"""CPU-only source preparation tests; no dataset, CUDA or actual checkpoint needed."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.solver._solver import BaseSolver
from tools.diagnostics.run_pairwise_ce_stage2 import (
    CONFIGS, prepare_run, sha256_file, validate_checkpoint,
)


def fixture():
    heads = {f"decoder.dec_score_head.{i}.weight": torch.zeros(3, 256) for i in range(3)}
    return {
        "last_epoch": 119, "model": heads,
        "ema": {"module": copy.deepcopy(heads), "updates": 14880},
        "optimizer": {"state": {0: {"step": torch.tensor(14880.)}},
                      "param_groups": [{"lr": 1e-4}, {"lr": 2e-4}]},
        "criterion": {}, "scaler": {"scale": 1024.},
        "lr_scheduler": {"last_epoch": 116},
        "lr_warmup_scheduler": {"last_step": 14880, "warmup_duration": 500},
    }


class Stage2PreparationTests(unittest.TestCase):
    def test_config_only_weight_diff_and_original_protocol(self):
        a, b = [load_config(str(ROOT / CONFIGS[p]), cfg={}) for p in ("w01", "w005")]
        for cfg in (a, b):
            self.assertEqual(cfg["epochs"], 132)
            self.assertEqual(cfg["optimizer"]["lr"], 2e-4)
            self.assertEqual(cfg["optimizer"]["params"][0]["lr"], 1e-4)
            self.assertEqual(cfg["lr_warmup_scheduler"]["warmup_duration"], 500)
            self.assertEqual(cfg["lr_scheduler"]["milestones"], [500])
            self.assertEqual(cfg["train_dataloader"]["collate_fn"]["stop_epoch"], 120)
            self.assertEqual(cfg["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"], 120)
            self.assertEqual(cfg["train_dataloader"]["total_batch_size"], 32)
        for cfg in (a, b):
            cfg.pop("__include__", None)
        a["DFINECriterion"]["pairwise_ce_weight"] = b["DFINECriterion"]["pairwise_ce_weight"]
        self.assertEqual(a, b)
        for profile, weight in (("w01", 0.1), ("w005", 0.05)):
            cfg = YAMLConfig(str(ROOT / CONFIGS[profile]))
            self.assertEqual(cfg.criterion.pairwise_ce_weight, weight)

    def test_reject_wrong_epochs_and_missing_states(self):
        for epoch in (118, 120, 126, 19):
            state = fixture()
            state["last_epoch"] = epoch
            with self.assertRaises(ValueError):
                validate_checkpoint(state)
        for key in ("optimizer", "ema", "scaler", "lr_warmup_scheduler"):
            state = fixture()
            del state[key]
            with self.assertRaises(ValueError):
                validate_checkpoint(state)
        state = fixture()
        state["optimizer"]["state"] = {}
        with self.assertRaises(ValueError):
            validate_checkpoint(state)

    def test_prepare_preserves_bytes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="dfine-stage2-test-") as tmp:
            tmp = Path(tmp)
            source = tmp / "source.pth"
            torch.save(fixture(), source)
            original_hash = sha256_file(source)
            for profile in CONFIGS:
                dest = tmp / profile
                metadata = prepare_run(source, dest, profile, use_amp=True)
                self.assertEqual(sha256_file(dest / "best_stg1.pth"), original_hash)
                self.assertEqual(sha256_file(source), original_hash)
                self.assertEqual(metadata["training_epochs"], 12)
                self.assertIn("-r", metadata["command"])
                self.assertNotIn("-t", metadata["command"])
                self.assertIn("--use-amp", metadata["command"])
                loaded = json.loads((dest / "stage2_source.json").read_text())
                self.assertEqual(loaded["source_sha256"], original_hash)
                with self.assertRaises(FileExistsError):
                    prepare_run(source, dest, profile)
            invalid = tmp / "invalid.pth"
            bad = fixture()
            bad["last_epoch"] = 19
            torch.save(bad, invalid)
            dest = tmp / "invalid-output"
            with self.assertRaises(ValueError):
                prepare_run(invalid, dest, "w01")
            self.assertFalse(dest.exists())

    def test_real_model_resume_preserves_states_and_new_loss_weight(self):
        # Exercise the EXISTING solver loader, rather than a new load path.
        def holder(profile):
            cfg = YAMLConfig(str(ROOT / CONFIGS[profile]))
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
            cfg.yaml_cfg["eval_spatial_size"] = [128, 128]
            solver = object.__new__(BaseSolver)
            solver.cfg = cfg
            solver.model = cfg.model
            solver.criterion = cfg.criterion
            solver.optimizer = cfg.optimizer
            solver.lr_scheduler = cfg.lr_scheduler
            solver.lr_warmup_scheduler = cfg.lr_warmup_scheduler
            solver.ema = cfg.ema
            solver.scaler = torch.amp.GradScaler("cpu", enabled=False)
            solver.last_epoch = -1
            return solver

        source = holder("w01")
        source.model.train()
        images = torch.rand(2, 3, 128, 128)
        targets = [
            {"labels": torch.tensor([1, 2]),
             "boxes": torch.tensor([[.3, .4, .2, .2], [.7, .6, .25, .3]])}
            for _ in range(2)
        ]
        outputs = source.model(images, targets)
        sum(source.criterion(outputs, targets).values()).backward()
        source.optimizer.step()
        source.ema.update(source.model)
        source.optimizer.zero_grad(set_to_none=True)
        source.last_epoch = 119
        source.ema.updates = 14880
        source.lr_warmup_scheduler.last_step = 14880
        source.lr_scheduler.last_epoch = 116
        for group, lr in zip(source.optimizer.param_groups,
                             source.lr_warmup_scheduler.warmup_end_values):
            group["lr"] = lr
        source_state = source.state_dict()
        self.assertTrue(source_state["optimizer"]["state"])
        with tempfile.TemporaryDirectory(prefix="dfine-stage2-resume-") as tmp:
            tmp = Path(tmp)
            checkpoint = tmp / "source.pth"
            torch.save(source_state, checkpoint)
            dest = tmp / "w005"
            prepare_run(checkpoint, dest, "w005", use_amp=True)
            resumed = holder("w005")
            resumed.load_resume_state(str(dest / "best_stg1.pth"))
            self.assertEqual(resumed.last_epoch, 119)
            self.assertEqual(list(range(resumed.last_epoch + 1, 132)), list(range(120, 132)))
            self.assertEqual(resumed.criterion.pairwise_ce_weight, 0.05)
            self.assertEqual(resumed.ema.updates, source.ema.updates)
            self.assertEqual(resumed.lr_scheduler.state_dict(), source.lr_scheduler.state_dict())
            self.assertEqual(resumed.lr_warmup_scheduler.state_dict(), source.lr_warmup_scheduler.state_dict())
            self.assertEqual(resumed.scaler.state_dict(), source.scaler.state_dict())
            for key, value in source.model.state_dict().items():
                torch.testing.assert_close(resumed.model.state_dict()[key], value, rtol=0, atol=0)
            for key, value in source.ema.module.state_dict().items():
                torch.testing.assert_close(resumed.ema.module.state_dict()[key], value, rtol=0, atol=0)
            original_opt = source.optimizer.state_dict()
            resumed_opt = resumed.optimizer.state_dict()
            self.assertEqual(resumed_opt["param_groups"], original_opt["param_groups"])
            for pid, values in original_opt["state"].items():
                for key, value in values.items():
                    torch.testing.assert_close(resumed_opt["state"][pid][key], value, rtol=0, atol=0)
            resumed.model.train()
            resumed.optimizer.zero_grad(set_to_none=True)
            losses = resumed.criterion(resumed.model(images, targets), targets)
            self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
            sum(losses.values()).backward()
            resumed.optimizer.step()
            resumed.ema.update(resumed.model)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
