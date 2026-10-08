"""Synthetic CPU checks for AQS scalar export; no dataset or GPU required."""

import copy
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.solver.aqs_parameters import checkpoint_aqs_report, runtime_aqs_report, save_aqs_report
from src.solver.det_solver import DetSolver
from src.zoo.dfine.dfine_decoder import AdaptiveQuerySelectionRefiner
from tools.diagnostics.export_aqs_parameters import export_checkpoint


def tiny_model(enabled=True):
    model = nn.Module()
    model.query_cls_refiner = AdaptiveQuerySelectionRefiner(8) if enabled else None
    return model


class AQSParameterTests(unittest.TestCase):
    def test_runtime_raw_effective_and_model_ema_are_independent(self):
        model = tiny_model()
        ema = SimpleNamespace(module=copy.deepcopy(model))
        with torch.no_grad():
            model.query_cls_refiner.threshold_logit.fill_(math.log(3))
            model.query_cls_refiner.residual_scale.fill_(-.2)
            ema.module.query_cls_refiner.threshold_logit.fill_(0)
            ema.module.query_cls_refiner.residual_scale.fill_(.4)
        report = runtime_aqs_report(model, ema, epoch=109)
        values = report["sources"]["model"]["query_cls_refiner"]
        self.assertAlmostEqual(values["threshold"], .75, places=6)
        self.assertAlmostEqual(values["effective_residual_scale"], math.tanh(-.2), places=7)
        self.assertEqual(values["temperature"], .1)
        self.assertEqual(report["sources"]["ema"]["query_cls_refiner"]["threshold"], .5)
        self.assertEqual(report["epoch"], 109)
        self.assertEqual(report["evaluation_weight_source"], "ema")

    def test_snapshot_detached_no_rng_grad_mode_or_weight_changes(self):
        model = tiny_model().train()
        (model.query_cls_refiner.residual_scale ** 2).backward()
        state = copy.deepcopy(model.state_dict())
        gradient = model.query_cls_refiner.residual_scale.grad.clone()
        rng = torch.get_rng_state().clone()
        report = runtime_aqs_report(model, epoch=1)
        self.assertTrue(model.training)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        torch.testing.assert_close(model.query_cls_refiner.residual_scale.grad, gradient, rtol=0, atol=0)
        for key, value in state.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        with torch.no_grad():
            model.query_cls_refiner.residual_scale.fill_(9)
        self.assertAlmostEqual(report["sources"]["model"]["query_cls_refiner"]["residual_scale"], .05)

    def test_disabled_baseline_is_noop_and_ddp_prefix_is_normalized(self):
        self.assertIsNone(runtime_aqs_report(tiny_model(False), epoch=0))
        wrapper = nn.Module()
        wrapper.module = tiny_model()
        report = runtime_aqs_report(wrapper, epoch=0)
        self.assertEqual(set(report["sources"]["model"]), {"query_cls_refiner"})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "aqs.json"
            save_aqs_report(None, path)
            self.assertFalse(path.exists())

    def test_checkpoint_prefixes_missing_ema_and_unknown_temperature(self):
        state = {"module.decoder.decoder." + k: v for k, v in tiny_model().state_dict().items()}
        report = checkpoint_aqs_report({"model": state, "last_epoch": 127}, "best_stg2.pth")
        values = report["sources"]["model"]["decoder.decoder.query_cls_refiner"]
        self.assertIsNone(values["temperature"])
        self.assertIsNone(values["temperature_source"])
        self.assertNotIn("ema", report["sources"])
        self.assertEqual(report["checkpoint_last_epoch"], 127)
        self.assertNotIn("epoch", report)
        report = checkpoint_aqs_report({"ema": {"module": state}}, "best_stg2.pth", temperature=.2)
        self.assertEqual(report["sources"]["ema"]["decoder.decoder.query_cls_refiner"]["temperature"], .2)

    def test_checkpoint_rejects_eval_pth_incomplete_or_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "No AQS"):
            checkpoint_aqs_report({"precision": torch.ones(2)}, "eval.pth")
        state = tiny_model().state_dict()
        del state["query_cls_refiner.residual_scale"]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            checkpoint_aqs_report({"model": state}, "x.pth")
        for value in (torch.tensor(float("nan")), torch.ones(2)):
            state = tiny_model().state_dict()
            state["query_cls_refiner.threshold_logit"] = value
            with self.assertRaises(ValueError):
                checkpoint_aqs_report({"model": state}, "x.pth")
        with self.assertRaises(ValueError):
            checkpoint_aqs_report({"model": tiny_model().state_dict()}, "x.pth", temperature=0)

    def test_old_checkpoint_exports_json_without_rewriting_pth(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "best_stg2.pth"
            model = tiny_model()
            ema = copy.deepcopy(model)
            with torch.no_grad():
                ema.query_cls_refiner.threshold_logit.fill_(1)
            torch.save({"model": model.state_dict(), "ema": {"module": ema.state_dict()},
                        "last_epoch": 10}, path)
            before = path.read_bytes()
            with redirect_stdout(io.StringIO()) as output:
                report = export_checkpoint(path)
            self.assertIn("unknown (not stored in checkpoint)", output.getvalue())
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(json.loads(path.with_name("best_stg2_aqs_parameters.json").read_text()), report)
            with self.assertRaises(FileExistsError):
                export_checkpoint(path)
            with self.assertRaises(ValueError):
                export_checkpoint(path, output_path=path, overwrite=True)
            with redirect_stdout(io.StringIO()):
                export_checkpoint(path, temperature=.1, overwrite=True)
            self.assertEqual(path.read_bytes(), before)

    def test_fit_snapshot_matches_evaluation_and_best_not_stage2_reload(self):
        # Model/EMA values are deliberately reset by the solver after the last
        # (worse) stage-2 evaluation. Export MUST retain pre-reset values.
        with tempfile.TemporaryDirectory() as folder:
            solver = DetSolver(SimpleNamespace(epochs=3, checkpoint_freq=12, clip_max_norm=.1, print_freq=1))
            solver.model = tiny_model()
            solver.ema = SimpleNamespace(module=copy.deepcopy(solver.model), decay=.99)
            solver.train = lambda: None
            solver.use_wandb = False
            solver.last_epoch = -1
            solver.output_dir = Path(folder)
            solver.train_dataloader = SimpleNamespace(set_epoch=Mock(), collate_fn=SimpleNamespace(
                stop_epoch=1, ema_restart_decay=.99))
            solver.val_dataloader = solver.criterion = solver.postprocessor = solver.evaluator = None
            solver.optimizer = solver.scaler = solver.writer = None
            solver.device = torch.device("cpu")
            solver.lr_warmup_scheduler = None
            solver.lr_scheduler = SimpleNamespace(step=Mock())
            solver.state_dict = lambda: {"model": solver.model.state_dict(),
                                         "ema": {"module": solver.ema.module.state_dict()},
                                         "last_epoch": solver.last_epoch}
            def reload_checkpoint(path):
                state = torch.load(path, map_location="cpu", weights_only=True)
                solver.model.load_state_dict(state["model"])
                solver.ema.module.load_state_dict(state["ema"]["module"])
                solver.last_epoch = state["last_epoch"]
            solver.load_resume_state = reload_checkpoint
            def train(*args, **kwargs):
                epoch = args[5]
                with torch.no_grad():
                    solver.model.query_cls_refiner.residual_scale.fill_(.1 * (epoch + 1))
                    solver.ema.module.query_cls_refiner.residual_scale.fill_(.2 * (epoch + 1))
                return {"loss": 1.}
            scores = iter((.1, .2, .15))
            def evaluate(*args, **kwargs):
                return {"coco_eval_bbox": [next(scores)]}, None
            with patch("src.solver.det_solver.stats", return_value=(123, "fake stats")), \
                 patch("src.solver.det_solver.train_one_epoch", side_effect=train), \
                 patch("src.solver.det_solver.evaluate", side_effect=evaluate), \
                 redirect_stdout(io.StringIO()) as output:
                solver.fit()
            final = json.loads((solver.output_dir / "aqs_parameters_final.json").read_text())
            best = json.loads((solver.output_dir / "aqs_parameters_best_stg2.json").read_text())
            self.assertEqual(final["epoch"], 2)
            self.assertEqual(best["epoch"], 1)
            self.assertAlmostEqual(final["sources"]["ema"]["query_cls_refiner"]["residual_scale"], .6)
            self.assertAlmostEqual(solver.ema.module.query_cls_refiner.residual_scale.item(), .2)
            saved = checkpoint_aqs_report(torch.load(solver.output_dir / "best_stg2.pth", weights_only=True),
                                          solver.output_dir / "best_stg2.pth", temperature=.1)
            self.assertEqual(best["sources"]["ema"]["query_cls_refiner"]["residual_scale"],
                             saved["sources"]["ema"]["query_cls_refiner"]["residual_scale"])
            logs = [json.loads(line) for line in (solver.output_dir / "log.txt").read_text().splitlines()]
            self.assertEqual(logs[-1]["aqs_parameters"], final["sources"])
            self.assertIn("final_evaluated_epoch", output.getvalue())
            self.assertIn("best_stg2", output.getvalue())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
