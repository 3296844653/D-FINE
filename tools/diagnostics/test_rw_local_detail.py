"""CPU regression checks; no real dataset, checkpoint download or full training."""

import copy
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.rw_local_detail import RWLocalDetailAdapter

BASE = "dfine_s_scbs_hrw_110.yml"
TRAIN = "dfine_s_scbs_hrw_110_rw_local_detail.yml"
DIAG = "dfine_s_scbs_hrw_110_rw_local_detail_diagnostics.yml"


def fresh_config(name, size=128):
    with patch("src.core.yaml_config.load_config", lambda path: load_config(path, cfg={})):
        config = YAMLConfig(str(ROOT / "configs/dfine" / name))
    config.yaml_cfg["HGNetv2"]["pretrained"] = False
    config.yaml_cfg["eval_spatial_size"] = [size, size]
    return config


class RWLocalDetailTests(unittest.TestCase):
    def test_zero_initialization_gradient_and_query_independence(self):
        torch.manual_seed(12)
        branch = RWLocalDetailAdapter(hidden_dim=32, p2_channels=16, detail_dim=16)
        p2 = torch.randn(2, 16, 16, 24, requires_grad=True)
        p3 = torch.randn(2, 32, 8, 12, requires_grad=True)
        query = torch.randn(2, 7, 32, requires_grad=True)
        boxes = torch.tensor([.5, .5, .6, .6]).expand(2, 7, 4).clone().requires_grad_()
        delta = branch(p2, p3, boxes, query)
        torch.testing.assert_close(delta, torch.zeros_like(query), rtol=0, atol=0)
        (delta * torch.randn_like(query)).sum().backward()
        self.assertGreater(branch.fuse[-1].weight.grad.abs().sum().item(), 0)
        self.assertEqual(branch.key_projection.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(boxes.grad)
        branch.zero_grad(set_to_none=True)
        p2.grad = p3.grad = query.grad = None
        torch.nn.init.normal_(branch.fuse[-1].weight, std=.1)
        actual = branch(p2, p3, boxes, query)
        actual.square().sum().backward()
        for name, parameter in branch.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        for feature in (p2, p3, query):
            self.assertGreater(feature.grad.abs().sum().item(), 0)
        self.assertIsNone(boxes.grad)
        order = torch.tensor([6, 1, 3, 0, 2, 5, 4])
        torch.testing.assert_close(branch(p2, p3, boxes[:, order], query[:, order]), actual[:, order])
        changed = query.detach().clone(); changed[:, 0] += torch.randn(32) * 20
        torch.testing.assert_close(branch(p2, p3, boxes, changed)[:, 1:], actual[:, 1:], rtol=0, atol=0)
        self.assertEqual(branch(p2, p3, boxes[:, :0], query[:, :0]).shape, (2, 0, 32))

    def test_fp32_sampling_coordinates_and_cpu_amp(self):
        branch = RWLocalDetailAdapter(hidden_dim=32, p2_channels=16, detail_dim=16)
        projected = torch.rand(2, 16, 12, 20)
        box = torch.tensor([[[.5, .5, .5, .6], [.9, .2, .4, .1]]] * 2)
        grid = 2 * (box[..., None, :2] + box[..., None, 2:] * branch.relative_grid) - 1
        expected = F.grid_sample(projected, grid.flatten(1, 2).unsqueeze(2),
                                 align_corners=False).squeeze(-1).transpose(1, 2).reshape(2, 2, 25, 16)
        torch.testing.assert_close(branch._sample(projected, grid), expected, rtol=0, atol=0)
        torch.nn.init.normal_(branch.fuse[-1].weight, std=.1)
        p2, p3, query = torch.rand(2, 16, 12, 20), torch.rand(2, 32, 6, 10), torch.rand(2, 2, 32)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            delta = branch(p2, p3, box, query)
            delta.float().square().sum().backward()
        self.assertTrue(torch.isfinite(delta).all())
        for name, parameter in branch.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_invalid_settings_and_feature_routing_guards(self):
        for kwargs in ({"hidden_dim": 0}, {"p2_channels": 0}, {"detail_dim": 10},
                       {"num_heads": 3}, {"grid_size": 1}, {"grid_size": 4},
                       {"grid_size": 5.0}, {"residual_scale": 0},
                       {"residual_scale": float("nan")}):
            with self.assertRaises(ValueError):
                RWLocalDetailAdapter(**kwargs)
        for kwargs in ({"use_aqs_refine": True}, {"use_p2_roi_cls": True},
                       {"use_rw_spatial_relation": True}, {"use_task_decoupled_heads": True},
                       {"eval_idx": 0}, {"layer_scale": 2},
                       {"rw_detail_class_ids": [1, 1]}, {"rw_detail_class_ids": [1, 3]},
                       {"rw_detail_class_ids": [1.0, 2]}):
            with self.assertRaises(ValueError):
                DFINETransformer(num_classes=3, use_rw_local_detail=True, **kwargs)
        model = fresh_config(TRAIN).model.eval()
        model.backbone.return_idx = [1, 2, 3]
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "return_idx"):
            model(torch.rand(2, 3, 128, 128))
        model.backbone.return_idx = [0, 1, 2, 3]
        model.encoder.use_p2_detail_fusion = True
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "separate ablations"):
            model(torch.rand(2, 3, 128, 128))

    def test_config_single_variable_and_baseline_initialization(self):
        baseline, trial = fresh_config(BASE), fresh_config(TRAIN)
        expected, actual = copy.deepcopy(baseline.yaml_cfg), copy.deepcopy(trial.yaml_cfg)
        for config in (expected, actual):
            config.pop("__include__", None)
            config.pop("output_dir", None)
        expected["HGNetv2"]["return_idx"] = [0, 1, 2, 3]
        expected["DFINETransformer"].update(
            use_rw_local_detail=True, rw_detail_p2_channels=64, rw_detail_dim=64,
            rw_detail_grid_size=5, rw_detail_num_heads=4, rw_detail_residual_scale=.1,
            rw_detail_class_ids=[1, 2],
        )
        self.assertEqual(expected, actual)
        diag = fresh_config(DIAG)
        self.assertTrue(diag.yaml_cfg["export_confusion_matrix"])
        self.assertTrue(diag.yaml_cfg["export_query_diagnostics"])
        for key in ("HGNetv2", "HybridEncoder", "DFINETransformer", "DFINECriterion",
                    "optimizer", "train_dataloader", "val_dataloader"):
            self.assertEqual(diag.yaml_cfg[key], trial.yaml_cfg[key])
        torch.manual_seed(21); base = baseline.model.eval()
        baseline_rng = torch.get_rng_state().clone()
        torch.manual_seed(21); model = trial.model.eval()
        torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
        for key, value in base.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        base_state = {k: v for k, v in model.state_dict().items()
                      if not k.startswith("decoder.rw_local_detail.")}
        base.load_state_dict(base_state, strict=True)
        added = sum(p.numel() for p in model.decoder.rw_local_detail.parameters())
        self.assertEqual(sum(p.numel() for p in model.parameters()) -
                         sum(p.numel() for p in base.parameters()), added)
        print("Local-detail added parameters:", added,
              "total:", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
        images = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            original, enhanced = base(images), model(images)
        for key in ("pred_boxes", "pred_logits"):
            torch.testing.assert_close(original[key], enhanced[key], rtol=0, atol=0)
        self.assertEqual(enhanced["pred_logits"].shape, (2, 300, 3))
        self.assertFalse(hasattr(base.decoder, "rw_local_detail"))

    def test_training_dn_aux_teacher_bbox_loss_optimizer_and_deploy(self):
        cfg = fresh_config(TRAIN)
        model, criterion = cfg.model, cfg.criterion
        self.assertFalse(criterion.use_pairwise_ce)
        self.assertFalse(criterion.use_class_margin)
        branch = model.decoder.rw_local_detail
        torch.nn.init.normal_(branch.fuse[-1].weight, std=.05)
        images = torch.rand(2, 3, 128, 128)
        targets = [
            {"labels": torch.tensor([0, 1, 2]),
             "boxes": torch.tensor([[.2, .3, .15, .2], [.5, .5, .2, .3], [.8, .7, .2, .2]])},
            {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)},
        ]
        model.train()
        torch.manual_seed(8); outputs = model(images, targets)
        model.decoder.use_rw_local_detail = False
        torch.manual_seed(8); original = model(images, targets)
        model.decoder.use_rw_local_detail = True
        for key in ("pred_boxes", "pred_corners", "ref_points"):
            torch.testing.assert_close(outputs[key], original[key], rtol=0, atol=0)
        torch.testing.assert_close(outputs["pred_logits"][..., 0], original["pred_logits"][..., 0], rtol=0, atol=0)
        self.assertGreater((outputs["pred_logits"] - original["pred_logits"]).abs().sum().item(), 0)
        for group in ("dn_outputs", "aux_outputs", "enc_aux_outputs"):
            for enhanced, old in zip(outputs[group], original[group]):
                for key in old:
                    if torch.is_tensor(old[key]):
                        torch.testing.assert_close(enhanced[key], old[key], rtol=0, atol=0)
        for key, value in original["pre_outputs"].items():
            torch.testing.assert_close(outputs["pre_outputs"][key], value, rtol=0, atol=0)
        losses = criterion(outputs, targets)
        self.assertFalse(any("pairwise_ce" in key or "margin" in key for key in losses))
        self.assertTrue(all(torch.isfinite(value).all() for value in losses.values()))
        sum(losses.values()).backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(branch.fuse[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(branch.p2_projection.conv.weight.grad.abs().sum().item(), 0)
        self.assertGreater(branch.p3_projection.conv.weight.grad.abs().sum().item(), 0)
        optimizer = cfg.optimizer
        identities = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertTrue(all(id(p) in identities for p in branch.parameters()))
        norm_parameters = (branch.p2_projection.norm.weight, branch.p3_projection.norm.weight,
                           branch.fuse.norm.weight, branch.query_projection.norm.weight)
        for parameter in norm_parameters:
            group = next(group for group in optimizer.param_groups if any(p is parameter for p in group["params"]))
            self.assertEqual(group["weight_decay"], 0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            deployed = copy.deepcopy(model).deploy()
            actual = deployed(images)
            for key in ("pred_boxes", "pred_logits"):
                self.assertTrue(torch.isfinite(actual[key]).all())
            # Random HGNet features have tied TopK scores. The original
            # encoder's conv/BN folding can reorder those ties (also without
            # this adapter). Verify decoder pruning with identical features.
            features = [torch.randn(2, 256, side, side) for side in (16, 8, 4)]
            p2 = torch.randn(2, 64, 32, 32)
            expected_decoder = model.decoder(features, p2_feat=p2)
            deployed_decoder = deployed.decoder(features, p2_feat=p2)
            for key in expected_decoder:
                torch.testing.assert_close(deployed_decoder[key], expected_decoder[key], rtol=1e-5, atol=1e-5)
            deployed.decoder.use_rw_local_detail = False
            disabled = deployed(images)
        torch.testing.assert_close(actual["pred_boxes"], disabled["pred_boxes"], rtol=0, atol=0)
        torch.testing.assert_close(actual["pred_logits"][..., 0], disabled["pred_logits"][..., 0], rtol=0, atol=0)

    def test_full_640_resolution_eval(self):
        model = fresh_config(TRAIN, size=640).model.eval()
        torch.nn.init.normal_(model.decoder.rw_local_detail.fuse[-1].weight, std=.02)
        with torch.no_grad():
            output = model(torch.rand(1, 3, 640, 640))
        self.assertEqual(output["pred_logits"].shape, (1, 300, 3))
        self.assertEqual(output["pred_boxes"].shape, (1, 300, 4))
        for value in output.values():
            self.assertTrue(torch.isfinite(value).all())

    def test_launcher_syntax_and_existing_results_are_preserved(self):
        script = ROOT / "tools/diagnostics/run_scbs_rw_local_detail_110.sh"
        subprocess.run(["bash", "-n", str(script)], check=True)
        with tempfile.TemporaryDirectory(prefix="dfine-rw-detail-test-") as directory:
            result = subprocess.run(["bash", str(script), directory], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("结果目录已存在", result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_strict_checkpoint_round_trip_and_full_cpu_amp_training(self):
        model = fresh_config(TRAIN).model
        torch.nn.init.normal_(model.decoder.rw_local_detail.fuse[-1].weight, std=.01)
        restored_cfg = fresh_config(TRAIN)
        restored = restored_cfg.model
        restored.load_state_dict(model.state_dict(), strict=True)
        images = torch.rand(2, 3, 128, 128)
        model.eval(); restored.eval()
        with torch.no_grad():
            expected, actual = model(images), restored(images)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        restored.train()
        for empty in (False, True):
            restored.zero_grad(set_to_none=True)
            targets = [dict(labels=torch.empty(0, dtype=torch.long), boxes=torch.empty(0, 4)) for _ in range(2)]
            if not empty:
                targets[0] = dict(labels=torch.tensor([1, 2]), boxes=torch.tensor([[.4, .5, .3, .3], [.7, .7, .2, .2]]))
            with torch.autocast("cpu", dtype=torch.bfloat16):
                output = restored(images, targets)
            losses = restored_cfg.criterion(output, targets)
            self.assertTrue(all(torch.isfinite(value).all() for value in losses.values()))
            sum(losses.values()).backward()
            for name, parameter in restored.decoder.rw_local_detail.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable; GPU FP16 check not run")
    def test_cuda_fp16_detector_training(self):
        cfg = fresh_config(TRAIN, size=640)
        model = cfg.model.cuda().train()
        criterion = cfg.criterion.cuda()
        torch.nn.init.normal_(model.decoder.rw_local_detail.fuse[-1].weight, std=.01)
        images = torch.rand(2, 3, 640, 640, device="cuda")
        targets = [dict(labels=torch.tensor([1, 2], device="cuda"),
                        boxes=torch.tensor([[.4, .5, .3, .3], [.7, .7, .2, .2]], device="cuda")),
                   dict(labels=torch.empty(0, dtype=torch.long, device="cuda"),
                        boxes=torch.empty(0, 4, device="cuda"))]
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(images, targets)
        losses = criterion(output, targets)
        self.assertTrue(all(torch.isfinite(value).all() for value in losses.values()))
        sum(losses.values()).backward()
        for name, parameter in model.decoder.rw_local_detail.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
