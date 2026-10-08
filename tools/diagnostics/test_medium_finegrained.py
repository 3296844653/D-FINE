"""CPU regression checks for the independent SCB-S MFFE experiment.

No dataset access, pretrained downloads, or full detector training. Run from
the repository root: python tools/diagnostics/test_medium_finegrained.py
"""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.zoo.dfine.hybrid_encoder import HybridEncoder
from src.zoo.dfine.medium_finegrained import MediumScaleFineGrainedEnhancement


def fresh_config(name):
    # The existing YAML loader has a shared default dict. Isolate test loads
    # without changing the project's loader or actual training workflow.
    with patch("src.core.yaml_config.load_config", lambda path: load_config(path, cfg={})):
        cfg = YAMLConfig(str(ROOT / "configs/dfine" / name))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["eval_spatial_size"] = [128, 128]
    return cfg


def small_encoder(**kwargs):
    return HybridEncoder(
        in_channels=[16, 32, 64], hidden_dim=32, dim_feedforward=64,
        depth_mult=0.34, expansion=0.5, mffe_mid_channels=8, **kwargs
    )


class MediumFineGrainedTests(unittest.TestCase):
    def test_shapes_gates_and_first_step_gradients(self):
        torch.manual_seed(10)
        module = MediumScaleFineGrainedEnhancement(32, 8)
        source = torch.randn(2, 32, 9, 13, requires_grad=True)
        fused = torch.randn_like(source, requires_grad=True)
        gates = []
        hook = module.branch_gate.register_forward_hook(
            lambda _, inputs, output: gates.append(output.detach().float().softmax(1))
        )
        output = module(source, fused)
        hook.remove()
        self.assertEqual(output.shape, fused.shape)
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(gates[0].sum(1), torch.ones(2, 9, 13))
        self.assertAlmostEqual(module.residual_scale().item(), 0.1, places=6)
        (output * torch.randn_like(output)).sum().backward()
        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        self.assertGreater(source.grad.abs().sum().item(), 0)
        self.assertGreater(fused.grad.abs().sum().item(), 0)
        # GroupNorm remains valid for a tiny feature map and batch size one.
        self.assertEqual(module(torch.rand(1, 32, 1, 1), torch.rand(1, 32, 1, 1)).shape,
                         (1, 32, 1, 1))

    def test_zero_scale_parity_and_bounded_scale(self):
        module = MediumScaleFineGrainedEnhancement(32, 8, alpha_init=0.0)
        source, fused = torch.randn(2, 32, 7, 11), torch.randn(2, 32, 7, 11)
        torch.testing.assert_close(module(source, fused), fused, rtol=0, atol=0)
        for raw_scale in (-100., 100.):
            with torch.no_grad():
                module.raw_alpha.fill_(raw_scale)
            self.assertLessEqual(abs(module.residual_scale().item()), 0.5)

    def test_cpu_autocast_forward_and_backward(self):
        module = MediumScaleFineGrainedEnhancement(32, 8)
        source = torch.randn(2, 32, 8, 12, requires_grad=True)
        fused = torch.randn_like(source, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = module(source, fused)
            loss = output.float().square().mean()
        self.assertTrue(torch.isfinite(output).all())
        loss.backward()
        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_checkpoint_outputs_gradients_and_saved_activation_memory(self):
        torch.manual_seed(12)
        ordinary = MediumScaleFineGrainedEnhancement(32, 8, use_checkpoint=False).train()
        recomputed = copy.deepcopy(ordinary)
        recomputed.use_checkpoint = True
        tensors = [torch.randn(2, 32, 16, 24), torch.randn(2, 32, 16, 24)]
        grad_weight = torch.randn_like(tensors[0])
        saved_bytes = []
        outputs, input_grads = [], []
        for module in (ordinary, recomputed):
            inputs = [x.detach().clone().requires_grad_() for x in tensors]
            records = []

            def pack(tensor):
                records.append(tensor.numel() * tensor.element_size())
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    output = module(*inputs)
            saved_bytes.append(sum(records))
            outputs.append(output.detach())
            (output * grad_weight).sum().backward()
            input_grads.append([x.grad for x in inputs])
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for a, b in zip(input_grads[0], input_grads[1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for (name, a), (_, b) in zip(ordinary.named_parameters(), recomputed.named_parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)
        # This measures tensors retained by the MFFE forward for backward,
        # NOT the full detector's CUDA peak or a guaranteed GPU capacity.
        self.assertLess(saved_bytes[1], saved_bytes[0])
        print(f"MFFE saved tensor bytes: ordinary={saved_bytes[0]}, "
              f"checkpoint={saved_bytes[1]}")
        # Non-reentrant checkpoint must still train the branch when the
        # upstream feature extractor is frozen (inputs need no gradients).
        recomputed.zero_grad(set_to_none=True)
        recomputed(*tensors).square().mean().backward()
        self.assertGreater(recomputed.detail_proj[0].weight.grad.abs().sum().item(), 0)
        recomputed.eval()
        with patch("src.zoo.dfine.medium_finegrained.checkpoint") as mocked:
            recomputed(*tensors)
            mocked.assert_not_called()
        recomputed.train()
        with patch("src.zoo.dfine.medium_finegrained.checkpoint") as mocked:
            with torch.no_grad():
                recomputed(*tensors)
            mocked.assert_not_called()

    def test_invalid_configuration_and_shapes(self):
        for kwargs in ({"mid_channels": 0}, {"mid_channels": 8.0},
                       {"alpha_init": -0.1}, {"alpha_init": 0.5},
                       {"alpha_init": float("nan")}, {"alpha_init": float("inf")}):
            with self.assertRaises(ValueError):
                MediumScaleFineGrainedEnhancement(32, **kwargs)
        module = MediumScaleFineGrainedEnhancement(32, 8)
        for source, fused in ((torch.rand(1, 32, 8, 8), torch.rand(1, 32, 4, 4)),
                              (torch.rand(1, 16, 8, 8), torch.rand(1, 16, 8, 8)),
                              (torch.rand(1, 32, 8), torch.rand(1, 32, 8))):
            with self.assertRaises(ValueError):
                module(source, fused)
        for flag in ("use_rfa_p3", "use_sfif", "use_p2_detail_fusion",
                     "use_encoder_highres_residual", "use_encoder_p3_joint_residual",
                     "use_p3_rfaconv_residual"):
            with self.assertRaisesRegex(ValueError, "ablated separately"):
                small_encoder(use_mffe=True, **{flag: True})
        with self.assertRaisesRegex(ValueError, "strides"):
            small_encoder(use_mffe=True, feat_strides=[4, 8, 16])

    def test_encoder_initialization_rng_disabled_parity_and_position(self):
        torch.manual_seed(21)
        baseline = small_encoder().eval()
        baseline_rng = torch.get_rng_state().clone()
        torch.manual_seed(21)
        trial = small_encoder(use_mffe=True).eval()
        torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
        self.assertFalse(hasattr(baseline, "mffe"))
        baseline_keys = set(baseline.state_dict())
        trial_keys = set(trial.state_dict())
        self.assertTrue(trial_keys - baseline_keys)
        self.assertTrue(all(k.startswith("mffe.") for k in trial_keys - baseline_keys))
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(trial.state_dict()[key], value, rtol=0, atol=0)
        features = [torch.randn(2, 16, 16, 16), torch.randn(2, 32, 8, 8),
                    torch.randn(2, 64, 4, 4)]
        trace, hooks = [], []
        for name, module in (("mffe_P3", trial.mffe[0]), ("mffe_P4", trial.mffe[1]),
                             ("pan_P4", trial.pan_blocks[0]), ("pan_P5", trial.pan_blocks[1])):
            hooks.append(module.register_forward_hook(
                lambda _, inputs, output, label=name: trace.append(label)
            ))
        with torch.no_grad():
            enhanced = trial(features)
            original = baseline(features)
        for hook in hooks:
            hook.remove()
        self.assertEqual(trace, ["mffe_P3", "mffe_P4", "pan_P4", "pan_P5"])
        for a, b in zip(enhanced, original):
            self.assertEqual(a.shape, b.shape)
            self.assertGreater((a - b).abs().sum().item(), 0)
        trial.use_mffe = False
        with torch.no_grad():
            disabled = trial(features)
        for a, b in zip(disabled, original):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_yaml_changes_only_module_and_diagnostics_use_correct_architecture(self):
        baseline = fresh_config("dfine_s_scbs_hrw.yml")
        trial = fresh_config("dfine_s_scbs_hrw_mffe.yml")
        expected, actual = copy.deepcopy(baseline.yaml_cfg), copy.deepcopy(trial.yaml_cfg)
        expected.pop("__include__", None)
        actual.pop("__include__", None)
        expected["HybridEncoder"].update(
            use_mffe=True, mffe_mid_channels=64, mffe_alpha_init=0.1, mffe_checkpoint=True
        )
        self.assertEqual(actual, expected)
        diag = fresh_config("dfine_s_scbs_hrw_mffe_diagnostics.yml")
        self.assertTrue(diag.yaml_cfg["HybridEncoder"]["use_mffe"])
        self.assertTrue(diag.yaml_cfg["export_confusion_matrix"])
        self.assertTrue(diag.yaml_cfg["export_query_diagnostics"])
        self.assertEqual(diag.yaml_cfg["query_diagnostic_conf_thresh"], 0.5)

    def test_full_detector_loss_gradients_optimizer_checkpoint_and_deploy(self):
        baseline_cfg = fresh_config("dfine_s_scbs_hrw.yml")
        trial_cfg = fresh_config("dfine_s_scbs_hrw_mffe.yml")
        torch.manual_seed(30)
        baseline = baseline_cfg.model.eval()
        torch.manual_seed(30)
        model = trial_cfg.model.eval()
        self.assertTrue(all(m.use_checkpoint for m in model.encoder.mffe))
        original_state = baseline.state_dict()
        for key, value in original_state.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        result = model.load_state_dict(original_state, strict=False)
        self.assertFalse(result.unexpected_keys)
        self.assertTrue(result.missing_keys)
        self.assertTrue(all(k.startswith("encoder.mffe.") for k in result.missing_keys))
        # Loading common weights into the unchanged architecture stays strict.
        baseline.load_state_dict({k: v for k, v in model.state_dict().items()
                                  if not k.startswith("encoder.mffe.")}, strict=True)
        baseline_total = sum(p.numel() for p in baseline.parameters())
        baseline_trainable = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
        extra = sum(p.numel() for p in model.encoder.mffe.parameters())
        self.assertEqual(sum(p.numel() for p in model.parameters()) -
                         sum(p.numel() for p in baseline.parameters()), extra)
        print(f"MFFE added parameters: {extra}; trainable total: {baseline_trainable + extra}; "
              f"all parameters: {baseline_total + extra}")
        print("Residual scales:", [m.residual_scale().item() for m in model.encoder.mffe])
        # Setting only the optional branch scale to zero recovers predictions.
        scales = [m.raw_alpha.detach().clone() for m in model.encoder.mffe]
        with torch.no_grad():
            for module in model.encoder.mffe:
                module.raw_alpha.zero_()
            images = torch.rand(2, 3, 128, 128)
            original, zero_scaled = baseline(images), model(images)
            for key in ("pred_logits", "pred_boxes"):
                torch.testing.assert_close(zero_scaled[key], original[key], rtol=0, atol=0)
            for module, scale in zip(model.encoder.mffe, scales):
                module.raw_alpha.copy_(scale)
        criterion = trial_cfg.criterion
        self.assertFalse(criterion.use_pairwise_ce)
        self.assertFalse(criterion.use_class_margin)
        self.assertFalse(criterion.use_class_balanced_vfl)
        for flag, value in vars(model.decoder).items():
            if flag.startswith("use_"):
                self.assertFalse(value, flag)
        targets = [
            {"labels": torch.tensor([0, 1, 2]), "boxes": torch.tensor([
                [.2, .3, .15, .2], [.5, .5, .2, .3], [.8, .7, .2, .2]])},
            {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)},
        ]
        model.train()
        output = model(images, targets)
        self.assertIn("dn_outputs", output)
        self.assertIn("aux_outputs", output)
        self.assertEqual(output["pred_logits"].shape, (2, 300, 3))
        self.assertEqual(output["pred_boxes"].shape, (2, 300, 4))
        losses = criterion(output, targets)
        self.assertTrue(losses)
        self.assertFalse(any("pairwise" in k or "margin" in k for k in losses))
        self.assertTrue(all(torch.isfinite(loss).all() for loss in losses.values()))
        sum(losses.values()).backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            if name.startswith("encoder.mffe."):
                self.assertIsNotNone(parameter.grad, name)
                self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        optimizer = trial_cfg.optimizer
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(id(p) in ids for p in model.encoder.mffe.parameters()))
        before_step = model.encoder.mffe[0].detail_proj[0].weight.detach().clone()
        optimizer.step()
        self.assertGreater((before_step - model.encoder.mffe[0].detail_proj[0].weight)
                           .abs().sum().item(), 0)
        model.eval()
        # A same-architecture checkpoint round trip includes every new tensor.
        restored = fresh_config("dfine_s_scbs_hrw_mffe.yml").model.eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            expected, actual = model(images), restored(images)
            for key in ("pred_logits", "pred_boxes"):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            deployed_model = copy.deepcopy(model).deploy()
            deployed = deployed_model(images)
            backbone_features = model.backbone(images)
            for a, b in zip(model.encoder(backbone_features),
                            deployed_model.encoder(backbone_features)):
                torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
            for key, value in model.encoder.mffe.state_dict().items():
                torch.testing.assert_close(deployed_model.encoder.mffe.state_dict()[key],
                                           value, rtol=0, atol=0)
            # Near-tied encoder scores can reorder selected queries after the
            # baseline's BN fusion, so full detector output ordering is NOT a
            # valid elementwise deployment-parity assertion here.
            for key in ("pred_logits", "pred_boxes"):
                self.assertTrue(torch.isfinite(deployed[key]).all())
                self.assertEqual(deployed[key].shape, expected[key].shape)
            baseline_deployed = copy.deepcopy(baseline).deploy()
            base_deploy_params = sum(p.numel() for p in baseline_deployed.parameters())
            self.assertEqual(base_deploy_params, 10178201)
            self.assertEqual(sum(p.numel() for p in deployed_model.parameters()),
                             base_deploy_params + extra)
            print(f"Log/profiler deployment parameters: {base_deploy_params + extra}")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
