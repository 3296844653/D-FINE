"""CPU checks for SCB-S highres residual; no real data, downloads or full training."""

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


TRAIN_CONFIG = "dfine_s_scbs_hrw_encoder_highres_residual_a03.yml"
DIAG_CONFIG = "dfine_s_scbs_hrw_encoder_highres_residual_a03_diagnostics.yml"


def fresh_config(name):
    # Isolate the original YAML loader's mutable default inside test processes.
    with patch("src.core.yaml_config.load_config", lambda path: load_config(path, cfg={})):
        cfg = YAMLConfig(str(ROOT / "configs/dfine" / name))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["eval_spatial_size"] = [128, 128]
    return cfg


def small_encoder(**kwargs):
    return HybridEncoder(
        in_channels=[16, 32, 64], hidden_dim=32, dim_feedforward=64,
        depth_mult=.34, expansion=.5, **kwargs,
    )


def copy_output(value):
    if isinstance(value, dict):
        return {key: copy_output(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_output(item) for item in value]
    return value


class EncoderHighresResidualTests(unittest.TestCase):
    def test_exact_post_pan_formula_and_unchanged_p4_p5(self):
        torch.manual_seed(11); baseline = small_encoder().eval()
        baseline_rng = torch.get_rng_state().clone()
        torch.manual_seed(11); trial = small_encoder(
            use_encoder_highres_residual=True, encoder_highres_alpha_init=.3,
        ).eval()
        torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(trial.state_dict()[key], value, rtol=0, atol=0)
        self.assertEqual(set(trial.state_dict()) - set(baseline.state_dict()),
                         {"encoder_highres_alpha"})
        self.assertFalse(hasattr(baseline, "encoder_highres_alpha"))
        self.assertAlmostEqual(trial.encoder_highres_alpha.item(), .3, places=6)
        for side_h, side_w in ((16, 24), (24, 32)):
            features = [torch.rand(2, 16, side_h, side_w),
                        torch.rand(2, 32, side_h // 2, side_w // 2),
                        torch.rand(2, 64, side_h // 4, side_w // 4)]
            captured, trace = {}, []
            hooks = [trial.input_proj[0].register_forward_hook(
                lambda module, inputs, output: captured.update(source=output.detach().clone()))]
            for index, layer in enumerate(trial.pan_blocks):
                hooks.append(layer.register_forward_hook(
                    lambda module, inputs, output, i=index: trace.append(f"pan{i}")))
            hooks.append(trial.register_forward_hook(
                lambda module, inputs, output: trace.append("encoder_output")))
            with torch.no_grad():
                original, enhanced = baseline(features), trial(features)
            for hook in hooks:
                hook.remove()
            self.assertEqual(trace, ["pan0", "pan1", "encoder_output"])
            expected = original[0] + trial.encoder_highres_alpha * captured["source"]
            torch.testing.assert_close(enhanced[0], expected, rtol=0, atol=0)
            for index in (1, 2):
                torch.testing.assert_close(enhanced[index], original[index], rtol=0, atol=0)
            self.assertGreater((enhanced[0] - original[0]).abs().sum().item(), 0)

    def test_zero_and_disabled_parity_and_signed_direct_alpha(self):
        torch.manual_seed(12); baseline = small_encoder().eval()
        torch.manual_seed(12); trial = small_encoder(
            use_encoder_highres_residual=True, encoder_highres_alpha_init=.3,
        ).eval()
        features = [torch.rand(2, 16, 16, 24), torch.rand(2, 32, 8, 12),
                    torch.rand(2, 64, 4, 6)]
        with torch.no_grad():
            original = baseline(features)
            trial.encoder_highres_alpha.zero_()
            zero_scaled = trial(features)
            trial.encoder_highres_alpha.fill_(.3)
            trial.use_encoder_highres_residual = False
            disabled = trial(features)
            for aa, bb, cc in zip(original, zero_scaled, disabled):
                torch.testing.assert_close(aa, bb, rtol=0, atol=0)
                torch.testing.assert_close(aa, cc, rtol=0, atol=0)
            trial.use_encoder_highres_residual = True
            source = trial.input_proj[0](features[0])
            # This old probe has an unconstrained scalar, NOT MFFE's tanh gate.
            for alpha in (-.7, .8):
                trial.encoder_highres_alpha.fill_(alpha)
                output = trial(features)
                expected = original[0] + trial.encoder_highres_alpha * source
                torch.testing.assert_close(output[0], expected, rtol=0, atol=0)

    def test_cpu_amp_gradients_and_shape_guards(self):
        trial = small_encoder(use_encoder_highres_residual=True,
                              encoder_highres_alpha_init=.3).train()
        features = [torch.rand(2, 16, 16, 24, requires_grad=True),
                    torch.rand(2, 32, 8, 12, requires_grad=True),
                    torch.rand(2, 64, 4, 6, requires_grad=True)]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = trial(features)
            loss = sum(x.float().square().mean() for x in output)
        self.assertTrue(all(torch.isfinite(x).all() for x in output))
        loss.backward()
        self.assertIsNotNone(trial.encoder_highres_alpha.grad)
        self.assertTrue(torch.isfinite(trial.encoder_highres_alpha.grad).all())
        self.assertGreater(trial.encoder_highres_alpha.grad.abs().item(), 0)
        for feature in features:
            self.assertIsNotNone(feature.grad)
            self.assertTrue(torch.isfinite(feature.grad).all())
        for flag in ("use_mffe", "use_sfif", "use_p2_detail_fusion", "use_rfa_p3",
                     "use_p3_rfaconv_residual", "use_encoder_p3_joint_residual"):
            with self.assertRaisesRegex(ValueError, "ablated separately"):
                small_encoder(use_encoder_highres_residual=True, **{flag: True})
        with self.assertRaisesRegex(ValueError, "P3-P5"):
            small_encoder(use_encoder_highres_residual=True, feat_strides=[4, 8, 16])

    def test_configs_are_independent_and_only_change_two_options(self):
        base_cfg = fresh_config("dfine_s_scbs_hrw.yml")
        trial_cfg = fresh_config(TRAIN_CONFIG)
        expected, actual = copy.deepcopy(base_cfg.yaml_cfg), copy.deepcopy(trial_cfg.yaml_cfg)
        for cfg in (expected, actual):
            cfg.pop("__include__", None)
        expected["HybridEncoder"].update(
            use_encoder_highres_residual=True, encoder_highres_alpha_init=.3,
        )
        self.assertEqual(expected, actual)
        self.assertEqual(trial_cfg.yaml_cfg["num_classes"], 3)
        self.assertEqual(trial_cfg.yaml_cfg["epochs"], 110)
        self.assertEqual(trial_cfg.yaml_cfg["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"], 100)
        self.assertEqual(trial_cfg.yaml_cfg["train_dataloader"]["collate_fn"]["stop_epoch"], 100)
        self.assertEqual(trial_cfg.yaml_cfg["train_dataloader"]["total_batch_size"], 32)
        model = trial_cfg.model
        for name, value in vars(model.decoder).items():
            if name.startswith("use_"):
                self.assertFalse(value, name)
        for name, value in vars(model.encoder).items():
            if name.startswith("use_") and isinstance(value, bool):
                self.assertEqual(value, name == "use_encoder_highres_residual", name)
        c = trial_cfg.criterion
        for flag in ("use_pairwise_ce", "use_class_margin", "use_class_balanced_vfl",
                     "use_query_validity", "use_query_objectness", "use_shape_iou",
                     "use_rw_isolated_specialist"):
            self.assertFalse(getattr(c, flag), flag)
        diag = fresh_config(DIAG_CONFIG)
        self.assertTrue(diag.yaml_cfg["HybridEncoder"]["use_encoder_highres_residual"])
        self.assertEqual(diag.yaml_cfg["HybridEncoder"]["encoder_highres_alpha_init"], .3)
        self.assertTrue(diag.yaml_cfg["export_confusion_matrix"])
        self.assertTrue(diag.yaml_cfg["export_query_diagnostics"])
        self.assertEqual(diag.yaml_cfg["query_diagnostic_conf_thresh"], .5)

    def test_full_detector_loss_optimizer_ema_checkpoint_and_deploy(self):
        base_cfg, trial_cfg = fresh_config("dfine_s_scbs_hrw.yml"), fresh_config(TRAIN_CONFIG)
        torch.manual_seed(21); baseline = base_cfg.model.eval()
        baseline_rng = torch.get_rng_state().clone()
        torch.manual_seed(21); model = trial_cfg.model.eval()
        torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
        extra_key = "encoder.encoder_highres_alpha"
        self.assertEqual(set(model.state_dict()) - set(baseline.state_dict()), {extra_key})
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in model.parameters()) -
                         sum(p.numel() for p in baseline.parameters()), 1)
        images = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            model.encoder.encoder_highres_alpha.zero_()
            original, zero_scaled = baseline(images), model(images)
            for key in ("pred_logits", "pred_boxes"):
                torch.testing.assert_close(zero_scaled[key], original[key], rtol=0, atol=0)
            model.encoder.encoder_highres_alpha.fill_(.3)
        targets = [
            {"labels": torch.tensor([0, 1, 2]), "boxes": torch.tensor([
                [.2, .3, .15, .2], [.5, .5, .2, .3], [.8, .7, .2, .2]])},
            {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)},
        ]
        model.train(); output = model(images, targets)
        self.assertEqual(output["pred_logits"].shape, (2, 300, 3))
        self.assertEqual(output["pred_boxes"].shape, (2, 300, 4))
        self.assertIn("dn_outputs", output)
        self.assertIn("aux_outputs", output)
        losses = trial_cfg.criterion(copy_output(output), targets)
        baseline.train(); old_output = baseline(images, targets)
        old_losses = base_cfg.criterion(copy_output(old_output), targets)
        self.assertEqual(set(losses), set(old_losses))
        self.assertTrue(all(torch.isfinite(v).all() for v in losses.values()))
        sum(losses.values()).backward()
        alpha = model.encoder.encoder_highres_alpha
        self.assertIsNotNone(alpha.grad)
        self.assertTrue(torch.isfinite(alpha.grad).all())
        self.assertGreater(alpha.grad.abs().item(), 0)
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer, ema = trial_cfg.optimizer, trial_cfg.ema
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn(id(alpha), ids)
        before = alpha.detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
        optimizer.step()
        self.assertGreater((alpha - before).abs().item(), 0)
        ema.update(model)
        self.assertTrue(torch.isfinite(ema.module.encoder.encoder_highres_alpha).all())
        model.eval()
        restored = fresh_config(TRAIN_CONFIG).model.eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            expected, actual = model(images), restored(images)
            for key in ("pred_logits", "pred_boxes"):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            deployed_model = copy.deepcopy(model).deploy()
            deployed = deployed_model(images)
            for key in ("pred_logits", "pred_boxes"):
                self.assertEqual(deployed[key].shape, expected[key].shape)
                self.assertTrue(torch.isfinite(deployed[key]).all())
            torch.testing.assert_close(deployed_model.encoder.encoder_highres_alpha,
                                       alpha, rtol=0, atol=0)
            self.assertEqual(sum(p.numel() for p in deployed_model.parameters()), 10178202)
        print("High-res Residual added parameters: 1; log/deploy total: 10178202")

    def test_full_detector_amp_training_with_mixed_and_empty_gt(self):
        cfg = fresh_config(TRAIN_CONFIG)
        model = cfg.model.train()
        images = torch.rand(2, 3, 128, 128)
        for targets in (
            [{"labels": torch.tensor([1, 2]), "boxes": torch.tensor([
                [.3, .4, .2, .2], [.7, .6, .2, .3]])},
             {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)}],
            [{"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)}
             for _ in range(2)],
        ):
            model.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                outputs = model(images, targets)
            losses = cfg.criterion(copy_output(outputs), targets)
            self.assertTrue(all(torch.isfinite(v).all() for v in losses.values()))
            sum(losses.values()).backward()
            self.assertIsNotNone(model.encoder.encoder_highres_alpha.grad)
            self.assertTrue(torch.isfinite(model.encoder.encoder_highres_alpha.grad).all())
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
