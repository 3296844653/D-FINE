"""CPU checks for the restored SCB-S AQS experiment; no dataset or downloads."""

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
from src.zoo.dfine.dfine_decoder import AdaptiveQuerySelectionRefiner, DFINETransformer


def fresh_config(name):
    # Avoid the existing YAML loader's shared mutable default in test processes.
    with patch("src.core.yaml_config.load_config", lambda path: load_config(path, cfg={})):
        config = YAMLConfig(str(ROOT / "configs/dfine" / name))
    config.yaml_cfg["HGNetv2"]["pretrained"] = False
    config.yaml_cfg["eval_spatial_size"] = [128, 128]
    return config


def old_formula(module, query, logits):
    """Original 26b851d equation, kept here as a regression reference."""
    importance = logits.sigmoid().amax(dim=-1)
    threshold = module.threshold_logit.sigmoid().to(dtype=importance.dtype)
    soft = torch.sigmoid((importance - threshold) / module.temperature)
    hard = (soft >= 0.5).to(dtype=soft.dtype)
    gate = hard.detach() - soft.detach() + soft
    context = (query * gate.unsqueeze(-1)).sum(dim=1, keepdim=True)
    context = context / gate.sum(dim=1, keepdim=True).clamp_min(1.0).unsqueeze(-1)
    context = context.expand(-1, query.shape[1], -1)
    delta = module.fuse(torch.cat([module.norm(query), context], dim=-1))
    return query + module.residual_scale.tanh().to(query.dtype) * gate.unsqueeze(-1).to(query.dtype) * delta


def copy_output(value):
    if isinstance(value, dict):
        return {key: copy_output(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_output(item) for item in value]
    return value


class AQSRefineTests(unittest.TestCase):
    def test_old_equation_zero_initialization_and_hard_selection(self):
        torch.manual_seed(10)
        module = AdaptiveQuerySelectionRefiner(32)
        query = torch.randn(2, 7, 32)
        logits = torch.tensor([[[1., -2., -3.], [-2., -2., -2.], [0., -1., -1.],
                                [2., 0., -2.], [-3., -3., -3.], [1., 0., 0.],
                                [-1., -1., -1.]]]).expand(2, -1, -1).clone()
        torch.testing.assert_close(module(query, logits), query, rtol=0, atol=0)
        # Start-up gradients intentionally reach the zero final projection
        # first; the rest learns after this projection has changed.
        module(query, logits).square().sum().backward()
        self.assertGreater(module.fuse[-1].weight.grad.abs().sum().item(), 0)
        torch.nn.init.normal_(module.fuse[-1].weight, std=.1)
        torch.nn.init.normal_(module.fuse[-1].bias, std=.1)
        actual = module(query, logits)
        torch.testing.assert_close(actual, old_formula(module, query, logits), rtol=0, atol=0)
        selected = logits.sigmoid().amax(-1) >= .5
        torch.testing.assert_close(actual[~selected], query[~selected], rtol=0, atol=0)
        self.assertGreater((actual[selected] - query[selected]).abs().sum().item(), 0)
        # No selected queries: safe normalization, exact identity, finite data.
        torch.testing.assert_close(module(query, torch.full_like(logits, -20)), query, rtol=0, atol=0)
        self.assertEqual(module(query[:, :0], logits[:, :0]).shape, (2, 0, 32))
        # Pooling is permutation-equivariant and independent between images.
        order = torch.tensor([6, 0, 3, 1, 4, 2, 5])
        torch.testing.assert_close(module(query[:, order], logits[:, order]), actual[:, order])
        modified = query.clone(); modified[0] += 100
        torch.testing.assert_close(module(modified, logits)[1], actual[1], rtol=0, atol=0)

    def test_straight_through_gradients_and_cpu_amp(self):
        module = AdaptiveQuerySelectionRefiner(32)
        torch.nn.init.normal_(module.fuse[-1].weight, std=.1)
        query = torch.randn(2, 9, 32, requires_grad=True)
        logits = torch.randn(2, 9, 3, requires_grad=True)
        logits.data[:, :2] = 1
        with torch.autocast("cpu", dtype=torch.bfloat16):
            refined = module(query, logits)
            loss = (refined.float() * torch.randn_like(query)).sum()
        self.assertTrue(torch.isfinite(refined).all())
        loss.backward()
        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(query.grad).all())

    def test_dn_and_detection_contexts_are_separate(self):
        transformer = DFINETransformer(
            num_classes=3, hidden_dim=32, feat_channels=[32, 32, 32],
            num_layers=3, layer_scale=1, use_aqs_refine=True,
        )
        decoder = transformer.decoder.train()
        module = decoder.query_cls_refiner
        torch.nn.init.normal_(module.fuse[-1].weight, std=.1)
        query, logits = torch.randn(2, 8, 32), torch.ones(2, 8, 3)
        meta = {"dn_num_split": [3, 5]}
        actual = decoder._apply_aqs_refiner(query, logits, meta)
        expected = torch.cat([module(query[:, :3], logits[:, :3]),
                              module(query[:, 3:], logits[:, 3:])], dim=1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        modified = query.clone(); modified[:, :3] += 100
        torch.testing.assert_close(decoder._apply_aqs_refiner(modified, logits, meta)[:, 3:],
                                   actual[:, 3:], rtol=0, atol=0)
        for bad in ({}, {"dn_num_split": [1, 2]}, {"dn_num_split": [3, 5.0]},
                    {"dn_num_split": [8]}, {"dn_num_split": [-1, 9]}):
            with self.assertRaises(ValueError):
                decoder._apply_aqs_refiner(query, logits, bad)
        decoder.eval()
        torch.testing.assert_close(decoder._apply_aqs_refiner(query, logits, None),
                                   module(query, logits), rtol=0, atol=0)

    def test_invalid_parameters_and_independent_experiment_guards(self):
        for kwargs in ({"threshold": 0}, {"threshold": 1}, {"threshold": float("nan")},
                       {"temperature": 0}, {"temperature": float("inf")},
                       {"residual_init": float("nan")}, {"hidden_dim": 0}):
            args = dict(hidden_dim=32); args.update(kwargs)
            with self.assertRaises(ValueError):
                AdaptiveQuerySelectionRefiner(**args)
        module = AdaptiveQuerySelectionRefiner(32)
        with self.assertRaises(ValueError):
            module(torch.randn(2, 8, 32), torch.randn(2, 7, 3))
        for flag in ("use_task_decoupled_heads", "use_local_evidence_cls",
                     "use_query_validity", "use_query_objectness", "use_p2_roi_cls",
                     "use_p2_query_init", "use_decoder_local_query_cls",
                     "use_rw_spatial_relation", "use_rw_isolated_specialist"):
            with self.assertRaisesRegex(ValueError, "ablated separately"):
                DFINETransformer(use_aqs_refine=True, **{flag: True})
        for kwargs in ({"eval_idx": 0}, {"layer_scale": 2}):
            with self.assertRaisesRegex(ValueError, "final decoder layer"):
                DFINETransformer(use_aqs_refine=True, **kwargs)

    def test_yaml_only_changes_aqs_and_baseline_initialization_is_preserved(self):
        base_cfg = fresh_config("dfine_s_scbs_hrw.yml")
        trial_cfg = fresh_config("dfine_s_scbs_hrw_aqs.yml")
        expected, actual = copy.deepcopy(base_cfg.yaml_cfg), copy.deepcopy(trial_cfg.yaml_cfg)
        for cfg in (expected, actual):
            cfg.pop("__include__", None)
        expected["DFINETransformer"].update(
            use_aqs_refine=True, aqs_threshold=.5,
            aqs_temperature=.1, aqs_residual_init=.05,
        )
        self.assertEqual(expected, actual)
        diag = fresh_config("dfine_s_scbs_hrw_aqs_diagnostics.yml")
        self.assertTrue(diag.yaml_cfg["DFINETransformer"]["use_aqs_refine"])
        self.assertTrue(diag.yaml_cfg["export_confusion_matrix"])
        self.assertTrue(diag.yaml_cfg["export_query_diagnostics"])
        self.assertEqual(diag.yaml_cfg["query_diagnostic_conf_thresh"], .5)
        torch.manual_seed(21); baseline = base_cfg.model.eval()
        baseline_rng = torch.get_rng_state().clone()
        torch.manual_seed(21); trial = trial_cfg.model.eval()
        torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
        new_prefix = "decoder.decoder.query_cls_refiner."
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(trial.state_dict()[key], value, rtol=0, atol=0)
        extra = set(trial.state_dict()) - set(baseline.state_dict())
        self.assertTrue(extra)
        self.assertTrue(all(k.startswith(new_prefix) for k in extra))
        self.assertIsNone(baseline.decoder.decoder.query_cls_refiner)
        self.assertFalse(trial.encoder.use_mffe)
        self.assertFalse(trial_cfg.criterion.use_pairwise_ce)
        self.assertFalse(trial_cfg.criterion.use_class_margin)
        self.assertFalse(trial_cfg.criterion.use_class_balanced_vfl)
        images = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            original, enhanced = baseline(images), trial(images)
        for key in ("pred_logits", "pred_boxes"):
            torch.testing.assert_close(enhanced[key], original[key], rtol=0, atol=0)

    def test_independent_110_config_changes_only_aqs_not_losses(self):
        base = fresh_config("dfine_s_scbs_hrw_110.yml")
        trial = fresh_config("dfine_s_scbs_hrw_110_aqs.yml")
        expected, actual = copy.deepcopy(base.yaml_cfg), copy.deepcopy(trial.yaml_cfg)
        for values in (expected, actual):
            values.pop("__include__", None)
            values.pop("output_dir", None)
        expected["DFINETransformer"].update(use_aqs_refine=True, aqs_threshold=.5,
                                            aqs_temperature=.1, aqs_residual_init=.05)
        self.assertEqual(expected, actual)
        self.assertEqual(trial.yaml_cfg["epochs"], 110)
        self.assertEqual(trial.yaml_cfg["train_dataloader"]["total_batch_size"], 32)
        self.assertEqual(trial.yaml_cfg["train_dataloader"]["collate_fn"]["stop_epoch"], 100)
        self.assertEqual(trial.yaml_cfg["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"], 100)
        self.assertEqual(trial.yaml_cfg["DFINECriterion"], base.yaml_cfg["DFINECriterion"])
        for name in ("use_pairwise_ce", "use_class_margin", "use_class_balanced_vfl",
                     "use_rw_query_calibration", "use_query_validity", "use_query_objectness",
                     "use_rw_isolated_specialist", "use_shape_iou"):
            self.assertFalse(getattr(trial.criterion, name), name)
        model = trial.model
        self.assertTrue(model.decoder.use_aqs_refine)
        self.assertFalse(model.decoder.use_rw_query_calibration)
        self.assertFalse(model.encoder.use_mffe)
        self.assertFalse(model.encoder.use_encoder_highres_residual)
        diag = fresh_config("dfine_s_scbs_hrw_110_aqs_diagnostics.yml")
        self.assertEqual(diag.yaml_cfg["epochs"], 110)
        self.assertTrue(diag.yaml_cfg["export_query_diagnostics"])
        self.assertTrue(diag.yaml_cfg["export_confusion_matrix"])
        self.assertEqual(diag.yaml_cfg["DFINECriterion"], base.yaml_cfg["DFINECriterion"])

    def test_final_layer_teacher_loss_optimizer_and_strict_checkpoint(self):
        base_cfg = fresh_config("dfine_s_scbs_hrw.yml")
        trial_cfg = fresh_config("dfine_s_scbs_hrw_aqs.yml")
        torch.manual_seed(30); baseline = base_cfg.model.train()
        torch.manual_seed(30); model = trial_cfg.model.train()
        branch = model.decoder.decoder.query_cls_refiner
        self.assertEqual(sum(p.numel() for p in branch.parameters()), 197634)
        # Force selected queries in the synthetic smoke test only; the real
        # baseline's original low-prior classification bias is NOT changed.
        with torch.no_grad():
            baseline.decoder.dec_score_head[-1].bias.fill_(1)
            model.decoder.dec_score_head[-1].bias.fill_(1)
        torch.nn.init.normal_(branch.fuse[-1].weight, std=.1)
        images = torch.rand(2, 3, 128, 128)
        targets = [
            {"labels": torch.tensor([0, 1, 2]), "boxes": torch.tensor([
                [.2, .3, .15, .2], [.5, .5, .2, .3], [.8, .7, .2, .2]])},
            {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)},
        ]
        torch.manual_seed(31); enhanced = model(images, targets)
        torch.manual_seed(31); original = baseline(images, targets)
        torch.testing.assert_close(enhanced["pred_boxes"], original["pred_boxes"], rtol=0, atol=0)
        self.assertGreater((enhanced["pred_logits"] - original["pred_logits"]).abs().sum().item(), 0)
        for aa, bb in zip(enhanced["aux_outputs"], original["aux_outputs"]):
            torch.testing.assert_close(aa["pred_logits"], bb["pred_logits"], rtol=0, atol=0)
            torch.testing.assert_close(aa["teacher_logits"], original["pred_logits"], rtol=0, atol=0)
            torch.testing.assert_close(aa["teacher_corners"], bb["teacher_corners"], rtol=0, atol=0)
        for group in ("pre_outputs",):
            for key in ("pred_logits", "pred_boxes"):
                torch.testing.assert_close(enhanced[group][key], original[group][key], rtol=0, atol=0)
        for aa, bb in zip(enhanced["enc_aux_outputs"], original["enc_aux_outputs"]):
            torch.testing.assert_close(aa["pred_logits"], bb["pred_logits"], rtol=0, atol=0)
        for aa, bb in zip(enhanced["dn_outputs"][:-1], original["dn_outputs"][:-1]):
            torch.testing.assert_close(aa["pred_logits"], bb["pred_logits"], rtol=0, atol=0)
        losses = trial_cfg.criterion(copy_output(enhanced), targets)
        old_losses = base_cfg.criterion(copy_output(original), targets)
        self.assertEqual(set(losses), set(old_losses))
        self.assertTrue(all(torch.isfinite(v).all() for v in losses.values()))
        sum(losses.values()).backward()
        for name, parameter in branch.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer = trial_cfg.optimizer
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(id(p) in ids for p in branch.parameters()))
        before = branch.fuse[-1].weight.detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
        optimizer.step()
        self.assertGreater((branch.fuse[-1].weight - before).abs().sum().item(), 0)
        model.eval()
        restored = fresh_config("dfine_s_scbs_hrw_aqs.yml").model.eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            expected, actual = model(images), restored(images)
            for key in ("pred_logits", "pred_boxes"):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            # Disabling refinement with identical weights cannot move boxes.
            model.decoder.use_aqs_refine = model.decoder.decoder.use_aqs_refine = False
            unrefined = model(images)
            torch.testing.assert_close(unrefined["pred_boxes"], expected["pred_boxes"], rtol=0, atol=0)
            model.decoder.use_aqs_refine = model.decoder.decoder.use_aqs_refine = True
            with torch.autocast("cpu", dtype=torch.bfloat16):
                amp = model(images)
            self.assertTrue(all(torch.isfinite(v).all() for v in amp.values()))
            deployed = copy.deepcopy(model).deploy()
            prediction = deployed(images)
            self.assertEqual(prediction["pred_logits"].shape, (2, 300, 3))
            self.assertTrue(all(torch.isfinite(v).all() for v in prediction.values()))
            self.assertEqual(sum(p.numel() for p in deployed.parameters()), 10375835)
        print("AQS added parameters: 197634; log/deploy total: 10375835")

    def test_full_detector_cpu_amp_training_and_empty_gt(self):
        cfg = fresh_config("dfine_s_scbs_hrw_aqs.yml")
        model = cfg.model.train()
        with torch.no_grad():
            model.decoder.dec_score_head[-1].bias.fill_(1)
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
            # Match det_engine: criterion is evaluated outside autocast.
            losses = cfg.criterion(copy_output(outputs), targets)
            self.assertTrue(all(torch.isfinite(v).all() for v in losses.values()))
            sum(losses.values()).backward()
            for name, parameter in model.decoder.decoder.query_cls_refiner.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(model.decoder.decoder.query_cls_refiner.fuse[-1]
                               .weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
