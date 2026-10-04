"""CPU regression checks; no dataset changes, downloads or detector training.

Run from repository root:
    python tools/diagnostics/test_pairwise_ce.py
"""
import copy
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core import YAMLConfig
from src.zoo.dfine.dfine_criterion import DFINECriterion
from src.zoo.dfine.matcher import HungarianMatcher


def criterion(**kwargs):
    return DFINECriterion(
        matcher=HungarianMatcher(
            {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2}, use_focal_loss=True
        ),
        weight_dict={"loss_vfl": 1, "loss_bbox": 5, "loss_giou": 2,
                     "loss_fgl": 0.15, "loss_ddf": 1.5},
        losses=["vfl", "boxes", "local"], num_classes=3, reg_max=32,
        alpha=0.75, gamma=2.0, **kwargs,
    )


def copy_output_tree(value):
    # Criterion adds dictionary metadata; retain the same differentiable tensors.
    if isinstance(value, dict):
        return {key: copy_output_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_output_tree(item) for item in value]
    return value


class PairwiseCETests(unittest.TestCase):
    def test_value_gradients_and_ignored_queries(self):
        logits = torch.tensor([[[4., 0., 1.], [2., 1., 0.],
                                [4., 2., 3.], [0., 8., -8.]]], requires_grad=True)
        targets = [{"labels": torch.tensor([1, 2, 0])}]
        indices = [(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))]
        loss = criterion(use_pairwise_ce=True).loss_labels_pairwise_ce(
            {"pred_logits": logits}, targets, indices, 3
        )
        expected = F.cross_entropy(
            torch.tensor([[0., 1.], [1., 0.]]), torch.tensor([0, 1]), reduction="sum"
        ) / 3
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertLess(logits.grad[0, 0, 1].item(), 0)
        self.assertGreater(logits.grad[0, 0, 2].item(), 0)
        self.assertGreater(logits.grad[0, 1, 1].item(), 0)
        self.assertLess(logits.grad[0, 1, 2].item(), 0)
        self.assertEqual(logits.grad[:, :, 0].abs().sum().item(), 0)
        self.assertEqual(logits.grad[:, 2:].abs().sum().item(), 0)

    def test_empty_and_no_pair(self):
        for labels, src in (([], []), ([0], [0])):
            logits = torch.randn(1, 3, 3, requires_grad=True)
            indices = [(torch.tensor(src, dtype=torch.long),
                        torch.arange(len(labels), dtype=torch.long))]
            loss = criterion(use_pairwise_ce=True).loss_labels_pairwise_ce(
                {"pred_logits": logits}, [{"labels": torch.tensor(labels, dtype=torch.long)}],
                indices, max(1, len(labels)),
            )
            loss.backward()
            self.assertEqual(loss.item(), 0)
            self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_quality_shift_cancels_and_amp(self):
        logits = torch.randn(1, 2, 3, requires_grad=True)
        quality = torch.randn(1, 2, 1, requires_grad=True)
        targets = [{"labels": torch.tensor([1, 2])}]
        indices = [(torch.arange(2), torch.arange(2))]
        c = criterion(use_pairwise_ce=True)
        original = c.loss_labels_pairwise_ce({"pred_logits": logits}, targets, indices, 2)
        shifted = c.loss_labels_pairwise_ce({"pred_logits": logits + quality}, targets, indices, 2)
        torch.testing.assert_close(original, shifted)
        shifted.backward()
        self.assertLess(quality.grad.abs().max().item(), 1e-7)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = c.loss_labels_pairwise_ce(
                {"pred_logits": logits.to(torch.bfloat16)}, targets, indices, 2
            )
        self.assertTrue(torch.isfinite(loss))

    def test_guards_and_no_parameters(self):
        self.assertEqual(list(criterion().state_dict()), list(criterion(use_pairwise_ce=True).state_dict()))
        for kwargs in ({"pairwise_ce_classes": [1, 1]}, {"pairwise_ce_classes": [1, 3]},
                       {"pairwise_ce_classes": [1.0, 2]}, {"pairwise_ce_weight": float("nan")},
                       {"pairwise_ce_weight": 0}, {"use_class_margin": True},
                       {"use_class_balanced_vfl": True}):
            with self.assertRaises(ValueError):
                criterion(use_pairwise_ce=True, **kwargs)

    def test_full_detector_dn_aux_loss_and_baseline_parity(self):
        torch.manual_seed(0)
        cfg = YAMLConfig(str(ROOT / "configs/dfine/dfine_s_scb3s_3cls_bs32_rw_pairwise_ce.yml"))
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        cfg.yaml_cfg["eval_spatial_size"] = [128, 128]
        model = cfg.model.train()
        enhanced = cfg.criterion
        self.assertTrue(enhanced.use_pairwise_ce)
        self.assertEqual(cfg.yaml_cfg["train_dataloader"]["total_batch_size"], 32)
        baseline = copy.deepcopy(enhanced)
        baseline.use_pairwise_ce = False
        targets = [
            {"labels": torch.tensor([0, 1, 2]),
             "boxes": torch.tensor([[.2, .3, .15, .2], [.5, .5, .2, .3], [.8, .7, .2, .2]])},
            {"labels": torch.tensor([1, 2]),
             "boxes": torch.tensor([[.3, .4, .2, .2], [.7, .6, .25, .3]])},
        ]
        images = torch.rand(2, 3, 128, 128)
        outputs = model(images, targets)
        self.assertIn("dn_outputs", outputs)
        self.assertIn("aux_outputs", outputs)
        original_losses = baseline(copy_output_tree(outputs), targets)
        new_losses = enhanced(copy_output_tree(outputs), targets)
        self.assertEqual(set(new_losses) - set(original_losses), {"loss_pairwise_ce"})
        for key in original_losses:
            torch.testing.assert_close(original_losses[key], new_losses[key], rtol=0, atol=0)
        # The new loss goes through classification, but not bbox/LQE heads.
        model.zero_grad(set_to_none=True)
        new_losses["loss_pairwise_ce"].backward(retain_graph=True)
        self.assertGreater(model.decoder.dec_score_head[-1].weight.grad.abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            if "bbox_head" in name or "lqe_layers" in name:
                if parameter.grad is not None:
                    self.assertLess(parameter.grad.abs().max().item(), 1e-7, name)
        model.zero_grad(set_to_none=True)
        sum(new_losses.values()).backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        model.eval()
        with torch.no_grad():
            prediction = model(images)
        self.assertEqual(tuple(prediction["pred_logits"].shape), (2, 300, 3))
        self.assertEqual(tuple(prediction["pred_boxes"].shape), (2, 300, 4))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
