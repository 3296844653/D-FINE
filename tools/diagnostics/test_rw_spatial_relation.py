"""CPU checks for the independent SCB-S spatial relation experiment."""
import copy
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core import YAMLConfig
from src.zoo.dfine.dfine_decoder import RWSpatialRelationRefiner, DFINETransformer


class RWRelationTests(unittest.TestCase):
    def test_branch_initialization_gradient_geometry_and_amp(self):
        branch = RWSpatialRelationRefiner(32, 3, 16, 3)
        feature = torch.randn(2, 32, 8, 8, requires_grad=True)
        boxes = torch.rand(2, 7, 4, requires_grad=True)
        query = torch.randn(2, 7, 32, requires_grad=True)
        self.assertEqual(branch(feature, boxes, query).abs().sum().item(), 0)
        torch.nn.init.normal_(branch.pair_head[-1].weight, std=.05)
        delta = branch(feature, boxes, query)
        self.assertEqual(delta[:, :, 0].abs().sum().item(), 0)
        torch.testing.assert_close(delta[:, :, 1], -delta[:, :, 2], rtol=0, atol=0)
        delta[:, :, 1].square().sum().backward()
        self.assertIsNone(boxes.grad)
        self.assertGreater(feature.grad.abs().sum().item(), 0)
        self.assertGreater(query.grad.abs().sum().item(), 0)
        for p in branch.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            amp = branch(feature, boxes, query)
        self.assertTrue(torch.isfinite(amp).all())

    def test_guards(self):
        for kwargs in ({"grid_size": 1}, {"relation_dim": 10},
                       {"class_ids": [1, 1]}, {"class_ids": [1, 3]},
                       {"class_ids": [1., 2]}):
            with self.assertRaises(ValueError):
                RWSpatialRelationRefiner(32, 3, **kwargs)
        with self.assertRaises(ValueError):
            DFINETransformer(num_classes=3, use_rw_spatial_relation=True,
                             use_decoder_local_query_cls=True)

    def test_full_model_parity_dn_loss_optimizer_and_deploy(self):
        torch.manual_seed(4)
        cfg = YAMLConfig(str(ROOT / "configs/dfine/dfine_s_scb3s_3cls_bs32_rw_spatial_relation.yml"))
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        cfg.yaml_cfg["eval_spatial_size"] = [128, 128]
        model = cfg.model.eval()
        loss_fn = cfg.criterion
        self.assertFalse(loss_fn.use_pairwise_ce)
        self.assertFalse(loss_fn.use_class_margin)
        self.assertEqual(cfg.yaml_cfg["epochs"], 132)
        self.assertEqual(cfg.yaml_cfg["train_dataloader"]["total_batch_size"], 32)
        branch = model.decoder.rw_spatial_relation
        print("Added parameters:", sum(p.numel() for p in branch.parameters()))
        print("Total parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        # Original architecture must accept exactly all non-branch keys.
        base_cfg = YAMLConfig(str(ROOT / "configs/dfine/dfine_s_scb3s_3cls_bs32.yml"))
        base_cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        base_cfg.yaml_cfg["eval_spatial_size"] = [128, 128]
        # Original load_config has a mutable default cache; explicitly isolate
        # the second model in this single-process regression test.
        base_cfg.yaml_cfg["DFINETransformer"]["use_rw_spatial_relation"] = False
        base = base_cfg.model.eval()
        baseline_state = {k: v for k, v in model.state_dict().items()
                          if not k.startswith("decoder.rw_spatial_relation.")}
        base.load_state_dict(baseline_state, strict=True)
        self.assertEqual(sum(p.numel() for p in model.parameters()) -
                         sum(p.numel() for p in base.parameters()), 75713)
        images = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            prediction = model(images)
            original = base(images)
        for key in ("pred_logits", "pred_boxes"):
            torch.testing.assert_close(prediction[key], original[key], rtol=0, atol=0)
        self.assertEqual(tuple(prediction["pred_logits"].shape), (2, 300, 3))
        # Nonzero branch changes only final ordinary classification, not DN/aux
        # scores, original distillation teacher, or the direct bbox computation.
        torch.nn.init.normal_(branch.pair_head[-1].weight, std=.01)
        targets = [
            {"labels": torch.tensor([0, 1, 2]),
             "boxes": torch.tensor([[.2, .3, .15, .2], [.5, .5, .2, .3], [.8, .7, .2, .2]])},
            {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)},
        ]
        model.train()
        torch.manual_seed(8)
        outputs = model(images, targets)
        model.decoder.use_rw_spatial_relation = False
        torch.manual_seed(8)
        unchanged = model(images, targets)
        model.decoder.use_rw_spatial_relation = True
        torch.testing.assert_close(outputs["pred_boxes"], unchanged["pred_boxes"], rtol=0, atol=0)
        torch.testing.assert_close(outputs["pred_logits"][..., 0], unchanged["pred_logits"][..., 0], rtol=0, atol=0)
        self.assertGreater((outputs["pred_logits"] - unchanged["pred_logits"]).abs().sum().item(), 0)
        for group in ("dn_outputs", "aux_outputs"):
            for enhanced, old in zip(outputs[group], unchanged[group]):
                for key in ("pred_logits", "teacher_logits"):
                    torch.testing.assert_close(enhanced[key], old[key], rtol=0, atol=0)
        losses = loss_fn(outputs, targets)
        self.assertFalse(any("pairwise_ce" in k or "margin" in k for k in losses))
        sum(losses.values()).backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all(), name)
        self.assertGreater(branch.pair_head[-1].weight.grad.abs().sum().item(), 0)
        optimizer = cfg.optimizer
        optimizer_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(optimizer_ids), len(set(optimizer_ids)))
        self.assertTrue(all(id(p) in optimizer_ids for p in branch.parameters()))
        optimizer.step()
        model.eval()
        with torch.no_grad():
            deployed_model = copy.deepcopy(model).deploy()
            deployed = deployed_model(images)
            deployed_model.decoder.use_rw_spatial_relation = False
            deployed_original = deployed_model(images)
        for key in ("pred_logits", "pred_boxes"):
            self.assertTrue(torch.isfinite(deployed[key]).all())
        self.assertEqual(tuple(deployed["pred_logits"].shape), (2, 300, 3))
        torch.testing.assert_close(deployed["pred_boxes"], deployed_original["pred_boxes"], rtol=0, atol=0)
        torch.testing.assert_close(deployed["pred_logits"][..., 0], deployed_original["pred_logits"][..., 0], rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
