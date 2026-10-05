"""Regression checks for isolated, score-preserving read/write expertise."""
import copy
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.solver.det_engine import clip_detector_gradients
from src.zoo.dfine.dfine_decoder import RWIsolatedSpecialist, DFINETransformer
from test_pairwise_ce import criterion, copy_output_tree


def config(name):
    # Isolate the original loader's mutable default between in-process configs.
    load_config.__defaults__[0].clear()
    c = YAMLConfig(str(ROOT / "configs/dfine" / name))
    c.yaml_cfg["HGNetv2"]["pretrained"] = False
    c.yaml_cfg["eval_spatial_size"] = [128, 128]
    return c


class IsolatedSpecialistTests(unittest.TestCase):
    def test_expert_gradient_isolation_and_amp(self):
        expert = RWIsolatedSpecialist(32, 3, 16, 3)
        tensors = [torch.randn(2, 32, 8, 8, requires_grad=True),
                   torch.rand(2, 7, 4, requires_grad=True),
                   torch.randn(2, 7, 32, requires_grad=True),
                   torch.randn(2, 7, 3, requires_grad=True)]
        pair = expert(*tensors)
        torch.testing.assert_close(expert.refine_logits(tensors[-1], pair), tensors[-1], rtol=0, atol=0)
        torch.nn.init.normal_(expert.refiner.pair_head[-1].weight, std=.1)
        F.cross_entropy(expert(*tensors).reshape(-1, 2), torch.arange(14) % 2).backward()
        for t in tensors:
            self.assertIsNone(t.grad)
        for p in expert.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            pair = expert(*tensors)
            corrected = expert.refine_logits(tensors[-1], pair)
        self.assertTrue(torch.isfinite(pair).all())
        torch.testing.assert_close(corrected.sort(-1).values, tensors[-1].sort(-1).values, rtol=0, atol=0)

    def test_score_preserving_swaps_and_ties(self):
        expert = RWIsolatedSpecialist(32, 3, 16, 3)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            logits = torch.tensor([[[4., 3., 2.], [-2., 3., 1.],
                                    [-2., 1., 3.], [4., 2., 2.]]], dtype=dtype)
            pair = torch.tensor([[[-1., 1.], [-1., 1.], [1., -1.], [1., -1.]]])
            refined = expert.refine_logits(logits, pair)
            torch.testing.assert_close(refined[..., 0], logits[..., 0], rtol=0, atol=0)
            torch.testing.assert_close(refined.sort(-1).values, logits.sort(-1).values, rtol=0, atol=0)
            self.assertEqual(refined[0, 0].argmax().item(), 0)
            self.assertEqual(refined[0, 1].argmax().item(), 2)
            self.assertEqual(refined[0, 2].argmax().item(), 1)
            torch.testing.assert_close(refined[0, 3], logits[0, 3], rtol=0, atol=0)

    def test_loss_filters_empty_cases_and_guards(self):
        c = criterion(use_rw_isolated_specialist=True)
        logits = torch.randn(1, 4, 2, requires_grad=True)
        boxes = torch.tensor([[[.2,.2,.1,.1], [.5,.5,.1,.1],
                               [.8,.8,.1,.1], [.2,.8,.1,.1]]], requires_grad=True)
        gt = boxes.detach()[0].clone()
        gt[3] = torch.tensor([.9,.1,.1,.1])
        targets = [{"labels": torch.tensor([1,2,0,1]), "boxes": gt}]
        out = {"pred_logits": torch.zeros(1,4,3), "pred_boxes": boxes,
               "rw_specialist_logits": logits, "rw_specialist_class_ids": (1,2)}
        indices = [(torch.arange(4), torch.arange(4))]
        loss = c.loss_rw_isolated_specialist(out, targets, indices, 4)
        expected = F.cross_entropy(logits[0,:2], torch.tensor([0,1]), reduction="sum") / 4
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(boxes.grad)
        self.assertEqual(logits.grad[0,2:].abs().sum().item(), 0)
        for labels, gt_boxes in (([], torch.empty(0,4)), ([0], gt[:1])):
            empty_targets = [{"labels": torch.tensor(labels,dtype=torch.long), "boxes": gt_boxes}]
            idx = [(torch.arange(len(labels)), torch.arange(len(labels)))]
            zero = c.loss_rw_isolated_specialist(out, empty_targets, idx, max(1,len(labels)))
            self.assertEqual(zero.item(), 0)
        for kwargs in ({"rw_specialist_class_ids":[1,1]}, {"rw_specialist_class_ids":[1,3]},
                       {"rw_specialist_weight":0}, {"rw_specialist_min_iou":float("nan")},
                       {"use_pairwise_ce":True}, {"use_class_margin":True}):
            with self.assertRaises(ValueError):
                criterion(use_rw_isolated_specialist=True, **kwargs)
        bad = dict(out, rw_specialist_class_ids=(2,1))
        with self.assertRaises(ValueError):
            c.loss_rw_isolated_specialist(bad, targets, indices, 4)
        with self.assertRaises(ValueError):
            DFINETransformer(num_classes=3, use_rw_spatial_relation=True, use_rw_isolated_specialist=True)

    def test_separate_clipping_preserves_baseline_clip_factor(self):
        model = torch.nn.Module()
        model.backbone = torch.nn.Linear(2,2,bias=False)
        model.decoder = torch.nn.Module()
        model.decoder.use_rw_isolated_specialist = True
        model.decoder.rw_isolated_specialist = torch.nn.Linear(2,2,bias=False)
        model.backbone.weight.grad = torch.ones_like(model.backbone.weight) * 3
        model.decoder.rw_isolated_specialist.weight.grad = torch.ones(2,2) * 100
        expected = torch.nn.Parameter(model.backbone.weight.detach().clone())
        expected.grad = model.backbone.weight.grad.clone()
        torch.nn.utils.clip_grad_norm_([expected], .1)
        clip_detector_gradients(model, .1)
        torch.testing.assert_close(model.backbone.weight.grad, expected.grad, rtol=0, atol=0)
        self.assertLessEqual(model.decoder.rw_isolated_specialist.weight.grad.norm().item(), .10001)

    def test_full_detector_original_loss_gradient_and_inference(self):
        torch.manual_seed(11)
        cfg = config("dfine_s_scb3s_3cls_bs32_rw_isolated_specialist.yml")
        model = cfg.model
        c = cfg.criterion
        torch.manual_seed(11)
        base_cfg = config("dfine_s_scb3s_3cls_bs32.yml")
        base = base_cfg.model
        base_state = base.state_dict()
        for k, v in base_state.items():
            torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
        self.assertFalse(base.decoder.use_rw_isolated_specialist)
        self.assertTrue(c.use_rw_isolated_specialist)
        self.assertFalse(c.use_pairwise_ce)
        self.assertEqual(cfg.yaml_cfg["epochs"],132)
        self.assertEqual(cfg.yaml_cfg["train_dataloader"]["total_batch_size"],32)
        branch = model.decoder.rw_isolated_specialist
        self.assertEqual(sum(p.numel() for p in branch.parameters()),75713)
        print("Added parameters:",sum(p.numel() for p in branch.parameters()))
        model.eval(); base.eval()
        images = torch.rand(2,3,128,128)
        with torch.no_grad():
            a,b = model(images),base(images)
        for key in ("pred_logits","pred_boxes"):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        torch.nn.init.normal_(branch.refiner.pair_head[-1].weight,std=.1)
        model.train(); base.train()
        targets = [{"labels":torch.tensor([0,1,2]),
                    "boxes":torch.tensor([[.2,.3,.15,.2],[.5,.5,.2,.3],[.8,.7,.2,.2]])},
                   {"labels":torch.empty(0,dtype=torch.long),"boxes":torch.empty(0,4)}]
        torch.manual_seed(12); outputs = model(images,targets)
        torch.manual_seed(12); old = base(images,targets)
        for key in ("pred_logits","pred_boxes"):
            torch.testing.assert_close(outputs[key],old[key],rtol=0,atol=0)
        for group in ("aux_outputs","dn_outputs","enc_aux_outputs"):
            for aa,bb in zip(outputs[group],old[group]):
                torch.testing.assert_close(aa["pred_logits"],bb["pred_logits"],rtol=0,atol=0)
        # Use zero IoU cut here to guarantee active samples on an untrained
        # synthetic model; the 0.5 filtering rule is tested separately above.
        c.rw_specialist_min_iou = 0
        new_losses = c(copy_output_tree(outputs),targets)
        old_losses = base_cfg.criterion(copy_output_tree(old),targets)
        self.assertEqual(set(new_losses)-set(old_losses),{"loss_rw_specialist"})
        for key in old_losses:
            torch.testing.assert_close(new_losses[key],old_losses[key],rtol=0,atol=0)
        new_losses["loss_rw_specialist"].backward(retain_graph=True)
        for name,p in model.named_parameters():
            if not name.startswith("decoder.rw_isolated_specialist."):
                self.assertIsNone(p.grad,name)
        self.assertGreater(branch.refiner.pair_head[-1].weight.grad.abs().sum().item(),0)
        model.zero_grad(set_to_none=True)
        sum(new_losses.values()).backward()
        sum(old_losses.values()).backward()
        for name,p in base.named_parameters():
            gradient = dict(model.named_parameters())[name].grad
            if p.grad is None:
                self.assertIsNone(gradient)
            else:
                torch.testing.assert_close(gradient,p.grad,rtol=0,atol=0)
        for p in model.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())
        optimizer = cfg.optimizer
        included = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(included),len(set(included)))
        self.assertTrue(all(id(p) in included for p in branch.parameters()))
        clip_detector_gradients(model,.1)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            refined = model(images)
            # Keep the architecture/state_dict for same-checkpoint evaluation.
            model.decoder.rw_specialist_apply_at_eval = False
            original = model(images)
            model.decoder.rw_specialist_apply_at_eval = True
        self.assertEqual(tuple(refined["pred_logits"].shape),(2,300,3))
        torch.testing.assert_close(refined["pred_boxes"],original["pred_boxes"],rtol=0,atol=0)
        torch.testing.assert_close(refined["pred_logits"][...,0],original["pred_logits"][...,0],rtol=0,atol=0)
        torch.testing.assert_close(refined["pred_logits"].sort(-1).values,original["pred_logits"].sort(-1).values,rtol=0,atol=0)
        with torch.no_grad():
            deployed = copy.deepcopy(model).deploy()(images)
        self.assertTrue(torch.isfinite(deployed["pred_logits"]).all())


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
