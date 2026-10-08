"""CPU regression tests: no dataset edits, downloads, or full training."""

import io
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.solver.det_engine import clip_detector_gradients
from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.rw_query_calibration import RWQueryCalibration
from tools.diagnostics.test_pairwise_ce import criterion, copy_output_tree


def config(name):
    with patch("src.core.yaml_config.load_config", lambda path: load_config(path, cfg={})):
        cfg = YAMLConfig(str(ROOT / "configs/dfine" / name))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["eval_spatial_size"] = [128, 128]
    return cfg


class RWCalibrationTests(unittest.TestCase):
    def test_identity_formula_bounds_and_other_class(self):
        head = RWQueryCalibration(32, 3, 4, dim=16)
        query, boxes, logits, corners = torch.randn(2, 6, 32), torch.rand(2, 6, 4), torch.randn(2, 6, 3), torch.randn(2, 6, 20)
        pair = head(query, boxes, logits, corners)
        torch.testing.assert_close(pair, logits[..., [1, 2]], rtol=0, atol=0)
        torch.testing.assert_close(head.refine_logits(logits, pair), logits, rtol=0, atol=0)
        with torch.no_grad():
            head.fuse[-1].bias.copy_(torch.tensor([1., -1.]))
        pair = head(query, boxes, logits, corners)
        expected = torch.tensor([-1., 2.]) * torch.tensor(1.).tanh()
        torch.testing.assert_close(pair - logits[..., [1, 2]], expected.expand_as(pair))
        self.assertLessEqual((pair - logits[..., [1, 2]]).abs().max().item(), 2.)
        torch.testing.assert_close(head.refine_logits(logits, pair)[..., 0], logits[..., 0], rtol=0, atol=0)
        # A contrast term can flip a wrong pair, not just boost both scores.
        with torch.no_grad():
            head.fuse[-1].bias.copy_(torch.tensor([0., -10.]))
        example = torch.tensor([[[-3., 1., 0.]]]).expand(2, 6, -1).clone()
        corrected = head.refine_logits(example, head(query, boxes, example, corners))
        self.assertTrue((corrected.argmax(-1) == 2).all())

    def test_neighbors_exclude_self_gray_overlap_and_other_images(self):
        head = RWQueryCalibration(32, 3, 4, dim=16, neighbor_count=2)
        boxes = torch.tensor([[[.2,.2,.15,.15], [.2,.2,.15,.15], [.26,.2,.15,.15], [.8,.8,.1,.1]]]).expand(2,-1,-1).clone()
        logits = torch.ones(2,4,3)
        weights = head.neighbor_weights(boxes, logits)
        self.assertEqual(weights.diagonal(dim1=1, dim2=2).sum().item(), 0.)
        self.assertEqual(weights[0,0,1].item(), 1.)
        self.assertEqual(weights[0,0,2].item(), 0.)
        self.assertEqual(weights[0,3].sum().item(), 0.)
        boxes[0] += 2
        torch.testing.assert_close(head.neighbor_weights(boxes, logits)[1], weights[1], rtol=0, atol=0)
        self.assertEqual(head.neighbor_weights(boxes[:,:1], logits[:,:1]).sum().item(), 0.)
        quality = head.quality_features(torch.zeros(2,4,20))
        torch.testing.assert_close(quality[...,:4], torch.ones(2,4,4))
        torch.testing.assert_close(quality[...,4:], torch.full((2,4,4), .2))

    def test_detached_inputs_and_amp_gradients(self):
        head = RWQueryCalibration(32,3,4,dim=16)
        torch.nn.init.normal_(head.fuse[-1].weight, std=.1)
        tensors = [torch.randn(2,7,32,requires_grad=True), torch.rand(2,7,4,requires_grad=True),
                   torch.randn(2,7,3,requires_grad=True), torch.randn(2,7,20,requires_grad=True)]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            pair = head(*tensors)
            loss = F.binary_cross_entropy_with_logits(pair, torch.rand_like(pair))
        loss.backward()
        self.assertTrue(torch.isfinite(pair).all())
        for tensor in tensors:
            self.assertIsNone(tensor.grad)
        for name, parameter in head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_clean_one_to_one_quality_targets_and_hard_negatives(self):
        c = criterion(use_rw_query_calibration=True, rw_calibration_min_negatives=2,
                      rw_calibration_negative_ratio=1)
        boxes = torch.tensor([[[.2,.2,.15,.15], [.6,.2,.15,.15], [.5,.8,.1,.1],
                               [.2,.2,.15,.15], [.9,.9,.1,.1], [.1,.7,.05,.05],
                               [.26,.2,.15,.15], [.8,.7,.05,.05]]], requires_grad=True)
        gt = torch.cat((boxes.detach()[0,:3], torch.tensor([[.05,.9,.05,.05]])))
        targets = [{"labels":torch.tensor([1,2,0,1]), "boxes":gt}]
        base_logits = torch.zeros(1,8,3)
        base_logits[0,2,1:] = 3
        base_logits[0,5,1:] = 2
        base_logits[0,7,1:] = -5
        base_logits.requires_grad_()
        pair = torch.zeros(1,8,2, requires_grad=True)
        outputs = {"pred_boxes":boxes, "pred_logits":base_logits,
                   "rw_calibration_logits":pair, "rw_calibration_class_ids":(1,2)}
        indices = [(torch.tensor([0,1,2,4]), torch.arange(4))]
        loss = c.loss_rw_query_calibration(outputs, targets, indices, 4)
        torch.testing.assert_close(loss, torch.tensor(2.).log())
        loss.backward()
        self.assertIsNone(boxes.grad)
        self.assertIsNone(base_logits.grad)
        self.assertLess(pair.grad[0,0,0].item(), 0)
        self.assertLess(pair.grad[0,1,1].item(), 0)
        self.assertGreater(pair.grad[0,2].sum().item(), 0)
        self.assertGreater(pair.grad[0,5].sum().item(), 0)
        # Duplicate, poor matched box, gray overlap, easy unsampled negative.
        self.assertEqual(pair.grad[0,[3,4,6,7]].abs().sum().item(), 0)
        gt[0,0] += .03  # IoU=2/3: positive target is quality, not a hard 1.
        expected_targets = torch.tensor([[2/3,0.], [0.,1.]])
        modified = pair.detach().clone().fill_(.7).requires_grad_()
        outputs["rw_calibration_logits"] = modified
        expected = (F.binary_cross_entropy_with_logits(modified[0,:2], expected_targets, reduction="none").mean(-1).sum()
                    + 2 * F.binary_cross_entropy_with_logits(modified[0,[2,5]], torch.zeros(2,2))) / 4
        torch.testing.assert_close(c.loss_rw_query_calibration(outputs, targets, indices, 4), expected)

    def test_empty_gt_and_no_eligible_queries(self):
        c = criterion(use_rw_query_calibration=True)
        logits = torch.randn(1,2,2, requires_grad=True)
        boxes = torch.tensor([[[.3,.3,.1,.1], [.8,.8,.1,.1]]])
        outputs = {"pred_logits":torch.zeros(1,2,3), "pred_boxes":boxes,
                   "rw_calibration_logits":logits, "rw_calibration_class_ids":(1,2)}
        empty = [{"labels":torch.empty(0,dtype=torch.long), "boxes":torch.empty(0,4)}]
        indices = [(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long))]
        loss = c.loss_rw_query_calibration(outputs, empty, indices, 1)
        torch.testing.assert_close(loss, F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)))
        bad = [{"labels":torch.tensor([1,2]), "boxes":1-boxes[0]}]
        zero = c.loss_rw_query_calibration(outputs, bad, [(torch.arange(2),torch.arange(2))], 2)
        self.assertEqual(zero.item(), 0)
        zero.backward()
        self.assertEqual(logits.grad.abs().sum().item(), 0)

    def _check_cpu_match_indices_on_accelerator(self, device):
        """Use real SciPy matching: CPU indices plus accelerator masks/GT."""
        c = criterion(use_rw_query_calibration=True)
        boxes_cpu = torch.tensor([[[.2,.2,.15,.15], [.6,.2,.15,.15],
                                   [.5,.8,.1,.1], [.9,.9,.05,.05]]])
        classes_cpu = torch.tensor([[[-2.,2.,-2.],[-2.,-2.,2.],
                                     [2.,-2.,-2.],[-3.,-3.,-3.]]])
        pair_cpu = torch.tensor([[[.7,-.4],[-.2,.8],[.3,.1],[-.6,-.8]]],requires_grad=True)
        for empty in (False,True):
            target_cpu = [{"labels":torch.empty(0,dtype=torch.long) if empty else torch.tensor([1,2,0]),
                           "boxes":torch.empty(0,4) if empty else boxes_cpu[0,:3].clone()}]
            target_device = [{k:v.to(device) for k,v in t.items()} for t in target_cpu]
            output_device = {"pred_boxes":boxes_cpu.to(device),
                             "pred_logits":classes_cpu.to(device),
                             "rw_calibration_class_ids":(1,2)}
            # Matcher deliberately keeps the returned indices on CPU.
            matches = c.matcher(output_device,target_device)["indices"]
            snapshot = [(a.clone(),b.clone()) for a,b in matches]
            self.assertTrue(all(a.device.type == b.device.type == "cpu" for a,b in matches))
            output_cpu = {"pred_boxes":boxes_cpu,"pred_logits":classes_cpu,
                          "rw_calibration_logits":pair_cpu,"rw_calibration_class_ids":(1,2)}
            expected = c.loss_rw_query_calibration(output_cpu,target_cpu,matches,3)
            dtypes = [torch.float32,torch.float16] if device.type == "cuda" else [torch.float32]
            for dtype in dtypes:
                pair_device = pair_cpu.detach().to(device=device,dtype=dtype).requires_grad_()
                output_device["rw_calibration_logits"] = pair_device
                loss = c.loss_rw_query_calibration(output_device,target_device,matches,3)
                self.assertEqual(loss.device.type,device.type)
                torch.testing.assert_close(loss.cpu(),expected,rtol=1e-3,atol=1e-4)
                loss.backward()
                self.assertTrue(torch.isfinite(pair_device.grad).all())
                # Also accept CPU targets in the stand-alone loss, without
                # mutating the match list used by the original detector losses.
                other = c.loss_rw_query_calibration(output_device,target_cpu,matches,3)
                torch.testing.assert_close(other,loss,rtol=0,atol=0)
            for (a,b),(old_a,old_b) in zip(matches,snapshot):
                self.assertEqual(a.device.type,"cpu")
                self.assertEqual(b.device.type,"cpu")
                torch.testing.assert_close(a,old_a,rtol=0,atol=0)
                torch.testing.assert_close(b,old_b,rtol=0,atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable on this host")
    def test_cuda_cpu_match_indices_nonempty_and_empty_gt(self):
        self._check_cpu_match_indices_on_accelerator(torch.device("cuda:0"))

    @unittest.skipUnless(hasattr(torch.backends,"mps") and torch.backends.mps.is_available(),
                         "MPS is unavailable on this host")
    def test_mps_cpu_match_indices_nonempty_and_empty_gt(self):
        self._check_cpu_match_indices_on_accelerator(torch.device("mps"))

    def test_configuration_guards_and_mismatch(self):
        for kwargs in ({"rw_calibration_class_ids":[1,1]}, {"rw_calibration_class_ids":[1,3]},
                       {"rw_calibration_weight":0}, {"rw_calibration_positive_iou":float("nan")},
                       {"rw_calibration_negative_iou":.6}, {"rw_calibration_negative_ratio":0},
                       {"use_pairwise_ce":True}, {"use_rw_isolated_specialist":True},
                       {"use_class_balanced_vfl":True}, {"use_query_objectness":True}):
            with self.assertRaises(ValueError):
                criterion(use_rw_query_calibration=True, **kwargs)
        for kwargs in ({"class_ids":[1,1]}, {"neighbor_count":0}, {"neighbor_iou":0},
                       {"common_limit":float("nan")}, {"contrast_limit":0}):
            with self.assertRaises(ValueError):
                RWQueryCalibration(32,3,4,**kwargs)
        with self.assertRaises(ValueError):
            DFINETransformer(num_classes=3, use_rw_query_calibration=True, use_aqs_refine=True)
        with self.assertRaises(ValueError):
            DFINETransformer(num_classes=3, use_rw_query_calibration=True, eval_idx=0)
        c = criterion(use_rw_query_calibration=True)
        with self.assertRaises(ValueError):
            c({"pred_logits":torch.zeros(1,2,3)}, [])
        with self.assertRaises(ValueError):
            criterion()({"pred_logits":torch.zeros(1,2,3),"rw_calibration_logits":torch.zeros(1,2,2)}, [])

    def test_clipping_keeps_original_scale(self):
        model = torch.nn.Module()
        model.backbone = torch.nn.Linear(2,2,bias=False)
        model.decoder = torch.nn.Module()
        model.decoder.use_rw_query_calibration = True
        model.decoder.rw_query_calibration = torch.nn.Linear(2,2,bias=False)
        model.backbone.weight.grad = torch.ones(2,2)*3
        model.decoder.rw_query_calibration.weight.grad = torch.ones(2,2)*100
        expected = torch.nn.Parameter(model.backbone.weight.detach().clone())
        expected.grad = model.backbone.weight.grad.clone()
        torch.nn.utils.clip_grad_norm_([expected],.1)
        clip_detector_gradients(model,.1)
        torch.testing.assert_close(model.backbone.weight.grad,expected.grad,rtol=0,atol=0)
        self.assertLessEqual(model.decoder.rw_query_calibration.weight.grad.norm().item(),.10001)

    def test_existing_classification_branches_with_current_valid_base(self):
        # Historical tests reference a removed pre-rename bs32 base filename.
        # Check their real modules against the current independent base instead.
        images = torch.rand(2,3,128,128)
        targets = [{"labels":torch.tensor([0,1,2]),
                    "boxes":torch.tensor([[.2,.3,.15,.2],[.5,.5,.2,.3],[.8,.7,.2,.2]])},
                   {"labels":torch.empty(0,dtype=torch.long),"boxes":torch.empty(0,4)}]
        for flag in ("use_rw_isolated_specialist", "use_rw_spatial_relation", "use_pairwise_ce"):
            cfg = config("dfine_s_scbs_hrw_110.yml")
            if flag == "use_pairwise_ce":
                cfg.yaml_cfg.setdefault("DFINECriterion",{})[flag] = True
            else:
                cfg.yaml_cfg.setdefault("DFINETransformer",{})[flag] = True
                if flag == "use_rw_isolated_specialist":
                    cfg.yaml_cfg.setdefault("DFINECriterion",{})[flag] = True
            model = cfg.model.train()
            self.assertFalse(model.decoder.use_rw_query_calibration)
            outputs = model(images,targets)
            losses = cfg.criterion(outputs,targets)
            self.assertNotIn("loss_rw_query_calibration",losses)
            loss = sum(losses.values())
            self.assertTrue(torch.isfinite(loss),flag)
            loss.backward()
            clip_detector_gradients(model,.1)
            with torch.no_grad():
                predicted = model.eval()(images)
            self.assertEqual(predicted["pred_logits"].shape,(2,300,3))
            self.assertTrue(torch.isfinite(predicted["pred_logits"]).all())

    def test_full_model_dn_loss_optimizer_ema_checkpoint_and_inference(self):
        torch.manual_seed(11)
        cfg = config("dfine_s_scbs_hrw_110_rw_query_calibration.yml")
        model = cfg.model
        torch.manual_seed(11)
        base_cfg = config("dfine_s_scbs_hrw_110.yml")
        base = base_cfg.model
        for name,value in base.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
        self.assertFalse(base.decoder.use_rw_query_calibration)
        self.assertEqual(cfg.yaml_cfg["epochs"],110)
        self.assertEqual(cfg.yaml_cfg["train_dataloader"]["collate_fn"]["stop_epoch"],100)
        self.assertEqual(cfg.yaml_cfg["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"],100)
        self.assertEqual(cfg.yaml_cfg["train_dataloader"]["total_batch_size"],32)
        branch = model.decoder.rw_query_calibration
        print("Added calibration parameters:",sum(p.numel() for p in branch.parameters()))
        images = torch.rand(2,3,128,128)
        model.eval(); base.eval()
        with torch.no_grad():
            a,b = model(images),base(images)
        for key in ("pred_boxes","pred_logits"):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        torch.nn.init.normal_(branch.fuse[-1].weight,std=.1)
        targets = [{"labels":torch.tensor([0,1,2]),
                    "boxes":torch.tensor([[.2,.3,.15,.2],[.5,.5,.2,.3],[.8,.7,.2,.2]])},
                   {"labels":torch.empty(0,dtype=torch.long),"boxes":torch.empty(0,4)}]
        model.train(); base.train()
        torch.manual_seed(12); outputs = model(images,targets)
        torch.manual_seed(12); old = base(images,targets)
        self.assertEqual(outputs["rw_calibration_logits"].shape,(2,300,2))
        for key in ("pred_logits","pred_boxes","pred_corners"):
            torch.testing.assert_close(outputs[key],old[key],rtol=0,atol=0)
        for group in ("aux_outputs","dn_outputs","enc_aux_outputs"):
            for aa,bb in zip(outputs[group],old[group]):
                torch.testing.assert_close(aa["pred_logits"],bb["pred_logits"],rtol=0,atol=0)
        losses = cfg.criterion(copy_output_tree(outputs),targets)
        old_losses = base_cfg.criterion(copy_output_tree(old),targets)
        self.assertEqual(set(losses)-set(old_losses),{"loss_rw_query_calibration"})
        for name,value in old_losses.items():
            torch.testing.assert_close(losses[name],value,rtol=0,atol=0)
        losses["loss_rw_query_calibration"].backward(retain_graph=True)
        for name,parameter in model.named_parameters():
            if not name.startswith("decoder.rw_query_calibration."):
                self.assertIsNone(parameter.grad,name)
        self.assertGreater(branch.fuse[-1].weight.grad.abs().sum().item(),0)
        model.zero_grad(set_to_none=True)
        sum(losses.values()).backward()
        sum(old_losses.values()).backward()
        for name,parameter in base.named_parameters():
            new_gradient = dict(model.named_parameters())[name].grad
            if parameter.grad is None:
                self.assertIsNone(new_gradient,name)
            else:
                torch.testing.assert_close(new_gradient,parameter.grad,rtol=0,atol=0)
        optimizer = cfg.optimizer
        included = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(included),len(set(included)))
        self.assertTrue(all(id(p) in included for p in branch.parameters()))
        clip_detector_gradients(model,.1)
        optimizer.step()
        cfg.ema.update(model)
        ema_keys = cfg.ema.module.state_dict()
        self.assertIn("decoder.rw_query_calibration.fuse.2.weight",ema_keys)
        model.eval()
        with torch.no_grad():
            enabled = model(images)
            model.decoder.rw_query_calibration_apply_at_eval = False
            disabled = model(images)
        torch.testing.assert_close(enabled["pred_boxes"],disabled["pred_boxes"],rtol=0,atol=0)
        torch.testing.assert_close(enabled["pred_logits"][...,0],disabled["pred_logits"][...,0],rtol=0,atol=0)
        self.assertGreater((enabled["pred_logits"][...,1:]-disabled["pred_logits"][...,1:]).abs().sum().item(),0)
        buffer = io.BytesIO()
        torch.save(model.state_dict(),buffer); buffer.seek(0)
        restored = config("dfine_s_scbs_hrw_110_rw_query_calibration.yml").model.eval()
        restored.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
        restored.decoder.rw_query_calibration_apply_at_eval = False
        with torch.no_grad():
            roundtrip = restored(images)
        torch.testing.assert_close(roundtrip["pred_logits"],disabled["pred_logits"],rtol=0,atol=0)
        diagnostics = config("dfine_s_scbs_hrw_110_rw_query_calibration_diagnostics.yml")
        self.assertTrue(diagnostics.yaml_cfg["export_query_diagnostics"])
        self.assertTrue(diagnostics.yaml_cfg["export_confusion_matrix"])
        self.assertTrue(diagnostics.yaml_cfg["DFINETransformer"]["use_rw_query_calibration"])
        # Old checkpoints remain strict-loadable when the new flag is off.
        base.load_state_dict(base.state_dict(),strict=True)
        model.eval()
        with torch.no_grad():
            deployed = copy.deepcopy(model).deploy()(images)
        self.assertTrue(torch.isfinite(deployed["pred_logits"]).all())
        self.assertTrue(torch.isfinite(deployed["pred_boxes"]).all())
        # Real S decoder in mixed precision, including clean empty-GT handling.
        model.train(); model.zero_grad(set_to_none=True)
        with torch.autocast("cpu",dtype=torch.bfloat16):
            amp_output = model(images,targets)
            amp_losses = cfg.criterion(amp_output,targets)
            amp_loss = sum(amp_losses.values())
        self.assertTrue(torch.isfinite(amp_loss))
        amp_loss.backward()
        for name,p in model.named_parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all(),name)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
