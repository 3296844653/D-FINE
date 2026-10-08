"""Second-layer AQS checks; synthetic CPU data, no downloads or full training."""

import copy
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.solver.aqs_parameters import runtime_aqs_report
from src.zoo.dfine.dfine_decoder import DFINETransformer, LQE
from tools.diagnostics.test_aqs_refine import copy_output, fresh_config


def targets(device="cpu", empty=False):
    blank = {"labels": torch.empty(0, dtype=torch.long, device=device),
             "boxes": torch.empty(0, 4, device=device)}
    labeled = {"labels": torch.tensor([0, 1, 2], device=device),
               "boxes": torch.tensor([[.2, .3, .15, .2], [.5, .5, .2, .3],
                                       [.8, .7, .2, .2]], device=device)}
    return [copy.deepcopy(blank if empty else labeled), copy.deepcopy(blank)]


def activate(model):
    # Synthetic-test-only confidence/weights: NEVER change real priors/config.
    with torch.no_grad():
        model.decoder.dec_score_head[1].bias.fill_(1)
        branch = model.decoder.decoder.query_cls_refiner
        nn.init.normal_(branch.fuse[-1].weight, std=.03)
        nn.init.normal_(branch.fuse[-1].bias, std=.01)
        nn.init.normal_(model.decoder.dec_bbox_head[2].layers[-1].weight, std=.01)


class AQSLayer2Tests(unittest.TestCase):
    def test_layer2_config_only_changes_position_and_output_directory(self):
        old = fresh_config("dfine_s_scbs_hrw_110_aqs.yml")
        new = fresh_config("dfine_s_scbs_hrw_110_aqs_layer2.yml")
        expected, actual = copy.deepcopy(old.yaml_cfg), copy.deepcopy(new.yaml_cfg)
        for cfg in (expected, actual):
            cfg.pop("__include__", None)
            cfg.pop("output_dir", None)
        expected["DFINETransformer"]["aqs_apply_layer"] = 2
        self.assertEqual(expected, actual)
        model = new.model
        self.assertEqual(model.decoder.eval_idx, 2)
        self.assertEqual(model.decoder.decoder.aqs_layer_idx, 1)
        self.assertEqual(len(model.decoder.decoder.layers), 3)
        self.assertEqual(new.yaml_cfg["epochs"], 110)
        self.assertEqual(new.yaml_cfg["train_dataloader"]["collate_fn"]["stop_epoch"], 100)
        self.assertEqual(new.yaml_cfg["DFINECriterion"], old.yaml_cfg["DFINECriterion"])
        self.assertFalse(model.encoder.use_mffe)
        diag = fresh_config("dfine_s_scbs_hrw_110_aqs_layer2_diagnostics.yml")
        self.assertEqual(diag.model.decoder.decoder.aqs_layer_idx, 1)
        self.assertTrue(diag.export_query_diagnostics)
        self.assertTrue(diag.yaml_cfg["export_confusion_matrix"])

    def test_zero_initialization_matches_baseline_with_identical_rng_and_heads(self):
        base_cfg = fresh_config("dfine_s_scbs_hrw_110.yml")
        cfg = fresh_config("dfine_s_scbs_hrw_110_aqs_layer2.yml")
        torch.manual_seed(41)
        base = base_cfg.model
        rng = torch.get_rng_state().clone()
        torch.manual_seed(41)
        trial = cfg.model
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for key, value in base.state_dict().items():
            torch.testing.assert_close(trial.state_dict()[key], value, rtol=0, atol=0)
        images = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            base.eval(); trial.eval()
            a, b = base(images), trial(images)
        for key in ("pred_logits", "pred_boxes"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        base.train(); trial.train()
        torch.manual_seed(42); a = base(images, targets())
        torch.manual_seed(42); b = trial(images, targets())
        for key in ("pred_logits", "pred_boxes", "pred_corners", "ref_points"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        for group in ("aux_outputs", "dn_outputs", "enc_aux_outputs"):
            for aa, bb in zip(a[group], b[group]):
                for key in ("pred_logits", "pred_boxes"):
                    torch.testing.assert_close(aa[key], bb[key], rtol=0, atol=0)

    def test_layer2_executes_once_and_refined_query_enters_layer3_in_inference(self):
        model = fresh_config("dfine_s_scbs_hrw_110_aqs_layer2.yml").model.eval()
        activate(model)
        baseline = copy.deepcopy(model)
        baseline.decoder.use_aqs_refine = baseline.decoder.decoder.use_aqs_refine = False
        decoder = model.decoder.decoder
        seen = {"refined": [], "layer3_input": [], "layer2_logits": []}
        def record_refiner(module, args, output):
            seen["refined"].append(output.detach().clone())
            seen["layer2_logits"].append(args[1].detach().clone())
        def record_layer3(module, args):
            seen["layer3_input"].append(args[0].detach().clone())
        hooks = [decoder.query_cls_refiner.register_forward_hook(record_refiner),
                 decoder.layers[2].register_forward_pre_hook(record_layer3)]
        images = torch.rand(2, 3, 128, 128)
        try:
            with torch.no_grad():
                result, original = model(images), baseline(images)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(len(seen["refined"]), 1)
        self.assertEqual(len(seen["layer3_input"]), 1)
        self.assertEqual(seen["layer2_logits"][0].shape, (2, 300, 3))
        torch.testing.assert_close(seen["layer3_input"][0], seen["refined"][0], rtol=0, atol=0)
        self.assertGreater((result["pred_logits"] - original["pred_logits"]).abs().sum().item(), 0)
        self.assertGreater((result["pred_boxes"] - original["pred_boxes"]).abs().sum().item(), 0)
        self.assertEqual(result["pred_boxes"].shape, (2, 300, 4))
        # A final-output-only loss must reach AQS: not just an aux-head change.
        model.zero_grad(set_to_none=True)
        model(images)["pred_logits"].square().sum().backward()
        self.assertGreater(decoder.query_cls_refiner.fuse[-1].weight.grad.abs().sum().item(), 0)

    def test_training_second_layer_aux_boxes_teacher_gradients_and_optimizer(self):
        cfg = fresh_config("dfine_s_scbs_hrw_110_aqs_layer2.yml")
        model = cfg.model.train()
        activate(model)
        baseline = copy.deepcopy(model)
        baseline.decoder.use_aqs_refine = baseline.decoder.decoder.use_aqs_refine = False
        images, labels = torch.rand(2, 3, 128, 128), targets()
        torch.manual_seed(43); trial = model(images, labels)
        torch.manual_seed(43); base = baseline(images, labels)
        self.assertEqual(len(trial["aux_outputs"]), 2)
        self.assertEqual(len(trial["dn_outputs"]), 3)
        # The layer-1 stream and pre-AQS layer-2 bbox/FDR are unchanged.
        for index in (0, 1):
            torch.testing.assert_close(trial["aux_outputs"][index]["pred_boxes"],
                                       base["aux_outputs"][index]["pred_boxes"], rtol=0, atol=0)
        torch.testing.assert_close(trial["aux_outputs"][0]["pred_logits"],
                                   base["aux_outputs"][0]["pred_logits"], rtol=0, atol=0)
        self.assertGreater((trial["aux_outputs"][1]["pred_logits"] -
                            base["aux_outputs"][1]["pred_logits"]).abs().sum().item(), 0)
        for aux in trial["aux_outputs"]:
            # Layer 3 has no extra AQS classification-only correction.
            torch.testing.assert_close(aux["teacher_logits"], trial["pred_logits"], rtol=0, atol=0)
        losses = cfg.criterion(copy_output(trial), labels)
        base_losses = cfg.criterion(copy_output(base), labels)
        self.assertEqual(set(losses), set(base_losses))
        self.assertTrue(all(torch.isfinite(value).all() for value in losses.values()))
        sum(losses.values()).backward()
        branch = model.decoder.decoder.query_cls_refiner
        for name, parameter in branch.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        # Production uses find_unused_parameters=False. Moving AQS must not
        # orphan an original trainable head in this mixed-GT training graph.
        unused = [name for name, parameter in model.named_parameters()
                  if parameter.requires_grad and parameter.grad is None]
        self.assertEqual(unused, [])
        ids = [id(p) for group in cfg.optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(id(p) in ids for p in branch.parameters()))
        before = branch.fuse[-1].weight.detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
        cfg.optimizer.step()
        self.assertGreater((branch.fuse[-1].weight - before).abs().sum().item(), 0)

    def test_strict_checkpoint_and_deploy_preserve_layer2_gate_heads(self):
        cfg = fresh_config("dfine_s_scbs_hrw_110_aqs_layer2.yml")
        model = cfg.model.eval()
        activate(model)
        restored = fresh_config("dfine_s_scbs_hrw_110_aqs_layer2.yml").model.eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        last_layer = fresh_config("dfine_s_scbs_hrw_110_aqs.yml").model.eval()
        self.assertEqual(set(model.state_dict()), set(last_layer.state_dict()))
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         sum(p.numel() for p in last_layer.parameters()))
        images = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            expected, actual = model(images), restored(images)
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
            deployed = copy.deepcopy(model).deploy()
            last_deployed = last_layer.deploy()
            full_prediction = deployed(images)
            self.assertTrue(all(torch.isfinite(v).all() for v in full_prediction.values()))
            self.assertIsInstance(deployed.decoder.dec_score_head[1], nn.Linear)
            self.assertIsInstance(deployed.decoder.decoder.lqe_layers[1], LQE)
            self.assertEqual(len(deployed.decoder.decoder.layers), 3)
            # Random, untrained HGNet features can produce tied encoder TopK
            # scores: conv/BN folding's normal rounding changes their ordering.
            # Test decoder pruning with SAME well-conditioned random features,
            # not two unrelated tied TopK selections from a random backbone.
            features = [torch.randn(2, 256, size, size) for size in (16, 8, 4)]
            expected = model.decoder(features)
            prediction = deployed.decoder(features)
            for key in expected:
                torch.testing.assert_close(prediction[key], expected[key], rtol=1e-5, atol=1e-5)
            gate_parameters = sum(p.numel() for p in deployed.decoder.dec_score_head[1].parameters())
            gate_parameters += sum(p.numel() for p in deployed.decoder.decoder.lqe_layers[1].parameters())
            self.assertEqual(sum(p.numel() for p in deployed.parameters()) -
                             sum(p.numel() for p in last_deployed.parameters()), gate_parameters)
        self.assertEqual(gate_parameters, 2180)
        report = runtime_aqs_report(model, epoch=109)
        self.assertEqual(report["sources"]["model"]["decoder.decoder.query_cls_refiner"]["decoder_layer"], 2)

    def test_cpu_amp_and_empty_gt(self):
        cfg = fresh_config("dfine_s_scbs_hrw_110_aqs_layer2.yml")
        model = cfg.model.train()
        with torch.no_grad():
            model.decoder.dec_score_head[1].bias.fill_(1)
        images = torch.rand(2, 3, 128, 128)
        for empty in (False, True):
            labels = targets(empty=empty)
            model.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                output = model(images, labels)
            losses = cfg.criterion(copy_output(output), labels)
            self.assertTrue(all(torch.isfinite(value).all() for value in losses.values()))
            sum(losses.values()).backward()
            for parameter in model.decoder.decoder.query_cls_refiner.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_invalid_layers_rejected_and_final_default_is_unchanged(self):
        for layer in (0, -2, 4, True, 2.0):
            with self.assertRaisesRegex(ValueError, "aqs_apply_layer"):
                DFINETransformer(num_layers=3, use_aqs_refine=True, aqs_apply_layer=layer)
        for kwargs in ({"eval_idx": 1}, {"layer_scale": 2}):
            with self.assertRaisesRegex(ValueError, "final decoder layer"):
                DFINETransformer(num_layers=3, use_aqs_refine=True, aqs_apply_layer=2, **kwargs)
        last = DFINETransformer(num_layers=3, use_aqs_refine=True)
        self.assertEqual(last.decoder.aqs_layer_idx, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable: real FP16 check must run on server")
    def test_cuda_fp16_mixed_and_empty_gt(self):
        cfg = fresh_config("dfine_s_scbs_hrw_110_aqs_layer2.yml")
        model, criterion = cfg.model.cuda().train(), cfg.criterion.cuda()
        with torch.no_grad():
            model.decoder.dec_score_head[1].bias.fill_(1)
        images = torch.rand(2, 3, 128, 128, device="cuda")
        for empty in (False, True):
            labels = targets(device="cuda", empty=empty)
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(images, labels)
            losses = criterion(copy_output(output), labels)
            self.assertTrue(all(torch.isfinite(value).all() for value in losses.values()))
            sum(losses.values()).backward()
            for parameter in model.decoder.decoder.query_cls_refiner.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
