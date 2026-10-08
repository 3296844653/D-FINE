"""Independent L2/L3 AQS checks; synthetic data, no downloads/full training."""

import copy
import io
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.solver.aqs_parameters import checkpoint_aqs_report, runtime_aqs_report
from src.zoo.dfine.dfine_decoder import DFINETransformer, LQE
from tools.diagnostics.export_aqs_parameters import export_checkpoint
from tools.diagnostics.test_aqs_layer2 import targets
from tools.diagnostics.test_aqs_refine import copy_output, fresh_config


CONFIG = "dfine_s_scbs_hrw_110_aqs_l2_l3.yml"


def activate(model):
    # TEST ONLY: confidence priors and projections in real configs are unchanged.
    with torch.no_grad():
        for index in (1, 2):
            model.decoder.dec_score_head[index].bias.fill_(1)
        for module in model.decoder.decoder.aqs_refiners.values():
            nn.init.normal_(module.fuse[-1].weight, std=.03)
            nn.init.normal_(module.fuse[-1].bias, std=.01)
        nn.init.normal_(model.decoder.dec_bbox_head[2].layers[-1].weight, std=.01)


class AQSL2L3Tests(unittest.TestCase):
    def test_config_only_adds_independent_layer_positions(self):
        old = fresh_config("dfine_s_scbs_hrw_110_aqs.yml")
        new = fresh_config(CONFIG)
        expected, actual = copy.deepcopy(old.yaml_cfg), copy.deepcopy(new.yaml_cfg)
        for cfg in (expected, actual):
            cfg.pop("__include__", None)
            cfg.pop("output_dir", None)
        expected["DFINETransformer"]["aqs_apply_layers"] = [2, 3]
        self.assertEqual(expected, actual)
        self.assertEqual(new.yaml_cfg["epochs"], 110)
        self.assertEqual(new.yaml_cfg["train_dataloader"]["total_batch_size"], 32)
        self.assertEqual(new.yaml_cfg["train_dataloader"]["collate_fn"]["stop_epoch"], 100)
        self.assertEqual(new.yaml_cfg["DFINECriterion"], old.yaml_cfg["DFINECriterion"])
        model = new.model
        self.assertEqual(model.decoder.eval_idx, 2)
        self.assertEqual(len(model.decoder.decoder.layers), 3)
        self.assertEqual(model.decoder.decoder.aqs_layer_indices, (1, 2))
        self.assertFalse(model.encoder.use_mffe)
        self.assertFalse(model.encoder.use_encoder_highres_residual)
        diag = fresh_config("dfine_s_scbs_hrw_110_aqs_l2_l3_diagnostics.yml")
        self.assertEqual(diag.model.decoder.decoder.aqs_layer_indices, (1, 2))
        self.assertTrue(diag.export_query_diagnostics)
        self.assertTrue(diag.yaml_cfg["export_confusion_matrix"])

    def test_two_refiners_share_no_parameters_and_start_at_defaults(self):
        decoder = DFINETransformer(num_layers=3, use_aqs_refine=True,
                                   aqs_apply_layers=[2, 3]).decoder
        l2, l3 = decoder.aqs_refiners["layer2"], decoder.aqs_refiners["layer3"]
        self.assertIsNone(decoder.query_cls_refiner)
        self.assertIsNone(decoder.aqs_layer_idx)
        self.assertFalse({id(p) for p in l2.parameters()} & {id(p) for p in l3.parameters()})
        self.assertFalse({p.data_ptr() for p in l2.parameters()} & {p.data_ptr() for p in l3.parameters()})
        for module in (l2, l3):
            self.assertEqual(module.threshold_logit.sigmoid().item(), .5)
            self.assertAlmostEqual(module.residual_scale.item(), .05)
            self.assertEqual(module.temperature, .1)
            self.assertEqual(module.fuse[-1].weight.abs().sum().item(), 0)
            self.assertEqual(sum(p.numel() for p in module.parameters()), 197634)
        with torch.no_grad():
            l2.threshold_logit.fill_(2)
            l2.residual_scale.fill_(.4)
        self.assertEqual(l3.threshold_logit.sigmoid().item(), .5)
        self.assertAlmostEqual(l3.residual_scale.item(), .05)

    def test_zero_start_preserves_baseline_weights_rng_and_train_eval_predictions(self):
        base_cfg, cfg = fresh_config("dfine_s_scbs_hrw_110.yml"), fresh_config(CONFIG)
        torch.manual_seed(71); base = base_cfg.model
        rng = torch.get_rng_state().clone()
        torch.manual_seed(71); trial = cfg.model
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for key, value in base.state_dict().items():
            torch.testing.assert_close(trial.state_dict()[key], value, rtol=0, atol=0)
        images = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            base.eval(); trial.eval()
            a, b = base(images), trial(images)
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        base.train(); trial.train()
        torch.manual_seed(72); a = base(images, targets())
        torch.manual_seed(72); b = trial(images, targets())
        for key in ("pred_logits", "pred_boxes", "pred_corners", "ref_points"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        for group in ("aux_outputs", "dn_outputs", "enc_aux_outputs"):
            for aa, bb in zip(a[group], b[group]):
                for key in ("pred_logits", "pred_boxes"):
                    torch.testing.assert_close(aa[key], bb[key], rtol=0, atol=0)

    def test_l2_reaches_d3_and_l3_changes_only_final_classification(self):
        model = fresh_config(CONFIG).model.eval()
        activate(model)
        only2 = copy.deepcopy(model)
        only2.decoder.decoder.aqs_layer_indices = (1,)  # identical weights; disable L3 only
        decoder = model.decoder.decoder
        seen = {"l2": [], "l3": [], "d3": []}
        def record(name):
            def hook(module, args, output):
                seen[name].append(output.detach().clone())
            return hook
        def before_d3(module, args):
            seen["d3"].append(args[0].detach().clone())
        hooks = [decoder.aqs_refiners["layer2"].register_forward_hook(record("l2")),
                 decoder.aqs_refiners["layer3"].register_forward_hook(record("l3")),
                 decoder.layers[2].register_forward_pre_hook(before_d3)]
        images = torch.rand(2, 3, 128, 128)
        try:
            with torch.no_grad():
                result, unrefined_final = model(images), only2(images)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual([len(seen[key]) for key in ("l2", "l3", "d3")], [1, 1, 1])
        torch.testing.assert_close(seen["d3"][0], seen["l2"][0], rtol=0, atol=0)
        torch.testing.assert_close(result["pred_boxes"], unrefined_final["pred_boxes"], rtol=0, atol=0)
        self.assertGreater((result["pred_logits"] - unrefined_final["pred_logits"]).abs().sum().item(), 0)
        model.zero_grad(set_to_none=True)
        model(images)["pred_logits"].square().sum().backward()
        for module in decoder.aqs_refiners.values():
            self.assertGreater(module.fuse[-1].weight.grad.abs().sum().item(), 0)

    def test_dn_isolation_for_each_independent_layer(self):
        decoder = DFINETransformer(num_classes=3, hidden_dim=32, feat_channels=[32, 32, 32],
                                   num_layers=3, use_aqs_refine=True,
                                   aqs_apply_layers=[2, 3]).decoder.train()
        query, logits = torch.randn(2, 8, 32), torch.ones(2, 8, 3)
        meta = {"dn_num_split": [3, 5]}
        for index in (1, 2):
            branch = decoder.aqs_refiners[f"layer{index + 1}"]
            nn.init.normal_(branch.fuse[-1].weight, std=.03)
            output = decoder._apply_aqs_refiner(query, logits, meta, layer_idx=index)
            expected = torch.cat([branch(query[:, :3], logits[:, :3]),
                                  branch(query[:, 3:], logits[:, 3:])], dim=1)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            changed = query.clone(); changed[:, :3] += 100
            torch.testing.assert_close(
                decoder._apply_aqs_refiner(changed, logits, meta, layer_idx=index)[:, 3:],
                output[:, 3:], rtol=0, atol=0,
            )
        with self.assertRaisesRegex(ValueError, "explicit layer_idx"):
            decoder._apply_aqs_refiner(query, logits, meta)

    def test_original_losses_teacher_and_both_gradients_optimizer(self):
        cfg = fresh_config(CONFIG)
        model = cfg.model.train()
        activate(model)
        only2 = copy.deepcopy(model)
        only2.decoder.decoder.aqs_layer_indices = (1,)
        images, labels = torch.rand(2, 3, 128, 128), targets()
        torch.manual_seed(73); both = model(images, labels)
        torch.manual_seed(73); unrefined_final = only2(images, labels)
        for aux in both["aux_outputs"]:
            # Distillation uses final PRE-L3-AQS scores, not calibrated logits.
            torch.testing.assert_close(aux["teacher_logits"], unrefined_final["pred_logits"], rtol=0, atol=0)
        self.assertGreater((both["pred_logits"] - unrefined_final["pred_logits"]).abs().sum().item(), 0)
        torch.testing.assert_close(both["pred_boxes"], unrefined_final["pred_boxes"], rtol=0, atol=0)
        losses = cfg.criterion(copy_output(both), labels)
        original_losses = cfg.criterion(copy_output(unrefined_final), labels)
        self.assertEqual(set(losses), set(original_losses))
        self.assertTrue(all(torch.isfinite(value).all() for value in losses.values()))
        sum(losses.values()).backward()
        before = {}
        for key, module in model.decoder.decoder.aqs_refiners.items():
            for name, parameter in module.named_parameters():
                self.assertIsNotNone(parameter.grad, (key, name))
                self.assertTrue(torch.isfinite(parameter.grad).all(), (key, name))
                self.assertGreater(parameter.grad.abs().sum().item(), 0, (key, name))
            before[key] = module.fuse[-1].weight.detach().clone()
        self.assertEqual([name for name, p in model.named_parameters()
                          if p.requires_grad and p.grad is None], [])
        optimizer = cfg.optimizer
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(id(p) in ids for module in model.decoder.decoder.aqs_refiners.values()
                            for p in module.parameters()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
        optimizer.step()
        for key, module in model.decoder.decoder.aqs_refiners.items():
            self.assertGreater((module.fuse[-1].weight - before[key]).abs().sum().item(), 0)

    def test_cpu_amp_mixed_and_empty_gt(self):
        cfg = fresh_config(CONFIG)
        model = cfg.model.train()
        with torch.no_grad():
            for index in (1, 2):
                model.decoder.dec_score_head[index].bias.fill_(1)
        images = torch.rand(2, 3, 128, 128)
        for empty in (False, True):
            labels = targets(empty=empty)
            model.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                output = model(images, labels)
            losses = cfg.criterion(copy_output(output), labels)
            self.assertTrue(all(torch.isfinite(value).all() for value in losses.values()))
            sum(losses.values()).backward()
            for module in model.decoder.decoder.aqs_refiners.values():
                for parameter in module.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_strict_loading_and_deploy_keep_both_gate_heads(self):
        model = fresh_config(CONFIG).model.eval()
        activate(model)
        restored = fresh_config(CONFIG).model.eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        single = fresh_config("dfine_s_scbs_hrw_110_aqs_layer2.yml").model.eval()
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(single.state_dict(), strict=True)
        # PyTorch may copy matching base keys before reporting missing keys.
        restored.load_state_dict(model.state_dict(), strict=True)
        extra = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in single.parameters())
        self.assertEqual(extra, 197634)
        with torch.no_grad():
            images = torch.rand(2, 3, 128, 128)
            expected, actual = model(images), restored(images)
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
            deployed = copy.deepcopy(model).deploy()
            for index in (1, 2):
                self.assertIsInstance(deployed.decoder.dec_score_head[index], nn.Linear)
                self.assertIsInstance(deployed.decoder.decoder.lqe_layers[index], LQE)
            features = [torch.randn(2, 256, size, size) for size in (16, 8, 4)]
            expected, actual = model.decoder(features), deployed.decoder(features)
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key], rtol=1e-5, atol=1e-5)
            self.assertEqual(sum(p.numel() for p in deployed.parameters()), 10575649)

    def test_runtime_and_checkpoint_export_all_four_independent_scalars(self):
        model = fresh_config(CONFIG).model.eval()
        ema = copy.deepcopy(model)
        with torch.no_grad():
            for source in (model, ema):
                source.decoder.decoder.aqs_refiners["layer2"].threshold_logit.fill_(math.log(.48 / .52))
                source.decoder.decoder.aqs_refiners["layer2"].residual_scale.fill_(.23)
                source.decoder.decoder.aqs_refiners["layer3"].threshold_logit.fill_(math.log(.53 / .47))
                source.decoder.decoder.aqs_refiners["layer3"].residual_scale.fill_(.14)
        runtime = runtime_aqs_report(model, SimpleNamespace(module=ema), epoch=109)
        prefixes = {"decoder.decoder.aqs_refiners.layer2", "decoder.decoder.aqs_refiners.layer3"}
        for source in ("model", "ema"):
            self.assertEqual(set(runtime["sources"][source]), prefixes)
            self.assertEqual({v["decoder_layer"] for v in runtime["sources"][source].values()}, {2, 3})
        state = {"model": model.state_dict(), "ema": {"module": ema.state_dict()}, "last_epoch": 109}
        report = checkpoint_aqs_report(state, "best_stg2.pth", temperature=.1)
        for source in ("model", "ema"):
            for prefix in prefixes:
                for key in ("threshold", "residual_scale", "effective_residual_scale", "decoder_layer"):
                    self.assertEqual(report["sources"][source][prefix][key], runtime["sources"][source][prefix][key])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "best_stg2.pth"
            torch.save(state, path)
            before = path.read_bytes()
            with redirect_stdout(io.StringIO()) as output:
                exported = export_checkpoint(path, temperature=.1)
            self.assertIn("[L2]", output.getvalue())
            self.assertIn("[L3]", output.getvalue())
            self.assertEqual(exported["sources"], report["sources"])
            self.assertEqual(path.read_bytes(), before)

    def test_invalid_multi_layer_config_rejected(self):
        for layers in ([], [2, 2], [0, 3], [2, 4], [True, 3], [2., 3], "2,3"):
            with self.assertRaisesRegex(ValueError, "aqs_apply_layers"):
                DFINETransformer(num_layers=3, use_aqs_refine=True, aqs_apply_layers=layers)
        with self.assertRaisesRegex(ValueError, "only one"):
            DFINETransformer(num_layers=3, use_aqs_refine=True, aqs_apply_layer=2, aqs_apply_layers=[2, 3])
        for kwargs in ({"eval_idx": 1}, {"layer_scale": 2}):
            with self.assertRaisesRegex(ValueError, "final decoder layer"):
                DFINETransformer(num_layers=3, use_aqs_refine=True, aqs_apply_layers=[2, 3], **kwargs)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable: run real FP16 check on server")
    def test_cuda_fp16_mixed_and_empty_gt(self):
        cfg = fresh_config(CONFIG)
        model, criterion = cfg.model.cuda().train(), cfg.criterion.cuda()
        with torch.no_grad():
            for index in (1, 2):
                model.decoder.dec_score_head[index].bias.fill_(1)
        images = torch.rand(2, 3, 128, 128, device="cuda")
        for empty in (False, True):
            labels = targets(device="cuda", empty=empty)
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(images, labels)
            losses = criterion(copy_output(output), labels)
            self.assertTrue(all(torch.isfinite(value).all() for value in losses.values()))
            sum(losses.values()).backward()
            for module in model.decoder.decoder.aqs_refiners.values():
                for parameter in module.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
