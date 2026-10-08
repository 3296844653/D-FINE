"""CPU checks for the MFFE P3-only ablation; no real dataset or downloads."""
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


def fresh_config(name):
    with patch("src.core.yaml_config.load_config", lambda path: load_config(path, cfg={})):
        cfg = YAMLConfig(str(ROOT / "configs/dfine" / name))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["eval_spatial_size"] = [128, 128]
    return cfg


def small_encoder(**kwargs):
    return HybridEncoder(in_channels=[16, 32, 64], hidden_dim=32,
                         dim_feedforward=64, depth_mult=0.34, expansion=0.5,
                         mffe_mid_channels=8, **kwargs)


def features():
    return [torch.randn(2, 16, 16, 16), torch.randn(2, 32, 8, 8), torch.randn(2, 64, 4, 4)]


class MFFEP3OnlyTests(unittest.TestCase):
    def test_default_keeps_old_two_branches_and_p3_initialization(self):
        torch.manual_seed(40); original = small_encoder(use_mffe=True).eval()
        original_rng = torch.get_rng_state().clone()
        torch.manual_seed(40); explicit = small_encoder(use_mffe=True, mffe_p3_only=False).eval()
        torch.testing.assert_close(torch.get_rng_state(), original_rng, rtol=0, atol=0)
        self.assertEqual(len(original.mffe), 2)
        self.assertEqual(list(original.state_dict()), list(explicit.state_dict()))
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
        torch.manual_seed(40); p3 = small_encoder(use_mffe=True, mffe_p3_only=True).eval()
        torch.testing.assert_close(torch.get_rng_state(), original_rng, rtol=0, atol=0)
        self.assertEqual(len(p3.mffe), 1)
        self.assertFalse(any(k.startswith('mffe.1.') for k in p3.state_dict()))
        for name, value in p3.state_dict().items():
            torch.testing.assert_close(value, original.state_dict()[name], rtol=0, atol=0)
        x = features()
        with torch.no_grad():
            for a, b in zip(original(x), explicit(x)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_p3_insertion_before_pan_and_no_direct_p4_branch(self):
        torch.manual_seed(41); baseline = small_encoder().eval()
        torch.manual_seed(41); p3 = small_encoder(use_mffe=True, mffe_p3_only=True).eval()
        trace, captures, hooks = [], {}, []
        def save(name, field, selector):
            def hook(module, inputs, output):
                captures[name] = selector(inputs, output).detach().clone()
                if field: trace.append(name)
            return hook
        hooks.append(p3.input_proj[0].register_forward_hook(save('source_p3', False, lambda i,o:o)))
        hooks.append(p3.fpn_blocks[0].register_forward_hook(save('fpn_p4', True, lambda i,o:o)))
        # The next FPN iteration applies a lateral 1x1 transform to P4
        # before that feature becomes the coarse input of the first PAN block.
        hooks.append(p3.lateral_convs[1].register_forward_hook(save('fpn_p4_for_pan', False, lambda i,o:o)))
        hooks.append(p3.fpn_blocks[1].register_forward_hook(save('fpn_p3', True, lambda i,o:o)))
        hooks.append(p3.mffe[0].register_forward_hook(save('mffe_p3', True, lambda i,o:i[0])))
        hooks.append(p3.pan_blocks[0].register_forward_hook(save('pan_p4', True, lambda i,o:i[0])))
        hooks.append(p3.pan_blocks[1].register_forward_hook(save('pan_p5', True, lambda i,o:i[0])))
        x = features()
        with torch.no_grad(): original, enhanced = baseline(x), p3(x)
        for hook in hooks: hook.remove()
        self.assertEqual(trace, ['fpn_p4', 'fpn_p3', 'mffe_p3', 'pan_p4', 'pan_p5'])
        torch.testing.assert_close(captures['mffe_p3'], captures['source_p3'], rtol=0, atol=0)
        # PAN receives the unchanged FPN P4 as its coarse input. Its fine
        # input still contains enhanced/downsampled P3, so P4/P5 can change.
        torch.testing.assert_close(captures['pan_p4'][:, 32:], captures['fpn_p4_for_pan'], rtol=0, atol=0)
        for a, b in zip(original, enhanced):
            self.assertEqual(a.shape, b.shape)
            self.assertTrue(torch.isfinite(b).all())
            self.assertGreater((a-b).abs().sum().item(), 0)

    def test_disabled_and_zero_scale_recover_baseline(self):
        torch.manual_seed(42); baseline = small_encoder().eval()
        torch.manual_seed(42); off = small_encoder(mffe_p3_only=True).eval()
        self.assertFalse(hasattr(off, 'mffe'))
        self.assertEqual(list(baseline.state_dict()), list(off.state_dict()))
        torch.manual_seed(42); trial = small_encoder(use_mffe=True, mffe_p3_only=True).eval()
        with torch.no_grad(): trial.mffe[0].raw_alpha.zero_()
        x = features()
        with torch.no_grad():
            expected, disabled, zero = baseline(x), off(x), trial(x)
        for a,b,c in zip(expected, disabled, zero):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            torch.testing.assert_close(a,c,rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            small_encoder(use_mffe=True, mffe_p3_only='true')

    def test_checkpoint_and_cpu_amp_encoder_gradients(self):
        torch.manual_seed(43)
        ordinary = small_encoder(use_mffe=True, mffe_p3_only=True, mffe_checkpoint=False).train()
        recomputed = copy.deepcopy(ordinary); recomputed.mffe[0].use_checkpoint = True
        x = features(); saved, gradients = [], []
        for model in (ordinary, recomputed):
            with torch.autocast('cpu', dtype=torch.bfloat16):
                output = model([v.detach().clone() for v in x])
                loss = sum(v.float().square().mean() for v in output)
            saved.append([v.detach() for v in output]); loss.backward()
            gradients.append({k:p.grad.detach().clone() for k,p in model.mffe.named_parameters()})
            for name,p in model.mffe.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
                self.assertGreater(p.grad.abs().sum().item(),0,name)
        for a,b in zip(saved[0],saved[1]): torch.testing.assert_close(a,b,rtol=0,atol=0)
        for key,value in gradients[0].items():
            torch.testing.assert_close(value,gradients[1][key],rtol=0,atol=0)

    def test_yaml_only_removes_p4_direct_enhancement(self):
        original = fresh_config('dfine_s_scbs_hrw_mffe.yml').yaml_cfg
        p3 = fresh_config('dfine_s_scbs_hrw_mffe_p3.yml').yaml_cfg
        original.pop('__include__',None); p3.pop('__include__',None)
        original['HybridEncoder']['mffe_p3_only'] = True
        self.assertEqual(original,p3)
        self.assertEqual(p3['train_dataloader']['total_batch_size'],32)
        self.assertEqual(p3['epochs'],110)
        self.assertEqual(p3['train_dataloader']['dataset']['transforms']['policy']['epoch'],100)
        self.assertEqual(p3['train_dataloader']['collate_fn']['stop_epoch'],100)
        diag = fresh_config('dfine_s_scbs_hrw_mffe_p3_diagnostics.yml').yaml_cfg
        self.assertTrue(diag['HybridEncoder']['mffe_p3_only'])
        self.assertTrue(diag['export_confusion_matrix'])
        self.assertTrue(diag['export_query_diagnostics'])
        self.assertEqual(diag['query_diagnostic_conf_thresh'],.5)
        self.assertEqual(diag['query_diagnostic_iou_thresh'],.5)

    def test_full_detector_losses_optimizer_and_checkpoint(self):
        base_cfg = fresh_config('dfine_s_scbs_hrw.yml')
        trial_cfg = fresh_config('dfine_s_scbs_hrw_mffe_p3.yml')
        torch.manual_seed(44); base = base_cfg.model.eval()
        torch.manual_seed(44); model = trial_cfg.model.eval()
        self.assertEqual(len(model.encoder.mffe),1)
        self.assertTrue(model.encoder.mffe[0].use_checkpoint)
        for name,value in base.state_dict().items():
            torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
        extra = sum(p.numel() for p in model.encoder.mffe.parameters())
        self.assertEqual(extra,56004)
        self.assertEqual(sum(p.numel() for p in model.parameters())-
                         sum(p.numel() for p in base.parameters()),extra)
        self.assertFalse(any(k.startswith('encoder.mffe.1.') for k in model.state_dict()))
        # Recover the exact baseline's prediction path when the one residual is zero.
        images = torch.rand(2,3,128,128)
        with torch.no_grad():
            scale = model.encoder.mffe[0].raw_alpha.detach().clone()
            model.encoder.mffe[0].raw_alpha.zero_()
            original, zero = base(images), model(images)
            for key in ('pred_logits','pred_boxes'):
                torch.testing.assert_close(original[key],zero[key],rtol=0,atol=0)
            model.encoder.mffe[0].raw_alpha.copy_(scale)
        criterion = trial_cfg.criterion
        for flag in ('use_pairwise_ce','use_class_margin','use_class_balanced_vfl'):
            self.assertFalse(getattr(criterion,flag))
        for name,value in vars(model.decoder).items():
            if name.startswith('use_'): self.assertFalse(value,name)
        targets = [dict(labels=torch.tensor([0,1,2]), boxes=torch.tensor([
            [.2,.3,.15,.2],[.5,.5,.2,.3],[.8,.7,.2,.2]])),
            dict(labels=torch.empty(0,dtype=torch.long),boxes=torch.empty(0,4))]
        model.train(); outputs = model(images, targets)
        self.assertIn('dn_outputs',outputs); self.assertIn('aux_outputs',outputs)
        self.assertEqual(outputs['pred_logits'].shape,(2,300,3))
        self.assertEqual(outputs['pred_boxes'].shape,(2,300,4))
        losses = criterion(outputs,targets)
        self.assertTrue(all(torch.isfinite(v).all() for v in losses.values()))
        self.assertFalse(any('pairwise' in k or 'margin' in k for k in losses))
        sum(losses.values()).backward()
        for name,param in model.encoder.mffe.named_parameters():
            self.assertIsNotNone(param.grad,name)
            self.assertTrue(torch.isfinite(param.grad).all(),name)
            self.assertGreater(param.grad.abs().sum().item(),0,name)
        optimizer = trial_cfg.optimizer
        ids = [id(p) for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertTrue(all(id(p) in ids for p in model.encoder.mffe.parameters()))
        old_weight = model.encoder.mffe[0].detail_proj[0].weight.detach().clone()
        optimizer.step()
        self.assertGreater((old_weight-model.encoder.mffe[0].detail_proj[0].weight).abs().sum().item(),0)
        model.eval()
        restored = fresh_config('dfine_s_scbs_hrw_mffe_p3.yml').model.eval()
        restored.load_state_dict(model.state_dict(),strict=True)
        with torch.no_grad():
            expected, actual = model(images), restored(images)
            for key in ('pred_logits','pred_boxes'):
                torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
        # Original two-branch checkpoints must not silently load as P3-only.
        original = fresh_config('dfine_s_scbs_hrw_mffe.yml').model.eval()
        with self.assertRaisesRegex(RuntimeError,'encoder.mffe.1.'):
            model.load_state_dict(original.state_dict(),strict=True)
        deployed = copy.deepcopy(restored).deploy()
        self.assertEqual(sum(p.numel() for p in deployed.parameters()),10234205)
        print(f'P3-only MFFE added parameters: {extra}; deploy/profiler parameters: 10234205')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
