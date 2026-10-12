"""Synthetic SADF correctness/regression tests, not effectiveness evidence."""
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.zoo.dfine.hybrid_encoder import HybridEncoder
from src.zoo.dfine.scale_aligned_detail_fusion import BoundedScaleAlignment, ScaleAlignedDetailFusion
from export_sadf_parameters import checkpoint_report, export_checkpoint

BASE = 'dfine_s_scbs_hrw_110.yml'
TRIAL = 'dfine_s_scbs_hrw_110_SADF.yml'


def config(name=TRIAL, size=128):
    with patch('src.core.yaml_config.load_config', lambda path: load_config(path, cfg={})):
        result = YAMLConfig(str(ROOT / 'configs/dfine' / name))
    result.yaml_cfg['HGNetv2']['pretrained'] = False
    result.yaml_cfg['eval_spatial_size'] = [size, size]
    return result


def small_encoder(**kwargs):
    return HybridEncoder(in_channels=[16,32,64], hidden_dim=32, dim_feedforward=64,
                         depth_mult=.34, expansion=.5, sadf_dim=16, **kwargs)


def features(device='cpu'):
    return [torch.randn(2,16,16,16,device=device), torch.randn(2,32,8,8,device=device),
            torch.randn(2,64,4,4,device=device)]


def targets(device='cpu', empty=False):
    result = [dict(labels=torch.empty(0,dtype=torch.long,device=device), boxes=torch.empty(0,4,device=device))
              for _ in range(2)]
    if not empty:
        result[0] = dict(labels=torch.tensor([0,1,2],device=device),
            boxes=torch.tensor([[.2,.3,.2,.2],[.5,.5,.3,.3],[.8,.7,.2,.2]],device=device))
    return result


def activate(module):
    # Test-only simulation of a trained, nonzero residual.
    nn.init.normal_(module.restore.weight, std=.02)


class SADFTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_config_single_variable_and_optimizer_coverage(self):
        base, trial = config(BASE), config()
        expected, actual = copy.deepcopy(base.yaml_cfg), copy.deepcopy(trial.yaml_cfg)
        for item in (expected, actual):
            item.pop('__include__', None); item.pop('output_dir', None)
        expected['HybridEncoder'].update(use_sadf=True, sadf_dim=64, sadf_offset_limit=2.,
            sadf_residual_init=.1, sadf_checkpoint=True)
        self.assertEqual(expected, actual)
        model = trial.model
        self.assertEqual(model.night_probe, 'none')
        for name,value in vars(model.decoder).items():
            if name.startswith('use_'):
                self.assertFalse(value, name)
        all_ids = [id(p) for group in trial.optimizer.param_groups for p in group['params']]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        for name,param in model.encoder.sadf.named_parameters():
            self.assertIn(id(param), all_ids, name)
            if 'norm' in name or name.endswith('bias'):
                group = next(g for g in trial.optimizer.param_groups if any(p is param for p in g['params']))
                self.assertEqual(group['weight_decay'], 0)
        extra = sum(p.numel() for p in model.encoder.sadf.parameters())
        self.assertEqual(extra, 162377)
        self.assertEqual(sum(p.numel() for p in model.parameters()) -
                         sum(p.numel() for p in base.model.parameters()), extra)
        print('SADF added parameters:', extra, 'raw total:', sum(p.numel() for p in model.parameters()))

    def test_baseline_state_rng_predictions_and_disabled_state(self):
        torch.manual_seed(61); base = config(BASE).model.eval(); rng = torch.get_rng_state().clone()
        torch.manual_seed(61); model = config().model.eval()
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for name,value in base.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        self.assertTrue(all(k.startswith('encoder.sadf.') for k in set(model.state_dict())-set(base.state_dict())))
        images = torch.rand(2,3,128,128)
        with torch.no_grad():
            expected, actual = base(images), model(images)
        for key in ('pred_logits','pred_boxes'):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        disabled = config(); disabled.yaml_cfg['HybridEncoder']['use_sadf'] = False
        torch.manual_seed(61); off = disabled.model.eval()
        self.assertEqual(set(off.state_dict()), set(base.state_dict()))
        self.assertFalse(hasattr(off.encoder, 'sadf'))

    def test_after_pan_source_routing_and_only_p3_directly_changes(self):
        torch.manual_seed(62); base = small_encoder().eval()
        torch.manual_seed(62); trial = small_encoder(use_sadf=True).eval(); activate(trial.sadf)
        trace, captured = [], {}
        def save_source(module, inputs, output): captured['source'] = output
        def inspect_sadf(module, inputs):
            trace.append('sadf'); captured['sadf_inputs'] = inputs
        hooks = [trial.input_proj[0].register_forward_hook(save_source),
                 trial.pan_blocks[0].register_forward_hook(lambda m,i,o: trace.append('pan4')),
                 trial.pan_blocks[1].register_forward_hook(lambda m,i,o: trace.append('pan5')),
                 trial.sadf.register_forward_pre_hook(inspect_sadf)]
        try:
            x = features()
            with torch.no_grad(): expected, actual = base(x), trial(x)
        finally:
            for hook in hooks: hook.remove()
        self.assertEqual(trace, ['pan4','pan5','sadf'])
        self.assertIs(captured['sadf_inputs'][0], captured['source'])
        self.assertGreater((actual[0]-expected[0]).abs().sum().item(), 0)
        for index in (1,2):
            torch.testing.assert_close(actual[index], expected[index], rtol=0, atol=0)

    def test_alignment_coordinates_bounds_and_direction(self):
        coarse = torch.randn(2,16,3,4)
        zero = torch.zeros(2,2,5,7)
        expected = F.interpolate(coarse, size=(5,7), mode='bilinear', align_corners=False)
        actual = BoundedScaleAlignment.sample(coarse, zero)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        shift = zero.clone(); shift[:,0] = 1
        shifted = BoundedScaleAlignment.sample(coarse, shift)
        torch.testing.assert_close(shifted[:,:,:,:-1], expected[:,:,:,1:], rtol=1e-5, atol=1e-6)
        alignment = BoundedScaleAlignment(16, 2.)
        with torch.no_grad(): alignment.offset.predict.bias.fill_(10)
        _, displacement = alignment(expected, coarse, expected)
        self.assertLessEqual(displacement.abs().max().item(), 2.)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            sampled = BoundedScaleAlignment.sample(coarse.bfloat16(), zero)
        self.assertEqual(sampled.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(sampled).all())

    def test_contrast_constant_boundaries_scale_weights_and_ablation_interfaces(self):
        x = [torch.ones(2,32,8,8), torch.randn(2,32,8,8), torch.randn(2,32,4,4), torch.randn(2,32,2,2)]
        module = ScaleAlignedDetailFusion(32,16,use_checkpoint=False).eval()
        _, aux = module(*x, return_aux=True)
        torch.testing.assert_close(aux['detail'], torch.zeros_like(aux['detail']), rtol=0, atol=1e-6)
        torch.testing.assert_close(aux['scale_weights'].sum(1), torch.ones(2,8,8))
        torch.testing.assert_close(aux['detail_weights'].sum(1), torch.ones(2,8,8))
        self.assertTrue(all(torch.isfinite(v).all() for v in aux.values()))
        count = sum(p.numel() for p in module.parameters())
        for option in ('enable_alignment','enable_detail','enable_scale_selection'):
            with self.subTest(option=option):
                changed = ScaleAlignedDetailFusion(32,16,use_checkpoint=False,**{option:False})
                activate(changed); output, maps = changed(*x, return_aux=True)
                self.assertEqual(sum(p.numel() for p in changed.parameters()), count)
                if option == 'enable_alignment':
                    self.assertEqual(maps['offset_p4'].abs().sum().item(), 0)
                if option == 'enable_scale_selection':
                    torch.testing.assert_close(maps['scale_weights'], torch.full((2,3,8,8),1/3))
                output.square().mean().backward()
                self.assertFalse([n for n,p in changed.named_parameters() if p.grad is None])

    def test_initial_projection_and_subsequent_internal_gradients(self):
        module = ScaleAlignedDetailFusion(32,16,use_checkpoint=False).train()
        x = [torch.randn(2,32,8,8), torch.randn(2,32,8,8), torch.randn(2,32,4,4), torch.randn(2,32,2,2)]
        optimizer = torch.optim.AdamW(module.parameters(), lr=.001)
        module(*x).square().mean().backward()
        self.assertGreater(module.restore.weight.grad.abs().sum().item(), 0)
        self.assertEqual(module.alignments[0].offset.predict.weight.grad.abs().sum().item(), 0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        module(*x).square().mean().backward()
        for name,param in module.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        for param in (module.detail_filters[0].weight, module.scale_selector.score.weight,
                      module.alignments[0].offset.predict.weight, module.joint_gates.predict.weight,
                      module.raw_alpha):
            self.assertGreater(param.grad.abs().sum().item(), 0)

    def test_checkpoint_recompute_cpu_amp_gradients_and_bn_stats(self):
        ordinary = small_encoder(use_sadf=True,sadf_checkpoint=False).train(); activate(ordinary.sadf)
        recomputed = copy.deepcopy(ordinary); recomputed.sadf.use_checkpoint = True
        outputs, grads = [], []
        x = features()
        for model in (ordinary, recomputed):
            with torch.autocast('cpu', dtype=torch.bfloat16):
                values = model(x); loss = sum(v.float().square().mean() for v in values)
            outputs.append([v.detach() for v in values]); loss.backward()
            grads.append({k:p.grad.detach().clone() for k,p in model.sadf.named_parameters()})
        for a,b in zip(outputs[0],outputs[1]): torch.testing.assert_close(a,b,rtol=0,atol=0)
        for key,value in grads[0].items(): torch.testing.assert_close(value,grads[1][key],rtol=0,atol=0)
        for key,value in ordinary.state_dict().items():
            if 'num_batches_tracked' in key:
                torch.testing.assert_close(value, recomputed.state_dict()[key], rtol=0, atol=0)

    def test_original_full_detector_loss_dn_teacher_optimizer_and_ema(self):
        cfg = config(); model = cfg.model.train(); activate(model.encoder.sadf)
        gt = targets(); images = torch.rand(2,3,128,128)
        output = model(images,gt)
        self.assertEqual(output['pred_logits'].shape, (2,300,3))
        self.assertEqual(output['pred_boxes'].shape, (2,300,4))
        self.assertIn('dn_outputs', output); self.assertIn('aux_outputs', output)
        loss = cfg.criterion(output,gt)
        baseline_cfg = config(BASE); baseline_cfg.model.train()
        original_loss = baseline_cfg.criterion(baseline_cfg.model(images,gt),gt)
        self.assertEqual(set(loss),set(original_loss))
        self.assertTrue(all(torch.isfinite(v).all() for v in loss.values()))
        sum(loss.values()).backward()
        self.assertFalse([n for n,p in model.named_parameters() if p.requires_grad and p.grad is None])
        for name,p in model.encoder.sadf.named_parameters(): self.assertTrue(torch.isfinite(p.grad).all(),name)
        before = model.encoder.sadf.restore.weight.detach().clone()
        cfg.optimizer.step(); cfg.ema.update(model)
        self.assertGreater((before-model.encoder.sadf.restore.weight).abs().sum().item(),0)
        self.assertTrue(torch.isfinite(cfg.ema.module.encoder.sadf.raw_alpha).all())
        model.zero_grad(set_to_none=True)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            empty = targets(empty=True); output = model(images,empty); loss = cfg.criterion(output,empty)
        sum(loss.values()).backward()
        self.assertTrue(all(torch.isfinite(v).all() for v in loss.values()))

    def test_strict_checkpoint_parameter_report_and_deploy(self):
        model = config().model.eval(); activate(model.encoder.sadf)
        buffer = io.BytesIO(); torch.save(model.state_dict(),buffer); buffer.seek(0)
        restored = config().model.eval()
        restored.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
        images = torch.rand(2,3,128,128)
        with torch.no_grad():
            expected, actual = model(images), restored(images)
            deployed = copy.deepcopy(model).deploy(); third = deployed(images)
        for key in ('pred_logits','pred_boxes'):
            torch.testing.assert_close(expected[key],actual[key],rtol=0,atol=0)
            self.assertEqual(expected[key].shape,third[key].shape)
            self.assertTrue(torch.isfinite(third[key]).all())
        # Untrained detector TopK ties can be reordered by original backbone
        # BN folding (also in baseline). Test the added branch/decoder with the
        # SAME input features rather than masking that with a loose tolerance.
        fixed = [torch.randn(2,256,s,s) for s in (16,8,4)]
        with torch.no_grad():
            source = torch.randn_like(fixed[0])
            torch.testing.assert_close(model.encoder.sadf(source,*fixed),
                deployed.encoder.sadf(source,*fixed),rtol=0,atol=0)
            before, after = model.decoder(fixed), deployed.decoder(fixed)
            for key in ('pred_logits','pred_boxes'):
                torch.testing.assert_close(before[key],after[key],rtol=1e-5,atol=1e-5)
        self.assertEqual(sum(p.numel() for p in deployed.parameters()),10178201+162377)
        state = {'model':model.state_dict(),'ema':{'module':restored.state_dict()},'last_epoch':109}
        with tempfile.TemporaryDirectory(prefix='dfine-sadf-test-') as directory:
            path = Path(directory)/'best_stg2.pth'; torch.save(state,path)
            report = export_checkpoint(path)
            self.assertEqual(set(report['sources']),{'model','ema'})
            self.assertAlmostEqual(report['sources']['ema']['channel_scale_mean'],.1,places=6)
            self.assertEqual(report['last_epoch'],109)
            with self.assertRaises(FileExistsError): export_checkpoint(path)
        with self.assertRaises(ValueError): checkpoint_report({'model':{}},'baseline.pth')

    def test_640_and_variable_resolution_outputs(self):
        model = config(size=640).model.eval(); activate(model.encoder.sadf)
        with torch.no_grad():
            output = model(torch.rand(1,3,640,640))
        self.assertEqual(output['pred_logits'].shape,(1,300,3))
        self.assertTrue(torch.isfinite(output['pred_logits']).all())
        self.assertTrue(torch.isfinite(output['pred_boxes']).all())
        encoder = small_encoder(use_sadf=True).eval()
        for height,width in ((16,24),(20,28)):
            x = [torch.randn(1,16,height,width),torch.randn(1,32,height//2,width//2),
                 torch.randn(1,64,height//4,width//4)]
            with torch.no_grad(): output = encoder(x)
            self.assertEqual(output[0].shape,(1,32,height,width))

    def test_invalid_settings_and_single_module_guards(self):
        for settings in ({'dim':7},{'offset_limit':5},{'offset_limit':float('nan')},
                         {'residual_init':0},{'residual_init':.5},{'use_checkpoint':'yes'}):
            with self.subTest(settings=settings), self.assertRaises(ValueError): ScaleAlignedDetailFusion(**settings)
        for settings in ({'use_mffe':True},{'use_encoder_highres_residual':True},
                         {'use_sfif':True},{'use_sadf':'yes'}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                small_encoder(**dict({'use_sadf':True},**settings))
        for name,value in (('night_probe','encoder_cross_gate'),):
            cfg = config(); cfg.yaml_cfg['DFINE'][name] = value
            with self.assertRaisesRegex(ValueError,'independently'): _ = cfg.model
        cfg = config(); cfg.yaml_cfg['DFINETransformer']['use_aqs_refine'] = True
        with self.assertRaisesRegex(ValueError,'independently'): _ = cfg.model

    def test_launcher_syntax_dry_run_and_existing_output_protection(self):
        scripts = ['tools/diagnostics/run_scbs_SADF_110.sh','tools/diagnostics/sync_scbs_SADF.sh']
        for script in scripts: subprocess.run(['bash','-n',str(ROOT/script)],check=True)
        # Only run this subprocess with the already-active testing interpreter's
        # environment. No server, GPU, filesystem output, or training is needed.
        for filename in (ROOT/'tools/diagnostics/scbs_SADF_sync_files.txt').read_text().splitlines():
            self.assertTrue((ROOT/filename).is_file(),filename)
        with tempfile.TemporaryDirectory(prefix='dfine-SADF-existing-') as folder:
            result = subprocess.run(['bash',str(ROOT/scripts[0]),'--dry-run',folder],
                                    capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('保留旧结果',result.stderr)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not available locally')
    def test_cuda_fp16_detector_original_loss_and_checkpoint(self):
        cfg = config(); model = cfg.model.cuda().train(); activate(model.encoder.sadf)
        criterion = cfg.criterion.cuda()
        gt = targets('cuda'); images = torch.rand(2,3,128,128,device='cuda')
        with torch.autocast('cuda',dtype=torch.float16):
            output = model(images,gt); loss = criterion(output,gt)
        scaler = torch.amp.GradScaler('cuda', init_scale=128)
        scaler.scale(sum(loss.values())).backward(); scaler.unscale_(cfg.optimizer)
        self.assertTrue(all(torch.isfinite(v).all() for v in loss.values()))
        for name,p in model.encoder.sadf.named_parameters():
            self.assertIsNotNone(p.grad,name); self.assertTrue(torch.isfinite(p.grad).all(),name)
        scaler.step(cfg.optimizer); scaler.update()


if __name__ == '__main__': unittest.main(verbosity=2)
