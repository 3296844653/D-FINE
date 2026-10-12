"""Synthetic SDRF correctness/regression tests; these do not establish AP gains."""
import copy
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.zoo.dfine.hybrid_encoder import HybridEncoder
from src.zoo.dfine.semantic_differential_relation_fusion import SemanticDifferentialRelationFusion as SDRF
from export_SDRF_parameters import checkpoint_report, export_checkpoint

BASE = 'dfine_s_scbs_hrw_110.yml'
TRIAL = 'dfine_s_scbs_hrw_110_SDRF.yml'
SETTINGS = dict(use_sdrf=True, sdrf_dim=64, sdrf_heads=4, sdrf_window=5,
                sdrf_halo=1, sdrf_residual_init=.1, sdrf_checkpoint=True)


def config(name=TRIAL, size=128):
    with patch('src.core.yaml_config.load_config', lambda path: load_config(path, cfg={})):
        result = YAMLConfig(str(ROOT / 'configs/dfine' / name))
    result.yaml_cfg['HGNetv2']['pretrained'] = False
    result.yaml_cfg['eval_spatial_size'] = [size, size]
    return result


def encoder(**kwargs):
    return HybridEncoder(in_channels=[16,32,64], hidden_dim=32, dim_feedforward=64,
        depth_mult=.34, expansion=.5, sdrf_dim=16, sdrf_heads=2, **kwargs)


def features(device='cpu'):
    return [torch.randn(2,16,16,16,device=device), torch.randn(2,32,8,8,device=device),
            torch.randn(2,64,4,4,device=device)]


def activate(module):
    nn.init.normal_(module.fuse.output.weight, std=.02)


def targets(device='cpu', empty=False):
    values = [dict(labels=torch.empty(0,dtype=torch.long,device=device),
                   boxes=torch.empty(0,4,device=device)) for _ in range(2)]
    if not empty:
        values[0] = dict(labels=torch.tensor([0,1,2],device=device),
            boxes=torch.tensor([[.2,.3,.2,.2],[.5,.5,.3,.3],[.8,.7,.2,.2]],device=device))
    return values


class SDRFTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(82)

    def assert_attention_invariants(self, aux):
        appearance, guided = aux['appearance_attention'], aux['guided_attention']
        d = aux['differential_attention']
        self.assertEqual(appearance.dtype,torch.float32)
        self.assertEqual(guided.dtype,appearance.dtype)
        self.assertEqual(d.dtype,appearance.dtype)
        # Two FP32 softmax/renormalization paths, subtraction and reduction
        # need a small rounding budget across CPU/PyTorch implementations.
        # 8 eps ~= 9.54e-7; do NOT relax to a training-scale tolerance.
        roundoff_atol = 8 * torch.finfo(appearance.dtype).eps
        for a in (appearance,guided):
            self.assertTrue(torch.isfinite(a).all())
            self.assertTrue((a >= 0).all())
            row_sum = a.sum(-1,dtype=torch.float64)
            torch.testing.assert_close(row_sum,torch.ones_like(row_sum),rtol=0,atol=roundoff_atol)
            self.assertEqual(a.masked_select(~aux['valid_keys'][:,None,None,:]).abs().sum().item(),0)
        # Keep the defining operation exact; only row-sum rounding is allowed.
        torch.testing.assert_close(d,guided-appearance,rtol=0,atol=0)
        row_sum = d.sum(-1,dtype=torch.float64)
        torch.testing.assert_close(row_sum,torch.zeros_like(row_sum),rtol=0,atol=roundoff_atol)

    def test_config_single_variable_optimizer_and_independence(self):
        base, trial = config(BASE), config()
        expected, actual = copy.deepcopy(base.yaml_cfg), copy.deepcopy(trial.yaml_cfg)
        for value in (expected,actual):
            value.pop('__include__',None); value.pop('output_dir',None)
        expected['HybridEncoder'].update(SETTINGS)
        self.assertEqual(expected,actual)
        model = trial.model
        self.assertEqual(model.night_probe,'none')
        self.assertTrue(all(not v for k,v in vars(model.decoder).items() if k.startswith('use_')))
        groups = trial.optimizer.param_groups
        ids = [id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))
        for name,p in model.encoder.sdrf.named_parameters():
            self.assertIn(id(p),ids,name)
            if 'norm' in name or name.endswith('bias'):
                group = next(g for g in groups if any(q is p for q in g['params']))
                self.assertEqual(group['weight_decay'],0)
        extra = sum(p.numel() for p in model.encoder.sdrf.parameters())
        self.assertEqual(sum(p.numel() for p in model.parameters())-
            sum(p.numel() for p in base.model.parameters()),extra)
        print('SDRF added parameters:',extra,'raw total:',sum(p.numel() for p in model.parameters()))
        bad = config(); bad.yaml_cfg['DFINETransformer']['use_aqs_refine'] = True
        with self.assertRaisesRegex(ValueError,'independently'): _ = bad.model

    def test_baseline_rng_state_outputs_and_disabled_schema(self):
        torch.manual_seed(44); base = config(BASE).model.eval(); rng = torch.get_rng_state().clone()
        torch.manual_seed(44); model = config().model.eval()
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        for key,value in base.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
        self.assertTrue(all(k.startswith('encoder.sdrf.') for k in set(model.state_dict())-set(base.state_dict())))
        with torch.no_grad():
            images = torch.rand(2,3,128,128); a,b = base(images),model(images)
        for key in ('pred_logits','pred_boxes'):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        disabled = config(); disabled.yaml_cfg['HybridEncoder']['use_sdrf'] = False
        self.assertEqual(set(disabled.model.state_dict()),set(base.state_dict()))

    def test_before_pan_routing_and_coarser_propagation(self):
        torch.manual_seed(50); base = encoder().eval()
        torch.manual_seed(50); model = encoder(use_sdrf=True).eval(); activate(model.sdrf)
        trace, seen = [],{}
        def capture_source(m,i,o): seen['source'] = o
        def capture_input(m,i): trace.append('sdrf'); seen['input'] = i
        hooks = [model.input_proj[0].register_forward_hook(capture_source),
            model.sdrf.register_forward_pre_hook(capture_input),
            model.pan_blocks[0].register_forward_hook(lambda m,i,o:trace.append('pan4')),
            model.pan_blocks[1].register_forward_hook(lambda m,i,o:trace.append('pan5'))]
        try:
            x = features()
            with torch.no_grad(): a,b = base(x),model(x)
        finally:
            for hook in hooks: hook.remove()
        self.assertEqual(trace,['sdrf','pan4','pan5'])
        self.assertIs(seen['input'][0],seen['source'])
        self.assertEqual(seen['input'][2].shape[-2:],(8,8))
        for left,right in zip(a,b): self.assertGreater((left-right).abs().sum().item(),0)

    def test_halo_block_coordinates_padding_and_no_wrap(self):
        m = SDRF(16,16,2,window=3,halo=1,use_checkpoint=False)
        x = torch.arange(36.).reshape(1,1,6,6)+1
        queries,keys = m._query_blocks(x),m._key_blocks(x)
        torch.testing.assert_close(queries[0,:,0],x[0,0,:3,:3].flatten())
        expected = torch.zeros(5,5); expected[1:,1:] = x[0,0,:4,:4]
        torch.testing.assert_close(keys[0,:,0],expected.flatten())
        # First block sees its right/bottom neighboring locations via halo.
        self.assertEqual(keys[0,9,0].item(),x[0,0,0,3].item())
        self.assertEqual(keys[1,5,0].item(),x[0,0,0,2].item())
        self.assertEqual(m.relative_index.min().item(),0)
        self.assertEqual(m.relative_index.max().item(),48)

    def test_attention_rows_masking_odd_small_shapes_and_semantic_effect(self):
        m = SDRF(16,16,2,use_checkpoint=False).eval(); activate(m)
        for height,width in ((9,13),(2,3),(1,1)):
            source = torch.randn(2,16,height,width); fused = torch.randn_like(source)
            coarse = torch.randn(2,16,max(1,height//2),max(1,width//2))
            out,aux = m(source,fused,coarse,return_aux=True)
            self.assertEqual(out.shape,source.shape)
            self.assert_attention_invariants(aux)
            self.assertTrue(torch.isfinite(out).all())
        source = torch.randn(2,16,9,13); coarse = torch.randn(2,16,4,7)
        a,aux = m(source,source,coarse,return_aux=True)
        b = m(source,source,coarse.flip(-1))
        self.assertGreater((a-b).abs().sum().item(),0)
        m.semantic_condition = False
        torch.testing.assert_close(m(source,source,coarse),source,rtol=0,atol=0)
        # With constant V, row-zero differential cannot convey a common value.
        m.semantic_condition = True
        with torch.no_grad():
            m.detail_qkv.weight[32:].zero_()
        _,aux = m(source,source,coarse,return_aux=True)
        self.assertEqual(aux['relation_message'].abs().sum().item(),0)

    def test_attention_roundoff_budget_still_rejects_broken_normalization(self):
        appearance = torch.tensor([.5,.5],dtype=torch.float32).reshape(1,1,1,2)
        guided = appearance.clone(); guided[...,1] += 2 * torch.finfo(torch.float32).eps
        aux = dict(appearance_attention=appearance,guided_attention=guided,
            differential_attention=guided-appearance,valid_keys=torch.ones(1,2,dtype=torch.bool))
        # Reproduce a row residual just above the previous 2e-7 limit.
        self.assertGreater(aux['differential_attention'].sum().item(),2e-7)
        self.assert_attention_invariants(aux)
        broken = dict(aux,guided_attention=guided * 1.001)
        broken['differential_attention'] = broken['guided_attention']-appearance
        with self.assertRaises(AssertionError): self.assert_attention_invariants(broken)
        broken = dict(aux,differential_attention=aux['differential_attention'] + 1e-4)
        with self.assertRaises(AssertionError): self.assert_attention_invariants(broken)

    def test_zero_initialization_then_internal_gradient_and_ablations(self):
        m = SDRF(16,16,2,use_checkpoint=False).train()
        x = torch.randn(2,16,9,13); c = torch.randn(2,16,5,7)
        optimizer = torch.optim.AdamW(m.parameters(),lr=.001)
        out = m(x,x,c); torch.testing.assert_close(out,x,rtol=0,atol=0)
        out.square().mean().backward()
        self.assertGreater(m.fuse.output.weight.grad.abs().sum().item(),0)
        self.assertEqual(m.semantic_query.weight.grad.abs().sum().item(),0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        m(x,x,c).square().mean().backward()
        for name,p in m.named_parameters():
            self.assertIsNotNone(p.grad,name); self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(p.grad.abs().sum().item(),0,name)
        count = sum(p.numel() for p in m.parameters())
        for semantic,difference in ((True,False),(False,False),(False,True)):
            ablation = SDRF(16,16,2,use_checkpoint=False,semantic_condition=semantic,differential=difference)
            self.assertEqual(sum(p.numel() for p in ablation.parameters()),count)
            activate(ablation); out = ablation(x,x,c); out.mean().backward()
            self.assertFalse([n for n,p in ablation.named_parameters() if p.grad is None])
            self.assertTrue(torch.isfinite(out).all())

    def test_checkpoint_recompute_cpu_amp_and_baseline_bn(self):
        a = encoder(use_sdrf=True,sdrf_checkpoint=False).train(); activate(a.sdrf)
        b = copy.deepcopy(a); b.sdrf.use_checkpoint = True
        x = features(); outputs,grads = [],[]
        for model in (a,b):
            with torch.autocast('cpu',dtype=torch.bfloat16):
                output = model(x); loss = sum(v.float().square().mean() for v in output)
            outputs.append(output); loss.backward()
            grads.append({n:p.grad.clone() for n,p in model.sdrf.named_parameters()})
        for left,right in zip(*outputs): torch.testing.assert_close(left,right,rtol=0,atol=0)
        for n,g in grads[0].items(): torch.testing.assert_close(g,grads[1][n],rtol=0,atol=0)
        for n,v in a.state_dict().items():
            if 'num_batches_tracked' in n: torch.testing.assert_close(v,b.state_dict()[n],rtol=0,atol=0)

    def test_full_detector_original_losses_dn_teacher_ema_and_empty_amp(self):
        cfg = config(); model = cfg.model.train(); activate(model.encoder.sdrf)
        images = torch.rand(2,3,128,128); gt = targets()
        output = model(images,gt); losses = cfg.criterion(output,gt)
        baseline = config(BASE); original = baseline.criterion(baseline.model.train()(images,gt),gt)
        self.assertEqual(set(losses),set(original))
        self.assertEqual(output['pred_logits'].shape,(2,300,3))
        self.assertIn('dn_outputs',output); self.assertIn('aux_outputs',output)
        self.assertTrue(all(torch.isfinite(v).all() for v in losses.values()))
        sum(losses.values()).backward()
        self.assertFalse([n for n,p in model.named_parameters() if p.requires_grad and p.grad is None])
        before = model.encoder.sdrf.fuse.output.weight.detach().clone()
        cfg.optimizer.step(); cfg.ema.update(model)
        self.assertGreater((before-model.encoder.sdrf.fuse.output.weight).abs().sum().item(),0)
        self.assertEqual(cfg.ema.module.encoder.sdrf.settings.dtype,torch.int64)
        model.zero_grad(set_to_none=True)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            empty = targets(empty=True); output = model(images,empty); loss = cfg.criterion(output,empty)
        sum(loss.values()).backward()
        self.assertTrue(all(torch.isfinite(v).all() for v in loss.values()))

    def test_strict_checkpoint_settings_report_and_deploy(self):
        model = config().model.eval(); activate(model.encoder.sdrf)
        saved = io.BytesIO(); torch.save(model.state_dict(),saved); saved.seek(0)
        restored = config().model.eval(); restored.load_state_dict(torch.load(saved,weights_only=True),strict=True)
        with torch.no_grad():
            images = torch.rand(2,3,128,128); expected = model(images); actual = restored(images)
            deployed = copy.deepcopy(restored).deploy(); deployed_output = deployed(images)
        for key in ('pred_logits','pred_boxes'):
            torch.testing.assert_close(expected[key],actual[key],rtol=0,atol=0)
            self.assertEqual(deployed_output[key].shape,expected[key].shape)
            self.assertTrue(torch.isfinite(deployed_output[key]).all())
        # Original backbone BN folding reorders untrained TopK ties, also in
        # baseline. Verify our branch and the original decoder on IDENTICAL
        # feature inputs; do not hide this with a loose end-to-end tolerance.
        fixed = [torch.randn(2,256,s,s) for s in (16,8,4)]
        source = torch.randn_like(fixed[0])
        with torch.no_grad():
            torch.testing.assert_close(model.encoder.sdrf(source,*fixed[:2]),
                deployed.encoder.sdrf(source,*fixed[:2]),rtol=0,atol=0)
            left,right = model.decoder(fixed),deployed.decoder(fixed)
        for key in ('pred_logits','pred_boxes'):
            torch.testing.assert_close(left[key],right[key],rtol=1e-5,atol=1e-5)
        self.assertEqual(sum(p.numel() for p in deployed.parameters()),10178201+70377)
        report = checkpoint_report(dict(model=model.state_dict(),ema=dict(module=model.state_dict()),last_epoch=109),'/tmp/best.pth')
        self.assertAlmostEqual(report['sources']['ema']['effective_residual_scale'],.1,places=6)
        with tempfile.TemporaryDirectory(prefix='sdrf-report-') as directory:
            path = Path(directory)/'best.pth'; torch.save(dict(model=model.state_dict()),path)
            export_checkpoint(path)
            with self.assertRaises(FileExistsError): export_checkpoint(path)
            self.assertTrue(path.is_file())
        a = SDRF(16,16,2)
        b = SDRF(16,16,2,semantic_condition=False)
        with self.assertRaisesRegex(RuntimeError,'settings disagree'): b.load_state_dict(a.state_dict(),strict=True)
        self.assertEqual(a.half().settings.dtype,torch.int64)

    def test_full_640_resolution(self):
        model = config(size=640).model.eval(); activate(model.encoder.sdrf)
        with torch.no_grad(): output = model(torch.rand(1,3,640,640))
        self.assertEqual(output['pred_boxes'].shape,(1,300,4))
        self.assertTrue(all(torch.isfinite(output[k]).all() for k in ('pred_logits','pred_boxes')))

    @unittest.skipUnless(torch.cuda.is_available(),'requires CUDA; run on the server')
    def test_cuda_fp16_full_detector_training(self):
        cfg = config(); model = cfg.model.cuda().train(); activate(model.encoder.sdrf)
        criterion = cfg.criterion.cuda(); gt = targets('cuda')
        with torch.autocast('cuda',dtype=torch.float16):
            output = model(torch.rand(2,3,128,128,device='cuda'),gt); losses = criterion(output,gt)
        sum(losses.values()).backward()
        self.assertTrue(all(torch.isfinite(v).all() for v in losses.values()))
        for n,p in model.encoder.sdrf.named_parameters(): self.assertTrue(torch.isfinite(p.grad).all(),n)

    def test_invalid_settings_and_scripts_preserve_outputs(self):
        for kwargs in (dict(window=4),dict(halo=0),dict(dim=17),dict(residual_init=0),dict(heads=True)):
            with self.assertRaises(ValueError): SDRF(**kwargs)
        for name in ('run_scbs_SDRF_110.sh','sync_scbs_SDRF.sh'):
            subprocess.run(['bash','-n',str(ROOT/'tools/diagnostics'/name)],check=True)
        with tempfile.TemporaryDirectory(prefix='sdrf-preserve-') as d:
            existing = Path(d)/'SDRF_existing'; existing.mkdir(); marker = existing/'log.txt'; marker.write_text('keep')
            result = subprocess.run(['bash',str(ROOT/'tools/diagnostics/run_scbs_SDRF_110.sh'),'--dry-run',str(existing)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0); self.assertEqual(marker.read_text(),'keep')
            new = Path(d)/'SDRF_new'
            result = subprocess.run(['bash',str(ROOT/'tools/diagnostics/run_scbs_SDRF_110.sh'),'--dry-run',str(new)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr); self.assertFalse(new.exists())
        manifest = (ROOT/'tools/diagnostics/scbs_SDRF_sync_files.txt').read_text().splitlines()
        self.assertEqual(len(manifest),len(set(manifest)))
        for relative in manifest:
            self.assertFalse(Path(relative).is_absolute())
            self.assertNotIn('..',Path(relative).parts)
            self.assertTrue((ROOT/relative).is_file())
            self.assertNotEqual(relative,'configs/dfine/'+BASE)
            self.assertNotIn(relative.split('/')[0],('dataset','output'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
