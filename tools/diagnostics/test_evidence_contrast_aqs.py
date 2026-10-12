"""Synthetic implementation checks, not AP/effectiveness/novelty evidence."""
import copy
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.zoo.dfine.evidence_contrast_aqs import EvidenceContrastAQS
from src.zoo.dfine.dfine_decoder import AdaptiveQuerySelectionRefiner
from src.solver.aqs_parameters import runtime_aqs_report
from tools.diagnostics.test_aqs_refine import fresh_config, copy_output
from tools.diagnostics.test_night_probes import targets
from tools.diagnostics.export_evidence_contrast_aqs import checkpoint_report, export_checkpoint

BASE = 'dfine_s_scbs_hrw_110.yml'
TRIAL = 'dfine_s_scbs_hrw_110_EvidenceContrastAQS.yml'


def inputs(device='cpu'):
    q = torch.randn(2,9,32,device=device)
    s = torch.tensor([[2.,-3.,-2.],[-2.,2.,-3.],[-3.,-2.,2.],[-.1,-.2,-3.],
                      [1.,.5,-2.],[.5,-3.,1.],[-3.,-3.,-3.],[2.,-2.,-1.],[-2.,2.,-3.]],device=device)
    return q,s[None].repeat(2,1,1),torch.rand(2,9,4,device=device),torch.randn(2,9,132,device=device)


def activate(module):
    nn.init.normal_(module.fuse[-1].weight,std=.02)


def activate_model(model):
    activate(model.decoder.decoder.query_cls_refiner)
    with torch.no_grad(): model.decoder.dec_score_head[1].bias.fill_(1)


class EvidenceContrastAQSTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_config_only_changes_aqs_mechanism_and_optimizer_coverage(self):
        baseline,trial = fresh_config(BASE),fresh_config(TRIAL)
        expected,actual = copy.deepcopy(baseline.yaml_cfg),copy.deepcopy(trial.yaml_cfg)
        for cfg in (expected,actual): cfg.pop('__include__',None); cfg.pop('output_dir',None)
        expected['DFINETransformer'].update(use_aqs_refine=True,aqs_refiner_type='EvidenceContrastAQS',
            aqs_apply_layer=2,aqs_threshold=.5,aqs_temperature=.1,aqs_residual_init=.1,
            aqs_evidence_dim=64,aqs_evidence_slots=4,aqs_evidence_heads=4)
        self.assertEqual(expected,actual)
        model = trial.model; module = model.decoder.decoder.query_cls_refiner
        self.assertIsInstance(module,EvidenceContrastAQS)
        self.assertEqual(model.decoder.decoder.aqs_layer_indices,(1,))
        self.assertEqual(model.night_probe,'none')
        self.assertEqual(sum(p.numel() for p in module.parameters()),245747)
        self.assertEqual(sum(p.numel() for p in model.parameters())-sum(p.numel() for p in baseline.model.parameters()),245747)
        ids = [id(p) for group in trial.optimizer.param_groups for p in group['params']]
        self.assertEqual(len(ids),len(set(ids)))
        for name,p in module.named_parameters():
            self.assertIn(id(p),ids,name)
            if 'norm' in name or name.endswith('bias'):
                group = next(g for g in trial.optimizer.param_groups if any(item is p for item in g['params']))
                self.assertEqual(group['weight_decay'],0)

    def test_original_state_rng_identity_and_legacy_aqs_compatibility(self):
        torch.manual_seed(81); base = fresh_config(BASE).model.eval(); rng = torch.get_rng_state().clone()
        torch.manual_seed(81); model = fresh_config(TRIAL).model.eval()
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        for key,value in base.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
        added = set(model.state_dict())-set(base.state_dict())
        self.assertTrue(all(k.startswith('decoder.decoder.query_cls_refiner.') for k in added))
        image = torch.rand(2,3,128,128)
        with torch.no_grad(): a,b = base(image),model(image)
        for key in ('pred_logits','pred_boxes'): torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
        legacy = fresh_config('dfine_s_scbs_hrw_110_aqs_layer2_r010.yml').model
        self.assertIsInstance(legacy.decoder.decoder.query_cls_refiner,AdaptiveQuerySelectionRefiner)
        self.assertFalse(any('settings' in k for k in legacy.decoder.decoder.query_cls_refiner.state_dict()))

    def test_zero_empty_donors_and_extreme_values_cpu_amp(self):
        module = EvidenceContrastAQS(32,3,dim=16); args = inputs()
        torch.testing.assert_close(module(*args),args[0],rtol=0,atol=0)
        activate(module)
        for value in (-1000.,-20.):
            q,s,b,c = args; s = torch.full_like(s,value)
            with torch.autocast('cpu',dtype=torch.bfloat16): output,aux = module(q,s,b,c,return_aux=True)
            torch.testing.assert_close(output,q,rtol=0,atol=0)
            self.assertEqual(aux['receiver_gate'].abs().sum().item(),0)
            self.assertTrue(all(torch.isfinite(v).all() for v in aux.values()))
            output.square().mean().backward(); module.zero_grad(set_to_none=True)
        q,s,b,c = args
        self.assertEqual(module(q[:,:0],s[:,:0],b[:,:0],c[:,:0]).shape,(2,0,32))
        self.assertTrue(torch.isfinite(module(q,torch.full_like(s,1000.),b,c)).all())

    def test_bank_semantics_support_contrast_permutation_and_batch_isolation(self):
        module = EvidenceContrastAQS(32,3,dim=16).eval(); activate(module)
        q,s,b,c = inputs(); output,aux = module(q,s,b,c,return_aux=True)
        self.assertEqual(aux['bank_mass'].shape,(2,12))
        torch.testing.assert_close(aux['slot_assignment'].sum(-1),torch.ones(2,9,3))
        torch.testing.assert_close(aux['support_attention'].sum(-1),torch.ones(2,4,9))
        self.assertGreater((aux['support']-aux['other']).abs().sum().item(),0)
        # No argmax lock: each selected query contributes softly to every class bank.
        self.assertTrue((aux['bank_mass']>0).all())
        self.assertTrue((aux['receiver_gate']>=0).all() and (aux['receiver_gate']<=1).all())
        self.assertGreater(aux['receiver_gate'][0,3].item(),0)  # low-score ambiguous receiver
        order = torch.tensor([8,3,6,1,5,0,4,7,2])
        torch.testing.assert_close(module(q[:,order],s[:,order],b[:,order],c[:,order]),output[:,order])
        modified = q.clone(); modified[0] *= 9
        torch.testing.assert_close(module(modified,s,b,c)[1],output[1],rtol=0,atol=0)

    def test_gradients_first_step_then_activated_detached_prediction_conditions(self):
        module = EvidenceContrastAQS(32,3,dim=16)
        q,s,b,c = [v.requires_grad_() for v in inputs()]
        module(q,s,b,c).square().mean().backward()
        self.assertGreater(module.fuse[-1].weight.grad.abs().sum().item(),0)
        self.assertEqual(module.threshold_logit.grad.item(),0)
        self.assertEqual(module.residual_scale.grad.item(),0)
        module.zero_grad(set_to_none=True); activate(module)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            output = module(q,s,b,c); loss = (output*torch.randn_like(output)).mean()
        loss.backward()
        for name,p in module.named_parameters():
            self.assertIsNotNone(p.grad,name); self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(p.grad.abs().sum().item(),0,name)
        self.assertIsNone(s.grad); self.assertIsNone(b.grad); self.assertIsNone(c.grad)
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_ablation_graph_coverage_and_uniform_concentration(self):
        args = inputs(); count = sum(p.numel() for p in EvidenceContrastAQS(32,3,dim=16).parameters())
        for flag in ('geometry','quality','contrast','adaptive_receiver'):
            module = EvidenceContrastAQS(32,3,dim=16,**{flag:False}); activate(module)
            self.assertEqual(sum(p.numel() for p in module.parameters()),count)
            output,aux = module(*args,return_aux=True); output.square().mean().backward()
            self.assertFalse([n for n,p in module.named_parameters() if p.grad is None])
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in module.parameters()))
            if flag == 'contrast': self.assertEqual(aux['difference'].abs().sum().item(),0)
            if flag == 'adaptive_receiver':
                torch.testing.assert_close(aux['receiver_gate'],aux['source_gate'])
        module = EvidenceContrastAQS(32,3,dim=16)
        q,s,b,c = args; _,aux = module(q,s,b,torch.zeros_like(c),return_aux=True)
        torch.testing.assert_close(aux['concentration'],torch.zeros(2,9),atol=1e-6,rtol=0)

    def test_dn_banks_never_leak_into_detection_and_empty_dn(self):
        cfg = fresh_config(TRIAL); decoder = cfg.model.decoder.decoder.train()
        activate(decoder.query_cls_refiner)
        q = torch.randn(2,9,256); _,s,b,c = inputs(); meta = {'dn_num_split':[3,6]}
        original = decoder._apply_aqs_refiner(q,s,meta,layer_idx=1,boxes=b,corners=c)
        altered = [v.clone() for v in (q,s,b,c)]
        for value in altered: value[:,:3] += 30
        changed = decoder._apply_aqs_refiner(*altered[:2],meta,layer_idx=1,boxes=altered[2],corners=altered[3])
        torch.testing.assert_close(original[:,3:],changed[:,3:],rtol=0,atol=0)
        direct = decoder.query_cls_refiner(q[:,3:],s[:,3:],b[:,3:],c[:,3:])
        torch.testing.assert_close(original[:,3:],direct,rtol=0,atol=0)
        empty = decoder._apply_aqs_refiner(q,s,{'dn_num_split':[0,9]},layer_idx=1,boxes=b,corners=c)
        torch.testing.assert_close(empty,decoder.query_cls_refiner(q,s,b,c),rtol=0,atol=0)
        with self.assertRaises(ValueError): decoder._apply_aqs_refiner(q,s,meta,layer_idx=1)

    def test_l2_boxes_unchanged_next_layer_receives_refined_features_original_losses(self):
        cfg = fresh_config(TRIAL); model = cfg.model.train(); activate_model(model)
        plain = copy.deepcopy(model); plain.decoder.use_aqs_refine = plain.decoder.decoder.use_aqs_refine = False
        image,gt = torch.rand(2,3,128,128),targets()
        torch.manual_seed(82); actual = model(image,gt)
        torch.manual_seed(82); expected = plain(image,gt)
        for layer in (0,1):
            torch.testing.assert_close(actual['aux_outputs'][layer]['pred_boxes'],expected['aux_outputs'][layer]['pred_boxes'],rtol=0,atol=0)
        self.assertGreater((actual['pred_logits']-expected['pred_logits']).abs().sum().item(),0)
        for aux in actual['aux_outputs']:
            torch.testing.assert_close(aux['teacher_logits'],actual['pred_logits'],rtol=0,atol=0)
        loss = cfg.criterion(copy_output(actual),gt)
        self.assertEqual(set(loss),set(cfg.criterion(copy_output(expected),gt)))
        sum(loss.values()).backward()
        self.assertFalse([n for n,p in model.named_parameters() if p.requires_grad and p.grad is None])
        for name,p in model.decoder.decoder.query_cls_refiner.named_parameters():
            self.assertTrue(torch.isfinite(p.grad).all(),name)
        before = model.decoder.decoder.query_cls_refiner.fuse[-1].weight.detach().clone()
        cfg.optimizer.step(); cfg.ema.update(model)
        self.assertGreater((before-model.decoder.decoder.query_cls_refiner.fuse[-1].weight).abs().sum().item(),0)
        trace = {}
        model.eval(); dec = model.decoder.decoder
        hooks = [dec.query_cls_refiner.register_forward_hook(lambda m,i,o:trace.update(refined=o.detach().clone())),
                 dec.layers[2].register_forward_pre_hook(lambda m,i:trace.update(next=i[0].detach().clone()))]
        try:
            with torch.no_grad(): model(image)
        finally:
            for hook in hooks: hook.remove()
        torch.testing.assert_close(trace['refined'],trace['next'],rtol=0,atol=0)

    def test_cpu_amp_training_with_dn_and_empty_gt(self):
        cfg = fresh_config(TRIAL); model = cfg.model.train(); activate_model(model)
        for empty in (False,True):
            model.zero_grad(set_to_none=True); gt = targets(empty=empty)
            with torch.autocast('cpu',dtype=torch.bfloat16):
                output = model(torch.rand(2,3,128,128),gt); loss = cfg.criterion(copy_output(output),gt)
            self.assertTrue(all(torch.isfinite(v).all() for v in loss.values()))
            sum(loss.values()).backward()
            for name,p in model.decoder.decoder.query_cls_refiner.named_parameters():
                self.assertIsNotNone(p.grad,name); self.assertTrue(torch.isfinite(p.grad).all(),name)

    def test_strict_checkpoint_export_runtime_report_and_deploy_fixed_features(self):
        model = fresh_config(TRIAL).model.eval(); activate_model(model)
        buffer = io.BytesIO(); torch.save(model.state_dict(),buffer); buffer.seek(0)
        restored = fresh_config(TRIAL).model.eval()
        restored.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
        fixed = [torch.randn(2,256,size,size) for size in (16,8,4)]
        deployed = copy.deepcopy(restored).deploy()
        with torch.no_grad():
            a,b,d = model.decoder(fixed),restored.decoder(fixed),deployed.decoder(fixed)
        for key in ('pred_logits','pred_boxes'):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            torch.testing.assert_close(a[key],d[key],rtol=1e-5,atol=1e-5)
        report = runtime_aqs_report(model,epoch=109)
        self.assertEqual(report['sources']['model']['decoder.decoder.query_cls_refiner']['decoder_layer'],2)
        checkpoint = dict(model=model.state_dict(),ema={'module':restored.state_dict()},last_epoch=109)
        with tempfile.TemporaryDirectory(prefix='dfine-evidence-aqs-') as folder:
            path = Path(folder)/'best_stg2.pth'; torch.save(checkpoint,path)
            report = export_checkpoint(path)
            values = report['sources']['ema']['decoder.decoder.query_cls_refiner']
            self.assertEqual(values['decoder_layer'],2); self.assertEqual(values['slots_per_class'],4)
            self.assertAlmostEqual(values['temperature'],.1,places=6)
            with self.assertRaises(FileExistsError): export_checkpoint(path)
        with self.assertRaises(ValueError): checkpoint_report({'model':{}},'baseline.pth')
        print('EvidenceContrastAQS raw/deploy parameters:',sum(p.numel() for p in model.parameters()),sum(p.numel() for p in deployed.parameters()))

    def test_full_640_inference_and_invalid_settings_independent_guards(self):
        cfg = fresh_config(TRIAL); cfg.yaml_cfg['eval_spatial_size'] = [640,640]
        model = cfg.model.eval(); activate_model(model)
        with torch.no_grad(): output = model(torch.rand(1,3,640,640))
        self.assertEqual(output['pred_logits'].shape,(1,300,3))
        self.assertTrue(torch.isfinite(output['pred_logits']).all() and torch.isfinite(output['pred_boxes']).all())
        for extra in ({'num_classes':1},{'slots':0},{'dim':7},{'threshold':1},{'temperature':0},{'geometry':'yes'}):
            kwargs = dict(hidden_dim=32,num_classes=3); kwargs.update(extra)
            with self.assertRaises(ValueError): EvidenceContrastAQS(**kwargs)
        cfg = fresh_config(TRIAL); cfg.yaml_cfg['DFINE']['night_probe'] = 'decoder_l2_local'
        cfg.yaml_cfg['DFINE']['allow_aqs_l2_local_combo'] = True
        with self.assertRaises(ValueError): _ = cfg.model
        cfg = fresh_config(TRIAL); cfg.yaml_cfg['HybridEncoder']['use_encoder_highres_residual'] = True
        with self.assertRaisesRegex(ValueError,'independent'): _ = cfg.model

    def test_checkpoint_rejects_silent_fixed_setting_or_layer_changes(self):
        model = fresh_config(TRIAL).model
        changed = fresh_config(TRIAL)
        changed.yaml_cfg['DFINETransformer']['aqs_temperature'] = .2
        with self.assertRaisesRegex(RuntimeError,'fixed settings differ'):
            changed.model.load_state_dict(model.state_dict(),strict=True)
        changed = fresh_config(TRIAL)
        changed.yaml_cfg['DFINETransformer']['aqs_apply_layer'] = 3
        with self.assertRaisesRegex(RuntimeError,'decoder layer differs'):
            changed.model.load_state_dict(model.state_dict(),strict=True)

    def test_launcher_manifest_syntax_and_no_overwrite(self):
        for name in ('run_scbs_EvidenceContrastAQS_110.sh','sync_scbs_EvidenceContrastAQS.sh'):
            subprocess.run(['bash','-n',str(ROOT/'tools/diagnostics'/name)],check=True)
        for filename in (ROOT/'tools/diagnostics/scbs_EvidenceContrastAQS_sync_files.txt').read_text().splitlines():
            self.assertTrue((ROOT/filename).is_file(),filename)
        with tempfile.TemporaryDirectory(prefix='EvidenceContrastAQS-existing-') as folder:
            result = subprocess.run(['bash',str(ROOT/'tools/diagnostics/run_scbs_EvidenceContrastAQS_110.sh'),'--dry-run',folder],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0); self.assertIn('保留旧结果',result.stderr)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not available locally')
    def test_cuda_fp16_full_detector_gradscaler(self):
        cfg = fresh_config(TRIAL); model = cfg.model.cuda().train(); activate_model(model)
        gt = targets('cuda'); criterion = cfg.criterion.cuda()
        with torch.autocast('cuda',dtype=torch.float16):
            output = model(torch.rand(2,3,128,128,device='cuda'),gt); loss = criterion(copy_output(output),gt)
        scaler = torch.amp.GradScaler('cuda',init_scale=128)
        scaler.scale(sum(loss.values())).backward(); scaler.unscale_(cfg.optimizer)
        self.assertTrue(all(torch.isfinite(v).all() for v in loss.values()))
        for name,p in model.decoder.decoder.query_cls_refiner.named_parameters():
            self.assertIsNotNone(p.grad,name); self.assertTrue(torch.isfinite(p.grad).all(),name)
        scaler.step(cfg.optimizer); scaler.update()


if __name__ == '__main__': unittest.main(verbosity=2)
