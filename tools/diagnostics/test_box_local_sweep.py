"""BoxLocal hyperparameter regression/integration tests; NOT evidence of gains."""
import copy
import gc
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.zoo.dfine.night_probes import BoxLocalQueryAdapter, AdaptiveROITokens, zero_output
from tools.diagnostics.box_local_sweep_spec import DEFAULTS, PROFILES, settings, module_name
from tools.diagnostics.scbs_box_local_sweep_experiment import check_config, parameters, on_off, pack, reference_summary
from tools.diagnostics.eval_box_local_sweep import evaluate


def config(profile=None,size=128,legacy=False):
    filename = f'dfine_s_scbs_hrw_110_{module_name(profile)}.yml' if profile else 'dfine_s_scbs_hrw_110.yml'
    with patch('src.core.yaml_config.load_config',lambda p:load_config(p,cfg={})):
        cfg = YAMLConfig(str(ROOT/'configs/dfine'/filename))
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    cfg.yaml_cfg['eval_spatial_size'] = [size,size]
    if legacy:cfg.yaml_cfg['DFINE']['night_probe'] = 'decoder_l2_local'
    return cfg


def branch(model):return model.decoder.decoder.night_local_adapter


def activate(module):torch.nn.init.normal_(module.fuse[-1].weight,std=.02)


def targets(device='cpu',empty=False):
    return [dict(labels=torch.tensor([0,1,2],device=device),
        boxes=torch.tensor([[.2,.3,.2,.2],[.5,.5,.3,.3],[.8,.7,.2,.2]],device=device))
        if i==0 and not empty else dict(labels=torch.empty(0,dtype=torch.long,device=device),
        boxes=torch.empty(0,4,device=device)) for i in range(2)]


class LegacyBox(nn.Module):
    """Frozen old construction/formula, for default schema and RNG comparisons."""
    def __init__(self,hidden_dim=32,dim=16):
        super().__init__()
        self.sampler = AdaptiveROITokens(hidden_dim,dim)
        self.fuse = nn.Sequential(nn.LayerNorm(2*dim),nn.Linear(2*dim,dim),nn.SiLU(),nn.Linear(dim,hidden_dim))
        zero_output(self.fuse[-1])
    def forward(self,q,b,p3):
        # Hard-coded historical 9-point, .15-offset formula.
        offsets = .15*self.sampler.offset(self.sampler.query_norm(q)).float().tanh()
        relative = (self.sampler.relative_grid+offsets.reshape(*q.shape[:2],9,2)).clamp(-.49,.49)
        position = self.sampler.position(relative).to(q.dtype)
        tokens = self.sampler._sample(self.sampler.p3_projection(p3),b,relative)+position+self.sampler.scale_embedding[0]
        tokens = self.sampler.token_norm(tokens)
        content = self.sampler.query_projection(self.sampler.query_norm(q))
        logits = (self.sampler.key(tokens)*content.unsqueeze(2)).sum(-1).float()/(self.sampler.dim**.5)
        local = (logits.softmax(2).to(tokens.dtype).unsqueeze(-1)*self.sampler.value(tokens)).sum(2)
        return q+.1*self.fuse(torch.cat((content,local),-1))


class BoxLocalSweepTests(unittest.TestCase):
    def test_legacy_default_tensor_schema_rng_outputs_and_gradients(self):
        torch.manual_seed(17);old = LegacyBox();rng = torch.get_rng_state().clone()
        torch.manual_seed(17);new = BoxLocalQueryAdapter(32,16)
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        self.assertEqual(list(old.state_dict()),list(new.state_dict()))
        for k,v in old.state_dict().items():torch.testing.assert_close(v,new.state_dict()[k],rtol=0,atol=0)
        activate(old);new.load_state_dict(old.state_dict(),strict=True)
        q = torch.randn(2,5,32);b = torch.tensor([.5,.5,.6,.7]).expand(2,5,4);p3 = torch.randn(2,32,12,12)
        before,after = old(q,b,p3),new(q,b,p3)
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        before.square().sum().backward();after.square().sum().backward()
        for (n,p),(n2,p2) in zip(old.named_parameters(),new.named_parameters()):
            self.assertEqual(n,n2);torch.testing.assert_close(p.grad,p2.grad,rtol=0,atol=0)
        clone = BoxLocalQueryAdapter(32,16);clone.load_state_dict(old.state_dict(),strict=True)

    def test_config_one_factor_baseline_identity_and_optimizer(self):
        torch.manual_seed(23);base = config().model.eval();rng = torch.get_rng_state().clone()
        image = torch.rand(2,3,128,128)
        with torch.no_grad():expected = base(image)
        self.assertFalse(any('night_local_adapter' in k for k in base.state_dict()))
        for profile in PROFILES:
            with self.subTest(group=profile):
                check_config(profile);spec = settings(profile)
                self.assertEqual(sum(spec[k]!=DEFAULTS[k] for k in DEFAULTS),1)
                torch.manual_seed(23);cfg = config(profile);model = cfg.model.eval();m = branch(model)
                torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
                for k,v in base.state_dict().items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
                with torch.no_grad():actual = model(image)
                for k in expected:torch.testing.assert_close(actual[k],expected[k],rtol=0,atol=0)
                self.assertEqual(m.sampler.num_points,spec['box_local_grid_size']**2)
                self.assertEqual(m.sampler.offset.out_features,2*m.sampler.num_points)
                self.assertEqual(m.dim,spec['box_local_dim']);self.assertEqual(m.residual_scale,spec['box_local_residual_scale'])
                self.assertFalse(model.decoder.use_aqs_refine)
                self.assertEqual(cfg.criterion.weight_dict,config().criterion.weight_dict)
                ids = [id(p) for g in cfg.optimizer.param_groups for p in g['params']]
                self.assertEqual(len(ids),len(set(ids)));self.assertTrue(all(id(p) in ids for p in m.parameters()))
                print(profile,'BoxLocal added parameters:',sum(p.numel() for p in m.parameters()),flush=True)
                del model,cfg;gc.collect()

    def test_sampling_grids_bounds_zero_offset_amp_and_detached_boxes(self):
        for grid in (2,3,5):
            for offset in (0.,.1,.15,.2):
                with self.subTest(grid=grid,offset=offset):
                    m = BoxLocalQueryAdapter(32,16,grid_size=grid,offset_scale=offset,record_settings=True);activate(m)
                    torch.nn.init.normal_(m.sampler.offset.weight,std=.1)
                    q = torch.randn(2,5,32,requires_grad=True)
                    b = torch.tensor([.5,.5,.6,.7]).expand(2,5,4).clone().requires_grad_()
                    p3 = torch.randn(2,32,12,12,requires_grad=True)
                    relative = m.sampler.relative_locations(q)
                    self.assertEqual(relative.shape,(2,5,grid**2,2));self.assertLessEqual(relative.detach().abs().max().item(),.490001)
                    if offset==0:
                        torch.testing.assert_close(relative,m.sampler.relative_grid.expand_as(relative),rtol=0,atol=0)
                    with torch.autocast('cpu',dtype=torch.bfloat16):out = m(q,b,p3)
                    out.square().mean().backward();self.assertIsNone(b.grad)
                    for n,p in m.named_parameters():
                        self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n)
                    if offset==0:self.assertEqual(m.sampler.offset.weight.grad.abs().sum().item(),0)
                    else:self.assertGreater(m.sampler.offset.weight.grad.abs().sum().item(),0)
                    changed = q.detach().clone();changed[:,0]+=5
                    torch.testing.assert_close(m(changed,b,p3)[:,1:],m(q,b,p3)[:,1:],rtol=0,atol=0)

    def test_full_detector_original_losses_dn_empty_ema_and_l2_routing(self):
        for profile in PROFILES:
            with self.subTest(group=profile):
                cfg = config(profile);model = cfg.model.train();m = branch(model)
                image,gt = torch.rand(2,3,128,128),targets()
                initial = sum(cfg.criterion(model(image,gt),gt).values());initial.backward()
                self.assertGreater(m.fuse[-1].weight.grad.abs().sum().item(),0)
                model.zero_grad(set_to_none=True);activate(m);observed=[]
                h = m.register_forward_pre_hook(lambda _,a:observed.append(a[0].shape[1]))
                with torch.autocast('cpu',dtype=torch.bfloat16):
                    out = model(image,gt);losses = cfg.criterion(out,gt);loss = sum(losses.values())
                h.remove();self.assertEqual(observed,[300]);self.assertTrue(torch.isfinite(loss));loss.backward()
                self.assertTrue(any('_dn_' in k for k in losses));self.assertTrue(any('_aux_' in k for k in losses))
                self.assertFalse(any('margin' in k or 'pairwise_ce' in k for k in losses))
                for n,p in m.named_parameters():self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n)
                cfg.optimizer.step();cfg.ema.update(model);config(profile).model.load_state_dict(cfg.ema.module.state_dict(),strict=True)
                with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
                    self.assertTrue(all(torch.isfinite(v) for v in cfg.criterion(model(image,targets(empty=True)),targets(empty=True)).values()))
                del cfg,model;gc.collect()

    def test_strict_nonlearned_settings_and_old_checkpoint_compatibility(self):
        m = BoxLocalQueryAdapter(32,16,residual_scale=.05,record_settings=True)
        BoxLocalQueryAdapter(32,16,residual_scale=.05,record_settings=True).load_state_dict(m.state_dict(),strict=True)
        with self.assertRaisesRegex(RuntimeError,'settings differ'):
            BoxLocalQueryAdapter(32,16,residual_scale=.2,record_settings=True).load_state_dict(m.state_dict(),strict=True)
        with self.assertRaisesRegex(RuntimeError,'settings differ'):
            BoxLocalQueryAdapter(32,16,residual_scale=.05,offset_scale=.1,record_settings=True).load_state_dict(m.state_dict(),strict=True)
        old = config(legacy=True).model
        config(legacy=True).model.load_state_dict(old.state_dict(),strict=True)
        self.assertFalse(any('_box_local_settings' in k for k in old.state_dict()))
        # New recorded configuration must not silently load unrecorded old settings.
        with self.assertRaises(RuntimeError):config('r005').model.load_state_dict(old.state_dict(),strict=True)

    def test_invalid_settings_and_unused_parameter_guards(self):
        for options in ({'dim':10},{'dim':True},{'grid_size':1},{'grid_size':True},
                {'offset_scale':-.1},{'offset_scale':True},{'residual_scale':-1},{'residual_scale':True},{'record_settings':1}):
            with self.subTest(options=options),self.assertRaises(ValueError):BoxLocalQueryAdapter(32,**options)
        cfg = config();cfg.yaml_cfg['DFINE']['box_local_offset_scale']=.1
        with self.assertRaises(ValueError):_ = cfg.model
        for owner,key,value in [('DFINETransformer','use_aqs_refine',True),('DFINE','serf_placement','SERF_Encoder')]:
            cfg = config('r005');cfg.yaml_cfg[owner][key]=value
            with self.assertRaises(ValueError):_ = cfg.model

    def test_report_fixed_values_and_runtime_off(self):
        for profile in PROFILES:
            with self.subTest(group=profile),tempfile.TemporaryDirectory() as tmp:
                model = config(profile).model.eval();m = branch(model);activate(m)
                q = torch.randn(2,5,256);b = torch.tensor([.5,.5,.6,.7]).expand(2,5,4);p3 = torch.randn(2,256,12,12)
                self.assertGreater((m(q,b,p3)-q).abs().sum().item(),0);m.enabled=False;self.assertIs(m(q,b,p3),q)
                cp,report = Path(tmp)/'best.pth',Path(tmp)/'parameters.json'
                torch.save(dict(model=model.state_dict(),ema=dict(module=model.state_dict()),last_epoch=99),cp)
                result = parameters(cp,report)['sources']['ema']
                self.assertEqual(result['dim'],settings(profile)['box_local_dim'])
                self.assertEqual(result['points'],settings(profile)['box_local_grid_size']**2)
                self.assertEqual(result['offset_scale'],settings(profile)['box_local_offset_scale'])
                self.assertEqual(result['residual_scale'],settings(profile)['box_local_residual_scale'])
                with self.assertRaises(FileExistsError):parameters(cp,report)
                del model;gc.collect()

    def test_640_forward_and_deploy_profiler(self):
        from calflops import calculate_flops
        for profile in ('r005','d032','d096','g2','g5','o000'):
            model = config(profile,640).model.eval();activate(branch(model))
            with torch.no_grad():
                out = model(torch.rand(1,3,640,640));self.assertEqual(out['pred_logits'].shape,(1,300,3))
                self.assertTrue(all(torch.isfinite(v).all() for v in out.values()))
                deployed = copy.deepcopy(model).deploy()
                features = [torch.randn(1,256,s,s) for s in (80,40,20)]
                before,after = model.decoder(features),deployed.decoder(features)
                for k in before:torch.testing.assert_close(before[k],after[k],rtol=3e-5,atol=3e-5)
                calculate_flops(deployed,input_shape=(1,3,640,640),print_results=False)
            del model,deployed;gc.collect()

    def test_wrapper_restore_and_real_coco_on_off_export(self):
        import yaml
        from PIL import Image
        original = BoxLocalQueryAdapter.__init__
        args = ['--test-only','-r','fake.pth','-u','diagnostic_box_local_output_enabled=false','diagnostic_box_local_ablation=box_off']
        def runner(*a,**kw):self.assertFalse(BoxLocalQueryAdapter(32,16).enabled)
        evaluate('off',args,runner=runner);self.assertIs(BoxLocalQueryAdapter.__init__,original)
        with self.assertRaises(ValueError):evaluate('off',['--test-only','-r','fake.pth'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);images = root/'images';images.mkdir()
            data = dict(images=[],annotations=[],categories=[dict(id=i,name=n) for i,n in enumerate(('hand-raising','read','write'))])
            for i in (1,2):
                Image.new('RGB',(128,128),(80*i,60,90)).save(images/f'{i}.png')
                data['images'].append(dict(id=i,file_name=f'{i}.png',width=128,height=128))
                for c in range(3):data['annotations'].append(dict(id=(i-1)*3+c+1,image_id=i,category_id=c,bbox=[10+c*25,20,40,50],area=2000,iscrowd=0))
            annotation = root/'val.json';annotation.write_text(json.dumps(data))
            env = dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg')
            for profile in ('r005','g5'):
                out = root/profile;out.mkdir();cfg = config(profile);model = cfg.model.eval();activate(branch(model))
                cp = out/'best.pth';torch.save(dict(model=model.state_dict(),ema=dict(module=model.state_dict()),last_epoch=99),cp)
                raw = copy.deepcopy(cfg.yaml_cfg);raw.pop('__include__',None)
                raw['val_dataloader'].update(total_batch_size=2,num_workers=0)
                raw['val_dataloader']['dataset'].update(img_folder=str(images),ann_file=str(annotation))
                for op in raw['val_dataloader']['dataset']['transforms']['ops']:
                    if op['type']=='Resize':op['size']=[128,128]
                path = out/'config.yml';path.write_text(yaml.safe_dump(raw))
                for mode in ('on','off'):
                    diag = out/'diagnostics_eval' if mode=='on' else out/f'{module_name(profile)}_off_eval'
                    command = [sys.executable,'-u',str(ROOT/'tools/diagnostics/eval_box_local_sweep.py'),'--mode',mode,
                        '-c',str(path),'-r',str(cp),'--test-only','-d','cpu','--seed=0','--output-dir',str(diag/'evaluation'),
                        '-u','export_confusion_matrix=true','export_query_diagnostics=true',
                        f'diagnostic_box_local_output_enabled={str(mode=="on").lower()}',f'diagnostic_box_local_ablation=box_{mode}']
                    result = subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stdout[-2500:]+result.stderr[-2500:])
                    result = subprocess.run([sys.executable,str(ROOT/'tools/diagnostics/analyze_detection_diagnostics.py'),
                        '--input-dir',str(diag/'evaluation'),'--output-dir',str(diag/'analysis')],cwd=ROOT,env=env,capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stdout[-2500:]+result.stderr[-2500:])
                on_off(profile,out);pack(module_name(profile),out)
                self.assertTrue((out/f'{module_name(profile)}_diagnostics.zip').is_file())
                del cfg,model;gc.collect()

    def test_script_default_all_selection_manifest_and_output_protection(self):
        launcher = ROOT/'tools/diagnostics/run_scbs_BoxLocalQueryAdapter_l2_sweep_110.sh'
        for script in (launcher,ROOT/'tools/diagnostics/sync_scbs_BoxLocalQueryAdapter_l2_sweep.sh'):
            subprocess.run(['bash','-n',str(script)],check=True)
        lines = (ROOT/'tools/diagnostics/scbs_BoxLocalQueryAdapter_l2_sweep_sync_files.txt').read_text().splitlines()
        self.assertEqual(len(lines),len(set(lines)));self.assertTrue(all((ROOT/p).is_file() for p in lines))
        self.assertNotIn('configs/dfine/dfine_s_scbs_hrw_110.yml',lines)
        self.assertFalse(any(p.startswith('dataset/') or p.endswith('.pth') for p in lines))
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ,SCBS_BOX_ROOT=tmp)
            default = subprocess.run(['bash',str(launcher),'--dry-run'],env=env,capture_output=True,text=True,check=True)
            self.assertIn('r005:',default.stdout);self.assertIn('r020:',default.stdout);self.assertNotIn('o010:',default.stdout)
            all_groups = subprocess.run(['bash',str(launcher),'--all','--dry-run'],env=env,capture_output=True,text=True,check=True)
            self.assertTrue(all(g+':' in all_groups.stdout for g in PROFILES));self.assertEqual(list(Path(tmp).iterdir()),[])
            existing = Path(tmp)/'dfine_s_scbs_hrw_epochs110_BoxLocalQueryAdapter_l2_r005_seed0_run1';existing.mkdir()
            marker = existing/'old.txt';marker.write_text('preserved')
            failed = subprocess.run(['bash',str(launcher),'--dry-run'],env=env,capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0);self.assertEqual(marker.read_text(),'preserved')

    def test_queue_csv_metrics_and_missing_reference_without_training(self):
        import tools.diagnostics.scbs_box_local_sweep as queue
        import tools.diagnostics.scbs_box_local_sweep_experiment as helper
        import tools.diagnostics.scbs_behavior_four_queue as shared
        from tools.diagnostics.scbs_instance_evidence_experiment import digest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def snapshot(g,out):(out/'experiment_inputs.json').write_text(json.dumps(dict(source_sha256={'train.py':digest(ROOT/'train.py')},annotation_sha256={},resolved_config={})))
            def stream(command,log,env):
                log.write_text('simulated subprocess, not real training');out = log.parent
                if log.name=='console.log':
                    (out/'log.txt').write_text('\n'.join(json.dumps(dict(epoch=i,test_coco_eval_bbox=[.5])) for i in range(110)));(out/'best_stg2.pth').write_bytes(b'toy')
                if log.name in ('on_analysis_console.log','off_analysis_console.log'):
                    folder = out/'diagnostics_eval' if log.name.startswith('on_') else out/f'{module_name("r005")}_off_eval'
                    (folder/'analysis').mkdir(parents=True)
                    fake = dict(saved_COCO_metrics=dict(AP=.5,AP50=.6,AP75=.5,AP_medium=.4,AP_large=.55),
                        saved_COCO_per_class={str(i):dict(AP=.5) for i in range(3)},validator_confusion=dict(unmatched_predictions=3),
                        confusion_pairs=[dict(gt='write',prediction='read',count=2)],best_query_cross_classification=dict(by_class=dict(write=dict(no_high_correct_selected_candidate=1))))
                    (folder/'analysis/summary.json').write_text(json.dumps(fake))
            old_handler = signal.getsignal(signal.SIGTERM)
            try:
                with patch.object(sys,'argv',['queue','run1','--root',tmp,'--groups','r005']),\
                     patch('tools.diagnostics.wait_for_gpu.wait_for_gpu'),patch.object(queue.socket,'socket'),\
                     patch.object(queue.subprocess,'run'),patch.object(shared,'stream',stream),\
                     patch.object(helper,'preflight'),patch.object(helper,'snapshot',snapshot),\
                     patch.object(helper,'parameters'),patch.object(helper,'on_off'):queue.main()
            finally:signal.signal(signal.SIGTERM,old_handler)
            result = root/'BoxLocalQueryAdapter_l2_sweep_epochs110_seed0_run1_r005'
            self.assertEqual(json.loads((result/'status.json').read_text())['state'],'complete')
            import csv
            with (result/'results.csv').open() as f:row = next(csv.DictReader(f))
            self.assertEqual(float(row['APm']),.4);self.assertEqual(float(row['write_AP']),.5);self.assertEqual(int(row['write_candidate_gap']),1)
            out = root/'dfine_s_scbs_hrw_epochs110_BoxLocalQueryAdapter_l2_r005_seed0_run1'
            self.assertTrue((out/f'{module_name("r005")}_diagnostics.zip').is_file())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable locally; check on server')
    def test_cuda_fp16_full_detector_training(self):
        for profile in PROFILES:
            cfg = config(profile);model = cfg.model.cuda().train();activate(branch(model));gt = targets('cuda')
            with torch.autocast('cuda',dtype=torch.float16):
                loss = sum(cfg.criterion(model(torch.rand(2,3,128,128,device='cuda'),gt),gt).values())
            loss.backward();self.assertTrue(torch.isfinite(loss))
            for n,p in model.named_parameters():
                if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all(),n)
            del cfg,model;gc.collect();torch.cuda.empty_cache()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable locally; branch-only memory check')
    def test_cuda_batch32_300_queries_all_groups(self):
        for profile in PROFILES:
            s = settings(profile)
            m = BoxLocalQueryAdapter(256,s['box_local_dim'],grid_size=s['box_local_grid_size'],offset_scale=s['box_local_offset_scale'],residual_scale=s['box_local_residual_scale'],record_settings=True).cuda().train();activate(m)
            q = torch.randn(32,300,256,device='cuda',requires_grad=True)
            b = torch.tensor([.5,.5,.5,.6],device='cuda').expand(32,300,4)
            p3 = torch.randn(32,256,80,80,device='cuda',requires_grad=True)
            with torch.autocast('cuda',dtype=torch.float16):out = m(q,b,p3)
            out.float().square().mean().backward();self.assertTrue(torch.isfinite(out).all())
            for n,p in m.named_parameters():self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n)
            del m,q,b,p3,out;gc.collect();torch.cuda.empty_cache()


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
