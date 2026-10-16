"""Synthetic correctness tests. Passing does NOT establish detection gains."""
import copy
import gc
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.zoo.dfine.behavior_refinement import BEHAVIOR_REFINEMENTS, build_behavior_refinement, attend
from tools.diagnostics.scbs_behavior_refinement_experiment import check_config, parameters, on_off, pack
from tools.diagnostics.eval_behavior_refinement import evaluate
from tools.diagnostics.scbs_behavior_four_queue import completed_training, stream


def config(name=None,size=128):
    filename = f'dfine_s_scbs_hrw_110_AQS_{name}.yml' if name else 'dfine_s_scbs_hrw_110_aqs_layer2_r010.yml'
    with patch('src.core.yaml_config.load_config',lambda p:load_config(p,cfg={})):
        cfg = YAMLConfig(str(ROOT/'configs/dfine'/filename))
    cfg.yaml_cfg['HGNetv2']['pretrained']=False
    cfg.yaml_cfg['eval_spatial_size']=[size,size]
    return cfg


def targets(device='cpu',empty=False):
    return [dict(labels=torch.tensor([0,1,2],device=device),
                 boxes=torch.tensor([[.2,.3,.15,.2],[.5,.5,.2,.3],[.8,.7,.2,.2]],device=device))
            if i==0 and not empty else dict(labels=torch.empty(0,dtype=torch.long,device=device),
                 boxes=torch.empty(0,4,device=device)) for i in range(2)]


def activate(module):
    for name,p in module.named_parameters():
        if name.endswith('up.weight'): torch.nn.init.normal_(p,std=.03)


def inputs(device='cpu'):
    q=torch.randn(2,5,32,device=device,requires_grad=True)
    b=torch.tensor([.5,.5,.6,.7],device=device).expand(2,5,4).clone().requires_grad_()
    s=torch.randn(2,5,3,device=device,requires_grad=True)
    c=torch.randn(2,5,4*9,device=device,requires_grad=True)
    memory=torch.randn(2,8*8+4*4+2*2,32,device=device,requires_grad=True)
    axes=torch.randn(3,32,device=device,requires_grad=True)
    return q,b,s,c,memory,axes


def run(module,x):
    q,b,s,c,memory,axes=x
    context=module.prepare(memory,[(8,8),(4,4),(2,2)])
    if module.name=='QueryContrastInitialization': return module(q,b,context)
    state=None
    if module.name=='ClassEvidenceTransport':
        state=module.update(q,b,context,None,axes,0)
        state=module.update(q,b,context,state,axes,1)
    return module(q,b,s,c,context=context,state=state,axes=axes)


class BehaviorRefinementTests(unittest.TestCase):
    def test_config_zero_initialization_baseline_rng_and_optimizer(self):
        torch.manual_seed(31); base=config().model.eval(); rng=torch.get_rng_state().clone()
        x=torch.rand(2,3,128,128)
        with torch.no_grad(): expected=base(x)
        self.assertFalse(any('behavior_refinement.' in k for k in base.state_dict()))
        for name in BEHAVIOR_REFINEMENTS:
            with self.subTest(module=name):
                check_config(name)
                torch.manual_seed(31); cfg=config(name); model=cfg.model.eval()
                torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
                for k,v in base.state_dict().items(): torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
                with torch.no_grad(): actual=model(x)
                for k in expected: torch.testing.assert_close(actual[k],expected[k],rtol=0,atol=0)
                ids=[id(p) for g in cfg.optimizer.param_groups for p in g['params']]
                self.assertEqual(len(ids),len(set(ids)))
                branch=model.decoder.behavior_refinement
                self.assertTrue(all(id(p) in ids for p in branch.parameters()))
                self.assertEqual(cfg.criterion.weight_dict,config().criterion.weight_dict)
                self.assertEqual(model.decoder.decoder.aqs_layer_indices,(1,))
                print(name,'added parameters:',sum(p.numel() for p in branch.parameters()),flush=True)
                del model,cfg;gc.collect()

    def test_local_amp_gradient_detached_coordinates_and_checkpoint(self):
        for name in BEHAVIOR_REFINEMENTS:
            with self.subTest(module=name):
                m=build_behavior_refinement(name,hidden_dim=32,dim=16,chunk_size=2,use_checkpoint=False)
                x=inputs(); initial=run(m,x)
                torch.testing.assert_close(initial,x[0] if name=='QueryContrastInitialization' else x[2],rtol=0,atol=0)
                initial.square().sum().backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in m.named_parameters() if n.endswith('up.weight')))
                m.zero_grad(set_to_none=True);activate(m);x=inputs()
                with torch.autocast('cpu',dtype=torch.bfloat16): out=run(m,x)
                out.square().sum().backward()
                for n,p in m.named_parameters():
                    self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n)
                self.assertIsNone(x[1].grad);self.assertIsNone(x[3].grad);self.assertIsNone(x[5].grad)
                self.assertGreater(x[0].grad.abs().sum().item(),0)
                if m.uses_maps:self.assertGreater(x[4].grad.abs().sum().item(),0)
                m.zero_grad(set_to_none=True);x=inputs();run(m,x).square().sum().backward()
                grads={n:p.grad.clone() for n,p in m.named_parameters()}
                m.zero_grad(set_to_none=True);m.use_checkpoint=True;run(m,x).square().sum().backward()
                for n,p in m.named_parameters():torch.testing.assert_close(p.grad,grads[n],rtol=2e-4,atol=3e-6)

    def test_permutation_null_runtime_off_and_strict_settings(self):
        h=torch.randn(2,3,16);v=torch.randn(2,9,16);valid=torch.zeros(2,9,dtype=torch.bool)
        value,availability,weights=attend(h,v,valid,torch.randn(3,16),torch.zeros(3))
        torch.testing.assert_close(weights.sum(-1),torch.ones(2,3));self.assertEqual(value.abs().sum(),0)
        self.assertEqual(availability.abs().sum(),0)
        for name in BEHAVIOR_REFINEMENTS:
            with self.subTest(module=name):
                m=build_behavior_refinement(name,hidden_dim=32,dim=16,chunk_size=2).eval();activate(m)
                x=inputs();before=run(m,x);order=torch.tensor([4,2,0,3,1])
                perm=tuple(t[:,order] if i<4 else t for i,t in enumerate(x))
                torch.testing.assert_close(run(m,perm),before[:,order],rtol=2e-5,atol=2e-6)
                m.enabled=False
                torch.testing.assert_close(run(m,x),x[0] if name=='QueryContrastInitialization' else x[2],rtol=0,atol=0)
                clone=build_behavior_refinement(name,hidden_dim=32,dim=16,chunk_size=2)
                clone.load_state_dict(m.state_dict(),strict=True)
                with self.assertRaisesRegex(RuntimeError,'settings differ'):
                    build_behavior_refinement(name,hidden_dim=32,dim=16,grid=5,chunk_size=2).load_state_dict(m.state_dict(),strict=True)
                for bad in ({'dim':10},{'grid':4},{'scale':True},{'scale':0},{'chunk_size':0}):
                    with self.assertRaises(ValueError):build_behavior_refinement(name,**bad)

    def test_full_detector_amp_original_losses_dn_ema(self):
        for name in BEHAVIOR_REFINEMENTS:
            with self.subTest(module=name):
                cfg=config(name);model=cfg.model.train();branch=model.decoder.behavior_refinement;activate(branch)
                image=torch.rand(2,3,128,128);gt=targets()
                with torch.autocast('cpu',dtype=torch.bfloat16):
                    out=model(image,gt);loss=sum(cfg.criterion(out,gt).values())
                self.assertTrue(torch.isfinite(loss));loss.backward()
                for n,p in model.named_parameters():
                    if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all(),n)
                for n,p in branch.named_parameters():self.assertIsNotNone(p.grad,n)
                cfg.optimizer.step();cfg.ema.update(model)
                config(name).model.load_state_dict(cfg.ema.module.state_dict(),strict=True)
                with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
                    empty=model(image,targets(empty=True))
                    self.assertTrue(all(torch.isfinite(v) for v in cfg.criterion(empty,targets(empty=True)).values()))
                # Same weights, DN seed and image. Score-only branches never change box calculations.
                model.eval()
                with torch.no_grad():
                    on=model(image);branch.enabled=False;off=model(image)
                if name!='QueryContrastInitialization':torch.testing.assert_close(on['pred_boxes'],off['pred_boxes'],rtol=0,atol=0)
                self.assertGreater((on['pred_logits']-off['pred_logits']).abs().sum().item(),0)
                del cfg,model,branch;gc.collect()

    def test_cet_three_stage_order_and_dn_isolation(self):
        cfg=config('ClassEvidenceTransport');model=cfg.model.train();m=model.decoder.behavior_refinement
        stages=[];original=m.update
        def update(query,boxes,context,state,axes,stage):
            stages.append((stage,query.shape[1]));return original(query,boxes,context,state,axes,stage)
        with patch.object(m,'update',update):model(torch.rand(2,3,128,128),targets())
        self.assertEqual(stages,[(0,300),(1,300),(2,300)])

    def test_640_and_deploy_profiler(self):
        from calflops import calculate_flops
        for name in BEHAVIOR_REFINEMENTS:
            with self.subTest(module=name):
                m=config(name,size=640).model.eval();activate(m.decoder.behavior_refinement)
                with torch.no_grad():
                    out=m(torch.rand(1,3,640,640))
                    self.assertEqual(out['pred_logits'].shape,(1,300,3));self.assertTrue(torch.isfinite(out['pred_logits']).all())
                    dep=copy.deepcopy(m).deploy()
                    features=[torch.randn(1,256,s,s) for s in (80,40,20)]
                    before=m.decoder(features);after=dep.decoder(features)
                    for k in before:torch.testing.assert_close(before[k],after[k],rtol=3e-5,atol=3e-5)
                    calculate_flops(dep,input_shape=(1,3,640,640),print_results=False)
                del m,dep;gc.collect()

    def test_eval_wrapper_restores_constructor_and_requires_actual_metadata(self):
        from src.zoo.dfine.behavior_refinement import BehaviorRefinementBase
        old=BehaviorRefinementBase.__init__
        args=['--test-only','-r','fake.pth','-u','diagnostic_behavior_refinement_output_enabled=false',
              'diagnostic_behavior_refinement_ablation=off']
        def runner(*args,**kw):self.assertFalse(build_behavior_refinement('ClassEvidenceTransport').enabled)
        evaluate('off',args,runner=runner)
        self.assertIs(BehaviorRefinementBase.__init__,old)
        with self.assertRaises(ValueError):evaluate('off',['--test-only','-r','fake.pth'])

    def test_launcher_dry_run_existing_results_and_log_completion(self):
        script=ROOT/'tools/diagnostics/run_scbs_AQS_BehaviorFour_110.sh'
        for shell in (script,ROOT/'tools/diagnostics/sync_scbs_AQS_BehaviorFour.sh'):
            subprocess.run(['bash','-n',str(shell)],check=True)
        manifest=(ROOT/'tools/diagnostics/scbs_AQS_BehaviorFour_sync_files.txt').read_text().splitlines()
        self.assertEqual(len(manifest),len(set(manifest)))
        self.assertTrue(all((ROOT/p).is_file() for p in manifest))
        self.assertFalse(any('dataset/' in p or p.endswith('.pth') or p.endswith('aqs_layer2_r010.yml') for p in manifest))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);env=dict(os.environ,SCBS_BEHAVIOR_ROOT=tmp)
            dry=subprocess.run(['bash',str(script),'--dry-run'],env=env,text=True,capture_output=True,check=True)
            self.assertTrue(all(name in dry.stdout for name in BEHAVIOR_REFINEMENTS));self.assertEqual(list(root.iterdir()),[])
            out=root/'dfine_s_scbs_hrw_epochs110_AQS_ClassEvidenceTransport_seed0_run1';out.mkdir()
            marker=out/'old.txt';marker.write_text('preserved')
            failed=subprocess.run(['bash',str(script),'--dry-run'],env=env,text=True,capture_output=True)
            self.assertNotEqual(failed.returncode,0);self.assertEqual(marker.read_text(),'preserved')
            records=[dict(epoch=i,test_coco_eval_bbox=[i/1000]) for i in range(110)]
            (out/'log.txt').write_text('\n'.join(json.dumps(x) for x in records));(out/'best_stg2.pth').touch()
            self.assertEqual(completed_training(out)['best_epoch'],109)
            (out/'log.txt').write_text('\n'.join(json.dumps(x) for x in records[:-1]))
            with self.assertRaises(RuntimeError):completed_training(out)
            stream([sys.executable,'-u','-c','print("streamed log")'],root/'stream.log',os.environ.copy())
            self.assertIn('streamed log',(root/'stream.log').read_text())
            with self.assertRaises(FileExistsError):stream([sys.executable,'-c','print(1)'],root/'stream.log',os.environ.copy())

    def test_component_switches_and_parameter_report(self):
        for name in BEHAVIOR_REFINEMENTS:
            with self.subTest(module=name):
                m=build_behavior_refinement(name,hidden_dim=32,dim=16,chunk_size=2).eval();activate(m)
                x=inputs();before=run(m,x)
                if name=='GeometryAwareQueryCompetition':
                    m.geometry_enabled=False
                    m.quality_enabled=False
                    changed=x[1].detach().clone();changed[:,:,0]=torch.linspace(.1,.9,5)
                    y=list(x);y[1]=changed
                    # Isolate geometry relation; also remove quality's visible-box size descriptors.
                    torch.testing.assert_close(run(m,x),run(m,tuple(y)),rtol=0,atol=0)
                m.contrast_enabled=False
                self.assertTrue(torch.isfinite(run(m,x)).all())
                m.quality_enabled=False
                if name!='QueryContrastInitialization':torch.testing.assert_close(run(m,x),x[2],rtol=0,atol=0)
                with tempfile.TemporaryDirectory() as tmp:
                    cp=Path(tmp)/'best.pth';report=Path(tmp)/'report.json'
                    state={'decoder.behavior_refinement.'+k:v for k,v in m.state_dict().items()}
                    torch.save(dict(model=state,ema=dict(module=state),last_epoch=99),cp)
                    result=parameters(cp,report)
                    self.assertEqual(result['sources']['ema']['module'],name)
                    self.assertAlmostEqual(result['sources']['ema']['effective_scale'],.1,places=6)
                    with self.assertRaises(FileExistsError):parameters(cp,report)

    def test_real_evaluation_export_on_off_and_pack(self):
        """Exercise train.py, strict EMA loading, exporter and analyzer on toy COCO."""
        import yaml
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);images=root/'images';images.mkdir()
            dataset=dict(images=[],annotations=[],categories=[dict(id=i,name=n) for i,n in enumerate(('hand-raising','read','write'))])
            for i in (1,2):
                Image.new('RGB',(128,128),(80*i,60,90)).save(images/f'{i}.png')
                dataset['images'].append(dict(id=i,file_name=f'{i}.png',width=128,height=128))
                for c in range(3):
                    dataset['annotations'].append(dict(id=(i-1)*3+c+1,image_id=i,category_id=c,
                        bbox=[10+c*25,20,40,50],area=2000,iscrowd=0))
            annotation=root/'val.json';annotation.write_text(json.dumps(dataset))
            env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg')
            for name in BEHAVIOR_REFINEMENTS:
                with self.subTest(module=name):
                    out=root/name;out.mkdir();cfg=config(name);model=cfg.model.eval();activate(model.decoder.behavior_refinement)
                    cp=out/'best.pth';torch.save(dict(model=model.state_dict(),ema=dict(module=model.state_dict()),last_epoch=99),cp)
                    raw=copy.deepcopy(cfg.yaml_cfg);raw.pop('__include__',None)
                    raw['val_dataloader'].update(total_batch_size=2,num_workers=0)
                    raw['val_dataloader']['dataset'].update(img_folder=str(images),ann_file=str(annotation))
                    for op in raw['val_dataloader']['dataset']['transforms']['ops']:
                        if op['type']=='Resize':op['size']=[128,128]
                    path=out/'config.yml';path.write_text(yaml.safe_dump(raw))
                    for mode in ('on','off'):
                        diag=out/'diagnostics_eval' if mode=='on' else out/f'{name}_off_eval'
                        args=[sys.executable,'-u',str(ROOT/'tools/diagnostics/eval_behavior_refinement.py'),'--mode',mode,
                              '-c',str(path),'-r',str(cp),'--test-only','-d','cpu','--seed=0','--output-dir',str(diag/'evaluation'),
                              '-u','export_confusion_matrix=true','export_query_diagnostics=true',
                              f'diagnostic_behavior_refinement_output_enabled={str(mode=="on").lower()}',
                              f'diagnostic_behavior_refinement_ablation={mode}']
                        evaluated=subprocess.run(args,cwd=ROOT,env=env,capture_output=True,text=True)
                        self.assertEqual(evaluated.returncode,0,evaluated.stdout[-1800:]+evaluated.stderr[-1800:])
                        analyzed=subprocess.run([sys.executable,str(ROOT/'tools/diagnostics/analyze_detection_diagnostics.py'),
                            '--input-dir',str(diag/'evaluation'),'--output-dir',str(diag/'analysis')],cwd=ROOT,env=env,capture_output=True,text=True)
                        self.assertEqual(analyzed.returncode,0,analyzed.stdout[-1800:]+analyzed.stderr[-1800:])
                    on_off(name,out);pack(name,out)
                    self.assertTrue((out/f'{name}_diagnostics.zip').is_file())
                    del cfg,model;gc.collect()

    def test_queue_state_summary_and_missing_reference_without_training(self):
        """Stub subprocess work, NOT training: audit orchestration/result preservation."""
        import tools.diagnostics.scbs_behavior_four_queue as queue
        import tools.diagnostics.scbs_behavior_refinement_experiment as helper
        from tools.diagnostics.scbs_instance_evidence_experiment import digest
        name=BEHAVIOR_REFINEMENTS[0]
        with tempfile.TemporaryDirectory() as tmp:
            port=7781
            root=Path(tmp)
            def snapshot(module,out):
                (out/'experiment_inputs.json').write_text(json.dumps(dict(source_sha256={'train.py':digest(ROOT/'train.py')},
                    annotation_sha256={},resolved_config={})))
            def execute(command,log,env):
                log.write_text('simulated subprocess; not real training')
                out=log.parent
                if log.name=='console.log':
                    rows=[dict(epoch=i,test_coco_eval_bbox=[.5]) for i in range(110)]
                    (out/'log.txt').write_text('\n'.join(json.dumps(x) for x in rows));(out/'best_stg2.pth').write_bytes(b'toy checkpoint')
                if log.name in ('on_analysis_console.log','off_analysis_console.log'):
                    folder=out/'diagnostics_eval' if log.name.startswith('on_') else out/f'{name}_off_eval'
                    (folder/'analysis').mkdir(parents=True)
                    (folder/'analysis/summary.json').write_text(json.dumps(dict(saved_COCO_metrics=dict(AP=.5,AP50=.6,AP75=.5,APm=.4,APl=.5),
                        saved_COCO_per_class={'hand-raising':{'AP':.5},'read':{'AP':.5},'write':{'AP':.5}})))
            args=['queue','run1','--root',str(root),'--port',str(port),'--modules',name]
            with patch.object(sys,'argv',args),patch('tools.diagnostics.wait_for_gpu.wait_for_gpu'),\
                 patch.object(queue.socket,'socket'),\
                 patch.object(queue,'stream',execute),patch.object(queue.subprocess,'run'),\
                 patch.object(helper,'preflight'),patch.object(helper,'snapshot',snapshot),\
                 patch.object(helper,'parameters'),patch.object(helper,'on_off'):
                queue.main()
            queue_dir=root/f'behavior_four_epochs110_seed0_run1_{name}'
            state=json.loads((queue_dir/'status.json').read_text())
            self.assertEqual(state['state'],'complete');self.assertEqual(len(state['completed']),1)
            self.assertIn(name,(queue_dir/'results.csv').read_text())
            out=root/f'dfine_s_scbs_hrw_epochs110_AQS_{name}_seed0_run1'
            self.assertTrue((out/f'{name}_diagnostics.zip').is_file())
            self.assertTrue(all(v['state']=='missing_or_incompatible' for v in json.loads((out/'comparison_status.json').read_text()).values()))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable locally; must run on server')
    def test_cuda_fp16_training(self):
        for name in BEHAVIOR_REFINEMENTS:
            with self.subTest(module=name):
                cfg=config(name);model=cfg.model.cuda().train();activate(model.decoder.behavior_refinement)
                with torch.autocast('cuda',dtype=torch.float16):
                    loss=sum(cfg.criterion(model(torch.rand(2,3,128,128,device='cuda'),targets('cuda')),targets('cuda')).values())
                loss.backward();self.assertTrue(torch.isfinite(loss))
                for n,p in model.named_parameters():
                    if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all(),n)
                del cfg,model;gc.collect();torch.cuda.empty_cache()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable locally; not a full-detector memory guarantee')
    def test_cuda_batch32_300_query_branches(self):
        for name in BEHAVIOR_REFINEMENTS:
            with self.subTest(module=name):
                m=build_behavior_refinement(name).cuda().train();activate(m)
                q=torch.randn(32,300,256,device='cuda',requires_grad=True)
                boxes=torch.tensor([.5,.5,.4,.6],device='cuda').expand(32,300,4)
                scores=torch.randn(32,300,3,device='cuda')
                corners=torch.randn(32,300,4*33,device='cuda')
                memory=torch.randn(32,8400,256,device='cuda',requires_grad=True)
                axes=torch.randn(3,256,device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):
                    context=m.prepare(memory,[(80,80),(40,40),(20,20)])
                    if name=='QueryContrastInitialization':y=m(q,boxes,context)
                    else:
                        state=None
                        if name=='ClassEvidenceTransport':
                            state=m.update(q,boxes,context,None,axes,0)
                            state=m.update(q,boxes,context,state,axes,1)
                        y=m(q,boxes,scores,corners,context=context,state=state,axes=axes)
                y.float().square().mean().backward()
                self.assertTrue(torch.isfinite(y).all())
                for n,p in m.named_parameters():
                    if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all(),n)
                del m,q,boxes,scores,corners,memory,axes,context,y
                if name=='ClassEvidenceTransport':del state
                gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
