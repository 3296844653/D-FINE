"""SERF synthetic/integration checks. Passing is NOT evidence of AP improvement."""
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
sys.path.insert(0, str(ROOT))
from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.zoo.dfine.spatial_evidence_relation_fusion import SERF_PLACEMENTS, SpatialEvidenceRelationFusion
from tools.diagnostics.scbs_serf_experiment import check_config, parameters, on_off, pack
from tools.diagnostics.eval_serf import evaluate


def config(name=None, size=128):
    filename = f'dfine_s_scbs_hrw_110_AQS_{name}.yml' if name else 'dfine_s_scbs_hrw_110_aqs_layer2_r010.yml'
    with patch('src.core.yaml_config.load_config', lambda p: load_config(p, cfg={})):
        cfg = YAMLConfig(str(ROOT / 'configs/dfine' / filename))
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    cfg.yaml_cfg['eval_spatial_size'] = [size, size]
    return cfg


def targets(device='cpu', empty=False):
    return [dict(labels=torch.tensor([0, 1, 2], device=device),
        boxes=torch.tensor([[.2,.3,.15,.2],[.5,.5,.2,.3],[.8,.7,.2,.2]], device=device))
        if i == 0 and not empty else dict(labels=torch.empty(0, dtype=torch.long, device=device),
        boxes=torch.empty(0, 4, device=device)) for i in range(2)]


def small(placement='SERF_Encoder', **options):
    return SpatialEvidenceRelationFusion(32, 64, placement, dim=16, window=4, halo=1, chunk=2, **options)


def activate(module):
    torch.nn.init.normal_(module.up.weight, std=.02)


class SERFTests(unittest.TestCase):
    def test_config_baseline_rng_initialization_and_optimizer(self):
        torch.manual_seed(31); base = config().model.eval(); rng = torch.get_rng_state().clone()
        image = torch.rand(2, 3, 128, 128)
        with torch.no_grad(): expected = base(image)
        self.assertFalse(any(k.startswith('serf.') for k in base.state_dict()))
        for name in SERF_PLACEMENTS:
            with self.subTest(placement=name):
                check_config(name)
                torch.manual_seed(31); cfg = config(name); model = cfg.model.eval()
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                for k, v in base.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
                with torch.no_grad(): actual = model(image)
                for k in expected: torch.testing.assert_close(actual[k], expected[k], rtol=0, atol=0)
                ids = [id(p) for g in cfg.optimizer.param_groups for p in g['params']]
                self.assertEqual(len(ids), len(set(ids)))
                self.assertTrue(all(id(p) in ids for p in model.serf.parameters()))
                self.assertEqual(cfg.criterion.weight_dict, config().criterion.weight_dict)
                self.assertEqual(model.decoder.decoder.aqs_layer_indices, (1,))
                self.assertEqual(model.decoder.num_queries, 300)
                self.assertEqual(model.serf.p4_channels, 256 if name == 'SERF_Encoder' else 512)
                print(name, 'added parameters:', sum(p.numel() for p in model.serf.parameters()), flush=True)
                del cfg, model; gc.collect()

    def test_halo_window_order_masking_centers_and_attention(self):
        m = small()
        image = torch.arange(64.).reshape(1, 1, 8, 8)
        windows = m._windows(image).reshape(4, 6, 6)
        torch.testing.assert_close(windows[0, 1:5, 1:5], image[0, 0, :4, :4])
        self.assertEqual(windows[0, 0].abs().sum(), 0)
        self.assertEqual(windows[0, :, 0].abs().sum(), 0)
        self.assertEqual(windows[-1, -1].abs().sum(), 0)
        self.assertEqual(windows[-1, :, -1].abs().sum(), 0)
        data = torch.randn(2, 36, 16)
        valid = torch.zeros(2, 36, dtype=torch.bool); valid[:, 7:11] = True
        value, a, centers, relation, read = m._interaction(data, data, torch.randn(2, 36, 4), valid)
        self.assertEqual(a[..., ~valid[0]].abs().sum(), 0)
        torch.testing.assert_close(a.sum(-1), torch.ones(2, 4))
        torch.testing.assert_close(relation.sum(-1), torch.ones(2, 4, 4))
        self.assertEqual(relation.diagonal(dim1=-2, dim2=-1).abs().sum(), 0)
        torch.testing.assert_close(read.sum(-1), torch.ones(2, 36))
        self.assertTrue(torch.isfinite(value).all())
        torch.testing.assert_close(centers, a @ m.coordinates)
        # Isolate geometry: all pixel appearances/semantics are identical.
        constant = torch.randn(2, 1, 16).expand(-1, 36, -1)
        assignment = torch.full((2, 36, 4), -10.)
        for slot in range(4): assignment[:, 7 + slot, slot] = 10
        shifted = assignment.clone(); shifted[:, :, [2, 3]] = shifted[:, :, [3, 2]]
        before = m._interaction(constant, constant, assignment, valid)[3]
        after = m._interaction(constant, constant, shifted, valid)[3]
        self.assertGreater((before - after).abs().sum().item(), 0)
        m.geometry_enabled = False
        before = m._interaction(constant, constant, assignment, valid)[3]
        after = m._interaction(constant, constant, shifted, valid)[3]
        torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)

    def test_odd_small_maps_zero_head_gradients_and_cpu_amp(self):
        for h, w in ((1, 1), (3, 2), (9, 7)):
            with self.subTest(shape=(h, w)):
                m = small(); p3 = torch.randn(2, 32, h, w, requires_grad=True)
                p4 = torch.randn(2, 64, max(1, (h+1)//2), max(1, (w+1)//2), requires_grad=True)
                initial = m(p3, p4)
                torch.testing.assert_close(initial, p3, rtol=0, atol=0)
                initial.square().mean().backward()
                self.assertGreater(m.up.weight.grad.abs().sum().item(), 0)
                m.zero_grad(set_to_none=True); activate(m)
                with torch.autocast('cpu', dtype=torch.bfloat16): output = m(p3, p4)
                output.square().mean().backward()
                self.assertEqual(output.shape, p3.shape); self.assertTrue(torch.isfinite(output).all())
                for n, p in m.named_parameters():
                    self.assertIsNotNone(p.grad, n); self.assertTrue(torch.isfinite(p.grad).all(), n)
                self.assertGreater(p4.grad.abs().sum().item(), 0)

    def test_checkpoint_chunk_equivalence_and_component_switches(self):
        m = small(use_checkpoint=False); activate(m)
        p3 = torch.randn(2, 32, 9, 7, requires_grad=True); p4 = torch.randn(2, 64, 5, 4, requires_grad=True)
        expected = m(p3, p4); expected.square().mean().backward()
        grads = {n: p.grad.clone() for n, p in m.named_parameters()}
        m.zero_grad(); m.use_checkpoint = True; actual = m(p3, p4); actual.square().mean().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for n, p in m.named_parameters(): torch.testing.assert_close(p.grad, grads[n], rtol=2e-5, atol=2e-7)
        m.eval(); m.chunk = 100
        torch.testing.assert_close(m(p3, p4), expected, rtol=2e-5, atol=2e-6)
        m.geometry_enabled = False; no_geometry = m(p3, p4)
        self.assertGreater((no_geometry - expected).abs().sum().item(), 0)
        m.relation_enabled = False; no_relation = m(p3, p4)
        self.assertGreater((no_relation - no_geometry).abs().sum().item(), 0)
        m.enabled = False
        self.assertIs(m(p3, p4), p3)

    def test_stage_routing_and_unchanged_aqs_call(self):
        for name in SERF_PLACEMENTS:
            with self.subTest(placement=name):
                model = config(name).model.eval(); activate(model.serf)
                features, decoder_inputs, aqs_queries = {}, [], []
                hooks = [model.backbone.register_forward_hook(lambda _, a, out: features.update(backbone=out)),
                    model.encoder.register_forward_pre_hook(lambda _, a: features.update(encoder_input=a[0])),
                    model.encoder.register_forward_hook(lambda _, a, out: features.update(encoder=out)),
                    model.decoder.register_forward_pre_hook(lambda _, a: decoder_inputs.append(a[0])),
                    model.decoder.decoder.query_cls_refiner.register_forward_pre_hook(lambda _, a: aqs_queries.append(a[0].shape[1]))]
                with torch.no_grad(): model(torch.rand(2, 3, 128, 128))
                for h in hooks: h.remove()
                source, downstream = (features['backbone'], features['encoder_input']) if name == 'SERF_Backbone' else (features['encoder'], decoder_inputs[0])
                self.assertGreater((source[0] - downstream[0]).abs().sum().item(), 0)
                for i in (1, 2): self.assertIs(source[i], downstream[i])
                self.assertEqual(aqs_queries, [300])

    def test_full_detector_original_losses_dn_empty_amp_ema_and_pretrained_schema(self):
        for name in SERF_PLACEMENTS:
            with self.subTest(placement=name):
                cfg = config(name); model = cfg.model.train(); activate(model.serf)
                # Original backbone pretrained state schema is not extended by the output adapter.
                config().model.backbone.load_state_dict(model.backbone.state_dict(), strict=True)
                image, gt = torch.rand(2, 3, 128, 128), targets()
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    output = model(image, gt); losses = cfg.criterion(output, gt); loss = sum(losses.values())
                self.assertTrue(torch.isfinite(loss)); self.assertTrue(any('_dn_' in k for k in losses))
                self.assertTrue(any('_aux_' in k for k in losses))
                loss.backward()
                for n, p in model.serf.named_parameters():
                    self.assertIsNotNone(p.grad, n); self.assertTrue(torch.isfinite(p.grad).all(), n)
                for n, p in model.named_parameters():
                    if p.grad is not None: self.assertTrue(torch.isfinite(p.grad).all(), n)
                cfg.optimizer.step(); cfg.ema.update(model)
                config(name).model.load_state_dict(cfg.ema.module.state_dict(), strict=True)
                with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
                    self.assertTrue(all(torch.isfinite(v) for v in cfg.criterion(model(image, targets(empty=True)), targets(empty=True)).values()))
                del cfg, model; gc.collect()

    def test_strict_settings_report_and_invalid_combinations(self):
        m = small(); clone = small(); clone.load_state_dict(m.state_dict(), strict=True)
        with self.assertRaisesRegex(RuntimeError, 'settings differ'):
            small('SERF_Backbone').load_state_dict(m.state_dict(), strict=True)
        for options in ({'dim': 10}, {'slots': 1}, {'halo': -1}, {'window': 1}, {'scale': True}, {'scale': 0}, {'chunk': 0}, {'heads': 3}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                settings = dict(dim=16, window=4, halo=1); settings.update(options)
                SpatialEvidenceRelationFusion(32, 64, 'SERF_Encoder', **settings)
        for owner, key, value in [('DFINE', 'night_probe', 'encoder_cross_gate'),
                ('DFINETransformer', 'behavior_refinement_module', 'GeometryAwareQueryCompetition'),
                ('DFINETransformer', 'use_aqs_refine', False), ('HGNetv2', 'return_idx', [0,1,2,3])]:
            cfg = config('SERF_Encoder'); cfg.yaml_cfg[owner][key] = value
            with self.assertRaises(ValueError): _ = cfg.model
        with self.assertRaises(ValueError): m(torch.randn(2, 32, 4, 4), torch.randn(2, 64, 5, 4))
        with tempfile.TemporaryDirectory() as tmp:
            cp, report = Path(tmp)/'best.pth', Path(tmp)/'parameters.json'
            state = {'serf.' + k: v for k, v in m.state_dict().items()}
            torch.save(dict(model=state, ema=dict(module=state), last_epoch=99), cp)
            result = parameters(cp, report)
            self.assertAlmostEqual(result['sources']['ema']['effective_scale'], .1, places=6)
            self.assertEqual(result['sources']['ema']['module'], 'SERF_Encoder')
            with self.assertRaises(FileExistsError): parameters(cp, report)

    def test_full_640_resolution_and_deploy_profiler(self):
        from calflops import calculate_flops
        for name in SERF_PLACEMENTS:
            with self.subTest(placement=name):
                model = config(name, 640).model.eval(); activate(model.serf)
                with torch.no_grad():
                    output = model(torch.rand(1, 3, 640, 640))
                    self.assertEqual(output['pred_logits'].shape, (1, 300, 3))
                    self.assertTrue(all(torch.isfinite(v).all() for v in output.values()))
                    deployed = copy.deepcopy(model).deploy()
                    p3 = torch.randn(1, model.serf.p3_channels, 80, 80)
                    p4 = torch.randn(1, model.serf.p4_channels, 40, 40)
                    torch.testing.assert_close(model.serf(p3, p4), deployed.serf(p3, p4), rtol=0, atol=0)
                    calculate_flops(deployed, input_shape=(1,3,640,640), print_results=False)
                del model, deployed; gc.collect()

    def test_eval_wrapper_and_real_export_on_off(self):
        import yaml
        from PIL import Image
        original = SpatialEvidenceRelationFusion.__init__
        args = ['--test-only', '-r', 'fake.pth', '-u', 'diagnostic_serf_output_enabled=false', 'diagnostic_serf_ablation=serf_off']
        def runner(*a, **kw): self.assertFalse(small().enabled)
        evaluate('off', args, runner=runner)
        self.assertIs(SpatialEvidenceRelationFusion.__init__, original)
        with self.assertRaises(ValueError): evaluate('off', ['--test-only', '-r', 'fake.pth'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); images = root/'images'; images.mkdir()
            data = dict(images=[], annotations=[], categories=[dict(id=i,name=n) for i,n in enumerate(('hand-raising','read','write'))])
            for i in (1,2):
                Image.new('RGB', (128,128), (80*i,60,90)).save(images/f'{i}.png')
                data['images'].append(dict(id=i,file_name=f'{i}.png',width=128,height=128))
                for c in range(3):
                    data['annotations'].append(dict(id=(i-1)*3+c+1,image_id=i,category_id=c,bbox=[10+c*25,20,40,50],area=2000,iscrowd=0))
            annotation = root/'val.json'; annotation.write_text(json.dumps(data))
            env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MPLBACKEND='Agg')
            for name in SERF_PLACEMENTS:
                out = root/name; out.mkdir(); cfg = config(name); model = cfg.model.eval(); activate(model.serf)
                cp = out/'best.pth'; torch.save(dict(model=model.state_dict(),ema=dict(module=model.state_dict()),last_epoch=99),cp)
                raw = copy.deepcopy(cfg.yaml_cfg); raw.pop('__include__',None)
                raw['val_dataloader'].update(total_batch_size=2,num_workers=0)
                raw['val_dataloader']['dataset'].update(img_folder=str(images),ann_file=str(annotation))
                for op in raw['val_dataloader']['dataset']['transforms']['ops']:
                    if op['type']=='Resize':op['size']=[128,128]
                path = out/'config.yml'; path.write_text(yaml.safe_dump(raw))
                for mode in ('on','off'):
                    diag = out/'diagnostics_eval' if mode=='on' else out/f'{name}_off_eval'
                    command = [sys.executable,'-u',str(ROOT/'tools/diagnostics/eval_serf.py'),'--mode',mode,
                        '-c',str(path),'-r',str(cp),'--test-only','-d','cpu','--seed=0','--output-dir',str(diag/'evaluation'),
                        '-u','export_confusion_matrix=true','export_query_diagnostics=true',
                        f'diagnostic_serf_output_enabled={str(mode=="on").lower()}',f'diagnostic_serf_ablation=serf_{mode}']
                    result = subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stdout[-2500:]+result.stderr[-2500:])
                    result = subprocess.run([sys.executable,str(ROOT/'tools/diagnostics/analyze_detection_diagnostics.py'),
                        '--input-dir',str(diag/'evaluation'),'--output-dir',str(diag/'analysis')],cwd=ROOT,env=env,capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stdout[-2500:]+result.stderr[-2500:])
                on_off(name,out); pack(name,out)
                self.assertTrue((out/f'{name}_diagnostics.zip').is_file())
                del cfg,model; gc.collect()

    def test_launch_dry_run_existing_output_protection_and_manifest(self):
        script = ROOT/'tools/diagnostics/run_scbs_AQS_SERF_110.sh'
        for shell in (script, ROOT/'tools/diagnostics/sync_scbs_AQS_SERF.sh'):
            subprocess.run(['bash','-n',str(shell)],check=True)
        manifest = (ROOT/'tools/diagnostics/scbs_AQS_SERF_sync_files.txt').read_text().splitlines()
        self.assertEqual(len(manifest),len(set(manifest)))
        self.assertTrue(all((ROOT/p).is_file() for p in manifest))
        self.assertFalse(any(p.startswith('dataset/') or p.endswith('.pth') or p.endswith('aqs_layer2_r010.yml') for p in manifest))
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ,SCBS_SERF_ROOT=tmp)
            dry = subprocess.run(['bash',str(script),'--dry-run'],env=env,text=True,capture_output=True,check=True)
            self.assertTrue(all(name in dry.stdout for name in SERF_PLACEMENTS))
            self.assertEqual(list(Path(tmp).iterdir()),[])
            existing = Path(tmp)/'dfine_s_scbs_hrw_epochs110_AQS_SERF_Backbone_seed0_run1'; existing.mkdir()
            marker = existing/'old.txt'; marker.write_text('preserved')
            failed = subprocess.run(['bash',str(script),'--dry-run'],env=env,text=True,capture_output=True)
            self.assertNotEqual(failed.returncode,0); self.assertEqual(marker.read_text(),'preserved')

    def test_queue_results_csv_and_missing_reference(self):
        import tools.diagnostics.scbs_serf_queue as queue
        import tools.diagnostics.scbs_serf_experiment as helper
        import tools.diagnostics.scbs_behavior_four_queue as shared
        from tools.diagnostics.scbs_instance_evidence_experiment import digest
        name = SERF_PLACEMENTS[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def snapshot(module,out):
                (out/'experiment_inputs.json').write_text(json.dumps(dict(source_sha256={'train.py':digest(ROOT/'train.py')},annotation_sha256={},resolved_config={})))
            def stream(command,log,env):
                log.write_text('simulated subprocess, NOT real training'); out = log.parent
                if log.name=='console.log':
                    (out/'log.txt').write_text('\n'.join(json.dumps(dict(epoch=i,test_coco_eval_bbox=[.5])) for i in range(110)))
                    (out/'best_stg2.pth').write_bytes(b'toy')
                if log.name in ('on_analysis_console.log','off_analysis_console.log'):
                    folder = out/'diagnostics_eval' if log.name.startswith('on_') else out/f'{name}_off_eval'
                    (folder/'analysis').mkdir(parents=True)
                    (folder/'analysis/summary.json').write_text(json.dumps(dict(saved_COCO_metrics=dict(AP=.5,AP50=.6,AP75=.5,AP_medium=.4,AP_large=.55),saved_COCO_per_class={})))
            old_handler = __import__('signal').getsignal(__import__('signal').SIGTERM)
            try:
                with patch.object(sys,'argv',['queue','run1','--root',tmp,'--modules',name]),\
                     patch('tools.diagnostics.wait_for_gpu.wait_for_gpu'),patch.object(queue.socket,'socket'),\
                     patch.object(queue.subprocess,'run'),patch.object(shared,'stream',stream),\
                     patch.object(helper,'preflight'),patch.object(helper,'snapshot',snapshot),\
                     patch.object(helper,'parameters'),patch.object(helper,'on_off'):
                    queue.main()
            finally:
                __import__('signal').signal(__import__('signal').SIGTERM,old_handler)
            folder = root/f'AQS_SERF_two_epochs110_seed0_run1_{name}'
            self.assertEqual(json.loads((folder/'status.json').read_text())['state'],'complete')
            import csv
            with (folder/'results.csv').open() as f: result = next(csv.DictReader(f))
            self.assertEqual(float(result['APm']),.4); self.assertEqual(float(result['APl']),.55)
            out = root/f'dfine_s_scbs_hrw_epochs110_AQS_{name}_seed0_run1'
            self.assertTrue((out/f'{name}_diagnostics.zip').is_file())
            self.assertTrue(all(v['state']=='missing_or_incompatible' for v in json.loads((out/'comparison_status.json').read_text()).values()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable locally; run on server')
    def test_cuda_fp16_full_detector_training(self):
        for name in SERF_PLACEMENTS:
            cfg = config(name); model = cfg.model.cuda().train(); activate(model.serf)
            gt = targets('cuda')
            with torch.autocast('cuda',dtype=torch.float16):
                loss = sum(cfg.criterion(model(torch.rand(2,3,128,128,device='cuda'),gt),gt).values())
            loss.backward(); self.assertTrue(torch.isfinite(loss))
            for n,p in model.named_parameters():
                if p.grad is not None: self.assertTrue(torch.isfinite(p.grad).all(),n)
            del cfg,model; gc.collect(); torch.cuda.empty_cache()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable locally; branch-only memory smoke test')
    def test_cuda_batch32_feature_branch(self):
        for name in SERF_PLACEMENTS:
            p4_channels = 256 if name=='SERF_Encoder' else 512
            m = SpatialEvidenceRelationFusion(256,p4_channels,name).cuda().train(); activate(m)
            p3 = torch.randn(32,256,80,80,device='cuda',requires_grad=True)
            p4 = torch.randn(32,p4_channels,40,40,device='cuda',requires_grad=True)
            with torch.autocast('cuda',dtype=torch.float16): output = m(p3,p4)
            output.float().square().mean().backward(); self.assertTrue(torch.isfinite(output).all())
            for n,p in m.named_parameters():
                self.assertIsNotNone(p.grad,n); self.assertTrue(torch.isfinite(p.grad).all(),n)
            del m,p3,p4,output; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
