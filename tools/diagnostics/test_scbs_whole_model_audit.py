"""Synthetic protocol/capture tests. No user-data training or downloads."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.core import YAMLConfig
from src.core.yaml_utils import load_config as detector_config
from tools.diagnostics import scbs_whole_model_audit as audit
from tools.diagnostics import scbs_feature_probe as feature
from tools.diagnostics.gt_box_classifier import load_config


def model_config(path):
    cfg = YAMLConfig(str(path)); cfg.yaml_cfg = detector_config(str(path),cfg={})
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    cfg.yaml_cfg['eval_spatial_size'] = [128,128]
    return cfg


class WholeModelTests(unittest.TestCase):
    def config(self):
        return load_config(ROOT/'configs/diagnostics/scbs_AQS_WholeModelAudit.yml')

    def fixture(self, root):
        cfg_path = root/'detector.yml'
        cfg_path.write_text('__include__:\n  - ' + str(ROOT/'configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml') +
            '\nHGNetv2:\n  pretrained: false\neval_spatial_size: [128,128]\n'+
            ''.join(f'{s}_dataloader:\n  dataset:\n    img_folder: {root/s}\n'
                    f'    ann_file: {root/(s+".json")}\n' for s in ('train','val')))
        torch.manual_seed(13)
        model = model_config(cfg_path).model.eval().requires_grad_(False)
        with torch.no_grad():
            model.decoder.enc_score_head.bias.zero_()
            model.decoder.enc_score_head.weight.mul_(100.)
            for head in model.decoder.dec_score_head:
                head.bias.fill_(2.)
        image = Image.fromarray(torch.randint(0,256,(128,128,3),dtype=torch.uint8).numpy())
        samples = to_tensor(image).unsqueeze(0)
        with torch.no_grad():
            prediction = model(samples)
        raw = prediction['pred_boxes'][0]
        ids = torch.where((raw[:,:2] > .2).all(-1) & (raw[:,:2] < .8).all(-1))[0][:3]
        self.assertEqual(len(ids),3)
        anns = []
        for i,q in enumerate(ids.tolist()):
            x,y = ((raw[q,:2]-raw[q,2:]/2)*128).tolist(); w,h = (raw[q,2:]*128).tolist()
            anns.append(dict(id=i+1,image_id=1,category_id=i,bbox=[x,y,w,h],area=w*h,iscrowd=0))
        for split in ('train','val'):
            (root/split).mkdir(); image.save(root/split/'one.png'); image.save(root/split/'empty.png')
            (root/f'{split}.json').write_text(json.dumps(dict(
                images=[dict(id=1,file_name='one.png',width=128,height=128),
                        dict(id=2,file_name='empty.png',width=128,height=128)],
                annotations=anns,categories=[dict(id=i,name=name) for i,name in feature.CATEGORIES])))
        state = copy.deepcopy(model.state_dict())
        checkpoint = root/'best_stg2.pth'
        torch.save(dict(model=state,ema={'module':state},last_epoch=109),checkpoint)
        config = self.config()
        config.update(detector_config=str(cfg_path),checkpoint=str(checkpoint),input_size=[128,128],
            progress_every=1,data={s:dict(image_dir=str(root/s),annotation=str(root/f'{s}.json')) for s in ('train','val')},
            probe=dict(seeds=[0],epochs=1,batch_size=8,learning_rate=.001,weight_decay=.01))
        return model,samples,state,config

    def test_explicit_aqs_opt_in_and_baseline_guard_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,_,_,config = self.fixture(Path(tmp))
            frozen,_,_,source = feature.build_frozen_detector(config,torch.device('cpu'))
            self.assertEqual(source,'ema'); self.assertFalse(frozen.training)
            self.assertEqual(frozen.decoder.decoder.aqs_layer_indices,(1,))
            baseline = copy.deepcopy(config); baseline.pop('detector_variant')
            with self.assertRaisesRegex(ValueError,'Baseline only'):
                feature.build_frozen_detector(baseline,torch.device('cpu'))
            invalid = copy.deepcopy(config); invalid['detector_variant']='aqs_l2_l3'
            with self.assertRaises(ValueError): feature.build_frozen_detector(invalid,torch.device('cpu'))
            path = Path(config['detector_config'])
            path.write_text(path.read_text()+'\nDFINETransformer:\n  aqs_apply_layer: 3\n')
            with self.assertRaisesRegex(ValueError,'single L2'):
                feature.build_frozen_detector(config,torch.device('cpu'))

    def test_passive_capture_exact_outputs_weights_rng_and_hook_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            model,x,state,_ = self.fixture(Path(tmp))
            with torch.no_grad():
                before = model(x); rng = torch.get_rng_state().clone()
                with audit.PipelineCapture(model) as capture:
                    actual,maps,queries,stages,_ = capture.forward(x)
                    self.assertFalse(capture.active); self.assertFalse(capture.proposals.active)
                    self.assertEqual(set(maps),set(audit.MAPS)); self.assertEqual(set(queries),set(audit.QUERIES))
                    self.assertEqual(set(stages),set(audit.STAGES))
                    for k in before: torch.testing.assert_close(before[k],actual[k],rtol=0,atol=0)
                    torch.testing.assert_close(stages['decoder_l2_pre_aqs']['boxes'],
                                               stages['decoder_l2_post_aqs']['boxes'],rtol=0,atol=0)
                    raw = stages['decoder_l3_raw_cls']['logits']; quality = stages['decoder_l3_lqe']['logits']
                    torch.testing.assert_close(raw[:,1]-raw[:,2],quality[:,1]-quality[:,2],rtol=0,atol=1e-6)
                    self.assertTrue(torch.equal(raw.argmax(-1),quality.argmax(-1)))
                    selected = stages['encoder_top300']['grid_ids']
                    torch.testing.assert_close(stages['encoder_all']['boxes'][selected],
                                               stages['encoder_top300']['boxes'],rtol=0,atol=0)
                self.assertFalse(capture.handles); self.assertFalse(capture.proposals.handles)
                after = model(x)
                for k in before: torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)
                torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value,state[key],rtol=0,atol=0)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))

    def test_geometric_class_and_non_argmax_coverage_are_distinct(self):
        stage = dict(boxes=torch.tensor([[.5,.5,.4,.4],[.1,.1,.1,.1]]),
            logits=torch.tensor([[3.,2.,1.],[-4.,-3.,-2.]]),valid=torch.tensor([True,True]))
        rows = audit.stage_gt_metrics(stage,torch.tensor([[.3,.3,.7,.7]]),torch.tensor([1]),
            torch.tensor([0]),[.5,.75],.5)
        c = rows[0]['coverage']['0.5']
        self.assertTrue(c['geometric']); self.assertTrue(c['gt_class_high_any_rank'])
        self.assertFalse(c['correct_top1']); self.assertFalse(c['correct_top1_high'])
        self.assertEqual(rows[0]['fixed']['gt_class_margin'],-1.)
        self.assertEqual(audit.stage_gt_metrics(stage,torch.empty(0,4),torch.empty(0,dtype=torch.long),
                                              torch.empty(0,dtype=torch.long),[.5],.5),[])
        self.assertIn('medium',audit.scopes(2,32**2)); self.assertIn('small',audit.scopes(2,32**2))

    def test_final_pairs_keep_duplicates_empty_images_and_class_errors(self):
        pairs = dict(boxes=torch.tensor([[0.,0.,10.,10.],[0.,0.,10.,10.],[30.,30.,40.,40.]]),
            labels=torch.tensor([0,2,1]),query_ids=torch.tensor([5,5,8]),scores=torch.tensor([.9,.8,.7]))
        gt = torch.tensor([[0.,0.,10.,10.]])
        rows,matrix = audit.final_error_records(1,'a.png',pairs,gt,torch.tensor([2]),[dict(id=4)],.5)
        self.assertEqual(int(matrix[2,0]),1)
        self.assertEqual(Counter(r['type'] for r in rows),
            Counter(matched_wrong_class=1,duplicate_or_competing_candidate=1,unmatched_low_overlap=1))
        empty,cm = audit.final_error_records(2,'empty.png',pairs,torch.empty(0,4),torch.empty(0,dtype=torch.long),[],.5)
        self.assertEqual(len(empty),3); self.assertEqual(int(cm[3,:3].sum()),3)

    def test_optional_tide_uses_full_predictions_and_reports_ap50(self):
        class FakeTIDE:
            BOX = 'box'
            def evaluate(self,a,b,mode):
                if (a,b,mode) != ('ground_truth','all_predictions','box'):
                    raise AssertionError('Changed TIDE input protocol')
            def summarize(self): print('Main errors: Cls Loc Both Dupe Bkg Miss')
        fake = SimpleNamespace(TIDE=FakeTIDE,datasets=SimpleNamespace(
            COCO=lambda path:'ground_truth',COCOResult=lambda path:'all_predictions'))
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(sys.modules,{'tidecv':fake}):
                status = audit.optional_tide('anns.json','predictions.json',Path(tmp))
            self.assertEqual(status['status'],'complete')
            self.assertIn('AP50',status['metric'])
            self.assertIn('Main errors',(Path(tmp)/'tide_summary.txt').read_text())

    def test_full_extract_probe_coco_zip_and_provenance_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _,_,state,config = self.fixture(root)
            checkpoint_sha = feature.digest(config['checkpoint'])
            output = root/'AQS_WholeModelAudit_audit1'
            report = audit.extract(config,output,torch.device('cpu'))
            self.assertTrue(report['metadata']['complete'])
            self.assertEqual(report['splits']['val']['images'],2)
            self.assertEqual(report['splits']['val']['GT'],3)
            self.assertEqual(report['splits']['val']['common_matched_gt'],3)
            self.assertIn('AP',report['validation_metrics']); self.assertIn('write',report['validation_metrics']['per_class'])
            predictions = json.loads((output/'val/coco_predictions.json').read_text())
            self.assertEqual(len(predictions),600)
            lines = [json.loads(line) for line in (output/'val/gt_stage_records.jsonl').read_text().splitlines()]
            self.assertEqual(len(lines),3)
            for row in lines:
                # Full-grid fixed ID must not silently be the ordinary query ID.
                self.assertEqual(row['stages']['encoder_top300']['fixed']['query_id'],row['fixed_query_id'])
                self.assertEqual(row['stages']['decoder_l1']['fixed']['query_id'],row['fixed_query_id'])
                torch.testing.assert_close(torch.tensor(row['stages']['encoder_all']['fixed']['iou']),
                    torch.tensor(row['stages']['encoder_top300']['fixed']['iou']),rtol=0,atol=0)
            probes = audit.train_probes(config,output,torch.device('cpu'))
            self.assertTrue(probes['complete']); self.assertEqual(len(probes['experiments']),18)
            summary = audit.finish(output)
            self.assertTrue(summary['complete'])
            post_transitions = [r for r in summary['stage_summary']['transitions']
                                if r['after'] == 'actual_postprocessor_topk']
            self.assertTrue(post_transitions)
            self.assertTrue(all(r['fixed'] == 0 for r in post_transitions))
            with zipfile.ZipFile(output.with_suffix('.zip')) as archive:
                self.assertTrue(any(name.endswith('summary.txt') for name in archive.namelist()))
                self.assertFalse(any(name.endswith('.pth') for name in archive.namelist()))
                self.assertTrue(any(name.endswith('gt_stage_records.jsonl') for name in archive.namelist()))
            self.assertEqual(checkpoint_sha,feature.digest(config['checkpoint']))
            for key,value in torch.load(config['checkpoint'],weights_only=False)['ema']['module'].items():
                torch.testing.assert_close(value,state[key],rtol=0,atol=0)
            with self.assertRaises(FileExistsError): audit.extract(config,output,torch.device('cpu'))
            with self.assertRaises(FileExistsError): audit.train_probes(config,output,torch.device('cpu'))
            with self.assertRaises(FileExistsError): audit.finish(output)
            bad = copy.deepcopy(config); bad['confidence_threshold']=.6
            with self.assertRaises(ValueError): audit.cache_pair(bad,output)

    def test_config_scripts_dry_run_and_output_preservation(self):
        config = self.config(); audit.validate_config(config)
        self.assertIn('aqs_layer2_r010',config['checkpoint'])
        self.assertEqual(config['probe']['seeds'],[0,1,2])
        for name in ('run_scbs_AQS_WholeModelAudit.sh','sync_scbs_AQS_WholeModelAudit.sh'):
            subprocess.run(['bash','-n',str(ROOT/'tools/diagnostics'/name)],check=True)
        manifest = (ROOT/'tools/diagnostics/scbs_AQS_WholeModelAudit_sync_files.txt').read_text().splitlines()
        self.assertFalse(any(p.startswith('src/') for p in manifest))
        self.assertFalse(any(p.startswith('configs/dfine/') for p in manifest))
        for name in manifest: self.assertTrue((ROOT/name).is_file(),name)
        script = ROOT/'tools/diagnostics/run_scbs_AQS_WholeModelAudit.sh'
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ,AQS_AUDIT_OUTPUT_ROOT=tmp)
            process = subprocess.run(['bash',str(script),'--dry-run','audit1'],env=env,capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            self.assertEqual(list(Path(tmp).iterdir()),[])
            output = Path(tmp)/'AQS_WholeModelAudit_audit1'; output.mkdir()
            marker = output/'old.txt'; marker.write_text('keep')
            process = subprocess.run(['bash',str(script),'--dry-run','audit1'],env=env,capture_output=True,text=True)
            self.assertNotEqual(process.returncode,0); self.assertEqual(marker.read_text(),'keep')
            for args in (['--dry-run','../audit1'],['--dry-run','audit0']):
                self.assertNotEqual(subprocess.run(['bash',str(script),*args],env=env,capture_output=True).returncode,0)

    def test_real_640_capture_and_exception_cleanup(self):
        cfg = model_config(ROOT/'configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml')
        cfg.yaml_cfg['eval_spatial_size'] = [640,640]
        model = cfg.model.eval().requires_grad_(False)
        with torch.no_grad():
            x = torch.rand(1,3,640,640)
            expected = model(x)
            with audit.PipelineCapture(model) as capture:
                actual,maps,queries,stages,shapes = capture.forward(x)
                self.assertEqual(shapes,[[80,80],[40,40],[20,20]])
                self.assertEqual(len(stages['encoder_all']['boxes']),8400)
                self.assertEqual(maps['backbone_p2'].shape[-2:],(160,160))
                for key in expected:
                    torch.testing.assert_close(expected[key],actual[key],rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'batch_size=1'):
                with audit.PipelineCapture(model) as capture:
                    capture.forward(torch.rand(2,3,128,128))
            self.assertFalse(capture.handles); self.assertFalse(capture.proposals.handles)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable on the local test host')
    def test_cuda_fp32_passive_capture(self):
        cfg = model_config(ROOT/'configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml')
        model = cfg.model.cuda().eval().requires_grad_(False)
        with torch.no_grad():
            x = torch.rand(1,3,128,128,device='cuda')
            expected = model(x)
            with audit.PipelineCapture(model) as capture:
                actual,*_ = capture.forward(x)
                for key in expected:
                    torch.testing.assert_close(expected[key],actual[key],rtol=0,atol=0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
