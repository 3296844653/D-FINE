"""CPU fixtures and simulated launcher only; never train/run a real GPU job."""

import copy
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core.yaml_utils import load_config
from tools.diagnostics.test_aqs_refine import fresh_config

SCRIPT = ROOT / 'tools/diagnostics/run_scbs_110_run2_diagnostics.sh'
CONFIG = 'dfine_s_scbs_hrw_110_run2_diagnostics.yml'
CHECKPOINT_SHA = '4ff8f014c4497e2d918de11f3fcc6d741bcc19315a9283c8b7786162154ee271'
ANNOTATION_SHA = '623bb8478049dd69d32a92183b0a322bdb8f79f72e6f6f52e491d48923bd5d37'

FAKE_CONDA = r'''
import hashlib, json, os, sys
from pathlib import Path
args = sys.argv[1:]
assert args[:3] == ['run', '--no-capture-output', '-n']
cmd = args[4:]
with Path(os.environ['RUN2_DIAG_TEST_CALLS']).open('a') as handle:
    handle.write(json.dumps(cmd) + '\n')
if cmd[:2] == ['python', '-']:
    code = sys.stdin.read()
    if len(cmd) == 5:
        if os.environ.get('RUN2_DIAG_TEST_PREFLIGHT_FAIL') == '1':
            sys.exit(19)
        checkpoint, output = Path(cmd[3]), Path(cmd[4])
        annotation = Path(os.environ['RUN2_DIAG_TEST_ANNOTATION'])
        inputs = dict(checkpoint=str(checkpoint.resolve()),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            annotation_file=str(annotation.resolve()),
            annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
            validation_dataset={'img_folder': '/fixture/images', 'ann_file': str(annotation)},
            images=1, GT=3, GT_by_category={'0':1,'1':1,'2':1}, weight_source='ema')
        output.mkdir(exist_ok=False)
        (output/'export_inputs.json').write_text(json.dumps(inputs))
    else:
        sys.argv = cmd[1:]
        exec(compile(code, '<production-run2-finalizer>', 'exec'))
elif cmd[0] == 'torchrun':
    print('SIMULATED LIVE VALIDATION LOG; NO TRAINING', flush=True)
    if os.environ.get('RUN2_DIAG_TEST_EVAL_FAIL') == '1':
        sys.exit(17)
    root = Path(cmd[cmd.index('--output-dir')+1])
    inputs = json.loads((root.parent/'export_inputs.json').read_text())
    folder = root/'query_diagnostics'; folder.mkdir(parents=True)
    (folder/'raw_queries').mkdir()
    (folder/'raw_queries/1.pt').write_bytes(b'MOCK raw query, not actual prediction')
    query = dict(complete=os.environ.get('RUN2_DIAG_TEST_INCOMPLETE') != '1', schema_version=2,
        num_classes=3, postprocessor_top_k=300, images=1, queries=300,
        checkpoint=inputs['checkpoint'], checkpoint_sha256=inputs['checkpoint_sha256'],
        confidence_threshold=.5, gt_iou_threshold=.5, neighbor_iou_threshold=.8)
    (folder/'metadata.json').write_text(json.dumps(query))
    for name in ('queries.csv','gt_coverage.csv','selected_predictions.csv','nearby_predictions.csv'):
        (folder/name).write_text('MOCK header\n')
    matrix_dir = root/'confusion_matrix'; matrix_dir.mkdir()
    matrix = dict(category_ids=[0,1,2],
        row_labels=['hand-raising','read','write','unmatched_predictions'],
        column_labels=['hand-raising','read','write','unmatched_GT'],
        confidence_threshold=.5, iou_threshold=.5,
        matching='class-blind greedy highest-IoU one-to-one (existing Validator)',
        matrix_counts=[[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,1,0,0]], GT_count=3,
        predictions_after_threshold=4,
        metadata=dict(checkpoint=inputs['checkpoint'], weight_source='ema',
                      validation_dataset=inputs['validation_dataset']))
    if os.environ.get('RUN2_DIAG_TEST_BAD_SOURCE') == '1':
        matrix['metadata']['weight_source'] = 'model'
    if os.environ.get('RUN2_DIAG_TEST_BAD_THRESHOLD') == '1':
        matrix['confidence_threshold'] = .3
    (matrix_dir/'confusion_matrix.json').write_text(json.dumps(matrix))
    for name in ('confusion_matrix/confusion_matrix.txt',
                 'confusion_matrix/confusion_matrix_counts.png',
                 'confusion_matrix/confusion_matrix_row_normalized.png',
                 'validation_metrics.json','per_class_metrics.json','eval.pth'):
        (root/name).write_bytes(b'MOCK ONLY')
elif 'analyze_detection_diagnostics.py' in cmd[1]:
    print('SIMULATED ANALYSIS LOG; NO RESCORING', flush=True)
    if os.environ.get('RUN2_DIAG_TEST_ANALYSIS_FAIL') == '1':
        sys.exit(23)
    root = Path(cmd[cmd.index('--output-dir')+1]); root.mkdir()
    inputs = json.loads((root.parent/'export_inputs.json').read_text())
    summary = dict(checkpoint=inputs['checkpoint'], checkpoint_sha256=inputs['checkpoint_sha256'],
        annotation_file=inputs['annotation_file'], weight_source='ema', images=1, queries=300,
        GT=3, confidence_threshold=.5, IoU_threshold=.5, selected_pairs_above_confidence=4)
    if os.environ.get('RUN2_DIAG_TEST_BAD_HASH') == '1':
        summary['checkpoint_sha256'] = 'wrong'
    if os.environ.get('RUN2_DIAG_TEST_BAD_COUNT') == '1':
        summary['queries'] = 299
    (root/'summary.json').write_text(json.dumps(summary))
    (root/'summary.txt').write_text('SIMULATED SUMMARY')
    for name in ('prediction_records.jsonl','best_query_cross_classification.json',
                 'best_query_cross_classification.jsonl','gt_best_iou_distribution.json',
                 'coco_pr_curves.json','score_distributions_and_quality.json',
                 'coco_pr_iou0.5.png','coco_pr_iou0.75.png','diagnostic_tp_fp_scores.png',
                 'score_quality_relationship.png','gt_best_iou_distribution.png'):
        if os.environ.get('RUN2_DIAG_TEST_MISSING_FILE') == '1' and name == 'coco_pr_curves.json':
            continue
        (root/name).write_text('MOCK')
    names = {'0':'hand-raising', '1':'read', '2':'write'}
    ranges = {'all':[0,1e10], 'small':[0,1024], 'medium':[1024,9216], 'large':[9216,1e10]}
    sizes = {area:dict(GT_count=3 if area == 'all' else 0,
                       class_counts={name:1 if area == 'all' else 0 for name in names.values()})
             for area in ranges}
    scale = dict(class_names=names, units='0~1; null means undefined, not zero', area_ranges=ranges,
                 metrics={cls:{area:dict(AP=.5, AP50=.6, AP75=.4, AR100=.8) for area in ranges} for cls in names})
    def group(count):
        return dict(GT_count=count, type_counts={k:count if k == 'F' else 0 for k in 'ABCDEFG'},
                    high_correct_selected_candidate_exists=count, no_high_correct_selected_candidate=0)
    cross = dict(confidence_threshold=.5, IoU_threshold=.5, all=group(3),
                 by_class={name:group(1) for name in names.values()},
                 by_size_and_class={area:{name:group(1 if area == 'all' else 0) for name in names.values()} for area in ranges})
    if os.environ.get('RUN2_DIAG_TEST_BAD_SCALE') == '1':
        scale['metrics']['2'].pop('medium')
    if os.environ.get('RUN2_DIAG_TEST_BAD_SCALE_UNIT') == '1':
        scale['metrics']['2']['medium']['AP'] = 32.75
    if os.environ.get('RUN2_DIAG_TEST_BAD_CROSS') == '1':
        cross['by_size_and_class']['medium']['write']['GT_count'] = 1
    (root/'gt_size_statistics.json').write_text(json.dumps(sizes))
    (root/'best_query_cross_classification.json').write_text(json.dumps(cross))
    if os.environ.get('RUN2_DIAG_TEST_MISSING_SCALE') != '1':
        (root/'coco_per_class_and_size.json').write_text(json.dumps(scale))
    (root/'gt_records.jsonl').write_text('{}\n'*3)
    if os.environ.get('RUN2_DIAG_TEST_CHANGED_CHECKPOINT') == '1':
        Path(inputs['checkpoint']).write_bytes(b'SIMULATED changed checkpoint')
    if os.environ.get('RUN2_DIAG_TEST_CHANGED_ANNOTATION') == '1':
        Path(inputs['annotation_file']).write_text('CHANGED')
else:
    raise RuntimeError(cmd)
'''


class SCBSRun2FullDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        folder = tempfile.TemporaryDirectory(prefix='scbs-run2-full-diag-')
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.output = self.root/'完整诊断 output'
        self.checkpoint = self.root/'best_stg2.pth'
        self.checkpoint.write_bytes(b'MOCK checkpoint; not model weights')
        self.annotation = self.root/'instances_val.json'; self.annotation.write_text('{}')
        self.calls = self.root/'calls.jsonl'
        self.conda = self.root/'fake-conda'
        self.conda.write_text('#!' + sys.executable + '\n' + FAKE_CONDA)
        self.conda.chmod(0o755)
        self.env = dict(os.environ, SCBS_RUN2_CONDA=str(self.conda), SCBS_RUN2_ENV='fixture',
                        SCBS_RUN2_CHECKPOINT=str(self.checkpoint), SCBS_RUN2_GPU='0',
                        RUN2_DIAG_TEST_CALLS=str(self.calls), RUN2_DIAG_TEST_ANNOTATION=str(self.annotation))
        for key in list(self.env):
            if key.startswith('RUN2_DIAG_TEST_') and key not in ('RUN2_DIAG_TEST_CALLS','RUN2_DIAG_TEST_ANNOTATION'):
                self.env.pop(key)

    def launch(self, *arguments):
        return subprocess.run(['bash', str(SCRIPT), str(self.output), *arguments],
                              cwd='/tmp', env=self.env, text=True, capture_output=True, timeout=35)

    def records(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def test_configuration_only_adds_exports_and_preserves_baseline(self):
        base, diag = fresh_config('dfine_s_scbs_hrw_110.yml'), fresh_config(CONFIG)
        expected, actual = copy.deepcopy(base.yaml_cfg), copy.deepcopy(diag.yaml_cfg)
        expected.update(export_confusion_matrix=True, export_query_diagnostics=True,
                        query_diagnostic_conf_thresh=.5, query_diagnostic_iou_thresh=.5,
                        query_diagnostic_neighbor_iou_thresh=.8)
        for config in (expected, actual):
            config.pop('__include__', None)
        self.assertEqual(expected, actual)
        torch.manual_seed(21); original = base.model
        torch.manual_seed(21); exported = diag.model
        self.assertEqual(set(original.state_dict()), set(exported.state_dict()))
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, exported.state_dict()[key], rtol=0, atol=0)
        self.assertFalse(exported.decoder.use_aqs_refine)
        self.assertFalse(exported.decoder.use_rw_local_detail)

    def test_syntax_and_default_identity(self):
        subprocess.run(['bash', '-n', str(SCRIPT)], check=True)
        codes = re.findall(r"<<'PY'\n(.*?)\nPY", SCRIPT.read_text(), flags=re.S)
        self.assertEqual(len(codes), 2)
        for code in codes:
            compile(code, str(SCRIPT), 'exec')
        self.assertIn('epochs110_seed0_run2', SCRIPT.read_text())
        self.assertIn('$run2_diag_run/diagnostics_export1', SCRIPT.read_text())
        self.assertIn(CHECKPOINT_SHA, codes[0]); self.assertIn(ANNOTATION_SHA, codes[0])

    def test_real_preflight_strict_states_counts_and_safe_hash_checks(self):
        image_dir = self.root/'images'; image_dir.mkdir()
        (image_dir/'fixture.jpg').write_bytes(b'file existence fixture')
        dataset = dict(images=[dict(id=1,file_name='fixture.jpg',width=100,height=100)],
            categories=[dict(id=i,name=n) for i,n in enumerate(('hand-raising','read','write'))],
            annotations=[dict(id=i+1,image_id=1,category_id=i,bbox=[i*20,0,10,10]) for i in range(3)])
        dataset['annotations'] += [dict(id=4,image_id=1,category_id=2,bbox=[0,0,10,10],iscrowd=1),
                                   dict(id=5,image_id=1,category_id=1,bbox=[200,0,10,10])]
        self.annotation.write_text(json.dumps(dataset))
        state_config = fresh_config('dfine_s_scbs_hrw_110.yml')
        # Anchors/valid_mask are persistent buffers: the real Run2 uses640.
        state_config.yaml_cfg['eval_spatial_size'] = [640, 640]
        state = state_config.model.state_dict()
        checkpoint = dict(last_epoch=109, model=state, ema={'module': state})
        torch.save(checkpoint, self.checkpoint)
        code = re.findall(r"<<'PY'\n(.*?)\nPY", SCRIPT.read_text(), flags=re.S)[0]

        def fixture_config(path, cfg):
            config = load_config(path, cfg=cfg)
            config['val_dataloader']['dataset'].update(img_folder=str(image_dir), ann_file=str(self.annotation))
            return config

        def check(output, checkpoint_hash=None, annotation_hash=None):
            # ONLY substitute expected fingerprint constants for test fixtures;
            # all production config/hash/state/GT validation code is executed.
            fixture_code = code.replace(CHECKPOINT_SHA, checkpoint_hash or hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())
            fixture_code = fixture_code.replace(ANNOTATION_SHA, annotation_hash or hashlib.sha256(self.annotation.read_bytes()).hexdigest())
            with patch('src.core.yaml_utils.load_config', fixture_config), redirect_stdout(io.StringIO()), \
                 patch.object(sys, 'argv', ['-', str(ROOT/'configs/dfine'/CONFIG), str(self.checkpoint), str(output)]):
                exec(compile(fixture_code, '<actual-run2-preflight>', 'exec'), {})

        check(self.output)
        inputs = json.loads((self.output/'export_inputs.json').read_text())
        self.assertEqual(inputs['GT'], 3)
        self.assertEqual(inputs['GT_by_category'], {'0':1,'1':1,'2':1})
        self.assertEqual(inputs['weight_source'], 'ema')
        for name, arguments in (('wrong-checkpoint', {'checkpoint_hash':'0'*64}),
                                ('wrong-annotation', {'annotation_hash':'0'*64})):
            output = self.root/name
            with self.assertRaisesRegex(AssertionError, '指纹'):
                check(output, **arguments)
            self.assertFalse(output.exists())
        for name, value, message in (
            ('no-ema', dict(last_epoch=109, model=state), '缺少model或EMA'),
            ('132', dict(checkpoint, last_epoch=131), '不是110轮'),
            ('other-module', dict(checkpoint, ema={'module':dict(state, **{'decoder.rw_local_detail.extra': torch.zeros(1)})}), 'Unexpected key'),
        ):
            torch.save(value, self.checkpoint)
            with self.assertRaisesRegex((AssertionError, RuntimeError), message):
                check(self.root/name)
            self.assertFalse((self.root/name).exists())

    def test_mock_workflow_only_validates_analyzes_logs_and_keeps_checkpoint(self):
        before = self.checkpoint.read_bytes()
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.records(); self.assertEqual(len(calls), 4)
        validation = calls[1]
        for argument in ('--test-only','--standalone','--local-addr=127.0.0.1','--use-amp','--seed=0'):
            self.assertIn(argument, validation)
        self.assertNotIn('-t', validation)
        self.assertEqual(validation[validation.index('-r')+1], str(self.checkpoint))
        self.assertTrue(validation[validation.index('-c')+1].endswith(CONFIG))
        self.assertEqual(self.checkpoint.read_bytes(), before)
        self.assertIn('SIMULATED LIVE', (self.output/'logs/evaluation.log').read_text())
        self.assertIn('SIMULATED ANALYSIS', (self.output/'logs/analysis.log').read_text())
        report = json.loads((self.output/'workflow_summary.json').read_text())
        self.assertTrue(report['complete']); self.assertEqual(report['queries'], 300)
        self.assertEqual(report['comparison_baseline'], 'SCB-S / D-FINE-S / 110-epoch Baseline Run2')
        self.assertTrue((self.output/'analysis/coco_per_class_and_size.json').is_file())
        self.assertIn('Run2', (self.output/'run2_diagnostics_summary.txt').read_text())

    def test_existing_results_preserved_including_dangling_symlink(self):
        self.output.mkdir(); (self.output/'keep.txt').write_text('keep')
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.output/'keep.txt').read_text(), 'keep')
        self.assertEqual(self.records(), [])
        link = self.root/'dangling'; link.symlink_to(self.root/'missing')
        self.output = link
        self.assertNotEqual(self.launch().returncode, 0)
        self.assertTrue(link.is_symlink()); self.assertEqual(self.records(), [])

    def test_missing_checkpoint_and_preflight_failure_create_no_results(self):
        self.env['SCBS_RUN2_CHECKPOINT'] = str(self.root/'missing.pth')
        self.assertNotEqual(self.launch().returncode, 0)
        self.assertFalse(self.output.exists()); self.assertEqual(self.records(), [])
        self.env['SCBS_RUN2_CHECKPOINT'] = str(self.checkpoint)
        self.env['RUN2_DIAG_TEST_PREFLIGHT_FAIL'] = '1'
        self.assertEqual(self.launch().returncode, 19)
        self.assertFalse(self.output.exists())

    def test_failed_evaluation_or_analysis_stops_without_success(self):
        for variable, code, calls in (('RUN2_DIAG_TEST_EVAL_FAIL',17,2), ('RUN2_DIAG_TEST_ANALYSIS_FAIL',23,3)):
            self.env[variable] = '1'; self.output = self.root/variable
            self.calls.unlink(missing_ok=True)
            result = self.launch()
            self.assertEqual(result.returncode, code, result.stdout + result.stderr)
            self.assertEqual(len(self.records()), calls)
            self.assertTrue((self.output/'logs/evaluation.log').is_file())
            self.assertFalse((self.output/'workflow_summary.json').exists())
            self.env.pop(variable)

    def test_incomplete_or_mismatched_outputs_never_marked_complete(self):
        for variable in ('RUN2_DIAG_TEST_INCOMPLETE','RUN2_DIAG_TEST_BAD_HASH','RUN2_DIAG_TEST_BAD_SOURCE',
                         'RUN2_DIAG_TEST_BAD_THRESHOLD','RUN2_DIAG_TEST_BAD_COUNT','RUN2_DIAG_TEST_MISSING_FILE',
                         'RUN2_DIAG_TEST_CHANGED_CHECKPOINT','RUN2_DIAG_TEST_CHANGED_ANNOTATION',
                         'RUN2_DIAG_TEST_MISSING_SCALE','RUN2_DIAG_TEST_BAD_SCALE',
                         'RUN2_DIAG_TEST_BAD_SCALE_UNIT','RUN2_DIAG_TEST_BAD_CROSS'):
            with self.subTest(variable=variable):
                self.output = self.root/variable; self.env[variable] = '1'
                result = self.launch()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.output/'workflow_summary.json').exists())
                self.env.pop(variable)

    def test_invalid_gpu_or_arguments_rejected_before_subprocess(self):
        self.env['SCBS_RUN2_GPU'] = '0,1'
        self.assertNotEqual(self.launch().returncode, 0)
        self.env['SCBS_RUN2_GPU'] = '0'
        self.assertNotEqual(self.launch('/tmp/extra').returncode, 0)
        self.assertEqual(self.records(), [])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
