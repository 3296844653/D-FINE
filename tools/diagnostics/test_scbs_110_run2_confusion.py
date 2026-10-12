"""CPU fixtures and mocked launchers only; never train or run server/GPU jobs."""
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
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core.yaml_utils import load_config
from src.solver.confusion_export import export_confusion_matrix
from src.solver.validator import Validator
from tools.diagnostics.test_aqs_refine import fresh_config

SCRIPT = ROOT / 'tools/diagnostics/run_scbs_110_run2_confusion.sh'
CONFIG = 'dfine_s_scbs_hrw_110_run2_confusion.yml'

FAKE_CONDA = r'''
import hashlib, json, os, sys
from pathlib import Path
args = sys.argv[1:]
assert args[:3] == ['run', '--no-capture-output', '-n'], args
cmd = args[4:]
with Path(os.environ['RUN2_TEST_CALLS']).open('a') as handle:
    handle.write(json.dumps(cmd) + '\n')
if cmd[:2] == ['python', '-']:
    code = sys.stdin.read()
    if len(cmd) == 5:
        if os.environ.get('RUN2_TEST_PREFLIGHT_FAIL') == '1':
            sys.exit(19)
        _, _, config, checkpoint, output = cmd
        checkpoint, output = Path(checkpoint), Path(output)
        annotation = Path(os.environ['RUN2_TEST_ANNOTATION'])
        inputs = dict(checkpoint=str(checkpoint.resolve()),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            annotation_file=str(annotation.resolve()),
            annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
            validation_dataset={'img_folder': '/fixture/images', 'ann_file': str(annotation)},
            images=1, GT=3, GT_by_category={'0': 1, '1': 1, '2': 1}, weight_source='ema',
            confidence_threshold=.5, iou_threshold=.5)
        output.mkdir(exist_ok=False)
        (output/'export_inputs.json').write_text(json.dumps(inputs))
        print('SIMULATED PREFLIGHT; NO GPU', flush=True)
    else:
        sys.argv = cmd[1:]
        exec(compile(code, '<real-finalizer>', 'exec'))
elif cmd[0] == 'torchrun':
    print('SIMULATED LIVE EVALUATION LOG; NO TRAINING', flush=True)
    if os.environ.get('RUN2_TEST_EVAL_FAIL') == '1':
        sys.exit(17)
    output = Path(cmd[cmd.index('--output-dir')+1])
    inputs = json.loads((output/'export_inputs.json').read_text())
    folder = output/'confusion_matrix'; folder.mkdir()
    matrix = dict(category_ids=[0,1,2], row_labels=['hand-raising','read','write','unmatched_predictions'],
        column_labels=['hand-raising','read','write','unmatched_GT'],
        confidence_threshold=.5, iou_threshold=.5,
        matching='class-blind greedy highest-IoU one-to-one (existing Validator)',
        matrix_counts=[[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,1,0,0]], GT_count=3,
        predictions_after_threshold=4,
        metadata=dict(checkpoint=inputs['checkpoint'], weight_source='ema',
                      validation_dataset=inputs['validation_dataset']))
    if os.environ.get('RUN2_TEST_BAD_SOURCE') == '1':
        matrix['metadata']['weight_source'] = 'model'
    if os.environ.get('RUN2_TEST_BAD_THRESHOLD') == '1':
        matrix['confidence_threshold'] = .3
    if os.environ.get('RUN2_TEST_BAD_COUNT') == '1':
        matrix['GT_count'] = 4
    (folder/'confusion_matrix.json').write_text(json.dumps(matrix))
    for filename in ('confusion_matrix/confusion_matrix.txt', 'confusion_matrix/confusion_matrix_counts.png',
                     'confusion_matrix/confusion_matrix_row_normalized.png', 'validation_metrics.json',
                     'per_class_metrics.json', 'eval.pth'):
        if os.environ.get('RUN2_TEST_INCOMPLETE') == '1' and filename.endswith('counts.png'):
            continue
        (output/filename).write_bytes(b'MOCK ONLY')
    if os.environ.get('RUN2_TEST_CHECKPOINT_CHANGED') == '1':
        Path(inputs['checkpoint']).write_bytes(b'SIMULATED changed checkpoint')
else:
    raise RuntimeError(cmd)
'''


class SCBS110Run2ConfusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        folder = tempfile.TemporaryDirectory(prefix='scbs-run2-confusion-test-')
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.output = self.root / '新诊断 output'
        self.checkpoint = self.root / 'best_stg2.pth'
        self.checkpoint.write_bytes(b'MOCK; not real checkpoint weights')
        self.annotation = self.root / 'instances_val.json'
        self.annotation.write_text('{}')
        self.calls = self.root / 'calls.jsonl'
        self.conda = self.root / 'fake-conda'
        self.conda.write_text('#!' + sys.executable + '\n' + FAKE_CONDA)
        self.conda.chmod(0o755)
        self.env = dict(os.environ, SCBS_RUN2_CONDA=str(self.conda), SCBS_RUN2_ENV='fixture',
                        SCBS_RUN2_CHECKPOINT=str(self.checkpoint), SCBS_RUN2_GPU='0',
                        RUN2_TEST_CALLS=str(self.calls), RUN2_TEST_ANNOTATION=str(self.annotation))
        for key in list(self.env):
            if key.startswith('RUN2_TEST_') and key not in ('RUN2_TEST_CALLS', 'RUN2_TEST_ANNOTATION'):
                self.env.pop(key)

    def run_workflow(self, *args):
        return subprocess.run(['bash', str(SCRIPT), str(self.output), *args], cwd='/tmp',
                              env=self.env, capture_output=True, text=True, timeout=35)

    def records(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def test_config_changes_only_export_and_model_state_is_identical(self):
        base, diag = fresh_config('dfine_s_scbs_hrw_110.yml'), fresh_config(CONFIG)
        expected, actual = copy.deepcopy(base.yaml_cfg), copy.deepcopy(diag.yaml_cfg)
        expected['export_confusion_matrix'] = True
        for config in (expected, actual):
            config.pop('__include__', None)
        self.assertEqual(expected, actual)
        self.assertFalse(actual.get('export_query_diagnostics', False))
        self.assertEqual(actual['epochs'], 110)
        self.assertEqual(actual['train_dataloader']['collate_fn']['stop_epoch'], 100)
        torch.manual_seed(21); original = base.model
        torch.manual_seed(21); exported = diag.model
        self.assertEqual(set(original.state_dict()), set(exported.state_dict()))
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, exported.state_dict()[key], rtol=0, atol=0)
        self.assertFalse(any('query_cls_refiner' in key for key in exported.state_dict()))

    def test_bash_and_embedded_python_syntax(self):
        subprocess.run(['bash', '-n', str(SCRIPT)], check=True)
        codes = re.findall(r"<<'PY'\n(.*?)\nPY", SCRIPT.read_text(), flags=re.S)
        self.assertEqual(len(codes), 2)
        for code in codes:
            compile(code, str(SCRIPT), 'exec')
        self.assertIn('epochs110_seed0_run2', SCRIPT.read_text())

    def test_real_preflight_counts_valid_gt_and_rejects_other_weights(self):
        image_dir = self.root / 'images'; image_dir.mkdir()
        (image_dir/'fixture.jpg').write_bytes(b'file-existence fixture')
        dataset = dict(images=[dict(id=1, file_name='fixture.jpg', width=100, height=100)],
                       categories=[dict(id=i, name=n) for i,n in enumerate(('hand-raising','read','write'))],
                       annotations=[dict(id=i+1, image_id=1, category_id=i, bbox=[i*20,0,10,10]) for i in range(3)])
        dataset['annotations'] += [dict(id=4,image_id=1,category_id=2,bbox=[0,0,10,10],iscrowd=1),
                                   dict(id=5,image_id=1,category_id=1,bbox=[200,0,10,10])]
        self.annotation.write_text(json.dumps(dataset))
        state = {'decoder.dec_score_head.2.weight': torch.zeros(3,256)}
        checkpoint = dict(last_epoch=109, model=state, ema={'module': state})
        torch.save(checkpoint, self.checkpoint)
        code = re.findall(r"<<'PY'\n(.*?)\nPY", SCRIPT.read_text(), flags=re.S)[0]
        original_loader = load_config
        def fixture_config(path, cfg):
            result = original_loader(path, cfg=cfg)
            result['val_dataloader']['dataset'].update(img_folder=str(image_dir), ann_file=str(self.annotation))
            return result
        def check(output):
            with patch('src.core.yaml_utils.load_config', fixture_config), redirect_stdout(io.StringIO()), \
                 patch.object(sys, 'argv', ['-', str(ROOT/'configs/dfine'/CONFIG), str(self.checkpoint), str(output)]):
                exec(compile(code, '<real-run2-preflight>', 'exec'), {})
        check(self.output)
        snapshot = json.loads((self.output/'export_inputs.json').read_text())
        self.assertEqual(snapshot['GT'], 3)
        self.assertEqual(snapshot['GT_by_category'], {'0':1,'1':1,'2':1})
        self.assertEqual(snapshot['checkpoint_sha256'], hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())
        cases = [('aqs', dict(checkpoint, model=dict(state, **{'decoder.decoder.query_cls_refiner.threshold_logit': torch.tensor(0.)})), '这是AQS'),
                 ('132', dict(checkpoint, last_epoch=131), '不是110轮'),
                 ('noema', dict(last_epoch=109, model=state), '缺少EMA')]
        for name, value, message in cases:
            with self.subTest(name=name):
                torch.save(value, self.checkpoint)
                output = self.root/name
                with self.assertRaisesRegex(AssertionError, message):
                    check(output)
                self.assertFalse(output.exists())

    def test_mocked_workflow_only_evaluates_and_logs_without_mutating_weights(self):
        before = self.checkpoint.read_bytes()
        result = self.run_workflow()
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        commands = self.records()
        self.assertEqual(len(commands), 3)
        evaluation = commands[1]
        self.assertEqual(evaluation[0], 'torchrun')
        for argument in ('--test-only', '--standalone', '--local-addr=127.0.0.1', '--use-amp', '--seed=0'):
            self.assertIn(argument, evaluation)
        self.assertNotIn('--master_port', evaluation)
        self.assertNotIn('-t', evaluation)
        self.assertEqual(evaluation[evaluation.index('-r')+1], str(self.checkpoint))
        self.assertEqual(self.checkpoint.read_bytes(), before)
        self.assertIn('SIMULATED LIVE', (self.output/'logs/evaluation.log').read_text())
        self.assertIn('SIMULATED LIVE', result.stdout)
        self.assertTrue(json.loads((self.output/'workflow_summary.json').read_text())['complete'])
        self.assertFalse((self.output/'query_diagnostics').exists())

    def test_existing_results_never_overwritten(self):
        self.output.mkdir(); marker=self.output/'keep.txt'; marker.write_text('keep')
        result=self.run_workflow()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('输出目录已存在', result.stderr)
        self.assertEqual(marker.read_text(), 'keep')
        self.assertFalse(self.calls.exists())

    def test_missing_checkpoint_and_preflight_failure_do_not_create_output(self):
        self.env['SCBS_RUN2_CHECKPOINT'] = str(self.root/'missing.pth')
        result=self.run_workflow()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('缺少110轮基线Run2权重', result.stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.calls.exists())
        self.env['SCBS_RUN2_CHECKPOINT'] = str(self.checkpoint)
        self.env['RUN2_TEST_PREFLIGHT_FAIL']='1'
        result=self.run_workflow()
        self.assertEqual(result.returncode, 19)
        self.assertFalse(self.output.exists())

    def test_evaluation_failure_is_not_reported_as_success(self):
        self.env['RUN2_TEST_EVAL_FAIL']='1'
        result=self.run_workflow()
        self.assertEqual(result.returncode, 17)
        self.assertFalse((self.output/'workflow_summary.json').exists())
        self.assertEqual(len(self.records()), 2)
        self.assertIn('退出码=17', result.stderr)
        self.assertTrue((self.output/'logs/evaluation.log').is_file())

    def test_incomplete_or_mismatched_export_never_marked_complete(self):
        for key in ('RUN2_TEST_INCOMPLETE','RUN2_TEST_BAD_SOURCE','RUN2_TEST_BAD_THRESHOLD',
                    'RUN2_TEST_BAD_COUNT','RUN2_TEST_CHECKPOINT_CHANGED'):
            with self.subTest(key=key):
                self.env[key]='1'; self.output=self.root/key
                result=self.run_workflow()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.output/'workflow_summary.json').exists())
                self.env.pop(key)

    def test_invalid_gpu_and_extra_arguments_rejected(self):
        self.env['SCBS_RUN2_GPU']='0,1'
        result=self.run_workflow()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('GPU必须', result.stderr)
        self.env['SCBS_RUN2_GPU']='0'
        result=self.run_workflow('extra')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('用法', result.stderr)
        self.assertFalse(self.calls.exists())

    def test_actual_export_preserves_original_class_blind_counts(self):
        gt=[dict(boxes=torch.tensor([[0.,0.,10.,10.],[20.,0.,30.,10.],[40.,0.,50.,10.]]), labels=torch.tensor([0,1,2]))]
        preds=[dict(boxes=torch.tensor([[0.,0.,10.,10.],[20.,0.,30.,10.],[40.,0.,50.,10.],[100.,0.,110.,10.]]),
                    labels=torch.tensor([0,2,1,1]), scores=torch.tensor([.9,.8,.7,.6]))]
        before=copy.deepcopy(preds)
        validator=Validator(gt,preds)
        validator.compute_metrics(extended=True)
        coco=SimpleNamespace(dataset={'categories':[dict(id=i,name=n) for i,n in enumerate(('hand-raising','read','write'))]})
        with redirect_stdout(io.StringIO()):
            report=export_confusion_matrix(validator,coco,self.root)
        self.assertEqual(report['matrix_counts'], [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,1,0,0]])
        self.assertEqual(report['misclassifications'], 2)
        self.assertEqual(report['confidence_threshold'], .5)
        self.assertEqual(report['iou_threshold'], .5)
        for key in preds[0]:
            torch.testing.assert_close(before[0][key],preds[0][key],rtol=0,atol=0)
        for name in ('confusion_matrix_counts.png','confusion_matrix_row_normalized.png','confusion_matrix.json','confusion_matrix.txt'):
            self.assertGreater((self.root/'confusion_matrix'/name).stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
