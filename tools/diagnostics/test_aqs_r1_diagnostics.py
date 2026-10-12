"""CPU config checks and MOCKED workflow tests; never train or use real CUDA."""
import copy
import hashlib
import io
import json
import os
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
from tools.diagnostics.test_aqs_refine import fresh_config
from src.core.yaml_utils import load_config

SCRIPT = ROOT / 'tools/diagnostics/run_scbs_aqs_r1_diagnostics.sh'
TRAIN = 'dfine_s_scbs_hrw_110_aqs_layer2_r010.yml'
DIAG = 'dfine_s_scbs_hrw_110_aqs_layer2_r010_diagnostics.yml'

# Mock only subprocess launchers; final completeness/hash check runs as real Python.
# This executable never calls torchrun, trains, or reads server-side datasets.
FAKE_CONDA = r'''
import hashlib, json, os, sys
from pathlib import Path
args = sys.argv[1:]
assert args[:3] == ['run', '--no-capture-output', '-n'], args
cmd = args[4:]
with Path(os.environ['AQS_DIAG_TEST_CALLS']).open('a') as h:
    h.write(json.dumps(cmd) + '\n')
if cmd[:2] == ['python', '-']:
    code = sys.stdin.read()
    if len(cmd) == 5:  # Config/config/checkpoint preflight: mocked server inputs.
        sys.exit(int(os.environ.get('AQS_DIAG_TEST_PREFLIGHT_EXIT', '0')))
    sys.argv = cmd[1:]
    exec(compile(code, '<real-final-workflow-verification>', 'exec'))
elif cmd[0] == 'torchrun':
    if os.environ.get('AQS_DIAG_TEST_EVAL_FAIL') == '1':
        print('SIMULATED EVALUATION FAILURE', flush=True)
        sys.exit(17)
    out = Path(cmd[cmd.index('--output-dir')+1])
    checkpoint = Path(cmd[cmd.index('-r')+1])
    (out/'query_diagnostics').mkdir(parents=True)
    metadata = dict(complete=os.environ.get('AQS_DIAG_TEST_INCOMPLETE') != '1',
                    checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    images=2, queries=600)
    (out/'query_diagnostics/metadata.json').write_text(json.dumps(metadata))
    print('SIMULATED VALIDATION LOG; NO TRAINING', flush=True)
elif 'analyze_detection_diagnostics.py' in cmd[1]:
    output = Path(cmd[cmd.index('--output-dir')+1]); output.mkdir()
    checkpoint = Path(os.environ['AQS_DIAG_CHECKPOINT'])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest, weight_source='ema',
                  images=2, queries=600, GT=5)
    if os.environ.get('AQS_DIAG_TEST_BAD_HASH') == '1':
        report['checkpoint_sha256'] = 'wrong'
    (output/'summary.json').write_text(json.dumps(report))
    (output/'summary.txt').write_text('SIMULATED SUMMARY')
elif 'export_aqs_parameters.py' in cmd[1]:
    output = Path(cmd[cmd.index('--output')+1])
    report = dict(checkpoint=cmd[cmd.index('--checkpoint')+1],
                  sources={'ema': {'decoder.decoder.query_cls_refiner': {'temperature': .1}}})
    output.write_text(json.dumps(report))
else:
    raise RuntimeError(cmd)
'''


class AQSR1DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='dfine-r1-diag-test-')
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.output = self.root / '诊断 output'
        self.checkpoint = self.root / 'best_stg2.pth'
        self.checkpoint.write_bytes(b'MOCK checkpoint; NOT actual model weights')
        self.calls = self.root / 'calls.jsonl'
        self.conda = self.root / 'fake-conda'
        self.conda.write_text('#!' + sys.executable + '\n' + FAKE_CONDA)
        self.conda.chmod(0o755)
        self.env = dict(os.environ, AQS_DIAG_CONDA=str(self.conda), AQS_DIAG_ENV='fake',
                        AQS_DIAG_CHECKPOINT=str(self.checkpoint), AQS_DIAG_GPU='0',
                        AQS_DIAG_TEST_CALLS=str(self.calls))
        for key in ('AQS_DIAG_TEST_PREFLIGHT_EXIT', 'AQS_DIAG_TEST_EVAL_FAIL',
                    'AQS_DIAG_TEST_INCOMPLETE', 'AQS_DIAG_TEST_BAD_HASH'):
            self.env.pop(key, None)

    def run_workflow(self, *args):
        return subprocess.run(['bash', str(SCRIPT), str(self.output), *args],
                              cwd=ROOT, env=self.env, capture_output=True, text=True, timeout=35)

    def records(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def test_config_only_adds_exports_and_keeps_r1_model(self):
        base, diag = fresh_config(TRAIN), fresh_config(DIAG)
        expected, actual = copy.deepcopy(base.yaml_cfg), copy.deepcopy(diag.yaml_cfg)
        expected.update(export_confusion_matrix=True, export_query_diagnostics=True,
                        query_diagnostic_conf_thresh=.5, query_diagnostic_iou_thresh=.5,
                        query_diagnostic_neighbor_iou_thresh=.8)
        for cfg in (expected, actual):
            cfg.pop('__include__', None)
        self.assertEqual(actual, expected)
        torch.manual_seed(15); original = base.model
        torch.manual_seed(15); model = diag.model
        for key, value in original.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        decoder = model.decoder.decoder
        self.assertEqual(decoder.aqs_layer_indices, (1,))
        self.assertEqual(len(decoder.aqs_refiners), 0)
        self.assertAlmostEqual(decoder.query_cls_refiner.threshold_logit.sigmoid().item(), .5)
        self.assertAlmostEqual(decoder.query_cls_refiner.residual_scale.item(), .1)
        self.assertEqual(decoder.query_cls_refiner.temperature, .1)
        self.assertEqual(diag.yaml_cfg['epochs'], 110)

    def test_shell_syntax(self):
        self.assertEqual(subprocess.run(['bash', '-n', str(SCRIPT)]).returncode, 0)

    def test_real_preflight_checks_config_validation_data_and_checkpoint(self):
        image_dir = self.root/'images'; image_dir.mkdir()
        (image_dir/'fixture.jpg').write_bytes(b'file-existence fixture; not a real image')
        annotation = self.root/'annotations.json'
        annotation.write_text(json.dumps(dict(images=[{'file_name': 'fixture.jpg'}],
            categories=[{'id': i, 'name': name} for i, name in enumerate(('hand-raising', 'read', 'write'))])))
        state = {'decoder.decoder.query_cls_refiner.threshold_logit': torch.tensor(0.),
                 'decoder.dec_score_head.2.weight': torch.zeros(3, 256)}
        torch.save({'last_epoch': 109, 'model': state}, self.checkpoint)
        # Execute the production preflight, only redirecting fixture dataset paths.
        code = SCRIPT.read_text().split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        def fixture_config(path, cfg):
            result = load_config(path, cfg=cfg)
            result['val_dataloader']['dataset'].update(img_folder=str(image_dir), ann_file=str(annotation))
            return result
        def check():
            with patch('src.core.yaml_utils.load_config', fixture_config), redirect_stdout(io.StringIO()), \
                 patch.object(sys, 'argv', ['-', str(ROOT/'configs/dfine'/DIAG),
                                           str(ROOT/'configs/dfine'/TRAIN), str(self.checkpoint)]):
                exec(compile(code, '<actual-R1-preflight>', 'exec'), {})
        check()
        torch.save({'last_epoch': 109, 'model': {}}, self.checkpoint)
        with self.assertRaisesRegex(AssertionError, '不是单层AQS'):
            check()
        torch.save({'last_epoch': 131, 'model': state}, self.checkpoint)
        with self.assertRaisesRegex(AssertionError, '不是110轮'):
            check()

    def test_only_validation_and_analysis_with_same_checkpoint_no_mutation(self):
        digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        result = self.run_workflow()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.records()
        self.assertEqual(len(calls), 5)
        evaluation = calls[1]
        self.assertEqual(evaluation[0], 'torchrun')
        self.assertIn('--test-only', evaluation)
        self.assertIn('--standalone', evaluation)
        self.assertIn('--local-addr=127.0.0.1', evaluation)
        self.assertNotIn('-t', evaluation)
        self.assertEqual(evaluation[evaluation.index('-r')+1], str(self.checkpoint))
        self.assertTrue(evaluation[evaluation.index('-c')+1].endswith(DIAG))
        self.assertEqual(calls[3][calls[3].index('--temperature')+1], '0.10')
        self.assertIn('SIMULATED VALIDATION LOG', (self.output/'logs/evaluation.log').read_text())
        report = json.loads((self.output/'workflow_summary.json').read_text())
        self.assertTrue(report['complete'])
        self.assertEqual(report['checkpoint_sha256'], digest)
        self.assertEqual(report['decoder_layer'], 2)
        self.assertEqual(hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(), digest)

    def test_existing_output_is_preserved_without_subprocesses(self):
        self.output.mkdir(); marker = self.output/'user.txt'; marker.write_text('keep')
        result = self.run_workflow()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_text(), 'keep')
        self.assertEqual(self.records(), [])

    def test_missing_checkpoint_creates_no_output(self):
        self.env['AQS_DIAG_CHECKPOINT'] = str(self.root/'missing.pth')
        self.assertNotEqual(self.run_workflow().returncode, 0)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.records(), [])

    def test_preflight_failure_creates_no_output(self):
        self.env['AQS_DIAG_TEST_PREFLIGHT_EXIT'] = '19'
        self.assertEqual(self.run_workflow().returncode, 19)
        self.assertFalse(self.output.exists())

    def test_evaluation_failure_stops_and_preserves_log(self):
        self.env['AQS_DIAG_TEST_EVAL_FAIL'] = '1'
        result = self.run_workflow()
        self.assertEqual(result.returncode, 17)
        self.assertEqual(len(self.records()), 2)
        self.assertTrue((self.output/'logs/evaluation.log').is_file())
        self.assertFalse((self.output/'workflow_summary.json').exists())

    def test_incomplete_or_different_hash_never_marks_complete(self):
        for setting in ('AQS_DIAG_TEST_INCOMPLETE', 'AQS_DIAG_TEST_BAD_HASH'):
            with self.subTest(setting=setting):
                self.output = self.root/setting
                self.env[setting] = '1'
                result = self.run_workflow()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.output/'workflow_summary.json').exists())
                self.env.pop(setting)

    def test_invalid_gpu_and_extra_arguments_are_rejected(self):
        self.env['AQS_DIAG_GPU'] = '0,1'
        self.assertNotEqual(self.run_workflow().returncode, 0)
        self.env['AQS_DIAG_GPU'] = '0'
        self.assertNotEqual(self.run_workflow('/tmp/extra-argument').returncode, 0)
        self.assertEqual(self.records(), [])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
