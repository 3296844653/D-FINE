"""CPU and simulated-launcher regressions; no real training or GPU job."""
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core.yaml_utils import load_config
from tools.diagnostics.analyze_detection_diagnostics import analyze, load_coco_pr
from tools.diagnostics.test_detection_diagnostics import fixture
from tools.diagnostics.test_rw_local_detail import fresh_config

SCRIPT = ROOT / 'tools/diagnostics/run_scbs_rw_local_detail_diagnostics.sh'
CONFIG = 'dfine_s_scbs_hrw_110_rw_local_detail_diagnostics.yml'

# Use the existing launcher fixture and execute the NEW production finalizer.
# Validation/analysis below are simulated; actual tensor aggregation is tested
# separately against real COCOeval output and independently constructed axes.
from tools.diagnostics.test_scbs_110_run2_diagnostics import FAKE_CONDA

FAKE_RW_CONDA = FAKE_CONDA.replace(
    "(root/'summary.json').write_text(json.dumps(summary))",
    """size_metrics = {str(c): {area: dict(AP=.3, AP50=.5, AP75=.4, AR100=.7)
                    for area in ('all','small','medium','large')} for c in range(3)}
    if os.environ.get('RW_DIAG_TEST_BAD_SIZE') == '1':
        size_metrics['2']['medium']['AP'] = 1.1
    summary['saved_COCO_per_class_and_size'] = size_metrics
    (root/'coco_per_class_and_size.json').write_text(json.dumps({'metrics':size_metrics}))
    if os.environ.get('RW_DIAG_TEST_SIZE_MISMATCH') == '1':
        summary['saved_COCO_per_class_and_size'] = {}
    (root/'summary.json').write_text(json.dumps(summary))""",
)


class RWLocalDetailDiagnosticLauncherTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        folder = tempfile.TemporaryDirectory(prefix='rw-detail-diagnostics-')
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.output = self.root / '局部诊断 output'
        self.checkpoint = self.root / 'best_stg2.pth'
        self.checkpoint.write_bytes(b'MOCK checkpoint; not real weights')
        self.annotation = self.root / 'instances_val.json'
        self.annotation.write_text('{}')
        self.calls = self.root / 'calls.jsonl'
        self.conda = self.root / 'fake-conda'
        self.conda.write_text('#!' + sys.executable + '\n' + FAKE_RW_CONDA)
        self.conda.chmod(0o755)
        self.env = dict(os.environ, RW_DETAIL_DIAG_CONDA=str(self.conda),
                        RW_DETAIL_DIAG_ENV='fixture', RW_DETAIL_DIAG_CHECKPOINT=str(self.checkpoint),
                        RW_DETAIL_DIAG_GPU='0', RUN2_DIAG_TEST_CALLS=str(self.calls),
                        RUN2_DIAG_TEST_ANNOTATION=str(self.annotation))
        for key in list(self.env):
            if key.startswith(('RUN2_DIAG_TEST_', 'RW_DIAG_TEST_')) and key not in (
                    'RUN2_DIAG_TEST_CALLS', 'RUN2_DIAG_TEST_ANNOTATION'):
                self.env.pop(key)

    def launch(self, *extra):
        return subprocess.run(['bash', str(SCRIPT), str(self.output), *extra],
                              cwd='/tmp', env=self.env, text=True, capture_output=True, timeout=35)

    def records(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def test_syntax_defaults_and_config_preserves_training(self):
        subprocess.run(['bash', '-n', str(SCRIPT)], check=True)
        codes = re.findall(r"<<'PY'\n(.*?)\nPY", SCRIPT.read_text(), re.S)
        self.assertEqual(len(codes), 2)
        for code in codes:
            compile(code, str(SCRIPT), 'exec')
        self.assertIn('epochs110_rw_local_detail_seed0_run1', SCRIPT.read_text())
        expected = load_config(str(ROOT / 'configs/dfine/dfine_s_scbs_hrw_110_rw_local_detail.yml'), cfg={})
        actual = load_config(str(ROOT / 'configs/dfine' / CONFIG), cfg={})
        expected.update(export_confusion_matrix=True, export_query_diagnostics=True,
                        query_diagnostic_conf_thresh=.5, query_diagnostic_iou_thresh=.5,
                        query_diagnostic_neighbor_iou_thresh=.8)
        for config in (expected, actual):
            config.pop('__include__', None)
        self.assertEqual(expected, actual)

    def test_real_preflight_validates_own_weights_dataset_and_config(self):
        images = self.root / 'images'; images.mkdir()
        (images / '1.jpg').write_bytes(b'existence fixture')
        dataset = dict(images=[dict(id=1, file_name='1.jpg', width=100, height=100)],
                       categories=[dict(id=c, name=n) for c, n in enumerate(('hand-raising', 'read', 'write'))],
                       annotations=[dict(id=c+1, image_id=1, category_id=c, bbox=[20*c, 0, 10, 10]) for c in range(3)])
        dataset['annotations'] += [dict(id=4, image_id=1, category_id=1, bbox=[200, 0, 10, 10]),
                                   dict(id=5, image_id=1, category_id=2, bbox=[0, 0, 10, 10], iscrowd=1)]
        self.annotation.write_text(json.dumps(dataset))
        state = fresh_config(CONFIG, size=640).model.state_dict()
        checkpoint = dict(last_epoch=102, model=state, ema={'module': state})
        torch.save(checkpoint, self.checkpoint)
        code = re.findall(r"<<'PY'\n(.*?)\nPY", SCRIPT.read_text(), re.S)[0]

        def check(output, changed_config=False):
            def fixture_config(path, cfg):
                # Patch recursive reads only at the top-level: load_config uses
                # a mutable default, so always start each config with cfg={}.
                with patch('src.core.yaml_utils.load_config', load_config):
                    config = load_config(path, cfg=cfg)
                config['val_dataloader']['dataset'].update(img_folder=str(images), ann_file=str(self.annotation))
                if changed_config and Path(path).name == CONFIG:
                    config['DFINETransformer']['rw_detail_residual_scale'] = .2
                return config

            with patch('src.core.yaml_utils.load_config', fixture_config), \
                 patch('src.core.yaml_config.load_config', lambda path: fixture_config(path, cfg={})), \
                 patch.object(sys, 'argv', ['-', str(ROOT / 'configs/dfine' / CONFIG), str(self.checkpoint), str(output)]), \
                 redirect_stdout(io.StringIO()):
                exec(compile(code, '<actual-rw-preflight>', 'exec'), {})

        check(self.output)
        inputs = json.loads((self.output / 'export_inputs.json').read_text())
        self.assertEqual(inputs['GT_by_category'], {'0': 1, '1': 1, '2': 1})
        self.assertEqual(inputs['checkpoint_epoch'], 102)
        self.assertEqual(inputs['module'], 'RWLocalDetailAdapter')
        self.assertEqual(inputs['module_state_statistics']['ema']['final_projection_weight_l2'], 0.)
        with self.assertRaisesRegex(AssertionError, '配置'):
            check(self.root / 'changed-config', changed_config=True)
        self.assertFalse((self.root / 'changed-config').exists())
        no_branch = {k: v for k, v in state.items() if not k.startswith('decoder.rw_local_detail.')}
        for name, value, error in (
                ('wrong-module', dict(checkpoint, model=no_branch), '不是局部细节'),
                ('no-ema', dict(last_epoch=102, model=state), '缺少model或EMA'),
                ('132-epoch', dict(checkpoint, last_epoch=127), '不是110轮'),
                ('wrong-shape', dict(checkpoint, ema={'module':dict(state, **{'decoder.rw_local_detail.fuse.up.weight':torch.zeros(1, 1)})}), 'size mismatch'),
                ('nonfinite', dict(checkpoint, model=dict(state, **{'decoder.rw_local_detail.fuse.up.weight':torch.full_like(state['decoder.rw_local_detail.fuse.up.weight'], float('nan'))})), '非有限值'),
        ):
            with self.subTest(case=name):
                torch.save(value, self.checkpoint)
                with self.assertRaisesRegex((AssertionError, RuntimeError), error):
                    check(self.root / name)
                self.assertFalse((self.root / name).exists())

    def test_workflow_validates_only_and_preserves_checkpoint(self):
        original = self.checkpoint.read_bytes()
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.records(); self.assertEqual(len(calls), 4)
        validation = calls[1]
        for option in ('--test-only', '--standalone', '--local-addr=127.0.0.1', '--use-amp', '--seed=0'):
            self.assertIn(option, validation)
        self.assertNotIn('-t', validation)
        self.assertEqual(validation[validation.index('-r')+1], str(self.checkpoint))
        self.assertTrue(validation[validation.index('-c')+1].endswith(CONFIG))
        self.assertEqual(self.checkpoint.read_bytes(), original)
        self.assertIn('SIMULATED LIVE', (self.output / 'logs/evaluation.log').read_text())
        self.assertIn('SIMULATED ANALYSIS', (self.output / 'logs/analysis.log').read_text())
        self.assertTrue(json.loads((self.output / 'workflow_summary.json').read_text())['complete'])
        self.assertIn('RW Local Detail', (self.output / 'rw_local_detail_diagnostics_summary.txt').read_text())

    def test_existing_outputs_and_invalid_arguments_stop_without_launching(self):
        self.output.mkdir(); (self.output / 'keep').write_text('keep')
        self.assertNotEqual(self.launch().returncode, 0)
        self.assertEqual((self.output / 'keep').read_text(), 'keep')
        self.output = self.root / 'dangling'; self.output.symlink_to(self.root / 'absent')
        self.assertNotEqual(self.launch().returncode, 0)
        self.assertTrue(self.output.is_symlink())
        self.output = self.root / 'new'
        self.env['RW_DETAIL_DIAG_GPU'] = '0,1'
        self.assertNotEqual(self.launch().returncode, 0)
        self.env['RW_DETAIL_DIAG_GPU'] = '0'
        self.assertNotEqual(self.launch('/tmp/extra').returncode, 0)
        self.assertEqual(self.records(), [])

    def test_failures_keep_partial_logs_without_success_report(self):
        for variable, expected, calls in (('RUN2_DIAG_TEST_PREFLIGHT_FAIL', 19, 1),
                                         ('RUN2_DIAG_TEST_EVAL_FAIL', 17, 2),
                                         ('RUN2_DIAG_TEST_ANALYSIS_FAIL', 23, 3)):
            with self.subTest(variable=variable):
                self.output = self.root / variable; self.calls.unlink(missing_ok=True)
                self.env[variable] = '1'
                result = self.launch()
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                self.assertEqual(len(self.records()), calls)
                self.assertFalse((self.output / 'workflow_summary.json').exists())
                if calls > 1:
                    self.assertTrue((self.output / 'logs/evaluation.log').is_file())
                else:
                    self.assertFalse(self.output.exists())
                self.env.pop(variable)

    def test_finalizer_rejects_incomplete_mixed_or_changed_artifacts(self):
        for variable in ('RUN2_DIAG_TEST_INCOMPLETE', 'RUN2_DIAG_TEST_BAD_HASH',
                         'RUN2_DIAG_TEST_BAD_SOURCE', 'RUN2_DIAG_TEST_BAD_THRESHOLD',
                         'RUN2_DIAG_TEST_BAD_COUNT', 'RUN2_DIAG_TEST_MISSING_FILE',
                         'RUN2_DIAG_TEST_CHANGED_CHECKPOINT', 'RUN2_DIAG_TEST_CHANGED_ANNOTATION',
                         'RW_DIAG_TEST_BAD_SIZE', 'RW_DIAG_TEST_SIZE_MISMATCH'):
            with self.subTest(variable=variable):
                self.output = self.root / variable; self.env[variable] = '1'
                result = self.launch()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.output / 'workflow_summary.json').exists())
                self.env.pop(variable)


class SavedCOCOClassSizeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_class_size_axes_and_undefined_are_not_mixed(self):
        params = SimpleNamespace(catIds=[0, 1, 2], imgIds=[7], iouThrs=np.array([.5, .75]),
                                 recThrs=np.array([0., .5, 1.]),
                                 areaRngLbl=['all', 'small', 'medium', 'large'],
                                 maxDets=[1, 10, 100], areaRng=[[0, 1e10], [0, 1024], [1024, 9216], [9216, 1e10]])
        precision = np.full((2, 3, 3, 4, 3), .01)
        recall = np.full((2, 3, 4, 3), .02)
        for k in range(3):
            for a in range(4):
                for t in range(2):
                    precision[t, :, k, a, 2] = .2 + k*.1 + a*.02 + t*.01
                    recall[t, k, a, 2] = .6 + k*.05 + a*.02 + t*.01
        precision[:, :, 2, 1, :] = -1; recall[:, 2, 1, :] = -1
        with tempfile.TemporaryDirectory(prefix='rw-coco-size-') as folder:
            path = Path(folder) / 'eval.pth'
            saved = dict(params=params, precision=precision, scores=precision.copy(), recall=recall)
            torch.save(saved, path)
            report = load_coco_pr(path, [0, 1, 2], [7])
            for k in range(3):
                for a, area in enumerate(params.areaRngLbl):
                    actual = report['saved_COCO_per_class_and_size'][str(k)][area]
                    if k == 2 and area == 'small':
                        self.assertTrue(all(value is None for value in actual.values()))
                    else:
                        self.assertAlmostEqual(actual['AP'], .205 + k*.1 + a*.02)
                        self.assertAlmostEqual(actual['AP50'], .2 + k*.1 + a*.02)
                        self.assertAlmostEqual(actual['AP75'], .21 + k*.1 + a*.02)
                        self.assertAlmostEqual(actual['AR100'], .605 + k*.05 + a*.02)
                self.assertEqual(report['saved_COCO_per_class'][str(k)], report['saved_COCO_per_class_and_size'][str(k)]['all'])
            valid = precision[:, :, :, 2, 2]
            self.assertAlmostEqual(report['saved_COCO_metrics']['AP_medium'], valid[valid>=0].mean())
            torch.save(dict(saved, recall=recall[:, :, :3, :]), path)
            with self.assertRaisesRegex(ValueError, 'axes'):
                load_coco_pr(path, [0, 1, 2], [7])

    def test_real_coco_export_writes_class_size_json_and_readable_summary(self):
        with tempfile.TemporaryDirectory(prefix='rw-real-coco-') as folder:
            root = Path(folder)
            with redirect_stdout(io.StringIO()):
                annotation, _, _, evaluation = fixture(root)
                summary = analyze(root, annotation)
            actual = json.loads((root / 'analysis/coco_per_class_and_size.json').read_text())
            self.assertEqual(actual['metrics'], summary['saved_COCO_per_class_and_size'])
            self.assertIn('write | medium | GT=0 | AP=--', (root / 'analysis/summary.txt').read_text())
            for k in range(3):
                valid = np.asarray(evaluation['precision'])[:, :, k, 1, 2]
                self.assertAlmostEqual(actual['metrics'][str(k)]['small']['AP'], valid[valid>=0].mean())
                self.assertEqual(actual['metrics'][str(k)]['small'], actual['metrics'][str(k)]['all'])
                self.assertTrue(all(value is None for value in actual['metrics'][str(k)]['medium'].values()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
