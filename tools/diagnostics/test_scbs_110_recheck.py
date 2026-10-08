"""Synthetic CPU-only tests; no real checkpoint, GPU, downloads or training."""
import copy
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from src.core.yaml_utils import load_config
from tools.diagnostics.gt_box_classifier import load_config as load_probe_config
from tools.diagnostics.analyze_detection_diagnostics import (
    analyze, best_query_evidence, score_first_match, summarize_best_query_evidence,
)
from tools.diagnostics.scbs_encoder_query_coverage import validate_config
from tools.diagnostics.test_detection_diagnostics import fixture

LAUNCHER = ROOT / 'tools/diagnostics/run_scbs_110_run1_diagnostics.sh'


class SCBS110RecheckTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def evidence(self, ious, scores, selected):
        return best_query_evidence(torch.tensor(ious), torch.tensor(scores), 1,
                                   torch.tensor(selected, dtype=torch.long), .5, .5)

    def test_new_config_is_110_only_and_old_132_is_preserved(self):
        new = load_probe_config(ROOT / 'configs/diagnostics/scbs_encoder_query_coverage_110_run1.yml')
        old = load_probe_config(ROOT / 'configs/diagnostics/scbs_encoder_query_coverage_run1.yml')
        validate_config(new)
        self.assertEqual(new['detector_config'], 'configs/dfine/dfine_s_scbs_hrw_110.yml')
        self.assertIn('epochs110_seed0_run1/best_stg2.pth', new['checkpoint'])
        expected = copy.deepcopy(old)
        expected.update(detector_config=new['detector_config'], checkpoint=new['checkpoint'])
        self.assertEqual(expected, new)
        self.assertIn('_132.yml', old['detector_config'])
        self.assertNotEqual(new['checkpoint'], old['checkpoint'])
        base = load_config(str(ROOT / new['detector_config']), cfg={})
        diag = load_config(str(ROOT / 'configs/dfine/dfine_s_scbs_hrw_110_diagnostics.yml'), cfg={})
        changed = {k for k in set(base) | set(diag) if base.get(k) != diag.get(k)}
        self.assertEqual(changed, {'__include__', 'export_confusion_matrix', 'export_query_diagnostics',
                                  'query_diagnostic_conf_thresh', 'query_diagnostic_iou_thresh',
                                  'query_diagnostic_neighbor_iou_thresh'})
        self.assertEqual(diag['epochs'], 110)
        self.assertEqual(diag['train_dataloader']['collate_fn']['stop_epoch'], 100)
        self.assertEqual(diag['train_dataloader']['dataset']['transforms']['policy']['epoch'], 100)
        for split in ('train', 'val'):
            self.assertEqual(new['data'][split]['annotation'], diag[f'{split}_dataloader']['dataset']['ann_file'])
            self.assertEqual(new['data'][split]['image_dir'], diag[f'{split}_dataloader']['dataset']['img_folder'])

    def test_all_seven_best_query_types(self):
        cases = {
            'A': ([1., .7], [[.1, .3, .1], [.1, .8, .1]], [4]),
            'B': ([1.], [[.1, .3, .1]], [1]),
            'C': ([1., .7], [[.9, .1, .1], [.1, .8, .1]], [0, 4]),
            'D': ([1., .7], [[.9, .1, .1], [.1, .4, .1]], [0, 4]),
            'E': ([1.], [[.1, .8, .1]], [0]),
            'F': ([1.], [[.1, .8, .1]], [1]),
            'G': ([.49], [[.1, .8, .1]], [1]),
        }
        for kind, args in cases.items():
            with self.subTest(kind=kind):
                row = self.evidence(*args)
                self.assertEqual(row['best_query_evidence_type'], kind)
                self.assertEqual(row['high_correct_selected_candidate_exists'], kind in ('A', 'C', 'F'))
        self.assertTrue(self.evidence(*cases['A'])['another_high_correct_selected_query_exists'])
        self.assertFalse(self.evidence(*cases['F'])['another_high_correct_selected_query_exists'])

    def test_flattened_secondary_class_is_not_called_a_classification_gap(self):
        row = self.evidence([1.], [[.9, .8, .1]], [0, 1])
        self.assertEqual(row['best_query_evidence_type'], 'C')
        self.assertTrue(row['best_query_GT_pair_high_and_selected'])
        self.assertTrue(row['high_correct_selected_candidate_exists'])
        self.assertFalse(row['another_high_correct_selected_query_exists'])
        self.assertFalse(row['any_overlapping_GT_top1_query_exists'])
        self.assertEqual(row['candidate_gap_reason'], 'none')

    def test_confidence_geometry_and_topk_are_distinct(self):
        self.assertEqual(self.evidence([.5], [[.1, .5, .1]], [1])['best_query_evidence_type'], 'F')
        low = self.evidence([1.], [[.1, .49, .1]], [1])
        dropped = self.evidence([1., .7], [[.1, .3, .1], [.1, .8, .1]], [1])
        geometry = self.evidence([.49], [[.1, .9, .1]], [1])
        self.assertEqual(low['candidate_gap_reason'], 'all_overlapping_GT_class_scores_below_confidence')
        self.assertEqual(dropped['best_query_evidence_type'], 'B')
        self.assertEqual(dropped['candidate_gap_reason'], 'high_GT_class_pair_not_selected_by_topk')
        self.assertEqual(geometry['candidate_gap_reason'], 'no_geometric_overlap')

    def test_candidate_existence_is_not_one_to_one_recall(self):
        scores = torch.tensor([[.1, .9, .1]])
        ious = torch.ones(1, 2)
        matches = score_first_match(ious, scores[:, 1], .5, torch.tensor([1]), torch.tensor([1, 1]))
        rows = []
        for g in range(2):
            row = best_query_evidence(ious[:, g], scores, 1, torch.tensor([1]), .5, .5)
            row.update(gt_class_id=1, area_original_annotation=2000., ignore=False,
                       class_aware_score_first_IoU05_high_matched=g in matches.tolist())
            rows.append(row)
        result = summarize_best_query_evidence(rows, {0: 'hand-raising', 1: 'read', 2: 'write'},
                                              {'medium': [1024., 9216.]}, .5, .5)
        self.assertEqual(result['all']['GT_count'], 2)
        self.assertEqual(result['all']['high_correct_selected_candidate_exists'], 2)
        self.assertEqual(result['all']['class_aware_score_first_IoU05_high_matched'], 1)
        self.assertEqual(result['all']['candidate_exists_but_not_one_to_one_high_matched'], 1)
        self.assertEqual(result['by_size_and_class']['medium']['read']['GT_count'], 2)
        self.assertEqual(result['by_class']['write']['GT_count'], 0)

    def test_end_to_end_existing_query_exports_without_changes_to_predictions(self):
        with tempfile.TemporaryDirectory(prefix='scbs-110-recheck-') as folder:
            root = Path(folder)
            annotation, _, _, _ = fixture(root, empty_second=True, dtype=torch.bfloat16)
            raw_path = root / 'query_diagnostics/raw_queries/1.pt'
            before = raw_path.read_bytes()
            summary = analyze(root, annotation)
            self.assertEqual(raw_path.read_bytes(), before)
            cross = summary['best_query_cross_classification']
            self.assertEqual(cross['all']['GT_count'], 5)
            self.assertEqual(cross['all']['type_counts'], dict(A=0, B=1, C=0, D=1, E=0, F=3, G=0))
            self.assertEqual(cross['by_class']['write']['no_high_correct_selected_candidate'], 1)
            rows = [json.loads(l) for l in (root / 'analysis/best_query_cross_classification.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 5)
            self.assertEqual(sum(r['class_aware_score_first_IoU05_high_matched'] for r in rows), 3)
            self.assertEqual(json.loads((root / 'analysis/best_query_cross_classification.json').read_text()), cross)
            with self.assertRaises(FileExistsError):
                analyze(root, annotation)

    def test_launcher_is_test_only_port_safe_and_keeps_existing_results(self):
        source = LAUNCHER.read_text()
        self.assertIn('--test-only', source)
        self.assertIn('--standalone --nnodes=1 --nproc_per_node=1', source)
        self.assertNotIn('--master_port', source)
        subprocess.run(['bash', '-n', str(LAUNCHER)], check=True)
        scripts = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.S)
        self.assertEqual(len(scripts), 2)
        for script in scripts:
            compile(script, str(LAUNCHER), 'exec')
        with tempfile.TemporaryDirectory(prefix='scbs-110-existing-') as folder:
            root = Path(folder)
            marker = root / 'keep.txt'
            marker.write_text('keep')
            result = subprocess.run(['bash', str(LAUNCHER), str(root)], cwd='/tmp',
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('输出目录已存在', result.stderr)
            self.assertEqual(marker.read_text(), 'keep')

    def test_workflow_checks_provenance_and_merges_readable_reports(self):
        finalizer = re.findall(r"<<'PY'\n(.*?)\nPY", LAUNCHER.read_text(), flags=re.S)[1]
        with tempfile.TemporaryDirectory(prefix='scbs-110-manifest-') as folder:
            root = Path(folder)
            analysis = dict(checkpoint='110.pth', checkpoint_sha256='a' * 64, weight_source='ema', GT=5)
            coverage = dict(complete=True, checkpoint_sha256='a' * 64, weight_source='ema',
                            splits={'val': {'coverage': {'all': {'gt_count': 5}}}})
            for path, value in (('analysis/summary.json', analysis), ('encoder_query_coverage/summary.json', coverage),
                                ('evaluation/query_diagnostics/metadata.json', {'complete': True})):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(value))
            (root / 'analysis/summary.txt').write_text('分类/分数检查\n')
            (root / 'encoder_query_coverage/summary.txt').write_text('Top300覆盖检查\n')
            result = subprocess.run([sys.executable, '-', str(root)], input=finalizer, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads((root / 'workflow_summary.json').read_text())['complete'])
            self.assertIn('Top300覆盖检查', (root / '110_run1_recheck_summary.txt').read_text())
            coverage['checkpoint_sha256'] = 'b' * 64
            (root / 'encoder_query_coverage/summary.json').write_text(json.dumps(coverage))
            result = subprocess.run([sys.executable, '-', str(root)], input=finalizer, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('两个步骤不是同一权重', result.stderr)


if __name__ == '__main__':
    unittest.main()
