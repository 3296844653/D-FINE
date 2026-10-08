"""Small CPU fixtures; never train/evaluate on the user's real dataset."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from PIL import Image
from src.core import YAMLConfig
from src.core.yaml_utils import load_config as load_detector_config
from src.solver.validator import Validator
from src.zoo.dfine.postprocessor import DFINEPostProcessor
from tools.diagnostics import scbs_feature_probe as audit
from tools.diagnostics.gt_box_classifier import load_config


class SCBSFeatureProbeTests(unittest.TestCase):
    def test_actual_topk_pairs_and_validator_parity_with_duplicates_and_empty_gt(self):
        outputs = {'pred_logits': torch.tensor([[[4., 3., -4.], [-4., -4., 2.]]]),
                   'pred_boxes': torch.tensor([[[.3, .3, .2, .2], [.7, .7, .2, .2]]])}
        post = DFINEPostProcessor(num_classes=3, num_top_queries=3)
        pairs = audit.selected_pairs(outputs, post, torch.tensor([[100, 100]]), .5)
        self.assertEqual(pairs['query_ids'].tolist(), [0, 0, 1])
        self.assertEqual(pairs['labels'].tolist(), [0, 1, 2])
        gt_boxes = pairs['boxes'][[0, 2]].clone(); gt_labels = torch.tensor([1, 2])
        ids, ious, matrix = audit.selected_assignment(pairs, gt_boxes, gt_labels, .5)
        validator = Validator([{'boxes': gt_boxes, 'labels': gt_labels}],
                              [{k: pairs[k] for k in ('boxes', 'labels', 'scores')}])
        validator.compute_metrics()
        self.assertEqual(matrix.tolist(), validator.conf_matrix.tolist())
        self.assertEqual(int((ids >= 0).sum()), 2)
        self.assertTrue((ious == 1).all())
        _, _, empty = audit.selected_assignment(pairs, torch.empty(0, 4), torch.empty(0, dtype=torch.long), .5)
        self.assertEqual(empty[3].tolist(), [1, 1, 1, 0])
        zero = audit.selected_pairs(outputs, post, torch.tensor([[100, 100]]), .99)
        self.assertEqual(len(zero['query_ids']), 0)
        _, _, no_pred = audit.selected_assignment(zero, gt_boxes, gt_labels, .5)
        self.assertEqual(no_pred[:, 3].tolist(), [0, 1, 1, 0])

    def test_low_query_scopes_partition_and_paired_metrics(self):
        valid = torch.ones(4, dtype=torch.bool)
        cache = dict(labels=torch.tensor([1, 2, 2, 0]), areas=torch.tensor([2000., 10000., 5000., 100.]),
            metadata={'confidence_threshold': .5}, groups={
                'iou': dict(valid=valid, query_ids=torch.arange(4), scores=torch.tensor([.1, .2, .3, .8])),
                'selected': dict(valid=torch.tensor([True, True, False, True]),
                                 query_ids=torch.tensor([5, 6, -1, 3]), original_labels=torch.tensor([1, 1, -1, 0]))})
        scopes = audit.group_scopes(cache, 'iou')
        self.assertEqual(int(scopes['low_score'].sum()), 3)
        partition = sum(scopes[key].int() for key in ('low_with_selected_correct', 'low_with_selected_wrong',
                                                       'low_without_selected_match'))
        self.assertTrue(torch.equal(partition, scopes['low_score'].int()))
        self.assertEqual(int(scopes['same_query_as_selected'].sum()), 1)
        report = audit.classify_metrics(torch.tensor([0, 1, 2]), torch.tensor([0, 2, 2]),
            torch.tensor([1, 1, 2]), torch.ones(3, dtype=torch.bool), [c[1] for c in audit.CATEGORIES])
        self.assertEqual((report['fixed'], report['broken'], report['net_fixed']), (1, 1, 0))
        self.assertEqual(report['read_write_errors'], 1)
        empty = audit.classify_metrics(torch.tensor([0]), torch.tensor([0]), torch.tensor([0]),
                                       torch.tensor([False]), [c[1] for c in audit.CATEGORIES])
        self.assertIsNone(empty['accuracy'])

    def test_roi_geometry_and_equal_parameter_controls(self):
        feature = torch.arange(256.).reshape(1, 1, 16, 16).repeat(1, 256, 1, 1)
        boxes = torch.tensor([[0., 0., .5, .5], [.5, .5, 1., 1.]])
        pooled = audit.pool_roi(feature, boxes, 3)
        self.assertEqual(pooled.shape, (2, 256))
        self.assertGreater(pooled[1].mean(), pooled[0].mean())
        self.assertEqual(audit.pool_roi(feature, torch.empty(0, 4), 3).shape, (0, 256))
        query = torch.randn(2, 256)
        cache = {'maps': {'encoder_p3': pooled, 'encoder_p4': pooled * 2},
                 'groups': {'selected': {'query': query, 'pred_p3': pooled + 1, 'pred_p4': pooled + 2}}}
        dims = [audit.probe_features(cache, 'selected', key).shape[-1] for key in (
            'query_only_capacity_control', 'query_plus_pred_roi', 'query_plus_gt_roi')]
        self.assertEqual(dims, [768, 768, 768])
        self.assertFalse(torch.equal(audit.probe_features(cache, 'selected', 'query_plus_pred_roi'),
                                     audit.probe_features(cache, 'selected', 'query_plus_gt_roi')))

    def test_normalization_is_train_only_and_fixed_head_training(self):
        settings = dict(epochs=2, batch_size=4, learning_rate=.01, weight_decay=.01)
        x = torch.randn(9, 8); y = torch.arange(9) % 3
        _, _, first = audit.fit_probe(x, y, torch.randn(3, 8), settings, 0, torch.device('cpu'), 'test')
        _, _, second = audit.fit_probe(x, y, torch.randn(3, 8) * 100, settings, 0, torch.device('cpu'), 'test')
        torch.testing.assert_close(first['mean'], x.mean(0), rtol=0, atol=0)
        for key in first['head']:
            torch.testing.assert_close(first['head'][key], second['head'][key], rtol=0, atol=0)

    def test_current_config_paths_and_no_old_dataset(self):
        config = load_config(ROOT / 'configs/diagnostics/scbs_feature_probe_run1.yml')
        audit.validate_config(config)
        self.assertEqual(config['detector_config'], 'configs/dfine/dfine_s_scbs_hrw_132.yml')
        self.assertIn('dfine_s_scbs_hrw_seed0_run1/best_stg2.pth', config['checkpoint'])
        self.assertEqual(config['probe']['seeds'], [0, 1, 2])
        for split in ('train', 'val'):
            self.assertIn('/SCB-S/', config['data'][split]['annotation'])
        broken = copy.deepcopy(config); broken['input_size'] = [641, 640]
        with self.assertRaises(ValueError): audit.validate_config(broken)

    def test_full_frozen_detector_extraction_and_small_probe_training(self):
        with tempfile.TemporaryDirectory(prefix='scbs-probe-fixture-') as temporary:
            root = Path(temporary)
            detector_path = root / 'detector.yml'
            detector_path.write_text('__include__:\n  - ' + str(ROOT / 'configs/dfine/dfine_s_scbs_hrw.yml') +
                '\nHGNetv2:\n  pretrained: false\neval_spatial_size: [128, 128]\n' +
                ''.join(f'{split}_dataloader:\n  dataset:\n    img_folder: {root / split}\n'
                        f'    ann_file: {root / (split + ".json")}\n' for split in ('train', 'val')))
            cfg = YAMLConfig(str(detector_path))
            cfg.yaml_cfg = load_detector_config(str(detector_path), cfg={})
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False
            torch.manual_seed(9); model = cfg.model.eval().requires_grad_(False)
            # Synthetic checkpoint only: force visible high scores so selected
            # and IoU candidate paths are both exercised without a trained GPU model.
            with torch.no_grad():
                # Remove the large initial foreground bias in this *random*
                # fixture, so tiny encoder outputs do not quantize into top-k
                # ties. No real/user checkpoint is altered by these fixtures.
                model.decoder.enc_score_head.bias.zero_()
                model.decoder.enc_score_head.weight.mul_(100.)
                for head in model.decoder.dec_score_head:
                    head.weight.zero_(); head.bias.fill_(2.)
            image = Image.fromarray(torch.randint(0, 256, (128, 128, 3), dtype=torch.uint8).numpy())
            from torchvision.transforms.functional import to_tensor
            with torch.no_grad(): outputs = model(to_tensor(image).unsqueeze(0))
            raw = outputs['pred_boxes'][0]
            usable = torch.where((raw[:, :2] > .2).all(-1) & (raw[:, :2] < .8).all(-1))[0][:3]
            self.assertEqual(len(usable), 3)
            anns = []
            for i, q in enumerate(usable.tolist()):
                box = raw[q]; x = float((box[0]-box[2]/2)*128); y = float((box[1]-box[3]/2)*128)
                w = float(box[2]*128); h = float(box[3]*128)
                anns.append(dict(id=i+1, image_id=1, category_id=i, bbox=[x, y, w, h], area=w*h, iscrowd=0))
            for split in ('train', 'val'):
                (root / split).mkdir(); image.save(root / split / 'one.png'); image.save(root / split / 'empty.png')
                (root / f'{split}.json').write_text(json.dumps(dict(
                    images=[dict(id=1, file_name='one.png', width=128, height=128),
                            dict(id=2, file_name='empty.png', width=128, height=128)],
                    annotations=anns, categories=[dict(id=i, name=n) for i, n in audit.CATEGORIES])))
            checkpoint = root / 'baseline.pth'
            state = copy.deepcopy(model.state_dict())
            torch.save(dict(ema={'module': state}, model=state, last_epoch=131), checkpoint)
            config = load_config(ROOT / 'configs/diagnostics/scbs_feature_probe_run1.yml')
            config.update(detector_config=str(detector_path), checkpoint=str(checkpoint), input_size=[128,128],
                progress_every=1, data={split: dict(image_dir=str(root/split), annotation=str(root/f'{split}.json'))
                                       for split in ('train','val')},
                probe=dict(seeds=[0], epochs=1, batch_size=8, learning_rate=.001, weight_decay=.01))
            target = root / 'output'; target.mkdir()
            audit.validate_config(config)
            frozen, _, _, source = audit.build_frozen_detector(config, torch.device('cpu'))
            self.assertEqual(source, 'ema')
            self.assertFalse(frozen.training)
            self.assertTrue(all(not p.requires_grad for p in frozen.parameters()))
            for key, value in frozen.state_dict().items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)
            with torch.no_grad():
                actual = frozen(to_tensor(image).unsqueeze(0))
            for key in outputs:
                if isinstance(outputs[key], torch.Tensor):
                    torch.testing.assert_close(actual[key], outputs[key], rtol=0, atol=0)
            report = audit.extract(config, target, torch.device('cpu'))
            self.assertTrue(report['complete']); self.assertEqual(report['val']['gt'], 3)
            self.assertEqual(report['val']['images'], 2)
            self.assertGreater(report['val']['scopes']['selected']['all_matched'], 0)
            for key, value in torch.load(checkpoint, weights_only=False)['ema']['module'].items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)
            summary = audit.train_probes(config, target, torch.device('cpu'))
            self.assertTrue(summary['complete']); self.assertEqual(len(summary['experiments']), 18)
            self.assertIn('selected', summary['original_reference'])
            self.assertIn('read_write_errors', summary['experiments'][0]['aggregate']['all_matched'])
            self.assertTrue((target/'summary.txt').is_file())
            with self.assertRaises(FileExistsError): audit.train_probes(config, target, torch.device('cpu'))
            train = torch.load(target/'train_features.pth', weights_only=False)
            val = torch.load(target/'val_features.pth', weights_only=False)
            changed = copy.deepcopy(config); changed['confidence_threshold'] = .6
            with self.assertRaises(ValueError): audit.validate_caches(train, val, changed)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
