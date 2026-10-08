"""CPU synthetic fixtures only; no user-data training or downloaded weights."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from src.core import YAMLConfig
from src.core.yaml_utils import load_config as load_detector_config
from tools.diagnostics import scbs_encoder_query_coverage as audit
from tools.diagnostics.gt_box_classifier import load_config


class EncoderQueryCoverageTests(unittest.TestCase):
    def settings(self):
        return load_config(ROOT/'configs/diagnostics/scbs_encoder_query_coverage_run1.yml')

    def test_config_uses_old_run1_and_not_new_budget(self):
        config = self.settings()
        audit.validate_config(config)
        self.assertEqual(config['detector_config'], 'configs/dfine/dfine_s_scbs_hrw_132.yml')
        self.assertIn('dfine_s_scbs_hrw_seed0_run1/best_stg2.pth', config['checkpoint'])
        self.assertEqual(config['splits'], ['val'])
        for key, value in (('input_size', [641,640]), ('iou_thresholds', [0.5,0.5]),
                           ('iou_thresholds', [float('nan')]), ('splits', ['test']),
                           ('save_selected_queries', 'true')):
            broken = copy.deepcopy(config); broken[key] = value
            with self.assertRaises(ValueError):
                audit.validate_config(broken)

    def test_geometry_is_unfiltered_and_not_class_matching_or_one_to_one(self):
        stage = dict(boxes=torch.tensor([[.5,.5,.4,.4], [.8,.8,.1,.1]]),
                     logits=torch.tensor([[-5.,-6.,-7.], [-7.,-6.,-5.]]),
                     valid=torch.tensor([True,False]))
        gt = torch.tensor([[.3,.3,.7,.7], [.3,.3,.7,.7]])
        matches = audit.best_matches(stage, gt, torch.tensor([0,1]))
        torch.testing.assert_close(matches['best_iou'], torch.ones(2))
        self.assertEqual(matches['best_index'].tolist(), [0,0])
        self.assertEqual(matches['class_iou'].tolist(), [1.,0.])
        torch.testing.assert_close(matches['valid_iou'], torch.ones(2))
        empty = audit.best_matches(stage, torch.empty(0,4), torch.empty(0,dtype=torch.long))
        self.assertTrue(all(len(v) == 0 for v in empty.values()))

    def test_invalid_anchor_coverage_is_separately_visible(self):
        stage = dict(boxes=torch.tensor([[.5,.5,.4,.4]]), logits=torch.ones(1,3),
                     valid=torch.tensor([False]))
        matches = audit.best_matches(stage, torch.tensor([[.3,.3,.7,.7]]), torch.tensor([0]))
        self.assertEqual(float(matches['best_iou'][0]), 1.)
        self.assertEqual(float(matches['valid_iou'][0]), 0.)

    def test_class_denominators_selection_loss_and_decoder_transitions(self):
        # all/top300/final: selection miss recovered; Decoder loss; neither;
        # successful throughout. A single candidate can cover multiple GT.
        values = [(1, [.9,.2,.8]), (2,[.9,.8,.2]), (2,[.1,.1,.1]), (0,[.9,.9,.9])]
        rows = []
        for category, ious in values:
            row = {'category_id': category}
            for stage, iou in zip(audit.STAGES, ious):
                row[stage] = dict(best_iou=iou, class_iou=iou/2, valid_iou=iou)
            rows.append(row)
        result = audit.summarize(rows, [.5,.75])
        all_stats = result['all']
        self.assertEqual(result['write']['gt_count'], 2)
        self.assertEqual(result['write']['stages']['encoder_top300']['0.5']['geometric']['fraction'], .5)
        transition = all_stats['transitions']['0.5']
        self.assertEqual(list(transition.values()), [1,1,1,1,1])
        self.assertEqual(all_stats['stages']['encoder_all']['0.5']['geometric']['covered'], 3)
        self.assertEqual(all_stats['stages']['encoder_top300']['0.5']['geometric']['covered'], 2)
        empty = audit.summarize([], [.5])
        self.assertIsNone(empty['write']['stages']['encoder_top300']['0.5']['geometric']['fraction'])

    def test_refuses_existing_results_before_loading_model(self):
        with tempfile.TemporaryDirectory(prefix='scbs-coverage-preserve-') as temporary:
            out = Path(temporary)
            (out/'summary.json').write_text('old results')
            with patch.object(audit, 'build_frozen_detector') as build:
                with self.assertRaises(FileExistsError):
                    audit.run(self.settings(), out, torch.device('cpu'))
                build.assert_not_called()
            self.assertEqual((out/'summary.json').read_text(), 'old results')

    def test_actual_baseline_capture_outputs_weights_and_hooks_unchanged(self):
        with tempfile.TemporaryDirectory(prefix='scbs-coverage-fixture-') as temporary:
            root = Path(temporary)
            config_path = root/'detector.yml'
            config_path.write_text('__include__:\n  - ' + str(ROOT/'configs/dfine/dfine_s_scbs_hrw_132.yml') +
                '\nHGNetv2:\n  pretrained: false\neval_spatial_size: [128,128]\n' +
                ''.join(f'{split}_dataloader:\n  dataset:\n    img_folder: {root/split}\n'
                        f'    ann_file: {root/(split+".json")}\n' for split in ('train','val')))
            cfg = YAMLConfig(str(config_path))
            cfg.yaml_cfg = load_detector_config(str(config_path), cfg={})
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False
            torch.manual_seed(17)
            model = cfg.model.eval().requires_grad_(False)
            with torch.no_grad():
                model.decoder.enc_score_head.bias.zero_()
                model.decoder.enc_score_head.weight.mul_(100.)
                # Make this synthetic pre-refinement observably different from
                # the true input references, without changing user checkpoints.
                model.decoder.pre_bbox_head.layers[-1].bias.fill_(.3)
            image = Image.fromarray(torch.randint(0,256,(128,128,3),dtype=torch.uint8).numpy())
            samples = to_tensor(image).unsqueeze(0)
            state = copy.deepcopy(model.state_dict())
            with torch.no_grad():
                original = model(samples)
                with audit.ProposalCapture(model) as capture:
                    decoded = {}
                    handle = model.decoder.decoder.register_forward_hook(
                        lambda module, args, value: decoded.update(pre_boxes=value[4].detach().clone()))
                    try:
                        outputs, stages, shapes = capture.forward(samples)
                    finally:
                        handle.remove()
                    self.assertFalse(capture.active)
                    self.assertEqual(shapes, [[16,16],[8,8],[4,4]])
                    self.assertEqual(stages['encoder_all']['boxes'].shape, (336,4))
                    self.assertEqual(stages['encoder_top300']['boxes'].shape, (300,4))
                    for key in original:
                        torch.testing.assert_close(outputs[key], original[key], rtol=0, atol=0)
                    ids = stages['encoder_top300']['grid_ids']
                    torch.testing.assert_close(stages['encoder_all']['boxes'][ids],
                                               stages['encoder_top300']['boxes'], rtol=0, atol=0)
                    # pre_bbox_head is a DIFFERENT, later refinement: ensure the
                    # capture did not accidentally use first-layer/pre-output boxes.
                    self.assertTrue(torch.equal(stages['encoder_top300']['boxes'],
                                                capture.values['reference_unact'][0].sigmoid()))
                    self.assertFalse(torch.equal(stages['encoder_top300']['boxes'], decoded['pre_boxes'][0]))
                    model.decoder.eval_spatial_size = None
                    try:
                        dynamic, dynamic_stages, _ = capture.forward(samples)
                    finally:
                        model.decoder.eval_spatial_size = [128,128]
                    for key in outputs:
                        torch.testing.assert_close(dynamic[key], outputs[key], rtol=0, atol=0)
                    for key in stages:
                        torch.testing.assert_close(dynamic_stages[key]['boxes'], stages[key]['boxes'], rtol=0, atol=0)
                self.assertFalse(capture.handles)
                after = model(samples)
            for key in original:
                torch.testing.assert_close(after[key], original[key], rtol=0, atol=0)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))

            checkpoint = root/'baseline.pth'
            torch.save(dict(ema={'module':state}, model=state, last_epoch=122), checkpoint)
            checkpoint_hash = audit.digest(checkpoint)
            usable = torch.where((original['pred_boxes'][0,:,:2] > .2).all(-1) &
                                 (original['pred_boxes'][0,:,:2] < .8).all(-1))[0][:3]
            self.assertEqual(len(usable), 3)
            anns = []
            for category, q in enumerate(usable.tolist()):
                box = original['pred_boxes'][0,q]
                x,y = ((box[:2]-box[2:]/2)*128).tolist(); w,h = (box[2:]*128).tolist()
                anns.append(dict(id=category+1, image_id=1, category_id=category,
                                 bbox=[x,y,w,h], area=w*h, iscrowd=0))
            for split in ('train','val'):
                (root/split).mkdir()
                image.save(root/split/'one.png'); image.save(root/split/'empty.png')
                (root/(split+'.json')).write_text(json.dumps(dict(
                    images=[dict(id=1,file_name='one.png',width=128,height=128),
                            dict(id=2,file_name='empty.png',width=128,height=128)],
                    annotations=anns, categories=[dict(id=i,name=n) for i,n in audit.CATEGORIES])))
            config = self.settings()
            config.update(detector_config=str(config_path), checkpoint=str(checkpoint),
                          input_size=[128,128], progress_every=1,
                          data={s:dict(image_dir=str(root/s),annotation=str(root/(s+'.json')))
                                for s in ('train','val')})
            with patch.object(audit, 'build_frozen_detector', wraps=audit.build_frozen_detector) as build:
                summary = audit.run(config, root/'results', torch.device('cpu'))
                self.assertEqual(build.call_count, 1)
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['weight_source'], 'ema')
            self.assertEqual(summary['checkpoint_epoch_metadata'], 122)
            self.assertEqual(summary['splits']['val']['gt_count'], 3)
            self.assertEqual(summary['splits']['val']['candidate_counts']['encoder_top300']['total'], 600)
            gt_records = [json.loads(line) for line in (root/'results/val/gt_records.jsonl').read_text().splitlines()]
            self.assertEqual(len(gt_records), 3)
            self.assertTrue(all(r['encoder_all']['best_iou'] >= r['encoder_top300']['best_iou'] for r in gt_records))
            packed = [json.loads(line) for line in (root/'results/val/selected_queries.jsonl').read_text().splitlines()]
            self.assertEqual(len(packed), 2)
            self.assertEqual(len(packed[0]['top300_grid_ids']), 300)
            self.assertEqual(len(packed[0]['encoder_top300_boxes']), 300)
            self.assertTrue((root/'results/summary.txt').is_file())
            self.assertEqual(audit.digest(checkpoint), checkpoint_hash)

    def test_hooks_are_removed_on_exception(self):
        from torch import nn
        class Broken(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Module()
                self.decoder.enc_output = nn.Identity()
                self.decoder.enc_score_head = nn.Identity()
                self.decoder.enc_bbox_head = nn.Identity()
                self.decoder.decoder = nn.Identity()
            def forward(self, value):
                self.decoder.enc_output(value)
                raise RuntimeError('fixture failure')
        model = Broken().eval()
        with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
            with audit.ProposalCapture(model) as capture:
                capture.forward(torch.ones(1,2,3))
        self.assertFalse(capture.active)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
