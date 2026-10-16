"""Parallel-fusion routing/gradient regression tests; no real dataset/training."""
import copy
import io
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
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.diagnostics.test_aqs_refine import copy_output, fresh_config
from tools.diagnostics.test_aqs_box_local_l2 import activate, branches
from tools.diagnostics.test_night_probes import targets

SERIAL = 'dfine_s_scbs_hrw_110_AQS_BoxLocalQueryAdapter_l2.yml'
CONFIG = 'dfine_s_scbs_hrw_110_AQS_BoxLocalQueryAdapter_l2_parallel.yml'
RUN = ROOT / 'tools/diagnostics/run_scbs_aqs_box_local_l2_parallel_110.sh'
COMMON_RUN = ROOT / 'tools/diagnostics/run_scbs_aqs_box_local_l2_110.sh'
SYNC = ROOT / 'tools/diagnostics/sync_scbs_aqs_box_local_l2_parallel.sh'


def observe(model, images, gt=None):
    """Record real detector branch inputs, outputs and the two L2 classifiers."""
    seen = dict(local=[], aqs=[], l3=[], score_inputs=[], scores=[])
    decoder = model.decoder.decoder
    clone = lambda x: x.detach().clone()
    hooks = [
        decoder.night_local_adapter.register_forward_hook(
            lambda m, a, r: seen['local'].append((clone(a[0]), clone(r)))),
        decoder.query_cls_refiner.register_forward_hook(
            lambda m, a, r: seen['aqs'].append((clone(a[0]), clone(a[1]), clone(r)))),
        decoder.layers[2].register_forward_pre_hook(
            lambda m, a: seen['l3'].append(clone(a[0]))),
        model.decoder.dec_score_head[1].register_forward_pre_hook(
            lambda m, a: seen['score_inputs'].append(clone(a[0]))),
        decoder.lqe_layers[1].register_forward_hook(
            lambda m, a, r: seen['scores'].append(clone(r))),
    ]
    try:
        result = model(images, gt)
    finally:
        for handle in hooks:
            handle.remove()
    return result, seen


class ParallelFusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(123)

    def test_only_fusion_changes_same_weights_rng_params_optimizer_and_losses(self):
        serial, parallel = fresh_config(SERIAL), fresh_config(CONFIG)
        expected, actual = copy.deepcopy(serial.yaml_cfg), copy.deepcopy(parallel.yaml_cfg)
        for cfg in (expected, actual):
            cfg.pop('__include__', None); cfg.pop('output_dir', None)
        expected['DFINE']['aqs_l2_local_fusion'] = 'parallel'
        self.assertEqual(expected, actual)
        torch.manual_seed(124); old = serial.model; rng = torch.get_rng_state().clone()
        torch.manual_seed(124); model = parallel.model
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertEqual(model.aqs_l2_local_fusion, 'parallel')
        self.assertEqual(model.decoder.decoder.aqs_l2_local_fusion, 'parallel')
        self.assertEqual(set(old.state_dict()), set(model.state_dict()))
        for name, value in old.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 10492533)
        ids = [id(p) for group in parallel.optimizer.param_groups for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(id(p) in ids for branch in branches(model) for p in branch.parameters()))
        self.assertEqual(parallel.yaml_cfg['DFINECriterion'], serial.yaml_cfg['DFINECriterion'])

    def test_zero_start_matches_baseline_in_train_and_eval(self):
        torch.manual_seed(125); base = fresh_config('dfine_s_scbs_hrw_110.yml').model
        torch.manual_seed(125); model = fresh_config(CONFIG).model
        images = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            for training in (False, True):
                base.train(training); model.train(training)
                gt = targets() if training else None
                torch.manual_seed(126); expected = base(images, gt)
                torch.manual_seed(126); actual = model(images, gt)
                for key in ('pred_logits', 'pred_boxes'):
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_parallel_original_query_original_scores_merge_and_dn_routing(self):
        model = fresh_config(CONFIG).model; activate(model)
        images = torch.rand(2, 3, 128, 128)
        for training, empty in ((False, False), (True, False), (True, True)):
            with self.subTest(training=training, empty=empty), torch.no_grad():
                model.train(training)
                output, seen = observe(model, images, targets(empty=empty) if training else None)
                query, local_output = seen['local'][0]
                self.assertEqual(query.shape[1], 300)
                raw_query = torch.cat([item[0] for item in seen['aqs']], dim=1)
                gate_scores = torch.cat([item[1] for item in seen['aqs']], dim=1)
                aqs_output = torch.cat([item[2] for item in seen['aqs']], dim=1)
                count = raw_query.shape[1] - 300
                torch.testing.assert_close(query, raw_query[:, count:], rtol=0, atol=0)
                torch.testing.assert_close(raw_query, seen['score_inputs'][0], rtol=0, atol=0)
                torch.testing.assert_close(gate_scores, seen['scores'][0], rtol=0, atol=0)
                delta = torch.cat((torch.zeros_like(raw_query[:, :count]), local_output-query), dim=1)
                expected = aqs_output + delta
                torch.testing.assert_close(seen['l3'][0], expected, rtol=0, atol=0)
                self.assertGreater((expected[:, count:]-aqs_output[:, count:]).abs().sum().item(), 0)
                if count:
                    torch.testing.assert_close(expected[:, :count], aqs_output[:, :count], rtol=0, atol=0)
                if training:
                    torch.testing.assert_close(seen['score_inputs'][1], expected, rtol=0, atol=0)
                    torch.testing.assert_close(output['aux_outputs'][1]['pred_logits'],
                                               seen['scores'][1][:, count:], rtol=0, atol=0)

    def test_local_change_does_not_change_aqs_gate_inputs_but_reaches_l3(self):
        model = fresh_config(CONFIG).model.eval(); activate(model)
        images = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            _, before = observe(model, images)
            model.decoder.decoder.night_local_adapter.fuse[-1].bias.add_(.7)
            _, after = observe(model, images)
        for old, new in zip(before['aqs'][0], after['aqs'][0]):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        self.assertGreater((after['l3'][0]-before['l3'][0]).abs().sum().item(), 0)

    def test_original_loss_gradients_unchanged_l2_boxes_and_teacher(self):
        cfg = fresh_config(CONFIG); model = cfg.model.train()
        with torch.no_grad():
            model.decoder.dec_score_head[1].bias.fill_(1)
        images, gt = torch.rand(2, 3, 128, 128), targets()
        sum(cfg.criterion(copy_output(model(images, gt)), gt).values()).backward()
        for branch in branches(model):
            self.assertGreater(branch.fuse[-1].weight.grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True); activate(model)
        control = copy.deepcopy(model)
        control.decoder.use_aqs_refine = control.decoder.decoder.use_aqs_refine = False
        control.decoder.decoder.night_local_adapter = None
        torch.manual_seed(127); output = model(images, gt)
        torch.manual_seed(127); original = control(images, gt)
        for i in (0, 1):
            torch.testing.assert_close(output['aux_outputs'][i]['pred_boxes'],
                                       original['aux_outputs'][i]['pred_boxes'], rtol=0, atol=0)
        for aux in output['aux_outputs']:
            torch.testing.assert_close(aux['teacher_logits'], output['pred_logits'], rtol=0, atol=0)
        losses = cfg.criterion(copy_output(output), gt)
        self.assertEqual(set(losses), set(cfg.criterion(copy_output(original), gt)))
        sum(losses.values()).backward()
        self.assertFalse([n for n, p in model.named_parameters() if p.requires_grad and p.grad is None])
        for branch in branches(model):
            for name, parameter in branch.named_parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        before = [b.fuse[-1].weight.detach().clone() for b in branches(model)]
        cfg.optimizer.step()
        for old, branch in zip(before, branches(model)):
            self.assertGreater((old-branch.fuse[-1].weight).abs().sum().item(), 0)

    def check_amp(self, device, dtype):
        cfg = fresh_config(CONFIG); model = cfg.model.to(device).train(); activate(model)
        criterion = cfg.criterion.to(device)
        for empty in (False, True):
            gt = targets(device=device, empty=empty); model.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=dtype):
                output = model(torch.rand(2, 3, 128, 128, device=device), gt)
                losses = criterion(copy_output(output), gt)
            self.assertEqual(output['pred_logits'].shape, (2, 300, 3))
            self.assertTrue(all(torch.isfinite(v).all() for v in losses.values()))
            sum(losses.values()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for branch in branches(model) for p in branch.parameters()))

    def test_cpu_amp_with_and_without_gt(self):
        self.check_amp('cpu', torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable locally')
    def test_cuda_fp16_with_and_without_gt(self):
        self.check_amp('cuda', torch.float16)

    def test_checkpoint_deploy_and_full_640_eval(self):
        cfg = fresh_config(CONFIG); cfg.yaml_cfg['eval_spatial_size'] = [640, 640]
        model = cfg.model.eval(); activate(model)
        state = io.BytesIO(); torch.save(model.state_dict(), state); state.seek(0)
        restored_cfg = fresh_config(CONFIG); restored_cfg.yaml_cfg['eval_spatial_size'] = [640, 640]
        restored = restored_cfg.model.eval()
        restored.load_state_dict(torch.load(state, weights_only=True), strict=True)
        with torch.no_grad():
            images = torch.rand(1, 3, 640, 640)
            expected, actual = model(images), restored(images)
            self.assertEqual(actual['pred_logits'].shape, (1, 300, 3))
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                self.assertTrue(torch.isfinite(actual[key]).all())
            deployed = copy.deepcopy(model).deploy()
            self.assertIsInstance(deployed.decoder.dec_score_head[1], nn.Linear)
            features = [torch.randn(1, 256, size, size) for size in (80, 40, 20)]
            expected, actual = model.decoder(features), deployed.decoder(features)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-5)

    def test_invalid_modes_and_non_combo_configs_are_rejected(self):
        for changes in ({'aqs_l2_local_fusion': 'unknown'}, {'aqs_l2_local_fusion': True},
                        {'allow_aqs_l2_local_combo': False}, {'night_probe': 'query_local_init'}):
            cfg = fresh_config(CONFIG); cfg.yaml_cfg['DFINE'].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                _ = cfg.model
        cfg = fresh_config(CONFIG); cfg.yaml_cfg['DFINETransformer']['aqs_apply_layer'] = 3
        with self.assertRaises(ValueError):
            _ = cfg.model
        for config in ('dfine_s_scbs_hrw_110.yml', 'dfine_s_scbs_hrw_110_night_n4.yml', SERIAL):
            self.assertEqual(fresh_config(config).model.aqs_l2_local_fusion, 'serial')

    def test_launcher_protocol_dry_run_sync_manifest_and_output_protection(self):
        for script in (RUN, COMMON_RUN, SYNC):
            subprocess.run(['bash', '-n', str(script)], check=True)
        code = re.findall(r"<<'PY'\n(.*?)\nPY", COMMON_RUN.read_text(), re.S)[0]
        with patch.object(sys, 'argv', ['-', str(ROOT/'configs/dfine'/CONFIG), '1', '1']), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                exec(compile(code, str(COMMON_RUN), 'exec'), {})
        self.assertEqual(result.exception.code, 0)
        with tempfile.TemporaryDirectory(prefix='aqs-parallel-launch-') as folder:
            env = dict(os.environ, SCBS_COMBO_ROOT=folder, SCBS_COMBO_CONDA='/not-installed/conda')
            for args in (['run1', '--dry-run'], ['run2', '--wait-gpu', '--diagnostics', '--dry-run']):
                result = subprocess.run(['bash', str(RUN), *args], env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(CONFIG, result.stdout)
                self.assertIn('_parallel_seed0_', result.stdout)
                self.assertIn('--seed=0', result.stdout)
            self.assertEqual(list(Path(folder).iterdir()), [])
            old = Path(folder)/'dfine_s_scbs_hrw_epochs110_AQS_BoxLocalQueryAdapter_l2_parallel_seed0_run1'
            old.mkdir(); (old/'keep').write_text('old results')
            result = subprocess.run(['bash', str(RUN), 'run1', '--dry-run'], env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((old/'keep').read_text(), 'old results')
        result = subprocess.run(['bash', str(SYNC), '--dry-run'], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('不连接服务器', result.stdout)
        manifest = ROOT/'tools/diagnostics/scbs_aqs_box_local_l2_parallel_sync_files.txt'
        files = manifest.read_text().splitlines()
        self.assertTrue(all((ROOT/name).is_file() for name in files))
        self.assertNotIn('configs/dfine/dfine_s_scbs_hrw_110.yml', files)
        self.assertNotIn('src/zoo/dfine/dfine_criterion.py', files)


if __name__ == '__main__':
    unittest.main(verbosity=2)
