"""Synthetic position-ablation checks; no dataset, downloads or full training."""
import copy
import io
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
from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.solver.aqs_parameters import runtime_aqs_report
from src.zoo.dfine.dfine_decoder import LQE

BASE = 'dfine_s_scbs_hrw_110.yml'
CONFIG = 'dfine_s_scbs_hrw_110_AQS_l2_BoxLocalQueryAdapter_query_init.yml'
SCRIPT = ROOT / 'tools/diagnostics/run_scbs_AQS_l2_BoxLocalQueryAdapter_query_init_110.sh'
AQS = dict(use_aqs_refine=True, aqs_apply_layer=2, aqs_threshold=.5,
           aqs_temperature=.1, aqs_residual_init=.1)


def config(name=CONFIG):
    with patch('src.core.yaml_config.load_config', lambda path: load_config(path, cfg={})):
        cfg = YAMLConfig(str(ROOT / 'configs/dfine' / name))
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    cfg.yaml_cfg['eval_spatial_size'] = [128, 128]
    return cfg


def targets(device='cpu', empty=False):
    gt = [dict(labels=torch.empty(0, dtype=torch.long, device=device),
               boxes=torch.empty(0, 4, device=device)) for _ in range(2)]
    if not empty:
        gt[0] = dict(labels=torch.tensor([0, 1, 2], device=device),
                     boxes=torch.tensor([[.2,.3,.2,.2], [.5,.5,.3,.3], [.8,.7,.2,.2]], device=device))
    return gt


def copy_output(value):
    if isinstance(value, dict):
        return {k: copy_output(v) for k, v in value.items()}
    if isinstance(value, list):
        return [copy_output(v) for v in value]
    return value


def branches(model):
    return model.decoder.night_query_initializer, model.decoder.decoder.query_cls_refiner


def activate(model):
    # Test-only selected queries and nonzero trained residuals. Real init unchanged.
    with torch.no_grad():
        model.decoder.dec_score_head[1].bias.fill_(1)
        for branch in branches(model):
            nn.init.normal_(branch.fuse[-1].weight, std=.03)
        nn.init.normal_(model.decoder.dec_bbox_head[2].layers[-1].weight, std=.01)


class AQSBoxLocalQueryInitTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_config_parameter_count_optimizer_and_original_loss(self):
        base, cfg = config(BASE), config()
        expected, actual = copy.deepcopy(base.yaml_cfg), copy.deepcopy(cfg.yaml_cfg)
        for item in (expected, actual):
            item.pop('__include__', None); item.pop('output_dir', None)
        expected['DFINE'].update(night_probe='query_local_init', allow_aqs_l2_query_init_combo=True)
        expected['DFINETransformer'].update(AQS)
        self.assertEqual(expected, actual)
        model = cfg.model
        local, aqs = config(BASE), config(BASE)
        local.yaml_cfg['DFINE']['night_probe'] = 'query_local_init'
        aqs.yaml_cfg['DFINETransformer'].update(AQS)
        count = lambda m: sum(p.numel() for p in m.parameters())
        self.assertEqual(count(model), count(local.model) + count(aqs.model) - count(base.model))
        old = config(BASE)
        old.yaml_cfg['DFINE'].update(night_probe='decoder_l2_local', allow_aqs_l2_local_combo=True)
        old.yaml_cfg['DFINETransformer'].update(AQS)
        self.assertEqual(count(model), count(old.model))
        self.assertTrue(model.allow_aqs_l2_query_init_combo)
        self.assertFalse(model.allow_aqs_l2_local_combo)
        self.assertIsNone(getattr(model.decoder.decoder, 'night_local_adapter', None))
        ids = [id(p) for group in cfg.optimizer.param_groups for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(id(p) in ids for branch in branches(model) for p in branch.parameters()))
        self.assertEqual(cfg.yaml_cfg['DFINECriterion'], base.yaml_cfg['DFINECriterion'])
        print('Query-init + AQS-L2 raw parameters:', count(model), flush=True)

    def test_baseline_rng_weights_zero_start_and_dn_outputs(self):
        torch.manual_seed(77); base = config(BASE).model
        rng = torch.get_rng_state().clone()
        torch.manual_seed(77); model = config().model
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        for name, value in base.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
        images = torch.rand(2, 3, 128, 128)
        base.eval(); model.eval()
        with torch.no_grad():
            expected, actual = base(images), model(images)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        base.train(); model.train(); gt = targets()
        torch.manual_seed(78); expected = base(images, gt)
        torch.manual_seed(78); actual = model(images, gt)
        for key in ('pred_logits', 'pred_boxes', 'pred_corners'):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        for key in ('dn_outputs', 'enc_aux_outputs'):
            for new, old in zip(actual[key], expected[key]):
                for field, value in old.items():
                    if torch.is_tensor(value):
                        torch.testing.assert_close(new[field], value, atol=0, rtol=0)

    def test_execution_order_and_dn_aqs_context_isolation(self):
        model = config().model.eval(); activate(model)
        seen, order = {}, []
        def local_hook(module, args, result):
            order.append('init'); seen['init'] = result.detach().clone()
            self.assertEqual(args[0].shape[1], 300)
        def layer_hook(index):
            def hook(module, args):
                order.append(f'L{index}'); seen[f'L{index}'] = args[0].detach().clone()
            return hook
        def aqs_hook(module, args, result):
            order.append('aqs'); seen['aqs'] = result.detach().clone()
        decoder = model.decoder.decoder
        hooks = [model.decoder.night_query_initializer.register_forward_hook(local_hook),
                 decoder.query_cls_refiner.register_forward_hook(aqs_hook)]
        hooks += [layer.register_forward_pre_hook(layer_hook(i+1)) for i, layer in enumerate(decoder.layers)]
        try:
            with torch.no_grad(): model(torch.rand(2, 3, 128, 128))
        finally:
            for hook in hooks: hook.remove()
        self.assertEqual(order, ['init', 'L1', 'L2', 'aqs', 'L3'])
        torch.testing.assert_close(seen['init'], seen['L1'], atol=0, rtol=0)
        torch.testing.assert_close(seen['aqs'], seen['L3'], atol=0, rtol=0)
        decoder.train()
        query, logits = torch.randn(2, 8, 256), torch.ones(2, 8, 3)
        meta = {'dn_num_split': [3, 5]}
        old = decoder._apply_aqs_refiner(query, logits, meta, layer_idx=1)
        changed = query.clone(); changed[:, :3] += 100
        new = decoder._apply_aqs_refiner(changed, logits, meta, layer_idx=1)
        torch.testing.assert_close(old[:, 3:], new[:, 3:], atol=0, rtol=0)

    def test_top300_proposals_and_dn_prefix_unchanged_local_content_changes(self):
        model = config().model.train(); activate(model)
        decoder = model.decoder
        memory, shapes = decoder._get_encoder_input([torch.randn(2,256,s,s) for s in (16,8,4)])
        dn_query, dn_box = torch.randn(2, 3, 256), torch.randn(2, 3, 4)
        seen = {}
        hook = decoder.night_query_initializer.register_forward_pre_hook(
            lambda module, args: seen.update(box_requires_grad=args[1].requires_grad,
                                             query_count=args[0].shape[1]))
        try:
            actual = decoder._get_decoder_input(memory, shapes, dn_query, dn_box)
        finally:
            hook.remove()
        decoder.night_probe = 'none'
        original = decoder._get_decoder_input(memory, shapes, dn_query, dn_box)
        self.assertEqual(seen, dict(box_requires_grad=False, query_count=300))
        torch.testing.assert_close(actual[0][:, :3], dn_query, atol=0, rtol=0)
        self.assertGreater((actual[0][:, 3:] - original[0][:, 3:]).abs().sum().item(), 0)
        # Equal selected anchors, encoder boxes and encoder logits for these weights.
        for new, old in zip(actual[1:], original[1:]):
            if isinstance(new, list):
                for a, b in zip(new, old): torch.testing.assert_close(a, b, atol=0, rtol=0)
            else:
                torch.testing.assert_close(new, old, atol=0, rtol=0)

    def test_zero_start_and_trained_gradients_teacher_and_optimizer(self):
        cfg = config(); model = cfg.model.train()
        with torch.no_grad(): model.decoder.dec_score_head[1].bias.fill_(1)
        images, gt = torch.rand(2, 3, 128, 128), targets()
        loss = cfg.criterion(copy_output(model(images, gt)), gt)
        sum(loss.values()).backward()
        for branch in branches(model):
            self.assertGreater(branch.fuse[-1].weight.grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True); activate(model)
        reference = copy.deepcopy(model)
        reference.decoder.night_probe = 'none'
        reference.decoder.use_aqs_refine = reference.decoder.decoder.use_aqs_refine = False
        torch.manual_seed(79); output = model(images, gt)
        torch.manual_seed(79); original = reference(images, gt)
        # Query-init can affect all later decoder boxes, not encoder proposals.
        for new, old in zip(output['enc_aux_outputs'], original['enc_aux_outputs']):
            for key in ('pred_boxes', 'pred_logits'):
                torch.testing.assert_close(new[key], old[key], atol=0, rtol=0)
        for aux in output['aux_outputs']:
            torch.testing.assert_close(aux['teacher_logits'], output['pred_logits'], atol=0, rtol=0)
        loss = cfg.criterion(copy_output(output), gt)
        self.assertEqual(set(loss), set(cfg.criterion(copy_output(original), gt)))
        sum(loss.values()).backward()
        self.assertFalse([n for n,p in model.named_parameters() if p.requires_grad and p.grad is None])
        for branch in branches(model):
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in branch.parameters()))
        before = [branch.fuse[-1].weight.detach().clone() for branch in branches(model)]
        cfg.optimizer.step()
        for old, branch in zip(before, branches(model)):
            self.assertGreater((old - branch.fuse[-1].weight).abs().sum().item(), 0)

    def test_cpu_amp_empty_gt_and_local_ordinary_queries_only(self):
        cfg = config(); model = cfg.model.train(); activate(model)
        sizes = []
        hook = model.decoder.night_query_initializer.register_forward_pre_hook(
            lambda module, args: sizes.append(args[0].shape[1]))
        try:
            for empty in (False, True):
                model.zero_grad(set_to_none=True); gt = targets(empty=empty)
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    output = model(torch.rand(2,3,128,128), gt)
                    loss = cfg.criterion(copy_output(output), gt)
                self.assertTrue(all(torch.isfinite(v).all() for v in loss.values()))
                sum(loss.values()).backward()
                self.assertEqual(output['pred_logits'].shape, (2,300,3))
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for branch in branches(model) for p in branch.parameters()))
        finally:
            hook.remove()
        self.assertEqual(sizes, [300, 300])

    def test_strict_checkpoint_deploy_and_aqs_export(self):
        model = config().model.eval(); activate(model)
        buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
        restored = config().model.eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        with torch.no_grad():
            images = torch.rand(2,3,128,128)
            expected, actual = model(images), restored(images)
            for key in expected: torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
            deployed = copy.deepcopy(model).deploy()
            self.assertIsInstance(deployed.decoder.dec_score_head[1], nn.Linear)
            self.assertIsInstance(deployed.decoder.decoder.lqe_layers[1], LQE)
            self.assertTrue(hasattr(deployed.decoder, 'night_query_initializer'))
            feats = [torch.randn(2,256,s,s) for s in (16,8,4)]
            expected, actual = model.decoder(feats), deployed.decoder(feats)
            for key in expected: torch.testing.assert_close(actual[key], expected[key], atol=1e-5, rtol=1e-5)
        report = runtime_aqs_report(model, epoch=109)
        self.assertEqual(report['sources']['model']['decoder.decoder.query_cls_refiner']['decoder_layer'], 2)

    def test_narrow_opt_in_and_legacy_guards(self):
        bad = [({'DFINE': {'allow_aqs_l2_query_init_combo': False}}, 'separate'),
               ({'DFINE': {'night_probe': 'decoder_l2_local'}}, 'only supported'),
               ({'DFINE': {'allow_aqs_l2_query_init_combo': 1}}, 'boolean'),
               ({'DFINE': {'allow_aqs_l2_local_combo': True}}, 'only one'),
               ({'DFINETransformer': {'use_aqs_refine': False}}, 'requires only'),
               ({'DFINETransformer': {'aqs_apply_layer': 3}}, 'requires only'),
               ({'DFINETransformer': {'aqs_apply_layer': -1, 'aqs_apply_layers': [2,3]}}, 'requires only'),
               ({'HybridEncoder': {'use_encoder_highres_residual': True}}, 'combined')]
        for change, message in bad:
            with self.subTest(change=change):
                cfg = config()
                for key, value in change.items(): cfg.yaml_cfg[key].update(value)
                with self.assertRaisesRegex(ValueError, message): _ = cfg.model
        # Old local-L2 opt-in must not silently authorize query initialization.
        cfg = config(BASE)
        cfg.yaml_cfg['DFINE'].update(night_probe='query_local_init', allow_aqs_l2_local_combo=True)
        cfg.yaml_cfg['DFINETransformer'].update(AQS)
        with self.assertRaisesRegex(ValueError, 'only supported'): _ = cfg.model
        self.assertFalse(config(BASE).model.allow_aqs_l2_query_init_combo)

    def test_full_640_eval_profiler_and_launcher_protection(self):
        from calflops import calculate_flops
        cfg = config(); cfg.yaml_cfg['eval_spatial_size'] = [640,640]
        model = cfg.model.eval(); activate(model)
        with torch.no_grad(): result = model(torch.rand(1,3,640,640))
        self.assertEqual(result['pred_logits'].shape, (1,300,3))
        self.assertTrue(all(torch.isfinite(v).all() for v in result.values()))
        calculate_flops(copy.deepcopy(model).deploy(), input_shape=(1,3,640,640), print_results=False)
        subprocess.run(['bash', '-n', str(SCRIPT)], check=True)
        code = re.findall(r"<<'PY'\n(.*?)\nPY", SCRIPT.read_text(), re.S)[0]
        with patch.object(sys, 'argv', ['-', str(ROOT/'configs/dfine'/CONFIG), '1']), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as result: exec(compile(code, str(SCRIPT), 'exec'), {})
        self.assertEqual(result.exception.code, 0)
        with tempfile.TemporaryDirectory(prefix='aqs-init-existing-') as folder:
            output = Path(folder)/'AQS_l2_BoxLocalQueryAdapter_query_init_old'; output.mkdir()
            sentinel = output/'keep'; sentinel.write_text('old results')
            result = subprocess.run(['bash', str(SCRIPT), str(output)], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('输出已存在', result.stderr)
            self.assertEqual(sentinel.read_text(), 'old results')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; verify real FP16 on server')
    def test_cuda_fp16_joint_training(self):
        cfg = config(); model = cfg.model.cuda().train(); activate(model)
        criterion = cfg.criterion.cuda()
        for empty in (False, True):
            model.zero_grad(set_to_none=True); gt = targets(device='cuda', empty=empty)
            with torch.autocast('cuda', dtype=torch.float16):
                output = model(torch.rand(2,3,128,128, device='cuda'), gt)
                loss = criterion(copy_output(output), gt)
            self.assertTrue(all(torch.isfinite(v).all() for v in loss.values()))
            sum(loss.values()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for branch in branches(model) for p in branch.parameters()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
