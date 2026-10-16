"""Joint AQS/local-L2 synthetic checks, never launch real full training."""
import copy
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
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.solver.aqs_parameters import runtime_aqs_report
from src.zoo.dfine.dfine_decoder import LQE
from tools.diagnostics.test_aqs_refine import copy_output, fresh_config
from tools.diagnostics.test_night_probes import targets

CONFIG = 'dfine_s_scbs_hrw_110_AQS_BoxLocalQueryAdapter_l2.yml'
SCRIPT = ROOT / 'tools/diagnostics/run_scbs_aqs_box_local_l2_110.sh'


def activate(model):
    with torch.no_grad():
        model.decoder.dec_score_head[1].bias.fill_(1)
        nn.init.normal_(model.decoder.decoder.query_cls_refiner.fuse[-1].weight, std=.03)
        nn.init.normal_(model.decoder.decoder.night_local_adapter.fuse[-1].weight, std=.03)
        nn.init.normal_(model.decoder.dec_bbox_head[2].layers[-1].weight, std=.01)


def branches(model):
    return (model.decoder.decoder.night_local_adapter, model.decoder.decoder.query_cls_refiner)


class AQSBoxLocalL2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_config_only_combines_tested_structures_and_original_loss(self):
        base, cfg = fresh_config('dfine_s_scbs_hrw_110.yml'), fresh_config(CONFIG)
        expected, actual = copy.deepcopy(base.yaml_cfg), copy.deepcopy(cfg.yaml_cfg)
        for obj in (expected, actual):
            obj.pop('__include__', None); obj.pop('output_dir', None)
        expected['DFINE'].update(night_probe='decoder_l2_local', allow_aqs_l2_local_combo=True,
            box_local_dim=64, box_local_grid_size=3, box_local_offset_scale=.15, box_local_residual_scale=.1)
        expected['DFINETransformer'].update(use_aqs_refine=True, aqs_apply_layer=2,
            aqs_threshold=.5, aqs_temperature=.1, aqs_residual_init=.1)
        self.assertEqual(expected, actual)
        combo = cfg.model
        local = fresh_config('dfine_s_scbs_hrw_110_night_n4.yml').model
        aqs = fresh_config('dfine_s_scbs_hrw_110_aqs_layer2_r010.yml').model
        self.assertTrue(combo.allow_aqs_l2_local_combo)
        adapter, refiner = branches(combo)
        self.assertEqual((adapter.dim, adapter.sampler.grid_size,
                          adapter.sampler.offset_scale, adapter.residual_scale), (64, 3, .15, .1))
        self.assertAlmostEqual(refiner.threshold_logit.sigmoid().item(), .5)
        self.assertEqual(refiner.temperature, .1)
        self.assertAlmostEqual(refiner.residual_scale.item(), .1)
        self.assertEqual(combo.decoder.decoder.aqs_layer_indices, (1,))
        self.assertEqual(sum(p.numel() for p in combo.parameters()),
                         sum(p.numel() for p in local.parameters()) + sum(p.numel() for p in aqs.parameters())
                         - sum(p.numel() for p in base.model.parameters()))
        ids = [id(p) for group in cfg.optimizer.param_groups for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(id(p) in ids for branch in branches(combo) for p in branch.parameters()))
        self.assertEqual(cfg.yaml_cfg['DFINECriterion'], base.yaml_cfg['DFINECriterion'])
        print('Combined raw parameters:', sum(p.numel() for p in combo.parameters()), flush=True)

    def test_baseline_rng_weights_and_zero_start_predictions_are_identical(self):
        torch.manual_seed(77); base = fresh_config('dfine_s_scbs_hrw_110.yml').model
        rng = torch.get_rng_state().clone()
        torch.manual_seed(77); combo = fresh_config(CONFIG).model
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        for key, value in base.state_dict().items():
            torch.testing.assert_close(combo.state_dict()[key], value, atol=0, rtol=0)
        images = torch.rand(2, 3, 128, 128)
        base.eval(); combo.eval()
        with torch.no_grad():
            original, actual = base(images), combo(images)
        for key in ('pred_logits', 'pred_boxes'):
            torch.testing.assert_close(actual[key], original[key], atol=0, rtol=0)
        base.train(); combo.train(); gt = targets()
        torch.manual_seed(78); original = base(images, gt)
        torch.manual_seed(78); actual = combo(images, gt)
        for key in ('pred_logits', 'pred_boxes', 'pred_corners'):
            torch.testing.assert_close(actual[key], original[key], atol=0, rtol=0)

    def test_local_then_aqs_then_l3_order_and_dn_context_isolation(self):
        model = fresh_config(CONFIG).model.eval(); activate(model)
        decoder = model.decoder.decoder
        seen = {}; order = []
        def local_hook(module, args, result):
            order.append('local'); seen['local'] = result.detach().clone()
        def aqs_hook(module, args, result):
            order.append('aqs'); seen['aqs_input'] = args[0].detach().clone(); seen['aqs_output'] = result.detach().clone()
        def l3_hook(module, args):
            order.append('l3'); seen['l3_input'] = args[0].detach().clone()
        hooks = [decoder.night_local_adapter.register_forward_hook(local_hook),
                 decoder.query_cls_refiner.register_forward_hook(aqs_hook),
                 decoder.layers[2].register_forward_pre_hook(l3_hook)]
        try:
            with torch.no_grad(): model(torch.rand(2, 3, 128, 128))
        finally:
            for hook in hooks: hook.remove()
        self.assertEqual(order, ['local', 'aqs', 'l3'])
        torch.testing.assert_close(seen['local'], seen['aqs_input'], atol=0, rtol=0)
        torch.testing.assert_close(seen['aqs_output'], seen['l3_input'], atol=0, rtol=0)
        decoder.train()
        query, logits = torch.randn(2, 8, 256), torch.ones(2, 8, 3)
        meta = {'dn_num_split':[3, 5]}
        old = decoder._apply_aqs_refiner(query, logits, meta, layer_idx=1)
        changed = query.clone(); changed[:, :3] += 100
        new = decoder._apply_aqs_refiner(changed, logits, meta, layer_idx=1)
        torch.testing.assert_close(new[:, 3:], old[:, 3:], atol=0, rtol=0)

    def test_training_gradients_current_l2_boxes_and_original_teacher_contract(self):
        cfg = fresh_config(CONFIG); model = cfg.model.train()
        # Synthetic randomly initialized queries start below AQS's hard gate.
        # Exercise selected queries without changing either zero output layer;
        # the real detector's original low-prior class bias remains unchanged.
        with torch.no_grad():
            model.decoder.dec_score_head[1].bias.fill_(1)
        images, gt = torch.rand(2, 3, 128, 128), targets()
        loss = cfg.criterion(copy_output(model(images, gt)), gt)
        sum(loss.values()).backward()
        for branch in branches(model):
            self.assertGreater(branch.fuse[-1].weight.grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True); activate(model)
        base = copy.deepcopy(model)
        base.decoder.use_aqs_refine = base.decoder.decoder.use_aqs_refine = False
        base.decoder.decoder.night_local_adapter = None
        torch.manual_seed(79); output = model(images, gt)
        torch.manual_seed(79); original = base(images, gt)
        for layer in (0, 1):
            torch.testing.assert_close(output['aux_outputs'][layer]['pred_boxes'],
                                       original['aux_outputs'][layer]['pred_boxes'], atol=0, rtol=0)
        torch.testing.assert_close(output['aux_outputs'][0]['pred_logits'],
                                   original['aux_outputs'][0]['pred_logits'], atol=0, rtol=0)
        for aux in output['aux_outputs']:
            torch.testing.assert_close(aux['teacher_logits'], output['pred_logits'], atol=0, rtol=0)
        loss = cfg.criterion(copy_output(output), gt)
        self.assertEqual(set(loss), set(cfg.criterion(copy_output(original), gt)))
        sum(loss.values()).backward()
        self.assertFalse([name for name,p in model.named_parameters() if p.requires_grad and p.grad is None])
        for branch in branches(model):
            for name, p in branch.named_parameters():
                self.assertTrue(torch.isfinite(p.grad).all(), name)
        before = [branch.fuse[-1].weight.detach().clone() for branch in branches(model)]
        cfg.optimizer.step()
        for old, branch in zip(before, branches(model)):
            self.assertGreater((old - branch.fuse[-1].weight).abs().sum().item(), 0)

    def test_cpu_amp_empty_gt_and_local_never_receives_dn_queries(self):
        cfg = fresh_config(CONFIG); model = cfg.model.train(); activate(model)
        sizes = []
        hook = model.decoder.decoder.night_local_adapter.register_forward_pre_hook(
            lambda module,args: sizes.append(args[0].shape[1]))
        try:
            for empty in (False, True):
                gt = targets(empty=empty); model.zero_grad(set_to_none=True)
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    output = model(torch.rand(2, 3, 128, 128), gt)
                    loss = cfg.criterion(copy_output(output), gt)
                self.assertTrue(all(torch.isfinite(v).all() for v in loss.values()))
                sum(loss.values()).backward()
                self.assertEqual(output['pred_logits'].shape, (2, 300, 3))
                for branch in branches(model):
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters()))
        finally:
            hook.remove()
        self.assertEqual(sizes, [300, 300])

    def test_strict_checkpoint_deploy_and_aqs_parameter_export(self):
        model = fresh_config(CONFIG).model.eval(); activate(model)
        buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
        restored = fresh_config(CONFIG).model.eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        with torch.no_grad():
            images = torch.rand(2, 3, 128, 128)
            original, result = model(images), restored(images)
            for key in original:
                torch.testing.assert_close(result[key], original[key], atol=0, rtol=0)
            deployed = copy.deepcopy(model).deploy()
            self.assertIsInstance(deployed.decoder.dec_score_head[1], nn.Linear)
            self.assertIsInstance(deployed.decoder.decoder.lqe_layers[1], LQE)
            feats = [torch.randn(2, 256, side, side) for side in (16, 8, 4)]
            original, result = model.decoder(feats), deployed.decoder(feats)
            for key in original:
                torch.testing.assert_close(result[key], original[key], atol=1e-5, rtol=1e-5)
        report = runtime_aqs_report(model, epoch=109)
        self.assertEqual(report['sources']['model']['decoder.decoder.query_cls_refiner']['decoder_layer'], 2)

    def test_only_explicit_l2_combination_allowed_and_legacy_guards_preserved(self):
        bad = [({'DFINE':{'allow_aqs_l2_local_combo':False}}, 'separate'),
               ({'DFINE':{'night_probe':'query_local_init'}}, 'only supported'),
               ({'DFINE':{'allow_aqs_l2_local_combo':1}}, 'boolean'),
               ({'DFINETransformer':{'use_aqs_refine':False}}, 'requires only'),
               ({'DFINETransformer':{'aqs_apply_layer':3}}, 'requires only'),
               ({'DFINETransformer':{'aqs_apply_layer':-1, 'aqs_apply_layers':[2,3]}}, 'requires only'),
               ({'HybridEncoder':{'use_encoder_highres_residual':True}}, 'combined')]
        for change, message in bad:
            with self.subTest(change=change):
                cfg = fresh_config(CONFIG)
                for key,value in change.items(): cfg.yaml_cfg[key].update(value)
                with self.assertRaisesRegex(ValueError, message): _ = cfg.model
        for name in ('dfine_s_scbs_hrw_110.yml', 'dfine_s_scbs_hrw_110_night_n4.yml',
                     'dfine_s_scbs_hrw_110_aqs_layer2_r010.yml'):
            self.assertFalse(fresh_config(name).model.allow_aqs_l2_local_combo)

    def test_full_640_eval_and_profiler(self):
        from calflops import calculate_flops
        cfg = fresh_config(CONFIG); cfg.yaml_cfg['eval_spatial_size'] = [640, 640]
        model = cfg.model.eval(); activate(model)
        with torch.no_grad():
            result = model(torch.rand(1, 3, 640, 640))
        self.assertEqual(result['pred_logits'].shape, (1, 300, 3))
        self.assertTrue(all(torch.isfinite(v).all() for v in result.values()))
        calculate_flops(copy.deepcopy(model).deploy(), input_shape=(1,3,640,640), print_results=False)

    def test_launcher_preflight_syntax_and_no_overwrite(self):
        subprocess.run(['bash','-n',str(SCRIPT)], check=True)
        code = re.findall(r"<<'PY'\n(.*?)\nPY", SCRIPT.read_text(), re.S)[0]
        with patch.object(sys, 'argv', ['-',str(ROOT/'configs/dfine'/CONFIG),'1']), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                exec(compile(code, str(SCRIPT), 'exec'), {})
        self.assertEqual(result.exception.code, 0)
        with tempfile.TemporaryDirectory(prefix='aqs-local-existing-') as folder:
            output = Path(folder)/'old'; output.mkdir(); sentinel = output/'keep'; sentinel.write_text('old results')
            result = subprocess.run(['bash',str(SCRIPT),str(output)], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(), 'old results')

    def test_launcher_local_dry_run_modes_and_old_path_argument(self):
        with tempfile.TemporaryDirectory(prefix='aqs-local-launcher-') as folder:
            env = dict(os.environ, SCBS_COMBO_ROOT=folder, SCBS_COMBO_CONDA='/not-installed/conda')
            for args in (['run2','--dry-run'], ['--dry-run','run2','--wait-gpu','--diagnostics'],
                         ['--dry-run',str(Path(folder)/'legacy_new_output')]):
                result = subprocess.run(['bash',str(SCRIPT),*args],env=env,text=True,capture_output=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('--seed=0',result.stdout)
                self.assertIn('--master_port=7781',result.stdout)
                self.assertNotIn(' --resume ',result.stdout)
            self.assertEqual(list(Path(folder).iterdir()),[])
            for args in (['run0','--dry-run'], ['run1','run2','--dry-run'], ['--unknown']):
                self.assertNotEqual(subprocess.run(['bash',str(SCRIPT),*args],env=env,
                    text=True,capture_output=True).returncode,0)
            old = Path(folder)/'dfine_s_scbs_hrw_epochs110_AQS_BoxLocalQueryAdapter_l2_seed0_run2'
            old.mkdir();(old/'keep').write_text('old result')
            result = subprocess.run(['bash',str(SCRIPT),'run2','--dry-run'],env=env,text=True,capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual((old/'keep').read_text(),'old result')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; verify real FP16 on server')
    def test_cuda_fp16_joint_training(self):
        cfg = fresh_config(CONFIG); model = cfg.model.cuda().train(); activate(model)
        criterion = cfg.criterion.cuda()
        for empty in (False, True):
            model.zero_grad(set_to_none=True); gt = targets(device='cuda', empty=empty)
            with torch.autocast('cuda', dtype=torch.float16):
                output = model(torch.rand(2,3,128,128,device='cuda'), gt)
                loss = criterion(copy_output(output), gt)
            self.assertTrue(all(torch.isfinite(v).all() for v in loss.values()))
            sum(loss.values()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for branch in branches(model) for p in branch.parameters()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
