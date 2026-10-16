"""Local implementation checks, not training results or evidence of accuracy gains."""
import copy
import ast
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
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.diagnostics.test_aqs_refine import fresh_config, copy_output
from tools.diagnostics.test_night_probes import targets
from tools.diagnostics.boxlocal_five_spec import VARIANTS, BOX_BASELINE, config_path, output_name
from tools.diagnostics.scbs_boxlocal_five import collect_result, validate_config, write_summary
from src.zoo.dfine.box_local_extensions import (
    ExtensionBase, PartRelationQueryInit, ReliableQueryCompetition,
    CrossScaleNeighborhoodFusion,
)


def config(tag):
    return fresh_config(Path(config_path(tag)).name)


def extension(model):
    return next(module for module in model.modules() if isinstance(module, ExtensionBase))


def activate(module):
    for child in module.modules():
        if isinstance(child, (nn.Conv2d, nn.Linear)) and torch.count_nonzero(child.weight) == 0:
            nn.init.normal_(child.weight, std=.02)
            if child.bias is not None:
                nn.init.normal_(child.bias, std=.02)


class BoxLocalFiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(310)

    def test_single_variable_original_weights_rng_zero_start_optimizer(self):
        torch.manual_seed(311)
        base_config = fresh_config(Path(BOX_BASELINE).name)
        base = base_config.model
        initial = copy.deepcopy(base.state_dict())
        rng = torch.get_rng_state().clone()
        images = torch.rand(2, 3, 128, 128)
        for tag in VARIANTS:
            with self.subTest(tag=tag):
                base.load_state_dict(initial, strict=True)
                validate_config(tag)
                cfg = config(tag)
                torch.manual_seed(311); model = cfg.model
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                for name, value in base.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
                self.assertEqual(model.night_probe, 'decoder_l2_local')
                self.assertFalse(model.decoder.use_aqs_refine)
                self.assertEqual(cfg.yaml_cfg['DFINECriterion'], base_config.yaml_cfg['DFINECriterion'])
                self.assertEqual(sum(isinstance(m, ExtensionBase) for m in model.modules()), 1)
                for training in (False, True):
                    base.train(training); model.train(training)
                    gt = targets() if training else None
                    with torch.no_grad():
                        torch.manual_seed(312); old = base(images, gt)
                        torch.manual_seed(312); new = model(images, gt)
                    for key in ('pred_logits', 'pred_boxes'):
                        torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)
                ids = [id(p) for group in cfg.optimizer.param_groups for p in group['params']]
                self.assertEqual(len(ids), len(set(ids)))
                self.assertTrue(all(id(p) in ids for p in extension(model).parameters()))
                group_lr = {id(p): group['lr'] for group in cfg.optimizer.param_groups for p in group['params']}
                self.assertTrue(all(group_lr[id(p)] == cfg.yaml_cfg['optimizer']['lr'] for p in extension(model).parameters()))

    def check_amp(self, device, dtype):
        images = torch.rand(2, 3, 128, 128, device=device)
        for tag in VARIANTS:
            cfg = config(tag); model = cfg.model.to(device).train(); module = extension(model)
            criterion = cfg.criterion.to(device)
            for empty in (False, True):
                with self.subTest(tag=tag, empty=empty):
                    gt = targets(device=device, empty=empty); model.zero_grad(set_to_none=True)
                    with torch.autocast(device, dtype=dtype):
                        output = model(images, gt)
                        losses = criterion(copy_output(output), gt)
                    self.assertEqual(output['pred_logits'].shape, (2, 300, 3))
                    self.assertTrue(all(torch.isfinite(v).all() for v in losses.values()))
                    sum(losses.values()).backward()
                    for name, parameter in model.named_parameters():
                        if not parameter.requires_grad:
                            continue
                        if empty and name == 'decoder.denoising_class_embed.weight':
                            # Original detector has no DN branch for an all-empty batch.
                            continue
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)
                    activate(module)  # Second pass exercises nonzero inner-branch gradients.
            old = [p.detach().clone() for p in module.parameters()]
            cfg.optimizer.step()
            self.assertTrue(any((a-b).abs().sum() > 0 for a, b in zip(old, module.parameters())))

    def test_cpu_amp_original_losses_nonempty_and_empty_gt_gradients(self):
        self.check_amp('cpu', torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable locally')
    def test_cuda_fp16_original_losses_and_gradients(self):
        self.check_amp('cuda', torch.float16)

    def test_routing_and_dn_exclusion(self):
        images, gt = torch.rand(2, 3, 128, 128), targets()
        for tag in VARIANTS:
            with self.subTest(tag=tag):
                model = config(tag).model.train(); module = extension(model); activate(module)
                control = copy.deepcopy(model); extension(control).enabled = False
                events, seen = [], {}
                def record(name):
                    def hook(m, args, result):
                        events.append(name)
                        seen[name] = (tuple(x.detach().clone() if torch.is_tensor(x) else x for x in args),
                                      result.detach().clone() if torch.is_tensor(result) else result)
                    return hook
                def before_l1(m, args):
                    events.append('L1')
                    seen['L1'] = args[0].detach().clone()
                handles = [model.backbone.register_forward_hook(record('backbone')),
                           model.encoder.register_forward_hook(record('encoder')),
                           module.register_forward_hook(record('extension')),
                           model.decoder.decoder.night_local_adapter.register_forward_hook(record('BoxLocal')),
                           model.decoder.decoder.layers[0].register_forward_pre_hook(before_l1),
                           model.decoder.decoder.layers[2].register_forward_pre_hook(lambda m,a: events.append('L3'))]
                try:
                    with torch.no_grad():
                        torch.manual_seed(313); actual = model(images, gt)
                        torch.manual_seed(313); old = control(images, gt)
                finally:
                    for handle in handles: handle.remove()
                self.assertEqual(seen['BoxLocal'][0][0].shape[1], 300)
                self.assertEqual(events.count('BoxLocal'), 1)
                self.assertLess(events.index('BoxLocal'), events.index('L3'))
                if tag == 'DDB':
                    self.assertLess(events.index('backbone'), events.index('extension'))
                    self.assertLess(events.index('extension'), events.index('encoder'))
                    self.assertEqual(seen['extension'][0][0].shape[1], model.encoder.in_channels[0])
                elif tag == 'SCR':
                    self.assertLess(events.index('backbone'), events.index('extension'))
                    self.assertLess(events.index('extension'), events.index('encoder'))
                    self.assertIs(module, model.encoder.box_semantic_extension)
                elif tag == 'CNF':
                    self.assertLess(events.index('encoder'), events.index('extension'))
                    self.assertLess(events.index('extension'), events.index('L1'))
                else:
                    self.assertEqual(seen['extension'][0][0].shape[1], 300)
                    for item in actual['enc_aux_outputs']:
                        torch.testing.assert_close(item['pred_logits'], old['enc_aux_outputs'][0]['pred_logits'], rtol=0, atol=0)
                        torch.testing.assert_close(item['pred_boxes'], old['enc_aux_outputs'][0]['pred_boxes'], rtol=0, atol=0)
                    if tag == 'PRQI':
                        self.assertLess(events.index('extension'), events.index('L1'))
                        # Fresh detector bbox output heads are zero initialized, so
                        # changed queries need not change boxes before training.
                        args, refined = seen['extension']
                        self.assertGreater((refined-args[0]).abs().sum().item(), 0)
                        count = actual['dn_meta']['dn_num_split'][0]
                        torch.testing.assert_close(seen['L1'][:, count:], refined, rtol=0, atol=0)
                    else:
                        self.assertLess(events.index('L3'), events.index('extension'))
                        torch.testing.assert_close(actual['pred_boxes'], old['pred_boxes'], rtol=0, atol=0)
                        for new, previous in zip(actual['aux_outputs'], old['aux_outputs']):
                            torch.testing.assert_close(new['pred_boxes'], previous['pred_boxes'], rtol=0, atol=0)
                            torch.testing.assert_close(new['pred_logits'], previous['pred_logits'], rtol=0, atol=0)
                        for new, previous in zip(actual['dn_outputs'], old['dn_outputs']):
                            torch.testing.assert_close(new['pred_logits'], previous['pred_logits'], rtol=0, atol=0)
                        self.assertGreater((actual['pred_logits']-old['pred_logits']).abs().sum().item(), 0)
                for aux in actual['aux_outputs']:
                    torch.testing.assert_close(aux['teacher_logits'], actual['pred_logits'], rtol=0, atol=0)

    def test_prqi_box_detach_null_outside_image_and_query_permutation(self):
        module = PartRelationQueryInit(channels=32, dim=16, chunk=2); activate(module)
        query = torch.randn(2, 5, 32, requires_grad=True)
        boxes = torch.tensor([.5, .5, .5, .5]).expand(2, 5, 4).clone().requires_grad_()
        feature = torch.randn(2, 32, 9, 11, requires_grad=True)
        output = module(query, boxes, feature)
        output.square().sum().backward()
        self.assertIsNone(boxes.grad)
        self.assertGreater(feature.grad.abs().sum().item(), 0)
        order = torch.tensor([4, 2, 0, 3, 1])
        torch.testing.assert_close(module(query[:, order], boxes[:, order], feature), output[:, order])
        self.assertTrue(torch.isfinite(module(query, torch.full_like(boxes, -5), feature)).all())
        self.assertEqual(module(query[:, :0], boxes[:, :0], feature).shape, (2, 0, 32))

    def test_rqc_reliability_proxy_null_and_permutation(self):
        module = ReliableQueryCompetition(channels=32, dim=16, chunk=2); activate(module)
        query = torch.randn(2, 5, 32)
        boxes = torch.tensor([.5, .5, .5, .5]).expand(2, 5, 4).clone()
        logits, corners = torch.randn(2, 5, 3), torch.randn(2, 5, 132)
        output = module(query, boxes, logits, corners)
        order = torch.tensor([4, 2, 0, 3, 1])
        torch.testing.assert_close(module(query[:, order], boxes[:, order], logits[:, order], corners[:, order]), output[:, order])
        quality = torch.zeros(2, 5, 12)
        low = module.reliability(torch.full_like(logits, -5), quality)
        high = module.reliability(torch.full_like(logits, 5), quality)
        self.assertTrue((high > low).all())
        quality[..., :4] = 1
        self.assertTrue((high > module.reliability(torch.full_like(logits, 5), quality)).all())
        # No non-self donors: both availability gates zero, exact unchanged logits.
        torch.testing.assert_close(module(query[:, :1], boxes[:, :1], logits[:, :1], corners[:, :1]), logits[:, :1], rtol=0, atol=0)
        self.assertEqual(module(query[:, :0], boxes[:, :0], logits[:, :0], corners[:, :0]).shape, (2, 0, 3))

    def test_cnf_odd_spatial_sizes_and_all_config_guards(self):
        module = CrossScaleNeighborhoodFusion(channels=32, dim=16)
        activate(module)
        result = module(torch.randn(2, 32, 21, 19), torch.randn(2, 32, 11, 10))
        self.assertEqual(result.shape, (2, 32, 21, 19))
        self.assertTrue(torch.isfinite(result).all())
        for dim, scale, chunk in ((8, .1, 32), (32, float('nan'), 32), (32, 0, 32), (32, .1, 0)):
            with self.assertRaises(ValueError):
                CrossScaleNeighborhoodFusion(dim=dim, scale=scale, chunk=chunk)
        for changes in ({'box_local_extension': 'unknown'}, {'box_local_dim': 32}, {'night_probe': 'none'},
                        {'allow_aqs_l2_local_combo': True}, {'box_extension_scale': 0}):
            cfg = config('DDB'); cfg.yaml_cfg['DFINE'].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError): _ = cfg.model

    def test_checkpoint_settings_full_640_and_deploy(self):
        for tag in VARIANTS:
            with self.subTest(tag=tag):
                cfg = config(tag); cfg.yaml_cfg['eval_spatial_size'] = [640, 640]
                model = cfg.model.eval(); activate(extension(model))
                restored = copy.deepcopy(model)
                serialized = io.BytesIO(); torch.save(model.state_dict(), serialized); serialized.seek(0)
                restored.load_state_dict(torch.load(serialized, weights_only=True), strict=True)
                with torch.no_grad():
                    images = torch.rand(1, 3, 640, 640)
                    expected, actual = model(images), restored(images)
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                        self.assertTrue(torch.isfinite(actual[key]).all())
                    deployed = copy.deepcopy(model).deploy()
                    actual = deployed(images)
                    for key in expected:
                        self.assertEqual(actual[key].shape, expected[key].shape)
                        self.assertTrue(torch.isfinite(actual[key]).all())
                    # Original reparameterized encoder can change TopK order on
                    # nearly tied random untrained features. Isolate the NEW
                    # hooks on identical inputs rather than loosen tolerances.
                    if tag == 'SCR':
                        raw = [torch.randn(1, c, size, size) for c, size in zip(model.encoder.in_channels, (80, 40, 20))]
                        for old, new in zip(model.encoder(raw), deployed.encoder(raw)):
                            torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-5)
                    features = [torch.randn(1, 256, size, size) for size in (80, 40, 20)]
                    expected, actual = model.decoder(features), deployed.decoder(features)
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-5)
                module = extension(model)
                invalid = copy.deepcopy(module.state_dict()); invalid['settings'][3] += 1
                with self.assertRaisesRegex(RuntimeError, 'settings differ'):
                    module.load_state_dict(invalid, strict=True)

    def test_launch_and_sync_dry_run_and_no_overwrite(self):
        launcher = ROOT/'tools/diagnostics/run_scbs_boxlocal_five_110.sh'
        sync = ROOT/'tools/diagnostics/sync_scbs_boxlocal_five.sh'
        for script in (launcher, sync): subprocess.run(['bash', '-n', str(script)], check=True)
        with tempfile.TemporaryDirectory(prefix='boxlocal-five-tests-') as folder:
            env = dict(os.environ, SCBS_BOXL_ROOT=folder, SCBS_BOXL_CONDA='/no/conda')
            result = subprocess.run(['bash', str(launcher), 'all', 'run2', '--wait-gpu', '--dry-run'], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count('训练：torchrun'), 5)
            self.assertIn('--seed=0', result.stdout)
            self.assertIn('export_query_diagnostics=false', result.stdout)
            self.assertEqual(list(Path(folder).iterdir()), [])
            output = Path(folder)/output_name('SCR', 0, 'run2'); output.mkdir()
            (output/'sentinel').write_text('existing result')
            result = subprocess.run(['bash', str(launcher), 'all', 'run2', '--dry-run'], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((output/'sentinel').read_text(), 'existing result')
        result = subprocess.run(['bash', str(sync), '--dry-run'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('不连接服务器', result.stdout)
        files = (ROOT/'tools/diagnostics/scbs_boxlocal_five_sync_files.txt').read_text().splitlines()
        self.assertTrue(all((ROOT/file).is_file() for file in files))
        self.assertIn(BOX_BASELINE, files)
        self.assertNotIn('configs/dfine/dfine_s_scbs_hrw_110.yml', files)
        self.assertNotIn('src/zoo/dfine/dfine_criterion.py', files)

    def test_sync_missing_base_bootstrap_and_preserve_existing(self):
        sync = ROOT/'tools/diagnostics/sync_scbs_boxlocal_five.sh'
        source = sync.read_text()
        code = re.findall(r"<<'PY'\n(.*?)\nPY", source, re.S)[0]
        tree = ast.parse(code)
        required = next(ast.literal_eval(node.value) for node in tree.body
                        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'required' for t in node.targets))
        original = (ROOT/BOX_BASELINE).read_bytes()
        sha = hashlib.sha256(original).hexdigest()
        iterdir = Path.iterdir
        with tempfile.TemporaryDirectory(prefix='boxlocal-sync-fixture-') as folder:
            remote = Path(folder)
            for name in required:
                file = remote/name; file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text('')
            probes = 'src/zoo/dfine/night_probes.py'
            (remote/probes).write_bytes((ROOT/probes).read_bytes())
            seed = remote/BOX_BASELINE
            def preflight():
                with patch.object(sys, 'argv', ['-', str(remote), '0', sha]), \
                     patch.object(Path, 'iterdir', lambda path: iter(()) if path == Path('/proc') else iterdir(path)), \
                     redirect_stdout(io.StringIO()):
                    exec(compile(code, str(sync), 'exec'), {})
            preflight()  # Missing night_n4 is allowed; the check itself is read-only.
            self.assertFalse(seed.exists())
            bootstrap = ['rsync', '-a', '--relative', '--ignore-existing',
                         './'+BOX_BASELINE, str(remote)+'/']
            subprocess.run(bootstrap, cwd=ROOT, check=True, capture_output=True)
            self.assertEqual(seed.read_bytes(), original)
            preflight()
            os.utime(seed, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
            before = seed.stat().st_mtime_ns
            subprocess.run(bootstrap, cwd=ROOT, check=True, capture_output=True)
            self.assertEqual(seed.stat().st_mtime_ns, before)
            # Main transfer's anchored exclude also protects the old config.
            self.assertIn('--exclude="/$box_seed"', source)
            changed = original+b'\n# server-owned configuration\n'
            seed.write_bytes(changed)
            with self.assertRaisesRegex(SystemExit, '保留原文件'):
                preflight()
            self.assertEqual(seed.read_bytes(), changed)
            subprocess.run(bootstrap, cwd=ROOT, check=True, capture_output=True)
            self.assertEqual(seed.read_bytes(), changed)
            subprocess.run(['rsync', '-a', '--relative', '--files-from',
                            str(ROOT/'tools/diagnostics/scbs_boxlocal_five_sync_files.txt'),
                            '--exclude=/' + BOX_BASELINE, str(ROOT)+'/', str(remote)+'/'],
                           cwd=ROOT, check=True, capture_output=True)
            self.assertEqual(seed.read_bytes(), changed)
            self.assertFalse((remote/'Users').exists())
            self.assertTrue((remote/'tools/diagnostics/run_scbs_boxlocal_five_110.sh').is_file())
            seed.unlink(); seed.symlink_to(remote/'missing-link-target')
            with self.assertRaisesRegex(SystemExit, '符号链接'):
                preflight()
            seed.unlink()
            (remote/'train.py').unlink()
            (remote/'tools/diagnostics/wait_for_gpu.py').unlink()
            with self.assertRaises(SystemExit) as failure:
                preflight()
            self.assertIn('train.py', str(failure.exception))
            self.assertIn('wait_for_gpu.py', str(failure.exception))

    def test_sync_post_transfer_checks_launcher_and_all_files(self):
        sync = ROOT/'tools/diagnostics/sync_scbs_boxlocal_five.sh'
        code = re.findall(r"<<'VERIFY_PY'\n(.*?)\nVERIFY_PY", sync.read_text(), re.S)[0]
        launcher = 'tools/diagnostics/run_scbs_boxlocal_five_110.sh'
        manifest = 'tools/diagnostics/scbs_boxlocal_five_sync_files.txt'
        with tempfile.TemporaryDirectory(prefix='boxlocal-sync-postcheck-') as folder:
            remote = Path(folder)
            subprocess.run(['rsync', '-a', '--relative', '--files-from', str(ROOT/manifest),
                            str(ROOT)+'/', str(remote)+'/'], check=True, capture_output=True)
            def postcheck():
                with patch.object(sys, 'argv', ['-', str(remote)]), redirect_stdout(io.StringIO()) as output:
                    exec(compile(code, str(sync), 'exec'), {})
                return output.getvalue()
            self.assertIn('启动脚本已核对', postcheck())
            (remote/launcher).unlink()
            with self.assertRaises(SystemExit) as failure:
                postcheck()
            self.assertIn(launcher, str(failure.exception))
            self.assertIn('不能开始训练', str(failure.exception))
            (remote/launcher).write_text('')
            with self.assertRaises(SystemExit):
                postcheck()

    def test_result_collection_definition_and_summary(self):
        with tempfile.TemporaryDirectory(prefix='boxlocal-five-results-') as folder:
            output = Path(folder); evaluation = output/'diagnostics_eval/evaluation'
            (evaluation/'confusion_matrix').mkdir(parents=True)
            (evaluation/'validation_metrics.json').write_text(json.dumps(dict(coco_eval_bbox=[.56, .73], bbox_per_class={'write': {'AP': .5}})))
            matrix = [[10, 1, 2, 3], [4, 20, 5, 6], [7, 8, 30, 9], [11, 12, 13, 0]]
            (evaluation/'confusion_matrix/confusion_matrix.json').write_text(json.dumps(dict(matrix_counts=matrix)))
            (output/'console.log').write_text('Model FLOPs:25.50 GFLOPS MACs:12.75 GMACs Params:10250091\n')
            result = collect_result('SCR', output, output/'best_stg2.pth')
            self.assertAlmostEqual(result['recall'], 60/105)
            self.assertAlmostEqual(result['precision'], 60/123)
            self.assertEqual((result['write_to_read'], result['read_to_write']), (8, 5))
            self.assertEqual(result['params'], 10250091)
            write_summary(output, [result], ['SCR'], 'complete')
            self.assertIn('| SCR |', (output/'summary.md').read_text())


if __name__ == '__main__':
    unittest.main(verbosity=2)
