"""Implementation/fixture checks, NOT formal GQC-only detection results."""
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
from zipfile import ZipFile

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.zoo.dfine.behavior_refinement import GeometryAwareQueryCompetition
from tools.diagnostics.scbs_gqc_only import (
    BASELINE, CONFIG, SETTINGS, collect_result, completed_training, output_name, report, validate_config,
)


def config(name=CONFIG, size=128):
    with patch('src.core.yaml_config.load_config', lambda p: load_config(p, cfg={})):
        cfg = YAMLConfig(str(ROOT/name))
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    cfg.yaml_cfg['eval_spatial_size'] = [size, size]
    return cfg


def targets(device='cpu', empty=False):
    return [dict(labels=torch.tensor([0, 1, 2], device=device), boxes=torch.tensor(
        [[.2, .3, .2, .2], [.5, .5, .3, .3], [.8, .7, .2, .2]], device=device))
        if i == 0 and not empty else dict(labels=torch.empty(0, dtype=torch.long, device=device),
                                         boxes=torch.empty(0, 4, device=device)) for i in range(2)]


def copy_output(value):
    if isinstance(value, dict): return {key: copy_output(item) for key, item in value.items()}
    if isinstance(value, list): return [copy_output(item) for item in value]
    return value


def activate(module):
    for name, parameter in module.named_parameters():
        if name.endswith('up.weight'):
            torch.nn.init.normal_(parameter, std=.03)


class GQCOnlyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(615)

    def test_protocol_no_aqs_boxlocal_baseline_weights_rng_and_zero_start(self):
        validate_config()
        base_cfg, cfg = config(BASELINE), config()
        expected, actual = copy.deepcopy(base_cfg.yaml_cfg), copy.deepcopy(cfg.yaml_cfg)
        expected['DFINETransformer'].update(SETTINGS)
        for item in (expected, actual): item.pop('__include__', None); item.pop('output_dir', None)
        self.assertEqual(expected, actual)
        torch.manual_seed(616); base = base_cfg.model; rng = torch.get_rng_state().clone()
        torch.manual_seed(616); model = cfg.model
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for name, value in base.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        self.assertFalse(model.decoder.use_aqs_refine)
        self.assertTrue(model.decoder.allow_gqc_standalone)
        self.assertEqual(model.night_probe, 'none')
        self.assertIsNone(getattr(model.decoder.decoder, 'night_local_adapter', None))
        self.assertFalse(any('threshold_logit' in name for name in model.state_dict()))
        branch = model.decoder.behavior_refinement
        self.assertIsInstance(branch, GeometryAwareQueryCompetition)
        self.assertEqual(sum(p.numel() for p in branch.parameters()), 54341)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 10277350)
        self.assertEqual(cfg.yaml_cfg['DFINECriterion'], base_cfg.yaml_cfg['DFINECriterion'])
        images = torch.rand(2, 3, 128, 128)
        for training in (False, True):
            base.train(training); model.train(training); gt = targets() if training else None
            with torch.no_grad():
                torch.manual_seed(617); old = base(images, gt)
                torch.manual_seed(617); new = model(images, gt)
            for key in ('pred_logits', 'pred_boxes'):
                torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)
        ids = [id(p) for group in cfg.optimizer.param_groups for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(id(p) in ids for p in branch.parameters()))

    def check_amp(self, device, dtype):
        cfg = config(); model = cfg.model.to(device).train(); branch = model.decoder.behavior_refinement
        criterion = cfg.criterion.to(device)
        images = torch.rand(2, 3, 128, 128, device=device)
        for empty in (False, True):
            gt = targets(device, empty); model.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=dtype):
                output = model(images, gt); losses = criterion(copy_output(output), gt)
            self.assertTrue(all(torch.isfinite(value).all() for value in losses.values()))
            sum(losses.values()).backward()
            self.assertGreater(branch.class_head.up.weight.grad.abs().sum().item(), 0)
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad or (empty and name == 'decoder.denoising_class_embed.weight'): continue
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            activate(branch)
        old = branch.class_head.up.weight.detach().clone()
        cfg.optimizer.step(); cfg.ema.update(model)
        self.assertGreater((old-branch.class_head.up.weight).abs().sum().item(), 0)
        config().model.load_state_dict(cfg.ema.module.state_dict(), strict=True)

    def test_cpu_amp_original_loss_gradients_empty_gt_optimizer_and_ema(self):
        self.check_amp('cpu', torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable locally; verify FP16 on server')
    def test_cuda_fp16_training(self):
        self.check_amp('cuda', torch.float16)

    def test_final_score_only_after_lqe_and_dn_isolation(self):
        cfg = config(); model = cfg.model.train(); branch = model.decoder.behavior_refinement; activate(branch)
        seen, lqe = [], []
        def observe(m, args, kwargs, result):
            seen.append(([x.detach().clone() for x in args], result.detach().clone()))
        handle = branch.register_forward_hook(observe, with_kwargs=True)
        lqe_handle = model.decoder.decoder.lqe_layers[2].register_forward_hook(lambda m,a,r: lqe.append(r.detach().clone()))
        try:
            images, gt = torch.rand(2, 3, 128, 128), targets()
            with torch.no_grad():
                torch.manual_seed(618); actual = model(images, gt)
                branch.enabled = False
                torch.manual_seed(618); old = model(images, gt)
        finally: handle.remove(); lqe_handle.remove()
        self.assertEqual(seen[0][0][0].shape, (2, 300, 256))
        count = actual['dn_meta']['dn_num_split'][0]
        torch.testing.assert_close(seen[0][0][2], lqe[0][:, count:], rtol=0, atol=0)
        torch.testing.assert_close(actual['pred_logits'], seen[0][1], rtol=0, atol=0)
        torch.testing.assert_close(actual['pred_boxes'], old['pred_boxes'], rtol=0, atol=0)
        self.assertGreater((actual['pred_logits']-old['pred_logits']).abs().sum().item(), 0)
        for key in ('aux_outputs', 'dn_outputs', 'enc_aux_outputs'):
            for new, previous in zip(actual[key], old[key]):
                for field in ('pred_boxes', 'pred_logits'):
                    torch.testing.assert_close(new[field], previous[field], rtol=0, atol=0)
        for aux in actual['aux_outputs']:
            torch.testing.assert_close(aux['teacher_logits'], actual['pred_logits'], rtol=0, atol=0)
        self.assertEqual(set(cfg.criterion(copy_output(actual), gt)), set(cfg.criterion(copy_output(old), gt)))

    def test_explicit_standalone_only_and_legacy_guards(self):
        for changes in ({'allow_gqc_standalone': False}, {'allow_gqc_standalone': 1},
                        {'use_aqs_refine': True}, {'behavior_refinement_module': 'none'},
                        {'behavior_refinement_module': 'ClassEvidenceTransport'}, {'use_query_objectness': True}):
            cfg = config(); cfg.yaml_cfg['DFINETransformer'].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError): _ = cfg.model
        for changes in ({'night_probe': 'decoder_l2_local'}, {'box_local_extension': 'ReliableQueryCompetition'},
                        {'finegrain_trial': 'PartInteractionL2'}):
            cfg = config(); cfg.yaml_cfg['DFINE'].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError): _ = cfg.model
        old = config('configs/dfine/dfine_s_scbs_hrw_110_AQS_GeometryAwareQueryCompetition.yml').model
        self.assertTrue(old.decoder.use_aqs_refine)
        self.assertFalse(old.decoder.allow_gqc_standalone)
        self.assertEqual(sum(p.numel() for p in old.decoder.behavior_refinement.parameters()), 54341)

    def test_checkpoint_640_deploy_and_profiler(self):
        from calflops import calculate_flops
        model = config(size=640).model.eval(); activate(model.decoder.behavior_refinement)
        saved = io.BytesIO(); torch.save(model.state_dict(), saved); saved.seek(0)
        restored = config(size=640).model.eval()
        restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
        with torch.no_grad():
            image = torch.rand(1, 3, 640, 640)
            expected, actual = model(image), restored(image)
            for key in expected: torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            self.assertEqual(actual['pred_logits'].shape, (1, 300, 3))
            deployed = copy.deepcopy(model).deploy()
            self.assertEqual(sum(p.numel() for p in deployed.parameters()), 10232542)
            features = [torch.randn(1, 256, size, size) for size in (80, 40, 20)]
            expected, actual = model.decoder(features), deployed.decoder(features)
            for key in expected: torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-5)
            flops, _, _ = calculate_flops(deployed, input_shape=(1, 3, 640, 640),
                output_as_string=False, print_results=False, print_detailed=False)
            self.assertGreater(flops, 0)
            print(f'GQC-only deploy params=10232542, calflops={flops/1e9:.6f}G', flush=True)

    def test_launch_dry_run_no_overwrite_sync_manifest_and_postcheck(self):
        launch = ROOT/'tools/diagnostics/run_scbs_gqc_only_110.sh'
        sync = ROOT/'tools/diagnostics/sync_scbs_gqc_only.sh'
        for script in (launch, sync): subprocess.run(['bash', '-n', str(script)], check=True)
        with tempfile.TemporaryDirectory(prefix='gqc-only-launch-') as folder:
            env = dict(os.environ, SCBS_GQC_ROOT=folder, SCBS_GQC_CONDA='/not/installed/conda')
            dry = subprocess.run(['bash', str(launch), 'run2', '--wait-gpu', '--seed', '1', '--dry-run'],
                                 env=env, capture_output=True, text=True)
            self.assertEqual(dry.returncode, 0, dry.stderr)
            self.assertIn(CONFIG, dry.stdout); self.assertIn('--seed=1', dry.stdout)
            self.assertIn('export_query_diagnostics=false', dry.stdout)
            self.assertEqual(list(Path(folder).iterdir()), [])
            output = Path(folder)/output_name(1, 'run2'); output.mkdir(); (output/'sentinel').write_text('preserve')
            failed = subprocess.run(['bash', str(launch), 'run2', '--seed', '1', '--dry-run'], env=env, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual((output/'sentinel').read_text(), 'preserve')
        dry = subprocess.run(['bash', str(sync), '--dry-run'], capture_output=True, text=True)
        self.assertEqual(dry.returncode, 0, dry.stderr); self.assertIn('不连接服务器', dry.stdout)
        manifest = ROOT/'tools/diagnostics/scbs_gqc_only_sync_files.txt'
        files = manifest.read_text().splitlines()
        self.assertTrue(all((ROOT/file).is_file() for file in files))
        self.assertNotIn(BASELINE, files)
        self.assertFalse(any('dataset/' in file or file.endswith(('.pth', 'night_n4.yml', 'dfine_criterion.py')) for file in files))
        code = re.findall(r"<<'VERIFY_PY'\n(.*?)\nVERIFY_PY", sync.read_text(), re.S)[0]
        with tempfile.TemporaryDirectory(prefix='gqc-only-postcheck-') as folder:
            remote = Path(folder)
            subprocess.run(['rsync', '-a', '--relative', '--files-from', str(manifest), str(ROOT)+'/', str(remote)+'/'], check=True, capture_output=True)
            def postcheck():
                with patch.object(sys, 'argv', ['-', str(remote)]), redirect_stdout(io.StringIO()) as stream:
                    exec(compile(code, str(sync), 'exec'), {})
                return stream.getvalue()
            self.assertIn('启动脚本已核对', postcheck())
            (remote/'tools/diagnostics/run_scbs_gqc_only_110.sh').unlink()
            with self.assertRaisesRegex(SystemExit, '不能开始训练'): postcheck()

    def test_completion_requires_all_110_epochs(self):
        with tempfile.TemporaryDirectory(prefix='gqc-only-completion-') as folder:
            output = Path(folder); log = output/'log.txt'
            log.write_text('\n'.join(json.dumps(dict(epoch=i)) for i in range(109)))
            with self.assertRaisesRegex(ValueError, '未记录完整'): completed_training(output)
            log.write_text('\n'.join(json.dumps(dict(epoch=i)) for i in range(110)))
            completed_training(output)

    def test_runner_orchestration_and_failure_retention_without_training(self):
        """Stub child commands only. No CUDA work or real model training."""
        import tools.diagnostics.scbs_gqc_only as runner
        resolved = validate_config()
        calls = []
        with tempfile.TemporaryDirectory(prefix='gqc-only-runner-fixture-') as folder:
            root = Path(folder)
            def execute(cmd, log, env):
                calls.append(cmd)
                log.write_text('Simulated child command, NOT actual training.\n')
                if log.name == 'console.log':
                    (log.parent/'log.txt').write_text('\n'.join(json.dumps(dict(epoch=i)) for i in range(110)))
                    (log.parent/'best_stg2.pth').write_bytes(b'fixture only, not a real checkpoint')
                else:
                    evaluation = log.parent/'diagnostics_eval/evaluation'; (evaluation/'confusion_matrix').mkdir(parents=True)
                    metrics = dict(coco_eval_bbox=[.5, .6], bbox_per_class={name: {'AP': .5} for name in ('hand-raising', 'read', 'write')})
                    (evaluation/'validation_metrics.json').write_text(json.dumps(metrics))
                    matrix = dict(row_labels=['hand-raising', 'read', 'write', 'unmatched_predictions'],
                        matrix_counts=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0]],
                        confidence_threshold=.5, iou_threshold=.5, metadata={'weight_source': 'ema'})
                    (evaluation/'confusion_matrix/confusion_matrix.json').write_text(json.dumps(matrix))
            with patch.dict(os.environ), patch.object(torch.cuda, 'is_available', return_value=True), \
                 patch('tools.diagnostics.wait_for_gpu.wait_for_gpu'), patch.object(runner.socket, 'socket'), \
                 patch.object(runner, 'validate_config', return_value=resolved), \
                 patch.object(runner, 'validate_data', return_value={'fixture': True}), \
                 patch.object(runner, 'run_logged', side_effect=execute), redirect_stdout(io.StringIO()):
                runner.main(['run1', '--root', str(root), '--wait-gpu'])
            output = root/output_name(0, 'run1')
            self.assertEqual(len(calls), 2)
            self.assertNotIn('-r', calls[0]); self.assertNotIn('-t', calls[0])
            self.assertIn('--test-only', calls[1]); self.assertIn('export_query_diagnostics=false', calls[1])
            self.assertEqual(json.loads((output/'status.json').read_text())['state'], 'complete')
            self.assertTrue((output/'GQC_only_diagnostics.zip').is_file())
            self.assertFalse(json.loads((output/'experiment_inputs.json').read_text())['AQS'])
            with patch.dict(os.environ), patch.object(torch.cuda, 'is_available', return_value=True), \
                 patch('tools.diagnostics.wait_for_gpu.wait_for_gpu'), patch.object(runner.socket, 'socket'), \
                 patch.object(runner, 'validate_config', return_value=resolved), \
                 patch.object(runner, 'validate_data', return_value={'fixture': True}), \
                 patch.object(runner, 'run_logged', side_effect=RuntimeError('simulated failure')), \
                 redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'simulated failure'):
                runner.main(['run2', '--root', str(root)])
            failed = root/output_name(0, 'run2')
            self.assertTrue(json.loads((failed/'status.json').read_text())['state'].startswith('incomplete'))
            self.assertTrue((failed/'experiment_inputs.json').is_file())

    def test_real_toy_coco_best_ema_export_report_and_archive(self):
        """Runs real CPU evaluation on TWO synthetic images, NOT the SCB-S experiment."""
        from PIL import Image
        import yaml
        with tempfile.TemporaryDirectory(prefix='gqc-only-coco-') as folder:
            output = Path(folder); images = output/'images'; images.mkdir()
            data = dict(info={}, images=[], annotations=[], categories=[dict(id=i, name=name)
                for i, name in enumerate(('hand-raising', 'read', 'write'))])
            for i in (1, 2):
                Image.new('RGB', (128, 128), (i*70, 60, 80)).save(images/f'{i}.png')
                data['images'].append(dict(id=i, file_name=f'{i}.png', width=128, height=128))
                for c in range(3):
                    data['annotations'].append(dict(id=(i-1)*3+c+1, image_id=i, category_id=c,
                        bbox=[10+c*25, 20, 30, 50], area=1500, iscrowd=0))
            annotation = output/'annotations.json'; annotation.write_text(json.dumps(data))
            cfg = config(); model = cfg.model.eval(); activate(model.decoder.behavior_refinement)
            checkpoint = output/'best_stg2.pth'
            torch.save(dict(model=model.state_dict(), ema=dict(module=model.state_dict()), last_epoch=99), checkpoint)
            raw = copy.deepcopy(cfg.yaml_cfg); raw.pop('__include__', None)
            raw['val_dataloader'].update(total_batch_size=2, num_workers=0)
            raw['val_dataloader']['dataset'].update(img_folder=str(images), ann_file=str(annotation))
            for op in raw['val_dataloader']['dataset']['transforms']['ops']:
                if op['type'] == 'Resize': op['size'] = [128, 128]
            path = output/'test_config.yml'; path.write_text(yaml.safe_dump(raw))
            env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MPLBACKEND='Agg')
            result = subprocess.run([sys.executable, '-u', str(ROOT/'train.py'), '-c', str(path), '-r', str(checkpoint),
                '--test-only', '-d', 'cpu', '--seed=0', '--output-dir', str(output/'diagnostics_eval/evaluation'),
                '-u', 'export_confusion_matrix=true', 'export_query_diagnostics=false'], cwd=ROOT, env=env,
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout[-2000:]+result.stderr[-2000:])
            (output/'console.log').write_text('Synthetic fixture, NOT SCB-S training.\n')
            measured = collect_result(output, checkpoint)
            self.assertEqual(measured['weight_source'], 'ema')
            report(output, measured)
            self.assertTrue((output/'summary.md').is_file())
            with ZipFile(output/'GQC_only_diagnostics.zip') as archive:
                names = archive.namelist()
                self.assertTrue(any(name.endswith('validation_metrics.json') for name in names))
                self.assertFalse(any(name.endswith(('.pth', '.pt')) for name in names))
                self.assertFalse(any('off_eval' in name for name in names))


if __name__ == '__main__':
    unittest.main(verbosity=2)
