"""Synthetic regression checks, not effectiveness/repeatability evidence."""

import copy
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.zoo.dfine.night_probes import NIGHT_PROBES, AdaptiveROITokens, BoxLocalQueryAdapter


def config(index=None, size=128):
    name = "dfine_s_scbs_hrw_110.yml" if index is None else f"dfine_s_scbs_hrw_110_night_n{index}.yml"
    with patch("src.core.yaml_config.load_config", lambda path: load_config(path, cfg={})):
        cfg = YAMLConfig(str(ROOT / "configs/dfine" / name))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["eval_spatial_size"] = [size, size]
    return cfg


def targets(device="cpu", empty=False):
    result = [dict(labels=torch.empty(0, dtype=torch.long, device=device),
                   boxes=torch.empty(0, 4, device=device)) for _ in range(2)]
    if not empty:
        result[0] = dict(labels=torch.tensor([0, 1, 2], device=device),
                         boxes=torch.tensor([[.2, .3, .2, .2], [.5, .5, .3, .3], [.8, .7, .2, .2]], device=device))
    return result


def branch_parameters(model):
    return {k: v for k, v in model.named_parameters() if "night_" in k}


def activate(model):
    """Simulate nonzero trained residuals; all baseline parameters unchanged."""
    for name, module in model.named_modules():
        if "night_" not in name:
            continue
        if name.endswith(".output") or name.endswith(".fuse.3") or name.endswith(".contrast.3") or name.endswith(".score.3"):
            torch.nn.init.normal_(module.weight, std=.02)


class NightProbeTests(unittest.TestCase):
    def test_configs_single_variable_and_identity_baseline_rng(self):
        base_cfg = config()
        torch.manual_seed(17)
        baseline = base_cfg.model.eval()
        state = baseline.state_dict()
        rng = torch.get_rng_state().clone()
        images = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            expected = baseline(images)
        for index, mode in enumerate(NIGHT_PROBES, 1):
            with self.subTest(mode=mode):
                cfg = config(index)
                old, new = copy.deepcopy(base_cfg.yaml_cfg), copy.deepcopy(cfg.yaml_cfg)
                for item in (old, new):
                    item.pop("__include__", None)
                    item.pop("output_dir", None)
                old["DFINE"]["night_probe"] = mode
                if index == 5:
                    old["HGNetv2"]["return_idx"] = [0, 1, 2, 3]
                self.assertEqual(old, new)
                torch.manual_seed(17)
                model = cfg.model.eval()
                torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
                for key, value in state.items():
                    torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
                self.assertFalse(model.decoder.use_aqs_refine)
                self.assertFalse(cfg.criterion.use_pairwise_ce)
                self.assertFalse(cfg.criterion.use_class_margin)
                self.assertTrue(branch_parameters(model))
                with torch.no_grad():
                    actual = model(images)
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
                optimizer = cfg.optimizer
                ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
                self.assertEqual(len(ids), len(set(ids)))
                self.assertTrue(all(id(p) in ids for p in branch_parameters(model).values()))
                print(mode, "added params:", sum(p.numel() for p in branch_parameters(model).values()), flush=True)

    def test_roi_sampling_bounds_detached_boxes_and_query_independence(self):
        adapter = BoxLocalQueryAdapter(32, 16)
        query = torch.randn(2, 5, 32, requires_grad=True)
        boxes = torch.tensor([.5, .5, .6, .7]).expand(2, 5, 4).clone().requires_grad_()
        p3 = torch.randn(2, 32, 12, 16, requires_grad=True)
        torch.nn.init.normal_(adapter.fuse[-1].weight, std=.1)
        torch.nn.init.normal_(adapter.sampler.offset.weight, std=.1)
        relative = adapter.sampler.relative_locations(query)
        self.assertLessEqual(float(relative.detach().abs().max()), .490001)
        actual = adapter(query, boxes, p3)
        actual.square().sum().backward()
        self.assertIsNone(boxes.grad)
        self.assertGreater(adapter.sampler.offset.weight.grad.abs().sum().item(), 0)
        for name, p in adapter.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        changed = query.detach().clone(); changed[:, 0] += 10
        torch.testing.assert_close(adapter(changed, boxes, p3)[:, 1:], actual[:, 1:], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            AdaptiveROITokens(32, 16, 8)(query, boxes, p3)

    def test_training_dn_amp_empty_gt_and_branch_gradients(self):
        for index, mode in enumerate(NIGHT_PROBES, 1):
            with self.subTest(mode=mode):
                cfg = config(index)
                model = cfg.model.train()
                images = torch.rand(2, 3, 128, 128)
                # At zero initialization the output projection must get a real
                # original-loss gradient (not just an isolated artificial loss).
                outputs = model(images, targets())
                losses = cfg.criterion(outputs, targets())
                self.assertTrue(all(torch.isfinite(v).all() for v in losses.values()))
                sum(losses.values()).backward()
                params = branch_parameters(model)
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in params.values()))
                model.zero_grad(set_to_none=True)
                activate(model)
                for empty in (False, True):
                    model.zero_grad(set_to_none=True)
                    gt = targets(empty=empty)
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        output = model(images, gt)
                        losses = cfg.criterion(output, gt)
                    self.assertTrue(all(torch.isfinite(v).all() for v in losses.values()))
                    sum(losses.values()).backward()
                    for name, p in params.items():
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                    self.assertFalse(any("margin" in key or "pairwise_ce" in key for key in losses))
                    self.assertEqual(output["pred_logits"].shape, (2, 300, 3))

    def test_final_heads_keep_bbox_hand_or_class_order_and_dn_teacher(self):
        images, gt = torch.rand(2, 3, 128, 128), targets()
        for index in (5, 6):
            with self.subTest(index=index):
                model = config(index).model.train()
                activate(model)
                torch.manual_seed(19); actual = model(images, gt)
                model.decoder.night_probe = "none"
                torch.manual_seed(19); original = model(images, gt)
                model.decoder.night_probe = NIGHT_PROBES[index - 1]
                for key in ("pred_boxes", "pred_corners", "ref_points"):
                    torch.testing.assert_close(actual[key], original[key], atol=0, rtol=0)
                if index == 5:
                    torch.testing.assert_close(actual["pred_logits"][..., 0], original["pred_logits"][..., 0], atol=0, rtol=0)
                else:
                    delta = actual["pred_logits"] - original["pred_logits"]
                    torch.testing.assert_close(delta[..., 0], delta[..., 1], atol=1e-6, rtol=1e-6)
                    torch.testing.assert_close(delta[..., 1], delta[..., 2], atol=1e-6, rtol=1e-6)
                    self.assertTrue(torch.equal(actual["pred_logits"].argmax(-1), original["pred_logits"].argmax(-1)))
                for group in ("dn_outputs", "aux_outputs", "enc_aux_outputs"):
                    for new, old in zip(actual[group], original[group]):
                        for key in old:
                            if torch.is_tensor(old[key]):
                                torch.testing.assert_close(new[key], old[key], atol=0, rtol=0)

    def test_checkpoint_round_trip_and_deploy_decoder_pruning(self):
        for index, mode in enumerate(NIGHT_PROBES, 1):
            with self.subTest(mode=mode):
                model = config(index).model.eval(); activate(model)
                buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
                restored = config(index).model.eval()
                restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
                images = torch.rand(2, 3, 128, 128)
                with torch.no_grad():
                    expected, actual = model(images), restored(images)
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
                    deployed = copy.deepcopy(model).deploy()
                    output = deployed(images)
                    self.assertTrue(all(torch.isfinite(v).all() for v in output.values()))
                    # Folding encoder BN can reorder randomly tied proposals;
                    # compare decoder pruning with the exact SAME features.
                    feats = [torch.randn(2, 256, side, side) for side in (16, 8, 4)]
                    kwargs = {"p2_feat": torch.randn(2, 64, 32, 32)} if index == 5 else {}
                    expected, actual = model.decoder(feats, **kwargs), deployed.decoder(feats, **kwargs)
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key], atol=1e-5, rtol=1e-5)

    def test_full_resolution_and_calflops(self):
        from calflops import calculate_flops
        for index in range(1, 7):
            with self.subTest(index=index):
                model = config(index, size=640).model.eval(); activate(model)
                with torch.no_grad():
                    output = model(torch.rand(1, 3, 640, 640))
                self.assertEqual(output["pred_logits"].shape, (1, 300, 3))
                self.assertTrue(all(torch.isfinite(v).all() for v in output.values()))
                calculate_flops(copy.deepcopy(model).deploy(), input_shape=(1, 3, 640, 640), print_results=False)

    def test_single_module_guards_and_default_no_new_parameters(self):
        base = config().model
        self.assertEqual(base.night_probe, "none")
        self.assertFalse(branch_parameters(base))
        for owner, flag in (("HGNetv2", "use_afta"), ("HybridEncoder", "use_mffe"),
                            ("DFINETransformer", "use_aqs_refine")):
            cfg = config(1); cfg.yaml_cfg[owner][flag] = True
            with self.assertRaisesRegex(ValueError, "separate|combined"):
                _ = cfg.model
        cfg = config(5); cfg.yaml_cfg["HGNetv2"]["return_idx"] = [1, 2, 3]
        model = cfg.model.eval()
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "return_idx"):
            model(torch.rand(2, 3, 128, 128))

    def test_launcher_syntax_dry_run_and_no_overwrite(self):
        script = ROOT / "tools/diagnostics/run_scbs_night_sweep_110.sh"
        subprocess.run(["bash", "-n", str(script)], check=True)
        import os
        with tempfile.TemporaryDirectory(prefix="dfine-night-test-") as directory:
            env = dict(os.environ, NIGHT_OUTPUT_ROOT=directory, NIGHT_PYTHON=sys.executable)
            result = subprocess.run(["bash", str(script), "--dry-run"], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])
            out = Path(directory) / "dfine_s_scbs_hrw_epochs110_night_n1_seed0_run1"
            out.mkdir(); (out / "sentinel").touch()
            result = subprocess.run(["bash", str(script), "--dry-run", "N1"], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("已存在", result.stderr)
            self.assertTrue((out / "sentinel").is_file())

    def test_launcher_simulated_failures_continue_and_live_logs(self):
        # Isolated fake executables only: NEVER launch actual full training.
        import json
        import os
        import shutil
        with tempfile.TemporaryDirectory(prefix="dfine-night-queue-test-") as directory:
            root = Path(directory)
            (root / "tools/diagnostics").mkdir(parents=True)
            (root / "configs/dfine").mkdir(parents=True)
            (root / "bin").mkdir()
            script = root / "tools/diagnostics/run_scbs_night_sweep_110.sh"
            shutil.copyfile(ROOT / "tools/diagnostics/run_scbs_night_sweep_110.sh", script)
            (root / "train.py").touch()
            for index in (1, 2, 3):
                (root / f"configs/dfine/dfine_s_scbs_hrw_110_night_n{index}.yml").touch()
            gpu = root / "bin/nvidia-smi"
            gpu.write_text("#!/bin/sh\nprintf '42\\n'\n", encoding="utf-8"); gpu.chmod(0o755)
            ps = root / "bin/ps"
            ps.write_text("#!/bin/sh\nprintf '/usr/libexec/gnome-remote-desktop-daemon\\n'\n", encoding="utf-8")
            ps.chmod(0o755)
            conda = root / "bin/test-conda"
            conda.write_text("#!" + sys.executable + "\n" + '''
import json, sys
from pathlib import Path
a = sys.argv[1:]
if 'torchrun' in a:
    out = Path(a[a.index('--output-dir')+1]); out.mkdir(parents=True, exist_ok=True)
    print('SIMULATED live training/evaluation output', flush=True)
    if '--test-only' in a:
        if 'night_n3_' in str(out): sys.exit(7)
        sys.exit(0)
    if 'night_n2_' in str(out): sys.exit(5)
    (out / 'best_stg2.pth').touch()
    sys.exit(0)
a = a[a.index('python')+1:]
name = Path(a[0]).name
if name == 'test_night_probes.py': sys.exit(0)
if name == 'analyze_detection_diagnostics.py':
    Path(a[a.index('--output-dir')+1]).mkdir(parents=True, exist_ok=True)
    sys.exit(0)
op = a[1]
if op == 'preflight': sys.exit(0)
if op == 'snapshot':
    (Path(a[3]) / 'experiment_inputs.json').write_text('{}')
    sys.exit(0)
if op == 'record':
    item = dict(index=int(a[2]), output=a[3], status=a[5], code=int(a[6]))
    with (Path(a[4]) / 'results.jsonl').open('a') as f:
        f.write(json.dumps(item)+'\\n')
    sys.exit(0)
sys.exit(9)
''', encoding="utf-8")
            conda.chmod(0o755)
            env = dict(os.environ, NIGHT_OUTPUT_ROOT=str(root / "output"),
                       NIGHT_CONDA=str(conda), PATH=str(root / "bin") + os.pathsep + os.environ["PATH"],
                       NIGHT_EXPORT="1", NIGHT_STOP_ON_ERROR="0", NIGHT_RUN_ID="1")
            result = subprocess.run(["bash", str(script), "N1", "N2", "N3"], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            records = root / "output/night_sweep_epochs110_seed0_run1/results.jsonl"
            self.assertTrue(records.is_file(), result.stdout + result.stderr)
            rows = [json.loads(line) for line in records.read_text().splitlines()]
            self.assertEqual([item["status"] for item in rows], ["completed", "training_failed", "diagnostics_failed"])
            self.assertEqual([item["code"] for item in rows], [0, 5, 7])
            self.assertIn("SIMULATED live", result.stdout)
            for index in (1, 2, 3):
                out = root / f"output/dfine_s_scbs_hrw_epochs110_night_n{index}_seed0_run1"
                self.assertIn("SIMULATED live", (out / "console.log").read_text())
            self.assertFalse((root / "output/dfine_s_scbs_hrw_epochs110_night_n2_seed0_run1/diagnostics_eval").exists())
            # Same reported PID, now a real compute command: reject it before
            # any new output directory, without killing or modifying anything.
            ps.write_text("#!/bin/sh\nprintf '/env/bin/python train.py\\n'\n", encoding="utf-8")
            env["NIGHT_RUN_ID"] = "2"
            result = subprocess.run(["bash", str(script), "N1"], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("仍有进程PID=42", result.stderr)
            self.assertFalse((root / "output/dfine_s_scbs_hrw_epochs110_night_n1_seed0_run2").exists())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable locally; server test required")
    def test_cuda_fp16_training_all_six(self):
        for index in range(1, 7):
            with self.subTest(index=index):
                cfg = config(index)
                model, criterion = cfg.model.cuda().train(), cfg.criterion.cuda()
                activate(model)
                images, gt = torch.rand(2, 3, 128, 128, device="cuda"), targets("cuda")
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(images, gt)
                    losses = criterion(output, gt)
                self.assertTrue(all(torch.isfinite(v).all() for v in losses.values()))
                sum(losses.values()).backward()
                for name, p in branch_parameters(model).items():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
