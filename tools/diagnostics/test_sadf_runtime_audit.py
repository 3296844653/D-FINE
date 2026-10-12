"""Synthetic audit tests. No real dataset/checkpoint/GPU or AP claims required."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from PIL import Image

from tools.diagnostics.audit_sadf_runtime import (
    VARIANTS, candidate_record, gt_annotations, load_configuration, main,
    pair_rows, roi_bounds, roi_statistics, sadf_intervention, scalar_maps,
    sha256, size_groups, summarize_rows,
)
from src.zoo.dfine.scale_aligned_detail_fusion import ScaleAlignedDetailFusion


def module_inputs():
    torch.manual_seed(54)
    module = ScaleAlignedDetailFusion(32, 16, use_checkpoint=False).eval()
    with torch.no_grad():
        module.restore.weight.normal_(std=.03)
        for alignment in module.alignments:
            alignment.offset.predict.weight.normal_(std=.03)
    values = [torch.randn(2,32,8,10), torch.randn(2,32,8,10),
              torch.randn(2,32,4,5), torch.randn(2,32,2,3)]
    return module, values


def record(annotation_id=1, covered=True):
    return dict(image_id=2, annotation_id=annotation_id, category_id=2,
                class_name="write", box_xyxy=[10.,20.,30.,70.], file_name="x.jpg",
                original_area=1000., size_groups=["small"], ignored=False,
                prediction={"correct_high_candidate": covered}, runtime={"relative_update": .1})


class SADFRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_capture_exact_predictions_no_state_or_gradient_changes(self):
        module, values = module_inputs()
        state = {k: v.clone() for k,v in module.state_dict().items()}
        with torch.inference_mode():
            expected = module(*values)
            capture = {}
            with sadf_intervention(module, "full", capture):
                actual = module(*values)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertIn("weight_p5", capture)
                self.assertTrue(all(v.shape == (2,8,10) and v.device.type == "cpu"
                                    and not v.requires_grad for v in capture.values()))
                maps = dict(capture)
            self.assertEqual(capture, {})
        self.assertNotIn("forward", module.__dict__)
        for name, value in state.items():
            torch.testing.assert_close(module.state_dict()[name], value, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in module.parameters()))
        self.assertTrue(all(torch.isfinite(v).all() for v in maps.values()))

    def test_interventions_restore_flags_even_after_exception(self):
        module, values = module_inputs()
        with torch.inference_mode():
            full = module(*values)
            for variant in VARIANTS:
                with self.subTest(variant=variant):
                    capture = {}
                    with sadf_intervention(module, variant, capture):
                        actual = module(*values)
                        if variant == "bypass":
                            torch.testing.assert_close(actual, values[1], rtol=0, atol=0)
                        if variant == "no_alignment":
                            self.assertEqual(float(capture["offset_p4_norm"].max()), 0.)
                        if variant == "no_detail":
                            self.assertEqual(float(capture["gated_detail_energy"].max()), 0.)
                        if variant == "equal_scales":
                            torch.testing.assert_close(capture["weight_p3"],
                                                       torch.full((2,8,10), 1/3))
                    self.assertTrue(module.enable_alignment and module.enable_detail and module.enable_scale_selection)
                    torch.testing.assert_close(module(*values), full, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "intentional"):
                with sadf_intervention(module, "no_detail"):
                    raise ValueError("intentional")
        self.assertTrue(module.enable_detail)
        self.assertNotIn("forward", module.__dict__)
        with self.assertRaises(RuntimeError):
            with sadf_intervention(module, "full"):
                pass

    def test_roi_original_coordinates_and_true_relative_residual(self):
        self.assertEqual(roi_bounds([40,20,80,60], (200,100), (10,20)), (4,2,8,6))
        before = torch.ones(1,4,10,20) * 2
        after = before + 1
        aux = {"detail": torch.ones(1,4,10,20)*3,
               "context_delta": torch.ones(1,4,10,20)*4,
               "detail_gate": torch.ones(1,4,10,20)*.2,
               "context_gate": torch.ones(1,4,10,20)*.5,
               "scale_weights": torch.ones(1,3,10,20)/3,
               "detail_weights": torch.ones(1,2,10,20)/2,
               "offset_p4": torch.ones(1,2,10,20)*2,
               "offset_p5": torch.zeros(1,2,10,20)}
        stats = roi_statistics(scalar_maps(before, after, aux, 2), 0,
                               [40,20,80,60], (200,100))
        self.assertAlmostEqual(stats["relative_update"], .5)
        self.assertAlmostEqual(stats["gated_detail_rms"], .6, places=6)
        self.assertAlmostEqual(stats["gated_context_rms"], 2.)
        self.assertAlmostEqual(stats["offset_p4_norm"], 2*2**.5, places=6)
        self.assertEqual(stats["offset_p4_near_bound_fraction"], 1.)
        self.assertEqual(stats["roi_p3_cells"], 16)
        self.assertEqual(size_groups(32**2), ["small", "medium"])
        self.assertEqual(size_groups(96**2), ["medium", "large"])

    def test_candidate_coverage_uses_correct_selected_pair_not_top1_or_TP(self):
        prediction = {"boxes": torch.tensor([[0.,0.,10.,10.],[0.,0.,10.,10.]]),
                      "labels": torch.tensor([1,2]), "scores": torch.tensor([.9,.6])}
        row = candidate_record(prediction, [0.,0.,10.,10.], 2)
        self.assertTrue(row["correct_high_candidate"])
        self.assertEqual(row["best_overlap_prediction"]["class_id"], 1)
        self.assertEqual(row["best_correct_overlap_prediction"]["class_id"], 2)
        prediction["scores"][1] = .49
        self.assertFalse(candidate_record(prediction, [0.,0.,10.,10.], 2)["correct_high_candidate"])
        empty = {"boxes": torch.empty(0,4), "labels": torch.empty(0,dtype=torch.long), "scores": torch.empty(0)}
        self.assertFalse(candidate_record(empty, [0.,0.,10.,10.], 2)["correct_high_candidate"])

    def test_pair_identity_and_group_denominators(self):
        before = [record(i, covered) for i, covered in enumerate((True,False,True,False), 1)]
        after = [record(i, covered) for i, covered in enumerate((False,True,True,False), 1)]
        paired = pair_rows(before, after)
        self.assertEqual([r["coverage_transition"] for r in paired],
                         ["lost", "gained", "both_covered", "both_uncovered"])
        after[-1]["ignored"] = True
        groups = summarize_rows(after)
        self.assertEqual(groups["write/all"]["GT_count"], 3)
        self.assertEqual(groups["write/all"]["correct_high_candidate_count"], 2)
        bad = copy.deepcopy(before); bad[0]["original_area"] += 1
        with self.assertRaises(ValueError): pair_rows(bad, paired)
        with self.assertRaises(ValueError): pair_rows(before+before[:1], paired)

    def test_full_synthetic_validation_and_preserved_inputs(self):
        with tempfile.TemporaryDirectory(prefix="SADF-audit-test-") as folder:
            root = Path(folder)
            image_dir = root / "images"; image_dir.mkdir()
            images = [{"id":1,"file_name":"one.jpg","width":128,"height":96},
                      {"id":2,"file_name":"two.jpg","width":160,"height":128}]
            for info in images:
                Image.new("RGB", (info["width"], info["height"]), (80,120,100)).save(image_dir / info["file_name"])
            annotations = [
                {"id":1,"image_id":1,"category_id":2,"bbox":[16,16,32,48],"area":1536,"iscrowd":0},
                {"id":2,"image_id":1,"category_id":1,"bbox":[64,16,24,24],"area":576,"iscrowd":0},
                {"id":3,"image_id":2,"category_id":0,"bbox":[12,12,64,64],"area":4096,"iscrowd":0},
                {"id":4,"image_id":2,"category_id":2,"bbox":[80,20,64,80],"area":5120,"iscrowd":0}]
            annotation_path = root / "annotations.json"
            annotation_path.write_text(json.dumps({"images":images,"annotations":annotations,
                "categories":[{"id":0,"name":"hand-raising"},{"id":1,"name":"read"},{"id":2,"name":"write"}]}))
            configs = []
            for name in ("dfine_s_scbs_hrw_110.yml", "dfine_s_scbs_hrw_110_SADF.yml"):
                path = root / name
                base = {"__include__":[str(ROOT / "configs/dfine" / name)],
                        "eval_spatial_size":[128,128], "HGNetv2":{"pretrained":False},
                        "val_dataloader":{"total_batch_size":2,"num_workers":0,
                          "dataset":{"img_folder":str(image_dir),"ann_file":str(annotation_path),
                            "transforms":{"ops":[{"type":"Resize","size":[128,128]},
                              {"type":"ConvertPILImage","dtype":"float32","scale":True}]}}}}
                path.write_text(yaml.safe_dump(base))
                configs.append(path)
            torch.manual_seed(6); baseline = load_configuration(configs[0], 2, 0).model.eval()
            torch.manual_seed(6); sadf = load_configuration(configs[1], 2, 0).model.eval()
            with torch.no_grad(): sadf.encoder.sadf.restore.weight.normal_(std=.02)
            baseline_path, sadf_path = root / "run2.pth", root / "SADF.pth"
            for model, path in ((baseline,baseline_path),(sadf,sadf_path)):
                torch.save({"model":model.state_dict(),"ema":{"module":model.state_dict()},"last_epoch":109}, path)
            digests = {p:sha256(p) for p in (baseline_path,sadf_path,annotation_path)}
            output = root / "SADF_audit"
            args = ["--config",str(configs[1]),"--baseline-config",str(configs[0]),
                    "--checkpoint",str(sadf_path),"--baseline-checkpoint",str(baseline_path),
                    "--output-dir",str(output),"--device","cpu","--batch-size","2","--workers","0",
                    "--expected-run2-sha256",digests[baseline_path],
                    "--expected-annotation-sha256",digests[annotation_path],
                    "--variants","full","no_alignment","bypass"]
            main(args+["--check-only"])
            self.assertFalse(output.exists())
            main(args)
            self.assertTrue(json.loads((output / "complete.json").read_text())["complete"])
            reports = json.loads((output / "SADF_interventions.json").read_text())
            self.assertEqual(set(reports), {"baseline_run2","full","no_alignment","bypass"})
            self.assertEqual(reports["full"]["GT_records"], 4)
            self.assertFalse(reports["full"]["partial_validation"])
            rows = [json.loads(line) for line in (output / "SADF_vs_Run2_gt_records.jsonl").read_text().splitlines()]
            self.assertEqual({r["annotation_id"] for r in rows}, {1,2,3,4})
            self.assertEqual(rows[0]["original_area"], 1536)
            self.assertIn("relative_update", rows[0]["runtime"])
            self.assertTrue((output / "SADF_runtime_examples.html").is_file())
            self.assertTrue(list(output.glob("SADF_maps_*.png")))
            summary = json.loads((output / "SADF_runtime_summary.json").read_text())["groups"]
            self.assertEqual(summary["write/medium"]["GT_count"], 2)
            for path, expected in digests.items(): self.assertEqual(sha256(path), expected)
            with self.assertRaises(FileExistsError): main(args)

    def test_launcher_syntax_and_sync_allowlist(self):
        files = (ROOT / "tools/diagnostics/scbs_SADF_runtime_audit_sync_files.txt").read_text().splitlines()
        self.assertTrue(files)
        for filename in files:
            self.assertTrue(filename.startswith("tools/diagnostics/"))
            self.assertTrue((ROOT / filename).is_file(), filename)
        for name in ("run_scbs_SADF_runtime_audit.sh", "sync_scbs_SADF_runtime_audit.sh"):
            subprocess.run(["bash","-n",str(ROOT / "tools/diagnostics" / name)], check=True)
        launcher = (ROOT / "tools/diagnostics/run_scbs_SADF_runtime_audit.sh").read_text()
        self.assertNotIn("train.py", launcher)
        self.assertNotIn("torchrun --", launcher)
        self.assertIn("--source ema", launcher)
        self.assertIn("tee", launcher)


if __name__ == "__main__":
    unittest.main(verbosity=2)
