"""CPU tests of query export and automatic diagnostics; no detector training."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from faster_coco_eval import COCO, COCOeval_faster
from torchvision.ops import box_convert

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.yaml_utils import load_config
from src.solver.query_diagnostics import QueryDiagnosticsWriter
from src.solver.confusion_export import export_confusion_matrix
from src.solver.validator import Validator
from src.zoo.dfine.postprocessor import DFINEPostProcessor
from tools.diagnostics.analyze_detection_diagnostics import (
    analyze, candidate_status, load_coco_pr, original_gt, score_first_match,
)


def fixture(root, empty_second=False, dtype=torch.float32, no_overlap=False, absent_writing=False):
    root = Path(root)
    annotation = root / "instances_val.json"
    images = [{"id": 1, "file_name": "1.jpg", "width": 120, "height": 20}]
    if empty_second:
        images.append({"id": 2, "file_name": "2.jpg", "width": 120, "height": 20})
    cats = [{"id": 0, "name": "hand-raising"}, {"id": 1, "name": "read"}, {"id": 2, "name": "write"}]
    gt_boxes = torch.tensor([[0,0,10,10], [20,0,30,10], [40,0,50,10], [60,0,70,10], [80,0,90,10]], dtype=torch.float32)
    gt_labels = torch.tensor([0, 1, 2, 1, 2])
    if absent_writing:
        gt_labels[gt_labels == 2] = 1
    anns = [{"id": i+1, "image_id": 1, "category_id": int(c), "bbox": [float(b[0]), float(b[1]), 10., 10.], "area": 100., "iscrowd": 0} for i, (b,c) in enumerate(zip(gt_boxes, gt_labels))]
    annotation.write_text(json.dumps({"images": images, "categories": cats, "annotations": anns}))
    boxes = torch.cat([gt_boxes, torch.tensor([[100,0,110,10], [20,0,30,10]], dtype=torch.float32)])
    if no_overlap:
        boxes[4] = torch.tensor([110,0,120,10])
    scores = torch.full((7, 3), .001)
    for q, cls, score in [(0,0,.9), (1,1,.9), (2,2,.9), (3,2,.9), (4,2,.49), (5,1,.8), (6,1,.6)]:
        scores[q,cls] = score
    output = {"pred_logits": torch.logit(scores).to(dtype)[None],
              "pred_boxes": (box_convert(boxes, "xyxy", "cxcywh") / torch.tensor([120.,20.,120.,20.])).to(dtype)[None]}
    post = DFINEPostProcessor(num_classes=3, num_top_queries=7)
    config = {"val_dataloader": {"dataset": {"ann_file": str(annotation)}}}
    writer = QueryDiagnosticsWriter(root, post, .5, .5, .8, config, None)
    gts, preds, coco_results = [], [], []
    for img in images:
        target = {"image_id": torch.tensor([img["id"]]), "image_path": img["file_name"],
                  "boxes": gt_boxes.clone() if img["id"]==1 else torch.zeros((0,4)),
                  "labels": gt_labels.clone() if img["id"]==1 else torch.zeros(0,dtype=torch.long)}
        sizes = torch.tensor([[120,20]])
        writer.record_batch(output, [target], sizes, (20,120))
        result = post(output, sizes)[0]
        gts.append(target); preds.append(result)
        for box, cls, score in zip(result["boxes"], result["labels"], result["scores"]):
            coco_results.append({"image_id":img["id"], "category_id":int(cls),
                                 "bbox":box_convert(box.float(), "xyxy", "xywh").tolist(), "score":float(score)})
    writer.finish()
    coco = COCO(str(annotation)); detections = coco.loadRes(coco_results)
    evaluator = COCOeval_faster(coco, detections, "bbox")
    evaluator.evaluate(); evaluator.accumulate()
    torch.save(evaluator.eval, root / "eval.pth")
    validator = Validator(gts, preds); validator.compute_metrics(extended=True)
    export_confusion_matrix(validator,coco,root,metadata={"weight_source":"ema"})
    return annotation, output, preds[0], evaluator.eval


class DetectionDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_configuration_keeps_baseline_unchanged(self):
        root = Path(__file__).resolve().parents[2]
        base = load_config(str(root / "configs/dfine/dfine_s_scbs_hrw_132.yml"), {})
        diag = load_config(str(root / "configs/dfine/dfine_s_scbs_hrw_confusion.yml"), {})
        changed = {k for k in set(base)|set(diag) if base.get(k)!=diag.get(k)}
        self.assertEqual(changed, {"__include__", "export_confusion_matrix", "export_query_diagnostics", "query_diagnostic_conf_thresh", "query_diagnostic_iou_thresh", "query_diagnostic_neighbor_iou_thresh"})
        self.assertEqual(base["num_classes"],3); self.assertEqual(base["epochs"],132)
        self.assertEqual(base["train_dataloader"]["total_batch_size"],32)
        self.assertTrue(diag["export_query_diagnostics"]); self.assertFalse(base.get("export_query_diagnostics",False))

    def test_end_to_end_official_pr_and_duplicate_handling(self):
        with tempfile.TemporaryDirectory(prefix="det-diag-test-") as folder:
            root=Path(folder); annotation, outputs, actual, evaluation=fixture(root)
            summary=analyze(root,annotation)
            self.assertEqual(summary["GT"],5); self.assertEqual(summary["queries"],7)
            self.assertEqual(summary["validator_confusion"]["misclassifications"],1)
            self.assertEqual(summary["validator_confusion"]["unmatched_predictions"],2)
            self.assertEqual(summary["oracle_candidate_status_counts"]["overlap_exists_but_GT_class_never_top1"],1)
            self.assertEqual(summary["oracle_candidate_status_counts"]["GT_class_top1_but_below_confidence"],1)
            records=[json.loads(l) for l in (root/"analysis/prediction_records.jsonl").read_text().splitlines()]
            duplicate=next(r for r in records if r["query_id"]==6)
            self.assertGreater(duplicate["max_same_class_gt_iou"],.99)
            self.assertFalse(duplicate["diagnostic_tp"])
            raw=torch.load(root/"query_diagnostics/raw_queries/1.pt",weights_only=False)
            qids=raw["selected_flat_indices"]//3
            self.assertTrue(torch.equal(raw["selected_scores"],actual["scores"].float()))
            self.assertTrue(torch.equal(raw["pred_boxes_xyxy"][qids],actual["boxes"].float()))
            pr=json.loads((root/"analysis/coco_pr_curves.json").read_text())
            expected=np.asarray(evaluation["precision"])[0,:,1,0,2]
            self.assertEqual(pr["curves"]["0.5"]["1"]["precision"],expected.tolist())
            valid=np.asarray(evaluation["precision"])[:,:,:,0,2]
            self.assertAlmostEqual(summary["saved_COCO_metrics"]["AP"],float(valid[valid>=0].mean()))
            for name in ("summary.json","summary.txt","gt_best_iou_distribution.json","gt_size_statistics.json","coco_pr_iou0.5.png","diagnostic_tp_fp_scores.png","score_quality_relationship.png"):
                self.assertGreater((root/"analysis"/name).stat().st_size,0)
            with self.assertRaises(FileExistsError): analyze(root,annotation)

    def test_original_device_rounding_is_preserved(self):
        with tempfile.TemporaryDirectory(prefix="det-diag-rounding-") as folder:
            root=Path(folder); _, _, actual, _=fixture(root,dtype=torch.bfloat16)
            raw=torch.load(root/"query_diagnostics/raw_queries/1.pt",weights_only=False)
            self.assertTrue(torch.equal(raw["selected_scores"],actual["scores"].float()))
            self.assertTrue(torch.equal(raw["pred_boxes_xyxy"][raw["selected_flat_indices"]//3],actual["boxes"].float()))
            self.assertFalse(torch.equal(raw["pred_scores"],raw["pred_logits"].sigmoid()))

    def test_empty_gt_image_and_incomplete_export(self):
        with tempfile.TemporaryDirectory(prefix="det-diag-empty-") as folder:
            root=Path(folder); annotation, _, _, _=fixture(root,empty_second=True)
            report=analyze(root,annotation)
            self.assertEqual(report["images"],2); self.assertEqual(report["GT"],5)
            metadata=root/"query_diagnostics/metadata.json"
            saved=json.loads(metadata.read_text());saved["complete"]=False;metadata.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError,"incomplete"):
                analyze(root,annotation,root/"new-analysis")

    def test_best_iou_is_not_actual_selection(self):
        ious=torch.tensor([.6,.95]); scores=torch.tensor([[.05,.9,.05],[.05,.1,.85]])
        status=candidate_status(ious,scores,1,torch.tensor([1]),.5,.5)
        self.assertEqual(status,"GT_class_top1_high_score_and_selected_candidate")
        self.assertEqual(int(ious.argmax()),1)
        self.assertEqual(int(scores[1].argmax()),2)

    def test_no_overlap_and_absent_category(self):
        with tempfile.TemporaryDirectory(prefix="det-diag-no-overlap-") as folder:
            root=Path(folder); annotation, _, _, _=fixture(root,no_overlap=True,absent_writing=True)
            report=analyze(root,annotation)
            self.assertEqual(report["oracle_candidate_status_counts"]["no_query_IoU_ge_threshold"],1)
            self.assertIsNone(report["saved_COCO_per_class"]["2"]["AP"])
            pr=json.loads((root/"analysis/coco_pr_curves.json").read_text())
            self.assertTrue(all(p is None for p in pr["curves"]["0.5"]["2"]["precision"]))
            coverage=json.loads((root/"analysis/gt_best_iou_distribution.json").read_text())
            self.assertEqual(coverage["coverage_at_IoU"]["0.5"]["count"],4)

    def test_one_to_one_and_class_checks(self):
        ious=torch.tensor([[1.],[1.]])
        self.assertEqual(score_first_match(ious,torch.tensor([.6,.9]),.5).tolist(),[-1,0])
        self.assertEqual(score_first_match(ious,torch.tensor([.6,.9]),.5,torch.tensor([1,2]),torch.tensor([1])).tolist(),[0,-1])
        self.assertEqual(score_first_match(torch.zeros(2,0),torch.tensor([.6,.9]),.5).tolist(),[-1,-1])

    def test_annotation_clipping_crowd_and_degenerate(self):
        anns=[{"id":1,"category_id":0,"bbox":[-5,0,10,10],"area":100},
              {"id":2,"category_id":0,"bbox":[0,0,10,10],"area":100,"iscrowd":1},
              {"id":3,"category_id":0,"bbox":[50,0,10,10],"area":100}]
        result=original_gt(anns,{"width":20,"height":20})
        self.assertEqual(len(result),1);self.assertEqual(result[0]["box"],[0,0,5,10])
        self.assertEqual(result[0]["area"],100)

    def test_wrong_annotations_and_incomplete_image_sets_fail(self):
        with tempfile.TemporaryDirectory(prefix="det-diag-wrong-") as folder:
            root=Path(folder); annotation, _, _, _=fixture(root)
            data=json.loads(annotation.read_text()); data["annotations"][0]["category_id"]=2
            annotation.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"disagree"):
                analyze(root,annotation)
            with self.assertRaisesRegex(ValueError,"category IDs"):
                load_coco_pr(root/"eval.pth",[0,1])
            data["images"].append({"id":2,"file_name":"2.jpg","width":120,"height":20})
            annotation.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"image IDs|entire annotation"):
                analyze(root,annotation)


if __name__=="__main__":
    unittest.main()
