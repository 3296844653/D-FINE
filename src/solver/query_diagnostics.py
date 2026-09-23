"""Validation-only export of D-FINE's final queries and postprocessor decisions.

This module does not classify detections as TP/FP or change evaluation. In
particular, max-overlap GT is a reference, not a one-to-one assigned match.
"""

import csv
import hashlib
import json
from pathlib import Path

import torch
from torchvision.ops import box_convert, box_iou

from .validator import scale_boxes


class QueryDiagnosticsWriter:
    """Stream raw queries, selected class-query pairs and nearby predictions."""

    def __init__(self, output_dir, postprocessor, conf_thresh, iou_thresh,
                 neighbor_iou_thresh, config, checkpoint_path):
        if not postprocessor.use_focal_loss or postprocessor.remap_mscoco_category:
            raise ValueError("Query export requires focal postprocessing without COCO label remapping")
        if postprocessor.deploy_mode:
            raise ValueError("Query export does not support deploy-mode postprocessing")
        if not all(0 <= value <= 1 for value in
                   (conf_thresh, iou_thresh, neighbor_iou_thresh)):
            raise ValueError("Diagnostic thresholds must be in [0, 1]")
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.neighbor_iou_thresh = float(neighbor_iou_thresh)
        self.num_classes = postprocessor.num_classes
        self.top_k = postprocessor.num_top_queries
        self.directory = Path(output_dir) / "query_diagnostics"
        # Never silently mix two checkpoints' records in one directory.
        self.directory.mkdir(parents=True, exist_ok=False)
        self.raw_dir = self.directory / "raw_queries"
        self.raw_dir.mkdir()
        self.images = 0
        self.queries = 0

        self.query_fields = ["image_id", "image_name", "query_id", "box_xyxy",
                             "best_class", "best_score", "second_score", "score_margin",
                             "max_gt_iou", "nearest_gt_index", "nearest_gt_class",
                             "selected_pair_count", "kept_pair_count"]
        self.query_fields += [f"logit_{c}" for c in range(self.num_classes)]
        self.query_fields += [f"score_{c}" for c in range(self.num_classes)]
        self.selected_fields = ["image_id", "image_name", "rank", "query_id", "class_id",
                                "score", "above_conf", "box_xyxy", "max_gt_iou",
                                "nearest_gt_index", "nearest_gt_class", "max_same_class_gt_iou",
                                "same_query_other_class_count", "other_query_neighbor_count"]
        self.gt_fields = ["image_id", "image_name", "gt_index", "gt_class", "box_xyxy",
                          "best_query_id", "best_query_iou", "best_query_class_score",
                          "best_overlap_class_score", "best_overlap_query_id",
                          "best_overlap_pair_selected", "best_overlap_pair_above_conf",
                          "any_overlap_pair_selected", "any_overlap_pair_above_conf"]
        self.near_fields = ["image_id", "image_name", "rank_a", "rank_b", "query_a",
                            "query_b", "class_a", "class_b", "score_a", "score_b",
                            "box_iou", "same_query", "same_class"]
        for name, fields in (("queries.csv", self.query_fields),
                             ("selected_predictions.csv", self.selected_fields),
                             ("gt_coverage.csv", self.gt_fields),
                             ("nearby_predictions.csv", self.near_fields)):
            with (self.directory / name).open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(fields)

        checkpoint = Path(checkpoint_path) if checkpoint_path else None
        checkpoint_sha256 = None
        if checkpoint and checkpoint.is_file():
            digest = hashlib.sha256()
            with checkpoint.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
            checkpoint_sha256 = digest.hexdigest()
        self.metadata = {
            "complete": False,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_sha256": checkpoint_sha256,
            "config": config,
            "confidence_threshold": self.conf_thresh,
            "gt_iou_threshold": self.iou_thresh,
            "neighbor_iou_threshold": self.neighbor_iou_thresh,
            "num_classes": self.num_classes,
            "postprocessor_top_k": self.top_k,
            "note": "All logits are final decoder logits (including LQE). Top-k is over query-class pairs. Max GT IoU is not a TP assignment.",
        }
        self._write_metadata()

    def _write_metadata(self):
        with (self.directory / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2, default=str)

    def _append(self, name, rows):
        with (self.directory / name).open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)

    @staticmethod
    def _box_text(box):
        return json.dumps([float(x) for x in box], separators=(",", ":"))

    def record_batch(self, outputs, targets, orig_target_sizes, input_hw):
        logits_batch = outputs["pred_logits"]
        boxes_batch = outputs["pred_boxes"]
        if logits_batch.ndim != 3 or boxes_batch.shape[:2] != logits_batch.shape[:2]:
            raise ValueError("Unexpected pred_logits / pred_boxes dimensions")
        if logits_batch.shape[-1] != self.num_classes or boxes_batch.shape[-1] != 4:
            raise ValueError("Prediction shape does not match the configured postprocessor")
        if len(targets) != logits_batch.shape[0] or len(orig_target_sizes) != len(targets):
            raise ValueError("Batch size mismatch in query export")
        if self.top_k > logits_batch.shape[1] * self.num_classes:
            raise ValueError("Postprocessor top-k exceeds query-class pair count")

        for b, target in enumerate(targets):
            # Select pairs on the original device, identically to DFINEPostProcessor.
            logits = logits_batch[b].detach()
            norm_boxes = boxes_batch[b].detach()
            if not torch.isfinite(logits).all() or not torch.isfinite(norm_boxes).all():
                raise ValueError("Non-finite query output during diagnostics")
            probs = logits.sigmoid()
            selected_scores, flat_index = torch.topk(probs.flatten(), self.top_k)
            qids = (flat_index // self.num_classes).cpu()
            labels = (flat_index % self.num_classes).cpu()
            selected_scores = selected_scores.cpu()
            logits = logits.float().cpu()
            probs = probs.float().cpu()
            norm_boxes = norm_boxes.float().cpu()
            width, height = [int(x) for x in orig_target_sizes[b].tolist()]
            box_scale = norm_boxes.new_tensor([width, height, width, height])
            pred_boxes = box_convert(norm_boxes, "cxcywh", "xyxy") * box_scale
            gt_boxes = scale_boxes(target["boxes"].detach().float().cpu().clone(),
                                   (height, width), tuple(input_hw))
            gt_labels = target["labels"].detach().long().cpu()
            if len(gt_boxes) != len(gt_labels):
                raise ValueError("GT boxes and labels have different lengths")
            if len(gt_labels) and (gt_labels.min() < 0 or gt_labels.max() >= self.num_classes):
                raise ValueError("GT class ID outside configured class range")
            ious = box_iou(pred_boxes, gt_boxes) if len(gt_boxes) else pred_boxes.new_zeros((len(pred_boxes), 0))
            max_ious, nearest_gt = ious.max(dim=1) if len(gt_boxes) else (
                pred_boxes.new_zeros(len(pred_boxes)), torch.full((len(pred_boxes),), -1))
            selected_boxes = pred_boxes[qids]
            kept = selected_scores >= self.conf_thresh
            image_id = int(target["image_id"].item()) if isinstance(target["image_id"], torch.Tensor) else int(target["image_id"])
            image_name = Path(target.get("image_path", str(image_id))).name
            nqueries = len(pred_boxes)

            # Exact CPU tensors permit later re-analysis without rerunning inference.
            torch.save({"image_id": image_id, "image_name": image_name,
                        "original_size_wh": (width, height), "input_size_hw": tuple(input_hw),
                        "pred_logits": logits, "pred_boxes_cxcywh": norm_boxes,
                        "gt_boxes_xyxy": gt_boxes, "gt_labels": gt_labels,
                        "selected_flat_indices": flat_index.cpu()},
                       self.raw_dir / f"{image_id}.pt")

            selected_count = torch.bincount(qids, minlength=nqueries)
            kept_count = torch.bincount(qids[kept], minlength=nqueries)
            best_scores, best_classes = probs.max(dim=1)
            second_scores = probs.topk(min(2, self.num_classes), dim=1).values[:, -1]
            query_rows = []
            for q in range(nqueries):
                g = int(nearest_gt[q])
                query_rows.append([image_id, image_name, q, self._box_text(pred_boxes[q]),
                                   int(best_classes[q]), float(best_scores[q]), float(second_scores[q]),
                                   float(best_scores[q] - second_scores[q]), float(max_ious[q]),
                                   g, int(gt_labels[g]) if g >= 0 else "",
                                   int(selected_count[q]), int(kept_count[q])]
                                  + logits[q].tolist() + probs[q].tolist())
            self._append("queries.csv", query_rows)

            # Only confidence-retained pairs are considered as nearby visible predictions.
            kept_ranks = torch.nonzero(kept).flatten()
            kept_ious = box_iou(selected_boxes[kept], selected_boxes[kept])
            neighbor_counts = [0] * self.top_k
            nearby_rows = []
            for a in range(len(kept_ranks)):
                rank_a = int(kept_ranks[a])
                for z in range(a + 1, len(kept_ranks)):
                    rank_b = int(kept_ranks[z])
                    overlap = float(kept_ious[a, z])
                    if overlap < self.neighbor_iou_thresh:
                        continue
                    same_query = int(qids[rank_a] == qids[rank_b])
                    if not same_query:
                        neighbor_counts[rank_a] += 1
                        neighbor_counts[rank_b] += 1
                    nearby_rows.append([image_id, image_name, rank_a, rank_b,
                                        int(qids[rank_a]), int(qids[rank_b]),
                                        int(labels[rank_a]), int(labels[rank_b]),
                                        float(selected_scores[rank_a]), float(selected_scores[rank_b]),
                                        overlap, same_query, int(labels[rank_a] == labels[rank_b])])
            self._append("nearby_predictions.csv", nearby_rows)

            selected_rows = []
            for rank in range(self.top_k):
                q, cls = int(qids[rank]), int(labels[rank])
                g = int(nearest_gt[q])
                same_class_gt = gt_labels == cls
                max_same_class = float(ious[q, same_class_gt].max()) if same_class_gt.any() else 0.0
                same_query_other_class = int(((qids == q) & (labels != cls) & kept).sum())
                selected_rows.append([image_id, image_name, rank, q, cls,
                                      float(selected_scores[rank]), int(kept[rank]),
                                      self._box_text(selected_boxes[rank]), float(max_ious[q]),
                                      g, int(gt_labels[g]) if g >= 0 else "", max_same_class,
                                      same_query_other_class, neighbor_counts[rank]])
            self._append("selected_predictions.csv", selected_rows)

            gt_rows = []
            for g in range(len(gt_labels)):
                cls = int(gt_labels[g])
                best_q = int(ious[:, g].argmax())
                overlap_qids = torch.nonzero(ious[:, g] >= self.iou_thresh).flatten()
                if len(overlap_qids):
                    best_overlap_q = int(overlap_qids[probs[overlap_qids, cls].argmax()])
                    best_overlap_score = float(probs[best_overlap_q, cls])
                    selected_mask = (qids == best_overlap_q) & (labels == cls)
                    is_selected = bool(selected_mask.any())
                    above_conf = bool((selected_mask & kept).any())
                    any_overlap = (labels == cls) & (ious[qids, g] >= self.iou_thresh)
                    any_selected = bool(any_overlap.any())
                    any_above_conf = bool((any_overlap & kept).any())
                else:
                    best_overlap_q, best_overlap_score, is_selected, above_conf = -1, 0.0, False, False
                    any_selected, any_above_conf = False, False
                gt_rows.append([image_id, image_name, g, cls, self._box_text(gt_boxes[g]),
                                best_q, float(ious[best_q, g]), float(probs[best_q, cls]),
                                best_overlap_score, best_overlap_q, int(is_selected), int(above_conf),
                                int(any_selected), int(any_above_conf)])
            self._append("gt_coverage.csv", gt_rows)
            self.images += 1
            self.queries += nqueries

    def finish(self):
        self.metadata.update(complete=True, images=self.images, queries=self.queries)
        self._write_metadata()
        print(f"Query diagnostics saved to {self.directory}")
