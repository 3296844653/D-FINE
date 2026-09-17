import copy
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from PIL import Image, ImageDraw
from torchvision.ops import box_iou


class Validator:
    def __init__(
        self,
        gt: List[Dict[str, torch.Tensor]],
        preds: List[Dict[str, torch.Tensor]],
        conf_thresh=0.5,
        iou_thresh=0.5,
        class_names=None,
    ) -> None:
        """
        Format example:
        gt = [{'labels': tensor([0]), 'boxes': tensor([[561.0, 297.0, 661.0, 359.0]])}, ...]
        len(gt) is the number of images
        bboxes are in format [x1, y1, x2, y2], absolute values
        """
        self.gt = gt
        self.preds = preds
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.class_names = class_names or {}
        self.thresholds = np.arange(0.2, 1.0, 0.05)
        self.conf_matrix = None
        self.filtered_preds = None

    def compute_metrics(self, extended=False) -> Dict[str, float]:
        filtered_preds = filter_preds(copy.deepcopy(self.preds), self.conf_thresh)
        self.filtered_preds = filtered_preds
        metrics = self._compute_main_metrics(filtered_preds)
        if not extended:
            metrics.pop("extended_metrics", None)
        return metrics

    def _compute_main_metrics(self, preds):
        (
            self.metrics_per_class,
            self.conf_matrix,
            self.class_to_idx,
        ) = self._compute_metrics_and_confusion_matrix(preds)
        tps, fps, fns = 0, 0, 0
        ious = []
        extended_metrics = {}
        for key, value in self.metrics_per_class.items():
            tps += value["TPs"]
            fps += value["FPs"]
            fns += value["FNs"]
            ious.extend(value["IoUs"])

            extended_metrics[f"precision_{key}"] = (
                value["TPs"] / (value["TPs"] + value["FPs"])
                if value["TPs"] + value["FPs"] > 0
                else 0
            )
            extended_metrics[f"recall_{key}"] = (
                value["TPs"] / (value["TPs"] + value["FNs"])
                if value["TPs"] + value["FNs"] > 0
                else 0
            )

            extended_metrics[f"iou_{key}"] = np.mean(value["IoUs"])

        precision = tps / (tps + fps) if (tps + fps) > 0 else 0
        recall = tps / (tps + fns) if (tps + fns) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        iou = np.mean(ious).item() if ious else 0
        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "TPs": tps,
            "FPs": fps,
            "FNs": fns,
            "extended_metrics": extended_metrics,
        }

    def _compute_matrix_multi_class(self, preds):
        metrics_per_class = defaultdict(lambda: {"TPs": 0, "FPs": 0, "FNs": 0, "IoUs": []})
        for pred, gt in zip(preds, self.gt):
            pred_boxes = pred["boxes"]
            pred_labels = pred["labels"]
            gt_boxes = gt["boxes"]
            gt_labels = gt["labels"]

            # isolate each class
            labels = torch.unique(torch.cat([pred_labels, gt_labels]))
            for label in labels:
                pred_cl_boxes = pred_boxes[pred_labels == label]  # filter by bool mask
                gt_cl_boxes = gt_boxes[gt_labels == label]

                n_preds = len(pred_cl_boxes)
                n_gts = len(gt_cl_boxes)
                if not (n_preds or n_gts):
                    continue
                if not n_preds:
                    metrics_per_class[label.item()]["FNs"] += n_gts
                    metrics_per_class[label.item()]["IoUs"].extend([0] * n_gts)
                    continue
                if not n_gts:
                    metrics_per_class[label.item()]["FPs"] += n_preds
                    metrics_per_class[label.item()]["IoUs"].extend([0] * n_preds)
                    continue

                ious = box_iou(pred_cl_boxes, gt_cl_boxes)  # matrix of all IoUs
                ious_mask = ious >= self.iou_thresh

                # indeces of boxes that have IoU >= threshold
                pred_indices, gt_indices = torch.nonzero(ious_mask, as_tuple=True)

                if not pred_indices.numel():  # no predicts matched gts
                    metrics_per_class[label.item()]["FNs"] += n_gts
                    metrics_per_class[label.item()]["IoUs"].extend([0] * n_gts)
                    metrics_per_class[label.item()]["FPs"] += n_preds
                    metrics_per_class[label.item()]["IoUs"].extend([0] * n_preds)
                    continue

                iou_values = ious[pred_indices, gt_indices]

                # sorting by IoU to match hgihest scores first
                sorted_indices = torch.argsort(-iou_values)
                pred_indices = pred_indices[sorted_indices]
                gt_indices = gt_indices[sorted_indices]
                iou_values = iou_values[sorted_indices]

                matched_preds = set()
                matched_gts = set()
                for pred_idx, gt_idx, iou in zip(pred_indices, gt_indices, iou_values):
                    if gt_idx.item() not in matched_gts and pred_idx.item() not in matched_preds:
                        matched_preds.add(pred_idx.item())
                        matched_gts.add(gt_idx.item())
                        metrics_per_class[label.item()]["TPs"] += 1
                        metrics_per_class[label.item()]["IoUs"].append(iou.item())

                unmatched_preds = set(range(n_preds)) - matched_preds
                unmatched_gts = set(range(n_gts)) - matched_gts
                metrics_per_class[label.item()]["FPs"] += len(unmatched_preds)
                metrics_per_class[label.item()]["IoUs"].extend([0] * len(unmatched_preds))
                metrics_per_class[label.item()]["FNs"] += len(unmatched_gts)
                metrics_per_class[label.item()]["IoUs"].extend([0] * len(unmatched_gts))
        return metrics_per_class

    def _compute_metrics_and_confusion_matrix(self, preds):
        # Initialize per-class metrics
        metrics_per_class = defaultdict(lambda: {"TPs": 0, "FPs": 0, "FNs": 0, "IoUs": []})

        # Collect all class IDs
        all_classes = set(self.class_names.keys())
        for pred in preds:
            all_classes.update(pred["labels"].tolist())
        for gt in self.gt:
            all_classes.update(gt["labels"].tolist())
        all_classes = sorted(list(all_classes))
        class_to_idx = {cls_id: idx for idx, cls_id in enumerate(all_classes)}
        n_classes = len(all_classes)
        conf_matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)  # +1 for background class

        for pred, gt in zip(preds, self.gt):
            pred_boxes = pred["boxes"]
            pred_labels = pred["labels"]
            gt_boxes = gt["boxes"]
            gt_labels = gt["labels"]

            n_preds = len(pred_boxes)
            n_gts = len(gt_boxes)

            if n_preds == 0 and n_gts == 0:
                continue

            ious = box_iou(pred_boxes, gt_boxes) if n_preds > 0 and n_gts > 0 else torch.tensor([])
            # Assign matches between preds and gts
            matched_pred_indices = set()
            matched_gt_indices = set()

            if ious.numel() > 0:
                # For each pred box, find the gt box with highest IoU
                ious_mask = ious >= self.iou_thresh
                pred_indices, gt_indices = torch.nonzero(ious_mask, as_tuple=True)
                iou_values = ious[pred_indices, gt_indices]

                # Sorting by IoU to match highest scores first
                sorted_indices = torch.argsort(-iou_values)
                pred_indices = pred_indices[sorted_indices]
                gt_indices = gt_indices[sorted_indices]
                iou_values = iou_values[sorted_indices]

                for pred_idx, gt_idx, iou in zip(pred_indices, gt_indices, iou_values):
                    if (
                        pred_idx.item() in matched_pred_indices
                        or gt_idx.item() in matched_gt_indices
                    ):
                        continue
                    matched_pred_indices.add(pred_idx.item())
                    matched_gt_indices.add(gt_idx.item())

                    pred_label = pred_labels[pred_idx].item()
                    gt_label = gt_labels[gt_idx].item()

                    pred_cls_idx = class_to_idx[pred_label]
                    gt_cls_idx = class_to_idx[gt_label]

                    # Update confusion matrix
                    conf_matrix[gt_cls_idx, pred_cls_idx] += 1

                    # Update per-class metrics
                    if pred_label == gt_label:
                        metrics_per_class[gt_label]["TPs"] += 1
                        metrics_per_class[gt_label]["IoUs"].append(iou.item())
                    else:
                        # Misclassification
                        metrics_per_class[gt_label]["FNs"] += 1
                        metrics_per_class[pred_label]["FPs"] += 1
                        metrics_per_class[gt_label]["IoUs"].append(0)
                        metrics_per_class[pred_label]["IoUs"].append(0)

            # Unmatched predictions (False Positives)
            unmatched_pred_indices = set(range(n_preds)) - matched_pred_indices
            for pred_idx in unmatched_pred_indices:
                pred_label = pred_labels[pred_idx].item()
                pred_cls_idx = class_to_idx[pred_label]
                # Update confusion matrix: background row
                conf_matrix[n_classes, pred_cls_idx] += 1
                # Update per-class metrics
                metrics_per_class[pred_label]["FPs"] += 1
                metrics_per_class[pred_label]["IoUs"].append(0)

            # Unmatched ground truths (False Negatives)
            unmatched_gt_indices = set(range(n_gts)) - matched_gt_indices
            for gt_idx in unmatched_gt_indices:
                gt_label = gt_labels[gt_idx].item()
                gt_cls_idx = class_to_idx[gt_label]
                # Update confusion matrix: background column
                conf_matrix[gt_cls_idx, n_classes] += 1
                # Update per-class metrics
                metrics_per_class[gt_label]["FNs"] += 1
                metrics_per_class[gt_label]["IoUs"].append(0)

        return metrics_per_class, conf_matrix, class_to_idx

    def save_plots(self, path_to_save, filename_prefix="") -> None:
        path_to_save = Path(path_to_save)
        path_to_save.mkdir(parents=True, exist_ok=True)
        prefix = f"{filename_prefix}_" if filename_prefix else ""



        original_conf_matrix = self.conf_matrix.copy() if self.conf_matrix is not None else None
        original_metrics_per_class = self.metrics_per_class
        original_class_to_idx = self.class_to_idx

        if self.conf_matrix is not None:
            class_labels = [
                self.class_names.get(cls_id, str(cls_id)) for cls_id in self.class_to_idx.keys()
            ] + ["background"]

            plt.figure(figsize=(10, 8))
            plt.imshow(self.conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.colorbar()
            tick_marks = np.arange(len(class_labels))
            plt.xticks(tick_marks, class_labels, rotation=45)
            plt.yticks(tick_marks, class_labels)

            # Add labels to each cell
            thresh = self.conf_matrix.max() / 2.0
            for i in range(self.conf_matrix.shape[0]):
                for j in range(self.conf_matrix.shape[1]):
                    plt.text(
                        j,
                        i,
                        format(self.conf_matrix[i, j], "d"),
                        horizontalalignment="center",
                        color="white" if self.conf_matrix[i, j] > thresh else "black",
                    )

            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            plt.savefig(path_to_save / f"{prefix}confusion_matrix.png")
            plt.close()

        thresholds = self.thresholds
        precisions, recalls, f1_scores = [], [], []

        # Store the original predictions to reset after each threshold
        original_preds = copy.deepcopy(self.preds)

        for threshold in thresholds:
            # Filter predictions based on the current threshold
            filtered_preds = filter_preds(copy.deepcopy(original_preds), threshold)
            # Compute metrics with the filtered predictions
            metrics = self._compute_main_metrics(filtered_preds)
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
            f1_scores.append(metrics["f1"])

        # Plot Precision and Recall vs Threshold
        plt.figure()
        plt.plot(thresholds, precisions, label="Precision", marker="o")
        plt.plot(thresholds, recalls, label="Recall", marker="o")
        plt.xlabel("Threshold")
        plt.ylabel("Value")
        plt.title("Precision and Recall vs Threshold")
        plt.legend()
        plt.grid(True)
        plt.savefig(path_to_save / f"{prefix}precision_recall_vs_threshold.png")
        plt.close()

        # Plot F1 Score vs Threshold
        plt.figure()
        plt.plot(thresholds, f1_scores, label="F1 Score", marker="o")
        plt.xlabel("Threshold")
        plt.ylabel("F1 Score")
        plt.title("F1 Score vs Threshold")
        plt.grid(True)
        plt.savefig(path_to_save / f"{prefix}f1_score_vs_threshold.png")
        plt.close()

        # Find the best threshold based on F1 Score (last occurence)
        best_idx = len(f1_scores) - np.argmax(f1_scores[::-1]) - 1
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        logger.info(
            f"Best Threshold: {round(best_threshold, 2)} with F1 Score: {round(best_f1, 3)}"
        )
        self.conf_matrix = original_conf_matrix
        self.metrics_per_class = original_metrics_per_class
        self.class_to_idx = original_class_to_idx

    def _class_name(self, class_id):
        if class_id is None:
            return "background"
        return self.class_names.get(int(class_id), str(int(class_id)))

    def _collect_error_records(self):
        """Collect errors with the same greedy IoU matching used by the matrix."""
        if self.filtered_preds is None:
            self.compute_metrics()

        records = []
        for image_index, (pred, gt) in enumerate(zip(self.filtered_preds, self.gt)):
            pred_boxes, pred_labels = pred["boxes"], pred["labels"]
            pred_scores = pred["scores"]
            gt_boxes, gt_labels = gt["boxes"], gt["labels"]
            matched_preds, matched_gts = set(), set()

            if len(pred_boxes) and len(gt_boxes):
                ious = box_iou(pred_boxes, gt_boxes)
                pred_indices, gt_indices = torch.nonzero(
                    ious >= self.iou_thresh, as_tuple=True
                )
                if pred_indices.numel():
                    order = torch.argsort(-ious[pred_indices, gt_indices])
                    for pair_index in order:
                        pred_index = int(pred_indices[pair_index])
                        gt_index = int(gt_indices[pair_index])
                        if pred_index in matched_preds or gt_index in matched_gts:
                            continue
                        matched_preds.add(pred_index)
                        matched_gts.add(gt_index)
                        pred_label = int(pred_labels[pred_index])
                        gt_label = int(gt_labels[gt_index])
                        if pred_label != gt_label:
                            records.append(
                                self._make_error_record(
                                    "misclassification",
                                    image_index,
                                    gt,
                                    gt_label,
                                    pred_label,
                                    gt_boxes[gt_index],
                                    pred_boxes[pred_index],
                                    pred_scores[pred_index],
                                    ious[pred_index, gt_index],
                                )
                            )

            for gt_index in sorted(set(range(len(gt_boxes))) - matched_gts):
                records.append(
                    self._make_error_record(
                        "false_negative",
                        image_index,
                        gt,
                        int(gt_labels[gt_index]),
                        None,
                        gt_boxes[gt_index],
                        None,
                        None,
                        None,
                    )
                )

            for pred_index in sorted(set(range(len(pred_boxes))) - matched_preds):
                records.append(
                    self._make_error_record(
                        "false_positive",
                        image_index,
                        gt,
                        None,
                        int(pred_labels[pred_index]),
                        None,
                        pred_boxes[pred_index],
                        pred_scores[pred_index],
                        None,
                    )
                )
        return records

    def _make_error_record(
        self,
        error_type,
        image_index,
        gt,
        gt_label,
        pred_label,
        gt_box,
        pred_box,
        score,
        iou,
    ):
        def box_list(box):
            return None if box is None else [round(float(x), 3) for x in box.tolist()]

        return {
            "error_type": error_type,
            "image_index": image_index,
            "image_id": gt.get("image_id", image_index),
            "image_path": gt.get("image_path", ""),
            "gt_label": gt_label,
            "gt_name": self._class_name(gt_label),
            "pred_label": pred_label,
            "pred_name": self._class_name(pred_label),
            "score": None if score is None else float(score),
            "iou": None if iou is None else float(iou),
            "gt_box": box_list(gt_box),
            "pred_box": box_list(pred_box),
        }

    def save_diagnostics(
        self, path_to_save, max_images_per_type=50, run_name=""
    ):
        """Save plots, confusion/summary CSVs, error rows and annotated samples."""
        path_to_save = Path(path_to_save)
        path_to_save.mkdir(parents=True, exist_ok=True)
        safe_run_name = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in str(run_name)
        ).strip("_")
        prefix = f"{safe_run_name}_" if safe_run_name else ""
        self.save_plots(path_to_save, filename_prefix=safe_run_name)

        class_ids = list(self.class_to_idx.keys())
        labels = [self._class_name(class_id) for class_id in class_ids] + ["background"]
        with (path_to_save / f"{prefix}confusion_matrix.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["true\\pred", *labels])
            for label, row in zip(labels, self.conf_matrix):
                writer.writerow([label, *row.tolist()])

        with (path_to_save / f"{prefix}per_class_errors.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            fields = ["class_id", "class_name", "TP", "FP", "FN", "precision", "recall"]
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for class_id in class_ids:
                values = self.metrics_per_class[class_id]
                tp, fp, fn = values["TPs"], values["FPs"], values["FNs"]
                writer.writerow(
                    {
                        "class_id": class_id,
                        "class_name": self._class_name(class_id),
                        "TP": tp,
                        "FP": fp,
                        "FN": fn,
                        "precision": tp / (tp + fp) if tp + fp else 0,
                        "recall": tp / (tp + fn) if tp + fn else 0,
                    }
                )

        records = self._collect_error_records()
        detail_fields = [
            "error_type", "image_id", "image_path", "gt_label", "gt_name",
            "pred_label", "pred_name", "score", "iou", "gt_box", "pred_box",
        ]
        with (path_to_save / f"{prefix}error_details.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=detail_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

        grouped = defaultdict(list)
        for record in records:
            grouped[(record["error_type"], record["image_id"], record["image_path"])].append(record)

        saved_counts = defaultdict(int)
        for (error_type, image_id, image_path), image_records in grouped.items():
            if saved_counts[error_type] >= int(max_images_per_type) or not image_path:
                continue
            try:
                image = Image.open(image_path).convert("RGB")
            except (FileNotFoundError, OSError) as error:
                logger.warning(f"Cannot open diagnostic image {image_path}: {error}")
                continue
            draw = ImageDraw.Draw(image)
            for record in image_records:
                if record["gt_box"] is not None:
                    draw.rectangle(record["gt_box"], outline="red", width=3)
                    draw.text(
                        (record["gt_box"][0], max(0, record["gt_box"][1] - 12)),
                        f"GT:{record['gt_name']}", fill="red",
                    )
                if record["pred_box"] is not None:
                    draw.rectangle(record["pred_box"], outline="cyan", width=3)
                    score_text = "" if record["score"] is None else f" {record['score']:.2f}"
                    draw.text(
                        (record["pred_box"][0], record["pred_box"][1]),
                        f"P:{record['pred_name']}{score_text}", fill="cyan",
                    )
            destination = path_to_save / f"{prefix}{error_type}_images"
            destination.mkdir(exist_ok=True)
            safe_stem = Path(image_path).stem.replace(" ", "_")
            image.save(
                destination / f"{prefix}{image_id}_{safe_stem}.jpg", quality=92
            )
            saved_counts[error_type] += 1

        logger.info(
            f"Saved diagnostics to {path_to_save} with {len(records)} error records"
        )


def filter_preds(preds, conf_thresh):
    for pred in preds:
        keep_idxs = pred["scores"] >= conf_thresh
        pred["scores"] = pred["scores"][keep_idxs]
        pred["boxes"] = pred["boxes"][keep_idxs]
        pred["labels"] = pred["labels"][keep_idxs]
    return preds


def scale_boxes(boxes, orig_shape, resized_shape):
    """
    boxes in format: [x1, y1, x2, y2], absolute values
    orig_shape: [height, width]
    resized_shape: [height, width]
    """
    scale_x = orig_shape[1] / resized_shape[1]
    scale_y = orig_shape[0] / resized_shape[0]
    boxes[:, 0] *= scale_x
    boxes[:, 2] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 3] *= scale_y
    return boxes
