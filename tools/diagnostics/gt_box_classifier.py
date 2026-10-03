"""Diagnose SCB-U behavior separability with GT-box classification.

This script deliberately bypasses object localization.  It dynamically crops
each COCO GT box, trains an ImageNet-pretrained ConvNeXt-Tiny with plain cross
entropy, and evaluates both ordinary classification metrics and COCO AP using
the original GT boxes plus the predicted class/score.

It is a diagnostic tool and does not modify or import the D-FINE detector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:  # D-FINE officially depends on faster-coco-eval.
    from faster_coco_eval import COCO, COCOeval_faster as COCOeval


Image.MAX_IMAGE_PIXELS = None


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a small standalone YAML with optional single-file inheritance."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    include = config.pop("__include__", None)
    if include is None:
        return config
    include_path = (path.parent / include).resolve()
    return _deep_update(load_config(include_path), config)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GTBoxClassificationDataset(Dataset):
    """One classification sample per non-crowd, valid COCO annotation."""

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        image_transform,
        padding: float,
        expected_categories: List[Tuple[int, str]] | None = None,
        full_image_resize: List[int] | Tuple[int, int] | None = None,
    ) -> None:
        if padding < 0:
            raise ValueError("GT-box padding must be non-negative")
        self.image_dir = Path(image_dir)
        self.annotation_file = str(annotation_file)
        self.coco = COCO(self.annotation_file)
        self.transform = image_transform
        self.padding = float(padding)
        if full_image_resize is not None:
            if len(full_image_resize) != 2 or any(int(value) <= 0 for value in full_image_resize):
                raise ValueError("full_image_resize must be [height, width] with positive values")
            self.full_image_resize = tuple(int(value) for value in full_image_resize)
        else:
            self.full_image_resize = None

        categories = sorted(self.coco.loadCats(self.coco.getCatIds()), key=lambda x: x["id"])
        self.categories = [(int(cat["id"]), str(cat["name"])) for cat in categories]
        if expected_categories is not None and self.categories != expected_categories:
            raise ValueError(
                "Train/validation category definitions differ: "
                f"{expected_categories} vs {self.categories}"
            )
        self.category_to_label = {
            category_id: label for label, (category_id, _) in enumerate(self.categories)
        }
        self.label_to_category = {
            label: category_id for category_id, label in self.category_to_label.items()
        }
        self.class_names = [name for _, name in self.categories]

        self.records: List[Dict[str, Any]] = []
        for annotation_id in sorted(self.coco.anns):
            annotation = self.coco.anns[annotation_id]
            if annotation.get("iscrowd", 0):
                continue
            x, y, width, height = map(float, annotation["bbox"])
            if width <= 1 or height <= 1:
                continue
            category_id = int(annotation["category_id"])
            if category_id not in self.category_to_label:
                continue
            image_info = self.coco.imgs[int(annotation["image_id"])]
            self.records.append(
                {
                    "annotation_id": int(annotation_id),
                    "image_id": int(annotation["image_id"]),
                    "file_name": str(image_info["file_name"]),
                    "bbox": [x, y, width, height],
                    "category_id": category_id,
                    "label": self.category_to_label[category_id],
                }
            )

        if not self.records:
            raise ValueError(f"No valid annotations found in {annotation_file}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = self.image_dir / record["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            x, y, width, height = record["bbox"]
            if self.full_image_resize is not None:
                # Match D-FINE validation preprocessing: the complete image is
                # resized to a fixed HxW before the object crop is extracted.
                # Scaling x/y independently is intentional because D-FINE's
                # fixed [640,640] Resize also changes the original aspect ratio.
                original_width, original_height = image.size
                resized_height, resized_width = self.full_image_resize
                scale_x = resized_width / original_width
                scale_y = resized_height / original_height
                image = transform_functional.resize(
                    image,
                    [resized_height, resized_width],
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
                x, width = x * scale_x, width * scale_x
                y, height = y * scale_y, height * scale_y
            pad_x, pad_y = width * self.padding, height * self.padding
            left = max(0.0, x - pad_x)
            top = max(0.0, y - pad_y)
            right = min(float(image.width), x + width + pad_x)
            bottom = min(float(image.height), y + height + pad_y)
            crop = image.crop((int(left), int(top), int(math.ceil(right)), int(math.ceil(bottom))))
        return self.transform(crop), int(record["label"]), int(index)

    def class_counts(self) -> List[int]:
        counts = [0] * len(self.categories)
        for record in self.records:
            counts[record["label"]] += 1
        return counts


def build_transforms(image_size: int, training: bool):
    operations: List[Any] = [transforms.Resize((image_size, image_size))]
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(operations)


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_tiny(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model


def confusion_metrics(confusion: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    true_count = confusion.sum(axis=1)
    pred_count = confusion.sum(axis=0)
    correct = np.diag(confusion).astype(np.float64)
    precision = np.divide(correct, pred_count, out=np.zeros_like(correct), where=pred_count > 0)
    recall = np.divide(correct, true_count, out=np.zeros_like(correct), where=true_count > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(correct),
        where=(precision + recall) > 0,
    )
    total = int(confusion.sum())
    return {
        "accuracy": float(correct.sum() / total) if total else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "num_samples": total,
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(true_count[index]),
            }
            for index, name in enumerate(class_names)
        },
    }


def save_confusion_matrix(
    confusion: np.ndarray, class_names: List[str], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["GT\\Pred", *class_names])
        for name, row in zip(class_names, confusion.tolist()):
            writer.writerow([name, *row])

    denominator = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion,
        denominator,
        out=np.zeros_like(confusion, dtype=np.float64),
        where=denominator > 0,
    )
    with (output_dir / "confusion_matrix_normalized.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["GT\\Pred", *class_names])
        for name, row in zip(class_names, normalized.tolist()):
            writer.writerow([name, *[f"{value:.6f}" for value in row]])

    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 7))
        image = axis.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        figure.colorbar(image, ax=axis)
        axis.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted class",
            ylabel="GT class",
            title="Normalized GT-box classification confusion matrix",
        )
        plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
        for row in range(len(class_names)):
            for column in range(len(class_names)):
                value = normalized[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}\n({confusion[row, column]})",
                    ha="center",
                    va="center",
                    color="white" if value > 0.5 else "black",
                    fontsize=8,
                )
        figure.tight_layout()
        figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
        plt.close(figure)
    except ImportError:
        print("matplotlib is unavailable; confusion CSV files were still saved.")


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    data_loader: DataLoader,
    dataset: GTBoxClassificationDataset,
    device: torch.device,
    amp: bool,
) -> Tuple[Dict[str, Any], np.ndarray, List[Dict[str, Any]]]:
    model.eval()
    num_classes = len(dataset.class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    predictions: List[Dict[str, Any]] = []

    for images, labels, indices in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp,
        ):
            logits = model(images)
        probabilities = logits.float().softmax(dim=-1)
        scores, predicted_labels = probabilities.max(dim=-1)

        labels_cpu = labels.cpu().tolist()
        predicted_cpu = predicted_labels.cpu().tolist()
        scores_cpu = scores.cpu().tolist()
        probabilities_cpu = probabilities.cpu().tolist()
        indices_cpu = indices.cpu().tolist()
        for true_label, predicted_label, score, class_scores, index in zip(
            labels_cpu, predicted_cpu, scores_cpu, probabilities_cpu, indices_cpu
        ):
            confusion[true_label, predicted_label] += 1
            record = dataset.records[index]
            predictions.append(
                {
                    "image_id": record["image_id"],
                    "category_id": dataset.label_to_category[predicted_label],
                    "bbox": record["bbox"],
                    "score": float(score),
                    "annotation_id": record["annotation_id"],
                    "gt_category_id": record["category_id"],
                    "gt_label": true_label,
                    "pred_label": predicted_label,
                    "class_scores": {
                        name: float(value)
                        for name, value in zip(dataset.class_names, class_scores)
                    },
                }
            )

    metrics = confusion_metrics(confusion, dataset.class_names)
    return metrics, confusion, predictions


def evaluate_coco(
    annotation_file: str,
    predictions: List[Dict[str, Any]],
    category_names: List[str],
    output_dir: Path,
) -> Dict[str, Any]:
    # COCO result JSON must contain only official detection-result fields.
    coco_predictions = [
        {
            "image_id": item["image_id"],
            "category_id": item["category_id"],
            "bbox": item["bbox"],
            "score": item["score"],
        }
        for item in predictions
    ]
    prediction_path = output_dir / "gtbox_classifier_predictions.json"
    prediction_path.write_text(
        json.dumps(coco_predictions, ensure_ascii=False), encoding="utf-8"
    )

    coco_gt = COCO(annotation_file)
    coco_dt = coco_gt.loadRes(str(prediction_path))
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = [float(value) for value in evaluator.stats]
    result: Dict[str, Any] = {
        "AP": stats[0],
        "AP50": stats[1],
        "AP75": stats[2],
        "AP_small": stats[3],
        "AP_medium": stats[4],
        "AP_large": stats[5],
        "AR1": stats[6],
        "AR10": stats[7],
        "AR100": stats[8],
    }

    precision = evaluator.eval["precision"]  # [IoU, recall, class, area, maxDets]
    per_class: Dict[str, float] = {}
    for class_index, class_name in enumerate(category_names):
        values = precision[:, :, class_index, 0, -1]
        valid = values[values > -1]
        per_class[class_name] = float(valid.mean()) if valid.size else float("nan")
    result["per_class_AP"] = per_class
    (output_dir / "coco_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def write_classification_outputs(
    metrics: Dict[str, Any],
    confusion: np.ndarray,
    predictions: List[Dict[str, Any]],
    dataset: GTBoxClassificationDataset,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classification_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "per_class_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for class_name, values in metrics["per_class"].items():
            writer.writerow(
                [
                    class_name,
                    values["precision"],
                    values["recall"],
                    values["f1"],
                    values["support"],
                ]
            )
    detailed = []
    for item in predictions:
        record = dataset.records_by_annotation[item["annotation_id"]]
        detailed.append(
            {
                **item,
                "file_name": record["file_name"],
                "gt_class": dataset.class_names[item["gt_label"]],
                "pred_class": dataset.class_names[item["pred_label"]],
            }
        )
    (output_dir / "classification_predictions_detailed.json").write_text(
        json.dumps(detailed, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "classification_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "annotation_id",
                "image_id",
                "file_name",
                "bbox_xywh",
                "gt_class",
                "pred_class",
                "pred_score",
                "correct",
                *[f"score_{name}" for name in dataset.class_names],
            ]
        )
        for item in detailed:
            writer.writerow(
                [
                    item["annotation_id"],
                    item["image_id"],
                    item["file_name"],
                    json.dumps(item["bbox"]),
                    item["gt_class"],
                    item["pred_class"],
                    item["score"],
                    int(item["gt_label"] == item["pred_label"]),
                    *[item["class_scores"][name] for name in dataset.class_names],
                ]
            )
    save_confusion_matrix(confusion, dataset.class_names, output_dir)


def run_training(config: Dict[str, Any], output_override: str | None) -> None:
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = config["training"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    output_dir = Path(output_override or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    image_size = int(model_cfg.get("image_size", 384))
    padding = float(data_cfg.get("padding", 0.0))
    full_image_resize = data_cfg.get("full_image_resize")
    train_dataset = GTBoxClassificationDataset(
        data_cfg["train_image_dir"],
        data_cfg["train_annotation"],
        build_transforms(image_size, training=True),
        padding,
        full_image_resize=full_image_resize,
    )
    val_dataset = GTBoxClassificationDataset(
        data_cfg["val_image_dir"],
        data_cfg["val_annotation"],
        build_transforms(image_size, training=False),
        padding,
        expected_categories=train_dataset.categories,
        full_image_resize=full_image_resize,
    )
    # Fast annotation lookup is used only when exporting detailed predictions.
    val_dataset.records_by_annotation = {
        record["annotation_id"]: record for record in val_dataset.records
    }

    batch_size = int(train_cfg.get("batch_size", 32))
    workers = int(train_cfg.get("num_workers", 4))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    print("Classes:", train_dataset.categories)
    print("Train samples:", len(train_dataset), train_dataset.class_counts())
    print("Validation samples:", len(val_dataset), val_dataset.class_counts())
    print(
        "Padding:", padding,
        "Full-image resize:", full_image_resize,
        "Crop classifier size:", image_size,
        "Device:", device,
    )

    model = build_model(
        len(train_dataset.class_names), bool(model_cfg.get("imagenet_pretrained", True))
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    epochs = int(train_cfg.get("epochs", 30))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_macro_f1 = -1.0
    best_path = output_dir / "best_macro_f1.pth"
    log_path = output_dir / "training_log.jsonl"
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * labels.numel()
            correct += int((logits.argmax(dim=-1) == labels).sum())
            seen += labels.numel()
        scheduler.step()

        val_metrics, _, _ = evaluate_classifier(
            model, val_loader, val_dataset, device, amp
        )
        epoch_log = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": loss_sum / max(seen, 1),
            "train_accuracy": correct / max(seen, 1),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        print(json.dumps(epoch_log, ensure_ascii=False))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_log, ensure_ascii=False) + "\n")

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_macro_f1": best_macro_f1,
                    "class_names": train_dataset.class_names,
                    "categories": train_dataset.categories,
                    "config": config,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    metrics, confusion, predictions = evaluate_classifier(
        model, val_loader, val_dataset, device, amp
    )
    write_classification_outputs(
        metrics, confusion, predictions, val_dataset, output_dir
    )
    coco_metrics = evaluate_coco(
        data_cfg["val_annotation"], predictions, val_dataset.class_names, output_dir
    )
    summary = {
        "best_epoch": int(checkpoint["epoch"]),
        "padding": padding,
        "full_image_resize": full_image_resize,
        "classification": metrics,
        "gtbox_coco": coco_metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Final summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def run_eval_only(
    config: Dict[str, Any], checkpoint_path: str, output_override: str | None
) -> None:
    seed_everything(int(config.get("seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg, model_cfg = config["data"], config["model"]
    output_dir = Path(output_override or config["output_dir"]) / "eval_only"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = GTBoxClassificationDataset(
        data_cfg["val_image_dir"],
        data_cfg["val_annotation"],
        build_transforms(int(model_cfg.get("image_size", 384)), training=False),
        float(data_cfg.get("padding", 0.0)),
        full_image_resize=data_cfg.get("full_image_resize"),
    )
    dataset.records_by_annotation = {
        record["annotation_id"]: record for record in dataset.records
    }
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 32)),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )
    model = build_model(len(dataset.class_names), pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    metrics, confusion, predictions = evaluate_classifier(
        model, loader, dataset, device, amp
    )
    write_classification_outputs(metrics, confusion, predictions, dataset, output_dir)
    coco_metrics = evaluate_coco(
        data_cfg["val_annotation"], predictions, dataset.class_names, output_dir
    )
    summary = {"classification": metrics, "gtbox_coco": coco_metrics}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="GT-box classifier YAML")
    parser.add_argument("--output-dir", default=None, help="Optional output override")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--eval-only requires --checkpoint")
        run_eval_only(config, args.checkpoint, args.output_dir)
    else:
        run_training(config, args.output_dir)


if __name__ == "__main__":
    main()
