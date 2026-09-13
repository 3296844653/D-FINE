"""Prepare leakage-aware thesis copies of SCB3-S and STBD-08.

The source datasets are never modified.  The generated datasets use hard links
for images (so they consume almost no additional disk space on the same
filesystem), cleaned COCO annotations, matching YOLO labels, and an audit
report describing every transformation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


SCB3_S_NAMES = ["hand_raising", "reading", "writing"]
STBD_NAMES = [
    "writing",
    "reading",
    "listening",
    "turning_around",
    "raising_hand",
    "standing",
    "discussing",
    "guiding",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def link_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def clip_bbox(
    bbox: list[float], width: int, height: int
) -> tuple[list[float] | None, set[str]]:
    if len(bbox) != 4 or not all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in bbox
    ):
        return None, {"non_finite_or_malformed"}

    x, y, box_width, box_height = map(float, bbox)
    flags: set[str] = set()
    if box_width <= 0 or box_height <= 0:
        flags.add("non_positive_size")
    if x < 0 or y < 0:
        flags.add("negative_origin")
    if x + box_width > width or y + box_height > height:
        flags.add("outside_image")

    x1 = min(max(x, 0.0), float(width))
    y1 = min(max(y, 0.0), float(height))
    x2 = min(max(x + box_width, 0.0), float(width))
    y2 = min(max(y + box_height, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        flags.add("degenerate_after_clipping")
        return None, flags
    return [x1, y1, x2 - x1, y2 - y1], flags


def clean_annotations(
    annotations: list[dict[str, Any]],
    width: int,
    height: int,
    category_map: dict[int, int],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    cleaned: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    seen: set[tuple[int, tuple[float, ...]]] = set()
    for annotation in annotations:
        old_category = int(annotation["category_id"])
        if old_category not in category_map:
            raise ValueError(f"Unexpected category id: {old_category}")
        bbox, flags = clip_bbox(annotation["bbox"], width, height)
        stats.update(flags)
        if bbox is None:
            stats["dropped_annotations"] += 1
            continue
        category = category_map[old_category]
        key = (category, tuple(round(value, 6) for value in bbox))
        if key in seen:
            stats["duplicate_annotations_removed"] += 1
            continue
        seen.add(key)
        item = deepcopy(annotation)
        item["category_id"] = category
        item["bbox"] = bbox
        item["area"] = bbox[2] * bbox[3]
        item["iscrowd"] = int(item.get("iscrowd", 0))
        cleaned.append(item)
    return cleaned, stats


def write_yolo_label(
    path: Path,
    annotations: list[dict[str, Any]],
    width: int,
    height: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for annotation in annotations:
        x, y, box_width, box_height = annotation["bbox"]
        center_x = (x + box_width / 2) / width
        center_y = (y + box_height / 2) / height
        normalized_width = box_width / width
        normalized_height = box_height / height
        values = (center_x, center_y, normalized_width, normalized_height)
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"Invalid normalized YOLO box in {path}: {values}")
        lines.append(
            f"{annotation['category_id']} "
            + " ".join(f"{value:.8f}" for value in values)
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def categories(names: list[str]) -> list[dict[str, Any]]:
    return [
        {"id": category_id, "name": name, "supercategory": "student_behavior"}
        for category_id, name in enumerate(names)
    ]


def create_data_yaml(output: Path, names: list[str]) -> None:
    rows = [
        f"path: {output}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    rows.extend(f"  {index}: {name}" for index, name in enumerate(names))
    (output / "data.yaml").write_text("\n".join(rows) + "\n", encoding="utf-8")


def prepare_scb3_s(source: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    datasets = {
        split: load_json(source / "annotations" / f"instances_{split}.json")
        for split in ("train", "val")
    }
    records_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    original_counts: dict[str, Any] = {}

    for split, dataset in datasets.items():
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in dataset["annotations"]:
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        original_counts[split] = {
            "images": len(dataset["images"]),
            "annotations": len(dataset["annotations"]),
        }
        for image in dataset["images"]:
            path = source / "images" / split / image["file_name"]
            if not path.is_file():
                raise FileNotFoundError(path)
            records_by_hash[sha1(path)].append(
                {
                    "split": split,
                    "image": image,
                    "path": path,
                    "annotations": annotations_by_image[int(image["id"])],
                }
            )

    cleaned_datasets = {
        split: {
            "info": {
                "description": "SCB-Dataset3-S thesis-clean v1",
                "source": str(source),
                "cleaning": (
                    "Exact-image deduplication; cross-split duplicate groups assigned "
                    "to train; best available annotation set selected; boxes clipped; "
                    "degenerate and duplicate annotations removed."
                ),
            },
            "licenses": datasets[split].get("licenses", []),
            "images": [],
            "annotations": [],
            "categories": categories(SCB3_S_NAMES),
        }
        for split in ("train", "val")
    }
    counters: Counter[str] = Counter()
    next_image_id = {"train": 1, "val": 1}
    next_annotation_id = {"train": 1, "val": 1}

    for digest, members in sorted(records_by_hash.items()):
        member_splits = {member["split"] for member in members}
        target_split = "train" if "train" in member_splits else "val"
        if len(members) > 1:
            counters["duplicate_image_files_removed"] += len(members) - 1
            counters["duplicate_image_groups"] += 1
        if len(member_splits) > 1:
            counters["cross_split_duplicate_groups_moved_to_train"] += 1
            counters["validation_files_removed_for_cross_split_leakage"] += sum(
                member["split"] == "val" for member in members
            )

        dimensions = {
            (int(member["image"]["width"]), int(member["image"]["height"]))
            for member in members
        }
        if len(dimensions) != 1:
            raise ValueError(f"Identical files disagree on dimensions: {members}")
        width, height = next(iter(dimensions))

        candidates = []
        for member in members:
            cleaned, annotation_stats = clean_annotations(
                member["annotations"], width, height, {0: 0, 1: 1, 2: 2}
            )
            candidates.append((len(cleaned), member, cleaned, annotation_stats))
        candidates.sort(
            key=lambda value: (
                value[0],
                value[1]["split"] == target_split,
                value[1]["image"]["file_name"],
            ),
            reverse=True,
        )
        _, annotation_member, selected_annotations, selected_stats = candidates[0]
        counters.update(selected_stats)
        if annotation_member["split"] != target_split:
            counters["annotation_sets_transferred_across_split"] += 1

        image_members = [m for m in members if m["split"] == target_split]
        image_member = min(image_members, key=lambda m: m["image"]["file_name"])
        filename = image_member["image"]["file_name"]
        image_id = next_image_id[target_split]
        next_image_id[target_split] += 1
        image_record = deepcopy(image_member["image"])
        image_record["id"] = image_id
        image_record["file_name"] = filename
        cleaned_datasets[target_split]["images"].append(image_record)

        finalized = []
        for annotation in selected_annotations:
            item = deepcopy(annotation)
            item["id"] = next_annotation_id[target_split]
            item["image_id"] = image_id
            next_annotation_id[target_split] += 1
            cleaned_datasets[target_split]["annotations"].append(item)
            finalized.append(item)

        link_image(
            image_member["path"], output / "images" / target_split / filename
        )
        write_yolo_label(
            output / "labels" / target_split / f"{Path(filename).stem}.txt",
            finalized,
            width,
            height,
        )

    for split in ("train", "val"):
        write_json(
            output / "annotations" / f"instances_{split}.json",
            cleaned_datasets[split],
        )
    create_data_yaml(output, SCB3_S_NAMES)
    return {
        "dataset": "SCB-Dataset3-S",
        "source": str(source),
        "output": str(output),
        "original": original_counts,
        "cleaned": {
            split: {
                "images": len(cleaned_datasets[split]["images"]),
                "annotations": len(cleaned_datasets[split]["annotations"]),
                "per_category": dict(
                    sorted(
                        Counter(
                            annotation["category_id"]
                            for annotation in cleaned_datasets[split]["annotations"]
                        ).items()
                    )
                ),
            }
            for split in ("train", "val")
        },
        "changes": dict(sorted(counters.items())),
        "remaining_limitation": (
            "The official filenames strongly suggest adjacent video frames across "
            "train and val. Exact duplicates are removed, but sequence-level leakage "
            "cannot be eliminated without authoritative source-video group metadata."
        ),
    }


def prepare_stbd(source: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    source_layout = {
        "train": ("train2017", "instances_train2017.json"),
        "val": ("val2017", "instances_val2017.json"),
    }
    old_to_new = {old_id: old_id - 1 for old_id in range(1, 9)}
    report: dict[str, Any] = {
        "dataset": "STBD-08",
        "source": str(source),
        "output": str(output),
        "category_id_mapping": old_to_new,
        "splits": {},
        "remaining_limitation": (
            "The class distribution is highly imbalanced and filenames suggest "
            "adjacent frames across train and val. Samples were not removed because "
            "these are intrinsic/protocol properties, not annotation-format errors."
        ),
    }

    for split, (image_folder, annotation_name) in source_layout.items():
        dataset = load_json(source / "annotations" / annotation_name)
        source_categories = {
            int(item["id"]): item["name"] for item in dataset["categories"]
        }
        if set(source_categories) != set(old_to_new):
            raise ValueError(f"Unexpected STBD categories: {source_categories}")
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in dataset["annotations"]:
            annotations_by_image[int(annotation["image_id"])].append(annotation)

        cleaned_dataset = {
            "info": {
                "description": "STBD-08 thesis-clean v1",
                "source": str(source),
                "cleaning": "Category ids remapped from 1-8 to 0-7; boxes validated.",
            },
            "licenses": dataset.get("licenses", []),
            "images": [],
            "annotations": [],
            "categories": categories(STBD_NAMES),
        }
        stats: Counter[str] = Counter()
        annotation_id = 1
        for image in dataset["images"]:
            filename = image["file_name"]
            source_image = source / image_folder / filename
            if not source_image.is_file():
                raise FileNotFoundError(source_image)
            width, height = int(image["width"]), int(image["height"])
            cleaned, annotation_stats = clean_annotations(
                annotations_by_image[int(image["id"])],
                width,
                height,
                old_to_new,
            )
            stats.update(annotation_stats)
            finalized = []
            for annotation in cleaned:
                item = deepcopy(annotation)
                item["id"] = annotation_id
                annotation_id += 1
                cleaned_dataset["annotations"].append(item)
                finalized.append(item)
            cleaned_dataset["images"].append(deepcopy(image))
            link_image(source_image, output / "images" / split / filename)
            write_yolo_label(
                output / "labels" / split / f"{Path(filename).stem}.txt",
                finalized,
                width,
                height,
            )

        write_json(
            output / "annotations" / f"instances_{split}.json", cleaned_dataset
        )
        report["splits"][split] = {
            "images": len(cleaned_dataset["images"]),
            "annotations": len(cleaned_dataset["annotations"]),
            "per_category": dict(
                sorted(
                    Counter(
                        annotation["category_id"]
                        for annotation in cleaned_dataset["annotations"]
                    ).items()
                )
            ),
            "changes": dict(sorted(stats.items())),
        }
    create_data_yaml(output, STBD_NAMES)
    return report


def write_report(output: Path, report: dict[str, Any]) -> None:
    write_json(output / "audit_report.json", report)
    lines = [
        f"# {report['dataset']} thesis-clean v1",
        "",
        f"- Source: `{report['source']}`",
        f"- Output: `{report['output']}`",
        "- Original source files were not modified.",
        "- Images in this directory are hard links to the source files.",
        "- Full machine-readable details are in `audit_report.json`.",
        "",
        "## Remaining limitation",
        "",
        report["remaining_limitation"],
        "",
    ]
    (output / "README_THESIS_DATASET.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scb3-s-source", type=Path, required=True)
    parser.add_argument("--scb3-s-output", type=Path, required=True)
    parser.add_argument("--stbd-source", type=Path, required=True)
    parser.add_argument("--stbd-output", type=Path, required=True)
    args = parser.parse_args()

    scb_report = prepare_scb3_s(args.scb3_s_source, args.scb3_s_output)
    write_report(args.scb3_s_output, scb_report)
    print(json.dumps(scb_report, ensure_ascii=False, indent=2))

    stbd_report = prepare_stbd(args.stbd_source, args.stbd_output)
    write_report(args.stbd_output, stbd_report)
    print(json.dumps(stbd_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
