"""Create SCB-U six-class COCO copies with labels 0..5.

Only categories[].id and annotations[].category_id are remapped. Original
annotations, images, boxes, names, YOLO IDs, splits and all other fields are
preserved. Existing outputs are accepted only when their contents match.
This tool needs only the Python standard library, not PyTorch or CUDA.
"""

import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path


def category_mapping(data):
    categories = data["categories"]
    if len(categories) != 6:
        raise ValueError("SCB-U must contain exactly six category definitions")
    ids = [category["id"] for category in categories]
    if any(type(value) is not int for value in ids) or len(set(ids)) != 6:
        raise ValueError("Category IDs must be six distinct integers")

    has_yolo_ids = ["yolo_id" in category for category in categories]
    if all(has_yolo_ids):
        mapping = {category["id"]: category["yolo_id"] for category in categories}
    elif any(has_yolo_ids):
        raise ValueError("Incomplete yolo_id metadata; refusing to guess the mapping")
    elif set(ids) == set(range(6)):
        mapping = {value: value for value in ids}
    elif set(ids) == set(range(1, 7)):
        mapping = {value: value - 1 for value in ids}
    else:
        raise ValueError("Non-contiguous category IDs require explicit yolo_id metadata")

    values = list(mapping.values())
    if any(type(value) is not int for value in values) or set(values) != set(range(6)):
        raise ValueError("The mapping must be one-to-one onto labels 0..5")
    return mapping


def remap_annotations(data, mapping):
    result = copy.deepcopy(data)
    for category in result["categories"]:
        category["id"] = mapping[category["id"]]
    for annotation in result["annotations"]:
        old_id = annotation["category_id"]
        if type(old_id) is not int or old_id not in mapping:
            raise ValueError(
                f"Annotation {annotation.get('id')} has undefined category_id {old_id!r}"
            )
        annotation["category_id"] = mapping[old_id]

    # Reverse the mapping and require exact structural equality: no other
    # annotation, image, metadata or geometry field may have changed.
    inverse = {new: old for old, new in mapping.items()}
    restored = copy.deepcopy(result)
    for category in restored["categories"]:
        category["id"] = inverse[category["id"]]
    for annotation in restored["annotations"]:
        annotation["category_id"] = inverse[annotation["category_id"]]
    if restored != data:
        raise RuntimeError("Unexpected changes outside category ID remapping")
    return result


def prepare(dataset_dir, output_dir=None):
    source_dir = Path(dataset_dir).resolve() / "annotations"
    destination = Path(output_dir).resolve() if output_dir else source_dir
    originals, converted, hashes, mappings = {}, {}, {}, {}
    for split in ("train", "val"):
        source = source_dir / f"instances_{split}.json"
        content = source.read_bytes()
        hashes[split] = hashlib.sha256(content).hexdigest()
        originals[split] = json.loads(content)
        mappings[split] = category_mapping(originals[split])
        converted[split] = remap_annotations(originals[split], mappings[split])

    if mappings["train"] != mappings["val"]:
        raise ValueError("Train/val category ID mappings disagree")
    names = [
        {category["id"]: category.get("name") for category in converted[split]["categories"]}
        for split in ("train", "val")
    ]
    if names[0] != names[1]:
        raise ValueError("Train/val category names disagree after mapping")

    report = {
        "mode": "category_id_only_zero_based_six_classes",
        "category_id_mapping": {str(old): new for old, new in sorted(mappings["train"].items())},
        "source_sha256": hashes,
        "splits": {},
        "note": "Original files untouched; no filtering, geometry edits, renaming or split changes.",
    }
    for split in ("train", "val"):
        report["splits"][split] = {
            "images": len(originals[split]["images"]),
            "annotations": len(originals[split]["annotations"]),
            "counts_before": dict(Counter(str(a["category_id"]) for a in originals[split]["annotations"])),
            "counts_after": dict(Counter(str(a["category_id"]) for a in converted[split]["annotations"])),
        }

    outputs = {
        destination / f"instances_{split}_0based.json": converted[split]
        for split in ("train", "val")
    }
    outputs[destination / "scbu_zero_based_report.json"] = report
    # Validate both splits and every existing output before writing anything.
    for path, expected in outputs.items():
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != expected:
            raise FileExistsError(f"Existing output differs; refusing to overwrite: {path}")
    for split in ("train", "val"):
        source = source_dir / f"instances_{split}.json"
        if hashlib.sha256(source.read_bytes()).hexdigest() != hashes[split]:
            raise RuntimeError(f"Source changed during conversion: {source}")

    destination.mkdir(parents=True, exist_ok=True)
    for path, expected in outputs.items():
        if path.exists():
            print(f"Verified existing: {path}", flush=True)
        else:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(expected, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
            print(f"Created: {path}", flush=True)
    print("Category IDs:", report["category_id_mapping"], flush=True)
    for split, summary in report["splits"].items():
        print(split, summary, flush=True)
    print("Original annotations and all boxes preserved. Labels now in 0..5.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", help="Optional destination for derived annotation files")
    args = parser.parse_args()
    prepare(args.dataset_dir, args.output_dir)


if __name__ == "__main__":
    main()
