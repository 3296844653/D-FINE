"""Convert the SCB-Dataset3-U COCO annotations from six to five classes.

The original hand-raising class (category 0) contains only six annotations and
is removed. The remaining category ids are shifted down by one while images,
boxes, annotation ids, and all other metadata are preserved.
"""

import argparse
import json
from pathlib import Path


CATEGORY_NAMES = [
    "reading",
    "writing",
    "using_phone",
    "bowing_head",
    "leaning_over_table",
]


def convert_annotation(source: Path, destination: Path) -> tuple[int, int]:
    with source.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    original_count = len(dataset["annotations"])
    converted_annotations = []
    for annotation in dataset["annotations"]:
        old_category_id = int(annotation["category_id"])
        if old_category_id == 0:
            continue
        if old_category_id not in range(1, 6):
            raise ValueError(
                f"Unexpected category id {old_category_id} in {source}"
            )

        converted = dict(annotation)
        converted["category_id"] = old_category_id - 1
        converted_annotations.append(converted)

    dataset["annotations"] = converted_annotations
    dataset["categories"] = [
        {"id": category_id, "name": name, "supercategory": "student_behavior"}
        for category_id, name in enumerate(CATEGORY_NAMES)
    ]

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False)
    temporary.replace(destination)

    return original_count, len(converted_annotations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for split in ("train", "val"):
        filename = f"instances_{split}.json"
        source = args.source_dir / filename
        destination = args.output_dir / filename
        before, after = convert_annotation(source, destination)
        print(
            f"{split}: {before} -> {after} annotations "
            f"({before - after} hand-raising annotations removed)"
        )


if __name__ == "__main__":
    main()
