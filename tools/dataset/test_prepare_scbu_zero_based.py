"""CPU-only preservation and safety tests for SCB-U category ID conversion."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "prepare_scbu_zero_based", Path(__file__).with_name("prepare_scbu_zero_based.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    return {
        "info": {"description": "unchanged metadata"},
        "images": [{"id": 7, "file_name": "example.jpg", "width": 640, "height": 480}],
        "categories": [
            {"id": i + 1, "name": f"class_{i}", "supercategory": "object", "yolo_id": i}
            for i in range(6)
        ],
        "annotations": [
            {"id": i + 1, "image_id": 7, "category_id": i + 1,
             "bbox": [-2.5, 3, 40, 80], "area": 3200, "iscrowd": int(i == 5),
             "segmentation": [], "custom": "preserve this"}
            for i in range(6)
        ],
    }


class SCBUZeroBasedTests(unittest.TestCase):
    def test_exact_mapping_and_preservation(self):
        data = fixture()
        before = copy.deepcopy(data)
        result = module.remap_annotations(data, module.category_mapping(data))
        self.assertEqual(data, before)
        self.assertEqual([c["id"] for c in result["categories"]], list(range(6)))
        self.assertEqual([a["category_id"] for a in result["annotations"]], list(range(6)))
        expected = copy.deepcopy(before)
        for category in expected["categories"]:
            category["id"] -= 1
        for annotation in expected["annotations"]:
            annotation["category_id"] -= 1
        self.assertEqual(result, expected)

    def test_explicit_yolo_ids_override_category_order(self):
        data = fixture()
        data["categories"].reverse()
        self.assertEqual(module.category_mapping(data), {i + 1: i for i in range(6)})

    def test_zero_and_one_based_fallbacks(self):
        for zero_based in (False, True):
            with self.subTest(zero_based=zero_based):
                data = fixture()
                for category in data["categories"]:
                    category.pop("yolo_id")
                    if zero_based:
                        category["id"] -= 1
                expected = {i: i for i in range(6)} if zero_based else {i + 1: i for i in range(6)}
                self.assertEqual(module.category_mapping(data), expected)

    def test_invalid_category_metadata_is_rejected(self):
        mutations = (
            lambda d: d["categories"].pop(),
            lambda d: d["categories"][0].update(id=2),
            lambda d: d["categories"][0].pop("yolo_id"),
            lambda d: d["categories"][0].update(yolo_id=5),
            lambda d: d["categories"][0].update(yolo_id=True),
        )
        for mutate in mutations:
            data = fixture()
            mutate(data)
            with self.assertRaises(ValueError):
                module.category_mapping(data)

    def test_undefined_annotation_category_is_rejected(self):
        data = fixture()
        data["annotations"][0]["category_id"] = 99
        with self.assertRaises(ValueError):
            module.remap_annotations(data, module.category_mapping(data))

    def create_sources(self, root, train=None, val=None):
        folder = root / "annotations"
        folder.mkdir()
        for split, data in (("train", train), ("val", val)):
            (folder / f"instances_{split}.json").write_text(
                json.dumps(data if data is not None else fixture()), encoding="utf-8"
            )
        return folder

    def test_originals_unchanged_and_repeated_runs_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = self.create_sources(root)
            before = {p: p.read_bytes() for p in folder.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()):
                report = module.prepare(root)
            created = {p: p.read_bytes() for p in folder.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.prepare(root), report)
            self.assertEqual({p: p.read_bytes() for p in folder.iterdir()}, created)
            self.assertTrue(all(path.read_bytes() == content for path, content in before.items()))
            self.assertEqual(report["splits"]["val"]["annotations"], 6)

    def test_different_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = self.create_sources(root)
            output = folder / "instances_val_0based.json"
            output.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                module.prepare(root)
            self.assertEqual(output.read_text(), "{}")
            self.assertFalse((folder / "instances_train_0based.json").exists())

    def test_bad_validation_split_does_not_write_training_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            val = fixture()
            val["categories"][0]["name"] = "different class"
            folder = self.create_sources(root, val=val)
            with self.assertRaises(ValueError):
                module.prepare(root)
            self.assertEqual(sorted(p.name for p in folder.iterdir()),
                             ["instances_train.json", "instances_val.json"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
