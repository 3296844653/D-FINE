"""SCB-S 110/100 schedule checks, preserving old 72/60 and 132/120 runs."""
import copy
import hashlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core.yaml_utils import load_config
from src.data.dataloader import BatchImageCollateFunction
from src.data.transforms.container import Compose


def config(name):
    return load_config(str(ROOT / "configs/dfine" / name), cfg={})


class StrongAugmentation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, sample):
        self.calls += 1
        return sample


class ResizeToBase(StrongAugmentation):
    pass


class SCBSShortScheduleTests(unittest.TestCase):
    def test_original_132_configuration_is_preserved_exactly(self):
        original = ROOT / "configs/dfine/dfine_s_scbs_hrw_132.yml"
        self.assertEqual(
            hashlib.sha256(original.read_bytes()).hexdigest(),
            "18f3614255117de3f966caafde4db2893eaeb39f06d85b77a9b47531ceafc851",
        )
        legacy = config(original.name)
        self.assertEqual(legacy["epochs"], 132)
        self.assertEqual(legacy["train_dataloader"]["collate_fn"]["stop_epoch"], 120)

    def test_only_three_training_schedule_values_changed(self):
        legacy = config("dfine_s_scbs_hrw_132.yml")
        current = config("dfine_s_scbs_hrw.yml")
        expected = copy.deepcopy(legacy)
        expected["epochs"] = 110
        expected["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"] = 100
        expected["train_dataloader"]["collate_fn"]["stop_epoch"] = 100
        self.assertEqual(current, expected)
        self.assertEqual(current["train_dataloader"]["total_batch_size"], 32)
        self.assertEqual(current["optimizer"]["lr"], 0.0002)
        self.assertEqual(current["lr_warmup_scheduler"]["warmup_duration"], 500)

    def test_current_experiments_all_inherit_identical_short_schedule(self):
        baseline = config("dfine_s_scbs_hrw.yml")
        names = (
            "aqs", "mffe", "mffe_p3", "encoder_highres_residual_a03",
            "rw_pairwise_ce_w005",
        )
        for name in names:
            with self.subTest(experiment=name):
                trial = config(f"dfine_s_scbs_hrw_{name}.yml")
                self.assertEqual(trial["epochs"], 110)
                self.assertEqual(trial["train_dataloader"], baseline["train_dataloader"])
                self.assertEqual(trial["val_dataloader"], baseline["val_dataloader"])
                self.assertEqual(trial["optimizer"], baseline["optimizer"])
                self.assertEqual(trial["lr_scheduler"], baseline["lr_scheduler"])
                self.assertEqual(trial["ema"], baseline["ema"])
        stop = baseline["train_dataloader"]["collate_fn"]["stop_epoch"]
        self.assertEqual(list(range(stop, baseline["epochs"])), list(range(100, 110)))

    def test_strong_augmentation_stops_at_epoch_100_but_resize_continues(self):
        policy = config("dfine_s_scbs_hrw.yml")["train_dataloader"]["dataset"]["transforms"]["policy"]
        strong, resize = StrongAugmentation(), ResizeToBase()
        transforms = Compose([strong, resize], policy={
            "name": policy["name"], "epoch": policy["epoch"],
            "ops": ["StrongAugmentation"],
        })
        for epoch in (99, 100, 109):
            dataset = SimpleNamespace(epoch=epoch)
            transforms((torch.zeros(1), {}, dataset))
        self.assertEqual(strong.calls, 1)
        self.assertEqual(resize.calls, 3)

    def test_multiscale_stops_at_epoch_100(self):
        values = copy.deepcopy(config("dfine_s_scbs_hrw.yml")["train_dataloader"]["collate_fn"])
        values.pop("type")
        collate = BatchImageCollateFunction(**values)
        image = torch.zeros(3, 640, 640)
        items = [(image, {"boxes": torch.empty(0, 4)})]
        with patch("src.data.dataloader.random.choice", return_value=480) as choose:
            collate.set_epoch(99)
            images, _ = collate(items)
            self.assertEqual(images.shape[-2:], (480, 480))
            for epoch in (100, 109):
                collate.set_epoch(epoch)
                images, _ = collate(items)
                self.assertEqual(images.shape[-2:], (640, 640))
            choose.assert_called_once()

    def test_old_run1_diagnostics_and_new_baseline_are_separate(self):
        old = config("dfine_s_scbs_hrw_confusion.yml")
        new = config("dfine_s_scbs_hrw_110_diagnostics.yml")
        self.assertEqual(old["epochs"], 132)
        self.assertEqual(new["epochs"], 110)
        self.assertEqual(old["train_dataloader"]["collate_fn"]["stop_epoch"], 120)
        self.assertEqual(new["train_dataloader"]["collate_fn"]["stop_epoch"], 100)
        for key in ("DFINE", "HGNetv2", "HybridEncoder", "DFINETransformer",
                    "DFINECriterion", "DFINEPostProcessor", "val_dataloader"):
            self.assertEqual(old[key], new[key])
        for cfg in (old, new):
            self.assertTrue(cfg["export_confusion_matrix"])
            self.assertTrue(cfg["export_query_diagnostics"])
            self.assertEqual(cfg["query_diagnostic_conf_thresh"], 0.5)
            self.assertEqual(cfg["query_diagnostic_iou_thresh"], 0.5)
        probe = load_config(str(ROOT / "configs/diagnostics/scbs_feature_probe_run1.yml"), cfg={})
        self.assertEqual(probe["detector_config"], "configs/dfine/dfine_s_scbs_hrw_132.yml")

    def test_previous_72_run_configuration_and_diagnostics_are_preserved(self):
        archived = ROOT / "configs/dfine/dfine_s_scbs_hrw_72.yml"
        self.assertEqual(
            hashlib.sha256(archived.read_bytes()).hexdigest(),
            "5238c5bcc2b7fcb45b3b8ee4ddb4a8ff46914f953e32507df6367910f649bb21",
        )
        baseline = config(archived.name)
        diagnostic = config("dfine_s_scbs_hrw_72_diagnostics.yml")
        self.assertEqual(diagnostic["epochs"], 72)
        self.assertEqual(diagnostic["train_dataloader"], baseline["train_dataloader"])
        self.assertEqual(diagnostic["train_dataloader"]["collate_fn"]["stop_epoch"], 60)
        self.assertEqual(diagnostic["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"], 60)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
