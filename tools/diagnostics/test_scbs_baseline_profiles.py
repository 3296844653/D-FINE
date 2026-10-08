"""Independent SCB-S baseline profiles: 80/72 and 110/100; no real data."""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.data.dataloader import BatchImageCollateFunction
from src.data.transforms.container import Compose
from tools.diagnostics.test_scbs_short_schedule import StrongAugmentation, ResizeToBase


PROFILES = ((80, 72), (110, 100))


def config(name):
    return load_config(str(ROOT / "configs/dfine" / name), cfg={})


def model_config(epochs):
    path = str(ROOT / "configs/dfine" / f"dfine_s_scbs_hrw_{epochs}.yml")
    with patch("src.core.yaml_config.load_config", lambda file: load_config(file, cfg={})):
        cfg = YAMLConfig(path)
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["eval_spatial_size"] = [128, 128]
    return cfg


class SCBSBaselineProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_both_profiles_directly_include_official_s_config(self):
        for epochs, stop in PROFILES:
            with self.subTest(epochs=epochs):
                path = ROOT / "configs/dfine" / f"dfine_s_scbs_hrw_{epochs}.yml"
                raw = yaml.safe_load(path.read_text())
                self.assertEqual(raw["__include__"], ["dfine_hgnetv2_s_coco.yml"])
                self.assertEqual(raw["epochs"], epochs)
                self.assertEqual(raw["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"], stop)
                self.assertEqual(raw["train_dataloader"]["collate_fn"]["stop_epoch"], stop)

    def test_only_training_cycle_and_output_directory_differ(self):
        short = config("dfine_s_scbs_hrw_80.yml")
        long = config("dfine_s_scbs_hrw_110.yml")
        expected = copy.deepcopy(short)
        expected["epochs"] = 110
        expected["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"] = 100
        expected["train_dataloader"]["collate_fn"]["stop_epoch"] = 100
        expected["output_dir"] = long["output_dir"]
        self.assertEqual(expected, long)
        self.assertNotEqual(short["output_dir"], long["output_dir"])
        self.assertEqual(short["train_dataloader"]["total_batch_size"], 32)
        self.assertEqual(short["num_classes"], 3)
        self.assertEqual(short["optimizer"]["lr"], 0.0002)
        self.assertEqual(short["lr_warmup_scheduler"]["warmup_duration"], 500)
        for name in ("HybridEncoder", "DFINETransformer", "DFINECriterion"):
            self.assertFalse(any(k.startswith("use_") and v is True for k, v in short[name].items()))

    def test_independent_110_matches_current_baseline(self):
        current = config("dfine_s_scbs_hrw.yml")
        independent = config("dfine_s_scbs_hrw_110.yml")
        current["output_dir"] = independent["output_dir"]
        self.assertEqual(current, independent)

    def test_phase_boundaries_and_second_stage_lengths(self):
        for epochs, stop in PROFILES:
            with self.subTest(epochs=epochs):
                cfg = config(f"dfine_s_scbs_hrw_{epochs}.yml")
                self.assertEqual(cfg["epochs"] - stop, 8 if epochs == 80 else 10)
                self.assertEqual(list(range(stop, cfg["epochs"])), list(range(stop, epochs)))
                policy = cfg["train_dataloader"]["dataset"]["transforms"]["policy"]
                strong, resize = StrongAugmentation(), ResizeToBase()
                transforms = Compose([strong, resize], policy={
                    "name": policy["name"], "epoch": policy["epoch"],
                    "ops": ["StrongAugmentation"],
                })
                for epoch in (stop - 1, stop, epochs - 1):
                    transforms((torch.zeros(1), {}, SimpleNamespace(epoch=epoch)))
                self.assertEqual(strong.calls, 1)
                self.assertEqual(resize.calls, 3)

    def test_multiscale_stops_at_each_profiles_boundary(self):
        for epochs, stop in PROFILES:
            with self.subTest(epochs=epochs):
                values = copy.deepcopy(config(f"dfine_s_scbs_hrw_{epochs}.yml")["train_dataloader"]["collate_fn"])
                values.pop("type")
                collate = BatchImageCollateFunction(**values)
                items = [(torch.zeros(3, 640, 640), {"boxes": torch.empty(0, 4)})]
                with patch("src.data.dataloader.random.choice", return_value=480) as choose:
                    collate.set_epoch(stop - 1)
                    images, _ = collate(items)
                    self.assertEqual(images.shape[-2:], (480, 480))
                    for epoch in (stop, epochs - 1):
                        collate.set_epoch(epoch)
                        images, _ = collate(items)
                        self.assertEqual(images.shape[-2:], (640, 640))
                    choose.assert_called_once()

    def test_each_diagnostic_uses_its_own_independent_profile(self):
        for epochs, stop in PROFILES:
            with self.subTest(epochs=epochs):
                name = f"dfine_s_scbs_hrw_{epochs}_diagnostics.yml"
                raw = yaml.safe_load((ROOT / "configs/dfine" / name).read_text())
                self.assertEqual(raw["__include__"], [f"dfine_s_scbs_hrw_{epochs}.yml"])
                baseline, diagnostic = config(f"dfine_s_scbs_hrw_{epochs}.yml"), config(name)
                baseline.pop("__include__"); diagnostic.pop("__include__")
                baseline.update(
                    export_confusion_matrix=True, export_query_diagnostics=True,
                    query_diagnostic_conf_thresh=0.5, query_diagnostic_iou_thresh=0.5,
                    query_diagnostic_neighbor_iou_thresh=0.8,
                )
                self.assertEqual(baseline, diagnostic)
                self.assertEqual(diagnostic["epochs"], epochs)
                self.assertEqual(diagnostic["train_dataloader"]["collate_fn"]["stop_epoch"], stop)

    def test_both_profiles_build_identical_models_without_downloads(self):
        configs = [model_config(epochs) for epochs, _ in PROFILES]
        for cfg, (epochs, _) in zip(configs, PROFILES):
            self.assertEqual(cfg.epochs, epochs)
        torch.manual_seed(23); short = configs[0].model.eval()
        torch.manual_seed(23); long = configs[1].model.eval()
        self.assertEqual(list(short.state_dict()), list(long.state_dict()))
        for key, value in short.state_dict().items():
            torch.testing.assert_close(value, long.state_dict()[key], rtol=0, atol=0)
        images = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            short_output, long_output = short(images), long(images)
        for key in ("pred_logits", "pred_boxes"):
            torch.testing.assert_close(short_output[key], long_output[key], rtol=0, atol=0)
        self.assertEqual(short_output["pred_logits"].shape, (1, 300, 3))
        self.assertEqual(short_output["pred_boxes"].shape, (1, 300, 4))
        self.assertFalse(hasattr(short.encoder, "mffe"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
