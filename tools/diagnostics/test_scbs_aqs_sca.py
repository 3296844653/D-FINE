"""Synthetic checks: SCA changes augmentation only, not AQS/model/loss/protocol."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.data._misc import convert_to_tv_tensor
from src.data.transforms import Compose
from scbs_sca_experiment import BASE,CONFIG,check_config,preflight,pack,digest


def config(path):
    with patch("src.core.yaml_config.load_config",lambda p:load_config(p,cfg={})):
        cfg=YAMLConfig(str(path))
    cfg.yaml_cfg["HGNetv2"]["pretrained"]=False
    cfg.yaml_cfg["eval_spatial_size"]=[128,128]
    return cfg


def sample(epoch=0):
    boxes=convert_to_tv_tensor(torch.tensor([[8.,8.,40.,45.],[45.,14.,88.,56.]]),
        "boxes",box_format="xyxy",spatial_size=(64,96))
    return (Image.new("RGB",(96,64),"white"),
        dict(boxes=boxes,labels=torch.tensor([1,2])),SimpleNamespace(epoch=epoch))


class SCATests(unittest.TestCase):
    def test_only_three_augmentation_parameters_changed(self):
        base=load_config(str(BASE),cfg={}); actual=load_config(str(CONFIG),cfg={})
        preflight(dry=True)
        check_config(actual,base)
        self.assertEqual(actual["DFINETransformer"],base["DFINETransformer"])
        self.assertEqual(actual["DFINECriterion"],base["DFINECriterion"])
        self.assertEqual(actual["optimizer"],base["optimizer"])
        self.assertEqual(actual["HGNetv2"],base["HGNetv2"])
        self.assertEqual(actual["HybridEncoder"],base["HybridEncoder"])
        self.assertEqual(actual["train_dataloader"]["collate_fn"],base["train_dataloader"]["collate_fn"])
        self.assertEqual(actual["val_dataloader"]["dataset"]["transforms"],base["val_dataloader"]["dataset"]["transforms"])
        for bad in ("epochs","batch","AQS","loss","SCF","crop_min_scale","operator_order","valfree"):
            changed=copy.deepcopy(actual)
            if bad == "epochs": changed["epochs"]=132
            elif bad == "batch": changed["train_dataloader"]["total_batch_size"]=16
            elif bad == "AQS": changed["DFINETransformer"]["aqs_residual_init"]=.05
            elif bad == "loss": changed["DFINECriterion"]["weight_dict"]["loss_vfl"]=2
            elif bad == "SCF": changed["HybridEncoder"]["use_subpixel_context_fusion"]=True
            elif bad == "crop_min_scale": changed["train_dataloader"]["dataset"]["transforms"]["ops"][2]["min_scale"]=.5
            elif bad == "operator_order": changed["train_dataloader"]["dataset"]["transforms"]["ops"].reverse()
            elif bad == "valfree": changed["val_free"]=True
            with self.assertRaises(AssertionError,msg=bad): check_config(changed,base)

    def test_real_registered_transforms_and_bounding_box_validity(self):
        params=load_config(str(CONFIG),cfg={})["train_dataloader"]["dataset"]["transforms"]
        compose=Compose(**copy.deepcopy({k:v for k,v in params.items() if k != "type"}))
        zoom,crop=compose.transforms[1:3]
        self.assertEqual(list(zoom.side_range),[1.,2.]); self.assertEqual(zoom.p,.3)
        self.assertEqual(crop.p,.5); self.assertEqual(crop.min_scale,.3)
        self.assertEqual(compose.policy["epoch"],100)
        for seed in range(20):
            torch.manual_seed(seed)
            image,target,dataset=compose(*sample())
            self.assertEqual(image.shape,(3,640,640)); self.assertEqual(image.dtype,torch.float32)
            self.assertEqual(target["boxes"].shape[0],target["labels"].shape[0])
            self.assertTrue(torch.isfinite(target["boxes"]).all())
            self.assertTrue(((target["boxes"] >= 0) & (target["boxes"] <= 1)).all())
            self.assertTrue(all(v in (1,2) for v in target["labels"].tolist()))
            self.assertEqual(dataset.epoch,0)

    def test_stop_epoch_skips_all_three_strong_augmentations(self):
        params=load_config(str(CONFIG),cfg={})["train_dataloader"]["dataset"]["transforms"]
        compose=Compose(**copy.deepcopy({k:v for k,v in params.items() if k != "type"}))
        def fail(self,*args): raise RuntimeError("Strong augmentation was called")
        for i in (0,1,2):
            name=type(compose.transforms[i]).__name__
            compose.transforms[i]=type(name,(torch.nn.Module,),dict(forward=fail))()
        image,target,_=compose(*sample(epoch=100))
        self.assertEqual(image.shape,(3,640,640)); self.assertEqual(len(target["labels"]),2)
        with self.assertRaisesRegex(RuntimeError,"Strong augmentation"):
            compose(*sample(epoch=99))

    def test_same_model_schema_weights_rng_predictions_and_parameter_count(self):
        torch.manual_seed(63); base=config(BASE).model.eval(); rng=torch.get_rng_state().clone()
        torch.manual_seed(63); trial=config(CONFIG).model.eval()
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        self.assertEqual(set(base.state_dict()),set(trial.state_dict()))
        for k,v in base.state_dict().items(): torch.testing.assert_close(v,trial.state_dict()[k],rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in base.parameters()),sum(p.numel() for p in trial.parameters()))
        self.assertFalse(any("subpixel_context_fusion" in k or "discriminative_evidence" in k for k in trial.state_dict()))
        image=torch.rand(2,3,128,128)
        with torch.no_grad(): a,b=base(image),trial(image)
        for k in a: torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)

    def test_launcher_syntax_dry_run_busy_gpu_and_existing_output_preservation(self):
        script=ROOT/"tools/diagnostics/run_scbs_AQS_SCA_110.sh"
        for s in (script,ROOT/"tools/diagnostics/sync_scbs_AQS_SCA.sh"):
            subprocess.run(["bash","-n",str(s)],check=True)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); env=dict(os.environ,SCBS_SCA_ROOT=tmp)
            dry=subprocess.run(["bash",str(script),"--dry-run"],env=env,capture_output=True,text=True)
            self.assertEqual(dry.returncode,0,dry.stderr); self.assertEqual(list(root.iterdir()),[])
            # Fake Conda only simulates an occupied GPU; no actual training.
            fake=root/"fake_conda"
            fake.write_text("#!/usr/bin/env python3\nimport sys\nsys.stdin.read()\nprint('GPU busy fixture')\nsys.exit(9)\n")
            fake.chmod(0o700); env["SCBS_SCA_CONDA"]=str(fake)
            failed=subprocess.run(["bash",str(script)],env=env,capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0)
            out=root/"dfine_s_scbs_hrw_epochs110_AQS_SCA_seed0_run1"
            self.assertFalse(out.exists())
            out.mkdir(); marker=out/"old.txt"; marker.write_text("keep")
            failed=subprocess.run(["bash",str(script),"--dry-run"],env=env,capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0); self.assertEqual(marker.read_text(),"keep")
            for args in (("--dry-run","../run1"),("--dry-run","run0"),("--dry-run","run1","extra")):
                self.assertNotEqual(subprocess.run(["bash",str(script),*args],env=env,capture_output=True).returncode,0)

    def test_sync_does_not_replace_model_or_original_configs(self):
        manifest=(ROOT/"tools/diagnostics/scbs_AQS_SCA_sync_files.txt").read_text().splitlines()
        configs=[p for p in manifest if p.startswith("configs/")]
        self.assertEqual(configs,["configs/dfine/dfine_s_scbs_hrw_110_AQS_SCA.yml"])
        self.assertFalse(any(p.startswith("src/") for p in manifest))
        for p in manifest: self.assertTrue((ROOT/p).is_file(),p)

    def test_diagnostic_zip_excludes_trained_weights_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); folder=root/"diagnostics_eval/evaluation"; folder.mkdir(parents=True)
            cp=root/"best_stg2.pth"; cp.write_bytes(b"must not modify or zip")
            original=digest(cp)
            (folder/"eval.pth").write_bytes(b"COCO evaluation fixture")
            (folder/"best_stg2.pth").write_bytes(b"must not archive")
            (folder/"confusion_matrix.json").write_text("{}")
            pack(root)
            with zipfile.ZipFile(root/"AQS_SCA_diagnostics.zip") as z:
                self.assertIn("diagnostics_eval/evaluation/eval.pth",z.namelist())
                self.assertFalse(any(n.endswith("best_stg2.pth") for n in z.namelist()))
            self.assertEqual(digest(cp),original)
            with self.assertRaises(FileExistsError): pack(root)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
