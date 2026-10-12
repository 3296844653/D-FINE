"""Synthetic integration checks, NOT evidence of experimental effectiveness."""
import copy
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.zoo.dfine.finegrain_trials import (
    FINEGRAIN_TRIALS, BoxScaleContrastL2, PartInteractionL2, PairEvidenceHead,
    WindowInteractionP3, sample_points, relative_grid,
)


def config(mode=None, size=128):
    suffix = "" if mode is None else "_"+mode
    with patch("src.core.yaml_config.load_config", lambda path: load_config(path, cfg={})):
        cfg = YAMLConfig(str(ROOT/f"configs/dfine/dfine_s_scbs_hrw_110{suffix}.yml"))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["eval_spatial_size"] = [size, size]
    return cfg


def targets(device="cpu", empty=False):
    data = [dict(labels=torch.empty(0, dtype=torch.long, device=device),
                 boxes=torch.empty(0, 4, device=device)) for _ in range(2)]
    if not empty:
        data[0] = dict(labels=torch.tensor([0,1,2], device=device),
                      boxes=torch.tensor([[.2,.3,.2,.2],[.5,.5,.3,.3],[.8,.7,.2,.2]], device=device))
    return data


def parameters(model):
    return {k:p for k,p in model.named_parameters() if "finegrain_" in k}


def activate(model):
    for name, module in model.named_modules():
        if "finegrain_" in name and (name.endswith(".output") or name.endswith(".fuse.3")):
            torch.nn.init.normal_(module.weight, std=.02)


class FinegrainTests(unittest.TestCase):
    def test_config_identity_rng_optimizer(self):
        torch.manual_seed(23); baseline = config().model.eval()
        original, rng = baseline.state_dict(), torch.get_rng_state().clone()
        images = torch.rand(2,3,128,128)
        with torch.no_grad(): expected = baseline(images)
        base_config = config().yaml_cfg
        for mode in FINEGRAIN_TRIALS:
            with self.subTest(mode=mode):
                cfg = config(mode)
                before, after = copy.deepcopy(base_config), copy.deepcopy(cfg.yaml_cfg)
                trial = {k:v for k,v in after["DFINE"].items() if k.startswith("finegrain_")}
                before["DFINE"].update(trial)
                for item in (before, after):
                    item.pop("__include__", None); item.pop("output_dir", None)
                self.assertEqual(before, after)
                torch.manual_seed(23); model = cfg.model.eval()
                torch.testing.assert_close(torch.get_rng_state(), rng, atol=0,rtol=0)
                for key, value in original.items():
                    torch.testing.assert_close(model.state_dict()[key], value, atol=0,rtol=0)
                with torch.no_grad(): actual = model(images)
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], atol=0,rtol=0)
                ids = [id(p) for group in cfg.optimizer.param_groups for p in group["params"]]
                self.assertEqual(len(ids),len(set(ids)))
                self.assertTrue(all(id(p) in ids for p in parameters(model).values()))
                groups={id(p):g for g in cfg.optimizer.param_groups for p in g["params"]}
                for name,p in parameters(model).items():
                    if "norm" in name or name.endswith(".bias"):
                        self.assertEqual(groups[id(p)]["weight_decay"],0,name)
                self.assertFalse(cfg.criterion.use_class_margin)
                self.assertFalse(cfg.criterion.use_pairwise_ce)
                self.assertFalse(model.decoder.use_aqs_refine)
                print(mode,"added parameters:",sum(p.numel() for p in parameters(model).values()),flush=True)

    def test_original_loss_gradients_amp_dn_and_empty(self):
        base = config(); base_model=base.model.train()
        gt=targets(); images=torch.rand(2,3,128,128)
        base_keys=set(base.criterion(base_model(images,gt),gt))
        for mode in FINEGRAIN_TRIALS:
            with self.subTest(mode=mode):
                cfg=config(mode); model=cfg.model.train()
                losses=cfg.criterion(model(images,gt),gt)
                self.assertEqual(set(losses),base_keys)
                sum(losses.values()).backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in parameters(model).values()))
                activate(model)
                for empty in (False,True):
                    model.zero_grad(set_to_none=True); gt=targets(empty=empty)
                    with torch.autocast("cpu",dtype=torch.bfloat16):
                        output=model(images,gt); losses=cfg.criterion(output,gt)
                    self.assertTrue(all(torch.isfinite(v).all() for v in losses.values()))
                    sum(losses.values()).backward()
                    for name,p in parameters(model).items():
                        self.assertIsNotNone(p.grad,name)
                        self.assertTrue(torch.isfinite(p.grad).all(),name)
                    self.assertEqual(output["pred_logits"].shape,(2,300,3))

    def test_roi_detached_boxes_masks_and_query_independence(self):
        q=torch.randn(2,5,32,requires_grad=True)
        boxes=torch.tensor([.5,.5,.6,.7]).expand(2,5,4).clone().requires_grad_()
        p3=torch.randn(2,32,12,16,requires_grad=True)
        p4=torch.randn(2,32,6,8,requires_grad=True)
        for cls in (BoxScaleContrastL2,PartInteractionL2):
            module=cls(channels=32,dim=16)
            torch.nn.init.normal_(module.fuse[-1].weight,std=.1)
            output=module(q,boxes,p3,p4); output.square().mean().backward()
            self.assertIsNone(boxes.grad)
            for name,p in module.named_parameters():
                self.assertIsNotNone(p.grad,name); self.assertTrue(torch.isfinite(p.grad).all(),name)
            changed=q.detach().clone(); changed[:,0]+=4
            torch.testing.assert_close(module(changed,boxes,p3,p4)[:,1:],output[:,1:],atol=0,rtol=0)
            outside=torch.tensor([2.,2.,.1,.1]).expand(2,5,4)
            torch.testing.assert_close(module(q,outside,p3,p4),q,atol=0,rtol=0)
        with torch.autocast("cpu",dtype=torch.bfloat16):
            values,valid=sample_points(p3.bfloat16(),boxes.bfloat16(),relative_grid(5))
        self.assertEqual(values.dtype,torch.bfloat16); self.assertTrue(valid.all())

    def test_window_padding_and_invalid_settings(self):
        for shape in ((7,11),(16,16),(3,4)):
            module=WindowInteractionP3(32,16,window=5)
            torch.nn.init.normal_(module.output.weight,std=.1)
            feature=torch.randn(2,32,*shape,requires_grad=True)
            output=module(feature); self.assertEqual(output.shape,feature.shape)
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(feature.grad).all())
        for kwargs in (dict(dim=17),dict(scale=0),dict(grid=4),dict(scale=float("nan"))):
            with self.assertRaises(ValueError): BoxScaleContrastL2(**kwargs)
        with self.assertRaises(ValueError): PartInteractionL2(parts=3)
        for owner,flag in (("HybridEncoder","use_sadf"),("DFINETransformer","use_aqs_refine")):
            cfg=config(FINEGRAIN_TRIALS[0]); cfg.yaml_cfg[owner][flag]=True
            with self.assertRaisesRegex(ValueError,"stack|independent"): _=cfg.model

    def test_encoder_p4_p5_direct_path_preserved(self):
        for mode in ("DetailOnlyCrossGate","WindowInteractionP3"):
            model=config(mode).model.eval(); activate(model)
            encoded,received=[],[]
            h1=model.encoder.register_forward_hook(lambda module,args,out: encoded.append(out))
            h2=model.decoder.register_forward_pre_hook(lambda module,args: received.append(args[0]))
            with torch.no_grad(): model(torch.rand(2,3,128,128))
            h1.remove(); h2.remove()
            for i in (1,2): self.assertIs(encoded[0][i],received[0][i])
            self.assertGreater((encoded[0][0]-received[0][0]).abs().sum().item(),0)

    def test_pair_head_fixed_bbox_hand_dn_teacher_and_independent_delta(self):
        model=config("PairEvidenceHead").model.train(); activate(model)
        images,gt=torch.rand(2,3,128,128),targets()
        torch.manual_seed(31); actual=model(images,gt)
        model.decoder.finegrain_trial="none"
        torch.manual_seed(31); original=model(images,gt)
        model.decoder.finegrain_trial="PairEvidenceHead"
        for key in ("pred_boxes","pred_corners","ref_points"):
            torch.testing.assert_close(actual[key],original[key],atol=0,rtol=0)
        torch.testing.assert_close(actual["pred_logits"][...,0],original["pred_logits"][...,0],atol=0,rtol=0)
        delta=actual["pred_logits"]-original["pred_logits"]
        self.assertGreater((delta[...,1]+delta[...,2]).abs().sum().item(),0)
        for group in ("dn_outputs","aux_outputs","enc_aux_outputs"):
            for new,old in zip(actual[group],original[group]):
                for key in old:
                    if torch.is_tensor(old[key]): torch.testing.assert_close(new[key],old[key],atol=0,rtol=0)

    def test_l2_routes_ordinary_only_current_bbox_preserved(self):
        images,gt=torch.rand(2,3,128,128),targets()
        for mode in ("BoxScaleContrastL2","PartInteractionL2"):
            model=config(mode).model.train(); activate(model)
            observed=[]
            hook=model.decoder.decoder.finegrain_local_adapter.register_forward_pre_hook(
                lambda module,args: observed.append(args[0].shape[1]))
            torch.manual_seed(37); actual=model(images,gt); hook.remove()
            adapter=model.decoder.decoder.finegrain_local_adapter
            del model.decoder.decoder.finegrain_local_adapter
            torch.manual_seed(37); original=model(images,gt)
            model.decoder.decoder.finegrain_local_adapter=adapter
            self.assertEqual(observed,[300])
            # L2 bbox produced BEFORE this residual; L3 bbox may change.
            torch.testing.assert_close(actual["aux_outputs"][1]["pred_boxes"],
                                       original["aux_outputs"][1]["pred_boxes"],atol=0,rtol=0)
            # Only L1/L2 DN predictions are unchanged. The original DN mask
            # permits DN -> ordinary attention; L3 DN can change indirectly.
            # Do NOT change baseline attention masking to force isolation.
            for new,old in zip(actual["dn_outputs"][:2],original["dn_outputs"][:2]):
                torch.testing.assert_close(new["pred_boxes"],old["pred_boxes"],atol=0,rtol=0)
                torch.testing.assert_close(new["pred_logits"],old["pred_logits"],atol=0,rtol=0)

    def test_checkpoint_settings_and_deploy(self):
        for mode in FINEGRAIN_TRIALS:
            model=config(mode).model.eval(); activate(model)
            buffer=io.BytesIO(); torch.save(model.state_dict(),buffer); buffer.seek(0)
            restored=config(mode).model.eval()
            restored.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
            wrong=config(mode); wrong.yaml_cfg["DFINE"]["finegrain_residual_scale"] = .2
            with self.assertRaisesRegex(RuntimeError,"settings"): wrong.model.load_state_dict(model.state_dict())
            with torch.no_grad():
                deployed=copy.deepcopy(model).deploy()
                feats=[torch.randn(2,256,side,side) for side in (16,8,4)]
                expected,actual=model.decoder(feats),deployed.decoder(feats)
                for key in expected: torch.testing.assert_close(actual[key],expected[key],atol=1e-5,rtol=1e-5)
                self.assertTrue(all(torch.isfinite(v).all() for v in deployed(torch.rand(2,3,128,128)).values()))

    def test_ema_fixed_settings_load_without_drift(self):
        from src.optim.ema import ModelEMA
        for mode in FINEGRAIN_TRIALS:
            cfg=config(mode); model=cfg.model.eval()
            ema=ModelEMA(model,decay=.9999,warmups=0)
            # Repeated arithmetic on float settings would eventually drift.
            # Integer encodings bypass EMA's float-only averaging unchanged.
            for _ in range(5): ema.update(model)
            restored=config(mode).model.eval()
            restored.load_state_dict(ema.module.state_dict(),strict=True)
            for name,value in model.state_dict().items():
                if "finegrain_" in name and name.endswith("settings"):
                    self.assertEqual(value.dtype,torch.int64)
                    torch.testing.assert_close(ema.module.state_dict()[name],value,atol=0,rtol=0)

    def test_full_640_and_calflops(self):
        from calflops import calculate_flops
        for mode in FINEGRAIN_TRIALS:
            with self.subTest(mode=mode):
                model=config(mode,size=640).model.eval(); activate(model)
                with torch.no_grad(): output=model(torch.rand(1,3,640,640))
                self.assertEqual(output["pred_logits"].shape,(1,300,3))
                self.assertTrue(all(torch.isfinite(v).all() for v in output.values()))
                calculate_flops(copy.deepcopy(model).deploy(),input_shape=(1,3,640,640),print_results=False)

    def test_launcher_dry_run_no_overwrite(self):
        script=ROOT/"tools/diagnostics/run_scbs_finegrain_five_110.sh"
        subprocess.run(["bash","-n",str(script)],check=True)
        with tempfile.TemporaryDirectory(prefix="dfine-finegrain-test-") as directory:
            env=dict(os.environ,FINEGRAIN_OUTPUT_ROOT=directory,FINEGRAIN_PYTHON=sys.executable)
            result=subprocess.run(["bash",str(script),"--dry-run"],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(list(Path(directory).iterdir()),[])
            output=Path(directory)/"dfine_s_scbs_hrw_epochs110_BoxScaleContrastL2_seed0_run1"
            output.mkdir()
            result=subprocess.run(["bash",str(script),"--dry-run","BoxScaleContrastL2"],env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0); self.assertIn("已存在",result.stderr)

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA not available locally")
    def test_cuda_fp16_all_five(self):
        for mode in FINEGRAIN_TRIALS:
            with self.subTest(mode=mode):
                cfg=config(mode); model=cfg.model.cuda().train(); activate(model)
                gt=targets("cuda")
                with torch.autocast("cuda",dtype=torch.float16):
                    losses=cfg.criterion(model(torch.rand(2,3,128,128,device="cuda"),gt),gt)
                self.assertTrue(all(torch.isfinite(v).all() for v in losses.values()))
                sum(losses.values()).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters(model).values()))
                del model,cfg,losses
                torch.cuda.empty_cache()


if __name__=="__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
