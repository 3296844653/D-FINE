"""SCF synthetic correctness/regression tests, NOT evidence of accuracy gains."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.zoo.dfine.subpixel_context_fusion import SubpixelContextFusion

NAME = "SubpixelContextFusion"


def config(new=True, size=128):
    name = f"dfine_s_scbs_hrw_110_AQS_{NAME}.yml" if new else "dfine_s_scbs_hrw_110_aqs_layer2_r010.yml"
    with patch("src.core.yaml_config.load_config", lambda p: load_config(p, cfg={})):
        cfg = YAMLConfig(str(ROOT / "configs/dfine" / name))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["eval_spatial_size"] = [size, size]
    return cfg


def features(device="cpu", h=7, w=9):
    return (torch.randn(2, 16, 2*h-1, 2*w, device=device, requires_grad=True),
        torch.randn(2, 32, h, w, device=device, requires_grad=True),
        torch.randn(2, 32, (h+1)//2, (w+1)//2, device=device, requires_grad=True))


def targets(device="cpu", empty=False):
    return [dict(labels=torch.tensor([0, 1, 2], device=device),
        boxes=torch.tensor([[.25,.3,.2,.3],[.5,.5,.3,.3],[.8,.7,.15,.2]], device=device))
        if i == 0 and not empty else dict(labels=torch.empty(0, dtype=torch.long, device=device),
        boxes=torch.empty(0, 4, device=device)) for i in range(2)]


class SCFTests(unittest.TestCase):
    def test_phase_coordinates_padding_and_no_wrap(self):
        image = torch.arange(15.).reshape(1,1,3,5)
        cells, valid = SubpixelContextFusion.subcells(image, 2, 3)
        self.assertEqual(cells[0, :, 0, 0, 0].tolist(), [0,1,5,6])
        self.assertEqual(cells[0, :, 0, 1, 2].tolist(), [14,0,0,0])
        self.assertEqual(valid[0, :, 1, 2].tolist(), [True,False,False,False])
        branch = SubpixelContextFusion(32,16,16,dilation=2)
        neighbors = branch.neighbors(torch.arange(9.).reshape(1,1,3,3))
        self.assertEqual(neighbors[0,4,0,0,0].item(),0)
        self.assertEqual(neighbors[0,8,0,0,0].item(),8)
        self.assertEqual(neighbors[0,0,0,0,0].item(),0)
        with self.assertRaisesRegex(ValueError,"aligned"):
            SubpixelContextFusion.subcells(image, 4, 3)

    def test_zero_initialization_then_nonzero_internal_gradients_amp(self):
        module = SubpixelContextFusion(32,16,16)
        x = features()
        out = module(*x)
        torch.testing.assert_close(out,x[1],rtol=0,atol=0)
        out.square().sum().backward()
        self.assertGreater(module.output.weight.grad.abs().sum().item(),0)
        self.assertEqual(module.fine_projection.conv.weight.grad.abs().sum().item(),0)
        module.zero_grad(set_to_none=True)
        torch.nn.init.normal_(module.output.weight,std=.03)
        x = features()
        with torch.autocast("cpu",dtype=torch.bfloat16):
            out = module(*x)
        out.square().sum().backward()
        for name,p in module.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(p.grad.abs().sum().item(),0,name)
        for value in x:
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(),0)

    def test_attention_padding_null_and_structural_ablations(self):
        module = SubpixelContextFusion(32,16,16).eval()
        torch.nn.init.normal_(module.output.weight,std=.03)
        x = features(h=1,w=2)
        out, aux = module(*x,return_aux=True)
        for name,count in (("phase",4),("context",9)):
            weights,valid = aux[name+"_weights"],aux[name+"_valid"].expand(2,-1,-1,-1)
            torch.testing.assert_close(weights.sum(1),torch.ones_like(weights[:,0]),rtol=1e-5,atol=1e-6)
            self.assertEqual(weights[:,:count].masked_select(~valid).count_nonzero().item(),0)
        with torch.no_grad():
            module.phase_null_bias.fill_(80); module.context_null_bias.fill_(80)
        rejected, null_aux = module(*x,return_aux=True)
        torch.testing.assert_close(rejected,x[1],rtol=0,atol=0)
        self.assertLess(null_aux["availability"].max().item(),1e-20)
        module.phase_null_bias.data.zero_(); module.context_null_bias.data.zero_()
        module.phase_selection=False
        _, uniform = module(*x,return_aux=True)
        self.assertEqual(uniform["phase_weights"][0,0,0,0].item(),uniform["phase_weights"][0,4,0,0].item())
        module.context_relations=False
        _, ablated = module(*x,return_aux=True)
        self.assertEqual(ablated["relation"].count_nonzero().item(),0)
        module.enabled=False
        self.assertIs(module(*x),x[1])

    def test_checkpoint_recompute_strict_settings_and_validation(self):
        torch.manual_seed(23); module=SubpixelContextFusion(32,16,16).train()
        torch.nn.init.normal_(module.output.weight,std=.03)
        clone=copy.deepcopy(module); clone.use_checkpoint=False
        a=features(); b=tuple(v.detach().clone().requires_grad_() for v in a)
        module(*a).sum().backward(); clone(*b).sum().backward()
        for (name,p),(other,q) in zip(module.named_parameters(),clone.named_parameters()):
            self.assertEqual(name,other)
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-5)
        clone.load_state_dict(module.state_dict(),strict=True)
        with self.assertRaisesRegex(RuntimeError,"settings disagree"):
            SubpixelContextFusion(32,16,16,dilation=1).load_state_dict(module.state_dict(),strict=True)
        for kwargs in ({"dim":15},{"dilation":True},{"residual_init":0},
                       {"residual_init":True},{"use_checkpoint":1}):
            with self.assertRaises(ValueError): SubpixelContextFusion(**kwargs)
        with self.assertRaises(ValueError): module(features()[0],torch.randn(2,31,7,9),features()[2])

    def test_config_single_variable_baseline_rng_schema_and_optimizer(self):
        from scbs_subpixel_context_experiment import preflight
        preflight(dry=True)
        torch.manual_seed(42); base=config(False).model.eval(); rng=torch.get_rng_state().clone()
        torch.manual_seed(42); cfg=config(); model=cfg.model.eval()
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        for key,value in base.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
        extras=set(model.state_dict())-set(base.state_dict())
        self.assertTrue(all(k.startswith("encoder.subpixel_context_fusion.") for k in extras))
        image=torch.rand(2,3,128,128)
        with torch.no_grad():
            a,b=base(image),model(image)
        for key in a: torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        disabled=config(); disabled.yaml_cfg["HybridEncoder"]["use_subpixel_context_fusion"]=False
        self.assertEqual(set(disabled.model.state_dict()),set(base.state_dict()))
        ids=[id(p) for group in cfg.optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(ids),len(set(ids)))
        for name,p in model.encoder.subpixel_context_fusion.named_parameters():
            self.assertIn(id(p),ids,name)
            if "norm" in name or "bias" in name:
                group=next(g for g in cfg.optimizer.param_groups if any(q is p for q in g["params"]))
                self.assertEqual(group["weight_decay"],0)
        extra=sum(p.numel() for p in model.parameters())-sum(p.numel() for p in base.parameters())
        self.assertEqual(extra,sum(p.numel() for p in model.encoder.subpixel_context_fusion.parameters()))
        print("SCF added parameters:",extra,"raw total:",sum(p.numel() for p in model.parameters()))
        for owner,option,value in (("DFINETransformer","aqs_apply_layer",3),
            ("DFINETransformer","use_discriminative_evidence_l2",True),
            ("HybridEncoder","use_sadf",True),("DFINE","night_probe","encoder_cross_gate")):
            invalid=config(); invalid.yaml_cfg[owner][option]=value
            with self.assertRaises(ValueError): _=invalid.model

    def test_before_pan_p2_routing_and_p4_p5_propagation(self):
        model=config().model.eval(); branch=model.encoder.subpixel_context_fusion
        torch.nn.init.normal_(branch.output.weight,std=.03)
        seen={}
        handles=[model.backbone.register_forward_hook(lambda m,i,o:seen.update(p2=o[0])),
            branch.register_forward_pre_hook(lambda m,i:seen.update(input=i)),
            branch.register_forward_hook(lambda m,i,o:seen.update(output=o)),
            model.encoder.downsample_convs[0].register_forward_pre_hook(lambda m,i:seen.update(pan=i[0]))]
        image=torch.rand(2,3,128,128)
        with torch.no_grad(): on=model(image)
        for h in handles: h.remove()
        self.assertIs(seen["input"][0],seen["p2"])
        self.assertIs(seen["pan"],seen["output"])
        raw=model.backbone(image)
        with torch.no_grad():
            a=model.encoder(raw); branch.enabled=False; b=model.encoder(raw)
        branch.enabled=True
        for i in range(3): self.assertGreater((a[i]-b[i]).abs().sum().item(),0)
        self.assertEqual(on["pred_logits"].shape,(2,300,3))

    def test_full_training_cpu_amp_losses_dn_optimizer_ema_empty(self):
        cfg=config(); model=cfg.model.train()
        torch.nn.init.normal_(model.encoder.subpixel_context_fusion.output.weight,std=.02)
        image=torch.rand(2,3,128,128); gt=targets()
        torch.manual_seed(18)
        with torch.autocast("cpu",dtype=torch.bfloat16):
            on=model(image,gt); losses=cfg.criterion(on,gt); loss=sum(losses.values())
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        for name,p in model.named_parameters():
            if p.grad is not None: self.assertTrue(torch.isfinite(p.grad).all(),name)
        for p in model.encoder.subpixel_context_fusion.parameters(): self.assertIsNotNone(p.grad)
        self.assertIn("dn_outputs",on); self.assertIn("aux_outputs",on)
        model.encoder.subpixel_context_fusion.enabled=False
        torch.manual_seed(18)
        with torch.autocast("cpu",dtype=torch.bfloat16): off=model(image,gt)
        self.assertEqual(set(losses),set(cfg.criterion(off,gt)))
        model.encoder.subpixel_context_fusion.enabled=True
        cfg.optimizer.step(); cfg.ema.update(model)
        clone=config().model; clone.load_state_dict(cfg.ema.module.state_dict(),strict=True)
        self.assertEqual(clone.encoder.subpixel_context_fusion.settings.dtype,torch.int64)
        with torch.no_grad(),torch.autocast("cpu",dtype=torch.bfloat16):
            empty=model(image,targets(empty=True))
            self.assertTrue(all(torch.isfinite(v) for v in cfg.criterion(empty,targets(empty=True)).values()))

    def test_full_640_eval_and_encoder_deploy(self):
        from calflops import calculate_flops
        model=config(size=640).model.eval()
        torch.nn.init.normal_(model.encoder.subpixel_context_fusion.output.weight,std=.02)
        with torch.no_grad():
            image=torch.rand(1,3,640,640); out=model(image)
            self.assertEqual(out["pred_logits"].shape,(1,300,3))
            self.assertTrue(all(torch.isfinite(v).all() for v in out.values()))
            raw=model.backbone(image)
            encoder=copy.deepcopy(model.encoder)
            before=encoder(raw)
            for m in encoder.modules():
                if hasattr(m,"convert_to_deploy"): m.convert_to_deploy()
            after=encoder(raw)
            for a,b in zip(before,after): torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
            calculate_flops(copy.deepcopy(model).deploy(),input_shape=(1,3,640,640),print_results=False)

    def test_runtime_wrapper_eval_guards_and_restoration(self):
        import eval_subpixel_context as wrapper
        from contextlib import redirect_stderr
        from io import StringIO
        original=SubpixelContextFusion.forward
        module=SubpixelContextFusion(32,16,16,use_checkpoint=False).eval()
        torch.nn.init.normal_(module.output.weight,std=.03)
        x=features()
        argv=sys.argv[:]
        for mode in ("off","on"):
            args=["wrapper","--module-mode",mode,"-r","best.pth","--test-only","-u",
                f"diagnostic_SubpixelContextFusion_output_enabled={str(mode=='on').lower()}"]
            def run(path,run_name):
                self.assertIn("--test-only",sys.argv)
                output=module(*x)
                if mode == "off": self.assertIs(output,x[1])
                else: self.assertGreater((output-x[1]).abs().sum().item(),0)
                self.assertTrue(module.enabled)
            with patch.object(sys,"argv",args),patch.object(wrapper.runpy,"run_path",run): wrapper.main()
            self.assertIs(SubpixelContextFusion.forward,original)
            self.assertEqual(sys.argv,argv)
        args=["wrapper","--module-mode","off","-r","best.pth","--test-only","-u",
            "diagnostic_SubpixelContextFusion_output_enabled=false"]
        with patch.object(sys,"argv",args),patch.object(wrapper.runpy,"run_path",side_effect=RuntimeError("eval failed")):
            with self.assertRaises(RuntimeError): wrapper.main()
        self.assertIs(SubpixelContextFusion.forward,original)
        for args in (["wrapper","--module-mode","on","-r","best.pth"],
            ["wrapper","--module-mode","on","--test-only"],
            ["wrapper","--module-mode","on","-r","best.pth","--test-only","-t","other.pth"],
            ["wrapper","--module-mode","on","-r","best.pth","--test-only","-u",
             "diagnostic_SubpixelContextFusion_output_enabled=false"]):
            with patch.object(sys,"argv",args),redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit): wrapper.main()

    def test_on_off_metadata_log_audit_and_zip_excludes_weights(self):
        from scbs_subpixel_context_experiment import on_off,pack,digest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cp=root/"best_stg2.pth"; cp.write_bytes(b"checkpoint fixture")
            row=dict(image_id=1,annotation_id=1,gt_class_id=2,area_original_annotation=100,
                ignore=False,image_name="one.jpg",box_xyxy=[1,2,11,12])
            summary=dict(checkpoint_sha256=digest(cp),weight_source="ema",GT=1,images=1,queries=300,
                class_names=["hand-raising","read","write"],confidence_threshold=.5,IoU_threshold=.5,
                saved_COCO_metrics={"AP":.55},saved_COCO_per_class={},validator_confusion={},
                best_query_cross_classification={},confusion_pairs={})
            for folder,enabled in (("diagnostics_eval",True),("SubpixelContextFusion_off_eval",False)):
                base=root/folder; (base/"analysis").mkdir(parents=True); (base/"evaluation/query_diagnostics").mkdir(parents=True)
                (base/"analysis/summary.json").write_text(json.dumps(summary))
                (base/"analysis/gt_records.jsonl").write_text(json.dumps(row)+"\n")
                meta=dict(complete=True,checkpoint_sha256=digest(cp),config=dict(
                    HybridEncoder={"use_subpixel_context_fusion":True},
                    DFINETransformer={"use_aqs_refine":True,"aqs_apply_layer":2},
                    diagnostic_SubpixelContextFusion_output_enabled=enabled))
                (base/"evaluation/query_diagnostics/metadata.json").write_text(json.dumps(meta))
                (base/"evaluation/eval.pth").write_bytes(b"COCO evaluation fixture")
                (base/"evaluation/best_stg2.pth").write_bytes(b"must not archive")
            (root/"log.txt").write_text(json.dumps(dict(epoch=109,test_coco_eval_bbox=[.55]))+"\npartial")
            on_off(root)
            report=json.loads((root/"SubpixelContextFusion_on_off.json").read_text())
            self.assertTrue(report["ON_matches_training_best_AP"])
            self.assertEqual(report["on_minus_off"]["saved_COCO_metrics"]["AP"],0)
            pack(root)
            with zipfile.ZipFile(root/"SubpixelContextFusion_diagnostics.zip") as z:
                self.assertFalse(any(n.endswith("best_stg2.pth") for n in z.namelist()))
                self.assertIn("diagnostics_eval/evaluation/eval.pth",z.namelist())
            with self.assertRaises(FileExistsError): pack(root)
            metadata=root/"SubpixelContextFusion_off_eval/evaluation/query_diagnostics/metadata.json"
            changed=json.loads(metadata.read_text()); changed["config"]["diagnostic_SubpixelContextFusion_output_enabled"]=True
            metadata.write_text(json.dumps(changed))
            with self.assertRaises(AssertionError): on_off(root)

    def test_launcher_output_preservation_and_reports(self):
        from scbs_subpixel_context_experiment import parameters
        script=ROOT/"tools/diagnostics/run_scbs_AQS_SubpixelContextFusion_110.sh"
        for name in (script,ROOT/"tools/diagnostics/sync_scbs_AQS_SubpixelContextFusion.sh"):
            subprocess.run(["bash","-n",str(name)],check=True)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); env=dict(os.environ,SCBS_SCF_ROOT=tmp)
            subprocess.run(["bash",str(script),"--dry-run"],env=env,check=True,capture_output=True)
            self.assertEqual(list(folder.iterdir()),[])
            out=folder/"dfine_s_scbs_hrw_epochs110_AQS_SubpixelContextFusion_seed0_run1"
            out.mkdir(); marker=out/"old.txt"; marker.write_text("preserve")
            failed=subprocess.run(["bash",str(script),"--dry-run"],env=env,capture_output=True)
            self.assertNotEqual(failed.returncode,0); self.assertEqual(marker.read_text(),"preserve")
            state={"encoder.subpixel_context_fusion."+k:v for k,v in SubpixelContextFusion().state_dict().items()}
            cp=folder/"best.pth"; torch.save(dict(model=state,ema=dict(module=state),last_epoch=105),cp)
            report=folder/"report.json"; parameters(cp,report)
            data=json.loads(report.read_text()); self.assertEqual(data["sources"]["ema"]["dim"],32)
            self.assertAlmostEqual(data["sources"]["ema"]["residual_scale"],.1,places=6)

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable locally; run on server")
    def test_cuda_fp16_full_training(self):
        cfg=config(); model=cfg.model.cuda().train(); gt=targets("cuda")
        torch.nn.init.normal_(model.encoder.subpixel_context_fusion.output.weight,std=.02)
        with torch.autocast("cuda",dtype=torch.float16):
            out=model(torch.rand(2,3,128,128,device="cuda"),gt)
            loss=sum(cfg.criterion(out,gt).values())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name,p in model.named_parameters():
            if p.grad is not None: self.assertTrue(torch.isfinite(p.grad).all(),name)
        torch.cuda.synchronize()

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable locally; no batch32 GPU memory claim")
    def test_cuda_batch32_640_training_smoke(self):
        # One synthetic iteration at nominal resolution/batch, not a guarantee
        # for the dataloader's larger multiscale inputs or real crowded batches.
        cfg=config(size=640); model=cfg.model.cuda().train()
        torch.nn.init.normal_(model.encoder.subpixel_context_fusion.output.weight,std=.02)
        image=torch.rand(32,3,640,640,device="cuda")
        gt=[targets("cuda")[i%2] for i in range(32)]
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda",dtype=torch.float16):
            out=model(image,gt); loss=sum(cfg.criterion(out,gt).values())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name,p in model.named_parameters():
            if p.grad is not None: self.assertTrue(torch.isfinite(p.grad).all(),name)
        torch.cuda.synchronize()
        print(f"SCF CUDA batch32@640 peak allocated={torch.cuda.max_memory_allocated()/2**30:.2f} GiB")


if __name__ == "__main__":
    torch.set_num_threads(2)
    if os.environ.get("SCF_REQUIRE_CUDA_TEST") == "1" and not torch.cuda.is_available():
        raise SystemExit("Launcher requires actual CUDA smoke tests; refusing CPU-only pass")
    unittest.main(verbosity=2)
