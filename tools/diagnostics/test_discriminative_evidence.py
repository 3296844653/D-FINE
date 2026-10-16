"""Synthetic correctness/regression tests, not real-dataset gain evidence."""
import copy
import gc
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.discriminative_evidence import DiscriminativeEvidenceL2
from scbs_discriminative_evidence_experiment import preflight, parameters, on_off
from scbs_instance_evidence_experiment import compare


def config(new=True, size=128):
    name = "dfine_s_scbs_hrw_110_AQS_DiscriminativeEvidenceL2.yml" if new else "dfine_s_scbs_hrw_110_aqs_layer2_r010.yml"
    with patch("src.core.yaml_config.load_config",lambda p:load_config(p,cfg={})):
        cfg = YAMLConfig(str(ROOT/"configs/dfine"/name))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["eval_spatial_size"] = [size,size]
    return cfg


def inputs(count=5, device="cpu"):
    return (torch.randn(2,count,32,device=device,requires_grad=True),
        torch.tensor([.5,.5,.6,.7],device=device).expand(2,count,4).clone().requires_grad_(),
        torch.randn(2,count,3,device=device,requires_grad=True),
        torch.randn(2,16,24,28,device=device,requires_grad=True),
        torch.randn(2,32,12,14,device=device,requires_grad=True),
        torch.randn(2,32,device=device,requires_grad=True))


def targets(device="cpu", empty=False):
    return [{"labels":torch.tensor([0,1,2],device=device),
             "boxes":torch.tensor([[.2,.3,.15,.2],[.5,.5,.2,.3],[.8,.7,.2,.2]],device=device)}
            if i == 0 and not empty else {"labels":torch.empty(0,dtype=torch.long,device=device),
             "boxes":torch.empty(0,4,device=device)} for i in range(2)]


class DiscriminativeEvidenceTests(unittest.TestCase):
    def test_zero_init_gradients_amp_and_no_coordinate_score_gradient(self):
        module = DiscriminativeEvidenceL2(32,16,16,chunk_size=2)
        x = inputs()
        out = module(*x)
        torch.testing.assert_close(out,x[0],rtol=0,atol=0)
        out.square().sum().backward()
        self.assertGreater(module.output.up.weight.grad.abs().sum().item(),0)
        self.assertEqual(module.p2_projection.conv.weight.grad.abs().sum().item(),0)
        self.assertIsNone(x[1].grad); self.assertIsNone(x[2].grad)
        self.assertIsNone(x[5].grad)
        module.zero_grad(set_to_none=True)
        torch.nn.init.normal_(module.output.up.weight,std=.04)
        x = inputs()
        with torch.autocast("cpu",dtype=torch.bfloat16): out = module(*x)
        out.square().sum().backward()
        for name,p in module.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(p.grad.abs().sum().item(),0,name)
        self.assertIsNone(x[1].grad); self.assertIsNone(x[2].grad)
        self.assertIsNone(x[5].grad)
        for i in (0,3,4): self.assertGreater(x[i].grad.abs().sum().item(),0)

    def test_class_hypotheses_null_mass_scale_gate_and_low_score_access(self):
        module = DiscriminativeEvidenceL2(32,16,16).eval()
        q, box, scores, p2, p3, classes = inputs()
        geometry = module.geometry(box)
        projected = (module.p2_projection(p2),module.p3_projection(p3))
        _, pair, weights, gate = module.evidence(q,geometry,*projected,classes)
        torch.testing.assert_close(weights.sum(-1),torch.ones_like(weights[...,0]))
        torch.testing.assert_close(gate.sum(-2),torch.ones_like(gate[...,0,:]))
        self.assertGreater((weights[...,0,:]-weights[...,1,:]).abs().sum().item(),0)
        self.assertEqual(pair.shape,(2,5,2,16))
        with torch.no_grad(): module.null_bias.fill_(30)
        _, rejected, weights, _ = module.evidence(q,geometry,*projected,classes)
        self.assertGreater(weights[...,-1].min().item(),.999)
        self.assertLess(rejected.abs().max().item(),1e-7)
        with torch.no_grad(): module.output.up.bias.fill_(1)
        torch.testing.assert_close(module(q,box,scores,p2,p3,classes),q,atol=2e-6,rtol=0)
        with torch.no_grad(): module.null_bias.zero_(); module.output.up.weight.normal_(std=.05)
        low_scores = torch.full_like(scores,-20)
        self.assertGreater((module(q,box,low_scores,p2,p3,classes)-q).abs().sum().item(),0)

    def test_clipping_empty_permutation_chunk_checkpoint_and_runtime_off(self):
        module = DiscriminativeEvidenceL2(32,16,16,chunk_size=2).eval()
        torch.nn.init.normal_(module.output.up.weight,std=.03)
        q, box, s, p2, p3, classes = inputs()
        out = module(q,box,s,p2,p3,classes)
        changed=q.detach().clone(); changed[:,0]+=10
        torch.testing.assert_close(module(changed,box,s,p2,p3,classes)[:,1:],out[:,1:],rtol=0,atol=0)
        order=torch.tensor([4,2,3,0,1])
        torch.testing.assert_close(module(q[:,order],box[:,order],s[:,order],p2,p3,classes),out[:,order],atol=2e-6,rtol=1e-5)
        outside=box.detach().clone(); outside[:,0]=torch.tensor([2.,2.,.1,.1])
        torch.testing.assert_close(module(q,outside,s,p2,p3,classes)[:,0],q[:,0],rtol=0,atol=0)
        self.assertEqual(module(q[:,:0],box[:,:0],s[:,:0],p2,p3,classes).shape,(2,0,32))
        module.train(); module.use_checkpoint=False
        module(q,box,s,p2,p3,classes).square().sum().backward()
        grads={k:p.grad.clone() for k,p in module.named_parameters()}
        module.zero_grad(set_to_none=True); module.use_checkpoint=True
        module(q,box,s,p2,p3,classes).square().sum().backward()
        for k,p in module.named_parameters(): torch.testing.assert_close(p.grad,grads[k],rtol=2e-5,atol=2e-6)
        saved=copy.deepcopy(module.state_dict()); module.enabled=False
        torch.testing.assert_close(module(q,box,s,p2,p3,classes),q,rtol=0,atol=0)
        for k,v in saved.items(): torch.testing.assert_close(module.state_dict()[k],v,rtol=0,atol=0)

    def test_config_aqs_original_rng_outputs_optimizer_and_loss_unchanged(self):
        preflight(dry=True)
        torch.manual_seed(31); baseline=config(False).model.eval(); rng=torch.get_rng_state().clone()
        torch.manual_seed(31); cfg=config(); model=cfg.model.eval()
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        for k,v in baseline.state_dict().items(): torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
        image=torch.rand(2,3,128,128)
        with torch.no_grad(): before=baseline(image); after=model(image)
        for k in before: torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)
        self.assertEqual(model.decoder.decoder.aqs_layer_indices,(1,))
        ids=[id(p) for g in cfg.optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(ids),len(set(ids)))
        module=model.decoder.decoder.discriminative_evidence_l2
        for name,p in module.named_parameters():
            self.assertIn(id(p),ids)
            if "norm" in name or "bias" in name:
                group=next(g for g in cfg.optimizer.param_groups if any(p is x for x in g["params"]))
                self.assertEqual(group["weight_decay"],0)
        self.assertEqual(cfg.criterion.weight_dict,config(False).criterion.weight_dict)
        print("DiscriminativeEvidenceL2 added parameters:",sum(p.numel() for p in module.parameters()),flush=True)

    def test_aqs_then_evidence_then_l3_dn_and_earlier_bbox_routing(self):
        cfg=config(); model=cfg.model.train(); decoder=model.decoder.decoder
        branch=decoder.discriminative_evidence_l2
        torch.nn.init.normal_(branch.output.up.weight,std=.04)
        captured={}
        def aqs_hook(m,args,out):
            if out.shape[1] == 300: captured["aqs"]=out.detach().clone()
        def branch_hook(m,args,out):
            captured["input"]=args[0].detach().clone(); captured["output"]=out.detach().clone()
        def l3_hook(m,args): captured["l3"]=args[0].detach().clone()
        handles=[decoder.query_cls_refiner.register_forward_hook(aqs_hook),
            branch.register_forward_hook(branch_hook),decoder.layers[2].register_forward_pre_hook(l3_hook)]
        image=torch.rand(2,3,128,128); gt=targets()
        torch.manual_seed(17); on=model(image,gt)
        for h in handles: h.remove()
        torch.testing.assert_close(captured["input"],captured["aqs"],rtol=0,atol=0)
        torch.testing.assert_close(captured["l3"][:,-300:],captured["output"],rtol=0,atol=0)
        self.assertEqual(captured["input"].shape,(2,300,256))
        branch.enabled=False; torch.manual_seed(17); off=model(image,gt); branch.enabled=True
        for key in on["pre_outputs"]:
            torch.testing.assert_close(on["pre_outputs"][key],off["pre_outputs"][key],rtol=0,atol=0)
        # L1 and L2 geometry were computed BEFORE evidence. L3 geometry may change.
        for i in (0,1):
            torch.testing.assert_close(on["aux_outputs"][i]["pred_boxes"],off["aux_outputs"][i]["pred_boxes"],rtol=0,atol=0)
        # Teachers are FINAL-layer outputs and may change through L3. Check
        # only the DN predictions actually computed at L2, not their teachers.
        for key in ("pred_logits","pred_boxes","pred_corners","ref_points"):
            torch.testing.assert_close(on["dn_outputs"][1][key],off["dn_outputs"][1][key],rtol=0,atol=0)
        self.assertGreater((on["pred_logits"]-off["pred_logits"]).abs().sum().item(),0)
        self.assertEqual(set(cfg.criterion(on,gt)),set(cfg.criterion(off,gt)))

    def test_full_original_training_cpu_amp_empty_optimizer_ema_strict_roundtrip(self):
        cfg=config(); model=cfg.model.train()
        branch=model.decoder.decoder.discriminative_evidence_l2
        torch.nn.init.normal_(branch.output.up.weight,std=.02)
        image=torch.rand(2,3,128,128); gt=targets()
        with torch.autocast("cpu",dtype=torch.bfloat16):
            out=model(image,gt); loss=sum(cfg.criterion(out,gt).values())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name,p in model.named_parameters():
            if p.grad is not None: self.assertTrue(torch.isfinite(p.grad).all(),name)
        for p in branch.parameters(): self.assertIsNotNone(p.grad)
        cfg.optimizer.step(); cfg.ema.update(model)
        clone=config().model; clone.load_state_dict(cfg.ema.module.state_dict(),strict=True)
        self.assertEqual(clone.decoder.decoder.discriminative_evidence_l2.settings.dtype,torch.int64)
        with torch.no_grad(),torch.autocast("cpu",dtype=torch.bfloat16):
            empty=model(image,targets(empty=True))
            self.assertTrue(all(torch.isfinite(v) for v in cfg.criterion(empty,targets(empty=True)).values()))

    def test_640_deploy_and_disabled_schema_invalid_settings(self):
        from calflops import calculate_flops
        model=config(size=640).model.eval()
        torch.nn.init.normal_(model.decoder.decoder.discriminative_evidence_l2.output.up.weight,std=.02)
        image=torch.rand(1,3,640,640)
        with torch.no_grad():
            out=model(image)
            self.assertEqual(out["pred_logits"].shape,(1,300,3))
            self.assertTrue(all(torch.isfinite(v).all() for v in out.values()))
            deployed=copy.deepcopy(model).deploy()
            # Hold encoder features/proposals fixed. Full untrained-detector
            # BN fusion can perturb tied TopK scores (also in AQS baseline).
            features=[torch.randn(1,256,s,s) for s in (80,40,20)]
            p2=torch.randn(1,64,160,160)
            before=model.decoder(features,p2_feat=p2)
            after=deployed.decoder(features,p2_feat=p2)
            for k in before: torch.testing.assert_close(after[k],before[k],atol=2e-5,rtol=2e-5)
            calculate_flops(copy.deepcopy(deployed),input_shape=(1,3,640,640),print_results=False)
        old=config(False).model
        self.assertFalse(any("discriminative_evidence" in k for k in old.state_dict()))
        for kwargs in ({"dim":10},{"grid":4},{"scale":0},{"scale":True},{"chunk_size":0},{"use_checkpoint":1}):
            with self.assertRaises(ValueError): DiscriminativeEvidenceL2(**kwargs)
        branch=DiscriminativeEvidenceL2()
        with self.assertRaisesRegex(RuntimeError,"settings differ"):
            DiscriminativeEvidenceL2(grid=5).load_state_dict(branch.state_dict(),strict=True)
        for options in ({"use_aqs_refine":False},{"aqs_apply_layer":3},{"num_classes":6},
                        {"instance_evidence_module":"PartEvidenceSelection"},{"use_rw_local_detail":True}):
            kwargs=dict(num_classes=3,num_layers=3,use_aqs_refine=True,aqs_apply_layer=2,use_discriminative_evidence_l2=True)
            kwargs.update(options)
            with self.assertRaises(ValueError): DFINETransformer(**kwargs)
        cfg=config(); cfg.yaml_cfg["DFINE"]["night_probe"]="decoder_l2_local"
        with self.assertRaises(ValueError): cfg.model

    def test_scripts_output_preservation_parameter_export_and_old_reference_schema(self):
        script=ROOT/"tools/diagnostics/run_scbs_AQS_DiscriminativeEvidenceL2_110.sh"
        subprocess.run(["bash","-n",str(script)],check=True)
        subprocess.run(["bash","-n",str(ROOT/"tools/diagnostics/sync_scbs_AQS_DiscriminativeEvidenceL2.sh")],check=True)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); env=dict(os.environ,SCBS_DER_ROOT=tmp)
            dry=subprocess.run(["bash",str(script),"--dry-run"],env=env,capture_output=True,text=True,check=True)
            self.assertIn("DiscriminativeEvidenceL2",dry.stdout); self.assertEqual(list(root.iterdir()),[])
            out=root/"dfine_s_scbs_hrw_epochs110_AQS_DiscriminativeEvidenceL2_seed0_run1"
            out.mkdir(); marker=out/"old.txt"; marker.write_text("preserve")
            failed=subprocess.run(["bash",str(script),"--dry-run"],env=env,capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0); self.assertEqual(marker.read_text(),"preserve")
            state={"decoder.decoder.discriminative_evidence_l2."+k:v for k,v in DiscriminativeEvidenceL2().state_dict().items()}
            cp=root/"best.pth"; torch.save(dict(model=state,ema=dict(module=state),last_epoch=109),cp)
            report=root/"params.json"; parameters(cp,report)
            self.assertEqual(json.loads(report.read_text())["sources"]["ema"]["settings"][6],2)
            reference=root/"reference"
            summary=dict(checkpoint="/one.pth",checkpoint_sha256="abc",weight_source="ema",GT=1,images=1,
                queries=300,class_names=["hand-raising","read","write"],confidence_threshold=.5,IoU_threshold=.5,
                saved_COCO_metrics={"AP":.5},saved_COCO_per_class={},validator_confusion={},
                best_query_cross_classification={},confusion_pairs={})
            row=dict(image_id=1,annotation_id=3,gt_class_id=2,area_original_annotation=100,
                     ignore=False,image_name="one.jpg",box_xyxy=[0.,0.,10.,10.])
            for folder in (reference/"analysis",out/"diagnostics_eval/analysis"):
                folder.mkdir(parents=True)
                (folder/"summary.json").write_text(json.dumps(summary))
                (folder/"gt_records.jsonl").write_text(json.dumps(row)+"\n")
            meta=reference/"evaluation/query_diagnostics/metadata.json"; meta.parent.mkdir(parents=True)
            decoder=dict(use_aqs_refine=True,aqs_apply_layer=2,aqs_threshold=.5,aqs_residual_init=.1,aqs_temperature=.1)
            meta.write_text(json.dumps(dict(complete=True,checkpoint_sha256="abc",config={"DFINETransformer":decoder})))
            compare(out,reference,"AQS_L2_R1")
            result=json.loads((out/"diagnostics_eval/compare_AQS_L2_R1.json").read_text())
            self.assertIn("saved_COCO_per_class_and_size",result["missing_metrics"]["reference"])
            self.assertIsNone(result["numeric_differences"]["saved_COCO_per_class_and_size"])
            # Same-EMA ablation provenance and optional old size table.
            for folder,enabled in ((out/"diagnostics_eval",True),(out/"DiscriminativeEvidenceL2_off_eval",False)):
                analysis=folder/"analysis"; analysis.mkdir(parents=True,exist_ok=True)
                (analysis/"summary.json").write_text(json.dumps(summary))
                (analysis/"gt_records.jsonl").write_text(json.dumps(row)+"\n")
                metadata=folder/"evaluation/query_diagnostics/metadata.json"
                metadata.parent.mkdir(parents=True)
                mode_decoder=dict(decoder,use_discriminative_evidence_l2=True)
                metadata.write_text(json.dumps(dict(complete=True,checkpoint_sha256="abc",
                    config={"DFINETransformer":mode_decoder,"diagnostic_DiscriminativeEvidenceL2_output_enabled":enabled})))
            on_off(out)
            report=json.loads((out/"DiscriminativeEvidenceL2_on_off.json").read_text())
            self.assertEqual(report["on_minus_off"]["saved_COCO_metrics"]["AP"],0)

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable locally; executed by server launcher")
    def test_cuda_fp16_full_training(self):
        cfg=config(); model=cfg.model.cuda().train(); criterion=cfg.criterion.cuda()
        torch.nn.init.normal_(model.decoder.decoder.discriminative_evidence_l2.output.up.weight,std=.02)
        with torch.autocast("cuda",dtype=torch.float16):
            loss=sum(criterion(model(torch.rand(2,3,128,128,device="cuda"),targets("cuda")),targets("cuda")).values())
        loss.backward(); self.assertTrue(torch.isfinite(loss))
        for name,p in model.named_parameters():
            if p.grad is not None: self.assertTrue(torch.isfinite(p.grad).all(),name)
        del model,cfg,criterion,loss; gc.collect(); torch.cuda.empty_cache()

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable locally; not a full-detector memory guarantee")
    def test_cuda_batch32_300_queries_branch(self):
        module=DiscriminativeEvidenceL2().cuda().train()
        torch.nn.init.normal_(module.output.up.weight,std=.02)
        q=torch.randn(32,300,256,device="cuda",requires_grad=True)
        box=torch.tensor([.5,.5,.3,.4],device="cuda").expand(32,300,4)
        s=torch.randn(32,300,3,device="cuda")
        p2=torch.randn(32,64,160,160,device="cuda",requires_grad=True)
        p3=torch.randn(32,256,80,80,device="cuda",requires_grad=True)
        classes=torch.randn(2,256,device="cuda")
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda",dtype=torch.float16): loss=module(q,box,s,p2,p3,classes).square().mean()
        loss.backward(); self.assertTrue(torch.isfinite(loss))
        print(f"New branch CUDA peak allocated: {torch.cuda.max_memory_allocated()/2**20:.1f} MiB",flush=True)
        del module,q,box,s,p2,p3,classes,loss; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
