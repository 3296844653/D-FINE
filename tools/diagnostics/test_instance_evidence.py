"""Synthetic correctness tests, NOT evidence of real-dataset accuracy gains."""
import copy
import gc
import json
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
from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.instance_evidence import INSTANCE_EVIDENCE_MODULES, build_instance_evidence

BASE = "dfine_s_scbs_hrw_110_aqs_layer2_r010.yml"


def config(mode=None, size=128):
    filename = BASE if mode is None else f"dfine_s_scbs_hrw_110_AQS_{mode}.yml"
    with patch("src.core.yaml_config.load_config", lambda p: load_config(p, cfg={})):
        cfg = YAMLConfig(str(ROOT / "configs/dfine" / filename))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["eval_spatial_size"] = [size, size]
    return cfg


def small_branch(mode, **kwargs):
    return build_instance_evidence(mode, hidden_dim=32, p2_channels=16, dim=16, **kwargs)


def branch_inputs(device="cpu", count=5):
    return (torch.randn(2, count, 32, device=device, requires_grad=True),
            torch.tensor([.5, .5, .6, .7], device=device).expand(2, count, 4).clone().requires_grad_(),
            torch.randn(2, 16, 24, 28, device=device, requires_grad=True),
            torch.randn(2, 32, 12, 14, device=device, requires_grad=True))


def targets(device="cpu", empty=False):
    return [{"labels": torch.empty(0, dtype=torch.long, device=device),
             "boxes": torch.empty(0, 4, device=device)} if empty or i == 1 else
            {"labels": torch.tensor([0, 1, 2], device=device),
             "boxes": torch.tensor([[.2,.3,.15,.2],[.5,.5,.2,.3],[.8,.7,.2,.2]], device=device)}
            for i in range(2)]


class InstanceEvidenceTests(unittest.TestCase):
    def test_zero_initialization_then_all_gradients_no_box_grad_cpu_amp(self):
        for mode in INSTANCE_EVIDENCE_MODULES:
            with self.subTest(mode=mode):
                branch = small_branch(mode, chunk_size=2)
                inputs = branch_inputs()
                delta = branch(*inputs)
                torch.testing.assert_close(delta, torch.zeros_like(delta), rtol=0, atol=0)
                (delta * torch.randn_like(delta)).sum().backward()
                self.assertGreater(branch.output.up.weight.grad.abs().sum().item(), 0)
                self.assertEqual(branch.p2_projection.conv.weight.grad.abs().sum().item(), 0)
                self.assertIsNone(inputs[1].grad)
                branch.zero_grad(set_to_none=True)
                torch.nn.init.normal_(branch.output.up.weight, std=.05)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    output = branch(*inputs)
                    output.square().sum().backward()
                for name, parameter in branch.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                for index in (0, 2, 3):
                    self.assertGreater(inputs[index].grad.abs().sum().item(), 0)
                self.assertIsNone(inputs[1].grad)
                self.assertTrue(torch.isfinite(output).all())
                torch.testing.assert_close(output[...,0], torch.zeros_like(output[...,0]), rtol=0, atol=0)

    def test_instance_isolation_clipping_empty_and_chunk_checkpoint_equivalence(self):
        for mode in INSTANCE_EVIDENCE_MODULES:
            with self.subTest(mode=mode):
                branch = small_branch(mode, chunk_size=2).eval()
                torch.nn.init.normal_(branch.output.up.weight, std=.05)
                query, box, p2, p3 = branch_inputs()
                actual = branch(query, box, p2, p3)
                changed = query.detach().clone(); changed[:,0] += 20
                torch.testing.assert_close(branch(changed, box, p2, p3)[:,1:], actual[:,1:], rtol=0, atol=0)
                order = torch.tensor([4, 1, 3, 0, 2])
                torch.testing.assert_close(branch(query[:,order], box[:,order], p2, p3), actual[:,order], atol=2e-6, rtol=2e-5)
                self.assertEqual(branch(query[:,:0], box[:,:0], p2, p3).shape, (2,0,3))
                outside = box.detach().clone(); outside[:,0] = torch.tensor([2.,2.,.1,.1])
                torch.testing.assert_close(branch(query, outside, p2, p3)[:,0], torch.zeros(2,3), rtol=0, atol=0)
                branch.train(); branch.use_checkpoint = False
                expected = branch(query, box, p2, p3)
                expected.square().sum().backward()
                gradients = {k:p.grad.clone() for k,p in branch.named_parameters()}
                branch.zero_grad(set_to_none=True); branch.use_checkpoint = True
                branch(query, box, p2, p3).square().sum().backward()
                for name, parameter in branch.named_parameters():
                    torch.testing.assert_close(parameter.grad, gradients[name], rtol=1e-5, atol=2e-6)

    def test_distinct_mechanisms_dual_stream_quantiles_and_directed_edges(self):
        q, boxes, p2, p3 = branch_inputs()
        part = small_branch("PartEvidenceSelection")
        geometry = part.clipped_geometry(boxes)
        content = part.query_projection(q)
        _, classes, assign, select, available = part.region_streams(
            content, part.p2_projection(p2), part.p3_projection(p3), geometry)
        torch.testing.assert_close(classes.sum(-1), torch.ones_like(classes[...,0]))
        torch.testing.assert_close(assign.sum(-1), torch.ones_like(assign[...,0]))
        torch.testing.assert_close(select.sum(-1), torch.ones_like(available))
        torch.testing.assert_close(available.sum(-1) + assign[..., -1].mean(-1), torch.ones_like(available[...,0]))
        with torch.no_grad():
            part.region_selection.weight.zero_(); part.region_selection.bias.zero_()
            part.region_selection.bias[-1] = 20
        *_, rejected = part.region_streams(content, part.p2_projection(p2), part.p3_projection(p3), geometry)
        self.assertLess(rejected.max().item(), 1e-7)

        sampling = small_branch("TrilinearDetailSampling")
        density = torch.ones(2, 5, 7, requires_grad=True)
        quantiles = sampling.inverse_cdf(density, 5)
        expected = (torch.arange(5)+.5)/5-.5
        torch.testing.assert_close(quantiles, expected.expand_as(quantiles), atol=1e-6, rtol=1e-5)
        concentrated = density.detach().clone(); concentrated[..., -1] = 100
        self.assertGreater(sampling.inverse_cdf(concentrated, 5).mean().item(), .3)
        quantiles.square().sum().backward()
        self.assertGreater(density.grad.abs().sum().item(), 0)
        tokens = torch.randn(2,5,49,16)
        attention = sampling.trilinear_attention(tokens)
        x = tokens.transpose(-1,-2)
        reference = ((x.softmax(-1) @ x.transpose(-1,-2)).softmax(-1) @ x).softmax(-1)
        torch.testing.assert_close(attention, reference)
        self.assertEqual(sampling.quantile_grid(attention.mean(-2)).shape, (2,5,25,2))

        interaction = small_branch("LocalInteractionEvidence")
        geometry = interaction.clipped_geometry(boxes)
        evidence, attention, weights, positions = interaction.interaction_evidence(
            interaction.query_projection(q), interaction.tokens(
                interaction.p2_projection(p2), interaction.p3_projection(p3), geometry))
        self.assertEqual(evidence.shape, (2,5,2,16))
        self.assertEqual(interaction.edges.shape, (6,2))
        self.assertTrue((interaction.edges[:,0] != interaction.edges[:,1]).all())
        torch.testing.assert_close(attention.sum(-1), torch.ones_like(positions[...,0]))
        torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[...,0]))
        self.assertGreater((weights[...,0,:]-weights[...,1,:]).abs().sum().item(), 0)

    def test_configs_only_add_one_module_to_aqs_and_preserve_initialization(self):
        torch.manual_seed(31); base_cfg = config(); base = base_cfg.model.eval()
        rng = torch.get_rng_state().clone()
        expected = copy.deepcopy(base_cfg.yaml_cfg)
        expected.pop("__include__", None); expected.pop("output_dir", None)
        images = torch.rand(2,3,128,128)
        with torch.no_grad(): original = base(images)
        for mode in INSTANCE_EVIDENCE_MODULES:
            with self.subTest(mode=mode):
                cfg = config(mode)
                actual = copy.deepcopy(cfg.yaml_cfg)
                actual.pop("__include__", None); actual.pop("output_dir", None)
                wanted = copy.deepcopy(expected)
                wanted["HGNetv2"]["return_idx"] = [0,1,2,3]
                wanted["DFINETransformer"].update(instance_evidence_module=mode,
                    instance_evidence_dim=64, instance_evidence_grid=7, instance_evidence_parts=3,
                    instance_evidence_p2_channels=64, instance_evidence_scale=.1,
                    instance_evidence_chunk=32, instance_evidence_checkpoint=True)
                self.assertEqual(wanted, actual)
                torch.manual_seed(31); model = cfg.model.eval()
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                for key, value in base.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
                with torch.no_grad(): enhanced = model(images)
                for key in original:
                    torch.testing.assert_close(enhanced[key], original[key], rtol=0, atol=0)
                self.assertEqual(model.decoder.decoder.aqs_layer_indices, (1,))
                self.assertFalse(any("margin" in k or "pairwise" in k for k in cfg.criterion.weight_dict))
                identities = [id(p) for group in cfg.optimizer.param_groups for p in group["params"]]
                self.assertEqual(len(identities), len(set(identities)))
                self.assertTrue(all(id(p) in identities for p in model.decoder.instance_evidence.parameters()))
                for name, parameter in model.decoder.instance_evidence.named_parameters():
                    if "norm" in name or "bias" in name:
                        group = next(g for g in cfg.optimizer.param_groups if any(p is parameter for p in g["params"]))
                        self.assertEqual(group["weight_decay"], 0)
                added = sum(p.numel() for p in model.decoder.instance_evidence.parameters())
                total = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"{mode}: added={added:,}, total trainable={total:,}", flush=True)
                del model, cfg; gc.collect()

    def test_full_detector_original_losses_dn_aux_teacher_and_checkpoint_ema(self):
        for mode in INSTANCE_EVIDENCE_MODULES:
            with self.subTest(mode=mode):
                cfg = config(mode); model = cfg.model.train()
                branch = model.decoder.instance_evidence
                torch.nn.init.normal_(branch.output.up.weight, std=.02)
                images = torch.rand(2,3,128,128); gt = targets()
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    torch.manual_seed(17); enhanced = model(images, gt)
                    model.decoder.instance_evidence_module = "none"
                    torch.manual_seed(17); old = model(images, gt)
                    model.decoder.instance_evidence_module = mode
                    for key in ("pred_boxes", "pred_corners", "ref_points"):
                        torch.testing.assert_close(enhanced[key], old[key], rtol=0, atol=0)
                    torch.testing.assert_close(enhanced["pred_logits"][...,0], old["pred_logits"][...,0], rtol=0, atol=0)
                    self.assertGreater((enhanced["pred_logits"]-old["pred_logits"]).abs().sum().item(),0)
                    for group in ("dn_outputs", "aux_outputs", "enc_aux_outputs"):
                        for new, before in zip(enhanced[group], old[group]):
                            for key in before:
                                if torch.is_tensor(before[key]):
                                    torch.testing.assert_close(new[key], before[key], rtol=0, atol=0)
                    for key in old["pre_outputs"]:
                        torch.testing.assert_close(enhanced["pre_outputs"][key], old["pre_outputs"][key], rtol=0, atol=0)
                    losses = cfg.criterion(enhanced, gt)
                    self.assertEqual(set(losses), set(cfg.criterion(old, gt)))
                    self.assertTrue(all(torch.isfinite(x) for x in losses.values()))
                    sum(losses.values()).backward()
                self.assertGreater(branch.p2_projection.conv.weight.grad.abs().sum().item(),0)
                self.assertGreater(branch.p3_projection.conv.weight.grad.abs().sum().item(),0)
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                cfg.optimizer.step(); cfg.ema.update(model)
                clone = config(mode).model
                clone.load_state_dict(model.state_dict(), strict=True)
                clone.load_state_dict(cfg.ema.module.state_dict(), strict=True)
                with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                    empty = model(images, targets(empty=True))
                    self.assertTrue(all(torch.isfinite(v) for v in cfg.criterion(empty, targets(empty=True)).values()))
                del cfg, model, clone, branch, enhanced, old, losses; gc.collect()

    def test_640_eval_deploy_and_settings_rejection(self):
        from calflops import calculate_flops
        for mode in INSTANCE_EVIDENCE_MODULES:
            with self.subTest(mode=mode):
                model = config(mode, 640).model.eval()
                torch.nn.init.normal_(model.decoder.instance_evidence.output.up.weight, std=.02)
                with torch.no_grad():
                    out = model(torch.rand(1,3,640,640))
                    self.assertEqual(out["pred_logits"].shape, (1,300,3))
                    self.assertTrue(all(torch.isfinite(v).all() for v in out.values()))
                    deployed = copy.deepcopy(model).deploy()
                    features = [torch.randn(1,256,s,s) for s in (80,40,20)]
                    p2 = torch.randn(1,64,160,160)
                    original = model.decoder(features, p2_feat=p2)
                    actual = deployed.decoder(features, p2_feat=p2)
                    for key in original:
                        torch.testing.assert_close(actual[key], original[key], rtol=1e-5, atol=1e-5)
                    calculate_flops(copy.deepcopy(deployed),input_shape=(1,3,640,640),print_results=False)
                branch = model.decoder.instance_evidence
                wrong = build_instance_evidence(mode, grid=5)
                with self.assertRaisesRegex(RuntimeError,"settings differ"):
                    wrong.load_state_dict(branch.state_dict(), strict=True)
                del model, deployed, branch; gc.collect()

    def test_invalid_settings_routing_and_disabled_schema(self):
        for kwargs in ({"dim":10},{"grid":4},{"parts":1},{"chunk_size":0},{"scale":0},
                       {"scale":float("nan")},{"use_checkpoint":1}):
            with self.assertRaises(ValueError): build_instance_evidence("PartEvidenceSelection", **kwargs)
        for kwargs in ({"num_classes":6},{"num_layers":2},{"eval_idx":0},
                       {"use_aqs_refine":True,"aqs_apply_layer":3},
                       {"use_rw_local_detail":True}):
            with self.assertRaises(ValueError):
                options = dict(num_classes=3, num_layers=3, instance_evidence_module="PartEvidenceSelection")
                options.update(kwargs); DFINETransformer(**options)
        cfg = config("PartEvidenceSelection")
        cfg.yaml_cfg["DFINE"]["finegrain_trial"] = "PairEvidenceHead"
        with self.assertRaises(ValueError): cfg.model
        cfg = config("PartEvidenceSelection")
        cfg.yaml_cfg["HGNetv2"]["return_idx"] = [1,2,3]
        with torch.no_grad(), self.assertRaisesRegex(ValueError,"return_idx"):
            cfg.model.eval()(torch.rand(2,3,128,128))
        cfg = config("PartEvidenceSelection")
        cfg.yaml_cfg["DFINETransformer"]["instance_evidence_module"] = "none"
        model = cfg.model
        self.assertFalse(hasattr(model.decoder,"instance_evidence"))
        self.assertFalse(any("instance_evidence" in k for k in model.state_dict()))

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable locally; launcher runs this on server")
    def test_cuda_fp16_full_training(self):
        for mode in INSTANCE_EVIDENCE_MODULES:
            cfg = config(mode); model = cfg.model.cuda().train(); criterion = cfg.criterion.cuda()
            torch.nn.init.normal_(model.decoder.instance_evidence.output.up.weight, std=.02)
            with torch.autocast("cuda",dtype=torch.float16):
                out = model(torch.rand(2,3,128,128,device="cuda"), targets("cuda"))
                loss = sum(criterion(out,targets("cuda")).values())
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for name,p in model.named_parameters():
                if p.grad is not None: self.assertTrue(torch.isfinite(p.grad).all(),name)
            del cfg, model, criterion, out, loss; gc.collect(); torch.cuda.empty_cache()

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable locally; no 4090 memory guarantee")
    def test_cuda_batch32_300_queries_branch_workspace(self):
        for mode in INSTANCE_EVIDENCE_MODULES:
            branch = build_instance_evidence(mode).cuda().train()
            torch.nn.init.normal_(branch.output.up.weight,std=.02)
            q = torch.randn(32,300,256,device="cuda",requires_grad=True)
            box = torch.tensor([.5,.5,.3,.4],device="cuda").expand(32,300,4)
            p2 = torch.randn(32,64,160,160,device="cuda",requires_grad=True)
            p3 = torch.randn(32,256,80,80,device="cuda",requires_grad=True)
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast("cuda",dtype=torch.float16): loss = branch(q,box,p2,p3).square().mean()
            loss.backward()
            print(f"{mode} isolated branch CUDA peak allocated: {torch.cuda.max_memory_allocated()/2**20:.1f} MiB",flush=True)
            self.assertTrue(torch.isfinite(loss))
            del branch, q, box, p2, p3, loss; gc.collect(); torch.cuda.empty_cache()

    def test_launcher_syntax_dry_run_and_output_preservation(self):
        script = ROOT/"tools/diagnostics/run_scbs_AQS_instance_evidence_110.sh"
        subprocess.run(["bash","-n",str(script)],check=True)
        subprocess.run(["bash","-n",str(ROOT/"tools/diagnostics/sync_scbs_AQS_instance_evidence.sh")],check=True)
        result = subprocess.run(["bash",str(script),"--dry-run","all","run9"],capture_output=True,text=True,check=True)
        for mode in INSTANCE_EVIDENCE_MODULES: self.assertIn(mode,result.stdout)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"dfine_s_scbs_hrw_epochs110_AQS_PartEvidenceSelection_seed0_run1"
            path.mkdir(); marker = path/"old.txt"; marker.write_text("preserve")
            import os
            env = dict(os.environ, SCBS_EVIDENCE_ROOT=tmp)
            result = subprocess.run(["bash",str(script),"--dry-run","all","run1"],env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(marker.read_text(),"preserve")
            self.assertEqual(len(list(Path(tmp).iterdir())),1)

    def test_parameter_export_model_ema_and_reference_comparison_guards(self):
        from scbs_instance_evidence_experiment import compare, parameters
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode in INSTANCE_EVIDENCE_MODULES:
                branch = small_branch(mode)
                state = {"decoder.instance_evidence."+k:v for k,v in branch.state_dict().items()}
                cp = root/f"{mode}.pth"
                torch.save(dict(model=state,ema=dict(module=state),last_epoch=109),cp)
                report = root/f"{mode}.json"; parameters(cp,report)
                data = json.loads(report.read_text())
                self.assertEqual(data["sources"]["model"]["module"],mode)
                self.assertEqual(data["sources"]["ema"]["residual_scale"],.1)
                with self.assertRaises(FileExistsError): parameters(cp,report)
            reference = root/"reference"
            output = root/"output"
            for analysis in (reference/"analysis",output/"diagnostics_eval/analysis"):
                analysis.mkdir(parents=True)
                row = dict(image_id=1,annotation_id=3,gt_class_id=2,area_original_annotation=100,
                           ignore=False,image_name="one.jpg",box_xyxy=[0.,0.,10.,10.])
                (analysis/"gt_records.jsonl").write_text(json.dumps(row)+"\n")
                summary = dict(checkpoint="/one.pth",checkpoint_sha256="abc",weight_source="ema",
                    GT=1,images=1,queries=300,class_names=["hand-raising","read","write"],
                    confidence_threshold=.5,IoU_threshold=.5,saved_COCO_metrics={"AP":.5},
                    saved_COCO_per_class={},saved_COCO_per_class_and_size={},validator_confusion={},
                    best_query_cross_classification={},confusion_pairs={})
                (analysis/"summary.json").write_text(json.dumps(summary))
            metadata = reference/"evaluation/query_diagnostics/metadata.json"
            metadata.parent.mkdir(parents=True)
            decoder = dict(use_aqs_refine=True,aqs_apply_layer=2,aqs_threshold=.5,
                           aqs_residual_init=.1,aqs_temperature=.1)
            meta = dict(complete=True,checkpoint_sha256="abc",config={"DFINETransformer":decoder})
            metadata.write_text(json.dumps(meta))
            compare(output,reference,"AQS_L2_R1")
            self.assertTrue((output/"diagnostics_eval/compare_AQS_L2_R1.json").is_file())
            decoder["aqs_apply_layer"] = 3; metadata.write_text(json.dumps(meta))
            with self.assertRaises(AssertionError): compare(output,reference,"AQS_L2_R1")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
