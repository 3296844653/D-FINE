"""AQS + SCF provenance, parameters, same-EMA ON/OFF and reference comparisons."""
import argparse
import copy
import json
import math
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core.yaml_utils import load_config
from scbs_instance_evidence_experiment import digest, compare, gt_identity, difference

NAME = "SubpixelContextFusion"
CONFIG = ROOT / f"configs/dfine/dfine_s_scbs_hrw_110_AQS_{NAME}.yml"
BASE = ROOT / "configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml"
SETTINGS = dict(use_subpixel_context_fusion=True, scf_p2_channels=64, scf_dim=32,
    scf_dilation=2, scf_residual_init=.1, scf_checkpoint=True)


def resolved():
    return load_config(str(CONFIG), cfg={})


def preflight(dry=False):
    expected, actual = copy.deepcopy(load_config(str(BASE), cfg={})), resolved()
    expected["HGNetv2"]["return_idx"] = [0,1,2,3]
    expected["HybridEncoder"].update(SETTINGS)
    for split in ("train","val"):
        folder = "/media/a5/5号机移动盘2/wq_datasets/SCB-S"
        expected[f"{split}_dataloader"]["dataset"].update(
            img_folder=f"{folder}/images/{split}", ann_file=f"{folder}/annotations/instances_{split}.json")
    for cfg in (expected,actual):
        cfg.pop("__include__",None); cfg.pop("output_dir",None)
    assert expected == actual, "除了SCF/P2输出路由和服务器数据地址，出现额外配置变化"
    decoder = actual["DFINETransformer"]
    assert decoder["use_aqs_refine"] and decoder["aqs_apply_layer"] == 2
    assert decoder.get("aqs_apply_layers") is None and decoder.get("aqs_refiner_type","legacy") == "legacy"
    assert (decoder["aqs_threshold"],decoder["aqs_residual_init"],decoder["aqs_temperature"]) == (.5,.1,.1)
    assert not any(v for k,v in decoder.items() if k.startswith("use_") and k != "use_aqs_refine" and isinstance(v,bool))
    assert decoder.get("instance_evidence_module","none") == "none"
    for owner in ("DFINECriterion","HGNetv2","HybridEncoder"):
        allowed = {"use_lab","use_subpixel_context_fusion"}
        assert not any(v for k,v in actual[owner].items() if k.startswith("use_") and k not in allowed and isinstance(v,bool)), owner
    assert all(actual["DFINE"].get(k,"none") == "none" for k in ("night_probe","finegrain_trial","literature_probe"))
    assert actual["epochs"] == 110 and actual["num_classes"] == 3 and not actual["remap_mscoco_category"]
    assert actual["use_ema"]
    assert actual["train_dataloader"]["collate_fn"]["stop_epoch"] == 100
    assert actual["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"] == 100
    assert all(actual[f"{s}_dataloader"]["total_batch_size"] == 32 for s in ("train","val"))
    print("配置通过：AQS-L2 R1 + SubpixelContextFusion，110/100，batch32，原损失。",flush=True)
    if dry: return
    import torch
    assert torch.cuda.is_available(), "CUDA不可用，禁止CPU完整训练"
    for split in ("train","val"):
        cfg = actual[f"{split}_dataloader"]["dataset"]
        folder, ann = Path(cfg["img_folder"]),Path(cfg["ann_file"])
        assert folder.is_dir() and ann.is_file(), f"缺少{split}数据"
        coco = json.loads(ann.read_text(encoding="utf-8"))
        assert sorted((c["id"],c["name"]) for c in coco["categories"]) == [(0,"hand-raising"),(1,"read"),(2,"write")]
        ids = {i["id"] for i in coco["images"]}
        assert len(ids) == len(coco["images"]) > 0
        assert all(a["image_id"] in ids and type(a["category_id"]) is int and 0 <= a["category_id"] < 3 for a in coco["annotations"])
        assert all((folder/i["file_name"]).is_file() for i in coco["images"]), f"{split}图片未完整同步"
        print(f"{split}: {len(ids)} images, annotation SHA256={digest(ann)}",flush=True)


def snapshot(output):
    cfg = resolved()
    files = (ROOT/"tools/diagnostics/scbs_AQS_SubpixelContextFusion_sync_files.txt").read_text().splitlines()
    report = dict(module=NAME, base="AQS-L2 R1", seed=0,
        initialization="same recipe from scratch, original backbone pretrained; no -r/-t; AQS trainable",
        resolved_config=cfg, source_sha256={p:digest(ROOT/p) for p in files if p},
        annotation_sha256={s:digest(cfg[f"{s}_dataloader"]["dataset"]["ann_file"]) for s in ("train","val")})
    with (output/"experiment_inputs.json").open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)


def parameters(cp_path, output):
    import torch
    cp = torch.load(cp_path,map_location="cpu",weights_only=True)
    prefix = "encoder.subpixel_context_fusion."
    states = {"model":cp["model"]}
    if "ema" in cp: states["ema"] = cp["ema"]["module"]
    report = dict(checkpoint=str(cp_path.resolve()),checkpoint_sha256=digest(cp_path),
        checkpoint_last_epoch=cp.get("last_epoch"),sources={},
        warning="Checkpoint epoch may be stale in stage2. Parameter norms/biases are not observed attention or contribution.")
    for name,state in states.items():
        settings = state[prefix+"settings"].tolist()
        raw = state[prefix+"raw_scale"].float().item()
        assert math.isfinite(raw)
        report["sources"][name] = dict(settings=settings,dim=settings[3],dilation=settings[4],
            residual_raw=raw,residual_scale=.5*math.tanh(raw),
            phase_null_bias=state[prefix+"phase_null_bias"].float().item(),
            context_null_bias=state[prefix+"context_null_bias"].float().item(),
            phase_bias=state[prefix+"phase_bias"].float().tolist(),
            context_bias=state[prefix+"context_bias"].float().tolist(),
            parameter_norms={k.removeprefix(prefix):float(v.float().norm()) for k,v in state.items()
                if k.startswith(prefix) and k != prefix+"settings"})
        print(f"{name} SCF residual scale={report['sources'][name]['residual_scale']:.8f}",flush=True)
    with output.open("x",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)


def on_off(output):
    paths = [output/"diagnostics_eval/analysis",output/"SubpixelContextFusion_off_eval/analysis"]
    on,off = [json.loads((p/"summary.json").read_text()) for p in paths]
    for key in ("checkpoint_sha256","weight_source","GT","images","queries","class_names","confidence_threshold","IoU_threshold"):
        assert on[key] == off[key], f"ON/OFF口径不同：{key}"
    assert on["weight_source"] == "ema"
    assert gt_identity(paths[0]) == gt_identity(paths[1])
    checkpoint=output/"best_stg2.pth"
    if checkpoint.is_file():
        assert digest(checkpoint) == on["checkpoint_sha256"], "评估后权重文件发生变化"
    parameter_path=output/"best_stg2_SubpixelContextFusion_parameters.json"
    if parameter_path.is_file():
        assert json.loads(parameter_path.read_text())["checkpoint_sha256"] == on["checkpoint_sha256"], "参数导出不是同一权重"
    for folder, enabled in zip(paths,(True,False)):
        meta = json.loads((folder.parent/"evaluation/query_diagnostics/metadata.json").read_text())
        assert meta["complete"] and meta["checkpoint_sha256"] == on["checkpoint_sha256"]
        cfg = meta["config"]
        assert cfg["diagnostic_SubpixelContextFusion_output_enabled"] is enabled
        assert cfg["HybridEncoder"]["use_subpixel_context_fusion"]
        assert cfg["DFINETransformer"]["use_aqs_refine"] and cfg["DFINETransformer"]["aqs_apply_layer"] == 2
    keys = ("saved_COCO_metrics","saved_COCO_per_class","saved_COCO_per_class_and_size",
        "validator_confusion","best_query_cross_classification","confusion_pairs")
    report = dict(checkpoint_sha256=on["checkpoint_sha256"], on={k:on.get(k) for k in keys},
        off={k:off.get(k) for k in keys},
        on_minus_off={k:difference(off[k],on[k]) if k in on and k in off else None for k in keys},
        AP_delta_unit="fraction; x100 = percentage points",
        warning="Same jointly trained EMA; AQS stays ON. OFF is not AQS-alone training. Encoder changes can affect TopK, final classes AND boxes.")
    best=None
    log=output/"log.txt"
    if log.is_file():
        for line in log.read_text().splitlines():
            try: row=json.loads(line)
            except ValueError: continue
            metrics=row.get("test_coco_eval_bbox")
            if isinstance(metrics,list) and metrics and isinstance(metrics[0],(int,float)) and math.isfinite(metrics[0]):
                if best is None or metrics[0] > best["AP"]:
                    best=dict(epoch=row.get("epoch"),AP=metrics[0])
    report["training_log_best"]=best
    report["ON_matches_training_best_AP"]=None if best is None else abs(on["saved_COCO_metrics"]["AP"]-best["AP"]) <= 1e-5
    if best is not None and not report["ON_matches_training_best_AP"]:
        print("注意：最佳权重独立ON评估与日志最优AP不一致；保留结果，请核对权重/数据/口径。",flush=True)
    with (output/"SubpixelContextFusion_on_off.json").open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(f"同一EMA ON-OFF ΔAP={100*(on['saved_COCO_metrics']['AP']-off['saved_COCO_metrics']['AP']):+.4f} pp",flush=True)


def pack(output):
    archive = output/"SubpixelContextFusion_diagnostics.zip"
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED) as z:
        for folder in ("diagnostics_eval","SubpixelContextFusion_off_eval"):
            for p in sorted((output/folder).rglob("*")):
                if p.is_file() and (p.suffix != ".pth" or p.name == "eval.pth"):
                    z.write(p,p.relative_to(output))
        for name in ("SubpixelContextFusion_on_off.json","best_stg2_SubpixelContextFusion_parameters.json",
                     "best_stg2_aqs_parameters.json","aqs_parameters_best_stg2.json","experiment_inputs.json"):
            p = output/name
            if p.is_file(): z.write(p,p.name)
    print("诊断压缩包：",archive,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="operation",required=True)
    p=sub.add_parser("preflight"); p.add_argument("--dry-run",action="store_true")
    for op in ("snapshot","on-off","pack"):
        p=sub.add_parser(op); p.add_argument("output",type=Path)
    p=sub.add_parser("parameters"); p.add_argument("checkpoint",type=Path); p.add_argument("output",type=Path)
    p=sub.add_parser("compare"); p.add_argument("output",type=Path); p.add_argument("reference",type=Path)
    p.add_argument("label",choices=("AQS_L2_R1","BaselineRun2"))
    args=parser.parse_args()
    if args.operation == "preflight": preflight(args.dry_run)
    elif args.operation == "parameters": parameters(args.checkpoint,args.output)
    elif args.operation == "compare": compare(args.output,args.reference,args.label)
    else: globals()[args.operation](args.output)


if __name__ == "__main__": main()
