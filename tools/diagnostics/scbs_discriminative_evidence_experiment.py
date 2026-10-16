"""Validate the AQS-based experiment and export checkpoint provenance/norms."""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core.yaml_utils import load_config
from scbs_instance_evidence_experiment import digest, compare, gt_identity, difference

NAME = "DiscriminativeEvidenceL2"
CONFIG = ROOT / f"configs/dfine/dfine_s_scbs_hrw_110_AQS_{NAME}.yml"
BASE = ROOT / "configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml"


def resolved():
    return load_config(str(CONFIG), cfg={})


def preflight(dry=False):
    base, actual = load_config(str(BASE), cfg={}), resolved()
    expected = copy.deepcopy(base)
    expected["HGNetv2"]["return_idx"] = [0,1,2,3]
    expected["DFINETransformer"].update(use_discriminative_evidence_l2=True,
        discriminative_evidence_dim=64, discriminative_evidence_grid=7,
        discriminative_evidence_p2_channels=64, discriminative_evidence_scale=.1,
        discriminative_evidence_chunk=32, discriminative_evidence_checkpoint=True)
    for split in ("train", "val"):
        dataset = f"/media/a5/5号机移动盘2/wq_datasets/SCB-S"
        expected[f"{split}_dataloader"]["dataset"].update(
            img_folder=f"{dataset}/images/{split}",
            ann_file=f"{dataset}/annotations/instances_{split}.json")
    for cfg in (expected, actual):
        cfg.pop("__include__",None); cfg.pop("output_dir",None)
    assert actual == expected, "存在新模块和数据绝对地址以外的配置变化"
    decoder = actual["DFINETransformer"]
    assert decoder["use_aqs_refine"] and decoder["aqs_apply_layer"] == 2
    assert decoder.get("aqs_apply_layers") is None and decoder.get("aqs_refiner_type","legacy") == "legacy"
    assert (decoder["aqs_threshold"],decoder["aqs_residual_init"],decoder["aqs_temperature"]) == (.5,.1,.1)
    assert actual["epochs"] == 110 and actual["num_classes"] == 3 and not actual["remap_mscoco_category"]
    assert actual["use_ema"]
    assert actual["train_dataloader"]["collate_fn"]["stop_epoch"] == 100
    assert actual["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"] == 100
    assert not any(v for k,v in decoder.items() if k.startswith("use_") and
                   k not in ("use_aqs_refine","use_discriminative_evidence_l2") and isinstance(v,bool))
    assert decoder.get("instance_evidence_module","none") == "none"
    assert all(actual["DFINE"].get(k,"none") == "none" for k in ("night_probe","finegrain_trial","literature_probe"))
    for owner in ("HGNetv2","HybridEncoder","DFINECriterion"):
        assert not any(v for k,v in actual[owner].items() if k.startswith("use_") and
                       k != "use_lab" and isinstance(v,bool)), owner
    for split in ("train","val"):
        assert actual[f"{split}_dataloader"]["total_batch_size"] == 32
    print("配置通过：AQS-L2 R1 + DiscriminativeEvidenceL2；110/100，batch32，原损失。",flush=True)
    if dry: return
    import torch
    assert torch.cuda.is_available(), "CUDA不可用"
    for split in ("train","val"):
        data = actual[f"{split}_dataloader"]["dataset"]
        folder, annotation = Path(data["img_folder"]),Path(data["ann_file"])
        assert folder.is_dir() and annotation.is_file(), f"缺少{split}数据"
        coco = json.loads(annotation.read_text(encoding="utf-8"))
        assert sorted((c["id"],c["name"]) for c in coco["categories"]) == [(0,"hand-raising"),(1,"read"),(2,"write")]
        ids = {i["id"] for i in coco["images"]}
        assert len(ids) == len(coco["images"]) > 0
        assert all(a["image_id"] in ids and type(a["category_id"]) is int and 0 <= a["category_id"] < 3 for a in coco["annotations"])
        assert all((folder/i["file_name"]).is_file() for i in coco["images"]), f"{split}缺少图片"
        print(f"{split}: {len(ids)} images, SHA256={digest(annotation)}",flush=True)


def snapshot(output):
    cfg = resolved()
    files = (ROOT/"tools/diagnostics/scbs_AQS_DiscriminativeEvidenceL2_sync_files.txt").read_text().splitlines()
    report = dict(module=NAME, base="AQS-L2 R1", seed=0,
        initialization="from scratch; AQS is trainable, not frozen; no -r/-t",
        resolved_config=cfg, source_sha256={p:digest(ROOT/p) for p in files if p},
        annotations_sha256={s:digest(cfg[f"{s}_dataloader"]["dataset"]["ann_file"]) for s in ("train","val")})
    with (output/"experiment_inputs.json").open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)


def parameters(checkpoint, output):
    import torch
    cp = torch.load(checkpoint, map_location="cpu",weights_only=True)
    prefix = "decoder.decoder.discriminative_evidence_l2."
    states = {"model":cp["model"]}
    if "ema" in cp: states["ema"] = cp["ema"]["module"]
    report = dict(checkpoint=str(checkpoint.resolve()),checkpoint_sha256=digest(checkpoint),
        checkpoint_last_epoch=cp.get("last_epoch"),sources={},
        note="Norms are not observed attention/residual contributions; last_epoch may be stale in best_stg2.")
    for name,state in states.items():
        settings = state[prefix+"settings"].tolist()
        hypothesis = state[prefix+"hypotheses"].float()
        report["sources"][name] = dict(settings=settings,
            null_bias=state[prefix+"null_bias"].float().tolist(),
            hypothesis_distance=float((hypothesis[0]-hypothesis[1]).norm()),
            norms={k.removeprefix(prefix):float(v.float().norm()) for k,v in state.items()
                   if k.startswith(prefix) and k != prefix+"settings"})
        print(f"{name}: {NAME}, output projection norm={report['sources'][name]['norms']['output.up.weight']:.6f}",flush=True)
    with output.open("x",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)


def on_off(output):
    on = output/"diagnostics_eval/analysis"
    off = output/"DiscriminativeEvidenceL2_off_eval/analysis"
    new, old = (json.loads((p/"summary.json").read_text()) for p in (on,off))
    for key in ("checkpoint_sha256","weight_source","GT","images","queries","class_names",
                "confidence_threshold","IoU_threshold"):
        assert old[key] == new[key], f"ON/OFF口径不一致：{key}"
    assert new["weight_source"] == "ema", "要求同一最佳EMA"
    assert gt_identity(on) == gt_identity(off)
    for folder, enabled in ((on.parent,True),(off.parent,False)):
        meta = json.loads((folder/"evaluation/query_diagnostics/metadata.json").read_text())
        assert meta["complete"] and meta["checkpoint_sha256"] == new["checkpoint_sha256"]
        cfg = meta["config"]
        assert cfg["diagnostic_DiscriminativeEvidenceL2_output_enabled"] is enabled, "缺少真实ON/OFF元数据标记"
        decoder = cfg["DFINETransformer"]
        assert decoder["use_discriminative_evidence_l2"] and decoder["use_aqs_refine"]
        assert decoder["aqs_apply_layer"] == 2
    keys = ("saved_COCO_metrics","saved_COCO_per_class","saved_COCO_per_class_and_size",
            "validator_confusion","best_query_cross_classification","confusion_pairs")
    report = dict(checkpoint_sha256=new["checkpoint_sha256"], on={k:new.get(k) for k in keys},
        off={k:old.get(k) for k in keys},
        on_minus_off={k:difference(old[k],new[k]) if k in old and k in new else None for k in keys},
        AP_delta_unit="fraction; x100 = percentage points",
        warning="Same jointly trained EMA, AQS remains ON. OFF is not AQS-alone training. "
                "Removing the L2 residual can change BOTH final classes and boxes via L3.")
    with (output/"DiscriminativeEvidenceL2_on_off.json").open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(f"同一权重 ON-OFF ΔAP={100*(new['saved_COCO_metrics']['AP']-old['saved_COCO_metrics']['AP']):+.4f} pp",flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="operation",required=True)
    q = sub.add_parser("preflight"); q.add_argument("--dry-run",action="store_true")
    q = sub.add_parser("snapshot"); q.add_argument("output",type=Path)
    q = sub.add_parser("parameters"); q.add_argument("checkpoint",type=Path); q.add_argument("output",type=Path)
    q = sub.add_parser("on-off"); q.add_argument("output",type=Path)
    q = sub.add_parser("compare"); q.add_argument("output",type=Path); q.add_argument("reference",type=Path)
    q.add_argument("label",choices=("AQS_L2_R1","BaselineRun2"))
    args = p.parse_args()
    if args.operation == "preflight": preflight(args.dry_run)
    elif args.operation == "snapshot": snapshot(args.output)
    elif args.operation == "parameters": parameters(args.checkpoint,args.output)
    elif args.operation == "on-off": on_off(args.output)
    else: compare(args.output,args.reference,args.label)


if __name__ == "__main__": main()
