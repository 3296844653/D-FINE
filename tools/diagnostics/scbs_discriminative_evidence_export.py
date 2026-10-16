"""Read-only checkpoint preflight, provenance audit and diagnostic packaging.

Torch/project imports are lazy: reporting/packaging tests need only Python's
standard library. Checkpoint/config/model/annotations are never modified.
"""
import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/"tools/diagnostics/scbs_AQS_DiscriminativeEvidenceL2_diagnostics_sync_files.txt"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def training_log(path):
    rows = []
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row,dict) and isinstance(row.get("epoch"),int):
                    metrics = row.get("test_coco_eval_bbox")
                    if (isinstance(metrics,list) and metrics and isinstance(metrics[0],(float,int))
                            and math.isfinite(metrics[0])):
                        rows.append(row)
    counts = Counter(row["epoch"] for row in rows)
    best = max(rows,key=lambda row:row["test_coco_eval_bbox"][0]) if rows else None
    return dict(path=str(path), records=len(rows), unique_epochs=len(counts),
        duplicate_epochs={str(k):v for k,v in counts.items() if v>1},
        max_epoch=max(counts) if counts else None,
        best=None if best is None else dict(epoch=best["epoch"],
            AP=best["test_coco_eval_bbox"][0], coco_metrics=best["test_coco_eval_bbox"],
            per_class=best.get("test_bbox_per_class")))


def prepare(checkpoint, bundle):
    import torch
    import sys
    sys.path.insert(0,str(ROOT))
    from scbs_discriminative_evidence_experiment import preflight, resolved
    preflight(dry=True)  # Config check only; do NOT require training images.
    cfg = resolved()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，未执行完整验证")
    data = cfg["val_dataloader"]["dataset"]
    folder, annotation = Path(data["img_folder"]),Path(data["ann_file"])
    if not folder.is_dir() or not annotation.is_file():
        raise FileNotFoundError("缺少原验证图片或标注")
    coco = json.loads(annotation.read_text(encoding="utf-8"))
    if sorted((c["id"],c["name"]) for c in coco["categories"]) != [(0,"hand-raising"),(1,"read"),(2,"write")]:
        raise ValueError("验证类别与SCB-S三类不一致")
    ids = {i["id"] for i in coco["images"]}
    if not ids or len(ids) != len(coco["images"]):
        raise ValueError("空验证集或重复image_id")
    if not all(a["image_id"] in ids and type(a["category_id"]) is int and 0 <= a["category_id"] < 3
               for a in coco["annotations"]):
        raise ValueError("验证标注类别或image_id错误")
    if not all((folder/i["file_name"]).is_file() for i in coco["images"]):
        raise FileNotFoundError("验证图片不完整")
    checkpoint = checkpoint.resolve()
    before = digest(checkpoint)
    cp = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(cp.get("ema"),dict) or "module" not in cp["ema"]:
        raise ValueError("缺少训练时的EMA；禁止退回model权重混用")
    decoder = cfg["DFINETransformer"]
    expected = [1, decoder["hidden_dim"], decoder["discriminative_evidence_p2_channels"],
        decoder["discriminative_evidence_dim"],decoder["discriminative_evidence_grid"],
        round(decoder["discriminative_evidence_scale"]*1e9),2]
    prefix = "decoder.decoder.discriminative_evidence_l2."
    for name,state in (("model",cp["model"]),("ema",cp["ema"]["module"])):
        if state[prefix+"settings"].tolist() != expected:
            raise ValueError(f"{name}: 证据模块配置与训练权重不一致")
        if state["decoder.dec_score_head.2.weight"].shape[0] != 3:
            raise ValueError("不是SCB-S三类权重")
        if "decoder.decoder.query_cls_refiner.threshold_logit" not in state:
            raise ValueError("缺少原单层AQS权重")
        if any("aqs_refiners.layer" in key for key in state):
            raise ValueError("不是单层AQS-L2 R1实验")
    if digest(checkpoint) != before:
        raise RuntimeError("检查过程中checkpoint变化；未导出")
    # New bundle only, so partial/old exports can never be overwritten.
    bundle.mkdir()
    (bundle/"logs").mkdir()
    files = [p for p in MANIFEST.read_text().splitlines() if p]
    report = dict(operation="evaluation_only_on_off", checkpoint=str(checkpoint),
        checkpoint_sha256=before, checkpoint_last_epoch=cp.get("last_epoch"),
        checkpoint_epoch_warning="last_epoch may lag the actual evaluated epoch after stage-2 reloads",
        weight_source="ema", AQS="unchanged legacy L2 R1; ON in both passes",
        config=cfg, validation_annotation=str(annotation), validation_annotation_sha256=digest(annotation),
        validation_images=len(ids), training_log=training_log(checkpoint.parent/"log.txt"),
        source_sha256={p:digest(ROOT/p) for p in files})
    with (bundle/"export_inputs.json").open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(f"只验证，检查通过：{len(ids)} images；checkpoint last_epoch={cp.get('last_epoch')}（非可靠最佳轮次）",flush=True)
    print(f"新导出目录：{bundle}",flush=True)


def audit(bundle):
    snapshot = json.loads((bundle/"export_inputs.json").read_text())
    if digest(snapshot["checkpoint"]) != snapshot["checkpoint_sha256"]:
        raise RuntimeError("导出期间训练权重变化，拒绝把两次评估认作同一权重")
    if digest(snapshot["validation_annotation"]) != snapshot["validation_annotation_sha256"]:
        raise RuntimeError("导出期间验证标注变化")
    on = bundle/"diagnostics_eval"
    off = bundle/"DiscriminativeEvidenceL2_off_eval"
    for folder,enabled in ((on,True),(off,False)):
        meta = json.loads((folder/"evaluation/query_diagnostics/metadata.json").read_text())
        summary = json.loads((folder/"analysis/summary.json").read_text())
        if not meta["complete"] or not meta["config"]["diagnostic_DiscriminativeEvidenceL2_output_enabled"] is enabled:
            raise ValueError("ON/OFF模式记录不完整或不一致")
        if meta["checkpoint_sha256"] != snapshot["checkpoint_sha256"] or summary["checkpoint_sha256"] != snapshot["checkpoint_sha256"]:
            raise ValueError("导出权重哈希与预检不一致")
        if summary["weight_source"] != "ema":
            raise ValueError("不是相同EMA权重")
    summary = json.loads((on/"analysis/summary.json").read_text())
    actual = summary["saved_COCO_metrics"]["AP"]
    best = snapshot["training_log"]["best"]
    delta = None if best is None else actual-best["AP"]
    # A mismatch is diagnostic evidence, not permission to fabricate an AP or
    # discard the already exported ON/OFF results. Record it explicitly.
    match = None if delta is None else abs(delta) <= 1e-5
    report = dict(weight_integrity_verified=True, actual_ON_AP=actual,
        expected_training_best=best, AP_difference_from_training_best=delta,
        matches_training_best_AP=match, tolerance_fraction=1e-5,
        warning="Missing log/mismatched AP requires investigation; OFF is not an AQS-alone model.")
    with (bundle/"checkpoint_evaluation_audit.json").open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(f"独立加载最佳EMA：ON AP={actual*100:.4f}；复现训练最佳AP：{match}",flush=True)
    if match is False:
        print("警告：未复现训练最佳AP；保留全部诊断，先检查权重/配置/数据/数值口径。",flush=True)


def package(bundle):
    target = bundle/(bundle.name+".zip")
    files = sorted(p for p in bundle.rglob("*") if p.is_file() and p != target)
    if any(p.suffix == ".pth" and p.name != "eval.pth" for p in files):
        raise ValueError("诊断压缩包不得包含训练checkpoint")
    with zipfile.ZipFile(target,"x",zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, path.relative_to(bundle))
    print(f"可下载的诊断压缩包：{target}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation",required=True)
    p = sub.add_parser("prepare"); p.add_argument("checkpoint",type=Path); p.add_argument("bundle",type=Path)
    p = sub.add_parser("audit"); p.add_argument("bundle",type=Path)
    p = sub.add_parser("package"); p.add_argument("bundle",type=Path)
    args = parser.parse_args()
    if args.operation == "prepare": prepare(args.checkpoint,args.bundle)
    elif args.operation == "audit": audit(args.bundle)
    else: package(args.bundle)


if __name__ == "__main__":
    main()
