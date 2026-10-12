"""Preflight/snapshot/result helpers for the independent six-module queue."""

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core.yaml_utils import load_config
from src.zoo.dfine.night_probes import NIGHT_PROBES


def resolved(index):
    return load_config(str(ROOT / f"configs/dfine/dfine_s_scbs_hrw_110_night_n{index}.yml"), cfg={})


def sources():
    paths = ["src/zoo/dfine/dfine.py", "src/zoo/dfine/dfine_decoder.py", "src/zoo/dfine/night_probes.py",
             "tools/diagnostics/run_scbs_night_sweep_110.sh", "tools/diagnostics/scbs_night_queue.py"]
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


def preflight(indices, dry_run):
    base = load_config(str(ROOT / "configs/dfine/dfine_s_scbs_hrw_110.yml"), cfg={})
    assert base["epochs"] == 110 and base["num_classes"] == 3 and base["use_ema"]
    assert base["train_dataloader"]["total_batch_size"] == base["val_dataloader"]["total_batch_size"] == 32
    assert base["train_dataloader"]["collate_fn"]["stop_epoch"] == 100
    assert base["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"] == 100
    for owner in ("DFINETransformer", "HybridEncoder", "DFINECriterion"):
        assert all(not value for key, value in base[owner].items() if key.startswith("use_") and isinstance(value, bool)), \
            f"基线启用了额外模块/损失：{owner}"
    assert not base["HGNetv2"].get("use_afta", False)
    assert not base["HGNetv2"].get("use_backbone_p3_local_probe", False)
    for index in indices:
        expected, actual = copy.deepcopy(base), resolved(index)
        expected["DFINE"]["night_probe"] = NIGHT_PROBES[index - 1]
        if index == 5:
            expected["HGNetv2"]["return_idx"] = [0, 1, 2, 3]
        for cfg in (expected, actual):
            cfg.pop("__include__", None)
            cfg.pop("output_dir", None)
        assert actual == expected, f"N{index}：模块以外的配置改变，停止队列"
        print(f"N{index} 配置通过：{NIGHT_PROBES[index - 1]} / 110-100 / batch32 / 原损失 / 无AQS", flush=True)
    if dry_run:
        return
    import torch
    assert torch.cuda.is_available(), "CUDA不可用，不启动完整CPU训练"
    for split in ("train", "val"):
        data = base[f"{split}_dataloader"]["dataset"]
        root, annotation = Path(data["img_folder"]), Path(data["ann_file"])
        assert root.is_dir() and annotation.is_file(), f"缺少{split}数据：{root} / {annotation}"
        coco = json.loads(annotation.read_text(encoding="utf-8"))
        assert sorted((c["id"], c["name"]) for c in coco["categories"]) == [(0, "hand-raising"), (1, "read"), (2, "write")]
        assert coco["images"], f"{split}数据为空"
        ids = {image["id"] for image in coco["images"]}
        assert len(ids) == len(coco["images"]), "重复image_id"
        assert all(a["image_id"] in ids and a["category_id"] in (0, 1, 2) for a in coco["annotations"])
        for image in coco["images"]:
            assert (root / image["file_name"]).is_file(), f"缺少图片：{image['file_name']}"
        print(f"{split}: {len(coco['images'])}图；annotation SHA256={hashlib.sha256(annotation.read_bytes()).hexdigest()}", flush=True)


def snapshot(index, output):
    cfg = resolved(index)
    annotation_hashes = {}
    for split in ("train", "val"):
        path = Path(cfg[f"{split}_dataloader"]["dataset"]["ann_file"])
        annotation_hashes[split] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    payload = dict(group=f"N{index}", mode=NIGHT_PROBES[index - 1], seed=0,
                   initialization="fresh detector + original HGNet pretrained backbone; no -r/-t",
                   resolved_config=cfg, source_sha256=sources(), annotation_sha256=annotation_hashes)
    with (output / "experiment_inputs.json").open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    print(f"已记录N{index}配置、seed和源码校验值", flush=True)


def record(index, output, queue, status, code):
    item = dict(group=f"N{index}", mode=NIGHT_PROBES[index - 1], output=str(output),
                status=status, exit_code=code, recorded_at=datetime.now(timezone.utc).isoformat())
    metrics_path = output / "diagnostics_eval/evaluation/validation_metrics.json"
    if metrics_path.is_file():
        item["validation_metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
    confusion_path = output / "diagnostics_eval/evaluation/confusion_matrix/confusion_matrix.json"
    if confusion_path.is_file():
        cm = json.loads(confusion_path.read_text(encoding="utf-8"))
        item["confusion_summary"] = {key: cm[key] for key in
            ("correct_class_matches", "misclassifications", "unmatched_predictions", "unmatched_GT", "per_class")}
        item["read_to_write"] = cm["matrix_counts"][1][2]
        item["write_to_read"] = cm["matrix_counts"][2][1]
    with (queue / "results.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(item, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="operation", required=True)
    check = sub.add_parser("preflight")
    check.add_argument("indices", nargs="+", type=int, choices=range(1, 7))
    check.add_argument("--dry-run", action="store_true")
    snap = sub.add_parser("snapshot")
    snap.add_argument("index", type=int, choices=range(1, 7)); snap.add_argument("output", type=Path)
    result = sub.add_parser("record")
    result.add_argument("index", type=int, choices=range(1, 7)); result.add_argument("output", type=Path)
    result.add_argument("queue", type=Path); result.add_argument("status"); result.add_argument("code", type=int)
    args = parser.parse_args()
    if args.operation == "preflight":
        preflight(args.indices, args.dry_run)
    elif args.operation == "snapshot":
        snapshot(args.index, args.output)
    else:
        record(args.index, args.output, args.queue, args.status, args.code)


if __name__ == "__main__":
    main()
