"""Configuration provenance, trained parameters and matched diagnostic comparisons."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core.yaml_utils import load_config
from src.zoo.dfine.instance_evidence import INSTANCE_EVIDENCE_MODULES

BASE = ROOT / "configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def resolved(mode):
    return load_config(str(ROOT / f"configs/dfine/dfine_s_scbs_hrw_110_AQS_{mode}.yml"), cfg={})


def preflight(modes, dry=False):
    base = load_config(str(BASE), cfg={})
    decoder = base["DFINETransformer"]
    assert decoder["use_aqs_refine"] and decoder["aqs_apply_layer"] == 2
    assert decoder.get("aqs_apply_layers") is None and decoder.get("aqs_refiner_type", "legacy") == "legacy"
    assert (decoder["aqs_threshold"], decoder["aqs_residual_init"], decoder["aqs_temperature"]) == (.5,.1,.1)
    assert base["epochs"] == 110 and base["num_classes"] == 3 and not base["remap_mscoco_category"]
    assert base["train_dataloader"]["total_batch_size"] == base["val_dataloader"]["total_batch_size"] == 32
    assert base["train_dataloader"]["collate_fn"]["stop_epoch"] == 100
    assert base["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"] == 100
    assert base["use_ema"]
    for owner in ("HybridEncoder", "DFINECriterion", "HGNetv2"):
        assert not any(v for k,v in base[owner].items() if k.startswith("use_") and k != "use_lab" and isinstance(v,bool)), owner
    assert not any(v for k,v in decoder.items() if k.startswith("use_") and k != "use_aqs_refine" and isinstance(v,bool))
    assert all(base["DFINE"].get(k,"none") == "none" for k in ("night_probe","finegrain_trial","literature_probe"))
    for mode in modes:
        expected, actual = copy.deepcopy(base), resolved(mode)
        expected["HGNetv2"]["return_idx"] = [0,1,2,3]
        expected["DFINETransformer"].update(instance_evidence_module=mode,
            instance_evidence_dim=64, instance_evidence_grid=7, instance_evidence_parts=3,
            instance_evidence_p2_channels=64, instance_evidence_scale=.1,
            instance_evidence_chunk=32, instance_evidence_checkpoint=True)
        for cfg in (expected,actual): cfg.pop("__include__",None); cfg.pop("output_dir",None)
        assert actual == expected, f"{mode}: AQS/训练/原损失之外出现额外配置变化"
        print(f"配置通过：AQS-L2 R1 + {mode}, 110/100, batch32, seed0", flush=True)
    if dry: return
    import torch
    assert torch.cuda.is_available(), "CUDA不可用，禁止CPU完整训练"
    for split in ("train","val"):
        data = base[f"{split}_dataloader"]["dataset"]
        folder, annotation = Path(data["img_folder"]), Path(data["ann_file"])
        assert folder.is_dir() and annotation.is_file(), f"缺少{split}数据"
        coco = json.loads(annotation.read_text(encoding="utf-8"))
        assert sorted((c["id"],c["name"]) for c in coco["categories"]) == [(0,"hand-raising"),(1,"read"),(2,"write")]
        ids = {i["id"] for i in coco["images"]}
        assert len(ids) == len(coco["images"]) > 0
        assert all(a["image_id"] in ids and type(a["category_id"]) is int and 0 <= a["category_id"] < 3 for a in coco["annotations"])
        assert all((folder/i["file_name"]).is_file() for i in coco["images"]), f"{split}图片未完整同步"
        print(f"{split}: {len(ids)} images, annotation SHA256={digest(annotation)}", flush=True)


def snapshot(mode, output):
    cfg = resolved(mode)
    manifest = ROOT / "tools/diagnostics/scbs_AQS_instance_evidence_sync_files.txt"
    files = [p for p in manifest.read_text().splitlines() if p.strip()]
    payload = dict(module=mode, base="AQS-L2 R1", seed=0,
        initialization="from scratch; original pretrained HGNet; no -r or -t; AQS weights remain trainable",
        resolved_config=cfg, source_sha256={p:digest(ROOT/p) for p in files},
        annotations_sha256={s:digest(cfg[f"{s}_dataloader"]["dataset"]["ann_file"]) for s in ("train","val")})
    with (output / "experiment_inputs.json").open("x",encoding="utf-8") as f:
        json.dump(payload,f,ensure_ascii=False,indent=2)


def parameters(checkpoint_path, output):
    import torch
    cp = torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    report = dict(checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=digest(checkpoint_path),
                  epoch=cp.get("last_epoch"), sources={})
    states = {"model":cp["model"]}
    if "ema" in cp and "module" in cp["ema"]: states["ema"] = cp["ema"]["module"]
    prefix = "decoder.instance_evidence."
    for source,state in states.items():
        settings = state[prefix + "_evidence_settings"].tolist()
        name = INSTANCE_EVIDENCE_MODULES[settings[0]]
        entry = dict(module=name, hidden_dim=settings[1], p2_channels=settings[2], dim=settings[3],
                     grid=settings[4], parts=settings[5], residual_scale=settings[6]/1e9,
                     parameter_norms={k.removeprefix(prefix):float(v.float().norm()) for k,v in state.items()
                                      if k.startswith(prefix) and not k.endswith("_evidence_settings")})
        if name == "PartEvidenceSelection":
            entry["region_selector_bias"] = state[prefix+"region_selection.bias"].tolist()
            entry["note"] = "Selector bias is NOT observed region relevance; requires feature-dependent visualization."
        if name == "LocalInteractionEvidence":
            entry["effective_role_centers"] = state[prefix+"role_centers"].float().tanh().tolist()
            entry["note"] = "Latent spatial slots, not supervised anatomical parts or detected objects."
        report["sources"][source] = entry
        print(f"{source} {name}: fixed scale={entry['residual_scale']}, head norm={entry['parameter_norms']['output.up.weight']:.6f}",flush=True)
    with output.open("x",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)


def gt_identity(analysis):
    keys = ("image_id","annotation_id","gt_class_id","area_original_annotation","ignore","image_name")
    result = {}
    with (analysis/"gt_records.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            identity = (row["image_id"],row["annotation_id"])
            assert identity not in result, "重复GT记录"
            result[identity] = tuple(row[k] for k in keys) + tuple(round(v,3) for v in row["box_xyxy"])
    return result


def difference(old,new):
    if isinstance(old,dict) and isinstance(new,dict):
        return {k:difference(old[k],new[k]) for k in old.keys() & new.keys()}
    if isinstance(old,(int,float)) and isinstance(new,(int,float)):
        return None if old < 0 or new < 0 else new-old
    return None


def check_reference(reference, label):
    old = json.loads((reference/"analysis/summary.json").read_text())
    assert old["weight_source"] == "ema", "要求同为最佳EMA权重"
    metadata = json.loads((reference/"evaluation/query_diagnostics/metadata.json").read_text())
    assert metadata["complete"] and metadata["checkpoint_sha256"] == old["checkpoint_sha256"]
    decoder = metadata["config"]["DFINETransformer"]
    assert decoder.get("instance_evidence_module","none") == "none", "参考不应带新增模块"
    if label == "AQS_L2_R1":
        assert decoder.get("use_aqs_refine",False) and decoder.get("aqs_apply_layer") == 2
        assert decoder.get("aqs_apply_layers") is None
        assert (decoder["aqs_threshold"],decoder["aqs_residual_init"],decoder["aqs_temperature"]) == (.5,.1,.1)
    else:
        assert not decoder.get("use_aqs_refine",False), "BaselineRun2不应包含AQS"
        assert old["checkpoint_sha256"] == "4ff8f014c4497e2d918de11f3fcc6d741bcc19315a9283c8b7786162154ee271", "不是此前核验的Baseline Run2"
    return old


def compare(output, reference, label):
    analysis = output/"diagnostics_eval/analysis"
    old = check_reference(reference,label)
    new = json.loads((analysis/"summary.json").read_text())
    for key in ("weight_source","GT","images","queries","class_names","confidence_threshold","IoU_threshold"):
        assert old[key] == new[key], f"诊断口径不同：{key}"
    assert gt_identity(reference/"analysis") == gt_identity(analysis), "GT图像/类别/面积/框不一致，拒绝比较"
    keys = ("saved_COCO_metrics","saved_COCO_per_class","saved_COCO_per_class_and_size",
            "validator_confusion","best_query_cross_classification")
    report = dict(reference=label, reference_checkpoint=old["checkpoint"],
        reference_checkpoint_sha256=old["checkpoint_sha256"], module_checkpoint=new["checkpoint"],
        module_checkpoint_sha256=new["checkpoint_sha256"], AP_delta_unit="fraction; x100 = percentage points",
        reference_metrics={k:old[k] for k in keys}, module_metrics={k:new[k] for k in keys},
        numeric_differences={k:difference(old[k],new[k]) for k in keys},
        reference_confusion_pairs=old["confusion_pairs"], module_confusion_pairs=new["confusion_pairs"],
        warning="Single-run diagnostic response, not stable gains or mechanism/novelty proof.")
    target = output/f"diagnostics_eval/compare_{label}.json"
    with target.open("x",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)
    print(f"{label}: ΔAP={100*(new['saved_COCO_metrics']['AP']-old['saved_COCO_metrics']['AP']):+.4f} pp; {target}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation",required=True)
    p = sub.add_parser("preflight"); p.add_argument("modes",nargs="+",choices=INSTANCE_EVIDENCE_MODULES)
    p.add_argument("--dry-run",action="store_true")
    p = sub.add_parser("snapshot"); p.add_argument("mode",choices=INSTANCE_EVIDENCE_MODULES); p.add_argument("output",type=Path)
    p = sub.add_parser("parameters"); p.add_argument("checkpoint",type=Path); p.add_argument("output",type=Path)
    p = sub.add_parser("compare"); p.add_argument("output",type=Path); p.add_argument("reference",type=Path)
    p.add_argument("label",choices=("AQS_L2_R1","BaselineRun2"))
    p = sub.add_parser("reference"); p.add_argument("reference",type=Path)
    p.add_argument("label",choices=("AQS_L2_R1","BaselineRun2"))
    args = parser.parse_args()
    if args.operation == "preflight": preflight(args.modes,args.dry_run)
    elif args.operation == "snapshot": snapshot(args.mode,args.output)
    elif args.operation == "parameters": parameters(args.checkpoint,args.output)
    elif args.operation == "compare": compare(args.output,args.reference,args.label)
    else:
        check_reference(args.reference,args.label)
        print(f"参考身份通过：{args.label}",flush=True)


if __name__ == "__main__": main()
