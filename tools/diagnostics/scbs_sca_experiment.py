"""Verify SCA's three augmentation changes on unchanged AQS-L2 R1."""
import argparse
import copy
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.core.yaml_utils import load_config

CONFIG=ROOT/"configs/dfine/dfine_s_scbs_hrw_110_AQS_SCA.yml"
BASE=ROOT/"configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml"


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda:f.read(1024*1024),b""): h.update(data)
    return h.hexdigest()


def resolved():
    return load_config(str(CONFIG),cfg={})


def check_config(actual,base):
    expected=copy.deepcopy(base)
    ops=expected["train_dataloader"]["dataset"]["transforms"]["ops"]
    for name in ("RandomZoomOut","RandomIoUCrop"):
        assert sum(op["type"] == name for op in ops) == 1,f"基础增强算子不唯一：{name}"
    zoom=next(op for op in ops if op["type"] == "RandomZoomOut")
    crop=next(op for op in ops if op["type"] == "RandomIoUCrop")
    assert zoom.get("p",.5) == .5 and list(zoom.get("side_range",[1.,4.])) == [1.,4.], "参考AQS已不是原始ZoomOut配方"
    assert crop["p"] == .8, "参考AQS已不是原始IoUCrop配方"
    zoom.update(side_range=[1.,2.],p=.3)
    crop["p"]=.5
    for split in ("train","val"):
        root="/media/a5/5号机移动盘2/wq_datasets/SCB-S"
        expected[f"{split}_dataloader"]["dataset"].update(
            img_folder=f"{root}/images/{split}",ann_file=f"{root}/annotations/instances_{split}.json")
    actual=copy.deepcopy(actual)
    for cfg in (expected,actual): cfg.pop("__include__",None); cfg.pop("output_dir",None)
    assert actual == expected,"三项SCA增强参数/服务器数据地址以外出现额外变化"
    d=actual["DFINETransformer"]
    assert d["use_aqs_refine"] and d["aqs_apply_layer"] == 2 and d["num_layers"] == 3
    assert d.get("aqs_apply_layers") is None and d.get("aqs_refiner_type","legacy") == "legacy"
    assert (d["aqs_threshold"],d["aqs_residual_init"],d["aqs_temperature"]) == (.5,.1,.1)
    assert not any(v for k,v in d.items() if k.startswith("use_") and k != "use_aqs_refine" and isinstance(v,bool))
    assert d.get("instance_evidence_module","none") == "none"
    assert d.get("literature_probe","none") == "none"
    for name in ("HGNetv2","HybridEncoder","DFINECriterion"):
        assert not any(v for k,v in actual[name].items() if k.startswith("use_") and k != "use_lab" and isinstance(v,bool)), name
    assert all(actual["DFINE"].get(k,"none") == "none" for k in ("night_probe","finegrain_trial"))
    assert not actual.get("val_free",False),"不迁入文档的val-free协议"
    assert actual["num_classes"] == 3 and not actual["remap_mscoco_category"]
    assert actual["epochs"] == 110 and actual["use_ema"]
    assert actual["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"] == 100
    assert actual["train_dataloader"]["collate_fn"]["stop_epoch"] == 100
    assert all(actual[f"{s}_dataloader"]["total_batch_size"] == 32 for s in ("train","val"))
    return actual


def preflight(dry=False):
    cfg=check_config(resolved(),load_config(str(BASE),cfg={}))
    print("配置通过：AQS-L2 R1 + SCA；ZoomOut [1,2]/p=.3；IoUCrop p=.5；110/100、batch32不变。",flush=True)
    if dry: return
    import torch
    assert torch.cuda.is_available(),"CUDA不可用，未启动完整训练"
    for split in ("train","val"):
        d=cfg[f"{split}_dataloader"]["dataset"]
        folder,ann=Path(d["img_folder"]),Path(d["ann_file"])
        assert folder.is_dir() and ann.is_file(),f"缺少{split}数据"
        coco=json.loads(ann.read_text(encoding="utf-8"))
        assert sorted((c["id"],c["name"]) for c in coco["categories"]) == [(0,"hand-raising"),(1,"read"),(2,"write")]
        ids={i["id"] for i in coco["images"]}
        assert len(ids) == len(coco["images"]) > 0
        assert all(a["image_id"] in ids and type(a["category_id"]) is int and 0 <= a["category_id"] < 3 for a in coco["annotations"])
        assert all((folder/i["file_name"]).is_file() for i in coco["images"]),f"{split}缺少图片"
        print(f"{split}: {len(ids)} images, annotation SHA256={digest(ann)}",flush=True)


def snapshot(output):
    cfg=resolved()
    files=(ROOT/"tools/diagnostics/scbs_AQS_SCA_sync_files.txt").read_text().splitlines()
    inherited=("configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml",
        "configs/dfine/dfine_s_scbs_hrw_110.yml","configs/dfine/dfine_hgnetv2_s_coco.yml",
        "configs/dfine/include/dataloader.yml","configs/dfine/include/optimizer.yml",
        "configs/dfine/include/dfine_hgnetv2.yml","configs/runtime.yml",
        "src/zoo/dfine/dfine.py","src/zoo/dfine/dfine_decoder.py","src/zoo/dfine/hybrid_encoder.py",
        "src/data/transforms/_transforms.py","src/data/transforms/container.py")
    payload=dict(experiment="AQS_SCA",base="AQS-L2 R1",seed=0,
        note="Training augmentation only; no new neural module, loss, DN300 or val-free. No inference ON/OFF ablation.",
        initialization="same recipe from scratch with original backbone pretraining; no -r/-t; AQS remains trainable",
        resolved_config=cfg,source_sha256={p:digest(ROOT/p) for p in [*files,*inherited] if p},
        annotations_sha256={s:digest(cfg[f"{s}_dataloader"]["dataset"]["ann_file"]) for s in ("train","val")})
    with (output/"experiment_inputs.json").open("x",encoding="utf-8") as f:
        json.dump(payload,f,ensure_ascii=False,indent=2)


def pack(output):
    archive=output/"AQS_SCA_diagnostics.zip"
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted((output/"diagnostics_eval").rglob("*")):
            if p.is_file() and (p.suffix != ".pth" or p.name == "eval.pth"):
                z.write(p,p.relative_to(output))
        for name in ("best_stg2_aqs_parameters.json","aqs_parameters_best_stg2.json","experiment_inputs.json"):
            p=output/name
            if p.is_file(): z.write(p,p.name)
    print("诊断压缩包：",archive,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="operation",required=True)
    p=sub.add_parser("preflight"); p.add_argument("--dry-run",action="store_true")
    for op in ("snapshot","pack"):
        p=sub.add_parser(op); p.add_argument("output",type=Path)
    p=sub.add_parser("compare"); p.add_argument("output",type=Path); p.add_argument("reference",type=Path)
    p.add_argument("label",choices=("AQS_L2_R1","BaselineRun2"))
    args=parser.parse_args()
    if args.operation == "preflight": preflight(args.dry_run)
    elif args.operation == "compare":
        from scbs_instance_evidence_experiment import compare
        compare(args.output,args.reference,args.label)
    else: globals()[args.operation](args.output)


if __name__ == "__main__": main()
