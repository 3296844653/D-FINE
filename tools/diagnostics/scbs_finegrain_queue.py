"""Config/data provenance and Run2 comparison for five structural candidates."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.core.yaml_utils import load_config
from src.zoo.dfine.finegrain_trials import FINEGRAIN_TRIALS

RUN2_SHA="4ff8f014c4497e2d918de11f3fcc6d741bcc19315a9283c8b7786162154ee271"
VAL_SHA="623bb8478049dd69d32a92183b0a322bdb8f79f72e6f6f52e491d48923bd5d37"


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def resolved(mode=None):
    suffix="" if mode is None else "_"+mode
    return load_config(str(ROOT/f"configs/dfine/dfine_s_scbs_hrw_110{suffix}.yml"),cfg={})


def settings(mode):
    values=dict(finegrain_trial=mode,finegrain_dim=64,finegrain_residual_scale=.10)
    if mode=="WindowInteractionP3": values["finegrain_window"]=5
    if mode in ("BoxScaleContrastL2","PartInteractionL2","PairEvidenceHead"):
        values["finegrain_grid_size"]=3 if mode=="BoxScaleContrastL2" else 5
    if mode=="PartInteractionL2": values["finegrain_parts"]=4
    return values


def preflight(modes,dry,checkpoint):
    base=resolved()
    assert base["epochs"]==110 and base["num_classes"]==3 and base["use_ema"]
    assert base["train_dataloader"]["total_batch_size"]==base["val_dataloader"]["total_batch_size"]==32
    assert base["train_dataloader"]["collate_fn"]["stop_epoch"]==100
    assert base["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"]==100
    assert base["DFINE"].get("night_probe","none")==base["DFINE"].get("finegrain_trial","none")=="none"
    assert base["DFINE"].get("literature_probe","none")=="none"
    for owner in ("DFINETransformer","HybridEncoder","DFINECriterion"):
        assert all(not v for k,v in base[owner].items() if k.startswith("use_") and isinstance(v,bool)),owner
    assert not base["HGNetv2"].get("use_afta",False)
    assert not base["HGNetv2"].get("use_backbone_p3_local_probe",False)
    for mode in modes:
        expected,actual=copy.deepcopy(base),resolved(mode)
        expected["DFINE"].update(settings(mode))
        for item in (expected,actual): item.pop("__include__",None); item.pop("output_dir",None)
        assert expected==actual,f"{mode}：模块之外的配置改变，停止"
        print(f"通过：{mode} / 110-100 / batch32 / seed0 / 原损失 / 无AQS",flush=True)
    if dry: return
    import torch
    assert torch.cuda.is_available(),"CUDA不可用，未启动CPU完整训练"
    for split in ("train","val"):
        data=base[f"{split}_dataloader"]["dataset"]
        folder,annotation=Path(data["img_folder"]),Path(data["ann_file"])
        assert folder.is_dir() and annotation.is_file(),f"缺少{split}数据"
        coco=json.loads(annotation.read_text())
        assert sorted((c["id"],c["name"]) for c in coco["categories"])==[(0,"hand-raising"),(1,"read"),(2,"write")]
        ids={i["id"] for i in coco["images"]}
        assert len(ids)==len(coco["images"])>0
        assert all(a["image_id"] in ids and a["category_id"] in (0,1,2) for a in coco["annotations"])
        assert all((folder/i["file_name"]).is_file() for i in coco["images"]),f"{split}存在未同步图片"
        if split=="val": assert digest(annotation)==VAL_SHA,"验证标注已变化，不能直接和现有Run2对照"
        print(f"{split}: {len(ids)}图；标注SHA256={digest(annotation)}",flush=True)
    if checkpoint is not None:
        assert checkpoint.is_file() and digest(checkpoint)==RUN2_SHA,"Run2权重不存在或不是已核验的Run2"


def snapshot(mode,output):
    cfg=resolved(mode)
    files=("src/zoo/dfine/dfine.py","src/zoo/dfine/dfine_decoder.py","src/zoo/dfine/finegrain_trials.py",
           "tools/diagnostics/scbs_finegrain_queue.py","tools/diagnostics/run_scbs_finegrain_five_110.sh")
    payload=dict(module=mode,seed=0,initialization="fresh detector + original pretrained HGNet; no -r/-t",
        resolved_config=cfg,source_sha256={p:digest(ROOT/p) for p in files},reference="Baseline110 Run2",
        annotation_sha256={s:digest(cfg[f"{s}_dataloader"]["dataset"]["ann_file"]) for s in ("train","val")})
    with (output/"experiment_inputs.json").open("x") as f: json.dump(payload,f,ensure_ascii=False,indent=2)


def differences(old,new):
    # Preserve nested metrics including missing/COCO=-1 entries; never convert
    # unavailable AP to a numerical gain. AP deltas are percentage points.
    if isinstance(old,dict) and isinstance(new,dict):
        return {key:differences(old[key],new[key]) for key in old.keys()&new.keys()}
    if isinstance(old,(int,float)) and isinstance(new,(int,float)):
        return None if old<0 or new<0 else new-old
    return None


def compare(output,reference):
    a=json.loads((reference/"analysis/summary.json").read_text())
    b=json.loads((output/"diagnostics_eval/analysis/summary.json").read_text())
    assert a["checkpoint_sha256"]==RUN2_SHA
    for key in ("weight_source","GT","images","class_names","confidence_threshold","IoU_threshold"):
        assert a[key]==b[key],f"对照口径不一致：{key}"
    assert a["weight_source"]=="ema"
    keys=("saved_COCO_metrics","saved_COCO_per_class","saved_COCO_per_class_and_size",
          "validator_confusion","best_query_cross_classification")
    # Full originals retained; downstream analysis can inspect per-GT crossings.
    report={"reference":"Baseline110 Run2","baseline_checkpoint_sha256":RUN2_SHA,
        "module_checkpoint_sha256":b["checkpoint_sha256"],"AP_delta_unit":"fraction (multiply by 100 for pp)",
        "baseline":{k:a[k] for k in keys},"module":{k:b[k] for k in keys},
        "numeric_differences":{k:differences(a[k],b[k]) for k in keys},
        "baseline_confusion_pairs":a["confusion_pairs"],"module_confusion_pairs":b["confusion_pairs"],
        "warning":"Single-run response, not repeatability or causal bottleneck proof; confusion counts are not COCO AP."}
    target=output/"diagnostics_eval/compare_BaselineRun2.json"
    with target.open("x") as f: json.dump(report,f,ensure_ascii=False,indent=2)
    delta=100*(b["saved_COCO_metrics"]["AP"]-a["saved_COCO_metrics"]["AP"])
    print(f"已对照Run2：ΔAP={delta:+.4f}个百分点；{target}",flush=True)


def record(mode,output,queue):
    item=dict(module=mode,output=str(output),status="completed")
    path=output/"diagnostics_eval/compare_BaselineRun2.json"
    if path.is_file(): item["comparison"]=json.loads(path.read_text())
    log=output/"log.txt"
    if log.is_file():
        rows=[json.loads(line) for line in log.read_text().splitlines() if line.strip().startswith("{")]
        candidates=[r for r in rows if "test_coco_eval_bbox" in r]
        if candidates: item["best_logged_epoch"]=max(candidates,key=lambda r:r["test_coco_eval_bbox"][0])
    with (queue/"results.jsonl").open("a") as f: f.write(json.dumps(item,ensure_ascii=False)+"\n")


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="operation",required=True)
    c=sub.add_parser("preflight"); c.add_argument("modes",nargs="+",choices=FINEGRAIN_TRIALS)
    c.add_argument("--dry-run",action="store_true"); c.add_argument("--run2-checkpoint",type=Path)
    for operation in ("snapshot","record"):
        q=sub.add_parser(operation); q.add_argument("mode",choices=FINEGRAIN_TRIALS); q.add_argument("output",type=Path)
        if operation=="record": q.add_argument("queue",type=Path)
    c=sub.add_parser("compare"); c.add_argument("output",type=Path); c.add_argument("reference",type=Path)
    a=p.parse_args()
    if a.operation=="preflight": preflight(a.modes,a.dry_run,a.run2_checkpoint)
    elif a.operation=="snapshot": snapshot(a.mode,a.output)
    elif a.operation=="compare": compare(a.output,a.reference)
    else: record(a.mode,a.output,a.queue)


if __name__=="__main__": main()
