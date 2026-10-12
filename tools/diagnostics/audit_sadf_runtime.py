"""Frozen SADF runtime audit. No training, checkpoint edits, or score replacement.

Uses the existing validation transforms/postprocessor/COCO evaluator. GT only
indexes diagnostic regions AFTER the detector forward. Inference interventions
on an already-trained model are NOT retrained architectural ablations.
"""
import argparse
import copy
import hashlib
import html
import json
import math
import sys
import types
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torchvision.ops import box_iou

from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.solver.confusion_export import export_confusion_matrix
from src.solver.det_engine import _bbox_metrics_per_class, _mean_valid_coco_values
from src.solver.validator import Validator, scale_boxes

VARIANTS = ("full", "no_alignment", "no_detail", "equal_scales", "bypass")
RUN2_SHA = "4ff8f014c4497e2d918de11f3fcc6d741bcc19315a9283c8b7786162154ee271"
ANNOTATION_SHA = "623bb8478049dd69d32a92183b0a322bdb8f79f72e6f6f52e491d48923bd5d37"
NOTES = [
    "Frozen inference only. Interventions are not retrained ablations or causal proof.",
    "bypass uses the SADF-trained detector with its P3 correction removed; it is NOT Baseline Run2.",
    "Coverage means existence of a selected Top300 query-class pair with correct class, score>=0.5 and IoU>=0.5. Not one-to-one TP/recall.",
    "Confusion uses the existing class-blind highest-IoU greedy Validator at score/IoU 0.5. COCO AP uses all scores and its own matching.",
    "GT regions use original-image coordinates mapped to the actual P3 grid; area groups use original COCO annotation area, not resized area or feature stride.",
    "Regions include intersecting P3 cells (floor/ceil, clipped). Heatmaps are P3-grid quantities, not anatomical attention or a proof of correct alignment.",
    "COCO area bounds are inclusive: exact 32^2/96^2 boundary objects belong to both adjacent groups.",
    "Channel gate means alone do not measure feature contributions. Gated feature RMS is also reported.",
    "Statistics weight each GT equally, not by its pixel area; ignored GT are retained in records but excluded from grouped runtime statistics.",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def load_configuration(path, batch_size=None, workers=None):
    raw = load_config(str(path), cfg={})
    with patch("src.core.yaml_config.load_config", lambda _: copy.deepcopy(raw)):
        cfg = YAMLConfig(str(path))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    if batch_size is not None:
        cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = batch_size
    if workers is not None:
        cfg.yaml_cfg["val_dataloader"]["num_workers"] = workers
    ops = cfg.yaml_cfg["val_dataloader"]["dataset"]["transforms"]["ops"]
    if [op["type"] for op in ops] != ["Resize", "ConvertPILImage"]:
        raise ValueError("Runtime ROI audit requires unchanged Resize + ConvertPILImage validation transforms")
    if cfg.yaml_cfg["val_dataloader"].get("shuffle", False):
        raise ValueError("Validation must not shuffle")
    return cfg


def load_frozen(cfg, checkpoint_path, source):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get(source)
    if source == "ema":
        state = state.get("module") if isinstance(state, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"{checkpoint_path}: missing requested {source} weights; no fallback")
    model = cfg.model
    model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()
           if value.is_floating_point()):
        raise ValueError("Nonfinite checkpoint parameters")
    model.eval().requires_grad_(False)
    return model, {"checkpoint": str(Path(checkpoint_path).resolve()),
                   "checkpoint_sha256": sha256(checkpoint_path),
                   "epoch": checkpoint.get("last_epoch"), "weight_source": source}


def scalar_maps(before, after, aux, offset_limit):
    """Reduce channels on the inference device; do not cache full feature tensors."""
    maps = {}
    before, after = before.float(), after.float()
    maps["input_energy"] = before.square().mean(1)
    maps["update_energy"] = (after - before).square().mean(1)
    maps["detail_energy"] = aux["detail"].float().square().mean(1)
    maps["context_energy"] = aux["context_delta"].float().square().mean(1)
    for name in ("detail", "context"):
        value = aux["detail" if name == "detail" else "context_delta"].float()
        gate = aux[name + "_gate"].float()
        maps[name + "_gate_mean"] = gate.mean(1)
        maps["gated_" + name + "_energy"] = (value * gate).square().mean(1)
    for i, name in enumerate(("p3", "p4", "p5")):
        maps["weight_" + name] = aux["scale_weights"][:, i].float()
    for i, size in enumerate((3, 5)):
        maps[f"detail_weight_{size}"] = aux["detail_weights"][:, i].float()
    for level in ("p4", "p5"):
        offset = aux["offset_" + level].float()
        maps[f"offset_{level}_x"] = offset[:, 0]
        maps[f"offset_{level}_y"] = offset[:, 1]
        maps[f"offset_{level}_norm"] = offset.square().sum(1).sqrt()
        maps[f"offset_{level}_near_bound_fraction"] = (offset.abs() >= .95 * offset_limit).float().mean(1)
    if any(not torch.isfinite(v).all() for v in maps.values()):
        raise ValueError("Nonfinite SADF runtime maps")
    # One device-to-host transfer per batch, independent of number of GT.
    stacked = torch.stack(list(maps.values()), 1).detach().cpu()
    return dict(zip(maps, stacked.unbind(1)))


@contextmanager
def sadf_intervention(module, variant, capture=None):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown SADF intervention: {variant}")
    if module.training or torch.is_grad_enabled():
        raise RuntimeError("SADF interventions require eval mode and no_grad/inference_mode")
    names = ("enable_alignment", "enable_detail", "enable_scale_selection")
    old_flags = [getattr(module, name) for name in names]
    local_forward = module.__dict__.get("forward")
    had_local_forward = "forward" in module.__dict__
    original = module.forward
    if variant == "no_alignment":
        module.enable_alignment = False
    elif variant == "no_detail":
        module.enable_detail = False
    elif variant == "equal_scales":
        module.enable_scale_selection = False

    def forward(this, source_p3, p3, p4, p5):
        if variant == "bypass":
            return p3
        if capture is None:
            return original(source_p3, p3, p4, p5)
        output, aux = original(source_p3, p3, p4, p5, return_aux=True)
        capture.clear()
        capture.update(scalar_maps(p3, output, aux, this.alignments[0].offset_limit))
        return output

    module.forward = types.MethodType(forward, module)
    try:
        yield
    finally:
        for name, value in zip(names, old_flags):
            setattr(module, name, value)
        if had_local_forward:
            module.forward = local_forward
        else:
            del module.forward
        if capture is not None:
            capture.clear()


def size_groups(area):
    return [name for name, lo, hi in (("small", 0, 32**2),
             ("medium", 32**2, 96**2), ("large", 96**2, float("inf"))) if lo <= area <= hi]


def roi_bounds(box, original_wh, map_hw):
    width, height = original_wh
    mh, mw = map_hw
    x1, y1, x2, y2 = box
    bounds = (max(0, min(mw, math.floor(x1 * mw / width))),
              max(0, min(mh, math.floor(y1 * mh / height))),
              max(0, min(mw, math.ceil(x2 * mw / width))),
              max(0, min(mh, math.ceil(y2 * mh / height))))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("Empty GT region on P3 grid")
    return bounds


def roi_statistics(maps, batch_index, box, original_wh):
    x1, y1, x2, y2 = roi_bounds(box, original_wh, next(iter(maps.values())).shape[-2:])
    means = {name: float(value[batch_index, y1:y2, x1:x2].double().mean())
             for name, value in maps.items()}
    result = {name: value for name, value in means.items() if not name.endswith("_energy")}
    for name in ("input", "update", "detail", "context", "gated_detail", "gated_context"):
        result[name + "_rms"] = math.sqrt(max(0., means[name + "_energy"]))
    result["relative_update"] = result["update_rms"] / (result["input_rms"] + 1e-12)
    result["roi_p3_cells"] = (x2 - x1) * (y2 - y1)
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("Nonfinite GT runtime statistic")
    return result


def gt_annotations(coco, image_id, target, original_boxes):
    """Recover original IDs/areas in EXACT dataset order and validate correspondence."""
    info = coco.imgs[image_id]
    width, height = map(int, target["orig_size"].tolist())
    if (width, height) != (info["width"], info["height"]):
        raise ValueError("Image dimensions differ from COCO metadata")
    items = []
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=[image_id])):
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        box = [min(width, max(0, x)), min(height, max(0, y)),
               min(width, max(0, x+w)), min(height, max(0, y+h))]
        if box[2] > box[0] and box[3] > box[1]:
            area = float(ann.get("area", w*h))
            if not math.isfinite(area) or area < 0:
                raise ValueError("Invalid original annotation area")
            items.append(dict(ann, original_box=box, original_area=area))
    expected = torch.tensor([a["original_box"] for a in items], dtype=torch.float32).reshape(-1, 4)
    labels = torch.tensor([a["category_id"] for a in items], dtype=torch.long)
    if (len(items) != len(original_boxes) or not torch.equal(labels, target["labels"].cpu())
            or not torch.allclose(expected, original_boxes, atol=1e-3, rtol=1e-5)):
        raise ValueError("GT IDs/boxes do not align with unchanged validation target order")
    return items


def candidate_record(prediction, box, category, confidence=.5, iou_threshold=.5):
    ious = box_iou(prediction["boxes"], torch.tensor([box], dtype=torch.float32)).flatten()
    overlap = ious >= iou_threshold
    correct = overlap & (prediction["labels"] == category)
    covered = bool((correct & (prediction["scores"] >= confidence)).any())
    def entry(index):
        return {"class_id": int(prediction["labels"][index]),
                "score": float(prediction["scores"][index]), "iou": float(ious[index]),
                "box_xyxy": prediction["boxes"][index].tolist()}
    closest = entry(int(ious.argmax())) if len(ious) else None
    best_correct = None
    if correct.any():
        ids = correct.nonzero().flatten()
        best_correct = entry(int(ids[prediction["scores"][ids].argmax()]))
    return {"correct_high_candidate": covered,
            "max_selected_iou": float(ious.max()) if len(ious) else 0.,
            "best_overlap_prediction": closest, "best_correct_overlap_prediction": best_correct}


def summarize_rows(rows, paired=False):
    groups = defaultdict(list)
    for row in rows:
        if row["ignored"]:
            continue
        for size in ["all", *row["size_groups"]]:
            label = row["class_name"] + "/" + size
            if paired:
                label += "/" + row["coverage_transition"]
            groups[label].append(row)
    results = {}
    for key, values in sorted(groups.items()):
        result = {"GT_count": len(values), "correct_high_candidate_count":
                  sum(v["prediction"]["correct_high_candidate"] for v in values), "runtime": {}}
        for name in values[0].get("runtime", {}):
            data = np.asarray([v["runtime"][name] for v in values], dtype=float)
            result["runtime"][name] = {"mean": float(data.mean()), "median": float(np.median(data)),
                                       "p10": float(np.quantile(data, .1)), "p90": float(np.quantile(data, .9))}
        results[key] = result
    return results


def coco_size_metrics(coco_eval):
    results = {}
    precision, recall = coco_eval.eval["precision"], coco_eval.eval["recall"]
    for index, category in enumerate(coco_eval.params.catIds):
        name = coco_eval.cocoGt.cats[category]["name"]
        results[name] = {}
        for area_index, area in enumerate(coco_eval.params.areaRngLbl):
            results[name][area] = {
                "AP": _mean_valid_coco_values(precision[:, :, index, area_index, -1]),
                "AR100": _mean_valid_coco_values(recall[:, index, area_index, -1])}
    return results


@torch.inference_mode()
def evaluate_variant(model, cfg, loader, output, metadata, variant, max_images=None):
    output.mkdir(exist_ok=False)
    evaluator = cfg.evaluator
    evaluator.cleanup()
    coco = evaluator.coco_eval["bbox"].cocoGt
    capture, rows, all_gt, all_predictions, seen = {}, [], [], [], []
    postprocessor = cfg.postprocessor.eval()
    if postprocessor.remap_mscoco_category or not postprocessor.use_focal_loss:
        raise ValueError("Audit requires original focal TopK postprocessing without category remapping")
    context = (nullcontext() if variant == "baseline_run2" else
               sadf_intervention(model.encoder.sadf, variant, capture if variant == "full" else None))
    device = next(model.parameters()).device
    with context:
        for batch_index, (samples, targets) in enumerate(loader):
            if max_images is not None:
                remaining = max_images - len(seen)
                if remaining <= 0:
                    break
                samples, targets = samples[:remaining], targets[:remaining]
            samples = samples.to(device)
            original_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
            # FP32 matches det_engine.evaluate (which does not enable AMP even
            # when --use-amp was passed to train.py --test-only).
            outputs = model(samples)
            predictions = postprocessor(outputs, original_sizes)
            evaluator.update({int(t["image_id"].item()): p for t, p in zip(targets, predictions)})
            for b, (target, prediction) in enumerate(zip(targets, predictions)):
                image_id = int(target["image_id"].item())
                if image_id in seen:
                    raise ValueError("Duplicate validation image")
                seen.append(image_id)
                cpu_prediction = {k: v.detach().float().cpu() if k != "labels" else v.detach().long().cpu()
                                  for k, v in prediction.items()}
                original_wh = tuple(int(v) for v in target["orig_size"].tolist())
                boxes = scale_boxes(target["boxes"].float().cpu().clone(),
                                    (original_wh[1], original_wh[0]), tuple(samples.shape[-2:]))
                all_gt.append({"boxes": boxes, "labels": target["labels"].cpu()})
                all_predictions.append(cpu_prediction)
                annotations = gt_annotations(coco, image_id, target, boxes)
                for ann, box in zip(annotations, boxes.tolist()):
                    row = {"image_id": image_id, "annotation_id": ann["id"],
                           "file_name": coco.imgs[image_id]["file_name"],
                           "original_wh": list(original_wh), "box_xyxy": box,
                           "category_id": ann["category_id"], "class_name": coco.cats[ann["category_id"]]["name"],
                           "original_area": ann["original_area"], "size_groups": size_groups(ann["original_area"]),
                           "ignored": bool(ann.get("ignore", False)),
                           "prediction": candidate_record(cpu_prediction, box, ann["category_id"])}
                    if variant == "full":
                        if not capture:
                            raise ValueError("SADF capture did not execute")
                        row["runtime"] = roi_statistics(capture, b, box, original_wh)
                    rows.append(row)
                # Retain reduced maps only for images with medium write GT.
                # Render at most 12 paired examples after Run2 matching.
                if variant == "full":
                    candidates = [a for a in annotations if a["category_id"] == 2
                                  and "medium" in size_groups(a["original_area"]) and not a.get("ignore", 0)]
                    if candidates:
                        snapshot = {k: v[b].clone() for k, v in capture.items()}
                        torch.save({"image_id": image_id, "maps": snapshot,
                                    "offset_limit": model.encoder.sadf.alignments[0].offset_limit},
                                   output / f"maps_{image_id}.pth")
            print(f"{variant}: {len(seen)}/{min(len(loader.dataset), max_images or len(loader.dataset))} images", flush=True)
    if not seen:
        raise ValueError("Empty validation run")
    for evaluation in evaluator.coco_eval.values():
        evaluation.params.imgIds = sorted(seen)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    validator = Validator(all_gt, all_predictions, conf_thresh=.5, iou_thresh=.5)
    diagnostic = validator.compute_metrics(extended=True)
    cm = export_confusion_matrix(validator, coco, output, metadata=dict(metadata, variant=variant))
    result = {"variant": variant, "images": len(seen), "GT_records": len(rows),
              "partial_validation": len(seen) != len(loader.dataset), "metadata": metadata,
              "coco_eval_bbox": evaluator.coco_eval["bbox"].stats.tolist(),
              "bbox_per_class": _bbox_metrics_per_class(evaluator.coco_eval["bbox"], diagnostic["extended_metrics"]),
              "coco_per_class_and_size": coco_size_metrics(evaluator.coco_eval["bbox"]),
              "confusion_matrix": cm, "candidate_coverage": summarize_rows(rows), "notes": NOTES}
    write_json(output / "metrics.json", result)
    with (output / "gt_records.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return result, rows


def pair_rows(baseline, sadf):
    def index(rows):
        out = {(r["image_id"], r["annotation_id"]): r for r in rows}
        if len(out) != len(rows):
            raise ValueError("Duplicate GT IDs")
        return out
    base, current = index(baseline), index(sadf)
    if base.keys() != current.keys():
        raise ValueError("Run2/SADF GT identity sets differ")
    result = []
    for key, row in current.items():
        old = base[key]
        for name in ("category_id", "box_xyxy", "original_area", "size_groups", "file_name", "ignored"):
            if row[name] != old[name]:
                raise ValueError(f"Run2/SADF GT mismatch: {name}")
        before, after = old["prediction"]["correct_high_candidate"], row["prediction"]["correct_high_candidate"]
        transition = ("both_covered" if before and after else "gained" if after else
                      "lost" if before else "both_uncovered")
        result.append(dict(row, baseline_prediction=old["prediction"], coverage_transition=transition))
    return result


def render_examples(output, rows, image_dir):
    """Readable figures linked from a self-contained local HTML index."""
    from PIL import Image
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Rectangle
    cards = []
    class_names = {0: "hand-raising", 1: "read", 2: "write"}
    def prediction_text(prediction):
        item = prediction.get("best_overlap_prediction") if prediction else None
        if item is None:
            return "no selected prediction"
        return (f"{class_names[item['class_id']]} / score={item['score']:.3f} / "
                f"IoU={item['iou']:.3f}")
    eligible = [r for r in rows if r["category_id"] == 2 and "medium" in r["size_groups"] and not r["ignored"]]
    chosen = []
    for transition in ("lost", "gained", "both_uncovered", "both_covered"):
        candidates = [r for r in eligible if r["coverage_transition"] == transition]
        candidates.sort(key=lambda r: (-r["runtime"]["relative_update"], r["image_id"], r["annotation_id"]))
        chosen.extend(candidates[:3])
    for row in chosen:
        snapshot = output / "full" / f"maps_{row['image_id']}.pth"
        data = torch.load(snapshot, map_location="cpu", weights_only=True)
        with Image.open(image_dir / row["file_name"]) as source:
            rgb = np.asarray(source.convert("RGB"))
        maps = data["maps"]
        values = [("P3 scale weight", maps["weight_p3"], 0, 1),
                  ("P4 scale weight", maps["weight_p4"], 0, 1),
                  ("P5 scale weight", maps["weight_p5"], 0, 1),
                  ("Gated detail RMS", maps["gated_detail_energy"].sqrt(), 0, None),
                  ("Gated context RMS", maps["gated_context_energy"].sqrt(), 0, None),
                  ("P4 displacement (P3 pixels)", maps["offset_p4_norm"], 0, data["offset_limit"]*math.sqrt(2)),
                  ("P5 displacement (P3 pixels)", maps["offset_p5_norm"], 0, data["offset_limit"]*math.sqrt(2)),
                  ("P3 update RMS", maps["update_energy"].sqrt(), 0, None)]
        fig = Figure(figsize=(15, 12)); FigureCanvasAgg(fig)
        axes = fig.subplots(3, 3).flatten()
        axes[0].imshow(rgb); axes[0].set_title("Green: GT / Orange: Run2 / Blue: SADF")
        gt = row["box_xyxy"]; width, height = row["original_wh"]
        for box, color in [(gt, "lime"),
                ((row.get("baseline_prediction", {}).get("best_overlap_prediction") or {}).get("box_xyxy"), "orange"),
                ((row["prediction"].get("best_overlap_prediction") or {}).get("box_xyxy"), "cyan")]:
            if box is not None:
                axes[0].add_patch(Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                           fill=False, edgecolor=color, linewidth=2))
        for ax, (name, value, lo, hi) in zip(axes[1:], values):
            artist = ax.imshow(value.numpy(), extent=(0, width, height, 0),
                               cmap="viridis", vmin=lo, vmax=hi)
            ax.add_patch(Rectangle((gt[0], gt[1]), gt[2]-gt[0], gt[3]-gt[1],
                                  fill=False, edgecolor="red", linewidth=1.5))
            if name.startswith("P4 displacement") or name.startswith("P5 displacement"):
                level = "p4" if name.startswith("P4") else "p5"
                mh, mw = value.shape
                step = max(1, min(mh, mw)//12)
                ys, xs = np.mgrid[step//2:mh:step, step//2:mw:step]
                ax.quiver((xs+.5)*width/mw, (ys+.5)*height/mh,
                          maps[f"offset_{level}_x"].numpy()[ys, xs]*width/mw,
                          maps[f"offset_{level}_y"].numpy()[ys, xs]*height/mh,
                          angles="xy", scale_units="xy", scale=1, color="white", width=.003)
            ax.set_title(name); fig.colorbar(artist, ax=ax, fraction=.035)
        for ax in axes:
            ax.set_xlim(0, width); ax.set_ylim(height, 0)
        title = f"image={row['image_id']} GT={row['annotation_id']} write/medium | {row.get('coverage_transition', '')}"
        fig.suptitle(title); fig.tight_layout()
        filename = f"SADF_maps_{row['image_id']}_{row['annotation_id']}.png"
        fig.savefig(output / filename, dpi=110)
        description = ("GT: write; Run2 best-overlap pair: " + prediction_text(row.get("baseline_prediction"))
                       + "; SADF best-overlap pair: " + prediction_text(row["prediction"]))
        cards.append(f"<h2>{html.escape(title)}</h2><p>{html.escape(row['file_name'])}</p>"
                     f"<p>{html.escape(description)}</p>"
                     f"<img style='width:100%;max-width:1500px' src='{filename}'>")
    page = ("<!doctype html><meta charset='utf-8'><title>SADF runtime audit</title>"
            "<h1>SADF 中尺度 write 内部诊断</h1><p>固定 EMA 权重；GT 仅用于统计。"
            "热图不是人体部位注意力或对齐正确性的证明；gained/lost 是候选覆盖变化，不是 COCO TP。"
            "图片里的预测框是Top300中与此GT最重叠的候选，可能低于0.5，"
            "也可能不是覆盖该GT的正确候选；类别/分数/IoU列在每张图上方。</p><p>"
            "每个候选覆盖变化组最多选3个残差相对变化较大的中尺度write样例，存在选择偏差；"
            "结论请看全部 GT 汇总。</p>" + "\n".join(cards))
    (output / "SADF_runtime_examples.html").write_text(page, encoding="utf-8")


def report_markdown(reports, paired=None):
    lines = ["# SADF 冻结权重内部诊断", "", "没有训练。分支关闭是训练后干预，不是重新训练的消融。", "",
             "| 版本 | AP | APm | read AP | write AP | medium write AP | write→read | read→write |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in reports.items():
        metric = result["coco_eval_bbox"]
        classes = result["coco_per_class_and_size"]
        cm = result["confusion_matrix"]
        ids = {label: i for i, label in enumerate(cm["row_labels"])}
        def ap(category, area="all"):
            value = classes[category][area]["AP"]
            return f"{value*100:.2f}" if value >= 0 else "NA"
        def overall(index):
            return f"{metric[index]*100:.2f}" if metric[index] >= 0 else "NA"
        lines.append(f"| {name} | {overall(0)} | {overall(4)} | {ap('read')} | "
                     f"{ap('write')} | {ap('write','medium')} | "
                     f"{cm['matrix_counts'][ids['write']][ids['read']]} | "
                     f"{cm['matrix_counts'][ids['read']][ids['write']]} |")
    lines += ["", "bypass ≠ 原始基线：其他参数仍是 SADF 训练后的参数。",
              "COCO AP 不使用固定 0.5 分数过滤；混淆计数使用 score≥0.5、IoU≥0.5。",
              "如 partial_validation=true，则是子集冒烟测试，不能与完整验证集已发表数值比较。"]
    if "full" in reports:
        lines += ["", "## 实际运行时统计（各GT等权均值）", "",
                  "| 类别/尺度 | GT数 | 正确高分候选覆盖数 | 残差相对变化% | P3权重% | P4权重% | P5权重% | P4偏移长度 | P5偏移长度 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        groups = reports["full"]["candidate_coverage"]
        for key in ("read/medium", "write/medium", "read/large", "write/large"):
            if key not in groups:
                continue
            group = groups[key]; runtime = group["runtime"]
            numbers = [runtime[k]["mean"] * scale for k, scale in (
                ("relative_update",100), ("weight_p3",100), ("weight_p4",100),
                ("weight_p5",100), ("offset_p4_norm",1), ("offset_p5_norm",1))]
            lines.append(f"| {key} | {group['GT_count']} | {group['correct_high_candidate_count']} | "
                         + " | ".join(f"{v:.3f}" for v in numbers) + " |")
        lines += ["", "偏移长度单位为P3像素，不是原图像素；残差相对变化不是alpha。权重大不自动代表更有效。",
                  "", "| 类别/尺度 | detail门控均值 | context门控均值 | 门控后detail RMS | 门控后context RMS |",
                  "|---|---:|---:|---:|---:|"]
        for key in ("read/medium", "write/medium", "read/large", "write/large"):
            if key in groups:
                values = groups[key]["runtime"]
                lines.append(f"| {key} | " + " | ".join(f"{values[k]['mean']:.4f}" for k in
                    ("detail_gate_mean","context_gate_mean","gated_detail_rms","gated_context_rms")) + " |")
    if paired is not None:
        lines += ["", "## 中尺度write：与Run2同一GT的候选覆盖变化", "",
                  "| 变化组 | GT数 | 残差相对变化% | P5权重% | detail门控均值 |",
                  "|---|---:|---:|---:|---:|"]
        groups = summarize_rows(paired, paired=True)
        for label in ("lost", "gained", "both_uncovered", "both_covered"):
            group = groups.get("write/medium/" + label)
            if group:
                runtime = group["runtime"]
                lines.append(f"| {label} | {group['GT_count']} | {runtime['relative_update']['mean']*100:.3f} | "
                             f"{runtime['weight_p5']['mean']*100:.3f} | {runtime['detail_gate_mean']['mean']:.4f} |")
        lines += ["", "lost/gained是正确高分候选覆盖的变化，不等于最终一对一检测错误或COCO TP。"]
    lines += ["", "## 全部统计与配对结果", "",
              "- SADF_runtime_summary.json：每类×原始尺度的实际残差/融合/门控/偏移统计。",
              "- SADF_vs_Run2_paired_groups.json：按同一 GT 的候选覆盖 gained/lost 分组统计。",
              "- SADF_vs_Run2_gt_records.jsonl：GT ID、框、预测、运行时统计逐项对照。",
              "- SADF_runtime_examples.html：少量中尺度 write 图示（仅辅助观察，不替代全量统计）。",
              "", "## 解释限制", "", *["- " + n for n in NOTES]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw"
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dfine/dfine_s_scbs_hrw_110_SADF.yml")
    parser.add_argument("--checkpoint", type=Path, default=Path(root) / "dfine_s_scbs_hrw_epochs110_SADF_seed0_run1/best_stg2.pth")
    parser.add_argument("--baseline-config", type=Path, default=ROOT / "configs/dfine/dfine_s_scbs_hrw_110.yml")
    parser.add_argument("--baseline-checkpoint", type=Path, default=Path(root) / "dfine_s_scbs_hrw_epochs110_seed0_run2/best_stg2.pth")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", choices=("ema", "model"), default="ema")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--expected-run2-sha256", default=RUN2_SHA)
    parser.add_argument("--expected-annotation-sha256", default=ANNOTATION_SHA)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if (args.batch_size < 1 or args.workers < 0 or (args.max_images is not None and args.max_images < 1)
            or len(set(args.variants)) != len(args.variants) or "full" not in args.variants):
        parser.error("Positive batch/max-images, nonnegative workers, and unique variants including full are required")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(f"Preserving existing output: {args.output_dir}")
    cfg = load_configuration(args.config, args.batch_size, args.workers)
    base_cfg = load_configuration(args.baseline_config, args.batch_size, args.workers)
    if cfg.yaml_cfg["val_dataloader"] != base_cfg.yaml_cfg["val_dataloader"]:
        raise ValueError("SADF/Run2 validation data/transforms/collate differ")
    for key in ("num_classes", "remap_mscoco_category", "use_focal_loss", "num_top_queries"):
        if cfg.yaml_cfg.get(key) != base_cfg.yaml_cfg.get(key):
            raise ValueError(f"SADF/Run2 postprocessing differs: {key}")
    data = cfg.yaml_cfg["val_dataloader"]["dataset"]
    annotation_sha = sha256(data["ann_file"])
    if args.expected_annotation_sha256 and annotation_sha != args.expected_annotation_sha256:
        raise ValueError("Annotation fingerprint is not the supplied Run2 validation dataset")
    annotation = json.loads(Path(data["ann_file"]).read_text(encoding="utf-8"))
    if sorted((c["id"], c["name"]) for c in annotation["categories"]) != [(0,"hand-raising"),(1,"read"),(2,"write")]:
        raise ValueError("Expected SCB-S official categories 0=hand-raising,1=read,2=write")
    if not annotation["images"]:
        raise ValueError("Empty annotation images")
    for key, items in (("image", annotation["images"]), ("annotation", annotation["annotations"])):
        if len({item["id"] for item in items}) != len(items):
            raise ValueError(f"Duplicate {key} IDs")
    for info in annotation["images"]:
        if not (Path(data["img_folder"]) / info["file_name"]).is_file():
            raise FileNotFoundError(info["file_name"])
    model, provenance = load_frozen(cfg, args.checkpoint, args.source)
    baseline, baseline_provenance = load_frozen(base_cfg, args.baseline_checkpoint, args.source)
    for configuration, info in ((cfg, provenance), (base_cfg, baseline_provenance)):
        if configuration.yaml_cfg.get("epochs") != 110 or not isinstance(info["epoch"], int) or not 0 <= info["epoch"] < 110:
            raise ValueError("Require a 110-epoch configuration/checkpoint")
    if args.expected_run2_sha256 and baseline_provenance["checkpoint_sha256"] != args.expected_run2_sha256:
        raise ValueError("Baseline fingerprint is not supplied Run2; no output created")
    if not hasattr(model.encoder, "sadf") or hasattr(baseline.encoder, "sadf"):
        raise ValueError("Require full SADF detector and disabled-module baseline")
    module = model.encoder.sadf
    if not all((module.enable_alignment, module.enable_detail, module.enable_scale_selection)):
        raise ValueError("Full checkpoint must have all SADF branches enabled")
    settings = module.settings.cpu()
    expected = torch.tensor([1, module.channels, module.dim, module.alignments[0].offset_limit,
                            module.MAX_RESIDUAL_SCALE, 1, 1, 1, int(module.use_checkpoint)])
    if not torch.equal(settings, expected):
        raise ValueError("SADF stored settings/config mismatch")
    print("检查通过：SADF最佳权重、Run2权重/标注、EMA来源、原验证变换。只评估，不训练。", flush=True)
    if args.check_only:
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not accidentally launch CPU full validation")
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    loader = cfg.val_dataloader
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {"complete": False, "SADF": provenance, "baseline_run2": baseline_provenance,
                "annotation_sha256": annotation_sha, "validation_dataset": data,
                "verified_supplied_run2_sha256": bool(args.expected_run2_sha256),
                "verified_supplied_annotation_sha256": bool(args.expected_annotation_sha256),
                "source": args.source, "batch_size": args.batch_size, "inference_dtype": "float32",
                "postprocessor_top_k": cfg.postprocessor.num_top_queries,
                "variants": args.variants, "max_images": args.max_images,
                "notes": NOTES}
    write_json(args.output_dir / "inputs.json", metadata)
    reports = {}
    baseline.to(device)
    reports["baseline_run2"], baseline_rows = evaluate_variant(baseline, cfg, loader,
        args.output_dir / "baseline_run2", baseline_provenance, "baseline_run2", args.max_images)
    baseline.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.to(device)
    # Always capture full first, even if CLI variant order differs.
    for variant in ("full", *[v for v in args.variants if v != "full"]):
        reports[variant], rows = evaluate_variant(model, cfg, loader, args.output_dir / variant,
            provenance, variant, args.max_images)
        if variant == "full":
            paired = pair_rows(baseline_rows, rows)
            write_json(args.output_dir / "SADF_runtime_summary.json", {"groups": summarize_rows(rows), "notes": NOTES})
            write_json(args.output_dir / "SADF_vs_Run2_paired_groups.json", {"groups": summarize_rows(paired, paired=True), "notes": NOTES})
            with (args.output_dir / "SADF_vs_Run2_gt_records.jsonl").open("x", encoding="utf-8") as handle:
                for row in paired:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    render_examples(args.output_dir, paired, Path(data["img_folder"]))
    write_json(args.output_dir / "SADF_interventions.json", reports)
    (args.output_dir / "SADF_runtime_report.md").write_text(report_markdown(reports, paired), encoding="utf-8")
    if sha256(args.checkpoint) != provenance["checkpoint_sha256"] or sha256(args.baseline_checkpoint) != baseline_provenance["checkpoint_sha256"] or sha256(data["ann_file"]) != annotation_sha:
        raise RuntimeError("Inputs changed during audit; report is incomplete")
    write_json(args.output_dir / "complete.json", dict(metadata, complete=True))
    print(report_markdown(reports, paired).split("## 全部统计")[0], flush=True)
    print("Finished. SADF runtime audit:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
