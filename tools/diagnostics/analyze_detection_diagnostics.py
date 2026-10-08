"""Analyze one complete validation export; do not train or rescore the detector.

COCO PR curves come from the saved COCO eval.pth, not a custom AP calculation.
GT-best-IoU is an oracle coverage statistic, not a prediction assignment.
Score histograms use an explicitly separate class-aware score-first IoU=0.5
one-to-one diagnostic, without COCO crowd/ignore or maxDets handling.
Read only trusted .pt/.pth artifacts produced by this project.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_convert, box_iou


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def finite_mean(values):
    return float(np.mean(values)) if len(values) else None


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {"count": len(values), "mean": finite_mean(values),
            "quantiles_0_10_25_50_75_90_100": np.quantile(values, [0, .1, .25, .5, .75, .9, 1]).tolist() if len(values) else [],
            "histogram_edges": np.linspace(0, 1, 11).tolist(),
            "histogram_counts": np.histogram(values, bins=np.linspace(0, 1, 11))[0].tolist(),
            "fraction_ge_threshold": {str(t): finite_mean(values >= t) for t in (.3, .5, .75, .9)}}


def pearson(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def score_first_match(ious, scores, threshold, pred_labels=None, gt_labels=None):
    """Highest score first; highest eligible IoU; each prediction/GT used once."""
    assignment = torch.full((len(scores),), -1, dtype=torch.long)
    used = torch.zeros(ious.shape[1], dtype=torch.bool)
    for p in torch.argsort(scores, descending=True, stable=True).tolist():
        eligible = (~used) & (ious[p] >= threshold)
        if pred_labels is not None:
            eligible &= gt_labels == pred_labels[p]
        if eligible.any():
            g = int(ious[p].masked_fill(~eligible, -1).argmax())
            assignment[p] = g
            used[g] = True
    return assignment


def load_coco_pr(eval_file, ids, image_ids=None):
    saved = torch.load(eval_file, map_location="cpu", weights_only=False)
    params = saved.get("params")
    if params is None or list(params.catIds) != ids:
        raise ValueError("COCO eval.pth category IDs do not match the annotations")
    if image_ids is not None and set(params.imgIds) != set(image_ids):
        raise ValueError("COCO eval.pth image IDs do not match the annotations")
    precision, score_grid = np.asarray(saved["precision"]), np.asarray(saved["scores"])
    if precision.shape != score_grid.shape or precision.ndim != 5:
        raise ValueError("Unexpected COCO precision/scores dimensions")
    area_index = list(params.areaRngLbl).index("all")
    maxdet_index = list(params.maxDets).index(100)
    if precision.shape[2] != len(ids):
        raise ValueError("COCO category dimension differs from annotations")
    def valid_mean(values):
        values = np.asarray(values)
        valid = values[values >= 0]
        return float(valid.mean()) if valid.size else None
    aggregate = {"AP": valid_mean(precision[:, :, :, area_index, maxdet_index])}
    for name, index in zip(params.areaRngLbl, range(len(params.areaRngLbl))):
        aggregate[f"AP_{name}"] = valid_mean(precision[:, :, :, index, maxdet_index])
    recall = np.asarray(saved["recall"])
    aggregate["AR100"] = valid_mean(recall[:, :, area_index, maxdet_index])
    per_class = {str(c): {"AP": valid_mean(precision[:, :, k, area_index, maxdet_index]),
                         "AR100": valid_mean(recall[:, k, area_index, maxdet_index])} for k, c in enumerate(ids)}
    curves = {}
    for threshold in (.5, .75):
        indices = np.flatnonzero(np.isclose(params.iouThrs, threshold))
        if len(indices) != 1:
            raise ValueError(f"Missing COCO IoU={threshold}")
        aggregate[f"AP{int(threshold*100)}"] = valid_mean(precision[int(indices[0]), :, :, area_index, maxdet_index])
        curves[str(threshold)] = {}
        for k, category_id in enumerate(ids):
            p = precision[int(indices[0]), :, k, area_index, maxdet_index]
            s = score_grid[int(indices[0]), :, k, area_index, maxdet_index]
            per_class[str(category_id)][f"AP{int(threshold*100)}"] = valid_mean(p)
            curves[str(threshold)][str(category_id)] = {
                "recall": np.asarray(params.recThrs).tolist(),
                "precision": [float(x) if x >= 0 else None for x in p],
                "score_at_recall_grid": [float(x) if p[i] >= 0 else None for i, x in enumerate(s)],
            }
    return {"definition": "Saved COCO interpolated precision at 101 recall points; all areas; maxDets=100. Not the conf=0.5 Validator matrix.",
            "curves": curves, "area_ranges": dict(zip(params.areaRngLbl, params.areaRng)),
            "saved_COCO_metrics": aggregate, "saved_COCO_per_class": per_class}


def original_gt(annotations, image):
    """Mirror dataset clipping, crowd exclusion and degenerate-box removal."""
    valid = []
    for ann in annotations:
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        box = [max(0, min(image["width"], x)), max(0, min(image["height"], y)),
               max(0, min(image["width"], x + w)), max(0, min(image["height"], y + h))]
        if box[2] > box[0] and box[3] > box[1]:
            area = float(ann.get("area", w * h))
            if not math.isfinite(area) or area < 0:
                raise ValueError("Invalid annotation area")
            valid.append({"annotation_id": ann["id"], "class_id": ann["category_id"],
                          "box": box, "area": area, "ignore": bool(ann.get("ignore", 0))})
    return valid


def candidate_status(ious, scores, gt_class, selected, confidence, threshold):
    overlapping = ious >= threshold
    if not overlapping.any():
        return "no_query_IoU_ge_threshold"
    top_class = scores.argmax(1)
    correct_first = overlapping & (top_class == gt_class)
    if not correct_first.any():
        return "overlap_exists_but_GT_class_never_top1"
    high = correct_first & (scores[:, gt_class] >= confidence)
    if not high.any():
        return "GT_class_top1_but_below_confidence"
    selected_queries = selected[selected % scores.shape[1] == gt_class] // scores.shape[1]
    if len(selected_queries) and high[selected_queries].any():
        return "GT_class_top1_high_score_and_selected_candidate"
    return "GT_class_top1_high_score_but_not_selected"


BEST_QUERY_TYPES = {
    "A": "best-IoU类别正确但低分；已有高分同类预测候选覆盖",
    "B": "best-IoU类别正确但低分；没有高分同类预测候选覆盖",
    "C": "best-IoU第一类别错误；已有高分同类预测候选覆盖",
    "D": "best-IoU第一类别错误；没有高分同类预测候选覆盖",
    "E": "best-IoU类别正确且高分；该GT类别项被后处理Top-k淘汰",
    "F": "best-IoU类别正确且高分；该GT类别项已被后处理选中",
    "G": "所有query的IoU均低于阈值",
}


def best_query_evidence(ious, scores, gt_class, selected, confidence, threshold):
    """Per-GT candidate evidence, NOT a one-to-one detection assignment.

    D-FINE selects query-class PAIRS, not just each query's argmax class. A
    selected, high-score GT-class pair counts even when it is second-ranked.
    Otherwise a wrong argmax can be incorrectly called a real detection gap.
    """
    best = int(ious.argmax())
    top_class = scores.argmax(1)
    gt_scores = scores[:, gt_class]
    overlapping = ious >= threshold
    pair_selected = torch.zeros(len(scores), dtype=torch.bool)
    selected_gt = selected[selected % scores.shape[1] == gt_class] // scores.shape[1]
    pair_selected[selected_gt] = True
    high_correct = overlapping & (gt_scores >= confidence) & pair_selected
    high_top1 = high_correct & (top_class == gt_class)
    other = torch.arange(len(scores)) != best
    covered = bool(high_correct.any())
    if not bool(overlapping.any()):
        kind = "G"
    elif int(top_class[best]) != gt_class:
        kind = "C" if covered else "D"
    elif float(gt_scores[best]) < confidence:
        kind = "A" if covered else "B"
    else:
        kind = "F" if bool(pair_selected[best]) else "E"
    if covered:
        gap = "none"
    elif not bool(overlapping.any()):
        gap = "no_geometric_overlap"
    elif not bool((overlapping & (gt_scores >= confidence)).any()):
        gap = "all_overlapping_GT_class_scores_below_confidence"
    else:
        gap = "high_GT_class_pair_not_selected_by_topk"
    return {
        "best_query_evidence_type": kind,
        "best_query_evidence_description": BEST_QUERY_TYPES[kind],
        "high_correct_selected_candidate_exists": covered,
        "high_correct_selected_candidate_query_ids": torch.nonzero(high_correct).flatten().tolist(),
        "another_high_correct_selected_query_exists": bool((high_correct & other).any()),
        "another_high_correct_top1_selected_query_exists": bool((high_top1 & other).any()),
        "best_query_GT_pair_high_and_selected": bool(high_correct[best]),
        "any_overlapping_GT_top1_query_exists": bool((overlapping & (top_class == gt_class)).any()),
        "candidate_gap_reason": gap,
    }


def summarize_best_query_evidence(records, names, area_ranges, confidence, threshold):
    def group(rows):
        counts = Counter(r["best_query_evidence_type"] for r in rows)
        return {
            "GT_count": len(rows), "ignored_GT_count": sum(r["ignore"] for r in rows),
            "type_counts": {k: counts[k] for k in BEST_QUERY_TYPES},
            "high_correct_selected_candidate_exists": sum(r["high_correct_selected_candidate_exists"] for r in rows),
            "no_high_correct_selected_candidate": sum(not r["high_correct_selected_candidate_exists"] for r in rows),
            "candidate_gap_reasons": dict(Counter(r["candidate_gap_reason"] for r in rows)),
            "class_aware_score_first_IoU05_high_matched": sum(r["class_aware_score_first_IoU05_high_matched"] for r in rows),
            "candidate_exists_but_not_one_to_one_high_matched": sum(
                r["high_correct_selected_candidate_exists"] and not r["class_aware_score_first_IoU05_high_matched"] for r in rows),
        }
    return {
        "definition": "GT-wise candidate cross-classification using the original selected query-class pairs and scores; NOT COCO TP/FN or a causal diagnosis.",
        "confidence_threshold": confidence, "IoU_threshold": threshold,
        "type_definitions": BEST_QUERY_TYPES,
        "all": group(records),
        "by_class": {name: group([r for r in records if r["gt_class_id"] == cls]) for cls, name in names.items()},
        "by_size_and_class": {
            size: {name: group([r for r in records if r["gt_class_id"] == cls and low <= r["area_original_annotation"] <= high])
                   for cls, name in names.items()} for size, (low, high) in area_ranges.items()
        },
        "warnings": [
            "Correct candidate means GT-class score>=threshold, IoU>=threshold and selected by the actual postprocessor. Its GT class need not be the query's argmax.",
            "A/C show an available correct candidate, not necessarily a redundant FP; the same candidate can overlap multiple GTs.",
            "B/D/no-high-candidate are candidate deficit evidence, not proof of the cause or of a COCO miss; annotation mistakes/occlusion may contribute.",
            "Class-aware score-first matches use IoU=0.5, one-to-one, no COCO ignore/crowd/maxDets rules; keep distinct from the original class-blind Validator matrix.",
            "COCO size boundaries are inclusive and can overlap. Ignore flags are retained; no annotation or teacher exclusion is applied.",
        ],
    }


def score_report(records, gt_count, class_ids, confidence):
    result = {}
    for key, subset in [("all", records)] + [(str(c), [r for r in records if r["class_id"] == c]) for c in class_ids]:
        tps = [r["score"] for r in subset if r["diagnostic_tp"]]
        fps = [r["score"] for r in subset if not r["diagnostic_tp"]]
        bins = []
        for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
            part = [r for r in subset if low <= r["score"] < high or (high == 1 and r["score"] == 1)]
            bins.append({"low": float(low), "high": float(high), "count": len(part),
                         "diagnostic_precision": finite_mean([r["diagnostic_tp"] for r in part]),
                         "mean_max_same_class_IoU": finite_mean([r["max_same_class_gt_iou"] for r in part])})
        denominator = sum(gt_count.values()) if key == "all" else gt_count[int(key)]
        sweep = []
        for threshold in (0, .1, .2, .3, .4, .5, .6, .7, .8, .9):
            keep = [r for r in subset if r["score"] >= threshold]
            tp = sum(r["diagnostic_tp"] for r in keep)
            sweep.append({"confidence": threshold, "predictions": len(keep), "TP": tp,
                          "FP": len(keep) - tp, "FN": denominator - tp,
                          "precision": tp / len(keep) if keep else None,
                          "recall": tp / denominator if denominator else None})
        result[key] = {"TP_scores": distribution(tps), "FP_scores": distribution(fps),
                       "TP_scores_above_confidence": distribution([s for s in tps if s >= confidence]),
                       "FP_scores_above_confidence": distribution([s for s in fps if s >= confidence]),
                       "pearson_score_vs_max_same_class_IoU": pearson([r["score"] for r in subset], [r["max_same_class_gt_iou"] for r in subset]),
                       "score_bins": bins, "diagnostic_threshold_sweep": sweep}
    return {"definition": "Class-aware score-first greedy one-to-one matching, IoU>=0.5, all exported top-k pairs, no COCO maxDets/ignore/crowd handling. FP means diagnostic unmatched, NOT proven background. Maximum IoU alone is not TP.",
            "confidence_threshold": confidence, "groups": result}


def plot_reports(output, pr, scores, coverage, names):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    def figure():
        fig = Figure(figsize=(8, 5)); FigureCanvasAgg(fig)
        return fig, fig.subplots()
    for threshold, curves in pr["curves"].items():
        fig, ax = figure()
        for category_id, values in curves.items():
            p = np.array([np.nan if x is None else x for x in values["precision"]])
            ax.plot(values["recall"], p, label=names[int(category_id)])
        ax.set(xlabel="Recall", ylabel="Interpolated precision", xlim=(0, 1), ylim=(0, 1), title=f"Saved COCO PR | IoU={threshold} | maxDets=100")
        ax.legend(); fig.tight_layout(); fig.savefig(output / f"coco_pr_iou{threshold}.png", dpi=160)
    fig, ax = figure()
    for category_id, name in names.items():
        hist = coverage["by_class"][str(category_id)]
        ax.stairs(hist["histogram_counts"], hist["histogram_edges"], label=name)
    ax.set(xlabel="GT-best IoU over ALL queries (oracle)", ylabel="GT count", title="Candidate coverage, NOT detector recall")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "gt_best_iou_distribution.png", dpi=160)
    fig = Figure(figsize=(max(8, 4 * len(names)), 4)); FigureCanvasAgg(fig)
    axes = np.atleast_1d(fig.subplots(1, len(names)))
    for ax, (category_id, name) in zip(axes, names.items()):
        group = scores["groups"][str(category_id)]
        for kind in ("TP_scores_above_confidence", "FP_scores_above_confidence"):
            hist = group[kind]
            ax.stairs(hist["histogram_counts"], hist["histogram_edges"], label=kind)
        ax.set(xlabel="Original final sigmoid score", ylabel="Prediction count", title=name)
        ax.legend()
    fig.suptitle(f"Diagnostic class-aware score-first matching | IoU>=0.5 | score>={scores['confidence_threshold']}")
    fig.tight_layout(); fig.savefig(output / "diagnostic_tp_fp_scores.png", dpi=160)
    fig, ax = figure()
    for category_id, name in names.items():
        bins = scores["groups"][str(category_id)]["score_bins"]
        ax.plot([(b["low"] + b["high"]) / 2 for b in bins],
                [b["mean_max_same_class_IoU"] if b["count"] else np.nan for b in bins], marker="o", label=name)
    ax.set(xlabel="Score-bin center", ylabel="Mean maximum same-class GT IoU", ylim=(0, 1), title="Geometric score relationship, NOT LQE branch attribution")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "score_quality_relationship.png", dpi=160)


def analyze(input_dir, annotation_file=None, output_dir=None, eval_file=None):
    input_dir = Path(input_dir)
    query_dir = input_dir / "query_diagnostics"
    metadata = json.loads((query_dir / "metadata.json").read_text())
    if not metadata.get("complete"):
        raise ValueError("Query export is incomplete; do not analyze a partially written directory")
    annotation_file = Path(annotation_file or metadata["config"]["val_dataloader"]["dataset"]["ann_file"])
    dataset = json.loads(annotation_file.read_text())
    categories = sorted(dataset["categories"], key=lambda c: c["id"])
    names = {c["id"]: c["name"] for c in categories}
    ids = list(names)
    if ids != list(range(metadata["num_classes"])):
        raise ValueError("Analysis requires non-remapped contiguous category IDs 0..C-1")
    pr = load_coco_pr(Path(eval_file or input_dir / "eval.pth"), ids, [i["id"] for i in dataset["images"]])
    output = Path(output_dir or input_dir / "analysis")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Analysis output is not empty; choose a new --output-dir to avoid mixing results")
    images = {x["id"]: x for x in dataset["images"]}
    annotations = defaultdict(list)
    for ann in dataset["annotations"]:
        annotations[ann["image_id"]].append(ann)
    files = sorted((query_dir / "raw_queries").glob("*.pt"))
    if len(files) != metadata["images"] or len(files) != len(images):
        raise ValueError("Export does not cover the entire annotation image set")
    confidence = float(metadata["confidence_threshold"])
    threshold = float(metadata["gt_iou_threshold"])
    gt_rows, pred_rows, gt_count, statuses = [], [], Counter(), Counter()
    seen, total_queries, legacy_scores = set(), 0, 0
    for number, path in enumerate(files, 1):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        image_id = int(raw["image_id"])
        if image_id not in images or image_id in seen:
            raise ValueError(f"Unexpected or duplicate image ID {image_id}")
        seen.add(image_id)
        gt_info = original_gt(annotations[image_id], images[image_id])
        gt = raw["gt_boxes_xyxy"].float(); labels = raw["gt_labels"].long()
        expected_boxes = torch.tensor([g["box"] for g in gt_info], dtype=torch.float32).reshape(-1, 4)
        if len(gt_info) != len(gt) or not torch.equal(labels, torch.tensor([g["class_id"] for g in gt_info], dtype=torch.long)) or not torch.allclose(gt, expected_boxes, atol=.02, rtol=1e-5):
            raise ValueError(f"Original annotations disagree with exported GT for image {image_id}; check dataset version/order/transforms")
        logits = raw["pred_logits"].float()
        if "pred_scores" in raw:
            scores = raw["pred_scores"].float()
        else:
            scores = logits.sigmoid(); legacy_scores += 1
        if "pred_boxes_xyxy" in raw:
            boxes = raw["pred_boxes_xyxy"].float()
        else:
            boxes = box_convert(raw["pred_boxes_cxcywh"].float(), "cxcywh", "xyxy") * torch.tensor(raw["original_size_wh"] * 2)
        if scores.shape != logits.shape or boxes.shape != (len(logits), 4) or scores.shape[1] != len(ids) or not torch.isfinite(scores).all() or not torch.isfinite(boxes).all():
            raise ValueError(f"Invalid raw prediction shapes/values: {path}")
        if len(scores) == 0 or (scores < 0).any() or (scores > 1).any():
            raise ValueError("No queries or invalid probability range")
        total_queries += len(scores)
        flat = raw["selected_flat_indices"].long()
        if flat.ndim != 1 or len(flat) != metadata["postprocessor_top_k"] or len(flat.unique()) != len(flat) or (flat < 0).any() or (flat >= scores.numel()).any():
            raise ValueError("Invalid saved postprocessor selection")
        qids, pred_labels = flat // len(ids), flat % len(ids)
        selected_scores = raw.get("selected_scores", scores.flatten()[flat]).float()
        if len(selected_scores) != len(flat) or not torch.allclose(selected_scores, scores.flatten()[flat], atol=1e-6, rtol=0):
            raise ValueError("Saved selected scores disagree with raw class scores")
        ious = box_iou(boxes, gt)
        high = selected_scores >= confidence
        high_indices = torch.nonzero(high).flatten()
        high_match = score_first_match(ious[qids[high]], selected_scores[high], threshold)
        matched_gt = {int(g): int(high_indices[p]) for p, g in enumerate(high_match.tolist()) if g >= 0}
        # This is a score-distribution diagnostic, distinct from the Validator's
        # class-blind IoU-first matrix and the saved official COCO PR curves.
        diagnostic_match = score_first_match(ious[qids], selected_scores, .5, pred_labels, labels)
        class_aware_high_matched_gt = {
            int(g) for p, g in enumerate(diagnostic_match.tolist()) if g >= 0 and bool(high[p])
        }
        for g, info in enumerate(gt_info):
            cls = int(labels[g]); gt_count[cls] += 1
            best = int(ious[:, g].argmax())
            status = candidate_status(ious[:, g], scores, cls, flat, confidence, threshold)
            statuses[status] += 1
            matched = matched_gt.get(g)
            best_class = int(scores[best].argmax())
            selected_flat = best * len(ids) + cls
            row = {"image_id": image_id, "image_name": images[image_id]["file_name"], "gt_index": g,
                   "annotation_id": info["annotation_id"], "gt_class_id": cls, "gt_class": names[cls],
                   "area_original_annotation": info["area"], "ignore": info["ignore"],
                   "box_xyxy": gt[g].tolist(), "best_query_id": best, "best_query_IoU": float(ious[best, g]),
                   "best_query_top_class": names[best_class], "best_query_GT_class_score": float(scores[best, cls]),
                   "best_query_all_class_scores": scores[best].tolist(), "best_query_GT_class_rank": 1 + int((scores[best] > scores[best, cls]).sum()),
                   "best_query_GT_pair_selected": bool((flat == selected_flat).any()),
                   "candidate_status": status, "score_first_high_match_query_id": int(qids[matched]) if matched is not None else None,
                   "score_first_high_match_class": names[int(pred_labels[matched])] if matched is not None else None,
                   "score_first_high_match_score": float(selected_scores[matched]) if matched is not None else None,
                   "score_first_high_match_IoU": float(ious[qids[matched], g]) if matched is not None else None,
                   "score_first_high_match_correct_class": bool(pred_labels[matched] == cls) if matched is not None else None}
            row.update(best_query_evidence(ious[:, g], scores, cls, flat, confidence, threshold))
            row["class_aware_score_first_IoU05_high_matched"] = g in class_aware_high_matched_gt
            gt_rows.append(row)
        for rank, (q, cls) in enumerate(zip(qids.tolist(), pred_labels.tolist())):
            g = int(diagnostic_match[rank])
            same = labels == cls
            pred_rows.append({"image_id": image_id, "image_name": images[image_id]["file_name"],
                              "rank": rank, "query_id": q, "class_id": cls, "class_name": names[cls],
                              "score": float(selected_scores[rank]), "above_diagnostic_confidence": bool(high[rank]),
                              "diagnostic_tp": g >= 0, "diagnostic_matched_gt_index": g if g >= 0 else None,
                              "diagnostic_matched_annotation_id": gt_info[g]["annotation_id"] if g >= 0 else None,
                              "diagnostic_matched_IoU": float(ious[q, g]) if g >= 0 else None,
                              "max_gt_iou": float(ious[q].max()) if len(gt) else 0.,
                              "max_same_class_gt_iou": float(ious[q, same].max()) if same.any() else 0.,
                              "box_xyxy": boxes[q].tolist()})
        if number % 100 == 0:
            print(f"Analyzed {number}/{len(files)} images", flush=True)
    if seen != set(images) or total_queries != metadata["queries"]:
        raise ValueError("Incomplete/mismatched image or query totals")
    coverage = {"definition": "Maximum IoU over ALL exported query boxes for each GT; class-blind, no confidence filter, no one-to-one assignment; oracle coverage NOT detector Recall.",
                "all": distribution([r["best_query_IoU"] for r in gt_rows]),
                "by_class": {str(c): distribution([r["best_query_IoU"] for r in gt_rows if r["gt_class_id"] == c]) for c in ids},
                "by_size": {}}
    sizes = {}
    for name, (low, high) in pr["area_ranges"].items():
        subset = [r for r in gt_rows if low <= r["area_original_annotation"] <= high]
        sizes[name] = {"inclusive_area_range": [float(low), float(high)], "GT_count": len(subset),
                       "saved_COCO_AP": pr["saved_COCO_metrics"].get(f"AP_{name}"),
                       "images_with_GT": len({r["image_id"] for r in subset}),
                       "class_counts": {names[c]: sum(r["gt_class_id"] == c for r in subset) for c in ids},
                       "GT_area_distribution": {"min": min((r["area_original_annotation"] for r in subset), default=None),
                                                "median": float(np.median([r["area_original_annotation"] for r in subset])) if subset else None,
                                                "max": max((r["area_original_annotation"] for r in subset), default=None)}}
        coverage["by_size"][name] = distribution([r["best_query_IoU"] for r in subset])
    coverage["coverage_at_IoU"] = {str(t): {"count": sum(r["best_query_IoU"] >= t for r in gt_rows),
                                           "fraction": finite_mean([r["best_query_IoU"] >= t for r in gt_rows])} for t in (.3, .5, .75, .9)}
    scores_report = score_report(pred_rows, gt_count, ids, confidence)
    cross = summarize_best_query_evidence(gt_rows, names, pr["area_ranges"], confidence, threshold)
    matrix_file = input_dir / "confusion_matrix/confusion_matrix.json"
    if not matrix_file.is_file():
        raise FileNotFoundError("Missing exported Validator confusion matrix; enable both export flags")
    matrix = json.loads(matrix_file.read_text())
    if matrix["category_ids"] != ids or matrix["GT_count"] != len(gt_rows) or matrix["confidence_threshold"] != confidence or matrix["iou_threshold"] != threshold:
        raise ValueError("Confusion matrix disagrees with query export")
    if matrix.get("metadata", {}).get("checkpoint") != metadata.get("checkpoint"):
        raise ValueError("Confusion and query metadata refer to different checkpoints")
    high_total = sum(r["above_diagnostic_confidence"] for r in pred_rows)
    if matrix["predictions_after_threshold"] != high_total:
        raise ValueError("Confusion matrix prediction total disagrees with saved postprocessor selection")
    matrix_counts = matrix["matrix_counts"]
    confusion_pairs = [{"gt": names[ids[i]], "prediction": names[ids[j]], "count": int(matrix_counts[i][j]),
                        "class_GT_count": gt_count[ids[i]],
                        "fraction_of_class_GT": matrix_counts[i][j] / gt_count[ids[i]] if gt_count[ids[i]] else None}
                       for i in range(len(ids)) for j in range(len(ids)) if i != j]
    warnings = ["One run cannot establish repeatability or small-object stability.",
                "GT-best-IoU and candidate_status are oracle candidate evidence, not detector TP/FN or a causal layer diagnosis.",
                "Unmatched/diagnostic FP predictions are not proven background; duplicates, localization, labels and out-of-scope targets may contribute.",
                "Final sigmoid scores include LQE; its separate contribution is not exported or attributed.",
                "Size ranges follow saved COCO inclusive boundaries; GT exactly on a boundary may belong to two ranges."]
    if sizes.get("small", {}).get("GT_count", 0) < 30:
        warnings.append("Small GT count is below 30: descriptive low-sample warning only, not a statistical reliability cutoff. Inspect counts/images/classes and repeat runs before interpreting APsmall.")
    if legacy_scores:
        warnings.append(f"{legacy_scores} legacy images lack exact saved scores; CPU sigmoid/box reconstruction can differ near thresholds. Prefer a new schema_version=2 export.")
    if any(a.get("iscrowd", 0) or a.get("ignore", 0) for a in dataset["annotations"]):
        warnings.append("Dataset includes crowd/ignore annotations. Official COCO curves honor COCO rules; custom diagnostic score histograms do not.")
    summary = {"checkpoint": metadata.get("checkpoint"), "checkpoint_sha256": metadata.get("checkpoint_sha256"),
               "weight_source": matrix.get("metadata", {}).get("weight_source"),
               "annotation_file": str(annotation_file), "images": len(files), "queries": total_queries, "GT": len(gt_rows),
               "selected_pairs": len(pred_rows), "selected_pairs_above_confidence": high_total,
               "confidence_threshold": confidence, "IoU_threshold": threshold, "class_names": names,
               "saved_COCO_metrics": pr["saved_COCO_metrics"], "saved_COCO_per_class": pr["saved_COCO_per_class"],
               "validator_confusion": {key: matrix[key] for key in ("matching", "correct_class_matches", "misclassifications", "unmatched_predictions", "unmatched_GT")},
               "confusion_pairs": sorted(confusion_pairs, key=lambda r: -r["count"]),
               "oracle_candidate_status_counts": dict(statuses), "GT_size_counts": sizes,
               "best_query_cross_classification": cross,
               "oracle_candidate_status_counts_by_class": {names[c]: dict(Counter(r["candidate_status"] for r in gt_rows if r["gt_class_id"] == c)) for c in ids},
               "high_confidence_score_first_class_blind_matches": {
                   names[c]: {"GT_count": gt_count[c],
                              "matched": sum(r["score_first_high_match_query_id"] is not None for r in gt_rows if r["gt_class_id"] == c),
                              "correct_class": sum(r["score_first_high_match_correct_class"] is True for r in gt_rows if r["gt_class_id"] == c),
                              "wrong_class": sum(r["score_first_high_match_correct_class"] is False for r in gt_rows if r["gt_class_id"] == c)} for c in ids},
               "repeatability": "not assessed; requires independent baseline runs", "warnings": warnings}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    write_json(output / "gt_best_iou_distribution.json", coverage)
    write_json(output / "gt_size_statistics.json", sizes)
    write_json(output / "coco_pr_curves.json", pr)
    write_json(output / "score_distributions_and_quality.json", scores_report)
    write_json(output / "best_query_cross_classification.json", cross)
    for filename, records in (("gt_records.jsonl", gt_rows), ("prediction_records.jsonl", pred_rows),
                              ("best_query_cross_classification.jsonl", gt_rows)):
        with (output / filename).open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    lines = ["Run1 检测诊断（只分析，不训练、不改分数、不修改标注）",
             f"图像 {len(files)}；GT {len(gt_rows)}；query {total_queries}",
             "保存的 COCO 结果（0~1）：" + json.dumps(pr["saved_COCO_metrics"], ensure_ascii=False),
             f"原 Validator 混淆矩阵：{summary['validator_confusion']}",
             "各类别保存的COCO指标（0~1）：" + json.dumps({names[int(c)]: v for c, v in pr["saved_COCO_per_class"].items()}, ensure_ascii=False),
             "类别混淆（原 Validator，GT → 预测）："]
    lines += [f"{r['gt']} → {r['prediction']}: {r['count']} / {r['class_GT_count']} GT" for r in summary["confusion_pairs"]]
    lines += ["GT-best IoU 候选覆盖（不是检测 Recall）：", json.dumps(coverage["coverage_at_IoU"], ensure_ascii=False),
              "原图 annotation area 尺寸分组：", json.dumps({k: v["GT_count"] for k, v in sizes.items()}, ensure_ascii=False),
              "候选证据分组（不是漏检原因判定）：", json.dumps(dict(statuses), ensure_ascii=False),
              "PR 曲线读取保存的 COCO eval.pth；分数直方图使用另一个明确标注的 score-first 类别匹配诊断。",
              "一次实验无法判断稳定性，也不能直接确认哪个网络层导致问题。"]
    lines += ["best-IoU交叉统计（候选证据，不是COCO漏检原因）："]
    lines += [f"{kind} {description}: {cross['all']['type_counts'][kind]}" for kind, description in BEST_QUERY_TYPES.items()]
    for name, stats in cross["by_class"].items():
        lines += [f"{name}: GT={stats['GT_count']}；无高分同类候选={stats['no_high_correct_selected_candidate']}；"
                  f"候选缺口={stats['candidate_gap_reasons']}；类别一致score-first IoU=0.5高分匹配={stats['class_aware_score_first_IoU05_high_matched']}"]
    if "medium" in cross["by_size_and_class"]:
        lines += ["中尺度类别×候选缺口：" + json.dumps(cross["by_size_and_class"]["medium"], ensure_ascii=False)]
    lines += ["NOTE: " + w for w in cross["warnings"]]
    lines += ["WARNING: " + w for w in warnings]
    text = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(text, encoding="utf-8")
    plot_reports(output, pr, scores_report, coverage, names)
    print(text, flush=True)
    print("Analysis saved:", output, flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Validation output containing query_diagnostics, confusion_matrix and eval.pth")
    parser.add_argument("--annotation-file", help="Override the metadata's validation annotation path, e.g. for local analysis")
    parser.add_argument("--output-dir", help="Empty/new output directory; default INPUT/analysis")
    parser.add_argument("--eval-file", help="Override the saved COCO eval.pth path")
    args = parser.parse_args()
    analyze(args.input_dir, args.annotation_file, args.output_dir, args.eval_file)


if __name__ == "__main__":
    main()
