"""Offline query audit. Read saved tensors only; never run or train a detector.

All statuses use the exporter class-blind, original-score-first diagnostic
matching. They are not COCO FP/FN counts or human-confirmed annotation errors.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torchvision.ops import box_iou
from tools.diagnostics.query_probe_export import score_match, xyxy


def overlap_bin(value, threshold):
    """Disjoint, threshold-relative bins. Labels retain actual boundaries."""
    if value <= 0:
        return 'IoU=0'
    edges = [0.0, threshold * .2, threshold * .6, threshold * .8, threshold]
    for left, right in zip(edges[:-1], edges[1:]):
        if value < right:
            return f'{left:g}<=IoU<{right:g}'
    return f'IoU>={threshold:g}'


def score_bin(value):
    """Descriptive score ranges only; never change diagnostic thresholds."""
    if value < .7:
        return 'score<0.7'
    return '0.7<=score<0.9' if value < .9 else 'score>=0.9'


def analyze(input_dir):
    summary = json.loads((input_dir / 'audit_summary.json').read_text())
    confidence = summary['config']['confidence_threshold']
    threshold = summary['config']['matching_iou']
    files = sorted((input_dir / 'raw_queries').glob('*.pth'))
    if not files:
        raise FileNotFoundError(f'No raw query files: {input_dir / "raw_queries"}')
    names = None
    counts, unmatched_classes, low_fixed_coverage = Counter(), Counter(), Counter()
    unmatched_detail, low_detail = Counter(), Counter()
    unmatched_cross = Counter()
    records = {'high_score_misclassifications': [], 'probe_fixed_high_score': [],
               'probe_broken_high_score': [], 'duplicate_candidates': [],
               'other_unmatched_high_score': [], 'gt_without_high_score_match': [],
               'low_score_iou_selected_fixed': []}
    original_cm = probe_cm = None
    for path in files:
        raw = torch.load(path, map_location='cpu', weights_only=False)
        if names is None:
            names = raw['class_names']
            original_cm = torch.zeros(len(names), len(names), dtype=torch.long)
            probe_cm = torch.zeros_like(original_cm)
        elif raw['class_names'] != names:
            raise ValueError('Class definitions differ between images')
        gt, boxes, logits = raw['gt_labels'], raw['gt_boxes_xyxy_normalized'], raw['pred_logits']
        scores, old = logits.sigmoid().max(-1)
        new = raw['probe_logits'].argmax(-1)
        ious = box_iou(xyxy(raw['pred_boxes_cxcywh_normalized']), boxes)
        assignment, status = score_match(ious, scores, threshold, confidence)
        if not torch.equal(assignment, raw['score_first_query_to_gt']):
            raise ValueError(f'Matching differs from export: {path}')
        covered = {int(g): q for q, g in enumerate(assignment.tolist()) if g >= 0}
        correct_queries = torch.tensor([q for g, q in covered.items() if old[q] == gt[g]],
                                       dtype=torch.long)
        pred_boxes = xyxy(raw['pred_boxes_cxcywh_normalized'])
        correct_overlaps = box_iou(pred_boxes, pred_boxes[correct_queries])
        counts.update(images=1, queries=len(scores), gt=len(gt))
        def row(q, g=None):
            result = {'image_id': raw['image_id'], 'file_name': raw['file_name'],
                      'query_index': q, 'original_class': names[int(old[q])],
                      'original_score': float(scores[q]), 'probe_class': names[int(new[q])]}
            if g is not None:
                result.update(annotation_id=raw['gt_annotation_ids'][g], gt_class=names[int(gt[g])],
                              iou=float(ious[q, g]))
            return result
        for q, category in enumerate(status):
            counts[category] += 1
            g = int(assignment[q])
            if g >= 0:
                y = int(gt[g])
                original_cm[y, old[q]] += 1
                probe_cm[y, new[q]] += 1
                correct, refined = int(old[q]) == y, int(new[q]) == y
                counts['original_correct'] += int(correct)
                counts['probe_correct'] += int(refined)
                if not correct:
                    records['high_score_misclassifications'].append(row(q, g))
                if not correct and refined:
                    records['probe_fixed_high_score'].append(row(q, g))
                if correct and not refined:
                    records['probe_broken_high_score'].append(row(q, g))
            elif category != 'below_threshold':
                g = int(ious[q].argmax()) if len(gt) else None
                item = row(q, g)
                item['diagnostic_status'] = category
                unmatched_classes[(category, names[int(old[q])])] += 1
                if category == 'duplicate_candidate':
                    item['score_first_reference_query'] = covered.get(g)
                    records['duplicate_candidates'].append(item)
                else:
                    # A small GT overlap does not identify the semantic content
                    # of an unmatched box; keep these as geometric evidence.
                    maximum = float(ious[q, g]) if g is not None else 0.0
                    group = overlap_bin(maximum, threshold)
                    score_group = score_bin(float(scores[q]))
                    neighbor_iou = float(correct_overlaps[q].max()) if len(correct_queries) else 0.0
                    neighbor_q = (int(correct_queries[correct_overlaps[q].argmax()])
                                  if len(correct_queries) else None)
                    item.update(max_gt_iou=maximum, gt_iou_group=group,
                                original_score_group=score_group,
                                max_iou_with_correct_high_score_prediction=neighbor_iou,
                                nearest_correct_high_score_query=neighbor_q)
                    unmatched_detail['iou_group:' + group] += 1
                    unmatched_detail['score_group:' + score_group] += 1
                    unmatched_detail['class:' + names[int(old[q])]] += 1
                    unmatched_cross[(names[int(old[q])], group, score_group)] += 1
                    if neighbor_iou >= .5:
                        unmatched_detail['overlap_correct_prediction_iou_ge_0.5'] += 1
                    if neighbor_iou >= .8:
                        unmatched_detail['overlap_correct_prediction_iou_ge_0.8'] += 1
                    records['other_unmatched_high_score'].append(item)
        for g in range(len(gt)):
            if g not in covered:
                low = (scores < confidence) & (ious[:, g] >= threshold)
                high = (scores >= confidence) & (ious[:, g] >= threshold)
                key = ('high_score_overlap_but_unassigned' if high.any() else
                       'only_low_score_overlap' if low.any() else 'no_query_overlap_at_threshold')
                counts['gt_without_high_score_match'] += 1
                counts['uncovered_gt_' + key] += 1
                item = {'image_id': raw['image_id'], 'file_name': raw['file_name'],
                        'annotation_id': raw['gt_annotation_ids'][g], 'gt_class': names[int(gt[g])],
                        'diagnostic_reason': key, 'low_score_candidate_count': int(low.sum()),
                        'low_score_original_correct_count': int((low & (old == gt[g])).sum()),
                        'low_score_probe_correct_count': int((low & (new == gt[g])).sum())}
                if low.any():
                    candidates = torch.where(low)[0]
                    q = int(candidates[scores[candidates].argmax()])
                    item['highest_original_score_low_candidate'] = row(q, g)
                    y = int(gt[g])
                    original_probs = logits[q].sigmoid()
                    incorrect_probs = original_probs.clone()
                    incorrect_probs[y] = -1
                    best_wrong = int(incorrect_probs.argmax())
                    item['highest_original_score_low_candidate'].update(
                        original_top1_is_gt=bool(old[q] == y),
                        original_gt_sigmoid=float(original_probs[y]),
                        strongest_wrong_class=names[best_wrong],
                        strongest_wrong_sigmoid=float(original_probs[best_wrong]),
                        gt_minus_strongest_wrong_sigmoid=float(original_probs[y]-original_probs[best_wrong]))
                    correct_low = low & (old == y)
                    if correct_low.any():
                        correct_ids = torch.where(correct_low)[0]
                        best_correct = int(correct_ids[scores[correct_ids].argmax()])
                        item['highest_score_correct_class_low_candidate'] = row(best_correct, g)
                    # Restrict this aggregate to GT with ONLY low-score overlap.
                    # Unassigned high-score overlaps are a separate matching issue.
                    if key == 'only_low_score_overlap':
                        low_detail['gt_count'] += 1
                        low_detail['highest_score_candidate_gt_top1' if old[q] == y
                                   else 'highest_score_candidate_wrong_top1'] += 1
                        low_detail['any_candidate_gt_top1' if correct_low.any()
                                   else 'no_candidate_gt_top1'] += 1
                        low_detail['probe_top1_gt_on_highest_score_candidate' if new[q] == y
                                   else 'probe_top1_wrong_on_highest_score_candidate'] += 1
                        low_detail['class:' + names[y]] += 1
                records['gt_without_high_score_match'].append(item)
            q = int(raw['iou_only_gt_to_query'][g])
            if q >= 0 and scores[q] < confidence and old[q] != gt[g] and new[q] == gt[g]:
                key = ('gt_has_high_score_correct_match' if g in covered and old[covered[g]] == gt[g]
                       else 'gt_has_high_score_wrong_class_match' if g in covered
                       else 'gt_without_high_score_match')
                low_fixed_coverage[key] += 1
                item = row(q, g)
                item['coverage_group'] = key
                item['high_score_match_query'] = covered.get(g)
                records['low_score_iou_selected_fixed'].append(item)
    expected = summary['counts']
    for key in ['images', 'queries', 'gt']:
        if counts[key] != expected[key]:
            raise ValueError(f'Incomplete/inconsistent input: {key}={counts[key]} vs {expected[key]}')
    previous = summary['comparisons']['score_first_high_confidence_matches']
    for current, prior in [('matched', 'count'), ('original_correct', 'original_correct'),
                           ('probe_correct', 'probe_correct')]:
        if counts[current] != previous[prior]:
            raise ValueError(f'Paired summary reconciliation failed: {current}')
    per_class = {}
    if sum(n for key, n in unmatched_detail.items() if key.startswith('iou_group:')) != len(records['other_unmatched_high_score']):
        raise ValueError('Unmatched IoU groups do not reconcile')
    if low_detail['gt_count'] != counts['uncovered_gt_only_low_score_overlap']:
        raise ValueError('Low-score GT groups do not reconcile')
    for c, name in enumerate(names):
        n = int(original_cm[c].sum())
        per_class[name] = {'matched_gt': n, 'original_correct': int(original_cm[c, c]),
                          'probe_correct': int(probe_cm[c, c]),
                          'original_accuracy': float(original_cm[c, c]/n) if n else None,
                          'probe_accuracy': float(probe_cm[c, c]/n) if n else None}
    pairs = sorted([{'gt': names[a], 'original_prediction': names[b],
                     'original_count': int(original_cm[a, b]), 'probe_count': int(probe_cm[a, b])}
                    for a in range(len(names)) for b in range(len(names))
                    if a != b and (original_cm[a, b] or probe_cm[a, b])],
                   key=lambda r: r['original_count'], reverse=True)
    return {'source': str(input_dir), 'confidence_threshold': confidence, 'matching_iou': threshold,
            'class_names': names, 'counts': dict(counts), 'per_class': per_class,
            'confusion_matrix_rows_gt_columns_prediction': {
                'original': original_cm.tolist(), 'probe': probe_cm.tolist()},
            'confusion_pairs': pairs,
            'unmatched_predictions_by_status_and_class': [
                {'status': key[0], 'class': key[1], 'count': n}
                for key, n in sorted(unmatched_classes.items())],
            'low_score_fixed_gt_coverage': dict(low_fixed_coverage),
            'other_unmatched_high_score_breakdown': dict(unmatched_detail),
            'other_unmatched_class_iou_score_cross': [
                {'class': k[0], 'gt_iou_group': k[1], 'score_group': k[2], 'count': n}
                for k, n in sorted(unmatched_cross.items())],
            'only_low_score_gt_classification_breakdown': dict(low_detail),
            'limitations': summary['limitations'] + [
                'Uncovered GT causes are geometric/score categories, not causal diagnoses.',
                'Teacher/scope/annotation issues are not automatically removed.',
                'Probe comparisons keep original scores and matching fixed; not detector AP.'],
            'records': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', help='Default: input-dir/offline_analysis')
    args = parser.parse_args()
    source = Path(args.input_dir)
    output = Path(args.output_dir) if args.output_dir else source / 'offline_analysis'
    torch.set_num_threads(4)
    result = analyze(source)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'query_error_analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    c = result['counts']
    lines = ['Query 离线诊断（不是 COCO AP/FP/FN）',
             f'图像 {c["images"]}；GT {c["gt"]}；query {c["queries"]}',
             f'阈值：原分类最高 sigmoid ≥ {result["confidence_threshold"]}；IoU ≥ {result["matching_iou"]}',
             f'高分匹配 {c.get("matched", 0)}；原类别正确 {c.get("original_correct", 0)}；探针正确 {c.get("probe_correct", 0)}',
             f'高分误分类 {len(result["records"]["high_score_misclassifications"])}；探针改对 {len(result["records"]["probe_fixed_high_score"])}；探针改错 {len(result["records"]["probe_broken_high_score"])}',
             f'高分重复候选 {c.get("duplicate_candidate", 0)}；其他高分未匹配 {len(result["records"]["other_unmatched_high_score"])}',
             f'无高分匹配 GT {c.get("gt_without_high_score_match", 0)}',
             '', '高分匹配的主要类别混淆（原分类头 → 探针的错误数量）：']
    lines += [f'{p["gt"]} → {p["original_prediction"]}: {p["original_count"]} → {p["probe_count"]}'
              for p in result['confusion_pairs']]
    lines += ['', '低分 IoU 候选改对后，其 GT 是否已被高分预测覆盖：',
              json.dumps(result['low_score_fixed_gt_coverage'], ensure_ascii=False),
              '', '无高分匹配 GT 的几何/分数组别：']
    lines += [f'{key}: {value}' for key, value in c.items() if key.startswith('uncovered_gt_')]
    lines += ['', '其他高分未匹配预测：GT 重叠、原分数、类别分组（各维度独立统计）：']
    unmatched_labels = {'overlap_correct_prediction_iou_ge_0.5': '与已有正确高分预测的 IoU≥0.5',
                        'overlap_correct_prediction_iou_ge_0.8': '与已有正确高分预测的 IoU≥0.8'}
    for key, value in sorted(result['other_unmatched_high_score_breakdown'].items()):
        label = unmatched_labels.get(key, key.replace('iou_group:', '最大GT重叠：')
                                     .replace('score_group:', '原分数：').replace('class:', '预测类别：'))
        lines.append(f'{label}: {value}')
    lines += ['', '只有低分候选覆盖的 GT：类别是否已经排第一：']
    low_labels = {'gt_count': '仅低分候选覆盖的GT总数',
                  'highest_score_candidate_gt_top1': '最高原分数候选：GT类别已排第一',
                  'highest_score_candidate_wrong_top1': '最高原分数候选：类别仍错误',
                  'any_candidate_gt_top1': '至少有一个低分候选的GT类别排第一',
                  'no_candidate_gt_top1': '所有重叠低分候选的第一类别均错误',
                  'probe_top1_gt_on_highest_score_candidate': '探针对最高原分数候选分类正确',
                  'probe_top1_wrong_on_highest_score_candidate': '探针对最高原分数候选分类错误'}
    lines += [f'{low_labels.get(key, key.replace("class:", "GT类别："))}: {value}'
              for key, value in sorted(result['only_low_score_gt_classification_breakdown'].items())]
    lines += ['', '解读：GT 类别排第一但低于阈值，只是评分不足的候选证据，不证明应直接提分。',
              '预测与已有正确预测重叠，只是冗余候选证据，可能也涉及相邻或遮挡目标。']
    lines += ['', '注意：重复候选不等于人工确认的重复误检。无重叠不等于真正背景。',
              '没有自动排除教师或修改标注。未使用探针 softmax 替换检测置信度。',
              '每条记录的图片名、GT 编号、query 编号均保存在 JSON records 中。']
    text = '\n'.join(lines) + '\n'
    (output / 'analysis_summary.txt').write_text(text, encoding='utf-8')
    print(text)
    print(f'Finished: {output}')


if __name__ == '__main__':
    main()
