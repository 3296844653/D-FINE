"""Export every validation query and audit the frozen stage classification probe.

No detector parameter, loss, postprocessor or COCO metric is changed. Diagnostic
matching is class-blind and is NOT the official detector evaluation. Probe
softmax is conditional on the five foreground classes, not object confidence.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from PIL import Image
from torchvision.ops import box_iou
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from tools.diagnostics.stage_feature_probe import build_detector, match_queries
from tools.diagnostics.gt_box_classifier import load_config, seed_everything


def xyxy(boxes):
    """Normalized cxcywh to xyxy, without changing detector box geometry."""
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                      boxes[:, :2] + boxes[:, 2:] / 2), -1)


def score_match(ious, scores, threshold, confidence):
    """Greedy class-blind diagnostic match, higher original score first.

    Uses top-1 per query only, unlike flattened query/class COCO postprocessing.
    Duplicate means overlap with an already assigned GT at the chosen threshold.
    """
    assignment = torch.full((len(scores),), -1, dtype=torch.long)
    status = ['below_threshold'] * len(scores)
    used = torch.zeros(ious.shape[1], dtype=torch.bool)
    for q in scores.argsort(descending=True).tolist():
        if scores[q] < confidence:
            continue
        available = (ious[q] >= threshold) & ~used
        if available.any():
            candidates = torch.where(available)[0]
            g = int(candidates[ious[q, candidates].argmax()])
            assignment[q] = g
            used[g] = True
            status[q] = 'matched'
        elif (ious[q] >= threshold).any():
            status[q] = 'duplicate_candidate'
        elif (ious[q] > 0).any():
            status[q] = 'unmatched_with_overlap'
        else:
            status[q] = 'unmatched_no_overlap'
    return assignment, status


def comparison_stats(gt, old, new, mask):
    """Paired counts with explicit denominators; empty accuracy is unavailable."""
    gt, old, new = gt[mask], old[mask], new[mask]
    a, b = old == gt, new == gt
    return {'count': len(gt), 'original_correct': int(a.sum()),
            'probe_correct': int(b.sum()), 'fixed': int((~a & b).sum()),
            'broken': int((a & ~b).sum()), 'both_wrong': int((~a & ~b).sum()),
            'original_accuracy': float(a.float().mean()) if len(gt) else None,
            'probe_accuracy': float(b.float().mean()) if len(gt) else None}


@torch.no_grad()
def run(config, output_dir, device):
    probe = torch.load(config['probe_checkpoint'], map_location='cpu', weights_only=False)
    if probe.get('stage') != 'decoder_query':
        raise ValueError('Expected a decoder_query linear probe, not an ROI probe')
    annotation = json.loads(Path(config['data']['val']['annotation']).read_text())
    categories = sorted(annotation['categories'], key=lambda c: c['id'])
    names = [c['name'] for c in categories]
    if names != probe['class_names']:
        raise ValueError('Probe and validation class order differ')
    cat_to_label = {c['id']: i for i, c in enumerate(categories)}
    by_image = {}
    for ann in annotation['annotations']:
        if not ann.get('iscrowd', 0) and ann['bbox'][2] > 1 and ann['bbox'][3] > 1:
            by_image.setdefault(ann['image_id'], []).append(ann)
    model, epoch, source = build_detector(config, device)
    # The original probe artifact predates provenance fields in its .pth file.
    # Verify its adjacent extraction report when present instead of assuming
    # any foreground classifier belongs to the selected detector checkpoint.
    provenance_path = Path(config['probe_checkpoint']).parent.parent / 'extraction_report.json'
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
        for key, expected in [('epoch', epoch), ('weight_source', source),
                              ('input_size', config['input_size'])]:
            if provenance.get(key) != expected:
                raise ValueError(f'Probe provenance mismatch for {key}')
        if provenance.get('checkpoint') != config['checkpoint']:
            raise ValueError('Probe was extracted from a different checkpoint path')
    else:
        print('Warning: probe extraction report missing; checkpoint provenance unverified.', flush=True)
    capture = {}
    def hook(module, args, value):
        capture['query'] = value
    handle = model.decoder.decoder.layers[model.decoder.eval_idx].register_forward_hook(hook)
    mean, std = probe['mean'].to(device), probe['std'].to(device)
    weight, bias = probe['head']['weight'].to(device), probe['head']['bias'].to(device)
    height, width = config['input_size']
    threshold, confidence = config['matching_iou'], config['confidence_threshold']
    neighbor_threshold = config['neighbor_iou_threshold']
    raw_dir = output_dir / 'raw_queries'
    raw_dir.mkdir(parents=True, exist_ok=True)
    paired, score_paired, counts = [], [], Counter()
    query_fields = ['image_id', 'file_name', 'query_index', 'original_class',
                    'original_score', 'probe_class', 'probe_softmax', 'max_gt_iou',
                    'nearest_gt_id', 'nearest_gt_class', 'score_match_status',
                    'score_match_gt_id', 'iou_only_gt_id', 'high_score_neighbor_count',
                    'x1', 'y1', 'x2', 'y2']
    query_fields += [f'{kind}_{name}' for kind in ['original_logit', 'original_sigmoid',
                                                'probe_logit', 'probe_softmax'] for name in names]
    gt_fields = ['image_id', 'file_name', 'annotation_id', 'gt_class', 'query_index',
                 'query_iou', 'original_class', 'original_score', 'probe_class',
                 'comparison', 'score_match_status', 'high_score_neighbor_count']
    try:
        with (output_dir / 'all_queries.csv').open('w', newline='', encoding='utf-8-sig') as qfile, \
                (output_dir / 'iou_selected_gt_comparison.csv').open('w', newline='', encoding='utf-8-sig') as gfile:
            qw, gw = csv.DictWriter(qfile, query_fields), csv.DictWriter(gfile, gt_fields)
            qw.writeheader()
            gw.writeheader()
            for index, info in enumerate(sorted(annotation['images'], key=lambda i: i['id'])):
                anns = by_image.get(info['id'], [])
                with Image.open(Path(config['data']['val']['image_dir']) / info['file_name']) as image:
                    rgb = image.convert('RGB')
                    ow, oh = rgb.size
                    tensor = TF.to_tensor(TF.resize(rgb, [height, width],
                        interpolation=InterpolationMode.BILINEAR, antialias=True))[None].to(device)
                capture.clear()
                outputs = model(tensor)
                query = capture['query'][0]
                if query.shape[-1] != weight.shape[-1]:
                    raise ValueError('Query channel dimension differs from probe')
                logits = outputs['pred_logits'][0].float().cpu()
                boxes = outputs['pred_boxes'][0].float().cpu()
                local_logits = (((query.float() - mean) / std) @ weight.T + bias).cpu()
                original_prob, local_prob = logits.sigmoid(), local_logits.softmax(-1)
                scores, old = original_prob.max(-1)
                new = local_prob.argmax(-1)
                gt_boxes = torch.tensor([[a['bbox'][0]/ow, a['bbox'][1]/oh,
                    (a['bbox'][0]+a['bbox'][2])/ow, (a['bbox'][1]+a['bbox'][3])/oh]
                    for a in anns], dtype=torch.float32).reshape(-1, 4).clamp(0, 1)
                gt_labels = torch.tensor([cat_to_label[a['category_id']] for a in anns], dtype=torch.long)
                pred_xy = xyxy(boxes)
                ious = box_iou(pred_xy, gt_boxes)
                selected, overlaps = match_queries(pred_xy, gt_boxes, threshold)
                assignment, status = score_match(ious, scores, threshold, confidence)
                for q in torch.where(assignment >= 0)[0].tolist():
                    g = int(assignment[q])
                    score_paired.append((int(gt_labels[g]), int(old[q]), int(new[q])))
                counts['gt_without_score_first_match'] += len(anns) - int((assignment >= 0).sum())
                neighbors = box_iou(pred_xy, pred_xy) >= neighbor_threshold
                neighbors.fill_diagonal_(False)
                neighbor_counts = (neighbors & (scores >= confidence)[None]).sum(-1)
                inverse = {int(q): g for g, q in enumerate(selected.tolist()) if q >= 0}
                for q in range(len(scores)):
                    nearest = int(ious[q].argmax()) if len(anns) else -1
                    overlap = float(ious[q, nearest]) if nearest >= 0 else 0.0
                    g = int(assignment[q])
                    diagnostic = status[q]
                    if g >= 0:
                        diagnostic = 'correct_class' if old[q] == gt_labels[g] else 'misclassification'
                    counts[diagnostic] += 1
                    row = dict(zip(query_fields[:18], [info['id'], info['file_name'], q,
                        names[int(old[q])], float(scores[q]), names[int(new[q])],
                        float(local_prob[q].max()), overlap,
                        anns[nearest]['id'] if nearest >= 0 else '',
                        names[int(gt_labels[nearest])] if nearest >= 0 else '', diagnostic,
                        anns[g]['id'] if g >= 0 else '',
                        anns[inverse[q]]['id'] if q in inverse else '', int(neighbor_counts[q]),
                        *[float(v) for v in (pred_xy[q] * torch.tensor([ow, oh, ow, oh]))]]))
                    for kind, values in [('original_logit', logits), ('original_sigmoid', original_prob),
                                         ('probe_logit', local_logits), ('probe_softmax', local_prob)]:
                        row.update({f'{kind}_{name}': float(values[q, c]) for c, name in enumerate(names)})
                    qw.writerow(row)
                for g, ann in enumerate(anns):
                    q = int(selected[g])
                    if q < 0:
                        gw.writerow({'image_id': info['id'], 'file_name': info['file_name'],
                            'annotation_id': ann['id'], 'gt_class': names[int(gt_labels[g])],
                            'query_index': -1, 'comparison': 'no_iou_match'})
                        counts['gt_no_iou_match'] += 1
                        continue
                    a, b = bool(old[q] == gt_labels[g]), bool(new[q] == gt_labels[g])
                    label = 'both_correct' if a and b else 'fixed' if b else 'broken' if a else 'both_wrong'
                    gw.writerow(dict(zip(gt_fields, [info['id'], info['file_name'], ann['id'],
                        names[int(gt_labels[g])], q, float(overlaps[g]), names[int(old[q])],
                        float(scores[q]), names[int(new[q])], label, status[q], int(neighbor_counts[q])])))
                    paired.append((int(gt_labels[g]), int(old[q]), int(new[q]),
                                   float(scores[q]), int(assignment[q]) == g))
                # Raw per-image data allow later matching/ranking checks without new inference.
                torch.save({'image_id': info['id'], 'file_name': info['file_name'],
                    'original_size_hw': [oh, ow], 'class_names': names,
                    'gt_annotation_ids': [a['id'] for a in anns], 'gt_labels': gt_labels,
                    'gt_boxes_xyxy_normalized': gt_boxes, 'pred_logits': logits,
                    'pred_boxes_cxcywh_normalized': boxes, 'decoder_queries': query.float().cpu(),
                    'probe_logits': local_logits, 'iou_only_gt_to_query': selected,
                    'score_first_query_to_gt': assignment}, raw_dir / f"{info['id']}.pth")
                counts['images'] += 1
                counts['queries'] += len(scores)
                counts['gt'] += len(anns)
                if (index + 1) % 25 == 0:
                    print(f"val: {index+1}/{len(annotation['images'])} images", flush=True)
    finally:
        handle.remove()
    if not paired:
        raise ValueError('No IoU-matched queries found')
    data = torch.tensor(paired)
    y, old, new = (data[:, i].long() for i in range(3))
    high = data[:, 3] >= confidence
    masks = {'all_iou_selected': torch.ones(len(y), dtype=torch.bool),
             'iou_selected_original_score_high': high,
             'iou_selected_original_score_low': ~high,
             'iou_selected_also_score_first_same_gt': data[:, 4].bool()}
    report = {'checkpoint': config['checkpoint'], 'epoch': epoch, 'weight_source': source,
              'probe_checkpoint': config['probe_checkpoint'], 'config': config,
              'probe_provenance_verified': provenance_path.is_file(),
              'counts': dict(counts), 'comparisons': {},
              'limitations': ['Diagnostic top-1 query matching, not official COCO evaluation.',
                 'Probe softmax is not object confidence; no scores are replaced.',
                 'GT scope and crowd filtering follow stage_feature_probe; labels are unchanged.',
                 'Duplicate/overlap status is geometric evidence, not human-confirmed error.']}
    for name, mask in masks.items():
        report['comparisons'][name] = comparison_stats(y, old, new, mask)
        report['comparisons'][name]['per_class'] = {
            label: comparison_stats(y, old, new, mask & (y == c)) for c, label in enumerate(names)}
    score_data = torch.tensor(score_paired, dtype=torch.long).reshape(-1, 3)
    sy, so, sn = score_data.unbind(1)
    report['comparisons']['score_first_high_confidence_matches'] = comparison_stats(
        sy, so, sn, torch.ones(len(sy), dtype=torch.bool))
    report['comparisons']['score_first_high_confidence_matches']['per_class'] = {
        label: comparison_stats(sy, so, sn, sy == c) for c, label in enumerate(names)}
    (output_dir / 'audit_summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: {a: b for a, b in v.items() if a != 'per_class'}
                      for k, v in report['comparisons'].items()}, indent=2), flush=True)
    print(f'Finished. Diagnostic files: {output_dir}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ['checkpoint', 'probe_checkpoint']:
        if not Path(config[key]).is_file():
            raise FileNotFoundError(f"Missing {key}: {config[key]}")
    if not 0 <= config['confidence_threshold'] <= 1 or not 0 < config['matching_iou'] <= 1:
        raise ValueError('Invalid confidence/IoU threshold')
    seed_everything(config['seed'])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'resolved_config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False))
    run(config, output_dir, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


if __name__ == '__main__':
    main()
