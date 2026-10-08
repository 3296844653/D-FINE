"""Build an offline, image-embedded review page from existing SCB-S diagnostics.

No inference, training, rescoring, or annotation modification is performed.
Only load trusted project-generated .pt exports.
"""
import argparse
import base64
import json
import math
import mimetypes
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import box_iou


CLASS_NAMES = {0: 'hand-raising', 1: 'read', 2: 'write'}
GROUPS = [
    ('B', 'B：类别正确但低分', lambda r: r['type'] == 'B'),
    ('D-no-correct', 'D：所有重叠 query 的最高分类都错误',
     lambda r: r['type'] == 'D' and not r['any_correct_top1_overlapping_query_exists']),
    ('D-low-correct', 'D：仍有低分正确类别 query',
     lambda r: r['type'] == 'D' and r['another_low_correct_query_exists']),
]


def read_records(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    mapped = {r['annotation_id']: r for r in rows}
    if len(mapped) != len(rows):
        raise ValueError('Duplicate annotation IDs in cross-classification records')
    return mapped


def select_cases(records, count, seed):
    rng, selected, group_data = random.Random(seed), [], []
    for key, title, predicate in GROUPS:
        pool = sorted([r for r in records.values() if r['gt_class_id'] == 2 and predicate(r)],
                      key=lambda r: r['annotation_id'])
        if len(pool) < count:
            raise ValueError(f'{key} has only {len(pool)} candidates; requested {count}')
        rng.shuffle(pool)
        picked, used_images = [], set()
        for row in pool:
            if row['image_id'] not in used_images:
                picked.append(row)
                used_images.add(row['image_id'])
                if len(picked) == count:
                    break
        for row in pool:
            if len(picked) == count:
                break
            if row['annotation_id'] not in {r['annotation_id'] for r in picked}:
                picked.append(row)
        selected.extend((key, title, r) for r in picked)
        group_data.append({'id': key, 'title': title, 'population': len(pool),
                           'sample_count': len(picked), 'image_count': len(used_images)})
    assert len({r['annotation_id'] for _, _, r in selected}) == len(selected)
    return selected, group_data


def crop_box(box, width, height):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    desired_h = max(bh * 1.65, bw * 2.0 / 1.2)
    desired_w = desired_h * 1.2
    scale = min(1.0, width / desired_w, height / desired_h)
    cw, ch = desired_w * scale, desired_h * scale
    left = min(max((x1 + x2) / 2 - cw / 2, 0), width - cw)
    top = min(max((y1 + y2) / 2 - ch / 2, 0), height - ch)
    return [math.floor(left), math.floor(top), math.ceil(left + cw), math.ceil(top + ch)]


def model_details(folder, record, cache):
    image_id = record['image_id']
    path = folder / 'query_diagnostics/raw_queries' / f'{image_id}.pt'
    if path not in cache:
        cache[path] = torch.load(path, map_location='cpu', weights_only=False)
    raw = cache[path]
    scores, boxes = raw['pred_scores'].float(), raw['pred_boxes_xyxy'].float()
    g, cls = record['gt_index'], record['gt_class_id']
    gt = raw['gt_boxes_xyxy'][g].float()
    assert int(raw['gt_labels'][g]) == cls
    assert torch.allclose(gt, torch.tensor(record['box_xyxy']), rtol=0, atol=1e-6)
    overlaps = box_iou(boxes, gt.reshape(1, 4))[:, 0]
    best = int(overlaps.argmax())
    assert best == record['best_query_id']
    assert abs(float(overlaps[best]) - record['best_query_IoU']) < 1e-6
    assert torch.allclose(scores[best], torch.tensor(record['best_query_all_class_scores']), rtol=0, atol=1e-6)
    top1 = scores.argmax(1)
    assert CLASS_NAMES[int(top1[best])] == record['best_query_top_class']
    confidence = record['confidence_threshold']
    flat, selected_scores = raw['selected_flat_indices'].long(), raw['selected_scores'].float()
    assert torch.allclose(scores.flatten()[flat], selected_scores, rtol=0, atol=1e-6)
    selected_pairs = torch.zeros_like(scores, dtype=torch.bool)
    selected_pairs.flatten()[flat] = True
    another_high = ((overlaps >= .5) & (top1 == cls) & (scores[:, cls] >= confidence)
                    & selected_pairs[:, cls] & (torch.arange(len(scores)) != best))
    assert bool(another_high.any()) == record['another_high_correct_query_exists']
    predictions = []
    for rank, (index, score) in enumerate(zip(flat.tolist(), selected_scores.tolist())):
        if score < confidence:
            continue
        q, label = divmod(index, scores.shape[1])
        predictions.append({'id': f'P{rank + 1}', 'rank': rank, 'query_id': q,
                            'class_id': label, 'class_name': CLASS_NAMES[label],
                            'query_top_class': CLASS_NAMES[int(top1[q])],
                            'score': score, 'iou': float(overlaps[q]), 'box': boxes[q].tolist()})
    low_correct = [q for q in range(len(scores)) if q != best and int(top1[q]) == cls
                   and float(overlaps[q]) >= .5 and float(scores[q, cls]) < confidence]
    low_correct.sort(key=lambda q: (-float(scores[q, cls]), -float(overlaps[q]), q))
    assert bool(low_correct) == record['another_low_correct_query_exists']
    assert any(p['class_id'] == cls and p['iou'] >= .5 for p in predictions) == record['any_high_GT_class_prediction_pair_exists']
    def query_data(q):
        return {'query_id': q, 'box': boxes[q].tolist(), 'class_id': int(top1[q]),
                'class_name': CLASS_NAMES[int(top1[q])], 'top_score': float(scores[q].max()),
                'gt_class_score': float(scores[q, cls]), 'scores': scores[q].tolist(),
                'iou': float(overlaps[q])}
    return {
        'type': record['type'], 'best': query_data(best),
        'low_correct': query_data(low_correct[0]) if low_correct else None,
        'another_high_correct_query_exists': record['another_high_correct_query_exists'],
        'high_GT_class_pair_exists': record['any_high_GT_class_prediction_pair_exists'],
        'predictions': predictions,
        'high_predictions_count': len(predictions),
        'nearby_high_predictions_count': sum(p['iou'] >= .1 for p in predictions),
        'score_first_diagnostic_outcome': record['score_first_outcome'],
    }, raw


def build(args):
    if args.output_dir.exists():
        raise FileExistsError('Choose a new output directory; existing reports are preserved.')
    baseline = read_records(args.cross_dir / 'run1_diagnostics_eval/gt_cross_classification.jsonl')
    experiment = read_records(args.cross_dir / 'rw_pairwise-ce-diagnostics_eval/gt_cross_classification.jsonl')
    if set(baseline) != set(experiment):
        raise ValueError('Different GT IDs between the two runs')
    original = json.loads((args.dataset_dir / 'annotations/instances_val.json').read_text())
    categories = {c['id']: c['name'] for c in original['categories']}
    if categories != CLASS_NAMES:
        raise ValueError('Local dataset category names/order do not match the exports')
    if len(original['images']) != 1026 or len(original['annotations']) != len(baseline):
        raise ValueError('Local validation dataset counts differ from the exports')
    image_info = {i['id']: i for i in original['images']}
    annotations = {a['id']: a for a in original['annotations']}
    source_meta = [json.loads((p / 'query_diagnostics/metadata.json').read_text())
                   for p in (args.baseline_dir, args.experiment_dir)]
    if not all(m['complete'] and m['schema_version'] == 2 for m in source_meta):
        raise ValueError('Requires complete schema-2 query exports')
    for name, meta in zip(['run1_diagnostics_eval', 'rw_pairwise-ce-diagnostics_eval'], source_meta):
        cross_summary = json.loads((args.cross_dir / name / 'summary.json').read_text())
        assert cross_summary['checkpoint_sha256'] == meta['checkpoint_sha256'], 'Different checkpoint from cross-classification'
    if source_meta[0]['config']['val_dataloader'] != source_meta[1]['config']['val_dataloader']:
        raise ValueError('Validation configurations differ')
    selected, groups = select_cases(baseline, args.per_group, args.seed)
    images, cases, cache = {}, [], {}
    for number, (group, group_title, first) in enumerate(selected, 1):
        second = experiment[first['annotation_id']]
        for field in ['image_id', 'image_name', 'gt_index', 'gt_class_id', 'box_xyxy', 'area_original_annotation']:
            assert first[field] == second[field], (first['annotation_id'], field)
        ann, info = annotations[first['annotation_id']], image_info[first['image_id']]
        assert ann['image_id'] == first['image_id'] and ann['category_id'] == first['gt_class_id']
        assert info['file_name'] == first['image_name']
        assert abs(ann['area'] - first['area_original_annotation']) < .01
        x, y, bw, bh = ann['bbox']
        expected = torch.tensor([max(0, x), max(0, y), min(info['width'], x + bw), min(info['height'], y + bh)])
        assert torch.allclose(expected, torch.tensor(first['box_xyxy']), rtol=1e-5, atol=.05)
        models, raws = [], []
        for folder, record in [(args.baseline_dir, first), (args.experiment_dir, second)]:
            detail, raw = model_details(folder, record, cache)
            models.append(detail); raws.append(raw)
        assert all(tuple(raw['original_size_wh']) == (info['width'], info['height']) for raw in raws)
        key = str(first['image_id'])
        if key not in images:
            picture = (args.dataset_dir / 'images/val' / first['image_name']).resolve()
            assert picture.is_relative_to((args.dataset_dir / 'images/val').resolve())
            with Image.open(picture) as opened:
                if opened.size != (info['width'], info['height']):
                    raise ValueError(f'Wrong image dimensions: {picture}')
            mime = mimetypes.guess_type(picture.name)[0] or 'image/jpeg'
            images[key] = {'width': info['width'], 'height': info['height'],
                           'src': f'data:{mime};base64,' + base64.b64encode(picture.read_bytes()).decode('ascii'),
                           'all_gt': [{'box': a['box_xyxy'], 'annotation_id': a['annotation_id'], 'class_name': a['gt_class']}
                                      for a in baseline.values() if a['image_id'] == first['image_id']]}
        cases.append({'number': number, 'id': str(first['annotation_id']), 'group': group,
                      'group_title': group_title, 'image_id': key, 'image_name': first['image_name'],
                      'annotation_id': first['annotation_id'], 'gt_index': first['gt_index'],
                      'gt_class_id': first['gt_class_id'], 'gt_class': first['gt_class'],
                      'gt_box': first['box_xyxy'], 'area': first['area_original_annotation'],
                      'crop': crop_box(first['box_xyxy'], info['width'], info['height']), 'models': models})
    source = {'baseline_checkpoint': source_meta[0]['checkpoint'],
              'experiment_checkpoint': source_meta[1]['checkpoint'],
              'baseline_sha256': source_meta[0]['checkpoint_sha256'],
              'experiment_sha256': source_meta[1]['checkpoint_sha256'], 'seed': args.seed,
              'sampling': 'Baseline-stratified; shuffle GTs, prefer different images per group; exploratory, not an unbiased accuracy estimate',
              'high_IoU': .75, 'confidence': .5, 'correct_candidate_IoU': .5,
              'nearby_prediction_display_IoU': .1}
    payload = {'source': source, 'groups': groups, 'images': images, 'cases': cases}
    template = Path(__file__).with_name('templates') / 'scbs_best_query_audit.html'
    text = template.read_text(encoding='utf-8')
    if text.count('__AUDIT_DATA__') != 1:
        raise ValueError('Template must have exactly one dataset placeholder')
    embedded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    embedded = embedded.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    html = text.replace('__AUDIT_DATA__', embedded)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'index.html').write_text(html, encoding='utf-8')
    manifest = {'source': source, 'groups': groups, 'cases': [{k: v for k, v in c.items() if k != 'models'} for c in cases]}
    (args.output_dir / 'sample_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Created {len(cases)} GT cases from {len(images)} original images; all input checks passed.')
    print('Web page:', args.output_dir / 'index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['cross-dir', 'baseline-dir', 'experiment-dir', 'dataset-dir', 'output-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--per-group', type=int, default=10)
    args = parser.parse_args()
    if args.per_group < 1:
        parser.error('--per-group must be positive')
    torch.set_num_threads(2)
    build(args)
