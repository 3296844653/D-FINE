"""Frozen SCB-S baseline probes; conditional classification, NEVER detector AP.

Two independently fixed GT/query pairings are retained: IoU-only Hungarian
oracle candidates, and actual flattened query-class top-k predictions matched
by the existing Validator's class-blind highest-IoU greedy rule. Only small
foreground classifiers are trained. Detector scores/boxes/weights never change.
"""
import argparse
import copy
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.ops import box_iou, roi_align
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.core import YAMLConfig
from src.core.yaml_utils import load_config as load_detector_config
from tools.diagnostics.gt_box_classifier import load_config, seed_everything, confusion_metrics
from tools.diagnostics.stage_feature_probe import match_queries

SCHEMA = 1
MAP_STAGES = ('backbone_p2', 'backbone_p3', 'backbone_p4', 'encoder_p3', 'encoder_p4')
PROBES = (*MAP_STAGES, 'decoder_query', 'query_only_capacity_control',
          'query_plus_pred_roi', 'query_plus_gt_roi')
CATEGORIES = [(0, 'hand-raising'), (1, 'read'), (2, 'write')]
LIMITATIONS = [
    'All probe metrics condition on matched foreground GT; not detector AP, precision or recall.',
    'IoU-only matches are oracle candidates; low-score gains do not prove detection gains.',
    'Selected matching uses actual flattened query-class top-k, not one top-1 per query.',
    'GT ROI is an oracle geometry control; predicted ROI is the deployable geometry comparison.',
    'Probe softmax is foreground-conditional, not object confidence; never replaces detector scores.',
    'Repeated-query control matches input size and parameter count, not effective feature rank or optimization dynamics.',
    'Representation, pooling, dimensions and probe optimization can affect separability; not a causal layer proof.',
    'Unmatched predictions are not proven background; annotations/teacher/scope are not edited or excluded.',
    'Extraction is FP32 eval, no augmentation, one image at a time; GPU/batch IoU ties may differ from an old export.',
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def validate_config(config):
    if len(config['input_size']) != 2 or any(type(v) is not int or v <= 0 or v % 32
                                           for v in config['input_size']):
        raise ValueError('input_size must be positive [height,width], multiples of 32')
    for field in ('matching_iou', 'confidence_threshold'):
        if not math.isfinite(config[field]) or not 0 < config[field] <= 1:
            raise ValueError(f'{field} must be in (0,1]')
    probe = config['probe']
    seeds = probe['seeds']
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('probe.seeds must contain unique nonnegative integers')
    for field in ('epochs', 'batch_size'):
        if type(probe[field]) is not int or probe[field] < 1:
            raise ValueError(f'probe.{field} must be positive')
    for field in ('learning_rate', 'weight_decay'):
        if not math.isfinite(probe[field]) or probe[field] < 0 or (field == 'learning_rate' and probe[field] == 0):
            raise ValueError(f'Invalid probe.{field}')
    for field in ('progress_every', 'num_threads', 'roi_size'):
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError(f'{field} must be positive')


def build_frozen_detector(config, device):
    # Fresh dictionary avoids the legacy YAML loader's mutable-default state.
    cfg = YAMLConfig(config['detector_config'])
    cfg.yaml_cfg = load_detector_config(config['detector_config'], cfg={})
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    cfg.yaml_cfg['eval_spatial_size'] = list(config['input_size'])
    if cfg.yaml_cfg['num_classes'] != 3 or cfg.yaml_cfg.get('remap_mscoco_category', False):
        raise ValueError('Use the SCB-S three-class, non-remapped baseline config')
    if cfg.yaml_cfg['HGNetv2']['name'] != 'B0' or cfg.yaml_cfg['DFINETransformer']['num_layers'] != 3:
        raise ValueError('This configuration expects the validated HGNetv2-B0 / D-FINE-S baseline')
    for split in ('train', 'val'):
        declared = cfg.yaml_cfg[f'{split}_dataloader']['dataset']
        for diagnostic_key, detector_key in (('image_dir', 'img_folder'), ('annotation', 'ann_file')):
            if Path(config['data'][split][diagnostic_key]).resolve() != Path(declared[detector_key]).resolve():
                raise ValueError(f'{split}: probe data paths differ from detector baseline config')
    model = cfg.model
    for component in (model.backbone, model.encoder, model.decoder):
        # HGNetv2's LearnableAffineBlock is part of the official B0 baseline,
        # not an experimental module. Its normal `use_lab: true` must be kept.
        enabled = [name for name, value in vars(component).items()
                   if name.startswith('use_') and name != 'use_lab'
                   and isinstance(value, bool) and value]
        if enabled:
            raise ValueError(f'Baseline only: experimental flags enabled: {enabled}')
    checkpoint = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
    source = 'ema' if 'ema' in checkpoint else 'model'
    state = checkpoint['ema']['module'] if source == 'ema' else checkpoint['model']
    model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, cfg.postprocessor.to(device).eval(), checkpoint.get('last_epoch'), source


def selected_pairs(outputs, postprocessor, original_size, confidence):
    """Recover query IDs for the actual unchanged focal top-k postprocessor."""
    if not postprocessor.use_focal_loss or postprocessor.remap_mscoco_category:
        raise ValueError('This audit requires the original focal, non-remapped postprocessor')
    results = postprocessor(outputs, original_size)[0]
    logits = outputs['pred_logits'][0]
    scores, flat = logits.sigmoid().flatten().topk(postprocessor.num_top_queries)
    qids, labels = flat // logits.shape[-1], flat % logits.shape[-1]
    if not torch.equal(labels, results['labels']) or not torch.equal(scores, results['scores']):
        raise ValueError('Query-class ID reconstruction differs from the actual postprocessor')
    keep = scores >= confidence
    return {key: value[keep] for key, value in dict(
        query_ids=qids, labels=labels, scores=scores, boxes=results['boxes']).items()}


def selected_assignment(pairs, gt_boxes, gt_labels, threshold, classes=3):
    """Same on-device sorting rule as Validator; predictions are query-class pairs.

    A query may appear more than once with distinct classes. Do not deduplicate
    it: original postprocessing/Validator permit those separate predictions.
    """
    n, m = len(gt_boxes), len(pairs['boxes'])
    assignment = torch.full((n,), -1, dtype=torch.long)
    overlaps = torch.zeros(n)
    matrix = torch.zeros(classes + 1, classes + 1, dtype=torch.long)
    used_predictions, used_gt = set(), set()
    if n and m:
        ious = box_iou(pairs['boxes'], gt_boxes)
        pred, gt = torch.nonzero(ious >= threshold, as_tuple=True)
        order = torch.argsort(-ious[pred, gt])
        for p, g in zip(pred[order].tolist(), gt[order].tolist()):
            if p in used_predictions or g in used_gt:
                continue
            used_predictions.add(p); used_gt.add(g)
            assignment[g] = p; overlaps[g] = float(ious[p, g])
            matrix[int(gt_labels[g]), int(pairs['labels'][p])] += 1
    for p in set(range(m)) - used_predictions:
        matrix[classes, int(pairs['labels'][p])] += 1
    for g in set(range(n)) - used_gt:
        matrix[int(gt_labels[g]), classes] += 1
    return assignment, overlaps, matrix


def pool_roi(feature, normalized_xyxy, side):
    if not len(normalized_xyxy):
        return torch.empty(0, feature.shape[1], device=feature.device)
    scale = normalized_xyxy.new_tensor([feature.shape[-1], feature.shape[-2]] * 2)
    pooled = roi_align(feature.float(), [normalized_xyxy.float() * scale],
                       output_size=side, spatial_scale=1, sampling_ratio=2, aligned=True)
    return pooled.mean((-1, -2))


def annotations_for_image(anns, image_id, original_size):
    """Mirror the baseline dataset's noncrowd/clipped positive-area filtering."""
    width, height = original_size
    kept, boxes = [], []
    for ann in anns:
        if ann.get('iscrowd', 0):
            continue
        if ann['category_id'] not in range(3):
            raise ValueError(f'Unknown category in image {image_id}')
        x, y, w, h = ann['bbox']
        box = [max(0, min(width, x)), max(0, min(height, y)),
               max(0, min(width, x + w)), max(0, min(height, y + h))]
        if box[2] > box[0] and box[3] > box[1]:
            kept.append(ann); boxes.append(box)
    return kept, torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)


def group_scopes(cache, group):
    current = cache['groups'][group]
    valid = current['valid']
    scopes = {'all_matched': valid}
    if group == 'selected':
        scopes.update(original_wrong=valid & (current['original_labels'] != cache['labels']),
                      original_correct=valid & (current['original_labels'] == cache['labels']),
                      read_write_confusion=valid & (
                          ((cache['labels'] == 1) & (current['original_labels'] == 2)) |
                          ((cache['labels'] == 2) & (current['original_labels'] == 1))))
    else:
        low = valid & (current['scores'] < cache['metadata']['confidence_threshold'])
        selected = cache['groups']['selected']
        correct_cover = selected['valid'] & (selected['original_labels'] == cache['labels'])
        scopes.update(high_score=valid & ~low, low_score=low,
                      low_with_selected_correct=low & correct_cover,
                      low_with_selected_wrong=low & selected['valid'] & ~correct_cover,
                      low_without_selected_match=low & ~selected['valid'],
                      same_query_as_selected=valid & selected['valid'] &
                      (current['query_ids'] == selected['query_ids']))
    for category_id, class_name in CATEGORIES:
        scopes[f'gt_{class_name}'] = valid & (cache['labels'] == category_id)
    area = cache['areas']
    # COCO area boundaries are inclusive, not disjoint at 32^2/96^2.
    scopes['medium_gt'] = valid & (area >= 32**2) & (area <= 96**2)
    return scopes


@torch.no_grad()
def extract(config, output, device):
    model, postprocessor, epoch, weight_source = build_frozen_detector(config, device)
    captures, hooks = {}, []
    def capture(name):
        def hook(module, args, value): captures[name] = value
        return hook
    for index, name in enumerate(MAP_STAGES[:3]):
        hooks.append(model.backbone.stages[index].register_forward_hook(capture(name)))
    hooks.append(model.encoder.register_forward_hook(capture('encoder')))
    hooks.append(model.decoder.decoder.layers[model.decoder.eval_idx].register_forward_hook(
        capture('query')))
    provenance = dict(schema_version=SCHEMA, config_sha256=config_digest(config),
                      checkpoint=str(config['checkpoint']), checkpoint_sha256=digest(config['checkpoint']),
                      epoch=epoch, weight_source=weight_source, input_size=config['input_size'],
                      confidence_threshold=config['confidence_threshold'], matching_iou=config['matching_iou'],
                      postprocessor_top_k=postprocessor.num_top_queries,
                      category_ids=[0, 1, 2], class_names=[c[1] for c in CATEGORIES],
                      matching_selected='class-blind greedy highest-IoU one-to-one on original-device query-class pairs',
                      matching_iou_only='class-blind Hungarian, maximize IoU-threshold matches then IoU; no score filter',
                      complete=False, limitations=LIMITATIONS)
    report = copy.deepcopy(provenance)
    height, width = config['input_size']
    try:
        for split in ('train', 'val'):
            data = config['data'][split]
            annotation = json.loads(Path(data['annotation']).read_text())
            categories = sorted([(c['id'], c['name']) for c in annotation['categories']])
            if categories != CATEGORIES:
                raise ValueError(f'{split}: expected official SCB-S categories {CATEGORIES}, got {categories}')
            by_image = {}
            for ann in annotation['annotations']:
                by_image.setdefault(ann['image_id'], []).append(ann)
            maps = {stage: [] for stage in MAP_STAGES}
            groups = {g: {key: [] for key in ('query', 'pred_p3', 'pred_p4', 'valid',
                                             'query_ids', 'original_labels', 'scores', 'ious')}
                      for g in ('iou', 'selected')}
            labels, areas, records = [], [], []
            matrix = torch.zeros(4, 4, dtype=torch.long)
            total_queries = 0
            print(f'{split}: extracting frozen FP32 features from {len(annotation["images"])} images', flush=True)
            for index, info in enumerate(sorted(annotation['images'], key=lambda row: row['id'])):
                image_path = Path(data['image_dir']) / info['file_name']
                with Image.open(image_path) as image:
                    rgb = image.convert('RGB'); original_size = rgb.size
                    image_tensor = TF.to_tensor(TF.resize(rgb, [height, width],
                        interpolation=InterpolationMode.BILINEAR, antialias=True)).unsqueeze(0).to(device)
                anns, gt_original = annotations_for_image(by_image.get(info['id'], []), info['id'], original_size)
                gt_labels = torch.tensor([a['category_id'] for a in anns], dtype=torch.long, device=device)
                gt = gt_original.to(device) / gt_original.new_tensor(list(original_size) * 2).to(device)
                # Match the evaluator's resize-and-scale-back floating-point path.
                gt_for_matching = gt_original.to(device) * gt.new_tensor(
                    [width / original_size[0], height / original_size[1]] * 2)
                gt_for_matching *= gt.new_tensor([original_size[0] / width, original_size[1] / height] * 2)
                captures.clear(); outputs = model(image_tensor)
                if not all(torch.isfinite(outputs[k]).all() for k in ('pred_logits', 'pred_boxes')):
                    raise ValueError(f'Non-finite detector output: {image_path}')
                raw_boxes = outputs['pred_boxes'][0].float()
                normalized_boxes = torch.cat((raw_boxes[:, :2] - raw_boxes[:, 2:] / 2,
                                              raw_boxes[:, :2] + raw_boxes[:, 2:] / 2), -1)
                scores, top1 = outputs['pred_logits'][0].sigmoid().max(-1)
                pairs = selected_pairs(outputs, postprocessor,
                    torch.tensor([original_size], device=device), config['confidence_threshold'])
                assigned, assigned_iou, current_matrix = selected_assignment(
                    pairs, gt_for_matching, gt_labels, config['matching_iou'])
                matrix += current_matrix
                iou_query, oracle_iou = match_queries(normalized_boxes, gt, config['matching_iou'])
                selected_query = torch.full((len(anns),), -1, dtype=torch.long)
                selected_label = torch.full_like(selected_query, -1)
                selected_score = torch.zeros(len(anns))
                found = assigned >= 0
                if found.any():
                    chosen = assigned[found].to(device)
                    selected_query[found] = pairs['query_ids'][chosen].cpu()
                    selected_label[found] = pairs['labels'][chosen].cpu()
                    selected_score[found] = pairs['scores'][chosen].cpu()
                encoder = captures['encoder']
                for stage in MAP_STAGES:
                    fmap = captures[stage] if stage.startswith('backbone') else encoder[0 if stage.endswith('p3') else 1]
                    maps[stage].append(pool_roi(fmap, gt, config['roi_size']).cpu())
                all_local = [pool_roi(encoder[level], normalized_boxes, config['roi_size']) for level in (0, 1)]
                query = captures['query'][0].float()
                for group, ids, overlaps in (('iou', iou_query, oracle_iou),
                                             ('selected', selected_query, assigned_iou)):
                    valid = ids >= 0; chosen = ids[valid].to(device)
                    fields = dict(valid=valid, query_ids=ids, ious=overlaps)
                    for key, value in (('query', query), ('pred_p3', all_local[0]), ('pred_p4', all_local[1])):
                        saved = torch.zeros(len(anns), value.shape[-1])
                        saved[valid] = value[chosen].cpu(); fields[key] = saved
                    old = torch.full((len(anns),), -1, dtype=torch.long)
                    conf = torch.zeros(len(anns))
                    if group == 'selected': old, conf = selected_label, selected_score
                    elif valid.any(): old[valid], conf[valid] = top1[chosen].cpu(), scores[chosen].cpu()
                    fields.update(original_labels=old, scores=conf)
                    for key, value in fields.items(): groups[group][key].append(value)
                labels.append(gt_labels.cpu())
                areas.append(torch.tensor([a.get('area', a['bbox'][2] * a['bbox'][3]) for a in anns]))
                records.extend([dict(image_id=info['id'], image_name=info['file_name'],
                                     annotation_id=a['id'], gt_class=CATEGORIES[a['category_id']][1]) for a in anns])
                total_queries += len(raw_boxes)
                if (index + 1) % config['progress_every'] == 0 or index + 1 == len(annotation['images']):
                    print(f'{split}: {index + 1}/{len(annotation["images"])} images; frozen weights unchanged', flush=True)
            if not sum(len(y) for y in labels):
                raise ValueError(f'{split}: no valid GT')
            cache = dict(metadata=copy.deepcopy(provenance), split=split,
                         annotation_sha256=digest(data['annotation']),
                         class_names=[c[1] for c in CATEGORIES], records=records,
                         labels=torch.cat(labels), areas=torch.cat(areas),
                         maps={k: torch.cat(v) for k, v in maps.items()},
                         groups={g: {k: torch.cat(v) for k, v in fields.items()} for g, fields in groups.items()},
                         original_selected_matrix=matrix)
            cache['metadata']['complete'] = True
            torch.save(cache, output / f'{split}_features.pth')
            report[split] = dict(images=len(annotation['images']), gt=len(cache['labels']), queries=total_queries,
                                annotation_sha256=cache['annotation_sha256'], original_selected_matrix=matrix.tolist(),
                                scopes={g: {name: int(mask.sum()) for name, mask in group_scopes(cache, g).items()}
                                        for g in groups})
            print(f'{split}: {json.dumps(report[split], ensure_ascii=False)}', flush=True)
    finally:
        for hook in hooks: hook.remove()
    if any(p.requires_grad or p.grad is not None for p in model.parameters()) or model.training:
        raise ValueError('Detector freeze invariant violated')
    report['complete'] = True
    (output / 'extraction_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def probe_features(cache, group, name):
    if name in MAP_STAGES: return cache['maps'][name]
    current = cache['groups'][group]; query = current['query']
    if name == 'decoder_query': return query
    if name == 'query_only_capacity_control':
        # Same 3D input/head parameter count as the local alternatives, no new evidence.
        return torch.cat((query, query, query), -1)
    if name == 'query_plus_pred_roi':
        return torch.cat((query, current['pred_p3'], current['pred_p4']), -1)
    if name == 'query_plus_gt_roi':
        return torch.cat((query, cache['maps']['encoder_p3'], cache['maps']['encoder_p4']), -1)
    raise ValueError(f'Unknown probe {name}')


def classify_metrics(labels, predictions, original, mask, names):
    count = int(mask.sum())
    if not count:
        return dict(count=0, accuracy=None, macro_f1=None, fixed=0, broken=0,
                    net_fixed=0, read_write_errors=0, original_correct=0, probe_correct=0, per_class=None,
                    confusion_matrix=None)
    gt, old, new = labels[mask], original[mask], predictions[mask]
    cm = torch.bincount(gt * len(names) + new, minlength=len(names)**2).reshape(len(names), len(names))
    result = confusion_metrics(cm.numpy(), names)
    fixed = int(((old != gt) & (new == gt)).sum())
    broken = int(((old == gt) & (new != gt)).sum())
    result.update(count=count, original_correct=int((old == gt).sum()), probe_correct=int((new == gt).sum()),
                  fixed=fixed, broken=broken, net_fixed=fixed-broken,
                  read_write_errors=int(cm[1, 2] + cm[2, 1]), confusion_matrix=cm.tolist())
    return result


def fit_probe(x, labels, validation_x, config, seed, device, progress_name):
    seed_everything(seed)
    mean = x.mean(0); std = x.std(0, unbiased=False).clamp_min(1e-5)
    standardized = (x - mean) / std
    val = (validation_x - mean) / std
    head = torch.nn.Linear(x.shape[1], 3).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    loader = DataLoader(TensorDataset(standardized, labels), batch_size=config['batch_size'],
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    for epoch in range(config['epochs']):
        for batch, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(batch.to(device)), target.to(device))
            if not torch.isfinite(loss): raise ValueError('Non-finite probe loss')
            loss.backward(); optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch + 1 == config['epochs']:
            print(f'{progress_name} seed={seed}: linear head {epoch+1}/{config["epochs"]}', flush=True)
    head.eval()
    with torch.no_grad():
        def predict(features):
            return torch.cat([head(batch.to(device)).argmax(-1).cpu()
                              for batch in features.split(config['batch_size'])])
        predictions = predict(val)
        train_accuracy = float((predict(standardized) == labels).float().mean())
    artifact = dict(head=head.cpu().state_dict(), mean=mean, std=std, seed=seed,
                    input_dim=x.shape[1], head_parameters=sum(p.numel() for p in head.parameters()))
    return predictions, train_accuracy, artifact


def validate_caches(train, val, config):
    expected = config_digest(config)
    for cache, split in ((train, 'train'), (val, 'val')):
        m = cache['metadata']
        if (not m['complete'] or m['schema_version'] != SCHEMA or cache['split'] != split
                or m['config_sha256'] != expected or cache['class_names'] != [c[1] for c in CATEGORIES]):
            raise ValueError(f'{split}: cache provenance/config/category mismatch')
        n = len(cache['labels'])
        if len(cache['records']) != n or len(cache['areas']) != n:
            raise ValueError('Cache GT row count mismatch')
        for value in cache['maps'].values():
            if value.shape[0] != n or not torch.isfinite(value).all(): raise ValueError('Invalid ROI features')
        for group in cache['groups'].values():
            if not torch.equal(group['valid'], group['query_ids'] >= 0): raise ValueError('Invalid query mask')
            for key, value in group.items():
                if len(value) != n or not torch.isfinite(value).all(): raise ValueError(f'Invalid query field {key}')
            if (group['original_labels'][group['valid']] < 0).any(): raise ValueError('Missing matched original label')
    if train['metadata']['checkpoint_sha256'] != val['metadata']['checkpoint_sha256']:
        raise ValueError('Train/val extracted from different checkpoints')
    if digest(config['checkpoint']) != train['metadata']['checkpoint_sha256']:
        raise ValueError('Checkpoint has changed since extraction')


def metric_stats(values):
    values = [v for v in values if v is not None]
    return dict(mean=statistics.mean(values) if values else None,
                sample_std=statistics.stdev(values) if len(values) > 1 else (0.0 if values else None))


def train_probes(config, output, device):
    train = torch.load(output / 'train_features.pth', map_location='cpu', weights_only=False)
    val = torch.load(output / 'val_features.pth', map_location='cpu', weights_only=False)
    validate_caches(train, val, config)
    if (output / 'probes').exists() or (output / 'summary.json').exists():
        raise FileExistsError('Existing probe results are preserved; use a new output directory')
    names = train['class_names']; summary = dict(complete=False, config=config,
        provenance=val['metadata'], original_selected_matrix=val['original_selected_matrix'].tolist(),
        limitations=LIMITATIONS, original_reference={}, experiments=[])
    lines = ['SCB-S 冻结特征诊断（不是 COCO AP，不是新的检测评估）',
             '准确率、write召回均限定在已匹配的前景GT；不能代表包含漏检的检测Recall。',
             'iou=不筛分数的IoU候选；selected=原后处理实际高分预测的一对一匹配。',
             'original_classifier=原分类结果；其他行为头为3次小分类器训练的均值±样本标准差。',
             '分组\t分类器\t匹配GT数\t准确率\twrite条件召回\t读写混淆数\t改对\t改错\t净改对']
    for group in ('iou', 'selected'):
        training_mask = train['groups'][group]['valid']
        if not training_mask.any() or not val['groups'][group]['valid'].any():
            raise ValueError(f'No matched GT in {group} group')
        scopes = group_scopes(val, group)
        original = val['groups'][group]['original_labels']
        reference = {scope: classify_metrics(val['labels'], original, original, mask, names)
                     for scope, mask in scopes.items()}
        summary['original_reference'][group] = reference
        a = reference['all_matched']
        lines.append(f'{group}\toriginal_classifier\t{a["count"]}\t{a["accuracy"]:.4f}'
                     f'\t{a["per_class"]["write"]["recall"]:.4f}\t{a["read_write_errors"]}\t—\t—\t—')
        print(lines[-1], flush=True)
        for name in PROBES:
            print(f'Training {group}/{name}: frozen cached features only', flush=True)
            x = probe_features(train, group, name)[training_mask]
            vx = probe_features(val, group, name)
            runs = []
            for seed in config['probe']['seeds']:
                predictions, train_accuracy, artifact = fit_probe(
                    x, train['labels'][training_mask], vx, config['probe'], seed, device, f'{group}/{name}')
                result = dict(seed=seed, train_accuracy=train_accuracy,
                              scopes={scope: classify_metrics(val['labels'], predictions, original, mask, names)
                                      for scope, mask in scopes.items()})
                target = output / 'probes' / group / name / f'seed{seed}'
                target.mkdir(parents=True)
                artifact.update(provenance=val['metadata'], group=group, probe=name, class_names=names)
                torch.save(artifact, target / 'head.pth')
                (target / 'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
                with (target / 'predictions.jsonl').open('w') as stream:
                    for i in torch.where(val['groups'][group]['valid'])[0].tolist():
                        record = dict(val['records'][i], query_id=int(val['groups'][group]['query_ids'][i]),
                            original_class=names[int(original[i])], probe_class=names[int(predictions[i])],
                            original_score=float(val['groups'][group]['scores'][i]),
                            match_iou=float(val['groups'][group]['ious'][i]),
                            scopes=[scope for scope, mask in scopes.items() if bool(mask[i])])
                        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                runs.append(result)
            aggregate = {}
            for scope in scopes:
                values = [r['scopes'][scope] for r in runs]
                aggregate[scope] = dict(count=values[0]['count'],
                    **{key: metric_stats([v[key] for v in values])
                       for key in ('accuracy', 'macro_f1', 'read_write_errors', 'fixed', 'broken', 'net_fixed')},
                    write_recall=metric_stats([v['per_class']['write']['recall'] if v['per_class'] else None for v in values]))
            entry = dict(group=group, probe=name, input_dim=x.shape[1], head_parameters=artifact['head_parameters'],
                         training_samples=int(training_mask.sum()), runs=runs, aggregate=aggregate)
            summary['experiments'].append(entry)
            a = aggregate['all_matched']; wr = a['write_recall']
            lines.append(f'{group}\t{name}\t{a["count"]}\t{a["accuracy"]["mean"]:.4f}±{a["accuracy"]["sample_std"]:.4f}'
                         f'\t{wr["mean"]:.4f}±{wr["sample_std"]:.4f}'
                         f'\t{a["read_write_errors"]["mean"]:.1f}\t{a["fixed"]["mean"]:.1f}'
                         f'\t{a["broken"]["mean"]:.1f}\t{a["net_fixed"]["mean"]:.1f}')
            print(lines[-1], flush=True)
    summary['complete'] = True
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (output / 'summary.txt').write_text('\n'.join(lines) + '\n\n' + '\n'.join(LIMITATIONS) + '\n')
    print('\n'.join(lines), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mode', choices=('all', 'extract', 'train'), default='all')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default=None)
    args = parser.parse_args(); config = load_config(args.config); validate_config(config)
    torch.set_num_threads(config['num_threads']); seed_everything(config['seed'])
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    output = Path(args.output_dir)
    if args.mode != 'train':
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('Existing output is preserved; use a new directory')
        for split in ('train', 'val'):
            data = config['data'][split]
            if not Path(data['image_dir']).is_dir() or not Path(data['annotation']).is_file():
                raise FileNotFoundError(f'{split}: missing dataset paths {data}')
        if not Path(config['checkpoint']).is_file(): raise FileNotFoundError(config['checkpoint'])
        output.mkdir(parents=True, exist_ok=True)
        (output / 'resolved_config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False))
        extract(config, output, device)
    else:
        report = json.loads((output / 'extraction_report.json').read_text())
        if not report['complete']: raise ValueError('Extraction incomplete')
    if args.mode != 'extract': train_probes(config, output, device)
    print(f'Finished: {output}', flush=True)


if __name__ == '__main__':
    main()
