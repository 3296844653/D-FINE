"""Frozen D-FINE stage probes. ROI probes use GT boxes, query matching uses IoU only.

Run from the repository root. Extraction uses deterministic detector validation
preprocessing on both splits; no augmentation or detector optimization occurs.
Probe accuracy measures linear accessibility, not information-theoretic loss.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset
from torchvision.ops import roi_align, box_iou
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from src.core.yaml_config import YAMLConfig
from tools.diagnostics.gt_box_classifier import (
    load_config, seed_everything, confusion_metrics, save_confusion_matrix,
)

STAGES = ['backbone_p2', 'backbone_p3', 'backbone_p4',
          'encoder_p3', 'encoder_p4', 'decoder_query']


def match_queries(pred_xyxy, gt_xyxy, threshold):
    """One-to-one matching, maximizing valid matches first then total IoU.

    No class score or confidence threshold is used. Unmatched GT retain -1.
    """
    matches = torch.full((len(gt_xyxy),), -1, dtype=torch.long)
    overlaps = torch.zeros(len(gt_xyxy))
    if not len(gt_xyxy):
        return matches, overlaps
    ious = box_iou(pred_xyxy.float(), gt_xyxy.float()).cpu()
    reward = (ious >= threshold).float() * (len(gt_xyxy) + 1) + ious
    query_ids, gt_ids = linear_sum_assignment(-reward.numpy())
    for query_id, gt_id in zip(query_ids, gt_ids):
        value = float(ious[query_id, gt_id])
        if value >= threshold:
            matches[gt_id] = int(query_id)
            overlaps[gt_id] = value
    return matches, overlaps


def build_detector(config, device):
    cfg = YAMLConfig(config['detector_config'], HGNetv2={'pretrained': False})
    model = cfg.model
    # A baseline checkpoint must load strictly: do not silently use random heads.
    checkpoint = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, checkpoint.get('last_epoch'), 'ema' if 'ema' in checkpoint else 'model'


@torch.no_grad()
def extract(config, output_dir, device):
    model, epoch, source = build_detector(config, device)
    captures = {}
    handles = []
    def capture(name):
        def hook(module, args, value):
            captures[name] = value
        return hook
    for index, name in enumerate(STAGES[:3]):
        handles.append(model.backbone.stages[index].register_forward_hook(capture(name)))
    handles.append(model.encoder.register_forward_hook(capture('encoder')))
    # Hook the evaluated final decoder layer, before prediction heads and LQE.
    handles.append(model.decoder.decoder.layers[model.decoder.eval_idx].register_forward_hook(
        capture('decoder_query')))
    categories_reference = None
    height, width = config.get('input_size', [640, 640])
    report = {'checkpoint': config['checkpoint'], 'epoch': epoch, 'weight_source': source,
              'input_size': [height, width], 'matching_iou': config['matching_iou']}
    try:
        for split in ['train', 'val']:
            data = config['data'][split]
            annotation = json.loads(Path(data['annotation']).read_text())
            categories = sorted(annotation['categories'], key=lambda c: c['id'])
            if categories_reference is not None and categories != categories_reference:
                raise ValueError('Train/val category definitions differ')
            categories_reference = categories
            cat_to_label = {c['id']: i for i, c in enumerate(categories)}
            by_image = {}
            for ann in annotation['annotations']:
                if not ann.get('iscrowd', 0) and ann['bbox'][2] > 1 and ann['bbox'][3] > 1:
                    by_image.setdefault(ann['image_id'], []).append(ann)
            features = {s: [] for s in STAGES}
            labels, records, valid_masks = [], [], []
            detector_labels = []
            for index, info in enumerate(sorted(annotation['images'], key=lambda i: i['id'])):
                anns = by_image.get(info['id'], [])
                if not anns:
                    continue
                with Image.open(Path(data['image_dir']) / info['file_name']) as image:
                    rgb = image.convert('RGB')
                    original_width, original_height = rgb.size
                    resized = TF.resize(rgb, [height, width],
                                        interpolation=InterpolationMode.BILINEAR, antialias=True)
                    tensor = TF.to_tensor(resized).unsqueeze(0).to(device)
                boxes = []
                for ann in anns:
                    x, y, w, h = ann['bbox']
                    boxes.append([x / original_width, y / original_height,
                                  (x+w) / original_width, (y+h) / original_height])
                gt = torch.tensor(boxes, device=device).clamp(0, 1)
                captures.clear()
                prediction = model(tensor)
                for stage in STAGES[:-1]:
                    feat = captures[stage] if stage.startswith('backbone') else captures['encoder'][
                        0 if stage.endswith('p3') else 1]
                    scaled = gt * gt.new_tensor([feat.shape[-1], feat.shape[-2]] * 2)
                    # Same 3x3 ROI sampler and global average for all map stages.
                    # Each stage contributes C channels; no stage-specific CNN is trained.
                    roi = roi_align(feat.float(), [scaled], output_size=3,
                                    spatial_scale=1.0, sampling_ratio=2, aligned=True)
                    features[stage].append(roi.mean(dim=(-1, -2)).cpu())
                pred = prediction['pred_boxes'][0].float()
                pred_xyxy = torch.cat((pred[:, :2]-pred[:, 2:]/2,
                                      pred[:, :2]+pred[:, 2:]/2), dim=-1)
                matched, ious = match_queries(pred_xyxy, gt, config['matching_iou'])
                valid = matched >= 0
                query = captures['decoder_query'][0]
                selected = torch.zeros(len(anns), query.shape[-1])
                selected[valid] = query[matched[valid].to(device)].float().cpu()
                original_labels = torch.full((len(anns),), -1, dtype=torch.long)
                original_labels[valid] = prediction['pred_logits'][0][
                    matched[valid].to(device)].argmax(-1).cpu()
                detector_labels.append(original_labels)
                features['decoder_query'].append(selected)
                valid_masks.append(valid)
                labels.append(torch.tensor([cat_to_label[a['category_id']] for a in anns]))
                records.extend([{'image_id': info['id'], 'file_name': info['file_name'],
                                 'annotation_id': ann['id'], 'gt_class': categories[cat_to_label[ann['category_id']]]['name'],
                                 'query_index': int(matched[i]), 'query_iou': float(ious[i])}
                                for i, ann in enumerate(anns)])
                if (index+1) % 25 == 0:
                    print(f'{split}: {index+1}/{len(annotation["images"])} images', flush=True)
            cache = {'features': {s: torch.cat(v) for s, v in features.items()},
                     'labels': torch.cat(labels), 'query_valid': torch.cat(valid_masks),
                     'detector_labels': torch.cat(detector_labels),
                     'records': records, 'class_names': [c['name'] for c in categories],
                     'metadata': report}
            torch.save(cache, output_dir / f'{split}_features.pth')
            mask, y = cache['query_valid'], cache['labels']
            report[split] = {'gt_count': len(y), 'matched_count': int(mask.sum()),
                             'coverage': float(mask.float().mean()),
                             'original_detector_matched_accuracy': (
                                 float((cache['detector_labels'][mask] == y[mask]).float().mean())
                                 if mask.any() else None),
                             'per_class_coverage': {
                                 name: float(mask[y == i].float().mean())
                                 for i, name in enumerate(cache['class_names'])}}
            print(split, report[split], flush=True)
    finally:
        for handle in handles:
            handle.remove()
    (output_dir / 'extraction_report.json').write_text(json.dumps(report, indent=2))


def train_probes(config, output_dir, device):
    train = torch.load(output_dir / 'train_features.pth', weights_only=False)
    val = torch.load(output_dir / 'val_features.pth', weights_only=False)
    if train['class_names'] != val['class_names']:
        raise ValueError('Feature cache categories differ')
    names = train['class_names']
    # All probes use the exact same GT training subset to compare stages fairly.
    training_mask = train['query_valid']
    common_val = val['query_valid']
    if not common_val.any():
        raise ValueError('No matched validation GT; no common-subset comparison is possible')
    results = []
    for stage in STAGES:
        seed_everything(config['seed'])
        x = train['features'][stage][training_mask].float()
        y = train['labels'][training_mask]
        if not len(y):
            raise ValueError('No matched training GT; check checkpoint and IoU threshold')
        mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-5)
        x = (x-mean)/std
        vx = (val['features'][stage].float()-mean)/std
        head = torch.nn.Linear(x.shape[1], len(names)).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=config['probe']['learning_rate'],
                                      weight_decay=config['probe']['weight_decay'])
        generator = torch.Generator().manual_seed(config['seed'])
        loader = DataLoader(TensorDataset(x, y), batch_size=config['probe']['batch_size'],
                            shuffle=True, generator=generator)
        # Fixed training duration; validation is never used for head selection.
        for epoch in range(config['probe']['epochs']):
            head.train()
            for batch, target in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(head(batch.to(device)), target.to(device))
                loss.backward()
                optimizer.step()
        head.eval()
        with torch.no_grad():
            predictions = torch.cat([head(b.to(device)).argmax(-1).cpu()
                                     for b in vx.split(config['probe']['batch_size'])])
            train_predictions = torch.cat([head(b.to(device)).argmax(-1).cpu()
                                           for b in x.split(config['probe']['batch_size'])])
        stage_dir = output_dir / stage
        stage_dir.mkdir(exist_ok=True)
        metrics = {}
        for scope, mask in [('common_matched', common_val),
                            ('all_gt', torch.ones_like(common_val) if stage != 'decoder_query' else common_val)]:
            confusion = np.zeros((len(names), len(names)), dtype=np.int64)
            for true, pred in zip(val['labels'][mask].tolist(), predictions[mask].tolist()):
                confusion[true, pred] += 1
            metrics[scope] = confusion_metrics(confusion, names)
            save_confusion_matrix(confusion, names, stage_dir / scope)
        if stage == 'decoder_query':
            # 'all_gt' would hide missed GT, so name this conditional scope accurately.
            metrics.pop('all_gt')
        metrics['train_accuracy'] = float((train_predictions == y).float().mean())
        metrics['input_dim'] = x.shape[1]
        metrics['head_parameters'] = sum(p.numel() for p in head.parameters())
        metrics['train_samples'] = len(y)
        torch.save({'head': head.state_dict(), 'mean': mean, 'std': std,
                    'class_names': names, 'stage': stage}, stage_dir / 'linear_probe.pth')
        (stage_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        with (stage_dir / 'val_predictions.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['annotation_id', 'image_id', 'file_name', 'GT', 'prediction',
                             'query_matched', 'query_iou'])
            for i, record in enumerate(val['records']):
                if stage == 'decoder_query' and not common_val[i]:
                    continue
                writer.writerow([record['annotation_id'], record['image_id'], record['file_name'],
                                 record['gt_class'], names[predictions[i]],
                                 bool(common_val[i]), record['query_iou']])
        row = {'stage': stage, **metrics}
        results.append(row)
        print(stage, json.dumps(metrics), flush=True)
    (output_dir / 'probe_summary.json').write_text(json.dumps(results, indent=2))
    with (output_dir / 'probe_comparison.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['stage', 'input_dim', 'head_parameters', 'train_samples',
                         'train_accuracy', 'common_val_samples', 'common_accuracy', 'common_macro_f1',
                         *[f'{name}_f1' for name in names]])
        for row in results:
            common = row['common_matched']
            writer.writerow([row['stage'], row['input_dim'], row['head_parameters'], row['train_samples'],
                             row['train_accuracy'], common['num_samples'], common['accuracy'],
                             common['macro_f1'], *[common['per_class'][n]['f1'] for n in names]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', help='Override baseline checkpoint')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mode', choices=['all', 'extract', 'train'], default='all')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.checkpoint:
        config['checkpoint'] = args.checkpoint
    seed_everything(config['seed'])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'resolved_config.json').write_text(json.dumps(config, indent=2))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.mode in ['all', 'extract']:
        extract(config, output_dir, device)
    if args.mode in ['all', 'train']:
        train_probes(config, output_dir, device)


if __name__ == '__main__':
    main()
