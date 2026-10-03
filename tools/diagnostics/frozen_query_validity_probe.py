"""Frozen baseline binary query probe followed by fixed-candidate COCO rescoring.

This is a diagnostic experiment, not a detector training modification. Positive
queries are correct-class, IoU>=0.5 one-to-one representatives. Other overlapping
queries are ignored, not labeled background. Negatives have max IoU<0.3 against
all GT including crowd. Labels use train/val annotations separately, never a
validation error list. No threshold or fusion coefficient is tuned on val.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.ops import box_iou
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from tools.diagnostics.stage_feature_probe import build_detector
from tools.diagnostics.query_probe_export import score_match, xyxy
from tools.diagnostics.gt_box_classifier import load_config, seed_everything, COCO, COCOeval
from src.core.yaml_config import YAMLConfig


def clean_labels(raw, all_gt_boxes, positive_iou=.5, negative_iou=.3):
    """1=clean positive, 0=clean geometric negative, -1=ambiguous/excluded.

    Select one correct-class query per GT by original score. This deliberately
    ignores duplicates and wrong-class, high-overlap queries. Low IoU can still
    include unannotated targets: these are candidate negatives, not verified ones.
    """
    boxes = xyxy(raw['pred_boxes_cxcywh_normalized'])
    scores, classes = raw['pred_logits'].sigmoid().max(-1)
    gt = raw['gt_labels']
    overlaps = box_iou(boxes, raw['gt_boxes_xyxy_normalized'])
    same_class = classes[:, None] == gt[None]
    assignment, _ = score_match(overlaps * same_class, scores, positive_iou, 0.0)
    all_overlaps = box_iou(boxes, all_gt_boxes)
    maximum = all_overlaps.amax(-1) if len(all_gt_boxes) else scores.new_zeros(len(scores))
    labels = torch.full((len(scores),), -1, dtype=torch.long)
    labels[maximum < negative_iou] = 0
    labels[assignment >= 0] = 1
    return labels, scores


def binary_metrics(labels, scores):
    """Exact tied-score ROC AUC, stepwise AP and trapezoidal PR AUC.

    PR AP and trapezoidal area differ; report both with explicit definitions.
    Undefined single-class metrics are null, not fabricated zero values.
    """
    y, s = np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)
    pos, neg = int(y.sum()), int(len(y)-y.sum())
    result = {'positive_count': pos, 'negative_count': neg,
              'positive_prevalence': pos/len(y) if len(y) else None}
    if not pos or not neg:
        return {**result, 'AUROC': None, 'PR_average_precision': None, 'PR_AUC_trapezoidal': None}
    order = np.argsort(-s, kind='stable')
    y, s = y[order], s[order]
    ends = np.r_[np.where(np.diff(s))[0], len(s)-1]
    tp = np.cumsum(y)[ends]
    fp = (ends+1)-tp
    recall, precision = tp/pos, tp/(ends+1)
    tpr, fpr = np.r_[0., recall], np.r_[0., fp/neg]
    trap = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return {**result, 'AUROC': float(trap(tpr, fpr)),
            'PR_average_precision': float(np.sum(np.diff(np.r_[0., recall])*precision)),
            'PR_AUC_trapezoidal': float(trap(np.r_[1., precision], np.r_[0., recall]))}


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {'count': 0}
    edges = np.linspace(0, 1, 11)
    return {'count': len(values), 'mean': float(values.mean()),
            'quantiles_05_25_50_75_95': np.quantile(values, [.05,.25,.5,.75,.95]).tolist(),
            'histogram_edges': edges.tolist(), 'histogram_counts': np.histogram(values, edges)[0].tolist()}


@torch.no_grad()
def extract(config, output, device):
    annotation = {split: json.loads(Path(config['data'][split]['annotation']).read_text())
                  for split in ['train', 'val']}
    categories = sorted(annotation['train']['categories'], key=lambda c: c['id'])
    if categories != sorted(annotation['val']['categories'], key=lambda c: c['id']):
        raise ValueError('Train/val categories differ')
    names = [c['name'] for c in categories]
    cat_to_label = {c['id']: i for i, c in enumerate(categories)}
    model, epoch, source = build_detector(config, device)
    post = YAMLConfig(config['detector_config'], HGNetv2={'pretrained': False}).postprocessor
    if not post.use_focal_loss or post.remap_mscoco_category:
        raise ValueError('This probe currently requires sigmoid scores and direct dataset category mapping')
    metadata = {'checkpoint': config['checkpoint'], 'epoch': epoch, 'weight_source': source,
                'input_size': config['input_size'], 'class_names': names,
                'num_top_queries': post.num_top_queries,
                'positive_iou': config['validity_probe']['positive_iou'],
                'negative_iou': config['validity_probe']['negative_iou']}
    metadata['annotation_sha256'] = {s: hashlib.sha256(Path(config['data'][s]['annotation']).read_bytes()).hexdigest()
                                     for s in ['train','val']}
    capture = {}
    handle = model.decoder.decoder.layers[model.decoder.eval_idx].register_forward_hook(
        lambda module, args, value: capture.update(query=value))
    val_dir = output / 'val_raw'
    val_dir.mkdir(parents=True, exist_ok=True)
    try:
        for split in ['train', 'val']:
            by_image = {}
            for ann in annotation[split]['annotations']:
                if ann['bbox'][2] > 1 and ann['bbox'][3] > 1:
                    by_image.setdefault(ann['image_id'], []).append(ann)
            features, labels, scores = [], [], []
            for index, info in enumerate(sorted(annotation[split]['images'], key=lambda x: x['id'])):
                anns = by_image.get(info['id'], [])
                original_size = (info['height'], info['width'])
                existing = Path(config.get('validation_raw_dir', '')) / f"{info['id']}.pth"
                if split == 'val' and existing.is_file():
                    # Reuse only when the adjacent export provenance matches.
                    report_path = existing.parent.parent / 'audit_summary.json'
                    report = json.loads(report_path.read_text())
                    if (report['checkpoint'] != config['checkpoint'] or report['epoch'] != epoch
                            or report['weight_source'] != source
                            or report['config']['input_size'] != config['input_size']):
                        raise ValueError('Existing val query export provenance differs')
                    raw = torch.load(existing, map_location='cpu', weights_only=False)
                    if raw['class_names'] != names or raw['original_size_hw'] != list(original_size):
                        raise ValueError('Existing val raw categories or dimensions differ')
                    # Replace GT from the current split annotation, never stale audit labels.
                else:
                    with Image.open(Path(config['data'][split]['image_dir']) / info['file_name']) as image:
                        image = image.convert('RGB')
                        if image.size != (info['width'], info['height']):
                            raise ValueError('Image size differs from COCO metadata')
                        tensor = TF.to_tensor(TF.resize(image, config['input_size'],
                            interpolation=InterpolationMode.BILINEAR, antialias=True))[None].to(device)
                    capture.clear()
                    pred = model(tensor)
                    raw = {'image_id': info['id'], 'file_name': info['file_name'],
                           'class_names': names, 'original_size_hw': list(original_size),
                           'pred_logits': pred['pred_logits'][0].float().cpu(),
                           'pred_boxes_cxcywh_normalized': pred['pred_boxes'][0].float().cpu(),
                           'decoder_queries': capture['query'][0].float().cpu()}
                oh, ow = original_size
                all_boxes = torch.tensor([[a['bbox'][0]/ow, a['bbox'][1]/oh,
                    (a['bbox'][0]+a['bbox'][2])/ow, (a['bbox'][1]+a['bbox'][3])/oh]
                    for a in anns], dtype=torch.float32).reshape(-1, 4).clamp(0, 1)
                valid = torch.tensor([not a.get('iscrowd', 0) for a in anns], dtype=torch.bool)
                raw['gt_boxes_xyxy_normalized'] = all_boxes[valid]
                raw['gt_labels'] = torch.tensor([cat_to_label[a['category_id']] for a in anns], dtype=torch.long)[valid]
                raw['gt_annotation_ids'] = [a['id'] for a in anns if not a.get('iscrowd', 0)]
                label, score = clean_labels(raw, all_boxes, metadata['positive_iou'], metadata['negative_iou'])
                features.append(raw['decoder_queries'])
                labels.append(label)
                scores.append(score)
                if split == 'val':
                    torch.save(raw, val_dir / f"{info['id']}.pth")
                if (index+1) % 25 == 0:
                    print(f"extract {split}: {index+1}/{len(annotation[split]['images'])}", flush=True)
            cache = {'features': torch.cat(features), 'labels': torch.cat(labels),
                     'original_scores': torch.cat(scores), 'metadata': metadata}
            torch.save(cache, output / f'{split}_binary_features.pth')
            print(split, {k: int((cache['labels']==v).sum()) for k,v in [('positive',1),('negative',0),('ignored',-1)]}, flush=True)
    finally:
        handle.remove()


def train(config, output, device):
    seed_everything(config['seed'])
    cache = torch.load(output / 'train_binary_features.pth', map_location='cpu', weights_only=False)
    valid = cache['labels'] >= 0
    x, y = cache['features'][valid].float(), cache['labels'][valid].float()
    pos, neg = int(y.sum()), int(len(y)-y.sum())
    if not pos or not neg:
        raise ValueError('Training requires both positive and negative clean queries')
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-5)
    x = (x-mean)/std
    head = torch.nn.Linear(x.shape[1], 1).to(device)
    settings = config['validity_probe']
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    loader = DataLoader(TensorDataset(x,y), batch_size=settings['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(config['seed']))
    # Train-only class balancing. This sigmoid is a ranking score, not a proved
    # calibrated objectness probability. Validation never selects an epoch.
    positive_weight = torch.tensor(neg/pos, device=device)
    for epoch in range(settings['epochs']):
        total = 0.
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(head(bx.to(device)).squeeze(-1),
                by.to(device), pos_weight=positive_weight)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(by)
        if (epoch+1) % 10 == 0:
            print(f'probe epoch {epoch+1}/{settings["epochs"]}: train loss {total/len(y):.6f}', flush=True)
    torch.save({'head': head.cpu().state_dict(), 'mean': mean, 'std': std,
                'metadata': cache['metadata'], 'training_settings': settings,
                'train_positive': pos, 'train_negative': neg}, output / 'validity_linear_probe.pth')


def evaluate(config, output):
    probe = torch.load(output / 'validity_linear_probe.pth', map_location='cpu', weights_only=False)
    cache = torch.load(output / 'val_binary_features.pth', map_location='cpu', weights_only=False)
    if cache['metadata'] != probe['metadata']:
        raise ValueError('Probe and validation cache provenance differ')
    for split in ['train','val']:
        current = hashlib.sha256(Path(config['data'][split]['annotation']).read_bytes()).hexdigest()
        if current != cache['metadata']['annotation_sha256'][split]:
            raise ValueError('Annotation changed since feature extraction')
    def predict(x):
        with torch.no_grad():
            return (((x.float()-probe['mean'])/probe['std']) @ probe['head']['weight'].T
                    + probe['head']['bias']).squeeze(-1).sigmoid()
    validity = predict(cache['features'])
    original = cache['original_scores']
    y = cache['labels']
    clean = y >= 0
    high = original >= .5
    metrics = {}
    for name, mask in [('all_clean',clean), ('high_score_clean',clean & high),
                       ('all_positives_vs_high_score_negatives', clean & ((y==1) | high))]:
        metrics[name] = {'original_score': binary_metrics(y[mask].numpy(), original[mask].numpy()),
                         'validity_score': binary_metrics(y[mask].numpy(), validity[mask].numpy())}
    distributions = {f'{kind}_{group}': distribution(values[mask].numpy())
        for kind, values in [('original', original), ('validity', validity)]
        for group, mask in [('positive',y==1), ('negative',y==0), ('high_score_negative',(y==0)&high)]}
    # Scientific histogram, fixed bins and train-independent display ranges.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (kind, values) in zip(axes, [('Original score',original), ('Validity score',validity)]):
        for label, mask in [('Positive',y==1), ('Negative',y==0), ('High-score negative',(y==0)&high)]:
            if mask.any():
                ax.hist(values[mask].numpy(), bins=np.linspace(0,1,21), density=True,
                        histtype='step', linewidth=1.5, label=label)
        ax.set(title=kind, xlabel='Score', ylabel='Density', xlim=(0,1))
        ax.legend()
    fig.tight_layout(); fig.savefig(output/'score_distributions.png',dpi=160); plt.close(fig)
    annotation = json.loads(Path(config['data']['val']['annotation']).read_text())
    categories = sorted(annotation['categories'], key=lambda c:c['id'])
    raw_files = sorted((output/'val_raw').glob('*.pth'))
    expected_ids = {i['id'] for i in annotation['images']}
    if {int(p.stem) for p in raw_files} != expected_ids:
        raise ValueError('Val raw image inventory is incomplete or contains stale files')
    original_predictions, rescored_predictions = [], []
    topk = cache['metadata']['num_top_queries']
    for file in raw_files:
        raw = torch.load(file, map_location='cpu', weights_only=False)
        scores = raw['pred_logits'].sigmoid()
        values, flat_index = scores.flatten().topk(topk)
        query_ids, class_ids = flat_index//scores.shape[1], flat_index%scores.shape[1]
        boxes = xyxy(raw['pred_boxes_cxcywh_normalized'])[query_ids].clone()
        oh, ow = raw['original_size_hw']
        boxes *= torch.tensor([ow,oh,ow,oh])
        boxes[:,2:] -= boxes[:,:2]
        valid_scores = predict(raw['decoder_queries'])[query_ids]
        for n in range(topk):
            record = {'image_id':raw['image_id'], 'category_id':categories[int(class_ids[n])]['id'],
                      'bbox':boxes[n].tolist(), 'score':float(values[n])}
            original_predictions.append(record)
            rescored_predictions.append({**record, 'score':float(values[n]*valid_scores[n])})
    # Preserve the EXACT original top-300 candidate identities: only scores differ.
    # Do not reselect classes, use NMS, filter thresholds or invent extra boxes.
    results = {}
    coco = COCO(config['data']['val']['annotation'])
    for name, predictions in [('baseline_original',original_predictions), ('baseline_rescored',rescored_predictions)]:
        (output/f'{name}_predictions.json').write_text(json.dumps(predictions))
        evaluator = COCOeval(coco, coco.loadRes(predictions), 'bbox')
        evaluator.params.imgIds = sorted(expected_ids)
        evaluator.params.catIds = [c['id'] for c in categories]
        evaluator.params.maxDets = [1,10,100]
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        results[name] = {name: float(evaluator.stats[i]) for name,i in
            [('AP',0),('AP50',1),('AP75',2),('APm',4),('APl',5),('AR100',8)]}
        per_class = {}
        for k, category in enumerate(categories):
            precision = evaluator.eval['precision'][:,:,k,0,-1]
            usable = precision[precision > -1]
            per_class[category['name']] = float(usable.mean()) if usable.size else None
        results[name]['per_class_AP'] = per_class
    delta = {key: (results['baseline_rescored'][key]-value)*100
             for key,value in results['baseline_original'].items() if key != 'per_class_AP'}
    report = {'metadata':cache['metadata'], 'binary_metrics':metrics, 'score_distributions':distributions,
              'coco':results, 'delta_percentage_points':delta,
              'limitations':['Clean-query AUC excludes ambiguity and does not prove all-query validity.',
                  'Geometric negatives may include annotation omissions or scope ambiguity.',
                  'Weighted-BCE sigmoid is a ranking score, not established calibrated probability.',
                  'COCO uses unchanged annotations and fixed original candidate boxes/classes.',
                  'No validation-selected thresholds, epochs or fusion coefficients.']}
    report['limitations'].append('A weak linear probe is not proof that no nonlinear validity information exists.')
    expected_ap = config.get('expected_baseline_ap')
    report['baseline_AP_reproduced_within_0.1_point'] = (abs(results['baseline_original']['AP']-expected_ap)<=.001
                                                        if expected_ap is not None else None)
    (output/'validity_probe_results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    text = json.dumps({'binary_metrics':metrics,'coco':results,'delta_percentage_points':delta,
                      'score_distribution_quantiles': {k: {a:b for a,b in v.items() if not a.startswith('histogram')}
                                                        for k,v in distributions.items()},
                      'baseline_AP_reproduced_within_0.1_point':report['baseline_AP_reproduced_within_0.1_point']},
                     indent=2,ensure_ascii=False)
    (output/'results_summary.txt').write_text(text)
    print(text,flush=True)
    print(f'Finished: {output}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--mode',choices=['all','extract','train','evaluate'],default='all')
    args=parser.parse_args()
    config=load_config(args.config)
    settings=config['validity_probe']
    if not 0<=settings['negative_iou']<settings['positive_iou']<=1:
        raise ValueError('Invalid clean sample IoU thresholds')
    if settings['epochs'] < 1 or settings['batch_size'] < 1 or settings['learning_rate'] <= 0 or settings['weight_decay'] < 0:
        raise ValueError('Invalid probe training settings')
    seed_everything(config['seed'])
    torch.set_num_threads(4)
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    (output/'resolved_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.mode in ['all','extract']: extract(config,output,device)
    if args.mode in ['all','train']: train(config,output,device)
    if args.mode in ['all','evaluate']: evaluate(config,output)


if __name__=='__main__':
    main()
