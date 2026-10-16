"""Frozen AQS pipeline audit. Only independent foreground linear probes train.

The detector stays in eval mode. Passive hooks capture actual intermediate
outputs; extra head readouts run AFTER inference with hooks inactive. Layer
readouts are diagnostics, not a replacement for final COCO predictions.
"""
import argparse
import copy
import csv
import contextlib
import io
import json
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from torchvision.ops import box_iou
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.zoo.dfine.dfine_decoder import distance2bbox, weighting_function
from tools.diagnostics import scbs_feature_probe as feature
from tools.diagnostics.scbs_encoder_query_coverage import ProposalCapture, xyxy
from tools.diagnostics.stage_feature_probe import match_queries
from tools.diagnostics.gt_box_classifier import load_config, seed_everything

SCHEMA = 1
SOURCE_FILES = ('tools/diagnostics/scbs_whole_model_audit.py',
                'tools/diagnostics/scbs_feature_probe.py',
                'tools/diagnostics/scbs_encoder_query_coverage.py',
                'src/zoo/dfine/dfine_decoder.py', 'src/zoo/dfine/hybrid_encoder.py',
                'src/zoo/dfine/dfine.py')
MAPS = ('backbone_p2', 'backbone_p3', 'backbone_p4', 'backbone_p5',
        'encoder_input_p3', 'encoder_input_p4', 'encoder_input_p5',
        'encoder_attention_p5',
        'fpn_p4_block', 'fpn_p3_block', 'encoder_p3', 'encoder_p4', 'encoder_p5')
QUERIES = ('query_init', 'decoder_l1', 'decoder_l2_pre_aqs',
           'decoder_l2_post_aqs', 'decoder_l3')
STAGES = ('encoder_all', 'encoder_top300', 'decoder_l1',
          'decoder_l2_pre_aqs', 'decoder_l2_post_aqs',
          'decoder_l3_raw_cls', 'decoder_l3_lqe')
LIMITATIONS = [
    'Coverage is per-GT oracle geometry; one query may cover multiple GT. It is not COCO Recall.',
    'Encoder-all boxes are counterfactual frozen-head readouts; normal inference regresses selected positions only.',
    'Intermediate classifiers/LQE are original frozen heads evaluated after inference; their confidence calibration differs.',
    'Fixed-query class transitions use the final IoU-only Hungarian matched subset; they do not imply detector TP transitions.',
    'Final postprocessing uses unchanged flattened query-class TopK, not one argmax class per query.',
    'Feature probes use GT ROI geometry and foreground labels. They do not measure background rejection or detector AP.',
    'Same ROI sampler, GT subset, optimization and seeds; channel dimensions/head parameter counts differ and are reported.',
    'Train-only feature standardization and probe fitting; validation labels never select probe weights or training duration.',
    'Spatial pooling, dimensions and probe capacity affect separability; a lower probe score is not causal proof of information loss.',
    'FP32 single-image evaluation may differ slightly from AMP/batched training validation; COCO AP is recomputed independently.',
    'Unmatched predictions are not proven background. Duplicate/localization candidates require annotation and image review.',
    'Stage diagnosis narrows hypotheses; changing that position must still pass controlled full detector experiments.',
]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('', encoding='utf-8'); return
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def validate_config(config):
    feature.validate_config(config)
    if config.get('detector_variant') != 'aqs_l2_r010':
        raise ValueError('WholeModelAudit requires explicit aqs_l2_r010 opt-in')
    thresholds = config['iou_thresholds']
    if (not thresholds or len(set(thresholds)) != len(thresholds)
            or any(isinstance(t, bool) or not 0 < t <= 1 for t in thresholds)):
        raise ValueError('iou_thresholds must be unique finite values in (0,1]')
    if config['matching_iou'] not in thresholds:
        raise ValueError('matching_iou must be present in iou_thresholds')


class PipelineCapture:
    """One unchanged inference, then extra readouts; no train-mode forward."""
    def __init__(self, model):
        self.model = model
        self.values = {}
        self.handles = []
        self.active = False
        self.proposals = ProposalCapture(model)

    def __enter__(self):
        self.proposals.__enter__()
        def hook(key):
            def capture(module, args, value):
                if self.active:
                    self.values[key] = value
            return capture
        decoder = self.model.decoder
        for i, module in enumerate(self.model.backbone.stages):
            self.handles.append(module.register_forward_hook(hook(f'backbone_p{i+2}')))
        for i, module in enumerate(self.model.encoder.input_proj):
            self.handles.append(module.register_forward_hook(hook(f'encoder_input_p{i+3}')))
        def attention_output(module, args):
            if self.active:
                self.values['encoder_attention_p5'] = args[0]
        self.handles.append(self.model.encoder.lateral_convs[0].register_forward_pre_hook(attention_output))
        self.handles.append(self.model.encoder.fpn_blocks[0].register_forward_hook(hook('fpn_p4_block')))
        self.handles.append(self.model.encoder.fpn_blocks[1].register_forward_hook(hook('fpn_p3_block')))
        self.handles.append(self.model.encoder.register_forward_hook(hook('encoder')))
        self.handles.append(decoder.decoder.register_forward_hook(hook('decoded')))
        def decoder_input(module, args):
            if self.active:
                self.values['query_init'] = args[0]
        self.handles.append(decoder.decoder.register_forward_pre_hook(decoder_input))
        for i in range(3):
            self.handles.append(decoder.cls_query_adapter[i].register_forward_hook(hook(f'cls_{i}')))
            self.handles.append(decoder.dec_bbox_head[i].register_forward_hook(hook(f'corner_delta_{i}')))
        def aqs_input(module, args):
            if self.active:
                self.values['aqs_input'] = args[0]
                self.values['aqs_gate_logits'] = args[1]
        refiner = decoder.decoder.query_cls_refiner
        self.handles.append(refiner.register_forward_pre_hook(aqs_input))
        self.handles.append(refiner.register_forward_hook(hook('aqs_output')))
        return self

    def __exit__(self, *args):
        self.active = False
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.proposals.__exit__(*args)

    @torch.no_grad()
    def forward(self, samples):
        if self.model.training or samples.shape[0] != 1:
            raise ValueError('Capture requires frozen eval and batch_size=1')
        self.values.clear(); self.active = True
        try:
            outputs, proposal_stages, shapes = self.proposals.forward(samples)
        finally:
            self.active = False
        decoder = self.model.decoder
        maps = {name: self.values[name] for name in MAPS if not name.startswith('encoder_p')}
        maps.update({f'encoder_p{i+3}': value for i, value in enumerate(self.values['encoder'])})
        queries = dict(query_init=self.values['query_init'], decoder_l1=self.values['cls_0'],
                       decoder_l2_pre_aqs=self.values['aqs_input'],
                       decoder_l2_post_aqs=self.values['aqs_output'], decoder_l3=self.values['cls_2'])
        project = getattr(decoder.decoder, 'project', None)
        if project is None:
            project = weighting_function(decoder.reg_max, decoder.up, decoder.reg_scale)
        initial = self.values['decoded'][4].detach()
        corners = 0
        layer_boxes, raw_logits, quality_logits = [], [], []
        for i in range(3):
            corners = self.values[f'corner_delta_{i}'] + corners
            boxes = distance2bbox(initial, decoder.integral(corners, project), decoder.reg_scale)
            raw = decoder.dec_score_head[i](self.values[f'cls_{i}'])
            quality = decoder.decoder.lqe_layers[i](raw, corners)
            layer_boxes.append(boxes[0]); raw_logits.append(raw[0]); quality_logits.append(quality[0])
        torch.testing.assert_close(layer_boxes[2], outputs['pred_boxes'][0], rtol=0, atol=0)
        torch.testing.assert_close(quality_logits[2], outputs['pred_logits'][0], rtol=0, atol=0)
        torch.testing.assert_close(quality_logits[1], self.values['aqs_gate_logits'][0], rtol=0, atol=0)
        post_raw = decoder.dec_score_head[1](queries['decoder_l2_post_aqs'])
        cumulative_l2 = self.values['corner_delta_0'] + self.values['corner_delta_1']
        post_logits = decoder.decoder.lqe_layers[1](post_raw, cumulative_l2)[0]
        # The current L2 box is computed before AQS; only L3 can alter it later.
        stages = {name: proposal_stages[name] for name in ('encoder_all', 'encoder_top300')}
        template = dict(grid_ids=proposal_stages['encoder_top300']['grid_ids'],
                        valid=torch.ones(300, dtype=torch.bool, device=samples.device))
        for name, boxes, logits in (
                ('decoder_l1', layer_boxes[0], quality_logits[0]),
                ('decoder_l2_pre_aqs', layer_boxes[1], quality_logits[1]),
                ('decoder_l2_post_aqs', layer_boxes[1], post_logits),
                ('decoder_l3_raw_cls', layer_boxes[2], raw_logits[2]),
                ('decoder_l3_lqe', layer_boxes[2], quality_logits[2])):
            stages[name] = dict(template, boxes=boxes, logits=logits)
        for name, value in queries.items():
            if value.shape != (1, 300, decoder.hidden_dim) or not torch.isfinite(value).all():
                raise ValueError(f'Invalid captured query feature: {name}')
        return outputs, maps, queries, stages, shapes


def stage_gt_metrics(stage, gt, labels, fixed_ids, thresholds, confidence):
    """No class/score matching restriction for geometric coverage."""
    if not len(gt):
        return []
    boxes = xyxy(stage['boxes'].float())
    ious = box_iou(gt.float(), boxes)
    logits = stage['logits'].float()
    scores = logits.sigmoid()
    predicted = logits.argmax(-1)
    best_iou, best_id = ious.max(-1)
    rows = []
    for g, category in enumerate(labels.tolist()):
        q = int(best_id[g]); fixed = int(fixed_ids[g])
        def readout(index):
            competing = logits[index].clone(); competing[category] = -torch.inf
            return dict(query_id=index, iou=float(ious[g, index]),
                        predicted_class=int(predicted[index]), class_correct=bool(predicted[index] == category),
                        gt_score=float(scores[index, category]), max_score=float(scores[index].max()),
                        gt_class_margin=float(logits[index, category] - competing.max()))
        coverage = {}
        for threshold in thresholds:
            geometrical = ious[g] >= threshold
            correct_top1 = predicted == category
            class_high = scores[:, category] >= confidence
            coverage[str(threshold)] = dict(
                geometric=bool(geometrical.any()),
                valid_anchor_geometric=bool((geometrical & stage['valid']).any()),
                correct_top1=bool((geometrical & correct_top1).any()),
                correct_top1_high=bool((geometrical & correct_top1 & class_high).any()),
                gt_class_high_any_rank=bool((geometrical & class_high).any()))
        rows.append(dict(best_iou=float(best_iou[g]), best=readout(q), fixed=readout(fixed), coverage=coverage))
    return rows


def scopes(category, area):
    values = ['all', feature.CATEGORIES[category][1]]
    for name, low, high in (('small', 0, 32**2), ('medium', 32**2, 96**2),
                            ('large', 96**2, float('inf'))):
        if low <= area <= high:
            values += [name, feature.CATEGORIES[category][1] + '/' + name]
    return values


def summarize_timeline(rows, thresholds):
    coverage, transitions = [], []
    scope_names = sorted({s for row in rows for s in row['scopes']})
    for scope in scope_names:
        current = [r for r in rows if scope in r['scopes']]
        for name in STAGES:
            for threshold in thresholds:
                counts = Counter(key for row in current
                                 for key, flag in row['stages'][name]['coverage'][str(threshold)].items() if flag)
                coverage.append(dict(scope=scope, stage=name, iou_threshold=threshold, gt_count=len(current),
                                     **{key: counts[key] for key in ('geometric', 'valid_anchor_geometric',
                                         'correct_top1', 'correct_top1_high', 'gt_class_high_any_rank')}))
        paired = [r for r in current if r['final_hungarian_matched']]
        for before, after in zip(STAGES, STAGES[1:]):
            for threshold in thresholds:
                for status in ('geometric', 'correct_top1', 'correct_top1_high', 'gt_class_high_any_rank'):
                    values = [(r['stages'][before]['coverage'][str(threshold)][status],
                               r['stages'][after]['coverage'][str(threshold)][status]) for r in current]
                    transitions.append(dict(scope=scope, before=before, after=after, iou_threshold=threshold,
                        comparison='any_candidate/' + status, gt_count=len(current),
                        fixed=sum(not a and b for a,b in values), broken=sum(a and not b for a,b in values),
                        both_correct=sum(a and b for a,b in values), both_wrong=sum(not a and not b for a,b in values)))
                # Track exactly the same final-matched query ID across layers.
                values = [(r['stages'][before]['fixed'], r['stages'][after]['fixed']) for r in paired]
                valid = [(a,b) for a,b in values if a['iou'] >= threshold and b['iou'] >= threshold]
                transitions.append(dict(scope=scope, before=before, after=after, iou_threshold=threshold,
                    comparison='fixed_query/class_given_both_boxes_pass_iou', gt_count=len(valid),
                    fixed=sum(not a['class_correct'] and b['class_correct'] for a,b in valid),
                    broken=sum(a['class_correct'] and not b['class_correct'] for a,b in valid),
                    both_correct=sum(a['class_correct'] and b['class_correct'] for a,b in valid),
                    both_wrong=sum(not a['class_correct'] and not b['class_correct'] for a,b in valid)))
        # Actual postprocessor coverage is saved only at matching_iou; its
        # explicit threshold is taken from each GT record, not list order.
        threshold = current[0]['postprocessor_coverage_iou'] if current else None
        if threshold is not None:
            values = [(r['stages']['decoder_l3_lqe']['coverage'][str(threshold)]['gt_class_high_any_rank'],
                       r['actual_postprocessor_correct_high_cover']) for r in current]
            transitions.append(dict(scope=scope,before='decoder_l3_lqe',after='actual_postprocessor_topk',
                iou_threshold=threshold,comparison='any_candidate/gt_class_high_any_rank',gt_count=len(current),
                fixed=sum(not a and b for a,b in values),broken=sum(a and not b for a,b in values),
                both_correct=sum(a and b for a,b in values),both_wrong=sum(not a and not b for a,b in values)))
    return coverage, transitions


def score_statistics(rows):
    table = []
    for scope in sorted({s for row in rows for s in row['scopes']}):
        current = [r for r in rows if scope in r['scopes'] and r['final_hungarian_matched']]
        for name in STAGES:
            values = [r['stages'][name]['fixed'] for r in current]
            scores = [r['gt_score'] for r in values]
            margins = [r['gt_class_margin'] for r in values]
            ious = [r['iou'] for r in values]
            correlation = None
            if len(values) > 1:
                a,b = torch.tensor(ious,dtype=torch.float64),torch.tensor(scores,dtype=torch.float64)
                if a.std() > 0 and b.std() > 0:
                    correlation = float(torch.corrcoef(torch.stack((a,b)))[0,1])
            table.append(dict(scope=scope,stage=name,count=len(values),
                gt_score_mean=statistics.mean(scores) if scores else None,
                gt_score_median=statistics.median(scores) if scores else None,
                gt_margin_mean=statistics.mean(margins) if margins else None,
                gt_margin_median=statistics.median(margins) if margins else None,
                mean_iou=statistics.mean(ious) if ious else None,
                iou_gt_score_pearson=correlation,
                class_correct=sum(r['class_correct'] for r in values)))
    return table


def final_error_records(image_id, image_name, pairs, gt, labels, anns, threshold):
    assignment, overlaps, matrix = feature.selected_assignment(pairs, gt, labels, threshold)
    used = set(assignment[assignment >= 0].tolist())
    ious = box_iou(pairs['boxes'], gt) if len(gt) else torch.empty(len(pairs['boxes']), 0, device=gt.device)
    rows = []
    for g, ann in enumerate(anns):
        p = int(assignment[g])
        if p < 0 or int(pairs['labels'][p]) != int(labels[g]):
            rows.append(dict(image_id=image_id, image_name=image_name, annotation_id=ann['id'],
                type='unmatched_GT' if p < 0 else 'matched_wrong_class',
                gt_class=int(labels[g]), gt_box=gt[g].tolist(),
                prediction_class=None if p < 0 else int(pairs['labels'][p]),
                query_id=None if p < 0 else int(pairs['query_ids'][p]),
                score=None if p < 0 else float(pairs['scores'][p]),
                prediction_box=None if p < 0 else pairs['boxes'][p].tolist(),
                iou=None if p < 0 else float(overlaps[g])))
    for p in range(len(pairs['boxes'])):
        if p in used:
            continue
        best_iou, g = (ious[p].max(0) if len(gt) else (torch.tensor(0.), torch.tensor(-1)))
        value, g = float(best_iou), int(g)
        kind = ('duplicate_or_competing_candidate' if value >= threshold else
                'localization_or_scope_candidate' if value >= .1 else 'unmatched_low_overlap')
        rows.append(dict(image_id=image_id, image_name=image_name,
            annotation_id=None if g < 0 else anns[g]['id'], type=kind,
            gt_class=None if g < 0 else int(labels[g]), gt_box=None if g < 0 else gt[g].tolist(),
            prediction_class=int(pairs['labels'][p]), query_id=int(pairs['query_ids'][p]),
            score=float(pairs['scores'][p]), prediction_box=pairs['boxes'][p].tolist(), iou=value))
    return rows, matrix


def evaluate_coco(annotation_path, predictions, curve_output=None):
    from faster_coco_eval import COCO, COCOeval_faster
    coco = COCO(str(annotation_path))
    if predictions:
        predicted = coco.loadRes(predictions)
    else:
        predicted = COCO()
        predicted.dataset = dict(images=coco.dataset['images'], categories=coco.dataset['categories'], annotations=[])
        predicted.createIndex()
    evaluator = COCOeval_faster(coco, predicted, 'bbox')
    evaluator.params.catIds = [0,1,2]
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    names = ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl')
    result = {name:float(value) if value >= 0 else None
              for name,value in zip(names,evaluator.stats[:12])}
    precision = evaluator.eval['precision']
    result['per_class'] = {}
    for k, (_,name) in enumerate(feature.CATEGORIES):
        values = precision[:,:,k,0,-1]; values = values[values > -1]
        result['per_class'][name] = float(values.mean()) if len(values) else None
    if curve_output is not None:
        curves = dict(recall_thresholds=evaluator.params.recThrs.tolist(),
                      iou_thresholds=evaluator.params.iouThrs.tolist(),curves=[],
                      note='Official COCO interpolated precision; -1 means unavailable. Full original TopK input, no .5 score cutoff.')
        for k,(_,name) in enumerate(feature.CATEGORIES):
            for a,area_name in enumerate(evaluator.params.areaRngLbl):
                curves['curves'].append(dict(category=name,area=area_name,
                    precision_by_iou=precision[:,:,k,a,-1].tolist()))
        write_json(curve_output,curves)
    return result


def optional_tide(annotation_path, predictions_path, output):
    """Use the official public API if installed; never auto-install packages."""
    try:
        from tidecv import TIDE, datasets
    except ImportError as error:
        status = dict(status='unavailable',reason=str(error),
            note='Core audit remains complete. Optional tidecv estimates idealized AP50 error contributions; fixed-threshold counts are not dAP.')
    else:
        try:
            tide = TIDE()
            tide.evaluate(datasets.COCO(str(annotation_path)),datasets.COCOResult(str(predictions_path)),mode=TIDE.BOX)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                tide.summarize()
            (Path(output)/'tide_summary.txt').write_text(stream.getvalue(),encoding='utf-8')
            status = dict(status='complete',metric='TIDE default AP50, not COCO AP50:95',
                note='Idealized error correction dAP is not an achievable training gain; labels/scope require human review.')
        except Exception as error:
            status = dict(status='failed',reason=f'{type(error).__name__}: {error}',
                          note='Optional TIDE failed; inspect before interpreting its error contributions.')
    write_json(Path(output)/'tide_status.json',status)
    return status


def preflight(config, device):
    validate_config(config)
    for split in ('train','val'):
        data = config['data'][split]
        annotation = json.loads(Path(data['annotation']).read_text(encoding='utf-8'))
        if sorted((c['id'],c['name']) for c in annotation['categories']) != feature.CATEGORIES:
            raise ValueError(f'{split}: unexpected SCB-S categories')
        if not annotation['images'] or not annotation['annotations']:
            raise ValueError(f'{split}: empty dataset')
        for image in annotation['images']:
            if not (Path(data['image_dir']) / image['file_name']).is_file():
                raise FileNotFoundError(f"Missing {split} image: {image['file_name']}")
    if not Path(config['checkpoint']).is_file():
        raise FileNotFoundError(config['checkpoint'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in the selected environment')
    import faster_coco_eval  # fail before extraction/output creation
    model, post, epoch, source = feature.build_frozen_detector(config, device)
    if model.decoder.eval_idx != 2 or model.decoder.hidden_dim != 256:
        raise ValueError('Audit expects the three-layer 256-channel D-FINE-S')
    if list(model.encoder.use_encoder_idx) != [2] or model.encoder.num_encoder_layers != 1:
        raise ValueError('Audit expects the original P5-only Encoder attention routing')
    return model, post, epoch, source


@torch.no_grad()
def extract(config, output, device):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Preserve existing results: {output}')
    model, postprocessor, epoch, source = preflight(config, device)
    checkpoint_sha = feature.digest(config['checkpoint'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'resolved_config.json', config)
    metadata = dict(schema=SCHEMA, complete=False, detector_variant=config['detector_variant'],
        config_sha256=feature.config_digest(config), checkpoint=config['checkpoint'], checkpoint_sha256=checkpoint_sha,
        weight_source=source, checkpoint_epoch_metadata=epoch, input_size=config['input_size'],
        category_ids=[0,1,2], confidence_threshold=config['confidence_threshold'],
        matching_iou=config['matching_iou'], iou_thresholds=config['iou_thresholds'],
        postprocessor_top_k=postprocessor.num_top_queries, limitations=LIMITATIONS,
        detector_config_sha256=feature.digest(config['detector_config']),
        resolved_detector_config=feature.load_detector_config(config['detector_config'],cfg={}),
        source_sha256={name:feature.digest(ROOT/name) for name in SOURCE_FILES})
    report = dict(metadata=metadata, splits={})
    write_json(output/'extraction_report.json', report)
    with PipelineCapture(model) as capture:
        for split in ('train','val'):
            print(f'开始 {split} 冻结特征提取；使用验证预处理，检测器保持eval。',flush=True)
            data = config['data'][split]
            annotation = json.loads(Path(data['annotation']).read_text(encoding='utf-8'))
            grouped = {}
            for ann in annotation['annotations']:
                grouped.setdefault(ann['image_id'], []).append(ann)
            features = {key: [] for key in MAPS + QUERIES}
            labels, areas, masks, baseline_classes, records = [], [], [], [], []
            timeline, predictions, errors = [], [], []
            matrix = torch.zeros(4,4, dtype=torch.long)
            images = sorted(annotation['images'], key=lambda x:x['id'])
            target = output/split; target.mkdir()
            with (target/'gt_stage_records.jsonl').open('w', encoding='utf-8') as stream:
                for index, info in enumerate(images):
                    with Image.open(Path(data['image_dir'])/info['file_name']) as image:
                        rgb = image.convert('RGB'); original_size = rgb.size
                        samples = TF.to_tensor(TF.resize(rgb, config['input_size'],
                            interpolation=InterpolationMode.BILINEAR, antialias=True)).unsqueeze(0).to(device)
                    anns, original_gt = feature.annotations_for_image(grouped.get(info['id'],[]), info['id'], original_size)
                    gt = original_gt.to(device) / original_gt.new_tensor(list(original_size)*2).to(device)
                    y = torch.tensor([a['category_id'] for a in anns], dtype=torch.long, device=device)
                    outputs, maps, queries, stages, shapes = capture.forward(samples)
                    ids, oracle_iou = match_queries(xyxy(outputs['pred_boxes'][0].float()), gt, config['matching_iou'])
                    mask = ids >= 0
                    fixed = ids.to(device).clone()
                    if len(gt):
                        fallback = box_iou(gt, xyxy(outputs['pred_boxes'][0].float())).argmax(-1)
                        fixed[~mask.to(device)] = fallback[~mask.to(device)]
                    for name in MAPS:
                        features[name].append(feature.pool_roi(maps[name], gt, config['roi_size']).cpu())
                    for name in QUERIES:
                        selected = torch.zeros(len(anns), queries[name].shape[-1])
                        selected[mask] = queries[name][0, ids[mask].to(device)].float().cpu()
                        features[name].append(selected)
                    old = torch.full((len(anns),), -1, dtype=torch.long)
                    if mask.any():
                        old[mask] = outputs['pred_logits'][0,ids[mask].to(device)].argmax(-1).cpu()
                    labels.append(y.cpu()); masks.append(mask); baseline_classes.append(old)
                    area = torch.tensor([a.get('area',a['bbox'][2]*a['bbox'][3]) for a in anns])
                    areas.append(area)
                    for g, ann in enumerate(anns):
                        records.append(dict(image_id=info['id'], image_name=info['file_name'],
                            annotation_id=ann['id'], category_id=ann['category_id'], area=float(area[g]),
                            fixed_query_id=int(fixed[g]), final_hungarian_matched=bool(mask[g]),
                            final_hungarian_iou=float(oracle_iou[g])))
                    if split == 'val':
                        per_stage = {s:stage_gt_metrics(v,gt,y,
                                     stages['encoder_top300']['grid_ids'][fixed] if s == 'encoder_all' else fixed,
                                     config['iou_thresholds'],config['confidence_threshold'])
                                     for s,v in stages.items()}
                        # Full original TopK scores for COCO; threshold .5 only for the diagnostic matrix.
                        pairs = feature.selected_pairs(outputs,postprocessor,
                            torch.tensor([original_size],device=device),0.0)
                        for p in range(len(pairs['boxes'])):
                            box = pairs['boxes'][p]; width_height = box[2:] - box[:2]
                            predictions.append(dict(image_id=info['id'], category_id=int(pairs['labels'][p]),
                                bbox=torch.cat((box[:2],width_height)).tolist(), score=float(pairs['scores'][p])))
                        high = {k:v[pairs['scores'] >= config['confidence_threshold']] for k,v in pairs.items()}
                        # Match original validator's resize and scale-back FP32 geometry.
                        h,w = config['input_size']; ow,oh = original_size
                        matched_gt = original_gt.to(device) * gt.new_tensor([w/ow,h/oh]*2)
                        matched_gt *= gt.new_tensor([ow/w,oh/h]*2)
                        current_errors,current_matrix = final_error_records(info['id'],info['file_name'],high,
                            matched_gt,y,anns,config['matching_iou'])
                        errors.extend(current_errors); matrix += current_matrix
                        for g, ann in enumerate(anns):
                            row = dict(records[-len(anns)+g], gt_box_original_xyxy=original_gt[g].tolist(),
                                original_size=list(original_size), scopes=scopes(ann['category_id'],float(area[g])),
                                postprocessor_coverage_iou=config['matching_iou'],
                                stages={s:per_stage[s][g] for s in STAGES})
                            row['actual_postprocessor_correct_high_cover'] = bool((
                                (box_iou(gt[g:g+1], xyxy(outputs['pred_boxes'][0].float()))[0,high['query_ids']] >= config['matching_iou'])
                                & (high['labels'] == ann['category_id'])).any())
                            timeline.append(row)
                            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    if index == 0 or (index+1) % config['progress_every'] == 0 or index+1 == len(images):
                        print(f'{split}: {index+1}/{len(images)} images; GT={sum(len(v) for v in labels)}; eval FP32',flush=True)
            cache = dict(metadata=copy.deepcopy(metadata), split=split, annotation_sha256=feature.digest(data['annotation']),
                features={k:torch.cat(v) for k,v in features.items()}, labels=torch.cat(labels),
                areas=torch.cat(areas), valid=torch.cat(masks), original_classes=torch.cat(baseline_classes), records=records)
            cache['metadata']['complete'] = True
            torch.save(cache, output/f'{split}_features.pth')
            report['splits'][split] = dict(images=len(images),GT=len(cache['labels']),
                common_matched_gt=int(cache['valid'].sum()), annotation_sha256=cache['annotation_sha256'])
            if split == 'val':
                coverage,transitions = summarize_timeline(timeline,config['iou_thresholds'])
                write_csv(target/'stage_coverage.csv',coverage)
                write_csv(target/'stage_transitions.csv',transitions)
                write_csv(target/'fixed_query_score_statistics.csv',score_statistics(timeline))
                write_json(target/'stage_summary.json',dict(coverage=coverage,transitions=transitions,
                    actual_postprocessor_correct_high_cover=sum(r['actual_postprocessor_correct_high_cover'] for r in timeline),
                    fixed_query_matching='final class-blind Hungarian; no score filter',limitations=LIMITATIONS))
                write_json(target/'coco_predictions.json',predictions)
                metrics = evaluate_coco(data['annotation'],predictions,target/'coco_pr_curves.json')
                write_json(target/'validation_metrics.json',metrics)
                report['optional_tide'] = optional_tide(data['annotation'],target/'coco_predictions.json',target)
                with (target/'final_error_records.jsonl').open('w',encoding='utf-8') as stream:
                    for row in errors:
                        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                write_json(target/'final_errors.json',dict(confusion_matrix=matrix.tolist(),
                    row_labels=[c[1] for c in feature.CATEGORIES]+['unmatched_predictions'],
                    column_labels=[c[1] for c in feature.CATEGORIES]+['unmatched_GT'],
                    counts=dict(Counter(r['type'] for r in errors)),
                    matching='class-blind greedy highest-IoU; original query-class TopK; conf threshold',
                    note='Counts, not AP error contributions; unmatched predictions are not proven background.'))
                report['validation_metrics'] = metrics
            write_json(output/'extraction_report.json',report)
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Frozen detector invariant violated')
    if checkpoint_sha != feature.digest(config['checkpoint']):
        raise ValueError('Checkpoint changed during extraction')
    if any(feature.digest(ROOT/name) != value for name,value in metadata['source_sha256'].items()):
        raise ValueError('Model/diagnostic source changed during extraction')
    report['metadata']['complete'] = True
    write_json(output/'extraction_report.json',report)
    return report


def cache_pair(config, output):
    train,val = (torch.load(Path(output)/f'{s}_features.pth',map_location='cpu',weights_only=False) for s in ('train','val'))
    for split,cache in (('train',train),('val',val)):
        m = cache['metadata']
        if (not m['complete'] or m['schema'] != SCHEMA or cache['split'] != split
                or m['config_sha256'] != feature.config_digest(config)
                or m['checkpoint_sha256'] != feature.digest(config['checkpoint'])
                or cache['annotation_sha256'] != feature.digest(config['data'][split]['annotation'])
                or m['detector_config_sha256'] != feature.digest(config['detector_config'])
                or m['resolved_detector_config'] != feature.load_detector_config(config['detector_config'],cfg={})
                or any(feature.digest(ROOT/name) != value for name,value in m['source_sha256'].items())):
            raise ValueError('Cache provenance mismatch; use a fresh output')
        if set(cache['features']) != set(MAPS+QUERIES) or len(cache['records']) != len(cache['labels']):
            raise ValueError('Cache stage/GT mismatch')
        if not torch.equal(cache['valid'], cache['original_classes'] >= 0):
            raise ValueError('Invalid common GT mask')
        for value in cache['features'].values():
            if len(value) != len(cache['labels']) or not torch.isfinite(value).all():
                raise ValueError('Invalid feature cache')
    if train['metadata']['checkpoint_sha256'] != val['metadata']['checkpoint_sha256']:
        raise ValueError('Train/val detector weights differ')
    return train,val


def probe_scopes(cache):
    names = sorted({s for r in cache['records'] for s in scopes(r['category_id'],r['area'])})
    return {name:cache['valid'] & torch.tensor([name in scopes(r['category_id'],r['area'])
                                              for r in cache['records']]) for name in names}


def train_probes(config, output, device):
    output = Path(output)
    if (output/'probes').exists() or (output/'probe_summary.json').exists():
        raise FileExistsError('Probe results already exist; preserve them')
    train,val = cache_pair(config,output)
    if not train['valid'].any() or not val['valid'].any():
        raise ValueError('No common matched GT for probes')
    masks = probe_scopes(val)
    names = [c[1] for c in feature.CATEGORIES]
    old = val['original_classes']
    result = dict(complete=False,metadata=val['metadata'],common_gt_train=int(train['valid'].sum()),
                  common_gt_val=int(val['valid'].sum()),limitations=LIMITATIONS,experiments=[])
    table = []
    for stage in MAPS+QUERIES:
        x = train['features'][stage][train['valid']].float()
        vx = val['features'][stage].float()
        runs = []
        for seed in config['probe']['seeds']:
            print(f'Probe {stage}, seed={seed}, train={len(x)}, channels={x.shape[-1]}',flush=True)
            predictions,accuracy,artifact = feature.fit_probe(x,train['labels'][train['valid']],vx,
                config['probe'],seed,device,stage)
            current = dict(seed=seed,train_accuracy=accuracy,
                scopes={s:feature.classify_metrics(val['labels'],predictions,old,mask,names) for s,mask in masks.items()})
            target = output/'probes'/stage/f'seed{seed}'; target.mkdir(parents=True)
            artifact.update(metadata=val['metadata'],probe=stage,class_names=names)
            torch.save(artifact,target/'head.pth'); write_json(target/'metrics.json',current)
            runs.append(current)
        aggregate = {}
        for scope in masks:
            values = [r['scopes'][scope] for r in runs]
            stats = {key:feature.metric_stats([r[key] for r in values])
                     for key in ('accuracy','macro_f1','read_write_errors','fixed','broken','net_fixed')}
            for category in names:
                stats[category+'_recall'] = feature.metric_stats([
                    r['per_class'][category]['recall'] if r['per_class'] else None for r in values])
            aggregate[scope] = dict(count=values[0]['count'],**stats)
            flat = dict(stage=stage,scope=scope,count=values[0]['count'],input_dim=x.shape[-1],
                        head_parameters=artifact['head_parameters'])
            for key,value in stats.items():
                flat[key+'_mean'] = value['mean']; flat[key+'_std'] = value['sample_std']
            table.append(flat)
        result['experiments'].append(dict(stage=stage,input_dim=x.shape[-1],
            head_parameters=artifact['head_parameters'],runs=runs,aggregate=aggregate))
    result['complete'] = True
    write_json(output/'probe_summary.json',result); write_csv(output/'feature_probes.csv',table)
    return result


def finish(output):
    output = Path(output)
    extraction = json.loads((output/'extraction_report.json').read_text())
    probes = json.loads((output/'probe_summary.json').read_text())
    if not extraction['metadata']['complete'] or not probes['complete']:
        raise ValueError('Incomplete audit')
    if extraction['metadata']['checkpoint_sha256'] != probes['metadata']['checkpoint_sha256']:
        raise ValueError('Probe and pipeline weights differ')
    stages = json.loads((output/'val/stage_summary.json').read_text())
    errors = json.loads((output/'val/final_errors.json').read_text())
    summary = dict(complete=True,metadata=extraction['metadata'],splits=extraction['splits'],
        validation_metrics=extraction['validation_metrics'],final_errors=errors,
        stage_summary=stages,feature_probes=probes['experiments'],
        optional_tide=extraction.get('optional_tide'),limitations=LIMITATIONS)
    write_json(output/'summary.json',summary)
    lines = ['AQS WholeModelAudit：整个模型的冻结诊断',
             f"权重：{extraction['metadata']['checkpoint']}",
             f"来源：{extraction['metadata']['weight_source']}；SHA256：{extraction['metadata']['checkpoint_sha256']}",
             'COCO验证指标（0～100）：'+str({k:round(v*100,4) for k,v in extraction['validation_metrics'].items() if isinstance(v,(int,float))}),
             '最终固定阈值诊断计数：'+str(errors['counts']),
             '阅读顺序：val/stage_coverage.csv → val/stage_transitions.csv → feature_probes.csv',
             '逐目标记录：val/gt_stage_records.jsonl；人工检查：val/final_error_records.jsonl',
             '分类探针均使用最终IoU匹配的同一GT子集；每个探针重复种子，不是重复训练检测器。',
             '', 'write目标：stage / IoU / 几何覆盖 / 正确top1 / 正确top1高分 / GT总数']
    for row in stages['coverage']:
        if row['scope'] == 'write' and row['iou_threshold'] == .5:
            lines.append(f"{row['stage']} / .5 / {row['geometric']} / {row['correct_top1']} / {row['correct_top1_high']} / {row['gt_count']}")
    lines += ['', '探针：stage / 全体匹配GT准确率 / write条件召回 / medium条件准确率']
    for experiment in probes['experiments']:
        a = experiment['aggregate']
        def mean(scope,key):
            return a.get(scope,{}).get(key,{}).get('mean')
        lines.append(f"{experiment['stage']} / {mean('all','accuracy')} / {mean('all','write_recall')} / {mean('medium','accuracy')}")
    lines += ['',*LIMITATIONS]
    (output/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive = output.parent/(output.name+'.zip')
    if archive.exists():
        raise FileExistsError(f'Archive exists: {archive}')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.txt','.log'):
                bundle.write(path,path.relative_to(output.parent))
    print('\n'.join(lines[:10]),flush=True)
    print(f'完成：{output}\n压缩包（不含权重和特征缓存）：{archive}',flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/diagnostics/scbs_AQS_WholeModelAudit.yml')
    parser.add_argument('--output-dir')
    parser.add_argument('--checkpoint')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--mode',choices=('all','extract','probe','pack'),default='all')
    parser.add_argument('--preflight',action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.checkpoint:
        config['checkpoint'] = str(Path(args.checkpoint).resolve())
    validate_config(config); torch.set_num_threads(config['num_threads']); seed_everything(config['seed'])
    device = torch.device(args.device)
    if args.preflight:
        preflight(config,device); print('模型、严格权重、类别、train/val图片和依赖检查通过。',flush=True); return
    if not args.output_dir:
        parser.error('--output-dir is required')
    if args.mode in ('all','extract'):
        extract(config,args.output_dir,device)
    if args.mode in ('all','probe'):
        train_probes(config,args.output_dir,device)
    if args.mode in ('all','pack'):
        finish(args.output_dir)


if __name__ == '__main__':
    main()
