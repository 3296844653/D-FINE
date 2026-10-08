"""Frozen SCB-S proposal coverage before selection, before/after Decoder.

This is oracle geometric coverage, NOT detector recall/AP. The original model
only regresses selected Encoder positions; all-position boxes are counterfactual
diagnostic estimates from its unchanged frozen regression head.
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from torchvision.ops import box_iou
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from tools.diagnostics.gt_box_classifier import load_config, seed_everything
from tools.diagnostics.scbs_feature_probe import (
    CATEGORIES, annotations_for_image, build_frozen_detector, config_digest, digest,
)

STAGES = ('encoder_all', 'encoder_top300', 'decoder_final')
LIMITATIONS = [
    'Per-GT maximum IoU, no score filter or one-to-one assignment; NOT detector Recall/AP.',
    'Encoder-all boxes are counterfactual frozen-head estimates: normal inference regresses only selected positions.',
    'Encoder-top300 boxes are the actual input reference boxes BEFORE Decoder/pre_bbox_head refinement.',
    'Main coverage includes the original candidate pool, even invalid-anchor positions; valid-anchor-only coverage is also reported.',
    'Class-consistent coverage uses argmax labels, not detection confidence or the final flattened query-class top-k.',
    'All-position geometry/top-k loss is diagnostic evidence, not a causal proof of a scoring or feature defect.',
    'Decoder-final is all 300 final query boxes, not confidence-filtered postprocessor detections.',
    'No teachers, classes, labels or out-of-scope people are manually excluded or modified.',
    'FP32 single-image eval; ties/numerics may differ from an older AMP/batched export.',
    'checkpoint_epoch_metadata is saved checkpoint metadata, NOT proof of total training epochs or best-log epoch.',
]


def validate_config(config):
    size = config['input_size']
    if len(size) != 2 or any(type(v) is not int or v <= 0 or v % 32 for v in size):
        raise ValueError('input_size must be positive [height,width], multiples of 32')
    thresholds = config['iou_thresholds']
    if (not thresholds or len(set(thresholds)) != len(thresholds)
            or any(isinstance(v, bool) or not math.isfinite(v) or not 0 < v <= 1 for v in thresholds)):
        raise ValueError('iou_thresholds must be unique numbers in (0,1]')
    if (not config['splits'] or len(set(config['splits'])) != len(config['splits'])
            or any(v not in ('train', 'val') for v in config['splits'])):
        raise ValueError('splits must contain train and/or val without duplicates')
    for key in ('progress_every', 'num_threads'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be positive')
    if type(config['seed']) is not int or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if type(config['save_selected_queries']) is not bool:
        raise ValueError('save_selected_queries must be a boolean')


def xyxy(boxes):
    return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                      boxes[..., :2] + boxes[..., 2:] / 2), dim=-1)


class ProposalCapture:
    """Passive hooks: no method patches, no changes to forward return values."""
    def __init__(self, model):
        self.model = model
        self.values = {}
        self.active = False
        self.handles = []

    def __enter__(self):
        decoder = self.model.decoder
        for module, key in ((decoder.enc_output, 'memory'), (decoder.enc_score_head, 'logits')):
            def hook(module, args, value, key=key):
                if self.active:
                    self.values[key] = value.detach()
            self.handles.append(module.register_forward_hook(hook))

        def bbox_hook(module, args, value):
            if self.active:
                self.values['selected_memory'] = args[0].detach()
                self.values['selected_deltas'] = value.detach()
        self.handles.append(decoder.enc_bbox_head.register_forward_hook(bbox_hook))

        def input_hook(module, args):
            if self.active:
                self.values['reference_unact'] = args[1].detach()
                self.values['spatial_shapes'] = copy.deepcopy(args[3])
        self.handles.append(decoder.decoder.register_forward_pre_hook(input_hook))
        return self

    def __exit__(self, *args):
        self.active = False
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @torch.no_grad()
    def forward(self, samples):
        if self.model.training or samples.shape[0] != 1:
            raise ValueError('Coverage capture requires eval and batch_size=1')
        self.values.clear()
        self.active = True
        try:
            outputs = self.model(samples)
        finally:
            self.active = False
        decoder = self.model.decoder
        if decoder.num_queries != 300 or decoder.query_select_method != 'default':
            raise ValueError('Use the baseline default Encoder selection with exactly 300 queries')
        memory, logits = self.values['memory'], self.values['logits']
        if logits.shape != (*memory.shape[:2], 3):
            raise ValueError('Expected three Encoder class logits per grid position')
        indices = logits.max(-1).values.topk(decoder.num_queries, dim=-1).indices
        selected = memory.gather(1, indices[..., None].expand(-1, -1, memory.shape[-1]))
        if not torch.equal(selected, self.values['selected_memory']):
            raise ValueError('Reconstructed Top300 differs from actual selected memory; inspect selection/ties')
        if decoder.eval_spatial_size is None:
            anchors, valid = decoder._generate_anchors(
                self.values['spatial_shapes'], dtype=memory.dtype, device=memory.device)
        else:
            anchors, valid = decoder.anchors, decoder.valid_mask
        if anchors.shape[:2] != memory.shape[:2]:
            raise ValueError('Anchor/grid count does not match captured Encoder memory')
        # The additional full-grid regression happens only after ordinary inference.
        # Hooks are inactive and every detector parameter remains frozen.
        all_boxes = (decoder.enc_bbox_head(memory) + anchors).sigmoid()
        selected_anchors = anchors.gather(1, indices[..., None].expand(-1, -1, 4))
        expected_refs = self.values['selected_deltas'] + selected_anchors
        torch.testing.assert_close(expected_refs, self.values['reference_unact'], rtol=0, atol=0)
        top_boxes = self.values['reference_unact'].sigmoid()
        # Full-grid and selected-grid GEMM may have different FP32 rounding.
        torch.testing.assert_close(all_boxes.gather(1, indices[..., None].expand(-1, -1, 4)),
                                   top_boxes, rtol=1e-4, atol=1e-5)
        # Use the actually executed selected regression at those positions, so
        # coverage near an IoU threshold has exact set inclusion across stages.
        all_boxes = all_boxes.clone()
        all_boxes.scatter_(1, indices[..., None].expand(-1, -1, 4), top_boxes)
        grid_ids = torch.arange(memory.shape[1], device=memory.device)
        valid = valid[0, :, 0].bool()
        top_ids = indices[0]
        stages = {
            'encoder_all': dict(boxes=all_boxes[0], logits=logits[0], grid_ids=grid_ids, valid=valid),
            'encoder_top300': dict(boxes=top_boxes[0], logits=logits[0, top_ids],
                                   grid_ids=top_ids, valid=valid[top_ids]),
            'decoder_final': dict(boxes=outputs['pred_boxes'][0], logits=outputs['pred_logits'][0],
                                  grid_ids=top_ids, valid=torch.ones_like(top_ids, dtype=torch.bool)),
        }
        for stage in stages.values():
            if not torch.isfinite(stage['boxes']).all() or not torch.isfinite(stage['logits']).all():
                raise ValueError('Non-finite proposal box/logit')
        return outputs, stages, self.values['spatial_shapes']


def best_matches(stage, gt_boxes, gt_labels):
    """GT labels are grouping/optional class-consistency, NOT selection input."""
    n = len(gt_boxes)
    if not n:
        return dict(best_iou=torch.empty(0), best_index=torch.empty(0, dtype=torch.long),
                    class_iou=torch.empty(0), valid_iou=torch.empty(0))
    ious = box_iou(xyxy(stage['boxes']), gt_boxes)
    best, ids = ious.max(dim=0)
    labels = stage['logits'].argmax(-1)
    class_ious = ious.masked_fill(labels[:, None] != gt_labels[None, :], -1)
    valid_ious = ious.masked_fill(~stage['valid'][:, None], -1)
    return dict(best_iou=best.cpu(), best_index=ids.cpu(),
                class_iou=class_ious.max(0).values.clamp_min(0).cpu(),
                valid_iou=valid_ious.max(0).values.clamp_min(0).cpu())


def summarize(records, thresholds):
    scopes = {'all': records}
    scopes.update({name: [r for r in records if r['category_id'] == category_id]
                   for category_id, name in CATEGORIES})
    result = {}
    for scope, subset in scopes.items():
        stage_results = {}
        for stage in STAGES:
            stats = {}
            for threshold in thresholds:
                count = len(subset)
                row = {}
                for field, key in (('best_iou', 'geometric'), ('class_iou', 'class_consistent'),
                                   ('valid_iou', 'valid_anchor_only')):
                    covered = sum(r[stage][field] >= threshold for r in subset)
                    row[key] = dict(covered=covered, total=count,
                                    fraction=covered / count if count else None)
                stats[str(threshold)] = row
            stage_results[stage] = stats
        transitions = {}
        for threshold in thresholds:
            counts = dict(covered_before_but_not_selected=0, covered_top300_and_final=0,
                          covered_top300_but_lost_after_decoder=0,
                          uncovered_top300_but_recovered_after_decoder=0,
                          uncovered_top300_and_final=0)
            for r in subset:
                before, selected, final = [r[s]['best_iou'] >= threshold for s in STAGES]
                if before and not selected:
                    counts['covered_before_but_not_selected'] += 1
                key = ('covered_top300_and_final' if selected and final else
                       'covered_top300_but_lost_after_decoder' if selected else
                       'uncovered_top300_but_recovered_after_decoder' if final else
                       'uncovered_top300_and_final')
                counts[key] += 1
            transitions[str(threshold)] = counts
        result[scope] = dict(gt_count=len(subset), stages=stage_results, transitions=transitions)
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def report_text(summary):
    lines = ['SCB-S Encoder Top300 / Decoder 覆盖率（不是检测Recall/AP）',
             '不筛置信度；每个GT独立取最大IoU；不做一对一匹配。',
             'Encoder-all是同一冻结回归头的选前诊断估计；Top300读取实际初始参考框。',
             '比例按全部有效非crowd GT计算；类别一致覆盖另存于summary.json。',
             'split\tGT类别\tGT数量\tIoU阈值\t选前覆盖率\tTop300覆盖率\t最终覆盖率\t筛选丢失数\tDecoder恢复数\tDecoder丢失数']
    for split, report in summary['splits'].items():
        for name, stats in report['coverage'].items():
            for threshold in summary['iou_thresholds']:
                key = str(threshold)
                fractions = [stats['stages'][s][key]['geometric']['fraction'] for s in STAGES]
                values = ['N/A' if v is None else f'{100*v:.2f}%' for v in fractions]
                transition = stats['transitions'][key]
                lines.append('\t'.join([split, name, str(stats['gt_count']), key, *values,
                    str(transition['covered_before_but_not_selected']),
                    str(transition['uncovered_top300_but_recovered_after_decoder']),
                    str(transition['covered_top300_but_lost_after_decoder'])]))
        lines.append(f'{split}候选计数：{json.dumps(report["candidate_counts"], ensure_ascii=False)}')
    return '\n'.join(lines + [''] + LIMITATIONS) + '\n'


@torch.no_grad()
def run(config, output, device):
    validate_config(config)
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Existing results are preserved; use a new output directory')
    if not Path(config['checkpoint']).is_file():
        raise FileNotFoundError(config['checkpoint'])
    for split in config['splits']:
        data = config['data'][split]
        if not Path(data['image_dir']).is_dir() or not Path(data['annotation']).is_file():
            raise FileNotFoundError(f'{split}: missing dataset paths {data}')
    model, _, epoch, source = build_frozen_detector(config, device)
    summary = dict(schema_version=1, complete=False, checkpoint=str(config['checkpoint']),
        checkpoint_sha256=digest(config['checkpoint']), checkpoint_epoch_metadata=epoch,
        weight_source=source, config_sha256=config_digest(config),
        script_sha256=digest(__file__), decoder_source_sha256=digest(ROOT/'src/zoo/dfine/dfine_decoder.py'),
        resolved_detector_config=load_config_resolved(config['detector_config']),
        input_size=config['input_size'], iou_thresholds=config['iou_thresholds'],
        category_ids=[c[0] for c in CATEGORIES], class_names=[c[1] for c in CATEGORIES],
        stage_definitions={
            'encoder_all': 'Frozen enc_bbox_head evaluated on ALL enc_output positions + original anchors; counterfactual diagnostic, no score filtering.',
            'encoder_top300': 'Actual 300 reference boxes supplied to TransformerDecoder before pre_bbox_head/decoder refinement.',
            'decoder_final': 'All 300 final pred_boxes; no postprocessor/score filtering.'},
        limitations=LIMITATIONS, splits={})
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'resolved_config.json', config)
    write_json(output/'summary.json', summary)
    with ProposalCapture(model) as capture:
        for split in config['splits']:
            data = config['data'][split]
            annotation = json.loads(Path(data['annotation']).read_text(encoding='utf-8'))
            if sorted((c['id'], c['name']) for c in annotation['categories']) != CATEGORIES:
                raise ValueError(f'{split}: expected official SCB-S category mapping {CATEGORIES}')
            by_image = {}
            for ann in annotation['annotations']:
                by_image.setdefault(ann['image_id'], []).append(ann)
            records, counts = [], {s: dict(total=0, invalid_anchor_positions=0) for s in STAGES}
            target = output/split
            target.mkdir()
            queries_file = (target/'selected_queries.jsonl').open('w', encoding='utf-8') if config['save_selected_queries'] else None
            try:
                with (target/'gt_records.jsonl').open('w', encoding='utf-8') as stream:
                    images = sorted(annotation['images'], key=lambda r: r['id'])
                    for index, info in enumerate(images):
                        with Image.open(Path(data['image_dir'])/info['file_name']) as image:
                            rgb = image.convert('RGB')
                            width, height = rgb.size
                            tensor = TF.to_tensor(TF.resize(rgb, config['input_size'],
                                interpolation=InterpolationMode.BILINEAR, antialias=True)).unsqueeze(0).to(device)
                        anns, gt = annotations_for_image(by_image.get(info['id'], []), info['id'], (width, height))
                        gt = gt.to(device) / gt.new_tensor([width, height, width, height]).to(device)
                        labels = torch.tensor([a['category_id'] for a in anns], dtype=torch.long, device=device)
                        _, stages, shapes = capture.forward(tensor)
                        matches = {s: best_matches(v, gt, labels) for s, v in stages.items()}
                        for stage, proposals in stages.items():
                            counts[stage]['total'] += len(proposals['boxes'])
                            if stage != 'decoder_final':
                                counts[stage]['invalid_anchor_positions'] += int((~proposals['valid']).sum())
                        cpu = {s: {k: v.cpu() for k, v in p.items()} for s, p in stages.items()}
                        for g, ann in enumerate(anns):
                            row = dict(image_id=info['id'], image_name=info['file_name'],
                                annotation_id=ann['id'], category_id=ann['category_id'],
                                gt_class=CATEGORIES[ann['category_id']][1], original_size=[width, height],
                                gt_box_normalized_xyxy=gt[g].cpu().tolist())
                            for s in STAGES:
                                best = int(matches[s]['best_index'][g]); proposal = cpu[s]
                                probabilities = proposal['logits'][best].sigmoid()
                                label = int(probabilities.argmax())
                                row[s] = {k: float(matches[s][k][g]) for k in ('best_iou','class_iou','valid_iou')}
                                row[s].update(best_candidate_index=best, grid_index=int(proposal['grid_ids'][best]),
                                    best_box_normalized_cxcywh=proposal['boxes'][best].tolist(),
                                    best_candidate_valid_anchor=bool(proposal['valid'][best]) if s != 'decoder_final' else None,
                                    predicted_class=CATEGORIES[label][1], max_score=float(probabilities.max()),
                                    gt_class_score=float(probabilities[ann['category_id']]))
                            records.append(row)
                            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                        if queries_file is not None:
                            packed = dict(image_id=info['id'], image_name=info['file_name'],
                                original_size=[width,height], spatial_shapes=shapes,
                                encoder_grid_count=len(stages['encoder_all']['boxes']),
                                top300_grid_ids=cpu['encoder_top300']['grid_ids'].tolist(),
                                top300_valid_anchors=cpu['encoder_top300']['valid'].tolist(),
                                encoder_top300_boxes=cpu['encoder_top300']['boxes'].tolist(),
                                encoder_top300_logits=cpu['encoder_top300']['logits'].tolist(),
                                decoder_final_boxes=cpu['decoder_final']['boxes'].tolist(),
                                decoder_final_logits=cpu['decoder_final']['logits'].tolist())
                            queries_file.write(json.dumps(packed, ensure_ascii=False, allow_nan=False) + '\n')
                        if (index+1) % config['progress_every'] == 0 or index+1 == len(images):
                            print(f'{split}: {index+1}/{len(images)} images; GT {len(records)}', flush=True)
            finally:
                if queries_file is not None:
                    queries_file.close()
            summary['splits'][split] = dict(images=len(images), gt_count=len(records),
                annotation_sha256=digest(data['annotation']), candidate_counts=counts,
                coverage=summarize(records, config['iou_thresholds']))
            write_json(output/'summary.json', summary)
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Frozen detector invariant violated')
    summary['complete'] = True
    write_json(output/'summary.json', summary)
    (output/'summary.txt').write_text(report_text(summary), encoding='utf-8')
    print(report_text(summary), flush=True)
    return summary


def load_config_resolved(path):
    from src.core.yaml_utils import load_config as detector_config
    return detector_config(path, cfg={})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', choices=('cpu','cuda'))
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    torch.set_num_threads(config['num_threads'])
    seed_everything(config['seed'])
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    run(config, args.output_dir, device)
    print(f'Finished: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
