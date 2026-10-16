"""Single-variable config, provenance and same-EMA diagnostics for SERF placements."""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core.yaml_utils import load_config
from src.zoo.dfine.spatial_evidence_relation_fusion import SERF_PLACEMENTS
from tools.diagnostics.scbs_instance_evidence_experiment import digest, difference, gt_identity
from tools.diagnostics.scbs_behavior_refinement_experiment import compare as compare_reference, pack

BASE = ROOT / 'configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml'
OPTIONS = dict(serf_dim=64, serf_window=8, serf_halo=2, serf_slots=4, serf_heads=4,
               serf_scale=.1, serf_chunk=128, serf_checkpoint=True)


def config_path(name):
    if name not in SERF_PLACEMENTS:
        raise ValueError('Unknown SERF placement')
    return ROOT / f'configs/dfine/dfine_s_scbs_hrw_110_AQS_{name}.yml'


def check_config(name):
    base, actual = load_config(str(BASE), cfg={}), load_config(str(config_path(name)), cfg={})
    expected = copy.deepcopy(base)
    expected['DFINE'].update(serf_placement=name, **OPTIONS)
    for cfg in (expected, actual):
        cfg.pop('__include__', None)
        cfg.pop('output_dir', None)
    if actual != expected:
        raise ValueError(f'{name}: config differs beyond the SERF branch')
    dec = actual['DFINETransformer']
    if (not dec.get('use_aqs_refine') or dec.get('aqs_apply_layer') != 2
            or dec.get('aqs_apply_layers') is not None or dec.get('aqs_refiner_type', 'legacy') != 'legacy'
            or (dec['aqs_threshold'], dec['aqs_residual_init'], dec['aqs_temperature']) != (.5, .1, .1)
            or actual['epochs'] != 110 or actual['num_classes'] != 3 or actual['remap_mscoco_category']
            or not actual['use_ema'] or actual['train_dataloader']['collate_fn']['stop_epoch'] != 100
            or actual['train_dataloader']['dataset']['transforms']['policy']['epoch'] != 100
            or any(actual[f'{split}_dataloader']['total_batch_size'] != 32 for split in ('train', 'val'))):
        raise ValueError('Keep plain AQS-L2 R1, SCB-S 110/100, batch32 and EMA')
    for owner in ('HGNetv2', 'HybridEncoder', 'DFINECriterion'):
        if any(v for k, v in actual[owner].items()
               if k.startswith('use_') and k != 'use_lab' and isinstance(v, bool)):
            raise ValueError('Other structure/loss active: ' + owner)
    if (any(v for k, v in dec.items() if k.startswith('use_') and k != 'use_aqs_refine' and isinstance(v, bool))
            or any(dec.get(k, 'none') != 'none' for k in
                   ('behavior_refinement_module', 'instance_evidence_module', 'literature_probe'))
            or any(actual['DFINE'].get(k, 'none') != 'none' for k in ('night_probe', 'finegrain_trial'))):
        raise ValueError('No other experimental module may be active')
    return actual


def preflight(names, dry=False):
    if not names or len(set(names)) != len(names):
        raise ValueError('Select distinct placements')
    for name in names:
        cfg = check_config(name)
        print(f'配置通过：AQS-L2 R1 + {name}; 110/100, batch32, seed0, 原损失。', flush=True)
    if dry:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; do not run full training on CPU')
    for split in ('train', 'val'):
        data = cfg[f'{split}_dataloader']['dataset']
        annotation = json.loads(Path(data['ann_file']).read_text(encoding='utf-8'))
        if sorted((c['id'], c['name']) for c in annotation['categories']) != [(0, 'hand-raising'), (1, 'read'), (2, 'write')]:
            raise ValueError('Unexpected SCB-S categories')
        ids = {r['id'] for r in annotation['images']}
        if not ids or len(ids) != len(annotation['images']):
            raise ValueError('Invalid image IDs')
        if any(a['image_id'] not in ids or a['category_id'] not in (0, 1, 2) for a in annotation['annotations']):
            raise ValueError('Invalid annotation ID/class')
        if any(not (Path(data['img_folder']) / r['file_name']).is_file() for r in annotation['images']):
            raise FileNotFoundError(f'{split}: image data incomplete; check server AQS base data paths')
        print(f'{split}: {len(ids)} images; annotations SHA256={digest(data["ann_file"])}', flush=True)


def snapshot(name, output):
    cfg = check_config(name)
    files = set((ROOT / 'tools/diagnostics/scbs_AQS_SERF_sync_files.txt').read_text().splitlines())
    files |= {str(p.relative_to(ROOT)) for p in (ROOT / 'src').rglob('*.py')} | {'train.py'}
    import yaml
    def includes(path):
        files.add(str(path.relative_to(ROOT)))
        raw = yaml.safe_load(path.read_text())
        for relative in raw.get('__include__', []):
            includes((path.parent / relative).resolve())
    includes(config_path(name))
    result = dict(module=name, base='AQS-L2 R1', seed=0, original_losses=True,
        initialization='Original HGNet pretraining; all original trainable parameters retained; no -r/-t',
        placement='post-FPN/PAN P3' if name == 'SERF_Encoder' else 'returned Backbone P3, before Encoder',
        resolved_config=cfg, source_sha256={p: digest(ROOT / p) for p in sorted(files) if p},
        annotation_sha256={s: digest(cfg[f'{s}_dataloader']['dataset']['ann_file']) for s in ('train', 'val')})
    with (Path(output) / 'experiment_inputs.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)


def parameters(checkpoint, output):
    import torch
    cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
    report = dict(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=digest(checkpoint),
        checkpoint_epoch_metadata=cp.get('last_epoch'), sources={},
        note='Norms are not observed contributions. Check log.txt for actual best epoch.')
    states = {'model': cp['model']}
    if 'ema' in cp:
        states['ema'] = cp['ema']['module']
    for source, state in states.items():
        settings = state['serf.settings'].tolist()
        scale = state['serf.raw_scale'].float()
        report['sources'][source] = dict(module=SERF_PLACEMENTS[settings[1]], settings=settings,
            raw_scale=float(scale), effective_scale=float(scale.tanh()),
            norms={k.removeprefix('serf.'): float(v.float().norm()) for k, v in state.items()
                   if k.startswith('serf.') and k != 'serf.settings'})
    with Path(output).open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report


def compare(output, reference, label):
    meta = json.loads((Path(reference) / 'evaluation/query_diagnostics/metadata.json').read_text())
    cfg = meta['config']
    if (cfg['DFINE'].get('serf_placement', 'none') != 'none'
            or any(v for k, v in cfg['HGNetv2'].items() if k.startswith('use_') and k != 'use_lab' and isinstance(v, bool))):
        raise ValueError('Reference must not contain SERF / experimental Backbone')
    return compare_reference(output, reference, label)


def on_off(name, output):
    out = Path(output)
    folders = [out / 'diagnostics_eval', out / f'{name}_off_eval']
    new, old = [json.loads((folder / 'analysis/summary.json').read_text()) for folder in folders]
    for key in ('checkpoint_sha256', 'weight_source', 'GT', 'images', 'queries', 'class_names', 'confidence_threshold', 'IoU_threshold'):
        if new[key] != old[key]:
            raise ValueError(f'ON/OFF protocol mismatch: {key}')
    if new['weight_source'] != 'ema' or gt_identity(folders[0] / 'analysis') != gt_identity(folders[1] / 'analysis'):
        raise ValueError('Use the same EMA and validation GT for ON/OFF')
    for folder, enabled in zip(folders, (True, False)):
        meta = json.loads((folder / 'evaluation/query_diagnostics/metadata.json').read_text())
        cfg, dec = meta['config'], meta['config']['DFINETransformer']
        if (not meta['complete'] or meta['checkpoint_sha256'] != new['checkpoint_sha256']
                or cfg.get('diagnostic_serf_output_enabled') is not enabled
                or cfg['DFINE']['serf_placement'] != name
                or not dec['use_aqs_refine'] or dec['aqs_apply_layer'] != 2
                or cfg.get('diagnostic_serf_ablation') != ('serf_on' if enabled else 'serf_off')):
            raise ValueError('SERF ON/OFF metadata does not describe actual runtime')
    keys = ('saved_COCO_metrics', 'saved_COCO_per_class', 'saved_COCO_per_class_and_size',
            'validator_confusion', 'best_query_cross_classification', 'confusion_pairs')
    result = dict(module=name, checkpoint_sha256=new['checkpoint_sha256'],
        on={k: new.get(k) for k in keys}, off={k: old.get(k) for k in keys},
        on_minus_off={k: difference(old[k], new[k]) if k in old and k in new else None for k in keys},
        note='Same jointly trained EMA; AQS stays ON. OFF is NOT separately trained AQS.',
        box_effect='Feature paths, query selection and boxes can change in BOTH placements.')
    with (out / f'{name}_on_off.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f'{name}: same EMA ON-OFF ΔAP={100*(new["saved_COCO_metrics"]["AP"]-old["saved_COCO_metrics"]["AP"]):+.4f} pp', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('preflight', 'parameters', 'compare', 'on-off', 'pack'))
    parser.add_argument('--modules', nargs='+', choices=SERF_PLACEMENTS)
    parser.add_argument('--module', choices=SERF_PLACEMENTS)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--label', choices=('AQS_L2_R1', 'BaselineRun2'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.operation == 'preflight': preflight(args.modules, args.dry_run)
    elif args.operation == 'parameters': parameters(args.checkpoint, args.output)
    elif args.operation == 'compare': compare(args.output, args.reference, args.label)
    elif args.operation == 'on-off': on_off(args.module, args.output)
    else: pack(args.module, args.output)


if __name__ == '__main__':
    main()
