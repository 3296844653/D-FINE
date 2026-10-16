"""Single-factor BoxLocal-L2 sweep provenance, parameters and diagnostic reports."""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core.yaml_utils import load_config
from tools.diagnostics.box_local_sweep_spec import DEFAULTS, PROFILES, settings, module_name
from tools.diagnostics.scbs_instance_evidence_experiment import digest, difference, gt_identity, compare as compare_saved
from tools.diagnostics.scbs_behavior_refinement_experiment import pack

BASE = ROOT / 'configs/dfine/dfine_s_scbs_hrw_110.yml'
PREFIX = 'decoder.decoder.night_local_adapter.'


def config_path(profile):
    return ROOT / f'configs/dfine/dfine_s_scbs_hrw_110_{module_name(profile)}.yml'


def check_config(profile):
    base, actual = load_config(str(BASE), cfg={}), load_config(str(config_path(profile)), cfg={})
    expected = copy.deepcopy(base)
    key, value = PROFILES[profile]
    expected['DFINE'].update(night_probe='decoder_l2_local', box_local_record_settings=True, **{key: value})
    for cfg in (actual, expected):
        cfg.pop('__include__', None); cfg.pop('output_dir', None)
    if actual != expected:
        raise ValueError(f'{profile}: more than the one intended parameter changed')
    if (base['epochs'] != 110 or base['num_classes'] != 3 or base['remap_mscoco_category']
            or not base['use_ema'] or base['HGNetv2']['name'] != 'B0'
            or base['HGNetv2'].get('return_idx') != [1,2,3]
            or base['DFINETransformer']['num_layers'] != 3
            or base['train_dataloader']['collate_fn']['stop_epoch'] != 100
            or base['train_dataloader']['dataset']['transforms']['policy']['epoch'] != 100
            or any(base[f'{s}_dataloader']['total_batch_size'] != 32 for s in ('train','val'))):
        raise ValueError('Expected original SCB-S / D-FINE-S 110/100, batch32, EMA')
    for owner in ('HGNetv2','HybridEncoder','DFINETransformer','DFINECriterion'):
        if any(v for k,v in base[owner].items() if k.startswith('use_') and k != 'use_lab' and isinstance(v,bool)):
            raise ValueError('No AQS, feature module or experimental loss may be active: ' + owner)
    if (any(base['DFINE'].get(k,'none') != 'none' for k in ('night_probe','finegrain_trial','serf_placement'))
            or any(base['DFINETransformer'].get(k,'none') != 'none' for k in
                   ('instance_evidence_module','behavior_refinement_module','literature_probe'))):
        raise ValueError('Only the standalone BoxLocal-L2 branch is allowed')
    effective = {k: actual['DFINE'].get(k,v) for k,v in DEFAULTS.items()}
    if effective != settings(profile):
        raise ValueError('Effective parameter values differ from the named profile')
    return actual


def preflight(profiles, dry=False):
    if not profiles or len(set(profiles)) != len(profiles):
        raise ValueError('Choose distinct BoxLocal parameter groups')
    for profile in profiles:
        cfg = check_config(profile)
        print(f'配置通过：{module_name(profile)} / {settings(profile)} / 无AQS、原损失。', flush=True)
    if dry: return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; do not launch full CPU training')
    for split in ('train','val'):
        data = cfg[f'{split}_dataloader']['dataset']
        ann = json.loads(Path(data['ann_file']).read_text(encoding='utf-8'))
        if sorted((c['id'],c['name']) for c in ann['categories']) != [(0,'hand-raising'),(1,'read'),(2,'write')]:
            raise ValueError('Unexpected SCB-S categories')
        ids = {r['id'] for r in ann['images']}
        if not ids or len(ids) != len(ann['images']): raise ValueError('Invalid image IDs')
        if any(a['image_id'] not in ids or a['category_id'] not in (0,1,2) for a in ann['annotations']):
            raise ValueError('Invalid annotation classes/IDs')
        if any(not (Path(data['img_folder'])/r['file_name']).is_file() for r in ann['images']):
            raise FileNotFoundError(f'{split}: missing images; check server baseline data paths')
        print(f'{split}: {len(ids)} images, annotation SHA256={digest(data["ann_file"])}', flush=True)


def snapshot(profile, output):
    cfg = check_config(profile)
    files = set((ROOT/'tools/diagnostics/scbs_BoxLocalQueryAdapter_l2_sweep_sync_files.txt').read_text().splitlines())
    files |= {str(p.relative_to(ROOT)) for p in (ROOT/'src').rglob('*.py')} | {'train.py'}
    import yaml
    def includes(path):
        files.add(str(path.relative_to(ROOT)))
        for relative in yaml.safe_load(path.read_text()).get('__include__',[]):
            includes((path.parent/relative).resolve())
    includes(config_path(profile))
    result = dict(module=module_name(profile), profile=profile, seed=0, settings=settings(profile),
        base='standalone BoxLocal-L2 dim64/grid3/offset0.15/residual0.10',
        initialization='From scratch, original HGNet pretraining; no -r/-t', original_losses=True,
        resolved_config=cfg, source_sha256={p:digest(ROOT/p) for p in sorted(files) if p},
        annotation_sha256={s:digest(cfg[f'{s}_dataloader']['dataset']['ann_file']) for s in ('train','val')})
    with (Path(output)/'experiment_inputs.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)


def parameters(checkpoint, output):
    import torch
    cp = torch.load(checkpoint,map_location='cpu',weights_only=True)
    result = dict(checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=digest(checkpoint),
        checkpoint_epoch_metadata=cp.get('last_epoch'),sources={},
        note='Fixed settings do not learn. Offset bias alone is NOT an observed sampling location; norms are NOT contributions.')
    states = {'model':cp['model']}
    if 'ema' in cp: states['ema'] = cp['ema']['module']
    for source,state in states.items():
        spec = state[PREFIX+'_box_local_settings'].tolist()
        result['sources'][source] = dict(hidden_dim=spec[1],dim=spec[2],grid_size=spec[3],points=spec[3]**2,
            offset_scale=spec[4]/1e9,residual_scale=spec[5]/1e9,
            offset_bias=state[PREFIX+'sampler.offset.bias'].tolist(),
            norms={k.removeprefix(PREFIX):float(v.float().norm()) for k,v in state.items()
                   if k.startswith(PREFIX) and not k.endswith('_box_local_settings')})
    with Path(output).open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
    return result


def reference_summary(reference, label):
    reference = Path(reference)
    meta = json.loads((reference/'evaluation/query_diagnostics/metadata.json').read_text())
    old = json.loads((reference/'analysis/summary.json').read_text())
    cfg, dec = meta['config'], meta['config']['DFINETransformer']
    if (not meta['complete'] or old['weight_source'] != 'ema'
            or meta['checkpoint_sha256'] != old['checkpoint_sha256'] or cfg['epochs'] != 110
            or any(v for k,v in dec.items() if k.startswith('use_') and isinstance(v,bool))
            or any(dec.get(k,'none') != 'none' for k in ('behavior_refinement_module','instance_evidence_module','literature_probe'))
            or cfg['DFINE'].get('finegrain_trial','none') != 'none' or cfg['DFINE'].get('serf_placement','none') != 'none'
            or any(v for k,v in cfg['HybridEncoder'].items() if k.startswith('use_') and isinstance(v,bool))
            or any(v for k,v in cfg['DFINECriterion'].items() if k.startswith('use_') and isinstance(v,bool))
            or any(v for k,v in cfg['HGNetv2'].items() if k.startswith('use_') and k != 'use_lab' and isinstance(v,bool))
            or cfg.get('diagnostic_box_local_output_enabled') is False):
        raise ValueError('Reference is not compatible standalone/default or original baseline EMA')
    wanted = 'decoder_l2_local' if label == 'BoxLocal_Default' else 'none'
    if cfg['DFINE'].get('night_probe','none') != wanted:
        raise ValueError('Reference module/placement differs from label')
    if label == 'BoxLocal_Default' and {k:cfg['DFINE'].get(k,v) for k,v in DEFAULTS.items()} != DEFAULTS:
        raise ValueError('Default BoxLocal reference has non-default parameters')
    return old


def compare(output, reference, label):
    out, reference = Path(output), Path(reference)
    old = reference_summary(reference,label)
    if label == 'BaselineRun2': return compare_saved(out,reference,label)
    new = json.loads((out/'diagnostics_eval/analysis/summary.json').read_text())
    for k in ('weight_source','GT','images','queries','class_names','confidence_threshold','IoU_threshold'):
        if old[k] != new[k]: raise ValueError('Different diagnostic protocol: ' + k)
    if gt_identity(reference/'analysis') != gt_identity(out/'diagnostics_eval/analysis'):
        raise ValueError('Reference and new GT identities differ')
    keys = ('saved_COCO_metrics','saved_COCO_per_class','saved_COCO_per_class_and_size',
            'validator_confusion','best_query_cross_classification','confusion_pairs')
    result = dict(reference=label,reference_checkpoint_sha256=old['checkpoint_sha256'],module_checkpoint_sha256=new['checkpoint_sha256'],
        reference_metrics={k:old.get(k) for k in keys},module_metrics={k:new.get(k) for k in keys},
        numeric_differences={k:difference(old[k],new[k]) if k in old and k in new else None for k in keys},
        note='One-factor single-run response, not a repeatability or causality proof; AP differences are fractions.')
    with (out/f'diagnostics_eval/compare_{label}.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)


def on_off(profile, output):
    out = Path(output); name = module_name(profile)
    dirs = [out/'diagnostics_eval',out/f'{name}_off_eval']
    new, old = [json.loads((d/'analysis/summary.json').read_text()) for d in dirs]
    for k in ('checkpoint_sha256','weight_source','GT','images','queries','class_names','confidence_threshold','IoU_threshold'):
        if new[k] != old[k]: raise ValueError('ON/OFF protocol mismatch: ' + k)
    if new['weight_source'] != 'ema' or gt_identity(dirs[0]/'analysis') != gt_identity(dirs[1]/'analysis'):
        raise ValueError('Use the same EMA and GT for ON/OFF')
    for folder, enabled in zip(dirs,(True,False)):
        m = json.loads((folder/'evaluation/query_diagnostics/metadata.json').read_text())
        cfg = m['config']
        if (not m['complete'] or m['checkpoint_sha256'] != new['checkpoint_sha256']
                or cfg.get('diagnostic_box_local_output_enabled') is not enabled
                or cfg.get('diagnostic_box_local_ablation') != ('box_on' if enabled else 'box_off')
                or cfg['DFINE']['night_probe'] != 'decoder_l2_local'
                or cfg['DFINETransformer'].get('use_aqs_refine',False)
                or {k:cfg['DFINE'].get(k,v) for k,v in DEFAULTS.items()} != settings(profile)):
            raise ValueError('ON/OFF metadata does not describe actual BoxLocal profile/runtime')
    keys = ('saved_COCO_metrics','saved_COCO_per_class','saved_COCO_per_class_and_size',
            'validator_confusion','best_query_cross_classification','confusion_pairs')
    result = dict(module=name,settings=settings(profile),checkpoint_sha256=new['checkpoint_sha256'],
        on={k:new.get(k) for k in keys},off={k:old.get(k) for k in keys},
        on_minus_off={k:difference(old[k],new[k]) if k in old and k in new else None for k in keys},
        note='Same jointly trained EMA; OFF is not an independently trained D-FINE baseline; L3 boxes can change.')
    with (out/f'{name}_on_off.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('preflight','parameters','compare','on-off','pack'))
    p.add_argument('--groups',nargs='+',choices=PROFILES)
    p.add_argument('--group',choices=PROFILES)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--reference',type=Path);p.add_argument('--label',choices=('BoxLocal_Default','BaselineRun2'))
    p.add_argument('--dry-run',action='store_true');a = p.parse_args()
    if a.operation == 'preflight':preflight(a.groups,a.dry_run)
    elif a.operation == 'parameters':parameters(a.checkpoint,a.output)
    elif a.operation == 'compare':compare(a.output,a.reference,a.label)
    elif a.operation == 'on-off':on_off(a.group,a.output)
    else:pack(module_name(a.group),a.output)


if __name__ == '__main__': main()
