"""Protocol, provenance, parameters and diagnostics for four independent modules."""
import argparse
import copy
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.core.yaml_utils import load_config
from src.zoo.dfine.behavior_refinement import BEHAVIOR_REFINEMENTS
from tools.diagnostics.scbs_instance_evidence_experiment import digest, compare as compare_saved, difference, gt_identity

BASE = ROOT/'configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml'


def config_path(name): return ROOT/f'configs/dfine/dfine_s_scbs_hrw_110_AQS_{name}.yml'


def check_config(name):
    base,actual = load_config(str(BASE),cfg={}),load_config(str(config_path(name)),cfg={})
    expected = copy.deepcopy(base)
    settings = dict(behavior_refinement_module=name,behavior_refinement_dim=64,
        behavior_refinement_scale=.1,behavior_refinement_chunk=32,behavior_refinement_checkpoint=True)
    if name != 'GeometryAwareQueryCompetition': settings['behavior_refinement_grid']=3
    expected['DFINETransformer'].update(settings)
    for cfg in (expected,actual): cfg.pop('__include__',None); cfg.pop('output_dir',None)
    if actual != expected: raise ValueError(f'{name}: configuration differs beyond the single new architecture')
    dec = actual['DFINETransformer']
    if (not dec['use_aqs_refine'] or dec['aqs_apply_layer'] != 2 or dec.get('aqs_apply_layers') is not None
            or dec.get('aqs_refiner_type','legacy') != 'legacy'
            or (dec['aqs_threshold'],dec['aqs_residual_init'],dec['aqs_temperature']) != (.5,.1,.1)):
        raise ValueError('AQS must remain plain L2 R1')
    if (actual['epochs'] != 110 or actual['num_classes'] != 3 or actual['remap_mscoco_category']
            or not actual['use_ema'] or actual['train_dataloader']['collate_fn']['stop_epoch'] != 100
            or actual['train_dataloader']['dataset']['transforms']['policy']['epoch'] != 100
            or any(actual[f'{split}_dataloader']['total_batch_size'] != 32 for split in ('train','val'))):
        raise ValueError('Expected original SCB-S 110/100, batch32, EMA protocol')
    for owner in ('HGNetv2','HybridEncoder','DFINECriterion'):
        if any(v for k,v in actual[owner].items() if k.startswith('use_') and k!='use_lab' and isinstance(v,bool)):
            raise ValueError('Other experimental structure/loss is active: '+owner)
    if (any(v for k,v in dec.items() if k.startswith('use_') and k!='use_aqs_refine' and isinstance(v,bool))
            or any(actual['DFINE'].get(k,'none')!='none' for k in ('night_probe','finegrain_trial','literature_probe'))):
        raise ValueError('A second experimental path is active')
    return actual


def compare(output,reference,label):
    metadata=json.loads((Path(reference)/'evaluation/query_diagnostics/metadata.json').read_text())
    cfg=metadata['config'];dec=cfg['DFINETransformer']
    if (dec.get('behavior_refinement_module','none')!='none'
            or any(v for k,v in cfg['HybridEncoder'].items() if k.startswith('use_') and isinstance(v,bool))
            or any(cfg['DFINE'].get(k,'none')!='none' for k in ('night_probe','finegrain_trial','literature_probe'))):
        raise ValueError('Reference must be plain AQS R1 / original Run2, without another structure')
    if cfg['epochs']!=110:raise ValueError('Reference is not the 110-epoch protocol')
    return compare_saved(Path(output),Path(reference),label)


def preflight(names,dry=False):
    if not names or len(set(names)) != len(names) or any(n not in BEHAVIOR_REFINEMENTS for n in names):
        raise ValueError('Select distinct independent behavior modules')
    cfg = None
    for name in names:
        cfg = check_config(name)
        print(f'配置通过：AQS-L2 R1 + {name}; 110/100, batch32, seed0, 原损失。',flush=True)
    if dry: return
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; full CPU training is not allowed')
    for split in ('train','val'):
        data = cfg[f'{split}_dataloader']['dataset']
        annotation = json.loads(Path(data['ann_file']).read_text(encoding='utf-8'))
        if sorted((c['id'],c['name']) for c in annotation['categories']) != [(0,'hand-raising'),(1,'read'),(2,'write')]:
            raise ValueError('Unexpected SCB-S categories')
        ids = {r['id'] for r in annotation['images']}
        if len(ids) != len(annotation['images']) or not ids: raise ValueError('Invalid image IDs')
        if any(a['image_id'] not in ids or a['category_id'] not in (0,1,2) for a in annotation['annotations']):
            raise ValueError('Invalid annotation ID/class')
        if any(not (Path(data['img_folder'])/r['file_name']).is_file() for r in annotation['images']):
            raise FileNotFoundError(f'{split}: image data incomplete')
        print(f'{split}: {len(ids)} images; annotations SHA256={digest(data["ann_file"])}',flush=True)


def snapshot(name,output):
    cfg = check_config(name)
    files = (ROOT/'tools/diagnostics/scbs_AQS_BehaviorFour_sync_files.txt').read_text().splitlines()
    # Record all Python model/data/loss code, not just the deployment delta.
    files=set(files)|{str(p.relative_to(ROOT)) for p in (ROOT/'src').rglob('*.py')}|{'train.py'}
    import yaml
    def includes(path):
        files.add(str(path.relative_to(ROOT)))
        raw=yaml.safe_load(path.read_text())
        for relative in raw.get('__include__',[]):includes((path.parent/relative).resolve())
    includes(config_path(name))
    report = dict(module=name,base='AQS-L2 R1',seed=0,initialization='original HGNet pretraining; detector/AQS trainable; no -r/-t',
        original_losses=True,resolved_config=cfg,source_sha256={p:digest(ROOT/p) for p in sorted(files) if p},
        annotation_sha256={s:digest(cfg[f'{s}_dataloader']['dataset']['ann_file']) for s in ('train','val')})
    with (Path(output)/'experiment_inputs.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False)


def parameters(checkpoint,output):
    import torch
    cp = torch.load(checkpoint,map_location='cpu',weights_only=True)
    report = dict(checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=digest(checkpoint),
        checkpoint_epoch_metadata=cp.get('last_epoch'),sources={},
        note='Parameter norms are not observed contributions; saved epoch metadata can be stale.')
    states = {'model':cp['model']}
    if 'ema' in cp: states['ema']=cp['ema']['module']
    prefix = 'decoder.behavior_refinement.'
    for source,state in states.items():
        settings = state[prefix+'settings'].tolist()
        raw = state[prefix+'raw_scale'].float()
        report['sources'][source] = dict(module=BEHAVIOR_REFINEMENTS[settings[1]],settings=settings,
            raw_scale=float(raw),effective_scale=float(raw.tanh()),
            norms={k.removeprefix(prefix):float(v.float().norm()) for k,v in state.items()
                   if k.startswith(prefix) and k != prefix+'settings'})
    with Path(output).open('x',encoding='utf-8') as stream: json.dump(report,stream,indent=2,ensure_ascii=False)
    return report


def on_off(name,output):
    output = Path(output)
    dirs = [output/'diagnostics_eval',output/f'{name}_off_eval']
    new,old = [json.loads((d/'analysis/summary.json').read_text()) for d in dirs]
    for key in ('checkpoint_sha256','weight_source','GT','images','queries','class_names','confidence_threshold','IoU_threshold'):
        if new[key] != old[key]: raise ValueError(f'ON/OFF protocol mismatch: {key}')
    if new['weight_source'] != 'ema' or gt_identity(dirs[0]/'analysis') != gt_identity(dirs[1]/'analysis'):
        raise ValueError('Use the same EMA and validation GT for ON/OFF')
    for folder,enabled in zip(dirs,(True,False)):
        meta = json.loads((folder/'evaluation/query_diagnostics/metadata.json').read_text())
        cfg = meta['config']; dec = cfg['DFINETransformer']
        if (not meta['complete'] or meta['checkpoint_sha256'] != new['checkpoint_sha256']
                or cfg.get('diagnostic_behavior_refinement_output_enabled') is not enabled
                or dec['behavior_refinement_module'] != name or not dec['use_aqs_refine'] or dec['aqs_apply_layer'] != 2):
            raise ValueError('ON/OFF metadata does not identify the actual module/AQS runtime')
    keys = ('saved_COCO_metrics','saved_COCO_per_class','saved_COCO_per_class_and_size',
            'validator_confusion','best_query_cross_classification','confusion_pairs')
    report = dict(module=name,checkpoint_sha256=new['checkpoint_sha256'],
        on={k:new.get(k) for k in keys},off={k:old.get(k) for k in keys},
        on_minus_off={k:difference(old[k],new[k]) if k in old and k in new else None for k in keys},
        note='Same jointly trained EMA, AQS stays ON; OFF is not separately trained AQS.',
        box_effect='query initialization can change boxes' if name=='QueryContrastInitialization' else 'final score path only; fixed-weight boxes must be unchanged')
    if name != 'QueryContrastInitialization':
        # Saved GT oracle geometry gives a second audit independent of aggregate AP.
        def geometry_rows(folder):
            with (folder/'analysis/gt_records.jsonl').open(encoding='utf-8') as stream:
                return {(r['image_id'],r['annotation_id']):r['best_query_IoU']
                        for r in (json.loads(line) for line in stream)}
        a,b = geometry_rows(dirs[0]),geometry_rows(dirs[1])
        if not a or a != b: raise ValueError('A score-only module unexpectedly changed fixed-weight geometry')
    with (output/f'{name}_on_off.json').open('x',encoding='utf-8') as stream: json.dump(report,stream,indent=2,ensure_ascii=False)
    print(f'{name}: same EMA ON-OFF ΔAP={100*(new["saved_COCO_metrics"]["AP"]-old["saved_COCO_metrics"]["AP"]):+.4f} pp',flush=True)


def pack(name,output):
    output = Path(output)
    archive = output/f'{name}_diagnostics.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.rglob('*')):
            if (path.is_file() and path.suffix in ('.json','.jsonl','.txt','.log','.csv','.png')
                    and 'raw_queries' not in path.parts):
                z.write(path,path.relative_to(output))
    print(f'诊断压缩包：{archive}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('preflight','snapshot','parameters','on-off','compare','pack'))
    parser.add_argument('--modules',nargs='+',choices=BEHAVIOR_REFINEMENTS)
    parser.add_argument('--module',choices=BEHAVIOR_REFINEMENTS)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--label',choices=('AQS_L2_R1','BaselineRun2'))
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    if args.operation=='preflight': preflight(args.modules,args.dry_run)
    elif args.operation=='snapshot': snapshot(args.module,args.output)
    elif args.operation=='parameters': parameters(args.checkpoint,args.output)
    elif args.operation=='on-off': on_off(args.module,args.output)
    elif args.operation=='compare': compare(args.output,args.reference,args.label)
    else: pack(args.module,args.output)


if __name__=='__main__': main()
