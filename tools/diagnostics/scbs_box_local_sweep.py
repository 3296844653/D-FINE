"""Standalone BoxLocal-L2 one-factor sweep; first two residual groups by default."""
import argparse
import csv
import json
import os
import re
import signal
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.diagnostics.box_local_sweep_spec import PROFILES, FIRST_GROUPS, settings, module_name


def first_existing(paths):
    return next((p for p in paths if (p/'analysis/summary.json').is_file()), paths[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',nargs='?',default='run1')
    selection = p.add_mutually_exclusive_group()
    selection.add_argument('--groups',nargs='+',choices=PROFILES)
    selection.add_argument('--all',action='store_true')
    p.add_argument('--wait-gpu',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--export-only',action='store_true')
    p.add_argument('--root',type=Path,default=Path(os.environ.get('SCBS_BOX_ROOT','/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw')))
    p.add_argument('--gpu',type=int,default=int(os.environ.get('SCBS_BOX_GPU','0')))
    p.add_argument('--port',type=int,default=int(os.environ.get('SCBS_BOX_PORT','7781')))
    a = p.parse_args()
    groups = list(PROFILES) if a.all else (a.groups or list(FIRST_GROUPS))
    if (not re.fullmatch(r'run[1-9][0-9]*',a.run) or not a.root.is_absolute() or a.gpu < 0
            or not 1024 <= a.port <= 65535 or len(set(groups)) != len(groups)):
        p.error('Invalid run/root/GPU/port or duplicated group')
    outputs = {g:a.root/f'dfine_s_scbs_hrw_epochs110_{module_name(g)}_seed0_{a.run}' for g in groups}
    suffix = 'all' if groups == list(PROFILES) else '_'.join(groups)
    queue = a.root/f'BoxLocalQueryAdapter_l2_sweep_epochs110_seed0_{a.run}_{suffix}{"_export" if a.export_only else ""}'

    def protected():
        if queue.exists() or queue.is_symlink():raise FileExistsError(f'Existing queue preserved: {queue}')
        for group,out in outputs.items():
            cfg = ROOT/f'configs/dfine/dfine_s_scbs_hrw_110_{module_name(group)}.yml'
            if not cfg.is_file():raise FileNotFoundError(cfg)
            if a.export_only:
                if not (out/'experiment_inputs.json').is_file():raise FileNotFoundError('Export requires original config/source snapshot')
                for path in (out/'diagnostics_eval',out/f'{module_name(group)}_off_eval',out/f'{module_name(group)}_diagnostics.zip'):
                    if path.exists() or path.is_symlink():raise FileExistsError(f'Existing diagnostics preserved: {path}')
            elif out.exists() or out.is_symlink():raise FileExistsError(f'Existing output preserved: {out}; select unrun groups or a new run ID')
            print(f'{group}: {settings(group)}\n  输出：{out}',flush=True)

    protected()
    if a.dry_run:
        print('仅演练；未创建目录、未启动训练。默认两组；--all选择全部9组。',flush=True);return
    from tools.diagnostics.wait_for_gpu import wait_for_gpu
    from tools.diagnostics.scbs_behavior_four_queue import stream,write_status,completed_training
    from tools.diagnostics.scbs_box_local_sweep_experiment import preflight,snapshot,parameters,on_off,compare,pack
    from tools.diagnostics.scbs_instance_evidence_experiment import digest
    if a.export_only:
        for out in outputs.values():completed_training(out)
    wait_for_gpu(a.gpu,wait=a.wait_gpu);protected()
    with socket.socket() as sock:sock.bind(('0.0.0.0',a.port))
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
    preflight(groups)
    print('开始合成/导出链路检查（不是精度实验）；通过后才启动完整训练。',flush=True)
    subprocess.run([sys.executable,'-u','tools/diagnostics/test_box_local_sweep.py'],cwd=ROOT,env=env,check=True)
    wait_for_gpu(a.gpu,wait=a.wait_gpu);protected()
    queue.mkdir(parents=True,exist_ok=False)
    completed,current,stage = [],None,'starting'
    launch = [sys.executable,'-u','-m','torch.distributed.run',f'--master_port={a.port}','--nproc_per_node=1']
    def record(state):write_status(queue/'status.json',state=state,group=current,stage=stage,
        completed=completed,outputs={k:str(v) for k,v in outputs.items()})
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for group,out in outputs.items():
            name = module_name(group);current,stage = group,'waiting';record('running')
            wait_for_gpu(a.gpu,wait=a.wait_gpu)
            with socket.socket() as sock:sock.bind(('0.0.0.0',a.port))
            cfg = f'configs/dfine/dfine_s_scbs_hrw_110_{name}.yml'
            if not a.export_only:
                out.mkdir(parents=True,exist_ok=False);snapshot(group,out)
                stage = 'training';record('running')
                stream([*launch,'train.py','-c',cfg,'--output-dir',str(out),'--use-amp','--seed=0'],out/'console.log',env)
            best = completed_training(out);cp = out/'best_stg2.pth'
            provenance = json.loads((out/'experiment_inputs.json').read_text())
            for file,sha in provenance['source_sha256'].items():
                if digest(ROOT/file) != sha:raise RuntimeError('Source changed during run: '+file)
            for split,sha in provenance['annotation_sha256'].items():
                if digest(provenance['resolved_config'][f'{split}_dataloader']['dataset']['ann_file']) != sha:
                    raise RuntimeError('Annotation changed during run: '+split)
            report = out/f'best_stg2_{name}_parameters.json'
            if not report.exists():parameters(cp,report)
            elif json.loads(report.read_text())['checkpoint_sha256'] != digest(cp):raise RuntimeError('Parameter report checkpoint mismatch')
            for mode in ('on','off'):
                stage = mode+'_diagnostics';record('running');wait_for_gpu(a.gpu,wait=a.wait_gpu)
                diag = out/'diagnostics_eval' if mode=='on' else out/f'{name}_off_eval'
                stream([*launch,'tools/diagnostics/eval_box_local_sweep.py','--mode',mode,'-c',cfg,'-r',str(cp),
                    '--test-only','--output-dir',str(diag/'evaluation'),'--use-amp','--seed=0','-u',
                    'export_confusion_matrix=true','export_query_diagnostics=true','query_diagnostic_conf_thresh=0.5',
                    'query_diagnostic_iou_thresh=0.5','query_diagnostic_neighbor_iou_thresh=0.8',
                    f'diagnostic_box_local_output_enabled={str(mode=="on").lower()}',
                    f'diagnostic_box_local_ablation=box_{mode}'],out/f'{mode}_diagnostics_console.log',env)
                stream([sys.executable,'-u','tools/diagnostics/analyze_detection_diagnostics.py','--input-dir',str(diag/'evaluation'),
                    '--output-dir',str(diag/'analysis')],out/f'{mode}_analysis_console.log',env)
            on_off(group,out)
            summary = json.loads((out/'diagnostics_eval/analysis/summary.json').read_text())
            metrics = summary['saved_COCO_metrics']
            if abs(metrics['AP']-best['AP']) > .0005:raise RuntimeError('ON diagnostic AP differs from stage2 best by >0.05 pp')
            stage = 'comparison';record('running')
            references = {
                'BoxLocal_Default':Path(os.environ.get('SCBS_BOX_DEFAULT_REFERENCE',str(first_existing([
                    a.root/'dfine_s_scbs_hrw_epochs110_BoxLocalQueryAdapter_l2_seed0_run1/diagnostics_eval',
                    a.root/'dfine_s_scbs_hrw_epochs110_night_n4_seed0_run1/diagnostics_eval'])))),
                'BaselineRun2':Path(os.environ.get('SCBS_BOX_RUN2_REFERENCE',str(first_existing([
                    a.root/'dfine_s_scbs_hrw_epochs110_seed0_run2/diagnostics_eval',
                    a.root/'dfine_s_scbs_hrw_epochs110_seed0_run2/diagnostics_export1',
                    a.root/'dfine_s_scbs_hrw_epochs110_seed0_run2_diagnostics_export1']))))}
            comparisons = {}
            for label,reference in references.items():
                try:
                    compare(out,reference,label);comparisons[label] = dict(state='compared',reference=str(reference))
                except (FileNotFoundError,AssertionError,KeyError,ValueError) as error:
                    comparisons[label] = dict(state='missing_or_incompatible',reference=str(reference),reason=str(error))
                    print(f'{label}未比较：{error}；保留新诊断，不伪造对照。',flush=True)
            with (out/'comparison_status.json').open('x',encoding='utf-8') as f:json.dump(comparisons,f,ensure_ascii=False,indent=2)
            pack(name,out)
            per_class = summary['saved_COCO_per_class'];counts = summary['validator_confusion']
            pairs = {(r['gt'],r['prediction']):r['count'] for r in summary['confusion_pairs']}
            completed.append({'group':group,'module':name,**settings(group),**best,**metrics,
                'APm':metrics['AP_medium'],'APl':metrics['AP_large'],
                'hand_AP':per_class['0']['AP'],'read_AP':per_class['1']['AP'],'write_AP':per_class['2']['AP'],
                'write_to_read':pairs.get(('write','read'),0),'read_to_write':pairs.get(('read','write'),0),
                'unmatched_predictions':counts['unmatched_predictions'],
                'write_candidate_gap':summary['best_query_cross_classification']['by_class']['write']['no_high_correct_selected_candidate'],
                'output':str(out)})
            record('running')
            with (queue/'results.csv').open('w',encoding='utf-8',newline='') as f:
                fields = ['group','module',*settings(group),'best_epoch','AP','AP50','AP75','APm','APl','hand_AP','read_AP','write_AP',
                          'write_to_read','read_to_write','unmatched_predictions','write_candidate_gap','output']
                writer = csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(completed)
        stage = 'complete';record('complete');print('所选BoxLocal-L2参数组与诊断全部完成：',queue,flush=True)
    except BaseException as error:
        record('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed')
        print(f'队列停止，已有结果保留：{error}',file=sys.stderr,flush=True);raise


if __name__ == '__main__':main()
