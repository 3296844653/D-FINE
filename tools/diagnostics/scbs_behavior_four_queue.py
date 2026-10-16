"""Four independent AQS + architecture runs, streamed logs and same-EMA diagnostics."""
import argparse
import csv
import json
import os
import re
import signal
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
MODULES=('ClassEvidenceTransport','QueryContrastInitialization','GeometryAwareQueryCompetition','PrototypeBoundaryRefinement')


def write_status(path,**data):
    data['updated_utc']=datetime.now(timezone.utc).isoformat()
    temporary=path.with_suffix('.pending')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)


def stream(command,log,env):
    """Only this queue's new process group is interrupted if the queue is cancelled."""
    print('执行：',' '.join(str(x) for x in command),flush=True)
    with log.open('x',encoding='utf-8') as output:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
        try:
            for line in proc.stdout:
                print(line,end='',flush=True);output.write(line);output.flush()
            result=proc.wait()
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGINT)
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
        finally:
            proc.stdout.close()
    if result:raise subprocess.CalledProcessError(result,command)


def completed_training(out):
    records=[json.loads(line) for line in (out/'log.txt').read_text().splitlines() if line.strip()]
    if [r['epoch'] for r in records] != list(range(110)):
        raise RuntimeError('Training log must contain exactly epochs 0..109; incomplete runs are not final results')
    best=max(records[100:],key=lambda r:r['test_coco_eval_bbox'][0])
    if not (out/'best_stg2.pth').is_file():raise FileNotFoundError('Missing best_stg2.pth')
    return dict(best_epoch=best['epoch'],AP=best['test_coco_eval_bbox'][0])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',nargs='?',default='run1')
    parser.add_argument('--modules',nargs='+',choices=MODULES,default=list(MODULES))
    parser.add_argument('--wait-gpu',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--export-only',action='store_true',help='Use existing completed training; never retrain or overwrite diagnostics')
    parser.add_argument('--root',type=Path,default=Path(os.environ.get('SCBS_BEHAVIOR_ROOT','/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw')))
    parser.add_argument('--gpu',type=int,default=int(os.environ.get('SCBS_BEHAVIOR_GPU','0')))
    parser.add_argument('--port',type=int,default=int(os.environ.get('SCBS_BEHAVIOR_PORT','7781')))
    args=parser.parse_args()
    if (not re.fullmatch(r'run[1-9][0-9]*',args.run) or not args.root.is_absolute() or args.gpu<0
            or not 1024<=args.port<=65535 or len(set(args.modules))!=len(args.modules)):
        parser.error('Invalid run number, root, GPU/port, or repeated module')
    outputs={name:args.root/f'dfine_s_scbs_hrw_epochs110_AQS_{name}_seed0_{args.run}' for name in args.modules}
    suffix='' if tuple(args.modules)==MODULES else '_'+ '_'.join(args.modules)
    queue=args.root/f'behavior_four_epochs110_seed0_{args.run}{suffix}{"_export" if args.export_only else ""}'
    def check_outputs():
        if queue.exists() or queue.is_symlink():raise FileExistsError(f'Queue already exists; preserved: {queue}')
        for name,out in outputs.items():
            cfg=ROOT/f'configs/dfine/dfine_s_scbs_hrw_110_AQS_{name}.yml'
            if not cfg.is_file():raise FileNotFoundError(cfg)
            if args.export_only:
                completed_training(out)
                for path in (out/'diagnostics_eval',out/f'{name}_off_eval',out/f'{name}_diagnostics.zip'):
                    if path.exists() or path.is_symlink():raise FileExistsError(f'Existing diagnostic preserved: {path}')
            elif out.exists() or out.is_symlink():raise FileExistsError(f'Existing result preserved: {out}')
            print(f'{name}: {out}',flush=True)
    check_outputs()
    if args.dry_run:
        print('仅演练；四组独立、顺序执行。未创建输出、未启动训练。',flush=True);return
    from tools.diagnostics.wait_for_gpu import wait_for_gpu
    from tools.diagnostics.scbs_behavior_refinement_experiment import preflight,snapshot,parameters,on_off,pack,compare
    from tools.diagnostics.scbs_instance_evidence_experiment import digest
    wait_for_gpu(args.gpu,wait=args.wait_gpu)
    check_outputs()
    with socket.socket() as sock:sock.bind(('0.0.0.0',args.port))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),PYTHONUNBUFFERED='1',MPLBACKEND='Agg')
    preflight(args.modules)
    # Tests must succeed before a training output directory is created.
    subprocess.run([sys.executable,'-u','tools/diagnostics/test_behavior_refinement.py'],cwd=ROOT,env=env,check=True)
    wait_for_gpu(args.gpu,wait=args.wait_gpu)
    check_outputs()
    queue.mkdir(parents=True,exist_ok=False)
    status=queue/'status.json';completed=[];current=None;stage='starting'
    launch=[sys.executable,'-u','-m','torch.distributed.run',f'--master_port={args.port}','--nproc_per_node=1']
    def record(state):write_status(status,state=state,module=current,stage=stage,completed=completed,outputs={k:str(v) for k,v in outputs.items()})
    # SIGTERM to the queue does not leave its own training process orphaned.
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for name,out in outputs.items():
            current=name;stage='training' if not args.export_only else 'exporting';record('running')
            wait_for_gpu(args.gpu,wait=args.wait_gpu)
            cfg=f'configs/dfine/dfine_s_scbs_hrw_110_AQS_{name}.yml'
            if not args.export_only:
                out.mkdir(parents=True,exist_ok=False);snapshot(name,out)
                stream([*launch,'train.py','-c',cfg,'--output-dir',str(out),'--use-amp','--seed=0'],out/'console.log',env)
            best=completed_training(out);cp=out/'best_stg2.pth'
            # A source change during training invalidates provenance rather than silently continuing.
            provenance=json.loads((out/'experiment_inputs.json').read_text())
            for file,sha in provenance['source_sha256'].items():
                if digest(ROOT/file)!=sha:raise RuntimeError(f'Source changed during this experiment: {file}')
            for split,sha in provenance['annotation_sha256'].items():
                path=provenance['resolved_config'][f'{split}_dataloader']['dataset']['ann_file']
                if digest(path)!=sha:raise RuntimeError(f'Annotation changed during this experiment: {split}')
            parameter_file=out/f'best_stg2_{name}_parameters.json'
            if not parameter_file.exists():parameters(cp,parameter_file)
            elif json.loads(parameter_file.read_text())['checkpoint_sha256']!=digest(cp):raise RuntimeError('Parameter report checkpoint mismatch')
            aqs_file=out/'best_stg2_aqs_parameters.json'
            if not aqs_file.exists():
                stream([sys.executable,'-u','tools/diagnostics/export_aqs_parameters.py','--checkpoint',str(cp),'--temperature','.1'],out/'aqs_parameters_console.log',env)
            for mode in ('on','off'):
                stage=f'{mode}_diagnostics';record('running');wait_for_gpu(args.gpu,wait=args.wait_gpu)
                diag=out/'diagnostics_eval' if mode=='on' else out/f'{name}_off_eval'
                stream([*launch,'tools/diagnostics/eval_behavior_refinement.py','--mode',mode,
                    '-c',cfg,'-r',str(cp),'--test-only','--output-dir',str(diag/'evaluation'),'--use-amp','--seed=0',
                    '-u','export_confusion_matrix=true','export_query_diagnostics=true',
                    'query_diagnostic_conf_thresh=0.5','query_diagnostic_iou_thresh=0.5','query_diagnostic_neighbor_iou_thresh=0.8',
                    f'diagnostic_behavior_refinement_output_enabled={str(mode=="on").lower()}',
                    f'diagnostic_behavior_refinement_ablation={mode}'],out/f'{mode}_diagnostics_console.log',env)
                stream([sys.executable,'-u','tools/diagnostics/analyze_detection_diagnostics.py',
                    '--input-dir',str(diag/'evaluation'),'--output-dir',str(diag/'analysis')],out/f'{mode}_analysis_console.log',env)
            on_off(name,out)
            summary=json.loads((out/'diagnostics_eval/analysis/summary.json').read_text())
            if abs(summary['saved_COCO_metrics']['AP']-best['AP'])>.0005:
                raise RuntimeError('ON diagnostic AP differs from stage2 best training AP by >0.05 percentage points')
            stage='comparison';record('running')
            references={
                'AQS_L2_R1':Path(os.environ.get('SCBS_BEHAVIOR_AQS_REFERENCE',str(args.root/'dfine_s_scbs_hrw_epochs110_aqs_layer2_r010_seed0_run1/diagnostics_eval'))),
                'BaselineRun2':Path(os.environ.get('SCBS_BEHAVIOR_RUN2_REFERENCE',str(args.root/'dfine_s_scbs_hrw_epochs110_seed0_run2/diagnostics_eval')))}
            comparison={}
            for label,reference in references.items():
                try:
                    compare(out,reference,label);comparison[label]=dict(state='compared',reference=str(reference))
                except (FileNotFoundError,AssertionError,KeyError,ValueError) as error:
                    comparison[label]=dict(state='missing_or_incompatible',reference=str(reference),reason=str(error))
                    print(f'{label}未比较：{error}；保留完整新诊断，不伪造对照。',flush=True)
            with (out/'comparison_status.json').open('x',encoding='utf-8') as handle:json.dump(comparison,handle,ensure_ascii=False,indent=2)
            pack(name,out)
            completed.append({'module':name,**best,**summary['saved_COCO_metrics'],
                'per_class':summary['saved_COCO_per_class'],'output':str(out)})
            record('running')
            with (queue/'results.csv').open('w',encoding='utf-8',newline='') as handle:
                fields=['module','best_epoch','AP','AP50','AP75','APm','APl','output']
                writer=csv.DictWriter(handle,fields,extrasaction='ignore');writer.writeheader();writer.writerows(completed)
        stage='complete';record('complete');print('四组所选实验和诊断全部完成：',queue,flush=True)
    except BaseException as error:
        record('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed')
        print(f'队列停止，保留已有结果：{error}',file=sys.stderr,flush=True);raise


if __name__=='__main__':main()
