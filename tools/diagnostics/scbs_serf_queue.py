"""Two independent SERF placements, sequential training and same-EMA diagnostics."""
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
sys.path.insert(0, str(ROOT))
MODULES = ('SERF_Encoder', 'SERF_Backbone')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', nargs='?', default='run1')
    parser.add_argument('--modules', nargs='+', choices=MODULES, default=list(MODULES))
    parser.add_argument('--wait-gpu', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('SCBS_SERF_ROOT', '/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw')))
    parser.add_argument('--gpu', type=int, default=int(os.environ.get('SCBS_SERF_GPU', '0')))
    parser.add_argument('--port', type=int, default=int(os.environ.get('SCBS_SERF_PORT', '7781')))
    args = parser.parse_args()
    if (not re.fullmatch(r'run[1-9][0-9]*', args.run) or not args.root.is_absolute()
            or args.gpu < 0 or not 1024 <= args.port <= 65535 or len(set(args.modules)) != len(args.modules)):
        parser.error('Invalid run number/root/GPU/port or duplicate placement')
    outputs = {name: args.root / f'dfine_s_scbs_hrw_epochs110_AQS_{name}_seed0_{args.run}' for name in args.modules}
    subset = '' if tuple(args.modules) == MODULES else '_' + '_'.join(args.modules)
    queue = args.root / f'AQS_SERF_two_epochs110_seed0_{args.run}{subset}{"_export" if args.export_only else ""}'

    def check_outputs():
        if queue.exists() or queue.is_symlink():
            raise FileExistsError(f'Existing queue preserved: {queue}')
        for name, out in outputs.items():
            cfg = ROOT / f'configs/dfine/dfine_s_scbs_hrw_110_AQS_{name}.yml'
            if not cfg.is_file():
                raise FileNotFoundError(cfg)
            if args.export_only:
                for p in (out / 'diagnostics_eval', out / f'{name}_off_eval', out / f'{name}_diagnostics.zip'):
                    if p.exists() or p.is_symlink():
                        raise FileExistsError(f'Existing diagnostic preserved: {p}')
                if not (out / 'experiment_inputs.json').is_file():
                    raise FileNotFoundError('Export requires the original source/config snapshot')
            elif out.exists() or out.is_symlink():
                raise FileExistsError(f'Existing result preserved: {out}')
            print(f'{name}: {out}', flush=True)

    check_outputs()
    if args.dry_run:
        print('仅演练：两组独立顺序训练，未创建目录、未占用GPU。', flush=True)
        return
    from tools.diagnostics.wait_for_gpu import wait_for_gpu
    from tools.diagnostics.scbs_behavior_four_queue import stream, completed_training, write_status
    from tools.diagnostics.scbs_serf_experiment import preflight, snapshot, parameters, on_off, compare, pack
    from tools.diagnostics.scbs_instance_evidence_experiment import digest
    if args.export_only:
        for out in outputs.values():
            completed_training(out)
    wait_for_gpu(args.gpu, wait=args.wait_gpu)
    check_outputs()
    with socket.socket() as sock:
        sock.bind(('0.0.0.0', args.port))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
    preflight(args.modules)
    # Real effect is assessed by training. These tests only check correctness.
    subprocess.run([sys.executable, '-u', 'tools/diagnostics/test_serf.py'], cwd=ROOT, env=env, check=True)
    wait_for_gpu(args.gpu, wait=args.wait_gpu)
    check_outputs()
    queue.mkdir(parents=True, exist_ok=False)
    status = queue / 'status.json'
    completed, current, stage = [], None, 'starting'
    launch = [sys.executable, '-u', '-m', 'torch.distributed.run', f'--master_port={args.port}', '--nproc_per_node=1']

    def record(state):
        write_status(status, state=state, module=current, stage=stage, completed=completed,
                     outputs={k: str(v) for k, v in outputs.items()})
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for name, out in outputs.items():
            current, stage = name, 'waiting'; record('running')
            wait_for_gpu(args.gpu, wait=args.wait_gpu)
            with socket.socket() as sock:
                sock.bind(('0.0.0.0', args.port))
            cfg = f'configs/dfine/dfine_s_scbs_hrw_110_AQS_{name}.yml'
            if not args.export_only:
                out.mkdir(parents=True, exist_ok=False)
                snapshot(name, out)
                stage = 'training'; record('running')
                stream([*launch, 'train.py', '-c', cfg, '--output-dir', str(out), '--use-amp', '--seed=0'], out / 'console.log', env)
            best = completed_training(out)
            cp = out / 'best_stg2.pth'
            provenance = json.loads((out / 'experiment_inputs.json').read_text())
            for file, sha in provenance['source_sha256'].items():
                if digest(ROOT / file) != sha:
                    raise RuntimeError(f'Source changed during experiment: {file}')
            for split, sha in provenance['annotation_sha256'].items():
                annotation = provenance['resolved_config'][f'{split}_dataloader']['dataset']['ann_file']
                if digest(annotation) != sha:
                    raise RuntimeError(f'Annotation changed: {split}')
            report = out / f'best_stg2_{name}_parameters.json'
            if not report.exists(): parameters(cp, report)
            elif json.loads(report.read_text())['checkpoint_sha256'] != digest(cp):
                raise RuntimeError('Parameter report checkpoint mismatch')
            if not (out / 'best_stg2_aqs_parameters.json').exists():
                stream([sys.executable, '-u', 'tools/diagnostics/export_aqs_parameters.py',
                        '--checkpoint', str(cp), '--temperature', '.1'], out / 'aqs_parameters_console.log', env)
            for mode in ('on', 'off'):
                stage = f'{mode}_diagnostics'; record('running')
                wait_for_gpu(args.gpu, wait=args.wait_gpu)
                diag = out / 'diagnostics_eval' if mode == 'on' else out / f'{name}_off_eval'
                stream([*launch, 'tools/diagnostics/eval_serf.py', '--mode', mode,
                    '-c', cfg, '-r', str(cp), '--test-only', '--output-dir', str(diag / 'evaluation'), '--use-amp', '--seed=0',
                    '-u', 'export_confusion_matrix=true', 'export_query_diagnostics=true',
                    'query_diagnostic_conf_thresh=0.5', 'query_diagnostic_iou_thresh=0.5', 'query_diagnostic_neighbor_iou_thresh=0.8',
                    f'diagnostic_serf_output_enabled={str(mode == "on").lower()}',
                    f'diagnostic_serf_ablation=serf_{mode}'], out / f'{mode}_diagnostics_console.log', env)
                stream([sys.executable, '-u', 'tools/diagnostics/analyze_detection_diagnostics.py',
                    '--input-dir', str(diag / 'evaluation'), '--output-dir', str(diag / 'analysis')], out / f'{mode}_analysis_console.log', env)
            on_off(name, out)
            summary = json.loads((out / 'diagnostics_eval/analysis/summary.json').read_text())
            if abs(summary['saved_COCO_metrics']['AP'] - best['AP']) > .0005:
                raise RuntimeError('ON diagnostic AP differs from stage2 best by >0.05 percentage points')
            stage = 'comparison'; record('running')
            references = {
                'AQS_L2_R1': Path(os.environ.get('SCBS_SERF_AQS_REFERENCE', str(args.root / 'dfine_s_scbs_hrw_epochs110_aqs_layer2_r010_seed0_run1/diagnostics_eval'))),
                'BaselineRun2': Path(os.environ.get('SCBS_SERF_RUN2_REFERENCE', str(args.root / 'dfine_s_scbs_hrw_epochs110_seed0_run2/diagnostics_eval')))}
            comparison = {}
            for label, reference in references.items():
                try:
                    compare(out, reference, label)
                    comparison[label] = dict(state='compared', reference=str(reference))
                except (FileNotFoundError, AssertionError, KeyError, ValueError) as error:
                    comparison[label] = dict(state='missing_or_incompatible', reference=str(reference), reason=str(error))
                    print(f'{label}未比较：{error}；保留诊断，不伪造对照。', flush=True)
            with (out / 'comparison_status.json').open('x', encoding='utf-8') as handle:
                json.dump(comparison, handle, ensure_ascii=False, indent=2)
            pack(name, out)
            metrics = summary['saved_COCO_metrics']
            completed.append({'module': name, **best, **metrics,
                'APm': metrics['AP_medium'], 'APl': metrics['AP_large'],
                'per_class': summary['saved_COCO_per_class'], 'output': str(out)})
            record('running')
            with (queue / 'results.csv').open('w', encoding='utf-8', newline='') as handle:
                fields = ['module', 'best_epoch', 'AP', 'AP50', 'AP75', 'APm', 'APl', 'output']
                writer = csv.DictWriter(handle, fields, extrasaction='ignore')
                writer.writeheader(); writer.writerows(completed)
        stage = 'complete'; record('complete')
        print('所选SERF位置训练和诊断全部完成：', queue, flush=True)
    except BaseException as error:
        record('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed')
        print(f'队列停止，已有结果保留：{error}', file=sys.stderr, flush=True)
        raise


if __name__ == '__main__':
    main()
