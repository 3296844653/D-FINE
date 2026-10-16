"""Run independent BoxLocal-L2 extensions; default exports best-EMA confusion only.

Dry-run is stdlib-only: no CUDA, dataset, conda, output mkdir or training.
Full query export/analysis is opt-in. No synthetic suite or ON/OFF runs here.
"""
import argparse
import copy
import hashlib
import inspect
import json
import os
import re
import shlex
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.diagnostics.boxlocal_five_spec import (
    BASELINE_AP, BASELINE_WRITE_AP, BOX_BASELINE, VARIANTS, config_path, output_name,
)

DEFAULT_ROOT = '/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw'


def validate_config(tag):
    from src.core.yaml_utils import load_config
    from src.zoo.dfine.dfine import DFINE
    from src.zoo.dfine.box_local_extensions import BOX_LOCAL_EXTENSIONS
    assert 'box_local_extension' in inspect.signature(DFINE).parameters, '五组扩展源码未同步'
    assert VARIANTS[tag][0] in BOX_LOCAL_EXTENSIONS
    actual = load_config(str(ROOT/config_path(tag)), cfg={})
    expected = copy.deepcopy(load_config(str(ROOT/BOX_BASELINE), cfg={}))
    expected['DFINE']['box_local_extension'] = VARIANTS[tag][0]
    stripped = copy.deepcopy(actual)
    for item in (expected, stripped):
        item.pop('__include__', None); item.pop('output_dir', None)
    if expected != stripped:
        raise ValueError(f'{tag}: 配置不止改变一个扩展，或偏离原BoxLocal-L2/损失/数据/训练设置')
    if (actual['epochs'] != 110 or actual['num_classes'] != 3
            or actual['train_dataloader']['collate_fn']['stop_epoch'] != 100
            or actual['train_dataloader']['dataset']['transforms']['policy']['epoch'] != 100
            or any(actual[f'{split}_dataloader']['total_batch_size'] != 32 for split in ('train', 'val'))):
        raise ValueError('需要110/100轮、三类别、batch32的原始协议')
    return actual


def validate_data(config):
    result = {}
    for split in ('train', 'val'):
        dataset = config[f'{split}_dataloader']['dataset']
        folder, annotation = Path(dataset['img_folder']), Path(dataset['ann_file'])
        raw = annotation.read_bytes(); data = json.loads(raw)
        if sorted((c['id'], c['name']) for c in data['categories']) != [(0, 'hand-raising'), (1, 'read'), (2, 'write')]:
            raise ValueError(f'{split}: 类别顺序与原始SCB-S不一致')
        if not folder.is_dir() or not data['images']:
            raise ValueError(f'{split}: 数据路径不存在或图片列表为空')
        for image in data['images']:
            if not (folder/image['file_name']).is_file():
                raise ValueError(f'{split}: 缺少图片 {image["file_name"]}')
        result[split] = dict(annotation_sha256=hashlib.sha256(raw).hexdigest(), images=len(data['images']),
                             GT=len(data['annotations']))
    return result


def command(config, output, args, *, checkpoint=None):
    cmd = ['torchrun', '--master_addr=127.0.0.1', f'--master_port={args.port}', '--nproc_per_node=1',
           'train.py', '-c', config, '--output-dir', str(output), '--use-amp', f'--seed={args.seed}']
    if checkpoint is not None:
        cmd += ['-r', str(checkpoint), '--test-only', '-u', 'export_confusion_matrix=true',
                f'export_query_diagnostics={str(args.diagnostics).lower()}',
                'query_diagnostic_conf_thresh=0.5', 'query_diagnostic_iou_thresh=0.5']
    return cmd


def ensure_new(path):
    if path.exists() or path.is_symlink():
        raise ValueError(f'保留已有结果，拒绝覆盖：{path}')


def run_logged(cmd, logfile, env):
    print(shlex.join(cmd), flush=True)
    with logfile.open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                stream.write(line); stream.flush(); print(line, end='', flush=True)
            status = process.wait()
        except BaseException:
            process.terminate()  # This runner's own child only.
            process.wait()
            raise
    if status:
        raise RuntimeError(f'命令退出码{status}，保留日志：{logfile}')


def snapshot(tag, output, config, args, data):
    import yaml
    files = {p for p in (ROOT/'src/zoo/dfine').glob('*.py')}
    files.update((ROOT/'src/nn/backbone').glob('*.py'))
    files.update(ROOT/name for name in ['train.py', __file__, 'tools/diagnostics/boxlocal_five_spec.py',
                 'tools/diagnostics/run_scbs_boxlocal_five_110.sh'])
    def includes(file):
        files.add(file)
        for child in yaml.safe_load(file.read_text()).get('__include__', []):
            includes((file.parent/child).resolve())
    includes(ROOT/config_path(tag))
    payload = dict(tag=tag, module=VARIANTS[tag][0], placement=VARIANTS[tag][1], seed=args.seed,
        baseline='default BoxLocal-L2, dim64/grid3/offset0.15/residual0.10, without AQS',
        initialization='fresh joint training; original HGNet pretraining; no -r/-t',
        resolved_config=config, data=data,
        source_sha256={str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest()
                       for file in sorted(files)},
        commands=dict(train=command(config_path(tag), output, args)))
    (output/'experiment_inputs.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    (output/'experiment_config.yml').write_text((ROOT/config_path(tag)).read_text())


def collect_result(tag, output, checkpoint):
    evaluation = output/'diagnostics_eval/evaluation'
    metrics = json.loads((evaluation/'validation_metrics.json').read_text())
    matrix = json.loads((evaluation/'confusion_matrix/confusion_matrix.json').read_text())
    counts = matrix['matrix_counts']; tp = sum(counts[i][i] for i in range(3))
    predicted = sum(sum(row[:3]) for row in counts); gt = sum(sum(row) for row in counts[:3])
    p, r = tp/max(predicted, 1), tp/max(gt, 1)
    ap = metrics['coco_eval_bbox'][0]; write = metrics['bbox_per_class']['write']['AP']
    console = (output/'console.log').read_text(errors='replace')
    profile = re.findall(r'Model FLOPs:([\d.]+) GFLOPS\s+MACs:[\d.]+ GMACs\s+Params:(\d+)', console)
    return dict(tag=tag, module=VARIANTS[tag][0], checkpoint=str(checkpoint), metrics=metrics,
        AP=ap, AP50=metrics['coco_eval_bbox'][1], write_AP=write, precision=p, recall=r,
        F1=2*p*r/max(p+r, 1e-12), delta_AP_pp=100*(ap-BASELINE_AP),
        delta_write_AP_pp=100*(write-BASELINE_WRITE_AP),
        write_to_read=counts[2][1], read_to_write=counts[1][2], confusion=matrix,
        FLOPs_G=float(profile[-1][0]) if profile else None,
        params=int(profile[-1][1]) if profile else None)


def write_summary(batch, results, selected, status):
    (batch/'results.json').write_text(json.dumps(dict(status=status, selected=selected, results=results), ensure_ascii=False, indent=2))
    text = ['# BoxLocal-L2 五组独立实验', '',
        '基线：mAP50:95=55.6658%，write AP=49.1097%。P/R/F1：conf0.5、IoU0.5，微平均。', '',
        '| 模型 | mAP50 | mAP50:95 | Recall | Precision | F1 | write AP | ΔAP(pp) | write→read | read→write |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for result in results:
        values = ' | '.join(f'{result[key]*100:.2f}' for key in ['AP50', 'AP', 'recall', 'precision', 'F1', 'write_AP'])
        text.append(f'| {result["tag"]} | {values} | {result["delta_AP_pp"]:+.2f} | {result["write_to_read"]} | {result["read_to_write"]} |')
    (batch/'summary.md').write_text('\n'.join(text)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('selection', nargs='?', default='all', choices=['all', *VARIANTS])
    parser.add_argument('run', nargs='?', default='run1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wait-gpu', action='store_true')
    parser.add_argument('--diagnostics', action='store_true', help='额外导出完整query数据并离线分析；默认仅最佳EMA指标/混淆矩阵')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--root', default=os.environ.get('SCBS_BOXL_ROOT', DEFAULT_ROOT))
    parser.add_argument('--gpu', type=int, default=int(os.environ.get('SCBS_BOXL_GPU', '0')))
    parser.add_argument('--port', type=int, default=int(os.environ.get('SCBS_BOXL_PORT', '7783')))
    args = parser.parse_args(argv)
    if not re.fullmatch(r'run[1-9][0-9]*', args.run) or args.seed < 0 or args.gpu < 0 or not 1024 <= args.port <= 65535:
        parser.error('需要run1/run2等编号、非负seed/GPU及1024..65535端口')
    root = Path(args.root)
    if not root.is_absolute():
        parser.error('输出根目录必须为绝对路径')
    selected = list(VARIANTS) if args.selection == 'all' else [args.selection]
    outputs = {tag: root/output_name(tag, args.seed, args.run) for tag in selected}
    batch = root/f'BoxLocalFive_epochs110_seed{args.seed}_{args.run}_{args.selection}'
    for output in [batch, *outputs.values()]:
        ensure_new(output)
    for tag in selected:
        if not (ROOT/config_path(tag)).is_file():
            raise ValueError(f'缺少配置：{config_path(tag)}')
    print('BoxLocal-L2固定基线；每组只新增一个模块；五组独立，AQS关闭。', flush=True)
    for tag in selected:
        print(f'{tag}: {VARIANTS[tag][0]} / {VARIANTS[tag][1]}\n输出：{outputs[tag]}', flush=True)
        if args.dry_run:
            print('训练：'+shlex.join(command(config_path(tag), outputs[tag], args)), flush=True)
            print('训练后自动评估：'+shlex.join(command(config_path(tag), outputs[tag]/'diagnostics_eval/evaluation', args,
                                                    checkpoint=outputs[tag]/'best_stg2.pth')), flush=True)
    if args.dry_run:
        print('仅演练：不加载PyTorch、不占GPU、不创建目录、不连接服务器；不执行ON/OFF。', flush=True)
        return
    configs = {tag: validate_config(tag) for tag in selected}
    os.chdir(ROOT)
    data = validate_data(configs[selected[0]])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    if not torch.cuda.is_available():
        raise ValueError('CUDA不可用，不启动CPU完整训练')
    from tools.diagnostics.wait_for_gpu import wait_for_gpu
    env = dict(os.environ, PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
    results = []
    try:
        for tag in selected:
            wait_for_gpu(args.gpu, wait=args.wait_gpu, idle_checks=1)
            # Waiting may be long. Re-read before launching, rather than record
            # a stale resolved config/data snapshot from queue start.
            configs[tag] = validate_config(tag)
            current_data = validate_data(configs[tag])
            if current_data != data:
                raise ValueError('等待/队列运行期间数据发生变化；停止，避免五组使用不同数据')
            with socket.socket() as port:
                port.bind(('127.0.0.1', args.port))
            if not batch.exists():
                batch.mkdir(parents=True, exist_ok=False)
            output = outputs[tag]; output.mkdir(parents=True, exist_ok=False)
            snapshot(tag, output, configs[tag], args, data)
            write_summary(batch, results, selected, f'training {tag}')
            run_logged(command(config_path(tag), output, args), output/'console.log', env)
            checkpoint = next((p for p in [output/'best_stg2.pth', output/'best_stg1.pth'] if p.is_file() and p.stat().st_size), None)
            if checkpoint is None:
                raise ValueError(f'{tag}: 训练未产生最佳权重，保留输出目录')
            evaluation = output/'diagnostics_eval/evaluation'
            run_logged(command(config_path(tag), evaluation, args, checkpoint=checkpoint), output/'evaluation_console.log', env)
            if args.diagnostics:
                run_logged([sys.executable, 'tools/diagnostics/analyze_detection_diagnostics.py',
                            '--input-dir', str(evaluation), '--output-dir', str(output/'diagnostics_eval/analysis')],
                           output/'analysis_console.log', env)
            results.append(collect_result(tag, output, checkpoint))
            write_summary(batch, results, selected, f'completed {tag}')
        write_summary(batch, results, selected, 'complete')
    except BaseException as error:
        if batch.exists():
            write_summary(batch, results, selected, f'incomplete: {type(error).__name__}: {error}')
        raise
    print(f'全部所选实验完成：{batch}/summary.md', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        print(f'错误：{error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
