"""Train the ORIGINAL GQC alone on plain D-FINE-S, then evaluate best EMA.

Dry-run uses stdlib only. No AQS, BoxLocal, RQC, synthetic pretraining tests or
ON/OFF evaluation. Query-level export is optional, metrics/confusion are default.
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
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CONFIG = 'configs/dfine/dfine_s_scbs_hrw_110_GeometryAwareQueryCompetition_only.yml'
BASELINE = 'configs/dfine/dfine_s_scbs_hrw_110.yml'
DEFAULT_ROOT = '/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw'
SETTINGS = dict(use_aqs_refine=False, allow_gqc_standalone=True,
    behavior_refinement_module='GeometryAwareQueryCompetition', behavior_refinement_dim=64,
    behavior_refinement_scale=.1, behavior_refinement_chunk=32, behavior_refinement_checkpoint=True)
BASELINE_AP = .5504686275416603
BASELINE_WRITE_AP = .47611529130649166


def validate_config():
    from src.core.yaml_utils import load_config
    from src.zoo.dfine.dfine_decoder import DFINETransformer
    if 'allow_gqc_standalone' not in inspect.signature(DFINETransformer).parameters:
        raise ValueError('服务器解码器未同步：缺少GQC单模块接口')
    actual = load_config(str(ROOT/CONFIG), cfg={})
    expected = copy.deepcopy(load_config(str(ROOT/BASELINE), cfg={}))
    expected['DFINETransformer'].update(SETTINGS)
    stripped = copy.deepcopy(actual)
    for config in (expected, stripped):
        config.pop('__include__', None); config.pop('output_dir', None)
    if expected != stripped or type(actual['DFINETransformer']['allow_gqc_standalone']) is not bool:
        raise ValueError('GQC单模块配置偏离原基线：不允许叠加AQS、BoxLocal或修改损失/训练设置')
    if (actual['epochs'] != 110 or actual['num_classes'] != 3 or not actual['use_ema']
            or actual['train_dataloader']['collate_fn']['stop_epoch'] != 100
            or actual['train_dataloader']['dataset']['transforms']['policy']['epoch'] != 100
            or any(actual[f'{split}_dataloader']['total_batch_size'] != 32 for split in ('train', 'val'))):
        raise ValueError('需要原110/100轮、三类、batch32、EMA协议')
    return actual


def validate_data(config):
    result = {}
    for split in ('train', 'val'):
        dataset = config[f'{split}_dataloader']['dataset']
        annotation, images = Path(dataset['ann_file']), Path(dataset['img_folder'])
        raw = annotation.read_bytes(); data = json.loads(raw)
        if sorted((c['id'], c['name']) for c in data['categories']) != [(0, 'hand-raising'), (1, 'read'), (2, 'write')]:
            raise ValueError(f'{split}: 类别顺序不符')
        if not images.is_dir() or not data['images']:
            raise ValueError(f'{split}: 图片路径不存在或数据为空')
        missing = [item['file_name'] for item in data['images'] if not (images/item['file_name']).is_file()]
        if missing:
            raise ValueError(f'{split}: 缺少图片，例如 {missing[:3]}')
        result[split] = dict(annotation=str(annotation.resolve()),
            annotation_sha256=hashlib.sha256(raw).hexdigest(), images=len(data['images']), GT=len(data['annotations']))
    return result


def output_name(seed, run):
    return f'dfine_s_scbs_hrw_epochs110_GQC_only_seed{seed}_{run}'


def command(output, args, checkpoint=None):
    cmd = ['torchrun', '--master_addr=127.0.0.1', f'--master_port={args.port}', '--nproc_per_node=1',
           'train.py', '-c', CONFIG, '--output-dir', str(output), '--use-amp', f'--seed={args.seed}']
    if checkpoint is not None:
        cmd += ['-r', str(checkpoint), '--test-only', '-u', 'export_confusion_matrix=true',
                f'export_query_diagnostics={str(args.diagnostics).lower()}',
                'query_diagnostic_conf_thresh=0.5', 'query_diagnostic_iou_thresh=0.5']
    return cmd


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
            process.terminate()  # Only this runner's own child.
            process.wait()
            raise
    if status:
        raise RuntimeError(f'命令退出码{status}，日志已保留：{logfile}')


def snapshot(output, config, args, data):
    import yaml
    files = set((ROOT/'src').rglob('*.py'))
    files.update(ROOT/name for name in ['train.py', 'tools/diagnostics/scbs_gqc_only.py',
        'tools/diagnostics/run_scbs_gqc_only_110.sh', 'tools/diagnostics/wait_for_gpu.py'])
    def include(file):
        if file in files:
            return
        files.add(file)
        for name in yaml.safe_load(file.read_text()).get('__include__', []):
            include((file.parent/name).resolve())
    include(ROOT/CONFIG)
    payload = dict(model='plain D-FINE-S + original GeometryAwareQueryCompetition',
        AQS=False, BoxLocal=False, RQC=False, seed=args.seed, data=data, resolved_config=config,
        initialization='fresh joint training, original HGNet pretraining; no detector -r/-t',
        source_sha256={str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest() for file in sorted(files)},
        train_command=command(output, args))
    (output/'experiment_inputs.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    (output/'experiment_config.yml').write_text((ROOT/CONFIG).read_text())


def completed_training(output):
    rows = [json.loads(line) for line in (output/'log.txt').read_text().splitlines() if line.strip()]
    epochs = {row['epoch'] for row in rows if 'epoch' in row}
    if not set(range(110)) <= epochs:
        raise ValueError('未记录完整0..109轮，不当作完整单模块实验；保留日志和权重')


def collect_result(output, checkpoint):
    evaluation = output/'diagnostics_eval/evaluation'
    metrics = json.loads((evaluation/'validation_metrics.json').read_text())
    matrix = json.loads((evaluation/'confusion_matrix/confusion_matrix.json').read_text())
    if matrix.get('row_labels', [])[:3] != ['hand-raising', 'read', 'write']:
        raise ValueError('混淆矩阵类别顺序不符')
    if matrix.get('confidence_threshold') != .5 or matrix.get('iou_threshold') != .5:
        raise ValueError('混淆矩阵阈值不是conf0.5/IoU0.5')
    if matrix.get('metadata', {}).get('weight_source') != 'ema':
        raise ValueError('预期最佳EMA评估，实际权重来源不符')
    counts = matrix['matrix_counts']; tp = sum(counts[i][i] for i in range(3))
    p = tp/max(sum(sum(row[:3]) for row in counts), 1)
    r = tp/max(sum(sum(row) for row in counts[:3]), 1)
    profile = re.findall(r'Model FLOPs:([\d.]+) GFLOPS\s+MACs:[\d.]+ GMACs\s+Params:(\d+)',
                         (output/'console.log').read_text(errors='replace'))
    ap = metrics['coco_eval_bbox'][0]; classes = metrics['bbox_per_class']
    return dict(checkpoint=str(checkpoint), weight_source='ema', metrics=metrics, confusion=matrix,
        AP=ap, AP50=metrics['coco_eval_bbox'][1], class_AP={name: classes[name]['AP'] for name in ('hand-raising', 'read', 'write')},
        precision=p, recall=r, F1=2*p*r/max(p+r, 1e-12),
        write_to_read=counts[2][1], read_to_write=counts[1][2],
        unmatched_predictions=sum(counts[3][:3]), unmatched_GT=sum(row[3] for row in counts[:3]),
        delta_AP_pp=100*(ap-BASELINE_AP), delta_write_AP_pp=100*(classes['write']['AP']-BASELINE_WRITE_AP),
        baseline_reference='historical Run2, NOT rerun here; verify protocol/data before comparison',
        FLOPs_G=float(profile[-1][0]) if profile else None, params=int(profile[-1][1]) if profile else None)


def report(output, result):
    (output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    lines = ['# D-FINE-S + GQC 单模块', '', 'AQS关闭，BoxLocal关闭；原始GQC，仅最终普通查询评分。',
        f'评估权重：{result["checkpoint"]}（EMA）', '',
        '| 模型 | mAP50 | mAP50:95 | hand AP | read AP | write AP |',
        '|---|---:|---:|---:|---:|---:|',
        '| 原始D-FINE（历史Run2） | 72.16 | 55.05 | 59.34 | 58.19 | 47.61 |',
        '| AQS（历史） | 73.27 | 55.87 | 59.44 | 58.44 | 49.74 |',
        '| AQS+GQC（历史） | 73.57 | 56.21 | 60.23 | 59.47 | 48.94 |',
        '| GQC单模块（本次） | ' + ' | '.join(f'{value*100:.2f}' for value in
            [result['AP50'], result['AP'], *result['class_AP'].values()]) + ' |', '',
        f'相对历史基线：整体AP {result["delta_AP_pp"]:+.2f} pp，write AP {result["delta_write_AP_pp"]:+.2f} pp。',
        f'conf0.5/IoU0.5微平均：Precision {result["precision"]*100:.2f}%，Recall {result["recall"]*100:.2f}%，F1 {result["F1"]*100:.2f}%。',
        f'write→read={result["write_to_read"]}，read→write={result["read_to_write"]}。', '',
        '历史参考不是本次重跑。先核对数据/训练口径，再用多seed验证增益；单次结果不证明稳定有效。']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    with ZipFile(output/'GQC_only_diagnostics.zip', 'x', ZIP_DEFLATED) as archive:
        for file in sorted(output.rglob('*')):
            if file.is_file() and file.suffix.lower() in ('.json', '.jsonl', '.txt', '.log', '.md', '.yml', '.png'):
                archive.write(file, file.relative_to(output))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', nargs='?', default='run1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wait-gpu', action='store_true')
    parser.add_argument('--diagnostics', action='store_true', help='额外导出完整query数据并分析，默认仅指标/混淆矩阵')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--root', default=os.environ.get('SCBS_GQC_ROOT', DEFAULT_ROOT))
    parser.add_argument('--gpu', type=int, default=int(os.environ.get('SCBS_GQC_GPU', '0')))
    parser.add_argument('--port', type=int, default=int(os.environ.get('SCBS_GQC_PORT', '7785')))
    args = parser.parse_args(argv)
    if not re.fullmatch(r'run[1-9][0-9]*', args.run) or args.seed < 0 or args.gpu < 0 or not 1024 <= args.port <= 65535:
        parser.error('需要run1/run2等编号、非负seed/GPU及1024..65535端口')
    root = Path(args.root)
    if not root.is_absolute():
        parser.error('输出根目录必须为绝对路径')
    output = root/output_name(args.seed, args.run)
    if output.exists() or output.is_symlink():
        raise ValueError(f'拒绝覆盖已有结果：{output}；请选择新run编号')
    if not (ROOT/CONFIG).is_file():
        raise ValueError(f'缺少配置：{CONFIG}')
    print(f'原始D-FINE + 原GQC单模块；AQS/BoxLocal关闭。\n输出：{output}', flush=True)
    if args.dry_run:
        print('训练：'+shlex.join(command(output, args)), flush=True)
        print('结束后评估：'+shlex.join(command(output/'diagnostics_eval/evaluation', args, output/'best_stg2.pth')), flush=True)
        print('仅演练：不连接服务器、不加载PyTorch、不占GPU、不创建目录；不执行ON/OFF。', flush=True)
        return
    os.chdir(ROOT); os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    config = validate_config(); data = validate_data(config)
    import torch
    if not torch.cuda.is_available():
        raise ValueError('CUDA不可用，不启动CPU完整训练')
    from tools.diagnostics.wait_for_gpu import wait_for_gpu
    wait_for_gpu(args.gpu, wait=args.wait_gpu, idle_checks=1)
    config = validate_config()
    if validate_data(config) != data:
        raise ValueError('等待期间数据发生变化；不启动实验')
    with socket.socket() as port:
        port.bind(('127.0.0.1', args.port))
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
    status = output/'status.json'
    def state(value):
        status.write_text(json.dumps(dict(state=value, output=str(output)), ensure_ascii=False, indent=2))
    try:
        state('preparing'); snapshot(output, config, args, data)
        state('training'); run_logged(command(output, args), output/'console.log', env)
        completed_training(output)
        checkpoint = next((file for file in (output/'best_stg2.pth', output/'best_stg1.pth') if file.is_file() and file.stat().st_size), None)
        if checkpoint is None:
            raise ValueError('训练未产生最佳权重；保留输出')
        state('evaluating best EMA')
        evaluation = output/'diagnostics_eval/evaluation'
        run_logged(command(evaluation, args, checkpoint), output/'evaluation_console.log', env)
        if args.diagnostics:
            run_logged([sys.executable, 'tools/diagnostics/analyze_detection_diagnostics.py', '--input-dir', str(evaluation),
                        '--output-dir', str(output/'diagnostics_eval/analysis')], output/'analysis_console.log', env)
        result = collect_result(output, checkpoint)
        state('complete'); report(output, result)
    except BaseException as error:
        state(f'incomplete: {type(error).__name__}: {error}')
        raise
    print(f'单模块实验完成：{output}/summary.md\n可下载：{output}/GQC_only_diagnostics.zip', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        print(f'错误：{error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
