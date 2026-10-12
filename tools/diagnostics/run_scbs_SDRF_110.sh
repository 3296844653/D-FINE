#!/usr/bin/env bash
# Independent SDRF training, parameter export and best-EMA diagnostics.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
sdrf_fail() { printf '错误：%s\n' "$*" >&2; exit 1; }
sdrf_dry=0
if [[ "${1:-}" == --dry-run ]]; then sdrf_dry=1; shift; fi
[[ "$#" -le 1 ]] || sdrf_fail '用法：bash tools/diagnostics/run_scbs_SDRF_110.sh [--dry-run] [新的绝对输出目录]'
sdrf_config='configs/dfine/dfine_s_scbs_hrw_110_SDRF.yml'
sdrf_output="${1:-/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_SDRF_seed0_run1}"
sdrf_gpu="${SCBS_SDRF_GPU:-0}"; sdrf_port="${SCBS_SDRF_PORT:-7781}"
sdrf_conda="${SCBS_SDRF_CONDA:-${CONDA_EXE:-conda}}"; sdrf_env="${SCBS_SDRF_ENV:-wq}"
[[ "$sdrf_output" == /* && "${sdrf_output##*/}" == *SDRF* ]] || sdrf_fail '输出须为含SDRF的绝对路径'
[[ "$sdrf_gpu" =~ ^[0-9]+$ && "$sdrf_port" =~ ^[0-9]+$ ]] || sdrf_fail 'GPU/端口须为整数'
[[ "$sdrf_port" -ge 1024 && "$sdrf_port" -le 65535 ]] || sdrf_fail '端口须在1024到65535之间'
[[ ! -e "$sdrf_output" && ! -L "$sdrf_output" ]] || sdrf_fail "输出已存在，保留旧结果：$sdrf_output"
for sdrf_file in train.py "$sdrf_config" tools/diagnostics/test_sdrf.py \
    tools/diagnostics/export_SDRF_parameters.py tools/diagnostics/analyze_detection_diagnostics.py; do
  [[ -f "$sdrf_file" ]] || sdrf_fail "缺少文件：$sdrf_file"
done
printf 'SDRF：110轮/停止增强100轮，batch32，seed0，原损失、不叠加AQS。\n输出：%s\n端口：%s\n' "$sdrf_output" "$sdrf_port"
if [[ "$sdrf_dry" == 1 ]]; then
  printf '仅演练，不读服务器数据、不创建输出、不启动训练。\n'
  printf '训练方式：conda run --no-capture-output -n %s torchrun --master_port=%s --nproc_per_node=1 train.py -c %s --output-dir "%s" --use-amp --seed=0\n' "$sdrf_env" "$sdrf_port" "$sdrf_config" "$sdrf_output"
  exit 0
fi
command -v "$sdrf_conda" >/dev/null 2>&1 || sdrf_fail '找不到Conda'
export CUDA_VISIBLE_DEVICES="$sdrf_gpu" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
"$sdrf_conda" run --no-capture-output -n "$sdrf_env" python - "$sdrf_config" "$sdrf_port" "$sdrf_gpu" <<'PY'
import copy
import json
import socket
import subprocess
import sys
from pathlib import Path
import torch
from src.core.yaml_utils import load_config

trial = load_config(sys.argv[1], cfg={})
expected = copy.deepcopy(load_config('configs/dfine/dfine_s_scbs_hrw_110.yml', cfg={}))
expected['HybridEncoder'].update(use_sdrf=True, sdrf_dim=64, sdrf_heads=4, sdrf_window=5,
    sdrf_halo=1, sdrf_residual_init=.1, sdrf_checkpoint=True)
for item in (trial, expected):
    item.pop('__include__',None); item.pop('output_dir',None)
assert trial == expected, '存在SDRF以外的配置变化，停止以保持独立对比'
assert trial['epochs'] == 110 and trial['num_classes'] == 3
assert trial['train_dataloader']['collate_fn']['stop_epoch'] == 100
assert trial['train_dataloader']['dataset']['transforms']['policy']['epoch'] == 100
assert trial['train_dataloader']['total_batch_size'] == trial['val_dataloader']['total_batch_size'] == 32
# Reject occupied GPU/port; never terminate another user's project.
processes = subprocess.run(['nvidia-smi','-i',sys.argv[3],'--query-compute-apps=pid,process_name',
    '--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.strip()
# Remote desktop may appear as C+G; it is not another training/LLM task.
desktop = {'Xorg','gnome-shell','gnome-remote-desktop-daemon'}
busy = [line for line in processes.splitlines() if Path(line.split(',',1)[-1].strip()).name not in desktop]
assert not busy, f'GPU仍有计算进程，未启动训练：{busy}'
with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
    sock.bind(('0.0.0.0',int(sys.argv[2])))
assert torch.cuda.is_available(), 'CUDA不可用，未启动CPU完整训练'
for split in ('train','val'):
    dataset = trial[f'{split}_dataloader']['dataset']; folder = Path(dataset['img_folder'])
    assert folder.is_dir(), f'图片目录不存在：{folder}'
    annotation = json.loads(Path(dataset['ann_file']).read_text(encoding='utf-8'))
    assert sorted((c['id'],c['name']) for c in annotation['categories']) == [(0,'hand-raising'),(1,'read'),(2,'write')]
    assert annotation['images'], f'{split}为空'
    for image in annotation['images']:
        assert (folder/image['file_name']).is_file(), f"缺少图片：{image['file_name']}"
    assert all(type(a['category_id']) is int and 0 <= a['category_id'] < 3 for a in annotation['annotations']), '标注类别越界'
print('数据、独立配置、GPU和端口检查通过。',flush=True)
PY
printf '\n[0/4] 合成验证（包含服务器CUDA FP16测试，不是效果验证）。\n'
"$sdrf_conda" run --no-capture-output -n "$sdrf_env" python -u tools/diagnostics/test_sdrf.py
mkdir -p -- "$(dirname "$sdrf_output")"
mkdir -- "$sdrf_output"
trap 'sdrf_status=$?; printf "SDRF实验未完成，退出码=%s；保留输出：%s\n" "$sdrf_status" "$sdrf_output" >&2; exit "$sdrf_status"' ERR
printf '\n[1/4] 从头完整训练SDRF。\n'
"$sdrf_conda" run --no-capture-output -n "$sdrf_env" \
torchrun --master_port="$sdrf_port" --nproc_per_node=1 train.py \
-c "$sdrf_config" --output-dir "$sdrf_output" --use-amp --seed=0 \
2>&1 | tee "$sdrf_output/console.log"
test -f "$sdrf_output/best_stg2.pth"
printf '\n[2/4] 导出最佳权重的model/EMA参数。\n'
"$sdrf_conda" run --no-capture-output -n "$sdrf_env" \
python -u tools/diagnostics/export_SDRF_parameters.py --checkpoint "$sdrf_output/best_stg2.pth"
printf '\n[3/4] 最佳EMA权重：混淆矩阵和完整query诊断。\n'
"$sdrf_conda" run --no-capture-output -n "$sdrf_env" \
torchrun --master_port="$sdrf_port" --nproc_per_node=1 train.py \
-c "$sdrf_config" -r "$sdrf_output/best_stg2.pth" --test-only \
--output-dir "$sdrf_output/diagnostics_eval/evaluation" --use-amp --seed=0 \
-u export_confusion_matrix=true export_query_diagnostics=true \
query_diagnostic_conf_thresh=0.5 query_diagnostic_iou_thresh=0.5 query_diagnostic_neighbor_iou_thresh=0.8 \
2>&1 | tee "$sdrf_output/diagnostics_console.log"
printf '\n[4/4] 类别×尺度、候选缺口、IoU-score和PR离线分析。\n'
"$sdrf_conda" run --no-capture-output -n "$sdrf_env" \
python -u tools/diagnostics/analyze_detection_diagnostics.py \
--input-dir "$sdrf_output/diagnostics_eval/evaluation" --output-dir "$sdrf_output/diagnostics_eval/analysis" \
2>&1 | tee "$sdrf_output/analysis_console.log"
test -s "$sdrf_output/diagnostics_eval/analysis/coco_per_class_and_size.json"
test -s "$sdrf_output/diagnostics_eval/analysis/best_query_cross_classification.json"
printf '\nSDRF训练与诊断完成：%s\n诊断请与110轮Baseline Run2同口径对比。\n' "$sdrf_output"
