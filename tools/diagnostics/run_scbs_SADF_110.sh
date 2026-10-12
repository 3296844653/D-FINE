#!/usr/bin/env bash
# Full independent SADF training -> best-EMA diagnostics -> offline analysis.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
sadf_fail() { printf '错误：%s\n' "$*" >&2; exit 1; }
sadf_dry=0
if [[ "${1:-}" == --dry-run ]]; then sadf_dry=1; shift; fi
[[ "$#" -le 1 ]] || sadf_fail '用法：bash tools/diagnostics/run_scbs_SADF_110.sh [--dry-run] [新的绝对输出目录]'
sadf_config='configs/dfine/dfine_s_scbs_hrw_110_SADF.yml'
sadf_output="${1:-/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_SADF_seed0_run1}"
sadf_conda="${SCBS_SADF_CONDA:-${CONDA_EXE:-conda}}"
sadf_env="${SCBS_SADF_ENV:-wq}"; sadf_gpu="${SCBS_SADF_GPU:-0}"
[[ "$sadf_output" == /* && "$sadf_gpu" =~ ^[0-9]+$ ]] || sadf_fail '输出须为绝对路径，GPU须为单个非负整数'
[[ "${sadf_output##*/}" == *SADF* ]] || sadf_fail '输出名称须包含 SADF'
[[ ! -e "$sadf_output" && ! -L "$sadf_output" ]] || sadf_fail "输出已存在，保留旧结果：$sadf_output"
command -v "$sadf_conda" >/dev/null 2>&1 || sadf_fail '找不到Conda'
export CUDA_VISIBLE_DEVICES="$sadf_gpu" PYTHONUNBUFFERED=1 MPLBACKEND=Agg

"$sadf_conda" run --no-capture-output -n "$sadf_env" python - "$sadf_config" "$sadf_dry" <<'PY'
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch
import torch
from src.core import YAMLConfig
from src.core.yaml_utils import load_config

trial = load_config(sys.argv[1], cfg={})
expected = copy.deepcopy(load_config('configs/dfine/dfine_s_scbs_hrw_110.yml', cfg={}))
expected['HybridEncoder'].update(use_sadf=True, sadf_dim=64, sadf_offset_limit=2.,
    sadf_residual_init=.1, sadf_checkpoint=True)
for item in (trial, expected):
    item.pop('__include__', None); item.pop('output_dir', None)
assert trial == expected, '配置修改了SADF之外的数据、损失、训练设置或模块'
assert trial['epochs'] == 110 and trial['num_classes'] == 3
assert trial['train_dataloader']['collate_fn']['stop_epoch'] == 100
assert trial['train_dataloader']['dataset']['transforms']['policy']['epoch'] == 100
assert trial['train_dataloader']['total_batch_size'] == trial['val_dataloader']['total_batch_size'] == 32
for name in ('train.py', 'tools/diagnostics/analyze_detection_diagnostics.py',
             'tools/diagnostics/export_sadf_parameters.py', 'tools/diagnostics/test_sadf.py'):
    assert Path(name).is_file(), f'缺少文件：{name}'
with patch('src.core.yaml_config.load_config', lambda path: load_config(path, cfg={})):
    cfg = YAMLConfig(sys.argv[1])
cfg.yaml_cfg['HGNetv2']['pretrained'] = False
model = cfg.model
assert hasattr(model.encoder, 'sadf'), 'SADF源码/接口未同步'
assert not model.decoder.use_aqs_refine and model.night_probe == 'none'
print('SADF完整机制检查通过：原始P3对比细节 -> P4/P5有界对齐 -> 尺度竞争 -> 双门控融合。', flush=True)
print('独立110/100轮、batch32、seed0、原损失；没有AQS/BoxLocal叠加。', flush=True)
if sys.argv[2] == '1':
    print('仅演练；不读服务器数据、不创建输出、不启动训练。', flush=True)
    sys.exit(0)
assert torch.cuda.is_available(), 'CUDA不可用，未启动CPU完整训练'
for split in ('train', 'val'):
    dataset = trial[f'{split}_dataloader']['dataset']; folder = Path(dataset['img_folder'])
    assert folder.is_dir(), f'{split}图片目录不存在：{folder}'
    annotation = json.loads(Path(dataset['ann_file']).read_text(encoding='utf-8'))
    assert sorted((c['id'], c['name']) for c in annotation['categories']) == [(0,'hand-raising'), (1,'read'), (2,'write')]
    assert annotation['images'], f'{split}为空'
    for image in annotation['images']:
        assert (folder / image['file_name']).is_file(), f"缺少图片：{image['file_name']}"
print('数据检查通过；请确认GPU空闲。', flush=True)
PY
[[ "$sadf_dry" == 0 ]] || exit 0

# Synthetic checks include CUDA FP16 when available, but are not AP evidence.
"$sadf_conda" run --no-capture-output -n "$sadf_env" python tools/diagnostics/test_sadf.py
mkdir -p -- "$(dirname "$sadf_output")"
mkdir -- "$sadf_output"
trap 'sadf_status=$?; printf "SADF实验未完成，退出码=%s；保留输出：%s\n" "$sadf_status" "$sadf_output" >&2; exit "$sadf_status"' ERR
printf '\n[1/4] 从头完整训练SADF：%s\n' "$sadf_output"
"$sadf_conda" run --no-capture-output -n "$sadf_env" \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py -c "$sadf_config" --output-dir "$sadf_output" --use-amp --seed=0 \
2>&1 | tee "$sadf_output/console.log"
test -f "$sadf_output/best_stg2.pth"

printf '\n[2/4] 导出最佳checkpoint的model/EMA参数。\n'
"$sadf_conda" run --no-capture-output -n "$sadf_env" \
python tools/diagnostics/export_sadf_parameters.py --checkpoint "$sadf_output/best_stg2.pth"

printf '\n[3/4] 最佳EMA权重：完整query/混淆矩阵诊断。\n'
"$sadf_conda" run --no-capture-output -n "$sadf_env" \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py -c "$sadf_config" -r "$sadf_output/best_stg2.pth" --test-only \
--output-dir "$sadf_output/diagnostics_eval/evaluation" --use-amp --seed=0 \
-u export_confusion_matrix=true export_query_diagnostics=true \
query_diagnostic_conf_thresh=0.5 query_diagnostic_iou_thresh=0.5 query_diagnostic_neighbor_iou_thresh=0.8 \
2>&1 | tee "$sadf_output/diagnostics_console.log"

printf '\n[4/4] 类别×尺度、候选缺口、IoU-score及PR离线分析；统一对比Baseline Run2。\n'
"$sadf_conda" run --no-capture-output -n "$sadf_env" \
python tools/diagnostics/analyze_detection_diagnostics.py \
--input-dir "$sadf_output/diagnostics_eval/evaluation" --output-dir "$sadf_output/diagnostics_eval/analysis" \
2>&1 | tee "$sadf_output/analysis_console.log"
test -s "$sadf_output/diagnostics_eval/analysis/coco_per_class_and_size.json"
test -s "$sadf_output/diagnostics_eval/analysis/best_query_cross_classification.json"
printf '\nSADF训练与诊断完成：%s\n' "$sadf_output"
