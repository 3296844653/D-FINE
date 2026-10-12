#!/usr/bin/env bash
# Full independent EvidenceContrastAQS training -> best-EMA diagnostics -> offline analysis.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
eca_fail() { printf '错误：%s\n' "$*" >&2; exit 1; }
eca_dry=0
if [[ "${1:-}" == --dry-run ]]; then eca_dry=1; shift; fi
[[ "$#" -le 1 ]] || eca_fail '用法：bash tools/diagnostics/run_scbs_EvidenceContrastAQS_110.sh [--dry-run] [新的绝对输出目录]'
eca_config='configs/dfine/dfine_s_scbs_hrw_110_EvidenceContrastAQS.yml'
eca_output="${1:-/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_EvidenceContrastAQS_seed0_run1}"
eca_conda="${SCBS_ECAQS_CONDA:-${CONDA_EXE:-conda}}"
eca_env="${SCBS_ECAQS_ENV:-wq}"; eca_gpu="${SCBS_ECAQS_GPU:-0}"
[[ "$eca_output" == /* && "$eca_gpu" =~ ^[0-9]+$ ]] || eca_fail '输出须为绝对路径，GPU须为单个非负整数'
[[ "${eca_output##*/}" == *EvidenceContrastAQS* ]] || eca_fail '输出名称须包含 EvidenceContrastAQS'
[[ ! -e "$eca_output" && ! -L "$eca_output" ]] || eca_fail "输出已存在，保留旧结果：$eca_output"
command -v "$eca_conda" >/dev/null 2>&1 || eca_fail '找不到Conda'
export CUDA_VISIBLE_DEVICES="$eca_gpu" PYTHONUNBUFFERED=1 MPLBACKEND=Agg

"$eca_conda" run --no-capture-output -n "$eca_env" python - "$eca_config" "$eca_dry" <<'PY'
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
expected['DFINETransformer'].update(use_aqs_refine=True, aqs_refiner_type='EvidenceContrastAQS',
    aqs_apply_layer=2, aqs_threshold=.5, aqs_temperature=.1, aqs_residual_init=.1,
    aqs_evidence_dim=64, aqs_evidence_slots=4, aqs_evidence_heads=4)
for item in (trial, expected):
    item.pop('__include__', None); item.pop('output_dir', None)
assert trial == expected, '配置修改了EvidenceContrastAQS之外的数据、损失、训练设置或模块'
assert trial['epochs'] == 110 and trial['num_classes'] == 3
assert trial['train_dataloader']['collate_fn']['stop_epoch'] == 100
assert trial['train_dataloader']['dataset']['transforms']['policy']['epoch'] == 100
assert trial['train_dataloader']['total_batch_size'] == trial['val_dataloader']['total_batch_size'] == 32
for name in ('train.py', 'tools/diagnostics/analyze_detection_diagnostics.py',
             'tools/diagnostics/export_evidence_contrast_aqs.py', 'tools/diagnostics/test_evidence_contrast_aqs.py'):
    assert Path(name).is_file(), f'缺少文件：{name}'
with patch('src.core.yaml_config.load_config', lambda path: load_config(path, cfg={})):
    cfg = YAMLConfig(sys.argv[1])
cfg.yaml_cfg['HGNetv2']['pretrained'] = False
model = cfg.model
assert model.decoder.decoder.aqs_refiner_type == 'EvidenceContrastAQS', '新源码/接口未同步'
assert model.decoder.use_aqs_refine and model.decoder.decoder.aqs_layer_indices == (1,)
assert model.night_probe == 'none'
print('EvidenceContrastAQS完整机制检查通过：源筛选 -> 类别/槽位证据库 -> 支持/竞争检索 -> 条件残差。', flush=True)
print('独立110/100轮、batch32、seed0、原损失；替换原AQS内部机制；没有SADF/BoxLocal叠加。', flush=True)
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
[[ "$eca_dry" == 0 ]] || exit 0

# Synthetic checks include CUDA FP16 when available, but are not AP evidence.
"$eca_conda" run --no-capture-output -n "$eca_env" python tools/diagnostics/test_evidence_contrast_aqs.py
mkdir -p -- "$(dirname "$eca_output")"
mkdir -- "$eca_output"
trap 'eca_status=$?; printf "EvidenceContrastAQS实验未完成，退出码=%s；保留输出：%s\n" "$eca_status" "$eca_output" >&2; exit "$eca_status"' ERR
printf '\n[1/4] 从头完整训练EvidenceContrastAQS：%s\n' "$eca_output"
"$eca_conda" run --no-capture-output -n "$eca_env" \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py -c "$eca_config" --output-dir "$eca_output" --use-amp --seed=0 \
2>&1 | tee "$eca_output/console.log"
test -f "$eca_output/best_stg2.pth"

printf '\n[2/4] 导出最佳checkpoint的model/EMA参数。\n'
"$eca_conda" run --no-capture-output -n "$eca_env" \
python tools/diagnostics/export_evidence_contrast_aqs.py --checkpoint "$eca_output/best_stg2.pth"

printf '\n[3/4] 最佳EMA权重：完整query/混淆矩阵诊断。\n'
"$eca_conda" run --no-capture-output -n "$eca_env" \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py -c "$eca_config" -r "$eca_output/best_stg2.pth" --test-only \
--output-dir "$eca_output/diagnostics_eval/evaluation" --use-amp --seed=0 \
-u export_confusion_matrix=true export_query_diagnostics=true \
query_diagnostic_conf_thresh=0.5 query_diagnostic_iou_thresh=0.5 query_diagnostic_neighbor_iou_thresh=0.8 \
2>&1 | tee "$eca_output/diagnostics_console.log"

printf '\n[4/4] 类别×尺度、候选缺口、IoU-score及PR离线分析；统一对比Baseline Run2。\n'
"$eca_conda" run --no-capture-output -n "$eca_env" \
python tools/diagnostics/analyze_detection_diagnostics.py \
--input-dir "$eca_output/diagnostics_eval/evaluation" --output-dir "$eca_output/diagnostics_eval/analysis" \
2>&1 | tee "$eca_output/analysis_console.log"
test -s "$eca_output/diagnostics_eval/analysis/coco_per_class_and_size.json"
test -s "$eca_output/diagnostics_eval/analysis/best_query_cross_classification.json"
printf '\nEvidenceContrastAQS训练与诊断完成：%s\n' "$eca_output"
