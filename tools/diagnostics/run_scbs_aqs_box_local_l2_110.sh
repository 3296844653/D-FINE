#!/usr/bin/env bash
# Joint structural ablation, fresh training. Never overwrite existing results.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
combo_fail() { printf '错误：%s\n' "$*" >&2; exit 1; }
combo_dry=0
if [[ "${1:-}" == '--dry-run' ]]; then combo_dry=1; shift; fi
[[ "$#" -le 1 ]] || combo_fail '用法：bash tools/diagnostics/run_scbs_aqs_box_local_l2_110.sh [--dry-run] [新输出目录]'
combo_config='configs/dfine/dfine_s_scbs_hrw_110_AQS_BoxLocalQueryAdapter_l2.yml'
combo_output="${1:-/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_AQS_BoxLocalQueryAdapter_l2_seed0_run1}"
combo_conda="${SCBS_COMBO_CONDA:-${CONDA_EXE:-conda}}"
combo_env="${SCBS_COMBO_ENV:-wq}"; combo_gpu="${SCBS_COMBO_GPU:-0}"
[[ "$combo_output" == /* && "$combo_gpu" =~ ^[0-9]+$ ]] || combo_fail '输出须为绝对路径，GPU须为单个非负整数'
[[ ! -e "$combo_output" && ! -L "$combo_output" ]] || combo_fail "输出已存在，保留旧结果：$combo_output"
command -v "$combo_conda" >/dev/null 2>&1 || combo_fail '找不到Conda'
export CUDA_VISIBLE_DEVICES="$combo_gpu" PYTHONUNBUFFERED=1 MPLBACKEND=Agg

"$combo_conda" run --no-capture-output -n "$combo_env" python - "$combo_config" "$combo_dry" <<'PY'
import copy
import inspect
import json
import sys
from pathlib import Path
import torch
from src.core.yaml_utils import load_config
from src.zoo.dfine.dfine import DFINE
from src.zoo.dfine.dfine_decoder import DFINETransformer

for filename in ('train.py', 'src/zoo/dfine/night_probes.py',
                 'tools/diagnostics/analyze_detection_diagnostics.py'):
    assert Path(filename).is_file(), f'缺少文件：{filename}'
assert 'allow_aqs_l2_local_combo' in inspect.signature(DFINE).parameters, 'DFINE组合开关未同步'
assert 'allow_aqs_l2_local_combo' in inspect.signature(DFINETransformer.configure_night_probe).parameters, 'Decoder组合支持未同步'
trial = load_config(sys.argv[1], cfg={})
expected = copy.deepcopy(load_config('configs/dfine/dfine_s_scbs_hrw_110.yml', cfg={}))
expected['DFINE'].update(night_probe='decoder_l2_local', allow_aqs_l2_local_combo=True)
expected['DFINETransformer'].update(use_aqs_refine=True, aqs_apply_layer=2,
    aqs_threshold=.5, aqs_temperature=.1, aqs_residual_init=.1)
for item in (expected, trial):
    item.pop('__include__', None); item.pop('output_dir', None)
assert trial == expected, '组合配置改动了其他模块、损失、数据或训练设置'
assert trial['epochs'] == 110 and trial['num_classes'] == 3
assert trial['train_dataloader']['collate_fn']['stop_epoch'] == 100
assert trial['train_dataloader']['dataset']['transforms']['policy']['epoch'] == 100
assert trial['train_dataloader']['total_batch_size'] == trial['val_dataloader']['total_batch_size'] == 32
print('检查通过：单模型 Local-L2 -> AQS-L2 -> L3；110/100、batch32、seed0、原损失。', flush=True)
if sys.argv[2] == '1':
    print('仅演练：未检查服务器数据/GPU，未创建输出，未启动训练。', flush=True)
    sys.exit(0)
for split in ('train', 'val'):
    data = trial[f'{split}_dataloader']['dataset']
    folder = Path(data['img_folder']); assert folder.is_dir(), f'缺少{split}图片目录'
    dataset = json.loads(Path(data['ann_file']).read_text(encoding='utf-8'))
    assert sorted((c['id'], c['name']) for c in dataset['categories']) == [(0, 'hand-raising'), (1, 'read'), (2, 'write')]
    assert dataset['images'], f'{split}集为空'
    for image in dataset['images']:
        assert (folder / image['file_name']).is_file(), f"缺少图片：{image['file_name']}"
assert torch.cuda.is_available(), 'CUDA不可用，未启动CPU完整训练'
print('数据检查通过；请确认GPU空闲。', flush=True)
PY
[[ "$combo_dry" == 0 ]] || exit 0

mkdir -p -- "$(dirname "$combo_output")"
mkdir -- "$combo_output"
trap 'combo_status=$?; printf "组合实验未完成，退出码=%s；保留结果：%s\n" "$combo_status" "$combo_output" >&2; exit "$combo_status"' ERR
printf '\n[1/3] 开始联合训练（不加载单模块权重）：%s\n' "$combo_output"
"$combo_conda" run --no-capture-output -n "$combo_env" \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py -c "$combo_config" --output-dir "$combo_output" --use-amp --seed=0 \
2>&1 | tee "$combo_output/console.log"
test -f "$combo_output/best_stg2.pth"

printf '\n[2/3] 本组合最佳EMA权重：导出完整诊断。\n'
"$combo_conda" run --no-capture-output -n "$combo_env" \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py -c "$combo_config" -r "$combo_output/best_stg2.pth" --test-only \
--output-dir "$combo_output/diagnostics_eval/evaluation" --use-amp --seed=0 \
-u export_confusion_matrix=true export_query_diagnostics=true \
query_diagnostic_conf_thresh=0.5 query_diagnostic_iou_thresh=0.5 query_diagnostic_neighbor_iou_thresh=0.8 \
2>&1 | tee "$combo_output/diagnostics_console.log"

printf '\n[3/3] 离线分析；后续统一与110轮Baseline Run2比较。\n'
"$combo_conda" run --no-capture-output -n "$combo_env" \
python tools/diagnostics/analyze_detection_diagnostics.py \
--input-dir "$combo_output/diagnostics_eval/evaluation" --output-dir "$combo_output/diagnostics_eval/analysis" \
2>&1 | tee "$combo_output/analysis_console.log"
test -s "$combo_output/diagnostics_eval/analysis/coco_per_class_and_size.json"
test -s "$combo_output/diagnostics_eval/analysis/best_query_cross_classification.json"
printf '\n组合实验完成，诊断目录：%s/diagnostics_eval\n' "$combo_output"
