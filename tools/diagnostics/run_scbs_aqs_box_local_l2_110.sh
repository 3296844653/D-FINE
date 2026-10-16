#!/usr/bin/env bash
# Default: fresh joint training only. Diagnostics are explicitly opt-in.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
combo_fail() { printf '错误：%s\n' "$*" >&2; exit 1; }
combo_dry=0; combo_diagnostics=0; combo_wait=0; combo_parallel=0; combo_destination=''
for combo_arg in "$@"; do
  case "$combo_arg" in
    --dry-run) combo_dry=1 ;;
    --diagnostics) combo_diagnostics=1 ;;
    --wait-gpu) combo_wait=1 ;;
    --parallel) combo_parallel=1 ;;
    --*) combo_fail "未知选项：$combo_arg" ;;
    *) [[ -z "$combo_destination" ]] || combo_fail '只接受一个run编号或新绝对输出目录'
       combo_destination="$combo_arg" ;;
  esac
done
combo_config='configs/dfine/dfine_s_scbs_hrw_110_AQS_BoxLocalQueryAdapter_l2.yml'
combo_name='AQS_BoxLocalQueryAdapter_l2'
if [[ "$combo_parallel" == 1 ]]; then
  combo_name="${combo_name}_parallel"
  combo_config="configs/dfine/dfine_s_scbs_hrw_110_${combo_name}.yml"
fi
combo_root="${SCBS_COMBO_ROOT:-/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw}"
combo_destination="${combo_destination:-run1}"
if [[ "$combo_destination" =~ ^run[1-9][0-9]*$ ]]; then
  combo_output="$combo_root/dfine_s_scbs_hrw_epochs110_${combo_name}_seed0_${combo_destination}"
elif [[ "$combo_destination" == /* ]]; then
  combo_output="$combo_destination"
else
  combo_fail '用法：[run1|run2|新绝对输出目录] [--dry-run] [--wait-gpu] [--diagnostics] [--parallel]'
fi
combo_conda="${SCBS_COMBO_CONDA:-${CONDA_EXE:-conda}}"
combo_env="${SCBS_COMBO_ENV:-wq}"; combo_gpu="${SCBS_COMBO_GPU:-0}"
combo_port="${SCBS_COMBO_PORT:-7781}"
[[ "$combo_output" == /* && "$combo_gpu" =~ ^[0-9]+$ ]] || combo_fail '输出须为绝对路径，GPU须为单个非负整数'
[[ "$combo_port" =~ ^[0-9]+$ ]] || combo_fail '端口须为整数'
[[ "$combo_port" -ge 1024 && "$combo_port" -le 65535 ]] || combo_fail '端口须为1024..65535'
[[ ! -e "$combo_output" && ! -L "$combo_output" ]] || combo_fail "输出已存在，保留旧结果：$combo_output"
for combo_file in train.py "$combo_config" tools/diagnostics/wait_for_gpu.py; do
  [[ -f "$combo_file" ]] || combo_fail "缺少文件：$combo_file"
done
if [[ "$combo_diagnostics" == 1 ]]; then
  [[ -f tools/diagnostics/analyze_detection_diagnostics.py ]] || combo_fail '缺少诊断分析工具'
fi
combo_train=("$combo_conda" run --no-capture-output -n "$combo_env"
  torchrun --master_addr=127.0.0.1 --master_port="$combo_port" --nproc_per_node=1
  train.py -c "$combo_config" --output-dir "$combo_output" --use-amp --seed=0)
if [[ "$combo_parallel" == 1 ]]; then
  printf '并行BoxLocal-L2 + AQS-L2 R1；同一L2 Query -> 两分支增量相加 -> L3。\n'
else
  printf '默认BoxLocal-L2 + AQS-L2 R1；Local -> AQS -> L3。\n'
fi
printf 'BoxLocal: dim64 / grid3 / offset0.15 / residual0.10。\n'
printf 'AQS: L2 / threshold0.50 / temperature0.10 / residual_init0.10。\n'
printf '110/100轮、batch32、seed0、原损失；从头联合训练，不加载旧训练权重。\n输出：%s\n' "$combo_output"
if [[ "$combo_dry" == 1 ]]; then
  printf '仅演练：不连接服务器、不占GPU、不创建目录。\n训练命令：'
  printf '%q ' "${combo_train[@]}"; printf '\n'
  printf 'GPU等待=%s；训练后完整诊断=%s；不会执行ON/OFF。\n' "$combo_wait" "$combo_diagnostics"
  exit 0
fi
command -v "$combo_conda" >/dev/null 2>&1 || combo_fail '找不到Conda'
export CUDA_VISIBLE_DEVICES="$combo_gpu" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
combo_wait_args=(--gpu "$combo_gpu" --idle-checks 1)
[[ "$combo_wait" == 0 ]] || combo_wait_args+=(--wait)
"$combo_conda" run --no-capture-output -n "$combo_env" \
python -u tools/diagnostics/wait_for_gpu.py "${combo_wait_args[@]}"

"$combo_conda" run --no-capture-output -n "$combo_env" python - "$combo_config" "$combo_dry" "$combo_parallel" <<'PY'
import copy
import inspect
import json
import sys
from pathlib import Path
import torch
from src.core.yaml_utils import load_config
from src.zoo.dfine.dfine import DFINE
from src.zoo.dfine.dfine_decoder import DFINETransformer

for filename in ('train.py', 'src/zoo/dfine/night_probes.py'):
    assert Path(filename).is_file(), f'缺少文件：{filename}'
assert 'allow_aqs_l2_local_combo' in inspect.signature(DFINE).parameters, 'DFINE组合开关未同步'
assert 'allow_aqs_l2_local_combo' in inspect.signature(DFINETransformer.configure_night_probe).parameters, 'Decoder组合支持未同步'
trial = load_config(sys.argv[1], cfg={})
expected = copy.deepcopy(load_config('configs/dfine/dfine_s_scbs_hrw_110.yml', cfg={}))
expected['DFINE'].update(night_probe='decoder_l2_local', allow_aqs_l2_local_combo=True,
    box_local_dim=64, box_local_grid_size=3, box_local_offset_scale=.15, box_local_residual_scale=.1)
expected['DFINETransformer'].update(use_aqs_refine=True, aqs_apply_layer=2,
    aqs_threshold=.5, aqs_temperature=.1, aqs_residual_init=.1)
parallel = len(sys.argv) > 3 and sys.argv[3] == '1'
if parallel:
    for method in (DFINE, DFINETransformer.configure_night_probe):
        assert 'aqs_l2_local_fusion' in inspect.signature(method).parameters, '并行融合源码尚未同步'
    expected['DFINE']['aqs_l2_local_fusion'] = 'parallel'
for item in (expected, trial):
    item.pop('__include__', None); item.pop('output_dir', None)
assert trial == expected, '组合配置改动了其他模块、损失、数据或训练设置'
assert trial['epochs'] == 110 and trial['num_classes'] == 3
assert trial['train_dataloader']['collate_fn']['stop_epoch'] == 100
assert trial['train_dataloader']['dataset']['transforms']['policy']['epoch'] == 100
assert trial['train_dataloader']['total_batch_size'] == trial['val_dataloader']['total_batch_size'] == 32
order = 'L2 -> (AQS增量 + BoxLocal增量) -> L3' if parallel else 'Local-L2 -> AQS-L2 -> L3'
print(f'检查通过：{order}；110/100、batch32、seed0、原损失。', flush=True)
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
print('数据、原损失及CUDA检查通过。', flush=True)
PY

"$combo_conda" run --no-capture-output -n "$combo_env" python - "$combo_port" <<'PORT'
import socket, sys
with socket.socket() as server:
    server.bind(('127.0.0.1', int(sys.argv[1])))
print('训练端口可用。', flush=True)
PORT

mkdir -p -- "$(dirname "$combo_output")"
mkdir -- "$combo_output"
trap 'combo_status=$?; printf "组合实验未完成，退出码=%s；保留结果：%s\n" "$combo_status" "$combo_output" >&2; exit "$combo_status"' ERR
"$combo_conda" run --no-capture-output -n "$combo_env" python - "$combo_config" "$combo_output" <<'SNAPSHOT'
import hashlib, json, sys
from pathlib import Path
import yaml
from src.core.yaml_utils import load_config
config, output = Path(sys.argv[1]), Path(sys.argv[2])
files = {'train.py', 'src/zoo/dfine/dfine.py', 'src/zoo/dfine/dfine_decoder.py',
         'src/zoo/dfine/night_probes.py', 'tools/diagnostics/run_scbs_aqs_box_local_l2_110.sh'}
def includes(path):
    files.add(str(path))
    for child in yaml.safe_load(path.read_text()).get('__include__', []):
        includes(path.parent / child)
includes(config)
resolved = load_config(str(config), cfg={})
fusion = resolved['DFINE'].get('aqs_l2_local_fusion', 'serial')
if fusion == 'parallel':
    files.add('tools/diagnostics/run_scbs_aqs_box_local_l2_parallel_110.sh')
snapshot = dict(seed=0, initialization='fresh joint training; original HGNet pretraining only; no -r/-t',
    fusion=fusion,
    order=(['Decoder L2 original Q and scores', 'AQS(Q,S) and BoxLocal(Q,B,P3)',
            'Q + AQS delta + BoxLocal delta', 'Decoder L3'] if fusion == 'parallel' else
           ['Decoder L2', 'BoxLocal-L2', 'AQS-L2', 'Decoder L3']), resolved_config=resolved,
    source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sorted(files)},
    annotation_sha256={split: hashlib.sha256(Path(resolved[f'{split}_dataloader']['dataset']['ann_file']).read_bytes()).hexdigest()
                       for split in ('train', 'val')})
with (output / 'experiment_inputs.json').open('x', encoding='utf-8') as stream:
    json.dump(snapshot, stream, ensure_ascii=False, indent=2)
with (output / 'experiment_config.yml').open('x', encoding='utf-8') as stream:
    stream.write(config.read_text())
SNAPSHOT
printf '\n开始联合训练；不运行整套合成测试或ON/OFF。\n'
"${combo_train[@]}" 2>&1 | tee "$combo_output/console.log"
test -s "$combo_output/best_stg2.pth"
if [[ "$combo_diagnostics" == 0 ]]; then
  printf '\n训练完成；日志、权重和配置已保存：%s\n' "$combo_output"
  exit 0
fi

printf '\n可选：本组合最佳EMA权重导出完整诊断，不执行ON/OFF。\n'
"$combo_conda" run --no-capture-output -n "$combo_env" \
torchrun --master_addr=127.0.0.1 --master_port="$combo_port" --nproc_per_node=1 \
train.py -c "$combo_config" -r "$combo_output/best_stg2.pth" --test-only \
--output-dir "$combo_output/diagnostics_eval/evaluation" --use-amp --seed=0 \
-u export_confusion_matrix=true export_query_diagnostics=true \
query_diagnostic_conf_thresh=0.5 query_diagnostic_iou_thresh=0.5 query_diagnostic_neighbor_iou_thresh=0.8 \
2>&1 | tee "$combo_output/diagnostics_console.log"

printf '\n离线分析；后续与原基线、BoxLocal单模块和AQS单模块分别比较。\n'
"$combo_conda" run --no-capture-output -n "$combo_env" \
python tools/diagnostics/analyze_detection_diagnostics.py \
--input-dir "$combo_output/diagnostics_eval/evaluation" --output-dir "$combo_output/diagnostics_eval/analysis" \
2>&1 | tee "$combo_output/analysis_console.log"
test -s "$combo_output/diagnostics_eval/analysis/coco_per_class_and_size.json"
test -s "$combo_output/diagnostics_eval/analysis/best_query_cross_classification.json"
printf '\n组合实验完成，诊断目录：%s/diagnostics_eval\n' "$combo_output"
