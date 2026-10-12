#!/usr/bin/env bash
# Fresh, single-module SCB-S experiment. Never resume/overwrite earlier results.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

rw_detail_fail() { printf '错误：%s\n' "$*" >&2; exit 1; }
[[ "$#" -le 1 ]] || rw_detail_fail '用法：bash tools/diagnostics/run_scbs_rw_local_detail_110.sh [新输出目录]'
rw_detail_output="${1:-/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_rw_local_detail_seed0_run1}"
rw_detail_conda="${CONDA_EXE:-conda}"
rw_detail_env="${RW_DETAIL_ENV:-wq}"
rw_detail_gpu="${RW_DETAIL_GPU:-0}"
rw_detail_config='configs/dfine/dfine_s_scbs_hrw_110_rw_local_detail.yml'
[[ "$rw_detail_output" == /* ]] || rw_detail_fail '输出目录必须是绝对路径'
[[ ! -e "$rw_detail_output" ]] || rw_detail_fail "结果目录已存在，保留旧结果：${rw_detail_output}；请指定新目录"
[[ "$rw_detail_gpu" =~ ^[0-9]+$ ]] || rw_detail_fail 'GPU必须是单个非负整数'
command -v "$rw_detail_conda" >/dev/null 2>&1 || rw_detail_fail '找不到Conda'
export CUDA_VISIBLE_DEVICES="$rw_detail_gpu" MPLBACKEND=Agg PYTHONUNBUFFERED=1

"$rw_detail_conda" run --no-capture-output -n "$rw_detail_env" python - "$rw_detail_config" <<'PY'
import copy
import inspect
import json
import sys
from pathlib import Path
import torch
from src.core.yaml_utils import load_config
from src.zoo.dfine.rw_local_detail import RWLocalDetailAdapter
from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.dfine import DFINE

assert Path('train.py').is_file(), '请在完整项目中运行'
assert 'use_rw_local_detail' in inspect.signature(DFINETransformer).parameters, 'Decoder源码未同步'
assert 'use_rw_local_detail' in inspect.getsource(DFINE.forward), 'P2路由源码未同步'
trial = load_config(sys.argv[1], cfg={})
expected = copy.deepcopy(load_config('configs/dfine/dfine_s_scbs_hrw_110.yml', cfg={}))
expected['HGNetv2']['return_idx'] = [0, 1, 2, 3]
expected['DFINETransformer'].update(
    use_rw_local_detail=True, rw_detail_p2_channels=64, rw_detail_dim=64,
    rw_detail_grid_size=5, rw_detail_num_heads=4, rw_detail_residual_scale=.1,
    rw_detail_class_ids=[1, 2],
)
for config in (expected, trial):
    config.pop('__include__', None)
    config.pop('output_dir', None)
assert trial == expected, '配置不是独立的局部细节实验，或修改了原训练/损失流程'
assert all(not value for key, value in trial['DFINETransformer'].items()
           if key.startswith('use_') and key != 'use_rw_local_detail'), '启用了其他query模块'
assert all(not value for key, value in trial['DFINECriterion'].items()
           if key.startswith('use_')), '启用了额外损失函数'
assert trial['epochs'] == 110 and trial['num_classes'] == 3
assert trial['train_dataloader']['total_batch_size'] == 32
assert trial['train_dataloader']['collate_fn']['stop_epoch'] == 100
for split in ('train', 'val'):
    data = trial[f'{split}_dataloader']['dataset']
    root = Path(data['img_folder'])
    assert root.is_dir(), f'缺少图片目录：{root}'
    annotation = json.loads(Path(data['ann_file']).read_text(encoding='utf-8'))
    assert sorted((c['id'], c['name']) for c in annotation['categories']) == [(0, 'hand-raising'), (1, 'read'), (2, 'write')]
    assert annotation['images'], f'{split}集为空'
    for image in annotation['images']:
        assert (root / image['file_name']).is_file(), f"缺少图片：{image['file_name']}"
assert torch.cuda.is_available(), 'CUDA不可用，未启动CPU完整训练'
print('检查通过：独立结构 / 110轮 / stop=100 / batch=32 / 原损失 / seed=0。', flush=True)
print('GPU：', torch.cuda.get_device_name(0), '；请确认没有其他训练占用。', flush=True)
PY

mkdir -p -- "$(dirname "$rw_detail_output")"
mkdir -- "$rw_detail_output"
trap 'rw_detail_status=$?; printf "训练未成功完成，退出码=%s；保留日志：%s\n" "$rw_detail_status" "$rw_detail_output/console.log" >&2; exit "$rw_detail_status"' ERR
printf '\n开始独立局部细节训练：%s\n' "$rw_detail_output"
"$rw_detail_conda" run --no-capture-output -n "$rw_detail_env" \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py -c "$rw_detail_config" --output-dir "$rw_detail_output" --use-amp --seed=0 \
2>&1 | tee "$rw_detail_output/console.log"
test -f "$rw_detail_output/best_stg2.pth"
printf '\n训练完成，最佳权重：%s\n' "$rw_detail_output/best_stg2.pth"
