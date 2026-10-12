#!/usr/bin/env bash
# Export R1's existing checkpoint; NEVER start training or rescore predictions.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

aqs_diag_fail() { printf '错误：%s\n' "$*" >&2; exit 1; }
[[ "$#" -le 1 ]] || aqs_diag_fail '用法：bash tools/diagnostics/run_scbs_aqs_r1_diagnostics.sh [新输出目录]'
aqs_diag_run='/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_aqs_layer2_r010_seed0_run1'
aqs_diag_checkpoint="${AQS_DIAG_CHECKPOINT:-$aqs_diag_run/best_stg2.pth}"
aqs_diag_output="${1:-$aqs_diag_run/diagnostics_eval}"
aqs_diag_config='configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010_diagnostics.yml'
aqs_diag_training='configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml'
aqs_diag_env="${AQS_DIAG_ENV:-wq}"
aqs_diag_conda="${AQS_DIAG_CONDA:-${CONDA_EXE:-conda}}"
aqs_diag_gpu="${AQS_DIAG_GPU:-0}"
[[ "$aqs_diag_gpu" =~ ^[0-9]+$ ]] || aqs_diag_fail 'GPU必须是单个非负整数'
[[ "$aqs_diag_output" == /* && "$aqs_diag_checkpoint" == /* ]] || aqs_diag_fail '输出和权重必须是绝对路径'
[[ ! -e "$aqs_diag_output" ]] || aqs_diag_fail "输出目录已存在，保留旧结果：${aqs_diag_output}；请传入新的输出目录"
test -f "$aqs_diag_checkpoint" || aqs_diag_fail "缺少R1权重：$aqs_diag_checkpoint"
command -v "$aqs_diag_conda" >/dev/null 2>&1 || aqs_diag_fail "找不到Conda：$aqs_diag_conda"
export CUDA_VISIBLE_DEVICES="$aqs_diag_gpu" PYTHONUNBUFFERED=1 MPLBACKEND=Agg

# Check only validation data: this workflow never needs the training images.
"$aqs_diag_conda" run --no-capture-output -n "$aqs_diag_env" \
python - "$aqs_diag_config" "$aqs_diag_training" "$aqs_diag_checkpoint" <<'PY'
import copy
import json
import sys
from pathlib import Path
import torch
from src.core.yaml_utils import load_config

for filename in ('train.py', 'tools/diagnostics/analyze_detection_diagnostics.py',
                 'tools/diagnostics/export_aqs_parameters.py'):
    assert Path(filename).is_file(), f'缺少脚本：{filename}'
diag = load_config(sys.argv[1], cfg={})
expected = copy.deepcopy(load_config(sys.argv[2], cfg={}))
options = dict(export_confusion_matrix=True, export_query_diagnostics=True,
               query_diagnostic_conf_thresh=.5, query_diagnostic_iou_thresh=.5,
               query_diagnostic_neighbor_iou_thresh=.8)
expected.update(options)
for cfg in (expected, diag):
    cfg.pop('__include__', None)
assert diag == expected, '诊断配置改变了R1的模型/数据/评估流程'
decoder = diag['DFINETransformer']
assert decoder['use_aqs_refine'] and decoder['aqs_apply_layer'] == 2
assert decoder.get('aqs_apply_layers') is None, '不得使用双层AQS'
assert (decoder['aqs_threshold'], decoder['aqs_residual_init'], decoder['aqs_temperature']) == (.5, .1, .1)
assert diag['epochs'] == 110 and diag['num_classes'] == 3 and not diag['remap_mscoco_category']
assert diag['train_dataloader']['collate_fn']['stop_epoch'] == 100
assert diag['train_dataloader']['total_batch_size'] == diag['val_dataloader']['total_batch_size'] == 32
data = diag['val_dataloader']['dataset']
assert Path(data['img_folder']).is_dir(), f"缺少验证图片目录：{data['img_folder']}"
dataset = json.loads(Path(data['ann_file']).read_text(encoding='utf-8'))
assert sorted((c['id'], c['name']) for c in dataset['categories']) == [(0, 'hand-raising'), (1, 'read'), (2, 'write')]
assert dataset['images'], '验证集为空'
for image in dataset['images']:
    assert (Path(data['img_folder']) / image['file_name']).is_file(), f"缺少验证图片：{image['file_name']}"
# Do not automatically fall back to arbitrary pickle execution.
checkpoint = torch.load(sys.argv[3], map_location='cpu', weights_only=True)
assert 0 <= checkpoint.get('last_epoch', -1) < 110, '不是110轮实验的训练权重'
state = checkpoint['model']
assert 'decoder.decoder.query_cls_refiner.threshold_logit' in state, '不是单层AQS权重'
assert not any('aqs_refiners.layer' in key for key in state), '不得使用双层AQS权重'
assert state['decoder.dec_score_head.2.weight'].shape[0] == 3, '权重类别数不匹配'
print('检查通过：R1 / L2 / 110轮，同一验证集、原后处理，conf=0.5、IoU=0.5。', flush=True)
print('权重epoch：', checkpoint['last_epoch'], '；配置初值不会覆盖已学习参数。请确认GPU空闲。', flush=True)
PY

# Atomic root creation: refuse even a partial earlier export, never overwrite it.
mkdir -- "$aqs_diag_output"
mkdir -- "$aqs_diag_output/logs"
trap 'aqs_diag_status=$?; printf "导出中断，退出码=%s；保留部分结果：%s\n" "$aqs_diag_status" "$aqs_diag_output" >&2; exit "$aqs_diag_status"' ERR
printf '\n[1/3] 只评估R1：导出混淆矩阵、全部300个query及原评估结果。\n'
"$aqs_diag_conda" run --no-capture-output -n "$aqs_diag_env" \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py -c "$aqs_diag_config" -r "$aqs_diag_checkpoint" --test-only \
--output-dir "$aqs_diag_output/evaluation" --use-amp --seed=0 \
2>&1 | tee "$aqs_diag_output/logs/evaluation.log"

printf '\n[2/3] 分析类别×尺度、best-IoU候选、PR和分数关系。\n'
"$aqs_diag_conda" run --no-capture-output -n "$aqs_diag_env" \
python tools/diagnostics/analyze_detection_diagnostics.py \
--input-dir "$aqs_diag_output/evaluation" --output-dir "$aqs_diag_output/analysis" \
2>&1 | tee "$aqs_diag_output/logs/analysis.log"

printf '\n[3/3] 读取同一最佳权重的model/EMA AQS参数。\n'
"$aqs_diag_conda" run --no-capture-output -n "$aqs_diag_env" \
python tools/diagnostics/export_aqs_parameters.py \
--checkpoint "$aqs_diag_checkpoint" --temperature 0.10 \
--output "$aqs_diag_output/aqs_parameters.json" \
2>&1 | tee "$aqs_diag_output/logs/aqs_parameters.log"

"$aqs_diag_conda" run --no-capture-output -n "$aqs_diag_env" \
python - "$aqs_diag_output" "$aqs_diag_checkpoint" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
root, checkpoint = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
query = json.loads((root / 'evaluation/query_diagnostics/metadata.json').read_text())
summary = json.loads((root / 'analysis/summary.json').read_text())
parameters = json.loads((root / 'aqs_parameters.json').read_text())
assert query['complete'], 'query导出未完整完成'
for report in (query, summary, parameters):
    assert Path(report['checkpoint']).resolve() == checkpoint, '不是同一R1权重'
digest = hashlib.sha256()
with checkpoint.open('rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(chunk)
assert query['checkpoint_sha256'] == summary['checkpoint_sha256'] == digest.hexdigest(), '权重哈希不一致'
assert summary['images'] == query['images'] and summary['queries'] == query['queries'], '导出计数不一致'
assert summary['queries'] == 300 * summary['images'], '不是完整300-query导出'
assert summary['weight_source'] in parameters['sources'], '缺少实际评估来源的AQS参数'
report = dict(complete=True, checkpoint=str(checkpoint), checkpoint_sha256=digest.hexdigest(),
              weight_source=summary['weight_source'], images=summary['images'], GT=summary['GT'],
              queries=summary['queries'], decoder_layer=2,
              note='Validation and offline analysis only; no retraining/rescoring. Not proof of a causal bottleneck.')
(root / 'workflow_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('\nR1诊断完成：', root, flush=True)
print('先查看：', root / 'analysis/summary.txt', flush=True)
PY
