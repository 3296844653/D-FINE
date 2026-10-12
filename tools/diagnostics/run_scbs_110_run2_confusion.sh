#!/usr/bin/env bash
# Export baseline Run2's existing confusion matrix; NEVER train or rescore.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

run2_confusion_fail() { printf '错误：%s\n' "$*" >&2; exit 1; }
[[ "$#" -le 1 ]] || run2_confusion_fail '用法：bash tools/diagnostics/run_scbs_110_run2_confusion.sh [新输出目录]'
run2_confusion_run='/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_seed0_run2'
run2_confusion_checkpoint="${SCBS_RUN2_CHECKPOINT:-$run2_confusion_run/best_stg2.pth}"
run2_confusion_output="${1:-$run2_confusion_run/confusion_eval}"
run2_confusion_config='configs/dfine/dfine_s_scbs_hrw_110_run2_confusion.yml'
run2_confusion_conda="${SCBS_RUN2_CONDA:-${CONDA_EXE:-conda}}"
run2_confusion_env="${SCBS_RUN2_ENV:-wq}"
run2_confusion_gpu="${SCBS_RUN2_GPU:-0}"
[[ "$run2_confusion_gpu" =~ ^[0-9]+$ ]] || run2_confusion_fail 'GPU必须是单个非负整数'
[[ "$run2_confusion_checkpoint" == /* && "$run2_confusion_output" == /* ]] || run2_confusion_fail '权重和输出必须是绝对路径'
[[ ! -e "$run2_confusion_output" && ! -L "$run2_confusion_output" ]] || run2_confusion_fail "输出目录已存在，保留旧结果：${run2_confusion_output}；请传入新的输出目录"
test -f "$run2_confusion_checkpoint" || run2_confusion_fail "缺少110轮基线Run2权重：$run2_confusion_checkpoint"
command -v "$run2_confusion_conda" >/dev/null 2>&1 || run2_confusion_fail "找不到Conda：$run2_confusion_conda"
export CUDA_VISIBLE_DEVICES="$run2_confusion_gpu" PYTHONUNBUFFERED=1 MPLBACKEND=Agg

# Check validation data only. Create a fresh output only after every check passes.
"$run2_confusion_conda" run --no-capture-output -n "$run2_confusion_env" \
python - "$run2_confusion_config" "$run2_confusion_checkpoint" "$run2_confusion_output" <<'PY'
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import torch
from src.core.yaml_utils import load_config

config_path, checkpoint_path, output_path = map(Path, sys.argv[1:])
for filename in ('train.py', 'src/solver/confusion_export.py'):
    assert Path(filename).is_file(), f'缺少脚本：{filename}'
diag = load_config(str(config_path), cfg={})
expected = copy.deepcopy(load_config('configs/dfine/dfine_s_scbs_hrw_110.yml', cfg={}))
expected['export_confusion_matrix'] = True
for config in (diag, expected):
    config.pop('__include__', None)
assert diag == expected, '诊断配置改变了110轮基线的模型/数据/评估流程'
assert not diag.get('export_query_diagnostics', False), '本次只导出混淆矩阵'
assert diag['epochs'] == 110 and diag['num_classes'] == 3 and not diag['remap_mscoco_category']
assert diag['train_dataloader']['collate_fn']['stop_epoch'] == 100
assert diag['train_dataloader']['dataset']['transforms']['policy']['epoch'] == 100
assert diag['train_dataloader']['total_batch_size'] == diag['val_dataloader']['total_batch_size'] == 32
data = diag['val_dataloader']['dataset']
annotation_path = Path(data['ann_file'])
dataset = json.loads(annotation_path.read_text(encoding='utf-8'))
assert sorted((c['id'], c['name']) for c in dataset['categories']) == [(0, 'hand-raising'), (1, 'read'), (2, 'write')]
assert dataset['images'], '验证集为空'
images = {image['id']: image for image in dataset['images']}
assert len(images) == len(dataset['images']), '验证集有重复image_id'
assert Path(data['img_folder']).is_dir(), f"缺少验证图片目录：{data['img_folder']}"
for image in dataset['images']:
    assert (Path(data['img_folder']) / image['file_name']).is_file(), f"缺少验证图片：{image['file_name']}"
annotations = defaultdict(list)
for item in dataset['annotations']:
    assert item['image_id'] in images and item['category_id'] in (0, 1, 2), '标注的图片或类别编号无效'
    if not item.get('iscrowd', 0):
        annotations[item['image_id']].append(item)
# Mirror CocoDetection's float32 clipping/nonempty-box rule for count verification.
class_counts = Counter({0: 0, 1: 0, 2: 0})
for image_id, items in annotations.items():
    image = images[image_id]
    boxes = torch.tensor([item['bbox'] for item in items], dtype=torch.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=image['width'])
    boxes[:, 1::2].clamp_(min=0, max=image['height'])
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    class_counts.update(item['category_id'] for item, valid in zip(items, keep.tolist()) if valid)

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
assert 0 <= checkpoint.get('last_epoch', -1) < 110, '不是110轮实验的训练权重'
state = checkpoint['model']
assert state['decoder.dec_score_head.2.weight'].shape[0] == 3, '权重类别数不匹配'
assert not any('query_cls_refiner' in key or 'aqs_refiners' in key for key in state), '这是AQS权重，不是基线'
assert 'ema' in checkpoint and 'module' in checkpoint['ema'], '缺少EMA权重，无法与本次AQS-R1的EMA评估直接对照'
assert checkpoint['ema']['module']['decoder.dec_score_head.2.weight'].shape[0] == 3, 'EMA类别数不匹配'
digest = hashlib.sha256()
with checkpoint_path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(chunk)
inputs = dict(checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=digest.hexdigest(),
              annotation_file=str(annotation_path.resolve()),
              annotation_sha256=hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
              validation_dataset=data, images=len(images),
              GT=sum(class_counts.values()), GT_by_category=dict(class_counts),
              weight_source='ema', confidence_threshold=.5, iou_threshold=.5)
output_path.mkdir(parents=True, exist_ok=False)
(output_path / 'export_inputs.json').write_text(json.dumps(inputs, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('检查通过：110轮基线 / EMA / 原验证集 / 只导出混淆矩阵。', flush=True)
print('权重：', checkpoint_path, flush=True)
print('图像：', inputs['images'], '；GT：', inputs['GT'], '。请确认GPU空闲。', flush=True)
PY

mkdir -- "$run2_confusion_output/logs"
trap 'run2_confusion_status=$?; printf "导出中断，退出码=%s；保留部分结果：%s\n" "$run2_confusion_status" "$run2_confusion_output" >&2; exit "$run2_confusion_status"' ERR
printf '\n正在评估110轮基线Run2，不训练；页面实时显示日志。\n'
"$run2_confusion_conda" run --no-capture-output -n "$run2_confusion_env" \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py -c "$run2_confusion_config" -r "$run2_confusion_checkpoint" --test-only \
--output-dir "$run2_confusion_output" --use-amp --seed=0 \
2>&1 | tee "$run2_confusion_output/logs/evaluation.log"

"$run2_confusion_conda" run --no-capture-output -n "$run2_confusion_env" \
python - "$run2_confusion_output" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
inputs = json.loads((root / 'export_inputs.json').read_text(encoding='utf-8'))
for filename in ('confusion_matrix/confusion_matrix.json', 'confusion_matrix/confusion_matrix.txt',
                 'confusion_matrix/confusion_matrix_counts.png', 'confusion_matrix/confusion_matrix_row_normalized.png',
                 'validation_metrics.json', 'per_class_metrics.json', 'eval.pth'):
    assert (root / filename).is_file() and (root / filename).stat().st_size > 0, f'未完整导出：{filename}'
matrix = json.loads((root / 'confusion_matrix/confusion_matrix.json').read_text(encoding='utf-8'))
assert matrix['category_ids'] == [0, 1, 2], '混淆矩阵类别顺序错误'
assert matrix['row_labels'] == ['hand-raising', 'read', 'write', 'unmatched_predictions']
assert matrix['column_labels'] == ['hand-raising', 'read', 'write', 'unmatched_GT']
assert matrix['confidence_threshold'] == matrix['iou_threshold'] == .5, '评估阈值与AQS-R1不同'
assert matrix['matching'] == 'class-blind greedy highest-IoU one-to-one (existing Validator)', '匹配规则改变了'
assert Path(matrix['metadata']['checkpoint']).resolve() == Path(inputs['checkpoint']), '不是同一Run2权重'
assert matrix['metadata']['weight_source'] == inputs['weight_source'] == 'ema', '未使用EMA评估'
assert matrix['metadata']['validation_dataset'] == inputs['validation_dataset'], '验证集配置不同'
counts = matrix['matrix_counts']
assert len(counts) == 4 and all(len(row) == 4 for row in counts), '矩阵尺寸错误'
assert counts[3][3] == 0 and all(isinstance(n, int) and n >= 0 for row in counts for n in row), '矩阵数值无效'
assert matrix['GT_count'] == inputs['GT'] == sum(sum(row) for row in counts[:3]), 'GT计数不一致'
for i in range(3):
    assert sum(counts[i]) == inputs['GT_by_category'][str(i)], '各类别GT计数不一致'
assert matrix['predictions_after_threshold'] == sum(sum(row[:3]) for row in counts), '预测计数不一致'
digest = hashlib.sha256()
with Path(inputs['checkpoint']).open('rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(chunk)
assert digest.hexdigest() == inputs['checkpoint_sha256'], '评估期间权重发生了变化'
assert hashlib.sha256(Path(inputs['annotation_file']).read_bytes()).hexdigest() == inputs['annotation_sha256'], '评估期间标注发生了变化'
report = dict(inputs, complete=True, experiment='110-epoch baseline Run2',
              matching=matrix['matching'], matrix='confusion_matrix/confusion_matrix.json',
              note='Validation only; no retraining, rescoring, query export or annotation changes. Diagnostic counts are not COCO AP.')
(root / 'workflow_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('\nRun2混淆矩阵导出完成：', root / 'confusion_matrix', flush=True)
print((root / 'confusion_matrix/confusion_matrix.txt').read_text(encoding='utf-8'), flush=True)
PY
