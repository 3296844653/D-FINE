#!/usr/bin/env bash
# Diagnose the EXISTING RW Local Detail experiment. Never train or rescore.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

rw_diag_fail() { printf '%s\n' "错误：$*" >&2; exit 1; }
[[ "$#" -le 1 ]] || rw_diag_fail '用法：bash tools/diagnostics/run_scbs_rw_local_detail_diagnostics.sh [新输出目录]'
rw_diag_run='/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_rw_local_detail_seed0_run1'
rw_diag_checkpoint="${RW_DETAIL_DIAG_CHECKPOINT:-$rw_diag_run/best_stg2.pth}"
rw_diag_output="${1:-$rw_diag_run/diagnostics_eval}"
rw_diag_config='configs/dfine/dfine_s_scbs_hrw_110_rw_local_detail_diagnostics.yml'
rw_diag_conda="${RW_DETAIL_DIAG_CONDA:-${CONDA_EXE:-conda}}"
rw_diag_env="${RW_DETAIL_DIAG_ENV:-wq}"
rw_diag_gpu="${RW_DETAIL_DIAG_GPU:-0}"
[[ "$rw_diag_gpu" =~ ^[0-9]+$ ]] || rw_diag_fail 'GPU必须是单个非负整数'
[[ "$rw_diag_checkpoint" == /* && "$rw_diag_output" == /* ]] || rw_diag_fail '权重和输出必须是绝对路径'
[[ ! -e "$rw_diag_output" && ! -L "$rw_diag_output" ]] || rw_diag_fail "输出已存在，保留旧结果：${rw_diag_output}；请传入新目录"
test -f "$rw_diag_checkpoint" || rw_diag_fail "缺少局部细节实验自己的最佳权重：$rw_diag_checkpoint"
command -v "$rw_diag_conda" >/dev/null 2>&1 || rw_diag_fail "找不到Conda：$rw_diag_conda"
export CUDA_VISIBLE_DEVICES="$rw_diag_gpu" PYTHONUNBUFFERED=1 MPLBACKEND=Agg

"$rw_diag_conda" run --no-capture-output -n "$rw_diag_env" \
python - "$rw_diag_config" "$rw_diag_checkpoint" "$rw_diag_output" <<'PY'
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import torch
from src.core import YAMLConfig
from src.core.yaml_utils import load_config

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

config_path, checkpoint_path, output_path = map(Path, sys.argv[1:])
for filename in ('train.py', 'src/zoo/dfine/rw_local_detail.py',
                 'src/solver/query_diagnostics.py', 'src/solver/confusion_export.py',
                 'tools/diagnostics/analyze_detection_diagnostics.py'):
    assert Path(filename).is_file(), f'缺少文件：{filename}'
diag = load_config(str(config_path), cfg={})
expected = copy.deepcopy(load_config('configs/dfine/dfine_s_scbs_hrw_110.yml', cfg={}))
expected['HGNetv2']['return_idx'] = [0, 1, 2, 3]
expected['DFINETransformer'].update(
    use_rw_local_detail=True, rw_detail_p2_channels=64, rw_detail_dim=64,
    rw_detail_grid_size=5, rw_detail_num_heads=4, rw_detail_residual_scale=.1,
    rw_detail_class_ids=[1, 2],
)
expected.update(export_confusion_matrix=True, export_query_diagnostics=True,
                query_diagnostic_conf_thresh=.5, query_diagnostic_iou_thresh=.5,
                query_diagnostic_neighbor_iou_thresh=.8)
for config in (expected, diag):
    config.pop('__include__', None)
    config.pop('output_dir', None)
assert diag == expected, '不是原独立RW Local Detail实验配置，或修改了模型/损失/数据'
assert diag['epochs'] == 110 and diag['num_classes'] == 3 and not diag['remap_mscoco_category']
assert diag['use_ema'], '必须使用EMA，与原最佳轮次评估一致'
assert diag['train_dataloader']['collate_fn']['stop_epoch'] == 100
assert diag['train_dataloader']['dataset']['transforms']['policy']['epoch'] == 100
assert diag['train_dataloader']['total_batch_size'] == diag['val_dataloader']['total_batch_size'] == 32
data = diag['val_dataloader']['dataset']
annotation_path = Path(data['ann_file'])
dataset = json.loads(annotation_path.read_text(encoding='utf-8'))
assert sorted((c['id'], c['name']) for c in dataset['categories']) == [(0, 'hand-raising'), (1, 'read'), (2, 'write')]
images = {image['id']: image for image in dataset['images']}
assert images and len(images) == len(dataset['images']), '验证集为空或image_id重复'
assert Path(data['img_folder']).is_dir(), '缺少验证图片目录'
for image in images.values():
    assert (Path(data['img_folder']) / image['file_name']).is_file(), f"缺少图片：{image['file_name']}"
annotations = defaultdict(list)
for item in dataset['annotations']:
    assert item['image_id'] in images and item['category_id'] in (0, 1, 2), '标注图片/类别编号无效'
    if not item.get('iscrowd', 0):
        annotations[item['image_id']].append(item)
class_counts = Counter({0: 0, 1: 0, 2: 0})
for image_id, items in annotations.items():
    image = images[image_id]
    boxes = torch.tensor([item['bbox'] for item in items], dtype=torch.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=image['width'])
    boxes[:, 1::2].clamp_(min=0, max=image['height'])
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    class_counts.update(item['category_id'] for item, valid in zip(items, keep.tolist()) if valid)

# This checkpoint has not been uploaded: snapshot its hash, do not invent one.
checkpoint_sha256 = sha256(checkpoint_path)
annotation_sha256 = sha256(annotation_path)
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
assert 0 <= checkpoint.get('last_epoch', -1) < 110, '不是110轮训练权重'
assert 'model' in checkpoint and 'module' in checkpoint.get('ema', {}), '缺少model或EMA权重'
model_config = YAMLConfig(str(config_path))
model_config.yaml_cfg['HGNetv2']['pretrained'] = False
model = model_config.model
assert model.decoder.use_rw_local_detail and not model.decoder.use_aqs_refine
state_statistics = {}
for name, state in (('model', checkpoint['model']), ('ema', checkpoint['ema']['module'])):
    assert 'decoder.rw_local_detail.fuse.up.weight' in state, '不是局部细节实验权重，不能使用基线/AQS'
    model.load_state_dict(state, strict=True)
    weight = state['decoder.rw_local_detail.fuse.up.weight'].float()
    bias = state['decoder.rw_local_detail.fuse.up.bias'].float()
    assert torch.isfinite(weight).all() and torch.isfinite(bias).all(), '局部分支权重非有限值'
    state_statistics[name] = dict(final_projection_weight_l2=float(weight.norm()),
                                  final_projection_weight_max_abs=float(weight.abs().max()),
                                  final_projection_bias_l2=float(bias.norm()))
inputs = dict(checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=checkpoint_sha256,
              checkpoint_epoch=checkpoint['last_epoch'], annotation_file=str(annotation_path.resolve()),
              annotation_sha256=annotation_sha256, validation_dataset=data, images=len(images),
              GT=sum(class_counts.values()), GT_by_category=dict(class_counts), weight_source='ema',
              module='RWLocalDetailAdapter', module_state_statistics=state_statistics,
              confidence_threshold=.5, iou_threshold=.5, neighbor_iou_threshold=.8)
output_path.mkdir(parents=True, exist_ok=False)
(output_path / 'export_inputs.json').write_text(json.dumps(inputs, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('检查通过：独立局部细节模块、严格加载model/EMA、110/100、原损失和完整验证集。', flush=True)
print('权重epoch：', inputs['checkpoint_epoch'], '；图像：', inputs['images'], '；GT：', inputs['GT'], flush=True)
PY

mkdir -- "$rw_diag_output/logs"
trap 'rw_diag_status=$?; printf "%s\n" "诊断未完成，退出码=${rw_diag_status}；保留部分结果：${rw_diag_output}" >&2; exit "$rw_diag_status"' ERR
printf '%s\n' '' '[1/2] 只验证局部细节实验：混淆矩阵、全部query与原COCO评估。'
"$rw_diag_conda" run --no-capture-output -n "$rw_diag_env" \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py -c "$rw_diag_config" -r "$rw_diag_checkpoint" --test-only \
--output-dir "$rw_diag_output/evaluation" --use-amp --seed=0 \
2>&1 | tee "$rw_diag_output/logs/evaluation.log"

printf '%s\n' '' '[2/2] 分析读写混淆、类别×尺度COCO指标和候选缺口。'
"$rw_diag_conda" run --no-capture-output -n "$rw_diag_env" \
python tools/diagnostics/analyze_detection_diagnostics.py \
--input-dir "$rw_diag_output/evaluation" --output-dir "$rw_diag_output/analysis" \
2>&1 | tee "$rw_diag_output/logs/analysis.log"

"$rw_diag_conda" run --no-capture-output -n "$rw_diag_env" python - "$rw_diag_output" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

root = Path(sys.argv[1])
inputs = json.loads((root / 'export_inputs.json').read_text(encoding='utf-8'))
required = (
    'evaluation/query_diagnostics/metadata.json', 'evaluation/query_diagnostics/queries.csv',
    'evaluation/query_diagnostics/gt_coverage.csv', 'evaluation/query_diagnostics/selected_predictions.csv',
    'evaluation/query_diagnostics/nearby_predictions.csv',
    'evaluation/confusion_matrix/confusion_matrix.json', 'evaluation/confusion_matrix/confusion_matrix.txt',
    'evaluation/validation_metrics.json', 'evaluation/per_class_metrics.json', 'evaluation/eval.pth',
    'analysis/summary.json', 'analysis/summary.txt', 'analysis/gt_records.jsonl',
    'analysis/prediction_records.jsonl', 'analysis/best_query_cross_classification.json',
    'analysis/best_query_cross_classification.jsonl', 'analysis/gt_best_iou_distribution.json',
    'analysis/gt_size_statistics.json', 'analysis/coco_pr_curves.json',
    'analysis/coco_per_class_and_size.json', 'analysis/score_distributions_and_quality.json',
    'analysis/coco_pr_iou0.5.png', 'analysis/coco_pr_iou0.75.png',
    'analysis/diagnostic_tp_fp_scores.png', 'analysis/score_quality_relationship.png',
    'analysis/gt_best_iou_distribution.png',
)
for filename in required:
    assert (root / filename).is_file() and (root / filename).stat().st_size > 0, f'未完整导出：{filename}'
query = json.loads((root / 'evaluation/query_diagnostics/metadata.json').read_text())
summary = json.loads((root / 'analysis/summary.json').read_text())
matrix = json.loads((root / 'evaluation/confusion_matrix/confusion_matrix.json').read_text())
assert query['complete'] and query['schema_version'] == 2, 'query导出不完整或schema过旧'
assert query['num_classes'] == 3 and query['postprocessor_top_k'] == 300
assert summary['weight_source'] == matrix['metadata']['weight_source'] == inputs['weight_source'] == 'ema'
for report in (query, summary, matrix['metadata']):
    assert Path(report['checkpoint']).resolve() == Path(inputs['checkpoint']), '步骤之间权重路径不同'
assert query['checkpoint_sha256'] == summary['checkpoint_sha256'] == inputs['checkpoint_sha256'] == sha256(inputs['checkpoint']), '权重指纹改变或不一致'
assert inputs['annotation_sha256'] == sha256(inputs['annotation_file']), '诊断期间标注发生变化'
assert matrix['metadata']['validation_dataset'] == inputs['validation_dataset'], '验证集配置改变'
assert Path(summary['annotation_file']).resolve() == Path(inputs['annotation_file'])
assert query['images'] == summary['images'] == inputs['images']
assert query['queries'] == summary['queries'] == 300 * inputs['images'], '不是完整300-query导出'
assert len(list((root / 'evaluation/query_diagnostics/raw_queries').glob('*.pt'))) == inputs['images']
assert summary['GT'] == matrix['GT_count'] == inputs['GT']
assert summary['confidence_threshold'] == matrix['confidence_threshold'] == query['confidence_threshold'] == .5
assert summary['IoU_threshold'] == matrix['iou_threshold'] == query['gt_iou_threshold'] == .5
assert query['neighbor_iou_threshold'] == .8
assert matrix['matching'] == 'class-blind greedy highest-IoU one-to-one (existing Validator)'
assert matrix['category_ids'] == [0, 1, 2]
assert matrix['row_labels'] == ['hand-raising', 'read', 'write', 'unmatched_predictions']
assert matrix['column_labels'] == ['hand-raising', 'read', 'write', 'unmatched_GT']
counts = matrix['matrix_counts']
assert len(counts) == 4 and all(len(row) == 4 for row in counts)
assert counts[3][3] == 0 and all(type(n) is int and n >= 0 for row in counts for n in row)
for category in range(3):
    assert sum(counts[category]) == inputs['GT_by_category'][str(category)]
assert matrix['predictions_after_threshold'] == sum(sum(row[:3]) for row in counts) == summary['selected_pairs_above_confidence']
with (root / 'analysis/gt_records.jsonl').open(encoding='utf-8') as handle:
    assert sum(1 for line in handle if line.strip()) == inputs['GT']
scale_report = json.loads((root / 'analysis/coco_per_class_and_size.json').read_text())
assert scale_report['metrics'] == summary['saved_COCO_per_class_and_size']
assert set(scale_report['metrics']) == {'0', '1', '2'}
for category in scale_report['metrics'].values():
    assert set(category) == {'all', 'small', 'medium', 'large'}
    for metrics in category.values():
        assert set(metrics) == {'AP', 'AP50', 'AP75', 'AR100'}
        assert all(value is None or (type(value) in (int, float) and 0 <= value <= 1)
                   for value in metrics.values())
report = dict(inputs, complete=True, experiment='110-epoch RW Local Detail diagnostics',
              queries=query['queries'], matching=matrix['matching'],
              note='Validation and offline analysis only; no training, rescoring or annotation edits. Candidate evidence is not COCO TP/FN or causal attribution.')
(root / 'workflow_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
text = '110轮 RW Local Detail 完整诊断\n' + (root / 'analysis/summary.txt').read_text(encoding='utf-8')
(root / 'rw_local_detail_diagnostics_summary.txt').write_text(text, encoding='utf-8')
print('\n局部细节诊断完整结束：', root, flush=True)
print('先查看：', root / 'rw_local_detail_diagnostics_summary.txt', flush=True)
PY
