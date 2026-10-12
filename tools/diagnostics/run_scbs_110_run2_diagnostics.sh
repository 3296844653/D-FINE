#!/usr/bin/env bash
# Run2 baseline: one validation + offline analysis, NEVER detector training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

run2_diag_fail() { printf '错误：%s\n' "$*" >&2; exit 1; }
[[ "$#" -le 1 ]] || run2_diag_fail '用法：bash tools/diagnostics/run_scbs_110_run2_diagnostics.sh [新输出目录]'
run2_diag_run='/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_seed0_run2'
run2_diag_checkpoint="${SCBS_RUN2_CHECKPOINT:-$run2_diag_run/best_stg2.pth}"
run2_diag_output="${1:-$run2_diag_run/diagnostics_export1}"
run2_diag_config='configs/dfine/dfine_s_scbs_hrw_110_run2_diagnostics.yml'
run2_diag_conda="${SCBS_RUN2_CONDA:-${CONDA_EXE:-conda}}"
run2_diag_env="${SCBS_RUN2_ENV:-wq}"
run2_diag_gpu="${SCBS_RUN2_GPU:-0}"
[[ "$run2_diag_gpu" =~ ^[0-9]+$ ]] || run2_diag_fail 'GPU必须是单个非负整数'
[[ "$run2_diag_checkpoint" == /* && "$run2_diag_output" == /* ]] || run2_diag_fail '权重和输出必须是绝对路径'
[[ ! -e "$run2_diag_output" && ! -L "$run2_diag_output" ]] || run2_diag_fail "输出目录已存在，保留旧结果：${run2_diag_output}；请传入新目录"
test -f "$run2_diag_checkpoint" || run2_diag_fail "缺少110轮基线Run2权重：$run2_diag_checkpoint"
command -v "$run2_diag_conda" >/dev/null 2>&1 || run2_diag_fail "找不到Conda：$run2_diag_conda"
export CUDA_VISIBLE_DEVICES="$run2_diag_gpu" PYTHONUNBUFFERED=1 MPLBACKEND=Agg

# Validate the *same* artifacts as the user's completed Run2 confusion export.
# Creating only the output snapshot follows all checks; no training data needed.
"$run2_diag_conda" run --no-capture-output -n "$run2_diag_env" \
python - "$run2_diag_config" "$run2_diag_checkpoint" "$run2_diag_output" <<'PY'
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import torch
from src.core import YAMLConfig
from src.core.yaml_utils import load_config

EXPECTED_CHECKPOINT_SHA256 = '4ff8f014c4497e2d918de11f3fcc6d741bcc19315a9283c8b7786162154ee271'
EXPECTED_ANNOTATION_SHA256 = '623bb8478049dd69d32a92183b0a322bdb8f79f72e6f6f52e491d48923bd5d37'

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

config_path, checkpoint_path, output_path = map(Path, sys.argv[1:])
for filename in ('train.py', 'src/solver/query_diagnostics.py', 'src/solver/confusion_export.py',
                 'tools/diagnostics/analyze_detection_diagnostics.py'):
    assert Path(filename).is_file(), f'缺少文件：{filename}'
diag = load_config(str(config_path), cfg={})
expected = copy.deepcopy(load_config('configs/dfine/dfine_s_scbs_hrw_110.yml', cfg={}))
expected.update(export_confusion_matrix=True, export_query_diagnostics=True,
                query_diagnostic_conf_thresh=.5, query_diagnostic_iou_thresh=.5,
                query_diagnostic_neighbor_iou_thresh=.8)
for config in (expected, diag):
    config.pop('__include__', None)
assert diag == expected, '诊断配置改变了110轮基线模型、损失或数据/评估流程'
assert diag['epochs'] == 110 and diag['num_classes'] == 3 and not diag['remap_mscoco_category']
assert diag['use_ema'], '本次必须与之前混淆矩阵一样使用EMA'
assert diag['train_dataloader']['collate_fn']['stop_epoch'] == 100
assert diag['train_dataloader']['dataset']['transforms']['policy']['epoch'] == 100
assert diag['train_dataloader']['total_batch_size'] == diag['val_dataloader']['total_batch_size'] == 32
checkpoint_sha256 = sha256(checkpoint_path)
assert checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256, '权重指纹不是此前提供的110轮Run2；停止，未创建输出'
data = diag['val_dataloader']['dataset']
annotation_path = Path(data['ann_file'])
annotation_sha256 = sha256(annotation_path)
assert annotation_sha256 == EXPECTED_ANNOTATION_SHA256, '标注指纹与Run2混淆矩阵不一致；停止，未创建输出'
dataset = json.loads(annotation_path.read_text(encoding='utf-8'))
assert sorted((c['id'], c['name']) for c in dataset['categories']) == [(0, 'hand-raising'), (1, 'read'), (2, 'write')]
assert dataset['images'], '验证集为空'
images = {image['id']: image for image in dataset['images']}
assert len(images) == len(dataset['images']), '验证集image_id重复'
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

# Safe checkpoint read: no arbitrary-pickle fallback. Strict-load both states
# against the disabled-experiment baseline; no downloading pretrained weights.
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
assert 0 <= checkpoint.get('last_epoch', -1) < 110, '不是110轮训练权重'
assert 'model' in checkpoint and 'module' in checkpoint.get('ema', {}), '缺少model或EMA权重'
model_config = YAMLConfig(str(config_path))
model_config.yaml_cfg['HGNetv2']['pretrained'] = False
model = model_config.model
for name, state in (('model', checkpoint['model']), ('ema', checkpoint['ema']['module'])):
    model.load_state_dict(state, strict=True)
    assert state['decoder.dec_score_head.2.weight'].shape[0] == 3, f'{name}类别数错误'
inputs = dict(checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=checkpoint_sha256,
              checkpoint_epoch=checkpoint['last_epoch'], annotation_file=str(annotation_path.resolve()),
              annotation_sha256=annotation_sha256, validation_dataset=data, images=len(images),
              GT=sum(class_counts.values()), GT_by_category=dict(class_counts), weight_source='ema',
              confidence_threshold=.5, iou_threshold=.5, neighbor_iou_threshold=.8)
output_path.mkdir(parents=True, exist_ok=False)
(output_path / 'export_inputs.json').write_text(json.dumps(inputs, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('检查通过：Run2权重/标注指纹、110/100、原模型/损失、EMA与完整验证图片。', flush=True)
print('图像：', inputs['images'], '；GT：', inputs['GT'], '；请确认GPU空闲。', flush=True)
PY

mkdir -- "$run2_diag_output/logs"
trap 'run2_diag_status=$?; printf "诊断未完成，退出码=%s；保留部分结果：%s\n" "$run2_diag_status" "$run2_diag_output" >&2; exit "$run2_diag_status"' ERR
printf '\n[1/2] 只验证Run2：全部300个query、原后处理选择、混淆矩阵、COCO评估。\n'
"$run2_diag_conda" run --no-capture-output -n "$run2_diag_env" \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py -c "$run2_diag_config" -r "$run2_diag_checkpoint" --test-only \
--output-dir "$run2_diag_output/evaluation" --use-amp --seed=0 \
2>&1 | tee "$run2_diag_output/logs/evaluation.log"

printf '\n[2/2] 离线整理：GT候选交叉分类、类别×尺度、PR、分数与IoU关系。\n'
"$run2_diag_conda" run --no-capture-output -n "$run2_diag_env" \
python tools/diagnostics/analyze_detection_diagnostics.py \
--input-dir "$run2_diag_output/evaluation" --output-dir "$run2_diag_output/analysis" \
2>&1 | tee "$run2_diag_output/logs/analysis.log"

"$run2_diag_conda" run --no-capture-output -n "$run2_diag_env" python - "$run2_diag_output" <<'PY'
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
    'evaluation/confusion_matrix/confusion_matrix_counts.png',
    'evaluation/confusion_matrix/confusion_matrix_row_normalized.png',
    'evaluation/validation_metrics.json', 'evaluation/per_class_metrics.json', 'evaluation/eval.pth',
    'analysis/summary.json', 'analysis/summary.txt', 'analysis/gt_records.jsonl',
    'analysis/prediction_records.jsonl', 'analysis/best_query_cross_classification.json',
    'analysis/best_query_cross_classification.jsonl', 'analysis/gt_best_iou_distribution.json',
    'analysis/gt_size_statistics.json', 'analysis/coco_pr_curves.json',
    'analysis/coco_per_class_and_size.json',
    'analysis/score_distributions_and_quality.json', 'analysis/coco_pr_iou0.5.png',
    'analysis/coco_pr_iou0.75.png', 'analysis/diagnostic_tp_fp_scores.png',
    'analysis/score_quality_relationship.png', 'analysis/gt_best_iou_distribution.png',
)
for filename in required:
    assert (root / filename).is_file() and (root / filename).stat().st_size > 0, f'未完整导出：{filename}'
query = json.loads((root / 'evaluation/query_diagnostics/metadata.json').read_text())
summary = json.loads((root / 'analysis/summary.json').read_text())
matrix = json.loads((root / 'evaluation/confusion_matrix/confusion_matrix.json').read_text())
assert query['complete'] and query['schema_version'] == 2, 'query导出未完整完成或使用了旧schema'
assert query['num_classes'] == 3 and query['postprocessor_top_k'] == 300, '类别或后处理Top-k不一致'
assert summary['weight_source'] == matrix['metadata']['weight_source'] == inputs['weight_source'] == 'ema'
for report in (query, summary, matrix['metadata']):
    assert Path(report['checkpoint']).resolve() == Path(inputs['checkpoint']), '步骤之间权重路径不同'
assert query['checkpoint_sha256'] == summary['checkpoint_sha256'] == inputs['checkpoint_sha256'] == sha256(inputs['checkpoint']), '权重指纹改变或不一致'
assert inputs['annotation_sha256'] == sha256(inputs['annotation_file']), '诊断期间标注发生了变化'
assert matrix['metadata']['validation_dataset'] == inputs['validation_dataset'], '验证集配置改变了'
assert Path(summary['annotation_file']).resolve() == Path(inputs['annotation_file'])
assert query['images'] == summary['images'] == inputs['images'], '图像计数不一致'
assert query['queries'] == summary['queries'] == 300 * inputs['images'], '未导出完整300-query结果'
assert len(list((root / 'evaluation/query_diagnostics/raw_queries').glob('*.pt'))) == inputs['images'], '原始query文件不完整'
assert summary['GT'] == matrix['GT_count'] == inputs['GT'], 'GT计数不一致'
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
    assert sum(counts[category]) == inputs['GT_by_category'][str(category)], '类别GT计数不一致'
assert matrix['predictions_after_threshold'] == sum(sum(row[:3]) for row in counts) == summary['selected_pairs_above_confidence']
sizes = json.loads((root / 'analysis/gt_size_statistics.json').read_text())
scale = json.loads((root / 'analysis/coco_per_class_and_size.json').read_text())
cross = json.loads((root / 'analysis/best_query_cross_classification.json').read_text())
class_names = {'0': 'hand-raising', '1': 'read', '2': 'write'}
areas = {'all', 'small', 'medium', 'large'}
assert scale['class_names'] == class_names and scale['units'].startswith('0~1'), '类别×尺度的类别映射或指标单位错误'
assert set(scale['area_ranges']) == areas and set(scale['metrics']) == set(class_names), '类别×尺度COCO指标不完整'
assert cross['confidence_threshold'] == cross['IoU_threshold'] == .5, '候选交叉统计阈值不一致'
assert cross['all']['GT_count'] == sizes['all']['GT_count'] == inputs['GT'], '候选/尺度统计GT总数不一致'
assert set(cross['by_class']) == set(class_names.values()) and set(cross['by_size_and_class']) == areas, '类别/尺度候选缺口统计不完整'
for category, name in class_names.items():
    assert set(scale['metrics'][category]) == areas, f'{name}缺少尺度COCO指标'
    assert cross['by_class'][name]['GT_count'] == inputs['GT_by_category'][category], f'{name}候选GT计数不一致'
    for area in areas:
        metrics = scale['metrics'][category][area]
        assert set(metrics) == {'AP', 'AP50', 'AP75', 'AR100'}, f'{name}/{area}缺少COCO指标'
        assert all(value is None or (type(value) in (int, float) and 0 <= value <= 1) for value in metrics.values()), f'{name}/{area}指标范围错误'
        group = cross['by_size_and_class'][area][name]
        assert group['GT_count'] == sizes[area]['class_counts'][name], f'{name}/{area}候选/尺度GT计数不一致'
        assert set(group['type_counts']) == set('ABCDEFG') and sum(group['type_counts'].values()) == group['GT_count'], f'{name}/{area}候选分组不完整'
        assert group['high_correct_selected_candidate_exists'] + group['no_high_correct_selected_candidate'] == group['GT_count'], f'{name}/{area}候选覆盖计数不一致'
with (root / 'analysis/gt_records.jsonl').open(encoding='utf-8') as handle:
    assert sum(1 for line in handle if line.strip()) == inputs['GT'], 'GT记录不完整'
report = dict(inputs, complete=True, experiment='110-epoch baseline Run2 full diagnostics',
              comparison_baseline='SCB-S / D-FINE-S / 110-epoch Baseline Run2',
              queries=query['queries'], matching=matrix['matching'],
              note='Validation + offline analysis only. No retraining, rescoring or annotation edits. Candidate evidence is not COCO TP/FN or causal layer identification.')
(root / 'workflow_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
text = '110轮基线 Run2 完整诊断\n' + (root / 'analysis/summary.txt').read_text(encoding='utf-8')
(root / 'run2_diagnostics_summary.txt').write_text(text, encoding='utf-8')
print('\nRun2诊断完整结束：', root, flush=True)
print('先查看：', root / 'run2_diagnostics_summary.txt', flush=True)
print('后续诊断统一与这份Run2比较；请上传整个结果目录，不只混淆矩阵。', flush=True)
PY
