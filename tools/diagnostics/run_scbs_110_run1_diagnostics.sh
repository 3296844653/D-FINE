#!/usr/bin/env bash
# Read the trained 110-round run1 checkpoint; NEVER train, resume or overwrite.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
SCBS_ENV="${SCBS_ENV:-wq}"
SCBS_GPU="${SCBS_GPU:-0}"
CHECKPOINT="/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_seed0_run1/best_stg2.pth"
CONFIG="configs/dfine/dfine_s_scbs_hrw_110_diagnostics.yml"
COVERAGE_CONFIG="configs/diagnostics/scbs_encoder_query_coverage_110_run1.yml"
OUTPUT="${1:-/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_seed0_run1/diagnostics_recheck}"
if [[ "$#" -gt 1 ]]; then
  printf '用法：bash tools/diagnostics/run_scbs_110_run1_diagnostics.sh [新输出目录]\n' >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  printf '输出目录已存在，保留旧结果：%s\n请传入一个新的输出目录。\n' "$OUTPUT" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$SCBS_GPU" PYTHONUNBUFFERED=1 MPLBACKEND=Agg

# Fail before creating result directories if source/config/data/weight is missing.
conda run --no-capture-output -n "$SCBS_ENV" python - "$CONFIG" "$COVERAGE_CONFIG" "$CHECKPOINT" <<'PY'
import json
import sys
from pathlib import Path
from src.core.yaml_utils import load_config
from tools.diagnostics.gt_box_classifier import load_config as load_probe_config
from tools.diagnostics.scbs_encoder_query_coverage import validate_config

for filename in ('train.py', 'tools/diagnostics/analyze_detection_diagnostics.py',
                 'tools/diagnostics/scbs_encoder_query_coverage.py'):
    assert Path(filename).is_file(), f'缺少脚本：{filename}'
diag = load_config(sys.argv[1], cfg={})
probe = load_probe_config(sys.argv[2])
validate_config(probe)
base = load_config(probe['detector_config'], cfg={})
assert diag['epochs'] == base['epochs'] == 110, '诊断必须使用110轮独立基线'
assert diag['train_dataloader']['dataset']['transforms']['policy']['epoch'] == 100
assert diag['train_dataloader']['collate_fn']['stop_epoch'] == 100
assert diag['export_query_diagnostics'] and diag['export_confusion_matrix']
assert diag['num_classes'] == 3 and not diag['remap_mscoco_category']
assert probe['checkpoint'] == sys.argv[3] and Path(sys.argv[3]).is_file(), '缺少110轮Run1权重'
for split in ('train', 'val'):
    data = diag[f'{split}_dataloader']['dataset']
    assert data['ann_file'] == probe['data'][split]['annotation']
    assert data['img_folder'] == probe['data'][split]['image_dir']
    assert Path(data['img_folder']).is_dir(), f'{split}图片目录不存在'
    dataset = json.loads(Path(data['ann_file']).read_text(encoding='utf-8'))
    assert sorted((c['id'], c['name']) for c in dataset['categories']) == [(0, 'hand-raising'), (1, 'read'), (2, 'write')]
    for image in dataset['images']:
        assert (Path(data['img_folder']) / image['file_name']).is_file(), f"缺少图片：{image['file_name']}"
print('检查通过：110/100、SCB-S类别、数据及Run1权重一致。确认GPU空闲后运行。', flush=True)
PY

mkdir -p -- "$OUTPUT/logs"
printf '\n[1/3] 只评估110轮Run1，导出混淆矩阵和全部query，不训练。\n'
conda run --no-capture-output -n "$SCBS_ENV" \
  torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
  -c "$CONFIG" -r "$CHECKPOINT" --test-only \
  --output-dir "$OUTPUT/evaluation" --use-amp --seed=0 \
  2>&1 | tee "$OUTPUT/logs/evaluation.log"

printf '\n[2/3] 自动统计类别×尺度、PR/分数和best-IoU冗余/缺口。\n'
conda run --no-capture-output -n "$SCBS_ENV" \
  python tools/diagnostics/analyze_detection_diagnostics.py \
  --input-dir "$OUTPUT/evaluation" --output-dir "$OUTPUT/analysis" \
  2>&1 | tee "$OUTPUT/logs/analysis.log"

printf '\n[3/3] 冻结同一权重，统计Encoder Top300与最终Decoder覆盖。\n'
conda run --no-capture-output -n "$SCBS_ENV" \
  python tools/diagnostics/scbs_encoder_query_coverage.py \
  --config "$COVERAGE_CONFIG" --output-dir "$OUTPUT/encoder_query_coverage" \
  2>&1 | tee "$OUTPUT/logs/encoder_query_coverage.log"

conda run --no-capture-output -n "$SCBS_ENV" python - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
analysis = json.loads((root / 'analysis/summary.json').read_text(encoding='utf-8'))
coverage = json.loads((root / 'encoder_query_coverage/summary.json').read_text(encoding='utf-8'))
export = json.loads((root / 'evaluation/query_diagnostics/metadata.json').read_text(encoding='utf-8'))
assert export['complete'] and coverage['complete'], '诊断未完整结束'
assert analysis['checkpoint_sha256'] and analysis['checkpoint_sha256'] == coverage['checkpoint_sha256'], '两个步骤不是同一权重'
assert analysis['weight_source'] == coverage['weight_source'], '两个步骤没有使用同一EMA/model权重'
assert analysis['GT'] == coverage['splits']['val']['coverage']['all']['gt_count'], 'GT数量不一致'
report = ((root / 'analysis/summary.txt').read_text(encoding='utf-8') + '\n'
          + (root / 'encoder_query_coverage/summary.txt').read_text(encoding='utf-8'))
(root / '110_run1_recheck_summary.txt').write_text(report, encoding='utf-8')
manifest = dict(complete=True, checkpoint=analysis['checkpoint'], checkpoint_sha256=analysis['checkpoint_sha256'],
                weight_source=analysis['weight_source'], GT=analysis['GT'],
                note='Only validation and offline analysis. Final-query export uses AMP/batches; coverage uses FP32/single-image. Not detector retraining or proof of a causal bottleneck.')
(root / 'workflow_summary.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('\n诊断成功完成；先查看：', root / '110_run1_recheck_summary.txt', flush=True)
PY
