# SCB-S run1 complete detection diagnostics

These files are diagnostic-only. Do not retrain or change the baseline/model,
annotations, scores, class labels, postprocessor top-k or matching thresholds.

## Files

- `configs/dfine/dfine_s_scbs_hrw_132.yml`: the preserved exact run1 baseline and server data paths. The current unsuffixed training config now uses 110 epochs.
- `configs/dfine/dfine_s_scbs_hrw_confusion.yml`: inherits it and enables both exports.
- `src/solver/query_diagnostics.py`: exports all queries; schema 2 also preserves
  the original-device sigmoid scores and postprocessor boxes for exact analysis.
- `tools/diagnostics/analyze_detection_diagnostics.py`: CPU-only offline analysis.
- `tools/diagnostics/test_detection_diagnostics.py`: synthetic CPU regression tests.

## Sync from the Mac Desktop checkout

This copies the historical baseline snapshot and diagnostic files, never data,
weights or results. It does not replace the current training configuration.

```bash
cd /Users/wq/Desktop/D-FINE
rsync -av --relative \
  ./configs/dfine/dfine_s_scbs_hrw_132.yml \
  ./configs/dfine/dfine_s_scbs_hrw_confusion.yml \
  ./src/solver/query_diagnostics.py \
  ./tools/diagnostics/analyze_detection_diagnostics.py \
  ./tools/diagnostics/test_detection_diagnostics.py \
  ./tools/diagnostics/SCBS_RUN1_DIAGNOSTICS.md \
  a5@100.76.151.67:/home/a5/MyProject/wq_project/D-FINE/
```

The server must also have the already implemented `det_engine.py`,
`det_solver.py`, `confusion_export.py` and BaseConfig query export flags.
They are unchanged in this task.

## 1. Export using run1's own best checkpoint (server)

Use a NEW output directory. The query writer intentionally refuses an existing
`query_diagnostics` directory, rather than mix checkpoints or append duplicates.
If this directory was previously used/interrupted, choose a new `_v2` directory;
do not remove previous results or restart baseline training.

```bash
cd /home/a5/MyProject/wq_project/D-FINE
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n wq \
torchrun --master_port=7781 --nproc_per_node=1 train.py \
  -c configs/dfine/dfine_s_scbs_hrw_confusion.yml \
  -r "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_seed0_run1/best_stg2.pth" \
  --test-only \
  --output-dir "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_seed0_run1/diagnostics_eval" \
  --use-amp --seed=0
```

Produces `confusion_matrix/`, `query_diagnostics/` (raw tensors and reference
tables), `eval.pth`, `validation_metrics.json`, and per-class evaluation metrics.
Metadata preserves checkpoint SHA256 and actual inference configuration.

## 2. Analyze the same export (server; no GPU required)

```bash
cd /home/a5/MyProject/wq_project/D-FINE
MPLBACKEND=Agg conda run --no-capture-output -n wq \
python tools/diagnostics/analyze_detection_diagnostics.py \
  --input-dir "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_seed0_run1/diagnostics_eval"
```

The validation annotation path is read from export metadata. When copying
diagnostics to another computer, pass `--annotation-file` pointing to the SAME
validation JSON at its new location. Raw GT/category/image counts and geometry
are checked against that JSON. No training images are required for offline analysis.

## Output in `diagnostics_eval/analysis/`

| Question | Output | Boundary of the interpretation |
|---|---|---|
| read/write confusion | `summary.json`/`.txt`, original confusion matrix | Counts use original Validator class-blind IoU-first one-to-one matching, conf>=0.5, IoU>=0.5. Wrong/ignored labels are not automatically corrected. |
| Coverage/localization | `gt_best_iou_distribution.json`/`.png`, `gt_records.jsonl` | All-query GT-best IoU is GT-assisted ORACLE coverage, not actual detector Recall. |
| Classification vs low score/selection | `gt_records.jsonl`, candidate status counts in `summary.json` | Scores of best-IoU queries and actual high-score diagnostic matches are kept separate. Overlap evidence does not prove a causal network-layer bottleneck. |
| Sizes | `gt_size_statistics.json`, coverage by size | Original annotation area and saved COCO inclusive area ranges; boundary GT can count in adjacent ranges. Counts/images/class mix shown. One run does not determine stability. |
| PR/ranking | `coco_pr_curves.json`, `coco_pr_iou0.5.png`, `coco_pr_iou0.75.png` | Read directly from saved COCO eval.pth, 101-point interpolated precision, maxDets=100. Not a new custom AP calculation. |
| Confidence separation | `score_distributions_and_quality.json`, `diagnostic_tp_fp_scores.png`, `prediction_records.jsonl` | JSON separates all top-k pairs from confidence>=0.5 predictions; the PNG displays only the high-score subset so low-score negatives do not dominate it. Separate class-aware score-first one-to-one IoU>=0.5 diagnostic; no COCO maxDets/ignore/crowd handling, not substituted for original matrix/AP. |
| Score/quality relationship | same JSON + `score_quality_relationship.png` | Uses geometric maximum same-class IoU, plus one-to-one TP status. A duplicate can have high IoU yet be FP. Correlation is not LQE causal attribution. |

All scores remain final sigmoid scores including LQE. The standalone LQE
component and pre-LQE class logits are NOT separately exported; network stages
cannot be diagnosed from final score correlations alone. No false claim that a
single run establishes stability, or that an unmatched box is true background.

The analyzer rejects incomplete exports, mismatched annotations/categories,
missing COCO results, inconsistent prediction counts and nonempty analysis
directories. For reruns, pass a new `--output-dir`, e.g. `analysis_v2`.
Only analyze trusted .pt/.pth files from this project (PyTorch pickle loading).

## Local regression checks

```bash
python tools/diagnostics/test_detection_diagnostics.py
```

Tests cover real Writer/Postprocessor/Validator/COCO integration, exact score and
box export (including CPU BF16 rounding), wrong-class and duplicate candidates,
low confidence, empty-GT images, oracle-vs-selected distinctions, clipping,
annotation mismatch, incomplete export and refusing output overwrite.
CPU tests are not a server checkpoint/data export or a CUDA performance test.
