# Existing DiscriminativeEvidenceL2 checkpoint: diagnostics ONLY

Use this after interrupted/resumed training completes. This is not the original
from-scratch launcher. It never executes detector training, tuning, optimizer
steps, or synthetic training tests. No new model/config/loss is introduced.

## Sync first (Mac)

```bash
cd /Users/wq/Desktop/D-FINE
bash tools/diagnostics/sync_scbs_AQS_DiscriminativeEvidenceL2_diagnostics.sh
```

Only the seven tools/doc files in the dedicated manifest are synced. Existing
server detector/config, shared analysis tools, annotations and weights remain
unchanged. The original completed experiment and its helpers must already be
present. Replacement tools are backed up; there is no --delete.

## Run second (server, inside tmux)

```bash
cd /home/a5/MyProject/wq_project/D-FINE
bash tools/diagnostics/export_scbs_AQS_DiscriminativeEvidenceL2.sh run1 export1
```

Checkpoint:
`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_AQS_DiscriminativeEvidenceL2_seed0_run1/best_stg2.pth`

New bundle directory INSIDE that experiment:
`DiscriminativeEvidenceL2_diagnostics_export1/`

An existing export (including a partial one) is never overwritten. Choose
`run1 export2` for another export. This does not change the training run number.
`--dry-run run1 export1` checks paths/options only, without GPU/file writes.

The GPU must be free and port7781 available; the script never kills other jobs.
It only needs validation images and annotations, not training images. If the
llama task still owns the GPU, the exporter stops before creating any result.

Overrides: `SCBS_DER_ROOT/CHECKPOINT/GPU/PORT/CONDA/ENV`. Use a checkpoint override
only for the SAME AQS+DiscriminativeEvidenceL2 trained experiment. To specify
renamed reference folders: `SCBS_DER_AQS_REFERENCE`, `SCBS_DER_RUN2_REFERENCE`.

## Outputs

- `export_inputs.json`: checkpoint/source/validation-annotation hashes, config,
  training-log best AP and duplicate-epoch information. The resumed run has
  duplicate epoch103; that is not an independent extra experiment.
- `aqs_parameters.json`, `DiscriminativeEvidenceL2_parameters.json`: read-only
  model/EMA parameter reports (norms are not observed attention utilization).
- ON: `diagnostics_eval/{evaluation,analysis}`.
- OFF: `DiscriminativeEvidenceL2_off_eval/{evaluation,analysis}`.
- In EACH pass: original COCO AP/class/size results, confusion JSON/PNG, all
  query records, candidate A/B/C/D gaps, score-IoU/PR and category-by-size tables.
- `DiscriminativeEvidenceL2_on_off.json`: identical EMA, thresholds and GT cohort.
- `checkpoint_evaluation_audit.json`: independently loaded ON AP versus the
  training log's best AP. For the uploaded run1 log, expected AP is55.233830.
  Mismatch is explicitly flagged, never silently replaced by the training AP.
- Available `compare_AQS_L2_R1.json`, `compare_BaselineRun2.json` inside ON
  diagnostics. Known reference directories are preferred; missing references
  are reported, not fabricated. Incompatible references fail instead of mixing.
- Terminal/log files in `logs/`, plus one downloadable ZIP named after the
  export folder. No training checkpoint is copied into that ZIP.

Both passes construct the SAME detector and strict-load the SAME trained EMA.
Do not set the model's use_discriminative_evidence_l2 flag false: that removes
its parameter schema. OFF bypasses only its runtime residual, keeping AQS ON.
Because the residual is before L3, final boxes as well as classes may change.
OFF remains jointly trained weights, NOT an independently trained AQS baseline.
ON/OFF cannot on its own prove which shared training layer caused lost AP.

The existing training helper checks EMA/hash/GT/threshold identities, and handles
older AQS exports missing the class-by-size table without fabricating zeros.
Default reference search is limited to AQS-R1 diagnostics_eval_v2/diagnostics_eval
and Run2 diagnostics_export1/diagnostics_eval. No broad recent-run selection.
