# AQS + three independent instance-evidence experiments

All code is edited in `/Users/wq/Desktop/D-FINE`. Nothing was automatically
uploaded or trained. These are **literature-inspired detection adaptations**,
not paper reproductions or proven thesis contributions.

## Controlled protocol

- Base: existing `dfine_s_scbs_hrw_110_aqs_layer2_r010.yml`.
- AQS: layer2, initial threshold=.50, residual=.10, fixed temperature=.10.
- 110 epochs, stop augmentation100, batch32, seed0, original pretrained HGNet-B0.
- Train each detector from scratch, not fine-tuning a completed AQS checkpoint.
  AQS weights keep learning normally; "fixed AQS" means fixed design/settings.
- Original VFL/bbox/GIoU/FGL/DDF and matcher are retained. No new loss.
- Each configuration adds **one** module, not all three simultaneously.
- P2 is returned additionally by the backbone. Original P3/P4/P5 computation
  and Encoder routing are unchanged. Decoder-projected P3 is the semantic input.
- New head runs AFTER final LQE and DN splitting, only on ordinary queries.
  It returns `[0, delta_read, delta_write]`, with independent read/write shifts.
- Predicted boxes are clipped/detached for sampling; no GT boxes or labels are
  consumed by the new branch. No hard confidence/AQS-selection gate.
- Direct bbox/corner/pre/DN/auxiliary outputs and AQS teacher logits are unchanged
  for the same model weights. Shared-feature training can still change all
  classes and localization; this is NOT a guarantee of unchanged trained boxes.
- Zero output initialization exactly reproduces AQS-alone initial outputs and
  preserves its initialization RNG. The last projection learns on step1;
  internal evidence branches begin learning after that projection becomes nonzero.
- Settings are persisted as an integer buffer, checked on strict loading, and
  immune to EMA averaging / half-precision conversion.

## The three modules

| Module | Mechanism | Difference from existing modules |
|---|---|---|
| PartEvidenceSelection | 7x7 cells plus 3x3 overlapping pooled regions; separate region class evidence and latent-part/irrelevant-region assignment; relevance-weighted evidence aggregation | Not just one query-attention pooled vector; classification and relevance are separate streams, with explicit irrelevant-region mass |
| TrilinearDetailSampling | Per-person normalized channel relation `(softmax(X) X^T)` and spatial attention; query-conditioned detail + averaged structure maps; axis inverse-CDF grids resample native P2/P3 | Not nine directly regressed offsets or uniform5x5 ROI pooling; coordinates are derived from attention density. It cannot recover already-lost pixel detail |
| LocalInteractionEvidence | Three latent region slots, directed non-self edges with appearance/difference/product/displacement; read/write select edges independently | Unlike PartInteractionL2, it acts on final logits, not L2 query; relation selection is class-conditioned, not simple average of all pairs |

Sources: PartNet (TMM2020, DOI10.1109/TMM.2019.2939747), TASN (CVPR2019,
https://openaccess.thecvf.com/content_CVPR_2019/html/Zheng_Looking_for_the_Devil_in_the_Details_Learning_Trilinear_Attention_CVPR_2019_paper.html),
iHOI (TMM2020, DOI10.1109/TMM.2019.2943753).

Deliberately NOT implemented: PartNet-Full multi-CNN crop ensemble / DPP448
proposals / BCE replacement; TASN image warping / part-master distillation;
iHOI explicit object detection / pose / gaze / human-object GT triplets.
Latent regions are not proven hands, books or intentions. Similar components
already exist in previous trials; this experiment alone does not establish novelty.

## First sync, then run

On the Mac:

```bash
cd /Users/wq/Desktop/D-FINE
bash tools/diagnostics/sync_scbs_AQS_instance_evidence.sh
```

On the server:

```bash
cd /home/a5/MyProject/wq_project/D-FINE
bash tools/diagnostics/run_scbs_AQS_instance_evidence_110.sh all run1
```

The three runs are sequential on GPU0 with port7781. Conda environment `wq`
and the previous explicit `--master_port` torchrun style are retained. Console
output is streamed and saved in each run's `console.log`; no hidden capture.
The launcher refuses existing output directories and occupied GPU/port. It never
kills another project, deletes results or silently changes batch size.

One group only / restart a new experiment number:

```bash
bash tools/diagnostics/run_scbs_AQS_instance_evidence_110.sh PartEvidenceSelection run1
bash tools/diagnostics/run_scbs_AQS_instance_evidence_110.sh TrilinearDetailSampling run2
```

Use `--dry-run all run1` to list plans without data access or output creation.
This is not resume support. An interrupted run is retained; do not overwrite it.

## Outputs and comparisons

Each output directory is named:

`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_AQS_<Module>_seed0_run1`

The launcher saves:

- `experiment_inputs.json`: resolved configuration and source/annotation hashes.
- Original training weights/logs and AQS model/EMA learned scalars.
- `best_stg2_<Module>_parameters.json`: module settings and model/EMA parameter
  norms; LocalInteractionEvidence also reports learned latent spatial centers.
  These values are not observed region relevance or proof of the mechanism.
- `diagnostics_eval/evaluation`: original COCO evaluation, confusion matrix,
  raw query predictions and best-checkpoint metadata.
- `diagnostics_eval/analysis`: GT records, class-by-size AP, PR, score/IoU,
  candidate gaps and A/B/C/D cross-classification from existing analysis.
- If complete reference diagnostics exist: `compare_AQS_L2_R1.json` and
  `compare_BaselineRun2.json`. Comparisons verify matching GT identities,
  original boxes/areas/classes, EMA source and diagnostic thresholds.

Default references are the existing AQS-R1 and BaselineRun2 output directories'
`diagnostics_eval` children. If they are stored elsewhere, provide explicit paths:

```bash
SCBS_EVIDENCE_AQS_REFERENCE="/absolute/path/to/AQS/diagnostics_eval" \
SCBS_EVIDENCE_RUN2_REFERENCE="/absolute/path/to/Run2/diagnostics_eval" \
bash tools/diagnostics/run_scbs_AQS_instance_evidence_110.sh all run1
```

Absent default references do not prevent training/export; the script reports the
missing comparison. Explicitly supplied incomplete references are rejected early.
An existing complete export can be compared later without retraining:

```bash
conda run --no-capture-output -n wq python tools/diagnostics/scbs_instance_evidence_experiment.py \
compare "/absolute/module/output" "/absolute/AQS/diagnostics_eval" AQS_L2_R1
```

First compare **new module + AQS vs AQS alone**; Run2 is the original baseline
reference, not a substitute for measuring the second module's incremental value.
Focus on COCO AP/read/write AP and both confusion directions, correct-high-query
gaps, PR/precision-recall; APm is secondary. Do not select a module solely on the
0.5-threshold confusion counts. Repeat positive results before claiming stability.

## Verification and resource limits

Run `python tools/diagnostics/test_instance_evidence.py`: CPU/BF16 gradients,
coordinate detachment, instance isolation, dual streams, inverse-CDF gradients,
directed edges, controlled configs/RNG, original losses/DN/aux/teacher, EMA strict
round trips, 640 eval/deploy and output preservation. CUDA tests run automatically
on the server, including isolated batch32 x300-query heads. Synthetic tests do
not validate accuracy, and isolated-head memory is not full-training memory.

Each branch uses query chunks32 and non-reentrant activation checkpointing.
The local Mac cannot verify full 4090 batch32 mixed-scale training memory or speed.
If the full run OOMs, retain the error and review memory; do not silently alter
only one group's batch size, resolution or training schedule.

Useful follow-up ablations only after a positive signal: module alone vs AQS,
uniform/no-relevance region weighting, no channel-relation vs uniform sampling,
no geometry or no class-specific relation selection, and capacity-matched
query-only/local-detail controls. Visualize learned regions before interpreting
them as behavior-specific evidence.
