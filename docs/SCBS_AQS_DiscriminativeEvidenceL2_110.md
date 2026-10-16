# AQS + DiscriminativeEvidenceL2

Research candidate, not established novelty or a guaranteed improvement.

## Evidence and aim

Use AQS-L2 R1 as the working baseline (AP55.8731, write49.7422), not the
original detector alone. Its write->read confusion is 240/684, read->write
156/2433, and 207 writing GT lack a correct high-confidence candidate at
confidence/IoU .5. All 684 have geometric candidates >=.5; that does not mean
localization/ranking are perfect. 2570 unmatched predictions are not all proven
background. Medium-scale performance is a secondary check, not a proven cause.

PES ON/OFF was 54.8023 vs54.7811, both below AQS. This does NOT establish that
the final head alone caused the joint training decrease, or that moving a head
to L2 fixes it. This new design tests a separate content-refinement hypothesis.

## Design

Order: L2 -> unchanged AQS-L2 -> DiscriminativeEvidenceL2 -> L3 -> original heads.

- Native backbone P2 and projected encoder P3, sampled in detached/clipped
  predicted L2 boxes. Query/box-size-conditioned fine/semantic channel mixing.
- Local P2 contrast and relative position alongside semantic appearance.
- Two hypotheses anchored by the original L2 read/write classifier weights
  (detached, projected) independently attend to ROI cells and
  an explicit null key. Null value is zero; real-cell mass is not renormalized.
- Separate evidence, their signed difference/product and null availability
  form a full-width query residual. Zero output initialization; fixed .1 scale.
- Real-cell availability also gates the full residual (including output bias):
  when both hypotheses defer to null, the update is suppressed. This is not a
  supervised validity score or a guarantee of reduced false positives.
- Soft ambiguity emphasis lies in [.5,1], never a high-confidence/class filter.
  Wrong high-score and low-score candidates can both receive evidence.
- Ordinary queries only, no GT ROIs, pose/gaze/object/part labels, new queries,
  class score boosts, extra loss, new matcher or train-time-only branch.

This is not just the old PES moved to L2 or a BoxLocal parameter sweep. It also
overlaps known class-conditioned local attention ideas; thesis originality
requires literature comparison and component/position ablations, not naming.

L1/L2 boxes are computed before the new residual. L3 classes AND boxes can change;
joint training changes shared features and possibly every output. DN queries
are not sampled/modified here, but original L3 self-attention lets DN read the
ordinary queries, so downstream DN outputs need NOT stay identical. No new
DN-to-ordinary information route is introduced. AQS's separate DN handling is
unchanged. Both AQS and the new branch remain trainable.

## Controlled experiment

110 epochs, stop augmentation100, batch32, seed0, standard pretrained HGNet-B0,
original losses/optimizer/schedule. New config includes only active new settings;
other experiments are off by default. AQS settings remain .50/.10/.10 at L2.
The new config overrides the inherited LOCAL dataset paths with the supplied
server SCB-S absolute paths; it does not edit any baseline config or annotations.

Added parameters: 101382 (not a measured FLOPs/latency gain).
Chunk32 and non-reentrant checkpointing bound ROI activations. CPU/BF16 tests
are synthetic; CUDA/FP16 and B32x300 branch tests run on server before training.
The isolated branch test does not guarantee full 4090 B32 memory headroom.

## Commands (sync first, run second)

Mac:

```bash
cd /Users/wq/Desktop/D-FINE
bash tools/diagnostics/sync_scbs_AQS_DiscriminativeEvidenceL2.sh
```

Server (inside tmux if the local computer will disconnect):

```bash
cd /home/a5/MyProject/wq_project/D-FINE
bash tools/diagnostics/run_scbs_AQS_DiscriminativeEvidenceL2_110.sh run1
```

Launcher refuses old outputs, occupied GPU/port and wrong/incomplete datasets;
does not kill any project or resume/overwrite checkpoints. Logs are displayed
and saved. Source replacements are backed up by scoped rsync, no --delete.

Output root:
`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_AQS_DiscriminativeEvidenceL2_seed0_run1`

Environment overrides: `SCBS_DER_ROOT/GPU/PORT/CONDA/ENV/SSH_TARGET`;
references: `SCBS_DER_AQS_REFERENCE`, `SCBS_DER_RUN2_REFERENCE` (directories
containing analysis/ and evaluation/). Reference comparison checks EMA/GT and
configuration identities. Missing old category-by-scale fields stay null,
never become fabricated zeros. Specify the actual reference locations if your
server folders were renamed. Missing references do not invalidate new exports.

## Automatically exported results

- Source/config/annotation hashes in experiment_inputs.json.
- Best model/EMA AQS scalars and new-module parameter norms/null bias.
  Norms are NOT measured attention distributions or evidence usage.
- ON: diagnostics_eval/{evaluation,analysis}: COCO AP, per-class x scale,
  class-blind confusion, all query records, PR/score-IoU and A/B/C/D GT diagnosis.
- OFF: DiscriminativeEvidenceL2_off_eval/{evaluation,analysis} using EXACTLY the
  same strict-loaded best EMA, AQS still ON. The runtime residual is disabled.
- DiscriminativeEvidenceL2_on_off.json and available AQS/Run2 comparisons.

OFF retains jointly trained weights and changes the L3 input, therefore may
change final boxes. It is neither the original AQS training nor frozen-box
offline re-scoring; ON/OFF alone cannot establish which training layer failed.

Judge incremental gains against AQS alone FIRST: AP, write/read AP, both confusion
directions, correct-high candidate gaps (B/D), false positives, PR and medium
class AP where both sides exist. Reduced write->read with increased read->write
or false positives is not a solved confusion problem. Single-run changes need
replication before claiming stable improvements or a thesis innovation.
