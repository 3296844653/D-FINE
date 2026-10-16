# AQS-L2 R1 + SubpixelContextFusion (SCF)

## Why this experiment

The verified AQS-L2 R1 best EMA has AP=55.87305, read AP=58.43816,
write AP=49.74225. At confidence/IoU=0.5, write→read=240/684 (35.09%),
read→write=156/2433 (6.41%). Writing correct-high-score query gap=207/684.
These are validation diagnostics, not train accuracy or proof of anatomical causes.
Medium writing has only 57 GT; small has 3 GT across all classes. Old AQS export
lacks the category×scale AP table, so Run2's medium writing AP is NOT AQS AP.

The preceding DiscriminativeEvidenceL2 run has AP=55.23383: lower than AQS.
Improved write recall at one threshold is not proof of improved AP/confusion.
Its resumed run is not a fully uninterrupted controlled experiment. SCF is a new
feature-side hypothesis, not a claim that its previous failure's cause is known.

## Location and mechanism

HGNet P2 (stride4) + FPN P3/P4 → SCF → original PAN → standard P3/P4/P5
→ original Decoder (including unchanged L2 AQS) → original class/bbox heads.

1. Project native P2 to 32 channels. Preserve the TL/TR/BL/BR subcells in each
   stride8 cell via pixel-unshuffle, not bilinear downsampling/ROI averaging.
2. P3 appearance + bilinearly aligned P4 semantics select the four fine cells.
   Phase position embeddings preserve relative layout. Padded cells are masked;
   a zero-valued null candidate allows rejection, without confidence thresholds.
3. Selected fine evidence conditions a 3×3 dilation2 P3 neighborhood read with
   relative-offset bias. Read differences to the center rather than blindly adding
   neighboring appearance; an independent null candidate can reject context.
4. Joint channel gates fuse selected detail, signed subcell contrast, relative
   context and their interaction. Zero-initialized output + nonzero bounded scale
   produces an exactly baseline-equivalent initial residual. PAN propagates it.

This is NOT supervised hand/pen/book detection, class-conditioned read/write
prototype matching, an AQS replacement, a new TopK selector, or a loss change.
No GT, predicted boxes or scores are module inputs. No third-party CUDA extension.
It can change ALL classes/scales, TopK identities, final boxes and class scores.
Checkpoint recomputes only the new GN branch; original BN/FPN/PAN runs once.
Added trainable parameters: 51,728 (~0.052M). Compare deployed parameter totals
using the project's existing profiler; raw training totals retain BN parameters.

Difference from earlier modules: local ROI/query/head methods depend on predicted
boxes or query axes; CrossGate has same-location coarse gating; SADF aligns P4/P5;
SDRF changes P3 relation weights with differential attention. SCF adds native P2
phase-preserving selection linked to P3 relative-context reading, before PAN.
Overlapping components are ordinary attention/gating/residuals: originality and
effectiveness are NOT established by implementation or one positive run.

## Recipe, acceptance and output

110 epochs, stop augmentation100, batch32, seed0; original optimizer, losses and
EMA; AQS threshold init0.50, raw residual init0.10, fixed T0.10, L2 only.
Same recipe from scratch, original HGNet pretraining, not AQS checkpoint tuning;
AQS remains trainable. Config contains only this experiment's enabled settings.

Primary comparator: AQS alone, not merely original baseline. Require AP/write AP
improvement plus BOTH confusion directions and candidate-gap checks at identical
thresholds; PR/score/IoU/category×scale evidence accompanies 0.5 diagnostics. A
write→read decrease with more read→write/FP or AP loss is not accepted as success.
APm is secondary evidence, not an asserted cause of read/write confusion. Repeat
promising runs and later A/B/AB controls before treating two modules as innovations.

Launcher refuses existing outputs and occupied GPU/port, never stops llama/other
projects. Runs tests BEFORE creating the experiment directory; trains with visible
logs, then evaluates SAME best EMA ON/OFF (AQS stays ON), exports model/EMA SCF
scales/null biases/weight norms, AQS parameters, diagnosis and ZIP. OFF is not an
independently trained AQS baseline. Integer settings enforce strict config matches.
ON/OFF report also checks checkpoint SHA and flags whether independent ON AP
matches the training-log best; stale stage2 `last_epoch` is not trusted as best epoch.
Server tests include one synthetic CUDA FP16 batch32@640 training iteration. This
is not a memory guarantee for larger multiscale inputs or real crowded batches;
if it fails, training does not start and no experiment directory is created.

Reference overrides: `SCBS_SCF_AQS_REFERENCE`, `SCBS_SCF_RUN2_REFERENCE`.
Absent references are reported, never fabricated. Optional root/GPU/port/conda
overrides: `SCBS_SCF_ROOT`, `SCBS_SCF_GPU`, `SCBS_SCF_PORT`, `SCBS_SCF_CONDA`.
If training/export is interrupted, preserve outputs and diagnose separately;
do not rerun the from-scratch launcher into an existing run directory.

## Synchronize first (Mac)

```bash
cd /Users/wq/Desktop/D-FINE
bash tools/diagnostics/sync_scbs_AQS_SubpixelContextFusion.sh
```

## Then run (server, GPU free)

```bash
cd /home/a5/MyProject/wq_project/D-FINE
tmux new-session -s scbs_SCF \
  "cd /home/a5/MyProject/wq_project/D-FINE && bash tools/diagnostics/run_scbs_AQS_SubpixelContextFusion_110.sh run1; exec bash"
```

Tmux: Control+B, release, then D; closing the Mac/SSH client does not stop this
server session. The shell remains open even if a preflight check fails, so its
error stays visible. If the named session already exists, use another name or
inspect it; do not interrupt its current task. No training was launched by
the assistant. CUDA FP16 test must run on server; local CPU/BF16 is not a GPU pass.

Output: `/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_AQS_SubpixelContextFusion_seed0_run1`.
ZIP: `SubpixelContextFusion_diagnostics.zip` inside that directory, NOT a trained
weight bundle. Checkpoint parameter norms/biases are not observed attention maps.
