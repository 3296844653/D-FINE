# Short20 weight comparison (not a full-training result)

Both runs use the same original SCB-S baseline `best_stg2.pth` (same bytes),
seed, hardware, batch32, full train/val splits and 640 evaluation. All detector
parameters remain trainable as in the inherited baseline; this is not a frozen
head probe. Only CE weight differs between the two short configurations.

- Control: `configs/dfine/dfine_s_scb3s_3cls_bs32_rw_pairwise_ce_short20_w01.yml`
- Experiment: `configs/dfine/dfine_s_scb3s_3cls_bs32_rw_pairwise_ce_short20_w005.yml`

Protocol shared by both: 20 fresh epochs, main LR 2e-5, backbone LR 1e-5,
100-step warmup, LR decay at milestone15. The strong augmentation policy is
disabled from epoch0; horizontal flip and the remaining transforms are retained.
Multiscale collate is disabled (fixed640). `stop_epoch=20` prevents the solver
from loading a nonexistent first-stage checkpoint: epochs0..19 never reach it.
EMA settings stay as inherited; the old checkpoint's EMA weights are loaded into
the model by `-t`, then a new EMA and optimizer are initialized for this run.

## How to run

In the existing Colab notebook, keep mount/sync/install and backup cells. In its
training cell, replace the run name and config and ADD `-t` with the original
baseline checkpoint path. Use `--use-amp --seed=0` for both runs. No `-r`, no
`--test-only` during these short training runs. Do not run the unmount/runtime
shutdown cell until results have been copied to Drive.

Original baseline checkpoint (Colab):

    /content/drive/MyDrive/D-FINE_outputs/dfine_s_scb3s_3cls_bs32_run1/best_stg2.pth

Suggested distinct output names:

    dfine_s_scb3s_3cls_bs32_rw_short20_w01_seed0
    dfine_s_scb3s_3cls_bs32_rw_short20_w005_seed0

The two runs must NOT chain weights. Both start from baseline, not from the
completed pairwise-CE run or the short control. Use new output folders; do not
mix with existing 132-epoch logs or weights.

Each short run's best checkpoint is `best_stg1.pth`. To export its matrix, invoke
its short config with `-r <short-run best_stg1.pth> --test-only`, a separate
output directory and `-u export_confusion_matrix=true`. The training solver does
not automatically export the confusion matrix just because the flag is enabled.

Compare the two short runs at identical epochs and using their best overall-AP
checkpoints. Report AP/read AP/write AP, both read/write confusion directions,
read/write Recall, unmatched GT and predictions. Lowering the weight is a
hypothesis about a tradeoff, not a guaranteed improvement. Do not select on just
one fluctuation, compare the short results directly with the old 132-epoch result,
or regard a negative short screen as proof the full-training variant cannot work.
Repeated manual tuning on validation risks overfitting it: this is one predeclared
0.05 candidate, not an open-ended validation search. Full training/repeats remain
necessary before claiming stable benefits.
