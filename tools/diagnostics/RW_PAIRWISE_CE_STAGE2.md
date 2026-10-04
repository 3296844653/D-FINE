# Shared-prefix stage-2 screening

This replaces, but does not delete, the unsuccessful fresh-optimizer short20
screen. No detector/solver/loss/data/metric code is changed by this experiment.

## Required source

Use the COMPLETE CE=0.1 experiment's `best_stg1.pth`, NOT baseline best_stg2,
NOT either short20 run, NOT completed CE best_stg2. Its `last_epoch` must be119.
From the provided full log, CE=0.1 first-stage best AP was approximately56.08
at epoch119; expect the pre-training resume evaluation to reproduce that value,
not baseline55.74. Stop and check source/data if it differs materially.

Colab source:

    /content/drive/MyDrive/D-FINE_outputs/dfine_s_scb3s_3cls_bs32_rw_pairwise_ce_run1/best_stg1.pth

Both runs resume the same checkpoint's model, EMA, optimizer, LR scheduler,
warmup and scaler. Retain original epochs132 and stop_epoch120. Do not reset
last_epoch, LR, optimizer, EMA, or move the augmentation boundary to0.

The original solver loads output/best_stg1 at the transition AND may reload it
after a validation regression/EMA adjustment. The launcher checks source state,
copies it byte-for-byte into a NEW output folder, records SHA256 and source states
in stage2_source.json, and invokes the unmodified train.py with -r. It refuses
old output folders, so do not overwrite earlier results or chain the two runs.

Profiles: w01=0.1; w005=0.05. Expanded configs differ only in CE weight.

## Colab training cell

After the usual GPU/mount/Git/sync/dependencies/data setup, run:

    PROFILE = "w01"  # run w005 separately afterward, same source
    RUN_NAME = f"dfine_s_scb3s_3cls_bs32_rw_stage2_{PROFILE}_seed0"

    CUDA_VISIBLE_DEVICES=0 python tools/diagnostics/run_pairwise_ce_stage2.py \
      --profile <PROFILE> \
      --checkpoint <original CE=0.1 best_stg1.pth> \
      --output-dir output/<RUN_NAME> --use-amp --seed=0 --master-port=7777

Keep the existing rsync-to-Drive backup after the training command. Only unmount
and disconnect after backup succeeds. Preparation errors must stop training.
Output names differ from full132 and short20 runs. If a name already exists, pick
a NEW suffix instead of deleting it. `--prepare-only` validates/copies without
launching; omit it for actual training. Do not launch prepare twice for one name.

## Interpretation

This compares weights only in epochs120..131 after a CE=0.1 prefix. It cannot
establish how CE=0.05 throughout132epochs would perform. Both forks reset RNG
from seed0 because the old loader does not restore RNG states; w01 is a necessary
paired control, not guaranteed bitwise reproduction of the old full run.

Keep the source immutable. Confirm stage2_source.json has the same source_sha256
in both runs. Logs should retain original LR states and start at Epoch:[120/132].
Use the initial resume AP as a sanity check, not a stage2 candidate result.

Compare both directions of read/write confusion, per-class AP/P/R, unmatched GT
and predictions, and global AP. Do not claim stable improvements from one pair.
If no best_stg2 file appears, consult log.txt: the original global-best saving
rules may not save any new best checkpoint when the source is never exceeded.
The launcher preserves those original rules and does not manufacture a best.

CPU preparation tests:

    python tools/diagnostics/test_pairwise_ce_stage2.py

These validate configuration differences, copied-state integrity, invalid-source
rejection and overwrite protection, not the actual Colab checkpoint or final AP.
