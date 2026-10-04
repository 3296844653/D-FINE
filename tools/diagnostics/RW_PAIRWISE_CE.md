# SCB-S read/write conditional classification experiment

Use the original SCB-S annotations, not the earlier cleaned dataset. Category IDs
are `0=hand-raising`, `1=read`, `2=write`.

## Hypothesis, not a diagnosis of cause

The observed matrix has 246 write-to-read and 154 read-to-write errors. This
identifies a target error type, but does not establish that the loss caused it.
This experiment tests a relative-class supervision change, not a new feature
extractor, a fix for occlusion, or a claimed novel research module.

For the final Hungarian-matched queries with GT in {read, write}, define

    p_pair(y|q) = exp(z_y) / (exp(z_read) + exp(z_write))
    L_pair = sum(-log p_pair(y|q)) / N_GT
    L_total = L_D-FINE + 0.1 * L_pair

`N_GT` is the distributed-average GT count already used by the criterion.
Unmatched queries, hand-raising, DN, intermediate decoder, pre-decoder and
encoder outputs receive no additional CE. Original VFL and all original loss
weights stay intact. This is not writing-only positive reweighting.

LQE adds a common scalar to class logits. It cancels from this two-class softmax,
so the added term does not directly train LQE or bbox heads. Shared decoder /
encoder / backbone features do receive classification gradients, and those
changes can indirectly affect localization. The matcher implementation and cost
are unchanged, but assignments may change after learned logits change.

Unlike the existing hinge Margin, conditional CE has no fixed-margin cutoff.
It may also overfit ambiguous or incorrect labels, and improving separation does
not guarantee calibrated scores or an AP gain.

## Implementation and controls

Only `src/zoo/dfine/dfine_criterion.py` changes model training code.
`use_pairwise_ce` defaults to false. No inference changes, parameters, buffers,
dataset edits, metric edits or new checkpoint keys are introduced. Existing
weights remain load-compatible. Train the comparison from the same initialization
procedure as baseline, NOT by resuming baseline's finished checkpoint.

Training config: `configs/dfine/dfine_s_scb3s_3cls_bs32_rw_pairwise_ce.yml`
inherits `dfine_s_scb3s_3cls_bs32.yml`: batch32, 132 epochs, same dataset, optimizer,
augmentation and architecture. Evaluation config with `_confusion.yml` suffix
adds only matrix export; it still requires `--test-only` when used for evaluation.

Run `python tools/diagnostics/test_pairwise_ce.py` from repository root. It tests
values, gradients, ignored samples, common-quality-shift invariance, input guards,
empty selections, CPU autocast, and full detector training forward/backward with
aux/DN outputs. Existing loss values must be exactly unchanged on identical
outputs when only this added term is enabled. This is a smoke test, not a complete
Colab CUDA/AMP training verification.

## Decision criteria

Compare on the SAME original SCB-S validation set and conf/IoU thresholds:
read->write, write->read, per-class AP/P/R, unmatched predictions, unmatched GT,
overall AP/AP50/AP75 and hand-raising metrics. Count changes alongside GT-normalized
rates. Do not compare error counts with the earlier 5205-GT cleaned dataset.

Both confusion directions matter. Decreasing write->read while increasing
read->write is a class tradeoff, not evidence that confusion is resolved. Lower
confusion counts caused by discarding more targets is not success. Use repeated
same-protocol runs to check whether an AP gain exceeds the new dataset's own
variation; do not reuse the old SCB-U standard deviation as SCB-S significance.

Training log should contain `loss_pairwise_ce`, without `_aux`, `_dn` or `_enc`
versions. Disable `use_pairwise_ce` to return to baseline loss behavior.
