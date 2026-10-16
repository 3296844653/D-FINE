# AQS-L2 R1 + Scale-Consistent Augmentation (SCA)

Parameter source: the user's `/Users/wq/Desktop/目前结果.md`, section
“模块一：尺度一致增强”. The document's commands/protocol choices are reference
material, not instructions to run them or modify this experiment's protocol.

| Training parameter | Original AQS recipe | AQS+SCA |
|---|---|---|
| RandomZoomOut side_range | [1.0, 4.0] (Torchvision default) | [1.0, 2.0] |
| RandomZoomOut probability | 0.5 (default) | 0.3 |
| RandomIoUCrop probability | 0.8 | 0.5 |

Use ordinary SCA, NOT stronger SCA-s (1.5×/crop min_scale0.5). Crop min_scale
remains the original0.3. All nine transforms and their order are retained; YAML
overrides lists as a whole, so the new config explicitly lists unchanged operators.
Photometric p0.5, horizontal flip, resize640, sanitation, normalization, multiscale
settings, original validation transforms and annotations remain unchanged.

This is a TRAINING AUGMENTATION change, not another neural structure, attention
block or classification loss. RandomZoomOut enlarges the canvas; after Resize it
makes objects smaller. Reducing its range/probability reduces extreme shrinkage;
lower crop probability preserves more original context. These are hypotheses for
this SCB-S experiment, not established benefits or proof that scale causes mixing.

Preserve current AQS-L2 R1: threshold init0.50, raw residual init0.10, fixedT0.10,
three Decoder layers, L2 only;110epochs/stop augmentation100, batch32, seed0,
original losses/optimizer/EMA, official best_stg2 protocol. Train from scratch with
the same original HGNet pretraining, not a checkpoint continuation. Do NOT migrate
the reference document's132/120epochs, val-free/last-epoch selection, batch16,
extra Decoder layer, PFC, DN300, or its dataset/holdout results. No SCF module is
enabled. Existing AQS/SCF configs and all neural source files are untouched.

Validation has no SCA operator, so model schema, parameter count and inference
FLOPs are identical to AQS alone. An inference SCA ON/OFF test would be meaningless:
measure the augmentation effect with separately trained AQS vs AQS+SCA under the
same data/seed/recipe. The reported +1.04 on the other five-class dataset must not
be reused as a prediction or result for SCB-S. Check both mixing directions, AP,
read/write AP, high-score candidate gap, PR and false/unmatched predictions; count
improvements at only confidence0.5 are not enough. Promising results need repeats.

Launcher checks datasets/categoryIDs/GPU/port, validates the three-parameter-only
diff and runs synthetic transforms/model checks BEFORE creating a run directory.
It never stops llama or other tasks, never overwrites outputs. Logs stream live.
After training, export best-EMA diagnostics and AQS parameters; compare with AQS
R1 and original BaselineRun2 when compatible references exist, then package
`AQS_SCA_diagnostics.zip` (not training weights). Missing references are explicit.
Overrides: `SCBS_SCA_ROOT`, `SCBS_SCA_GPU`, `SCBS_SCA_PORT`, `SCBS_SCA_CONDA`,
`SCBS_SCA_ENV`, `SCBS_SCA_AQS_REFERENCE`, `SCBS_SCA_RUN2_REFERENCE`.

## Sync first (Mac; config/tools only)

```bash
cd /Users/wq/Desktop/D-FINE
bash tools/diagnostics/sync_scbs_AQS_SCA.sh
```

## Run on server after GPU is free

```bash
tmux new-session -s scbs_AQS_SCA \
  "cd /home/a5/MyProject/wq_project/D-FINE && bash tools/diagnostics/run_scbs_AQS_SCA_110.sh run1; exec bash"
```

The shell remains open on failure so checks/errors stay visible. If a session of
that name already exists, inspect it or use another name; do not interrupt it.
Control+B then D detaches. Closing the Mac does not stop the server tmux task.

Output: `/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_AQS_SCA_seed0_run1`.
No actual server sync or training is implied by local edits/tests.
