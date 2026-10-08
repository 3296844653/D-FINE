# SCB-S / D-FINE-S：独立的 80/72 与 110/100 基线

两份配置各自直接继承官方 S 配置 `dfine_hgnetv2_s_coco.yml`，不继承可变的 `dfine_s_scbs_hrw.yml`，也不互相继承。

| 配置 | 总轮次 | policy.epoch / collate_fn.stop_epoch | 第二阶段 |
|---|---:|---:|---:|
| `configs/dfine/dfine_s_scbs_hrw_80.yml` | 80 | 72 | 8 轮，epoch 72–79 |
| `configs/dfine/dfine_s_scbs_hrw_110.yml` | 110 | 100 | 10 轮，epoch 100–109 |

到指定 epoch 时，同时关闭 policy 指定的强增强与多尺度，原 best_stg1 回载、EMA 和最佳权重保存流程不改。训练周期三项不同，默认输出目录也分开；模型结构、原损失、数据划分与服务器路径、三类顺序、batch=32、学习率、warmup、EMA 参数、输入及评估规则完全一致，不开启实验模块。

这是比较两套训练周期，并非只改变总 epochs 的单变量实验。精度、收敛情况及实际耗时需要运行后判断，不能预先保证哪套更好。

现有主配置 `dfine_s_scbs_hrw.yml` 及模块配置仍是 110/100，不自动改成 80/72。旧 72/60、132/120 配置、权重、日志和数据均不修改。选定基线周期后，模块需要与相同周期的基线比较。

## 同步到服务器（Mac 终端）

本次未自动同步或运行服务器训练。只同步以下配置和检查文件，不上传数据或权重，也不删除文件；服务器若有独立修改，先保存再同步。

```bash
cd /Users/wq/Desktop/D-FINE
rsync -avR \
  configs/dfine/dfine_s_scbs_hrw_80.yml \
  configs/dfine/dfine_s_scbs_hrw_110.yml \
  configs/dfine/dfine_s_scbs_hrw_80_diagnostics.yml \
  configs/dfine/dfine_s_scbs_hrw_110_diagnostics.yml \
  tools/diagnostics/test_scbs_baseline_profiles.py \
  tools/diagnostics/test_scbs_short_schedule.py \
  tools/diagnostics/SCBS_BASELINE_80_VS_110.md \
  a5@100.76.151.67:/home/a5/MyProject/wq_project/D-FINE/
```

假设服务器已有完整项目和官方 S 配置依赖。新检查脚本复用已有 `test_scbs_short_schedule.py` 中的无数据测试组件。

## 80 轮基线（服务器）

```bash
cd /home/a5/MyProject/wq_project/D-FINE
test ! -e "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs80_seed0_run1" && \
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
conda run --no-capture-output -n wq \
torchrun --master_port=7781 --nproc_per_node=1 train.py \
  -c configs/dfine/dfine_s_scbs_hrw_80.yml \
  --output-dir "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs80_seed0_run1" \
  --use-amp --seed=0
```

## 110 轮基线（服务器）

80 轮完成后再运行，不在同一张 GPU 上同时启动两组训练。

```bash
cd /home/a5/MyProject/wq_project/D-FINE
test ! -e "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_seed0_run1" && \
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
conda run --no-capture-output -n wq \
torchrun --master_port=7781 --nproc_per_node=1 train.py \
  -c configs/dfine/dfine_s_scbs_hrw_110.yml \
  --output-dir "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_seed0_run1" \
  --use-amp --seed=0
```

若输出目录存在，换 run 编号，不覆盖或续训旧结果；若端口被占用，换端口。两组均不使用 `-r` 或 `-t`，采用相同预训练初始化策略和 seed。需要复验时保持训练参数和 seed 不变，只改变输出 run 编号。

## 各自导出诊断

```bash
cd /home/a5/MyProject/wq_project/D-FINE
PROFILE_EPOCHS=80
PROFILE_RUN=1
PROFILE_DIR="/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs${PROFILE_EPOCHS}_seed0_run${PROFILE_RUN}"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
conda run --no-capture-output -n wq \
torchrun --master_port=7781 --nproc_per_node=1 train.py \
  -c "configs/dfine/dfine_s_scbs_hrw_${PROFILE_EPOCHS}_diagnostics.yml" \
  -r "$PROFILE_DIR/best_stg2.pth" \
  --test-only --output-dir "$PROFILE_DIR/diagnostics_eval" \
  --use-amp --seed=0
```

另一组将 `PROFILE_EPOCHS` 改成 110；`PROFILE_RUN` 必须与实际训练目录一致。使用新诊断目录，检查权重属于对应周期。best_stg2 保存仍遵循原 solver 的改善判断；若不存在，先查看训练日志，不填其他实验权重。

## 结果比较

记录各组最高 AP 所在轮次及同一 checkpoint 的 AP、AP50、AP75、APm、APl、AR100、hand/read/write AP 和实际训练时间；观察最后阶段的 AP 趋势。单次比较只作初步判断，重复运行确认波动后再决定长期采用哪个周期。

## 无数据检查

```bash
conda run --no-capture-output -n wq python tools/diagnostics/test_scbs_baseline_profiles.py
```

验证独立继承、配置仅周期和输出目录不同、两处停止时点正确、第二阶段长度、诊断各自对应，以及相同随机初始化下两组模型参数与前向结果一致。本地合成检查不代表完整 GPU 训练或精度验证。
