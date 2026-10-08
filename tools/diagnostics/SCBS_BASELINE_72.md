# SCB-S / D-FINE-S：历史 72 轮基线复现

当前训练配置已改为 110/100，见 `SCBS_BASELINE_110.md`。本文件只用于复现已运行的历史 72/60 基线，对应配置固定为 `configs/dfine/dfine_s_scbs_hrw_72.yml`：

```yaml
epochs: 72
train_dataloader:
  dataset:
    transforms:
      policy:
        epoch: 60
  collate_fn:
    stop_epoch: 60
```

epoch 从 0 开始：0–59 为前 60 轮；60–71 为后 12 轮。第二阶段沿用原实现：关闭 policy 指定的强增强和多尺度，读取 best_stg1，调整 EMA，保留 best_stg1/best_stg2 的保存流程。不更改模型、损失、数据、batch=32、学习率、500-step warmup 或 EMA 参数，也不按比例调整学习率。

现有模块继承不带轮数后缀的当前配置，因此已经统一为 110/100；本文件的历史 72/60 周期不会改变它们。旧 Colab 或名称含 `scb3s`、`132` 的独立配置，以及 SCB5/SCB-U/其他规模均未修改。

原 132/120 基线完整保存在 `configs/dfine/dfine_s_scbs_hrw_132.yml`，旧 run1 的混淆矩阵配置、冻结特征探针继续引用它。现有权重、数据、日志未修改。重新导出旧特征时使用新输出目录，不覆盖已生成的缓存。

新 72 轮实验必须重新建立自己的基线；不能直接用旧五次 132 轮 AP 均值判断新模块的收益。短训练是否充分收敛、是否稳定，仍需运行验证。训练轮次减少约 45.5%，不保证实际耗时等比例下降或精度不下降。

## 同步配置（Mac；本次并未自动同步）

下列命令不上传数据/权重，不删除服务器文件。服务器上若有独立修改，先保存再同步。

```bash
cd /Users/wq/Desktop/D-FINE
rsync -avR \
  configs/dfine/dfine_s_scbs_hrw_72.yml \
  configs/dfine/dfine_s_scbs_hrw_132.yml \
  configs/dfine/dfine_s_scbs_hrw_72_diagnostics.yml \
  configs/dfine/dfine_s_scbs_hrw_confusion.yml \
  configs/diagnostics/scbs_feature_probe_run1.yml \
  tools/diagnostics/test_scbs_short_schedule.py \
  tools/diagnostics/SCBS_BASELINE_72.md \
  a5@100.76.151.67:/home/a5/MyProject/wq_project/D-FINE/
```

历史 72 轮诊断必须同时同步 `dfine_s_scbs_hrw_72.yml`。已有模块引用 `dfine_s_scbs_hrw.yml`，使用的是当前 110/100 周期，不是此处的历史周期。

## 新基线训练（服务器）

使用新目录，不续训旧 132 轮 checkpoint；若目录已存在，换 run 编号。

```bash
cd /home/a5/MyProject/wq_project/D-FINE
test ! -e "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs72_seed0_run1" && \
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
conda run --no-capture-output -n wq \
torchrun --master_port=7781 --nproc_per_node=1 train.py \
  -c configs/dfine/dfine_s_scbs_hrw_72.yml \
  --output-dir "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs72_seed0_run1" \
  --use-amp --seed=0
```

历史基线启动配置应显示 epochs=72、policy.epoch=60、collate_fn.stop_epoch=60；日志最终 epoch=71。不要用此处的历史权重替代新 110 轮基线的对照。

## 导出本次短基线的诊断

```bash
cd /home/a5/MyProject/wq_project/D-FINE
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
conda run --no-capture-output -n wq \
torchrun --master_port=7781 --nproc_per_node=1 train.py \
  -c configs/dfine/dfine_s_scbs_hrw_72_diagnostics.yml \
  -r "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs72_seed0_run1/best_stg2.pth" \
  --test-only \
  --output-dir "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs72_seed0_run1/diagnostics_eval" \
  --use-amp --seed=0
```

沿用原评分、IoU 阈值和后处理，不通过改变评估规则制造提升。best_stg2 的保存仍遵循原 solver 的改善判断；若不存在，先检查训练日志，不要随意把其他实验权重填入命令。

## 无数据检查

```bash
conda run --no-capture-output -n wq python tools/diagnostics/test_scbs_short_schedule.py
```

该检查当前验证 110/100 周期及旧 72/60、132/120 配置的保留和诊断隔离，不是完整 GPU 训练或效果验证。
