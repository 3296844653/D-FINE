# SCB-S / D-FINE-S：110 轮基线，epoch 100 停增强

独立训练配置为 `configs/dfine/dfine_s_scbs_hrw_110.yml`，直接继承官方 S 配置，不继承可变的主配置。与另一个 80/72 基线对比时见 `SCBS_BASELINE_80_VS_110.md`。主配置 `dfine_s_scbs_hrw.yml` 仍保持 110/100：

```yaml
epochs: 110
train_dataloader:
  dataset:
    transforms:
      policy:
        epoch: 100
  collate_fn:
    stop_epoch: 100
```

epoch 从 0 开始：0–99 为前 100 轮；100–109 为后 10 轮。epoch 100 同时停止 policy 指定的强增强与 collate 多尺度，沿用原 solver 读取 best_stg1、刷新 EMA 和保存最佳权重的流程。

仅修改训练周期。模型结构、损失、类别顺序、数据路径、batch=32、优化器、学习率、500-step warmup、EMA 参数与评估规则保持不变。AQS、MFFE、MFFE P3-only、Encoder High-res Residual、rw_pairwise_ce_w005 自动继承相同的 110/100 周期。

已运行的 72/60 配置完整保存为 `dfine_s_scbs_hrw_72.yml`，72 轮诊断固定继承这个快照。132/120 配置和旧 run1 诊断、冻结特征探针不变。不修改权重、日志或数据集。新模块应与新 110 轮基线比较，而不是直接拿旧 72/132 轮 AP 判断收益。

## 同步到服务器（Mac 终端）

本次仅修改本地文件，未自动同步或启动训练。下列命令不删除服务器文件，也不上传数据或权重；服务器如果有独立修改，先保存再同步。

```bash
cd /Users/wq/Desktop/D-FINE
rsync -avR \
  configs/dfine/dfine_s_scbs_hrw.yml \
  configs/dfine/dfine_s_scbs_hrw_110.yml \
  configs/dfine/dfine_s_scbs_hrw_72.yml \
  configs/dfine/dfine_s_scbs_hrw_72_diagnostics.yml \
  configs/dfine/dfine_s_scbs_hrw_110_diagnostics.yml \
  tools/diagnostics/test_scbs_short_schedule.py \
  tools/diagnostics/SCBS_BASELINE_110.md \
  a5@100.76.151.67:/home/a5/MyProject/wq_project/D-FINE/
```

已有模块配置直接继承同步后的基线，即使用新周期。此同步命令假设服务器已经有原项目、模块源码、132 轮配置快照及其依赖。

## 基线训练（服务器）

使用新输出目录，不续训旧 72/132 轮权重。输出目录若存在，请换 run 编号。若端口正在被使用，也需要更换端口。

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

启动配置应显示 epochs=110、policy.epoch=100、collate_fn.stop_epoch=100；最后一轮日志为 epoch 109。是否充分收敛、精度及重复运行波动仍需实际实验验证。

## 本次基线诊断（服务器）

110轮Run1的轻量复核（包括自动交叉统计与Encoder Top300覆盖率）见
`tools/diagnostics/SCBS_110_RUN1_RECHECK.md`。建议用其中的一条命令，不需要重新训练。
下面仍保留仅导出混淆矩阵与query的命令。

```bash
cd /home/a5/MyProject/wq_project/D-FINE
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
conda run --no-capture-output -n wq \
torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
  -c configs/dfine/dfine_s_scbs_hrw_110_diagnostics.yml \
  -r "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_seed0_run1/best_stg2.pth" \
  --test-only \
  --output-dir "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_seed0_run1/diagnostics_eval" \
  --use-amp --seed=0
```

诊断输出使用新目录。best_stg2 是否保存仍按原 solver 的指标改善判断；如果不存在，请先检查训练日志，不要填入其他实验的权重。

## 无数据检查

```bash
conda run --no-capture-output -n wq python tools/diagnostics/test_scbs_short_schedule.py
```

检查仅三项训练设置变化、模块继承一致、epoch 100 停强增强和多尺度，以及旧 72/132 轮配置完整保留和诊断隔离。此检查不是服务器完整训练或精度验证。
