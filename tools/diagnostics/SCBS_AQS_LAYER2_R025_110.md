# SCB-S / AQS-L2-3：阈值0.50、残差初值0.25

只调整第二层单层 AQS 的原始残差初值，从0.05改为0.25；第三层不添加 AQS，仍接收第二层增强后的 query。两份原 L2 配置保留。

保持阈值初值0.50、固定温度0.10、110轮/100轮停增强、batch32、seed0、三类、数据路径、原损失及模块位置不变。不使用双层配置，不叠加其他模块。

残差系数仍可学习，实际缩放是 `tanh(rho)`：初始 `rho=0.25` 对应约0.244919，不是把有效缩放固定为0.25。MLP末层仍零初始化，初始前向为恒等增强，不保证训练后效果提升。

## 在Mac同步新增文件

服务器已有当前 L2 源码以及110轮基线配置时，只需同步本次新增文件；无需更新数据集或源码。

```bash
cd /Users/wq/Desktop/D-FINE
rsync -avR --backup \
  --backup-dir="/home/a5/MyProject/wq_project/D-FINE-sync-backups/aqs_l2_r025_$(date +%Y%m%d_%H%M%S)" \
  configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r025.yml \
  configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r025_diagnostics.yml \
  tools/diagnostics/SCBS_AQS_LAYER2_R025_110.md \
  a5@100.76.151.67:/home/a5/MyProject/wq_project/D-FINE/
```

## 在服务器从头训练

先确认GPU0空闲。使用新输出目录，不加 `-r/-t`，避免加载旧权重覆盖新的可学习参数初值。下列脚本发现输出目录已存在就停止，不覆盖旧结果。

```bash
bash <<'BASH'
set -euo pipefail
cd /home/a5/MyProject/wq_project/D-FINE
AQS_L2_R025_OUT="/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_aqs_layer2_r025_seed0_run1"
if [[ -e "$AQS_L2_R025_OUT" ]]; then
  printf '输出目录已存在，请使用新的运行编号：%s\n' "$AQS_L2_R025_OUT"
  exit 1
fi
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
conda run --no-capture-output -n wq \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py -c configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r025.yml \
--output-dir "$AQS_L2_R025_OUT" --use-amp --seed=0
BASH
```

## 训练后诊断

```bash
cd /home/a5/MyProject/wq_project/D-FINE
AQS_L2_R025_RUN="/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_aqs_layer2_r025_seed0_run1"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
conda run --no-capture-output -n wq \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py -c configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r025_diagnostics.yml \
-r "$AQS_L2_R025_RUN/best_stg2.pth" --test-only \
--output-dir "$AQS_L2_R025_RUN/diagnostics_eval" --use-amp --seed=0
```

参数仍自动分别记录在每轮日志、最后轮评估JSON和最佳权重JSON中。也可单独读取本实验的最佳权重：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
conda run --no-capture-output -n wq \
python tools/diagnostics/export_aqs_parameters.py \
--checkpoint "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_aqs_layer2_r025_seed0_run1/best_stg2.pth" \
--temperature 0.10
```

对照同设备、同训练和评估流程的原110轮L2实验（0.50/0.05）。重点比较总AP、APm、read/write AP、AR100及最佳EMA的阈值/残差。原基线用于辅助对照；不要将本次初值改动误当成新增结构或已证明的机制改进。
