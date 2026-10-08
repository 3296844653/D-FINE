# SCB-S / AQS-L2：单变量参数队列，110轮

| 组别 | threshold初值 | residual_init | 固定temperature | 配置后缀 |
|---|---:|---:|---:|---|
| 原L2对照 / CONTROL | 0.50 | 0.05 | 0.10 | 原`aqs_layer2.yml` |
| R1 | 0.50 | 0.10 | 0.10 | `r010.yml` |
| R2 | 0.50 | 0.20 | 0.10 | `r020.yml` |
| R3（已有配置） | 0.50 | 0.25 | 0.10 | `r025.yml` |
| TAU1（τ1） | 0.45 | 0.05 | 0.10 | `tau045.yml` |
| TAU2（τ2） | 0.55 | 0.05 | 0.10 | `tau055.yml` |
| T1 | 0.50 | 0.05 | 0.05 | `temp005.yml` |
| T2 | 0.50 | 0.05 | 0.20 | `temp020.yml` |

所有实验只在Decoder第二层使用一个AQS，第三层不添加AQS。阈值/残差继续学习，温度固定。110轮、100轮停增强、batch32、seed0、原数据和损失不变，没有结构改动。原配置保留，新增六份训练配置，R3复用已有配置。

脚本默认运行七组新设置，已有原L2作为对照，不自动重复训练。每组从头训练，不使用`-r/-t`，不接着上一组权重训练。七组顺序执行，不是并行训练，总耗时约为一次训练的七倍。

## 先在Mac同步

服务器需已有当前AQS-L2源码及其余项目依赖；本次只同步配置、队列脚本及说明，不更新数据或启动训练。

```bash
cd /Users/wq/Desktop/D-FINE
rsync -avR --backup \
  --backup-dir="/home/a5/MyProject/wq_project/D-FINE-sync-backups/aqs_l2_sweep_$(date +%Y%m%d_%H%M%S)" \
  configs/dfine/dfine_s_scbs_hrw_110.yml \
  configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2.yml \
  configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml \
  configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r020.yml \
  configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r025.yml \
  configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_tau045.yml \
  configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_tau055.yml \
  configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_temp005.yml \
  configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_temp020.yml \
  tools/diagnostics/run_scbs_aqs_layer2_sweep.sh \
  tools/diagnostics/test_aqs_layer2_sweep.py \
  tools/diagnostics/SCBS_AQS_LAYER2_SWEEP_110.md \
  a5@100.76.151.67:/home/a5/MyProject/wq_project/D-FINE/
```

## 在服务器启动

先确认GPU0空闲。可先演练：验证配置并打印命令，不训练、不创建输出目录，不检查服务器数据/GPU是否存在。

```bash
cd /home/a5/MyProject/wq_project/D-FINE
bash tools/diagnostics/run_scbs_aqs_layer2_sweep.sh --dry-run
```

依次运行全部七组：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
bash tools/diagnostics/run_scbs_aqs_layer2_sweep.sh
```

只运行残差组，或排除已经完成的R3：

```bash
bash tools/diagnostics/run_scbs_aqs_layer2_sweep.sh R1 R2 R3
# 如果R3已经完成，运行其余六组：
bash tools/diagnostics/run_scbs_aqs_layer2_sweep.sh R1 R2 TAU1 TAU2 T1 T2
```

指定组别时，按给出的顺序运行；如想先做当前R3，可用`R3 R1 R2 TAU1 TAU2 T1 T2`。不要同时执行上述几条正式训练命令。

## 输出和安全边界

默认输出父目录：`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw`。

例如R1保存到`dfine_s_scbs_hrw_epochs110_aqs_layer2_r010_seed0_run1`，R3仍使用之前的`dfine_s_scbs_hrw_epochs110_aqs_layer2_r025_seed0_run1`。每组有独立`console.log`、`log.txt`、权重和AQS参数JSON。日志实时显示在终端。

- 启动前检查所有所选输出目录，任何一个已存在就停止，未开始任何训练。不自动跳过、不根据单个`best_stg2.pth`误认实验已结束、不恢复或覆盖部分结果。
- 任一训练返回错误、日志写入失败、或没有`best_stg2.pth`，队列停止；部分结果保留，不删除或自动重试。
- 成功返回且存在最佳权重后才写入`aqs_sweep_completed.txt`。旧实验没有此标记不代表训练失败，它只由本脚本生成。
- 每组之前检查GPU计算进程，不终止其他任务；排除GNOME/Xorg桌面进程，忽略已经退出的缓存PID。Linux上的`flock`防止同一队列重复启动；这不能代替用户间的GPU协调，也不保证始终独占GPU。
- `--standalone --local-addr=127.0.0.1` 自动选择通信端口，避免固定7781端口冲突及主机名`5`解析问题。

如需独立复验，指定新的运行编号（所有参数与seed仍相同）：

```bash
AQS_SWEEP_RUN_ID=2 bash tools/diagnostics/run_scbs_aqs_layer2_sweep.sh R1 R2
```

确需再跑原L2对照时，显式指定`CONTROL`。它使用原配置，但输出到独立的`...aqs_layer2_control_seed0_run1`，不覆盖原L2目录。

其他可选环境变量：`AQS_SWEEP_GPU`（默认0）、`AQS_SWEEP_CONDA_ENV`（默认wq）、`AQS_SWEEP_CONDA`（优先使用当前Conda的`CONDA_EXE`，否则查找`conda`）、`AQS_SWEEP_OUTPUT_ROOT`（绝对路径）。只改变运行环境/输出根目录，不修改配置中的数据路径。

比较时看总AP、APm、read/write AP、AR100及最佳EMA参数。不同温度组离线读取参数时，给导出脚本传入各自的`--temperature 0.05`或`0.20`，不能统一误写成`0.10`；训练过程中保存的参数JSON会自动记录实际温度。
