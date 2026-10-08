# AQS 训练后参数导出

只增加只读日志与 JSON 导出，不改变模块、损失、优化器、随机数或 checkpoint 的保存/回载规则。AQS 未启用时不新增日志字段或参数文件。

## 以后训练自动输出

训练配置、命令不变。每轮 `log.txt` 增加 `aqs_parameters`，TensorBoard 增加 `AQS/model/...` 与 `AQS/ema/...`。训练正常完成后，终端显示最后一轮评估参数及已有的最佳阶段参数。

原实验输出目录新增：

- `aqs_parameters_final.json`：最后一轮**评估时**的参数，正常训练结束时写出。
- `aqs_parameters_last_evaluated.json`：每轮更新，即使随后中断也保留最后已评估的一轮。
- `aqs_parameters_best_stg2.json`：与 `best_stg2.pth` 保存动作同步，只有该权重确实保存后才有。
- `aqs_parameters_best_stg1.json`：同理对应第一阶段最佳权重。

同时输出 model 和 EMA。对照检测评估通常看 EMA。`epoch` 是训练循环的 **0-based** 轮次。

**原 solver 在第二阶段可能回载 best_stg1，last.pth 仅在第一阶段更新。因此这里在评估后、回载前采样，不把训练结束时可能回退的内存参数或 last.pth 当作最后一轮参数。** 旧实验如果没保存最后一轮权重/快照，不能事后恢复那一轮；但仍可以精确读取已有 best_stg2。

## 已训练实验：不重训，不推理，无需 GPU 或数据

在服务器运行：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
conda run --no-capture-output -n wq \
python tools/diagnostics/export_aqs_parameters.py \
  --checkpoint "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_aqs_seed0_run1/best_stg2.pth" \
  --temperature 0.10
```

输出保存在权重旁边：`best_stg2_aqs_parameters.json`。已有 JSON 时默认拒绝覆盖；明确重导可添加 `--overwrite`，不改动 `.pth`。

旧110轮实验则把 checkpoint 改为：

```text
/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_aqs_seed0_run1/best_stg2.pth
```

脚本只需要 PyTorch，使用 `weights_only=True` 安全读取，不自动回退到执行任意 pickle。必须传模型权重，不是 `eval.pth`（后者是 COCO 评估结果）。

## 字段说明

| 字段 | 含义 |
|---|---|
| `threshold_logit` | 学到的原始阈值参数 |
| `threshold` | `sigmoid(threshold_logit)`，实际 query 选择阈值 |
| `residual_scale` | 学到的原始残差参数 |
| `effective_residual_scale` | `tanh(residual_scale)`，前向实际残差系数 |
| `temperature` | 固定温度，不参与学习 |

温度不在 checkpoint state_dict 中；单独读取旧权重时不提供 `--temperature` 就显示 unknown。提供的值须来自**这次旧实验**的原配置，JSON 会标记为用户提供，不伪装成权重恢复值。MLP/LayerNorm 的全部权重仍在模型 checkpoint 中，本导出是关键标量而非全部张量。

仅凭残差系数接近零，不能断言 AQS 无作用，还取决于 MLP 输出的幅度。参数变化也不等于 AP 提升。

## Mac 同步此次修改（不上传数据、不启动训练）

```bash
cd /Users/wq/Desktop/D-FINE
rsync -avR --backup \
  --backup-dir="/home/a5/MyProject/wq_project/D-FINE-sync-backups/aqs_parameters_$(date +%Y%m%d_%H%M%S)" \
  src/solver/det_solver.py \
  src/solver/aqs_parameters.py \
  tools/diagnostics/export_aqs_parameters.py \
  tools/diagnostics/test_aqs_parameters.py \
  tools/diagnostics/AQS_PARAMETER_EXPORT.md \
  a5@100.76.151.67:/home/a5/MyProject/wq_project/D-FINE/
```

若只要读取旧权重、不需要未来自动日志，只须同步 `src/solver/aqs_parameters.py` 和 `tools/diagnostics/export_aqs_parameters.py`。小样本检查：`python tools/diagnostics/test_aqs_parameters.py`。
