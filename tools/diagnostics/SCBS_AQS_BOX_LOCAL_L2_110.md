# AQS-L2 R1 + BoxLocalQueryAdapter-L2 联合消融

这是同一个 D-FINE-S 同时启用两个独立模块、进行一次联合训练，不是同时启动两个训练任务，
也不是将两份训练好的权重相加，更不是已证明有效的新融合模块。单模块配置及实现保持不变。

执行顺序固定为：

```text
Decoder L2 query + L2预测框 + Encoder P3
                ↓
BoxLocalQueryAdapter（框内9点自适应采样，普通query）
                ↓
当前分类头 + LQE 得分
                ↓
AQS-L2（高置信query门控与query上下文）
                ↓
Decoder L3 → 最终类别/框预测
```

局部模块先提供图像细节，AQS 再利用局部增强后的 query 及其分数聚合上下文：

\[
Q_l=Q_2+0.1 f_l(Q_2,\operatorname{Sample}(P3,\operatorname{detach}(b_2)))
\]

\[
s_l=\max_c\sigma(\operatorname{LQE}(\operatorname{Cls}_2(Q_l),d_2)_c),\quad
g_i=\mathbf{1}[s_{l,i}\geq\sigma(\theta)]
\]

\[
C_l=\frac{\sum_i g_i Q_{l,i}}{\max(1,\sum_i g_i)},\quad
Q_a=Q_l+\tanh(\rho)g\,f_a([\operatorname{LN}(Q_l),C_l])
\]

AQS 使用原有 straight-through 门控训练方式，温度固定为 0.10。阈值初值 0.50、
原始残差参数初值 0.10，二者继续学习；不使用训练结束后的数值替代初值。
局部模块保持原有维度、0.1 残差缩放和零输出初始化。DN query 不进入局部采样，
DN/普通 query 的 AQS 上下文分开计算。

L2 当前 bbox/FDR 先计算，两个模块不直接修改它；更新后的 query 进入 L3，因此最终 bbox
仍可能变化。原 VFL、bbox 损失、匹配、Top300、数据及110/100训练流程均保持不变。

## 命名与同步、运行

组合配置：`configs/dfine/dfine_s_scbs_hrw_110_AQS_BoxLocalQueryAdapter_l2.yml`。
配置文件和输出目录均带 `AQS_BoxLocalQueryAdapter_l2`，不使用 N 编号代替模块名。
后续改动先给本地同步命令，再给服务器运行命令。

先在本地 Mac 终端同步本次源码、配置、诊断工具；不覆盖服务器基线配置，不删除任何服务器文件：

```bash
cd /Users/wq/Desktop/D-FINE
rsync -avR --backup --suffix=".before_AQS_BoxLocalQueryAdapter_l2_$(date +%Y%m%d_%H%M%S)" \
  src/zoo/dfine/dfine.py \
  src/zoo/dfine/dfine_decoder.py \
  src/zoo/dfine/night_probes.py \
  configs/dfine/dfine_s_scbs_hrw_110_AQS_BoxLocalQueryAdapter_l2.yml \
  tools/diagnostics/run_scbs_aqs_box_local_l2_110.sh \
  tools/diagnostics/analyze_detection_diagnostics.py \
  a5@100.76.151.67:/home/a5/MyProject/wq_project/D-FINE/
```

再在服务器终端运行：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
bash tools/diagnostics/run_scbs_aqs_box_local_l2_110.sh
```

默认输出：
`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_AQS_BoxLocalQueryAdapter_l2_seed0_run1`

脚本顺序执行完整训练、最佳 EMA 评估、离线分析，日志实时显示并落盘。
旧目录存在即停止，可传入新的绝对输出目录。`--dry-run` 只核对配置和源码接口，不训练。

## 验证目标

统一与 Baseline Run2 比较，同时与 AQS-L2 R1、Local-L2 单模块比较：AP、write AP、
read↔write混淆、write Precision/Recall、write/中尺度write候选缺口、举手退化及未匹配预测。
组合要证明单模块不能同时获得的收益，不预设两个模块的提升可以直接相加。
若之后将其重新设计为一个融合模块，那是另一项结构实验，不能沿用本次组合结果作证。
