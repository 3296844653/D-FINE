# AQS + BoxLocal-L2 并行残差融合

第一轮只改变连接方式：两个分支读取同一份原始 L2 Query，分别得到增量后相加，再送入 L3。
这是待验证的结构实验，不预设一定优于单模块或旧串联组合。

## 计算与固定设置

设 Q 为 L2 原始分类 Query，B 为已预测的 L2 框，S 为原始分类头加 LQE 的分数：

```
dA = AQS(Q, S) - Q
dB = BoxLocal(Q, B, P3) - Q
Q_out = Q + dA + dB
```

- AQS 的门控、上下文与 BoxLocal 的采样均读取更新前的 Q；S 不来自局部更新后的 Query。
- 两个增量已经包含原模块的残差系数，不二次缩放，不重复相加原始 Q。
- “并行”指数据流分支；实现不要求 GPU 同时执行，两分支仍联合反向传播。
- BoxLocal：dim64、3×3 网格、offset0.15、固定 residual0.10。
- AQS：只在 L2，threshold 初值0.50、temperature0.10、raw residual 初值0.10；阈值/残差继续学习。
- 保留各分支末层零初始化，不新增参数；组合总参数10,492,533，与旧串联组合相同。
- BoxLocal 只处理普通300个 Query；DN 局部增量为零，AQS 的 DN/普通 Query 仍分开汇总。
- 融合结果进入 L2 分类和 L3。当前 L2 框已计算完，不被此次融合直接改写；L3 框可以变化。
- 原始损失、110轮/100轮停止增强、batch32、seed0、原始数据和 HGNet 预训练设置保持一致。
- 本轮不加入类别模糊度权重、额外损失、可学习融合门或新尺度。

独立配置：`configs/dfine/dfine_s_scbs_hrw_110_AQS_BoxLocalQueryAdapter_l2_parallel.yml`。
旧配置默认 `aqs_l2_local_fusion: serial`；只有新配置显式设为 `parallel`。
两个版本的张量键相同，但前向含义不同：评估必须使用各自的配置，不能把旧串联权重切到新配置当作重新训练的并行结果。

## Mac 同步

```bash
cd /Users/wq/Desktop/D-FINE
bash tools/diagnostics/sync_scbs_aqs_box_local_l2_parallel.sh
```

白名单同步两个模型源码文件及此实验的配置/工具，带备份、不删除文件，不复制数据、权重或旧输出。
此次新增模型计算路径，需要更新共享模型源码；若服务器此项目仍在运行，脚本退出且不替换文件。
也可加 `--wait` 自动等待 D-FINE 任务结束后同步；Mac 需保持联网。其他项目仅占用 GPU 时不妨碍同步。
`--dry-run` 只预览白名单，不连接 SSH。

## 服务器训练

建议在现有 tmux 会话执行：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
bash tools/diagnostics/run_scbs_aqs_box_local_l2_parallel_110.sh run1 --wait-gpu
```

默认只训练，不自动执行合成测试或 ON/OFF。
需要训练后自动导出混淆矩阵和完整诊断，在上述命令末尾加 `--diagnostics`。
等待 GPU 使用原有工具；GPU 空闲即启动，不终止其他进程。
可用 `SCBS_COMBO_GPU`（默认0）、`SCBS_COMBO_PORT`（默认7781）调整设备/端口。

新输出：
`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_AQS_BoxLocalQueryAdapter_l2_parallel_seed0_run1`。
目录已存在就拒绝覆盖；换 run2 仅改变目录编号，仍为 seed0，不是不同随机种子复验或断点续训。
脚本不加载旧单模块/串联实验权重。每次保存配置、融合模式、代码/标注哈希和 console.log。

可在 Mac 演练启动命令，无需 CUDA/Conda：

```bash
bash tools/diagnostics/run_scbs_aqs_box_local_l2_parallel_110.sh run1 --dry-run
```

## 验证与结果判断

本地合成检查入口：`python tools/diagnostics/test_aqs_box_local_l2_parallel.py`。
检查真实前向路由、原始门控分数、精确增量合并、DN隔离、原损失梯度、空标注/AMP、640前向、权重保存加载与部署。
旧串联回归入口：`python tools/diagnostics/test_aqs_box_local_l2.py`。
这些检查证明实现与训练接口可用，不证明精度提升；本机无 CUDA 时 FP16 项跳过。

主要对照：仅 AQS-L2 55.87 AP、仅 BoxLocal-L2 55.67 AP、旧串联组合55.06 AP（均为已有单次结果）。
同时比较整体/各类别 AP，以及相同 conf0.5/IoU0.5 下两个方向的 read/write 混淆和漏检。
只优于旧串联、但仍低于最佳单模块，不能认为已经充分利用两模块优势；有正向信号后再做多随机种子确认。
