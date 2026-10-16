# BoxLocalQueryAdapter-L2：单因素参数实验

只修改 `/Users/wq/Desktop/D-FINE`，再同步到服务器训练。
这是此前**单独BoxLocal-L2**的参数实验，不叠加AQS、SERF或其他模块，
不改损失、数据、优化器、Top300和DN路由，不从旧权重微调。
继承服务器110轮基线：100轮停止增强、batch32、seed0、原HGNet预训练。

## 组别

默认：dim64 / grid3×3 / offset0.15 / residual0.10，沿用已有结果，不重跑。

| 组别 | 唯一变化 | 验证目标 |
|---|---|---|
| r005 | 固定残差0.05 | 减弱增强 |
| r020 | 固定残差0.20 | 增强是否过强/不足 |
| o010 | 偏移幅度上限0.10 | 更受约束的取样 |
| o020 | 偏移幅度上限0.20 | 更自由的取样；仍限制在框内 |
| d032 | 内部维度32 | 较小容量 |
| d096 | 内部维度96 | 较大容量 |
| g2 | 2×2，共4点 | 网格结构消融 |
| g5 | 5×5，共25点 | 网格结构消融 |
| o000 | 偏移幅度0 | 固定网格对照，仍保留query条件局部注意力 |

前缀均为`dfine_s_scbs_hrw_110_BoxLocalQueryAdapter_l2_`，例如r005配置
是`configs/dfine/dfine_s_scbs_hrw_110_BoxLocalQueryAdapter_l2_r005.yml`。
各配置只列启用模块、参数记录开关和本组唯一改动，其他值用旧默认值。

残差和偏移幅度上限是**固定超参数，不自动学习**。偏移预测权重、
注意力投影和融合MLP继续由原检测损失训练。o000的偏移头梯度为0，
但模块仍读取局部证据。0.20偏移更容易触及±0.49边界，不能假定一定更好。
网格组同时改变点数、初始网格布局和偏移头尺寸，不是单纯改变一个数值
缩放系数；属于结构消融。dim变化也改变参数量/计算量，应一起报告。

默认旧配置的参数键、初始化随机数状态、前向公式保持兼容。
新实验开启`box_local_record_settings: true`，将固定参数作为INT64
buffer保存到checkpoint，不是新增可学习参数。严格加载时拒绝参数不同
的配置；评估新权重必须使用对应组的配置，不能套旧默认配置。

## 先同步（Mac）

```bash
cd /Users/wq/Desktop/D-FINE
bash tools/diagnostics/sync_scbs_BoxLocalQueryAdapter_l2_sweep.sh
```

同步白名单中的共享源码、新配置和工具；不覆盖服务器基线配置，
不删除旧结果/权重，不修改标注。替换源码有服务器同级备份。
如果D-FINE训练、评估或队列仍在运行，拒绝同步共享源码；其他项目不停止。
服务器原`dfine_s_scbs_hrw_110.yml`必须存在并指向原SCB-S数据。

## 第一轮：默认只跑残差两组（服务器SSH终端）

```bash
cd /home/a5/MyProject/wq_project/D-FINE
tmux new-session -d -s scbs_BoxLocal_sweep \
  'cd /home/a5/MyProject/wq_project/D-FINE && bash tools/diagnostics/run_scbs_BoxLocalQueryAdapter_l2_sweep_110.sh run1 --wait-gpu; exec bash'
tmux attach -t scbs_BoxLocal_sweep
```

Ctrl+B，松开，再按D脱离tmux，可关闭Mac。服务器重启仍会中断。
已存在同名tmux时先attach查看，不删除会话。日志实时展示并保存console.log。
GPU被计算任务占用则等待，不杀llama/其他任务；端口占用则报错，不抢占。

## 其他选择

仅预览，不创建目录、不占GPU、不启动训练：

```bash
bash tools/diagnostics/run_scbs_BoxLocalQueryAdapter_l2_sweep_110.sh run1 --all --dry-run
```

如果还没有启动任何组，`--all --wait-gpu`顺序跑全部9组。不是并行挤占GPU，
也不会叠加9组参数；每组独立完整训练。总耗时至少约9倍单组训练时间再加诊断，
不会保证一个晚上跑完。

残差两组完成后，在tmux内运行其余7组（不重复前两组）：

```bash
bash tools/diagnostics/run_scbs_BoxLocalQueryAdapter_l2_sweep_110.sh run1 --wait-gpu \
  --groups o010 o020 d032 d096 g2 g5 o000
```

只跑指定组也用`--groups`。任何选中输出存在则停止，保留旧结果。
`--all`不会跳过旧组；已有组应明确排除，或用run2等新编号。

## 输出和对比

输出根：`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw`。
各组例如`dfine_s_scbs_hrw_epochs110_BoxLocalQueryAdapter_l2_r005_seed0_run1`。
队列目录`BoxLocalQueryAdapter_l2_sweep_epochs110_seed0_run1_r005_r020`，
包含status.json和results.csv（AP数值0–1，不是百分数）。

每组自动执行完整训练→最佳stage2 EMA→参数导出→ON/OFF评估→诊断分析→ZIP。
完整0..109轮日志、best_stg2.pth和原始源码/标注哈希均需通过核验。
ON/OFF是同一联合训练EMA的功能诊断，不等于独立训练基线；关闭Box也可能
改变L3定位。ZIP不打包权重和raw_queries；完整数据留服务器。

自动比较两份对照：

- **BoxLocal_Default**：已有BoxLocal-L2的默认参数结果。
- **BaselineRun2**：此前核验的110轮原基线Run2。

默认查找模块名输出目录和旧night_n4输出的diagnostics_eval；Run2查找
diagnostics_eval、diagnostics_export1或根目录下对应*_diagnostics_export1。
找不到时在comparison_status.json中明确记录，仍保留新结果，不伪造对照。
可在启动前设置`SCBS_BOX_DEFAULT_REFERENCE`、`SCBS_BOX_RUN2_REFERENCE`
为实际包含analysis和evaluation的诊断目录。

关注整体AP/read/write AP、两方向混淆、APm与中尺度write AP、正确高分候选缺口、
PR和未匹配预测，不只看confidence=0.5下的计数；未匹配不等于确定背景。
参数搜索不等于结构创新，单次增益需复验，也不应反复用验证集搜索后把它当
独立测试集证据。固定尺度参数不会输出“训练后学到的新值”。

`--export-only`仅用于完整训练且尚未导出诊断的原目录，不恢复训练、不覆盖诊断，
还需原源码哈希一致。中断队列保留已有日志和结果，不清理其他进程。

## 验证

启动前自动运行test_box_local_sweep.py：旧默认权重/RNG/数值/梯度兼容、
9组单因素配置、网格/框约束/坐标detach、固定网格、CPU AMP、原损失/DN/
aux/空GT/EMA、严格参数加载、640前向/部署、微型COCO全链路和文件保护。
CUDA FP16及batch32×300-query分支检查仅在服务器执行；分支检查不保证
完整batch32所有多尺度图像的峰值显存。OOM会停止并保留日志，不悄悄改batch。
合成测试通过不表示实际精度提升。
