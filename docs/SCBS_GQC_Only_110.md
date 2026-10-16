# 原始D-FINE-S + GQC 单模块

这是对**原始GeometryAwareQueryCompetition**的独立训练实验，不是RQC，也不是关闭联合模型中的AQS再推理。

- 直接继承原始`dfine_s_scbs_hrw_110.yml`，不继承AQS或BoxLocal配置。
- AQS、BoxLocal及其他新增模块关闭；原骨干、编码器、Top300查询和三层Decoder不变。
- 原GQC结构/参数不改：dim64、残差初始化0.10、chunk32、gradient checkpoint开启，新增54,341参数。
- 本地640×640、batch1的部署calflops统计：参数10.232542M、FLOPs约24.954056G；不是实测推理延迟。
- 在L3原始分类和LQE之后，只修正300个普通查询的最终分数；不直接改框、不处理DN查询。训练仍可通过原损失间接影响整网。
- 保持原损失与权重、110轮/第100轮停止增强、batch32、EMA和原HGNet预训练；不从AQS/GQC/BoxLocal权重继续训练。
- 构造分支保持原模型RNG；输出零初始化，与原始D-FINE起点等价。仅显式`allow_gqc_standalone: true`允许这个单模块，其他三个旧行为模块仍要求原AQS-L2设置。
- 新输出目录带`GQC_only`，不覆盖旧配置、实验结果、数据或权重。

## 推荐：现在提交，在服务器后台排队

不必等待现有实验结束才能上传，也不用先执行下面的共享源码同步。Mac执行一次：

```bash
cd /Users/wq/Desktop/D-FINE
bash tools/diagnostics/queue_scbs_gqc_only_110.sh run1
```

看到“服务器排队提交成功”之后即可关闭Mac。之前若仍在Mac运行`sync_scbs_gqc_only.sh --wait`且停留在等待阶段，可先Ctrl+C取消那个等待；不会结束服务器训练。

这个入口只把白名单文件传到服务器`/home/a5/MyProject/wq_project/D-FINE-pending/GQC_only_seed0_run1/payload`，不覆盖正在使用的D-FINE。随后由服务器tmux后台：

1. 在该任务的`workspace`中创建独立源码/配置副本，叠加这次GQC修改；不复制`.git`、旧输出或大体积数据/权重。
2. 数据与预训练权重通过原路径/软链接复用；原项目源码不修改。
3. 等原D-FINE训练/评估/队列进程整体结束，而非在五组实验之间GPU短暂空闲时插队。
4. 再等GPU无计算进程后启动GQC单模块，训练/评估/打包全部自动完成，默认不做ON/OFF。

队列状态、日志位于待运行目录，不在原D-FINE项目中。服务器查看：

```bash
tmux attach -t scbs_gqc_only_seed0_run1
```

或：

```bash
tail -f /home/a5/MyProject/wq_project/D-FINE-pending/GQC_only_seed0_run1/queue.log
```

`queue_status.json`记录`waiting_project`、`waiting_gpu`、`running`、`complete`或`failed`等状态。完成后tmux会话可能退出，但文件保留。正式结果仍在原移动盘输出根目录，名称与手动启动相同。

演练提交加`--dry-run`，不联网、不创建任务。可用`--seed 1`、`--gpu 0`、`--port 7785`、`--root /绝对路径`、`--environment wq`、`--diagnostics`覆盖相应设置。旧待运行目录、同名tmux会话或已有输出均拒绝覆盖；改用未使用的run编号，run编号不改变seed。
提交时会在服务器查找并记录Conda完整路径，不依赖后台shell是否初始化Conda；必要时用`--conda-exe /home/a5/MySoftware/Anaconda3/bin/conda`显式指定服务器路径（不要填Mac路径）。

这是服务器tmux中的协作排队，不是Slurm：新GQC任务通过文件锁互斥，不承诺全服务器任务严格FIFO，也无法阻止其他用户手动抢卡。服务器重启会中断，不能自动恢复。独立副本会保留供复现，不自动删除。

下面是原有手动同步/启动方式；已使用排队提交时**不要再重复执行**。

## Mac先同步

```bash
cd /Users/wq/Desktop/D-FINE
bash tools/diagnostics/sync_scbs_gqc_only.sh
```

不依赖此前缺失的`night_n4.yml`，也不要求先同步五组BoxLocal实验。只需服务器现有原始项目依赖。若当前D-FINE任务仍使用共享源码，先等其结束，或给同步命令加`--wait`自动等待；不会结束任何任务。被替换源码有备份。

看到“启动脚本已核对”和“GQC单模块同步完成”后，再到服务器运行。Mac本地演练加`--dry-run`，不连接服务器。

## 服务器tmux内运行

```bash
cd /home/a5/MyProject/wq_project/D-FINE
bash tools/diagnostics/run_scbs_gqc_only_110.sh run1 --wait-gpu
```

默认Conda环境wq、seed0、GPU0、端口7785。`--wait-gpu`等其他计算进程退出后启动，不会停止任务，也不是多用户GPU预订锁。若需要断开Mac，在服务器tmux内运行即可。

只演练命令：

```bash
bash tools/diagnostics/run_scbs_gqc_only_110.sh run1 --dry-run
```

默认不跑合成测试、ON/OFF、完整query导出。训练完成自动检查0..109轮日志，评估best_stg2的EMA，导出三类AP和混淆矩阵；缺少best_stg2时明确记录best_stg1回退。完整query分析仅在额外加`--diagnostics`时执行。

若旧输出存在，使用新run编号；不自动覆盖/续训。`run2`只变输出编号，不改变seed；不同种子要加`--seed 1`。输出根目录可用`--root /绝对路径`覆盖；环境可用`SCBS_GQC_ENV`覆盖。

## 结束后取哪些文件

默认输出：`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_GQC_only_seed0_run1`。

- `GQC_only_diagnostics.zip`：可直接下载分析，含配置/源码和标注SHA256、训练日志、最终指标、混淆矩阵和汇总，不含权重或原始query张量。
- `summary.md`、`summary.json`：本次结果、三类AP、conf0.5/IoU0.5微平均P/R/F1、read↔write误分和相对历史基线变化。
- `diagnostics_eval/evaluation/validation_metrics.json`及`confusion_matrix/`：实际最佳EMA评估。
- `best_stg1.pth`、`best_stg2.pth`、`last.pth`和`log.txt`：服务器保留，用于复验和恢复。
- `experiment_inputs.json`：解析配置、源码/数据SHA256、实际命令与seed；`status.json`显示训练/评估/未完成状态。

历史参考（不是本次重跑）：原D-FINE AP55.0469%/write47.6115%，AQS AP55.8731%/write49.7422%，AQS+GQC AP56.2120%/write48.9361%。先核对数据与协议，再判断GQC单模块相对原基线是否提高，并检查hand/read/write是否协同改善；有信号后用多seed复验。

## 实现检查

```bash
python tools/diagnostics/test_gqc_only.py
python tools/diagnostics/test_gqc_only_queue.py
```

实现检查与合成数据评估不代表正式增益；本机无CUDA，真实CUDA FP16检查需在服务器环境执行。本次修改不自动同步或启动服务器训练。
