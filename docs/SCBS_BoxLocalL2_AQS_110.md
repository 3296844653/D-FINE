# 默认 BoxLocal-L2 + AQS-L2 R1

仅修改桌面 `/Users/wq/Desktop/D-FINE`。现有模型源码已经支持这一组合，
本次显式固定参数并简化启动入口，不另造模块、不改变原单模块配置。
这与此前同配置的 AQS + BoxLocal-L2 是同一结构，并不是一种新变体；
已有完整结果可以直接使用，重跑不能算新的结构创新。

## 固定结构与协议

- 顺序：L2 已预测框 → BoxLocal-L2 → AQS-L2 → L3。
- BoxLocal：dim64、3×3 网格、offset0.15、固定 residual0.10。
- AQS：只在L2；threshold初值0.50、固定temperature0.10、raw residual初值0.10。
  AQS阈值和残差参数继续学习，不冻结。
- 110轮，100轮停止增强，batch32，seed0，原始损失/数据/Top300/DN。
- 使用原HGNet预训练进行从头联合训练；不恢复单模块checkpoint、不拼接两份权重。
  `run2`只改变输出目录，不等于seed1，也不等于继续上次训练。
- 不开启新的settings buffer，保持旧组合checkpoint的模型键兼容。

## Mac：只同步配置和工具

```bash
cd /Users/wq/Desktop/D-FINE
bash tools/diagnostics/sync_scbs_aqs_box_local_l2.sh
```

只传白名单6个文件，带备份且不删除文件。不会同步模型源码、数据、权重、
原始基线配置或旧输出，所以不要求其他实验停机。仅当同一个组合正在运行
时拒绝替换它的配置/脚本。服务器需已有BoxLocal扫描时期的模型组合接口；
若缺失会明确报错，不悄悄覆盖共享源码。`--dry-run`仅本地预览，不连接SSH。

## 服务器：默认一条命令只训练

在服务器的tmux会话内：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
bash tools/diagnostics/run_scbs_aqs_box_local_l2_110.sh run2
```

默认不运行整套合成测试、不做ON/OFF、不自动导出大型query诊断。
保留数据、配置、GPU和端口的必要检查。GPU忙时退出且不停止其他项目。
希望排队时在同一命令后加 `--wait-gpu`，GPU空闲即继续，无额外30秒确认。
希望训练后自动导出最佳EMA混淆矩阵及query诊断、离线分析时加
`--diagnostics`，仍不做ON/OFF。

若要新建tmux并一直保留窗口以查看报错，可在服务器执行：

```bash
tmux new-session -s scbs_Box_AQS \
  'cd /home/a5/MyProject/wq_project/D-FINE && bash tools/diagnostics/run_scbs_aqs_box_local_l2_110.sh run2 --wait-gpu; exec bash'
```

同名会话已存在就用 `tmux attach -t scbs_Box_AQS`，不删除旧会话。
Ctrl+B，松开，再按D脱离后可以关闭Mac；服务器重启仍会中断。

## 输出与保护

默认根目录：`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw`。
run2输出：`dfine_s_scbs_hrw_epochs110_AQS_BoxLocalQueryAdapter_l2_seed0_run2`。
目录已存在就拒绝启动，不自动续训、不覆盖、不自动删除；换一个未使用编号。
旧的绝对输出目录参数仍兼容。不同实验应使用不同输出目录。

保存 `console.log`、训练产生的 `log.txt`/权重，以及
`experiment_config.yml`、`experiment_inputs.json`（解析配置、相关源码和标注哈希）。
完整诊断只在 `--diagnostics` 时生成于 `diagnostics_eval/`。
默认GPU0、端口7781；可用 `SCBS_COMBO_GPU`、`SCBS_COMBO_PORT`、
`SCBS_COMBO_ROOT`、`SCBS_COMBO_ENV`、`SCBS_COMBO_CONDA`覆盖运行环境。
调换GPU/端口不改变模型；本入口固定seed0。

只预览训练命令，无需本机Conda/PyTorch，不占GPU、不创建输出：

```bash
bash tools/diagnostics/run_scbs_aqs_box_local_l2_110.sh run2 --dry-run
```

本地正确性检查：`python tools/diagnostics/test_aqs_box_local_l2.py`。
包括参数、Local→AQS→L3路由、原损失、DN隔离、梯度、空标注/AMP、
checkpoint/部署及输出保护；CUDA FP16测试仅在有CUDA的环境执行。
合成测试不证明精度。最终与原基线、仅BoxLocal-L2、仅AQS-L2分别比较。
