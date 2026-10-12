# SCB-S：六个位置的独立结构筛选

## 本轮目的与边界

这是六组**独立实验**，不是把六个模块叠在一个模型。每组使用原SCB-S数据、D-FINE-S、110轮、100轮停止增强、batch32、seed=0、EMA，保持原VFL/框/分布损失与匹配器。

现有正向信号：AQS-L2 R1相对110轮Baseline Run2整体AP约+0.83个百分点、write AP约+2.13，但write固定阈值正确匹配只增加4个，APm未改善。RW Local Detail对固定阈值write匹配有正向信号，但整体AP未提高。它们支持继续验证“局部判别特征/候选评分”假设，不证明唯一瓶颈所在。

本轮**不叠加AQS**，不加载AQS/基线训练结束权重，不改变损失函数、不增加query、不改类别映射，不改现有实验配置。都是从原HGNet预训练Backbone开始完整训练。只要没有`DFINE.night_probe`，原基线没有新增可学习参数。

## 六组改动与验证目标

| 组别 | 修改位置 | 结构操作 | 优先核对 |
|---|---|---|---|
| N1 | Backbone P3输出→Encoder | 3×3/5×5深度卷积＋局部高通残差 | write/read AP、APm；P4/P5的Backbone计算不变 |
| N2 | Encoder输出P3 | 用P4语义门控P3局部对比细节 | APm与中尺度write AP，不只整体AP |
| N3 | Encoder→Decoder query初始化 | 同一Top300候选从预测框内P3取局部信息 | 正确高分候选缺口是否缩小；不是新的TopK选择器 |
| N4 | Decoder第2层 | 在L2预测框内query条件取样，更新普通query后送入L3 | write AP、读写混淆、Recall、APm |
| N5 | 最终读写分类 | query自适应P2/P3取样，输出相反的read/write logit修正 | write→read与read→write是否同时减少、write AP |
| N6 | 最终分类分数（LQE后） | query＋P3局部外观＋框分布确定性→共同logit修正 | write AP、未匹配预测、PR与候选缺口，不能只看0.5计数 |

配置：`configs/dfine/dfine_s_scbs_hrw_110_night_n1.yml`至`..._n6.yml`。只写本组开关，N5额外返回P2；其余现有模块默认关闭。当前实现限定SCB-S的0=hand-raising、1=read、2=write与三层D-FINE-S，错误组合会拒绝运行。

## 主要公式

所有新增输出投影W_out、b_out均为0初始值，默认残差尺度固定0.1；不是再次搜索初始值。

N1：h=Proj(P3)，d=h−AvgPool3(h)，P3'=P3+0.1·W_out[h,DWConv3(h),DWConv5(h),d]。

N2：f=Proj(P3)，c=Resize(Proj(P4))，g=σ(Conv[f,c])，P3'=P3+0.1·W_out[g⊙DWConv3(f−AvgPool3(f)),c]。

N3/N4：相对取样位置r_j=clip(r_grid,j+0.15·tanh(Offset(LN(q))),−0.49,0.49)，图像归一化位置x_j=stopgrad(b_center)+stopgrad(b_wh)⊙r_j。

取样后，a_j=softmax_j(K(t_j)^T Q(q)/√64)，l=Σ_j a_j V(t_j)，q'=q+0.1·MLP[Q(q),l]。

N3在原Top300选完后、DN拼接前更新query，用encoder预测框而非GT。N4在L2框已计算后更新普通query并传入L3；L2本轮框不直接改变，L3框可能改变。DN不参与局部query更新，也不混入普通query的上下文。

N5：对P2/P3各取9点，d=0.5·tanh(MLP[Q(q),l])，z'_hand=z_hand，z'_read=z_read−d，z'_write=z_write+d。不是固定5×5格，也不是把同一残差同时抬高读写分数。

N6：u=[各边框bin概率峰值, 各边归一化熵]，d=0.5·tanh(MLP[Q(q),l,u])，z'_c=z_c+d，对同一query所有类使用相同修正，不直接改变类间排序或预测框。d由原VFL监督，**不能称为已校准的有效目标概率**。

所有框坐标/bin确定性在新增分类取样支路中detach；共享特征仍由原训练目标更新。因此“不直接改bbox计算”不意味着完整训练后的定位必然完全不变。N5/N6只改最终普通检测logits，DN、pre、aux预测与分类蒸馏teacher保持未修正版本。

## 服务器运行

本地桌面项目运行以下命令，同步明确白名单中的源码、六组配置和工具（保留服务器110轮基线配置/数据路径；rsync不delete，替换文件有服务器备份）：

```bash
cd /Users/wq/Desktop/D-FINE
bash tools/diagnostics/sync_scbs_night_probes.sh --dry-run
bash tools/diagnostics/sync_scbs_night_probes.sh
```

默认SSH目标读取当前已配置的服务器地址`a5@100.76.151.67`，若变更可设置`NIGHT_SSH_TARGET`。此脚本不启动训练。服务器原`dfine_s_scbs_hrw_110.yml`必须已存在且是110/100/batch32原损失配置；正式启动队列会核对。

同步后在服务器运行：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
bash tools/diagnostics/run_scbs_night_sweep_110.sh --dry-run
bash tools/diagnostics/run_scbs_night_sweep_110.sh
```

默认优先顺序N4→N6→N5→N2→N1→N3。一张4090逐组运行，不同时挤六组。每组110轮完整训练；若每组约2小时，六组约12小时**再加诊断开销**，不能保证一个晚上全部完成。只运行前三组：

```bash
bash tools/diagnostics/run_scbs_night_sweep_110.sh N4 N6 N5
```

若之后运行剩余组，队列记录目录不能覆盖，使用新的队列编号：

```bash
NIGHT_RUN_ID=2 bash tools/diagnostics/run_scbs_night_sweep_110.sh N2 N1 N3
```

如需离开SSH，建议在已有tmux会话中运行（脚本不自动安装/创建会话，不提交后台作业）。默认Conda环境wq、GPU0，torchrun使用standalone动态端口，不复用固定7781。

结果根目录：`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/`。

各组：`dfine_s_scbs_hrw_epochs110_night_n{1..6}_seed0_run1/`；含console.log、权重、experiment_inputs.json。训练成功后自动在`diagnostics_eval/evaluation/`导出本组best_stg2/EMA混淆矩阵和全部query，在`diagnostics_eval/analysis/`产生分析。结果汇总：`night_sweep_epochs110_seed0_run1/results.jsonl`。

训练/诊断失败会保留全部日志并记录失败状态，默认继续下一组；未成功训练不会打印“训练成功”，也不会导出其他权重。GPU仍被进程占用则停队列，绝不杀其他任务。可以`NIGHT_STOP_ON_ERROR=1`首错即停，`NIGHT_EXPORT=0`只训练。

## 验证与判定

运行前脚本检查所有配置、完整训练/验证图片和类别、GPU是否空闲，以及六组CUDA FP16小样本反向。小样本检查**不保证batch32所有多尺度输入的峰值显存**；若OOM，本组标失败，保留日志，不自动改batch（避免不公平比较）。

完整工程回归：

```bash
conda run --no-capture-output -n wq python tools/diagnostics/test_night_probes.py
```

看同一个checkpoint/EMA的整体AP、read/write AP、APm、两方向混淆、write P/R、真实候选缺口。几何覆盖与fixed-threshold counts不是COCO AP/FP/FN，未匹配预测不等于确定背景。

当前110轮基线三次均值AP=55.16±0.11；一次新实验超过该均值只是筛选信号，不证明模块稳定有效。选择目标错误确实下降且总体指标不受明显损伤的1–2组，后续再做复验/机制消融；不要把本轮六个筛选模块直接宣称为六个创新点。
