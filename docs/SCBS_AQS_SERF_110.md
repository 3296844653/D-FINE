# AQS + SERF：Encoder / Backbone 两个独立位置

本地工程：`/Users/wq/Desktop/D-FINE`。服务器训练，不在 Mac 上训练。

目的：检验关系化的局部特征是否能改善 read↔write 混淆。并不预设
Encoder 是主要瓶颈，也不保证改动有收益或已经具备论文创新性。

## 两组位置

| 配置/模块 | 输入 | 更新位置 | 保留的路径 |
|---|---|---|---|
| SERF_Encoder | 完整 FPN/PAN 后 P3/P4，均256通道 | 更新 Encoder 返回的 P3 → Decoder | P4/P5 Encoder 返回值不改 |
| SERF_Backbone | HGNetv2-B0 阶段输出 P3/P4，256/512通道 | 更新返回的 P3 → Encoder | 骨干 P4/P5 计算及返回值不改 |

Backbone 版是**阶段输出适配器**，不是改写骨干内部阶段，也不是同时
增强 P3 与 P4。P4 提供上下文。Encoder 版在最终 PAN 后，区别于此前
SDRF 的 PAN 前语义差分注意力。两组均只增加一个 SERF，不能同时开启。

核心超参数相同，P4 输入通道不同，所以输入投影参数量不同；不是严格
等参数的位置消融。新增分支均放在 DFINE.serf，使用相同优化器默认组，
不因 Backbone 名称而使用骨干低学习率。原骨干参数及预训练加载不变。

## 结构

1. P3 与 P4 分别投影到64维，P4上采样对齐。
2. P3划分8×8窗口，附加2像素halo；仅真实像素参与权重归一化。
3. 根据局部外观和上下文学习4个证据槽，每槽池化得到外观、语义、
   加权空间中心。证据槽不是人工标注的手、笔、书本或人体部件。
4. 证据间进行4头交互：外观/语义关联 + 相对中心位置偏置，排除自连接。
   用「其他证据消息 − 当前证据值」构造关系信息。
5. 每个像素重新选择证据，重建空间细节；结合局部对比特征与P4语义，
   生成P3残差。仅写回窗口核心，halo不复制输出、不跨边界环绕。

这不是对P4简单门控P3，也不是新增多个卷积核的平均融合，亦不是
预测框ROI分类头。它利用关系信息检验「有候选框，但行为证据不够
有区分力」这一假设。计算覆盖所有位置/类别，不用GT框、不只提高write分数。

零初始化输出投影；残差有效初值0.10，由tanh限制范围并继续学习。
初始完整模型与AQS基线一致；第一步先训练输出投影，随后内部支路获得梯度。
只用GroupNorm/LayerNorm，不增加BatchNorm；采用非重入梯度检查点节省显存。
训练时FP32注意力归一化，兼容AMP。batch32分支显存检查不等于完整640检测器
的显存保证，正式训练若OOM应保留日志并再处理，不能悄悄改变batch或分辨率。

## 固定协议及判断

继承当前服务器 AQS-L2 R1：110轮，100轮停止增强，batch32，seed0，
原损失、原数据/增强/优化器，AQS初始化0.50 / 0.10 / 0.10；从头训练，
使用原HGNet预训练。不会从AQS最佳权重微调，不与GQC或其他结构叠加。
配置仅列本组启用项，不写其他关闭项。

主要比较独立训练的AQS R1，原Baseline Run2为辅助比较。
优先看整体AP、read/write AP和双方混淆率，辅以AP75、APm和类别×尺度AP、
正确高分候选缺口与PR。不能仅凭confidence=0.5时的混淆计数认定有效；
不能仅凭write上涨而忽略read/hand和总体AP下降。单次收益需后续复验。

训练结束自动导出最佳EMA的模块参数、AQS参数、ON/OFF诊断、混淆矩阵、
GT交叉分类、类别×尺度AP、PR与得分统计，以及诊断ZIP。参数范数不是
观察到的实际贡献；ON/OFF用同一联合训练权重，不能代替独立AQS对照。
两种位置的OFF都可能改变候选选择与bbox，而不是仅关最终分类分数。
可额外用eval_serf.py的no_geometry/no_relation测试组件响应，但不能冒充
重训消融。当前队列自动只执行ON和OFF。

## 先同步（Mac本地终端）

```bash
cd /Users/wq/Desktop/D-FINE
bash tools/diagnostics/sync_scbs_AQS_SERF.sh
```

只同步清单内的代码和两份新配置；不修改原基线配置、数据或权重。
服务器原AQS配置必须存在，数据路径按服务器的原配置继承。
如果D-FINE旧任务仍在运行，同步脚本会停止，避免中途改源码。
被替换代码备份在服务器工程同级的 D-FINE-AQS-SERF-backup-*。

## 启动（服务器SSH终端）

```bash
cd /home/a5/MyProject/wq_project/D-FINE
tmux new-session -d -s scbs_SERF \
  'cd /home/a5/MyProject/wq_project/D-FINE && bash tools/diagnostics/run_scbs_AQS_SERF_110.sh run1 --wait-gpu; exec bash'
tmux attach -t scbs_SERF
```

先 Encoder 后 Backbone，独立训练而非同时叠加。GPU被占用时等待；不会
杀死llama或其他项目。Ctrl+B，松开后按D，脱离tmux后可关闭Mac，服务器
仍继续。服务器关机/重启不能由tmux保护。已有同名tmux请先查看，不删除。

不训练的预览：

```bash
bash tools/diagnostics/run_scbs_AQS_SERF_110.sh run1 --dry-run
```

只跑一种：追加`--modules SERF_Encoder`或`--modules SERF_Backbone`。
输出目录已存在即停止，保留旧结果；新实验用run2等编号。

输出根：`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw`。

- dfine_s_scbs_hrw_epochs110_AQS_SERF_Encoder_seed0_run1
- dfine_s_scbs_hrw_epochs110_AQS_SERF_Backbone_seed0_run1
- AQS_SERF_two_epochs110_seed0_run1/status.json、results.csv（指标0–1）

每组上传`SERF_Encoder_diagnostics.zip`或`SERF_Backbone_diagnostics.zip`即可
进行主要分析；完整权重和raw_queries留服务器。若对照诊断目录缺失，
comparison_status.json明确记录，不会丢弃本次结果或伪造比较。可设置
SCBS_SERF_AQS_REFERENCE、SCBS_SERF_RUN2_REFERENCE为实际diagnostics_eval目录。

队列只接受完整0..109轮的日志和best_stg2.pth；中断结果不能冒充完整实验。
`--export-only`用于训练完整、尚未生成诊断的原目录，并核对原始源码哈希；
不覆盖已存在诊断。中途更改源码后不能使用此模式冒充原协议。

## 测试边界

服务器启动训练前运行 test_serf.py。包括奇数/小尺寸、padding/halo、
初始基线一致、反向梯度、CPU BF16、完整检测器DN/aux/空GT/EMA、
严格权重、两位置路由、640前向/部署/计算量工具、真实微型COCO导出和
队列文件保护。CUDA FP16与batch32分支测试只在有CUDA的服务器执行。
这些检查通过只说明代码可用，不说明模块有效。
