# AQS-L2 + BoxLocalQueryAdapter at query initialization

这是单模型同时启用两个既有模块的位置消融，不是把两者改成一个新模块。
原 Local-L2 + AQS-L2 配置、脚本和历史结果全部保留。

## 本次唯一的位置变化

同一 Encoder Top300 候选 -> BoxLocalQueryAdapter -> Decoder L1 -> L2 -> AQS-L2 -> L3。

- 原组合的局部模块在 L2 后；本次移到 Query 初始化，仅调整普通 query 内容。
- 使用已有 Encoder P3、已有预测框内九点 query 条件取样、已有局部注意力与 0.1 残差。
- 不更改 TopK 排序、候选数量、Encoder proposal 框计算，不使用 GT 框或验证集标签。
- 局部取样框坐标 detach；局部内容可沿 Encoder 特征反向传播。
- 在 DN 拼接之前执行局部初始化；普通 query 和 DN query 的 AQS 上下文仍分开。
- AQS 仍仅在 L2 后执行，阈值初值 0.50、残差参数初值 0.10、固定温度 0.10；阈值及残差参数继续学习。
- 沿用 110 epochs / 100 停增强 / batch32 / seed0，原损失和预训练方式不变。
- 两模块输出投影仍为零初始化，共享基线权重与基线随机数状态不受新增模块初始化影响。

“不改 proposal 框计算”不代表训练后框不变：初始化 query 变了，后续 L1-L3 的分类及回归都会可能变化；训练也会更新共享网络。
本实验检验两个位置能否互补，不能预先证明混淆减少或 AP 提升。

## 运行

先同步最新源码，再在服务器仓库根目录执行：

```bash
bash tools/diagnostics/run_scbs_AQS_l2_BoxLocalQueryAdapter_query_init_110.sh
```

仅检查配置/源码、不创建输出、不启动训练：

```bash
bash tools/diagnostics/run_scbs_AQS_l2_BoxLocalQueryAdapter_query_init_110.sh --dry-run
```

默认配置：`configs/dfine/dfine_s_scbs_hrw_110_AQS_l2_BoxLocalQueryAdapter_query_init.yml`。
默认结果：`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_AQS_l2_BoxLocalQueryAdapter_query_init_seed0_run1`。
目录存在时脚本拒绝覆盖；重复实验可提供一个含 `AQS_l2_BoxLocalQueryAdapter_query_init` 的新绝对输出目录。

脚本从头联合训练，实时输出并保存 `console.log`，完成后用本组合 `best_stg2.pth` 的 EMA 进行 test-only 诊断及离线分析。
`diagnostics_eval/evaluation` 保存混淆矩阵和 query 导出；`diagnostics_eval/analysis` 保存 COCO 类别×尺度与候选交叉分类统计。
训练或分析失败会返回非零状态，保留已产生结果，不谎报完成。

## 比较与判断

以相同验证标注、预处理、EMA、阈值与匹配规则，与 110 轮 Baseline Run2 比较；同时参考 AQS-L2 R1、单独 query-init 局部模块及旧 L2 串联组合。
核对整体 AP / APm / 三类 AP，read->write、write->read，write TP/FP、未匹配预测和正确高分候选缺口。
诊断计数不是 COCO AP/FP/FN，未匹配预测也不能直接叫背景误检或重复框。
这仍是一次新组合试验，单次提升不等于稳定有效。

## 合成回归测试

```bash
conda run --no-capture-output -n wq python tools/diagnostics/test_aqs_box_local_query_init.py
```

测试使用随机输入、禁用预训练下载，无需服务器数据，且不启动完整训练。
涵盖配置隔离、零初始化、模块执行顺序、Top300/proposal 与 DN 输入保留、梯度、空 GT、AMP、严格 checkpoint 加载、部署和脚本保护。
无 CUDA 时跳过 FP16 测试；合成测试只能证明实现可运行，不证明实际精度收益。
