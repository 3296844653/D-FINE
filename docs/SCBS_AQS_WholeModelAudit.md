# AQS WholeModelAudit

诊断对象是整个 D-FINE-S + 单层 L2 AQS R1，110/100轮、batch32。
使用已有 best_stg2.pth 的 EMA 严格加载并冻结检测器。正常 eval 模式下
被动 hook 捕获输出；额外分类头读数在原 forward 完成之后计算。
仅独立的前景线性分类探针训练，检测器不会续训或改变参数。

## 操作

Mac 同步：

```bash
cd /Users/wq/Desktop/D-FINE
bash tools/diagnostics/sync_scbs_AQS_WholeModelAudit.sh
```

服务器运行（先等待GPU空闲）：

```bash
tmux new-session -s scbs_AQS_Audit \
  "cd /home/a5/MyProject/wq_project/D-FINE && bash tools/diagnostics/run_scbs_AQS_WholeModelAudit.sh audit1; exec bash"
```

默认根目录是已有 AQS R1 输出目录，新增 `AQS_WholeModelAudit_audit1/` 和
同名 `.zip`，旧结果保留。压缩包不包含模型、特征缓存或探针权重。
控制台日志写到同层 `AQS_WholeModelAudit_audit1_console.log`。
若权重路径改变，服务器命令前设置环境变量，例如
`export AQS_AUDIT_CHECKPOINT="/绝对路径/best_stg2.pth"`。

先演练可运行 `bash tools/diagnostics/run_scbs_AQS_WholeModelAudit.sh --dry-run audit1`。
希望排在当前GPU任务之后自动启动，可使用：

```bash
tmux new-session -d -s scbs_AQS_Audit_wait \
  "cd /home/a5/MyProject/wq_project/D-FINE && bash tools/diagnostics/run_scbs_AQS_WholeModelAudit.sh --wait-gpu audit1; exec bash"
```

后台tmux保留等待和诊断，SSH断开或本地Mac关闭后仍会继续。
查看时执行 `tmux attach -t scbs_AQS_Audit_wait`。
默认每30秒检查GPU计算进程，连续两次空闲后启动。桌面显示进程不计入
计算占用；未知进程名仍按占用处理。GPU查询失败会停止，避免误判空闲。
等待期间不会创建诊断输出；开始前再次检查编号，保留已有结果。
llama-server是常驻服务，即使请求处理完仍可能占显存；需要其使用者
结束服务，队列才会继续。无需等待的原命令仍保留遇占用即退出的行为。

只提取与单独训练探针分别支持 Python `--mode extract` / `--mode probe`。
已提取完整缓存后可运行同配置 `--mode probe`，然后 `--mode pack`；
已有探针结果拒绝覆盖。中断的部分缓存不能当完整结果使用。

## 流程与记录

1. Backbone P2/P3/P4/P5；Encoder输入投影P3/P4/P5；P5注意力之后；FPN两个融合块；
   Encoder最终P3/P4/P5。在同一GT框内ROIAlign 3×3并平均池化。
   `fpn_p4_block` 是首个FPN融合块的输出，之后还经过侧向卷积；
   `fpn_p3_block` 是第二个FPN块的输出，不等同于所有Encoder内部位置。
2. Encoder全网格冻结回归头估计 → 真正Top300输入框。
   全网格估计属于反事实诊断，正常推理只回归选中的网格。
3. 同一300个query的初始化、L1分类特征、L2 AQS前后、L3分类特征。
   逐GT统计各阶段IoU=.5/.75/.9几何覆盖、正确top1、正确top1高分，
   另保留不要求top1的GT类别高分覆盖（原后处理允许非top1类别）。
4. 相邻阶段统计任意候选覆盖的恢复/丢失。固定query追踪采用最终
   IoU-only Hungarian的一对一匹配；未匹配GT仅保留最佳IoU回退轨迹，
   不纳入固定query的条件分类转移统计。全网格阶段通过实际Top300
   网格ID恢复同一候选的位置，不使用query编号直接索引全网格。
5. L3原分类头 → 原LQE → 原query×class TopK后处理。
   LQE给同一query的各类别添加共同logit，不能改变它的类别argmax；
   但可改变跨query排序和阈值通过情况。
6. 最终完整TopK预测重算COCO指标；conf=.5只用于诊断矩阵/错误样本，
   不用于截断COCO AP输入。所有图像（包含无GT图像）均参与评估。
7. train/val均以相同640×640验证预处理提取冻结特征。以最终几何
   Hungarian匹配的共同GT子集训练/比较18种线性探针，每种seed0/1/2。
   标准化只用train；固定50轮；验证集不参与拟合或模型选择。

## 主要输出

| 文件 | 用途 |
|---|---|
| summary.txt / summary.json | 整合诊断及局限说明 |
| val/stage_coverage.csv | 各阶段×类别×尺度×IoU候选覆盖 |
| val/stage_transitions.csv | 改对/改错；候选覆盖恢复/丢失 |
| val/fixed_query_score_statistics.csv | 相同query的GT类别分数、类别差值、IoU与分数关系 |
| val/gt_stage_records.jsonl | 同一GT逐阶段框/分数差/类别/固定query轨迹 |
| feature_probes.csv / probe_summary.json | 18个位置的探针均值和标准差、参数量、类别召回 |
| val/validation_metrics.json | 原最终预测的独立COCO AP及分类别AP |
| val/coco_pr_curves.json | COCO各类别×尺度×IoU的插值PR曲线 |
| val/tide_status.json / tide_summary.txt | 若服务器已安装tidecv，自动执行官方TIDE默认AP50错误分析 |
| val/final_errors.json | 固定阈值矩阵、错误数量 |
| val/final_error_records.jsonl | 人工检查所需图片名、GT/预测框、分数与类别 |
| extraction_report.json / resolved_config.json | 完整性、权重来源/哈希、输入与协议 |

## 解释边界

阶段间分数校准不同，要结合类别logit差值、几何覆盖与无分数条件分类，
不能仅比较0.5通过数。Probe只测前景类别可读性；不同通道数意味着
不同线性头参数量，已明确报告。其条件召回不是检测Recall，不能直接
代替分类头的物体置信度。GT ROI是诊断用理想位置，推理不能使用GT。

“重复/竞争候选”“定位/范围候选”等是诊断候选类型；未匹配预测可能
对应标注遗漏或范围外目标。错误计数不等于AP损失。若已安装tidecv，
工具调用其官方API输出理想修正dAP（默认AP50，不等同于COCO AP50:95）。
未安装时明确记录unavailable，核心诊断继续完成；不会自动安装包。
需要这一可选统计时，可在运行前执行 `conda run -n wq pip install tidecv`。

整个模型的诊断用于定位可疑阶段，不能证明改动某层一定提升性能。
修改后还需相同训练协议与AQS基线作完整对照。

参考：[线性探针](https://arxiv.org/abs/1610.01644)、
[TIDE错误分析](https://dbolya.com/tide/)。
