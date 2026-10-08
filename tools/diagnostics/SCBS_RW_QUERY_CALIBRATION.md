# SCB-S：110/100 读写候选校准独立实验

## 依据与边界

110轮Run1诊断中，684个writing GT全部有IoU≥0.5的Encoder Top300和最终Decoder框，但只有483个有分数≥0.5、被实际后处理选中的正确类别候选。其余201个中，179个有类别排第一但低分的候选，22个的所有重叠候选第一类别均错误。原Validator另记录write→read 231次、read→write 164次；这些统计不是同一种匹配规则，不能相减，也不能把22理解成全部类别混淆数量。

因此本次同时检验**读写相对偏好**和**正确类别的分数排序**，不增加query，不改定位。以上证据不能证明具体网络层导致了问题；阈值0.5的缺口也不等于COCO漏检，单纯跨过0.5不保证AP提高。

## 位置与结构

位置：最终Decoder输出（包含原LQE调整）→分离DN query→本模块→原postprocessor。

只处理普通300个query。对于第i个query，输入为：

- 冻结梯度的最终query表征q_i；
- 同一图像最多4个IoU≥0.5重叠邻居的加权表征；排除自己；
- 自身及邻居的原sigmoid类别分数；
- 四条边的回归分布归一化熵和最大概率（8维，质量线索而非真实IoU）。

令h_i=SiLU(Linear(LayerNorm(stopgrad(q_i))))，邻居权重为

    w_ij ∝ IoU(b_i,b_j) × max(sigmoid(z_j,read), sigmoid(z_j,write))

在满足条件的Top4邻居上归一化；无邻居时上下文为0。IoU和权重也停止梯度。拼接自身、邻域、分数与分布特征后，小MLP输出a_i、d_i：

    g_i = 0.5 × tanh(a_i)
    r_i = 1.5 × tanh(d_i)
    z'_i,read  = z_i,read  + g_i + r_i
    z'_i,write = z_i,write + g_i - r_i

g控制读写共同的分数变化，r控制两者之间的偏好。两个方向都可以升降，不是统一提升writing；hand-raising logit和bbox不变。最终线性层零初始化，初始严格等于基线。新增26,242参数。

区别：Pairwise CE只是增加旧分类头的损失；旧Objectness仅做共同行为评分；旧Isolated Specialist只交换已有读写分数。本模块使用局部候选关系和回归分布线索，同时改变读写分数与相对偏好。它是一个待验证的机制，不宣称已解决问题或已构成论文创新。

## 训练与保护

原final Hungarian一对一匹配、VFL、bbox/FGL、蒸馏、Encoder/辅助/DN输出不变。训练pred_logits仍为原结果，额外输出rw_calibration_logits供新分支监督；评估时使用经过校准的读写logit。

正样本仅为原final Hungarian匹配到read/write、且IoU≥0.5的query，正确类别目标为当前IoU、错误类别目标为0。不是把所有重叠候选都提分。

负样本候选为：未匹配且maxIoU≤0.3的query，或IoU≥0.5的其他类别匹配query；读写目标均为0。按原读写最高分选hard negatives，每图最多max(16,3×正样本数)。重复框/中间重叠/低质量匹配不参与这项附加监督。低IoU仅表示相对当前标注的非匹配候选，不证明其为真实背景。

令单个query损失为两个类别的BCE均值，则每图附加项为

    sum(正样本BCE) + max(正样本数,1) × mean(所选负样本BCE)
    L = L_original + 0.1 × sum(上述附加项) / 全局GT数

所有原模型输入停止梯度，只训练小校准分支；该分支单独裁剪梯度，防止改变原模型梯度裁剪系数。创建分支时保护原初始化随机数流。关闭开关时旧基线无新增参数。AMP scaler/优化器仍由训练器共同管理，不能据此保证完整重训后的原模型权重逐bit相同。

实验继承独立110轮/100停止增强基线，batch32、seed0，其余实验默认关闭且禁止叠加分类实验。不得与132轮均值直接比较。没有修改数据集或标注，没有把验证集GT送入模型，也没有按人工错误列表训练。

## 服务器：完整训练

先同步这次改动的Python文件、配置以及本说明/测试到服务器。必须包含新rw_query_calibration.py；只传YAML不够。

```bash
cd /home/a5/MyProject/wq_project/D-FINE
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
conda run --no-capture-output -n wq \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py -c configs/dfine/dfine_s_scbs_hrw_110_rw_query_calibration.yml \
--output-dir "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_rw_query_calibration_seed0_run1" \
--use-amp --seed=0
```

确认GPU空闲、输出目录尚不存在；新编号不要覆盖旧结果。这里从头按基线流程训练，不使用-t/-r。--local-addr避免服务器数字主机名5被解析成错误IP，--standalone避免固定7781端口冲突。

## 同一权重：开启/关闭校准的成对评估

用新实验自己的best_stg2.pth，保持其他设置相同。off仍构建模块并加载其权重，只在评估时跳过校准，不是加载旧基线权重：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
SCBS_CAL_ROOT="/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_rw_query_calibration_seed0_run1"
for SCBS_CAL_MODE in on off; do
  SCBS_CAL_APPLY=True
  if [ "$SCBS_CAL_MODE" = off ]; then SCBS_CAL_APPLY=False; fi
  test ! -e "$SCBS_CAL_ROOT/diagnostics_$SCBS_CAL_MODE" || exit 1
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
  conda run --no-capture-output -n wq \
  torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
  train.py -c configs/dfine/dfine_s_scbs_hrw_110_rw_query_calibration_diagnostics.yml \
  -r "$SCBS_CAL_ROOT/best_stg2.pth" --test-only \
  --output-dir "$SCBS_CAL_ROOT/diagnostics_$SCBS_CAL_MODE" --use-amp --seed=0 \
  -u "DFINETransformer.rw_query_calibration_apply_at_eval=$SCBS_CAL_APPLY" || exit 1
  conda run --no-capture-output -n wq \
  python tools/diagnostics/analyze_detection_diagnostics.py \
  --input-dir "$SCBS_CAL_ROOT/diagnostics_$SCBS_CAL_MODE" \
  --output-dir "$SCBS_CAL_ROOT/analysis_$SCBS_CAL_MODE" || exit 1
done
```

看COCO AP、read/write AP、PR、候选缺口是否一起改善，并检查混淆与FP是否增加。正确候选数量增加但AP下降，不算成功。hand logit不变不等于hand AP一定不变，因为全类别Top-k竞争可能变化。模型可能仍增加重复候选，本机制不是NMS或重复框消除。

先与110轮基线55.16±0.11比较；同权重on/off用于区分校准效果和重训波动。初跑正向再补重复实验。若AP不升且读写错误没有改善，停止该方向，不继续只凭0.5阈值提分。

## 本地测试（不训练完整实验）

```bash
python tools/diagnostics/test_rw_query_calibration.py
```

覆盖公式/零初始化、邻居与自排除、干净一对一监督/软IoU目标、忽略重复/灰区、空GT、输入梯度隔离、独立裁剪、完整S前后向与DN、优化器/EMA、checkpoint和CPU AMP。真实4090训练与AP提升需服务器实验确认。

本次本地验证：9项专项测试通过；以当前有效配置覆盖旧读写模块，相关回归选择集50项通过（排除3个仍依赖已改名旧配置的历史测试）；额外CPU Gloo DDP、find_unused_parameters=False连续两步前向/反向/裁剪/优化器更新通过。这些是合成小样本逻辑测试，不是SCB-S训练或AP实测。

### CUDA设备索引修复

首次服务器启动在`positives = source[clean & is_pair]`报设备不一致。原SciPy HungarianMatcher返回CPU索引，而GPU训练中的布尔筛选条件在CUDA；此前CPU检查没有覆盖这一混合设备路径。

修复只在新附加损失内部把source/gt_indices的局部副本及所需GT转到校准logits所在设备，不修改原match list，公式、采样阈值、参数和模块位置不变。测试新增真实Matcher的CPU索引配CUDA预测/GT，覆盖非空及空GT、FP32/FP16、损失值与CPU参考一致、原索引不被改动。CUDA不可用时明确跳过该测试；服务器可先运行`python tools/diagnostics/test_rw_query_calibration.py`验证。
