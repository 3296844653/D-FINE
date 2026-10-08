# SCB-S / D-FINE-S：旧 AQS 的独立迁移实验

## 目的与边界

旧 SCB-U-S 日志中，AQS 相对 run8 的 reading / writing AP 有正向信号，
但单次结果不能证明稳定有效。本次是在 SCB-S 复验这套机制，不是承诺
提升、复用旧数据集权重，或宣称新增论文创新。

恢复来源：本仓库 Git `26b851d` 中的 `AdaptiveQuerySelectionRefiner`。
保留原核心公式、零初始化的最后投影、参数默认值、DN/检测 query 分组
和未增强分类分数作为普通 query 定位蒸馏 teacher 的处理。不恢复该提交
中的其他实验代码。本次三类顺序：0=hand-raising、1=read、2=write。

## 插入位置与公式

Encoder → 原 Decoder 最后一层 Q → 原分类头 + LQE 得到 z → **AQS** →
同一个分类头 + 同一个 LQE → 原后处理。

对每个 query，定义：

\[
s_i=\max_c\sigma(z_{ic}),\quad \tau=\sigma(\theta),\quad
u_i=\sigma((s_i-\tau)/T),\quad h_i=\mathbf{1}[u_i\geq0.5].
\]

前向采用硬门控，反向采用 straight-through 梯度：

\[
g_i=\operatorname{stopgrad}(h_i-u_i)+u_i,\qquad
C=\frac{\sum_i g_i Q_i}{\max(1,\sum_i g_i)}.
\]

\[
\widetilde Q_i=Q_i+\tanh(\rho)g_i
\operatorname{MLP}([\operatorname{LN}(Q_i);C]),\qquad
\widetilde z_i=\operatorname{Classifier}(\widetilde Q_i)+\operatorname{LQE}.
\]

- tau 初始 0.50，可学习；T=0.10 固定；rho 初始 0.05，实际残差系数为
  tanh(0.05)，不是直接使用 0.05。不要与诊断时固定的置信度阈值混淆。
- MLP 最后一层权重/偏置初始化为零；同权重起点输出与基线相同。
  初始梯度先到最后投影，其他分支参数随后获得梯度，这是原实现设计。
- 三个类别都参与，不把旧五类的 read/write 编号硬编码到三类实验中。
- 不增加 query 数、P2 特征、新损失或新后处理，也不叠加 MFFE。
- 同一次前向中回归仍使用原 Q，AQS 不直接修改 bbox/FDR；但分类梯度和
  匹配会影响共享模型的训练，不能声称训练后的定位绝对不变。
- DN query 与普通检测 query 分开聚合上下文，避免 GT-derived DN 信息
  进入普通 query。只有最后一层分类增强；pre、encoder、早期 aux 保留。
- 普通 query 的 GO-LSD 使用未增强的最终分类 teacher；DN loss 延续
  旧实现的独立分组与最终 DN 分类输出。原有损失代码和权重不改。
- 新增 197,634 参数。当前三类 S 模型日志/deploy 总参数 10,375,835。
- 上述门控优先选择高置信度 query，不意味着会直接挽救所有低分候选。

## 配置

训练：`configs/dfine/dfine_s_scbs_hrw_aqs.yml`，仅继承当前服务器基线并
增加 AQS 四项参数。当前与新基线统一为 110 epochs、epoch 100 停增强；
三类、数据路径、batch=32、多尺度、优化器和学习率计划保持一致。
旧 72/132 轮结果不能直接作为本次 110 轮实验的对照；使用新输出目录，建议名称带 `epochs110`。

诊断：`configs/dfine/dfine_s_scbs_hrw_aqs_diagnostics.yml`，使用本实验
自己训练的最佳权重，导出混淆矩阵及完整 query 诊断。

## 同步（Mac 本地终端）

```bash
cd /Users/wq/Desktop/D-FINE
rsync -avR \
  configs/dfine/dfine_s_scbs_hrw.yml \
  src/zoo/dfine/dfine_decoder.py \
  configs/dfine/dfine_s_scbs_hrw_aqs.yml \
  configs/dfine/dfine_s_scbs_hrw_aqs_diagnostics.yml \
  tools/diagnostics/test_aqs_refine.py \
  tools/diagnostics/SCBS_AQS.md \
  a5@100.76.151.67:/home/a5/MyProject/wq_project/D-FINE/
```

这要求服务器已有当前基线和已验证的诊断工具。同步不使用 --delete，
不删除数据或旧实验结果；不会自动推送 Git 或启动服务器训练。

## 检查与训练（服务器终端）

CPU 小规模检查，无数据、无预训练下载、不进行完整训练：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
conda run --no-capture-output -n wq \
python tools/diagnostics/test_aqs_refine.py
```

正式训练，从头训练，不加 -r / -t，不覆盖旧输出：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n wq \
torchrun \
  --master_port=7783 \
  --nproc_per_node=1 \
  train.py \
  -c configs/dfine/dfine_s_scbs_hrw_aqs.yml \
  --output-dir "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_aqs_seed0_run1" \
  --use-amp \
  --seed=0
```

若输出目录已有结果，换新 run 编号。AQS 用轻量 query MLP，没有额外
高分辨率激活分支，但 CPU 检查不是 4090 batch=32 显存保证。

## 训练后诊断

```bash
cd /home/a5/MyProject/wq_project/D-FINE
AQS_DIR="/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_aqs_seed0_run1"
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n wq \
torchrun \
  --master_port=7783 \
  --nproc_per_node=1 \
  train.py \
  -c configs/dfine/dfine_s_scbs_hrw_aqs_diagnostics.yml \
  -r "$AQS_DIR/best_stg2.pth" \
  --test-only \
  --output-dir "$AQS_DIR/diagnostics_eval" \
  --use-amp \
  --seed=0
```

成功导出后在同一终端执行：

```bash
conda run --no-capture-output -n wq \
python tools/diagnostics/analyze_detection_diagnostics.py \
  --input-dir "$AQS_DIR/diagnostics_eval"
```

优先对照 writing AP、reading AP、write→read / read→write 和 writing
Recall，辅以类别×尺度及全类别误检/漏检计数。单次只有总 AP 微涨不能
确认为有效；若目标错误改善，再做独立复验。未匹配预测不等于背景。

## 本地验证范围

7 项 CPU 检查覆盖旧公式、硬选择与零初始化、STE 梯度、CPU bfloat16
训练/推理、混合/全空 GT、DN 隔离、配置只改变 AQS、原参数初始化、
teacher/aux/pre/encoder 路径、原损失反向、优化器、严格权重加载及部署。
额外直接读取 Git 26b851d 的旧 AQS 类，核对初始化、前向和梯度在 FP32
及 CPU bfloat16 下逐元素一致；并与修改前的实际 Decoder 源文件核对，
AQS 关闭时训练/推理输出及参数初始化逐元素一致。

未执行完整训练或真实 CUDA/4090 batch=32 检查；上述验证不保证 AP
提升，也不保证每个训练批次的显存峰值。真实训练仅在服务器进行。
