# SCB-S / D-FINE-S：Encoder High-res Residual，110/100，alpha 初始 0.3

## 目的与位置

独立迁移已有 SCB-U Encoder High-resolution Residual probe。旧数据集的
读写 AP 正向信号尚未证明稳定有效；本次在当前 SCB-S 基线上复验，不
叠加 AQS、MFFE、Pairwise CE 或其他模块，不复用 SCB-U 训练权重。

现有 `src/zoo/dfine/hybrid_encoder.py` 已包含正确位置的实现，本次不
重写该实现，也不修改基线源码、基线配置或数据集。

这里的 high-res 是原生三尺度中最高分辨率的 **P3，stride=8**，不是
新增 stride=4 的 P2。实现先保存融合前的通道投影 P3：

\[
H_3=\operatorname{InputProj}_3(B_3).
\]

原 Encoder 完成 P5 AIFI、自顶向下 FPN 和自底向上 PAN 后得到：

\[
(E_3,E_4,E_5)=\operatorname{Encoder}_{baseline}(B_3,B_4,B_5).
\]

只在整个 FPN/PAN **之后**补回 P3：

\[
\widetilde E_3=E_3+\alpha H_3,\qquad
\widetilde E_4=E_4,\qquad \widetilde E_5=E_5.
\]

- alpha 是单个直接可学习标量，初始 0.3。没有 sigmoid/tanh 幅度限制，
  不采用 MFFE 的系数公式。
- 放在融合之后，测试原 P3 信息绕过融合后是否更有用。没有在 PAN
  之前改变中间特征，故同权重前向的 P4/P5 输出不变。
- 最终 P3 进入原 Decoder，所以会影响共享分类/定位；不能声称最终
  预测框不变或这是一个仅分类实验。
- P3 不是 COCO medium 标签；模块并不按 GT 尺寸或类别选择目标。
- 新增 1 个参数。当前三类 S 模型日志/deploy 参数由 10,178,201 变为
  10,178,202；四舍五入均为 10.18M。

## 两个新配置

训练：`configs/dfine/dfine_s_scbs_hrw_110_encoder_highres_residual_a03.yml`。
明确继承 `dfine_s_scbs_hrw_110.yml`，增加开启标志、alpha 初始值及独立输出目录。
服务器 SCB-S 路径、三类顺序（0=hand-raising、1=read、2=write）、
batch=32、多尺度、原损失和学习率保持一致；110 epochs、epoch 100 停增强。
旧 72/132 轮结果不能直接作为本次 110 轮实验的对照；新输出目录明确带 `epochs110`。
不增加其他 alpha 变体配置。

诊断：`configs/dfine/dfine_s_scbs_hrw_110_encoder_highres_residual_a03_diagnostics.yml`。
加载此实验自己的 `best_stg2.pth`，导出相同阈值的混淆矩阵及 query。
初始化值不覆盖 checkpoint 中已学习到的 alpha。

原来不带 `110` 的 High-res 配置保留不改；本次不要覆盖它的旧输出，也
不要使用 AQS 的权重续训。AQS 队列、C1 配置均不变，本组不属于 AQS 队列。

## 同步（Mac 本地终端）

```bash
cd /Users/wq/Desktop/D-FINE
rsync -avR --backup \
  --backup-dir="/home/a5/MyProject/wq_project/D-FINE-sync-backups/highres110_$(date +%Y%m%d_%H%M%S)" \
  configs/dfine/dfine_s_scbs_hrw_110.yml \
  src/zoo/dfine/hybrid_encoder.py \
  src/zoo/dfine/medium_finegrained.py \
  configs/dfine/dfine_s_scbs_hrw_110_encoder_highres_residual_a03.yml \
  configs/dfine/dfine_s_scbs_hrw_110_encoder_highres_residual_a03_diagnostics.yml \
  tools/diagnostics/test_encoder_highres_residual.py \
  tools/diagnostics/SCBS_ENCODER_HIGHRES_RESIDUAL.md \
  a5@100.76.151.67:/home/a5/MyProject/wq_project/D-FINE/
```

同步 MFFE 源文件只是满足现有 Encoder 的导入依赖，不在此实验开启。
服务器应已同步当前基线和诊断工具。不删除旧文件，不覆盖旧实验输出。

## 检查与训练（服务器终端）

小规模 CPU 检查，无真实数据/下载，不进行完整训练：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
conda run --no-capture-output -n wq \
python tools/diagnostics/test_encoder_highres_residual.py
```

从头训练，不使用 -r / -t，不加载基线或旧 SCB-U 权重：

```bash
bash <<'BASH'
set -euo pipefail
cd /home/a5/MyProject/wq_project/D-FINE
HIGHRES110_DIR="/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_encoder_highres_residual_a03_seed0_run1"
if [[ -e "$HIGHRES110_DIR" ]]; then
  printf '输出已存在，请使用新的运行编号：%s\n' "$HIGHRES110_DIR"
  exit 1
fi
mkdir "$HIGHRES110_DIR"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
conda run --no-capture-output -n wq \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py \
  -c configs/dfine/dfine_s_scbs_hrw_110_encoder_highres_residual_a03.yml \
  --output-dir "$HIGHRES110_DIR" --use-amp --seed=0 \
  2>&1 | tee "$HIGHRES110_DIR/console.log"
BASH
```

这是另一个独立训练，不要与 AQS 同时占用同一张 4090。先用 `nvidia-smi`
确认 GPU 空闲；自动通信端口不等于 GPU 独占。日志实时显示，并保存在
自己的 `console.log`。脚本不恢复/覆盖旧输出；启动失败后的目录也保留。

## 训练后诊断（服务器）

```bash
cd /home/a5/MyProject/wq_project/D-FINE
HIGHRES_DIR="/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_encoder_highres_residual_a03_seed0_run1"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
conda run --no-capture-output -n wq \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
  train.py \
  -c configs/dfine/dfine_s_scbs_hrw_110_encoder_highres_residual_a03_diagnostics.yml \
  -r "$HIGHRES_DIR/best_stg2.pth" \
  --test-only \
  --output-dir "$HIGHRES_DIR/diagnostics_eval" \
  --use-amp \
  --seed=0
```

成功后在同一终端生成分析报告：

```bash
conda run --no-capture-output -n wq \
python tools/diagnostics/analyze_detection_diagnostics.py \
  --input-dir "$HIGHRES_DIR/diagnostics_eval"
```

读取最佳权重中的 alpha（只读取自己训练的可信 checkpoint）：

```bash
conda run --no-capture-output -n wq \
python - "$HIGHRES_DIR/best_stg2.pth" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("last_epoch:", checkpoint.get("last_epoch"))
for source, state in (
    ("ema", checkpoint.get("ema", {}).get("module", {})),
    ("model", checkpoint.get("model", {})),
):
    matches = [(key, value) for key, value in state.items()
               if key.endswith("encoder.encoder_highres_alpha")]
    if not matches:
        print(source, "未找到 High-res Residual alpha")
    for key, value in matches:
        print(source, key, "=", value.item())
PY
```

EMA 是实际使用 EMA 评估时的对应系数。alpha 非零不等于模块有效。
这里读取的是最佳 checkpoint，不一定是最后一轮的 alpha。

## 判断是否有效

总体 AP 对照同设备、同设置的 SCB-S 110/100 基线（三次 AP 均值约
55.16 ± 0.11），不对比 132 轮五次均值；逐项错误对照相同 110 轮 run1。
优先看 writing AP、reading AP、write→read / read→write、writing Recall，
并补充中尺度类别 AP。只有总 AP 微涨但目标错误未改善，不算解决读写
问题。若单次有清晰正向信号，再做独立复验；单次不证明稳定有效，也
不直接证明信息丢失发生于哪一层。未匹配预测不等于真实背景。

本地检查范围包括准确的融合后残差公式、P4/P5 不变、零系数/关闭回归、
矩形特征尺寸、系数梯度/更新、配置单变量、原损失/DN/aux、CPU bfloat16、
空 GT、优化器、EMA、严格权重加载及部署。不执行完整训练或真实 CUDA
显存检查，不修改原模型实现来追求测试通过。

测试脚本覆盖新配置及旧配置兼容性；本地检查不代表 SCB-S 上的 AP、
混淆数量或 CUDA 显存已得到验证。效果需运行这组完整训练后确认。
