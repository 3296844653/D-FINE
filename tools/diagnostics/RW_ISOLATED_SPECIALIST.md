# SCB-S 隔离训练的 read/write 空间关系专家

## 当前证据与本次假设

固定 confidence=0.5、IoU=0.5、EMA 和相同验证集：

| 指标 | Baseline | Pairwise CE | Spatial Relation |
|---|---:|---:|---:|
| read→write | 154 | 155 | 144 |
| write→read | 245 | 221 | 222 |
| 两类混淆合计 | 399 | 376 | 366 |
| 未匹配预测 | 2484 | 2365 | 2477 |

Spatial Relation 的训练日志最佳总 AP 为55.58，低于 baseline55.74；
read/write AP 分别+0.49/+0.10，但举手 AP -1.08。
这是单次实验，不能证明瓶颈位置或认定举手下降一定由共享梯度造成。

本次不是叠加原全网络 Pairwise CE，也不是重新换一种 attention。
保留有局部正向响应的空间关系分支，改为独立训练的二分类专家，
验证“保持主检测训练与检测分数不变，只纠正 read/write 类别选择”是否更合适。
这是项目实验设计，不保证提升，也不宣称已具备论文创新性。
本次不直接解决大量未匹配候选、重复框或定位错误。

## 与旧模块的区别

| 项目 | 原 Spatial Relation | 本次 RW Isolated Specialist |
|---|---|---|
| 输入 P3/query | 允许梯度回传 | detach，只训练专家自身 |
| 主训练 pred_logits | 加 read/write 残差 | 保留原始输出 |
| Matcher / VFL / GO-LSD / DN | 最终主 logits 会改变 | 输入保持原路径 |
| 专家监督 | 通过主 VFL | 原 Hungarian 匹配中 read/write、IoU≥0.5 的条件 CE |
| 推理 | 直接加减 logit | 判断相反时交换 read/write 的原分数 |
| 梯度裁剪 | 主网络和新增分支一起裁剪 | 分组裁剪，避免专家改变主网络裁剪系数 |

位置仍在最后 decoder 层、DN/普通 query 分离之后；
使用 encoder P3 经 decoder input projection 后的特征，九个框内 token 保留坐标，
局部 self-attention + query-conditioned cross-attention 提供证据。
框、query、P3 和基线 logits 全部 detach 后送入专家；推理不读取 GT。

设原 read/write logits 为 z_r,z_w，专家输出残差 d：

```text
g_0 = stopgrad(z_r - z_w)
d   = SpatialRelation(stopgrad(F_P3), stopgrad(b), stopgrad(q))
g   = g_0 + 2d
u   = [g/2, -g/2]

L_expert = sum CE(u_i, pair_label_i) / N_GT
           i 属于原最终层 Hungarian 匹配，GT为read/write，且IoU≥0.5
L_total = L_original_DFINE + 0.1 L_expert
```

这里的 IoU 是当前匹配 query 的预测框与其 GT 框的 IoU，不重新挑选另一批 query；
GT只用于训练。早期没有合格正样本时，专家损失为图相连的0，原检测仍正常训练。
举手、未匹配、低IoU、辅助层和DN不参与专家 CE。
新增日志键 `train_loss_rw_specialist` 是已乘0.1的损失；没有额外 `loss_pairwise_ce`。

推理规则：g与g_0符号相反时交换(z_r,z_w)，否则保留；任一判断平局时不交换。
仍用原 sigmoid 和 postprocessor，不拿专家 softmax 概率替代检测置信度。
每个query的全部分数数值集合、最大分数、举手logit和bbox均保持不变，只重分配read/write标签。

这是训练/推理任务不同：主检测训练保持原输出，专家独立学条件判别，验证/推理应用标签交换。
条件 CE 不以精度校准为目标，所以选择保留原检测分数。
剩余风险包括专家错误交换、背景候选也可能更换read/write类别、相同分数的top-k边界、
AMP溢出导致共同跳步、EMA/最佳epoch选择不同、GPU随机性和阶段2恢复点差异。
因此不承诺举手 AP 或主训练轨迹一定与旧基线数值完全相同。

## 文件与配置

- `src/zoo/dfine/dfine_decoder.py`：`RWIsolatedSpecialist` 和普通query接入。
- `src/zoo/dfine/dfine_criterion.py`：独立专家监督，原损失函数不替换。
- `src/solver/det_engine.py`：只对启用该实验的模型分组裁剪；其它实验沿用原全局裁剪。
- `configs/dfine/dfine_s_scb3s_3cls_bs32_rw_isolated_specialist.yml`：仅当前实验参数。

新配置直接继承 baseline 的 `dfine_s_scb3s_3cls_bs32.yml`，
不是继承 Spatial Relation 或 Pairwise CE 实验配置；其余模块默认关闭。
仍为原 SCB3-S、类别0=hand-raising/1=read/2=write、132 epochs、batch32、原增强/优化器/EMA。
新增参数75,713，旧实验模块与配置保留，不更改标注。

## Git → Colab：保持原流程

桌面终端提交：

```bash
cd /Users/wq/Desktop/D-FINE
git add src/zoo/dfine/dfine_decoder.py src/zoo/dfine/dfine_criterion.py \
  src/solver/det_engine.py \
  configs/dfine/dfine_s_scb3s_3cls_bs32_rw_isolated_specialist.yml \
  tools/diagnostics/test_rw_isolated_specialist.py \
  tools/diagnostics/RW_ISOLATED_SPECIALIST.md
git commit -m "Add isolated score-preserving read-write specialist"
git push
```

原 ipynb 中先挂载云盘、同步最新 Git 和完整数据、安装原环境。
不需要另建 Colab 配置；数据仍用 `dataset/SCB3-S/`。
不能只重新执行训练格而仍使用上一轮缓存源码。

下面替换原训练单元格，完整训练，不加 -t/-r：

```python
%cd /content/D-FINE
from pathlib import Path
import os
import subprocess

RUN_NAME = "dfine_s_scb3s_3cls_bs32_rw_isolated_specialist_run1"
CONFIG = "configs/dfine/dfine_s_scb3s_3cls_bs32_rw_isolated_specialist.yml"
local = Path("/content/D-FINE/output") / RUN_NAME
cloud = Path("/content/drive/MyDrive/D-FINE_outputs") / RUN_NAME

assert os.path.ismount("/content/drive"), "请先挂载云盘，不要仅创建 drive 目录"
assert Path("train.py").is_file() and Path(CONFIG).is_file(), "请先同步最新项目"
assert "class RWIsolatedSpecialist" in Path("src/zoo/dfine/dfine_decoder.py").read_text(), "仍是旧源码"
assert not local.exists() and not cloud.exists(), "不要覆盖旧结果，请使用新运行编号"
for split in ("train", "val"):
    assert Path(f"dataset/SCB3-S/annotations/instances_{split}.json").is_file(), "数据未同步"
    assert Path(f"dataset/SCB3-S/images/{split}").is_dir(), "图片未同步"

env = dict(os.environ, MPLBACKEND="Agg", CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
print("正在完整训练：", RUN_NAME, flush=True)
try:
    subprocess.run([
        "torchrun", "--master_port=7777", "--nproc_per_node=1", "train.py",
        "-c", CONFIG, "--output-dir", str(local), "--use-amp", "--seed=0",
    ], env=env, check=True)
    assert (local / "best_stg2.pth").is_file(), "请检查最佳权重保存情况"
finally:
    if local.is_dir():
        cloud.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-av", str(local) + "/", str(cloud) + "/"], check=True)
        print("输出已复制到挂载的云盘目录：", cloud)
print("训练成功完成。")
```

完成后运行下面的评估格，用同一份新权重分别关闭/开启专家推理，
这样直接比较类别交换本身的收益，不需要额外完整训练；同时和旧baseline比较整体结果。
`rw_specialist_apply_at_eval=false`只关闭推理应用，不删除专家结构，仍能严格加载同一权重。
导出同阈值混淆矩阵并直接打印，避免依赖网页找文件：

```python
try:
    for mode in ("off", "on"):
        evaluation = local / f"specialist_{mode}_eval"
        subprocess.run([
            "torchrun", "--master_port=7777", "--nproc_per_node=1", "train.py",
            "-c", CONFIG, "-r", str(local / "best_stg2.pth"), "--test-only",
            "--output-dir", str(evaluation), "--use-amp", "--seed=0",
            "-u", "export_confusion_matrix=true",
            f"DFINETransformer.rw_specialist_apply_at_eval={'true' if mode == 'on' else 'false'}",
        ], env=env, check=True)
        matrix = evaluation / "confusion_matrix/confusion_matrix.json"
        assert matrix.is_file()
        print("专家推理：", mode)
        print(matrix.read_text(encoding="utf-8"))
finally:
    if local.is_dir():
        cloud.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-av", str(local) + "/", str(cloud) + "/"], check=True)
```

## 判定与验证

目标：减少两方向混淆，同时提高read/write AP，并避免明显损害举手AP、AP75和Recall。
若混淆减少但AP仍不升，说明保留旧分数的类别修正也不足以获得整体收益，
需要检查排序/定位/未匹配候选，而不是认为问题已全部解决。
Colab先完整跑这一套，参数搜索和重复实验留待4090。

本地测试命令：`python tools/diagnostics/test_rw_isolated_specialist.py`。
覆盖输入梯度隔离、原主网络初始化/损失/梯度一致、合格样本筛选、空样本、DN/aux、
独立裁剪、原分数保留、参数优化器覆盖、CPU BF16 autocast与部署形状兼容。
另执行原 train_one_epoch 的 CPU FP32/BF16 单批次、EMA/optimizer 与 calflops 启动检查。
尚未执行T4 CUDA FP16及完整训练，CPU通过不等于T4精度或数值结果保证。
