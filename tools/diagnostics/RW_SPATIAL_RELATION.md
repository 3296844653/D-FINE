# SCB-S read/write Spatial Relation 分支：独立 Colab 结构实验

## 依据与边界

原始 SCB-S batch32 基线的混淆矩阵中，write→read 为 245，read→write 为 154。
Pairwise CE 一次完整实验分别变成 221 和 155：有部分收益，但两个方向没有同时改善。
这支持继续研究 read/write 判别，不证明特征损失必然发生在某一层。

本分支检验的假设：最终 query 使用框内局部空间关系，是否比只使用整体表征更有利于 read/write 判别。
不是已经证明有效的模块，也不宣称学术新颖性；需通过实验和文献检索另行确认。

## 插入位置与原理

代码：`src/zoo/dfine/dfine_decoder.py` 中 `RWSpatialRelationRefiner`。
在最后一层 decoder、普通 query 与 DN query 分离之后，对最终分类 logits 加残差；
位于 postprocessor/sigmoid/top-k 之前，训练与评估采用同一分支。

从 encoder 的 P3 经 decoder input projection 后的特征 F 采样，不使用 GT 框。
每个预测框 b 内取 3×3 的九个点，并将相对二维坐标作为位置编码加入 token。
设最终 query 为 q，采样点相对位置为 r_i：

```text
t_i = BilinearSample(Conv1x1(F), stopgrad(b), r_i) + Linear(r_i)
T   = [t_1, ..., t_9]
R   = LayerNorm(T + SelfAttention(LayerNorm(T)))
u   = Linear(LayerNorm(q))
e   = CrossAttention(u, R, R)
d   = MLP(concat(u, e))
z'_hand  = z_hand
z'_read  = z_read  + d
z'_write = z_write - d
```

位置编码让九个局部 token 在关系建模之前不被平均掉；self-attention 交换局部信息，
query-conditioned cross-attention 再选择与当前目标有关的证据。
不同于已有 DecoderLocalQueryClassificationRefiner 的“采样→空间平均→拼接 MLP”。
九个位置不是已标定的手、笔或书本，不能保证对应某个身体部位，也不能恢复图像里不存在的细节。

最后一层 MLP 权重/偏置为零；在相同原权重下，初始输出与 baseline 相同。
新增参数 75,713。bbox 坐标采样前 detach，没有新增 bbox 回归分支；
但 query 和 encoder 特征不 detach，分类梯度会影响共享网络，所以不能保证训练后的 bbox 不变。
对类别0直接 logit 修正为0，不代表训练后举手 AP 必然不变。
read/write 两个 logit 的和保持不变，差增加 2d；原 sigmoid 后处理保留，置信度仍可能变化。

## 单变量配置

`configs/dfine/dfine_s_scb3s_3cls_bs32_rw_spatial_relation.yml`
继承原始 `dfine_s_scb3s_3cls_bs32.yml`：132 epochs、batch32、原优化器/增强/评估。
类别 0=hand-raising、1=read、2=write；数据路径继续使用 `dataset/SCB3-S/`。
只开启本分支，关闭原有实验模块、Pairwise CE、Margin 和 class-balanced VFL。
原 VFL、FDR、GO-LSD 损失保留；辅助层使用未经过新分支的 teacher logits，DN 不参与该分支。
原 matcher 算法不改，但最终 logits 改变后匹配结果可能改变，这是该结构的训练效应。

## 本地提交与 Colab

在桌面 `/Users/wq/Desktop/D-FINE` 提交推送这些文件，随后照原 ipynb 流程同步最新 Git、
数据、依赖到 `/content/D-FINE`。不另建 Colab 专用数据配置。

```bash
cd /Users/wq/Desktop/D-FINE
git add src/zoo/dfine/dfine_decoder.py \
  configs/dfine/dfine_s_scb3s_3cls_bs32_rw_spatial_relation.yml \
  tools/diagnostics/test_rw_spatial_relation.py \
  tools/diagnostics/RW_SPATIAL_RELATION.md
git commit -m "Add independent SCB-S read-write spatial relation branch"
git push
```

下列单元格代替原训练单元格；先完成原来的挂载网盘、同步项目/数据、安装依赖步骤。
这是完整训练，不使用 -t/-r，不是短训练或从 baseline 最佳权重微调。
运行名已存在时不要覆盖旧实验，修改 run1 为新的运行编号。

```python
%cd /content/D-FINE
from pathlib import Path
import json
import os
import subprocess

RUN_NAME = "dfine_s_scb3s_3cls_bs32_rw_spatial_relation_run1"
CONFIG = "configs/dfine/dfine_s_scb3s_3cls_bs32_rw_spatial_relation.yml"
local = Path("/content/D-FINE/output") / RUN_NAME
cloud = Path("/content/drive/MyDrive/D-FINE_outputs") / RUN_NAME

assert Path("/content/drive/MyDrive").is_dir(), "请先挂载谷歌网盘"
assert Path("train.py").is_file(), "请先按原流程同步完整项目到 /content/D-FINE"
assert Path(CONFIG).is_file(), "新配置尚未同步，请检查 Git 分支与推送"
assert not local.exists() and not cloud.exists(), "运行目录已存在，请使用新的 RUN_NAME"
for split in ("train", "val"):
    root = Path("dataset/SCB3-S")
    ann = root / "annotations" / f"instances_{split}.json"
    assert ann.is_file(), f"数据没有同步：{ann}"
    content = json.loads(ann.read_text())
    names = {c["id"]: c["name"] for c in content["categories"]}
    assert names == {0: "hand-raising", 1: "read", 2: "write"}, names
    missing = [im["file_name"] for im in content["images"]
               if not (root / "images" / split / im["file_name"]).is_file()]
    assert not missing, f"{split} 缺少图片，例如 {missing[:3]}"
    print(split, len(content["images"]), "images", names)

env = dict(os.environ, MPLBACKEND="Agg", CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
command = [
    "torchrun", "--master_port=7777", "--nproc_per_node=1", "train.py",
    "-c", CONFIG, "--output-dir", str(local), "--use-amp", "--seed=0",
]
print("正在完整训练：", RUN_NAME, flush=True)
try:
    subprocess.run(command, env=env, check=True)
    assert (local / "best_stg2.pth").is_file(), "训练结束但没有 best_stg2，请检查日志"
finally:
    if local.is_dir():
        cloud.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-av", str(local) + "/", str(cloud) + "/"], check=True)
        print("已有输出已备份至：", cloud)
print("训练成功完成。")
```

完成后，保持运行时连接，可用以下单元格直接导出当前最佳权重的混淆矩阵，不重新训练：

```python
evaluation = local / "confusion_eval"
try:
    subprocess.run([
        "torchrun", "--master_port=7777", "--nproc_per_node=1", "train.py",
        "-c", CONFIG, "-r", str(local / "best_stg2.pth"), "--test-only",
        "--output-dir", str(evaluation), "--use-amp", "--seed=0",
        "-u", "export_confusion_matrix=true",
    ], env=env, check=True)
    assert (evaluation / "confusion_matrix" / "confusion_matrix.json").is_file()
finally:
    if local.is_dir():
        cloud.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-av", str(local) + "/", str(cloud) + "/"], check=True)
print("混淆矩阵已保存：", cloud / "confusion_eval/confusion_matrix/confusion_matrix.json")
```

## 判断是否值得继续

对比同一 Colab baseline：AP、read/write AP、read→write 和 write→read 数量、
两类 Precision/Recall，并检查 hand-raising AP、AP75 和总 FP/FN 是否恶化。
混淆矩阵统一 score=0.5、IoU=0.5；低 IoU 未匹配预测不直接等同真正背景。
混淆减少但 read/write Recall 同时下降，可能只是少输出了预测，不能直接当作解决混淆。
一次 Colab 收益只做筛选，4090 再复验与调参，不宣称稳定提升。

验证命令：`python tools/diagnostics/test_rw_spatial_relation.py`。
本地 CPU 检查覆盖零残差输出一致性、参数覆盖、训练/DN/辅助路径、空 GT 图像、反向传播、
BF16 autocast、部署模式形状与分支隔离。尚未进行 T4 CUDA FP16 或完整训练测试，
部署兼容性检查不等同部署前后检测指标完全一致。
