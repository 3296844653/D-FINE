# SCB-S：P2/P3 Query-conditioned Local Detail（独立结构实验）

## 为什么做

110轮 Baseline Run2 对比 AQS-L2-R1：write AP 47.61→49.74，但 write→read
仍为240，read→write为156。AQS主要改善预测可靠性，不能视为已经解决细粒度分类。
这些数字支持进一步测试局部分类信息，不证明 Decoder 是唯一根因，也不保证本模块有效。

这次只测试一个机制：最终query能否通过预测框内的高分辨率细节与语义信息，
形成更适合读写分类的特征。不改损失、不叠加AQS、不改变原Encoder的P3–P5路径。

## 位置与计算

HGNetV2-B0额外返回原有P2（stride=4，64通道），原P3–P5仍走原Encoder。
Encoder P3（stride=8，256通道）与P2分别投影到64维；对每个最终普通query的
预测框采样5×5点，共50个局部token。点带有框内相对坐标和尺度标识。
query作为4头attention的查询，从自己框内的token选择信息；不是全局query上下文池化。

令 b 是归一化cxcywh预测框，p=(u,v)是框内相对坐标：

    x = b_cx + u*b_w,  y = b_cy + v*b_h
    t_(scale,p) = sample(Project(F_scale), x, y) + Position(p) + Scale(scale)
    a_(head,p) = softmax_p(Q_head · K_(head,p) / sqrt(d_head))
    local = concat_heads(sum_p a_(head,p) * V_(head,p))
    delta_Q = 0.1 * MLP(concat(Project(Q), local))
    z_new = z_original + mask_read_write * W_cls * delta_Q

W_cls复用原最后一层分类器权重；z_original已含原分类bias和LQE修正，
不重新计算LQE、不重复添加bias。hand logit直接增量为0。
MLP最后一层零初始化，固定0.1缩放；不额外引入可学习alpha/门限或新loss。

注意：这是“分类特征残差”而不是直接制造一对正负相反的读写logit。
以前RW Spatial Relation只用P3、通过单标量做(+delta,-delta)；
P2 ROI probe主要均值池化。本次用P2+P3、query条件的空间选择、复用分类器。
局部texture仍可能带来噪声，因此先独立验证，不宣称机制已经被证明。

## 训练边界

- 放在最终Decoder之后、普通/DN拆分之后、最终分类输出之前。
- 训练和推理都用预测框，不用GT框/标签来取特征，不使用0.5分数门控。
- 采样坐标detach，新增分类分支不向bbox坐标反向传播。
- 原DN、Encoder/pre/较早层aux分类输出和原DDF分类teacher不被直接替换。
- 最终pred_logits会进入原matcher/VFL，匹配结果可以随分类变化；不能声称匹配不变。
- 原bbox前向计算不改，但分类梯度会更新共享特征，长期训练仍可能改变定位和hand AP。
- 默认关闭；旧基线/AQS配置不需要修改。与现有query实验互斥，第一版不能叠加AQS。
- 从原HGNet预训练正常完整训练，不用`-r`续训，也不用`-t`加载基线最优权重。
- 110轮/增强停止100轮/batch32/seed0，和当前110轮基线对照，而不是132轮均值。

增加71,552参数，当前配置训练参数总数10,294,559。
先降维再采样，FP32计算采样坐标、AMP计算其余分支；需要服务器检查实际GPU显存。

## 运行

只在用户桌面的D-FINE修改、再同步到服务器；本文件不自动同步或启动训练。
Mac终端同步本次文件（不上传数据集/旧输出；服务器同名源码先备份）：

```bash
cd /Users/wq/Desktop/D-FINE
rsync -avR --backup --backup-dir="/home/a5/MyProject/wq_project/D-FINE-sync-backups/rw-local-detail-$(date +%Y%m%d-%H%M%S)" \
  src/zoo/dfine/dfine.py src/zoo/dfine/dfine_decoder.py src/zoo/dfine/rw_local_detail.py \
  configs/dfine/dfine_s_scbs_hrw_110_rw_local_detail.yml \
  configs/dfine/dfine_s_scbs_hrw_110_rw_local_detail_diagnostics.yml \
  tools/diagnostics/test_rw_local_detail.py \
  tools/diagnostics/run_scbs_rw_local_detail_110.sh \
  tools/diagnostics/SCBS_RW_LOCAL_DETAIL_110.md \
  a5@100.76.151.67:/home/a5/MyProject/wq_project/D-FINE/
```

前提：服务器已经有当前110轮基线配置及当前完整源码；rsync会传送两份现有源码的
完整文件。如服务器另有未同步的源码修改，应先核对，备份不会自动合并这些修改。
服务器进入项目后：

```bash
cd /home/a5/MyProject/wq_project/D-FINE
bash tools/diagnostics/run_scbs_rw_local_detail_110.sh
```

默认输出：
`/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_rw_local_detail_seed0_run1`。
页面/终端实时输出同时保存console.log；目录已存在就停止，可传入新输出目录。
standalone自动分配端口，不使用之前被占用的7781。

训练完成后，用自己的最佳权重导出（不是AQS或基线权重）：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n wq \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py -c configs/dfine/dfine_s_scbs_hrw_110_rw_local_detail_diagnostics.yml \
-r '/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_rw_local_detail_seed0_run1/best_stg2.pth' \
--test-only --output-dir '/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_rw_local_detail_seed0_run1/diagnostics_eval' \
--use-amp --seed=0
```

使用新诊断目录，不覆盖旧导出。重点看write/read AP、双向混淆、正确匹配数、
未匹配GT与预测数、APm、Recall。混淆下降同时漏检上升不能简单算修正成功。
若独立实验无正信号，不直接叠加AQS或不断调参；若有，再复验或做AQS组合消融。

## 验证

```bash
conda run --no-capture-output -n wq python tools/diagnostics/test_rw_local_detail.py
```

覆盖零初始化/基线初始化RNG、配置单变量、FP32采样、普通query与DN边界、
原teacher、bbox不变、loss反传、优化器参数、strict权重读写、640推理、CPU AMP和空GT。
有CUDA时额外运行640分辨率FP16训练冒烟测试（batch2，不代表batch32显存测试）。
原Encoder部署BN融合可能重排未训练随机网络的TopK并列候选，
所以Decoder部署一致性使用同一组特征单独验证，同时检查完整部署输出有限。
