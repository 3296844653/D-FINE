# SCB-S：AQS 110轮独立复验

训练配置`configs/dfine/dfine_s_scbs_hrw_110_aqs.yml`直接继承独立110轮基线，停止增强100轮、batch32、seed0、原数据路径及类别顺序不变。仅启用已有AQS，参数保持threshold=0.5、temperature=0.1、residual_init=0.05；不叠加候选校准、Pairwise CE、MFFE或High-res Residual。不修改原损失、损失权重或匹配器实现。

结构和公式见`SCBS_AQS.md`。本次不是新增AQS版本，只检查原机制在110/100训练流程下是否仍有收益。分类输出变化会影响匹配和共享特征的训练，不能宣称训练后的框完全相同。

## Mac同步

```bash
cd /Users/wq/Desktop/D-FINE
rsync -avR --backup \
  --backup-dir="/home/a5/MyProject/wq_project/D-FINE-sync-backups/aqs110_$(date +%Y%m%d_%H%M%S)" \
  configs/dfine/dfine_s_scbs_hrw_110.yml \
  configs/dfine/dfine_s_scbs_hrw_110_aqs.yml \
  configs/dfine/dfine_s_scbs_hrw_110_aqs_diagnostics.yml \
  tools/diagnostics/test_aqs_refine.py \
  tools/diagnostics/SCBS_AQS_110.md \
  a5@100.76.151.67:/home/a5/MyProject/wq_project/D-FINE/
```

服务器须保留已经同步过的AQS源码。本次没有修改模型/损失Python源码，不需要再次覆盖它们。

## 服务器训练

确认GPU空闲且新输出目录不存在，不覆盖旧132轮AQS结果。

```bash
cd /home/a5/MyProject/wq_project/D-FINE
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
conda run --no-capture-output -n wq \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py -c configs/dfine/dfine_s_scbs_hrw_110_aqs.yml \
--output-dir "/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_aqs_seed0_run1" \
--use-amp --seed=0
```

这是按基线流程从头训练，不添加-r/-t，不加载132轮AQS或110轮基线的训练完成权重。

## 训练后诊断

```bash
cd /home/a5/MyProject/wq_project/D-FINE
SCBS_AQS110_RUN="/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_aqs_seed0_run1"
test ! -e "$SCBS_AQS110_RUN/diagnostics_eval" && \
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
conda run --no-capture-output -n wq \
torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
train.py -c configs/dfine/dfine_s_scbs_hrw_110_aqs_diagnostics.yml \
-r "$SCBS_AQS110_RUN/best_stg2.pth" --test-only \
--output-dir "$SCBS_AQS110_RUN/diagnostics_eval" --use-amp --seed=0
```

成功后运行`tools/diagnostics/analyze_detection_diagnostics.py --input-dir "$SCBS_AQS110_RUN/diagnostics_eval"`，使用wq环境。对照110轮基线，不拿132轮均值作直接对照。

当前110轮三次基线AP为55.15、55.05、55.27，均值55.16±0.11。先看总AP、read/write AP和双向混淆、writing Recall/Precision；微小提升只算候选信号，需要复验。没有预先确认AQS在110轮有效，结构测试通过也不等于AP提升。

小样本代码检查：`python tools/diagnostics/test_aqs_refine.py`，无数据下载或完整训练。新增检查保证独立110配置只启用AQS、原损失不变。

本次8项CPU检查通过，包括完整S前后向、空GT、DN隔离、混合精度、优化器和权重加载。未启动服务器完整训练或进行真实4090显存/AP验证。
