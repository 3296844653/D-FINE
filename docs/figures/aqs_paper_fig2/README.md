# 原论文Figure 2基础上加入AQS-L2

来源：用户提供的 `/Users/wq/Desktop/D-FINE.pdf` 第5页Figure 2。
原论文图2是FDR结构示意，并非完整骨干/编码器逐算子展开图。此版本保留原图布局，
只在Decoder Layer 2至Layer 3之间的query路径增加AQS；不把AQS插到分布、加权函数或框更新路径。

## 文件及编辑范围

- `dfine_paper_fig2_with_aqs_l2.drawio`：原图锁定为背景；新增AQS、连接线和下方放大分支为原生可编辑图形。
- `dfine_paper_fig2_with_aqs_l2.svg`：原图作为内嵌PNG背景；新增内容为矢量。
- `dfine_paper_fig2_with_aqs_l2.png`：高分辨率预览。
- `../../../output/pdf/dfine_paper_fig2_with_aqs_l2.pdf` 对应工程 `output/pdf/`：合并原PDF图形，保留原始字体、路径及已有图片；新增内容为矢量。
- `original_fig2_background.png`：从原PDF渲染的图2背景，仅供SVG/draw.io使用，不是新实验图片。
- `generate_paper_figure.py`：生成程序，依赖本工程之前架构图的Scene定义。

重要：draw.io中的原论文背景不是逐元素重建，所以不能声称整张原图每个元素都可编辑。
论文插图优先使用PDF版本；SVG中原有曲线/示例图的清晰度由内嵌背景分辨率决定。
保留先前的整体架构图和内部图，未覆盖原文件。

## 结构依据

采用当前 `dfine_s_scbs_hrw_110_aqs_layer2_r010.yml` 与 `AdaptiveQuerySelectionRefiner`：
L2 query和当前ClsHead2+LQE2 logits进入AQS，更新query后传递到L3。
L2当前FDR先于AQS计算；框不作为AQS输入，不直接由AQS修改。L3框可受到更新query的间接影响。
保持300个普通query、原损失与原GO-LSD；训练DN和普通query分组计算，不相互池化。
放大分支补充原图未显示的分类logits门控输入，不表示修改FDR。

原图中的风筝、分布曲线和框仍然只是原论文示意，不能作为SCB-S行为检测结果或AQS收益的证据。

建议论文图注：在D-FINE原有FDR框架中加入AQS的检测结构。AQS位于第二层Decoder之后，
根据当前分类置信度构建门控上下文并残差增强query，更新query传递至第三层Decoder；
原有分布残差细化、非均匀加权及框更新路径保持不变。图改绘自D-FINE Figure 2。

本次只生成文档图，不修改训练代码或服务器，原始论文PDF保持不变。
