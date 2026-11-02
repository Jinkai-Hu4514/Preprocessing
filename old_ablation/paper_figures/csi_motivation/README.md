# CSI 预处理研究动机图

该图根据用户提供的草图及其中文说明重新绘制。它是研究动机示意图，不是实验结果图。

## 文件与使用

- `csi_preprocessing_motivation.drawio`：完整可编辑源文件，含 English - IEEE 和 中文 - 论文两个页面。用桌面 draw.io 的“文件 → 打开”选择此文件；每页含 117 个独立的文字、线条或形状对象，曲线由可编辑折线构成，未嵌入截图。
- `csi_preprocessing_motivation_en.svg`、`csi_preprocessing_motivation_zh.svg`：矢量图，文字保留为文本。
- `csi_preprocessing_motivation_en.png`、`csi_preprocessing_motivation_zh.png`：600 dpi 位图。
- `../../output/pdf/csi_preprocessing_motivation_en.pdf`、`../../output/pdf/csi_preprocessing_motivation_zh.pdf`：PDF 为纯矢量，使用的字体已嵌入。论文排版优先使用 PDF。
- `build_figure.py`：可复现的图稿生成脚本；`figure_trace.json`：来源、设计参数与使用限制。

设计宽度为 7.16 英寸（181.864 mm），适合 IEEE 双栏通栏图。正文约 9.28 pt，最小文字约 8.56 pt，小节标题约 10.31 pt。该信息量不宜直接缩成单栏图。

配色：蓝 `#0072B2`、深橙 `#B76500`、紫 `#8056A3`，配以深灰文字和白色背景。这是本图的设计选择，并非 IEEE 指定色卡。三种处理还分别使用实线、虚线、点划线和 P1/P2/P3 编号，便于不依赖颜色辨认。

## 建议图注（英文）

**Conceptual motivation for evaluating CSI preprocessing across recognition models.** (a) Raw channel state information (CSI) contains motion-related variations together with nuisance effects, including gain variation, drift, outliers, and phase distortions. The amplitude trace illustrates amplitude-related effects; phase impairments are noted separately. (b) Changes in preprocessing methods, their order, and their parameters can preserve, attenuate, or transform task-relevant cues. P1–P3 denote illustrative alternative configurations, not a prescribed pipeline order or a universal quality ranking. (c) The configurations are compared using the same data split, training protocol, and model architecture, with the model trained separately for each processed input. Symbols s1–s3 denote the resulting recognition scores without asserting their values or ordering. (d) Repeating the comparison across model architectures allows the dependence of preprocessing effects on the model to be assessed; si,A and si,B denote the scores for configuration Pi with models A and B, respectively. All waveforms are synthetic illustrations, and all scores are symbolic.

## 建议图注（中文）

**CSI 预处理影响识别性能的研究动机示意图。**（a）原始信道状态信息（CSI）同时包含人体动作相关变化和增益变化、漂移、异常值、相位失真等干扰；幅值曲线仅示意幅值相关现象，相位误差另行说明。（b）预处理方法、顺序及参数的变化，可能保留、削弱或改变任务相关信息。P1–P3 表示示例处理配置，不表示固定处理顺序或普遍的优劣排序。（c）在相同数据划分、训练设置及模型架构下，对不同预处理后的输入分别训练和测试，所得识别性能以 s1–s3 表示，不预设具体数值或大小关系。（d）进一步在不同模型架构上重复比较，以考察预处理效果对模型的依赖性；si,A 与 si,B 分别表示配置 Pi 在模型 A 和模型 B 上的识别性能。所有波形均为人工构造的示意，性能均以符号表示。

## 建议正文表述

原始 CSI 不仅包含人体动作相关变化，还受到硬件增益变化、相位误差及环境干扰等因素的影响，因此通常需要进行预处理。然而，预处理在抑制干扰的同时，也可能改变或削弱对识别任务有用的信息。不同预处理方法及其参数配置会产生不同的数据表征，从而可能使相同模型架构在相同数据划分和训练设置下获得不同的识别性能。此外，不同模型的特征提取能力可能进一步影响其对预处理的响应。因此，有必要在受控实验条件下，系统考察预处理配置及其与模型选择之间的关系。

## LaTeX

从仓库根目录编译时可使用：

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{output/pdf/csi_preprocessing_motivation_en.pdf}
  \caption{Conceptual motivation for evaluating CSI preprocessing across recognition models. Raw CSI contains both motion-related variations and nuisance effects. Alternative preprocessing configurations may change the retained cues and lead to different recognition scores under a fixed architecture and training protocol. Extending the comparison across architectures examines model dependence. All waveforms are synthetic and all scores are symbolic.}
  \label{fig:csi-preprocessing-motivation}
\end{figure*}
```

## 参考规范与论文

1. [IEEE Author Center: Resolution and Size](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/)：推荐矢量图，说明通栏与单栏尺寸。
2. [IEEE Author Center: Improve Your Graphics](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/improve-your-graphics/)：字体嵌入、推荐字体、实际尺寸下的字号，以及颜色与线型共同编码。
3. [Proceedings of the IEEE: Guidelines for Figures and Tables](https://proceedingsoftheieee.ieee.org/resources/guidelines-for-figures-and-tables/)：颜色对比、可读性、统一排版和简洁标注。IEEE 没有在这些规范中要求唯一的一组图稿颜色。
4. Ratnam et al., [Optimal preprocessing of WiFi CSI for sensing applications](https://arxiv.org/abs/2307.12126), IEEE Transactions on Wireless Communications, 2024, [DOI](https://doi.org/10.1109/TWC.2024.3376332)：供增益／相位误差与预处理表述核对，图 6–7 可作为同领域曲线图的参考。本图未复用该论文的图或数据。
5. Wang et al., [A Survey on Wi-Fi Sensing Generalizability: Taxonomy, Techniques, Datasets, and Future Research Prospects](https://arxiv.org/abs/2503.08008), arXiv v4, 2026，作者稿标注已被 IEEE Communications Surveys & Tutorials 接收：作为同领域流程与特征学习表述的参考。本图未复用其图或数据。

检查日期：2026-09-06。已核查官方规范、论文文本及下载的作者稿原图。TWC 论文图 6 使用红、绿、蓝、黑曲线，并结合线型及圆形、三角形、方形标记；COMST 综述图 4 以浅橙、浅紫、浅黄、浅蓝底区分模块，使用深色文字。本文借鉴其“颜色与线型共同编码”“浅色底与深色文字”的表达原则，所选色值由本图独立确定，未逐图取色或复制他人图稿。

交付检查：两个 draw.io 页面分别包含 117 个原生对象；SVG 通过 XML 解析；两份 PDF 均为单页，包含 0 个位图对象，所有字体资源均已嵌入；中英文渲染及英文灰度渲染已目视检查。桌面 draw.io 窗口未连接到本次可用的电脑控制接口，因此未完成在用户桌面编辑器中的打开验证。
