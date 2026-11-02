# Bandpass：ElderAL 与 XRF55 服务器验证

这套实验只在服务器读取数据，默认路径与服务器一致：

- `sdp_dataset/elderAL`
- `sdp_dataset/xrf55/wifi`

所有结果统一写入一个新目录：

```text
ablation/bandpass_server_results/
├── signal_analysis/   # 信号统计、Elder/XRF55 科研图
├── sign_ablation/     # 7×5 分类消融、模型和配对效应
└── final_report/      # 最终合并报告与完整性审计
```

四个文件的分工：

- `bandpass_server_signal_analysis.py`：不训练模型，只验证 Elder 旁路机制，
  并统计 XRF55 七种方法的负值和方向变化；XRF55 默认只使用前三个用户。
- `bandpass_server_sign_ablation.py`：训练模型，用 2×2 和补充分支验证
  “采样率”和“符号折叠”是否真的影响准确率。
- `bandpass_server_final_report.py`：不读数据、不训练，只检查前两步是否跑完整，
  然后合并成一份可汇报结论。
- `README_bandpass_server.md`：运行顺序、指标含义和判定标准。

本地只需要做无数据自检：

```bash
python ablation/bandpass_server_signal_analysis.py --self-test
python ablation/bandpass_server_sign_ablation.py --self-test
```

## 1. 先跑信号层验证

```bash
python ablation/bandpass_server_signal_analysis.py \
  --elder-root sdp_dataset/elderAL \
  --xrf-root sdp_dataset/xrf55/wifi \
  --xrf-user-count 3 \
  --workers 8 \
  --resume
```

XRF55 只选择文件名中数值排序后的前三个用户。标准数据应为
`3 × 55动作 × 20次重复 = 3300` 个文件，而不是全部11个用户的12100个文件。

结果目录：

```text
ablation/bandpass_server_results/signal_analysis/
```

它回答三个问题：

1. ElderAL 有多少样本因为原始帧数 `T < 28` 被源码直接返回？
2. XRF55 各去噪方法处理后，负值比例、负值能量和样本间分布分别是多少？
3. 把输出取绝对值后，相邻帧的上升/下降方向有多少被改变？

主要结果：

- `elder_bandpass_bypass_overview.{png,pdf,svg}`
- `elder_bandpass_before_after.{png,pdf,svg}`
- `elder_bandpass_summary.json`
- `xrf55_denoiser_negative_distribution.{png,pdf,svg}`
- `xrf55_rise_fall_and_abs_effect.{png,pdf,svg}`
- `xrf55_negative_method_summary.csv`
- `signal_report_summary.md`

其中 XRF55 的统计位置固定在：

```text
去噪之后
→ IQR 之前
→ 插值之前
→ padding 之前
→ 归一化之前
```

因此这里出现的负值确实来自原始输入或去噪算法，不会和后面的 z-score
负值混在一起。

## 2. 再跑分类因果消融

```bash
python ablation/bandpass_server_sign_ablation.py \
  --data-path sdp_dataset/xrf55/wifi \
  --workers 4 \
  --gpu 1 \
  --resume
```

结果目录：

```text
ablation/bandpass_server_results/sign_ablation/
```

正式实验为 7 个分支、5 个模型 seed。核心是严格的 2×2：

```text
采样率：1000 Hz / 200 Hz
表示方式：legacy abs / 保留正负号
```

固定不变的条件：

- 前 3 个真实用户 ID；
- repetition 1–12 训练、13–16 验证、17–20 测试；
- cubic15、resize1000、ResNet1D；
- 归一化只使用训练集统计量；
- seed：42、49、514、654、886；
- 50 epochs。

三个补充分支用于定位具体是哪一步造成影响：

- `bp_fs200_signed_iqr_absnorm`：IQR 保留符号，但归一化前仍取绝对值；
- `bp_fs200_signed_no_iqr`：保留符号且去掉 IQR；
- `savgol_reference`：Savgol 对照。

主要结果：

- `training_summary.csv`
- `training_aggregate.csv`
- `paired_seed_effects.csv`
- `paired_effects.csv`
- `figures/bandpass_sampling_sign_ablation.{png,pdf,svg}`
- `report_summary.md`

每个 case 还会独立保存：

- 配置哈希；
- 源数据顺序哈希；
- 完整用户/动作/repetition 网格检查；
- 预处理诊断；
- 测试样本清单；
- 每个 seed 的 checkpoint、预测和日志。

预处理缓存使用源数据 dtype 的磁盘映射文件，避免把全部原始 XRF55 和全部
中间数组同时放进内存。缓存会占用较多磁盘，但能支持断点续跑。

## 3. 汇报时怎样解释

### 基线是什么

基线是静态环境、多径、设备增益和很慢漂移形成的信号底座。Bandpass 同时去掉
低于 0.5 Hz 和高于 50 Hz 的成分，所以 `原始信号 − Bandpass 输出` 不能全部
叫作基线。

### 上升和下降是什么

正值、负值只表示 Bandpass 零中心的两侧。上升和下降看的是相邻帧差：

```text
x[t] - x[t-1] > 0：上升
x[t] - x[t-1] < 0：下降
```

取绝对值会在负半轴上翻转部分斜率方向，跨零点时也会改变波形关系。

另外，z-score 之后出现的负数只表示“低于训练集均值”，已经不是 Bandpass
零中心的负半轴。汇报时不能把这两种负数混在一起。

### 什么时候能说“符号折叠是原因”

重点看同一采样率下：

```text
sign_effect_fs200
= bp_fs200_signed − bp_fs200_legacy_abs
```

只有 5 个预注册 seed 全部完成，并且：

- 平均差值为正；
- 至少 4/5 个 seed 为正；
- 训练随机性的配对 bootstrap 区间不跨 0；

才可以说：后续 IQR/`abs` 折叠 Bandpass 正负号，是 XRF55 性能下降的原因之一。

再看：

```text
sampling_effect_signed
= bp_fs200_signed − bp_fs1000_signed
```

它表示修正符号问题以后，采样率设置本身还剩多少影响。

注意：5 个 seed 只反映训练随机性，不是 5 份独立数据集，不能把误差条说成
“对整个 XRF55 总体的置信区间”。

## 4. 生成唯一的最终汇报

两套实验都跑完后执行：

```bash
python ablation/bandpass_server_final_report.py
```

输出：

- `ablation/bandpass_server_results/final_report/bandpass_final_report.md`
- `ablation/bandpass_server_results/final_report/bandpass_result_audit.json`

只有完整扫描、四组核心科研图、官方 7 case × 5 seed、50 epochs、3 个用户
以及所有 checkpoint/预测/测试清单均通过检查时，报告才会标记 `COMPLETE`。
缺文件、少 seed 或烟测结果一律标记 `PRELIMINARY`，并默认返回非零退出码，
防止把半成品误当成正式结论。

## 5. 复数输入的边界

“负值”和“保留符号”只对实数 XRF55 有定义。分类消融若检测到真正的复数 CSI
会直接报错，防止把复数静默当成有正负号的幅度。

信号统计若确实需要研究复数数据的幅度，可显式使用：

```bash
python ablation/bandpass_server_signal_analysis.py \
  --only xrf \
  --xrf-complex-policy amplitude
```

此时报告只能称为 `abs(CSI)` 幅度实验，不能直接拿来证明实数符号折叠机制。
