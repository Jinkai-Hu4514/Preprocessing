# Bandpass 在 XRF55 上效果不佳：服务器实验汇报摘要

# INCOMPLETE

**当前输出不是正式实验结论。所有差值只能当烟测/探索结果，禁止据此下强因果结论。**

未完成原因：
- valid artifact-backed results must be exactly the official 35: missing=[('bp_fs1000_legacy_abs', 42), ('bp_fs1000_legacy_abs', 49), ('bp_fs1000_legacy_abs', 514), ('bp_fs1000_legacy_abs', 654), ('bp_fs1000_legacy_abs', 886), ('bp_fs1000_signed', 42), ('bp_fs1000_signed', 49), ('bp_fs1000_signed', 514), ('bp_fs1000_signed', 654), ('bp_fs1000_signed', 886), ('bp_fs200_legacy_abs', 49), ('bp_fs200_legacy_abs', 514), ('bp_fs200_legacy_abs', 654), ('bp_fs200_legacy_abs', 886), ('bp_fs200_signed', 49), ('bp_fs200_signed', 514), ('bp_fs200_signed', 654), ('bp_fs200_signed', 886), ('bp_fs200_signed_iqr_absnorm', 42), ('bp_fs200_signed_iqr_absnorm', 49), ('bp_fs200_signed_iqr_absnorm', 514), ('bp_fs200_signed_iqr_absnorm', 654), ('bp_fs200_signed_iqr_absnorm', 886), ('bp_fs200_signed_no_iqr', 42), ('bp_fs200_signed_no_iqr', 49), ('bp_fs200_signed_no_iqr', 514), ('bp_fs200_signed_no_iqr', 654), ('bp_fs200_signed_no_iqr', 886), ('savgol_reference', 42), ('savgol_reference', 49), ('savgol_reference', 514), ('savgol_reference', 654), ('savgol_reference', 886)], unexpected=[]
- key effect missing: sampling_effect_legacy
- key effect missing: sampling_effect_signed
- key effect missing: sign_effect_fs1000
- key effect sign_effect_fs200 is not paired on all 5 official seeds
- key effect missing: sampling_x_sign_interaction
- key effect missing: signed_iqr_then_absnorm_vs_legacy_fs200
- key effect missing: signed_norm_vs_signed_iqr_absnorm
- key effect missing: remove_iqr_effect_signed_fs200
- key effect missing: signed_fs200_vs_savgol

## 先把三个概念说清楚

- **基线**：CSI 中由静态环境、设备增益和很慢漂移造成的整体底座。Bandpass 去掉低于 0.5 Hz 的慢变化，同时也去掉高于 50 Hz 的部分，所以“原始值减 Bandpass”不能全部叫基线。
- **正值/负值**：Bandpass 把信号拉到零附近后，正负只表示位于零中心的哪一侧，不直接表示上升或下降。
- **上升/下降**：看相邻帧差值 `x[t]-x[t-1]`。差值为正才叫上升，差值为负才叫下降。
- **z-score 后的负数**：只表示该数值低于训练集均值，已经不是 Bandpass 输出的负半轴，二者不能混为一谈。

## 分类结果

| 实验分支 | 完成度 | 配置 | 测试准确率（均值 ± SD） |
|---|---:|---|---:|
| bp_fs1000_legacy_abs | 0/5 | bandpass; fs=1000 Hz; IQR=legacy_abs; norm=abs; epochs=50; users=3 | — |
| bp_fs1000_signed | 0/5 | bandpass; fs=1000 Hz; IQR=signed; norm=signed; epochs=50; users=3 | — |
| bp_fs200_legacy_abs | 1/5 | bandpass; fs=200 Hz; IQR=legacy_abs; norm=abs; epochs=50; users=3 | 62.58% ± 0.00% |
| bp_fs200_signed | 1/5 | bandpass; fs=200 Hz; IQR=signed; norm=signed; epochs=50; users=3 | 64.70% ± 0.00% |
| bp_fs200_signed_iqr_absnorm | 0/5 | bandpass; fs=200 Hz; IQR=signed; norm=abs; epochs=50; users=3 | — |
| bp_fs200_signed_no_iqr | 0/5 | bandpass; fs=200 Hz; IQR=none; norm=signed; epochs=50; users=3 | — |
| savgol_reference | 0/5 | savgol; fs=N/A; IQR=legacy_abs; norm=abs; epochs=50; users=3 | — |

## 因果判定

- **INCOMPLETE：不进行正式因果判定。** 下列数值如存在，只用于检查代码和决定是否继续跑满。
- 探索性符号效应（200 Hz）：+2.12 个百分点（1/5 对 seed）。

## 汇报时必须加的一句话

正式结论要求预注册的 5 个模型 seed 全部完成。这里的重复只改变模型随机种子，所以误差条和 bootstrap 区间只反映训练随机性，不能当成对整个 XRF55 总体的统计置信区间。
