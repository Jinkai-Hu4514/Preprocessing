# Bandpass 服务器实验最终审计与汇报

> 状态：**PRELIMINARY**

> 科学判定：**证据未完成，只能作为初步结果**

当前报告不能当作正式结论。缺失项列在文末；只有官方范围全部跑完后才能去掉 PRELIMINARY。

## 一句话结论

实验文件或预注册 seed 尚未全部完成，现在只能汇报信号机制，不能下最终性能归因结论。

## 1. ElderAL：为什么 Bandpass 看起来还可以

- 源码阈值：严格 `T < 28` 直接返回原始数据。
- 有效样本：2404；旁路样本：2010，占 83.61%。
- 旁路样本处理前后逐值完全相同的比例：100.00%。

大白话：这部分样本没有真正经过 Bandpass，所以它们不会被滤坏。因此 ElderAL 上的结果不能直接解释成“Bandpass 特别适合该数据集”。

## 2. XRF55：去噪后负值和 abs 方向折叠

- 与分类实验一致，信号统计使用前三个用户：[1, 2, 3]。

| 方法 | 有意义负值率 | 负值能量占比 | abs后方向改变或丢失率 |
|---|---:|---:|---:|
| raw | 0.000% | 0.000% | 0.000% |
| wavelet | 0.000% | 0.000% | 0.000% |
| butterworth_o5_c0.3 | 0.000% | 0.000% | 0.000% |
| savgol_w7_p3 | 0.000% | 0.000% | 0.000% |
| bandpass_fs1000 | 49.759% | 50.457% | 49.759% |
| bandpass_fs200 | 49.991% | 50.323% | 49.960% |
| hampel_w5_s3 | 0.000% | 0.000% | 0.000% |

- **基线**是静态环境、设备增益和很慢漂移形成的信号底座。Bandpass 还会去掉 50 Hz 以上成分，所以 `raw−Bandpass` 不能全部叫基线。
- **正/负**只表示 Bandpass 零中心两侧；**上升/下降**看相邻帧差值 `x[t]−x[t−1]`。
- 取绝对值会把 `+a` 和 `−a` 变成同一个 `a`；负半轴上的局部斜率还可能反向，某些跨零变化会被压平。

## 3. 官方 7 case × 5 seed 分类消融

| case | 完成 seed | 测试准确率均值 ± SD |
|---|---:|---:|
| bp_fs1000_legacy_abs | 0/5 | N/A ± N/A |
| bp_fs1000_signed | 0/5 | N/A ± N/A |
| bp_fs200_legacy_abs | 1/5 | 62.58% ± 0.00% |
| bp_fs200_signed | 1/5 | 64.70% ± 0.00% |
| bp_fs200_signed_iqr_absnorm | 0/5 | N/A ± N/A |
| bp_fs200_signed_no_iqr | 0/5 | N/A ± N/A |
| savgol_reference | 0/5 | N/A ± N/A |

## 4. 关键配对效应

正数表示定义左边的分支准确率更高。单位为测试准确率百分点。

| 效应 | 配对 seed | 平均差值 | 正向 seed | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| sampling_effect_legacy | 0/5 | N/A | N/A | [N/A, N/A] |
| sampling_effect_signed | 0/5 | N/A | N/A | [N/A, N/A] |
| sign_effect_fs1000 | 0/5 | N/A | N/A | [N/A, N/A] |
| sign_effect_fs200 | 1/5 | +2.12 | 100% | [+2.12, +2.12] |
| signed_iqr_then_absnorm_vs_legacy_fs200 | 0/5 | N/A | N/A | [N/A, N/A] |
| signed_norm_vs_signed_iqr_absnorm | 0/5 | N/A | N/A | [N/A, N/A] |
| remove_iqr_effect_signed_fs200 | 0/5 | N/A | N/A | [N/A, N/A] |
| signed_fs200_vs_savgol | 0/5 | N/A | N/A | [N/A, N/A] |
| sampling_x_sign_interaction | 0/5 | N/A | N/A | [N/A, N/A] |

最关键的两个数：

- 固定 200 Hz 后的符号效应（signed−legacy abs）：+2.12 个百分点。
- 保留符号后的采样率效应（200−1000 Hz）：N/A 个百分点。

误差条和 bootstrap 区间只反映五个模型训练 seed 的随机性，不是对整个 XRF55 总体的置信区间。

## 5. 文件与完整性审计

| 科研图组 | PNG | PDF | SVG |
|---|---:|---:|---:|
| elder_bypass_overview | ✓ | ✓ | ✓ |
| elder_before_after | ✓ | ✓ | ✓ |
| xrf_negative_distribution | ✓ | ✓ | ✓ |
| sampling_sign_ablation | ✓ | ✓ | ✓ |

### 未完成或不一致项

- **signal_not_official_complete**：信号分析完成状态没有明确记录 official_complete=true。（/home/test/bupt_hjk/ablation/bandpass_server_results/signal_analysis/signal_study_summary.json）
- **elder_not_official_complete**：Elder 摘要没有明确记录 official_complete=true。（/home/test/bupt_hjk/ablation/bandpass_server_results/signal_analysis/elder_bandpass_summary.json）
- **xrf_not_official_complete**：XRF 摘要没有明确记录 official_complete=true。（/home/test/bupt_hjk/ablation/bandpass_server_results/signal_analysis/xrf55_negative_summary.json）
- **training_not_official_complete**：训练完成状态没有明确记录 official_complete=true。（/home/test/bupt_hjk/ablation/bandpass_server_results/sign_ablation/completion_status.json）
- **training_seeds_incomplete**：官方 7×5 缺少 33 组：bp_fs1000_legacy_abs/seed=42, bp_fs1000_legacy_abs/seed=49, bp_fs1000_legacy_abs/seed=514, bp_fs1000_legacy_abs/seed=654, bp_fs1000_legacy_abs/seed=886, bp_fs1000_signed/seed=42, bp_fs1000_signed/seed=49, bp_fs1000_signed/seed=514 …（/home/test/bupt_hjk/ablation/bandpass_server_results/sign_ablation/training_summary.csv）
- **training_aggregate_case_missing**：training_aggregate.csv 缺少 bp_fs1000_legacy_abs。（/home/test/bupt_hjk/ablation/bandpass_server_results/sign_ablation/training_aggregate.csv）
- **training_aggregate_case_missing**：training_aggregate.csv 缺少 bp_fs1000_signed。（/home/test/bupt_hjk/ablation/bandpass_server_results/sign_ablation/training_aggregate.csv）
- **training_aggregate_case_missing**：training_aggregate.csv 缺少 bp_fs200_signed_iqr_absnorm。（/home/test/bupt_hjk/ablation/bandpass_server_results/sign_ablation/training_aggregate.csv）
- **training_aggregate_case_missing**：training_aggregate.csv 缺少 bp_fs200_signed_no_iqr。（/home/test/bupt_hjk/ablation/bandpass_server_results/sign_ablation/training_aggregate.csv）
- **training_aggregate_case_missing**：training_aggregate.csv 缺少 savgol_reference。（/home/test/bupt_hjk/ablation/bandpass_server_results/sign_ablation/training_aggregate.csv）
- **paired_effect_missing**：paired_effects.csv 缺少 sampling_effect_legacy。（/home/test/bupt_hjk/ablation/bandpass_server_results/sign_ablation/paired_effects.csv）
- **paired_effect_missing**：paired_effects.csv 缺少 sampling_effect_signed。（/home/test/bupt_hjk/ablation/bandpass_server_results/sign_ablation/paired_effects.csv）
- **paired_effect_missing**：paired_effects.csv 缺少 sign_effect_fs1000。（/home/test/bupt_hjk/ablation/bandpass_server_results/sign_ablation/paired_effects.csv）
- **paired_effect_seeds_incomplete**：sign_effect_fs200 的配对 seed 不是预注册五个。（/home/test/bupt_hjk/ablation/bandpass_server_results/sign_ablation/paired_effects.csv）
- **paired_seed_effect_missing**：paired_seed_effects.csv 缺少 sign_effect_fs200/seed=49。（/home/test/bupt_hjk/ablation/bandpass_server_results/sign_ablation/paired_seed_effects.csv）
- **paired_seed_effect_missing**：paired_seed_effects.csv 缺少 sign_effect_fs200/seed=514。（/home/test/bupt_hjk/ablation/bandpass_server_results/sign_ablation/paired_seed_effects.csv）
- **paired_seed_effect_missing**：paired_seed_effects.csv 缺少 sign_effect_fs200/seed=654。（/home/test/bupt_hjk/ablation/bandpass_server_results/sign_ablation/paired_seed_effects.csv）
- **paired_seed_effect_missing**：paired_seed_effects.csv 缺少 sign_effect_fs200/seed=886。（/home/test/bupt_hjk/ablation/bandpass_server_results/sign_ablation/paired_seed_effects.csv）
- **paired_effect_missing**：paired_effects.csv 缺少 signed_iqr_then_absnorm_vs_legacy_fs200。（/home/test/bupt_hjk/ablation/bandpass_server_results/sign_ablation/paired_effects.csv）
- **paired_effect_missing**：paired_effects.csv 缺少 signed_norm_vs_signed_iqr_absnorm。（/home/test/bupt_hjk/ablation/bandpass_server_results/sign_ablation/paired_effects.csv）
- **paired_effect_missing**：paired_effects.csv 缺少 remove_iqr_effect_signed_fs200。（/home/test/bupt_hjk/ablation/bandpass_server_results/sign_ablation/paired_effects.csv）
- **paired_effect_missing**：paired_effects.csv 缺少 signed_fs200_vs_savgol。（/home/test/bupt_hjk/ablation/bandpass_server_results/sign_ablation/paired_effects.csv）
- **paired_effect_missing**：paired_effects.csv 缺少 sampling_x_sign_interaction。（/home/test/bupt_hjk/ablation/bandpass_server_results/sign_ablation/paired_effects.csv）
