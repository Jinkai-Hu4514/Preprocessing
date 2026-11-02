# elderAL / xrf55 80 全量测试与 Widar 相位 preset 测试分析

生成日期：2026-07-10。主指标统一使用 `test_acc`；`best_val_acc` 中旧 CSV 的百分制数值已折算为 0-1 比例后展示。不同数据集的任务、划分和模型不同，跨数据集只能看相对表现，不能把绝对准确率当成严格同榜。

## 1. 总览表
| 测试集 | 总/成功/失败 | 口径 | 平均 test | 中位 test | 最佳项 | 最佳 test | 最佳 val |
| --- | --- | --- | --- | --- | --- | --- | --- |
| elderAL 80 全量 | 80 / 80 / 0 | 固定模型 csitime | 79.53% | 82.59% | elderAL_037 / savgol_w7_p3+iqr+min-max+linear64 / csitime | 92.59% | 94.57% |
| xrf55 80 全量 | 80 / 80 / 0 | 固定模型 resnet1d | 74.75% | 83.18% | xrf55_034 / savgol_w7_p3+iqr+z-score+cubic15 / resnet1d | 85.15% | 90.45% |
| Widar 新相位 preset | 90 / 84 / 6 | 6 preset x 15 模型；cnn2d 全部 OOM | 62.07% | 61.82% | activity_detection / mlpmodel | 74.55% | 75.17% |
| Widar 旧 preset | 114 / 84 / 30 | 旧 condition_v2；多大模型失败/跳过 | 59.36% | 58.89% | gesture_recognition / mlpmodel | 68.69% | 70.24% |
| Widar 全量 pipeline 已有结果 | 5 / 5 / 0 | 当前 summary 只有前 5 个组合，不是完整 320 | 70.30% | 70.51% | widar_002 / wavelet+iqr+linear+z-score+cubic15 / mlpmodel | 70.71% | 72.56% |
| elderAL 旧 preset | 114 / 106 / 8 | 6 preset x 多模型 | 58.87% | 64.81% | high_quality / csitime | 91.48% | 92.51% |
| xrf55 旧 preset | 114 / 84 / 30 | 6 preset x 多模型 | 73.92% | 79.85% | fast / cnn1dmodel | 84.70% | 90.61% |

## 2. 关键对比
| 对比 | 新/全量 | 旧/基线 | 差值 | 判断 |
| --- | --- | --- | --- | --- |
| elderAL 全量 vs 旧 preset 同模型 | 92.59% | 91.48% | +1.11 pp | 全量最佳略高，主要收益来自 savgol+IQR+min-max 的组合搜索。 |
| xrf55 全量 vs 旧 preset 同模型 | 85.15% | 84.09% | +1.06 pp | 全量最佳小幅高于旧 preset；xrf55 对除 bandpass 外的处理不太敏感。 |
| xrf55 全量 vs 旧 preset 全模型最佳 | 85.15% | 84.70% | +0.45 pp | 全量 resnet1d 只比旧 preset 的 fast+cnn1d 高 0.45pp。 |
| Widar 新相位 preset vs 旧 preset 最佳 | 74.55% | 68.69% | +5.86 pp | 新相位输入把 Widar 最佳从 68.69% 推到 74.55%。 |
| Widar 新相位 preset vs Widar 全量已有最佳 | 74.55% | 70.71% | +3.84 pp | 新 preset 的 activity_detection+mlpmodel 超过当前 full_tests_new 已完成 5 项。 |
| Widar 新旧相同 84 对均值 | 62.07% | 59.36% | +2.72 pp | 相同 preset+model 共 84 对：61 个提升、23 个下降，中位提升 +2.22pp。 |

## 3. 80 全量算法测试 Top 配置
### elderAL 80 全量 Top 10
| id | pipeline | model | test | val | 耗时(s) |
| --- | --- | --- | --- | --- | --- |
| elderAL_037 | savgol_w7_p3+iqr+min-max+linear64 | csitime | 92.59% | 94.57% | 177.98 |
| elderAL_040 | savgol_w7_p3+iqr+min-max+decimate64 | csitime | 92.59% | 93.82% | 177.56 |
| elderAL_038 | savgol_w7_p3+iqr+min-max+cubic64 | csitime | 92.22% | 94.19% | 181.77 |
| elderAL_024 | butterworth_o5_c0.3+iqr+min-max+decimate64 | csitime | 91.85% | 92.88% | 23.95 |
| elderAL_039 | savgol_w7_p3+iqr+min-max+nearest64 | csitime | 91.48% | 95.88% | 181.62 |
| elderAL_023 | butterworth_o5_c0.3+iqr+min-max+nearest64 | csitime | 91.11% | 92.70% | 24.13 |
| elderAL_045 | savgol_w7_p3+outlier_z-score+min-max+linear64 | csitime | 90.37% | 95.13% | 176.96 |
| elderAL_014 | wavelet+outlier_z-score+min-max+cubic64 | csitime | 90.00% | 90.64% | 54.62 |
| elderAL_021 | butterworth_o5_c0.3+iqr+min-max+linear64 | csitime | 90.00% | 92.88% | 24.16 |
| elderAL_047 | savgol_w7_p3+outlier_z-score+min-max+nearest64 | csitime | 88.89% | 94.19% | 177.78 |

### xrf55 80 全量 Top 10
| id | pipeline | model | test | val | 耗时(s) |
| --- | --- | --- | --- | --- | --- |
| xrf55_034 | savgol_w7_p3+iqr+z-score+cubic15 | resnet1d | 85.15% | 90.45% | 482.14 |
| xrf55_007 | wavelet+iqr+min-max+nearest15 | resnet1d | 85.00% | 90.45% | 199.32 |
| xrf55_026 | butterworth_o5_c0.3+outlier_z-score+z-score+cubic15 | resnet1d | 85.00% | 90.76% | 467.92 |
| xrf55_032 | butterworth_o5_c0.3+outlier_z-score+min-max+decimate15 | resnet1d | 85.00% | 90.00% | 62.29 |
| xrf55_040 | savgol_w7_p3+iqr+min-max+decimate15 | resnet1d | 85.00% | 90.45% | 99.52 |
| xrf55_045 | savgol_w7_p3+outlier_z-score+min-max+linear15 | resnet1d | 84.85% | 90.45% | 238.07 |
| xrf55_024 | butterworth_o5_c0.3+iqr+min-max+decimate15 | resnet1d | 84.70% | 89.39% | 65.56 |
| xrf55_010 | wavelet+outlier_z-score+z-score+cubic15 | resnet1d | 84.39% | 89.70% | 459.99 |
| xrf55_013 | wavelet+outlier_z-score+min-max+linear15 | resnet1d | 84.39% | 90.91% | 214.71 |
| xrf55_004 | wavelet+iqr+z-score+decimate15 | resnet1d | 84.24% | 90.61% | 88.55 |

### Widar full_tests_new 当前已有 5 项
| id | pipeline | model | test | val | 耗时(s) |
| --- | --- | --- | --- | --- | --- |
| widar_002 | wavelet+iqr+linear+z-score+cubic15 | mlpmodel | 70.71% | 72.56% | 2124.07 |
| widar_001 | wavelet+iqr+linear+z-score+linear15 | mlpmodel | 70.61% | 72.37% | 1571.40 |
| widar_003 | wavelet+iqr+linear+z-score+nearest15 | mlpmodel | 70.51% | 72.66% | 1501.98 |
| widar_004 | wavelet+iqr+linear+z-score+decimate15 | mlpmodel | 70.20% | 72.17% | 1211.83 |
| widar_005 | wavelet+iqr+linear+min-max+linear15 | mlpmodel | 69.49% | 71.11% | 1767.40 |

## 4. 全量算法因子影响
### elderAL
| 因子 | 均值最佳水平 | 均值 | 该水平最佳 | 均值最差水平 | 均值 | 均值差 |
| --- | --- | --- | --- | --- | --- | --- |
| denoise | savgol_w7_p3 | 85.67% | 92.59% | hampel_w5_s3 | 64.19% | +21.48 pp |
| outliers | outlier_z-score | 80.50% | 90.37% | iqr | 78.56% | +1.94 pp |
| normalize | min-max | 81.69% | 92.59% | z-score | 77.36% | +4.33 pp |
| interpolate | decimate64 | 81.19% | 92.59% | cubic64 | 78.59% | +2.59 pp |

### xrf55
| 因子 | 均值最佳水平 | 均值 | 该水平最佳 | 均值最差水平 | 均值 | 均值差 |
| --- | --- | --- | --- | --- | --- | --- |
| denoise | wavelet | 83.78% | 85.00% | bandpass_0.5-50 | 40.09% | +43.68 pp |
| outliers | outlier_z-score | 74.84% | 85.00% | iqr | 74.66% | +0.18 pp |
| normalize | min-max | 74.89% | 85.00% | z-score | 74.61% | +0.28 pp |
| interpolate | decimate15 | 75.08% | 85.00% | nearest15 | 74.42% | +0.67 pp |

## 5. Widar 新相位 preset：按 preset 汇总
| preset | 成功数 | 平均 test | 中位 test | 最佳模型 | 最佳 test | 最佳 val |
| --- | --- | --- | --- | --- | --- | --- |
| activity_detection | 14 | 63.16% | 61.82% | mlpmodel | 74.55% | 75.17% |
| fast | 14 | 63.84% | 64.24% | mlpmodel | 71.62% | 74.20% |
| gesture_recognition | 14 | 62.95% | 62.53% | mlpmodel | 73.64% | 75.17% |
| high_quality | 14 | 63.46% | 62.07% | mlpmodel | 74.44% | 74.78% |
| localization | 14 | 61.82% | 61.67% | that | 68.48% | 71.01% |
| robust | 14 | 57.20% | 56.67% | that | 65.86% | 66.09% |

## 6. Widar 新相位 preset：按模型汇总
| model | 成功数 | 平均 test | 最佳 preset | 最佳 test |
| --- | --- | --- | --- | --- |
| attentiongru | 6 | 58.59% | high_quality | 61.92% |
| bilstmattention | 6 | 64.21% | activity_detection | 69.29% |
| cnn1dmodel | 6 | 63.48% | high_quality | 69.39% |
| csimodel | 6 | 61.14% | activity_detection | 63.43% |
| csitime | 6 | 62.24% | fast | 67.88% |
| ei | 6 | 63.05% | fast | 66.26% |
| fewsense | 6 | 58.80% | fast | 61.92% |
| lstmmodel | 6 | 57.04% | localization | 60.10% |
| mlpmodel | 6 | 70.42% | activity_detection | 74.55% |
| pa_csi | 6 | 61.20% | fast | 65.05% |
| resnet1d | 6 | 61.50% | fast | 68.59% |
| resnet2d | 6 | 64.49% | high_quality | 68.59% |
| that | 6 | 61.84% | localization | 68.48% |
| wiflexformer | 6 | 61.01% | fast | 63.43% |

## 7. Widar 新旧相同组合差值
### 按 preset
| preset | 共同组合数 | 平均差值 | 提升数 | 下降数 |
| --- | --- | --- | --- | --- |
| activity_detection | 14 | +3.46 pp | 10 | 4 |
| fast | 14 | +4.14 pp | 11 | 3 |
| gesture_recognition | 14 | +2.04 pp | 10 | 4 |
| high_quality | 14 | +3.18 pp | 11 | 3 |
| localization | 14 | +4.46 pp | 13 | 1 |
| robust | 14 | -0.98 pp | 6 | 8 |

### 按模型
| model | 共同组合数 | 平均差值 | 提升数 | 下降数 |
| --- | --- | --- | --- | --- |
| attentiongru | 6 | -0.66 pp | 2 | 4 |
| bilstmattention | 6 | +5.59 pp | 5 | 1 |
| cnn1dmodel | 6 | +2.93 pp | 5 | 1 |
| csimodel | 6 | +2.37 pp | 5 | 1 |
| csitime | 6 | +3.13 pp | 5 | 1 |
| ei | 6 | +3.74 pp | 6 | 0 |
| fewsense | 6 | -1.16 pp | 2 | 4 |
| lstmmodel | 6 | -0.27 pp | 1 | 5 |
| mlpmodel | 6 | +5.29 pp | 5 | 1 |
| pa_csi | 6 | +3.37 pp | 6 | 0 |
| resnet1d | 6 | +3.82 pp | 5 | 1 |
| resnet2d | 6 | +6.43 pp | 5 | 1 |
| that | 6 | +2.61 pp | 5 | 1 |
| wiflexformer | 6 | +0.84 pp | 4 | 2 |

### 提升最大的 12 项
| preset | model | 旧 test | 新 test | 差值 |
| --- | --- | --- | --- | --- |
| activity_detection | bilstmattention | 57.68% | 69.29% | +11.62 pp |
| high_quality | resnet2d | 58.59% | 68.59% | +10.00 pp |
| localization | resnet2d | 54.55% | 64.44% | +9.90 pp |
| activity_detection | mlpmodel | 64.95% | 74.55% | +9.60 pp |
| high_quality | mlpmodel | 65.05% | 74.44% | +9.39 pp |
| fast | resnet1d | 60.00% | 68.59% | +8.59 pp |
| high_quality | bilstmattention | 59.80% | 68.18% | +8.38 pp |
| localization | csitime | 53.84% | 62.22% | +8.38 pp |
| gesture_recognition | resnet2d | 58.38% | 66.46% | +8.08 pp |
| localization | resnet1d | 52.93% | 60.81% | +7.88 pp |
| fast | csitime | 60.10% | 67.88% | +7.78 pp |
| fast | resnet2d | 59.39% | 66.97% | +7.58 pp |

### 下降最大的 12 项
| preset | model | 旧 test | 新 test | 差值 |
| --- | --- | --- | --- | --- |
| localization | fewsense | 64.24% | 57.27% | -6.97 pp |
| robust | fewsense | 63.23% | 57.17% | -6.06 pp |
| robust | wiflexformer | 58.08% | 52.83% | -5.25 pp |
| robust | resnet2d | 56.97% | 53.43% | -3.54 pp |
| robust | attentiongru | 57.58% | 54.24% | -3.33 pp |
| fast | attentiongru | 60.10% | 57.47% | -2.63 pp |
| gesture_recognition | lstmmodel | 57.58% | 55.35% | -2.22 pp |
| robust | cnn1dmodel | 56.36% | 54.34% | -2.02 pp |
| robust | resnet1d | 56.57% | 54.85% | -1.72 pp |
| high_quality | that | 59.19% | 57.88% | -1.31 pp |
| activity_detection | csitime | 61.21% | 60.20% | -1.01 pp |
| robust | mlpmodel | 63.33% | 62.32% | -1.01 pp |

## 8. 失败项
| 测试 | 失败数 | 失败模型分布 |
| --- | --- | --- |
| Widar 新相位 preset | 6 | cnn2dmodel:6 |
| Widar 旧 preset | 30 | cnn2dmodel:6, efficientnetcsi:6, visiontransformercsi:6, mambacsi:6, graphneuralcsi:6 |
| elderAL 旧 preset | 8 | efficientnetcsi:4, visiontransformercsi:4 |
| xrf55 旧 preset | 30 | cnn2dmodel:6, efficientnetcsi:6, visiontransformercsi:6, mambacsi:6, graphneuralcsi:6 |

## 9. 结论

- elderAL：当前最佳是 `elderAL_037 / savgol_w7_p3+iqr+min-max+linear64 / csitime`，test=92.59%。`elderAL_040` 同为 92.59%，但 `elderAL_037` 验证集更高。全量搜索比旧 preset 的 `high_quality+csitime` 高 +1.11pp。
- xrf55：当前最佳是 `xrf55_034 / savgol_w7_p3+iqr+z-score+cubic15 / resnet1d`，test=85.15%。它只比旧 preset 全模型最佳高 +0.45pp，说明旧 preset 已经接近最优；但 `bandpass_0.5-50` 在 xrf55 上均值只有 40.09%，明显破坏信号。
- Widar：新相位 preset 最佳是 `activity_detection+mlpmodel`，test=74.55%，比旧 preset 最佳 `gesture_recognition+mlpmodel` 高 +5.86pp，比当前 full_tests_new 已有最佳高 +3.84pp。
- Widar 使用相位后总体变好：相同 84 对 preset+model 平均 +2.72pp、61 对提升；提升最大的模型族是 resnet2d、bilstmattention、mlpmodel。
- Widar 不是所有项都变好：`robust` preset 平均 -0.98pp，`fewsense` 和 `lstmmodel` 平均也略降。原因更像是 robust phase sanitization/wavelet/z-score 与新增相位通道叠加后过度清洗或放大相位噪声，部分时序/轻量模型吃不下新增通道的信息。
- 如果只问“目前最好的是哪个”：跨所有这些结果按 test_acc 数值最高是 elderAL 的 `elderAL_037`/`elderAL_040`，92.59%；但各数据集不可严格横比。Widar 范围内最好是 `new_preset_tests/widar+activity_detection+mlpmodel`，74.55%。
