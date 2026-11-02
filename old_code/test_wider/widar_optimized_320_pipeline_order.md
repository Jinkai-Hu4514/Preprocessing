# Widar 320 种 pipeline 的公共前缀优化执行顺序

本文档对应 `SDP/test_wider/full_test_widar_new.py`，并保留原 `full_test_widar.py` 的 320 种算法配置、`combo_id`、`combo_name` 和组合顺序；新脚本只改变重复预处理的复用方式。

## 核心规则

320 种组合可以按以下前三步划分为 40 个公共前缀：

```text
denoise → outliers → calibrate
5 × 2 × 4 = 40 个公共前缀
```

每个公共前缀对应 8 个组合：4 个 z-score 组合和 4 个 min-max 组合。

Widar 的 z-score 是特殊流程。为了保留带正负号的归一化幅度和正确相位，当前源码的实际执行顺序是：

```text
公共前缀 → interpolate → z-score → [normalized_amplitude, phase]
```

min-max 不走这个特殊分支，实际执行顺序仍然是：

```text
公共前缀 → min-max → interpolate
```

因此每个公共前缀的复用结构为：

```text
公共前缀
├─ linear15   → z-score → 训练
├─ cubic15    → z-score → 训练
├─ nearest15  → z-score → 训练
├─ decimate15 → z-score → 训练
└─ min-max（只执行一次）
   ├─ linear15   → 训练
   ├─ cubic15    → 训练
   ├─ nearest15  → 训练
   └─ decimate15 → 训练
```

> 注意：下面的 `combo_name` 保持原配置命名顺序；“实际执行流程”则反映 Widar + z-score 在当前源码中的真实顺序。

## 40 个公共前缀及其 320 个组合

### 公共前缀 01：wavelet+iqr+linear

公共缓存结果：`wavelet → iqr → linear`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_001 | `wavelet+iqr+linear+z-score+linear15` | `wavelet → iqr → linear → linear15 → z-score → 训练` |
| widar_002 | `wavelet+iqr+linear+z-score+cubic15` | `wavelet → iqr → linear → cubic15 → z-score → 训练` |
| widar_003 | `wavelet+iqr+linear+z-score+nearest15` | `wavelet → iqr → linear → nearest15 → z-score → 训练` |
| widar_004 | `wavelet+iqr+linear+z-score+decimate15` | `wavelet → iqr → linear → decimate15 → z-score → 训练` |
| widar_005 | `wavelet+iqr+linear+min-max+linear15` | `wavelet → iqr → linear → min-max → linear15 → 训练` |
| widar_006 | `wavelet+iqr+linear+min-max+cubic15` | `wavelet → iqr → linear → min-max → cubic15 → 训练` |
| widar_007 | `wavelet+iqr+linear+min-max+nearest15` | `wavelet → iqr → linear → min-max → nearest15 → 训练` |
| widar_008 | `wavelet+iqr+linear+min-max+decimate15` | `wavelet → iqr → linear → min-max → decimate15 → 训练` |

### 公共前缀 02：wavelet+iqr+polynomial_d3

公共缓存结果：`wavelet → iqr → polynomial_d3`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_009 | `wavelet+iqr+polynomial_d3+z-score+linear15` | `wavelet → iqr → polynomial_d3 → linear15 → z-score → 训练` |
| widar_010 | `wavelet+iqr+polynomial_d3+z-score+cubic15` | `wavelet → iqr → polynomial_d3 → cubic15 → z-score → 训练` |
| widar_011 | `wavelet+iqr+polynomial_d3+z-score+nearest15` | `wavelet → iqr → polynomial_d3 → nearest15 → z-score → 训练` |
| widar_012 | `wavelet+iqr+polynomial_d3+z-score+decimate15` | `wavelet → iqr → polynomial_d3 → decimate15 → z-score → 训练` |
| widar_013 | `wavelet+iqr+polynomial_d3+min-max+linear15` | `wavelet → iqr → polynomial_d3 → min-max → linear15 → 训练` |
| widar_014 | `wavelet+iqr+polynomial_d3+min-max+cubic15` | `wavelet → iqr → polynomial_d3 → min-max → cubic15 → 训练` |
| widar_015 | `wavelet+iqr+polynomial_d3+min-max+nearest15` | `wavelet → iqr → polynomial_d3 → min-max → nearest15 → 训练` |
| widar_016 | `wavelet+iqr+polynomial_d3+min-max+decimate15` | `wavelet → iqr → polynomial_d3 → min-max → decimate15 → 训练` |

### 公共前缀 03：wavelet+iqr+stc

公共缓存结果：`wavelet → iqr → stc`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_017 | `wavelet+iqr+stc+z-score+linear15` | `wavelet → iqr → stc → linear15 → z-score → 训练` |
| widar_018 | `wavelet+iqr+stc+z-score+cubic15` | `wavelet → iqr → stc → cubic15 → z-score → 训练` |
| widar_019 | `wavelet+iqr+stc+z-score+nearest15` | `wavelet → iqr → stc → nearest15 → z-score → 训练` |
| widar_020 | `wavelet+iqr+stc+z-score+decimate15` | `wavelet → iqr → stc → decimate15 → z-score → 训练` |
| widar_021 | `wavelet+iqr+stc+min-max+linear15` | `wavelet → iqr → stc → min-max → linear15 → 训练` |
| widar_022 | `wavelet+iqr+stc+min-max+cubic15` | `wavelet → iqr → stc → min-max → cubic15 → 训练` |
| widar_023 | `wavelet+iqr+stc+min-max+nearest15` | `wavelet → iqr → stc → min-max → nearest15 → 训练` |
| widar_024 | `wavelet+iqr+stc+min-max+decimate15` | `wavelet → iqr → stc → min-max → decimate15 → 训练` |

### 公共前缀 04：wavelet+iqr+robust

公共缓存结果：`wavelet → iqr → robust`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_025 | `wavelet+iqr+robust+z-score+linear15` | `wavelet → iqr → robust → linear15 → z-score → 训练` |
| widar_026 | `wavelet+iqr+robust+z-score+cubic15` | `wavelet → iqr → robust → cubic15 → z-score → 训练` |
| widar_027 | `wavelet+iqr+robust+z-score+nearest15` | `wavelet → iqr → robust → nearest15 → z-score → 训练` |
| widar_028 | `wavelet+iqr+robust+z-score+decimate15` | `wavelet → iqr → robust → decimate15 → z-score → 训练` |
| widar_029 | `wavelet+iqr+robust+min-max+linear15` | `wavelet → iqr → robust → min-max → linear15 → 训练` |
| widar_030 | `wavelet+iqr+robust+min-max+cubic15` | `wavelet → iqr → robust → min-max → cubic15 → 训练` |
| widar_031 | `wavelet+iqr+robust+min-max+nearest15` | `wavelet → iqr → robust → min-max → nearest15 → 训练` |
| widar_032 | `wavelet+iqr+robust+min-max+decimate15` | `wavelet → iqr → robust → min-max → decimate15 → 训练` |

### 公共前缀 05：wavelet+outlier_z-score+linear

公共缓存结果：`wavelet → outlier_z-score → linear`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_033 | `wavelet+outlier_z-score+linear+z-score+linear15` | `wavelet → outlier_z-score → linear → linear15 → z-score → 训练` |
| widar_034 | `wavelet+outlier_z-score+linear+z-score+cubic15` | `wavelet → outlier_z-score → linear → cubic15 → z-score → 训练` |
| widar_035 | `wavelet+outlier_z-score+linear+z-score+nearest15` | `wavelet → outlier_z-score → linear → nearest15 → z-score → 训练` |
| widar_036 | `wavelet+outlier_z-score+linear+z-score+decimate15` | `wavelet → outlier_z-score → linear → decimate15 → z-score → 训练` |
| widar_037 | `wavelet+outlier_z-score+linear+min-max+linear15` | `wavelet → outlier_z-score → linear → min-max → linear15 → 训练` |
| widar_038 | `wavelet+outlier_z-score+linear+min-max+cubic15` | `wavelet → outlier_z-score → linear → min-max → cubic15 → 训练` |
| widar_039 | `wavelet+outlier_z-score+linear+min-max+nearest15` | `wavelet → outlier_z-score → linear → min-max → nearest15 → 训练` |
| widar_040 | `wavelet+outlier_z-score+linear+min-max+decimate15` | `wavelet → outlier_z-score → linear → min-max → decimate15 → 训练` |

### 公共前缀 06：wavelet+outlier_z-score+polynomial_d3

公共缓存结果：`wavelet → outlier_z-score → polynomial_d3`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_041 | `wavelet+outlier_z-score+polynomial_d3+z-score+linear15` | `wavelet → outlier_z-score → polynomial_d3 → linear15 → z-score → 训练` |
| widar_042 | `wavelet+outlier_z-score+polynomial_d3+z-score+cubic15` | `wavelet → outlier_z-score → polynomial_d3 → cubic15 → z-score → 训练` |
| widar_043 | `wavelet+outlier_z-score+polynomial_d3+z-score+nearest15` | `wavelet → outlier_z-score → polynomial_d3 → nearest15 → z-score → 训练` |
| widar_044 | `wavelet+outlier_z-score+polynomial_d3+z-score+decimate15` | `wavelet → outlier_z-score → polynomial_d3 → decimate15 → z-score → 训练` |
| widar_045 | `wavelet+outlier_z-score+polynomial_d3+min-max+linear15` | `wavelet → outlier_z-score → polynomial_d3 → min-max → linear15 → 训练` |
| widar_046 | `wavelet+outlier_z-score+polynomial_d3+min-max+cubic15` | `wavelet → outlier_z-score → polynomial_d3 → min-max → cubic15 → 训练` |
| widar_047 | `wavelet+outlier_z-score+polynomial_d3+min-max+nearest15` | `wavelet → outlier_z-score → polynomial_d3 → min-max → nearest15 → 训练` |
| widar_048 | `wavelet+outlier_z-score+polynomial_d3+min-max+decimate15` | `wavelet → outlier_z-score → polynomial_d3 → min-max → decimate15 → 训练` |

### 公共前缀 07：wavelet+outlier_z-score+stc

公共缓存结果：`wavelet → outlier_z-score → stc`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_049 | `wavelet+outlier_z-score+stc+z-score+linear15` | `wavelet → outlier_z-score → stc → linear15 → z-score → 训练` |
| widar_050 | `wavelet+outlier_z-score+stc+z-score+cubic15` | `wavelet → outlier_z-score → stc → cubic15 → z-score → 训练` |
| widar_051 | `wavelet+outlier_z-score+stc+z-score+nearest15` | `wavelet → outlier_z-score → stc → nearest15 → z-score → 训练` |
| widar_052 | `wavelet+outlier_z-score+stc+z-score+decimate15` | `wavelet → outlier_z-score → stc → decimate15 → z-score → 训练` |
| widar_053 | `wavelet+outlier_z-score+stc+min-max+linear15` | `wavelet → outlier_z-score → stc → min-max → linear15 → 训练` |
| widar_054 | `wavelet+outlier_z-score+stc+min-max+cubic15` | `wavelet → outlier_z-score → stc → min-max → cubic15 → 训练` |
| widar_055 | `wavelet+outlier_z-score+stc+min-max+nearest15` | `wavelet → outlier_z-score → stc → min-max → nearest15 → 训练` |
| widar_056 | `wavelet+outlier_z-score+stc+min-max+decimate15` | `wavelet → outlier_z-score → stc → min-max → decimate15 → 训练` |

### 公共前缀 08：wavelet+outlier_z-score+robust

公共缓存结果：`wavelet → outlier_z-score → robust`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_057 | `wavelet+outlier_z-score+robust+z-score+linear15` | `wavelet → outlier_z-score → robust → linear15 → z-score → 训练` |
| widar_058 | `wavelet+outlier_z-score+robust+z-score+cubic15` | `wavelet → outlier_z-score → robust → cubic15 → z-score → 训练` |
| widar_059 | `wavelet+outlier_z-score+robust+z-score+nearest15` | `wavelet → outlier_z-score → robust → nearest15 → z-score → 训练` |
| widar_060 | `wavelet+outlier_z-score+robust+z-score+decimate15` | `wavelet → outlier_z-score → robust → decimate15 → z-score → 训练` |
| widar_061 | `wavelet+outlier_z-score+robust+min-max+linear15` | `wavelet → outlier_z-score → robust → min-max → linear15 → 训练` |
| widar_062 | `wavelet+outlier_z-score+robust+min-max+cubic15` | `wavelet → outlier_z-score → robust → min-max → cubic15 → 训练` |
| widar_063 | `wavelet+outlier_z-score+robust+min-max+nearest15` | `wavelet → outlier_z-score → robust → min-max → nearest15 → 训练` |
| widar_064 | `wavelet+outlier_z-score+robust+min-max+decimate15` | `wavelet → outlier_z-score → robust → min-max → decimate15 → 训练` |

### 公共前缀 09：butterworth_o5_c0.3+iqr+linear

公共缓存结果：`butterworth_o5_c0.3 → iqr → linear`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_065 | `butterworth_o5_c0.3+iqr+linear+z-score+linear15` | `butterworth_o5_c0.3 → iqr → linear → linear15 → z-score → 训练` |
| widar_066 | `butterworth_o5_c0.3+iqr+linear+z-score+cubic15` | `butterworth_o5_c0.3 → iqr → linear → cubic15 → z-score → 训练` |
| widar_067 | `butterworth_o5_c0.3+iqr+linear+z-score+nearest15` | `butterworth_o5_c0.3 → iqr → linear → nearest15 → z-score → 训练` |
| widar_068 | `butterworth_o5_c0.3+iqr+linear+z-score+decimate15` | `butterworth_o5_c0.3 → iqr → linear → decimate15 → z-score → 训练` |
| widar_069 | `butterworth_o5_c0.3+iqr+linear+min-max+linear15` | `butterworth_o5_c0.3 → iqr → linear → min-max → linear15 → 训练` |
| widar_070 | `butterworth_o5_c0.3+iqr+linear+min-max+cubic15` | `butterworth_o5_c0.3 → iqr → linear → min-max → cubic15 → 训练` |
| widar_071 | `butterworth_o5_c0.3+iqr+linear+min-max+nearest15` | `butterworth_o5_c0.3 → iqr → linear → min-max → nearest15 → 训练` |
| widar_072 | `butterworth_o5_c0.3+iqr+linear+min-max+decimate15` | `butterworth_o5_c0.3 → iqr → linear → min-max → decimate15 → 训练` |

### 公共前缀 10：butterworth_o5_c0.3+iqr+polynomial_d3

公共缓存结果：`butterworth_o5_c0.3 → iqr → polynomial_d3`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_073 | `butterworth_o5_c0.3+iqr+polynomial_d3+z-score+linear15` | `butterworth_o5_c0.3 → iqr → polynomial_d3 → linear15 → z-score → 训练` |
| widar_074 | `butterworth_o5_c0.3+iqr+polynomial_d3+z-score+cubic15` | `butterworth_o5_c0.3 → iqr → polynomial_d3 → cubic15 → z-score → 训练` |
| widar_075 | `butterworth_o5_c0.3+iqr+polynomial_d3+z-score+nearest15` | `butterworth_o5_c0.3 → iqr → polynomial_d3 → nearest15 → z-score → 训练` |
| widar_076 | `butterworth_o5_c0.3+iqr+polynomial_d3+z-score+decimate15` | `butterworth_o5_c0.3 → iqr → polynomial_d3 → decimate15 → z-score → 训练` |
| widar_077 | `butterworth_o5_c0.3+iqr+polynomial_d3+min-max+linear15` | `butterworth_o5_c0.3 → iqr → polynomial_d3 → min-max → linear15 → 训练` |
| widar_078 | `butterworth_o5_c0.3+iqr+polynomial_d3+min-max+cubic15` | `butterworth_o5_c0.3 → iqr → polynomial_d3 → min-max → cubic15 → 训练` |
| widar_079 | `butterworth_o5_c0.3+iqr+polynomial_d3+min-max+nearest15` | `butterworth_o5_c0.3 → iqr → polynomial_d3 → min-max → nearest15 → 训练` |
| widar_080 | `butterworth_o5_c0.3+iqr+polynomial_d3+min-max+decimate15` | `butterworth_o5_c0.3 → iqr → polynomial_d3 → min-max → decimate15 → 训练` |

### 公共前缀 11：butterworth_o5_c0.3+iqr+stc

公共缓存结果：`butterworth_o5_c0.3 → iqr → stc`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_081 | `butterworth_o5_c0.3+iqr+stc+z-score+linear15` | `butterworth_o5_c0.3 → iqr → stc → linear15 → z-score → 训练` |
| widar_082 | `butterworth_o5_c0.3+iqr+stc+z-score+cubic15` | `butterworth_o5_c0.3 → iqr → stc → cubic15 → z-score → 训练` |
| widar_083 | `butterworth_o5_c0.3+iqr+stc+z-score+nearest15` | `butterworth_o5_c0.3 → iqr → stc → nearest15 → z-score → 训练` |
| widar_084 | `butterworth_o5_c0.3+iqr+stc+z-score+decimate15` | `butterworth_o5_c0.3 → iqr → stc → decimate15 → z-score → 训练` |
| widar_085 | `butterworth_o5_c0.3+iqr+stc+min-max+linear15` | `butterworth_o5_c0.3 → iqr → stc → min-max → linear15 → 训练` |
| widar_086 | `butterworth_o5_c0.3+iqr+stc+min-max+cubic15` | `butterworth_o5_c0.3 → iqr → stc → min-max → cubic15 → 训练` |
| widar_087 | `butterworth_o5_c0.3+iqr+stc+min-max+nearest15` | `butterworth_o5_c0.3 → iqr → stc → min-max → nearest15 → 训练` |
| widar_088 | `butterworth_o5_c0.3+iqr+stc+min-max+decimate15` | `butterworth_o5_c0.3 → iqr → stc → min-max → decimate15 → 训练` |

### 公共前缀 12：butterworth_o5_c0.3+iqr+robust

公共缓存结果：`butterworth_o5_c0.3 → iqr → robust`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_089 | `butterworth_o5_c0.3+iqr+robust+z-score+linear15` | `butterworth_o5_c0.3 → iqr → robust → linear15 → z-score → 训练` |
| widar_090 | `butterworth_o5_c0.3+iqr+robust+z-score+cubic15` | `butterworth_o5_c0.3 → iqr → robust → cubic15 → z-score → 训练` |
| widar_091 | `butterworth_o5_c0.3+iqr+robust+z-score+nearest15` | `butterworth_o5_c0.3 → iqr → robust → nearest15 → z-score → 训练` |
| widar_092 | `butterworth_o5_c0.3+iqr+robust+z-score+decimate15` | `butterworth_o5_c0.3 → iqr → robust → decimate15 → z-score → 训练` |
| widar_093 | `butterworth_o5_c0.3+iqr+robust+min-max+linear15` | `butterworth_o5_c0.3 → iqr → robust → min-max → linear15 → 训练` |
| widar_094 | `butterworth_o5_c0.3+iqr+robust+min-max+cubic15` | `butterworth_o5_c0.3 → iqr → robust → min-max → cubic15 → 训练` |
| widar_095 | `butterworth_o5_c0.3+iqr+robust+min-max+nearest15` | `butterworth_o5_c0.3 → iqr → robust → min-max → nearest15 → 训练` |
| widar_096 | `butterworth_o5_c0.3+iqr+robust+min-max+decimate15` | `butterworth_o5_c0.3 → iqr → robust → min-max → decimate15 → 训练` |

### 公共前缀 13：butterworth_o5_c0.3+outlier_z-score+linear

公共缓存结果：`butterworth_o5_c0.3 → outlier_z-score → linear`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_097 | `butterworth_o5_c0.3+outlier_z-score+linear+z-score+linear15` | `butterworth_o5_c0.3 → outlier_z-score → linear → linear15 → z-score → 训练` |
| widar_098 | `butterworth_o5_c0.3+outlier_z-score+linear+z-score+cubic15` | `butterworth_o5_c0.3 → outlier_z-score → linear → cubic15 → z-score → 训练` |
| widar_099 | `butterworth_o5_c0.3+outlier_z-score+linear+z-score+nearest15` | `butterworth_o5_c0.3 → outlier_z-score → linear → nearest15 → z-score → 训练` |
| widar_100 | `butterworth_o5_c0.3+outlier_z-score+linear+z-score+decimate15` | `butterworth_o5_c0.3 → outlier_z-score → linear → decimate15 → z-score → 训练` |
| widar_101 | `butterworth_o5_c0.3+outlier_z-score+linear+min-max+linear15` | `butterworth_o5_c0.3 → outlier_z-score → linear → min-max → linear15 → 训练` |
| widar_102 | `butterworth_o5_c0.3+outlier_z-score+linear+min-max+cubic15` | `butterworth_o5_c0.3 → outlier_z-score → linear → min-max → cubic15 → 训练` |
| widar_103 | `butterworth_o5_c0.3+outlier_z-score+linear+min-max+nearest15` | `butterworth_o5_c0.3 → outlier_z-score → linear → min-max → nearest15 → 训练` |
| widar_104 | `butterworth_o5_c0.3+outlier_z-score+linear+min-max+decimate15` | `butterworth_o5_c0.3 → outlier_z-score → linear → min-max → decimate15 → 训练` |

### 公共前缀 14：butterworth_o5_c0.3+outlier_z-score+polynomial_d3

公共缓存结果：`butterworth_o5_c0.3 → outlier_z-score → polynomial_d3`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_105 | `butterworth_o5_c0.3+outlier_z-score+polynomial_d3+z-score+linear15` | `butterworth_o5_c0.3 → outlier_z-score → polynomial_d3 → linear15 → z-score → 训练` |
| widar_106 | `butterworth_o5_c0.3+outlier_z-score+polynomial_d3+z-score+cubic15` | `butterworth_o5_c0.3 → outlier_z-score → polynomial_d3 → cubic15 → z-score → 训练` |
| widar_107 | `butterworth_o5_c0.3+outlier_z-score+polynomial_d3+z-score+nearest15` | `butterworth_o5_c0.3 → outlier_z-score → polynomial_d3 → nearest15 → z-score → 训练` |
| widar_108 | `butterworth_o5_c0.3+outlier_z-score+polynomial_d3+z-score+decimate15` | `butterworth_o5_c0.3 → outlier_z-score → polynomial_d3 → decimate15 → z-score → 训练` |
| widar_109 | `butterworth_o5_c0.3+outlier_z-score+polynomial_d3+min-max+linear15` | `butterworth_o5_c0.3 → outlier_z-score → polynomial_d3 → min-max → linear15 → 训练` |
| widar_110 | `butterworth_o5_c0.3+outlier_z-score+polynomial_d3+min-max+cubic15` | `butterworth_o5_c0.3 → outlier_z-score → polynomial_d3 → min-max → cubic15 → 训练` |
| widar_111 | `butterworth_o5_c0.3+outlier_z-score+polynomial_d3+min-max+nearest15` | `butterworth_o5_c0.3 → outlier_z-score → polynomial_d3 → min-max → nearest15 → 训练` |
| widar_112 | `butterworth_o5_c0.3+outlier_z-score+polynomial_d3+min-max+decimate15` | `butterworth_o5_c0.3 → outlier_z-score → polynomial_d3 → min-max → decimate15 → 训练` |

### 公共前缀 15：butterworth_o5_c0.3+outlier_z-score+stc

公共缓存结果：`butterworth_o5_c0.3 → outlier_z-score → stc`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_113 | `butterworth_o5_c0.3+outlier_z-score+stc+z-score+linear15` | `butterworth_o5_c0.3 → outlier_z-score → stc → linear15 → z-score → 训练` |
| widar_114 | `butterworth_o5_c0.3+outlier_z-score+stc+z-score+cubic15` | `butterworth_o5_c0.3 → outlier_z-score → stc → cubic15 → z-score → 训练` |
| widar_115 | `butterworth_o5_c0.3+outlier_z-score+stc+z-score+nearest15` | `butterworth_o5_c0.3 → outlier_z-score → stc → nearest15 → z-score → 训练` |
| widar_116 | `butterworth_o5_c0.3+outlier_z-score+stc+z-score+decimate15` | `butterworth_o5_c0.3 → outlier_z-score → stc → decimate15 → z-score → 训练` |
| widar_117 | `butterworth_o5_c0.3+outlier_z-score+stc+min-max+linear15` | `butterworth_o5_c0.3 → outlier_z-score → stc → min-max → linear15 → 训练` |
| widar_118 | `butterworth_o5_c0.3+outlier_z-score+stc+min-max+cubic15` | `butterworth_o5_c0.3 → outlier_z-score → stc → min-max → cubic15 → 训练` |
| widar_119 | `butterworth_o5_c0.3+outlier_z-score+stc+min-max+nearest15` | `butterworth_o5_c0.3 → outlier_z-score → stc → min-max → nearest15 → 训练` |
| widar_120 | `butterworth_o5_c0.3+outlier_z-score+stc+min-max+decimate15` | `butterworth_o5_c0.3 → outlier_z-score → stc → min-max → decimate15 → 训练` |

### 公共前缀 16：butterworth_o5_c0.3+outlier_z-score+robust

公共缓存结果：`butterworth_o5_c0.3 → outlier_z-score → robust`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_121 | `butterworth_o5_c0.3+outlier_z-score+robust+z-score+linear15` | `butterworth_o5_c0.3 → outlier_z-score → robust → linear15 → z-score → 训练` |
| widar_122 | `butterworth_o5_c0.3+outlier_z-score+robust+z-score+cubic15` | `butterworth_o5_c0.3 → outlier_z-score → robust → cubic15 → z-score → 训练` |
| widar_123 | `butterworth_o5_c0.3+outlier_z-score+robust+z-score+nearest15` | `butterworth_o5_c0.3 → outlier_z-score → robust → nearest15 → z-score → 训练` |
| widar_124 | `butterworth_o5_c0.3+outlier_z-score+robust+z-score+decimate15` | `butterworth_o5_c0.3 → outlier_z-score → robust → decimate15 → z-score → 训练` |
| widar_125 | `butterworth_o5_c0.3+outlier_z-score+robust+min-max+linear15` | `butterworth_o5_c0.3 → outlier_z-score → robust → min-max → linear15 → 训练` |
| widar_126 | `butterworth_o5_c0.3+outlier_z-score+robust+min-max+cubic15` | `butterworth_o5_c0.3 → outlier_z-score → robust → min-max → cubic15 → 训练` |
| widar_127 | `butterworth_o5_c0.3+outlier_z-score+robust+min-max+nearest15` | `butterworth_o5_c0.3 → outlier_z-score → robust → min-max → nearest15 → 训练` |
| widar_128 | `butterworth_o5_c0.3+outlier_z-score+robust+min-max+decimate15` | `butterworth_o5_c0.3 → outlier_z-score → robust → min-max → decimate15 → 训练` |

### 公共前缀 17：savgol_w7_p3+iqr+linear

公共缓存结果：`savgol_w7_p3 → iqr → linear`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_129 | `savgol_w7_p3+iqr+linear+z-score+linear15` | `savgol_w7_p3 → iqr → linear → linear15 → z-score → 训练` |
| widar_130 | `savgol_w7_p3+iqr+linear+z-score+cubic15` | `savgol_w7_p3 → iqr → linear → cubic15 → z-score → 训练` |
| widar_131 | `savgol_w7_p3+iqr+linear+z-score+nearest15` | `savgol_w7_p3 → iqr → linear → nearest15 → z-score → 训练` |
| widar_132 | `savgol_w7_p3+iqr+linear+z-score+decimate15` | `savgol_w7_p3 → iqr → linear → decimate15 → z-score → 训练` |
| widar_133 | `savgol_w7_p3+iqr+linear+min-max+linear15` | `savgol_w7_p3 → iqr → linear → min-max → linear15 → 训练` |
| widar_134 | `savgol_w7_p3+iqr+linear+min-max+cubic15` | `savgol_w7_p3 → iqr → linear → min-max → cubic15 → 训练` |
| widar_135 | `savgol_w7_p3+iqr+linear+min-max+nearest15` | `savgol_w7_p3 → iqr → linear → min-max → nearest15 → 训练` |
| widar_136 | `savgol_w7_p3+iqr+linear+min-max+decimate15` | `savgol_w7_p3 → iqr → linear → min-max → decimate15 → 训练` |

### 公共前缀 18：savgol_w7_p3+iqr+polynomial_d3

公共缓存结果：`savgol_w7_p3 → iqr → polynomial_d3`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_137 | `savgol_w7_p3+iqr+polynomial_d3+z-score+linear15` | `savgol_w7_p3 → iqr → polynomial_d3 → linear15 → z-score → 训练` |
| widar_138 | `savgol_w7_p3+iqr+polynomial_d3+z-score+cubic15` | `savgol_w7_p3 → iqr → polynomial_d3 → cubic15 → z-score → 训练` |
| widar_139 | `savgol_w7_p3+iqr+polynomial_d3+z-score+nearest15` | `savgol_w7_p3 → iqr → polynomial_d3 → nearest15 → z-score → 训练` |
| widar_140 | `savgol_w7_p3+iqr+polynomial_d3+z-score+decimate15` | `savgol_w7_p3 → iqr → polynomial_d3 → decimate15 → z-score → 训练` |
| widar_141 | `savgol_w7_p3+iqr+polynomial_d3+min-max+linear15` | `savgol_w7_p3 → iqr → polynomial_d3 → min-max → linear15 → 训练` |
| widar_142 | `savgol_w7_p3+iqr+polynomial_d3+min-max+cubic15` | `savgol_w7_p3 → iqr → polynomial_d3 → min-max → cubic15 → 训练` |
| widar_143 | `savgol_w7_p3+iqr+polynomial_d3+min-max+nearest15` | `savgol_w7_p3 → iqr → polynomial_d3 → min-max → nearest15 → 训练` |
| widar_144 | `savgol_w7_p3+iqr+polynomial_d3+min-max+decimate15` | `savgol_w7_p3 → iqr → polynomial_d3 → min-max → decimate15 → 训练` |

### 公共前缀 19：savgol_w7_p3+iqr+stc

公共缓存结果：`savgol_w7_p3 → iqr → stc`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_145 | `savgol_w7_p3+iqr+stc+z-score+linear15` | `savgol_w7_p3 → iqr → stc → linear15 → z-score → 训练` |
| widar_146 | `savgol_w7_p3+iqr+stc+z-score+cubic15` | `savgol_w7_p3 → iqr → stc → cubic15 → z-score → 训练` |
| widar_147 | `savgol_w7_p3+iqr+stc+z-score+nearest15` | `savgol_w7_p3 → iqr → stc → nearest15 → z-score → 训练` |
| widar_148 | `savgol_w7_p3+iqr+stc+z-score+decimate15` | `savgol_w7_p3 → iqr → stc → decimate15 → z-score → 训练` |
| widar_149 | `savgol_w7_p3+iqr+stc+min-max+linear15` | `savgol_w7_p3 → iqr → stc → min-max → linear15 → 训练` |
| widar_150 | `savgol_w7_p3+iqr+stc+min-max+cubic15` | `savgol_w7_p3 → iqr → stc → min-max → cubic15 → 训练` |
| widar_151 | `savgol_w7_p3+iqr+stc+min-max+nearest15` | `savgol_w7_p3 → iqr → stc → min-max → nearest15 → 训练` |
| widar_152 | `savgol_w7_p3+iqr+stc+min-max+decimate15` | `savgol_w7_p3 → iqr → stc → min-max → decimate15 → 训练` |

### 公共前缀 20：savgol_w7_p3+iqr+robust

公共缓存结果：`savgol_w7_p3 → iqr → robust`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_153 | `savgol_w7_p3+iqr+robust+z-score+linear15` | `savgol_w7_p3 → iqr → robust → linear15 → z-score → 训练` |
| widar_154 | `savgol_w7_p3+iqr+robust+z-score+cubic15` | `savgol_w7_p3 → iqr → robust → cubic15 → z-score → 训练` |
| widar_155 | `savgol_w7_p3+iqr+robust+z-score+nearest15` | `savgol_w7_p3 → iqr → robust → nearest15 → z-score → 训练` |
| widar_156 | `savgol_w7_p3+iqr+robust+z-score+decimate15` | `savgol_w7_p3 → iqr → robust → decimate15 → z-score → 训练` |
| widar_157 | `savgol_w7_p3+iqr+robust+min-max+linear15` | `savgol_w7_p3 → iqr → robust → min-max → linear15 → 训练` |
| widar_158 | `savgol_w7_p3+iqr+robust+min-max+cubic15` | `savgol_w7_p3 → iqr → robust → min-max → cubic15 → 训练` |
| widar_159 | `savgol_w7_p3+iqr+robust+min-max+nearest15` | `savgol_w7_p3 → iqr → robust → min-max → nearest15 → 训练` |
| widar_160 | `savgol_w7_p3+iqr+robust+min-max+decimate15` | `savgol_w7_p3 → iqr → robust → min-max → decimate15 → 训练` |

### 公共前缀 21：savgol_w7_p3+outlier_z-score+linear

公共缓存结果：`savgol_w7_p3 → outlier_z-score → linear`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_161 | `savgol_w7_p3+outlier_z-score+linear+z-score+linear15` | `savgol_w7_p3 → outlier_z-score → linear → linear15 → z-score → 训练` |
| widar_162 | `savgol_w7_p3+outlier_z-score+linear+z-score+cubic15` | `savgol_w7_p3 → outlier_z-score → linear → cubic15 → z-score → 训练` |
| widar_163 | `savgol_w7_p3+outlier_z-score+linear+z-score+nearest15` | `savgol_w7_p3 → outlier_z-score → linear → nearest15 → z-score → 训练` |
| widar_164 | `savgol_w7_p3+outlier_z-score+linear+z-score+decimate15` | `savgol_w7_p3 → outlier_z-score → linear → decimate15 → z-score → 训练` |
| widar_165 | `savgol_w7_p3+outlier_z-score+linear+min-max+linear15` | `savgol_w7_p3 → outlier_z-score → linear → min-max → linear15 → 训练` |
| widar_166 | `savgol_w7_p3+outlier_z-score+linear+min-max+cubic15` | `savgol_w7_p3 → outlier_z-score → linear → min-max → cubic15 → 训练` |
| widar_167 | `savgol_w7_p3+outlier_z-score+linear+min-max+nearest15` | `savgol_w7_p3 → outlier_z-score → linear → min-max → nearest15 → 训练` |
| widar_168 | `savgol_w7_p3+outlier_z-score+linear+min-max+decimate15` | `savgol_w7_p3 → outlier_z-score → linear → min-max → decimate15 → 训练` |

### 公共前缀 22：savgol_w7_p3+outlier_z-score+polynomial_d3

公共缓存结果：`savgol_w7_p3 → outlier_z-score → polynomial_d3`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_169 | `savgol_w7_p3+outlier_z-score+polynomial_d3+z-score+linear15` | `savgol_w7_p3 → outlier_z-score → polynomial_d3 → linear15 → z-score → 训练` |
| widar_170 | `savgol_w7_p3+outlier_z-score+polynomial_d3+z-score+cubic15` | `savgol_w7_p3 → outlier_z-score → polynomial_d3 → cubic15 → z-score → 训练` |
| widar_171 | `savgol_w7_p3+outlier_z-score+polynomial_d3+z-score+nearest15` | `savgol_w7_p3 → outlier_z-score → polynomial_d3 → nearest15 → z-score → 训练` |
| widar_172 | `savgol_w7_p3+outlier_z-score+polynomial_d3+z-score+decimate15` | `savgol_w7_p3 → outlier_z-score → polynomial_d3 → decimate15 → z-score → 训练` |
| widar_173 | `savgol_w7_p3+outlier_z-score+polynomial_d3+min-max+linear15` | `savgol_w7_p3 → outlier_z-score → polynomial_d3 → min-max → linear15 → 训练` |
| widar_174 | `savgol_w7_p3+outlier_z-score+polynomial_d3+min-max+cubic15` | `savgol_w7_p3 → outlier_z-score → polynomial_d3 → min-max → cubic15 → 训练` |
| widar_175 | `savgol_w7_p3+outlier_z-score+polynomial_d3+min-max+nearest15` | `savgol_w7_p3 → outlier_z-score → polynomial_d3 → min-max → nearest15 → 训练` |
| widar_176 | `savgol_w7_p3+outlier_z-score+polynomial_d3+min-max+decimate15` | `savgol_w7_p3 → outlier_z-score → polynomial_d3 → min-max → decimate15 → 训练` |

### 公共前缀 23：savgol_w7_p3+outlier_z-score+stc

公共缓存结果：`savgol_w7_p3 → outlier_z-score → stc`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_177 | `savgol_w7_p3+outlier_z-score+stc+z-score+linear15` | `savgol_w7_p3 → outlier_z-score → stc → linear15 → z-score → 训练` |
| widar_178 | `savgol_w7_p3+outlier_z-score+stc+z-score+cubic15` | `savgol_w7_p3 → outlier_z-score → stc → cubic15 → z-score → 训练` |
| widar_179 | `savgol_w7_p3+outlier_z-score+stc+z-score+nearest15` | `savgol_w7_p3 → outlier_z-score → stc → nearest15 → z-score → 训练` |
| widar_180 | `savgol_w7_p3+outlier_z-score+stc+z-score+decimate15` | `savgol_w7_p3 → outlier_z-score → stc → decimate15 → z-score → 训练` |
| widar_181 | `savgol_w7_p3+outlier_z-score+stc+min-max+linear15` | `savgol_w7_p3 → outlier_z-score → stc → min-max → linear15 → 训练` |
| widar_182 | `savgol_w7_p3+outlier_z-score+stc+min-max+cubic15` | `savgol_w7_p3 → outlier_z-score → stc → min-max → cubic15 → 训练` |
| widar_183 | `savgol_w7_p3+outlier_z-score+stc+min-max+nearest15` | `savgol_w7_p3 → outlier_z-score → stc → min-max → nearest15 → 训练` |
| widar_184 | `savgol_w7_p3+outlier_z-score+stc+min-max+decimate15` | `savgol_w7_p3 → outlier_z-score → stc → min-max → decimate15 → 训练` |

### 公共前缀 24：savgol_w7_p3+outlier_z-score+robust

公共缓存结果：`savgol_w7_p3 → outlier_z-score → robust`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_185 | `savgol_w7_p3+outlier_z-score+robust+z-score+linear15` | `savgol_w7_p3 → outlier_z-score → robust → linear15 → z-score → 训练` |
| widar_186 | `savgol_w7_p3+outlier_z-score+robust+z-score+cubic15` | `savgol_w7_p3 → outlier_z-score → robust → cubic15 → z-score → 训练` |
| widar_187 | `savgol_w7_p3+outlier_z-score+robust+z-score+nearest15` | `savgol_w7_p3 → outlier_z-score → robust → nearest15 → z-score → 训练` |
| widar_188 | `savgol_w7_p3+outlier_z-score+robust+z-score+decimate15` | `savgol_w7_p3 → outlier_z-score → robust → decimate15 → z-score → 训练` |
| widar_189 | `savgol_w7_p3+outlier_z-score+robust+min-max+linear15` | `savgol_w7_p3 → outlier_z-score → robust → min-max → linear15 → 训练` |
| widar_190 | `savgol_w7_p3+outlier_z-score+robust+min-max+cubic15` | `savgol_w7_p3 → outlier_z-score → robust → min-max → cubic15 → 训练` |
| widar_191 | `savgol_w7_p3+outlier_z-score+robust+min-max+nearest15` | `savgol_w7_p3 → outlier_z-score → robust → min-max → nearest15 → 训练` |
| widar_192 | `savgol_w7_p3+outlier_z-score+robust+min-max+decimate15` | `savgol_w7_p3 → outlier_z-score → robust → min-max → decimate15 → 训练` |

### 公共前缀 25：bandpass_0.5-50+iqr+linear

公共缓存结果：`bandpass_0.5-50 → iqr → linear`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_193 | `bandpass_0.5-50+iqr+linear+z-score+linear15` | `bandpass_0.5-50 → iqr → linear → linear15 → z-score → 训练` |
| widar_194 | `bandpass_0.5-50+iqr+linear+z-score+cubic15` | `bandpass_0.5-50 → iqr → linear → cubic15 → z-score → 训练` |
| widar_195 | `bandpass_0.5-50+iqr+linear+z-score+nearest15` | `bandpass_0.5-50 → iqr → linear → nearest15 → z-score → 训练` |
| widar_196 | `bandpass_0.5-50+iqr+linear+z-score+decimate15` | `bandpass_0.5-50 → iqr → linear → decimate15 → z-score → 训练` |
| widar_197 | `bandpass_0.5-50+iqr+linear+min-max+linear15` | `bandpass_0.5-50 → iqr → linear → min-max → linear15 → 训练` |
| widar_198 | `bandpass_0.5-50+iqr+linear+min-max+cubic15` | `bandpass_0.5-50 → iqr → linear → min-max → cubic15 → 训练` |
| widar_199 | `bandpass_0.5-50+iqr+linear+min-max+nearest15` | `bandpass_0.5-50 → iqr → linear → min-max → nearest15 → 训练` |
| widar_200 | `bandpass_0.5-50+iqr+linear+min-max+decimate15` | `bandpass_0.5-50 → iqr → linear → min-max → decimate15 → 训练` |

### 公共前缀 26：bandpass_0.5-50+iqr+polynomial_d3

公共缓存结果：`bandpass_0.5-50 → iqr → polynomial_d3`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_201 | `bandpass_0.5-50+iqr+polynomial_d3+z-score+linear15` | `bandpass_0.5-50 → iqr → polynomial_d3 → linear15 → z-score → 训练` |
| widar_202 | `bandpass_0.5-50+iqr+polynomial_d3+z-score+cubic15` | `bandpass_0.5-50 → iqr → polynomial_d3 → cubic15 → z-score → 训练` |
| widar_203 | `bandpass_0.5-50+iqr+polynomial_d3+z-score+nearest15` | `bandpass_0.5-50 → iqr → polynomial_d3 → nearest15 → z-score → 训练` |
| widar_204 | `bandpass_0.5-50+iqr+polynomial_d3+z-score+decimate15` | `bandpass_0.5-50 → iqr → polynomial_d3 → decimate15 → z-score → 训练` |
| widar_205 | `bandpass_0.5-50+iqr+polynomial_d3+min-max+linear15` | `bandpass_0.5-50 → iqr → polynomial_d3 → min-max → linear15 → 训练` |
| widar_206 | `bandpass_0.5-50+iqr+polynomial_d3+min-max+cubic15` | `bandpass_0.5-50 → iqr → polynomial_d3 → min-max → cubic15 → 训练` |
| widar_207 | `bandpass_0.5-50+iqr+polynomial_d3+min-max+nearest15` | `bandpass_0.5-50 → iqr → polynomial_d3 → min-max → nearest15 → 训练` |
| widar_208 | `bandpass_0.5-50+iqr+polynomial_d3+min-max+decimate15` | `bandpass_0.5-50 → iqr → polynomial_d3 → min-max → decimate15 → 训练` |

### 公共前缀 27：bandpass_0.5-50+iqr+stc

公共缓存结果：`bandpass_0.5-50 → iqr → stc`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_209 | `bandpass_0.5-50+iqr+stc+z-score+linear15` | `bandpass_0.5-50 → iqr → stc → linear15 → z-score → 训练` |
| widar_210 | `bandpass_0.5-50+iqr+stc+z-score+cubic15` | `bandpass_0.5-50 → iqr → stc → cubic15 → z-score → 训练` |
| widar_211 | `bandpass_0.5-50+iqr+stc+z-score+nearest15` | `bandpass_0.5-50 → iqr → stc → nearest15 → z-score → 训练` |
| widar_212 | `bandpass_0.5-50+iqr+stc+z-score+decimate15` | `bandpass_0.5-50 → iqr → stc → decimate15 → z-score → 训练` |
| widar_213 | `bandpass_0.5-50+iqr+stc+min-max+linear15` | `bandpass_0.5-50 → iqr → stc → min-max → linear15 → 训练` |
| widar_214 | `bandpass_0.5-50+iqr+stc+min-max+cubic15` | `bandpass_0.5-50 → iqr → stc → min-max → cubic15 → 训练` |
| widar_215 | `bandpass_0.5-50+iqr+stc+min-max+nearest15` | `bandpass_0.5-50 → iqr → stc → min-max → nearest15 → 训练` |
| widar_216 | `bandpass_0.5-50+iqr+stc+min-max+decimate15` | `bandpass_0.5-50 → iqr → stc → min-max → decimate15 → 训练` |

### 公共前缀 28：bandpass_0.5-50+iqr+robust

公共缓存结果：`bandpass_0.5-50 → iqr → robust`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_217 | `bandpass_0.5-50+iqr+robust+z-score+linear15` | `bandpass_0.5-50 → iqr → robust → linear15 → z-score → 训练` |
| widar_218 | `bandpass_0.5-50+iqr+robust+z-score+cubic15` | `bandpass_0.5-50 → iqr → robust → cubic15 → z-score → 训练` |
| widar_219 | `bandpass_0.5-50+iqr+robust+z-score+nearest15` | `bandpass_0.5-50 → iqr → robust → nearest15 → z-score → 训练` |
| widar_220 | `bandpass_0.5-50+iqr+robust+z-score+decimate15` | `bandpass_0.5-50 → iqr → robust → decimate15 → z-score → 训练` |
| widar_221 | `bandpass_0.5-50+iqr+robust+min-max+linear15` | `bandpass_0.5-50 → iqr → robust → min-max → linear15 → 训练` |
| widar_222 | `bandpass_0.5-50+iqr+robust+min-max+cubic15` | `bandpass_0.5-50 → iqr → robust → min-max → cubic15 → 训练` |
| widar_223 | `bandpass_0.5-50+iqr+robust+min-max+nearest15` | `bandpass_0.5-50 → iqr → robust → min-max → nearest15 → 训练` |
| widar_224 | `bandpass_0.5-50+iqr+robust+min-max+decimate15` | `bandpass_0.5-50 → iqr → robust → min-max → decimate15 → 训练` |

### 公共前缀 29：bandpass_0.5-50+outlier_z-score+linear

公共缓存结果：`bandpass_0.5-50 → outlier_z-score → linear`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_225 | `bandpass_0.5-50+outlier_z-score+linear+z-score+linear15` | `bandpass_0.5-50 → outlier_z-score → linear → linear15 → z-score → 训练` |
| widar_226 | `bandpass_0.5-50+outlier_z-score+linear+z-score+cubic15` | `bandpass_0.5-50 → outlier_z-score → linear → cubic15 → z-score → 训练` |
| widar_227 | `bandpass_0.5-50+outlier_z-score+linear+z-score+nearest15` | `bandpass_0.5-50 → outlier_z-score → linear → nearest15 → z-score → 训练` |
| widar_228 | `bandpass_0.5-50+outlier_z-score+linear+z-score+decimate15` | `bandpass_0.5-50 → outlier_z-score → linear → decimate15 → z-score → 训练` |
| widar_229 | `bandpass_0.5-50+outlier_z-score+linear+min-max+linear15` | `bandpass_0.5-50 → outlier_z-score → linear → min-max → linear15 → 训练` |
| widar_230 | `bandpass_0.5-50+outlier_z-score+linear+min-max+cubic15` | `bandpass_0.5-50 → outlier_z-score → linear → min-max → cubic15 → 训练` |
| widar_231 | `bandpass_0.5-50+outlier_z-score+linear+min-max+nearest15` | `bandpass_0.5-50 → outlier_z-score → linear → min-max → nearest15 → 训练` |
| widar_232 | `bandpass_0.5-50+outlier_z-score+linear+min-max+decimate15` | `bandpass_0.5-50 → outlier_z-score → linear → min-max → decimate15 → 训练` |

### 公共前缀 30：bandpass_0.5-50+outlier_z-score+polynomial_d3

公共缓存结果：`bandpass_0.5-50 → outlier_z-score → polynomial_d3`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_233 | `bandpass_0.5-50+outlier_z-score+polynomial_d3+z-score+linear15` | `bandpass_0.5-50 → outlier_z-score → polynomial_d3 → linear15 → z-score → 训练` |
| widar_234 | `bandpass_0.5-50+outlier_z-score+polynomial_d3+z-score+cubic15` | `bandpass_0.5-50 → outlier_z-score → polynomial_d3 → cubic15 → z-score → 训练` |
| widar_235 | `bandpass_0.5-50+outlier_z-score+polynomial_d3+z-score+nearest15` | `bandpass_0.5-50 → outlier_z-score → polynomial_d3 → nearest15 → z-score → 训练` |
| widar_236 | `bandpass_0.5-50+outlier_z-score+polynomial_d3+z-score+decimate15` | `bandpass_0.5-50 → outlier_z-score → polynomial_d3 → decimate15 → z-score → 训练` |
| widar_237 | `bandpass_0.5-50+outlier_z-score+polynomial_d3+min-max+linear15` | `bandpass_0.5-50 → outlier_z-score → polynomial_d3 → min-max → linear15 → 训练` |
| widar_238 | `bandpass_0.5-50+outlier_z-score+polynomial_d3+min-max+cubic15` | `bandpass_0.5-50 → outlier_z-score → polynomial_d3 → min-max → cubic15 → 训练` |
| widar_239 | `bandpass_0.5-50+outlier_z-score+polynomial_d3+min-max+nearest15` | `bandpass_0.5-50 → outlier_z-score → polynomial_d3 → min-max → nearest15 → 训练` |
| widar_240 | `bandpass_0.5-50+outlier_z-score+polynomial_d3+min-max+decimate15` | `bandpass_0.5-50 → outlier_z-score → polynomial_d3 → min-max → decimate15 → 训练` |

### 公共前缀 31：bandpass_0.5-50+outlier_z-score+stc

公共缓存结果：`bandpass_0.5-50 → outlier_z-score → stc`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_241 | `bandpass_0.5-50+outlier_z-score+stc+z-score+linear15` | `bandpass_0.5-50 → outlier_z-score → stc → linear15 → z-score → 训练` |
| widar_242 | `bandpass_0.5-50+outlier_z-score+stc+z-score+cubic15` | `bandpass_0.5-50 → outlier_z-score → stc → cubic15 → z-score → 训练` |
| widar_243 | `bandpass_0.5-50+outlier_z-score+stc+z-score+nearest15` | `bandpass_0.5-50 → outlier_z-score → stc → nearest15 → z-score → 训练` |
| widar_244 | `bandpass_0.5-50+outlier_z-score+stc+z-score+decimate15` | `bandpass_0.5-50 → outlier_z-score → stc → decimate15 → z-score → 训练` |
| widar_245 | `bandpass_0.5-50+outlier_z-score+stc+min-max+linear15` | `bandpass_0.5-50 → outlier_z-score → stc → min-max → linear15 → 训练` |
| widar_246 | `bandpass_0.5-50+outlier_z-score+stc+min-max+cubic15` | `bandpass_0.5-50 → outlier_z-score → stc → min-max → cubic15 → 训练` |
| widar_247 | `bandpass_0.5-50+outlier_z-score+stc+min-max+nearest15` | `bandpass_0.5-50 → outlier_z-score → stc → min-max → nearest15 → 训练` |
| widar_248 | `bandpass_0.5-50+outlier_z-score+stc+min-max+decimate15` | `bandpass_0.5-50 → outlier_z-score → stc → min-max → decimate15 → 训练` |

### 公共前缀 32：bandpass_0.5-50+outlier_z-score+robust

公共缓存结果：`bandpass_0.5-50 → outlier_z-score → robust`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_249 | `bandpass_0.5-50+outlier_z-score+robust+z-score+linear15` | `bandpass_0.5-50 → outlier_z-score → robust → linear15 → z-score → 训练` |
| widar_250 | `bandpass_0.5-50+outlier_z-score+robust+z-score+cubic15` | `bandpass_0.5-50 → outlier_z-score → robust → cubic15 → z-score → 训练` |
| widar_251 | `bandpass_0.5-50+outlier_z-score+robust+z-score+nearest15` | `bandpass_0.5-50 → outlier_z-score → robust → nearest15 → z-score → 训练` |
| widar_252 | `bandpass_0.5-50+outlier_z-score+robust+z-score+decimate15` | `bandpass_0.5-50 → outlier_z-score → robust → decimate15 → z-score → 训练` |
| widar_253 | `bandpass_0.5-50+outlier_z-score+robust+min-max+linear15` | `bandpass_0.5-50 → outlier_z-score → robust → min-max → linear15 → 训练` |
| widar_254 | `bandpass_0.5-50+outlier_z-score+robust+min-max+cubic15` | `bandpass_0.5-50 → outlier_z-score → robust → min-max → cubic15 → 训练` |
| widar_255 | `bandpass_0.5-50+outlier_z-score+robust+min-max+nearest15` | `bandpass_0.5-50 → outlier_z-score → robust → min-max → nearest15 → 训练` |
| widar_256 | `bandpass_0.5-50+outlier_z-score+robust+min-max+decimate15` | `bandpass_0.5-50 → outlier_z-score → robust → min-max → decimate15 → 训练` |

### 公共前缀 33：hampel_w5_s3+iqr+linear

公共缓存结果：`hampel_w5_s3 → iqr → linear`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_257 | `hampel_w5_s3+iqr+linear+z-score+linear15` | `hampel_w5_s3 → iqr → linear → linear15 → z-score → 训练` |
| widar_258 | `hampel_w5_s3+iqr+linear+z-score+cubic15` | `hampel_w5_s3 → iqr → linear → cubic15 → z-score → 训练` |
| widar_259 | `hampel_w5_s3+iqr+linear+z-score+nearest15` | `hampel_w5_s3 → iqr → linear → nearest15 → z-score → 训练` |
| widar_260 | `hampel_w5_s3+iqr+linear+z-score+decimate15` | `hampel_w5_s3 → iqr → linear → decimate15 → z-score → 训练` |
| widar_261 | `hampel_w5_s3+iqr+linear+min-max+linear15` | `hampel_w5_s3 → iqr → linear → min-max → linear15 → 训练` |
| widar_262 | `hampel_w5_s3+iqr+linear+min-max+cubic15` | `hampel_w5_s3 → iqr → linear → min-max → cubic15 → 训练` |
| widar_263 | `hampel_w5_s3+iqr+linear+min-max+nearest15` | `hampel_w5_s3 → iqr → linear → min-max → nearest15 → 训练` |
| widar_264 | `hampel_w5_s3+iqr+linear+min-max+decimate15` | `hampel_w5_s3 → iqr → linear → min-max → decimate15 → 训练` |

### 公共前缀 34：hampel_w5_s3+iqr+polynomial_d3

公共缓存结果：`hampel_w5_s3 → iqr → polynomial_d3`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_265 | `hampel_w5_s3+iqr+polynomial_d3+z-score+linear15` | `hampel_w5_s3 → iqr → polynomial_d3 → linear15 → z-score → 训练` |
| widar_266 | `hampel_w5_s3+iqr+polynomial_d3+z-score+cubic15` | `hampel_w5_s3 → iqr → polynomial_d3 → cubic15 → z-score → 训练` |
| widar_267 | `hampel_w5_s3+iqr+polynomial_d3+z-score+nearest15` | `hampel_w5_s3 → iqr → polynomial_d3 → nearest15 → z-score → 训练` |
| widar_268 | `hampel_w5_s3+iqr+polynomial_d3+z-score+decimate15` | `hampel_w5_s3 → iqr → polynomial_d3 → decimate15 → z-score → 训练` |
| widar_269 | `hampel_w5_s3+iqr+polynomial_d3+min-max+linear15` | `hampel_w5_s3 → iqr → polynomial_d3 → min-max → linear15 → 训练` |
| widar_270 | `hampel_w5_s3+iqr+polynomial_d3+min-max+cubic15` | `hampel_w5_s3 → iqr → polynomial_d3 → min-max → cubic15 → 训练` |
| widar_271 | `hampel_w5_s3+iqr+polynomial_d3+min-max+nearest15` | `hampel_w5_s3 → iqr → polynomial_d3 → min-max → nearest15 → 训练` |
| widar_272 | `hampel_w5_s3+iqr+polynomial_d3+min-max+decimate15` | `hampel_w5_s3 → iqr → polynomial_d3 → min-max → decimate15 → 训练` |

### 公共前缀 35：hampel_w5_s3+iqr+stc

公共缓存结果：`hampel_w5_s3 → iqr → stc`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_273 | `hampel_w5_s3+iqr+stc+z-score+linear15` | `hampel_w5_s3 → iqr → stc → linear15 → z-score → 训练` |
| widar_274 | `hampel_w5_s3+iqr+stc+z-score+cubic15` | `hampel_w5_s3 → iqr → stc → cubic15 → z-score → 训练` |
| widar_275 | `hampel_w5_s3+iqr+stc+z-score+nearest15` | `hampel_w5_s3 → iqr → stc → nearest15 → z-score → 训练` |
| widar_276 | `hampel_w5_s3+iqr+stc+z-score+decimate15` | `hampel_w5_s3 → iqr → stc → decimate15 → z-score → 训练` |
| widar_277 | `hampel_w5_s3+iqr+stc+min-max+linear15` | `hampel_w5_s3 → iqr → stc → min-max → linear15 → 训练` |
| widar_278 | `hampel_w5_s3+iqr+stc+min-max+cubic15` | `hampel_w5_s3 → iqr → stc → min-max → cubic15 → 训练` |
| widar_279 | `hampel_w5_s3+iqr+stc+min-max+nearest15` | `hampel_w5_s3 → iqr → stc → min-max → nearest15 → 训练` |
| widar_280 | `hampel_w5_s3+iqr+stc+min-max+decimate15` | `hampel_w5_s3 → iqr → stc → min-max → decimate15 → 训练` |

### 公共前缀 36：hampel_w5_s3+iqr+robust

公共缓存结果：`hampel_w5_s3 → iqr → robust`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_281 | `hampel_w5_s3+iqr+robust+z-score+linear15` | `hampel_w5_s3 → iqr → robust → linear15 → z-score → 训练` |
| widar_282 | `hampel_w5_s3+iqr+robust+z-score+cubic15` | `hampel_w5_s3 → iqr → robust → cubic15 → z-score → 训练` |
| widar_283 | `hampel_w5_s3+iqr+robust+z-score+nearest15` | `hampel_w5_s3 → iqr → robust → nearest15 → z-score → 训练` |
| widar_284 | `hampel_w5_s3+iqr+robust+z-score+decimate15` | `hampel_w5_s3 → iqr → robust → decimate15 → z-score → 训练` |
| widar_285 | `hampel_w5_s3+iqr+robust+min-max+linear15` | `hampel_w5_s3 → iqr → robust → min-max → linear15 → 训练` |
| widar_286 | `hampel_w5_s3+iqr+robust+min-max+cubic15` | `hampel_w5_s3 → iqr → robust → min-max → cubic15 → 训练` |
| widar_287 | `hampel_w5_s3+iqr+robust+min-max+nearest15` | `hampel_w5_s3 → iqr → robust → min-max → nearest15 → 训练` |
| widar_288 | `hampel_w5_s3+iqr+robust+min-max+decimate15` | `hampel_w5_s3 → iqr → robust → min-max → decimate15 → 训练` |

### 公共前缀 37：hampel_w5_s3+outlier_z-score+linear

公共缓存结果：`hampel_w5_s3 → outlier_z-score → linear`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_289 | `hampel_w5_s3+outlier_z-score+linear+z-score+linear15` | `hampel_w5_s3 → outlier_z-score → linear → linear15 → z-score → 训练` |
| widar_290 | `hampel_w5_s3+outlier_z-score+linear+z-score+cubic15` | `hampel_w5_s3 → outlier_z-score → linear → cubic15 → z-score → 训练` |
| widar_291 | `hampel_w5_s3+outlier_z-score+linear+z-score+nearest15` | `hampel_w5_s3 → outlier_z-score → linear → nearest15 → z-score → 训练` |
| widar_292 | `hampel_w5_s3+outlier_z-score+linear+z-score+decimate15` | `hampel_w5_s3 → outlier_z-score → linear → decimate15 → z-score → 训练` |
| widar_293 | `hampel_w5_s3+outlier_z-score+linear+min-max+linear15` | `hampel_w5_s3 → outlier_z-score → linear → min-max → linear15 → 训练` |
| widar_294 | `hampel_w5_s3+outlier_z-score+linear+min-max+cubic15` | `hampel_w5_s3 → outlier_z-score → linear → min-max → cubic15 → 训练` |
| widar_295 | `hampel_w5_s3+outlier_z-score+linear+min-max+nearest15` | `hampel_w5_s3 → outlier_z-score → linear → min-max → nearest15 → 训练` |
| widar_296 | `hampel_w5_s3+outlier_z-score+linear+min-max+decimate15` | `hampel_w5_s3 → outlier_z-score → linear → min-max → decimate15 → 训练` |

### 公共前缀 38：hampel_w5_s3+outlier_z-score+polynomial_d3

公共缓存结果：`hampel_w5_s3 → outlier_z-score → polynomial_d3`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_297 | `hampel_w5_s3+outlier_z-score+polynomial_d3+z-score+linear15` | `hampel_w5_s3 → outlier_z-score → polynomial_d3 → linear15 → z-score → 训练` |
| widar_298 | `hampel_w5_s3+outlier_z-score+polynomial_d3+z-score+cubic15` | `hampel_w5_s3 → outlier_z-score → polynomial_d3 → cubic15 → z-score → 训练` |
| widar_299 | `hampel_w5_s3+outlier_z-score+polynomial_d3+z-score+nearest15` | `hampel_w5_s3 → outlier_z-score → polynomial_d3 → nearest15 → z-score → 训练` |
| widar_300 | `hampel_w5_s3+outlier_z-score+polynomial_d3+z-score+decimate15` | `hampel_w5_s3 → outlier_z-score → polynomial_d3 → decimate15 → z-score → 训练` |
| widar_301 | `hampel_w5_s3+outlier_z-score+polynomial_d3+min-max+linear15` | `hampel_w5_s3 → outlier_z-score → polynomial_d3 → min-max → linear15 → 训练` |
| widar_302 | `hampel_w5_s3+outlier_z-score+polynomial_d3+min-max+cubic15` | `hampel_w5_s3 → outlier_z-score → polynomial_d3 → min-max → cubic15 → 训练` |
| widar_303 | `hampel_w5_s3+outlier_z-score+polynomial_d3+min-max+nearest15` | `hampel_w5_s3 → outlier_z-score → polynomial_d3 → min-max → nearest15 → 训练` |
| widar_304 | `hampel_w5_s3+outlier_z-score+polynomial_d3+min-max+decimate15` | `hampel_w5_s3 → outlier_z-score → polynomial_d3 → min-max → decimate15 → 训练` |

### 公共前缀 39：hampel_w5_s3+outlier_z-score+stc

公共缓存结果：`hampel_w5_s3 → outlier_z-score → stc`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_305 | `hampel_w5_s3+outlier_z-score+stc+z-score+linear15` | `hampel_w5_s3 → outlier_z-score → stc → linear15 → z-score → 训练` |
| widar_306 | `hampel_w5_s3+outlier_z-score+stc+z-score+cubic15` | `hampel_w5_s3 → outlier_z-score → stc → cubic15 → z-score → 训练` |
| widar_307 | `hampel_w5_s3+outlier_z-score+stc+z-score+nearest15` | `hampel_w5_s3 → outlier_z-score → stc → nearest15 → z-score → 训练` |
| widar_308 | `hampel_w5_s3+outlier_z-score+stc+z-score+decimate15` | `hampel_w5_s3 → outlier_z-score → stc → decimate15 → z-score → 训练` |
| widar_309 | `hampel_w5_s3+outlier_z-score+stc+min-max+linear15` | `hampel_w5_s3 → outlier_z-score → stc → min-max → linear15 → 训练` |
| widar_310 | `hampel_w5_s3+outlier_z-score+stc+min-max+cubic15` | `hampel_w5_s3 → outlier_z-score → stc → min-max → cubic15 → 训练` |
| widar_311 | `hampel_w5_s3+outlier_z-score+stc+min-max+nearest15` | `hampel_w5_s3 → outlier_z-score → stc → min-max → nearest15 → 训练` |
| widar_312 | `hampel_w5_s3+outlier_z-score+stc+min-max+decimate15` | `hampel_w5_s3 → outlier_z-score → stc → min-max → decimate15 → 训练` |

### 公共前缀 40：hampel_w5_s3+outlier_z-score+robust

公共缓存结果：`hampel_w5_s3 → outlier_z-score → robust`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| widar_313 | `hampel_w5_s3+outlier_z-score+robust+z-score+linear15` | `hampel_w5_s3 → outlier_z-score → robust → linear15 → z-score → 训练` |
| widar_314 | `hampel_w5_s3+outlier_z-score+robust+z-score+cubic15` | `hampel_w5_s3 → outlier_z-score → robust → cubic15 → z-score → 训练` |
| widar_315 | `hampel_w5_s3+outlier_z-score+robust+z-score+nearest15` | `hampel_w5_s3 → outlier_z-score → robust → nearest15 → z-score → 训练` |
| widar_316 | `hampel_w5_s3+outlier_z-score+robust+z-score+decimate15` | `hampel_w5_s3 → outlier_z-score → robust → decimate15 → z-score → 训练` |
| widar_317 | `hampel_w5_s3+outlier_z-score+robust+min-max+linear15` | `hampel_w5_s3 → outlier_z-score → robust → min-max → linear15 → 训练` |
| widar_318 | `hampel_w5_s3+outlier_z-score+robust+min-max+cubic15` | `hampel_w5_s3 → outlier_z-score → robust → min-max → cubic15 → 训练` |
| widar_319 | `hampel_w5_s3+outlier_z-score+robust+min-max+nearest15` | `hampel_w5_s3 → outlier_z-score → robust → min-max → nearest15 → 训练` |
| widar_320 | `hampel_w5_s3+outlier_z-score+robust+min-max+decimate15` | `hampel_w5_s3 → outlier_z-score → robust → min-max → decimate15 → 训练` |

## 执行数量变化

| 处理阶段 | 原流程执行次数 | 公共前缀优化后 |
|---|---:|---:|
| denoise | 320 | 40 |
| outliers | 320 | 40 |
| calibrate | 320 | 40 |
| min-max | 160 | 40 |
| z-score | 160 | 160（必须在各插值分支后执行） |
| interpolate | 320 | 320 |
| 模型训练 | 320 | 320 |

公共前缀应当一次只缓存一个；完成该前缀对应的 8 个组合后释放，再处理下一个公共前缀，以免 Widar 中间数据占用过多内存。
