# Gait 320 种 pipeline 的公共前缀优化执行顺序

本文档对应 `SDP/test_gait/full_test_gait_new.py`。脚本保留原 `full_test_gait.py` 的 `combo_id`、`combo_name` 和组合顺序，只复用重复的预处理结果。

## 核心规则

320 种组合按前三步划分为 40 个公共前缀：

```text
denoise → outliers → calibrate
5 × 2 × 4 = 40 个公共前缀
```

每个公共前缀对应 8 个组合。Gait + z-score 为了保留带正负号的归一化幅度和真实相位，实际执行顺序是：

```text
公共前缀 → interpolate → z-score → [normalized_amplitude, phase] → 训练
```

min-max 保持普通 pipeline 的执行顺序：

```text
公共前缀 → min-max → interpolate → [amplitude, phase] → 训练
```

每个公共前缀的复用结构如下：

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

> `combo_name` 继续使用配置字典的命名顺序；“实际执行流程”展示程序真正执行的顺序。

## 40 个公共前缀及完整 320 个组合

### 公共前缀 01：wavelet+iqr+linear

公共缓存结果：`wavelet → iqr → linear`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_001 | `wavelet+iqr+linear+z-score+linear15` | `wavelet → iqr → linear → linear15 → z-score → 训练` |
| gait_002 | `wavelet+iqr+linear+z-score+cubic15` | `wavelet → iqr → linear → cubic15 → z-score → 训练` |
| gait_003 | `wavelet+iqr+linear+z-score+nearest15` | `wavelet → iqr → linear → nearest15 → z-score → 训练` |
| gait_004 | `wavelet+iqr+linear+z-score+decimate15` | `wavelet → iqr → linear → decimate15 → z-score → 训练` |
| gait_005 | `wavelet+iqr+linear+min-max+linear15` | `wavelet → iqr → linear → min-max → linear15 → 训练` |
| gait_006 | `wavelet+iqr+linear+min-max+cubic15` | `wavelet → iqr → linear → min-max → cubic15 → 训练` |
| gait_007 | `wavelet+iqr+linear+min-max+nearest15` | `wavelet → iqr → linear → min-max → nearest15 → 训练` |
| gait_008 | `wavelet+iqr+linear+min-max+decimate15` | `wavelet → iqr → linear → min-max → decimate15 → 训练` |

### 公共前缀 02：wavelet+iqr+polynomial_d3

公共缓存结果：`wavelet → iqr → polynomial_d3`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_009 | `wavelet+iqr+polynomial_d3+z-score+linear15` | `wavelet → iqr → polynomial_d3 → linear15 → z-score → 训练` |
| gait_010 | `wavelet+iqr+polynomial_d3+z-score+cubic15` | `wavelet → iqr → polynomial_d3 → cubic15 → z-score → 训练` |
| gait_011 | `wavelet+iqr+polynomial_d3+z-score+nearest15` | `wavelet → iqr → polynomial_d3 → nearest15 → z-score → 训练` |
| gait_012 | `wavelet+iqr+polynomial_d3+z-score+decimate15` | `wavelet → iqr → polynomial_d3 → decimate15 → z-score → 训练` |
| gait_013 | `wavelet+iqr+polynomial_d3+min-max+linear15` | `wavelet → iqr → polynomial_d3 → min-max → linear15 → 训练` |
| gait_014 | `wavelet+iqr+polynomial_d3+min-max+cubic15` | `wavelet → iqr → polynomial_d3 → min-max → cubic15 → 训练` |
| gait_015 | `wavelet+iqr+polynomial_d3+min-max+nearest15` | `wavelet → iqr → polynomial_d3 → min-max → nearest15 → 训练` |
| gait_016 | `wavelet+iqr+polynomial_d3+min-max+decimate15` | `wavelet → iqr → polynomial_d3 → min-max → decimate15 → 训练` |

### 公共前缀 03：wavelet+iqr+stc

公共缓存结果：`wavelet → iqr → stc`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_017 | `wavelet+iqr+stc+z-score+linear15` | `wavelet → iqr → stc → linear15 → z-score → 训练` |
| gait_018 | `wavelet+iqr+stc+z-score+cubic15` | `wavelet → iqr → stc → cubic15 → z-score → 训练` |
| gait_019 | `wavelet+iqr+stc+z-score+nearest15` | `wavelet → iqr → stc → nearest15 → z-score → 训练` |
| gait_020 | `wavelet+iqr+stc+z-score+decimate15` | `wavelet → iqr → stc → decimate15 → z-score → 训练` |
| gait_021 | `wavelet+iqr+stc+min-max+linear15` | `wavelet → iqr → stc → min-max → linear15 → 训练` |
| gait_022 | `wavelet+iqr+stc+min-max+cubic15` | `wavelet → iqr → stc → min-max → cubic15 → 训练` |
| gait_023 | `wavelet+iqr+stc+min-max+nearest15` | `wavelet → iqr → stc → min-max → nearest15 → 训练` |
| gait_024 | `wavelet+iqr+stc+min-max+decimate15` | `wavelet → iqr → stc → min-max → decimate15 → 训练` |

### 公共前缀 04：wavelet+iqr+robust

公共缓存结果：`wavelet → iqr → robust`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_025 | `wavelet+iqr+robust+z-score+linear15` | `wavelet → iqr → robust → linear15 → z-score → 训练` |
| gait_026 | `wavelet+iqr+robust+z-score+cubic15` | `wavelet → iqr → robust → cubic15 → z-score → 训练` |
| gait_027 | `wavelet+iqr+robust+z-score+nearest15` | `wavelet → iqr → robust → nearest15 → z-score → 训练` |
| gait_028 | `wavelet+iqr+robust+z-score+decimate15` | `wavelet → iqr → robust → decimate15 → z-score → 训练` |
| gait_029 | `wavelet+iqr+robust+min-max+linear15` | `wavelet → iqr → robust → min-max → linear15 → 训练` |
| gait_030 | `wavelet+iqr+robust+min-max+cubic15` | `wavelet → iqr → robust → min-max → cubic15 → 训练` |
| gait_031 | `wavelet+iqr+robust+min-max+nearest15` | `wavelet → iqr → robust → min-max → nearest15 → 训练` |
| gait_032 | `wavelet+iqr+robust+min-max+decimate15` | `wavelet → iqr → robust → min-max → decimate15 → 训练` |

### 公共前缀 05：wavelet+outlier_z-score+linear

公共缓存结果：`wavelet → outlier_z-score → linear`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_033 | `wavelet+outlier_z-score+linear+z-score+linear15` | `wavelet → outlier_z-score → linear → linear15 → z-score → 训练` |
| gait_034 | `wavelet+outlier_z-score+linear+z-score+cubic15` | `wavelet → outlier_z-score → linear → cubic15 → z-score → 训练` |
| gait_035 | `wavelet+outlier_z-score+linear+z-score+nearest15` | `wavelet → outlier_z-score → linear → nearest15 → z-score → 训练` |
| gait_036 | `wavelet+outlier_z-score+linear+z-score+decimate15` | `wavelet → outlier_z-score → linear → decimate15 → z-score → 训练` |
| gait_037 | `wavelet+outlier_z-score+linear+min-max+linear15` | `wavelet → outlier_z-score → linear → min-max → linear15 → 训练` |
| gait_038 | `wavelet+outlier_z-score+linear+min-max+cubic15` | `wavelet → outlier_z-score → linear → min-max → cubic15 → 训练` |
| gait_039 | `wavelet+outlier_z-score+linear+min-max+nearest15` | `wavelet → outlier_z-score → linear → min-max → nearest15 → 训练` |
| gait_040 | `wavelet+outlier_z-score+linear+min-max+decimate15` | `wavelet → outlier_z-score → linear → min-max → decimate15 → 训练` |

### 公共前缀 06：wavelet+outlier_z-score+polynomial_d3

公共缓存结果：`wavelet → outlier_z-score → polynomial_d3`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_041 | `wavelet+outlier_z-score+polynomial_d3+z-score+linear15` | `wavelet → outlier_z-score → polynomial_d3 → linear15 → z-score → 训练` |
| gait_042 | `wavelet+outlier_z-score+polynomial_d3+z-score+cubic15` | `wavelet → outlier_z-score → polynomial_d3 → cubic15 → z-score → 训练` |
| gait_043 | `wavelet+outlier_z-score+polynomial_d3+z-score+nearest15` | `wavelet → outlier_z-score → polynomial_d3 → nearest15 → z-score → 训练` |
| gait_044 | `wavelet+outlier_z-score+polynomial_d3+z-score+decimate15` | `wavelet → outlier_z-score → polynomial_d3 → decimate15 → z-score → 训练` |
| gait_045 | `wavelet+outlier_z-score+polynomial_d3+min-max+linear15` | `wavelet → outlier_z-score → polynomial_d3 → min-max → linear15 → 训练` |
| gait_046 | `wavelet+outlier_z-score+polynomial_d3+min-max+cubic15` | `wavelet → outlier_z-score → polynomial_d3 → min-max → cubic15 → 训练` |
| gait_047 | `wavelet+outlier_z-score+polynomial_d3+min-max+nearest15` | `wavelet → outlier_z-score → polynomial_d3 → min-max → nearest15 → 训练` |
| gait_048 | `wavelet+outlier_z-score+polynomial_d3+min-max+decimate15` | `wavelet → outlier_z-score → polynomial_d3 → min-max → decimate15 → 训练` |

### 公共前缀 07：wavelet+outlier_z-score+stc

公共缓存结果：`wavelet → outlier_z-score → stc`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_049 | `wavelet+outlier_z-score+stc+z-score+linear15` | `wavelet → outlier_z-score → stc → linear15 → z-score → 训练` |
| gait_050 | `wavelet+outlier_z-score+stc+z-score+cubic15` | `wavelet → outlier_z-score → stc → cubic15 → z-score → 训练` |
| gait_051 | `wavelet+outlier_z-score+stc+z-score+nearest15` | `wavelet → outlier_z-score → stc → nearest15 → z-score → 训练` |
| gait_052 | `wavelet+outlier_z-score+stc+z-score+decimate15` | `wavelet → outlier_z-score → stc → decimate15 → z-score → 训练` |
| gait_053 | `wavelet+outlier_z-score+stc+min-max+linear15` | `wavelet → outlier_z-score → stc → min-max → linear15 → 训练` |
| gait_054 | `wavelet+outlier_z-score+stc+min-max+cubic15` | `wavelet → outlier_z-score → stc → min-max → cubic15 → 训练` |
| gait_055 | `wavelet+outlier_z-score+stc+min-max+nearest15` | `wavelet → outlier_z-score → stc → min-max → nearest15 → 训练` |
| gait_056 | `wavelet+outlier_z-score+stc+min-max+decimate15` | `wavelet → outlier_z-score → stc → min-max → decimate15 → 训练` |

### 公共前缀 08：wavelet+outlier_z-score+robust

公共缓存结果：`wavelet → outlier_z-score → robust`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_057 | `wavelet+outlier_z-score+robust+z-score+linear15` | `wavelet → outlier_z-score → robust → linear15 → z-score → 训练` |
| gait_058 | `wavelet+outlier_z-score+robust+z-score+cubic15` | `wavelet → outlier_z-score → robust → cubic15 → z-score → 训练` |
| gait_059 | `wavelet+outlier_z-score+robust+z-score+nearest15` | `wavelet → outlier_z-score → robust → nearest15 → z-score → 训练` |
| gait_060 | `wavelet+outlier_z-score+robust+z-score+decimate15` | `wavelet → outlier_z-score → robust → decimate15 → z-score → 训练` |
| gait_061 | `wavelet+outlier_z-score+robust+min-max+linear15` | `wavelet → outlier_z-score → robust → min-max → linear15 → 训练` |
| gait_062 | `wavelet+outlier_z-score+robust+min-max+cubic15` | `wavelet → outlier_z-score → robust → min-max → cubic15 → 训练` |
| gait_063 | `wavelet+outlier_z-score+robust+min-max+nearest15` | `wavelet → outlier_z-score → robust → min-max → nearest15 → 训练` |
| gait_064 | `wavelet+outlier_z-score+robust+min-max+decimate15` | `wavelet → outlier_z-score → robust → min-max → decimate15 → 训练` |

### 公共前缀 09：butterworth_o5_c0.3+iqr+linear

公共缓存结果：`butterworth_o5_c0.3 → iqr → linear`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_065 | `butterworth_o5_c0.3+iqr+linear+z-score+linear15` | `butterworth_o5_c0.3 → iqr → linear → linear15 → z-score → 训练` |
| gait_066 | `butterworth_o5_c0.3+iqr+linear+z-score+cubic15` | `butterworth_o5_c0.3 → iqr → linear → cubic15 → z-score → 训练` |
| gait_067 | `butterworth_o5_c0.3+iqr+linear+z-score+nearest15` | `butterworth_o5_c0.3 → iqr → linear → nearest15 → z-score → 训练` |
| gait_068 | `butterworth_o5_c0.3+iqr+linear+z-score+decimate15` | `butterworth_o5_c0.3 → iqr → linear → decimate15 → z-score → 训练` |
| gait_069 | `butterworth_o5_c0.3+iqr+linear+min-max+linear15` | `butterworth_o5_c0.3 → iqr → linear → min-max → linear15 → 训练` |
| gait_070 | `butterworth_o5_c0.3+iqr+linear+min-max+cubic15` | `butterworth_o5_c0.3 → iqr → linear → min-max → cubic15 → 训练` |
| gait_071 | `butterworth_o5_c0.3+iqr+linear+min-max+nearest15` | `butterworth_o5_c0.3 → iqr → linear → min-max → nearest15 → 训练` |
| gait_072 | `butterworth_o5_c0.3+iqr+linear+min-max+decimate15` | `butterworth_o5_c0.3 → iqr → linear → min-max → decimate15 → 训练` |

### 公共前缀 10：butterworth_o5_c0.3+iqr+polynomial_d3

公共缓存结果：`butterworth_o5_c0.3 → iqr → polynomial_d3`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_073 | `butterworth_o5_c0.3+iqr+polynomial_d3+z-score+linear15` | `butterworth_o5_c0.3 → iqr → polynomial_d3 → linear15 → z-score → 训练` |
| gait_074 | `butterworth_o5_c0.3+iqr+polynomial_d3+z-score+cubic15` | `butterworth_o5_c0.3 → iqr → polynomial_d3 → cubic15 → z-score → 训练` |
| gait_075 | `butterworth_o5_c0.3+iqr+polynomial_d3+z-score+nearest15` | `butterworth_o5_c0.3 → iqr → polynomial_d3 → nearest15 → z-score → 训练` |
| gait_076 | `butterworth_o5_c0.3+iqr+polynomial_d3+z-score+decimate15` | `butterworth_o5_c0.3 → iqr → polynomial_d3 → decimate15 → z-score → 训练` |
| gait_077 | `butterworth_o5_c0.3+iqr+polynomial_d3+min-max+linear15` | `butterworth_o5_c0.3 → iqr → polynomial_d3 → min-max → linear15 → 训练` |
| gait_078 | `butterworth_o5_c0.3+iqr+polynomial_d3+min-max+cubic15` | `butterworth_o5_c0.3 → iqr → polynomial_d3 → min-max → cubic15 → 训练` |
| gait_079 | `butterworth_o5_c0.3+iqr+polynomial_d3+min-max+nearest15` | `butterworth_o5_c0.3 → iqr → polynomial_d3 → min-max → nearest15 → 训练` |
| gait_080 | `butterworth_o5_c0.3+iqr+polynomial_d3+min-max+decimate15` | `butterworth_o5_c0.3 → iqr → polynomial_d3 → min-max → decimate15 → 训练` |

### 公共前缀 11：butterworth_o5_c0.3+iqr+stc

公共缓存结果：`butterworth_o5_c0.3 → iqr → stc`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_081 | `butterworth_o5_c0.3+iqr+stc+z-score+linear15` | `butterworth_o5_c0.3 → iqr → stc → linear15 → z-score → 训练` |
| gait_082 | `butterworth_o5_c0.3+iqr+stc+z-score+cubic15` | `butterworth_o5_c0.3 → iqr → stc → cubic15 → z-score → 训练` |
| gait_083 | `butterworth_o5_c0.3+iqr+stc+z-score+nearest15` | `butterworth_o5_c0.3 → iqr → stc → nearest15 → z-score → 训练` |
| gait_084 | `butterworth_o5_c0.3+iqr+stc+z-score+decimate15` | `butterworth_o5_c0.3 → iqr → stc → decimate15 → z-score → 训练` |
| gait_085 | `butterworth_o5_c0.3+iqr+stc+min-max+linear15` | `butterworth_o5_c0.3 → iqr → stc → min-max → linear15 → 训练` |
| gait_086 | `butterworth_o5_c0.3+iqr+stc+min-max+cubic15` | `butterworth_o5_c0.3 → iqr → stc → min-max → cubic15 → 训练` |
| gait_087 | `butterworth_o5_c0.3+iqr+stc+min-max+nearest15` | `butterworth_o5_c0.3 → iqr → stc → min-max → nearest15 → 训练` |
| gait_088 | `butterworth_o5_c0.3+iqr+stc+min-max+decimate15` | `butterworth_o5_c0.3 → iqr → stc → min-max → decimate15 → 训练` |

### 公共前缀 12：butterworth_o5_c0.3+iqr+robust

公共缓存结果：`butterworth_o5_c0.3 → iqr → robust`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_089 | `butterworth_o5_c0.3+iqr+robust+z-score+linear15` | `butterworth_o5_c0.3 → iqr → robust → linear15 → z-score → 训练` |
| gait_090 | `butterworth_o5_c0.3+iqr+robust+z-score+cubic15` | `butterworth_o5_c0.3 → iqr → robust → cubic15 → z-score → 训练` |
| gait_091 | `butterworth_o5_c0.3+iqr+robust+z-score+nearest15` | `butterworth_o5_c0.3 → iqr → robust → nearest15 → z-score → 训练` |
| gait_092 | `butterworth_o5_c0.3+iqr+robust+z-score+decimate15` | `butterworth_o5_c0.3 → iqr → robust → decimate15 → z-score → 训练` |
| gait_093 | `butterworth_o5_c0.3+iqr+robust+min-max+linear15` | `butterworth_o5_c0.3 → iqr → robust → min-max → linear15 → 训练` |
| gait_094 | `butterworth_o5_c0.3+iqr+robust+min-max+cubic15` | `butterworth_o5_c0.3 → iqr → robust → min-max → cubic15 → 训练` |
| gait_095 | `butterworth_o5_c0.3+iqr+robust+min-max+nearest15` | `butterworth_o5_c0.3 → iqr → robust → min-max → nearest15 → 训练` |
| gait_096 | `butterworth_o5_c0.3+iqr+robust+min-max+decimate15` | `butterworth_o5_c0.3 → iqr → robust → min-max → decimate15 → 训练` |

### 公共前缀 13：butterworth_o5_c0.3+outlier_z-score+linear

公共缓存结果：`butterworth_o5_c0.3 → outlier_z-score → linear`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_097 | `butterworth_o5_c0.3+outlier_z-score+linear+z-score+linear15` | `butterworth_o5_c0.3 → outlier_z-score → linear → linear15 → z-score → 训练` |
| gait_098 | `butterworth_o5_c0.3+outlier_z-score+linear+z-score+cubic15` | `butterworth_o5_c0.3 → outlier_z-score → linear → cubic15 → z-score → 训练` |
| gait_099 | `butterworth_o5_c0.3+outlier_z-score+linear+z-score+nearest15` | `butterworth_o5_c0.3 → outlier_z-score → linear → nearest15 → z-score → 训练` |
| gait_100 | `butterworth_o5_c0.3+outlier_z-score+linear+z-score+decimate15` | `butterworth_o5_c0.3 → outlier_z-score → linear → decimate15 → z-score → 训练` |
| gait_101 | `butterworth_o5_c0.3+outlier_z-score+linear+min-max+linear15` | `butterworth_o5_c0.3 → outlier_z-score → linear → min-max → linear15 → 训练` |
| gait_102 | `butterworth_o5_c0.3+outlier_z-score+linear+min-max+cubic15` | `butterworth_o5_c0.3 → outlier_z-score → linear → min-max → cubic15 → 训练` |
| gait_103 | `butterworth_o5_c0.3+outlier_z-score+linear+min-max+nearest15` | `butterworth_o5_c0.3 → outlier_z-score → linear → min-max → nearest15 → 训练` |
| gait_104 | `butterworth_o5_c0.3+outlier_z-score+linear+min-max+decimate15` | `butterworth_o5_c0.3 → outlier_z-score → linear → min-max → decimate15 → 训练` |

### 公共前缀 14：butterworth_o5_c0.3+outlier_z-score+polynomial_d3

公共缓存结果：`butterworth_o5_c0.3 → outlier_z-score → polynomial_d3`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_105 | `butterworth_o5_c0.3+outlier_z-score+polynomial_d3+z-score+linear15` | `butterworth_o5_c0.3 → outlier_z-score → polynomial_d3 → linear15 → z-score → 训练` |
| gait_106 | `butterworth_o5_c0.3+outlier_z-score+polynomial_d3+z-score+cubic15` | `butterworth_o5_c0.3 → outlier_z-score → polynomial_d3 → cubic15 → z-score → 训练` |
| gait_107 | `butterworth_o5_c0.3+outlier_z-score+polynomial_d3+z-score+nearest15` | `butterworth_o5_c0.3 → outlier_z-score → polynomial_d3 → nearest15 → z-score → 训练` |
| gait_108 | `butterworth_o5_c0.3+outlier_z-score+polynomial_d3+z-score+decimate15` | `butterworth_o5_c0.3 → outlier_z-score → polynomial_d3 → decimate15 → z-score → 训练` |
| gait_109 | `butterworth_o5_c0.3+outlier_z-score+polynomial_d3+min-max+linear15` | `butterworth_o5_c0.3 → outlier_z-score → polynomial_d3 → min-max → linear15 → 训练` |
| gait_110 | `butterworth_o5_c0.3+outlier_z-score+polynomial_d3+min-max+cubic15` | `butterworth_o5_c0.3 → outlier_z-score → polynomial_d3 → min-max → cubic15 → 训练` |
| gait_111 | `butterworth_o5_c0.3+outlier_z-score+polynomial_d3+min-max+nearest15` | `butterworth_o5_c0.3 → outlier_z-score → polynomial_d3 → min-max → nearest15 → 训练` |
| gait_112 | `butterworth_o5_c0.3+outlier_z-score+polynomial_d3+min-max+decimate15` | `butterworth_o5_c0.3 → outlier_z-score → polynomial_d3 → min-max → decimate15 → 训练` |

### 公共前缀 15：butterworth_o5_c0.3+outlier_z-score+stc

公共缓存结果：`butterworth_o5_c0.3 → outlier_z-score → stc`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_113 | `butterworth_o5_c0.3+outlier_z-score+stc+z-score+linear15` | `butterworth_o5_c0.3 → outlier_z-score → stc → linear15 → z-score → 训练` |
| gait_114 | `butterworth_o5_c0.3+outlier_z-score+stc+z-score+cubic15` | `butterworth_o5_c0.3 → outlier_z-score → stc → cubic15 → z-score → 训练` |
| gait_115 | `butterworth_o5_c0.3+outlier_z-score+stc+z-score+nearest15` | `butterworth_o5_c0.3 → outlier_z-score → stc → nearest15 → z-score → 训练` |
| gait_116 | `butterworth_o5_c0.3+outlier_z-score+stc+z-score+decimate15` | `butterworth_o5_c0.3 → outlier_z-score → stc → decimate15 → z-score → 训练` |
| gait_117 | `butterworth_o5_c0.3+outlier_z-score+stc+min-max+linear15` | `butterworth_o5_c0.3 → outlier_z-score → stc → min-max → linear15 → 训练` |
| gait_118 | `butterworth_o5_c0.3+outlier_z-score+stc+min-max+cubic15` | `butterworth_o5_c0.3 → outlier_z-score → stc → min-max → cubic15 → 训练` |
| gait_119 | `butterworth_o5_c0.3+outlier_z-score+stc+min-max+nearest15` | `butterworth_o5_c0.3 → outlier_z-score → stc → min-max → nearest15 → 训练` |
| gait_120 | `butterworth_o5_c0.3+outlier_z-score+stc+min-max+decimate15` | `butterworth_o5_c0.3 → outlier_z-score → stc → min-max → decimate15 → 训练` |

### 公共前缀 16：butterworth_o5_c0.3+outlier_z-score+robust

公共缓存结果：`butterworth_o5_c0.3 → outlier_z-score → robust`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_121 | `butterworth_o5_c0.3+outlier_z-score+robust+z-score+linear15` | `butterworth_o5_c0.3 → outlier_z-score → robust → linear15 → z-score → 训练` |
| gait_122 | `butterworth_o5_c0.3+outlier_z-score+robust+z-score+cubic15` | `butterworth_o5_c0.3 → outlier_z-score → robust → cubic15 → z-score → 训练` |
| gait_123 | `butterworth_o5_c0.3+outlier_z-score+robust+z-score+nearest15` | `butterworth_o5_c0.3 → outlier_z-score → robust → nearest15 → z-score → 训练` |
| gait_124 | `butterworth_o5_c0.3+outlier_z-score+robust+z-score+decimate15` | `butterworth_o5_c0.3 → outlier_z-score → robust → decimate15 → z-score → 训练` |
| gait_125 | `butterworth_o5_c0.3+outlier_z-score+robust+min-max+linear15` | `butterworth_o5_c0.3 → outlier_z-score → robust → min-max → linear15 → 训练` |
| gait_126 | `butterworth_o5_c0.3+outlier_z-score+robust+min-max+cubic15` | `butterworth_o5_c0.3 → outlier_z-score → robust → min-max → cubic15 → 训练` |
| gait_127 | `butterworth_o5_c0.3+outlier_z-score+robust+min-max+nearest15` | `butterworth_o5_c0.3 → outlier_z-score → robust → min-max → nearest15 → 训练` |
| gait_128 | `butterworth_o5_c0.3+outlier_z-score+robust+min-max+decimate15` | `butterworth_o5_c0.3 → outlier_z-score → robust → min-max → decimate15 → 训练` |

### 公共前缀 17：savgol_w7_p3+iqr+linear

公共缓存结果：`savgol_w7_p3 → iqr → linear`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_129 | `savgol_w7_p3+iqr+linear+z-score+linear15` | `savgol_w7_p3 → iqr → linear → linear15 → z-score → 训练` |
| gait_130 | `savgol_w7_p3+iqr+linear+z-score+cubic15` | `savgol_w7_p3 → iqr → linear → cubic15 → z-score → 训练` |
| gait_131 | `savgol_w7_p3+iqr+linear+z-score+nearest15` | `savgol_w7_p3 → iqr → linear → nearest15 → z-score → 训练` |
| gait_132 | `savgol_w7_p3+iqr+linear+z-score+decimate15` | `savgol_w7_p3 → iqr → linear → decimate15 → z-score → 训练` |
| gait_133 | `savgol_w7_p3+iqr+linear+min-max+linear15` | `savgol_w7_p3 → iqr → linear → min-max → linear15 → 训练` |
| gait_134 | `savgol_w7_p3+iqr+linear+min-max+cubic15` | `savgol_w7_p3 → iqr → linear → min-max → cubic15 → 训练` |
| gait_135 | `savgol_w7_p3+iqr+linear+min-max+nearest15` | `savgol_w7_p3 → iqr → linear → min-max → nearest15 → 训练` |
| gait_136 | `savgol_w7_p3+iqr+linear+min-max+decimate15` | `savgol_w7_p3 → iqr → linear → min-max → decimate15 → 训练` |

### 公共前缀 18：savgol_w7_p3+iqr+polynomial_d3

公共缓存结果：`savgol_w7_p3 → iqr → polynomial_d3`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_137 | `savgol_w7_p3+iqr+polynomial_d3+z-score+linear15` | `savgol_w7_p3 → iqr → polynomial_d3 → linear15 → z-score → 训练` |
| gait_138 | `savgol_w7_p3+iqr+polynomial_d3+z-score+cubic15` | `savgol_w7_p3 → iqr → polynomial_d3 → cubic15 → z-score → 训练` |
| gait_139 | `savgol_w7_p3+iqr+polynomial_d3+z-score+nearest15` | `savgol_w7_p3 → iqr → polynomial_d3 → nearest15 → z-score → 训练` |
| gait_140 | `savgol_w7_p3+iqr+polynomial_d3+z-score+decimate15` | `savgol_w7_p3 → iqr → polynomial_d3 → decimate15 → z-score → 训练` |
| gait_141 | `savgol_w7_p3+iqr+polynomial_d3+min-max+linear15` | `savgol_w7_p3 → iqr → polynomial_d3 → min-max → linear15 → 训练` |
| gait_142 | `savgol_w7_p3+iqr+polynomial_d3+min-max+cubic15` | `savgol_w7_p3 → iqr → polynomial_d3 → min-max → cubic15 → 训练` |
| gait_143 | `savgol_w7_p3+iqr+polynomial_d3+min-max+nearest15` | `savgol_w7_p3 → iqr → polynomial_d3 → min-max → nearest15 → 训练` |
| gait_144 | `savgol_w7_p3+iqr+polynomial_d3+min-max+decimate15` | `savgol_w7_p3 → iqr → polynomial_d3 → min-max → decimate15 → 训练` |

### 公共前缀 19：savgol_w7_p3+iqr+stc

公共缓存结果：`savgol_w7_p3 → iqr → stc`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_145 | `savgol_w7_p3+iqr+stc+z-score+linear15` | `savgol_w7_p3 → iqr → stc → linear15 → z-score → 训练` |
| gait_146 | `savgol_w7_p3+iqr+stc+z-score+cubic15` | `savgol_w7_p3 → iqr → stc → cubic15 → z-score → 训练` |
| gait_147 | `savgol_w7_p3+iqr+stc+z-score+nearest15` | `savgol_w7_p3 → iqr → stc → nearest15 → z-score → 训练` |
| gait_148 | `savgol_w7_p3+iqr+stc+z-score+decimate15` | `savgol_w7_p3 → iqr → stc → decimate15 → z-score → 训练` |
| gait_149 | `savgol_w7_p3+iqr+stc+min-max+linear15` | `savgol_w7_p3 → iqr → stc → min-max → linear15 → 训练` |
| gait_150 | `savgol_w7_p3+iqr+stc+min-max+cubic15` | `savgol_w7_p3 → iqr → stc → min-max → cubic15 → 训练` |
| gait_151 | `savgol_w7_p3+iqr+stc+min-max+nearest15` | `savgol_w7_p3 → iqr → stc → min-max → nearest15 → 训练` |
| gait_152 | `savgol_w7_p3+iqr+stc+min-max+decimate15` | `savgol_w7_p3 → iqr → stc → min-max → decimate15 → 训练` |

### 公共前缀 20：savgol_w7_p3+iqr+robust

公共缓存结果：`savgol_w7_p3 → iqr → robust`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_153 | `savgol_w7_p3+iqr+robust+z-score+linear15` | `savgol_w7_p3 → iqr → robust → linear15 → z-score → 训练` |
| gait_154 | `savgol_w7_p3+iqr+robust+z-score+cubic15` | `savgol_w7_p3 → iqr → robust → cubic15 → z-score → 训练` |
| gait_155 | `savgol_w7_p3+iqr+robust+z-score+nearest15` | `savgol_w7_p3 → iqr → robust → nearest15 → z-score → 训练` |
| gait_156 | `savgol_w7_p3+iqr+robust+z-score+decimate15` | `savgol_w7_p3 → iqr → robust → decimate15 → z-score → 训练` |
| gait_157 | `savgol_w7_p3+iqr+robust+min-max+linear15` | `savgol_w7_p3 → iqr → robust → min-max → linear15 → 训练` |
| gait_158 | `savgol_w7_p3+iqr+robust+min-max+cubic15` | `savgol_w7_p3 → iqr → robust → min-max → cubic15 → 训练` |
| gait_159 | `savgol_w7_p3+iqr+robust+min-max+nearest15` | `savgol_w7_p3 → iqr → robust → min-max → nearest15 → 训练` |
| gait_160 | `savgol_w7_p3+iqr+robust+min-max+decimate15` | `savgol_w7_p3 → iqr → robust → min-max → decimate15 → 训练` |

### 公共前缀 21：savgol_w7_p3+outlier_z-score+linear

公共缓存结果：`savgol_w7_p3 → outlier_z-score → linear`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_161 | `savgol_w7_p3+outlier_z-score+linear+z-score+linear15` | `savgol_w7_p3 → outlier_z-score → linear → linear15 → z-score → 训练` |
| gait_162 | `savgol_w7_p3+outlier_z-score+linear+z-score+cubic15` | `savgol_w7_p3 → outlier_z-score → linear → cubic15 → z-score → 训练` |
| gait_163 | `savgol_w7_p3+outlier_z-score+linear+z-score+nearest15` | `savgol_w7_p3 → outlier_z-score → linear → nearest15 → z-score → 训练` |
| gait_164 | `savgol_w7_p3+outlier_z-score+linear+z-score+decimate15` | `savgol_w7_p3 → outlier_z-score → linear → decimate15 → z-score → 训练` |
| gait_165 | `savgol_w7_p3+outlier_z-score+linear+min-max+linear15` | `savgol_w7_p3 → outlier_z-score → linear → min-max → linear15 → 训练` |
| gait_166 | `savgol_w7_p3+outlier_z-score+linear+min-max+cubic15` | `savgol_w7_p3 → outlier_z-score → linear → min-max → cubic15 → 训练` |
| gait_167 | `savgol_w7_p3+outlier_z-score+linear+min-max+nearest15` | `savgol_w7_p3 → outlier_z-score → linear → min-max → nearest15 → 训练` |
| gait_168 | `savgol_w7_p3+outlier_z-score+linear+min-max+decimate15` | `savgol_w7_p3 → outlier_z-score → linear → min-max → decimate15 → 训练` |

### 公共前缀 22：savgol_w7_p3+outlier_z-score+polynomial_d3

公共缓存结果：`savgol_w7_p3 → outlier_z-score → polynomial_d3`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_169 | `savgol_w7_p3+outlier_z-score+polynomial_d3+z-score+linear15` | `savgol_w7_p3 → outlier_z-score → polynomial_d3 → linear15 → z-score → 训练` |
| gait_170 | `savgol_w7_p3+outlier_z-score+polynomial_d3+z-score+cubic15` | `savgol_w7_p3 → outlier_z-score → polynomial_d3 → cubic15 → z-score → 训练` |
| gait_171 | `savgol_w7_p3+outlier_z-score+polynomial_d3+z-score+nearest15` | `savgol_w7_p3 → outlier_z-score → polynomial_d3 → nearest15 → z-score → 训练` |
| gait_172 | `savgol_w7_p3+outlier_z-score+polynomial_d3+z-score+decimate15` | `savgol_w7_p3 → outlier_z-score → polynomial_d3 → decimate15 → z-score → 训练` |
| gait_173 | `savgol_w7_p3+outlier_z-score+polynomial_d3+min-max+linear15` | `savgol_w7_p3 → outlier_z-score → polynomial_d3 → min-max → linear15 → 训练` |
| gait_174 | `savgol_w7_p3+outlier_z-score+polynomial_d3+min-max+cubic15` | `savgol_w7_p3 → outlier_z-score → polynomial_d3 → min-max → cubic15 → 训练` |
| gait_175 | `savgol_w7_p3+outlier_z-score+polynomial_d3+min-max+nearest15` | `savgol_w7_p3 → outlier_z-score → polynomial_d3 → min-max → nearest15 → 训练` |
| gait_176 | `savgol_w7_p3+outlier_z-score+polynomial_d3+min-max+decimate15` | `savgol_w7_p3 → outlier_z-score → polynomial_d3 → min-max → decimate15 → 训练` |

### 公共前缀 23：savgol_w7_p3+outlier_z-score+stc

公共缓存结果：`savgol_w7_p3 → outlier_z-score → stc`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_177 | `savgol_w7_p3+outlier_z-score+stc+z-score+linear15` | `savgol_w7_p3 → outlier_z-score → stc → linear15 → z-score → 训练` |
| gait_178 | `savgol_w7_p3+outlier_z-score+stc+z-score+cubic15` | `savgol_w7_p3 → outlier_z-score → stc → cubic15 → z-score → 训练` |
| gait_179 | `savgol_w7_p3+outlier_z-score+stc+z-score+nearest15` | `savgol_w7_p3 → outlier_z-score → stc → nearest15 → z-score → 训练` |
| gait_180 | `savgol_w7_p3+outlier_z-score+stc+z-score+decimate15` | `savgol_w7_p3 → outlier_z-score → stc → decimate15 → z-score → 训练` |
| gait_181 | `savgol_w7_p3+outlier_z-score+stc+min-max+linear15` | `savgol_w7_p3 → outlier_z-score → stc → min-max → linear15 → 训练` |
| gait_182 | `savgol_w7_p3+outlier_z-score+stc+min-max+cubic15` | `savgol_w7_p3 → outlier_z-score → stc → min-max → cubic15 → 训练` |
| gait_183 | `savgol_w7_p3+outlier_z-score+stc+min-max+nearest15` | `savgol_w7_p3 → outlier_z-score → stc → min-max → nearest15 → 训练` |
| gait_184 | `savgol_w7_p3+outlier_z-score+stc+min-max+decimate15` | `savgol_w7_p3 → outlier_z-score → stc → min-max → decimate15 → 训练` |

### 公共前缀 24：savgol_w7_p3+outlier_z-score+robust

公共缓存结果：`savgol_w7_p3 → outlier_z-score → robust`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_185 | `savgol_w7_p3+outlier_z-score+robust+z-score+linear15` | `savgol_w7_p3 → outlier_z-score → robust → linear15 → z-score → 训练` |
| gait_186 | `savgol_w7_p3+outlier_z-score+robust+z-score+cubic15` | `savgol_w7_p3 → outlier_z-score → robust → cubic15 → z-score → 训练` |
| gait_187 | `savgol_w7_p3+outlier_z-score+robust+z-score+nearest15` | `savgol_w7_p3 → outlier_z-score → robust → nearest15 → z-score → 训练` |
| gait_188 | `savgol_w7_p3+outlier_z-score+robust+z-score+decimate15` | `savgol_w7_p3 → outlier_z-score → robust → decimate15 → z-score → 训练` |
| gait_189 | `savgol_w7_p3+outlier_z-score+robust+min-max+linear15` | `savgol_w7_p3 → outlier_z-score → robust → min-max → linear15 → 训练` |
| gait_190 | `savgol_w7_p3+outlier_z-score+robust+min-max+cubic15` | `savgol_w7_p3 → outlier_z-score → robust → min-max → cubic15 → 训练` |
| gait_191 | `savgol_w7_p3+outlier_z-score+robust+min-max+nearest15` | `savgol_w7_p3 → outlier_z-score → robust → min-max → nearest15 → 训练` |
| gait_192 | `savgol_w7_p3+outlier_z-score+robust+min-max+decimate15` | `savgol_w7_p3 → outlier_z-score → robust → min-max → decimate15 → 训练` |

### 公共前缀 25：bandpass_0.5-50+iqr+linear

公共缓存结果：`bandpass_0.5-50 → iqr → linear`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_193 | `bandpass_0.5-50+iqr+linear+z-score+linear15` | `bandpass_0.5-50 → iqr → linear → linear15 → z-score → 训练` |
| gait_194 | `bandpass_0.5-50+iqr+linear+z-score+cubic15` | `bandpass_0.5-50 → iqr → linear → cubic15 → z-score → 训练` |
| gait_195 | `bandpass_0.5-50+iqr+linear+z-score+nearest15` | `bandpass_0.5-50 → iqr → linear → nearest15 → z-score → 训练` |
| gait_196 | `bandpass_0.5-50+iqr+linear+z-score+decimate15` | `bandpass_0.5-50 → iqr → linear → decimate15 → z-score → 训练` |
| gait_197 | `bandpass_0.5-50+iqr+linear+min-max+linear15` | `bandpass_0.5-50 → iqr → linear → min-max → linear15 → 训练` |
| gait_198 | `bandpass_0.5-50+iqr+linear+min-max+cubic15` | `bandpass_0.5-50 → iqr → linear → min-max → cubic15 → 训练` |
| gait_199 | `bandpass_0.5-50+iqr+linear+min-max+nearest15` | `bandpass_0.5-50 → iqr → linear → min-max → nearest15 → 训练` |
| gait_200 | `bandpass_0.5-50+iqr+linear+min-max+decimate15` | `bandpass_0.5-50 → iqr → linear → min-max → decimate15 → 训练` |

### 公共前缀 26：bandpass_0.5-50+iqr+polynomial_d3

公共缓存结果：`bandpass_0.5-50 → iqr → polynomial_d3`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_201 | `bandpass_0.5-50+iqr+polynomial_d3+z-score+linear15` | `bandpass_0.5-50 → iqr → polynomial_d3 → linear15 → z-score → 训练` |
| gait_202 | `bandpass_0.5-50+iqr+polynomial_d3+z-score+cubic15` | `bandpass_0.5-50 → iqr → polynomial_d3 → cubic15 → z-score → 训练` |
| gait_203 | `bandpass_0.5-50+iqr+polynomial_d3+z-score+nearest15` | `bandpass_0.5-50 → iqr → polynomial_d3 → nearest15 → z-score → 训练` |
| gait_204 | `bandpass_0.5-50+iqr+polynomial_d3+z-score+decimate15` | `bandpass_0.5-50 → iqr → polynomial_d3 → decimate15 → z-score → 训练` |
| gait_205 | `bandpass_0.5-50+iqr+polynomial_d3+min-max+linear15` | `bandpass_0.5-50 → iqr → polynomial_d3 → min-max → linear15 → 训练` |
| gait_206 | `bandpass_0.5-50+iqr+polynomial_d3+min-max+cubic15` | `bandpass_0.5-50 → iqr → polynomial_d3 → min-max → cubic15 → 训练` |
| gait_207 | `bandpass_0.5-50+iqr+polynomial_d3+min-max+nearest15` | `bandpass_0.5-50 → iqr → polynomial_d3 → min-max → nearest15 → 训练` |
| gait_208 | `bandpass_0.5-50+iqr+polynomial_d3+min-max+decimate15` | `bandpass_0.5-50 → iqr → polynomial_d3 → min-max → decimate15 → 训练` |

### 公共前缀 27：bandpass_0.5-50+iqr+stc

公共缓存结果：`bandpass_0.5-50 → iqr → stc`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_209 | `bandpass_0.5-50+iqr+stc+z-score+linear15` | `bandpass_0.5-50 → iqr → stc → linear15 → z-score → 训练` |
| gait_210 | `bandpass_0.5-50+iqr+stc+z-score+cubic15` | `bandpass_0.5-50 → iqr → stc → cubic15 → z-score → 训练` |
| gait_211 | `bandpass_0.5-50+iqr+stc+z-score+nearest15` | `bandpass_0.5-50 → iqr → stc → nearest15 → z-score → 训练` |
| gait_212 | `bandpass_0.5-50+iqr+stc+z-score+decimate15` | `bandpass_0.5-50 → iqr → stc → decimate15 → z-score → 训练` |
| gait_213 | `bandpass_0.5-50+iqr+stc+min-max+linear15` | `bandpass_0.5-50 → iqr → stc → min-max → linear15 → 训练` |
| gait_214 | `bandpass_0.5-50+iqr+stc+min-max+cubic15` | `bandpass_0.5-50 → iqr → stc → min-max → cubic15 → 训练` |
| gait_215 | `bandpass_0.5-50+iqr+stc+min-max+nearest15` | `bandpass_0.5-50 → iqr → stc → min-max → nearest15 → 训练` |
| gait_216 | `bandpass_0.5-50+iqr+stc+min-max+decimate15` | `bandpass_0.5-50 → iqr → stc → min-max → decimate15 → 训练` |

### 公共前缀 28：bandpass_0.5-50+iqr+robust

公共缓存结果：`bandpass_0.5-50 → iqr → robust`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_217 | `bandpass_0.5-50+iqr+robust+z-score+linear15` | `bandpass_0.5-50 → iqr → robust → linear15 → z-score → 训练` |
| gait_218 | `bandpass_0.5-50+iqr+robust+z-score+cubic15` | `bandpass_0.5-50 → iqr → robust → cubic15 → z-score → 训练` |
| gait_219 | `bandpass_0.5-50+iqr+robust+z-score+nearest15` | `bandpass_0.5-50 → iqr → robust → nearest15 → z-score → 训练` |
| gait_220 | `bandpass_0.5-50+iqr+robust+z-score+decimate15` | `bandpass_0.5-50 → iqr → robust → decimate15 → z-score → 训练` |
| gait_221 | `bandpass_0.5-50+iqr+robust+min-max+linear15` | `bandpass_0.5-50 → iqr → robust → min-max → linear15 → 训练` |
| gait_222 | `bandpass_0.5-50+iqr+robust+min-max+cubic15` | `bandpass_0.5-50 → iqr → robust → min-max → cubic15 → 训练` |
| gait_223 | `bandpass_0.5-50+iqr+robust+min-max+nearest15` | `bandpass_0.5-50 → iqr → robust → min-max → nearest15 → 训练` |
| gait_224 | `bandpass_0.5-50+iqr+robust+min-max+decimate15` | `bandpass_0.5-50 → iqr → robust → min-max → decimate15 → 训练` |

### 公共前缀 29：bandpass_0.5-50+outlier_z-score+linear

公共缓存结果：`bandpass_0.5-50 → outlier_z-score → linear`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_225 | `bandpass_0.5-50+outlier_z-score+linear+z-score+linear15` | `bandpass_0.5-50 → outlier_z-score → linear → linear15 → z-score → 训练` |
| gait_226 | `bandpass_0.5-50+outlier_z-score+linear+z-score+cubic15` | `bandpass_0.5-50 → outlier_z-score → linear → cubic15 → z-score → 训练` |
| gait_227 | `bandpass_0.5-50+outlier_z-score+linear+z-score+nearest15` | `bandpass_0.5-50 → outlier_z-score → linear → nearest15 → z-score → 训练` |
| gait_228 | `bandpass_0.5-50+outlier_z-score+linear+z-score+decimate15` | `bandpass_0.5-50 → outlier_z-score → linear → decimate15 → z-score → 训练` |
| gait_229 | `bandpass_0.5-50+outlier_z-score+linear+min-max+linear15` | `bandpass_0.5-50 → outlier_z-score → linear → min-max → linear15 → 训练` |
| gait_230 | `bandpass_0.5-50+outlier_z-score+linear+min-max+cubic15` | `bandpass_0.5-50 → outlier_z-score → linear → min-max → cubic15 → 训练` |
| gait_231 | `bandpass_0.5-50+outlier_z-score+linear+min-max+nearest15` | `bandpass_0.5-50 → outlier_z-score → linear → min-max → nearest15 → 训练` |
| gait_232 | `bandpass_0.5-50+outlier_z-score+linear+min-max+decimate15` | `bandpass_0.5-50 → outlier_z-score → linear → min-max → decimate15 → 训练` |

### 公共前缀 30：bandpass_0.5-50+outlier_z-score+polynomial_d3

公共缓存结果：`bandpass_0.5-50 → outlier_z-score → polynomial_d3`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_233 | `bandpass_0.5-50+outlier_z-score+polynomial_d3+z-score+linear15` | `bandpass_0.5-50 → outlier_z-score → polynomial_d3 → linear15 → z-score → 训练` |
| gait_234 | `bandpass_0.5-50+outlier_z-score+polynomial_d3+z-score+cubic15` | `bandpass_0.5-50 → outlier_z-score → polynomial_d3 → cubic15 → z-score → 训练` |
| gait_235 | `bandpass_0.5-50+outlier_z-score+polynomial_d3+z-score+nearest15` | `bandpass_0.5-50 → outlier_z-score → polynomial_d3 → nearest15 → z-score → 训练` |
| gait_236 | `bandpass_0.5-50+outlier_z-score+polynomial_d3+z-score+decimate15` | `bandpass_0.5-50 → outlier_z-score → polynomial_d3 → decimate15 → z-score → 训练` |
| gait_237 | `bandpass_0.5-50+outlier_z-score+polynomial_d3+min-max+linear15` | `bandpass_0.5-50 → outlier_z-score → polynomial_d3 → min-max → linear15 → 训练` |
| gait_238 | `bandpass_0.5-50+outlier_z-score+polynomial_d3+min-max+cubic15` | `bandpass_0.5-50 → outlier_z-score → polynomial_d3 → min-max → cubic15 → 训练` |
| gait_239 | `bandpass_0.5-50+outlier_z-score+polynomial_d3+min-max+nearest15` | `bandpass_0.5-50 → outlier_z-score → polynomial_d3 → min-max → nearest15 → 训练` |
| gait_240 | `bandpass_0.5-50+outlier_z-score+polynomial_d3+min-max+decimate15` | `bandpass_0.5-50 → outlier_z-score → polynomial_d3 → min-max → decimate15 → 训练` |

### 公共前缀 31：bandpass_0.5-50+outlier_z-score+stc

公共缓存结果：`bandpass_0.5-50 → outlier_z-score → stc`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_241 | `bandpass_0.5-50+outlier_z-score+stc+z-score+linear15` | `bandpass_0.5-50 → outlier_z-score → stc → linear15 → z-score → 训练` |
| gait_242 | `bandpass_0.5-50+outlier_z-score+stc+z-score+cubic15` | `bandpass_0.5-50 → outlier_z-score → stc → cubic15 → z-score → 训练` |
| gait_243 | `bandpass_0.5-50+outlier_z-score+stc+z-score+nearest15` | `bandpass_0.5-50 → outlier_z-score → stc → nearest15 → z-score → 训练` |
| gait_244 | `bandpass_0.5-50+outlier_z-score+stc+z-score+decimate15` | `bandpass_0.5-50 → outlier_z-score → stc → decimate15 → z-score → 训练` |
| gait_245 | `bandpass_0.5-50+outlier_z-score+stc+min-max+linear15` | `bandpass_0.5-50 → outlier_z-score → stc → min-max → linear15 → 训练` |
| gait_246 | `bandpass_0.5-50+outlier_z-score+stc+min-max+cubic15` | `bandpass_0.5-50 → outlier_z-score → stc → min-max → cubic15 → 训练` |
| gait_247 | `bandpass_0.5-50+outlier_z-score+stc+min-max+nearest15` | `bandpass_0.5-50 → outlier_z-score → stc → min-max → nearest15 → 训练` |
| gait_248 | `bandpass_0.5-50+outlier_z-score+stc+min-max+decimate15` | `bandpass_0.5-50 → outlier_z-score → stc → min-max → decimate15 → 训练` |

### 公共前缀 32：bandpass_0.5-50+outlier_z-score+robust

公共缓存结果：`bandpass_0.5-50 → outlier_z-score → robust`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_249 | `bandpass_0.5-50+outlier_z-score+robust+z-score+linear15` | `bandpass_0.5-50 → outlier_z-score → robust → linear15 → z-score → 训练` |
| gait_250 | `bandpass_0.5-50+outlier_z-score+robust+z-score+cubic15` | `bandpass_0.5-50 → outlier_z-score → robust → cubic15 → z-score → 训练` |
| gait_251 | `bandpass_0.5-50+outlier_z-score+robust+z-score+nearest15` | `bandpass_0.5-50 → outlier_z-score → robust → nearest15 → z-score → 训练` |
| gait_252 | `bandpass_0.5-50+outlier_z-score+robust+z-score+decimate15` | `bandpass_0.5-50 → outlier_z-score → robust → decimate15 → z-score → 训练` |
| gait_253 | `bandpass_0.5-50+outlier_z-score+robust+min-max+linear15` | `bandpass_0.5-50 → outlier_z-score → robust → min-max → linear15 → 训练` |
| gait_254 | `bandpass_0.5-50+outlier_z-score+robust+min-max+cubic15` | `bandpass_0.5-50 → outlier_z-score → robust → min-max → cubic15 → 训练` |
| gait_255 | `bandpass_0.5-50+outlier_z-score+robust+min-max+nearest15` | `bandpass_0.5-50 → outlier_z-score → robust → min-max → nearest15 → 训练` |
| gait_256 | `bandpass_0.5-50+outlier_z-score+robust+min-max+decimate15` | `bandpass_0.5-50 → outlier_z-score → robust → min-max → decimate15 → 训练` |

### 公共前缀 33：hampel_w5_s3+iqr+linear

公共缓存结果：`hampel_w5_s3 → iqr → linear`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_257 | `hampel_w5_s3+iqr+linear+z-score+linear15` | `hampel_w5_s3 → iqr → linear → linear15 → z-score → 训练` |
| gait_258 | `hampel_w5_s3+iqr+linear+z-score+cubic15` | `hampel_w5_s3 → iqr → linear → cubic15 → z-score → 训练` |
| gait_259 | `hampel_w5_s3+iqr+linear+z-score+nearest15` | `hampel_w5_s3 → iqr → linear → nearest15 → z-score → 训练` |
| gait_260 | `hampel_w5_s3+iqr+linear+z-score+decimate15` | `hampel_w5_s3 → iqr → linear → decimate15 → z-score → 训练` |
| gait_261 | `hampel_w5_s3+iqr+linear+min-max+linear15` | `hampel_w5_s3 → iqr → linear → min-max → linear15 → 训练` |
| gait_262 | `hampel_w5_s3+iqr+linear+min-max+cubic15` | `hampel_w5_s3 → iqr → linear → min-max → cubic15 → 训练` |
| gait_263 | `hampel_w5_s3+iqr+linear+min-max+nearest15` | `hampel_w5_s3 → iqr → linear → min-max → nearest15 → 训练` |
| gait_264 | `hampel_w5_s3+iqr+linear+min-max+decimate15` | `hampel_w5_s3 → iqr → linear → min-max → decimate15 → 训练` |

### 公共前缀 34：hampel_w5_s3+iqr+polynomial_d3

公共缓存结果：`hampel_w5_s3 → iqr → polynomial_d3`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_265 | `hampel_w5_s3+iqr+polynomial_d3+z-score+linear15` | `hampel_w5_s3 → iqr → polynomial_d3 → linear15 → z-score → 训练` |
| gait_266 | `hampel_w5_s3+iqr+polynomial_d3+z-score+cubic15` | `hampel_w5_s3 → iqr → polynomial_d3 → cubic15 → z-score → 训练` |
| gait_267 | `hampel_w5_s3+iqr+polynomial_d3+z-score+nearest15` | `hampel_w5_s3 → iqr → polynomial_d3 → nearest15 → z-score → 训练` |
| gait_268 | `hampel_w5_s3+iqr+polynomial_d3+z-score+decimate15` | `hampel_w5_s3 → iqr → polynomial_d3 → decimate15 → z-score → 训练` |
| gait_269 | `hampel_w5_s3+iqr+polynomial_d3+min-max+linear15` | `hampel_w5_s3 → iqr → polynomial_d3 → min-max → linear15 → 训练` |
| gait_270 | `hampel_w5_s3+iqr+polynomial_d3+min-max+cubic15` | `hampel_w5_s3 → iqr → polynomial_d3 → min-max → cubic15 → 训练` |
| gait_271 | `hampel_w5_s3+iqr+polynomial_d3+min-max+nearest15` | `hampel_w5_s3 → iqr → polynomial_d3 → min-max → nearest15 → 训练` |
| gait_272 | `hampel_w5_s3+iqr+polynomial_d3+min-max+decimate15` | `hampel_w5_s3 → iqr → polynomial_d3 → min-max → decimate15 → 训练` |

### 公共前缀 35：hampel_w5_s3+iqr+stc

公共缓存结果：`hampel_w5_s3 → iqr → stc`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_273 | `hampel_w5_s3+iqr+stc+z-score+linear15` | `hampel_w5_s3 → iqr → stc → linear15 → z-score → 训练` |
| gait_274 | `hampel_w5_s3+iqr+stc+z-score+cubic15` | `hampel_w5_s3 → iqr → stc → cubic15 → z-score → 训练` |
| gait_275 | `hampel_w5_s3+iqr+stc+z-score+nearest15` | `hampel_w5_s3 → iqr → stc → nearest15 → z-score → 训练` |
| gait_276 | `hampel_w5_s3+iqr+stc+z-score+decimate15` | `hampel_w5_s3 → iqr → stc → decimate15 → z-score → 训练` |
| gait_277 | `hampel_w5_s3+iqr+stc+min-max+linear15` | `hampel_w5_s3 → iqr → stc → min-max → linear15 → 训练` |
| gait_278 | `hampel_w5_s3+iqr+stc+min-max+cubic15` | `hampel_w5_s3 → iqr → stc → min-max → cubic15 → 训练` |
| gait_279 | `hampel_w5_s3+iqr+stc+min-max+nearest15` | `hampel_w5_s3 → iqr → stc → min-max → nearest15 → 训练` |
| gait_280 | `hampel_w5_s3+iqr+stc+min-max+decimate15` | `hampel_w5_s3 → iqr → stc → min-max → decimate15 → 训练` |

### 公共前缀 36：hampel_w5_s3+iqr+robust

公共缓存结果：`hampel_w5_s3 → iqr → robust`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_281 | `hampel_w5_s3+iqr+robust+z-score+linear15` | `hampel_w5_s3 → iqr → robust → linear15 → z-score → 训练` |
| gait_282 | `hampel_w5_s3+iqr+robust+z-score+cubic15` | `hampel_w5_s3 → iqr → robust → cubic15 → z-score → 训练` |
| gait_283 | `hampel_w5_s3+iqr+robust+z-score+nearest15` | `hampel_w5_s3 → iqr → robust → nearest15 → z-score → 训练` |
| gait_284 | `hampel_w5_s3+iqr+robust+z-score+decimate15` | `hampel_w5_s3 → iqr → robust → decimate15 → z-score → 训练` |
| gait_285 | `hampel_w5_s3+iqr+robust+min-max+linear15` | `hampel_w5_s3 → iqr → robust → min-max → linear15 → 训练` |
| gait_286 | `hampel_w5_s3+iqr+robust+min-max+cubic15` | `hampel_w5_s3 → iqr → robust → min-max → cubic15 → 训练` |
| gait_287 | `hampel_w5_s3+iqr+robust+min-max+nearest15` | `hampel_w5_s3 → iqr → robust → min-max → nearest15 → 训练` |
| gait_288 | `hampel_w5_s3+iqr+robust+min-max+decimate15` | `hampel_w5_s3 → iqr → robust → min-max → decimate15 → 训练` |

### 公共前缀 37：hampel_w5_s3+outlier_z-score+linear

公共缓存结果：`hampel_w5_s3 → outlier_z-score → linear`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_289 | `hampel_w5_s3+outlier_z-score+linear+z-score+linear15` | `hampel_w5_s3 → outlier_z-score → linear → linear15 → z-score → 训练` |
| gait_290 | `hampel_w5_s3+outlier_z-score+linear+z-score+cubic15` | `hampel_w5_s3 → outlier_z-score → linear → cubic15 → z-score → 训练` |
| gait_291 | `hampel_w5_s3+outlier_z-score+linear+z-score+nearest15` | `hampel_w5_s3 → outlier_z-score → linear → nearest15 → z-score → 训练` |
| gait_292 | `hampel_w5_s3+outlier_z-score+linear+z-score+decimate15` | `hampel_w5_s3 → outlier_z-score → linear → decimate15 → z-score → 训练` |
| gait_293 | `hampel_w5_s3+outlier_z-score+linear+min-max+linear15` | `hampel_w5_s3 → outlier_z-score → linear → min-max → linear15 → 训练` |
| gait_294 | `hampel_w5_s3+outlier_z-score+linear+min-max+cubic15` | `hampel_w5_s3 → outlier_z-score → linear → min-max → cubic15 → 训练` |
| gait_295 | `hampel_w5_s3+outlier_z-score+linear+min-max+nearest15` | `hampel_w5_s3 → outlier_z-score → linear → min-max → nearest15 → 训练` |
| gait_296 | `hampel_w5_s3+outlier_z-score+linear+min-max+decimate15` | `hampel_w5_s3 → outlier_z-score → linear → min-max → decimate15 → 训练` |

### 公共前缀 38：hampel_w5_s3+outlier_z-score+polynomial_d3

公共缓存结果：`hampel_w5_s3 → outlier_z-score → polynomial_d3`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_297 | `hampel_w5_s3+outlier_z-score+polynomial_d3+z-score+linear15` | `hampel_w5_s3 → outlier_z-score → polynomial_d3 → linear15 → z-score → 训练` |
| gait_298 | `hampel_w5_s3+outlier_z-score+polynomial_d3+z-score+cubic15` | `hampel_w5_s3 → outlier_z-score → polynomial_d3 → cubic15 → z-score → 训练` |
| gait_299 | `hampel_w5_s3+outlier_z-score+polynomial_d3+z-score+nearest15` | `hampel_w5_s3 → outlier_z-score → polynomial_d3 → nearest15 → z-score → 训练` |
| gait_300 | `hampel_w5_s3+outlier_z-score+polynomial_d3+z-score+decimate15` | `hampel_w5_s3 → outlier_z-score → polynomial_d3 → decimate15 → z-score → 训练` |
| gait_301 | `hampel_w5_s3+outlier_z-score+polynomial_d3+min-max+linear15` | `hampel_w5_s3 → outlier_z-score → polynomial_d3 → min-max → linear15 → 训练` |
| gait_302 | `hampel_w5_s3+outlier_z-score+polynomial_d3+min-max+cubic15` | `hampel_w5_s3 → outlier_z-score → polynomial_d3 → min-max → cubic15 → 训练` |
| gait_303 | `hampel_w5_s3+outlier_z-score+polynomial_d3+min-max+nearest15` | `hampel_w5_s3 → outlier_z-score → polynomial_d3 → min-max → nearest15 → 训练` |
| gait_304 | `hampel_w5_s3+outlier_z-score+polynomial_d3+min-max+decimate15` | `hampel_w5_s3 → outlier_z-score → polynomial_d3 → min-max → decimate15 → 训练` |

### 公共前缀 39：hampel_w5_s3+outlier_z-score+stc

公共缓存结果：`hampel_w5_s3 → outlier_z-score → stc`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_305 | `hampel_w5_s3+outlier_z-score+stc+z-score+linear15` | `hampel_w5_s3 → outlier_z-score → stc → linear15 → z-score → 训练` |
| gait_306 | `hampel_w5_s3+outlier_z-score+stc+z-score+cubic15` | `hampel_w5_s3 → outlier_z-score → stc → cubic15 → z-score → 训练` |
| gait_307 | `hampel_w5_s3+outlier_z-score+stc+z-score+nearest15` | `hampel_w5_s3 → outlier_z-score → stc → nearest15 → z-score → 训练` |
| gait_308 | `hampel_w5_s3+outlier_z-score+stc+z-score+decimate15` | `hampel_w5_s3 → outlier_z-score → stc → decimate15 → z-score → 训练` |
| gait_309 | `hampel_w5_s3+outlier_z-score+stc+min-max+linear15` | `hampel_w5_s3 → outlier_z-score → stc → min-max → linear15 → 训练` |
| gait_310 | `hampel_w5_s3+outlier_z-score+stc+min-max+cubic15` | `hampel_w5_s3 → outlier_z-score → stc → min-max → cubic15 → 训练` |
| gait_311 | `hampel_w5_s3+outlier_z-score+stc+min-max+nearest15` | `hampel_w5_s3 → outlier_z-score → stc → min-max → nearest15 → 训练` |
| gait_312 | `hampel_w5_s3+outlier_z-score+stc+min-max+decimate15` | `hampel_w5_s3 → outlier_z-score → stc → min-max → decimate15 → 训练` |

### 公共前缀 40：hampel_w5_s3+outlier_z-score+robust

公共缓存结果：`hampel_w5_s3 → outlier_z-score → robust`

| combo_id | combo_name | 实际执行流程 |
|---|---|---|
| gait_313 | `hampel_w5_s3+outlier_z-score+robust+z-score+linear15` | `hampel_w5_s3 → outlier_z-score → robust → linear15 → z-score → 训练` |
| gait_314 | `hampel_w5_s3+outlier_z-score+robust+z-score+cubic15` | `hampel_w5_s3 → outlier_z-score → robust → cubic15 → z-score → 训练` |
| gait_315 | `hampel_w5_s3+outlier_z-score+robust+z-score+nearest15` | `hampel_w5_s3 → outlier_z-score → robust → nearest15 → z-score → 训练` |
| gait_316 | `hampel_w5_s3+outlier_z-score+robust+z-score+decimate15` | `hampel_w5_s3 → outlier_z-score → robust → decimate15 → z-score → 训练` |
| gait_317 | `hampel_w5_s3+outlier_z-score+robust+min-max+linear15` | `hampel_w5_s3 → outlier_z-score → robust → min-max → linear15 → 训练` |
| gait_318 | `hampel_w5_s3+outlier_z-score+robust+min-max+cubic15` | `hampel_w5_s3 → outlier_z-score → robust → min-max → cubic15 → 训练` |
| gait_319 | `hampel_w5_s3+outlier_z-score+robust+min-max+nearest15` | `hampel_w5_s3 → outlier_z-score → robust → min-max → nearest15 → 训练` |
| gait_320 | `hampel_w5_s3+outlier_z-score+robust+min-max+decimate15` | `hampel_w5_s3 → outlier_z-score → robust → min-max → decimate15 → 训练` |

## 数量核对

- 公共前缀：40 个。
- 每个公共前缀：8 个最终组合。
- 完整组合：`40 × 8 = 320` 个。
- 首个组合：`gait_001 = wavelet+iqr+linear+z-score+linear15`。
- 最后组合：`gait_320 = hampel_w5_s3+outlier_z-score+robust+min-max+decimate15`。

