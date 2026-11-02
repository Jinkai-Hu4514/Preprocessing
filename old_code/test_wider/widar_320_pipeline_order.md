# Widar 320 种 pipeline 组合实际运行顺序

来源文件：`SDP/test_wider/full_test_widar.py`

实际生成逻辑来自 `build_pipeline_combinations()`，顺序就是 `itertools.product(DENOISE_OPTIONS, OUTLIER_OPTIONS, CALIBRATE_OPTIONS, NORMALIZE_OPTIONS, INTERPOLATE_OPTIONS)` 的展开顺序。

也就是外层到内层依次为：

```text
denoise -> outliers -> calibrate -> normalize -> interpolate
```

## 算法选项顺序

### denoise

- `wavelet`: `{"method": "wavelet"}`
- `butterworth_o5_c0.3`: `{"method": "butterworth", "order": 5, "cutoff": 0.3}`
- `savgol_w7_p3`: `{"method": "savgol", "window_length": 7, "polyorder": 3}`
- `bandpass_0.5-50`: `{"method": "bandpass", "order": 4, "low_freq": 0.5, "high_freq": 50.0, "fs": 1000.0}`
- `hampel_w5_s3`: `{"method": "hampel", "window_size": 5, "n_sigma": 3.0}`

### outliers

- `iqr`: `{"method": "iqr", "factor": 1.5}`
- `outlier_z-score`: `{"method": "z-score", "factor": 3.0}`

### calibrate

- `linear`: `{"method": "linear"}`
- `polynomial_d3`: `{"method": "polynomial", "degree": 3}`
- `stc`: `{"method": "stc"}`
- `robust`: `{"method": "robust"}`

### normalize

- `z-score`: `{"method": "z-score"}`
- `min-max`: `{"method": "min-max"}`

### interpolate

- `linear15`: `{"method": "linear", "target_K": 15}`
- `cubic15`: `{"method": "cubic", "target_K": 15}`
- `nearest15`: `{"method": "nearest", "target_K": 15}`
- `decimate15`: `{"method": "decimate", "target_K": 15}`

## 320 组实际运行顺序

| 序号 | combo_id | combo_name | denoise | outliers | calibrate | normalize | interpolate |
|---:|---|---|---|---|---|---|---|
| 1 | widar_001 | wavelet+iqr+linear+z-score+linear15 | wavelet | iqr | linear | z-score | linear15 |
| 2 | widar_002 | wavelet+iqr+linear+z-score+cubic15 | wavelet | iqr | linear | z-score | cubic15 |
| 3 | widar_003 | wavelet+iqr+linear+z-score+nearest15 | wavelet | iqr | linear | z-score | nearest15 |
| 4 | widar_004 | wavelet+iqr+linear+z-score+decimate15 | wavelet | iqr | linear | z-score | decimate15 |
| 5 | widar_005 | wavelet+iqr+linear+min-max+linear15 | wavelet | iqr | linear | min-max | linear15 |
| 6 | widar_006 | wavelet+iqr+linear+min-max+cubic15 | wavelet | iqr | linear | min-max | cubic15 |
| 7 | widar_007 | wavelet+iqr+linear+min-max+nearest15 | wavelet | iqr | linear | min-max | nearest15 |
| 8 | widar_008 | wavelet+iqr+linear+min-max+decimate15 | wavelet | iqr | linear | min-max | decimate15 |
| 9 | widar_009 | wavelet+iqr+polynomial_d3+z-score+linear15 | wavelet | iqr | polynomial_d3 | z-score | linear15 |
| 10 | widar_010 | wavelet+iqr+polynomial_d3+z-score+cubic15 | wavelet | iqr | polynomial_d3 | z-score | cubic15 |
| 11 | widar_011 | wavelet+iqr+polynomial_d3+z-score+nearest15 | wavelet | iqr | polynomial_d3 | z-score | nearest15 |
| 12 | widar_012 | wavelet+iqr+polynomial_d3+z-score+decimate15 | wavelet | iqr | polynomial_d3 | z-score | decimate15 |
| 13 | widar_013 | wavelet+iqr+polynomial_d3+min-max+linear15 | wavelet | iqr | polynomial_d3 | min-max | linear15 |
| 14 | widar_014 | wavelet+iqr+polynomial_d3+min-max+cubic15 | wavelet | iqr | polynomial_d3 | min-max | cubic15 |
| 15 | widar_015 | wavelet+iqr+polynomial_d3+min-max+nearest15 | wavelet | iqr | polynomial_d3 | min-max | nearest15 |
| 16 | widar_016 | wavelet+iqr+polynomial_d3+min-max+decimate15 | wavelet | iqr | polynomial_d3 | min-max | decimate15 |
| 17 | widar_017 | wavelet+iqr+stc+z-score+linear15 | wavelet | iqr | stc | z-score | linear15 |
| 18 | widar_018 | wavelet+iqr+stc+z-score+cubic15 | wavelet | iqr | stc | z-score | cubic15 |
| 19 | widar_019 | wavelet+iqr+stc+z-score+nearest15 | wavelet | iqr | stc | z-score | nearest15 |
| 20 | widar_020 | wavelet+iqr+stc+z-score+decimate15 | wavelet | iqr | stc | z-score | decimate15 |
| 21 | widar_021 | wavelet+iqr+stc+min-max+linear15 | wavelet | iqr | stc | min-max | linear15 |
| 22 | widar_022 | wavelet+iqr+stc+min-max+cubic15 | wavelet | iqr | stc | min-max | cubic15 |
| 23 | widar_023 | wavelet+iqr+stc+min-max+nearest15 | wavelet | iqr | stc | min-max | nearest15 |
| 24 | widar_024 | wavelet+iqr+stc+min-max+decimate15 | wavelet | iqr | stc | min-max | decimate15 |
| 25 | widar_025 | wavelet+iqr+robust+z-score+linear15 | wavelet | iqr | robust | z-score | linear15 |
| 26 | widar_026 | wavelet+iqr+robust+z-score+cubic15 | wavelet | iqr | robust | z-score | cubic15 |
| 27 | widar_027 | wavelet+iqr+robust+z-score+nearest15 | wavelet | iqr | robust | z-score | nearest15 |
| 28 | widar_028 | wavelet+iqr+robust+z-score+decimate15 | wavelet | iqr | robust | z-score | decimate15 |
| 29 | widar_029 | wavelet+iqr+robust+min-max+linear15 | wavelet | iqr | robust | min-max | linear15 |
| 30 | widar_030 | wavelet+iqr+robust+min-max+cubic15 | wavelet | iqr | robust | min-max | cubic15 |
| 31 | widar_031 | wavelet+iqr+robust+min-max+nearest15 | wavelet | iqr | robust | min-max | nearest15 |
| 32 | widar_032 | wavelet+iqr+robust+min-max+decimate15 | wavelet | iqr | robust | min-max | decimate15 |
| 33 | widar_033 | wavelet+outlier_z-score+linear+z-score+linear15 | wavelet | outlier_z-score | linear | z-score | linear15 |
| 34 | widar_034 | wavelet+outlier_z-score+linear+z-score+cubic15 | wavelet | outlier_z-score | linear | z-score | cubic15 |
| 35 | widar_035 | wavelet+outlier_z-score+linear+z-score+nearest15 | wavelet | outlier_z-score | linear | z-score | nearest15 |
| 36 | widar_036 | wavelet+outlier_z-score+linear+z-score+decimate15 | wavelet | outlier_z-score | linear | z-score | decimate15 |
| 37 | widar_037 | wavelet+outlier_z-score+linear+min-max+linear15 | wavelet | outlier_z-score | linear | min-max | linear15 |
| 38 | widar_038 | wavelet+outlier_z-score+linear+min-max+cubic15 | wavelet | outlier_z-score | linear | min-max | cubic15 |
| 39 | widar_039 | wavelet+outlier_z-score+linear+min-max+nearest15 | wavelet | outlier_z-score | linear | min-max | nearest15 |
| 40 | widar_040 | wavelet+outlier_z-score+linear+min-max+decimate15 | wavelet | outlier_z-score | linear | min-max | decimate15 |
| 41 | widar_041 | wavelet+outlier_z-score+polynomial_d3+z-score+linear15 | wavelet | outlier_z-score | polynomial_d3 | z-score | linear15 |
| 42 | widar_042 | wavelet+outlier_z-score+polynomial_d3+z-score+cubic15 | wavelet | outlier_z-score | polynomial_d3 | z-score | cubic15 |
| 43 | widar_043 | wavelet+outlier_z-score+polynomial_d3+z-score+nearest15 | wavelet | outlier_z-score | polynomial_d3 | z-score | nearest15 |
| 44 | widar_044 | wavelet+outlier_z-score+polynomial_d3+z-score+decimate15 | wavelet | outlier_z-score | polynomial_d3 | z-score | decimate15 |
| 45 | widar_045 | wavelet+outlier_z-score+polynomial_d3+min-max+linear15 | wavelet | outlier_z-score | polynomial_d3 | min-max | linear15 |
| 46 | widar_046 | wavelet+outlier_z-score+polynomial_d3+min-max+cubic15 | wavelet | outlier_z-score | polynomial_d3 | min-max | cubic15 |
| 47 | widar_047 | wavelet+outlier_z-score+polynomial_d3+min-max+nearest15 | wavelet | outlier_z-score | polynomial_d3 | min-max | nearest15 |
| 48 | widar_048 | wavelet+outlier_z-score+polynomial_d3+min-max+decimate15 | wavelet | outlier_z-score | polynomial_d3 | min-max | decimate15 |
| 49 | widar_049 | wavelet+outlier_z-score+stc+z-score+linear15 | wavelet | outlier_z-score | stc | z-score | linear15 |
| 50 | widar_050 | wavelet+outlier_z-score+stc+z-score+cubic15 | wavelet | outlier_z-score | stc | z-score | cubic15 |
| 51 | widar_051 | wavelet+outlier_z-score+stc+z-score+nearest15 | wavelet | outlier_z-score | stc | z-score | nearest15 |
| 52 | widar_052 | wavelet+outlier_z-score+stc+z-score+decimate15 | wavelet | outlier_z-score | stc | z-score | decimate15 |
| 53 | widar_053 | wavelet+outlier_z-score+stc+min-max+linear15 | wavelet | outlier_z-score | stc | min-max | linear15 |
| 54 | widar_054 | wavelet+outlier_z-score+stc+min-max+cubic15 | wavelet | outlier_z-score | stc | min-max | cubic15 |
| 55 | widar_055 | wavelet+outlier_z-score+stc+min-max+nearest15 | wavelet | outlier_z-score | stc | min-max | nearest15 |
| 56 | widar_056 | wavelet+outlier_z-score+stc+min-max+decimate15 | wavelet | outlier_z-score | stc | min-max | decimate15 |
| 57 | widar_057 | wavelet+outlier_z-score+robust+z-score+linear15 | wavelet | outlier_z-score | robust | z-score | linear15 |
| 58 | widar_058 | wavelet+outlier_z-score+robust+z-score+cubic15 | wavelet | outlier_z-score | robust | z-score | cubic15 |
| 59 | widar_059 | wavelet+outlier_z-score+robust+z-score+nearest15 | wavelet | outlier_z-score | robust | z-score | nearest15 |
| 60 | widar_060 | wavelet+outlier_z-score+robust+z-score+decimate15 | wavelet | outlier_z-score | robust | z-score | decimate15 |
| 61 | widar_061 | wavelet+outlier_z-score+robust+min-max+linear15 | wavelet | outlier_z-score | robust | min-max | linear15 |
| 62 | widar_062 | wavelet+outlier_z-score+robust+min-max+cubic15 | wavelet | outlier_z-score | robust | min-max | cubic15 |
| 63 | widar_063 | wavelet+outlier_z-score+robust+min-max+nearest15 | wavelet | outlier_z-score | robust | min-max | nearest15 |
| 64 | widar_064 | wavelet+outlier_z-score+robust+min-max+decimate15 | wavelet | outlier_z-score | robust | min-max | decimate15 |
| 65 | widar_065 | butterworth_o5_c0.3+iqr+linear+z-score+linear15 | butterworth_o5_c0.3 | iqr | linear | z-score | linear15 |
| 66 | widar_066 | butterworth_o5_c0.3+iqr+linear+z-score+cubic15 | butterworth_o5_c0.3 | iqr | linear | z-score | cubic15 |
| 67 | widar_067 | butterworth_o5_c0.3+iqr+linear+z-score+nearest15 | butterworth_o5_c0.3 | iqr | linear | z-score | nearest15 |
| 68 | widar_068 | butterworth_o5_c0.3+iqr+linear+z-score+decimate15 | butterworth_o5_c0.3 | iqr | linear | z-score | decimate15 |
| 69 | widar_069 | butterworth_o5_c0.3+iqr+linear+min-max+linear15 | butterworth_o5_c0.3 | iqr | linear | min-max | linear15 |
| 70 | widar_070 | butterworth_o5_c0.3+iqr+linear+min-max+cubic15 | butterworth_o5_c0.3 | iqr | linear | min-max | cubic15 |
| 71 | widar_071 | butterworth_o5_c0.3+iqr+linear+min-max+nearest15 | butterworth_o5_c0.3 | iqr | linear | min-max | nearest15 |
| 72 | widar_072 | butterworth_o5_c0.3+iqr+linear+min-max+decimate15 | butterworth_o5_c0.3 | iqr | linear | min-max | decimate15 |
| 73 | widar_073 | butterworth_o5_c0.3+iqr+polynomial_d3+z-score+linear15 | butterworth_o5_c0.3 | iqr | polynomial_d3 | z-score | linear15 |
| 74 | widar_074 | butterworth_o5_c0.3+iqr+polynomial_d3+z-score+cubic15 | butterworth_o5_c0.3 | iqr | polynomial_d3 | z-score | cubic15 |
| 75 | widar_075 | butterworth_o5_c0.3+iqr+polynomial_d3+z-score+nearest15 | butterworth_o5_c0.3 | iqr | polynomial_d3 | z-score | nearest15 |
| 76 | widar_076 | butterworth_o5_c0.3+iqr+polynomial_d3+z-score+decimate15 | butterworth_o5_c0.3 | iqr | polynomial_d3 | z-score | decimate15 |
| 77 | widar_077 | butterworth_o5_c0.3+iqr+polynomial_d3+min-max+linear15 | butterworth_o5_c0.3 | iqr | polynomial_d3 | min-max | linear15 |
| 78 | widar_078 | butterworth_o5_c0.3+iqr+polynomial_d3+min-max+cubic15 | butterworth_o5_c0.3 | iqr | polynomial_d3 | min-max | cubic15 |
| 79 | widar_079 | butterworth_o5_c0.3+iqr+polynomial_d3+min-max+nearest15 | butterworth_o5_c0.3 | iqr | polynomial_d3 | min-max | nearest15 |
| 80 | widar_080 | butterworth_o5_c0.3+iqr+polynomial_d3+min-max+decimate15 | butterworth_o5_c0.3 | iqr | polynomial_d3 | min-max | decimate15 |
| 81 | widar_081 | butterworth_o5_c0.3+iqr+stc+z-score+linear15 | butterworth_o5_c0.3 | iqr | stc | z-score | linear15 |
| 82 | widar_082 | butterworth_o5_c0.3+iqr+stc+z-score+cubic15 | butterworth_o5_c0.3 | iqr | stc | z-score | cubic15 |
| 83 | widar_083 | butterworth_o5_c0.3+iqr+stc+z-score+nearest15 | butterworth_o5_c0.3 | iqr | stc | z-score | nearest15 |
| 84 | widar_084 | butterworth_o5_c0.3+iqr+stc+z-score+decimate15 | butterworth_o5_c0.3 | iqr | stc | z-score | decimate15 |
| 85 | widar_085 | butterworth_o5_c0.3+iqr+stc+min-max+linear15 | butterworth_o5_c0.3 | iqr | stc | min-max | linear15 |
| 86 | widar_086 | butterworth_o5_c0.3+iqr+stc+min-max+cubic15 | butterworth_o5_c0.3 | iqr | stc | min-max | cubic15 |
| 87 | widar_087 | butterworth_o5_c0.3+iqr+stc+min-max+nearest15 | butterworth_o5_c0.3 | iqr | stc | min-max | nearest15 |
| 88 | widar_088 | butterworth_o5_c0.3+iqr+stc+min-max+decimate15 | butterworth_o5_c0.3 | iqr | stc | min-max | decimate15 |
| 89 | widar_089 | butterworth_o5_c0.3+iqr+robust+z-score+linear15 | butterworth_o5_c0.3 | iqr | robust | z-score | linear15 |
| 90 | widar_090 | butterworth_o5_c0.3+iqr+robust+z-score+cubic15 | butterworth_o5_c0.3 | iqr | robust | z-score | cubic15 |
| 91 | widar_091 | butterworth_o5_c0.3+iqr+robust+z-score+nearest15 | butterworth_o5_c0.3 | iqr | robust | z-score | nearest15 |
| 92 | widar_092 | butterworth_o5_c0.3+iqr+robust+z-score+decimate15 | butterworth_o5_c0.3 | iqr | robust | z-score | decimate15 |
| 93 | widar_093 | butterworth_o5_c0.3+iqr+robust+min-max+linear15 | butterworth_o5_c0.3 | iqr | robust | min-max | linear15 |
| 94 | widar_094 | butterworth_o5_c0.3+iqr+robust+min-max+cubic15 | butterworth_o5_c0.3 | iqr | robust | min-max | cubic15 |
| 95 | widar_095 | butterworth_o5_c0.3+iqr+robust+min-max+nearest15 | butterworth_o5_c0.3 | iqr | robust | min-max | nearest15 |
| 96 | widar_096 | butterworth_o5_c0.3+iqr+robust+min-max+decimate15 | butterworth_o5_c0.3 | iqr | robust | min-max | decimate15 |
| 97 | widar_097 | butterworth_o5_c0.3+outlier_z-score+linear+z-score+linear15 | butterworth_o5_c0.3 | outlier_z-score | linear | z-score | linear15 |
| 98 | widar_098 | butterworth_o5_c0.3+outlier_z-score+linear+z-score+cubic15 | butterworth_o5_c0.3 | outlier_z-score | linear | z-score | cubic15 |
| 99 | widar_099 | butterworth_o5_c0.3+outlier_z-score+linear+z-score+nearest15 | butterworth_o5_c0.3 | outlier_z-score | linear | z-score | nearest15 |
| 100 | widar_100 | butterworth_o5_c0.3+outlier_z-score+linear+z-score+decimate15 | butterworth_o5_c0.3 | outlier_z-score | linear | z-score | decimate15 |
| 101 | widar_101 | butterworth_o5_c0.3+outlier_z-score+linear+min-max+linear15 | butterworth_o5_c0.3 | outlier_z-score | linear | min-max | linear15 |
| 102 | widar_102 | butterworth_o5_c0.3+outlier_z-score+linear+min-max+cubic15 | butterworth_o5_c0.3 | outlier_z-score | linear | min-max | cubic15 |
| 103 | widar_103 | butterworth_o5_c0.3+outlier_z-score+linear+min-max+nearest15 | butterworth_o5_c0.3 | outlier_z-score | linear | min-max | nearest15 |
| 104 | widar_104 | butterworth_o5_c0.3+outlier_z-score+linear+min-max+decimate15 | butterworth_o5_c0.3 | outlier_z-score | linear | min-max | decimate15 |
| 105 | widar_105 | butterworth_o5_c0.3+outlier_z-score+polynomial_d3+z-score+linear15 | butterworth_o5_c0.3 | outlier_z-score | polynomial_d3 | z-score | linear15 |
| 106 | widar_106 | butterworth_o5_c0.3+outlier_z-score+polynomial_d3+z-score+cubic15 | butterworth_o5_c0.3 | outlier_z-score | polynomial_d3 | z-score | cubic15 |
| 107 | widar_107 | butterworth_o5_c0.3+outlier_z-score+polynomial_d3+z-score+nearest15 | butterworth_o5_c0.3 | outlier_z-score | polynomial_d3 | z-score | nearest15 |
| 108 | widar_108 | butterworth_o5_c0.3+outlier_z-score+polynomial_d3+z-score+decimate15 | butterworth_o5_c0.3 | outlier_z-score | polynomial_d3 | z-score | decimate15 |
| 109 | widar_109 | butterworth_o5_c0.3+outlier_z-score+polynomial_d3+min-max+linear15 | butterworth_o5_c0.3 | outlier_z-score | polynomial_d3 | min-max | linear15 |
| 110 | widar_110 | butterworth_o5_c0.3+outlier_z-score+polynomial_d3+min-max+cubic15 | butterworth_o5_c0.3 | outlier_z-score | polynomial_d3 | min-max | cubic15 |
| 111 | widar_111 | butterworth_o5_c0.3+outlier_z-score+polynomial_d3+min-max+nearest15 | butterworth_o5_c0.3 | outlier_z-score | polynomial_d3 | min-max | nearest15 |
| 112 | widar_112 | butterworth_o5_c0.3+outlier_z-score+polynomial_d3+min-max+decimate15 | butterworth_o5_c0.3 | outlier_z-score | polynomial_d3 | min-max | decimate15 |
| 113 | widar_113 | butterworth_o5_c0.3+outlier_z-score+stc+z-score+linear15 | butterworth_o5_c0.3 | outlier_z-score | stc | z-score | linear15 |
| 114 | widar_114 | butterworth_o5_c0.3+outlier_z-score+stc+z-score+cubic15 | butterworth_o5_c0.3 | outlier_z-score | stc | z-score | cubic15 |
| 115 | widar_115 | butterworth_o5_c0.3+outlier_z-score+stc+z-score+nearest15 | butterworth_o5_c0.3 | outlier_z-score | stc | z-score | nearest15 |
| 116 | widar_116 | butterworth_o5_c0.3+outlier_z-score+stc+z-score+decimate15 | butterworth_o5_c0.3 | outlier_z-score | stc | z-score | decimate15 |
| 117 | widar_117 | butterworth_o5_c0.3+outlier_z-score+stc+min-max+linear15 | butterworth_o5_c0.3 | outlier_z-score | stc | min-max | linear15 |
| 118 | widar_118 | butterworth_o5_c0.3+outlier_z-score+stc+min-max+cubic15 | butterworth_o5_c0.3 | outlier_z-score | stc | min-max | cubic15 |
| 119 | widar_119 | butterworth_o5_c0.3+outlier_z-score+stc+min-max+nearest15 | butterworth_o5_c0.3 | outlier_z-score | stc | min-max | nearest15 |
| 120 | widar_120 | butterworth_o5_c0.3+outlier_z-score+stc+min-max+decimate15 | butterworth_o5_c0.3 | outlier_z-score | stc | min-max | decimate15 |
| 121 | widar_121 | butterworth_o5_c0.3+outlier_z-score+robust+z-score+linear15 | butterworth_o5_c0.3 | outlier_z-score | robust | z-score | linear15 |
| 122 | widar_122 | butterworth_o5_c0.3+outlier_z-score+robust+z-score+cubic15 | butterworth_o5_c0.3 | outlier_z-score | robust | z-score | cubic15 |
| 123 | widar_123 | butterworth_o5_c0.3+outlier_z-score+robust+z-score+nearest15 | butterworth_o5_c0.3 | outlier_z-score | robust | z-score | nearest15 |
| 124 | widar_124 | butterworth_o5_c0.3+outlier_z-score+robust+z-score+decimate15 | butterworth_o5_c0.3 | outlier_z-score | robust | z-score | decimate15 |
| 125 | widar_125 | butterworth_o5_c0.3+outlier_z-score+robust+min-max+linear15 | butterworth_o5_c0.3 | outlier_z-score | robust | min-max | linear15 |
| 126 | widar_126 | butterworth_o5_c0.3+outlier_z-score+robust+min-max+cubic15 | butterworth_o5_c0.3 | outlier_z-score | robust | min-max | cubic15 |
| 127 | widar_127 | butterworth_o5_c0.3+outlier_z-score+robust+min-max+nearest15 | butterworth_o5_c0.3 | outlier_z-score | robust | min-max | nearest15 |
| 128 | widar_128 | butterworth_o5_c0.3+outlier_z-score+robust+min-max+decimate15 | butterworth_o5_c0.3 | outlier_z-score | robust | min-max | decimate15 |
| 129 | widar_129 | savgol_w7_p3+iqr+linear+z-score+linear15 | savgol_w7_p3 | iqr | linear | z-score | linear15 |
| 130 | widar_130 | savgol_w7_p3+iqr+linear+z-score+cubic15 | savgol_w7_p3 | iqr | linear | z-score | cubic15 |
| 131 | widar_131 | savgol_w7_p3+iqr+linear+z-score+nearest15 | savgol_w7_p3 | iqr | linear | z-score | nearest15 |
| 132 | widar_132 | savgol_w7_p3+iqr+linear+z-score+decimate15 | savgol_w7_p3 | iqr | linear | z-score | decimate15 |
| 133 | widar_133 | savgol_w7_p3+iqr+linear+min-max+linear15 | savgol_w7_p3 | iqr | linear | min-max | linear15 |
| 134 | widar_134 | savgol_w7_p3+iqr+linear+min-max+cubic15 | savgol_w7_p3 | iqr | linear | min-max | cubic15 |
| 135 | widar_135 | savgol_w7_p3+iqr+linear+min-max+nearest15 | savgol_w7_p3 | iqr | linear | min-max | nearest15 |
| 136 | widar_136 | savgol_w7_p3+iqr+linear+min-max+decimate15 | savgol_w7_p3 | iqr | linear | min-max | decimate15 |
| 137 | widar_137 | savgol_w7_p3+iqr+polynomial_d3+z-score+linear15 | savgol_w7_p3 | iqr | polynomial_d3 | z-score | linear15 |
| 138 | widar_138 | savgol_w7_p3+iqr+polynomial_d3+z-score+cubic15 | savgol_w7_p3 | iqr | polynomial_d3 | z-score | cubic15 |
| 139 | widar_139 | savgol_w7_p3+iqr+polynomial_d3+z-score+nearest15 | savgol_w7_p3 | iqr | polynomial_d3 | z-score | nearest15 |
| 140 | widar_140 | savgol_w7_p3+iqr+polynomial_d3+z-score+decimate15 | savgol_w7_p3 | iqr | polynomial_d3 | z-score | decimate15 |
| 141 | widar_141 | savgol_w7_p3+iqr+polynomial_d3+min-max+linear15 | savgol_w7_p3 | iqr | polynomial_d3 | min-max | linear15 |
| 142 | widar_142 | savgol_w7_p3+iqr+polynomial_d3+min-max+cubic15 | savgol_w7_p3 | iqr | polynomial_d3 | min-max | cubic15 |
| 143 | widar_143 | savgol_w7_p3+iqr+polynomial_d3+min-max+nearest15 | savgol_w7_p3 | iqr | polynomial_d3 | min-max | nearest15 |
| 144 | widar_144 | savgol_w7_p3+iqr+polynomial_d3+min-max+decimate15 | savgol_w7_p3 | iqr | polynomial_d3 | min-max | decimate15 |
| 145 | widar_145 | savgol_w7_p3+iqr+stc+z-score+linear15 | savgol_w7_p3 | iqr | stc | z-score | linear15 |
| 146 | widar_146 | savgol_w7_p3+iqr+stc+z-score+cubic15 | savgol_w7_p3 | iqr | stc | z-score | cubic15 |
| 147 | widar_147 | savgol_w7_p3+iqr+stc+z-score+nearest15 | savgol_w7_p3 | iqr | stc | z-score | nearest15 |
| 148 | widar_148 | savgol_w7_p3+iqr+stc+z-score+decimate15 | savgol_w7_p3 | iqr | stc | z-score | decimate15 |
| 149 | widar_149 | savgol_w7_p3+iqr+stc+min-max+linear15 | savgol_w7_p3 | iqr | stc | min-max | linear15 |
| 150 | widar_150 | savgol_w7_p3+iqr+stc+min-max+cubic15 | savgol_w7_p3 | iqr | stc | min-max | cubic15 |
| 151 | widar_151 | savgol_w7_p3+iqr+stc+min-max+nearest15 | savgol_w7_p3 | iqr | stc | min-max | nearest15 |
| 152 | widar_152 | savgol_w7_p3+iqr+stc+min-max+decimate15 | savgol_w7_p3 | iqr | stc | min-max | decimate15 |
| 153 | widar_153 | savgol_w7_p3+iqr+robust+z-score+linear15 | savgol_w7_p3 | iqr | robust | z-score | linear15 |
| 154 | widar_154 | savgol_w7_p3+iqr+robust+z-score+cubic15 | savgol_w7_p3 | iqr | robust | z-score | cubic15 |
| 155 | widar_155 | savgol_w7_p3+iqr+robust+z-score+nearest15 | savgol_w7_p3 | iqr | robust | z-score | nearest15 |
| 156 | widar_156 | savgol_w7_p3+iqr+robust+z-score+decimate15 | savgol_w7_p3 | iqr | robust | z-score | decimate15 |
| 157 | widar_157 | savgol_w7_p3+iqr+robust+min-max+linear15 | savgol_w7_p3 | iqr | robust | min-max | linear15 |
| 158 | widar_158 | savgol_w7_p3+iqr+robust+min-max+cubic15 | savgol_w7_p3 | iqr | robust | min-max | cubic15 |
| 159 | widar_159 | savgol_w7_p3+iqr+robust+min-max+nearest15 | savgol_w7_p3 | iqr | robust | min-max | nearest15 |
| 160 | widar_160 | savgol_w7_p3+iqr+robust+min-max+decimate15 | savgol_w7_p3 | iqr | robust | min-max | decimate15 |
| 161 | widar_161 | savgol_w7_p3+outlier_z-score+linear+z-score+linear15 | savgol_w7_p3 | outlier_z-score | linear | z-score | linear15 |
| 162 | widar_162 | savgol_w7_p3+outlier_z-score+linear+z-score+cubic15 | savgol_w7_p3 | outlier_z-score | linear | z-score | cubic15 |
| 163 | widar_163 | savgol_w7_p3+outlier_z-score+linear+z-score+nearest15 | savgol_w7_p3 | outlier_z-score | linear | z-score | nearest15 |
| 164 | widar_164 | savgol_w7_p3+outlier_z-score+linear+z-score+decimate15 | savgol_w7_p3 | outlier_z-score | linear | z-score | decimate15 |
| 165 | widar_165 | savgol_w7_p3+outlier_z-score+linear+min-max+linear15 | savgol_w7_p3 | outlier_z-score | linear | min-max | linear15 |
| 166 | widar_166 | savgol_w7_p3+outlier_z-score+linear+min-max+cubic15 | savgol_w7_p3 | outlier_z-score | linear | min-max | cubic15 |
| 167 | widar_167 | savgol_w7_p3+outlier_z-score+linear+min-max+nearest15 | savgol_w7_p3 | outlier_z-score | linear | min-max | nearest15 |
| 168 | widar_168 | savgol_w7_p3+outlier_z-score+linear+min-max+decimate15 | savgol_w7_p3 | outlier_z-score | linear | min-max | decimate15 |
| 169 | widar_169 | savgol_w7_p3+outlier_z-score+polynomial_d3+z-score+linear15 | savgol_w7_p3 | outlier_z-score | polynomial_d3 | z-score | linear15 |
| 170 | widar_170 | savgol_w7_p3+outlier_z-score+polynomial_d3+z-score+cubic15 | savgol_w7_p3 | outlier_z-score | polynomial_d3 | z-score | cubic15 |
| 171 | widar_171 | savgol_w7_p3+outlier_z-score+polynomial_d3+z-score+nearest15 | savgol_w7_p3 | outlier_z-score | polynomial_d3 | z-score | nearest15 |
| 172 | widar_172 | savgol_w7_p3+outlier_z-score+polynomial_d3+z-score+decimate15 | savgol_w7_p3 | outlier_z-score | polynomial_d3 | z-score | decimate15 |
| 173 | widar_173 | savgol_w7_p3+outlier_z-score+polynomial_d3+min-max+linear15 | savgol_w7_p3 | outlier_z-score | polynomial_d3 | min-max | linear15 |
| 174 | widar_174 | savgol_w7_p3+outlier_z-score+polynomial_d3+min-max+cubic15 | savgol_w7_p3 | outlier_z-score | polynomial_d3 | min-max | cubic15 |
| 175 | widar_175 | savgol_w7_p3+outlier_z-score+polynomial_d3+min-max+nearest15 | savgol_w7_p3 | outlier_z-score | polynomial_d3 | min-max | nearest15 |
| 176 | widar_176 | savgol_w7_p3+outlier_z-score+polynomial_d3+min-max+decimate15 | savgol_w7_p3 | outlier_z-score | polynomial_d3 | min-max | decimate15 |
| 177 | widar_177 | savgol_w7_p3+outlier_z-score+stc+z-score+linear15 | savgol_w7_p3 | outlier_z-score | stc | z-score | linear15 |
| 178 | widar_178 | savgol_w7_p3+outlier_z-score+stc+z-score+cubic15 | savgol_w7_p3 | outlier_z-score | stc | z-score | cubic15 |
| 179 | widar_179 | savgol_w7_p3+outlier_z-score+stc+z-score+nearest15 | savgol_w7_p3 | outlier_z-score | stc | z-score | nearest15 |
| 180 | widar_180 | savgol_w7_p3+outlier_z-score+stc+z-score+decimate15 | savgol_w7_p3 | outlier_z-score | stc | z-score | decimate15 |
| 181 | widar_181 | savgol_w7_p3+outlier_z-score+stc+min-max+linear15 | savgol_w7_p3 | outlier_z-score | stc | min-max | linear15 |
| 182 | widar_182 | savgol_w7_p3+outlier_z-score+stc+min-max+cubic15 | savgol_w7_p3 | outlier_z-score | stc | min-max | cubic15 |
| 183 | widar_183 | savgol_w7_p3+outlier_z-score+stc+min-max+nearest15 | savgol_w7_p3 | outlier_z-score | stc | min-max | nearest15 |
| 184 | widar_184 | savgol_w7_p3+outlier_z-score+stc+min-max+decimate15 | savgol_w7_p3 | outlier_z-score | stc | min-max | decimate15 |
| 185 | widar_185 | savgol_w7_p3+outlier_z-score+robust+z-score+linear15 | savgol_w7_p3 | outlier_z-score | robust | z-score | linear15 |
| 186 | widar_186 | savgol_w7_p3+outlier_z-score+robust+z-score+cubic15 | savgol_w7_p3 | outlier_z-score | robust | z-score | cubic15 |
| 187 | widar_187 | savgol_w7_p3+outlier_z-score+robust+z-score+nearest15 | savgol_w7_p3 | outlier_z-score | robust | z-score | nearest15 |
| 188 | widar_188 | savgol_w7_p3+outlier_z-score+robust+z-score+decimate15 | savgol_w7_p3 | outlier_z-score | robust | z-score | decimate15 |
| 189 | widar_189 | savgol_w7_p3+outlier_z-score+robust+min-max+linear15 | savgol_w7_p3 | outlier_z-score | robust | min-max | linear15 |
| 190 | widar_190 | savgol_w7_p3+outlier_z-score+robust+min-max+cubic15 | savgol_w7_p3 | outlier_z-score | robust | min-max | cubic15 |
| 191 | widar_191 | savgol_w7_p3+outlier_z-score+robust+min-max+nearest15 | savgol_w7_p3 | outlier_z-score | robust | min-max | nearest15 |
| 192 | widar_192 | savgol_w7_p3+outlier_z-score+robust+min-max+decimate15 | savgol_w7_p3 | outlier_z-score | robust | min-max | decimate15 |
| 193 | widar_193 | bandpass_0.5-50+iqr+linear+z-score+linear15 | bandpass_0.5-50 | iqr | linear | z-score | linear15 |
| 194 | widar_194 | bandpass_0.5-50+iqr+linear+z-score+cubic15 | bandpass_0.5-50 | iqr | linear | z-score | cubic15 |
| 195 | widar_195 | bandpass_0.5-50+iqr+linear+z-score+nearest15 | bandpass_0.5-50 | iqr | linear | z-score | nearest15 |
| 196 | widar_196 | bandpass_0.5-50+iqr+linear+z-score+decimate15 | bandpass_0.5-50 | iqr | linear | z-score | decimate15 |
| 197 | widar_197 | bandpass_0.5-50+iqr+linear+min-max+linear15 | bandpass_0.5-50 | iqr | linear | min-max | linear15 |
| 198 | widar_198 | bandpass_0.5-50+iqr+linear+min-max+cubic15 | bandpass_0.5-50 | iqr | linear | min-max | cubic15 |
| 199 | widar_199 | bandpass_0.5-50+iqr+linear+min-max+nearest15 | bandpass_0.5-50 | iqr | linear | min-max | nearest15 |
| 200 | widar_200 | bandpass_0.5-50+iqr+linear+min-max+decimate15 | bandpass_0.5-50 | iqr | linear | min-max | decimate15 |
| 201 | widar_201 | bandpass_0.5-50+iqr+polynomial_d3+z-score+linear15 | bandpass_0.5-50 | iqr | polynomial_d3 | z-score | linear15 |
| 202 | widar_202 | bandpass_0.5-50+iqr+polynomial_d3+z-score+cubic15 | bandpass_0.5-50 | iqr | polynomial_d3 | z-score | cubic15 |
| 203 | widar_203 | bandpass_0.5-50+iqr+polynomial_d3+z-score+nearest15 | bandpass_0.5-50 | iqr | polynomial_d3 | z-score | nearest15 |
| 204 | widar_204 | bandpass_0.5-50+iqr+polynomial_d3+z-score+decimate15 | bandpass_0.5-50 | iqr | polynomial_d3 | z-score | decimate15 |
| 205 | widar_205 | bandpass_0.5-50+iqr+polynomial_d3+min-max+linear15 | bandpass_0.5-50 | iqr | polynomial_d3 | min-max | linear15 |
| 206 | widar_206 | bandpass_0.5-50+iqr+polynomial_d3+min-max+cubic15 | bandpass_0.5-50 | iqr | polynomial_d3 | min-max | cubic15 |
| 207 | widar_207 | bandpass_0.5-50+iqr+polynomial_d3+min-max+nearest15 | bandpass_0.5-50 | iqr | polynomial_d3 | min-max | nearest15 |
| 208 | widar_208 | bandpass_0.5-50+iqr+polynomial_d3+min-max+decimate15 | bandpass_0.5-50 | iqr | polynomial_d3 | min-max | decimate15 |
| 209 | widar_209 | bandpass_0.5-50+iqr+stc+z-score+linear15 | bandpass_0.5-50 | iqr | stc | z-score | linear15 |
| 210 | widar_210 | bandpass_0.5-50+iqr+stc+z-score+cubic15 | bandpass_0.5-50 | iqr | stc | z-score | cubic15 |
| 211 | widar_211 | bandpass_0.5-50+iqr+stc+z-score+nearest15 | bandpass_0.5-50 | iqr | stc | z-score | nearest15 |
| 212 | widar_212 | bandpass_0.5-50+iqr+stc+z-score+decimate15 | bandpass_0.5-50 | iqr | stc | z-score | decimate15 |
| 213 | widar_213 | bandpass_0.5-50+iqr+stc+min-max+linear15 | bandpass_0.5-50 | iqr | stc | min-max | linear15 |
| 214 | widar_214 | bandpass_0.5-50+iqr+stc+min-max+cubic15 | bandpass_0.5-50 | iqr | stc | min-max | cubic15 |
| 215 | widar_215 | bandpass_0.5-50+iqr+stc+min-max+nearest15 | bandpass_0.5-50 | iqr | stc | min-max | nearest15 |
| 216 | widar_216 | bandpass_0.5-50+iqr+stc+min-max+decimate15 | bandpass_0.5-50 | iqr | stc | min-max | decimate15 |
| 217 | widar_217 | bandpass_0.5-50+iqr+robust+z-score+linear15 | bandpass_0.5-50 | iqr | robust | z-score | linear15 |
| 218 | widar_218 | bandpass_0.5-50+iqr+robust+z-score+cubic15 | bandpass_0.5-50 | iqr | robust | z-score | cubic15 |
| 219 | widar_219 | bandpass_0.5-50+iqr+robust+z-score+nearest15 | bandpass_0.5-50 | iqr | robust | z-score | nearest15 |
| 220 | widar_220 | bandpass_0.5-50+iqr+robust+z-score+decimate15 | bandpass_0.5-50 | iqr | robust | z-score | decimate15 |
| 221 | widar_221 | bandpass_0.5-50+iqr+robust+min-max+linear15 | bandpass_0.5-50 | iqr | robust | min-max | linear15 |
| 222 | widar_222 | bandpass_0.5-50+iqr+robust+min-max+cubic15 | bandpass_0.5-50 | iqr | robust | min-max | cubic15 |
| 223 | widar_223 | bandpass_0.5-50+iqr+robust+min-max+nearest15 | bandpass_0.5-50 | iqr | robust | min-max | nearest15 |
| 224 | widar_224 | bandpass_0.5-50+iqr+robust+min-max+decimate15 | bandpass_0.5-50 | iqr | robust | min-max | decimate15 |
| 225 | widar_225 | bandpass_0.5-50+outlier_z-score+linear+z-score+linear15 | bandpass_0.5-50 | outlier_z-score | linear | z-score | linear15 |
| 226 | widar_226 | bandpass_0.5-50+outlier_z-score+linear+z-score+cubic15 | bandpass_0.5-50 | outlier_z-score | linear | z-score | cubic15 |
| 227 | widar_227 | bandpass_0.5-50+outlier_z-score+linear+z-score+nearest15 | bandpass_0.5-50 | outlier_z-score | linear | z-score | nearest15 |
| 228 | widar_228 | bandpass_0.5-50+outlier_z-score+linear+z-score+decimate15 | bandpass_0.5-50 | outlier_z-score | linear | z-score | decimate15 |
| 229 | widar_229 | bandpass_0.5-50+outlier_z-score+linear+min-max+linear15 | bandpass_0.5-50 | outlier_z-score | linear | min-max | linear15 |
| 230 | widar_230 | bandpass_0.5-50+outlier_z-score+linear+min-max+cubic15 | bandpass_0.5-50 | outlier_z-score | linear | min-max | cubic15 |
| 231 | widar_231 | bandpass_0.5-50+outlier_z-score+linear+min-max+nearest15 | bandpass_0.5-50 | outlier_z-score | linear | min-max | nearest15 |
| 232 | widar_232 | bandpass_0.5-50+outlier_z-score+linear+min-max+decimate15 | bandpass_0.5-50 | outlier_z-score | linear | min-max | decimate15 |
| 233 | widar_233 | bandpass_0.5-50+outlier_z-score+polynomial_d3+z-score+linear15 | bandpass_0.5-50 | outlier_z-score | polynomial_d3 | z-score | linear15 |
| 234 | widar_234 | bandpass_0.5-50+outlier_z-score+polynomial_d3+z-score+cubic15 | bandpass_0.5-50 | outlier_z-score | polynomial_d3 | z-score | cubic15 |
| 235 | widar_235 | bandpass_0.5-50+outlier_z-score+polynomial_d3+z-score+nearest15 | bandpass_0.5-50 | outlier_z-score | polynomial_d3 | z-score | nearest15 |
| 236 | widar_236 | bandpass_0.5-50+outlier_z-score+polynomial_d3+z-score+decimate15 | bandpass_0.5-50 | outlier_z-score | polynomial_d3 | z-score | decimate15 |
| 237 | widar_237 | bandpass_0.5-50+outlier_z-score+polynomial_d3+min-max+linear15 | bandpass_0.5-50 | outlier_z-score | polynomial_d3 | min-max | linear15 |
| 238 | widar_238 | bandpass_0.5-50+outlier_z-score+polynomial_d3+min-max+cubic15 | bandpass_0.5-50 | outlier_z-score | polynomial_d3 | min-max | cubic15 |
| 239 | widar_239 | bandpass_0.5-50+outlier_z-score+polynomial_d3+min-max+nearest15 | bandpass_0.5-50 | outlier_z-score | polynomial_d3 | min-max | nearest15 |
| 240 | widar_240 | bandpass_0.5-50+outlier_z-score+polynomial_d3+min-max+decimate15 | bandpass_0.5-50 | outlier_z-score | polynomial_d3 | min-max | decimate15 |
| 241 | widar_241 | bandpass_0.5-50+outlier_z-score+stc+z-score+linear15 | bandpass_0.5-50 | outlier_z-score | stc | z-score | linear15 |
| 242 | widar_242 | bandpass_0.5-50+outlier_z-score+stc+z-score+cubic15 | bandpass_0.5-50 | outlier_z-score | stc | z-score | cubic15 |
| 243 | widar_243 | bandpass_0.5-50+outlier_z-score+stc+z-score+nearest15 | bandpass_0.5-50 | outlier_z-score | stc | z-score | nearest15 |
| 244 | widar_244 | bandpass_0.5-50+outlier_z-score+stc+z-score+decimate15 | bandpass_0.5-50 | outlier_z-score | stc | z-score | decimate15 |
| 245 | widar_245 | bandpass_0.5-50+outlier_z-score+stc+min-max+linear15 | bandpass_0.5-50 | outlier_z-score | stc | min-max | linear15 |
| 246 | widar_246 | bandpass_0.5-50+outlier_z-score+stc+min-max+cubic15 | bandpass_0.5-50 | outlier_z-score | stc | min-max | cubic15 |
| 247 | widar_247 | bandpass_0.5-50+outlier_z-score+stc+min-max+nearest15 | bandpass_0.5-50 | outlier_z-score | stc | min-max | nearest15 |
| 248 | widar_248 | bandpass_0.5-50+outlier_z-score+stc+min-max+decimate15 | bandpass_0.5-50 | outlier_z-score | stc | min-max | decimate15 |
| 249 | widar_249 | bandpass_0.5-50+outlier_z-score+robust+z-score+linear15 | bandpass_0.5-50 | outlier_z-score | robust | z-score | linear15 |
| 250 | widar_250 | bandpass_0.5-50+outlier_z-score+robust+z-score+cubic15 | bandpass_0.5-50 | outlier_z-score | robust | z-score | cubic15 |
| 251 | widar_251 | bandpass_0.5-50+outlier_z-score+robust+z-score+nearest15 | bandpass_0.5-50 | outlier_z-score | robust | z-score | nearest15 |
| 252 | widar_252 | bandpass_0.5-50+outlier_z-score+robust+z-score+decimate15 | bandpass_0.5-50 | outlier_z-score | robust | z-score | decimate15 |
| 253 | widar_253 | bandpass_0.5-50+outlier_z-score+robust+min-max+linear15 | bandpass_0.5-50 | outlier_z-score | robust | min-max | linear15 |
| 254 | widar_254 | bandpass_0.5-50+outlier_z-score+robust+min-max+cubic15 | bandpass_0.5-50 | outlier_z-score | robust | min-max | cubic15 |
| 255 | widar_255 | bandpass_0.5-50+outlier_z-score+robust+min-max+nearest15 | bandpass_0.5-50 | outlier_z-score | robust | min-max | nearest15 |
| 256 | widar_256 | bandpass_0.5-50+outlier_z-score+robust+min-max+decimate15 | bandpass_0.5-50 | outlier_z-score | robust | min-max | decimate15 |
| 257 | widar_257 | hampel_w5_s3+iqr+linear+z-score+linear15 | hampel_w5_s3 | iqr | linear | z-score | linear15 |
| 258 | widar_258 | hampel_w5_s3+iqr+linear+z-score+cubic15 | hampel_w5_s3 | iqr | linear | z-score | cubic15 |
| 259 | widar_259 | hampel_w5_s3+iqr+linear+z-score+nearest15 | hampel_w5_s3 | iqr | linear | z-score | nearest15 |
| 260 | widar_260 | hampel_w5_s3+iqr+linear+z-score+decimate15 | hampel_w5_s3 | iqr | linear | z-score | decimate15 |
| 261 | widar_261 | hampel_w5_s3+iqr+linear+min-max+linear15 | hampel_w5_s3 | iqr | linear | min-max | linear15 |
| 262 | widar_262 | hampel_w5_s3+iqr+linear+min-max+cubic15 | hampel_w5_s3 | iqr | linear | min-max | cubic15 |
| 263 | widar_263 | hampel_w5_s3+iqr+linear+min-max+nearest15 | hampel_w5_s3 | iqr | linear | min-max | nearest15 |
| 264 | widar_264 | hampel_w5_s3+iqr+linear+min-max+decimate15 | hampel_w5_s3 | iqr | linear | min-max | decimate15 |
| 265 | widar_265 | hampel_w5_s3+iqr+polynomial_d3+z-score+linear15 | hampel_w5_s3 | iqr | polynomial_d3 | z-score | linear15 |
| 266 | widar_266 | hampel_w5_s3+iqr+polynomial_d3+z-score+cubic15 | hampel_w5_s3 | iqr | polynomial_d3 | z-score | cubic15 |
| 267 | widar_267 | hampel_w5_s3+iqr+polynomial_d3+z-score+nearest15 | hampel_w5_s3 | iqr | polynomial_d3 | z-score | nearest15 |
| 268 | widar_268 | hampel_w5_s3+iqr+polynomial_d3+z-score+decimate15 | hampel_w5_s3 | iqr | polynomial_d3 | z-score | decimate15 |
| 269 | widar_269 | hampel_w5_s3+iqr+polynomial_d3+min-max+linear15 | hampel_w5_s3 | iqr | polynomial_d3 | min-max | linear15 |
| 270 | widar_270 | hampel_w5_s3+iqr+polynomial_d3+min-max+cubic15 | hampel_w5_s3 | iqr | polynomial_d3 | min-max | cubic15 |
| 271 | widar_271 | hampel_w5_s3+iqr+polynomial_d3+min-max+nearest15 | hampel_w5_s3 | iqr | polynomial_d3 | min-max | nearest15 |
| 272 | widar_272 | hampel_w5_s3+iqr+polynomial_d3+min-max+decimate15 | hampel_w5_s3 | iqr | polynomial_d3 | min-max | decimate15 |
| 273 | widar_273 | hampel_w5_s3+iqr+stc+z-score+linear15 | hampel_w5_s3 | iqr | stc | z-score | linear15 |
| 274 | widar_274 | hampel_w5_s3+iqr+stc+z-score+cubic15 | hampel_w5_s3 | iqr | stc | z-score | cubic15 |
| 275 | widar_275 | hampel_w5_s3+iqr+stc+z-score+nearest15 | hampel_w5_s3 | iqr | stc | z-score | nearest15 |
| 276 | widar_276 | hampel_w5_s3+iqr+stc+z-score+decimate15 | hampel_w5_s3 | iqr | stc | z-score | decimate15 |
| 277 | widar_277 | hampel_w5_s3+iqr+stc+min-max+linear15 | hampel_w5_s3 | iqr | stc | min-max | linear15 |
| 278 | widar_278 | hampel_w5_s3+iqr+stc+min-max+cubic15 | hampel_w5_s3 | iqr | stc | min-max | cubic15 |
| 279 | widar_279 | hampel_w5_s3+iqr+stc+min-max+nearest15 | hampel_w5_s3 | iqr | stc | min-max | nearest15 |
| 280 | widar_280 | hampel_w5_s3+iqr+stc+min-max+decimate15 | hampel_w5_s3 | iqr | stc | min-max | decimate15 |
| 281 | widar_281 | hampel_w5_s3+iqr+robust+z-score+linear15 | hampel_w5_s3 | iqr | robust | z-score | linear15 |
| 282 | widar_282 | hampel_w5_s3+iqr+robust+z-score+cubic15 | hampel_w5_s3 | iqr | robust | z-score | cubic15 |
| 283 | widar_283 | hampel_w5_s3+iqr+robust+z-score+nearest15 | hampel_w5_s3 | iqr | robust | z-score | nearest15 |
| 284 | widar_284 | hampel_w5_s3+iqr+robust+z-score+decimate15 | hampel_w5_s3 | iqr | robust | z-score | decimate15 |
| 285 | widar_285 | hampel_w5_s3+iqr+robust+min-max+linear15 | hampel_w5_s3 | iqr | robust | min-max | linear15 |
| 286 | widar_286 | hampel_w5_s3+iqr+robust+min-max+cubic15 | hampel_w5_s3 | iqr | robust | min-max | cubic15 |
| 287 | widar_287 | hampel_w5_s3+iqr+robust+min-max+nearest15 | hampel_w5_s3 | iqr | robust | min-max | nearest15 |
| 288 | widar_288 | hampel_w5_s3+iqr+robust+min-max+decimate15 | hampel_w5_s3 | iqr | robust | min-max | decimate15 |
| 289 | widar_289 | hampel_w5_s3+outlier_z-score+linear+z-score+linear15 | hampel_w5_s3 | outlier_z-score | linear | z-score | linear15 |
| 290 | widar_290 | hampel_w5_s3+outlier_z-score+linear+z-score+cubic15 | hampel_w5_s3 | outlier_z-score | linear | z-score | cubic15 |
| 291 | widar_291 | hampel_w5_s3+outlier_z-score+linear+z-score+nearest15 | hampel_w5_s3 | outlier_z-score | linear | z-score | nearest15 |
| 292 | widar_292 | hampel_w5_s3+outlier_z-score+linear+z-score+decimate15 | hampel_w5_s3 | outlier_z-score | linear | z-score | decimate15 |
| 293 | widar_293 | hampel_w5_s3+outlier_z-score+linear+min-max+linear15 | hampel_w5_s3 | outlier_z-score | linear | min-max | linear15 |
| 294 | widar_294 | hampel_w5_s3+outlier_z-score+linear+min-max+cubic15 | hampel_w5_s3 | outlier_z-score | linear | min-max | cubic15 |
| 295 | widar_295 | hampel_w5_s3+outlier_z-score+linear+min-max+nearest15 | hampel_w5_s3 | outlier_z-score | linear | min-max | nearest15 |
| 296 | widar_296 | hampel_w5_s3+outlier_z-score+linear+min-max+decimate15 | hampel_w5_s3 | outlier_z-score | linear | min-max | decimate15 |
| 297 | widar_297 | hampel_w5_s3+outlier_z-score+polynomial_d3+z-score+linear15 | hampel_w5_s3 | outlier_z-score | polynomial_d3 | z-score | linear15 |
| 298 | widar_298 | hampel_w5_s3+outlier_z-score+polynomial_d3+z-score+cubic15 | hampel_w5_s3 | outlier_z-score | polynomial_d3 | z-score | cubic15 |
| 299 | widar_299 | hampel_w5_s3+outlier_z-score+polynomial_d3+z-score+nearest15 | hampel_w5_s3 | outlier_z-score | polynomial_d3 | z-score | nearest15 |
| 300 | widar_300 | hampel_w5_s3+outlier_z-score+polynomial_d3+z-score+decimate15 | hampel_w5_s3 | outlier_z-score | polynomial_d3 | z-score | decimate15 |
| 301 | widar_301 | hampel_w5_s3+outlier_z-score+polynomial_d3+min-max+linear15 | hampel_w5_s3 | outlier_z-score | polynomial_d3 | min-max | linear15 |
| 302 | widar_302 | hampel_w5_s3+outlier_z-score+polynomial_d3+min-max+cubic15 | hampel_w5_s3 | outlier_z-score | polynomial_d3 | min-max | cubic15 |
| 303 | widar_303 | hampel_w5_s3+outlier_z-score+polynomial_d3+min-max+nearest15 | hampel_w5_s3 | outlier_z-score | polynomial_d3 | min-max | nearest15 |
| 304 | widar_304 | hampel_w5_s3+outlier_z-score+polynomial_d3+min-max+decimate15 | hampel_w5_s3 | outlier_z-score | polynomial_d3 | min-max | decimate15 |
| 305 | widar_305 | hampel_w5_s3+outlier_z-score+stc+z-score+linear15 | hampel_w5_s3 | outlier_z-score | stc | z-score | linear15 |
| 306 | widar_306 | hampel_w5_s3+outlier_z-score+stc+z-score+cubic15 | hampel_w5_s3 | outlier_z-score | stc | z-score | cubic15 |
| 307 | widar_307 | hampel_w5_s3+outlier_z-score+stc+z-score+nearest15 | hampel_w5_s3 | outlier_z-score | stc | z-score | nearest15 |
| 308 | widar_308 | hampel_w5_s3+outlier_z-score+stc+z-score+decimate15 | hampel_w5_s3 | outlier_z-score | stc | z-score | decimate15 |
| 309 | widar_309 | hampel_w5_s3+outlier_z-score+stc+min-max+linear15 | hampel_w5_s3 | outlier_z-score | stc | min-max | linear15 |
| 310 | widar_310 | hampel_w5_s3+outlier_z-score+stc+min-max+cubic15 | hampel_w5_s3 | outlier_z-score | stc | min-max | cubic15 |
| 311 | widar_311 | hampel_w5_s3+outlier_z-score+stc+min-max+nearest15 | hampel_w5_s3 | outlier_z-score | stc | min-max | nearest15 |
| 312 | widar_312 | hampel_w5_s3+outlier_z-score+stc+min-max+decimate15 | hampel_w5_s3 | outlier_z-score | stc | min-max | decimate15 |
| 313 | widar_313 | hampel_w5_s3+outlier_z-score+robust+z-score+linear15 | hampel_w5_s3 | outlier_z-score | robust | z-score | linear15 |
| 314 | widar_314 | hampel_w5_s3+outlier_z-score+robust+z-score+cubic15 | hampel_w5_s3 | outlier_z-score | robust | z-score | cubic15 |
| 315 | widar_315 | hampel_w5_s3+outlier_z-score+robust+z-score+nearest15 | hampel_w5_s3 | outlier_z-score | robust | z-score | nearest15 |
| 316 | widar_316 | hampel_w5_s3+outlier_z-score+robust+z-score+decimate15 | hampel_w5_s3 | outlier_z-score | robust | z-score | decimate15 |
| 317 | widar_317 | hampel_w5_s3+outlier_z-score+robust+min-max+linear15 | hampel_w5_s3 | outlier_z-score | robust | min-max | linear15 |
| 318 | widar_318 | hampel_w5_s3+outlier_z-score+robust+min-max+cubic15 | hampel_w5_s3 | outlier_z-score | robust | min-max | cubic15 |
| 319 | widar_319 | hampel_w5_s3+outlier_z-score+robust+min-max+nearest15 | hampel_w5_s3 | outlier_z-score | robust | min-max | nearest15 |
| 320 | widar_320 | hampel_w5_s3+outlier_z-score+robust+min-max+decimate15 | hampel_w5_s3 | outlier_z-score | robust | min-max | decimate15 |
