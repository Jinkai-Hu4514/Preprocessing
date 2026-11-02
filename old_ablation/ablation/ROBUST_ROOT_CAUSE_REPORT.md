# Robust 导致 Widar / Gait 准确率下降的根因

## 结论

根因已经锁定在 `robust_phase_sanitization()` 的时间去趋势步骤，而不是普通的幅度归一化，也不是单独的复数插值：

> Robust 对每根天线的每个子载波分别估计一个时间斜率，只使用开头 50 个包（本数据约 50 ms），再把各不相同的斜率外推到模型使用的 1500 帧（约 1.5 s）。这既会删掉真实的人体动作时间趋势，也会让不同子载波产生不断累积的相对旋转，最终把频率方向相位关系绕到接近随机。

`first 50` 的长距离外推会加剧问题，但不是唯一根因：把同样 50 个拟合点均匀铺满全序列仍然会破坏相位。真正结构性的错误是“沿时间、逐子载波独立去趋势”。其他相位校准候选是在每一帧沿频率轴去硬件相位误差，两者并不是同一种语义。

Gait 后续的笛卡尔复数插值/decimate 会把已经打乱的相位转换为虚假幅度低谷，是很强的次级放大器；Widar 上该交互较弱。`min-max` 的额外下降主要来自当前执行顺序不一致以及相位/幅度尺度失衡，不能解释 Robust 的基础降点。

本实验没有修改 `SDP/.../src/wsdp` 下的任何源码。实验代码与新结果均在 `ablation/` 下。

## 1. 现有准确率结果的严格配对

从已有 320 组合结果中，只比较除 `calibrate=linear/robust` 外完全相同的行。Widar 有 80 对；Gait 当前已完成部分有 48 对。

| 数据集 | 配对数 | Robust − Linear | z-score 子集 | min-max 子集 |
|---|---:|---:|---:|---:|
| Widar | 80 | **−10.49 pp** | −8.55 pp（40 对） | −12.43 pp（40 对） |
| Gait | 48 | **−47.80 pp** | −37.28 pp（24 对） | −58.33 pp（24 对） |

所有配对明细见 `robust_rootcause_results/accuracy_pairs.csv`，聚合见 `accuracy_effects.json`。

### 最关键的排除实验：z-score + nearest15

Nearest 不混合相邻复数点；linear 与 robust 又都保持原始幅度。随后 z-score 只由幅度计算。因此该配对的幅度通道相同，差别只来自相位校准。

| 数据集 | 配对数 | Linear 均值 | Robust 均值 | 差值 |
|---|---:|---:|---:|---:|
| Widar | 10 | 72.37% | 63.47% | **−8.90 pp** |
| Gait | 6 | 94.01% | 69.01% | **−25.01 pp** |

与图片相同的 `savgol_w7_p3 + IQR + z-score + nearest15` 条件为：

- Widar：75.56% → 63.43%，**−12.12 pp**；
- Gait：95.87% → 66.74%，**−29.13 pp**。

所以复数插值抵消不可能是 Robust 基础降点的必要条件；即使完全不混合复数点，损坏后的相位本身也会使准确率大幅下降。

## 2. 源码实际做了什么

对复数 CSI `H[t,f,a]`，源码先对每个 `(f,a)` 沿时间独立 unwrap：

```text
phase[t,f,a] = unwrap_t(angle(H[t,f,a]))
```

然后每帧、每根天线减掉跨子载波中位相位：

```text
centered[t,f,a] = phase[t,f,a] - median_f(phase[t,f,a])
```

再令 `m=min(T,50)`，对每个 `(f,a)` 分别计算前 50 帧全部点对斜率的中位数：

```text
slope[f,a] = median((centered[j,f,a]-centered[i,f,a])/(j-i)), 0<=i<j<m
```

最后把它外推到整个序列：

```text
corrected[t,f,a] = centered[t,f,a] - t*slope[f,a]
```

两个子载波的附加相对相位因此为：

```text
-(slope[f2,a] - slope[f1,a]) * t
```

只要相邻斜率有微小差异，误差就随帧号线性累积；包裹到 `[-π,π]` 后会反复绕圈。当差值模 `2π` 接近 `π` 时，两个复数向量方向相反。

## 3. 本次组件消融

运行命令：

```text
python ablation/robust_rootcause_ablation.py \
  --dataset all \
  --max-samples-gait 36 \
  --max-samples-widar 4 \
  --diagnostic-samples 36 \
  --equivalence-samples 1 \
  --prefix savgol_iqr \
  --skip-probe \
  --seed 42
```

本机保存的是抽样数据：Gait 36 个诊断样本都来自 user1，Widar 4 个样本都属于同一类别，不能据此重新训练分类器。故准确率证据使用上面的完整服务器结果；本地真实 CSI 用于验证内部信号机制。源码等价性检查中，本实验的 `robust_first50` 与源函数最大复数误差为 0。

### 3.1 “50 帧”是什么时间

| 数据集 | 样本数 | 平均总帧数 | 中位帧间隔 | 前 50 帧跨度 | 前 50 占全样本 | 外推到 1500 帧倍数 |
|---|---:|---:|---:|---:|---:|---:|
| Gait | 36 | 2984.7 | 1000 μs | 49.00 ms | 1.68% | 30.59× |
| Widar | 4 | 1945.3 | 1000 μs | 53.52 ms | 2.57% | 30.59× |

Robust 源码并不使用 timestamp；它只用 `np.arange(T)`。因此“50”是 50 个包，不是秒。采样率越高，同样 50 帧覆盖的物理动作时间越短；丢包或不均匀采样也完全没有进入斜率分母。

### 3.2 前 50 帧斜率不代表全动作

| 数据集 | first50 与 full-span 斜率相关系数 | 斜率 MAE（rad/frame） | 到帧 1499 的相邻未包裹漂移 | 包裹后平均旋转 | 超过 π/2 比例 |
|---|---:|---:|---:|---:|---:|
| Gait | 0.299 | 0.1357 | 153.1 rad | 1.547 rad | 49.1% |
| Widar | 0.371 | 0.0951 | 74.9 rad | 1.499 rad | 45.1% |

约一半相邻子载波到模型序列末端会被额外旋转超过 90°。平均包裹旋转已经接近独立随机相位的 `π/2`。

### 3.3 哪一个内部步骤造成破坏

下表以 `common_only` 为基准。“Linear coherence”定义为 `|interp(H)| / interp(|H|)` 的均值；越低说明笛卡尔插值的向量抵消越严重。

#### Gait（36 个样本）

| 条件 | 后 1/4 相邻相位跳变 | 相对旋转 >π/2 | Linear coherence |
|---|---:|---:|---:|
| common_only | 0.907 rad | 0.0% | 0.935 |
| robust_shared_first50 | 0.907 rad | 0.0% | 0.935 |
| robust_window_limited | 1.282 rad | 31.1% | 0.858 |
| robust_fullspan50 | 1.544 rad | 48.3% | 0.818 |
| **robust_first50（源码）** | **1.562 rad** | **49.8%** | **0.817** |

#### Widar（4 个样本）

| 条件 | 后 1/4 相邻相位跳变 | 相对旋转 >π/2 | Linear coherence |
|---|---:|---:|---:|
| common_only | 0.812 rad | 0.0% | 0.953 |
| robust_shared_first50 | 0.812 rad | 0.0% | 0.953 |
| robust_window_limited | 1.117 rad | 20.1% | 0.903 |
| robust_fullspan50 | 1.538 rad | 48.4% | 0.814 |
| **robust_first50（源码）** | **1.526 rad** | **45.7%** | **0.836** |

这组消融给出三个关键判据：

1. `common_only` 不改变相邻子载波关系；它只是每帧的共同旋转。它可能删除绝对公共相位信息，但不能制造频率方向相消。
2. 把 30 个独立斜率改成每根天线共享一个斜率后，结果与 `common_only` 几乎完全相同。说明破坏相对相位的必要因素是“每个子载波独立斜率”。
3. 在第 49 帧后停止继续增长修正可明显恢复，但不能完全恢复，说明长距离外推是放大因素。把 50 个点均匀铺满全序列仍然接近随机，说明仅把 first50 改成 all-frames 并不能修好；沿时间删除逐子载波趋势本身就与动作信息冲突。

## 4. 为什么插值会继续放大 Gait 的损失

Linear/cubic 在频率轴上分别插值复数的实部和虚部，decimate 也分别滤实部和虚部。若两个复数相位接近反向，混合后幅度会变小，形成原数据不存在的低谷。

信号层：

- Gait 的 linear coherence：linear calibration 0.981，Robust 0.817；Robust 下约 11.3% 的插值点低于幅度插值参考的一半。
- Widar 的 linear coherence：linear calibration 0.998，Robust 0.836；Robust 下约 9.3% 的点低于一半。

完整准确率的 `Robust × interpolation` 交互（相对 nearest）为：

| 数据集 | linear15 | cubic15 | decimate15 |
|---|---:|---:|---:|
| Widar | −0.81 pp | −0.72 pp | +0.22 pp |
| Gait | **−10.74 pp** | **−10.05 pp** | **−20.40 pp** |

因此插值是 Gait 的强放大器，但不是 Widar/Gait 共同的首要根因。

## 5. 为什么 Robust + min-max 更差

已有 320 表的 matched interaction 为：

- Widar：min-max 使 Robust 相对 Linear 的损失再扩大 **3.89 pp**；
- Gait：再扩大 **21.05 pp**。

但当前两条路径执行顺序不同：

```text
z-score: prefix -> interpolate -> z-score -> explicit [amplitude, phase]
min-max: prefix -> min-max complex -> interpolate -> Dataset abs/angle
```

所以这不是纯 normalization 2×2，不能把全部交互直接归因于 min-max。

本次样本统计给出的主要放大机制是通道尺度：

| 数据集 | Robust phase std | z-score amplitude std | phase/amp | min-max amplitude std | phase/amp |
|---|---:|---:|---:|---:|---:|
| Gait | 1.814 | 1.000 | 1.81× | 0.219 | **8.31×** |
| Widar | 1.817 | 1.000 | 1.82× | 0.215 | **8.54×** |

Min-max 后，已经被 Robust 打乱的 phase 通道在标准差上约为 amplitude 的 8 倍，更容易主导 MLP。Min-max 产生精确零幅度、使 `angle(0)=0` 的比例只有约 0.06%–0.07%，属于较小因素。本实验没有发现 min-max 会单独制造比原 Robust 更强的相消。

## 6. 对图片中三个猜想的最终判定

1. **“公共相位包含动作信息，被 Robust 删除”**：代码上成立，可能有贡献，但不是 Robust 相对 Linear 的主因。Linear 也会去除每包频域拟合的截距；而 `common_only` 不破坏相邻子载波关系。本机抽样只有单类，不能声称已经证明公共相位的类别信息量。
2. **“前 50 帧斜率外推使相位差越来越大”**：成立，但需要修正表述。未包裹的相对修正随帧号线性增大；包裹相位会反复绕圈，不是单调增大。停止外推能部分恢复，但 full-span 仍坏，因此 first50 是放大器，逐子载波时间去趋势才是根因。
3. **“方向相反后复数插值相消”**：成立，且在 Gait 上会额外造成约 10–20 pp 损失；但 nearest 条件仍大降，故它不是基础根因。

## 7. 代码级根因优先级

1. **轴与任务语义错配**：把人体动作的时间趋势当作硬件误差，逐 `(subcarrier, antenna)` 删除。
2. **独立斜率破坏相对相位**：小斜率差乘以 1500 帧后绕圈，频率相位结构接近随机。
3. **短窗口长外推**：约 50 ms 的估计外推约 30 倍，放大初始噪声、动作起始差异及 unwrap 分支差异。
4. **Gait 的复数插值/decimate 放大**：把相位错误进一步变成幅度错误。
5. **min-max 尺度与顺序交互**：让损坏相位相对幅度占据更大数值尺度。
6. **可能的次级问题**：源码先让各子载波独立 temporal unwrap、之后才减 common phase；不同 `2π` 分支可能被斜率估计器当成趋势。

## 8. 复现文件

- 实验程序：`ablation/robust_rootcause_ablation.py`
- 完整配对：`ablation/robust_rootcause_results/accuracy_pairs.csv`
- 准确率聚合：`ablation/robust_rootcause_results/accuracy_effects.json`
- Gait 信号诊断：`ablation/robust_rootcause_results/gait/`
- Widar 信号诊断：`ablation/robust_rootcause_results/widar/`

运行中只读取原源码和既有结果，不会写入或修改 WSDP 源码。
