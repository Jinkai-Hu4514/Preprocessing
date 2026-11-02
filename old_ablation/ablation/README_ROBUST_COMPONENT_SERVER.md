# Robust 组件消融：简化服务器版

入口：`ablation/robust_component_server_ablation.py`

## 数据流程

```text
readers.load_data（自动跳过格式不匹配和读取失败文件）
→ 解析、排序并过滤空样本/短样本
→ savgol(window=7, polyorder=3) + IQR(factor=1.5)
→ phase condition
→ nearest(target_K=15)
→ per-sample z-score amplitude + wrapped phase
→ truncate/pad to 1500
→ mlpmodel
```

原始数据只读取一次，公共 `savgol+IQR` 只处理一次。公共前缀保存在内存中，
每次只构造当前条件的模型输入；条件训练结束后立即释放。

脚本不会再创建以下缓存：

```text
cache/prefix_xxx/000000.npy
cache/prefix_xxx/000001.npy
...
```

磁盘只保留 checkpoint、训练 history、测试预测、split 和结果汇总。旧版已经生成的
`ablation/robust_component_server_results/<dataset>/cache` 不会再使用，可以在确认
旧任务已经停止后手动删除。新版默认输出到：

```text
ablation/robust_component_results/
```

## 固定口径

- 模型 seed：42
- group split seed：42
- Gait：60 epochs
- Widar：80 epochs
- 模型：`mlpmodel`
- 默认设备：`cuda:1`
- 最终输入：`float32 (N,1500,15,6)`，显式 `[amplitude, phase]`
- `robust_first50` 直接调用当前源码 `robust_phase_sanitization()`

`core` 包含八个条件：

1. `linear_reference`
2. `robust_first50`
3. `no_calibration`
4. `common_only`
5. `detrend_first50_only`
6. `robust_shared_first50`
7. `robust_window_limited`
8. `robust_fullspan50`

## 服务器运行

先检查路径和 GPU：

```bash
python -u ablation/robust_component_server_ablation.py \
  --dataset gait --suite core --dry-run

python -u ablation/robust_component_server_ablation.py \
  --dataset widar --suite core --dry-run
```

正式运行：

```bash
python -u ablation/robust_component_server_ablation.py \
  --dataset gait --suite core --workers 4

python -u ablation/robust_component_server_ablation.py \
  --dataset widar --suite core --workers 4
```

默认已经指定 `cuda:1`，不要同时设置 `CUDA_VISIBLE_DEVICES=1`。后者会把物理 GPU 1
重新编号为进程内的 `cuda:0`。如果服务器调度器只暴露一张卡，应显式传入调度器
给出的逻辑设备编号。

只验证读取与预处理、不训练：

```bash
python -u ablation/robust_component_server_ablation.py \
  --dataset gait \
  --conditions robust_first50 \
  --max-samples 20 \
  --preprocess-only \
  --workers 2
```

使用 `--max-samples` 的输出只能作为冒烟测试，不能作为正式 ACC。

## 内存与结果

无磁盘缓存意味着需要足够内存。全量 Gait 建议至少准备约 64 GiB 可用 RAM；运行时
会打印公共前缀和当前模型输入占用。Gait 与 Widar 建议顺序运行。

重点结果：

```text
ablation/robust_component_results/<dataset>/summary.csv
ablation/robust_component_results/<dataset>/contrasts.json
ablation/robust_component_results/<dataset>/split_indices.npz
ablation/robust_component_results/<dataset>/<condition>/seed_42/status.json
ablation/robust_component_results/<dataset>/<condition>/seed_42/test_predictions.npz
```

判断方式：

- `robust_first50 << common_only`：逐子载波时间去趋势是主要损失源。
- `robust_shared_first50 ≈ common_only`：不同子载波使用不同斜率造成破坏。
- `robust_window_limited > robust_first50`：长期外推放大损失。
- `robust_fullspan50 > robust_first50`：只使用前 50 帧估计斜率不稳定。
- `robust_fullspan50` 仍低于 `common_only`：逐子载波去趋势本身删除了动作信息。
