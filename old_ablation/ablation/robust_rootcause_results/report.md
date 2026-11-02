# Robust phase sanitization root-cause ablation

This file is generated from the committed 320-combination accuracy tables and the real-CSI component diagnostics in this directory.

## Paired full-training evidence

| Dataset | Pairs | Robust − linear | z-score | min-max | min-max interaction |
|---|---:|---:|---:|---:|---:|
| widar | 80 | -10.49 pp | -8.55 pp | -12.43 pp | -3.89 pp |
| gait | 48 | -47.80 pp | -37.28 pp | -58.33 pp | -21.05 pp |

A negative interaction means min-max amplifies the loss caused by robust.

## Signal/component runs

- gait: ok, samples=180, probe=True, duration=181.6s
- widar: ok, samples=180, probe=True, duration=123.4s

## Interpretation rule

- A large loss for nearest15 proves interpolation cancellation is not required.
- Recovery from robust_first50 to robust_window_limited identifies long-horizon extrapolation; recovery to robust_fullspan50 identifies the first-50 fit window.
- A remaining loss in common_only identifies removal of label-bearing common phase.
- Lower Cartesian cancellation ratios for robust_first50 identify complex interpolation as an amplifier, not the sole root cause.

See `accuracy_effects.json`, each dataset's `slope_diagnostics.json`, `variant_diagnostics.json`, and `probe_accuracy.json` for the numeric evidence.
