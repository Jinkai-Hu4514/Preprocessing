# Bandpass paper figures

All quantitative panels are rendered directly from the local experiment outputs. PNG files are intended for quick review; PDF and SVG files are the publication/editable versions.

## Suggested captions

**Table X. Classification performance of five denoisers over the 80 XRF55 preprocessing configurations.** Each denoiser is evaluated under the same 16 downstream configurations. Bandpass achieves the lowest accuracy in all 16 matched cells. All runs use seed 42.

**Fig. X. Sampling-rate mismatch and its classification effect on XRF55.** (a) Effective zero-phase response of the configured and sampling-rate-aligned filters on the physical 200-Hz time axis. Setting the filter parameter to 1000 Hz shifts the intended 0.5–50-Hz passband to 0.1–10 Hz. (b) Test accuracy under the matched IQR + z-score + cubic15 pipeline. Aligning the parameter with 200 Hz increases accuracy from 38.18% to 61.97%, while a 23.18-percentage-point gap to the matched savgol result remains. Results in (b) are single-seed measurements.

**Fig. Y. Sign folding after bandpass filtering on XRF55.** (a) Illustrative trace showing that absolute-value processing reflects the negative half-wave above zero. (b) Full audit of 3,300 samples. Approximately half of the bandpass output lies on the meaningful negative half-axis, and absolute-value processing changes or removes approximately half of the local slope directions. Panel (a) is illustrative; the population-level statement is supported by panel (b).

**Fig. Z. Short-sequence bypass behavior of bandpass filtering on ElderAL.** (a) Distribution of raw sequence lengths. Because zero-phase filtering requires at least 28 frames, 2,010 of 2,404 valid segments (83.61%) are returned unchanged. (b) Input–output NRMSE for bypassed and actually filtered segments. All bypassed segments are exact matches, whereas the 394 filtered segments have a median NRMSE of 0.897. NRMSE quantifies transformation magnitude and should not be interpreted by itself as classification-information loss.

## Reproduction

Run the four scripts in this directory from the repository root:

```bash
python ablation/bandpass_paper_figures/plot_table_x.py
python ablation/bandpass_paper_figures/plot_fig_x_sampling_rate.py
python ablation/bandpass_paper_figures/plot_fig_y_sign_folding.py
python ablation/bandpass_paper_figures/plot_fig_z_elder_bypass.py
```
