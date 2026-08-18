# M77 retinal-motion causal chain

This production path tests a visual-only causal chain with the frozen M77
epoch-279 checkpoint and RR100 population. Every neural probe and natural-movie
replay supplies the same all-zero 42-dimensional behavior vector.

## Analysis contract

- Input and supervision are both native 240 Hz.
- Every fixation has 60 real history samples plus a distinct 240-sample,
  one-second scoring interval.
- Primary eye traces are zero-phase filtered on the continuous raw DDPI grid
  (100-Hz passband, 118-Hz stopband) before sampling at 240 Hz. Raw-resampled
  traces are retained as a sensitivity control. Saved saccade epochs receive
  50-ms guards, and a conservative 30-deg/s peak-speed fallback rejects
  residual saccade-like jumps missed by the event file.
- The primary spectrum is computed from each directly rendered 151-pixel M77
  movie with a spatial Tukey window, temporal mean removal, and two DPSS
  tapers. Positive and negative temporal frequencies are folded because the
  grating bank measures orientation, not motion direction. No instantaneous
  velocity or `k dot v` approximation is used.
- The auxiliary 255-pixel spectrum is for clean population visualization only.
  It never replaces the exact 151-pixel scorer aperture for prediction.
- M77 outputs are expected spike counts per native bin. Reported mean rates are
  multiplied by 240 Hz; expected spikes are summed across the one-second
  interval.
- All motion effects use the matched stabilized rendering of the same image.

## Production order

Run commands in the `yatesfv` environment from the repository root.

1. Build the shared fixation bank:

   ```bash
   python paper/fig4/spatiotemporal_tuning/build_real_fixation_bank.py \
     --out-dir outputs/dekel240_paper/m77_epoch279/retinal_causal_chain/fixation_bank \
     --n-traces 100 --candidates-per-session 24
   ```

2. Measure dense RR100 tuning (25 SF × 33 nonzero TF × 8 orientations × 32
   phases, plus matched TF=0 controls):

   ```bash
   python paper/fig4/spatiotemporal_tuning/run_native_periodic_tuning.py \
     /mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/dekel240/D240M77c_dekel_native240_freqmasked_floor0p5_gnlrnalpha0p1_presplit_s201/analysis_candidates/epoch=279-val_bps_overall=0.5912.ckpt \
     --dataset-config paper/model_selection/configs/multi_240_long_split3_dekel35.yaml \
     --grid-mode dense --dense-n-spatial 25 --dense-n-temporal 33 \
     --dense-n-orientations 8 --n-phases 32 \
     --out-dir outputs/dekel240_paper/m77_epoch279/retinal_causal_chain/dense_tuning
   ```

3. Run renderer-faithful causal-chain shards with
   `run_m77_retinal_causal_chain.py`. Production shards must use disjoint image
   ranges and the complete trace range. Repeat with `--trace-kind raw` and with
   `--trace-transform rotate90` for the two replay controls.

4. Merge and analyze shards:

   ```bash
   python paper/fig4/spatiotemporal_tuning/analyze_m77_retinal_causal_chain.py \
     <filtered shard directories> --raw-shards <raw shard directories> \
     --rotated-shards <rotated shard directories> \
     --out-dir outputs/dekel240_paper/m77_epoch279/retinal_causal_chain/analysis
   ```

5. Generate the clean auxiliary power visualization and direct-rendered versus
   full-trajectory signal-processing validation with
   `run_m77_retinal_power_visualization.py` and
   `validate_m77_retinal_spectrum.py`.

6. Run the 20-image × 20-trace nonlinear audit with
   `audit_m77_nonlinear_sharpening.py`. The tangent is a symmetric finite
   directional derivative around stabilization. A direct JVP is undefined in
   exact-zero split-ReLU patches entering Lp pooling, so the command compares
   0.10 and 0.05 central steps and saves their relative disagreement.

## Claim gates

The generated summaries expose gates; figure text must follow them.

- Stabilized dynamic power must be numerically zero.
- Direct-rendered versus complete-trajectory spectrum cosine must be at least
  0.97 in aggregate and 0.85 for the median image–trace pair.
- The signed full joint SF×TF×orientation projection must predict held-out rate
  changes and improve over total-power and separable controls under crossed
  image/trace validation.
- Rate and spatial-map SSI increases require hierarchical confidence intervals
  above zero.
- Nonlinear sharpening requires the full replay to increase RR100 spatial
  information and the matched tangent replay to underestimate that increase.
- Filtered-versus-raw dependence is reported rather than hidden if qualitative
  conclusions differ.

Boundary or ambiguous fitted tuning peaks are censored. Peak fits are only
annotations: projections always use the complete measured tuning tensor.
