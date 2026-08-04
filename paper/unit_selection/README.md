# Unit selection

`generate_session_configs.py` defines **which units enter every analysis in the
paper**. It is a one-shot preprocessing script, not a figure script, but it sits
here because its output is the population that fig1--fig4 and both supplements
all inherit.

## What it does

For each session it computes spike-triggered energies (STEs) from the
**gaborium** (RF-mapping) condition, derives a per-unit SNR, and writes

```
experiments/dataset_configs/sessions/<session>.yaml
```

with four unit lists: `cids`, `visual`, `qcmissing`, `qccontam`.

**`cids` is set equal to `visual`.** The two spike-sorting QC lists are recorded
in the YAML but are *not* applied to `cids` — nothing downstream intersects
them. If you are looking for the criterion that decides the analyzed
population, it is `visual` and nothing else.

## The criterion

A unit is "visual" if, in the gaborium condition,

```
cluster_snr > 5   AND   num_spikes >= 100
```

- `num_spikes` counts spikes in bins with valid eye tracking (`robs * dpi_valid`).
- `cluster_snr` is the peak STE SNR over 20 lags:
  `signal = |STE - median_xy(STE)|`, smoothed with
  `gaussian_filter(signal, [0, 2, 2, 2])`; `noise` is the spatial median of the
  lag-0 signal map; `snr_per_lag = max_xy(signal) / noise`; `cluster_snr` is its
  maximum across lags.

Downstream, the per-session `cids` are read by `models/config_loader.py`
(`load_dataset_configs`) via the parent config's `session_dir`, so any dataset
config built on `experiments/dataset_configs/multi_basic_*.yaml` gets this
population. Analysis-specific criteria (firing rate, split-half PSTH R^2,
session unit floor) are applied *on top* of it — see
`paper/covariance_decomposition/derive.py`.

## Do not confuse this with the Figure 1C receptive-field filter

Figure 1C draws RF contours using a **different and stricter** screen that has
no effect on the analyzed population:

|  | unit selection (this script) | Fig 1C contours |
|---|---|---|
| SNR threshold | `> 5` | `> 9` |
| spike threshold | `>= 100` | `> 200` |
| STE smoothing | `gaussian_filter(..., [0, 2, 2, 2])` | `gaussian_filter(..., [0, 1, 1, 1])` |
| extra screen | none | contour circularity `>= 0.9` |
| effect | sets `cids` for all analyses | selects which contours are drawn |

See `paper/fig1/rf_contours.py` (local `compute_snr`, sigma 1) and
`eval/sta_ste.py:compute_snr` (also sigma 1). Because the smoothing differs, the
two SNR values are not on the same scale and the thresholds are not comparable.
This distinction has repeatedly been mistaken for an inconsistency; it is not.

## Running it

The output path is relative to the current working directory, so run it from
the repository root:

```bash
cd VisionCore
uv run python paper/unit_selection/generate_session_configs.py
```

Other fixed parameters: `n_lags = 20`, `missing_thresh = 25`,
`contam_thresh = 50` (the latter two only populate `qcmissing` / `qccontam`).

Regenerating the YAMLs changes the analyzed population and invalidates every
cache under `outputs/cache/`.
