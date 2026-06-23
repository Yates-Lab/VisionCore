"""
Figure 2 precomputation: load raw data, run LOTC covariance decomposition,
then derive every per-window/per-panel statistic the fig2 panel scripts
need. Cached as a single bundle so panel scripts load instantly.

Rowley/Luke sessions use a Rowley-specific inclusion adapter before the
decomposition: start from sorter contamination-passing V1 units when available,
replace the older Gaborium/YAML visual list with primary-eye dots RF SNR, and
drop units with excess FixRSVP missingness. The downstream plotted-cell quality
gates (rate + FixRSVP split-half PSTH R^2) remain shared across subjects.

Bundle keys returned by ``load_fig2_data(refresh=False)``:
    session_results, metrics, m_by_window, subject_per_neuron_by_window,
    alpha_stats, fano_stats, nc_stats,
    sub_names, sub_subjects, pr_fem_list, pr_psth_list,
    overlap_k1_list, overlap_k_list, var_p_given_f, var_f_given_p,
    spectra_psth, spectra_fem,
    WINDOWS_MS, WINDOWS_BINS, SUBJECTS, SUBJECT_COLORS,
    session_names, subjects, n_sessions,
    SUBSPACE_WINDOW_IDX, SUBSPACE_K,
    config (dict of analysis parameters).

Two caches on disk:
    CACHE_DIR/fig2_decomposition_yates_rowley.pkl   raw per-session decompositions
    CACHE_DIR/fig2_derived_yates_rowley.pkl         derived bundle (everything above)

Set REFRESH=True at the top, pass refresh=True to load_fig2_data(), or
delete the cache file to force recompute.

Usage:
    from compute_fig2_data import load_fig2_data
    data = load_fig2_data()
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import dill

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR
from VisionCore.covariance import (
    cov_to_corr,
    project_to_psd,
    get_upper_triangle,
    align_fixrsvp_trials,
    run_covariance_decomposition,
)
from VisionCore.stats import (
    geomean,
    iqr_25_75,
    bootstrap_mean_ci,
    fisher_z_mean,
    emp_p_one_sided,
    wilcoxon_signed_rank,
    paired_valid,
)
from VisionCore.subspace import (
    participation_ratio,
    symmetric_subspace_overlap,
    directional_variance_capture,
)
from DataYatesV1 import get_free_device


# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
REFRESH = False              # set True to force recompute of derived bundle
DT = 1 / 120                 # seconds per bin (native 240 Hz sampling)
WINDOW_BINS = [1, 2, 3, 6]   # counting windows in bins (6 @ 120 Hz = 50 ms = stim refresh)
N_SHUFFLES = 100             # shuffle null iterations
N_STAGE1_WORKERS = 8         # parallel session decompositions (set >1 to fan out)
N_STAGE1_GPUS = 2            # GPUs to distribute workers across (round-robin)
MIN_RATE_HZ = 2.0            # firing-rate inclusion threshold
MIN_PSTH_R2 = 0.05           # split-half PSTH R^2 inclusion threshold
N_PSTH_SPLITS = 100          # random halvings for split-half PSTH reliability
ROWLEY_DOTS_SNR_THRESH = 5.0 # visual criterion from test_rowley12 dots RF path
ROWLEY_MAX_NAN_FRAC = 0.20   # max FixRSVP NaN fraction within valid bins
ROWLEY_PROCESSED_ROOT = Path("/mnt/ssd2/RowleyMarmoV1V2/processed")
ROWLEY_STEP07_VISUAL_RF_GEOMETRY = (
    VISIONCORE_ROOT.parent
    / "DataRowleyV1V2"
    / "outputs"
    / "luke_step07_rf_survey"
    / "luke_step07_visual_rf_geometry.csv"
)
ROWLEY_STEP07_RELIABILITY = (
    VISIONCORE_ROOT
    / "outputs"
    / "stats"
    / "fig2_rf_compare"
    / "luke_step07_visual_rf_valid_split_half_reliability.csv"
)
ROWLEY_DOTS_ROI_DEG = np.array([[-5, 5], [-5, 5]], dtype=np.float32)
ROWLEY_DOTS_DXY_DEG = 0.2
ROWLEY_DOTS_STA_LAGS = np.arange(2, 8)
INTERCEPT_MODE = "below_threshold"
INTERCEPT_THRESHOLD = 0.05
INTERCEPT_KWARGS = {"threshold": INTERCEPT_THRESHOLD}
MIN_VAR = 0                  # minimum variance for correlation computation
EPS_RHO = 1e-3               # floor for correlation denominators
SUBJECTS = ["Allen", "Logan", "Luke"]
SUBJECT_COLORS = {"Allen": "tab:blue", "Logan": "tab:green", "Luke": "tab:orange"}

DATASET_CONFIGS_PATH = (
    VISIONCORE_ROOT
    / "experiments"
    / "dataset_configs"
    / "multi_basic_120_long_yates_rowley.yaml"
)

SUBSPACE_WINDOW_IDX = 1      # second window (4 bins ≈ 16.67 ms)
SUBSPACE_K = 5

DECOMP_CACHE = CACHE_DIR / "fig2_decomposition_yates_rowley.pkl"
DERIVED_CACHE = CACHE_DIR / "fig2_derived_yates_rowley.pkl"

# Until the Yates+Rowley pipeline is recomputed, fall back to the existing
# Yates-only caches so the figure renders for layout iteration. Once the
# *_yates_rowley caches exist, they take precedence automatically.
if not DECOMP_CACHE.exists():
    DECOMP_CACHE = CACHE_DIR / "fig2_decomposition.pkl"
if not DERIVED_CACHE.exists():
    DERIVED_CACHE = CACHE_DIR / "fig2_derived.pkl"


def _is_rowley_config(cfg):
    """Return True for Rowley-session configs."""
    return str(cfg.get("lab", "")).lower() == "rowley"


def _rowley_initial_cid_pool(cfg):
    """Return the Rowley pre-decomposition CID pool and its source field."""
    if _is_luke_step07_session(cfg):
        reliable = _rowley_step07_reliable_cids(cfg)
        if ROWLEY_STEP07_RELIABILITY.exists():
            return reliable, "step07_split_half_psth_r2"
        return _rowley_step07_visual_rf_valid_cids(cfg), "step07_visual_rf_valid"
    for key in ("sortercontam", "qccontam", "cids"):
        values = np.asarray(cfg.get(key, []), dtype=int)
        if values.size > 0:
            return values, key
    return np.asarray([], dtype=int), "none"


def _is_luke_step07_session(cfg):
    return (
        str(cfg.get("session", "")).startswith("Luke_")
        and ROWLEY_STEP07_VISUAL_RF_GEOMETRY.exists()
    )


def _rowley_step07_visual_rf_valid_cids(cfg):
    """Luke step07 visual units with plausible dots RFs.

    These units are measured in the DataRowleyV1V2 step07 RF survey using raw
    dots-calibration STAs, PRL correction, and an RF-valid gate of eccentricity
    <= 1 deg and half-peak diameter <= 2 deg. The shared fig2 rate and
    split-half PSTH reliability gates are still applied downstream.
    """
    session = cfg.get("session")
    eye = cfg.get("eye")
    if not session or not eye or not ROWLEY_STEP07_VISUAL_RF_GEOMETRY.exists():
        return np.asarray([], dtype=int)

    try:
        import pandas as pd

        survey = pd.read_csv(ROWLEY_STEP07_VISUAL_RF_GEOMETRY)
    except Exception as e:
        print(f"  [{session}] Could not load step07 RF-valid units: {e}")
        return np.asarray([], dtype=int)

    keep = (
        (survey["session"].astype(str) == str(session))
        & (survey["eye"].astype(str) == str(eye))
        & (survey["status"].astype(str) == "ok")
        & survey["rf_valid"].astype(bool)
    )
    return np.asarray(survey.loc[keep, "cluster_id"], dtype=int)


def _rowley_step07_reliable_cids(cfg):
    """Luke step07 RF-valid units that passed the diagnostic split-half gate."""
    session = cfg.get("session")
    eye = cfg.get("eye")
    if not session or not eye or not ROWLEY_STEP07_RELIABILITY.exists():
        return np.asarray([], dtype=int)

    try:
        import pandas as pd

        reliability = pd.read_csv(ROWLEY_STEP07_RELIABILITY)
    except Exception as e:
        print(f"  [{session}] Could not load step07 reliability units: {e}")
        return np.asarray([], dtype=int)

    keep = (
        (reliability["session"].astype(str) == str(session))
        & (reliability["eye"].astype(str) == str(eye))
        & (reliability["status"].astype(str) == "ok")
        & reliability["passes_psth_r2_gate"].astype(str).str.lower().eq("true")
    )
    return np.asarray(reliability.loc[keep, "cid"], dtype=int)


def _resolve_rowley_session_root(dataset_directory):
    """Walk up from a dataset dir to the Rowley session root."""
    p = Path(dataset_directory)
    for candidate in [p, *p.parents]:
        if (candidate / "dpi_calibration").exists() or (
            candidate / "dots_calibration"
        ).exists():
            return candidate
    return p.parent


def _resolve_rowley_dataset_directory(cfg):
    """Normalize stale Rowley dataset paths before prepare_data loads them."""
    dataset_dir = Path(cfg.get("directory", ""))
    if dataset_dir.exists():
        return str(dataset_dir), cfg.get("eye", "right")

    candidates = []
    if "/processed_declan/" in str(dataset_dir):
        candidates.append(Path(str(dataset_dir).replace(
            "/processed_declan/", "/processed/"
        )))

    session_name = cfg.get("session")
    eye = cfg.get("eye", "right")
    if session_name:
        candidates.append(ROWLEY_PROCESSED_ROOT / session_name / "datasets" / f"{eye}_eye")
        candidates.append(
            ROWLEY_PROCESSED_ROOT / session_name / "datasets_gaussian" / f"{eye}_eye"
        )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate), eye

        parent = candidate.parent
        for alt_eye in ("left", "right"):
            sibling = parent / f"{alt_eye}_eye"
            if sibling.exists():
                print(
                    f"  [{session_name}] Rowley directory fallback: "
                    f"{dataset_dir} -> {sibling} (eye={alt_eye})"
                )
                return str(sibling), alt_eye

    return str(dataset_dir), eye


def _to_numpy(x):
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(x)


def _as_bool_1d(x, n_expected=None):
    x = np.asarray(x).reshape(-1)
    x = x > 0.5 if x.dtype != bool else x
    if n_expected is not None and len(x) != n_expected:
        raise ValueError(f"Expected {n_expected} values, got {len(x)}")
    return x.astype(bool)


def _nearest_resample_bool(sample_times, values, target_times):
    sample_times = np.asarray(sample_times, dtype=np.float64)
    values = _as_bool_1d(values, len(sample_times))
    target_times = np.asarray(target_times, dtype=np.float64)
    right_idx = np.searchsorted(sample_times, target_times, side="left")
    right_idx = np.clip(right_idx, 0, len(sample_times) - 1)
    left_idx = np.clip(right_idx - 1, 0, len(sample_times) - 1)
    choose_left = (
        np.abs(target_times - sample_times[left_idx])
        <= np.abs(sample_times[right_idx] - target_times)
    )
    return values[np.where(choose_left, left_idx, right_idx)]


def _interp_xy(sample_times, xy, target_times):
    sample_times = np.asarray(sample_times, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float32)
    target_times = np.asarray(target_times, dtype=np.float64)
    return np.column_stack([
        np.interp(target_times, sample_times, xy[:, 0]),
        np.interp(target_times, sample_times, xy[:, 1]),
    ]).astype(np.float32)


def _load_rowley_dots_snr_for_cids(cfg, cids):
    """Load primary-eye dots RF SNR values for the requested Rowley CIDs."""
    session_root = _resolve_rowley_session_root(cfg.get("directory", ""))
    eye = cfg.get("eye", "right")
    cids = np.asarray(cids)

    cal_path = (
        session_root
        / "dots_calibration"
        / f"{eye}_eye"
        / "calibration_results.npz"
    )
    if cal_path.exists():
        data = np.load(cal_path, allow_pickle=True)
        if "calibration_cluster_ids" in data and "optimized_max_snr" in data:
            snr_cids = np.asarray(data["calibration_cluster_ids"])
            max_snr = np.asarray(data["optimized_max_snr"])
            snr_by_cid = {
                int(cid): float(snr) for cid, snr in zip(snr_cids, max_snr)
            }
            snr = np.array([snr_by_cid.get(int(cid), np.nan) for cid in cids])
            if np.isfinite(snr).any():
                return snr
            print(
                f"  [{cfg['session']}] Dots calibration IDs did not match "
                f"requested Rowley CIDs in {cal_path}; trying fallback cache"
            )
        else:
            print(
                f"  [{cfg['session']}] Dots calibration result missing ID/SNR "
                f"keys: {cal_path}; trying fallback cache"
            )

    snr_path = session_root / "dpi_calibration" / f"{eye}_eye" / "dots_rf_snr.npz"
    if snr_path.exists():
        data = np.load(snr_path, allow_pickle=True)
        snr_cids = np.asarray(data["cids"])
        max_snr = np.asarray(data["max_snr"])
        snr_by_cid = {int(cid): float(snr) for cid, snr in zip(snr_cids, max_snr)}
        return np.array([snr_by_cid.get(int(cid), np.nan) for cid in cids])

    return _compute_rowley_dots_snr(session_root, eye, cids, snr_path)


def _prepare_rowley_unit_mapping(cfg, rowley_cids):
    """Validate Rowley CIDs against fixRSVP metadata.

    Rowley session YAMLs store the V1 sorter/QC pool as cluster IDs.
    The newer processed_declan datasets preserve those IDs in metadata, while
    older raw datasets may only expose robs columns. Prefer metadata IDs when
    present so loading sorter/QC-passing V1 units does not fall back to the older
    Gaborium visual list.
    """
    from DataYatesV1 import DictDataset

    fix_path = Path(cfg["directory"]) / "fixrsvp.dset"
    if not fix_path.exists():
        print(f"  [{cfg['session']}] Rowley fixrsvp missing: {fix_path}")
        return None

    dset = DictDataset.load(fix_path)
    n_cols = int(dset["robs"].shape[1])
    rowley_cids = np.asarray(rowley_cids, dtype=int)
    if rowley_cids.size == 0 or np.any(rowley_cids < 0):
        print(
            f"  [{cfg['session']}] Skipping Rowley session: empty/negative "
            "CID pool values"
        )
        return None

    region = np.asarray(dset.metadata.get("region", []))
    if region.shape[0] != n_cols:
        print(
            f"  [{cfg['session']}] Skipping Rowley session: fixrsvp region "
            "metadata is missing or has the wrong length"
        )
        return None

    region_ok = region == cfg.get("region", "V1")
    if not np.any(region_ok):
        print(f"  [{cfg['session']}] Skipping Rowley session: no V1 columns found")
        return None

    stored_cids = None
    for key in ("cluster_ids", "all_cids", "cids"):
        value = dset.metadata.get(key, None)
        if value is None:
            continue
        value = np.asarray(value)
        if value.ndim == 1 and value.shape[0] == n_cols:
            stored_cids = value.astype(int, copy=False)
            break

    if stored_cids is not None:
        cid_to_col = {int(cid): col for col, cid in enumerate(stored_cids.tolist())}
        col_indices = np.array(
            [cid_to_col.get(int(cid), -1) for cid in rowley_cids],
            dtype=int,
        )
        missing = rowley_cids[col_indices < 0]
        if missing.size:
            print(
                f"  [{cfg['session']}] Skipping Rowley session: CID pool "
                f"CIDs missing from fixrsvp metadata, e.g. {missing[:10].tolist()}"
            )
            return None
        non_region = rowley_cids[~region_ok[col_indices]]
        if non_region.size:
            print(
                f"  [{cfg['session']}] Skipping Rowley session: CID pool "
                f"contains non-{cfg.get('region', 'V1')} CIDs, "
                f"e.g. {non_region[:10].tolist()}"
            )
            return None

        # prepare_data will map these cluster IDs through the same metadata.
        return rowley_cids, rowley_cids

    if np.max(rowley_cids) >= n_cols:
        print(
            f"  [{cfg['session']}] Skipping Rowley session: fixrsvp has no "
            f"cluster_ids metadata and CID pool values do not index columns "
            f"(n_cols={n_cols}, range={rowley_cids.min()}-{rowley_cids.max()})"
        )
        return None

    if not np.all(region_ok[rowley_cids]):
        non_region = rowley_cids[~region_ok[rowley_cids]]
        print(
            f"  [{cfg['session']}] Skipping Rowley session: CID pool column "
            f"indices include non-{cfg.get('region', 'V1')} units, "
            f"e.g. {non_region[:10].tolist()}"
        )
        return None

    return rowley_cids, rowley_cids


def _compute_rowley_dots_snr(session_root, eye, target_cids, cache_path):
    """Compute dots RF SNR from Rowley dots-calibration inputs."""
    import pandas as pd
    from DataYatesV1 import DictDataset
    from DataRowleyV1V2.dots_calibration.training import (
        bin_dots_to_stimulus,
        calculate_rf_snr,
    )
    from DataRowleyV1V2.utils.rf import calc_sta

    dots_path = session_root / "dpi_calibration" / "dots_binned_data.dset"
    eye_dir = session_root / "dpi_calibration" / f"{eye}_eye"
    dpi_csv = eye_dir / "calibrated_dpi.csv"
    params_path = eye_dir / "calibration_params.npz"
    required = [dots_path, dpi_csv, params_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "dots RF SNR cache missing and cannot recompute; missing "
            + ", ".join(missing)
        )

    print(f"  Recomputing Rowley dots RF SNR from {dots_path}")
    dots_dset = DictDataset.load(dots_path)
    robs_all = _to_numpy(dots_dset["robs"])
    dots_cids = np.asarray(
        dots_dset.metadata.get("cids", np.arange(robs_all.shape[1]))
    )
    dots_index = {int(cid): idx for idx, cid in enumerate(dots_cids.tolist())}
    matched = np.array([dots_index.get(int(cid), -1) for cid in target_cids], dtype=int)
    found = matched >= 0
    if not found.any():
        raise ValueError("dots RF SNR recompute found no matching cluster IDs")

    params = np.load(params_path, allow_pickle=True)
    ppd = float(np.asarray(params["ppd"]).reshape(-1)[0])
    dpi_df = pd.read_csv(dpi_csv, usecols=["t_ephys", "i", "j", "valid"])
    sample_times = dpi_df["t_ephys"].to_numpy(dtype=np.float64)
    gaze_pix = dpi_df[["i", "j"]].to_numpy(dtype=np.float32)
    gaze_valid = dpi_df["valid"].to_numpy()
    valid_samples = (
        np.isfinite(sample_times)
        & _as_bool_1d(gaze_valid, len(sample_times))
        & np.all(np.isfinite(gaze_pix), axis=1)
    )
    if valid_samples.sum() < 2:
        raise ValueError("dots RF SNR recompute has too few calibrated gaze samples")

    t_bins = _to_numpy(dots_dset["t_bins"]).astype(np.float64)
    dots_pix = _to_numpy(dots_dset["dots_pix"]).astype(np.float32)
    robs = robs_all.astype(np.float32)

    gaze_interp = _interp_xy(sample_times[valid_samples], gaze_pix[valid_samples], t_bins)
    gaze_valid_interp = _nearest_resample_bool(sample_times, gaze_valid, t_bins)

    roi_pix = np.flipud(ROWLEY_DOTS_ROI_DEG * ppd)
    dxy_pix = ROWLEY_DOTS_DXY_DEG * ppd
    i_edges = np.arange(roi_pix[0, 0], roi_pix[0, 1] + dxy_pix, dxy_pix)
    j_edges = np.arange(roi_pix[1, 0], roi_pix[1, 1] + dxy_pix, dxy_pix)

    stim = bin_dots_to_stimulus(
        dots_pix, gaze_interp, i_edges, j_edges
    )[gaze_valid_interp]
    robs_valid = robs[gaze_valid_interp]
    if stim.shape[0] == 0:
        raise ValueError("dots RF SNR recompute has no valid dots frames")

    stas = calc_sta(
        stim[..., None],
        robs_valid,
        ROWLEY_DOTS_STA_LAGS,
        reverse_correlate=False,
        progress=False,
    ).squeeze().cpu().numpy()
    max_snr_all, _, _ = calculate_rf_snr(stas, ROWLEY_DOTS_DXY_DEG)

    max_snr_target = np.full(len(target_cids), np.nan, dtype=np.float32)
    max_snr_target[found] = max_snr_all[matched[found]].astype(np.float32, copy=False)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, cids=target_cids, max_snr=max_snr_target)
    except OSError as e:
        print(f"  Warning: could not save dots RF SNR cache {cache_path}: {e}")
    return max_snr_target


def _split_half_psth_r2(robs, n_splits, seed=42, min_valid_bins=10,
                        min_trials_per_half=2):
    """NaN-aware split-half PSTH R^2, matching the Rowley test scripts."""
    n_trials, _, n_units = robs.shape
    rng = np.random.default_rng(seed)
    r2_sum = np.zeros(n_units)
    r2_count = np.zeros(n_units, dtype=int)
    if n_trials < 2:
        return np.full(n_units, np.nan)

    for _ in range(n_splits):
        perm = rng.permutation(n_trials)
        half = n_trials // 2
        if half < min_trials_per_half:
            break
        idx_a = perm[:half]
        idx_b = perm[half:2 * half]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            psth_a = np.nanmean(robs[idx_a], axis=0)
            psth_b = np.nanmean(robs[idx_b], axis=0)
        cnt_a = np.sum(np.isfinite(robs[idx_a]), axis=0)
        cnt_b = np.sum(np.isfinite(robs[idx_b]), axis=0)
        for j in range(n_units):
            a, b = psth_a[:, j], psth_b[:, j]
            ok_t = (
                np.isfinite(a)
                & np.isfinite(b)
                & (cnt_a[:, j] >= min_trials_per_half)
                & (cnt_b[:, j] >= min_trials_per_half)
            )
            if ok_t.sum() < min_valid_bins:
                continue
            if np.std(a[ok_t]) <= 0 or np.std(b[ok_t]) <= 0:
                continue
            r = np.corrcoef(a[ok_t], b[ok_t])[0, 1]
            if np.isfinite(r):
                r2_sum[j] += r * r
                r2_count[j] += 1

    return np.divide(
        r2_sum,
        r2_count,
        out=np.full_like(r2_sum, np.nan, dtype=float),
        where=r2_count > 0,
    )


def _apply_rowley_inclusion(cfg, robs, eyepos, valid_mask, neuron_mask, meta):
    """Apply Rowley/Luke visual and missing-data gates before decomposition."""
    loaded_cids = np.asarray(cfg.get("cids", np.arange(meta["n_neurons_total"])))
    loaded_dots_ids = np.asarray(cfg.get("_rowley_dots_cids", loaded_cids))
    cids_used = loaded_cids[neuron_mask]
    dots_ids_used = loaded_dots_ids[neuron_mask]

    dots_snr = _load_rowley_dots_snr_for_cids(cfg, dots_ids_used)
    pool_source = str(cfg.get("_rowley_pool_source", ""))
    if pool_source in {"step07_visual_rf_valid", "step07_split_half_psth_r2"}:
        visual_ok = np.ones(len(dots_ids_used), dtype=bool)
    else:
        visual_ok = np.isfinite(dots_snr) & (dots_snr >= ROWLEY_DOTS_SNR_THRESH)

    eye_valid = np.isfinite(np.sum(eyepos, axis=2))
    valid_bins = eye_valid[:, :, None]
    n_valid_bins = max(float(eye_valid.sum()), 1.0)
    nan_frac = (np.isnan(robs) & valid_bins).sum(axis=(0, 1)) / n_valid_bins
    nan_ok = nan_frac <= ROWLEY_MAX_NAN_FRAC

    keep = visual_ok & nan_ok
    print(
        f"  [{cfg['session']}] Rowley visual gate ({pool_source or 'dots_snr'}): "
        f"{int(visual_ok.sum())}/{len(visual_ok)} pass"
    )
    print(
        f"  [{cfg['session']}] Rowley NaN fraction <= {ROWLEY_MAX_NAN_FRAC}: "
        f"{int(nan_ok.sum())}/{len(nan_ok)} pass"
    )
    print(
        f"  [{cfg['session']}] Rowley pre-decomp units: "
        f"{int(keep.sum())}/{len(keep)} pass both"
    )

    if keep.sum() < 2:
        return None, None, None, None, meta

    robs = robs[:, :, keep]
    neuron_mask = neuron_mask[keep]
    cids_used = cids_used[keep]
    dots_ids_used = dots_ids_used[keep]
    dots_snr = dots_snr[keep]
    nan_frac = nan_frac[keep]
    valid_mask = (
        np.isfinite(np.sum(robs, axis=2))
        & np.isfinite(np.sum(eyepos, axis=2))
    )

    meta = dict(meta)
    meta.update({
        "n_neurons_after_spike_gate": int(len(keep)),
        "n_neurons_after_rowley_visual_gate": int(visual_ok.sum()),
        "n_neurons_after_rowley_nan_gate": int(nan_ok.sum()),
        "n_neurons_used": int(robs.shape[2]),
        "rowley_cids_used": cids_used.tolist(),
        "rowley_dots_cids_used": dots_ids_used.tolist(),
        "rowley_dots_snr_thresh": ROWLEY_DOTS_SNR_THRESH,
        "rowley_dots_snr": dots_snr.tolist(),
        "rowley_pool_source": pool_source,
        "rowley_max_nan_frac": ROWLEY_MAX_NAN_FRAC,
        "rowley_nan_frac": nan_frac.tolist(),
    })
    return robs, eyepos, valid_mask, neuron_mask, meta


# ---------------------------------------------------------------------------
# Stage 1: per-session decomposition (cached as DECOMP_CACHE)
# ---------------------------------------------------------------------------

def _load_contam_rate(session_name, subject, n_neurons_total):
    """Per-neuron min contamination rate from QC data (or None)."""
    if subject in ("Allen", "Logan"):
        from DataYatesV1.utils.io import YatesV1Session
        try:
            sess = YatesV1Session(session_name)
            refractory = np.load(
                sess.sess_dir / 'qc' / 'refractory' / 'refractory.npz'
            )
            min_contam_props = refractory['min_contam_props']
            return np.array([
                np.min(min_contam_props[i]) for i in range(len(min_contam_props))
            ])
        except Exception as e:
            print(f"  Warning: Could not load QC data: {e}")
            return None
    raise NotImplementedError(
        f"QC loading not implemented for subject {subject}"
    )


def _compute_one_session(cfg):
    """Per-session work: returns the result dict, or None to skip.

    Module-level so it's picklable for multiprocessing. Imports its own
    heavy modules so spawned workers initialize cleanly.
    """
    if str(VISIONCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(VISIONCORE_ROOT))
    from models.data import prepare_data

    session_name = cfg["session"]
    subject = session_name.split("_")[0]
    if subject not in SUBJECTS:
        return None

    cfg = dict(cfg)
    is_rowley = _is_rowley_config(cfg)
    if is_rowley:
        # Rowley session YAML `cids` is still tied to the older Gaborium visual
        # criterion. Load the sorter contamination-passing V1 pool when present,
        # then apply the dots RF visual gate after trial alignment.
        rowley_cids, rowley_pool_source = _rowley_initial_cid_pool(cfg)
        if rowley_cids.size == 0:
            print(
                f"  [{session_name}] Skipping: Rowley config has no "
                "sortercontam/qccontam/cids"
            )
            return None
        print(
            f"  [{session_name}] Rowley initial unit pool: "
            f"{rowley_pool_source} ({rowley_cids.size} CIDs)"
        )
        cfg["_rowley_pool_source"] = rowley_pool_source
        cfg["directory"], cfg["eye"] = _resolve_rowley_dataset_directory(cfg)
        mapped = _prepare_rowley_unit_mapping(cfg, rowley_cids)
        if mapped is None:
            return None
        rowley_cids, rowley_dots_cids = mapped
        cfg["cids"] = rowley_cids.tolist()
        cfg["_rowley_dots_cids"] = rowley_dots_cids.tolist()

    if "fixrsvp" not in cfg["types"]:
        cfg["types"] = cfg["types"] + ["fixrsvp"]

    print(f"\n--- {session_name} ({subject}) ---")
    try:
        train_data, val_data, cfg = prepare_data(cfg, strict=False)
    except Exception as e:
        print(f"  [{session_name}] Skipping: {e}")
        return None

    try:
        dset_idx = train_data.get_dataset_index("fixrsvp")
    except (ValueError, KeyError):
        print(f"  [{session_name}] Skipping: no fixrsvp data")
        return None
    fixrsvp_dset = train_data.dsets[dset_idx]

    align_kwargs = dict(
        valid_time_bins=120,
        min_fix_dur=20,
        min_total_spikes=0,
    )
    if is_rowley:
        align_kwargs.update(
            fixation_radius=1.5,
            fixation_center="median_valid",
            require_dpi_valid=True,
        )
    robs, eyepos, valid_mask, neuron_mask, meta = align_fixrsvp_trials(
        fixrsvp_dset,
        **align_kwargs,
    )
    if robs is None or robs.shape[0] < 10:
        print(f"  [{session_name}] Skipping: insufficient data ({meta})")
        return None
    print(f"  [{session_name}] Trials: {meta['n_trials_good']}/{meta['n_trials_total']}, "
          f"Neurons: {meta['n_neurons_used']}/{meta['n_neurons_total']}")

    if is_rowley:
        try:
            robs, eyepos, valid_mask, neuron_mask, meta = _apply_rowley_inclusion(
                cfg, robs, eyepos, valid_mask, neuron_mask, meta
            )
        except Exception as e:
            print(f"  [{session_name}] Skipping Rowley session: {e}")
            return None
        if robs is None or robs.shape[2] < 2:
            print(f"  [{session_name}] Skipping: too few Rowley units after inclusion")
            return None

    n_units = robs.shape[2]
    n_spikes_per_unit = np.nansum(robs, axis=(0, 1))
    n_valid_bins_per_unit = np.sum(np.isfinite(robs), axis=(0, 1))
    rate_hz = np.where(
        n_valid_bins_per_unit > 0,
        n_spikes_per_unit / np.maximum(n_valid_bins_per_unit, 1) / DT,
        np.nan,
    )

    psth_r2 = _split_half_psth_r2(robs, N_PSTH_SPLITS, seed=42)
    n_rel = int(np.sum(np.isfinite(psth_r2) & (psth_r2 > MIN_PSTH_R2)))
    print(
        f"  [{session_name}] FixRSVP PSTH R^2 > {MIN_PSTH_R2}: "
        f"{n_rel}/{n_units} pass plotted-unit gate"
    )

    # If a worker pool pinned us to a single GPU via CUDA_VISIBLE_DEVICES,
    # use cuda:0 directly. get_free_device queries nvidia-smi and may
    # return a physical index that doesn't exist in this worker's CUDA view.
    import os, torch
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if torch.cuda.is_available() and cvd and "," not in cvd:
        device = torch.device("cuda:0")
        print(f"  [{session_name}] using pinned device cuda:0 "
              f"(CUDA_VISIBLE_DEVICES={cvd})")
    else:
        device = get_free_device()
    results, mats = run_covariance_decomposition(
        robs, eyepos, valid_mask,
        window_sizes_bins=WINDOW_BINS,
        dt=DT,
        n_shuffles=N_SHUFFLES,
        intercept_mode=INTERCEPT_MODE,
        intercept_kwargs=INTERCEPT_KWARGS,
        seed=42,
        device=str(device),
    )

    psth = robs.mean(axis=0)

    try:
        contam_rate = _load_contam_rate(
            session_name, subject, meta['n_neurons_total']
        )
    except NotImplementedError:
        contam_rate = None
        print(f"  [{session_name}] QC: contamination not available for {subject}")

    return {
        "session": session_name,
        "subject": subject,
        "results": results,
        "mats": mats,
        "neuron_mask": neuron_mask,
        "meta": meta,
        "psth": psth,
        "rate_hz": rate_hz,
        "psth_r2": psth_r2,
        "cids": np.asarray(cfg.get("cids", []))[neuron_mask].tolist(),
        "qc": {"contam_rate": contam_rate},
    }


def _worker_init(gpu_queue):
    """Pool initializer: claim a GPU and pin this worker to it.

    Sets CUDA_VISIBLE_DEVICES BEFORE any CUDA op runs. Subsequent
    `torch.cuda` queries in this process see exactly one GPU (renumbered 0).
    """
    import os
    gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[worker pid={os.getpid()}] pinned to physical GPU {gpu_id}", flush=True)


def _run_session_pool(cfgs, n_workers, n_gpus=1):
    """Run _compute_one_session over cfgs, returning results in input order.

    n_workers=1 → in-process sequential (bit-identical to legacy path).
    n_workers>1 → spawn-Pool with workers round-robin pinned to n_gpus GPUs.
    """
    if n_workers <= 1:
        return [_compute_one_session(cfg) for cfg in cfgs]

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    n_gpus = max(int(n_gpus), 1)
    gpu_queue = ctx.Queue()
    for i in range(n_workers):
        gpu_queue.put(i % n_gpus)
    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(gpu_queue,),
    ) as pool:
        return pool.map(_compute_one_session, cfgs)


def _compute_session_results(n_workers=None, n_gpus=None):
    """Run LOTC decomposition for every session listed in the dataset config."""
    if n_workers is None:
        n_workers = N_STAGE1_WORKERS
    if n_gpus is None:
        n_gpus = N_STAGE1_GPUS

    if str(VISIONCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(VISIONCORE_ROOT))
    from models.config_loader import load_dataset_configs

    dataset_configs = load_dataset_configs(str(DATASET_CONFIGS_PATH))
    per_session = _run_session_pool(dataset_configs, n_workers, n_gpus)
    session_results = [r for r in per_session if r is not None]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(DECOMP_CACHE, "wb") as f:
        dill.dump(session_results, f)
    print(f"\nCached {len(session_results)} sessions to {DECOMP_CACHE}")
    return session_results


def _load_or_compute_session_results(refresh=False):
    if DECOMP_CACHE.exists() and not refresh:
        print(f"Loading cached decomposition from {DECOMP_CACHE}")
        with open(DECOMP_CACHE, "rb") as f:
            return dill.load(f)
    return _compute_session_results()


# ---------------------------------------------------------------------------
# Stage 2: derive per-window metrics
# ---------------------------------------------------------------------------

def _metrics_one(sr, w_idx):
    """Per-(window, session) contribution to the metrics dict for w_idx.
    Returns None if the session has too few valid neurons in this window."""
    if w_idx >= len(sr["results"]):
        return None
    res = sr["results"][w_idx]
    mats = sr["mats"][w_idx]

    Ctotal = mats["Total"]
    Cpsth = mats["PSTH"]
    Crate = mats["Intercept"]
    Cfem = mats["FEM"]

    CnoiseU = 0.5 * ((Ctotal - Cpsth) + (Ctotal - Cpsth).T)
    CnoiseC = 0.5 * ((Ctotal - Crate) + (Ctotal - Crate).T)

    erate = res["Erates"]
    rate_hz_ds = sr["rate_hz"]
    psth_r2_ds = sr["psth_r2"]
    valid = (
        np.isfinite(erate)
        & np.isfinite(rate_hz_ds)
        & (rate_hz_ds > MIN_RATE_HZ)
        & np.isfinite(psth_r2_ds)
        & (psth_r2_ds > MIN_PSTH_R2)
        & (np.diag(Ctotal) > MIN_VAR)
        & np.isfinite(np.diag(Crate))
        & np.isfinite(np.diag(Cpsth))
    )
    if valid.sum() < 2:
        return None

    diag_psth = np.diag(Cpsth)[valid]
    diag_rate = np.diag(Crate)[valid]
    alpha = diag_psth / diag_rate

    ff_u = np.diag(CnoiseU)[valid] / erate[valid]
    ff_c = np.diag(CnoiseC)[valid] / erate[valid]

    NoiseCorrU = cov_to_corr(
        project_to_psd(CnoiseU[np.ix_(valid, valid)]), min_var=MIN_VAR
    )
    NoiseCorrC = cov_to_corr(
        project_to_psd(CnoiseC[np.ix_(valid, valid)]), min_var=MIN_VAR
    )
    rho_u_full = get_upper_triangle(NoiseCorrU)
    rho_c_full = get_upper_triangle(NoiseCorrC)
    pair_ok = np.isfinite(rho_u_full) & np.isfinite(rho_c_full)
    rho_u = rho_u_full[pair_ok]
    rho_c = rho_c_full[pair_ok]

    if len(rho_u) > 0:
        rho_u_meanz = fisher_z_mean(rho_u, eps=EPS_RHO)
        rho_c_meanz = fisher_z_mean(rho_c, eps=EPS_RHO)
        rho_delta_meanz = rho_c_meanz - rho_u_meanz
    else:
        rho_u_meanz = rho_c_meanz = rho_delta_meanz = np.nan

    n_valid_ds = int(valid.sum())
    shuff_alphas = []
    ds_shuff_var_c = []
    shuff_rho_c_meanz_list, shuff_rho_delta_meanz_list = [], []
    shuff_rho_subject_list = []
    if "Shuffled_Intercepts" in mats and len(mats["Shuffled_Intercepts"]) > 0:
        for Crate_shuf in mats["Shuffled_Intercepts"]:
            diag_rate_shuf = np.diag(Crate_shuf)[valid]
            alpha_shuf = diag_psth / diag_rate_shuf
            shuff_alphas.append(1 - alpha_shuf)

            CnoiseC_shuf = Ctotal - Crate_shuf
            CnoiseC_shuf = 0.5 * (CnoiseC_shuf + CnoiseC_shuf.T)
            ds_shuff_var_c.append(np.diag(CnoiseC_shuf)[valid])
            NC_shuf = cov_to_corr(
                project_to_psd(CnoiseC_shuf[np.ix_(valid, valid)]),
                min_var=MIN_VAR,
            )
            rho_c_shuf = get_upper_triangle(NC_shuf)
            ok = np.isfinite(rho_c_shuf) & pair_ok
            if ok.sum() > 0:
                shuff_rho_c_meanz_list.append(
                    fisher_z_mean(rho_c_shuf[ok], eps=EPS_RHO)
                )
                shuff_rho_delta_meanz_list.append(
                    fisher_z_mean(rho_c_shuf[ok], eps=EPS_RHO)
                    - fisher_z_mean(rho_u_full[ok[:len(rho_u_full)]], eps=EPS_RHO)
                )
                shuff_rho_subject_list.append(sr["subject"])

    return dict(
        subject=sr["subject"],
        session=sr["session"],
        n_valid=n_valid_ds,
        alpha=alpha,
        ff_uncorr=ff_u,
        ff_corr=ff_c,
        erate=erate[valid],
        rho_uncorr=rho_u,
        rho_corr=rho_c,
        rho_u_meanz=rho_u_meanz,
        rho_c_meanz=rho_c_meanz,
        rho_delta_meanz=rho_delta_meanz,
        Ctotal=Ctotal[np.ix_(valid, valid)],
        Cpsth=Cpsth[np.ix_(valid, valid)],
        Crate=Crate[np.ix_(valid, valid)],
        CnoiseU=CnoiseU[np.ix_(valid, valid)],
        CnoiseC=CnoiseC[np.ix_(valid, valid)],
        Cfem=Cfem[np.ix_(valid, valid)],
        shuff_alphas=shuff_alphas,
        ds_shuff_var_c=np.asarray(ds_shuff_var_c) if ds_shuff_var_c else None,
        shuff_rho_c_meanz=shuff_rho_c_meanz_list,
        shuff_rho_delta_meanz=shuff_rho_delta_meanz_list,
        shuff_rho_subject=shuff_rho_subject_list,
    )


def _compute_metrics(session_results, windows_ms, windows_bins, n_jobs=-1):
    """Parallelized across (window, session) pairs.

    The original loop body was pure numpy and per-session-independent — the
    only cross-session step is final concatenation, which we do after the
    parallel fan-out below. joblib preserves input order, so session order is
    preserved exactly.
    """
    from joblib import Parallel, delayed

    n_windows = len(windows_ms)
    tasks = [(w_idx, sr_i) for w_idx in range(n_windows)
             for sr_i in range(len(session_results))]
    flat = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_metrics_one)(session_results[sr_i], w_idx)
        for (w_idx, sr_i) in tasks
    )

    by_w = [[] for _ in range(n_windows)]
    for (w_idx, _), r in zip(tasks, flat):
        by_w[w_idx].append(r)

    metrics = []
    for w_idx in range(n_windows):
        all_alpha, all_ff_uncorr, all_ff_corr, all_erate = [], [], [], []
        all_rho_uncorr, all_rho_corr = [], []
        rho_u_meanz_by_ds, rho_c_meanz_by_ds, rho_delta_meanz_by_ds = [], [], []
        all_Ctotal, all_Cpsth, all_Crate, all_CnoiseU, all_CnoiseC, all_Cfem = (
            [], [], [], [], [], []
        )
        shuff_alphas = []
        shuff_rho_delta_meanz, shuff_rho_c_meanz, shuff_rho_subject = [], [], []
        subject_by_ds, subject_per_neuron, subject_per_pair = [], [], []
        session_per_neuron = []
        shuff_var_c_blocks, shuff_var_c_nvalid = [], []

        for r in by_w[w_idx]:
            if r is None:
                continue
            subject_by_ds.append(r["subject"])
            all_alpha.append(r["alpha"])
            subject_per_neuron.extend([r["subject"]] * r["n_valid"])
            session_per_neuron.extend([r["session"]] * r["n_valid"])
            all_ff_uncorr.append(r["ff_uncorr"])
            all_ff_corr.append(r["ff_corr"])
            all_erate.append(r["erate"])
            all_rho_uncorr.append(r["rho_uncorr"])
            all_rho_corr.append(r["rho_corr"])
            subject_per_pair.extend([r["subject"]] * len(r["rho_uncorr"]))

            if len(r["rho_uncorr"]) > 0:
                rho_u_meanz_by_ds.append(r["rho_u_meanz"])
                rho_c_meanz_by_ds.append(r["rho_c_meanz"])
                rho_delta_meanz_by_ds.append(r["rho_delta_meanz"])

            all_Ctotal.append(r["Ctotal"])
            all_Cpsth.append(r["Cpsth"])
            all_Crate.append(r["Crate"])
            all_CnoiseU.append(r["CnoiseU"])
            all_CnoiseC.append(r["CnoiseC"])
            all_Cfem.append(r["Cfem"])

            shuff_alphas.extend(r["shuff_alphas"])
            shuff_rho_c_meanz.extend(r["shuff_rho_c_meanz"])
            shuff_rho_delta_meanz.extend(r["shuff_rho_delta_meanz"])
            shuff_rho_subject.extend(r["shuff_rho_subject"])

            shuff_var_c_blocks.append(r["ds_shuff_var_c"])
            shuff_var_c_nvalid.append(r["n_valid"])

        # Pool shuffle-null corrected variances across sessions into [N_total, S_min].
        # Sessions are truncated to the smallest shuffle count so each column b is one
        # coherent null draw pooled over all neurons; the row order matches `erate`.
        # Sessions without shuffles contribute NaN rows (dropped per-column later).
        present_S = [b.shape[0] for b in shuff_var_c_blocks if b is not None]
        if present_S:
            S_min = min(present_S)
            rows = []
            for blk, nv in zip(shuff_var_c_blocks, shuff_var_c_nvalid):
                if blk is None:
                    rows.append(np.full((nv, S_min), np.nan))
                else:
                    rows.append(blk[:S_min].T)  # [n_valid, S_min]
            shuff_var_c = np.concatenate(rows, axis=0)
        else:
            shuff_var_c = np.empty((0, 0))

        metrics.append({
            "window_ms": windows_ms[w_idx],
            "window_bins": windows_bins[w_idx],
            "alpha": np.concatenate(all_alpha) if all_alpha else np.array([]),
            "uncorr": np.concatenate(all_ff_uncorr) if all_ff_uncorr else np.array([]),
            "corr": np.concatenate(all_ff_corr) if all_ff_corr else np.array([]),
            "erate": np.concatenate(all_erate) if all_erate else np.array([]),
            "rho_uncorr": (
                np.concatenate(all_rho_uncorr) if all_rho_uncorr else np.array([])
            ),
            "rho_corr": (
                np.concatenate(all_rho_corr) if all_rho_corr else np.array([])
            ),
            "rho_u_meanz_by_ds": np.array(rho_u_meanz_by_ds),
            "rho_c_meanz_by_ds": np.array(rho_c_meanz_by_ds),
            "rho_delta_meanz_by_ds": np.array(rho_delta_meanz_by_ds),
            "subject_by_ds": subject_by_ds,
            "subject_per_neuron": np.array(subject_per_neuron),
            "session_per_neuron": np.array(session_per_neuron),
            "subject_per_pair": np.array(subject_per_pair),
            "shuff_var_c": shuff_var_c,
            "Ctotal": all_Ctotal,
            "Cpsth": all_Cpsth,
            "Crate": all_Crate,
            "CnoiseU": all_CnoiseU,
            "CnoiseC": all_CnoiseC,
            "Cfem": all_Cfem,
            "shuff_alphas": shuff_alphas,
            "shuff_rho_delta_meanz": np.array(shuff_rho_delta_meanz),
            "shuff_rho_c_meanz": np.array(shuff_rho_c_meanz),
            "shuff_rho_subject": np.array(shuff_rho_subject),
        })

        m = metrics[-1]
        print(f"Window {windows_ms[w_idx]:.1f} ms ({windows_bins[w_idx]} bins): "
              f"{len(m['alpha'])} neurons, "
              f"{len(m['rho_uncorr'])} pairs, "
              f"{len(m['shuff_alphas'])} shuffle iterations")
    return metrics


# ---------------------------------------------------------------------------
# Stage 3: per-panel summaries
# ---------------------------------------------------------------------------

def _compute_alpha_stats(metrics, windows_ms):
    m_by_window = []
    subject_per_neuron_by_window = []
    alpha_stats = {}
    for w_idx, m_dict in enumerate(metrics):
        alpha = m_dict["alpha"]
        m_raw = 1 - alpha
        subj_raw = m_dict["subject_per_neuron"]

        in_range = np.isfinite(m_raw) & (m_raw >= 0.0) & (m_raw <= 1.0)
        n_total = int(np.isfinite(m_raw).sum())
        n_dropped = int(n_total - in_range.sum())
        m = m_raw[in_range]
        m_by_window.append(m)
        subject_per_neuron_by_window.append(subj_raw[in_range])

        mean_m, (ci_lo, ci_hi) = bootstrap_mean_ci(m, nboot=5000, seed=0)
        med_m = float(np.nanmedian(m))
        q25, q75 = iqr_25_75(m)

        shuff_m = [
            s[np.isfinite(s) & (s >= 0.0) & (s <= 1.0)]
            for s in m_dict["shuff_alphas"]
        ]
        shuff_m = [s for s in shuff_m if s.size > 0]
        if len(shuff_m) > 0:
            null_means = np.array([np.nanmean(s) for s in shuff_m])
            null_mean_ci = (
                float(np.percentile(null_means, 2.5)),
                float(np.percentile(null_means, 97.5)),
            )
            p_emp = emp_p_one_sided(null_means, mean_m, direction="less")
        else:
            null_mean_ci = (np.nan, np.nan)
            p_emp = np.nan

        alpha_stats[windows_ms[w_idx]] = {
            "n": len(m), "mean": mean_m, "ci": (ci_lo, ci_hi),
            "median": med_m, "iqr": (q25, q75),
            "null_ci": null_mean_ci, "p_emp": p_emp,
            "n_dropped": n_dropped, "n_total": n_total,
        }
    return m_by_window, subject_per_neuron_by_window, alpha_stats


def _slope_through_origin(erate, var):
    """LS slope of var = slope * erate forced through the origin (nan-safe)."""
    ok = np.isfinite(erate) & np.isfinite(var) & (erate > 0) & (var >= 0)
    if ok.sum() < 3:
        return np.nan
    e, v = erate[ok], var[ok]
    return float(np.sum(e * v) / np.sum(e ** 2))


def _clustered_slope_bootstrap(erate, var_u, var_c, sessions, nboot=5000, seed=0):
    """
    Session-clustered bootstrap of slope-through-origin Fano factors.

    Neurons within a session share the same trials and eye-movement traces, so
    they are not independent samples. A naive neuron-level bootstrap therefore
    underestimates the true uncertainty. We instead resample whole sessions with
    replacement, pool every neuron from the drawn sessions, and refit the pooled
    slopes each iteration -- propagating session-level uncertainty.

    Fallback: with a single session we cannot estimate between-session variance,
    so we resample neurons within that session (the only uncertainty available)
    rather than return a degenerate, zero-width interval.

    Returns dict with 95% CIs for the uncorrected slope, corrected slope, and
    their difference (uncorrected - corrected), plus a one-sided bootstrap p for
    H0: corrected slope >= uncorrected slope (i.e. no FEM-driven inflation).

    Vectorized: per-session sufficient statistics (sum e*v and sum e^2) are
    precomputed, then bootstrap draws are gathered+summed in a single matrix
    op. Equivalent to the per-iteration Python loop but ~100x faster.
    """
    erate = np.asarray(erate, dtype=float)
    var_u = np.asarray(var_u, dtype=float)
    var_c = np.asarray(var_c, dtype=float)
    sessions = np.asarray(sessions)
    uniq = np.unique(sessions)
    rng = np.random.default_rng(seed)
    n = len(erate)

    if uniq.size >= 2:
        # Sufficient statistics per session: slope = sum(e*v) / sum(e^2). The
        # cluster bootstrap concatenates whole sessions, so a session draw is
        # just a sum over the per-session sufficient statistics.
        K = uniq.size
        sum_ee = np.zeros(K)
        sum_eu = np.zeros(K)
        sum_ec = np.zeros(K)
        for i, s in enumerate(uniq):
            m = sessions == s
            e = erate[m]; vu = var_u[m]; vc = var_c[m]
            ok = np.isfinite(e) & np.isfinite(vu) & np.isfinite(vc)
            e, vu, vc = e[ok], vu[ok], vc[ok]
            sum_ee[i] = np.sum(e * e)
            sum_eu[i] = np.sum(e * vu)
            sum_ec[i] = np.sum(e * vc)

        # draws: [nboot, K] session indices, with replacement
        draws = rng.integers(0, K, size=(nboot, K))
        D_ee = sum_ee[draws].sum(axis=1)
        D_eu = sum_eu[draws].sum(axis=1)
        D_ec = sum_ec[draws].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            su = np.where(D_ee > 0, D_eu / D_ee, np.nan)
            sc = np.where(D_ee > 0, D_ec / D_ee, np.nan)
    else:
        # Neuron-level bootstrap (single-session fallback). Same vectorized
        # gather pattern.
        ok = np.isfinite(erate) & np.isfinite(var_u) & np.isfinite(var_c)
        e_v = erate[ok]; vu_v = var_u[ok]; vc_v = var_c[ok]
        if e_v.size == 0:
            su = np.full(nboot, np.nan)
            sc = np.full(nboot, np.nan)
        else:
            idx = rng.integers(0, e_v.size, size=(nboot, e_v.size))
            E = e_v[idx]
            D_ee = np.sum(E * E, axis=1)
            D_eu = np.sum(E * vu_v[idx], axis=1)
            D_ec = np.sum(E * vc_v[idx], axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                su = np.where(D_ee > 0, D_eu / D_ee, np.nan)
                sc = np.where(D_ee > 0, D_ec / D_ee, np.nan)

    diff = su - sc

    def _ci(a):
        a = a[np.isfinite(a)]
        if a.size == 0:
            return (np.nan, np.nan)
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

    diff_f = diff[np.isfinite(diff)]
    p = (float((np.sum(diff_f <= 0) + 1) / (diff_f.size + 1))
         if diff_f.size else np.nan)
    return {
        "unc_ci": _ci(su), "cor_ci": _ci(sc), "diff_ci": _ci(diff),
        "p": p, "n_sessions": int(uniq.size),
    }


def _compute_fano_stats(metrics, windows_ms):
    fano_stats = {}
    for w_idx, m_dict in enumerate(metrics):
        ff_u, ff_c, erate = m_dict["uncorr"], m_dict["corr"], m_dict["erate"]
        ff_u_v, ff_c_v, mask = paired_valid(ff_u, ff_c, positive=True)
        erate_v = erate[mask]
        subject_labels_v = m_dict["subject_per_neuron"][mask]
        session_labels_v = m_dict["session_per_neuron"][mask]
        n_valid = len(ff_u_v)

        g_unc = geomean(ff_u_v)
        g_cor = geomean(ff_c_v)
        ratio = g_cor / g_unc
        pct_red = (1 - ratio) * 100

        _, p_wil = wilcoxon_signed_rank(ff_c_v, ff_u_v, alternative="less")

        var_u = ff_u_v * erate_v
        var_c = ff_c_v * erate_v

        # Pooled slope-through-origin Fano factors with session-clustered CIs.
        slope_unc = _slope_through_origin(erate_v, var_u)
        slope_cor = _slope_through_origin(erate_v, var_c)
        boot = _clustered_slope_bootstrap(
            erate_v, var_u, var_c, session_labels_v, nboot=5000, seed=0
        )
        slope_unc_ci = boot["unc_ci"]
        slope_cor_ci = boot["cor_ci"]
        slope_diff = slope_unc - slope_cor
        slope_diff_ci = boot["diff_ci"]
        p_slope = boot["p"]

        # Per-subject slopes + clustered CIs (drive the per-subject fitted lines
        # and shaded bands in panel D).
        per_subject = {}
        for subj in np.unique(subject_labels_v):
            s_mask = subject_labels_v == subj
            e_s, vu_s, vc_s = erate_v[s_mask], var_u[s_mask], var_c[s_mask]
            sess_s = session_labels_v[s_mask]
            boot_s = _clustered_slope_bootstrap(
                e_s, vu_s, vc_s, sess_s, nboot=2000, seed=0
            )
            su_s = _slope_through_origin(e_s, vu_s)
            sc_s = _slope_through_origin(e_s, vc_s)
            per_subject[str(subj)] = {
                "slope_unc": su_s, "slope_cor": sc_s,
                "slope_unc_ci": boot_s["unc_ci"], "slope_cor_ci": boot_s["cor_ci"],
                "slope_diff": su_s - sc_s, "slope_diff_ci": boot_s["diff_ci"],
                "p_slope": boot_s["p"], "n_sessions": boot_s["n_sessions"],
                "n": int(s_mask.sum()),
            }

        # FEM-shuffle null on the corrected slope. `shuff_var_c` holds, per neuron,
        # the corrected noise variance recomputed with eye positions shuffled
        # ([N_neurons, S]); column b is one null draw pooled across sessions. For
        # each draw we refit the pooled corrected slope and measure how much
        # variance the correction strips off by chance. The real reduction is
        # significant if it exceeds this null reduction.
        shuff_var_c = m_dict.get("shuff_var_c", np.empty((0, 0)))
        if shuff_var_c.size and shuff_var_c.shape[0] == mask.shape[0]:
            svc = shuff_var_c[mask]  # [n_valid, S]
            null_slope_cor = np.array([
                _slope_through_origin(erate_v, svc[:, b])
                for b in range(svc.shape[1])
            ])
            null_slope_cor = null_slope_cor[np.isfinite(null_slope_cor)]
        else:
            null_slope_cor = np.array([])

        if null_slope_cor.size:
            null_reduction = slope_unc - null_slope_cor
            obs_reduction = slope_unc - slope_cor
            slope_cor_null_ci = (
                float(np.percentile(null_slope_cor, 2.5)),
                float(np.percentile(null_slope_cor, 97.5)),
            )
            p_emp_slope = emp_p_one_sided(
                null_reduction, obs_reduction, direction="greater"
            )
        else:
            slope_cor_null_ci = (np.nan, np.nan)
            p_emp_slope = np.nan

        fano_stats[windows_ms[w_idx]] = {
            "n": n_valid, "g_unc": g_unc, "g_cor": g_cor,
            "ratio": ratio, "pct_red": pct_red, "p_wil": p_wil,
            "slope_unc": slope_unc, "slope_cor": slope_cor,
            "slope_unc_ci": slope_unc_ci, "slope_cor_ci": slope_cor_ci,
            "slope_diff": slope_diff, "slope_diff_ci": slope_diff_ci,
            "p_slope": p_slope, "n_sessions": boot["n_sessions"],
            "slope_cor_null_ci": slope_cor_null_ci, "p_emp_slope": p_emp_slope,
            "null_ratio_ci": (np.nan, np.nan),
            "per_subject": per_subject,
            "erate": erate_v, "var_u": var_u, "var_c": var_c,
            "subject_per_neuron": subject_labels_v,
            "session_per_neuron": session_labels_v,
        }
    return fano_stats


def _compute_nc_stats(metrics, windows_ms):
    nc_stats = {}
    for w_idx, m_dict in enumerate(metrics):
        rho_u = m_dict["rho_uncorr"]
        rho_c = m_dict["rho_corr"]
        n_pairs = len(rho_u)

        z_u_ds = m_dict["rho_u_meanz_by_ds"]
        z_c_ds = m_dict["rho_c_meanz_by_ds"]
        dz_ds = m_dict["rho_delta_meanz_by_ds"]
        n_ds = len(z_u_ds)

        z_u_mean, z_u_ci = bootstrap_mean_ci(z_u_ds, nboot=5000, seed=0)
        z_c_mean, z_c_ci = bootstrap_mean_ci(z_c_ds, nboot=5000, seed=0)
        dz_mean, dz_ci = bootstrap_mean_ci(dz_ds, nboot=5000, seed=0)

        if n_ds >= 5:
            _, p_wil = wilcoxon_signed_rank(z_c_ds, z_u_ds, alternative="less")
        else:
            p_wil = np.nan

        shuff_dz = m_dict["shuff_rho_delta_meanz"]
        shuff_subj = m_dict["shuff_rho_subject"]
        if len(shuff_dz) > 0:
            null_dz_ci = (
                float(np.percentile(shuff_dz, 2.5)),
                float(np.percentile(shuff_dz, 97.5)),
            )
            p_emp_dz = emp_p_one_sided(shuff_dz, dz_mean, direction="less")
        else:
            null_dz_ci = (np.nan, np.nan)
            p_emp_dz = np.nan

        null_dz_ci_by_subject = {}
        for subj in SUBJECTS:
            s_mask = shuff_subj == subj
            if s_mask.sum() > 0:
                null_dz_ci_by_subject[subj] = (
                    float(np.percentile(shuff_dz[s_mask], 2.5)),
                    float(np.percentile(shuff_dz[s_mask], 97.5)),
                )
            else:
                null_dz_ci_by_subject[subj] = (np.nan, np.nan)

        nc_stats[windows_ms[w_idx]] = {
            "n_pairs": n_pairs, "n_ds": n_ds,
            "z_u_mean": z_u_mean, "z_u_ci": z_u_ci,
            "z_c_mean": z_c_mean, "z_c_ci": z_c_ci,
            "dz_mean": dz_mean, "dz_ci": dz_ci,
            "p_wil": p_wil, "null_dz_ci": null_dz_ci, "p_emp_dz": p_emp_dz,
            "null_dz_ci_by_subject": null_dz_ci_by_subject,
            "rho_u": rho_u, "rho_c": rho_c,
        }
    return nc_stats


def _subspace_one_session(sr, session_name, subject):
    """Pure per-session subspace computation. Returns None if the session
    has too few valid neurons to participate in panel G."""
    w_idx = SUBSPACE_WINDOW_IDX
    if w_idx >= len(sr["mats"]):
        return None
    mats = sr["mats"][w_idx]
    Cpsth = mats["PSTH"]
    Crate = mats["Intercept"]
    Ctotal = mats["Total"]
    Cfem = Crate - Cpsth

    erate = sr["results"][w_idx]["Erates"]
    rate_hz_ds = sr["rate_hz"]
    psth_r2_ds = sr["psth_r2"]
    valid = (
        np.isfinite(erate)
        & np.isfinite(rate_hz_ds)
        & (rate_hz_ds > MIN_RATE_HZ)
        & np.isfinite(psth_r2_ds)
        & (psth_r2_ds > MIN_PSTH_R2)
        & (np.diag(Ctotal) > MIN_VAR)
        & np.isfinite(np.diag(Crate))
        & np.isfinite(np.diag(Cpsth))
    )
    if valid.sum() < SUBSPACE_K + 1:
        return None

    Cpsth_v = Cpsth[np.ix_(valid, valid)]
    Cfem_v = Cfem[np.ix_(valid, valid)]
    Ctotal_v = Ctotal[np.ix_(valid, valid)]

    Cpsth_psd = project_to_psd(Cpsth_v)
    Cfem_psd = project_to_psd(Cfem_v)

    w_psth, V_psth = np.linalg.eigh(Cpsth_psd)
    w_fem, V_fem = np.linalg.eigh(Cfem_psd)
    w_psth, V_psth = w_psth[::-1], V_psth[:, ::-1]
    w_fem, V_fem = w_fem[::-1], V_fem[:, ::-1]

    k = min(SUBSPACE_K, int(valid.sum()) - 1)
    U_psth = V_psth[:, :k]
    U_fem = V_fem[:, :k]
    tr_total = np.trace(Ctotal_v)

    # Shuffle null on Cfem only: same Cpsth (and its subspace) as real, but
    # FEM rebuilt from eye-shuffled Intercepts.
    null_x, null_y, null_ok, null_ok1 = [], [], [], []
    shuff_intercepts = mats.get("Shuffled_Intercepts", []) or []
    for Crate_shuf in shuff_intercepts:
        Crate_shuf = np.asarray(Crate_shuf, dtype=np.float64)
        Cfem_shuf = Crate_shuf[np.ix_(valid, valid)] - Cpsth_v
        Cfem_shuf_psd = project_to_psd(Cfem_shuf)
        w_shuf, V_shuf = np.linalg.eigh(Cfem_shuf_psd)
        V_shuf = V_shuf[:, ::-1]
        U_fem_shuf = V_shuf[:, :k]
        null_x.append(directional_variance_capture(Cpsth_psd, U_fem_shuf))
        null_y.append(directional_variance_capture(Cfem_shuf_psd, U_psth))
        null_ok.append(symmetric_subspace_overlap(U_psth, U_fem_shuf))
        null_ok1.append(symmetric_subspace_overlap(V_psth[:, :1], V_shuf[:, :1]))

    return dict(
        session_name=session_name,
        subject=subject,
        pr_psth=participation_ratio(Cpsth_psd),
        pr_fem=participation_ratio(Cfem_psd),
        overlap_k=symmetric_subspace_overlap(U_psth, U_fem),
        overlap_k1=symmetric_subspace_overlap(V_psth[:, :1], V_fem[:, :1]),
        var_p_given_f=directional_variance_capture(Cpsth_psd, U_fem),
        var_f_given_p=directional_variance_capture(Cfem_psd, U_psth),
        spectrum_psth=w_psth / tr_total,
        spectrum_fem=w_fem / tr_total,
        null_var_p_given_f=null_x,
        null_var_f_given_p=null_y,
        null_overlap_k=null_ok,
        null_overlap_k1=null_ok1,
    )


def _compute_subspace(session_results, session_names, subjects, n_jobs=-1):
    """Per-session work runs in parallel across CPU cores (joblib).

    Stage 2 only — pure numpy, no torch/CUDA, safe to fan out.
    """
    from joblib import Parallel, delayed

    per_session = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_subspace_one_session)(sr, session_names[i], subjects[i])
        for i, sr in enumerate(session_results)
    )

    sub_names, sub_subjects = [], []
    pr_fem_list, pr_psth_list = [], []
    overlap_k1_list, overlap_k_list = [], []
    var_p_given_f, var_f_given_p = [], []
    spectra_psth, spectra_fem = [], []
    null_var_p_given_f, null_var_f_given_p = [], []
    null_overlap_k, null_overlap_k1 = [], []
    null_session_idx, null_subjects = [], []

    for r in per_session:
        if r is None:
            continue
        sub_names.append(r["session_name"])
        sub_subjects.append(r["subject"])
        pr_psth_list.append(r["pr_psth"])
        pr_fem_list.append(r["pr_fem"])
        overlap_k_list.append(r["overlap_k"])
        overlap_k1_list.append(r["overlap_k1"])
        var_p_given_f.append(r["var_p_given_f"])
        var_f_given_p.append(r["var_f_given_p"])
        spectra_psth.append(r["spectrum_psth"])
        spectra_fem.append(r["spectrum_fem"])

        sess_pos = len(sub_names) - 1
        n_draws = len(r["null_var_p_given_f"])
        null_var_p_given_f.extend(r["null_var_p_given_f"])
        null_var_f_given_p.extend(r["null_var_f_given_p"])
        null_overlap_k.extend(r["null_overlap_k"])
        null_overlap_k1.extend(r["null_overlap_k1"])
        null_session_idx.extend([sess_pos] * n_draws)
        null_subjects.extend([r["subject"]] * n_draws)

    return dict(
        sub_names=sub_names,
        sub_subjects=sub_subjects,
        pr_fem_list=pr_fem_list,
        pr_psth_list=pr_psth_list,
        overlap_k1_list=overlap_k1_list,
        overlap_k_list=overlap_k_list,
        var_p_given_f=var_p_given_f,
        var_f_given_p=var_f_given_p,
        spectra_psth=spectra_psth,
        spectra_fem=spectra_fem,
        null_var_p_given_f=null_var_p_given_f,
        null_var_f_given_p=null_var_f_given_p,
        null_overlap_k=null_overlap_k,
        null_overlap_k1=null_overlap_k1,
        null_session_idx=null_session_idx,
        null_subjects=null_subjects,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_fig2_data(refresh=False, refresh_decomposition=False):
    """Load the derived fig2 bundle, recomputing if needed.

    ``refresh`` rebuilds only the derived bundle (stage 2). The expensive
    per-session decomposition (stage 1, GPU) is preserved unless
    ``refresh_decomposition=True``.
    """
    refresh = refresh or REFRESH
    if DERIVED_CACHE.exists() and not refresh:
        print(f"Loading cached fig2 derived bundle from {DERIVED_CACHE}")
        with open(DERIVED_CACHE, "rb") as f:
            return dill.load(f)

    session_results = _load_or_compute_session_results(refresh=refresh_decomposition)

    windows_ms = [r["window_ms"] for r in session_results[0]["results"]]
    windows_bins = [r["window_bins"] for r in session_results[0]["results"]]
    session_names = [sr["session"] for sr in session_results]
    subjects = [sr["subject"] for sr in session_results]
    n_sessions = len(session_results)
    print(f"\nLoaded {n_sessions} sessions: {session_names}")
    print(f"Windows (bins): {windows_bins} -> (ms): "
          f"{[f'{w:.1f}' for w in windows_ms]}")

    metrics = _compute_metrics(session_results, windows_ms, windows_bins)
    m_by_window, subject_per_neuron_by_window, alpha_stats = _compute_alpha_stats(
        metrics, windows_ms
    )
    fano_stats = _compute_fano_stats(metrics, windows_ms)
    nc_stats = _compute_nc_stats(metrics, windows_ms)
    subspace = _compute_subspace(session_results, session_names, subjects)

    bundle = dict(
        session_results=session_results,
        metrics=metrics,
        m_by_window=m_by_window,
        subject_per_neuron_by_window=subject_per_neuron_by_window,
        alpha_stats=alpha_stats,
        fano_stats=fano_stats,
        nc_stats=nc_stats,
        WINDOWS_MS=windows_ms,
        WINDOWS_BINS=windows_bins,
        SUBJECTS=SUBJECTS,
        SUBJECT_COLORS=SUBJECT_COLORS,
        session_names=session_names,
        subjects=subjects,
        n_sessions=n_sessions,
        SUBSPACE_WINDOW_IDX=SUBSPACE_WINDOW_IDX,
        SUBSPACE_K=SUBSPACE_K,
        config=dict(
            DT=DT, WINDOW_BINS=WINDOW_BINS, N_SHUFFLES=N_SHUFFLES,
            MIN_RATE_HZ=MIN_RATE_HZ, MIN_PSTH_R2=MIN_PSTH_R2,
            N_PSTH_SPLITS=N_PSTH_SPLITS, INTERCEPT_MODE=INTERCEPT_MODE,
            INTERCEPT_KWARGS=INTERCEPT_KWARGS, MIN_VAR=MIN_VAR, EPS_RHO=EPS_RHO,
        ),
        **subspace,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(DERIVED_CACHE, "wb") as f:
        dill.dump(bundle, f)
    print(f"\nCached fig2 derived bundle to {DERIVED_CACHE}")
    return bundle


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Precompute fig2 derived data.")
    p.add_argument("-r", "--refresh", action="store_true",
                   help="Force recompute of derived bundle (keeps decomposition cache).")
    p.add_argument("--recompute-decomposition", action="store_true",
                   help="Also drop decomposition cache and rerun raw decomposition.")
    args, _ = p.parse_known_args()

    load_fig2_data(
        refresh=args.refresh or args.recompute_decomposition,
        refresh_decomposition=args.recompute_decomposition,
    )
