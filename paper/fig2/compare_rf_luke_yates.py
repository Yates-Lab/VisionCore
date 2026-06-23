"""Compare RF eccentricity for Luke versus Allen/Logan.

The output uses degrees/arcminutes, never raw pixels. Yates RFs are measured
from cached Gaborium STE maps using each session's gaborium ``ppd`` and
``roi_src`` metadata. Rowley/Luke RFs are measured from dots calibration maps,
using the calibration ``roi_deg``/``dxy_deg`` grid and session ``ppd``. Rowley
dots calibration STAs are saved before the FaceCal PRL bias correction in
DataRowleyV1V2, so their centers are shifted into the final PRL-corrected gaze
frame using the per-session ``dpi_calibration/*/calibration_params.npz`` bias.

Equivalent RF diameters are retained in the unit CSV as source-specific
diagnostics only. They should not be interpreted across subjects here because
Luke comes from coarse dots-calibration STAs while Allen/Logan come from
fine-resolution Gaborium STE maps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.ndimage import gaussian_filter, label


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ROWLEY_REPO = ROOT.parent / "DataRowleyV1V2"
if ROWLEY_REPO.exists() and str(ROWLEY_REPO) not in sys.path:
    sys.path.insert(0, str(ROWLEY_REPO))

from DataYatesV1.utils.io import YatesV1Session  # noqa: E402
from DataRowleyV1V2.shifter.preprocess import get_contour_mask_and_properties  # noqa: E402
from eval.sta_ste import cache_path as yates_sta_cache_path  # noqa: E402
from eval.sta_ste import compute_snr  # noqa: E402
from models.config_loader import load_dataset_configs  # noqa: E402
from VisionCore.paths import CACHE_DIR, FIGURES_DIR, STATS_DIR  # noqa: E402


DATASET_CONFIGS_PATH = (
    ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long_yates_rowley.yaml"
)
FIG2_LOOK_DERIVED = CACHE_DIR / "fig2_derived_yates_rowley_look.pkl"
OUT_DIR = FIGURES_DIR / "fig2_rf_compare"
STAT_DIR = STATS_DIR / "fig2_rf_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)
STEP07_LUKE_VISUAL_RF_GEOMETRY = (
    ROWLEY_REPO / "outputs" / "luke_step07_rf_survey" / "luke_step07_visual_rf_geometry.csv"
)

ROWLEY_PROCESSED_ROOT = Path("/mnt/ssd2/RowleyMarmoV1V2/processed")
ROWLEY_DOTS_SNR_THRESH = 10.0
ROWLEY_DOTS_MIN_SPIKES = 2000
ROWLEY_FOVEAL_DEPTH_BAND_UM = 1500.0
YATES_STE_SNR_THRESH = 5.0
MIN_COMPONENT_PIXELS = 3
MEASUREMENT_NOTE = """RF comparison notes
===================

Eccentricity is the intended cross-subject comparison in this analysis.

The source-specific equivalent RF diameter columns in rf_unit_metrics.csv and
rf_subject_summary.csv are retained only as diagnostics. They are not comparable
between Luke and Allen/Logan because Luke is measured from dots-calibration STAs
on a 0.2 deg grid, while Allen/Logan are measured from Gaborium STE maps on a
1/37.504766 deg grid. The contour area is therefore strongly quantized for Luke.
Luke dots RF geometry uses the same raw half-peak contour measurement as
DataRowleyV1V2/scripts/diagnostics/survey_luke_step07_rf_geometry.py.
"""


def _session_subject(session_name: str) -> str:
    return session_name.split("_")[0]


def _rowley_initial_cid_pool(cfg: dict) -> tuple[np.ndarray, str]:
    for key in ("sortercontam", "qccontam", "cids"):
        values = np.asarray(cfg.get(key, []), dtype=int)
        if values.size:
            return values, key
    return np.asarray([], dtype=int), "none"


def _rowley_session_root(cfg: dict) -> Path:
    p = Path(cfg.get("directory", ""))
    for candidate in [p, *p.parents]:
        if (candidate / "dots_calibration").exists():
            return candidate
    return ROWLEY_PROCESSED_ROOT / cfg["session"]


def _component_metrics(
    image: np.ndarray,
    x_deg_grid: np.ndarray,
    y_deg_grid: np.ndarray,
    dxy_deg: float,
    threshold_frac: float = 0.5,
) -> dict | None:
    """Return center/eccentricity/area for the signed absolute-peak component."""
    img = np.asarray(image, dtype=float)
    centered = img - np.nanmedian(img)
    if not np.isfinite(centered).any():
        return None

    peak_flat = int(np.nanargmax(np.abs(centered)))
    peak_signed = float(np.ravel(centered)[peak_flat])
    if not np.isfinite(peak_signed) or peak_signed == 0:
        return None
    polarity = 1.0 if peak_signed > 0 else -1.0
    signed = polarity * centered
    peak = float(abs(peak_signed))

    mask = signed >= threshold_frac * peak
    labels, n_labels = label(mask)
    if n_labels < 1:
        return None

    peak_rc = np.unravel_index(peak_flat, centered.shape)
    component_id = labels[peak_rc]
    if component_id == 0:
        return None

    comp = labels == component_id
    if int(comp.sum()) < MIN_COMPONENT_PIXELS:
        return None

    weights = np.clip(signed, 0, None) * comp
    if float(weights.sum()) <= 0:
        weights = comp.astype(float)

    x0 = float(np.nansum(x_deg_grid * weights) / np.nansum(weights))
    y0 = float(np.nansum(y_deg_grid * weights) / np.nansum(weights))
    area_deg2 = float(comp.sum() * (dxy_deg ** 2))
    diameter_deg = float(2.0 * np.sqrt(area_deg2 / np.pi))
    eccentricity_deg = float(np.hypot(x0, y0))
    return {
        "rf_x_deg": x0,
        "rf_y_deg": y0,
        "eccentricity_deg": eccentricity_deg,
        "eccentricity_arcmin": 60.0 * eccentricity_deg,
        "rf_area_deg2": area_deg2,
        "rf_equiv_diameter_deg": diameter_deg,
        "rf_equiv_diameter_arcmin": 60.0 * diameter_deg,
        "component_pixels": int(comp.sum()),
        "peak_value": peak_signed,
        "rf_polarity": int(polarity),
    }


def _rowley_step07_contour_metrics(
    image: np.ndarray,
    i_centers_deg: np.ndarray,
    j_centers_deg: np.ndarray,
    dxy_deg: float,
    threshold_frac: float = 0.5,
) -> dict | None:
    """Measure Luke dots RFs with the step07 raw half-peak contour convention."""
    img = np.asarray(image, dtype=float)
    if not np.isfinite(img).any():
        return None

    peak_pos = np.unravel_index(int(np.nanargmax(np.abs(img))), img.shape)
    peak_value = float(img[peak_pos])
    if not np.isfinite(peak_value) or peak_value == 0:
        return None
    polarity = 1.0 if peak_value >= 0 else -1.0
    aligned = polarity * img
    threshold = threshold_frac * abs(peak_value)
    contour, mask, area_px, center_rc = get_contour_mask_and_properties(aligned, threshold)
    if contour is None or area_px <= 0 or not np.isfinite(center_rc).all():
        return None

    ctr_row, ctr_col = center_rc
    y0 = float(np.interp(ctr_row, np.arange(len(i_centers_deg)), i_centers_deg))
    x0 = float(np.interp(ctr_col, np.arange(len(j_centers_deg)), j_centers_deg))
    area_deg2 = float(area_px) * dxy_deg * dxy_deg
    diameter_deg = float(2.0 * np.sqrt(area_deg2 / np.pi))
    eccentricity_deg = float(np.hypot(x0, y0))
    return {
        "rf_x_deg": x0,
        "rf_y_deg": y0,
        "eccentricity_deg": eccentricity_deg,
        "eccentricity_arcmin": 60.0 * eccentricity_deg,
        "rf_area_deg2": area_deg2,
        "rf_equiv_diameter_deg": diameter_deg,
        "rf_equiv_diameter_arcmin": 60.0 * diameter_deg,
        "component_pixels": int(area_px),
        "peak_value": peak_value,
        "rf_polarity": int(polarity),
        "rowley_rf_measurement": "step07_raw_half_peak_contour",
    }


def _rowley_prl_bias(root: Path, eye: str) -> dict:
    params_path = root / "dpi_calibration" / f"{eye}_eye" / "calibration_params.npz"
    if not params_path.exists():
        return {
            "rowley_prl_bias_row_px": np.nan,
            "rowley_prl_bias_col_px": np.nan,
            "rowley_prl_bias_mag_deg": np.nan,
            "rowley_prl_bias_source": "missing",
        }
    z = np.load(params_path, allow_pickle=True)
    bias_pix = np.asarray(z.get("bias_pix", np.array([np.nan, np.nan])), dtype=float)
    ppd = float(z.get("ppd", np.nan))
    if bias_pix.size != 2 or not np.isfinite(ppd) or ppd <= 0:
        mag_deg = np.nan
    else:
        mag_deg = float(np.linalg.norm(bias_pix) / ppd)
    return {
        "rowley_prl_bias_row_px": float(bias_pix[0]) if bias_pix.size >= 1 else np.nan,
        "rowley_prl_bias_col_px": float(bias_pix[1]) if bias_pix.size >= 2 else np.nan,
        "rowley_prl_bias_mag_deg": mag_deg,
        "rowley_prl_bias_source": str(params_path),
    }


def _rowley_rf_quality_mask(z: np.lib.npyio.NpzFile, snr: np.ndarray, n_spikes: np.ndarray) -> np.ndarray:
    """Conservative Rowley dots-RF mask, matching shifter RF diagnostics."""
    keep = np.isfinite(snr) & (snr >= ROWLEY_DOTS_SNR_THRESH)
    keep &= np.isfinite(n_spikes) & (n_spikes >= ROWLEY_DOTS_MIN_SPIKES)

    region = None
    if "calibration_region" in z.files:
        region = np.asarray(z["calibration_region"]).astype(str)
        if len(region) == len(keep):
            keep &= region == "V1"

    if "calibration_depth_um" in z.files:
        depth = np.asarray(z["calibration_depth_um"], dtype=float)
        if len(depth) == len(keep):
            depth_ref = depth[np.isfinite(depth)]
            if region is not None and len(region) == len(keep):
                depth_ref = depth[(region == "V1") & np.isfinite(depth)]
            if depth_ref.size:
                shallow_min = float(np.nanmax(depth_ref) - ROWLEY_FOVEAL_DEPTH_BAND_UM)
                keep &= depth >= shallow_min

    return keep


def _apply_rowley_prl_shift(metrics: dict, bias: dict, ppd: float) -> dict:
    """Shift pre-PRL dot STA centers into the final PRL-corrected gaze frame."""
    row_bias = float(bias.get("rowley_prl_bias_row_px", np.nan))
    col_bias = float(bias.get("rowley_prl_bias_col_px", np.nan))
    out = dict(metrics)
    out["rf_x_deg_pre_prl"] = float(metrics["rf_x_deg"])
    out["rf_y_deg_pre_prl"] = float(metrics["rf_y_deg"])
    out["eccentricity_deg_pre_prl"] = float(metrics["eccentricity_deg"])
    out["rowley_prl_shift_applied"] = False
    if np.isfinite(row_bias) and np.isfinite(col_bias) and np.isfinite(ppd) and ppd > 0:
        # Saved dots STAs used gaze_pre. Final gaze is gaze_pre - bias_pix, so
        # dot-relative coordinates in the final frame are shifted by +bias_pix.
        # This Rowley diagnostic branch reports row/i as positive y; conventional
        # visual elevation would flip both the pre-PRL row coordinate and this
        # row-bias term, leaving eccentricity unchanged.
        out["rf_x_deg"] = float(metrics["rf_x_deg"] + col_bias / ppd)
        out["rf_y_deg"] = float(metrics["rf_y_deg"] + row_bias / ppd)
        out["eccentricity_deg"] = float(np.hypot(out["rf_x_deg"], out["rf_y_deg"]))
        out["eccentricity_arcmin"] = 60.0 * out["eccentricity_deg"]
        out["rowley_prl_shift_applied"] = True
    return out


def _load_fig2_plotted_cids() -> dict[str, set[int]]:
    if not FIG2_LOOK_DERIVED.exists():
        return {}
    with open(FIG2_LOOK_DERIVED, "rb") as f:
        data = dill.load(f)
    min_rate = float(data["config"]["MIN_RATE_HZ"])
    min_r2 = float(data["config"]["MIN_PSTH_R2"])
    out: dict[str, set[int]] = {}
    for sr in data["session_results"]:
        cids = np.asarray(sr.get("cids", []), dtype=int)
        rate = np.asarray(sr.get("rate_hz", []), dtype=float)
        r2 = np.asarray(sr.get("psth_r2", []), dtype=float)
        keep = np.isfinite(rate) & (rate > min_rate) & np.isfinite(r2) & (r2 > min_r2)
        out[sr["session"]] = set(cids[keep].astype(int).tolist())
    return out


def _extract_yates_rows(cfg: dict, plotted_by_session: dict[str, set[int]]) -> list[dict]:
    session = cfg["session"]
    subject = _session_subject(session)
    path = yates_sta_cache_path(session)
    if not path.exists():
        return []

    z = np.load(path)
    stes = np.asarray(z["stes"])
    num_spikes = np.asarray(z["num_spikes"])
    snr, peak_lag, _ = compute_snr(stes)

    sess = YatesV1Session(session)
    dset = sess.get_dataset("gaborium")
    if dset is None:
        return []
    ppd = float(dset.metadata["ppd"])
    roi_origin = np.asarray(dset.metadata["roi_src"][:, 0], dtype=float)
    cluster_ids = np.asarray(sess.get_cluster_ids(), dtype=int)
    del dset

    candidate = set(np.asarray(cfg.get("cids", cfg.get("visual", [])), dtype=int).tolist())
    plotted = plotted_by_session.get(session, set())
    rows = []
    for row_idx, cid in enumerate(cluster_ids):
        cid = int(cid)
        if cid not in candidate:
            continue
        lag = int(peak_lag[row_idx])
        img = gaussian_filter(stes[row_idx, lag] - np.nanmedian(stes[row_idx, lag]), 1.0)
        h, w = img.shape
        rr, cc = np.indices((h, w))
        y_deg = -(roi_origin[0] + rr) / ppd
        x_deg = (roi_origin[1] + cc) / ppd
        dxy_deg = 1.0 / ppd
        metrics = _component_metrics(img, x_deg, y_deg, dxy_deg)
        if metrics is None:
            continue
        rows.append({
            "session": session,
            "subject": subject,
            "lab": "Yates",
            "rf_source": "gaborium_ste",
            "cid": cid,
            "ppd": ppd,
            "pool": "yaml_visual",
            "passes_visual_gate": bool(snr[row_idx] >= YATES_STE_SNR_THRESH),
            "fig2_plotted": cid in plotted,
            "rf_snr": float(snr[row_idx]),
            "n_spikes_rf_stim": float(num_spikes[row_idx]),
            "peak_lag": lag,
            **metrics,
        })
    return rows


def _extract_rowley_rows(cfg: dict, plotted_by_session: dict[str, set[int]]) -> list[dict]:
    session = cfg["session"]
    subject = _session_subject(session)
    root = _rowley_session_root(cfg)
    eye = cfg.get("eye", "right")
    cal_path = root / "dots_calibration" / f"{eye}_eye" / "calibration_results.npz"
    if not cal_path.exists():
        return []

    z = np.load(cal_path, allow_pickle=True)
    stas = np.asarray(z["optimized_stas"], dtype=float)
    snr = np.asarray(z["optimized_max_snr"], dtype=float)
    n_spikes = np.asarray(z.get("n_spikes", np.full(stas.shape[0], np.nan)), dtype=float)
    if "calibration_cluster_ids" in z.files:
        cal_cids = np.asarray(z["calibration_cluster_ids"], dtype=int)
    else:
        cal_cids = np.arange(stas.shape[0], dtype=int)

    roi_deg = np.asarray(z["roi_deg"], dtype=float)
    dxy_deg = float(z["dxy_deg"])
    ppd = float(z.get("ppd", np.nan))
    prl_bias = _rowley_prl_bias(root, eye)
    rowley_keep = _rowley_rf_quality_mask(z, snr, n_spikes)

    # Dots maps are indexed as row/i and col/j in gaze-centered coordinates.
    # Reconstruct the axes exactly as DataRowleyV1V2 does: build pixel edges
    # from roi_deg and session ppd, then divide the centers back by ppd.
    # DataRowleyV1V2 shifter diagnostics report azimuth=j/ppd and row/i as y.
    n_i, n_j = stas.shape[-2:]
    roi_pix = np.flipud(roi_deg * ppd)
    dxy_pix = dxy_deg * ppd
    i_edges = np.arange(roi_pix[0, 0], roi_pix[0, 1] + dxy_pix, dxy_pix)
    j_edges = np.arange(roi_pix[1, 0], roi_pix[1, 1] + dxy_pix, dxy_pix)
    i_centers = (i_edges[:-1] + i_edges[1:]) / (2.0 * ppd)
    j_centers = (j_edges[:-1] + j_edges[1:]) / (2.0 * ppd)
    if len(i_centers) != n_i:
        i_centers = np.linspace(roi_deg[0, 0] + dxy_deg / 2, roi_deg[0, 1] - dxy_deg / 2, n_i)
    if len(j_centers) != n_j:
        j_centers = np.linspace(roi_deg[1, 0] + dxy_deg / 2, roi_deg[1, 1] - dxy_deg / 2, n_j)
    ii, jj = np.meshgrid(i_centers, j_centers, indexing="ij")
    y_deg = ii
    x_deg = jj

    pool_cids, pool_source = _rowley_initial_cid_pool(cfg)
    pool = set(pool_cids.astype(int).tolist())
    plotted = plotted_by_session.get(session, set())
    cid_to_row = {int(cid): i for i, cid in enumerate(cal_cids.tolist())}

    rows = []
    for cid in sorted(pool):
        row_idx = cid_to_row.get(int(cid))
        if row_idx is None:
            continue
        if not bool(rowley_keep[row_idx]):
            continue
        lag_scores = np.nanmax(stas[row_idx], axis=(1, 2))
        lag = int(np.nanargmax(lag_scores))
        img = stas[row_idx, lag] - np.nanmedian(stas[row_idx, lag])
        metrics = _rowley_step07_contour_metrics(img, i_centers, j_centers, dxy_deg)
        if metrics is None:
            continue
        metrics = _apply_rowley_prl_shift(metrics, prl_bias, ppd)
        rows.append({
            "session": session,
            "subject": subject,
            "lab": "Rowley",
            "rf_source": "dots_calibration_sta_prl_shifted",
            "cid": int(cid),
            "ppd": ppd,
            "pool": pool_source,
            "passes_visual_gate": True,
            "fig2_plotted": int(cid) in plotted,
            "rf_snr": float(snr[row_idx]),
            "n_spikes_rf_stim": float(n_spikes[row_idx]),
            "rowley_rf_snr_thresh": ROWLEY_DOTS_SNR_THRESH,
            "rowley_rf_min_spikes": ROWLEY_DOTS_MIN_SPIKES,
            "rowley_foveal_depth_band_um": ROWLEY_FOVEAL_DEPTH_BAND_UM,
            "peak_lag": lag,
            **prl_bias,
            **metrics,
        })
    return rows


def _summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for subject, g in df.groupby("subject", sort=False):
        rows.append({
            "subset": label,
            "subject": subject,
            "n_units": int(len(g)),
            "n_sessions": int(g["session"].nunique()),
            "eccentricity_deg_median": float(g["eccentricity_deg"].median()),
            "eccentricity_deg_iqr25": float(g["eccentricity_deg"].quantile(0.25)),
            "eccentricity_deg_iqr75": float(g["eccentricity_deg"].quantile(0.75)),
            "rf_diameter_arcmin_median_source_specific": float(g["rf_equiv_diameter_arcmin"].median()),
            "rf_diameter_arcmin_iqr25_source_specific": float(g["rf_equiv_diameter_arcmin"].quantile(0.25)),
            "rf_diameter_arcmin_iqr75_source_specific": float(g["rf_equiv_diameter_arcmin"].quantile(0.75)),
        })
    return pd.DataFrame(rows)


def _pairwise_tests(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    metric = "eccentricity_deg"
    for a, b in (("Luke", "Allen"), ("Luke", "Logan")):
        xa = df.loc[df["subject"] == a, metric].dropna().to_numpy()
        xb = df.loc[df["subject"] == b, metric].dropna().to_numpy()
        if len(xa) == 0 or len(xb) == 0:
            continue
        stat, p = sp_stats.mannwhitneyu(xa, xb, alternative="two-sided")
        rows.append({
            "subset": label,
            "metric": metric,
            "comparison": f"{a} vs {b}",
            "n_a": int(len(xa)),
            "n_b": int(len(xb)),
            "median_a": float(np.nanmedian(xa)),
            "median_b": float(np.nanmedian(xb)),
            "mannwhitney_u": float(stat),
            "p_value": float(p),
        })
    return pd.DataFrame(rows)


def _load_step07_luke_visual_rf_valid() -> pd.DataFrame:
    if not STEP07_LUKE_VISUAL_RF_GEOMETRY.exists():
        return pd.DataFrame()

    src = pd.read_csv(STEP07_LUKE_VISUAL_RF_GEOMETRY)
    keep = (src["status"] == "ok") & src["rf_valid"].astype(bool)
    src = src[keep].copy()
    if src.empty:
        return pd.DataFrame()

    return pd.DataFrame({
        "session": src["session"].astype(str),
        "subject": "Luke",
        "lab": "Rowley",
        "rf_source": "step07_visual_rf_valid_dots_sta",
        "cid": src["cluster_id"].astype(int),
        "ppd": src["ppd_dataset"].astype(float),
        "pool": "step07_visual_rf_valid",
        "passes_visual_gate": True,
        "fig2_plotted": False,
        "rf_snr": src["dots_snr"].astype(float),
        "n_spikes_rf_stim": src["n_spikes"].astype(float),
        "peak_lag": src["peak_lag"].astype(int),
        "rf_x_deg": src["rf_center_az_deg"].astype(float),
        "rf_y_deg": src["rf_center_el_deg"].astype(float),
        "eccentricity_deg": src["eccentricity_deg"].astype(float),
        "eccentricity_arcmin": 60.0 * src["eccentricity_deg"].astype(float),
        "rf_area_deg2": src["rf_area_deg2"].astype(float),
        "rf_equiv_diameter_deg": src["rf_equiv_diameter_deg"].astype(float),
        "rf_equiv_diameter_arcmin": 60.0 * src["rf_equiv_diameter_deg"].astype(float),
        "component_pixels": src["rf_area_px"].astype(float),
        "peak_value": src["peak_value"].astype(float),
        "rf_polarity": np.where(src["polarity"].astype(str) == "ON", 1, -1),
        "rowley_rf_measurement": "step07_raw_half_peak_contour",
        "step07_rf_valid": True,
    })


def _plot(df: pd.DataFrame, out_path: Path) -> None:
    colors = {"Allen": "tab:blue", "Logan": "tab:green", "Luke": "tab:orange"}
    subjects = ["Allen", "Logan", "Luke"]
    fig, ax = plt.subplots(1, 1, figsize=(4.4, 3.4), constrained_layout=True)
    rng = np.random.default_rng(7)
    for i, subject in enumerate(subjects):
        vals = df.loc[df["subject"] == subject, "eccentricity_deg"].dropna().to_numpy()
        if vals.size == 0:
            continue
        x = i + rng.uniform(-0.18, 0.18, size=vals.size)
        ax.scatter(x, vals, s=11, alpha=0.35, color=colors[subject], linewidths=0)
        q25, med, q75 = np.nanpercentile(vals, [25, 50, 75])
        ax.plot([i - 0.22, i + 0.22], [med, med], color="k", lw=1.5)
        ax.plot([i, i], [q25, q75], color="k", lw=1.0)
    ax.set_xticks(range(len(subjects)), subjects)
    ax.set_ylabel("RF eccentricity from gaze center (deg)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_path, dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    cfgs = load_dataset_configs(str(DATASET_CONFIGS_PATH))
    plotted_by_session = _load_fig2_plotted_cids()

    rows: list[dict] = []
    for cfg in cfgs:
        subject = _session_subject(cfg["session"])
        if subject not in {"Allen", "Logan", "Luke"}:
            continue
        if cfg.get("lab", "").lower() == "rowley":
            rows.extend(_extract_rowley_rows(cfg, plotted_by_session))
        else:
            rows.extend(_extract_yates_rows(cfg, plotted_by_session))

    df = pd.DataFrame(rows)
    unit_csv = STAT_DIR / "rf_unit_metrics.csv"
    df.to_csv(unit_csv, index=False)

    visual = df[df["passes_visual_gate"]].copy()
    plotted = df[df["fig2_plotted"]].copy()
    step07_luke_visual = _load_step07_luke_visual_rf_valid()
    visual_with_step07_luke = pd.concat(
        [visual[visual["subject"] != "Luke"].copy(), step07_luke_visual],
        ignore_index=True,
    )
    summary = pd.concat([
        _summary(visual, "visual_rf_candidate"),
        _summary(plotted, "fig2_plotted"),
    ], ignore_index=True)
    tests = pd.concat([
        _pairwise_tests(visual, "visual_rf_candidate"),
        _pairwise_tests(plotted, "fig2_plotted"),
    ], ignore_index=True)
    summary_csv = STAT_DIR / "rf_subject_summary.csv"
    tests_csv = STAT_DIR / "rf_pairwise_tests.csv"
    notes_path = STAT_DIR / "rf_measurement_notes.txt"
    summary.to_csv(summary_csv, index=False)
    tests.to_csv(tests_csv, index=False)
    notes_path.write_text(MEASUREMENT_NOTE)

    if len(step07_luke_visual):
        step07_unit_csv = STAT_DIR / "rf_unit_metrics_luke_step07_visual_rf_valid.csv"
        step07_summary_csv = STAT_DIR / "rf_subject_summary_luke_step07_visual_rf_valid.csv"
        step07_tests_csv = STAT_DIR / "rf_pairwise_tests_luke_step07_visual_rf_valid.csv"
        visual_with_step07_luke.to_csv(step07_unit_csv, index=False)
        step07_summary = _summary(
            visual_with_step07_luke,
            "visual_rf_candidate_luke_step07_visual_rf_valid",
        )
        step07_tests = _pairwise_tests(
            visual_with_step07_luke,
            "visual_rf_candidate_luke_step07_visual_rf_valid",
        )
        step07_summary.to_csv(step07_summary_csv, index=False)
        step07_tests.to_csv(step07_tests_csv, index=False)
        _plot(
            visual_with_step07_luke,
            OUT_DIR / "rf_eccentricity_visual_candidates_luke_step07_visual_rf_valid.png",
        )

    _plot(visual, OUT_DIR / "rf_eccentricity_visual_candidates.png")
    if len(plotted):
        _plot(plotted, OUT_DIR / "rf_eccentricity_fig2_plotted.png")

    print(f"Wrote {len(df)} unit rows to {unit_csv}")
    print(summary.to_string(index=False))
    print(f"Wrote pairwise tests to {tests_csv}")
    print(f"Wrote measurement notes to {notes_path}")
    if len(step07_luke_visual):
        print(
            "Wrote Luke step07 visual RF-valid comparison "
            f"(n={len(step07_luke_visual)}) to {OUT_DIR}"
        )
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
