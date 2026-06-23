"""Test FixRSVP split-half PSTH reliability for Luke step07 RF-valid units."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ROWLEY_REPO = ROOT.parent / "DataRowleyV1V2"
if ROWLEY_REPO.exists() and str(ROWLEY_REPO) not in sys.path:
    sys.path.insert(0, str(ROWLEY_REPO))

from DataRowleyV1V2.utils.datasets import DictDataset  # noqa: E402
from VisionCore.covariance import align_fixrsvp_trials  # noqa: E402
from VisionCore.paths import FIGURES_DIR, STATS_DIR  # noqa: E402


STEP07_VISUAL_RF = (
    ROWLEY_REPO / "outputs" / "luke_step07_rf_survey" / "luke_step07_visual_rf_geometry.csv"
)
OUT_DIR = FIGURES_DIR / "fig2_rf_compare"
STAT_DIR = STATS_DIR / "fig2_rf_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

DT = 1.0 / 120.0
N_SPLITS = 100
MIN_PSTH_R2 = 0.05
MIN_RATE_HZ = 0.5


def _split_half_psth_r2(
    robs: np.ndarray,
    n_splits: int,
    seed: int = 42,
    min_valid_bins: int = 10,
    min_trials_per_half: int = 2,
) -> np.ndarray:
    """NaN-aware split-half PSTH R2, matching fig2 compute_fig2_data.py."""
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
        idx_b = perm[half : 2 * half]
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


def _session_root_from_calibration_path(path: str) -> Path:
    return Path(path).parents[2]


def _load_step07_units() -> pd.DataFrame:
    df = pd.read_csv(STEP07_VISUAL_RF)
    keep = (df["status"] == "ok") & df["rf_valid"].astype(bool)
    return df[keep].copy()


def _analyze_session(session_df: pd.DataFrame) -> list[dict]:
    first = session_df.iloc[0]
    session = str(first["session"])
    eye = str(first["eye"])
    root = _session_root_from_calibration_path(str(first["calibration_path"]))
    dset_path = root / "datasets" / f"{eye}_eye" / "fixrsvp.dset"
    if not dset_path.exists():
        return [
            {
                "session": session,
                "eye": eye,
                "cid": int(row["cluster_id"]),
                "status": "missing_fixrsvp",
                "fixrsvp_path": str(dset_path),
            }
            for _, row in session_df.iterrows()
        ]

    dset = DictDataset.load(dset_path)
    cluster_ids = np.asarray(dset.metadata.get("cluster_ids", []), dtype=int)
    if cluster_ids.size == 0:
        n_cols = np.asarray(dset.covariates["robs"]).shape[1]
        cluster_ids = np.arange(n_cols, dtype=int)

    robs, _, _, neuron_mask, meta = align_fixrsvp_trials(
        dset,
        valid_time_bins=120,
        min_fix_dur=20,
        min_total_spikes=0,
        fixation_radius=1.5,
        fixation_center="median_valid",
        require_dpi_valid=True,
    )
    if robs is None:
        return [
            {
                "session": session,
                "eye": eye,
                "cid": int(row["cluster_id"]),
                "status": "alignment_failed",
                "fixrsvp_path": str(dset_path),
                **meta,
            }
            for _, row in session_df.iterrows()
        ]

    aligned_cids = cluster_ids[np.asarray(neuron_mask, dtype=int)]
    cid_to_col = {int(cid): i for i, cid in enumerate(aligned_cids)}
    psth_r2 = _split_half_psth_r2(robs, N_SPLITS, seed=42)
    n_spikes = np.nansum(robs, axis=(0, 1))
    n_valid_bins = np.sum(np.isfinite(robs), axis=(0, 1))
    rate_hz = np.where(
        n_valid_bins > 0,
        n_spikes / np.maximum(n_valid_bins, 1) / DT,
        np.nan,
    )

    rows = []
    for _, row in session_df.iterrows():
        cid = int(row["cluster_id"])
        col = cid_to_col.get(cid)
        base = {
            "session": session,
            "eye": eye,
            "cid": cid,
            "status": "ok" if col is not None else "missing_after_alignment",
            "fixrsvp_path": str(dset_path),
            "n_trials_total": int(meta.get("n_trials_total", 0)),
            "n_trials_good": int(meta.get("n_trials_good", 0)),
            "n_neurons_total": int(meta.get("n_neurons_total", 0)),
            "n_neurons_aligned": int(meta.get("n_neurons_used", 0)),
            "dots_snr": float(row["dots_snr"]),
            "dots_rf_eccentricity_deg": float(row["eccentricity_deg"]),
            "dots_rf_diameter_deg": float(row["rf_equiv_diameter_deg"]),
        }
        if col is not None:
            base.update(
                {
                    "fixrsvp_col": int(col),
                    "psth_r2": float(psth_r2[col]),
                    "rate_hz": float(rate_hz[col]),
                    "n_spikes_fixrsvp": float(n_spikes[col]),
                    "n_valid_bins_fixrsvp": int(n_valid_bins[col]),
                    "passes_psth_r2_gate": bool(
                        np.isfinite(psth_r2[col]) and psth_r2[col] > MIN_PSTH_R2
                    ),
                    "passes_rate_gate": bool(
                        np.isfinite(rate_hz[col]) and rate_hz[col] > MIN_RATE_HZ
                    ),
                    "passes_fig2_plotted_unit_gate": bool(
                        np.isfinite(psth_r2[col])
                        and psth_r2[col] > MIN_PSTH_R2
                        and np.isfinite(rate_hz[col])
                        and rate_hz[col] > MIN_RATE_HZ
                    ),
                }
            )
        rows.append(base)
    return rows


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(["session", "eye"], sort=True):
        ok = g[g["status"] == "ok"]
        rows.append(
            {
                "session": keys[0],
                "eye": keys[1],
                "n_step07_rf_valid": int(len(g)),
                "n_found_in_fixrsvp": int(len(ok)),
                "n_psth_r2_pass": int(ok["passes_psth_r2_gate"].sum()) if len(ok) else 0,
                "n_rate_pass": int(ok["passes_rate_gate"].sum()) if len(ok) else 0,
                "n_fig2_plotted_gate_pass": int(ok["passes_fig2_plotted_unit_gate"].sum())
                if len(ok)
                else 0,
                "psth_r2_median": float(ok["psth_r2"].median()) if len(ok) else np.nan,
                "psth_r2_iqr25": float(ok["psth_r2"].quantile(0.25)) if len(ok) else np.nan,
                "psth_r2_iqr75": float(ok["psth_r2"].quantile(0.75)) if len(ok) else np.nan,
                "rate_hz_median": float(ok["rate_hz"].median()) if len(ok) else np.nan,
            }
        )
    overall = df[df["status"] == "ok"]
    rows.append(
        {
            "session": "ALL",
            "eye": "",
            "n_step07_rf_valid": int(len(df)),
            "n_found_in_fixrsvp": int(len(overall)),
            "n_psth_r2_pass": int(overall["passes_psth_r2_gate"].sum()) if len(overall) else 0,
            "n_rate_pass": int(overall["passes_rate_gate"].sum()) if len(overall) else 0,
            "n_fig2_plotted_gate_pass": int(overall["passes_fig2_plotted_unit_gate"].sum())
            if len(overall)
            else 0,
            "psth_r2_median": float(overall["psth_r2"].median()) if len(overall) else np.nan,
            "psth_r2_iqr25": float(overall["psth_r2"].quantile(0.25)) if len(overall) else np.nan,
            "psth_r2_iqr75": float(overall["psth_r2"].quantile(0.75)) if len(overall) else np.nan,
            "rate_hz_median": float(overall["rate_hz"].median()) if len(overall) else np.nan,
        }
    )
    return pd.DataFrame(rows)


def _plot(df: pd.DataFrame, out_path: Path) -> None:
    ok = df[df["status"] == "ok"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), constrained_layout=True)
    axes[0].hist(ok["psth_r2"].dropna(), bins=np.linspace(0, 1, 31), color="0.35")
    axes[0].axvline(MIN_PSTH_R2, color="tab:red", lw=1.2)
    axes[0].set_xlabel("FixRSVP split-half PSTH R2")
    axes[0].set_ylabel("Units")
    sc = axes[1].scatter(
        ok["dots_rf_eccentricity_deg"],
        ok["psth_r2"],
        c=ok["rate_hz"],
        s=28,
        cmap="viridis",
        alpha=0.85,
        linewidths=0,
    )
    axes[1].axhline(MIN_PSTH_R2, color="tab:red", lw=1.2)
    axes[1].set_xlabel("Dots RF eccentricity (deg)")
    axes[1].set_ylabel("FixRSVP split-half PSTH R2")
    fig.colorbar(sc, ax=axes[1], label="FixRSVP rate (Hz)")
    fig.savefig(out_path, dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    units = _load_step07_units()
    rows: list[dict] = []
    for _, session_df in units.groupby(["session", "eye"], sort=True):
        rows.extend(_analyze_session(session_df))

    df = pd.DataFrame(rows)
    detail_path = STAT_DIR / "luke_step07_visual_rf_valid_split_half_reliability.csv"
    summary_path = STAT_DIR / "luke_step07_visual_rf_valid_split_half_reliability_summary.csv"
    fig_path = OUT_DIR / "luke_step07_visual_rf_valid_split_half_reliability.png"
    df.to_csv(detail_path, index=False)
    summary = _summarize(df)
    summary.to_csv(summary_path, index=False)
    _plot(df, fig_path)

    print(summary.to_string(index=False))
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
