"""Step-through QC for the V1-RR100 movie-medoid population.

Open this file in VS Code/Jupyter as a percent-cell script and run cells one at
a time. It uses saved population specs and cached activation movies; it does not
run the VisionCore model.
"""

# %%
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:  # pragma: no cover - notebook convenience only.
    def display(obj):
        print(obj)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from declan.redundancy_resolved_v1_population import apply_population_view, load_population_view
from declan.run_cached_rr_medoid_compression_frontier import (
    BASE_VERSION,
    CONSTRUCTION_RECON_CSV,
    OUT_DIR as FINGERPRINT_OUT_DIR,
    _safe_slug,
    combine_similarity_matrices,
    correlation_from_fingerprints,
    corr_1d,
    load_activation_movie,
    movie_paths_from_reconstruction_csv,
)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
    }
)

# %% Knobs
RR100_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75"
    "_medoidPosthocminRepcomplete0p45_movieMedoid"
)
RR100_MEAN_VERSION = (
    "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75"
    "_medoidPosthocminRepcomplete0p45_mean"
)
RR192_VERSION = (
    "V1-RR_complete_0p65_moviesplit0p75_pair0p60_rec4_blockjkP0p50n5L0p50n4_merge2nd1.01"
)

POPULATION_SHORT_NAME = "V1-RR100 movie-medoid"
SAVE_FIGURES = True
FORCE_RECOMPUTE = False
N_HELDOUT_BACKIMAGE_CASES = 6
N_GROUPS_TO_PLOT = 8
MAX_MEMBERS_PER_GROUP_PLOT = 7
N_SHARP_SINGLETONS = 16
N_SHARP_REPS = 16
RANDOM_SEED = 7
ALLOW_NEGATIVE_AFFINE_GAIN = False
AFFINE_GAIN_TAG = "signed" if ALLOW_NEGATIVE_AFFINE_GAIN else "nonnegative"

QC_DIR = ROOT / "outputs" / "redundancy_resolved_v1_twin" / f"rr100_movie_medoid_qc_{_safe_slug(RR100_VERSION)}"
FIG_DIR = QC_DIR / "figures"
QC_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("population:", RR100_VERSION)
print("qc_dir:", QC_DIR)


# %% Small helpers
def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _as_table(rows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    return rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)


def _zscore_1d(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32)
    return (y - np.nanmean(y)) / (np.nanstd(y) + eps)


def center_pixel_traces(movie: np.ndarray, channels: np.ndarray | list[int]) -> np.ndarray:
    y = np.asarray(movie, dtype=np.float32)
    channels = np.asarray(channels, dtype=int)
    cy = y.shape[2] // 2
    cx = y.shape[3] // 2
    return y[:, channels, cy, cx]


def spatial_mean_traces(movie: np.ndarray, channels: np.ndarray | list[int]) -> np.ndarray:
    y = np.asarray(movie, dtype=np.float32)
    channels = np.asarray(channels, dtype=int)
    return y[:, channels].mean(axis=(2, 3))


def flatten_unit_movie(movie: np.ndarray, channels: np.ndarray | list[int]) -> np.ndarray:
    y = np.asarray(movie, dtype=np.float32)
    channels = np.asarray(channels, dtype=int)
    return np.transpose(y[:, channels], (1, 0, 2, 3)).reshape(channels.size, -1)


def corr_matrix_for_population_movie(movie: np.ndarray, population_view, normalization: str = "zscore") -> np.ndarray:
    reduced = apply_population_view(movie, population_view).astype(np.float32, copy=False)
    fp = np.transpose(reduced, (1, 0, 2, 3)).reshape(reduced.shape[1], -1)
    if normalization != "none":
        fp -= np.nanmean(fp, axis=1, keepdims=True)
    if normalization == "zscore":
        fp /= np.nanstd(fp, axis=1, keepdims=True) + 1e-8
    elif normalization not in {"none", "center"}:
        raise ValueError(f"Unknown normalization {normalization!r}")
    return correlation_from_fingerprints(fp)


def plot_corr_heatmap(corr: np.ndarray, *, title: str, save_name: str | None = None):
    corr = np.asarray(corr, dtype=np.float32)
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    dist = np.clip(1.0 - corr.astype(np.float64), 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    order = leaves_list(linkage(squareform(dist, checks=False), method="average")) if corr.shape[0] > 2 else np.arange(corr.shape[0])
    offdiag = corr[np.triu_indices(corr.shape[0], k=1)]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    for ax, mat, subtitle in (
        (axes[0], corr, "spec order"),
        (axes[1], corr[np.ix_(order, order)], "clustered order"),
    ):
        im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(subtitle)
        ax.set_xlabel("representative")
        ax.set_ylabel("representative")
    axes[2].hist(offdiag[np.isfinite(offdiag)], bins=np.linspace(-1, 1, 61), color="0.35")
    axes[2].axvline(0, color="0.2", lw=0.8)
    axes[2].set_xlabel("off-diagonal corr")
    axes[2].set_ylabel("pairs")
    axes[2].set_title("pair distribution")
    fig.colorbar(im, ax=axes[:2], shrink=0.8, label="corr")
    fig.suptitle(title, fontsize=10)
    if SAVE_FIGURES and save_name:
        fig.savefig(FIG_DIR / save_name, dpi=180, bbox_inches="tight")
    return fig


def selected_channels_from_reps(representatives: pd.DataFrame) -> np.ndarray:
    reps = representatives.sort_values("rep_idx")
    channels = reps["selected_channel"].fillna(-1).astype(int).to_numpy()
    if np.any(channels < 0):
        raise ValueError("All RR100 representatives should have selected_channel set.")
    return channels


def representative_member_lists(representatives: pd.DataFrame) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for row in representatives.to_dict("records"):
        out[int(row["rep_idx"])] = [int(ch) for ch in row["members"]]
    return out


def channel_to_rep_from_cluster_membership(cluster_membership: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mem = np.asarray(cluster_membership, dtype=np.float32)
    represented = np.sum(mem > 0, axis=0) == 1
    channel_to_rep = np.argmax(mem[:, represented] > 0, axis=0)
    return channel_to_rep.astype(np.int32, copy=False), represented


def expand_reduced_to_full(reduced: np.ndarray, cluster_membership: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channel_to_rep, represented = channel_to_rep_from_cluster_membership(cluster_membership)
    expanded = np.asarray(reduced, dtype=np.float32)[:, channel_to_rep]
    return expanded, represented


def channelwise_affine_sufficient_stats(
    expanded: np.ndarray,
    full: np.ndarray,
    *,
    chunk_channels: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_all = np.asarray(expanded, dtype=np.float32)
    y_all = np.asarray(full, dtype=np.float32)
    if x_all.shape != y_all.shape:
        raise ValueError(f"expanded/full shape mismatch: {x_all.shape} vs {y_all.shape}")
    n_channels = x_all.shape[1]
    n_samples = int(x_all.shape[0] * x_all.shape[2] * x_all.shape[3])
    n = np.full(n_channels, n_samples, dtype=np.float64)
    sum_x = np.zeros(n_channels, dtype=np.float64)
    sum_y = np.zeros(n_channels, dtype=np.float64)
    sum_x2 = np.zeros(n_channels, dtype=np.float64)
    sum_xy = np.zeros(n_channels, dtype=np.float64)
    axes = (0, 2, 3)
    for start in range(0, n_channels, int(chunk_channels)):
        stop = min(start + int(chunk_channels), n_channels)
        x = x_all[:, start:stop]
        y = y_all[:, start:stop]
        sum_x[start:stop] = np.sum(x, axis=axes, dtype=np.float64)
        sum_y[start:stop] = np.sum(y, axis=axes, dtype=np.float64)
        sum_x2[start:stop] = np.sum(x * x, axis=axes, dtype=np.float64)
        sum_xy[start:stop] = np.sum(x * y, axis=axes, dtype=np.float64)
    return n, sum_x, sum_y, sum_x2, sum_xy


def affine_params_from_sufficient_stats(
    n: np.ndarray,
    sum_x: np.ndarray,
    sum_y: np.ndarray,
    sum_x2: np.ndarray,
    sum_xy: np.ndarray,
    eps: float = 1e-8,
    allow_negative_gain: bool = ALLOW_NEGATIVE_AFFINE_GAIN,
) -> tuple[np.ndarray, np.ndarray]:
    mean_x = sum_x / np.maximum(n, 1)
    mean_y = sum_y / np.maximum(n, 1)
    var_x = sum_x2 / np.maximum(n, 1) - mean_x * mean_x
    cov_xy = sum_xy / np.maximum(n, 1) - mean_x * mean_y
    gain = cov_xy / np.maximum(var_x, eps)
    if not allow_negative_gain:
        gain = np.maximum(gain, 0.0)
    offset = mean_y - gain * mean_x
    return gain.astype(np.float32), offset.astype(np.float32)


def fit_per_channel_affine(expanded: np.ndarray, full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit y ~= gain * x + offset separately for every represented channel."""
    return affine_params_from_sufficient_stats(*channelwise_affine_sufficient_stats(expanded, full))


def apply_per_channel_affine(expanded: np.ndarray, gain: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return (
        np.asarray(expanded, dtype=np.float32) * np.asarray(gain, dtype=np.float32)[None, :, None, None]
        + np.asarray(offset, dtype=np.float32)[None, :, None, None]
    )


def reconstruction_metric_row(
    full: np.ndarray,
    expanded: np.ndarray,
    *,
    version: str,
    case_label: str,
    n_representatives: int,
    expansion_mode: str,
) -> dict[str, Any]:
    residual = expanded - full
    per_channel_corr = np.asarray([corr_1d(full[:, ch], expanded[:, ch]) for ch in range(full.shape[1])], dtype=np.float32)
    return {
        "version": version,
        "case": case_label,
        "expansion_mode": expansion_mode,
        "n_time": int(full.shape[0]),
        "n_representatives": int(n_representatives),
        "n_represented_channels": int(full.shape[1]),
        "global_rate_corr": corr_1d(full, expanded),
        "rmse": float(np.sqrt(np.nanmean(residual * residual))),
        "nrmse_by_full_std": float(np.sqrt(np.nanmean(residual * residual)) / max(float(np.nanstd(full)), 1e-8)),
        "min_channel_corr": float(np.nanmin(per_channel_corr)),
        "p05_channel_corr": float(np.nanpercentile(per_channel_corr, 5)),
        "median_channel_corr": float(np.nanmedian(per_channel_corr)),
    }


def reconstruction_metrics_for_case(
    movie: np.ndarray,
    population_view,
    *,
    case_label: str,
    include_same_case_affine: bool = True,
    affine_params: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    reduced = apply_population_view(movie, population_view).astype(np.float32, copy=False)
    expanded, represented = expand_reduced_to_full(reduced, population_view.cluster_membership)
    full = np.asarray(movie, dtype=np.float32)[:, represented]
    rows = [
        reconstruction_metric_row(
            full,
            expanded,
            version=population_view.name,
            case_label=case_label,
            n_representatives=reduced.shape[1],
            expansion_mode="raw_copy",
        )
    ]
    if include_same_case_affine:
        gain, offset = fit_per_channel_affine(expanded, full)
        rows.append(
            reconstruction_metric_row(
                full,
                apply_per_channel_affine(expanded, gain, offset),
                version=population_view.name,
                case_label=case_label,
                n_representatives=reduced.shape[1],
                expansion_mode="same_case_channel_affine",
            )
        )
    if affine_params is not None:
        gain, offset = affine_params
        rows.append(
            reconstruction_metric_row(
                full,
                apply_per_channel_affine(expanded, gain, offset),
                version=population_view.name,
                case_label=case_label,
                n_representatives=reduced.shape[1],
                expansion_mode="construction_fit_channel_affine",
            )
        )
    return rows


def fit_affine_params_from_cases(population_view, cases: list[tuple[str, Path]]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Fit per-channel affine copy-back gains from a list of cached movies."""
    channel_to_rep, represented = channel_to_rep_from_cluster_membership(population_view.cluster_membership)
    n_channels = int(np.sum(represented))
    n = np.zeros(n_channels, dtype=np.float64)
    sum_x = np.zeros(n_channels, dtype=np.float64)
    sum_y = np.zeros(n_channels, dtype=np.float64)
    sum_x2 = np.zeros(n_channels, dtype=np.float64)
    sum_xy = np.zeros(n_channels, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for case, path in cases:
        print(f"fit affine {population_view.name} | {case}")
        movie = load_activation_movie(path)
        reduced = apply_population_view(movie, population_view).astype(np.float32, copy=False)
        expanded = reduced[:, channel_to_rep]
        full = np.asarray(movie, dtype=np.float32)[:, represented]
        case_n, case_sum_x, case_sum_y, case_sum_x2, case_sum_xy = channelwise_affine_sufficient_stats(
            expanded,
            full,
        )
        n += case_n
        sum_x += case_sum_x
        sum_y += case_sum_y
        sum_x2 += case_sum_x2
        sum_xy += case_sum_xy
        samples = int(case_n[0]) if case_n.size else 0
        rows.append({"version": population_view.name, "case": case, "n_samples_per_channel": samples})
        del movie, reduced, expanded, full
    gain, offset = affine_params_from_sufficient_stats(n, sum_x, sum_y, sum_x2, sum_xy)
    return gain, offset, pd.DataFrame(rows)


def spatial_sharpness(maps: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(maps, dtype=np.float32).reshape(maps.shape[0], -1)
    p99 = np.nanpercentile(x, 99, axis=1)
    p50 = np.nanpercentile(x, 50, axis=1)
    p25 = np.nanpercentile(x, 25, axis=1)
    return (p99 - p50) / (np.abs(p50 - p25) + eps)


def pick_peak_frame(movie: np.ndarray, channels: np.ndarray | list[int]) -> int:
    y = np.asarray(movie, dtype=np.float32)
    channels = np.asarray(channels, dtype=int)
    if channels.size == 0:
        return int(y.shape[0] - 1)
    score = np.nanpercentile(y[:, channels].reshape(y.shape[0], -1), 99, axis=1)
    return int(np.nanargmax(score))


# %% Step 1: load RR100 specs and inspect membership bookkeeping
rr100_view = load_population_view(version_name=RR100_VERSION)
rr100_mean_view = load_population_view(version_name=RR100_MEAN_VERSION)
try:
    rr192_view = load_population_view(version_name=RR192_VERSION)
except FileNotFoundError:
    rr192_view = None

membership = rr100_view.membership
cluster_membership = rr100_view.cluster_membership
assert membership is not None and cluster_membership is not None
representatives = pd.DataFrame(rr100_view.meta["representatives"]).sort_values("rep_idx").reset_index(drop=True)
selected_channels = selected_channels_from_reps(representatives)
member_lists = representative_member_lists(representatives)

summary_rows = [
    {
        "version": rr100_view.name,
        "pooling_mode": rr100_view.meta.get("pooling_mode"),
        "n_representatives": rr100_view.n_units,
        "n_input_channels": rr100_view.input_channels,
        "max_pooling_nnz": int((membership > 0).sum(axis=1).max()),
        "max_cluster_size": int((cluster_membership > 0).sum(axis=1).max()),
        "n_groups": int((representatives["n_members"] > 1).sum()),
        "n_singletons": int((representatives["n_members"] == 1).sum()),
    }
]
display(_as_table(summary_rows))
display(
    representatives.sort_values("n_members", ascending=False)
    [["rep_idx", "kind", "n_members", "selected_channel", "selected_ccnorm", "sessions"]]
    .head(20)
)


# %% Step 2: choose construction and held-out cached movies
construction_cases = movie_paths_from_reconstruction_csv(CONSTRUCTION_RECON_CSV)
construction_case_labels = {case for case, _path in construction_cases if not case.startswith("FixRSVP_")}


def discover_heldout_backimage_cases(n_cases: int = N_HELDOUT_BACKIMAGE_CASES) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path, int]] = []
    for json_path in sorted(FINGERPRINT_OUT_DIR.glob("activation_movie_*_trace000_stored_frames*_lag32_out151x151_scale1.json")):
        with json_path.open(encoding="utf-8") as f:
            meta = json.load(f)
        image_key = str(meta.get("image_key", json_path.stem))
        if image_key in construction_case_labels:
            continue
        npz_path = json_path.with_suffix(".npz")
        if not npz_path.exists():
            continue
        rows.append((image_key, npz_path, int(meta.get("activation_shape", [0])[0])))
    # Prefer a deterministic spread through the cached BackImage library.
    rows = sorted(rows, key=lambda x: x[0].lower())
    return [(case, path) for case, path, _n in rows[: int(n_cases)]]


heldout_cases = discover_heldout_backimage_cases()
case_table = pd.DataFrame(
    [
        {"split": "construction", "case": case, "path": str(path)}
        for case, path in construction_cases
    ]
    + [
        {"split": "heldout_cached", "case": case, "path": str(path)}
        for case, path in heldout_cases
    ]
)
display(case_table)


# %% Step 3: representative correlation heatmaps on construction cases
def compute_population_corr_for_cases(population_view, cases: list[tuple[str, Path]], *, combine_mode: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    corr_parts: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for case, path in cases:
        print(f"corr {population_view.name} | {case}")
        movie = load_activation_movie(path)
        corr = corr_matrix_for_population_movie(movie, population_view)
        corr_parts.append(corr)
        offdiag = corr[np.triu_indices(corr.shape[0], k=1)]
        rows.append(
            {
                "version": population_view.name,
                "case": case,
                "n_representatives": int(corr.shape[0]),
                "offdiag_p95": float(np.nanpercentile(offdiag, 95)),
                "offdiag_p99": float(np.nanpercentile(offdiag, 99)),
                "offdiag_max": float(np.nanmax(offdiag)),
            }
        )
        del movie, corr
    combined = combine_similarity_matrices(corr_parts, mode=combine_mode)
    return combined, rows


construction_corr_path = QC_DIR / "rr100_movie_medoid_construction_corr_min.npz"
if construction_corr_path.exists() and not FORCE_RECOMPUTE:
    with np.load(construction_corr_path) as data:
        construction_corr = np.asarray(data["corr"], dtype=np.float32)
    construction_corr_rows = pd.read_csv(QC_DIR / "rr100_movie_medoid_construction_corr_inputs.csv").to_dict("records")
else:
    construction_corr, construction_corr_rows = compute_population_corr_for_cases(
        rr100_view,
        construction_cases,
        combine_mode="min",
    )
    np.savez_compressed(
        construction_corr_path,
        corr=construction_corr.astype(np.float32),
        case_labels=np.asarray([case for case, _path in construction_cases], dtype=str),
    )
    _write_csv(QC_DIR / "rr100_movie_medoid_construction_corr_inputs.csv", construction_corr_rows)

display(_as_table(construction_corr_rows))
fig = plot_corr_heatmap(
    construction_corr,
    title=f"{POPULATION_SHORT_NAME}: construction-case min representative corr",
    save_name="rr100_movie_medoid_construction_corr_heatmap.png",
)
plt.show()


# %% Step 4: representative correlation heatmaps on held-out cached BackImage cases
heldout_corr_path = QC_DIR / "rr100_movie_medoid_heldout_corr_min.npz"
if heldout_cases:
    if heldout_corr_path.exists() and not FORCE_RECOMPUTE:
        with np.load(heldout_corr_path) as data:
            heldout_corr = np.asarray(data["corr"], dtype=np.float32)
        heldout_corr_rows = pd.read_csv(QC_DIR / "rr100_movie_medoid_heldout_corr_inputs.csv").to_dict("records")
    else:
        heldout_corr, heldout_corr_rows = compute_population_corr_for_cases(
            rr100_view,
            heldout_cases,
            combine_mode="min",
        )
        np.savez_compressed(
            heldout_corr_path,
            corr=heldout_corr.astype(np.float32),
            case_labels=np.asarray([case for case, _path in heldout_cases], dtype=str),
        )
        _write_csv(QC_DIR / "rr100_movie_medoid_heldout_corr_inputs.csv", heldout_corr_rows)

    display(_as_table(heldout_corr_rows))
    fig = plot_corr_heatmap(
        heldout_corr,
        title=f"{POPULATION_SHORT_NAME}: held-out cached BackImage min representative corr",
        save_name="rr100_movie_medoid_heldout_corr_heatmap.png",
    )
    plt.show()
else:
    print("No held-out cached BackImage cases found.")


# %% Step 5: medoid-vs-cluster group quality across cases
def group_quality_for_case(movie: np.ndarray, case_label: str, representatives: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in representatives.to_dict("records"):
        rep_idx = int(row["rep_idx"])
        members = np.asarray(row["members"], dtype=int)
        if members.size <= 1:
            continue
        selected = int(row["selected_channel"])
        centroid = np.nanmean(movie[:, members], axis=1)
        selected_vec = movie[:, selected]
        member_corrs = np.asarray([corr_1d(movie[:, int(ch)], centroid) for ch in members], dtype=np.float32)
        selected_member_corrs = np.asarray([corr_1d(selected_vec, movie[:, int(ch)]) for ch in members if int(ch) != selected], dtype=np.float32)
        finite_member = member_corrs[np.isfinite(member_corrs)]
        finite_selected = selected_member_corrs[np.isfinite(selected_member_corrs)]
        rows.append(
            {
                "case": case_label,
                "rep_idx": rep_idx,
                "group_label": int(row["group_label"]),
                "n_members": int(members.size),
                "selected_channel": selected,
                "selected_to_centroid_corr": corr_1d(selected_vec, centroid),
                "min_member_centroid_corr": float(np.nanmin(finite_member)) if finite_member.size else np.nan,
                "median_member_centroid_corr": float(np.nanmedian(finite_member)) if finite_member.size else np.nan,
                "min_selected_member_corr": float(np.nanmin(finite_selected)) if finite_selected.size else np.nan,
                "median_selected_member_corr": float(np.nanmedian(finite_selected)) if finite_selected.size else np.nan,
                "members": ",".join(str(int(ch)) for ch in members),
            }
        )
    return rows


group_quality_csv = QC_DIR / "rr100_movie_medoid_group_quality.csv"
all_quality_cases = construction_cases + heldout_cases
if group_quality_csv.exists() and not FORCE_RECOMPUTE:
    group_quality = pd.read_csv(group_quality_csv)
else:
    quality_rows: list[dict[str, Any]] = []
    for case, path in all_quality_cases:
        print(f"group quality {case}")
        movie = load_activation_movie(path)
        quality_rows.extend(group_quality_for_case(movie, case, representatives))
        del movie
    group_quality = _as_table(quality_rows)
    group_quality.to_csv(group_quality_csv, index=False)

display(
    group_quality.sort_values(["min_selected_member_corr", "selected_to_centroid_corr"])
    .head(30)
    .style.format(precision=3)
)


# %% Step 6: plot group-quality summary metrics
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
bins = np.linspace(-1.0, 1.0, 65)
axes[0, 0].hist(group_quality["selected_to_centroid_corr"], bins=bins, alpha=0.75, color="tab:blue")
axes[0, 0].axvline(0, color="0.25", lw=0.8)
axes[0, 0].set_title("Selected medoid to cluster centroid")
axes[0, 0].set_xlabel("corr")
axes[0, 0].set_ylabel("group x case")

axes[0, 1].hist(group_quality["min_selected_member_corr"], bins=bins, alpha=0.75, color="tab:orange")
axes[0, 1].axvline(0, color="0.25", lw=0.8)
axes[0, 1].set_title("Worst selected-medoid/member corr")
axes[0, 1].set_xlabel("corr")

scatter = axes[1, 0].scatter(
    group_quality["n_members"],
    group_quality["selected_to_centroid_corr"],
    c=group_quality["min_selected_member_corr"],
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    s=np.clip(group_quality["n_members"].to_numpy() * 4, 12, 180),
    alpha=0.75,
    linewidths=0,
)
axes[1, 0].axhline(0, color="0.25", lw=0.8)
axes[1, 0].set_xlabel("cluster size")
axes[1, 0].set_ylabel("selected to centroid corr")
axes[1, 0].set_title("Quality vs size")
fig.colorbar(scatter, ax=axes[1, 0], shrink=0.8, label="min selected-member corr")

case_summary = (
    group_quality.groupby("case")
    .agg(
        worst_selected_member=("min_selected_member_corr", "min"),
        median_selected_centroid=("selected_to_centroid_corr", "median"),
    )
    .reset_index()
)
x = np.arange(len(case_summary))
axes[1, 1].plot(x, case_summary["worst_selected_member"], "o-", label="worst selected-member")
axes[1, 1].plot(x, case_summary["median_selected_centroid"], "o-", label="median selected-centroid")
axes[1, 1].axhline(0, color="0.25", lw=0.8)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(case_summary["case"], rotation=45, ha="right", fontsize=7)
axes[1, 1].set_ylabel("corr")
axes[1, 1].set_title("Per-case group quality")
axes[1, 1].legend(frameon=False, fontsize=8)
fig.suptitle(f"{POPULATION_SHORT_NAME}: medoid group quality", fontsize=11)
if SAVE_FIGURES:
    fig.savefig(FIG_DIR / "rr100_movie_medoid_group_quality_summary.png", bbox_inches="tight")
plt.show()


# %% Step 7: reconstruction metrics for RR100 medoid, RR100 mean, and RR192
# Expansion modes:
# - raw_copy: copy each representative response back to every original channel in its cluster.
# - same_case_channel_affine: fit one gain/offset per restored channel on the same case; useful as an upper bound.
# - construction_fit_channel_affine: fit gains/offsets on construction cases and apply them to each case.
#   This is the fairest held-out test when we allow "same shape but different gain" units to merge.
reconstruction_csv = QC_DIR / f"rr100_reconstruction_metrics_with_{AFFINE_GAIN_TAG}_affine.csv"
affine_params_path = QC_DIR / f"construction_fit_{AFFINE_GAIN_TAG}_affine_params.npz"
reference_views = [rr100_view, rr100_mean_view] + ([rr192_view] if rr192_view is not None else [])
if reconstruction_csv.exists() and not FORCE_RECOMPUTE:
    reconstruction = pd.read_csv(reconstruction_csv)
else:
    affine_params_by_version: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    affine_fit_rows: list[dict[str, Any]] = []
    if affine_params_path.exists() and not FORCE_RECOMPUTE:
        with np.load(affine_params_path) as data:
            for view in reference_views:
                slug = _safe_slug(view.name)
                affine_params_by_version[view.name] = (
                    np.asarray(data[f"gain_{slug}"], dtype=np.float32),
                    np.asarray(data[f"offset_{slug}"], dtype=np.float32),
                )
        affine_fit_table = pd.read_csv(QC_DIR / f"construction_fit_{AFFINE_GAIN_TAG}_affine_params_cases.csv")
    else:
        arrays: dict[str, np.ndarray] = {}
        for view in reference_views:
            gain, offset, fit_table = fit_affine_params_from_cases(view, construction_cases)
            affine_params_by_version[view.name] = (gain, offset)
            slug = _safe_slug(view.name)
            arrays[f"gain_{slug}"] = gain
            arrays[f"offset_{slug}"] = offset
            affine_fit_rows.extend(fit_table.to_dict("records"))
        np.savez_compressed(affine_params_path, **arrays)
        affine_fit_table = _as_table(affine_fit_rows)
        affine_fit_table.to_csv(QC_DIR / f"construction_fit_{AFFINE_GAIN_TAG}_affine_params_cases.csv", index=False)

    recon_rows: list[dict[str, Any]] = []
    for case, path in all_quality_cases:
        print(f"reconstruction {case}")
        movie = load_activation_movie(path)
        for view in reference_views:
            recon_rows.extend(
                reconstruction_metrics_for_case(
                    movie,
                    view,
                    case_label=case,
                    include_same_case_affine=True,
                    affine_params=affine_params_by_version[view.name],
                )
            )
        del movie
    reconstruction = _as_table(recon_rows)
    reconstruction.to_csv(reconstruction_csv, index=False)

display(reconstruction.sort_values(["case", "version", "expansion_mode"]).style.format(precision=3))

def short_reconstruction_label(version: str, expansion_mode: str) -> str:
    if version == RR100_VERSION:
        base = "RR100 medoid"
    elif version == RR100_MEAN_VERSION:
        base = "RR100 mean"
    elif version == RR192_VERSION:
        base = "RR192"
    else:
        base = version[-18:]
    mode = {
        "raw_copy": "raw",
        "same_case_channel_affine": "same-case scale",
        "construction_fit_channel_affine": "construction-fit scale",
    }.get(str(expansion_mode), str(expansion_mode))
    return f"{base} | {mode}"


fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.4), constrained_layout=True)
for (version, expansion_mode), sub in reconstruction.groupby(["version", "expansion_mode"], sort=False):
    sub = sub.reset_index(drop=True)
    label = short_reconstruction_label(version, expansion_mode)
    axes[0].plot(sub["case"], sub["global_rate_corr"], "o-", label=label)
    axes[1].plot(sub["case"], sub["p05_channel_corr"], "o-", label=label)
    axes[2].plot(sub["case"], sub["nrmse_by_full_std"], "o-", label=label)
for ax, ylabel, title in (
    (axes[0], "global reconstruction corr", "Pool-expand global reconstruction"),
    (axes[1], "p05 channel corr", "Weak-channel shape reconstruction"),
    (axes[2], "NRMSE / full std", "Amplitude-sensitive error"),
):
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(reconstruction["case"].unique())))
    ax.set_xticklabels(list(reconstruction["case"].unique()), rotation=45, ha="right", fontsize=7)
axes[0].legend(frameon=False, fontsize=6, ncol=1)
if SAVE_FIGURES:
    fig.savefig(FIG_DIR / f"rr100_reconstruction_metrics_with_{AFFINE_GAIN_TAG}_affine.png", bbox_inches="tight")
plt.show()


# %% Step 7b: summarize how much channel-wise scaling changes the reconstruction story
reconstruction_delta = (
    reconstruction.pivot_table(
        index=["version", "case"],
        columns="expansion_mode",
        values=["global_rate_corr", "p05_channel_corr", "nrmse_by_full_std"],
    )
    .reset_index()
)
reconstruction_delta.columns = [
    "_".join([str(part) for part in col if str(part)])
    if isinstance(col, tuple)
    else str(col)
    for col in reconstruction_delta.columns
]
if "global_rate_corr_raw_copy" in reconstruction_delta and "global_rate_corr_construction_fit_channel_affine" in reconstruction_delta:
    reconstruction_delta["delta_global_corr_construction_fit_minus_raw"] = (
        reconstruction_delta["global_rate_corr_construction_fit_channel_affine"]
        - reconstruction_delta["global_rate_corr_raw_copy"]
    )
if "nrmse_by_full_std_raw_copy" in reconstruction_delta and "nrmse_by_full_std_construction_fit_channel_affine" in reconstruction_delta:
    reconstruction_delta["delta_nrmse_construction_fit_minus_raw"] = (
        reconstruction_delta["nrmse_by_full_std_construction_fit_channel_affine"]
        - reconstruction_delta["nrmse_by_full_std_raw_copy"]
    )
reconstruction_delta.to_csv(QC_DIR / f"rr100_reconstruction_{AFFINE_GAIN_TAG}_affine_delta_summary.csv", index=False)
display(
    reconstruction_delta[
        [
            col
            for col in [
                "version",
                "case",
                "global_rate_corr_raw_copy",
                "global_rate_corr_construction_fit_channel_affine",
                "delta_global_corr_construction_fit_minus_raw",
                "nrmse_by_full_std_raw_copy",
                "nrmse_by_full_std_construction_fit_channel_affine",
                "delta_nrmse_construction_fit_minus_raw",
            ]
            if col in reconstruction_delta.columns
        ]
    ].style.format(precision=3)
)


# %% Step 8: choose groups for detailed plots
largest_reps = (
    representatives.query("n_members > 1")
    .sort_values("n_members", ascending=False)["rep_idx"]
    .head(N_GROUPS_TO_PLOT)
    .astype(int)
    .tolist()
)
worst_reps = (
    group_quality.sort_values(["min_selected_member_corr", "selected_to_centroid_corr"])
    .drop_duplicates("rep_idx")["rep_idx"]
    .head(N_GROUPS_TO_PLOT)
    .astype(int)
    .tolist()
)
GROUPS_TO_PLOT = list(dict.fromkeys(largest_reps + worst_reps))[: max(N_GROUPS_TO_PLOT, 1) * 2]
CASE_TO_PLOT = heldout_cases[0] if heldout_cases else construction_cases[0]
print("case_to_plot:", CASE_TO_PLOT[0])
print("largest:", largest_reps)
print("worst:", worst_reps)
print("plot:", GROUPS_TO_PLOT)


# %% Step 9: single-frame activation maps for selected groups
def plot_group_activation_maps(movie: np.ndarray, groups: list[int], *, case_label: str):
    reps_by_idx = {int(row["rep_idx"]): row for row in representatives.to_dict("records")}
    n_rows = len(groups)
    n_cols = MAX_MEMBERS_PER_GROUP_PLOT + 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.35 * n_cols, 1.35 * n_rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(n_rows, n_cols)
    for row_idx, rep_idx in enumerate(groups):
        row = reps_by_idx[int(rep_idx)]
        members = np.asarray(row["members"], dtype=int)
        selected = int(row["selected_channel"])
        shown_members = list(dict.fromkeys([selected] + [int(ch) for ch in members if int(ch) != selected]))[:MAX_MEMBERS_PER_GROUP_PLOT]
        frame = pick_peak_frame(movie, [selected])
        centroid_map = np.nanmean(movie[frame, members], axis=0)
        arrays = [movie[frame, int(ch)] for ch in shown_members] + [movie[frame, selected], centroid_map]
        finite = np.concatenate([a.ravel() for a in arrays])
        vmin, vmax = np.nanpercentile(finite, [1, 99.5])
        for col_idx, ax in enumerate(axes[row_idx]):
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx < len(shown_members):
                ch = int(shown_members[col_idx])
                ax.imshow(movie[frame, ch], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
                suffix = " selected" if ch == selected else ""
                ax.set_title(f"ch {ch}{suffix}", fontsize=7)
            elif col_idx == MAX_MEMBERS_PER_GROUP_PLOT:
                ax.imshow(movie[frame, selected], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_title("medoid", fontsize=7)
            elif col_idx == MAX_MEMBERS_PER_GROUP_PLOT + 1:
                ax.imshow(centroid_map, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_title("mean", fontsize=7)
            else:
                ax.set_axis_off()
        axes[row_idx, 0].set_ylabel(f"rep {rep_idx}\nn={len(members)}\nt={frame}", rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle(f"{POPULATION_SHORT_NAME}: group maps on {case_label}", fontsize=11)
    return fig


plot_case_label, plot_case_path = CASE_TO_PLOT
plot_movie = load_activation_movie(plot_case_path)
fig = plot_group_activation_maps(plot_movie, GROUPS_TO_PLOT, case_label=plot_case_label)
if SAVE_FIGURES:
    fig.savefig(FIG_DIR / f"rr100_group_activation_maps_{_safe_slug(plot_case_label)}.png", bbox_inches="tight")
plt.show()


# %% Step 10: center-pixel traces for the same groups
def plot_group_center_traces(movie: np.ndarray, groups: list[int], *, case_label: str, max_members: int = 8):
    reps_by_idx = {int(row["rep_idx"]): row for row in representatives.to_dict("records")}
    n = len(groups)
    n_cols = 2
    n_rows = int(math.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 2.4 * n_rows), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    for ax in axes:
        ax.set_axis_off()
    t = np.arange(movie.shape[0])
    for ax, rep_idx in zip(axes, groups):
        ax.set_axis_on()
        row = reps_by_idx[int(rep_idx)]
        members = np.asarray(row["members"], dtype=int)
        selected = int(row["selected_channel"])
        shown = list(dict.fromkeys([selected] + [int(ch) for ch in members if int(ch) != selected]))[:max_members]
        traces = center_pixel_traces(movie, shown)
        for trace, ch in zip(traces.T, shown):
            if int(ch) == selected:
                ax.plot(t, trace, color="black", lw=1.8, label=f"selected {ch}")
            else:
                ax.plot(t, trace, color="0.65", lw=0.8, alpha=0.55)
        centroid = center_pixel_traces(movie, members).mean(axis=1)
        ax.plot(t, centroid, color="tab:orange", lw=1.4, alpha=0.9, label="member mean")
        ax.set_title(f"rep {rep_idx}, n={len(members)}", fontsize=9)
        ax.set_xlabel("frame")
        ax.set_ylabel("center pixel rate")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"{POPULATION_SHORT_NAME}: center-pixel traces on {case_label}", fontsize=11)
    return fig


fig = plot_group_center_traces(plot_movie, GROUPS_TO_PLOT, case_label=plot_case_label)
if SAVE_FIGURES:
    fig.savefig(FIG_DIR / f"rr100_group_center_pixel_traces_{_safe_slug(plot_case_label)}.png", bbox_inches="tight")
plt.show()


# %% Step 11: sharp selected representatives and sharp singleton examples
def sharp_rep_rows(movie: np.ndarray, reps: pd.DataFrame, *, singleton_only: bool, n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_reps = reps.query("n_members == 1") if singleton_only else reps
    for row in candidate_reps.to_dict("records"):
        rep_idx = int(row["rep_idx"])
        ch = int(row["selected_channel"])
        frame = pick_peak_frame(movie, [ch])
        score = float(spatial_sharpness(movie[frame, [ch]])[0])
        rows.append(
            {
                "rep_idx": rep_idx,
                "channel": ch,
                "n_members": int(row["n_members"]),
                "frame": int(frame),
                "sharpness": score,
                "selected_ccnorm": float(row.get("selected_ccnorm", np.nan)),
            }
        )
    rows.sort(key=lambda r: (-float(r["sharpness"]), int(r["rep_idx"])))
    return rows[: int(n)]


def plot_sharp_gallery(movie: np.ndarray, rows: list[dict[str, Any]], *, title: str):
    n = len(rows)
    n_cols = min(8, max(1, n))
    n_rows = int(math.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.55 * n_cols, 1.65 * n_rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(n_rows, n_cols)
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, row in zip(axes.ravel(), rows):
        ch = int(row["channel"])
        frame = int(row["frame"])
        arr = movie[frame, ch]
        vmin, vmax = np.nanpercentile(arr, [1, 99.5])
        ax.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"rep {int(row['rep_idx'])}\nch {ch} t={frame}", fontsize=7)
        ax.set_axis_on()
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    return fig


sharp_singletons = sharp_rep_rows(plot_movie, representatives, singleton_only=True, n=N_SHARP_SINGLETONS)
sharp_reps = sharp_rep_rows(plot_movie, representatives, singleton_only=False, n=N_SHARP_REPS)
_write_csv(QC_DIR / f"rr100_sharp_singletons_{_safe_slug(plot_case_label)}.csv", sharp_singletons)
_write_csv(QC_DIR / f"rr100_sharp_representatives_{_safe_slug(plot_case_label)}.csv", sharp_reps)
display(_as_table(sharp_singletons).head(20).style.format(precision=3))

fig = plot_sharp_gallery(plot_movie, sharp_singletons, title=f"{POPULATION_SHORT_NAME}: sharp singletons on {plot_case_label}")
if SAVE_FIGURES:
    fig.savefig(FIG_DIR / f"rr100_sharp_singletons_{_safe_slug(plot_case_label)}.png", bbox_inches="tight")
plt.show()

fig = plot_sharp_gallery(plot_movie, sharp_reps, title=f"{POPULATION_SHORT_NAME}: sharp selected reps on {plot_case_label}")
if SAVE_FIGURES:
    fig.savefig(FIG_DIR / f"rr100_sharp_representatives_{_safe_slug(plot_case_label)}.png", bbox_inches="tight")
plt.show()


# %% Step 12: save a compact manifest of generated QC artifacts
manifest = {
    "population": RR100_VERSION,
    "qc_dir": str(QC_DIR),
    "fig_dir": str(FIG_DIR),
    "construction_cases": [case for case, _path in construction_cases],
    "heldout_cases": [case for case, _path in heldout_cases],
    "key_outputs": {
        "construction_corr": str(construction_corr_path),
        "heldout_corr": str(heldout_corr_path) if heldout_cases else None,
        "group_quality": str(group_quality_csv),
        "reconstruction": str(reconstruction_csv),
    },
}
(QC_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
display(manifest)
