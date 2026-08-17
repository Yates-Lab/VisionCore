#!/usr/bin/env python3
"""Short-fixation smoke test for the retinal-motion/passband mechanism.

The analysis deliberately avoids interpreting individual low temporal-frequency
FFT bins.  It uses native 240-Hz, 60-frame windows that fit the M66 temporal
stems, partitions temporal power into four broad bands, and compares a
frequency-domain passband prediction with the stems' measured linear
preactivation energy on locally rendered natural-image movies.
"""

from __future__ import annotations

import argparse
import base64
import gc
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.signal.windows import dpss
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_WINDOWS = ROOT / (
    "outputs/fixation_statistics_by_stimulus_all_sessions_after_review/"
    "backimage_image_structure_reviewed_v2_screenfiltered_yfix/"
    "backimage_image_fem_windows.csv"
)
CHECKPOINT = Path(
    "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/dekel240/"
    "D240M66a_m63e31_m64be63_readout_coord_teacher0p4_s201/"
    "analysis_candidates/epoch=031-endpoint.ckpt"
)
GALLERY = ROOT / "outputs/cache_fig4_original/fig4_coherence_gallery.npz"
SELECTED_PATCH = ROOT / (
    "outputs/dekel240_paper/m66_final_snapshot/unit_maps_m66/cache/selected_patch.npy"
)
DEFAULT_OUT = ROOT / (
    "outputs/dekel240_paper/m66_final_snapshot/fig4_new_ending/"
    "rucci_short_fixation_smoke"
)
DEFAULT_VIS = Path(
    "/home/jake/.codex/visualizations/2026/08/14/"
    "019ffe32-dd4b-7ab1-ba87-15e22155ada2/"
    "rucci-short-fixation-smoke.html"
)

RATE_HZ = 240.0
N_LAGS = 60
PPD = 37.50476617
SPATIAL_FFT = 64
TF_EDGES = np.asarray([0.0, 12.0, 32.0, 64.0, 120.0001])
TF_LABELS = ("slow\n0–12", "mid\n12–32", "fast\n32–64", "very fast\n64–120")
SF_EDGES = np.asarray([0.0, 2.0, 4.0, 8.0, 12.0001])
SF_LABELS = ("0–2", "2–4", "4–8", "8–12")
CONDITIONS = ("stable", "drift", "residual", "full")
CONDITION_LABELS = {
    "stable": "stabilized",
    "drift": "endpoint drift",
    "residual": "detrended residual",
    "full": "measured motion",
}
BRANCHES = ("base", "auxiliary", "residual")
BRANCH_LABELS = {"base": "main", "auxiliary": "auxiliary", "residual": "residual"}
COLORS = {
    "stable": "#777777",
    "drift": "#2F78B7",
    "residual": "#E28E2B",
    "full": "#C84C36",
    "base": "#2F78B7",
    "auxiliary": "#59A14F",
    "residual_branch": "#C84C36",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-fixations", type=int, default=48)
    parser.add_argument("--n-sessions", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--visualization", type=Path, default=DEFAULT_VIS)
    return parser.parse_args()


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def normalize_patch(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value, dtype=np.float64)
    lo, hi = np.nanpercentile(image, (0.5, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    # This is the scorer's input convention after uint-like standardization:
    # (255 * normalized_image - 127) / 255.
    return (np.clip((image - lo) / (hi - lo), 0.0, 1.0) - 127.0 / 255.0).astype(np.float32)


def load_patches() -> np.ndarray:
    with np.load(GALLERY, allow_pickle=True) as archive:
        patches = [normalize_patch(item) for item in np.asarray(archive["patches"])]
    selected = np.asarray(np.load(SELECTED_PATCH), dtype=np.float32)
    size = 151
    y0 = (selected.shape[0] - size) // 2
    x0 = (selected.shape[1] - size) // 2
    patches.append(normalize_patch(selected[y0 : y0 + size, x0 : x0 + size]))
    return np.stack(patches)


def select_epochs(n_fixations: int, n_sessions: int) -> pd.DataFrame:
    rows = pd.read_csv(SOURCE_WINDOWS)
    keys = ["session", "trial_idx", "epoch_start_local", "epoch_stop_local"]
    epochs = rows.drop_duplicates(keys).copy()
    epochs["epoch_samples"] = epochs["epoch_stop_local"] - epochs["epoch_start_local"]
    epochs = epochs.loc[epochs["epoch_samples"] >= N_LAGS].copy()
    sessions = np.asarray(sorted(epochs["session"].astype(str).unique()))
    if len(sessions) > int(n_sessions):
        chosen_index = np.linspace(0, len(sessions) - 1, int(n_sessions)).round().astype(int)
        sessions = sessions[chosen_index]
    per_session = max(1, int(np.ceil(int(n_fixations) / len(sessions))))
    selected = []
    for session in sessions:
        sub = epochs.loc[epochs["session"].astype(str).eq(str(session))].sort_values(
            ["epoch_samples", "trial_idx", "epoch_start_local"]
        )
        index = np.linspace(0, len(sub) - 1, min(per_session, len(sub))).round().astype(int)
        selected.append(sub.iloc[np.unique(index)])
    result = pd.concat(selected, ignore_index=True)
    if len(result) > int(n_fixations):
        keep = np.linspace(0, len(result) - 1, int(n_fixations)).round().astype(int)
        result = result.iloc[np.unique(keep)].copy()
    return result.reset_index(drop=True)


def load_trace_conditions(epochs: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    from models.data.datasets import DictDataset

    condition_rows = {name: [] for name in CONDITIONS}
    provenance = []
    for session, session_rows in epochs.groupby("session", sort=True):
        dataset_path = Path(
            f"/mnt/ssd/YatesMarmoV1/processed/{session}/datasets/backimage.dset"
        )
        dataset = DictDataset.load(str(dataset_path))
        eyepos = as_numpy(dataset["eyepos"]).astype(np.float64)
        t_bins = as_numpy(dataset["t_bins"]).astype(np.float64)
        trial_inds = as_numpy(dataset.covariates["trial_inds"]).reshape(-1).astype(int)
        dt = float(np.median(np.diff(t_bins)))
        if not np.isclose(dt, 1.0 / RATE_HZ, rtol=0, atol=2e-5):
            raise RuntimeError(f"{session} is not native 240 Hz: dt={dt}.")
        for row in session_rows.itertuples(index=False):
            trial_global_start = int(row.global_start) - int(row.local_start)
            epoch_start = trial_global_start + int(row.epoch_start_local)
            epoch_stop = trial_global_start + int(row.epoch_stop_local)
            start = epoch_start + (epoch_stop - epoch_start - N_LAGS) // 2
            stop = start + N_LAGS
            if stop > len(eyepos) or not np.all(trial_inds[start:stop] == int(row.trial_idx)):
                raise RuntimeError(f"Invalid epoch/global-index mapping for {session}, trial {row.trial_idx}.")
            full = eyepos[start:stop].copy()
            full -= full[0]
            alpha = np.linspace(0.0, 1.0, N_LAGS)[:, None]
            drift = alpha * full[-1]
            residual = full - drift
            condition_rows["stable"].append(np.zeros_like(full, dtype=np.float32))
            condition_rows["drift"].append(drift.astype(np.float32))
            condition_rows["residual"].append(residual.astype(np.float32))
            condition_rows["full"].append(full.astype(np.float32))
            provenance.append(
                {
                    "session": str(session),
                    "trial_idx": int(row.trial_idx),
                    "epoch_start_global": int(epoch_start),
                    "epoch_stop_global": int(epoch_stop),
                    "window_start_global": int(start),
                    "window_stop_global": int(stop),
                    "epoch_samples": int(epoch_stop - epoch_start),
                    "epoch_duration_s_native240": float((epoch_stop - epoch_start) / RATE_HZ),
                    "window_duration_s_native240": float(N_LAGS / RATE_HZ),
                }
            )
        del dataset, eyepos, t_bins, trial_inds
        gc.collect()
    return (
        {name: np.asarray(values, dtype=np.float32) for name, values in condition_rows.items()},
        pd.DataFrame(provenance),
    )


def load_kernels() -> dict[str, np.ndarray]:
    from eval.load_twin import load_twin

    model, _ = load_twin(CHECKPOINT.resolve(), device="cpu", verbose=False)
    cores = {
        "base": model.model.convnet,
        "auxiliary": model.model.auxiliary_convnet,
        "residual": model.model.residual_convnet,
    }
    kernels = {
        name: core.effective_temporal_weight().detach().float().cpu().numpy()[:, 0]
        for name, core in cores.items()
    }
    del model
    gc.collect()
    for name, value in kernels.items():
        if value.shape[1:] != (N_LAGS, 7, 7):
            raise ValueError(f"Unexpected {name} kernel shape {value.shape}.")
    return kernels


def spatial_centers() -> np.ndarray:
    positions = np.asarray([40.0, 75.0, 110.0])
    yy, xx = np.meshgrid(positions, positions, indexing="ij")
    return np.column_stack((yy.ravel(), xx.ravel()))


def render_local_movie(patch: np.ndarray, trace: np.ndarray, centers: np.ndarray) -> np.ndarray:
    offset = np.arange(-3.0, 4.0)
    dy, dx = np.meshgrid(offset, offset, indexing="ij")
    output = np.empty((len(centers), N_LAGS, 7, 7), dtype=np.float32)
    for time_index, (eye_x, eye_y) in enumerate(np.asarray(trace)):
        sample_y = centers[:, 0, None, None] + dy[None] - float(eye_y) * PPD
        sample_x = centers[:, 1, None, None] + dx[None] + float(eye_x) * PPD
        output[:, time_index] = map_coordinates(
            patch,
            (sample_y, sample_x),
            order=1,
            mode="reflect",
            prefilter=False,
        )
    return output


def measured_linear_drive(
    patches: np.ndarray,
    traces: dict[str, np.ndarray],
    kernels: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, dict[str, float]]]]:
    centers = spatial_centers()
    n_fixations = int(next(iter(traces.values())).shape[0])
    drive = {
        name: np.zeros((n_fixations, len(CONDITIONS)), dtype=np.float64)
        for name in BRANCHES
    }
    for fixation in range(n_fixations):
        for condition_index, condition in enumerate(CONDITIONS):
            branch_sum = {name: 0.0 for name in BRANCHES}
            branch_count = {name: 0 for name in BRANCHES}
            for patch in patches:
                movie = render_local_movie(patch, traces[condition][fixation], centers)
                # M66 lag index zero is the newest frame.
                lagged = movie[:, ::-1]
                for name in BRANCHES:
                    values = np.einsum(
                        "ctyx,ptyx->pc",
                        kernels[name].astype(np.float64),
                        lagged.astype(np.float64),
                        optimize=True,
                    )
                    branch_sum[name] += float(np.sum(values**2))
                    branch_count[name] += int(values.size)
            for name in BRANCHES:
                drive[name][fixation, condition_index] = branch_sum[name] / branch_count[name]
        if (fixation + 1) % 8 == 0:
            print(f"measured first-layer drive {fixation + 1}/{n_fixations}", flush=True)

    rng = np.random.default_rng(20260816)
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for name in BRANCHES:
        stable = np.maximum(drive[name][:, 0], 1e-30)
        summary[name] = {}
        for condition_index, condition in enumerate(CONDITIONS):
            condition_drive = drive[name][:, condition_index]
            point = float(np.mean(condition_drive) / max(float(np.mean(stable)), 1e-30))
            boot = np.empty(2000, dtype=np.float64)
            for index in range(len(boot)):
                sample = rng.integers(0, len(stable), len(stable))
                boot[index] = float(
                    np.mean(condition_drive[sample])
                    / max(float(np.mean(stable[sample])), 1e-30)
                )
            summary[name][condition] = {
                "mean_ratio_to_stable": point,
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
            }
    return drive, summary


def image_power(patches: np.ndarray) -> np.ndarray:
    centers = spatial_centers()
    window = np.outer(np.hanning(SPATIAL_FFT), np.hanning(SPATIAL_FFT))
    values = []
    half = SPATIAL_FFT // 2
    for patch in patches:
        for center_y, center_x in centers:
            y0 = int(round(center_y)) - half
            x0 = int(round(center_x)) - half
            crop = np.asarray(patch[y0 : y0 + SPATIAL_FFT, x0 : x0 + SPATIAL_FFT], dtype=np.float64)
            if crop.shape != (SPATIAL_FFT, SPATIAL_FFT):
                continue
            transformed = np.fft.fft2(crop * window, norm="ortho")
            values.append(np.abs(transformed) ** 2)
    return np.mean(values, axis=0)


def band_sum(value: np.ndarray, sf: np.ndarray, tf_abs: np.ndarray) -> np.ndarray:
    result = np.zeros((len(TF_EDGES) - 1, len(SF_EDGES) - 1), dtype=np.float64)
    for ti in range(len(TF_EDGES) - 1):
        tf_mask = (tf_abs >= TF_EDGES[ti]) & (tf_abs < TF_EDGES[ti + 1])
        for si in range(len(SF_EDGES) - 1):
            sf_mask = (sf >= SF_EDGES[si]) & (sf < SF_EDGES[si + 1])
            result[ti, si] = float(np.sum(value[tf_mask][:, sf_mask]))
    return result


def spectral_analysis(
    patches: np.ndarray,
    traces: dict[str, np.ndarray],
    kernels: dict[str, np.ndarray],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, float]],
    np.ndarray,
    np.ndarray,
]:
    spatial_power = image_power(patches)
    spatial_axis = np.fft.fftfreq(SPATIAL_FFT, d=1.0 / PPD)
    fy, fx = np.meshgrid(spatial_axis, spatial_axis, indexing="ij")
    sf = np.sqrt(fx**2 + fy**2)
    temporal = np.fft.fftfreq(N_LAGS, d=1.0 / RATE_HZ)
    tf_abs = np.abs(temporal)
    tapers = dpss(N_LAGS, NW=1.5, Kmax=2, sym=False)

    n_fixations = int(next(iter(traces.values())).shape[0])
    phase_power_by_condition: dict[str, np.ndarray] = {}
    phase_power_by_fixation: dict[str, list[np.ndarray]] = {name: [] for name in CONDITIONS}
    for condition in CONDITIONS:
        aggregate = np.zeros((N_LAGS, SPATIAL_FFT, SPATIAL_FFT), dtype=np.float64)
        for fixation in range(n_fixations):
            eye = np.asarray(traces[condition][fixation], dtype=np.float64)
            dot = eye[:, 0, None, None] * fx[None] + eye[:, 1, None, None] * fy[None]
            phase = np.exp(-2j * np.pi * dot)
            power = np.zeros_like(aggregate)
            for taper in tapers:
                transformed = np.fft.fft(phase * taper[:, None, None], axis=0, norm="ortho")
                power += np.abs(transformed) ** 2
            power /= len(tapers)
            aggregate += power
            phase_power_by_fixation[condition].append(power)
        phase_power_by_condition[condition] = aggregate / n_fixations

    retinal_power = {
        condition: phase_power_by_condition[condition] * spatial_power[None]
        for condition in CONDITIONS
    }
    retinal_bands = {
        condition: band_sum(value, sf, tf_abs) for condition, value in retinal_power.items()
    }

    filter_power: dict[str, np.ndarray] = {}
    for name in BRANCHES:
        transformed = np.fft.fftn(
            kernels[name],
            s=(N_LAGS, SPATIAL_FFT, SPATIAL_FFT),
            axes=(-3, -2, -1),
            norm="ortho",
        )
        filter_power[name] = np.sum(np.abs(transformed) ** 2, axis=0)

    matched_bands: dict[str, dict[str, np.ndarray]] = {name: {} for name in BRANCHES}
    predicted_ratio: dict[str, dict[str, float]] = {name: {} for name in BRANCHES}
    for name in BRANCHES:
        stable_total = float(np.sum(retinal_power["stable"] * filter_power[name]))
        for condition in CONDITIONS:
            matched = retinal_power[condition] * filter_power[name]
            matched_bands[name][condition] = band_sum(matched, sf, tf_abs)
            predicted_ratio[name][condition] = float(np.sum(matched) / max(stable_total, 1e-30))
    return retinal_bands, matched_bands, predicted_ratio, sf, tf_abs


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_band_heatmap(
    axis: plt.Axes,
    value: np.ndarray,
    *,
    title: str,
    scale: float,
    show_y: bool,
) -> None:
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-scale, vmax=scale)
    image = axis.imshow(value, origin="lower", aspect="auto", cmap="RdBu_r", norm=norm)
    axis.set_xticks(np.arange(len(SF_LABELS)), SF_LABELS)
    axis.set_yticks(np.arange(len(TF_LABELS)), TF_LABELS if show_y else [""] * len(TF_LABELS))
    axis.set_xlabel("SF (cycles/deg)")
    if show_y:
        axis.set_ylabel("broad TF band (Hz)")
    axis.set_title(title, pad=4)
    for row in range(value.shape[0]):
        for col in range(value.shape[1]):
            number = float(value[row, col])
            if abs(number) >= 0.08 * scale:
                axis.text(col, row, f"{number:+.0f}", ha="center", va="center", fontsize=5.8)
    return image


def render(
    out_dir: Path,
    traces: dict[str, np.ndarray],
    provenance: pd.DataFrame,
    retinal_bands: dict[str, np.ndarray],
    matched_bands: dict[str, dict[str, np.ndarray]],
    predicted_ratio: dict[str, dict[str, float]],
    measured_summary: dict[str, dict[str, dict[str, float]]],
) -> tuple[Path, dict[str, float]]:
    configure()
    stable_retinal_total = max(float(np.sum(retinal_bands["stable"])), 1e-30)
    retinal_delta = 100.0 * (retinal_bands["full"] - retinal_bands["stable"]) / stable_retinal_total
    matched_delta = {}
    for name in BRANCHES:
        stable = max(float(np.sum(matched_bands[name]["stable"])), 1e-30)
        matched_delta[name] = 100.0 * (
            matched_bands[name]["full"] - matched_bands[name]["stable"]
        ) / stable
    heat_scale = max(
        float(np.max(np.abs(retinal_delta))),
        *(float(np.max(np.abs(value))) for value in matched_delta.values()),
        1e-6,
    )

    figure = plt.figure(figsize=(10.2, 6.2))
    grid = figure.add_gridspec(2, 4, height_ratios=(0.92, 1.0), hspace=0.53, wspace=0.48)

    trajectory_axis = figure.add_subplot(grid[0, 0])
    path_lengths = np.sum(np.linalg.norm(np.diff(traces["full"], axis=1), axis=2), axis=1)
    example = int(np.argmin(np.abs(path_lengths - np.median(path_lengths))))
    trajectory_axis.plot(
        traces["full"][example, :, 0] * 60,
        traces["full"][example, :, 1] * 60,
        color=COLORS["full"],
        lw=1.4,
        label="full",
    )
    trajectory_axis.plot(
        traces["drift"][example, :, 0] * 60,
        traces["drift"][example, :, 1] * 60,
        color=COLORS["drift"],
        lw=1.3,
        label="endpoint drift",
    )
    trajectory_axis.plot(
        traces["residual"][example, :, 0] * 60,
        traces["residual"][example, :, 1] * 60,
        color=COLORS["residual"],
        lw=1.1,
        label="residual",
    )
    trajectory_axis.scatter([0], [0], s=10, color="#222222", zorder=4)
    trajectory_axis.set_aspect("equal", adjustable="datalim")
    trajectory_axis.set_xlabel("horizontal displacement (arcmin)")
    trajectory_axis.set_ylabel("vertical displacement (arcmin)")
    trajectory_axis.set_title("A  Exact within-fixation decomposition")
    trajectory_axis.legend(frameon=False, fontsize=6.4)

    retinal_axis = figure.add_subplot(grid[0, 1])
    image = draw_band_heatmap(
        retinal_axis,
        retinal_delta,
        title="B  Motion-added retinal power",
        scale=heat_scale,
        show_y=True,
    )

    components_axis = figure.add_subplot(grid[0, 2])
    x = np.arange(len(TF_LABELS))
    width = 0.24
    for offset, condition in enumerate(("drift", "residual", "full")):
        value = retinal_bands[condition].sum(axis=1)
        stable = retinal_bands["stable"].sum(axis=1)
        delta = 100.0 * (value - stable) / stable_retinal_total
        components_axis.bar(
            x + (offset - 1) * width,
            delta,
            width=width,
            color=COLORS[condition],
            label=CONDITION_LABELS[condition],
        )
    components_axis.axhline(0, color="#777777", lw=0.7)
    components_axis.set_xticks(x, TF_LABELS)
    components_axis.set_ylabel("retinal power change\n(% stable total)")
    components_axis.set_title("C  Drift and residual redistribute power")
    components_axis.legend(frameon=False, fontsize=6.3)

    validation_axis = figure.add_subplot(grid[0, 3])
    pred = []
    obs = []
    for name in BRANCHES:
        color_key = "residual_branch" if name == "residual" else name
        for condition, marker in zip(("drift", "residual", "full"), ("o", "s", "^")):
            px = float(predicted_ratio[name][condition])
            oy = float(measured_summary[name][condition]["mean_ratio_to_stable"])
            pred.append(px)
            obs.append(oy)
            validation_axis.scatter(
                px,
                oy,
                s=28,
                marker=marker,
                color=COLORS[color_key],
                edgecolor="white",
                linewidth=0.45,
            )
    low = min(min(pred), min(obs), 0.95)
    high = max(max(pred), max(obs), 1.05)
    margin = 0.07 * (high - low)
    validation_axis.plot([low - margin, high + margin], [low - margin, high + margin], color="#777777", lw=0.8)
    validation_axis.set_xlim(low - margin, high + margin)
    validation_axis.set_ylim(low - margin, high + margin)
    correlation = float(pearsonr(pred, obs).statistic)
    validation_axis.set_xlabel("broad-band spectral prediction\n(relative to stable)")
    validation_axis.set_ylabel("measured linear drive\n(relative to stable)")
    validation_axis.set_title(f"D  Prediction vs convolution  r={correlation:.2f}")
    validation_axis.text(
        0.03,
        0.97,
        "○ drift   □ residual   △ full\nblue main · green auxiliary · red residual",
        transform=validation_axis.transAxes,
        va="top",
        fontsize=6.2,
    )

    for column, name in enumerate(BRANCHES):
        axis = figure.add_subplot(grid[1, column])
        draw_band_heatmap(
            axis,
            matched_delta[name],
            title=f"{chr(69 + column)}  {BRANCH_LABELS[name]} stem: matched power",
            scale=heat_scale,
            show_y=column == 0,
        )

    drive_axis = figure.add_subplot(grid[1, 3])
    condition_x = np.arange(3)
    width = 0.24
    for branch_index, name in enumerate(BRANCHES):
        color_key = "residual_branch" if name == "residual" else name
        values = []
        low_error = []
        high_error = []
        for condition in ("drift", "residual", "full"):
            item = measured_summary[name][condition]
            values.append(item["mean_ratio_to_stable"])
            low_error.append(item["mean_ratio_to_stable"] - item["ci95_low"])
            high_error.append(item["ci95_high"] - item["mean_ratio_to_stable"])
        drive_axis.bar(
            condition_x + (branch_index - 1) * width,
            values,
            yerr=np.asarray([low_error, high_error]),
            capsize=2,
            width=width,
            color=COLORS[color_key],
            label=BRANCH_LABELS[name],
        )
    drive_axis.axhline(1.0, color="#777777", lw=0.7)
    drive_axis.set_xticks(condition_x, ("drift", "residual", "full"))
    drive_axis.set_ylabel("measured preactivation power\n(relative to stabilized)")
    drive_axis.set_title("H  Actual M66 first-layer drive")
    drive_axis.legend(frameon=False, fontsize=6.4)

    n_fixations = int(len(provenance))
    median_duration = float(provenance["epoch_duration_s_native240"].median())
    figure.suptitle(
        "Short-fixation Rucci smoke test: motion moves image power into M66 passbands",
        x=0.02,
        y=0.99,
        ha="left",
        fontsize=11,
        fontweight="semibold",
    )
    figure.text(
        0.985,
        0.015,
        f"{n_fixations} clean epochs ({provenance.session.nunique()} sessions; median {median_duration:.2f} s), "
        "5 natural-image patches × 9 positions; 60 native samples at 240 Hz; TF values are integrated bands",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#555555",
    )
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.12, top=0.90)
    path = out_dir / "rucci_short_fixation_smoke.png"
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(out_dir / f"rucci_short_fixation_smoke.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path, {"prediction_measured_pearson_r": correlation}


def write_visualization(path: Path, png_path: Path, n_fixations: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = base64.b64encode(png_path.read_bytes()).decode("ascii")
    path.write_text(
        f'''<div id="rucci-short-fixation-smoke" style="width:100%;color:var(--foreground)">
  <figure style="margin:0;display:grid;gap:6px">
    <figcaption style="font-weight:500">Short-fixation retinal transfer and M66 passband drive</figcaption>
    <img src="data:image/png;base64,{data}" alt="Smoke-test figure using {n_fixations} native 240 hertz clean fixation epochs. It shows drift and residual decomposition, broad spatial-temporal power bands, M66 stem-matched power, and validation against measured first-layer convolution energy." style="width:100%;height:auto;border:1px solid var(--border);border-radius:8px;background:var(--card)">
  </figure>
</div>
''',
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    epochs = select_epochs(int(args.n_fixations), int(args.n_sessions))
    traces, provenance = load_trace_conditions(epochs)
    patches = load_patches()
    kernels = load_kernels()
    measured, measured_summary = measured_linear_drive(patches, traces, kernels)
    retinal_bands, matched_bands, predicted_ratio, _sf, _tf = spectral_analysis(
        patches, traces, kernels
    )
    png, figure_metrics = render(
        args.out_dir,
        traces,
        provenance,
        retinal_bands,
        matched_bands,
        predicted_ratio,
        measured_summary,
    )
    provenance.to_csv(args.out_dir / "fixation_provenance.csv", index=False)
    np.savez_compressed(
        args.out_dir / "rucci_short_fixation_smoke_arrays.npz",
        conditions=np.asarray(CONDITIONS),
        branches=np.asarray(BRANCHES),
        sf_edges=SF_EDGES,
        tf_edges=TF_EDGES,
        **{f"trace_{name}": value for name, value in traces.items()},
        **{f"retinal_band_{name}": value for name, value in retinal_bands.items()},
        **{
            f"matched_band_{branch}_{condition}": matched_bands[branch][condition]
            for branch in BRANCHES
            for condition in CONDITIONS
        },
        **{f"measured_drive_{name}": value for name, value in measured.items()},
    )
    report = {
        "analysis": "short-fixation broad-band retinal transfer and measured M66 first-layer drive",
        "checkpoint": str(CHECKPOINT),
        "n_fixations": int(len(provenance)),
        "n_sessions": int(provenance.session.nunique()),
        "sessions": sorted(provenance.session.unique().tolist()),
        "n_image_patches": int(len(patches)),
        "n_spatial_positions_per_patch": int(len(spatial_centers())),
        "sampling_rate_hz": RATE_HZ,
        "model_history_samples": N_LAGS,
        "model_history_s": N_LAGS / RATE_HZ,
        "epoch_duration_s_native240": {
            "min": float(provenance.epoch_duration_s_native240.min()),
            "median": float(provenance.epoch_duration_s_native240.median()),
            "max": float(provenance.epoch_duration_s_native240.max()),
        },
        "temporal_bands_hz": TF_EDGES.tolist(),
        "spatial_bands_cpd": SF_EDGES.tolist(),
        "detrending": "endpoint-line drift plus exact zero-endpoint residual; full = drift + residual",
        "spectral_estimator": "two DPSS tapers, NW=1.5, 60 native samples; power integrated over predefined broad bands",
        "measured_drive": "mean squared effective-temporal-kernel preactivation before bias, normalization, and nonlinearity",
        "measured_drive_ratio_to_stable": measured_summary,
        "spectral_predicted_drive_ratio_to_stable": predicted_ratio,
        **figure_metrics,
        "timing_audit": "all source datasets were read from t_bins at 240 Hz; CSV duration metadata are not used",
        "claim_boundary": "first-layer linear drive smoke test; no unit-specific Jacobian or nonlinear SSI attribution is claimed",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    write_visualization(args.visualization, png, int(len(provenance)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
