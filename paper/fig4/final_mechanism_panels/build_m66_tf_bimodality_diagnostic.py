#!/usr/bin/env python3
"""Diagnose the apparent bimodality of M66 output-unit temporal tuning.

The standard grating probe contains only five non-zero temporal frequencies.
This script contrasts the hard winning probe frequency with a continuous
response-weighted temporal center, and shows representative unit curves.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr


ROOT = Path(__file__).resolve().parents[3]
TUNING = ROOT / (
    "outputs/dekel240_paper/final/fig4_real_trace/frequency_tuning_m66/"
    "frequency_tuning_grouped.csv"
)
EFFECTS = ROOT / (
    "outputs/dekel240_paper/m66_final_snapshot/fig4_new_ending/"
    "rucci_unit_specific/per_unit_motion_effects.csv"
)
DEFAULT_OUT = ROOT / (
    "outputs/dekel240_paper/m66_final_snapshot/fig4_new_ending/"
    "tf_bimodality_diagnostic"
)
DEFAULT_VIS = Path(
    "/home/jake/.codex/visualizations/2026/08/14/"
    "019ffe32-dd4b-7ab1-ba87-15e22155ada2/"
    "m66-temporal-bimodality.html"
)
M66_CHECKPOINT = Path(
    "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/dekel240/"
    "D240M66a_m63e31_m64be63_readout_coord_teacher0p4_s201/"
    "analysis_candidates/epoch=031-endpoint.ckpt"
)

EXAMPLE_UNITS = [55, 45, 60, 3, 17, 92, 63, 89]
CLASS_COLORS = {
    0.2: "#2A788E",
    0.8: "#4AA56D",
    3.2: "#7A5AA6",
    12.8: "#E1783D",
    47.2: "#B84A42",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--visualization", type=Path, default=DEFAULT_VIS)
    return parser.parse_args()


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_population() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    raw = pd.read_csv(TUNING)
    dynamic = raw.loc[raw.temporal_hz.gt(0.0)]
    curves = (
        dynamic.groupby(["unit_index", "temporal_hz"], sort=True)
        .response_amp_rms.max()
        .unstack()
        .sort_index(axis=1)
    )
    effects = pd.read_csv(EFFECTS).set_index("unit_index")
    active_ids = effects.index[effects.active.astype(bool)]
    curves = curves.loc[active_ids]
    # Three otherwise-active output units have grating amplitudes at numerical
    # underflow (~1e-37).  They have no estimable TF tuning and should not be
    # silently assigned to the first bin by argmax.
    curves = curves.loc[curves.max(axis=1).gt(1e-10)]
    tf = curves.columns.to_numpy(dtype=float)
    amplitude = curves.to_numpy(dtype=float)
    normalized = amplitude / amplitude.max(axis=1, keepdims=True)
    weights = normalized / np.maximum(normalized.sum(axis=1, keepdims=True), 1e-30)
    hard_peak = tf[np.argmax(normalized, axis=1)]
    continuous_center = 10.0 ** np.sum(weights * np.log10(tf)[None, :], axis=1)
    hard_index = np.argmax(normalized, axis=1)
    interpolated_peak = np.empty(len(normalized), dtype=float)
    interpolated_peak_censored = np.zeros(len(normalized), dtype=bool)
    log_tf = np.log2(tf)
    for row_index, (curve, peak_index) in enumerate(zip(normalized, hard_index)):
        if peak_index == 0:
            interpolated_peak[row_index] = tf[0]
            interpolated_peak_censored[row_index] = True
            continue
        if peak_index == len(tf) - 1:
            interpolated_peak[row_index] = tf[-1]
            interpolated_peak_censored[row_index] = True
            continue
        coefficients = np.polyfit(
            log_tf[peak_index - 1 : peak_index + 2],
            curve[peak_index - 1 : peak_index + 2],
            2,
        )
        if coefficients[0] < 0:
            vertex = -coefficients[1] / (2.0 * coefficients[0])
        else:
            vertex = log_tf[peak_index]
        vertex = np.clip(vertex, log_tf[peak_index - 1], log_tf[peak_index + 1])
        interpolated_peak[row_index] = 2.0**vertex
    sorted_response = np.sort(normalized, axis=1)
    winner_margin = sorted_response[:, -1] - sorted_response[:, -2]
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-30)), axis=1) / np.log(
        len(tf)
    )

    population = effects.loc[curves.index].copy()
    population["hard_peak_tf_hz"] = hard_peak
    population["interpolated_peak_tf_hz"] = interpolated_peak
    population["interpolated_peak_censored"] = interpolated_peak_censored
    population["continuous_tf_center_hz"] = continuous_center
    population["winner_margin"] = winner_margin
    population["normalized_entropy"] = entropy
    return population, tf, normalized


def example_notes(unit: int) -> str:
    return {
        55: "genuinely low-pass",
        45: "broad; low wins",
        60: "low/high near-tie",
        3: "rare 0.8-Hz winner",
        17: "mid-frequency winner",
        92: "broad; high wins",
        63: "high-frequency biased",
        89: "strong high-frequency bias",
    }[unit]


def render_examples(
    out_dir: Path, population: pd.DataFrame, tf: np.ndarray, normalized: np.ndarray
) -> Path:
    configure()
    response = pd.DataFrame(normalized, index=population.index, columns=tf)
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.25), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.82, bottom=0.12, wspace=0.24, hspace=0.53)
    fig.suptitle(
        "Individual M66 units are usually broad—not two-peaked",
        x=0.065,
        y=0.965,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.905,
        "Each curve is the unit's maximum grating response across spatial frequency and orientation.\n"
        "The star is the hard winner; the dashed line is a local-quadratic peak estimate in log frequency.",
        ha="left",
        va="top",
        fontsize=10.2,
        color="#40464D",
    )

    for ax, unit in zip(axes.flat, EXAMPLE_UNITS):
        row = population.loc[unit]
        y = response.loc[unit].to_numpy(dtype=float)
        peak = float(row.hard_peak_tf_hz)
        interpolated_peak = float(row.interpolated_peak_tf_hz)
        center = float(row.continuous_tf_center_hz)
        gain = float(row.ssi_change_percent)
        color = CLASS_COLORS[peak]
        ax.plot(tf, y, color=color, lw=2.2, marker="o", ms=4.6, zorder=3)
        peak_index = int(np.argmax(y))
        ax.scatter(
            [tf[peak_index]], [y[peak_index]], marker="*", s=105, color=color,
            edgecolor="white", linewidth=0.8, zorder=5,
        )
        ax.axvline(interpolated_peak, color="#20262B", ls=(0, (3, 2)), lw=1.15, alpha=0.8)
        ax.fill_between(tf, 0, y, color=color, alpha=0.09)
        ax.set_xscale("log")
        ax.set_xlim(0.14, 66)
        ax.set_ylim(-0.03, 1.09)
        ax.set_xticks(tf)
        ax.set_xticklabels(["0.2", "0.8", "3.2", "12.8", "47"])
        ax.set_yticks([0, 0.5, 1.0])
        ax.grid(axis="y", color="#E1E5E8", lw=0.75)
        ax.set_title(f"u{unit:03d} · {example_notes(unit)}", loc="left", fontweight="bold")
        ax.text(
            0.02,
            0.05,
            f"interpolated peak {'≤0.20' if row.interpolated_peak_censored else f'{interpolated_peak:.2f}'} Hz\n"
            f"center {center:.2f} Hz   "
            f"motion ΔSSI {gain:+.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.2,
            color="#31373C",
        )

    for ax in axes[:, 0]:
        ax.set_ylabel("normalized response")
    for ax in axes[1, :]:
        ax.set_xlabel("temporal frequency (Hz)")
    fig.text(
        0.985,
        0.025,
        "M66 standard grating probe · active output units",
        ha="right",
        color="#687078",
        fontsize=8.2,
    )
    path = out_dir / "m66_example_temporal_tuning_curves.png"
    fig.savefig(path, dpi=190, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return path


def render_population(
    out_dir: Path, population: pd.DataFrame, tf: np.ndarray, normalized: np.ndarray
) -> tuple[Path, dict[str, float]]:
    configure()
    peak = population.hard_peak_tf_hz.to_numpy(dtype=float)
    interpolated_peak = population.interpolated_peak_tf_hz.to_numpy(dtype=float)
    censored = population.interpolated_peak_censored.to_numpy(dtype=bool)
    center = population.continuous_tf_center_hz.to_numpy(dtype=float)
    margin = population.winner_margin.to_numpy(dtype=float)
    gain = population.ssi_change_percent.to_numpy(dtype=float)
    counts = pd.Series(peak).value_counts().reindex(tf, fill_value=0)
    rho, p_value = spearmanr(center, gain)

    fig = plt.figure(figsize=(13.2, 7.2), facecolor="white")
    grid = fig.add_gridspec(
        2, 2, left=0.07, right=0.975, top=0.82, bottom=0.105,
        wspace=0.27, hspace=0.42, width_ratios=(0.91, 1.09),
    )
    fig.suptitle(
        "Why M66's preferred temporal frequency looks bimodal",
        x=0.07,
        y=0.965,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.905,
        "The split is partly real: a lower-bound low-pass group and a broad band-pass group. "
        "Five-bin argmax exaggerates their separation.",
        ha="left",
        va="top",
        fontsize=10.2,
        color="#40464D",
    )

    ax = fig.add_subplot(grid[0, 0])
    colors = [CLASS_COLORS[float(value)] for value in tf]
    bars = ax.bar(np.arange(len(tf)), counts.to_numpy(), color=colors, width=0.72)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 1.3, str(int(count)), ha="center", fontsize=9)
    ax.set_xticks(np.arange(len(tf)), ["0.2", "0.8", "3.2", "12.8", "47.2"])
    ax.set_xlabel("winning probe TF (Hz)")
    ax.set_ylabel("number of active units")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.set_title("A  Hard winner: apparently bimodal", loc="left")
    ax.text(
        0.98,
        0.95,
        f"{100*np.mean(margin < 0.30):.0f}% are within 30%\nof the runner-up",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#50575D",
    )

    ax = fig.add_subplot(grid[0, 1])
    interior = interpolated_peak[~censored]
    bins = np.geomspace(0.35, 14.0, 15)
    ax.hist(interior, bins=bins, color="#D88755", alpha=0.7, edgecolor="white")
    ax.bar(
        0.2,
        int(np.sum(censored)),
        width=0.07,
        color="#2A788E",
        edgecolor="white",
        align="center",
    )
    ax.text(0.2, int(np.sum(censored)) + 1.0, f"{int(np.sum(censored))}\ncensored", ha="center", va="bottom", fontsize=8.2)
    ax.set_xscale("log")
    ax.set_xlim(0.14, 15.0)
    ax.set_xticks([0.2, 0.5, 1, 2, 4, 8, 12], ["≤0.2", "0.5", "1", "2", "4", "8", "12"])
    ax.set_ylim(0, max(58, int(np.sum(censored)) * 1.12))
    ax.set_xlabel("local-quadratic peak estimate (Hz)")
    ax.set_ylabel("number of tuned units")
    ax.set_title("B  Interpolation resolves the high branch", loc="left")
    ax.text(
        0.98,
        0.94,
        "Low-pass peaks cannot be\nlocalized below the 0.2-Hz boundary",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        color="#50575D",
    )

    ax = fig.add_subplot(grid[1, 0])
    log_center = np.log10(center)
    bins = np.geomspace(0.5, 7.0, 15)
    ax.hist(center, bins=bins, color="#6C8FB4", alpha=0.45, edgecolor="white")
    x = np.geomspace(0.5, 7.0, 500)
    kde = gaussian_kde(log_center)
    density = kde(np.log10(x))
    density = density / density.max() * 15.5
    ax.plot(x, density, color="#235A85", lw=2.3)
    median_center = float(np.median(center))
    ax.axvline(median_center, color="#20262B", ls=(0, (4, 2)), lw=1.25)
    ax.text(median_center * 1.05, 16.2, f"median {median_center:.2f} Hz", va="top", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 7.0)
    ax.set_xticks([0.5, 1, 2, 4, 7], ["0.5", "1", "2", "4", "7"])
    ax.set_ylim(0, 17)
    ax.set_xlabel("response-weighted TF center (Hz)")
    ax.set_ylabel("units / scaled density")
    ax.set_title("C  Overall response mass forms a continuum", loc="left")
    ax.text(
        0.98,
        0.94,
        f"center vs motion gain\nSpearman ρ = {rho:.3f}, p = {p_value:.2g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.3,
        color="#50575D",
    )

    ax = fig.add_subplot(grid[1, 1])
    order = np.argsort(center)
    cmap = LinearSegmentedColormap.from_list("response", ["#F4F6F7", "#7BB6BC", "#133E5A"])
    image = ax.imshow(normalized[order], aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(tf)), ["0.2", "0.8", "3.2", "12.8", "47.2"])
    ax.set_xlabel("probe TF (Hz)")
    ax.set_ylabel("units, sorted by continuous center")
    ax.set_yticks([0, len(order) // 2, len(order) - 1], [f"{center[order[0]]:.2f}", f"{center[order[len(order)//2]]:.2f}", f"{center[order[-1]]:.2f} Hz"])
    ax.set_title("D  Broad curves shift gradually across units", loc="left")
    cbar = fig.colorbar(image, ax=ax, fraction=0.042, pad=0.025)
    cbar.set_label("normalized response")

    fig.text(
        0.975,
        0.025,
        f"Local quadratic uses the winning point and its two log-TF neighbors · n={len(population)} active, grating-responsive units",
        ha="right",
        color="#687078",
        fontsize=8.2,
    )
    path = out_dir / "m66_tf_bimodality_population_diagnostic.png"
    fig.savefig(path, dpi=190, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    metrics = {
        "n_active": int(len(population)),
        "n_low_peak_censored": int(np.sum(censored)),
        "median_interior_interpolated_peak_hz": float(np.median(interior)),
        "median_continuous_center_hz": median_center,
        "fraction_margin_lt_0p2": float(np.mean(margin < 0.20)),
        "fraction_margin_lt_0p3": float(np.mean(margin < 0.30)),
        "median_normalized_entropy": float(np.median(population.normalized_entropy)),
        "rho_continuous_center_vs_ssi_gain": float(rho),
        "p_continuous_center_vs_ssi_gain": float(p_value),
    }
    return path, metrics


def load_stem_temporal_spectra() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import torch
    from eval.load_twin import load_twin

    model, _ = load_twin(M66_CHECKPOINT, device="cpu", verbose=False)
    cores = {
        "main": model.model.convnet,
        "auxiliary": model.model.auxiliary_convnet,
        "residual": model.model.residual_convnet,
    }
    results = {}
    for name, core in cores.items():
        weights = core.effective_temporal_weight().detach().float().cpu()[:, 0]
        spectrum = torch.fft.fftn(
            weights, s=(2048, 64, 64), dim=(-3, -2, -1)
        ).abs().square().numpy()
        tf_linear = np.fft.rfftfreq(2048, d=1.0 / 240.0)
        power_linear = spectrum[:, : len(tf_linear)].sum(axis=(-2, -1))
        power_linear /= np.maximum(power_linear.max(axis=1, keepdims=True), 1e-30)
        tf = np.geomspace(0.1171875, 60.0, 180)
        power = np.stack(
            [np.interp(tf, tf_linear, row) for row in power_linear], axis=0
        )
        peak_tf = tf[np.argmax(power, axis=1)]
        order = np.argsort(peak_tf)
        results[name] = (tf, power[order], peak_tf[order])
    return results


def render_stem_spectra(out_dir: Path) -> Path:
    configure()
    results = load_stem_temporal_spectra()
    fig = plt.figure(figsize=(13.2, 4.25), facecolor="white")
    grid = fig.add_gridspec(
        1, 4, left=0.065, right=0.98, top=0.74, bottom=0.18,
        wspace=0.28, width_ratios=(1, 0.68, 1, 0.92),
    )
    fig.suptitle(
        "The first temporal stem already contains slow and band-pass channel families",
        x=0.065,
        y=0.96,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.87,
        "Rows are complete learned spatiotemporal filters, normalized within filter and sorted by spectral peak. "
        "Downstream output units mix these channels nonlinearly.",
        ha="left",
        va="top",
        fontsize=10.0,
        color="#40464D",
    )
    cmap = LinearSegmentedColormap.from_list(
        "stem_power", ["#F7F7F4", "#88B7B2", "#184A60"]
    )
    group_rows = []
    for column, name in enumerate(("main", "auxiliary", "residual")):
        ax = fig.add_subplot(grid[0, column])
        tf, power, peak_tf = results[name]
        log_tf = np.log(tf)
        log_edges = np.empty(len(tf) + 1, dtype=float)
        log_edges[1:-1] = 0.5 * (log_tf[:-1] + log_tf[1:])
        log_edges[0] = log_tf[0] - 0.5 * (log_tf[1] - log_tf[0])
        log_edges[-1] = log_tf[-1] + 0.5 * (log_tf[-1] - log_tf[-2])
        tf_edges = np.exp(log_edges)
        mesh = ax.pcolormesh(
            tf_edges,
            np.arange(len(power) + 1),
            power,
            shading="auto",
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        ax.scatter(
            peak_tf,
            np.arange(len(power)) + 0.5,
            s=18,
            marker="o",
            facecolors="none",
            edgecolors="#E86D34",
            linewidths=0.8,
        )
        ax.set_xscale("log")
        ax.set_xlim(0.1, 60)
        ax.set_xticks([0.2, 0.8, 3.2, 12.8, 47.2], ["0.2", "0.8", "3.2", "12.8", "47"])
        ax.set_ylim(len(power), 0)
        ax.set_yticks([])
        ax.set_xlabel("temporal frequency (Hz)")
        ax.set_title(f"{name} stem · {len(power)} filters", loc="left")
        low_count = int(np.sum(peak_tf <= 0.2))
        ax.text(
            0.98,
            0.04,
            f"{low_count}/{len(power)} peak ≤0.2 Hz",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.1,
            color="#30363B",
        )
        group_rows.append(
            [
                low_count,
                int(np.sum((peak_tf > 0.2) & (peak_tf <= 8))),
                int(np.sum((peak_tf > 8) & (peak_tf <= 20))),
                int(np.sum(peak_tf > 20)),
            ]
        )

    ax = fig.add_subplot(grid[0, 3])
    groups = np.asarray(group_rows)
    x = np.arange(4)
    bottom = np.zeros(4)
    branch_colors = ["#317C93", "#65A66E", "#D8753C"]
    for row, name, color in zip(groups, ("main", "auxiliary", "residual"), branch_colors):
        ax.bar(x, row, bottom=bottom, color=color, width=0.68, label=name)
        bottom += row
    for index, total in enumerate(bottom):
        ax.text(index, total + 0.5, str(int(total)), ha="center", fontsize=8.5)
    ax.set_xticks(x, ["≤0.2", "0.2–8", "8–20", ">20"])
    ax.set_xlabel("filter spectral peak (Hz)")
    ax.set_ylabel("number of first-stem filters")
    ax.set_ylim(0, max(bottom) * 1.18)
    ax.set_title("Filter-bank composition", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.text(
        0.98,
        0.035,
        "Orange circles mark within-filter spectral maxima · M66 checkpoint epoch 31",
        ha="right",
        color="#687078",
        fontsize=8.1,
    )
    path = out_dir / "m66_first_stem_temporal_families.png"
    fig.savefig(path, dpi=190, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return path


def encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def write_html(path: Path, examples: Path, population: Path, stems: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<div style="font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#17212b;background:#fff;padding:14px;line-height:1.35">
  <div style="max-width:1280px;margin:0 auto">
    <img alt="Eight representative M66 temporal tuning curves" src="data:image/png;base64,{encode(examples)}" style="width:100%;height:auto;display:block;border:1px solid #e6eaed;border-radius:10px" />
    <img alt="Population diagnostic of apparent M66 temporal-frequency bimodality" src="data:image/png;base64,{encode(population)}" style="width:100%;height:auto;display:block;margin-top:18px;border:1px solid #e6eaed;border-radius:10px" />
    <img alt="Learned M66 first-stem temporal filter families" src="data:image/png;base64,{encode(stems)}" style="width:100%;height:auto;display:block;margin-top:18px;border:1px solid #e6eaed;border-radius:10px" />
  </div>
</div>"""
    path.write_text(html)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    population, tf, normalized = load_population()
    examples = render_examples(args.out_dir, population, tf, normalized)
    diagnostic, metrics = render_population(args.out_dir, population, tf, normalized)
    stems = render_stem_spectra(args.out_dir)
    population.to_csv(args.out_dir / "m66_tf_population_continuous_summary.csv")
    pd.Series(metrics).to_json(args.out_dir / "metrics.json", indent=2)
    write_html(args.visualization, examples, diagnostic, stems)
    print(examples)
    print(diagnostic)
    print(stems)
    print(args.visualization)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
