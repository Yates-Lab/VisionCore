#!/usr/bin/env python3
"""Render a paired population audit of real-neuron grating tuning predictions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODELS = ("Model",)
COLORS = {"Model": "#0072b2"}
DEFAULT_INPUT = ROOT / "outputs/dekel240_paper/real_neuron_grating_tuning"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--models",
        default=",".join(MODELS),
        help="Comma-separated output labels to compare in the requested order",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })


def parse_model_labels(value: str) -> tuple[str, ...]:
    labels = tuple(label.strip() for label in value.split(",") if label.strip())
    if not labels:
        raise ValueError("--models must contain at least one label")
    if len(set(labels)) != len(labels):
        raise ValueError("--models contains duplicate labels")
    return labels


def load_metrics(input_dir: Path) -> dict[str, pd.DataFrame]:
    result = {}
    for model in MODELS:
        path = input_dir / model / "unit_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        table = pd.read_csv(path)
        # Normalize legacy caches into scope-neutral names.  Older all-repeat
        # runs were correct, but their column names still said ``test``.
        if "n_spikes" not in table and "test_spikes" in table:
            table["n_spikes"] = table["test_spikes"]
        if "grating_bps" not in table and "grating_test_bps" in table:
            table["grating_bps"] = table["grating_test_bps"]
        table["key"] = table.session.astype(str) + ":" + table.cid.astype(int).astype(str)
        if table.key.duplicated().any():
            raise ValueError(f"Duplicate cells in {path}")
        result[model] = table
    return result


def load_data_scope(input_dir: Path) -> str:
    path = input_dir / MODELS[0] / "provenance.json"
    if path.exists():
        return str(json.loads(path.read_text()).get("data_scope", "heldout"))
    return "all" if input_dir.name.endswith("_all_data") else "heldout"


def paired_tables(tables: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], set[str]]:
    common = set.intersection(*(set(table.key) for table in tables.values()))
    return {
        model: table.loc[table.key.isin(common)].sort_values("key").reset_index(drop=True)
        for model, table in tables.items()
    }, common


def hierarchical_bootstrap(
    table: pd.DataFrame,
    column: str,
    *,
    n_boot: int,
    seed: int,
    statistic=np.nanmedian,
) -> tuple[float, float, float, int]:
    work = table.loc[np.isfinite(pd.to_numeric(table[column], errors="coerce"))].copy()
    if work.empty:
        return float("nan"), float("nan"), float("nan"), 0
    sessions = work.session.unique()
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    grouped = {session: work.loc[work.session.eq(session)] for session in sessions}
    for idx in range(n_boot):
        sampled_sessions = rng.choice(sessions, len(sessions), replace=True)
        values = []
        for session in sampled_sessions:
            group = grouped[session]
            take = rng.integers(0, len(group), len(group))
            values.extend(group.iloc[take][column].to_numpy(dtype=np.float64))
        boots[idx] = statistic(np.asarray(values, dtype=np.float64))
    center = statistic(work[column].to_numpy(dtype=np.float64))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(center), float(lo), float(hi), int(len(work))


def hierarchical_spearman(
    table: pd.DataFrame,
    x: str,
    y: str,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, int]:
    good = np.isfinite(table[x]) & np.isfinite(table[y])
    work = table.loc[good].copy()
    if len(work) < 5:
        return float("nan"), float("nan"), float("nan"), len(work)
    sessions = work.session.unique()
    groups = {session: work.loc[work.session.eq(session)] for session in sessions}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        parts = []
        for session in rng.choice(sessions, len(sessions), replace=True):
            group = groups[session]
            parts.append(group.iloc[rng.integers(0, len(group), len(group))])
        draw = pd.concat(parts, ignore_index=True)
        boots.append(spearmanr(draw[x], draw[y]).statistic)
    center = float(spearmanr(work[x], work[y]).statistic)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return center, float(lo), float(hi), len(work)


def quality_keys(reference: pd.DataFrame) -> dict[str, set[str]]:
    enough = reference.n_spikes.ge(50)
    return {
        "sf": set(reference.loc[enough & reference.sf_split_half.ge(0.2), "key"]),
        "orientation": set(reference.loc[enough & reference.ori_split_half.ge(0.2), "key"]),
        "phase": set(reference.loc[
            enough & reference.data_phase_vector_consistency.ge(0.5), "key"
        ]),
        "latency": set(reference.loc[
            enough & reference.temporal_split_half.ge(0.2) & reference.data_lag_boundary.eq(0), "key"
        ]),
        "sf_preference": set(reference.loc[
            enough & reference.sf_split_half.ge(0.2) & reference.data_sf_boundary.eq(0), "key"
        ]),
    }


def build_summary(
    tables: dict[str, pd.DataFrame], gates: dict[str, set[str]], n_boot: int, seed: int
) -> pd.DataFrame:
    specs = [
        ("SF curve correlation", "sf_curve_corr", "sf", "higher"),
        ("orientation curve correlation", "ori_curve_corr", "orientation", "higher"),
        ("phase curve correlation", "phase_curve_corr", "phase", "higher"),
        ("latency-profile correlation", "temporal_curve_corr", "latency", "higher"),
        ("preferred SF error (octaves)", "sf_error_octaves", "sf_preference", "lower"),
        ("preferred orientation error (deg)", "ori_error_deg", "orientation", "lower"),
        ("preferred phase error (deg)", "phase_error_deg", "phase", "lower"),
        ("peak-latency error (ms)", "lag_error_ms", "latency", "lower"),
        ("model F1/F0", "model_f1_f0", "phase", "descriptive"),
        ("grating BPS", "grating_bps", "all", "higher"),
    ]
    rows = []
    for model_idx, model in enumerate(MODELS):
        table = tables[model]
        for metric_idx, (label, column, gate, direction) in enumerate(specs):
            work = table if gate == "all" else table.loc[table.key.isin(gates[gate])]
            center, lo, hi, n = hierarchical_bootstrap(
                work, column, n_boot=n_boot, seed=seed + 97 * model_idx + metric_idx
            )
            rows.append({
                "model": model,
                "metric": label,
                "column": column,
                "quality_gate": gate,
                "direction": direction,
                "median": center,
                "ci_low": lo,
                "ci_high": hi,
                "n_units": n,
                "n_sessions": int(work.loc[np.isfinite(work[column]), "session"].nunique()),
            })
        phase_work = table.loc[table.key.isin(gates["phase"])]
        center, lo, hi, n = hierarchical_spearman(
            phase_work, "data_debiased_f1_f0", "model_f1_f0",
            n_boot=n_boot, seed=seed + 1000 + model_idx,
        )
        rows.append({
            "model": model,
            "metric": "F1/F0 population Spearman",
            "column": "data_debiased_f1_f0:model_f1_f0",
            "quality_gate": "phase",
            "direction": "higher",
            "median": center,
            "ci_low": lo,
            "ci_high": hi,
            "n_units": n,
            "n_sessions": int(phase_work.session.nunique()),
        })
    return pd.DataFrame(rows)


def _point_ci(axis, x, row, color, offset=0.0, marker="o", label=None):
    axis.errorbar(
        x + offset,
        row["median"],
        yerr=[[row["median"] - row["ci_low"]], [row["ci_high"] - row["median"]]],
        fmt=marker,
        color=color,
        markeredgecolor="white",
        markeredgewidth=0.7,
        markersize=6.5,
        linewidth=1.35,
        capsize=2.5,
        label=label,
        zorder=3,
    )


def plot_summary(
    summary: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    gates: dict[str, set[str]],
    output_dir: Path,
    data_scope: str,
) -> None:
    configure_matplotlib()
    fig = plt.figure(figsize=(14.2, 8.4), constrained_layout=False)
    grid = fig.add_gridspec(2, 4, left=0.06, right=0.985, top=0.88, bottom=0.10, wspace=0.38, hspace=0.42)
    title = (
        f"{MODELS[0]} recovery of real V1 grating tuning"
        if len(MODELS) == 1
        else "Comparison of model recovery of real V1 grating tuning"
    )
    fig.suptitle(title, x=0.06, ha="left", fontsize=18, fontweight="bold")
    common_n = len(tables[MODELS[0]])
    n_sessions = int(tables[MODELS[0]].session.nunique())
    session_label = "session" if n_sessions == 1 else "sessions"
    scope_text = (
        "All recorded forage-grating trials (descriptive fit assay; includes training data)"
        if data_scope == "all" else "Held-out final 15% of forage-grating trials"
    )
    fig.text(
        0.06, 0.915,
        f"{scope_text} · {n_sessions} {session_label} · "
        f"{common_n} cells with model predictions "
        "· data-defined conditions · noise-debiased recorded F1/F0",
        ha="left", va="center", fontsize=10, color="#444444",
    )

    curve_specs = [
        ("sf", "SF", "SF curve correlation"),
        ("orientation", "orientation", "orientation curve correlation"),
        ("phase", "phase", "phase curve correlation"),
        ("latency", "latency", "latency-profile correlation"),
    ]
    ax = fig.add_subplot(grid[0, :2])
    offsets = np.linspace(-0.24, 0.24, len(MODELS))
    for model, offset in zip(MODELS, offsets):
        rows = summary.loc[summary.model.eq(model)].set_index("metric")
        for idx, (_, _, metric) in enumerate(curve_specs):
            _point_ci(ax, idx, rows.loc[metric], COLORS[model], offset=offset, label=model if idx == 0 else None)
    reference = tables[MODELS[0]]
    for idx, (gate, _, _) in enumerate(curve_specs):
        rel_col = {"sf": "sf_split_half", "orientation": "ori_split_half", "phase": "phase_split_half", "latency": "temporal_split_half"}[gate]
        work = reference.loc[reference.key.isin(gates[gate])]
        value = np.nanmedian(work[rel_col])
        ax.scatter(idx, value, marker="_", s=180, linewidths=2.2, color="#666666", zorder=2)
    ax.set_title("A  Recovered tuning-curve shape", loc="left")
    ax.set_ylabel("model–data correlation")
    ax.set_xticks(range(len(curve_specs)), [label for _, label, _ in curve_specs])
    ax.set_ylim(-0.1, 1.03)
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, ncol=4, loc="lower left")
    ax.text(0.99, 0.04, "gray ticks: neural split-half reliability", transform=ax.transAxes, ha="right", fontsize=8.5, color="#666666")

    small_specs = [
        (grid[0, 2], "B  Preferred SF", "preferred SF error (octaves)", "absolute error (octaves)", 0),
        (grid[0, 3], "C  Preferred orientation", "preferred orientation error (deg)", "absolute error (deg)", 1),
        (grid[1, 0], "D  Preferred retinal phase", "preferred phase error (deg)", "absolute error (deg)", 2),
        (grid[1, 1], "E  Response latency", "peak-latency error (ms)", "absolute error (ms)", 3),
        (grid[1, 2], "F  Phase modulation", "model F1/F0", "standard F1/F0", 4),
        (grid[1, 3], "G  Poisson likelihood", "grating BPS", "bits/spike", 5),
    ]
    for subplot, title, metric, ylabel, seed_idx in small_specs:
        axis = fig.add_subplot(subplot)
        rows = summary.loc[summary.metric.eq(metric)].set_index("model")
        x_offset = 0
        if metric == "model F1/F0":
            reference = tables[MODELS[0]].loc[
                tables[MODELS[0]].key.isin(gates["phase"])
            ]
            center, lo, hi, _ = hierarchical_bootstrap(
                reference,
                "data_debiased_f1_f0",
                n_boot=2000,
                seed=20260819 + seed_idx,
            )
            observed = pd.Series({"median": center, "ci_low": lo, "ci_high": hi})
            _point_ci(axis, 0, observed, "#111111", marker="s")
            x_offset = 1
        for idx, model in enumerate(MODELS):
            _point_ci(axis, idx + x_offset, rows.loc[model], COLORS[model])
        axis.set_title(title, loc="left")
        axis.set_ylabel(ylabel)
        labels = (["recorded"] if x_offset else []) + list(MODELS)
        axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        axis.grid(axis="y", color="#e0e0e0", linewidth=0.7)
        if metric == "grating BPS":
            axis.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
        else:
            axis.set_ylim(bottom=0)
        n_text = ", ".join(f"{model} n={int(rows.loc[model, 'n_units'])}" for model in MODELS)
        # Keep sample-size text clear of the zero-reference line in the BPS panel.
        n_y = 0.08 if metric == "grating BPS" else 0.02
        axis.text(0.02, n_y, n_text, transform=axis.transAxes, fontsize=6.7, color="#666666", va="bottom")

    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"real_neuron_grating_tuning_summary.{suffix}", dpi=220 if suffix == "png" else None)
    plt.close(fig)


def plot_preference_scatter(tables: dict[str, pd.DataFrame], gates: dict[str, set[str]], output_dir: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(
        len(MODELS),
        3,
        figsize=(11.2, 4.5 * len(MODELS)),
        squeeze=False,
        constrained_layout=True,
    )
    sf_values = np.concatenate([
        table.loc[table.key.isin(gates["sf_preference"]), ["data_preferred_sf_cpd", "model_preferred_sf_cpd"]]
        .to_numpy(dtype=float).ravel()
        for table in tables.values()
    ])
    sf_values = sf_values[np.isfinite(sf_values) & (sf_values > 0)]
    sf_min = float(2.0 ** np.floor(np.log2(np.min(sf_values))))
    sf_max = float(2.0 ** np.ceil(np.log2(np.max(sf_values))))
    sf_ticks = 2.0 ** np.arange(np.log2(sf_min), np.log2(sf_max) + 1)
    for row, model in enumerate(MODELS):
        table = tables[model]
        sf = table.loc[table.key.isin(gates["sf_preference"])]
        ori = table.loc[table.key.isin(gates["orientation"])]
        phase = table.loc[table.key.isin(gates["phase"])]
        ax = axes[row, 0]
        interior = sf.model_sf_boundary.eq(0)
        ax.scatter(
            sf.loc[interior, "data_preferred_sf_cpd"],
            sf.loc[interior, "model_preferred_sf_cpd"],
            s=9, alpha=0.28, color=COLORS[model], edgecolors="none",
            label="interior model peak",
        )
        ax.scatter(
            sf.loc[~interior, "data_preferred_sf_cpd"],
            sf.loc[~interior, "model_preferred_sf_cpd"],
            s=14, alpha=0.55, facecolors="none", edgecolors=COLORS[model], linewidths=0.65,
            label="model peak at tested boundary",
        )
        ax.plot([sf_min, sf_max], [sf_min, sf_max], color="#777777", linewidth=0.8)
        ax.set(
            xscale="log", yscale="log",
            xlim=(sf_min / 1.12, sf_max * 1.12), ylim=(sf_min / 1.12, sf_max * 1.12),
            xticks=sf_ticks, yticks=sf_ticks,
        )
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_ylabel(f"{model} prediction")
        if row == 0:
            ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        ax = axes[row, 1]
        ori_delta = (
            (ori.model_preferred_ori_deg.to_numpy() - ori.data_preferred_ori_deg.to_numpy() + 90.0)
            % 180.0
        ) - 90.0
        ori_unwrapped = ori.data_preferred_ori_deg.to_numpy() + ori_delta
        ax.scatter(ori.data_preferred_ori_deg, ori_unwrapped, s=9, alpha=0.28, color=COLORS[model], edgecolors="none")
        ax.plot([0, 180], [0, 180], color="#777777", linewidth=0.8)
        ax.set(xlim=(0, 180), ylim=(-45, 225), xticks=[0, 45, 90, 135, 180], yticks=[0, 45, 90, 135, 180])
        ax = axes[row, 2]
        ax.scatter(
            phase.data_debiased_f1_f0,
            phase.model_f1_f0,
            s=9,
            alpha=0.28,
            color=COLORS[model],
            edgecolors="none",
        )
        upper = max(
            1.0,
            float(
                np.nanpercentile(
                    np.r_[phase.data_debiased_f1_f0, phase.model_f1_f0], 98
                )
            ),
        )
        ax.plot([0, upper], [0, upper], color="#777777", linewidth=0.8)
        ax.set(xlim=(0, upper), ylim=(0, upper))
    for col, title in enumerate(("preferred SF (cycles/deg)", "preferred orientation (nearest axial equivalent)", "F1/F0")):
        axes[0, col].set_title(title, fontweight="bold")
        axes[-1, col].set_xlabel("recorded neuron")
    fig.suptitle("Measured versus predicted grating tuning", fontsize=17, fontweight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"real_neuron_grating_tuning_scatter.{suffix}", dpi=220 if suffix == "png" else None)
    plt.close(fig)


def _normalize_curve(values: np.ndarray, mode: str = "range") -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if mode == "mean":
        denom = np.nanmean(values)
        return values / denom if np.isfinite(denom) and abs(denom) > 1e-9 else np.full_like(values, np.nan)
    lo, hi = np.nanmin(values), np.nanmax(values)
    return (values - lo) / (hi - lo) if np.isfinite(hi - lo) and hi - lo > 1e-9 else np.full_like(values, np.nan)


def load_unit_curves(input_dir: Path, model: str, session: str, cid: int) -> dict[str, np.ndarray]:
    path = input_dir / model / "sessions" / f"{session}_curves.npz"
    with np.load(path) as archive:
        cids = archive["cids"].astype(int)
        matches = np.flatnonzero(cids == int(cid))
        if len(matches) != 1:
            raise KeyError((model, session, cid))
        row = int(matches[0])
        result = {key: archive[key].copy() for key in archive.files if key not in ("cids", "canonical_channels")}
    for key in ("data_temporal", "model_temporal", "data_sf", "model_sf", "data_ori", "model_ori", "data_phase", "model_phase"):
        result[key] = result[key][row]
    return result


def select_examples(reference: pd.DataFrame, gates: dict[str, set[str]], n: int = 8) -> pd.DataFrame:
    keys = gates["sf"] & gates["orientation"] & gates["phase"] & gates["latency"]
    work = reference.loc[reference.key.isin(keys)].copy()
    # Example selection is based only on neural repeatability, never on which
    # model happened to fit a cell well.
    work["quality"] = work[
        ["sf_split_half", "ori_split_half", "phase_split_half", "temporal_split_half"]
    ].mean(axis=1)
    work = work.loc[work.quality.ge(work.quality.median())].sort_values("data_preferred_sf_cpd")
    if len(work) <= n:
        return work
    positions = np.rint(np.linspace(0, len(work) - 1, n)).astype(int)
    return work.iloc[positions]


def plot_examples(
    input_dir: Path, tables: dict[str, pd.DataFrame], gates: dict[str, set[str]],
    output_dir: Path, data_scope: str,
) -> pd.DataFrame:
    configure_matplotlib()
    selected = select_examples(tables[MODELS[0]], gates)
    with PdfPages(output_dir / "real_neuron_grating_tuning_examples.pdf") as pdf:
        for _, unit in selected.iterrows():
            session, cid = str(unit.session), int(unit.cid)
            curves = {model: load_unit_curves(input_dir, model, session, cid) for model in MODELS}
            axes_x = {
                "temporal": curves[MODELS[0]]["lags_ms"],
                "sf": curves[MODELS[0]]["sfs"],
                "ori": curves[MODELS[0]]["oris"],
                "phase": curves[MODELS[0]]["phase_bins_deg"],
            }
            fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.3), constrained_layout=True)
            scope_label = "all-trial" if data_scope == "all" else "held-out"
            fig.suptitle(f"{session} · cid {cid} · {scope_label} real grating tuning", fontsize=14, fontweight="bold")
            specs = [
                ("temporal", "data_temporal", "model_temporal", "response latency", "lag after grating state (ms)", "range"),
                ("sf", "data_sf", "model_sf", "spatial frequency", "cycles/deg", "range"),
                ("ori", "data_ori", "model_ori", "orientation", "orientation (deg)", "range"),
                ("phase", "data_phase", "model_phase", "retinal phase", "phase (deg)", "mean"),
            ]
            for axis, (kind, data_key, model_key, title, xlabel, norm) in zip(axes, specs):
                x = axes_x[kind]
                data = _normalize_curve(curves[MODELS[0]][data_key], norm)
                axis.plot(x, data, color="#111111", linewidth=2.2, marker="o", markersize=3.5, label="recorded")
                for model in MODELS:
                    model_x = curves[model][{"temporal": "lags_ms", "sf": "sfs", "ori": "oris", "phase": "phase_bins_deg"}[kind]]
                    axis.plot(model_x, _normalize_curve(curves[model][model_key], norm), color=COLORS[model], linewidth=1.35, alpha=0.95, label=model)
                axis.set_title(title)
                axis.set_xlabel(xlabel)
                axis.set_ylabel("mean-normalized rate" if norm == "mean" else "normalized modulation")
                axis.grid(color="#e5e5e5", linewidth=0.6)
                if kind == "sf":
                    axis.set_xscale("log", base=2)
                    axis.set_xticks(x)
                    axis.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
            axes[0].legend(frameon=False, ncol=5, bbox_to_anchor=(0, 1.24), loc="lower left")
            pdf.savefig(fig)
            plt.close(fig)
    selected.to_csv(output_dir / "example_cells.csv", index=False)
    return selected


def plot_session_atlases(
    input_dir: Path, tables: dict[str, pd.DataFrame], output_dir: Path, data_scope: str,
) -> None:
    configure_matplotlib()
    atlas_dir = output_dir / "per_session_pdfs"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    reference = tables[MODELS[0]]
    combined_path = output_dir / "all_real_neurons_grating_tuning.pdf"
    with PdfPages(combined_path) as combined_pdf:
        for session, session_table in reference.groupby("session", sort=True):
            with PdfPages(atlas_dir / f"{session}_real_grating_tuning.pdf") as session_pdf:
                for start in range(0, len(session_table), 6):
                    page = session_table.iloc[start : start + 6]
                    fig, axes = plt.subplots(len(page), 4, figsize=(12.8, 1.75 * len(page) + 1.0), squeeze=False, constrained_layout=True)
                    scope_label = "all-trial" if data_scope == "all" else "held-out"
                    fig.suptitle(
                        f"{session} · {scope_label} real-grating responses",
                        x=0.01, ha="left", fontsize=14, fontweight="bold",
                    )
                    for row, (_, unit) in enumerate(page.iterrows()):
                        cid = int(unit.cid)
                        model_curves = {model: load_unit_curves(input_dir, model, session, cid) for model in MODELS}
                        specs = [
                            ("lags_ms", "data_temporal", "model_temporal", "lag (ms)", "range"),
                            ("sfs", "data_sf", "model_sf", "SF (c/deg)", "range"),
                            ("oris", "data_ori", "model_ori", "orientation (deg)", "range"),
                            ("phase_bins_deg", "data_phase", "model_phase", "retinal phase (deg)", "mean"),
                        ]
                        for col, (xkey, dkey, mkey, xlabel, norm) in enumerate(specs):
                            axis = axes[row, col]
                            base = model_curves[MODELS[0]]
                            axis.plot(base[xkey], _normalize_curve(base[dkey], norm), color="#111111", linewidth=1.7)
                            for model in MODELS:
                                curve = model_curves[model]
                                axis.plot(curve[xkey], _normalize_curve(curve[mkey], norm), color=COLORS[model], linewidth=0.95)
                            axis.grid(color="#e7e7e7", linewidth=0.5)
                            if row == len(page) - 1:
                                axis.set_xlabel(xlabel, fontsize=8)
                            if row == 0:
                                axis.set_title(
                                    ("response latency", "spatial frequency", "orientation", "retinal phase")[col],
                                    fontsize=9, fontweight="bold",
                                )
                            if col == 0:
                                axis.set_ylabel(f"cid {cid}\nnormalized", fontsize=8)
                            if xkey == "sfs":
                                axis.set_xscale("log", base=2)
                                axis.set_xticks(curve[xkey])
                                axis.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
                    handles = [mpl.lines.Line2D([], [], color="#111111", lw=2, label="recorded")]
                    handles += [mpl.lines.Line2D([], [], color=COLORS[m], lw=1.4, label=m) for m in MODELS]
                    fig.legend(handles=handles, frameon=False, ncol=5, loc="upper right", bbox_to_anchor=(0.995, 0.998))
                    session_pdf.savefig(fig)
                    combined_pdf.savefig(fig)
                    plt.close(fig)


def main() -> None:
    global MODELS, COLORS
    args = parse_args()
    MODELS = parse_model_labels(args.models)
    palette = plt.get_cmap("tab10")
    COLORS = {
        model: COLORS.get(model, mpl.colors.to_hex(palette(index % 10)))
        for index, model in enumerate(MODELS)
    }
    output_dir = args.output_dir or args.input_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_metrics(args.input_dir)
    tables, common = paired_tables(raw)
    data_scope = load_data_scope(args.input_dir)
    gates = quality_keys(tables[MODELS[0]])
    summary = build_summary(tables, gates, args.bootstrap, args.seed)
    summary.to_csv(output_dir / "population_summary.csv", index=False)
    availability = pd.DataFrame({
        "model": list(MODELS),
        "available_canonical_units": [len(raw[model]) for model in MODELS],
        "paired_common_units": [len(common)] * len(MODELS),
        "sessions": [raw[model].session.nunique() for model in MODELS],
    })
    availability.to_csv(output_dir / "population_availability.csv", index=False)
    gates_json = {name: {"n_units": len(keys), "keys": sorted(keys)} for name, keys in gates.items()}
    (output_dir / "quality_gates.json").write_text(json.dumps(gates_json, indent=2) + "\n")
    plot_summary(summary, tables, gates, output_dir, data_scope)
    plot_preference_scatter(tables, gates, output_dir)
    selected = plot_examples(args.input_dir, tables, gates, output_dir, data_scope)
    plot_session_atlases(args.input_dir, tables, output_dir, data_scope)
    report = {
        "canonical_population": 756,
        "available_canonical_units": {
            model: int(len(raw[model])) for model in MODELS
        },
        "missing_canonical_units": {
            model: int(756 - len(raw[model])) for model in MODELS
        },
        "paired_common_units": len(common),
        "sessions": int(tables[MODELS[0]].session.nunique()),
        "models": list(MODELS),
        "data_scope": data_scope,
        "quality_gate_counts": {name: len(keys) for name, keys in gates.items()},
        "example_cells": selected[["session", "cid"]].to_dict("records"),
        "all_available_units_rendered_in_session_atlases": True,
        "combined_all_unit_atlas": "all_real_neurons_grating_tuning.pdf",
        "interpretation_guard": (
            "The real experiment has SF, orientation, and gaze-induced retinal phase. "
            "Its temporal result is a response-lag profile, not temporal-frequency tuning."
        ),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
