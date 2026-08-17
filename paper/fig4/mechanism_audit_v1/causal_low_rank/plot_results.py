#!/usr/bin/env python3
"""Build the three causal-low-rank candidate figures from saved products only.

This module deliberately has no model, cache, or optimizer imports.  It reads
only consolidated evaluation products, refuses to synthesize absent values,
and records any unavailable figure in ``plot_manifest.json``.  Negative map
recovery and SSI-transfer values are retained exactly as saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/causal_low_rank_v1"
CONTRAST_ORDER = ("low_0_to_2", "high_0_to_1", "high_1_to_3")
CONTRAST_LABEL = {
    "low_0_to_2": "Lower-SF sharpening\n0× → 2×",
    "high_0_to_1": "Higher-SF sharpening\n0× → 1×",
    "high_1_to_3": "Higher-SF reversal\n1× → 3×",
}
CONTRAST_SHORT = {
    "low_0_to_2": "lower-SF sharpening",
    "high_0_to_1": "higher-SF sharpening",
    "high_1_to_3": "higher-SF reversal",
}
CONTRAST_COLOR = {
    "low_0_to_2": "#168B7A",
    "high_0_to_1": "#B4558B",
    "high_1_to_3": "#D87620",
}
METHOD_STYLE = {
    "learned": ("#173F5F", "o", "-", "learned causal"),
    "movement_pca": ("#68737D", "s", "--", "movement PCA"),
    "readout_svd": ("#7D5BA6", "^", "-.", "readout SVD"),
}
RANK_TICKS = (0, 1, 2, 4, 8, 16, 32, 128)
EPS = 1e-30


class DataUnavailable(RuntimeError):
    """Raised when a figure cannot be made without inventing missing data."""


@dataclass(frozen=True)
class Inputs:
    root: Path
    rank_summary: Path
    baseline_results: Path
    per_unit_results: Path
    cross_scale_results: Path
    objective_maps: Path
    statistics: Path
    bootstrap_results: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figure-data-dir", type=Path)
    parser.add_argument(
        "--selected-rank",
        action="append",
        default=[],
        metavar="CONTRAST=RANK",
        help="Plot-only rank declaration; repeat for multiple contrasts.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return an error if any candidate figure lacks a required saved product.",
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def input_paths(root: Path) -> Inputs:
    def resolve(name: str) -> Path:
        direct = root / name
        evaluation = root / "evaluation" / name
        return direct if direct.exists() or not evaluation.exists() else evaluation

    return Inputs(
        root=root,
        rank_summary=resolve("rank_summary.csv"),
        baseline_results=resolve("baseline_results.csv"),
        per_unit_results=resolve("per_unit_results.csv"),
        cross_scale_results=resolve("cross_scale_results.csv"),
        objective_maps=resolve("objective_representative_exact_maps.npz"),
        statistics=resolve("statistics.json"),
        bootstrap_results=resolve("bootstrap_results.npz"),
    )


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise DataUnavailable(f"{label} is unavailable: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise DataUnavailable(f"{label} is empty: {path}") from error


def require_columns(frame: pd.DataFrame, names: Iterable[str], label: str) -> None:
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise DataUnavailable(f"{label} lacks required columns: {', '.join(missing)}")


def numeric(frame: pd.DataFrame, names: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for name in names:
        if name in result:
            result[name] = pd.to_numeric(result[name], errors="coerce")
    return result


def parse_rank_overrides(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected CONTRAST=RANK, received {item!r}")
        contrast, raw = item.split("=", 1)
        if contrast not in CONTRAST_ORDER:
            raise ValueError(f"Unknown contrast {contrast!r}")
        rank = int(raw)
        if rank <= 0 or rank > 128:
            raise ValueError(f"Invalid rank {rank}")
        result[contrast] = rank
    return result


def read_statistics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _rank_from_statistics(value: Any, contrast: str, ancestors: tuple[str, ...] = ()) -> int | None:
    """Find an explicitly declared selected rank, never a test-chosen optimum."""
    if isinstance(value, dict):
        direct = value.get(contrast)
        if isinstance(direct, (int, float)) and any("rank" in key.lower() for key in ancestors):
            return int(direct)
        if isinstance(direct, dict):
            for key in ("selected_rank", "retained_rank", "final_rank", "rank"):
                candidate = direct.get(key)
                if isinstance(candidate, (int, float)):
                    return int(candidate)
        context = ancestors + tuple(str(key) for key in value.keys() if str(key) == contrast)
        for key, child in value.items():
            if key in ("selected_rank", "retained_rank", "final_rank"):
                if contrast in ancestors and isinstance(child, (int, float)):
                    return int(child)
            found = _rank_from_statistics(child, contrast, context + (str(key),))
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _rank_from_statistics(child, contrast, ancestors)
            if found is not None:
                return found
    return None


def selected_ranks(
    statistics: dict[str, Any], rank_table: pd.DataFrame, overrides: dict[str, int]
) -> dict[str, int]:
    result = dict(overrides)
    for contrast in CONTRAST_ORDER:
        if contrast in result:
            continue
        declared = _rank_from_statistics(statistics, contrast)
        if declared is not None and 0 < declared <= 128:
            result[contrast] = declared
            continue
        subset = rank_table.loc[
            rank_table.contrast.eq(contrast)
            & rank_table.method.eq("learned")
            & rank_table.stage.eq("crossval")
        ]
        ranks = sorted(set(pd.to_numeric(subset["rank"], errors="coerce").dropna().astype(int)))
        if len(ranks) == 1:
            result[contrast] = ranks[0]
    return result


def canonical_axes_stable(statistics: dict[str, Any]) -> bool:
    accepted = {
        "canonical_axes_stable",
        "stable_canonical_axes",
        "canonical_axis_stability_passed",
        "interpret_canonical_axes",
    }

    def visit(value: Any) -> bool:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in accepted and child is True:
                    return True
                if visit(child):
                    return True
        elif isinstance(value, list):
            return any(visit(child) for child in value)
        return False

    return visit(statistics)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, destination: Path) -> list[str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, options in (
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": 600}),
    ):
        path = destination.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="white", **options)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def _preferred_stage(table: pd.DataFrame) -> str:
    crossed = table.loc[
        table.stage.eq("crossval") & table.method.eq("learned")
    ]
    crossed_complete = all(
        pd.to_numeric(
            crossed.loc[crossed.contrast.eq(contrast), "fold"], errors="coerce"
        ).nunique()
        == 4
        for contrast in CONTRAST_ORDER
    )
    if crossed_complete:
        return "crossval"
    if bool(table.stage.eq("screening").any()):
        return "screening"
    if bool(table.stage.eq("crossval").any()):
        return "crossval"
    raise DataUnavailable("Neither crossed-validation nor screening rank results are available")


def _summarize_curve(points: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["contrast", "stage", "method", "rank"]
    for key, frame in points.groupby(keys, dropna=False):
        values = pd.to_numeric(frame[metric], errors="coerce").dropna().to_numpy(float)
        if not len(values):
            continue
        rows.append(
            {
                **dict(zip(keys, key)),
                "metric": metric,
                "mean": float(np.mean(values)),
                "ci_low": float(np.percentile(values, 2.5)),
                "ci_high": float(np.percentile(values, 97.5)),
                "ci_source": "held-out-fold percentile",
                "n_values": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def _random_summary(baselines: pd.DataFrame, stage: str, metric: str) -> pd.DataFrame:
    subset = baselines.loc[
        baselines.stage.eq(stage) & baselines.method.isin(("random_haar", "random"))
    ]
    # A budget-limited run can complete the predeclared screening fold first,
    # then save its random controls under the crossval product namespace for
    # later fold-wise consolidation.  These rows are the same held-out fold;
    # use them only when no stage-labelled screening random control exists.
    random_source = stage
    if subset.empty and stage == "screening":
        subset = baselines.loc[
            baselines.stage.eq("crossval")
            & baselines.method.isin(("random_haar", "random"))
            & pd.to_numeric(baselines.get("fold"), errors="coerce").eq(0)
        ]
        random_source = "crossval_fold0_for_screening"
    rows = []
    for (contrast, rank), frame in subset.groupby(["contrast", "rank"]):
        values = pd.to_numeric(frame[metric], errors="coerce").dropna().to_numpy(float)
        if not len(values):
            continue
        rows.append(
            {
                "contrast": contrast,
                "stage": random_source,
                "method": "random_haar",
                "rank": int(rank),
                "metric": metric,
                "mean": float(np.mean(values)),
                "ci_low": float(np.percentile(values, 2.5)),
                "ci_high": float(np.percentile(values, 97.5)),
                "ci_source": (
                    "Haar-draw percentile on predeclared screening fold"
                    if random_source != "crossval"
                    else "Haar-draw percentile pooled over held-out folds"
                ),
                "n_values": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def _plot_method_curve(ax: plt.Axes, summary: pd.DataFrame, method: str) -> None:
    frame = summary.loc[summary.method.eq(method)].sort_values("rank")
    if frame.empty:
        return
    color, marker, linestyle, label = METHOD_STYLE[method]
    x = frame["rank"].to_numpy(float)
    y = frame["mean"].to_numpy(float)
    if method == "learned" and {"ci_low", "ci_high"}.issubset(frame.columns):
        low = pd.to_numeric(frame["ci_low"], errors="coerce").to_numpy(float)
        high = pd.to_numeric(frame["ci_high"], errors="coerce").to_numpy(float)
        if np.any(np.isfinite(low) & np.isfinite(high)):
            ax.fill_between(x, low, high, color=color, alpha=0.11, linewidth=0, zorder=2)
    ax.plot(
        x,
        y,
        color=color,
        marker=marker,
        markersize=3.8,
        markeredgewidth=0.7,
        linewidth=1.5 if method == "learned" else 1.05,
        linestyle=linestyle,
        zorder=5 if method == "learned" else 3,
    )
    ax.annotate(
        label,
        (x[-1], y[-1]),
        xytext=(4, {"learned": 7, "movement_pca": 0, "readout_svd": -7}[method]),
        textcoords="offset points",
        color=color,
        fontsize=6.2,
        va="center",
        clip_on=False,
    )


def load_bootstrap_summary(path: Path) -> pd.DataFrame:
    """Read common long-form bootstrap NPZ layouts without assuming results.

    Accepted layouts are (i) JSON records under ``summary_json`` or
    ``records_json``; (ii) a structured ``summary`` array; (iii) parallel
    one-dimensional arrays; or (iv) draw arrays named with both a contrast
    key and an exact metric name.  Unrecognized archives remain unused and
    are reported in the plot manifest rather than guessed.
    """
    if not path.is_file():
        return pd.DataFrame()
    metrics = (
        "map_r2_sufficiency",
        "map_r2_necessity",
        "ssi_fraction_transferred",
        "ssi_fraction_removed",
    )
    records: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=False) as archive:
        for key in ("summary_json", "records_json"):
            if key in archive:
                raw = np.asarray(archive[key]).item()
                decoded = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
                if isinstance(decoded, dict):
                    decoded = decoded.get("summary", decoded.get("records", []))
                if isinstance(decoded, list):
                    records.extend(item for item in decoded if isinstance(item, dict))
        if "summary" in archive and np.asarray(archive["summary"]).dtype.names:
            structured = np.asarray(archive["summary"])
            for row in structured:
                records.append(
                    {
                        name: (row[name].decode() if isinstance(row[name], bytes) else row[name].item())
                        for name in structured.dtype.names or ()
                    }
                )
        if not records:
            one_dimensional = {
                key: np.asarray(archive[key])
                for key in archive.files
                if np.asarray(archive[key]).ndim == 1
            }
            lengths = {len(value) for value in one_dimensional.values()}
            if len(lengths) == 1 and lengths and next(iter(lengths)) > 0 and {
                "contrast",
                "rank",
            }.issubset(one_dimensional):
                length = next(iter(lengths))
                for index in range(length):
                    records.append(
                        {
                            key: (
                                value[index].decode()
                                if isinstance(value[index], bytes)
                                else value[index].item()
                            )
                            for key, value in one_dimensional.items()
                        }
                    )
            else:
                for key, value in one_dimensional.items():
                    contrast = next((item for item in CONTRAST_ORDER if item in key), None)
                    metric = next((item for item in metrics if item in key), None)
                    if contrast is None or metric is None or not len(value):
                        continue
                    rank = None
                    for token in key.replace("-", "_").split("_"):
                        if token.startswith("rank") and token[4:].isdigit():
                            rank = int(token[4:])
                    if rank is None:
                        continue
                    finite = pd.to_numeric(pd.Series(value), errors="coerce").dropna().to_numpy(float)
                    if len(finite):
                        records.append(
                            {
                                "contrast": contrast,
                                "rank": rank,
                                "method": "learned",
                                "metric": metric,
                                "mean": float(np.mean(finite)),
                                "ci_low": float(np.percentile(finite, 2.5)),
                                "ci_high": float(np.percentile(finite, 97.5)),
                                "n_bootstrap": len(finite),
                            }
                        )
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame = frame.rename(
        columns={
            "lower": "ci_low",
            "upper": "ci_high",
            "lower_ci": "ci_low",
            "upper_ci": "ci_high",
            "bootstrap_value": "value",
        }
    )
    if "method" not in frame:
        frame["method"] = "learned"
    if "metric" not in frame:
        wide_rows = []
        identity = [name for name in ("contrast", "rank", "method", "fold") if name in frame]
        for metric in metrics:
            if metric not in frame:
                continue
            for _, row in frame.iterrows():
                item = {name: row[name] for name in identity}
                item.update(
                    {
                        "metric": metric,
                        "mean": row[metric],
                        "ci_low": row.get(metric + "_ci_low", np.nan),
                        "ci_high": row.get(metric + "_ci_high", np.nan),
                    }
                )
                wide_rows.append(item)
        frame = pd.DataFrame(wide_rows)
    if "value" in frame and {"contrast", "rank", "metric"}.issubset(frame.columns):
        rows = []
        for key, group in frame.groupby(["contrast", "rank", "method", "metric"], dropna=False):
            values = pd.to_numeric(group.value, errors="coerce").dropna().to_numpy(float)
            if len(values):
                rows.append(
                    {
                        "contrast": key[0],
                        "rank": key[1],
                        "method": key[2],
                        "metric": key[3],
                        "mean": float(np.mean(values)),
                        "ci_low": float(np.percentile(values, 2.5)),
                        "ci_high": float(np.percentile(values, 97.5)),
                        "n_bootstrap": len(values),
                    }
                )
        frame = pd.DataFrame(rows)
    required = {"contrast", "rank", "metric", "ci_low", "ci_high"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    return numeric(frame, ("rank", "mean", "ci_low", "ci_high"))


def apply_bootstrap_intervals(summary: pd.DataFrame, bootstrap: pd.DataFrame, metric: str) -> pd.DataFrame:
    if summary.empty or bootstrap.empty:
        return summary
    result = summary.copy()
    source = bootstrap.loc[
        bootstrap.metric.eq(metric) & bootstrap.method.eq("learned")
    ]
    for index, row in result.loc[result.method.eq("learned")].iterrows():
        match = source.loc[
            source.contrast.eq(row.contrast)
            & pd.to_numeric(source["rank"], errors="coerce").eq(int(row["rank"]))
        ]
        if match.empty:
            continue
        low = pd.to_numeric(match.ci_low, errors="coerce").dropna()
        high = pd.to_numeric(match.ci_high, errors="coerce").dropna()
        if len(low) and len(high):
            result.loc[index, "ci_low"] = float(low.iloc[0])
            result.loc[index, "ci_high"] = float(high.iloc[0])
            result.loc[index, "ci_source"] = "crossed bootstrap"
            if "mean" in match and np.isfinite(pd.to_numeric(match["mean"], errors="coerce").iloc[0]):
                result.loc[index, "mean"] = float(pd.to_numeric(match["mean"], errors="coerce").iloc[0])
    return result


def figure1(
    rank_table: pd.DataFrame,
    baseline_table: pd.DataFrame,
    bootstrap_table: pd.DataFrame,
    figure_dir: Path,
    data_dir: Path,
) -> tuple[list[str], dict[str, Any]]:
    needed = {
        "stage",
        "contrast",
        "fold",
        "rank",
        "method",
        "map_r2_sufficiency",
        "map_r2_necessity",
        "ssi_fraction_transferred",
        "ssi_fraction_removed",
    }
    require_columns(rank_table, needed, "rank_summary.csv")
    require_columns(baseline_table, ("stage", "contrast", "rank", "method"), "baseline_results.csv")
    rank_table = numeric(rank_table, ("fold", "rank"))
    stage = _preferred_stage(rank_table)
    points = rank_table.loc[rank_table.stage.eq(stage)].copy()
    if not all(bool(points.contrast.eq(key).any()) for key in CONTRAST_ORDER):
        missing = [key for key in CONTRAST_ORDER if not bool(points.contrast.eq(key).any())]
        raise DataUnavailable(f"Figure 1 lacks {stage} rows for: {', '.join(missing)}")
    metrics = (
        ("map_r2_sufficiency", "Sufficiency map recovery, $R^2$"),
        ("map_r2_necessity", "Necessity map recovery, $R^2$"),
        ("ssi", "Exact SSI transfer fraction"),
    )
    plot_rows: list[pd.DataFrame] = []
    fig, axes = plt.subplots(3, 3, figsize=(10.6, 7.25), sharex=True)
    for column, contrast in enumerate(CONTRAST_ORDER):
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, column]
            metric_names = (
                ("ssi_fraction_transferred", "sufficiency"),
                ("ssi_fraction_removed", "necessity"),
            ) if metric == "ssi" else ((metric, None),)
            for source_metric, transfer_kind in metric_names:
                summary = _summarize_curve(points, source_metric)
                summary = apply_bootstrap_intervals(summary, bootstrap_table, source_metric)
                summary = summary.loc[summary.contrast.eq(contrast)].copy()
                summary["transfer_kind"] = transfer_kind or "map"
                plot_rows.append(summary)
                random = _random_summary(baseline_table, stage, source_metric)
                random = random.loc[random.contrast.eq(contrast)].copy()
                random["transfer_kind"] = transfer_kind or "map"
                plot_rows.append(random)

                if metric == "ssi":
                    for method in METHOD_STYLE:
                        frame = summary.loc[summary.method.eq(method)].sort_values("rank")
                        if frame.empty:
                            continue
                        color, marker, _, method_label = METHOD_STYLE[method]
                        linestyle = "-" if transfer_kind == "sufficiency" else ":"
                        marker = marker if transfer_kind == "sufficiency" else "x"
                        ax.plot(
                            frame["rank"],
                            frame["mean"],
                            color=color,
                            linestyle=linestyle,
                            marker=marker,
                            markersize=3.5,
                            linewidth=1.35 if method == "learned" else 0.9,
                            zorder=5 if method == "learned" else 3,
                        )
                    if not random.empty:
                        random = random.sort_values("rank")
                        alpha = 0.16 if transfer_kind == "sufficiency" else 0.08
                        ax.fill_between(
                            random["rank"], random["ci_low"], random["ci_high"],
                            color="#AEB5BA", alpha=alpha, linewidth=0, zorder=1,
                        )
                else:
                    for method in METHOD_STYLE:
                        _plot_method_curve(ax, summary, method)
                    if not random.empty:
                        random = random.sort_values("rank")
                        ax.fill_between(
                            random["rank"], random["ci_low"], random["ci_high"],
                            color="#AEB5BA", alpha=0.30, linewidth=0, zorder=1,
                        )
                        ax.plot(random["rank"], random["mean"], color="#8C969D", lw=0.7)

            identity_metric = "ssi_fraction_transferred" if metric == "ssi" else metric
            identity = points.loc[
                points.contrast.eq(contrast)
                & points.method.eq("identity")
                & pd.to_numeric(points[identity_metric], errors="coerce").notna()
            ]
            if not identity.empty:
                y_identity = float(pd.to_numeric(identity[identity_metric], errors="coerce").mean())
                ax.scatter([128], [y_identity], marker="*", s=34, color="#111111", zorder=7)
                ax.annotate(
                    "identity", (128, y_identity), xytext=(-3, 6), textcoords="offset points",
                    ha="right", fontsize=6.2, color="#111111",
                )
            if row == 0:
                ax.set_title(CONTRAST_LABEL[contrast], color=CONTRAST_COLOR[contrast], fontweight="semibold")
            if column == 0:
                ax.set_ylabel(ylabel)
            if row == 2:
                ax.set_xlabel("Subspace rank")
            ax.axhline(0, color="#C5C9CC", lw=0.65, zorder=0)
            ax.axhline(1, color="#D8DADC", lw=0.55, linestyle="--", zorder=0)
            ax.set_xscale("symlog", linthresh=1, linscale=0.7, base=2)
            ax.set_xlim(0, 165)
            ax.set_xticks(RANK_TICKS)
            ax.set_xticklabels([str(value) for value in RANK_TICKS])
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#ECEEEF", linewidth=0.55, zorder=0)
            if metric == "ssi" and column == 0:
                ax.text(
                    0.02, 0.98, "solid ○ sufficient\ndotted × necessary",
                    transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color="#3C4449",
                )

    fig.suptitle(
        "Is the movement-dependent RR100 map transformation low-dimensional?",
        fontsize=12,
        fontweight="semibold",
        y=1.005,
    )
    learned_folds = pd.to_numeric(
        points.loc[points.method.eq("learned"), "fold"], errors="coerce"
    ).nunique()
    evidence_label = (
        "four-fold crossed test"
        if stage == "crossval" and learned_folds == 4
        else "predeclared screening fold only"
    )
    fig.text(
        0.995,
        0.008,
        f"{evidence_label}; "
        "points/lines are held-out; random shading is the 2.5–97.5% interval",
        ha="right",
        va="bottom",
        fontsize=6.6,
        color="#50575C",
    )
    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([value for value in plot_rows if not value.empty], ignore_index=True).to_csv(
        data_dir / "figure1_rank_curves.csv", index=False
    )
    points.to_csv(data_dir / "figure1_heldout_fold_points.csv", index=False)
    outputs = save_figure(fig, figure_dir / "causal_subspace_figure1_rank_curves")
    return outputs, {
        "stage": stage,
        "heldout_folds": int(learned_folds),
        "evidence_label": evidence_label,
        "n_fold_rows": int(len(points)),
        "crossed_bootstrap_intervals_used": bool(
            not bootstrap_table.empty
            and bool(pd.concat([value for value in plot_rows if not value.empty], ignore_index=True)
                     .ci_source.eq("crossed bootstrap").any())
        ),
    }


def _decode_strings(values: np.ndarray) -> list[str]:
    return [item.decode() if isinstance(item, bytes) else str(item) for item in values.tolist()]


def load_objective_maps(path: Path) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    if not path.is_file():
        raise DataUnavailable(f"objective held-out maps are unavailable: {path}")
    with np.load(path, allow_pickle=False) as archive:
        for name in ("maps", "map_names", "metadata_json"):
            if name not in archive:
                raise DataUnavailable(f"objective maps lack {name!r}: {path}")
        maps = np.asarray(archive["maps"], dtype=np.float64)
        names = _decode_strings(np.asarray(archive["map_names"]))
        raw = np.asarray(archive["metadata_json"]).item()
        metadata = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    if maps.ndim != 4 or len(metadata) != len(maps) or maps.shape[1] != len(names):
        raise DataUnavailable(
            f"objective map/metadata dimensions disagree: maps={maps.shape}, names={len(names)}, metadata={len(metadata)}"
        )
    return maps, names, metadata


def exact_ssi_from_gain(gain: np.ndarray) -> float:
    value = np.asarray(gain, dtype=np.float64)
    return float(np.mean(value * np.log2(value + 1e-8)))


def safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > EPS else float("nan")


def map_r2(prediction: np.ndarray, target: np.ndarray, baseline: np.ndarray) -> float:
    denominator = float(np.square(target - baseline).sum())
    return 1.0 - safe_fraction(float(np.square(prediction - target).sum()), denominator)


def _objective_row(
    metadata: list[dict[str, Any]],
    rank_table: pd.DataFrame,
    contrast: str,
    rank: int,
) -> int:
    candidates = [
        index
        for index, row in enumerate(metadata)
        if str(row.get("contrast")) == contrast
        and int(row.get("rank", -1)) == int(rank)
        and str(row.get("stage", "crossval")) == "crossval"
    ]
    if not candidates:
        raise DataUnavailable(f"No crossed held-out objective map for {contrast}, rank {rank}")
    folds = {int(metadata[index].get("fold", -1)): index for index in candidates}
    performance = rank_table.loc[
        rank_table.contrast.eq(contrast)
        & rank_table.stage.eq("crossval")
        & rank_table.method.eq("learned")
        & pd.to_numeric(rank_table["rank"], errors="coerce").eq(rank)
    ].copy()
    if performance.empty:
        return min(candidates)
    joint = 0.5 * (
        pd.to_numeric(performance["map_r2_sufficiency"], errors="coerce")
        + pd.to_numeric(performance["map_r2_necessity"], errors="coerce")
    )
    target = float(np.nanmedian(joint))
    ordered = np.argsort(np.abs(joint.to_numpy(float) - target), kind="stable")
    for ordinal in ordered:
        fold = int(performance.iloc[int(ordinal)].fold)
        if fold in folds:
            return folds[fold]
    return min(candidates)


def figure2(
    rank_table: pd.DataFrame,
    maps: np.ndarray,
    names: list[str],
    metadata: list[dict[str, Any]],
    ranks: dict[str, int],
    figure_dir: Path,
    data_dir: Path,
) -> tuple[list[str], dict[str, Any]]:
    missing_ranks = [key for key in CONTRAST_ORDER if key not in ranks]
    if missing_ranks:
        raise DataUnavailable(
            "Figure 2 requires explicitly selected ranks for: " + ", ".join(missing_ranks)
        )
    required_names = (
        "representative_gain_a",
        "representative_gain_b",
        "representative_target_minus_baseline",
        "representative_gain_sufficiency",
        "representative_sufficiency_residual",
        "representative_gain_necessity",
    )
    absent = sorted(set(required_names) - set(names))
    if absent:
        raise DataUnavailable("Objective map archive lacks: " + ", ".join(absent))
    name_to_index = {name: index for index, name in enumerate(names)}
    chosen = [_objective_row(metadata, rank_table, key, ranks[key]) for key in CONTRAST_ORDER]
    payload: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    rendered: list[list[np.ndarray]] = []
    normalized_maps: list[np.ndarray] = []
    difference_maps: list[np.ndarray] = []
    for contrast, archive_row in zip(CONTRAST_ORDER, chosen):
        source = {name: maps[archive_row, name_to_index[name]] for name in required_names}
        gain_a = source["representative_gain_a"]
        gain_b = source["representative_gain_b"]
        gain_s = source["representative_gain_sufficiency"]
        gain_n = source["representative_gain_necessity"]
        target_delta = gain_b - gain_a
        suff_delta = gain_s - gain_a
        residual = gain_b - gain_s
        row_maps = [gain_a, gain_b, target_delta, suff_delta, residual, gain_n]
        rendered.append(row_maps)
        normalized_maps.extend((gain_a, gain_b, gain_s, gain_n))
        difference_maps.extend((target_delta, suff_delta, residual))
        ssi_a, ssi_b = exact_ssi_from_gain(gain_a), exact_ssi_from_gain(gain_b)
        ssi_s, ssi_n = exact_ssi_from_gain(gain_s), exact_ssi_from_gain(gain_n)
        effect = ssi_b - ssi_a
        record = {
            "contrast": contrast,
            "rank": ranks[contrast],
            "archive_row": archive_row,
            **metadata[archive_row],
            "map_r2_sufficiency": map_r2(gain_s, gain_b, gain_a),
            "map_r2_necessity": map_r2(gain_n, gain_a, gain_b),
            "ssi_a_bits": ssi_a,
            "ssi_b_bits": ssi_b,
            "ssi_sufficiency_bits": ssi_s,
            "ssi_necessity_bits": ssi_n,
            "ssi_fraction_transferred": safe_fraction(ssi_s - ssi_a, effect),
            "ssi_fraction_removed": safe_fraction(ssi_b - ssi_n, effect),
            "ssi_recomputed_from_saved_gain_maps": True,
        }
        records.append(record)
        prefix = contrast + "__"
        for label, value in zip(
            ("baseline", "target", "target_delta", "sufficient_delta", "residual", "necessity"),
            row_maps,
        ):
            payload[prefix + label] = np.asarray(value, dtype=np.float32)

    all_norm = np.concatenate([value.ravel() for value in normalized_maps])
    vmin, vmax = float(np.nanmin(all_norm)), float(np.nanmax(all_norm))
    dmax = float(np.nanmax(np.abs(np.concatenate([value.ravel() for value in difference_maps]))))
    dmax = max(dmax, 1e-12)
    seq_norm = Normalize(vmin=vmin, vmax=vmax)
    diff_norm = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)
    fig, axes = plt.subplots(3, 6, figsize=(12.8, 6.75))
    column_titles = (
        "Baseline $g_A$",
        "Target $g_B$",
        "Complete $g_B-g_A$",
        "Sufficient $g_{suff}-g_A$",
        "Residual $g_B-g_{suff}$",
        "After necessity removal $g_{nec}$",
    )
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=8.5, pad=5)
    for row, (contrast, record, row_maps) in enumerate(zip(CONTRAST_ORDER, records, rendered)):
        for column, value in enumerate(row_maps):
            is_difference = column in (2, 3, 4)
            axes[row, column].imshow(
                value,
                origin="lower",
                cmap="RdBu_r" if is_difference else "viridis",
                norm=diff_norm if is_difference else seq_norm,
                interpolation="nearest",
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            for spine in axes[row, column].spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(
            CONTRAST_LABEL[contrast] + f"\nrank {ranks[contrast]}, fold {int(record.get('fold', -1))}",
            color=CONTRAST_COLOR[contrast],
            fontweight="semibold",
            rotation=0,
            ha="right",
            va="center",
            labelpad=8,
        )
        annotations = (
            f"SSI {record['ssi_a_bits']:.3f}",
            f"SSI {record['ssi_b_bits']:.3f}",
            f"ΔSSI {record['ssi_b_bits'] - record['ssi_a_bits']:+.3f}",
            f"$R^2_s$ {record['map_r2_sufficiency']:.2f}\nSSI {record['ssi_sufficiency_bits']:.3f}; $F^{{SSI}}_s$ {record['ssi_fraction_transferred']:.2f}",
            "target − sufficient",
            f"$R^2_n$ {record['map_r2_necessity']:.2f}\nSSI {record['ssi_necessity_bits']:.3f}; $F^{{SSI}}_n$ {record['ssi_fraction_removed']:.2f}",
        )
        for column, text in enumerate(annotations):
            axes[row, column].text(
                0.5,
                -0.055,
                text,
                transform=axes[row, column].transAxes,
                ha="center",
                va="top",
                fontsize=6.6,
                color="#343A3E",
                linespacing=1.0,
            )

    seq_cax = fig.add_axes((0.20, 0.092, 0.28, 0.014))
    seq_bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=seq_norm, cmap="viridis"),
        cax=seq_cax,
        orientation="horizontal",
    )
    seq_bar.set_label("Mean-normalized rate, $g=r/\\bar r$ (one scale across all normalized maps)")
    diff_cax = fig.add_axes((0.61, 0.092, 0.28, 0.014))
    diff_bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=diff_norm, cmap="RdBu_r"),
        cax=diff_cax,
        orientation="horizontal",
    )
    diff_bar.set_label("Change in mean-normalized rate (one symmetric scale across all difference maps)")
    fig.suptitle(
        "What spatial transformation does the retained ConvGRU subspace carry?",
        fontsize=12,
        fontweight="semibold",
        y=1.01,
    )
    complete_crossed = all(
        pd.to_numeric(
            rank_table.loc[
                rank_table.contrast.eq(contrast)
                & rank_table.stage.eq("crossval")
                & rank_table.method.eq("learned")
                & pd.to_numeric(rank_table["rank"], errors="coerce").eq(ranks[contrast]),
                "fold",
            ],
            errors="coerce",
        ).nunique()
        == 4
        for contrast in CONTRAST_ORDER
    )
    selection_note = (
        "Each held-out example is selected objectively from the fold nearest median crossed-test performance; no map is independently rescaled."
        if complete_crossed
        else "Predeclared screening fold only; each example is the evaluator-predeclared median record within that fold; no map is independently rescaled."
    )
    fig.text(
        0.995,
        0.002,
        selection_note,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#50575C",
    )
    fig.subplots_adjust(left=0.12, right=0.995, top=0.91, bottom=0.205, wspace=0.08, hspace=0.27)
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(data_dir / "figure2_objective_example_metadata.csv", index=False)
    np.savez_compressed(data_dir / "figure2_objective_maps.npz", **payload)
    outputs = save_figure(fig, figure_dir / "causal_subspace_figure2_objective_maps")
    return outputs, {
        "selected_rows": chosen,
        "normalized_map_limits": [vmin, vmax],
        "difference_map_limits": [-dmax, dmax],
        "selection_rule": (
            "fold nearest median joint sufficiency/necessity held-out recovery; evaluator-predeclared median record within fold"
            if complete_crossed
            else "predeclared screening fold; evaluator-predeclared median record within fold"
        ),
        "four_fold_crossed_complete": bool(complete_crossed),
    }


def _dose_points(frame: pd.DataFrame, contrast: str, rank: int) -> pd.DataFrame:
    subset = frame.loc[
        frame.contrast.eq(contrast)
        & frame.method.eq("learned")
        & pd.to_numeric(frame["rank"], errors="coerce").eq(rank)
    ].copy()
    if subset.empty:
        raise DataUnavailable(f"No cross-scale rows for {contrast}, rank {rank}")
    needed = (
        "fold",
        "scale_a",
        "scale_b",
        "ssi_a_bits",
        "ssi_b_bits",
        "ssi_sufficiency_bits",
        "ssi_necessity_bits",
    )
    require_columns(subset, needed, "cross_scale_results.csv")
    rows = []
    for fold, fold_frame in subset.groupby("fold"):
        first = fold_frame.iloc[0]
        rows.append(
            {
                "contrast": contrast,
                "rank": rank,
                "fold": int(fold),
                "scale": float(first.scale_a),
                "condition": "intact",
                "ssi_bits": float(first.ssi_a_bits),
            }
        )
        for condition, column in (
            ("sufficient", "ssi_sufficiency_bits"),
            ("necessity", "ssi_necessity_bits"),
        ):
            rows.append(
                {
                    "contrast": contrast,
                    "rank": rank,
                    "fold": int(fold),
                    "scale": float(first.scale_a),
                    "condition": condition,
                    "ssi_bits": float(first.ssi_a_bits),
                }
            )
        for _, item in fold_frame.iterrows():
            for condition, column in (
                ("intact", "ssi_b_bits"),
                ("sufficient", "ssi_sufficiency_bits"),
                ("necessity", "ssi_necessity_bits"),
            ):
                rows.append(
                    {
                        "contrast": contrast,
                        "rank": rank,
                        "fold": int(fold),
                        "scale": float(item.scale_b),
                        "condition": condition,
                        "ssi_bits": float(item[column]),
                    }
                )
    result = pd.DataFrame(rows).drop_duplicates(
        ["contrast", "rank", "fold", "scale", "condition"], keep="first"
    )
    expected = {0.0, 0.5, 1.0, 2.0, 3.0}
    for fold, fold_frame in result.groupby("fold"):
        observed = set(np.round(fold_frame.loc[fold_frame.condition.eq("intact"), "scale"], 6))
        if observed != expected:
            raise DataUnavailable(
                f"Incomplete intact dose curve for {contrast}, fold {fold}: observed {sorted(observed)}"
            )
    return result


def _plot_dose(ax: plt.Axes, dose: pd.DataFrame, contrast: str) -> None:
    color = CONTRAST_COLOR[contrast]
    styles = {
        "intact": ("#22282C", "o", "-", "intact"),
        "sufficient": (color, "o", "--", "sufficient"),
        "necessity": ("#59717C", "s", ":", "after necessity removal"),
    }
    for condition, (line_color, marker, linestyle, label) in styles.items():
        frame = dose.loc[dose.condition.eq(condition)]
        for _, fold in frame.groupby("fold"):
            fold = fold.sort_values("scale")
            ax.plot(fold.scale, fold.ssi_bits, color=line_color, lw=0.55, alpha=0.20)
        summary = frame.groupby("scale", as_index=False).ssi_bits.mean().sort_values("scale")
        ax.plot(
            summary.scale,
            summary.ssi_bits,
            color=line_color,
            marker=marker,
            markerfacecolor="white" if condition == "sufficient" else line_color,
            markeredgewidth=0.8,
            markersize=4.2,
            linestyle=linestyle,
            linewidth=1.55,
            zorder=4,
        )
        ax.annotate(
            label,
            (float(summary.scale.iloc[-1]), float(summary.ssi_bits.iloc[-1])),
            xytext=(5, {"intact": 7, "sufficient": 0, "necessity": -7}[condition]),
            textcoords="offset points",
            color=line_color,
            fontsize=6.5,
            va="center",
            clip_on=False,
        )
    ax.set_xticks((0, 0.5, 1, 2, 3))
    ax.set_xlabel("Movement amplitude (× measured FEM)")
    ax.set_ylabel("Exact population SSI (bits/spike)")
    ax.set_title(CONTRAST_LABEL[contrast] + " subspace", color=color, fontweight="semibold")
    ax.grid(axis="y", color="#ECEEEF", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)


def figure3(
    cross_scale: pd.DataFrame,
    per_unit: pd.DataFrame,
    ranks: dict[str, int],
    axes_stable: bool,
    figure_dir: Path,
    data_dir: Path,
) -> tuple[list[str], dict[str, Any]]:
    required_contrasts = ("low_0_to_2", "high_0_to_1")
    missing_ranks = [key for key in required_contrasts if key not in ranks]
    if missing_ranks:
        raise DataUnavailable("Figure 3 requires selected ranks for: " + ", ".join(missing_ranks))
    require_columns(cross_scale, ("contrast", "rank", "method"), "cross_scale_results.csv")
    dose = pd.concat(
        [_dose_points(cross_scale, key, ranks[key]) for key in required_contrasts],
        ignore_index=True,
    )
    needed_unit = (
        "stage",
        "contrast",
        "fold",
        "rank",
        "method",
        "unit_index",
        "sf_split_metric",
        "map_r2_sufficiency",
        "map_r2_necessity",
    )
    require_columns(per_unit, needed_unit, "per_unit_results.csv")
    selected_unit_rows = []
    for contrast in CONTRAST_ORDER:
        if contrast not in ranks:
            continue
        selected_unit_rows.append(
            per_unit.loc[
                per_unit.stage.eq("crossval")
                & per_unit.contrast.eq(contrast)
                & per_unit.method.eq("learned")
                & pd.to_numeric(per_unit["rank"], errors="coerce").eq(ranks[contrast])
            ].copy()
        )
    unit_points = pd.concat(selected_unit_rows, ignore_index=True)
    if unit_points.empty:
        raise DataUnavailable("No selected-rank crossed per-unit results are available")
    for name in ("sf_split_metric", "map_r2_sufficiency", "map_r2_necessity"):
        unit_points[name] = pd.to_numeric(unit_points[name], errors="coerce")
    unit_summary = (
        unit_points.groupby(["contrast", "rank", "unit_index"], as_index=False)
        .agg(
            sf_split_metric=("sf_split_metric", "first"),
            map_r2_sufficiency=("map_r2_sufficiency", "mean"),
            map_r2_necessity=("map_r2_necessity", "mean"),
            n_folds=("fold", "nunique"),
        )
    )

    fig = plt.figure(figsize=(10.8, 7.0))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.08), hspace=0.42, wspace=0.28)
    _plot_dose(fig.add_subplot(grid[0, 0]), dose.loc[dose.contrast.eq("low_0_to_2")], "low_0_to_2")
    _plot_dose(fig.add_subplot(grid[0, 1]), dose.loc[dose.contrast.eq("high_0_to_1")], "high_0_to_1")
    for column, (metric, title) in enumerate(
        (
            ("map_r2_sufficiency", "Per-unit sufficient map recovery"),
            ("map_r2_necessity", "Per-unit necessary map recovery"),
        )
    ):
        ax = fig.add_subplot(grid[1, column])
        for contrast in CONTRAST_ORDER:
            frame = unit_summary.loc[unit_summary.contrast.eq(contrast)]
            if frame.empty:
                continue
            ax.scatter(
                frame.sf_split_metric,
                frame[metric],
                s=10,
                facecolors="none",
                edgecolors=CONTRAST_COLOR[contrast],
                linewidths=0.65,
                alpha=0.62,
            )
            finite = frame.loc[np.isfinite(frame.sf_split_metric) & np.isfinite(frame[metric])]
            if len(finite) >= 3:
                order = np.argsort(finite.sf_split_metric.to_numpy(float))
                x = finite.sf_split_metric.to_numpy(float)[order]
                y = finite[metric].to_numpy(float)[order]
                window = max(3, int(math.ceil(len(y) / 7)))
                smooth = pd.Series(y).rolling(window, center=True, min_periods=2).median().to_numpy()
                ax.plot(x, smooth, color=CONTRAST_COLOR[contrast], lw=1.2)
        ax.axhline(0, color="#AEB4B8", lw=0.7)
        ax.axvline(0.5, color="#C5C9CC", lw=0.7, linestyle="--")
        ax.set_xlabel("Historical SF split metric")
        ax.set_ylabel("Held-out map-effect recovery, $R^2$")
        ax.set_title(title, fontweight="semibold")
        ax.grid(axis="y", color="#ECEEEF", linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        if column == 1:
            for offset, contrast in enumerate(CONTRAST_ORDER):
                if bool(unit_summary.contrast.eq(contrast).any()):
                    ax.text(
                        1.01,
                        0.98 - 0.075 * offset,
                        CONTRAST_SHORT[contrast],
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=6.5,
                        color=CONTRAST_COLOR[contrast],
                    )

    fig.suptitle(
        "How do RR100 populations read the retained causal state?",
        fontsize=12,
        fontweight="semibold",
        y=0.99,
    )
    heldout_folds = int(pd.to_numeric(dose["fold"], errors="coerce").nunique())
    evidence_prefix = (
        "Four-fold crossed test. "
        if heldout_folds == 4
        else "Predeclared screening fold only. "
    )
    fig.text(
        0.995,
        0.006,
        (
            evidence_prefix
            + "Stable canonical axes were declared, but this conservative panel still reports only subspace-level effects."
            if axes_stable
            else evidence_prefix
            + "Canonical axes are not interpreted: the panel reports only subspace-level causal recovery."
        ),
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#50575C",
    )
    fig.subplots_adjust(left=0.085, right=0.90, top=0.91, bottom=0.10)
    data_dir.mkdir(parents=True, exist_ok=True)
    dose.to_csv(data_dir / "figure3_ssi_dose_curves.csv", index=False)
    unit_summary.to_csv(data_dir / "figure3_per_unit_recovery_vs_sf.csv", index=False)
    outputs = save_figure(fig, figure_dir / "causal_subspace_figure3_population_readout")
    return outputs, {
        "selected_ranks": {key: ranks[key] for key in ranks if key in CONTRAST_ORDER},
        "canonical_axes_explicitly_stable": bool(axes_stable),
        "latent_axes_shown": False,
        "n_per_unit_summary_rows": int(len(unit_summary)),
        "heldout_folds": heldout_folds,
    }


def main() -> int:
    args = parse_args()
    configure_style()
    source = args.input_dir.resolve()
    inputs = input_paths(source)
    figure_dir = (args.output_dir or source / "figures").resolve()
    data_dir = (args.figure_data_dir or source / "figure_data").resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    overrides = parse_rank_overrides(args.selected_rank)
    statistics = read_statistics(inputs.statistics)
    statuses: dict[str, Any] = {}
    all_outputs: list[str] = []

    try:
        rank_table = read_csv(inputs.rank_summary, "rank summary")
    except DataUnavailable as error:
        rank_table = pd.DataFrame()
        statuses["figure1"] = {"status": "unavailable", "reason": str(error)}
        statuses["figure2"] = {"status": "unavailable", "reason": str(error)}
        statuses["figure3"] = {"status": "unavailable", "reason": str(error)}
    else:
        chosen_ranks = selected_ranks(statistics, rank_table, overrides)
        try:
            baselines = read_csv(inputs.baseline_results, "baseline results")
            bootstrap = load_bootstrap_summary(inputs.bootstrap_results)
            outputs, detail = figure1(rank_table, baselines, bootstrap, figure_dir, data_dir)
            statuses["figure1"] = {"status": "complete", "outputs": outputs, **detail}
            all_outputs.extend(outputs)
        except DataUnavailable as error:
            statuses["figure1"] = {"status": "unavailable", "reason": str(error)}

        try:
            maps, names, metadata = load_objective_maps(inputs.objective_maps)
            outputs, detail = figure2(
                rank_table, maps, names, metadata, chosen_ranks, figure_dir, data_dir
            )
            statuses["figure2"] = {"status": "complete", "outputs": outputs, **detail}
            all_outputs.extend(outputs)
        except DataUnavailable as error:
            statuses["figure2"] = {"status": "unavailable", "reason": str(error)}

        try:
            cross_scale = read_csv(inputs.cross_scale_results, "cross-scale results")
            per_unit = read_csv(inputs.per_unit_results, "per-unit results")
            outputs, detail = figure3(
                cross_scale,
                per_unit,
                chosen_ranks,
                canonical_axes_stable(statistics),
                figure_dir,
                data_dir,
            )
            statuses["figure3"] = {"status": "complete", "outputs": outputs, **detail}
            all_outputs.extend(outputs)
        except DataUnavailable as error:
            statuses["figure3"] = {"status": "unavailable", "reason": str(error)}

    manifest = {
        "analysis": "causal_low_rank_saved_products_plotting",
        "saved_products_only": True,
        "model_or_optimizer_imported": False,
        "negative_recovery_or_transfer_clipped": False,
        "individual_map_rescaling": False,
        "figure2_ssi_source": "recomputed exactly from saved mean-normalized gain maps",
        "input_root": str(source),
        "input_files": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in vars(inputs).items()
            if name != "root"
        },
        "selected_rank_overrides": overrides,
        "statuses": statuses,
        "outputs": all_outputs,
    }
    (figure_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, status in statuses.items():
        print(f"{name}: {status['status']}" + (f" — {status['reason']}" if status['status'] != "complete" else ""))
    if args.strict and any(value["status"] != "complete" for value in statuses.values()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
