#!/usr/bin/env python3
"""Render the final paired Ryan-versus-M66 population comparison.

The validation panel aligns units by ``(session, cid)``.  The FixRSVP panels
align the exact Figure-3 population by ``(session, source unit index)`` and
verify that Ryan and M66 were scored against identical observations.  Median
confidence intervals use a paired hierarchical bootstrap that resamples
sessions and then neurons within each sampled session.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RYAN_VAL = (
    ROOT / "outputs/dekel240_evaluation/Ryan_05_lr5e-4_epoch471_val_full.json"
)
DEFAULT_M66_VAL = ROOT / "outputs/dekel240_evaluation/M66a_epoch31_val_full.json"
DEFAULT_RYAN_FIX = ROOT / "outputs/cache/fig3_digitaltwin.pkl"
DEFAULT_M66_FIX = (
    ROOT / "outputs/dekel240_paper/final/cache_fig3/fig3_digitaltwin.pkl"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/dekel240_evaluation/M66_vs_Ryan_final_comparison"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ryan-val", type=Path, default=DEFAULT_RYAN_VAL)
    parser.add_argument("--m66-val", type=Path, default=DEFAULT_M66_VAL)
    parser.add_argument("--ryan-fix", type=Path, default=DEFAULT_RYAN_FIX)
    parser.add_argument("--m66-fix", type=Path, default=DEFAULT_M66_FIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def _resolve_npz(report_path: Path, report: dict) -> Path:
    path = Path(report["per_unit_bps_npz"])
    if not path.is_absolute():
        path = report_path.parent / path
    return path


def load_validation(reference_path: Path, candidate_path: Path):
    reference_report = json.loads(reference_path.read_text())
    candidate_report = json.loads(candidate_path.read_text())
    if reference_report["split"] != "val" or candidate_report["split"] != "val":
        raise RuntimeError("Both in-distribution reports must use the validation split")

    rows = []
    with (
        np.load(_resolve_npz(reference_path, reference_report), allow_pickle=False) as ref,
        np.load(_resolve_npz(candidate_path, candidate_report), allow_pickle=False) as cand,
    ):
        ref_names = ref["session_names"].astype(str)
        cand_names = cand["session_names"].astype(str)
        if not np.array_equal(ref_names, cand_names):
            raise RuntimeError("Validation session order differs")
        for index, session in enumerate(ref_names):
            ref_cids = ref[f"cids_{index}"].astype(np.int64)
            cand_cids = cand[f"cids_{index}"].astype(np.int64)
            if not np.array_equal(ref_cids, cand_cids):
                raise RuntimeError(f"Validation cids differ for {session}")
            ref_bps = ref[f"bps_{index}"].astype(np.float64)
            cand_bps = cand[f"bps_{index}"].astype(np.float64)
            valid = np.isfinite(ref_bps) & np.isfinite(cand_bps)
            for cid, left, right in zip(
                ref_cids[valid], ref_bps[valid], cand_bps[valid], strict=True
            ):
                rows.append(
                    {
                        "metric": "validation_bps",
                        "session": str(session),
                        "unit_id": int(cid),
                        "ryan": float(left),
                        "m66": float(right),
                    }
                )
    return pd.DataFrame(rows), reference_report, candidate_report


def _load_fix_cache(path: Path) -> dict[str, dict]:
    with path.open("rb") as stream:
        values = dill.load(stream)
    return {str(value["session"]): value for value in values}


def load_fixrsvp(reference_path: Path, candidate_path: Path) -> pd.DataFrame:
    reference = _load_fix_cache(reference_path)
    candidate = _load_fix_cache(candidate_path)
    if set(reference) != set(candidate):
        raise RuntimeError("FixRSVP session sets differ")

    rows = []
    for session in sorted(reference):
        left = reference[session]
        right = candidate[session]
        ref_units = np.asarray(left["neuron_mask"], dtype=np.int64)
        cand_units = np.asarray(right["neuron_mask"], dtype=np.int64)
        if not np.array_equal(ref_units, cand_units):
            raise RuntimeError(f"FixRSVP neuron masks differ for {session}")
        ref_robs = np.asarray(left["robs_used"])
        cand_robs = np.asarray(right["robs_used"])
        if not np.array_equal(ref_robs, cand_robs, equal_nan=True):
            raise RuntimeError(f"FixRSVP observations differ for {session}")

        # Recompute the paired CCnorm values with one shared, current CCmax.
        # Ryan's historical cache predates the present seeded 2 x 500-split
        # estimator, so its stored Monte-Carlo CCmax differs slightly even
        # though the underlying observations are identical.  M66's production
        # cache uses the current estimator and has been independently reproduced
        # from the canonical observations.  Using that data-only denominator for
        # both models makes the comparison exactly paired and also avoids a
        # model-dependent "unstable CCnorm" exclusion.
        ref_ccabs = np.asarray(left["ccabs"], dtype=np.float64)
        cand_ccabs = np.asarray(right["ccabs"], dtype=np.float64)
        shared_ccmax = np.asarray(right["ccmax"], dtype=np.float64)
        valid = (
            np.isfinite(ref_ccabs)
            & np.isfinite(cand_ccabs)
            & np.isfinite(shared_ccmax)
            & (shared_ccmax > 1e-3)
            # Keep the exact same cells in both columns while preserving the
            # paper protocol's two-seed stability gate.  The intersection is
            # symmetric; neither model gets a population-composition advantage.
            & np.isfinite(np.asarray(left["ccnorm"], dtype=np.float64))
            & np.isfinite(np.asarray(right["ccnorm"], dtype=np.float64))
        )
        for unit, ref_value, cand_value, ceiling in zip(
            ref_units[valid],
            ref_ccabs[valid] / shared_ccmax[valid],
            cand_ccabs[valid] / shared_ccmax[valid],
            shared_ccmax[valid],
            strict=True,
        ):
            rows.append(
                {
                    "metric": "fixrsvp_ccnorm",
                    "session": session,
                    "unit_id": int(unit),
                    "ryan": float(ref_value),
                    "m66": float(cand_value),
                    "shared_ccmax": float(ceiling),
                }
            )

        for metric, key in {"fixrsvp_single_trial_r2": "ve_model"}.items():
            ref_values = np.asarray(left[key], dtype=np.float64)
            cand_values = np.asarray(right[key], dtype=np.float64)
            valid = np.isfinite(ref_values) & np.isfinite(cand_values)
            for unit, ref_value, cand_value in zip(
                ref_units[valid], ref_values[valid], cand_values[valid], strict=True
            ):
                rows.append(
                    {
                        "metric": metric,
                        "session": session,
                        "unit_id": int(unit),
                        "ryan": float(ref_value),
                        "m66": float(cand_value),
                    }
                )
    return pd.DataFrame(rows)


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def hierarchical_bootstrap(
    table: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    official_bps: bool = False,
) -> dict:
    groups = [
        group[["ryan", "m66"]].to_numpy(np.float64)
        for _, group in table.groupby("session", sort=True)
    ]
    if not groups:
        raise RuntimeError("No paired values to bootstrap")
    observed = table[["ryan", "m66"]].to_numpy(np.float64)
    observed_difference = observed[:, 1] - observed[:, 0]

    rng = np.random.default_rng(seed)
    draws = np.empty((n_bootstrap, 4), dtype=np.float64)
    official_draws = (
        np.empty((n_bootstrap, 3), dtype=np.float64) if official_bps else None
    )
    n_sessions = len(groups)
    for draw_index in range(n_bootstrap):
        session_indices = rng.integers(0, n_sessions, size=n_sessions)
        sampled = []
        session_scores = []
        for session_index in session_indices:
            values = groups[int(session_index)]
            unit_indices = rng.integers(0, len(values), size=len(values))
            resampled = values[unit_indices]
            sampled.append(resampled)
            if official_bps:
                session_scores.append(np.clip(resampled, 0.0, None).mean(axis=0))
        paired = np.concatenate(sampled, axis=0)
        differences = paired[:, 1] - paired[:, 0]
        draws[draw_index] = (
            np.median(paired[:, 0]),
            np.median(paired[:, 1]),
            np.median(differences),
            np.mean(differences > 0),
        )
        if official_bps:
            scores = np.asarray(session_scores).mean(axis=0)
            official_draws[draw_index] = scores[0], scores[1], scores[1] - scores[0]

    names = ("ryan", "m66", "paired_difference", "fraction_m66_greater")
    summary = {
        "n_units": int(len(table)),
        "n_sessions": int(table["session"].nunique()),
        "ryan": {"median": float(np.median(observed[:, 0]))},
        "m66": {"median": float(np.median(observed[:, 1]))},
        "paired_difference": {"median": float(np.median(observed_difference))},
        "fraction_m66_greater": {"estimate": float(np.mean(observed_difference > 0))},
        "bootstrap": {
            "method": "paired hierarchical bootstrap: sessions, then neurons",
            "n_draws": int(n_bootstrap),
            "seed": int(seed),
        },
    }
    for index, name in enumerate(names):
        summary[name]["ci95"] = [
            float(value) for value in np.quantile(draws[:, index], [0.025, 0.975])
        ]
    summary["paired_difference"]["bootstrap_probability_m66_le_ryan"] = float(
        np.mean(draws[:, 2] <= 0)
    )
    if official_bps:
        summary["official_validation_aggregation_bootstrap"] = {
            "method": "zero-clipped neuron mean, then equal-session mean",
            "ryan": {
                "ci95": [
                    float(value)
                    for value in np.quantile(official_draws[:, 0], [0.025, 0.975])
                ]
            },
            "m66": {
                "ci95": [
                    float(value)
                    for value in np.quantile(official_draws[:, 1], [0.025, 0.975])
                ]
            },
            "difference": {
                "ci95": [
                    float(value)
                    for value in np.quantile(official_draws[:, 2], [0.025, 0.975])
                ],
                "bootstrap_probability_m66_le_ryan": float(
                    np.mean(official_draws[:, 2] <= 0)
                ),
            },
        }
    return summary


def _format_value(value: float, metric: str) -> str:
    digits = 3 if metric != "fixrsvp_single_trial_r2" else 4
    return f"{value:.{digits}f}"


def _format_interval(interval: list[float], metric: str) -> str:
    return (
        f"[{_format_value(interval[0], metric)}, "
        f"{_format_value(interval[1], metric)}]"
    )


def render_panel(
    ax,
    table: pd.DataFrame,
    summary: dict,
    *,
    title: str,
    axis_label: str,
    metric: str,
    official_scores: tuple[float, float] | None = None,
) -> None:
    values = table[["ryan", "m66"]].to_numpy(np.float64)
    combined = values.ravel()
    span = float(combined.max() - combined.min())
    padding = max(0.04 * span, 0.005)
    low = float(combined.min() - padding)
    high = float(combined.max() + padding)

    ax.plot([low, high], [low, high], color="0.55", linewidth=1.0, zorder=0)
    ax.scatter(
        values[:, 0],
        values[:, 1],
        s=10,
        alpha=0.20,
        color="#2d6fa3",
        edgecolors="none",
        rasterized=True,
    )
    ax.scatter(
        [summary["ryan"]["median"]],
        [summary["m66"]["median"]],
        marker="D",
        s=62,
        color="#c44e36",
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
        label="population medians",
    )
    ax.set(
        title=title,
        xlabel=f"Ryan twin {axis_label}",
        ylabel=f"M66 {axis_label}",
        xlim=(low, high),
        ylim=(low, high),
    )
    ax.set_aspect("equal", adjustable="box")

    lines = [
        f"n={summary['n_units']:,} neurons / {summary['n_sessions']} sessions",
        (
            f"Ryan median {_format_value(summary['ryan']['median'], metric)} "
            f"{_format_interval(summary['ryan']['ci95'], metric)}"
        ),
        (
            f"M66 median {_format_value(summary['m66']['median'], metric)} "
            f"{_format_interval(summary['m66']['ci95'], metric)}"
        ),
        (
            f"paired median Δ {_format_value(summary['paired_difference']['median'], metric)} "
            f"{_format_interval(summary['paired_difference']['ci95'], metric)}"
        ),
        f"M66 higher for {100 * summary['fraction_m66_greater']['estimate']:.1f}%",
    ]
    if official_scores is not None:
        lines.insert(
            1,
            f"official score {official_scores[0]:.4f} → {official_scores[1]:.4f}",
        )
    ax.text(
        0.03,
        0.97,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.2,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": "0.85",
            "alpha": 0.92,
        },
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation, ryan_val_report, m66_val_report = load_validation(
        args.ryan_val.resolve(), args.m66_val.resolve()
    )
    fixrsvp = load_fixrsvp(args.ryan_fix.resolve(), args.m66_fix.resolve())
    paired = pd.concat([validation, fixrsvp], ignore_index=True)
    paired.to_csv(args.output_dir / "paired_neuron_metrics.csv", index=False)

    metric_order = (
        "validation_bps",
        "fixrsvp_ccnorm",
        "fixrsvp_single_trial_r2",
    )
    summaries = {}
    for metric_index, metric in enumerate(metric_order):
        table = paired.loc[paired.metric == metric].copy()
        summaries[metric] = hierarchical_bootstrap(
            table,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + metric_index,
            official_bps=metric == "validation_bps",
        )

    validation_official = {
        "ryan": float(ryan_val_report["bps_overall"]),
        "m66": float(m66_val_report["bps_overall"]),
    }
    validation_official["difference"] = (
        validation_official["m66"] - validation_official["ryan"]
    )
    summaries["validation_bps"]["official_validation_aggregation"] = (
        validation_official
    )

    report = {
        "model_selection": (
            "M66 is the final mechanistic twin; higher-scoring M70 was rejected "
            "because it failed the production Figure-4 mechanism gate."
        ),
        "inputs": {
            "ryan_validation": str(args.ryan_val.resolve()),
            "m66_validation": str(args.m66_val.resolve()),
            "ryan_fixrsvp": str(args.ryan_fix.resolve()),
            "m66_fixrsvp": str(args.m66_fix.resolve()),
        },
        "metrics": summaries,
        "fixrsvp_ccnorm_definition": {
            "numerator": "per-model CCabs on the identical >=20-trial time bins",
            "denominator": (
                "one shared data-only CCmax from the current seeded 2 x 500-split "
                "estimator in the M66 production cache"
            ),
            "population": (
                "paired intersection of the paper protocol's finite/stable cells; "
                "the exact same neurons enter both columns"
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    summary_rows = []
    for metric in metric_order:
        summary = summaries[metric]
        summary_rows.append(
            {
                "metric": metric,
                "n_neurons": summary["n_units"],
                "n_sessions": summary["n_sessions"],
                "ryan_median": summary["ryan"]["median"],
                "ryan_ci95_low": summary["ryan"]["ci95"][0],
                "ryan_ci95_high": summary["ryan"]["ci95"][1],
                "m66_median": summary["m66"]["median"],
                "m66_ci95_low": summary["m66"]["ci95"][0],
                "m66_ci95_high": summary["m66"]["ci95"][1],
                "paired_median_difference": summary["paired_difference"]["median"],
                "difference_ci95_low": summary["paired_difference"]["ci95"][0],
                "difference_ci95_high": summary["paired_difference"]["ci95"][1],
                "fraction_m66_greater": summary["fraction_m66_greater"]["estimate"],
            }
        )
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "summary.csv", index=False)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.65), constrained_layout=True)
    panels = (
        (
            "validation_bps",
            "A  In-distribution validation",
            "bits/spike",
            (validation_official["ryan"], validation_official["m66"]),
        ),
        (
            "fixrsvp_ccnorm",
            "B  FixRSVP generalization",
            "CCnorm (shared CCmax)",
            None,
        ),
        (
            "fixrsvp_single_trial_r2",
            "C  FixRSVP generalization",
            "single-trial R²",
            None,
        ),
    )
    for ax, (metric, title, label, official) in zip(axes, panels, strict=True):
        render_panel(
            ax,
            paired.loc[paired.metric == metric],
            summaries[metric],
            title=title,
            axis_label=label,
            metric=metric,
            official_scores=official,
        )
    fig.suptitle(
        "Ryan twin versus final mechanistic twin M66",
        fontsize=13,
        fontweight="bold",
    )
    stem = args.output_dir / "ryan_vs_m66_final_comparison"
    fig.savefig(stem.with_suffix(".png"), dpi=260)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
