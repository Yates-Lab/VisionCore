#!/usr/bin/env python3
"""Split Ryan/M66 FixRSVP captured rate variance into PSTH and FEM terms.

The Figure-3 panel-D numerator is

    captured = Var(y) - Var(y - p) = 2 Cov(y, p) - Var(p).

On the exact same scored samples, write the prediction as its repeat-mean
component plus its within-repeat component, ``p = p_psth + p_fem``.  The two
prediction components are sample-orthogonal, so

    captured_psth = 2 Cov(y, p_psth) - Var(p_psth)
    captured_fem  = 2 Cov(y, p_fem)  - Var(p_fem)

and their sum is the panel-D numerator to numerical precision.  The terms are
reported both as contributions to empirical ``diag(Crate)`` and normalized by
their corresponding empirical covariance-decomposition components,
``diag(Cpsth)`` and ``diag(Crate - Cpsth)``.

"FEM" here follows the paper's covariance-decomposition label.  Because the
twins also receive behaviour, the within-repeat prediction can include any
trial-specific covariate coupled to the response, not only retinal image shifts.
The stabilized-retina ablation is needed to assign that term causally to FEMs.
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "paper" / "fig3", ROOT / "paper" / "covariance_decomposition"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _fig3_explainable_variance import (  # noqa: E402
    MIN_SCORED_WINDOWS,
    figure2_valid_mask,
    matched_count_indices,
)
import derive  # noqa: E402


RYAN_CACHE = ROOT / "outputs/cache/fig3_digitaltwin.pkl"
M66_CACHE = ROOT / "outputs/dekel240_evaluation/M66a_epoch31_fixrsvp_traces.pkl"
ALIGNED_CACHE = ROOT / "outputs/dekel240_paper/final/cache_fig3/covdecomp_aligned_sessions.pkl"
EMPIRICAL_CACHE = ROOT / "outputs/dekel240_paper/final/cache_fig3/covdecomp_empirical.pkl"
ABLATION_CACHE = ROOT / "outputs/dekel240_paper/final/cache_fig3/fig3_ablation_inference.pkl"
OUTPUT = ROOT / "outputs/dekel240_evaluation/M66_vs_Ryan_fixrsvp_variance_components"
MIN_SESSION_UNITS = 10


def sample_cov(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 2:
        return np.nan
    return float(np.cov(left, right, ddof=1)[0, 1])


def sample_var(values):
    values = np.asarray(values, dtype=float)
    return float(np.var(values, ddof=1)) if values.size >= 2 else np.nan


def prediction_components(prediction, phase):
    """Return repeat-mean and within-repeat prediction on retained samples."""
    prediction = np.asarray(prediction, dtype=float)
    phase = np.asarray(phase, dtype=int)
    unique, inverse = np.unique(phase, return_inverse=True)
    counts = np.bincount(inverse)
    means = np.bincount(inverse, weights=prediction) / counts
    between = means[inverse]
    within = prediction - between
    return between, within, int(len(unique))


def decompose_captured_variance(robs, predictions, eyepos, dfs):
    """Exact additive split of the panel-D numerator on one common mask."""
    robs = np.asarray(robs, dtype=float)
    dfs = np.asarray(dfs, dtype=float)
    eyepos = np.asarray(eyepos, dtype=float)
    predictions = {key: np.asarray(value, dtype=float) for key, value in predictions.items()}
    n_units = robs.shape[2]

    trial, time = matched_count_indices(figure2_valid_mask(robs, eyepos))
    base_y = robs[trial, time]
    base_dfs = dfs[trial, time]
    base_predictions = {key: value[trial, time] for key, value in predictions.items()}

    output = {
        key: {
            term: np.full(n_units, np.nan, dtype=float)
            for term in ("total", "psth", "fem", "closure_error")
        }
        for key in predictions
    }
    n_windows = np.zeros(n_units, dtype=int)
    n_phases = np.zeros(n_units, dtype=int)

    for unit in range(n_units):
        keep = (
            np.isfinite(base_y[:, unit])
            & np.isfinite(base_dfs[:, unit])
            & (base_dfs[:, unit] > 0)
        )
        for value in base_predictions.values():
            keep &= np.isfinite(value[:, unit])
        n_windows[unit] = int(keep.sum())
        if n_windows[unit] < MIN_SCORED_WINDOWS:
            continue

        y = base_y[keep, unit]
        phase = time[keep]
        for key, value in base_predictions.items():
            p = value[keep, unit]
            p_psth, p_fem, phases = prediction_components(p, phase)
            n_phases[unit] = phases
            total = sample_var(y) - sample_var(y - p)
            psth = 2.0 * sample_cov(y, p_psth) - sample_var(p_psth)
            fem = 2.0 * sample_cov(y, p_fem) - sample_var(p_fem)
            output[key]["total"][unit] = total
            output[key]["psth"][unit] = psth
            output[key]["fem"][unit] = fem
            output[key]["closure_error"][unit] = total - psth - fem
    return output, n_windows, n_phases


def hierarchical_ci(rows, key, rng, n_bootstrap=5000):
    sessions = sorted({row["session"] for row in rows if np.isfinite(row[key])})
    groups = {
        session: np.asarray(
            [row[key] for row in rows if row["session"] == session and np.isfinite(row[key])]
        )
        for session in sessions
    }
    groups = {key_: value for key_, value in groups.items() if len(value)}
    sessions = sorted(groups)
    if not sessions:
        return [None, None]
    bootstrap = np.empty(n_bootstrap, dtype=float)
    for sample in range(n_bootstrap):
        chunks = []
        for session in rng.choice(sessions, len(sessions), replace=True):
            values = groups[session]
            chunks.append(values[rng.integers(0, len(values), len(values))])
        bootstrap[sample] = np.median(np.concatenate(chunks))
    return np.quantile(bootstrap, [0.025, 0.975]).tolist()


def metric_summary(rows, key, rng):
    values = np.asarray([row[key] for row in rows], dtype=float)
    finite = np.isfinite(values)
    return {
        "n": int(finite.sum()),
        "median": float(np.median(values[finite])) if finite.any() else None,
        "hierarchical_95ci": hierarchical_ci(rows, key, rng),
    }


def main():
    with RYAN_CACHE.open("rb") as stream:
        ryan = {row["session"]: row for row in pickle.load(stream)}
    with M66_CACHE.open("rb") as stream:
        m66 = {row["session"]: row for row in pickle.load(stream)}
    with ALIGNED_CACHE.open("rb") as stream:
        aligned = {row["session"]: row for row in pickle.load(stream)}
    with EMPIRICAL_CACHE.open("rb") as stream:
        empirical = {row["session"]: row for row in pickle.load(stream)}
    with ABLATION_CACHE.open("rb") as stream:
        ablation_payload = pickle.load(stream)
    ablation = {row["session"]: row for row in ablation_payload["results"]}

    rows = []
    session_counts = {}
    m66_cache_differences = []
    closure_errors = []
    for session in sorted(set(ryan) & set(m66) & set(aligned) & set(empirical)):
        left = ryan[session]
        right = m66[session]
        unit_ids = np.asarray(left["neuron_mask"], dtype=int)
        if not np.array_equal(unit_ids, np.asarray(right["neuron_mask"], dtype=int)):
            raise RuntimeError(f"{session}: neuron masks differ")
        robs = np.asarray(left["robs_used"], dtype=float)
        dfs = np.asarray(left["dfs_used"], dtype=float)
        if not np.array_equal(robs, np.asarray(right["robs_used"]), equal_nan=True):
            raise RuntimeError(f"{session}: observations differ")
        if not np.array_equal(dfs, np.asarray(right["dfs_used"]), equal_nan=True):
            raise RuntimeError(f"{session}: filters differ")

        aligned_row = aligned[session]
        aligned_ids = np.asarray(aligned_row["neuron_mask"], dtype=int)
        included = set(
            aligned_ids[
                np.isfinite(aligned_row["rate_hz"])
                & (np.asarray(aligned_row["rate_hz"]) > derive.MIN_RATE_HZ)
                & np.isfinite(aligned_row["psth_r2"])
                & (np.asarray(aligned_row["psth_r2"]) > derive.MIN_PSTH_R2)
            ].tolist()
        )
        selected = np.asarray([unit in included for unit in unit_ids], dtype=bool)
        if int(selected.sum()) < MIN_SESSION_UNITS:
            continue
        session_counts[session] = int(selected.sum())

        split, n_windows, n_phases = decompose_captured_variance(
            robs,
            {
                "ryan": np.asarray(left["rhat_used"], dtype=float),
                "m66": np.asarray(right["rhat_used"], dtype=float),
            },
            np.asarray(right["eyepos_used"], dtype=float),
            dfs,
        )

        empirical_row = empirical[session]
        empirical_ids = np.asarray(empirical_row["neuron_mask"], dtype=int)
        empirical_column = {int(unit): index for index, unit in enumerate(empirical_ids)}
        window = next(value for value in empirical_row["windows"] if value["window_bins"] == 1)
        block = window["targets"]["full"]
        c_rate_all = np.diag(np.asarray(block["Crate"], dtype=float))
        c_psth_all = np.diag(np.asarray(block["Cpsth"], dtype=float))

        cached_m66 = None
        cached_column = {}
        if session in ablation:
            cached_m66 = np.asarray(ablation[session]["captured_variance"]["intact"], dtype=float)
            cached_column = {
                int(unit): index
                for index, unit in enumerate(np.asarray(ablation[session]["neuron_mask"], dtype=int))
            }

        for local, unit in enumerate(unit_ids):
            if not selected[local] or int(unit) not in empirical_column:
                continue
            column = empirical_column[int(unit)]
            c_rate = float(c_rate_all[column])
            c_psth = float(c_psth_all[column])
            c_fem = c_rate - c_psth
            row = {
                "session": session,
                "unit_id": int(unit),
                "n_windows": int(n_windows[local]),
                "n_phases": int(n_phases[local]),
                "c_rate": c_rate,
                "c_psth": c_psth,
                "c_fem": c_fem,
            }
            for model in ("ryan", "m66"):
                for component in ("total", "psth", "fem", "closure_error"):
                    row[f"{model}_captured_{component}"] = float(split[model][component][local])
                if c_rate > 0:
                    row[f"{model}_q_total"] = row[f"{model}_captured_total"] / c_rate
                    row[f"{model}_q_psth"] = row[f"{model}_captured_psth"] / c_rate
                    row[f"{model}_q_fem"] = row[f"{model}_captured_fem"] / c_rate
                else:
                    row[f"{model}_q_total"] = np.nan
                    row[f"{model}_q_psth"] = np.nan
                    row[f"{model}_q_fem"] = np.nan
                row[f"{model}_psth_efficiency"] = (
                    row[f"{model}_captured_psth"] / c_psth if c_psth > 0 else np.nan
                )
                row[f"{model}_fem_efficiency"] = (
                    row[f"{model}_captured_fem"] / c_fem if c_fem > 0 else np.nan
                )
                closure_errors.append(abs(row[f"{model}_captured_closure_error"]))
            for component in ("total", "psth", "fem"):
                row[f"delta_q_{component}_m66_minus_ryan"] = (
                    row[f"m66_q_{component}"] - row[f"ryan_q_{component}"]
                )
            for component in ("psth", "fem"):
                row[f"delta_{component}_efficiency_m66_minus_ryan"] = (
                    row[f"m66_{component}_efficiency"]
                    - row[f"ryan_{component}_efficiency"]
                )
            rows.append(row)

            if cached_m66 is not None and int(unit) in cached_column:
                cached = cached_m66[cached_column[int(unit)]]
                current = row["m66_captured_total"]
                if np.isfinite(cached) and np.isfinite(current):
                    m66_cache_differences.append(abs(float(cached - current)))

    rng = np.random.default_rng(20260816)
    paired_panel_d = [
        row for row in rows
        if np.isfinite(row["ryan_q_total"])
        and np.isfinite(row["m66_q_total"])
        and row["n_windows"] >= MIN_SCORED_WINDOWS
    ]
    # A within-repeat/FEM term is not identifiable when every retained phase
    # appears only once.  This occurs for one session after the panel-D common
    # all-neuron validity frame is applied.  Keep those units in the published
    # total score, but exclude them from claims about the PSTH/FEM split.
    component_rows = [
        row for row in paired_panel_d if row["n_windows"] > row["n_phases"]
    ]
    component_efficiency = {
        "psth": [
            row for row in component_rows
            if np.isfinite(row["ryan_psth_efficiency"])
            and np.isfinite(row["m66_psth_efficiency"])
        ],
        "fem": [
            row for row in component_rows
            if np.isfinite(row["ryan_fem_efficiency"])
            and np.isfinite(row["m66_fem_efficiency"])
        ],
    }

    metrics = {}
    for component in ("total", "psth", "fem"):
        subset = paired_panel_d if component == "total" else component_rows
        metrics[f"q_{component}"] = {
            model: metric_summary(subset, f"{model}_q_{component}", rng)
            for model in ("ryan", "m66")
        }
        metrics[f"q_{component}"]["paired_delta_m66_minus_ryan"] = metric_summary(
            subset, f"delta_q_{component}_m66_minus_ryan", rng
        )
    metrics["q_total_on_component_population"] = {
        model: metric_summary(component_rows, f"{model}_q_total", rng)
        for model in ("ryan", "m66")
    }
    metrics["q_total_on_component_population"]["paired_delta_m66_minus_ryan"] = (
        metric_summary(component_rows, "delta_q_total_m66_minus_ryan", rng)
    )
    for component in ("psth", "fem"):
        subset = component_efficiency[component]
        metrics[f"{component}_efficiency"] = {
            model: metric_summary(subset, f"{model}_{component}_efficiency", rng)
            for model in ("ryan", "m66")
        }
        metrics[f"{component}_efficiency"]["paired_delta_m66_minus_ryan"] = metric_summary(
            subset, f"delta_{component}_efficiency_m66_minus_ryan", rng
        )

    report = {
        "model_predictions": "Figure-3 affine-adjusted FixRSVP predictions",
        "window_bins": 1,
        "window_ms": 1000 / 120,
        "population": {
            "minimum_rate_hz": derive.MIN_RATE_HZ,
            "minimum_split_half_psth_r2": derive.MIN_PSTH_R2,
            "minimum_units_per_session": MIN_SESSION_UNITS,
            "minimum_scored_windows": MIN_SCORED_WINDOWS,
            "n_sessions_before_score_filter": len(session_counts),
            "n_units_before_score_filter": int(sum(session_counts.values())),
            "n_sessions_panel_d": len({row["session"] for row in paired_panel_d}),
            "n_units_panel_d": len(paired_panel_d),
            "n_sessions_component_split": len(
                {row["session"] for row in component_rows}
            ),
            "n_units_component_split": len(component_rows),
            "n_units_component_unidentifiable": len(paired_panel_d) - len(component_rows),
            "n_units_psth_efficiency": len(component_efficiency["psth"]),
            "n_units_fem_efficiency": len(component_efficiency["fem"]),
        },
        "definitions": {
            "q_total": "captured_total / empirical Crate",
            "q_psth": "captured_PSTH / empirical Crate (additive contribution)",
            "q_fem": "captured_FEM / empirical Crate (additive contribution)",
            "psth_efficiency": "captured_PSTH / empirical Cpsth",
            "fem_efficiency": "captured_FEM / empirical (Crate - Cpsth)",
            "identity": "q_total = q_psth + q_fem per unit",
            "component_support": (
                "PSTH/FEM summaries require at least one repeated phase "
                "(n_windows > n_phases); total q retains the full panel-D population"
            ),
        },
        "audit": {
            "maximum_absolute_additive_closure_error": float(max(closure_errors, default=np.nan)),
            "maximum_absolute_m66_panel_d_cache_difference": float(
                max(m66_cache_differences, default=np.nan)
            ),
            "m66_cache_comparisons": len(m66_cache_differences),
        },
        "metrics": metrics,
        "interpretation_caveat": (
            "The within-repeat term is FEM-linked/trial-specific. Because the models also "
            "receive behavior, causal attribution specifically to retinal FEM requires the "
            "stabilized-retina ablation."
        ),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (OUTPUT / "paired_unit_variance_components.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
