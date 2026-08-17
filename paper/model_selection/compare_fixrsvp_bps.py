#!/usr/bin/env python3
"""Compare raw and Figure-3-affine FixRSVP Poisson bits/spike.

The canonical Ryan Figure-3 cache stores only affine-calibrated predictions.
Its upstream raw FixRSVP predictions are still present in the eval-stack
caches.  This script recovers the per-cell affine map from uniquely matched
response rows, validates that the map reconstructs the cached prediction to
sub-per-mille relative error, and then inverts it.  The candidate evaluator
cache records its affine parameters explicitly.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
RYAN_CACHE = ROOT / "outputs/cache/fig3_digitaltwin.pkl"
M66_CACHE = ROOT / "outputs/dekel240_evaluation/M66a_epoch31_fixrsvp_traces.pkl"
SESSION_ARCHIVE = (
    ROOT
    / "outputs/dekel240_evaluation/Ryan_05_lr5e-4_epoch471_val_full_per_unit.npz"
)
EVAL_ROOT = Path(
    "/mnt/ssd/YatesMarmoV1/conv_model_fits/eval_stack/"
    "learned_resnet_concat_convgru_gaussian_lr1e-3_wd1e-5_cls1.0_bs256_ga4"
)
OUTPUT = ROOT / "outputs/dekel240_evaluation/M66_vs_Ryan_fixrsvp_bps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache", type=Path, default=RYAN_CACHE)
    parser.add_argument("--candidate-cache", type=Path, default=M66_CACHE)
    parser.add_argument("--candidate-label", default="M66")
    parser.add_argument("--session-archive", type=Path, default=SESSION_ARCHIVE)
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def bits_per_spike(prediction, observation, valid):
    """Exact NumPy equivalent of eval.eval_stack_utils.bits_per_spike."""
    prediction = np.where(
        valid, np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )
    observation = np.where(
        valid, np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )
    weights = valid.astype(float)
    duration = np.maximum(weights.sum(axis=0), 1.0)
    spikes = np.maximum((weights * observation).sum(axis=0), 1.0)
    null_rate = spikes / duration
    model_ll = observation * np.log(prediction + 1e-8) - prediction
    null_ll = observation * np.log(null_rate[None, :] + 1e-8) - null_rate[None, :]
    return ((model_ll - null_ll) * weights).sum(axis=0) / spikes / np.log(2.0)


def _row_key(observation, filters):
    return (
        np.ascontiguousarray(observation, dtype=np.float32).tobytes()
        + np.ascontiguousarray(filters, dtype=np.float32).tobytes()
    )


def recover_ryan_raw(
    session, adjusted, observation, filters, neuron_mask, index, *, eval_root=EVAL_ROOT
):
    prefix = eval_root.name
    path = eval_root / f"{prefix}_dataset{index}_bps_cache.pt"
    cached = torch.load(path, map_location="cpu", weights_only=False)["fixrsvp"]
    raw_observation = np.asarray(cached["robs"])[:, neuron_mask]
    raw_filters = np.asarray(cached["dfs"])[:, neuron_mask]
    raw_prediction = np.asarray(cached["rhat"])[:, neuron_mask]

    n_units = len(neuron_mask)
    fig_observation = observation.reshape(-1, n_units)
    fig_filters = filters.reshape(-1, n_units)
    fig_adjusted = adjusted.reshape(-1, n_units)

    raw_rows = {}
    for row in range(len(raw_observation)):
        raw_rows.setdefault(
            _row_key(raw_observation[row], raw_filters[row]), []
        ).append(row)
    fig_rows = {}
    for row in range(len(fig_observation)):
        if np.isfinite(fig_observation[row]).all() and np.isfinite(fig_filters[row]).all():
            fig_rows.setdefault(
                _row_key(fig_observation[row], fig_filters[row]), []
            ).append(row)
    pairs = [
        (fig_rows[key][0], raw_rows[key][0])
        for key in fig_rows.keys() & raw_rows.keys()
        if len(fig_rows[key]) == 1 and len(raw_rows[key]) == 1
    ]
    if len(pairs) < 100:
        raise RuntimeError(f"{session}: only {len(pairs)} unique raw-row matches")
    fig_index = np.asarray([pair[0] for pair in pairs])
    raw_index = np.asarray([pair[1] for pair in pairs])

    scales = np.empty(n_units)
    offsets = np.empty(n_units)
    relative_rmse = np.empty(n_units)
    for unit in range(n_units):
        valid = (
            (fig_filters[fig_index, unit] > 0)
            & np.isfinite(fig_adjusted[fig_index, unit])
            & np.isfinite(raw_prediction[raw_index, unit])
        )
        design = np.column_stack(
            [raw_prediction[raw_index[valid], unit], np.ones(valid.sum())]
        )
        target = fig_adjusted[fig_index[valid], unit]
        scale, offset = np.linalg.lstsq(design, target, rcond=None)[0]
        fitted = design @ np.asarray([scale, offset])
        scales[unit] = scale
        offsets[unit] = offset
        relative_rmse[unit] = np.sqrt(np.mean((fitted - target) ** 2)) / max(
            np.std(target), 1e-9
        )
    if np.max(relative_rmse) > 1e-3 or np.min(scales) <= 0:
        raise RuntimeError(
            f"{session}: raw recovery failed validation; "
            f"max relative RMSE={np.max(relative_rmse):.6g}"
        )
    raw = (adjusted - offsets[None, None, :]) / scales[None, None, :]
    audit = {
        "matched_unique_rows": len(pairs),
        "median_relative_rmse": float(np.median(relative_rmse)),
        "max_relative_rmse": float(np.max(relative_rmse)),
    }
    return raw, audit


def training_style_overall(by_session):
    session_means = []
    for values in by_session.values():
        finite = values[np.isfinite(values)]
        if len(finite):
            session_means.append(np.clip(finite, 0.0, None).mean())
    return float(np.mean(session_means))


def hierarchical_ci(rows, key, rng, n_bootstrap=5000):
    sessions = sorted({row["session"] for row in rows})
    groups = {
        session: np.asarray([row[key] for row in rows if row["session"] == session])
        for session in sessions
    }
    bootstrap = np.empty(n_bootstrap)
    for sample in range(n_bootstrap):
        chunks = []
        for session in rng.choice(sessions, len(sessions), replace=True):
            values = groups[session]
            chunks.append(values[rng.integers(0, len(values), len(values))])
        bootstrap[sample] = np.median(np.concatenate(chunks))
    return np.quantile(bootstrap, [0.025, 0.975]).tolist()


def render_comparison(rows, report, candidate_label, out_dir):
    """Render paired raw/affine BPS scatters with hierarchical summaries."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), constrained_layout=True)
    for axis, mode, title in zip(
        axes,
        ("raw", "adjusted"),
        ("Unadjusted predictions", "Positive affine adjustment"),
        strict=True,
    ):
        x = np.asarray([row[f"ryan_{mode}"] for row in rows], dtype=float)
        y = np.asarray([row[f"candidate_{mode}"] for row in rows], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        bounds = np.quantile(np.concatenate((x, y)), [0.005, 0.995])
        padding = 0.06 * max(float(bounds[1] - bounds[0]), 1e-6)
        limits = (float(bounds[0] - padding), float(bounds[1] + padding))
        axis.scatter(x, y, s=8, alpha=0.20, color="#4c78a8", linewidths=0)
        axis.plot(limits, limits, color="0.45", lw=1.1, zorder=0)
        axis.set(xlim=limits, ylim=limits, aspect="equal", adjustable="box")
        axis.set_xlabel("Twin bits/spike")
        axis.set_ylabel(f"{candidate_label} bits/spike")
        axis.set_title(title, fontweight="bold")

        block = report[mode]
        ryan_median = block["unit_median"]["ryan"]
        candidate_median = block["unit_median"]["candidate"]
        axis.scatter(
            [ryan_median],
            [candidate_median],
            marker="D",
            s=58,
            color="#c84c36",
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        ryan_ci = block["hierarchical_95ci"]["ryan_unit_median"]
        candidate_ci = block["hierarchical_95ci"]["candidate_unit_median"]
        delta_ci = block["hierarchical_95ci"]["paired_delta_median"]
        delta = block["paired_delta_median_candidate_minus_ryan"]
        overall = block["training_style_overall"]
        text = (
            f"n={len(x):,}\n"
            f"unit median: {ryan_median:.3f} → {candidate_median:.3f}\n"
            f"95% CI: [{ryan_ci[0]:.3f}, {ryan_ci[1]:.3f}] → "
            f"[{candidate_ci[0]:.3f}, {candidate_ci[1]:.3f}]\n"
            f"paired Δ median: {delta:+.3f} "
            f"[{delta_ci[0]:+.3f}, {delta_ci[1]:+.3f}]\n"
            f"training-style overall: {overall['ryan']:.3f} → "
            f"{overall['candidate']:.3f}"
        )
        axis.text(
            0.03,
            0.97,
            text,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.3,
            bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none", "pad": 3},
        )
        axis.grid(color="0.9", linewidth=0.6)
    fig.suptitle(
        f"FixRSVP Poisson likelihood: Twin versus {candidate_label}\n"
        "identical observations, neurons, and data-valid support",
        fontsize=12,
        fontweight="bold",
    )
    png = out_dir / "fixrsvp_bps_comparison.png"
    fig.savefig(png, dpi=220, facecolor="white")
    fig.savefig(png.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    ryan = {
        value["session"]: value
        for value in pickle.load(args.reference_cache.open("rb"))
    }
    candidate = {
        value["session"]: value
        for value in pickle.load(args.candidate_cache.open("rb"))
    }
    with np.load(args.session_archive) as archive:
        all_session_names = archive["session_names"].astype(str).tolist()

    metrics = (
        "ryan_raw",
        "candidate_raw",
        "ryan_adjusted",
        "candidate_adjusted",
    )
    by_session = {metric: {} for metric in metrics}
    rows = []
    recovery_audit = {}
    for session in sorted(ryan):
        reference = ryan[session]
        candidate_session = candidate[session]
        neuron_mask = np.asarray(reference["neuron_mask"], dtype=int)
        if not np.array_equal(neuron_mask, candidate_session["neuron_mask"]):
            raise RuntimeError(f"{session}: neuron masks differ")
        observation = np.asarray(reference["robs_used"], dtype=float)
        filters = np.asarray(reference["dfs_used"], dtype=float)
        if not np.array_equal(
            observation, np.asarray(candidate_session["robs_used"]), equal_nan=True
        ):
            raise RuntimeError(f"{session}: observations differ")
        if not np.array_equal(
            filters, np.asarray(candidate_session["dfs_used"]), equal_nan=True
        ):
            raise RuntimeError(f"{session}: filters differ")

        ryan_adjusted = np.asarray(reference["rhat_used"], dtype=float)
        candidate_adjusted = np.asarray(candidate_session["rhat_used"], dtype=float)
        candidate_raw = (
            candidate_adjusted
            - np.asarray(candidate_session["affine_offset"])[None, None, :]
        ) / np.asarray(candidate_session["affine_scale"])[None, None, :]
        ryan_raw, audit = recover_ryan_raw(
            session,
            ryan_adjusted,
            observation,
            filters,
            neuron_mask,
            all_session_names.index(session),
            eval_root=args.eval_root,
        )
        recovery_audit[session] = audit

        valid = (
            np.isfinite(observation)
            & np.isfinite(filters)
            & (filters > 0)
            & np.isfinite(ryan_raw)
            & np.isfinite(candidate_raw)
            & np.isfinite(ryan_adjusted)
            & np.isfinite(candidate_adjusted)
        )
        n_units = len(neuron_mask)
        flat_observation = observation.reshape(-1, n_units)
        flat_valid = valid.reshape(-1, n_units)
        values = {
            "ryan_raw": bits_per_spike(
                ryan_raw.reshape(-1, n_units), flat_observation, flat_valid
            ),
            "candidate_raw": bits_per_spike(
                candidate_raw.reshape(-1, n_units), flat_observation, flat_valid
            ),
            "ryan_adjusted": bits_per_spike(
                ryan_adjusted.reshape(-1, n_units), flat_observation, flat_valid
            ),
            "candidate_adjusted": bits_per_spike(
                candidate_adjusted.reshape(-1, n_units), flat_observation, flat_valid
            ),
        }
        for metric in metrics:
            by_session[metric][session] = values[metric]
        for local, source_unit in enumerate(neuron_mask):
            if all(np.isfinite(values[metric][local]) for metric in metrics):
                row = {
                    "session": session,
                    "source_unit_index": int(source_unit),
                    **{metric: float(values[metric][local]) for metric in metrics},
                }
                row["delta_raw_candidate_minus_ryan"] = (
                    row["candidate_raw"] - row["ryan_raw"]
                )
                row["delta_adjusted_candidate_minus_ryan"] = (
                    row["candidate_adjusted"] - row["ryan_adjusted"]
                )
                rows.append(row)

    rng = np.random.default_rng(20260816)
    report = {
        "n_sessions": len(ryan),
        "n_units": len(rows),
        "candidate_label": args.candidate_label,
        "candidate_cache": str(args.candidate_cache.resolve()),
        "shared_mask": True,
        "adjustment": "per-unit positive affine fit on the same FixRSVP trials",
        "raw_recovery_audit": {
            "minimum_matched_unique_rows": min(
                value["matched_unique_rows"] for value in recovery_audit.values()
            ),
            "maximum_relative_rmse": max(
                value["max_relative_rmse"] for value in recovery_audit.values()
            ),
        },
        "raw": {},
        "adjusted": {},
    }
    for mode in ("raw", "adjusted"):
        left = np.asarray([row[f"ryan_{mode}"] for row in rows])
        right = np.asarray([row[f"candidate_{mode}"] for row in rows])
        delta_key = f"delta_{mode}_candidate_minus_ryan"
        delta = np.asarray([row[delta_key] for row in rows])
        report[mode] = {
            "training_style_overall": {
                "ryan": training_style_overall(by_session[f"ryan_{mode}"]),
                "candidate": training_style_overall(
                    by_session[f"candidate_{mode}"]
                ),
            },
            "unit_median": {
                "ryan": float(np.median(left)),
                "candidate": float(np.median(right)),
            },
            "paired_delta_median_candidate_minus_ryan": float(np.median(delta)),
            "paired_delta_mean_candidate_minus_ryan": float(np.mean(delta)),
            "candidate_win_fraction": float(np.mean(delta > 0)),
            "negative_bps_fraction": {
                "ryan": float(np.mean(left < 0)),
                "candidate": float(np.mean(right < 0)),
            },
            "hierarchical_95ci": {
                "ryan_unit_median": hierarchical_ci(rows, f"ryan_{mode}", rng),
                "candidate_unit_median": hierarchical_ci(
                    rows, f"candidate_{mode}", rng
                ),
                "paired_delta_median": hierarchical_ci(rows, delta_key, rng),
            },
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.out_dir / "paired_unit_bps.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    render_comparison(rows, report, args.candidate_label, args.out_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
