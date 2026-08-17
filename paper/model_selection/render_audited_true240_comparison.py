#!/usr/bin/env python3
"""Render an audited Ryan-versus-native-240 production comparison.

Validation units are paired by ``(session, cid)``.  FixRSVP units are paired
by ``(session, source unit index)`` after exact observation/filter checks.
Ryan and the candidate are rescored with one explicit data-only support and
must have identical per-unit CCmax.  The script aborts on any masking or
CCnorm-identity violation rather than rendering a misleading population plot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import dill
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.model_selection import evaluate_dekel_fixrsvp as fixrsvp_eval
from paper.model_selection.audit_fixrsvp_trace_cache import independent_ccabs

_ccnorm_on_fixed_support = fixrsvp_eval._ccnorm_on_fixed_support


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    validation = parser.add_mutually_exclusive_group(required=False)
    validation.add_argument(
        "--paired-val",
        type=Path,
        help="Exact common-support report from evaluate_paired_validation_models.py",
    )
    validation.add_argument(
        "--separate-val-reports",
        action="store_true",
        help="Use legacy separate reports (requires --ryan-val and --candidate-val)",
    )
    parser.add_argument("--ryan-val", type=Path)
    parser.add_argument("--candidate-val", type=Path)
    parser.add_argument(
        "--ryan-fix",
        type=Path,
        default=ROOT / "outputs/cache/fig3_digitaltwin.pkl",
    )
    parser.add_argument("--candidate-fix", type=Path, required=True)
    parser.add_argument(
        "--candidate-metric-audit",
        type=Path,
        help=(
            "Independent audit_fixrsvp_trace_cache.py report. When supplied, "
            "reuse its verified data-only CCmax and recompute only each model's "
            "CCabs, avoiding redundant split-half resampling."
        ),
    )
    parser.add_argument(
        "--fix-only",
        action="store_true",
        help="Compare two FixRSVP trace caches without a validation panel.",
    )
    parser.add_argument("--reference-label", default="Ryan")
    parser.add_argument("--candidate-label", default="native-240 candidate")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--max-fix-sessions", type=int, default=None)
    parser.add_argument(
        "--ccnorm-splits",
        type=int,
        default=None,
        help="Testing override; production defaults to the Figure-3 value (500 per seed).",
    )
    return parser.parse_args()


def _resolve_npz(report_path: Path, report: dict) -> Path:
    path = Path(report["per_unit_bps_npz"])
    return path if path.is_absolute() else report_path.parent / path


def load_validation(reference_path: Path, candidate_path: Path):
    reference_report = json.loads(reference_path.read_text())
    candidate_report = json.loads(candidate_path.read_text())
    if reference_report["split"] != "val" or candidate_report["split"] != "val":
        raise RuntimeError("Both validation reports must use split='val'")
    reference_rate = int(
        reference_report.get("score_rate_hz", reference_report.get("supervision_rate_hz", -1))
    )
    candidate_rate = int(
        candidate_report.get("score_rate_hz", candidate_report.get("supervision_rate_hz", -1))
    )
    if reference_rate != 120 or candidate_rate != 120:
        raise RuntimeError(
            "Production validation comparison requires both likelihoods on the "
            f"same 120-Hz count grid; got {reference_rate} and {candidate_rate}"
        )
    reference_counts = reference_report.get("samples_by_session")
    candidate_counts = candidate_report.get(
        "pairs_by_session", candidate_report.get("samples_by_session")
    )
    if reference_counts is None or candidate_counts is None:
        raise RuntimeError("Validation reports must record per-session sample counts")
    if reference_counts != candidate_counts:
        differences = {
            name: (reference_counts.get(name), candidate_counts.get(name))
            for name in sorted(set(reference_counts) | set(candidate_counts))
            if reference_counts.get(name) != candidate_counts.get(name)
        }
        raise RuntimeError(
            "Validation 120-Hz sample supports differ by session: "
            f"{differences}"
        )
    rows = []
    with (
        np.load(_resolve_npz(reference_path, reference_report)) as reference,
        np.load(_resolve_npz(candidate_path, candidate_report)) as candidate,
    ):
        ref_names = reference["session_names"].astype(str)
        cand_names = candidate["session_names"].astype(str)
        if not np.array_equal(ref_names, cand_names):
            raise RuntimeError("Validation session order differs")
        for index, session in enumerate(ref_names):
            ref_cids = reference[f"cids_{index}"].astype(np.int64)
            cand_cids = candidate[f"cids_{index}"].astype(np.int64)
            if not np.array_equal(ref_cids, cand_cids):
                raise RuntimeError(f"Validation cids differ for {session}")
            left = reference[f"bps_{index}"].astype(np.float64)
            right = candidate[f"bps_{index}"].astype(np.float64)
            valid = np.isfinite(left) & np.isfinite(right)
            rows.extend(
                {
                    "metric": "validation_bps",
                    "session": str(session),
                    "unit_id": int(cid),
                    "ryan": float(ref_value),
                    "candidate": float(cand_value),
                }
                for cid, ref_value, cand_value in zip(
                    ref_cids[valid], left[valid], right[valid], strict=True
                )
            )
    return pd.DataFrame(rows), reference_report, candidate_report


def load_paired_validation(report_path: Path):
    """Load a comparison already scored on one exact support mask."""
    report = json.loads(report_path.read_text())
    if report.get("split") != "val" or int(report.get("score_rate_hz", -1)) != 120:
        raise RuntimeError("Paired validation must be a 120-Hz validation report")
    archive_path = Path(report["per_unit_npz"])
    if not archive_path.is_absolute():
        archive_path = report_path.parent / archive_path
    rows = []
    with np.load(archive_path) as archive:
        names = archive["session_names"].astype(str)
        expected = set(report["shared_geometry_by_session"])
        if set(names) != expected:
            raise RuntimeError("Paired validation sessions disagree with geometry audit")
        for index, session in enumerate(names):
            cids = archive[f"cids_{index}"].astype(np.int64)
            candidate = archive[f"candidate_bps_{index}"].astype(np.float64)
            reference = archive[f"reference_bps_{index}"].astype(np.float64)
            counts = archive[f"common_valid_count_{index}"].astype(np.int64)
            if not (len(cids) == len(candidate) == len(reference) == len(counts)):
                raise RuntimeError(f"{session}: paired validation arrays differ in length")
            valid = np.isfinite(candidate) & np.isfinite(reference) & (counts > 0)
            rows.extend(
                {
                    "metric": "validation_bps",
                    "session": str(session),
                    "unit_id": int(cid),
                    "ryan": float(ref_value),
                    "candidate": float(candidate_value),
                }
                for cid, ref_value, candidate_value in zip(
                    cids[valid], reference[valid], candidate[valid], strict=True
                )
            )
    return pd.DataFrame(rows), report


def _load_cache(path: Path) -> dict[str, dict]:
    with path.open("rb") as stream:
        values = dill.load(stream)
    return {str(value["session"]): value for value in values}


def _load_candidate_metric_audit(path: Path, candidate_path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    if not payload.get("independent_implementation") or not payload.get("all_checks_passed"):
        raise RuntimeError("candidate metric audit did not pass independently")
    audited_trace = Path(payload["trace_cache"]).resolve()
    if audited_trace != candidate_path.resolve():
        raise RuntimeError(
            f"metric audit covers {audited_trace}, not {candidate_path.resolve()}"
        )
    sessions = {str(row["session"]): row for row in payload.get("sessions", [])}
    if len(sessions) != int(payload.get("n_sessions", -1)):
        raise RuntimeError("candidate metric audit has duplicate or missing sessions")
    required_true = (
        "support_exact_match",
        "prediction_complete_on_support",
        "stability_mask_exact_match",
    )
    for session, row in sessions.items():
        if not all(bool(row.get(key)) for key in required_true):
            raise RuntimeError(f"candidate metric audit failed for {session}")
        for key in (
            "ccabs_max_abs_error",
            "ccmax_max_abs_error",
            "ccnorm_identity_max_abs_error",
        ):
            if float(row.get(key, np.inf)) > 2e-12:
                raise RuntimeError(f"candidate metric audit {key} failed for {session}")
    return sessions


def load_fixrsvp(
    reference_path: Path,
    candidate_path: Path,
    max_sessions=None,
    candidate_metric_audit: Path | None = None,
):
    reference = _load_cache(reference_path)
    candidate = _load_cache(candidate_path)
    names = sorted(set(reference) & set(candidate))
    if set(reference) != set(candidate) and max_sessions is None:
        raise RuntimeError("FixRSVP session sets differ")
    if max_sessions is not None:
        names = names[: int(max_sessions)]
    audited_sessions = (
        _load_candidate_metric_audit(candidate_metric_audit, candidate_path)
        if candidate_metric_audit is not None
        else None
    )
    if audited_sessions is not None and not set(names).issubset(audited_sessions):
        missing = sorted(set(names) - set(audited_sessions))
        raise RuntimeError(f"candidate metric audit lacks sessions: {missing}")

    rows = []
    audits = []
    for ordinal, session in enumerate(names, 1):
        print(f"[{ordinal}/{len(names)}] audited FixRSVP metrics: {session}", flush=True)
        left = reference[session]
        right = candidate[session]
        ref_units = np.asarray(left["neuron_mask"], dtype=np.int64)
        cand_units = np.asarray(right["neuron_mask"], dtype=np.int64)
        if not np.array_equal(ref_units, cand_units):
            raise RuntimeError(f"FixRSVP neuron masks differ for {session}")
        for key in ("robs_used", "dfs_used"):
            if not np.array_equal(
                np.asarray(left[key]), np.asarray(right[key]), equal_nan=True
            ):
                raise RuntimeError(f"FixRSVP {key} differs for {session}")

        if audited_sessions is None:
            ref_cc = _ccnorm_on_fixed_support(
                left["robs_used"], left["rhat_used"], left["dfs_used"]
            )
            cand_cc = _ccnorm_on_fixed_support(
                right["robs_used"], right["rhat_used"], right["dfs_used"]
            )
            ccmax_source = "independently recomputed for both models"
        else:
            support = fixrsvp_eval._data_score_mask(
                right["robs_used"], right["dfs_used"]
            )
            if not np.array_equal(support, np.asarray(right["ccnorm_support"], bool)):
                raise AssertionError(f"{session}: stored support changed after audit")
            ccmax = np.asarray(right["ccmax"], dtype=np.float64)
            unstable = np.asarray(right["ccnorm_unstable"], dtype=bool)
            candidate_ccabs = independent_ccabs(
                right["robs_used"], right["rhat_used"], support
            )
            reference_ccabs = independent_ccabs(
                left["robs_used"], left["rhat_used"], support
            )

            def cached_ceiling_metrics(ccabs):
                with np.errstate(divide="ignore", invalid="ignore"):
                    ccnorm = np.asarray(ccabs, dtype=np.float64) / ccmax
                ccnorm[unstable] = np.nan
                return {
                    "support": support,
                    "ccabs": np.asarray(ccabs, dtype=np.float64),
                    "ccmax": ccmax,
                    "ccnorm": ccnorm,
                    "unstable": unstable,
                }

            cand_cc = cached_ceiling_metrics(candidate_ccabs)
            ref_cc = cached_ceiling_metrics(reference_ccabs)
            if not np.allclose(
                cand_cc["ccabs"], right["ccabs"], rtol=0, atol=2e-12, equal_nan=True
            ) or not np.allclose(
                cand_cc["ccnorm"], right["ccnorm"], rtol=0, atol=2e-12, equal_nan=True
            ):
                raise AssertionError(f"{session}: audited candidate metrics changed")
            ccmax_source = "candidate data-only ceiling verified by independent audit"
        support_equal = np.array_equal(ref_cc["support"], cand_cc["support"])
        max_ccmax_difference = float(
            np.nanmax(np.abs(ref_cc["ccmax"] - cand_cc["ccmax"]))
        )
        ccmax_equal = np.allclose(
            ref_cc["ccmax"], cand_cc["ccmax"], rtol=0, atol=0, equal_nan=True
        )
        stability_equal = np.array_equal(
            ref_cc["unstable"], cand_cc["unstable"]
        )
        if not support_equal or not ccmax_equal or not stability_equal:
            raise AssertionError(
                f"{session}: shared CCnorm audit failed "
                f"(support={support_equal}, stability={stability_equal}, "
                f"max CCmax delta={max_ccmax_difference})"
            )

        stable = np.isfinite(ref_cc["ccnorm"]) & np.isfinite(cand_cc["ccnorm"])
        raw_delta = cand_cc["ccabs"] - ref_cc["ccabs"]
        normalized_delta = cand_cc["ccnorm"] - ref_cc["ccnorm"]
        if np.any(raw_delta[stable] * normalized_delta[stable] < -1e-14):
            raise AssertionError(
                f"{session}: CCabs and CCnorm model differences disagree in sign"
            )
        for metric, ref_values, cand_values, valid in (
            ("fixrsvp_ccabs", ref_cc["ccabs"], cand_cc["ccabs"], stable),
            ("fixrsvp_ccnorm", ref_cc["ccnorm"], cand_cc["ccnorm"], stable),
        ):
            rows.extend(
                {
                    "metric": metric,
                    "session": session,
                    "unit_id": int(unit),
                    "ryan": float(ref_value),
                    "candidate": float(cand_value),
                }
                for unit, ref_value, cand_value in zip(
                    ref_units[valid],
                    np.asarray(ref_values)[valid],
                    np.asarray(cand_values)[valid],
                    strict=True,
                )
            )

        support = np.asarray(cand_cc["support"], dtype=bool)
        observation = np.where(support, right["robs_used"], np.nan)
        ref_prediction = np.where(support, left["rhat_used"], np.nan)
        cand_prediction = np.where(support, right["rhat_used"], np.nan)
        ref_r2 = fixrsvp_eval._variance_explained_float64(
            ref_prediction, observation
        )
        cand_r2 = fixrsvp_eval._variance_explained_float64(
            cand_prediction, observation
        )
        valid_r2 = np.isfinite(ref_r2) & np.isfinite(cand_r2)
        rows.extend(
            {
                "metric": "fixrsvp_single_trial_r2",
                "session": session,
                "unit_id": int(unit),
                "ryan": float(ref_value),
                "candidate": float(cand_value),
            }
            for unit, ref_value, cand_value in zip(
                ref_units[valid_r2], ref_r2[valid_r2], cand_r2[valid_r2], strict=True
            )
        )
        audits.append(
            {
                "session": session,
                "n_units": int(len(ref_units)),
                "n_stable_common_ccnorm": int(stable.sum()),
                "n_supported_samples": int(ref_cc["support"].sum()),
                "support_exact_match": support_equal,
                "ccmax_exact_match": ccmax_equal,
                "stability_mask_exact_match": stability_equal,
                "max_abs_ccmax_difference": max_ccmax_difference,
                "ccabs_ccnorm_delta_sign_consistent": True,
                "ccmax_source": ccmax_source,
                "single_trial_r2_recomputed_float64_on_shared_support": True,
            }
        )
    return pd.DataFrame(rows), audits


def hierarchical_summary(table: pd.DataFrame, n_bootstrap: int, seed: int) -> dict:
    groups = [
        group[["ryan", "candidate"]].to_numpy(np.float64)
        for _, group in table.groupby("session", sort=True)
    ]
    if not groups:
        raise RuntimeError("No paired values")
    observed = table[["ryan", "candidate"]].to_numpy(np.float64)
    delta = observed[:, 1] - observed[:, 0]
    rng = np.random.default_rng(seed)
    draws = np.empty((n_bootstrap, 4), dtype=np.float64)
    for draw in range(n_bootstrap):
        chunks = []
        for group_index in rng.integers(0, len(groups), size=len(groups)):
            group = groups[int(group_index)]
            chunks.append(group[rng.integers(0, len(group), size=len(group))])
        values = np.concatenate(chunks)
        difference = values[:, 1] - values[:, 0]
        draws[draw] = (
            np.median(values[:, 0]),
            np.median(values[:, 1]),
            np.median(difference),
            np.mean(difference > 0),
        )
    keys = ("ryan", "candidate", "paired_difference", "fraction_candidate_greater")
    summary = {
        "n_units": int(len(table)),
        "n_sessions": int(table.session.nunique()),
        "ryan": {"median": float(np.median(observed[:, 0]))},
        "candidate": {"median": float(np.median(observed[:, 1]))},
        "paired_difference": {"median": float(np.median(delta))},
        "fraction_candidate_greater": {"estimate": float(np.mean(delta > 0))},
        "bootstrap": {
            "method": "paired hierarchical bootstrap: sessions, then neurons",
            "n_draws": int(n_bootstrap),
            "seed": int(seed),
        },
    }
    for index, key in enumerate(keys):
        summary[key]["ci95"] = [
            float(value) for value in np.quantile(draws[:, index], [0.025, 0.975])
        ]
    return summary


def render(
    tables: pd.DataFrame,
    summaries: dict,
    reference_label: str,
    candidate_label: str,
    output: Path,
) -> None:
    all_definitions = (
        ("validation_bps", "common held-out validation", "bits/spike"),
        ("fixrsvp_ccabs", "FixRSVP raw PSTH", "CCabs"),
        ("fixrsvp_ccnorm", "FixRSVP noise-adjusted", "CCnorm"),
        ("fixrsvp_single_trial_r2", "FixRSVP single-trial", "$R^2$"),
    )
    definitions = tuple(
        definition for definition in all_definitions
        if definition[0] in summaries
    )
    fig, axes = plt.subplots(
        1, len(definitions), figsize=(3.55 * len(definitions), 4.4),
        squeeze=False,
    )
    for axis, (metric, title, unit_label) in zip(axes[0], definitions, strict=True):
        table = tables.loc[tables.metric.eq(metric)]
        summary = summaries[metric]
        values = table[["ryan", "candidate"]].to_numpy(float)
        # The training/selection reduction explicitly clips negative per-unit
        # BPS to zero before averaging.  A few nearly silent cells can have
        # enormous negative raw BPS and otherwise collapse the scientifically
        # relevant part of the validation scatter into a single pixel.  Match
        # the training display convention while retaining the raw values in
        # the CSV and all numerical summaries.
        n_negative_clipped = 0
        if metric == "validation_bps":
            n_negative_clipped = int(np.any(values < 0, axis=1).sum())
            plot_values = np.clip(values, 0.0, None)
        else:
            plot_values = values
        full_lo = float(np.nanmin(plot_values))
        full_hi = float(np.nanmax(plot_values))
        robust_lo, robust_hi = np.nanquantile(plot_values, [0.01, 0.99])
        robust_span = float(robust_hi - robust_lo)
        use_robust_limits = (full_hi - full_lo) > 3.0 * max(robust_span, 1e-12)
        lo = float(robust_lo if use_robust_limits else full_lo)
        hi = float(robust_hi if use_robust_limits else full_hi)
        pad = max(0.04 * (hi - lo), 0.004)
        axis.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.55", lw=0.9)
        display_values = np.clip(plot_values, lo - pad, hi + pad)
        n_edge_clipped = int(np.any(display_values != plot_values, axis=1).sum())
        axis.scatter(
            display_values[:, 0], display_values[:, 1], s=9, alpha=0.18,
            color="#2D6FA3", edgecolors="none", rasterized=True,
        )
        axis.scatter(
            summary["ryan"]["median"], summary["candidate"]["median"],
            marker="D", s=52, color="#C44E36", edgecolor="white", lw=0.7,
        )
        axis.set(
            title=title,
            xlabel=f"{reference_label} {unit_label}",
            ylabel=f"{candidate_label} {unit_label}",
            xlim=(lo - pad, hi + pad),
            ylim=(lo - pad, hi + pad),
        )
        axis.set_aspect("equal", adjustable="box")
        delta = summary["paired_difference"]
        precision = 4 if metric == "fixrsvp_single_trial_r2" else 3
        display_notes = []
        if n_negative_clipped:
            display_notes.append(f"{n_negative_clipped} BPS<0 shown at 0")
        if n_edge_clipped:
            display_notes.append(f"{n_edge_clipped} outliers edge-clipped")
        clipped_note = (
            "\n" + "; ".join(display_notes) if display_notes else ""
        )
        support_note = ""
        if metric == "validation_bps" and "common_source_bins" in summary:
            support_note = (
                f"\n{summary['common_source_bins']:,} shared 120-Hz bins; "
                f"{summary['fraction_candidate_validation']:.1%}/"
                f"{summary['fraction_reference_validation']:.1%} of each split"
            )
        axis.text(
            0.03,
            0.97,
            (
                f"n={summary['n_units']:,} / {summary['n_sessions']} sessions\n"
                f"{reference_label} {summary['ryan']['median']:.{precision}f}\n"
                f"{candidate_label} {summary['candidate']['median']:.{precision}f}\n"
                f"paired Δ {delta['median']:+.{precision}f}\n"
                f"95% CI [{delta['ci95'][0]:+.{precision}f}, {delta['ci95'][1]:+.{precision}f}]"
                f"{support_note}"
                f"{clipped_note}"
            ),
            transform=axis.transAxes,
            va="top",
            fontsize=7.6,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.85", "alpha": 0.9},
        )
        axis.grid(alpha=0.16)
    fig.suptitle(
        f"Audited common-unit comparison: {reference_label} versus {candidate_label}",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), pad=0.8, w_pad=1.1)
    save_options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.08}
    fig.savefig(output, dpi=190, **save_options)
    fig.savefig(output.with_suffix(".pdf"), **save_options)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.ccnorm_splits is not None:
        if args.ccnorm_splits < 1:
            raise ValueError("--ccnorm-splits must be positive")
        fixrsvp_eval.CCNORM_N_SPLITS = int(args.ccnorm_splits)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    has_separate_validation = args.ryan_val is not None or args.candidate_val is not None
    if args.fix_only:
        if args.paired_val is not None or has_separate_validation or args.separate_val_reports:
            raise ValueError("--fix-only cannot be combined with validation inputs")
        validation = pd.DataFrame(
            columns=["metric", "session", "unit_id", "ryan", "candidate"]
        )
        validation_inputs = {}
        validation_bps = None
    elif args.paired_val is not None:
        if args.ryan_val is not None or args.candidate_val is not None:
            raise ValueError("--paired-val cannot be combined with separate validation paths")
        validation, paired_validation_report = load_paired_validation(
            args.paired_val.resolve()
        )
        validation_inputs = {"paired_validation": str(args.paired_val.resolve())}
        validation_bps = {
            "support": "exact common held-out source bins and per-unit data filters",
            "ryan": float(paired_validation_report["reference_bps_overall"]),
            "candidate": float(paired_validation_report["candidate_bps_overall"]),
            "difference": float(
                paired_validation_report["candidate_bps_overall"]
                - paired_validation_report["reference_bps_overall"]
            ),
            "common_source_bins": int(
                sum(paired_validation_report["shared_geometry_by_session"].values())
            ),
            "candidate_validation_source_bins": int(
                sum(paired_validation_report["candidate_geometry_by_session"].values())
            ),
            "reference_validation_source_bins": int(
                sum(paired_validation_report["reference_geometry_by_session"].values())
            ),
        }
        validation_bps["fraction_candidate_validation"] = (
            validation_bps["common_source_bins"]
            / validation_bps["candidate_validation_source_bins"]
        )
        validation_bps["fraction_reference_validation"] = (
            validation_bps["common_source_bins"]
            / validation_bps["reference_validation_source_bins"]
        )
    elif args.separate_val_reports or has_separate_validation:
        if args.ryan_val is None or args.candidate_val is None:
            raise ValueError(
                "--separate-val-reports requires --ryan-val and --candidate-val"
            )
        validation, ryan_report, candidate_report = load_validation(
            args.ryan_val.resolve(), args.candidate_val.resolve()
        )
        validation_inputs = {
            "ryan_validation": str(args.ryan_val.resolve()),
            "candidate_validation": str(args.candidate_val.resolve()),
        }
        validation_bps = {
            "support": "separate reports with identical recorded sample counts",
            "ryan": float(ryan_report["bps_overall"]),
            "candidate": float(candidate_report["bps_overall"]),
            "difference": float(
                candidate_report["bps_overall"] - ryan_report["bps_overall"]
            ),
        }
    else:
        raise ValueError(
            "Choose --paired-val, --separate-val-reports, or --fix-only"
        )
    fixrsvp, audits = load_fixrsvp(
        args.ryan_fix.resolve(),
        args.candidate_fix.resolve(),
        max_sessions=args.max_fix_sessions,
        candidate_metric_audit=(
            args.candidate_metric_audit.resolve()
            if args.candidate_metric_audit is not None
            else None
        ),
    )
    paired = (
        fixrsvp.reset_index(drop=True)
        if validation.empty
        else pd.concat([validation, fixrsvp], ignore_index=True)
    )
    paired.to_csv(args.out_dir / "paired_common_unit_metrics.csv", index=False)
    metric_order = (
        "validation_bps",
        "fixrsvp_ccabs",
        "fixrsvp_ccnorm",
        "fixrsvp_single_trial_r2",
    )
    metric_order = tuple(
        metric for metric in metric_order if paired.metric.eq(metric).any()
    )
    summaries = {
        metric: hierarchical_summary(
            paired.loc[paired.metric.eq(metric)],
            args.n_bootstrap,
            args.seed + index,
        )
        for index, metric in enumerate(metric_order)
    }
    if validation_bps is not None and "validation_bps" in summaries:
        for key in (
            "common_source_bins",
            "fraction_candidate_validation",
            "fraction_reference_validation",
        ):
            if key in validation_bps:
                summaries["validation_bps"][key] = validation_bps[key]
    report = {
        "reference_label": args.reference_label,
        "candidate_label": args.candidate_label,
        "inputs": {
            **validation_inputs,
            "reference_fixrsvp": str(args.ryan_fix.resolve()),
            "candidate_fixrsvp": str(args.candidate_fix.resolve()),
            **(
                {"candidate_metric_audit": str(args.candidate_metric_audit.resolve())}
                if args.candidate_metric_audit is not None
                else {}
            ),
        },
        "validation_bps": validation_bps,
        "metric_invariants": {
            "support": "identical finite observation and nonzero finite data-filter samples",
            "noise_ceiling": "identical data-only CCmax for reference and candidate per unit",
            "identity": "CCnorm = CCabs / CCmax exactly on every retained unit",
            "delta_order": "candidate-minus-reference CCabs and CCnorm have identical sign per unit",
            "sessions": audits,
        },
        "metrics": summaries,
    }
    (args.out_dir / "audited_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    output = args.out_dir / "audited_common_unit_comparison.png"
    render(
        paired,
        summaries,
        args.reference_label,
        args.candidate_label,
        output,
    )
    print(json.dumps(report, indent=2))
    print(output)


if __name__ == "__main__":
    main()
