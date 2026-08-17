#!/usr/bin/env python3
"""Localize why M66 matches validation likelihood but trails Ryan on FixRSVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RYAN_FIX = ROOT / "outputs/cache/fig3_digitaltwin.pkl"
DEFAULT_M66_FIX = ROOT / "outputs/dekel240_paper/final/cache_fig3/fig3_digitaltwin.pkl"
DEFAULT_RYAN_VAL = ROOT / "outputs/dekel240_evaluation/Ryan_05_lr5e-4_epoch471_val_full_per_unit.npz"
DEFAULT_M66_VAL = ROOT / "outputs/dekel240_evaluation/M66a_epoch31_val_full_per_unit.npz"
DEFAULT_M66_CHECKPOINT = Path(
    "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/dekel240/"
    "D240M66a_m63e31_m64be63_readout_coord_teacher0p4_s201/"
    "analysis_candidates/epoch=031-endpoint.ckpt"
)
DEFAULT_FEM = ROOT / "outputs/dekel240_evaluation/M66_vs_Ryan_fixrsvp_femfraction_aligned/paired_femfraction.csv"
DEFAULT_OUTPUT = ROOT / "outputs/dekel240_evaluation/M66_vs_Ryan_failure_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ryan-fix", type=Path, default=DEFAULT_RYAN_FIX)
    parser.add_argument("--m66-fix", type=Path, default=DEFAULT_M66_FIX)
    parser.add_argument("--ryan-val", type=Path, default=DEFAULT_RYAN_VAL)
    parser.add_argument("--m66-val", type=Path, default=DEFAULT_M66_VAL)
    parser.add_argument("--m66-checkpoint", type=Path, default=DEFAULT_M66_CHECKPOINT)
    parser.add_argument("--fem-csv", type=Path, default=DEFAULT_FEM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_sessions(path: Path) -> dict[str, dict]:
    with path.open("rb") as stream:
        rows = dill.load(stream)
    return {str(row["session"]): row for row in rows}


def load_validation(reference_path: Path, candidate_path: Path) -> pd.DataFrame:
    rows = []
    with (
        np.load(reference_path, allow_pickle=False) as reference,
        np.load(candidate_path, allow_pickle=False) as candidate,
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
            ref_bps = reference[f"bps_{index}"].astype(float)
            cand_bps = candidate[f"bps_{index}"].astype(float)
            valid = np.isfinite(ref_bps) & np.isfinite(cand_bps)
            for cid, left, right in zip(
                ref_cids[valid], ref_bps[valid], cand_bps[valid], strict=True
            ):
                rows.append(
                    {
                        "session": str(session),
                        "cid": int(cid),
                        "ryan_val_bps": float(left),
                        "m66_val_bps": float(right),
                        "delta_val_bps": float(right - left),
                    }
                )
    return pd.DataFrame(rows)


def trace_features(trace: np.ndarray, eligible: np.ndarray) -> dict[str, float]:
    trace = np.asarray(trace, dtype=float)
    valid = np.asarray(eligible, dtype=bool) & np.isfinite(trace)
    values = trace[valid]
    if len(values) < 20:
        return {"std": np.nan, "roughness": np.nan, "hf15_fraction": np.nan}
    centered = values - values.mean()
    variance = float(np.mean(centered**2))
    if variance <= 1e-12:
        return {"std": 0.0, "roughness": np.nan, "hf15_fraction": np.nan}
    roughness = float(np.mean(np.diff(values) ** 2) / (2.0 * variance))
    frequency = np.fft.rfftfreq(len(values), d=1 / 120.0)
    power = np.abs(np.fft.rfft(centered)) ** 2
    non_dc = frequency > 0
    denominator = float(power[non_dc].sum())
    hf_fraction = (
        float(power[frequency >= 15.0].sum() / denominator)
        if denominator > 0
        else np.nan
    )
    return {
        "std": float(np.sqrt(variance)),
        "roughness": roughness,
        "hf15_fraction": hf_fraction,
    }


def finite_summary(values) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "median": None, "q25": None, "q75": None}
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def spearman_summary(frame: pd.DataFrame, left: str, right: str) -> dict:
    values = frame[[left, right]].to_numpy(float)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 3:
        return {"n": int(len(values)), "rho": None, "p": None}
    result = spearmanr(values[:, 0], values[:, 1])
    return {"n": int(len(values)), "rho": float(result.statistic), "p": float(result.pvalue)}


def grouped_summary(frame: pd.DataFrame, group: str) -> dict:
    output = {}
    for name, values in frame.groupby(group, observed=True):
        output[str(name)] = {
            "n": int(len(values)),
            "delta_ccabs": finite_summary(values["delta_ccabs"]),
            "delta_ccnorm_shared": finite_summary(values["delta_ccnorm_shared"]),
            "delta_single_trial_r2": finite_summary(values["delta_single_trial_r2"]),
            "delta_val_bps": finite_summary(values["delta_val_bps"]),
        }
    return output


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ryan = load_sessions(args.ryan_fix.resolve())
    m66 = load_sessions(args.m66_fix.resolve())
    if set(ryan) != set(m66):
        raise RuntimeError("FixRSVP session sets differ")

    checkpoint = torch.load(args.m66_checkpoint, map_location="cpu", weights_only=False)
    dataset_cids = checkpoint["hyper_parameters"]["dataset_cids"]
    rows = []
    for session in sorted(ryan):
        left = ryan[session]
        right = m66[session]
        units = np.asarray(left["neuron_mask"], dtype=int)
        if not np.array_equal(units, np.asarray(right["neuron_mask"], dtype=int)):
            raise RuntimeError(f"Neuron masks differ for {session}")
        robs = np.asarray(left["robs_used"], dtype=float)
        dfs = np.asarray(left["dfs_used"], dtype=float)
        if not np.array_equal(robs, np.asarray(right["robs_used"]), equal_nan=True):
            raise RuntimeError(f"Observations differ for {session}")
        if not np.array_equal(dfs, np.asarray(right["dfs_used"]), equal_nan=True):
            raise RuntimeError(f"Filters differ for {session}")

        data_valid = np.isfinite(robs) & np.isfinite(dfs) & (dfs != 0)
        n_valid = data_valid.sum(axis=0)
        ryan_trials = np.asarray(left["rhat_used"], dtype=float)
        m66_trials = np.asarray(right["rhat_used"], dtype=float)
        ryan_ccabs = np.asarray(left["ccabs"], dtype=float)
        m66_ccabs = np.asarray(right["ccabs"], dtype=float)
        shared_ccmax = np.asarray(right["ccmax"], dtype=float)
        session_cids = np.asarray(dataset_cids[session], dtype=int)
        for local, source_unit in enumerate(units):
            eligible = n_valid[:, local] >= 20
            data_features = trace_features(np.asarray(right["robs_mean"])[:, local], eligible)
            ryan_features = trace_features(np.asarray(left["rhat_mean"])[:, local], eligible)
            m66_features = trace_features(np.asarray(right["rhat_mean"])[:, local], eligible)
            ceiling = float(shared_ccmax[local])
            shared_valid = (
                np.isfinite(ryan_ccabs[local])
                and np.isfinite(m66_ccabs[local])
                and np.isfinite(ceiling)
                and ceiling > 1e-3
            )

            sample_mask = (
                data_valid[:, :, local]
                & np.isfinite(ryan_trials[:, :, local])
                & np.isfinite(m66_trials[:, :, local])
            )
            observation = robs[:, :, local][sample_mask]
            observation_variance = float(np.var(observation))

            def variance_decomposition(prediction):
                prediction = prediction[:, :, local][sample_mask]
                covariance = float(
                    np.mean(
                        (observation - observation.mean())
                        * (prediction - prediction.mean())
                    )
                )
                return (
                    covariance / observation_variance,
                    float(np.var(prediction)) / observation_variance,
                )

            ryan_covariance_ratio, ryan_prediction_variance_ratio = (
                variance_decomposition(ryan_trials)
            )
            m66_covariance_ratio, m66_prediction_variance_ratio = (
                variance_decomposition(m66_trials)
            )
            rows.append(
                {
                    "session": session,
                    "subject": session.split("_")[0],
                    "source_unit_index": int(source_unit),
                    "cid": int(session_cids[source_unit]),
                    "ccmax": ceiling,
                    "ryan_ccabs": float(ryan_ccabs[local]),
                    "m66_ccabs": float(m66_ccabs[local]),
                    "delta_ccabs": float(m66_ccabs[local] - ryan_ccabs[local]),
                    "ryan_ccnorm_shared": float(ryan_ccabs[local] / ceiling) if shared_valid else np.nan,
                    "m66_ccnorm_shared": float(m66_ccabs[local] / ceiling) if shared_valid else np.nan,
                    "delta_ccnorm_shared": float((m66_ccabs[local] - ryan_ccabs[local]) / ceiling)
                    if shared_valid
                    else np.nan,
                    "ryan_single_trial_r2": float(np.asarray(left["ve_model"])[local]),
                    "m66_single_trial_r2": float(np.asarray(right["ve_model"])[local]),
                    "delta_single_trial_r2": float(
                        np.asarray(right["ve_model"])[local] - np.asarray(left["ve_model"])[local]
                    ),
                    "ryan_covariance_ratio": ryan_covariance_ratio,
                    "m66_covariance_ratio": m66_covariance_ratio,
                    "delta_covariance_ratio": (
                        m66_covariance_ratio - ryan_covariance_ratio
                    ),
                    "ryan_prediction_variance_ratio": ryan_prediction_variance_ratio,
                    "m66_prediction_variance_ratio": m66_prediction_variance_ratio,
                    "delta_prediction_variance_ratio": (
                        m66_prediction_variance_ratio
                        - ryan_prediction_variance_ratio
                    ),
                    "mean_rate_hz": float(np.nanmean(np.where(data_valid[:, :, local], robs[:, :, local], np.nan)) * 120),
                    "data_psth_std": data_features["std"],
                    "ryan_psth_std": ryan_features["std"],
                    "m66_psth_std": m66_features["std"],
                    "data_roughness": data_features["roughness"],
                    "ryan_roughness": ryan_features["roughness"],
                    "m66_roughness": m66_features["roughness"],
                    "data_hf15_fraction": data_features["hf15_fraction"],
                    "ryan_hf15_fraction": ryan_features["hf15_fraction"],
                    "m66_hf15_fraction": m66_features["hf15_fraction"],
                }
            )

    table = pd.DataFrame(rows)
    validation = load_validation(args.ryan_val.resolve(), args.m66_val.resolve())
    table = table.merge(validation, on=["session", "cid"], how="left", validate="one_to_one")
    fem = pd.read_csv(args.fem_csv)
    fem = fem.rename(
        columns={
            "unit_id": "source_unit_index",
            "empirical": "empirical_fem_fraction",
            "model": "m66_fem_fraction",
            "ryan": "ryan_fem_fraction",
        }
    )
    table = table.merge(
        fem,
        on=["session", "source_unit_index"],
        how="left",
        validate="one_to_one",
    )
    table["fem_fraction_advantage_ryan"] = (
        table["ryan_fem_fraction"] - table["m66_fem_fraction"]
    )
    table["reliability_stratum"] = pd.cut(
        table["ccmax"],
        bins=[-np.inf, 0.60, 0.85, np.inf],
        labels=["low (≤0.60)", "medium (0.60–0.85)", "high (>0.85)"],
    )
    table["data_hf_quartile"] = pd.qcut(
        table["data_hf15_fraction"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"]
    )
    table["data_roughness_quartile"] = pd.qcut(
        table["data_roughness"], 4, labels=["Q1 smooth", "Q2", "Q3", "Q4 rough"]
    )
    table.to_csv(args.output_dir / "paired_failure_metrics.csv", index=False)

    reliable = table["ccmax"] > 0.85
    matched_val = table["delta_val_bps"].notna()
    report = {
        "n_units": int(len(table)),
        "n_sessions": int(table.session.nunique()),
        "mask_contract": {
            "observations_and_filters_exact_across_models": True,
            "minimum_trials_for_ccabs_time_bin": 20,
            "shared_ccmax_source": str(args.m66_fix.resolve()),
        },
        "overall": {
            key: finite_summary(table[key])
            for key in (
                "delta_ccabs",
                "delta_ccnorm_shared",
                "delta_single_trial_r2",
                "delta_val_bps",
                "ryan_covariance_ratio",
                "m66_covariance_ratio",
                "delta_covariance_ratio",
                "ryan_prediction_variance_ratio",
                "m66_prediction_variance_ratio",
                "delta_prediction_variance_ratio",
                "data_hf15_fraction",
                "ryan_hf15_fraction",
                "m66_hf15_fraction",
                "data_roughness",
                "ryan_roughness",
                "m66_roughness",
                "data_psth_std",
                "ryan_psth_std",
                "m66_psth_std",
            )
        },
        "reliable_cells": {
            "n": int(reliable.sum()),
            "delta_ccabs": finite_summary(table.loc[reliable, "delta_ccabs"]),
            "delta_ccnorm_shared": finite_summary(table.loc[reliable, "delta_ccnorm_shared"]),
            "delta_single_trial_r2": finite_summary(
                table.loc[reliable, "delta_single_trial_r2"]
            ),
        },
        "by_reliability": grouped_summary(table, "reliability_stratum"),
        "by_subject": grouped_summary(table, "subject"),
        "by_data_hf_quartile": grouped_summary(table, "data_hf_quartile"),
        "by_data_roughness_quartile": grouped_summary(
            table, "data_roughness_quartile"
        ),
        "associations": {
            "ryan_ccabs_advantage_vs_ccmax": spearman_summary(
                table.assign(ryan_advantage=-table.delta_ccabs),
                "ryan_advantage",
                "ccmax",
            ),
            "ryan_ccabs_advantage_vs_data_hf": spearman_summary(
                table.assign(ryan_advantage=-table.delta_ccabs),
                "ryan_advantage",
                "data_hf15_fraction",
            ),
            "ryan_ccabs_advantage_vs_data_roughness": spearman_summary(
                table.assign(ryan_advantage=-table.delta_ccabs),
                "ryan_advantage",
                "data_roughness",
            ),
            "fix_ccabs_delta_vs_validation_bps_delta": spearman_summary(
                table, "delta_ccabs", "delta_val_bps"
            ),
            "fix_ccabs_delta_vs_fem_fraction_advantage_ryan": spearman_summary(
                table, "delta_ccabs", "fem_fraction_advantage_ryan"
            ),
        },
        "validation_overlap": {
            "n": int(matched_val.sum()),
            "m66_fix_wins_fraction": float(np.mean(table.loc[matched_val, "delta_ccabs"] > 0)),
            "m66_validation_wins_fraction": float(
                np.mean(table.loc[matched_val, "delta_val_bps"] > 0)
            ),
            "both_win_fraction": float(
                np.mean(
                    (table.loc[matched_val, "delta_ccabs"] > 0)
                    & (table.loc[matched_val, "delta_val_bps"] > 0)
                )
            ),
        },
        "fem_fraction": {
            "n": int(table["m66_fem_fraction"].notna().sum()),
            "empirical": finite_summary(table["empirical_fem_fraction"]),
            "ryan": finite_summary(table["ryan_fem_fraction"]),
            "m66": finite_summary(table["m66_fem_fraction"]),
            "ryan_minus_m66": finite_summary(table["fem_fraction_advantage_ryan"]),
        },
    }
    session_summary = (
        table.groupby(["subject", "session"], as_index=False)
        .agg(
            n_units=("cid", "size"),
            median_delta_ccabs=("delta_ccabs", "median"),
            median_delta_ccnorm_shared=("delta_ccnorm_shared", "median"),
            median_delta_single_trial_r2=("delta_single_trial_r2", "median"),
            median_delta_val_bps=("delta_val_bps", "median"),
        )
    )
    session_summary.to_csv(args.output_dir / "session_summary.csv", index=False)
    report["session_association_fix_vs_validation"] = spearman_summary(
        session_summary, "median_delta_ccabs", "median_delta_val_bps"
    )
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.7), constrained_layout=True)
    reliability_order = ["low (≤0.60)", "medium (0.60–0.85)", "high (>0.85)"]
    reliability_values = [
        table.loc[table.reliability_stratum == label, "delta_ccabs"].dropna().to_numpy()
        for label in reliability_order
    ]
    axes[0].boxplot(reliability_values, labels=["low", "medium", "high"], showfliers=False)
    axes[0].axhline(0, color="0.55", linewidth=1)
    axes[0].set(
        title="A  Shape loss spans reliability levels",
        xlabel="data reliability (CCmax)",
        ylabel="M66 − Ryan CCabs",
    )

    hf_order = ["Q1 low", "Q2", "Q3", "Q4 high"]
    hf_values = [
        table.loc[table.data_hf_quartile == label, "delta_ccabs"].dropna().to_numpy()
        for label in hf_order
    ]
    axes[1].boxplot(hf_values, labels=["Q1", "Q2", "Q3", "Q4"], showfliers=False)
    axes[1].axhline(0, color="0.55", linewidth=1)
    axes[1].set(
        title="B  Not primarily a fast-PSTH failure",
        xlabel="data PSTH power above 15 Hz",
        ylabel="M66 − Ryan CCabs",
    )

    overlap = table.dropna(subset=["delta_val_bps", "delta_ccabs"])
    axes[2].scatter(
        overlap["delta_val_bps"],
        overlap["delta_ccabs"],
        s=9,
        alpha=0.22,
        color="#2d6fa3",
        edgecolors="none",
        rasterized=True,
    )
    axes[2].axhline(0, color="0.55", linewidth=1)
    axes[2].axvline(0, color="0.55", linewidth=1)
    association = report["associations"]["fix_ccabs_delta_vs_validation_bps_delta"]
    axes[2].text(
        0.04,
        0.96,
        f"Spearman ρ={association['rho']:.2f}",
        transform=axes[2].transAxes,
        va="top",
    )
    axes[2].set(
        title="C  Validation gains do not transfer",
        xlabel="M66 − Ryan validation bits/spike",
        ylabel="M66 − Ryan FixRSVP CCabs",
    )
    stem = args.output_dir / "m66_fixrsvp_failure_localization"
    fig.savefig(stem.with_suffix(".png"), dpi=240)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
