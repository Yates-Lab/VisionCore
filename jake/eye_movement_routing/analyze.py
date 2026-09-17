"""Exploratory population routing from the selected Figure 4 replay.

Groups are defined by independently assayed TF/SF tuning. Movement ordering
uses spectral power on selection images only; effects use separate images.
This script does not modify the production Figure 4 or manuscript.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.fig4.spatiotemporal_tuning._spectral_shards import load_and_merge_shards

COLORS = ("#D58120", "#197EAC")
LABELS = ("Higher SF / lower TF", "Lower SF / higher TF")
SEED = 20260914


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def group_effects(rate, spikes, ssi, groups, bins, image_rows, trace_draws=None):
    """Median unit effect after averaging matched images and bin trajectories."""
    rr = rate[image_rows].mean(axis=0)
    counts = spikes[image_rows].sum(axis=0)
    info = (spikes[image_rows] * ssi[image_rows]).sum(axis=0)
    result = np.empty((2, 2, len(bins)))
    for b, rows in enumerate(bins if trace_draws is None else trace_draws):
        rate_bin = rr[rows].mean(axis=0)
        ssi_bin = info[rows].sum(axis=0) / np.maximum(counts[rows].sum(axis=0), 1e-12)
        for m, values in enumerate((rate_bin, ssi_bin)):
            effect = 100 * (values[1] / np.maximum(values[0], 1e-12) - 1)
            for g, units in enumerate(groups):
                result[m, g, b] = np.median(effect[units])
    return result


def match_paths(path_length, spectral_balance):
    """Contrast spectral balance within path deciles, at most 15% mismatch."""
    pairs = []
    for stratum in np.array_split(np.argsort(path_length), 10):
        ordered = stratum[np.argsort(spectral_balance[stratum])]
        n = len(stratum)//3
        low, high = ordered[:n], ordered[-n:]
        cost = np.abs(np.log(path_length[low, None]/path_length[high][None, :]))
        left, right = linear_sum_assignment(cost)
        pairs.extend((low[i], high[j]) for i, j in zip(left, right) if cost[i, j] < np.log(1.15))
    return np.asarray(pairs, dtype=int)


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})


def draw_curves(ax, center, low, high, ylabel):
    x = np.arange(5)
    for g in range(2):
        ax.fill_between(x, low[g], high[g], color=COLORS[g], alpha=.14, linewidth=0)
        ax.plot(x, center[g], "o-", color=COLORS[g], lw=2.3, ms=5, label=LABELS[g])
    ax.axhline(0, color=".65", lw=.8, zorder=0)
    ax.set_xticks(x, ["1", "2", "3", "4", "5"])
    ax.set_xlabel("Movement spectral balance (quintile)\n← slower-tuned drive     faster-tuned drive →")
    ax.set_ylabel(ylabel)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=ROOT / "outputs/no_phase_readout_comparison_20260910/rank1/figure4")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs/eye_movement_routing_20260914")
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = json.loads((ROOT / "manuscript/analysis/selected_model_bundle.json").read_text())
    shards = [args.bundle / f"matrix_spectral_replay/shard_0{i}/causal_chain_shard.npz" for i in range(2)]
    for shard in shards:
        metadata = json.loads((shard.parent / "summary.json").read_text())
        if metadata["checkpoint_sha256"] != selected["checkpoint_sha256"]:
            raise ValueError("Replay checkpoint differs from selected manuscript checkpoint")
    data = load_and_merge_shards(shards)
    if not np.array_equal(data["motion_scales"], [0, 1]):
        raise ValueError("Expected stabilized, measured condition order")
    tuning_path = args.bundle / "all_available_yu_tuning/tuning_summary.csv"
    tuning = pd.read_csv(tuning_path).set_index("unit_index").loc[data["unit_indices"]].reset_index()
    if tuning[["session", "cid"]].duplicated().any():
        raise ValueError("Repeated unit identity")
    valid = np.flatnonzero(tuning.validated_for_figure4.to_numpy(bool))
    speed = np.log2(tuning.exact_twin_yu_preferred_tf_hz / tuning.exact_twin_yu_preferred_sf_cpd).to_numpy()
    order = valid[np.argsort(speed[valid], kind="stable")]
    n = len(order) // 3
    groups = np.array([order[:n], order[-n:]])
    # This split and the tuning groups do not consult replay response outcomes.
    permutation = np.random.default_rng(SEED).permutation(len(data["image_indices"]))
    selection_images, evaluation_images = permutation[:20], permutation[20:]
    power = np.mean(data["joint_passband_power"][selection_images, :, 1], axis=0)
    log_power = np.log(np.maximum(power, 1e-30))
    spectral_balance = log_power[:, groups[1]].mean(axis=1) - log_power[:, groups[0]].mean(axis=1)
    bins = np.array_split(np.argsort(spectral_balance, kind="stable"), 5)
    trace_table = pd.read_csv(args.bundle / "response_matrix_40img_x_200fix/merged/trace_feature_table.csv")
    if not np.array_equal(trace_table.trace_bank_index, data["trace_indices"]):
        raise ValueError("Trace identities do not match")
    trace_table["spectral_balance_log_ratio"] = spectral_balance
    trace_table["routing_quintile"] = 0
    for b, rows in enumerate(bins):
        trace_table.loc[rows, "routing_quintile"] = b + 1
    trace_table.to_csv(args.out_dir / "movement_groups.csv", index=False)
    tuning["routing_group"] = "excluded middle third or unvalidated"
    for g, rows in enumerate(groups):
        tuning.loc[rows, "routing_group"] = LABELS[g]
    tuning.to_csv(args.out_dir / "unit_groups.csv", index=False)
    rate, spikes, ssi = (data[k].astype(np.float64) for k in ("mean_rate", "expected_spikes", "map_ssi"))
    point = group_effects(rate, spikes, ssi, groups, bins, evaluation_images)
    rng = np.random.default_rng(SEED + 1)
    boot = np.empty((args.bootstrap, *point.shape))
    # Groups are fixed. Images and movements, shared by both groups, are the
    # two resampled experimental axes; cells are not independent repetitions.
    for b in range(args.bootstrap):
        rows = rng.choice(evaluation_images, len(evaluation_images), replace=True)
        draws = [rng.choice(v, len(v), replace=True) for v in bins]
        boot[b] = group_effects(rate, spikes, ssi, groups, bins, rows, draws)
    lo, hi = np.quantile(boot, [.025, .975], axis=0)
    interaction = boot[:, :, 1, -1] - boot[:, :, 0, -1] - boot[:, :, 1, 0] + boot[:, :, 0, 0]
    # Four evenly spaced members from each bin, chosen before decoding outcomes.
    decode_traces = np.concatenate([b[[4, 14, 25, 35]] for b in bins])
    decode_images = np.sort(evaluation_images)[::2]
    np.savez_compressed(args.out_dir / "routing_analysis.npz", groups=groups,
                        selection_images=selection_images, evaluation_images=evaluation_images,
                        spectral_balance=spectral_balance, bins=np.array(bins),
                        center=point, ci_low=lo, ci_high=hi, bootstrap_effects=boot,
                        decode_traces=decode_traces, decode_images=decode_images)
    report = {
        "status": "exploratory model analysis; no attention manipulation",
        "checkpoint_sha256": selected["checkpoint_sha256"], "seed": SEED,
        "group_rule": "equal-sized outer thirds of log2(TF/SF), among 145 strictly validated model units",
        "group_sizes": [n, n], "selection_images": selection_images.tolist(),
        "evaluation_images": evaluation_images.tolist(),
        "movement_order": "log geometric-mean passband power ratio (faster/slower group), selection images only",
        "same_movies_for_both_groups": True,
        "interval": "95% crossed bootstrap of evaluation image patches and trajectories within fixed quintiles; units held fixed",
        "outcomes": {},
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in [*shards, tuning_path]},
        "limits": ["Movement quintiles are not path-length matched.",
                   "Group rule and movement ordering were explored in this task; this is not a preregistered confirmation.",
                   "Evaluation images were not used to order movements; all predictions come from the same fitted model.",
                   "SSI is the existing spatial information measure, not measured decoding accuracy."]}
    for m, label in enumerate(["rate_percent", "ssi_percent"]):
        report["outcomes"][label] = {"center": point[m].tolist(), "ci_low": lo[m].tolist(), "ci_high": hi[m].tolist(),
            "group_by_movement_interaction_ci95": np.quantile(interaction[:, m], [.025,.975]).tolist()}
    path_length = trace_table.rendered_path_length_arcmin.to_numpy()
    pairs = match_paths(path_length, spectral_balance)
    matched = group_effects(rate, spikes, ssi, groups, pairs.T, evaluation_images)
    matched_boot = []
    for _ in range(args.bootstrap):
        rows = rng.choice(evaluation_images, len(evaluation_images), replace=True)
        pair_draw = pairs[rng.integers(len(pairs), size=len(pairs))]
        matched_boot.append(group_effects(rate, spikes, ssi, groups, pair_draw.T, rows))
    matched_boot = np.asarray(matched_boot)
    matched_interaction = matched_boot[:,:,1,1]-matched_boot[:,:,0,1]-matched_boot[:,:,1,0]+matched_boot[:,:,0,0]
    report['path_matched_control'] = {
        'rule':'outer spectral-balance thirds within path deciles; unique optimal pairs with <15% path mismatch',
        'n_pairs':len(pairs), 'pairs':pairs.tolist(),
        'median_path_arcmin':np.median(path_length[pairs], axis=0).tolist(),
        'center':matched.tolist(),
        'interaction_ci95':np.quantile(matched_interaction, [.025,.975],axis=0).tolist(),
        'axes':['outcome (rate, SSI)','unit group','lower/higher spectral balance within path stratum']}
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for m, ax in enumerate(axes):
        draw_curves(ax, point[m], lo[m], hi[m], ["Firing-rate change (%)", "Spatial information change (%)"][m])
        ax.set_title(["A   Population response gain", "B   Spatial information per spike"][m], loc="left", fontweight="bold", pad=15)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("The same eye movements affect two tuning-defined populations differently", fontsize=13, y=.99)
    fig.text(.5, .015, "48 model units per group · 20 evaluation images × 200 measured trajectories · bands: 95% image/trajectory bootstrap", ha="center", fontsize=8, color=".4")
    fig.tight_layout(rect=[0,.07,1,.95])
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(args.out_dir / f"routing_two_panels.{ext}", dpi=180)
    print(json.dumps(report["outcomes"], indent=2), flush=True)


if __name__ == "__main__":
    main()
