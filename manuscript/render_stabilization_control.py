#!/usr/bin/env python3
"""Paired comparison of history-, trial-, and global stabilization on Figure 3 C/D.

The replays use the production scoring code and population. They do not fit a
new encoding model or compute a new FEM variance decomposition.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from analysis_selection import SOURCE_ROOT, source_path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_trial_replay_acceptance(trial_cache, accepted_sha256):
    """Return the verified accepted digest, or None when no exception was requested."""
    if accepted_sha256 is None:
        return None
    actual = digest(trial_cache)
    if accepted_sha256 != actual:
        raise ValueError(
            f"Accepted trial replay SHA-256 does not match {trial_cache}: "
            f"expected {accepted_sha256}, actual {actual}"
        )
    return actual


def replay_check_record(
    reference, metric, condition, maximum, tolerance, accepted_trial_sha256
):
    """Record a replay check, failing unless this exact trial cache was accepted."""
    record = {"maximum_absolute_difference": maximum, "tolerance": tolerance}
    if maximum <= tolerance:
        return record
    if reference == "trial" and accepted_trial_sha256 is not None:
        return {
            **record,
            "status": "accepted",
            "accepted_trial_replay_sha256": accepted_trial_sha256,
        }
    raise AssertionError(
        f"Unchanged {reference} replay differs: {metric}/{condition} {maximum}"
    )


def provenance_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(SOURCE_ROOT))
    except ValueError:
        return str(path)


def assemble_control_rows(global_payload, history_payload, trial_payload):
    """Validate both controls and order them like the production cache."""
    controls = {
        "history_endpoint": history_payload,
        "trial_centroid": trial_payload,
    }
    for reference, payload in controls.items():
        if payload.get("schema_version") != global_payload.get("schema_version"):
            raise ValueError(f"{reference} and main-figure caches use different scoring schemas")
        if not payload.get("complete") or payload.get("checkpoint_path") != global_payload.get("checkpoint_path"):
            raise ValueError(f"Incomplete or mismatched {reference} control")

    global_rows = global_payload["results"]
    expected_sessions = {row["session"] for row in global_rows}
    ordered = {}
    checks = {}
    identity_keys = (
        "neuron_mask", "ccmax", "ccnorm_unstable", "matched_var_y",
        "matched_n_windows", "n_base_windows",
    )
    for reference, payload in controls.items():
        by_session = {row["session"]: row for row in payload["results"]}
        if len(by_session) != len(payload["results"]) or set(by_session) != expected_sessions:
            raise ValueError(f"{reference} does not cover the exact main-figure session set")
        rows = [by_session[row["session"]] for row in global_rows]
        checks[reference] = {}
        for global_row, control_row in zip(global_rows, rows):
            if control_row.get("stabilization_reference") != reference:
                raise ValueError(f"Unexpected {reference} stabilization reference")
            session_checks = {}
            for key in identity_keys:
                first = np.asarray(global_row[key])
                second = np.asarray(control_row[key])
                try:
                    same = np.array_equal(first, second, equal_nan=True)
                except TypeError:
                    same = np.array_equal(first, second)
                session_checks[key] = bool(same)
            if not all(session_checks.values()):
                raise AssertionError(
                    f"Data-only scoring quantities changed: {global_row['session']} "
                    f"{reference} {session_checks}"
                )
            checks[reference][global_row["session"]] = session_checks
        ordered[reference] = rows
    return ordered["history_endpoint"], ordered["trial_centroid"], checks


def adjacent_scope_contrasts(values, sessions, mask, bootstrap):
    """Directly pair each neighboring retinal-stabilization scope per unit."""
    return {
        second + "_minus_" + first: bootstrap(
            values[first], values[second], sessions, mask
        )
        for first, second in (
            ("full", "history"),
            ("history", "trial"),
            ("trial", "global"),
        )
    }


def statistics_tex(metrics):
    macros = {}
    for name, record in metrics.items():
        prefix = "StabC" if name == "ccnorm" else "StabD"
        macros[prefix+"Units"] = str(record["n_units"])
        for condition, median in record["medians"].items():
            macros[prefix+condition.capitalize()] = f"{median:.3f}"
        for reference in ("full", "global"):
            stat = record["contrasts"]["history_minus_"+reference]
            macros[prefix+reference.capitalize()+"Delta"] = f"{stat['median']:+.3f}"
            macros[prefix+reference.capitalize()+"CI"] = f"[{stat['ci_low']:+.3f}, {stat['ci_high']:+.3f}]"
        for suffix, contrast in (("MotionLoss", "full_minus_history"),
                                 ("ExtraLoss", "full_minus_retinal"),
                                 ("CostDifference", "retinal_minus_history"),
                                 ("FullToHistory", "history_minus_full"),
                                 ("HistoryToTrial", "trial_minus_history"),
                                 ("TrialToGlobal", "global_minus_trial")):
            stat = record["contrasts"][contrast]
            macros[prefix+suffix] = f"{stat['median']:.3f}"
            macros[prefix+suffix+"CI"] = f"[{stat['ci_low']:+.3f}, {stat['ci_high']:+.3f}]"
    return ("% Generated by render_stabilization_control.py from paired production-population scores.\n" +
            "".join(f"\\newcommand{{\\{key}}}{{{value}}}\n" for key, value in sorted(macros.items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", type=Path, default=SOURCE_ROOT / "outputs/stabilization_control_20260915",
                        help="Directory containing the retained history-local replay")
    parser.add_argument("--trial-inference-dir", type=Path, default=ROOT / "outputs/trial_stabilization_control",
                        help="Directory containing the trial-centroid replay")
    parser.add_argument("--out-dir", type=Path, default=HERE / "build/stabilization_control")
    parser.add_argument(
        "--accept-trial-replay-sha256",
        help="Accept numerical replay drift only for this exact trial cache",
    )
    args = parser.parse_args()
    selection = json.loads((HERE / "analysis/selected_model_bundle.json").read_text())
    manifest = json.loads((SOURCE_ROOT / selection["bundle"] / "figure3/run_manifest.json").read_text())
    os.environ.update({
        key: str(source_path(value)) if Path(value).is_absolute() else value
        for key, value in manifest["environment"].items()
    })
    if cache_dir := os.environ.get("VISIONCORE_MANUSCRIPT_EMPIRICAL_CACHE_DIR"):
        cache_dir = Path(cache_dir).expanduser().resolve()
        os.environ.update(
            FIG3_COVDECOMP_CACHE_PATH=str(cache_dir / "covdecomp_empirical.pkl"),
            FIG3_COVDECOMP_DERIVED_CACHE_PATH=str(cache_dir / "covdecomp_derived.pkl"),
            COVDECOMP_ALIGNED_CACHE_PATH=str(cache_dir / "covdecomp_aligned_sessions.pkl"),
        )
    os.environ["MPLCONFIGDIR"] = str(HERE / "build/mpl")
    sys.path[:0] = [str(ROOT / "paper/fig3"), str(ROOT)]
    import dill
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import _fig3_ablation_data as ab
    from generate_figure3 import _session_cluster_bootstrap, INTACT_COLOR, ABLATED_COLOR, STABILIZED_COLOR

    history_path = args.inference_dir / "history_stabilized.pkl"
    history_manifest = json.loads((args.inference_dir / "run_manifest.json").read_text())
    trial_path = args.trial_inference_dir / "trial_stabilized.pkl"
    accepted_trial_sha256 = validate_trial_replay_acceptance(
        trial_path, args.accept_trial_replay_sha256
    )
    trial_manifest_path = args.trial_inference_dir / "trial_stabilized_run_manifest.json"
    if not trial_manifest_path.exists():
        trial_manifest_path = args.trial_inference_dir / "run_manifest.json"
    trial_manifest = json.loads(trial_manifest_path.read_text())
    for control_path, run_manifest, reference in (
        (history_path, history_manifest, "history_endpoint"),
        (trial_path, trial_manifest, "trial_centroid"),
    ):
        if (
            run_manifest["selection"] != selection
            or run_manifest.get("stabilization_reference", "history_endpoint") != reference
            or digest(control_path) != run_manifest["local_cache_sha256"]
        ):
            raise ValueError(f"{reference} replay provenance differs from selected model or completed cache")
    history_payload = dill.load(history_path.open("rb"))
    trial_payload = dill.load(trial_path.open("rb"))
    global_path = source_path(manifest["environment"]["FIG3_ABLATION_CACHE_PATH"])
    global_payload = dill.load(global_path.open("rb"))
    global_rows = global_payload["results"]
    history_rows, trial_rows, data_checks = assemble_control_rows(
        global_payload, history_payload, trial_payload
    )
    included = ab._load_fig2_included_sessions()
    global_rows = ab._attach_fig2_derived([r for r in global_rows if r["session"] in included])
    history_rows = ab._attach_fig2_derived([r for r in history_rows if r["session"] in included])
    trial_rows = ab._attach_fig2_derived([r for r in trial_rows if r["session"] in included])
    global_data = ab.aggregate(global_rows)
    history_data = ab.aggregate(history_rows)
    trial_data = ab.aggregate(trial_rows)
    pop = ab._fig2_cd_population(global_rows)
    for reference, rows in (("history", history_rows), ("trial", trial_rows)):
        if not np.array_equal(pop, ab._fig2_cd_population(rows)):
            raise AssertionError(f"{reference} population inclusion changed")
    sessions = np.asarray(global_data["sessions"])
    metrics = {
        "ccnorm": {"full": global_data["ccnorm"]["intact"],
                   "retinal": global_data["ccnorm"]["zeroed"],
                   "global": global_data["ccnorm"]["stabilized"],
                   "trial": trial_data["ccnorm"]["stabilized"],
                   "history": history_data["ccnorm"]["stabilized"]},
        "fraction": {"psth": global_data["explainable_fraction"]["psth"],
                     "full": global_data["explainable_fraction"]["intact"],
                     "retinal": global_data["explainable_fraction"]["zeroed"],
                     "global": global_data["explainable_fraction"]["stabilized"],
                     "trial": trial_data["explainable_fraction"]["stabilized"],
                     "history": history_data["explainable_fraction"]["stabilized"]}}
    # Replaying the unchanged inputs is a numerical reproducibility check.
    # Mixed precision may vary slightly across GPU/batch implementations; a
    # 0.005 absolute score tolerance is much smaller than the main effects.
    replay_checks = {}
    for reference, control_data in (("history", history_data), ("trial", trial_data)):
        for metric, tolerance in (("ccnorm", .002), ("explainable_fraction", .005)):
            for condition in ("intact", "zeroed"):
                a, b = global_data[metric][condition], control_data[metric][condition]
                valid = pop & np.isfinite(a) & np.isfinite(b)
                maximum = float(np.max(np.abs(a[valid] - b[valid])))
                key = f"{metric}/{condition}" if reference == "history" else f"trial/{metric}/{condition}"
                replay_checks[key] = replay_check_record(
                    reference,
                    metric,
                    condition,
                    maximum,
                    tolerance,
                    accepted_trial_sha256,
                )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive = HERE / "analysis/stabilization_control"
    archive.mkdir(parents=True, exist_ok=True)
    colors = {"psth": ".55", "full": INTACT_COLOR, "retinal": ABLATED_COLOR,
              "global": STABILIZED_COLOR, "history": "#009e73", "trial": "#7b3294"}
    labels = {"psth": "PSTH", "full": "Full", "retinal": "Retinal",
              "global": "Global", "history": "History-local", "trial": "Trial"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35))
    fig.subplots_adjust(left=.10, right=.98, bottom=.25, top=.88, wspace=.43)
    reports, arrays = {}, {"sessions": sessions, "population": pop,
                          "neuron_ids": np.concatenate([r["neuron_mask"] for r in global_rows])}
    for index, (name, values) in enumerate(metrics.items()):
        ax = axes[index]
        mask = pop & np.logical_and.reduce([np.isfinite(value) for value in values.values()])
        keys = list(values)
        groups = [np.asarray(values[key])[mask] for key in keys]
        positions = np.arange(len(keys), dtype=float)
        bp = ax.boxplot(groups, positions=positions, widths=.54, whis=(10, 90),
                        showfliers=False, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 1.2})
        for patch, key in zip(bp["boxes"], keys):
            patch.set_facecolor(colors[key]); patch.set_alpha(.5)
        ax.set_xticks(positions, [labels[key] for key in keys], rotation=30, ha="right")
        ax.set_title(["Trial-averaged prediction", "Single-trial prediction"][index], fontsize=9)
        ax.set_ylabel(["Normalized correlation", "Fraction of conditional rate\nvariance explained"][index])
        ax.text(-.20, 1.11, "AB"[index], transform=ax.transAxes, fontsize=11, weight="bold")
        if name == "ccnorm":
            ax.set_ylim(.2, 1.)
        else:
            ax.set_ylim(-.3, .8)
            ax.axhline(0, color=".4", lw=.7)
            ax.axhline(float(np.median(values["psth"][mask])), color=".55", lw=.7, ls="--")
        # Guard against cropping any displayed whisker.
        lo, hi = ax.get_ylim()
        if any(np.percentile(v, 10) < lo or np.percentile(v, 90) > hi for v in groups):
            raise AssertionError(f"Displayed {name} whisker outside plotting bounds")
        reports[name] = {"n_units": int(mask.sum()), "n_sessions": len(np.unique(sessions[mask])),
                         "medians": {key: float(np.median(values[key][mask])) for key in keys},
                         "contrasts": {}}
        for key in keys:
            arrays[name+"_"+key] = np.asarray(values[key])
        arrays[name+"_mask"] = mask
        # Compare the ablation costs within each unit before pooling. Their
        # difference is (Full - History) - (Full - Retinal) = Retinal - History;
        # subtracting two population medians would estimate a different quantity.
        for first, second in (("full", "history"), ("global", "history"),
                              ("history", "full"), ("retinal", "full"),
                              ("history", "retinal")):
            stat = _session_cluster_bootstrap(values[first], values[second], sessions, mask)
            reports[name]["contrasts"][second+"_minus_"+first] = stat
        reports[name]["contrasts"].update(
            adjacent_scope_contrasts(
                values, sessions, mask, _session_cluster_bootstrap
            )
        )
    for suffix in ("pdf", "png"):
        fig.savefig(args.out_dir / f"stabilization_control.{suffix}", dpi=180)
    plt.close(fig)
    main_stats = json.loads((SOURCE_ROOT / selection["bundle"] / "figure3/figures/figure3_manifest.json").read_text())
    for name, panel in (("ccnorm", "panel_c_stats"), ("fraction", "panel_d_stats")):
        original = main_stats[panel]
        if reports[name]["n_units"] != original["n_units"]:
            raise AssertionError(f"Supplement {name} population differs from the main figure")
        for key, source_key in (("full", "intact"), ("retinal", "zeroed"), ("global", "stabilized")):
            expected = (original["medians"][source_key] if name == "ccnorm"
                        else original["conditions"][source_key]["median"])
            if reports[name]["medians"][key] != expected:
                raise AssertionError(f"Supplement {name}/{key} does not reproduce the main-figure median")
    np.savez_compressed(archive / "paired_scores.npz", **arrays)
    report = {"checkpoint_sha256": selection["checkpoint_sha256"],
              "global_cache": provenance_path(global_path), "global_cache_sha256": digest(global_path),
              "local_cache": provenance_path(history_path), "local_cache_sha256": digest(history_path),
              "trial_cache": provenance_path(trial_path), "trial_cache_sha256": digest(trial_path),
              "accepted_trial_replay_sha256": accepted_trial_sha256,
              "data_identity_checks": data_checks, "unchanged_input_replay": replay_checks,
              "history_render_audits": history_manifest["sessions"],
              "trial_render_audits": trial_manifest["sessions"],
              "trial_boundary_audit": trial_manifest["trial_boundary_audit"],
              "trial_input_provenance": trial_manifest["input_provenance"],
              "metrics": reports,
              "ablation_cost_comparison": {
                  "recent_retinal_motion": "full_minus_history",
                  "explicit_extraretinal_input": "full_minus_retinal",
                  "paired_cost_difference": "retinal_minus_history",
                  "positive_difference": "larger cost of removing recent retinal motion"},
              "uncertainty": "10000 paired session bootstrap samples; fitted checkpoint held fixed",
              "paired_scores_sha256": digest(archive / "paired_scores.npz")}
    (archive / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    binding = {"summary": "manuscript/analysis/stabilization_control/summary.json",
               "summary_sha256": digest(archive / "summary.json")}
    (HERE / "analysis/stabilization_control.json").write_text(json.dumps(binding, indent=2) + "\n")
    (HERE / "stabilization_stats.tex").write_text(statistics_tex(reports))
    import shutil
    shutil.copy2(args.out_dir / "stabilization_control.pdf", HERE / "figures/stabilization_control.pdf")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
