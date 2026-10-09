#!/usr/bin/env python3
"""Render the stabilization-control supplement for the selected manuscript model.

The analysis and plotting live in paper/supp_stabilization_control. This entry
point binds them to the manuscript's selected bundle, retained replay caches,
and installed statistics and figure files.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from analysis_selection import SOURCE_ROOT, source_path, selected_analysis

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from paper.supp_stabilization_control.analysis import (  # noqa: E402,F401
    adjacent_scope_contrasts, assemble_control_rows, digest, paired_metrics,
    replay_check_record, statistics_tex, summarize, unchanged_input_replay_checks,
    validate_replay_provenance, validate_trial_replay_acceptance,
)


def provenance_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(SOURCE_ROOT))
    except ValueError:
        return str(path)


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
    selection = selected_analysis()
    bundle = SOURCE_ROOT / selection["bundle"]
    final = json.loads((bundle / "FINAL_MANIFEST.json").read_text())
    manifest = json.loads((bundle / "figure3/run_manifest.json").read_text())
    if (final.get("model_spec_sha256") != selection.get("model_spec_sha256")
            or digest(source_path(final["model_spec"])) != selection["model_spec_sha256"]
            or manifest.get("model_spec_sha256") != selection["model_spec_sha256"]
            or manifest.get("checkpoint_sha256") != selection["checkpoint_sha256"]):
        raise ValueError("Selected Figure 3 model identity changed")
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import _fig3_ablation_data as ab
    from generate_figure3 import _session_cluster_bootstrap
    from paper.supp_stabilization_control import figure

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
        validate_replay_provenance(control_path, run_manifest, reference, selection)
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
    metrics = paired_metrics(global_data, history_data, trial_data)
    replay_checks = unchanged_input_replay_checks(
        global_data, {"history": history_data, "trial": trial_data}, pop, accepted_trial_sha256
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive = HERE / "analysis/stabilization_control"
    archive.mkdir(parents=True, exist_ok=True)
    reports, scores = summarize(metrics, sessions, pop, _session_cluster_bootstrap)
    arrays = {"sessions": sessions, "population": pop,
              "neuron_ids": np.concatenate([r["neuron_mask"] for r in global_rows]), **scores}
    fig = figure.plot(arrays, np.load(archive / "panel_a_example.npz"))
    figure.save(fig, args.out_dir)
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
