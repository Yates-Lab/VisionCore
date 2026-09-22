#!/usr/bin/env python3
"""Replay a small shortlist from the existing Figure 4 example audit."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "manuscript/build/mpl"))


def install_reviewed(source, review, summary, selection, shortlist):
    """Pin the highest-SSI-gain candidate, after reviewing the shortlist."""
    import numpy as np
    from paper.fig4.spatiotemporal_tuning.activation_map_metrics import map_statistics
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    winner = shortlist.iloc[0]
    path = review / "example_0.npz"
    with np.load(path) as data:
        for key in ("image_index", "trace_index", "unit_index", "endpoint_frame"):
            if int(data[key]) != int(winner[key]):
                raise ValueError("Reviewed candidate does not match the deterministic SSI ranking")
        stats = map_statistics(np.stack([data["stable_rate_map"], data["motion_rate_map"]]), float(data["output_rate_hz"]))
        np.testing.assert_allclose(stats["rate_spikes_s"], [winner.stable_rate_spikes_s, winner.motion_rate_spikes_s], rtol=2e-5)
        np.testing.assert_allclose(stats["ssi_bits_per_spike"], [winner.stable_ssi_bits_per_spike, winner.motion_ssi_bits_per_spike], rtol=2e-5)
        anchor = summary["trace_window_selection"]["anchor_chronological_frame_index"]
        np.testing.assert_allclose(data["stable_history"][anchor], data["motion_history"][anchor], atol=1e-5, rtol=0)
    archive = ROOT / "manuscript/analysis/figure4_example"
    archive.mkdir(exist_ok=True)
    shutil.copy2(path, archive / "selected_example.npz")
    shutil.copy2(review / "shortlist.csv", archive / "shortlist.csv")
    selected = winner.drop(labels=["selected"], errors="ignore").to_dict()
    for i, name in enumerate(("stable", "motion")):
        selected[name+"_rate_spikes_s"] = float(stats["rate_spikes_s"][i])
        selected[name+"_ssi_bits_per_spike"] = float(stats["ssi_bits_per_spike"][i])
    selected["rate_change_spikes_s"] = float(np.diff(stats["rate_spikes_s"])[0])
    selected["rate_change_percent"] = float(100*(stats["rate_spikes_s"][1]/stats["rate_spikes_s"][0]-1))
    selected["ssi_change_bits_per_spike"] = float(np.diff(stats["ssi_bits_per_spike"])[0])
    selected["ssi_change_percent"] = float(100*(stats["ssi_bits_per_spike"][1]/stats["ssi_bits_per_spike"][0]-1))
    summary["selected"] = selected
    summary["response_selection"]["ranking"] = "largest absolute SSI gain within the original top-8-unit and bounded response gates"
    summary["response_selection"]["selection_tier"] += "; ranked by absolute SSI gain"
    summary["response_selection"]["all_candidates_file"] = str((source / "candidate_metrics.csv").relative_to(ROOT))
    summary["response_selection"]["population_unit_effects_file"] = str((source / "population_unit_effects.csv").relative_to(ROOT))
    summary["example_revision"] = {
        "n_pairs_replayed_for_visual_review": len(shortlist), "selected_review_index": 0,
        "map_replay_relative_tolerance": 2e-5,
        "source_sha256": {str((source/name).relative_to(ROOT)): digest(source/name)
                          for name in ("candidate_metrics.csv", "summary.json", "image_candidates.csv", "trace_candidates.csv")}}
    (archive / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    binding = {"model_bundle": selection["bundle"], "checkpoint_sha256": selection["checkpoint_sha256"],
               "audit_dir": str(archive.relative_to(ROOT)),
               "source_sha256": {str((archive/name).relative_to(ROOT)): digest(archive/name)
                                  for name in ("selected_example.npz", "summary.json", "shortlist.csv")}}
    (ROOT / "manuscript/analysis/figure4_example_selection.json").write_text(json.dumps(binding, indent=2)+"\n")
    print(f"Installed image {int(winner.image_index)}, trace {int(winner.trace_index)}, unit {int(winner.unit_index)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "manuscript/build/figure4_example_review")
    parser.add_argument("--install-reviewed", action="store_true", help="Install the already-replayed highest-SSI-gain example without inference")
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from paper.fig4.spatiotemporal_tuning.activation_map_metrics import score_histories, map_statistics
    from paper.fig4.spatiotemporal_tuning.retinal_replay import render_movies
    from paper.fig4.upstream.real_trace_matrix.core import extract_patch
    from paper.fig4.upstream.real_trace_matrix.model import RealTraceMatrixScorer

    selection = json.loads((ROOT / "manuscript/analysis/selected_model_bundle.json").read_text())
    bundle = ROOT / selection["bundle"]
    source = bundle / "figure4/panel_a_exemplar_audit"
    summary = json.loads((source / "summary.json").read_text())
    checkpoint = Path(summary["checkpoint"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection["checkpoint_sha256"]:
        raise ValueError("Example replay checkpoint differs from the manuscript")
    metrics = pd.read_csv(source / "candidate_metrics.csv")
    eligible = metrics.loc[(metrics.stable_rate_spikes_s >= .5) & (metrics.stable_ssi_bits_per_spike >= .05)]
    units = eligible.groupby("unit_index").population_joint_rank_score.max().sort_values(ascending=False).head(8).index
    eligible = eligible.loc[
        eligible.unit_index.isin(units) & eligible.rate_change_percent.between(25, 200)
        & eligible.ssi_change_percent.between(10, 200) & eligible.ssi_change_bits_per_spike.between(.02, .25)
        & (eligible.motion_rate_spikes_s <= 50) & (eligible.normalized_map_rms_change >= .12)]
    # Keep all existing gates and unit preselection. Inspect distinct image/unit
    # pairs ranked by absolute SSI gain, rather than overall response-map change.
    shortlist = eligible.sort_values(
        ["ssi_change_bits_per_spike", "ssi_change_percent", "image_index", "trace_index", "unit_index"],
        ascending=[False, False, True, True, True]).drop_duplicates(["image_index", "unit_index"]).head(8)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shortlist.to_csv(args.out_dir / "shortlist.csv", index=False)
    if args.install_reviewed:
        install_reviewed(source, args.out_dir, summary, selection, shortlist)
        return
    images = pd.read_csv(source / "image_candidates.csv")
    traces = np.load(summary["trace_source"])
    anchor = summary["trace_window_selection"]["anchor_chronological_frame_index"]
    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=checkpoint,
        dataset_configs=ROOT / "paper/model_selection/configs/multi_240_long_split3_dekel35_allgratings.yaml",
        population_spec_dir=bundle / "figure4/all_available_population_spec",
        population_version=summary["population_version"], device=args.device, strict=True,
        mcfarland_outputs=ROOT / "scripts/mcfarland_outputs_mono.pkl")
    fig, axes = plt.subplots(len(shortlist), 4, figsize=(10, 2.5 * len(shortlist)))
    cache = {}
    replay = []
    for index, (_, row) in enumerate(shortlist.iterrows()):
        image_index, trace_index, unit = (int(row[key]) for key in ("image_index", "trace_index", "unit_index"))
        endpoint = int(row.endpoint_frame)
        window = traces[trace_index, endpoint-59:endpoint+1].copy()
        window -= window[anchor]
        patch, _ = extract_patch(images.loc[images.image_index.eq(image_index)].iloc[0],
                                 canvas_cache=cache, patch_size_px=540)
        histories = render_movies(patch, np.stack([np.zeros_like(window), window]), device=args.device)
        maps = score_histories(scorer, histories, batch_size=2)[:, unit]
        stats = map_statistics(maps, scorer.output_rate_hz)
        np.testing.assert_allclose(stats["rate_spikes_s"], [row.stable_rate_spikes_s, row.motion_rate_spikes_s], rtol=2e-5)
        np.testing.assert_allclose(stats["ssi_bits_per_spike"], [row.stable_ssi_bits_per_spike, row.motion_ssi_bits_per_spike], rtol=2e-5)
        np.testing.assert_allclose(histories[0, anchor], histories[1, anchor], atol=1e-5, rtol=0)
        np.savez_compressed(args.out_dir / f"example_{index}.npz",
            image_index=image_index, trace_index=trace_index, unit_index=unit, endpoint_frame=endpoint,
            output_rate_hz=scorer.output_rate_hz, trace_window_xy_filtered_endpoint_aligned=window,
            stable_history=histories[0], motion_history=histories[1], stable_rate_map=maps[0], motion_rate_map=maps[1])
        stable, moving = stats["normalized_map"]
        common_max = max(stable.max(), moving.max())
        axes[index, 0].imshow(histories[0, anchor], cmap="gray", vmin=0, vmax=255)
        axes[index, 0].set_title(f"#{index}: image {image_index}, trace {trace_index}, unit {unit}", fontsize=9)
        for col, gain, name, si, rate in ((1, stable, "Stabilized", *[stats[k][0] for k in ("ssi_bits_per_spike", "rate_spikes_s")]),
                                         (2, moving, "Measured", *[stats[k][1] for k in ("ssi_bits_per_spike", "rate_spikes_s")])):
            axes[index, col].imshow(gain, cmap="viridis", vmin=0, vmax=common_max, interpolation="nearest")
            axes[index, col].set_title(f"{name}: {si:.3f} bits/spike, {rate:.1f} spikes/s", fontsize=9)
        delta = moving-stable
        axes[index, 3].imshow(delta, cmap="RdBu_r", vmin=-abs(delta).max(), vmax=abs(delta).max(), interpolation="nearest")
        axes[index, 3].set_title(f"Change: SSI +{row.ssi_change_percent:.0f}%", fontsize=9)
        for ax in axes[index]:
            ax.axis("off")
        replay.append({"review_index": index, "image_index": image_index, "trace_index": trace_index, "unit_index": unit,
                       "rates": stats["rate_spikes_s"].tolist(), "ssi": stats["ssi_bits_per_spike"].tolist()})
        print(f"Replayed example {index}: SSI {stats['ssi_bits_per_spike']}", flush=True)
    fig.tight_layout()
    fig.savefig(args.out_dir / "contact_sheet.png", dpi=130)
    plt.close(fig)
    (args.out_dir / "replay_checks.json").write_text(json.dumps(replay, indent=2)+"\n")


if __name__ == "__main__":
    main()
