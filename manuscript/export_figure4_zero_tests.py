#!/usr/bin/env python3
"""Test the displayed Figure 4 population effects against zero improvement.

Use each curve's existing paired bootstrap scheme, retaining its estimand and
all source data. This quantifies uncertainty of population summaries, separately
from the across-unit distributions shown by the boxplots. No model inference.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "build/mpl"))


def bootstrap_zero_p(draws):
    """Two-sided percentile-bootstrap sign tails, with finite-sample correction."""
    import numpy as np
    values = np.asarray(draws, float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite bootstrap draws")
    tail = np.minimum((values <= 0).sum(axis=0), (values >= 0).sum(axis=0))
    return np.minimum(1., 2. * (tail + 1) / (len(values) + 1))


def holm(p):
    import numpy as np
    p = np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def main():
    import numpy as np
    import pandas as pd
    from paper.fig4.spatiotemporal_tuning.population_response import load_matrix, _population_arrays, quantile_bins, crossed_population_bootstrap
    from paper.fig4.spatiotemporal_tuning._spectral_shards import load_and_merge_shards
    from paper.fig4.spatiotemporal_tuning._figure4_renderer import _direct_mechanism_values, _clustered_curve

    selection = json.loads((HERE / "analysis/selected_model_bundle.json").read_text())
    bundle = ROOT / selection["bundle"]
    original = json.loads((bundle / "figure4/production_figure4/figure/summary.json").read_text())
    out = HERE / "analysis/figure4_zero_tests"
    out.mkdir(exist_ok=True)
    source_paths = [HERE / "analysis/selected_model_bundle.json", bundle / "figure4/production_figure4/figure/summary.json"]
    records, archives = [], {}

    def add(panel, outcome, x, centers, draws, expected, scheme, seed):
        np.testing.assert_allclose(centers, expected, rtol=1e-12, atol=1e-12)
        key = panel + "_" + outcome
        archives[key] = np.asarray(draws)
        ps = bootstrap_zero_p(draws)
        ci = np.quantile(draws, [.025, .975], axis=0).T
        for i in range(len(centers)):
            records.append({"panel": panel, "outcome": outcome, "index": i, "x": float(x[i]),
                            "estimate": float(centers[i]), "p_raw": float(ps[i]), "ci95": ci[i].tolist(),
                            "n_bootstrap": len(draws), "seed": seed, "resampling": scheme,
                            "draws_key": key})

    # B: reproduce the original crossed image × fixation × unit bootstrap.
    matrix = bundle / "figure4/response_matrix_40img_x_200fix/merged"
    source_paths.extend(matrix/name for name in ("summary.json", "mean_rate_matrix.npy", "ssi_matrix.npy", "expected_spikes_matrix.npy",
                        "stabilized_mean_rate_by_image.npy", "stabilized_ssi_by_image.npy", "stabilized_expected_spikes_by_image.npy", "trace_feature_table.csv"))
    arrays, traces, summary = load_matrix(matrix)
    reduced = _population_arrays(arrays, np.arange(summary["n_units"]))
    path = traces.rendered_path_length_arcmin.to_numpy()
    bins = quantile_bins(path, 8)
    b_draws, b_centers, xs = [], [], []
    for i in range(8):
        trace_indices = np.flatnonzero(bins == i)
        centers, _, _, draws = crossed_population_bootstrap(reduced, trace_indices,
            n_bootstrap=3000, rng=np.random.default_rng(20260823+1000*i), return_draws=True)
        b_draws.append(draws); b_centers.append(centers); xs.append(np.median(path[trace_indices]))
        print(f"B bin {i+1}/8 complete", flush=True)
    for column, outcome in enumerate(("rate", "SSI")):
        add("B", outcome, xs, np.asarray(b_centers)[:, column], np.stack(b_draws, axis=1)[:, :, column],
            original["panels"]["B"]["curves"][outcome]["effect_percent"],
            "crossed images, fixation histories, and units; measured/stabilized pairing retained", "20260823 + 1000*bin")
    del arrays, reduced

    # F (source panel G): preserve the published units/within-unit histories scheme.
    shards = [bundle / f"figure4/matrix_spectral_replay/shard_{i:02d}/causal_chain_shard.npz" for i in range(2)]
    source_paths.extend(shards)
    data = load_and_merge_shards(shards)
    values = _direct_mechanism_values(data)
    for i, (key, outcome) in enumerate((("rate_percent", "rate"), ("ssi_percent", "SSI"))):
        xs, centers, _, _, draws = _clustered_curve(values["passband_percentile"], values[key],
            n_bootstrap=5000, seed=20260825+i, return_draws=True)
        add("F", outcome, xs, centers, draws, original["panels"]["G"][key]["binned_center"],
            "units, then fixation histories within each unit; fixed image ensemble", 20260825+i)
        print(f"F {outcome} complete", flush=True)

    # G (source panel H): the same unit bootstrap used for cumulative readouts.
    trajectory_path = bundle / "figure4/top_passband_stage_trajectory_10img_x_10fix/top_passband_stage_trajectory.npz"
    source_paths.append(trajectory_path)
    trajectory = np.load(trajectory_path)
    for i, (key, outcome, report_key) in enumerate((("unit_temporal_modulation_points", "temporal", "temporal_modulation"),
                                                  ("unit_ssi_delta_bits_per_spike", "SSI", "spatial_sharpening"))):
        value = np.asarray(trajectory[key], float)
        seed = 20260845+1000*i
        units = np.random.default_rng(seed).integers(0, value.shape[1], size=(5000, value.shape[1]))
        draws = np.median(value[:, units], axis=2).T
        add("G", outcome, np.arange(3), np.median(value, axis=1), draws,
            original["panels"]["H"][report_key]["center"], "units; fixed high-engagement movie ensemble", seed)
    adjusted = holm([row["p_raw"] for row in records])
    for row, p in zip(records, adjusted):
        row["p_holm"] = float(p)
        row["significant_positive"] = bool(row["estimate"] > 0 and p < .05)
    np.savez_compressed(out / "bootstrap_draws.npz", **archives)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report = {"checkpoint_sha256": selection["checkpoint_sha256"],
              "null": "zero measured-minus-stabilized population effect in each displayed bin or stage",
              "method": "two-sided percentile-bootstrap sign-tail probabilities; 2*(min(n<=0,n>=0)+1)/(B+1), capped at 1",
              "multiplicity": "Holm familywise correction across all 32 displayed effects in B, F, and G",
              "scope": "conditional on the fitted twin, with the same resampling units as the plotted confidence intervals; not training or between-animal uncertainty",
              "records": records, "draws_sha256": digest(out / "bootstrap_draws.npz"),
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_paths}}
    (out / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(pd.DataFrame(records)[["panel", "outcome", "index", "p_raw", "p_holm", "significant_positive"]].to_string(index=False))


if __name__ == "__main__":
    main()
