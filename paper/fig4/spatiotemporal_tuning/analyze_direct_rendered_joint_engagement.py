#!/usr/bin/env python3
"""Measure M77 SF×TF engagement from the retinal movies actually rendered.

This is the primary, renderer-faithful companion to the ideal Fourier-shift
calculation.  Each selected image is shifted through the exact Figure 4 crop,
padding, and bilinear-interpolation path for every selected eye trajectory.
The resulting exact 151×151 scorer crop receives a spatial FFT at each time and a
two-DPSS temporal spectrum after removal of the finite-window temporal mean.
Only then is dynamic power accumulated on the measured SF×TF×orientation grid
and compared with the model's raw periodic-grating response tensor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.analyze_image_specific_joint_engagement import (
    actual_causal_effects,
    bootstrap_median,
    bootstrap_spearman,
    circular_orientation_weights,
    interpolate_tuning_to_fft_bins,
    log_interpolation_weights,
    selectivity_bits,
    unit_correlations,
    validate_matrix_contract,
    validate_native240_trace_contract,
)
from paper.fig4.spatiotemporal_tuning.compute_native_rucci_overlap import (
    EPS,
    frequency_grid,
    load_matrix_trace_bank,
    observed_tuning_tensor,
)
from paper.fig4.spatiotemporal_tuning.validate_trajectory_phase_spectrum import (
    folded_multitaper_power,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import (
    OUT_SIZE,
    PPD,
    _eye_deg_to_norm,
    _shift_movie_with_eye,
    _standardize_uint_like,
)


BASE = ROOT / "outputs/dekel240_paper/m77_epoch279"
DEFAULT_MATRIX = BASE / "figure4_real_trace_pilot_corrected/merged"
DEFAULT_GROUPED = BASE / "periodic_tuning_respaced/frequency_tuning_grouped.csv"
DEFAULT_AUDIT = BASE / "periodic_tuning_respaced/fit_audit/m77_tuning_fit_audit.csv"
DEFAULT_OUT = (
    BASE
    / "figure4_real_trace_pilot_corrected/direct_rendered_joint_engagement"
)
METHOD_VERSION = "direct_rendered_movie_dpss_nw1p5_k2_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--grouped-csv", type=Path, default=DEFAULT_GROUPED)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--n-traces",
        type=int,
        default=16,
        help="Evenly sample this many retained trajectories; 0 uses all.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--expected-checkpoint-sha256", default=None)
    parser.add_argument("--expected-dataset-configs-sha256", default=None)
    parser.add_argument("--require-native-240-contract", action="store_true")
    return parser.parse_args()


def render_movies(
    patch: np.ndarray,
    traces: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    """Render [trace,time,height,width] through the exact scorer geometry."""
    import torch

    trace = np.asarray(traces, dtype=np.float32)
    if trace.ndim != 3 or trace.shape[-1] != 2:
        raise ValueError(f"traces must be [trace,time,2], got {trace.shape}")
    n_trace, n_time = trace.shape[:2]
    image = _standardize_uint_like(patch)
    eye = torch.from_numpy(trace.reshape(-1, 2)).to(device)
    eye_norm = _eye_deg_to_norm(eye, ppd=PPD, img_size=image.shape, torch=torch)
    base = torch.from_numpy(image).to(device=device, dtype=torch.float32)
    repeated = base.unsqueeze(0).expand(n_trace * n_time, -1, -1)
    with torch.no_grad():
        shifted = _shift_movie_with_eye(
            repeated,
            eye_norm,
            out_size=OUT_SIZE,
            scale_factor=1.0,
            torch=torch,
        )
    return shifted.reshape(n_trace, n_time, *OUT_SIZE).cpu().numpy()


def mode_to_grid_matrix(
    kxy: np.ndarray,
    spatial_cpd: np.ndarray,
    orientations_deg: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Sparse linear interpolation from Fourier modes to SF×orientation bins."""
    radial = np.linalg.norm(kxy, axis=1)
    sf0, sf1, sw0, sw1, resolved = log_interpolation_weights(radial, spatial_cpd)
    ori0, ori1, ow0, ow1 = circular_orientation_weights(kxy, orientations_deg)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    mode = np.arange(len(kxy), dtype=int)
    for sf_index, sf_weight in ((sf0, sw0), (sf1, sw1)):
        for ori_index, ori_weight in ((ori0, ow0), (ori1, ow1)):
            weight = sf_weight * ori_weight * resolved
            keep = weight > 0
            rows.append(mode[keep])
            columns.append(
                sf_index[keep] * len(orientations_deg) + ori_index[keep]
            )
            values.append(weight[keep])
    matrix = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(len(kxy), len(spatial_cpd) * len(orientations_deg)),
    ).tocsr()
    return matrix, resolved


def movie_power_cube(
    movie: np.ndarray,
    *,
    flat_index: np.ndarray,
    mode_to_grid: sparse.csr_matrix,
    n_spatial: int,
    n_orientation: int,
    frame_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    coefficient = np.fft.fft2(movie, axes=(-2, -1)) / float(
        movie.shape[-2] * movie.shape[-1]
    )
    selected = coefficient.reshape(len(movie), -1)[:, flat_index].T
    temporal_hz, mode_power = folded_multitaper_power(selected, frame_rate_hz)
    flat_cube = mode_to_grid.T @ mode_power
    cube = np.asarray(flat_cube).reshape(
        n_spatial, n_orientation, len(temporal_hz)
    ).transpose(0, 2, 1)
    return temporal_hz, cube


def engagement_from_cubes(
    cubes: np.ndarray,
    normalized_tuning: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute joint and marginal controls from identical dynamic-power cubes."""
    power = np.asarray(cubes, dtype=np.float64)
    tuning = np.asarray(normalized_tuning, dtype=np.float64)
    joint = np.einsum("isto,usto->iu", power, tuning, optimize=True)
    tuning_sfo = tuning.sum(axis=2)
    tuning_tf = tuning.sum(axis=(1, 3))
    separable_tuning = np.einsum(
        "uso,ut->usto", tuning_sfo, tuning_tf, optimize=True
    )
    separable = np.einsum(
        "isto,usto->iu", power, separable_tuning, optimize=True
    )
    tf_marginal = np.einsum(
        "it,ut->iu", power.sum(axis=(1, 3)), tuning_tf, optimize=True
    )
    sf_marginal = np.einsum(
        "iso,uso->iu", power.sum(axis=2), tuning_sfo, optimize=True
    )
    total = power.sum(axis=(1, 2, 3))

    def fraction(value: np.ndarray) -> np.ndarray:
        return np.divide(
            value,
            total[:, None],
            out=np.zeros_like(value),
            where=total[:, None] > 0,
        )

    return {
        "joint": joint,
        "separable": separable,
        "tf_marginal": tf_marginal,
        "sf_marginal": sf_marginal,
        "joint_fraction": fraction(joint),
        "separable_fraction": fraction(separable),
        "tf_marginal_fraction": fraction(tf_marginal),
        "sf_marginal_fraction": fraction(sf_marginal),
        "total_dynamic_power": np.broadcast_to(total[:, None], joint.shape).copy(),
    }


def render_population_summary(
    path: Path,
    correlations: dict[str, np.ndarray],
    predictor_summary: dict[str, tuple[float, tuple[float, float]]],
    coupling: np.ndarray,
    ssi_percent: np.ndarray,
    trusted: np.ndarray,
    coupling_summary: tuple[float, float, tuple[float, float]],
) -> None:
    order = (
        "joint_fraction",
        "separable_fraction",
        "tf_marginal_fraction",
        "sf_marginal_fraction",
        "total_dynamic_power",
    )
    labels = ("joint", "separable", "TF", "SF×ori", "total power")
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), constrained_layout=True)
    rng = np.random.default_rng(17)
    for index, name in enumerate(order):
        values = correlations[name]
        finite = values[np.isfinite(values)]
        axes[0].scatter(
            index + rng.uniform(-0.14, 0.14, len(finite)),
            finite,
            s=13,
            color="0.68",
            alpha=0.30,
            edgecolor="none",
        )
        median, interval = predictor_summary[name]
        axes[0].errorbar(
            index,
            median,
            yerr=[[median - interval[0]], [interval[1] - median]],
            fmt="D",
            color="#2474A8" if index == 0 else "#666666",
            ms=6,
            capsize=3,
        )
    axes[0].axhline(0, color="0.5", lw=0.8)
    axes[0].set(
        title="A  Same-image prediction of moving − stabilized rate",
        ylabel="within-unit Spearman ρ",
        xticks=np.arange(len(order)),
        xticklabels=labels,
    )
    axes[0].grid(axis="y", alpha=0.17)

    axes[1].scatter(
        coupling[~trusted], ssi_percent[~trusted], s=24, color="0.72", alpha=0.45
    )
    axes[1].scatter(
        coupling[trusted],
        ssi_percent[trusted],
        s=30,
        color="#2E7D49",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.4,
    )
    axes[1].axhline(0, color="0.5", lw=0.8)
    axes[1].set(
        title=(
            "B  Does joint SF–TF coupling explain the SSI increase?\n"
            f"Spearman ρ={coupling_summary[0]:.2f}, p={coupling_summary[1]:.2g}"
        ),
        xlabel="joint − separable alignment selectivity (bits)",
        ylabel="moving versus stabilized SSI (%)",
    )
    axes[1].grid(alpha=0.17)
    figure.suptitle(
        "M77: retinal power measured from the actually rendered movies",
        fontsize=13,
        fontweight="semibold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir = args.matrix_dir.resolve()
    matrix_audit = validate_matrix_contract(matrix_dir)
    provenance = matrix_audit.get("validated_common_provenance") or {}
    expected_provenance = {
        "model_provenance.model.checkpoint_sha256": args.expected_checkpoint_sha256,
        "model_provenance.model.dataset_configs_sha256": args.expected_dataset_configs_sha256,
    }
    for key, expected in expected_provenance.items():
        if expected is not None and provenance.get(key) != expected:
            raise RuntimeError(
                f"Merged matrix provenance mismatch for {key}: "
                f"{provenance.get(key)!r} versus expected {expected!r}"
            )
    images = pd.read_csv(matrix_dir / "image_feature_table.csv").sort_values(
        "image_index"
    ).reset_index(drop=True)
    traces, dt, trace_rows, trace_contract = load_matrix_trace_bank(
        matrix_dir, int(args.n_traces)
    )
    if args.require_native_240_contract:
        validate_native240_trace_contract(trace_contract)
    frame_rate_hz = 1.0 / dt
    units, spatial, probe_tf, orientation, raw_tuning, _ = observed_tuning_tensor(
        pd.read_csv(args.grouped_csv)
    )
    grid = frequency_grid()
    kxy = np.asarray(grid["kxy"], dtype=np.float64)
    flat_index = np.asarray(grid["flat_index"], dtype=int)
    distributor, resolved_modes = mode_to_grid_matrix(kxy, spatial, orientation)

    cubes: list[np.ndarray] = []
    temporal_hz: np.ndarray | None = None
    canvas_cache: dict = {}
    for image_row, image in images.iterrows():
        patch, _ = extract_patch(
            image, canvas_cache=canvas_cache, patch_size_px=540
        )
        movies = render_movies(patch, traces, device=args.device)
        trace_cubes: list[np.ndarray] = []
        for movie in movies:
            frequency, cube = movie_power_cube(
                movie,
                flat_index=flat_index,
                mode_to_grid=distributor,
                n_spatial=len(spatial),
                n_orientation=len(orientation),
                frame_rate_hz=frame_rate_hz,
            )
            if temporal_hz is None:
                temporal_hz = frequency
            elif not np.array_equal(temporal_hz, frequency):
                raise RuntimeError("Rendered movies returned inconsistent TF grids")
            trace_cubes.append(cube)
        cubes.append(np.mean(trace_cubes, axis=0))
        print(f"direct rendered spectrum image {image_row + 1}/{len(images)}", flush=True)
    image_cubes = np.asarray(cubes, dtype=np.float64)
    temporal_hz = np.asarray(temporal_hz, dtype=np.float64)

    normalized_tuning = interpolate_tuning_to_fft_bins(
        raw_tuning, probe_tf, temporal_hz
    )
    engagement = engagement_from_cubes(image_cubes, normalized_tuning)
    causal = actual_causal_effects(matrix_dir, units)
    correlations = {
        name: unit_correlations(value, causal["rate_delta"])
        for name, value in engagement.items()
    }
    predictor_summary = {
        name: bootstrap_median(
            value, n_bootstrap=args.n_bootstrap, seed=args.seed + index
        )
        for index, (name, value) in enumerate(correlations.items())
    }
    joint_ssi = selectivity_bits(engagement["joint_fraction"])
    separable_ssi = selectivity_bits(engagement["separable_fraction"])
    coupling = joint_ssi - separable_ssi
    coupling_summary = bootstrap_spearman(
        coupling,
        causal["ssi_percent"],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 30,
    )
    audit = pd.read_csv(args.audit_csv).set_index("unit_index")
    trusted = np.asarray(
        [
            unit in audit.index and str(audit.loc[unit, "audit_category"]) == "trusted"
            for unit in units
        ],
        dtype=bool,
    )

    table = pd.DataFrame(
        {
            "unit_index": units,
            # Stable column aliases consumed by the production mechanism builder.
            "joint_engagement_ssi_bits": joint_ssi,
            "separable_engagement_ssi_bits": separable_ssi,
            "joint_minus_separable_engagement_ssi_bits": coupling,
            "joint_alignment_selectivity_bits": joint_ssi,
            "separable_alignment_selectivity_bits": separable_ssi,
            "joint_minus_separable_selectivity_bits": coupling,
            "ssi_percent_vs_stabilized": causal["ssi_percent"],
            "joint_alignment_fraction_vs_rate_delta_spearman": correlations[
                "joint_fraction"
            ],
            "separable_alignment_fraction_vs_rate_delta_spearman": correlations[
                "separable_fraction"
            ],
            "joint_alignment_vs_rate_delta_spearman": correlations[
                "joint_fraction"
            ],
            "separable_alignment_vs_rate_delta_spearman": correlations[
                "separable_fraction"
            ],
            "tf_alignment_vs_rate_delta_spearman": correlations[
                "tf_marginal_fraction"
            ],
            "sf_orientation_alignment_vs_rate_delta_spearman": correlations[
                "sf_marginal_fraction"
            ],
            "total_dynamic_power_vs_rate_delta_spearman": correlations[
                "total_dynamic_power"
            ],
            "joint_passband_power_vs_rate_delta_spearman": correlations["joint"],
            "separable_passband_power_vs_rate_delta_spearman": correlations[
                "separable"
            ],
            "tf_passband_power_vs_rate_delta_spearman": correlations[
                "tf_marginal"
            ],
            "sf_passband_power_vs_rate_delta_spearman": correlations[
                "sf_marginal"
            ],
            "joint_peak_audit_trusted": trusted,
        }
    )
    table.to_csv(args.out_dir / "unit_direct_rendered_joint_engagement.csv", index=False)
    np.savez_compressed(
        args.out_dir / "direct_rendered_joint_engagement.npz",
        spectrum_method_version=np.asarray(METHOD_VERSION),
        unit_indices=units,
        spatial_cpd=spatial,
        temporal_hz=temporal_hz,
        periodic_probe_temporal_hz=probe_tf,
        orientation_deg=orientation,
        image_rendered_movie_power=image_cubes.astype(np.float32),
        joint_engagement=engagement["joint"].astype(np.float32),
        separable_engagement=engagement["separable"].astype(np.float32),
        tf_marginal_engagement=engagement["tf_marginal"].astype(np.float32),
        sf_marginal_engagement=engagement["sf_marginal"].astype(np.float32),
        joint_alignment_fraction=engagement["joint_fraction"].astype(np.float32),
        separable_alignment_fraction=engagement["separable_fraction"].astype(np.float32),
        tf_marginal_alignment_fraction=engagement["tf_marginal_fraction"].astype(np.float32),
        sf_marginal_alignment_fraction=engagement["sf_marginal_fraction"].astype(np.float32),
        total_dynamic_power=engagement["total_dynamic_power"][:, 0].astype(np.float32),
        moving_rate=causal["moving_rate"].astype(np.float32),
        stabilized_rate=causal["stable_rate"].astype(np.float32),
        moving_ssi=causal["moving_ssi"].astype(np.float32),
        stabilized_ssi=causal["stable_ssi"].astype(np.float32),
        trace_rows=np.asarray(trace_rows, dtype=np.int64),
    )
    figure = args.out_dir / "m77_direct_rendered_population_summary.png"
    render_population_summary(
        figure,
        correlations,
        predictor_summary,
        coupling,
        causal["ssi_percent"],
        trusted,
        coupling_summary,
    )
    summary = {
        "analysis": "renderer-faithful image-specific SF×TF×orientation engagement",
        "spectrum_method_version": METHOD_VERSION,
        "definition": (
            "spatial FFT of every exactly rendered 151x151 retinal-movie frame, "
            "followed by mean removal and two-DPSS temporal power; no instantaneous "
            "velocity or ideal-translation assumption enters the primary estimator"
        ),
        "n_images": int(len(images)),
        "n_traces": int(len(traces)),
        "n_units": int(len(units)),
        "n_resolved_fourier_modes": int(np.count_nonzero(resolved_modes)),
        "trace_rows": np.asarray(trace_rows, dtype=int).tolist(),
        "trace_contract": trace_contract,
        "matrix_audit": matrix_audit,
        "predictor_within_unit_rate_delta": {
            key: {
                "median_spearman": float(value[0]),
                "bootstrap_ci95": [float(x) for x in value[1]],
            }
            for key, value in predictor_summary.items()
        },
        "coupling_excess_ssi_vs_ssi_percent": {
            "rho": coupling_summary[0],
            "p": coupling_summary[1],
            "bootstrap_ci95": list(coupling_summary[2]),
        },
        "claim_boundary": (
            "The primary retinal spectrum matches the scorer renderer exactly, but "
            "the selected eye traces originate at 120 Hz and are linearly interpolated "
            "onto the model's 240-Hz output grid. The periodic bank is direction-collapsed."
        ),
        "figure": str(figure.resolve()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
