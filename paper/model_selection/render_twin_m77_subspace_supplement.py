#!/usr/bin/env python3
"""Render the paired Twin-versus-M77 response-subspace supplement.

The two models have different native stimulus lattices (Twin: 120 Hz,
32x25x25 local support; M77: 240 Hz, 60x35x35 support).  We therefore compare
the same frozen biological units using coordinate-invariant quantities:
held-out response fidelity versus rank and cumulative Jacobian energy versus
rank.  Native physical basis filters are shown side by side, but no principal
angle is reported across incompatible feature grids.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.nonlinear_phase_causal.response_subspace import (
    dct_filters_to_movies as twin_dct_filters_to_movies,
)
from paper.model_selection._m77_response_subspace_impl import (
    dct_filters_to_movies as m77_dct_filters_to_movies,
)


DEFAULT_TWIN = ROOT / (
    "outputs/figures/fig4/nonlinear_phase_causal_v1/response_subspace_pilot"
)
DEFAULT_M77 = ROOT / (
    "outputs/dekel240_paper/m77_epoch279/response_subspace_matched"
)
DEFAULT_OUT = ROOT / (
    "outputs/dekel240_paper/m77_epoch279/twin_m77_subspace_supplement"
)

TWIN_COLOR = "#E67E22"
M77_COLOR = "#2C7FB8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twin-dir", type=Path, default=DEFAULT_TWIN)
    parser.add_argument("--m77-dir", type=Path, default=DEFAULT_M77)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--atlas-rank", type=int, default=8)
    parser.add_argument("--modes-per-unit", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selection(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    if "unit_index" not in table:
        raise ValueError(f"{path} lacks unit_index")
    if table.unit_index.duplicated().any():
        raise ValueError(f"{path} contains duplicate unit_index values")
    return table.reset_index(drop=True)


def validate_matching_units(twin: pd.DataFrame, m77: pd.DataFrame) -> np.ndarray:
    twin_units = twin.unit_index.to_numpy(dtype=int)
    m77_units = m77.unit_index.to_numpy(dtype=int)
    if not np.array_equal(twin_units, m77_units):
        raise ValueError(
            "Twin and M77 unit selections are not identical and in the same order: "
            f"Twin={twin_units.tolist()}, M77={m77_units.tolist()}"
        )
    return twin_units


def cumulative_energy(singular_values: np.ndarray) -> np.ndarray:
    values = np.asarray(singular_values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected [unit, mode] singular values, got {values.shape}")
    energy = np.square(values)
    total = energy.sum(axis=1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("Every unit must have positive Jacobian energy")
    return np.cumsum(energy, axis=1) / total


def rank_at_fraction(cumulative: np.ndarray, fraction: float = 0.8) -> np.ndarray:
    value = np.asarray(cumulative, dtype=np.float64)
    reached = value >= float(fraction)
    result = np.argmax(reached, axis=1) + 1
    result[~reached.any(axis=1)] = value.shape[1] + 1
    return result


def bootstrap_median(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    count: int,
) -> tuple[float, float, float]:
    value = np.asarray(values, dtype=np.float64)
    value = value[np.isfinite(value)]
    if not len(value):
        return math.nan, math.nan, math.nan
    rows = rng.integers(0, len(value), size=(int(count), len(value)))
    medians = np.median(value[rows], axis=1)
    return (
        float(np.median(value)),
        float(np.percentile(medians, 2.5)),
        float(np.percentile(medians, 97.5)),
    )


def paired_fidelity(
    twin_path: Path,
    m77_path: Path,
) -> tuple[pd.DataFrame, list[int]]:
    tables = []
    for model, path in (("Twin", twin_path), ("M77", m77_path)):
        table = pd.read_csv(path)
        required = {"unit_index", "rank", "test_rate_r2"}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{path} lacks {sorted(missing)}")
        selected = table[["unit_index", "rank", "test_rate_r2"]].copy()
        selected["model"] = model
        tables.append(selected)
    common_ranks = sorted(
        set(tables[0]["rank"].astype(int)).intersection(
            tables[1]["rank"].astype(int)
        )
    )
    combined = pd.concat(tables, ignore_index=True)
    combined = combined.loc[combined["rank"].isin(common_ranks)].copy()
    for rank in common_ranks:
        left = combined.loc[
            combined.model.eq("Twin") & combined["rank"].eq(rank), "unit_index"
        ].to_numpy(dtype=int)
        right = combined.loc[
            combined.model.eq("M77") & combined["rank"].eq(rank), "unit_index"
        ].to_numpy(dtype=int)
        if not np.array_equal(left, right):
            raise ValueError(f"Rank {rank} fidelity rows are not paired")
    return combined, common_ranks


def load_twin_energy(directory: Path) -> np.ndarray:
    candidates = sorted(directory.glob("rank*_active_gradient_spectrum.npz"))
    if not candidates:
        raise FileNotFoundError(f"No Twin active-gradient spectrum in {directory}")
    # All rank fits store the same model-gradient spectrum. Prefer the file
    # with the largest fitted rank and verify its cached cumulative values.
    def fitted_rank(path: Path) -> int:
        return int(path.name.split("_", 1)[0].removeprefix("rank"))

    path = max(candidates, key=fitted_rank)
    values = np.load(path)
    computed = cumulative_energy(values["singular_values"])
    if "cumulative_energy_fraction" in values:
        np.testing.assert_allclose(
            computed,
            values["cumulative_energy_fraction"],
            rtol=2e-5,
            atol=2e-6,
        )
    return computed


def load_m77_energy(directory: Path) -> np.ndarray:
    values = np.load(directory / "active_subspace.npz")
    return cumulative_energy(values["singular_values"])


def fidelity_summary(
    fidelity: pd.DataFrame,
    ranks: list[int],
    *,
    seed: int,
    bootstrap: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for model in ("Twin", "M77"):
        for rank in ranks:
            values = fidelity.loc[
                fidelity.model.eq(model) & fidelity["rank"].eq(rank),
                "test_rate_r2",
            ].to_numpy(dtype=float)
            median, low, high = bootstrap_median(
                values, rng=rng, count=bootstrap
            )
            rows.append(
                {
                    "model": model,
                    "rank": int(rank),
                    "median_test_rate_r2": median,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_units": int(np.isfinite(values).sum()),
                }
            )
    return pd.DataFrame(rows)


def render_summary(
    fidelity: pd.DataFrame,
    summary: pd.DataFrame,
    ranks: list[int],
    twin_energy: np.ndarray,
    m77_energy: np.ndarray,
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.35), constrained_layout=True)
    colors = {"Twin": TWIN_COLOR, "M77": M77_COLOR}
    for model in ("Twin", "M77"):
        part = summary.loc[summary.model.eq(model)].sort_values("rank")
        axes[0].plot(
            part["rank"].to_numpy(dtype=float),
            part.median_test_rate_r2.to_numpy(dtype=float),
            "o-",
            color=colors[model],
            lw=2,
            label=model,
        )
        axes[0].fill_between(
            part["rank"].to_numpy(dtype=float),
            part.ci95_low.to_numpy(dtype=float),
            part.ci95_high.to_numpy(dtype=float),
            color=colors[model],
            alpha=0.17,
            linewidth=0,
        )
    axes[0].set(
        title="A  Response fidelity",
        xlabel="private subspace rank",
        ylabel="held-out response $R^2$",
        xticks=ranks,
    )
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.18)

    shown = min(32, twin_energy.shape[1], m77_energy.shape[1])
    x = np.arange(1, shown + 1)
    for model, energy in (("Twin", twin_energy), ("M77", m77_energy)):
        for row in energy:
            axes[1].plot(x, row[:shown], color=colors[model], alpha=0.11, lw=0.7)
        axes[1].plot(
            x,
            np.median(energy[:, :shown], axis=0),
            color=colors[model],
            lw=2.2,
            label=model,
        )
    axes[1].axhline(0.8, color="0.45", ls="--", lw=1)
    axes[1].set(
        title="B  Jacobian energy concentration",
        xlabel="gradient subspace rank",
        ylabel="cumulative energy",
        ylim=(0, 1.02),
    )
    axes[1].grid(alpha=0.18)

    twin_rank = rank_at_fraction(twin_energy)
    m77_rank = rank_at_fraction(m77_energy)
    limit = max(int(twin_rank.max()), int(m77_rank.max())) + 1
    axes[2].plot([0, limit], [0, limit], color="0.65", lw=1, zorder=0)
    axes[2].scatter(
        twin_rank,
        m77_rank,
        color=M77_COLOR,
        edgecolor="white",
        linewidth=0.5,
        s=38,
    )
    axes[2].set(
        title="C  Rank needed for 80% energy",
        xlabel="Twin rank",
        ylabel="M77 rank",
        xlim=(0, limit),
        ylim=(0, limit),
    )
    axes[2].grid(alpha=0.18)
    figure.suptitle(
        "Twin and M77 have different response-active function spaces on the same units",
        fontsize=12,
        fontweight="semibold",
    )
    figure.savefig(output.with_suffix(".png"), dpi=220, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def load_native_filters(
    twin_dir: Path,
    m77_dir: Path,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    twin_state_path = twin_dir / f"rank{rank}_state.pt"
    if not twin_state_path.exists():
        raise FileNotFoundError(twin_state_path)
    twin_state = torch.load(twin_state_path, map_location="cpu", weights_only=False)
    twin_physical = twin_state.weights / twin_state.feature_std[None, None]
    twin_filters = twin_dct_filters_to_movies(twin_physical).cpu().numpy()

    m77_values = np.load(m77_dir / "active_subspace.npz")
    m77_directions = np.asarray(m77_values["directions"], dtype=np.float32)
    if rank > m77_directions.shape[1]:
        raise ValueError(
            f"M77 has {m77_directions.shape[1]} directions, requested rank {rank}"
        )
    scaling = np.load(m77_dir / "bank/feature_scaling.npz")
    m77_physical = torch.from_numpy(
        m77_directions[:, :rank] / np.asarray(scaling["std"])[None, None]
    )
    m77_filters = m77_dct_filters_to_movies(m77_physical).cpu().numpy()
    if twin_filters.shape[:2] != m77_filters.shape[:2]:
        raise ValueError(
            f"Native filter unit/rank shapes differ: {twin_filters.shape} vs "
            f"{m77_filters.shape}"
        )
    return twin_filters, m77_filters


def render_filter_atlas(
    units: np.ndarray,
    twin_filters: np.ndarray,
    m77_filters: np.ndarray,
    *,
    modes: int,
    output: Path,
) -> None:
    modes = min(int(modes), twin_filters.shape[1], m77_filters.shape[1])
    with PdfPages(output) as pdf:
        for unit_position, unit_index in enumerate(units):
            figure, axes = plt.subplots(
                2,
                modes + 1,
                figsize=(2.05 * (modes + 1), 4.5),
                constrained_layout=True,
                squeeze=False,
            )
            for row, (model, filters, rate, color) in enumerate(
                (
                    ("Twin", twin_filters, 120.0, TWIN_COLOR),
                    ("M77", m77_filters, 240.0, M77_COLOR),
                )
            ):
                profile_axis = axes[row, 0]
                for mode in range(modes):
                    movie = filters[unit_position, mode]
                    energy = np.sqrt(np.mean(np.square(movie), axis=(1, 2)))
                    profile_axis.plot(
                        np.arange(len(energy)) / rate * 1000.0,
                        energy / max(float(energy.max()), 1e-12),
                        lw=1.2,
                        label=f"mode {mode + 1}",
                    )
                    peak = int(np.argmax(energy))
                    spatial = movie[peak]
                    limit = max(float(np.percentile(np.abs(spatial), 99)), 1e-12)
                    extent = np.asarray(
                        [-spatial.shape[1], spatial.shape[1], -spatial.shape[0], spatial.shape[0]],
                        dtype=float,
                    ) / (2 * 37.50476617)
                    axes[row, mode + 1].imshow(
                        spatial,
                        cmap="RdBu_r",
                        vmin=-limit,
                        vmax=limit,
                        origin="lower",
                        extent=extent,
                        interpolation="nearest",
                    )
                    axes[row, mode + 1].set_title(
                        f"mode {mode + 1}\npeak {peak / rate * 1000:.0f} ms",
                        fontsize=8,
                    )
                    axes[row, mode + 1].set_xlabel("visual deg")
                    if mode == 0:
                        axes[row, mode + 1].set_ylabel("visual deg")
                profile_axis.set(
                    title=f"{model}: temporal RMS",
                    xlabel="native filter time (ms)",
                    ylabel="normalized energy",
                    ylim=(-0.03, 1.05),
                )
                profile_axis.spines["left"].set_color(color)
                profile_axis.legend(frameon=False, fontsize=6, ncol=2)
                profile_axis.grid(alpha=0.15)
            figure.suptitle(
                f"RR100 unit {int(unit_index)} · native response-active basis filters",
                fontsize=12,
                fontweight="semibold",
            )
            figure.text(
                0.5,
                0.012,
                "Basis signs/order are not biological labels; compare localization and smoothness. "
                "Twin and M77 retain their native grids.",
                ha="center",
                fontsize=7,
                color="0.35",
            )
            pdf.savefig(figure, facecolor="white")
            plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.atlas_rank < 1 or args.modes_per_unit < 1:
        raise ValueError("atlas-rank and modes-per-unit must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    twin_selection_path = args.twin_dir / "unit_selection.csv"
    m77_selection_path = args.m77_dir / "unit_selection.csv"
    twin_selection = load_selection(twin_selection_path)
    m77_selection = load_selection(m77_selection_path)
    units = validate_matching_units(twin_selection, m77_selection)

    fidelity, ranks = paired_fidelity(
        args.twin_dir / "heldout_response_fidelity.csv",
        args.m77_dir / "heldout_response_fidelity.csv",
    )
    summary = fidelity_summary(
        fidelity,
        ranks,
        seed=args.seed,
        bootstrap=args.bootstrap,
    )
    fidelity.to_csv(args.out_dir / "paired_unit_rank_fidelity.csv", index=False)
    summary.to_csv(args.out_dir / "fidelity_summary.csv", index=False)

    twin_energy = load_twin_energy(args.twin_dir)
    m77_energy = load_m77_energy(args.m77_dir)
    if twin_energy.shape[0] != len(units) or m77_energy.shape[0] != len(units):
        raise ValueError("Gradient spectra do not have one row per frozen unit")
    render_summary(
        fidelity,
        summary,
        ranks,
        twin_energy,
        m77_energy,
        args.out_dir / "twin_m77_subspace_summary",
    )

    twin_filters, m77_filters = load_native_filters(
        args.twin_dir,
        args.m77_dir,
        args.atlas_rank,
    )
    render_filter_atlas(
        units,
        twin_filters,
        m77_filters,
        modes=args.modes_per_unit,
        output=args.out_dir / "twin_m77_native_filter_atlas.pdf",
    )

    files = [
        twin_selection_path,
        m77_selection_path,
        args.twin_dir / "heldout_response_fidelity.csv",
        args.m77_dir / "heldout_response_fidelity.csv",
        args.m77_dir / "active_subspace.npz",
    ]
    manifest = {
        "analysis": "paired Twin versus M77 response-active subspaces",
        "unit_indices": units.tolist(),
        "models": ["Twin", "M77"],
        "comparison_contract": (
            "same frozen RR100 units; fidelity and cumulative Jacobian energy are "
            "compared within each model; native filters are displayed without "
            "cross-grid principal angles"
        ),
        "native_lattices": {
            "Twin": "120 Hz, 32x25x25 local support",
            "M77": "240 Hz, 60x35x35 support",
        },
        "input_sha256": {str(path.resolve()): sha256(path) for path in files},
    }
    (args.out_dir / "subspace_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote paired subspace supplement to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
