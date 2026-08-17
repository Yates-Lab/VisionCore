#!/usr/bin/env python3
"""Fit response-active subspaces for the native-240 M77 model.

The analysis evaluates RR100 units at the centre of M77's 35-pixel crop.  A
frozen selection CSV can be supplied so Twin and M77 are compared on exactly
the same biological units.  The fitted object is a local explanation of each
teacher model, so response fidelity and gradient-energy rank can be compared
despite the teachers' different native temporal and spatial lattices.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.upstream.real_trace_matrix.model import (
    load_mcfarland_outputs,
    load_pinned_multidataset_model,
    load_population_view,
)
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_POPULATION_SPEC_DIR,
    RR100_VERSION,
)


DEFAULT_CHECKPOINT = Path(
    "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/dekel240/"
    "D240M77c_dekel_native240_freqmasked_floor0p5_gnlrnalpha0p1_presplit_s201/"
    "analysis_candidates/epoch=279-val_bps_overall=0.5912.ckpt"
)
DEFAULT_DATASET_CONFIG = ROOT / (
    "paper/model_selection/configs/multi_240_long_split3_dekel35.yaml"
)
DEFAULT_UNIT_TABLE = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
    "merged/unit_feature_table.csv"
)
DEFAULT_OUT = ROOT / "outputs/dekel240_paper/m77_epoch279/response_subspace_matched"
MCFARLAND_OUTPUTS = ROOT / "scripts/mcfarland_outputs_mono.pkl"
TWIN_SUBSPACE_SUMMARY = ROOT / (
    "outputs/figures/fig4/nonlinear_phase_causal_v1/"
    "response_subspace_pilot/rank_summary.csv"
)

FRAME_RATE_HZ = 240.0
PPD = 37.50476617
N_LAGS = 60
INPUT_SIZE = 35
SCAFFOLD_SIZE = 9
DCT_MAX_SPATIAL_CPD = 14.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-label", default="M77")
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--unit-table", type=Path, default=DEFAULT_UNIT_TABLE)
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=None,
        help=(
            "Frozen unit selection to reuse verbatim. For the paired supplement, "
            "pass the Twin response-subspace unit_selection.csv."
        ),
    )
    parser.add_argument(
        "--robust-tuning-summary",
        type=Path,
        default=None,
        help="Optional native periodic-tuning summary used to replace the historical SF grouping",
    )
    parser.add_argument(
        "--periodic-tuning-csv",
        type=Path,
        default=None,
        help="Required with --robust-tuning-summary to define current response amplitudes",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=96)
    parser.add_argument("--n-gradient", type=int, default=128)
    parser.add_argument("--movie-batch-size", type=int, default=4)
    parser.add_argument("--ranks", default="1,2,4,8")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument(
        "--ridge-grid",
        default="1e-5,1e-4,1e-3,1e-2,1e-1",
        help="Training-only ridge grid; selected separately for every unit/rank",
    )
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--overwrite-bank", action="store_true")
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def dct_feature_mask(device: torch.device | str = "cpu") -> torch.Tensor:
    spatial_cpd = (
        torch.arange(INPUT_SIZE, dtype=torch.float32, device=device)
        * PPD
        / (2 * INPUT_SIZE)
    )
    radial = torch.sqrt(spatial_cpd[:, None].square() + spatial_cpd[None].square())
    spatial = radial <= DCT_MAX_SPATIAL_CPD
    return spatial[None].expand(N_LAGS, -1, -1).reshape(-1)


N_DCT_FEATURES = int(dct_feature_mask().sum())


def orthonormal_dct(n: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    positions = torch.arange(n, dtype=dtype) + 0.5
    frequencies = torch.arange(n, dtype=dtype)[:, None]
    basis = torch.cos(math.pi * frequencies * positions[None] / float(n))
    basis[0] *= math.sqrt(1.0 / n)
    if n > 1:
        basis[1:] *= math.sqrt(2.0 / n)
    return basis


def r2_score(target: np.ndarray, prediction: np.ndarray, axis: int = 0) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = np.sum((target - prediction) ** 2, axis=axis)
    total = np.sum(
        (target - np.mean(target, axis=axis, keepdims=True)) ** 2,
        axis=axis,
    )
    return 1.0 - residual / np.maximum(total, 1e-12)


def deterministic_unit_subset(table: pd.DataFrame, n_per_group: int = 6) -> pd.DataFrame:
    """Choose units evenly across prior-OSI ranks in two frozen SF groups."""
    required = {
        "unit_index",
        "sf_split_metric",
        "prior_orientation_selectivity_index",
        "dynamic_peak_response_amp",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing unit-selection columns: {sorted(missing)}")
    work = table.copy()
    work["response_subspace_group"] = np.where(
        pd.to_numeric(work.sf_split_metric, errors="coerce") < 0.5,
        "lower-SF",
        "higher-SF",
    )
    selected_parts = []
    quantiles = np.linspace(0.05, 0.95, int(n_per_group))
    for group in ("lower-SF", "higher-SF"):
        part = work.loc[work.response_subspace_group.eq(group)].copy()
        threshold = float(part.dynamic_peak_response_amp.median())
        part = (
            part.loc[part.dynamic_peak_response_amp.ge(threshold)]
            .sort_values(
                ["prior_orientation_selectivity_index", "unit_index"],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        if len(part) < n_per_group:
            raise ValueError(f"Only {len(part)} units in {group}")
        indices = np.rint(quantiles * (len(part) - 1)).astype(int)
        if len(np.unique(indices)) != len(indices):
            raise ValueError((group, indices.tolist()))
        selected = part.iloc[indices].copy()
        selected["selection_osi_quantile"] = quantiles
        selected["selection_rule"] = (
            "nearest rank to evenly spaced 0.05..0.95 prior-OSI quantiles among "
            "units at or above their SF-group median response amplitude"
        )
        selected["selection_min_dynamic_peak_response_amp"] = threshold
        selected_parts.append(selected)
    result = pd.concat(selected_parts, ignore_index=True)
    result.insert(0, "subset_index", np.arange(len(result), dtype=int))
    return result


class DirectPopulationReadout(nn.Module):
    """Dependency-light exact one-hot RR100 readout."""

    def __init__(
        self,
        feature_weights: torch.Tensor,
        bias: torch.Tensor,
        space_weights: torch.Tensor,
    ):
        super().__init__()
        self.features = nn.Conv2d(
            feature_weights.shape[1], feature_weights.shape[0], 1, bias=False
        )
        self.features.weight = nn.Parameter(feature_weights, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)
        self.space_weights = nn.Parameter(space_weights[:, None], requires_grad=False)
        self.n_units = int(len(bias))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(value)
        spatial = F.conv2d(
            feature_map,
            self.space_weights,
            groups=self.n_units,
            padding="valid",
        )
        return spatial + self.bias[None, :, None, None]


def movies_to_dct(movie: torch.Tensor) -> torch.Tensor:
    """Orthonormal 3-D DCT restricted to interpretable spatial frequencies."""
    if tuple(movie.shape[1:]) != (1, N_LAGS, INPUT_SIZE, INPUT_SIZE):
        raise ValueError(tuple(movie.shape))
    temporal = orthonormal_dct(N_LAGS, dtype=movie.dtype).to(movie.device)
    spatial = orthonormal_dct(INPUT_SIZE, dtype=movie.dtype).to(movie.device)
    value = torch.matmul(temporal, movie[:, 0].flatten(start_dim=2))
    value = value.reshape(-1, N_LAGS, INPUT_SIZE, INPUT_SIZE)
    value = torch.matmul(spatial, value)
    value = torch.matmul(value, spatial.T)
    return value.flatten(start_dim=1)[:, dct_feature_mask(movie.device)]


def dct_filters_to_movies(coefficients: torch.Tensor) -> torch.Tensor:
    if coefficients.shape[-1] != N_DCT_FEATURES:
        raise ValueError(tuple(coefficients.shape))
    leading = coefficients.shape[:-1]
    full = coefficients.new_zeros(*leading, N_LAGS * INPUT_SIZE * INPUT_SIZE)
    full[..., dct_feature_mask(coefficients.device)] = coefficients
    temporal = orthonormal_dct(N_LAGS, dtype=coefficients.dtype).to(coefficients.device)
    spatial = orthonormal_dct(INPUT_SIZE, dtype=coefficients.dtype).to(coefficients.device)
    value = full.reshape(-1, N_LAGS, INPUT_SIZE, INPUT_SIZE)
    value = torch.matmul(spatial.T, value)
    value = torch.matmul(value, spatial)
    value = torch.matmul(temporal.T, value.flatten(start_dim=2))
    return value.reshape(*leading, N_LAGS, INPUT_SIZE, INPUT_SIZE)


def generate_movie_batch(
    batch_size: int,
    *,
    seed: int,
    device: str,
) -> torch.Tensor:
    """Fixed-covariance Gaussian contexts on M77's normalized input scale."""
    generator = torch.Generator(device=device).manual_seed(int(seed))
    white = torch.randn(
        int(batch_size), 1, N_LAGS, INPUT_SIZE, INPUT_SIZE,
        generator=generator, device=device,
    )
    smooth3 = F.avg_pool3d(white, kernel_size=3, stride=1, padding=1)
    smooth7 = F.avg_pool3d(white, kernel_size=7, stride=1, padding=3)
    weights = torch.tensor([0.15, 0.85, 0.50], device=device)
    weights = weights / weights.square().sum().sqrt()
    movie = weights[0] * white + weights[1] * smooth3 + weights[2] * smooth7
    movie = movie - movie.mean(dim=(2, 3, 4), keepdim=True)
    movie = movie / movie.std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
    return (0.16 * movie).clamp(-0.5, 0.5)


@dataclass
class M77Teacher:
    model: object
    device: str

    def zero_behavior(self, batch_size: int, dtype: torch.dtype) -> torch.Tensor | None:
        modulator = getattr(self.model.model, "modulator", None)
        behavior_dim = getattr(modulator, "behavior_dim", None) if modulator is not None else None
        if behavior_dim is None:
            return None
        return torch.zeros(int(batch_size), int(behavior_dim), device=self.device, dtype=dtype)


def canonical_rr100_rows(model, outputs: list[dict]) -> list[dict]:
    """Reproduce the channel ordering used to freeze the existing RR100 view.

    The original helper selected positions in each session result using the
    stored CCnorm array.  We retain that exact historical ordering and attach
    the biological cell ID needed to find the corresponding model readout row.

    McFarland result rows are aligned to ``cids_used`` when that field is
    present.  ``cids`` can be a larger pre-mask list, so indexing it by a
    CCnorm-row position silently assigns the wrong biological cell whenever
    the evaluation dropped cells.  This mirrors ``scripts.spatial_info`` and
    only falls back to ``cids`` when it has exactly one entry per score row.
    """
    output_by_session = {str(value["sess"]): value for value in outputs}
    rows = []
    for model_index, session in enumerate(model.names):
        if session not in output_by_session:
            continue
        output = output_by_session[session]
        scores = np.asarray(output["ccnorm"]["ccnorm"])
        source_cids = None
        for key in ("cids_used", "cids"):
            candidate = np.asarray(output.get(key, []))
            if candidate.ndim == 1 and candidate.size == scores.size:
                source_cids = candidate.astype(int, copy=False)
                break
        if source_cids is None:
            raise ValueError(
                f"McFarland output {session!r} has {scores.size} CCnorm rows "
                "but no equally sized cids_used/cids array"
            )
        passing = np.flatnonzero(scores > 0.5)
        for source_position in passing:
            rows.append({
                "session": str(session),
                "model_readout_index": int(model_index),
                "historical_source_position": int(source_position),
                "source_cid": int(source_cids[source_position]),
                "ccnorm": float(scores[source_position]),
            })
    return rows


def build_m77_rr100_readout(
    teacher: M77Teacher,
    population_view,
    canonical_rows: list[dict],
    rr_unit_indices: np.ndarray,
) -> DirectPopulationReadout:
    """Map requested frozen RR100 units onto M77 by session and biological CID."""
    membership = np.asarray(population_view.membership, dtype=np.float64)
    rr_unit_indices = np.asarray(rr_unit_indices, dtype=int)
    selected_all = np.argmax(np.abs(membership), axis=1)
    expected = np.zeros_like(membership)
    expected[np.arange(len(selected_all)), selected_all] = 1.0
    if not np.array_equal(membership[rr_unit_indices], expected[rr_unit_indices]):
        raise ValueError("Requested RR100 rows must have exact positive one-hot membership")
    if membership.shape[1] != len(canonical_rows):
        raise ValueError(
            f"RR100 view expects {membership.shape[1]} canonical channels, "
            f"reconstructed {len(canonical_rows)}"
        )

    feature_weights, biases, masks = [], [], []
    missing = []
    for rr_unit in rr_unit_indices:
        row = canonical_rows[int(selected_all[int(rr_unit)])]
        model_index = int(row["model_readout_index"])
        session_readout = teacher.model.model.readouts[model_index]
        configured_cids = list(
            map(int, teacher.model.model.dataset_configs[model_index].get("cids", []))
        )
        try:
            local_index = configured_cids.index(int(row["source_cid"]))
        except ValueError:
            missing.append({"rr_unit": int(rr_unit), **row})
            continue
        feature_weights.append(session_readout.features.weight.detach()[local_index].clone())
        biases.append(session_readout.bias.detach()[local_index].clone())
        masks.append(
            session_readout.compute_gaussian_mask(
                SCAFFOLD_SIZE,
                SCAFFOLD_SIZE,
                torch.device(teacher.device),
            )[local_index].detach().clone()
        )
    if missing:
        raise RuntimeError(
            "Selected RR100 units are absent from M77's stored CID subset: "
            + json.dumps(missing)
        )
    return DirectPopulationReadout(
        torch.stack(feature_weights),
        torch.stack(biases),
        torch.stack(masks),
    ).to(teacher.device).eval()


def load_teacher(args: argparse.Namespace, selection: pd.DataFrame):
    model, _ = load_pinned_multidataset_model(
        checkpoint_path=args.checkpoint,
        dataset_configs=args.dataset_config,
        device=args.device,
        strict=True,
    )
    outputs, _ = load_mcfarland_outputs(MCFARLAND_OUTPUTS)
    population_view, _, _, _ = load_population_view(
        spec_dir=DEFAULT_POPULATION_SPEC_DIR,
        version_name=RR100_VERSION,
    )
    teacher = M77Teacher(model=model, device=args.device)
    readout = build_m77_rr100_readout(
        teacher,
        population_view,
        canonical_rr100_rows(model, outputs),
        selection.unit_index.to_numpy(int),
    )
    with torch.no_grad():
        movie = generate_movie_batch(1, seed=args.seed - 1, device=args.device)
        behavior = teacher.zero_behavior(1, movie.dtype)
        core = teacher.model.model.core_forward(movie, behavior)
        if tuple(core.shape[-3:]) != (1, SCAFFOLD_SIZE, SCAFFOLD_SIZE):
            raise RuntimeError(f"Unexpected M77 core shape {tuple(core.shape)}")
        output = readout(core[:, :, -1])
        if tuple(output.shape) != (1, len(selection), 1, 1):
            raise RuntimeError(f"Unexpected RR100 readout shape {tuple(output.shape)}")
    return teacher, readout


def preactivation(
    teacher: M77Teacher,
    readout: DirectPopulationReadout,
    movie: torch.Tensor,
) -> torch.Tensor:
    behavior = teacher.zero_behavior(len(movie), movie.dtype)
    core = teacher.model.model.core_forward(movie, behavior)
    return readout(core[:, :, -1])[:, :, 0, 0]


def make_bank(
    args: argparse.Namespace,
    teacher: M77Teacher,
    readout: DirectPopulationReadout,
    selection: pd.DataFrame,
) -> None:
    bank = args.output_dir / "bank"
    bank.mkdir(parents=True, exist_ok=True)
    counts = {"train": args.n_train, "test": args.n_test}
    for split_number, (split, count) in enumerate(counts.items()):
        feature_path = bank / f"{split}_features.npy"
        target_path = bank / f"{split}_z.npy"
        expected = (int(count), N_DCT_FEATURES)
        if (
            not args.overwrite_bank
            and feature_path.exists()
            and target_path.exists()
            and np.load(feature_path, mmap_mode="r").shape == expected
        ):
            print(f"bank {split}: reusing {feature_path}", flush=True)
            continue
        features = np.lib.format.open_memmap(
            feature_path, mode="w+", dtype=np.float16, shape=expected,
        )
        target = np.lib.format.open_memmap(
            target_path, mode="w+", dtype=np.float32, shape=(int(count), len(selection)),
        )
        split_seed = args.seed + split_number * 1_000_000
        for start in range(0, int(count), args.movie_batch_size):
            stop = min(start + args.movie_batch_size, int(count))
            movie = generate_movie_batch(stop - start, seed=split_seed + start, device=args.device)
            with torch.no_grad():
                features[start:stop] = movies_to_dct(movie).cpu().numpy().astype(np.float16)
                target[start:stop] = preactivation(teacher, readout, movie).cpu().numpy()
            if start == 0 or stop == count or stop % 64 == 0:
                print(f"bank {split}: {stop}/{count}", flush=True)
        features.flush(); target.flush()

    train = np.asarray(np.load(bank / "train_features.npy", mmap_mode="r"), dtype=np.float32)
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)
    np.savez(bank / "feature_scaling.npz", mean=mean, std=std)


def measure_gradients(
    args: argparse.Namespace,
    teacher: M77Teacher,
    readout: DirectPopulationReadout,
    selection: pd.DataFrame,
) -> Path:
    path = args.output_dir / "bank/train_gradient_dct.npy"
    count = min(int(args.n_gradient), int(args.n_train))
    shape = (count, len(selection), N_DCT_FEATURES)
    if (
        path.exists()
        and not args.overwrite_bank
        and np.load(path, mmap_mode="r").shape == shape
    ):
        print(f"gradients: reusing {path}", flush=True)
        return path
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    for start in range(0, count, args.movie_batch_size):
        stop = min(start + args.movie_batch_size, count)
        movie = generate_movie_batch(stop - start, seed=args.seed + start, device=args.device)
        movie.requires_grad_(True)
        z = preactivation(teacher, readout, movie)
        for unit in range(len(selection)):
            gradient = torch.autograd.grad(
                z[:, unit].sum(),
                movie,
                retain_graph=unit + 1 < len(selection),
                create_graph=False,
            )[0]
            output[start:stop, unit] = movies_to_dct(gradient).detach().cpu().numpy().astype(np.float16)
        if start == 0 or stop == count or stop % 32 == 0:
            print(f"gradients: {stop}/{count} contexts", flush=True)
    output.flush()
    return path


@torch.no_grad()
def active_directions(
    gradient_path: Path,
    feature_std: torch.Tensor,
    max_rank: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top gradient-covariance modes via the smaller context-space Gram matrix."""
    source = np.load(gradient_path, mmap_mode="r")
    directions, spectra = [], []
    for unit in range(source.shape[1]):
        gradient = torch.from_numpy(np.asarray(source[:, unit], dtype=np.float32)).to(device)
        gradient = gradient * feature_std[None]
        gram = gradient @ gradient.T
        eigenvalue, eigenvector = torch.linalg.eigh(gram)
        order = eigenvalue.argsort(descending=True)
        eigenvalue = eigenvalue[order].clamp_min(0)
        eigenvector = eigenvector[:, order]
        singular = eigenvalue.sqrt()
        top = min(max_rank, int((singular > 1e-8).sum()))
        basis = (eigenvector[:, :top].T @ gradient) / singular[:top, None].clamp_min(1e-8)
        if top < max_rank:
            basis = F.pad(basis, (0, 0, 0, max_rank - top))
        directions.append(basis[:max_rank])
        spectra.append(singular)
        print(f"active directions: unit {unit + 1}/{source.shape[1]}", flush=True)
    return torch.stack(directions), torch.stack(spectra)


def design_matrix(projected: torch.Tensor) -> torch.Tensor:
    columns = [torch.ones(len(projected), 1, device=projected.device), projected]
    for left in range(projected.shape[1]):
        for right in range(left, projected.shape[1]):
            columns.append((projected[:, left] * projected[:, right])[:, None])
    return torch.cat(columns, dim=1)


def quadratic_design_columns(rank: int) -> int:
    return 1 + int(rank) + int(rank) * (int(rank) + 1) // 2


@torch.no_grad()
def solve_ridge(design: torch.Tensor, target: torch.Tensor, ridge: float) -> torch.Tensor:
    gram = design.T @ design
    penalty = torch.eye(len(gram), device=design.device, dtype=design.dtype) * (
        float(ridge) * len(design)
    )
    penalty[0, 0] = 0
    return torch.linalg.solve(gram + penalty, design.T @ target)


@torch.no_grad()
def select_ridge_training_only(
    design: torch.Tensor,
    target_z: torch.Tensor,
    ridge_grid: list[float],
) -> tuple[float, float]:
    """Choose ridge on a deterministic 20% training split using rate R2."""
    if len(design) < 10:
        raise ValueError("ridge selection requires at least ten training samples")
    rows = torch.arange(len(design), device=design.device)
    validation = rows.remainder(5).eq(0)
    fit = ~validation
    target_rate = F.softplus(target_z[validation]).cpu().numpy()
    scores = []
    for ridge in ridge_grid:
        theta = solve_ridge(design[fit], target_z[fit], ridge)
        prediction = F.softplus(design[validation] @ theta).cpu().numpy()
        scores.append(float(r2_score(target_rate, prediction, axis=0)))
    best = int(np.nanargmax(scores))
    return float(ridge_grid[best]), float(scores[best])


@torch.no_grad()
def fit_and_score(
    args: argparse.Namespace,
    selection: pd.DataFrame,
    directions: torch.Tensor,
    spectra: torch.Tensor,
) -> pd.DataFrame:
    bank = args.output_dir / "bank"
    scaling = np.load(bank / "feature_scaling.npz")
    mean = torch.from_numpy(scaling["mean"]).to(args.device)
    std = torch.from_numpy(scaling["std"]).to(args.device)

    def load(split: str):
        x = torch.from_numpy(
            np.asarray(np.load(bank / f"{split}_features.npy"), dtype=np.float32)
        ).to(args.device)
        z = torch.from_numpy(
            np.asarray(np.load(bank / f"{split}_z.npy"), dtype=np.float32)
        ).to(args.device)
        return (x - mean) / std, z

    x_train, z_train = load("train")
    x_test, z_test = load("test")
    ridge_grid = [
        float(value) for value in args.ridge_grid.split(",") if value.strip()
    ] or [float(args.ridge)]
    if any(value <= 0 for value in ridge_grid):
        raise ValueError("--ridge-grid entries must be positive")
    rows = []
    for rank in [int(v) for v in args.ranks.split(",") if v.strip()]:
        for unit in range(len(selection)):
            basis = directions[unit, :rank]
            train_design = design_matrix(x_train @ basis.T)
            test_design = design_matrix(x_test @ basis.T)
            selected_ridge, validation_rate_r2 = select_ridge_training_only(
                train_design,
                z_train[:, unit],
                ridge_grid,
            )
            theta = solve_ridge(
                train_design,
                z_train[:, unit],
                selected_ridge,
            )
            z_prediction = test_design @ theta
            rate = F.softplus(z_test[:, unit])
            rate_prediction = F.softplus(z_prediction)
            target_np = rate.cpu().numpy()
            prediction_np = rate_prediction.cpu().numpy()
            cumulative = spectra[unit].square().cumsum(0) / spectra[unit].square().sum().clamp_min(1e-12)
            rows.append({
                "subset_index": int(selection.subset_index.iloc[unit]),
                "unit_index": int(selection.unit_index.iloc[unit]),
                "response_subspace_group": selection.response_subspace_group.iloc[unit],
                "rank": rank,
                "test_rate_r2": float(r2_score(target_np, prediction_np, axis=0)),
                "test_rate_correlation": float(np.corrcoef(target_np, prediction_np)[0, 1]),
                "test_z_r2": float(r2_score(z_test[:, unit].cpu().numpy(), z_prediction.cpu().numpy(), axis=0)),
                "gradient_energy_fraction": float(cumulative[min(rank, len(cumulative)) - 1]),
                "selected_ridge": selected_ridge,
                "validation_rate_r2": validation_rate_r2,
                "quadratic_design_columns": int(train_design.shape[1]),
                "train_samples_per_coefficient": float(
                    len(train_design) / train_design.shape[1]
                ),
            })
        median = np.median([row["test_rate_r2"] for row in rows if row["rank"] == rank])
        print(f"rank {rank}: median held-out rate R2={median:.3f}", flush=True)
    return pd.DataFrame(rows)


def render(
    args: argparse.Namespace,
    metrics: pd.DataFrame,
    selection: pd.DataFrame,
    directions: torch.Tensor,
    spectra: torch.Tensor,
) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.25))
    ranks = sorted(metrics["rank"].unique())
    for _, part in metrics.groupby("unit_index"):
        axes[0].plot(part["rank"], part.test_rate_r2, color="0.65", lw=0.8, alpha=0.7)
    median = metrics.groupby("rank").test_rate_r2.median().reindex(ranks)
    axes[0].plot(
        ranks, median, "o-", color="#2369a2", lw=2.5,
        label=f"{args.model_label} median",
    )
    if TWIN_SUBSPACE_SUMMARY.exists():
        twin = pd.read_csv(TWIN_SUBSPACE_SUMMARY)
        twin = twin.loc[twin["rank"].isin(ranks)]
        axes[0].plot(
            twin["rank"], twin["median_test_rate_r2"], "s--",
            color="#d06a2e", lw=1.8, label="Twin median",
        )
    axes[0].axhline(0.8, color="0.45", ls="--", lw=1)
    axes[0].set(xlabel="private subspace rank", ylabel="held-out response $R^2$", xticks=ranks)
    axes[0].legend(frameon=False)

    energy = spectra.square().cumsum(1) / spectra.square().sum(1, keepdim=True).clamp_min(1e-12)
    shown = min(16, energy.shape[1])
    x = np.arange(1, shown + 1)
    for unit in range(len(selection)):
        axes[1].plot(x, energy[unit, :shown].cpu(), color="0.7", lw=0.8, alpha=0.7)
    axes[1].plot(x, energy[:, :shown].median(0).values.cpu(), color="#c45432", lw=2.5)
    axes[1].axhline(0.8, color="0.45", ls="--", lw=1)
    axes[1].set(xlabel="gradient subspace rank", ylabel="cumulative Jacobian energy", ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(args.output_dir / "rank_response_fidelity.png", dpi=220)
    fig.savefig(args.output_dir / "rank_response_fidelity.pdf")
    plt.close(fig)

    scaling = np.load(args.output_dir / "bank/feature_scaling.npz")
    std = torch.from_numpy(scaling["std"]).to(directions.device)
    atlas_rank = min(8, directions.shape[1])
    physical = directions[:, :atlas_rank] / std[None, None]
    filters = dct_filters_to_movies(physical).cpu().numpy()
    fig, axes = plt.subplots(
        len(selection), atlas_rank,
        figsize=(1.5 * atlas_rank, 1.25 * len(selection)), squeeze=False,
    )
    for unit in range(len(selection)):
        for mode in range(atlas_rank):
            movie = filters[unit, mode]
            lag = int(np.argmax(np.square(movie).sum(axis=(1, 2))))
            image = movie[lag]
            limit = float(np.percentile(np.abs(image), 99)) or 1.0
            axes[unit, mode].imshow(
                image, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest"
            )
            axes[unit, mode].set_xticks([]); axes[unit, mode].set_yticks([])
            if unit == 0:
                axes[unit, mode].set_title(f"mode {mode + 1}")
            if mode == 0:
                axes[unit, mode].set_ylabel(
                    f"u{int(selection.unit_index.iloc[unit]):03d}\n{1000 * lag / FRAME_RATE_HZ:.0f} ms"
                )
    fig.suptitle(
        f"{args.model_label} response-active filters (peak-energy frame)", y=0.998
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / f"rank{atlas_rank}_filter_atlas.png", dpi=220)
    fig.savefig(args.output_dir / f"rank{atlas_rank}_filter_atlas.pdf")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.selection_csv is not None:
        selection = pd.read_csv(args.selection_csv)
        required = {"unit_index", "response_subspace_group"}
        missing = required.difference(selection.columns)
        if missing:
            raise ValueError(
                f"Frozen selection is missing columns: {sorted(missing)}"
            )
        if selection.unit_index.duplicated().any():
            raise ValueError("Frozen selection contains duplicate unit_index values")
        selection = selection.reset_index(drop=True)
        selection["subset_index"] = np.arange(len(selection), dtype=int)
    else:
        unit_table = pd.read_csv(args.unit_table)
        if args.robust_tuning_summary is not None:
            if args.periodic_tuning_csv is None:
                raise ValueError(
                    "--robust-tuning-summary requires --periodic-tuning-csv"
                )
            robust = pd.read_csv(args.robust_tuning_summary)
            periodic = pd.read_csv(args.periodic_tuning_csv)
            amplitudes = (
                periodic.loc[periodic.temporal_hz.gt(0)]
                .groupby("unit_index", as_index=False)
                .response_amp_rms.max()
                .rename(
                    columns={
                        "response_amp_rms": "current_dynamic_peak_response_amp"
                    }
                )
            )
            unit_table = unit_table.merge(
                robust[["unit_index", "low_sf_censored", "preferred_sf_cpd"]],
                on="unit_index",
                how="inner",
                validate="one_to_one",
            ).merge(
                amplitudes,
                on="unit_index",
                how="inner",
                validate="one_to_one",
            )
            unit_table["sf_split_metric"] = np.where(
                unit_table.low_sf_censored.astype(bool),
                0.0,
                unit_table.preferred_sf_cpd.astype(float),
            )
            unit_table["dynamic_peak_response_amp"] = (
                unit_table.current_dynamic_peak_response_amp.astype(float)
            )
        selection = deterministic_unit_subset(unit_table)
    selection.to_csv(args.output_dir / "unit_selection.csv", index=False)
    print(f"selected RR100 units: {selection.unit_index.tolist()}", flush=True)
    ranks = [int(value) for value in args.ranks.split(",") if value.strip()]
    if not ranks or min(ranks) < 1:
        raise ValueError("--ranks must contain positive integers")
    required_train = 4 * quadratic_design_columns(max(ranks))
    if int(args.n_train) < required_train:
        raise ValueError(
            f"n_train={args.n_train} underpowers the rank-{max(ranks)} quadratic "
            f"model ({quadratic_design_columns(max(ranks))} coefficients); "
            f"require at least {required_train} training contexts"
        )
    teacher, readout = load_teacher(args, selection)
    make_bank(args, teacher, readout, selection)
    gradient_path = measure_gradients(args, teacher, readout, selection)
    del readout, teacher
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    scaling = np.load(args.output_dir / "bank/feature_scaling.npz")
    feature_std = torch.from_numpy(scaling["std"]).to(args.device)
    directions, spectra = active_directions(
        gradient_path, feature_std, max(ranks), args.device
    )
    np.savez_compressed(
        args.output_dir / "active_subspace.npz",
        directions=directions.cpu().numpy().astype(np.float32),
        singular_values=spectra.cpu().numpy().astype(np.float32),
    )
    metrics = fit_and_score(args, selection, directions, spectra)
    metrics.to_csv(args.output_dir / "heldout_response_fidelity.csv", index=False)
    render(args, metrics, selection, directions, spectra)
    summary = metrics.groupby("rank").agg(
        median_test_rate_r2=("test_rate_r2", "median"),
        min_test_rate_r2=("test_rate_r2", "min"),
        median_test_rate_correlation=("test_rate_correlation", "median"),
        median_gradient_energy_fraction=("gradient_energy_fraction", "median"),
    ).reset_index()
    summary.to_csv(args.output_dir / "rank_summary.csv", index=False)
    save_json(args.output_dir / "run_summary.json", {
        "checkpoint": str(args.checkpoint.resolve()),
        "model_label": str(args.model_label),
        "unit_selection_tuning": (
            {"frozen_selection_csv": str(args.selection_csv.resolve())}
            if args.selection_csv is not None
            else {
                "robust_tuning_summary": str(args.robust_tuning_summary.resolve()),
                "periodic_tuning_csv": str(args.periodic_tuning_csv.resolve()),
                "sf_groups": "low-SF censored versus resolved periodic SF peak",
            }
            if args.robust_tuning_summary is not None
            else "historical unit table"
        ),
        "protocol": (
            "Exact stimulus Jacobians and gradient-covariance subspaces on fixed-covariance "
            "Gaussian 60x35x35 movies; RR100 readouts evaluated at the native crop centre; "
            "behavior fixed to its standardized zero context."
        ),
        "n_dct_features": N_DCT_FEATURES,
        "n_train": int(args.n_train),
        "n_test": int(args.n_test),
        "n_gradient": int(min(args.n_gradient, args.n_train)),
        "ridge_selection": {
            "grid": [float(value) for value in args.ridge_grid.split(",") if value.strip()],
            "selection_support": "deterministic 20% split of training contexts only",
            "minimum_training_contexts_per_quadratic_coefficient": 4,
        },
        "elapsed_seconds": time.time() - started,
        "rank_summary": summary.to_dict(orient="records"),
        "success_gate": "median held-out response R2 >= 0.8",
        "gate_passed_ranks": summary.loc[
            summary.median_test_rate_r2.ge(0.8), "rank"
        ].astype(int).tolist(),
    })
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
