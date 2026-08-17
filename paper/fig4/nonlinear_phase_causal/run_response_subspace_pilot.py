#!/usr/bin/env python3
"""Fit response-predictive spatiotemporal subspaces for a subset of RR100."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.phase_spatial_followup.run_exact_subset import build_direct_readout
from paper.fig4.nonlinear_phase_causal.response_subspace import (
    INPUT_SIZE,
    N_DCT_FEATURES,
    N_LAGS,
    OUTPUT_SIZE,
    PATCH_SIZE,
    LowRankLQ,
    ResponseSubspaceState,
    convolutional_predict,
    dct_filters_to_movies,
    dct_frequency_penalty,
    deterministic_unit_subset,
    extract_local_patches,
    generate_movie_batch,
    patches_to_dct,
    r2_score,
    response_grid_offsets,
)
from paper.fig4.nonlinear_phase_causal.run_experiment import preactivation
from paper.fig4.upstream.real_trace_matrix.model import RealTraceMatrixScorer
from paper.fig4.upstream.run_real_trace_matrix import (
    DEFAULT_DATASET_CONFIGS,
    DEFAULT_POPULATION_SPEC_DIR,
    MODEL_CHECKPOINT_PATH,
    RR100_VERSION,
)


DEFAULT_OUT = ROOT / "outputs/figures/fig4/nonlinear_phase_causal_v1/response_subspace_pilot"
UNIT_TABLE = ROOT / (
    "outputs/active_sensing_movie_information/"
    "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1/"
    "merged/unit_feature_table.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-train-movies", type=int, default=768)
    parser.add_argument("--n-val-movies", type=int, default=96)
    parser.add_argument("--n-test-movies", type=int, default=96)
    parser.add_argument("--movie-batch-size", type=int, default=8)
    parser.add_argument("--ranks", default="1,2,4,8")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--fit-batch-size", type=int, default=512)
    parser.add_argument("--locations-per-movie", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--smoothness", type=float, default=0.003)
    parser.add_argument("--ste-iterations", type=int, default=3)
    parser.add_argument("--gradient-movies", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--overwrite-bank", action="store_true")
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_teacher(device: str):
    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=MODEL_CHECKPOINT_PATH,
        dataset_configs=DEFAULT_DATASET_CONFIGS,
        population_spec_dir=DEFAULT_POPULATION_SPEC_DIR,
        rr100_version=RR100_VERSION,
        device=device,
        strict=True,
        mcfarland_outputs=ROOT / "scripts/mcfarland_outputs_mono.pkl",
    )
    return scorer, build_direct_readout(scorer).eval()


def make_bank(args: argparse.Namespace, selection: pd.DataFrame) -> None:
    bank = args.output_dir / "bank"
    bank.mkdir(parents=True, exist_ok=True)
    unit_indices = selection.unit_index.to_numpy(int)
    offsets = response_grid_offsets()
    n_grid = len(offsets)
    n_features = N_DCT_FEATURES
    split_counts = {
        "train": args.n_train_movies,
        "val": args.n_val_movies,
        "test": args.n_test_movies,
    }
    for split_number, (split, n_movies) in enumerate(split_counts.items()):
        feature_path = bank / f"{split}_features.npy"
        target_path = bank / f"{split}_z.npy"
        full_path = bank / f"{split}_z_full.npy"
        if feature_path.exists() and target_path.exists() and full_path.exists() and not args.overwrite_bank:
            print(f"bank {split}: reusing {feature_path}", flush=True)
            continue
        features = np.lib.format.open_memmap(
            feature_path, mode="w+", dtype=np.float16,
            shape=(n_movies * n_grid, n_features),
        )
        targets = np.lib.format.open_memmap(
            target_path, mode="w+", dtype=np.float32,
            shape=(n_movies * n_grid, len(unit_indices)),
        )
        full_targets = np.lib.format.open_memmap(
            full_path, mode="w+", dtype=np.float16,
            shape=(n_movies, len(unit_indices), OUTPUT_SIZE, OUTPUT_SIZE),
        )
        rms_rows, mix_rows, clip_rows = [], [], []
        start_time = time.time()
        for start in range(0, n_movies, args.movie_batch_size):
            stop = min(start + args.movie_batch_size, n_movies)
            movie, meta = generate_movie_batch(
                stop - start,
                seed=args.seed + split_number * 1_000_000 + start,
                device=args.device,
            )
            with torch.no_grad():
                behavior = scorer._zero_behavior(len(movie), movie.dtype)
                z = preactivation(scorer.model.model, readout, movie, behavior)[:, unit_indices]
                patches = extract_local_patches(movie, offsets)
                coefficients = patches_to_dct(patches)
            row0, row1 = start * n_grid, stop * n_grid
            features[row0:row1] = coefficients.reshape(-1, n_features).cpu().numpy().astype(np.float16)
            out_y = torch.as_tensor(offsets[:, 0] + OUTPUT_SIZE // 2, device=z.device)
            out_x = torch.as_tensor(offsets[:, 1] + OUTPUT_SIZE // 2, device=z.device)
            selected_z = z[:, :, out_y, out_x].permute(0, 2, 1)
            targets[row0:row1] = selected_z.reshape(-1, len(unit_indices)).cpu().numpy()
            full_targets[start:stop] = z.cpu().numpy().astype(np.float16)
            rms_rows.append(meta["rms"])
            mix_rows.append(meta["mix_weights"])
            clip_rows.append(meta["clipped_fraction"])
            if start == 0 or stop == n_movies or stop % 128 == 0:
                print(f"bank {split}: {stop}/{n_movies} movies", flush=True)
        features.flush(); targets.flush()
        full_targets.flush()
        np.savez_compressed(
            bank / f"{split}_stimulus_metadata.npz",
            rms=np.concatenate(rms_rows),
            mix_weights=np.concatenate(mix_rows),
            clipped_fraction=np.concatenate(clip_rows),
        )
        print(f"bank {split}: finished in {time.time() - start_time:.1f}s", flush=True)
    train = np.load(bank / "train_features.npy", mmap_mode="r")
    sums = np.zeros(train.shape[1], dtype=np.float64)
    squares = np.zeros(train.shape[1], dtype=np.float64)
    for start in range(0, len(train), 2048):
        value = np.asarray(train[start : start + 2048], dtype=np.float32)
        sums += value.sum(axis=0, dtype=np.float64)
        squares += np.square(value, dtype=np.float64).sum(axis=0)
    mean = sums / len(train)
    var = squares / len(train) - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-10))
    np.savez(bank / "feature_scaling.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))
    save_json(bank / "manifest.json", {
        "analysis": "RR100 response-predictive local spatiotemporal subspace",
        "target": "exact twin preactivation z at response-map pixels",
        "decoder": "per-unit full linear plus quadratic function of private rank-k projections",
        "movie_counts": split_counts,
        "response_grid_offsets_yx": offsets.tolist(),
        "input_output_alignment": "input shift 2 pixels = output shift 1 pixel; empirically >0.999999 correlation",
        "local_movie_shape": [N_LAGS, PATCH_SIZE, PATCH_SIZE],
        "feature_basis": "all 32 temporal DCT modes crossed with spatial DCT modes <=14 c/deg",
        "n_features": n_features,
        "stimulus": "fixed-covariance Gaussian movies; fixed RMS 0.16; clipped to [-0.5,0.5]",
        "unit_indices": unit_indices.tolist(),
        "seed": args.seed,
        "elapsed_seconds": time.time() - started,
    })


def measure_response_gradients(
    args: argparse.Namespace,
    selection: pd.DataFrame,
    scorer: RealTraceMatrixScorer,
    readout,
) -> None:
    """Measure exact local gradients only as an active-subspace estimator."""
    path = args.output_dir / "bank/train_response_gradient_dct.npy"
    n_movies = min(int(args.gradient_movies), int(args.n_train_movies))
    expected_shape = (n_movies, len(selection), N_DCT_FEATURES)
    if path.exists() and not args.overwrite_bank and np.load(path, mmap_mode="r").shape == expected_shape:
        print(f"response gradients: reusing {path}", flush=True)
        return
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=expected_shape)
    unit_indices = selection.unit_index.to_numpy(int)
    for start in range(0, n_movies, args.movie_batch_size):
        stop = min(start + args.movie_batch_size, n_movies)
        movie, _ = generate_movie_batch(
            stop - start,
            seed=args.seed + start,
            device=args.device,
        )
        movie.requires_grad_(True)
        behavior = scorer._zero_behavior(len(movie), movie.dtype)
        z = preactivation(scorer.model.model, readout, movie, behavior)[:, unit_indices]
        for subset_index in range(len(unit_indices)):
            gradient = torch.autograd.grad(
                z[:, subset_index, OUTPUT_SIZE // 2, OUTPUT_SIZE // 2].sum(),
                movie,
                retain_graph=subset_index + 1 < len(unit_indices),
                create_graph=False,
            )[0]
            local = gradient[:, 0, :, 75 - PATCH_SIZE // 2 : 75 + PATCH_SIZE // 2 + 1,
                             75 - PATCH_SIZE // 2 : 75 + PATCH_SIZE // 2 + 1]
            coefficients = patches_to_dct(local[:, None])[:, 0]
            output[start:stop, subset_index] = coefficients.detach().cpu().numpy().astype(np.float16)
        if start == 0 or stop == n_movies or stop % 64 == 0:
            print(f"response gradients: {stop}/{n_movies} contexts", flush=True)
    output.flush()


@torch.no_grad()
def active_gradient_initialization(
    gradient_path: Path,
    feature_std: torch.Tensor,
    rank: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return response-active directions in standardized stimulus coordinates."""
    gradient = torch.from_numpy(np.asarray(np.load(gradient_path))).to(device).float()
    # dx_raw/dx_standardized = feature_std.
    gradient = gradient * feature_std[None, None]
    directions, spectra = [], []
    for unit in range(gradient.shape[1]):
        _, singular, vh = torch.linalg.svd(gradient[:, unit], full_matrices=False)
        directions.append(vh[:rank])
        spectra.append(singular)
    return torch.stack(directions), torch.stack(spectra)


def make_dense_design(args: argparse.Namespace) -> None:
    """Use many translated response-map pixels per movie without more teacher calls."""
    bank = args.output_dir / "bank"
    split_counts = {
        "train": args.n_train_movies,
        "val": args.n_val_movies,
        "test": args.n_test_movies,
    }
    n_features = N_DCT_FEATURES
    location_count = int(args.locations_per_movie)
    candidates = np.stack(
        np.meshgrid(np.arange(-20, 21), np.arange(-20, 21), indexing="ij"), axis=-1
    ).reshape(-1, 2)
    for split_number, (split, n_movies) in enumerate(split_counts.items()):
        feature_path = bank / f"{split}_dense_features.npy"
        target_path = bank / f"{split}_dense_z.npy"
        offset_path = bank / f"{split}_dense_offsets.npy"
        expected = (n_movies * location_count, n_features)
        if feature_path.exists() and target_path.exists() and offset_path.exists() and not args.overwrite_bank:
            observed = np.load(feature_path, mmap_mode="r").shape
            if observed == expected:
                print(f"dense design {split}: reusing {feature_path}", flush=True)
                continue
        features = np.lib.format.open_memmap(
            feature_path, mode="w+", dtype=np.float16, shape=expected,
        )
        targets = np.lib.format.open_memmap(
            target_path, mode="w+", dtype=np.float32,
            shape=(n_movies * location_count, np.load(bank / f"{split}_z_full.npy", mmap_mode="r").shape[1]),
        )
        offsets_saved = np.lib.format.open_memmap(
            offset_path, mode="w+", dtype=np.int8, shape=(n_movies, location_count, 2),
        )
        full_target = np.load(bank / f"{split}_z_full.npy", mmap_mode="r")
        for start in range(0, n_movies, args.movie_batch_size):
            stop = min(start + args.movie_batch_size, n_movies)
            movie, _ = generate_movie_batch(
                stop - start,
                seed=args.seed + split_number * 1_000_000 + start,
                device=args.device,
            )
            rng = np.random.default_rng(args.seed + 7_000_000 + split_number * 1_000_000 + start)
            offsets = candidates[rng.choice(len(candidates), size=location_count, replace=False)]
            with torch.no_grad():
                coefficients = patches_to_dct(extract_local_patches(movie, offsets))
            row0, row1 = start * location_count, stop * location_count
            features[row0:row1] = coefficients.reshape(-1, n_features).cpu().numpy().astype(np.float16)
            batch_target = torch.from_numpy(np.asarray(full_target[start:stop], dtype=np.float32)).to(args.device)
            out_y = torch.as_tensor(offsets[:, 0] + OUTPUT_SIZE // 2, device=args.device)
            out_x = torch.as_tensor(offsets[:, 1] + OUTPUT_SIZE // 2, device=args.device)
            selected = batch_target[:, :, out_y, out_x].permute(0, 2, 1)
            targets[row0:row1] = selected.reshape(-1, selected.shape[-1]).cpu().numpy()
            offsets_saved[start:stop] = offsets[None]
        features.flush(); targets.flush(); offsets_saved.flush()
        print(f"dense design {split}: {expected[0]} translated response examples", flush=True)

    train = np.load(bank / "train_dense_features.npy", mmap_mode="r")
    sums = np.zeros(train.shape[1], dtype=np.float64)
    squares = np.zeros(train.shape[1], dtype=np.float64)
    for start in range(0, len(train), 2048):
        value = np.asarray(train[start : start + 2048], dtype=np.float32)
        sums += value.sum(axis=0, dtype=np.float64)
        squares += np.square(value, dtype=np.float64).sum(axis=0)
    mean = sums / len(train)
    variance = squares / len(train) - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-10))
    np.savez(bank / "feature_scaling.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))


def _unit_qr(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.qr(value.transpose(1, 2), mode="reduced").Q.transpose(1, 2)


@torch.no_grad()
def ste_initialization(
    x: torch.Tensor,
    y: torch.Tensor,
    rank: int,
    *,
    seed: int,
    iterations: int,
) -> torch.Tensor:
    """Matrix-free STA/STE directions for deterministic Gaussian stimulation."""
    n_units, n_features = y.shape[1], x.shape[1]
    width = min(n_features, max(2 * rank, rank + 4))
    generator = torch.Generator(device=x.device).manual_seed(int(seed))
    basis = torch.randn(n_units, width, n_features, generator=generator, device=x.device)
    centered = y - y.mean(dim=0, keepdim=True)
    sta = torch.matmul(centered.T, x.float()) / len(x)
    sta = sta / sta.norm(dim=1, keepdim=True).clamp_min(1e-8)
    basis[:, 0] = sta
    basis = _unit_qr(basis)

    def covariance_operator(vectors: torch.Tensor) -> torch.Tensor:
        flat = vectors.reshape(n_units * width, n_features)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            projected = torch.matmul(x, flat.T.to(x.dtype)).reshape(len(x), n_units, width)
            weighted = projected * centered[:, :, None].to(projected.dtype)
            result = torch.matmul(x.T, weighted.reshape(len(x), -1)).T
        return result.float().reshape(n_units, width, n_features) / len(x)

    # A^2 power iteration retains both positive and negative covariance modes.
    for iteration in range(max(1, int(iterations))):
        basis = _unit_qr(covariance_operator(basis))
        basis = _unit_qr(covariance_operator(basis))
        print(f"  STE subspace iteration {iteration + 1}/{iterations}", flush=True)
    applied = covariance_operator(basis)
    small = torch.einsum("uwp,uvp->uwv", basis, applied)
    small = 0.5 * (small + small.transpose(1, 2))
    eigenvalue, eigenvector = torch.linalg.eigh(small)
    ordering = eigenvalue.abs().argsort(dim=1, descending=True)
    ordered = torch.gather(eigenvector, 2, ordering[:, None].expand(-1, width, -1))
    directions = torch.einsum("uwr,uwp->urp", ordered[:, :, :rank], basis)
    # If an STA is informative, guarantee it a candidate dimension.
    sta_strength = torch.matmul(centered.T, x.float()).norm(dim=1) / len(x)
    for unit in range(n_units):
        if float(sta_strength[unit]) > 1e-5:
            directions[unit, -1] = sta[unit]
    return _unit_qr(directions)


@torch.no_grad()
def initialize_decoder(model: LowRankLQ, x: torch.Tensor, y: torch.Tensor) -> None:
    """Solve the L+Q decoder exactly for the initial response subspace."""
    model.bias.zero_(); model.linear.zero_(); model.quadratic.zero_()
    for unit in range(model.n_units):
        projected = torch.matmul(x.float(), model.weights[unit].T)
        columns = [torch.ones(len(x), 1, device=x.device), projected]
        pairs = []
        for left in range(model.rank):
            for right in range(left, model.rank):
                columns.append((projected[:, left] * projected[:, right])[:, None])
                pairs.append((left, right))
        design = torch.cat(columns, dim=1)
        gram = design.T @ design
        ridge = torch.eye(len(gram), device=x.device) * (1e-4 * len(x))
        ridge[0, 0] = 0.0
        theta = torch.linalg.solve(gram + ridge, design.T @ y[:, unit])
        model.bias[unit] = theta[0]
        model.linear[unit] = theta[1 : 1 + model.rank]
        for coefficient, (left, right) in zip(theta[1 + model.rank :], pairs):
            if left == right:
                model.quadratic[unit, left, right] = coefficient
            else:
                model.quadratic[unit, left, right] = coefficient / 2
                model.quadratic[unit, right, left] = coefficient / 2


def initialize_from_response(
    model: LowRankLQ,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int,
    iterations: int,
) -> None:
    """Initialize with matrix-free spike-triggered average/covariance modes."""
    with torch.no_grad():
        model.weights.copy_(ste_initialization(x, y, model.rank, seed=seed, iterations=iterations))
        initialize_decoder(model, x, y)


def fit_rank(args: argparse.Namespace, rank: int, selection: pd.DataFrame) -> tuple[ResponseSubspaceState, pd.DataFrame]:
    bank = args.output_dir / "bank"
    scaling = np.load(bank / "feature_scaling.npz")
    feature_mean = torch.from_numpy(scaling["mean"]).to(args.device)
    feature_std = torch.from_numpy(scaling["std"]).to(args.device)

    def load_split(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.from_numpy(np.asarray(np.load(bank / f"{split}_dense_features.npy"))).to(args.device)
        target = torch.from_numpy(np.asarray(np.load(bank / f"{split}_dense_z.npy"))).to(args.device)
        features = ((features.float() - feature_mean) / feature_std).to(torch.float16)
        return features, target

    x_train, z_train = load_split("train")
    x_val, z_val = load_split("val")
    x_test, z_test = load_split("test")
    target_mean = z_train.mean(dim=0)
    target_std = z_train.std(dim=0).clamp_min(1e-4)
    y_train = (z_train - target_mean) / target_std
    y_val = (z_val - target_mean) / target_std
    y_test = (z_test - target_mean) / target_std
    torch.manual_seed(args.seed + rank * 101)
    model = LowRankLQ(z_train.shape[1], rank, x_train.shape[1]).to(args.device)
    gradient_path = bank / "train_response_gradient_dct.npy"
    if gradient_path.exists():
        gradient_basis, gradient_spectrum = active_gradient_initialization(
            gradient_path, feature_std, rank, args.device
        )
        with torch.no_grad():
            model.weights.copy_(gradient_basis)
            initialize_decoder(model, x_train, y_train)
        energy = gradient_spectrum.square()
        np.savez(
            args.output_dir / f"rank{rank}_active_gradient_spectrum.npz",
            singular_values=gradient_spectrum.cpu().numpy(),
            cumulative_energy_fraction=(energy.cumsum(1) / energy.sum(1, keepdim=True)).cpu().numpy(),
        )
        print(f"rank {rank}: initialized from exact response-active gradients", flush=True)
    else:
        initialize_from_response(
            model,
            x_train,
            y_train,
            seed=args.seed + rank * 313,
            iterations=args.ste_iterations,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    penalty = dct_frequency_penalty().to(args.device)
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        initial_prediction = model(x_val).float()
        best_loss = float(F.mse_loss(initial_prediction, y_val))
    best_epoch, best_state = -1, copy.deepcopy(model.state_dict())
    generator = torch.Generator(device=args.device).manual_seed(args.seed + rank * 1009)
    history = [{
        "rank": rank,
        "epoch": -1,
        "train_normalized_mse": np.nan,
        "val_normalized_mse": best_loss,
    }]
    print(f"rank {rank} STE/LQ initialization: val {best_loss:.4f}", flush=True)
    started_fit = time.time()
    for epoch in range(args.epochs):
        permutation = torch.randperm(len(x_train), generator=generator, device=args.device)
        model.train()
        train_loss = 0.0
        for start in range(0, len(x_train), args.fit_batch_size):
            ids = permutation[start : start + args.fit_batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(x_train[ids])
                mse = F.mse_loss(prediction.float(), y_train[ids])
                loss = (
                    mse
                    + args.smoothness * model.smoothness_loss(penalty)
                    + 0.01 * model.orthogonality_loss()
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += float(mse.detach()) * len(ids)
        model.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            val_prediction = model(x_val).float()
            val_loss = float(F.mse_loss(val_prediction, y_val))
        train_loss /= len(x_train)
        history.append({"rank": rank, "epoch": epoch, "train_normalized_mse": train_loss, "val_normalized_mse": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"rank {rank} epoch {epoch + 1}: train {train_loss:.4f}, val {val_loss:.4f}", flush=True)
        if epoch - best_epoch >= 18:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.retract()
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction_norm = model(x_test).float()
    z_prediction = prediction_norm * target_std + target_mean
    rate_target = F.softplus(z_test)
    rate_prediction = F.softplus(z_prediction)
    z_r2 = r2_score(z_test.cpu().numpy(), z_prediction.cpu().numpy(), axis=0)
    rate_r2 = r2_score(rate_target.cpu().numpy(), rate_prediction.cpu().numpy(), axis=0)
    correlation = np.asarray([
        np.corrcoef(rate_target[:, unit].cpu(), rate_prediction[:, unit].cpu())[0, 1]
        for unit in range(z_test.shape[1])
    ])
    metrics = selection[["subset_index", "unit_index", "response_subspace_group", "sf_split_metric", "prior_orientation_selectivity_index"]].copy()
    metrics["rank"] = rank
    metrics["test_z_r2"] = z_r2
    metrics["test_rate_r2"] = rate_r2
    metrics["test_rate_correlation"] = correlation
    metrics["best_epoch"] = best_epoch
    metrics["best_val_normalized_mse"] = best_loss
    metrics["fit_seconds"] = time.time() - started_fit
    state = ResponseSubspaceState(
        rank=rank,
        weights=model.weights.detach().cpu(),
        linear=model.linear.detach().cpu(),
        quadratic=model.quadratic.detach().cpu(),
        bias=model.bias.detach().cpu(),
        target_mean=target_mean.detach().cpu(),
        target_std=target_std.detach().cpu(),
        feature_mean=feature_mean.detach().cpu(),
        feature_std=feature_std.detach().cpu(),
    )
    torch.save(state, args.output_dir / f"rank{rank}_state.pt")
    pd.DataFrame(history).to_csv(args.output_dir / f"rank{rank}_training_history.csv", index=False)
    return state, metrics


def full_map_validation(args: argparse.Namespace, state: ResponseSubspaceState, selection: pd.DataFrame) -> pd.DataFrame:
    target = np.load(args.output_dir / "bank/test_z_full.npy", mmap_mode="r")
    predictions = []
    for start in range(0, args.n_test_movies, args.movie_batch_size):
        stop = min(start + args.movie_batch_size, args.n_test_movies)
        movie, _ = generate_movie_batch(
            stop - start,
            seed=args.seed + 2_000_000 + start,
            device=args.device,
        )
        with torch.no_grad():
            z, _ = convolutional_predict(movie, state)
        predictions.append(z.cpu().numpy())
    prediction = np.concatenate(predictions)
    rows = []
    for subset_index, unit_index in enumerate(selection.unit_index.to_numpy(int)):
        z_target = np.asarray(target[:, subset_index], dtype=np.float32).ravel()
        z_prediction = prediction[:, subset_index].ravel()
        rate_target = np.logaddexp(0.0, z_target)
        rate_prediction = np.logaddexp(0.0, z_prediction)
        rows.append({
            "subset_index": subset_index,
            "unit_index": unit_index,
            "rank": state.rank,
            "full_map_z_r2": float(r2_score(z_target, z_prediction, axis=0)),
            "full_map_rate_r2": float(r2_score(rate_target, rate_prediction, axis=0)),
            "full_map_rate_correlation": float(np.corrcoef(rate_target, rate_prediction)[0, 1]),
        })
    return pd.DataFrame(rows)


def render(args: argparse.Namespace, metrics: pd.DataFrame, states: dict[int, ResponseSubspaceState], selection: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    for unit, part in metrics.groupby("unit_index"):
        color = "#2676b8" if part.response_subspace_group.iloc[0] == "lower-SF" else "#d85832"
        axes[0].plot(part["rank"], part.test_rate_r2, color=color, alpha=0.45, lw=1)
    summary = metrics.groupby(["response_subspace_group", "rank"]).test_rate_r2.agg(["median", "min", "max"]).reset_index()
    for group, part in summary.groupby("response_subspace_group"):
        color = "#2676b8" if group == "lower-SF" else "#d85832"
        axes[0].plot(part["rank"], part["median"], "o-", color=color, lw=2.5, label=group)
    axes[0].axhline(0.8, ls="--", color="0.45", lw=1)
    axes[0].set(xlabel="private subspace rank", ylabel="held-out response $R^2$", xticks=sorted(states))
    axes[0].legend(frameon=False)
    pivot = metrics.pivot(index="unit_index", columns="rank", values="test_rate_r2").loc[selection.unit_index]
    image = axes[1].imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axes[1].set(xlabel="rank", ylabel="predeclared RR100 unit", xticks=np.arange(len(pivot.columns)), xticklabels=pivot.columns)
    axes[1].set_yticks(np.arange(len(pivot)), labels=[f"u{x:03d}" for x in pivot.index])
    fig.colorbar(image, ax=axes[1], label="held-out response $R^2$", fraction=0.045)
    fig.tight_layout()
    fig.savefig(args.output_dir / "rank_response_fidelity.png", dpi=220)
    fig.savefig(args.output_dir / "rank_response_fidelity.pdf")
    plt.close(fig)

    atlas_rank = max([rank for rank in states if rank <= 8], default=min(states))
    state = states[atlas_rank]
    physical = state.weights / state.feature_std[None, None]
    filters = dct_filters_to_movies(physical).numpy()
    fig, axes = plt.subplots(len(selection), atlas_rank, figsize=(1.55 * atlas_rank, 1.35 * len(selection)), squeeze=False)
    for unit in range(len(selection)):
        for mode in range(atlas_rank):
            movie = filters[unit, mode]
            lag = int(np.argmax(np.square(movie).sum(axis=(1, 2))))
            value = movie[lag]
            limit = np.percentile(np.abs(value), 99)
            axes[unit, mode].imshow(value, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
            axes[unit, mode].set_xticks([]); axes[unit, mode].set_yticks([])
            if unit == 0:
                axes[unit, mode].set_title(f"mode {mode + 1}")
            if mode == 0:
                axes[unit, mode].set_ylabel(f"u{int(selection.unit_index.iloc[unit]):03d}\nlag {lag}")
    fig.suptitle(f"Rank-{atlas_rank} response-predictive filters (peak-energy lag)", y=0.995)
    fig.tight_layout()
    fig.savefig(args.output_dir / f"rank{atlas_rank}_filter_atlas.png", dpi=220)
    fig.savefig(args.output_dir / f"rank{atlas_rank}_filter_atlas.pdf")
    plt.close(fig)


def main() -> int:
    global scorer, readout, started
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    table = pd.read_csv(UNIT_TABLE)
    selection = deterministic_unit_subset(table)
    selection.to_csv(args.output_dir / "unit_selection.csv", index=False)
    print("selected units:", selection.unit_index.tolist(), flush=True)
    scorer, readout = load_teacher(args.device)
    make_bank(args, selection)
    measure_response_gradients(args, selection, scorer, readout)
    del readout, scorer
    torch.cuda.empty_cache()
    make_dense_design(args)
    ranks = [int(value) for value in args.ranks.split(",") if value.strip()]
    states, metric_rows, full_rows = {}, [], []
    for rank in ranks:
        state, metric = fit_rank(args, rank, selection)
        states[rank] = state
        metric_rows.append(metric)
        full_rows.append(full_map_validation(args, state, selection))
        print(
            f"rank {rank}: median sampled-map rate R2={metric.test_rate_r2.median():.3f}; "
            f"full-map={full_rows[-1].full_map_rate_r2.median():.3f}",
            flush=True,
        )
    metrics = pd.concat(metric_rows, ignore_index=True)
    full = pd.concat(full_rows, ignore_index=True)
    metrics = metrics.merge(full, on=["subset_index", "unit_index", "rank"], how="left")
    metrics.to_csv(args.output_dir / "heldout_response_fidelity.csv", index=False)
    render(args, metrics, states, selection)
    summary = metrics.groupby("rank").agg(
        median_test_rate_r2=("test_rate_r2", "median"),
        min_test_rate_r2=("test_rate_r2", "min"),
        median_full_map_rate_r2=("full_map_rate_r2", "median"),
        min_full_map_rate_r2=("full_map_rate_r2", "min"),
    ).reset_index()
    summary.to_csv(args.output_dir / "rank_summary.csv", index=False)
    save_json(args.output_dir / "run_summary.json", {
        "elapsed_seconds": time.time() - started,
        "rank_summary": summary.to_dict(orient="records"),
        "success_gate": "median full-map held-out response R2 >= 0.8",
        "gate_passed_ranks": summary.loc[summary.median_full_map_rate_r2.ge(0.8), "rank"].astype(int).tolist(),
    })
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
