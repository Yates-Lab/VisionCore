#!/usr/bin/env python3
"""Fit minimal shared-subspace models that can replace the RR100 twin."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.nonlinear_phase_causal.reduced_twin import (
    PopulationNonlinearity,
    ReducedTwin,
    ReducedTwinState,
    normalized_mse,
)
from paper.fig4.nonlinear_phase_causal.response_subspace import generate_movie_batch, r2_score


SOURCE = ROOT / "outputs/figures/fig4/nonlinear_phase_causal_v1/response_subspace_pilot"
DEFAULT_OUT = ROOT / "outputs/figures/fig4/nonlinear_phase_causal_v1/reduced_twin_pilot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ranks", default="8,16,32,64,128")
    parser.add_argument("--architectures", default="linear;32;64,32;128,64")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seeds", default="17,29")
    parser.add_argument("--movie-batch-size", type=int, default=8)
    parser.add_argument("--teacher-seed", type=int, default=20260813)
    return parser.parse_args()


def parse_architectures(value: str) -> list[tuple[int, ...]]:
    result = []
    for item in value.split(";"):
        item = item.strip()
        result.append(()) if item in ("", "linear") else result.append(tuple(int(x) for x in item.split(",")))
    return result


def architecture_name(hidden: tuple[int, ...]) -> str:
    return "linear" if not hidden else "mlp_" + "x".join(str(value) for value in hidden)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


@torch.no_grad()
def shared_active_basis(source: Path, feature_std: torch.Tensor, max_rank: int, device: str):
    gradient = torch.from_numpy(
        np.asarray(np.load(source / "bank/train_response_gradient_dct.npy"))
    ).to(device).float()
    gradient = gradient * feature_std[None, None]
    matrix = gradient.reshape(-1, gradient.shape[-1])
    _, singular, basis = torch.pca_lowrank(
        matrix,
        q=int(max_rank),
        center=False,
        niter=6,
    )
    energy = singular.square()
    return basis.T.contiguous(), singular, energy.cumsum(0) / energy.sum()


def load_projected_data(
    source: Path,
    basis: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    device: str,
):
    splits = {}
    for split in ("train", "val", "test"):
        features = torch.from_numpy(
            np.asarray(np.load(source / f"bank/{split}_dense_features.npy"))
        ).to(device)
        target_z = torch.from_numpy(
            np.asarray(np.load(source / f"bank/{split}_dense_z.npy"))
        ).to(device).float()
        generators = []
        for start in range(0, len(features), 2048):
            standardized = (features[start : start + 2048].float() - feature_mean) / feature_std
            generators.append(standardized @ basis.T)
        splits[split] = (torch.cat(generators), torch.nn.functional.softplus(target_z))
    return splits


def fit_decoder(
    generators: dict[str, torch.Tensor],
    rank: int,
    hidden: tuple[int, ...],
    *,
    dropout: float,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    device: str,
):
    x_train, y_train = generators["train"]
    x_val, y_val = generators["val"]
    generator_mean = x_train[:, :rank].mean(0)
    generator_std = x_train[:, :rank].std(0).clamp_min(1e-5)
    target_std = y_train.std(0).clamp_min(1e-5)

    def normalize(x):
        return (x[:, :rank] - generator_mean) / generator_std

    torch.manual_seed(int(seed))
    model = PopulationNonlinearity(
        rank,
        y_train.shape[1],
        hidden_dims=hidden,
        dropout=dropout if hidden else 0.0,
    ).to(device)
    # Start at each unit's mean firing rate.  The RR100 rates are well below
    # softplus(0), so the default zero-centered initialization otherwise spends
    # much of optimization merely learning the baseline.
    with torch.no_grad():
        output_layer = model.network[-1]
        output_layer.weight.normal_(mean=0.0, std=0.01)
        mean_rate = y_train.mean(0).clamp_min(1e-6)
        output_layer.bias.copy_(torch.log(torch.expm1(mean_rate)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_loss, best_state, best_epoch = math.inf, None, -1
    history = []
    rng = torch.Generator(device=device).manual_seed(seed + 1000)
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(x_train), generator=rng, device=device)
        train_loss = 0.0
        for start in range(0, len(x_train), batch_size):
            ids = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(normalize(x_train[ids]))
                loss = normalized_mse(prediction.float(), y_train[ids], target_std)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += float(loss.detach()) * len(ids)
        model.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            val_prediction = model(normalize(x_val)).float()
        val_loss = float(normalized_mse(val_prediction, y_val, target_std))
        history.append({"epoch": epoch, "train_normalized_mse": train_loss / len(x_train), "val_normalized_mse": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, generator_mean, generator_std, target_std, best_loss, best_epoch, pd.DataFrame(history)


@torch.no_grad()
def sampled_metrics(model, generators, generator_mean, generator_std, rank, selection):
    x_test, target = generators["test"]
    normalized = (x_test[:, :rank] - generator_mean) / generator_std
    prediction = model(normalized).float()
    target_np, prediction_np = target.cpu().numpy(), prediction.cpu().numpy()
    rows = []
    for subset_index, unit_index in enumerate(selection.unit_index.to_numpy(int)):
        rows.append({
            "subset_index": subset_index,
            "unit_index": unit_index,
            "sampled_rate_r2": float(r2_score(target_np[:, subset_index], prediction_np[:, subset_index], axis=0)),
            "sampled_rate_correlation": float(np.corrcoef(target_np[:, subset_index], prediction_np[:, subset_index])[0, 1]),
            "sampled_normalized_rmse": float(np.sqrt(np.mean(((target_np[:, subset_index] - prediction_np[:, subset_index]) / max(target_np[:, subset_index].std(), 1e-8)) ** 2))),
        })
    return pd.DataFrame(rows)


@torch.no_grad()
def full_map_metrics(args, reduced: ReducedTwin, selection: pd.DataFrame):
    target_z = np.load(args.source_dir / "bank/test_z_full.npy", mmap_mode="r")
    predictions = []
    n_movies = target_z.shape[0]
    for start in range(0, n_movies, args.movie_batch_size):
        stop = min(start + args.movie_batch_size, n_movies)
        movie, _ = generate_movie_batch(
            stop - start,
            seed=args.teacher_seed + 2_000_000 + start,
            device=args.device,
        )
        predictions.append(reduced(movie).cpu().numpy())
    prediction = np.concatenate(predictions)
    target = np.logaddexp(0.0, np.asarray(target_z, dtype=np.float32))
    rows = []
    for subset_index, unit_index in enumerate(selection.unit_index.to_numpy(int)):
        true = target[:, subset_index].ravel()
        pred = prediction[:, subset_index].ravel()
        rows.append({
            "subset_index": subset_index,
            "unit_index": unit_index,
            "full_map_rate_r2": float(r2_score(true, pred, axis=0)),
            "full_map_rate_correlation": float(np.corrcoef(true, pred)[0, 1]),
            "full_map_normalized_rmse": float(np.sqrt(np.mean(((true - pred) / max(true.std(), 1e-8)) ** 2))),
        })
    return pd.DataFrame(rows)


def render(output_dir: Path, summary: pd.DataFrame, metrics: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3))
    for name, part in summary.groupby("architecture"):
        axes[0].plot(part["rank"], part.median_full_map_rate_r2, "o-", label=name)
    axes[0].axhline(0.8, color="0.4", ls="--", lw=1)
    axes[0].set(xlabel="shared subspace rank", ylabel="median held-out full-map response $R^2$", xscale="log", xticks=sorted(summary["rank"].unique()))
    axes[0].set_xticklabels(sorted(summary["rank"].unique()))
    axes[0].legend(frameon=False, fontsize=7)
    best = summary.sort_values(["median_full_map_rate_r2", "parameter_count"], ascending=[False, True]).iloc[0]
    chosen = metrics.loc[(metrics["rank"] == best["rank"]) & metrics.architecture.eq(best.architecture)]
    axes[1].bar(np.arange(len(chosen)), chosen.full_map_rate_r2, color="#3178a8")
    axes[1].axhline(0.8, color="0.4", ls="--", lw=1)
    axes[1].set(xlabel="predeclared RR100 unit", ylabel="held-out full-map response $R^2$", xticks=np.arange(len(chosen)), xticklabels=[f"u{x:03d}" for x in chosen.unit_index], ylim=(-0.05, 1))
    axes[1].tick_params(axis="x", rotation=55)
    axes[1].set_title(f"best: rank {int(best['rank'])}, {best.architecture}")
    fig.tight_layout()
    fig.savefig(output_dir / "reduced_twin_response_fidelity.png", dpi=220)
    fig.savefig(output_dir / "reduced_twin_response_fidelity.pdf")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    selection = pd.read_csv(args.source_dir / "unit_selection.csv")
    scaling = np.load(args.source_dir / "bank/feature_scaling.npz")
    feature_mean = torch.from_numpy(scaling["mean"]).to(args.device)
    feature_std = torch.from_numpy(scaling["std"]).to(args.device)
    ranks = [int(value) for value in args.ranks.split(",") if value.strip()]
    architectures = parse_architectures(args.architectures)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    basis, singular, cumulative = shared_active_basis(
        args.source_dir, feature_std, max(ranks), args.device
    )
    torch.save({"basis": basis.cpu(), "singular_values": singular.cpu(), "cumulative_energy": cumulative.cpu()}, args.output_dir / "shared_active_basis.pt")
    generators = load_projected_data(
        args.source_dir, basis, feature_mean, feature_std, args.device
    )

    all_metrics, model_rows = [], []
    for rank in ranks:
        for hidden in architectures:
            name = architecture_name(hidden)
            candidates = []
            for seed in seeds:
                fitted = fit_decoder(
                    generators, rank, hidden,
                    dropout=args.dropout,
                    seed=seed,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    device=args.device,
                )
                candidates.append((fitted[4], seed, fitted))
            _, best_seed, fitted = min(candidates, key=lambda item: item[0])
            decoder, generator_mean, generator_std, target_std, val_loss, best_epoch, history = fitted
            state = ReducedTwinState(
                rank=rank,
                unit_indices=torch.from_numpy(selection.unit_index.to_numpy(int)),
                basis=basis[:rank].detach().cpu(),
                feature_mean=feature_mean.detach().cpu(),
                feature_std=feature_std.detach().cpu(),
                generator_mean=generator_mean.detach().cpu(),
                generator_std=generator_std.detach().cpu(),
                hidden_dims=hidden,
                dropout=args.dropout if hidden else 0.0,
                decoder_state={key: value.detach().cpu() for key, value in decoder.state_dict().items()},
            )
            reduced = ReducedTwin(state).to(args.device).eval()
            sampled = sampled_metrics(decoder, generators, generator_mean, generator_std, rank, selection)
            full = full_map_metrics(args, reduced, selection)
            metric = sampled.merge(full, on=["subset_index", "unit_index"])
            metric["rank"] = rank
            metric["architecture"] = name
            metric["seed"] = best_seed
            metric["best_epoch"] = best_epoch
            metric["val_normalized_mse"] = val_loss
            all_metrics.append(metric)
            parameter_count = sum(value.numel() for value in reduced.parameters()) + rank * basis.shape[1]
            model_rows.append({
                "rank": rank,
                "architecture": name,
                "seed": best_seed,
                "best_epoch": best_epoch,
                "val_normalized_mse": val_loss,
                "parameter_count": parameter_count,
                "median_sampled_rate_r2": metric.sampled_rate_r2.median(),
                "min_sampled_rate_r2": metric.sampled_rate_r2.min(),
                "median_full_map_rate_r2": metric.full_map_rate_r2.median(),
                "min_full_map_rate_r2": metric.full_map_rate_r2.min(),
            })
            torch.save(state, args.output_dir / f"reduced_twin_rank{rank}_{name}.pt")
            history.to_csv(args.output_dir / f"training_rank{rank}_{name}.csv", index=False)
            print(f"rank {rank:3d} {name:12s}: median full-map R2={metric.full_map_rate_r2.median():.3f}, min={metric.full_map_rate_r2.min():.3f}", flush=True)

    metrics = pd.concat(all_metrics, ignore_index=True)
    summary = pd.DataFrame(model_rows).sort_values(["rank", "parameter_count"])
    metrics.to_csv(args.output_dir / "unit_response_fidelity.csv", index=False)
    summary.to_csv(args.output_dir / "model_selection.csv", index=False)
    render(args.output_dir, summary, metrics)
    passing = summary.loc[summary.median_full_map_rate_r2.ge(0.8)].sort_values("parameter_count")
    selected = None if passing.empty else passing.iloc[0].to_dict()
    save_json(args.output_dir / "run_summary.json", {
        "analysis": "deployable RR100 shared-subspace reduced twin with learned population nonlinearity",
        "source_method": "Fixational Transients shared IRF subspace + normalized flexible response nonlinearity",
        "ranks": ranks,
        "architectures": [architecture_name(value) for value in architectures],
        "selection_rule": "smallest model with median held-out full-map response R2 >= 0.8",
        "selected_model": selected,
        "elapsed_seconds": time.time() - started,
    })
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
