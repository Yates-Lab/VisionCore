#!/usr/bin/env python3
"""Jointly distill the RR100 twin into filters plus a response nonlinearity.

Unlike the frozen active-subspace pilot, this script optimizes the shared
spatiotemporal filters for held-out firing-rate prediction.  The resulting
model is exported as the same standalone ``ReducedTwin`` used for movie-level
and FEM counterfactuals.
"""

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
import torch.nn as nn
import torch.nn.functional as F


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
from paper.fig4.nonlinear_phase_causal.run_reduced_twin_pilot import (
    SOURCE,
    architecture_name,
    parse_architectures,
    shared_active_basis,
)


DEFAULT_OUT = ROOT / "outputs/figures/fig4/nonlinear_phase_causal_v1/distilled_twin_pilot"


class DistillationModel(nn.Module):
    """Trainable shared linear front end and population response function."""

    def __init__(
        self,
        initial_basis: torch.Tensor,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
        mean_rate: torch.Tensor,
    ):
        super().__init__()
        self.basis = nn.Parameter(initial_basis.clone().float())
        self.generator_norm = nn.BatchNorm1d(
            initial_basis.shape[0], affine=False, momentum=0.05
        )
        self.decoder = PopulationNonlinearity(
            initial_basis.shape[0], output_dim, hidden_dims, dropout
        )
        with torch.no_grad():
            output_layer = self.decoder.network[-1]
            output_layer.weight.normal_(mean=0.0, std=0.01)
            output_layer.bias.copy_(torch.log(torch.expm1(mean_rate.clamp_min(1e-6))))

    def forward(self, standardized_features: torch.Tensor) -> torch.Tensor:
        generators = F.linear(standardized_features, self.basis)
        return self.decoder(self.generator_norm(generators))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ranks", default="16,32,64,128")
    parser.add_argument("--architectures", default="64,32;128,64")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--basis-learning-rate", type=float, default=0.0002)
    parser.add_argument("--decoder-learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--movie-batch-size", type=int, default=8)
    parser.add_argument("--teacher-seed", type=int, default=20260813)
    return parser.parse_args()


def load_standardized_data(source, feature_mean, feature_std, device):
    splits = {}
    for split in ("train", "val", "test"):
        features = torch.from_numpy(
            np.asarray(np.load(source / f"bank/{split}_dense_features.npy"))
        ).to(device)
        target_z = torch.from_numpy(
            np.asarray(np.load(source / f"bank/{split}_dense_z.npy"))
        ).to(device).float()
        standardized = ((features.float() - feature_mean) / feature_std).half()
        splits[split] = (standardized, F.softplus(target_z))
        del features, target_z
    return splits


def fit_model(
    data,
    initial_basis,
    hidden,
    args,
):
    x_train, y_train = data["train"]
    x_val, y_val = data["val"]
    target_std = y_train.std(0).clamp_min(1e-5)
    torch.manual_seed(args.seed)
    model = DistillationModel(
        initial_basis,
        y_train.shape[1],
        hidden,
        args.dropout,
        y_train.mean(0),
    ).to(args.device)
    decoder_parameters = list(model.generator_norm.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": [model.basis], "lr": args.basis_learning_rate, "weight_decay": 1e-5},
            {"params": decoder_parameters, "lr": args.decoder_learning_rate, "weight_decay": 1e-4},
        ]
    )
    rng = torch.Generator(device=args.device).manual_seed(args.seed + 1000)
    best_loss, best_state, best_epoch = math.inf, None, -1
    history = []
    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(len(x_train), generator=rng, device=args.device)
        train_loss = 0.0
        for start in range(0, len(x_train), args.batch_size):
            ids = permutation[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(x_train[ids])
                loss = normalized_mse(prediction.float(), y_train[ids], target_std)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += float(loss.detach()) * len(ids)
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(x_val), args.batch_size):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    predictions.append(model(x_val[start : start + args.batch_size]).float())
        val_loss = float(normalized_mse(torch.cat(predictions), y_val, target_std))
        history.append(
            {
                "epoch": epoch,
                "train_normalized_mse": train_loss / len(x_train),
                "val_normalized_mse": val_loss,
            }
        )
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= args.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, target_std, best_loss, best_epoch, pd.DataFrame(history)


@torch.no_grad()
def sampled_metrics(model, data, selection, batch_size):
    features, target = data["test"]
    prediction = []
    for start in range(0, len(features), batch_size):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction.append(model(features[start : start + batch_size]).float())
    prediction = torch.cat(prediction).cpu().numpy()
    target = target.cpu().numpy()
    rows = []
    for subset_index, unit_index in enumerate(selection.unit_index.to_numpy(int)):
        true, pred = target[:, subset_index], prediction[:, subset_index]
        rows.append(
            {
                "subset_index": subset_index,
                "unit_index": unit_index,
                "sampled_rate_r2": float(r2_score(true, pred, axis=0)),
                "sampled_rate_correlation": float(np.corrcoef(true, pred)[0, 1]),
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def full_map_metrics(args, reduced, selection):
    target_z = np.load(args.source_dir / "bank/test_z_full.npy", mmap_mode="r")
    predictions = []
    for start in range(0, len(target_z), args.movie_batch_size):
        stop = min(start + args.movie_batch_size, len(target_z))
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
        true, pred = target[:, subset_index].ravel(), prediction[:, subset_index].ravel()
        rows.append(
            {
                "subset_index": subset_index,
                "unit_index": unit_index,
                "full_map_rate_r2": float(r2_score(true, pred, axis=0)),
                "full_map_rate_correlation": float(np.corrcoef(true, pred)[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def render(output_dir, summary, metrics):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for name, part in summary.groupby("architecture"):
        axes[0].plot(part["rank"], part.median_full_map_rate_r2, "o-", label=name)
    axes[0].axhline(0.8, color="0.4", ls="--", lw=1)
    axes[0].set(xlabel="jointly learned filter rank", ylabel="median held-out response $R^2$", xscale="log")
    axes[0].legend(frameon=False)
    best = summary.sort_values(["median_full_map_rate_r2", "parameter_count"], ascending=[False, True]).iloc[0]
    chosen = metrics.loc[(metrics["rank"] == best["rank"]) & metrics.architecture.eq(best.architecture)]
    axes[1].bar(np.arange(len(chosen)), chosen.full_map_rate_r2, color="#3178a8")
    axes[1].axhline(0.8, color="0.4", ls="--", lw=1)
    axes[1].set(
        xlabel="predeclared RR100 unit",
        ylabel="held-out full-map response $R^2$",
        xticks=np.arange(len(chosen)),
        xticklabels=[f"u{x:03d}" for x in chosen.unit_index],
        ylim=(-0.05, 1),
    )
    axes[1].tick_params(axis="x", rotation=55)
    fig.tight_layout()
    fig.savefig(output_dir / "distilled_twin_response_fidelity.png", dpi=220)
    fig.savefig(output_dir / "distilled_twin_response_fidelity.pdf")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    selection = pd.read_csv(args.source_dir / "unit_selection.csv")
    scaling = np.load(args.source_dir / "bank/feature_scaling.npz")
    feature_mean = torch.from_numpy(scaling["mean"]).to(args.device).float()
    feature_std = torch.from_numpy(scaling["std"]).to(args.device).float()
    ranks = [int(value) for value in args.ranks.split(",") if value.strip()]
    architectures = parse_architectures(args.architectures)
    initial_basis, singular, cumulative = shared_active_basis(
        args.source_dir, feature_std, max(ranks), args.device
    )
    data = load_standardized_data(args.source_dir, feature_mean, feature_std, args.device)
    model_rows, metric_rows = [], []
    for rank in ranks:
        for hidden in architectures:
            name = architecture_name(hidden)
            model, _, val_loss, best_epoch, history = fit_model(
                data, initial_basis[:rank], hidden, args
            )
            norm = model.generator_norm
            state = ReducedTwinState(
                rank=rank,
                unit_indices=torch.from_numpy(selection.unit_index.to_numpy(int)),
                basis=model.basis.detach().cpu(),
                feature_mean=feature_mean.detach().cpu(),
                feature_std=feature_std.detach().cpu(),
                generator_mean=norm.running_mean.detach().cpu(),
                generator_std=torch.sqrt(norm.running_var.detach().cpu() + norm.eps),
                hidden_dims=hidden,
                dropout=args.dropout,
                decoder_state={key: value.detach().cpu() for key, value in model.decoder.state_dict().items()},
            )
            reduced = ReducedTwin(state).to(args.device).eval()
            sampled = sampled_metrics(model, data, selection, args.batch_size)
            full = full_map_metrics(args, reduced, selection)
            metrics = sampled.merge(full, on=["subset_index", "unit_index"])
            metrics["rank"] = rank
            metrics["architecture"] = name
            metric_rows.append(metrics)
            parameter_count = rank * model.basis.shape[1] + sum(p.numel() for p in model.decoder.parameters())
            row = {
                "rank": rank,
                "architecture": name,
                "best_epoch": best_epoch,
                "val_normalized_mse": val_loss,
                "parameter_count": parameter_count,
                "median_sampled_rate_r2": metrics.sampled_rate_r2.median(),
                "min_sampled_rate_r2": metrics.sampled_rate_r2.min(),
                "median_full_map_rate_r2": metrics.full_map_rate_r2.median(),
                "min_full_map_rate_r2": metrics.full_map_rate_r2.min(),
            }
            model_rows.append(row)
            torch.save(state, args.output_dir / f"distilled_twin_rank{rank}_{name}.pt")
            history.to_csv(args.output_dir / f"training_rank{rank}_{name}.csv", index=False)
            print(
                f"rank {rank:3d} {name:12s}: median full-map R2="
                f"{row['median_full_map_rate_r2']:.3f}, min={row['min_full_map_rate_r2']:.3f}",
                flush=True,
            )
    summary = pd.DataFrame(model_rows).sort_values(["rank", "parameter_count"])
    metrics = pd.concat(metric_rows, ignore_index=True)
    summary.to_csv(args.output_dir / "model_selection.csv", index=False)
    metrics.to_csv(args.output_dir / "unit_response_fidelity.csv", index=False)
    render(args.output_dir, summary, metrics)
    passing = summary.loc[summary.median_full_map_rate_r2.ge(0.8)].sort_values("parameter_count")
    selected = None if passing.empty else passing.iloc[0].to_dict()
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "analysis": "joint response-distillation into spatiotemporal filters plus learned nonlinearity",
                "selection_rule": "smallest model with median held-out full-map response R2 >= 0.8",
                "selected_model": selected,
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        )
        + "\n"
    )
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
