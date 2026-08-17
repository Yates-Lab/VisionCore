#!/usr/bin/env python3
"""Supervised linear bottleneck plus MLP distillation of RR100 responses."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.nonlinear_phase_causal.reduced_twin import (
    PopulationNonlinearity,
    ReducedTwin,
    ReducedTwinState,
    normalized_mse,
)
from paper.fig4.nonlinear_phase_causal.response_subspace import r2_score
from paper.fig4.nonlinear_phase_causal.run_reduced_twin_pilot import (
    architecture_name,
    parse_architectures,
)


SOURCE_STATE = ROOT / (
    "outputs/figures/fig4/nonlinear_phase_causal_v1/reduced_twin_full_frequency/"
    "reduced_twin_rank128_mlp_128x64.pt"
)
SOURCE_BANK = ROOT / (
    "outputs/figures/fig4/nonlinear_phase_causal_v1/"
    "reduced_twin_natural_calibration_all_scales/bank"
)
DEFAULT_OUT = ROOT / "outputs/figures/fig4/nonlinear_phase_causal_v1/linear_mlp_reduction"
SELECTION = ROOT / (
    "outputs/figures/fig4/nonlinear_phase_causal_v1/response_subspace_pilot/"
    "unit_selection.csv"
)


class LinearMLP(nn.Module):
    """Learned response-predictive linear encoder followed by an MLP."""

    def __init__(self, input_dim, rank, output_dim, hidden, dropout, mean_rate):
        super().__init__()
        self.encoder = nn.Linear(input_dim, rank, bias=False)
        self.generator_norm = nn.BatchNorm1d(rank, affine=False, momentum=0.05)
        self.decoder = PopulationNonlinearity(rank, output_dim, hidden, dropout)
        with torch.no_grad():
            self.encoder.weight.zero_()
            diagonal = min(input_dim, rank)
            self.encoder.weight[:diagonal, :diagonal].copy_(torch.eye(diagonal))
            self.encoder.weight.add_(0.005 * torch.randn_like(self.encoder.weight))
            output = self.decoder.network[-1]
            output.weight.normal_(mean=0.0, std=0.01)
            output.bias.copy_(torch.log(torch.expm1(mean_rate.clamp_min(1e-6))))

    def forward(self, normalized_source_generators):
        encoded = self.encoder(normalized_source_generators)
        return self.decoder(self.generator_norm(encoded))

    def orthogonality_loss(self):
        weight = self.encoder.weight
        normalized = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
        gram = normalized @ normalized.T
        identity = torch.eye(len(weight), device=weight.device, dtype=weight.dtype)
        return (gram - identity).square().mean()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, default=SOURCE_STATE)
    parser.add_argument("--source-bank", type=Path, default=SOURCE_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ranks", default="4,8,16,32,64")
    parser.add_argument("--architectures", default="64,32")
    parser.add_argument("--seeds", default="17,29")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--encoder-learning-rate", type=float, default=0.0003)
    parser.add_argument("--decoder-learning-rate", type=float, default=0.001)
    parser.add_argument("--orthogonality", type=float, default=0.001)
    return parser.parse_args()


def load_data(path, device):
    raw = {}
    for split in ("train", "val", "test"):
        x = torch.from_numpy(np.asarray(np.load(path / f"{split}_generators.npy"))).to(device).float()
        y = torch.from_numpy(np.asarray(np.load(path / f"{split}_rates.npy"))).to(device).float()
        raw[split] = (x, y)
    mean = raw["train"][0].mean(0)
    std = raw["train"][0].std(0).clamp_min(1e-5)
    data = {
        split: (((x - mean) / std).half(), y)
        for split, (x, y) in raw.items()
    }
    return raw, data, mean, std


def fit(data, rank, hidden, seed, args):
    x_train, y_train = data["train"]
    x_val, y_val = data["val"]
    target_std = y_train.std(0).clamp_min(1e-5)
    torch.manual_seed(seed)
    model = LinearMLP(
        x_train.shape[1], rank, y_train.shape[1], hidden, args.dropout, y_train.mean(0)
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_learning_rate, "weight_decay": 1e-5},
            {"params": model.decoder.parameters(), "lr": args.decoder_learning_rate, "weight_decay": 1e-4},
        ]
    )
    rng = torch.Generator(device=args.device).manual_seed(seed + 1000)
    best_loss, best_epoch, best_state = math.inf, -1, None
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
                response_loss = normalized_mse(prediction.float(), y_train[ids], target_std)
                loss = response_loss + args.orthogonality * model.orthogonality_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += float(response_loss.detach()) * len(ids)
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(x_val), args.batch_size):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    predictions.append(model(x_val[start : start + args.batch_size]).float())
        val_loss = float(normalized_mse(torch.cat(predictions), y_val, target_std))
        history.append(
            {"epoch": epoch, "train_normalized_mse": train_loss / len(x_train), "val_normalized_mse": val_loss}
        )
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= args.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, best_loss, best_epoch, pd.DataFrame(history)


@torch.no_grad()
def test_metrics(model, data, selection, batch_size):
    x, target = data["test"]
    prediction = []
    for start in range(0, len(x), batch_size):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction.append(model(x[start : start + batch_size]).float())
    prediction = torch.cat(prediction).cpu().numpy()
    target = target.cpu().numpy()
    rows = []
    for subset_index, unit_index in enumerate(selection.unit_index.to_numpy(int)):
        true, pred = target[:, subset_index], prediction[:, subset_index]
        rows.append(
            {
                "subset_index": subset_index,
                "unit_index": unit_index,
                "test_rate_r2": float(r2_score(true, pred, axis=0)),
                "test_rate_correlation": float(np.corrcoef(true, pred)[0, 1]),
                "test_rate_normalized_rmse": float(
                    np.sqrt(np.mean(((true - pred) / max(true.std(), 1e-8)) ** 2))
                ),
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def export_state(model, source_state, raw_mean, raw_std):
    weight = model.encoder.weight.detach()
    transform = weight / raw_std[None]
    basis = transform.cpu() @ source_state.basis.float()
    source_center = (weight * raw_mean[None] / raw_std[None]).sum(1)
    norm = model.generator_norm
    generator_mean = norm.running_mean.detach() + source_center
    generator_std = torch.sqrt(norm.running_var.detach() + norm.eps)
    state = ReducedTwinState(
        rank=len(weight),
        unit_indices=source_state.unit_indices,
        basis=basis.cpu(),
        feature_mean=source_state.feature_mean,
        feature_std=source_state.feature_std,
        generator_mean=generator_mean.cpu(),
        generator_std=generator_std.cpu(),
        hidden_dims=tuple(
            layer.out_features
            for layer in model.decoder.network[:-1]
            if isinstance(layer, nn.Linear)
        ),
        dropout=float(
            next((layer.p for layer in model.decoder.network if isinstance(layer, nn.Dropout)), 0.0)
        ),
        decoder_state={key: value.detach().cpu() for key, value in model.decoder.state_dict().items()},
    )
    return state, transform


@torch.no_grad()
def export_audit(model, state, transform, raw_test, normalized_test, device):
    reduced = ReducedTwin(state).to(device).eval()
    count = min(4096, len(raw_test))
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        direct = model(normalized_test[:count]).float()
        composed_generators = raw_test[:count] @ transform.T
        exported = reduced.decode_generators(composed_generators).float()
    return float((direct - exported).abs().max())


def render_rank_curve(output_dir, summary):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.1))
    for architecture, part in summary.groupby("architecture"):
        axes[0].plot(part["rank"], part.median_test_rate_r2, "o-", label=architecture)
    axes[0].axhline(0.8, color="0.4", ls="--", lw=1)
    axes[0].set(xlabel="learned bottleneck rank", ylabel="median held-out response $R^2$", xscale="log")
    axes[0].legend(frameon=False)
    axes[1].plot(summary.parameter_count / 1e6, summary.median_test_rate_r2, "o-")
    for _, row in summary.iterrows():
        axes[1].annotate(f"r{int(row['rank'])}", (row.parameter_count / 1e6, row.median_test_rate_r2), xytext=(3, 3), textcoords="offset points")
    axes[1].set(xlabel="deployable parameters (millions)", ylabel="median held-out response $R^2$")
    fig.tight_layout()
    fig.savefig(output_dir / "linear_mlp_rank_curve.png", dpi=220)
    fig.savefig(output_dir / "linear_mlp_rank_curve.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    source_state = torch.load(args.source_state, map_location="cpu", weights_only=False)
    selection = pd.read_csv(SELECTION)
    raw, data, raw_mean, raw_std = load_data(args.source_bank, args.device)
    ranks = [int(value) for value in args.ranks.split(",") if value.strip()]
    architectures = parse_architectures(args.architectures)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows, all_metrics = [], []
    for rank in ranks:
        for hidden in architectures:
            name = architecture_name(hidden)
            candidates = []
            for seed in seeds:
                fitted = fit(data, rank, hidden, seed, args)
                candidates.append((fitted[1], seed, fitted))
            _, seed, fitted = min(candidates, key=lambda value: value[0])
            model, val_loss, best_epoch, history = fitted
            metric = test_metrics(model, data, selection, args.batch_size)
            state, transform = export_state(model, source_state, raw_mean, raw_std)
            audit = export_audit(
                model, state, transform, raw["test"][0], data["test"][0], args.device
            )
            metric["rank"] = rank
            metric["architecture"] = name
            all_metrics.append(metric)
            parameter_count = state.basis.numel() + sum(value.numel() for value in model.decoder.parameters())
            row = {
                "rank": rank,
                "architecture": name,
                "seed": seed,
                "best_epoch": best_epoch,
                "val_normalized_mse": val_loss,
                "parameter_count": parameter_count,
                "median_test_rate_r2": metric.test_rate_r2.median(),
                "min_test_rate_r2": metric.test_rate_r2.min(),
                "median_test_rate_correlation": metric.test_rate_correlation.median(),
                "export_max_abs_error": audit,
            }
            rows.append(row)
            torch.save(state, args.output_dir / f"linear_mlp_rank{rank}_{name}.pt")
            np.savez(
                args.output_dir / f"linear_encoder_rank{rank}_{name}.npz",
                weight=model.encoder.weight.detach().cpu().numpy(),
                source_generator_mean=raw_mean.cpu().numpy(),
                source_generator_std=raw_std.cpu().numpy(),
                composed_basis=state.basis.numpy(),
            )
            history.to_csv(args.output_dir / f"training_rank{rank}_{name}.csv", index=False)
            print(
                f"rank {rank:2d} {name}: test median R2={row['median_test_rate_r2']:.3f}, "
                f"min={row['min_test_rate_r2']:.3f}, params={parameter_count:,}",
                flush=True,
            )
    summary = pd.DataFrame(rows).sort_values(["rank", "parameter_count"])
    metrics = pd.concat(all_metrics, ignore_index=True)
    summary.to_csv(args.output_dir / "model_selection.csv", index=False)
    metrics.to_csv(args.output_dir / "unit_response_fidelity.csv", index=False)
    render_rank_curve(args.output_dir, summary)
    passing = summary.loc[summary.median_test_rate_r2.ge(0.8)].sort_values("parameter_count")
    if len(passing):
        selected = passing.iloc[0]
        rule = "smallest model with median held-out sampled response R2 >= 0.8"
    else:
        selected = summary.sort_values(["median_test_rate_r2", "parameter_count"], ascending=[False, True]).iloc[0]
        rule = "no model passed 0.8; diagnostic selection is highest held-out sampled response R2"
    payload = {
        "analysis": "supervised linear encoder followed by response MLP",
        "source_state": str(args.source_state.resolve()),
        "source_bank": str(args.source_bank.resolve()),
        "selection_rule": rule,
        "selected_model": selected.to_dict(),
        "selected_state": str(
            (args.output_dir / f"linear_mlp_rank{int(selected['rank'])}_{selected['architecture']}.pt").resolve()
        ),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
