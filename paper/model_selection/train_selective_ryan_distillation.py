#!/usr/bin/env python3
"""Refine a smooth student using selective Ryan prediction targets.

By default the visual cores and feature-level behavior pathway remain frozen.
Real spike likelihood is always optimized for every unit.  Ryan supplies an
auxiliary Poisson target only for units on which its deterministic held-out BPS
exceeds the smooth student's BPS by a configurable margin.  An explicit
``--allow-visual-trainable`` opt-in supports audited late-core experiments with
separate core learning rate, parent anchoring, and spatial smoothness; ordinary
head refinements still cannot move any visual weight or copy Ryan's pixel-scale
Jacobian geometry.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.model_selection.cache_ryan_teacher_rates import sha256_tensor, unwrap


class TeacherRateView(torch.utils.data.Dataset):
    """Attach cached teacher rates using the student's local split position."""

    def __init__(
        self,
        dataset,
        rates: torch.Tensor,
        rate_key: str = "teacher_rate",
    ):
        if len(dataset) != len(rates):
            raise ValueError(
                f"Dataset/cache length differs: {len(dataset)} != {len(rates)}"
            )
        if not rate_key:
            raise ValueError("Cached-rate key must be nonempty")
        self.dataset = dataset
        self.rates = rates
        self.rate_key = rate_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        item[self.rate_key] = self.rates[index]
        return item


def load_bps_archive(report_path: Path):
    """Return session -> (cids, per-unit BPS) from an evaluator report."""
    report = json.loads(report_path.read_text())
    archive_path = Path(report["per_unit_bps_npz"])
    if not archive_path.is_absolute():
        archive_path = (report_path.parent / archive_path).resolve()
    archive = np.load(archive_path, allow_pickle=False)
    output = {}
    for index, raw_name in enumerate(archive["session_names"]):
        name = str(raw_name)
        output[name] = (
            np.asarray(archive[f"cids_{index}"], dtype=np.int64),
            np.asarray(archive[f"bps_{index}"], dtype=np.float32),
        )
    return output


def unit_teacher_weights(
    teacher_report: Path,
    student_report: Path,
    sessions: list[str],
    cids_by_session: dict[str, list[int]],
    margin: float,
    scale: float,
) -> tuple[dict[int, torch.Tensor], dict]:
    """Build continuous unit weights from held-out teacher advantage."""
    if scale <= 0:
        raise ValueError("Teacher advantage scale must be positive")
    teacher = load_bps_archive(teacher_report)
    student = load_bps_archive(student_report)
    weights = {}
    report = {}
    for dataset_idx, name in enumerate(sessions):
        if name not in teacher or name not in student:
            raise RuntimeError(f"Missing BPS archive entry for {name}")
        teacher_cids, teacher_bps = teacher[name]
        student_cids, student_bps = student[name]
        expected_cids = np.asarray(cids_by_session[name], dtype=np.int64)
        if not (
            np.array_equal(teacher_cids, student_cids)
            and np.array_equal(teacher_cids, expected_cids)
        ):
            raise RuntimeError(f"Unit identities/order differ for {name}")
        advantage = teacher_bps - student_bps
        finite = np.isfinite(advantage)
        value = np.zeros_like(advantage, dtype=np.float32)
        value[finite] = np.clip((advantage[finite] - margin) / scale, 0.0, 1.0)
        weights[dataset_idx] = torch.from_numpy(value)
        report[name] = {
            "n_units": int(len(value)),
            "n_teacher_weighted": int((value > 0).sum()),
            "mean_teacher_weight": float(value.mean()),
            "mean_teacher_advantage_bps": float(np.nanmean(advantage)),
        }
    return weights, report


def weighted_teacher_loss(
    prediction: torch.Tensor,
    teacher_rate: torch.Tensor,
    dfs: torch.Tensor,
    unit_weight: torch.Tensor,
) -> torch.Tensor:
    """Masked Poisson cross-entropy with unit-specific teacher selection."""
    teacher_rate = teacher_rate.float()
    prediction = prediction.float()
    finite = torch.isfinite(teacher_rate)
    target = torch.nan_to_num(teacher_rate, nan=0.0, posinf=0.0, neginf=0.0)
    mask = dfs.float()
    if mask.ndim == 1:
        mask = mask[:, None]
    mask = mask * finite * unit_weight[None, :]
    denominator = mask.sum()
    if denominator <= 0:
        return prediction.sum() * 0.0
    loss = F.poisson_nll_loss(
        prediction, target, log_input=False, full=False, reduction="none"
    )
    return (loss * mask).sum() / denominator


def weighted_teacher_shape_loss(
    prediction: torch.Tensor,
    teacher_rate: torch.Tensor,
    dfs: torch.Tensor,
    unit_weight: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Match teacher modulation after per-unit affine normalization.

    The ordinary Poisson distillation term is deliberately sensitive to rate
    scale and offset.  Figure-3 CC/CCnorm instead measures the temporal shape
    after an affine calibration, so a small complementary correlation loss
    makes that objective explicit without importing any teacher derivative.
    Statistics are computed within each homogeneous-session minibatch and
    invalid observations are masked independently for every unit.
    """
    teacher_rate = teacher_rate.float()
    prediction = prediction.float()
    finite = torch.isfinite(teacher_rate)
    target = torch.nan_to_num(teacher_rate, nan=0.0, posinf=0.0, neginf=0.0)
    mask = dfs.float()
    if mask.ndim == 1:
        mask = mask[:, None]
    mask = mask * finite
    count = mask.sum(dim=0)
    safe_count = count.clamp_min(1.0)

    prediction_mean = (prediction * mask).sum(dim=0) / safe_count
    target_mean = (target * mask).sum(dim=0) / safe_count
    prediction_centered = prediction - prediction_mean[None, :]
    target_centered = target - target_mean[None, :]
    prediction_var = (
        prediction_centered.square() * mask
    ).sum(dim=0) / safe_count
    target_var = (target_centered.square() * mask).sum(dim=0) / safe_count
    covariance = (
        prediction_centered * target_centered * mask
    ).sum(dim=0) / safe_count
    correlation = covariance / (
        prediction_var.clamp_min(eps).sqrt()
        * target_var.clamp_min(eps).sqrt()
    )
    correlation = correlation.clamp(-1.0, 1.0)

    valid_unit = (
        (count >= 2)
        & (prediction_var > eps)
        & (target_var > eps)
        & torch.isfinite(correlation)
    )
    weight = unit_weight.float() * valid_unit.float()
    denominator = weight.sum()
    if denominator <= 0:
        return prediction.sum() * 0.0
    return ((1.0 - correlation) * weight).sum() / denominator


def relative_parameter_anchor_loss(named_parameters, reference):
    """Mean squared drift relative to each parameter's parent energy.

    Averaging one dimensionless ratio per tensor keeps the strength stable
    when a refinement includes both a million-weight convolution and small
    normalization vectors.  The reference is captured before the first
    optimizer step, so the loss is exactly zero at initialization.
    """
    terms = []
    for name, parameter in named_parameters:
        target = reference[name]
        denominator = target.float().square().sum().clamp_min(1.0e-12)
        terms.append((parameter.float() - target.float()).square().sum() / denominator)
    if not terms:
        raise ValueError("Parameter anchoring requires at least one tensor")
    return torch.stack(terms).mean()


def spatial_laplacian_loss(named_parameters):
    """Dekel-style spatial curvature penalty for trainable convolution kernels."""
    from training.regularizers import laplacian_penalty

    terms = [
        laplacian_penalty(parameter, (-2, -1), reduction="dekel")
        for _, parameter in named_parameters
        if parameter.ndim >= 4
    ]
    if not terms:
        raise ValueError("Spatial Laplacian regularization matched no kernels")
    return torch.stack(terms).sum()


_UNIT_PARAMETER_PATTERN = re.compile(
    r"^model\.(?:readouts|auxiliary_readouts|residual_readouts|"
    r"auxiliary_residual_readouts|residual_visual_readouts)\.(\d+)\."
    r"|^model\.(?:output_modulator|distilled_output_modulator)\."
    r"(?:offset_layers|gain_layers)\.(\d+)\."
)


def install_unit_isolation(
    named_parameters,
    teacher_weights,
    allowed_shared_patterns=(),
):
    """Mask and snapshot unit rows so unselected predictions cannot drift.

    This mode deliberately rejects shared tensors.  It is valid only when the
    caller has frozen the shared behavior encoder and selected unit-indexed
    Gaussian/readout or gain/offset parameters.  Gradient hooks prevent
    unselected rows from affecting global gradient clipping; the returned
    restore callback also reverses AdamW and proximal updates on those rows.
    """
    references = []
    reports = {}
    unexpected = []
    allowed_shared = []
    for name, parameter in named_parameters:
        match = _UNIT_PARAMETER_PATTERN.match(name)
        if match is None:
            if any(pattern in name for pattern in allowed_shared_patterns):
                allowed_shared.append(name)
            else:
                unexpected.append(name)
            continue
        dataset_idx = int(match.group(1) or match.group(2))
        if dataset_idx not in teacher_weights:
            raise RuntimeError(
                f"No teacher weights for dataset {dataset_idx} ({name})"
            )
        selected = teacher_weights[dataset_idx] > 0
        if parameter.ndim == 0 or parameter.shape[0] != selected.numel():
            raise RuntimeError(
                "Unit-isolated parameter does not lead with the neuron axis: "
                f"{name} has {tuple(parameter.shape)}, expected "
                f"{selected.numel()} rows"
            )
        selected = selected.to(device=parameter.device)
        gradient_mask = selected.to(dtype=parameter.dtype).reshape(
            (-1,) + (1,) * (parameter.ndim - 1)
        )
        parameter.register_hook(
            lambda gradient, mask=gradient_mask: gradient * mask
        )
        unselected = ~selected
        references.append(
            (parameter, unselected, parameter.detach().clone())
        )
        reports[name] = {
            "dataset_idx": dataset_idx,
            "n_rows": int(selected.numel()),
            "n_trainable_rows": int(selected.sum()),
        }
    if unexpected:
        raise RuntimeError(
            "Unit isolation requires exclusively unit-indexed trainable "
            "parameters. Freeze shared encoders; unexpected tensors: "
            f"{unexpected[:10]}"
        )
    unmatched_patterns = [
        pattern
        for pattern in allowed_shared_patterns
        if not any(pattern in name for name in allowed_shared)
    ]
    if unmatched_patterns:
        raise RuntimeError(
            "Allowed shared isolation patterns matched no trainable "
            f"parameters: {unmatched_patterns}"
        )
    if not references:
        raise RuntimeError("Unit isolation matched no trainable parameters")

    @torch.no_grad()
    def restore_unselected_rows():
        for parameter, unselected, reference in references:
            parameter[unselected] = reference[unselected]

    return restore_unselected_rows, reports


def save_checkpoint(
    parent_checkpoint: dict,
    model,
    output: Path,
    epoch: int,
    global_step: int,
    distillation_metadata: dict,
) -> None:
    """Write an evaluator-compatible, self-contained student checkpoint."""
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    parent_keys = set(parent_checkpoint["state_dict"])
    if set(state) != parent_keys:
        missing = sorted(parent_keys - set(state))[:10]
        extra = sorted(set(state) - parent_keys)[:10]
        raise RuntimeError(
            f"Checkpoint state contract changed; missing={missing}, extra={extra}"
        )
    artifact = dict(parent_checkpoint)
    artifact["state_dict"] = state
    artifact["epoch"] = int(epoch)
    artifact["global_step"] = int(global_step)
    artifact["optimizer_states"] = []
    artifact["lr_schedulers"] = []
    # Parent callback state contains its old validation score and checkpoint
    # ranking.  Retaining it would make diagnostics label these new weights
    # with M48's score even though exact evaluation has not yet run.
    artifact["callbacks"] = {}
    artifact["selective_ryan_distillation"] = distillation_metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student_checkpoint", type=Path)
    parser.add_argument("--teacher-cache-dir", type=Path, required=True)
    parser.add_argument("--teacher-evaluation", type=Path, required=True)
    parser.add_argument("--student-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-datasets", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--steps-per-epoch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--distillation-strength", type=float, default=0.25)
    parser.add_argument(
        "--teacher-shape-strength",
        type=float,
        default=0.0,
        help=(
            "Optional affine-invariant within-batch correlation loss against "
            "Ryan, complementary to Poisson distillation."
        ),
    )
    parser.add_argument(
        "--parent-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional same-grid cache of the starting student's train rates. "
            "Required when preserving units that do not receive Ryan targets."
        ),
    )
    parser.add_argument(
        "--student-preservation-strength",
        type=float,
        default=0.0,
        help=(
            "Poisson self-distillation strength on the complement of each "
            "unit's Ryan-teacher weight."
        ),
    )
    parser.add_argument("--teacher-advantage-margin", type=float, default=0.01)
    parser.add_argument("--teacher-advantage-scale", type=float, default=0.04)
    parser.add_argument(
        "--trainable-patterns", default="readouts,output_modulator"
    )
    parser.add_argument(
        "--freeze-unselected-unit-parameters",
        action="store_true",
        help=(
            "Exactly preserve unit-indexed parameter rows whose Ryan teacher "
            "weight is zero. Shared trainable tensors are rejected."
        ),
    )
    parser.add_argument(
        "--allow-shared-unit-isolation-patterns",
        default="",
        help=(
            "Comma-separated explicit opt-in for shared tensors that may "
            "train alongside isolated neuron rows. Intended only for a new "
            "identity-gated residual path."
        ),
    )
    parser.add_argument(
        "--allow-visual-trainable",
        action="store_true",
        help="Explicit opt-in for tightly controlled late-core refinement.",
    )
    parser.add_argument(
        "--core-learning-rate",
        type=float,
        default=None,
        help="Learning rate for convnet/modulator tensors; defaults to head LR.",
    )
    parser.add_argument(
        "--parent-anchor-strength",
        type=float,
        default=0.0,
        help="Dimensionless relative-weight anchor applied to trainable visual tensors.",
    )
    parser.add_argument(
        "--visual-spatial-laplacian-strength",
        type=float,
        default=0.0,
        help="Dekel-reduced spatial Laplacian penalty on trainable visual kernels.",
    )
    parser.add_argument("--gradient-clip-val", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--milestones", default="3,7,15,31")
    parser.add_argument("--enable-logging", action="store_true")
    parser.add_argument("--project-name", default="model_selection")
    parser.add_argument("--experiment-name", default="D240M53_selective_ryan_distill")
    args = parser.parse_args()

    if args.distillation_strength < 0:
        raise ValueError("Distillation strength must be nonnegative")
    if args.teacher_shape_strength < 0:
        raise ValueError("Teacher shape strength must be nonnegative")
    if args.student_preservation_strength < 0:
        raise ValueError("Student preservation strength must be nonnegative")
    if (args.parent_cache_dir is None) != (
        args.student_preservation_strength == 0
    ):
        raise ValueError(
            "--parent-cache-dir and a positive "
            "--student-preservation-strength must be supplied together"
        )
    allowed_shared_isolation_patterns = tuple(
        value.strip()
        for value in args.allow_shared_unit_isolation_patterns.split(",")
        if value.strip()
    )
    if (
        allowed_shared_isolation_patterns
        and not args.freeze_unselected_unit_parameters
    ):
        raise ValueError(
            "Shared isolation patterns require "
            "--freeze-unselected-unit-parameters"
        )
    milestones = {
        int(value) for value in args.milestones.split(",") if value.strip()
    }
    if any(epoch < 0 or epoch >= args.epochs for epoch in milestones):
        raise ValueError("Milestones must be zero-based epochs within --epochs")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    from eval.load_twin import load_twin
    from training.pl_modules import MultiDatasetDM
    from training.pl_modules.multidataset_model import _adamw_param_groups_named
    from training.regularizers import get_excluded_params_for_weight_decay
    from training.schedulers import LinearWarmupCosineAnnealingLR

    checkpoint_path = args.student_checkpoint.resolve()
    parent_checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    cfg_dir = (parent_checkpoint.get("hyper_parameters", {}) or {}).get("cfg_dir")
    if cfg_dir is None:
        raise ValueError("Student checkpoint does not record cfg_dir")
    model, info = load_twin(
        checkpoint_path, device=str(device), verbose=False
    )
    sessions = model.names[: args.max_datasets]
    patterns = tuple(
        value.strip() for value in args.trainable_patterns.split(",") if value.strip()
    )
    if not patterns:
        raise ValueError("At least one trainable pattern is required")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(any(pattern in name for pattern in patterns))
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names:
        raise RuntimeError(f"Patterns {patterns} matched no parameters")
    visual_trainable = [
        name
        for name in trainable_names
        if "convnet" in name or name.startswith("model.modulator")
    ]
    if visual_trainable and not args.allow_visual_trainable:
        raise RuntimeError(
            "Selective distillation requires frozen visual/feature behavior paths; "
            "pass --allow-visual-trainable only for an explicitly audited late-core "
            f"refinement. Unexpected trainable parameters: {visual_trainable[:10]}"
        )
    if args.parent_anchor_strength < 0 or args.visual_spatial_laplacian_strength < 0:
        raise ValueError("Visual regularization strengths must be nonnegative")
    if not visual_trainable and (
        args.parent_anchor_strength > 0
        or args.visual_spatial_laplacian_strength > 0
    ):
        raise ValueError("Visual regularization was requested with no trainable visual tensors")
    visual_named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name in visual_trainable
    ]
    visual_reference = {
        name: parameter.detach().clone()
        for name, parameter in visual_named_parameters
    }

    datamodule = MultiDatasetDM(
        cfg_dir=cfg_dir,
        max_ds=args.max_datasets,
        batch=args.batch_size,
        workers=args.num_workers,
        steps_per_epoch=args.steps_per_epoch,
        dset_dtype="uint8",
        homogeneous_batches=True,
    )
    datamodule.setup("fit")
    if datamodule.names != sessions:
        raise RuntimeError("Student checkpoint and data-module session order differ")

    cache_manifest = json.loads(
        (args.teacher_cache_dir / "manifest.json").read_text()
    )
    if Path(cache_manifest["student_dataset_config"]).resolve() != Path(cfg_dir).resolve():
        raise RuntimeError("Teacher cache was built for a different student config")
    cache_reports = {}
    parent_cache_reports = {}
    parent_cache_manifest = None
    if args.parent_cache_dir is not None:
        parent_cache_manifest = json.loads(
            (args.parent_cache_dir / "manifest.json").read_text()
        )
        if Path(parent_cache_manifest["teacher_checkpoint"]).resolve() != checkpoint_path:
            raise RuntimeError(
                "Parent cache was not generated from the starting student checkpoint"
            )
        if Path(parent_cache_manifest["student_dataset_config"]).resolve() != Path(
            cfg_dir
        ).resolve():
            raise RuntimeError("Parent cache was built for a different student config")
        if (
            int(parent_cache_manifest.get("factor", -1)) != 1
            or int(parent_cache_manifest.get("phase", -1)) != 0
        ):
            raise RuntimeError(
                "Parent prediction preservation requires a same-grid cache "
                "with factor=1 and phase=0"
            )
    for dataset_idx, name in enumerate(sessions):
        cache_path = args.teacher_cache_dir / f"{dataset_idx:02d}_{name}_train.pt"
        artifact = torch.load(cache_path, map_location="cpu", weights_only=False)
        metadata = artifact["metadata"]
        dataset = datamodule.train_dsets[name]
        if metadata["session"] != name or metadata["split"] != "train":
            raise RuntimeError(f"Cache identity mismatch in {cache_path}")
        current_hash = sha256_tensor(unwrap(dataset).inds)
        if metadata["student_inds_sha256"] != current_hash:
            raise RuntimeError(f"Student split indices changed for {name}")
        if list(metadata["cids"]) != list(info["cids_by_session"][name]):
            raise RuntimeError(f"Cached unit identities changed for {name}")
        dataset = TeacherRateView(dataset, artifact["rates"], "teacher_rate")
        cache_reports[name] = metadata
        if args.parent_cache_dir is not None:
            parent_path = (
                args.parent_cache_dir / f"{dataset_idx:02d}_{name}_train.pt"
            )
            parent_artifact = torch.load(
                parent_path, map_location="cpu", weights_only=False
            )
            parent_metadata = parent_artifact["metadata"]
            if parent_metadata["session"] != name or parent_metadata["split"] != "train":
                raise RuntimeError(f"Parent cache identity mismatch in {parent_path}")
            if parent_metadata["student_inds_sha256"] != current_hash:
                raise RuntimeError(f"Parent cache split indices changed for {name}")
            if list(parent_metadata["cids"]) != list(info["cids_by_session"][name]):
                raise RuntimeError(f"Parent cache unit identities changed for {name}")
            dataset = TeacherRateView(
                dataset, parent_artifact["rates"], "parent_rate"
            )
            parent_cache_reports[name] = parent_metadata
        datamodule.train_dsets[name] = dataset

    teacher_weights, weight_report = unit_teacher_weights(
        args.teacher_evaluation.resolve(),
        args.student_evaluation.resolve(),
        sessions,
        info["cids_by_session"],
        args.teacher_advantage_margin,
        args.teacher_advantage_scale,
    )
    teacher_weights = {
        index: value.to(device) for index, value in teacher_weights.items()
    }
    preservation_weights = {
        index: (1.0 - value).clamp_(0.0, 1.0)
        for index, value in teacher_weights.items()
    }
    restore_unselected_rows = None
    unit_isolation_report = {}
    if args.freeze_unselected_unit_parameters:
        restore_unselected_rows, unit_isolation_report = install_unit_isolation(
            [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ],
            teacher_weights,
            allowed_shared_patterns=allowed_shared_isolation_patterns,
        )

    excluded_names = set(get_excluded_params_for_weight_decay(model.reg_terms))
    param_groups = _adamw_param_groups_named(
        list(model.named_parameters()),
        wd=args.weight_decay,
        excluded_names=excluded_names,
        core_lr=(
            args.learning_rate
            if args.core_learning_rate is None
            else args.core_learning_rate
        ),
        head_lr=args.learning_rate,
    )
    optimizer = torch.optim.AdamW(
        param_groups, betas=(0.9, 0.999), eps=1.0e-8
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        warmup_start_lr=0.0,
        eta_min=0.0,
    )

    run = None
    if args.enable_logging:
        import wandb

        wandb_config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        run = wandb.init(
            project=args.project_name,
            name=args.experiment_name,
            config={
                **wandb_config,
                "student_checkpoint": str(checkpoint_path),
                "teacher_cache_dir": str(args.teacher_cache_dir.resolve()),
                "trainable_parameter_count": int(
                    sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
                ),
            },
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "student_checkpoint": str(checkpoint_path),
        "teacher_cache_dir": str(args.teacher_cache_dir.resolve()),
        "teacher_evaluation": str(args.teacher_evaluation.resolve()),
        "student_evaluation": str(args.student_evaluation.resolve()),
        "distillation_strength": args.distillation_strength,
        "teacher_shape_strength": args.teacher_shape_strength,
        "teacher_advantage_margin": args.teacher_advantage_margin,
        "teacher_advantage_scale": args.teacher_advantage_scale,
        "trainable_patterns": list(patterns),
        "trainable_names": trainable_names,
        "unit_weight_report": weight_report,
        "cache_report": cache_reports,
        "parent_cache_report": parent_cache_reports,
        "unit_isolation_report": unit_isolation_report,
        "allowed_shared_unit_isolation_patterns": list(
            allowed_shared_isolation_patterns
        ),
        "history": [],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2))

    loader = datamodule.train_dataloader()
    global_step = int(parent_checkpoint.get("global_step", 0))
    model.train()
    for epoch in range(args.epochs):
        setter = getattr(loader.batch_sampler, "set_epoch", None)
        if setter is not None:
            setter(epoch)
        iterator = iter(loader)
        totals = {
            "actual": 0.0,
            "teacher": 0.0,
            "teacher_shape": 0.0,
            "student_preservation": 0.0,
            "parent_anchor": 0.0,
            "visual_laplacian": 0.0,
            "total": 0.0,
        }
        session_steps = {name: 0 for name in sessions}
        for _ in range(args.steps_per_epoch):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            if isinstance(batch, list):
                if len(batch) != 1:
                    raise RuntimeError("Expected one homogeneous session per batch")
                batch = batch[0]
            batch = {
                key: value.to(device, non_blocking=True)
                if torch.is_tensor(value)
                else value
                for key, value in batch.items()
            }
            dataset_idx = int(batch["dataset_idx"][0])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(
                    batch["stim"],
                    dataset_idx,
                    batch.get("behavior"),
                    batch.get("history"),
                    batch.get("output_behavior"),
                )
            actual_loss = model.loss_fn(
                {
                    "rhat": prediction.float(),
                    "robs": batch["robs"].float(),
                    "dfs": batch["dfs"].float(),
                }
            )
            teacher_loss = weighted_teacher_loss(
                prediction,
                batch["teacher_rate"],
                batch["dfs"],
                teacher_weights[dataset_idx],
            )
            total_loss = actual_loss + args.distillation_strength * teacher_loss
            teacher_shape_loss = prediction.new_zeros((), dtype=torch.float32)
            if args.teacher_shape_strength > 0:
                teacher_shape_loss = weighted_teacher_shape_loss(
                    prediction,
                    batch["teacher_rate"],
                    batch["dfs"],
                    teacher_weights[dataset_idx],
                )
                total_loss = (
                    total_loss
                    + args.teacher_shape_strength * teacher_shape_loss
                )
            student_preservation = prediction.new_zeros((), dtype=torch.float32)
            if args.student_preservation_strength > 0:
                student_preservation = weighted_teacher_loss(
                    prediction,
                    batch["parent_rate"],
                    batch["dfs"],
                    preservation_weights[dataset_idx],
                )
                total_loss = (
                    total_loss
                    + args.student_preservation_strength * student_preservation
                )
            parent_anchor = prediction.new_zeros((), dtype=torch.float32)
            if args.parent_anchor_strength > 0:
                parent_anchor = relative_parameter_anchor_loss(
                    visual_named_parameters, visual_reference
                )
                total_loss = total_loss + args.parent_anchor_strength * parent_anchor
            visual_laplacian = prediction.new_zeros((), dtype=torch.float32)
            if args.visual_spatial_laplacian_strength > 0:
                visual_laplacian = spatial_laplacian_loss(visual_named_parameters)
                total_loss = (
                    total_loss
                    + args.visual_spatial_laplacian_strength * visual_laplacian
                )
            for regularizer in model.reg_terms:
                if any(parameter.requires_grad for parameter in regularizer.params):
                    regularizer_loss = regularizer.loss(epoch)
                    if torch.isfinite(regularizer_loss):
                        total_loss = total_loss + regularizer_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.gradient_clip_val,
            )
            optimizer.step()
            fallback_lr = max(float(group["lr"]) for group in optimizer.param_groups)
            for regularizer in model.reg_terms:
                if any(parameter.requires_grad for parameter in regularizer.params):
                    regularizer.prox(epoch, fallback_lr, optimizer=optimizer)
            if restore_unselected_rows is not None:
                restore_unselected_rows()

            totals["actual"] += float(actual_loss.detach())
            totals["teacher"] += float(teacher_loss.detach())
            totals["teacher_shape"] += float(teacher_shape_loss.detach())
            totals["student_preservation"] += float(
                student_preservation.detach()
            )
            totals["parent_anchor"] += float(parent_anchor.detach())
            totals["visual_laplacian"] += float(visual_laplacian.detach())
            totals["total"] += float(total_loss.detach())
            session_steps[sessions[dataset_idx]] += 1
            global_step += 1

        scheduler.step()
        record = {
            "epoch": epoch,
            "actual_loss": totals["actual"] / args.steps_per_epoch,
            "teacher_loss": totals["teacher"] / args.steps_per_epoch,
            "teacher_shape_loss": (
                totals["teacher_shape"] / args.steps_per_epoch
            ),
            "student_preservation_loss": (
                totals["student_preservation"] / args.steps_per_epoch
            ),
            "parent_anchor_loss": totals["parent_anchor"] / args.steps_per_epoch,
            "visual_laplacian_loss": totals["visual_laplacian"] / args.steps_per_epoch,
            "total_loss": totals["total"] / args.steps_per_epoch,
            "learning_rate": max(float(group["lr"]) for group in optimizer.param_groups),
            "global_step": global_step,
            "session_steps": session_steps,
        }
        metadata["history"].append(record)
        print(json.dumps(record), flush=True)
        if run is not None:
            run.log(
                {key: value for key, value in record.items() if key != "session_steps"},
                step=global_step,
            )
        (args.output_dir / "manifest.json").write_text(
            json.dumps(metadata, indent=2)
        )
        if epoch in milestones:
            save_checkpoint(
                parent_checkpoint,
                model,
                args.output_dir / "analysis_candidates" / f"epoch={epoch:03d}-endpoint.ckpt",
                epoch,
                global_step,
                metadata,
            )
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
