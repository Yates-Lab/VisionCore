"""Exact ConvGRU transport and candidate-realignment interventions.

All interventions are expressed as a delta from the literal concatenated
convolution.  This matters on CUDA: executing the current and recurrent
halves as two convolutions changes accumulation order slightly.  Adding only
``modified_recurrent - original_recurrent`` to the literal preactivation
makes every identity/no-shift endpoint exactly the frozen implementation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.equations import (
    project_coordinates,
    project_native,
    split_conv2d_contributions,
    zero_hidden_like,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.registration import (
    normalized_multichannel_xcorr,
)


ANALYSIS_SEED = 20260812
KernelMode = Literal["intact", "center_only", "permute_offcenter", "zero"]
RealignmentTarget = Literal["none", "candidate_p", "complementary_q"]


@dataclass(frozen=True)
class KernelIntervention:
    """One frozen-weight recurrent-kernel condition.

    ``gate_mode`` applies to the hidden halves of the reset and update
    kernels.  ``candidate_mode`` applies only to the hidden half of the
    candidate kernel after reset gating.  ``reset_hidden_each_step`` is the
    explicit no-recurrence reference and is not treated as a transport
    ablation.
    """

    name: str
    gate_mode: KernelMode = "intact"
    candidate_mode: KernelMode = "intact"
    permutation_seed: int | None = None
    reset_hidden_each_step: bool = False
    interpretation: str = ""


OFFSET_PERMUTATION_SEEDS: tuple[int, ...] = (
    ANALYSIS_SEED + 101,
    ANALYSIS_SEED + 211,
    ANALYSIS_SEED + 307,
)


TRANSPORT_INTERVENTIONS: tuple[KernelIntervention, ...] = (
    KernelIntervention(
        "intact",
        interpretation="Literal frozen ConvGRU; exact numerical identity control.",
    ),
    KernelIntervention(
        "candidate_recurrent_center_only",
        candidate_mode="center_only",
        interpretation="Remove only off-center candidate hidden-to-hidden transport.",
    ),
    KernelIntervention(
        "gate_recurrent_center_only",
        gate_mode="center_only",
        interpretation="Remove only off-center reset/update hidden-to-hidden transport.",
    ),
    KernelIntervention(
        "all_recurrent_center_only",
        gate_mode="center_only",
        candidate_mode="center_only",
        interpretation="Remove every off-center hidden-to-hidden spatial tap.",
    ),
    *tuple(
        KernelIntervention(
            f"all_recurrent_offset_permuted_seed_{seed}",
            gate_mode="permute_offcenter",
            candidate_mode="permute_offcenter",
            permutation_seed=seed,
            interpretation=(
                "Apply one fixed global permutation of the eight off-center offsets to "
                "all recurrent kernels; every channel matrix, center tap, and norm is retained."
            ),
        )
        for seed in OFFSET_PERMUTATION_SEEDS
    ),
    KernelIntervention(
        "no_recurrence_reference",
        reset_hidden_each_step=True,
        interpretation=(
            "Reset hidden state to zero before every internal step; current input processing "
            "remains intact. This is a reference, not the primary spatial-transport control."
        ),
    ),
)


@dataclass(frozen=True)
class RealignmentReplay:
    """Output and applied shifts from one exact candidate realignment replay."""

    sequence: torch.Tensor
    applied_shift_yx_px: torch.Tensor
    oracle_valid: torch.Tensor


def _spatial_positions(kernel_size: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    if int(kernel_size) < 1 or int(kernel_size) % 2 != 1:
        raise ValueError("Recurrent spatial kernels must have positive odd width")
    center = (int(kernel_size) // 2, int(kernel_size) // 2)
    offcenter = [
        (row, column)
        for row in range(int(kernel_size))
        for column in range(int(kernel_size))
        if (row, column) != center
    ]
    return offcenter, center


def deterministic_offcenter_permutation(kernel_size: int, seed: int) -> np.ndarray:
    """Return a fixed non-identity permutation of off-center tap indices."""
    positions, _ = _spatial_positions(kernel_size)
    permutation = np.random.default_rng(int(seed)).permutation(len(positions))
    if len(permutation) > 1 and np.array_equal(permutation, np.arange(len(permutation))):
        permutation = np.roll(permutation, 1)
    return permutation.astype(np.int64)


def transform_recurrent_kernel(
    hidden_weight: torch.Tensor,
    mode: KernelMode,
    *,
    seed: int | None = None,
) -> torch.Tensor:
    """Transform only spatial taps of an ``[out,in,k,k]`` recurrent kernel."""
    if hidden_weight.ndim != 4 or hidden_weight.shape[-1] != hidden_weight.shape[-2]:
        raise ValueError(f"Expected square OIHW kernel, found {tuple(hidden_weight.shape)}")
    width = int(hidden_weight.shape[-1])
    positions, center = _spatial_positions(width)
    if mode == "intact":
        return hidden_weight
    transformed = hidden_weight.clone()
    if mode == "zero":
        return transformed.zero_()
    if mode == "center_only":
        transformed.zero_()
        transformed[..., center[0], center[1]] = hidden_weight[..., center[0], center[1]]
        return transformed
    if mode != "permute_offcenter":
        raise ValueError(f"Unknown recurrent-kernel mode: {mode}")
    if seed is None:
        raise ValueError("An explicit fixed seed is required for offset permutation")
    permutation = deterministic_offcenter_permutation(width, int(seed))
    for destination, source_index in zip(positions, permutation.tolist()):
        source = positions[int(source_index)]
        transformed[..., destination[0], destination[1]] = hidden_weight[..., source[0], source[1]]
    transformed[..., center[0], center[1]] = hidden_weight[..., center[0], center[1]]
    return transformed


def recurrent_kernel_invariants(
    original: torch.Tensor,
    transformed: torch.Tensor,
) -> dict[str, float | bool]:
    """Numerical audit of center, total norm, and off-center matrix multiset."""
    if original.shape != transformed.shape:
        raise ValueError("Kernel invariant check requires matching shapes")
    positions, center = _spatial_positions(int(original.shape[-1]))
    original_off = torch.stack([original[..., row, column] for row, column in positions])
    transformed_off = torch.stack([transformed[..., row, column] for row, column in positions])
    original_norms = torch.sort(original_off.flatten(1).square().sum(1)).values
    transformed_norms = torch.sort(transformed_off.flatten(1).square().sum(1)).values
    return {
        "center_max_abs": float(
            (original[..., center[0], center[1]] - transformed[..., center[0], center[1]])
            .abs()
            .max()
            .detach()
            .cpu()
        ),
        "total_norm_relative_error": float(
            (
                original.square().sum().sqrt() - transformed.square().sum().sqrt()
            ).abs().detach().cpu()
            / max(float(original.square().sum().sqrt().detach().cpu()), 1e-30)
        ),
        "offcenter_tap_norm_multiset_equal": bool(
            torch.allclose(original_norms, transformed_norms, atol=1e-7, rtol=1e-7)
        ),
    }


def _modified_literal_preactivation(
    layer: torch.nn.Conv2d,
    current: torch.Tensor,
    hidden: torch.Tensor,
    mode: KernelMode,
    *,
    seed: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return modified literal preactivation and original/modified recurrence."""
    _, original_recurrent, literal = split_conv2d_contributions(layer, current, hidden)
    if mode == "intact":
        return literal, original_recurrent, original_recurrent
    input_channels = int(current.shape[1])
    hidden_weight = layer.weight[:, input_channels:]
    transformed_weight = transform_recurrent_kernel(hidden_weight, mode, seed=seed)
    modified_recurrent = F.conv2d(
        hidden,
        transformed_weight,
        None,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=1,
    )
    # Delta-from-literal gives exact identity when the transform is identity.
    return literal + (modified_recurrent - original_recurrent), original_recurrent, modified_recurrent


def transport_intervention_step(
    cell: torch.nn.Module,
    x_t: torch.Tensor,
    h_previous: torch.Tensor | None,
    intervention: KernelIntervention,
) -> torch.Tensor:
    """Execute one exact VisionCore GRU step under a recurrent-kernel condition."""
    if h_previous is None or intervention.reset_hidden_each_step:
        h_previous = zero_hidden_like(cell, x_t)
    if intervention.name == "intact":
        return cell(x_t, h_previous)
    seed = intervention.permutation_seed
    z_pre, _, _ = _modified_literal_preactivation(
        cell.update_gate, x_t, h_previous, intervention.gate_mode, seed=seed
    )
    r_pre, _, _ = _modified_literal_preactivation(
        cell.reset_gate, x_t, h_previous, intervention.gate_mode, seed=seed
    )
    z = torch.sigmoid(z_pre)
    r = torch.sigmoid(r_pre)
    reset_hidden = r * h_previous
    candidate_pre, _, _ = _modified_literal_preactivation(
        cell.out_gate,
        x_t,
        reset_hidden,
        intervention.candidate_mode,
        seed=seed,
    )
    candidate = torch.tanh(candidate_pre)
    return (1.0 - z) * h_previous + z * candidate


def replay_transport_intervention(
    cell: torch.nn.Module,
    sequence: torch.Tensor,
    intervention: KernelIntervention,
) -> torch.Tensor:
    """Replay one ``B,C,T,H,W`` window and return the complete state sequence."""
    if sequence.ndim != 5:
        raise ValueError(f"Expected BCTHW sequence, found {tuple(sequence.shape)}")
    state: torch.Tensor | None = None
    outputs: list[torch.Tensor] = []
    for internal_step in range(int(sequence.shape[2])):
        state = transport_intervention_step(
            cell, sequence[:, :, internal_step], state, intervention
        )
        outputs.append(state)
    return torch.stack(outputs, dim=2)


def fourier_shift_2d(
    value: torch.Tensor,
    shift_yx_px: torch.Tensor,
    *,
    preserve_channel_norm: bool = True,
) -> torch.Tensor:
    """Periodically translate BCHW maps by batched subpixel shifts.

    Positive ``dy`` moves content down and positive ``dx`` moves it right.
    The Fourier operator changes phase only.  A tiny channel-wise correction
    compensates real-Nyquist roundoff after ``irfft2`` and makes the declared
    no-amplitude-change constraint explicit.
    """
    if value.ndim != 4:
        raise ValueError(f"Expected BCHW maps, found {tuple(value.shape)}")
    shifts = torch.as_tensor(shift_yx_px, dtype=value.dtype, device=value.device)
    if shifts.ndim == 1:
        shifts = shifts[None].expand(value.shape[0], -1)
    if shifts.shape != (value.shape[0], 2):
        raise ValueError(f"Expected shifts {(value.shape[0], 2)}, found {tuple(shifts.shape)}")
    height, width = map(int, value.shape[-2:])
    ky = torch.fft.fftfreq(height, device=value.device, dtype=value.dtype)[None, :, None]
    kx = torch.fft.rfftfreq(width, device=value.device, dtype=value.dtype)[None, None, :]
    phase_argument = (
        ky * shifts[:, 0, None, None] + kx * shifts[:, 1, None, None]
    )
    phase = torch.exp((-2j * math.pi) * phase_argument)[:, None]
    shifted = torch.fft.irfft2(
        torch.fft.rfft2(value, dim=(-2, -1)) * phase,
        s=(height, width),
        dim=(-2, -1),
    )
    if preserve_channel_norm:
        before = value.square().sum(dim=(-2, -1), keepdim=True).sqrt()
        after = shifted.square().sum(dim=(-2, -1), keepdim=True).sqrt()
        scale = torch.where(before > 1e-20, before / after.clamp_min(1e-20), torch.ones_like(before))
        shifted = shifted * scale
    # Preserve the declared identity endpoint bit-for-bit for zero-shift
    # samples; an FFT round trip must never masquerade as an intervention.
    zero = shifts.eq(0).all(dim=1)
    if bool(zero.any()):
        shifted = torch.where(zero[:, None, None, None], value, shifted)
    return shifted


def oracle_candidate_shift_yx(
    candidate_current: torch.Tensor,
    candidate_recurrent: torch.Tensor,
    basis: torch.Tensor,
    *,
    max_lag_px: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upper-bound shift that aligns P recurrent evidence to P current evidence."""
    current_p = project_coordinates(candidate_current, basis)
    recurrent_p = project_coordinates(candidate_recurrent, basis)
    peak = normalized_multichannel_xcorr(current_p, recurrent_p, max_lag_px=max_lag_px)
    # The measured lag is recurrent content relative to current content; the
    # corrective content translation is its negative.
    shift = torch.stack([-peak.lag_y_px, -peak.lag_x_px], dim=1)
    shift = torch.where(peak.valid[:, None], shift, torch.zeros_like(shift))
    return shift, peak.valid


def realignment_step(
    cell: torch.nn.Module,
    x_t: torch.Tensor,
    h_previous: torch.Tensor | None,
    basis: torch.Tensor,
    *,
    shift_yx_px: torch.Tensor | None,
    target: RealignmentTarget,
    oracle: bool = False,
    max_oracle_lag_px: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shift only P or Q of the recurrent candidate contribution.

    Reset/update gates, current candidate input, retained state, and the
    complementary candidate component remain literal and untouched.
    """
    if h_previous is None:
        h_previous = zero_hidden_like(cell, x_t)
    if target == "none":
        zeros = torch.zeros((x_t.shape[0], 2), dtype=x_t.dtype, device=x_t.device)
        valid = torch.ones(x_t.shape[0], dtype=torch.bool, device=x_t.device)
        return cell(x_t, h_previous), zeros, valid

    _, _, z_literal = split_conv2d_contributions(cell.update_gate, x_t, h_previous)
    _, _, r_literal = split_conv2d_contributions(cell.reset_gate, x_t, h_previous)
    z = torch.sigmoid(z_literal)
    r = torch.sigmoid(r_literal)
    reset_hidden = r * h_previous
    candidate_current, candidate_recurrent, candidate_literal = split_conv2d_contributions(
        cell.out_gate, x_t, reset_hidden
    )
    if oracle:
        applied, valid = oracle_candidate_shift_yx(
            candidate_current,
            candidate_recurrent,
            basis,
            max_lag_px=int(max_oracle_lag_px),
        )
    else:
        if shift_yx_px is None:
            raise ValueError("Eye/control realignment requires explicit feature-map shifts")
        applied = torch.as_tensor(shift_yx_px, dtype=x_t.dtype, device=x_t.device)
        if applied.ndim == 1:
            applied = applied[None].expand(x_t.shape[0], -1)
        if applied.shape != (x_t.shape[0], 2):
            raise ValueError(f"Expected shifts {(x_t.shape[0], 2)}, found {tuple(applied.shape)}")
        valid = torch.isfinite(applied).all(dim=1)
        applied = torch.where(valid[:, None], applied, torch.zeros_like(applied))

    candidate_p = project_native(candidate_recurrent, basis)
    candidate_q = candidate_recurrent - candidate_p
    if target == "candidate_p":
        modified_recurrent = candidate_q + fourier_shift_2d(candidate_p, applied)
    elif target == "complementary_q":
        modified_recurrent = candidate_p + fourier_shift_2d(candidate_q, applied)
    else:
        raise ValueError(f"Unknown realignment target: {target}")
    # Preserve literal arithmetic at zero shift and change only recurrent
    # candidate evidence for nonzero shifts.
    candidate_pre = candidate_literal + (modified_recurrent - candidate_recurrent)
    candidate = torch.tanh(candidate_pre)
    h_t = (1.0 - z) * h_previous + z * candidate
    return h_t, applied, valid


def replay_realignment(
    cell: torch.nn.Module,
    sequence: torch.Tensor,
    basis: torch.Tensor,
    *,
    shift_yx_by_step: torch.Tensor | None,
    target: RealignmentTarget,
    oracle: bool = False,
    max_oracle_lag_px: int = 4,
) -> RealignmentReplay:
    """Replay an eight-step within-window candidate realignment condition."""
    if sequence.ndim != 5:
        raise ValueError(f"Expected BCTHW sequence, found {tuple(sequence.shape)}")
    if not oracle and target != "none":
        if shift_yx_by_step is None:
            raise ValueError("Non-oracle realignment requires B,T,2 shifts")
        shifts = torch.as_tensor(
            shift_yx_by_step, dtype=sequence.dtype, device=sequence.device
        )
        if shifts.shape != (sequence.shape[0], sequence.shape[2], 2):
            raise ValueError(
                f"Expected shifts {(sequence.shape[0], sequence.shape[2], 2)}, "
                f"found {tuple(shifts.shape)}"
            )
    else:
        shifts = None
    state: torch.Tensor | None = None
    outputs: list[torch.Tensor] = []
    applied: list[torch.Tensor] = []
    valid: list[torch.Tensor] = []
    for internal_step in range(int(sequence.shape[2])):
        # There is no recurrent evidence at step zero.  Bypass the
        # intervention before argument validation so an oracle replay (which
        # intentionally has no requested shift tensor) retains the literal
        # zero-state endpoint instead of raising or running an FFT.
        if internal_step == 0 and target != "none":
            state = cell(sequence[:, :, internal_step], None)
            actual = torch.zeros(
                (sequence.shape[0], 2), dtype=sequence.dtype, device=sequence.device
            )
            step_valid = torch.ones(
                sequence.shape[0], dtype=torch.bool, device=sequence.device
            )
            outputs.append(state)
            applied.append(actual)
            valid.append(step_valid)
            continue
        requested = None if shifts is None else shifts[:, internal_step]
        state, actual, step_valid = realignment_step(
            cell,
            sequence[:, :, internal_step],
            state,
            basis,
            shift_yx_px=requested,
            target=target,
            oracle=oracle and internal_step > 0,
            max_oracle_lag_px=max_oracle_lag_px,
        )
        outputs.append(state)
        applied.append(actual)
        valid.append(step_valid)
    return RealignmentReplay(
        sequence=torch.stack(outputs, dim=2),
        applied_shift_yx_px=torch.stack(applied, dim=1),
        oracle_valid=torch.stack(valid, dim=1),
    )


def matched_random_directions(
    shifts_yx: np.ndarray,
    *,
    labels: Sequence[str],
    seed: int = ANALYSIS_SEED + 911,
) -> np.ndarray:
    """Deterministic random directions with each requested magnitude retained."""
    values = np.asarray(shifts_yx, dtype=np.float64)
    if values.shape[-1] != 2 or values.shape[:-1] != (len(labels),):
        raise ValueError("Random-direction labels must match an N by 2 shift array")
    result = np.zeros_like(values)
    for index, label in enumerate(labels):
        digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
        integer = int.from_bytes(digest[:8], "little")
        angle = np.random.default_rng(integer).uniform(0.0, 2.0 * math.pi)
        magnitude = float(np.linalg.norm(values[index]))
        result[index] = magnitude * np.asarray([math.sin(angle), math.cos(angle)])
    return result.astype(np.float32)
