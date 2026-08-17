"""Literal decomposition of the ConvGRU implementation used by Figure 4.

The names here follow :mod:`models.modules.recurrent`, not a textbook GRU.
In particular, VisionCore's ``update_gate`` is the *candidate-write* fraction:

    h_t = (1 - z_t) h_{t-1} + z_t n_t.

No hook-derived or algebraically simplified surrogate is used.  For each
concatenated convolution, the recurrent contribution is evaluated directly as
the bias-free hidden-kernel operation.  The current contribution is then
defined as the literal concatenated convolution minus that recurrent term.
Consequently current plus recurrent reconstructs the literal preactivation to
numerical precision, and the current term owns both the bias and the small
floating-point cross-accumulation correction of the fused convolution.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GRUStepTerms:
    """All exact terms in one VisionCore ``ConvGRUCell`` update."""

    x_t: torch.Tensor
    h_previous: torch.Tensor
    update_input_preactivation: torch.Tensor
    update_recurrent_preactivation: torch.Tensor
    update_preactivation: torch.Tensor
    update_gate: torch.Tensor
    reset_input_preactivation: torch.Tensor
    reset_recurrent_preactivation: torch.Tensor
    reset_preactivation: torch.Tensor
    reset_gate: torch.Tensor
    reset_hidden: torch.Tensor
    candidate_current_preactivation: torch.Tensor
    candidate_recurrent_preactivation: torch.Tensor
    candidate_preactivation: torch.Tensor
    candidate_state: torch.Tensor
    retained_state_contribution: torch.Tensor
    new_state_contribution: torch.Tensor
    h_t: torch.Tensor

    def hidden_space_terms(self) -> dict[str, torch.Tensor]:
        """Return only 128-channel terms to which P/Q can be applied."""
        excluded = {"x_t"}
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name not in excluded
        }


def split_conv2d_contributions(
    layer: torch.nn.Conv2d,
    x: torch.Tensor,
    hidden: torch.Tensor,
    *,
    hidden_is_last: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose ``layer(cat([x, hidden]))`` around the direct recurrent term.

    The recurrent contribution is exactly ``K_h * hidden`` with no bias.  The
    current contribution is the residual ``literal - recurrent`` and therefore
    owns the bias and any floating-point cross-accumulation correction incurred
    by the literal fused convolution.  The literal value remains canonical for
    all gates and state updates.  The function validates the actual checkpoint
    layout rather than assuming 128 input channels.
    """
    if layer.groups != 1:
        raise ValueError("Exact ConvGRU split requires groups=1")
    if layer.padding_mode != "zeros":
        raise ValueError(
            f"Exact audit expects Conv2d zero padding, found {layer.padding_mode!r}"
        )
    hidden_channels = int(hidden.shape[1])
    input_channels = int(x.shape[1])
    if input_channels + hidden_channels != int(layer.weight.shape[1]):
        raise ValueError(
            "Concatenated channel count does not match ConvGRU kernel: "
            f"{input_channels}+{hidden_channels}!={layer.weight.shape[1]}"
        )
    if not hidden_is_last:
        raise NotImplementedError("The frozen checkpoint concatenates [x, h]")
    hidden_weight = layer.weight[:, input_channels:]
    kwargs = {
        "stride": layer.stride,
        "padding": layer.padding,
        "dilation": layer.dilation,
        "groups": 1,
    }
    recurrent = F.conv2d(hidden, hidden_weight, None, **kwargs)
    # Preserve the literal concatenated convolution for the actual update.
    # cuDNN can accumulate separate input/hidden convolutions in a materially
    # different order.  Assign that cross-accumulation correction to the
    # current residual while keeping the biologically relevant recurrent term
    # equal to the direct bias-free hidden-kernel operation.
    literal = layer(torch.cat([x, hidden], dim=1))
    current = literal - recurrent
    return current, recurrent, literal


def zero_hidden_like(cell: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Construct the exact zero initial state used by ``ConvGRUCell``."""
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW input, found {tuple(x.shape)}")
    hidden_size = int(getattr(cell, "hidden_size"))
    return torch.zeros(
        x.shape[0], hidden_size, x.shape[-2], x.shape[-1],
        dtype=x.dtype, device=x.device,
    )


def instrument_convgru_step(
    cell: torch.nn.Module,
    x_t: torch.Tensor,
    h_previous: torch.Tensor | None,
    *,
    verify: bool = True,
    atol: float = 2e-6,
    rtol: float = 2e-6,
    split_atol: float = 5e-4,
    split_rtol: float = 1e-4,
) -> GRUStepTerms:
    """Evaluate and decompose one update using the checkpoint's conventions."""
    if h_previous is None:
        h_previous = zero_hidden_like(cell, x_t)
    z_x, z_h, z_pre = split_conv2d_contributions(cell.update_gate, x_t, h_previous)
    r_x, r_h, r_pre = split_conv2d_contributions(cell.reset_gate, x_t, h_previous)
    z = torch.sigmoid(z_pre)
    r = torch.sigmoid(r_pre)
    reset_hidden = r * h_previous
    n_x, n_h, n_pre = split_conv2d_contributions(cell.out_gate, x_t, reset_hidden)
    candidate = torch.tanh(n_pre)
    retained = (1.0 - z) * h_previous
    new = z * candidate
    h_t = retained + new
    terms = GRUStepTerms(
        x_t=x_t,
        h_previous=h_previous,
        update_input_preactivation=z_x,
        update_recurrent_preactivation=z_h,
        update_preactivation=z_pre,
        update_gate=z,
        reset_input_preactivation=r_x,
        reset_recurrent_preactivation=r_h,
        reset_preactivation=r_pre,
        reset_gate=r,
        reset_hidden=reset_hidden,
        candidate_current_preactivation=n_x,
        candidate_recurrent_preactivation=n_h,
        candidate_preactivation=n_pre,
        candidate_state=candidate,
        retained_state_contribution=retained,
        new_state_contribution=new,
        h_t=h_t,
    )
    if verify:
        literal = cell(x_t, h_previous)
        if not torch.allclose(h_t, literal, atol=atol, rtol=rtol):
            maximum = float((h_t - literal).abs().max().detach().cpu())
            raise RuntimeError(
                "Instrumented terms do not reconstruct literal ConvGRUCell output; "
                f"max_abs={maximum:.8g}"
            )
        for label, split_sum, literal in (
            ("update", z_x + z_h, z_pre),
            ("reset", r_x + r_h, r_pre),
            ("candidate", n_x + n_h, n_pre),
        ):
            if not torch.allclose(
                split_sum, literal, atol=split_atol, rtol=split_rtol
            ):
                maximum = float((split_sum - literal).abs().max().detach().cpu())
                raise RuntimeError(
                    f"{label} preactivation split is not exact; max_abs={maximum:.8g}"
                )
    return terms


def replay_convgru_cell(
    cell: torch.nn.Module,
    sequence: torch.Tensor,
    hidden: torch.Tensor | None = None,
    *,
    verify: bool = True,
) -> list[GRUStepTerms]:
    """Replay a ``B,C,T,H,W`` sequence and retain its exact step terms."""
    if sequence.ndim != 5:
        raise ValueError(f"Expected BCTHW sequence, found {tuple(sequence.shape)}")
    result: list[GRUStepTerms] = []
    state = hidden
    for step in range(int(sequence.shape[2])):
        terms = instrument_convgru_step(
            cell, sequence[:, :, step], state, verify=verify
        )
        result.append(terms)
        state = terms.h_t
    if verify:
        literal = torch.stack([term.h_t for term in result], dim=2)
        # ConvGRU itself owns a list of cells; here we validate the one-cell
        # temporal recurrence explicitly rather than constructing a wrapper.
        state = hidden
        reference: list[torch.Tensor] = []
        for step in range(int(sequence.shape[2])):
            state = cell(sequence[:, :, step], state)
            reference.append(state)
        reference_tensor = torch.stack(reference, dim=2)
        torch.testing.assert_close(literal, reference_tensor, atol=2e-6, rtol=2e-6)
    return result


def project_coordinates(value: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Return U^T value for a shared channel basis U (C by k)."""
    if value.ndim != 4 or basis.ndim != 2 or value.shape[1] != basis.shape[0]:
        raise ValueError(
            f"Projection shape mismatch: value={tuple(value.shape)}, basis={tuple(basis.shape)}"
        )
    return torch.einsum("ck,bcyx->bkyx", basis, value)


def project_native(value: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Return P value in the native hidden-channel coordinates."""
    coords = project_coordinates(value, basis)
    return torch.einsum("ck,bkyx->bcyx", basis, coords)


def complementary_native(value: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Return Q value=(I-UU^T)value without constructing a 128x128 matrix."""
    return value - project_native(value, basis)


def term_energy_summary(
    terms: GRUStepTerms,
    basis: torch.Tensor,
    names: Iterable[str] | None = None,
    *,
    comparison_bases: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Compact exact energy reductions for hidden-space update terms.

    ``basis`` defines the learned P/Q split.  ``comparison_bases`` adds
    preregistered rank-matched views such as the RR100 readout-SVD basis.  The
    latter are reported as subspace energy only: their orthogonal complements
    are not relabeled Q, which is reserved for the learned projector.
    """
    available = terms.hidden_space_terms()
    selected = tuple(available) if names is None else tuple(names)
    comparisons = {} if comparison_bases is None else dict(comparison_bases)
    output: dict[str, torch.Tensor] = {}
    for name in selected:
        value = available[name]
        p_coords = project_coordinates(value, basis)
        total = value.square().sum(dim=(1, 2, 3))
        p_energy = p_coords.square().sum(dim=(1, 2, 3))
        q_energy = (total - p_energy).clamp_min(0.0)
        output[f"{name}__total_energy"] = total
        output[f"{name}__p_energy"] = p_energy
        output[f"{name}__q_energy"] = q_energy
        output[f"{name}__p_energy_per_dimension"] = p_energy / float(basis.shape[1])
        output[f"{name}__q_energy_per_dimension"] = q_energy / float(
            basis.shape[0] - basis.shape[1]
        )
        for label, comparison_basis in comparisons.items():
            if not label or "__" in label:
                raise ValueError(
                    "Comparison-basis labels must be nonempty and cannot contain '__'"
                )
            comparison_coords = project_coordinates(value, comparison_basis)
            comparison_energy = comparison_coords.square().sum(dim=(1, 2, 3))
            output[f"{name}__{label}_energy"] = comparison_energy
            output[f"{name}__{label}_energy_per_dimension"] = (
                comparison_energy / float(comparison_basis.shape[1])
            )
    return output
