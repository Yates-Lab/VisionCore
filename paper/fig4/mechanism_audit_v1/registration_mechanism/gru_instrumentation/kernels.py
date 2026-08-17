"""P/Q offset-energy audit for the frozen ConvGRU recurrent kernels."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch


RECURRENT_PARAMETER_NAMES: dict[str, str] = {
    "candidate": "model.recurrent.cells.0.out_gate.weight",
    "reset_gate": "model.recurrent.cells.0.reset_gate.weight",
    "update_gate": "model.recurrent.cells.0.update_gate.weight",
}


def validate_basis(basis: np.ndarray | torch.Tensor, *, channels: int = 128, rank: int = 8) -> np.ndarray:
    value = np.asarray(basis, dtype=np.float64)
    if value.shape != (channels, rank):
        raise ValueError(f"Expected basis {(channels, rank)}, found {value.shape}")
    if not np.allclose(value.T @ value, np.eye(rank), atol=2e-5, rtol=2e-5):
        raise ValueError("Projector basis is not orthonormal")
    return value


def recurrent_kernel_half(weight: np.ndarray | torch.Tensor, *, hidden_channels: int = 128) -> np.ndarray:
    """Return K_h from a kernel over ``cat([x,h])`` after checking layout."""
    value = np.asarray(weight, dtype=np.float64)
    if value.ndim != 4 or value.shape[0] != hidden_channels:
        raise ValueError(f"Unexpected ConvGRU kernel shape {value.shape}")
    if value.shape[1] <= hidden_channels:
        raise ValueError("ConvGRU kernel has no distinct current-input block")
    recurrent = value[:, -hidden_channels:, :, :]
    if recurrent.shape != (hidden_channels, hidden_channels, value.shape[2], value.shape[3]):
        raise ValueError("Hidden-to-hidden kernel split failed")
    return recurrent


def pq_offset_energies(kernel_matrix: np.ndarray, basis: np.ndarray) -> dict[str, float]:
    """Compute the four orthogonal Frobenius-energy blocks without Q itself."""
    k = np.asarray(kernel_matrix, dtype=np.float64)
    raw_basis = np.asarray(basis)
    if raw_basis.ndim != 2:
        raise ValueError(f"Expected a matrix basis, found {raw_basis.shape}")
    u = validate_basis(raw_basis, channels=k.shape[0], rank=raw_basis.shape[1])
    if k.shape != (u.shape[0], u.shape[0]):
        raise ValueError(f"Expected square hidden-channel matrix, found {k.shape}")
    utu = u.T @ k @ u
    pp = float(np.square(utu).sum())
    # ||P K||_F^2 = ||U^T K||_F^2 and ||K P||_F^2 = ||K U||_F^2.
    pk = float(np.square(u.T @ k).sum())
    kp = float(np.square(k @ u).sum())
    total = float(np.square(k).sum())
    pq = max(pk - pp, 0.0)
    qp = max(kp - pp, 0.0)
    qq = max(total - pp - pq - qp, 0.0)
    return {
        "energy_pp": pp,
        "energy_pq": pq,
        "energy_qp": qp,
        "energy_qq": qq,
        "energy_total": total,
        "decomposition_residual": total - (pp + pq + qp + qq),
    }


def kernel_offset_rows(
    recurrent_kernel: np.ndarray,
    basis: np.ndarray,
    *,
    kernel_name: str,
    projector_label: str,
    contrast: str,
    fold: int,
) -> list[dict[str, object]]:
    """One auditable energy row per spatial cross-correlation offset."""
    kernel = np.asarray(recurrent_kernel, dtype=np.float64)
    if kernel.ndim != 4 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"Expected Cout,Cin,Ky,Kx recurrent kernel, found {kernel.shape}")
    height, width = map(int, kernel.shape[-2:])
    if height % 2 != 1 or width % 2 != 1:
        raise ValueError("ConvGRU spatial kernel must have an odd center tap")
    center_y, center_x = height // 2, width // 2
    rows: list[dict[str, object]] = []
    for ky in range(height):
        for kx in range(width):
            energy = pq_offset_energies(kernel[:, :, ky, kx], basis)
            dy = ky - center_y
            dx = kx - center_x
            rows.append(
                {
                    "contrast": contrast,
                    "fold": int(fold),
                    "projector": projector_label,
                    "kernel": kernel_name,
                    # PyTorch Conv2d uses cross-correlation.  This is the input
                    # sampling offset relative to the output location.
                    "input_offset_y": int(dy),
                    "input_offset_x": int(dx),
                    "is_center": bool(dy == 0 and dx == 0),
                    **energy,
                }
            )
    return rows


def add_kernel_summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Attach center/off-center fractions and directional energy moments."""
    if not rows:
        return rows
    energy_names = ("energy_pp", "energy_pq", "energy_qp", "energy_qq", "energy_total")
    totals = {name: sum(float(row[name]) for row in rows) for name in energy_names}
    centers = {
        name: sum(float(row[name]) for row in rows if bool(row["is_center"]))
        for name in energy_names
    }
    off = {name: max(totals[name] - centers[name], 0.0) for name in energy_names}
    n_offcenter = max(sum(not bool(row["is_center"]) for row in rows), 1)
    for row in rows:
        dy = float(row["input_offset_y"])
        dx = float(row["input_offset_x"])
        for name in energy_names:
            prefix = name.removeprefix("energy_")
            row[f"{prefix}_center_fraction"] = centers[name] / max(totals[name], 1e-30)
            row[f"{prefix}_offcenter_fraction"] = off[name] / max(totals[name], 1e-30)
            row[f"{prefix}_center_energy"] = centers[name]
            row[f"{prefix}_offcenter_total_energy"] = off[name]
            row[f"{prefix}_offcenter_mean_energy_per_tap"] = off[name] / n_offcenter
            row[f"{prefix}_center_to_offcenter_mean_ratio"] = centers[name] / max(
                off[name] / n_offcenter, 1e-30
            )
            row[f"{prefix}_offcenter_direction_moment_y"] = (
                sum(
                    float(other[name]) * float(other["input_offset_y"])
                    for other in rows
                    if not bool(other["is_center"])
                )
                / max(off[name], 1e-30)
            )
            row[f"{prefix}_offcenter_direction_moment_x"] = (
                sum(
                    float(other[name]) * float(other["input_offset_x"])
                    for other in rows
                    if not bool(other["is_center"])
                )
                / max(off[name], 1e-30)
            )
        opposite = next(
            (
                other
                for other in rows
                if int(other["input_offset_y"]) == -int(dy)
                and int(other["input_offset_x"]) == -int(dx)
            ),
            None,
        )
        for name in energy_names:
            prefix = name.removeprefix("energy_")
            row[f"{prefix}_opposite_offset_asymmetry"] = (
                0.0 if opposite is None else float(row[name]) - float(opposite[name])
            )
            row[f"{prefix}_opposite_offset_normalized_asymmetry"] = (
                0.0
                if opposite is None
                else (float(row[name]) - float(opposite[name]))
                / max(float(row[name]) + float(opposite[name]), 1e-30)
            )
    return rows


def analyze_checkpoint_kernels(
    state_dict: Mapping[str, torch.Tensor],
    projectors: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Analyze candidate/reset/update hidden kernels for all supplied bases."""
    output: list[dict[str, object]] = []
    for projector in projectors:
        basis = validate_basis(np.asarray(projector["basis"]))
        for label, parameter_name in RECURRENT_PARAMETER_NAMES.items():
            if parameter_name not in state_dict:
                raise KeyError(f"Frozen checkpoint lacks {parameter_name}")
            recurrent = recurrent_kernel_half(state_dict[parameter_name].detach().cpu().numpy())
            rows = kernel_offset_rows(
                recurrent,
                basis,
                kernel_name=label,
                projector_label=str(projector.get("projector", "learned_p")),
                contrast=str(projector["contrast"]),
                fold=int(projector["fold"]),
            )
            output.extend(add_kernel_summaries(rows))
    return output


def write_kernel_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Refusing to write an empty kernel audit")
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
