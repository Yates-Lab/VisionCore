from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import torch

from scripts.mcfarland_sim import eye_deg_to_norm

StoredEyeConvention = Literal["visual_xy", "visual_x_negy"]


DEFAULT_STORED_EYE_CONVENTION: StoredEyeConvention = "visual_xy"


def _as_eye_array(eyepos: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(eyepos, torch.Tensor):
        arr = eyepos.detach().cpu().numpy()
    else:
        arr = np.asarray(eyepos)
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError(f"Expected 2 values for 1D eye position, got shape {arr.shape}")
        arr = arr.reshape(1, 2)
    if arr.shape[-1] != 2:
        raise ValueError(f"Expected eye positions with trailing dimension 2, got shape {arr.shape}")
    return np.asarray(arr, dtype=np.float32)


def dataset_eyepos_to_visual_deg(
    eyepos: np.ndarray | torch.Tensor,
    stored_convention: StoredEyeConvention = DEFAULT_STORED_EYE_CONVENTION,
) -> np.ndarray:
    arr = _as_eye_array(eyepos).copy()
    if stored_convention == "visual_xy":
        return arr
    if stored_convention == "visual_x_negy":
        arr[..., 1] *= -1.0
        return arr
    raise ValueError(f"Unknown stored eye convention: {stored_convention}")


def stored_eyepos_to_eye_norm(
    eyepos: np.ndarray | torch.Tensor,
    ppd: float,
    img_size: Sequence[int],
    stored_convention: StoredEyeConvention = DEFAULT_STORED_EYE_CONVENTION,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    visual_deg = dataset_eyepos_to_visual_deg(eyepos, stored_convention=stored_convention)
    eye_tensor = torch.as_tensor(visual_deg, dtype=torch.float32)
    if device is not None:
        eye_tensor = eye_tensor.to(device)
    return eye_deg_to_norm(eye_tensor, float(ppd), tuple(int(v) for v in img_size[-2:]))


def visual_eye_deg_to_canonical_shift_px(
    eyepos_visual_deg: np.ndarray | torch.Tensor,
    ppd: float,
) -> np.ndarray:
    visual_deg = _as_eye_array(eyepos_visual_deg)
    out = np.empty_like(visual_deg, dtype=np.float32)
    out[..., 0] = visual_deg[..., 0] * float(ppd)
    out[..., 1] = -visual_deg[..., 1] * float(ppd)
    return out


def stored_eyepos_to_canonical_shift_px(
    eyepos: np.ndarray | torch.Tensor,
    ppd: float,
    stored_convention: StoredEyeConvention = DEFAULT_STORED_EYE_CONVENTION,
) -> np.ndarray:
    visual_deg = dataset_eyepos_to_visual_deg(eyepos, stored_convention=stored_convention)
    return visual_eye_deg_to_canonical_shift_px(visual_deg, ppd)
