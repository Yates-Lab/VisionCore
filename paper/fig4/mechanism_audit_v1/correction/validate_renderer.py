#!/usr/bin/env python3
"""Validate corrected retinal coordinates against the legacy renderer where comparable."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.mechanism_audit_v1.correction.common import BANK_DIR, LEGACY_MATRIX_DIR, OUT_DIR, write_json
from paper.fig4.mechanism_audit_v1.correction.scoring import make_corrected_causal_stims
from paper.fig4.upstream.real_trace_matrix.core import extract_patch
from paper.fig4.upstream.real_trace_matrix.model import (
    N_LAGS,
    OUT_SIZE,
    PPD,
    _standardize_uint_like,
    _trace_xy_to_twin_helper_order,
    make_counterfactual_stim,
)


def main() -> int:
    import torch

    images = pd.read_csv(LEGACY_MATRIX_DIR / "image_feature_table.csv").sort_values("image_index").reset_index(drop=True)
    patch, _ = extract_patch(images.iloc[0], canvas_cache={}, patch_size_px=540)
    image = _standardize_uint_like(patch)
    with np.load(BANK_DIR / "corrected_history_trajectory_banks.npz") as archive:
        true = np.asarray(archive["true_history_xy"][:1], dtype=np.float32)
        held = np.asarray(archive["held_initial_history_xy"][:1], dtype=np.float32)
        scored = np.asarray(archive["stored_scored_trace_xy"][0], dtype=np.float32)
    new_true = make_corrected_causal_stims(image, true, torch=torch)
    new_held = make_corrected_causal_stims(image, held, torch=torch)
    full_stack = np.broadcast_to(
        image[None], (scored.shape[0] + N_LAGS + 1, *image.shape)
    ).copy()
    legacy_eye = torch.from_numpy(_trace_xy_to_twin_helper_order(scored))
    legacy = make_counterfactual_stim(
        full_stack,
        legacy_eye,
        ppd=PPD,
        scale_factor=1.0,
        n_lags=N_LAGS,
        out_size=OUT_SIZE,
    )
    if tuple(new_true.shape) != (40, 1, 32, 151, 151):
        raise AssertionError(new_true.shape)
    if tuple(legacy.shape) != (41, 1, 32, 151, 151):
        raise AssertionError(legacy.shape)
    # Corrected outputs 31:39 and legacy embedded outputs 32:40 use exactly
    # the same scored source samples 0:39.  Equality proves coordinate and lag
    # ordering are unchanged where the histories genuinely overlap.
    overlap_error = float(torch.max(torch.abs(new_true[31:] - legacy[32:])).item())
    held_initial_lag_error = float(
        torch.max(torch.abs(new_held[0, :, 1:] - new_held[0, :, :1])).item()
    )
    if overlap_error != 0.0 or held_initial_lag_error != 0.0:
        raise AssertionError(
            {"overlap_error": overlap_error, "held_initial_lag_error": held_initial_lag_error}
        )
    write_json(
        OUT_DIR / "renderer_validation.json",
        {
            "actual_image_index": 0,
            "actual_trajectory_index": 0,
            "corrected_stimulus_shape": list(new_true.shape),
            "legacy_stimulus_shape": list(legacy.shape),
            "corrected_vs_legacy_valid_overlap_max_abs_pixel_error": overlap_error,
            "held_prefix_first_output_across_lag_max_abs_pixel_error": held_initial_lag_error,
            "coordinate_convention": "input histories are [x_deg,y_deg]; identical to legacy after its preflip+internal-flip pair",
            "validation_conclusion": "rendered image coordinates and lag order match exactly where source histories overlap",
        },
    )
    print("Corrected renderer validated exactly on the nine uncontaminated legacy outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
