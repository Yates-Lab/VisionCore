"""Spatial-registration estimators for exact ConvGRU update terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .equations import complementary_native, project_coordinates


@dataclass(frozen=True)
class CorrelationPeak:
    """Batched normalized multi-channel cross-correlation result.

    ``lag_*`` follows the protocol's sampling convention
    ``sum A(x,y) B(x+dx,y+dy)``.  Thus, if B is a copy of A whose content was
    displaced right by two pixels, ``lag_x`` is +2 pixels.
    """

    zero_lag_correlation: torch.Tensor
    best_lag_correlation: torch.Tensor
    lag_y_px: torch.Tensor
    lag_x_px: torch.Tensor
    peak_sharpness: torch.Tensor
    residual_mismatch: torch.Tensor
    valid: torch.Tensor


def _as_bchw(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3:
        return value.unsqueeze(0)
    if value.ndim != 4:
        raise ValueError(f"Expected CHW or BCHW map, found {tuple(value.shape)}")
    return value


def normalized_multichannel_xcorr(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    max_lag_px: int = 4,
    subpixel_factor: int = 4,
    variance_epsilon: float = 1e-12,
) -> CorrelationPeak:
    """Bounded Fourier cross-correlation with subpixel peak localization.

    Each channel is spatially mean-centered independently.  Zero padding makes
    the evaluated integer correlations linear rather than circular.  The
    cross-spectrum is zero-padded in frequency before inversion, producing an
    explicit bounded Fourier subpixel grid (one quarter pixel by default).

    The denominator is the product of the two complete centered-map norms,
    exactly matching the normalization written in the audit protocol.  Search
    lags are small relative to the 64x64 map, and zero padding prevents wrap.
    """
    a = _as_bchw(a).float()
    b = _as_bchw(b).float()
    if a.shape != b.shape:
        raise ValueError(f"Cross-correlation shape mismatch: {a.shape} != {b.shape}")
    if max_lag_px < 1:
        raise ValueError("max_lag_px must be at least one")
    if int(subpixel_factor) < 1:
        raise ValueError("subpixel_factor must be positive")
    subpixel_factor = int(subpixel_factor)
    a = a - a.mean(dim=(-2, -1), keepdim=True)
    b = b - b.mean(dim=(-2, -1), keepdim=True)
    height, width = map(int, a.shape[-2:])
    if max_lag_px >= min(height, width) // 2:
        raise ValueError("max_lag_px must be smaller than half the map width")
    padded_shape = (2 * height, 2 * width)
    spectrum_a = torch.fft.rfft2(a, s=padded_shape)
    spectrum_b = torch.fft.rfft2(b, s=padded_shape)
    # conj(A) B yields sum A(x,y) B(x+dx,y+dy) at positive array lag.
    cross_spectrum = (spectrum_a.conj() * spectrum_b).sum(dim=1)
    # Band-limited interpolation of the correlation surface: pad the centered
    # full-frequency y axis and the nonnegative rFFT x axis, then invert at the
    # enlarged size.  Multiplication by factor^2 restores the original FFT
    # normalization at integer sample locations.
    if subpixel_factor > 1:
        shifted = torch.fft.fftshift(cross_spectrum, dim=(-2,))
        target_y = padded_shape[0] * subpixel_factor
        target_x_half = padded_shape[1] * subpixel_factor // 2 + 1
        pad_y = target_y - shifted.shape[-2]
        pad_top = pad_y // 2
        shifted = F.pad(
            shifted,
            (0, target_x_half - shifted.shape[-1], pad_top, pad_y - pad_top),
        )
        cross_spectrum = torch.fft.ifftshift(shifted, dim=(-2,))
    upsampled_shape = (
        padded_shape[0] * subpixel_factor,
        padded_shape[1] * subpixel_factor,
    )
    correlation = torch.fft.irfft2(cross_spectrum, s=upsampled_shape)
    correlation = correlation * float(subpixel_factor**2)
    denominator = torch.sqrt(
        a.square().sum(dim=(1, 2, 3)) * b.square().sum(dim=(1, 2, 3))
    )
    valid = denominator > float(variance_epsilon)
    correlation = correlation / denominator.clamp_min(float(variance_epsilon))[:, None, None]

    lag_ticks = torch.arange(
        -int(max_lag_px) * subpixel_factor,
        int(max_lag_px) * subpixel_factor + 1,
        device=a.device,
        dtype=torch.long,
    )
    y_index = torch.remainder(lag_ticks, upsampled_shape[0])
    x_index = torch.remainder(lag_ticks, upsampled_shape[1])
    window = correlation.index_select(1, y_index).index_select(2, x_index)
    window_flat = window.flatten(1)
    best_index = torch.argmax(window_flat, dim=1)
    side = int(window.shape[-1])
    iy = torch.div(best_index, side, rounding_mode="floor")
    ix = torch.remainder(best_index, side)
    batch = torch.arange(len(a), device=a.device)
    c0 = window[batch, iy, ix]

    # Exclude a one-feature-pixel peak neighborhood for a nonlocal sharpness
    # measure, independent of the chosen subpixel grid density.
    gy = torch.arange(side, device=a.device)[None, :, None]
    gx = torch.arange(side, device=a.device)[None, None, :]
    near = (gy - iy[:, None, None]).abs().le(subpixel_factor) & (
        gx - ix[:, None, None]
    ).abs().le(subpixel_factor)
    second = window.masked_fill(near, -torch.inf).flatten(1).max(dim=1).values
    second = torch.where(torch.isfinite(second), second, c0)

    best_value = c0
    lag_y = lag_ticks[iy].to(a.dtype) / float(subpixel_factor)
    lag_x = lag_ticks[ix].to(a.dtype) / float(subpixel_factor)
    zero_index = int(max_lag_px) * subpixel_factor
    zero = window[:, zero_index, zero_index]
    nan = torch.full_like(zero, torch.nan)
    return CorrelationPeak(
        zero_lag_correlation=torch.where(valid, zero, nan),
        best_lag_correlation=torch.where(valid, best_value, nan),
        lag_y_px=torch.where(valid, lag_y, nan),
        lag_x_px=torch.where(valid, lag_x, nan),
        peak_sharpness=torch.where(valid, best_value - second, nan),
        residual_mismatch=torch.where(valid, 1.0 - best_value, nan),
        valid=valid,
    )


def feature_views(value: torch.Tensor, learned_basis: torch.Tensor) -> dict[str, torch.Tensor]:
    """Produce candidate-P and complementary-Q maps for registration."""
    return {
        "learned_p": project_coordinates(value, learned_basis),
        # Q has no privileged 120-axis basis.  Keeping the exact native-space
        # residual is coordinate-invariant and still compact because it is not
        # persisted, only reduced online.
        "learned_q": complementary_native(value, learned_basis),
    }


def registration_step_metrics(
    current: torch.Tensor,
    h_previous: torch.Tensor,
    recurrent: torch.Tensor,
    bases: Mapping[str, torch.Tensor],
    *,
    max_lag_px: int = 4,
) -> dict[str, dict[str, torch.Tensor]]:
    """Compare raw and recurrently transformed retained evidence to current."""
    result: dict[str, dict[str, torch.Tensor]] = {}
    batch_size = int(current.shape[0])

    def add_result(
        name: str,
        raw: CorrelationPeak,
        transformed: CorrelationPeak,
        selection: slice,
    ) -> None:
        result[name] = {
            "raw_zero_lag_correlation": raw.zero_lag_correlation[selection],
            "raw_best_lag_correlation": raw.best_lag_correlation[selection],
            "raw_lag_y_px": raw.lag_y_px[selection],
            "raw_lag_x_px": raw.lag_x_px[selection],
            "raw_peak_sharpness": raw.peak_sharpness[selection],
            "raw_residual_mismatch": raw.residual_mismatch[selection],
            "recurrent_zero_lag_correlation": transformed.zero_lag_correlation[selection],
            "recurrent_best_lag_correlation": transformed.best_lag_correlation[selection],
            "recurrent_lag_y_px": transformed.lag_y_px[selection],
            "recurrent_lag_x_px": transformed.lag_x_px[selection],
            "recurrent_peak_sharpness": transformed.peak_sharpness[selection],
            "recurrent_residual_mismatch": transformed.residual_mismatch[selection],
            # Content displacement applied to the retained evidence.  A raw
            # sampling lag +d is corrected by a content shift -d, hence
            # recurrent residual minus raw lag.  This sign matches the
            # synthetic calibration's current-minus-previous feature shift.
            "transport_y_px": transformed.lag_y_px[selection] - raw.lag_y_px[selection],
            "transport_x_px": transformed.lag_x_px[selection] - raw.lag_x_px[selection],
            "zero_lag_alignment_improvement": (
                transformed.zero_lag_correlation[selection] - raw.zero_lag_correlation[selection]
            ),
            "best_lag_alignment_improvement": (
                transformed.best_lag_correlation[selection] - raw.best_lag_correlation[selection]
            ),
            "valid": raw.valid[selection] & transformed.valid[selection],
        }

    # All rank-k coordinate projections share the same channel count, so they
    # can be concatenated on the batch axis and transformed by four FFT calls
    # total per step (two for P-like views, two for Q-like views), rather than
    # one FFT triplet per subspace.  Namespaced ``...::learned_q`` keys are
    # recognized so all contrasts can be reduced together.
    groups: dict[str, list[tuple[str, torch.Tensor]]] = {"p": [], "q": []}
    for name, basis in bases.items():
        groups["q" if name.split("::")[-1] == "learned_q" else "p"].append((name, basis))
    for kind, entries in groups.items():
        if not entries:
            continue
        if kind == "p":
            current_views = [project_coordinates(current, basis) for _, basis in entries]
            previous_views = [project_coordinates(h_previous, basis) for _, basis in entries]
            recurrent_views = [project_coordinates(recurrent, basis) for _, basis in entries]
        else:
            current_views = [complementary_native(current, basis) for _, basis in entries]
            previous_views = [complementary_native(h_previous, basis) for _, basis in entries]
            recurrent_views = [complementary_native(recurrent, basis) for _, basis in entries]
        raw = normalized_multichannel_xcorr(
            torch.cat(current_views, dim=0),
            torch.cat(previous_views, dim=0),
            max_lag_px=max_lag_px,
        )
        transformed = normalized_multichannel_xcorr(
            torch.cat(current_views, dim=0),
            torch.cat(recurrent_views, dim=0),
            max_lag_px=max_lag_px,
        )
        for index, (name, _) in enumerate(entries):
            selection = slice(index * batch_size, (index + 1) * batch_size)
            add_result(name, raw, transformed, selection)
    return result


def fit_shift_calibration(
    eye_displacement_deg: np.ndarray,
    measured_feature_lag_px: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Fit the empirical eye-degree -> feature-pixel map including sign/cross-talk."""
    eye = np.asarray(eye_displacement_deg, dtype=np.float64)
    feature = np.asarray(measured_feature_lag_px, dtype=np.float64)
    if eye.ndim != 2 or feature.shape != eye.shape or eye.shape[1] != 2:
        raise ValueError("Calibration arrays must both have shape (observations, 2)")
    design = np.column_stack([eye, np.ones(len(eye))])
    coefficients, _, _, _ = np.linalg.lstsq(design, feature, rcond=None)
    predicted = design @ coefficients
    residual = feature - predicted
    sse = np.square(residual).sum(axis=0)
    centered = feature - feature.mean(axis=0, keepdims=True)
    sst = np.square(centered).sum(axis=0)
    r2 = 1.0 - sse / np.maximum(sst, 1e-30)
    return {
        "matrix_feature_px_per_eye_deg": coefficients[:2].T,
        "intercept_feature_px": coefficients[2],
        "r2_by_feature_component": r2,
        "median_vector_error_px": float(np.median(np.linalg.norm(residual, axis=1))),
    }


def apply_shift_calibration(
    eye_displacement_deg: np.ndarray,
    matrix_feature_px_per_eye_deg: np.ndarray,
) -> np.ndarray:
    eye = np.asarray(eye_displacement_deg, dtype=np.float64)
    matrix = np.asarray(matrix_feature_px_per_eye_deg, dtype=np.float64)
    if matrix.shape != (2, 2) or eye.shape[-1] != 2:
        raise ValueError("Expected (...,2) eye displacement and a 2x2 calibration matrix")
    return eye @ matrix.T


# Exact union of retinal-lag supports derived from this checkpoint's executed
# frontend/ResNet graph (learned temporal k=16; RB1 temporal k=3 + MaxPool3d
# stride 2 and cropped stride-2 shortcut; RB2 temporal k=3).
FIG4_CONVGRU_INPUT_LAG_SUPPORTS: tuple[tuple[int, int], ...] = (
    (0, 17),
    (0, 19),
    (0, 21),
    (0, 23),
    (2, 25),
    (4, 27),
    (6, 29),
    (8, 31),
)


def derive_fig4_convgru_input_lag_supports(model_config: Mapping[str, object]) -> tuple[tuple[int, int], ...]:
    """Propagate exact temporal dependencies through the frozen core graph.

    This intentionally fails if any architecture value relevant to the eight
    supports changes; it is an audit of this checkpoint rather than a generic
    receptive-field calculator.
    """
    frontend = dict(model_config["frontend"])["params"]  # type: ignore[index]
    convnet = dict(model_config["convnet"])["params"]  # type: ignore[index]
    recurrent = dict(model_config["recurrent"])["params"]  # type: ignore[index]
    if int(frontend["kernel_size"]) != 16:  # type: ignore[index]
        raise ValueError("Frozen temporal frontend kernel is no longer 16")
    blocks = convnet["block_configs"]  # type: ignore[index]
    if len(blocks) != 2:
        raise ValueError("Frozen ResNet no longer has exactly two blocks")
    temporal_kernels = [int(block["conv_params"]["kernel_size"][0]) for block in blocks]
    pool = blocks[0]["pool_params"]
    if temporal_kernels != [3, 3] or int(pool["kernel_size"]) != 2 or int(pool["stride"]) != 2:
        raise ValueError("Frozen ResNet temporal convolution/pooling contract changed")
    if int(recurrent["n_layers"]) != 1 or int(recurrent["hidden_dim"]) != 128:  # type: ignore[index]
        raise ValueError("Frozen ConvGRU depth/width contract changed")

    # Each set contains retinal lag indices contributing to one time position.
    support: list[set[int]] = [set(range(q, q + 16)) for q in range(17)]

    def causal_conv(sequence: list[set[int]], kernel: int) -> list[set[int]]:
        return [
            set().union(*(sequence[max(0, q - offset)] for offset in range(kernel)))
            for q in range(len(sequence))
        ]

    rb1_main_prepool = causal_conv(support, 3)
    rb1_main = [
        rb1_main_prepool[2 * q] | rb1_main_prepool[2 * q + 1]
        for q in range(8)
    ]
    rb1_shortcut_before_crop = [support[2 * q] for q in range(9)]
    rb1_shortcut = rb1_shortcut_before_crop[-8:]
    rb1 = [main | shortcut for main, shortcut in zip(rb1_main, rb1_shortcut)]
    rb2_main = causal_conv(rb1, 3)
    rb2 = [main | shortcut for main, shortcut in zip(rb2_main, rb1)]
    result = tuple((min(value), max(value)) for value in rb2)
    if result != FIG4_CONVGRU_INPUT_LAG_SUPPORTS:
        raise RuntimeError(f"Derived supports changed unexpectedly: {result}")
    return result


def support_midpoint_lags(
    supports: tuple[tuple[int, int], ...] = FIG4_CONVGRU_INPUT_LAG_SUPPORTS,
) -> np.ndarray:
    """Nominal structural anchor for composite evidence at each GRU step."""
    return np.asarray([(lo + hi) / 2.0 for lo, hi in supports], dtype=np.float64)


def _linear_sample_history(history: np.ndarray, index: float) -> np.ndarray:
    lo = int(np.floor(index))
    hi = int(np.ceil(index))
    if lo < 0 or hi >= len(history):
        raise IndexError(f"Fractional history index {index} lies outside [0,{len(history)-1}]")
    if lo == hi:
        return np.asarray(history[lo], dtype=np.float64)
    weight = float(index - lo)
    return (1.0 - weight) * history[lo] + weight * history[hi]


def internal_step_eye_displacements(
    history_xy_deg: np.ndarray,
    scored_frame: int,
    *,
    current_history_index_at_frame_zero: int = 31,
    anchors_lag: np.ndarray | None = None,
) -> np.ndarray:
    """Eye displacement from previous to current *internal* evidence support.

    The scored windows are current-to-oldest.  Consequently increasing GRU
    step follows evidence backward in retinal time.  Row zero is NaN because
    the cell has no preceding recurrent evidence.  This is intentionally not
    mislabeled as recurrence across the 40 scored outputs.
    """
    history = np.asarray(history_xy_deg, dtype=np.float64)
    if history.shape != (71, 2):
        raise ValueError(f"Expected one corrected (71,2) history, found {history.shape}")
    anchors = support_midpoint_lags() if anchors_lag is None else np.asarray(anchors_lag, dtype=np.float64)
    current_index = int(current_history_index_at_frame_zero) + int(scored_frame)
    eye_at_anchor = np.stack(
        [_linear_sample_history(history, current_index - float(lag)) for lag in anchors]
    )
    displacement = np.full_like(eye_at_anchor, np.nan)
    displacement[1:] = eye_at_anchor[1:] - eye_at_anchor[:-1]
    return displacement


def vector_transport_statistics(
    measured_transport: np.ndarray,
    expected_retinal_feature_shift: np.ndarray,
) -> dict[str, float]:
    """Component regressions and vector agreement required by section 8.4."""
    measured = np.asarray(measured_transport, dtype=np.float64)
    expected = np.asarray(expected_retinal_feature_shift, dtype=np.float64)
    if measured.shape != expected.shape or measured.ndim != 2 or measured.shape[1] != 2:
        raise ValueError("Transport arrays must have matching shape (observations,2)")
    valid = np.isfinite(measured).all(axis=1) & np.isfinite(expected).all(axis=1)
    measured = measured[valid]
    expected = expected[valid]
    if len(measured) < 3:
        return {key: float("nan") for key in (
            "slope_x", "intercept_x", "r2_x", "slope_y", "intercept_y", "r2_y",
            "vector_correlation", "median_displacement_error_px", "vector_variance_explained",
        )}

    def regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        design = np.column_stack([x, np.ones(len(x))])
        coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        prediction = design @ coef
        sse = float(np.square(y - prediction).sum())
        sst = float(np.square(y - y.mean()).sum())
        return float(coef[0]), float(coef[1]), float(1.0 - sse / max(sst, 1e-30))

    sx, ix, r2x = regression(expected[:, 0], measured[:, 0])
    sy, iy, r2y = regression(expected[:, 1], measured[:, 1])
    expected_flat = expected.ravel()
    measured_flat = measured.ravel()
    vector_corr = float(np.corrcoef(expected_flat, measured_flat)[0, 1])
    error = np.linalg.norm(measured - expected, axis=1)
    variance_explained = 1.0 - float(np.square(measured - expected).sum()) / max(
        float(np.square(expected - expected.mean(axis=0, keepdims=True)).sum()), 1e-30
    )
    return {
        "slope_x": sx,
        "intercept_x": ix,
        "r2_x": r2x,
        "slope_y": sy,
        "intercept_y": iy,
        "r2_y": r2y,
        "vector_correlation": vector_corr,
        "median_displacement_error_px": float(np.median(error)),
        "vector_variance_explained": float(variance_explained),
    }
