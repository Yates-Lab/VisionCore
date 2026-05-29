"""Trace selection, exact retinal rendering, and local pyramid controls."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .common import DT, N_LAGS, OUT_SIZE, PPD, make_counterfactual_stim
from .eye_controls import detect_microsaccade_events


@dataclass
class TraceExample:
    example_id: str
    kind: str
    source_trace_index: int
    window_start: int
    window_stop: int
    trace: np.ndarray
    threshold_deg_s: float
    n_events_in_window: int
    event_onset: int | None
    event_offset: int | None
    peak_speed_deg_s: float | None
    rms_displacement_deg: float
    path_length_deg: float
    max_speed_deg_s: float


def _trace_stats(trace: np.ndarray, dt: float = DT) -> dict[str, float]:
    tr = np.asarray(trace, dtype=np.float64)
    centered = tr - tr.mean(axis=0, keepdims=True)
    steps = np.diff(tr, axis=0)
    speed = np.linalg.norm(np.diff(tr, axis=0, prepend=tr[:1]), axis=1) / dt
    return {
        "rms_displacement_deg": float(np.sqrt(np.mean(np.sum(centered * centered, axis=1)))),
        "path_length_deg": float(np.sum(np.linalg.norm(steps, axis=1))) if steps.size else 0.0,
        "max_speed_deg_s": float(np.max(speed)) if speed.size else 0.0,
    }


def _overlapping_events(events: list[dict[str, int | float]], start: int, stop: int) -> list[dict[str, int | float]]:
    return [
        event for event in events
        if int(event["offset"]) >= start and int(event["onset"]) < stop
    ]


def select_trace_examples(
    eye_traces: np.ndarray,
    durations: np.ndarray,
    *,
    t_max: int,
    n_each: int,
    seed: int,
    stride: int = 8,
) -> list[TraceExample]:
    """Select real windows with either zero or exactly one detected microsaccade."""
    rng = np.random.default_rng(seed)
    fixation_candidates: list[tuple[float, int, int, np.ndarray, float]] = []
    ms_candidates: list[tuple[float, int, int, np.ndarray, float, dict[str, int | float]]] = []

    for trace_idx, duration in enumerate(durations):
        duration = int(duration)
        if duration < t_max:
            continue
        source = np.asarray(eye_traces[trace_idx, :duration], dtype=np.float32)
        if np.isnan(source).any():
            good = np.where(~np.isnan(source[:, 0]))[0]
            if len(good) < t_max:
                continue
            source = source[good]
            duration = source.shape[0]
        events, event_mask, threshold = detect_microsaccade_events(source, min_samples=1)
        for start in range(0, duration - t_max + 1, stride):
            stop = start + t_max
            window = source[start:stop]
            if window.shape[0] != t_max or np.isnan(window).any():
                continue
            stats = _trace_stats(window)
            overlaps = _overlapping_events(events, start, stop)
            if len(overlaps) == 0 and not bool(np.any(event_mask[start:stop])):
                score = stats["path_length_deg"] + 5.0 * stats["max_speed_deg_s"] * DT + 0.5 * stats["rms_displacement_deg"]
                fixation_candidates.append((score, trace_idx, start, window.copy(), threshold))

        for event in events:
            onset = int(event["onset"])
            start = max(0, min(onset - t_max // 3, duration - t_max))
            stop = start + t_max
            window = source[start:stop]
            if window.shape[0] != t_max or np.isnan(window).any():
                continue
            overlaps = _overlapping_events(events, start, stop)
            if len(overlaps) != 1:
                continue
            rel_onset = int(overlaps[0]["onset"]) - start
            rel_offset = int(overlaps[0]["offset"]) - start
            if rel_onset < 4 or rel_offset > t_max - 5:
                continue
            stats = _trace_stats(window)
            peak = float(overlaps[0]["peak_speed_deg_s"])
            score = -peak + 0.1 * stats["path_length_deg"]
            ms_candidates.append((score, trace_idx, start, window.copy(), threshold, overlaps[0]))

    fixation_candidates.sort(key=lambda x: x[0])
    ms_candidates.sort(key=lambda x: x[0])

    def take_diverse(candidates: list[tuple], n: int) -> list[tuple]:
        chosen = []
        seen: set[int] = set()
        for candidate in candidates:
            source_idx = int(candidate[1])
            if source_idx in seen and len(candidates) >= n:
                continue
            chosen.append(candidate)
            seen.add(source_idx)
            if len(chosen) == n:
                return chosen
        remaining = [c for c in candidates if c not in chosen]
        rng.shuffle(remaining)
        return (chosen + remaining)[:n]

    fixation = take_diverse(fixation_candidates, n_each)
    microsaccade = take_diverse(ms_candidates, n_each)
    if len(fixation) < n_each:
        raise RuntimeError(f"Only found {len(fixation)} fixation-only windows; requested {n_each}")
    if len(microsaccade) < n_each:
        raise RuntimeError(f"Only found {len(microsaccade)} one-microsaccade windows; requested {n_each}")

    out: list[TraceExample] = []
    for i, (_score, trace_idx, start, window, threshold) in enumerate(fixation):
        stats = _trace_stats(window)
        out.append(TraceExample(
            example_id=f"fixation_{i:02d}",
            kind="fixation",
            source_trace_index=int(trace_idx),
            window_start=int(start),
            window_stop=int(start + t_max),
            trace=window.astype(np.float32),
            threshold_deg_s=float(threshold),
            n_events_in_window=0,
            event_onset=None,
            event_offset=None,
            peak_speed_deg_s=None,
            **stats,
        ))
    for i, (_score, trace_idx, start, window, threshold, event) in enumerate(microsaccade):
        stats = _trace_stats(window)
        out.append(TraceExample(
            example_id=f"microsaccade_{i:02d}",
            kind="microsaccade",
            source_trace_index=int(trace_idx),
            window_start=int(start),
            window_stop=int(start + t_max),
            trace=window.astype(np.float32),
            threshold_deg_s=float(threshold),
            n_events_in_window=1,
            event_onset=int(event["onset"]) - int(start),
            event_offset=int(event["offset"]) - int(start),
            peak_speed_deg_s=float(event["peak_speed_deg_s"]),
            **stats,
        ))
    return out


def retinal_movie_from_image_trace(
    image: np.ndarray,
    trace: np.ndarray,
    *,
    t_max: int,
    crop_center_offset_px: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Return the current-frame retinal movie aligned to ``trace[t]``.

    The shared model path returns a lagged tensor.  Lag index 0 is the current
    model frame, and output index 1 corresponds to trace sample 0 because the
    reconstruction prepends lag history.  This function keeps the exact model
    crop/rendering path but removes that display-only lag offset.
    """
    eye_stim = model_lag_tensor_from_image_trace(
        image,
        trace,
        t_max=t_max,
        crop_center_offset_px=crop_center_offset_px,
    )
    return aligned_current_retinal_movie(eye_stim.detach().cpu().numpy(), t_max=t_max)


def model_lag_tensor_from_image_trace(
    image: np.ndarray,
    trace: np.ndarray,
    *,
    t_max: int,
    crop_center_offset_px: tuple[float, float] = (0.0, 0.0),
):
    """Return the raw lagged model-input tensor from the shared reconstruction path."""
    import torch

    img = np.asarray(image, dtype=np.float32)
    tr = np.asarray(trace[:t_max], dtype=np.float32)
    full_stack = np.broadcast_to(
        img[None, :, :],
        (t_max + N_LAGS + 1, *img.shape),
    ).copy()
    offset_x, offset_y = (float(crop_center_offset_px[0]), float(crop_center_offset_px[1]))
    if offset_x == 0.0 and offset_y == 0.0:
        return make_counterfactual_stim(
            full_stack,
            torch.from_numpy(tr).float(),
            ppd=PPD,
            n_lags=N_LAGS,
            out_size=OUT_SIZE,
        )

    from _common import _embed_time_lags, _eye_deg_to_norm, _shift_movie_with_eye

    eye_pos = torch.from_numpy(tr).float()
    eye_norm = _eye_deg_to_norm(torch.fliplr(eye_pos), PPD, full_stack.shape[1:3])
    h, w = img.shape
    center = (
        2.0 * offset_x / max(w - 1, 1),
        2.0 * offset_y / max(h - 1, 1),
    )
    eye_movie = _shift_movie_with_eye(
        torch.from_numpy(full_stack[:eye_pos.shape[0] + N_LAGS]).float(),
        torch.cat([eye_norm[:N_LAGS], eye_norm], dim=0),
        out_size=OUT_SIZE,
        center=center,
        scale_factor=1.0,
        mode="bilinear",
    )
    return _embed_time_lags(eye_movie, n_lags=N_LAGS)


def aligned_current_retinal_movie(retinal: np.ndarray, *, t_max: int) -> np.ndarray:
    """Extract lag-0 frames so movie frame ``t`` matches eye-trace sample ``t``."""
    arr = np.asarray(retinal, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        movie = arr
    elif arr.ndim == 4:
        # time x lags x height x width; lag 0 is the current model frame.
        movie = arr[:, 0]
    elif arr.ndim == 5:
        # time x channel x lags x height x width.
        movie = arr[:, 0, 0]
    else:
        raise ValueError(f"Cannot extract display movie from retinal tensor shape {arr.shape}")
    if movie.shape[0] >= t_max + 1:
        movie = movie[1:t_max + 1]
    else:
        movie = movie[:t_max]
    if movie.shape[0] != t_max:
        raise ValueError(f"Aligned retinal movie has {movie.shape[0]} frames, expected {t_max}")
    return movie.astype(np.float32)


def aligned_model_lag_cubes(retinal: np.ndarray, *, t_max: int, n_lags: int = N_LAGS) -> np.ndarray:
    """Return ``(time, lag, height, width)`` cubes aligned to eye-trace samples.

    Lag index 0 is the current model frame.  Cube ``t`` is the 32-frame input
    history whose current frame corresponds to eye-trace sample ``t``.
    """
    arr = np.asarray(retinal, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 4:
        cubes = arr
    elif arr.ndim == 5:
        cubes = arr[:, 0]
    else:
        raise ValueError(f"Cannot extract lag cubes from retinal tensor shape {arr.shape}")
    if cubes.shape[0] >= t_max + 1:
        cubes = cubes[1:t_max + 1]
    else:
        cubes = cubes[:t_max]
    if cubes.shape[0] != t_max:
        raise ValueError(f"Aligned lag cubes have {cubes.shape[0]} samples, expected {t_max}")
    if cubes.shape[1] != n_lags:
        raise ValueError(f"Expected {n_lags} lag frames, got {cubes.shape[1]}")
    return cubes.astype(np.float32)


def model_lag_cubes_from_image_trace(
    image: np.ndarray,
    trace: np.ndarray,
    *,
    t_max: int,
    crop_center_offset_px: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Return aligned model lag cubes using the exact shared crop/rendering path."""
    eye_stim = model_lag_tensor_from_image_trace(
        image,
        trace,
        t_max=t_max,
        crop_center_offset_px=crop_center_offset_px,
    )
    return aligned_model_lag_cubes(eye_stim.detach().cpu().numpy(), t_max=t_max)


def model_crop_centers_px(
    trace: np.ndarray,
    image_shape: tuple[int, int],
    ppd: float = PPD,
    crop_center_offset_px: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Source-image crop centers corresponding to the model's eye-shifted crop.

    This mirrors ``make_counterfactual_stim``: the Ryan common path flips the
    two eye-position columns before converting degrees to grid-sample shifts.
    Returned coordinates are ``(x_px, y_px)`` for plotting on the source image.
    """
    tr = np.asarray(trace, dtype=np.float64)
    h, w = int(image_shape[0]), int(image_shape[1])
    offset_x, offset_y = (float(crop_center_offset_px[0]), float(crop_center_offset_px[1]))
    x = (w - 1) / 2.0 + offset_x - tr[:, 1] * float(ppd)
    y = (h - 1) / 2.0 + offset_y + tr[:, 0] * float(ppd)
    return np.column_stack([x, y]).astype(np.float32)


def local_phase_scramble_roi(
    trace: np.ndarray,
    image_shape: tuple[int, int],
    *,
    margin_px: int | None = None,
    ppd: float = PPD,
    out_size: tuple[int, int] = OUT_SIZE,
    crop_center_offset_px: tuple[float, float] = (0.0, 0.0),
) -> dict[str, int | float]:
    """Return the source-image ROI containing all trace-sampled model crops."""
    centers = model_crop_centers_px(
        trace,
        image_shape,
        ppd=ppd,
        crop_center_offset_px=crop_center_offset_px,
    )
    h, w = int(image_shape[0]), int(image_shape[1])
    crop_h, crop_w = int(out_size[0]), int(out_size[1])
    margin = crop_h // 4 if margin_px is None else int(margin_px)
    if centers.size == 0:
        raise ValueError("Cannot define a local phase-scramble ROI for an empty trace.")

    x0 = int(np.floor(np.min(centers[:, 0] - crop_w / 2.0) - margin))
    x1 = int(np.ceil(np.max(centers[:, 0] + crop_w / 2.0) + margin))
    y0 = int(np.floor(np.min(centers[:, 1] - crop_h / 2.0) - margin))
    y1 = int(np.ceil(np.max(centers[:, 1] + crop_h / 2.0) + margin))
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w, x1))
    y1 = max(y0 + 1, min(h, y1))
    return {
        "roi_x0": x0,
        "roi_x1": x1,
        "roi_y0": y0,
        "roi_y1": y1,
        "roi_width_px": int(x1 - x0),
        "roi_height_px": int(y1 - y0),
        "roi_area_fraction": float(((x1 - x0) * (y1 - y0)) / max(h * w, 1)),
        "roi_margin_px": int(margin),
        "crop_width_px": int(crop_w),
        "crop_height_px": int(crop_h),
        "crop_center_offset_x_px": float(crop_center_offset_px[0]),
        "crop_center_offset_y_px": float(crop_center_offset_px[1]),
        "trace_center_x_min": float(np.min(centers[:, 0])),
        "trace_center_x_max": float(np.max(centers[:, 0])),
        "trace_center_y_min": float(np.min(centers[:, 1])),
        "trace_center_y_max": float(np.max(centers[:, 1])),
    }


_PYRAMID_CACHE = {}


def _padded_even_patch(patch: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Reflect-pad a patch to even dimensions for stable pyramid reconstruction."""
    arr = np.asarray(patch, dtype=np.float32)
    pad_h = int(arr.shape[0] % 2)
    pad_w = int(arr.shape[1] % 2)
    if pad_h == 0 and pad_w == 0:
        return arr, (0, 0)
    padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="reflect")
    return padded.astype(np.float32), (pad_h, pad_w)


def _crop_even_padding(arr: np.ndarray, padding: tuple[int, int]) -> np.ndarray:
    pad_h, pad_w = padding
    h_stop = arr.shape[0] - int(pad_h) if pad_h else arr.shape[0]
    w_stop = arr.shape[1] - int(pad_w) if pad_w else arr.shape[1]
    return arr[:h_stop, :w_stop].astype(np.float32)


def _steerable_pyramid(shape: tuple[int, int], *, height: int = 3, order: int = 3):
    key = (int(shape[0]), int(shape[1]), int(height), int(order))
    if key not in _PYRAMID_CACHE:
        from plenoptic.simulate import SteerablePyramidFreq

        _PYRAMID_CACHE[key] = SteerablePyramidFreq(
            (key[0], key[1]),
            height=height,
            order=order,
            is_complex=True,
            downsample=False,
            tight_frame=False,
        )
    return _PYRAMID_CACHE[key]


def _patch_to_tensor(patch: np.ndarray):
    import torch

    return torch.from_numpy(np.asarray(patch, dtype=np.float32))[None, None]


def _tensor_to_patch(tensor) -> np.ndarray:
    return tensor.detach().cpu().squeeze().numpy().astype(np.float32)


def _zero_like_pyramid_coeffs(coeffs):
    return OrderedDict((key, value * 0) for key, value in coeffs.items())


def _copy_pyramid_coeffs(coeffs):
    return OrderedDict((key, value.clone()) for key, value in coeffs.items())


def _reconstruct_pyramid_patch(pyr, coeffs, padding: tuple[int, int]) -> np.ndarray:
    recon = pyr.recon_pyr(coeffs)
    return _crop_even_padding(_tensor_to_patch(recon), padding)


def _phase_scramble_pyramid_coeffs(coeffs, rng: np.random.Generator):
    import torch

    scrambled = OrderedDict()
    mag_error_num = 0.0
    mag_error_den = 0.0
    n_complex = 0
    for key, value in coeffs.items():
        if torch.is_complex(value):
            n_complex += 1
            magnitude = torch.abs(value)
            phase_np = rng.uniform(-np.pi, np.pi, size=tuple(value.shape)).astype(np.float32)
            phase = torch.from_numpy(phase_np).to(value.device)
            out = torch.polar(magnitude, phase)
            mag_error_num += float(torch.linalg.norm(torch.abs(out) - magnitude).cpu())
            mag_error_den += float(torch.linalg.norm(magnitude).cpu())
            scrambled[key] = out
        else:
            scrambled[key] = value.clone()
    return scrambled, float(mag_error_num / max(mag_error_den, 1e-12)), int(n_complex)


def _pyramid_level_indices(coeffs) -> list[int]:
    """Return the spatial-frequency level indices present in pyramid coeffs."""
    return sorted({int(key[0]) for key in coeffs if isinstance(key, tuple)})


def _selected_band_coeffs(coeffs, band: str):
    selected = _zero_like_pyramid_coeffs(coeffs)
    levels = _pyramid_level_indices(coeffs)
    if not levels:
        raise ValueError("No oriented pyramid levels found.")
    low_level = levels[-1]
    if band == "sf_high":
        keep = {"residual_highpass"}
        keep.update(key for key in coeffs if isinstance(key, tuple) and int(key[0]) == 0)
    elif band == "sf_mid_high":
        if len(levels) < 3:
            raise ValueError("sf_mid_high requires at least three pyramid levels.")
        keep = {key for key in coeffs if isinstance(key, tuple) and int(key[0]) == levels[1]}
    elif band == "sf_mid_low":
        if len(levels) < 4:
            raise ValueError("sf_mid_low requires a four-level pyramid.")
        keep = {key for key in coeffs if isinstance(key, tuple) and int(key[0]) == levels[2]}
    elif band == "sf_mid":
        if len(levels) < 2:
            raise ValueError("sf_mid requires at least two pyramid levels.")
        keep = {key for key in coeffs if isinstance(key, tuple) and int(key[0]) == 1}
    elif band == "sf_low":
        keep = {"residual_lowpass"}
        keep.update(key for key in coeffs if isinstance(key, tuple) and int(key[0]) == low_level)
    else:
        raise ValueError(f"Unsupported pyramid band: {band}")
    for key in keep:
        if key in coeffs:
            selected[key] = coeffs[key].clone()
    return selected


def _paste_local_control(
    src: np.ndarray,
    patch_raw: np.ndarray,
    roi: dict[str, int | float],
    *,
    clip: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    y0, y1 = int(roi["roi_y0"]), int(roi["roi_y1"])
    x0, x1 = int(roi["roi_x0"]), int(roi["roi_x1"])
    src_min = float(np.nanmin(src))
    src_max = float(np.nanmax(src))
    if clip:
        patch = np.clip(patch_raw, src_min, src_max).astype(np.float32)
    else:
        patch = np.asarray(patch_raw, dtype=np.float32)
    out = src.copy()
    out[y0:y1, x0:x1] = patch
    outside = np.ones(src.shape, dtype=bool)
    outside[y0:y1, x0:x1] = False
    outside_changed = float(np.mean(out[outside] != src[outside])) if np.any(outside) else 0.0
    return out.astype(np.float32), {
        "source_min": src_min,
        "source_max": src_max,
        "raw_mean": float(np.mean(patch_raw)),
        "raw_std": float(np.std(patch_raw)),
        "clipped_mean": float(np.mean(patch)),
        "clipped_std": float(np.std(patch)),
        "clipping_fraction": float(np.mean((patch_raw < src_min) | (patch_raw > src_max))),
        "outside_roi_changed_fraction": outside_changed,
    }


def pyramid_local_image_controls(
    image: np.ndarray,
    trace: np.ndarray,
    rng: np.random.Generator,
    *,
    margin_px: int | None = None,
    clip: bool = True,
    ppd: float = PPD,
    out_size: tuple[int, int] = OUT_SIZE,
    crop_center_offset_px: tuple[float, float] = (0.0, 0.0),
    height: int = 3,
    order: int = 3,
    sf_bands: tuple[str, ...] = ("sf_low", "sf_mid", "sf_high"),
) -> tuple[dict[str, np.ndarray], list[dict[str, float | int | str]]]:
    """Build local complex-steerable-pyramid phase and SF-band controls.

    The source image outside the trace ROI is unchanged.  The three SF-band
    controls preserve natural band energy: no contrast renormalization is
    applied before clipping to the source image range.
    """
    src = np.asarray(image, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError(f"Expected a 2D source image, got shape {src.shape}")
    roi = local_phase_scramble_roi(
        trace,
        src.shape,
        margin_px=margin_px,
        ppd=ppd,
        out_size=out_size,
        crop_center_offset_px=crop_center_offset_px,
    )
    y0, y1 = int(roi["roi_y0"]), int(roi["roi_y1"])
    x0, x1 = int(roi["roi_x0"]), int(roi["roi_x1"])
    patch = src[y0:y1, x0:x1].astype(np.float32)
    padded, padding = _padded_even_patch(patch)
    pyr = _steerable_pyramid(padded.shape, height=height, order=order)
    coeffs = pyr(_patch_to_tensor(padded))
    recon_patch = _reconstruct_pyramid_patch(pyr, _copy_pyramid_coeffs(coeffs), padding)
    recon_error = float(np.linalg.norm(recon_patch - patch) / (np.linalg.norm(patch) + 1e-12))

    controls: dict[str, np.ndarray] = {}
    audits: list[dict[str, float | int | str]] = []
    common = {
        **roi,
        "pyramid_height": int(height),
        "pyramid_order": int(order),
        "padded_height_px": int(padded.shape[0]),
        "padded_width_px": int(padded.shape[1]),
        "pad_bottom_px": int(padding[0]),
        "pad_right_px": int(padding[1]),
        "original_roi_mean": float(np.mean(patch)),
        "original_roi_std": float(np.std(patch)),
        "pyramid_reconstruction_relative_error": recon_error,
    }

    scrambled_coeffs, coeff_mag_error, n_complex_phase_scrambled = _phase_scramble_pyramid_coeffs(coeffs, rng)
    phase_patch = _reconstruct_pyramid_patch(pyr, scrambled_coeffs, padding)
    out, audit = _paste_local_control(src, phase_patch, roi, clip=clip)
    controls["pyramid_phase_scrambled"] = out
    audits.append({
        **common,
        **audit,
        "control": "pyramid_phase_scrambled",
        "complex_coeff_magnitude_relative_error": coeff_mag_error,
        "n_complex_phase_scrambled_bands": n_complex_phase_scrambled,
        "real_residual_policy": "preserved",
        "band_component_mean": float("nan"),
        "band_component_std": float("nan"),
        "band_energy_fraction": float("nan"),
    })

    for band in sf_bands:
        band_coeffs = _selected_band_coeffs(coeffs, band)
        component = _reconstruct_pyramid_patch(pyr, band_coeffs, padding)
        if band != "sf_low":
            band_patch = component + float(np.mean(patch))
        else:
            band_patch = component
        out, audit = _paste_local_control(src, band_patch, roi, clip=clip)
        controls[band] = out
        audits.append({
            **common,
            **audit,
            "control": band,
            "complex_coeff_magnitude_relative_error": 0.0,
            "n_complex_phase_scrambled_bands": 0,
            "real_residual_policy": "selected_band",
            "band_component_mean": float(np.mean(component)),
            "band_component_std": float(np.std(component)),
            "band_energy_fraction": float(
                (np.std(component) ** 2) / max(float(np.std(patch) ** 2), 1e-12)
            ),
        })
    return controls, audits


def trace_example_row(example: TraceExample) -> dict[str, Any]:
    return {
        "example_id": example.example_id,
        "kind": example.kind,
        "source_trace_index": example.source_trace_index,
        "window_start": example.window_start,
        "window_stop": example.window_stop,
        "threshold_deg_s": example.threshold_deg_s,
        "n_events_in_window": example.n_events_in_window,
        "event_onset": example.event_onset,
        "event_offset": example.event_offset,
        "peak_speed_deg_s": example.peak_speed_deg_s,
        "rms_displacement_deg": example.rms_displacement_deg,
        "path_length_deg": example.path_length_deg,
        "max_speed_deg_s": example.max_speed_deg_s,
    }
