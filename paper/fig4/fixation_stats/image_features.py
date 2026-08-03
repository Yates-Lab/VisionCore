from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from scipy import ndimage


@lru_cache(maxsize=128)
def _cached_session(session_name: str):
    from DataYatesV1 import get_session

    subject, date = session_name.split("_", 1)
    return get_session(subject, date)


@lru_cache(maxsize=64)
def _backimage_canvas(session_name: str, trial_idx: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    from DataYatesV1.exp.backimage import BackImageTrial
    from PIL import Image as PILImage

    sess = _cached_session(session_name)
    trial = BackImageTrial(sess.exp["D"][int(trial_idx)], sess.exp["S"])
    image = trial.get_image()
    if image.ndim == 3:
        image = image.mean(axis=2)
    image = image.astype(np.float32)
    sr = sess.exp["S"]["screenRect"].astype(int)
    height = int(sr[3] - sr[1])
    width = int(sr[2] - sr[0])
    canvas = np.full((height, width), float(trial.bkgnd), dtype=np.float32)
    x0, y0, x1, y1 = [int(v) for v in trial.dest_rect]
    h, w = y1 - y0, x1 - x0
    if image.shape[:2] != (h, w):
        image = np.asarray(PILImage.fromarray(image.astype(np.float32), mode="F").resize((w, h), resample=2), dtype=np.float32)
    y0c, y1c = max(0, y0), min(height, y1)
    x0c, x1c = max(0, x0), min(width, x1)
    sy0, sy1 = y0c - y0, h - (y1 - y1c)
    sx0, sx1 = x0c - x0, w - (x1 - x1c)
    canvas[y0c:y1c, x0c:x1c] = image[sy0:sy1, sx0:sx1]
    ppd = float(sess.exp["S"]["pixPerDeg"])
    return canvas, ppd, (height, width)


@lru_cache(maxsize=8192)
def backimage_trial_geometry(session_name: str, trial_idx: int) -> dict[str, Any]:
    from DataYatesV1.exp.backimage import BackImageTrial

    sess = _cached_session(session_name)
    trial = BackImageTrial(sess.exp["D"][int(trial_idx)], sess.exp["S"])
    sr = sess.exp["S"]["screenRect"].astype(int)
    height = int(sr[3] - sr[1])
    width = int(sr[2] - sr[0])
    x0, y0, x1, y1 = [int(v) for v in trial.dest_rect]
    return {
        "screen_height_px": height,
        "screen_width_px": width,
        "screen_shape": (height, width),
        "ppd": float(sess.exp["S"]["pixPerDeg"]),
        "center_x_px": width / 2.0,
        "center_y_px": height / 2.0,
        "dest_rect": (x0, y0, x1, y1),
        "background": float(trial.bkgnd),
    }


def gaze_deg_to_screen_px(gaze_xy_deg: np.ndarray, *, ppd: float, screen_shape: tuple[int, int]) -> np.ndarray:
    gaze = np.asarray(gaze_xy_deg, dtype=np.float64)
    height, width = int(screen_shape[0]), int(screen_shape[1])
    out = np.empty_like(gaze, dtype=np.float64)
    out[..., 0] = width / 2.0 + gaze[..., 0] * float(ppd)
    out[..., 1] = height / 2.0 - gaze[..., 1] * float(ppd)
    return out


def screen_px_to_gaze_deg(screen_xy_px: np.ndarray, *, ppd: float, screen_shape: tuple[int, int]) -> np.ndarray:
    xy = np.asarray(screen_xy_px, dtype=np.float64)
    height, width = int(screen_shape[0]), int(screen_shape[1])
    out = np.empty_like(xy, dtype=np.float64)
    out[..., 0] = (xy[..., 0] - width / 2.0) / float(ppd)
    out[..., 1] = -(xy[..., 1] - height / 2.0) / float(ppd)
    return out


def image_axis_rad_to_gaze_axis_rad(axis_rad: float | np.ndarray) -> float | np.ndarray:
    """Convert image-array axis angle (+row down) to gaze angle (+y up)."""
    return -np.asarray(axis_rad)


def _radial_log_power_slope(
    power: np.ndarray,
    rr_cpd: np.ndarray,
    *,
    min_cpd: float = 0.5,
    max_cpd: float = 16.0,
    n_bins: int = 12,
) -> float:
    power = np.asarray(power, dtype=np.float64)
    rr = np.asarray(rr_cpd, dtype=np.float64)
    valid = (rr >= float(min_cpd)) & (rr <= float(max_cpd)) & np.isfinite(power) & (power > 0.0)
    if np.count_nonzero(valid) < 8:
        return float("nan")

    edges = np.geomspace(float(min_cpd), float(max_cpd), int(n_bins) + 1)
    centers: list[float] = []
    means: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = valid & (rr >= lo) & (rr < hi)
        if np.count_nonzero(mask) < 3:
            continue
        centers.append(float(np.sqrt(lo * hi)))
        means.append(float(np.mean(power[mask])))
    if len(means) < 4:
        return float("nan")
    x = np.log(np.asarray(centers, dtype=np.float64))
    y = np.log(np.asarray(means, dtype=np.float64))
    return float(np.polyfit(x, y, 1)[0])


def _failure_features(error: Exception | str) -> dict[str, Any]:
    return {
        "image_feature_ok": False,
        "image_feature_error": str(error),
    }


def local_backimage_features(
    *,
    session_name: str,
    trial_idx: int,
    gaze_xy_deg: np.ndarray,
    patch_radius_deg: float,
) -> dict[str, Any]:
    try:
        canvas, ppd, (height, width) = _backimage_canvas(session_name, int(trial_idx))
        geometry = backimage_trial_geometry(session_name, int(trial_idx))
    except Exception as exc:
        return _failure_features(exc)

    try:
        cx, cy = gaze_deg_to_screen_px(gaze_xy_deg, ppd=ppd, screen_shape=(height, width))
        rad = max(2, int(round(float(patch_radius_deg) * ppd)))
        x0, x1 = max(0, int(round(cx)) - rad), min(width, int(round(cx)) + rad + 1)
        y0, y1 = max(0, int(round(cy)) - rad), min(height, int(round(cy)) + rad + 1)
        if x1 <= x0 or y1 <= y0:
            return _failure_features("patch_outside_screen")
        patch = np.asarray(canvas[y0:y1, x0:x1], dtype=np.float64)
        if patch.size < 16:
            return _failure_features("patch_too_small")

        dest_x0, dest_y0, dest_x1, dest_y1 = geometry["dest_rect"]
        yy, xx = np.indices(patch.shape)
        screen_x = xx + x0
        screen_y = yy + y0
        inside_image = (
            (screen_x >= dest_x0)
            & (screen_x < dest_x1)
            & (screen_y >= dest_y0)
            & (screen_y < dest_y1)
        )
        background = float(geometry["background"])
        patch_fraction_inside_image = float(np.mean(inside_image))
        patch_fraction_background = float(np.mean(np.isclose(patch, background, atol=1e-6)))
        distance_to_image_border_px = float(min(cx - dest_x0, dest_x1 - cx, cy - dest_y0, dest_y1 - cy))

        gx = ndimage.sobel(patch, axis=1, mode="nearest")
        gy = ndimage.sobel(patch, axis=0, mode="nearest")
        grad_energy = gx * gx + gy * gy
        grad_mag = np.sqrt(grad_energy)
        jxx = float(np.mean(gx * gx))
        jyy = float(np.mean(gy * gy))
        jxy = float(np.mean(gx * gy))
        coherence_den = jxx + jyy
        coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2) / coherence_den if coherence_den > 0 else np.nan
        gradient_orientation_image = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
        edge_orientation_image = gradient_orientation_image + np.pi / 2.0
        gradient_orientation = image_axis_rad_to_gaze_axis_rad(gradient_orientation_image)
        edge_orientation = image_axis_rad_to_gaze_axis_rad(edge_orientation_image)

        demeaned = patch - float(np.mean(patch))
        amp = np.abs(np.fft.fftshift(np.fft.fft2(demeaned)))
        yy, xx = np.indices(amp.shape)
        fy = yy - (amp.shape[0] - 1) / 2.0
        fx = xx - (amp.shape[1] - 1) / 2.0
        rr = np.hypot(fy, fx)
        high = amp[rr >= np.quantile(rr, 0.75)]
        total_power = float(np.sum(amp ** 2))
        high_power = float(np.sum(high ** 2))
        power = amp ** 2
        freq_mask = rr > 0
        if np.any(freq_mask) and total_power > 0:
            w = power[freq_mask]
            fxv = fx[freq_mask]
            fyv = fy[freq_mask]
            cxx = float(np.sum(w * fxv * fxv) / (np.sum(w) + 1e-12))
            cyy = float(np.sum(w * fyv * fyv) / (np.sum(w) + 1e-12))
            cxy = float(np.sum(w * fxv * fyv) / (np.sum(w) + 1e-12))
            spectrum_anisotropy = np.sqrt((cxx - cyy) ** 2 + 4.0 * cxy ** 2) / (cxx + cyy) if (cxx + cyy) > 0 else np.nan
            spectrum_orientation_image = 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)
            spectrum_orientation = image_axis_rad_to_gaze_axis_rad(spectrum_orientation_image)
        else:
            spectrum_anisotropy = np.nan
            spectrum_orientation_image = np.nan
            spectrum_orientation = np.nan

        fy_cpd = np.fft.fftshift(np.fft.fftfreq(patch.shape[0], d=1.0 / float(ppd)))
        fx_cpd = np.fft.fftshift(np.fft.fftfreq(patch.shape[1], d=1.0 / float(ppd)))
        rr_cpd = np.hypot(*np.meshgrid(fx_cpd, fy_cpd))
        non_dc_power = float(np.sum(power[rr_cpd > 0]))
        power_slope = _radial_log_power_slope(power, rr_cpd)
        amplitude_slope = 0.5 * power_slope if np.isfinite(power_slope) else float("nan")

        def band_fraction(lo: float, hi: float | None) -> float:
            if non_dc_power <= 0:
                return float("nan")
            mask = rr_cpd >= lo
            if hi is not None:
                mask &= rr_cpd < hi
            mask &= rr_cpd > 0
            return float(np.sum(power[mask]) / non_dc_power)

        return {
            "image_feature_ok": True,
            "image_feature_error": "",
            "image_patch_center_x_px": float(cx),
            "image_patch_center_y_px": float(cy),
            "image_patch_radius_px": int(rad),
            "image_patch_fraction_inside_image": patch_fraction_inside_image,
            "image_patch_fraction_background": patch_fraction_background,
            "image_patch_distance_to_image_border_px": distance_to_image_border_px,
            "image_patch_mean": float(np.mean(patch)),
            "image_patch_std": float(np.std(patch)),
            "image_patch_rms_contrast": float(np.std(patch) / (abs(float(np.mean(patch))) + 1e-6)),
            "image_gradient_energy": float(np.mean(grad_energy)),
            "image_edge_density": float(np.mean(grad_mag > (np.mean(grad_mag) + np.std(grad_mag)))),
            "image_orientation_coherence": float(coherence),
            "image_gradient_axis_deg": float(np.degrees(gradient_orientation)),
            "image_edge_axis_deg": float(np.degrees(edge_orientation)),
            "image_gradient_orientation_deg": float(np.degrees(gradient_orientation)),
            "image_edge_orientation_deg": float(np.degrees(edge_orientation)),
            "image_dominant_orientation_deg": float(np.degrees(edge_orientation)),
            "image_gradient_axis_array_deg": float(np.degrees(gradient_orientation_image)),
            "image_edge_axis_array_deg": float(np.degrees(edge_orientation_image)),
            "image_spectrum_anisotropy": float(spectrum_anisotropy),
            "image_spectrum_orientation_deg": float(np.degrees(spectrum_orientation)),
            "image_spectrum_orientation_array_deg": float(np.degrees(spectrum_orientation_image)),
            "image_high_freq_power_fraction": high_power / total_power if total_power > 0 else float("nan"),
            "image_power_0_2_cpd_fraction": band_fraction(0.0, 2.0),
            "image_power_2_4_cpd_fraction": band_fraction(2.0, 4.0),
            "image_power_4_8_cpd_fraction": band_fraction(4.0, 8.0),
            "image_power_8plus_cpd_fraction": band_fraction(8.0, None),
            "image_power_slope_0p5_16_cpd": power_slope,
            "image_amplitude_slope_0p5_16_cpd": amplitude_slope,
            "image_power_slope_deviation_from_1f": power_slope + 2.0 if np.isfinite(power_slope) else float("nan"),
            "image_amplitude_slope_deviation_from_1f": amplitude_slope + 1.0 if np.isfinite(amplitude_slope) else float("nan"),
            "image_abs_power_slope_deviation_from_1f": abs(power_slope + 2.0) if np.isfinite(power_slope) else float("nan"),
            "image_abs_amplitude_slope_deviation_from_1f": abs(amplitude_slope + 1.0) if np.isfinite(amplitude_slope) else float("nan"),
        }
    except Exception as exc:
        return _failure_features(exc)
