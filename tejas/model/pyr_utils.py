import torch
import numpy as np
from plenoptic.simulate import SteerablePyramidFreq


def get_pyr_band_frequencies(pyr, image_shape, ppd):
    """
    Estimate per-band peak spatial frequency from impulse reconstruction.
    Returns list of dicts with cycles/pixel and cycles/degree.
    """
    H, W = image_shape
    empty = torch.zeros((1, 1, H, W), dtype=torch.float32, device=pyr.lo0mask.device)
    base = pyr(empty)
    tuple_keys = [k for k in base.keys() if isinstance(k, tuple)]
    if len(tuple_keys) == 0:
        return []
    by = base[tuple_keys[0]].shape[-2] // 2
    bx = base[tuple_keys[0]].shape[-1] // 2

    out = []
    for level, ori in tuple_keys:
        coeffs = {k: torch.zeros_like(v) for k, v in base.items()}
        coeffs[(level, ori)][:, :, by, bx] = 1.0
        filt = pyr.recon_pyr(coeffs, [level], [ori]).squeeze().detach().cpu().numpy()
        F = np.abs(np.fft.rfft2(filt))
        fy = np.fft.fftfreq(H, d=1.0)
        fx = np.fft.rfftfreq(W, d=1.0)
        ky, kx = np.unravel_index(np.argmax(F), F.shape)
        cpp = float(np.hypot(fy[ky], fx[kx]))  # cycles per pixel
        out.append({
            "level": int(level),
            "orientation": int(ori),
            "cycles_per_pixel": cpp,
            "cycles_per_degree": float(cpp * ppd),
            "wavelength_pixels": float(np.inf if cpp == 0 else 1.0 / cpp),
            "cycles_across_width": float(cpp * W),
            "cycles_across_height": float(cpp * H),
        })
    return out


def get_sf_info(source, ppd=None, return_full=False):
    """
    Summarize steerable-pyramid spatial frequency metadata.

    Parameters
    ----------
    source : list[dict] or model-like
        Either precomputed output from `get_pyr_band_frequencies(...)` or an
        object exposing `.pyr` and `.image_shape`.
    ppd : float or None
        Required when `source` is model-like and frequency info must be computed.
    return_full : bool
        If True, return a dict with both averaged and per-band metadata.
        If False, return only {level: mean_cpd}.

    Notes
    -----
    Orientation angles reported here are index-based placeholders and are not
    calibrated from reconstructed basis filters. Use
    `calibrate_pyr_orientation_labels(...)` for physically measured bar
    orientation and `to_display_orientation_convention(...)` for paper-style
    mirrored display labels.
    """
    if isinstance(source, list):
        freq_info = source
    else:
        if ppd is None:
            raise ValueError("ppd must be provided when source is a model-like object.")
        freq_info = get_pyr_band_frequencies(source.pyr, source.image_shape, ppd)

    levels_to_cpd = {}
    level_orientation_to_cpd = {}
    for info in freq_info:
        level = int(info["level"])
        orientation = int(info["orientation"])
        cpd = float(info["cycles_per_degree"])
        if level not in levels_to_cpd:
            levels_to_cpd[level] = []
            level_orientation_to_cpd[level] = {}
        levels_to_cpd[level].append(cpd)
        level_orientation_to_cpd[level][orientation] = cpd

    levels_to_cpd = {level: np.mean(cpds).item() for level, cpds in levels_to_cpd.items()}
    if not return_full:
        return levels_to_cpd

    orientations = sorted({int(info["orientation"]) for info in freq_info})
    num_orient = max(orientations) + 1 if orientations else 0
    orientation_index_degrees = [float(ori * 180.0 / max(num_orient, 1)) for ori in orientations]
    orientation_display_degrees = to_display_orientation_convention(orientation_index_degrees)
    return {
        "levels_to_cpd": levels_to_cpd,
        "level_orientation_to_cpd": level_orientation_to_cpd,
        "orientations": orientations,
        # Backward-compatible key: historically this referred to index-based
        # angles. Keep it for existing callers.
        "orientation_degrees": orientation_index_degrees,
        "orientation_index_degrees": orientation_index_degrees,
        "orientation_display_degrees": orientation_display_degrees,
        "freq_info": freq_info,
    }


def estimate_bar_orientation_deg_from_filter(filt_2d, rad_min=0.03):
    """
    Estimate bar orientation (deg in [0, 180)) from a reconstructed basis filter.
    """
    h, w = filt_2d.shape
    spec = np.abs(np.fft.fftshift(np.fft.fft2(filt_2d))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(h, d=1.0))
    fx = np.fft.fftshift(np.fft.fftfreq(w, d=1.0))
    xx, yy = np.meshgrid(fx, fy, indexing="xy")
    rr = np.sqrt(xx * xx + yy * yy)
    mask = rr >= rad_min
    ww = spec * mask
    denom = ww.sum()
    if denom <= 0:
        return float("nan")
    mx = (ww * xx).sum() / denom
    my = (ww * yy).sum() / denom
    x0 = xx - mx
    y0 = yy - my
    cxx = (ww * x0 * x0).sum() / denom
    cyy = (ww * y0 * y0).sum() / denom
    cxy = (ww * x0 * y0).sum() / denom
    cov = np.array([[cxx, cxy], [cxy, cyy]], dtype=np.float64)
    _, vecs = np.linalg.eigh(cov)
    vx, vy = vecs[:, 1]
    theta_freq = np.degrees(np.arctan2(vy, vx)) % 180.0
    return float((theta_freq + 90.0) % 180.0)


def circular_mean_deg_180(angles_deg):
    """
    Circular mean for orientations defined modulo 180 degrees.
    """
    arr = np.asarray([a for a in angles_deg if np.isfinite(a)], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    doubled = np.deg2rad(2.0 * arr)
    mean_angle = 0.5 * np.arctan2(np.sin(doubled).mean(), np.cos(doubled).mean())
    return float(np.degrees(mean_angle) % 180.0)


def to_display_orientation_convention(angles_deg):
    """
    Map orientation angles to the mirrored display convention used in figures.

    For modulo-180 orientation, this is:
      display = (180 - angle) mod 180
    which yields paper-style ordering (e.g., 90, 60, 30, 0, 150, 120).
    """
    arr = np.asarray(angles_deg, dtype=np.float64)
    if arr.size == 0:
        return []
    return [float(v) for v in np.mod(180.0 - arr, 180.0)]


def calibrate_pyr_orientation_labels(
    pyr,
    image_shape,
    num_orientations,
    scales=None,
    device=None,
    rad_min=0.03,
):
    """
    Calibrate orientation index -> bar orientation (deg) from basis reconstructions.
    Returns (orientation_degrees, orientation_check), where:
      - orientation_degrees are calibrated bar orientations
      - orientation_check also includes mirrored display-convention angles
    """
    if device is None:
        device = pyr.lo0mask.device
    dummy = torch.zeros(1, 1, *image_shape, dtype=torch.float32, device=device)
    pyr_template = pyr(dummy)
    tuple_keys = [k for k in pyr_template.keys() if isinstance(k, tuple)]
    if len(tuple_keys) == 0:
        fallback = [float(o * 180.0 / max(num_orientations, 1)) for o in range(num_orientations)]
        return fallback, {"status": "no_tuple_keys"}

    if scales is None:
        scales = sorted({int(k[0]) for k in tuple_keys})
    else:
        scales = [int(s) for s in scales]

    per_scale = {}
    per_orientation = {ori: [] for ori in range(num_orientations)}

    for scale_idx in scales:
        scale_map = {}
        for orient_idx in range(num_orientations):
            coeffs = {k: torch.zeros_like(v) for k, v in pyr_template.items()}
            k = (int(scale_idx), int(orient_idx))
            if k not in coeffs:
                continue
            hh, ww = coeffs[k].shape[-2:]
            cy, cx = hh // 2, ww // 2
            coeffs[k][:, :, cy, cx] = 1.0
            filt = pyr.recon_pyr(coeffs, [int(scale_idx)], [int(orient_idx)]).squeeze(0).squeeze(0).detach().cpu().numpy()
            bar_deg = estimate_bar_orientation_deg_from_filter(filt, rad_min=rad_min)
            scale_map[int(orient_idx)] = float(bar_deg)
            per_orientation[int(orient_idx)].append(float(bar_deg))
        per_scale[int(scale_idx)] = scale_map

    calibrated = []
    for ori in range(num_orientations):
        mean_deg = circular_mean_deg_180(per_orientation.get(ori, []))
        if not np.isfinite(mean_deg):
            mean_deg = float(ori * 180.0 / max(num_orientations, 1))
        calibrated.append(float(mean_deg))

    expected_step = 180.0 / max(num_orientations, 1)
    wrapped = np.mod(np.asarray(calibrated, dtype=np.float64), 180.0)
    diffs = np.diff(np.concatenate([wrapped, wrapped[:1] + 180.0]))
    check = {
        "status": "ok",
        "calibration_scales": scales,
        "per_scale_orientation_degrees": per_scale,
        "orientation_degrees": calibrated,
        "orientation_display_degrees": to_display_orientation_convention(calibrated),
        "expected_step_deg": float(expected_step),
        "measured_steps_deg": [float(d) for d in diffs],
        "max_step_error_deg": float(np.max(np.abs(diffs - expected_step))),
    }
    return calibrated, check

def find_pyr_size_and_height_for_lowest_cpd(
    lowest_cpd_target,
    ppd,
    order=5,
    validate=True,
    rel_tolerance=0.0,
):
    """
    Analytic design for this steerable-pyramid implementation:
      cpd(level i) ~= ppd * 2^(-(i+2))
    so lowest level for height h has:
      lowest_cpd ~= ppd / 2^(h+1)
    and valid height needs:
      min(H, W) >= 2^(h+2)
    """
    if lowest_cpd_target <= 0:
        raise ValueError("lowest_cpd_target must be positive.")
    if ppd <= 0:
        raise ValueError("ppd must be positive.")
    if rel_tolerance < 0:
        raise ValueError("rel_tolerance must be non-negative.")

    # Strict height that guarantees approx_lowest_cpd <= target.
    strict_height = int(max(1, np.ceil(np.log2(ppd / lowest_cpd_target)) - 1))
    # Candidate one level smaller (higher cpd, smaller image). Use if within tolerance.
    candidate_height = max(1, strict_height - 1)
    strict_approx = float(ppd / (2 ** (strict_height + 1)))
    candidate_approx = float(ppd / (2 ** (candidate_height + 1)))

    allowed_cpd = lowest_cpd_target * (1.0 + rel_tolerance)
    # Analytic default choice
    height = candidate_height if candidate_approx <= allowed_cpd else strict_height

    def _validate_height(h):
        image_shape = (2 ** (h + 2), 2 ** (h + 2))
        pyr = SteerablePyramidFreq(
            image_shape=image_shape,
            height=h,
            order=order,
            is_complex=False,
            downsample=False,
        )
        freq_info = get_pyr_band_frequencies(pyr, image_shape, ppd)
        sf_meta = get_sf_info(freq_info, return_full=True)
        levels_to_cpd = sf_meta["levels_to_cpd"]
        min_level = max(levels_to_cpd.keys())
        return float(levels_to_cpd[min_level]), int(min_level), levels_to_cpd, freq_info

    # If validating, prefer the smaller height when measured cpd is within tolerance.
    if validate and candidate_height < strict_height:
        cand_cpd, cand_level, cand_levels, cand_freq_info = _validate_height(candidate_height)
        if cand_cpd <= allowed_cpd:
            height = candidate_height
            validated_lowest_cpd = cand_cpd
            validated_lowest_level = cand_level
            validated_levels_to_cpd = cand_levels
            validated_freq_info = cand_freq_info
        else:
            strict_cpd, strict_level, strict_levels, strict_freq_info = _validate_height(strict_height)
            height = strict_height
            validated_lowest_cpd = strict_cpd
            validated_lowest_level = strict_level
            validated_levels_to_cpd = strict_levels
            validated_freq_info = strict_freq_info

    size = int(2 ** (height + 2))
    image_shape = (size, size)
    approx_lowest_cpd = float(ppd / (2 ** (height + 1)))

    out = {
        "image_shape": image_shape,
        "height": height,
        "lowest_level": int(height - 1),
        "approx_lowest_cpd": approx_lowest_cpd,
        "target_cpd": float(lowest_cpd_target),
        "allowed_cpd_with_tolerance": float(allowed_cpd),
        "rel_tolerance": float(rel_tolerance),
        "strict_height": int(strict_height),
        "candidate_height": int(candidate_height),
        "strict_approx_lowest_cpd": float(strict_approx),
        "candidate_approx_lowest_cpd": float(candidate_approx),
        "formula": "lowest_cpd ~= ppd / 2^(height+1), min_size = 2^(height+2)",
    }

    if validate:
        if "validated_lowest_cpd" not in locals():
            v_cpd, v_level, v_levels, v_freq_info = _validate_height(height)
            validated_lowest_cpd = v_cpd
            validated_lowest_level = v_level
            validated_levels_to_cpd = v_levels
            validated_freq_info = v_freq_info
        out["validated_lowest_level"] = int(validated_lowest_level)
        out["validated_lowest_cpd"] = float(validated_lowest_cpd)
        out["levels_to_cpd"] = validated_levels_to_cpd
        out["freq_info"] = validated_freq_info
    return out

# #%%
# order = 3
# imsize = 64

# import plenoptic as po
# from plenoptic.simulate import SteerablePyramidFreq
# import itertools
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# dtype = torch.float32
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pyr = SteerablePyramidFreq(height=3, image_shape=[imsize, imsize], order=order).to(
#     DEVICE
# )
# empty_image = torch.zeros((1, 1, imsize, imsize), dtype=dtype).to(DEVICE)
# pyr_coeffs = pyr.forward(empty_image)

# # insert a 1 in the center of each coefficient...
# for k, v in pyr.pyr_size.items():
#     mid = (v[0] // 2, v[1] // 2)
#     pyr_coeffs[k][..., mid[0], mid[1]] = 1

# # ... and then reconstruct this dummy image to visualize the filter.
# reconList = []
# for scale, ori in itertools.product(range(pyr.num_scales), range(pyr.num_orientations)):
#     reconList.append(pyr.recon_pyr(pyr_coeffs, [scale], [ori]))

# po.imshow(reconList, col_wrap=order + 1, vrange="indep1", zoom=2);