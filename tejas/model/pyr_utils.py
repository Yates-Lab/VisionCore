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
def get_sf_info(model, ppd):
    freq_info = get_pyr_band_frequencies(model.pyr, model.image_shape, ppd)
    levels_to_cpd = {}
    for info in freq_info:
        if info['level'] not in levels_to_cpd:
            levels_to_cpd[info['level']] = []
        levels_to_cpd[info['level']].append(info['cycles_per_degree'])
    levels_to_cpd = {level: np.mean(cpds).item() for level, cpds in levels_to_cpd.items()}
    return levels_to_cpd

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
        levels_to_cpd = {}
        for info in freq_info:
            if info["level"] not in levels_to_cpd:
                levels_to_cpd[info["level"]] = []
            levels_to_cpd[info["level"]].append(info["cycles_per_degree"])
        levels_to_cpd = {level: np.mean(cpds).item() for level, cpds in levels_to_cpd.items()}
        min_level = max(levels_to_cpd.keys())
        return float(levels_to_cpd[min_level]), int(min_level), levels_to_cpd

    # If validating, prefer the smaller height when measured cpd is within tolerance.
    if validate and candidate_height < strict_height:
        cand_cpd, cand_level, cand_levels = _validate_height(candidate_height)
        if cand_cpd <= allowed_cpd:
            height = candidate_height
            validated_lowest_cpd = cand_cpd
            validated_lowest_level = cand_level
            validated_levels_to_cpd = cand_levels
        else:
            strict_cpd, strict_level, strict_levels = _validate_height(strict_height)
            height = strict_height
            validated_lowest_cpd = strict_cpd
            validated_lowest_level = strict_level
            validated_levels_to_cpd = strict_levels

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
            v_cpd, v_level, v_levels = _validate_height(height)
            validated_lowest_cpd = v_cpd
            validated_lowest_level = v_level
            validated_levels_to_cpd = v_levels
        out["validated_lowest_level"] = int(validated_lowest_level)
        out["validated_lowest_cpd"] = float(validated_lowest_cpd)
        out["levels_to_cpd"] = validated_levels_to_cpd
    return out