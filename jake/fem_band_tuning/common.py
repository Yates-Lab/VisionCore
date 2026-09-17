"""Laplacian scene interventions for the Figure 4 nonlinear twin.

No image-specific normalization is allowed after the bands are constructed.
Band interventions are replayed throughout one causal history; only its final
output bin is scored. The DC component is always held fixed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 20260914
LEVELS = 5  # five Laplacian detail levels plus the coarse residual
STRENGTHS = np.array([0.5, 0.75, 1.0, 1.25, 1.5], dtype=np.float32)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    def convert(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'Cannot serialize {type(value)}')
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False, default=convert) + '\n')


def selected_sources():
    selected_path = ROOT / 'manuscript/analysis/selected_model_bundle.json'
    selected = json.loads(selected_path.read_text())
    bundle = ROOT / selected['bundle'] / 'figure4'
    source_path = bundle / 'top_passband_stage_trajectory_10img_x_10fix/summary.json'
    source = json.loads(source_path.read_text())
    if source['checkpoint_sha256'] != selected['checkpoint_sha256']:
        raise ValueError('Figure 4 summary and current selected checkpoint disagree')
    return selected, source, bundle, selected_path, source_path


def laplacian_bands(image, levels=LEVELS):
    """Exact reconstructing OpenCV binomial pyramid, all bands at input size.

    Bands include the expanded coarse residual with its mean removed. Means
    of the expanded detail bands are also assigned to the fixed DC image.
    Filtering precedes cropping, so the neural crop boundary is not a pyramid
    boundary. Dyadic bands overlap and are not sharp Fourier cutoffs.
    """
    gaussian = [np.asarray(image, dtype=np.float32)]
    for _ in range(levels):
        gaussian.append(cv2.pyrDown(gaussian[-1]))
    native = [gaussian[k] - cv2.pyrUp(gaussian[k+1], dstsize=gaussian[k].shape[::-1])
              for k in range(levels)] + [gaussian[-1]]
    expanded = []
    for k, band in enumerate(native):
        for j in range(k-1, -1, -1):
            band = cv2.pyrUp(band, dstsize=gaussian[j].shape[::-1])
        expanded.append(band)
    bands = np.stack(expanded)
    means = bands.mean(axis=(1, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    return bands - means, float(means.sum())


def interventions(n_bands=LEVELS+1):
    """Shared baseline, single-band sweeps, and untouched joint validation probes."""
    offsets = [np.zeros(n_bands, dtype=np.float32)]
    rows = [{'kind': 'baseline', 'band': -1, 'strength': 1.0}]
    for k in range(n_bands):
        for a in STRENGTHS:
            if a == 1:
                continue
            d = np.zeros(n_bands, dtype=np.float32)
            d[k] = a - 1
            offsets.append(d)
            rows.append({'kind': 'single_band', 'band': k, 'strength': float(a)})
    rng = np.random.default_rng(SEED)
    for _ in range(4):
        d = rng.choice([-0.2, 0.2], n_bands).astype(np.float32)
        for sign in [1, -1]:
            offsets.append(d * sign)
            rows.append({'kind': 'joint_validation', 'band': -1, 'strength': None})
    return np.stack(offsets), rows


def sweep_indices(rows, n_bands=LEVELS+1):
    indices = np.zeros((n_bands, len(STRENGTHS)), dtype=int)
    for k in range(n_bands):
        for j, a in enumerate(STRENGTHS):
            if a != 1:
                indices[k, j] = next(i for i, row in enumerate(rows)
                    if row['kind'] == 'single_band' and row['band'] == k and row['strength'] == float(a))
    return indices


def render_basis(fields, trace, *, device):
    """Return [component,1,newest-first lag,y,x], using Figure 4 geometry.

    fields[0] is the reference image in uint-like units; the rest are signed
    band images. All normalization constants are fixed across interventions.
    """
    import torch
    from paper.fig4.upstream.real_trace_matrix.model import (
        _eye_deg_to_norm, _shift_movie_with_eye, PPD, OUT_SIZE)
    value = torch.as_tensor(fields, dtype=torch.float32, device=device)
    eye = torch.as_tensor(trace, dtype=torch.float32, device=device)
    norm = _eye_deg_to_norm(eye, ppd=PPD, img_size=fields.shape[-2:], torch=torch)
    movie = _shift_movie_with_eye(value[None].expand(len(trace), -1, -1, -1), norm,
        out_size=OUT_SIZE, scale_factor=1.0, torch=torch)
    history = movie.flip(0).permute(1, 0, 2, 3).unsqueeze(1).contiguous() / 255.0
    history[0] -= 127.0 / 255.0
    return history


def center_rates(scorer, history, *, audit=False):
    """One native exact-CID readout per neuron at the central map position.

    Cropping the final feature map before the trained spatial readout avoids
    evaluating unused translated copies. This is audited against the full
    Figure 4 rate-map function, never used as an approximation to the core.
    """
    module = scorer.model.model
    if scorer.readout.has_phase_branch:
        raise ValueError('This analysis is pinned to the selected no-phase twin')
    behavior = scorer._zero_behavior(len(history), history.dtype)
    feature = module.core_forward_spatial_map(history, behavior)
    h, w = scorer.readout.space_weights.shape[-2:]
    cy = (feature.shape[-2] - h + 1) // 2
    cx = (feature.shape[-1] - w + 1) // 2
    logits = scorer.readout(feature[..., cy:cy+h, cx:cx+w])
    value = (module.activation(logits) + scorer.readout.post_activation_baseline[None, :, None, None])
    value = value * scorer.readout.available_mask[None, :, None, None]
    value = scorer.apply_population_view(value, scorer.population_view)[..., 0, 0]
    error = None
    if audit:
        full = scorer.apply_population_view(scorer._compute_rate_map(history), scorer.population_view)
        ref = full[..., full.shape[-2]//2, full.shape[-1]//2]
        error = float((ref-value).abs().max().item())
        if not scorer.torch.allclose(ref, value, rtol=2e-5, atol=3e-6):
            raise ValueError(f'Central readout equivalence failed: {error}')
    return value * scorer.output_rate_hz, error


def gain_statistics(sweeps):
    """sweeps[..., band,strength,unit] in Hz; derivatives per band coefficient."""
    low, midlow, base, midhigh, high = np.moveaxis(sweeps, -2, 0)
    gain = (midhigh-midlow) / 0.5
    gain_wide = high-low
    curvature = (midhigh + midlow - 2*base) / (0.25**2)
    return gain, gain_wide, curvature


def three_way_residual(array):
    """Balanced ANOVA interaction for axes [scene,eye,neuron,...]."""
    out = np.asarray(array, dtype=np.float64)
    for axis in (0, 1, 2):
        out = out - out.mean(axis=axis, keepdims=True)
    return out
