"""Stimulus loading and small image-control helpers."""
from __future__ import annotations

import numpy as np

from .common import PPD, StimulusSpec


def phase_scramble_image(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    match_mean: bool = True,
    match_std: bool = True,
) -> np.ndarray:
    """Fourier phase-scramble a real image while preserving amplitude."""
    x = np.asarray(image, dtype=np.float64)
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    amp = np.abs(np.fft.fft2(x - mu))
    random_phase = np.angle(np.fft.fft2(rng.normal(size=x.shape)))
    random_phase[0, 0] = 0.0
    y = np.fft.ifft2(amp * np.exp(1j * random_phase)).real
    if match_std:
        y = y / (np.std(y) + 1e-12) * sigma
    if match_mean:
        y = y + mu
    return y.astype(np.float32)


def amplitude_spectrum(image: np.ndarray) -> np.ndarray:
    """Return shifted log amplitude spectrum for QC display."""
    x = np.asarray(image, dtype=np.float64)
    amp = np.abs(np.fft.fftshift(np.fft.fft2(x - np.mean(x), norm="ortho")))
    return np.log1p(amp).astype(np.float32)


def amplitude_spectrum_relative_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Compare full Fourier amplitudes for two same-shaped images."""
    fa = np.abs(np.fft.fft2(np.asarray(a, dtype=np.float64) - np.mean(a)))
    fb = np.abs(np.fft.fft2(np.asarray(b, dtype=np.float64) - np.mean(b)))
    return float(np.linalg.norm(fa - fb) / (np.linalg.norm(fa) + eps))


def shift_image(
    image: np.ndarray,
    dx_deg: float,
    dy_deg: float,
    *,
    ppd: float = PPD,
    mode: str = "nearest",
    order: int = 1,
) -> np.ndarray:
    """Shift an image by visual-angle units without circular wraparound."""
    from scipy.ndimage import shift

    shifted = shift(
        np.asarray(image, dtype=np.float32),
        shift=(float(dy_deg) * ppd, float(dx_deg) * ppd),
        order=order,
        mode=mode,
        prefilter=False,
    )
    return shifted.astype(np.float32)


def load_natural_images(n_images: int, indices: tuple[int, ...] | None = None) -> list[tuple[StimulusSpec, np.ndarray]]:
    """Load natural images using the project FIXRsvp image stack helper."""
    from scripts.mcfarland_sim import get_fixrsvp_stack

    stack = get_fixrsvp_stack(frames_per_im=1, prefix="nat").astype(np.float32)
    chosen = tuple(range(min(int(n_images), stack.shape[0]))) if indices is None else tuple(indices)
    out: list[tuple[StimulusSpec, np.ndarray]] = []
    for idx in chosen:
        if idx < 0 or idx >= stack.shape[0]:
            raise ValueError(f"natural image index {idx} out of range for {stack.shape[0]} images")
        out.append((StimulusSpec(family="natural", image_index=int(idx), seed=0), stack[idx].astype(np.float32)))
    return out
