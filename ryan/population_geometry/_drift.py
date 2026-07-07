"""Brownian (diffusive) fixational drift generator + drift statistics.

2-D Brownian motion with diffusion constant kappa (deg^2 / s): each axis takes
independent Gaussian increments with variance 2*kappa*dt, so the per-axis mean
squared displacement grows as MSD_1d(tau) = 2*kappa*tau and the 2-D (radial) MSD
as MSD_2d(tau) = 4*kappa*tau. kappa is matched to real marmoset drift by fitting
that slope to measured fixation traces (see estimate_kappa).
"""
from __future__ import annotations

import numpy as np

DT = 1.0 / 120.0  # s per frame (120 Hz), matches the twin


def brownian_drift(n_frames: int, kappa: float, rng: np.random.Generator,
                   dt: float = DT, x0=(0.0, 0.0)) -> np.ndarray:
    """Return an (n_frames, 2) Brownian drift trace in degrees, starting at x0."""
    step_sd = np.sqrt(2.0 * kappa * dt)
    steps = rng.normal(0.0, step_sd, size=(n_frames, 2))
    steps[0] = 0.0
    return np.asarray(x0, dtype=np.float32) + np.cumsum(steps, axis=0).astype(np.float32)


def msd_curve(traces: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean squared displacement vs lag (in frames) over a set of traces.

    traces: (K, T, 2) with NaN padding allowed. Returns (lags, msd_2d) where
    lags are in frames (1..max_lag) and msd_2d is the radial MSD in deg^2.
    """
    lags = np.arange(1, max_lag + 1)
    msd = np.full(max_lag, np.nan)
    for li, lag in enumerate(lags):
        sq = []
        for tr in traces:
            valid = np.isfinite(tr[:, 0])
            n = int(valid.sum())
            if n <= lag:
                continue
            seg = tr[:n]
            d = seg[lag:] - seg[:-lag]
            sq.append(np.sum(d ** 2, axis=1))
        if sq:
            msd[li] = np.concatenate(sq).mean()
    return lags, msd


def estimate_kappa(traces: np.ndarray, dt: float = DT,
                   fit_lags: int = 10) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate kappa (deg^2/s) from traces by fitting MSD_2d = 4*kappa*tau + c.

    The intercept c absorbs the eye-tracker measurement-noise floor (white noise
    adds a constant to the MSD at all lags); fitting it out avoids inflating the
    diffusive slope. Fits over the first `fit_lags` lags. Returns (kappa, lags, msd).
    """
    lags, msd = msd_curve(traces, max_lag=max(fit_lags, 20))
    tau = lags[:fit_lags] * dt
    y = msd[:fit_lags]
    ok = np.isfinite(y)
    A = np.stack([tau[ok], np.ones(ok.sum())], axis=1)     # [tau, 1]
    slope, _c = np.linalg.lstsq(A, y[ok], rcond=None)[0]
    return float(slope / 4.0), lags, msd


def _self_test():
    """Validate the generator: recovered kappa should match the injected value."""
    rng = np.random.default_rng(0)
    kappa_true = 0.01  # deg^2/s
    T = 120
    traces = np.stack([brownian_drift(T, kappa_true, rng) for _ in range(2000)])
    kappa_hat, lags, msd = estimate_kappa(traces, fit_lags=10)
    # theoretical MSD_2d = 4 kappa tau
    theo = 4 * kappa_true * (lags * DT)
    rel_err = np.nanmean(np.abs(msd[:10] - theo[:10]) / theo[:10])
    print(f"injected kappa={kappa_true}, recovered={kappa_hat:.5f} "
          f"({100*abs(kappa_hat-kappa_true)/kappa_true:.1f}% err)")
    print(f"MSD vs theory mean rel err (first 10 lags): {rel_err:.3f}")
    assert abs(kappa_hat - kappa_true) / kappa_true < 0.1, "kappa recovery off"
    assert rel_err < 0.1, "MSD does not follow 4*kappa*tau"
    print("drift generator self-test passed.")


if __name__ == "__main__":
    _self_test()
