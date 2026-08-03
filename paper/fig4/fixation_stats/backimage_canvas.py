"""Rebuild a BackImage trial's full-screen luminance canvas, and crop from it.

Split out of `image_features` and `run_backimage_twin_drift_geometry` so that
consumers needing only the canvas -- notably figure 4's panel-A/C schematic --
do not have to import those modules' analysis halves. Both modules re-export
what they used to define, so their own callers are unaffected.

`DataYatesV1` and `PIL` are imported inside the functions: this module sits on
the figure build's import path, and a session-loading dependency should be
paid for only by the code that actually reads a session.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


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


def _clip_patch(canvas: np.ndarray, center_xy_px: tuple[float, float], size_px: int) -> np.ndarray:
    half = int(size_px) // 2
    cx, cy = float(center_xy_px[0]), float(center_xy_px[1])
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    out = np.full((int(size_px), int(size_px)), float(np.nanmean(canvas)), dtype=np.float32)
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(canvas.shape[1], x0 + int(size_px))
    src_y1 = min(canvas.shape[0], y0 + int(size_px))
    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    if src_x1 > src_x0 and src_y1 > src_y0:
        out[dst_y0 : dst_y0 + src_y1 - src_y0, dst_x0 : dst_x0 + src_x1 - src_x0] = canvas[src_y0:src_y1, src_x0:src_x1]
    return out
