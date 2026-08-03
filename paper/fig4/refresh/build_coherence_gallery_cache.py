#!/usr/bin/env python3
"""One-time cache builder for D's local-edge-coherence example gallery.

Picks one real BackImage window per coherence bin (matching the same
COHERENCE_ORDER bins used by panels H/I/J), crops a 151x151 px patch around
its gaze center (same convention as D's own "151 x 151 crop"), and caches
the normalized patches + provenance to disk. This is the slow step (each
canvas load pulls a full-resolution stimulus frame via DataYatesV1, ~20s
each) -- _fig4_ssi_common.py only ever reads the cached .npz, mirroring
how paper/fig4/_fig4_contour_schematic.py's load_real_payload() reads
a pre-built cache instead of touching DataYatesV1 at figure-render time.

Run manually to (re)build the cache:
    uv run python paper/fig4/refresh/build_coherence_gallery_cache.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from VisionCore.paths import VISIONCORE_ROOT as ROOT

import _fig4_imports  # noqa: F401  (puts the fig4 directory on sys.path)
from fixation_stats.image_features import _backimage_canvas, gaze_deg_to_screen_px  # noqa: E402
from _fig4_contour_schematic import normalize_image  # noqa: E402

WINDOWS_CSV = (
    ROOT
    / "outputs"
    / "fixation_statistics_by_stimulus_all_sessions_after_review"
    / "backimage_image_structure_reviewed_v2_screenfiltered_yfix"
    / "backimage_image_fem_windows.csv"
)
# Same four bins as behavior_model_bridge._fig4_bridge.COHERENCE_ORDER.
COHERENCE_BANDS = (
    (0.0, 0.2, "0-0.2"),
    (0.2, 0.5, "0.2-0.5"),
    (0.5, 0.8, "0.5-0.8"),
    (0.8, 1.01, "0.8-1"),
)
CROP_SIZE_PX = 151
OUT_DIR = ROOT / "outputs" / "fig" / "ssi_figure_v2" / "panels" / "cache"
CACHE_NPZ = OUT_DIR / "coherence_gallery.npz"
PROVENANCE_JSON = OUT_DIR / "coherence_gallery_provenance.json"


def _clip_patch(canvas: np.ndarray, center_xy_px: tuple[float, float], size_px: int) -> np.ndarray:
    half = int(size_px) // 2
    cx, cy = float(center_xy_px[0]), float(center_xy_px[1])
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    out = np.full((int(size_px), int(size_px)), float(np.nanmean(canvas)), dtype=np.float32)
    src_x0, src_y0 = max(0, x0), max(0, y0)
    src_x1 = min(canvas.shape[1], x0 + int(size_px))
    src_y1 = min(canvas.shape[0], y0 + int(size_px))
    dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
    if src_x1 > src_x0 and src_y1 > src_y0:
        out[dst_y0 : dst_y0 + src_y1 - src_y0, dst_x0 : dst_x0 + src_x1 - src_x0] = canvas[src_y0:src_y1, src_x0:src_x1]
    return out


def select_representative_rows(windows_csv: Path = WINDOWS_CSV) -> pd.DataFrame:
    """One row per coherence bin, closest to that bin's midpoint (deterministic)."""
    cols = [
        "session",
        "stimulus",
        "trial_idx",
        "mean_x_deg",
        "mean_y_deg",
        "image_feature_ok",
        "image_orientation_coherence",
        "image_patch_radius_px",
    ]
    windows = pd.read_csv(windows_csv, usecols=cols)
    windows = windows[windows["stimulus"].astype(str).eq("backimage")].copy()
    ok = windows["image_feature_ok"].astype(str).str.lower().isin(["true", "1", "yes"])
    windows = windows[ok].copy()
    windows["image_orientation_coherence"] = pd.to_numeric(windows["image_orientation_coherence"], errors="coerce")
    windows = windows[np.isfinite(windows["image_orientation_coherence"])].copy()

    rows = []
    for lo, hi, label in COHERENCE_BANDS:
        sub = windows[
            (windows["image_orientation_coherence"] >= lo) & (windows["image_orientation_coherence"] < hi)
        ].copy()
        if sub.empty:
            continue
        midpoint = (lo + min(hi, 1.0)) / 2.0
        sub["dist_to_mid"] = (sub["image_orientation_coherence"] - midpoint).abs()
        sub = sub.sort_values(["dist_to_mid", "session", "trial_idx"])
        chosen = sub.iloc[0].copy()
        chosen["coherence_bin"] = label
        chosen["coherence_bin_low"] = lo
        chosen["coherence_bin_high"] = min(hi, 1.0)
        rows.append(chosen)
    return pd.DataFrame(rows)


def build_cache(out_dir: Path = OUT_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_representative_rows()

    patches = []
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    for _, row in selected.iterrows():
        key = (str(row["session"]), int(row["trial_idx"]))
        if key not in canvas_cache:
            canvas_cache[key] = _backimage_canvas(*key)
        canvas, ppd, screen_shape = canvas_cache[key]
        center_px = gaze_deg_to_screen_px(
            np.asarray([float(row["mean_x_deg"]), float(row["mean_y_deg"])]), ppd=ppd, screen_shape=screen_shape
        )
        patch = _clip_patch(canvas, (float(center_px[0]), float(center_px[1])), CROP_SIZE_PX)
        patches.append(normalize_image(patch).astype(np.float32))

    patches_arr = np.stack(patches, axis=0)
    coherence_values = selected["image_orientation_coherence"].to_numpy(dtype=np.float64)
    radius_px = selected["image_patch_radius_px"].to_numpy(dtype=np.float64)
    bin_labels = selected["coherence_bin"].to_numpy(dtype=object)

    np.savez(
        CACHE_NPZ,
        patches=patches_arr,
        coherence_values=coherence_values,
        radius_px=radius_px,
        bin_labels=bin_labels,
        crop_size_px=np.asarray([CROP_SIZE_PX]),
    )

    provenance = {
        "source_csv": str(WINDOWS_CSV.relative_to(ROOT)),
        "crop_size_px": CROP_SIZE_PX,
        "selection_rule": "row per COHERENCE_BANDS bin closest to the bin midpoint coherence value",
        "examples": [
            {
                "coherence_bin": str(row["coherence_bin"]),
                "image_orientation_coherence": float(row["image_orientation_coherence"]),
                "session": str(row["session"]),
                "trial_idx": int(row["trial_idx"]),
                "image_patch_radius_px": float(row["image_patch_radius_px"]),
            }
            for _, row in selected.iterrows()
        ],
    }
    PROVENANCE_JSON.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {"npz": CACHE_NPZ, "provenance_json": PROVENANCE_JSON}


def main() -> None:
    paths = build_cache()
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
