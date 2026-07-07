"""Load natural image patches from the backimage dataset for in-silico drift.

Renders BackImage natural images, resamples to the model's PPD, and returns
full-field (patch_px x patch_px) grayscale crops (0..255) suitable as the static
stimulus that the gaze-shift -> OUT_SIZE crop pipeline drifts over.
"""
from __future__ import annotations

import numpy as np

from _pop_common import PPD


def load_backimage_patches(n_patches: int, rng: np.random.Generator,
                           session: str = "Allen_2022-04-08",
                           target_ppd: float = PPD, patch_px: int = 540,
                           margin: int = 60, min_std: float = 5.0) -> np.ndarray:
    """Return (n_patches, patch_px, patch_px) natural patches at target_ppd."""
    from DataYatesV1.utils.io import get_session
    from DataYatesV1.exp.backimage import BackImageTrial
    from models.data.datasets import DictDataset
    from PIL import Image

    subj, date = session.split("_", 1)
    sess = get_session(subj, date)
    exp = sess.exp
    ppd = float(exp["S"]["pixPerDeg"])

    dset = DictDataset.load(str(sess.sess_dir / "datasets" / "backimage.dset"))
    trial_ids = np.unique(np.asarray(dset.covariates["trial_inds"]).ravel().astype(int))
    del dset
    scale = target_ppd / ppd

    patches = []
    tries = 0
    max_tries = n_patches * 40
    while len(patches) < n_patches and tries < max_tries:
        tries += 1
        ti = int(rng.choice(trial_ids))
        try:
            img = BackImageTrial(exp["D"][ti], exp["S"]).get_image()
        except Exception:
            continue
        if img.ndim == 3:
            img = img.mean(axis=2)
        img = img.astype(np.float32)
        if abs(scale - 1.0) > 1e-3:
            new = (int(round(img.shape[1] * scale)), int(round(img.shape[0] * scale)))
            img = np.asarray(Image.fromarray(img).resize(new, resample=2), dtype=np.float32)
        H, W = img.shape
        if H < patch_px + 2 * margin or W < patch_px + 2 * margin:
            continue
        y = int(rng.integers(margin, H - patch_px - margin))
        x = int(rng.integers(margin, W - patch_px - margin))
        patch = img[y:y + patch_px, x:x + patch_px]
        if patch.std() < min_std:            # skip near-uniform (background) crops
            continue
        patches.append(np.clip(patch, 0, 255).astype(np.float32))
    if len(patches) < n_patches:
        print(f"  WARNING: only {len(patches)}/{n_patches} patches from {session}")
    return np.stack(patches)
