"""Quick visual picker for fixRSVP images.

Loads the full marmoset RSVP image library and renders every image in a
grid with its `image_id` labeled, so you can pick a non-aversive one to
put in `PREFERRED_FIXRSVP_IMAGE_IDS` (see `_fig3a_data.py`).

Usage:
    uv run declan/fig3/browse_fixrsvp_images.py
"""
from __future__ import annotations

import math
import re

import matplotlib.pyplot as plt
import numpy as np

from _fig3_data import FIG_DIR


def main():
    from DataYatesV1.exp.support import get_rsvp_fix_stim

    images = get_rsvp_fix_stim()
    pat = re.compile(r"^im(\d+)$")
    entries = []
    for key, val in images.items():
        m = pat.match(key)
        if m is None:
            continue
        img = np.asarray(val)
        if img.ndim == 3:
            img = img.mean(axis=2)
        entries.append((int(m.group(1)), img.astype(np.uint8)))
    entries.sort(key=lambda e: e[0])
    if not entries:
        raise RuntimeError("No images found in rsvpFixStim.mat")

    n = len(entries)
    ncols = 6
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 1.6, nrows * 1.6),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (img_id, img) in zip(axes.ravel(), entries):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"id={img_id}", fontsize=8)

    fig.suptitle(f"fixRSVP image library  ({n} images)", fontsize=10)
    fig.tight_layout(pad=0.4)

    out_png = FIG_DIR / "fixrsvp_image_browser.png"
    out_pdf = FIG_DIR / "fixrsvp_image_browser.pdf"
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
