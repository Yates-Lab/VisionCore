"""
Render a single fixRSVP stimulus to a PNG for inclusion in the fig1a SVG.

The image is rendered exactly as it appears on screen: ``FixRsvpTrial.get_rois``
applies the on-screen Gaussian aperture over the mid-gray screen background, so
the saved patch carries the same windowing the marmoset fixates. No matplotlib
is used — the uint8 grayscale array is written straight to PNG with PIL.

Usage:
    uv run paper/fig1/render_fig1a_stimulus.py [--image-id 18]
"""

import argparse
from pathlib import Path
from PIL import Image, ImageEnhance

from generate_fig1b import pick_representative_session, _load_fixrsvp_stimulus

# The fig1a SVG raster assets (dpieg.png, fig1a_reference_crop.png) live next to
# fig1a.svg, so the stimulus PNG goes here too for the Illustrator workflow.
HERE = Path(__file__).resolve().parent

# Contrast boost over the as-on-screen patch. 1.0 = unchanged; 1.35 = +35%.
# Applied about the patch mean (the mid-gray screen background), so the
# background stays put while the face structure gains contrast.
CONTRAST_FACTOR = 1.35


def render_stimulus(image_id=18, out_path=None):
    name, _ = pick_representative_session()
    image, half_deg = _load_fixrsvp_stimulus(name, image_id=image_id)
    if out_path is None:
        out_path = HERE / f"fig1a_stimulus_im{image_id:02d}.png"
    # `image` is a uint8 grayscale patch with row 0 at the top (screen
    # orientation), spanning +/- half_deg about screen centre.
    img = Image.fromarray(image, mode="L")
    img = ImageEnhance.Contrast(img).enhance(CONTRAST_FACTOR)
    img.save(out_path)
    print(f"Session {name}: image_id={image_id} -> {image.shape[0]}x{image.shape[1]} px "
          f"(+/-{half_deg:.3f} deg), contrast x{CONTRAST_FACTOR:g}")
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Render a fixRSVP stimulus to PNG.")
    p.add_argument("--image-id", type=int, default=18,
                   help="fixRSVP image id to render (default: 18).")
    args = p.parse_args()
    render_stimulus(image_id=args.image_id)
