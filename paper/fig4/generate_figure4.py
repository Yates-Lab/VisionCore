#!/usr/bin/env python3
"""Figure 4: single-spike information and contour-relative FEM.

Real compositing, not one shared matplotlib canvas. Every displayed panel (A-H)
is rendered as its own independent figure, sized from
PANEL_BOXES (_fig4_layout.py -- measured directly off
ssi_figure_v2_3.pdf, the Illustrator-fixed reference artwork), then placed
onto a blank page at that measured position via pypdf. No panel's drawing
code shares a coordinate tree with any other panel's, so one panel's
opaque axes background can no longer paint over a neighboring panel's text
-- the recurring failure mode in _fig4_ssi_common.py's nested
ax.inset_axes() approach.

Module names carry the DISPLAYED panel letter. The internal layout keys below
("BC", "D", "EF", "G", "I", "J", "K") are the SOURCE letters from the measured
Illustrator reference, kept so the manifest can state provenance; DISPLAY_SPECS
maps one to the other and is the single source of truth for what a reader sees.

Usage:
    uv run python paper/fig4/generate_figure4.py [--allow-missing]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from textwrap import wrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from pypdf import PdfReader, PdfWriter, Transformation

from VisionCore.paths import VISIONCORE_ROOT as ROOT

import _fig4_paths as _paths

import _fig4_layout as layout  # noqa: E402
import panel_a_motion_schematic  # noqa: E402
import panel_c_contour_relative_stimulus  # noqa: E402
import panel_b_path_bins  # noqa: E402
import panel_d_path_bins  # noqa: E402
import panel_e_rms_excursion  # noqa: E402
import panel_f_unwrapped_edge_coherence  # noqa: E402
import _fig4_panel_header  # noqa: E402
import panel_g_match_advantage  # noqa: E402
import panel_h_patch_radius_alignment_slope  # noqa: E402

OUT_DIR = _paths.FIG_DIR
# One panels directory, used both as the default here and by compose(). These
# had drifted apart -- the default said "panels_v4" while compose() wrote
# "panels_v3" -- so the default was dead and the name was a lie either way.
PANELS_OUT_DIR = _paths.PANELS_DIR
TITLE_TEXT = "Single-spike information and contour-relative FEM"
BCEF_PAIR_LABELS = {
    "BC": ("B", "C"),
    "EF": ("E", "F"),
}
V4_LAYOUT_BOXES = {
    # x, y-from-top, width, height in inches. These are hand-tuned for the
    # combined-panel v4 layout rather than inherited directly from the older
    # Illustrator reference boxes.
    "A": (0.0944, 0.1200, 5.6000, 3.9939),
    "BC": (5.7600, 0.1200, 2.5000, 3.9939),
    "D": (0.0944, 3.9250, 2.9000, 3.8800),
    "EF": (3.0444, 3.9250, 2.6000, 3.8800),
    "G": (5.6849, 3.9250, 2.5962, 3.8800),
    "I": (0.0944, 7.8656, 2.6704, 3.0306),
    "J": (2.9148, 7.8656, 2.6704, 3.0306),
    "K": (5.7352, 7.8656, 2.6704, 3.0306),
}
DISPLAY_SPECS = {
    "A": {
        "label": "A",
        "title": "FEMs sharpen spatial coding",
    },
    "BC": {
        "label": "B",
        "title": "Path length separates low- and\nhigh-SF benefit",
    },
    "D": {
        "label": "C",
        "title": "Local contours define the\nrelevant image axis",
    },
    "EF": {
        "label": "D",
        "title": "Contour alignment exposes a\nhigh-SF limit",
        "xlabel": "path length (arcmin; irrespective of\nspatial footprint)",
    },
    "G": {
        "label": "E",
        "title": "Across-contour spread limits\nhigh-SF benefit",
    },
    "I": {
        "label": "F",
        "title": "Real FEM spread is contour-aligned",
    },
    "J": {
        "label": "G",
        "title": "Contour-matched FEMs beat\nrotations for aligned high-SF units",
    },
    "K": {
        "label": "H",
        "title": "Edge following saturates near\nfoveal scale",
    },
}

SCHEMATIC_REQUIRED_INPUTS = (
    _paths.UNIT_MAPS_NPZ,
    _paths.UNIT_MAPS_SELECTED_PATCH_NPY,
    _paths.UNIT_MAPS_SSI_ALL_UNITS_CSV,
    _paths.UNIT_MAPS_ORIENTATION_GROUPS_CSV,
    _paths.IMAGE_FEATURE_TABLE_CSV,
    _paths.TRACE_XY_NPY,
    _paths.TRACE_COMPONENT_MOVIE_METRICS_CSV,
    _paths.SF_TUNING_UNIT_GROUPS_CSV,
    _paths.SCHEMATIC_FINAL_MAPS_NPZ,
    _paths.SCHEMATIC_FINAL_MAP_UNIT_METRICS_CSV,
    _paths.SCHEMATIC_TRACE_CENTER40_CSV,
    _paths.SCHEMATIC_STIMULUS_PAYLOAD_NPZ,
)

PANEL_REQUIRED_INPUTS = {
    "A": (
        *SCHEMATIC_REQUIRED_INPUTS,
        _paths.PANEL_A_NETWORK_ICON_PDF,
        _paths.PANEL_A_LAYOUT_OVERRIDES_JSON,
    ),
    "BC": (_paths.STORY_PANEL_B_VALUES_CSV,),
    "D": (
        *SCHEMATIC_REQUIRED_INPUTS,
        _paths.PANEL_D_LAYOUT_OVERRIDES_JSON,
        _paths.COHERENCE_GALLERY_NPZ,
    ),
    "EF": (_paths.STORY_PANEL_B_VALUES_CSV,),
    "G": (
        _paths.PATH_BINS_VALUES_CSV,
        _paths.PATH_BINS_LAST_BIN_CONTRASTS_CSV,
        _paths.PATH_BINS_TRACE_BANK_REFERENCE_CSV,
        _paths.PATH_BINS_POPULATIONS_CSV,
    ),
    "I": (
        _paths.EDGE_COHERENCE_PROFILES_CSV,
        _paths.EDGE_COHERENCE_RANDOM_BASELINE_CSV,
    ),
    "J": (_paths.BRIDGE_PREDICTION_BY_COHERENCE_SUMMARY_CSV,),
    "K": (_paths.PATCH_RADIUS_ALIGNMENT_SLOPE_CSV,),
}


def _union_box(*boxes: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    return (left, top, right - left, bottom - top)


def bcef_pair_boxes() -> dict[str, tuple[float, float, float, float]]:
    return {pair_key: V4_LAYOUT_BOXES[pair_key] for pair_key in BCEF_PAIR_LABELS}


def v4_placement_boxes() -> dict[str, tuple[float, float, float, float]]:
    """Final page placement boxes.

    The source panel identities still come from the measured reference layout,
    but v4 uses a deliberate grid tuned after B/C and E/F were consolidated.
    """
    return dict(V4_LAYOUT_BOXES)


def build_title_panel(page_w_in: float, out_dir: Path = PANELS_OUT_DIR) -> Path:
    """The one page-level element that isn't inside any lettered panel --
    built the same way as every panel, so the compositor has exactly one
    code path (no special-casing) for placing it."""
    fig = plt.figure(figsize=(page_w_in, 0.42))
    fig.text(0.5, 0.5, TITLE_TEXT, ha="center", va="center", fontsize=13.5, fontweight="bold")
    out_path = out_dir / "panel_title.pdf"
    fig.savefig(out_path, transparent=True)
    plt.close(fig)
    return out_path


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _summarize_missing_inputs(paths: list[Path]) -> str:
    names = [path.name for path in paths]
    shown = ", ".join(names[:5])
    if len(names) > 5:
        shown = f"{shown}, +{len(names) - 5} more"
    return "\n".join(wrap(shown, width=48))


def _summarize_exception(exc: BaseException) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    if not lines:
        return exc.__class__.__name__
    return "\n".join(wrap(lines[0], width=48))


def _build_missing_panel_pdf(
    key: str,
    *,
    figsize: tuple[float, float],
    out_dir: Path,
    missing: list[Path] | None = None,
    exception: BaseException | None = None,
) -> Path:
    display = DISPLAY_SPECS[key]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    ax.add_patch(
        patches.Rectangle(
            (0.035, 0.045),
            0.93,
            0.87,
            facecolor="#fff8f6",
            edgecolor="#c0392b",
            linewidth=0.9,
        )
    )
    ax.text(0.07, 0.86, display["label"], ha="left", va="center", fontsize=12, fontweight="bold")
    ax.text(
        0.18,
        0.86,
        display["title"].replace("\n", " "),
        ha="left",
        va="center",
        fontsize=8.2,
        color="#222222",
    )
    ax.text(
        0.50,
        0.56,
        "Missing cached input",
        ha="center",
        va="center",
        fontsize=9.6,
        fontweight="bold",
        color="#c0392b",
    )
    if missing:
        detail = _summarize_missing_inputs(missing)
    elif exception is not None:
        detail = _summarize_exception(exception)
    else:
        detail = "No cache details were reported."
    ax.text(
        0.50,
        0.43,
        detail,
        ha="center",
        va="center",
        fontsize=6.8,
        color="#7a2d22",
        linespacing=1.12,
    )
    ax.text(
        0.50,
        0.20,
        "--allow-missing smoke render only",
        ha="center",
        va="center",
        fontsize=6.6,
        color="#7a2d22",
    )

    out_path = out_dir / f"panel_{key.lower()}_missing.pdf"
    fig.savefig(out_path, transparent=True)
    plt.close(fig)
    return out_path


def _build_real_or_missing(
    key: str,
    builder: Callable[[], Path],
    *,
    figsize: tuple[float, float],
    out_dir: Path,
) -> Path:
    missing = [path for path in PANEL_REQUIRED_INPUTS.get(key, ()) if not path.exists()]
    if _paths.ALLOW_MISSING and missing:
        print(
            f"WARNING: rendering placeholder for panel {DISPLAY_SPECS[key]['label']} "
            f"({len(missing)} missing cached input(s)).",
            file=sys.stderr,
        )
        return _build_missing_panel_pdf(key, figsize=figsize, out_dir=out_dir, missing=missing)
    try:
        return builder()
    except (FileNotFoundError, _paths.Fig4MissingInput) as exc:
        if not _paths.ALLOW_MISSING:
            raise
        print(
            f"WARNING: rendering placeholder for panel {DISPLAY_SPECS[key]['label']}: {exc}",
            file=sys.stderr,
        )
        return _build_missing_panel_pdf(key, figsize=figsize, out_dir=out_dir, exception=exc)


def build_all_panels(out_dir: Path = PANELS_OUT_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    placement_boxes = v4_placement_boxes()

    paths: dict[str, Path] = {}
    paths["A"] = _build_real_or_missing(
        "A",
        lambda: panel_a_motion_schematic.build_panel(
            figsize=placement_boxes["A"][2:4],
            out_dir=out_dir,
            panel_label=DISPLAY_SPECS["A"]["label"],
            panel_title=DISPLAY_SPECS["A"]["title"],
        ),
        figsize=placement_boxes["A"][2:4],
        out_dir=out_dir,
    )
    paths["D"] = _build_real_or_missing(
        "D",
        lambda: panel_c_contour_relative_stimulus.build_panel(
            figsize=placement_boxes["D"][2:4],
            out_dir=out_dir,
            panel_label=DISPLAY_SPECS["D"]["label"],
            panel_title=DISPLAY_SPECS["D"]["title"],
        ),
        figsize=placement_boxes["D"][2:4],
        out_dir=out_dir,
    )
    # Panels B and D share their drawing code (_fig4_path_bins) but not their
    # layout, so each owns its own module. This loop used to branch on
    # `pair_key` six times to pick axes box, padding and header placement; those
    # choices now live with the panel they describe.
    for pair_key, panel_module in (("BC", panel_b_path_bins), ("EF", panel_d_path_bins)):
        display = DISPLAY_SPECS[pair_key]
        paths[pair_key] = _build_real_or_missing(
            pair_key,
            lambda pair_key=pair_key, panel_module=panel_module, display=display: panel_module.build_panel(
                figsize=placement_boxes[pair_key][2:4],
                out_dir=out_dir,
                panel_label=display["label"],
                panel_title=display["title"],
                panel_subtitle=display.get("subtitle"),
                xlabel=display.get("xlabel"),
            ),
            figsize=placement_boxes[pair_key][2:4],
            out_dir=out_dir,
        )
    display_g = DISPLAY_SPECS["G"]
    paths["G"] = _build_real_or_missing(
        "G",
        lambda: panel_e_rms_excursion.build_panel(
            out_dir=out_dir,
            figsize=placement_boxes["G"][2:4],
            panel_label=display_g["label"],
            panel_title=display_g["title"],
        )["pdf"],
        figsize=placement_boxes["G"][2:4],
        out_dir=out_dir,
    )
    paths["I"] = _build_real_or_missing(
        "I",
        lambda: panel_f_unwrapped_edge_coherence.build_panel(
            out_dir=out_dir,
            figsize=placement_boxes["I"][2:4],
            label=DISPLAY_SPECS["I"]["label"],
            title=DISPLAY_SPECS["I"]["title"],
        )["pdf"],
        figsize=placement_boxes["I"][2:4],
        out_dir=out_dir,
    )
    paths["J"] = _build_real_or_missing(
        "J",
        lambda: panel_g_match_advantage.build_panel(
            out_dir=out_dir,
            figsize=placement_boxes["J"][2:4],
            label=DISPLAY_SPECS["J"]["label"],
            title=DISPLAY_SPECS["J"]["title"],
        )["pdf"],
        figsize=placement_boxes["J"][2:4],
        out_dir=out_dir,
    )
    paths["K"] = _build_real_or_missing(
        "K",
        lambda: panel_h_patch_radius_alignment_slope.build_panel(
            out_dir=out_dir,
            figsize=placement_boxes["K"][2:4],
            label=DISPLAY_SPECS["K"]["label"],
            title=DISPLAY_SPECS["K"]["title"],
        )["pdf"],
        figsize=placement_boxes["K"][2:4],
        out_dir=out_dir,
    )
    return paths


def build_degraded_banner(page_w_in: float, missing: list, out_dir: Path) -> Path:
    """A red banner naming the missing inputs, stamped across the top of a
    degraded render.

    The console warning is not enough on its own: the PDF is the artifact that
    gets circulated, and a degraded panel A differs from a real one across
    roughly a sixth of the page while looking entirely plausible. Whatever says
    "this is not publishable" has to be on the page.
    """
    fig = plt.figure(figsize=(page_w_in, 0.46))
    names = ", ".join(path.name for path in missing[:3])
    if len(missing) > 3:
        names += f", +{len(missing) - 3} more"
    fig.text(0.5, 0.68, "DEGRADED RENDER - NOT PUBLISHABLE", ha="center", va="center",
             fontsize=12, fontweight="bold", color="#c0392b")
    fig.text(0.5, 0.26, f"{len(missing)} cached input(s) missing: {names}",
             ha="center", va="center", fontsize=7.5, color="#c0392b")
    out_path = out_dir / "panel_degraded_banner.pdf"
    fig.savefig(out_path, transparent=True)
    plt.close(fig)
    return out_path


def _write_sidecars(out_dir: Path, manifest: dict) -> None:
    """Caption, README, and manifest next to the figure, as fig3 does."""
    caption = """Figure 4. Single-spike information and contour-relative fixational eye movements.

(A) Fixational eye movements sharpen spatial coding: a real FEM trace drives the retinal image across a local image contour, and the resulting instantaneous single-spike information (SSI) maps are shown for the model input cube. (B) Trace path length separates low- and high-spatial-frequency benefit; SSI is expressed as a percentage of each cell's own stabilized baseline. (C) Local image contours define the relevant image axis: the contour-relative frame is set by the dominant edge orientation within the analysis aperture. (D) Resolving path length along the contour-relative axes exposes a high-SF limit that total path length obscures. (E) Across-contour spread, not along-contour spread, is what limits the high-SF benefit. (F) Real FEM spread is contour-aligned: the unwrapped position-spread profile departs from a random-orientation baseline, and the departure grows with edge coherence. (G) Contour-matched FEM traces beat orientation-rotated controls for aligned high-SF units, with the advantage concentrated at high edge coherence. (H) Edge following saturates near the foveal spatial scale: the alignment slope flattens as the analysis patch radius grows.
"""
    (out_dir / "figure4_caption.md").write_text(caption, encoding="utf-8")

    readme = """# Figure 4

Generated by `paper/fig4/generate_figure4.py`.

Single-spike information carried by fixational eye movements, resolved in a
contour-relative frame. Panel A draws the model-input cube using figure 3's
panel-A builder (`paper/fig3/generate_fig3a.py`) -- a hard cross-figure
dependency, imported unguarded so that a break in fig3 fails here rather than
silently changing what panel A depicts.

## Layout

Each displayed panel is rendered as an independent PDF and composited onto a
blank page with pypdf, so no panel shares a coordinate tree with another.
Module names carry the **displayed** letter; the internal layout keys retain the
**source** letters measured off `reference/ssi_figure_v2_3.pdf`, and
`DISPLAY_SPECS` in the entry point maps between them.

## Directory layout

- `*.py` here -- the figure itself: `panel_[a-h]_*.py` and the `_fig4_*` modules
  they share (`_fig4_style`, `_fig4_brackets`, `_fig4_broken_axis`,
  `_fig4_schematic_axes`, `_fig4_ssi_common`, `_fig4_contour_schematic`, ...).
  Everything here is on the build's import path.
- `refresh/` -- the analysis that regenerates the cached inputs. Nothing here is
  imported by the build. These are run directly and import the shared modules
  above via `refresh/_fig4_imports.py`, which puts this directory on `sys.path`.
- `fixation_stats/` -- BackImage fixation/eye-movement feature extraction, used
  by `refresh/`; the build touches only `fixation_stats.backimage_canvas`.
- `reference/` -- the regression baseline and the Illustrator source artwork,
  plus the original handoff documentation (marked historical).

## Inputs

All cached inputs live flat under `outputs/cache/` as `fig4_*`.
`_fig4_paths.REFRESH_SOURCES` records which script regenerates each one. The
build preflights every required input and reports all missing files at once.
Several caches come from analyses that need the raw recordings and do not run
in this repo; those are marked `(upstream)`.

Historical handoff bundles can be migrated into the flat cache namespace with
`paper/fig4/stage_cache_overlay.py`. The compact cache tarball is partial; a
full old output tree supplies the lower-root RR100, merged-bank, CI-bearing
path-bin, producer-schema edge-coherence and schematic-stimulus artifacts.

`--allow-missing` renders placeholder panels instead of failing, and stamps a
red "DEGRADED RENDER - NOT PUBLISHABLE" banner across the page. It is for layout
iteration only.

## Regression test

`tests/test_fig4_regression.py` rebuilds the figure and requires a maximum
channel difference of zero against `reference/figure4_reference.pdf`, the
original build this figure was refactored from.

## Outputs
- `figure4.pdf`
- `figure4_caption.md`
- `figure4_manifest.json`
- `panels/` (the per-panel PDFs the compositor places)
"""
    (out_dir / "figure4_README.md").write_text(readme, encoding="utf-8")


def _place(writer: PdfWriter, base_page, source_pdf: Path, x_in: float, y_in_from_top: float, page_h_pt: float) -> None:
    reader = PdfReader(str(source_pdf))
    panel_page = reader.pages[0]
    actual_h_pt = float(panel_page.mediabox.height)
    tx = x_in * 72.0
    ty = page_h_pt - y_in_from_top * 72.0 - actual_h_pt
    base_page.merge_transformed_page(panel_page, Transformation().translate(tx=tx, ty=ty))


def compose(out_dir: Path = OUT_DIR, allow_missing: bool = False) -> dict[str, Path]:
    _paths.ALLOW_MISSING = allow_missing
    # Check every cached input before drawing anything: a run that is going to
    # fail should say so in one shot, listing all of what is absent, rather
    # than dying partway through panel G on whichever file it happened to open
    # first.
    _paths.check_required_inputs()

    out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = PANELS_OUT_DIR if out_dir == OUT_DIR else out_dir / "panels"
    page_w_in, page_h_in = layout.PAGE_SIZE_IN
    page_w_pt, page_h_pt = page_w_in * 72.0, page_h_in * 72.0

    panel_paths = build_all_panels(panels_dir)

    writer = PdfWriter()
    writer.add_blank_page(width=page_w_pt, height=page_h_pt)
    base_page = writer.pages[0]

    placement_boxes = v4_placement_boxes()
    for key in ["A", "BC", "D", "EF", "G", "I", "J", "K"]:
        x_in, y_in, _w_in, _h_in = placement_boxes[key]
        _place(writer, base_page, panel_paths[key], x_in, y_in, page_h_pt)

    # Only stamped when inputs are genuinely absent, so a normal build is
    # byte-for-byte unaffected by this branch existing.
    degraded = _paths.missing_inputs()
    if degraded:
        _place(writer, base_page, build_degraded_banner(page_w_in, degraded, panels_dir),
               0.0, 0.0, page_h_pt)

    out_pdf = out_dir / "figure4.pdf"
    with open(out_pdf, "wb") as f:
        writer.write(f)

    provenance = {
        "figure": "figure4",
        "architecture": "composited: each panel is an independently rendered PDF, placed at a "
        "position in the hand-tuned v4 grid; source panel identities retain the measured "
        "ssi_figure_v2_3.pdf reference boxes for provenance (see _fig4_layout.py, measured "
        "off reference/ssi_figure_v2_3.pdf)",
        "page_size_in": [page_w_in, page_h_in],
        "display_panels": {
            display["label"]: {
                "title": display["title"],
                "subtitle": display.get("subtitle"),
                "source_key": key,
                "source_script": str(panel_paths[key].name),
                "placement_box_in_x_y_w_h": list(placement_boxes[key]),
                "combined_source_letters": list(BCEF_PAIR_LABELS[key]) if key in BCEF_PAIR_LABELS else None,
                "content_note": "formerly Panel H RMS-excursion dose curve" if key == "G" else None,
            }
            for key, display in DISPLAY_SPECS.items()
        },
        "source_layout_panels": {
            letter: {
                "measured_box_in_x_y_w_h": list(layout.PANEL_BOXES[letter]),
                "display_panel": (
                    "B"
                    if letter in {"B", "C"}
                    else "D"
                    if letter in {"E", "F"}
                    else DISPLAY_SPECS[letter]["label"]
                    if letter in DISPLAY_SPECS
                    else None
                ),
            }
            for letter in layout.PANEL_BOXES
        },
        "omitted_layout_boxes": {
            "H": {
                "measured_box_in_x_y_w_h": list(layout.PANEL_BOXES["H"]),
                "reason": "RMS-excursion content moved to the former G placement box after G's explainer was folded into D.",
            }
        },
        "new_layout_boxes": {
            key: {"placement_box_in_x_y_w_h": list(box)}
            for key, box in placement_boxes.items()
        },
        "combined_axis_groups": {
            pair_key: {
                "letters": list(letters),
                "placement_box_in_x_y_w_h": list(placement_boxes[pair_key]),
                "source_script": str(panel_paths[pair_key].name),
            }
            for pair_key, letters in BCEF_PAIR_LABELS.items()
        },
        "output_pdf": _relative_to_root(out_pdf),
    }
    manifest_path = out_dir / "figure4_manifest.json"
    manifest_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_sidecars(out_dir, provenance)

    return {"pdf": out_pdf, "manifest_json": manifest_path}


def parse_args():
    # No --recompute counterpart to fig3's: fig4 composes from cached inputs
    # only, and regenerating those is the job of the scripts in refresh/, which
    # need the raw recordings and run for hours. Conflating the two behind one
    # flag would make an expensive, differently-provisioned operation look like
    # a routine rebuild.
    parser = argparse.ArgumentParser(
        description="Compose figure 4 (single-spike information and contour-relative FEM)."
    )
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory for figure outputs (default: canonical fig4 dir).")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Render placeholder panels for missing cached inputs "
                             "instead of failing. For layout iteration only -- the "
                             "resulting figure is not publishable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = OUT_DIR if args.out_dir is None else Path(args.out_dir)
    paths = compose(out_dir=out_dir, allow_missing=args.allow_missing)
    for key, path in paths.items():
        print(path)


if __name__ == "__main__":
    main()
