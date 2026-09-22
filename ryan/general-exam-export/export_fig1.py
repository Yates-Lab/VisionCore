"""Export figure-1 panel data for the general-exam oral deck.

Figure 1 is the phenomenon: the same image lands on a different retina on every
repeat, so a "repeat" is not a repeat. The talk walks it in two reveals, each of
which needs the *same* underlying arrays plotted twice with only the row order
changed:

    B     gaze cloud during fixation                  (context)
    C     foveal RF contours                          (context)
    D-G   single unit: STA, trial fixations, raster    <- reveal 1
          unsorted -> sorted by gaze, and the PSTH split it produces
    H-J   population: eye traces, raster, PSTH         <- reveal 2

Panels D-G and H-J come straight out of the payload caches that
``generate_fig1d.load_cell_payload`` and ``generate_fig1f.load_panel_payload``
already maintain, so nothing is recomputed here. Panel A is a hand-drawn
Illustrator schematic and is not part of this export.

Ragged per-trial spike times are flattened to a values array plus offsets,
because ``.npz`` cannot hold an object array without pickle.

Usage (on solo):
    uv run --directory ~/v1-fovea/VisionCore ryan/general-exam-export/export_fig1.py
"""

from __future__ import annotations

import numpy as np

from _export_common import add_paper_path, save_panel

add_paper_path("fig1")

import matplotlib                                             # noqa: E402
matplotlib.use("Agg")

from VisionCore.paths import CACHE_DIR                        # noqa: E402
import generate_fig1b as b                                    # noqa: E402
import generate_fig1c as c                                    # noqa: E402
import generate_fig1d as d                                    # noqa: E402
import generate_fig1f as f                                    # noqa: E402


GAZE_CACHE = CACHE_DIR / "fig1_gaze"
RF_CACHE = CACHE_DIR / "fig1_rf_contours"
SINGLE_CACHE = CACHE_DIR / "fig1_single_cell"
POP_CACHE = CACHE_DIR / "fig1_population"

PANEL_B_IMAGE_ID = 18


def _flatten_ragged(seq):
    """Flatten a nested sequence of 1-D arrays into (values, offsets).

    ``offsets`` has len(seq) + 1 entries; item ``i`` is
    ``values[offsets[i]:offsets[i + 1]]``.
    """
    parts = [np.atleast_1d(np.asarray(x, dtype=float)).ravel() for x in seq]
    lengths = [len(p) for p in parts]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    values = np.concatenate(parts) if parts else np.zeros(0)
    return values, offsets


def _ragged_offsets(seq):
    """Offsets alone, for several parallel ragged arrays of matching lengths."""
    lengths = [len(np.atleast_1d(np.asarray(x))) for x in seq]
    return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)


# ── B: gaze distribution during fixation ────────────────────────────
def export_panel_b():
    session_name, eyepos = b.pick_representative_session()
    image, extent_deg = b._load_fixrsvp_stimulus(session_name, image_id=PANEL_B_IMAGE_ID)

    save_panel(
        "fig1_b_gaze",
        {
            "eyepos_deg": np.asarray(eyepos, dtype=np.float32),
            "stimulus_image": np.asarray(image, dtype=np.float32),
            "stimulus_extent_deg": float(extent_deg),
            "view_margin_deg": float(b.EXTENT_VIEW_MARGIN_DEG),
            "fix_radius_deg": float(b.FIX_RADIUS_DEG),
            "analysis_radius_deg": float(b.ANALYSIS_RADIUS_DEG),
            "session": session_name,
            "image_id": PANEL_B_IMAGE_ID,
        },
        source="paper/fig1/generate_fig1b.py:plot_panel_b",
        caches=[GAZE_CACHE],
        notes="2-D histogram of eyepos_deg with filled percentile contours; "
              "reference circles at fix_radius_deg and analysis_radius_deg. "
              "stimulus_image spans +/- stimulus_extent_deg on both axes.",
    )


# ── C: foveal RF contours ───────────────────────────────────────────
def _normalize_hulls(hulls):
    """Yield (cell_id, points) from whichever shape ``hulls`` arrives in."""
    if isinstance(hulls, dict):
        for cid, pts in hulls.items():
            yield int(cid), np.asarray(pts, dtype=float)
        return
    for item in hulls:
        if isinstance(item, (tuple, list)) and len(item) == 2 and np.ndim(item[1]) == 2:
            yield int(item[0]), np.asarray(item[1], dtype=float)
        else:
            yield -1, np.asarray(item, dtype=float)


def export_panel_c():
    by_subject = c._load_all_contours()

    xs, ys, offsets, subjects, sessions, cells = [], [], [0], [], [], []
    for subject, entries in by_subject.items():
        for session_name, hulls in entries:
            for cid, pts in _normalize_hulls(hulls):
                pts = np.asarray(pts, dtype=float).reshape(-1, 2)
                xs.append(pts[:, 0])
                ys.append(pts[:, 1])
                offsets.append(offsets[-1] + len(pts))
                subjects.append(subject)
                sessions.append(session_name)
                cells.append(cid)

    save_panel(
        "fig1_c_rfcontours",
        {
            "contour_x_deg": np.concatenate(xs) if xs else np.zeros(0),
            "contour_y_deg": np.concatenate(ys) if ys else np.zeros(0),
            "contour_offsets": np.asarray(offsets, dtype=np.int64),
            "contour_subject": np.asarray(subjects).astype("U32"),
            "contour_session": np.asarray(sessions).astype("U32"),
            "contour_cell": np.asarray(cells, dtype=np.int64),
        },
        source="paper/fig1/generate_fig1c.py:plot_panel_c",
        caches=[RF_CACHE],
        notes="Convex-hull RF contours, one closed polygon per unit: "
              "contour i is contour_{x,y}_deg[offsets[i]:offsets[i+1]].",
    )


# ── D-G: single-unit gaze-sorted raster ─────────────────────────────
def export_single_unit():
    p = d.load_cell_payload()
    sta = d._sta_centered_in_degrees(p["session"], p["cell"], lag=p["peak_lag"])

    segs = p["segments"]
    spike_values, spike_offsets = _flatten_ragged(p["spike_times_all"])
    tbin_values, tbin_offsets = _flatten_ragged(p["trial_t_bins_all"])

    save_panel(
        "fig1_dg_single_unit",
        {
            # receptive field (panel E left)
            "sta_image": np.asarray(sta["image"], dtype=np.float32),
            "sta_extent_deg": np.asarray(sta["extent"], dtype=float),
            "peak_lag": int(p["peak_lag"]),
            "max_orientation_deg": float(p["max_orientation"]),
            # per-trial data (panels D, F, G)
            "eyepos_all_deg": np.asarray(p["eyepos_all"], dtype=np.float32),
            "robs_cell_all": np.asarray(p["robs_cell_all"], dtype=np.float32),
            "spike_times_values": spike_values,
            "spike_times_offsets": spike_offsets,
            "trial_t_bins_values": tbin_values,
            "trial_t_bins_offsets": tbin_offsets,
            # segment structure: the gaze sort that turns D into F
            "segment_start": np.asarray([s["start"] for s in segs], dtype=np.int64),
            "segment_end": np.asarray([s["end"] for s in segs], dtype=np.int64),
            # Segments hold different numbers of trials, so the per-trial
            # arrays are flattened against one shared offsets array.
            "segment_offsets": _ragged_offsets([s["iix"] for s in segs]),
            "segment_trials": np.concatenate(
                [np.asarray(s["iix"], dtype=np.int64) for s in segs]),
            "segment_distances": np.concatenate(
                [np.asarray(s["distances"], dtype=float) for s in segs]),
            "segment_signed_proj": np.concatenate(
                [np.asarray(s["signed_proj"], dtype=float) for s in segs]),
            "segment_cx": np.asarray([s["cx"] for s in segs], dtype=float),
            "segment_cy": np.asarray([s["cy"] for s in segs], dtype=float),
            "segment_slope": np.asarray([s["slope"] for s in segs], dtype=float),
            "example_segment_idx": int(p["example_segment_idx"]),
            "total_window": np.asarray(p["total_window"], dtype=np.int64),
            "session": p["session"],
            "cell": int(p["cell"]),
        },
        source="paper/fig1/generate_fig1d.py:load_cell_payload",
        caches=[SINGLE_CACHE],
        notes="Segment i owns trials segment_trials[segment_offsets[i]:"
              "segment_offsets[i+1]], with matching slices of "
              "segment_distances / segment_signed_proj. "
              "Unsorted reveal: draw trials in native order. Sorted reveal: "
              "within each segment, order trials by segment_signed_proj "
              "(projection onto the axis orthogonal to the preferred "
              "orientation). Axes and color scale must be identical between "
              "the two frames; only row order changes.",
    )


# ── H-J: population gaze-clustered raster ───────────────────────────
def export_population():
    q = f.load_panel_payload()
    spike_values, spike_offsets, trial_idx, unit_idx = [], [0], [], []
    for t, per_unit in enumerate(q["spike_times_trials"]):
        for u, ts in enumerate(per_unit):
            ts = np.atleast_1d(np.asarray(ts, dtype=float)).ravel()
            spike_values.append(ts)
            spike_offsets.append(spike_offsets[-1] + len(ts))
            trial_idx.append(t)
            unit_idx.append(u)

    tbin_values, tbin_offsets = _flatten_ragged(q["trial_t_bins"])

    save_panel(
        "fig1_hj_population",
        {
            "eyepos_deg": np.asarray(q["eyepos"], dtype=np.float32),
            # ragged spikes, indexed by (trial, unit) pairs
            "spike_times_values": np.concatenate(spike_values) if spike_values else np.zeros(0),
            "spike_times_offsets": np.asarray(spike_offsets, dtype=np.int64),
            "spike_trial_index": np.asarray(trial_idx, dtype=np.int64),
            "spike_unit_index": np.asarray(unit_idx, dtype=np.int64),
            "trial_t_bins_values": tbin_values,
            "trial_t_bins_offsets": tbin_offsets,
            "n_cells": int(len(q["cids"])),
            "cids": np.asarray(q["cids"], dtype=np.int64),
            # the two gaze clusters that drive the reveal
            "cluster_trials": np.asarray(q["iix"], dtype=np.int64),
            "cluster_labels": np.asarray(q["clusters"], dtype=np.int64),
            "segment_start": int(q["segment_start"]),
            "segment_end": int(q["segment_end"]),
            "raster_start": int(q["raster_start"]),
            "raster_end": int(q["raster_end"]),
            "peak_lag": int(q["peak_lag"]),
            "session": q["session"],
        },
        source="paper/fig1/generate_fig1f.py:load_panel_payload",
        caches=[POP_CACHE],
        notes="robs is deliberately omitted: the panel uses it only for the "
              "cell count, exported here as n_cells. cluster_labels (0/1) "
              "index into cluster_trials, not into all trials. Cluster 0 = "
              "lower-rate (blue), cluster 1 = higher-rate (red).",
    )


def main():
    print("== panel B (gaze cloud)")
    export_panel_b()
    print("== panel C (RF contours)")
    export_panel_c()
    print("== panels D-G (single unit)")
    export_single_unit()
    print("== panels H-J (population)")
    export_population()


if __name__ == "__main__":
    main()
