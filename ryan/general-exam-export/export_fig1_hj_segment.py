"""Export the fixRSVP stimulus behind figure 1's gaze-clustered segment.

The general-exam oral deck opens Part 3's phenomenon at the population level:
two groups of trials whose gaze differed by a third of a degree, and the very
different population responses they produced (manuscript figure 1 H-J). The
talk's version of panel H adds something the printed panel does not have — the
two clusters' gaze positions drawn ON THE IMAGE the animal was viewing, so the
audience can see how little of the picture separates the two conditions.

That needs the stimulus, and ``fig1_hj_population.npz`` does not carry it.
Neither does the fig1f pickle cache: ``generate_fig1f._load_fixrsvp_data``
keeps robs, eyepos, spike times and t-bins, and drops ``image_ids``. So this
export re-runs ``get_fixrsvp_data`` for the population session and renders the
patches on screen during the clustering segment.

WHAT "THE IMAGE" MEANS HERE, AND WHAT IT DOES NOT. fixRSVP flashes a new image
every 12 bins at 240 Hz — 50 ms, 20 Hz. The clustering segment is bins 46-78,
so roughly three images pass during it. There is no single image for the
window, and a slide that implies otherwise is lying about the paradigm. This
export therefore ships EVERY patch shown in the segment, in order, plus the one
the population was most likely responding to at the segment's midpoint
(midpoint minus the population peak lag). A generator may back the gaze scatter
with that one; the speaker says "one of the images flashed during this window".

THE ORIGIN IS THE SESSION GAZE CENTROID, not raw tracker coordinates. Panel B
centres its gaze cloud on the centroid, and the trial movie subtracts it too,
because the residual is a calibration offset and an audience reads any offset
between the marker and the middle of the image as fixational error. Printed
panel H plots raw eyepos, which is fine for a trace against time and wrong for
a marker on a picture. The centroid is exported rather than applied so the deck
generator can subtract it from the traces and the scatter together and stay
self-consistent.

Trial selection is not repeated here. It is imported from
``generate_fig1f`` — the same cluster assignment and the same dropped display
rows that produce the eight rows of panel I — so a trial exported here is a row
of that raster.

Usage (on solo):
    uv run --directory ~/v1-fovea/VisionCore \
        ryan/general-exam-export/export_fig1_hj_segment.py
"""

from __future__ import annotations

import numpy as np

from _export_common import add_paper_path, save_panel

add_paper_path("fig1")

import matplotlib                                             # noqa: E402
matplotlib.use("Agg")

from VisionCore.paths import VISIONCORE_ROOT                  # noqa: E402
from eval.fixrsvp import get_fixrsvp_data                     # noqa: E402
import generate_fig1b as b                                    # noqa: E402
import generate_fig1f as f                                    # noqa: E402


# The population session, from the panel that defines the clusters. NOT
# generate_fig1d's session — the single-unit raster comes from a different day.
SUBJECT = f.SUBJECT
DATE = f.DATE
DATASET_CONFIGS_PATH = f.DATASET_CONFIGS_PATH


def _load_trials():
    data = get_fixrsvp_data(
        SUBJECT, DATE, DATASET_CONFIGS_PATH,
        use_cached_data=True,
        salvageable_mismatch_time_threshold=25,
        verbose=False,
    )
    return (
        np.asarray(data["eyepos"], dtype=np.float64),        # (NT, T, 2) deg
        np.asarray(data["image_ids"], dtype=np.int64),       # (NT, T), -1 = blank
    )


def _density_centroid(eyepos_samples):
    """Centre of the gaze distribution, by the definition panel B uses.

    Copied from ``generate_fig1b.plot_panel_b`` pass 1 by way of
    ``export_trial_movie._density_centroid``: smooth the 2-D gaze histogram,
    take the region holding the top 50% of the mass, and return its
    mass-weighted centroid.
    """
    from scipy.ndimage import gaussian_filter

    pad = 0.5
    edges = np.linspace(-b.HIST_RANGE_DEG - pad, b.HIST_RANGE_DEG + pad,
                        b.HIST_BINS + 1)
    H, xe, ye = np.histogram2d(eyepos_samples[:, 0], eyepos_samples[:, 1],
                               bins=[edges, edges])
    Hs = gaussian_filter(H, sigma=1.5)
    level_50 = b._percentile_levels(Hs, [50])[0]
    mask = Hs >= level_50
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    Xg, Yg = np.meshgrid(xc, yc, indexing="ij")
    w = Hs[mask]
    return np.array([np.average(Xg[mask], weights=w),
                     np.average(Yg[mask], weights=w)])


def _render_stimulus_bank(image_ids_needed):
    """Render each fixRSVP image id to the on-screen patch it produces.

    Mirrors ``generate_fig1b._load_all_fixrsvp_stimuli`` (and is a deliberate
    copy for the reason documented in ``export_trial_movie``: that function
    renders only the curated preferred ids and memoises them under a fixed
    cache name, which widening would poison). The patch is sampled
    symmetrically about ``centerPix`` out to the trial's ``faceRadius``, so it
    carries the same Gaussian aperture over mid-gray the animal sees, and its
    half-width in degrees is ``round(radius*ppd)/ppd``.

    ``image_ids_needed`` are dataset ids (0-indexed); the raw trial history is
    1-indexed, so raw id = dataset id + 1.
    """
    from DataYatesV1.utils.io import get_session
    from DataYatesV1.exp.general import get_trial_protocols
    from DataYatesV1.exp.fix_rsvp import FixRsvpTrial

    sess = get_session(SUBJECT, DATE)
    exp = sess.exp
    protocols = get_trial_protocols(exp)
    idxs = [i for i, p in enumerate(protocols)
            if p == "FixRsvpStim" and FixRsvpTrial.is_valid(exp["D"][i])]
    if not idxs:
        raise RuntimeError(f"No valid FixRsvp trials in {SUBJECT}_{DATE}")

    ppd = float(exp["S"]["pixPerDeg"])
    cx, cy = (int(round(v)) for v in np.asarray(exp["S"]["centerPix"]).ravel()[:2])

    trials = [FixRsvpTrial(exp["D"][i], exp["S"]) for i in idxs]

    patches, half_degs, rendered_ids = [], [], []
    for dataset_id in image_ids_needed:
        raw_id = int(dataset_id) + 1
        for trial in trials:
            hits = np.where(np.asarray(trial.image_ids) == raw_id)[0]
            if len(hits) == 0:
                continue
            half = int(round(float(trial.radius) * ppd))
            roi = np.array([[cy - half, cy + half],
                            [cx - half, cx + half]], dtype=int)
            patch = np.asarray(trial.get_rois(int(hits[0]), roi=roi))
            patches.append(patch.squeeze().astype(np.uint8))
            half_degs.append(half / ppd)
            rendered_ids.append(int(dataset_id))
            break
        else:
            raise RuntimeError(
                f"fixRSVP image id {dataset_id} (raw {raw_id}) is shown during "
                f"the clustering segment but appears in no trial of "
                f"{SUBJECT}_{DATE}"
            )

    half_deg = float(np.unique(np.round(half_degs, 9)).item())
    print(f"  rendered {len(patches)} stimulus patches, "
          f"{patches[0].shape[0]}x{patches[0].shape[1]} px = +/-{half_deg:.3f} deg")
    return (np.stack(patches), np.asarray(rendered_ids, dtype=np.int64), half_deg)


def _displayed_trials():
    """The eight trials of panel I, in display order, with cluster labels."""
    payload = f.load_panel_payload()
    c0, c1 = f._ordered_cluster_trials(payload["iix"], payload["clusters"])
    c0, c1 = f._apply_drop_rows(c0, c1)
    trials = np.asarray(list(c0) + list(c1), dtype=np.int64)
    labels = np.asarray([0] * len(c0) + [1] * len(c1), dtype=np.int64)
    return payload, trials, labels


def export_segment_stimulus():
    payload, trials, labels = _displayed_trials()
    seg_s = int(payload["segment_start"])
    seg_e = int(payload["segment_end"])
    peak_lag = int(payload["peak_lag"])
    print(f"  segment bins [{seg_s}, {seg_e}) = "
          f"{seg_s * f.DT * 1000:.0f}-{seg_e * f.DT * 1000:.0f} ms, "
          f"peak lag {peak_lag} bins ({peak_lag * f.DT * 1000:.0f} ms)")
    print(f"  display trials: {trials.tolist()} "
          f"(cluster {labels.tolist()})")

    eyepos, image_ids = _load_trials()

    # Every displayed trial must have seen the same images at the same times,
    # or the panel's premise — same stimulus, different gaze — is false for
    # this window. fixRSVP replays one fixed sequence, so this should hold;
    # it is asserted rather than assumed because the whole slide rests on it.
    seg_ids = image_ids[np.ix_(trials, np.arange(seg_s, seg_e))]
    unique_rows = np.unique(seg_ids, axis=0)
    if len(unique_rows) != 1:
        raise RuntimeError(
            "displayed trials do not share one image sequence over the "
            f"clustering segment; {len(unique_rows)} distinct sequences found:\n"
            f"{unique_rows}"
        )
    seg_sequence = unique_rows[0]
    if np.any(seg_sequence < 0):
        raise RuntimeError("blank bins inside the clustering segment")
    seg_unique = np.unique(seg_sequence)
    print(f"  segment shows {len(seg_unique)} images: {seg_unique.tolist()}")

    # The image most plausibly driving the population at the middle of the
    # window: the segment's midpoint, walked back by the population peak lag.
    mid_bin = (seg_s + seg_e) // 2
    driving_bin = max(mid_bin - peak_lag, 0)
    driving_id = int(image_ids[trials[0], driving_bin])
    if driving_id < 0:
        raise RuntimeError("no image on screen at the segment's driving bin")
    print(f"  driving image at bin {driving_bin}: id {driving_id}")

    patches, patch_ids, half_deg = _render_stimulus_bank(seg_unique)

    # Origin: the session's own gaze centroid, computed exactly as panel B and
    # the trial movie compute theirs.
    session_samples = eyepos[image_ids >= 0]
    session_samples = session_samples[np.all(np.isfinite(session_samples), axis=1)]
    session_samples = session_samples[
        np.all(np.abs(session_samples) < b.FIX_RADIUS_DEG, axis=1)]
    centroid = _density_centroid(session_samples)
    print(f"  gaze centroid: ({centroid[0]:+.3f}, {centroid[1]:+.3f}) deg "
          f"from {len(session_samples)} fixation samples")

    save_panel(
        "fig1_hj_segment",
        {
            "display_trials": trials,
            "display_cluster": labels,
            "segment_start": seg_s,
            "segment_end": seg_e,
            "peak_lag": peak_lag,
            "segment_image_sequence": seg_sequence.astype(np.int64),
            "stimulus_patches": patches,
            "stimulus_patch_ids": patch_ids,
            "stimulus_half_deg": float(half_deg),
            "driving_image_id": int(driving_id),
            "driving_bin": int(driving_bin),
            "gaze_centroid_deg": centroid.astype(np.float64),
            "fix_radius_deg": float(b.FIX_RADIUS_DEG),
            "analysis_radius_deg": float(b.ANALYSIS_RADIUS_DEG),
            "dt_s": float(f.DT),
            "session": f"{SUBJECT}_{DATE}",
        },
        source="ryan/general-exam-export/export_fig1_hj_segment.py "
               "(eval.fixrsvp.get_fixrsvp_data + DataYatesV1.exp.fix_rsvp.FixRsvpTrial)",
        notes="The fixRSVP stimulus behind figure 1's gaze-clustering segment, "
              "for the talk's population slides. display_trials are the eight "
              "rows of panel I in display order (generate_fig1f clusters, with "
              "DROP_DISPLAY_ROWS applied); they index the trial axis of "
              "fig1_hj_population.npz. All eight share one image sequence over "
              "bins [segment_start, segment_end): segment_image_sequence, one "
              "entry per bin, ids indexing stimulus_patch_ids into "
              "stimulus_patches. Images flash at 20 Hz, so the segment spans "
              "several of them and NO single patch is 'the' stimulus for the "
              "window; driving_image_id is the one on screen at the segment "
              "midpoint minus peak_lag, offered as the representative backdrop. "
              "Each patch spans +/-stimulus_half_deg about screen centre and "
              "carries the on-screen Gaussian aperture over mid-gray. "
              "Subtract gaze_centroid_deg from fig1_hj_population's eyepos_deg "
              "to put the fixation target at patch centre; printed panel H does "
              "not, because a trace against time needs no common origin with an "
              "image.",
    )


def main():
    print(f"== fixRSVP clustering-segment stimulus ({SUBJECT}_{DATE})")
    export_segment_stimulus()


if __name__ == "__main__":
    main()
