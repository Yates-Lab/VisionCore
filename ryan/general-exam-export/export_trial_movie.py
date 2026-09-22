"""Export one fixRSVP trial as a stimulus movie for the general-exam oral deck.

The talk's Part 3 opens by claiming that a "repeat" is not a repeat: the monitor
shows the same flashed image every time, but the eye is somewhere new, so the
retina receives something new. That claim is currently made in words over a
static figure. This export supplies the animation that makes it visible — one
trial, played back with the measured gaze marker riding over the flashed images.

What a fixRSVP trial is, as far as this export is concerned:

    the marmoset holds fixation while natural images flash at 20 Hz (a new
    image every 12 bins at 240 Hz) at screen centre. Eye position is tracked
    at 240 Hz by the DPI. A trial ends when fixation breaks, so trials run
    anywhere from 40 to 508 bins (0.17-2.1 s).

Three arrays carry it: ``image_id`` per time bin (-1 = nothing on screen),
``eyepos`` per time bin in degrees, and a stack of the distinct stimulus
patches those ids index into. The deck's render script composites them.

Stimulus patches are rendered here rather than shipped as ids because the image
bank (``get_rsvp_fix_stim``) and the Gaussian-aperture compositing that puts an
image on the mid-gray screen both live in DataYatesV1, which does not exist on
the deck machine. The render mirrors
``generate_fig1b._load_all_fixrsvp_stimuli`` — same ROI convention, same
``half_deg`` geometry — so a patch exported here drops onto the panel-B gaze
cloud in register. It is a deliberate copy rather than an import: that function
renders only the curated ``PREFERRED_FIXRSVP_IMAGE_IDS`` and memoises them into
a shared cache under a fixed name, and widening it to every id in a trial would
poison panel B's cache.

Three candidate trials are exported, not one. Which trial reads best at slide
scale is a question about the render, not about summary statistics: pure-drift
trials can look like nothing is happening, and microsaccade trials can look like
a failure to fixate. Both kinds are included so the choice is made from the
animation.

Session and trial indexing follow ``paper/fig1/generate_fig1d.py`` — same
subject, same date, same ``get_fixrsvp_data`` call — so trial ``k`` here is
trial ``k`` of the raster the talk shows two slides later.

Usage (on solo):
    uv run --directory ~/v1-fovea/VisionCore ryan/general-exam-export/export_trial_movie.py
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
import generate_fig1d as d                                    # noqa: E402


# The trial movie is drawn from the same session/cell context as the raster
# slides, so the audience is watching the experiment those trials came from.
SUBJECT = d.SUBJECT
DATE = d.DATE
DATASET_CONFIGS_PATH = str(
    VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_240_rsvp.yaml"
)

N_CANDIDATES = 3
# A trial has to be long enough to be worth animating. 240 bins = 1 s = ~20
# flashes; below that the movie is over before the audience has parsed it.
MIN_BINS = 240
# A trial also has to be one the analysis would use: mostly inside the 0.5 deg
# window that generate_fig1d sorts trials within.
MIN_FRAC_IN_WINDOW = 0.8
# Same threshold generate_fig1d uses to call a sample-to-sample jump a
# microsaccade, reported per candidate so the trial choice can be made on
# drift-versus-microsaccade grounds.
MICROSACCADE_THRESHOLD = d.MICROSACCADE_THRESHOLD


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
        np.asarray(data["trial_t_bins"], dtype=np.float64),  # (NT, T) ephys seconds
    )


def _trial_stats(eyepos_trial, image_id_trial, centroid):
    """Summarise one trial: length, image count, gaze extent, microsaccades.

    Gaze is measured relative to ``centroid`` — the fixation target — because
    the criterion that matters is the analysis window, which is defined about
    the target and not about wherever this trial happened to sit.
    """
    on = image_id_trial >= 0
    eye = eyepos_trial[on]
    finite = np.all(np.isfinite(eye), axis=1)
    if not finite.any():
        return None
    eye = eye[finite] - centroid
    steps = np.linalg.norm(np.diff(eye, axis=0), axis=1)
    in_window = np.linalg.norm(eye, axis=1) < b.ANALYSIS_RADIUS_DEG
    return {
        "n_bins": int(on.sum()),
        "n_images": int(len(np.unique(image_id_trial[on]))),
        "n_nonfinite": int((~finite).sum()),
        "n_in_window": int(in_window.sum()),
        "frac_in_window": float(in_window.mean()),
        "offset_deg": float(np.linalg.norm(np.median(eye, axis=0))),
        "x_range_deg": float(np.ptp(eye[:, 0])),
        "y_range_deg": float(np.ptp(eye[:, 1])),
        "n_microsaccades": int((steps > MICROSACCADE_THRESHOLD).sum()),
    }


def _pick_candidates(eyepos, image_ids, centroid):
    """Return the trials to export, ranked by time spent in the analysis window.

    Length alone is the wrong criterion, and picking on it produced a first
    draft whose marker sat in the corner of the frame: the session's longest
    trials are ones where the animal held a steady but *offset* gaze, ~0.9 deg
    below the target. Those trials are long, clean, and excluded from the
    analysis the next slides show. Ranking by in-window samples keeps the movie
    and the raster describing the same trials.
    """
    scored = []
    for t in range(len(image_ids)):
        stats = _trial_stats(eyepos[t], image_ids[t], centroid)
        if stats is None or stats["n_bins"] < MIN_BINS or stats["n_nonfinite"]:
            continue
        if stats["frac_in_window"] < MIN_FRAC_IN_WINDOW:
            continue
        scored.append((stats["n_in_window"], t, stats))
    scored.sort(key=lambda item: -item[0])
    picked = scored[:N_CANDIDATES]
    for _, t, stats in picked:
        print(f"  trial {t:3d}: {stats['n_bins']:3d} bins "
              f"({stats['frac_in_window']:.0%} in window), "
              f"{stats['n_images']:2d} images, "
              f"offset {stats['offset_deg']:.2f} deg, "
              f"gaze {stats['x_range_deg']:.2f} x {stats['y_range_deg']:.2f} deg, "
              f"{stats['n_microsaccades']} microsaccade steps")
    return [t for _, t, _ in picked], [stats for _, _, stats in picked]


def _density_centroid(eyepos_samples):
    """Centre of the gaze distribution, by the definition panel B uses.

    Copied from ``generate_fig1b.plot_panel_b`` pass 1: smooth the 2-D gaze
    histogram, take the region holding the top 50% of the mass, and return its
    mass-weighted centroid. A median would be close but not identical, and the
    movie's marker has to sit where panel B's cloud sits.

    Computed from this session's own samples rather than panel B's
    representative session, which is picked by sample count and need not be
    this one.
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

    Mirrors ``generate_fig1b._load_all_fixrsvp_stimuli``: the patch is sampled
    symmetrically about ``centerPix`` out to the trial's ``faceRadius``, so it
    carries the same Gaussian aperture over mid-gray that the animal sees, and
    its half-width in degrees is ``round(radius*ppd)/ppd``.

    ``image_ids_needed`` are dataset ids (0-indexed). The raw trial history is
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
                f"fixRSVP image id {dataset_id} (raw {raw_id}) is used by an "
                f"exported trial but appears in no trial of {SUBJECT}_{DATE}"
            )

    half_deg = float(np.unique(np.round(half_degs, 9)).item())
    print(f"  rendered {len(patches)} stimulus patches, "
          f"{patches[0].shape[0]}x{patches[0].shape[1]} px = +/-{half_deg:.3f} deg")
    return (np.stack(patches), np.asarray(rendered_ids, dtype=np.int64), half_deg)


def export_trial_movie():
    eyepos, image_ids, t_bins = _load_trials()

    # Calibration offset, not fixational error: panel B centres the gaze cloud
    # on the session centroid before drawing it, and the movie has to use the
    # same origin or the marker will sit off the fixation point for reasons the
    # audience will read as drift. Computed before trial selection, which is
    # defined relative to it.
    session_samples = eyepos[image_ids >= 0]
    session_samples = session_samples[np.all(np.isfinite(session_samples), axis=1)]
    session_samples = session_samples[
        np.all(np.abs(session_samples) < b.FIX_RADIUS_DEG, axis=1)]
    centroid = _density_centroid(session_samples)
    print(f"  gaze centroid: ({centroid[0]:+.3f}, {centroid[1]:+.3f}) deg "
          f"from {len(session_samples)} fixation samples")

    trials, stats = _pick_candidates(eyepos, image_ids, centroid)
    if not trials:
        raise RuntimeError("no trial met the export criteria")

    trials = np.asarray(trials, dtype=np.int64)
    eye = eyepos[trials]
    ids = image_ids[trials]
    times = t_bins[trials]

    # Time relative to each trial's first bin: the deck has no use for the
    # session's ephys clock, and absolute times would only invite a reader to
    # compare them across trials, which means nothing here.
    t_rel = times - times[:, [0]]

    needed = np.unique(ids[ids >= 0])
    patches, patch_ids, half_deg = _render_stimulus_bank(needed)

    save_panel(
        "fig1_trial_movie",
        {
            "trial_index": trials,
            "t_rel_s": t_rel.astype(np.float32),
            "eyepos_deg": eye.astype(np.float32),
            "gaze_centroid_deg": centroid.astype(np.float64),
            "image_id": ids.astype(np.int64),
            "stimulus_patches": patches,
            "stimulus_patch_ids": patch_ids,
            "stimulus_half_deg": float(half_deg),
            "fix_radius_deg": float(b.FIX_RADIUS_DEG),
            "analysis_radius_deg": float(b.ANALYSIS_RADIUS_DEG),
            "microsaccade_threshold_deg": float(MICROSACCADE_THRESHOLD),
            "dt_s": float(d.DT),
            "session": f"{SUBJECT}_{DATE}",
            "trial_n_bins": np.asarray([s["n_bins"] for s in stats], dtype=np.int64),
            "trial_n_images": np.asarray([s["n_images"] for s in stats], dtype=np.int64),
            "trial_n_microsaccades": np.asarray(
                [s["n_microsaccades"] for s in stats], dtype=np.int64),
            "trial_frac_in_window": np.asarray(
                [s["frac_in_window"] for s in stats], dtype=np.float64),
        },
        source="ryan/general-exam-export/export_trial_movie.py "
               "(eval.fixrsvp.get_fixrsvp_data + DataYatesV1.exp.fix_rsvp.FixRsvpTrial)",
        notes="Three candidate fixRSVP trials for the Part 3 stimulus animation. "
              "Row i of every per-trial array is trial_index[i] of the same "
              "get_fixrsvp_data call generate_fig1d uses, so trial numbering "
              "matches the raster panels. image_id is per time bin, -1 = "
              "nothing on screen; ids index stimulus_patch_ids into "
              "stimulus_patches. Each patch spans +/-stimulus_half_deg on both "
              "axes about screen centre and already carries the on-screen "
              "Gaussian aperture over mid-gray. eyepos_deg is raw; subtract "
              "gaze_centroid_deg to put the fixation target at the origin, "
              "which is what panel B does. Trailing bins of a short trial are "
              "padded by the loader: use image_id >= 0 to find the real ones.",
    )


def main():
    print(f"== fixRSVP trial movie ({SUBJECT}_{DATE})")
    export_trial_movie()


if __name__ == "__main__":
    main()
