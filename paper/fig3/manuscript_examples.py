"""Select real, paired movie illustrations without using neural effect sizes."""
from copy import deepcopy
import json
import numpy as np


def prepare_examples(assets):
    from DataYatesV1.utils.io import YatesV1Session
    from DataYatesV1.utils.data.datasets import DictDataset
    from _fig3_ablation_data import _center_crop_spatial, build_stabilized_stim
    from _fig3_data import FIG_DIR

    assets = deepcopy(assets)
    session = YatesV1Session(assets.session)
    raw = DictDataset.load(session.sess_dir / "datasets/fixrsvp.dset")
    stim = _center_crop_spatial(raw["stim"].numpy(), (35, 35))
    eye = raw["eyepos"].numpy()
    valid = raw["dpi_valid"].numpy().ravel() > 0
    trial = raw["trial_inds"].numpy().ravel()
    candidates = []
    # Scan at 50-ms increments; require a valid central fixation throughout.
    # Rank on retinal contrast and gaze excursion, independently of the twin.
    for endpoint in range(59, len(stim), 12):
        ix = np.arange(endpoint - 59, endpoint + 1)
        if trial[ix[0]] != trial[endpoint] or not valid[ix].all():
            continue
        if not np.isfinite(eye[ix]).all() or np.max(np.linalg.norm(eye[ix], axis=1)) >= .5:
            continue
        contrast = float(np.std(stim[ix].astype(float), axis=(1, 2)).mean())
        excursion = float(np.max(np.linalg.norm(eye[ix] - eye[endpoint], axis=1)))
        if contrast >= 15 and .02 <= excursion <= .4:
            candidates.append((contrast * excursion, endpoint))
    if not candidates:
        raise ValueError("No valid high-contrast central FixRSVP example")
    finalists = sorted(candidates, reverse=True)[:24]
    global_stim, alignment_maxabs, n_trials = build_stabilized_stim(
        assets.session, ((stim.astype(np.float32)-127)/255)[:, None], 1)
    if alignment_maxabs:
        raise AssertionError("Global stabilization is not aligned with the recorded movie")
    global_pixels = np.rint(global_stim[:, 0] * 255 + 127)
    scored = []
    for _, endpoint in finalists:
        stable = global_pixels[endpoint-59:endpoint+1]
        moving = stim[endpoint-59:endpoint+1].astype(np.float32)
        rms = float(np.sqrt(np.mean((moving-stable)**2)))
        scored.append((rms, endpoint, moving, stable))
    rms, endpoint, moving, stable = max(scored, key=lambda row: row[0])
    assets.lag_cube = moving
    assets.stab_lag_cube = stable
    # All three cached candidates were selected for held-out prediction quality.
    # Choose the clearest evoked transient for this small schematic trace.
    strengths = [float((np.percentile(p["rhat_rate"], 95) - np.percentile(p["rhat_rate"], 20)) /
                       max(np.mean(p["rhat_rate"]), 1e-8)) for p in assets.psth_neurons]
    selected = int(np.argmax(strengths))
    assets.psth_neurons = [assets.psth_neurons[selected]] + [
        p for i, p in enumerate(assets.psth_neurons) if i != selected]
    provenance = {"session": assets.session, "trial": int(trial[endpoint]),
                  "native_endpoint": endpoint, "history_frames": 60, "sample_rate_hz": 240,
                  "selection": "24 largest contrast-times-excursion candidates; largest moving/stabilized pixel RMS",
                  "n_eligible_windows": len(candidates), "n_finalists": len(finalists),
                  "moving_stabilized_rms_pixel_levels": rms,
                  "stabilization": "session-global centroid ROI from the production ablation; RSVP flashes preserved",
                  "current_frame_identical": bool(np.array_equal(moving[-1], stable[-1])),
                  "display_frame_replacement": False,
                  "psth_selection": "largest predicted transient/mean among three cached high-CCnorm examples",
                  "psth_candidate_scores": strengths, "psth_selected_index": selected,
                  "render_audit": {"anchor": "session_global", "alignment_maxabs": alignment_maxabs,
                                   "n_trials_stabilized": n_trials}}
    assets.manuscript_example_provenance = provenance
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "example_selection.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return assets
