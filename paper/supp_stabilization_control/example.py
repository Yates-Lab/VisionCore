#!/usr/bin/env python3
"""Extract one measured history and its stabilized counterparts for panel A.

The four retinal movies are rendered from the raw 240-Hz fixRSVP dataset with
the same frozen-ROI rules as the scored replays: paper/fig3/_fig3_ablation_data.py
(session-global and trial-centroid) and paper/fig3/history_stabilization.py
(history endpoint). The result is retained so the figure renders without raw data.

    uv run python paper/supp_stabilization_control/example.py --list drift --session Allen_2022-03-02
    uv run python paper/supp_stabilization_control/example.py \\
        --session Allen_2022-02-16 --trial 12 --endpoint 140
"""
import argparse
from pathlib import Path

import numpy as np

N_LAGS = 60                 # 250 ms at 240 Hz
MODEL_HW = (35, 35)
FIX_RADIUS = 1.0            # deg; match _fig3_ablation_data
CENTROID_RADIUS = 0.5       # deg; match _fig3_ablation_data
SACCADE_SPEED = 8.0         # deg/s; display-only example selection
DRIFT_MAX_SPEED = 3.0       # deg/s on the smoothed trace; excludes microsaccades
CONDITIONS = ("full", "history", "trial", "global")


def center_crop(array, target_hw=MODEL_HW):
    height, width = array.shape[-2:]
    top, left = (height - target_hw[0]) // 2, (width - target_hw[1]) // 2
    return array[..., top:top + target_hw[0], left:left + target_hw[1]]


def centroid_index(dpi_pix, valid):
    """Sample whose gaze is nearest the centroid of the valid samples."""
    indices = np.flatnonzero(valid)
    if not len(indices):
        return None
    centroid = dpi_pix[indices].mean(axis=0)
    return int(indices[np.argmin(((dpi_pix[indices] - centroid) ** 2).sum(axis=1))])


def post_saccade_endpoints(eye_deg, sample_rate=240., delay=10, n_lags=N_LAGS):
    """Samples shortly after an isolated fast gaze shift, with a full history.

    Returns (endpoint, amplitude_deg) pairs. A history qualifies when exactly
    one supra-threshold speed event falls inside it.
    """
    speed = np.hypot(*np.gradient(eye_deg, axis=0).T) * sample_rate
    fast = speed > SACCADE_SPEED
    onsets = np.flatnonzero(fast & ~np.r_[False, fast[:-1]])
    out = []
    for onset in onsets:
        offset = onset
        while offset + 1 < len(fast) and fast[offset + 1]:
            offset += 1
        endpoint = offset + delay
        start = endpoint - n_lags + 1
        if start < 0 or endpoint >= len(eye_deg):
            continue
        inside = onsets[(onsets >= start) & (onsets <= endpoint)]
        if len(inside) != 1:
            continue
        amplitude = float(np.hypot(*(eye_deg[min(offset + 2, len(eye_deg) - 1)]
                                     - eye_deg[max(onset - 2, 0)])))
        out.append((int(endpoint), amplitude))
    return out


def drift_endpoint(eye_deg, sample_rate=240., n_lags=N_LAGS, guard=12, smooth=9):
    """Saccade-free history with the largest net gaze displacement in a trial.

    Returns (endpoint, displacement_deg), or None when no history and its
    `guard` samples on either side stay below DRIFT_MAX_SPEED.
    """
    pad = smooth // 2
    padded = np.pad(eye_deg, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.stack([np.convolve(padded[:, axis], np.ones(smooth) / smooth, mode="valid")
                         for axis in range(2)], axis=1)
    fast = np.hypot(*np.gradient(smoothed, axis=0).T) * sample_rate > DRIFT_MAX_SPEED
    n_fast = np.r_[0, np.cumsum(fast)]
    best = None
    for endpoint in range(n_lags - 1 + guard, len(eye_deg) - guard):
        start = endpoint - n_lags + 1
        if n_fast[endpoint + guard + 1] - n_fast[start - guard]:
            continue
        displacement = float(np.hypot(*(smoothed[endpoint] - smoothed[start])))
        if best is None or displacement > best[1]:
            best = (endpoint, displacement)
    return best


class Session:
    def __init__(self, session_name):
        from DataYatesV1.utils.data.datasets import DictDataset
        from DataYatesV1.utils.general import get_clock_functions
        from DataYatesV1.utils.io import YatesV1Session

        self.name = session_name
        self.session = YatesV1Session(session_name)
        self.ptb2ephys, _ = get_clock_functions(self.session.exp)
        raw = DictDataset.load(self.session.sess_dir / "datasets" / "fixrsvp.dset")
        self.stim = raw["stim"].numpy()
        self.trial_ids = raw["trial_inds"].numpy().astype(int).ravel()
        self.times = raw["t_bins"].numpy().ravel()
        self.rois = raw["roi"].numpy().astype(int)
        self.dpi_pix = raw["dpi_pix"].numpy()
        self.eye = raw["eyepos"].numpy()
        self.dpi_valid = raw["dpi_valid"].numpy().ravel() > 0
        radius = np.hypot(self.eye[:, 0], self.eye[:, 1])
        self.fixation, self.central = radius < FIX_RADIUS, radius < CENTROID_RADIUS
        self.global_index = centroid_index(self.dpi_pix, self.central & self.dpi_valid)

    def trial_index(self, trial_id):
        rows = self.trial_ids == trial_id
        index = centroid_index(self.dpi_pix, rows & self.central & self.dpi_valid)
        if index is None:
            index = centroid_index(self.dpi_pix, rows & self.fixation & self.dpi_valid)
        return index

    def candidates(self, kind="saccade"):
        """Rank histories: post-saccade by frozen-gaze separation, drift by displacement."""
        out = []
        for trial_id in np.unique(self.trial_ids):
            rows = np.flatnonzero(self.trial_ids == trial_id)
            if not (self.dpi_valid[rows] & self.central[rows]).all():
                continue
            anchor = self.trial_index(trial_id)
            if kind == "drift":
                found = drift_endpoint(self.eye[rows])
                endpoints = [found] if found else []
            else:
                endpoints = post_saccade_endpoints(self.eye[rows])
            for endpoint, amplitude in endpoints:
                frozen = self.eye[[rows[endpoint], anchor, self.global_index]] * 60
                gaps = [np.hypot(*(frozen[i] - frozen[j])) for i, j in ((0, 1), (0, 2), (1, 2))]
                out.append({"session": self.name, "trial": int(trial_id), "endpoint": endpoint,
                            "trial_samples": len(rows), "movement_arcmin": amplitude * 60,
                            "min_gap_arcmin": float(min(gaps))})
        key = "movement_arcmin" if kind == "drift" else "min_gap_arcmin"
        return sorted(out, key=lambda record: -record[key])

    def extract(self, trial_id, endpoint):
        """Gaze trace and the four 60-frame retinal movies ending at `endpoint`."""
        from DataYatesV1.exp.fix_rsvp import FixRsvpTrial

        rows = np.flatnonzero(self.trial_ids == trial_id)
        history = rows[endpoint - N_LAGS + 1:endpoint + 1]
        if endpoint - N_LAGS + 1 < 0 or len(history) != N_LAGS:
            raise ValueError("History does not fit inside the trial")
        anchors = {"history": int(history[-1]), "trial": self.trial_index(trial_id),
                   "global": self.global_index}
        trial = FixRsvpTrial(self.session.exp["D"][trial_id], self.session.exp["S"])
        start = np.where(trial.image_ids == 2)[0][0]
        flips = self.ptb2ephys(trial.flip_times[start:])
        images = np.searchsorted(flips, self.times[history], side="right") - 1 + start
        if not np.array_equal(trial.get_rois(images, roi=self.rois[history]), self.stim[history]):
            raise AssertionError("Re-rendered measured history differs from the stored stimulus")
        movies = {"full": self.stim[history]}
        for condition, anchor in anchors.items():
            roi = np.repeat(self.rois[anchor][None], N_LAGS, axis=0)
            movies[condition] = trial.get_rois(images, roi=roi)
        if not np.array_equal(movies["history"][-1], movies["full"][-1]):
            raise AssertionError("History-local stabilization changed the current frame")
        center = self.eye[self.global_index]
        out = {"session": self.name, "trial": trial_id, "endpoint": endpoint,
               "time_ms": (self.times[rows] - self.times[rows[0]]) * 1e3,
               "gaze_arcmin": (self.eye[rows] - center) * 60,
               "image_index": images - images[0]}
        for condition in CONDITIONS:
            out["movie_" + condition] = center_crop(movies[condition])
            if condition != "full":
                out["frozen_arcmin_" + condition] = (self.eye[anchors[condition]] - center) * 60
        return out


def main():
    from VisionCore.paths import VISIONCORE_ROOT

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", nargs="+", required=True)
    parser.add_argument("--list", choices=("saccade", "drift"),
                        help="Rank candidate histories of this kind and exit")
    parser.add_argument("--trial", type=int)
    parser.add_argument("--endpoint", type=int, help="Sample within the trial")
    parser.add_argument("--out", type=Path,
                        default=VISIONCORE_ROOT
                        / "manuscript/analysis/stabilization_control/panel_a_example.npz")
    args = parser.parse_args()
    if args.list:
        key = "movement_arcmin" if args.list == "drift" else "min_gap_arcmin"
        records = sorted((r for name in args.session for r in Session(name).candidates(args.list)),
                         key=lambda record: -record[key])
        for record in records[:25]:
            print("{session} trial={trial} endpoint={endpoint} samples={trial_samples} "
                  "movement={movement_arcmin:.1f}' min_gap={min_gap_arcmin:.1f}'".format(**record))
        return
    example = Session(args.session[0]).extract(args.trial, args.endpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **example)
    print(args.out)


if __name__ == "__main__":
    main()
