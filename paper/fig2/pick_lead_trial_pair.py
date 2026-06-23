# %% Scratch: pick a trial pair for the fig2 lead-in panel.
"""
Scan Allen_2022-03-04 fixrsvp trials for unit 151 (orig 151).

Constraints:
    * 0–750 ms window (90 bins @ 120 Hz), all bins valid in BOTH trials.
    * Each trial should have ~2 saccades in the window (microsaccade-ish
      velocity events).
    * Pair should expose a within-window time bin with very small |Δeye|
      AND a separate bin with very large |Δeye|, on the same axis (h or v).

Run:
    uv run VisionCore/ryan/fig2/pick_lead_trial_pair.py
"""
import sys
import pickle

import numpy as np

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR
from VisionCore.covariance import align_fixrsvp_trials

TARGET_SESSION = "Allen_2022-03-04"
TARGET_UNIT_ORIG = 151
WINDOW_BINS = 90          # 750 ms @ 120 Hz
DT = 1.0 / 120.0
TOP_K = 15

SACC_SPEED_THRESH = 30.0  # deg/s — peak speed to count as a saccade
SACC_MIN_SEP_BINS = 6     # separate events by at least this many bins
SACC_COUNT_TARGET = 2     # prefer trials with this many saccades in-window
SACC_COUNT_TOL = 1        # accept SACC_COUNT_TARGET ± tol

PAIRS_CACHE = CACHE_DIR / f"fig2_lead_pair_scan_{TARGET_SESSION}.pkl"


def load_trials():
    if PAIRS_CACHE.exists():
        with open(PAIRS_CACHE, "rb") as f:
            return pickle.load(f)

    if str(VISIONCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(VISIONCORE_ROOT))
    from models.config_loader import load_dataset_configs
    from models.data import prepare_data

    cfg_path = (
        VISIONCORE_ROOT / "experiments" / "dataset_configs"
        / "multi_basic_120_long.yaml"
    )
    dataset_configs = load_dataset_configs(str(cfg_path))
    cfg = next(c for c in dataset_configs if c["session"] == TARGET_SESSION)
    if "fixrsvp" not in cfg["types"]:
        cfg["types"] = cfg["types"] + ["fixrsvp"]

    train_data, _, cfg = prepare_data(cfg, strict=False)
    dset_idx = train_data.get_dataset_index("fixrsvp")
    fixrsvp_dset = train_data.dsets[dset_idx]

    robs, eyepos, valid_mask, neuron_mask, meta = align_fixrsvp_trials(
        fixrsvp_dset, valid_time_bins=120, min_fix_dur=20, min_total_spikes=0,
    )
    payload = {
        "robs": robs, "eyepos": eyepos, "valid_mask": valid_mask,
        "neuron_mask": neuron_mask, "meta": meta,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(PAIRS_CACHE, "wb") as f:
        pickle.dump(payload, f)
    return payload


def count_saccades(eyepos_trial):
    """Count saccade-like velocity events within WINDOW_BINS.

    eyepos_trial: (WINDOW_BINS, 2)
    Returns (count, peak_bins)
    """
    # central-difference velocity in deg/s
    e = eyepos_trial
    v = np.zeros_like(e)
    v[1:-1] = (e[2:] - e[:-2]) / (2.0 * DT)
    v[0] = (e[1] - e[0]) / DT
    v[-1] = (e[-1] - e[-2]) / DT
    speed = np.sqrt((v ** 2).sum(axis=-1))

    # Find peaks > thresh with min separation
    above = speed > SACC_SPEED_THRESH
    peaks = []
    i = 0
    n = len(speed)
    while i < n:
        if above[i]:
            # find local peak in contiguous above-thresh segment
            j = i
            while j < n and above[j]:
                j += 1
            local = i + int(np.argmax(speed[i:j]))
            if not peaks or (local - peaks[-1]) >= SACC_MIN_SEP_BINS:
                peaks.append(local)
            i = j
        else:
            i += 1
    return len(peaks), peaks


def main():
    pkt = load_trials()
    robs = pkt["robs"]
    eyepos = pkt["eyepos"]
    valid_mask = pkt["valid_mask"]
    neuron_mask = pkt["neuron_mask"]

    print(f"session={TARGET_SESSION}  trials={robs.shape[0]}  "
          f"n_time={robs.shape[1]}  units={robs.shape[2]}")

    if TARGET_UNIT_ORIG not in np.asarray(neuron_mask):
        avail = sorted(np.asarray(neuron_mask).tolist())
        print(f"orig unit {TARGET_UNIT_ORIG} not in mask. Available "
              f"orig IDs: {avail[:20]}... (n={len(avail)})")
        return
    j = int(np.where(np.asarray(neuron_mask) == TARGET_UNIT_ORIG)[0][0])
    print(f"unit (post-mask) idx = {j}")

    W = WINDOW_BINS
    n_trials = robs.shape[0]

    # Per-trial: fully valid in window? saccade count?
    full_valid = valid_mask[:, :W].all(axis=1)
    sacc_counts = np.full(n_trials, -1, dtype=int)
    sacc_peaks = [None] * n_trials
    for t in range(n_trials):
        if full_valid[t]:
            c, peaks = count_saccades(eyepos[t, :W])
            sacc_counts[t] = c
            sacc_peaks[t] = peaks

    lo = SACC_COUNT_TARGET - SACC_COUNT_TOL
    hi = SACC_COUNT_TARGET + SACC_COUNT_TOL
    eligible = np.where(full_valid & (sacc_counts >= lo) & (sacc_counts <= hi))[0]
    print(f"\nfull-valid trials: {full_valid.sum()}/{n_trials}")
    print(f"with {lo}-{hi} saccades in window: {len(eligible)}")
    if len(eligible) < 2:
        print("Not enough eligible trials. Relax constraints.")
        # Print histogram of counts for diagnosis
        for k in range(0, 6):
            print(f"  trials with {k} saccades (full_valid): "
                  f"{((sacc_counts == k) & full_valid).sum()}")
        return

    # Score pairs by spread (d_far - d_close) on each axis.
    def scan(eye_axis):
        out = []
        for ai in range(len(eligible)):
            for bi in range(ai + 1, len(eligible)):
                a, b = int(eligible[ai]), int(eligible[bi])
                e_a = eye_axis[a, :W]
                e_b = eye_axis[b, :W]
                d = np.abs(e_a - e_b)
                t_close = int(np.argmin(d))
                t_far = int(np.argmax(d))
                if t_close == t_far:
                    continue
                d_close = float(d[t_close])
                d_far = float(d[t_far])
                # Bonus for bin separation in time (cleaner arrows)
                t_sep = abs(t_far - t_close)
                score = (d_far - d_close) + 0.005 * t_sep
                out.append((a, b, t_close, t_far, d_close, d_far, t_sep, score))
        out.sort(key=lambda x: -x[7])
        return out

    eye_h = eyepos[..., 0]
    eye_v = eyepos[..., 1]

    print("\n--- Horizontal axis: top pairs ---")
    for a, b, tc, tf, dc, df, ts, sc in scan(eye_h)[:TOP_K]:
        spk_a = int(robs[a, :W, j].sum())
        spk_b = int(robs[b, :W, j].sum())
        print(f"  pair=({a:3d},{b:3d})  close_t={tc:3d} (Δ={dc:.3f}°)  "
              f"far_t={tf:3d} (Δ={df:.3f}°)  Δt_bins={ts:3d}  "
              f"sacc=({sacc_counts[a]},{sacc_counts[b]})  "
              f"spk=({spk_a},{spk_b})")

    print("\n--- Vertical axis: top pairs ---")
    for a, b, tc, tf, dc, df, ts, sc in scan(eye_v)[:TOP_K]:
        spk_a = int(robs[a, :W, j].sum())
        spk_b = int(robs[b, :W, j].sum())
        print(f"  pair=({a:3d},{b:3d})  close_t={tc:3d} (Δ={dc:.3f}°)  "
              f"far_t={tf:3d} (Δ={df:.3f}°)  Δt_bins={ts:3d}  "
              f"sacc=({sacc_counts[a]},{sacc_counts[b]})  "
              f"spk=({spk_a},{spk_b})")


if __name__ == "__main__":
    main()
