"""Aligned-session cache builder.

One-shot loader that mirrors `legacy.compute_fig2_data._compute_one_session`'s
data-prep phase, then pickles the per-session arrays so every downstream
script can read them without touching `prepare_data` / `DataYatesV1`. After
`cache/aligned_sessions.pkl` exists, the methods folder is self-contained:
both pipelines re-derive their results from the same aligned arrays.

Schema (per session):
    session            str
    subject            str
    robs               (n_trials, valid_time_bins, n_neurons_used) float32
    eyepos             (n_trials, valid_time_bins, 2) float32
    valid_mask         (n_trials, valid_time_bins) bool
    neuron_mask        (n_neurons_used,) int   -- indices into the original
                                                   neuron axis (kept so QC
                                                   indices, which live in the
                                                   pre-mask space, still line
                                                   up).
    rate_hz            (n_neurons_used,) float64 -- firing rate inclusion stat
    psth_r2            (n_neurons_used,) float64 -- split-half PSTH R^2
                                                    inclusion stat (cached so
                                                    no pipeline recomputes it).
    contam_rate        (n_neurons_used,) float64 or None -- min QC contam
                                                            (Allen/Logan only).
    n_trials_total     int                 -- from align_fixrsvp_trials meta
    n_neurons_total    int
    schema_version     int

The cache uses the same constants as `legacy.compute_fig2_data` (DT,
min_total_spikes=0, MIN_RATE_HZ, MIN_PSTH_R2, N_PSTH_SPLITS, fixation_radius=1.0,
min_fix_dur=20, valid_time_bins=120). Sessions that legacy skips are skipped
here too (same try/except).
"""
from __future__ import annotations

import sys
from pathlib import Path

import dill
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
CACHE_PATH = THIS_DIR / "cache" / "aligned_sessions.pkl"

SCHEMA_VERSION = 1
DT = 1 / 120
N_PSTH_SPLITS = 100
VALID_TIME_BINS = 120
MIN_FIX_DUR = 20
MIN_TOTAL_SPIKES = 0
FIXATION_RADIUS = 1.0


def _compute_inclusion_stats(robs):
    """Per-cell rate_hz and split-half PSTH R^2. Matches the legacy formulae
    in `_compute_one_session` exactly."""
    n_units = robs.shape[2]
    n_spikes_per_unit = np.nansum(robs, axis=(0, 1))
    n_valid_bins_per_unit = np.sum(np.isfinite(robs), axis=(0, 1))
    rate_hz = np.where(
        n_valid_bins_per_unit > 0,
        n_spikes_per_unit / np.maximum(n_valid_bins_per_unit, 1) / DT,
        np.nan,
    )

    rng = np.random.default_rng(42)
    r2_sum = np.zeros(n_units)
    r2_cnt = np.zeros(n_units, dtype=int)
    n_trials_r2 = robs.shape[0]
    for _ in range(N_PSTH_SPLITS):
        perm = rng.permutation(n_trials_r2)
        h = n_trials_r2 // 2
        psth_a = np.nanmean(robs[perm[:h]], axis=0)
        psth_b = np.nanmean(robs[perm[h:2 * h]], axis=0)
        for j in range(n_units):
            a, b = psth_a[:, j], psth_b[:, j]
            ok_t = np.isfinite(a) & np.isfinite(b)
            if ok_t.sum() < 3 or np.std(a[ok_t]) == 0 or np.std(b[ok_t]) == 0:
                continue
            r = np.corrcoef(a[ok_t], b[ok_t])[0, 1]
            if np.isfinite(r):
                r2_sum[j] += r * r
                r2_cnt[j] += 1
    psth_r2 = np.where(r2_cnt > 0, r2_sum / np.maximum(r2_cnt, 1), np.nan)
    return rate_hz, psth_r2


def _align_one_session(cfg):
    """Build one aligned-session record. Returns None on skip (matching legacy
    skip semantics)."""
    from VisionCore.paths import VISIONCORE_ROOT
    if str(VISIONCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(VISIONCORE_ROOT))
    from models.data import prepare_data
    from legacy.covariance import align_fixrsvp_trials
    from legacy.compute_fig2_data import _load_contam_rate, SUBJECTS

    session_name = cfg["session"]
    subject = session_name.split("_")[0]
    if subject not in SUBJECTS:
        return None

    if "fixrsvp" not in cfg["types"]:
        cfg["types"] = cfg["types"] + ["fixrsvp"]

    print(f"\n--- {session_name} ({subject}) ---")
    try:
        train_data, _val_data, cfg = prepare_data(cfg, strict=False)
    except Exception as e:
        print(f"  [{session_name}] Skipping: {e}")
        return None

    try:
        dset_idx = train_data.get_dataset_index("fixrsvp")
    except (ValueError, KeyError):
        print(f"  [{session_name}] Skipping: no fixrsvp data")
        return None
    fixrsvp_dset = train_data.dsets[dset_idx]

    robs, eyepos, valid_mask, neuron_mask, meta = align_fixrsvp_trials(
        fixrsvp_dset,
        valid_time_bins=VALID_TIME_BINS,
        min_fix_dur=MIN_FIX_DUR,
        min_total_spikes=MIN_TOTAL_SPIKES,
        fixation_radius=FIXATION_RADIUS,
    )
    if robs is None or robs.shape[0] < 10:
        print(f"  [{session_name}] Skipping: insufficient data ({meta})")
        return None
    print(
        f"  [{session_name}] Trials: {meta['n_trials_good']}/{meta['n_trials_total']}, "
        f"Neurons: {meta['n_neurons_used']}/{meta['n_neurons_total']}"
    )

    rate_hz, psth_r2 = _compute_inclusion_stats(robs)

    try:
        contam_rate = _load_contam_rate(
            session_name, subject, meta["n_neurons_total"]
        )
    except NotImplementedError:
        contam_rate = None

    # Subset contam to the kept neurons (it is in the pre-mask space).
    if contam_rate is not None and neuron_mask is not None:
        contam_rate = np.asarray(contam_rate)[neuron_mask]

    return {
        "session": session_name,
        "subject": subject,
        "robs": robs.astype(np.float32),
        "eyepos": eyepos.astype(np.float32),
        "valid_mask": valid_mask.astype(bool),
        "neuron_mask": neuron_mask,
        "rate_hz": rate_hz,
        "psth_r2": psth_r2,
        "contam_rate": contam_rate,
        "n_trials_total": meta["n_trials_total"],
        "n_trials_good": meta["n_trials_good"],
        "n_neurons_total": meta["n_neurons_total"],
        "n_neurons_used": meta["n_neurons_used"],
        "schema_version": SCHEMA_VERSION,
    }


def build_cache(force=False):
    """Build cache/aligned_sessions.pkl. Returns the list of session dicts."""
    if CACHE_PATH.exists() and not force:
        print(f"Loading cached aligned sessions from {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            return dill.load(f)

    from VisionCore.paths import VISIONCORE_ROOT
    if str(VISIONCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(VISIONCORE_ROOT))
    from models.config_loader import load_dataset_configs
    from legacy.compute_fig2_data import DATASET_CONFIGS_PATH

    cfgs = load_dataset_configs(str(DATASET_CONFIGS_PATH))
    sessions = []
    for cfg in cfgs:
        rec = _align_one_session(cfg)
        if rec is not None:
            sessions.append(rec)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        dill.dump(sessions, f)
    print(f"\nCached {len(sessions)} aligned sessions to {CACHE_PATH}")
    return sessions


def load_cache():
    """Cheap accessor for downstream scripts: errors loudly if not built yet."""
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"{CACHE_PATH} does not exist. Run `uv run python data_loading.py` "
            "to build it once."
        )
    with open(CACHE_PATH, "rb") as f:
        return dill.load(f)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build aligned-session cache.")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild even if cache exists.")
    args = ap.parse_args()
    build_cache(force=args.force)
