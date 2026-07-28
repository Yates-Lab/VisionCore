"""
Build the per-session fixRSVP trial-scan caches the fig2 lead-in picker needs.

``pick_lead_example.py`` searches for the panel A/B example over
(unit x trial pair x alignment), but it can only see sessions that have a
``fig2_lead_pair_scan_<session>.pkl`` cache. Historically only the committed
session (Allen_2022-03-04) had one. This script builds the rest, so the search
can range over every session that appears in the figure.

Each cache holds the trial-aligned fixRSVP arrays for one session:
``robs`` (n_trials, n_time, C), ``eyepos`` (n_trials, n_time, 2),
``valid_mask``, ``neuron_mask``, ``meta`` -- exactly the payload
``align_fixrsvp_trials`` returns, matching the existing cache's format.

Sessions without usable fixRSVP data are reported and skipped.

Run:
    uv run paper/fig2/build_lead_scan_caches.py
    uv run paper/fig2/build_lead_scan_caches.py --sessions Allen_2022-02-16 Logan_2020-01-06
    uv run paper/fig2/build_lead_scan_caches.py --refresh
"""
import argparse
import pickle
import sys
import traceback

import numpy as np

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR
from VisionCore.covariance import align_fixrsvp_trials

DATASET_CONFIG = (
    VISIONCORE_ROOT / "experiments" / "dataset_configs"
    / "multi_basic_120_long.yaml"
)
# Alignment parameters copied verbatim from the original scan-cache build
# (ryan/fig2/pick_lead_trial_pair.py) so every session's cache is comparable
# with the committed Allen_2022-03-04 one.
VALID_TIME_BINS = 120
MIN_FIX_DUR = 20
MIN_TOTAL_SPIKES = 0

MIN_TRIALS = 8          # fewer than this and no pair search is worth running


def scan_cache_path(session):
    return CACHE_DIR / f"fig2_lead_pair_scan_{session}.pkl"


def fig2_sessions():
    """Sessions that actually appear in the figure (the derived fig2 bundle)."""
    from compute_fig2_data import load_fig2_data
    data = load_fig2_data()
    return [s["session"] for s in data["session_results"]]


def build_session(session, refresh=False):
    """Build (or load) one session's scan cache. Returns a status string."""
    path = scan_cache_path(session)
    if path.exists() and not refresh:
        with open(path, "rb") as f:
            pkt = pickle.load(f)
        return "cached", pkt

    if str(VISIONCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(VISIONCORE_ROOT))
    from models.config_loader import load_dataset_configs
    from models.data import prepare_data

    dataset_configs = load_dataset_configs(str(DATASET_CONFIG))
    cfg = next((c for c in dataset_configs if c["session"] == session), None)
    if cfg is None:
        return "not-in-config", None
    if "fixrsvp" not in cfg["types"]:
        cfg["types"] = list(cfg["types"]) + ["fixrsvp"]

    train_data, _, cfg = prepare_data(cfg, strict=False)
    try:
        dset_idx = train_data.get_dataset_index("fixrsvp")
    except (KeyError, ValueError, IndexError):
        return "no-fixrsvp", None
    fixrsvp_dset = train_data.dsets[dset_idx]

    robs, eyepos, valid_mask, neuron_mask, meta = align_fixrsvp_trials(
        fixrsvp_dset,
        valid_time_bins=VALID_TIME_BINS,
        min_fix_dur=MIN_FIX_DUR,
        min_total_spikes=MIN_TOTAL_SPIKES,
    )
    if robs is None:
        return "align-failed", None
    if robs.shape[0] < MIN_TRIALS:
        return f"too-few-trials({robs.shape[0]})", None

    payload = {
        "robs": robs, "eyepos": eyepos, "valid_mask": valid_mask,
        "neuron_mask": neuron_mask, "meta": meta,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return "built", payload


def main():
    p = argparse.ArgumentParser(description="Build fig2 lead-in scan caches.")
    p.add_argument("--sessions", nargs="*", default=None,
                   help="Sessions to build (default: every fig2 session).")
    p.add_argument("--refresh", action="store_true",
                   help="Rebuild even if the cache already exists.")
    args, _ = p.parse_known_args()

    sessions = args.sessions if args.sessions else fig2_sessions()
    print(f"Building fixRSVP scan caches for {len(sessions)} session(s).\n")

    rows = []
    for i, session in enumerate(sessions, 1):
        print(f"[{i}/{len(sessions)}] {session} ... ", end="", flush=True)
        try:
            status, pkt = build_session(session, refresh=args.refresh)
        except Exception as exc:                       # noqa: BLE001
            print(f"FAILED ({type(exc).__name__}: {exc})")
            traceback.print_exc(limit=2)
            rows.append((session, "error", 0, 0))
            continue
        if pkt is None:
            print(status)
            rows.append((session, status, 0, 0))
            continue
        n_tr, n_t, n_u = pkt["robs"].shape
        print(f"{status}: {n_tr} trials x {n_t} bins x {n_u} units")
        rows.append((session, status, n_tr, n_u))

    ok = [r for r in rows if r[1] in ("built", "cached")]
    print(f"\n{len(ok)}/{len(rows)} sessions usable "
          f"({sum(r[2] for r in ok)} trials, {sum(r[3] for r in ok)} units total).")
    bad = [r for r in rows if r not in ok]
    if bad:
        print("Skipped:")
        for session, status, _, _ in bad:
            print(f"  {session}: {status}")


if __name__ == "__main__":
    main()
