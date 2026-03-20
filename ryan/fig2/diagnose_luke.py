"""
Diagnostic script: Why do Luke sessions produce bad/missing results in Figure 2?

Checks:
1. fixrsvp data availability per session
2. Trial counts and durations
3. Neuron counts surviving each filter stage
4. Comparison with Allen/Logan sessions
"""
import sys
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from VisionCore.paths import VISIONCORE_ROOT
from VisionCore.covariance import align_fixrsvp_trials, extract_valid_segments
from models.config_loader import load_dataset_configs
from models.data import prepare_data

DATASET_CONFIGS_PATH = VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"
MIN_TOTAL_SPIKES = 500
WINDOW_BINS = [2, 4, 8, 16]
MIN_SEG_LEN = 36

dataset_configs = load_dataset_configs(str(DATASET_CONFIGS_PATH))

print("=" * 80)
print("DIAGNOSTIC: fixRSVP data availability and neuron survival per session")
print("=" * 80)

summary = []

for cfg in dataset_configs:
    session_name = cfg["session"]
    subject = session_name.split("_")[0]
    if subject not in ["Allen", "Logan", "Luke"]:
        continue

    # Ensure fixrsvp is requested
    if "fixrsvp" not in cfg["types"]:
        cfg["types"] = cfg["types"] + ["fixrsvp"]

    print(f"\n{'='*60}")
    print(f"Session: {session_name} ({subject})")
    print(f"{'='*60}")

    try:
        train_data, val_data, cfg_out = prepare_data(cfg, strict=False)
    except Exception as e:
        print(f"  FAILED to load: {e}")
        summary.append({"session": session_name, "subject": subject, "error": str(e)})
        continue

    # Check what datasets are available
    available_types = []
    for i, dset in enumerate(train_data.dsets):
        covs = dset.covariates if hasattr(dset, 'covariates') else dset
        keys = list(covs.keys()) if hasattr(covs, 'keys') else dir(covs)
        n_time = covs['robs'].shape[0] if 'robs' in covs else 0
        n_cells = covs['robs'].shape[1] if 'robs' in covs else 0
        available_types.append((i, n_time, n_cells))
        print(f"  Dataset {i}: T={n_time}, NC={n_cells}, keys={sorted(k for k in covs.keys() if not k.startswith('_'))}")

    # Get fixrsvp
    try:
        dset_idx = train_data.get_dataset_index("fixrsvp")
    except (ValueError, KeyError):
        print("  NO fixrsvp dataset found!")
        summary.append({"session": session_name, "subject": subject, "error": "no fixrsvp"})
        continue

    fixrsvp_dset = train_data.dsets[dset_idx]
    covs = fixrsvp_dset.covariates if hasattr(fixrsvp_dset, 'covariates') else fixrsvp_dset

    robs_flat = np.asarray(covs['robs'])
    eyepos_flat = np.asarray(covs['eyepos'])
    trial_inds = np.asarray(covs['trial_inds']).ravel()
    psth_inds = np.asarray(covs['psth_inds']).ravel()

    n_trials_raw = len(np.unique(trial_inds))
    n_time_total = robs_flat.shape[0]
    n_cells_total = robs_flat.shape[1]
    T_max = int(psth_inds.max()) + 1

    print(f"\n  fixrsvp raw stats:")
    print(f"    Total time bins: {n_time_total}")
    print(f"    Total neurons: {n_cells_total}")
    print(f"    Unique trials: {n_trials_raw}")
    print(f"    Max within-trial bins (T): {T_max}")
    print(f"    Mean spikes/neuron: {robs_flat.sum(0).mean():.1f}")
    print(f"    Median spikes/neuron: {np.median(robs_flat.sum(0)):.1f}")

    # Eye position stats
    eye_dist = np.hypot(eyepos_flat[:, 0], eyepos_flat[:, 1])
    fixation = eye_dist < 1.0
    print(f"    Fixation fraction (r<1deg): {fixation.mean():.3f}")
    print(f"    Eye distance: median={np.median(eye_dist):.3f}, "
          f"mean={eye_dist.mean():.3f}, max={eye_dist.max():.3f}")

    # Run align_fixrsvp_trials
    robs, eyepos, valid_mask, neuron_mask, meta = align_fixrsvp_trials(
        fixrsvp_dset,
        valid_time_bins=120,
        min_fix_dur=20,
        min_total_spikes=MIN_TOTAL_SPIKES,
    )

    print(f"\n  After align_fixrsvp_trials (min_spikes={MIN_TOTAL_SPIKES}):")
    print(f"    Trials: {meta['n_trials_good']}/{meta['n_trials_total']}")
    print(f"    Neurons: {meta['n_neurons_used']}/{meta['n_neurons_total']}")

    if robs is None:
        print("    -> SKIPPED (insufficient data)")
        summary.append({
            "session": session_name, "subject": subject,
            "n_trials_raw": n_trials_raw, "n_time": n_time_total,
            "n_cells": n_cells_total, "trials_good": meta['n_trials_good'],
            "neurons_used": meta['n_neurons_used'], "error": "insufficient after align"
        })
        continue

    # Check valid segments per window
    print(f"\n  Valid segments and windows:")
    segments = extract_valid_segments(valid_mask, min_len_bins=MIN_SEG_LEN)
    print(f"    Valid segments (min_len={MIN_SEG_LEN}): {len(segments)}")
    if segments:
        seg_lens = [s[2] for s in segments]
        print(f"    Segment lengths: min={min(seg_lens)}, max={max(seg_lens)}, "
              f"mean={np.mean(seg_lens):.1f}, total_bins={sum(seg_lens)}")

    # Estimate n_samples per window
    for wb in WINDOW_BINS:
        n_windows_est = sum(max(0, slen - max(wb, wb) + 1) for _, _, slen in segments)
        print(f"    Window {wb} bins ({wb/240*1000:.1f}ms): ~{n_windows_est} samples "
              f"{'OK' if n_windows_est >= 100 else 'TOO FEW (<100)'}")

    # Neuron spike count distribution
    total_spikes_per_neuron = np.nansum(robs[:, :, :], axis=(0, 1))
    n_above_500 = (total_spikes_per_neuron >= 500).sum()
    n_above_200 = (total_spikes_per_neuron >= 200).sum()
    n_above_100 = (total_spikes_per_neuron >= 100).sum()
    print(f"\n  Neuron spike counts (after align, {len(neuron_mask)} neurons):")
    print(f"    >=500 spikes: {n_above_500}")
    print(f"    >=200 spikes: {n_above_200}")
    print(f"    >=100 spikes: {n_above_100}")
    print(f"    Spike count distribution: "
          f"min={total_spikes_per_neuron.min():.0f}, "
          f"median={np.median(total_spikes_per_neuron):.0f}, "
          f"max={total_spikes_per_neuron.max():.0f}")

    # Second neuron filter in metrics extraction (line 275 of generate_figure2.py)
    # This uses per-WINDOW spike counts, not total
    print(f"\n  Per-window neuron survival (second filter in metrics extraction):")
    for wb in WINDOW_BINS:
        # Rough estimate: spikes scale with window size
        # The actual filter uses: erate * n_samples >= MIN_TOTAL_SPIKES
        # where erate is spikes per sample (window)
        pass

    summary.append({
        "session": session_name, "subject": subject,
        "n_trials_raw": n_trials_raw, "n_time": n_time_total,
        "n_cells": n_cells_total,
        "trials_good": meta['n_trials_good'],
        "neurons_used": meta['n_neurons_used'],
        "n_segments": len(segments),
        "total_seg_bins": sum(s[2] for s in segments) if segments else 0,
        "error": None
    })

print("\n\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Session':<25} {'Subject':<8} {'T_total':>8} {'NC':>5} {'Trials':>8} "
      f"{'GoodTr':>8} {'Neurons':>8} {'Segments':>9} {'SegBins':>9} {'Status'}")
print("-" * 120)

for s in summary:
    if s.get("error") and "n_trials_raw" not in s:
        print(f"{s['session']:<25} {s['subject']:<8} {'ERROR: ' + s['error']}")
        continue
    print(f"{s['session']:<25} {s['subject']:<8} "
          f"{s.get('n_time', 'N/A'):>8} {s.get('n_cells', 'N/A'):>5} "
          f"{s.get('n_trials_raw', 'N/A'):>8} {s.get('trials_good', 'N/A'):>8} "
          f"{s.get('neurons_used', 'N/A'):>8} "
          f"{s.get('n_segments', 'N/A'):>9} {s.get('total_seg_bins', 'N/A'):>9} "
          f"{'OK' if not s.get('error') else s['error']}")
