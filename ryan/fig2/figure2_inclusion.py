# %% Imports and configuration
"""
Figure 2 inclusion criteria exploration.

Computes per-neuron quality metrics and plots each against 1-alpha
to identify useful inclusion thresholds. Standalone — no dependency
on the figure 2 cache.
"""
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from VisionCore.paths import VISIONCORE_ROOT, FIGURES_DIR
from VisionCore.covariance import (
    align_fixrsvp_trials,
    run_covariance_decomposition,
)
from DataYatesV1 import get_free_device
from DataYatesV1.utils.io import YatesV1Session
from DataYatesV1.exp.general import get_trial_protocols
from DataYatesV1.exp.fix_rsvp import FixRsvpTrial

# Matplotlib publication defaults
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

DT = 1 / 240
WINDOW_BINS = [2]           # smallest window only
N_SHUFFLES = 0              # no shuffles needed for inclusion criteria
MIN_TOTAL_SPIKES = 500
MIN_VAR = 0
SUBJECTS = ["Allen", "Logan"]
DEVICE = get_free_device()
DATASET_CONFIGS_PATH = VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"
FIG_DIR = FIGURES_DIR / "fig2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ONSET_PRE = 0.2      # seconds before trial onset
ONSET_POST = 0.3     # seconds after trial onset
N_SPLITS = 20         # number of random 50/50 splits for PSTH self-consistency

# Detect interactive IPython session
try:
    get_ipython()  # type: ignore[name-defined]
    INTERACTIVE = True
except NameError:
    INTERACTIVE = False


def show_or_close(fig):
    if INTERACTIVE:
        plt.show()
    else:
        plt.close(fig)


# %% Load data and compute per-session metrics

if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))

from models.config_loader import load_dataset_configs
from models.data import prepare_data

dataset_configs = load_dataset_configs(str(DATASET_CONFIGS_PATH))
session_data = []

for cfg in dataset_configs:
    session_name = cfg["session"]
    subject = session_name.split("_")[0]
    if subject not in SUBJECTS:
        continue

    if "fixrsvp" not in cfg["types"]:
        cfg["types"] = cfg["types"] + ["fixrsvp"]

    print(f"\n--- {session_name} ({subject}) ---")
    try:
        train_data, val_data, cfg = prepare_data(cfg, strict=False)
    except Exception as e:
        print(f"  Skipping: {e}")
        continue

    try:
        dset_idx = train_data.get_dataset_index("fixrsvp")
    except (ValueError, KeyError):
        print("  Skipping: no fixrsvp data")
        continue
    fixrsvp_dset = train_data.dsets[dset_idx]

    # Trial-align
    robs, eyepos, valid_mask, neuron_mask, meta = align_fixrsvp_trials(
        fixrsvp_dset,
        valid_time_bins=120,
        min_fix_dur=20,
        min_total_spikes=MIN_TOTAL_SPIKES,
    )
    if robs is None or robs.shape[0] < 10:
        print(f"  Skipping: insufficient data ({meta})")
        continue
    print(f"  Trials: {meta['n_trials_good']}/{meta['n_trials_total']}, "
          f"Neurons: {meta['n_neurons_used']}/{meta['n_neurons_total']}")

    # Covariance decomposition (smallest window only)
    results, mats = run_covariance_decomposition(
        robs, eyepos, valid_mask,
        window_sizes_bins=WINDOW_BINS,
        dt=DT,
        n_shuffles=N_SHUFFLES,
        intercept_mode="lowest_bin",
        seed=42,
        device=str(DEVICE),
    )

    # Load YatesV1Session for raw spike times and true trial onsets
    # The fixrsvp dataset skips the first image per trial, so t_bins starts
    # late. Instead, load the actual FixRsvpTrial objects and use the first
    # flip time converted to ephys clock.
    try:
        sess = YatesV1Session(session_name)
        spike_times = sess.ks_results.spike_times
        spike_clusters = sess.ks_results.spike_clusters
        cids_all = np.unique(spike_clusters)
        cids_used = cids_all[neuron_mask]

        exp = sess.exp
        ptb2ephys = sess.ptb2ephys
        protocols = get_trial_protocols(exp)

        # Build map from original trial index -> true onset in ephys time
        trial_onset_map = {}
        for iT in range(len(exp['D'])):
            if protocols[iT] != 'FixRsvpStim':
                continue
            if not FixRsvpTrial.is_valid(exp['D'][iT]):
                continue
            trial_obj = FixRsvpTrial(exp['D'][iT], exp['S'])
            trial_onset_map[iT] = float(ptb2ephys(trial_obj.flip_times[0]))

        # Match dataset trial_inds to experiment trial indices
        covs = fixrsvp_dset.covariates if hasattr(fixrsvp_dset, 'covariates') else fixrsvp_dset
        trial_inds = np.asarray(covs['trial_inds']).ravel()
        unique_trials = np.unique(trial_inds).astype(int)
        trial_onset_times = np.array([
            trial_onset_map[tid] for tid in unique_trials if tid in trial_onset_map
        ])
        print(f"  True onsets: {len(trial_onset_times)} trials")
    except Exception as e:
        print(f"  Warning: Could not load session data: {e}")
        spike_times = None
        spike_clusters = None
        cids_used = None
        covs = fixrsvp_dset.covariates if hasattr(fixrsvp_dset, 'covariates') else fixrsvp_dset
        t_bins = np.asarray(covs['t_bins']).ravel()
        trial_inds = np.asarray(covs['trial_inds']).ravel()
        unique_trials = np.unique(trial_inds)
        trial_onset_times = np.array([t_bins[trial_inds == tid].min() for tid in unique_trials])

    session_data.append({
        "session": session_name,
        "subject": subject,
        "robs": robs,                         # (n_trials, n_time, n_neurons_used)
        "neuron_mask": neuron_mask,
        "meta": meta,
        "results": results[0],                # smallest window only
        "mats": mats[0],
        "trial_onset_times": trial_onset_times,
        "spike_times": spike_times,
        "spike_clusters": spike_clusters,
        "cids_used": cids_used,
    })

print(f"\nLoaded {len(session_data)} sessions")

# %% Compute 1-alpha (FEM modulation fraction) per neuron

all_one_minus_alpha = []
all_subjects = []

for sd in session_data:
    res = sd["results"]
    mat = sd["mats"]

    Ctotal = mat["Total"]
    Cpsth = mat["PSTH"]
    Crate = mat["Intercept"]

    erate = res["Erates"]
    total_spikes = erate * res["n_samples"]

    valid = (
        np.isfinite(erate)
        & (total_spikes >= MIN_TOTAL_SPIKES)
        & (np.diag(Ctotal) > MIN_VAR)
        & np.isfinite(np.diag(Crate))
        & np.isfinite(np.diag(Cpsth))
    )
    if valid.sum() < 3:
        continue

    alpha = np.clip(np.diag(Cpsth)[valid] / np.diag(Crate)[valid], 0, 1)
    all_one_minus_alpha.append(1 - alpha)
    all_subjects.extend([sd["subject"]] * valid.sum())

    # Store valid mask for use by criteria computations
    sd["valid"] = valid

all_one_minus_alpha = np.concatenate(all_one_minus_alpha)
all_subjects = np.array(all_subjects)
print(f"Total neurons with valid 1-alpha: {len(all_one_minus_alpha)}")

# %% Criterion 1: Contamination rate

all_contam = []

for sd in session_data:
    if "valid" not in sd:
        continue
    valid = sd["valid"]
    subject = sd["subject"]
    session_name = sd["session"]

    try:
        sess = YatesV1Session(session_name)
        refractory = np.load(
            sess.sess_dir / 'qc' / 'refractory' / 'refractory.npz'
        )
        min_contam_props = refractory['min_contam_props']
        contam_rate = np.array([
            np.min(min_contam_props[i]) for i in range(len(min_contam_props))
        ])
        # contam_rate is (n_neurons_total,); neuron_mask selects which neurons
        # are in the decomposition, then valid further filters
        contam_for_decomp = contam_rate[sd["neuron_mask"]]
        all_contam.append(contam_for_decomp[valid] * 100)  # as percentage
    except Exception as e:
        print(f"  {session_name}: contamination unavailable ({e})")
        all_contam.append(np.full(valid.sum(), np.nan))

all_contam = np.concatenate(all_contam)
print(f"Contamination: {np.isfinite(all_contam).sum()}/{len(all_contam)} neurons with data")

# %% Criterion 2: PSTH self-consistency (split-half r²)

all_psth_r2 = []
rng = np.random.default_rng(42)

for sd in session_data:
    if "valid" not in sd:
        continue
    valid = sd["valid"]
    robs = sd["robs"]  # (n_trials, n_time, n_neurons_used)
    n_trials = robs.shape[0]
    n_neurons_used = robs.shape[2]

    # r² for each neuron, averaged over valid splits
    r2_sum = np.zeros(n_neurons_used)
    r2_count = np.zeros(n_neurons_used, dtype=int)

    for _ in range(N_SPLITS):
        perm = rng.permutation(n_trials)
        half = n_trials // 2
        psth_a = np.nanmean(robs[perm[:half]], axis=0)  # (n_time, n_neurons_used)
        psth_b = np.nanmean(robs[perm[half:2*half]], axis=0)

        for j in range(n_neurons_used):
            a, b = psth_a[:, j], psth_b[:, j]
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.sum() > 2 and np.std(a[finite]) > 0 and np.std(b[finite]) > 0:
                r2_sum[j] += np.corrcoef(a[finite], b[finite])[0, 1] ** 2
                r2_count[j] += 1

    r2_per_neuron = np.where(r2_count > 0, r2_sum / r2_count, np.nan)

    # valid mask filters neurons within the decomposition; neuron_mask already
    # selected which neurons from the full dataset are in robs. The valid mask
    # is over the same n_neurons_used axis.
    all_psth_r2.append(r2_per_neuron[valid])

all_psth_r2 = np.concatenate(all_psth_r2)
print(f"PSTH r²: mean={np.nanmean(all_psth_r2):.3f}, "
      f"median={np.nanmedian(all_psth_r2):.3f}")

# %% Criterion 3: Onset responsiveness (raw spike times)

all_mod_idx = []
all_onset_pval = []

for sd in session_data:
    if "valid" not in sd:
        continue
    valid = sd["valid"]
    spike_times = sd["spike_times"]
    spike_clusters = sd["spike_clusters"]
    cids_used = sd["cids_used"]
    trial_onsets = sd["trial_onset_times"]

    if spike_times is None or cids_used is None:
        all_mod_idx.append(np.full(valid.sum(), np.nan))
        all_onset_pval.append(np.full(valid.sum(), np.nan))
        continue

    n_neurons_used = len(cids_used)
    mod_idx = np.full(n_neurons_used, np.nan)
    onset_pval = np.full(n_neurons_used, np.nan)

    for j, cid in enumerate(cids_used):
        st = spike_times[spike_clusters == cid]
        pre_counts = np.empty(len(trial_onsets))
        post_counts = np.empty(len(trial_onsets))

        for t_idx, t0 in enumerate(trial_onsets):
            pre_counts[t_idx] = np.sum((st >= t0 - ONSET_PRE) & (st < t0))
            post_counts[t_idx] = np.sum((st >= t0) & (st < t0 + ONSET_POST))

        mean_pre = pre_counts.mean()
        mean_post = post_counts.mean()
        denom = mean_post + mean_pre
        if denom > 0:
            mod_idx[j] = (mean_post - mean_pre) / denom
        else:
            mod_idx[j] = 0.0

        # Wilcoxon signed-rank test (two-sided)
        if np.any(post_counts != pre_counts):
            _, onset_pval[j] = sp_stats.wilcoxon(post_counts, pre_counts)
        else:
            onset_pval[j] = 1.0

    sd["mod_idx"] = mod_idx
    sd["onset_pval"] = onset_pval
    all_mod_idx.append(mod_idx[valid])
    all_onset_pval.append(onset_pval[valid])

all_mod_idx = np.concatenate(all_mod_idx)
all_onset_pval = np.concatenate(all_onset_pval)
print(f"Onset modulation index: mean={np.nanmean(all_mod_idx):.3f}, "
      f"median={np.nanmedian(all_mod_idx):.3f}")
print(f"Onset significant (p<0.05): "
      f"{np.sum(all_onset_pval < 0.05)}/{np.isfinite(all_onset_pval).sum()}")

# %% Criterion 4: Firing rate

all_firing_rate = []

for sd in session_data:
    if "valid" not in sd:
        continue
    valid = sd["valid"]
    robs = sd["robs"]  # (n_trials, n_time, n_neurons_used)

    # Total spikes and total non-NaN bins per neuron
    n_spikes = np.nansum(robs, axis=(0, 1))  # (n_neurons_used,)
    n_bins = np.sum(np.isfinite(robs[:, :, 0:1] * np.ones((1, 1, robs.shape[2]))),
                    axis=(0, 1))  # broadcast to count per-neuron
    # Simpler: count non-NaN entries per neuron
    n_bins = np.sum(~np.isnan(robs), axis=(0, 1))  # (n_neurons_used,)
    firing_rate = np.where(n_bins > 0, n_spikes / (n_bins * DT), np.nan)

    all_firing_rate.append(firing_rate[valid])

all_firing_rate = np.concatenate(all_firing_rate)
print(f"Firing rate: mean={np.nanmean(all_firing_rate):.1f} Hz, "
      f"median={np.nanmedian(all_firing_rate):.1f} Hz")

# %% Diagnostic plots: criteria vs 1-alpha

SUBJECT_COLORS = {"Allen": "C0", "Logan": "C1"}
N_BINS_RUNNING = 10


def running_median_iqr(x, y, n_bins=N_BINS_RUNNING):
    """Compute running median and IQR of y in quantile bins of x."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < n_bins:
        return np.array([]), np.array([]), np.array([]), np.array([])
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    medians = np.empty(n_bins)
    q25 = np.empty(n_bins)
    q75 = np.empty(n_bins)
    for i in range(n_bins):
        mask = (x >= edges[i]) & (x < edges[i + 1] if i < n_bins - 1 else x <= edges[i + 1])
        if mask.sum() > 0:
            medians[i] = np.median(y[mask])
            q25[i] = np.percentile(y[mask], 25)
            q75[i] = np.percentile(y[mask], 75)
        else:
            medians[i] = q25[i] = q75[i] = np.nan
    return centers, medians, q25, q75


fig, axes = plt.subplots(1, 4, figsize=(20, 5))

criteria = [
    (all_contam, "Contamination rate (%)", "contam"),
    (all_psth_r2, "PSTH self-consistency (r²)", "psth_r2"),
    #(all_mod_idx, "Onset modulation index", "mod_idx"),
    (all_onset_pval, "Onset p-value", "onset_pval"),
    (all_firing_rate, "Firing rate (Hz)", "firing_rate"),
]

for ax, (crit_vals, xlabel, name) in zip(axes, criteria):
    for subj in SUBJECTS:
        mask = all_subjects == subj
        finite = np.isfinite(crit_vals[mask]) & np.isfinite(all_one_minus_alpha[mask])
        ax.scatter(
            crit_vals[mask][finite],
            all_one_minus_alpha[mask][finite],
            alpha=0.3, s=10, label=subj, color=SUBJECT_COLORS[subj],
        )
    # Make x axis log scale for p-values
    if name == "onset_pval":
        ax.set_xscale("log")

    # Running median over all subjects
    centers, medians, q25, q75 = running_median_iqr(crit_vals, all_one_minus_alpha)
    if len(centers) > 0:
        ax.plot(centers, medians, 'k-', lw=2, zorder=5)
        ax.fill_between(centers, q25, q75, color='k', alpha=0.15, zorder=4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("1 - α")
    ax.legend(fontsize=8)

fig.suptitle("Inclusion criteria vs FEM modulation fraction", fontsize=14)
fig.tight_layout()
fig.savefig(FIG_DIR / "inclusion_diagnostics.pdf", bbox_inches="tight")
print(f"Saved to {FIG_DIR / 'inclusion_diagnostics.pdf'}")
show_or_close(fig)

# %% Debug PDF: per-session onset PSTHs

from matplotlib.backends.backend_pdf import PdfPages

PSTH_BIN_S = 0.01  # 10 ms bins for onset PSTH
UNITS_PER_PAGE = 10

psth_edges = np.arange(-ONSET_PRE, ONSET_POST + PSTH_BIN_S, PSTH_BIN_S)
psth_centers = 0.5 * (psth_edges[:-1] + psth_edges[1:])

for sd in session_data:
    if sd["spike_times"] is None or sd["cids_used"] is None:
        continue
    if "mod_idx" not in sd:
        continue

    session_name = sd["session"]
    spike_times = sd["spike_times"]
    spike_clusters = sd["spike_clusters"]
    cids_used = sd["cids_used"]
    trial_onsets = sd["trial_onset_times"]
    mod_idx = sd["mod_idx"]
    onset_pval = sd["onset_pval"]
    n_units = len(cids_used)

    pdf_path = FIG_DIR / f"onset_psth_debug_{session_name}.pdf"
    with PdfPages(pdf_path) as pdf:
        for page_start in range(0, n_units, UNITS_PER_PAGE):
            page_end = min(page_start + UNITS_PER_PAGE, n_units)
            n_on_page = page_end - page_start

            fig, axes = plt.subplots(n_on_page, 1, figsize=(8, 2.5 * n_on_page),
                                     squeeze=False)

            for i, j in enumerate(range(page_start, page_end)):
                ax = axes[i, 0]
                cid = cids_used[j]
                st = spike_times[spike_clusters == cid]

                # Build peri-onset histogram averaged across trials
                counts_per_trial = np.zeros((len(trial_onsets), len(psth_centers)))
                for t_idx, t0 in enumerate(trial_onsets):
                    counts, _ = np.histogram(st - t0, bins=psth_edges)
                    counts_per_trial[t_idx] = counts

                # Convert to firing rate (Hz)
                mean_rate = counts_per_trial.mean(axis=0) / PSTH_BIN_S
                sem_rate = counts_per_trial.std(axis=0) / (np.sqrt(len(trial_onsets)) * PSTH_BIN_S)

                ax.plot(psth_centers * 1000, mean_rate, 'k-', lw=1)
                ax.fill_between(psth_centers * 1000, mean_rate - sem_rate,
                                mean_rate + sem_rate, color='k', alpha=0.2)
                ax.axvline(0, color='r', ls='--', lw=0.8, alpha=0.6)

                # Annotate with metrics
                mi = mod_idx[j]
                pv = onset_pval[j]
                pv_str = f"{pv:.1e}" if np.isfinite(pv) and pv < 0.001 else f"{pv:.3f}"
                ax.set_title(f"unit {cid}  |  mod_idx={mi:.3f}  |  p={pv_str}",
                             fontsize=9, loc='left')
                ax.set_ylabel("Rate (Hz)", fontsize=8)
                if i == n_on_page - 1:
                    ax.set_xlabel("Time from onset (ms)")

            fig.suptitle(f"{session_name} — onset PSTHs", fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"  Saved {pdf_path.name} ({n_units} units)")

print("Onset PSTH debug PDFs complete.")
