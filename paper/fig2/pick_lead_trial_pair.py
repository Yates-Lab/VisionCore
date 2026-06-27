"""
Pick a cleaner example trial pair for the Figure 2 lead-in (panels A/B).

Reuses the existing scan cache (``fig2_lead_pair_scan_<session>.pkl`` produced
by the old ryan/fig2 picker), so no data prep is needed.

Selection vs. the old picker adds the constraint that the meeting flagged: the
committed pair (49, 68) has a microsaccade in BOTH trials at ~290 ms, so the two
horizontal traces step at the same instant and read as one horizontal + one
vertical line. The new score therefore PENALIZES coincident saccades (peaks in
the two trials close in time) and REWARDS a clear time separation between each
trial's saccades, while still requiring -- as before -- a small-Delta-e
(matched) window and a large-Delta-e (divergent) window on the chosen axis, plus
a visibly different spike train.

Outputs a multi-page PDF of the top candidates (eye trace on the chosen axis +
offset spike trains, matched/divergent windows marked) for visual selection.

Run:
    uv run paper/fig2/pick_lead_trial_pair.py
"""
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from VisionCore.paths import CACHE_DIR
from _panel_common import FIG_DIR
from generate_panel_example import SESSION, _binned_rate

WINDOW_BINS = 72              # 600 ms @ 120 Hz -- matches the plotted panel
DT = 1.0 / 120.0
WIN_HALF = 3                  # half-width (bins) of the matched/divergent windows
TOP_K = 16
LEAD_UNIT = 110              # v3 example cell (chosen via survey_lead_cells.py)

SACC_SPEED_THRESH = 30.0      # deg/s peak speed to count as a saccade
SACC_MIN_SEP_BINS = 6         # separate a trial's own events by >= this many bins
SACC_COUNT_LO, SACC_COUNT_HI = 1, 3   # acceptable saccade count per trial in-window
SACC_PAD_BINS = 3             # bins around a saccade also treated as non-fixation

COINCIDENCE_BINS = 6          # cross-trial saccade peaks within this -> coincident
COINCIDENCE_PENALTY = 0.25    # deg-equivalent penalty per coincident saccade pair
TIME_SEP_BONUS = 0.004        # reward per bin of close/far window separation
SPIKE_DIFF_BONUS = 0.15       # reward for a distinguishable spike train (normalized)

PAIR_SCAN_CACHE = CACHE_DIR / f"fig2_lead_pair_scan_{SESSION}.pkl"


def _speed(eye_trial):
    """Central-difference eye speed (deg/s). eye_trial: (W, 2)."""
    e = eye_trial
    v = np.zeros_like(e)
    v[1:-1] = (e[2:] - e[:-2]) / (2.0 * DT)
    v[0] = (e[1] - e[0]) / DT
    v[-1] = (e[-1] - e[-2]) / DT
    return np.sqrt((v ** 2).sum(axis=-1))


def _fixation_mask(eye_trial, pad=SACC_PAD_BINS):
    """Boolean (W,): True where the eye is fixating (not in/near a saccade).
    Bins above the speed threshold are dilated by ``pad`` so window centers
    can't land on a saccade's flank."""
    sacc = _speed(eye_trial) > SACC_SPEED_THRESH
    mask = sacc.copy()
    for s in range(1, pad + 1):
        mask[s:] |= sacc[:-s]
        mask[:-s] |= sacc[s:]
    return ~mask


def _saccade_peaks(eye_trial):
    """Bins of saccade-like velocity peaks within the window. eye_trial: (W, 2)."""
    speed = _speed(eye_trial)
    above = speed > SACC_SPEED_THRESH
    peaks = []
    i, n = 0, len(speed)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            local = i + int(np.argmax(speed[i:j]))
            if not peaks or (local - peaks[-1]) >= SACC_MIN_SEP_BINS:
                peaks.append(local)
            i = j
        else:
            i += 1
    return peaks


def _coincident_count(peaks_a, peaks_b):
    """Number of saccade peaks in A that have a B peak within COINCIDENCE_BINS."""
    if not peaks_a or not peaks_b:
        return 0
    pb = np.asarray(peaks_b)
    return int(sum(np.min(np.abs(pb - pa)) < COINCIDENCE_BINS for pa in peaks_a))


def _spike_diff(r_a, r_b):
    """Normalized L1 difference between the two spike trains (0 = identical)."""
    denom = float(r_a.sum() + r_b.sum())
    if denom <= 0:
        return 0.0
    return float(np.abs(r_a - r_b).sum() / denom)


def main(unit=LEAD_UNIT):
    out_pdf = FIG_DIR / f"lead_panel_candidates_u{unit}.pdf"
    with open(PAIR_SCAN_CACHE, "rb") as f:
        pkt = pickle.load(f)
    robs = pkt["robs"]
    eyepos = pkt["eyepos"]
    valid_mask = pkt["valid_mask"]
    neuron_mask = np.asarray(pkt["neuron_mask"])
    j = int(np.where(neuron_mask == unit)[0][0])

    W = WINDOW_BINS
    n_trials = robs.shape[0]
    full_valid = valid_mask[:, :W].all(axis=1)

    peaks = [None] * n_trials
    fixmask = [None] * n_trials
    for t in range(n_trials):
        if full_valid[t]:
            peaks[t] = _saccade_peaks(eyepos[t, :W])
            fixmask[t] = _fixation_mask(eyepos[t, :W])
    sacc_ok = np.array([
        full_valid[t] and (SACC_COUNT_LO <= len(peaks[t]) <= SACC_COUNT_HI)
        for t in range(n_trials)
    ])
    eligible = np.where(sacc_ok)[0]
    print(f"session={SESSION}  unit={unit}  trials={n_trials}  "
          f"full-valid={full_valid.sum()}  eligible={len(eligible)}")
    if len(eligible) < 2:
        print("Not enough eligible trials; relax SACC_COUNT bounds.")
        return

    # Window center is valid only if the whole [t-WIN_HALF, t+WIN_HALF] span is
    # a fixation in BOTH trials -- so matched/divergent windows never sit on (or
    # on the flank of) a saccade. Precompute the per-trial "window fully in
    # fixation" mask by a min-filter of the fixation mask.
    win_ok = [None] * n_trials
    for t in range(n_trials):
        if fixmask[t] is None:
            continue
        fm = fixmask[t]
        ok = fm.copy()
        for s in range(1, WIN_HALF + 1):
            ok[s:] &= fm[:-s]
            ok[:-s] &= fm[s:]
        ok[:WIN_HALF] = False
        ok[W - WIN_HALF:] = False
        win_ok[t] = ok

    candidates = []
    for axis in (0, 1):
        eye_ax = eyepos[..., axis]
        for ai in range(len(eligible)):
            for bi in range(ai + 1, len(eligible)):
                a, b = int(eligible[ai]), int(eligible[bi])
                valid_center = win_ok[a] & win_ok[b]
                if valid_center.sum() < 2:
                    continue
                d = np.abs(eye_ax[a, :W] - eye_ax[b, :W])
                d_masked = np.where(valid_center, d, np.nan)
                t_close = int(np.nanargmin(d_masked))
                t_far = int(np.nanargmax(d_masked))
                if t_close == t_far:
                    continue
                d_close, d_far = float(d[t_close]), float(d[t_far])
                t_sep = abs(t_far - t_close)
                ncoin = _coincident_count(peaks[a], peaks[b])
                _, ra_b = _binned_rate(robs[a, :W, j])
                _, rb_b = _binned_rate(robs[b, :W, j])
                sdiff = _spike_diff(ra_b, rb_b)
                score = (
                    (d_far - d_close)
                    + TIME_SEP_BONUS * t_sep
                    + SPIKE_DIFF_BONUS * sdiff
                    - COINCIDENCE_PENALTY * ncoin
                )
                candidates.append(dict(
                    axis=axis, a=a, b=b, t_close=t_close, t_far=t_far,
                    d_close=d_close, d_far=d_far, t_sep=t_sep,
                    ncoin=ncoin, sdiff=sdiff, score=score,
                ))

    candidates.sort(key=lambda c: -c["score"])
    print(f"\nTop {TOP_K} candidates (coincident-saccade-penalized):")
    print(f"{'axis':>4} {'a':>4} {'b':>4} {'t_cl':>5} {'t_far':>5} "
          f"{'d_cl':>6} {'d_far':>6} {'coin':>4} {'sdiff':>6} {'score':>7}")
    for c in candidates[:TOP_K]:
        print(f"{'h' if c['axis']==0 else 'v':>4} {c['a']:>4} {c['b']:>4} "
              f"{c['t_close']:>5} {c['t_far']:>5} {c['d_close']:>6.3f} "
              f"{c['d_far']:>6.3f} {c['ncoin']:>4} {c['sdiff']:>6.3f} "
              f"{c['score']:>7.3f}")

    t_ms = np.arange(W) * DT * 1000.0
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for c in candidates[:TOP_K]:
            axis, a, b = c["axis"], c["a"], c["b"]
            e_a = eyepos[a, :W, axis]
            e_b = eyepos[b, :W, axis]
            t_spk, r_a = _binned_rate(robs[a, :W, j])
            _, r_b = _binned_rate(robs[b, :W, j])

            fig, axs = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
            fig.suptitle(
                f"axis={'h' if axis==0 else 'v'}  trials=({a},{b})  "
                f"coin={c['ncoin']}  sdiff={c['sdiff']:.3f}  "
                f"score={c['score']:.3f}\n"
                f"matched t={c['t_close']} (Δ={c['d_close']:.3f}°)  "
                f"divergent t={c['t_far']} (Δ={c['d_far']:.3f}°)",
                fontsize=9,
            )
            for win, color, name in (
                (c["t_close"], "0.80", "matched"),
                (c["t_far"], "0.88", "divergent"),
            ):
                t0 = max(0, win - WIN_HALF) * DT * 1000.0
                t1 = min(W, win + WIN_HALF) * DT * 1000.0
                for ax in axs:
                    ax.axvspan(t0, t1, color=color, zorder=-1)
                axs[0].text(0.5 * (t0 + t1), 0.97, name, transform=axs[0].get_xaxis_transform(),
                            ha="center", va="top", fontsize=8)

            axs[0].plot(t_ms, e_a, color="k", lw=1.6, ls="-", label=f"trial {a}")
            axs[0].plot(t_ms, e_b, color="k", lw=1.6, ls="--", label=f"trial {b}")
            for pk in (peaks[a] or []):
                axs[0].axvline(t_ms[pk], color="tab:blue", ls=":", alpha=0.5)
            for pk in (peaks[b] or []):
                axs[0].axvline(t_ms[pk], color="tab:orange", ls=":", alpha=0.5)
            axs[0].set_ylabel(f"eye {'h' if axis==0 else 'v'} (°)")
            axs[0].legend(loc="upper right", fontsize=8, frameon=False)

            ymax = float(max(r_a.max(), r_b.max(), 1e-6))
            offset = 1.25 * ymax
            axs[1].step(t_spk, r_b + offset, color="k", lw=1.0, ls="--",
                        where="mid")
            axs[1].step(t_spk, r_a, color="k", lw=1.0, ls="-", where="mid")
            axs[1].set_xlabel("time from fixation onset (ms)")
            axs[1].set_ylabel("spike rate (offset, 25 ms bins)")
            axs[1].set_yticks([])
            axs[1].set_xlim(0, t_ms[-1])

            pdf.savefig(fig, bbox_inches="tight", dpi=120)
            plt.close(fig)
    print(f"\nSaved {out_pdf}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Pick a lead trial pair for a unit.")
    p.add_argument("--unit", type=int, default=LEAD_UNIT,
                   help=f"Original unit id (default {LEAD_UNIT}).")
    args, _ = p.parse_known_args()
    main(unit=args.unit)
