"""
Joint search for the Figure 2 lead-in example: (unit x trial pair x alignment).

The committed example was chosen in two independent stages -- ``survey_lead_cells``
ranked units on panel-B criteria alone (rate, Cfem/Ctotal, low sigma_int/Ctotal),
then ``pick_lead_trial_pair`` ranked trial pairs for that one unit on eye-geometry
criteria plus a *global* spike-train difference. Neither stage scored the thing
panel A is meant to demonstrate: that the two spike trains should track each other
where the eye trajectories are matched and diverge where they are not.

This picker scores that contrast directly. For each trial pair it finds the
saccade that brings the two trajectories into alignment, splits the 600 ms
window there into a pre-alignment (large Delta-e) and post-alignment (small
Delta-e) epoch, and for every unit computes the Pearson correlation between the
two trials' 25 ms binned rates within each epoch. The headline quantity is

    contrast = r_post - r_pre

i.e. how much more the spike trains covary once the eyes are following the same
trajectory. Eye geometry, saccade structure, spike counts, and panel-B unit
quality enter as hard gates so the ranking is a pure read on the visual claim.

Outputs a ranked table plus a multipage PDF that previews each candidate through
the *real* panel A/B renderers, so what you see is what the figure would show.

Run:
    uv run paper/fig2/pick_lead_example.py
    uv run paper/fig2/pick_lead_example.py --top 40
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpecFromSubplotSpec

from _panel_common import FIG_DIR
from generate_panel_example import (
    SESSION, DT, WINDOW_BINS, RATE_BIN_FACTOR, _FIG2_RC,
    EXAMPLE_UNIT, TRIAL_A, TRIAL_B, EYE_AXIS, W1, W2,
    _load_trial_pair, _decompose_all_units,
    plot_eye_rate_example, plot_unaccounted_variance_panel,
)
from pick_lead_trial_pair import (
    _saccade_peaks, _fixation_mask, _coincident_count,
    SACC_COUNT_LO, SACC_COUNT_HI, WIN_HALF,
)
from survey_lead_cells import _score_units
from build_lead_scan_caches import fig2_sessions, scan_cache_path

# --- epoch construction -----------------------------------------------------
SPLIT_PAD_BINS = 3          # bins around the aligning saccade excluded from both epochs
MIN_EPOCH_RATE_BINS = 6     # >= 150 ms of 25 ms bins per epoch, so r is meaningful
SACC_MATCH_BINS = 4         # saccade peak must fall within this of the split

# --- eye-geometry gates (deg, on the chosen axis) ---------------------------
# All three are on the *mean* |Delta e| within an epoch, so they are stricter
# than the Delta-e the shaded windows report (those pick the epoch extremum).
MIN_D_PRE = 0.5             # trajectories must be genuinely far apart before alignment
MAX_D_POST = 0.15           # mean |Delta e| after alignment must be small
MIN_D_DROP = 0.25           # mean |Delta e| must fall by at least this across the split

# --- spike gates: keep the correlations off degenerate near-silent epochs ----
MIN_SPIKES_PER_EPOCH = 5    # per trial, per epoch
MIN_NONZERO_BINS = 3        # per trial, per epoch

# --- panel-B quality gates (percentiles among units passing survey_lead_cells) ---
MIN_FEM_FRAC_PCT = 50.0     # Cfem/Ctotal must be at least this percentile
MAX_INT_FRAC_PCT = 75.0     # sigma_int/Ctotal must be at most this percentile
# Panel B draws three levels: total, eye-blind, and the internal-noise floor.
# A unit whose estimated internal floor is ~0 collapses the bottom two into the
# x-axis, so the panel loses the structure it exists to show (and a cell with no
# internal variability at all is an estimation artifact, not a real neuron).
MIN_INT_FRAC = 0.10         # sigma_int/Ctotal floor, absolute

# --- shaded-window placement ------------------------------------------------
EDGE_CLEARANCE_BINS = 8     # keep window centers this far from the panel edges

MIN_R_POST = 0.5            # post-alignment trains must genuinely track


def _pearson_cols(x, y):
    """Per-column Pearson r between (n, C) arrays. NaN where either is constant."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = x - x.mean(0)
    yc = y - y.mean(0)
    denom = np.sqrt((xc ** 2).sum(0) * (yc ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (xc * yc).sum(0) / denom, np.nan)


def _epoch_rate_bins(t_split, n_rate_bins, factor=RATE_BIN_FACTOR,
                     pad=SPLIT_PAD_BINS):
    """Rate-bin indices lying wholly before / after the padded split."""
    starts = np.arange(n_rate_bins) * factor
    ends = starts + factor
    pre = np.flatnonzero(ends <= t_split - pad)
    post = np.flatnonzero(starts >= t_split + pad)
    return pre, post


def _pick_windows(d, t_split, win_ok, half=WIN_HALF):
    """Divergent window in the pre epoch (max Delta e), matched window in the
    post epoch (min Delta e), both restricted to saccade-free centers and kept
    clear of the panel edges (a window flush against the axis reads badly)."""
    n = len(d)
    idx = np.arange(n)
    inset = (idx >= EDGE_CLEARANCE_BINS) & (idx < n - EDGE_CLEARANCE_BINS)
    win_ok = win_ok & inset
    pre_ok = win_ok & (idx + half < t_split - SPLIT_PAD_BINS)
    post_ok = win_ok & (idx - half > t_split + SPLIT_PAD_BINS)
    if not pre_ok.any() or not post_ok.any():
        return None, None
    t_div = int(np.flatnonzero(pre_ok)[np.argmax(d[pre_ok])])
    t_mat = int(np.flatnonzero(post_ok)[np.argmin(d[post_ok])])
    return (t_div - half, t_div + half), (t_mat - half, t_mat + half)


def _unit_gate(allu):
    """Boolean mask over units: survey_lead_cells' quality filter, plus
    percentile gates on the FEM and internal-noise variance fractions. Also
    returns that survey's per-unit panel-B score (0-1) for the composite rank."""
    score, fem_frac, int_frac, valid = _score_units(allu)
    ok = valid.copy()
    if ok.sum() == 0:
        return ok, fem_frac, int_frac, score
    fem_thr = np.nanpercentile(fem_frac[valid], MIN_FEM_FRAC_PCT)
    int_thr = np.nanpercentile(int_frac[valid], MAX_INT_FRAC_PCT)
    ok &= (fem_frac >= fem_thr) & (int_frac <= int_thr) \
        & (int_frac >= MIN_INT_FRAC)
    return ok, fem_frac, int_frac, score


def search_session(session):
    """All candidates for one session. Returns (candidates, allu)."""
    pair = _load_trial_pair(session)
    robs = np.nan_to_num(pair["robs"], nan=0.0)
    eyepos = pair["eyepos"]
    valid_mask = pair["valid_mask"]
    neuron_mask = np.asarray(pair["neuron_mask"])

    allu = _decompose_all_units(session=session)
    if not np.array_equal(np.asarray(allu["neuron_mask"]), neuron_mask):
        raise RuntimeError("Unit ordering differs between the decomposition "
                           "cache and the pair-scan cache.")
    unit_ok, fem_frac, int_frac, unit_score = _unit_gate(allu)
    units = np.flatnonzero(unit_ok)

    W = WINDOW_BINS
    n_trials = robs.shape[0]
    if robs.shape[1] < W:
        raise RuntimeError(f"{session}: only {robs.shape[1]} time bins, need {W}")
    n_rate_bins = W // RATE_BIN_FACTOR
    # (n_trials, n_rate_bins, C) spike counts in the 25 ms counting window.
    R = robs[:, :W, :].reshape(n_trials, n_rate_bins, RATE_BIN_FACTOR, -1).sum(2)
    R = R[..., units]

    full_valid = valid_mask[:, :W].all(axis=1)
    peaks = [_saccade_peaks(eyepos[t, :W]) if full_valid[t] else None
             for t in range(n_trials)]
    sacc_ok = np.array([
        full_valid[t] and (SACC_COUNT_LO <= len(peaks[t]) <= SACC_COUNT_HI)
        for t in range(n_trials)
    ])
    eligible = np.flatnonzero(sacc_ok)

    # Window centers usable for the shaded Delta-e windows: whole span in fixation.
    win_ok = [None] * n_trials
    for t in range(n_trials):
        if not full_valid[t]:
            continue
        fm = _fixation_mask(eyepos[t, :W])
        ok = fm.copy()
        for s in range(1, WIN_HALF + 1):
            ok[s:] &= fm[:-s]
            ok[:-s] &= fm[s:]
        ok[:WIN_HALF] = False
        ok[W - WIN_HALF:] = False
        win_ok[t] = ok

    print(f"  {session}: trials={n_trials} eligible={len(eligible)} "
          f"units={len(units)}/{len(neuron_mask)} pass the panel-B gate", end="")

    min_split = MIN_EPOCH_RATE_BINS * RATE_BIN_FACTOR + SPLIT_PAD_BINS
    max_split = W - min_split
    candidates = []
    n_geom = 0

    for axis in (0, 1):
        eye_ax = eyepos[..., axis]
        for ai in range(len(eligible)):
            a = int(eligible[ai])
            for bi in range(ai + 1, len(eligible)):
                b = int(eligible[bi])
                if _coincident_count(peaks[a], peaks[b]):
                    continue
                d = np.abs(eye_ax[a, :W] - eye_ax[b, :W])

                # The split must sit on a saccade in exactly one trial -- the
                # one that brings the trajectories together.
                splits = [p for p in (peaks[a] or []) + (peaks[b] or [])
                          if min_split <= p <= max_split]
                best = None
                for t_split in splits:
                    d_pre = float(d[:t_split - SPLIT_PAD_BINS].mean())
                    d_post = float(d[t_split + SPLIT_PAD_BINS:].mean())
                    if d_pre < MIN_D_PRE or d_post > MAX_D_POST \
                            or (d_pre - d_post) < MIN_D_DROP:
                        continue
                    if best is None or (d_pre - d_post) > best[1] - best[2]:
                        best = (t_split, d_pre, d_post)
                if best is None:
                    continue
                t_split, d_pre, d_post = best
                n_geom += 1

                pre_bins, post_bins = _epoch_rate_bins(t_split, n_rate_bins)
                if len(pre_bins) < MIN_EPOCH_RATE_BINS \
                        or len(post_bins) < MIN_EPOCH_RATE_BINS:
                    continue

                w_div, w_mat = _pick_windows(
                    d, t_split, win_ok[a] & win_ok[b])
                if w_div is None:
                    continue

                xa_pre, xb_pre = R[a][pre_bins], R[b][pre_bins]
                xa_post, xb_post = R[a][post_bins], R[b][post_bins]

                # Spike gates, per trial per epoch, vectorized over units.
                enough = (
                    (xa_pre.sum(0) >= MIN_SPIKES_PER_EPOCH)
                    & (xb_pre.sum(0) >= MIN_SPIKES_PER_EPOCH)
                    & (xa_post.sum(0) >= MIN_SPIKES_PER_EPOCH)
                    & (xb_post.sum(0) >= MIN_SPIKES_PER_EPOCH)
                    & ((xa_pre > 0).sum(0) >= MIN_NONZERO_BINS)
                    & ((xb_pre > 0).sum(0) >= MIN_NONZERO_BINS)
                    & ((xa_post > 0).sum(0) >= MIN_NONZERO_BINS)
                    & ((xb_post > 0).sum(0) >= MIN_NONZERO_BINS)
                )
                if not enough.any():
                    continue

                r_pre = _pearson_cols(xa_pre, xb_pre)
                r_post = _pearson_cols(xa_post, xb_post)
                contrast = r_post - r_pre
                keep = (enough & np.isfinite(contrast)
                        & (r_post >= MIN_R_POST))
                for k in np.flatnonzero(keep):
                    j = int(units[k])
                    candidates.append(dict(
                        session=session,
                        unit=int(neuron_mask[j]), j=j, axis=axis, a=a, b=b,
                        t_split=int(t_split), d_pre=d_pre, d_post=d_post,
                        r_pre=float(r_pre[k]), r_post=float(r_post[k]),
                        contrast=float(contrast[k]),
                        w_div=w_div, w_mat=w_mat,
                        n_pre=len(pre_bins), n_post=len(post_bins),
                        fem_frac=float(fem_frac[j]),
                        int_frac=float(int_frac[j]),
                        rate=float(allu["rate_hz"][j]),
                        unit_score=float(unit_score[j]),
                    ))

    print(f"  ->  {n_geom} geometries, {len(candidates)} candidates")
    return candidates, allu


def search(sessions):
    """Search every session and pool the candidates under one global ranking."""
    all_candidates = []
    allu_by_session = {}
    print(f"Searching {len(sessions)} session(s):")
    for session in sessions:
        try:
            cands, allu = search_session(session)
        except FileNotFoundError:
            print(f"  {session}: no scan cache -- run build_lead_scan_caches.py")
            continue
        except Exception as exc:                        # noqa: BLE001
            print(f"  {session}: skipped ({type(exc).__name__}: {exc})")
            continue
        allu_by_session[session] = allu
        all_candidates.extend(cands)

    print(f"\n{len(all_candidates)} candidates pooled across "
          f"{len(allu_by_session)} session(s).")
    if not all_candidates:
        return all_candidates, allu_by_session

    # Composite: half the panel-A contrast, half the unit's panel-B quality --
    # ranking on contrast alone surfaces cells whose decomposition curve is
    # nearly flat, which makes for a poor panel B. The contrast rank is computed
    # over the pooled set so sessions compete on the same scale; unit_score is
    # already a within-session percentile from survey_lead_cells.
    contrasts = np.array([c["contrast"] for c in all_candidates])
    contrast_rank = np.argsort(np.argsort(contrasts)) / max(len(contrasts) - 1, 1)
    for c, cr in zip(all_candidates, contrast_rank):
        c["contrast_rank"] = float(cr)
        c["composite"] = 0.5 * float(cr) + 0.5 * c["unit_score"]
    return all_candidates, allu_by_session


def _report_committed():
    """Where the currently committed example lands on the new metric."""
    pair = _load_trial_pair(SESSION)
    robs = np.nan_to_num(pair["robs"], nan=0.0)
    eyepos = pair["eyepos"]
    neuron_mask = np.asarray(pair["neuron_mask"])
    j = int(np.flatnonzero(neuron_mask == EXAMPLE_UNIT)[0])
    W = WINDOW_BINS
    n_rate_bins = W // RATE_BIN_FACTOR

    d = np.abs(eyepos[TRIAL_A, :W, EYE_AXIS] - eyepos[TRIAL_B, :W, EYE_AXIS])
    # Split midway between the committed divergent and matched windows.
    t_split = (W1[1] + W2[0]) // 2
    pre_bins, post_bins = _epoch_rate_bins(t_split, n_rate_bins)
    R = robs[:, :W, j].reshape(-1, n_rate_bins, RATE_BIN_FACTOR).sum(2)
    r_pre = float(_pearson_cols(R[TRIAL_A][pre_bins, None],
                                R[TRIAL_B][pre_bins, None])[0])
    r_post = float(_pearson_cols(R[TRIAL_A][post_bins, None],
                                 R[TRIAL_B][post_bins, None])[0])
    print(f"\nCommitted example ({SESSION}, unit {EXAMPLE_UNIT}, trials "
          f"{TRIAL_A}/{TRIAL_B}, axis={'h' if EYE_AXIS==0 else 'v'}, "
          f"split={t_split}):")
    print(f"  d_pre={d[:t_split].mean():.3f}  d_post={d[t_split:].mean():.3f}  "
          f"r_pre={r_pre:+.3f}  r_post={r_post:+.3f}  "
          f"contrast={r_post - r_pre:+.3f}")
    return dict(r_pre=r_pre, r_post=r_post, contrast=r_post - r_pre)


def _render_page(pdf, c, allu_by_session, rank):
    """Preview one candidate through the real panel A / panel B renderers."""
    allu = allu_by_session[c["session"]]
    j = c["j"]
    decomp = {
        "bin_centers": allu["bin_centers"],
        "cum_crate": allu["cum_crate"][:, j],
        "Ctotal": float(allu["Ctotal"][j]),
        "Crate": float(allu["Crate"][j]),
        "Cpsth": float(allu["Cpsth"][j]),
        "sigma_int": float(allu["sigma_int"][j]),
    }

    fig = plt.figure(figsize=(7.0, 3.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.34,
                             left=0.09, right=0.975, top=0.78, bottom=0.15)
    gs_a = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 0],
                                   height_ratios=[1.0, 1.0], hspace=0.18)
    ax_eye = fig.add_subplot(gs_a[0, 0])
    ax_spk = fig.add_subplot(gs_a[1, 0], sharex=ax_eye)
    plot_eye_rate_example(ax_eye, ax_spk, unit_orig=c["unit"],
                          trial_a=c["a"], trial_b=c["b"], eye_axis=c["axis"],
                          w1=c["w_div"], w2=c["w_mat"], session=c["session"])
    # Mark the aligning saccade and the two epochs the correlations came from.
    t_split_ms = c["t_split"] * DT * 1000.0
    for ax in (ax_eye, ax_spk):
        ax.axvline(t_split_ms, color="tab:green", lw=1.0, ls="--", zorder=5)
    ax_eye.text(t_split_ms, 1.01, "align", transform=ax_eye.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7, color="tab:green")
    trans = ax_spk.get_xaxis_transform()
    ax_spk.text(0.5 * t_split_ms, -0.30, f"$r_{{pre}}$={c['r_pre']:+.2f}",
                transform=trans, ha="center", va="top", fontsize=7.5,
                color="crimson")
    ax_spk.text(0.5 * (t_split_ms + WINDOW_BINS * DT * 1000.0), -0.30,
                f"$r_{{post}}$={c['r_post']:+.2f}", transform=trans,
                ha="center", va="top", fontsize=7.5, color="tab:blue")

    ax_b = fig.add_subplot(outer[0, 1])
    plot_unaccounted_variance_panel(ax_b, decomp=decomp, caption=False)
    ax_b.xaxis.label.set_size(8)
    ax_b.yaxis.label.set_size(8)
    ax_b.tick_params(labelsize=7)

    fig.suptitle(
        f"rank {rank}   {c['session']}   unit {c['unit']}   "
        f"trials ({c['a']},{c['b']})   "
        f"axis={'h' if c['axis']==0 else 'v'}   split bin {c['t_split']}\n"
        f"contrast={c['contrast']:+.3f}  (r_pre={c['r_pre']:+.3f}, "
        f"r_post={c['r_post']:+.3f}; n={c['n_pre']}/{c['n_post']} bins)   "
        f"Δe {c['d_pre']:.2f}°→{c['d_post']:.2f}°\n"
        f"rate={c['rate']:.1f} Hz   Cfem/Ctot={c['fem_frac']:.2f}   "
        f"Σint/Ctot={c['int_frac']:.2f}   composite={c['composite']:.2f}",
        fontsize=7.5, y=0.995,
    )
    pdf.savefig(fig, bbox_inches="tight", dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description="Joint unit x trial-pair x alignment search for the fig2 lead-in.")
    p.add_argument("--top", type=int, default=24,
                   help="Number of candidates to render (default 24).")
    p.add_argument("--max-per-unit", type=int, default=2,
                   help="Cap pages per unit so the PDF spans distinct cells.")
    p.add_argument("--rank", choices=("composite", "contrast"),
                   default="composite",
                   help="composite = half panel-A contrast, half panel-B unit "
                        "quality (default); contrast = panel-A contrast only.")
    p.add_argument("--sessions", nargs="*", default=None,
                   help="Sessions to search (default: every fig2 session that "
                        "has a scan cache).")
    p.add_argument("--out", default="lead_example_candidates.pdf",
                   help="PDF filename under the fig2 figures dir.")
    args, _ = p.parse_known_args()

    if args.sessions:
        sessions = args.sessions
    else:
        sessions = [s for s in fig2_sessions() if scan_cache_path(s).exists()]
        if not sessions:
            sessions = [SESSION]

    candidates, allu_by_session = search(sessions)
    candidates.sort(key=lambda c: -c[args.rank])
    _report_committed()
    if not candidates:
        print("\nNo candidates passed. Loosen MAX_D_POST / MIN_D_DROP / MIN_R_POST.")
        return

    seen = {}
    shortlist = []
    for c in candidates:
        key = (c["session"], c["unit"])
        n = seen.get(key, 0)
        if n >= args.max_per_unit:
            continue
        seen[key] = n + 1
        shortlist.append(c)
        if len(shortlist) >= args.top:
            break

    print(f"\nTop {len(shortlist)} by {args.rank} (max {args.max_per_unit} per unit):")
    print(f"{'session':>17} {'unit':>5} {'a':>4} {'b':>4} {'ax':>3} {'split':>6} "
          f"{'r_pre':>7} {'r_post':>7} {'contr':>7} {'d_pre':>6} {'d_post':>7} "
          f"{'rate':>6} {'fem':>5} {'int':>5} {'comp':>5}")
    for c in shortlist:
        print(f"{c['session']:>17} {c['unit']:>5} {c['a']:>4} {c['b']:>4} "
              f"{'h' if c['axis']==0 else 'v':>3} {c['t_split']:>6} "
              f"{c['r_pre']:>+7.3f} {c['r_post']:>+7.3f} {c['contrast']:>+7.3f} "
              f"{c['d_pre']:>6.2f} {c['d_post']:>7.2f} {c['rate']:>6.1f} "
              f"{c['fem_frac']:>5.2f} {c['int_frac']:>5.2f} "
              f"{c['composite']:>5.2f}")

    n_sess = len({c["session"] for c in shortlist})
    print(f"\nShortlist spans {n_sess} session(s).")

    out = FIG_DIR / args.out
    with PdfPages(out) as pdf, plt.rc_context(_FIG2_RC):
        for rank, c in enumerate(shortlist, 1):
            _render_page(pdf, c, allu_by_session, rank)
    print(f"Saved {out} ({len(shortlist)} pages, best -> worst).")


if __name__ == "__main__":
    main()
