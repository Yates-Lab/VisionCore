r"""Panel-D control: is the model-vs-empirical 1-alpha mismatch real or an
artifact of comparing two different estimators?

Panel D compares a clean ANOVA on the model's deterministic rates against the
full empirical pipeline (run_covariance_decomposition) on noisy observed spikes.
This script removes that estimator confound: it Poisson-samples the model rates
and pushes the simulated spikes through the *identical* pipeline used on real
data, with the same window (1 bin), the same intercept mode (below_threshold,
0.05 deg), and the cached eye trajectories.

STATUS (2026-05-28) -- PRELIMINARY, NOT YET FAITHFUL. Deferred to a future
session for an independent audit. Running this fig3-aligned version does NOT
reproduce fig2 (obs_pipeline median ~0.476 vs fig2 ~0.725 on session 0), so it
is confounded: the fig3 alignment (min_total_spikes=200, fixation<1deg, its own
trial/segment structure, dfs==0 -> NaN->0 zero-fill) differs from fig2's
align_fixrsvp_trials. Before this control can answer artifact-vs-real, obs_pipeline
must first match fig2. The FAITHFUL version should prepare simulated model spikes
in fig2's exact align_fixrsvp_trials frame.

LEADING HYPOTHESIS for why the pipeline and the clean ANOVA disagree (Ryan's
insight): the below_threshold intercept estimates Crate from PAIRS of trials with
eye-trajectory distance < 0.05 deg. During fixation eye position is ~Gaussian, so
*close pairs* are over-represented at CENTRAL eye positions (the squared-density
integral is dominated by the high-density center). The pipeline therefore
integrates rate variance over a NARROWER, more-central eye-position distribution
than the ANOVA, which uses ALL (trial, phase) samples and so sees the full
fixational distribution (including peripheral drifts that shift the stimulus
further on the retina). If FEM-driven variance grows with eccentricity, the two
estimators measure FEM over different distributions and are not directly
comparable -- which could dissolve the "model is more sensitive than the neurons"
reading of panel D. Test next session: compute the model 1-alpha on the SAME
close-pair distribution the pipeline uses (or measure both over a matched
eye-position distribution) so model and data are compared on equal footing.

Localized evidence so far (session 0, simulated Poisson spikes): pipeline
components correlate with the model's analytic components (PSTH r=0.74, Crate
r=0.91) but are scale-inflated (PSTH ~2.7x, Crate ~1.3x), and the differential
inflation of the PSTH numerator deflates the pipeline's 1-alpha.

Four per-cell 1-alpha estimates are compared (good cells, ccmax > threshold):

  model_analytic : rate_variance_components(rhat)         -- clean, no Poisson.
  sim_pipeline   : pipeline on Poisson(rhat)              -- known input + analysis.
  obs_pipeline   : pipeline on observed robs, SAME align  -- data + analysis.
  fig2           : 1 - sr['alpha']                        -- data + analysis (fig2 align).

Interpretation:
  * sim_pipeline ~ obs_pipeline, both < model_analytic  => the pipeline (e.g. the
    0.05 deg intercept) deflates 1-alpha; the panel-D gap is an analysis artifact.
  * sim_pipeline ~ model_analytic but still > obs        => real model property:
    the model's rates carry more FEM fraction than the neurons'.
  * obs_pipeline ~ fig2 is an alignment sanity check.

Usage:
    uv run declan/fig3/simulate_fig3d_control.py [--n-sim 5]
"""
import argparse
import numpy as np
import dill
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

from VisionCore.covariance import (
    run_covariance_decomposition, rate_variance_components,
)
from _fig3_data import (
    DT, FIG_DIR, CACHE_DIR, SUBJECTS, SUBJECT_COLORS, CCMAX_THRESHOLD,
    configure_matplotlib, load_fig3_data,
)

RESULT_CACHE = CACHE_DIR / "fig4d_simulation_control.pkl"
THRESHOLD = 0.05            # below_threshold intercept (deg), matches fig2
WINDOW_BINS = [1]          # 1-bin counting window, matches empirical mats[0]
MIN_TRIALS_PER_PHASE = 10


def _alpha_from_mats(mats0):
    """1-alpha per cell from a decomposition mats dict (window 0)."""
    psth = np.diag(mats0["PSTH"]).astype(float)
    rate = np.diag(mats0["Intercept"]).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.clip(psth / rate, 0.0, 1.0)
    return 1.0 - alpha


def _decompose(spikes, eyepos, valid_mask, seed=42):
    """Run the empirical pipeline once; return per-cell 1-alpha for window 0."""
    _, mats = run_covariance_decomposition(
        spikes, eyepos, valid_mask,
        window_sizes_bins=WINDOW_BINS, dt=DT, n_shuffles=0,
        intercept_mode="below_threshold",
        intercept_kwargs={"threshold": THRESHOLD},
        seed=seed, device="cuda",
    )
    if not mats:
        return None
    return _alpha_from_mats(mats[0])


def compute_control(data, n_sim=5, seed=0):
    """Per-cell 1-alpha under the four estimators, across all sessions."""
    rows = {k: [] for k in ("subj", "ccmax", "model_analytic",
                            "sim_pipeline", "obs_pipeline", "fig2")}
    for si, sr in enumerate(data["session_results"]):
        rhat = sr["rhat_used"]              # (trials, time, neurons), rescaled rate
        robs = sr["robs_used"]
        eyepos = sr["eyepos_used"]
        valid = sr["valid_mask"]
        n_neurons = rhat.shape[2]
        print(f"[{si+1}/{len(data['session_results'])}] {sr['session']} "
              f"({sr['subject']}): {rhat.shape[0]} trials, {n_neurons} neurons")

        # clean analytic model 1-alpha (per neuron), valid = eye-finite bins
        model_an = np.array([
            rate_variance_components(
                rhat[:, :, ni], valid=valid,
                min_trials_per_phase=MIN_TRIALS_PER_PHASE,
            )["one_minus_alpha"]
            for ni in range(n_neurons)
        ])

        # observed spikes through the identical pipeline (same alignment)
        obs_pipe = _decompose(robs, eyepos, valid, seed=42)

        # Poisson draws from the model rate through the identical pipeline
        lam = np.clip(np.nan_to_num(rhat, nan=0.0), 0.0, None)
        sim_draws = []
        for r in range(n_sim):
            rng = np.random.default_rng(seed + 1000 * si + r)
            spk = rng.poisson(lam).astype(np.float32)
            out = _decompose(spk, eyepos, valid, seed=42)
            if out is not None:
                sim_draws.append(out)
        sim_pipe = (np.nanmean(np.stack(sim_draws), axis=0)
                    if sim_draws else np.full(n_neurons, np.nan))

        fig2 = 1.0 - np.asarray(sr["alpha"], dtype=float)

        rows["subj"].extend([sr["subject"]] * n_neurons)
        rows["ccmax"].extend(np.asarray(sr["ccmax"], dtype=float))
        rows["model_analytic"].extend(model_an)
        rows["sim_pipeline"].extend(sim_pipe if obs_pipe is not None else [np.nan] * n_neurons)
        rows["obs_pipeline"].extend(obs_pipe if obs_pipe is not None
                                    else [np.nan] * n_neurons)
        rows["fig2"].extend(fig2)

    out = {k: (np.asarray(v) if k != "subj" else np.asarray(v, dtype=object).astype(str))
           for k, v in rows.items()}
    return out


def _pair_stats(name, x, y, mask):
    ok = mask & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        print(f"  {name}: N={ok.sum()} (too few)")
        return
    a, b = x[ok], y[ok]
    rho = spearmanr(a, b).correlation
    r = pearsonr(a, b)[0]
    print(f"  {name}: N={ok.sum()}, Spearman ρ={rho:.3f}, Pearson r={r:.3f}, "
          f"median(y-x)={np.median(b - a):+.3f}, "
          f"median x={np.median(a):.3f}, median y={np.median(b):.3f}")


def report(res):
    good = res["ccmax"] > CCMAX_THRESHOLD
    print(f"\n=== Panel-D simulation control: {good.sum()} good cells ===")
    for subj in SUBJECTS + ["All"]:
        m = good if subj == "All" else (good & (res["subj"] == subj))
        print(f"\n[{subj}] N_good={m.sum()}")
        _pair_stats("sim_pipeline   vs model_analytic", res["model_analytic"],
                    res["sim_pipeline"], m)
        _pair_stats("sim_pipeline   vs obs_pipeline   ", res["obs_pipeline"],
                    res["sim_pipeline"], m)
        _pair_stats("obs_pipeline   vs fig2           ", res["fig2"],
                    res["obs_pipeline"], m)
        _pair_stats("model_analytic vs fig2  (panel D)", res["fig2"],
                    res["model_analytic"], m)


def make_figure(res):
    good = res["ccmax"] > CCMAX_THRESHOLD
    pairs = [
        ("model_analytic", "sim_pipeline", "model analytic", "sim pipeline"),
        ("obs_pipeline", "sim_pipeline", "obs pipeline", "sim pipeline"),
        ("fig2", "obs_pipeline", "fig2 (empirical)", "obs pipeline"),
        ("fig2", "model_analytic", "fig2 (empirical)", "model analytic"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.5))
    for ax, (xk, yk, xl, yl) in zip(axes.ravel(), pairs):
        ax.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.5)
        for subj in SUBJECTS:
            m = good & (res["subj"] == subj) & np.isfinite(res[xk]) & np.isfinite(res[yk])
            ax.scatter(res[xk][m], res[yk][m], s=5, alpha=0.5,
                       color=SUBJECT_COLORS[subj], label=subj)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel(rf"{xl}  $1-\alpha$")
        ax.set_ylabel(rf"{yl}  $1-\alpha$")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "panel_d_simulation_control.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"\nSaved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sim", type=int, default=5)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    configure_matplotlib()
    data = load_fig3_data()
    if "eyepos_used" not in data["session_results"][0]:
        raise RuntimeError(
            "Cache lacks eyepos_used. Regenerate via load_fig3_data(recompute=True)."
        )

    if RESULT_CACHE.exists() and not args.recompute:
        print(f"Loading control results from {RESULT_CACHE}")
        with open(RESULT_CACHE, "rb") as f:
            res = dill.load(f)
    else:
        res = compute_control(data, n_sim=args.n_sim)
        with open(RESULT_CACHE, "wb") as f:
            dill.dump(res, f)
        print(f"Cached control results to {RESULT_CACHE}")

    report(res)
    make_figure(res)


if __name__ == "__main__":
    main()
