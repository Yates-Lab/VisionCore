import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def plot_digitaltwin_paper_figure(metrics, traces, title="Digital Twin Performance"):
    """
    Generates the publication-quality Digital Twin figure.
    
    Layout:
    [ A: Example Trace 1 ] [ B: Example Trace 2 ] 
    [ C: Var Expl Scatter] [ D: Ratio vs Mod  ] [ E: CC vs Reliability ]
    """
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2])

    # --- Panels A & B: Example Traces ---
    # We will pick the "best" examples (high FEM modulation, high improvement)
    # You can also pass specific indices if you want fixed examples
    
    # Find good examples: High FEM modulation (low alpha) AND High Improvement
    candidates = np.where(
        (metrics['fem_mod'] > 0.4) & 
        (metrics['improvement_ratio'] > 1.2) &
        (metrics['var_total'] > 0.05)
    )[0]
    
    # Sort by improvement ratio to get striking examples
    best_inds = candidates[np.argsort(metrics['improvement_ratio'][candidates])[::-1]]
    
    if len(best_inds) >= 2:
        ex_indices = [best_inds[0], best_inds[1]] # Top 2
    else:
        ex_indices = [0, 1] # Fallback

    ax_ex1 = fig.add_subplot(gs[0, :]) # Full width top for nice long trace? Or split?
    # Let's split top row
    ax_ex1 = fig.add_subplot(gs[0, :])
    _plot_example_trace(ax_ex1, traces, ex_indices[0], metrics)
    
    ax_ex2 = fig.add_subplot(gs[1, :])
    _plot_example_trace(ax_ex2, traces, ex_indices[1], metrics)

    # --- Panel C: Variance Explained (Model vs PSTH) ---
    ax_var = fig.add_subplot(gs[2, 0])
    
    # Filter for plot clarity
    valid = (metrics['var_total'] > 0.01) & (metrics['var_psth'] > 0.001)
    x = metrics['var_expl_psth'][valid]
    y = metrics['var_expl_model'][valid]
    c = metrics['fem_mod'][valid]
    
    sc = ax_var.scatter(x, y, c=c, cmap='viridis', s=10, alpha=0.7, vmin=0, vmax=0.8)
    
    # Identity line
    max_val = np.percentile(np.concatenate([x,y]), 99)
    ax_var.plot([0, max_val], [0, max_val], 'k--', lw=1)
    
    ax_var.set_xlim(0, max_val)
    ax_var.set_ylim(0, max_val)
    ax_var.set_xlabel("Var Explained by PSTH")
    ax_var.set_ylabel("Var Explained by Model")
    ax_var.set_title("Model Captures\nFEM Variance")
    
    # Inset colorbar
    cbar = plt.colorbar(sc, ax=ax_var, shrink=0.6)
    cbar.set_label("FEM Mod (1-$\\alpha$)")

    # --- Panel D: Improvement Ratio vs FEM Modulation ---
    ax_ratio = fig.add_subplot(gs[2, 1])
    
    x_mod = metrics['fem_mod'][valid]
    y_ratio = metrics['improvement_ratio'][valid]
    
    # Clip extreme outliers for visualization
    mask = (y_ratio < 5) & (x_mod > -0.2)
    
    sns.regplot(
        x=x_mod[mask], y=y_ratio[mask], 
        ax=ax_ratio, 
        scatter_kws={'s': 5, 'alpha': 0.2, 'color': 'gray'},
        line_kws={'color': 'red'}
    )
    ax_ratio.axhline(1.0, color='k', linestyle='--')
    ax_ratio.set_xlabel("FEM Modulation (1 - $\\alpha$)")
    ax_ratio.set_ylabel("Improvement (Model / PSTH)")
    ax_ratio.set_title("Improvement Scales\nwith Modulation")

    # --- Panel E: Performance vs Reliability ---
    ax_perf = fig.add_subplot(gs[2, 2])
    
    # Reliability proxy: Var_PSTH / Var_Total (Signal-to-Total-Variance)
    # A neuron driven purely by noise has 0 reliability here.
    # Note: This is an approximation. CCmax is better if available.
    reliability = metrics['var_psth'][valid] / metrics['var_total'][valid]
    perf = metrics['cc_norm'][valid]
    
    ax_perf.scatter(reliability, perf, s=10, c='k', alpha=0.4)
    ax_perf.set_xlabel("Signal Reliability")
    ax_perf.set_ylabel("Model $CC_{norm}$")
    ax_perf.set_title("Model Fits Reliable Units")
    
    plt.tight_layout()
    return fig

def _plot_example_trace(ax, traces, global_idx, metrics):
    """
    Helper to plot a raster-like PSTH overlay for a single neuron.
    """
    # Unpack location
    ds_idx = metrics['ds_idx'][global_idx]
    n_idx  = metrics['n_idx'][global_idx]
    
    # Get data
    robs = traces[ds_idx]['robs'][:, :, n_idx] # (Trials, Time)
    rhat = traces[ds_idx]['rhat'][:, :, n_idx]
    
    # Calculate Mean +/- IQR/SE
    # We want to show the trial-to-trial variability envelope
    
    # Mean
    mu_obs = np.nanmean(robs, axis=0)
    mu_hat = np.nanmean(rhat, axis=0)
    
    # Variability (IQR)
    # We use IQR because spikes are poisson-like and noisy
    lo_obs = np.nanpercentile(robs, 25, axis=0)
    hi_obs = np.nanpercentile(robs, 75, axis=0)
    
    lo_hat = np.nanpercentile(rhat, 25, axis=0)
    hi_hat = np.nanpercentile(rhat, 75, axis=0)
    
    t = np.arange(len(mu_obs)) * (1000/120) # ms
    
    # Plot Data (Black/Gray)
    ax.plot(t, mu_obs, 'k-', lw=2, label='Data Mean')
    ax.fill_between(t, lo_obs, hi_obs, color='k', alpha=0.15, label='Data IQR')
    
    # Plot Model (Red/Pink)
    ax.plot(t, mu_hat, 'r--', lw=1.5, label='Model Mean')
    # ax.fill_between(t, lo_hat, hi_hat, color='r', alpha=0.15, label='Model IQR')
    
    # Plot specific interesting trials? 
    # Optional: Plot 1-2 single trials if they track well
    # ax.plot(t, robs[0], 'k-', lw=0.5, alpha=0.3)
    # ax.plot(t, rhat[0], 'r-', lw=0.5, alpha=0.3)

    # Styling
    ax.set_title(f"Neuron {ds_idx}-{n_idx} | FEM Mod: {metrics['fem_mod'][global_idx]:.2f} | Improvement: {metrics['improvement_ratio'][global_idx]:.1f}x")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Firing Rate")
    ax.legend(loc='upper right', frameon=False, fontsize='small')
    sns.despine(ax=ax)