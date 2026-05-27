# %% Visualize top candidate trial pairs for fig2 lead panel.
"""
Plot the top candidate (trial-a, trial-b) pairs as 2-row mini-panels:
    top    = eye position (h or v) for the 2 trials, vertical markers at
             the proposed close-bin (blue) and far-bin (red).
    bottom = spike rate over time for the 2 trials, plotted as separate
             line traces offset vertically on shared axes, with a scale bar.

Run after pick_lead_trial_pair.py (uses the same cache).
"""
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from VisionCore.paths import CACHE_DIR, FIGURES_DIR

TARGET_SESSION = "Allen_2022-03-04"
TARGET_UNIT_ORIG = 151
WINDOW_BINS = 90
DT = 1.0 / 120.0

OUT = FIGURES_DIR / "fig2" / "lead_panel_candidates.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Hand-picked from picker output above.
# (axis, trial_a, trial_b, t_close, t_far)
CANDIDATES = [
    ("h", 49, 68, 42, 2),    # sacc=(2,2)  — top pick
    ("h", 54, 68, 81, 8),    # sacc=(1,2)
    ("h", 9, 68, 77, 4),     # sacc=(1,2)
    ("h", 9, 59, 71, 4),     # sacc=(1,1)
    ("h", 9, 35, 57, 4),     # sacc=(1,1)
    ("h", 9, 34, 70, 4),     # sacc=(1,2)
    ("h", 9, 82, 76, 4),     # sacc=(1,2)
    ("h", 49, 59, 36, 2),    # sacc=(2,1)
    ("h", 9, 13, 37, 4),     # sacc=(1,1)
    ("h", 9, 14, 89, 4),     # sacc=(1,2)
    ("v", 41, 68, 76, 35),   # sacc=(1,2)
    ("v", 43, 59, 6, 80),    # sacc=(2,1)
    ("v", 34, 41, 44, 24),   # sacc=(2,1)
    ("v", 21, 41, 45, 11),   # sacc=(2,1)
    ("v", 14, 59, 12, 86),   # sacc=(2,1)
    ("v", 13, 59, 1, 81),    # sacc=(1,1)
    ("v", 15, 41, 60, 28),   # sacc=(1,1)
    ("v", 59, 68, 12, 75),   # sacc=(1,2)
]


def main():
    with open(CACHE_DIR / f"fig2_lead_pair_scan_{TARGET_SESSION}.pkl", "rb") as f:
        pkt = pickle.load(f)
    robs = pkt["robs"]
    eyepos = pkt["eyepos"]
    valid_mask = pkt["valid_mask"]
    neuron_mask = pkt["neuron_mask"]
    j = int(np.where(np.asarray(neuron_mask) == TARGET_UNIT_ORIG)[0][0])

    W = WINDOW_BINS
    t_ms = np.arange(W) * DT * 1000.0

    with PdfPages(OUT) as pdf:
        for axis, a, b, t_c, t_f in CANDIDATES:
            ax_idx = 0 if axis == "h" else 1
            e_a = eyepos[a, :W, ax_idx]
            e_b = eyepos[b, :W, ax_idx]
            r_a = robs[a, :W, j] / DT
            r_b = robs[b, :W, j] / DT
            va = valid_mask[a, :W].all()
            vb = valid_mask[b, :W].all()

            fig, axs = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1]})
            fig.suptitle(
                f"axis={axis}  trials=({a},{b})  fully_valid=({va},{vb})  "
                f"t_close={t_c} (Δ={abs(e_a[t_c]-e_b[t_c]):.3f}°)  "
                f"t_far={t_f} (Δ={abs(e_a[t_f]-e_b[t_f]):.3f}°)",
                fontsize=10,
            )

            # Top: eye traces
            ax = axs[0]
            ax.plot(t_ms, e_a, color="tab:blue", lw=1.5, label=f"trial {a}")
            ax.plot(t_ms, e_b, color="tab:orange", lw=1.5, label=f"trial {b}")
            ax.axvline(t_ms[t_c], color="tab:blue", ls=":", alpha=0.6)
            ax.axvline(t_ms[t_f], color="tab:red", ls=":", alpha=0.6)
            ax.set_ylabel(f"eye {axis} (°)")
            ax.legend(loc="upper right", fontsize=8, frameon=False)

            # Bottom: rate traces, vertically offset on shared axes
            ax = axs[1]
            ymax = float(max(r_a.max(), r_b.max(), 1e-6))
            offset = 1.2 * ymax
            ax.step(t_ms, r_b + offset, color="tab:orange", lw=1.0,
                    where="mid", label=f"trial {b}")
            ax.step(t_ms, r_a, color="tab:blue", lw=1.0,
                    where="mid", label=f"trial {a}")
            # baselines
            ax.axhline(0, color="0.7", lw=0.6)
            ax.axhline(offset, color="0.7", lw=0.6)
            ax.axvline(t_ms[t_c], color="tab:blue", ls=":", alpha=0.6)
            ax.axvline(t_ms[t_f], color="tab:red", ls=":", alpha=0.6)

            # Scale bar (vertical, spk/s) at right edge
            sb_len = round(ymax / 2 / 10) * 10  # nearest 10 spk/s
            sb_len = max(sb_len, 10)
            sb_x = t_ms[-1] + 12
            ax.plot([sb_x, sb_x], [0, sb_len], color="k", lw=2,
                    clip_on=False)
            ax.text(sb_x + 4, sb_len / 2, f"{sb_len} spk/s",
                    rotation=90, va="center", ha="left", fontsize=8,
                    clip_on=False)
            ax.set_xlabel("time from fixation onset (ms)")
            ax.set_yticks([])
            ax.set_xlim(0, t_ms[-1] + 4)
            ax.spines["left"].set_visible(False)

            pdf.savefig(fig, bbox_inches="tight", dpi=120)
            plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
