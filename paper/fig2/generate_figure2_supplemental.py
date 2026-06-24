"""
Figure 2 supplemental outputs:
    - Per-session covariance heatmaps (Total / FEM / PSTH / Noise-corrected)
    - Stats report text file (alpha, fano, noise correlation, subspace)

Both rely on the cached fig2 derived bundle (compute_fig2_data.load_fig2_data).
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

from VisionCore.covariance import project_to_psd
from _panel_common import FIG_DIR, STAT_DIR
from compute_fig2_data import load_fig2_data


HEATMAP_WINDOW_IDX = 3  # 80 ms


def write_covariance_heatmaps(data=None, window_idx=HEATMAP_WINDOW_IDX):
    if data is None:
        data = load_fig2_data()
    cmap = plt.get_cmap("RdBu")
    for ds_idx, sr in enumerate(data["session_results"]):
        if window_idx >= len(sr["mats"]):
            continue
        mats = sr["mats"][window_idx]
        Crate_raw = mats["Intercept"]

        hm_valid = (np.isfinite(np.diag(Crate_raw))
                    & np.isfinite(np.diag(mats["PSTH"])))
        ix = np.ix_(hm_valid, hm_valid)

        Ctotal = project_to_psd(mats["Total"][ix])
        Cpsth = project_to_psd(mats["PSTH"][ix])
        Cfem = project_to_psd(Crate_raw[ix] - mats["PSTH"][ix])
        CnoiseC = project_to_psd(mats["Total"][ix] - Crate_raw[ix])

        v = np.nanmax(np.abs(Ctotal)) * 0.5

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"{sr['session']} ({sr['subject']})", fontsize=14)
        for ax, mat, title, vscale in zip(
            axs,
            [Ctotal, Cfem, Cpsth, CnoiseC],
            ["Total", "FEM", "PSTH", "Noise (Corrected)"],
            [1.0, 1.0, 0.5, 1.0],
        ):
            ax.imshow(mat, cmap=cmap, interpolation="nearest",
                      vmin=-v * vscale, vmax=v * vscale)
            ax.set_title(title)
            ax.axis("off")
        out = FIG_DIR / f"figure2_cov_decomp_session{ds_idx}.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved {out}")


class _Tee:
    def __init__(self, file, stream):
        self.file = file
        self.stream = stream

    def write(self, data):
        self.file.write(data)
        self.stream.write(data)

    def flush(self):
        self.file.flush()
        self.stream.flush()


def write_stats_report(data=None):
    if data is None:
        data = load_fig2_data()
    WINDOWS_MS = data["WINDOWS_MS"]
    WINDOWS_BINS = data["WINDOWS_BINS"]
    SUBJECTS = data["SUBJECTS"]
    alpha_stats = data["alpha_stats"]
    fano_stats = data["fano_stats"]
    nc_stats = data["nc_stats"]
    subjects = data["subjects"]
    n_sessions = data["n_sessions"]
    pr_fem_list = data["pr_fem_list"]
    pr_psth_list = data["pr_psth_list"]
    var_p_given_f = data["var_p_given_f"]
    var_f_given_p = data["var_f_given_p"]

    stats_file = STAT_DIR / "figure2_stats.txt"
    with open(stats_file, "w") as f:
        old_stdout = sys.stdout
        sys.stdout = _Tee(f, old_stdout)
        try:
            print("=" * 80)
            print("FIGURE 2: LOTC COVARIANCE DECOMPOSITION STATISTICS")
            print("=" * 80)
            print(f"Sessions: {n_sessions} ({', '.join(sorted(set(subjects)))})")
            print(f"Windows (bins): {WINDOWS_BINS} -> (ms): "
                  f"{[f'{w:.1f}' for w in WINDOWS_MS]}")

            print("\n" + "=" * 80)
            print("ALPHA (FEM MODULATION FRACTION)")
            print("=" * 80)
            for w in WINDOWS_MS:
                s = alpha_stats[w]
                print(f"\nWindow {w:.1f} ms (N={s['n']}):")
                print(f"  1-alpha: mean={s['mean']:.3f} "
                      f"[{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]")
                print(f"  median={s['median']:.3f} "
                      f"IQR=[{s['iqr'][0]:.3f}, {s['iqr'][1]:.3f}]")
                print(f"  Shuffle null 95% CI: "
                      f"[{s['null_ci'][0]:.3f}, {s['null_ci'][1]:.3f}]")
                print(f"  Empirical p={s['p_emp']:.4f}")

            print("\n" + "=" * 80)
            print("FANO FACTOR STATISTICS")
            print("=" * 80)
            for w in WINDOWS_MS:
                s = fano_stats[w]
                print(f"\nWindow {w:.1f} ms (N={s['n']}):")
                print(f"  Uncorr: gmean={s['g_unc']:.3f}")
                print(f"  Corr:   gmean={s['g_cor']:.3f}")
                print(f"  Ratio={s['ratio']:.3f} "
                      f"({s['pct_red']:.1f}% reduction)")
                print(f"  Wilcoxon p={s['p_wil']:.3g}")
                print(f"  Pop FF: uncorr={s['slope_unc']:.3f}, "
                      f"corr={s['slope_cor']:.3f}")
                print(f"  Slope diff={s['slope_diff']:.3f} CI "
                      f"[{s['slope_diff_ci'][0]:.3f}, "
                      f"{s['slope_diff_ci'][1]:.3f}]")

            print("\n" + "=" * 80)
            print("NOISE CORRELATION STATISTICS")
            print("=" * 80)
            for w in WINDOWS_MS:
                s = nc_stats[w]
                print(f"\nWindow {w:.1f} ms ({s['n_pairs']} pairs, "
                      f"{s['n_ds']} datasets):")
                print(f"  z_uncorr = {s['z_u_mean']:.4f} "
                      f"[{s['z_u_ci'][0]:.4f}, {s['z_u_ci'][1]:.4f}]")
                print(f"  z_corr   = {s['z_c_mean']:.4f} "
                      f"[{s['z_c_ci'][0]:.4f}, {s['z_c_ci'][1]:.4f}]")
                print(f"  delta_z  = {s['dz_mean']:.4f} "
                      f"[{s['dz_ci'][0]:.4f}, {s['dz_ci'][1]:.4f}]")
                print(f"  Wilcoxon p={s['p_wil']:.3g}, "
                      f"empirical p={s['p_emp_dz']:.4f}")

            print("\n" + "=" * 80)
            print("SUBSPACE STATISTICS")
            print("=" * 80)
            print(f"  PR(FEM):  mean={np.mean(pr_fem_list):.3f}")
            print(f"  PR(PSTH): mean={np.mean(pr_psth_list):.3f}")
            print(f"  X: mean={np.mean(var_p_given_f):.3f}")
            print(f"  Y: mean={np.mean(var_f_given_p):.3f}")

            print("\n" + "=" * 80)
            print("END OF STATISTICS")
            print("=" * 80)
        finally:
            sys.stdout = old_stdout
    print(f"Stats saved to {stats_file}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate fig2 supplemental outputs.")
    p.add_argument("--no-heatmaps", action="store_true",
                   help="Skip per-session covariance heatmaps.")
    p.add_argument("--no-stats", action="store_true",
                   help="Skip stats report.")
    args, _ = p.parse_known_args()
    data = load_fig2_data()
    if not args.no_heatmaps:
        write_covariance_heatmaps(data=data)
    if not args.no_stats:
        write_stats_report(data=data)
