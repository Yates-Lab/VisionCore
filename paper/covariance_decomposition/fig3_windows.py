"""The counting-window contract shared by Figure 2 and Figure 3.

Figure 3 deliberately uses TWO different counting windows, and this module is
the single place either is named. Restating them locally is what let them drift
apart silently once already.

FIG2_REPORTED_WINDOW_BINS (3 bins, 25 ms)
    The window Figure 2 reports. Any quantity a reader will compare against a
    Figure 2 number must be computed here -- notably Figure 3 panel E, whose
    neuron f_FEM axis is the same quantity Figure 2 panel C reports. Figure 2's
    variance-based panels need this window: at one bin the uncorrected Fano
    factor is 1.058, essentially Poisson, so there is almost no excess variance
    left to attribute to fixational eye movements.

FIG3_SINGLETRIAL_WINDOW_BINS (1 bin, 8.33 ms)
    The twin's native resolution, used by Figure 3 panel D. The twin predicts at
    120 Hz and panel D asks how much single-trial structure it captures, so
    scoring it at its own resolution is the natural comparison. Summing to 25 ms
    also lifts the trial-average baseline (+0.038 median) about twice as much as
    it lifts the twin (+0.019): the twin's advantage lives in fast structure
    that averaging washes out. On the reported population that contrast is not
    significant at either window (twin vs trial-average p = 0.073 at 8.33 ms,
    0.196 at 25 ms; on a fixed unit set, 0.030 vs 0.212), so the window is
    chosen on the native-resolution argument, not on a significance threshold.
    Panel D's headline stabilization contrast is robust either way
    (p < 1e-4 at both).

Panel C is unaffected: it is a per-bin variance-explained ratio with no counting
window at all.

Both windows must exist in `decompose.WINDOW_BINS_DEFAULT`, since Figure 3 reads
its denominators from that decomposition.
"""

FIG2_REPORTED_WINDOW_BINS = 3
FIG3_SINGLETRIAL_WINDOW_BINS = 1

DT = 1 / 120


def window_ms(window_bins):
    """Counting-window duration in milliseconds."""
    return window_bins * DT * 1000.0


def fig2_history_bins():
    """Figure 2's fixed matching history in bins, read from `decompose.py`.

    Derived rather than restated: this is the constant that silently drifted
    between Figure 2 and Figure 3 once already.
    """
    import sys
    from pathlib import Path
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    from decompose import T_HIST_MS_DEFAULT, DT as _DT
    return int(round(T_HIST_MS_DEFAULT / (_DT * 1000)))
