"""Summarize existing sub-degree saccades without rerunning the detector.

Run from the repository root:
    PYTHONPATH=. python scripts/analyze_small_saccade_durations.py

Only reads source data. Exports event IDs, quality flags, summaries, and plots.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/visioncore-saccade-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from VisionCore.paths import STATS_DIR


PROTOCOLS = ("backimage", "fixrsvp", "gaborium", "gratings")
LABELS = {"backimage": "Free viewing", "fixrsvp": "Fixation task"}
COLORS = {"backimage": "#2878a2", "fixrsvp": "#d87832"}


def match_trials(events: pd.DataFrame, dataset: Path) -> dict:
    """Match the entire fitted event to one contiguous trial (bin-edge bounds)."""
    payload = torch.load(dataset, weights_only=False, mmap=True, map_location="cpu")
    cov = payload["covariates"]
    time = cov["t_bins"].numpy().reshape(-1)
    trial = cov["trial_inds"].numpy().reshape(-1)
    delta = np.diff(time)
    assert len(time) == len(trial) and np.all(np.isfinite(time))
    assert np.all(delta > 0), f"Nonmonotonic time: {dataset}"
    dt = float(np.median(delta[np.diff(trial) == 0]))
    assert np.isfinite(dt) and dt > 0
    breaks = (np.diff(trial) != 0) | (delta > 1.5 * dt)
    edges = np.r_[0, np.flatnonzero(breaks) + 1, len(time)]
    start = time[edges[:-1]] - dt / 2
    stop = time[edges[1:] - 1] + dt / 2
    onset = events.start_time.to_numpy()
    offset = events.end_time.to_numpy()
    pos = np.searchsorted(start, onset, side="right") - 1
    safe = np.clip(pos, 0, len(start) - 1)
    inside = (pos >= 0) & (onset >= start[safe]) & (offset <= stop[safe])
    inside &= np.isfinite(onset) & np.isfinite(offset) & (offset > onset)
    all_ids = events.index[inside]
    prior = events.loc[all_ids, "protocol_matches"]
    events.loc[all_ids, "protocol_matches"] = prior.map(
        lambda value: f"{value};{dataset.stem}" if value else dataset.stem)
    overlap = events.index[inside & (events.protocol != "unassigned").to_numpy()]
    inside &= (events.protocol == "unassigned").to_numpy()
    ids = events.index[inside]
    events.loc[ids, "protocol"] = dataset.stem
    events.loc[ids, "trial_idx"] = trial[edges[:-1]][safe[inside]]
    events.loc[ids, "trial_start_s"] = start[safe[inside]]
    events.loc[ids, "trial_end_s"] = stop[safe[inside]]

    # Sensitivity flag only: the saved detector already screens native eye validity.
    valid = cov["dpi_valid"].numpy().reshape(-1) == 1
    invalid_prefix = np.r_[0, np.cumsum(~valid)]
    lo = np.searchsorted(time + dt / 2, onset[inside], side="right")
    hi = np.searchsorted(time - dt / 2, offset[inside], side="left")
    events.loc[ids, "dataset_eye_valid"] = invalid_prefix[hi] == invalid_prefix[lo]
    # A small number of exported trials overlap between different protocols.
    # Keep those events in the overall inventory, but do not assign a task.
    events.loc[overlap, "protocol"] = "ambiguous"
    events.loc[overlap, "trial_idx"] = -1
    events.loc[overlap, ["trial_start_s", "trial_end_s"]] = np.nan
    events.loc[overlap, "dataset_eye_valid"] = False
    return {"protocol": dataset.stem, "dataset_path": str(dataset),
            "n_trials": len(start), "bin_width_ms": dt * 1000,
            "trial_exposure_s": float(np.sum(stop - start))}


def summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n_events": 0, "n_sessions": 0}
    duration = frame.duration_ms.to_numpy()
    q = np.percentile(duration, [5, 25, 50, 75, 95])
    medians = frame.groupby("session").duration_ms.median()
    return {"n_events": len(frame), "n_sessions": frame.session.nunique(),
            "mean_ms": float(duration.mean()), "sd_ms": float(duration.std(ddof=1)),
            **{f"p{p}_ms": float(v) for p, v in zip([5, 25, 50, 75, 95], q)},
            "min_ms": float(duration.min()), "max_ms": float(duration.max()),
            "median_amplitude_deg": float(frame.amplitude_deg.median()),
            "median_of_session_medians_ms": float(medians.median()),
            "min_session_median_ms": float(medians.min()),
            "max_session_median_ms": float(medians.max())}


def make_plots(events: pd.DataFrame, per_session: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    bins = np.linspace(0, 100, 101)
    for protocol, label in LABELS.items():
        data = events[events.protocol == protocol]
        color = COLORS[protocol]
        axes[0].hist(data.duration_ms, bins=bins, density=True, histtype="step",
                     linewidth=2, color=color,
                     label=f"{label} (n={len(data):,})")
        axes[0].axvline(data.duration_ms.median(), color=color, ls=":", lw=1)
        amp_bin = pd.cut(data.amplitude_deg, np.linspace(0, 1, 6), right=False)
        grouped = data.groupby(amp_bin, observed=True).duration_ms
        quantiles = grouped.quantile([.25, .5, .75]).unstack()
        x = np.array([interval.mid for interval in quantiles.index])
        axes[1].plot(x, quantiles[.5], "o-", color=color, label=label)
        axes[1].fill_between(x, quantiles[.25], quantiles[.75], color=color, alpha=.15)
        sess = per_session[per_session.protocol == protocol]
        for subject, marker in [("Allen", "o"), ("Logan", "^")]:
            sub = sess[sess.subject == subject]
            axes[2].scatter(sub.native_sample_interval_ms, sub.p50_ms,
                            color=color, marker=marker, alpha=.75,
                            label=f"{label}: {subject}")
    axes[0].set(xlabel="Fitted duration (ms)", ylabel="Probability density",
                xlim=(0, 100), title="Saccades with amplitude <1°")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set(xlabel="Amplitude (deg)", ylabel="Duration (ms)",
                title="Median and middle 50%", xlim=(0, 1))
    axes[2].set(xlabel="Native sample interval (ms; first 20k samples)",
                ylabel="Session median duration (ms)", title="Session / sampling variation")
    axes[2].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "duration_summary.png", dpi=180)
    fig.savefig(out / "duration_summary.pdf")
    plt.close(fig)


def smoothing_check(out: Path) -> None:
    """Idealized input sanity check; this is not a deconvolution or new detector."""
    dt = .0018525
    time = np.arange(-.1, .1, dt)

    def asymmetric_gaussian(t, mu, amplitude, sigma1, sigma2, baseline):
        sigma = np.where(t < mu, sigma1, sigma2)
        return amplitude * np.exp(-.5 * ((t - mu) / sigma) ** 2) + baseline

    rows = []
    for duration in [0, .005, .01, .015, .02, .03]:
        position = ((time >= 0).astype(float) if duration == 0 else
                    (1 - np.cos(np.pi * np.clip(time / duration, 0, 1))) / 2)
        speed = gaussian_filter1d(np.abs(np.diff(savgol_filter(position, 11, 3))) / dt, 2)
        keep = (time[:-1] >= -.025) & (time[:-1] <= duration + .025)
        x, y = time[:-1][keep], speed[keep]
        mu = x[np.argmax(y)]
        sigma = np.sqrt(np.mean((x - mu) ** 2 * y) / y.sum())
        fit, _ = curve_fit(asymmetric_gaussian, x, y,
                           p0=[mu, y.max(), sigma, sigma, (y[0] + y[-1]) / 2],
                           maxfev=5000)
        width = np.sqrt(-2 * np.log(.1)) * (abs(fit[2]) + abs(fit[3]))
        rows.append({"input_movement_ms": duration * 1000,
                     "fitted_duration_ms": width * 1000, "sample_interval_ms": dt * 1000,
                     "input": "unit step" if duration == 0 else "unit half-cosine ramp",
                     "fit_padding_ms": 25})
    pd.DataFrame(rows).to_csv(out / "smoothing_sanity_check.csv", index=False)


def raw_examples(events: pd.DataFrame, root: Path, session: str, out: Path) -> None:
    """Show six typical-duration events, selected before inspecting raw traces."""
    import h5py

    data = events[events.session == session]
    examples = []
    for protocol in LABELS:
        candidates = data[(data.protocol == protocol) & data.dataset_eye_valid]
        for target in [.3, .6, .9]:
            near = candidates[(candidates.amplitude_deg - target).abs() < .07]
            if near.empty:
                continue
            examples.append(near.loc[(near.duration_ms - near.duration_ms.median()).abs().idxmin()].copy())
    if not examples:
        return
    print(f"Reading native eye trace for examples: {session}", flush=True)
    dpi = pd.read_csv(root / session / "dpi/ddpi.csv",
                      usecols=["t_ephys", "dpi_i", "dpi_j", "valid"])
    with h5py.File(root.parent / "mat" / f"{session}_struct.mat") as mat:
        ppd = float(mat["S/pixPerDeg"][()].squeeze())
        center = mat["S/centerPix"][()].ravel()[::-1]
    time = dpi.t_ephys.to_numpy()
    position = np.fliplr(dpi[["dpi_i", "dpi_j"]].to_numpy() - center) / ppd * [1, -1]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), layout="constrained")
    raw_rows = []
    for ax, event in zip(axes.flat, examples):
        start, end = int(event.start_idx), int(event.end_idx)
        assert time[start] <= event.mu <= time[end], "Native indices no longer align with saved fit"
        window = np.arange(max(0, start - 35), min(len(time), end + 35))
        relative_time = (time[window] - event.start_time) * 1000
        displacement = position[end] - position[start]
        event["current_amplitude_deg"] = np.linalg.norm(displacement)
        direction = displacement / event.current_amplitude_deg
        projected = (position[window] - position[start]) @ direction
        perpendicular = (position[window] - position[start]) @ np.array([-direction[1], direction[0]])
        ax.plot(relative_time, projected, "o-", ms=2, color="#2878a2", label="Raw, along movement")
        ax.plot(relative_time, perpendicular, lw=1, color="#888888", label="Raw, perpendicular")
        ax.axvspan(0, event.duration_ms, color="#d87832", alpha=.15, label="Saved fitted duration")
        ax.set(xlim=(-30, 65), xlabel="Time from fitted onset (ms)", ylabel="Displacement (deg)",
               title=f"{event.protocol}: saved {event.amplitude_deg:.2f}° / {event.duration_ms:.1f} ms\n"
                     f"t={event.start_time:.3f} s, trial {event.trial_idx}")
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[0, 0]:
            ax.legend(fontsize=7, frameon=False)
        for i, sample in enumerate(window):
            raw_rows.append({"source_event_index": event.source_event_index, "session": session,
                             "sample_idx": sample, "time_s": time[sample],
                             "time_from_fitted_onset_ms": relative_time[i],
                             "projected_deg": projected[i], "perpendicular_deg": perpendicular[i],
                             "valid": dpi.valid.iloc[sample]})
    for ax in list(axes.flat)[len(examples):]:
        ax.set_visible(False)
    fig.savefig(out / "raw_trace_examples.png", dpi=180)
    fig.savefig(out / "raw_trace_examples.pdf")
    plt.close(fig)
    pd.DataFrame(examples).to_csv(out / "example_saccades.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(out / "example_raw_traces.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path,
                        default=Path("/mnt/ssd/YatesMarmoV1/processed"))
    parser.add_argument("--output-dir", type=Path,
                        default=STATS_DIR / "small_saccade_durations")
    parser.add_argument("--max-duration-ms", type=float, default=250.)
    parser.add_argument("--raw-example-session", default="",
                        help="Optional session for native eye-trace validation, e.g. Allen_2022-04-08")
    args = parser.parse_args()
    assert args.max_duration_ms > 0
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frames, inventory, session_inventory = [], [], []
    for session_dir in sorted(args.processed_root.iterdir()):
        if not session_dir.is_dir():
            continue
        source = session_dir / "saccades/saccades.json"
        if not source.exists():
            session_inventory.append({"session": session_dir.name,
                                      "status": "no_saved_saccades"})
            continue
        events = pd.DataFrame(json.loads(source.read_text()))
        events["source_event_index"] = events.index
        events["source_json"] = str(source)
        events["session"] = session_dir.name
        events["subject"] = session_dir.name.split("_")[0]
        events["duplicate"] = events.duplicated(["start_idx", "end_idx"], keep="first")
        events["amplitude_deg"] = np.hypot(events.end_x - events.start_x,
                                          events.end_y - events.start_y)
        events["duration_ms"] = (events.end_time - events.start_time) * 1000
        n_raw = len(events)
        events = events[np.isfinite(events.amplitude_deg) &
                        (events.amplitude_deg > 0) & (events.amplitude_deg < 1)].copy()
        events["valid_duration"] = (np.isfinite(events.duration_ms) &
                                    (events.duration_ms > 0) &
                                    (events.duration_ms <= args.max_duration_ms))
        events["positive_peak"] = np.isfinite(events.A) & (events.A > 0)
        events["included"] = ~events.duplicate & events.valid_duration & events.positive_peak
        events["protocol"] = "unassigned"
        events["protocol_matches"] = ""
        events["trial_idx"] = -1
        events["trial_start_s"] = np.nan
        events["trial_end_s"] = np.nan
        events["dataset_eye_valid"] = False
        dpi = session_dir / "dpi/ddpi.csv"
        sample_time = pd.read_csv(dpi, usecols=["t_ephys"], nrows=20000).t_ephys.to_numpy()
        delta = np.diff(sample_time)
        interval = float(np.median(delta[np.isfinite(delta) & (delta > 0)]) * 1000)
        events["native_sample_interval_ms"] = interval
        for protocol in PROTOCOLS:
            dataset = session_dir / "datasets" / f"{protocol}.dset"
            if dataset.exists():
                inv = match_trials(events, dataset)
                inventory.append({"session": session_dir.name, **inv})
        session_inventory.append({"session": session_dir.name, "status": "analyzed",
                                  "n_raw": n_raw, "n_subdegree_before_qc": len(events),
                                  "n_subdegree_duplicates": int(events.duplicate.sum()),
                                  "n_subdegree_bad_duration": int((~events.valid_duration).sum()),
                                  "n_subdegree_nonpositive_peak": int((~events.positive_peak).sum()),
                                  "n_subdegree_included": int(events.included.sum()),
                                  "native_sample_interval_ms": interval})
        frames.append(events)
        print(f"{session_dir.name}: {len(events):,} subdegree, "
              f"{events.included.sum():,} after QC", flush=True)
    audit = pd.concat(frames, ignore_index=True)
    selected = audit[audit.included].copy()
    audit.to_csv(out / "subdegree_events_with_qc.csv", index=False)
    selected.to_csv(out / "subdegree_saccades.csv", index=False)
    selected[selected.protocol.isin(LABELS)].to_csv(
        out / "freeview_and_fixation_saccades.csv", index=False)
    pd.DataFrame(inventory).to_csv(out / "dataset_inventory.csv", index=False)
    pd.DataFrame(session_inventory).to_csv(out / "session_inventory.csv", index=False)
    summaries = []
    for protocol in ("all_saved", *PROTOCOLS, "unassigned", "ambiguous", "backimage_and_fixrsvp"):
        data = selected if protocol == "all_saved" else (
            selected[selected.protocol.isin(LABELS)] if protocol == "backimage_and_fixrsvp"
            else selected[selected.protocol == protocol])
        summaries.append({"protocol": protocol, "subject": "pooled", **summary(data)})
        for subject, group in data.groupby("subject"):
            summaries.append({"protocol": protocol, "subject": subject, **summary(group)})
    stats = pd.DataFrame(summaries)
    stats.to_csv(out / "duration_summary.csv", index=False)
    per_session = pd.DataFrame([
        {"session": session, "subject": group.subject.iloc[0], "protocol": protocol,
         "native_sample_interval_ms": group.native_sample_interval_ms.iloc[0],
         **summary(group)}
        for (session, protocol), group in selected.groupby(["session", "protocol"])
    ])
    per_session.to_csv(out / "duration_by_session.csv", index=False)
    amp_rows, sensitivity_rows = [], []
    for protocol in LABELS:
        data = selected[selected.protocol == protocol]
        for lo, hi in zip(np.arange(0, 1, .2), np.arange(.2, 1.01, .2)):
            amp_rows.append({"protocol": protocol, "amplitude_min_deg": lo,
                             "amplitude_max_deg": hi,
                             **summary(data[(data.amplitude_deg >= lo) & (data.amplitude_deg < hi)])})
        for label, subset in [
            ("primary", data),
            ("duration_le_100ms", data[data.duration_ms <= 100]),
            ("dataset_eye_valid", data[data.dataset_eye_valid]),
            ("sample_interval_le_2ms", data[data.native_sample_interval_ms <= 2]),
            ("exclude_amplitude_lt_0.1deg", data[data.amplitude_deg >= .1]),
        ]:
            sensitivity_rows.append({"protocol": protocol, "filter": label, **summary(subset)})
    pd.DataFrame(amp_rows).to_csv(out / "duration_by_amplitude.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(out / "sensitivity.csv", index=False)
    make_plots(selected, per_session, out)
    smoothing_check(out)
    if args.raw_example_session:
        raw_examples(selected, args.processed_root, args.raw_example_session, out)
    methods = {
        "source_root": str(args.processed_root),
        "amplitude_definition": "hypot(end_x - start_x, end_y - start_y), saved positions in degrees",
        "amplitude_selection": "0 < amplitude_deg < 1 (strict upper bound)",
        "duration_definition": "1000 * (end_time - start_time), fitted 10% velocity boundaries",
        "duration_qc": f"finite, 0 < duration_ms <= {args.max_duration_ms}, finite A > 0",
        "deduplication": "one event per (session, start_idx, end_idx); keep first source row",
        "protocol_matching": "entire fitted interval within one contiguous trial; t_bins edges +/- dt/2",
        "sample_interval_estimate": "median positive finite t_ephys difference in first 20,000 native samples",
        "native_validity": "saved detector already excludes invalid samples; dataset-valid sensitivity also exported",
        "detector_source": "/home/jake/repos/DataYatesV1/DataYatesV1/utils/detect_saccades.py",
        "uncertainty": "IQR describes event spread, not a confidence interval; sessions and subjects are repeated observations",
        "limitations": [
            "Saved detector smooths position (11-sample Savitzky-Golay) and speed (Gaussian sigma=2 samples).",
            "Fitted durations depend on smoothing, sampling rate, and the 10% boundary definition.",
            "Saved amplitude uses the padded detection-window endpoints, not fitted-time interpolated endpoints.",
            "No raw re-detection or manual validation of every event; tiny events may be missed by the saved threshold.",
            "Detector splitting code duplicates the first subwindow; duplicate rows removed, missing second events not recovered.",
            "Only sessions with saved JSON detections were analyzed; unassigned events include intertrial periods and unmatched protocols.",
            "Events within overlapping exported trials from different protocols are labeled ambiguous and excluded from task summaries.",
            "The idealized smoothing check produces about 26 ms for an instantaneous step. Cached widths alone do not resolve the unsmoothed movement duration.",
            "Raw examples use current dpi calibration; saved endpoint coordinates can differ because the source files have been updated since detection. The amplitude criterion uses saved coordinates."
        ],
    }
    (out / "methods.json").write_text(json.dumps(methods, indent=2) + "\n")
    print(stats[stats.subject == "pooled"].to_string(index=False))
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
