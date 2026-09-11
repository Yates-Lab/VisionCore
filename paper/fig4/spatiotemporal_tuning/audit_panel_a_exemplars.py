#!/usr/bin/env python3
"""Select a legible, auditable Panel-A retinal-motion example.

Trace windows and natural images are selected without looking at model responses.
The selected 250-ms trace is a genuinely measured, low-pass-filtered fixation
segment. It is translated so measured-motion and stabilized histories share
the retinal frame at the model's resolved mean peak temporal lag; only the
surrounding motion history differs.

The final unit/image/trace combination is explicitly an illustrative example.
It is selected from the deterministic candidate grid using disclosed response
clarity gates.  Every scored candidate is retained in ``candidate_metrics.csv``.
Population inference must come from the factorial replay, never this audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.retinal_replay import (  # noqa: E402
    render_movies,
)
from paper.fig4.spatiotemporal_tuning._spectral_shards import (  # noqa: E402
    load_and_merge_shards,
)
from paper.fig4.upstream.real_trace_matrix.core import extract_patch  # noqa: E402
from paper.fig4.upstream.real_trace_matrix.model import (  # noqa: E402
    RealTraceMatrixScorer,
)
from paper.fig4.spatiotemporal_tuning.activation_map_metrics import (  # noqa: E402
    map_statistics,
    model_mean_peak_lag,
    score_histories,
)


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-table", type=Path, required=True)
    parser.add_argument("--trace-bank", type=Path, required=True)
    parser.add_argument("--rucci-ensemble", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--population-spec-dir", type=Path, required=True)
    parser.add_argument("--population-version", required=True)
    parser.add_argument("--mcfarland-outputs", type=Path, required=True)
    parser.add_argument(
        "--population-shards",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Completed factorial replay used to preselect a robust example unit. "
            "The final image/trace endpoint remains an explicitly illustrative selection."
        ),
    )
    parser.add_argument(
        "--population-matrix-dir",
        type=Path,
        default=None,
        help=(
            "Completed image×fixation response matrix used to preselect a robust "
            "example unit. This is the preferred input when Panel B and Panel A "
            "share the same 250-ms replay bank."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--history-samples", type=int, default=60)
    parser.add_argument("--patch-size-px", type=int, default=540)
    parser.add_argument("--n-image-candidates", type=int, default=12)
    parser.add_argument("--image-selection", choices=("contrast_quantiles", "central_detail"),
                        default="contrast_quantiles")
    parser.add_argument("--n-trace-candidates-per-subject", type=int, default=6)
    parser.add_argument(
        "--population-unit-candidates",
        type=int,
        default=8,
        help=(
            "Number of independently population-ranked units allowed into the "
            "bounded endpoint-clarity audit."
        ),
    )
    parser.add_argument("--map-batch-size", type=int, default=6)
    parser.add_argument("--minimum-window-span-deg", type=float, default=0.35)
    parser.add_argument("--maximum-window-span-deg", type=float, default=1.25)
    parser.add_argument("--maximum-window-peak-speed-deg-s", type=float, default=100.0)
    parser.add_argument("--minimum-stable-rate-spikes-s", type=float, default=0.5)
    parser.add_argument("--minimum-stable-ssi", type=float, default=0.05)
    parser.add_argument("--minimum-rate-change-percent", type=float, default=25.0)
    parser.add_argument("--maximum-rate-change-percent", type=float, default=200.0)
    parser.add_argument("--minimum-ssi-change-bits", type=float, default=0.02)
    parser.add_argument("--minimum-ssi-change-percent", type=float, default=10.0)
    parser.add_argument("--maximum-ssi-change-percent", type=float, default=200.0)
    parser.add_argument("--maximum-ssi-change-bits", type=float, default=0.25)
    parser.add_argument("--maximum-motion-rate-spikes-s", type=float, default=50.0)
    parser.add_argument("--minimum-normalized-map-rms-change", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _rank01(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True).fillna(0.0)


def _window_metrics(window: np.ndarray, rate_hz: float) -> dict[str, float]:
    steps = np.linalg.norm(np.diff(window, axis=0), axis=1)
    relative = window - window[-1]
    return {
        "window_span_deg": float(np.max(np.linalg.norm(relative, axis=1))),
        "window_path_length_deg": float(np.sum(steps)),
        "window_net_displacement_deg": float(np.linalg.norm(window[-1] - window[0])),
        "window_peak_speed_deg_s": float(np.max(steps) * float(rate_hz)),
        "window_rms_radius_deg": float(
            np.sqrt(np.mean(np.sum(relative * relative, axis=1)))
        ),
    }


def _window_temporal_spectrum(
    window: np.ndarray, rate_hz: float
) -> dict[str, float]:
    """Summarize a filtered eye-position window without an external row join."""
    value = np.asarray(window, dtype=np.float64)
    centered = value - np.mean(value, axis=0, keepdims=True)
    taper = np.hanning(len(centered))[:, None]
    spectrum = np.fft.rfft(centered * taper, axis=0)
    power = np.sum(np.abs(spectrum) ** 2, axis=1)
    frequency = np.fft.rfftfreq(len(centered), d=1.0 / float(rate_hz))
    dynamic = frequency > 0
    mass = float(np.sum(power[dynamic]))
    centroid = float(
        np.sum(frequency[dynamic] * power[dynamic]) / max(mass, EPS)
    )
    return {
        "power_centroid_hz": centroid,
        "dynamic_power_mass": mass,
    }


def select_trace_windows(
    traces: np.ndarray,
    trace_table: pd.DataFrame,
    *,
    history_samples: int,
    per_subject: int,
    minimum_span: float,
    maximum_span: float,
    maximum_peak_speed: float,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Select high-motion but bounded windows independently of model response."""
    rate_hz = float(trace_table.target_rate_hz.iloc[0])
    rows: list[dict[str, Any]] = []
    windows: dict[int, np.ndarray] = {}
    first_complete_endpoint = 2 * int(history_samples) - 1
    for trace_index, trace in enumerate(np.asarray(traces, dtype=np.float32)):
        if "event_class" in trace_table and trace_table.iloc[trace_index].event_class == "excluded":
            continue
        best: dict[str, Any] | None = None
        for endpoint in range(first_complete_endpoint, len(trace)):
            window = trace[endpoint - int(history_samples) + 1 : endpoint + 1]
            metrics = _window_metrics(window, rate_hz)
            if not (
                float(minimum_span) <= metrics["window_span_deg"] <= float(maximum_span)
                and metrics["window_peak_speed_deg_s"] <= float(maximum_peak_speed)
            ):
                continue
            score = (
                metrics["window_span_deg"]
                + 0.10 * min(metrics["window_path_length_deg"], 2.0)
            )
            candidate = {
                "trace_index": int(trace_index),
                "endpoint_frame": int(endpoint),
                "motion_selection_score": float(score),
                **_window_temporal_spectrum(window, rate_hz),
                **metrics,
            }
            if best is None or score > float(best["motion_selection_score"]):
                best = candidate
                windows[int(trace_index)] = np.asarray(window - window[-1], dtype=np.float32)
        if best is None:
            continue
        table_row = trace_table.loc[trace_table.trace_index.eq(int(trace_index))]
        if len(table_row) != 1:
            raise ValueError(f"trace {trace_index} does not resolve uniquely")
        row = table_row.iloc[0]
        best.update(
            {
                "subject": str(row.session).split("_")[0],
                "saved_microsaccade_count": int(row.get("verified_microsaccade_count", row.saved_microsaccade_count)),
                "session": str(row.session),
            }
        )
        rows.append(best)
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise RuntimeError("no trace windows passed the outcome-independent motion gates")
    low_centroid, high_centroid = candidates.power_centroid_hz.quantile(
        (0.25, 0.75)
    )
    candidates["spectral_regime_code"] = np.select(
        (
            candidates.power_centroid_hz <= float(low_centroid),
            candidates.power_centroid_hz >= float(high_centroid),
        ),
        (0, 1),
        default=-1,
    ).astype(int)
    # Prefer the rapid-transient quartile and detector-positive microsaccade
    # epochs, while preserving a fallback to any bounded real trace.
    candidates["event_priority"] = (
        candidates.spectral_regime_code.eq(1).astype(int)
        + candidates.saved_microsaccade_count.gt(0).astype(int)
    )
    if "event_class" in trace_table:
        candidates["spectral_regime_code"] = candidates.saved_microsaccade_count.gt(0).astype(int)
        candidates["event_priority"] = candidates.saved_microsaccade_count.gt(0).astype(int)
    selected = []
    for _, group in candidates.groupby("subject", sort=True):
        ordered = group.sort_values(
            ["event_priority", "motion_selection_score", "power_centroid_hz"],
            ascending=[False, False, False],
        )
        selected.append(ordered.head(int(per_subject)))
    selected_frame = pd.concat(selected, ignore_index=True)
    selected_frame = selected_frame.sort_values(
        ["subject", "event_priority", "motion_selection_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    selected_windows = {
        int(row.trace_index): windows[int(row.trace_index)]
        for row in selected_frame.itertuples()
    }
    return selected_frame, selected_windows


def select_images(table: pd.DataFrame, count: int, *, method: str = "contrast_quantiles",
                  patch_size_px: int = 540) -> pd.DataFrame:
    """Choose image candidates from image features without using neural responses."""
    valid = table.copy()
    if "image_feature_ok" in valid:
        valid = valid.loc[valid.image_feature_ok.astype(bool)]
    valid = valid.loc[np.isfinite(valid.image_patch_rms_contrast)]
    if method == "central_detail":
        # Evaluate image structure before running the model. The central half
        # of the rendered field excludes detail confined to the patch border.
        cache = {}
        detail = []
        for _, row in valid.iterrows():
            patch, _ = extract_patch(row, canvas_cache=cache, patch_size_px=patch_size_px)
            frame = render_movies(patch, np.zeros((1, 1, 2), dtype=np.float32),
                                  device="cpu")[0, 0]
            h, w = frame.shape
            center = frame[h // 4:3 * h // 4, w // 4:3 * w // 4].astype(float)
            gy, gx = np.gradient(center)
            detail.append(float(np.sqrt(np.mean(gx * gx + gy * gy))))
        valid["central_gradient_rms"] = detail
        valid = valid.sort_values(["central_gradient_rms", "image_index"],
                                  ascending=[False, True])
        if len(valid) < int(count):
            raise RuntimeError(f"only {len(valid)} valid images for {count} requested candidates")
        return valid.head(int(count)).copy().reset_index(drop=True)
    if method != "contrast_quantiles":
        raise ValueError(f"unknown image selection method: {method}")
    valid = valid.sort_values("image_patch_rms_contrast").reset_index(drop=True)
    if len(valid) < int(count):
        raise RuntimeError(f"only {len(valid)} valid images for {count} requested candidates")
    positions = np.unique(
        np.round(np.linspace(0, len(valid) - 1, int(count))).astype(int)
    )
    if len(positions) != int(count):
        raise RuntimeError("contrast-quantile image selection produced duplicate rows")
    return valid.iloc[positions].copy().reset_index(drop=True)


def population_unit_effects(paths: list[Path]) -> pd.DataFrame:
    """Rank units by robust rate and SSI gains in the complete replay."""
    data = load_and_merge_shards(paths)
    scales = np.asarray(data["motion_scales"], dtype=float)
    stable = int(np.flatnonzero(np.isclose(scales, 0.0))[0])
    motion = int(np.flatnonzero(np.isclose(scales, 1.0))[0])
    rate_stable = np.nanmedian(np.asarray(data["mean_rate"])[:, :, stable], axis=(0, 1))
    rate_motion = np.nanmedian(np.asarray(data["mean_rate"])[:, :, motion], axis=(0, 1))
    ssi_stable = np.nanmedian(np.asarray(data["map_ssi"])[:, :, stable], axis=(0, 1))
    ssi_motion = np.nanmedian(np.asarray(data["map_ssi"])[:, :, motion], axis=(0, 1))
    frame = pd.DataFrame(
        {
            "unit_index": np.asarray(data["unit_indices"], dtype=int),
            "population_rate_change_percent": 100.0
            * (rate_motion - rate_stable)
            / np.maximum(rate_stable, EPS),
            "population_rate_change_spikes_s": rate_motion - rate_stable,
            "population_ssi_change_percent": 100.0
            * (ssi_motion - ssi_stable)
            / np.maximum(ssi_stable, EPS),
            "population_ssi_change_bits_per_spike": ssi_motion - ssi_stable,
        }
    )
    positive = frame.loc[
        frame.population_rate_change_percent.gt(0.0)
        & frame.population_ssi_change_percent.gt(0.0)
    ].copy()
    if positive.empty:
        raise RuntimeError("no unit has positive population rate and SSI effects")
    positive["population_joint_rank_score"] = (
        _rank01(positive.population_rate_change_percent)
        + _rank01(positive.population_rate_change_spikes_s)
        + _rank01(positive.population_ssi_change_percent)
        + _rank01(positive.population_ssi_change_bits_per_spike)
    )
    frame = frame.merge(
        positive[["unit_index", "population_joint_rank_score"]],
        on="unit_index",
        how="left",
    )
    frame["population_joint_rank_score"] = frame.population_joint_rank_score.fillna(0.0)
    return frame


def population_unit_effects_from_matrix(matrix_dir: Path) -> pd.DataFrame:
    """Rank units from the same matched motion/stabilized matrix used in Panel B."""
    matrix_dir = Path(matrix_dir)
    summary = json.loads((matrix_dir / "summary.json").read_text())
    n_images = int(summary["n_images"])
    n_traces = int(summary["n_traces"])
    bin_seconds = float(summary["validated_common_provenance"]["bin_seconds"])
    if not np.isfinite(bin_seconds) or bin_seconds <= 0:
        raise ValueError("matrix summary has an invalid bin_seconds")
    rate_motion = np.load(matrix_dir / "mean_rate_matrix.npy")
    ssi_motion = np.load(matrix_dir / "ssi_matrix.npy")
    rate_stable = np.load(matrix_dir / "stabilized_mean_rate_by_image.npy")
    ssi_stable = np.load(matrix_dir / "stabilized_ssi_by_image.npy")
    if rate_motion.ndim != 2 or ssi_motion.shape != rate_motion.shape:
        raise ValueError("moving rate and SSI matrices must be aligned 2-D arrays")
    n_units = int(rate_motion.shape[1])
    expected_moving = (n_images * n_traces, n_units)
    expected_stable = (n_images, n_units)
    if rate_motion.shape != expected_moving:
        raise ValueError(
            f"moving response matrix has shape {rate_motion.shape}, expected {expected_moving}"
        )
    if rate_stable.shape != expected_stable or ssi_stable.shape != expected_stable:
        raise ValueError("stabilized matrices do not match the image and unit axes")
    rate_motion = rate_motion.reshape(n_images, n_traces, n_units)
    ssi_motion = ssi_motion.reshape(n_images, n_traces, n_units)
    motion_rate_center = np.nanmedian(rate_motion, axis=(0, 1))
    motion_ssi_center = np.nanmedian(ssi_motion, axis=(0, 1))
    stable_rate_center = np.nanmedian(rate_stable, axis=0)
    stable_ssi_center = np.nanmedian(ssi_stable, axis=0)
    frame = pd.DataFrame(
        {
            "unit_index": np.arange(n_units, dtype=int),
            "population_rate_change_percent": 100.0
            * (motion_rate_center - stable_rate_center)
            / np.maximum(stable_rate_center, EPS),
            "population_rate_change_spikes_s": motion_rate_center - stable_rate_center,
            "population_ssi_change_percent": 100.0
            * (motion_ssi_center - stable_ssi_center)
            / np.maximum(stable_ssi_center, EPS),
            "population_ssi_change_bits_per_spike": motion_ssi_center
            - stable_ssi_center,
        }
    )
    positive = frame.loc[
        frame.population_rate_change_percent.gt(0.0)
        & frame.population_ssi_change_percent.gt(0.0)
    ].copy()
    if positive.empty:
        raise RuntimeError("no unit has positive population rate and SSI effects")
    positive["population_joint_rank_score"] = (
        _rank01(positive.population_rate_change_percent)
        + _rank01(positive.population_rate_change_spikes_s)
        + _rank01(positive.population_ssi_change_percent)
        + _rank01(positive.population_ssi_change_bits_per_spike)
    )
    frame = frame.merge(
        positive[["unit_index", "population_joint_rank_score"]],
        on="unit_index",
        how="left",
    )
    frame["population_joint_rank_score"] = frame.population_joint_rank_score.fillna(0.0)
    return frame


def assert_matrix_checkpoint(matrix_dir: Path, checkpoint: Path) -> str:
    """Reject population preselection computed by a different twin."""
    summary = json.loads((Path(matrix_dir) / "summary.json").read_text())
    observed = summary.get("validated_common_provenance", {}).get(
        "model_provenance.model.checkpoint_sha256"
    )
    if observed is None:
        observed = summary.get("model_provenance", {}).get("model", {}).get(
            "checkpoint_sha256"
        )
    if observed is None:
        raise ValueError("population response matrix lacks checkpoint provenance")
    expected = sha256(Path(checkpoint))
    if str(observed) != expected:
        raise ValueError(
            "population response matrix and Panel-A scorer use different checkpoints"
        )
    return expected


def choose_example(metrics: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.Series, str]:
    eligible = metrics.loc[
        metrics.stable_rate_spikes_s.ge(float(args.minimum_stable_rate_spikes_s))
        & metrics.stable_ssi_bits_per_spike.ge(float(args.minimum_stable_ssi))
    ].copy()
    if eligible.empty:
        raise RuntimeError("no unit passed the baseline rate and SSI gates")
    if "population_joint_rank_score" in eligible:
        count = max(1, int(getattr(args, "population_unit_candidates", 8)))
        unit_scores = (
            eligible.groupby("unit_index").population_joint_rank_score.max()
            .sort_values(ascending=False)
        )
        unit_scores = unit_scores.loc[unit_scores.gt(0.0)].head(count)
        if unit_scores.empty:
            raise RuntimeError("population replay did not nominate any positive-effect units")
        eligible = eligible.loc[eligible.unit_index.isin(unit_scores.index)].copy()
        unit_selection = f"top-{len(unit_scores)} population-robust units + "
    else:
        unit_selection = ""
    strict = eligible.loc[
        eligible.rate_change_percent.ge(float(args.minimum_rate_change_percent))
        & eligible.rate_change_percent.le(float(args.maximum_rate_change_percent))
        & eligible.ssi_change_bits_per_spike.ge(float(args.minimum_ssi_change_bits))
        & eligible.ssi_change_bits_per_spike.le(float(args.maximum_ssi_change_bits))
        & eligible.ssi_change_percent.ge(float(args.minimum_ssi_change_percent))
        & eligible.ssi_change_percent.le(float(args.maximum_ssi_change_percent))
        & eligible.motion_rate_spikes_s.le(float(args.maximum_motion_rate_spikes_s))
        & eligible.normalized_map_rms_change.ge(
            float(args.minimum_normalized_map_rms_change)
        )
    ].copy()
    if not strict.empty:
        pool = strict
        tier = unit_selection + "bounded disclosed endpoint-clarity gates"
    else:
        pool = eligible.loc[
            eligible.rate_change_percent.gt(0.0)
            & eligible.ssi_change_bits_per_spike.gt(0.0)
        ].copy()
        tier = (
            unit_selection
            + "positive-effect fallback; strict clarity gates had no candidates"
        )
    if pool.empty:
        raise RuntimeError("no baseline-responsive example had positive rate and SSI changes")
    # The bounded gates prevent a denominator-sensitive extreme from winning.
    # Within the population-preselected unit and interpretable endpoint range,
    # select the map pair with the clearest normalized spatial change.
    pool["clarity_score"] = _rank01(pool.normalized_map_rms_change)
    sort_columns = ["clarity_score", "stable_rate_spikes_s", "unit_index"]
    ascending = [False, False, True]
    if "population_joint_rank_score" in pool:
        sort_columns.insert(1, "population_joint_rank_score")
        ascending.insert(1, False)
    pool = pool.sort_values(sort_columns, ascending=ascending)
    return pool.iloc[0], tier


def main() -> None:
    args = parse_args()
    if args.population_shards and args.population_matrix_dir is not None:
        raise ValueError(
            "use either --population-shards or --population-matrix-dir, not both"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.population_matrix_dir is not None:
        assert_matrix_checkpoint(args.population_matrix_dir, args.checkpoint)
    traces = np.load(args.trace_bank / "trace_xy_filtered.npy")
    trace_table = pd.read_csv(args.trace_bank / "trace_table.csv")
    if not args.rucci_ensemble.exists():
        raise FileNotFoundError(args.rucci_ensemble)
    trace_candidates, windows = select_trace_windows(
        traces,
        trace_table,
        history_samples=int(args.history_samples),
        per_subject=int(args.n_trace_candidates_per_subject),
        minimum_span=float(args.minimum_window_span_deg),
        maximum_span=float(args.maximum_window_span_deg),
        maximum_peak_speed=float(args.maximum_window_peak_speed_deg_s),
    )
    image_table = pd.read_csv(args.image_table)
    image_candidates = select_images(image_table, int(args.n_image_candidates),
                                     method=args.image_selection, patch_size_px=args.patch_size_px)
    trace_candidates.to_csv(args.out_dir / "trace_candidates.csv", index=False)
    image_candidates.to_csv(args.out_dir / "image_candidates.csv", index=False)
    population_units = None
    population_lookup: dict[int, dict[str, float]] = {}
    if args.population_shards:
        population_units = population_unit_effects(args.population_shards)
    elif args.population_matrix_dir is not None:
        population_units = population_unit_effects_from_matrix(
            args.population_matrix_dir
        )
    if population_units is not None:
        population_units.to_csv(args.out_dir / "population_unit_effects.csv", index=False)
        population_lookup = {
            int(row.unit_index): {
                key: float(getattr(row, key))
                for key in (
                    "population_rate_change_percent",
                    "population_rate_change_spikes_s",
                    "population_ssi_change_percent",
                    "population_ssi_change_bits_per_spike",
                    "population_joint_rank_score",
                )
            }
            for row in population_units.itertuples()
        }

    scorer = RealTraceMatrixScorer.load(
        checkpoint_path=args.checkpoint.resolve(),
        dataset_configs=args.dataset_config.resolve(),
        population_spec_dir=args.population_spec_dir.resolve(),
        population_version=str(args.population_version),
        device=str(args.device),
        strict=True,
        mcfarland_outputs=args.mcfarland_outputs.resolve(),
    )
    if int(scorer.n_lags) != int(args.history_samples):
        raise RuntimeError(
            f"model expects {scorer.n_lags} history frames, not {args.history_samples}"
        )
    trace_indices = trace_candidates.trace_index.to_numpy(dtype=int)
    lag_contract = model_mean_peak_lag(scorer.model.model.convnet)
    anchor_lag = int(lag_contract["resolved_rounded_peak_lag_frames"])
    anchor_index = int(args.history_samples) - 1 - anchor_lag
    trace_windows = np.stack([windows[int(index)] for index in trace_indices])
    trace_windows = trace_windows - trace_windows[:, anchor_index : anchor_index + 1]
    metrics_rows: list[dict[str, Any]] = []
    saved: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    canvas_cache: dict[tuple[str, int], tuple[np.ndarray, float, tuple[int, int]]] = {}
    for image_number, image_row in image_candidates.iterrows():
        patch, _ = extract_patch(
            image_row, canvas_cache=canvas_cache, patch_size_px=int(args.patch_size_px)
        )
        histories = render_movies(
            patch,
            np.concatenate(
                (
                    np.zeros((1, int(args.history_samples), 2), dtype=np.float32),
                    trace_windows,
                ),
                axis=0,
            ),
            device=str(args.device),
        )
        maps = score_histories(scorer, histories, batch_size=int(args.map_batch_size))
        stable = map_statistics(maps[:1], float(scorer.output_rate_hz))
        motion = map_statistics(maps[1:], float(scorer.output_rate_hz))
        stable_rate = stable["rate_spikes_s"][0]
        stable_ssi = stable["ssi_bits_per_spike"][0]
        stable_gain = stable["normalized_map"][0]
        for trace_local, trace_index in enumerate(trace_indices):
            motion_rate = motion["rate_spikes_s"][trace_local]
            motion_ssi = motion["ssi_bits_per_spike"][trace_local]
            motion_gain = motion["normalized_map"][trace_local]
            rate_change = motion_rate - stable_rate
            ssi_change = motion_ssi - stable_ssi
            map_rms = np.sqrt(np.mean((motion_gain - stable_gain) ** 2, axis=(-2, -1)))
            trace_row = trace_candidates.loc[
                trace_candidates.trace_index.eq(int(trace_index))
            ].iloc[0]
            for unit_index in range(int(scorer.n_units)):
                metrics_rows.append(
                    {
                        "image_index": int(image_row.image_index),
                        "trace_index": int(trace_index),
                        "endpoint_frame": int(trace_row.endpoint_frame),
                        "unit_index": int(unit_index),
                        "subject": str(trace_row.subject),
                        "spectral_regime_code": int(trace_row.spectral_regime_code),
                        "power_centroid_hz": float(trace_row.power_centroid_hz),
                        "saved_microsaccade_count": int(
                            trace_row.saved_microsaccade_count
                        ),
                        "window_span_deg": float(trace_row.window_span_deg),
                        "window_path_length_deg": float(
                            trace_row.window_path_length_deg
                        ),
                        "window_peak_speed_deg_s": float(
                            trace_row.window_peak_speed_deg_s
                        ),
                        "stable_rate_spikes_s": float(stable_rate[unit_index]),
                        "motion_rate_spikes_s": float(motion_rate[unit_index]),
                        "rate_change_spikes_s": float(rate_change[unit_index]),
                        "rate_change_percent": float(
                            100.0 * rate_change[unit_index] / max(stable_rate[unit_index], EPS)
                        ),
                        "stable_ssi_bits_per_spike": float(stable_ssi[unit_index]),
                        "motion_ssi_bits_per_spike": float(motion_ssi[unit_index]),
                        "ssi_change_bits_per_spike": float(ssi_change[unit_index]),
                        "ssi_change_percent": float(
                            100.0 * ssi_change[unit_index] / max(stable_ssi[unit_index], EPS)
                        ),
                        "normalized_map_rms_change": float(map_rms[unit_index]),
                        **population_lookup.get(int(unit_index), {}),
                    }
                )
            saved[(int(image_row.image_index), int(trace_index))] = {
                "stable_maps": maps[0],
                "motion_maps": maps[trace_local + 1],
            }
        print(
            f"Panel-A audit image {image_number + 1}/{len(image_candidates)}",
            flush=True,
        )
    metrics = pd.DataFrame(metrics_rows)
    selected, selection_tier = choose_example(metrics, args)
    metrics["selected"] = (
        metrics.image_index.eq(int(selected.image_index))
        & metrics.trace_index.eq(int(selected.trace_index))
        & metrics.unit_index.eq(int(selected.unit_index))
    )
    metrics.to_csv(args.out_dir / "candidate_metrics.csv", index=False)
    key = (int(selected.image_index), int(selected.trace_index))
    data = saved[key]
    unit_index = int(selected.unit_index)
    window = windows[int(selected.trace_index)]
    window = window - window[anchor_index]
    selected_image = image_candidates.loc[
        image_candidates.image_index.eq(int(selected.image_index))
    ]
    if len(selected_image) != 1:
        raise RuntimeError("selected Panel-A image did not resolve uniquely")
    selected_patch, _ = extract_patch(
        selected_image.iloc[0],
        canvas_cache=canvas_cache,
        patch_size_px=int(args.patch_size_px),
    )
    selected_histories = render_movies(
        selected_patch,
        np.stack((np.zeros_like(window), window)),
        device=str(args.device),
    )
    np.savez_compressed(
        args.out_dir / "selected_example.npz",
        image_index=np.asarray(int(selected.image_index)),
        trace_index=np.asarray(int(selected.trace_index)),
        endpoint_frame=np.asarray(int(selected.endpoint_frame)),
        unit_index=np.asarray(unit_index),
        output_rate_hz=np.asarray(float(scorer.output_rate_hz)),
        trace_window_xy_filtered_endpoint_aligned=np.asarray(window, dtype=np.float32),
        stable_history=np.asarray(selected_histories[0], dtype=np.float32),
        motion_history=np.asarray(selected_histories[1], dtype=np.float32),
        stable_rate_map=np.asarray(data["stable_maps"][unit_index], dtype=np.float32),
        motion_rate_map=np.asarray(data["motion_maps"][unit_index], dtype=np.float32),
    )
    summary = {
        "analysis": "response-audited illustrative Panel-A exemplar",
        "model_label": str(args.model_label or args.checkpoint.stem),
        "population_version": str(args.population_version),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "trace_source": str((args.trace_bank / "trace_xy_filtered.npy").resolve()),
        "trace_filtering": "production low-pass-filtered real fixation bank",
        "rucci_ensemble_provenance": {
            "path": str(args.rucci_ensemble.resolve()),
            "sha256": sha256(args.rucci_ensemble),
            "used_for_endpoint_selection": False,
        },
        "trace_window_selection": {
            "uses_neural_response": False,
            "duration_ms": 1000.0 * int(args.history_samples) / float(scorer.input_rate_hz),
            "complete_analysis_window_only": True,
            "alignment_kind": "resolved_model_mean_peak_lag",
            "anchor_chronological_frame_index": int(anchor_index),
            "anchor_lag_frames": int(anchor_lag),
            "anchor_time_before_prediction_ms": float(
                1000.0 * anchor_lag / float(scorer.output_rate_hz)
            ),
            "alignment": (
                "measured window translated to zero at the rounded mean learned "
                "first-layer peak-energy lag; stabilized and motion histories share "
                "the model's most influential retinal frame"
            ),
            "candidate_count": int(len(trace_candidates)),
            "minimum_span_deg": float(args.minimum_window_span_deg),
            "maximum_span_deg": float(args.maximum_window_span_deg),
            "maximum_peak_speed_deg_s": float(args.maximum_window_peak_speed_deg_s),
        },
        "model_peak_lag": lag_contract,
        "image_selection": {
            "uses_neural_response": False,
            "method": ("largest RMS image gradient in the central half of the rendered field after image QC"
                       if args.image_selection == "central_detail" else
                       "even quantiles of natural-image RMS contrast after image QC"),
            "policy": args.image_selection,
            "candidate_count": int(len(image_candidates)),
        },
        "response_selection": {
            "illustrative_not_inferential": True,
            "selection_tier": selection_tier,
            "all_candidates_file": "candidate_metrics.csv",
            "unit_preselection": (
                f"top {int(args.population_unit_candidates)} units by joint rank across "
                "rate-percent, rate-absolute, SSI-percent, and SSI-absolute gains "
                "in the completed factorial replay"
                if population_units is not None
                else "none"
            ),
            "population_unit_effects_file": (
                "population_unit_effects.csv" if population_units is not None else None
            ),
            "baseline_rate_gate_spikes_s": float(args.minimum_stable_rate_spikes_s),
            "baseline_ssi_gate_bits_per_spike": float(args.minimum_stable_ssi),
            "rate_change_gate_percent": float(args.minimum_rate_change_percent),
            "rate_change_ceiling_percent": float(args.maximum_rate_change_percent),
            "ssi_change_gate_bits_per_spike": float(args.minimum_ssi_change_bits),
            "ssi_change_ceiling_bits_per_spike": float(args.maximum_ssi_change_bits),
            "ssi_change_gate_percent": float(args.minimum_ssi_change_percent),
            "ssi_change_ceiling_percent": float(args.maximum_ssi_change_percent),
            "motion_rate_ceiling_spikes_s": float(args.maximum_motion_rate_spikes_s),
            "normalized_map_rms_change_gate": float(
                args.minimum_normalized_map_rms_change
            ),
        },
        "selected": {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in selected.to_dict().items()
        },
        "n_scored_combinations": int(len(metrics)),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["selected"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
