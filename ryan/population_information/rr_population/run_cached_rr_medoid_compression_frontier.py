"""Cached post-hoc compression sweep for medoid-oriented V1-RR populations.

This script uses activation movies and final RR labels already written by
``inspect_canonical_twin_channel_redundancy.py``. It avoids importing that
interactive script, which executes many notebook cells at import time.

Run from the repo root::

    .venv/bin/python -m declan.run_cached_rr_medoid_compression_frontier
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "redundancy_resolved_v1_twin" / "step1_activation_fingerprints"
BASE_VERSION = "V1-RR_MS_min_complete0p65_split0p75_pair0p60_anyfail_finalsplit0p75"
BASE_LABELS_PATH = OUT_DIR / f"multistim_final_labels_{BASE_VERSION}.npy"
CONSTRUCTION_RECON_CSV = OUT_DIR / f"candidate_audit_reconstruction_{BASE_VERSION}_MultiStim_construction_final.csv"
UNIT_TABLE_CSV = OUT_DIR / "canonical_shared_twin_unit_table.csv"

TARGET_REPRESENTATIVES = 100
THRESHOLDS = (
    0.02,
    0.05,
    0.08,
    0.10,
    0.12,
    0.15,
    0.18,
    0.20,
    0.22,
    0.25,
    0.28,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
)
SIMILARITY_MODE = "min"
LINKAGE_METHOD = "complete"
FINGERPRINT_NORMALIZATION = "zscore"


def _safe_slug(value: object, max_len: int = 128) -> str:
    text = str(value)
    text = Path(text).stem if "/" in text else text
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    slug = "_".join(part for part in slug.split("_") if part)
    return slug[:max_len] or "unnamed"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if not math.isfinite(value) else value
    return str(obj)


def load_unit_rows(path: Path = UNIT_TABLE_CSV) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        for key in ("channel", "source_unit_index", "model_readout_index", "mcfarland_output_index"):
            if key in row and row[key] != "":
                row[key] = int(row[key])
        if "ccnorm" in row and row["ccnorm"] != "":
            row["ccnorm"] = float(row["ccnorm"])
        out.append(row)
    return out


def representative_members_from_labels(labels: np.ndarray) -> tuple[list[np.ndarray], list[tuple[str, int]]]:
    labels_arr = np.asarray(labels, dtype=int)
    rep_members: list[np.ndarray] = []
    rep_sources: list[tuple[str, int]] = []
    for group_id in sorted(set(labels_arr[labels_arr >= 0])):
        rep_members.append(np.flatnonzero(labels_arr == int(group_id)))
        rep_sources.append(("group", int(group_id)))
    for channel in np.flatnonzero(labels_arr == -1):
        rep_members.append(np.asarray([int(channel)], dtype=int))
        rep_sources.append(("singleton", int(channel)))
    return rep_members, rep_sources


def population_counts(labels: np.ndarray, version: str) -> dict[str, Any]:
    labels_arr = np.asarray(labels, dtype=int)
    n_groups = len(set(labels_arr[labels_arr >= 0]))
    n_singletons = int(np.sum(labels_arr == -1))
    return {
        "version": version,
        "n_representatives": int(n_groups + n_singletons),
        "n_groups": int(n_groups),
        "n_singletons": n_singletons,
        "n_excluded": int(np.sum(labels_arr == -2)),
        "largest_group": int(
            max([np.sum(labels_arr == gid) for gid in sorted(set(labels_arr[labels_arr >= 0]))] or [1])
        ),
    }


def movie_paths_from_reconstruction_csv(path: Path, *, max_cases: int | None = None) -> list[tuple[str, Path]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cases = list(dict.fromkeys(row["case"] for row in rows))
    if max_cases is not None and max_cases > 0:
        cases = cases[: int(max_cases)]
    out: list[tuple[str, Path]] = []
    for case in cases:
        if case.startswith("FixRSVP_"):
            movie_path = OUT_DIR / "activation_movie_fixrsvp_dynamic_fpi6_frames192_lag32_out151x151_scale1.npz"
        else:
            case_slug = _safe_slug(case)
            matches = sorted(OUT_DIR.glob(f"activation_movie_{case_slug}_trace000_stored_frames*_lag32_out151x151_scale1.npz"))
            if not matches:
                raise FileNotFoundError(f"No cached activation movie found for case {case!r} (slug {case_slug})")
            movie_path = matches[0]
        out.append((case, movie_path))
    return out


def load_activation_movie(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "activation_movie" not in data:
            raise KeyError(f"{path} has no activation_movie array")
        return np.asarray(data["activation_movie"], dtype=np.float32)


def representative_fingerprints_from_movie(
    movie: np.ndarray,
    labels: np.ndarray,
    *,
    normalization: str = FINGERPRINT_NORMALIZATION,
    eps: float = 1e-8,
) -> np.ndarray:
    y = np.asarray(movie, dtype=np.float32)
    if y.ndim != 4:
        raise ValueError(f"Expected T x C x H x W movie, got {y.shape}")
    rep_members, _ = representative_members_from_labels(labels)
    n_samples = int(y.shape[0] * y.shape[2] * y.shape[3])
    rep_fp = np.empty((len(rep_members), n_samples), dtype=np.float32)
    for rep_idx, members in enumerate(rep_members):
        rep_map = np.nanmean(y[:, members], axis=1)
        rep_fp[rep_idx] = rep_map.reshape(-1)
    if normalization == "none":
        return rep_fp
    rep_fp -= np.nanmean(rep_fp, axis=1, keepdims=True)
    if normalization == "center":
        return rep_fp
    if normalization == "zscore":
        rep_fp /= np.nanstd(rep_fp, axis=1, keepdims=True) + eps
        return rep_fp
    raise ValueError(f"Unknown normalization {normalization!r}")


def correlation_from_fingerprints(fingerprints: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(fingerprints, dtype=np.float32)
    x -= np.nanmean(x, axis=1, keepdims=True)
    norms = np.sqrt(np.nansum(x * x, axis=1, keepdims=True))
    x = x / np.maximum(norms, eps)
    corr = x @ x.T
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float32, copy=False)


def combine_similarity_matrices(corrs: list[np.ndarray], mode: str = SIMILARITY_MODE) -> np.ndarray:
    stack = np.stack([np.asarray(corr, dtype=np.float32) for corr in corrs], axis=0)
    if mode == "min":
        out = np.nanmin(stack, axis=0)
    elif mode == "mean":
        out = np.nanmean(stack, axis=0)
    else:
        raise ValueError(f"Unknown similarity mode {mode!r}")
    out = np.asarray(out, dtype=np.float32)
    np.fill_diagonal(out, 1.0)
    return out


def redundancy_groups_from_corr(corr: np.ndarray, threshold: float, *, method: str = LINKAGE_METHOD) -> np.ndarray:
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except ImportError as exc:
        raise RuntimeError("scipy is required for clustering") from exc
    dist = np.clip(1.0 - np.asarray(corr, dtype=np.float64), 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method=method)
    raw_labels = fcluster(z, t=1.0 - float(threshold), criterion="distance") - 1
    counts = np.bincount(raw_labels)
    out = raw_labels.copy()
    out[np.isin(out, np.where(counts == 1)[0])] = -1
    return out.astype(int)


def merge_representatives_from_corr(
    labels: np.ndarray,
    rep_corr: np.ndarray,
    threshold: float,
    *,
    method: str = LINKAGE_METHOD,
) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=int)
    rep_members, _ = representative_members_from_labels(labels_arr)
    rep_labels = redundancy_groups_from_corr(rep_corr, float(threshold), method=method)
    new_labels = np.full(labels_arr.shape, -2, dtype=int)
    next_group = 0
    for rep_cluster_id in sorted(set(rep_labels[rep_labels >= 0])):
        rep_indices = np.flatnonzero(rep_labels == rep_cluster_id)
        members = np.concatenate([rep_members[int(idx)] for idx in rep_indices])
        if members.size >= 2:
            new_labels[members] = next_group
            next_group += 1
        elif members.size == 1:
            new_labels[int(members[0])] = -1
    for rep_idx in np.flatnonzero(rep_labels == -1):
        members = rep_members[int(rep_idx)]
        if members.size >= 2:
            new_labels[members] = next_group
            next_group += 1
        elif members.size == 1:
            new_labels[int(members[0])] = -1
    new_labels[labels_arr == -2] = -2
    return new_labels


def corr_1d(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    if denom < eps:
        return float("nan")
    return float(np.dot(aa, bb) / denom)


def population_membership_from_labels(
    labels: np.ndarray,
    *,
    pooling_mode: str,
    medoid_channels: dict[int, int] | None = None,
) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=int)
    group_ids = sorted(set(labels_arr[labels_arr >= 0]))
    singletons = list(np.flatnonzero(labels_arr == -1))
    mem = np.zeros((len(group_ids) + len(singletons), labels_arr.size), dtype=np.float32)
    medoid_channels = {} if medoid_channels is None else {int(k): int(v) for k, v in medoid_channels.items()}
    for rep_idx, group_id in enumerate(group_ids):
        members = np.flatnonzero(labels_arr == int(group_id))
        if pooling_mode == "medoid":
            selected = int(medoid_channels.get(int(group_id), int(members[0])))
            mem[rep_idx, selected] = 1.0
        elif pooling_mode == "mean":
            mem[rep_idx, members] = 1.0 / float(members.size)
        else:
            raise ValueError(f"Unknown pooling_mode {pooling_mode!r}")
    for rep_idx, channel in enumerate(singletons, start=len(group_ids)):
        mem[rep_idx, int(channel)] = 1.0
    return mem


def cluster_membership_from_labels(labels: np.ndarray) -> np.ndarray:
    return population_membership_from_labels(labels, pooling_mode="mean")


def select_ccnorm_medoid_channels(labels: np.ndarray, unit_rows: list[dict[str, Any]]) -> dict[int, int]:
    labels_arr = np.asarray(labels, dtype=int)
    medoids: dict[int, int] = {}
    for group_id in sorted(set(labels_arr[labels_arr >= 0])):
        members = np.flatnonzero(labels_arr == int(group_id))
        scores = []
        for channel in members:
            ccnorm = float(unit_rows[int(channel)].get("ccnorm", np.nan)) if int(channel) < len(unit_rows) else np.nan
            scores.append((ccnorm if np.isfinite(ccnorm) else -np.inf, -int(channel), int(channel)))
        medoids[int(group_id)] = max(scores)[2]
    return medoids


def select_movie_medoid_channels(
    labels: np.ndarray,
    unit_rows: list[dict[str, Any]],
    movie_paths: list[tuple[str, Path]],
) -> tuple[dict[int, int], list[dict[str, Any]]]:
    """Choose one real channel per group using cached construction movies."""
    labels_arr = np.asarray(labels, dtype=int)
    group_ids = sorted(set(labels_arr[labels_arr >= 0]))
    stats: dict[tuple[int, int], dict[str, Any]] = {}
    for group_id in group_ids:
        for channel in np.flatnonzero(labels_arr == int(group_id)):
            ccnorm = float(unit_rows[int(channel)].get("ccnorm", np.nan)) if int(channel) < len(unit_rows) else np.nan
            stats[(int(group_id), int(channel))] = {
                "group_id": int(group_id),
                "channel": int(channel),
                "n_members": int(np.sum(labels_arr == int(group_id))),
                "ccnorm": ccnorm,
                "member_centroid_corrs": [],
            }

    for case, path in movie_paths:
        print(f"  select movie medoids: {case}", flush=True)
        movie = load_activation_movie(path)
        for group_id in group_ids:
            members = np.flatnonzero(labels_arr == int(group_id))
            if members.size < 2:
                continue
            centroid = np.nanmean(movie[:, members], axis=1)
            for channel in members:
                stats[(int(group_id), int(channel))]["member_centroid_corrs"].append(
                    corr_1d(movie[:, int(channel)], centroid)
                )
        del movie

    medoids: dict[int, int] = {}
    detail_rows: list[dict[str, Any]] = []
    for group_id in group_ids:
        members = np.flatnonzero(labels_arr == int(group_id))
        candidates: list[dict[str, Any]] = []
        for channel in members:
            row = stats[(int(group_id), int(channel))]
            corrs = np.asarray(row["member_centroid_corrs"], dtype=np.float32)
            finite = corrs[np.isfinite(corrs)]
            out = {
                "group_id": int(group_id),
                "channel": int(channel),
                "n_members": int(members.size),
                "n_cases": int(finite.size),
                "worst_member_centroid_corr": float(np.nanmin(finite)) if finite.size else np.nan,
                "median_member_centroid_corr": float(np.nanmedian(finite)) if finite.size else np.nan,
                "ccnorm": float(row["ccnorm"]),
            }
            candidates.append(out)

        def score(row: dict[str, Any]) -> tuple[float, float, float, int]:
            worst = float(row["worst_member_centroid_corr"])
            median = float(row["median_member_centroid_corr"])
            ccnorm = float(row["ccnorm"])
            return (
                worst if np.isfinite(worst) else -np.inf,
                median if np.isfinite(median) else -np.inf,
                ccnorm if np.isfinite(ccnorm) else -np.inf,
                -int(row["channel"]),
            )

        selected = max(candidates, key=score)
        medoids[int(group_id)] = int(selected["channel"])
        for row in candidates:
            out = dict(row)
            out["selected"] = bool(int(row["channel"]) == int(selected["channel"]))
            detail_rows.append(out)
    return medoids, detail_rows


def expand_population_to_full(
    reduced: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    y = np.asarray(reduced, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=int)
    group_ids = sorted(set(labels_arr[labels_arr >= 0]))
    singletons = list(np.flatnonzero(labels_arr == -1))
    expanded = np.zeros((y.shape[0], labels_arr.size, y.shape[2], y.shape[3]), dtype=np.float32)
    for rep_idx, group_id in enumerate(group_ids):
        members = np.flatnonzero(labels_arr == int(group_id))
        expanded[:, members] = y[:, rep_idx, None]
    for rep_idx, channel in enumerate(singletons, start=len(group_ids)):
        expanded[:, int(channel)] = y[:, rep_idx]
    return expanded


def audit_pooling_on_movies(
    version: str,
    labels: np.ndarray,
    membership: np.ndarray,
    movie_paths: list[tuple[str, Path]],
    *,
    pooling_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    valid = np.asarray(labels, dtype=int) != -2
    for case, path in movie_paths:
        print(f"  audit {pooling_mode}: {case}", flush=True)
        movie = load_activation_movie(path)
        reduced = np.einsum("rc,tchw->trhw", membership, movie, optimize=True).astype(np.float32, copy=False)
        expanded = expand_population_to_full(reduced, labels)
        full_valid = movie[:, valid]
        expanded_valid = expanded[:, valid]
        residual = expanded_valid - full_valid
        per_channel_corr = np.asarray(
            [corr_1d(full_valid[:, ch], expanded_valid[:, ch]) for ch in range(full_valid.shape[1])],
            dtype=np.float32,
        )
        rows.append(
            {
                "version": version,
                "case": case,
                "pooling_mode": pooling_mode,
                "n_time": int(movie.shape[0]),
                "n_channels": int(full_valid.shape[1]),
                "n_representatives": int(membership.shape[0]),
                "global_rate_corr": corr_1d(full_valid, expanded_valid),
                "rmse": float(np.sqrt(np.nanmean(residual * residual))),
                "nrmse_by_full_std": float(np.sqrt(np.nanmean(residual * residual)) / max(float(np.nanstd(full_valid)), 1e-8)),
                "min_channel_corr": float(np.nanmin(per_channel_corr)),
                "p05_channel_corr": float(np.nanpercentile(per_channel_corr, 5)),
                "median_channel_corr": float(np.nanmedian(per_channel_corr)),
            }
        )
        del movie, reduced, expanded, full_valid, expanded_valid, residual, per_channel_corr
    global_corrs = np.asarray([float(row["global_rate_corr"]) for row in rows], dtype=np.float32)
    return (
        {
            "version": version,
            "pooling_mode": pooling_mode,
            "n_cases": int(len(rows)),
            "n_representatives": int(membership.shape[0]),
            "mean_global_reconstruction_corr": float(np.nanmean(global_corrs)),
            "min_global_reconstruction_corr": float(np.nanmin(global_corrs)),
            "median_global_reconstruction_corr": float(np.nanmedian(global_corrs)),
            "mean_p05_channel_corr": float(np.nanmean([float(row["p05_channel_corr"]) for row in rows])),
            "min_p05_channel_corr": float(np.nanmin([float(row["p05_channel_corr"]) for row in rows])),
        },
        rows,
    )


def save_population_spec(
    labels: np.ndarray,
    unit_rows: list[dict[str, Any]],
    version: str,
    *,
    pooling_mode: str,
    medoid_channels: dict[int, int] | None = None,
    selection_metadata: dict[str, Any] | None = None,
) -> Path:
    labels_arr = np.asarray(labels, dtype=int)
    group_ids = sorted(set(labels_arr[labels_arr >= 0]))
    singletons = list(np.flatnonzero(labels_arr == -1))
    excluded = list(map(int, np.flatnonzero(labels_arr == -2)))
    medoid_channels = {} if medoid_channels is None else {int(k): int(v) for k, v in medoid_channels.items()}
    reps: list[dict[str, Any]] = []
    for rep_idx, group_id in enumerate(group_ids):
        members = list(map(int, np.flatnonzero(labels_arr == int(group_id))))
        selected = int(medoid_channels.get(int(group_id), members[0])) if pooling_mode == "medoid" else None
        selected_row = unit_rows[selected] if selected is not None and selected < len(unit_rows) else {}
        ccnorms = [float(unit_rows[ch].get("ccnorm", np.nan)) for ch in members if ch < len(unit_rows)]
        sessions = sorted(set(str(unit_rows[ch].get("session", "")) for ch in members if ch < len(unit_rows)))
        reps.append(
            {
                "rep_idx": int(rep_idx),
                "kind": "group",
                "group_label": int(group_id),
                "members": members,
                "n_members": int(len(members)),
                "pooling_mode": pooling_mode,
                "selected_channel": selected,
                "selected_session": str(selected_row.get("session", "")) if selected_row else None,
                "selected_source_unit_index": (
                    int(selected_row["source_unit_index"])
                    if selected_row and "source_unit_index" in selected_row
                    else None
                ),
                "selected_ccnorm": float(selected_row.get("ccnorm", np.nan)) if selected_row else np.nan,
                "sessions": sessions,
                "mean_ccnorm": float(np.nanmean(ccnorms)) if ccnorms else np.nan,
            }
        )
    for rep_idx, channel in enumerate(singletons, start=len(group_ids)):
        row = unit_rows[int(channel)] if int(channel) < len(unit_rows) else {}
        reps.append(
            {
                "rep_idx": int(rep_idx),
                "kind": "singleton",
                "group_label": -1,
                "members": [int(channel)],
                "n_members": 1,
                "pooling_mode": "singleton",
                "selected_channel": int(channel),
                "selected_session": str(row.get("session", "")),
                "selected_source_unit_index": int(row["source_unit_index"]) if "source_unit_index" in row else None,
                "selected_ccnorm": float(row.get("ccnorm", np.nan)),
                "sessions": [str(row.get("session", ""))],
                "mean_ccnorm": float(row.get("ccnorm", np.nan)),
            }
        )
    spec = {
        "version": version,
        "pooling_mode": pooling_mode,
        "selection_metadata": dict(selection_metadata or {}),
        "n_input_channels": int(labels_arr.size),
        "n_representatives": int(len(reps)),
        "n_groups": int(len(group_ids)),
        "n_singletons": int(len(singletons)),
        "n_excluded": int(len(excluded)),
        "excluded_channels": excluded,
        "representatives": reps,
    }
    slug = _safe_slug(version)
    json_path = OUT_DIR / f"population_spec_{slug}.json"
    npz_path = OUT_DIR / f"population_spec_{slug}.npz"
    membership = population_membership_from_labels(labels_arr, pooling_mode=pooling_mode, medoid_channels=medoid_channels)
    cluster_membership = cluster_membership_from_labels(labels_arr)
    _save_json(json_path, spec)
    np.savez_compressed(
        npz_path,
        labels=labels_arr,
        membership=membership,
        cluster_membership=cluster_membership,
        pooling_mode=np.asarray(pooling_mode),
    )
    return json_path


def plot_frontier(rows: list[dict[str, Any]], path: Path) -> None:
    thresholds = np.asarray([float(row["threshold"]) for row in rows], dtype=np.float32)
    n_reps = np.asarray([int(row["n_representatives"]) for row in rows], dtype=np.int32)
    largest = np.asarray([int(row["largest_group"]) for row in rows], dtype=np.int32)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    axes[0].plot(thresholds, n_reps, "-o", lw=1.6, ms=4)
    axes[0].axhline(TARGET_REPRESENTATIVES, color="0.25", ls="--", lw=1.0)
    axes[0].set_xlabel("merge threshold")
    axes[0].set_ylabel("representatives")
    axes[0].set_title("Post-hoc compression")
    axes[1].plot(thresholds, largest, "-o", lw=1.6, ms=4, color="tab:orange")
    axes[1].set_xlabel("merge threshold")
    axes[1].set_ylabel("largest original-channel cluster")
    axes[1].set_title("Largest cluster")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    labels = np.load(args.labels)
    unit_rows = load_unit_rows(args.unit_table)
    movie_paths = movie_paths_from_reconstruction_csv(args.reconstruction_csv, max_cases=args.max_cases)
    print(f"Base: {population_counts(labels, BASE_VERSION)}", flush=True)
    print("Cases:", ", ".join(case for case, _path in movie_paths), flush=True)

    rep_corrs: list[np.ndarray] = []
    repcorr_rows: list[dict[str, Any]] = []
    for case, path in movie_paths:
        print(f"Computing representative corr: {case}", flush=True)
        movie = load_activation_movie(path)
        rep_fp = representative_fingerprints_from_movie(movie, labels, normalization=args.normalization)
        rep_corr = correlation_from_fingerprints(rep_fp)
        rep_corrs.append(rep_corr)
        offdiag = rep_corr[np.triu_indices(rep_corr.shape[0], k=1)]
        repcorr_rows.append(
            {
                "case": case,
                "n_representatives": int(rep_corr.shape[0]),
                "offdiag_corr_p95": float(np.nanpercentile(offdiag, 95)),
                "offdiag_corr_p99": float(np.nanpercentile(offdiag, 99)),
                "offdiag_corr_max": float(np.nanmax(offdiag)),
            }
        )
        del movie, rep_fp, rep_corr

    combined_corr = combine_similarity_matrices(rep_corrs, mode=args.similarity_mode)
    slug_base = _safe_slug(
        f"{BASE_VERSION}_cachedMedoid_posthoc_{args.similarity_mode}_{args.linkage}"
        f"_target{args.target}_cases{len(movie_paths)}"
    )
    np.savez_compressed(
        OUT_DIR / f"cached_medoid_compression_representative_corr_{slug_base}.npz",
        rep_corr=combined_corr.astype(np.float32),
        thresholds=np.asarray(args.thresholds, dtype=np.float32),
        case_labels=np.asarray([case for case, _path in movie_paths], dtype=str),
    )
    _write_csv(OUT_DIR / f"cached_medoid_compression_representative_corr_inputs_{slug_base}.csv", repcorr_rows)

    frontier_rows: list[dict[str, Any]] = []
    labels_by_version: dict[str, np.ndarray] = {}
    for threshold in args.thresholds:
        version = f"{BASE_VERSION}_medoidPosthoc{args.similarity_mode}Rep{args.linkage}{float(threshold):.2f}".replace("0.", "0p")
        compressed_labels = merge_representatives_from_corr(
            labels,
            combined_corr,
            threshold=float(threshold),
            method=args.linkage,
        )
        row = population_counts(compressed_labels, version)
        row.update(
            {
                "base_version": BASE_VERSION,
                "threshold": float(threshold),
                "target_representatives": int(args.target),
                "distance_to_target": int(abs(int(row["n_representatives"]) - int(args.target))),
                "similarity_mode": args.similarity_mode,
                "linkage": args.linkage,
            }
        )
        frontier_rows.append(row)
        labels_by_version[version] = compressed_labels

    _write_csv(OUT_DIR / f"cached_medoid_compression_frontier_summary_{slug_base}.csv", frontier_rows)
    plot_frontier(frontier_rows, OUT_DIR / f"cached_medoid_compression_frontier_summary_{slug_base}.png")

    at_or_above = [row for row in frontier_rows if int(row["n_representatives"]) >= int(args.target)]
    selected_rows = []
    if at_or_above:
        selected_rows.append(min(at_or_above, key=lambda row: (int(row["distance_to_target"]), int(row["n_representatives"]))))
    selected_rows.append(min(frontier_rows, key=lambda row: (int(row["distance_to_target"]), abs(int(row["n_representatives"]) - int(args.target)))))
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in selected_rows:
        if row["version"] not in seen:
            unique.append(row)
            seen.add(row["version"])

    selected_summary_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    for row in unique:
        version = str(row["version"])
        candidate_labels = labels_by_version[version]
        np.save(OUT_DIR / f"cached_medoid_compression_candidate_labels_{_safe_slug(version)}.npy", candidate_labels)
        medoids, medoid_detail_rows = select_movie_medoid_channels(candidate_labels, unit_rows, movie_paths)
        medoid_selection_path = OUT_DIR / f"cached_medoid_selection_{_safe_slug(version)}.csv"
        _write_csv(medoid_selection_path, medoid_detail_rows)
        metadata = {
            "base_version": BASE_VERSION,
            "target_representatives": int(args.target),
            "threshold": float(row["threshold"]),
            "similarity_mode": args.similarity_mode,
            "linkage": args.linkage,
            "medoid_selection": "max worst member-centroid corr across cached construction movies",
            "medoid_selection_table": str(medoid_selection_path),
            "cases": [case for case, _path in movie_paths],
        }
        mean_version = f"{version}_mean"
        medoid_version = f"{version}_movieMedoid"
        save_population_spec(candidate_labels, unit_rows, mean_version, pooling_mode="mean", selection_metadata=metadata)
        save_population_spec(
            candidate_labels,
            unit_rows,
            medoid_version,
            pooling_mode="medoid",
            medoid_channels=medoids,
            selection_metadata=metadata,
        )
        mean_membership = population_membership_from_labels(candidate_labels, pooling_mode="mean")
        medoid_membership = population_membership_from_labels(
            candidate_labels,
            pooling_mode="medoid",
            medoid_channels=medoids,
        )
        mean_summary, mean_rows = audit_pooling_on_movies(
            mean_version,
            candidate_labels,
            mean_membership,
            movie_paths,
            pooling_mode="mean",
        )
        medoid_summary, medoid_rows = audit_pooling_on_movies(
            medoid_version,
            candidate_labels,
            medoid_membership,
            movie_paths,
            pooling_mode="medoid",
        )
        selected = dict(row)
        selected.update(
            {
                "mean_version": mean_version,
                "medoid_version": medoid_version,
                "mean_global_reconstruction_corr": mean_summary["mean_global_reconstruction_corr"],
                "medoid_global_reconstruction_corr": medoid_summary["mean_global_reconstruction_corr"],
                "mean_min_global_reconstruction_corr": mean_summary["min_global_reconstruction_corr"],
                "medoid_min_global_reconstruction_corr": medoid_summary["min_global_reconstruction_corr"],
                "mean_mean_p05_channel_corr": mean_summary["mean_p05_channel_corr"],
                "medoid_mean_p05_channel_corr": medoid_summary["mean_p05_channel_corr"],
            }
        )
        selected_summary_rows.append(selected)
        reconstruction_rows.extend(mean_rows)
        reconstruction_rows.extend(medoid_rows)

    _write_csv(OUT_DIR / f"cached_medoid_compression_selected_candidates_{slug_base}.csv", selected_summary_rows)
    _write_csv(OUT_DIR / f"cached_medoid_compression_selected_reconstruction_{slug_base}.csv", reconstruction_rows)
    print("Selected candidates:")
    for row in selected_summary_rows:
        print(row)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=BASE_LABELS_PATH)
    parser.add_argument("--unit-table", type=Path, default=UNIT_TABLE_CSV)
    parser.add_argument("--reconstruction-csv", type=Path, default=CONSTRUCTION_RECON_CSV)
    parser.add_argument("--target", type=int, default=TARGET_REPRESENTATIVES)
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(THRESHOLDS))
    parser.add_argument("--similarity-mode", choices=["min", "mean"], default=SIMILARITY_MODE)
    parser.add_argument("--linkage", choices=["complete", "average", "weighted", "single"], default=LINKAGE_METHOD)
    parser.add_argument("--normalization", choices=["none", "center", "zscore"], default=FINGERPRINT_NORMALIZATION)
    parser.add_argument("--max-cases", type=int, default=0, help="0 means use all construction cases.")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    if args.max_cases <= 0:
        args.max_cases = None
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    run(args)


if __name__ == "__main__":
    main()
