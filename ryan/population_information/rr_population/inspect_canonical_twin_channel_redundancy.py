#!/usr/bin/env python3
# %% [markdown]
# # Canonical V1 Twin Channel Redundancy: Activation Fingerprints
#
# Interactive first-pass inspection for the redundancy-resolved V1 twin project.
#
# This script deliberately uses percent cells so it can be opened as a Jupyter/
# VS Code notebook while remaining easy to diff. The core object is the full
# spatial response movie
#
#     T x C x H x W
#
# from the canonical shared twin readout on a real BackImage fixation trace.
# Channel fingerprints are then built as
#
#     C x (T * H * W)
#
# so the intentional convolutional spatial samples are preserved as samples of
# each channel, while clustering/merging candidates remain channels.

# %% Imports and configuration
from __future__ import annotations

import csv
import json
import os
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is for notebook display only.
    pd = None


#ROOT = Path(__file__).resolve().parents[1]
ROOT = Path('/home/ryanress/v1-fovea/VisionCore/')
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


BACKIMAGE_RESULTS_PATH = ROOT / "declan" / "backimage_fixation_results.pkl"

# Pinned canonical digital twin — the SAME behavior-conditioned twin used by
# paper/fig3 (see paper/fig3/_fig3_data.py). We pin an explicit checkpoint +
# in-repo config rather than scan_checkpoints/model_index so readout sizes stay
# matched to the checkpoint. See VisionCore/TWIN_IMPROVEMENTS.md for why the old
# get_model_and_dataset_configs() path drifted and broke.
CANONICAL_TWIN_CHECKPOINT = (
    "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/digital_twin_120/"
    "2026-03-31_11-33-32_learned_resnet_concat_convgru_gaussian/"
    "learned_resnet_concat_convgru_gaussian_lr1e-3_wd1e-5_cls1.0_bs256_ga4/"
    "epoch=374-val_bps_overall=0.6395.ckpt"
)
CANONICAL_TWIN_CONFIG = ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"

# Per-unit CCNORM (twin-vs-data noise-corrected correlation) comes from the fig3
# inference cache, which is computed against THIS twin — so it is aligned to the
# readout cids. This replaces the old multi-GB scripts/mcfarland_outputs*.pkl
# dependency, which indexed a different unit set. See TWIN_IMPROVEMENTS.md item 4.
FIG3_CCNORM_CACHE = ROOT / "outputs" / "cache" / "fig3_digitaltwin.pkl"

OUT_DIR = ROOT / "outputs" / "redundancy_resolved_v1_twin" / "step1_activation_fingerprints"
MCFARLAND_OUTPUT_CANDIDATES = (
    ROOT / "scripts" / "mcfarland_outputs_mono.pkl",
    ROOT / "scripts" / "mcfarland_outputs.pkl",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_MODE = "standard"

N_LAGS = 32
OUT_SIZE = (151, 151)
SCALE_FACTOR = 1.0
PPD = 37.50476617
BATCH_SIZE = 32

# Start small enough to remain pleasant in a notebook. Increase once the first
# response movie looks healthy.
IMAGE_KEY: str | None = None
IMAGE_RANK = 0
TRACE_INDEX = 0
MAX_FRAMES: int | None = 160
CENTER_EYE_TRACE = False

LOAD_CACHE_IF_AVAILABLE = True
SAVE_ACTIVATION_CACHE = True
SAVE_FIGURES = True

RANDOM_SEED = 7
N_RANDOM_CHANNELS = 12
N_TOP_PAIRS = 24
FINGERPRINT_NORMALIZATION = "zscore"  # one of: "none", "center", "zscore"
RUN_TSNE = True
RUN_BROAD_HELDOUT_VALIDATION = False  # slow 3x2x2 comparison; cached outputs already exist for routine runs
REDUNDANCY_THRESHOLDS = [0.6, 0.75, 0.9]
REDUNDANCY_LINKAGE_METHOD = "complete"
REDUNDANCY_LINKAGE_METHODS = ["complete", "average"]
FIRST_CLUSTER_THRESHOLD = 0.65  # first-stage V1-RR construction threshold
EMBEDDING_AUDIT_N_GROUPS = 6
EMBEDDING_AUDIT_MIN_GROUP_SIZE = 3
EMBEDDING_AUDIT_MAX_TRACE_MEMBERS = 5
EMBEDDING_AUDIT_MAX_MAP_MEMBERS = 2
N_FINGERPRINT_CASES = 20          # training fingerprints — more cases = more robust corr matrix
N_HELDOUT_FINGERPRINT_CASES = 8   # held-out cases: used for movie-based split + fingerprint diagnostics
HELDOUT_START_RANK = N_FINGERPRINT_CASES
SILENCE_RATE_THRESHOLD = 0.02     # channels with mean activation < this across all fingerprint cases are excluded
REDUNDANCY_VALIDATION_SPECS = [
    {
        "name": "V1-RR124_raw_complete_0p60",
        "similarity": "raw_fingerprint_corr",
        "method": "complete",
        "threshold": 0.60,
    },
    {
        "name": "V1-RR236_raw_complete_0p75",
        "similarity": "raw_fingerprint_corr",
        "method": "complete",
        "threshold": 0.75,
    },
    {
        "name": "V1-RR224_pcacos_complete_0p60",
        "similarity": "pca_cosine",
        "method": "complete",
        "threshold": 0.60,
    },
    {
        "name": "V1-RR372_pcacos_complete_0p75",
        "similarity": "pca_cosine",
        "method": "complete",
        "threshold": 0.75,
    },
]
N_VALIDATION_GROUPS_TO_PLOT = 6
N_VALIDATION_MAP_MEMBERS = 4
VALIDATION_TRACE_FRAMES = 120
TRACE_PLOT_MODES = ("center_pixel", "spatial_mean")
TRACE_PLOT_ZSCORE_BY_MODE = {"center_pixel": False, "spatial_mean": True}
ACTIVATION_MAP_CMAP = "gray"
MAP_FRAME_MODE = "peak"  # one of: "peak", "final", "index"
MAP_FRAME_INDEX: int | None = None
N_SINGLETON_MAPS_TO_PLOT = 12
N_SHARP_MAPS_TO_PLOT = 12
N_LABELED_SHARP_SINGLETONS = 24

# Post-hoc compression frontier. This does not replace the conservative RR by
# default; it asks whether the already-vetted representatives can be merged
# further while keeping multi-stimulus response quality acceptable.
RUN_POSTHOC_COMPRESSION_FRONTIER = True
COMPRESSION_TARGET_REPRESENTATIVES = 100
COMPRESSION_THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]
COMPRESSION_LINKAGE_METHOD = "complete"
COMPRESSION_SIMILARITY_MODE = "min"  # one of: "min", "mean"
COMPRESSION_PROMOTION_MAX_GROUPS_BELOW_0P60 = 0
COMPRESSION_PROMOTION_MIN_RECONSTRUCTION_CORR = 0.75
RUN_COMPRESSION_CANDIDATE_PLOTS = True
SAVE_COMPRESSION_POPULATION_SPECS = False

# Mean-pooled specs are ensemble centroids. Medoid specs keep one real modeled
# unit per cluster, which preserves sharper/cell-like activation maps.
SAVE_MEDOID_POPULATION_SPECS = True
RUN_MEDOID_POPULATION_AUDITS = True
MEDOID_SELECTION_METRIC_SPACE = "movie_t_h_w"

# FixRSVP audit. This is a stimulus-diversity check for the already-built V1-RR
# labels, not another rule for creating/splitting clusters.
RUN_FIXRSVP_AUDIT = True
FIXRSVP_STIMULUS_TYPE = "fixrsvp"
FIXRSVP_FRAMES_PER_IMAGE = 6
FIXRSVP_NUM_FRAMES = 192
FIXRSVP_FRAME: int | None = None

# Multi-stimulus candidate. "min" is conservative: pairwise similarity must be
# strong in both stimulus-specific matrices before a merge is initialized.
MULTISTIM_SIMILARITY_MODE = "min"  # one of: "min", "mean"
MULTISTIM_SPLIT_CASE_FRACTION = 0.0        # 0.0 = split on any construction-stimulus failure
MULTISTIM_PAIRWISE_CASE_FRACTION = 0.0
MULTISTIM_FINAL_FORCE_SPLIT_CENTROID_THRESHOLD = 0.75
SAVE_MULTISTIM_POPULATION_SPEC = True

OUT_DIR.mkdir(parents=True, exist_ok=True)


# %% Lightweight data containers
@dataclass(frozen=True)
class BackimageCase:
    image_key: str
    entry: dict[str, Any]
    image: np.ndarray
    eyepos: np.ndarray
    trace_index: int
    centered_eye_trace: bool


@dataclass(frozen=True)
class ModelBundle:
    model: torch.nn.Module
    readout: torch.nn.Module
    outputs: list[dict[str, Any]]
    unit_rows: list[dict[str, Any]]
    device: str
    dataset_config: dict[str, Any] | None = None


# %% General helpers
def _safe_slug(value: object, max_len: int = 96) -> str:
    text = str(value)
    text = Path(text).stem if "/" in text else text
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    slug = "_".join(part for part in slug.split("_") if part)
    return slug[:max_len] or "unnamed"


def _as_table(rows: list[dict[str, Any]]):
    if pd is not None:
        return pd.DataFrame(rows)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _channel_label(channel: int, unit_rows: list[dict[str, Any]] | None = None) -> str:
    if unit_rows is None or channel >= len(unit_rows):
        return f"ch {channel}"
    row = unit_rows[channel]
    sess = str(row.get("session", "session?"))
    cid = row.get("source_unit_index", "?")
    return f"ch {channel}\n{sess}:{cid}"


def _zscore_1d(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - float(np.nanmean(x))) / (float(np.nanstd(x)) + eps)


# %% BackImage loading helpers
def _to_gray_float32(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3]
        arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
    elif arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = arr[:3]
        arr = 0.2989 * arr[0] + 0.5870 * arr[1] + 0.1140 * arr[2]
    arr = np.squeeze(arr).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image after conversion, got {arr.shape}")
    if np.nanmax(arr) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0.0, 255.0).astype(np.float32)


def _image_search_dirs() -> list[Path]:
    dirs: list[Path] = []

    try:
        import importlib

        support = importlib.import_module("DataYatesV1.exp.support")
        get_backimage_directory = getattr(support, "get_backimage_directory", None)
        if callable(get_backimage_directory):
            dirs.append(Path(get_backimage_directory()))
    except Exception:
        pass

    fallback = Path("/home/declan/DataYatesV1/DataYatesV1/exp/SupportData/Backgrounds")
    dirs.extend([fallback, ROOT / "declan", ROOT / "data", ROOT / "datasets"])

    for raw in os.environ.get("VC_IMAGE_DIRS", "").split(":"):
        raw = raw.strip()
        if raw:
            dirs.append(Path(raw))

    unique: list[Path] = []
    seen: set[str] = set()
    for path in dirs:
        key = str(path)
        if key not in seen and path.exists():
            seen.add(key)
            unique.append(path)
    return unique


def find_image_on_disk(filename: str, search_dirs: Iterable[Path] | None = None) -> Path | None:
    basename = Path(filename).name
    if not basename:
        return None
    for base in search_dirs or _image_search_dirs():
        if not base.is_dir():
            continue
        for root, _dirs, files in os.walk(base):
            if basename in files:
                return Path(root) / basename
    return None


def load_image_for_entry(entry: dict[str, Any], image_key: str) -> np.ndarray:
    if entry.get("image") is not None:
        return _to_gray_float32(entry["image"])

    for key in ("image_path", "path", "filepath", "file_path", "img_path"):
        raw_path = entry.get(key)
        if isinstance(raw_path, (list, tuple)):
            raw_path = raw_path[0] if raw_path else None
        if isinstance(raw_path, bytes):
            raw_path = raw_path.decode("utf-8", errors="ignore")
        if isinstance(raw_path, str) and Path(raw_path).exists():
            return _read_image_file(Path(raw_path))

    found = find_image_on_disk(image_key)
    if found is not None:
        return _read_image_file(found)

    raise FileNotFoundError(
        f"Could not resolve image for BackImage key {image_key!r}. "
        "Try setting VC_IMAGE_DIRS to the directory containing the background images."
    )


def _read_image_file(path: Path) -> np.ndarray:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return _to_gray_float32(np.asarray(image))
    except Exception:
        import imageio.v2 as imageio

        return _to_gray_float32(imageio.imread(path))


def normalize_eye_trace(raw: object, center: bool = False) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size % 2:
            arr = arr[:-1]
        arr = arr.reshape(-1, 2)
    elif arr.ndim == 2:
        if arr.shape[1] == 2:
            pass
        elif arr.shape[0] == 2:
            arr = arr.T
        elif arr.shape[1] > 2:
            arr = arr[:, :2]
        else:
            raise ValueError(f"Could not interpret eye trace shape {arr.shape}")
    else:
        flat = arr.reshape(-1)
        if flat.size % 2:
            flat = flat[:-1]
        arr = flat.reshape(-1, 2)

    finite_rows = np.isfinite(arr).all(axis=1)
    if not finite_rows.all():
        first_bad = int(np.where(~finite_rows)[0][0])
        arr = arr[:first_bad]
    if center and arr.size:
        arr = arr - np.nanmean(arr, axis=0, keepdims=True)
    return arr.astype(np.float32)


def split_entry_eye_traces(entry: dict[str, Any], center: bool = False) -> tuple[list[np.ndarray], str]:
    raw = entry.get("eyepos", [])
    n_trials_raw = entry.get("n_trials", 0)
    try:
        n_trials = int(n_trials_raw)
    except Exception:
        n_trials = 0

    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        traces = [normalize_eye_trace(arr[i, :, :2], center=center) for i in range(arr.shape[0])]
        return traces, "array_Tx2_per_trial"

    if arr.ndim == 2 and arr.shape[1] >= 2:
        arr = arr[:, :2].astype(np.float32)
        if n_trials > 1 and arr.shape[0] >= n_trials:
            frames_per_trial = int(arr.shape[0] // n_trials)
            usable = int(frames_per_trial * n_trials)
            if frames_per_trial >= 1 and usable > 0:
                traces_3d = arr[:usable].reshape(n_trials, frames_per_trial, 2)
                traces = [normalize_eye_trace(traces_3d[i], center=center) for i in range(n_trials)]
                return traces, "concatenated_samples_split_by_n_trials"
        return [normalize_eye_trace(arr, center=center)], "single_trace_array"

    if isinstance(raw, (list, tuple)):
        traces = [normalize_eye_trace(trace, center=center) for trace in raw]
        return traces, "list_of_traces"

    return [normalize_eye_trace(raw, center=center)], "single_trace_fallback"


def load_backimage_results(path: Path = BACKIMAGE_RESULTS_PATH) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected BackImage results dict at {path}, got {type(payload).__name__}")
    return payload


def summarize_backimage_results(results: dict[str, Any], n_trace_lengths: int = 20):
    rows: list[dict[str, Any]] = []
    for image_key, entry in results.items():
        eyepos_raw = entry.get("eyepos", []) if isinstance(entry, dict) else []
        traces, split_mode = split_entry_eye_traces(entry) if isinstance(entry, dict) else ([], "not_a_dict")
        lengths: list[int] = []
        for trace in traces[:n_trace_lengths]:
            try:
                lengths.append(int(normalize_eye_trace(trace).shape[0]))
            except Exception:
                pass
        eyepos_arr = np.asarray(eyepos_raw)
        rows.append(
            {
                "image_key": image_key,
                "n_trials_field": entry.get("n_trials", np.nan) if isinstance(entry, dict) else np.nan,
                "n_eye_traces": len(traces),
                "n_eye_samples_raw": int(eyepos_arr.shape[0]) if eyepos_arr.ndim >= 1 else 0,
                "median_trace_frames": float(np.median(lengths)) if lengths else np.nan,
                "min_trace_frames_sample": int(np.min(lengths)) if lengths else np.nan,
                "split_mode": split_mode,
                "has_cached_image": bool(isinstance(entry, dict) and entry.get("image") is not None),
            }
        )
    rows.sort(key=lambda row: (row["n_eye_traces"], row["n_trials_field"]), reverse=True)
    return _as_table(rows)


def select_backimage_case(
    results: dict[str, Any],
    image_key: str | None = IMAGE_KEY,
    image_rank: int = IMAGE_RANK,
    trace_index: int = TRACE_INDEX,
    max_frames: int | None = MAX_FRAMES,
    center_eye_trace: bool = CENTER_EYE_TRACE,
) -> BackimageCase:
    if image_key is None:
        def _rank_entry(item: tuple[str, Any]) -> tuple[int, int]:
            _key, candidate = item
            if not isinstance(candidate, dict):
                return (0, 0)
            traces, _split_mode = split_entry_eye_traces(candidate)
            try:
                n_trials = int(candidate.get("n_trials", 0))
            except Exception:
                n_trials = 0
            return (len(traces), n_trials)

        ordered = sorted(
            results.items(),
            key=_rank_entry,
            reverse=True,
        )
        image_key, entry = ordered[int(image_rank)]
    else:
        entry = results[image_key]

    if not isinstance(entry, dict):
        raise TypeError(f"BackImage entry {image_key!r} is not a dict")
    traces, split_mode = split_entry_eye_traces(entry, center=center_eye_trace)
    if not traces:
        raise ValueError(f"BackImage entry {image_key!r} has no eyepos traces")

    trace_index = int(np.clip(trace_index, 0, len(traces) - 1))
    eyepos = normalize_eye_trace(traces[trace_index], center=False)
    if max_frames is not None:
        eyepos = eyepos[: int(max_frames)]
    if eyepos.shape[0] < 2:
        raise ValueError(
            f"Selected trace is too short after cleanup: {eyepos.shape}. "
            f"Entry split mode was {split_mode!r}; try a different TRACE_INDEX or inspect this entry."
        )

    image = load_image_for_entry(entry, str(image_key))
    return BackimageCase(
        image_key=str(image_key),
        entry=entry,
        image=image,
        eyepos=eyepos,
        trace_index=trace_index,
        centered_eye_trace=center_eye_trace,
    )


def plot_backimage_case(case: BackimageCase, max_points: int = 2000):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].imshow(case.image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("BackImage")
    axes[0].axis("off")

    eye = case.eyepos[:max_points]
    axes[1].plot(eye[:, 0], label="x", linewidth=1.2)
    axes[1].plot(eye[:, 1], label="y", linewidth=1.2)
    axes[1].set_title("Eye Trace")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("deg")
    axes[1].legend(frameon=False)

    axes[2].plot(eye[:, 0], eye[:, 1], linewidth=1.0)
    axes[2].scatter(eye[0, 0], eye[0, 1], s=28, label="start")
    axes[2].scatter(eye[-1, 0], eye[-1, 1], s=28, label="end")
    axes[2].set_title("Eye Path")
    axes[2].set_xlabel("x deg")
    axes[2].set_ylabel("y deg")
    axes[2].axis("equal")
    axes[2].legend(frameon=False)
    return fig


# %% Model/readout helpers
def _load_pickle_or_dill(path: Path) -> Any:
    try:
        import dill

        with path.open("rb") as handle:
            return dill.load(handle)
    except ImportError:
        with path.open("rb") as handle:
            return pickle.load(handle)


def load_mcfarland_outputs(candidates: Iterable[Path] = MCFARLAND_OUTPUT_CANDIDATES) -> list[dict[str, Any]]:
    for path in candidates:
        if path.exists():
            outputs = _load_pickle_or_dill(path)
            if not isinstance(outputs, list):
                raise TypeError(f"Expected list in {path}, got {type(outputs).__name__}")
            return outputs
    candidates_s = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find any mcfarland outputs file among: {candidates_s}")


def build_readout_unit_rows(
    model: torch.nn.Module,
    outputs: list[dict[str, Any]],
    ccnorm_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    sessions = [str(output.get("sess", "")) for output in outputs]
    model_names = [str(name) for name in getattr(model, "names", [])]
    rows: list[dict[str, Any]] = []

    for model_readout_index, session in enumerate(model_names):
        if session not in sessions:
            continue
        output_index = sessions.index(session)
        output = outputs[output_index]
        ccnorm_payload = output.get("ccnorm", {})
        ccnorm = np.asarray(ccnorm_payload.get("ccnorm", []), dtype=np.float32)
        if ccnorm.size == 0:
            continue
        source_unit_indices = np.where(ccnorm > float(ccnorm_threshold))[0]
        for source_unit_index in source_unit_indices:
            rows.append(
                {
                    "channel": len(rows),
                    "session": session,
                    "source_unit_index": int(source_unit_index),
                    "ccnorm": float(ccnorm[source_unit_index]),
                    "model_readout_index": int(model_readout_index),
                    "mcfarland_output_index": int(output_index),
                }
            )
    return rows


def load_ccnorm_outputs_from_fig3_cache(
    model: torch.nn.Module,
    cache_path: Path = FIG3_CCNORM_CACHE,
) -> list[dict[str, Any]]:
    """Build per-session CCNORM ``outputs`` from the fig3 inference cache.

    Produces the same list-of-dicts shape the readout builders consume
    (``{"sess": name, "ccnorm": {"ccnorm": array}}``), but sourced from the
    fig3 cache instead of scripts/mcfarland_outputs*.pkl. The cache stores
    ``ccnorm`` over a spike-thresholded subset of each session's units, indexed
    by ``neuron_mask`` into the readout's cids; we scatter those values back into
    a full readout-length array (unmeasured units -> NaN, excluded by the
    ccnorm>threshold selection downstream).
    """
    import dill

    if not cache_path.exists():
        raise FileNotFoundError(
            f"fig3 inference cache not found at {cache_path}. Run "
            "`uv run python paper/fig3/generate_figure3.py` (or load_fig3_data) first."
        )
    with cache_path.open("rb") as handle:
        session_results = dill.load(handle)

    names = list(getattr(model, "names", []))
    outputs: list[dict[str, Any]] = []
    for entry in session_results:
        session = entry.get("session")
        if session not in names:
            continue
        dataset_idx = names.index(session)
        n_units = int(model.model.readouts[dataset_idx].bias.shape[0])
        ccnorm = np.asarray(entry.get("ccnorm", []), dtype=np.float32)
        neuron_mask = np.asarray(entry.get("neuron_mask", []), dtype=np.int64)
        full = np.full(n_units, np.nan, dtype=np.float32)
        if (
            neuron_mask.size
            and ccnorm.size
            and len(neuron_mask) == len(ccnorm)
            and int(neuron_mask.max()) < n_units
        ):
            full[neuron_mask] = ccnorm
        else:
            raise ValueError(
                f"fig3 cache CCNORM for {session!r} does not map into the twin "
                f"readout ({len(ccnorm)} values, mask max "
                f"{int(neuron_mask.max()) if neuron_mask.size else 'n/a'}, "
                f"readout n_units {n_units})."
            )
        outputs.append({"sess": session, "ccnorm": {"ccnorm": full}})
    if not outputs:
        raise ValueError(
            "No fig3 cache sessions matched the twin's dataset names; cannot "
            "build CCNORM outputs."
        )
    return outputs


def load_model_bundle(device: str = DEVICE, mode: str = MODEL_MODE) -> ModelBundle:
    from spatial_info import get_spatial_readout
    from eval.eval_stack_multidataset import load_model
    from models.config_loader import load_dataset_configs

    # Pin the fig3 canonical twin explicitly (see CANONICAL_TWIN_CHECKPOINT).
    model, _model_info = load_model(checkpoint_path=CANONICAL_TWIN_CHECKPOINT, device=device)
    model.model.eval()
    if hasattr(model.model, "convnet"):
        model.model.convnet.use_checkpointing = False
    model = model.to(device).eval()

    # Behavior transforms (eye velocity + eye position) are session-independent;
    # keep one config around so we can rebuild the twin's behavior input.
    dataset_configs = load_dataset_configs(str(CANONICAL_TWIN_CONFIG))

    outputs = load_ccnorm_outputs_from_fig3_cache(model)
    readout = get_spatial_readout(model, outputs).to(device).eval()
    # build_readout_unit_rows iterates model.names in the same order and applies
    # the same ccnorm>0.5 selection get_spatial_readout uses, so channel indices
    # line up with the concatenated population readout.
    unit_rows = build_readout_unit_rows(model, outputs, ccnorm_threshold=0.5)
    return ModelBundle(
        model=model,
        readout=readout,
        outputs=outputs,
        unit_rows=unit_rows,
        device=device,
        dataset_config=dataset_configs[0],
    )


def build_behavior_from_eyepos(eyepos: np.ndarray, dataset_config: dict[str, Any]) -> torch.Tensor:
    """Reproduce the twin's behavior input from an eye trace.

    The concat twin was trained on a 42-dim behavior vector derived entirely
    from eye movement: eye velocity through an acausal raised-cosine temporal
    basis (+ splitrelu) plus raw eye position. We rebuild it by running the
    exact training transforms (the ``expose_as: behavior`` entries in the
    dataset config) on the same eye trace this analysis uses to shift the
    retinal stimulus, so eye movement drives both the retinal shift and the
    modulation — as it did during training.

    Returns a (T, behavior_dim) tensor aligned frame-for-frame with ``eyepos``.
    """
    from models.data.transforms import make_pipeline

    eye = torch.from_numpy(np.asarray(eyepos, dtype=np.float32))
    parts: list[torch.Tensor] = []
    for spec in dataset_config.get("transforms", {}).values():
        if spec.get("expose_as") != "behavior":
            continue
        pipeline = make_pipeline(spec.get("ops", []), dataset_config)
        parts.append(pipeline(eye).float())
    if not parts:
        raise ValueError(
            "No 'behavior' transforms found in dataset_config; cannot build "
            "behavior input for the concat twin."
        )
    return torch.cat(parts, dim=-1)


# %% Activation movie construction and caching
def activation_cache_path(case: BackimageCase) -> Path:
    slug = _safe_slug(case.image_key)
    frames = case.eyepos.shape[0]
    centered = "centered" if case.centered_eye_trace else "stored"
    name = (
        f"activation_movie_{slug}_trace{case.trace_index:03d}_{centered}"
        f"_frames{frames}_lag{N_LAGS}_out{OUT_SIZE[0]}x{OUT_SIZE[1]}_scale{SCALE_FACTOR:g}.npz"
    )
    return OUT_DIR / name


def build_retinal_stimulus(case: BackimageCase) -> torch.Tensor:
    from mcfarland_sim import shift_movie_with_eye
    from spatial_info import embed_time_lags
    from scripts.fixrsvp_eye_conventions import stored_eyepos_to_eye_norm

    if case.eyepos.shape[0] < N_LAGS:
        raise ValueError(f"Need at least N_LAGS={N_LAGS} eye samples, got {case.eyepos.shape[0]}")

    image = np.asarray(case.image, dtype=np.float32)
    eyepos_t = torch.from_numpy(case.eyepos).float()
    eye_norm = stored_eyepos_to_eye_norm(eyepos_t, PPD, image.shape[-2:], device=eyepos_t.device)

    movie = torch.from_numpy(image).float().unsqueeze(0).expand(case.eyepos.shape[0] + N_LAGS, -1, -1)
    eye_movie = shift_movie_with_eye(
        movie,
        torch.cat([eye_norm[:N_LAGS], eye_norm], dim=0),
        out_size=OUT_SIZE,
        scale_factor=SCALE_FACTOR,
        mode="bilinear",
    )
    stim = embed_time_lags(eye_movie, n_lags=N_LAGS)
    return (stim - 127.0) / 255.0


def compute_activation_movie(
    case: BackimageCase,
    bundle: ModelBundle,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    from spatial_info import compute_rate_map_batched

    stim = build_retinal_stimulus(case)
    behavior = build_behavior_from_eyepos(case.eyepos, bundle.dataset_config)
    # build_retinal_stimulus prepends N_LAGS frames, so embed_time_lags emits one
    # extra leading "warmup" output frame. Drop it so stim[i] aligns frame-for-
    # frame with eyepos[i] and behavior[i] (T == len(eyepos)).
    if stim.shape[0] == behavior.shape[0] + 1:
        stim = stim[1:]
    if behavior.shape[0] != stim.shape[0]:
        raise ValueError(
            f"Behavior/stimulus time mismatch: stim T={stim.shape[0]} vs "
            f"behavior T={behavior.shape[0]}"
        )
    with torch.no_grad():
        y = compute_rate_map_batched(
            bundle.model, bundle.readout, stim, batch_size=batch_size, behavior=behavior
        )
    y_np = y.detach().cpu().numpy().astype(np.float32) if isinstance(y, torch.Tensor) else np.asarray(y, dtype=np.float32)
    if y_np.ndim != 4:
        raise ValueError(f"Expected activation movie T x C x H x W, got {y_np.shape}")
    return y_np


def save_activation_cache(
    path: Path,
    activation_movie: np.ndarray,
    case: BackimageCase,
    bundle: ModelBundle,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unit_sessions = np.asarray([str(row.get("session", "")) for row in bundle.unit_rows])
    source_unit_indices = np.asarray([int(row.get("source_unit_index", -1)) for row in bundle.unit_rows], dtype=np.int64)
    ccnorm = np.asarray([float(row.get("ccnorm", np.nan)) for row in bundle.unit_rows], dtype=np.float32)
    metadata = {
        "analysis": "canonical_twin_channel_redundancy_step1_activation_fingerprints",
        "image_key": case.image_key,
        "trace_index": case.trace_index,
        "centered_eye_trace": case.centered_eye_trace,
        "activation_shape": list(activation_movie.shape),
        "n_lags": N_LAGS,
        "out_size": list(OUT_SIZE),
        "scale_factor": SCALE_FACTOR,
        "ppd": PPD,
        "model_mode": MODEL_MODE,
        "device": bundle.device,
    }
    np.savez_compressed(
        path,
        activation_movie=activation_movie.astype(np.float32),
        image=case.image.astype(np.float32),
        eyepos=case.eyepos.astype(np.float32),
        unit_sessions=unit_sessions,
        source_unit_indices=source_unit_indices,
        ccnorm=ccnorm,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    _save_json(path.with_suffix(".json"), metadata)
    _write_csv(path.with_name(path.stem + "_unit_table.csv"), bundle.unit_rows)


def load_activation_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as npz:
        metadata_json = str(npz["metadata_json"].item()) if "metadata_json" in npz else "{}"
        return {
            "activation_movie": np.asarray(npz["activation_movie"], dtype=np.float32),
            "image": np.asarray(npz["image"], dtype=np.float32) if "image" in npz else None,
            "eyepos": np.asarray(npz["eyepos"], dtype=np.float32) if "eyepos" in npz else None,
            "metadata": json.loads(metadata_json),
        }


def fixrsvp_activation_cache_path(
    *,
    stimulus_type: str = FIXRSVP_STIMULUS_TYPE,
    frames_per_image: int = FIXRSVP_FRAMES_PER_IMAGE,
    num_frames: int = FIXRSVP_NUM_FRAMES,
    frame: int | None = FIXRSVP_FRAME,
) -> Path:
    frame_part = "dynamic" if frame is None else f"frame{int(frame):03d}"
    name = (
        f"activation_movie_{_safe_slug(stimulus_type)}_{frame_part}"
        f"_fpi{int(frames_per_image)}_frames{int(num_frames)}"
        f"_lag{N_LAGS}_out{OUT_SIZE[0]}x{OUT_SIZE[1]}_scale{SCALE_FACTOR:g}.npz"
    )
    return OUT_DIR / name


def _coerce_grayscale_movie(stack: np.ndarray) -> np.ndarray:
    arr = np.asarray(stack, dtype=np.float32)
    if arr.ndim == 3:
        return arr
    if arr.ndim != 4:
        raise ValueError(f"Expected stimulus stack T x H x W or T x C/H x H/W x C, got {arr.shape}")
    if arr.shape[-1] in (1, 3, 4):
        return np.nanmean(arr[..., : min(arr.shape[-1], 3)], axis=-1).astype(np.float32)
    if arr.shape[1] in (1, 3, 4):
        return np.nanmean(arr[:, : min(arr.shape[1], 3)], axis=1).astype(np.float32)
    raise ValueError(f"Could not infer channel axis for stimulus stack with shape {arr.shape}")


def build_fixrsvp_stimulus(
    *,
    stimulus_type: str = FIXRSVP_STIMULUS_TYPE,
    frames_per_image: int = FIXRSVP_FRAMES_PER_IMAGE,
    num_frames: int = FIXRSVP_NUM_FRAMES,
    frame: int | None = FIXRSVP_FRAME,
    n_lags: int = N_LAGS,
    out_size: tuple[int, int] = OUT_SIZE,
    scale_factor: float = SCALE_FACTOR,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Build a static-eye FixRSVP stimulus in the same VisionCore format as BackImage.

    This keeps the audit focused on visual diversity: RSVP images change over time,
    but eye position is fixed at the center, and the stimulus is cropped to OUT_SIZE
    before lag embedding.
    """
    from mcfarland_sim import shift_movie_with_eye
    from spatial_info import embed_time_lags, make_stimulus_stack

    num_frames = int(num_frames)
    if num_frames < 1:
        raise ValueError(f"FIXRSVP_NUM_FRAMES must be positive, got {num_frames}")
    needed_raw_frames = num_frames + int(n_lags) - 1

    raw_stack = make_stimulus_stack(
        type=stimulus_type,
        frame=frame,
        frames_per_im=int(frames_per_image),
        num_frames=needed_raw_frames,
    )
    raw_movie = _coerce_grayscale_movie(np.asarray(raw_stack, dtype=np.float32))
    source_frames = int(raw_movie.shape[0])
    if source_frames < needed_raw_frames:
        repeats = int(np.ceil(needed_raw_frames / max(source_frames, 1)))
        raw_movie = np.tile(raw_movie, (repeats, 1, 1))
    raw_movie = raw_movie[:needed_raw_frames]

    movie_t = torch.from_numpy(raw_movie.astype(np.float32, copy=False))
    eye_norm = torch.zeros((movie_t.shape[0], 2), dtype=movie_t.dtype)
    retinal_movie = shift_movie_with_eye(
        movie_t,
        eye_norm,
        out_size=out_size,
        scale_factor=scale_factor,
        mode="bilinear",
    )
    stim = embed_time_lags(retinal_movie, n_lags=int(n_lags))
    if stim.shape[0] != num_frames:
        raise RuntimeError(f"Expected {num_frames} lagged frames, got {stim.shape[0]}")

    metadata = {
        "analysis": "canonical_twin_channel_redundancy_fixrsvp_audit",
        "stimulus_type": stimulus_type,
        "frames_per_image": int(frames_per_image),
        "requested_model_frames": num_frames,
        "source_frames_before_tiling": source_frames,
        "raw_frames_after_tiling": int(raw_movie.shape[0]),
        "frame": None if frame is None else int(frame),
        "n_lags": int(n_lags),
        "out_size": list(out_size),
        "scale_factor": float(scale_factor),
        "stim_shape": list(stim.shape),
        "static_center_eye_trace": True,
    }
    return (stim - 127.0) / 255.0, metadata


def compute_fixrsvp_activation_movie(
    bundle: ModelBundle,
    *,
    stimulus_type: str = FIXRSVP_STIMULUS_TYPE,
    frames_per_image: int = FIXRSVP_FRAMES_PER_IMAGE,
    num_frames: int = FIXRSVP_NUM_FRAMES,
    frame: int | None = FIXRSVP_FRAME,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, dict[str, Any]]:
    from spatial_info import compute_rate_map_batched

    stim, metadata = build_fixrsvp_stimulus(
        stimulus_type=stimulus_type,
        frames_per_image=frames_per_image,
        num_frames=num_frames,
        frame=frame,
    )
    # This stimulus holds the eye static at center (see build_fixrsvp_stimulus).
    # For a behavior-conditioned twin that means zero eye velocity + centered eye
    # position, i.e. a zero behavior vector. We build it directly rather than
    # running an all-zero eye trace through the training transforms, whose
    # maxnorm step would divide by zero; the correct static-eye behavior is
    # exactly zeros. None-modulator twins get behavior=None (unchanged).
    modulator = getattr(bundle.model.model, "modulator", None)
    if modulator is not None:
        behavior_dim = int(getattr(modulator, "behavior_dim", 0))
        behavior = torch.zeros((stim.shape[0], behavior_dim), dtype=stim.dtype)
        metadata["behavior"] = "static_center_zero"
        metadata["behavior_dim"] = behavior_dim
    else:
        behavior = None
        metadata["behavior"] = "none"
    with torch.no_grad():
        y = compute_rate_map_batched(
            bundle.model, bundle.readout, stim, batch_size=int(batch_size), behavior=behavior
        )
    y_np = y.detach().cpu().numpy().astype(np.float32) if isinstance(y, torch.Tensor) else np.asarray(y, dtype=np.float32)
    if y_np.ndim != 4:
        raise ValueError(f"Expected activation movie T x C x H x W, got {y_np.shape}")
    metadata["activation_shape"] = list(y_np.shape)
    metadata["model_mode"] = MODEL_MODE
    metadata["device"] = bundle.device
    return y_np, metadata


def save_fixrsvp_activation_cache(path: Path, activation_movie: np.ndarray, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        activation_movie=np.asarray(activation_movie, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    _save_json(path.with_suffix(".json"), metadata)


# %% Activation diagnostics
TRACE_MODE_LABELS = {
    "spatial_mean": "spatial-mean",
    "center_pixel": "center-pixel",
}


def trace_mode_label(trace_mode: str) -> str:
    return TRACE_MODE_LABELS.get(str(trace_mode), str(trace_mode).replace("_", "-"))


def trace_mode_zscore_default(trace_mode: str) -> bool:
    return bool(TRACE_PLOT_ZSCORE_BY_MODE.get(str(trace_mode), True))


def activation_summary_rows(activation_movie: np.ndarray) -> list[dict[str, Any]]:
    y = np.asarray(activation_movie, dtype=np.float32)
    rows = [
        {"metric": "shape_T_C_H_W", "value": " x ".join(str(v) for v in y.shape)},
        {"metric": "finite_fraction", "value": float(np.isfinite(y).mean())},
        {"metric": "global_min", "value": float(np.nanmin(y))},
        {"metric": "global_max", "value": float(np.nanmax(y))},
        {"metric": "global_mean", "value": float(np.nanmean(y))},
        {"metric": "global_std", "value": float(np.nanstd(y))},
    ]
    channel_std = np.nanstd(y, axis=(0, 2, 3))
    rows.extend(
        [
            {"metric": "channel_std_median", "value": float(np.nanmedian(channel_std))},
            {"metric": "near_silent_channels_std_lt_1e-6", "value": int(np.sum(channel_std < 1e-6))},
        ]
    )
    return rows


def spatial_mean_traces(activation_movie: np.ndarray) -> np.ndarray:
    y = np.asarray(activation_movie, dtype=np.float32)
    return y.mean(axis=(2, 3))


def center_pixel_traces(activation_movie: np.ndarray) -> np.ndarray:
    """Center-pixel model responses from the post-activation rate-map movie."""
    y = np.asarray(activation_movie, dtype=np.float32)
    center_y = y.shape[2] // 2
    center_x = y.shape[3] // 2
    return y[:, :, center_y, center_x]


def activation_traces(activation_movie: np.ndarray, trace_mode: str = "spatial_mean") -> np.ndarray:
    """Collapse each activation map to a T x C trace for visual diagnostics."""
    if trace_mode == "spatial_mean":
        return spatial_mean_traces(activation_movie)
    if trace_mode == "center_pixel":
        return center_pixel_traces(activation_movie)
    raise ValueError(f"Unknown trace_mode {trace_mode!r}; expected one of {sorted(TRACE_MODE_LABELS)}")


def channel_variance_rank(activation_movie: np.ndarray) -> np.ndarray:
    y = np.asarray(activation_movie, dtype=np.float32)
    return np.argsort(np.nanstd(y, axis=(0, 2, 3)))[::-1]


def plot_activation_distributions(
    activation_movie: np.ndarray,
    *,
    trace_mode: str = "spatial_mean",
):
    y = np.asarray(activation_movie, dtype=np.float32)
    channel_mean = np.nanmean(y, axis=(0, 2, 3))
    channel_std = np.nanstd(y, axis=(0, 2, 3))
    traces = activation_traces(y, trace_mode=trace_mode)
    trace_label = trace_mode_label(trace_mode)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    axes[0].hist(channel_mean, bins=60, color="0.25")
    axes[0].set_title("Channel Mean")
    axes[0].set_xlabel("activation")
    axes[0].set_ylabel("channels")

    axes[1].hist(channel_std, bins=60, color="0.25")
    axes[1].set_title("Channel Std")
    axes[1].set_xlabel("activation std")

    axes[2].plot(np.nanmean(traces, axis=1), color="black", linewidth=1.2)
    axes[2].fill_between(
        np.arange(traces.shape[0]),
        np.nanpercentile(traces, 10, axis=1),
        np.nanpercentile(traces, 90, axis=1),
        color="0.7",
        alpha=0.5,
        linewidth=0,
    )
    axes[2].set_title(f"Population Activity ({trace_label})")
    axes[2].set_xlabel("model frame")
    axes[2].set_ylabel(f"{trace_label} activation")
    return fig


def plot_activation_eye_trace_alignment(
    activation_movie: np.ndarray,
    eyepos: np.ndarray,
    *,
    speed_percentile: float = 95.0,
    trace_mode: str = "spatial_mean",
) -> plt.Figure:
    """Overlay eye motion with population activity to audit motion-linked transients."""
    y = np.asarray(activation_movie, dtype=np.float32)
    eye = np.asarray(eyepos, dtype=np.float32)
    traces = activation_traces(y, trace_mode=trace_mode)
    trace_label = trace_mode_label(trace_mode)
    n = min(y.shape[0], eye.shape[0])
    if n < 2:
        raise ValueError(f"Need at least two aligned frames, got activation={y.shape[0]} eye={eye.shape[0]}")

    eye = eye[:n]
    pop = np.nanmean(traces[:n], axis=1)
    d_eye = np.diff(eye, axis=0, prepend=eye[:1])
    speed = np.sqrt(np.sum(d_eye * d_eye, axis=1))
    speed_cut = float(np.nanpercentile(speed, speed_percentile))
    fast = speed >= speed_cut
    t = np.arange(n)

    fig, axes = plt.subplots(3, 1, figsize=(12, 6.2), sharex=True, constrained_layout=True)
    axes[0].plot(t, eye[:, 0], label="x", linewidth=1.0)
    axes[0].plot(t, eye[:, 1], label="y", linewidth=1.0)
    axes[0].set_ylabel("eye position (deg)")
    axes[0].set_title("Stored eye trace aligned to activation movie")
    axes[0].legend(frameon=False, loc="upper right", ncol=2)

    axes[1].plot(t, speed, color="#444444", linewidth=1.0)
    axes[1].fill_between(t, 0.0, speed, where=fast, color="#d95f02", alpha=0.25, linewidth=0)
    axes[1].axhline(speed_cut, color="#d95f02", linestyle="--", linewidth=1.0, label=f"p{speed_percentile:.0f}")
    axes[1].set_ylabel("eye speed\n(deg/frame)")
    axes[1].set_title("Frame-to-frame eye displacement")
    axes[1].legend(frameon=False, loc="upper right")

    pop_z = _zscore_1d(pop)
    speed_z = _zscore_1d(speed)
    axes[2].plot(t, pop_z, color="black", linewidth=1.2, label=f"population {trace_label}")
    axes[2].plot(t, speed_z, color="#d95f02", linewidth=0.9, alpha=0.8, label="eye speed")
    axes[2].fill_between(t, np.nanmin(pop_z), np.nanmax(pop_z), where=fast, color="#d95f02", alpha=0.08, linewidth=0)
    axes[2].set_ylabel("z-score")
    axes[2].set_xlabel("model frame")
    axes[2].set_title("Do population transients align with rapid eye motion?")
    axes[2].legend(frameon=False, loc="upper right")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return fig


def plot_channel_trace_snippets(
    activation_movie: np.ndarray,
    channels: Iterable[int],
    unit_rows: list[dict[str, Any]] | None = None,
    frame_slice: slice = slice(0, 120),
    zscore: bool | None = None,
    trace_mode: str = "spatial_mean",
):
    traces = activation_traces(activation_movie, trace_mode=trace_mode)
    trace_label = trace_mode_label(trace_mode)
    if zscore is None:
        zscore = trace_mode_zscore_default(trace_mode)
    channels = [int(c) for c in channels]
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    for channel in channels:
        trace = traces[frame_slice, channel]
        if zscore:
            trace = _zscore_1d(trace)
        ax.plot(np.arange(trace.shape[0]), trace, linewidth=1.0, alpha=0.85, label=_channel_label(channel, unit_rows))
    ax.set_title(f"{trace_label.title()} Channel Trace Snippets")
    ax.set_xlabel("model frame")
    ax.set_ylabel("z-scored activation" if zscore else "model activation")
    if len(channels) <= 12:
        ax.legend(frameon=False, ncol=2, fontsize=8)
    return fig


def select_activation_map_frame(
    activation_movie: np.ndarray,
    channels: Iterable[int] | None = None,
    *,
    frame_mode: str = MAP_FRAME_MODE,
    frame_index: int | None = MAP_FRAME_INDEX,
    peak_percentile: float = 99.0,
) -> int:
    """Pick one model frame for single-frame activation-map display."""
    y = np.asarray(activation_movie, dtype=np.float32)
    if y.shape[0] < 1:
        raise ValueError("Activation movie has no frames")
    mode = str(frame_mode)
    if mode == "final":
        return int(y.shape[0] - 1)
    if mode == "index":
        idx = int(0 if frame_index is None else frame_index)
        return int(np.clip(idx, 0, y.shape[0] - 1))
    if mode != "peak":
        raise ValueError(f"Unknown MAP_FRAME_MODE {frame_mode!r}; expected 'peak', 'final', or 'index'")

    if channels is None:
        channel_arr = np.arange(y.shape[1], dtype=int)
    else:
        channel_arr = np.asarray([int(ch) for ch in channels], dtype=int)
        channel_arr = channel_arr[(0 <= channel_arr) & (channel_arr < y.shape[1])]
    if channel_arr.size == 0:
        return int(y.shape[0] - 1)

    shared_map = np.nanmean(y[:, channel_arr], axis=1)
    flat = shared_map.reshape(shared_map.shape[0], -1)
    frame_scores = np.nanpercentile(flat, peak_percentile, axis=1)
    if not np.any(np.isfinite(frame_scores)):
        return int(y.shape[0] - 1)
    return int(np.nanargmax(frame_scores))


def map_frame_title(frame: int, frame_mode: str = MAP_FRAME_MODE) -> str:
    if frame_mode == "peak":
        return f"peak-shared t={int(frame)}"
    if frame_mode == "final":
        return f"final t={int(frame)}"
    return f"t={int(frame)}"


def plot_channel_map_strip(
    activation_movie: np.ndarray,
    channels: Iterable[int],
    frames: Iterable[int],
    unit_rows: list[dict[str, Any]] | None = None,
    robust: bool = True,
):
    y = np.asarray(activation_movie, dtype=np.float32)
    channels = [int(c) for c in channels]
    frames = [int(f) for f in frames if 0 <= int(f) < y.shape[0]]
    if not channels or not frames:
        raise ValueError("Need at least one channel and one valid frame")

    fig, axes = plt.subplots(
        len(channels),
        len(frames),
        figsize=(1.8 * len(frames), 1.75 * len(channels)),
        squeeze=False,
        constrained_layout=True,
    )
    for r, channel in enumerate(channels):
        vals = y[:, channel]
        if robust:
            vmin, vmax = np.nanpercentile(vals, [2, 98])
        else:
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = None, None
        for c, frame in enumerate(frames):
            ax = axes[r, c]
            ax.imshow(y[frame, channel], cmap=ACTIVATION_MAP_CMAP, vmin=vmin, vmax=vmax)
            if r == 0:
                ax.set_title(f"t={frame}")
            if c == 0:
                ax.set_ylabel(_channel_label(channel, unit_rows), fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    return fig


def plot_top_pair_overview(
    activation_movie: np.ndarray,
    pairs: list[dict[str, Any]],
    unit_rows: list[dict[str, Any]] | None = None,
    n_pairs: int = 8,
    frame_slice: slice = slice(0, 120),
    trace_mode: str = "spatial_mean",
    zscore: bool | None = None,
):
    y = np.asarray(activation_movie, dtype=np.float32)
    traces = activation_traces(y, trace_mode=trace_mode)
    trace_label = trace_mode_label(trace_mode)
    if zscore is None:
        zscore = trace_mode_zscore_default(trace_mode)
    selected = pairs[:n_pairs]
    if not selected:
        raise ValueError("No pairs to plot")

    fig, axes = plt.subplots(len(selected), 3, figsize=(13, 2.0 * len(selected)), squeeze=False, constrained_layout=True)
    for r, pair in enumerate(selected):
        a = int(pair["channel_a"])
        b = int(pair["channel_b"])
        corr = float(pair["corr"])
        ax = axes[r, 0]
        ta = traces[frame_slice, a]
        tb = traces[frame_slice, b]
        if zscore:
            ta = _zscore_1d(ta)
            tb = _zscore_1d(tb)
        ax.plot(ta, label=_channel_label(a, unit_rows), linewidth=1.0)
        ax.plot(tb, label=_channel_label(b, unit_rows), linewidth=1.0)
        ax.set_title(f"rank {pair.get('rank', r + 1)} corr={corr:.3f}")
        ax.set_xlabel("frame")
        ax.set_ylabel(f"{trace_label} {'z' if zscore else 'activation'}")
        ax.legend(frameon=False, fontsize=7)

        map_frame = select_activation_map_frame(y, channels=[a, b])
        map_a = y[map_frame, a]
        map_b = y[map_frame, b]
        vmin = float(np.nanpercentile(np.stack([map_a, map_b]), 2))
        vmax = float(np.nanpercentile(np.stack([map_a, map_b]), 98))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = None, None
        axes[r, 1].imshow(map_a, cmap=ACTIVATION_MAP_CMAP, vmin=vmin, vmax=vmax)
        axes[r, 1].set_title(f"A {map_frame_title(map_frame)}")
        axes[r, 2].imshow(map_b, cmap=ACTIVATION_MAP_CMAP, vmin=vmin, vmax=vmax)
        axes[r, 2].set_title(f"B {map_frame_title(map_frame)}")
        for c in (1, 2):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    return fig


# %% Fingerprints and similarity
def make_channel_fingerprints(
    activation_movie: np.ndarray,
    normalization: str = FINGERPRINT_NORMALIZATION,
    eps: float = 1e-6,
) -> np.ndarray:
    y = np.asarray(activation_movie, dtype=np.float32)
    fingerprints = np.transpose(y, (1, 0, 2, 3)).reshape(y.shape[1], -1)
    if normalization == "none":
        return fingerprints.astype(np.float32, copy=False)
    fingerprints = fingerprints - np.nanmean(fingerprints, axis=1, keepdims=True)
    if normalization == "center":
        return fingerprints.astype(np.float32, copy=False)
    if normalization == "zscore":
        fingerprints = fingerprints / (np.nanstd(fingerprints, axis=1, keepdims=True) + eps)
        return fingerprints.astype(np.float32, copy=False)
    raise ValueError(f"Unknown fingerprint normalization {normalization!r}")


def channel_correlation_from_fingerprints(fingerprints: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(fingerprints, dtype=np.float32)
    x = x - np.nanmean(x, axis=1, keepdims=True)
    norms = np.sqrt(np.nansum(x * x, axis=1, keepdims=True))
    z = x / np.maximum(norms, eps)
    corr = z @ z.T
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float32)


def top_correlated_pairs(
    corr: np.ndarray,
    unit_rows: list[dict[str, Any]] | None = None,
    n_pairs: int = N_TOP_PAIRS,
    min_corr: float | None = None,
) -> list[dict[str, Any]]:
    corr = np.asarray(corr, dtype=np.float32)
    ii, jj = np.triu_indices(corr.shape[0], k=1)
    vals = corr[ii, jj]
    order = np.argsort(vals)[::-1]
    rows: list[dict[str, Any]] = []
    for idx in order:
        value = float(vals[idx])
        if min_corr is not None and value < min_corr:
            break
        a = int(ii[idx])
        b = int(jj[idx])
        row: dict[str, Any] = {
            "rank": len(rows) + 1,
            "channel_a": a,
            "channel_b": b,
            "corr": value,
        }
        if unit_rows is not None and a < len(unit_rows) and b < len(unit_rows):
            row.update(
                {
                    "session_a": unit_rows[a].get("session", ""),
                    "source_unit_a": unit_rows[a].get("source_unit_index", ""),
                    "ccnorm_a": unit_rows[a].get("ccnorm", np.nan),
                    "session_b": unit_rows[b].get("session", ""),
                    "source_unit_b": unit_rows[b].get("source_unit_index", ""),
                    "ccnorm_b": unit_rows[b].get("ccnorm", np.nan),
                    "same_session": unit_rows[a].get("session", "") == unit_rows[b].get("session", ""),
                }
            )
        rows.append(row)
        if len(rows) >= int(n_pairs):
            break
    return rows


def plot_correlation_heatmap(
    corr: np.ndarray,
    channel_order: Iterable[int] | None = None,
):
    corr = np.asarray(corr, dtype=np.float32)
    if channel_order is None:
        order = np.arange(corr.shape[0])
    else:
        order = np.asarray(list(channel_order), dtype=int)
    sub = corr[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(sub, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(f"Channel Fingerprint Correlation ({len(order)} channels)")
    ax.set_xlabel("ordered channel")
    ax.set_ylabel("ordered channel")
    fig.colorbar(im, ax=ax, shrink=0.8, label="corr")
    return fig


def plot_reduced_corr_heatmap(
    corr: np.ndarray,
    labels: np.ndarray,
    title: str = "Representative correlation",
) -> "plt.Figure":
    """
    Plot the N_reps x N_reps pooled-representative correlation after merging.

    Group representatives are the mean of their member channels, so their
    correlation is computed as M @ corr @ M.T and then diagonal-normalized.
    Singletons (label=-1) pass through. Excluded units (label=-2) are omitted.
    The ordering matches the population spec rep_idx: groups first, then singletons.
    """
    labels = np.asarray(labels, dtype=int)
    corr_arr = np.asarray(corr, dtype=np.float64)
    group_ids = sorted(set(labels[labels >= 0]))
    singleton_channels = list(np.flatnonzero(labels == -1))
    n_reps = len(group_ids) + len(singleton_channels)
    membership = np.zeros((n_reps, labels.size), dtype=np.float64)

    for rep_idx, gid in enumerate(group_ids):
        members = np.flatnonzero(labels == gid)
        membership[rep_idx, members] = 1.0 / members.size
    for rep_idx, ch in enumerate(singleton_channels, start=len(group_ids)):
        membership[rep_idx, int(ch)] = 1.0

    rep_cov = membership @ corr_arr @ membership.T
    rep_std = np.sqrt(np.maximum(np.diag(rep_cov), 1e-12))
    sub = rep_cov / np.maximum(np.outer(rep_std, rep_std), 1e-12)
    sub = np.clip(sub, -1.0, 1.0)
    np.fill_diagonal(sub, 1.0)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(sub, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    if group_ids and singleton_channels:
        ax.axhline(len(group_ids) - 0.5, color="0.1", linewidth=0.7, alpha=0.55)
        ax.axvline(len(group_ids) - 0.5, color="0.1", linewidth=0.7, alpha=0.55)
    ax.set_title(f"{title} ({n_reps} representatives)")
    ax.set_xlabel("representative index")
    ax.set_ylabel("representative index")
    fig.colorbar(im, ax=ax, shrink=0.8, label="corr")
    return fig


def save_fingerprint_cache(
    path: Path,
    fingerprints: np.ndarray,
    corr: np.ndarray,
    top_pairs: list[dict[str, Any]],
    unit_mean_rate: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "fingerprints": fingerprints.astype(np.float32),
        "corr": corr.astype(np.float32),
        "top_pairs_json": np.asarray(json.dumps(top_pairs, sort_keys=True, default=_json_default)),
    }
    if unit_mean_rate is not None:
        arrays["unit_mean_rate"] = np.asarray(unit_mean_rate, dtype=np.float32)
    np.savez_compressed(path, **arrays)
    _write_csv(path.with_suffix(".top_pairs.csv"), top_pairs)


# %% Optional embedding view
def compute_channel_embedding(
    fingerprints: np.ndarray,
    run_tsne: bool = RUN_TSNE,
    seed: int = RANDOM_SEED,
    pca_dim: int = 30,
) -> tuple[np.ndarray, str, np.ndarray]:
    """Returns (embedding_2d, title, pca_scores). pca_scores is C x pca_dim."""
    try:
        from sklearn.decomposition import PCA
    except Exception as exc:
        raise RuntimeError(f"scikit-learn is needed for embeddings: {exc}") from exc

    x = np.asarray(fingerprints, dtype=np.float32)
    n_components = min(int(pca_dim), x.shape[0] - 1, x.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    scores = pca.fit_transform(x)
    if not run_tsne:
        if scores.shape[1] < 2:
            scores = np.pad(scores, ((0, 0), (0, 2 - scores.shape[1])))
        return scores[:, :2], "PCA", scores

    try:
        from sklearn.manifold import TSNE
    except Exception as exc:
        raise RuntimeError(f"scikit-learn TSNE import failed: {exc}") from exc

    perplexity = min(30.0, max(5.0, (x.shape[0] - 1) / 4.0))
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
    )
    return tsne.fit_transform(scores), "t-SNE on PCA fingerprints", scores


def pca_cosine_similarity(pca_scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Cosine similarity matrix from PCA scores (C x D).

    More robust than raw fingerprint correlation because:
    - Invariant to per-unit amplitude (a dim unit isn't penalised)
    - Operates in the PCA-denoised space, reducing frame-noise sensitivity
    This is not the t-SNE affinity itself; it is a denoised-space similarity
    that can be compared against raw fingerprint correlation without relying on
    distances in the distorted 2D t-SNE embedding.
    """
    s = np.asarray(pca_scores, dtype=np.float64)
    norms = np.linalg.norm(s, axis=1, keepdims=True)
    s_norm = s / np.maximum(norms, eps)
    sim = s_norm @ s_norm.T
    return np.clip(sim, -1.0, 1.0).astype(np.float32)


def redundancy_groups_from_corr(
    corr: np.ndarray,
    threshold: float,
    method: str = REDUNDANCY_LINKAGE_METHOD,
) -> np.ndarray:
    """
    Hierarchical clustering on fingerprint correlation.

    With method="complete", every within-group pair must exceed the threshold.
    With method="average", the average linkage distance must pass threshold, so
    groups can be larger but may contain some weaker pairwise matches.

    Singletons (no partner above threshold) get label -1.
    """
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except ImportError as exc:
        raise RuntimeError(f"scipy is required for redundancy grouping: {exc}") from exc
    if method not in {"complete", "average", "weighted", "single"}:
        raise ValueError(f"Unsupported linkage method {method!r}")
    dist = np.clip(1.0 - np.asarray(corr, dtype=np.float64), 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method=method)
    raw_labels = fcluster(Z, t=1.0 - threshold, criterion="distance") - 1  # 0-indexed
    counts = np.bincount(raw_labels)
    raw_labels = raw_labels.copy()
    raw_labels[np.isin(raw_labels, np.where(counts == 1)[0])] = -1
    return raw_labels


def _within_group_pairwise_corr(corr: np.ndarray, members: np.ndarray) -> np.ndarray:
    members = np.asarray(members, dtype=int)
    if members.size < 2:
        return np.asarray([], dtype=np.float32)
    sub = np.asarray(corr, dtype=np.float32)[np.ix_(members, members)]
    ii, jj = np.triu_indices(members.size, k=1)
    return sub[ii, jj]


def redundancy_group_rows(
    corr: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    method: str,
) -> list[dict[str, Any]]:
    corr_arr = np.asarray(corr, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for group_id in sorted(set(labels[labels >= 0])):
        members = np.flatnonzero(labels == group_id)
        pair_corr = _within_group_pairwise_corr(corr, members)
        sub_corr = corr_arr[np.ix_(members, members)]
        centroid_norm = float(np.sqrt(max(float(np.mean(sub_corr)), 1e-12)))
        centroid_corr = np.mean(sub_corr, axis=1) / centroid_norm
        eigvals = np.linalg.eigvalsh(sub_corr.astype(np.float64))
        eigvals = np.clip(eigvals, 0.0, None)
        pc1_fraction = float(np.max(eigvals) / np.sum(eigvals)) if np.sum(eigvals) > 0 else np.nan
        rows.append(
            {
                "method": method,
                "threshold": float(threshold),
                "group_id": int(group_id),
                "size": int(members.size),
                "min_pair_corr": float(np.min(pair_corr)) if pair_corr.size else np.nan,
                "median_pair_corr": float(np.median(pair_corr)) if pair_corr.size else np.nan,
                "mean_pair_corr": float(np.mean(pair_corr)) if pair_corr.size else np.nan,
                "min_centroid_corr": float(np.min(centroid_corr)),
                "median_centroid_corr": float(np.median(centroid_corr)),
                "pc1_fraction_from_corr": pc1_fraction,
                "members": ",".join(str(int(m)) for m in members),
            }
        )
    rows.sort(key=lambda row: (-int(row["size"]), float(row["min_pair_corr"])))
    return rows


def summarize_redundancy_grouping(
    corr: np.ndarray,
    thresholds: list[float],
    methods: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for method in methods:
        for threshold in thresholds:
            labels = redundancy_groups_from_corr(corr, threshold, method=method)
            group_rows = redundancy_group_rows(corr, labels, threshold=threshold, method=method)
            detail_rows.extend(group_rows)
            group_sizes = np.asarray([row["size"] for row in group_rows], dtype=np.float32)
            n_groups = len(group_rows)
            n_singletons = int(np.sum(labels < 0))
            n_redundant = int(np.sum(labels >= 0))
            after_merge = int(n_groups + n_singletons)
            weakest_pairs = np.asarray([row["min_pair_corr"] for row in group_rows], dtype=np.float32)
            weakest_centroids = np.asarray([row["min_centroid_corr"] for row in group_rows], dtype=np.float32)
            pc1_fractions = np.asarray([row["pc1_fraction_from_corr"] for row in group_rows], dtype=np.float32)
            summary_rows.append(
                {
                    "method": method,
                    "threshold": float(threshold),
                    "n_input_units": int(labels.size),
                    "n_after_merge": after_merge,
                    "n_removed": int(labels.size - after_merge),
                    "n_groups": n_groups,
                    "n_redundant_units": n_redundant,
                    "n_singletons": n_singletons,
                    "largest_group": int(np.max(group_sizes)) if group_sizes.size else 1,
                    "median_group_size": float(np.median(group_sizes)) if group_sizes.size else 1.0,
                    "weakest_group_min_corr": float(np.min(weakest_pairs)) if weakest_pairs.size else np.nan,
                    "groups_with_min_pair_below_threshold": int(np.sum(weakest_pairs < threshold)) if weakest_pairs.size else 0,
                    "weakest_group_min_centroid_corr": float(np.min(weakest_centroids)) if weakest_centroids.size else np.nan,
                    "median_group_pc1_fraction": float(np.median(pc1_fractions)) if pc1_fractions.size else np.nan,
                }
            )
    return summary_rows, detail_rows


def add_similarity_label(rows: list[dict[str, Any]], similarity: str) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    for row in rows:
        out = {"similarity": similarity}
        out.update(row)
        labeled.append(out)
    return labeled


def summarize_redundancy_grouping_labeled(
    similarity: str,
    corr: np.ndarray,
    thresholds: list[float],
    methods: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows, detail_rows = summarize_redundancy_grouping(
        corr,
        thresholds=thresholds,
        methods=methods,
    )
    return add_similarity_label(summary_rows, similarity), add_similarity_label(detail_rows, similarity)


def plot_redundancy_metric_grid(
    summary_rows: list[dict[str, Any]],
    metrics: list[tuple[str, str]] | None = None,
) -> plt.Figure:
    if metrics is None:
        metrics = [
            ("n_after_merge", "units after merge"),
            ("n_removed", "units removed"),
            ("largest_group", "largest group"),
            ("groups_with_min_pair_below_threshold", "groups with min pair below threshold"),
            ("weakest_group_min_centroid_corr", "weakest group centroid similarity"),
            ("median_group_pc1_fraction", "median group PC1 fraction"),
        ]

    similarities = sorted({str(row.get("similarity", "")) for row in summary_rows})
    methods = sorted({str(row.get("method", "")) for row in summary_rows})
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    colors = {name: color_cycle[i % len(color_cycle)] for i, name in enumerate(similarities)}
    linestyles = {method: "-" if method == "complete" else "--" if method == "average" else ":" for method in methods}

    n_cols = 3
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.1 * n_cols, 3.6 * n_rows), squeeze=False, constrained_layout=True)
    axes_flat = axes.ravel()

    for ax, (metric, ylabel) in zip(axes_flat, metrics):
        for similarity in similarities:
            for method in methods:
                rows = [
                    row
                    for row in summary_rows
                    if str(row.get("similarity", "")) == similarity and str(row.get("method", "")) == method
                ]
                rows.sort(key=lambda row: float(row["threshold"]))
                if not rows:
                    continue
                x = np.asarray([float(row["threshold"]) for row in rows], dtype=float)
                y = np.asarray([float(row.get(metric, np.nan)) for row in rows], dtype=float)
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=1.8,
                    color=colors[similarity],
                    linestyle=linestyles[method],
                    label=f"{similarity} / {method}",
                )
        ax.set_xlabel("similarity threshold")
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(metrics):]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    fig.suptitle("Redundancy grouping metrics across similarity, linkage, and threshold", fontsize=11)
    return fig


def _corr_to_centroid(fingerprints: np.ndarray, members: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(fingerprints, dtype=np.float32)[np.asarray(members, dtype=int)]
    centroid = np.mean(x, axis=0)
    x_centered = x - np.mean(x, axis=1, keepdims=True)
    centroid_centered = centroid - float(np.mean(centroid))
    numerator = x_centered @ centroid_centered
    denom = np.linalg.norm(x_centered, axis=1) * max(float(np.linalg.norm(centroid_centered)), eps)
    return np.asarray(numerator / np.maximum(denom, eps), dtype=np.float32)


def _pc1_fraction_from_group_fingerprints(fingerprints: np.ndarray, members: np.ndarray) -> float:
    members = np.asarray(members, dtype=int)
    if members.size < 2:
        return float("nan")
    group_corr = channel_correlation_from_fingerprints(np.asarray(fingerprints, dtype=np.float32)[members])
    eigvals = np.linalg.eigvalsh(group_corr.astype(np.float64))
    eigvals = np.clip(eigvals, 0.0, None)
    total = float(np.sum(eigvals))
    return float(np.max(eigvals) / total) if total > 0 else float("nan")


def validate_redundancy_candidate(
    version_name: str,
    labels: np.ndarray,
    train_fingerprints: np.ndarray,
    heldout_fingerprints: np.ndarray,
    spec: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    labels = np.asarray(labels, dtype=int)
    group_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    for group_id in sorted(set(labels[labels >= 0])):
        members = np.flatnonzero(labels == group_id)
        train_corrs = _corr_to_centroid(train_fingerprints, members)
        heldout_corrs = _corr_to_centroid(heldout_fingerprints, members)
        heldout_residual = np.clip(1.0 - heldout_corrs * heldout_corrs, 0.0, 1.0)
        heldout_pairwise = _pairwise_corr_from_fingerprints(heldout_fingerprints, members)
        finite_pairwise = heldout_pairwise[np.isfinite(heldout_pairwise)]
        row = {
            "version": version_name,
            "similarity": spec["similarity"],
            "method": spec["method"],
            "threshold": float(spec["threshold"]),
            "group_id": int(group_id),
            "size": int(members.size),
            "train_min_centroid_corr": float(np.min(train_corrs)),
            "train_median_centroid_corr": float(np.median(train_corrs)),
            "heldout_min_centroid_corr": float(np.min(heldout_corrs)),
            "heldout_median_centroid_corr": float(np.median(heldout_corrs)),
            "heldout_p05_centroid_corr": float(np.percentile(heldout_corrs, 5)),
            "heldout_median_residual_fraction": float(np.median(heldout_residual)),
            "heldout_pc1_fraction": _pc1_fraction_from_group_fingerprints(heldout_fingerprints, members),
            "heldout_min_pairwise_corr": float(np.nanmin(finite_pairwise)) if finite_pairwise.size else np.nan,
            "heldout_median_pairwise_corr": float(np.nanmedian(finite_pairwise)) if finite_pairwise.size else np.nan,
            "members": ",".join(str(int(m)) for m in members),
        }
        group_rows.append(row)
        for member, train_corr, heldout_corr, residual in zip(members, train_corrs, heldout_corrs, heldout_residual):
            member_rows.append(
                {
                    "version": version_name,
                    "similarity": spec["similarity"],
                    "method": spec["method"],
                    "threshold": float(spec["threshold"]),
                    "group_id": int(group_id),
                    "group_size": int(members.size),
                    "channel": int(member),
                    "train_centroid_corr": float(train_corr),
                    "heldout_centroid_corr": float(heldout_corr),
                    "heldout_residual_fraction": float(residual),
                }
            )
    group_rows.sort(key=lambda row: (float(row["heldout_min_pairwise_corr"]), float(row["heldout_min_centroid_corr"]), -int(row["size"])))

    n_groups = len(group_rows)
    n_singletons = int(np.sum(labels == -1))
    n_excluded = int(np.sum(labels == -2))
    n_after_merge = int(n_groups + n_singletons)
    heldout_member_corrs = np.asarray([row["heldout_centroid_corr"] for row in member_rows], dtype=np.float32)
    heldout_group_min = np.asarray([row["heldout_min_centroid_corr"] for row in group_rows], dtype=np.float32)
    heldout_group_pc1 = np.asarray([row["heldout_pc1_fraction"] for row in group_rows], dtype=np.float32)
    heldout_group_min_pairwise = np.asarray(
        [row["heldout_min_pairwise_corr"] for row in group_rows], dtype=np.float32
    )
    summary = {
        "version": version_name,
        "similarity": spec["similarity"],
        "method": spec["method"],
        "threshold": float(spec["threshold"]),
        "n_input_units": int(labels.size),
        "n_after_merge": n_after_merge,
        "n_removed": int(labels.size - n_excluded - n_after_merge),
        "n_groups": n_groups,
        "n_redundant_units": int(np.sum(labels >= 0)),
        "n_singletons": n_singletons,
        "n_excluded": n_excluded,
        "largest_group": int(max([row["size"] for row in group_rows], default=1)),
        "heldout_min_member_centroid_corr": float(np.min(heldout_member_corrs)) if heldout_member_corrs.size else np.nan,
        "heldout_p05_member_centroid_corr": float(np.percentile(heldout_member_corrs, 5)) if heldout_member_corrs.size else np.nan,
        "heldout_median_member_centroid_corr": float(np.median(heldout_member_corrs)) if heldout_member_corrs.size else np.nan,
        "heldout_worst_group_min_centroid_corr": float(np.min(heldout_group_min)) if heldout_group_min.size else np.nan,
        "heldout_median_group_min_centroid_corr": float(np.median(heldout_group_min)) if heldout_group_min.size else np.nan,
        "heldout_median_group_pc1_fraction": float(np.median(heldout_group_pc1)) if heldout_group_pc1.size else np.nan,
        "groups_with_heldout_min_centroid_below_0p75": int(np.sum(heldout_group_min < 0.75)) if heldout_group_min.size else 0,
        "groups_with_heldout_min_centroid_below_0p60": int(np.sum(heldout_group_min < 0.60)) if heldout_group_min.size else 0,
        "heldout_worst_group_min_pairwise_corr": float(np.nanmin(heldout_group_min_pairwise)) if heldout_group_min_pairwise.size else np.nan,
        "groups_with_heldout_min_pairwise_below_0p75": int(np.nansum(heldout_group_min_pairwise < 0.75)) if heldout_group_min_pairwise.size else 0,
        "groups_with_heldout_min_pairwise_below_0p60": int(np.nansum(heldout_group_min_pairwise < 0.60)) if heldout_group_min_pairwise.size else 0,
    }
    return summary, group_rows, member_rows


def _flat_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    aa = aa - float(np.mean(aa))
    bb = bb - float(np.mean(bb))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom < eps:
        return float("nan")
    return float(np.dot(aa, bb) / denom)


def _pairwise_corr_from_fingerprints(
    fingerprints: np.ndarray,
    members: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Upper-triangle pairwise Pearson correlations for a group, computed from fingerprint vectors."""
    members = np.asarray(members, dtype=int)
    if members.size < 2:
        return np.asarray([], dtype=np.float32)
    x = np.asarray(fingerprints, dtype=np.float64)[members]       # (n, D)
    x_c = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(x_c, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    x_n = x_c / norms
    corr_matrix = x_n @ x_n.T                                     # (n, n)
    ii, jj = np.triu_indices(members.size, k=1)
    return corr_matrix[ii, jj].astype(np.float32)


def _spatial_ssi_single_frame_np(rate_maps: np.ndarray, eps: float = 1e-8) -> float:
    y = np.asarray(rate_maps, dtype=np.float64)
    if y.ndim != 3:
        raise ValueError(f"Expected C x H x W rate maps, got {y.shape}")
    flat = y.reshape(y.shape[0], -1)
    rbar = flat.mean(axis=1)
    gain = flat / (rbar[:, None] + eps)
    unit_bits = np.mean(gain * np.log2(gain + eps), axis=1)
    weights = rbar / max(float(np.sum(rbar)), eps)
    return float(np.sum(weights * unit_bits))


def _spatial_ssi_timecourse_np(movie: np.ndarray) -> np.ndarray:
    y = np.asarray(movie, dtype=np.float32)
    if y.ndim != 4:
        raise ValueError(f"Expected T x C x H x W movie, got {y.shape}")
    return np.asarray([_spatial_ssi_single_frame_np(y[t]) for t in range(y.shape[0])], dtype=np.float32)


def _group_quality_from_channel_first_array(
    version_name: str,
    labels: np.ndarray,
    channel_first: np.ndarray,
    *,
    case_label: str,
    metric_space: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    x = np.asarray(channel_first, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    if x.shape[0] != labels.size:
        raise ValueError(f"Array has {x.shape[0]} channels, labels have {labels.size}")

    group_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    for group_id in sorted(set(labels[labels >= 0])):
        members = np.flatnonzero(labels == group_id)
        if members.size <= 1:
            continue
        centroid = np.nanmean(x[members], axis=0)
        corrs = np.asarray([_flat_corr(x[m], centroid) for m in members], dtype=np.float32)
        stds = np.asarray([float(np.nanstd(x[m])) for m in members], dtype=np.float32)
        finite_corrs = corrs[np.isfinite(corrs)]
        worst_pos = int(np.nanargmin(corrs)) if finite_corrs.size else 0

        # Pairwise member-member correlations — catches bad merges where each member
        # looks acceptable relative to the centroid but members disagree with each other.
        vecs = [np.asarray(x[m], dtype=np.float64).ravel() for m in members]
        pairwise = np.asarray(
            [_flat_corr(vecs[i], vecs[j]) for i in range(len(members)) for j in range(i + 1, len(members))],
            dtype=np.float32,
        )
        finite_pairwise = pairwise[np.isfinite(pairwise)]
        worst_pair_pos = int(np.nanargmin(pairwise)) if finite_pairwise.size else 0
        if finite_pairwise.size:
            pair_idx, worst_pair = 0, (int(members[0]), int(members[1]))
            for pi in range(len(members)):
                for pj in range(pi + 1, len(members)):
                    if pair_idx == worst_pair_pos:
                        worst_pair = (int(members[pi]), int(members[pj]))
                    pair_idx += 1
        else:
            worst_pair = (-1, -1)

        # Leave-one-member-out centroid correlations are a stricter jackknife
        # check: each member is compared with the mean of the rest of its group,
        # so a member cannot make its own centroid look better.
        x_flat = np.asarray(x[members], dtype=np.float64).reshape(members.size, -1)
        x_centered = x_flat - np.nanmean(x_flat, axis=1, keepdims=True)
        member_norms = np.linalg.norm(x_centered, axis=1)
        loo_centered = (np.nansum(x_centered, axis=0, keepdims=True) - x_centered) / max(members.size - 1, 1)
        loo_norms = np.linalg.norm(loo_centered, axis=1)
        loo_corrs = np.sum(x_centered * loo_centered, axis=1) / np.maximum(member_norms * loo_norms, 1e-8)
        loo_corrs = np.asarray(loo_corrs, dtype=np.float32)
        finite_loo_corrs = loo_corrs[np.isfinite(loo_corrs)]
        worst_loo_pos = int(np.nanargmin(loo_corrs)) if finite_loo_corrs.size else 0

        row = {
            "version": version_name,
            "case": case_label,
            "metric_space": metric_space,
            "group_id": int(group_id),
            "size": int(members.size),
            "min_member_centroid_corr": float(np.nanmin(corrs)) if finite_corrs.size else np.nan,
            "p05_member_centroid_corr": float(np.nanpercentile(corrs, 5)) if finite_corrs.size else np.nan,
            "median_member_centroid_corr": float(np.nanmedian(corrs)) if finite_corrs.size else np.nan,
            "worst_member": int(members[worst_pos]),
            "worst_member_corr": float(corrs[worst_pos]),
            "min_pairwise_member_corr": float(np.nanmin(finite_pairwise)) if finite_pairwise.size else np.nan,
            "median_pairwise_member_corr": float(np.nanmedian(finite_pairwise)) if finite_pairwise.size else np.nan,
            "worst_pair_a": int(worst_pair[0]),
            "worst_pair_b": int(worst_pair[1]),
            "worst_pair_corr": float(pairwise[worst_pair_pos]) if finite_pairwise.size else np.nan,
            "min_leave_one_out_centroid_corr": float(np.nanmin(finite_loo_corrs)) if finite_loo_corrs.size else np.nan,
            "p05_leave_one_out_centroid_corr": float(np.nanpercentile(finite_loo_corrs, 5)) if finite_loo_corrs.size else np.nan,
            "median_leave_one_out_centroid_corr": float(np.nanmedian(finite_loo_corrs)) if finite_loo_corrs.size else np.nan,
            "worst_leave_one_out_member": int(members[worst_loo_pos]),
            "worst_leave_one_out_corr": float(loo_corrs[worst_loo_pos]) if finite_loo_corrs.size else np.nan,
            "min_member_std": float(np.nanmin(stds)),
            "median_member_std": float(np.nanmedian(stds)),
            "centroid_std": float(np.nanstd(centroid)),
            "members": ",".join(str(int(m)) for m in members),
        }
        group_rows.append(row)
        for member, corr_value, loo_corr_value, std_value in zip(members, corrs, loo_corrs, stds):
            member_rows.append(
                {
                    "version": version_name,
                    "case": case_label,
                    "metric_space": metric_space,
                    "group_id": int(group_id),
                    "group_size": int(members.size),
                    "channel": int(member),
                    "member_centroid_corr": float(corr_value),
                    "leave_one_out_centroid_corr": float(loo_corr_value),
                    "member_std": float(std_value),
                }
            )
    group_rows.sort(key=lambda row: (float(row["min_pairwise_member_corr"]), float(row["min_member_centroid_corr"]), -int(row["size"])))
    return group_rows, member_rows


def _expand_group_means_to_full(movie: np.ndarray, labels: np.ndarray) -> np.ndarray:
    y = np.asarray(movie, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    if y.ndim != 4:
        raise ValueError(f"Expected T x C x H x W movie, got {y.shape}")
    if y.shape[1] != labels.size:
        raise ValueError(f"Movie has {y.shape[1]} channels, labels have {labels.size}")
    expanded = np.zeros_like(y)
    singleton_mask = labels == -1           # true singletons pass through
    if np.any(singleton_mask):
        expanded[:, singleton_mask] = y[:, singleton_mask]
    # label == -2 (excluded/silent): left as zeros — not included in reconstruction
    for group_id in sorted(set(labels[labels >= 0])):
        members = np.flatnonzero(labels == group_id)
        centroid = np.nanmean(y[:, members], axis=1)
        expanded[:, members] = centroid[:, None]
    return expanded


def population_membership_from_labels(
    labels: np.ndarray,
    *,
    pooling_mode: str = "mean",
    medoid_channels: dict[int, int] | None = None,
) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=int)
    pooling_mode = str(pooling_mode)
    if pooling_mode not in {"mean", "medoid"}:
        raise ValueError(f"Unknown pooling_mode {pooling_mode!r}; expected 'mean' or 'medoid'")
    medoid_channels = {} if medoid_channels is None else {int(k): int(v) for k, v in medoid_channels.items()}
    group_ids = sorted(set(labels_arr[labels_arr >= 0]))
    singleton_channels = list(np.flatnonzero(labels_arr == -1))
    membership = np.zeros((len(group_ids) + len(singleton_channels), labels_arr.size), dtype=np.float32)

    for rep_idx, group_id in enumerate(group_ids):
        members = np.flatnonzero(labels_arr == group_id)
        if pooling_mode == "medoid":
            selected = int(medoid_channels.get(int(group_id), int(members[0])))
            if selected not in set(map(int, members)):
                raise ValueError(f"Medoid channel {selected} is not in group {group_id}")
            membership[rep_idx, selected] = 1.0
        else:
            membership[rep_idx, members] = 1.0 / max(int(members.size), 1)
    for rep_idx, channel in enumerate(singleton_channels, start=len(group_ids)):
        membership[rep_idx, int(channel)] = 1.0
    return membership


def _expand_population_membership_to_full(
    movie: np.ndarray,
    labels: np.ndarray,
    membership: np.ndarray,
) -> np.ndarray:
    y = np.asarray(movie, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=int)
    mem = np.asarray(membership, dtype=np.float32)
    if y.ndim != 4:
        raise ValueError(f"Expected T x C x H x W movie, got {y.shape}")
    if y.shape[1] != labels_arr.size or mem.shape[1] != labels_arr.size:
        raise ValueError(
            f"Movie/labels/membership mismatch: movie C={y.shape[1]}, labels={labels_arr.size}, membership={mem.shape}"
        )
    reduced = np.einsum("rc,tchw->trhw", mem, y, optimize=True).astype(np.float32, copy=False)
    expanded = np.zeros_like(y)
    group_ids = sorted(set(labels_arr[labels_arr >= 0]))
    for rep_idx, group_id in enumerate(group_ids):
        members = np.flatnonzero(labels_arr == group_id)
        expanded[:, members] = reduced[:, rep_idx, None]
    for rep_idx, channel in enumerate(np.flatnonzero(labels_arr == -1), start=len(group_ids)):
        expanded[:, int(channel)] = reduced[:, rep_idx]
    return expanded


def _pool_expand_reconstruction_row(
    version_name: str,
    labels: np.ndarray,
    movie: np.ndarray,
    *,
    case_label: str,
) -> dict[str, Any]:
    y_full = np.asarray(movie, dtype=np.float32)
    expanded_full = _expand_group_means_to_full(y_full, labels)

    # Restrict all metrics to non-excluded channels (label != -2)
    valid_mask = np.asarray(labels, dtype=int) != -2
    y = y_full[:, valid_mask]
    expanded = expanded_full[:, valid_mask]

    residual = expanded - y
    per_channel_corr = np.asarray(
        [_flat_corr(y[:, ch], expanded[:, ch]) for ch in range(y.shape[1])], dtype=np.float32
    )
    full_ssi = _spatial_ssi_timecourse_np(y)
    expanded_ssi = _spatial_ssi_timecourse_np(expanded)
    full_std = float(np.nanstd(y))
    return {
        "version": version_name,
        "case": case_label,
        "n_time": int(y.shape[0]),
        "n_channels": int(y.shape[1]),       # valid channels only
        "n_excluded": int(np.sum(~valid_mask)),
        "global_rate_corr": _flat_corr(y, expanded),
        "rmse": float(np.sqrt(np.nanmean(residual * residual))),
        "nrmse_by_full_std": float(np.sqrt(np.nanmean(residual * residual)) / max(full_std, 1e-8)),
        "min_channel_corr": float(np.nanmin(per_channel_corr)),
        "p05_channel_corr": float(np.nanpercentile(per_channel_corr, 5)),
        "median_channel_corr": float(np.nanmedian(per_channel_corr)),
        "mean_full_ssi": float(np.nanmean(full_ssi)),
        "mean_expanded_ssi": float(np.nanmean(expanded_ssi)),
        "mean_abs_expanded_minus_full_ssi": float(np.nanmean(np.abs(expanded_ssi - full_ssi))),
        "max_abs_expanded_minus_full_ssi": float(np.nanmax(np.abs(expanded_ssi - full_ssi))),
        "ssi_timecourse_corr_expanded_full": _flat_corr(full_ssi, expanded_ssi),
    }


def _pool_expand_reconstruction_row_from_membership(
    version_name: str,
    labels: np.ndarray,
    membership: np.ndarray,
    movie: np.ndarray,
    *,
    case_label: str,
    pooling_mode: str,
) -> dict[str, Any]:
    y_full = np.asarray(movie, dtype=np.float32)
    expanded_full = _expand_population_membership_to_full(y_full, labels, membership)

    valid_mask = np.asarray(labels, dtype=int) != -2
    y = y_full[:, valid_mask]
    expanded = expanded_full[:, valid_mask]

    residual = expanded - y
    per_channel_corr = np.asarray(
        [_flat_corr(y[:, ch], expanded[:, ch]) for ch in range(y.shape[1])], dtype=np.float32
    )
    full_ssi = _spatial_ssi_timecourse_np(y)
    expanded_ssi = _spatial_ssi_timecourse_np(expanded)
    full_std = float(np.nanstd(y))
    return {
        "version": version_name,
        "case": case_label,
        "pooling_mode": str(pooling_mode),
        "n_time": int(y.shape[0]),
        "n_channels": int(y.shape[1]),
        "n_representatives": int(np.asarray(membership).shape[0]),
        "n_excluded": int(np.sum(~valid_mask)),
        "global_rate_corr": _flat_corr(y, expanded),
        "rmse": float(np.sqrt(np.nanmean(residual * residual))),
        "nrmse_by_full_std": float(np.sqrt(np.nanmean(residual * residual)) / max(full_std, 1e-8)),
        "min_channel_corr": float(np.nanmin(per_channel_corr)),
        "p05_channel_corr": float(np.nanpercentile(per_channel_corr, 5)),
        "median_channel_corr": float(np.nanmedian(per_channel_corr)),
        "mean_full_ssi": float(np.nanmean(full_ssi)),
        "mean_expanded_ssi": float(np.nanmean(expanded_ssi)),
        "mean_abs_expanded_minus_full_ssi": float(np.nanmean(np.abs(expanded_ssi - full_ssi))),
        "max_abs_expanded_minus_full_ssi": float(np.nanmax(np.abs(expanded_ssi - full_ssi))),
        "ssi_timecourse_corr_expanded_full": _flat_corr(full_ssi, expanded_ssi),
    }


def audit_population_membership_on_movies(
    version_name: str,
    labels: np.ndarray,
    membership: np.ndarray,
    movies: list[np.ndarray],
    case_labels: list[str],
    *,
    pooling_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reconstruction_rows = [
        _pool_expand_reconstruction_row_from_membership(
            version_name,
            labels,
            membership,
            movie,
            case_label=case_label,
            pooling_mode=pooling_mode,
        )
        for movie, case_label in zip(movies, case_labels)
    ]
    recon_corr = np.asarray([float(row["global_rate_corr"]) for row in reconstruction_rows], dtype=np.float32)
    ssi_abs = np.asarray([float(row["mean_abs_expanded_minus_full_ssi"]) for row in reconstruction_rows], dtype=np.float32)
    return (
        {
            "version": version_name,
            "pooling_mode": str(pooling_mode),
            "n_cases": int(len(movies)),
            "n_input_units": int(np.asarray(labels).size),
            "n_representatives": int(np.asarray(membership).shape[0]),
            "mean_global_reconstruction_corr": float(np.nanmean(recon_corr)) if recon_corr.size else np.nan,
            "min_global_reconstruction_corr": float(np.nanmin(recon_corr)) if recon_corr.size else np.nan,
            "mean_abs_expanded_minus_full_ssi": float(np.nanmean(ssi_abs)) if ssi_abs.size else np.nan,
            "max_mean_abs_expanded_minus_full_ssi": float(np.nanmax(ssi_abs)) if ssi_abs.size else np.nan,
        },
        reconstruction_rows,
    )


def audit_redundancy_labels_on_heldout_movies(
    version_name: str,
    labels: np.ndarray,
    heldout_movies: list[np.ndarray],
    heldout_case_labels: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Spatial/movie audit for a candidate compact population on held-out activation movies."""
    group_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    for movie, case_label in zip(heldout_movies, heldout_case_labels):
        movie_group_rows, movie_member_rows = _group_quality_from_channel_first_array(
            version_name,
            labels,
            np.moveaxis(movie, 1, 0),
            case_label=case_label,
            metric_space="movie_t_h_w",
        )
        mean_map_group_rows, mean_map_member_rows = _group_quality_from_channel_first_array(
            version_name,
            labels,
            np.nanmean(movie, axis=0),
            case_label=case_label,
            metric_space="mean_map_h_w",
        )
        group_rows.extend(movie_group_rows)
        group_rows.extend(mean_map_group_rows)
        member_rows.extend(movie_member_rows)
        member_rows.extend(mean_map_member_rows)
        reconstruction_rows.append(
            _pool_expand_reconstruction_row(version_name, labels, movie, case_label=case_label)
        )

    movie_rows_only = [row for row in group_rows if str(row["metric_space"]) == "movie_t_h_w"]
    mean_map_rows_only = [row for row in group_rows if str(row["metric_space"]) == "mean_map_h_w"]
    movie_group_min = np.asarray([float(row["min_member_centroid_corr"]) for row in movie_rows_only], dtype=np.float32)
    movie_group_min_pairwise = np.asarray(
        [float(row.get("min_pairwise_member_corr", np.nan)) for row in movie_rows_only], dtype=np.float32
    )
    mean_map_group_min = np.asarray([float(row["min_member_centroid_corr"]) for row in mean_map_rows_only], dtype=np.float32)
    recon_corr = np.asarray([float(row["global_rate_corr"]) for row in reconstruction_rows], dtype=np.float32)
    n_groups = len(set(labels[labels >= 0]))
    n_singletons = int(np.sum(labels == -1))   # true singletons only; label=-2 = excluded
    n_excluded = int(np.sum(labels == -2))
    summary = {
        "version": version_name,
        "n_cases": int(len(heldout_movies)),
        "n_input_units": int(labels.size),
        "n_representatives": int(n_groups + n_singletons),
        "n_groups": int(n_groups),
        "n_singletons": int(n_singletons),
        "n_excluded": int(n_excluded),
        "movie_worst_group_min_centroid_corr": float(np.nanmin(movie_group_min)) if movie_group_min.size else np.nan,
        "movie_median_group_min_centroid_corr": float(np.nanmedian(movie_group_min)) if movie_group_min.size else np.nan,
        "movie_groups_below_0p75": int(np.sum(movie_group_min < 0.75)) if movie_group_min.size else 0,
        "movie_groups_below_0p60": int(np.sum(movie_group_min < 0.60)) if movie_group_min.size else 0,
        "movie_worst_group_min_pairwise_corr": float(np.nanmin(movie_group_min_pairwise)) if movie_group_min_pairwise.size else np.nan,
        "movie_groups_below_0p75_pairwise": int(np.nansum(movie_group_min_pairwise < 0.75)) if movie_group_min_pairwise.size else 0,
        "movie_groups_below_0p60_pairwise": int(np.nansum(movie_group_min_pairwise < 0.60)) if movie_group_min_pairwise.size else 0,
        "mean_map_worst_group_min_centroid_corr": float(np.nanmin(mean_map_group_min)) if mean_map_group_min.size else np.nan,
        "mean_map_groups_below_0p75": int(np.sum(mean_map_group_min < 0.75)) if mean_map_group_min.size else 0,
        "mean_global_reconstruction_corr": float(np.nanmean(recon_corr)) if recon_corr.size else np.nan,
        "min_global_reconstruction_corr": float(np.nanmin(recon_corr)) if recon_corr.size else np.nan,
    }
    return summary, group_rows, member_rows, reconstruction_rows


def plot_stimulus_cluster_audit_summary(
    group_rows: list[dict[str, Any]],
    reconstruction_rows: list[dict[str, Any]],
    *,
    version_name: str,
    stimulus_label: str,
) -> plt.Figure:
    movie_rows = [row for row in group_rows if str(row.get("metric_space")) == "movie_t_h_w"]
    mean_map_rows = [row for row in group_rows if str(row.get("metric_space")) == "mean_map_h_w"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.3), constrained_layout=True)
    axes_flat = axes.ravel()
    bins = np.linspace(-0.2, 1.0, 49)

    if not movie_rows:
        for ax in axes_flat:
            ax.axis("off")
        axes_flat[0].text(0.5, 0.5, "No multi-member groups to audit", ha="center", va="center")
        fig.suptitle(f"{stimulus_label} cluster audit - {version_name}", fontsize=11)
        return fig

    movie_min = np.asarray([float(row["min_member_centroid_corr"]) for row in movie_rows], dtype=np.float32)
    movie_pair = np.asarray([float(row.get("min_pairwise_member_corr", np.nan)) for row in movie_rows], dtype=np.float32)
    movie_loo = np.asarray([float(row.get("min_leave_one_out_centroid_corr", np.nan)) for row in movie_rows], dtype=np.float32)
    movie_size = np.asarray([int(row["size"]) for row in movie_rows], dtype=np.int32)
    mean_map_min = np.asarray([float(row["min_member_centroid_corr"]) for row in mean_map_rows], dtype=np.float32)

    axes_flat[0].hist(movie_min, bins=bins, histtype="stepfilled", alpha=0.45, label="movie")
    if mean_map_min.size:
        axes_flat[0].hist(mean_map_min, bins=bins, histtype="step", linewidth=1.8, label="mean map")
    for threshold, linestyle in ((0.60, ":"), (0.75, "--")):
        axes_flat[0].axvline(threshold, color="0.25", linestyle=linestyle, linewidth=1.0)
    axes_flat[0].set_xlabel("min member-centroid corr")
    axes_flat[0].set_ylabel("groups")
    axes_flat[0].set_title("Group coherence")
    axes_flat[0].legend(frameon=False, fontsize=8)
    axes_flat[0].text(
        0.02,
        0.98,
        f"{int(np.sum(movie_min < 0.60))} groups < .60\n{int(np.sum(movie_min < 0.75))} groups < .75",
        transform=axes_flat[0].transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    finite_pair = movie_pair[np.isfinite(movie_pair)]
    finite_loo = movie_loo[np.isfinite(movie_loo)]
    if finite_pair.size:
        axes_flat[1].hist(finite_pair, bins=bins, histtype="stepfilled", alpha=0.38, label="pairwise")
    if finite_loo.size:
        axes_flat[1].hist(finite_loo, bins=bins, histtype="step", linewidth=1.8, label="leave-one-out")
    for threshold, linestyle in ((0.50, ":"), (0.60, "--")):
        axes_flat[1].axvline(threshold, color="0.25", linestyle=linestyle, linewidth=1.0)
    axes_flat[1].set_xlabel("correlation")
    axes_flat[1].set_ylabel("groups")
    axes_flat[1].set_title("Hidden-merge stress tests")
    axes_flat[1].legend(frameon=False, fontsize=8)

    color_values = np.where(np.isfinite(movie_pair), movie_pair, movie_min)
    scatter = axes_flat[2].scatter(
        movie_size,
        movie_min,
        c=color_values,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        s=np.clip(movie_size * 4, 16, 180),
        alpha=0.72,
        linewidths=0,
    )
    axes_flat[2].axhline(0.75, color="0.25", linestyle="--", linewidth=1.0)
    axes_flat[2].axhline(0.60, color="0.25", linestyle=":", linewidth=1.0)
    axes_flat[2].set_xlabel("group size")
    axes_flat[2].set_ylabel("movie min member-centroid corr")
    axes_flat[2].set_title("Quality vs group size")
    fig.colorbar(scatter, ax=axes_flat[2], shrink=0.82, label="min pairwise corr")

    if reconstruction_rows:
        recon = reconstruction_rows[0]
        metrics = [
            ("global_rate_corr", "global"),
            ("median_channel_corr", "median ch"),
            ("p05_channel_corr", "p05 ch"),
            ("min_channel_corr", "min ch"),
            ("ssi_timecourse_corr_expanded_full", "SSI time"),
        ]
        vals = [float(recon.get(key, np.nan)) for key, _label in metrics]
        finite_vals = np.asarray([value for value in vals if np.isfinite(value)], dtype=np.float32)
        x = np.arange(len(metrics))
        axes_flat[3].bar(x, vals, color=["tab:green", "tab:purple", "tab:orange", "tab:red", "tab:blue"], alpha=0.75)
        axes_flat[3].axhline(0, color="0.2", linewidth=0.8)
        lower = min(-0.05, float(np.min(finite_vals)) - 0.05) if finite_vals.size else -0.05
        axes_flat[3].set_ylim(lower, 1.02)
        axes_flat[3].set_xticks(x)
        axes_flat[3].set_xticklabels([label for _key, label in metrics], rotation=20)
        axes_flat[3].set_ylabel("correlation")
        axes_flat[3].set_title(
            f"Pool-expand reconstruction\nNRMSE={float(recon.get('nrmse_by_full_std', np.nan)):.3f}"
        )
    else:
        axes_flat[3].axis("off")

    fig.suptitle(f"{stimulus_label} cluster audit - {version_name}", fontsize=11)
    return fig


def audit_redundancy_labels_on_movie_blocks(
    version_name: str,
    labels: np.ndarray,
    heldout_movies: list[np.ndarray],
    heldout_case_labels: list[str],
    *,
    n_blocks: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Generic robustness audit over temporal blocks of held-out natural movies.

    This keeps the validation panel clean: groups are tested against independent
    BackImage movie blocks, not against the downstream image/condition suite.
    """
    group_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    for movie, case_label in zip(heldout_movies, heldout_case_labels):
        y = np.asarray(movie, dtype=np.float32)
        if y.ndim != 4:
            raise ValueError(f"Expected T x C x H x W movie, got {y.shape}")
        edges = np.linspace(0, y.shape[0], int(n_blocks) + 1, dtype=int)
        for block_idx, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
            if stop <= start:
                continue
            block_rows, block_member_rows = _group_quality_from_channel_first_array(
                version_name,
                labels,
                np.moveaxis(y[start:stop], 1, 0),
                case_label=str(case_label),
                metric_space="movie_t_block_h_w",
            )
            for row in block_rows:
                row["block_idx"] = int(block_idx)
                row["block_start"] = int(start)
                row["block_stop"] = int(stop)
                row["block_frames"] = int(stop - start)
            for row in block_member_rows:
                row["block_idx"] = int(block_idx)
                row["block_start"] = int(start)
                row["block_stop"] = int(stop)
                row["block_frames"] = int(stop - start)
            group_rows.extend(block_rows)
            member_rows.extend(block_member_rows)
    return group_rows, member_rows


def _bad_group_ids_from_block_jackknife_audit(
    group_rows: list[dict[str, Any]],
    *,
    pairwise_threshold: float,
    pairwise_min_blocks: int,
    loo_centroid_threshold: float,
    loo_centroid_min_blocks: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Flag groups with repeated blockwise pairwise or jackknife-centroid failures."""
    metrics_by_group: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"pairwise": [], "loo": []}
    )
    for row in group_rows:
        if str(row.get("metric_space")) != "movie_t_block_h_w":
            continue
        gid = int(row["group_id"])
        metrics_by_group[gid]["pairwise"].append(float(row.get("min_pairwise_member_corr", np.nan)))
        metrics_by_group[gid]["loo"].append(float(row.get("min_leave_one_out_centroid_corr", np.nan)))

    bad_ids: list[int] = []
    summary_rows: list[dict[str, Any]] = []
    for gid, metrics in sorted(metrics_by_group.items()):
        pairwise = np.asarray(metrics["pairwise"], dtype=np.float32)
        loo = np.asarray(metrics["loo"], dtype=np.float32)
        pairwise_finite = pairwise[np.isfinite(pairwise)]
        loo_finite = loo[np.isfinite(loo)]
        pairwise_fail_count = int(np.sum(pairwise_finite < float(pairwise_threshold)))
        loo_fail_count = int(np.sum(loo_finite < float(loo_centroid_threshold)))
        fails_pairwise = pairwise_fail_count >= int(pairwise_min_blocks)
        fails_loo = loo_fail_count >= int(loo_centroid_min_blocks)
        if fails_pairwise or fails_loo:
            bad_ids.append(int(gid))
        summary_rows.append(
            {
                "group_id": int(gid),
                "n_blocks": int(max(pairwise_finite.size, loo_finite.size)),
                "pairwise_threshold": float(pairwise_threshold),
                "pairwise_fail_count": pairwise_fail_count,
                "pairwise_fail_fraction": float(pairwise_fail_count / pairwise_finite.size) if pairwise_finite.size else 0.0,
                "min_pairwise": float(np.nanmin(pairwise_finite)) if pairwise_finite.size else np.nan,
                "median_pairwise": float(np.nanmedian(pairwise_finite)) if pairwise_finite.size else np.nan,
                "loo_centroid_threshold": float(loo_centroid_threshold),
                "loo_centroid_fail_count": loo_fail_count,
                "loo_centroid_fail_fraction": float(loo_fail_count / loo_finite.size) if loo_finite.size else 0.0,
                "min_loo_centroid": float(np.nanmin(loo_finite)) if loo_finite.size else np.nan,
                "median_loo_centroid": float(np.nanmedian(loo_finite)) if loo_finite.size else np.nan,
                "fails_pairwise": bool(fails_pairwise),
                "fails_loo_centroid": bool(fails_loo),
                "flagged": bool(fails_pairwise or fails_loo),
            }
        )
    return bad_ids, summary_rows


def plot_heldout_validation_summary(
    summary_rows: list[dict[str, Any]],
    member_rows: list[dict[str, Any]],
) -> plt.Figure:
    versions = [str(row["version"]) for row in summary_rows]
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), constrained_layout=True)
    axes_flat = axes.ravel()

    axes_flat[0].bar(versions, [float(row["n_after_merge"]) for row in summary_rows], color=colors[: len(versions)])
    axes_flat[0].set_ylabel("units after merge")
    axes_flat[0].set_title("Compression")
    axes_flat[0].tick_params(axis="x", rotation=20)

    axes_flat[1].bar(
        versions,
        [float(row["heldout_worst_group_min_centroid_corr"]) for row in summary_rows],
        color=colors[: len(versions)],
    )
    axes_flat[1].axhline(0.75, color="0.3", linestyle="--", linewidth=1.0)
    axes_flat[1].set_ylim(0.0, 1.02)
    axes_flat[1].set_ylabel("worst group min corr")
    axes_flat[1].set_title("Held-out weakest group")
    axes_flat[1].tick_params(axis="x", rotation=20)

    bins = np.linspace(0.0, 1.0, 41)
    for i, version in enumerate(versions):
        vals = [
            float(row["heldout_centroid_corr"])
            for row in member_rows
            if str(row["version"]) == version
        ]
        axes_flat[2].hist(vals, bins=bins, histtype="step", linewidth=2.0, color=colors[i % len(colors)], label=version)
    axes_flat[2].axvline(0.75, color="0.3", linestyle="--", linewidth=1.0)
    axes_flat[2].set_xlabel("held-out unit-to-group centroid corr")
    axes_flat[2].set_ylabel("units")
    axes_flat[2].set_title("Member reconstruction distribution")
    axes_flat[2].legend(frameon=False, fontsize=8)

    axes_flat[3].bar(
        versions,
        [float(row["groups_with_heldout_min_centroid_below_0p75"]) for row in summary_rows],
        color=colors[: len(versions)],
    )
    axes_flat[3].set_ylabel("groups")
    axes_flat[3].set_title("Groups with held-out min corr < 0.75")
    axes_flat[3].tick_params(axis="x", rotation=20)
    fig.suptitle("Held-out redundancy reconstruction checks", fontsize=12)
    return fig


def select_validation_groups_to_plot(
    group_rows: list[dict[str, Any]],
    version_name: str,
    n_groups: int = N_VALIDATION_GROUPS_TO_PLOT,
) -> list[dict[str, Any]]:
    rows = [row for row in group_rows if str(row["version"]) == version_name]
    largest = sorted(rows, key=lambda row: (-int(row["size"]), _group_row_min_centroid_corr(row)))
    weakest = sorted(rows, key=lambda row: (_group_row_min_centroid_corr(row), -int(row["size"])))
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in largest[: max(1, n_groups // 2)] + weakest:
        gid = int(row["group_id"])
        if gid in seen:
            continue
        selected.append(row)
        seen.add(gid)
        if len(selected) >= n_groups:
            break
    return selected


def _members_from_group_row(row: dict[str, Any]) -> list[int]:
    return [int(part) for part in str(row["members"]).split(",") if part.strip()]


def _group_row_min_centroid_corr(row: dict[str, Any]) -> float:
    for key in ("heldout_min_centroid_corr", "min_member_centroid_corr", "min_centroid_corr"):
        if key in row:
            return float(row[key])
    return float("nan")


def plot_group_trace_overlays(
    activation_movie: np.ndarray,
    group_rows: list[dict[str, Any]],
    unit_rows: list[dict[str, Any]] | None = None,
    max_frames: int = VALIDATION_TRACE_FRAMES,
    trace_mode: str = "spatial_mean",
    zscore: bool | None = None,
) -> plt.Figure:
    traces = activation_traces(activation_movie, trace_mode=trace_mode)
    trace_label = trace_mode_label(trace_mode)
    if zscore is None:
        zscore = trace_mode_zscore_default(trace_mode)
    n = len(group_rows)
    fig, axes = plt.subplots(n, 1, figsize=(12.5, max(2.1 * n, 3.0)), squeeze=False, constrained_layout=True)
    for ax, row in zip(axes.ravel(), group_rows):
        members = _members_from_group_row(row)
        frame_slice = slice(0, min(max_frames, traces.shape[0]))
        group_traces = traces[frame_slice][:, members]
        if zscore:
            plot_traces = np.stack([_zscore_1d(group_traces[:, i]) for i in range(group_traces.shape[1])], axis=1)
            centroid = _zscore_1d(np.nanmean(group_traces, axis=1))
        else:
            plot_traces = group_traces
            centroid = np.nanmean(group_traces, axis=1)
        for i, member in enumerate(members):
            label = _channel_label(member, unit_rows).replace("\n", " ")
            ax.plot(plot_traces[:, i], color="0.65", linewidth=0.9, alpha=0.55, label=label if i < 4 else None)
        ax.plot(centroid, color="black", linewidth=2.0, label="group mean")
        min_corr = _group_row_min_centroid_corr(row)
        metric_space = str(row.get("metric_space", "heldout"))
        ax.set_title(
            f"{row['version']} | group {row['group_id']} | n={row['size']} | "
            f"{metric_space} min centroid corr={min_corr:.3f} | {trace_label}",
            fontsize=9,
        )
        ax.set_ylabel("z" if zscore else "model activation")
        ax.grid(True, alpha=0.18)
        if len(members) <= 4:
            ax.legend(frameon=False, fontsize=7, ncol=2)
    axes[-1, 0].set_xlabel("held-out movie frame")
    return fig


def plot_group_activation_map_panels(
    activation_movie: np.ndarray,
    group_rows: list[dict[str, Any]],
    unit_rows: list[dict[str, Any]] | None = None,
    max_members: int = N_VALIDATION_MAP_MEMBERS,
    frame_mode: str = MAP_FRAME_MODE,
) -> plt.Figure:
    y = np.asarray(activation_movie, dtype=np.float32)
    n_rows = len(group_rows)
    n_cols = max_members + 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.15 * n_cols, 2.15 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    for r, row in enumerate(group_rows):
        members = _members_from_group_row(row)
        shown_members = members[:max_members]
        map_frame = select_activation_map_frame(y, channels=members, frame_mode=frame_mode)
        maps = [np.nanmean(y[map_frame, members], axis=0)]
        maps.extend(y[map_frame, member] for member in shown_members)
        vals = np.stack(maps)
        vmin, vmax = np.nanpercentile(vals, [2, 98])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = None, None
        titles = [f"group mean\n{map_frame_title(map_frame, frame_mode)}"] + [
            _channel_label(member, unit_rows).replace("\n", " ") for member in shown_members
        ]
        for c in range(n_cols):
            ax = axes[r, c]
            if c < len(maps):
                ax.imshow(maps[c], cmap=ACTIVATION_MAP_CMAP, vmin=vmin, vmax=vmax)
                ax.set_title(titles[c], fontsize=7)
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(
            f"{row['version']}\ng{row['group_id']} n={row['size']}\nmin={_group_row_min_centroid_corr(row):.2f}",
            fontsize=7,
        )
    fig.suptitle("Single-frame activation maps for selected redundancy groups", fontsize=11)
    return fig


def _channel_membership_status(channel: int, labels: np.ndarray | None) -> tuple[str, int | None]:
    if labels is None:
        return "unlabeled", None
    labels_arr = np.asarray(labels, dtype=int)
    if channel < 0 or channel >= labels_arr.size:
        return "unlabeled", None
    group_id = int(labels_arr[channel])
    if group_id == -1:
        return "singleton", -1
    if group_id == -2:
        return "excluded", -2
    return f"group {group_id}", group_id


def spatial_map_sharpness_scores(maps: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Spatial concentration score for C x H x W maps.

    A uniform positive map scores near 1; a very concentrated map scores higher.
    This is only a visual selector, not a clustering criterion.
    """
    arr = np.asarray(maps, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.size == 0:
        return np.empty((0,), dtype=np.float32)
    flat = arr.reshape(arr.shape[0], -1)
    baseline = np.nanpercentile(flat, 5, axis=1, keepdims=True)
    positive = np.maximum(flat - baseline, 0.0)
    positive = np.where(np.isfinite(positive), positive, 0.0)
    total = np.nansum(positive, axis=1, keepdims=True)
    weights = np.divide(
        positive,
        np.maximum(total, eps),
        out=np.zeros_like(positive, dtype=np.float32),
        where=total > eps,
    )
    scores = np.nansum(weights * weights, axis=1) * flat.shape[1]
    return scores.astype(np.float32, copy=False)


def channel_spatial_sharpness_rows(
    activation_movie: np.ndarray,
    channels: Iterable[int] | None = None,
    labels: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    y = np.asarray(activation_movie, dtype=np.float32)
    if channels is None:
        channel_arr = np.arange(y.shape[1], dtype=int)
    else:
        channel_arr = np.asarray([int(ch) for ch in channels], dtype=int)
        channel_arr = channel_arr[(0 <= channel_arr) & (channel_arr < y.shape[1])]
    if channel_arr.size == 0:
        return []

    final_frame = int(y.shape[0] - 1)
    final_maps = y[final_frame, channel_arr]
    flat_t_c_p = np.transpose(y[:, channel_arr], (1, 0, 2, 3)).reshape(channel_arr.size, y.shape[0], -1)
    frame_scores = np.nanpercentile(flat_t_c_p, 99, axis=2)
    peak_frames = np.zeros(channel_arr.size, dtype=int)
    for i in range(channel_arr.size):
        if np.any(np.isfinite(frame_scores[i])):
            peak_frames[i] = int(np.nanargmax(frame_scores[i]))
        else:
            peak_frames[i] = final_frame
    peak_maps = y[peak_frames, channel_arr]

    peak_scores = spatial_map_sharpness_scores(peak_maps)
    final_scores = spatial_map_sharpness_scores(final_maps)
    peak_activation = np.nanmax(peak_maps.reshape(channel_arr.size, -1), axis=1)
    final_activation = np.nanmax(final_maps.reshape(channel_arr.size, -1), axis=1)

    rows: list[dict[str, Any]] = []
    for i, channel in enumerate(channel_arr):
        membership, group_id = _channel_membership_status(int(channel), labels)
        score_pair = np.asarray([peak_scores[i], final_scores[i]], dtype=np.float32)
        score = float(np.nanmax(score_pair)) if np.any(np.isfinite(score_pair)) else float("nan")
        rows.append(
            {
                "channel": int(channel),
                "membership": membership,
                "group_id": group_id,
                "sharpness_score": score,
                "peak_frame": int(peak_frames[i]),
                "final_frame": final_frame,
                "peak_frame_sharpness": float(peak_scores[i]),
                "final_frame_sharpness": float(final_scores[i]),
                "peak_activation": float(peak_activation[i]),
                "final_activation": float(final_activation[i]),
            }
        )
    return rows


def select_sharp_channel_rows(
    activation_movie: np.ndarray,
    channels: Iterable[int] | None = None,
    labels: np.ndarray | None = None,
    n_channels: int = N_SHARP_MAPS_TO_PLOT,
) -> list[dict[str, Any]]:
    rows = channel_spatial_sharpness_rows(activation_movie, channels=channels, labels=labels)
    rows.sort(
        key=lambda row: (
            -float(row["sharpness_score"]) if np.isfinite(float(row["sharpness_score"])) else np.inf,
            -float(row["peak_activation"]) if np.isfinite(float(row["peak_activation"])) else np.inf,
            int(row["channel"]),
        )
    )
    return rows[: int(n_channels)]


def _channel_display_maps(
    activation_movie: np.ndarray,
    channel: int,
) -> list[tuple[str, np.ndarray]]:
    y = np.asarray(activation_movie, dtype=np.float32)
    maps_t_h_w = y[:, int(channel)]
    flat = maps_t_h_w.reshape(maps_t_h_w.shape[0], -1)
    frame_scores = np.nanpercentile(flat, 99, axis=1)
    if np.any(np.isfinite(frame_scores)):
        peak_frame = int(np.nanargmax(frame_scores))
    else:
        peak_frame = 0
    final_frame = int(maps_t_h_w.shape[0] - 1)
    return [
        (f"peak t={peak_frame}", maps_t_h_w[peak_frame]),
        (f"final t={final_frame}", maps_t_h_w[final_frame]),
    ]


def plot_channel_activation_map_gallery(
    activation_movie: np.ndarray,
    channel_rows: list[dict[str, Any]],
    *,
    title: str,
    labels: np.ndarray | None = None,
    unit_rows: list[dict[str, Any]] | None = None,
    robust: bool = True,
) -> plt.Figure:
    if not channel_rows:
        raise ValueError("No channels supplied for activation-map gallery")

    y = np.asarray(activation_movie, dtype=np.float32)
    rows = channel_rows
    n_rows = len(rows)
    n_cols = 2
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.45 * n_cols, max(1.8 * n_rows, 3.0)),
        squeeze=False,
        constrained_layout=True,
    )
    for r, row in enumerate(rows):
        channel = int(row["channel"])
        maps = _channel_display_maps(y, channel)
        stack = np.stack([activation_map for _label, activation_map in maps])
        if robust:
            vmin, vmax = np.nanpercentile(stack, [1, 99])
        else:
            vmin, vmax = float(np.nanmin(stack)), float(np.nanmax(stack))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = vmax = None

        membership = str(row.get("membership", ""))
        if not membership:
            membership, _ = _channel_membership_status(channel, labels)
        sharpness = row.get("sharpness_score", np.nan)
        ylabel = (
            f"{_channel_label(channel, unit_rows)}\n"
            f"{membership}\n"
            f"sharp={float(sharpness):.1f}"
        )
        for c, (map_label, activation_map) in enumerate(maps):
            ax = axes[r, c]
            ax.imshow(activation_map, cmap=ACTIVATION_MAP_CMAP, vmin=vmin, vmax=vmax)
            ax.set_title(map_label, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=7)

    fig.suptitle(title, fontsize=11)
    return fig


def plot_redundancy_embedding(
    embedding: np.ndarray,
    embedding_name: str,
    corr: np.ndarray,
    thresholds: list[float],
    method: str = REDUNDANCY_LINKAGE_METHOD,
) -> plt.Figure:
    """One panel per threshold: embedding colored by redundancy group, singletons in grey."""
    n = len(thresholds)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.2), constrained_layout=True)
    if n == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab20")
    for ax, threshold in zip(axes, thresholds):
        labels = redundancy_groups_from_corr(corr, threshold, method=method)
        group_ids = sorted(set(labels[labels >= 0]))
        n_redundant = int((labels >= 0).sum())
        singleton_mask = labels < 0
        ax.scatter(
            embedding[singleton_mask, 0], embedding[singleton_mask, 1],
            s=12, color="#cccccc", alpha=0.45, linewidths=0, zorder=1, label=f"singleton ({singleton_mask.sum()})",
        )
        for k, gid in enumerate(group_ids):
            mask = labels == gid
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                s=22, color=cmap(k % 20 / 19), alpha=0.88, linewidths=0, zorder=2,
                label=f"g{k} (n={int(mask.sum())})",
            )
        n_groups = len(group_ids)
        n_singletons = int(singleton_mask.sum())
        n_after_merge = n_groups + n_singletons
        ax.set_title(
            f"{method} linkage, corr > {threshold:.2f}  |  {len(labels)} -> {n_after_merge} units\n"
            f"{n_groups} redundant groups ({n_redundant} units in groups)",
            fontsize=9,
        )
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        if n_groups <= 15:
            ax.legend(markerscale=1.4, fontsize=6, loc="best", framealpha=0.5)
    fig.suptitle(f"{embedding_name} | {method} linkage", fontsize=10)
    return fig


def plot_embedding_labeled_singletons(
    embedding: np.ndarray,
    embedding_name: str,
    labels: np.ndarray,
    singleton_rows: list[dict[str, Any]],
    *,
    title: str,
    unit_rows: list[dict[str, Any]] | None = None,
) -> plt.Figure:
    """Embedding view with the sharpest singleton channels explicitly labeled."""
    emb = np.asarray(embedding, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=int)
    if emb.ndim != 2 or emb.shape[1] < 2:
        raise ValueError(f"Expected embedding shape C x 2+, got {emb.shape}")
    if labels_arr.shape[0] != emb.shape[0]:
        raise ValueError(f"Label length {labels_arr.shape[0]} does not match embedding rows {emb.shape[0]}")

    singleton_channels = np.flatnonzero(labels_arr == -1)
    grouped_channels = np.flatnonzero(labels_arr >= 0)
    excluded_channels = np.flatnonzero(labels_arr == -2)
    highlighted = np.asarray([int(row["channel"]) for row in singleton_rows], dtype=int)

    fig, ax = plt.subplots(figsize=(10.5, 8.5), constrained_layout=True)
    if excluded_channels.size:
        ax.scatter(
            emb[excluded_channels, 0],
            emb[excluded_channels, 1],
            s=10,
            color="#dddddd",
            alpha=0.18,
            linewidths=0,
            label=f"excluded ({excluded_channels.size})",
            zorder=0,
        )
    if grouped_channels.size:
        ax.scatter(
            emb[grouped_channels, 0],
            emb[grouped_channels, 1],
            s=13,
            color="#9fb7d7",
            alpha=0.30,
            linewidths=0,
            label=f"grouped ({grouped_channels.size})",
            zorder=1,
        )
    if singleton_channels.size:
        ax.scatter(
            emb[singleton_channels, 0],
            emb[singleton_channels, 1],
            s=15,
            color="#bdbdbd",
            alpha=0.45,
            linewidths=0,
            label=f"singletons ({singleton_channels.size})",
            zorder=2,
        )

    if highlighted.size:
        sharpness = np.asarray(
            [float(row.get("sharpness_score", np.nan)) for row in singleton_rows],
            dtype=np.float32,
        )
        finite = np.isfinite(sharpness)
        if np.any(finite):
            color_values = sharpness
            scatter = ax.scatter(
                emb[highlighted, 0],
                emb[highlighted, 1],
                s=70,
                c=color_values,
                cmap="viridis",
                edgecolors="black",
                linewidths=0.7,
                alpha=0.95,
                label=f"sharpest labeled ({highlighted.size})",
                zorder=4,
            )
            fig.colorbar(scatter, ax=ax, shrink=0.76, label="spatial sharpness")
        else:
            ax.scatter(
                emb[highlighted, 0],
                emb[highlighted, 1],
                s=70,
                color="#f28e2b",
                edgecolors="black",
                linewidths=0.7,
                alpha=0.95,
                label=f"sharpest labeled ({highlighted.size})",
                zorder=4,
            )

        offsets = [
            (5, 6),
            (7, -8),
            (-26, 6),
            (-28, -8),
            (9, 10),
            (9, -12),
            (-34, 10),
            (-34, -12),
        ]
        for rank, row in enumerate(singleton_rows, start=1):
            channel = int(row["channel"])
            dx, dy = offsets[(rank - 1) % len(offsets)]
            label = f"{rank}: ch {channel}"
            ax.annotate(
                label,
                xy=(float(emb[channel, 0]), float(emb[channel, 1])),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7,
                ha="left" if dx >= 0 else "right",
                va="center",
                arrowprops={"arrowstyle": "-", "color": "0.25", "linewidth": 0.45, "alpha": 0.75},
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.85},
                zorder=5,
            )

    ax.set_title(f"{title}\n{embedding_name}", fontsize=10)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(frameon=False, fontsize=8, loc="best")
    return fig


def select_embedding_audit_groups(
    labels: np.ndarray,
    corr: np.ndarray,
    *,
    threshold: float,
    method: str,
    n_groups: int = EMBEDDING_AUDIT_N_GROUPS,
    min_group_size: int = EMBEDDING_AUDIT_MIN_GROUP_SIZE,
) -> list[int]:
    """Pick a few visually useful groups for embedding-linked activation audits."""
    rows = redundancy_group_rows(corr, labels, threshold=threshold, method=method)
    candidates = [
        row for row in rows
        if int(row["size"]) >= int(min_group_size)
    ]
    if not candidates:
        candidates = rows
    # Prefer large, internally coherent groups; they are usually the easiest
    # groups to visually audit in a compact summary figure.
    candidates.sort(
        key=lambda row: (
            -int(row["size"]),
            -float(row.get("min_pair_corr", np.nan)),
            -float(row.get("min_centroid_corr", np.nan)),
        )
    )
    return [int(row["group_id"]) for row in candidates[: int(n_groups)]]


def plot_embedding_cluster_activation_audit(
    embedding: np.ndarray,
    embedding_name: str,
    corr: np.ndarray,
    labels: np.ndarray,
    activation_movie: np.ndarray,
    *,
    threshold: float,
    method: str = REDUNDANCY_LINKAGE_METHOD,
    group_ids: list[int] | None = None,
    unit_rows: list[dict[str, Any]] | None = None,
    trace_frames: int = VALIDATION_TRACE_FRAMES,
    max_trace_members: int = EMBEDDING_AUDIT_MAX_TRACE_MEMBERS,
    max_map_members: int = EMBEDDING_AUDIT_MAX_MAP_MEMBERS,
    trace_mode: str = "spatial_mean",
    zscore: bool | None = None,
    frame_mode: str = MAP_FRAME_MODE,
) -> plt.Figure:
    """
    Show the base-cluster embedding plus activation trace/map insets.

    The embedding is a navigation aid only; the cluster membership comes from
    the high-dimensional fingerprint correlation labels at `threshold`.
    """
    from matplotlib.patches import ConnectionPatch

    emb = np.asarray(embedding, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    y = np.asarray(activation_movie, dtype=np.float32)
    traces = activation_traces(y, trace_mode=trace_mode)
    trace_label = trace_mode_label(trace_mode)
    if zscore is None:
        zscore = trace_mode_zscore_default(trace_mode)

    if group_ids is None:
        group_ids = select_embedding_audit_groups(
            labels,
            corr,
            threshold=threshold,
            method=method,
            n_groups=EMBEDDING_AUDIT_N_GROUPS,
            min_group_size=EMBEDDING_AUDIT_MIN_GROUP_SIZE,
        )
    group_ids = [int(gid) for gid in group_ids if np.sum(labels == int(gid)) >= 2]
    if not group_ids:
        raise ValueError("No multi-member groups available for embedding activation audit")

    group_rows = {
        int(row["group_id"]): row
        for row in redundancy_group_rows(corr, labels, threshold=threshold, method=method)
    }
    n_selected = min(len(group_ids), EMBEDDING_AUDIT_N_GROUPS)
    group_ids = group_ids[:n_selected]

    fig = plt.figure(figsize=(16, 10), constrained_layout=False)
    ax_embed = fig.add_axes([0.29, 0.16, 0.42, 0.68])
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, max(10, n_selected)))[:n_selected]

    excluded_mask = labels == -2
    singleton_mask = labels == -1
    other_group_mask = labels >= 0
    selected_mask = np.zeros(labels.shape, dtype=bool)
    for gid in group_ids:
        selected_mask |= labels == gid
    other_group_mask &= ~selected_mask

    ax_embed.scatter(
        emb[excluded_mask, 0],
        emb[excluded_mask, 1],
        s=10,
        color="#dddddd",
        alpha=0.22,
        linewidths=0,
        zorder=0,
        label="excluded",
    )
    ax_embed.scatter(
        emb[singleton_mask, 0],
        emb[singleton_mask, 1],
        s=12,
        color="#cfcfcf",
        alpha=0.35,
        linewidths=0,
        zorder=1,
        label="singletons",
    )
    ax_embed.scatter(
        emb[other_group_mask, 0],
        emb[other_group_mask, 1],
        s=16,
        color="#9fb7d7",
        alpha=0.35,
        linewidths=0,
        zorder=2,
        label="other grouped units",
    )

    group_centroids: dict[int, np.ndarray] = {}
    for k, gid in enumerate(group_ids):
        members = np.flatnonzero(labels == gid)
        color = colors[k]
        group_centroid = np.nanmean(emb[members], axis=0)
        group_centroids[gid] = group_centroid
        ax_embed.scatter(
            emb[members, 0],
            emb[members, 1],
            s=42,
            color=color,
            alpha=0.92,
            edgecolors="black",
            linewidths=0.35,
            zorder=4,
        )
        ax_embed.text(
            float(group_centroid[0]),
            float(group_centroid[1]),
            f"g{gid}",
            color="black",
            fontsize=8,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": color, "alpha": 0.9},
            zorder=5,
        )

    n_groups = len(set(labels[labels >= 0]))
    n_singletons = int(np.sum(labels == -1))
    ax_embed.set_title(
        f"{embedding_name}\n{method} linkage, corr > {threshold:.2f}: "
        f"{n_groups + n_singletons} reps ({n_groups} groups + {n_singletons} singletons)",
        fontsize=10,
    )
    ax_embed.set_xlabel("dim 1")
    ax_embed.set_ylabel("dim 2")
    ax_embed.legend(frameon=False, fontsize=7, loc="best")

    panel_positions = [
        (0.03, 0.70, 0.23, 0.24),
        (0.03, 0.40, 0.23, 0.24),
        (0.03, 0.10, 0.23, 0.24),
        (0.74, 0.70, 0.23, 0.24),
        (0.74, 0.40, 0.23, 0.24),
        (0.74, 0.10, 0.23, 0.24),
    ]
    trace_n = min(int(trace_frames), y.shape[0], traces.shape[0])
    trace_x = np.arange(trace_n)

    for k, (gid, panel) in enumerate(zip(group_ids, panel_positions)):
        members = np.flatnonzero(labels == gid)
        color = colors[k]
        x0, y0, w, h = panel
        trace_ax = fig.add_axes([x0, y0 + h * 0.42, w, h * 0.53])
        map_h = h * 0.30
        map_y = y0 + h * 0.04

        member_trace_std = np.nanstd(traces[:trace_n, members], axis=0)
        shown_members = members[np.argsort(member_trace_std)[::-1]][: int(max_trace_members)]
        shown_maps = shown_members[: int(max_map_members)]

        group_traces = traces[:trace_n, members]
        for member in shown_members:
            member_trace = traces[:trace_n, int(member)]
            if zscore:
                member_trace = _zscore_1d(member_trace)
            trace_ax.plot(
                trace_x,
                member_trace,
                color=color,
                linewidth=0.75,
                alpha=0.45,
            )
        centroid_trace = np.nanmean(group_traces, axis=1)
        if zscore:
            centroid_trace = _zscore_1d(centroid_trace)
        trace_ax.plot(trace_x, centroid_trace, color="black", linewidth=1.6, label="group mean")
        row = group_rows.get(gid, {})
        trace_ax.set_title(
            f"g{gid} n={members.size}  min pair={float(row.get('min_pair_corr', np.nan)):.2f} | {trace_label}",
            fontsize=8,
        )
        trace_ax.set_xticks([])
        trace_ax.set_yticks([])
        trace_ax.spines["top"].set_visible(False)
        trace_ax.spines["right"].set_visible(False)
        trace_ax.spines["left"].set_visible(False)
        trace_ax.spines["bottom"].set_visible(False)

        map_frame = select_activation_map_frame(y, channels=members, frame_mode=frame_mode)
        maps: list[tuple[str, np.ndarray]] = [
            (f"mean {map_frame_title(map_frame, frame_mode)}", np.nanmean(y[map_frame, members], axis=0)),
        ]
        for member in shown_maps:
            maps.append((f"ch {int(member)}", y[map_frame, int(member)]))
        stack = np.stack([m for _label, m in maps])
        vmin, vmax = np.nanpercentile(stack, [2, 98])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = vmax = None
        map_count = len(maps)
        gap = 0.004
        map_w = (w - gap * (map_count - 1)) / map_count
        for map_idx, (map_label, activation_map) in enumerate(maps):
            map_ax = fig.add_axes([x0 + map_idx * (map_w + gap), map_y, map_w, map_h])
            map_ax.imshow(activation_map, cmap=ACTIVATION_MAP_CMAP, vmin=vmin, vmax=vmax)
            map_ax.set_title(map_label, fontsize=6)
            map_ax.set_xticks([])
            map_ax.set_yticks([])
            for spine in map_ax.spines.values():
                spine.set_visible(False)

        connection = ConnectionPatch(
            xyA=(float(group_centroids[gid][0]), float(group_centroids[gid][1])),
            coordsA=ax_embed.transData,
            xyB=(0.5, 0.5),
            coordsB=trace_ax.transAxes,
            axesA=ax_embed,
            axesB=trace_ax,
            arrowstyle="->",
            color=color,
            linewidth=1.1,
            alpha=0.9,
            shrinkA=4,
            shrinkB=4,
            mutation_scale=10,
        )
        fig.add_artist(connection)

    fig.suptitle(
        "Embedding-linked base-cluster activation audit "
        f"({trace_label} {'z-scored' if zscore else 'raw'} traces; single-frame maps; "
        "cluster membership from high-dimensional fingerprints)",
        fontsize=12,
    )
    return fig


# %% Step 1: load and summarize BackImage fixation cases
backimage_results = load_backimage_results(BACKIMAGE_RESULTS_PATH)
backimage_summary = summarize_backimage_results(backimage_results)
backimage_summary.head(12) if pd is not None else backimage_summary[:12]


# %% Step 2: select one image/trace and inspect the retinal drive
case = select_backimage_case(
    backimage_results,
    image_key=IMAGE_KEY,
    image_rank=IMAGE_RANK,
    trace_index=TRACE_INDEX,
    max_frames=MAX_FRAMES,
    center_eye_trace=CENTER_EYE_TRACE,
)
fig_case = plot_backimage_case(case)
if SAVE_FIGURES:
    fig_case.savefig(OUT_DIR / f"case_{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.png", dpi=160)
plt.show()


# %% Step 3: load canonical shared twin/readout and source-unit metadata
bundle = load_model_bundle(device=DEVICE, mode=MODEL_MODE)
unit_table = _as_table(bundle.unit_rows)
_write_csv(OUT_DIR / "canonical_shared_twin_unit_table.csv", bundle.unit_rows)
unit_table.head(12) if pd is not None else unit_table[:12]


# %% Step 4: compute or load the full T x C x H x W activation movie
cache_path = activation_cache_path(case)
if LOAD_CACHE_IF_AVAILABLE and cache_path.exists():
    payload = load_activation_cache(cache_path)
    activation_movie = payload["activation_movie"]
    print(f"Loaded activation movie cache: {cache_path}")
else:
    activation_movie = compute_activation_movie(case, bundle, batch_size=BATCH_SIZE)
    if SAVE_ACTIVATION_CACHE:
        save_activation_cache(cache_path, activation_movie, case, bundle)
        print(f"Saved activation movie cache: {cache_path}")

activation_summary = _as_table(activation_summary_rows(activation_movie))
activation_summary


# %% Step 5: sanity plots for the activation movie
for trace_mode in TRACE_PLOT_MODES:
    trace_slug = _safe_slug(trace_mode)
    fig_dist = plot_activation_distributions(activation_movie, trace_mode=trace_mode)
    if SAVE_FIGURES:
        fig_dist.savefig(
            OUT_DIR / f"activation_distributions_{trace_slug}_{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.png",
            dpi=160,
        )
    plt.show()

    fig_eye = plot_activation_eye_trace_alignment(activation_movie, case.eyepos, trace_mode=trace_mode)
    if SAVE_FIGURES:
        fig_eye.savefig(
            OUT_DIR / f"activation_eye_trace_alignment_{trace_slug}_{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.png",
            dpi=160,
        )
    plt.show()

variance_order = channel_variance_rank(activation_movie)
rng = np.random.default_rng(RANDOM_SEED)
active_pool = variance_order[: max(50, min(activation_movie.shape[1], 250))]
random_channels = rng.choice(active_pool, size=min(N_RANDOM_CHANNELS, active_pool.size), replace=False)
for trace_mode in TRACE_PLOT_MODES:
    trace_slug = _safe_slug(trace_mode)
    fig_traces = plot_channel_trace_snippets(
        activation_movie,
        random_channels,
        unit_rows=bundle.unit_rows,
        trace_mode=trace_mode,
    )
    if SAVE_FIGURES:
        fig_traces.savefig(
            OUT_DIR / f"random_channel_traces_{trace_slug}_{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.png",
            dpi=160,
        )
    plt.show()


# %% Step 6: inspect spatial activation maps for active channels
preview_channels = variance_order[: min(6, activation_movie.shape[1])]
preview_frames = np.linspace(0, activation_movie.shape[0] - 1, num=min(8, activation_movie.shape[0]), dtype=int)
fig_maps = plot_channel_map_strip(activation_movie, preview_channels, preview_frames, unit_rows=bundle.unit_rows)
if SAVE_FIGURES:
    fig_maps.savefig(OUT_DIR / f"active_channel_map_strip_{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.png", dpi=180)
plt.show()

sharp_channel_rows = select_sharp_channel_rows(
    activation_movie,
    n_channels=N_SHARP_MAPS_TO_PLOT,
)
_write_csv(
    OUT_DIR / f"sharp_channel_map_examples_{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.csv",
    sharp_channel_rows,
)
if sharp_channel_rows:
    fig_sharp_maps = plot_channel_activation_map_gallery(
        activation_movie,
        sharp_channel_rows,
        title=f"Sharp activation-map examples - {_safe_slug(case.image_key)}",
        unit_rows=bundle.unit_rows,
    )
    if SAVE_FIGURES:
        fig_sharp_maps.savefig(
            OUT_DIR / f"sharp_channel_activation_maps_{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.png",
            dpi=180,
            bbox_inches="tight",
        )
    plt.show()


# %% Step 7: accumulate C x (T*H*W) fingerprints across N_FINGERPRINT_CASES images
# Fingerprints from each case are concatenated along the feature axis so that
# correlation is estimated from many more spatial-temporal samples, making the
# similarity matrix robust to single-image stimulus-drive confounds.
#
# Combined cache: changing N_FINGERPRINT_CASES invalidates it (name encodes N).
# Individual per-case activation movies are cached separately so only new cases
# need model inference when N is increased.
fingerprint_cache_path = OUT_DIR / f"multi_case_fingerprints_n{N_FINGERPRINT_CASES}_{FINGERPRINT_NORMALIZATION}.npz"
fingerprint_case_labels: list[str] = []

if LOAD_CACHE_IF_AVAILABLE and fingerprint_cache_path.exists():
    _fp_data = np.load(fingerprint_cache_path, allow_pickle=True)
    fingerprints = np.asarray(_fp_data["fingerprints"], dtype=np.float32)
    corr = np.asarray(_fp_data["corr"], dtype=np.float32)
    top_pairs = json.loads(str(_fp_data["top_pairs_json"].item()))
    unit_mean_rate = (
        np.asarray(_fp_data["unit_mean_rate"], dtype=np.float32)
        if "unit_mean_rate" in _fp_data
        else None
    )
    print(f"Loaded combined fingerprint cache ({N_FINGERPRINT_CASES} cases): fingerprints={fingerprints.shape}  corr={corr.shape}")
else:
    fingerprint_parts: list[np.ndarray] = []
    _case_mean_rates: list[np.ndarray] = []

    for img_rank in range(N_FINGERPRINT_CASES):
        try:
            fp_case = select_backimage_case(
                backimage_results,
                image_key=None,
                image_rank=img_rank,
                trace_index=0,
                max_frames=MAX_FRAMES,
                center_eye_trace=CENTER_EYE_TRACE,
            )
        except (IndexError, KeyError, ValueError, StopIteration):
            break
        except FileNotFoundError as _e:
            print(f"  skipping rank {img_rank}: {_e}")
            continue
        fp_cache = activation_cache_path(fp_case)
        if LOAD_CACHE_IF_AVAILABLE and fp_cache.exists():
            fp_payload = load_activation_cache(fp_cache)
            fp_movie = fp_payload["activation_movie"]
        else:
            fp_movie = compute_activation_movie(fp_case, bundle, batch_size=BATCH_SIZE)
            if SAVE_ACTIVATION_CACHE:
                save_activation_cache(fp_cache, fp_movie, fp_case, bundle)
        fingerprint_parts.append(make_channel_fingerprints(fp_movie, normalization=FINGERPRINT_NORMALIZATION))
        _case_mean_rates.append(np.asarray(fp_movie, dtype=np.float32).mean(axis=(0, 2, 3)))
        fingerprint_case_labels.append(fp_case.image_key)
        print(f"  case {img_rank}: {fp_case.image_key}  movie={fp_movie.shape}")

    fingerprints = np.concatenate(fingerprint_parts, axis=1)
    unit_mean_rate = np.stack(_case_mean_rates, axis=0).mean(axis=0) if _case_mean_rates else None
    print(f"Fingerprints: {len(fingerprint_parts)} cases -> shape {fingerprints.shape}")
    corr = channel_correlation_from_fingerprints(fingerprints)
    top_pairs = top_correlated_pairs(corr, unit_rows=bundle.unit_rows, n_pairs=N_TOP_PAIRS)
    save_fingerprint_cache(fingerprint_cache_path, fingerprints, corr, top_pairs, unit_mean_rate=unit_mean_rate)
    print(f"Saved combined fingerprint cache: {fingerprint_cache_path}")

top_pairs_table = _as_table(top_pairs)
top_pairs_table


# %% Step 7a: silence detection — exclude units with near-zero mean activation
# Units below SILENCE_RATE_THRESHOLD across all fingerprint cases produce unstable
# z-scored fingerprints (dividing by ~0 std) and corrupt the correlation matrix.
# They are masked from clustering and excluded from the final population spec.
if unit_mean_rate is not None:
    bad_unit_mask = unit_mean_rate < SILENCE_RATE_THRESHOLD
    n_bad = int(np.sum(bad_unit_mask))
    print(
        f"Silence filter (threshold={SILENCE_RATE_THRESHOLD}): "
        f"{n_bad}/{len(bad_unit_mask)} units excluded  "
        f"(mean rate range: {unit_mean_rate.min():.4f} – {unit_mean_rate.max():.4f})"
    )
    if n_bad > 0:
        bad_channels = np.flatnonzero(bad_unit_mask)
        for ch in bad_channels:
            row = bundle.unit_rows[ch] if ch < len(bundle.unit_rows) else {}
            print(f"  excluded ch {ch}: session={row.get('session','?')}, ccnorm={row.get('ccnorm', float('nan')):.3f}, mean_rate={unit_mean_rate[ch]:.4f}")
else:
    bad_unit_mask = np.zeros(corr.shape[0], dtype=bool)
    print("unit_mean_rate not available — no silence filtering applied (re-run Step 7 without cache to populate)")

corr_filtered = corr.copy()
corr_filtered[bad_unit_mask, :] = 0.0
corr_filtered[:, bad_unit_mask] = 0.0


# %% Step 8: compare complete vs average linkage grouping at the same thresholds
grouping_summary_rows, grouping_detail_rows = summarize_redundancy_grouping_labeled(
    "raw_fingerprint_corr",
    corr_filtered,
    thresholds=REDUNDANCY_THRESHOLDS,
    methods=REDUNDANCY_LINKAGE_METHODS,
)
_write_csv(OUT_DIR / "redundancy_linkage_summary.csv", grouping_summary_rows)
_write_csv(OUT_DIR / "redundancy_linkage_group_details.csv", grouping_detail_rows)
grouping_summary_table = _as_table(grouping_summary_rows)
grouping_summary_table


# %% Step 9: visualize similarity structure and suspiciously redundant pairs
fig_corr = plot_correlation_heatmap(corr, channel_order=variance_order)
if SAVE_FIGURES:
    fig_corr.savefig(OUT_DIR / f"fingerprint_corr_heatmap_{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.png", dpi=170)
plt.show()

for trace_mode in TRACE_PLOT_MODES:
    trace_slug = _safe_slug(trace_mode)
    fig_pairs = plot_top_pair_overview(
        activation_movie,
        top_pairs,
        unit_rows=bundle.unit_rows,
        n_pairs=min(8, len(top_pairs)),
        trace_mode=trace_mode,
    )
    if SAVE_FIGURES:
        fig_pairs.savefig(
            OUT_DIR / f"top_correlated_pair_overview_{trace_slug}_{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.png",
            dpi=170,
        )
    plt.show()


# %% Step 10: redundancy atlas - embedding colored by fingerprint-correlation groups
# Similarity = raw fingerprint correlation; grouping uses REDUNDANCY_LINKAGE_METHOD.
# Singletons (no partner above threshold) are grey; redundant groups share a color.
# Adjust REDUNDANCY_THRESHOLDS at the top to explore different merge aggressiveness.
pca_scores = None
try:
    embedding, embedding_name, pca_scores = compute_channel_embedding(fingerprints, run_tsne=RUN_TSNE)
    fig_embed = plot_redundancy_embedding(
        embedding,
        embedding_name,
        corr,
        REDUNDANCY_THRESHOLDS,
        method=REDUNDANCY_LINKAGE_METHOD,
    )
    if SAVE_FIGURES:
        suffix = "tsne" if RUN_TSNE else "pca"
        fig_embed.savefig(
            OUT_DIR
            / f"redundancy_embedding_{REDUNDANCY_LINKAGE_METHOD}_{suffix}_{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.png",
            dpi=170,
        )
    plt.show()

    base_cluster_labels = redundancy_groups_from_corr(
        corr_filtered,
        FIRST_CLUSTER_THRESHOLD,
        method=REDUNDANCY_LINKAGE_METHOD,
    )
    base_cluster_labels[bad_unit_mask] = -2
    for trace_mode in TRACE_PLOT_MODES:
        trace_slug = _safe_slug(trace_mode)
        fig_embed_audit = plot_embedding_cluster_activation_audit(
            embedding,
            embedding_name,
            corr_filtered,
            base_cluster_labels,
            activation_movie,
            threshold=FIRST_CLUSTER_THRESHOLD,
            method=REDUNDANCY_LINKAGE_METHOD,
            unit_rows=bundle.unit_rows,
            trace_mode=trace_mode,
        )
        if SAVE_FIGURES:
            fig_embed_audit.savefig(
                OUT_DIR
                / (
                    f"redundancy_embedding_cluster_activation_audit_{trace_slug}_"
                    f"{REDUNDANCY_LINKAGE_METHOD}_corr{FIRST_CLUSTER_THRESHOLD:.2f}_{suffix}_"
                    f"{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.png"
                ).replace("0.", "0p"),
                dpi=190,
                bbox_inches="tight",
            )
        plt.show()

    base_singleton_rows = select_sharp_channel_rows(
        activation_movie,
        channels=np.flatnonzero(base_cluster_labels == -1),
        labels=base_cluster_labels,
        n_channels=N_SINGLETON_MAPS_TO_PLOT,
    )
    _write_csv(
        OUT_DIR
        / (
            f"base_cluster_singleton_map_examples_"
            f"{REDUNDANCY_LINKAGE_METHOD}_corr{FIRST_CLUSTER_THRESHOLD:.2f}_{suffix}_"
            f"{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.csv"
        ).replace("0.", "0p"),
        base_singleton_rows,
    )
    if base_singleton_rows:
        fig_base_singletons = plot_channel_activation_map_gallery(
            activation_movie,
            base_singleton_rows,
            title=(
                f"Sharp singleton activation maps - "
                f"{REDUNDANCY_LINKAGE_METHOD} corr>{FIRST_CLUSTER_THRESHOLD:.2f}"
            ),
            labels=base_cluster_labels,
            unit_rows=bundle.unit_rows,
        )
        if SAVE_FIGURES:
            fig_base_singletons.savefig(
                OUT_DIR
                / (
                    f"base_cluster_singleton_activation_maps_"
                    f"{REDUNDANCY_LINKAGE_METHOD}_corr{FIRST_CLUSTER_THRESHOLD:.2f}_{suffix}_"
                    f"{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.png"
                ).replace("0.", "0p"),
                dpi=180,
                bbox_inches="tight",
            )
        plt.show()

    for linkage_method in REDUNDANCY_LINKAGE_METHODS:
        if linkage_method == REDUNDANCY_LINKAGE_METHOD:
            continue
        fig_embed_method = plot_redundancy_embedding(
            embedding,
            embedding_name,
            corr,
            REDUNDANCY_THRESHOLDS,
            method=linkage_method,
        )
        if SAVE_FIGURES:
            fig_embed_method.savefig(
                OUT_DIR
                / f"redundancy_embedding_{linkage_method}_{suffix}_{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.png",
                dpi=170,
            )
        plt.show()
except (RuntimeError, ImportError) as exc:
    print(f"Skipping redundancy embedding cell: {exc}")


# %% Step 11: redundancy atlas using PCA-cosine similarity
# PCA-cosine similarity, since t-SNE distances in 2D are NOT meaningful - t-SNE distorts global distances to
# preserve local topology. But the PCA space it operates on carries the same
# neighbourhood structure without that distortion. Cosine similarity in PCA space
# is amplitude-invariant and denoised, making it more robust than raw fingerprint
# correlation for deciding which units are truly redundant.
pca_cosine_sim = None
try:
    if pca_scores is None:
        raise RuntimeError("Run the embedding cell first so pca_scores is available.")
    pca_cosine_sim = pca_cosine_similarity(pca_scores)
    fig_embed_pca = plot_redundancy_embedding(
        embedding,
        f"{embedding_name} - PCA-cosine similarity",
        pca_cosine_sim,
        REDUNDANCY_THRESHOLDS,
        method=REDUNDANCY_LINKAGE_METHOD,
    )
    if SAVE_FIGURES:
        suffix = "tsne" if RUN_TSNE else "pca"
        fig_embed_pca.savefig(
            OUT_DIR
            / f"redundancy_embedding_pcacosine_{REDUNDANCY_LINKAGE_METHOD}_{suffix}_{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.png",
            dpi=170,
        )
    plt.show()

    for linkage_method in REDUNDANCY_LINKAGE_METHODS:
        if linkage_method == REDUNDANCY_LINKAGE_METHOD:
            continue
        fig_embed_pca_method = plot_redundancy_embedding(
            embedding,
            f"{embedding_name} - PCA-cosine similarity",
            pca_cosine_sim,
            REDUNDANCY_THRESHOLDS,
            method=linkage_method,
        )
        if SAVE_FIGURES:
            fig_embed_pca_method.savefig(
                OUT_DIR
                / f"redundancy_embedding_pcacosine_{linkage_method}_{suffix}_{_safe_slug(case.image_key)}_trace{case.trace_index:03d}.png",
                dpi=170,
            )
        plt.show()
except (RuntimeError, ImportError) as exc:
    print(f"Skipping PCA-cosine redundancy embedding cell: {exc}")


# %% Step 12: metric plots across similarity x linkage x threshold
# This is the compact 3 x 2 x 2 comparison:
# - thresholds: REDUNDANCY_THRESHOLDS
# - linkage methods: complete and average
# - similarity matrices: raw fingerprint correlation and PCA-cosine similarity
combined_grouping_summary_table = None
try:
    if pca_cosine_sim is None:
        raise RuntimeError("Run Step 11 first so pca_cosine_sim is available.")
    pca_grouping_summary_rows, pca_grouping_detail_rows = summarize_redundancy_grouping_labeled(
        "pca_cosine",
        pca_cosine_sim,
        thresholds=REDUNDANCY_THRESHOLDS,
        methods=REDUNDANCY_LINKAGE_METHODS,
    )
    combined_grouping_summary_rows = grouping_summary_rows + pca_grouping_summary_rows
    combined_grouping_detail_rows = grouping_detail_rows + pca_grouping_detail_rows
    _write_csv(OUT_DIR / "redundancy_similarity_linkage_summary.csv", combined_grouping_summary_rows)
    _write_csv(OUT_DIR / "redundancy_similarity_linkage_group_details.csv", combined_grouping_detail_rows)

    fig_metrics = plot_redundancy_metric_grid(combined_grouping_summary_rows)
    if SAVE_FIGURES:
        fig_metrics.savefig(OUT_DIR / "redundancy_similarity_linkage_metric_grid.png", dpi=170, bbox_inches="tight")
    plt.show()

    combined_grouping_summary_table = _as_table(combined_grouping_summary_rows)
except RuntimeError as exc:
    print(f"Skipping redundancy metric grid: {exc}")
combined_grouping_summary_table


# %% Step 13: held-out validation for candidate compact twins
# Group memberships are fixed from the training similarity matrix. On held-out
# BackImage cases, each original unit is compared with the mean activation
# fingerprint of its assigned group. This tests whether a proposed merge remains
# redundant on unseen natural-image/eye-trace drive.
#
# Movies are always loaded (needed for the spatial movie audit later).
# Fingerprints are cached separately: loading the combined fingerprint cache
# skips re-concatenation and re-normalization on reruns.
heldout_case_labels: list[str] = []
heldout_movies: list[np.ndarray] = []
heldout_cases: list[BackimageCase] = []
_heldout_fp_cache_path = (
    OUT_DIR
    / f"multi_case_heldout_fingerprints_n{N_HELDOUT_FINGERPRINT_CASES}_start{HELDOUT_START_RANK}_{FINGERPRINT_NORMALIZATION}.npz"
)

for img_rank in range(HELDOUT_START_RANK, HELDOUT_START_RANK + N_HELDOUT_FINGERPRINT_CASES):
    try:
        heldout_case = select_backimage_case(
            backimage_results,
            image_key=None,
            image_rank=img_rank,
            trace_index=0,
            max_frames=MAX_FRAMES,
            center_eye_trace=CENTER_EYE_TRACE,
        )
    except (IndexError, KeyError, ValueError, StopIteration):
        break
    except FileNotFoundError as _e:
        print(f"  skipping rank {img_rank}: {_e}")
        continue
    heldout_cache = activation_cache_path(heldout_case)
    if LOAD_CACHE_IF_AVAILABLE and heldout_cache.exists():
        heldout_payload = load_activation_cache(heldout_cache)
        heldout_movie = heldout_payload["activation_movie"]
    else:
        heldout_movie = compute_activation_movie(heldout_case, bundle, batch_size=BATCH_SIZE)
        if SAVE_ACTIVATION_CACHE:
            save_activation_cache(heldout_cache, heldout_movie, heldout_case, bundle)
    heldout_case_labels.append(heldout_case.image_key)
    heldout_movies.append(heldout_movie)
    heldout_cases.append(heldout_case)
    print(f"  heldout case {img_rank}: {heldout_case.image_key}  movie={heldout_movie.shape}")

if heldout_movies:
    if LOAD_CACHE_IF_AVAILABLE and _heldout_fp_cache_path.exists():
        _heldout_data = np.load(_heldout_fp_cache_path, allow_pickle=True)
        heldout_fingerprints = np.asarray(_heldout_data["fingerprints"], dtype=np.float32)
        print(f"Loaded heldout fingerprint cache ({N_HELDOUT_FINGERPRINT_CASES} cases): {heldout_fingerprints.shape}")
    else:
        heldout_fp_parts = [
            make_channel_fingerprints(m, normalization=FINGERPRINT_NORMALIZATION) for m in heldout_movies
        ]
        heldout_fingerprints = np.concatenate(heldout_fp_parts, axis=1)
        _heldout_fp_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(_heldout_fp_cache_path, fingerprints=heldout_fingerprints.astype(np.float32))
        print(f"Saved heldout fingerprint cache: {_heldout_fp_cache_path}")
    print(f"Held-out fingerprints: {len(heldout_movies)} cases -> shape {heldout_fingerprints.shape}")
else:
    heldout_fingerprints = None
    print("No held-out fingerprints were built.")

validation_summary_rows: list[dict[str, Any]] = []
validation_group_rows: list[dict[str, Any]] = []
validation_member_rows: list[dict[str, Any]] = []

similarity_matrices = {"raw_fingerprint_corr": corr_filtered}
if pca_cosine_sim is not None:
    similarity_matrices["pca_cosine"] = pca_cosine_sim

if heldout_fingerprints is not None and RUN_BROAD_HELDOUT_VALIDATION:
    for spec in REDUNDANCY_VALIDATION_SPECS:
        similarity_name = str(spec["similarity"])
        if similarity_name not in similarity_matrices:
            print(f"Skipping {spec['name']}: missing similarity matrix {similarity_name!r}")
            continue
        labels = redundancy_groups_from_corr(
            similarity_matrices[similarity_name],
            threshold=float(spec["threshold"]),
            method=str(spec["method"]),
        )
        summary_row, group_rows, member_rows = validate_redundancy_candidate(
            str(spec["name"]),
            labels,
            train_fingerprints=fingerprints,
            heldout_fingerprints=heldout_fingerprints,
            spec=spec,
        )
        validation_summary_rows.append(summary_row)
        validation_group_rows.extend(group_rows)
        validation_member_rows.extend(member_rows)

    _write_csv(OUT_DIR / "redundancy_heldout_validation_summary.csv", validation_summary_rows)
    _write_csv(OUT_DIR / "redundancy_heldout_validation_groups.csv", validation_group_rows)
    _write_csv(OUT_DIR / "redundancy_heldout_validation_members.csv", validation_member_rows)

    validation_summary_table = _as_table(validation_summary_rows)
    fig_validation = plot_heldout_validation_summary(validation_summary_rows, validation_member_rows)
    if SAVE_FIGURES:
        fig_validation.savefig(OUT_DIR / "redundancy_heldout_validation_summary.png", dpi=170, bbox_inches="tight")
    plt.show()

    if heldout_movies:
        plot_movie = heldout_movies[0]
        for summary_row in validation_summary_rows:
            version_name = str(summary_row["version"])
            selected_groups = select_validation_groups_to_plot(
                validation_group_rows,
                version_name=version_name,
                n_groups=N_VALIDATION_GROUPS_TO_PLOT,
            )
            if not selected_groups:
                continue
            version_slug = _safe_slug(version_name)
            for trace_mode in TRACE_PLOT_MODES:
                trace_slug = _safe_slug(trace_mode)
                fig_traces = plot_group_trace_overlays(
                    plot_movie,
                    selected_groups,
                    unit_rows=bundle.unit_rows,
                    trace_mode=trace_mode,
                )
                if SAVE_FIGURES:
                    fig_traces.savefig(
                        OUT_DIR / f"heldout_group_trace_overlays_{trace_slug}_{version_slug}.png",
                        dpi=170,
                        bbox_inches="tight",
                    )
                plt.show()

            fig_maps = plot_group_activation_map_panels(plot_movie, selected_groups, unit_rows=bundle.unit_rows)
            if SAVE_FIGURES:
                fig_maps.savefig(OUT_DIR / f"heldout_group_activation_maps_{version_slug}.png", dpi=170, bbox_inches="tight")
            plt.show()
else:
    validation_summary_table = None
    if heldout_fingerprints is not None:
        print(
            "Skipping broad held-out validation "
            "(RUN_BROAD_HELDOUT_VALIDATION=False); Step 14 will still validate the patched candidate."
        )

validation_summary_table


# %% Step 14: patch bad groups using recursive movie-aware sub-clustering
# Strategy: start from complete-linkage fingerprint groups, audit them on held-out
# activation movies, and recursively split groups whose movie responses fail either
# a member-to-centroid gate or a worst-pair gate.  Pairwise failures matter because
# a centroid can hide two internally inconsistent subclusters.

PATCH_BASE_THRESHOLD           = FIRST_CLUSTER_THRESHOLD   # initial fingerprint merge threshold (more aggressive than before)
MOVIE_SPLIT_CENTROID_THRESHOLD = 0.75   # movie centroid below this flags a group for splitting
MOVIE_SPLIT_CASE_FRACTION      = 0.55    # group is flagged if > this fraction of held-out cases fail
MOVIE_SPLIT_PAIRWISE_THRESHOLD = 0.60   # worst member-member corr below this also flags a group
MOVIE_SPLIT_PAIRWISE_CASE_FRACTION = 0.55
MOVIE_SPLIT_SUBCLUSTER_THRESHOLD = 0.75 # sub-clustering threshold within each flagged group
MOVIE_SPLIT_MAX_PASSES         = 4
MOVIE_BLOCK_JACKKNIFE_N_BLOCKS = 4      # generic robustness audit: split each held-out movie into temporal blocks
MOVIE_BLOCK_PAIRWISE_THRESHOLD = 0.50    # repeated weak member-member corr flags unstable groups
MOVIE_BLOCK_PAIRWISE_MIN_BLOCKS = 5
MOVIE_BLOCK_LOO_CENTROID_THRESHOLD = 0.50  # leave-one-member-out centroid corr, repeated across blocks
MOVIE_BLOCK_LOO_CENTROID_MIN_BLOCKS = 4
SECOND_PASS_MERGE_THRESHOLD    = 1.01   # set > 1.0 to disable second-pass merge
                                         # Centroid fingerprints are noise-averaged and systematically
                                         # more correlated than raw channels; 0.80 still over-merges,
                                         # creating super-groups that fail movie validation in Stage 4.
                                         # Stage 1 at 0.65 already handles the correlated-block problem.
PATCHED_VERSION_NAME = (
    f"V1-RR_complete_{PATCH_BASE_THRESHOLD:.2f}"
    f"_moviesplit{MOVIE_SPLIT_CENTROID_THRESHOLD:.2f}"
    f"_pair{MOVIE_SPLIT_PAIRWISE_THRESHOLD:.2f}"
    f"_rec{MOVIE_SPLIT_MAX_PASSES}"
    f"_blockjkP{MOVIE_BLOCK_PAIRWISE_THRESHOLD:.2f}n{MOVIE_BLOCK_PAIRWISE_MIN_BLOCKS}"
    f"L{MOVIE_BLOCK_LOO_CENTROID_THRESHOLD:.2f}n{MOVIE_BLOCK_LOO_CENTROID_MIN_BLOCKS}"
    f"_merge2nd{SECOND_PASS_MERGE_THRESHOLD:.2f}"
).replace("0.", "0p")


def patch_bad_groups(
    base_labels: np.ndarray,
    corr: np.ndarray,
    bad_group_ids: list[int],
    split_threshold: float,
    method: str = "complete",
) -> np.ndarray:
    """Re-cluster each bad group at split_threshold; promote sub-groups to new labels."""
    new_labels = base_labels.copy()
    next_label = int(np.max(new_labels[new_labels >= 0])) + 1 if np.any(new_labels >= 0) else 0

    for group_id in bad_group_ids:
        members = np.flatnonzero(new_labels == group_id)
        # Mark all members as singletons first; sub-groups will override
        new_labels[members] = -1
        if members.size < 2:
            continue
        sub_corr = np.asarray(corr)[np.ix_(members, members)]
        sub_labels = redundancy_groups_from_corr(sub_corr, split_threshold, method=method)
        for sub_id in sorted(set(sub_labels[sub_labels >= 0])):
            sub_members = members[sub_labels == sub_id]
            if sub_members.size >= 2:
                new_labels[sub_members] = next_label
                next_label += 1

    return new_labels


def _bad_group_ids_from_movie_audit(
    group_rows: list[dict[str, Any]],
    *,
    centroid_threshold: float,
    centroid_case_fraction: float,
    pairwise_threshold: float | None,
    pairwise_case_fraction: float,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Flag groups whose held-out movie quality fails centroid or pairwise gates."""
    metrics_by_group: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"centroid": [], "pairwise": []}
    )
    for row in group_rows:
        if str(row.get("metric_space")) != "movie_t_h_w":
            continue
        gid = int(row["group_id"])
        metrics_by_group[gid]["centroid"].append(float(row["min_member_centroid_corr"]))
        metrics_by_group[gid]["pairwise"].append(float(row.get("min_pairwise_member_corr", np.nan)))

    bad_ids: list[int] = []
    summary_rows: list[dict[str, Any]] = []
    for gid, metrics in sorted(metrics_by_group.items()):
        centroid = np.asarray(metrics["centroid"], dtype=np.float32)
        pairwise = np.asarray(metrics["pairwise"], dtype=np.float32)
        centroid_fail_frac = float(np.mean(centroid < float(centroid_threshold))) if centroid.size else 0.0
        if pairwise_threshold is None:
            pairwise_fail_frac = 0.0
            pairwise_min = np.nan
        else:
            finite_pairwise = pairwise[np.isfinite(pairwise)]
            pairwise_fail_frac = (
                float(np.mean(finite_pairwise < float(pairwise_threshold)))
                if finite_pairwise.size
                else 0.0
            )
            pairwise_min = float(np.nanmin(finite_pairwise)) if finite_pairwise.size else np.nan
        fails_centroid = centroid_fail_frac > float(centroid_case_fraction)
        fails_pairwise = (
            pairwise_threshold is not None
            and pairwise_fail_frac > float(pairwise_case_fraction)
        )
        if fails_centroid or fails_pairwise:
            bad_ids.append(gid)
        summary_rows.append(
            {
                "group_id": int(gid),
                "n_cases": int(centroid.size),
                "centroid_fail_frac": centroid_fail_frac,
                "pairwise_fail_frac": pairwise_fail_frac,
                "min_centroid": float(np.nanmin(centroid)) if centroid.size else np.nan,
                "min_pairwise": pairwise_min,
                "fails_centroid": bool(fails_centroid),
                "fails_pairwise": bool(fails_pairwise),
                "flagged": bool(fails_centroid or fails_pairwise),
            }
        )
    return bad_ids, summary_rows


def recursive_movie_split_labels(
    base_labels: np.ndarray,
    split_corr: np.ndarray,
    heldout_movies: list[np.ndarray],
    heldout_case_labels: list[str],
    *,
    version_name: str,
    centroid_threshold: float,
    centroid_case_fraction: float,
    pairwise_threshold: float | None,
    pairwise_case_fraction: float,
    split_threshold: float,
    max_passes: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Repeatedly audit and split failing groups until no movie gates fail."""
    labels = np.asarray(base_labels, dtype=int).copy()
    pass_rows: list[dict[str, Any]] = []
    for pass_idx in range(1, int(max_passes) + 1):
        summary, group_rows, _, _ = audit_redundancy_labels_on_heldout_movies(
            f"{version_name}_pass{pass_idx}",
            labels,
            heldout_movies,
            heldout_case_labels,
        )
        bad_group_ids, gate_rows = _bad_group_ids_from_movie_audit(
            group_rows,
            centroid_threshold=centroid_threshold,
            centroid_case_fraction=centroid_case_fraction,
            pairwise_threshold=pairwise_threshold,
            pairwise_case_fraction=pairwise_case_fraction,
        )
        for row in gate_rows:
            if row["flagged"]:
                row = dict(row)
                row["pass_idx"] = int(pass_idx)
                row["group_size"] = int(np.sum(labels == int(row["group_id"])))
                pass_rows.append(row)

        n_groups = len(set(labels[labels >= 0]))
        n_singletons = int(np.sum(labels == -1))
        n_excluded = int(np.sum(labels == -2))
        print(
            f"\nRecursive movie split pass {pass_idx}: "
            f"{n_groups + n_singletons} reps "
            f"({n_groups} groups + {n_singletons} singletons + {n_excluded} excluded); "
            f"{len(bad_group_ids)} groups flagged"
        )
        if bad_group_ids:
            for gid in bad_group_ids[:20]:
                row = next(r for r in gate_rows if int(r["group_id"]) == int(gid))
                print(
                    f"  group {gid}: size={int(np.sum(labels == gid))}, "
                    f"min_centroid={row['min_centroid']:.3f}, "
                    f"centroid_fail={row['centroid_fail_frac']:.2f}, "
                    f"min_pairwise={row['min_pairwise']:.3f}, "
                    f"pairwise_fail={row['pairwise_fail_frac']:.2f}"
                )
            if len(bad_group_ids) > 20:
                print(f"  ... {len(bad_group_ids) - 20} more flagged groups")
        else:
            break

        next_labels = patch_bad_groups(
            labels,
            split_corr,
            bad_group_ids,
            split_threshold=split_threshold,
            method="complete",
        )
        if np.array_equal(next_labels, labels):
            print("  No label changes after splitting; stopping to avoid an infinite loop.")
            break
        labels = next_labels

    return labels, pass_rows


def compute_representative_fingerprints(
    fingerprints: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """
    One fingerprint per representative: group centroid (mean of members) or singleton channel.

    Returns
    -------
    rep_fp     : (n_reps, fp_dim) float32 — z-scored centroid fingerprint per rep
    rep_source : list of ("group", group_id) or ("singleton", channel_idx)
    """
    group_ids = sorted(set(labels[labels >= 0]))
    singleton_channels = list(np.flatnonzero(labels == -1))
    fp = np.asarray(fingerprints, dtype=np.float32)
    parts: list[np.ndarray] = []
    sources: list[tuple[str, int]] = []
    for gid in group_ids:
        members = np.flatnonzero(labels == gid)
        parts.append(fp[members].mean(axis=0))
        sources.append(("group", int(gid)))
    for ch in singleton_channels:
        parts.append(fp[ch])
        sources.append(("singleton", int(ch)))
    rep_fp = np.stack(parts, axis=0) if parts else np.empty((0, fp.shape[1]), dtype=np.float32)
    return rep_fp, sources


def second_pass_merge(
    labels: np.ndarray,
    fingerprints: np.ndarray,
    threshold: float,
    method: str = "complete",
) -> np.ndarray:
    """
    Re-cluster representative fingerprints at `threshold`, then map merged clusters
    back to a full-channel label array.

    Representatives whose centroids cluster together have all their underlying
    channels folded into one new group label.  Excluded units (label == -2) are
    preserved unchanged.
    """
    rep_fp, rep_sources = compute_representative_fingerprints(fingerprints, labels)
    if rep_fp.shape[0] < 2:
        return labels.copy()

    rep_corr = channel_correlation_from_fingerprints(rep_fp)
    rep_labels = redundancy_groups_from_corr(rep_corr, threshold=threshold, method=method)

    new_labels = labels.copy()
    next_label = int(np.max(new_labels[new_labels >= 0])) + 1 if np.any(new_labels >= 0) else 0

    for new_gid in sorted(set(rep_labels[rep_labels >= 0])):
        rep_indices = np.flatnonzero(rep_labels == new_gid)
        if rep_indices.size < 2:
            continue  # singleton in second-pass rep space; no change needed
        merged_channels: list[int] = []
        for ri in rep_indices:
            kind, source_id = rep_sources[ri]
            if kind == "group":
                merged_channels.extend(np.flatnonzero(labels == source_id).tolist())
            else:
                merged_channels.append(source_id)
        for ch in merged_channels:
            new_labels[ch] = next_label
        next_label += 1

    return new_labels


def representative_members_from_labels(labels: np.ndarray) -> tuple[list[np.ndarray], list[tuple[str, int]]]:
    """Return representative member channel arrays in population-spec order."""
    labels_arr = np.asarray(labels, dtype=int)
    members_by_rep: list[np.ndarray] = []
    sources: list[tuple[str, int]] = []
    for gid in sorted(set(labels_arr[labels_arr >= 0])):
        members = np.flatnonzero(labels_arr == gid)
        if members.size:
            members_by_rep.append(members.astype(int, copy=False))
            sources.append(("group", int(gid)))
    for ch in np.flatnonzero(labels_arr == -1):
        members_by_rep.append(np.asarray([int(ch)], dtype=int))
        sources.append(("singleton", int(ch)))
    return members_by_rep, sources


def merge_representatives_from_corr(
    labels: np.ndarray,
    rep_corr: np.ndarray,
    threshold: float,
    *,
    method: str = COMPRESSION_LINKAGE_METHOD,
) -> np.ndarray:
    """
    Merge existing representatives using a representative-level correlation matrix.

    Existing multi-unit groups are preserved unless their representative is merged
    with another representative. Excluded units remain excluded.
    """
    labels_arr = np.asarray(labels, dtype=int)
    rep_members, _rep_sources = representative_members_from_labels(labels_arr)
    corr_arr = np.asarray(rep_corr, dtype=np.float32)
    if corr_arr.shape != (len(rep_members), len(rep_members)):
        raise ValueError(
            f"Representative corr shape {corr_arr.shape} does not match {len(rep_members)} reps"
        )

    rep_labels = redundancy_groups_from_corr(corr_arr, threshold=float(threshold), method=method)
    new_labels = np.full(labels_arr.shape, -2, dtype=int)
    next_group = 0

    for rep_cluster_id in sorted(set(rep_labels[rep_labels >= 0])):
        rep_indices = np.flatnonzero(rep_labels == rep_cluster_id)
        merged_members = np.concatenate([rep_members[int(idx)] for idx in rep_indices])
        if merged_members.size >= 2:
            new_labels[merged_members] = next_group
            next_group += 1
        elif merged_members.size == 1:
            new_labels[int(merged_members[0])] = -1

    for rep_idx in np.flatnonzero(rep_labels == -1):
        members = rep_members[int(rep_idx)]
        if members.size >= 2:
            new_labels[members] = next_group
            next_group += 1
        elif members.size == 1:
            new_labels[int(members[0])] = -1

    new_labels[labels_arr == -2] = -2
    return new_labels


def representative_corrs_from_movies(
    labels: np.ndarray,
    movies: list[np.ndarray],
    case_labels: list[str],
    *,
    normalization: str = FINGERPRINT_NORMALIZATION,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Compute one representative-correlation matrix per stimulus movie."""
    rep_corrs: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for movie, case_label in zip(movies, case_labels):
        fp = make_channel_fingerprints(movie, normalization=normalization)
        rep_fp, rep_sources = compute_representative_fingerprints(fp, labels)
        rep_corr = channel_correlation_from_fingerprints(rep_fp)
        rep_corrs.append(rep_corr.astype(np.float32, copy=False))
        offdiag = rep_corr[np.triu_indices(rep_corr.shape[0], k=1)] if rep_corr.shape[0] > 1 else np.asarray([])
        rows.append(
            {
                "case": str(case_label),
                "n_representatives": int(rep_corr.shape[0]),
                "n_group_reps": int(sum(kind == "group" for kind, _ in rep_sources)),
                "n_singleton_reps": int(sum(kind == "singleton" for kind, _ in rep_sources)),
                "offdiag_corr_p95": float(np.nanpercentile(offdiag, 95)) if offdiag.size else np.nan,
                "offdiag_corr_p99": float(np.nanpercentile(offdiag, 99)) if offdiag.size else np.nan,
                "offdiag_corr_max": float(np.nanmax(offdiag)) if offdiag.size else np.nan,
            }
        )
    return rep_corrs, rows


def combine_similarity_matrices(
    corr_matrices: list[np.ndarray],
    *,
    mode: str = COMPRESSION_SIMILARITY_MODE,
) -> np.ndarray:
    if not corr_matrices:
        raise ValueError("Need at least one representative-correlation matrix")
    stack = np.stack([np.asarray(corr, dtype=np.float32) for corr in corr_matrices], axis=0)
    if mode == "min":
        combined = np.nanmin(stack, axis=0)
    elif mode == "mean":
        combined = np.nanmean(stack, axis=0)
    else:
        raise ValueError(f"Unknown COMPRESSION_SIMILARITY_MODE={mode!r}")
    combined = np.asarray(combined, dtype=np.float32)
    np.fill_diagonal(combined, 1.0)
    return combined


def _compression_frontier_plot(
    rows: list[dict[str, Any]],
    *,
    title: str,
    target_representatives: int = COMPRESSION_TARGET_REPRESENTATIVES,
) -> plt.Figure:
    if not rows:
        raise ValueError("No compression frontier rows to plot")
    thresholds = np.asarray([float(row["threshold"]) for row in rows], dtype=np.float32)
    n_reps = np.asarray([int(row["n_representatives"]) for row in rows], dtype=np.int32)
    worst = np.asarray([float(row["movie_worst_group_min_centroid_corr"]) for row in rows], dtype=np.float32)
    below_060 = np.asarray([int(row["movie_groups_below_0p60"]) for row in rows], dtype=np.int32)
    below_075 = np.asarray([int(row["movie_groups_below_0p75"]) for row in rows], dtype=np.int32)
    recon = np.asarray([float(row["mean_global_reconstruction_corr"]) for row in rows], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(thresholds, n_reps, "-o", linewidth=1.8, markersize=4)
    ax.axhline(target_representatives, color="0.25", linestyle="--", linewidth=1.0, label=f"target {target_representatives}")
    ax.set_xlabel("representative merge threshold")
    ax.set_ylabel("n representatives")
    ax.set_title("Compression")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.plot(n_reps, worst, "-o", linewidth=1.8, markersize=4)
    ax.axhline(0.75, color="0.35", linestyle="--", linewidth=1.0, label="0.75")
    ax.axhline(0.60, color="0.55", linestyle=":", linewidth=1.0, label="0.60")
    ax.set_xlabel("n representatives")
    ax.set_ylabel("worst movie member-centroid corr")
    ax.set_title("Worst group quality")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.plot(n_reps, below_075, "-o", linewidth=1.6, markersize=4, label="below .75")
    ax.plot(n_reps, below_060, "-o", linewidth=1.6, markersize=4, label="below .60")
    ax.set_xlabel("n representatives")
    ax.set_ylabel("group-case failures")
    ax.set_title("Failure counts")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    ax.plot(n_reps, recon, "-o", linewidth=1.8, markersize=4)
    ax.axhline(COMPRESSION_PROMOTION_MIN_RECONSTRUCTION_CORR, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_xlabel("n representatives")
    ax.set_ylabel("mean pool-expand reconstruction corr")
    ax.set_title("Reconstruction")

    fig.suptitle(title, fontsize=12)
    return fig


patched_summary_table = None
patched_labels = None
patched_group_rows = None
n_patched_groups = n_patched_singletons = n_patched_excluded = n_patched_total = 0

if heldout_movies:
    # ------------------------------------------------------------------
    # Stage 1: fingerprint cluster at PATCH_BASE_THRESHOLD
    # ------------------------------------------------------------------
    base_labels = redundancy_groups_from_corr(
        corr_filtered, threshold=PATCH_BASE_THRESHOLD, method="complete"
    )
    base_labels[bad_unit_mask] = -2
    _n1g = len(set(base_labels[base_labels >= 0]))
    _n1s = int(np.sum(base_labels == -1))
    print(f"Stage 1 (fingerprint cluster @ {PATCH_BASE_THRESHOLD}): {_n1g + _n1s} reps "
          f"({_n1g} groups + {_n1s} singletons + {int(np.sum(base_labels == -2))} excluded)")

    # ------------------------------------------------------------------
    # Stage 2: recursive movie-based split
    # Audit the current labels on held-out activation movies, then sub-cluster
    # any group that repeatedly fails either a member-centroid or pairwise gate.
    # ------------------------------------------------------------------
    if heldout_fingerprints is not None:
        split_corr = channel_correlation_from_fingerprints(heldout_fingerprints)
        split_corr[bad_unit_mask, :] = 0.0
        split_corr[:, bad_unit_mask] = 0.0
        np.fill_diagonal(split_corr, 1.0)
        split_corr_source = "heldout_movie_fingerprints"
    else:
        split_corr = corr_filtered
        split_corr_source = "training_fingerprints"

    print(
        "\nStage 2 (recursive movie split): "
        f"centroid<{MOVIE_SPLIT_CENTROID_THRESHOLD} in >{MOVIE_SPLIT_CASE_FRACTION:.0%} cases, "
        f"pairwise<{MOVIE_SPLIT_PAIRWISE_THRESHOLD} in >{MOVIE_SPLIT_PAIRWISE_CASE_FRACTION:.0%} cases; "
        f"sub-cluster @ {MOVIE_SPLIT_SUBCLUSTER_THRESHOLD} using {split_corr_source}"
    )
    split_labels, recursive_split_rows = recursive_movie_split_labels(
        base_labels,
        split_corr,
        heldout_movies,
        heldout_case_labels,
        version_name=PATCHED_VERSION_NAME,
        centroid_threshold=MOVIE_SPLIT_CENTROID_THRESHOLD,
        centroid_case_fraction=MOVIE_SPLIT_CASE_FRACTION,
        pairwise_threshold=MOVIE_SPLIT_PAIRWISE_THRESHOLD,
        pairwise_case_fraction=MOVIE_SPLIT_PAIRWISE_CASE_FRACTION,
        split_threshold=MOVIE_SPLIT_SUBCLUSTER_THRESHOLD,
        max_passes=MOVIE_SPLIT_MAX_PASSES,
    )
    split_labels[bad_unit_mask] = -2
    _write_csv(
        OUT_DIR / f"recursive_movie_split_flags_{_safe_slug(PATCHED_VERSION_NAME)}.csv",
        recursive_split_rows,
    )
    _n2g = len(set(split_labels[split_labels >= 0]))
    _n2s = int(np.sum(split_labels == -1))
    print(f"  After split: {_n2g + _n2s} reps ({_n2g} groups + {_n2s} singletons)")

    # ------------------------------------------------------------------
    # Stage 3: second-pass merge on representative fingerprints
    # Cluster the group centroids + singletons at the same threshold.
    # This collapses the correlated blocks visible in the post-merge heatmap.
    # ------------------------------------------------------------------
    if SECOND_PASS_MERGE_THRESHOLD <= 1.0:
        merged_labels = second_pass_merge(
            split_labels, fingerprints, threshold=SECOND_PASS_MERGE_THRESHOLD
        )
    else:
        merged_labels = split_labels.copy()
        print("\nStage 3 disabled: SECOND_PASS_MERGE_THRESHOLD > 1.0")
    merged_labels[bad_unit_mask] = -2
    _n3g = len(set(merged_labels[merged_labels >= 0]))
    _n3s = int(np.sum(merged_labels == -1))
    _n3x = int(np.sum(merged_labels == -2))
    print(f"\nStage 3 (second-pass merge @ {SECOND_PASS_MERGE_THRESHOLD}): {_n3g + _n3s} reps "
          f"({_n3g} groups + {_n3s} singletons + {_n3x} excluded)")

    # ------------------------------------------------------------------
    # Stage 4: re-run movie-based split on Stage 3 output when Stage 3 is active.
    # Stage 3 merges centroids in fingerprint space, which can inadvertently
    # re-create bad merges that Stage 2 had already cleaned up.
    # ------------------------------------------------------------------
    if SECOND_PASS_MERGE_THRESHOLD <= 1.0:
        print("\nStage 4 (movie re-validation after second-pass merge)")
        final_labels, recursive_resplit_rows = recursive_movie_split_labels(
            merged_labels,
            split_corr,
            heldout_movies,
            heldout_case_labels,
            version_name=f"{PATCHED_VERSION_NAME}_postmerge",
            centroid_threshold=MOVIE_SPLIT_CENTROID_THRESHOLD,
            centroid_case_fraction=MOVIE_SPLIT_CASE_FRACTION,
            pairwise_threshold=MOVIE_SPLIT_PAIRWISE_THRESHOLD,
            pairwise_case_fraction=MOVIE_SPLIT_PAIRWISE_CASE_FRACTION,
            split_threshold=MOVIE_SPLIT_SUBCLUSTER_THRESHOLD,
            max_passes=MOVIE_SPLIT_MAX_PASSES,
        )
        _write_csv(
            OUT_DIR / f"recursive_movie_resplit_flags_{_safe_slug(PATCHED_VERSION_NAME)}.csv",
            recursive_resplit_rows,
        )
    else:
        print("\nStage 4 skipped: Stage 3 merge is disabled")
        final_labels = merged_labels.copy()
    final_labels[bad_unit_mask] = -2

    # ------------------------------------------------------------------
    # Stage 5: generic blockwise jackknife robustness split
    # Split groups that are unstable across independent temporal blocks of the
    # held-out BackImage movies. This is deliberately not the downstream
    # validation image/condition suite; it tests whether the group is a stable
    # response family under generic natural-drive resampling.
    # ------------------------------------------------------------------
    print(
        "\nStage 5 (generic block-jackknife robustness): "
        f"{MOVIE_BLOCK_JACKKNIFE_N_BLOCKS} blocks/case, "
        f"pairwise<{MOVIE_BLOCK_PAIRWISE_THRESHOLD} in >= {MOVIE_BLOCK_PAIRWISE_MIN_BLOCKS} blocks, "
        f"LOO-centroid<{MOVIE_BLOCK_LOO_CENTROID_THRESHOLD} in >= {MOVIE_BLOCK_LOO_CENTROID_MIN_BLOCKS} blocks"
    )
    block_group_rows, block_member_rows = audit_redundancy_labels_on_movie_blocks(
        PATCHED_VERSION_NAME,
        final_labels,
        heldout_movies,
        heldout_case_labels,
        n_blocks=MOVIE_BLOCK_JACKKNIFE_N_BLOCKS,
    )
    block_bad_group_ids, block_flag_rows = _bad_group_ids_from_block_jackknife_audit(
        block_group_rows,
        pairwise_threshold=MOVIE_BLOCK_PAIRWISE_THRESHOLD,
        pairwise_min_blocks=MOVIE_BLOCK_PAIRWISE_MIN_BLOCKS,
        loo_centroid_threshold=MOVIE_BLOCK_LOO_CENTROID_THRESHOLD,
        loo_centroid_min_blocks=MOVIE_BLOCK_LOO_CENTROID_MIN_BLOCKS,
    )
    for row in block_flag_rows:
        row["group_size"] = int(np.sum(final_labels == int(row["group_id"])))
    _write_csv(
        OUT_DIR / f"block_jackknife_group_quality_{_safe_slug(PATCHED_VERSION_NAME)}.csv",
        block_group_rows,
    )
    _write_csv(
        OUT_DIR / f"block_jackknife_member_quality_{_safe_slug(PATCHED_VERSION_NAME)}.csv",
        block_member_rows,
    )
    _write_csv(
        OUT_DIR / f"block_jackknife_split_flags_{_safe_slug(PATCHED_VERSION_NAME)}.csv",
        block_flag_rows,
    )
    print(f"  {len(block_bad_group_ids)} groups flagged for generic robustness sub-clustering")
    if block_bad_group_ids:
        flagged_rows = [row for row in block_flag_rows if bool(row["flagged"])]
        flagged_rows.sort(
            key=lambda row: (
                -int(row["pairwise_fail_count"]),
                -int(row["loo_centroid_fail_count"]),
                float(row["min_pairwise"]),
                float(row["min_loo_centroid"]),
            )
        )
        for row in flagged_rows[:20]:
            print(
                f"    group {int(row['group_id'])}: size={int(row['group_size'])}, "
                f"pair_fail={int(row['pairwise_fail_count'])}/{int(row['n_blocks'])}, "
                f"loo_fail={int(row['loo_centroid_fail_count'])}/{int(row['n_blocks'])}, "
                f"min_pair={float(row['min_pairwise']):.3f}, "
                f"min_loo={float(row['min_loo_centroid']):.3f}"
            )
        if len(flagged_rows) > 20:
            print(f"    ... {len(flagged_rows) - 20} more flagged groups")

        jackknife_labels = patch_bad_groups(
            final_labels,
            split_corr,
            block_bad_group_ids,
            split_threshold=MOVIE_SPLIT_SUBCLUSTER_THRESHOLD,
            method="complete",
        )
        jackknife_labels[bad_unit_mask] = -2
        if np.array_equal(jackknife_labels, final_labels):
            print("  Block-jackknife split produced no label changes.")
        else:
            final_labels = jackknife_labels
            _n5g = len(set(final_labels[final_labels >= 0]))
            _n5s = int(np.sum(final_labels == -1))
            print(f"  After block-jackknife split: {_n5g + _n5s} reps ({_n5g} groups + {_n5s} singletons)")

    n_patched_groups = len(set(final_labels[final_labels >= 0]))
    n_patched_singletons = int(np.sum(final_labels == -1))
    n_patched_excluded = int(np.sum(final_labels == -2))
    n_patched_total = n_patched_groups + n_patched_singletons
    print(f"Final: {n_patched_total} reps "
          f"({n_patched_groups} groups + {n_patched_singletons} singletons + {n_patched_excluded} excluded)")

    patched_labels = final_labels
    np.save(OUT_DIR / f"patched_labels_{_safe_slug(PATCHED_VERSION_NAME)}.npy", patched_labels)

    # ------------------------------------------------------------------
    # Fingerprint validation on the final labels (diagnostic, no longer
    # drives splitting — that role now belongs to the movie audit above).
    # ------------------------------------------------------------------
    if heldout_fingerprints is not None:
        patched_spec = {
            "similarity": "raw_fingerprint_corr",
            "method": "complete+recursive_movie_split+block_jackknife",
            "threshold": PATCH_BASE_THRESHOLD,
            "movie_centroid_threshold": MOVIE_SPLIT_CENTROID_THRESHOLD,
            "movie_centroid_case_fraction": MOVIE_SPLIT_CASE_FRACTION,
            "movie_pairwise_threshold": MOVIE_SPLIT_PAIRWISE_THRESHOLD,
            "movie_pairwise_case_fraction": MOVIE_SPLIT_PAIRWISE_CASE_FRACTION,
            "movie_split_threshold": MOVIE_SPLIT_SUBCLUSTER_THRESHOLD,
            "movie_split_max_passes": MOVIE_SPLIT_MAX_PASSES,
            "block_jackknife_n_blocks": MOVIE_BLOCK_JACKKNIFE_N_BLOCKS,
            "block_jackknife_pairwise_threshold": MOVIE_BLOCK_PAIRWISE_THRESHOLD,
            "block_jackknife_pairwise_min_blocks": MOVIE_BLOCK_PAIRWISE_MIN_BLOCKS,
            "block_jackknife_loo_centroid_threshold": MOVIE_BLOCK_LOO_CENTROID_THRESHOLD,
            "block_jackknife_loo_centroid_min_blocks": MOVIE_BLOCK_LOO_CENTROID_MIN_BLOCKS,
            "split_corr_source": split_corr_source,
        }
        patched_summary, patched_group_rows, patched_member_rows = validate_redundancy_candidate(
            PATCHED_VERSION_NAME,
            patched_labels,
            train_fingerprints=fingerprints,
            heldout_fingerprints=heldout_fingerprints,
            spec=patched_spec,
        )
        if validation_summary_rows:
            all_summary_rows = validation_summary_rows + [patched_summary]
            all_member_rows = validation_member_rows + patched_member_rows
            all_group_rows = validation_group_rows + patched_group_rows
            _write_csv(OUT_DIR / "redundancy_heldout_validation_summary.csv", all_summary_rows)
            _write_csv(OUT_DIR / "redundancy_heldout_validation_groups.csv", all_group_rows)
            _write_csv(OUT_DIR / "redundancy_heldout_validation_members.csv", all_member_rows)
        else:
            version_slug = _safe_slug(PATCHED_VERSION_NAME)
            all_summary_rows = [patched_summary]
            all_member_rows = patched_member_rows
            all_group_rows = patched_group_rows
            _write_csv(OUT_DIR / f"redundancy_heldout_validation_summary_{version_slug}.csv", all_summary_rows)
            _write_csv(OUT_DIR / f"redundancy_heldout_validation_groups_{version_slug}.csv", all_group_rows)
            _write_csv(OUT_DIR / f"redundancy_heldout_validation_members_{version_slug}.csv", all_member_rows)
        fig_patched = plot_heldout_validation_summary(all_summary_rows, all_member_rows)
        if SAVE_FIGURES:
            fig_patched.savefig(OUT_DIR / "redundancy_heldout_validation_summary_with_patched.png", dpi=170, bbox_inches="tight")
        plt.show()
        patched_summary_table = _as_table(all_summary_rows)
else:
    print("Skipping Step 14: heldout_movies not available.")

patched_summary_table


# %% Step 15: final QC checks and save portable population spec
# Verify coverage, inspect group-size distribution and session balance,
# then write a JSON + NPZ spec that the spatial-info notebook can load
# to reduce the 756-channel activation movie to the selected representatives.

def _build_population_spec(
    labels: np.ndarray,
    unit_rows: list[dict[str, Any]],
    version_name: str,
    *,
    pooling_mode: str = "mean",
    medoid_channels: dict[int, int] | None = None,
    selection_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a serialisable spec for the reduced population.

    Each entry in ``representatives`` lists:
      - rep_idx      : output slot index (0 … N_reps-1)
      - kind         : "group" or "singleton"
      - members      : list of input channel indices belonging to this rep
      - n_members    : group size (1 for singletons)
      - session      : session name(s) — comma-joined for multi-session groups
      - mean_ccnorm  : mean ccnorm across members (nan if unavailable)
    """
    pooling_mode = str(pooling_mode)
    if pooling_mode not in {"mean", "medoid"}:
        raise ValueError(f"Unknown pooling_mode {pooling_mode!r}; expected 'mean' or 'medoid'")
    medoid_channels = {} if medoid_channels is None else {int(k): int(v) for k, v in medoid_channels.items()}
    group_ids = sorted(set(labels[labels >= 0]))
    singleton_ids = list(np.flatnonzero(labels == -1))   # label=-2 = excluded (silent/bad)
    excluded_ids = list(np.flatnonzero(labels == -2))
    reps: list[dict[str, Any]] = []

    for rep_idx, gid in enumerate(group_ids):
        members = list(map(int, np.flatnonzero(labels == gid)))
        if pooling_mode == "medoid":
            selected_channel = int(medoid_channels.get(int(gid), members[0]))
            if selected_channel not in members:
                raise ValueError(f"Medoid channel {selected_channel} is not in group {gid}: {members}")
        else:
            selected_channel = None
        sessions = sorted(set(str(unit_rows[m].get("session", "")) for m in members if m < len(unit_rows)))
        ccnorms = [float(unit_rows[m].get("ccnorm", float("nan"))) for m in members if m < len(unit_rows)]
        mean_ccnorm = float(np.nanmean(ccnorms)) if ccnorms else float("nan")
        selected_row = unit_rows[selected_channel] if selected_channel is not None and selected_channel < len(unit_rows) else {}
        reps.append({
            "rep_idx": rep_idx,
            "kind": "group",
            "group_label": int(gid),
            "members": members,
            "n_members": len(members),
            "pooling_mode": pooling_mode,
            "selected_channel": selected_channel,
            "selected_session": str(selected_row.get("session", "")) if selected_row else None,
            "selected_source_unit_index": (
                int(selected_row["source_unit_index"])
                if selected_row and "source_unit_index" in selected_row
                else None
            ),
            "selected_ccnorm": (
                float(selected_row.get("ccnorm", float("nan")))
                if selected_row
                else float("nan")
            ),
            "sessions": sessions,
            "mean_ccnorm": mean_ccnorm,
        })

    for rep_idx, ch in enumerate(singleton_ids, start=len(group_ids)):
        row = unit_rows[ch] if ch < len(unit_rows) else {}
        reps.append({
            "rep_idx": rep_idx,
            "kind": "singleton",
            "group_label": -1,
            "members": [int(ch)],
            "n_members": 1,
            "pooling_mode": "singleton",
            "selected_channel": int(ch),
            "selected_session": str(row.get("session", "")),
            "selected_source_unit_index": (
                int(row["source_unit_index"])
                if "source_unit_index" in row
                else None
            ),
            "selected_ccnorm": float(row.get("ccnorm", float("nan"))),
            "sessions": [str(row.get("session", ""))],
            "mean_ccnorm": float(row.get("ccnorm", float("nan"))),
        })

    return {
        "version": version_name,
        "pooling_mode": pooling_mode,
        "selection_metadata": dict(selection_metadata or {}),
        "n_input_channels": int(labels.size),
        "n_representatives": len(reps),
        "n_groups": len(group_ids),
        "n_singletons": len(singleton_ids),
        "n_excluded": len(excluded_ids),
        "excluded_channels": excluded_ids,
        "representatives": reps,
    }


def save_population_spec(
    labels: np.ndarray,
    unit_rows: list[dict[str, Any]],
    version_name: str,
    out_dir: Path,
    *,
    pooling_mode: str = "mean",
    medoid_channels: dict[int, int] | None = None,
    selection_metadata: dict[str, Any] | None = None,
) -> Path:
    spec = _build_population_spec(
        labels,
        unit_rows,
        version_name,
        pooling_mode=pooling_mode,
        medoid_channels=medoid_channels,
        selection_metadata=selection_metadata,
    )
    slug = _safe_slug(version_name)
    json_path = out_dir / f"population_spec_{slug}.json"
    _save_json(json_path, spec)

    # Compact membership matrix: shape (n_reps, n_input_channels), float32 for easy matmul
    n_reps = spec["n_representatives"]
    n_ch = int(labels.size)
    membership = np.zeros((n_reps, n_ch), dtype=np.float32)
    cluster_membership = np.zeros((n_reps, n_ch), dtype=np.float32)
    for rep in spec["representatives"]:
        for ch in rep["members"]:
            cluster_membership[rep["rep_idx"], ch] = 1.0 / rep["n_members"]
        if spec["pooling_mode"] == "medoid" and rep["kind"] == "group":
            membership[rep["rep_idx"], int(rep["selected_channel"])] = 1.0
        else:
            for ch in rep["members"]:
                membership[rep["rep_idx"], ch] = 1.0 / rep["n_members"]  # pre-normalised for mean pooling

    npz_path = out_dir / f"population_spec_{slug}.npz"
    np.savez_compressed(
        npz_path,
        labels=labels,
        membership=membership,
        cluster_membership=cluster_membership,
        pooling_mode=np.asarray(spec["pooling_mode"]),
    )
    print(f"Population spec saved:\n  {json_path}\n  {npz_path}")
    return json_path


def select_medoid_channels_from_member_rows(
    labels: np.ndarray,
    member_rows: list[dict[str, Any]],
    unit_rows: list[dict[str, Any]],
    *,
    metric_space: str = MEDOID_SELECTION_METRIC_SPACE,
    version_name: str | None = None,
) -> tuple[dict[int, int], list[dict[str, Any]]]:
    """
    Pick one actual channel per multi-member group.

    The preferred medoid maximizes worst-case leave-one-out centroid correlation
    across stimulus cases, then median leave-one-out correlation, then ccnorm.
    """
    labels_arr = np.asarray(labels, dtype=int)
    rows_by_group_member: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    metric_versions: set[str] = set()
    for row in member_rows:
        if str(row.get("metric_space")) != str(metric_space):
            continue
        metric_versions.add(str(row.get("version")))
        if version_name is not None and str(row.get("version")) != str(version_name):
            continue
        rows_by_group_member[(int(row["group_id"]), int(row["channel"]))].append(row)
    if version_name is None and len(metric_versions) > 1:
        raise ValueError(
            "Medoid selection member_rows contain multiple versions; pass version_name=... "
            f"to avoid mixing metrics across candidates: {sorted(metric_versions)}"
        )

    medoids: dict[int, int] = {}
    detail_rows: list[dict[str, Any]] = []
    for group_id in sorted(set(labels_arr[labels_arr >= 0])):
        members = np.flatnonzero(labels_arr == group_id)
        candidates: list[dict[str, Any]] = []
        for member in members:
            channel = int(member)
            ccnorm = (
                float(unit_rows[channel].get("ccnorm", float("nan")))
                if channel < len(unit_rows)
                else float("nan")
            )
            rows = rows_by_group_member.get((int(group_id), channel), [])
            loo = np.asarray(
                [float(row.get("leave_one_out_centroid_corr", np.nan)) for row in rows],
                dtype=np.float32,
            )
            centroid = np.asarray(
                [float(row.get("member_centroid_corr", np.nan)) for row in rows],
                dtype=np.float32,
            )
            finite_loo = loo[np.isfinite(loo)]
            finite_centroid = centroid[np.isfinite(centroid)]
            candidates.append(
                {
                    "group_id": int(group_id),
                    "channel": channel,
                    "n_members": int(members.size),
                    "n_cases": int(len(rows)),
                    "worst_leave_one_out_centroid_corr": (
                        float(np.nanmin(finite_loo)) if finite_loo.size else np.nan
                    ),
                    "median_leave_one_out_centroid_corr": (
                        float(np.nanmedian(finite_loo)) if finite_loo.size else np.nan
                    ),
                    "worst_member_centroid_corr": (
                        float(np.nanmin(finite_centroid)) if finite_centroid.size else np.nan
                    ),
                    "median_member_centroid_corr": (
                        float(np.nanmedian(finite_centroid)) if finite_centroid.size else np.nan
                    ),
                    "ccnorm": ccnorm,
                }
            )

        def _score(row: dict[str, Any]) -> tuple[float, float, float, float, float, int]:
            return (
                float(row["worst_leave_one_out_centroid_corr"]) if np.isfinite(float(row["worst_leave_one_out_centroid_corr"])) else -np.inf,
                float(row["median_leave_one_out_centroid_corr"]) if np.isfinite(float(row["median_leave_one_out_centroid_corr"])) else -np.inf,
                float(row["worst_member_centroid_corr"]) if np.isfinite(float(row["worst_member_centroid_corr"])) else -np.inf,
                float(row["median_member_centroid_corr"]) if np.isfinite(float(row["median_member_centroid_corr"])) else -np.inf,
                float(row["ccnorm"]) if np.isfinite(float(row["ccnorm"])) else -np.inf,
                -int(row["channel"]),
            )

        selected = max(candidates, key=_score)
        medoids[int(group_id)] = int(selected["channel"])
        for row in candidates:
            out = dict(row)
            out["selected"] = bool(int(row["channel"]) == int(selected["channel"]))
            out["selection_metric_space"] = str(metric_space)
            detail_rows.append(out)

    return medoids, detail_rows


def save_medoid_population_variant(
    *,
    labels: np.ndarray,
    unit_rows: list[dict[str, Any]],
    version_name: str,
    movies: list[np.ndarray],
    case_labels: list[str],
    out_dir: Path,
    member_rows: list[dict[str, Any]] | None = None,
    metric_space: str = MEDOID_SELECTION_METRIC_SPACE,
) -> tuple[Path | None, list[dict[str, Any]], dict[str, Any] | None]:
    """Save and audit a one-real-unit-per-cluster medoid population spec."""
    if not SAVE_MEDOID_POPULATION_SPECS:
        return None, [], None
    if not movies:
        print(f"Skipping medoid population for {version_name}: no movies available for medoid selection.")
        return None, [], None

    source_member_rows = member_rows
    selection_audit_version = version_name
    if source_member_rows is None:
        selection_audit_version = f"{version_name}_medoid_selection"
        _, _, source_member_rows, _ = audit_redundancy_labels_on_heldout_movies(
            selection_audit_version,
            labels,
            movies,
            case_labels,
        )
    else:
        metric_versions = sorted(
            {
                str(row.get("version"))
                for row in source_member_rows
                if str(row.get("metric_space")) == str(metric_space)
            }
        )
        if version_name in metric_versions:
            selection_audit_version = version_name
        elif len(metric_versions) == 1:
            selection_audit_version = metric_versions[0]
        elif len(metric_versions) > 1:
            raise ValueError(
                "Medoid selection received multiple metric versions. "
                f"Filter member_rows or pass rows for one version only: {metric_versions}"
            )

    medoid_channels, medoid_detail_rows = select_medoid_channels_from_member_rows(
        labels,
        source_member_rows,
        unit_rows,
        metric_space=metric_space,
        version_name=selection_audit_version,
    )
    medoid_version = f"{version_name}_medoid"
    medoid_slug = _safe_slug(medoid_version)
    _write_csv(out_dir / f"medoid_selection_{medoid_slug}.csv", medoid_detail_rows)

    selection_metadata = {
        "selection": "cluster_medoid",
        "selection_metric_space": metric_space,
        "selection_rule": (
            "max worst leave-one-out centroid corr across stimulus cases; "
            "tie-break median LOO, member-centroid corr, ccnorm"
        ),
        "base_version": version_name,
        "selection_audit_version": selection_audit_version,
        "n_selection_cases": int(len(movies)),
        "selection_cases": [str(label) for label in case_labels],
    }
    spec_path = save_population_spec(
        labels,
        unit_rows,
        medoid_version,
        out_dir,
        pooling_mode="medoid",
        medoid_channels=medoid_channels,
        selection_metadata=selection_metadata,
    )

    audit_summary: dict[str, Any] | None = None
    if RUN_MEDOID_POPULATION_AUDITS:
        mean_membership = population_membership_from_labels(labels, pooling_mode="mean")
        medoid_membership = population_membership_from_labels(
            labels,
            pooling_mode="medoid",
            medoid_channels=medoid_channels,
        )
        mean_summary, mean_recon_rows = audit_population_membership_on_movies(
            f"{version_name}_mean_membership",
            labels,
            mean_membership,
            movies,
            case_labels,
            pooling_mode="mean",
        )
        medoid_summary, medoid_recon_rows = audit_population_membership_on_movies(
            medoid_version,
            labels,
            medoid_membership,
            movies,
            case_labels,
            pooling_mode="medoid",
        )
        audit_summary = {
            "version": medoid_version,
            "base_version": version_name,
            "n_representatives": int(medoid_membership.shape[0]),
            "mean_pool_reconstruction_corr": mean_summary["mean_global_reconstruction_corr"],
            "medoid_pool_reconstruction_corr": medoid_summary["mean_global_reconstruction_corr"],
            "delta_reconstruction_corr_medoid_minus_mean": (
                float(medoid_summary["mean_global_reconstruction_corr"])
                - float(mean_summary["mean_global_reconstruction_corr"])
            ),
            "mean_pool_min_reconstruction_corr": mean_summary["min_global_reconstruction_corr"],
            "medoid_pool_min_reconstruction_corr": medoid_summary["min_global_reconstruction_corr"],
            "mean_pool_mean_abs_ssi_delta": mean_summary["mean_abs_expanded_minus_full_ssi"],
            "medoid_pool_mean_abs_ssi_delta": medoid_summary["mean_abs_expanded_minus_full_ssi"],
        }
        _write_csv(out_dir / f"medoid_membership_audit_summary_{medoid_slug}.csv", [audit_summary])
        _write_csv(
            out_dir / f"medoid_membership_audit_reconstruction_{medoid_slug}.csv",
            mean_recon_rows + medoid_recon_rows,
        )
        print(
            f"Medoid population saved for {version_name}: "
            f"mean recon={float(mean_summary['mean_global_reconstruction_corr']):.3f}, "
            f"medoid recon={float(medoid_summary['mean_global_reconstruction_corr']):.3f}"
        )

    return spec_path, medoid_detail_rows, audit_summary


def plot_population_qc(
    labels: np.ndarray,
    unit_rows: list[dict[str, Any]],
    version_name: str,
    patched_group_rows: list[dict[str, Any]],
) -> plt.Figure:
    group_ids = sorted(set(labels[labels >= 0]))
    group_sizes = [int(np.sum(labels == gid)) for gid in group_ids]
    singleton_ids = np.flatnonzero(labels == -1)  # label=-2 = excluded, not a representative
    all_rep_ccnorm = []
    for gid in group_ids:
        members = np.flatnonzero(labels == gid)
        ccnorms = [float(unit_rows[m].get("ccnorm", float("nan"))) for m in members if m < len(unit_rows)]
        all_rep_ccnorm.append(float(np.nanmean(ccnorms)) if ccnorms else float("nan"))
    for ch in singleton_ids:
        all_rep_ccnorm.append(float(unit_rows[ch].get("ccnorm", float("nan"))) if ch < len(unit_rows) else float("nan"))
    orig_ccnorm = [float(row.get("ccnorm", float("nan"))) for row in unit_rows]

    sessions_orig = [str(row.get("session", "unknown")) for row in unit_rows]
    session_names = sorted(set(sessions_orig))
    orig_counts = [sessions_orig.count(s) for s in session_names]

    rep_sessions: list[str] = []
    for gid in group_ids:
        members = np.flatnonzero(labels == gid)
        # Assign rep to the most common session in the group
        sess_list = [sessions_orig[m] for m in members if m < len(sessions_orig)]
        dominant = max(set(sess_list), key=sess_list.count) if sess_list else "unknown"
        rep_sessions.append(dominant)
    for ch in singleton_ids:
        rep_sessions.append(sessions_orig[ch] if ch < len(sessions_orig) else "unknown")
    rep_counts = [rep_sessions.count(s) for s in session_names]

    heldout_group_min = [float(row["heldout_min_centroid_corr"]) for row in patched_group_rows]

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), constrained_layout=True)
    axes = axes.ravel()

    # 1. Group size distribution
    max_size = max(group_sizes) if group_sizes else 1
    size_bins = np.arange(1.5, max_size + 1.5)
    axes[0].hist(group_sizes, bins=size_bins, edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("group size (members)")
    axes[0].set_ylabel("number of groups")
    axes[0].set_title(f"Group sizes  (n_groups={len(group_ids)}, n_singletons={singleton_ids.size})")

    # 2. Per-session unit counts: original vs reduced
    x = np.arange(len(session_names))
    w = 0.35
    axes[1].bar(x - w / 2, orig_counts, w, label=f"original ({len(unit_rows)})", alpha=0.8)
    axes[1].bar(x + w / 2, rep_counts, w, label=f"{version_name} ({len(all_rep_ccnorm)})", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(session_names, rotation=25, ha="right", fontsize=7)
    axes[1].set_ylabel("units")
    axes[1].set_title("Session balance: original vs reduced")
    axes[1].legend(frameon=False, fontsize=7)

    # 3. ccnorm distribution: original vs representative
    bins = np.linspace(0.0, 1.0, 31)
    axes[2].hist(orig_ccnorm, bins=bins, histtype="step", linewidth=2.0, label=f"original ({len(orig_ccnorm)})", color="C0")
    axes[2].hist(all_rep_ccnorm, bins=bins, histtype="step", linewidth=2.0, label=f"representatives ({len(all_rep_ccnorm)})", color="C1")
    axes[2].set_xlabel("ccnorm")
    axes[2].set_ylabel("units")
    axes[2].set_title("ccnorm: original vs representative")
    axes[2].legend(frameon=False, fontsize=8)

    # 4. Heldout min centroid corr distribution for groups
    axes[3].hist(heldout_group_min, bins=np.linspace(0.0, 1.0, 31), edgecolor="white", linewidth=0.5)
    axes[3].axvline(0.75, color="0.3", linestyle="--", linewidth=1.2, label="0.75")
    axes[3].set_xlabel("heldout min centroid corr")
    axes[3].set_ylabel("groups")
    axes[3].set_title("Held-out group quality (should be clean above 0.75)")
    axes[3].legend(frameon=False, fontsize=8)

    # 5. Cumulative ccnorm (original vs rep)
    sorted_orig = np.sort([v for v in orig_ccnorm if not np.isnan(v)])
    sorted_rep = np.sort([v for v in all_rep_ccnorm if not np.isnan(v)])
    axes[4].plot(sorted_orig, np.linspace(0, 1, sorted_orig.size), label="original", linewidth=2.0, color="C0")
    axes[4].plot(sorted_rep, np.linspace(0, 1, sorted_rep.size), label="representatives", linewidth=2.0, color="C1")
    axes[4].set_xlabel("ccnorm")
    axes[4].set_ylabel("CDF")
    axes[4].set_title("ccnorm CDF")
    axes[4].legend(frameon=False, fontsize=8)

    # 6. Text summary
    axes[5].axis("off")
    summary_lines = [
        f"Version:  {version_name}",
        f"Input channels:  {int(labels.size)}",
        f"Representatives: {len(all_rep_ccnorm)}",
        f"  • groups:      {len(group_ids)}",
        f"  • singletons:  {int(singleton_ids.size)}",
        f"Largest group:   {max(group_sizes) if group_sizes else 'n/a'}",
        f"Mean group size: {np.mean(group_sizes):.1f}" if group_sizes else "Mean group size: n/a",
        f"Heldout worst group min: {min(heldout_group_min):.3f}" if heldout_group_min else "",
        f"Groups below 0.75:  {sum(v < 0.75 for v in heldout_group_min)}",
    ]
    axes[5].text(0.05, 0.95, "\n".join(summary_lines), transform=axes[5].transAxes,
                 va="top", ha="left", fontsize=9, family="monospace")
    fig.suptitle(f"Final QC — {version_name}", fontsize=11)
    return fig


if patched_labels is not None and patched_group_rows is not None:
    # Sanity checks
    assert patched_labels.size == len(bundle.unit_rows), (
        f"Label array size {patched_labels.size} != n_units {len(bundle.unit_rows)}"
    )
    assert n_patched_singletons + n_patched_groups == n_patched_total, (
        "Representative count mismatch after patching"
    )
    print(f"Coverage check passed: all {patched_labels.size} input channels accounted for.")

    spec_path = save_population_spec(patched_labels, bundle.unit_rows, PATCHED_VERSION_NAME, OUT_DIR)

    fig_qc = plot_population_qc(patched_labels, bundle.unit_rows, PATCHED_VERSION_NAME, patched_group_rows)
    if SAVE_FIGURES:
        fig_qc.savefig(OUT_DIR / f"population_qc_{_safe_slug(PATCHED_VERSION_NAME)}.png", dpi=170, bbox_inches="tight")
    plt.show()
else:
    spec_path = None
    print("Skipping QC: patched_labels not available (run Step 14 first).")

spec_path

# %% Step 16: spatial/movie audit for the saved compact population
# Fingerprint validation asks whether grouped channels have similar response
# fingerprints.  This downstream audit asks the stricter question needed for
# SSI-style analyses: does replacing each group with its rate-map centroid
# reconstruct the held-out activation movies and their spatial information?

spatial_movie_audit_summary_table = None

if patched_labels is not None and heldout_movies and heldout_case_labels:
    (
        spatial_movie_summary,
        spatial_movie_group_rows,
        spatial_movie_member_rows,
        spatial_movie_reconstruction_rows,
    ) = audit_redundancy_labels_on_heldout_movies(
        PATCHED_VERSION_NAME,
        patched_labels,
        heldout_movies,
        heldout_case_labels,
    )
    slug = _safe_slug(PATCHED_VERSION_NAME)
    _write_csv(OUT_DIR / f"spatial_movie_audit_summary_{slug}.csv", [spatial_movie_summary])
    _write_csv(OUT_DIR / f"spatial_movie_audit_groups_{slug}.csv", spatial_movie_group_rows)
    _write_csv(OUT_DIR / f"spatial_movie_audit_members_{slug}.csv", spatial_movie_member_rows)
    _write_csv(OUT_DIR / f"spatial_movie_audit_reconstruction_{slug}.csv", spatial_movie_reconstruction_rows)

    movie_rows = [
        row for row in spatial_movie_group_rows
        if str(row.get("metric_space")) == "movie_t_h_w"
    ]
    mean_map_rows = [
        row for row in spatial_movie_group_rows
        if str(row.get("metric_space")) == "mean_map_h_w"
    ]
    for threshold in (0.60, 0.75):
        failing_group_ids = {
            int(row["group_id"])
            for row in movie_rows
            if float(row["min_member_centroid_corr"]) < threshold
        }
        extra_reps = 0
        for gid in failing_group_ids:
            extra_reps += int(np.sum(patched_labels == gid)) - 1
        print(
            f"Movie audit split-to-singletons estimate @ {threshold:.2f}: "
            f"{len(failing_group_ids)} groups, +{extra_reps} reps, "
            f"estimated n={int(spatial_movie_summary.get('n_representatives', 0) or len(set(patched_labels[patched_labels >= 0])) + np.sum(patched_labels == -1) + extra_reps)}"
        )

    print("Spatial/movie audit summary:")
    print(spatial_movie_summary)
    spatial_movie_audit_summary_table = _as_table([spatial_movie_summary])

    if SAVE_FIGURES and movie_rows:
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0), constrained_layout=True)
        movie_min = np.asarray([float(row["min_member_centroid_corr"]) for row in movie_rows], dtype=np.float32)
        movie_size = np.asarray([int(row["size"]) for row in movie_rows], dtype=np.int32)
        axes[0].hist(movie_min, bins=np.linspace(-0.2, 1.0, 49), edgecolor="white", linewidth=0.5)
        axes[0].axvline(0.75, color="0.3", linestyle="--", linewidth=1.0)
        axes[0].set_xlabel("movie min member-centroid corr")
        axes[0].set_ylabel("groups x heldout cases")
        axes[0].set_title("Held-out movie group quality")

        axes[1].scatter(movie_size, movie_min, s=14, alpha=0.45)
        axes[1].axhline(0.75, color="0.3", linestyle="--", linewidth=1.0)
        axes[1].set_xlabel("group size")
        axes[1].set_ylabel("movie min member-centroid corr")
        axes[1].set_title("Quality vs group size")

        if mean_map_rows:
            mean_map_by_key = {
                (str(row["case"]), int(row["group_id"])): float(row["min_member_centroid_corr"])
                for row in mean_map_rows
            }
            paired_x = []
            paired_y = []
            for row in movie_rows:
                key = (str(row["case"]), int(row["group_id"]))
                if key in mean_map_by_key:
                    paired_x.append(float(row["min_member_centroid_corr"]))
                    paired_y.append(float(mean_map_by_key[key]))
            axes[2].scatter(paired_x, paired_y, s=14, alpha=0.45)
            axes[2].plot([-0.2, 1.0], [-0.2, 1.0], color="0.3", linestyle="--", linewidth=0.8)
            axes[2].set_xlabel("movie min corr")
            axes[2].set_ylabel("mean-map min corr")
            axes[2].set_title("Movie vs mean-map quality")
        else:
            axes[2].axis("off")

        fig.suptitle(f"Spatial/movie audit - {PATCHED_VERSION_NAME}", fontsize=11)
        fig.savefig(OUT_DIR / f"spatial_movie_audit_summary_{slug}.png", dpi=170, bbox_inches="tight")
        plt.show()
else:
    print("Skipping spatial/movie audit: patched labels or held-out movies are unavailable.")

spatial_movie_audit_summary_table


# %% Step 17: reduced-channel fingerprint correlation heatmap for the vetted population
# Shows between-representative similarity using the raw fingerprint corr matrix.
# Upper-left = groups (sorted by group id), lower-right = singletons.
# Ideally all off-diagonal entries are low after merging; any remaining hot spots
# indicate candidate further merges.
if patched_labels is not None:
    fig_reduced_corr = plot_reduced_corr_heatmap(
        corr_filtered,
        patched_labels,
        title=f"Post-merge representative correlation — {PATCHED_VERSION_NAME}",
    )
    if SAVE_FIGURES:
        fig_reduced_corr.savefig(
            OUT_DIR / f"reduced_corr_heatmap_{_safe_slug(PATCHED_VERSION_NAME)}.png",
            dpi=170,
            bbox_inches="tight",
        )
    plt.show()
else:
    print("Skipping reduced heatmap: patched_labels not available.")

# %% Step 18: FixRSVP stimulus audit for the vetted population
# BackImage drove the construction/validation above. This cell asks whether the
# same V1-RR groups remain coherent when driven by the visually diverse FixRSVP
# image stream. It is diagnostic only: no clusters are split here.
fixrsvp_audit_summary_table = None
fixrsvp_movie_group_table = None

if RUN_FIXRSVP_AUDIT and patched_labels is not None:
    fixrsvp_cache = fixrsvp_activation_cache_path(
        stimulus_type=FIXRSVP_STIMULUS_TYPE,
        frames_per_image=FIXRSVP_FRAMES_PER_IMAGE,
        num_frames=FIXRSVP_NUM_FRAMES,
        frame=FIXRSVP_FRAME,
    )
    if LOAD_CACHE_IF_AVAILABLE and fixrsvp_cache.exists():
        fixrsvp_payload = load_activation_cache(fixrsvp_cache)
        fixrsvp_activation_movie = fixrsvp_payload["activation_movie"]
        fixrsvp_metadata = fixrsvp_payload["metadata"]
        print(f"Loaded FixRSVP activation movie cache: {fixrsvp_cache}")
    else:
        fixrsvp_activation_movie, fixrsvp_metadata = compute_fixrsvp_activation_movie(
            bundle,
            stimulus_type=FIXRSVP_STIMULUS_TYPE,
            frames_per_image=FIXRSVP_FRAMES_PER_IMAGE,
            num_frames=FIXRSVP_NUM_FRAMES,
            frame=FIXRSVP_FRAME,
            batch_size=BATCH_SIZE,
        )
        if SAVE_ACTIVATION_CACHE:
            save_fixrsvp_activation_cache(fixrsvp_cache, fixrsvp_activation_movie, fixrsvp_metadata)
            print(f"Saved FixRSVP activation movie cache: {fixrsvp_cache}")

    fixrsvp_label = (
        f"FixRSVP_{FIXRSVP_STIMULUS_TYPE}_fpi{FIXRSVP_FRAMES_PER_IMAGE}"
        f"_frames{FIXRSVP_NUM_FRAMES}"
        f"_{'dynamic' if FIXRSVP_FRAME is None else f'frame{FIXRSVP_FRAME:03d}'}"
    )
    fixrsvp_slug = _safe_slug(f"{PATCHED_VERSION_NAME}_{fixrsvp_label}")
    print(f"FixRSVP activation movie: {fixrsvp_activation_movie.shape}")

    _write_csv(
        OUT_DIR / f"fixrsvp_activation_summary_{fixrsvp_slug}.csv",
        activation_summary_rows(fixrsvp_activation_movie),
    )

    for trace_mode in TRACE_PLOT_MODES:
        trace_slug = _safe_slug(trace_mode)
        fig_fixrsvp_dist = plot_activation_distributions(
            fixrsvp_activation_movie,
            trace_mode=trace_mode,
        )
        if SAVE_FIGURES:
            fig_fixrsvp_dist.savefig(
                OUT_DIR / f"fixrsvp_activation_distributions_{trace_slug}_{fixrsvp_slug}.png",
                dpi=160,
                bbox_inches="tight",
            )
        plt.show()

    fixrsvp_fingerprints = make_channel_fingerprints(
        fixrsvp_activation_movie,
        normalization=FINGERPRINT_NORMALIZATION,
    )
    fixrsvp_corr = channel_correlation_from_fingerprints(fixrsvp_fingerprints)
    fixrsvp_corr_filtered = fixrsvp_corr.copy()
    if "bad_unit_mask" in globals():
        _bad_mask = np.asarray(bad_unit_mask, dtype=bool)
        if _bad_mask.shape[0] == fixrsvp_corr_filtered.shape[0]:
            fixrsvp_corr_filtered[_bad_mask, :] = 0.0
            fixrsvp_corr_filtered[:, _bad_mask] = 0.0
            np.fill_diagonal(fixrsvp_corr_filtered, 1.0)

    fixrsvp_top_pairs = top_correlated_pairs(
        fixrsvp_corr_filtered,
        unit_rows=bundle.unit_rows,
        n_pairs=N_TOP_PAIRS,
    )
    _write_csv(OUT_DIR / f"fixrsvp_top_pairs_{fixrsvp_slug}.csv", fixrsvp_top_pairs)
    np.savez_compressed(
        OUT_DIR / f"fixrsvp_channel_corr_{fixrsvp_slug}_{FINGERPRINT_NORMALIZATION}.npz",
        corr=fixrsvp_corr.astype(np.float32),
        corr_filtered=fixrsvp_corr_filtered.astype(np.float32),
        top_pairs_json=np.asarray(json.dumps(fixrsvp_top_pairs, sort_keys=True, default=_json_default)),
        metadata_json=np.asarray(json.dumps(fixrsvp_metadata, sort_keys=True, default=_json_default)),
    )

    (
        fixrsvp_summary,
        fixrsvp_group_rows,
        fixrsvp_member_rows,
        fixrsvp_reconstruction_rows,
    ) = audit_redundancy_labels_on_heldout_movies(
        PATCHED_VERSION_NAME,
        patched_labels,
        [fixrsvp_activation_movie],
        [fixrsvp_label],
    )
    _write_csv(OUT_DIR / f"fixrsvp_cluster_audit_summary_{fixrsvp_slug}.csv", [fixrsvp_summary])
    _write_csv(OUT_DIR / f"fixrsvp_cluster_audit_groups_{fixrsvp_slug}.csv", fixrsvp_group_rows)
    _write_csv(OUT_DIR / f"fixrsvp_cluster_audit_members_{fixrsvp_slug}.csv", fixrsvp_member_rows)
    _write_csv(
        OUT_DIR / f"fixrsvp_cluster_audit_reconstruction_{fixrsvp_slug}.csv",
        fixrsvp_reconstruction_rows,
    )

    print("FixRSVP audit summary:")
    print(fixrsvp_summary)
    fixrsvp_audit_summary_table = _as_table([fixrsvp_summary])

    if SAVE_FIGURES:
        fig_fixrsvp_quality = plot_stimulus_cluster_audit_summary(
            fixrsvp_group_rows,
            fixrsvp_reconstruction_rows,
            version_name=PATCHED_VERSION_NAME,
            stimulus_label=fixrsvp_label,
        )
        fig_fixrsvp_quality.savefig(
            OUT_DIR / f"fixrsvp_cluster_audit_summary_{fixrsvp_slug}.png",
            dpi=170,
            bbox_inches="tight",
        )
        plt.show()

        fig_fixrsvp_corr = plot_reduced_corr_heatmap(
            fixrsvp_corr_filtered,
            patched_labels,
            title=f"FixRSVP representative correlation — {PATCHED_VERSION_NAME}",
        )
        fig_fixrsvp_corr.savefig(
            OUT_DIR / f"fixrsvp_reduced_corr_heatmap_{fixrsvp_slug}.png",
            dpi=170,
            bbox_inches="tight",
        )
        plt.show()

        fixrsvp_movie_rows = [
            row for row in fixrsvp_group_rows
            if str(row.get("metric_space")) == "movie_t_h_w"
        ]
        selected_fixrsvp_groups = select_validation_groups_to_plot(
            fixrsvp_movie_rows,
            PATCHED_VERSION_NAME,
            n_groups=N_VALIDATION_GROUPS_TO_PLOT,
        )
        if selected_fixrsvp_groups:
            for trace_mode in TRACE_PLOT_MODES:
                trace_slug = _safe_slug(trace_mode)
                fig_fixrsvp_traces = plot_group_trace_overlays(
                    fixrsvp_activation_movie,
                    selected_fixrsvp_groups,
                    unit_rows=bundle.unit_rows,
                    max_frames=VALIDATION_TRACE_FRAMES,
                    trace_mode=trace_mode,
                )
                fig_fixrsvp_traces.savefig(
                    OUT_DIR / f"fixrsvp_group_trace_overlays_{trace_slug}_{fixrsvp_slug}.png",
                    dpi=170,
                    bbox_inches="tight",
                )
                plt.show()

            fig_fixrsvp_maps = plot_group_activation_map_panels(
                fixrsvp_activation_movie,
                selected_fixrsvp_groups,
                unit_rows=bundle.unit_rows,
                max_members=N_VALIDATION_MAP_MEMBERS,
            )
            fig_fixrsvp_maps.savefig(
                OUT_DIR / f"fixrsvp_group_activation_maps_{fixrsvp_slug}.png",
                dpi=170,
                bbox_inches="tight",
            )
            plt.show()

    fixrsvp_movie_group_table = _as_table(
        [
            row for row in fixrsvp_group_rows
            if str(row.get("metric_space")) == "movie_t_h_w"
        ]
    )
elif not RUN_FIXRSVP_AUDIT:
    print("Skipping FixRSVP audit: RUN_FIXRSVP_AUDIT=False.")
else:
    print("Skipping FixRSVP audit: patched_labels not available.")

fixrsvp_audit_summary_table


# %% Step 19: reciprocal stimulus-generalization audit
# Directional question:
#   1. BackImage-built V1-RR -> FixRSVP test   (already exposed problems above)
#   2. FixRSVP-initialized groups -> BackImage test
#   3. FixRSVP-built groups -> BackImage test, after applying the same recursive
#      within-stimulus split logic on the FixRSVP movie.
#
# This tells us whether the apparent failure is asymmetric: perhaps FixRSVP is a
# better redundancy-discovery stimulus that still generalizes back to BackImage.
directional_generalization_table = None
fixrsvp_init_labels = None
fixrsvp_built_labels = None

if RUN_FIXRSVP_AUDIT and heldout_movies and heldout_case_labels:
    if "fixrsvp_activation_movie" not in globals() or "fixrsvp_corr_filtered" not in globals():
        fixrsvp_cache = fixrsvp_activation_cache_path(
            stimulus_type=FIXRSVP_STIMULUS_TYPE,
            frames_per_image=FIXRSVP_FRAMES_PER_IMAGE,
            num_frames=FIXRSVP_NUM_FRAMES,
            frame=FIXRSVP_FRAME,
        )
        if LOAD_CACHE_IF_AVAILABLE and fixrsvp_cache.exists():
            fixrsvp_payload = load_activation_cache(fixrsvp_cache)
            fixrsvp_activation_movie = fixrsvp_payload["activation_movie"]
            fixrsvp_metadata = fixrsvp_payload["metadata"]
            print(f"Loaded FixRSVP activation movie cache: {fixrsvp_cache}")
        else:
            fixrsvp_activation_movie, fixrsvp_metadata = compute_fixrsvp_activation_movie(
                bundle,
                stimulus_type=FIXRSVP_STIMULUS_TYPE,
                frames_per_image=FIXRSVP_FRAMES_PER_IMAGE,
                num_frames=FIXRSVP_NUM_FRAMES,
                frame=FIXRSVP_FRAME,
                batch_size=BATCH_SIZE,
            )
            if SAVE_ACTIVATION_CACHE:
                save_fixrsvp_activation_cache(fixrsvp_cache, fixrsvp_activation_movie, fixrsvp_metadata)
                print(f"Saved FixRSVP activation movie cache: {fixrsvp_cache}")

        fixrsvp_fingerprints = make_channel_fingerprints(
            fixrsvp_activation_movie,
            normalization=FINGERPRINT_NORMALIZATION,
        )
        fixrsvp_corr = channel_correlation_from_fingerprints(fixrsvp_fingerprints)
        fixrsvp_corr_filtered = fixrsvp_corr.copy()
        if "bad_unit_mask" in globals():
            _bad_mask = np.asarray(bad_unit_mask, dtype=bool)
            if _bad_mask.shape[0] == fixrsvp_corr_filtered.shape[0]:
                fixrsvp_corr_filtered[_bad_mask, :] = 0.0
                fixrsvp_corr_filtered[:, _bad_mask] = 0.0
                np.fill_diagonal(fixrsvp_corr_filtered, 1.0)

    fixrsvp_label = (
        f"FixRSVP_{FIXRSVP_STIMULUS_TYPE}_fpi{FIXRSVP_FRAMES_PER_IMAGE}"
        f"_frames{FIXRSVP_NUM_FRAMES}"
        f"_{'dynamic' if FIXRSVP_FRAME is None else f'frame{FIXRSVP_FRAME:03d}'}"
    )
    directional_slug = _safe_slug(
        f"directional_{PATCHED_VERSION_NAME}_{fixrsvp_label}_backimageheldout{len(heldout_movies)}"
    )

    def _rep_counts(labels: np.ndarray) -> dict[str, int]:
        labels_arr = np.asarray(labels, dtype=int)
        n_groups = len(set(labels_arr[labels_arr >= 0]))
        n_singletons = int(np.sum(labels_arr == -1))
        return {
            "n_representatives": int(n_groups + n_singletons),
            "n_groups": int(n_groups),
            "n_singletons": n_singletons,
            "n_excluded": int(np.sum(labels_arr == -2)),
        }

    def _make_init_labels(similarity: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        labels = redundancy_groups_from_corr(
            similarity,
            threshold=PATCH_BASE_THRESHOLD,
            method=REDUNDANCY_LINKAGE_METHOD,
        )
        if mask is not None and np.asarray(mask).shape[0] == labels.shape[0]:
            labels[np.asarray(mask, dtype=bool)] = -2
        return labels

    def _directional_row(
        *,
        candidate: str,
        construction_stimulus: str,
        test_stimulus: str,
        labels: np.ndarray,
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        row = {
            "candidate": candidate,
            "construction_stimulus": construction_stimulus,
            "test_stimulus": test_stimulus,
        }
        row.update(_rep_counts(labels))
        n_group_case_tests = int(row["n_groups"]) * int(summary.get("n_cases", 0))
        row["n_group_case_tests"] = n_group_case_tests
        for key in (
            "n_cases",
            "movie_worst_group_min_centroid_corr",
            "movie_median_group_min_centroid_corr",
            "movie_groups_below_0p75",
            "movie_groups_below_0p60",
            "movie_worst_group_min_pairwise_corr",
            "movie_groups_below_0p75_pairwise",
            "movie_groups_below_0p60_pairwise",
            "mean_map_worst_group_min_centroid_corr",
            "mean_map_groups_below_0p75",
            "mean_global_reconstruction_corr",
            "min_global_reconstruction_corr",
        ):
            row[key] = summary.get(key, np.nan)
        denom = max(int(row["n_group_case_tests"]), 1)
        row["movie_groups_below_0p75_fraction"] = float(row["movie_groups_below_0p75"]) / denom
        row["movie_groups_below_0p60_fraction"] = float(row["movie_groups_below_0p60"]) / denom
        row["movie_groups_below_0p75_pairwise_fraction"] = (
            float(row["movie_groups_below_0p75_pairwise"]) / denom
        )
        row["movie_groups_below_0p60_pairwise_fraction"] = (
            float(row["movie_groups_below_0p60_pairwise"]) / denom
        )
        return row

    directional_summary_rows: list[dict[str, Any]] = []
    directional_group_rows: list[dict[str, Any]] = []
    directional_reconstruction_rows: list[dict[str, Any]] = []

    # A. BackImage-built V1-RR tested on FixRSVP. This is the reciprocal of the
    # new FixRSVP-first rows and is repeated here so the comparison table is self-contained.
    if patched_labels is not None:
        backimage_built_name = f"{PATCHED_VERSION_NAME}__built_BackImage__test_FixRSVP"
        (
            bi_to_fx_summary,
            bi_to_fx_groups,
            _bi_to_fx_members,
            bi_to_fx_recon,
        ) = audit_redundancy_labels_on_heldout_movies(
            backimage_built_name,
            patched_labels,
            [fixrsvp_activation_movie],
            [fixrsvp_label],
        )
        directional_summary_rows.append(
            _directional_row(
                candidate=backimage_built_name,
                construction_stimulus="BackImage",
                test_stimulus="FixRSVP",
                labels=patched_labels,
                summary=bi_to_fx_summary,
            )
        )
        directional_group_rows.extend(bi_to_fx_groups)
        directional_reconstruction_rows.extend(bi_to_fx_recon)

    # B. Initial clusters from FixRSVP correlation only, tested on held-out BackImage.
    fixrsvp_init_name = (
        f"V1-RRfix_init_complete_{PATCH_BASE_THRESHOLD:.2f}"
        f"__built_FixRSVP__test_BackImage"
    ).replace("0.", "0p")
    fixrsvp_init_labels = _make_init_labels(
        fixrsvp_corr_filtered,
        bad_unit_mask if "bad_unit_mask" in globals() else None,
    )
    (
        fx_init_to_bi_summary,
        fx_init_to_bi_groups,
        _fx_init_to_bi_members,
        fx_init_to_bi_recon,
    ) = audit_redundancy_labels_on_heldout_movies(
        fixrsvp_init_name,
        fixrsvp_init_labels,
        heldout_movies,
        heldout_case_labels,
    )
    directional_summary_rows.append(
        _directional_row(
            candidate=fixrsvp_init_name,
            construction_stimulus="FixRSVP_init_only",
            test_stimulus="BackImage",
            labels=fixrsvp_init_labels,
            summary=fx_init_to_bi_summary,
        )
    )
    directional_group_rows.extend(fx_init_to_bi_groups)
    directional_reconstruction_rows.extend(fx_init_to_bi_recon)

    # C. FixRSVP-initialized and recursively split on FixRSVP, then tested on BackImage.
    fixrsvp_built_name = (
        f"V1-RRfix_complete_{PATCH_BASE_THRESHOLD:.2f}"
        f"_fixsplit{MOVIE_SPLIT_CENTROID_THRESHOLD:.2f}"
        f"_pair{MOVIE_SPLIT_PAIRWISE_THRESHOLD:.2f}"
        f"_rec{MOVIE_SPLIT_MAX_PASSES}"
        f"__built_FixRSVP__test_BackImage"
    ).replace("0.", "0p")
    fixrsvp_built_labels, fixrsvp_recursive_split_rows = recursive_movie_split_labels(
        fixrsvp_init_labels,
        fixrsvp_corr_filtered,
        [fixrsvp_activation_movie],
        [fixrsvp_label],
        version_name=fixrsvp_built_name,
        centroid_threshold=MOVIE_SPLIT_CENTROID_THRESHOLD,
        centroid_case_fraction=MOVIE_SPLIT_CASE_FRACTION,
        pairwise_threshold=MOVIE_SPLIT_PAIRWISE_THRESHOLD,
        pairwise_case_fraction=MOVIE_SPLIT_PAIRWISE_CASE_FRACTION,
        split_threshold=MOVIE_SPLIT_SUBCLUSTER_THRESHOLD,
        max_passes=MOVIE_SPLIT_MAX_PASSES,
    )
    if "bad_unit_mask" in globals():
        _bad_mask = np.asarray(bad_unit_mask, dtype=bool)
        if _bad_mask.shape[0] == fixrsvp_built_labels.shape[0]:
            fixrsvp_built_labels[_bad_mask] = -2

    (
        fx_built_to_bi_summary,
        fx_built_to_bi_groups,
        _fx_built_to_bi_members,
        fx_built_to_bi_recon,
    ) = audit_redundancy_labels_on_heldout_movies(
        fixrsvp_built_name,
        fixrsvp_built_labels,
        heldout_movies,
        heldout_case_labels,
    )
    directional_summary_rows.append(
        _directional_row(
            candidate=fixrsvp_built_name,
            construction_stimulus="FixRSVP",
            test_stimulus="BackImage",
            labels=fixrsvp_built_labels,
            summary=fx_built_to_bi_summary,
        )
    )
    directional_group_rows.extend(fx_built_to_bi_groups)
    directional_reconstruction_rows.extend(fx_built_to_bi_recon)

    np.save(OUT_DIR / f"fixrsvp_init_labels_{directional_slug}.npy", fixrsvp_init_labels)
    np.save(OUT_DIR / f"fixrsvp_built_labels_{directional_slug}.npy", fixrsvp_built_labels)
    _write_csv(
        OUT_DIR / f"fixrsvp_recursive_split_flags_{directional_slug}.csv",
        fixrsvp_recursive_split_rows,
    )
    _write_csv(
        OUT_DIR / f"directional_generalization_summary_{directional_slug}.csv",
        directional_summary_rows,
    )
    _write_csv(
        OUT_DIR / f"directional_generalization_groups_{directional_slug}.csv",
        directional_group_rows,
    )
    _write_csv(
        OUT_DIR / f"directional_generalization_reconstruction_{directional_slug}.csv",
        directional_reconstruction_rows,
    )

    print("Directional generalization summary:")
    for row in directional_summary_rows:
        print(row)
    directional_generalization_table = _as_table(directional_summary_rows)

    if SAVE_FIGURES:
        candidates = [str(row["construction_stimulus"]) for row in directional_summary_rows]
        x = np.arange(len(directional_summary_rows))
        fig_dir, axes = plt.subplots(2, 3, figsize=(15.5, 7.8), constrained_layout=True)
        axes_flat = axes.ravel()

        axes_flat[0].bar(x, [int(row["n_representatives"]) for row in directional_summary_rows], color="0.35")
        axes_flat[0].set_ylabel("representatives")
        axes_flat[0].set_title("Compression")

        axes_flat[1].bar(
            x,
            [float(row["movie_worst_group_min_centroid_corr"]) for row in directional_summary_rows],
            color="tab:red",
            alpha=0.75,
        )
        axes_flat[1].axhline(0.75, color="0.25", linestyle="--", linewidth=1.0)
        axes_flat[1].axhline(0.60, color="0.25", linestyle=":", linewidth=1.0)
        axes_flat[1].set_ylabel("worst group min corr")
        axes_flat[1].set_title("Worst group on test stimulus")

        axes_flat[2].bar(
            x,
            [float(row["movie_median_group_min_centroid_corr"]) for row in directional_summary_rows],
            color="tab:blue",
            alpha=0.75,
        )
        axes_flat[2].axhline(0.75, color="0.25", linestyle="--", linewidth=1.0)
        axes_flat[2].set_ylabel("median group min corr")
        axes_flat[2].set_title("Typical group on test stimulus")

        axes_flat[3].bar(
            x,
            [float(row["movie_groups_below_0p75_fraction"]) for row in directional_summary_rows],
            color="tab:orange",
            alpha=0.8,
        )
        axes_flat[3].set_ylim(0.0, 1.0)
        axes_flat[3].set_ylabel("fraction below .75")
        axes_flat[3].set_title("Centroid failures\n(group-case tests)")

        axes_flat[4].bar(
            x,
            [float(row["movie_groups_below_0p75_pairwise_fraction"]) for row in directional_summary_rows],
            color="tab:purple",
            alpha=0.75,
        )
        axes_flat[4].set_ylim(0.0, 1.0)
        axes_flat[4].set_ylabel("fraction below .75")
        axes_flat[4].set_title("Pairwise failures\n(group-case tests)")

        axes_flat[5].bar(
            x,
            [float(row["mean_global_reconstruction_corr"]) for row in directional_summary_rows],
            color="tab:green",
            alpha=0.75,
        )
        axes_flat[5].set_ylim(0.0, 1.02)
        axes_flat[5].set_ylabel("global reconstruction corr")
        axes_flat[5].set_title("Pool-expand reconstruction")

        for ax in axes_flat:
            ax.set_xticks(x)
            ax.set_xticklabels(candidates, rotation=18, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.18)
        fig_dir.suptitle("Directional generalization: construction stimulus vs test stimulus", fontsize=12)
        fig_dir.savefig(
            OUT_DIR / f"directional_generalization_summary_{directional_slug}.png",
            dpi=170,
            bbox_inches="tight",
        )
        plt.show()
elif not RUN_FIXRSVP_AUDIT:
    print("Skipping directional generalization: RUN_FIXRSVP_AUDIT=False.")
else:
    print("Skipping directional generalization: need held-out BackImage movies and FixRSVP audit variables.")

directional_generalization_table


def inspect_rr_candidate_on_movies(
    *,
    version_name: str,
    labels: np.ndarray,
    audit_movies: list[np.ndarray],
    audit_case_labels: list[str],
    audit_name: str,
    corr_for_heatmap: np.ndarray | None = None,
    unit_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the standard audit/plot suite for one candidate on one stimulus set."""
    if not audit_movies:
        raise ValueError(f"No audit movies supplied for {version_name} / {audit_name}")

    summary, group_rows, member_rows, reconstruction_rows = audit_redundancy_labels_on_heldout_movies(
        version_name,
        labels,
        audit_movies,
        audit_case_labels,
    )
    slug = _safe_slug(f"{version_name}_{audit_name}")
    _write_csv(OUT_DIR / f"candidate_audit_summary_{slug}.csv", [summary])
    _write_csv(OUT_DIR / f"candidate_audit_groups_{slug}.csv", group_rows)
    _write_csv(OUT_DIR / f"candidate_audit_members_{slug}.csv", member_rows)
    _write_csv(OUT_DIR / f"candidate_audit_reconstruction_{slug}.csv", reconstruction_rows)

    print(f"{audit_name} audit summary for {version_name}:")
    print(summary)

    if SAVE_FIGURES:
        fig_quality = plot_stimulus_cluster_audit_summary(
            group_rows,
            reconstruction_rows,
            version_name=version_name,
            stimulus_label=audit_name,
        )
        fig_quality.savefig(
            OUT_DIR / f"candidate_audit_summary_{slug}.png",
            dpi=170,
            bbox_inches="tight",
        )
        plt.show()

        if corr_for_heatmap is not None:
            fig_heatmap = plot_reduced_corr_heatmap(
                corr_for_heatmap,
                labels,
                title=f"{audit_name} representative correlation - {version_name}",
            )
            fig_heatmap.savefig(
                OUT_DIR / f"candidate_reduced_corr_heatmap_{slug}.png",
                dpi=170,
                bbox_inches="tight",
            )
            plt.show()

        plot_case_label = str(audit_case_labels[0])
        plot_movie = audit_movies[0]
        plot_rows = [
            row for row in group_rows
            if str(row.get("metric_space")) == "movie_t_h_w"
            and str(row.get("case")) == plot_case_label
        ]
        if not plot_rows:
            plot_rows = [
                row for row in group_rows
                if str(row.get("metric_space")) == "movie_t_h_w"
            ]
        selected_groups = select_validation_groups_to_plot(
            plot_rows,
            version_name,
            n_groups=N_VALIDATION_GROUPS_TO_PLOT,
        )
        if selected_groups:
            for trace_mode in TRACE_PLOT_MODES:
                trace_slug = _safe_slug(trace_mode)
                fig_traces = plot_group_trace_overlays(
                    plot_movie,
                    selected_groups,
                    unit_rows=unit_rows,
                    max_frames=VALIDATION_TRACE_FRAMES,
                    trace_mode=trace_mode,
                )
                fig_traces.savefig(
                    OUT_DIR / f"candidate_group_trace_overlays_{trace_slug}_{slug}.png",
                    dpi=170,
                    bbox_inches="tight",
                )
                plt.show()

            fig_maps = plot_group_activation_map_panels(
                plot_movie,
                selected_groups,
                unit_rows=unit_rows,
                max_members=N_VALIDATION_MAP_MEMBERS,
            )
            fig_maps.savefig(
                OUT_DIR / f"candidate_group_activation_maps_{slug}.png",
                dpi=170,
                bbox_inches="tight",
            )
            plt.show()

            singleton_rows = select_sharp_channel_rows(
                plot_movie,
                channels=np.flatnonzero(np.asarray(labels, dtype=int) == -1),
                labels=labels,
                n_channels=N_SINGLETON_MAPS_TO_PLOT,
            )
            _write_csv(
                OUT_DIR / f"candidate_singleton_map_examples_{slug}.csv",
                singleton_rows,
            )
            if singleton_rows:
                fig_singletons = plot_channel_activation_map_gallery(
                    plot_movie,
                    singleton_rows,
                    title=f"{audit_name} sharp singleton activation maps - {version_name}",
                    labels=labels,
                    unit_rows=unit_rows,
                )
                fig_singletons.savefig(
                    OUT_DIR / f"candidate_singleton_activation_maps_{slug}.png",
                    dpi=180,
                    bbox_inches="tight",
                )
                plt.show()

            sharp_rows = select_sharp_channel_rows(
                plot_movie,
                channels=np.flatnonzero(np.asarray(labels, dtype=int) != -2),
                labels=labels,
                n_channels=N_SHARP_MAPS_TO_PLOT,
            )
            _write_csv(
                OUT_DIR / f"candidate_sharp_map_examples_{slug}.csv",
                sharp_rows,
            )
            if sharp_rows:
                fig_sharp = plot_channel_activation_map_gallery(
                    plot_movie,
                    sharp_rows,
                    title=f"{audit_name} sharp activation-map examples - {version_name}",
                    labels=labels,
                    unit_rows=unit_rows,
                )
                fig_sharp.savefig(
                    OUT_DIR / f"candidate_sharp_activation_maps_{slug}.png",
                    dpi=180,
                    bbox_inches="tight",
                )
                plt.show()

    return summary, group_rows, member_rows, reconstruction_rows


def _candidate_count_row(candidate: str, labels: np.ndarray) -> dict[str, Any]:
    labels_arr = np.asarray(labels, dtype=int)
    n_groups = len(set(labels_arr[labels_arr >= 0]))
    n_singletons = int(np.sum(labels_arr == -1))
    n_excluded = int(np.sum(labels_arr == -2))
    return {
        "candidate": candidate,
        "n_representatives": int(n_groups + n_singletons),
        "n_groups": int(n_groups),
        "n_singletons": n_singletons,
        "n_excluded": n_excluded,
    }


# %% Step 20: FixRSVP-first candidate inspection suite
# The previous cell answered the directional question numerically. This cell is
# intentionally visual: it shows the FixRSVP-initialized/built population on its
# construction stimulus first, then on held-out BackImage, before any
# multi-stimulus RR is constructed.
fixrsvp_candidate_inspection_table = None

if fixrsvp_built_labels is not None and "fixrsvp_activation_movie" in globals():
    fixrsvp_candidate_name = fixrsvp_built_name if "fixrsvp_built_name" in globals() else "V1-RRfix"
    fixrsvp_candidate_rows = [_candidate_count_row(fixrsvp_candidate_name, fixrsvp_built_labels)]

    fx_construct_summary, _, _, _ = inspect_rr_candidate_on_movies(
        version_name=fixrsvp_candidate_name,
        labels=fixrsvp_built_labels,
        audit_movies=[fixrsvp_activation_movie],
        audit_case_labels=[fixrsvp_label],
        audit_name="FixRSVP_construction",
        corr_for_heatmap=fixrsvp_corr_filtered,
        unit_rows=bundle.unit_rows,
    )
    fx_generalization_summary, _, _, _ = inspect_rr_candidate_on_movies(
        version_name=fixrsvp_candidate_name,
        labels=fixrsvp_built_labels,
        audit_movies=heldout_movies,
        audit_case_labels=heldout_case_labels,
        audit_name="BackImage_generalization",
        corr_for_heatmap=None,
        unit_rows=bundle.unit_rows,
    )
    fixrsvp_candidate_rows[0].update({
        "construction_movie_worst": fx_construct_summary["movie_worst_group_min_centroid_corr"],
        "construction_movie_median": fx_construct_summary["movie_median_group_min_centroid_corr"],
        "construction_reconstruction_corr": fx_construct_summary["mean_global_reconstruction_corr"],
        "backimage_movie_worst": fx_generalization_summary["movie_worst_group_min_centroid_corr"],
        "backimage_movie_median": fx_generalization_summary["movie_median_group_min_centroid_corr"],
        "backimage_reconstruction_corr": fx_generalization_summary["mean_global_reconstruction_corr"],
    })
    _write_csv(
        OUT_DIR / f"fixrsvp_candidate_inspection_{_safe_slug(fixrsvp_candidate_name)}.csv",
        fixrsvp_candidate_rows,
    )
    fixrsvp_candidate_inspection_table = _as_table(fixrsvp_candidate_rows)
else:
    print("Skipping FixRSVP-first candidate inspection: run Step 19 first.")

fixrsvp_candidate_inspection_table


# %% Step 21: multi-stimulus RR candidate construction and inspection
# This comes after the single-stimulus candidates so the decision points remain
# visible. Default mode is conservative intersection-style similarity:
#   multi_corr = min(BackImage corr, FixRSVP corr)
# Set MULTISTIM_SIMILARITY_MODE="mean" above to inspect a less conservative
# balanced-average candidate.
multistim_labels = None
multistim_inspection_table = None

if RUN_FIXRSVP_AUDIT and heldout_movies and "fixrsvp_corr_filtered" in globals():
    if MULTISTIM_SIMILARITY_MODE == "min":
        multistim_corr = np.minimum(corr_filtered, fixrsvp_corr_filtered)
    elif MULTISTIM_SIMILARITY_MODE == "mean":
        multistim_corr = 0.5 * np.asarray(corr_filtered, dtype=np.float32) + 0.5 * np.asarray(fixrsvp_corr_filtered, dtype=np.float32)
    else:
        raise ValueError(f"Unknown MULTISTIM_SIMILARITY_MODE={MULTISTIM_SIMILARITY_MODE!r}")
    multistim_corr = np.asarray(multistim_corr, dtype=np.float32)
    if "bad_unit_mask" in globals():
        _bad_mask = np.asarray(bad_unit_mask, dtype=bool)
        if _bad_mask.shape[0] == multistim_corr.shape[0]:
            multistim_corr[_bad_mask, :] = 0.0
            multistim_corr[:, _bad_mask] = 0.0
    np.fill_diagonal(multistim_corr, 1.0)

    multistim_version_name = (
        f"V1-RR_MS_{MULTISTIM_SIMILARITY_MODE}"
        f"_complete{PATCH_BASE_THRESHOLD:.2f}"
        f"_split{MOVIE_SPLIT_CENTROID_THRESHOLD:.2f}"
        f"_pair{MOVIE_SPLIT_PAIRWISE_THRESHOLD:.2f}"
        f"_anyfail"
    ).replace("0.", "0p")

    multistim_base_labels = redundancy_groups_from_corr(
        multistim_corr,
        threshold=PATCH_BASE_THRESHOLD,
        method=REDUNDANCY_LINKAGE_METHOD,
    )
    if "bad_unit_mask" in globals():
        _bad_mask = np.asarray(bad_unit_mask, dtype=bool)
        if _bad_mask.shape[0] == multistim_base_labels.shape[0]:
            multistim_base_labels[_bad_mask] = -2
    print("Multi-stimulus base:", _candidate_count_row(multistim_version_name + "_base", multistim_base_labels))

    multistim_construction_movies = [fixrsvp_activation_movie] + heldout_movies
    multistim_construction_case_labels = [fixrsvp_label] + heldout_case_labels
    multistim_labels, multistim_split_rows = recursive_movie_split_labels(
        multistim_base_labels,
        multistim_corr,
        multistim_construction_movies,
        multistim_construction_case_labels,
        version_name=multistim_version_name,
        centroid_threshold=MOVIE_SPLIT_CENTROID_THRESHOLD,
        centroid_case_fraction=MULTISTIM_SPLIT_CASE_FRACTION,
        pairwise_threshold=MOVIE_SPLIT_PAIRWISE_THRESHOLD,
        pairwise_case_fraction=MULTISTIM_PAIRWISE_CASE_FRACTION,
        split_threshold=MOVIE_SPLIT_SUBCLUSTER_THRESHOLD,
        max_passes=MOVIE_SPLIT_MAX_PASSES,
    )
    if "bad_unit_mask" in globals():
        _bad_mask = np.asarray(bad_unit_mask, dtype=bool)
        if _bad_mask.shape[0] == multistim_labels.shape[0]:
            multistim_labels[_bad_mask] = -2

    multistim_slug = _safe_slug(multistim_version_name)
    np.save(OUT_DIR / f"multistim_labels_{multistim_slug}.npy", multistim_labels)
    _write_csv(OUT_DIR / f"multistim_recursive_split_flags_{multistim_slug}.csv", multistim_split_rows)

    ms_construct_summary, _, _, _ = inspect_rr_candidate_on_movies(
        version_name=multistim_version_name,
        labels=multistim_labels,
        audit_movies=multistim_construction_movies,
        audit_case_labels=multistim_construction_case_labels,
        audit_name="MultiStim_construction",
        corr_for_heatmap=multistim_corr,
        unit_rows=bundle.unit_rows,
    )
    ms_fix_summary, _, _, _ = inspect_rr_candidate_on_movies(
        version_name=multistim_version_name,
        labels=multistim_labels,
        audit_movies=[fixrsvp_activation_movie],
        audit_case_labels=[fixrsvp_label],
        audit_name="FixRSVP_component",
        corr_for_heatmap=None,
        unit_rows=bundle.unit_rows,
    )
    ms_backimage_summary, _, _, _ = inspect_rr_candidate_on_movies(
        version_name=multistim_version_name,
        labels=multistim_labels,
        audit_movies=heldout_movies,
        audit_case_labels=heldout_case_labels,
        audit_name="BackImage_component",
        corr_for_heatmap=None,
        unit_rows=bundle.unit_rows,
    )

    multistim_rows = [_candidate_count_row(multistim_version_name, multistim_labels)]
    multistim_rows[0].update({
        "multi_movie_worst": ms_construct_summary["movie_worst_group_min_centroid_corr"],
        "multi_movie_median": ms_construct_summary["movie_median_group_min_centroid_corr"],
        "multi_reconstruction_corr": ms_construct_summary["mean_global_reconstruction_corr"],
        "fixrsvp_movie_worst": ms_fix_summary["movie_worst_group_min_centroid_corr"],
        "fixrsvp_movie_median": ms_fix_summary["movie_median_group_min_centroid_corr"],
        "backimage_movie_worst": ms_backimage_summary["movie_worst_group_min_centroid_corr"],
        "backimage_movie_median": ms_backimage_summary["movie_median_group_min_centroid_corr"],
    })
    _write_csv(OUT_DIR / f"multistim_candidate_inspection_{multistim_slug}.csv", multistim_rows)

    if SAVE_MULTISTIM_POPULATION_SPEC:
        save_population_spec(multistim_labels, bundle.unit_rows, multistim_version_name, OUT_DIR)

    multistim_inspection_table = _as_table(multistim_rows)
else:
    print("Skipping multi-stimulus RR: need FixRSVP corr and held-out BackImage movies.")

multistim_inspection_table


# %% Step 22: final cleanup - force-split unresolved multi-stimulus failures
# The recursive split can leave a tiny tail when the split-correlation matrix
# still groups members that fail in a specific held-out movie. This final cleanup
# is deliberately blunt and transparent: any group with movie member-centroid
# quality below the final gate on any construction-battery movie is converted to
# true singletons. Pairwise failures remain diagnostic unless they also depress
# member-centroid quality.
multistim_final_labels = None
multistim_final_inspection_table = None

if multistim_labels is not None and "multistim_construction_movies" in globals():
    final_gate = float(MULTISTIM_FINAL_FORCE_SPLIT_CENTROID_THRESHOLD)
    cleanup_summary, cleanup_group_rows, _, _ = audit_redundancy_labels_on_heldout_movies(
        f"{multistim_version_name}_pre_final_cleanup",
        multistim_labels,
        multistim_construction_movies,
        multistim_construction_case_labels,
    )
    cleanup_movie_rows = [
        row for row in cleanup_group_rows
        if str(row.get("metric_space")) == "movie_t_h_w"
    ]
    cleanup_fail_rows = [
        row for row in cleanup_movie_rows
        if float(row.get("min_member_centroid_corr", np.nan)) < final_gate
    ]
    cleanup_bad_group_ids = sorted({int(row["group_id"]) for row in cleanup_fail_rows})

    multistim_final_labels = np.asarray(multistim_labels, dtype=int).copy()
    for gid in cleanup_bad_group_ids:
        multistim_final_labels[multistim_final_labels == gid] = -1
    if "bad_unit_mask" in globals():
        _bad_mask = np.asarray(bad_unit_mask, dtype=bool)
        if _bad_mask.shape[0] == multistim_final_labels.shape[0]:
            multistim_final_labels[_bad_mask] = -2

    multistim_final_version_name = (
        f"{multistim_version_name}"
        f"_finalsplit{final_gate:.2f}"
    ).replace("0.", "0p")
    multistim_final_slug = _safe_slug(multistim_final_version_name)

    cleanup_rows = []
    for row in cleanup_fail_rows:
        out = dict(row)
        out["final_force_split_threshold"] = final_gate
        out["n_members_forced_to_singletons"] = int(np.sum(np.asarray(multistim_labels) == int(row["group_id"])))
        cleanup_rows.append(out)
    _write_csv(
        OUT_DIR / f"multistim_final_force_split_flags_{multistim_final_slug}.csv",
        cleanup_rows,
    )
    np.save(OUT_DIR / f"multistim_final_labels_{multistim_final_slug}.npy", multistim_final_labels)

    print(
        f"Final cleanup: {len(cleanup_bad_group_ids)} groups force-split "
        f"({sum(int(np.sum(np.asarray(multistim_labels) == gid)) for gid in cleanup_bad_group_ids)} channels) "
        f"at movie centroid < {final_gate:.2f}"
    )
    if cleanup_bad_group_ids:
        for row in sorted(cleanup_fail_rows, key=lambda r: float(r["min_member_centroid_corr"]))[:20]:
            print(
                f"  case={row['case']} group={int(row['group_id'])} size={int(row['size'])} "
                f"min_centroid={float(row['min_member_centroid_corr']):.3f} "
                f"min_pairwise={float(row.get('min_pairwise_member_corr', np.nan)):.3f} "
                f"members={row['members']}"
            )

    (
        ms_final_construct_summary,
        ms_final_construct_group_rows,
        ms_final_construct_member_rows,
        ms_final_construct_reconstruction_rows,
    ) = inspect_rr_candidate_on_movies(
        version_name=multistim_final_version_name,
        labels=multistim_final_labels,
        audit_movies=multistim_construction_movies,
        audit_case_labels=multistim_construction_case_labels,
        audit_name="MultiStim_construction_final",
        corr_for_heatmap=multistim_corr if "multistim_corr" in globals() else None,
        unit_rows=bundle.unit_rows,
    )
    ms_final_fix_summary, _, _, _ = inspect_rr_candidate_on_movies(
        version_name=multistim_final_version_name,
        labels=multistim_final_labels,
        audit_movies=[fixrsvp_activation_movie],
        audit_case_labels=[fixrsvp_label],
        audit_name="FixRSVP_component_final",
        corr_for_heatmap=None,
        unit_rows=bundle.unit_rows,
    )
    ms_final_backimage_summary, _, _, _ = inspect_rr_candidate_on_movies(
        version_name=multistim_final_version_name,
        labels=multistim_final_labels,
        audit_movies=heldout_movies,
        audit_case_labels=heldout_case_labels,
        audit_name="BackImage_component_final",
        corr_for_heatmap=None,
        unit_rows=bundle.unit_rows,
    )

    multistim_final_rows = [_candidate_count_row(multistim_final_version_name, multistim_final_labels)]
    multistim_final_rows[0].update({
        "n_force_split_groups": len(cleanup_bad_group_ids),
        "n_force_split_channels": int(sum(np.sum(np.asarray(multistim_labels) == gid) for gid in cleanup_bad_group_ids)),
        "final_gate": final_gate,
        "multi_movie_worst": ms_final_construct_summary["movie_worst_group_min_centroid_corr"],
        "multi_movie_median": ms_final_construct_summary["movie_median_group_min_centroid_corr"],
        "multi_reconstruction_corr": ms_final_construct_summary["mean_global_reconstruction_corr"],
        "fixrsvp_movie_worst": ms_final_fix_summary["movie_worst_group_min_centroid_corr"],
        "fixrsvp_movie_median": ms_final_fix_summary["movie_median_group_min_centroid_corr"],
        "backimage_movie_worst": ms_final_backimage_summary["movie_worst_group_min_centroid_corr"],
        "backimage_movie_median": ms_final_backimage_summary["movie_median_group_min_centroid_corr"],
    })
    _write_csv(
        OUT_DIR / f"multistim_final_candidate_inspection_{multistim_final_slug}.csv",
        multistim_final_rows,
    )

    save_population_spec(
        multistim_final_labels,
        bundle.unit_rows,
        multistim_final_version_name,
        OUT_DIR,
    )
    save_medoid_population_variant(
        labels=multistim_final_labels,
        unit_rows=bundle.unit_rows,
        version_name=multistim_final_version_name,
        movies=multistim_construction_movies,
        case_labels=multistim_construction_case_labels,
        out_dir=OUT_DIR,
        member_rows=ms_final_construct_member_rows,
    )
    multistim_final_inspection_table = _as_table(multistim_final_rows)
else:
    print("Skipping final cleanup: run Step 21 first.")

multistim_final_inspection_table


# %% Step 23: labeled t-SNE/PCA positions for the sharpest final singletons
# This is a visual locator for the singleton units that still look most like
# compact cell-like response maps. The embedding is the same fingerprint-space
# view computed in Step 10; labels come from the final RR candidate when present.
sharp_singleton_embedding_table = None

if "embedding" not in globals() or embedding is None:
    if "fingerprints" in globals():
        embedding, embedding_name, pca_scores = compute_channel_embedding(fingerprints, run_tsne=RUN_TSNE)
    else:
        embedding = None
        embedding_name = "embedding unavailable"

if multistim_final_labels is not None:
    singleton_embedding_labels = np.asarray(multistim_final_labels, dtype=int)
    singleton_embedding_version = multistim_final_version_name
elif multistim_labels is not None:
    singleton_embedding_labels = np.asarray(multistim_labels, dtype=int)
    singleton_embedding_version = multistim_version_name
elif patched_labels is not None:
    singleton_embedding_labels = np.asarray(patched_labels, dtype=int)
    singleton_embedding_version = PATCHED_VERSION_NAME
else:
    singleton_embedding_labels = None
    singleton_embedding_version = "labels unavailable"

if embedding is not None and singleton_embedding_labels is not None:
    singleton_reference_movies: list[tuple[str, np.ndarray]] = []
    if heldout_movies:
        singleton_reference_movies.append((f"BackImage_{heldout_case_labels[0]}", heldout_movies[0]))
    elif "activation_movie" in globals():
        singleton_reference_movies.append((f"BackImage_{case.image_key}", activation_movie))
    if "fixrsvp_activation_movie" in globals():
        singleton_reference_movies.append((f"FixRSVP_{fixrsvp_label}", fixrsvp_activation_movie))

    singleton_rows_all: list[dict[str, Any]] = []
    singleton_channels = np.flatnonzero(singleton_embedding_labels == -1)
    for reference_label, reference_movie in singleton_reference_movies:
        sharp_singleton_rows = select_sharp_channel_rows(
            reference_movie,
            channels=singleton_channels,
            labels=singleton_embedding_labels,
            n_channels=N_LABELED_SHARP_SINGLETONS,
        )
        for rank, row in enumerate(sharp_singleton_rows, start=1):
            row["rank"] = int(rank)
            row["reference"] = reference_label
            row["version"] = singleton_embedding_version
        singleton_rows_all.extend(sharp_singleton_rows)

        ref_slug = _safe_slug(reference_label)
        version_slug = _safe_slug(singleton_embedding_version)
        _write_csv(
            OUT_DIR / f"sharp_singleton_embedding_labels_{version_slug}_{ref_slug}.csv",
            sharp_singleton_rows,
        )
        if sharp_singleton_rows:
            fig_sharp_singleton_embed = plot_embedding_labeled_singletons(
                embedding,
                embedding_name,
                singleton_embedding_labels,
                sharp_singleton_rows,
                title=(
                    f"Sharpest singleton positions - {singleton_embedding_version}\n"
                    f"reference: {reference_label}"
                ),
                unit_rows=bundle.unit_rows if "bundle" in globals() else None,
            )
            if SAVE_FIGURES:
                fig_sharp_singleton_embed.savefig(
                    OUT_DIR / f"sharp_singleton_embedding_labels_{version_slug}_{ref_slug}.png",
                    dpi=190,
                    bbox_inches="tight",
                )
            plt.show()

    if singleton_rows_all:
        _write_csv(
            OUT_DIR / f"sharp_singleton_embedding_labels_{_safe_slug(singleton_embedding_version)}_all_references.csv",
            singleton_rows_all,
        )
        sharp_singleton_embedding_table = _as_table(singleton_rows_all)
    else:
        print("No singleton channels available for labeled embedding plot.")
elif embedding is None:
    print("Skipping sharp singleton embedding labels: Step 10 embedding/fingerprints are unavailable.")
else:
    print("Skipping sharp singleton embedding labels: no RR label array is available.")

sharp_singleton_embedding_table


# %% Step 24: post-hoc compression frontier toward a biological-sized RR
# The conservative multi-stimulus RR asks "what can we safely merge?" This cell
# asks a second, explicitly post-hoc question: after those decisions, can we
# merge representatives further using a stimulus-consolidated representative
# similarity matrix, and how much quality is lost as we approach ~128 reps?
compression_frontier_table = None
compression_repcorr_table = None
compressed_candidate_inspection_table = None
compression_frontier_labels: dict[str, np.ndarray] = {}

if RUN_POSTHOC_COMPRESSION_FRONTIER:
    if multistim_final_labels is not None:
        compression_base_labels = np.asarray(multistim_final_labels, dtype=int)
        compression_base_version = multistim_final_version_name
    elif multistim_labels is not None:
        compression_base_labels = np.asarray(multistim_labels, dtype=int)
        compression_base_version = multistim_version_name
    elif patched_labels is not None:
        compression_base_labels = np.asarray(patched_labels, dtype=int)
        compression_base_version = PATCHED_VERSION_NAME
    else:
        compression_base_labels = None
        compression_base_version = "labels_unavailable"

    compression_movies: list[np.ndarray] = []
    compression_case_labels: list[str] = []
    if "multistim_construction_movies" in globals() and multistim_construction_movies:
        compression_movies = list(multistim_construction_movies)
        compression_case_labels = list(multistim_construction_case_labels)
    else:
        if "fixrsvp_activation_movie" in globals():
            compression_movies.append(fixrsvp_activation_movie)
            compression_case_labels.append(fixrsvp_label)
        if heldout_movies:
            compression_movies.extend(heldout_movies)
            compression_case_labels.extend(heldout_case_labels)

    if compression_base_labels is None:
        print("Skipping compression frontier: no base RR labels are available.")
    elif not compression_movies:
        print("Skipping compression frontier: no stimulus movies are available.")
    else:
        base_counts = _candidate_count_row(compression_base_version, compression_base_labels)
        print("Compression base:", base_counts)

        rep_corrs_by_case, repcorr_rows = representative_corrs_from_movies(
            compression_base_labels,
            compression_movies,
            compression_case_labels,
        )
        compression_rep_corr = combine_similarity_matrices(
            rep_corrs_by_case,
            mode=COMPRESSION_SIMILARITY_MODE,
        )
        compression_slug_base = _safe_slug(
            f"{compression_base_version}_posthoc_{COMPRESSION_SIMILARITY_MODE}_{COMPRESSION_LINKAGE_METHOD}"
        )
        _write_csv(
            OUT_DIR / f"compression_representative_corr_inputs_{compression_slug_base}.csv",
            repcorr_rows,
        )
        np.savez_compressed(
            OUT_DIR / f"compression_representative_corr_{compression_slug_base}.npz",
            rep_corr=compression_rep_corr.astype(np.float32),
            thresholds=np.asarray(COMPRESSION_THRESHOLDS, dtype=np.float32),
            case_labels=np.asarray(compression_case_labels, dtype=str),
        )
        compression_repcorr_table = _as_table(repcorr_rows)

        frontier_rows: list[dict[str, Any]] = []
        frontier_reconstruction_rows: list[dict[str, Any]] = []
        frontier_group_rows: list[dict[str, Any]] = []
        frontier_member_rows: list[dict[str, Any]] = []

        for threshold in COMPRESSION_THRESHOLDS:
            compressed_version = (
                f"{compression_base_version}"
                f"_posthoc{COMPRESSION_SIMILARITY_MODE}"
                f"Rep{COMPRESSION_LINKAGE_METHOD}{float(threshold):.2f}"
            ).replace("0.", "0p")
            compressed_labels = merge_representatives_from_corr(
                compression_base_labels,
                compression_rep_corr,
                threshold=float(threshold),
                method=COMPRESSION_LINKAGE_METHOD,
            )
            summary, group_rows, member_rows, reconstruction_rows = audit_redundancy_labels_on_heldout_movies(
                compressed_version,
                compressed_labels,
                compression_movies,
                compression_case_labels,
            )
            summary.update(
                {
                    "base_version": compression_base_version,
                    "base_n_representatives": int(base_counts["n_representatives"]),
                    "threshold": float(threshold),
                    "compression_similarity_mode": COMPRESSION_SIMILARITY_MODE,
                    "compression_linkage_method": COMPRESSION_LINKAGE_METHOD,
                    "target_representatives": int(COMPRESSION_TARGET_REPRESENTATIVES),
                    "distance_to_target": int(abs(int(summary["n_representatives"]) - int(COMPRESSION_TARGET_REPRESENTATIVES))),
                    "n_removed_from_base": int(base_counts["n_representatives"] - int(summary["n_representatives"])),
                }
            )
            summary["passes_promotion_gates"] = bool(
                int(summary["movie_groups_below_0p60"]) <= int(COMPRESSION_PROMOTION_MAX_GROUPS_BELOW_0P60)
                and float(summary["mean_global_reconstruction_corr"]) >= float(COMPRESSION_PROMOTION_MIN_RECONSTRUCTION_CORR)
            )
            frontier_rows.append(summary)
            frontier_group_rows.extend(group_rows)
            frontier_member_rows.extend(member_rows)
            frontier_reconstruction_rows.extend(reconstruction_rows)
            compression_frontier_labels[compressed_version] = compressed_labels

        _write_csv(OUT_DIR / f"compression_frontier_summary_{compression_slug_base}.csv", frontier_rows)
        _write_csv(OUT_DIR / f"compression_frontier_groups_{compression_slug_base}.csv", frontier_group_rows)
        _write_csv(OUT_DIR / f"compression_frontier_members_{compression_slug_base}.csv", frontier_member_rows)
        _write_csv(OUT_DIR / f"compression_frontier_reconstruction_{compression_slug_base}.csv", frontier_reconstruction_rows)

        if SAVE_FIGURES and frontier_rows:
            fig_frontier = _compression_frontier_plot(
                frontier_rows,
                title=(
                    f"Post-hoc compression frontier - {compression_base_version}\n"
                    f"rep corr={COMPRESSION_SIMILARITY_MODE}, linkage={COMPRESSION_LINKAGE_METHOD}"
                ),
            )
            fig_frontier.savefig(
                OUT_DIR / f"compression_frontier_summary_{compression_slug_base}.png",
                dpi=170,
                bbox_inches="tight",
            )
            plt.show()

        passing_rows = [row for row in frontier_rows if bool(row["passes_promotion_gates"])]
        selected_rows: list[dict[str, Any]] = []
        if passing_rows:
            selected_rows.append(
                min(
                    passing_rows,
                    key=lambda row: (
                        int(row["distance_to_target"]),
                        int(row["n_representatives"]),
                        -float(row["threshold"]),
                    ),
                )
            )
        if frontier_rows:
            selected_rows.append(
                min(
                    frontier_rows,
                    key=lambda row: (
                        int(row["distance_to_target"]),
                        int(row["movie_groups_below_0p60"]),
                        -float(row["mean_global_reconstruction_corr"]),
                    ),
                )
            )

        unique_selected: list[dict[str, Any]] = []
        seen_versions: set[str] = set()
        for row in selected_rows:
            version = str(row["version"])
            if version in seen_versions:
                continue
            seen_versions.add(version)
            unique_selected.append(row)

        selected_summary_rows: list[dict[str, Any]] = []
        for row in unique_selected:
            version = str(row["version"])
            labels_for_version = compression_frontier_labels[version]
            slug = _safe_slug(version)
            np.save(OUT_DIR / f"compression_candidate_labels_{slug}.npy", labels_for_version)
            selected_summary_rows.append(dict(row))

            if SAVE_COMPRESSION_POPULATION_SPECS or SAVE_MEDOID_POPULATION_SPECS:
                save_population_spec(labels_for_version, bundle.unit_rows, version, OUT_DIR)

            save_medoid_population_variant(
                labels=labels_for_version,
                unit_rows=bundle.unit_rows,
                version_name=version,
                movies=compression_movies,
                case_labels=compression_case_labels,
                out_dir=OUT_DIR,
                member_rows=[
                    member_row
                    for member_row in frontier_member_rows
                    if str(member_row.get("version")) == version
                ],
            )

            if RUN_COMPRESSION_CANDIDATE_PLOTS:
                corr_for_heatmap = (
                    multistim_corr
                    if "multistim_corr" in globals() and np.asarray(multistim_corr).shape[0] == labels_for_version.size
                    else None
                )
                inspect_rr_candidate_on_movies(
                    version_name=version,
                    labels=labels_for_version,
                    audit_movies=compression_movies,
                    audit_case_labels=compression_case_labels,
                    audit_name="Posthoc_compression_selected",
                    corr_for_heatmap=corr_for_heatmap,
                    unit_rows=bundle.unit_rows if "bundle" in globals() else None,
                )

        if selected_summary_rows:
            _write_csv(
                OUT_DIR / f"compression_selected_candidates_{compression_slug_base}.csv",
                selected_summary_rows,
            )
            compressed_candidate_inspection_table = _as_table(selected_summary_rows)

        compression_frontier_table = _as_table(frontier_rows)
elif not RUN_POSTHOC_COMPRESSION_FRONTIER:
    print("Skipping post-hoc compression frontier: RUN_POSTHOC_COMPRESSION_FRONTIER=False.")

compression_frontier_table
