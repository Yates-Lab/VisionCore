"""
Portable loaders for redundancy-resolved V1 twin population specs.

Typical notebook usage::

    from declan.redundancy_resolved_v1_population import (
        apply_population_pooling,
        load_population_view,
    )

    view = load_population_view()      # default: current selected reduced-population spec
    reduced = apply_population_pooling(full_movie_tchw, view.membership)
    # reduced: (T, view.n_units, H, W)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for _p in (str(ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_DEFAULT_SPEC_DIR = ROOT / "outputs" / "redundancy_resolved_v1_twin" / "step1_activation_fingerprints"
_DEFAULT_VERSION_NAME = "V1-RR_complete_0p65_moviesplit0p75_pair0p60_rec4_blockjkP0p50n5L0p50n4_merge2nd1.01"


def _safe_slug(value: object, max_len: int = 96) -> str:
    text = str(value)
    text = Path(text).stem if "/" in text else text
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    slug = "_".join(part for part in slug.split("_") if part)
    return slug[:max_len] or "unnamed"


def _strip_population_prefix(stem: str) -> str:
    prefix = "population_spec_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


@dataclass(frozen=True)
class PopulationView:
    """A post-activation population transform for canonical twin rate maps."""

    name: str
    membership: np.ndarray | None
    input_channels: int
    n_units: int
    meta: dict[str, Any]
    labels: np.ndarray | None = None
    cluster_membership: np.ndarray | None = None


@dataclass
class ModelBundle:
    model: Any
    readout: Any
    outputs: list
    unit_rows: list
    device: str


@dataclass
class PopulationBundle:
    model: Any
    readout: Any
    outputs: list
    unit_rows: list
    device: str
    population: PopulationView


def _coerce_spec_dir(spec_dir: Path | str | None = None) -> Path:
    return Path(spec_dir) if spec_dir is not None else _DEFAULT_SPEC_DIR


def _spec_error_message(spec_dir: Path, version_name: str) -> str:
    available = list_population_specs(spec_dir)
    if available:
        choices = "\n".join(
            f"  - {row['version']} ({row['n_representatives']} reps; "
            f"pooling={row['pooling_mode']}; slug={row['slug']})"
            for row in available
        )
    else:
        choices = "  - none found"
    return (
        f"Population spec for {version_name!r} was not found in {spec_dir}.\n"
        "Run inspect_canonical_twin_channel_redundancy.py through the population-spec save step, "
        "or pass version_name=... for one of:\n"
        f"{choices}"
    )


def list_population_specs(spec_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """List saved ``population_spec_*.npz`` files with their JSON metadata."""
    spec_dir = _coerce_spec_dir(spec_dir)
    rows: list[dict[str, Any]] = []
    for npz_path in sorted(spec_dir.glob("population_spec_*.npz")):
        slug = _strip_population_prefix(npz_path.stem)
        json_path = npz_path.with_suffix(".json")
        meta: dict[str, Any] = {}
        if json_path.exists():
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(npz_path) as data:
            membership_shape = tuple(int(v) for v in data["membership"].shape)
            labels_shape = tuple(int(v) for v in data["labels"].shape) if "labels" in data else None
        rows.append(
            {
                "version": str(meta.get("version", slug)),
                "slug": slug,
                "n_representatives": int(meta.get("n_representatives", membership_shape[0])),
                "n_input_channels": int(meta.get("n_input_channels", membership_shape[1])),
                "pooling_mode": str(meta.get("pooling_mode", "mean")),
                "membership_shape": membership_shape,
                "labels_shape": labels_shape,
                "npz_path": npz_path,
                "json_path": json_path if json_path.exists() else None,
                "meta": meta,
            }
        )
    return rows


def resolve_population_spec_paths(
    spec_dir: Path | str | None = None,
    *,
    version_name: str | None = None,
) -> tuple[Path, Path | None]:
    """
    Resolve a population spec by human version name or filename slug.

    The inspection script uses ``_safe_slug`` for filenames, so a version like
    ``merge2nd1.01`` is saved as ``merge2nd1_01`` on disk.
    """
    spec_dir = _coerce_spec_dir(spec_dir)
    version_name = _DEFAULT_VERSION_NAME if version_name is None else str(version_name)
    candidate_slugs = list(dict.fromkeys([version_name, _safe_slug(version_name)]))
    for slug in candidate_slugs:
        npz_path = spec_dir / f"population_spec_{slug}.npz"
        if npz_path.exists():
            json_path = npz_path.with_suffix(".json")
            return npz_path, json_path if json_path.exists() else None

    matches = [
        row
        for row in list_population_specs(spec_dir)
        if row["version"] == version_name
        or row["slug"] == version_name
        or row["slug"] == _safe_slug(version_name)
    ]
    if len(matches) == 1:
        row = matches[0]
        return Path(row["npz_path"]), row["json_path"]
    if len(matches) > 1:
        choices = ", ".join(str(row["npz_path"]) for row in matches)
        raise RuntimeError(f"Ambiguous population spec for {version_name!r}: {choices}")
    raise FileNotFoundError(_spec_error_message(spec_dir, version_name))


def load_population_spec(
    spec_dir: Path | str | None = None,
    *,
    version_name: str | None = None,
) -> dict[str, Any]:
    """
    Load the population spec from the NPZ and JSON files produced by the
    redundancy-inspection script (Step 15).

    Returns a dict with keys:
      - ``membership``  : float32 array of shape (N_reps, 756), pre-normalised population transform.
                          Mean-pooled specs average grouped channels; medoid specs are one-hot.
                          ``reduced = membership @ full_channels_flat``
      - ``labels``      : int array of shape (756,), group assignment (-1 = singleton)
      - ``meta``        : dict parsed from the JSON spec (version, n_representatives, ...)
      - ``spec_dir``    : Path where the files were found
    """
    spec_dir = _coerce_spec_dir(spec_dir)
    npz_path, json_path = resolve_population_spec_paths(spec_dir, version_name=version_name)
    with np.load(npz_path) as data:
        membership = np.asarray(data["membership"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int32)
        saved_cluster_membership = (
            np.asarray(data["cluster_membership"], dtype=np.float32)
            if "cluster_membership" in data
            else None
        )
    meta: dict[str, Any]
    if json_path is not None and json_path.exists():
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        meta = {"version": _strip_population_prefix(npz_path.stem)}
    cluster_membership = (
        saved_cluster_membership
        if saved_cluster_membership is not None
        else population_cluster_membership(labels, meta, n_reps=membership.shape[0])
    )
    if cluster_membership.shape != membership.shape:
        raise ValueError(
            "cluster_membership shape does not match membership shape: "
            f"{cluster_membership.shape} vs {membership.shape}"
        )
    spec: dict[str, Any] = {
        "membership": membership,
        "labels": labels,
        "cluster_membership": cluster_membership,
        "spec_dir": spec_dir,
        "npz_path": npz_path,
        "json_path": json_path,
        "meta": meta,
    }

    return spec


def _member_lists_from_labels(labels: np.ndarray) -> list[list[int]]:
    labels_arr = np.asarray(labels, dtype=np.int32)
    group_ids = sorted(set(int(v) for v in labels_arr[labels_arr >= 0]))
    member_lists = [list(map(int, np.flatnonzero(labels_arr == group_id))) for group_id in group_ids]
    member_lists.extend([[int(ch)] for ch in np.flatnonzero(labels_arr == -1)])
    return member_lists


def population_cluster_membership(
    labels: np.ndarray,
    meta: dict[str, Any] | None = None,
    *,
    n_reps: int | None = None,
) -> np.ndarray:
    """
    Build a representative-to-input-channel assignment matrix.

    This differs from ``membership`` for medoid specs. ``membership`` is the
    actual pooling transform and is one-hot for medoids; this matrix records all
    channels that belong to each cluster, with rows normalized to average the
    full cluster. Diagnostics, group-size summaries, and expand-to-full audits
    should use this matrix.
    """
    labels_arr = np.asarray(labels, dtype=np.int32)
    meta = dict(meta or {})
    reps = meta.get("representatives")
    if reps:
        inferred_n_reps = max(int(rep["rep_idx"]) for rep in reps) + 1
        if n_reps is None:
            n_reps = inferred_n_reps
        member_lists: list[list[int]] = [[] for _ in range(int(n_reps))]
        for rep in reps:
            rep_idx = int(rep["rep_idx"])
            if rep_idx < 0 or rep_idx >= int(n_reps):
                raise ValueError(f"Representative index {rep_idx} outside n_reps={n_reps}")
            member_lists[rep_idx] = [int(ch) for ch in rep.get("members", [])]
    else:
        member_lists = _member_lists_from_labels(labels_arr)
        if n_reps is None:
            n_reps = len(member_lists)

    if n_reps is None:
        n_reps = len(member_lists)
    if len(member_lists) != int(n_reps):
        raise ValueError(f"Expected {n_reps} representative member lists, got {len(member_lists)}")

    cluster_membership = np.zeros((int(n_reps), int(labels_arr.size)), dtype=np.float32)
    for rep_idx, members in enumerate(member_lists):
        if not members:
            continue
        members_arr = np.asarray(members, dtype=np.int64)
        if np.any(members_arr < 0) or np.any(members_arr >= labels_arr.size):
            raise ValueError(f"Representative {rep_idx} has members outside channel range: {members}")
        cluster_membership[rep_idx, members_arr] = 1.0 / float(members_arr.size)
    return cluster_membership


def load_population_view(
    spec_dir: Path | str | None = None,
    *,
    version_name: str | None = None,
) -> PopulationView:
    """Load a saved redundancy-resolved population as a post-activation view."""
    spec = load_population_spec(spec_dir, version_name=version_name)
    membership = np.asarray(spec["membership"], dtype=np.float32)
    labels = np.asarray(spec["labels"], dtype=np.int32)
    cluster_membership = np.asarray(spec["cluster_membership"], dtype=np.float32)
    meta = dict(spec.get("meta", {}))
    return PopulationView(
        name=str(meta.get("version", _DEFAULT_VERSION_NAME)),
        membership=membership,
        input_channels=int(membership.shape[1]),
        n_units=int(membership.shape[0]),
        meta=meta,
        labels=labels,
        cluster_membership=cluster_membership,
    )


def full_population_view(n_channels: int, name: str | None = None) -> PopulationView:
    """Identity population view for the canonical full readout."""
    return PopulationView(
        name=name or f"full_{int(n_channels)}",
        membership=None,
        input_channels=int(n_channels),
        n_units=int(n_channels),
        meta={"version": name or f"full_{int(n_channels)}", "n_input_channels": int(n_channels)},
        labels=None,
        cluster_membership=None,
    )


def apply_population_pooling(
    activation_movie: np.ndarray,
    membership: np.ndarray,
) -> np.ndarray:
    """
    Reduce a full T x C x H x W activation movie to T x N_rr x H x W.

    A single snapshot with shape C x H x W is also accepted and returns
    N_rr x H x W.

    ``membership`` is the (N_reps, C) pre-normalised matrix from ``load_population_spec()``.
    Mean-pooled specs produce group centroids; medoid specs select one actual
    modeled unit per group; singletons pass through.

    Parameters
    ----------
    activation_movie : (T, C, H, W) or (C, H, W) float array, full twin output
    membership       : (N_reps, C) float32, from spec["membership"]

    Returns
    -------
    (T, N_reps, H, W) or (N_reps, H, W) float32
    """
    return apply_population_view(activation_movie, membership)


def _apply_membership_numpy(
    activation_movie: np.ndarray,
    membership: np.ndarray,
) -> np.ndarray:
    movie = np.asarray(activation_movie, dtype=np.float32)
    mem = np.asarray(membership, dtype=np.float32)

    squeeze_time = False
    if movie.ndim == 3:
        movie = movie[np.newaxis]
        squeeze_time = True
    elif movie.ndim != 4:
        raise ValueError(f"Expected (T, C, H, W) or (C, H, W), got {movie.shape}")
    T, C, H, W = movie.shape
    n_rr, n_ch = mem.shape
    if n_ch != C:
        raise ValueError(f"membership has {n_ch} input channels, movie has {C}")

    flat = movie.reshape(T, C, H * W)               # (T, C, H*W)
    reduced_flat = np.einsum("rc,tcs->trs", mem, flat)  # (T, N_rr, H*W)
    reduced = reduced_flat.reshape(T, n_rr, H, W)
    return reduced[0] if squeeze_time else reduced


def _apply_membership_torch(activation_movie: Any, membership: np.ndarray) -> Any:
    import torch

    movie = activation_movie
    mem = torch.as_tensor(membership, dtype=movie.dtype, device=movie.device)

    squeeze_time = False
    if movie.ndim == 3:
        movie = movie.unsqueeze(0)
        squeeze_time = True
    elif movie.ndim != 4:
        raise ValueError(f"Expected (T, C, H, W) or (C, H, W), got {tuple(movie.shape)}")
    T, C, H, W = movie.shape
    n_rr, n_ch = mem.shape
    if n_ch != C:
        raise ValueError(f"membership has {n_ch} input channels, movie has {C}")

    flat = movie.reshape(T, C, H * W)
    reduced_flat = torch.einsum("rc,tcs->trs", mem, flat)
    reduced = reduced_flat.reshape(T, n_rr, H, W)
    return reduced[0] if squeeze_time else reduced


def apply_population_view(
    activation_movie: Any,
    population: PopulationView | np.ndarray,
) -> Any:
    """
    Apply a population view to a post-activation rate map.

    ``activation_movie`` may be numpy or torch and may have shape
    ``(T, C, H, W)`` or ``(C, H, W)``.  Pooling happens after the model
    activation, so a reduced population represents a mean of rates, not a mean of logits.
    """
    membership = population.membership if isinstance(population, PopulationView) else population
    if membership is None:
        return activation_movie

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.is_tensor(activation_movie):
        return _apply_membership_torch(activation_movie, np.asarray(membership, dtype=np.float32))
    return _apply_membership_numpy(activation_movie, np.asarray(membership, dtype=np.float32))


def cubes_to_visioncore_stim(cubes: np.ndarray):
    """Convert Jake lag cubes to VisionCore's normalized ``(T, 1, lag, H, W)`` input."""
    import torch

    arr = np.asarray(cubes, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[:, None]
    elif arr.ndim != 5:
        raise ValueError(f"Expected lag cubes with shape (T, lag, H, W), got {arr.shape}")
    return torch.from_numpy((arr - 127.0) / 255.0)


def compute_population_rate_map_batched(
    model: Any,
    readout: Any,
    stim: Any,
    population: PopulationView | None = None,
    *,
    batch_size: int = 32,
) -> Any:
    """
    Compute canonical rate maps, then apply an optional post-activation population view.

    This is the safest plug-in point for reduced populations because the
    canonical ``compute_rate_map_batched`` already applies the model activation.
    """
    from spatial_info import compute_rate_map_batched

    full_rate_map = compute_rate_map_batched(model, readout, stim, batch_size=int(batch_size))
    if population is None:
        return full_rate_map
    return apply_population_view(full_rate_map, population)


def load_canonical_twin_bundle(device: str | None = None, mode: str = "standard"):
    """
    Load the full 756-channel ModelBundle without importing the redundancy script.

    The redundancy script has no __main__ guard, so importing it would execute all
    its top-level analysis code.  This function replicates the three steps of
    load_model_bundle() directly: load model, load mcfarland outputs, build readout.

    Parameters
    ----------
    device : "cpu", "cuda", "cuda:0", etc., or None to auto-detect
    mode   : model mode string passed to scripts.utils.get_model_and_dataset_configs
             (default "standard")
    """
    import pickle
    import torch

    from spatial_info import get_spatial_readout
    from utils import get_model_and_dataset_configs

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # mcfarland output candidates (same list as the redundancy script)
    _candidates = (
        ROOT / "scripts" / "mcfarland_outputs_mono.pkl",
        ROOT / "scripts" / "mcfarland_outputs.pkl",
    )

    def _load_outputs() -> list[dict[str, Any]]:
        for path in _candidates:
            if path.exists():
                with open(path, "rb") as fh:
                    try:
                        import dill
                        fh.seek(0)
                        obj = dill.load(fh)
                    except Exception:
                        fh.seek(0)
                        obj = pickle.load(fh)
                if not isinstance(obj, list):
                    raise TypeError(f"Expected list in {path}, got {type(obj).__name__}")
                return obj
        raise FileNotFoundError(
            f"McFarland outputs not found. Tried: {[str(p) for p in _candidates]}"
        )

    model, _ = get_model_and_dataset_configs(mode=mode)
    model = model.to(device).eval()
    outputs = _load_outputs()
    readout, unit_rows = get_spatial_readout(model, outputs, return_unit_rows=True)
    readout = readout.to(device).eval()
    return ModelBundle(model=model, readout=readout, outputs=outputs, unit_rows=unit_rows, device=device)


def load_population_bundle(
    *,
    population: str | PopulationView = "reduced",
    device: str | None = None,
    mode: str = "standard",
    spec_dir: Path | str | None = None,
    version_name: str | None = None,
) -> PopulationBundle:
    """
    Load the canonical twin plus a named population view.

    ``population="full"`` returns an identity view over all canonical channels.
    ``population="reduced"`` returns the default redundancy-resolved
    post-activation rate pooling view. Pass ``version_name=...`` to select a
    specific saved spec.
    """
    bundle = load_canonical_twin_bundle(device=device, mode=mode)
    if isinstance(population, PopulationView):
        view = population
    elif str(population).lower() in {"full", "full756", "canonical"}:
        view = full_population_view(len(bundle.unit_rows), name="full")
    elif str(population).lower() in {"reduced", "redundancy_resolved", "redundancy-resolved"}:
        view = load_population_view(spec_dir, version_name=version_name)
    else:
        view = load_population_view(spec_dir, version_name=str(population))

    if view.input_channels != len(bundle.unit_rows):
        raise ValueError(
            f"Population view expects {view.input_channels} channels, "
            f"canonical bundle has {len(bundle.unit_rows)}."
        )
    return PopulationBundle(
        model=bundle.model,
        readout=bundle.readout,
        outputs=bundle.outputs,
        unit_rows=bundle.unit_rows,
        device=bundle.device,
        population=view,
    )
