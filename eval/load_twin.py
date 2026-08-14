"""One way to load a digital twin, for every canonical analysis.

Readout sizes are not stored in older checkpoints. `MultiDatasetModel.__init__`
rebuilds each per-session head to `len(dataset_config['cids'])`, read at load
time from mutable YAMLs under `experiments/dataset_configs/sessions/`. Edit a
session's cids after training and the checkpoint stops loading, with a wall of
`size mismatch for model.readouts.N.*` errors that names no session.

This module makes that failure legible and, where possible, impossible:

* readout sizes come from the checkpoint's own ``state_dict``;
* cids come from the checkpoint's snapshot when it has one, else from YAML,
  which is then used for cell *identity* only;
* a mismatch raises `CidDriftError` naming the drifting session and both
  counts, before any model is built.

Prefer this over `scan_checkpoints` + `model_index=0` for anything canonical:
"best in directory" silently repoints as new checkpoints appear.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

_READOUT_RE = re.compile(r"^model\.readouts\.(\d+)\.(mean|bias)$")


class CidDriftError(RuntimeError):
    """A checkpoint's readouts disagree with the cids resolved for it."""


def readout_sizes_from_state_dict(state_dict):
    """Per-session readout unit counts, taken from the checkpoint itself.

    Parameters
    ----------
    state_dict : dict
        A Lightning checkpoint's `state_dict`.

    Returns
    -------
    list of int
        Unit count per readout, ordered by readout index.
    """
    sizes = {}
    for key, value in state_dict.items():
        m = _READOUT_RE.match(key)
        if m is None:
            continue
        idx = int(m.group(1))
        # `mean` is (n_units, 2); `bias` is (n_units,). Either gives the count,
        # and both agree, so first-writer-wins is fine.
        sizes.setdefault(idx, int(value.shape[0]))

    if not sizes:
        raise ValueError(
            "Checkpoint state_dict contains no readout parameters "
            "(expected keys like 'model.readouts.0.mean'). This does not look "
            "like a MultiDatasetModel checkpoint.")

    # Sort numerically: readouts.10 must not land between readouts.1 and .2.
    return [sizes[i] for i in sorted(sizes)]


def _load_cids_from_config(cfg_path):
    """session name -> cids, resolved from a parent dataset config."""
    from models.config_loader import load_dataset_configs

    cfgs = load_dataset_configs(str(cfg_path))
    return {c["session"]: list(c["cids"]) for c in cfgs}


def resolve_checkpoint_cids(checkpoint, dataset_configs_path=None):
    """Resolve per-session cids for a checkpoint.

    Parameters
    ----------
    checkpoint : dict
        Loaded checkpoint (needs `hyper_parameters`).
    dataset_configs_path : str or Path, optional
        Override for the parent dataset config. Use when the recorded `cfg_dir`
        points somewhere that no longer exists, e.g. another user's home
        directory.

    Returns
    -------
    cids_by_session : dict
    source : {'checkpoint', 'yaml'}
        Where the cids came from. 'checkpoint' is self-contained and immune to
        later YAML edits.
    """
    hparams = checkpoint.get("hyper_parameters", {}) or {}

    snapshot = hparams.get("dataset_cids", None)
    if snapshot and dataset_configs_path is None:
        return {k: list(v) for k, v in snapshot.items()}, "checkpoint"

    cfg_path = dataset_configs_path or hparams.get("cfg_dir", None)
    if cfg_path is None:
        raise ValueError(
            "Checkpoint has neither a `dataset_cids` snapshot nor a `cfg_dir` "
            "hparam, so its readout population cannot be resolved. Pass "
            "dataset_configs_path= explicitly.")

    return _load_cids_from_config(cfg_path), "yaml"


def validate_readout_sizes(cids_by_session, readout_sizes, checkpoint_path):
    """Raise `CidDriftError` if resolved cids disagree with the checkpoint.

    Parameters
    ----------
    cids_by_session : dict
        session name -> cids, in the config's session order.
    readout_sizes : list of int
        Per-readout unit counts from the checkpoint's state_dict.
    checkpoint_path : str or Path
        Only used in the error message.
    """
    names = list(cids_by_session)

    if len(names) != len(readout_sizes):
        raise CidDriftError(
            f"Checkpoint {checkpoint_path} has {len(readout_sizes)} readouts "
            f"but the dataset config resolves {len(names)} sessions. The "
            f"config has gained or lost sessions since training; point at the "
            f"config that produced this checkpoint.")

    drift = [
        (name, len(cids_by_session[name]), size)
        for name, size in zip(names, readout_sizes)
        if len(cids_by_session[name]) != size
    ]
    if drift:
        lines = "\n".join(
            f"  session {name}: yaml {n_yaml} vs ckpt {n_ckpt}"
            for name, n_yaml, n_ckpt in drift)
        raise CidDriftError(
            f"Readout population drift in {len(drift)} of {len(names)} "
            f"sessions for checkpoint {checkpoint_path}:\n{lines}\n"
            f"The session YAMLs were edited after this checkpoint was trained. "
            f"Check out the config revision that produced it, or retrain.")


def load_twin(checkpoint_path, device="cuda", dataset_configs_path=None,
              verbose=True):
    """Load a digital twin from an explicit, pinned checkpoint path.

    Parameters
    ----------
    checkpoint_path : str or Path
        Explicit path. Never "best in directory" for a canonical result.
    device : str
        Device to move the model to.
    dataset_configs_path : str or Path, optional
        Override the parent dataset config recorded in the checkpoint.
    verbose : bool
        Print a short summary.

    Returns
    -------
    model : MultiDatasetModel
    info : dict
        `path`, `epoch`, `cids_by_session`, `cids_source`, `readout_sizes`,
        `n_units`, `behavior_dim`, `modulator_type`.
    """
    from eval.eval_stack_multidataset import load_model

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu",
                            weights_only=False)
    if "state_dict" not in checkpoint:
        raise ValueError(f"{checkpoint_path} has no state_dict.")

    readout_sizes = readout_sizes_from_state_dict(checkpoint["state_dict"])
    cids_by_session, cids_source = resolve_checkpoint_cids(
        checkpoint, dataset_configs_path=dataset_configs_path)

    # Fail here, with a session name, rather than inside load_state_dict.
    validate_readout_sizes(cids_by_session, readout_sizes, checkpoint_path)

    model, model_info = load_model(checkpoint_path=str(checkpoint_path),
                                   device=device, verbose=verbose)

    model_cfg = (checkpoint.get("hyper_parameters", {}) or {}).get(
        "model_config_dict", {}) or {}
    modulator = (model_cfg.get("modulator", {}) or {})

    info = dict(model_info)
    info.update({
        "cids_by_session": cids_by_session,
        "cids_source": cids_source,
        "readout_sizes": readout_sizes,
        "n_units": sum(readout_sizes),
        "modulator_type": modulator.get("type", None),
        "behavior_dim": (modulator.get("params", {}) or {}).get("behavior_dim", None),
    })

    if verbose:
        print(f"  cids from {cids_source}; {len(readout_sizes)} sessions, "
              f"{info['n_units']} units; modulator={info['modulator_type']}")
        if cids_source == "yaml":
            print("  NOTE: this checkpoint predates cid snapshotting, so its "
                  "readout population depends on the session YAMLs staying put.")

    return model, info
