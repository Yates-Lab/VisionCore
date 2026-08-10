"""Generate model configs for the capacity ladder.

A single scalar `width` scales every channel dimension in the shared core:
the ResNet stem, both residual stages, the modulator embedding, and the
ConvGRU hidden state. Readout parameters are set per-session at build time
and are not touched here.

Channel counts are rounded to a multiple of `ROUND_TO` so that the scaled
models stay tensor-core friendly; the realized parameter count is therefore
not exactly quadratic in `width`. Always read realized counts from
`measure_model()` rather than predicting them.

Usage
-----
    from paper.model_selection.gen_configs import write_ladder
    paths = write_ladder()          # writes configs/width_*.yaml
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from VisionCore.paths import VISIONCORE_ROOT

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"

# The architecture the paper model uses. All ladder rungs are this config
# with the core channel dimensions scaled.
BASE_MODEL_CONFIG = (
    VISIONCORE_ROOT / "experiments" / "model_configs"
    / "learned_resnet_concat_convgru_gaussian.yaml"
)

# Width ladder. 1.0 reproduces the current paper model (4.92M params).
# Spans below and above it so the val-loss curve has a left arm and can be
# pushed past the ~80M saturation point reported for mouse visual cortex
# foundation models (Willeke et al. 2026).
WIDTH_LADDER = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0)

ROUND_TO = 4
MIN_CHANNELS = 4


def _scale(value: int, width: float) -> int:
    """Scale a channel count, rounded to a multiple of ROUND_TO."""
    scaled = value * width
    rounded = int(round(scaled / ROUND_TO) * ROUND_TO)
    return max(MIN_CHANNELS, rounded)


def scale_model_config(config: dict, width: float) -> dict:
    """Return a copy of `config` with all core channel dimensions scaled.

    Scales, in order: ResNet stem output channels, the per-stage residual
    block channels, the modulator feature embedding, and the ConvGRU hidden
    dimension. Kernel sizes, normalization, activations, pooling and the
    regularization schedule are left untouched, so the only thing varying
    across the ladder is capacity.
    """
    cfg = copy.deepcopy(config)

    convnet = cfg["convnet"]["params"]
    convnet["channels"] = [_scale(c, width) for c in convnet["channels"]]
    convnet["stem_config"]["out_channels"] = _scale(
        convnet["stem_config"]["out_channels"], width
    )

    if "modulator" in cfg and cfg["modulator"] is not None:
        mod = cfg["modulator"].get("params", {})
        if "feature_dim" in mod:
            # behavior_dim is set by the dataset (42-d gaze vector); only the
            # learned embedding width scales.
            mod["feature_dim"] = _scale(mod["feature_dim"], width)

    if "recurrent" in cfg and cfg["recurrent"] is not None:
        rec = cfg["recurrent"].get("params", {})
        if "hidden_dim" in rec:
            rec["hidden_dim"] = _scale(rec["hidden_dim"], width)

    return cfg


def strip_adapter(config: dict) -> dict:
    """Return a copy with the per-dataset `AffineAdapter` replaced by identity.

    The adapter exists to reconcile datasets recorded at different spatial
    scales, which was a live concern when a second lab's data was expected. It
    is not one now: every session comes through the same preprocessing.

    What it was actually doing, read from the E1a checkpoint's own weights:
    the learned x/y scales start at `softplus(0) = 0.693` and converge to a
    mean of ~0.99 -- the adapter spends its capacity climbing back to the
    identity resample -- retaining only a ~±20% per-session spread. Its
    Gaussian pre-blur is *fixed* at sigma = 1.0 on all 30 datasets and cannot
    move, because `AffineAdapter._blur` builds its kernel from `sigma.item()`
    and so detaches the graph; `log_sigma` receives no gradient despite the
    method documenting itself as differentiable in sigma.

    Removing it is therefore mostly a simplification, but not purely one: it
    also removes that fixed sigma = 1 pre-blur. `type: none` maps to
    `nn.Identity` in `create_frontend`, so this is a configuration change and
    no model code is deleted -- existing checkpoints keep loading.
    """
    cfg = copy.deepcopy(config)
    cfg["adapter"] = {"type": "none", "params": {}}
    return cfg


def width_name(width: float, suffix: str = "") -> str:
    """Filesystem-safe name for a ladder rung, e.g. 0.25 -> 'width0p25'."""
    return "width" + f"{width:g}".replace(".", "p") + suffix


def set_frontend_channels(config: dict, num_channels: int) -> dict:
    """Return a copy of `config` with the temporal frontend widened.

    The frontend is a learned temporal filter bank -- `num_channels` kernels
    over a 16-frame window -- and it is deliberately *not* scaled by
    `scale_model_config`. The default of 4 is a biological prior: midget and
    parasol, ON and OFF, the retinal channels an achromatic stimulus drives.
    A fixed set of retinal channel types should not grow with cortical
    capacity, so the invariance is intended.

    What is untested is whether the prior binds. Every temporal structure the
    model can represent passes through this basis, and its width relative to
    the blocks falls from 1:8 at width 0.25 to 1:128 at width 4.0 -- so even a
    principled bottleneck becomes a progressively tighter one up the ladder.

    Widening it is therefore a scientific manipulation, not a tuning knob: if
    a wider frontend fits better, the cost of the retinal analogy is what has
    been measured.
    """
    cfg = copy.deepcopy(config)
    params = cfg.get("frontend", {}).get("params")
    if params is None or "num_channels" not in params:
        raise ValueError("frontend has no num_channels to set")
    params["num_channels"] = int(num_channels)
    return cfg


def write_ladder(widths=WIDTH_LADDER, base=BASE_MODEL_CONFIG,
                 out_dir=CONFIG_DIR, with_adapter=True,
                 suffix=None, frontend_channels=None) -> dict[float, Path]:
    """Write one model config per width. Returns {width: path}.

    `with_adapter=False` writes the no-adapter variant under a distinct name
    rather than overwriting the adapter-on rungs, so a run manifest's config
    path says unambiguously which architecture it trained.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if suffix is None:
        suffix = "" if with_adapter else "_noadapter"
        if frontend_channels is not None:
            suffix += f"_fe{int(frontend_channels)}"

    with open(base) as fh:
        base_cfg = yaml.safe_load(fh)
    if not with_adapter:
        base_cfg = strip_adapter(base_cfg)
    if frontend_channels is not None:
        base_cfg = set_frontend_channels(base_cfg, frontend_channels)

    paths = {}
    for width in widths:
        cfg = scale_model_config(base_cfg, width)
        path = out_dir / f"{width_name(width, suffix)}.yaml"
        header = (
            f"# Capacity-ladder rung: width={width:g}\n"
            f"# Generated by paper/model_selection/gen_configs.py\n"
            f"# Base: {Path(base).name}\n"
            f"# Adapter: {'AffineAdapter' if with_adapter else 'none (identity)'}\n"
            + (f"# Frontend: {int(frontend_channels)} temporal channels "
               f"(default 4)\n" if frontend_channels is not None else "")
        )
        with open(path, "w") as fh:
            fh.write(header)
            yaml.safe_dump(cfg, fh, sort_keys=False)
        paths[width] = path
    return paths


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-adapter", action="store_true",
                    help="Write the identity-adapter ladder (_noadapter suffix)")
    args = ap.parse_args()

    for width, path in write_ladder(with_adapter=not args.no_adapter).items():
        print(f"width={width:g}  ->  {path}")
