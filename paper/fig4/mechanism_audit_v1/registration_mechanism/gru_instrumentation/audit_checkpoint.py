#!/usr/bin/env python3
"""Audit exact Figure 4 ConvGRU equations and recurrent offset kernels on CPU."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.modules.recurrent import ConvGRUCell
from paper.fig4.mechanism_audit_v1.causal_low_rank.common import sha256_file, write_json
from paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.equations import (
    instrument_convgru_step,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.kernels import (
    RECURRENT_PARAMETER_NAMES,
    analyze_checkpoint_kernels,
    write_kernel_csv,
)
from paper.fig4.mechanism_audit_v1.registration_mechanism.gru_instrumentation.registration import (
    FIG4_CONVGRU_INPUT_LAG_SUPPORTS,
    derive_fig4_convgru_input_lag_supports,
    support_midpoint_lags,
)
from paper.fig4.upstream.run_real_trace_matrix import (
    MODEL_CHECKPOINT_PATH,
    MODEL_CHECKPOINT_SHA256,
)


DEFAULT_OUT = ROOT / "outputs/figures/fig4/mechanism_audit_v1/registration_mechanism_v1"
CAUSAL_FITS = ROOT / "outputs/figures/fig4/mechanism_audit_v1/causal_low_rank_v1/fits/crossval"
CONTRASTS = ("low_0_to_2", "high_0_to_1", "high_1_to_3")
FOLDS = (0, 1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--equations-only",
        action="store_true",
        help="Write the exact equation audit without requiring all 12 fold projectors.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_hash = sha256_file(path)
    if path.resolve() == Path(MODEL_CHECKPOINT_PATH).resolve() and observed_hash != MODEL_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Frozen Figure 4 checkpoint hash changed: {observed_hash} != {MODEL_CHECKPOINT_SHA256}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint lacks a state_dict")
    hparams = checkpoint.get("hyper_parameters", {})
    return state, {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": observed_hash,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "model_config_dict": hparams.get("model_config_dict"),
    }


def cell_from_checkpoint(state: dict[str, torch.Tensor]) -> ConvGRUCell:
    prefix = "model.recurrent.cells.0."
    update = state[prefix + "update_gate.weight"]
    hidden_channels = int(update.shape[0])
    input_channels = int(update.shape[1] - hidden_channels)
    kernel_size = int(update.shape[-1])
    if update.shape != (128, 384, 3, 3):
        raise RuntimeError(f"Frozen checkpoint ConvGRU shape changed: {tuple(update.shape)}")
    cell = ConvGRUCell(input_channels, hidden_channels, kernel_size)
    cell.load_state_dict(
        {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        },
        strict=True,
    )
    return cell.eval()


def numerical_equation_check(cell: ConvGRUCell) -> dict[str, float]:
    generator = torch.Generator(device="cpu").manual_seed(20260812)
    x = torch.randn(2, 256, 11, 13, generator=generator)
    h = torch.randn(2, 128, 11, 13, generator=generator)
    with torch.no_grad():
        terms = instrument_convgru_step(cell, x, h, verify=True)
        literal = cell(x, h)
    return {
        "h_reconstruction_max_abs": float((terms.h_t - literal).abs().max()),
        "retained_plus_new_max_abs": float(
            (
                terms.retained_state_contribution
                + terms.new_state_contribution
                - terms.h_t
            ).abs().max()
        ),
        "candidate_split_max_abs": float(
            (
                terms.candidate_current_preactivation
                + terms.candidate_recurrent_preactivation
                - terms.candidate_preactivation
            ).abs().max()
        ),
    }


def discover_projectors(*, complete: bool) -> list[dict[str, object]]:
    projectors: list[dict[str, object]] = []
    missing: list[Path] = []
    for contrast in CONTRASTS:
        for fold in FOLDS:
            path = CAUSAL_FITS / contrast / f"fold_{fold}" / "rank_008/U.npy"
            if not path.is_file():
                missing.append(path)
                continue
            projectors.append(
                {
                    "basis": np.asarray(np.load(path), dtype=np.float32),
                    "projector": "learned_p",
                    "contrast": contrast,
                    "fold": fold,
                    "source_path": str(path),
                }
            )
    if complete and missing:
        raise RuntimeError(
            "Kernel inference requires all 12 fold-specific projectors; missing:\n"
            + "\n".join(str(path) for path in missing)
        )
    return projectors


def equation_markdown(provenance: dict[str, Any], checks: dict[str, float]) -> str:
    recurrent_source = Path(inspect.getsourcefile(ConvGRUCell) or "")
    source_hash = sha256_file(recurrent_source)
    supports = ", ".join(f"[{lo}..{hi}]" for lo, hi in FIG4_CONVGRU_INPUT_LAG_SUPPORTS)
    anchors = ", ".join(f"{value:g}" for value in support_midpoint_lags())
    return f"""# Exact ConvGRU equation audit

## Frozen implementation

- Checkpoint: `{provenance['checkpoint']}`
- Checkpoint SHA-256: `{provenance['checkpoint_sha256']}`
- Checkpoint epoch: {provenance['checkpoint_epoch']}
- Source: `{recurrent_source}`
- Source SHA-256: `{source_hash}`
- One cell; current input 256 channels; hidden state 128 channels; all three kernels 3×3 with PyTorch zero padding of one pixel.
- Serialized kernels are exactly `(128, 384, 3, 3)`, ordered as 256 current-input channels followed by 128 hidden channels.

The source uses the following equations (asterisk is PyTorch spatial cross-correlation):

```text
a_z,recurrent_t = K_zh * h_(t-1)
a_z,current_t   = literal_z_t - a_z,recurrent_t
z_t = sigmoid(a_z,current_t + a_z,recurrent_t)
a_r,recurrent_t = K_rh * h_(t-1)
a_r,current_t   = literal_r_t - a_r,recurrent_t
r_t = sigmoid(a_r,current_t + a_r,recurrent_t)
a_current_t   = literal_n_t - a_recurrent_t
a_recurrent_t = K_nh * (r_t ⊙ h_(t-1))
n_t = tanh(a_current_t + a_recurrent_t)
retained_t = (1 - z_t) ⊙ h_(t-1)
new_t      = z_t ⊙ n_t
h_t = retained_t + new_t
```

Here `literal_n_t` is the checkpoint's literal fused convolution on
`[x_t, r_t ⊙ h_(t-1)]`.  Thus `a_recurrent_t` remains exactly the direct,
bias-free hidden-kernel contribution, whereas `a_current_t` owns the candidate
bias and the small floating-point cross-accumulation correction required to
reconstruct the fused result.  The same decomposition is used for update and
reset preactivations, and literal fused values remain canonical for all gates.

The module's object name `update_gate` can be misleading: **`z_t` is the
candidate-write fraction**, while `(1-z_t)` retains the old state.  Candidate
input bias is contained in `a_current_t` and is added exactly once.

The deterministic CPU test reconstructed the literal hidden update with maximum
absolute error `{checks['h_reconstruction_max_abs']:.8g}` and reconstructed
`h_t = retained_t + new_t` with error
`{checks['retained_plus_new_max_abs']:.8g}`.  The direct recurrent term plus the
literal-minus-recurrent current residual reconstructed the candidate
preactivation with maximum error `{checks['candidate_split_max_abs']:.8g}`.
The literal concatenated convolution remains the canonical value used to
advance the recurrent state.

## Exact internal-time convention

The corrected model input is a 32-lag tensor ordered `lag 0 = current` through
`lag 31 = oldest`.  The learned valid temporal convolution (kernel 16) and the
first ResNet block's `MaxPool3d(stride=2)` reduce this to **eight** ConvGRU
steps.  The ConvGRU loop consumes those steps in array order, so it progresses
from relatively newer evidence toward relatively older evidence **within each
independently scored 32-lag window**.  It is not recurrence across the 40
scored movie outputs.

The exact structural union of retinal-lag supports for steps 0..7 is:

```text
{supports}
```

Their nominal structural midpoint lags are `{anchors}` samples.  Because each
input is a nonlinear mixture over a broad support, these midpoints are only
predeclared anchors; sign and feature-pixel scale must be measured by the
synthetic translation calibration before comparing inferred transport with
eye displacement.

## Kernel-offset convention

PyTorch `Conv2d` executes cross-correlation.  The offset `(dy,dx)` in
`kernel_offset_energy.csv` therefore labels the sampled input location
`h[y+dy,x+dx]` contributing to output location `(y,x)`.  Off-center energy is
descriptive and is not, by itself, evidence for spatial registration.
"""


def run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state, provenance = load_checkpoint(args.checkpoint)
    cell = cell_from_checkpoint(state)
    checks = numerical_equation_check(cell)
    derived_supports = derive_fig4_convgru_input_lag_supports(
        provenance["model_config_dict"]
    )
    report = equation_markdown(provenance, checks)
    (args.output_dir / "gru_equation_audit.md").write_text(report, encoding="utf-8")
    projectors = discover_projectors(complete=not args.equations_only)
    kernel_rows: list[dict[str, object]] = []
    if projectors:
        kernel_rows = analyze_checkpoint_kernels(state, projectors)
        projector_set_complete = len(projectors) == len(CONTRASTS) * len(FOLDS)
        for row in kernel_rows:
            row["projector_set_complete"] = projector_set_complete
            row["projectors_observed"] = len(projectors)
            row["projectors_expected"] = len(CONTRASTS) * len(FOLDS)
        write_kernel_csv(args.output_dir / "kernel_offset_energy.csv", kernel_rows)
    payload = {
        **provenance,
        "source": str(Path(inspect.getsourcefile(ConvGRUCell) or "").resolve()),
        "source_sha256": sha256_file(Path(inspect.getsourcefile(ConvGRUCell) or "")),
        "cell_shapes": {
            label: list(state[name].shape)
            for label, name in RECURRENT_PARAMETER_NAMES.items()
        },
        "equation_reconstruction": checks,
        "input_lag_order": "0=current through 31=oldest",
        "convgru_steps": 8,
        "convgru_input_lag_supports": derived_supports,
        "convgru_input_lag_supports_derived_from_checkpoint_config": True,
        "projectors_analyzed": len(projectors),
        "kernel_offset_rows": len(kernel_rows),
        "equations_only": bool(args.equations_only),
    }
    write_json(args.output_dir / "checkpoint_gru_audit.json", payload)
    print(report)
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
