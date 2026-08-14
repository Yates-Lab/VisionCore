"""Measured (not predicted) cost of a model configuration.

Everything here runs the real model on a real batch. Nothing is extrapolated
from parameter counts: gradient checkpointing trades compute for memory and
changes the FLOP count, ConvGRU activation memory scales with sequence length
in ways that are awkward to predict analytically, and channel rounding in
`gen_configs` breaks the clean quadratic scaling you would otherwise assume.

Three measurements:
  count_params  -- parameter counts by component
  measure_flops -- forward and forward+backward FLOPs for one batch
  measure_step  -- peak GPU memory and wall-clock time for a full optimizer step
"""
from __future__ import annotations

import time
from contextlib import nullcontext

import torch


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
def count_params(model) -> dict:
    """Parameter counts, split into shared core and per-session readouts."""
    inner = getattr(model, "model", model)
    total = sum(p.numel() for p in inner.parameters())
    readout = sum(
        p.numel() for n, p in inner.named_parameters() if n.startswith("readouts")
    )
    per_component = {}
    for name in ("adapter", "frontend", "convnet", "modulator", "recurrent", "readouts"):
        per_component[name] = sum(
            p.numel() for n, p in inner.named_parameters() if n.startswith(name)
        )
    return {
        "params_total": total,
        "params_core": total - readout,
        "params_readout": readout,
        **{f"params_{k}": v for k, v in per_component.items()},
    }


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------
def measure_flops(model, batch) -> dict:
    """Measure forward and forward+backward FLOPs for one training step.

    `batch` is whatever the data module yields: a single dict, or (the
    default multi-session path) a list of per-session dicts that together
    make up one optimizer step. Both are handled, and the returned count
    covers the whole step.

    Uses torch's FlopCounterMode, which counts actually-dispatched
    matmul/conv work. Under gradient checkpointing the backward count
    includes the recomputed forward, which is what belongs on the x-axis of
    a compute-scaling plot.
    """
    from torch.utils.flop_counter import FlopCounterMode

    inner = getattr(model, "model", model)
    batches = [batch] if isinstance(batch, dict) else batch

    # Forward only.
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        for b in batches:
            inner(b["stim"], int(b["dataset_idx"][0]), beh=b.get("behavior"))
    fwd = counter.get_total_flops()

    # Forward + backward, through the real training step so that the count
    # includes the loss and regularization terms.
    counter = FlopCounterMode(display=False)
    with counter:
        loss = model.training_step(batch, 0)
        loss.backward()
    fwd_bwd = counter.get_total_flops()

    inner.zero_grad(set_to_none=True)
    return {"flops_forward": fwd, "flops_fwd_bwd": fwd_bwd}


# ---------------------------------------------------------------------------
# Memory and throughput
# ---------------------------------------------------------------------------
def measure_step(model, batch, device, n_warmup: int = 2, n_timed: int = 5,
                 lr: float = 1e-3, precision: str = "bf16-mixed") -> dict:
    """Run real optimizer steps; report peak memory and median step time.

    Creates a real AdamW optimizer so that optimizer state memory (two
    moments per parameter) is included in the peak. Uses `training_step` so
    the measured cost includes the loss and the regularization terms, not
    just the forward pass.

    Returns peak memory in GiB and step time in seconds. Raises
    torch.cuda.OutOfMemoryError if the configuration does not fit; callers
    are expected to catch it and record the failure.
    """
    inner = getattr(model, "model", model)
    optimizer = torch.optim.AdamW(inner.parameters(), lr=lr)

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "bf16-mixed"
        else nullcontext()
    )

    def _one_step():
        optimizer.zero_grad(set_to_none=True)
        with autocast:
            loss = model.training_step(batch, 0)
        loss.backward()
        optimizer.step()
        return loss

    for _ in range(n_warmup):
        _one_step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    times = []
    for _ in range(n_timed):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = _one_step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    peak = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)

    times = sorted(times)
    return {
        "peak_mem_gib": peak / 1024**3,
        "reserved_mem_gib": reserved / 1024**3,
        "step_time_s": times[len(times) // 2],
        "loss": float(loss.detach().float().item()),
    }
