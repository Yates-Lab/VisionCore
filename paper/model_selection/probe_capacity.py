"""Measure what each rung of the capacity ladder actually costs on this GPU.

Builds every width in the ladder, pulls a real training batch from the real
data module, and runs real optimizer steps. Reports parameter count, measured
FLOPs, peak memory and step time, with and without gradient checkpointing,
across a few batch sizes. Configurations that do not fit are recorded as OOM
rather than crashing the sweep.

The output table is what the training budget should be planned against. Do not
substitute predicted parameter counts or analytic FLOPs for it.

Usage
-----
    uv run python paper/model_selection/probe_capacity.py \
        --widths 0.25 0.5 1.0 2.0 3.0 4.0 \
        --batch-sizes 64 128 256 \
        --max-datasets 4
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

from VisionCore.paths import VISIONCORE_ROOT

HERE = Path(__file__).resolve().parent
if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))
if str(HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent.parent))

from paper.model_selection.gen_configs import (  # noqa: E402
    WIDTH_LADDER, write_ladder, width_name,
)
from paper.model_selection.measure import (  # noqa: E402
    count_params, measure_flops, measure_step,
)

DATASET_CONFIG = (
    VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"
)
OUT_PATH = HERE / "capacity_probe.json"


def _free(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_batch(batch_size: int, max_datasets: int, dset_dtype: str, device,
              max_scan: int = 64, homogeneous: bool = False):
    """Pull one real, full-size training batch from the real data module.

    The loader groups samples by dataset, so an arbitrary batch can be a
    short tail batch. Memory measured on a short batch would understate the
    real cost, so scan until a batch of the requested size appears and fall
    back to the largest one seen.
    """
    from training.pl_modules import MultiDatasetDM

    dm = MultiDatasetDM(
        cfg_dir=str(DATASET_CONFIG),
        max_ds=max_datasets,
        batch=batch_size,
        workers=0,
        steps_per_epoch=max_scan,
        enable_curriculum=False,
        dset_dtype=dset_dtype,
        homogeneous_batches=homogeneous,
    )
    dm.setup("fit")

    def _total(b):
        return sum(d["robs"].shape[0] for d in ([b] if isinstance(b, dict) else b))

    best = None
    for i, batch in enumerate(dm.train_dataloader()):
        if best is None or _total(batch) > _total(best):
            best = batch
        if _total(batch) == batch_size:
            break
        if i >= max_scan:
            break

    got = _total(best)
    if got != batch_size:
        print(f"    WARNING: requested batch {batch_size}, largest found {got}")

    def _to_device(d):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()}

    batch = _to_device(best) if isinstance(best, dict) else [_to_device(d) for d in best]
    return batch, dm


def build_model(width: float, max_datasets: int, checkpointing: bool, device,
                with_adapter: bool = True):
    from training.pl_modules import MultiDatasetModel

    # The adapter blurs and grid-samples the raw stimulus, which is the largest
    # tensor in the network, so whether it is present changes peak memory by
    # more than its 90 parameters suggest. Probe the architecture that will
    # actually be trained.
    config_path = write_ladder(widths=[width], with_adapter=with_adapter)[width]
    model = MultiDatasetModel(
        model_cfg=str(config_path),
        cfg_dir=str(DATASET_CONFIG),
        lr=1e-3,
        wd=1e-5,
        max_ds=max_datasets,
        compile_model=False,
    )
    model.model.convnet.use_checkpointing = checkpointing
    model = model.to(device)
    model.train()
    return model


def probe_one(width, batch_size, checkpointing, batch, max_datasets, device,
              with_adapter=True):
    """Measure one (width, batch_size, checkpointing) cell. Returns a dict."""
    row = {
        "width": width,
        "batch_size": batch_size,
        "gradient_checkpointing": checkpointing,
    }
    model = None
    try:
        model = build_model(width, max_datasets, checkpointing, device,
                            with_adapter=with_adapter)
        row.update(count_params(model))
        row.update(measure_step(model, batch, device))
        try:
            row.update(measure_flops(model, batch))
        except Exception as exc:  # FLOP counter is the optional part
            row["flops_error"] = f"{type(exc).__name__}: {exc}"
        row["status"] = "ok"
    except torch.cuda.OutOfMemoryError as exc:
        row["status"] = "oom"
        row["error"] = str(exc).split("\n")[0]
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _free(model)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--widths", type=float, nargs="+", default=list(WIDTH_LADDER))
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--max-datasets", type=int, default=4,
                   help="Sessions to load. Affects readout params only (small); "
                        "keep low to make the probe fast.")
    p.add_argument("--dset-dtype", type=str, default="bfloat16")
    p.add_argument("--checkpointing", type=str, nargs="+",
                   default=["off", "on"], choices=["off", "on"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--homogeneous-batches", action="store_true",
                   help="One session per step (single large forward) instead of "
                        "~30 sequential sub-forwards. Changes training dynamics, "
                        "not just throughput.")
    p.add_argument("--no-adapter", action="store_true",
                   help="Probe the identity-adapter architecture")
    p.add_argument("--out", type=Path, default=OUT_PATH)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("medium")

    props = torch.cuda.get_device_properties(device)
    print(f"Device: {props.name}  ({props.total_memory / 1024**3:.1f} GiB)")
    print(f"Dataset config: {DATASET_CONFIG.name}  ({args.max_datasets} sessions)\n")

    rows = []
    for batch_size in args.batch_sizes:
        print(f"=== batch_size={batch_size}: loading real batch ===")
        batch, dm = get_batch(batch_size, args.max_datasets, args.dset_dtype, device,
                              homogeneous=args.homogeneous_batches)
        sub = [batch] if isinstance(batch, dict) else batch
        shapes = {
            "n_sub_batches": len(sub),
            "total_samples": sum(d["robs"].shape[0] for d in sub),
            "sub_batch_sizes": [d["robs"].shape[0] for d in sub],
            "stim": list(sub[0]["stim"].shape[1:]),
        }
        print(f"    {shapes['n_sub_batches']} sub-batches, "
              f"{shapes['total_samples']} total samples, "
              f"stim {shapes['stim']}\n")

        for ckpt_flag in args.checkpointing:
            checkpointing = ckpt_flag == "on"
            for width in args.widths:
                row = probe_one(width, batch_size, checkpointing, batch,
                                args.max_datasets, device,
                                with_adapter=not args.no_adapter)
                row["batch_shapes"] = shapes
                row["homogeneous_batches"] = args.homogeneous_batches
                row["with_adapter"] = not args.no_adapter
                rows.append(row)

                tag = f"w={width:<4g} bs={batch_size:<4d} ckpt={ckpt_flag:<3s}"
                if row["status"] == "ok":
                    print(f"  {tag}  {row['params_total']/1e6:7.2f}M params  "
                          f"{row['peak_mem_gib']:6.2f} GiB  "
                          f"{row['step_time_s']*1000:8.1f} ms/step")
                else:
                    print(f"  {tag}  {row['status'].upper()}: "
                          f"{row.get('error', '')[:70]}")

                with open(args.out, "w") as fh:
                    json.dump(rows, fh, indent=2)

        _free(batch, dm)
        print()

    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
