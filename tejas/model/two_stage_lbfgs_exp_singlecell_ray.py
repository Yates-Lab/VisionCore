import json
import os
import shutil
import subprocess
import sys
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import torch
from models.losses import PoissonBPSAggregator
from torch.utils.data import DataLoader

from two_stage_core import TwoStage
from two_stage_trainer import eval_step
from util import get_dataset_info


DATASET_CONFIGS_PATH = os.getenv(
    "DATASET_CONFIGS_PATH",
    "/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_gaborium_20lags.yaml",
)
SUBJECT = os.getenv("SUBJECT", "Allen")
DATE = os.getenv("DATE", "2022-04-13")
IMAGE_SHAPE = tuple(
    int(x.strip()) for x in os.getenv("IMAGE_SHAPE", "41,41").split(",") if x.strip()
)
PYR_LEVELS = int(os.getenv("PYR_LEVELS", "3"))
DEFAULT_OUTPUT_ROOT = (
    "/home/tejas/VisionCore/tejas/model/final_runs"
)
DEFAULT_SESSION_NAME = f"{SUBJECT}_{DATE}"


def parse_cell_ids(raw, n_total_cells):
    s = str(raw).strip().lower()
    if s in {"all", "*"}:
        return list(range(int(n_total_cells)))
    ids = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not ids:
        raise ValueError("CELL_IDS must be 'all' or comma-separated indices.")
    for cid in ids:
        if cid < 0 or cid >= n_total_cells:
            raise ValueError(f"cell id {cid} out of range [0, {n_total_cells - 1}]")
    return sorted(set(ids))


def resolve_cell_ids(raw):
    info = get_dataset_info(DATASET_CONFIGS_PATH, SUBJECT, DATE, IMAGE_SHAPE)
    return parse_cell_ids(raw=raw, n_total_cells=int(info["robs"].shape[1]))


def crop_slice(crop_size):
    return slice(crop_size, -crop_size) if int(crop_size) > 0 else slice(None)


def prepare_batch_one_lag(batch, lag_index, crop_size, device="cuda"):
    b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    ys = crop_slice(crop_size)
    xs = crop_slice(crop_size)
    b["stim"] = b["stim"][:, :, [int(lag_index)], ys, xs]
    return b


def build_task_env():
    env = {
        "TRAINING_PROTOCOL": "single_cell",
        "NUM_EPOCHS": os.getenv("NUM_EPOCHS", "1"),
        "BATCH_SIZE": os.getenv("BATCH_SIZE", "2048"),
        "NUM_WORKERS": os.getenv("TASK_NUM_WORKERS", "4"),
        "PERSISTENT_WORKERS": os.getenv("PERSISTENT_WORKERS", "1"),
        "PIN_MEMORY": os.getenv("PIN_MEMORY", "1"),
        "PREFETCH_FACTOR": os.getenv("PREFETCH_FACTOR", "2"),
        "PRECOMPUTE_PYRAMID": os.getenv("PRECOMPUTE_PYRAMID", "1"),
        "PRECOMPUTE_DENOMINATOR": os.getenv("PRECOMPUTE_DENOMINATOR", "1"),
        "CACHE_DTYPE": os.getenv("CACHE_DTYPE", "float32"),
        "SAVE_BEST_PNGS": os.getenv("SAVE_BEST_PNGS", "1"),
        "SAVE_PNG_MODE": os.getenv("SAVE_PNG_MODE", "best"),
        "SKIP_TRAIN_EVAL": os.getenv("SKIP_TRAIN_EVAL", "0"),
        "OUTPUT_NONLINEARITY": os.getenv("OUTPUT_NONLINEARITY", "exp"),
        "SPARSITY_MODE": os.getenv("SPARSITY_MODE", "prox_l1"),
        "LOCALITY_MODE": os.getenv("LOCALITY_MODE", "weighted_l21"),
        "FIXED_LAMBDA_PROX": os.getenv("FIXED_LAMBDA_PROX", "1e-1"),
        "FIXED_LAMBDA_LOCAL": os.getenv("FIXED_LAMBDA_LOCAL", "0"),
        "PROX_MULTS": os.getenv("PROX_MULTS", "1.0"),
        "LOCAL_MULTS": os.getenv("LOCAL_MULTS", "1.0"),
        "SEED": os.getenv("SEED", "0"),
        "USE_GROUP_SEED": os.getenv("USE_GROUP_SEED", "0"),
        "TIME_BREAKDOWN": os.getenv("TIME_BREAKDOWN", "1"),
        "LBFGS_MAX_ITER": os.getenv("LBFGS_MAX_ITER", "10"),
        "LBFGS_HISTORY_SIZE": os.getenv("LBFGS_HISTORY_SIZE", "10"),
        "LBFGS_LR": os.getenv("LBFGS_LR", "0.1"),
        "PYR_LEVELS": os.getenv("PYR_LEVELS", str(PYR_LEVELS)),
    }
    return env


def copy_cell_checkpoint(cell_dir, checkpoints_dir, cell_id):
    src = cell_dir / "checkpoints" / f"cell_{int(cell_id):03d}_best.pt"
    if src.exists():
        shutil.copy2(src, checkpoints_dir / src.name)


def verify_checkpoint_bps(session_dir, results_df, batch_size):
    info = get_dataset_info(DATASET_CONFIGS_PATH, SUBJECT, DATE, IMAGE_SHAPE)
    val_dset = info["val_dset"]
    crop_size = info["crop_size"]
    val_dset.to("cpu")
    val_loader = DataLoader(
        val_dset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    verification_rows = []
    checkpoints_dir = Path(session_dir) / "checkpoints"
    for row in results_df.itertuples(index=False):
        ckpt_path = checkpoints_dir / f"cell_{int(row.cell_id):03d}_best.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = TwoStage(**ckpt["model_kwargs"])
        model.load_state_dict(ckpt["model_state"])
        model.cuda()
        model.eval()

        val_agg = PoissonBPSAggregator()
        lag_index = int(ckpt["lag_index"])
        group_cell_ids = [int(x) for x in ckpt["group_cell_ids"]]
        for batch in val_loader:
            batch = prepare_batch_one_lag(
                batch=batch,
                lag_index=lag_index,
                crop_size=crop_size,
                device="cuda",
            )
            out = eval_step(model=model, batch=batch, cell_ids=group_cell_ids, use_resolver=True)
            val_agg(out)
        reloaded_val_bps = float(val_agg.closure().detach().cpu().numpy().reshape(-1)[int(ckpt["neuron_idx"])])
        recorded_val_bps = float(row.best_val_bps)
        verification_rows.append(
            {
                "cell_id": int(row.cell_id),
                "recorded_val_bps": recorded_val_bps,
                "reloaded_val_bps": reloaded_val_bps,
                "abs_diff": abs(reloaded_val_bps - recorded_val_bps),
            }
        )
        del model
        torch.cuda.empty_cache()

    verification_df = pd.DataFrame(verification_rows).sort_values("cell_id").reset_index(drop=True)
    verification_df.to_csv(Path(session_dir) / "checkpoint_verification.csv", index=False)
    return verification_df


def compute_init_state_files(cell_ids, output_root, seed):
    info = get_dataset_info(DATASET_CONFIGS_PATH, SUBJECT, DATE, IMAGE_SHAPE)
    train_dset = info["train_dset"]
    peak_lags = np.asarray(info["peak_lags"]).reshape(-1)
    state_root = Path(output_root) / "_init_states"
    state_root.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    state_paths = {}
    for cid in cell_ids:
        torch_path = state_root / f"cell_{int(cid):03d}_torch.pt"
        numpy_path = state_root / f"cell_{int(cid):03d}_numpy.pkl"
        torch.save(torch.get_rng_state(), torch_path)
        with numpy_path.open("wb") as f:
            pickle.dump(np.random.get_state(), f)

        model = TwoStage(
            image_shape=IMAGE_SHAPE,
            n_neurons=1,
            n_lags=1,
            height=PYR_LEVELS,
            order=5,
            lowest_cpd_target=1.0,
            ppd=train_dset.dsets[0].metadata["ppd"],
            rel_tolerance=0.3,
            validate_cpd=True,
            beta_init=0.0,
            init_weight_scale=1e-4,
            beta_as_parameter=True,
            clamp_beta_min=None,
            hann_window_power=2,
            output_nonlinearity="exp",
        )
        del model
        state_paths[int(cid)] = {
            "torch": str(torch_path),
            "numpy": str(numpy_path),
        }
    return state_paths


def copy_cell_pngs(cell_dir, output_root, cell_id):
    for stale in output_root.glob(f"cell_{int(cell_id):03d}_*_diagnostics.png"):
        stale.unlink(missing_ok=True)
    for src in sorted(cell_dir.glob(f"cell_{int(cell_id):03d}_*_diagnostics.png")):
        shutil.copy2(src, output_root / src.name)


def run_cell_job(cell_id, repo_root, output_root, task_env, init_state_paths):
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    import pandas as pd
    import ray

    cell_dir = Path(output_root) / f"cell_{int(cell_id):03d}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    run_log = cell_dir / "run.log"
    start_s = time.perf_counter()

    env = os.environ.copy()
    env.update(task_env)
    env["CELL_IDS"] = str(int(cell_id))
    env["OUTPUT_DIR"] = str(cell_dir)
    env["SAVE_BEST_CHECKPOINTS"] = "1"
    env["CHECKPOINT_DIR"] = str(cell_dir / "checkpoints")
    env.pop("LBFGS_MAX_EVAL", None)
    env["INIT_TORCH_RNG_STATE_PATH"] = str(init_state_paths["torch"])
    env["INIT_NUMPY_RNG_STATE_PATH"] = str(init_state_paths["numpy"])

    accelerator_ids = ray.get_runtime_context().get_accelerator_ids()
    gpu_ids = accelerator_ids.get("GPU", [])
    env["RAY_ASSIGNED_GPU_IDS"] = ",".join(str(x) for x in gpu_ids)

    cmd = [sys.executable, "tejas/model/two_stage_lbfgs_exp_multicell_gridsearch.py"]
    with run_log.open("w") as f:
        f.write(f"cell_id={int(cell_id)}\n")
        f.write(f"ray_gpu_ids={gpu_ids}\n")
        f.write(f"command={' '.join(cmd)}\n")
        f.write(f"task_env={json.dumps(task_env, sort_keys=True)}\n\n")
        f.flush()
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    elapsed_s = time.perf_counter() - start_s
    if proc.returncode != 0:
        raise RuntimeError(
            f"Cell {cell_id} failed with exit code {proc.returncode}. "
            f"See {run_log}."
        )

    best_csv = cell_dir / "best_val_bps.csv"
    if not best_csv.exists():
        raise FileNotFoundError(f"Missing expected output: {best_csv}")
    df = pd.read_csv(best_csv)
    if len(df) != 1:
        raise ValueError(f"Expected exactly one row in {best_csv}, found {len(df)}")
    row = df.iloc[0].to_dict()
    row["cell_id"] = int(row["cell_id"])
    row["elapsed_s"] = float(elapsed_s)
    row["gpu_ids"] = ",".join(str(x) for x in gpu_ids)
    row["output_dir"] = str(cell_dir)
    return row


def main():
    repo_root = Path(__file__).resolve().parents[2]
    final_runs_root = Path(os.getenv("FINAL_RUNS_ROOT", DEFAULT_OUTPUT_ROOT))
    session_name = os.getenv("SESSION_NAME", DEFAULT_SESSION_NAME)
    session_dir = final_runs_root / session_name
    pngs_dir = session_dir / "pngs"
    checkpoints_dir = session_dir / "checkpoints"
    cells_dir = session_dir / "cells"
    session_dir.mkdir(parents=True, exist_ok=True)
    pngs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    cells_dir.mkdir(parents=True, exist_ok=True)

    cell_ids = resolve_cell_ids(os.getenv("CELL_IDS", "all"))
    init_state_reference_ids = resolve_cell_ids(os.getenv("INIT_STATE_REFERENCE_CELL_IDS", os.getenv("CELL_IDS", "all")))
    gpu_fraction = float(os.getenv("RAY_GPU_FRACTION", "0.125"))
    task_cpus = float(os.getenv("RAY_TASK_CPUS", "4"))
    task_env = build_task_env()
    state_paths_by_cell = compute_init_state_files(
        cell_ids=init_state_reference_ids,
        output_root=session_dir,
        seed=int(task_env["SEED"]),
    )

    print(f"session_dir={session_dir}")
    print(f"n_cells={len(cell_ids)}")
    print(f"ray_gpu_fraction={gpu_fraction}")
    print(f"ray_task_cpus={task_cpus}")
    print(f"task_env={json.dumps(task_env, sort_keys=True)}")

    ray.init(
        ignore_reinit_error=True,
        log_to_driver=True,
        include_dashboard=False,
        runtime_env={
            "py_executable": sys.executable,
            "excludes": [
                ".git/**",
                ".venv/**",
                "tejas/model/final_pngs*/**",
                "tejas/model/fast_final_pngs*/**",
                "tejas/model/ray_sweep*/**",
                "tejas/model/lambda_prox_sweep*/**",
                "tejas/model/final_runs/**",
                "tejas/model/single_cell_protocol_allcells_lambda_prox_1e-1_local0*/**",
                "tejas/model/grouped_allcells_lambda_prox_1e-1_local0*/**",
                "tejas/model/_debug*/**",
                "tejas/model/_bench*/**",
                "tejas/model/_ray_subset_parity/**",
                "tejas/model/ray_single_cell_protocol_allcells_lambda_prox_1e-1_local0_closer_to_file/**",
            ],
        },
    )

    worker = ray.remote(num_gpus=gpu_fraction, num_cpus=task_cpus)(run_cell_job)

    t0 = time.perf_counter()
    pending = [
        worker.remote(
            cell_id=int(cid),
            repo_root=str(repo_root),
            output_root=str(cells_dir),
            task_env=task_env,
            init_state_paths=state_paths_by_cell[int(cid)],
        )
        for cid in cell_ids
    ]

    results = []
    total = len(pending)
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        result = ray.get(done[0])
        results.append(result)
        print(
            f"[done {len(results):03d}/{total:03d}] "
            f"cell={int(result['cell_id']):03d} "
            f"val_bps={float(result['best_val_bps']):.4f} "
            f"elapsed_s={float(result['elapsed_s']):.2f} "
            f"gpu_ids={result['gpu_ids']}"
        )

    total_elapsed_s = time.perf_counter() - t0
    results_df = pd.DataFrame(results).sort_values("cell_id").reset_index(drop=True)
    results_df.to_csv(session_dir / "ray_cell_results.csv", index=False)

    best_df = results_df[
        [
            "cell_id",
            "best_val_bps",
            "train_bps_at_best",
            "best_lag_group",
            "best_epoch",
            "best_prox_mult",
            "best_local_mult",
        ]
    ].copy()
    best_df.to_csv(session_dir / "best_val_bps.csv", index=False)

    for row in results_df.itertuples(index=False):
        copy_cell_pngs(
            cell_dir=Path(row.output_dir),
            output_root=pngs_dir,
            cell_id=int(row.cell_id),
        )
        copy_cell_checkpoint(
            cell_dir=Path(row.output_dir),
            checkpoints_dir=checkpoints_dir,
            cell_id=int(row.cell_id),
        )

    verification_df = verify_checkpoint_bps(
        session_dir=session_dir,
        results_df=results_df,
        batch_size=int(task_env["BATCH_SIZE"]),
    )

    summary = {
        "n_cells": int(len(results_df)),
        "total_elapsed_s": float(total_elapsed_s),
        "mean_elapsed_s_per_cell": float(results_df["elapsed_s"].mean()),
        "median_elapsed_s_per_cell": float(results_df["elapsed_s"].median()),
        "mean_best_val_bps": float(results_df["best_val_bps"].mean()),
        "median_best_val_bps": float(results_df["best_val_bps"].median()),
        "max_checkpoint_reload_abs_diff": float(verification_df["abs_diff"].max()),
        "ray_gpu_fraction": float(gpu_fraction),
        "ray_task_cpus": float(task_cpus),
        "session_dir": str(session_dir),
        "task_env": task_env,
    }
    with (session_dir / "ray_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    ray.shutdown()


if __name__ == "__main__":
    main()
