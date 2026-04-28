import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from DataYatesV1 import get_complete_sessions


FINAL_RUNS_ROOT = Path(
    os.getenv("FINAL_RUNS_ROOT", "/home/tejas/VisionCore/tejas/model/final_runs")
)
SESSION_FILTER = os.getenv("SESSION_FILTER", "Allen_")
CLEAN_SESSION_DIRS = str(os.getenv("CLEAN_SESSION_DIRS", "0")).strip().lower() in {"1", "true", "yes"}


def discover_allen_sessions():
    session_names_env = os.getenv("SESSION_NAMES")
    if session_names_env:
        return [s.strip() for s in session_names_env.split(",") if s.strip()]
    sessions = sorted(
        sess.name
        for sess in get_complete_sessions()
        if sess.name.startswith(SESSION_FILTER)
    )
    return sessions


def append_csv_row(path: Path, fieldnames, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_session(session_name: str):
    subject, date = session_name.split("_", 1)
    session_dir = FINAL_RUNS_ROOT / session_name
    if CLEAN_SESSION_DIRS and session_dir.exists():
        import shutil

        shutil.rmtree(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    run_log = session_dir / "batch_run.log"

    env = os.environ.copy()
    env.update(
        {
            "SUBJECT": subject,
            "DATE": date,
            "SESSION_NAME": session_name,
            "FINAL_RUNS_ROOT": str(FINAL_RUNS_ROOT),
            "CELL_IDS": os.getenv("CELL_IDS", "all"),
            "INIT_STATE_REFERENCE_CELL_IDS": os.getenv("INIT_STATE_REFERENCE_CELL_IDS", "all"),
            "RAY_GPU_FRACTION": os.getenv("RAY_GPU_FRACTION", "0.125"),
            "RAY_TASK_CPUS": os.getenv("RAY_TASK_CPUS", "4"),
            "TASK_NUM_WORKERS": os.getenv("TASK_NUM_WORKERS", "4"),
            "SAVE_BEST_PNGS": os.getenv("SAVE_BEST_PNGS", "1"),
            "SAVE_PNG_MODE": os.getenv("SAVE_PNG_MODE", "best"),
            "SKIP_TRAIN_EVAL": os.getenv("SKIP_TRAIN_EVAL", "0"),
            "PYR_LEVELS": os.getenv("PYR_LEVELS", "3"),
        }
    )

    cmd = [
        "/home/tejas/VisionCore/.venv/bin/python",
        "tejas/model/two_stage_lbfgs_exp_singlecell_ray.py",
    ]

    start_s = time.perf_counter()
    with run_log.open("w") as f:
        f.write(f"session_name={session_name}\n")
        f.write(f"subject={subject}\n")
        f.write(f"date={date}\n")
        f.write(f"command={' '.join(cmd)}\n")
        f.write(f"env={json.dumps({k: env[k] for k in ['SUBJECT','DATE','SESSION_NAME','FINAL_RUNS_ROOT','CELL_IDS','INIT_STATE_REFERENCE_CELL_IDS','RAY_GPU_FRACTION','RAY_TASK_CPUS','TASK_NUM_WORKERS','SAVE_BEST_PNGS','SAVE_PNG_MODE','SKIP_TRAIN_EVAL','PYR_LEVELS']}, sort_keys=True)}\n\n")
        f.flush()
        proc = subprocess.run(
            cmd,
            cwd="/home/tejas/VisionCore",
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed_s = time.perf_counter() - start_s
    return proc.returncode, elapsed_s, session_dir


def main():
    sessions = discover_allen_sessions()
    print(f"discovered {len(sessions)} Allen sessions")
    for s in sessions:
        print(s)

    summary_csv = FINAL_RUNS_ROOT / "allen_batch_summary.csv"
    failures_csv = FINAL_RUNS_ROOT / "allen_batch_failures.csv"

    for idx, session_name in enumerate(sessions, start=1):
        print(f"\n[{idx}/{len(sessions)}] starting {session_name}")
        try:
            returncode, elapsed_s, session_dir = run_session(session_name)
            row = {
                "session_name": session_name,
                "status": "ok" if returncode == 0 else "failed",
                "returncode": int(returncode),
                "elapsed_s": float(elapsed_s),
                "session_dir": str(session_dir),
            }
            append_csv_row(
                summary_csv,
                ["session_name", "status", "returncode", "elapsed_s", "session_dir"],
                row,
            )
            if returncode != 0:
                append_csv_row(
                    failures_csv,
                    ["session_name", "returncode", "elapsed_s", "session_dir"],
                    {
                        "session_name": session_name,
                        "returncode": int(returncode),
                        "elapsed_s": float(elapsed_s),
                        "session_dir": str(session_dir),
                    },
                )
                print(f"[failed] {session_name} returncode={returncode}")
            else:
                print(f"[ok] {session_name} elapsed_s={elapsed_s:.2f}")
        except Exception as e:
            append_csv_row(
                summary_csv,
                ["session_name", "status", "returncode", "elapsed_s", "session_dir"],
                {
                    "session_name": session_name,
                    "status": "exception",
                    "returncode": -1,
                    "elapsed_s": 0.0,
                    "session_dir": str(FINAL_RUNS_ROOT / session_name),
                },
            )
            append_csv_row(
                failures_csv,
                ["session_name", "returncode", "elapsed_s", "session_dir"],
                {
                    "session_name": session_name,
                    "returncode": -1,
                    "elapsed_s": 0.0,
                    "session_dir": str(FINAL_RUNS_ROOT / session_name),
                },
            )
            print(f"[exception] {session_name}: {e}")

    print("\nall sessions attempted")
    print(f"summary_csv={summary_csv}")
    print(f"failures_csv={failures_csv}")


if __name__ == "__main__":
    main()
