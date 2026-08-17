#!/usr/bin/env python3
"""Snapshot every native-240 validation result and render an LRN learning curve."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = Path(
    "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/dekel240"
)
OUT = ROOT / "outputs/dekel240_paper/training_monitor/native240_lrn"
HISTORY = OUT / "validation_history.csv"
STATUS = OUT / "run_status.json"
PLATEAU_PATIENCE_VALIDATIONS = 12
VALIDATION_EPOCH_STRIDE = 4

RUNS = {
    "M73 · GroupNorm, unregularized": (
        "D240M73c_dekel_native240_freqmasked_floor0p5_nostructuralreg_mlp_s201",
        "#555B61",
    ),
    "M74 · GroupNorm, half-smooth": (
        "D240M74c_dekel_native240_freqmasked_floor0p5_halfsmoothonly_mlp_s201",
        "#8A9095",
    ),
    "M75 · pure LRN": (
        "D240M75c_dekel_native240_freqmasked_floor0p5_lrn_s201",
        "#2D7E91",
    ),
    "M76 · GroupNorm→LRN": (
        "D240M76c_dekel_native240_freqmasked_floor0p5_historical_gnlrn_presplit_s201",
        "#DF7438",
    ),
    "M77 · GroupNorm→LRN α=.1": (
        "D240M77c_dekel_native240_freqmasked_floor0p5_gnlrnalpha0p1_presplit_s201",
        "#A64B73",
    ),
    "M78 · GroupNorm→LRN α=.03": (
        "D240M78c_dekel_native240_freqmasked_floor0p5_gnlrnalpha0p03_presplit_s201",
        "#3B7F6A",
    ),
}
PATTERN = re.compile(r"epoch=(\d+)-val_bps_overall=([-+]?\d*\.?\d+)\.ckpt$")
WANDB_ROOT = ROOT / "logs/wandb"


def scan_checkpoints() -> list[dict]:
    rows = []
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    for label, (directory, _) in RUNS.items():
        for path in (CHECKPOINT_ROOT / directory).glob("epoch=*-val_bps_overall=*.ckpt"):
            match = PATTERN.search(path.name)
            if match is None:
                continue
            rows.append(
                {
                    "observed_at": timestamp,
                    "run": label,
                    "epoch": int(match.group(1)),
                    "val_bps_overall": float(match.group(2)),
                    "checkpoint": str(path),
                    "source": "checkpoint",
                }
            )
    return rows


def scan_wandb_history() -> list[dict]:
    """Read validation points even when they did not enter checkpoint top-k."""
    try:
        from wandb.proto import wandb_internal_pb2 as wandb_pb
        from wandb.sdk.internal.datastore import DataStore
    except ImportError:
        return []

    directory_to_label = {directory: label for label, (directory, _) in RUNS.items()}
    rows = []
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    for run_dir in WANDB_ROOT.glob("offline-run-*"):
        metadata_path = run_dir / "files/wandb-metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
        args = metadata.get("args", [])
        try:
            experiment_name = args[args.index("--experiment_name") + 1]
        except (ValueError, IndexError):
            continue
        label = directory_to_label.get(experiment_name)
        if label is None:
            continue
        run_files = list(run_dir.glob("run-*.wandb"))
        if not run_files:
            continue

        store = DataStore()
        try:
            store.open_for_scan(str(run_files[0]))
        except (AssertionError, OSError):
            # W&B creates the run file before writing its LevelDB header while
            # a newly launched run is still preparing datasets.
            continue
        while True:
            try:
                data = store.scan_data()
            except Exception:
                # An active run can end in a record that is still being appended.
                break
            if data is None:
                break
            record = wandb_pb.Record()
            record.ParseFromString(data)
            if not record.HasField("history"):
                continue
            values = {
                "/".join(item.nested_key) if item.nested_key else item.key: item.value_json
                for item in record.history.item
            }
            if "epoch" not in values or "val_bps_overall" not in values:
                continue
            rows.append(
                {
                    "observed_at": timestamp,
                    "run": label,
                    "epoch": int(json.loads(values["epoch"])),
                    "val_bps_overall": float(json.loads(values["val_bps_overall"])),
                    "checkpoint": "",
                    "source": "wandb history",
                }
            )
    return rows


def scan() -> pd.DataFrame:
    return pd.DataFrame(scan_checkpoints() + scan_wandb_history())


def update_history(observed: pd.DataFrame) -> pd.DataFrame:
    if HISTORY.exists():
        history = pd.read_csv(HISTORY)
        if "source" not in history:
            history["source"] = "checkpoint"
        combined = pd.concat([history, observed], ignore_index=True)
    else:
        combined = observed.copy()
    # Logged history is more precise than the four-decimal checkpoint filename and
    # includes poor validation points that did not enter the retained top-k.
    combined["source_priority"] = combined.source.eq("wandb history").astype(int)
    combined = combined.sort_values(
        ["run", "epoch", "observed_at", "source_priority"]
    )
    combined = combined.drop_duplicates(["run", "epoch"], keep="last")
    combined = combined.drop(columns="source_priority")
    combined.to_csv(HISTORY, index=False)
    return combined


def summarize_active_runs(history: pd.DataFrame) -> dict:
    """Write an explicit, conservative plateau audit for active candidates."""
    payload = {
        "plateau_rule": (
            f"no new validation best for {PLATEAU_PATIENCE_VALIDATIONS} "
            "consecutive 4-epoch validation checks"
        ),
        "runs": {},
    }
    for label in (
        "M76 · GroupNorm→LRN",
        "M77 · GroupNorm→LRN α=.1",
        "M78 · GroupNorm→LRN α=.03",
    ):
        rows = history.loc[history.run.eq(label)].sort_values("epoch")
        if rows.empty:
            continue
        best_position = int(rows.val_bps_overall.to_numpy().argmax())
        best = rows.iloc[best_position]
        latest = rows.iloc[-1]
        observed_since_best = int(len(rows) - best_position - 1)
        # Checkpoint directories retain only top-k epochs.  When W&B history is
        # unavailable (currently M78), counting files badly understates how
        # many validation checks have elapsed.  The trainer validates every
        # four epochs, so the epoch gap is the conservative authoritative
        # count for early-stopping decisions.
        epoch_gap = int(latest.epoch - best.epoch)
        elapsed_checks = max(
            observed_since_best,
            epoch_gap // VALIDATION_EPOCH_STRIDE,
        )
        history_complete = bool(
            rows.loc[rows.epoch.ge(best.epoch), "source"].eq("wandb history").all()
        )
        payload["runs"][label] = {
            "latest_epoch": int(latest.epoch),
            "latest_val_bps": float(latest.val_bps_overall),
            "best_epoch": int(best.epoch),
            "best_val_bps": float(best.val_bps_overall),
            "validations_since_best": elapsed_checks,
            "observed_validation_rows_since_best": observed_since_best,
            "epochs_since_best": epoch_gap,
            "validation_history_complete": history_complete,
            "validation_count_basis": (
                "complete wandb history"
                if history_complete
                else f"epoch gap / {VALIDATION_EPOCH_STRIDE}; checkpoint top-k is incomplete"
            ),
            "plateau_candidate": bool(
                elapsed_checks >= PLATEAU_PATIENCE_VALIDATIONS
            ),
        }
    STATUS.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def render(history: pd.DataFrame) -> Path:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.65))
    display_history = history.loc[history.epoch.ge(31)].copy()
    for label, (_, color) in RUNS.items():
        selected = display_history.loc[display_history.run.eq(label)].sort_values("epoch")
        if selected.empty:
            continue
        if label.startswith("M73") or label.startswith("M74"):
            best = float(selected.val_bps_overall.max())
            ax.axhline(
                best,
                color=color,
                lw=1.35,
                ls=(0, (4, 2)),
                label=f"{label}: best {best:.4f}",
            )
            continue
        ax.plot(
            selected.epoch,
            selected.val_bps_overall,
            color=color,
            marker="o",
            ms=5,
            lw=2.1,
            label=label,
        )
        best = selected.loc[selected.val_bps_overall.idxmax()]
        ax.scatter(best.epoch, best.val_bps_overall, s=58, marker="*", color=color, zorder=4)
        ax.annotate(
            f"best {best.val_bps_overall:.3f}",
            (best.epoch, best.val_bps_overall),
            xytext=(5, 7),
            textcoords="offset points",
            ha="left",
            fontsize=8,
            color=color,
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation bits/spike")
    ax.set_title(
        "Native-240 normalization controls",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", color="#E1E5E8", lw=0.7)
    ax.set_xlim(left=28)
    ax.set_ylim(0.25, 0.61)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(
        0.985,
        0.015,
        "epochs <31 omitted for readability; stars mark best observed validation; checkpoint directory retains top 3",
        ha="right",
        fontsize=7.4,
        color="#687078",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = OUT / "native240_lrn_validation_curve.png"
    fig.savefig(path, dpi=190, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return path


def render_matched(history: pd.DataFrame) -> Path | None:
    labels = [
        "M76 · GroupNorm→LRN",
        "M77 · GroupNorm→LRN α=.1",
        "M78 · GroupNorm→LRN α=.03",
    ]
    selected = history.loc[history.run.isin(labels)].copy()
    m78 = selected.loc[selected.run.eq(labels[2])]
    if m78.empty:
        return None
    max_epoch = int(m78.epoch.max())
    selected = selected.loc[selected.epoch.le(max_epoch) & selected.epoch.ge(31)]
    colors = {label: RUNS[label][1] for label in labels}

    fig, (curve_axis, delta_axis) = plt.subplots(
        1, 2, figsize=(10.4, 4.4), gridspec_kw={"width_ratios": [1.65, 1]}
    )
    for label in labels:
        rows = selected.loc[selected.run.eq(label)].sort_values("epoch")
        curve_axis.plot(
            rows.epoch,
            rows.val_bps_overall,
            color=colors[label],
            marker="o",
            ms=5,
            lw=2,
            label=label,
        )
        latest = rows.iloc[-1]
        y_offset = {labels[0]: 8, labels[1]: -14, labels[2]: 20}[label]
        curve_axis.annotate(
            f"{latest.val_bps_overall:.3f}",
            (latest.epoch, latest.val_bps_overall),
            xytext=(5, y_offset),
            textcoords="offset points",
            ha="left",
            fontsize=8,
            color=colors[label],
        )

    matched = selected.pivot(index="epoch", columns="run", values="val_bps_overall")
    delta_axis.axhline(0, color="#6B737A", lw=0.9)
    for offset, comparison in zip((-1.15, 1.15), labels[1:]):
        common = matched[[labels[0], comparison]].dropna()
        delta = common[comparison] - common[labels[0]]
        delta_axis.bar(
            delta.index + offset,
            delta.values,
            width=2.1,
            color=colors[comparison],
            alpha=0.88,
            label=f"{comparison.split('·', 1)[0].strip()} − M76",
        )
    curve_axis.set(
        xlabel="epoch",
        ylabel="validation bits/spike",
        title="matched training trajectories",
    )
    delta_axis.set(
        xlabel="matched epoch",
        ylabel="difference from M76 (bits/spike)",
        title="normalization cost at matched epochs",
    )
    curve_axis.grid(axis="y", color="#E1E5E8", lw=0.7)
    delta_axis.grid(axis="y", color="#E1E5E8", lw=0.7)
    curve_axis.legend(frameon=False, fontsize=8, loc="lower right")
    delta_axis.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        "Local divisive competition preserves matched-epoch predictive performance",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    path = OUT / "native240_lrn_matched_epoch_curve.png"
    fig.savefig(path, dpi=190, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    observed = scan()
    if observed.empty:
        raise SystemExit("No matching checkpoints found")
    history = update_history(observed)
    status = summarize_active_runs(history)
    figure = render(history)
    matched_figure = render_matched(history)
    print(history.to_string(index=False))
    print(figure)
    if matched_figure is not None:
        print(matched_figure)
    print(STATUS)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
