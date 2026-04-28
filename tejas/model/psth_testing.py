#%%
from tejas.rsvp_util import get_fixrsvp_data
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml


SUBJECT = "Allen"
DATE = "2022-04-13"
CELL_NUMBER = 16
DURATION_S = 0.5
DATASET_CONFIGS_PATH = "/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_60_rsvp.yaml"
OUT_PATH = (
    f"/home/tejas/VisionCore/tejas/model/"
    f"psth_testing_{Path(DATASET_CONFIGS_PATH).stem}_{SUBJECT}_{DATE}_cell_{CELL_NUMBER:03d}.png"
)


def get_target_rate_hz():
    with open(DATASET_CONFIGS_PATH, "r") as f:
        return float(yaml.safe_load(f).get("sampling", {}).get("target_rate", 240))


def plot_fixrsvp_psth_and_spike_raster(data, cell_number, target_rate_hz):
    robs = np.transpose(data["robs"], (2, 0, 1))
    t_trial = np.arange(robs.shape[-1], dtype=np.float32) / float(target_rate_hz)
    valid_t = t_trial <= DURATION_S
    robs = robs[:, :, valid_t]
    t_trial = t_trial[valid_t]
    psth = np.nanmean(robs, axis=1)
    psth_ste = np.nanstd(robs, axis=1) / np.sqrt(np.sum(np.isfinite(robs), axis=1))

    fig, ax = plt.subplots(figsize=(6, 4))
    for i_trial in range(robs.shape[1]):
        spikes = np.nan_to_num(robs[cell_number, i_trial], nan=0.0) > 0
        spike_times = t_trial[spikes]
        if len(spike_times) > 0:
            ax.eventplot(spike_times, lineoffsets=i_trial, linelengths=0.8, color="k", alpha=0.7)
    ax.set_ylabel("Trial")
    ax.set_xlabel("Time (s)")

    ax2 = ax.twinx()
    ax2.plot(t_trial, psth[cell_number], color="tab:blue")
    ax2.fill_between(
        t_trial,
        psth[cell_number] - psth_ste[cell_number],
        psth[cell_number] + psth_ste[cell_number],
        color="tab:blue",
        alpha=0.3,
    )
    ax2.set_ylabel("Firing Rate (Hz)")
    ax.set_title(f"FixRSVP PSTH and Spike Raster - Unit {data['cids'][cell_number]}")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print("saved", OUT_PATH)


def main():
    target_rate_hz = get_target_rate_hz()
    data = get_fixrsvp_data(
        SUBJECT,
        DATE,
        DATASET_CONFIGS_PATH,
        use_cached_data=True,
        verbose=False,
    )
    print("robs", data["robs"].shape)
    print("dfs", data["dfs"].shape)
    print("eyepos", data["eyepos"].shape)
    print("stim", data["stim"].shape)
    plot_fixrsvp_psth_and_spike_raster(data, CELL_NUMBER, target_rate_hz)


if __name__ == "__main__":
    main()

# %%
