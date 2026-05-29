"""Interactive scratch script for inspecting the twininfo analysis.

Open this file in VS Code, Spyder, or another editor that understands ``#%%``
cells.  The cells intentionally use the same production helpers as
``jake.twininfo.pipeline`` so intermediate plots match the real analysis.
"""
from __future__ import annotations

#%% Path bootstrap
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    """Find the VisionCore checkout so this script works from any CWD."""
    candidates: list[Path] = []
    if "__file__" in globals():
        candidates.extend(Path(__file__).resolve().parents)
    candidates.extend(Path.cwd().resolve().parents)
    candidates.insert(0, Path.cwd().resolve())
    for candidate in candidates:
        if (candidate / "VisionCore" / "paths.py").exists() and (candidate / "jake" / "twininfo").exists():
            return candidate
    raise RuntimeError(
        "Could not find the VisionCore repo root. Add /home/jake/repos/VisionCore to sys.path manually."
    )


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print(f"repo root: {REPO_ROOT}")


#%% Imports and configuration
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import numpy as np

from jake.twininfo.activation_movies import condition_lag_cubes
from jake.twininfo.common import (
    DEFAULT_CCMAX_THRESHOLD,
    DT,
    N_LAGS,
    OUT_SIZE,
    OUTPUT_DIR,
    load_digital_twin,
)
from jake.twininfo.image_selection import SF_BANDS_4, crop_rows, select_image_crops
from jake.twininfo.lagcube_information import (
    block_current_samples,
    block_endpoint_lag_cubes,
    cumulative_pattern_fisher,
    cumulative_spatial_ssi,
    finite_difference_shift_set,
    run_lag_cube_rates,
    run_shifted_lag_cube_rate_maps,
    unique_shifts,
)
from jake.twininfo.retinal_examples import model_crop_centers_px
from jake.twininfo.retinal_movies import (
    CONDITION_LABELS,
    save_stimulus_comparison_mp4,
    stimulus_condition_movies,
)
from jake.twininfo.population import build_analysis_population
from jake.twininfo.trace_selection import run_trace_selection_step

try:
    from IPython.display import HTML, display
except Exception:  # pragma: no cover - only used in notebooks.
    HTML = None
    display = None


SEED = 0
IMAGE_INDICES = (24, 29, 30)
N_CROPS_PER_IMAGE = 1
N_EXAMPLES_PER_KIND = 2
T_MAX = 128
TRACE_STRIDE = 8
POPULATION_SIZE = 16
POPULATION_SELECTION = "top_performance"
PERFORMANCE_METRIC = "ccnorm"
MIN_PERFORMANCE_SCORE = None  # Example: 0.5 keeps only units with ccnorm >= 0.5.
POPULATION_GRID_POSITION_MODE = "full_grid"
POPULATION_GRID_STRIDE = 1
DEDUPLICATE_UNITS = True
DEDUPE_CORRELATION_THRESHOLD = 0.95
BATCH_SIZE = 64

EXAMPLE_KIND = "microsaccade"  # "fixation" or "microsaccade"
EXAMPLE_RANK_WITHIN_KIND = 0
IMAGE_INDEX = 24
CROP_RANK = 0

SAVE_MP4S = False
SHOW_JS_ANIMATION = False
RUN_INFORMATION_CELL = False

INTERACTIVE_DIR = OUTPUT_DIR / "interactive-script"
FIGURE_DIR = INTERACTIVE_DIR / "figures"
METADATA_DIR = INTERACTIVE_DIR / "metadata"
MOVIE_DIR = INTERACTIVE_DIR / "movies"
for path in (FIGURE_DIR, METADATA_DIR, MOVIE_DIR):
    path.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)


#%% Load the digital twin and pick a small top-performing unit population
model, model_info, device = load_digital_twin()
population, population_rows = build_analysis_population(
    model,
    N=POPULATION_SIZE,
    rng=rng,
    selection=POPULATION_SELECTION,
    performance_metric=PERFORMANCE_METRIC,
    min_performance_score=MIN_PERFORMANCE_SCORE,
    ccmax_threshold=DEFAULT_CCMAX_THRESHOLD,
    grid_position_mode=POPULATION_GRID_POSITION_MODE,
    grid_stride=POPULATION_GRID_STRIDE,
    deduplicate_units=DEDUPLICATE_UNITS,
    dedupe_correlation_threshold=DEDUPE_CORRELATION_THRESHOLD,
    dedupe_battery_seed=SEED,
    dedupe_batch_size=BATCH_SIZE,
)

print(f"device: {device}")
print(f"biological twins: {len(set(int(row['global_unit_idx']) for row in population_rows))}")
print(f"simulated neurons in full rate map: {len(population.unit_ids)}")
print(f"interactive outputs: {INTERACTIVE_DIR}")
print("top selected units:")
for row in population_rows[: min(10, len(population_rows))]:
    print({
        key: row[key]
        for key in (
            "global_unit_idx",
            "session_name",
            "original_neuron_id",
            "performance_score",
            "ccnorm",
            "ccmax",
            "grid_row",
            "grid_col",
        )
    })


#%% Select real eye traces and write the trace-selection QC figures
examples = run_trace_selection_step(
    figure_dir=FIGURE_DIR,
    metadata_dir=METADATA_DIR,
    seed=SEED,
    n_examples_per_kind=N_EXAMPLES_PER_KIND,
    t_max=T_MAX,
    stride=TRACE_STRIDE,
    model=model,
)

for example in examples:
    print(
        example.example_id,
        example.kind,
        "events=",
        example.n_events_in_window,
        "event_onset=",
        example.event_onset,
    )


#%% Select natural images and middle-band-energy crop hotspots
image_by_index, selected_crops, image_qc_paths = select_image_crops(
    image_indices=IMAGE_INDICES,
    n_crops_per_image=N_CROPS_PER_IMAGE,
    figure_dir=FIGURE_DIR,
    metadata_path=METADATA_DIR / "02_image_crop_hotspots.csv",
    seed=SEED,
    t_max=T_MAX,
    trace_arrays=[example.trace for example in examples],
)
crop_table = crop_rows(selected_crops)

print("image QC figures:")
for path in image_qc_paths:
    print(" ", path)
print("selected crops:")
for row in crop_table:
    print(row)


#%% Choose one image/crop/eye-trace example for hands-on plotting
kind_examples = [example for example in examples if example.kind == EXAMPLE_KIND]
example = kind_examples[EXAMPLE_RANK_WITHIN_KIND]
image = image_by_index[IMAGE_INDEX]
crop = next(
    row
    for row in crop_table
    if int(row["image_index"]) == IMAGE_INDEX and int(row["crop_rank"]) == CROP_RANK
)
crop_offset = (float(crop["offset_x_px"]), float(crop["offset_y_px"]))

print("chosen example:", example.example_id, example.kind)
print("chosen image:", IMAGE_INDEX)
print("chosen crop:", crop)


#%% Plot the chosen source image, retinal crop path, and eye trace
def plot_source_and_trace(image: np.ndarray, trace: np.ndarray, crop_offset: tuple[float, float]) -> None:
    centers = model_crop_centers_px(
        trace,
        image.shape,
        crop_center_offset_px=crop_offset,
    )
    speed = np.linalg.norm(np.diff(trace, axis=0, prepend=trace[:1]), axis=1) / DT
    t = np.arange(trace.shape[0]) * DT
    vmin, vmax = np.percentile(image, [1, 99])

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    axs[0].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    axs[0].plot(centers[:, 0], centers[:, 1], color="#1f77b4", lw=1.0)
    axs[0].scatter(centers[0, 0], centers[0, 1], color="green", s=20, label="start")
    axs[0].scatter(centers[-1, 0], centers[-1, 1], color="red", s=20, label="end")
    axs[0].add_patch(Rectangle(
        (centers[0, 0] - OUT_SIZE[1] / 2, centers[0, 1] - OUT_SIZE[0] / 2),
        OUT_SIZE[1],
        OUT_SIZE[0],
        fill=False,
        edgecolor="#d62728",
        linewidth=1.0,
    ))
    axs[0].set_title("source image + crop path")
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].plot(t, trace[:, 0] * 60.0, label="x")
    axs[1].plot(t, trace[:, 1] * 60.0, label="y")
    if example.event_onset is not None:
        axs[1].axvline(example.event_onset * DT, color="0.1", ls="--", lw=1.0)
    axs[1].set_title("eye position")
    axs[1].set_xlabel("time (s)")
    axs[1].set_ylabel("arcmin")
    axs[1].legend(frameon=False)

    axs[2].plot(t, speed, color="0.2")
    if example.event_onset is not None:
        axs[2].axvline(example.event_onset * DT, color="0.1", ls="--", lw=1.0)
    axs[2].set_title("eye speed")
    axs[2].set_xlabel("time (s)")
    axs[2].set_ylabel("deg/s")

    for ax in axs[1:]:
        ax.grid(color="0.9", lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()


plot_source_and_trace(image, example.trace, crop_offset)


#%% Render retinal movies for real, stabilized, phase-shuffled, and SF-band controls
movies, control_images, pyramid_audits = stimulus_condition_movies(
    image,
    example.trace,
    t_max=T_MAX,
    seed=SEED + IMAGE_INDEX * 1009 + CROP_RANK * 37,
    crop_center_offset_px=crop_offset,
)

print("movie shapes:")
for condition, movie in movies.items():
    print(f"  {condition:25s} {movie.shape}")
print("pyramid audit rows:")
for row in pyramid_audits:
    print(row)


#%% Plot selected movie frames side by side
def show_movie_frames(
    movies: dict[str, np.ndarray],
    condition_order: tuple[str, ...],
    frames: tuple[int, ...] = (0, 31, 63, 95, 127),
) -> None:
    n_rows = len(frames)
    n_cols = len(condition_order)
    vals = np.concatenate([movies[c].ravel()[:: max(1, movies[c].size // 50000)] for c in condition_order])
    vmin, vmax = np.percentile(vals, [1, 99])
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.8 * n_rows), squeeze=False)
    for r, frame in enumerate(frames):
        frame = min(int(frame), min(movies[c].shape[0] for c in condition_order) - 1)
        for c, condition in enumerate(condition_order):
            axs[r, c].imshow(movies[condition][frame], cmap="gray", vmin=vmin, vmax=vmax)
            if r == 0:
                axs[r, c].set_title(CONDITION_LABELS.get(condition, condition))
            if c == 0:
                axs[r, c].set_ylabel(f"frame {frame}")
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
    fig.tight_layout()


show_movie_frames(movies, ("real", "stabilized", "pyramid_phase_scrambled"))
show_movie_frames(movies, SF_BANDS_4)


#%% Play the retinal movies inside a notebook or interactive Python session
def animate_movie_comparison(
    movies: dict[str, np.ndarray],
    condition_order: tuple[str, ...],
    *,
    interval_ms: int = 40,
) -> FuncAnimation:
    vals = np.concatenate([movies[c].ravel()[:: max(1, movies[c].size // 50000)] for c in condition_order])
    vmin, vmax = np.percentile(vals, [1, 99])
    n_frames = min(movies[c].shape[0] for c in condition_order)
    fig, axs = plt.subplots(1, len(condition_order), figsize=(3.2 * len(condition_order), 3.4))
    axs = np.atleast_1d(axs)
    ims = []
    for ax, condition in zip(axs, condition_order, strict=True):
        im = ax.imshow(movies[condition][0], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(CONDITION_LABELS.get(condition, condition))
        ax.set_xticks([])
        ax.set_yticks([])
        ims.append(im)

    def update(frame: int):
        for im, condition in zip(ims, condition_order, strict=True):
            im.set_data(movies[condition][frame])
        fig.suptitle(f"frame {frame}/{n_frames - 1}")
        return ims

    return FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=False)


anim_phase = animate_movie_comparison(movies, ("real", "stabilized", "pyramid_phase_scrambled"))
anim_sf = animate_movie_comparison(movies, SF_BANDS_4)

if SHOW_JS_ANIMATION and HTML is not None and display is not None:
    display(HTML(anim_phase.to_jshtml()))


#%% Optionally save MP4s for this chosen example
if SAVE_MP4S:
    base = f"{example.example_id}_image{IMAGE_INDEX:02d}_crop{CROP_RANK:02d}"
    save_stimulus_comparison_mp4(
        MOVIE_DIR / f"interactive_phase_{base}.mp4",
        movies,
        condition_order=("real", "stabilized", "pyramid_phase_scrambled"),
        title=f"{example.kind} {base}: phase controls",
        fps=30,
    )
    save_stimulus_comparison_mp4(
        MOVIE_DIR / f"interactive_sf_{base}.mp4",
        movies,
        condition_order=SF_BANDS_4,
        title=f"{example.kind} {base}: SF bands",
        fps=30,
    )
    print("saved MP4s to", MOVIE_DIR)


#%% Build and inspect the exact model lag cubes
LAG_CONDITIONS = ("real", "stabilized", "pyramid_phase_scrambled") + SF_BANDS_4
cubes_by_condition = {
    condition: condition_lag_cubes(
        image,
        example.trace,
        condition=condition,
        seed=SEED + IMAGE_INDEX * 1009 + CROP_RANK * 37,
        t_max=T_MAX,
        crop_center_offset_px=crop_offset,
    )
    for condition in LAG_CONDITIONS
}
blocks_by_condition = {
    condition: block_endpoint_lag_cubes(cubes)[0]
    for condition, cubes in cubes_by_condition.items()
}
_stable_blocks = blocks_by_condition["stabilized"]
print("lag cube shape:", cubes_by_condition["real"].shape, "(time, lag, height, width)")
print("lag sample shape:", _stable_blocks.shape, "(time, lag, height, width)")
print(
    "stabilized max abs diff across all lags:",
    float(np.max(np.abs(_stable_blocks - _stable_blocks[:, :1]))),
)


#%% Plot one lag cube. Lag 0 is the current frame; higher lag is farther back.
def plot_lag_cube(condition: str = "real", block_index: int = 0) -> None:
    lags = np.array([0, 1, 2, 4, 8, 16, 24, N_LAGS - 1], dtype=int)
    block = blocks_by_condition[condition][block_index]
    vals = block[lags]
    vmin, vmax = np.percentile(vals, [1, 99])
    fig, axs = plt.subplots(2, 4, figsize=(10, 5.2), squeeze=False)
    for ax, lag, frame in zip(axs.ravel(), lags, vals, strict=True):
        ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"lag {int(lag)}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{condition}: lag sample {block_index}")
    fig.tight_layout()


plot_lag_cube("real", block_index=0)
plot_lag_cube("stabilized", block_index=0)
plot_lag_cube("pyramid_phase_scrambled", block_index=0)


#%% Run the model on full 128-frame lag cubes and plot population rates
RATE_CONDITIONS = ("real", "stabilized", "pyramid_phase_scrambled")
rates_by_condition: dict[str, np.ndarray] = {}
rate_maps_by_condition: dict[str, np.ndarray] = {}

for condition in RATE_CONDITIONS:
    rates, rate_map = run_lag_cube_rates(
        model,
        population,
        device,
        cubes_by_condition[condition],
        batch_size=BATCH_SIZE,
        return_rate_map=True,
    )
    rates_by_condition[condition] = rates
    if rate_map is not None:
        rate_maps_by_condition[condition] = rate_map
    print(condition, "rates", rates.shape, "rate map", None if rate_map is None else rate_map.shape)


fig, axs = plt.subplots(1, 2, figsize=(11, 4))
frame_times = np.arange(cubes_by_condition["real"].shape[0]) * DT
for condition, rates in rates_by_condition.items():
    axs[0].plot(frame_times, rates.mean(axis=1), label=CONDITION_LABELS.get(condition, condition))
    axs[1].plot(frame_times, rates.sum(axis=1), label=CONDITION_LABELS.get(condition, condition))
axs[0].set_title("mean rate over sampled units")
axs[0].set_ylabel("spikes/s")
axs[1].set_title("population summed rate")
for ax in axs:
    ax.set_xlabel("time (s)")
    ax.grid(color="0.9", lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
axs[0].legend(frameon=False)
fig.tight_layout()


#%% Plot 3x3 spatial activation maps for the most variable real units
def plot_activation_grid(condition: str = "real", frame: int = 0, units: np.ndarray | None = None) -> None:
    rate_map = rate_maps_by_condition[condition]
    if units is None:
        scores = np.nanvar(rate_maps_by_condition["real"], axis=(0, 2, 3))
        units = np.argsort(scores)[-9:][::-1]
    units = np.asarray(units, dtype=int)[:9]
    fig, axs = plt.subplots(3, 3, figsize=(7.5, 7.5), squeeze=False)
    for ax, unit in zip(axs.ravel(), units, strict=True):
        vals = rate_map[:, unit]
        vmin, vmax = np.percentile(vals, [1, 99.5])
        ax.imshow(rate_map[frame, unit], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"unit {int(unit)}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{condition} activation maps, block frame {frame}")
    fig.tight_layout()


top_units = np.argsort(np.nanvar(rate_maps_by_condition["real"], axis=(0, 2, 3)))[-9:][::-1]
plot_activation_grid("real", frame=0, units=top_units)
plot_activation_grid("stabilized", frame=0, units=top_units)
plot_activation_grid("pyramid_phase_scrambled", frame=0, units=top_units)


#%% Optional: compute cumulative FI and SSI for this one example.
# This is slower because it reruns the model at shifted retinal positions.
# It writes interactive figures only; rerun the production pipeline to update
# figures under a production output folder.
if RUN_INFORMATION_CELL:
    block_times = block_current_samples(blocks_by_condition["real"].shape[0], n_lags=N_LAGS) * DT
    fisher_shifts = finite_difference_shift_set(fisher_step_arcmin=0.5)
    all_shifts = unique_shifts(fisher_shifts)
    info_by_condition = {}

    for condition in LAG_CONDITIONS:
        print("information condition:", condition)
        rate_maps_by_shift = run_shifted_lag_cube_rate_maps(
            model,
            population,
            device,
            blocks_by_condition[condition],
            all_shifts,
            batch_size=BATCH_SIZE,
        )
        fisher = cumulative_pattern_fisher(
            rate_maps_by_shift,
            fisher_step_arcmin=0.5,
            dt=DT,
        )
        ssi = cumulative_spatial_ssi(rate_maps_by_shift[(0.0, 0.0)], dt=DT)
        info_by_condition[condition] = {"fisher": fisher, "ssi": ssi}

    def plot_cumulative_information(
        condition_order: tuple[str, ...],
        *,
        title: str,
        save_name: str,
    ) -> None:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        for condition in condition_order:
            info = info_by_condition[condition]
            label = CONDITION_LABELS.get(condition, condition)
            axs[0].plot(
                block_times,
                info["fisher"]["cumulative_fisher_pattern"],
                marker="o",
                label=label,
            )
            axs[1].plot(
                block_times,
                info["ssi"]["cumulative_spatial_ssi_bits"],
                marker="o",
                label=label,
            )
        axs[0].set_title("cumulative pattern FI")
        axs[1].set_title("cumulative spatial SSI (bits)")
        for ax in axs:
            ax.set_xlabel("time (s)")
            ax.grid(color="0.9", lw=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axs[0].legend(frameon=False)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / save_name, dpi=180, bbox_inches="tight", facecolor="white")

    phase_conditions = ("real", "stabilized", "pyramid_phase_scrambled")
    sf_conditions = ("real",) + SF_BANDS_4
    plot_cumulative_information(
        phase_conditions,
        title=f"{example.example_id} image {IMAGE_INDEX} crop {CROP_RANK}: phase controls",
        save_name="interactive_cumulative_information_phase.pdf",
    )
    plot_cumulative_information(
        sf_conditions,
        title=f"{example.example_id} image {IMAGE_INDEX} crop {CROP_RANK}: spatial-frequency bands",
        save_name="interactive_cumulative_information_sf.pdf",
    )

    gain_conditions = ("real", "pyramid_phase_scrambled") + SF_BANDS_4
    stable_fi = info_by_condition["stabilized"]["fisher"]["cumulative_fisher_pattern"][-1]
    stable_ssi = info_by_condition["stabilized"]["ssi"]["cumulative_spatial_ssi_bits"][-1]
    fi_gain = [
        info_by_condition[condition]["fisher"]["cumulative_fisher_pattern"][-1] - stable_fi
        for condition in gain_conditions
    ]
    ssi_gain = [
        info_by_condition[condition]["ssi"]["cumulative_spatial_ssi_bits"][-1] - stable_ssi
        for condition in gain_conditions
    ]
    labels = [CONDITION_LABELS.get(condition, condition) for condition in gain_conditions]

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(gain_conditions))
    axs[0].bar(x, fi_gain, color="0.35")
    axs[1].bar(x, ssi_gain, color="0.35")
    axs[0].set_title("final pattern FI gain over stabilized")
    axs[1].set_title("final cumulative spatial SSI gain over stabilized (bits)")
    for ax in axs:
        ax.axhline(0.0, color="0.1", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", color="0.9", lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"{example.example_id} image {IMAGE_INDEX} crop {CROP_RANK}: final information gains")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "interactive_final_information_gain.pdf", dpi=180, bbox_inches="tight", facecolor="white")

    for condition, info in info_by_condition.items():
        spatial_ssi = np.asarray(info["ssi"]["cumulative_spatial_ssi_bits"])
        print(
            condition,
            "negative spatial SSI increments:",
            int((np.diff(spatial_ssi) < -1e-9).sum()),
        )
    print("saved cumulative information figures to", FIGURE_DIR)
else:
    print("Set RUN_INFORMATION_CELL = True to compute cumulative FI/SSI for this example.")
