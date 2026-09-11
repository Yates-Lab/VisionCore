"""Shared, production-only rendering primitives for Figure 4.

These functions are the small source closure required by the canonical A--H
renderer. They deliberately exclude the historical exploratory composers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image as PILImage, ImageDraw
from scipy.optimize import least_squares
from VisionCore.figure_typography import apply_font_floor

EPS = np.finfo(np.float64).eps
BLUE = "#0072B2"
ORANGE = "#D55E00"
PURPLE = "#6A51A3"
ROLE_COLORS = (BLUE, ORANGE)
ROLE_NAMES = ("low SF / high TF", "high SF / low TF")
PASSBAND_RESPONSE_FRACTION = 0.55
YU_EFFECTIVE_CONDITIONS = 36
YU_FIT_EPS = 1e-10
CYAN = "#00A6B8"
CABINET_ALPHA = np.deg2rad(45.0)
CABINET_DEPTH = 0.5
DEPTH_VEC = np.array(
    [-np.cos(CABINET_ALPHA) * CABINET_DEPTH, np.sin(CABINET_ALPHA) * CABINET_DEPTH]
)
SCREEN_YAW_DEG = -22.0


def skewed_log_gaussian(log_frequency: np.ndarray, preferred_log_frequency: float, width: float, skew: float) -> np.ndarray:
    """Yu et al. Eqs. 2/3/5, with the analytic zero-skew limit.

    The frequencies and preference are already log2 transformed.  The small
    denominator guard only handles the removable numerical singularity; it
    does not clip the function or alter its finite tails.
    """
    delta = np.asarray(log_frequency, dtype=np.float64) - np.asarray(preferred_log_frequency, dtype=np.float64)
    width = float(width)
    skew = float(skew)
    if abs(skew) < 1e-05:
        return np.exp(-0.5 * np.square(delta / max(width, EPS)))
    denominator = width - skew * delta
    denominator = np.where(np.abs(denominator) < 1e-06, np.copysign(1e-06, np.where(denominator == 0, 1.0, denominator)), denominator)
    return np.exp(-0.5 * np.square(delta / denominator)) - np.exp(-1.0 / np.square(skew))


def yu_surface(parameters: np.ndarray, log_sf: np.ndarray, log_tf: np.ndarray, *, inseparable: bool) -> np.ndarray:
    """Evaluate Yu R0 (7 parameters) or R1 (8 parameters)."""
    amplitude, sf_star, sigma_s, zeta_s, tf_star, sigma_t, zeta_t = parameters[:7]
    q = float(parameters[7]) if inseparable else 0.0
    spatial = skewed_log_gaussian(log_sf, sf_star, sigma_s, zeta_s)
    tf_center = tf_star + q * (log_sf - sf_star)
    temporal = skewed_log_gaussian(log_tf, tf_center, sigma_t, zeta_t)
    return float(amplitude) * spatial * temporal


def _orientation_surface(unit: pd.DataFrame, *, value_column: str='mean_rate', preferred_orientation_deg: float | None=None) -> tuple[float, pd.DataFrame]:
    if value_column not in unit:
        raise ValueError(f'unit tuning table lacks {value_column!r}')
    values = pd.to_numeric(unit[value_column], errors='coerce')
    dynamic = unit.loc[unit.temporal_hz.gt(0) & values.notna()].copy()
    if dynamic.empty:
        raise ValueError('unit has no dynamic grating responses')
    orientation = float(preferred_orientation_deg) if preferred_orientation_deg is not None else float(dynamic.loc[dynamic[value_column].idxmax()].probe_orientation_deg)
    surface = dynamic.loc[np.isclose(dynamic.probe_orientation_deg, orientation)].copy()
    if surface.empty:
        raise ValueError(f'unit has no samples at orientation {orientation:g} deg')
    return (orientation, surface)


def _surface_arrays(surface: pd.DataFrame, *, value_column: str='mean_rate') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = surface.pivot(index='temporal_hz', columns='spatial_cpd', values=value_column).sort_index(axis=0).sort_index(axis=1)
    sf = pivot.columns.to_numpy(dtype=float)
    tf = pivot.index.to_numpy(dtype=float)
    response = pivot.to_numpy(dtype=float)
    sf_grid, tf_grid = np.meshgrid(sf, tf, indexing='xy')
    return (sf_grid, tf_grid, response)


def full_support_six_by_six(sf_grid: np.ndarray, tf_grid: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select 36 acquired samples spanning the complete measured SF×TF bank.

    This display grid is deliberately distinct from the peak-centered 36
    conditions used to fit the Yu model. Indices are spaced across the full
    logarithmic acquisition grid, including both endpoints. No response values
    are interpolated or extrapolated.
    """
    sf_grid = np.asarray(sf_grid)
    tf_grid = np.asarray(tf_grid)
    response = np.asarray(response)
    if sf_grid.shape != response.shape or tf_grid.shape != response.shape:
        raise ValueError('SF, TF, and response grids must have matching shapes')
    if response.shape[0] < 6 or response.shape[1] < 6:
        raise ValueError('measured bank must provide at least six SF and TF samples')
    sf_index = np.rint(np.linspace(0, response.shape[1] - 1, 6)).astype(int)
    tf_index = np.rint(np.linspace(0, response.shape[0] - 1, 6)).astype(int)
    selection = np.ix_(tf_index, sf_index)
    return (sf_grid[selection], tf_grid[selection], response[selection])


def measured_full_support_surface(unit: pd.DataFrame, *, value_column: str='mean_rate', preferred_orientation_deg: float | None=None) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Return a coarse, full-support view of actual grating measurements."""
    orientation, orientation_surface = _orientation_surface(unit, value_column=value_column, preferred_orientation_deg=preferred_orientation_deg)
    sf_grid, tf_grid, raw_response = _surface_arrays(orientation_surface, value_column=value_column)
    baseline = float(np.nanquantile(raw_response, 0.02))
    response = np.maximum(raw_response - baseline, 0.0)
    display_sf, display_tf, display_response = full_support_six_by_six(sf_grid, tf_grid, response)
    return (orientation, baseline, display_sf, display_tf, display_response)


def _radial_temporal_quadrature(spatial_cpd: np.ndarray, temporal_hz: np.ndarray) -> np.ndarray:
    """Integration weights for radial two-dimensional SF and positive TF."""
    spatial = np.asarray(spatial_cpd, dtype=float)
    temporal = np.asarray(temporal_hz, dtype=float)
    if spatial.ndim != 1 or temporal.ndim != 1 or len(spatial) < 2 or (len(temporal) < 2):
        raise ValueError('power grids must be one-dimensional')
    if np.any(np.diff(spatial) <= 0) or np.any(np.diff(temporal) <= 0):
        raise ValueError('power grids must be strictly increasing')
    return 2.0 * np.pi * spatial[:, None] * np.gradient(spatial)[:, None] * np.gradient(temporal)[None, :]


def yu_tuning_surface(fit_row: pd.Series, spatial_cpd: np.ndarray, temporal_hz: np.ndarray) -> np.ndarray:
    """Evaluate the selected Yu et al. R0/R1 fit on an arbitrary log grid."""
    spatial = np.asarray(spatial_cpd, dtype=np.float64)
    temporal = np.asarray(temporal_hz, dtype=np.float64)
    if np.any(spatial <= 0) or np.any(temporal <= 0):
        raise ValueError('Yu tuning grids must contain positive frequencies')
    sf_grid, tf_grid = np.meshgrid(spatial, temporal, indexing='ij')
    parameters = np.asarray([1.0, np.log2(float(fit_row.preferred_sf_cpd)), float(fit_row.sigma_s), float(fit_row.zeta_s), np.log2(float(fit_row.preferred_tf_hz)), float(fit_row.sigma_t), float(fit_row.zeta_t), float(fit_row.q)])
    selected_model = str(fit_row.get('selected_model', fit_row.get('selected_model_yu36', '')))
    if selected_model not in ('R0', 'R1'):
        raise ValueError('tuning fit row must declare selected_model or selected_model_yu36')
    surface = yu_surface(parameters, np.log2(sf_grid), np.log2(tf_grid), inseparable=selected_model == 'R1')
    surface = np.clip(surface, 0.0, None)
    return surface / max(float(np.max(surface)), EPS)


def support_limited_yu_tuning_surface(fit_row: pd.Series, spatial_cpd: np.ndarray, temporal_hz: np.ndarray) -> np.ndarray:
    """Evaluate a Yu fit only inside the SF×TF support used to estimate it."""
    required = ('measured_min_sf_cpd', 'measured_max_sf_cpd', 'measured_min_tf_hz', 'measured_max_tf_hz')
    missing = [name for name in required if name not in fit_row.index]
    if missing:
        raise ValueError('full-support tuning row is missing measured frequency bounds: ' + ', '.join(missing))
    spatial = np.asarray(spatial_cpd, dtype=float)
    temporal = np.asarray(temporal_hz, dtype=float)
    surface = yu_tuning_surface(fit_row, spatial, temporal)
    supported = (spatial[:, None] >= float(fit_row.measured_min_sf_cpd)) & (spatial[:, None] <= float(fit_row.measured_max_sf_cpd)) & (temporal[None, :] >= float(fit_row.measured_min_tf_hz)) & (temporal[None, :] <= float(fit_row.measured_max_tf_hz))
    return np.where(supported, surface, np.nan)


def spatial_information(rate_map: np.ndarray) -> tuple[float, float, np.ndarray]:
    value = np.clip(np.asarray(rate_map, dtype=np.float64), 0.0, None)
    mean = float(value.mean())
    gain = value / max(mean, EPS)
    information = float(np.mean(gain * np.log2(np.maximum(gain, EPS))))
    return (mean, information, gain)


def _draw_spacetime_cube(axis: plt.Axes, movie: np.ndarray, title: str, *, outline: str='#00A6B8') -> None:
    corners = box_corners_3d((0.0, 0.0, 0.0), (1.65, 1.28, 1.55), yaw_deg=-40.0, pitch_deg=0.0, roll_deg=0.0)
    projected = _draw_lag_cube(axis, np.asarray(movie, dtype=np.float64)[::-1], corners, outline=outline, edge_width=1.15, zorder=3.0)
    extent = np.ptp(projected, axis=0)
    axis.set_xlim(projected[:, 0].min() - 0.08 * extent[0], projected[:, 0].max() + 0.08 * extent[0])
    axis.set_ylim(projected[:, 1].min() - 0.18 * extent[1], projected[:, 1].max() + 0.12 * extent[1])
    axis.set_aspect('equal')
    axis.axis('off')
    axis.set_title(title, fontsize=7.7, pad=2)
    axis.annotate('250 ms', xy=(0.72, 0.04), xytext=(0.26, 0.04), xycoords='axes fraction', textcoords='axes fraction', arrowprops={'arrowstyle': '-|>', 'lw': 0.8, 'color': '0.35'}, ha='center', va='center', fontsize=6.1, color='0.35')


def configure() -> None:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.2, 'axes.titlesize': 7.5, 'axes.titleweight': 'semibold', 'axes.labelsize': 6.8, 'xtick.labelsize': 6.0, 'ytick.labelsize': 6.0, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def _panel_title(subfigure, label: str, title: str, *, size: float=8.4) -> None:
    subfigure.text(0.0, 0.985, label, ha='left', va='top', fontsize=10.0, fontweight='bold')
    subfigure.text(0.07, 0.985, title, ha='left', va='top', fontsize=size, fontweight='semibold')


def _frequency_axes(axis: plt.Axes, *, show_y: bool) -> None:
    axis.set_xscale('log', base=2)
    axis.set_yscale('log', base=2)
    axis.set_xticks((1, 2, 4, 8), ('1', '2', '4', '8'))
    axis.set_yticks((1, 4, 16, 64), ('1', '4', '16', '64'))
    axis.set_xlabel('SF (cycles/deg)')
    if show_y:
        axis.set_ylabel('TF (Hz)')
    else:
        axis.set_yticklabels([])


def _draw_network(axis: plt.Axes) -> None:
    """One schematic feature-plane stack, without individual stages/readouts.

    The uniform grids are vector symbols, not measured feature activations;
    the actual, checkpoint-derived activation maps appear to the right.
    """
    axis.set_box_aspect(1.0)
    axis.set_anchor('C')
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    horizontal = np.array((0.57, 0.14))
    vertical = np.array((0.0, 0.49))
    for index, (fill, edge) in enumerate(zip(
        ('#eef3f7', '#e1eaf2', '#d4e2ee', '#bdd4e4'),
        ('#8174a2', '#7186a6', '#5679a4', '#5791ad'),
    )):
        origin = np.array((0.12 + 0.065 * index, 0.34 - 0.055 * index))
        corners = [origin, origin + horizontal, origin + horizontal + vertical, origin + vertical]
        axis.add_patch(Polygon(corners, facecolor=fill, edgecolor=edge, lw=0.8, zorder=3 * index))
        for fraction in np.linspace(0, 1, 9)[1:-1]:
            for start, step in ((origin + fraction * horizontal, vertical),
                                (origin + fraction * vertical, horizontal)):
                axis.plot([start[0], start[0] + step[0]], [start[1], start[1] + step[1]],
                          color='white', lw=0.35, alpha=0.65, zorder=3 * index + 1)
    axis.axis('off')


def _load_panel_a_audit(audit_dir: Path) -> tuple[dict[str, np.ndarray | float | int | str], dict[str, object]]:
    archive_path = Path(audit_dir) / 'selected_example.npz'
    summary_path = Path(audit_dir) / 'summary.json'
    if not archive_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f'Panel-A audit requires {archive_path} and {summary_path}')
    with np.load(archive_path, allow_pickle=False) as archive:
        example = {key: archive[key] for key in archive.files}
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    for key in ('unit_index', 'image_index', 'trace_index', 'output_rate_hz', 'motion_rate_map', 'stable_rate_map', 'motion_history', 'stable_history', 'trace_window_xy_filtered_endpoint_aligned'):
        if key not in example:
            raise ValueError(f'Panel-A audit archive lacks {key}')
    motion_history = np.asarray(example['motion_history'], dtype=float)
    stable_history = np.asarray(example['stable_history'], dtype=float)
    trace = np.asarray(example['trace_window_xy_filtered_endpoint_aligned'], dtype=float)
    if motion_history.shape != stable_history.shape or motion_history.ndim != 3:
        raise ValueError('Panel-A motion and stabilized histories must be matched 3-D arrays')
    if trace.ndim != 2 or trace.shape[1] != 2:
        raise ValueError('Panel-A filtered trace must have shape [time,2]')
    alignment = summary.get('trace_window_selection', {})
    alignment_kind = str(alignment.get('alignment_kind', 'endpoint'))
    if alignment_kind == 'resolved_model_mean_peak_lag':
        anchor_index = int(alignment['anchor_chronological_frame_index'])
        if not 0 <= anchor_index < len(trace):
            raise ValueError('Panel-A model-lag anchor is outside the filtered history')
        if not np.allclose(trace[anchor_index], 0.0, atol=1e-06, rtol=0.0):
            raise ValueError('Panel-A filtered trace is not zero at the model-lag anchor')
        if not np.allclose(motion_history[anchor_index], stable_history[anchor_index], atol=1e-05, rtol=0.0):
            raise ValueError('Panel-A histories do not share the model-lag anchor frame')
    elif alignment_kind == 'endpoint':
        anchor_index = len(trace) - 1
        if not np.allclose(trace[-1], 0.0, atol=1e-06, rtol=0.0):
            raise ValueError('Panel-A filtered trace is not endpoint-aligned')
        if not np.allclose(motion_history[-1], stable_history[-1], atol=1e-05, rtol=0.0):
            raise ValueError('Panel-A endpoint-matched histories do not share the current frame')
    else:
        raise ValueError(f'unsupported Panel-A history alignment: {alignment_kind}')
    if np.asarray(example['motion_rate_map']).shape != np.asarray(example['stable_rate_map']).shape:
        raise ValueError('Panel-A response maps are not shape matched')
    result: dict[str, np.ndarray | float | int | str] = {**example, 'unit_index': int(np.asarray(example['unit_index'])), 'image_index': int(np.asarray(example['image_index'])), 'trace_index': int(np.asarray(example['trace_index'])), 'endpoint_frame': int(np.asarray(example['endpoint_frame'])), 'output_rate_hz': float(np.asarray(example['output_rate_hz'])), 'alignment_kind': alignment_kind, 'alignment_anchor_frame': int(anchor_index), 'alignment_anchor_lag_frames': int(alignment.get('anchor_lag_frames', len(trace) - 1 - anchor_index)), 'alignment_anchor_time_ms': float(alignment.get('anchor_time_before_prediction_ms', 1000.0 * (len(trace) - 1 - anchor_index) / float(example['output_rate_hz']))), 'selection_kind': str(summary.get('response_selection', {}).get('selection_tier', 'audited'))}
    return (result, summary)


def draw_panel_a(subfigure, example: dict[str, np.ndarray | float | int | str]) -> dict[str, object]:
    _panel_title(subfigure, 'A', '')
    unit_index = int(example['unit_index'])
    image_index = int(example['image_index'])
    trace_index = int(example['trace_index'])
    output_rate_hz = float(example['output_rate_hz'])
    motion_map = np.asarray(example['motion_rate_map'], dtype=float)
    stable_map = np.asarray(example['stable_rate_map'], dtype=float)
    motion_history = np.asarray(example['motion_history'], dtype=float)
    stable_history = np.asarray(example['stable_history'], dtype=float)
    motion_mean, motion_ssi, motion_gain = spatial_information(motion_map)
    stable_mean, stable_ssi, stable_gain = spatial_information(stable_map)
    ink, muted = '#303b45', '#617080'
    for x, label in ((0.145, 'Retinal history'), (0.385, 'Predictive model'),
                     (0.625, 'Activation maps'), (0.855, 'Difference')):
        subfigure.text(x, 0.965, label, ha='center', va='top',
                       fontsize=7.7, fontweight='semibold', color=ink)
    subfigure.text(0.625, 0.905, 'response / spatial mean', ha='center', va='top', fontsize=6.1, color=muted)
    subfigure.text(0.855, 0.905, 'motion − stabilized', ha='center', va='top', fontsize=6.1, color=muted)
    map_low = min(float(np.quantile(motion_gain, 0.01)), float(np.quantile(stable_gain, 0.01)))
    map_high = max(float(np.quantile(motion_gain, 0.995)), float(np.quantile(stable_gain, 0.995)))
    conditions = (('Measured motion', motion_mean, motion_ssi, motion_gain, motion_history), ('Stabilized', stable_mean, stable_ssi, stable_gain, stable_history))
    cube_axes, map_axes = [], []
    for row, (condition, mean, information, gain, history) in enumerate(conditions):
        cube_axis = subfigure.add_axes((0.03, 0.535 - 0.415 * row, 0.23, 0.33))
        _draw_spacetime_cube(cube_axis, history, condition, outline='#5791ad')
        cube_axis.title.set_fontsize(6.8)
        cube_axis.title.set_color(ink)
        map_axis = subfigure.add_axes((0.53, 0.555 - 0.415 * row, 0.19, 0.30))
        map_axis.imshow(gain, cmap='viridis', vmin=map_low, vmax=map_high)
        map_axis.set_xticks([])
        map_axis.set_yticks([])
        for spine in map_axis.spines.values():
            spine.set_visible(True)
            spine.set_color('#d8dfe3')
            spine.set_linewidth(0.5)
        map_axis.text(0.5, -0.065, f'{output_rate_hz * mean:.2f} spikes/s\n{information:.3f} bits/spike',
                      transform=map_axis.transAxes, ha='center', va='top', color=muted, fontsize=6.1)
        cube_axes.append(cube_axis)
        map_axes.append(map_axis)
    network_axis = subfigure.add_axes((0.295, 0.27, 0.18, 0.49))
    _draw_network(network_axis)
    network_axis.text(0.5, -0.09, 'same model, both conditions', ha='center', va='top',
                      transform=network_axis.transAxes, fontsize=6.1, color=muted)
    difference_axis = subfigure.add_axes((0.775, 0.265, 0.16, 0.505))
    difference = motion_gain - stable_gain
    difference_limit = max(float(np.quantile(np.abs(difference), 0.995)), EPS)
    image = difference_axis.imshow(difference, cmap='RdBu_r', vmin=-difference_limit, vmax=difference_limit)
    difference_axis.set_xticks([])
    difference_axis.set_yticks([])
    for spine in difference_axis.spines.values():
        spine.set_visible(True)
        spine.set_color('#d8dfe3')
        spine.set_linewidth(0.5)
    relative = 100.0 * (motion_ssi - stable_ssi) / max(stable_ssi, EPS)
    rate_relative = 100.0 * (motion_mean - stable_mean) / max(stable_mean, EPS)
    # Resolve equal-aspect image axes before attaching arrows to their edges.
    subfigure.canvas.draw()
    network = network_axis.get_position()
    difference_box = difference_axis.get_position()
    colorbar_axis = subfigure.add_axes((difference_box.x1 + 0.008, difference_box.y0,
                                      0.006, difference_box.height))
    colorbar = subfigure.colorbar(image, cax=colorbar_axis,
                                ticks=MaxNLocator(nbins=3, symmetric=True, prune='both'))
    colorbar.outline.set_linewidth(0.5)
    colorbar.ax.tick_params(labelsize=6.1, pad=1.5, length=2, width=0.5)
    def arrow(start, stop, *, curve=0.0):
        subfigure.add_artist(FancyArrowPatch(start, stop, transform=subfigure.transFigure,
                             arrowstyle='-|>', connectionstyle=f'arc3,rad={curve}',
                             mutation_scale=7.0, linewidth=0.8, color=ink, shrinkA=0, shrinkB=0))
    for row, (cube_axis, map_axis) in enumerate(zip(cube_axes, map_axes)):
        cube, response = cube_axis.get_position(), map_axis.get_position()
        arrow((cube.x1 + 0.008, cube.y0 + 0.54 * cube.height),
              (network.x0 + 0.09 * network.width, network.y0 + (0.68 - 0.27 * row) * network.height),
              curve=0.08 if row else -0.08)
        arrow((network.x0 + 0.92 * network.width, network.y0 + (0.68 - 0.27 * row) * network.height),
              (response.x0 - 0.009, response.y0 + 0.5 * response.height),
              curve=-0.08 if row else 0.08)
        arrow((response.x1 + 0.009, response.y0 + 0.5 * response.height),
              (0.745, 0.518 + (0.025 if row == 0 else -0.025)))
    # The top response minus the bottom response yields the actual difference.
    operation = subfigure.add_axes((0.734, 0.488, 0.022, 0.06))
    operation.set_box_aspect(1)
    operation.add_patch(Circle((0.5, 0.5), 0.46, facecolor='white', edgecolor=muted, lw=0.7))
    operation.text(0.5, 0.5, '−', ha='center', va='center', fontsize=7.0, color=ink)
    operation.axis('off')
    arrow((0.758, 0.518), (difference_box.x0 - 0.005, 0.518))
    return {'unit_index': int(unit_index), 'image_index': image_index, 'trace_index': trace_index, 'selection_kind': str(example.get('selection_kind', 'unknown')), 'alignment_kind': str(example.get('alignment_kind', 'endpoint')), 'alignment_anchor_frame': int(example.get('alignment_anchor_frame', 59)), 'alignment_anchor_lag_frames': int(example.get('alignment_anchor_lag_frames', 0)), 'alignment_anchor_time_ms': float(example.get('alignment_anchor_time_ms', 0.0)), 'stable_rate_spikes_s': float(output_rate_hz * stable_mean), 'motion_rate_spikes_s': float(output_rate_hz * motion_mean), 'rate_change_percent': float(rate_relative), 'stable_ssi_bits_per_spike': float(stable_ssi), 'motion_ssi_bits_per_spike': float(motion_ssi), 'ssi_change_bits_per_spike': float(motion_ssi - stable_ssi), 'ssi_change_percent': float(relative), 'network_icon_contains_spatial_readout': False}


def _exemplar_tuning(tuning_table: pd.DataFrame, fits: pd.DataFrame) -> dict[str, object]:
    units = tuple(fits.index.astype(int))
    measured = []
    for unit in units:
        fit = fits.loc[unit]
        unit_table = tuning_table.loc[tuning_table.unit_index.eq(unit)]
        if unit_table.empty:
            raise ValueError(f'exemplar u{unit:03d} is absent from the tuning table')
        if 'source_unit_index' in unit_table and 'source_unit_index' in fits:
            observed = unit_table.source_unit_index.drop_duplicates().to_numpy(dtype=int)
            expected = int(fit.source_unit_index)
            if len(observed) != 1 or int(observed[0]) != expected:
                raise ValueError(f'exemplar u{unit:03d} identity mismatch: tuning source {observed.tolist()} versus fit source {expected}')
        orientation, _, sf_grid, tf_grid, response = measured_full_support_surface(unit_table, value_column=str(fit.get('response_column', 'mean_rate')), preferred_orientation_deg=float(fit.preferred_orientation_deg))
        measured.append({'orientation_deg': float(orientation), 'spatial': sf_grid[0].copy(), 'temporal': tf_grid[:, 0].copy(), 'response': response})
    return {'units': units, 'measured': measured}


def routing_metrics(archive: dict[str, np.ndarray], fits: pd.DataFrame) -> dict[str, object]:
    spatial = np.asarray(archive['spatial_cpd'], dtype=float)
    temporal = np.asarray(archive['temporal_hz'], dtype=float)
    regime_power = np.asarray(archive['spectral_regime_power_distribution'], dtype=float)
    regime_code = np.asarray(archive['spectral_regime_code'], dtype=int)
    dynamic_mass = np.asarray(archive['per_trace_dynamic_power_mass'], dtype=float)
    integrals = np.asarray(archive['spectral_regime_integrals'], dtype=float)
    if regime_power.shape != (2, len(spatial), len(temporal)):
        raise ValueError('Rucci regime power has inconsistent dimensions')
    if not np.allclose(integrals, 1.0, atol=1e-06, rtol=0.0):
        raise ValueError('conditional Rucci regime distributions must integrate to one')
    positive = regime_power[regime_power > 0]
    floor = max(float(np.quantile(positive, 0.01)), EPS)
    log_power = np.log10(np.maximum(regime_power, floor))
    display_low, display_high = np.quantile(log_power, (0.01, 0.99))
    contrast = np.log2(np.maximum(regime_power[1], floor) / np.maximum(regime_power[0], floor))
    contrast_limit = min(4.0, max(1.0, float(np.quantile(np.abs(contrast), 0.98))))
    surfaces = [support_limited_yu_tuning_surface(fits.loc[unit], spatial, temporal) for unit in fits.index.astype(int)]
    quadrature = _radial_temporal_quadrature(spatial, temporal)
    lasso_fraction = np.asarray([[float(np.sum(power[surface >= PASSBAND_RESPONSE_FRACTION] * quadrature[surface >= PASSBAND_RESPONSE_FRACTION])) for power in regime_power] for surface in surfaces])
    gain = lasso_fraction / np.maximum(lasso_fraction[:, [0]], EPS)
    routing_separation = float(gain[0, 1] / max(gain[1, 1], EPS)) if gain.shape[0] >= 2 else float('nan')
    raw_mass_ratio = float(np.median(dynamic_mass[regime_code == 1]) / max(np.median(dynamic_mass[regime_code == 0]), EPS))
    trace_metadata = {key: np.asarray(archive[key], dtype=dtype) for key, dtype in (('per_trace_power_centroid_hz', float), ('speed_deg_s', float), ('path_length_arcmin', float), ('microsaccade_count', int)) if key in archive}
    trace_metadata['regime_selection'] = str(np.asarray(archive.get('spectral_regime_selection', 'centroid_quartiles')).item())
    trace_metadata['regime_names'] = ('drift', 'microsaccades') if trace_metadata['regime_selection'] == 'events' else ('drift-rich', 'rapid-transient')
    return {'spatial': spatial, 'temporal': temporal, 'regime_power': regime_power, 'regime_code': regime_code, 'log_power': log_power, 'display_limits': (float(display_low), float(display_high)), 'contrast': contrast, 'contrast_limit': contrast_limit, 'surfaces': surfaces, 'lasso_fraction': lasso_fraction, 'lasso_gain': gain, 'routing_separation': routing_separation, 'raw_mass_ratio': raw_mass_ratio, 'integrals': integrals, **trace_metadata}


def _overlay_passband(axis: plt.Axes, spatial: np.ndarray, temporal: np.ndarray, surface: np.ndarray, color: str) -> None:
    """Draw the one authoritative fitted-passband contour and white halo."""
    axis.contour(spatial, temporal, surface.T, levels=(PASSBAND_RESPONSE_FRACTION,), colors=('white',), linewidths=2.1)
    axis.contour(spatial, temporal, surface.T, levels=(PASSBAND_RESPONSE_FRACTION,), colors=(color,), linewidths=1.0)


def _overlay_passbands(axis: plt.Axes, spatial: np.ndarray, temporal: np.ndarray, surfaces: list[np.ndarray]) -> None:
    for color, surface in zip(ROLE_COLORS, surfaces):
        _overlay_passband(axis, spatial, temporal, surface, color)


def select_population_units(data: dict[str, np.ndarray], unit_indices: list[int] | np.ndarray) -> dict[str, np.ndarray]:
    """Subset every unit-indexed replay tensor by explicit unit identity."""
    available = np.asarray(data['unit_indices'], dtype=int)
    requested = np.asarray(unit_indices, dtype=int)
    lookup = {int(unit): index for index, unit in enumerate(available)}
    missing = [int(unit) for unit in requested if int(unit) not in lookup]
    if missing:
        raise ValueError(f'validated tuning units are absent from mechanism replay: {missing}')
    positions = np.asarray([lookup[int(unit)] for unit in requested], dtype=int)
    selected: dict[str, np.ndarray] = {}
    for key, raw in data.items():
        value = np.asarray(raw)
        if key == 'unit_indices':
            selected[key] = requested.copy()
        elif value.ndim > 1 and value.shape[-1] == len(available):
            selected[key] = np.take(value, positions, axis=-1)
        else:
            selected[key] = value
    return selected


def _render_panel(output: Path, figsize: tuple[float, float], draw, *args, margins: tuple[float, float, float, float]=(0.08, 0.97, 0.12, 0.87), text_replacements: dict[str, str] | None=None, **kwargs) -> dict[str, object]:
    """Render one fixed-size vector panel for deterministic page composition."""
    figure = plt.figure(figsize=figsize, facecolor='white')
    left, right, bottom, top = margins
    figure.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    report = draw(figure, *args, **kwargs)
    if text_replacements:
        from matplotlib.text import Text
        for artist in figure.findobj(match=lambda obj: isinstance(obj, Text)):
            if artist.get_text() in text_replacements:
                artist.set_text(text_replacements[artist.get_text()])
    apply_font_floor(figure)
    figure.savefig(output, facecolor='white')
    plt.close(figure)
    return report


def _compose_page(output: Path, placements: list[tuple[Path, float, float]], *, page_width_in: float=8.5, page_height_in: float=11.0) -> None:
    """Place fixed-size vector panels on a PDF page."""
    from pypdf import PdfReader, PdfWriter, Transformation
    writer = PdfWriter()
    page = writer.add_blank_page(width=float(page_width_in) * 72.0, height=float(page_height_in) * 72.0)
    for source, x_in, y_in in placements:
        source_page = PdfReader(str(source)).pages[0]
        tx = 72.0 * float(x_in)
        ty = 72.0 * page_height_in - 72.0 * float(y_in) - float(source_page.mediabox.height)
        page.merge_transformed_page(source_page, Transformation().translate(tx=tx, ty=ty))
    with output.open('wb') as handle:
        writer.write(handle)


def _render_page_png(pdf: Path, png: Path) -> None:
    executable = shutil.which('pdftoppm')
    if executable is None:
        raise RuntimeError('pdftoppm is required to render the Figure-4 PNG')
    subprocess.run([executable, '-png', '-singlefile', '-r', '300', str(pdf), str(png.with_suffix(''))], check=True)


def _load_shard_summaries(paths: list[Path]) -> list[dict[str, object]]:
    summaries = []
    for raw_path in paths:
        path = Path(raw_path)
        summary_path = path / 'summary.json' if path.is_dir() else path.parent / 'summary.json'
        if not summary_path.exists():
            raise FileNotFoundError(f'missing replay provenance: {summary_path}')
        summaries.append(json.loads(summary_path.read_text(encoding='utf-8')))
    return summaries


def _summary_checkpoint_digest(summary: dict[str, object]) -> str:
    direct = str(summary.get('checkpoint_sha256', ''))
    if direct:
        return direct
    model = summary.get('model_provenance', {}).get('model', {})
    nested = str(model.get('checkpoint_sha256', ''))
    if nested:
        return nested
    shard_digests = {str(shard.get('model_provenance', {}).get('model', {}).get('checkpoint_sha256', '')) for shard in summary.get('shard_summaries', [])}
    shard_digests.discard('')
    return next(iter(shard_digests)) if len(shard_digests) == 1 else ''

def _perspective_coeffs(src_corners, dst_corners):
    """Solve for PIL.Image.PERSPECTIVE coefficients.

    PIL maps output(x', y') back to input(x, y) as:
        x = (a x' + b y' + c) / (g x' + h y' + 1)
        y = (d x' + e y' + f) / (g x' + h y' + 1)

    `src_corners`: 4×2 source-image pixel coords (where to sample FROM)
    `dst_corners`: 4×2 output-image pixel coords (target locations in the
                   *output canvas*; we use the bounding box of dst as the
                   canvas size).
    """
    M = []
    for (sx, sy), (dx, dy) in zip(src_corners, dst_corners):
        M.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        M.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
    A = np.array(M, dtype=np.float64)
    B = np.array(src_corners, dtype=np.float64).reshape(8)
    coeffs, *_ = np.linalg.lstsq(A, B, rcond=None)
    return coeffs

def cabinet_project(p3):
    """Project (N×3) (or shape-(3,)) world points to 2D screen coords."""
    p3 = np.asarray(p3, dtype=float)
    return p3[..., :2] + p3[..., 2:3] * DEPTH_VEC

def _R_x(angle_deg):
    a = np.deg2rad(angle_deg)
    c, s = (np.cos(a), np.sin(a))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _R_y(angle_deg):
    a = np.deg2rad(angle_deg)
    c, s = (np.cos(a), np.sin(a))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def _R_z(angle_deg):
    a = np.deg2rad(angle_deg)
    c, s = (np.cos(a), np.sin(a))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def _euler_rotation(yaw_deg, pitch_deg=0.0, roll_deg=0.0):
    return _R_y(yaw_deg) @ _R_x(pitch_deg) @ _R_z(roll_deg)

def box_corners_3d(front_center, size, *, yaw_deg=SCREEN_YAW_DEG, pitch_deg=0.0, roll_deg=0.0):
    """World corners of a rotated axis-aligned box (front-face center anchor)."""
    cx, cy, cz = front_center
    w, h, d = size
    w2, h2 = (w / 2.0, h / 2.0)
    local = np.array([[-w2, -h2, 0.0], [+w2, -h2, 0.0], [+w2, +h2, 0.0], [-w2, +h2, 0.0], [-w2, -h2, d], [+w2, -h2, d], [+w2, +h2, d], [-w2, +h2, d]])
    R = _euler_rotation(yaw_deg, pitch_deg, roll_deg)
    return local @ R.T + np.array([cx, cy, cz])

def _draw_quad_image(ax, image, dst_quad, *, src_corners=None, zorder=2, alpha=1.0, auto_contrast=False, out_res=512):
    """Warp `image` into `dst_quad` (4×2 data coords, LL,LR,UR,UL)."""
    H, W = image.shape[:2]
    if src_corners is None:
        src_corners = np.array([[0, H], [W, H], [W, 0], [0, 0]], dtype=np.float64)
    bx0, by0 = (dst_quad[:, 0].min(), dst_quad[:, 1].min())
    bx1, by1 = (dst_quad[:, 0].max(), dst_quad[:, 1].max())
    sx = out_res / (bx1 - bx0)
    sy = out_res / (by1 - by0)
    dst_px = np.column_stack([(dst_quad[:, 0] - bx0) * sx, (by1 - dst_quad[:, 1]) * sy])
    coeffs = _perspective_coeffs(src_corners, dst_px)
    pil = PILImage.fromarray(image.astype(np.uint8))
    if pil.mode != 'L':
        pil = pil.convert('L')
    if auto_contrast:
        arr = np.asarray(pil, dtype=np.float32)
        vmin, vmax = np.percentile(arr, [1, 99])
        if vmax > vmin:
            arr = np.clip((arr - vmin) / (vmax - vmin) * 255.0, 0, 255)
            pil = PILImage.fromarray(arr.astype(np.uint8))
    warped = pil.transform((out_res, out_res), PILImage.PERSPECTIVE, coeffs, resample=PILImage.BILINEAR)
    mask = PILImage.new('L', (out_res, out_res), 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in dst_px], fill=255)
    rgba = np.dstack([np.array(warped)] * 3 + [np.array(mask)])
    ax.imshow(rgba, extent=[bx0, bx1, by0, by1], origin='upper', zorder=zorder, interpolation='bilinear', alpha=alpha)

def _draw_lag_cube(ax, cube, corners3d, *, outline=CYAN, edge_width=1.4, zorder=4):
    """Texture front, top, and left faces of a 3D box from a (T,H,W) cube."""
    n_lags, H, W = cube.shape
    vmin, vmax = np.percentile(cube, [2, 98])
    if vmax <= vmin:
        vmax = vmin + 1.0

    def _norm(arr):
        a = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
        return (a * 255).astype(np.uint8)
    p2 = cabinet_project(corners3d)
    fLL, fLR, fUR, fUL = (p2[0], p2[1], p2[2], p2[3])
    bLL, bLR, bUR, bUL = (p2[4], p2[5], p2[6], p2[7])
    front_img = _norm(cube[0])
    _draw_quad_image(ax, front_img, np.array([fLL, fLR, fUR, fUL]), zorder=zorder + 0.3)
    top_img = _norm(cube[::-1, 0, :])
    _draw_quad_image(ax, top_img, np.array([fUL, fUR, bUR, bUL]), zorder=zorder + 0.2)
    left_img = _norm(cube[:, :, 0]).T
    _draw_quad_image(ax, left_img, np.array([fLL, bLL, bUL, fUL]), zorder=zorder + 0.1)
    for quad in (np.array([fLL, fLR, fUR, fUL]), np.array([fUL, fUR, bUR, bUL]), np.array([fLL, bLL, bUL, fUL])):
        ax.add_patch(Polygon(quad, closed=True, fill=False, edgecolor=outline, linewidth=edge_width, zorder=zorder + 0.5))
    return p2

@dataclass(frozen=True)
class SurfaceFit:
    model: str
    parameters: np.ndarray
    prediction: np.ndarray
    rss: float
    r2: float
    bic_dense: float
    bic_yu36: float
    success: bool

def fit_probability(bic_r0: float, bic_r1: float) -> float:
    """Stable form of Yu Eq. 7: P(R1 | R0, R1)."""
    delta = float(np.clip(0.5 * (bic_r1 - bic_r0), -700.0, 700.0))
    return float(1.0 / (1.0 + np.exp(delta)))

def _fit_bounds(log_sf: np.ndarray, log_tf: np.ndarray, inseparable: bool):
    lower = [0.0, float(log_sf.min() - 1.0), 0.12, -0.85, float(log_tf.min() - 1.0), 0.12, -0.85]
    upper = [8.0, float(log_sf.max() + 1.0), 4.5, 0.85, float(log_tf.max() + 1.0), 4.5, 0.85]
    if inseparable:
        lower.append(-1.0)
        upper.append(1.5)
    return (np.asarray(lower), np.asarray(upper))

def coefficient_of_determination(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = np.asarray(observed) - np.asarray(predicted)
    centered = np.asarray(observed) - float(np.mean(observed))
    denominator = float(np.sum(np.square(centered)))
    if denominator <= YU_FIT_EPS:
        return float('nan')
    return 1.0 - float(np.sum(np.square(residual))) / denominator

def bic_from_rss(rss: float, n: int, k: int) -> float:
    return float(n * np.log(max(float(rss) / n, YU_FIT_EPS)) + k * np.log(n))

def fit_yu_surface(log_sf: np.ndarray, log_tf: np.ndarray, response: np.ndarray, *, inseparable: bool, initial_from_r0: np.ndarray | None=None) -> SurfaceFit:
    """Fit R0/R1 with bounded multistart least squares."""
    x = np.asarray(log_sf, dtype=np.float64).ravel()
    y = np.asarray(log_tf, dtype=np.float64).ravel()
    z = np.asarray(response, dtype=np.float64).ravel()
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = (x[keep], y[keep], z[keep])
    if len(z) < 25 or float(np.ptp(z)) <= YU_FIT_EPS:
        raise ValueError('surface requires at least 25 finite, nonconstant conditions')
    scale = max(float(np.max(z)), YU_FIT_EPS)
    normalized = z / scale
    peak = int(np.argmax(normalized))
    lower, upper = _fit_bounds(x, y, inseparable)
    starts: list[np.ndarray] = []
    if initial_from_r0 is None:
        for width_s, width_t, skew_s, skew_t in ((0.8, 0.9, 0.0, 0.0), (1.4, 1.5, 0.0, 0.0), (2.2, 2.2, 0.0, 0.0), (1.4, 1.5, 0.25, 0.2), (1.4, 1.5, -0.2, 0.2)):
            base = np.array([1.0, x[peak], width_s, skew_s, y[peak], width_t, skew_t], dtype=float)
            if inseparable:
                base = np.r_[base, 0.0]
            starts.append(base)
    else:
        base = np.asarray(initial_from_r0, dtype=float)
        if len(base) != 7:
            raise ValueError('initial_from_r0 must contain seven parameters')
        starts = [np.r_[base, q] for q in (-0.4, 0.0, 0.25, 0.65, 1.0)]
    best = None
    for start in starts:
        start = np.minimum(np.maximum(start, lower + 1e-06), upper - 1e-06)
        result = least_squares(lambda p: yu_surface(p, x, y, inseparable=inseparable) - normalized, start, bounds=(lower, upper), max_nfev=2500, xtol=1e-10, ftol=1e-10, gtol=1e-10)
        candidate_rss = float(np.sum(np.square(result.fun)))
        if best is None or candidate_rss < best[0]:
            best = (candidate_rss, result)
    assert best is not None
    normalized_rss, result = best
    prediction = yu_surface(result.x, x, y, inseparable=inseparable) * scale
    rss = normalized_rss * scale ** 2
    n_parameters = 8 if inseparable else 7
    mse = rss / len(z)
    return SurfaceFit(model='R1' if inseparable else 'R0', parameters=np.r_[result.x[0] * scale, result.x[1:]], prediction=prediction, rss=rss, r2=coefficient_of_determination(z, prediction), bic_dense=bic_from_rss(rss, len(z), n_parameters), bic_yu36=float(YU_EFFECTIVE_CONDITIONS * np.log(max(mse, YU_FIT_EPS)) + n_parameters * np.log(YU_EFFECTIVE_CONDITIONS)), success=bool(result.success))
