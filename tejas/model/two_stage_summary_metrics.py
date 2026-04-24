import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from matplotlib.ticker import FuncFormatter
from torch.utils.data import DataLoader

from models.config_loader import load_dataset_configs
from two_stage_core import TwoStage
from util import get_dataset_from_config

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DEFAULT_RUNS_ROOT = Path("/home/tejas/VisionCore/tejas/model/final_runs")
DEFAULT_BATCH_SUMMARY = DEFAULT_RUNS_ROOT / "allen_batch_summary.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_ROOT / "summary_metrics"
DEFAULT_DATASET_CONFIGS = "/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_gaborium_20lags.yaml"
DATAYATES_ROOT = "/home/tejas/DataYatesV1"
EPS = 1e-12
MIN_R2_FULL_FOR_LI = 1e-3
_MODEL_CONFIGS_BY_SESSION = None
_PHASE_TUNING_CACHE = {}
_RF_CONTOUR_CACHE = {}
_PPD_CACHE = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load saved two-stage checkpoints and compute per-cell summary metrics "
            "(linearity fractions, linearity index, separability ratio) without retraining."
        )
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--batch-summary-csv", type=Path, default=DEFAULT_BATCH_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--session-prefix", type=str, default="Allen_")
    parser.add_argument("--session-names", type=str, default="")
    parser.add_argument("--max-cells-per-session", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    return parser.parse_args()


def resolve_session_dirs(runs_root, batch_summary_csv, session_prefix, session_names):
    if session_names:
        wanted = [x.strip() for x in str(session_names).split(",") if x.strip()]
        session_dirs = [Path(runs_root) / name for name in wanted]
        return [p for p in session_dirs if p.exists()]

    if Path(batch_summary_csv).exists():
        summary = pd.read_csv(batch_summary_csv)
        if "status" in summary.columns:
            summary = summary[summary["status"].astype(str).str.lower() == "ok"]
        if "session_name" in summary.columns:
            summary = summary[summary["session_name"].astype(str).str.startswith(str(session_prefix))]
        session_dirs = [Path(p) for p in summary["session_dir"].tolist()]
        return [p for p in session_dirs if p.exists()]

    session_dirs = sorted(Path(runs_root).glob(f"{session_prefix}*"))
    return [p for p in session_dirs if p.is_dir()]


def get_model_config_by_session():
    global _MODEL_CONFIGS_BY_SESSION
    if _MODEL_CONFIGS_BY_SESSION is None:
        configs = load_dataset_configs(DEFAULT_DATASET_CONFIGS)
        _MODEL_CONFIGS_BY_SESSION = {str(cfg["session"]): cfg for cfg in configs if "session" in cfg}
    return _MODEL_CONFIGS_BY_SESSION


def load_bps_map(session_dir):
    csv_path = Path(session_dir) / "best_val_bps.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if "cell_id" not in df.columns:
        return {}
    return {int(row["cell_id"]): row for _, row in df.iterrows()}


def parse_session_name(session_name):
    parts = str(session_name).split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Expected session name like Subject_YYYY-MM-DD, got: {session_name}")
    return parts[0], parts[1]


def get_model_true_cids_for_session(session_name):
    configs = get_model_config_by_session()
    if str(session_name) not in configs:
        raise KeyError(f"Session {session_name} not found in {DEFAULT_DATASET_CONFIGS}")
    cids = configs[str(session_name)].get("cids", None)
    if cids is None:
        raise KeyError(f"Session {session_name} config is missing cids.")
    return [int(x) for x in cids]


def get_gratings_true_cids_for_session(session_name):
    subject, date = parse_session_name(session_name)
    dataset_configs_path = "/mnt/sata/YatesMarmoV1/conv_model_fits/data_configs/multi_dataset_basic_for_metrics_gratings"
    yaml_files = [
        f
        for f in Path(dataset_configs_path).iterdir()
        if f.suffix == ".yaml" and "base" not in f.name and date in f.name and subject in f.name
    ]
    if not yaml_files:
        return None
    with yaml_files[0].open("r") as f:
        session_cfg = yaml.safe_load(f)
    cids = session_cfg.get("cids", None)
    if cids is None:
        return None
    return [int(x) for x in cids]


def get_ppd_for_session(session_name):
    if session_name in _PPD_CACHE:
        return _PPD_CACHE[session_name]
    if DATAYATES_ROOT not in sys.path:
        sys.path.insert(0, DATAYATES_ROOT)
    from DataYatesV1 import get_session  # noqa: WPS433

    subject, date = parse_session_name(session_name)
    sess = get_session(subject, date)
    ppd = float(sess.exp["S"]["pixPerDeg"])
    _PPD_CACHE[session_name] = ppd
    return ppd


def load_phase_tuning_map(session_name):
    if session_name in _PHASE_TUNING_CACHE:
        return _PHASE_TUNING_CACHE[session_name]

    if DATAYATES_ROOT not in sys.path:
        sys.path.insert(0, DATAYATES_ROOT)
    from DataYatesV1 import get_session  # noqa: WPS433

    subject, date = parse_session_name(session_name)
    sess = get_session(subject, date)
    true_cluster_ids = np.unique(sess.ks_results.spike_clusters)
    model_true_cids = get_model_true_cids_for_session(session_name)

    def _empty_phase_map():
        return {
            int(subset_idx): {
                "true_cid": int(true_cid),
                "gratings_unit_index": None,
                "phase_modulation_index": float("nan"),
                "phase_modulation_valid": False,
                "peak_orientation_deg": float("nan"),
                "peak_orientation_valid": False,
                "peak_sf_cpd": float("nan"),
                "peak_sf_valid": False,
            }
            for subset_idx, true_cid in enumerate(model_true_cids)
        }

    cache_path = Path(f"/mnt/ssd/YatesMarmoV1/metrics/gratings_analysis/{subject}_{date}/gratings_results.npz")
    try:
        if cache_path.exists():
            gratings_results = np.load(cache_path, allow_pickle=True)["gratings_results"].item()
        else:
            from DataYatesV1.models.config_loader import load_dataset_configs as dy_load_dataset_configs  # noqa: WPS433
            from DataYatesV1.utils.data import prepare_data as dy_prepare_data  # noqa: WPS433
            from jake.multidataset_ddp.gratings_analysis import gratings_analysis  # noqa: WPS433

            dataset_configs_path = "/mnt/sata/YatesMarmoV1/conv_model_fits/data_configs/multi_dataset_basic_for_metrics_gratings"
            yaml_files = [
                f
                for f in Path(dataset_configs_path).iterdir()
                if f.suffix == ".yaml" and "base" not in f.name and date in f.name and subject in f.name
            ]
            if not yaml_files:
                _PHASE_TUNING_CACHE[session_name] = _empty_phase_map()
                return _PHASE_TUNING_CACHE[session_name]
            dataset_configs = dy_load_dataset_configs([yaml_files[0].name], dataset_configs_path)
            train_dset, _, dataset_config = dy_prepare_data(dataset_configs[0], strict=False)
            train_dset_loaded = train_dset[:]
            robs = train_dset_loaded["robs"].numpy()
            sf = train_dset.dsets[0]["sf"][train_dset.inds[:, 1]]
            ori = train_dset.dsets[0]["ori"][train_dset.inds[:, 1]].numpy()
            phases = train_dset.dsets[0]["stim_phase"][train_dset.inds[:, 1]].numpy()
            dt = 1 / dataset_config["sampling"]["target_rate"]
            n_lags = dataset_config["keys_lags"]["stim"][-1]
            max_eye_movement = 10
            gratings_validity_filter = np.logical_and.reduce(
                [
                    np.abs(train_dset.dsets[0]["eyepos"][train_dset.inds[:, 1], 0]) < max_eye_movement,
                    np.abs(train_dset.dsets[0]["eyepos"][train_dset.inds[:, 1], 1]) < max_eye_movement,
                    train_dset.dsets[0]["dpi_valid"][train_dset.inds[:, 1]],
                ]
            ).astype(np.float32)
            gratings_validity_filter = gratings_validity_filter[:, None].repeat(robs.shape[1], axis=1)
            gratings_results = gratings_analysis(
                robs=robs,
                sf=sf,
                ori=ori,
                phases=phases,
                dt=dt,
                n_lags=n_lags,
                dfs=gratings_validity_filter,
                min_spikes=30,
            )
            gratings_results["phases"] = phases
            gratings_results["robs"] = robs
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, gratings_results=gratings_results)
    except Exception:
        _PHASE_TUNING_CACHE[session_name] = _empty_phase_map()
        return _PHASE_TUNING_CACHE[session_name]
    modulation_index = np.asarray(gratings_results["modulation_index"], dtype=float).reshape(-1)
    sine_fit_results = gratings_results["sine_fit_results"]
    peak_sf_idx = np.asarray(gratings_results.get("peak_sf_idx", []), dtype=int).reshape(-1)
    peak_ori_idx = np.asarray(gratings_results.get("peak_ori_idx", []), dtype=int).reshape(-1)
    sfs = np.asarray(gratings_results.get("sfs", []), dtype=float).reshape(-1)
    oris = np.asarray(gratings_results.get("oris", []), dtype=float).reshape(-1)
    expected_n = int(modulation_index.shape[0])
    gratings_true_cids = get_gratings_true_cids_for_session(session_name)
    if gratings_true_cids is not None and len(gratings_true_cids) == expected_n:
        cid_to_full_idx = {int(cid): int(i) for i, cid in enumerate(gratings_true_cids)}
    elif len(true_cluster_ids) == expected_n:
        cid_to_full_idx = {int(cid): int(i) for i, cid in enumerate(true_cluster_ids.tolist())}
    else:
        _PHASE_TUNING_CACHE[session_name] = _empty_phase_map()
        return _PHASE_TUNING_CACHE[session_name]
    if peak_sf_idx.shape[0] != expected_n or peak_ori_idx.shape[0] != expected_n:
        _PHASE_TUNING_CACHE[session_name] = _empty_phase_map()
        return _PHASE_TUNING_CACHE[session_name]

    phase_map = {}
    for subset_idx, true_cid in enumerate(model_true_cids):
        full_idx = cid_to_full_idx.get(int(true_cid), None)
        fit = None if full_idx is None else sine_fit_results[int(full_idx)]
        modulation = float("nan") if full_idx is None else float(modulation_index[int(full_idx)])
        peak_ori = float("nan")
        peak_sf = float("nan")
        peak_ori_valid = False
        peak_sf_valid = False
        if full_idx is not None:
            sf_idx = int(peak_sf_idx[int(full_idx)])
            ori_idx = int(peak_ori_idx[int(full_idx)])
            if 0 <= sf_idx < len(sfs):
                peak_sf = float(sfs[sf_idx])
                peak_sf_valid = np.isfinite(peak_sf)
            if 0 <= ori_idx < len(oris):
                peak_ori = float(oris[ori_idx])
                peak_ori_valid = np.isfinite(peak_ori)
        phase_map[int(subset_idx)] = {
            "true_cid": int(true_cid),
            "gratings_unit_index": (None if full_idx is None else int(full_idx)),
            "phase_modulation_index": modulation,
            "phase_modulation_valid": bool(
                full_idx is not None and fit is not None and np.isfinite(modulation)
            ),
            "peak_orientation_deg": peak_ori,
            "peak_orientation_valid": bool(full_idx is not None and peak_ori_valid),
            "peak_sf_cpd": peak_sf,
            "peak_sf_valid": bool(full_idx is not None and peak_sf_valid),
        }

    _PHASE_TUNING_CACHE[session_name] = phase_map
    return phase_map


def load_rf_contour_map(session_name):
    if session_name in _RF_CONTOUR_CACHE:
        return _RF_CONTOUR_CACHE[session_name]

    if DATAYATES_ROOT not in sys.path:
        sys.path.insert(0, DATAYATES_ROOT)
    from DataYatesV1 import get_session  # noqa: WPS433
    from tejas.metrics.gaborium import get_rf_contour_metrics  # noqa: WPS433

    subject, date = parse_session_name(session_name)
    sess = get_session(subject, date)
    true_cluster_ids = np.unique(sess.ks_results.spike_clusters)
    cid_to_full_idx = {int(cid): int(i) for i, cid in enumerate(true_cluster_ids.tolist())}
    model_true_cids = get_model_true_cids_for_session(session_name)

    def _empty_rf_map():
        return {
            int(subset_idx): {
                "data_sqrt_area_contour_deg": float("nan"),
                "data_area_contour_deg2": float("nan"),
                "data_rf_contour_valid": False,
            }
            for subset_idx, _ in enumerate(model_true_cids)
        }

    try:
        contour_metrics = get_rf_contour_metrics(date, subject)
    except Exception:
        _RF_CONTOUR_CACHE[session_name] = _empty_rf_map()
        return _RF_CONTOUR_CACHE[session_name]

    rf_map = {}
    for subset_idx, true_cid in enumerate(model_true_cids):
        full_idx = cid_to_full_idx.get(int(true_cid), None)
        metric = {} if full_idx is None else contour_metrics.get(int(full_idx), {})
        sqrt_area = float(metric.get("sqrt_area_contour_deg", float("nan")))
        area_deg2 = float(metric.get("area_contour_deg2", float("nan")))
        valid = bool(metric.get("valid", False) and np.isfinite(sqrt_area))
        rf_map[int(subset_idx)] = {
            "data_sqrt_area_contour_deg": sqrt_area,
            "data_area_contour_deg2": area_deg2,
            "data_rf_contour_valid": valid,
        }

    _RF_CONTOUR_CACHE[session_name] = rf_map
    return rf_map


def contour_area_px2_from_map(image_2d):
    image = np.asarray(image_2d, dtype=float)
    if image.ndim != 2 or not np.isfinite(image).any():
        return float("nan")
    ptp = float(np.ptp(image))
    if ptp > 0:
        norm = (image - float(np.min(image))) / ptp
    else:
        norm = image.copy()
    if not np.isfinite(norm).all() or float(np.var(norm)) <= 1e-6:
        return float("nan")
    from skimage import measure  # noqa: WPS433

    contours = measure.find_contours(norm, 0.5)
    if not contours:
        return float("nan")
    contour = sorted(contours, key=len)[-1]
    x = contour[:, 0]
    y = contour[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area)


def model_band_rf_size_metrics(w_pos_map, w_neg_map, ppd):
    amp = np.sqrt(np.asarray(w_pos_map, dtype=float) ** 2 + np.asarray(w_neg_map, dtype=float) ** 2)
    sqrt_area_vals_deg = []
    area_vals_deg2 = []
    for scale_idx in range(int(amp.shape[0])):
        for orient_idx in range(int(amp.shape[1])):
            area_px2 = contour_area_px2_from_map(amp[scale_idx, orient_idx])
            if np.isfinite(area_px2):
                area_vals_deg2.append(float(area_px2 / (float(ppd) ** 2)))
                sqrt_area_vals_deg.append(float(np.sqrt(area_px2) / float(ppd)))
    vals = np.asarray(sqrt_area_vals_deg, dtype=float)
    area_vals = np.asarray(area_vals_deg2, dtype=float)
    if vals.size == 0:
        return {
            "model_band_sqrt_area_contour_mean_deg": float("nan"),
            "model_band_sqrt_area_contour_median_deg": float("nan"),
            "model_band_sqrt_area_contour_max_deg": float("nan"),
            "model_band_area_contour_mean_deg2": float("nan"),
            "model_band_area_contour_median_deg2": float("nan"),
            "model_band_area_contour_max_deg2": float("nan"),
            "model_band_contour_valid_count": 0,
        }
    return {
        "model_band_sqrt_area_contour_mean_deg": float(np.nanmean(vals)),
        "model_band_sqrt_area_contour_median_deg": float(np.nanmedian(vals)),
        "model_band_sqrt_area_contour_max_deg": float(np.nanmax(vals)),
        "model_band_area_contour_mean_deg2": float(np.nanmean(area_vals)),
        "model_band_area_contour_median_deg2": float(np.nanmedian(area_vals)),
        "model_band_area_contour_max_deg2": float(np.nanmax(area_vals)),
        "model_band_contour_valid_count": int(vals.size),
    }


def crop_slice(crop_size):
    return slice(int(crop_size), -int(crop_size)) if int(crop_size) > 0 else slice(None)


def build_session_eval_context(session_name, image_shape, batch_size, num_workers, device):
    subject, date = parse_session_name(session_name)
    _, val_dset, _ = get_dataset_from_config(
        subject=subject,
        date=date,
        dataset_configs_path=DEFAULT_DATASET_CONFIGS,
    )
    val_dset.to("cpu")
    sample = val_dset[0]
    stim = sample["stim"]
    if stim.ndim not in {4, 5}:
        raise ValueError(f"Unexpected stim ndim={stim.ndim} for session {session_name}")
    stim_hw = int(stim.shape[-1])
    crop_size = max((stim_hw - int(image_shape[-1])) // 2, 0)
    loader = DataLoader(
        val_dset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=str(device).startswith("cuda"),
    )
    return {"crop_size": crop_size, "val_loader": loader}


def extract_windowed_maps_and_flat_weights(ckpt):
    kwargs = dict(ckpt["model_kwargs"])
    state = ckpt["model_state"]
    n_neurons = int(kwargs["n_neurons"])
    n_lags = int(kwargs["n_lags"])
    n_scales = int(kwargs["height"])
    n_orientations = int(kwargs["order"]) + 1
    image_shape = tuple(int(x) for x in kwargs["image_shape"])
    neuron_idx = int(ckpt.get("neuron_idx", 0))
    lag_idx = 0 if n_lags == 1 else int(ckpt.get("lag_index", 0))

    w_pos_flat = state["w_pos.weight"].detach().to(dtype=torch.float32, device="cpu")[neuron_idx]
    w_neg_flat = state["w_neg.weight"].detach().to(dtype=torch.float32, device="cpu")[neuron_idx]
    hann_flat = state.get("hann_flat", None)
    if hann_flat is not None:
        hann_flat = hann_flat.detach().to(dtype=torch.float32, device="cpu")
        w_pos_win_flat = w_pos_flat * hann_flat
        w_neg_win_flat = w_neg_flat * hann_flat
    else:
        w_pos_win_flat = w_pos_flat
        w_neg_win_flat = w_neg_flat

    shape = (n_neurons, n_lags, n_scales, n_orientations, *image_shape)
    w_pos_map = w_pos_win_flat.reshape(1, n_lags, n_scales, n_orientations, *image_shape)[0, lag_idx]
    w_neg_map = w_neg_win_flat.reshape(1, n_lags, n_scales, n_orientations, *image_shape)[0, lag_idx]
    return w_pos_map.to(torch.float64), w_neg_map.to(torch.float64), w_pos_win_flat, w_neg_win_flat


def linear_energy_components(w_pos, w_neg):
    inv_sqrt2 = float(2.0 ** -0.5)
    return {"w_linear": (w_pos - w_neg) * inv_sqrt2, "w_energy": (w_pos + w_neg) * inv_sqrt2}


def separable_from_marginals(connectivity):
    scale_margin = connectivity.sum(dim=(1, 2, 3), keepdim=False)
    orient_margin = connectivity.sum(dim=(0, 2, 3), keepdim=False)
    spatial_margin = connectivity.sum(dim=(0, 1), keepdim=False)
    total = float(connectivity.sum().item())
    if total <= EPS:
        return torch.zeros_like(connectivity)
    return (
        scale_margin[:, None, None, None]
        * orient_margin[None, :, None, None]
        * spatial_margin[None, None, :, :]
    ) / (total ** 2)


def explained_variance(target, approx):
    y = target.reshape(-1)
    yhat = approx.reshape(-1)
    sst = torch.sum((y - y.mean()) ** 2)
    sst_value = float(sst.item())
    if sst_value <= EPS:
        return float("nan"), 0.0
    sse = torch.sum((y - yhat) ** 2)
    return float((1.0 - sse / sst).item()), sst_value


def separability_metrics(component_map):
    connectivity = component_map.square()
    separable = separable_from_marginals(connectivity)
    return explained_variance(connectivity, separable)


def total_connectivity_map_from_maps(w_pos_map, w_neg_map):
    return (w_pos_map.square() + w_neg_map.square()).to(torch.float64)


def weighted_log_stats(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return float("nan"), float("nan")
    v = values[mask]
    w = weights[mask]
    wsum = max(float(w.sum()), EPS)
    mean_v = float(np.sum(w * v) / wsum)
    var_v = float(np.sum(w * (v - mean_v) ** 2) / wsum)
    return mean_v, float(np.sqrt(max(var_v, 0.0)))


def weighted_orientation_stats_deg(angles_deg, weights):
    angles_deg = np.asarray(angles_deg, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(angles_deg) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return float("nan"), float("nan")
    theta = np.deg2rad(angles_deg[mask])
    w = weights[mask]
    wsum = max(float(w.sum()), EPS)
    c = float(np.sum(w * np.cos(2.0 * theta)) / wsum)
    s = float(np.sum(w * np.sin(2.0 * theta)) / wsum)
    mean_theta = 0.5 * np.arctan2(s, c)
    if mean_theta < 0:
        mean_theta += np.pi
    r = np.sqrt(max(c * c + s * s, 0.0))
    if r <= 0:
        std_theta = np.pi / np.sqrt(12.0)
    else:
        std_theta = 0.5 * np.sqrt(max(-2.0 * np.log(min(r, 1.0)), 0.0))
    return float(np.rad2deg(mean_theta)), float(np.rad2deg(std_theta))


def spectral_bandwidth_metrics(connectivity_map, orientation_degrees, scale_cpds):
    orient_weights = connectivity_map.sum(dim=(0, 2, 3)).detach().cpu().numpy()
    scale_weights = connectivity_map.sum(dim=(1, 2, 3)).detach().cpu().numpy()
    scale_cpds = np.asarray(scale_cpds, dtype=float)
    orientation_degrees = np.asarray(orientation_degrees, dtype=float)
    log_scale_cpds = np.log2(np.clip(scale_cpds, a_min=EPS, a_max=None))
    sf_centroid_log2, sf_bandwidth_oct = weighted_log_stats(log_scale_cpds, scale_weights)
    ori_centroid_deg, ori_bandwidth_deg = weighted_orientation_stats_deg(orientation_degrees, orient_weights)
    return {
        "orientation_centroid_deg": float(ori_centroid_deg),
        "orientation_bandwidth_deg": float(ori_bandwidth_deg),
        "sf_centroid_log2_cpd": float(sf_centroid_log2),
        "sf_centroid_cpd": float(2.0 ** sf_centroid_log2) if np.isfinite(sf_centroid_log2) else float("nan"),
        "sf_bandwidth_octaves": float(sf_bandwidth_oct),
    }


def init_running_stats(n_cells, device):
    z = torch.zeros(int(n_cells), dtype=torch.float64, device=device)
    return {"sw": z.clone(), "sx": z.clone(), "sy": z.clone(), "sxx": z.clone(), "syy": z.clone(), "sxy": z.clone()}


def update_running_stats(stats, pred, target, mask):
    w = mask.to(dtype=torch.float64)
    x = pred.to(dtype=torch.float64)
    y = target.to(dtype=torch.float64)
    stats["sw"] += w.sum(dim=0)
    stats["sx"] += (w * x).sum(dim=0)
    stats["sy"] += (w * y).sum(dim=0)
    stats["sxx"] += (w * x * x).sum(dim=0)
    stats["syy"] += (w * y * y).sum(dim=0)
    stats["sxy"] += (w * x * y).sum(dim=0)


def finalize_running_r2(stats):
    sw = stats["sw"].clamp_min(EPS)
    mx = stats["sx"] / sw
    my = stats["sy"] / sw
    cov = stats["sxy"] / sw - (mx * my)
    vx = stats["sxx"] / sw - (mx * mx)
    vy = stats["syy"] / sw - (my * my)
    r2 = (cov * cov) / (vx * vy + EPS)
    return torch.clamp(r2, min=0.0, max=1.0)


def load_session_checkpoint_records(session_dir, max_cells_per_session):
    checkpoints_dir = Path(session_dir) / "checkpoints"
    if not checkpoints_dir.exists():
        return [], "missing checkpoints_dir"
    ckpt_paths = sorted(checkpoints_dir.glob("cell_*_best.pt"))
    if max_cells_per_session and max_cells_per_session > 0:
        ckpt_paths = ckpt_paths[: int(max_cells_per_session)]
    if not ckpt_paths:
        return [], "no checkpoint files"

    bps_map = load_bps_map(session_dir)
    phase_map = load_phase_tuning_map(Path(session_dir).name)
    rf_contour_map = load_rf_contour_map(Path(session_dir).name)
    ppd = get_ppd_for_session(Path(session_dir).name)
    reference_model = None
    orientation_degrees = None
    scale_cpds = None
    records = []
    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cell_id = int(ckpt["cell_id"])
        lag_index = int(ckpt.get("lag_index", 0))
        neuron_idx = int(ckpt.get("neuron_idx", 0))
        kwargs = dict(ckpt["model_kwargs"])
        if reference_model is None:
            reference_model = TwoStage(**kwargs)
            orientation_degrees = np.asarray(
                getattr(reference_model, "orientation_display_degrees", reference_model.orientation_degrees),
                dtype=float,
            )
            scale_cpds = np.asarray(getattr(reference_model, "used_scale_cpd", []), dtype=float)
        output_nonlinearity = str(kwargs.get("output_nonlinearity", "exp")).strip().lower()
        beta = float(ckpt["model_state"]["beta"].detach().reshape(-1)[neuron_idx].item())
        w_pos_map, w_neg_map, w_pos_win_flat, w_neg_win_flat = extract_windowed_maps_and_flat_weights(ckpt)
        model_rf_metrics = model_band_rf_size_metrics(
            w_pos_map.detach().cpu().numpy(),
            w_neg_map.detach().cpu().numpy(),
            ppd=ppd,
        )
        comp = linear_energy_components(w_pos_map, w_neg_map)
        total_connectivity = total_connectivity_map_from_maps(w_pos_map, w_neg_map)
        spectral_metrics = spectral_bandwidth_metrics(
            connectivity_map=total_connectivity,
            orientation_degrees=orientation_degrees[: total_connectivity.shape[1]],
            scale_cpds=scale_cpds[: total_connectivity.shape[0]],
        )
        sr_linear, sst_linear = separability_metrics(comp["w_linear"])
        sr_energy, sst_energy = separability_metrics(comp["w_energy"])
        combined_den = sst_linear + sst_energy
        separability_ratio = (
            float((sr_linear * sst_linear + sr_energy * sst_energy) / combined_den)
            if combined_den > EPS
            else float("nan")
        )

        row = {
            "session_name": str(Path(session_dir).name),
            "cell_id": int(cell_id),
            "neuron_idx": int(neuron_idx),
            "lag_index": int(lag_index),
            "checkpoint_path": str(ckpt_path),
            "best_train_bps": float(ckpt.get("best_train_bps", float("nan"))),
            "best_val_bps": float(ckpt.get("best_val_bps", float("nan"))),
            "separability_ratio": float(separability_ratio),
            "separability_ratio_linear": float(sr_linear),
            "separability_ratio_energy": float(sr_energy),
            "orientation_centroid_deg": spectral_metrics["orientation_centroid_deg"],
            "orientation_bandwidth_deg": spectral_metrics["orientation_bandwidth_deg"],
            "sf_centroid_log2_cpd": spectral_metrics["sf_centroid_log2_cpd"],
            "sf_centroid_cpd": spectral_metrics["sf_centroid_cpd"],
            "sf_bandwidth_octaves": spectral_metrics["sf_bandwidth_octaves"],
            "true_cid": int(phase_map.get(int(cell_id), {}).get("true_cid", -1)),
            "gratings_unit_index": phase_map.get(int(cell_id), {}).get("gratings_unit_index", None),
            "phase_modulation_index": float(
                phase_map.get(int(cell_id), {}).get("phase_modulation_index", float("nan"))
            ),
            "phase_modulation_valid": bool(
                phase_map.get(int(cell_id), {}).get("phase_modulation_valid", False)
            ),
            "peak_orientation_deg": float(
                phase_map.get(int(cell_id), {}).get("peak_orientation_deg", float("nan"))
            ),
            "peak_orientation_valid": bool(
                phase_map.get(int(cell_id), {}).get("peak_orientation_valid", False)
            ),
            "peak_sf_cpd": float(
                phase_map.get(int(cell_id), {}).get("peak_sf_cpd", float("nan"))
            ),
            "peak_sf_valid": bool(
                phase_map.get(int(cell_id), {}).get("peak_sf_valid", False)
            ),
            "data_sqrt_area_contour_deg": float(
                rf_contour_map.get(int(cell_id), {}).get("data_sqrt_area_contour_deg", float("nan"))
            ),
            "data_area_contour_deg2": float(
                rf_contour_map.get(int(cell_id), {}).get("data_area_contour_deg2", float("nan"))
            ),
            "data_rf_contour_valid": bool(
                rf_contour_map.get(int(cell_id), {}).get("data_rf_contour_valid", False)
            ),
            **model_rf_metrics,
            "output_nonlinearity": output_nonlinearity,
            "model_kwargs": kwargs,
            "beta": beta,
            "w_pos_win_flat": w_pos_win_flat,
            "w_neg_win_flat": w_neg_win_flat,
        }
        bps_row = bps_map.get(cell_id, None)
        if bps_row is not None:
            for col, val in bps_row.items():
                if col not in row:
                    row[col] = val
        records.append(row)
    del reference_model
    return records, None


def build_feature_model(model_kwargs, device):
    feature_model = TwoStage(**model_kwargs).to(device)
    feature_model.eval()
    return feature_model


def evaluate_session_response_metrics(records, session_ctx, device):
    if not records:
        return {}

    reference_kwargs = dict(records[0]["model_kwargs"])
    feature_model = build_feature_model(reference_kwargs, device=device)
    inv_sqrt2 = float(2.0 ** -0.5)
    ys = crop_slice(session_ctx["crop_size"])
    xs = crop_slice(session_ctx["crop_size"])

    groups = defaultdict(list)
    for rec in records:
        groups[(int(rec["lag_index"]), str(rec["output_nonlinearity"]))].append(rec)

    group_state = {}
    for key, recs in groups.items():
        w_pos = torch.stack([r["w_pos_win_flat"] for r in recs], dim=0).to(device=device, dtype=torch.float32)
        w_neg = torch.stack([r["w_neg_win_flat"] for r in recs], dim=0).to(device=device, dtype=torch.float32)
        beta = torch.tensor([r["beta"] for r in recs], device=device, dtype=torch.float32).unsqueeze(0)
        cell_ids = [int(r["cell_id"]) for r in recs]
        group_state[key] = {
            "records": recs,
            "w_pos": w_pos,
            "w_neg": w_neg,
            "w_lin": inv_sqrt2 * (w_pos - w_neg),
            "w_eng": inv_sqrt2 * (w_pos + w_neg),
            "beta": beta,
            "cell_ids": cell_ids,
            "stats_full": init_running_stats(len(recs), device=device),
            "stats_lin": init_running_stats(len(recs), device=device),
            "stats_eng": init_running_stats(len(recs), device=device),
        }

    with torch.no_grad():
        for batch in session_ctx["val_loader"]:
            batch_gpu = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            for (lag_index, output_nonlinearity), state in group_state.items():
                stim_lag = batch_gpu["stim"][:, :, [int(lag_index)], ys, xs]
                pos_feats, neg_feats, _ = feature_model.get_pyr_feats({"stim": stim_lag})
                z_full = F.linear(pos_feats, state["w_pos"]) + F.linear(neg_feats, state["w_neg"])
                z_lin = F.linear(pos_feats, state["w_lin"]) - F.linear(neg_feats, state["w_lin"])
                z_eng = F.linear(pos_feats, state["w_eng"]) + F.linear(neg_feats, state["w_eng"])
                if output_nonlinearity == "exp":
                    pred_full = torch.exp(state["beta"] + z_full)
                    pred_lin = torch.exp(state["beta"] + z_lin)
                    pred_eng = torch.exp(state["beta"] + z_eng)
                elif output_nonlinearity == "relu":
                    pred_full = (state["beta"] + F.relu(z_full)).clamp_min(1e-6)
                    pred_lin = (state["beta"] + F.relu(z_lin)).clamp_min(1e-6)
                    pred_eng = (state["beta"] + F.relu(z_eng)).clamp_min(1e-6)
                else:
                    raise ValueError(f"Unsupported output_nonlinearity: {output_nonlinearity}")

                robs = batch_gpu["robs"][:, state["cell_ids"]]
                dfs = batch_gpu["dfs"][:, state["cell_ids"]]
                update_running_stats(state["stats_full"], pred_full, robs, dfs)
                update_running_stats(state["stats_lin"], pred_lin, robs, dfs)
                update_running_stats(state["stats_eng"], pred_eng, robs, dfs)

    response_metrics = {}
    for state in group_state.values():
        r2_full = finalize_running_r2(state["stats_full"]).detach().cpu().numpy().reshape(-1)
        r2_lin = finalize_running_r2(state["stats_lin"]).detach().cpu().numpy().reshape(-1)
        r2_eng = finalize_running_r2(state["stats_eng"]).detach().cpu().numpy().reshape(-1)
        for idx, rec in enumerate(state["records"]):
            full = float(r2_full[idx])
            lin = float(r2_lin[idx])
            eng = float(r2_eng[idx])
            response_metrics[int(rec["cell_id"])] = {
                "r2_full": full,
                "r2_linear": lin,
                "r2_energy": eng,
                "frac_linear": lin / max(full, EPS),
                "frac_energy": eng / max(full, EPS),
            }

    del feature_model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return response_metrics


def build_rows_for_session(records, response_metrics):
    rows = []
    for rec in records:
        cell_id = int(rec["cell_id"])
        resp = response_metrics.get(cell_id, {})
        rows.append(
            {
                "session_name": rec["session_name"],
                "cell_id": cell_id,
                "neuron_idx": rec["neuron_idx"],
                "lag_index": rec["lag_index"],
                "checkpoint_path": rec["checkpoint_path"],
                "best_train_bps": rec["best_train_bps"],
                "best_val_bps": rec["best_val_bps"],
                "frac_linear": float(resp.get("frac_linear", float("nan"))),
                "frac_energy": float(resp.get("frac_energy", float("nan"))),
                "linearity_index": float(resp.get("frac_linear", float("nan")) - resp.get("frac_energy", float("nan"))),
                "r2_full": float(resp.get("r2_full", float("nan"))),
                "r2_linear": float(resp.get("r2_linear", float("nan"))),
                "r2_energy": float(resp.get("r2_energy", float("nan"))),
                "linearity_valid": bool(
                    np.isfinite(resp.get("r2_full", float("nan")))
                    and float(resp.get("r2_full", float("nan"))) >= float(MIN_R2_FULL_FOR_LI)
                ),
                "separability_ratio": rec["separability_ratio"],
                "separability_ratio_linear": rec["separability_ratio_linear"],
                "separability_ratio_energy": rec["separability_ratio_energy"],
                "orientation_centroid_deg": rec["orientation_centroid_deg"],
                "orientation_bandwidth_deg": rec["orientation_bandwidth_deg"],
                "sf_centroid_log2_cpd": rec["sf_centroid_log2_cpd"],
                "sf_centroid_cpd": rec["sf_centroid_cpd"],
                "sf_bandwidth_octaves": rec["sf_bandwidth_octaves"],
                "true_cid": rec["true_cid"],
                "gratings_unit_index": rec["gratings_unit_index"],
                "phase_modulation_index": rec["phase_modulation_index"],
                "phase_modulation_valid": rec["phase_modulation_valid"],
                "peak_orientation_deg": rec["peak_orientation_deg"],
                "peak_orientation_valid": rec["peak_orientation_valid"],
                "peak_sf_cpd": rec["peak_sf_cpd"],
                "peak_sf_valid": rec["peak_sf_valid"],
                "data_sqrt_area_contour_deg": rec["data_sqrt_area_contour_deg"],
                "data_area_contour_deg2": rec["data_area_contour_deg2"],
                "data_rf_contour_valid": rec["data_rf_contour_valid"],
                "model_band_sqrt_area_contour_mean_deg": rec["model_band_sqrt_area_contour_mean_deg"],
                "model_band_sqrt_area_contour_median_deg": rec["model_band_sqrt_area_contour_median_deg"],
                "model_band_sqrt_area_contour_max_deg": rec["model_band_sqrt_area_contour_max_deg"],
                "model_band_area_contour_mean_deg2": rec["model_band_area_contour_mean_deg2"],
                "model_band_area_contour_median_deg2": rec["model_band_area_contour_median_deg2"],
                "model_band_area_contour_max_deg2": rec["model_band_area_contour_max_deg2"],
                "model_band_contour_valid_count": rec["model_band_contour_valid_count"],
                **{
                    k: v
                    for k, v in rec.items()
                    if k
                    in {
                        "train_bps_at_best",
                        "best_lag_group",
                        "best_epoch",
                        "best_prox_mult",
                        "best_local_mult",
                    }
                },
            }
        )
    return rows


def format_progress(done, total, width=28):
    total = max(int(total), 1)
    done = max(0, min(int(done), total))
    filled = int(round(width * done / total))
    return f"[{'#' * filled}{'.' * (width - filled)}] {done}/{total}"


def plot_separability_vs_linearity(df, output_path):
    plot_df = df[
        np.isfinite(df["linearity_index"])
        & np.isfinite(df["separability_ratio"])
        & df["linearity_valid"].astype(bool)
    ].copy()
    if plot_df.empty:
        return
    x = plot_df["linearity_index"].to_numpy(dtype=float)
    y = plot_df["separability_ratio"].to_numpy(dtype=float)
    y_plot = np.clip(y, 0.0, 1.0)

    fig = plt.figure(figsize=(6.2, 5.6))
    grid = fig.add_gridspec(
        4,
        4,
        left=0.12,
        right=0.92,
        bottom=0.12,
        top=0.94,
        wspace=0.06,
        hspace=0.06,
    )
    ax = fig.add_subplot(grid[1:, :3])
    ax_histx = fig.add_subplot(grid[0, :3], sharex=ax)
    ax_histy = fig.add_subplot(grid[1:, 3], sharey=ax)

    color = "#3cb44b"
    ax.scatter(x, y_plot, s=22, c=color, alpha=0.9, edgecolors="none")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Linearity index")
    ax.set_ylabel("Separability ratio")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    bins_x = np.linspace(-1.0, 1.0, 17)
    bins_y = np.linspace(0.0, 1.0, 17)
    ax_histx.hist(x, bins=bins_x, color=color, alpha=0.95)
    ax_histy.hist(y_plot, bins=bins_y, orientation="horizontal", color=color, alpha=0.95)
    ax_histx.axis("off")
    ax_histy.axis("off")
    fig.suptitle(f"All sessions: separability vs linearity (n={len(plot_df)})", fontsize=11)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_linear_vs_energy(df, output_path):
    plot_df = df[np.isfinite(df["frac_linear"]) & np.isfinite(df["frac_energy"])].copy()
    if plot_df.empty:
        return
    x = plot_df["frac_linear"].to_numpy(dtype=float)
    y = plot_df["frac_energy"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(5.1, 5.1))
    color = "#3777c8"
    ax.scatter(x, y, s=22, c=color, alpha=0.9, edgecolors="none")
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1.0, alpha=0.7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Fraction of variance explained by linear term")
    ax.set_ylabel("Fraction of variance explained by energy term")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"All sessions: linear vs energy fractions (n={len(plot_df)})", fontsize=11)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_spectral_summary(df, output_path, well_fit_threshold=0.1):
    if "best_val_bps" not in df.columns:
        return
    plot_df = df[
        np.isfinite(df["sf_centroid_cpd"])
        & np.isfinite(df["sf_bandwidth_octaves"])
        & np.isfinite(df["orientation_centroid_deg"])
        & np.isfinite(df["orientation_bandwidth_deg"])
        & (df["best_val_bps"].astype(float) >= float(well_fit_threshold))
    ].copy()
    if plot_df.empty:
        return

    green = "#39b54a"
    # Match the paper's right-half log-polar layout: 90 at top, 0 at right,
    # then 150/120 descending along the lower-right arc.
    theta_mean_deg = np.mod(90.0 - plot_df["orientation_centroid_deg"].to_numpy(dtype=float), 180.0)
    theta_mean = np.deg2rad(theta_mean_deg)
    theta_std = np.deg2rad(plot_df["orientation_bandwidth_deg"].to_numpy(dtype=float))
    sf_mean_log = np.log2(plot_df["sf_centroid_cpd"].to_numpy(dtype=float))
    sf_std_log = plot_df["sf_bandwidth_octaves"].to_numpy(dtype=float)
    x_sf = plot_df["sf_centroid_cpd"].to_numpy(dtype=float)
    y_sf = plot_df["sf_bandwidth_octaves"].to_numpy(dtype=float)
    y_ori = plot_df["orientation_bandwidth_deg"].to_numpy(dtype=float)
    x_lo = max(1.0, np.floor(np.nanmin(x_sf) * 2.0) / 2.0 - 0.1)
    x_hi = np.ceil(np.nanmax(x_sf) * 2.0) / 2.0 + 0.3
    xticks = np.arange(np.ceil(x_lo * 2.0) / 2.0, x_hi + 0.001, 0.5)

    fig = plt.figure(figsize=(7.0, 5.8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], hspace=0.18, wspace=0.18)
    ax_polar = fig.add_subplot(grid[:, 0], projection="polar")
    ax_sf = fig.add_subplot(grid[0, 1])
    ax_ori = fig.add_subplot(grid[1, 1], sharex=ax_sf)

    tt = np.linspace(0.0, 2.0 * np.pi, 200)
    # Draw smaller contours first so larger ellipses do not completely wash out the panel.
    order = np.argsort(sf_std_log * np.maximum(theta_std, 1e-6))
    for idx in order:
        th0 = theta_mean[idx]
        ths = theta_std[idx]
        r0 = sf_mean_log[idx]
        rs = sf_std_log[idx]
        th = np.mod(th0 + ths * np.cos(tt), np.pi)
        rr = np.clip(r0 + rs * np.sin(tt), a_min=np.log2(0.95), a_max=np.log2(10.0))
        # Outline-only reads much better than filled contours at this population size.
        ax_polar.plot(th, rr, color=green, alpha=0.10, linewidth=0.7)

    rticks_vals = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0], dtype=float)
    rticks = np.log2(rticks_vals)
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_thetamin(0)
    ax_polar.set_thetamax(180)
    ax_polar.set_thetagrids([0, 30, 60, 90, 120, 150], labels=["90", "60", "30", "0", "150", "120"])
    ax_polar.set_ylim(np.log2(0.95), np.log2(10.0))
    ax_polar.set_yticks(rticks)
    ax_polar.set_yticklabels([f"{v:g}" for v in rticks_vals], fontsize=9)
    ax_polar.set_rlabel_position(180)
    ax_polar.grid(alpha=0.65, linewidth=0.9)
    ax_polar.set_title("a  Allen population", loc="left", fontsize=12, fontweight="bold")
    ax_polar.text(0.5, -0.12, "Orientation (deg)", transform=ax_polar.transAxes, ha="center", va="top")
    ax_polar.text(-0.07, 0.02, "Spatial frequency (c/deg)", transform=ax_polar.transAxes, rotation=90, ha="center", va="bottom")

    ax_sf.scatter(x_sf, y_sf, s=34, color=green, alpha=0.95, edgecolors="none")
    ax_ori.scatter(x_sf, y_ori, s=34, color=green, alpha=0.95, edgecolors="none")
    ax_sf.set_title("b", loc="left", fontsize=12, fontweight="bold")
    ax_ori.set_title("c", loc="left", fontsize=12, fontweight="bold")
    ax_sf.set_ylabel("Spatial frequency\nbandwidth (octaves)")
    ax_ori.set_ylabel("Orientation\nbandwidth (deg)")
    ax_ori.set_xlabel("Spatial frequency\ncentroid (c/deg)")
    formatter = FuncFormatter(lambda v, pos: f"{v:g}")
    for ax in (ax_sf, ax_ori):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xscale("linear")
        ax.set_xlim(x_lo, x_hi)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis="x", labelrotation=0)
    ax_sf.set_ylim(-0.02, max(0.9, np.nanmax(y_sf) * 1.08))
    ax_ori.set_ylim(max(0.0, np.nanmin(y_ori) * 0.9), np.nanmax(y_ori) * 1.08)

    fig.suptitle(
        f"Spectral bandwidth of afferent connection weights (well-fit cells, n={len(plot_df)})",
        fontsize=11,
        y=0.98,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_spectral_summary_by_session(df, output_path, well_fit_threshold=0.1):
    if "best_val_bps" not in df.columns:
        return
    plot_df = df[
        np.isfinite(df["sf_centroid_cpd"])
        & np.isfinite(df["sf_bandwidth_octaves"])
        & np.isfinite(df["orientation_centroid_deg"])
        & np.isfinite(df["orientation_bandwidth_deg"])
        & (df["best_val_bps"].astype(float) >= float(well_fit_threshold))
    ].copy()
    if plot_df.empty or "session_name" not in plot_df.columns:
        return

    sessions = sorted(plot_df["session_name"].astype(str).unique().tolist())
    cmap = plt.get_cmap("tab20")
    session_to_color = {sess: cmap(i % cmap.N) for i, sess in enumerate(sessions)}

    fig = plt.figure(figsize=(8.4, 6.1))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], hspace=0.20, wspace=0.22)
    ax_polar = fig.add_subplot(grid[:, 0], projection="polar")
    ax_sf = fig.add_subplot(grid[0, 1])
    ax_ori = fig.add_subplot(grid[1, 1], sharex=ax_sf)

    tt = np.linspace(0.0, 2.0 * np.pi, 200)
    rticks_vals = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0], dtype=float)
    rticks = np.log2(rticks_vals)

    x_all = plot_df["sf_centroid_cpd"].to_numpy(dtype=float)
    x_lo = max(1.0, np.floor(np.nanmin(x_all) * 2.0) / 2.0 - 0.1)
    x_hi = np.ceil(np.nanmax(x_all) * 2.0) / 2.0 + 0.3
    xticks = np.arange(np.ceil(x_lo * 2.0) / 2.0, x_hi + 0.001, 0.5)
    formatter = FuncFormatter(lambda v, pos: f"{v:g}")

    for sess in sessions:
        sess_df = plot_df[plot_df["session_name"].astype(str) == sess]
        color = session_to_color[sess]
        theta_mean_deg = np.mod(90.0 - sess_df["orientation_centroid_deg"].to_numpy(dtype=float), 180.0)
        theta_mean = np.deg2rad(theta_mean_deg)
        theta_std = np.deg2rad(sess_df["orientation_bandwidth_deg"].to_numpy(dtype=float))
        sf_mean_log = np.log2(sess_df["sf_centroid_cpd"].to_numpy(dtype=float))
        sf_std_log = sess_df["sf_bandwidth_octaves"].to_numpy(dtype=float)

        order = np.argsort(sf_std_log * np.maximum(theta_std, 1e-6))
        for idx in order:
            th0 = theta_mean[idx]
            ths = theta_std[idx]
            r0 = sf_mean_log[idx]
            rs = sf_std_log[idx]
            th = np.mod(th0 + ths * np.cos(tt), np.pi)
            rr = np.clip(r0 + rs * np.sin(tt), a_min=np.log2(0.95), a_max=np.log2(10.0))
            ax_polar.plot(th, rr, color=color, alpha=0.14, linewidth=0.75)

        ax_sf.scatter(
            sess_df["sf_centroid_cpd"].to_numpy(dtype=float),
            sess_df["sf_bandwidth_octaves"].to_numpy(dtype=float),
            s=22,
            color=color,
            alpha=0.9,
            edgecolors="none",
            label=sess,
        )
        ax_ori.scatter(
            sess_df["sf_centroid_cpd"].to_numpy(dtype=float),
            sess_df["orientation_bandwidth_deg"].to_numpy(dtype=float),
            s=22,
            color=color,
            alpha=0.9,
            edgecolors="none",
        )

    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_thetamin(0)
    ax_polar.set_thetamax(180)
    ax_polar.set_thetagrids([0, 30, 60, 90, 120, 150], labels=["90", "60", "30", "0", "150", "120"])
    ax_polar.set_ylim(np.log2(0.95), np.log2(10.0))
    ax_polar.set_yticks(rticks)
    ax_polar.set_yticklabels([f"{v:g}" for v in rticks_vals], fontsize=9)
    ax_polar.set_rlabel_position(180)
    ax_polar.grid(alpha=0.65, linewidth=0.9)
    ax_polar.set_title("a  Session-colored", loc="left", fontsize=12, fontweight="bold")
    ax_polar.text(0.5, -0.12, "Orientation (deg)", transform=ax_polar.transAxes, ha="center", va="top")
    ax_polar.text(-0.07, 0.02, "Spatial frequency (c/deg)", transform=ax_polar.transAxes, rotation=90, ha="center", va="bottom")

    ax_sf.set_title("b", loc="left", fontsize=12, fontweight="bold")
    ax_ori.set_title("c", loc="left", fontsize=12, fontweight="bold")
    ax_sf.set_ylabel("Spatial frequency\nbandwidth (octaves)")
    ax_ori.set_ylabel("Orientation\nbandwidth (deg)")
    ax_ori.set_xlabel("Spatial frequency\ncentroid (c/deg)")
    for ax in (ax_sf, ax_ori):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xscale("linear")
        ax.set_xlim(x_lo, x_hi)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(formatter)
    ax_sf.set_ylim(-0.02, max(0.9, np.nanmax(plot_df["sf_bandwidth_octaves"].to_numpy(dtype=float)) * 1.08))
    ori_vals = plot_df["orientation_bandwidth_deg"].to_numpy(dtype=float)
    ax_ori.set_ylim(max(0.0, np.nanmin(ori_vals) * 0.9), np.nanmax(ori_vals) * 1.08)

    handles, labels = ax_sf.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False, fontsize=8)

    fig.suptitle(
        f"Spectral bandwidth by session color (well-fit cells, n={len(plot_df)})",
        fontsize=11,
        y=0.98,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_phase_modulation_vs_linearity(
    df,
    output_path,
    min_best_val_bps=None,
    title_prefix=None,
    min_linearity_index=None,
):
    if "phase_modulation_index" not in df.columns:
        return
    plot_df = df[
        np.isfinite(df["linearity_index"])
        & np.isfinite(df["phase_modulation_index"])
        & df["phase_modulation_valid"].astype(bool)
        & df["linearity_valid"].astype(bool)
    ].copy()
    if min_best_val_bps is not None:
        if "best_val_bps" not in plot_df.columns:
            return
        plot_df = plot_df[np.isfinite(plot_df["best_val_bps"]) & (plot_df["best_val_bps"] > float(min_best_val_bps))].copy()
    if min_linearity_index is not None:
        plot_df = plot_df[plot_df["linearity_index"].astype(float) >= float(min_linearity_index)].copy()
    if plot_df.empty:
        return

    x = plot_df["linearity_index"].to_numpy(dtype=float)
    y = plot_df["phase_modulation_index"].to_numpy(dtype=float)
    corr = float(np.corrcoef(x, y)[0, 1]) if len(plot_df) >= 2 else float("nan")
    title = "Phase modulation vs linearity"
    if title_prefix:
        title = f"{title_prefix} {title}".strip()
    if min_best_val_bps is not None:
        title = f"{title} (val BPS > {float(min_best_val_bps):g})"
    if min_linearity_index is not None:
        title = f"{title}, LI >= {float(min_linearity_index):g}"

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.scatter(x, y, s=24, color="#6a51a3", alpha=0.85, edgecolors="none")
    ax.set_xlabel("Linearity index")
    ax.set_ylabel("Phase modulation index")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"{title} (n={len(plot_df)}, r={corr:.3f})", fontsize=11)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_r2_component_diagnostics(df, output_path):
    cols = ["r2_full", "r2_linear", "r2_energy"]
    plot_df = df.copy()
    for col in cols:
        plot_df = plot_df[np.isfinite(plot_df[col])]
    if plot_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), sharey=True)
    bins = np.logspace(-8, 0, 50)
    colors = {
        "r2_full": "#444444",
        "r2_linear": "#3777c8",
        "r2_energy": "#39b54a",
    }
    titles = {
        "r2_full": "Full map",
        "r2_linear": "Linear component",
        "r2_energy": "Energy component",
    }
    for ax, col in zip(axes, cols):
        vals = plot_df[col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            continue
        ax.hist(vals, bins=bins, color=colors[col], alpha=0.9)
        ax.set_xscale("log")
        ax.set_title(titles[col], fontsize=11)
        ax.set_xlabel("r²")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Count")
    fig.suptitle("r² component diagnostics", fontsize=12)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_best_val_bps_histogram(df, output_path):
    if "best_val_bps" not in df.columns:
        return
    vals = df["best_val_bps"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.hist(vals, bins=50, color="#4c78a8", alpha=0.9)
    ax.axvline(float(np.nanmedian(vals)), color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xlabel("Best validation BPS")
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        f"Histogram of best validation BPS across all cells (n={vals.size}, median={np.nanmedian(vals):.3f})",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_peak_sf_vs_centroid(df, output_path, well_fit_threshold=0.1):
    if "best_val_bps" not in df.columns:
        return
    plot_df = df[
        np.isfinite(df["peak_sf_cpd"])
        & np.isfinite(df["sf_centroid_cpd"])
        & df["peak_sf_valid"].astype(bool)
        & (df["best_val_bps"].astype(float) >= float(well_fit_threshold))
    ].copy()
    if plot_df.empty:
        return

    x = plot_df["peak_sf_cpd"].to_numpy(dtype=float)
    y = plot_df["sf_centroid_cpd"].to_numpy(dtype=float)
    corr = float(np.corrcoef(x, y)[0, 1]) if len(plot_df) >= 2 else float("nan")
    lo = max(0.0, float(np.nanmin([x.min(), y.min()])) * 0.95)
    hi = float(np.nanmax([x.max(), y.max()])) * 1.05

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.scatter(x, y, s=24, color="#1f78b4", alpha=0.85, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.0, alpha=0.7)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Peak spatial frequency from gratings (c/deg)")
    ax.set_ylabel("Model spatial frequency centroid (c/deg)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        f"Peak spatial frequency vs centroid (well-fit cells, n={len(plot_df)}, r={corr:.3f})",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_peak_orientation_vs_centroid(df, output_path, well_fit_threshold=0.1):
    if "best_val_bps" not in df.columns:
        return
    plot_df = df[
        np.isfinite(df["peak_orientation_deg"])
        & np.isfinite(df["orientation_centroid_deg"])
        & df["peak_orientation_valid"].astype(bool)
        & (df["best_val_bps"].astype(float) >= float(well_fit_threshold))
    ].copy()
    if plot_df.empty:
        return

    x = np.mod(plot_df["peak_orientation_deg"].to_numpy(dtype=float), 180.0)
    y = np.mod(plot_df["orientation_centroid_deg"].to_numpy(dtype=float), 180.0)
    corr = float(np.corrcoef(x, y)[0, 1]) if len(plot_df) >= 2 else float("nan")

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.scatter(x, y, s=24, color="#e6550d", alpha=0.85, edgecolors="none")
    ax.plot([0, 180], [0, 180], "--", color="black", linewidth=1.0, alpha=0.7)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 180)
    ax.set_xlabel("Peak orientation from gratings (deg)")
    ax.set_ylabel("Model orientation centroid (deg)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        f"Peak orientation vs centroid (well-fit cells, n={len(plot_df)}, r={corr:.3f})",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rf_size_comparison(df, output_path, model_col, title_label, well_fit_threshold=0.1):
    if "best_val_bps" not in df.columns:
        return
    plot_df = df.copy()
    plot_df["data_sqrt_area_contour_deg_num"] = pd.to_numeric(plot_df["data_sqrt_area_contour_deg"], errors="coerce")
    plot_df[f"{model_col}_num"] = pd.to_numeric(plot_df[model_col], errors="coerce")
    plot_df["best_val_bps_num"] = pd.to_numeric(plot_df["best_val_bps"], errors="coerce")
    plot_df = plot_df[
        np.isfinite(plot_df["data_sqrt_area_contour_deg_num"])
        & np.isfinite(plot_df[f"{model_col}_num"])
        & plot_df["data_rf_contour_valid"].astype(bool)
        & (plot_df["best_val_bps_num"] >= float(well_fit_threshold))
    ].copy()
    if plot_df.empty:
        return

    x = plot_df["data_sqrt_area_contour_deg_num"].to_numpy(dtype=float)
    y = plot_df[f"{model_col}_num"].to_numpy(dtype=float)
    corr = float(np.corrcoef(x, y)[0, 1]) if len(plot_df) >= 2 else float("nan")
    lo = min(float(np.nanmin(x)), float(np.nanmin(y)))
    hi = max(float(np.nanmax(x)), float(np.nanmax(y)))
    pad = 0.05 * max(hi - lo, 1e-3)

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.scatter(x, y, s=24, color="#2ca25f", alpha=0.85, edgecolors="none")
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="black", linewidth=1.0, alpha=0.7)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Data STE sqrt contour area (deg)")
    ax.set_ylabel(f"Model sqrt contour area ({title_label}, deg)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        f"Data vs model RF size ({title_label}, well-fit cells, n={len(plot_df)}, r={corr:.3f})",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_by_session(df):
    if df.empty:
        return pd.DataFrame()
    grouped = (
        df.groupby("session_name", dropna=False)
        .agg(
            n_cells=("cell_id", "count"),
            mean_best_val_bps=("best_val_bps", "mean"),
            median_best_val_bps=("best_val_bps", "median"),
            mean_linearity_index=("linearity_index", "mean"),
            n_linearity_valid=("linearity_valid", "sum"),
            mean_separability_ratio=("separability_ratio", "mean"),
            mean_frac_linear=("frac_linear", "mean"),
            mean_frac_energy=("frac_energy", "mean"),
            mean_sf_centroid_cpd=("sf_centroid_cpd", "mean"),
            mean_sf_bandwidth_octaves=("sf_bandwidth_octaves", "mean"),
            mean_orientation_bandwidth_deg=("orientation_bandwidth_deg", "mean"),
            mean_phase_modulation_index=("phase_modulation_index", "mean"),
            n_phase_modulation_valid=("phase_modulation_valid", "sum"),
        )
        .reset_index()
        .sort_values("session_name")
        .reset_index(drop=True)
    )
    return grouped


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session_dirs = resolve_session_dirs(
        runs_root=args.runs_root,
        batch_summary_csv=args.batch_summary_csv,
        session_prefix=args.session_prefix,
        session_names=args.session_names,
    )
    if not session_dirs:
        raise FileNotFoundError("No session directories found to summarize.")

    session_records = {}
    skipped_sessions = []
    total_cells = 0
    for session_dir in session_dirs:
        records, reason = load_session_checkpoint_records(session_dir, args.max_cells_per_session)
        if reason is not None:
            skipped_sessions.append({"session_name": session_dir.name, "reason": reason})
            continue
        session_records[session_dir.name] = records
        total_cells += len(records)

    if total_cells == 0:
        raise RuntimeError("No checkpoint metrics were computed.")

    pbar = tqdm(total=total_cells, unit="cell") if tqdm is not None else None
    rows = []
    done_cells = 0
    for session_dir in session_dirs:
        session_name = session_dir.name
        records = session_records.get(session_name, [])
        if not records:
            continue
        session_ctx = build_session_eval_context(
            session_name=session_name,
            image_shape=records[0]["model_kwargs"]["image_shape"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )
        response_metrics = evaluate_session_response_metrics(records=records, session_ctx=session_ctx, device=args.device)
        rows.extend(build_rows_for_session(records, response_metrics))
        done_cells += len(records)
        msg = f"{format_progress(done_cells, total_cells)} session={session_name}"
        if pbar is not None:
            pbar.update(len(records))
            pbar.set_description(msg)
        else:
            print(msg, flush=True)

    if pbar is not None:
        pbar.close()

    metrics_df = pd.DataFrame(rows).sort_values(["session_name", "cell_id"]).reset_index(drop=True)
    if metrics_df.empty:
        raise RuntimeError("No checkpoint metrics were computed.")

    per_session_df = summarize_by_session(metrics_df)
    metrics_csv = output_dir / "per_cell_metrics.csv"
    sessions_csv = output_dir / "per_session_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    per_session_df.to_csv(sessions_csv, index=False)

    plot_separability_vs_linearity(metrics_df, output_dir / "separability_vs_linearity.png")
    plot_linear_vs_energy(metrics_df, output_dir / "linear_vs_energy_fraction.png")
    plot_spectral_summary(metrics_df, output_dir / "spectral_bandwidth_summary.png")
    plot_spectral_summary_by_session(metrics_df, output_dir / "seperate_session_color_spectral_bandwidth_summary.png")
    plot_phase_modulation_vs_linearity(metrics_df, output_dir / "phase_modulation_vs_linearity.png")
    plot_phase_modulation_vs_linearity(
        metrics_df,
        output_dir / "strong_fits_phase_modulation_vs_linearity.png",
        min_best_val_bps=0.1,
        title_prefix="Strong fits",
        min_linearity_index=-1.5,
    )
    plot_r2_component_diagnostics(metrics_df, output_dir / "r2_component_diagnostics.png")
    plot_best_val_bps_histogram(metrics_df, output_dir / "best_val_bps_histogram.png")
    plot_peak_sf_vs_centroid(metrics_df, output_dir / "peak_sf_vs_sf_centroid.png")
    plot_peak_orientation_vs_centroid(metrics_df, output_dir / "peak_orientation_vs_orientation_centroid.png")
    plot_rf_size_comparison(
        metrics_df,
        output_dir / "rf_size_data_vs_model_mean.png",
        model_col="model_band_sqrt_area_contour_mean_deg",
        title_label="mean over bands",
    )
    plot_rf_size_comparison(
        metrics_df,
        output_dir / "rf_size_data_vs_model_median.png",
        model_col="model_band_sqrt_area_contour_median_deg",
        title_label="median over bands",
    )
    plot_rf_size_comparison(
        metrics_df,
        output_dir / "rf_size_data_vs_model_max.png",
        model_col="model_band_sqrt_area_contour_max_deg",
        title_label="max over bands",
    )

    summary = {
        "n_sessions": int(metrics_df["session_name"].nunique()),
        "n_cells": int(len(metrics_df)),
        "sessions": sorted(metrics_df["session_name"].astype(str).unique().tolist()),
        "metrics_csv": str(metrics_csv),
        "sessions_csv": str(sessions_csv),
        "separability_plot": str(output_dir / "separability_vs_linearity.png"),
        "linear_energy_plot": str(output_dir / "linear_vs_energy_fraction.png"),
        "spectral_bandwidth_plot": str(output_dir / "spectral_bandwidth_summary.png"),
        "session_color_spectral_bandwidth_plot": str(
            output_dir / "seperate_session_color_spectral_bandwidth_summary.png"
        ),
        "phase_modulation_vs_linearity_plot": str(output_dir / "phase_modulation_vs_linearity.png"),
        "strong_fits_phase_modulation_vs_linearity_plot": str(
            output_dir / "strong_fits_phase_modulation_vs_linearity.png"
        ),
        "r2_component_diagnostics_plot": str(output_dir / "r2_component_diagnostics.png"),
        "best_val_bps_histogram_plot": str(output_dir / "best_val_bps_histogram.png"),
        "peak_sf_vs_sf_centroid_plot": str(output_dir / "peak_sf_vs_sf_centroid.png"),
        "peak_orientation_vs_orientation_centroid_plot": str(
            output_dir / "peak_orientation_vs_orientation_centroid.png"
        ),
        "rf_size_data_vs_model_mean_plot": str(output_dir / "rf_size_data_vs_model_mean.png"),
        "rf_size_data_vs_model_median_plot": str(output_dir / "rf_size_data_vs_model_median.png"),
        "rf_size_data_vs_model_max_plot": str(output_dir / "rf_size_data_vs_model_max.png"),
        "skipped_sessions": skipped_sessions,
        "notes": [
            "Linear and energy fractions are computed from response-space r^2 values against recorded activity, matching the paper text rather than a weight-power decomposition.",
            "The corrected implementation batches all cells within a session/lag group together, avoiding a separate validation sweep for every checkpoint.",
            f"Linearity-based plots mark cells as valid only when r2_full >= {MIN_R2_FULL_FOR_LI:g} to avoid ratio blow-ups from near-zero denominators.",
            "Orientation/spatial-frequency statistics are computed from total afferent weight magnitudes, using weighted means and standard deviations over orientation and log2 spatial frequency.",
            "Phase tuning uses gratings modulation_index, joined via model subset index -> true cids from VisionCore config -> full gratings unit index from the raw session cluster ordering.",
            "Separability is computed on nonnegative connection-strength maps (squared linear/energy maps), using the product of scale, orientation, and spatial marginals.",
            "The reported separability_ratio is the variance-weighted average of the linear and energy component separabilities.",
        ],
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
