# NOTE: LBFGS variant of two_stage.py (optimizer/training loop changed).
#%%
from DataYatesV1 import get_gaborium_sta_ste, get_session, plot_stas, enable_autoreload,calc_sta
enable_autoreload()
from models.losses import MaskedPoissonNLLLoss, PoissonBPSAggregator, MaskedLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import schedulefree
import numpy as np
#%%

from util import get_dataset_from_config
dataset_configs_path = '/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_gaborium_20lags.yaml'
train_dset, val_dset, dataset_config = get_dataset_from_config('Allen', '2022-04-13', dataset_configs_path)
cids = dataset_config['cids']

#%%
train_dset_loaded = train_dset[:]


stim = train_dset_loaded['stim']
robs = train_dset_loaded['robs']
dfs = train_dset_loaded['dfs']

n_lags = 5
# Calculate spike-triggered averages (STAs)
stas = calc_sta(stim.detach().cpu().squeeze()[:, 0, 5:-5, 5:-5],
                robs.cpu(),
                range(n_lags),
                dfs=dfs.cpu().squeeze(),
                progress=True).cpu().squeeze().numpy()

# # Calculate spike-triggered second moments (STEs)
# # Uses squared stimulus values via stim_modifier
stes = calc_sta(stim.detach().cpu().squeeze()[:, 0, 5:-5, 5:-5],
                robs.cpu(),
                range(n_lags),
                dfs=dfs.cpu().squeeze(),
                stim_modifier=lambda x: x**2,
                progress=True).cpu().squeeze().numpy()

# plot_stas(stas[:, :, None, :, :])
# plt.show()
# plot_stas(stes[:, :, None, :, :])
# plt.show()
peak_lags = np.array([stes[cc].std((1,2)).argmax() for cc in range(stes.shape[0])])


#%%
from pyr_utils import (
    find_pyr_size_and_height_for_lowest_cpd,
    get_pyr_band_frequencies,
    get_sf_info,
    calibrate_pyr_orientation_labels,
    to_display_orientation_convention,
)

# Example:
cfg = find_pyr_size_and_height_for_lowest_cpd(
    lowest_cpd_target=1.0,
    ppd=train_dset.dsets[0].metadata["ppd"],
    order=3,
    rel_tolerance=0.3,
    validate=True,
)
print(cfg)
#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
from plenoptic.simulate import SteerablePyramidFreq
import plenoptic as po


class TwoStage(nn.Module):
    def __init__(
        self,
        image_shape,
        n_neurons,
        n_lags=1,
        height=3,
        order=5,
        lowest_cpd_target=None,
        ppd=None,
        rel_tolerance=0.0,
        validate_cpd=True,
        beta_init=0.0,
        init_weight_scale=1.0,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.image_shape = image_shape  # user-facing spatial shape
        self.n_lags = n_lags
        self.height = height  # user-facing number of scales used in readout
        self.order = order
        self.lowest_cpd_target = lowest_cpd_target
        self.ppd = ppd
        self.rel_tolerance = rel_tolerance
        self.validate_cpd = validate_cpd
        self.init_weight_scale = init_weight_scale

        if self.lowest_cpd_target is not None:
            if self.ppd is None:
                raise ValueError("ppd must be provided when lowest_cpd_target is set.")
            cfg = find_pyr_size_and_height_for_lowest_cpd(
                lowest_cpd_target=self.lowest_cpd_target,
                ppd=self.ppd,
                order=self.order,
                rel_tolerance=self.rel_tolerance,
                validate=self.validate_cpd,
            )
            self.pyr_image_shape = tuple(cfg["image_shape"])
            self.pyr_height = int(cfg["height"])
            self.pyr_cfg = cfg
        else:
            self.pyr_image_shape = tuple(self.image_shape)
            self.pyr_height = int(self.height)
            self.pyr_cfg = None

        if self.height > self.pyr_height:
            raise ValueError(
                f"user height ({self.height}) cannot exceed internal pyramid height ({self.pyr_height})."
            )

        self.used_scales = list(range(self.pyr_height - self.height, self.pyr_height))
        self.pyr = SteerablePyramidFreq(
            self.pyr_image_shape,
            height=self.pyr_height,
            order=self.order,
            is_complex=False,
            downsample=False,
        )
        self.pyr_freq_info = []
        self.pyr_levels_to_cpd = {}
        # User-facing orientation degrees follow the visualization convention.
        self.orientation_degrees = [float(o * 180.0 / (self.order + 1)) for o in range(self.order + 1)]
        # Keep an explicit calibrated (non-mirrored) variant for diagnostics.
        self.orientation_calibrated_degrees = list(self.orientation_degrees)
        if self.ppd is not None:
            self.pyr_freq_info = get_pyr_band_frequencies(self.pyr, self.pyr_image_shape, self.ppd)
            sf_meta = get_sf_info(self.pyr_freq_info, return_full=True)
            self.pyr_levels_to_cpd = sf_meta["levels_to_cpd"]
            if len(sf_meta["orientation_degrees"]) == (self.order + 1):
                self.orientation_calibrated_degrees = sf_meta["orientation_degrees"]
        # Calibrate orientation labels from basis reconstructions so axis labels/icons
        # match bar orientation in image space (not raw frequency-axis convention).
        self.orientation_calibrated_degrees, self.orientation_check = calibrate_pyr_orientation_labels(
            self.pyr,
            image_shape=self.pyr_image_shape,
            num_orientations=(self.order + 1),
            scales=self.used_scales,
            device=self.pyr.lo0mask.device,
        )
        # Display convention for afferent-map figure: mirror handedness to match
        # paper-style orientation ordering (e.g., 90, 60, 30, 0, 150, 120).
        self.orientation_display_degrees = to_display_orientation_convention(self.orientation_calibrated_degrees)
        # Make default orientation metadata match what is shown in the figure.
        self.orientation_degrees = list(self.orientation_display_degrees)
        self.used_scale_cpd = [self.pyr_levels_to_cpd.get(scale, float("nan")) for scale in self.used_scales]

        n_bands = len(self.used_scales) * (order + 1)
        n_feat_per_half = n_bands * n_lags * image_shape[0] * image_shape[1]
        self.w_pos = nn.Linear(n_feat_per_half, n_neurons, bias=False)
        self.w_neg = nn.Linear(n_feat_per_half, n_neurons, bias=False)
        with torch.no_grad():
            self.w_pos.weight.mul_(self.init_weight_scale)
            self.w_neg.weight.mul_(self.init_weight_scale)
        hann_y = torch.hann_window(image_shape[0], periodic=False)
        hann_x = torch.hann_window(image_shape[1], periodic=False)
        hann_2d = torch.outer(hann_y, hann_x) ** 2
        hann_2d = hann_2d / hann_2d.max().clamp_min(1e-8)
        self.register_buffer(
            "hann_flat",
            hann_2d.reshape(-1).repeat(n_bands * n_lags),
        )
        self.alpha_pos = nn.Parameter(torch.ones(n_neurons))
        self.alpha_neg = nn.Parameter(torch.ones(n_neurons))
        self.beta = nn.Parameter(torch.ones(n_neurons) * beta_init)

    def _windowed_weight(self, weight):
        return weight * self.hann_flat.unsqueeze(0)

    def _get_center_crop_slices(self):
        h, w = self.image_shape
        ph, pw = self.pyr_image_shape
        if ph < h or pw < w:
            raise ValueError("Internal pyramid image shape must be >= user image_shape.")
        y0 = (ph - h) // 2
        x0 = (pw - w) // 2
        return slice(y0, y0 + h), slice(x0, x0 + w)

    def _pad_to_pyr_shape(self, x4d):
        # x4d: (N, 1, H, W)
        h, w = x4d.shape[-2:]
        ph, pw = self.pyr_image_shape
        if (h, w) == (ph, pw):
            return x4d
        pad_h = ph - h
        pad_w = pw - w
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Input image larger than internal pyramid image shape.")
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return F.pad(x4d, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")

    @property
    def positive_afferent_map_unwindowed(self):
        '''
        Returns the positive afferent map as a tensor of shape (n_neurons, n_lags, height, order+1, *image_shape)
        '''
        return self.w_pos.weight.reshape(
            self.n_neurons, self.n_lags, self.height, self.order+1, *self.image_shape
        )
    
    @property
    def negative_afferent_map_unwindowed(self):
        '''
        Returns the negative afferent map as a tensor of shape (n_neurons, n_lags, height, order+1, *image_shape)
        '''
        return self.w_neg.weight.reshape(
            self.n_neurons, self.n_lags, self.height, self.order+1, *self.image_shape
        )


    @property
    def positive_afferent_map(self):
        '''
        Returns the positive afferent map as a tensor of shape (n_neurons, n_lags, height, order+1, *image_shape)
        '''
        return self._windowed_weight(self.w_pos.weight).reshape(
            self.n_neurons, self.n_lags, self.height, self.order+1, *self.image_shape
        )
    
    @property
    def negative_afferent_map(self):
        '''
        Returns the negative afferent map as a tensor of shape (n_neurons, n_lags, height, order+1, *image_shape)
        '''
        return self._windowed_weight(self.w_neg.weight).reshape(
            self.n_neurons, self.n_lags, self.height, self.order+1, *self.image_shape
        )

    @property
    def linear_receptive_field(self):
        """
        Linear receptive fields in pixel space from (w+ - w-)/2.
        Returns shape: (n_neurons, n_lags, H, W)
        """
        assert self.n_neurons == 1 and self.n_lags == 1, "linear_receptive_field currently expects n_neurons == 1 and n_lags == 1"
        w_linear = 0.5 * (self.positive_afferent_map - self.negative_afferent_map)
        dummy = torch.zeros(1, 1, *self.pyr_image_shape, device=w_linear.device, dtype=w_linear.dtype)
        pyr_template = self.pyr(dummy)
        ys, xs = self._get_center_crop_slices()
        pyr_coeffs = {}
        for k, v in pyr_template.items():
            if isinstance(k, tuple):
                scale_idx, orient_idx = k
                pyr_coeffs[k] = torch.zeros_like(v)
                if scale_idx in self.used_scales:
                    local_scale_idx = self.used_scales.index(scale_idx)
                    pyr_coeffs[k][:, :, ys, xs] = w_linear[0, 0, local_scale_idx, orient_idx].unsqueeze(0).unsqueeze(0)
            else:
                pyr_coeffs[k] = torch.zeros_like(v)
        rf_full = self.pyr.recon_pyr(pyr_coeffs).squeeze(0).squeeze(0)
        rf = rf_full[ys, xs]
        return rf.unsqueeze(0).unsqueeze(0)

    @property
    def energy_receptive_fields(self):
        """
        Energy receptive field images in pixel space from we = (w+ + w-)/2.
        Returns a tuple: (exc_rf, inh_rf), each shape (n_neurons, n_lags, H, W),
        where exc_rf is built from we > 0 and inh_rf from we < 0.

        Characteristic-image approximation: for each spatial-frequency/orientation
        band, pick a global sign (+/-) greedily to reduce destructive interference
        during reconstruction.
        """
        assert self.n_neurons == 1 and self.n_lags == 1, "energy_receptive_fields currently expects n_neurons == 1 and n_lags == 1"
        w_energy = 0.5 * (self.positive_afferent_map + self.negative_afferent_map)
        w_exc = F.relu(w_energy)
        w_inh = F.relu(-w_energy)

        dummy = torch.zeros(1, 1, *self.pyr_image_shape, device=w_energy.device, dtype=w_energy.dtype)
        pyr_template = self.pyr(dummy)
        ys, xs = self._get_center_crop_slices()

        def _zero_coeffs():
            return {k: torch.zeros_like(v) for k, v in pyr_template.items()}

        def _characteristic_recon(w_nonneg):
            band_contrib = {}
            band_energy = {}
            for k in pyr_template:
                if not isinstance(k, tuple):
                    continue
                scale_idx, orient_idx = k
                if scale_idx not in self.used_scales:
                    continue
                local_scale_idx = self.used_scales.index(scale_idx)
                band_map = w_nonneg[0, 0, local_scale_idx, orient_idx]
                energy = float((band_map * band_map).sum().item())
                if energy <= 0.0:
                    continue
                coeffs_single = _zero_coeffs()
                coeffs_single[k][:, :, ys, xs] = band_map.unsqueeze(0).unsqueeze(0)
                contrib = self.pyr.recon_pyr(coeffs_single).squeeze(0).squeeze(0)[ys, xs]
                band_contrib[k] = contrib
                band_energy[k] = energy

            ordered = sorted(band_energy.keys(), key=lambda kk: band_energy[kk], reverse=True)
            accum = torch.zeros(
                (self.image_shape[0], self.image_shape[1]),
                device=w_nonneg.device,
                dtype=w_nonneg.dtype,
            )
            band_sign = {}
            for k in ordered:
                contrib = band_contrib[k]
                dot = torch.sum(accum * contrib)
                sign = -1.0 if dot < 0 else 1.0
                band_sign[k] = sign
                accum = accum + sign * contrib

            coeffs_final = _zero_coeffs()
            for k in ordered:
                scale_idx, orient_idx = k
                local_scale_idx = self.used_scales.index(scale_idx)
                band_map = w_nonneg[0, 0, local_scale_idx, orient_idx]
                coeffs_final[k][:, :, ys, xs] = (
                    band_sign[k] * band_map
                ).unsqueeze(0).unsqueeze(0)

            rf_full = self.pyr.recon_pyr(coeffs_final).squeeze(0).squeeze(0)
            return rf_full[ys, xs]

        rf_exc = _characteristic_recon(w_exc)
        rf_inh = _characteristic_recon(w_inh)
        return rf_exc.unsqueeze(0).unsqueeze(0), rf_inh.unsqueeze(0).unsqueeze(0)
    
    def get_pyr_feats(self, x):
        with torch.no_grad():
            s = x['stim'][:, 0]  # [B, 1, lags, H, W] -> [B, lags, H, W]
            B, L, H, W = s.shape
            s4d = s.reshape(B * L, 1, H, W)
            s4d = self._pad_to_pyr_shape(s4d)
            pyr_out = self.pyr(s4d, scales=list(self.used_scales))
            ys, xs = self._get_center_crop_slices()
            pos_feats, neg_feats = [], []
            for k, v in pyr_out.items():
                if isinstance(k, tuple) and (k[0] in self.used_scales):
                    vc = v[..., ys, xs]
                    pos_feats.append(F.relu(vc).reshape(B, -1))
                    neg_feats.append(F.relu(-vc).reshape(B, -1))

            pos_feats = torch.cat(pos_feats, dim=-1)
            neg_feats = torch.cat(neg_feats, dim=-1)

            return pos_feats, neg_feats, pyr_out

    def forward(self, x):
        pos_feats, neg_feats, _ = self.get_pyr_feats(x)

        z = F.linear(pos_feats, self._windowed_weight(self.w_pos.weight)) + F.linear(
            neg_feats, self._windowed_weight(self.w_neg.weight)
        )
        # print(z.shape, pos_feats.shape, neg_feats.shape)

        # x['rhat'] = (self.beta + self.alpha_pos * F.relu(z) + self.alpha_neg * F.relu(-z)).clamp(min=1e-6)
        x['rhat'] = (self.beta + F.relu(z)).clamp(min=1e-6)
        return x


def sparsity_penalty(model):
    w_star = torch.sqrt(model.w_pos.weight**2 + model.w_neg.weight**2)
    # w_star = get_w_star(model.positive_afferent_map[0, 0], model.negative_afferent_map[0, 0])
    return w_star.norm(1) / w_star.norm(2)


def prox_group_l21_(w_pos, w_neg, tau, eps=1e-12):
    """
    In-place proximal step on paired weights (w_pos, w_neg):
    prox_{tau * ||.||_2}(v) per feature pair.
    """
    if tau <= 0:
        return
    with torch.no_grad():
        norm = torch.sqrt(w_pos.pow(2) + w_neg.pow(2) + eps)
        scale = (1.0 - tau / norm).clamp_min(0.0)
        w_pos.mul_(scale)
        w_neg.mul_(scale)


def get_w_star(positive_afferent_map, negative_afferent_map, eps=1e-8):
    assert positive_afferent_map.ndim == 4 and negative_afferent_map.ndim == 4 and positive_afferent_map.shape == negative_afferent_map.shape, \
        "Expected 4D (height, order+1, H, W), same shape"
    return torch.sqrt(positive_afferent_map**2 + negative_afferent_map**2 + eps)



def weighted_variance_along_dim(w_star, dim, circular=False, eps=1e-8):
    assert w_star.ndim == 4, "Expected w_star to be 4D (height, order+1, H, W)"
    size = w_star.shape[dim]
    coords = torch.arange(size, device=w_star.device, dtype=w_star.dtype)
    shape = [1] * w_star.ndim
    shape[dim] = size
    coord = coords.view(*shape)
    w_sum = w_star.sum().clamp_min(eps)
    if circular:
        theta = 2 * torch.pi * coord / max(size, 1)
        c = (w_star * torch.cos(theta)).sum() / w_sum
        s = (w_star * torch.sin(theta)).sum() / w_sum
        return (1.0 - torch.sqrt((c * c + s * s).clamp(min=0.0, max=1.0))).clamp_min(0.0)
    mean = (w_star * coord).sum() / w_sum
    second = (w_star * coord * coord).sum() / w_sum
    return (second - mean * mean).clamp_min(0.0)


def locality_penalty_from_maps(positive_afferent_map, negative_afferent_map, circular_dims=None, eps=1e-8):
    """
    Convolution-style 4D locality penalty over (scale, orientation, y, x).

    Matches the `locality_conv` idea by applying distance-weighted convolution
    on squared energy marginals per dimension, then summing contributions.
    Orientation can be treated as circular via `circular_dims={1}`.
    Input shape: (height, order+1, H, W).
    """
    w_star = get_w_star(positive_afferent_map, negative_afferent_map, eps=eps)
    # Match locality_conv behavior by using squared magnitude.
    e = w_star.pow(2)
    circular_dims = set() if circular_dims is None else set(circular_dims)
    n_s, n_o, n_v, n_h = [int(x) for x in e.shape]

    # FFT domain shape: linear dims use full-conv size (2n-1), circular dims keep size n.
    fft_shape = [
        n_s if 0 in circular_dims else (2 * n_s - 1),
        n_o if 1 in circular_dims else (2 * n_o - 1),
        n_v if 2 in circular_dims else (2 * n_v - 1),
        n_h if 3 in circular_dims else (2 * n_h - 1),
    ]
    denom = float(2 * (n_s + n_o + n_v + n_h) ** 2)

    def _axis_distance(n, circular, fft_n, device, dtype):
        if circular:
            idx = torch.arange(fft_n, device=device, dtype=dtype)
            return torch.minimum(idx, (fft_n - idx).to(dtype)).pow(2) / max(denom, 1.0)
        offs = torch.arange(fft_n, device=device, dtype=dtype) - (n - 1)
        return offs.pow(2) / max(denom, 1.0)

    ds = _axis_distance(n_s, (0 in circular_dims), fft_shape[0], e.device, e.dtype)
    do = _axis_distance(n_o, (1 in circular_dims), fft_shape[1], e.device, e.dtype)
    dv = _axis_distance(n_v, (2 in circular_dims), fft_shape[2], e.device, e.dtype)
    dh = _axis_distance(n_h, (3 in circular_dims), fft_shape[3], e.device, e.dtype)

    # Joint 4D distance kernel (single 4D volume convolution).
    k = (
        ds[:, None, None, None]
        + do[None, :, None, None]
        + dv[None, None, :, None]
        + dh[None, None, None, :]
    )

    # Embed energy in FFT volume.
    e_pad = e.new_zeros(tuple(fft_shape))
    e_pad[:n_s, :n_o, :n_v, :n_h] = e

    conv_full = torch.fft.ifftn(torch.fft.fftn(e_pad) * torch.fft.fftn(k)).real

    # Extract "same" region along linear dims, direct region along circular dims.
    s0 = 0 if 0 in circular_dims else (n_s - 1)
    o0 = 0 if 1 in circular_dims else (n_o - 1)
    v0 = 0 if 2 in circular_dims else (n_v - 1)
    h0 = 0 if 3 in circular_dims else (n_h - 1)
    conv_same = conv_full[s0:s0 + n_s, o0:o0 + n_o, v0:v0 + n_v, h0:h0 + n_h]

    l_local = torch.sum(e * conv_same)
    z = l_local.new_zeros(())
    return l_local, (z, z, z, z)


def visualize_afferent_map(model, figsize=None, title=None, eps=1e-8, show_examples=True):
    """Visualize model afferent maps with hue=on/off proportion and saturation=amplitude."""
    import matplotlib.colors as mcolors
    import matplotlib.patches as patches

    def _draw_gabor_icon(ax, theta_deg, sf_rank, sf_count, color="#36b76f"):
        # Paper-style cartoon: two touching oriented lobes, one filled and one unfilled.
        frac = 0.0 if sf_count <= 1 else float(sf_rank) * 1.5 / float(sf_count - 1) 
        lobe_len = 0.9 - 0.22 * frac
        lobe_thick = lobe_len * 0.451 - 0.05 * frac
        # Side-by-side touching means center shift along the MINOR axis (perpendicular).
        sep = lobe_thick * 1 # touching / slight overlap for visibility
        th_perp = np.deg2rad(theta_deg + 90.0)
        dx = 0.5 * sep * np.cos(th_perp)
        dy = 0.5 * sep * np.sin(th_perp)
        c1 = (0.5 - dx, 0.5 - dy)
        c2 = (0.5 + dx, 0.5 + dy)
        # Unfilled lobe
        ax.add_patch(
            patches.Ellipse(
                c1, width=lobe_len, height=lobe_thick, angle=theta_deg,
                facecolor="white", edgecolor=color, linewidth=2.4
            )
        )
        # Filled lobe
        ax.add_patch(
            patches.Ellipse(
                c2, width=lobe_len, height=lobe_thick, angle=theta_deg,
                facecolor=color, edgecolor=color, linewidth=2.4, alpha=0.6
            )
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")

    w_plus = model.positive_afferent_map[0, 0].detach().cpu().numpy()
    w_minus = model.negative_afferent_map[0, 0].detach().cpu().numpy()
    assert w_plus.ndim == 4 and w_minus.ndim == 4 and w_plus.shape == w_minus.shape, "Expected 4D (height, order+1, H, W), same shape"
    height, n_orient, _, _ = w_plus.shape
    row_order = np.arange(height)
    if getattr(model, "ppd", None) is not None:
        cpd_arr = np.asarray(model.used_scale_cpd[:height], dtype=float)
        finite_idx = np.where(np.isfinite(cpd_arr))[0]
        nonfinite_idx = np.where(~np.isfinite(cpd_arr))[0]
        if finite_idx.size > 0:
            row_order = np.concatenate(
                [finite_idx[np.argsort(cpd_arr[finite_idx])], nonfinite_idx]
            )
            w_plus = w_plus[row_order]
            w_minus = w_minus[row_order]
    w_star = np.sqrt(w_plus**2 + w_minus**2)
    w_max = w_star.max() + eps
    sat = np.clip(w_star / w_max, 0, 1)
    angle = np.arctan2(w_minus, w_plus)
    # Match paper legend orientation: right=On excitation, up=Off excitation, left=On inhibition, down=Off inhibition
    hue = (1.0 / 3.0 - angle / (2 * np.pi)) % 1.0
    val = np.ones_like(hue)
    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    if figsize is None:
        figsize = (2 * n_orient, 2 * height)
    fig, axes = plt.subplots(height, n_orient, figsize=figsize, squeeze=False)
    for i in range(height):
        for j in range(n_orient):
            axes[i, j].imshow(rgb[i, j])
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    display_orientations = getattr(model, "orientation_display_degrees", model.orientation_degrees)
    x_labels = [f"{deg:.0f}" for deg in display_orientations[:n_orient]]
    if getattr(model, "ppd", None) is None:
        y_labels = [f"scale {model.used_scales[idx]}" for idx in row_order]
        y_axis_label = "Band spatial frequency (scale)"
    else:
        cpd_vals = np.asarray(model.used_scale_cpd[:height], dtype=float)[row_order]
        y_labels = [f"{cpd:.2f}" if np.isfinite(cpd) else "n/a" for cpd in cpd_vals]
        y_axis_label = "Band spatial frequency (cpd)"
    for j, lbl in enumerate(x_labels):
        axes[-1, j].set_xlabel(lbl)
    for i, lbl in enumerate(y_labels):
        axes[i, 0].set_ylabel(lbl, rotation=0, va="center", ha="right", labelpad=14)
    fig.supxlabel("Band orientation (deg)", y=0.08)
    fig.supylabel(y_axis_label, x=0.12)
    if title:
        fig.suptitle(title)
    plt.tight_layout(rect=(0.16, 0.14, 0.98, 0.95))

    if show_examples:
        ori_examples = [float(d) for d in display_orientations[:n_orient]]

        # Left-side example cartoon gabors for spatial-frequency progression.
        # Keep these vertically oriented to match the paper-style side legend.
        for i in range(height):
            pos = axes[i, 0].get_position()
            box_h = pos.height * 0.30
            box_w = pos.width * 0.30
            x0 = pos.x0 - box_w * 2.75
            y0 = pos.y0 + (pos.height - box_h) * 0.5
            gax = fig.add_axes([x0, y0, box_w, box_h])
            _draw_gabor_icon(gax, theta_deg=90.0, sf_rank=i, sf_count=height)

        # Bottom example cartoon gabors for orientation progression.
        sf_mid_rank = max(0, height // 2)
        for j in range(n_orient):
            pos = axes[-1, j].get_position()
            box_h = pos.height * 0.30
            box_w = pos.width * 0.30
            x0 = pos.x0 + (pos.width - box_w) * 0.5
            y0 = pos.y0 - box_h * 2.05
            gax = fig.add_axes([x0, y0, box_w, box_h])
            _draw_gabor_icon(gax, theta_deg=ori_examples[j], sf_rank=sf_mid_rank, sf_count=height)

    return fig, axes


def render_energy_component_rgb(component, hue_rgb, amp_scale=None, carrier_scale=None, bg_gray=0.90, eps=1e-8):
    """
    Render one energy component using:
    - phase-invariant power envelope (|component|) for color strength
    - signed carrier (component) for light/dark sinusoidal structure
    """
    arr = component.detach().cpu().numpy() if torch.is_tensor(component) else np.asarray(component)
    arr = np.asarray(arr, dtype=np.float32)
    abs_arr = np.abs(arr)
    if amp_scale is None:
        amp_scale = float(np.percentile(abs_arr, 99))
    if carrier_scale is None:
        carrier_scale = float(abs_arr.max())
    amp = np.clip(abs_arr / (amp_scale + eps), 0.0, 1.0)
    carrier = np.clip(arr / (carrier_scale + eps), -1.0, 1.0)

    # Warm/cool hue saturation from power envelope.
    hue = np.asarray(hue_rgb, dtype=np.float32).reshape(1, 1, 3)
    base = np.full((*arr.shape, 3), fill_value=bg_gray, dtype=np.float32)
    rgb = (1.0 - amp[..., None]) * base + amp[..., None] * hue

    # Signed carrier controls local brightness to reveal sinusoidal phase.
    light = 0.5 + 0.5 * carrier
    rgb = rgb * (0.65 + 0.35 * light[..., None])
    return np.clip(rgb, 0.0, 1.0)


lambda_reg = 1e-4
gamma_local = lambda_reg * 4/20#4/20 #4/20#1/10 
sparsity_mode = "ratio_l1_l2"  # options: "ratio_l1_l2", "prox_l1"
lambda_prox = 1e-4  # used only when sparsity_mode == "prox_l1"
lambda_local_prox = 1e-1  # optional locality weight in prox mode
circular_dims = {1}
# circular_dims = {}
losses = []
crop_size = 5
cell_ids = [66]
num_epochs = 80
target_bps_val = 2
plot_every = 1  # set >0 to enable periodic plotting

spike_loss = MaskedPoissonNLLLoss(pred_key='rhat', target_key='robs', mask_key='dfs')
# spike_loss =  MaskedLoss(nn.MSELoss(reduction='none'), pred_key='rhat', target_key='robs', mask_key='dfs')
# n_lags = len(dataset_config['keys_lags']['stim'])
n_lags = 1
# image_shape = train_dset[0]['stim'].shape[2:]
image_shape = (41, 41)
# num_neurons = len(dataset_config['cids'])
num_neurons = 1
# beta_init = robs[:, cell_ids[0]].mean().item()
beta_init = 0.0
model = TwoStage(
    image_shape=image_shape,
    n_neurons=num_neurons,
    n_lags=n_lags,
    height=3,
    order=5,
    lowest_cpd_target=1.0,
    ppd=train_dset.dsets[0].metadata["ppd"],
    rel_tolerance=0.3,
    validate_cpd=True,
    beta_init=beta_init,
    init_weight_scale=1e-4,
)


def _resolve_output_indices(requested_ids, out_dict):
    """
    Resolve prediction and target indices for single-cell or multi-cell runs.
    Returns: (pred_idx_for_rhat, target_idx_for_robs_dfs)
    """
    n_rhat = int(out_dict["rhat"].shape[1])
    n_robs = int(out_dict["robs"].shape[1])
    n_dfs = int(out_dict["dfs"].shape[1])
    if n_robs != n_dfs:
        raise ValueError(
            f"Inconsistent target widths: robs={n_robs}, dfs={n_dfs}"
        )

    # Targets/masks usually come from full population tensors.
    if len(requested_ids) == 1 and n_robs == 1:
        target_idx = [0]
    elif min(requested_ids) >= -n_robs and max(requested_ids) < n_robs:
        target_idx = list(requested_ids)
    else:
        raise IndexError(
            f"Requested cell_ids={requested_ids} out of bounds for robs/dfs width {n_robs}."
        )

    # Predictions may be local model outputs (e.g. n_rhat == 1).
    if len(target_idx) == 1 and n_rhat == 1:
        pred_idx = [0]
    elif n_rhat == len(target_idx):
        pred_idx = list(range(n_rhat))
    elif min(target_idx) >= -n_rhat and max(target_idx) < n_rhat:
        pred_idx = list(target_idx)
    else:
        raise ValueError(
            f"Cannot align rhat width {n_rhat} with requested cells {requested_ids} "
            f"and target width {n_robs}."
        )

    return pred_idx, target_idx

model.cuda()
torch.cuda.empty_cache()
train_dset.to('cpu')
val_dset.to('cpu')
batch_size = 20024  # 64

train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)
val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.LBFGS(
    model.parameters(),
    # lr=0.2,
    # max_iter=5,
    # history_size=10,
    # line_search_fn="strong_wolfe",
    # lr=1,
    # max_iter=10000,
    # tolerance_grad=1e-6,
    # # tolerance_change=1e-8,
    # tolerance_change=1e-6,
    # # line_search_fn='strong_wolfe'
    # history_size=10,
    lr=1,
    max_iter=5,
    history_size=10,
    line_search_fn="strong_wolfe",
)

for epoch in range(num_epochs):
    train_agg = PoissonBPSAggregator()
    val_agg = PoissonBPSAggregator()
    prox_tau_last = 0.0
    
    for i, batch in enumerate(tqdm(train_loader)):
        model.train()
        batch = {k: v.cuda() for k, v in batch.items()}
        batch['stim'] = batch['stim'][:, :, peak_lags[cell_ids], crop_size:-crop_size, crop_size:-crop_size]
        step_stats = {}

        def closure():
            optimizer.zero_grad()
            out = model(batch)
            pred_idx, target_idx = _resolve_output_indices(cell_ids, out)
            out['rhat'] = out['rhat'][:, pred_idx]
            out['robs'] = out['robs'][:, target_idx]
            out['dfs'] = out['dfs'][:, target_idx]
            assert out['dfs'].shape == out['rhat'].shape == out['robs'].shape

            poisson_loss = spike_loss(out)
            pos_map = model.positive_afferent_map[0, 0]
            neg_map = model.negative_afferent_map[0, 0]
            l_local, _ = locality_penalty_from_maps(pos_map, neg_map, circular_dims=circular_dims)
            if sparsity_mode == "ratio_l1_l2":
                l_sparse = sparsity_penalty(model)
                # gamma_local_local = 0.05 / max(l_local.detach().item(), 1e-12)
                gamma_local_local = gamma_local
                reg_term = lambda_reg * l_sparse * (1.0 + gamma_local_local * l_local)
            elif sparsity_mode == "prox_l1":
                l_sparse = l_local.new_zeros(())
                gamma_local_local = 0.0
                reg_term = lambda_local_prox * l_local
            else:
                raise ValueError(f"Unknown sparsity_mode: {sparsity_mode}")
            loss = poisson_loss + reg_term
            loss.backward()
            step_stats["poisson"] = float(poisson_loss.detach().item())
            step_stats["sparse"] = float(l_sparse.detach().item())
            step_stats["local"] = float(l_local.detach().item())
            step_stats["reg"] = float(reg_term.detach().item())
            step_stats["gamma_local"] = float(gamma_local_local)
            step_stats["loss"] = float(loss.detach().item())
            return loss

        optimizer.step(closure)
        if sparsity_mode == "prox_l1":
            lr = float(optimizer.param_groups[0].get("lr", 1e-3))
            prox_tau_last = lr * lambda_prox
            prox_group_l21_(model.w_pos.weight, model.w_neg.weight, tau=prox_tau_last)
        losses.append(step_stats["loss"])

        poisson_last = step_stats["poisson"]
        sparse_last = step_stats["sparse"]
        local_last = step_stats["local"]
        reg_last = step_stats["reg"]
        gamma_local = step_stats["gamma_local"]

        with torch.no_grad():
            out = model(batch)
            pred_idx, target_idx = _resolve_output_indices(cell_ids, out)
            out['rhat'] = out['rhat'][:, pred_idx]
            out['robs'] = out['robs'][:, target_idx]
            out['dfs'] = out['dfs'][:, target_idx]
            train_agg(out)
    
    for batch in val_loader:
        model.eval()
        with torch.no_grad():
            batch = {k: v.cuda() for k, v in batch.items()}
            batch['stim'] = batch['stim'][:, :, peak_lags[cell_ids], crop_size:-crop_size, crop_size:-crop_size]
            out = model(batch)
            pred_idx, target_idx = _resolve_output_indices(cell_ids, out)
            out['rhat'] = out['rhat'][:, pred_idx]
            out['robs'] = out['robs'][:, target_idx]
            out['dfs'] = out['dfs'][:, target_idx]
            assert out['dfs'].shape == out['rhat'].shape == out['robs'].shape
            val_agg(out)
    bps = train_agg.closure().cpu().numpy()
    bps_val = val_agg.closure().cpu().numpy()

    if plot_every > 0 and (epoch % plot_every == 0):
        fig, axes = visualize_afferent_map(model, title=f"Cell {cell_ids[0]}")
        plt.show()
        sta_img = stas[cell_ids[0], peak_lags[cell_ids[0]]]
        energy_exc_rf, energy_inh_rf = model.energy_receptive_fields
        energy_exc_np = energy_exc_rf[0, 0].detach().cpu().numpy()
        energy_inh_np = energy_inh_rf[0, 0].detach().cpu().numpy()
        joint_abs = np.concatenate([np.abs(energy_exc_np).reshape(-1), np.abs(energy_inh_np).reshape(-1)])
        joint_amp_scale = float(np.percentile(joint_abs, 99))
        joint_carrier_scale = float(joint_abs.max())
        exc_rgb = render_energy_component_rgb(
            energy_exc_np,
            hue_rgb=(0.95, 0.70, 0.35),
            amp_scale=joint_amp_scale,
            carrier_scale=joint_carrier_scale,
        )
        inh_rgb = render_energy_component_rgb(
            energy_inh_np,
            hue_rgb=(0.45, 0.70, 0.95),
            amp_scale=joint_amp_scale,
            carrier_scale=joint_carrier_scale,
        )
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(model.linear_receptive_field[0, 0].detach().cpu().numpy(), cmap='coolwarm_r')
        axes[0].set_title("Linear RF")
        axes[0].axis('off')
        axes[1].imshow(exc_rgb)
        axes[1].set_title("Energy Exc RF")
        axes[1].axis('off')
        axes[2].imshow(inh_rgb)
        axes[2].set_title("Energy Inh RF")
        axes[2].axis('off')
        axes[3].imshow(sta_img, cmap='coolwarm_r')
        axes[3].set_title(f"STA (cell {cell_ids[0]})")
        axes[3].axis('off')
        plt.tight_layout()
        plt.show()
        locality_factor = gamma_local * local_last
        print(
            f"mode={sparsity_mode}, poisson={poisson_last:.6f}, "
            f"L_sparse={sparse_last:.6f}, L_local={local_last:.6f}, "
            f"gamma*L_local={locality_factor:.6f} ({100.0 * locality_factor:.2f}%), "
            f"prox_tau={prox_tau_last:.6e}, reg={reg_last:.6f}"
        ) 
        # plt.plot(losses)
        # plt.show()

        print("beta:", model.beta.item())
        print(bps.item())
        print(bps_val.item())

    bps_train_scalar = float(np.asarray(bps).reshape(-1)[0])
    bps_val_scalar = float(np.asarray(bps_val).reshape(-1)[0])
    print(
        f"epoch={epoch:03d} bps_train={bps_train_scalar:.4f} "
        f"bps_val={bps_val_scalar:.4f} poisson={poisson_last:.6f} reg={reg_last:.6f}"
    )
    if bps_val_scalar >= target_bps_val:
        print(f"Reached target validation BPS {bps_val_scalar:.4f} >= {target_bps_val:.4f} at epoch {epoch}.")
        break
    

# %%
