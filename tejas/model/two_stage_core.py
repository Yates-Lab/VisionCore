import torch
import torch.nn as nn
import torch.nn.functional as F
from plenoptic.simulate import SteerablePyramidFreq

from pyr_utils import (
    calibrate_pyr_orientation_labels,
    find_pyr_size_and_height_for_lowest_cpd,
    get_pyr_band_frequencies,
    get_sf_info,
    to_display_orientation_convention,
)


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
        beta_as_parameter=True,
        clamp_beta_min=None,
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
        self.beta_as_parameter = bool(beta_as_parameter)
        self.clamp_beta_min = clamp_beta_min

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
        beta_tensor = torch.ones(n_neurons) * beta_init
        if self.beta_as_parameter:
            self.beta = nn.Parameter(beta_tensor)
        else:
            self.register_buffer("beta", beta_tensor)

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
        """
        Returns the positive afferent map as a tensor of shape (n_neurons, n_lags, height, order+1, *image_shape)
        """
        return self.w_pos.weight.reshape(
            self.n_neurons, self.n_lags, self.height, self.order + 1, *self.image_shape
        )

    @property
    def negative_afferent_map_unwindowed(self):
        """
        Returns the negative afferent map as a tensor of shape (n_neurons, n_lags, height, order+1, *image_shape)
        """
        return self.w_neg.weight.reshape(
            self.n_neurons, self.n_lags, self.height, self.order + 1, *self.image_shape
        )

    @property
    def positive_afferent_map(self):
        """
        Returns the positive afferent map as a tensor of shape (n_neurons, n_lags, height, order+1, *image_shape)
        """
        return self._windowed_weight(self.w_pos.weight).reshape(
            self.n_neurons, self.n_lags, self.height, self.order + 1, *self.image_shape
        )

    @property
    def negative_afferent_map(self):
        """
        Returns the negative afferent map as a tensor of shape (n_neurons, n_lags, height, order+1, *image_shape)
        """
        return self._windowed_weight(self.w_neg.weight).reshape(
            self.n_neurons, self.n_lags, self.height, self.order + 1, *self.image_shape
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
            s = x["stim"][:, 0]  # [B, 1, lags, H, W] -> [B, lags, H, W]
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
        if self.clamp_beta_min is not None:
            with torch.no_grad():
                self.beta.clamp_(min=float(self.clamp_beta_min))

        x["rhat"] = (self.beta + self.alpha_pos * F.relu(z) + self.alpha_neg * F.relu(-z)).clamp(min=1e-6)
        return x
