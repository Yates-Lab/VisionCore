"""Two-stage model per Oleskiw et al. 2024: steerable pyramid + pos/neg readout + piecewise-linear output."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from plenoptic.simulate import SteerablePyramidFreq


class TwoStage(nn.Module):
    def __init__(self, image_shape, n_neurons, n_lags=1, height=3, order=5):
        super().__init__()
        self.image_shape = image_shape
        self.n_lags = n_lags
        self.height = height
        self.order = order
        self.pyr = SteerablePyramidFreq(
            image_shape, height=height, order=order,
            is_complex=False, downsample=False,
        )
        n_bands = height * (order + 1)
        self.n_bands = n_bands
        H, W = image_shape[0], image_shape[1]
        n_feat_per_half = n_bands * n_lags * H * W
        self.n_feat_per_half = n_feat_per_half
        self.w_pos = nn.Linear(n_feat_per_half, n_neurons, bias=False)
        self.w_neg = nn.Linear(n_feat_per_half, n_neurons, bias=False)
        self.alpha_pos = nn.Parameter(torch.ones(n_neurons))
        self.alpha_neg = nn.Parameter(torch.ones(n_neurons))
        self.beta = nn.Parameter(torch.zeros(n_neurons))
        self.register_buffer("feat_scale", torch.tensor(1.0))  # set from data
        # Optional LN generator path: z_total = z_pyr + gamma * (generator / gen_scale)
        self.gamma_ln = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("gen_scale", torch.tensor(1.0))

    def get_pyr_feats(self, x):
        with torch.no_grad():
            s = x["stim"][:, 0]  # [B, 1, lags, H, W] -> [B, lags, H, W]
            B, L, H, W = s.shape
            pyr_out = self.pyr(s.reshape(B * L, 1, H, W))
            pos_feats, neg_feats = [], []
            for k, v in pyr_out.items():
                if isinstance(k, tuple):
                    pos_feats.append(F.relu(v).reshape(B, -1))
                    neg_feats.append(F.relu(-v).reshape(B, -1))
            pos_feats = torch.cat(pos_feats, dim=1)
            neg_feats = torch.cat(neg_feats, dim=1)
            pos_feats = pos_feats / self.feat_scale
            neg_feats = neg_feats / self.feat_scale
            return pos_feats, neg_feats, pyr_out

    def sta_in_pyramid_weights(self, sta_numpy, device):
        """Compute w_pos, w_neg from STA (n_lags, H, W) so readout approximates STA in pyramid space."""
        import numpy as np
        L, H, W = sta_numpy.shape
        assert (H, W) == self.image_shape
        sta_t = torch.from_numpy(sta_numpy).float().to(device)
        with torch.no_grad():
            # sta_t (L, H, W) -> (L, 1, H, W) for pyramid
            x = sta_t.unsqueeze(1)
            pyr_out = self.pyr(x)
            parts = []
            for k, v in pyr_out.items():
                if isinstance(k, tuple):
                    # v (L, 1, H, W) -> flatten to (L*H*W,)
                    parts.append(v.reshape(-1))
            sta_pyr = torch.cat(parts, dim=0)
            n = sta_pyr.numel()
            # Scale so that z = w@feats has std ~1 when feats have std~1 (avoid huge rhat)
            sta_pyr = sta_pyr / (sta_pyr.norm() + 1e-8)
            scale = (n ** -0.5) * 10.0  # ~1/sqrt(n) so z_std ~ 10; then we fit beta/alpha
            w_pos = F.relu(sta_pyr) * scale
            w_neg = F.relu(-sta_pyr) * scale
        return w_pos, w_neg

    def forward(self, x):
        pos_feats, neg_feats, _ = self.get_pyr_feats(x)
        z = self.w_pos(pos_feats) + self.w_neg(neg_feats)
        if "generator" in x and x["generator"] is not None:
            g = x["generator"].reshape(z.shape[0], -1).to(z.device) / self.gen_scale.clamp(min=1e-6)
            z = z + self.gamma_ln * g
        x["rhat"] = (self.beta + self.alpha_pos * F.softplus(z)).clamp(min=1e-6)
        return x


def sparsity_penalty(model):
    """Paper Eq 5-6: L1/L2 ratio on weight amplitudes w* = sqrt(w+^2 + w-^2)."""
    w_star = torch.sqrt(model.w_pos.weight ** 2 + model.w_neg.weight ** 2)
    return w_star.norm(1) / (w_star.norm(2) + 1e-8)


def locality_penalty(model, gamma_spatial=1.0, gamma_spectral=1.0):
    """Paper Eq 7-8: penalize spatial and spectral dispersion of weight magnitudes w*."""
    w_pos = model.w_pos.weight  # (n_neurons, n_feat)
    w_neg = model.w_neg.weight
    w_star = torch.sqrt(w_pos ** 2 + w_neg ** 2)  # (n_neurons, n_feat)
    n_bands = model.n_bands
    n_lags = model.n_lags
    H, W = model.image_shape[0], model.image_shape[1]
    # Reshape to (n_neurons, n_bands, n_lags, H, W)
    w = w_star.reshape(w_star.shape[0], n_bands, n_lags, H, W)
    w = w + 1e-8
    total = w.sum(dim=(1, 2, 3, 4), keepdim=True)
    w_n = w / total.clamp(min=1e-8)
    # Spatial variance: over H, W (indices as positions)
    h_idx = torch.arange(H, device=w.device, dtype=w.dtype).view(1, 1, 1, H, 1).expand_as(w)
    v_idx = torch.arange(W, device=w.device, dtype=w.dtype).view(1, 1, 1, 1, W).expand_as(w)
    mean_h = (w_n * h_idx).sum()
    mean_v = (w_n * v_idx).sum()
    var_h = (w_n * (h_idx - mean_h) ** 2).sum()
    var_v = (w_n * (v_idx - mean_v) ** 2).sum()
    spatial_disp = torch.sqrt(var_h + var_v + 1e-8)
    # Spectral variance: over bands (and lags)
    band_idx = torch.arange(n_bands, device=w.device, dtype=w.dtype).view(1, n_bands, 1, 1, 1).expand_as(w)
    mean_b = (w_n * band_idx).sum()
    var_band = (w_n * (band_idx - mean_b) ** 2).sum()
    spectral_disp = torch.sqrt(var_band + 1e-8)
    return gamma_spatial * spatial_disp + gamma_spectral * spectral_disp
