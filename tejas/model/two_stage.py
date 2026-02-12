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
from pyr_utils import find_pyr_size_and_height_for_lowest_cpd

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

        n_bands = len(self.used_scales) * (order + 1)
        n_feat_per_half = n_bands * n_lags * image_shape[0] * image_shape[1]
        self.w_pos = nn.Linear(n_feat_per_half, n_neurons, bias=False)
        self.w_neg = nn.Linear(n_feat_per_half, n_neurons, bias=False)
        hann_y = torch.hann_window(image_shape[0], periodic=False)
        hann_x = torch.hann_window(image_shape[1], periodic=False)
        hann_2d = torch.outer(hann_y, hann_x)
        hann_2d = hann_2d / hann_2d.max().clamp_min(1e-8)
        self.register_buffer(
            "hann_flat",
            hann_2d.reshape(-1).repeat(n_bands * n_lags),
        )
        self.alpha_pos = nn.Parameter(torch.ones(n_neurons))
        self.alpha_neg = nn.Parameter(torch.ones(n_neurons))
        self.beta = nn.Parameter(torch.zeros(n_neurons))

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

        x['rhat'] = (self.beta + self.alpha_pos * F.relu(z) + self.alpha_neg * F.relu(-z)).clamp(min=1e-6)
        return x


def sparsity_penalty(model):
    w_star = torch.sqrt(model.w_pos.weight**2 + model.w_neg.weight**2)
    return w_star.norm(1) / w_star.norm(2)


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
    w_star = get_w_star(positive_afferent_map, negative_afferent_map, eps=eps)
    circular_dims = set() if circular_dims is None else set(circular_dims)
    sigma_s2 = weighted_variance_along_dim(w_star, 0, circular=(0 in circular_dims), eps=eps)
    sigma_o2 = weighted_variance_along_dim(w_star, 1, circular=(1 in circular_dims), eps=eps)
    sigma_v2 = weighted_variance_along_dim(w_star, 2, circular=(2 in circular_dims), eps=eps)
    sigma_h2 = weighted_variance_along_dim(w_star, 3, circular=(3 in circular_dims), eps=eps)
    l_local = torch.sqrt(sigma_h2 + sigma_v2 + eps) + torch.sqrt(sigma_s2 + sigma_o2 + eps)
    return l_local, (sigma_h2, sigma_v2, sigma_o2, sigma_s2)


def visualize_afferent_map(positive_afferent_map, negative_afferent_map, figsize=None, title=None, eps=1e-8):
    """Visualize 4D afferent maps (height, order+1, H, W) with hue = on/off proportion, saturation = amplitude w*."""
    import matplotlib.colors as mcolors
    w_plus = positive_afferent_map.detach().cpu().numpy() if torch.is_tensor(positive_afferent_map) else np.asarray(positive_afferent_map)
    w_minus = negative_afferent_map.detach().cpu().numpy() if torch.is_tensor(negative_afferent_map) else np.asarray(negative_afferent_map)
    assert w_plus.ndim == 4 and w_minus.ndim == 4 and w_plus.shape == w_minus.shape, "Expected 4D (height, order+1, H, W), same shape"
    height, n_orient, _, _ = w_plus.shape
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
    axes[0, 0].figure.supylabel("Band spatial frequency")
    axes[-1, n_orient // 2].set_xlabel("Band orientation (deg)")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    return fig, axes




spike_loss = MaskedPoissonNLLLoss(pred_key='rhat', target_key='robs', mask_key='dfs')
# n_lags = len(dataset_config['keys_lags']['stim'])
n_lags = 1
# image_shape = train_dset[0]['stim'].shape[2:]
image_shape = (41, 41)
# num_neurons = len(dataset_config['cids'])
num_neurons = 1
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
)

model.cuda()
torch.cuda.empty_cache()
train_dset.to('cpu')
val_dset.to('cpu')
batch_size = 1024 # 64

train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)
val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = schedulefree.RAdamScheduleFree(model.parameters())
# optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-3)
lambda_reg = 1e-2
gamma_local = lambda_reg * 1/20#1/10 
circular_dims = {1}
# circular_dims = {}
losses = []
crop_size = 5
cell_ids = [14]
for epoch in range(100):
    train_agg = PoissonBPSAggregator()
    val_agg = PoissonBPSAggregator()
    
    for i, batch in enumerate(tqdm(train_loader)):
        optimizer.train()
        batch = {k: v.cuda() for k, v in batch.items()}
        batch['stim'] = batch['stim'][:, :, peak_lags[cell_ids], crop_size:-crop_size, crop_size:-crop_size]
        out = model(batch)
        out['robs'] = out['robs'][:, cell_ids]
        out['dfs'] = out['dfs'][:, cell_ids]
        assert out['dfs'].shape == out['rhat'].shape == out['robs'].shape
        
        poisson_loss = spike_loss(out)
        pos_map = model.positive_afferent_map[0, 0]
        neg_map = model.negative_afferent_map[0, 0]
        l_sparse = sparsity_penalty(model)
        l_local, _ = locality_penalty_from_maps(pos_map, neg_map, circular_dims=circular_dims)
        reg_term = lambda_reg * l_sparse * (1.0 + gamma_local * l_local)
        loss = poisson_loss + reg_term
        poisson_last = poisson_loss.detach()
        sparse_last = l_sparse.detach()
        local_last = l_local.detach()
        reg_last = reg_term.detach()
        losses.append(loss.item())
        
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()


        with torch.no_grad():
            train_agg(out)
    
    for batch in val_loader:
        optimizer.eval()
        with torch.no_grad():
            batch = {k: v.cuda() for k, v in batch.items()}
            batch['stim'] = batch['stim'][:, :, peak_lags[cell_ids], crop_size:-crop_size, crop_size:-crop_size]
            out = model(batch)
            out['robs'] = out['robs'][:, cell_ids]
            out['dfs'] = out['dfs'][:, cell_ids]
            assert out['dfs'].shape == out['rhat'].shape == out['robs'].shape
            val_agg(out)
    bps = train_agg.closure().cpu().numpy()
    bps_val = val_agg.closure().cpu().numpy()

    # plt.plot(losses)
    # plt.show()
    pos = model.positive_afferent_map[0, 0]   # (height, order+1, H, W)
    neg = model.negative_afferent_map[0, 0]
    fig, axes = visualize_afferent_map(pos, neg, title=f"Cell {cell_ids[0]}")
    plt.show()
    sta_img = stas[cell_ids[0], peak_lags[cell_ids[0]]]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(model.linear_receptive_field[0, 0].detach().cpu().numpy(), cmap='coolwarm_r')
    axes[0].set_title("Linear RF")
    axes[0].axis('off')
    axes[1].imshow(sta_img, cmap='coolwarm_r')
    axes[1].set_title(f"STA (cell {cell_ids[0]})")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()
    locality_factor = gamma_local * local_last.item()
    print(
        f"poisson={poisson_last.item():.6f}, L_sparse={sparse_last.item():.6f}, "
        f"L_local={local_last.item():.6f}, gamma*L_local={locality_factor:.6f} "
        f"({100.0 * locality_factor:.2f}%), reg={reg_last.item():.6f}"
    )
    print(bps.item())
    print(bps_val.item())

#%%

# %%
