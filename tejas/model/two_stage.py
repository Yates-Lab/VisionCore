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
stas = calc_sta(stim.detach().cpu().squeeze()[:, 0, 10:-10, 10:-10],
                robs.cpu(),
                range(n_lags),
                dfs=dfs.cpu().squeeze(),
                progress=True).cpu().squeeze().numpy()

# # Calculate spike-triggered second moments (STEs)
# # Uses squared stimulus values via stim_modifier
stes = calc_sta(stim.detach().cpu().squeeze()[:, 0, 10:-10, 10:-10],
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from plenoptic.simulate import SteerablePyramidFreq
import plenoptic as po
class TwoStage(nn.Module):
    def __init__(self, image_shape, n_neurons, n_lags=1, height=3, order=5):
        super().__init__()
        self.pyr = SteerablePyramidFreq(image_shape, height=height, order=order,
                                         is_complex=False, downsample=False)
        self.n_neurons = n_neurons
        self.image_shape = image_shape
        self.n_lags = n_lags
        self.height = height
        self.order = order
        n_bands = height * (order + 1)  # 18
        n_feat_per_half = n_bands * n_lags * image_shape[0] * image_shape[1]
        self.w_pos = nn.Linear(n_feat_per_half, n_neurons, bias=False)
        self.w_neg = nn.Linear(n_feat_per_half, n_neurons, bias=False)
        self.alpha_pos = nn.Parameter(torch.ones(n_neurons))
        self.alpha_neg = nn.Parameter(torch.ones(n_neurons))
        self.beta = nn.Parameter(torch.zeros(n_neurons))
    
    @property
    def positive_afferent_map(self):
        '''
        Returns the positive afferent map as a tensor of shape (n_neurons, n_lags, height, order+1, *image_shape)
        '''
        return self.w_pos.weight.reshape(self.n_neurons, self.n_lags, self.height, self.order+1, *self.image_shape)
    
    @property
    def negative_afferent_map(self):
        '''
        Returns the negative afferent map as a tensor of shape (n_neurons, n_lags, height, order+1, *image_shape)
        '''
        return self.w_neg.weight.reshape(self.n_neurons, self.n_lags, self.height, self.order+1, *self.image_shape)
    
    def get_pyr_feats(self, x):
        with torch.no_grad():
            s = x['stim'][:, 0]  # [B, 1, lags, H, W] -> [B, lags, H, W]
            B, L, H, W = s.shape
            pyr_out = self.pyr(s.reshape(B * L, 1, H, W))
            pos_feats, neg_feats = [], []
            for k, v in pyr_out.items():
                if isinstance(k, tuple):
                    pos_feats.append(F.relu(v).reshape(B, -1))
                    neg_feats.append(F.relu(-v).reshape(B, -1))

            pos_feats = torch.cat(pos_feats, dim=-1)
            neg_feats = torch.cat(neg_feats, dim=-1)

            return pos_feats, neg_feats, pyr_out

    def forward(self, x):
        pos_feats, neg_feats, _ = self.get_pyr_feats(x)
        
        z = self.w_pos(pos_feats) + self.w_neg(neg_feats)
        # print(z.shape, pos_feats.shape, neg_feats.shape)

        x['rhat'] = (self.beta + self.alpha_pos * F.relu(z) + self.alpha_neg * F.relu(-z)).clamp(min=1e-6)
        return x


def sparsity_penalty(model):
    w_star = torch.sqrt(model.w_pos.weight**2 + model.w_neg.weight**2)
    return w_star.norm(1) / w_star.norm(2)




spike_loss = MaskedPoissonNLLLoss(pred_key='rhat', target_key='robs', mask_key='dfs')
# n_lags = len(dataset_config['keys_lags']['stim'])
n_lags = 1
# image_shape = train_dset[0]['stim'].shape[2:]
image_shape = (41, 41)
# num_neurons = len(dataset_config['cids'])
num_neurons = 1
model = TwoStage(image_shape=image_shape, n_neurons=num_neurons, n_lags=n_lags, height=3, order=5)

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
lambda_sparsity = 1e-2
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
        sparsity_loss = lambda_sparsity * sparsity_penalty(model)
        loss = poisson_loss + sparsity_loss
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
    fig, axes = visualize_afferent_map(pos, neg, title="Cell 14")
    plt.show()
    print(bps.item())
    print(bps_val.item())
    

#%%
def visualize_afferent_map(positive_afferent_map, negative_afferent_map, figsize=None, title=None, eps=1e-8):
    """Visualize 4D afferent maps (height, order+1, H, W) with hue = on/off proportion, saturation = amplitude w*."""
    import matplotlib.colors as mcolors
    w_plus = positive_afferent_map.detach().cpu().numpy() if torch.is_tensor(positive_afferent_map) else np.asarray(positive_afferent_map)
    w_minus = negative_afferent_map.detach().cpu().numpy() if torch.is_tensor(negative_afferent_map) else np.asarray(negative_afferent_map)
    assert w_plus.ndim == 4 and w_minus.ndim == 4 and w_plus.shape == w_minus.shape, "Expected 4D (height, order+1, H, W), same shape"
    height, n_orient, H, W = w_plus.shape
    w_star = np.sqrt(w_plus**2 + w_minus**2)
    w_max = w_star.max() + eps
    sat = np.clip(w_star / w_max, 0, 1)
    angle = np.arctan2(w_minus, w_plus)
    hue = (angle + np.pi) / (2 * np.pi) % 1.0
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


# %%
