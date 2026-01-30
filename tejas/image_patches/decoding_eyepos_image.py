#%%
import os
from pathlib import Path
# Device options
use_gpu = True
gpu_index = 0
if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
# from DataYatesV1.models.config_loader import load_dataset_configs
# from DataYatesV1.utils.data import prepare_data
from models.config_loader import load_dataset_configs
from models.data import prepare_data
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from DataYatesV1 import  get_complete_sessions
import matplotlib.patheffects as pe 
from DataYatesV1.exp.fix_rsvp import FixRsvpTrial
from DataYatesV1.utils.general import get_clock_functions
import torch
import torch.nn as nn
import numpy as np
import plenoptic as po
from plenoptic.simulate import SteerablePyramidFreq
from tqdm import tqdm

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['pdf.compression'] = 0
mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.resample'] = False
import contextlib
import schedulefree

from scripts.mcfarland_sim import get_fixrsvp_stack
from DataYatesV1.exp.support import get_rsvp_fix_stim

# support_images = get_rsvp_fix_stim()
# stack_images = get_fixrsvp_stack()
#%%
rsvp_images = torch.from_numpy(get_fixrsvp_stack(frames_per_im=1))
ppd = 37.50476617
#get the central 2.5 degrees of the image
window_size_pixels = 2 * ppd
start_x = int(rsvp_images.shape[1] // 2 - window_size_pixels // 2)
end_x = int(rsvp_images.shape[1] // 2 + window_size_pixels // 2)
start_y = int(rsvp_images.shape[2] // 2 - window_size_pixels // 2)
end_y = int(rsvp_images.shape[2] // 2 + window_size_pixels // 2)
rsvp_images_cropped = rsvp_images[:, start_x:end_x, start_y:end_y]
#%%
rsvp_images_cropped_compare = rsvp_images_cropped[:20]

#%%
import torch
import torch.nn.functional as F
from tqdm import tqdm

import torch
import torch.nn.functional as F
from tqdm import tqdm

def topk_similar_patches_torch_gpu(
    imgs: torch.Tensor,          # [B,H,W] on GPU
    patch: int = 30,
    k: int = 25,
    proj_dim: int = None,        # set None to disable
    query_chunk: int = 1024,
    key_chunk: int = 65536,
    drop_self: bool = True,
    pad_mode: str = "reflect",
    eps: float = 1e-6,
    seed: int = 0,
    spatial_order: str = "yx",   # "yx" (pos=y*W+x) or "xy" (pos=x*H+y)
):
    """
    Exact top-k NCC-like similarity using torch on GPU (chunked).
    Returns nb_b, nb_y, nb_x, sims with shape [B,H,W,k]
    """
    assert imgs.ndim == 3 and imgs.is_cuda
    assert spatial_order in ("yx", "xy")
    B, H, W = imgs.shape
    device = imgs.device
    N = B * H * W
    D = patch * patch

    # ---- Extract patches ----
    x = imgs[:, None].float()
    pad = patch // 2
    x = F.pad(x, (pad, pad, pad, pad), mode=pad_mode)
    patches = F.unfold(x, kernel_size=patch, stride=1)      # [B,D,H*W]
    X = patches.permute(0, 2, 1).reshape(-1, D)            # [N,D]

    # ---- Normalize (zero-mean + L2) ----
    X = X - X.mean(dim=1, keepdim=True)
    X = X / (X.norm(dim=1, keepdim=True) + eps)

    # ---- Optional random projection ----
    if proj_dim is not None and proj_dim < D:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        R = torch.randn(D, proj_dim, device=device, generator=g) / (D ** 0.5)
        Z = X @ R
        Z = Z / (Z.norm(dim=1, keepdim=True) + eps)
    else:
        Z = X  # [N,d]

    out_ids  = torch.empty((N, k), device=device, dtype=torch.long)
    out_sims = torch.empty((N, k), device=device, dtype=torch.float32)

    for qs in tqdm(range(0, N, query_chunk), desc="torch topk (queries)", unit="chunk"):
        qe = min(qs + query_chunk, N)
        Q = Z[qs:qe]  # [q,d]

        best_sims = torch.full((qe - qs, k), -1e9, device=device)
        best_ids  = torch.full((qe - qs, k), -1, device=device, dtype=torch.long)

        for ks in range(0, N, key_chunk):
            ke = min(ks + key_chunk, N)
            K = Z[ks:ke]  # [m,d]
            sims = Q @ K.T

            if drop_self:
                q_ids = torch.arange(qs, qe, device=device)[:, None]
                k_ids = torch.arange(ks, ke, device=device)[None, :]
                sims = sims.masked_fill(q_ids == k_ids, -1e9)

            cand_sims = torch.cat([best_sims, sims], dim=1)
            cand_ids_block = torch.arange(ks, ke, device=device, dtype=torch.long)[None, :].expand(qe - qs, -1)
            cand_ids = torch.cat([best_ids, cand_ids_block], dim=1)

            best_sims, idx = torch.topk(cand_sims, k=k, dim=1)
            best_ids = torch.gather(cand_ids, 1, idx)

        out_sims[qs:qe] = best_sims
        out_ids[qs:qe]  = best_ids

    # ---- Reshape queries to [B,H,W,k] correctly depending on spatial order ----
    if spatial_order == "yx":
        ids4  = out_ids.view(B, H, W, k)
        sims4 = out_sims.view(B, H, W, k)
    else:  # spatial_order == "xy"
        # queries are laid out as [B, W, H, k] (x-major), then permute to [B,H,W,k]
        ids4  = out_ids.view(B, W, H, k).permute(0, 2, 1, 3).contiguous()
        sims4 = out_sims.view(B, W, H, k).permute(0, 2, 1, 3).contiguous()

    # ---- Decode neighbor ids -> neighbor (b,y,x) ----
    nb_b = ids4 // (H * W)
    r = ids4 % (H * W)

    if spatial_order == "yx":
        nb_y = r // W
        nb_x = r % W
    else:
        nb_x = r // H
        nb_y = r % H

    return nb_b, nb_y, nb_x, sims4
#%%

imgs = rsvp_images_cropped_compare.cuda()   # [20,74,74]
nb_b, nb_y, nb_x, sims = topk_similar_patches_torch_gpu(
    imgs,
    patch=10,
    k=10,
    proj_dim=None,
    query_chunk=512,
    key_chunk=32768,
    spatial_order="xy",   # <-- IMPORTANT
)
#%%
for order in ["yx", "xy"]:
    nb_b, nb_y, nb_x, sims = topk_similar_patches_torch_gpu(
        imgs, patch=10, k=10, proj_dim=None,
        query_chunk=512, key_chunk=32768,
        spatial_order=order
    )
    print("order =", order)
    verify_match(imgs, nb_b, nb_y, nb_x, sims, query_b=5, query_y=12, query_x=72, patch=10, rank=0)
#%%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def _crop_patch_reflect(img2d: torch.Tensor, y: int, x: int, patch: int) -> torch.Tensor:
    """
    img2d: [H,W] torch tensor (cpu or cuda)
    returns: [patch, patch] tensor
    Uses reflect padding so boundary pixels work.
    """
    H, W = img2d.shape
    pad = patch // 2
    # pad order for 2D via 4D: [1,1,H,W]
    t = img2d[None, None].float()
    t = F.pad(t, (pad, pad, pad, pad), mode="reflect")
    # shift coords by pad
    y0 = y
    x0 = x
    patch_t = t[0, 0, y0:y0 + patch, x0:x0 + patch]
    return patch_t

def show_topk_patch_matches(
    imgs: torch.Tensor,          # [B,H,W]
    nb_b: torch.Tensor,          # [B,H,W,k]
    nb_y: torch.Tensor,          # [B,H,W,k]
    nb_x: torch.Tensor,          # [B,H,W,k]
    sims: torch.Tensor,          # [B,H,W,k]
    query_b: int,
    query_y: int,
    query_x: int,
    patch: int,
    topn: int = 10,
    show_full_images: bool = False,
    vmax_mode: str = "per_patch",   # "per_patch" or "global"
):
    """
    Displays query patch and top-n matched patches.
    Optionally shows the full images with markers for query and matches.

    vmax_mode:
      - "per_patch": each patch auto-scales (best for structure)
      - "global": use same vmin/vmax across all shown patches (best for intensity comparison)
    """
    assert imgs.ndim == 3, f"imgs expected [B,H,W], got {imgs.shape}"
    assert nb_b.ndim == 4 and sims.ndim == 4, "neighbors/sims expected [B,H,W,k]"
    B, H, W = imgs.shape
    k = sims.shape[-1]
    topn = min(topn, k)

    # Move minimal stuff to CPU for plotting
    imgs_cpu = imgs.detach().cpu()
    nb_b_q = nb_b[query_b, query_y, query_x, :topn].detach().cpu()
    nb_y_q = nb_y[query_b, query_y, query_x, :topn].detach().cpu()
    nb_x_q = nb_x[query_b, query_y, query_x, :topn].detach().cpu()
    sims_q = sims[query_b, query_y, query_x, :topn].detach().cpu()

    # Build patches
    q_patch = _crop_patch_reflect(imgs_cpu[query_b], query_y, query_x, patch)

    patches = []
    meta = []
    for i in range(topn):
        b_i = int(nb_b_q[i])
        y_i = int(nb_y_q[i])
        x_i = int(nb_x_q[i])
        s_i = float(sims_q[i])
        p_i = _crop_patch_reflect(imgs_cpu[b_i], y_i, x_i, patch)
        patches.append(p_i)
        meta.append((b_i, y_i, x_i, s_i))

    # Decide intensity scaling for patch grid
    if vmax_mode == "global":
        all_vals = torch.stack([q_patch] + patches, dim=0)
        vmin = float(all_vals.min())
        vmax = float(all_vals.max())
    else:
        vmin = vmax = None  # per-patch autoscale

    # ---- Plot: query + topn ----
    cols = min(6, topn + 1)
    rows = (topn + 1 + cols - 1) // cols
    fig = plt.figure(figsize=(3 * cols, 3 * rows))
    ax0 = plt.subplot(rows, cols, 1)
    ax0.imshow(q_patch.numpy(), cmap="gray", vmin=vmin, vmax=vmax)
    ax0.set_title(f"Query (b,y,x)=({query_b},{query_y},{query_x})")
    ax0.axis("off")

    for i in range(topn):
        ax = plt.subplot(rows, cols, i + 2)
        ax.imshow(patches[i].numpy(), cmap="gray", vmin=vmin, vmax=vmax)
        b_i, y_i, x_i, s_i = meta[i]
        ax.set_title(f"#{i+1} sim={s_i:.3f}\n({b_i},{y_i},{x_i})")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # ---- Optional: show full images with markers ----
    if show_full_images:
        # Show query image + up to first 5 match images (or fewer if duplicates)
        show_m = min(5, topn)
        fig2 = plt.figure(figsize=(4 * (1 + show_m), 4))
        axq = plt.subplot(1, 1 + show_m, 1)
        axq.imshow(imgs_cpu[query_b].numpy(), cmap="gray")
        axq.scatter([query_x], [query_y], s=60)  # marker
        axq.set_title(f"Query image b={query_b}")
        axq.axis("off")

        for i in range(show_m):
            b_i, y_i, x_i, s_i = meta[i]
            ax = plt.subplot(1, 1 + show_m, i + 2)
            ax.imshow(imgs_cpu[b_i].numpy(), cmap="gray")
            ax.scatter([x_i], [y_i], s=60)
            ax.set_title(f"#{i+1} b={b_i}\nsim={s_i:.3f}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

#%%

show_topk_patch_matches(
    imgs, nb_b, nb_y, nb_x, sims,
    query_b=5, query_y=12, query_x=72,
    patch=10, topn=10,
    show_full_images=True,
    vmax_mode="global",   # recommended for sanity
)

#%%
def best_pixel_by_topk(
    sims: torch.Tensor,            # [B,H,W,k]
    nb_b: torch.Tensor = None,     # [B,H,W,k]
    mode: str = "mean",            # "mean" | "kth" | "count"
    threshold: float = 0.8,
    exclude_same_image: bool = False,
    min_valid_neighbors: int = 1,  # <- new
):
    assert sims.ndim == 4
    B, H, W, k = sims.shape
    device = sims.device

    if exclude_same_image:
        assert nb_b is not None and nb_b.shape == sims.shape
        q_b = torch.arange(B, device=device)[:, None, None, None]
        mask = (nb_b != q_b)
        sims_m = sims.masked_fill(~mask, -1e9)
        valid = mask.sum(dim=-1)  # [B,H,W]

        if mode == "mean":
            denom = valid.clamp(min=1)
            sims_sum = sims_m.masked_fill(sims_m < -1e8, 0.0).sum(dim=-1)
            score = sims_sum / denom
        elif mode == "kth":
            score = sims_m.min(dim=-1).values
        elif mode == "count":
            score = ((sims_m > threshold) & mask).sum(dim=-1).float()
        else:
            raise ValueError("mode must be: mean | kth | count")

        # guard: require at least min_valid_neighbors
        score = score.masked_fill(valid < min_valid_neighbors, -1e9)

    else:
        if mode == "mean":
            score = sims.mean(dim=-1)
        elif mode == "kth":
            score = sims.min(dim=-1).values
        elif mode == "count":
            score = (sims > threshold).sum(dim=-1).float()
        else:
            raise ValueError("mode must be: mean | kth | count")

    flat_idx = torch.argmax(score).item()
    b = flat_idx // (H * W)
    r = flat_idx % (H * W)
    y = r // W
    x = r % W
    return (b, y, x), score

(best_b, best_y, best_x), score_map = best_pixel_by_topk(
    sims, nb_b=nb_b, mode="mean",
    exclude_same_image=True,
    min_valid_neighbors=5
)
show_topk_patch_matches(
    imgs, nb_b, nb_y, nb_x, sims,
    query_b=best_b, query_y=best_y, query_x=best_x,
    patch=10, topn=5, show_full_images=True,
    vmax_mode="global",
)

#%%
import torch
import torch.nn.functional as F

def _extract_patch_like_unfold(img2d: torch.Tensor, y: int, x: int, patch: int, pad_mode="reflect"):
    """
    Matches the patch definition used by F.pad(...)+F.unfold(...):
    pad by patch//2 and take window starting at (y,x) in the padded image.
    """
    pad = patch // 2
    t = img2d[None, None].float()
    t = F.pad(t, (pad, pad, pad, pad), mode=pad_mode)
    return t[0, 0, y:y+patch, x:x+patch].contiguous()

def _ncc_like_sim(p: torch.Tensor, q: torch.Tensor, eps=1e-6):
    p = p.flatten()
    q = q.flatten()
    p = p - p.mean()
    q = q - q.mean()
    p = p / (p.norm() + eps)
    q = q / (q.norm() + eps)
    return (p * q).sum()

def verify_match(
    imgs: torch.Tensor, nb_b, nb_y, nb_x, sims,
    query_b: int, query_y: int, query_x: int,
    patch: int,
    rank: int = 0,
    pad_mode: str = "reflect",
):
    """
    Checks whether reported sims match direct recomputation for a given neighbor rank.
    """
    imgs_cpu = imgs.detach().cpu()

    b2 = int(nb_b[query_b, query_y, query_x, rank].item())
    y2 = int(nb_y[query_b, query_y, query_x, rank].item())
    x2 = int(nb_x[query_b, query_y, query_x, rank].item())

    reported = float(sims[query_b, query_y, query_x, rank].item())

    p1 = _extract_patch_like_unfold(imgs_cpu[query_b], query_y, query_x, patch, pad_mode=pad_mode)
    p2 = _extract_patch_like_unfold(imgs_cpu[b2], y2, x2, patch, pad_mode=pad_mode)

    direct = float(_ncc_like_sim(p1, p2).item())

    print(f"Query (b,y,x)=({query_b},{query_y},{query_x})")
    print(f"Neighbor rank {rank}: (b,y,x)=({b2},{y2},{x2})")
    print(f"Reported sim = {reported:.8f}")
    print(f"Direct   sim = {direct:.8f}")
    print(f"Abs diff     = {abs(reported-direct):.3e}")
verify_match(imgs, nb_b, nb_y, nb_x, sims, query_b=5, query_y=12, query_x=72, patch=10, rank=0)
verify_match(imgs, nb_b, nb_y, nb_x, sims, query_b=5, query_y=12, query_x=72, patch=10, rank=5)
