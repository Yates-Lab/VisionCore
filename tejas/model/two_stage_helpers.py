import matplotlib.pyplot as plt
import numpy as np
import torch


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
        sep = lobe_thick * 1  # touching / slight overlap for visibility
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
