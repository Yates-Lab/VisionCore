import matplotlib.pyplot as plt
import numpy as np
import torch


def afferent_hue_from_signed_maps(w_plus, w_minus):
    """Map signed on/off responses to hue used in afferent visualizations."""
    angle = np.arctan2(w_minus, w_plus)
    # Keep exact mapping used by visualize_afferent_map.
    return (1.0 / 3.0 - angle / (2 * np.pi)) % 1.0


def afferent_saturation_from_signed_maps(w_plus, w_minus, eps=1e-8):
    """Map afferent magnitude to saturation in [0, 1]."""
    w_star = np.sqrt(w_plus**2 + w_minus**2)
    w_max = w_star.max() + eps
    return np.clip(w_star / w_max, 0, 1)


def afferent_rgb_from_signed_maps(w_plus, w_minus, eps=1e-8):
    """Render afferent map RGB using shared hue/saturation logic."""
    import matplotlib.colors as mcolors

    hue = afferent_hue_from_signed_maps(w_plus, w_minus)
    sat = afferent_saturation_from_signed_maps(w_plus, w_minus, eps=eps)
    val = np.ones_like(hue)
    hsv = np.stack([hue, sat, val], axis=-1)
    return mcolors.hsv_to_rgb(hsv)


def draw_afferent_colorwheel(
    ax=None,
    resolution=401,
    add_labels=True,
    title=None,
    background=0.90,
    axis_limit=1.35,
):
    """
    Draw the afferent-map colorwheel used by visualize_afferent_map.

    If `ax` is None, creates and returns a standalone figure+axis.
    If `ax` is provided, draws in-place and returns (None, ax).
    """
    import matplotlib.colors as mcolors

    created_fig = False
    if ax is None:
        created_fig = True
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
    else:
        fig = None

    lin = np.linspace(-1.0, 1.0, int(resolution), dtype=float)
    xx, yy = np.meshgrid(lin, lin)
    rr = np.sqrt(xx**2 + yy**2)
    wheel_mask = rr <= 1.0

    hue = afferent_hue_from_signed_maps(xx, yy)
    sat = np.clip(rr, 0.0, 1.0)
    val = np.ones_like(sat)
    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)

    bg = np.array([background, background, background], dtype=float)
    rgb[~wheel_mask] = bg
    ax.imshow(rgb, origin="lower", extent=(-1.0, 1.0, -1.0, 1.0), interpolation="nearest")

    # Dashed cardinal axes through colorwheel.
    ax.plot([-axis_limit, axis_limit], [0, 0], color="k", linestyle=(0, (4, 4)), linewidth=1.8, alpha=0.85)
    ax.plot([0, 0], [-axis_limit, axis_limit], color="k", linestyle=(0, (4, 4)), linewidth=1.8, alpha=0.85)

    # Arrowheads at axis ends.
    ah = dict(arrowstyle="-|>", color="k", linewidth=1.6, mutation_scale=12)
    ax.annotate("", xy=(axis_limit, 0), xytext=(axis_limit - 0.20, 0), arrowprops=ah)
    ax.annotate("", xy=(-axis_limit, 0), xytext=(-axis_limit + 0.20, 0), arrowprops=ah)
    ax.annotate("", xy=(0, axis_limit), xytext=(0, axis_limit - 0.20), arrowprops=ah)
    ax.annotate("", xy=(0, -axis_limit), xytext=(0, -axis_limit + 0.20), arrowprops=ah)

    if add_labels:
        ax.text(0.0, axis_limit + 0.09, "'Off' excitation", ha="center", va="bottom", fontsize=12, style="italic")
        ax.text(axis_limit + 0.09, 0.0, "'On' excitation", ha="left", va="center", fontsize=12, style="italic")
        ax.text(0.0, -axis_limit - 0.09, "'Off' inhibition", ha="center", va="top", fontsize=12, style="italic")
        ax.text(-axis_limit - 0.09, 0.0, "'On' inhibition", ha="right", va="center", fontsize=12, style="italic")

    if title:
        ax.set_title(title)
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect("equal")
    ax.axis("off")

    if created_fig:
        return fig, ax
    return None, ax


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


def visualize_afferent_map(
    model,
    figsize=None,
    title=None,
    eps=1e-8,
    show_examples=True,
    show_colorwheel=False,
    colorwheel_kwargs=None,
    neuron_idx=0,
    lag_idx=0,
):
    """Visualize model afferent maps with hue=on/off proportion and saturation=amplitude."""
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

    w_plus = model.positive_afferent_map[neuron_idx, lag_idx].detach().cpu().numpy()
    w_minus = model.negative_afferent_map[neuron_idx, lag_idx].detach().cpu().numpy()
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
    rgb = afferent_rgb_from_signed_maps(w_plus, w_minus, eps=eps)
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
    right_edge = 0.82 if show_colorwheel else 0.98
    plt.tight_layout(rect=(0.16, 0.14, right_edge, 0.95))

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

    if show_colorwheel:
        # Anchor colorwheel to afferent map panel bounds.
        positions = [axes[i, j].get_position() for i in range(height) for j in range(n_orient)]
        x1 = max(p.x1 for p in positions)
        y0 = min(p.y0 for p in positions)
        y1 = max(p.y1 for p in positions)
        panel_h = y1 - y0
        wheel_h = panel_h * 0.82
        wheel_w = 0.14
        wheel_x0 = min(0.99 - wheel_w, x1 + 0.02)
        wheel_y0 = y0 + (panel_h - wheel_h) * 0.5
        cax = fig.add_axes([wheel_x0, wheel_y0, wheel_w, wheel_h])
        _wheel_defaults = dict(add_labels=True, axis_limit=1.30)
        _wheel_defaults.update(colorwheel_kwargs or {})
        draw_afferent_colorwheel(ax=cax, **_wheel_defaults)

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


def show_epoch_diagnostics(
    model,
    stas,
    peak_lags,
    cell_id,
    sparsity_mode,
    poisson_last,
    sparse_last,
    local_last,
    gamma_local,
    prox_tau_last,
    reg_last,
    bps,
    bps_val,
    phase=None,
    epoch=None,
    show_colorwheel=True,
    colorwheel_kwargs=None,
    neuron_idx=0,
    lag_idx=0,
    save_dir=None,
    save_prefix=None,
    close_figs=False,
    show_plots=True,
):
    bps_np = np.asarray(bps.detach().cpu().numpy() if torch.is_tensor(bps) else bps).reshape(-1)
    bps_val_np = np.asarray(bps_val.detach().cpu().numpy() if torch.is_tensor(bps_val) else bps_val).reshape(-1)
    if 0 <= int(neuron_idx) < bps_np.size:
        bps_cell = float(bps_np[int(neuron_idx)])
        bps_val_cell = float(bps_val_np[int(neuron_idx)])
    else:
        bps_cell = float(bps_np.mean()) if bps_np.size else float("nan")
        bps_val_cell = float(bps_val_np.mean()) if bps_val_np.size else float("nan")

    aff_title = f"Cell {cell_id} | train_bps={bps_cell:.3f} val_bps={bps_val_cell:.3f}"
    if phase is not None and epoch is not None:
        aff_title = f"{aff_title} | {phase} e{int(epoch):03d}"
    elif phase is not None:
        aff_title = f"{aff_title} | {phase}"

    aff_fig, axes = visualize_afferent_map(
        model,
        title=aff_title,
        show_colorwheel=show_colorwheel,
        colorwheel_kwargs=colorwheel_kwargs,
        neuron_idx=neuron_idx,
        lag_idx=lag_idx,
    )
    if show_plots:
        plt.show()
    sta_img = stas[cell_id, peak_lags[cell_id]]
    energy_exc_rf, energy_inh_rf = model.energy_receptive_fields_at(
        neuron_idx=neuron_idx, lag_idx=lag_idx
    )
    energy_exc_np = energy_exc_rf[0, 0].detach().cpu().numpy()
    energy_inh_np = energy_inh_rf[0, 0].detach().cpu().numpy()
    joint_abs = np.concatenate(
        [np.abs(energy_exc_np).reshape(-1), np.abs(energy_inh_np).reshape(-1)]
    )
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
    rf_fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(
        model.linear_receptive_field_at(neuron_idx=neuron_idx, lag_idx=lag_idx)[0, 0]
        .detach()
        .cpu()
        .numpy(),
        cmap="coolwarm_r",
    )
    axes[0].set_title("Linear RF")
    axes[0].axis("off")
    axes[1].imshow(exc_rgb)
    axes[1].set_title("Energy Exc RF")
    axes[1].axis("off")
    axes[2].imshow(inh_rgb)
    axes[2].set_title("Energy Inh RF")
    axes[2].axis("off")
    axes[3].imshow(sta_img, cmap="coolwarm_r")
    axes[3].set_title(f"STA (cell {cell_id})")
    axes[3].axis("off")
    rf_fig.suptitle(f"train_bps={bps_cell:.3f} | val_bps={bps_val_cell:.3f}", y=0.98)
    plt.tight_layout()
    if save_dir is not None:
        import os

        def _fig_to_rgb_arr(fig):
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
            return rgba[..., :3]

        prefix = save_prefix or f"cell_{cell_id}"
        os.makedirs(save_dir, exist_ok=True)
        aff_img = _fig_to_rgb_arr(aff_fig)
        rf_img = _fig_to_rgb_arr(rf_fig)

        pad = 16
        out_w = max(aff_img.shape[1], rf_img.shape[1])

        def _pad_to_width(img, width):
            if img.shape[1] == width:
                return img
            left = (width - img.shape[1]) // 2
            right = width - img.shape[1] - left
            return np.pad(
                img,
                ((0, 0), (left, right), (0, 0)),
                mode="constant",
                constant_values=255,
            )

        aff_pad = _pad_to_width(aff_img, out_w)
        rf_pad = _pad_to_width(rf_img, out_w)
        gap = np.full((pad, out_w, 3), 255, dtype=np.uint8)
        combined = np.concatenate([aff_pad, gap, rf_pad], axis=0)
        out_path = os.path.join(save_dir, f"{prefix}_diagnostics.png")
        plt.imsave(out_path, combined)
    if show_plots:
        plt.show()
    if close_figs:
        plt.close("all")

    locality_factor = gamma_local * local_last
    print(
        f"mode={sparsity_mode}, poisson={poisson_last:.6f}, "
        f"L_sparse={sparse_last:.6f}, L_local={local_last:.6f}, "
        f"gamma*L_local={locality_factor:.6f} ({100.0 * locality_factor:.2f}%), "
        f"prox_tau={prox_tau_last:.6e}, reg={reg_last:.6f}"
    )
    print("beta:", model.beta.detach().cpu().numpy().tolist())
    if bps_np.size == 1:
        print(float(bps_np[0]))
    else:
        print(np.round(bps_np, 4).tolist())
    if bps_val_np.size == 1:
        print(float(bps_val_np[0]))
    else:
        print(np.round(bps_val_np, 4).tolist())
    if phase is not None and epoch is not None:
        print(f"phase={phase}, epoch={epoch}")
    elif phase is not None:
        print(f"phase={phase}")


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
