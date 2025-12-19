
#%% Generic Imports

# this is to suppress errors in the attempts at compilation that happen for one of the loaded models because it crashed
import sys
sys.path.append('..')
import numpy as np

from DataYatesV1 import enable_autoreload, get_free_device, get_session, get_complete_sessions
from models.data import prepare_data
from models.config_loader import load_dataset_configs

import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
import torch.nn.functional as F

# embed TrueType fonts in PDF/PS
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

# (optional) pick a clean sans‐serif
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

enable_autoreload()

device = get_free_device()

import torch

enable_autoreload()
device = get_free_device()

#%% Utilities


#-----------
# Get Stimuli
#-----------
def get_fixrsvp_stack(full_size=600, frames_per_im=3):
    from DataYatesV1.exp.fix_rsvp import get_rsvp_fix_stim

    images = get_rsvp_fix_stim()
    im_size = images['im01'].shape
    window = np.hanning(im_size[0])[:, None] * np.hanning(im_size[1])[None, :]
    num_images = 60
    center_pix = full_size // 2 - im_size[0] // 2
    full_stack = np.zeros((num_images*frames_per_im, full_size, full_size))

    frame_counter = 0
    for im_id in range(60):
        im = images[f'im{im_id+1:02d}'].mean(axis=2).astype(np.float32)
        im -= 127
        im /= 255
        # place in center
        for _ in range(frames_per_im):
            full_stack[frame_counter, center_pix:center_pix+im_size[0], center_pix:center_pix+im_size[1]] = window * im
            frame_counter += 1

    return full_stack


def sample_fix_durations(n, dist="lognormal", mean=0.25, sigma=0.35, rng=None):
    """
    Sample fixation durations (seconds).
    dist:
      - "lognormal": mean is approx mean seconds, sigma is log-space std (heavy-tailed)
      - "gamma": mean=mean, sigma=std in seconds (converted to shape/scale)
    """
    rng = np.random.default_rng() if rng is None else rng

    if dist == "lognormal":
        # Convert desired mean (approx) to lognormal parameters.
        # For lognormal: E[T] = exp(mu + 0.5*sigma^2)
        mu = np.log(max(mean, 1e-9)) - 0.5 * sigma**2
        return rng.lognormal(mean=mu, sigma=sigma, size=n)

    if dist == "gamma":
        # Given mean and std (sigma), solve shape k and scale theta:
        # mean = k*theta, var = k*theta^2
        std = sigma
        var = std**2
        k = (mean**2) / max(var, 1e-12)
        theta = var / max(mean, 1e-12)
        return rng.gamma(shape=k, scale=theta, size=n)

    raise ValueError("dist must be 'lognormal' or 'gamma'.")

# ----------------------------
# Microsaccade main sequence
# ----------------------------
def main_sequence_peak_vel(amp_deg, v0=20.0, vmax=200.0, a0=1.0):
    """
    Simple saturating main sequence: v_peak = v0 + (vmax - v0)*(1 - exp(-amp/a0))
    Units: deg/s.
    Reasonable microsaccade regime: amp ~ 0.05–1 deg, v_peak ~ 20–200 deg/s.
    """
    amp_deg = np.asarray(amp_deg)
    return v0 + (vmax - v0) * (1.0 - np.exp(-amp_deg / max(a0, 1e-9)))

# ----------------------------
# Core simulator
# ----------------------------
def simulate_eye_trace(
    T_total=10.0,
    dt=0.001,
    x0=(0.0, 0.0),

    # Fixations (drift)
    fix_dist="lognormal",
    fix_mean=0.25,
    fix_spread=0.35,          # lognormal sigma (log-space) OR gamma std (sec)
    D=0.02,                   # diffusion constant in deg^2/s (position Brownian)

    # Microsaccades
    ms_rate_boost=1.0,        # optional multiplier on how often you microsaccade (via fixation durations)
    ms_dur_mean=0.020,        # seconds (typical micro-saccade 10–30 ms)
    ms_dur_jitter=0.005,      # seconds std
    ms_amp_dist="lognormal",
    ms_amp_mean=0.25,         # deg
    ms_amp_spread=0.45,       # lognormal sigma (log-space) or gamma std (deg) if using gamma
    ms_dir_kappa=0.0,         # von Mises concentration; 0 => uniform directions

    # Velocity profile shape
    profile_sigma_frac=0.18,  # Gaussian profile width as fraction of ms duration
    use_sigmoid_gate=True,
    gate_sharpness=10.0,      # higher => sharper onset/offset gating

    rng=None
):
    """
    Returns:
      t: (N,)
      pos: (N,2) in deg
      vel: (N,2) in deg/s
      state: (N,) int {0=fixation, 1=microsaccade}
    """
    rng = np.random.default_rng() if rng is None else rng
    n_steps = int(np.round(T_total / dt))
    t = np.arange(n_steps) * dt

    pos = np.zeros((n_steps, 2), dtype=float)
    vel = np.zeros((n_steps, 2), dtype=float)
    state = np.zeros(n_steps, dtype=int)

    pos[0] = np.array(x0, dtype=float)

    # Brownian motion in position: dx ~ N(0, 2D dt)
    # If you want to think "velocities", v ~ N(0, 2D/dt) so that dx = v*dt matches above.
    pos_noise_std = np.sqrt(2.0 * D * dt)

    i = 0
    while i < n_steps - 1:
        # --- Fixation segment ---
        Tf = sample_fix_durations(
            n=1, dist=fix_dist, mean=fix_mean / max(ms_rate_boost, 1e-9), sigma=fix_spread, rng=rng
        )[0]
        fix_len = max(1, int(np.round(Tf / dt)))
        j_end = min(n_steps, i + fix_len)

        # random-walk in position (drift)
        for k in range(i, j_end - 1):
            dpos = rng.normal(0.0, pos_noise_std, size=2)
            pos[k + 1] = pos[k] + dpos
            vel[k] = dpos / dt
            state[k] = 0

        i = j_end
        if i >= n_steps - 1:
            break

        # --- Microsaccade segment ---
        Ts = max(0.005, rng.normal(ms_dur_mean, ms_dur_jitter))
        ms_len = max(2, int(np.round(Ts / dt)))
        j_end = min(n_steps, i + ms_len)

        # amplitude (deg)
        if ms_amp_dist == "lognormal":
            # lognormal with approximate mean=ms_amp_mean (same conversion trick)
            mu = np.log(max(ms_amp_mean, 1e-9)) - 0.5 * ms_amp_spread**2
            amp = float(rng.lognormal(mu, ms_amp_spread))
        elif ms_amp_dist == "gamma":
            std = ms_amp_spread
            var = std**2
            kshape = (ms_amp_mean**2) / max(var, 1e-12)
            theta = var / max(ms_amp_mean, 1e-12)
            amp = float(rng.gamma(kshape, theta))
        else:
            raise ValueError("ms_amp_dist must be 'lognormal' or 'gamma'.")

        # direction
        if ms_dir_kappa <= 0:
            theta = rng.uniform(0, 2*np.pi)
        else:
            # von Mises around 0; rotate if you want a preferred direction
            theta = rng.vonmises(mu=0.0, kappa=ms_dir_kappa)
        disp = amp * np.array([np.cos(theta), np.sin(theta)], dtype=float)

        # build 1D speed profile, then apply to 2D direction
        nseg = j_end - i
        tt = np.arange(nseg) * dt
        Tseg = tt[-1] + dt

        # Gaussian bump centered mid-saccade
        sigma_t = max(profile_sigma_frac * Tseg, 1e-6)
        center = 0.5 * Tseg
        bump = np.exp(-0.5 * ((tt - center) / sigma_t)**2)

        if use_sigmoid_gate:
            # gate to enforce near-zero at boundaries
            # gate = sigmoid(rise) * sigmoid(fall)
            rise = 1.0 / (1.0 + np.exp(-gate_sharpness * (tt / Tseg - 0.15)))
            fall = 1.0 / (1.0 + np.exp(-gate_sharpness * (0.85 - tt / Tseg)))
            bump = bump * rise * fall

        # Convert bump to velocity so that integral equals displacement magnitude
        # Let v(t) = s * bump(t) along direction u; then ∫ v dt = s * ∫ bump dt = amp
        area = np.sum(bump) * dt
        u = disp / (np.linalg.norm(disp) + 1e-12)

        # Set peak velocity via main sequence, but still enforce exact displacement.
        v_peak_target = float(main_sequence_peak_vel(amp))
        bump_max = float(np.max(bump))
        s_from_peak = v_peak_target / max(bump_max, 1e-12)
        s_from_area = amp / max(area, 1e-12)

        # Blend: enforce displacement exactly, but nudge toward main-sequence peak.
        # If you want strict peak matching, set alpha=1 and accept slight disp error, or resample.
        alpha = 0.3
        s = (1 - alpha) * s_from_area + alpha * s_from_peak

        # Now *renormalize* to enforce displacement exactly:
        s = s * (amp / max(s * area, 1e-12))

        v_seg = (s * bump)[:, None] * u[None, :]  # (nseg,2)

        # integrate
        for k in range(nseg):
            idx = i + k
            if idx >= n_steps - 1:
                break
            vel[idx] = v_seg[k]
            pos[idx + 1] = pos[idx] + vel[idx] * dt
            state[idx] = 1

        i = j_end

    return t, pos, vel, state


def eye_deg_to_norm(
    eye_deg: torch.Tensor,   # (T,2) in degrees, (x_deg,y_deg), y positive UP
    ppd: float,              # pixels per degree
    img_size,                # (H,W)
):
    """
    Convert eye position from degrees (relative to image center)
    to grid_sample normalized coordinates [-1,1].

    Returns: (T,2) tensor (x_norm,y_norm)
    """
    H, W = img_size
    eye_deg = eye_deg.to(dtype=torch.float32)

    # degrees -> pixels
    x_pix = eye_deg[:, 0] * ppd
    y_pix = eye_deg[:, 1] * ppd

    # pixels -> normalized [-1,1]
    x_norm = 2.0 * x_pix / (W - 1)
    y_norm = -2.0 * y_pix / (H - 1)  # minus because grid_sample y goes down

    return torch.stack((x_norm, y_norm), dim=-1)

def eye_deg_to_pix(
    eye_deg: torch.Tensor,   # (T,2) in degrees, (x_deg,y_deg), y positive UP
    ppd: float,              # pixels per degree
):
    """
    Convert eye position from degrees to pixels (relative to image center).

    Returns: (T,2) tensor (x_pix,y_pix) where x,y are in pixels relative to center
    """
    eye_deg = eye_deg.to(dtype=torch.float32)

    # degrees -> pixels (keep relative to center)
    x_pix = eye_deg[:, 0] * ppd
    y_pix = eye_deg[:, 1] * ppd

    return torch.stack((x_pix, y_pix), dim=-1)

def shift_movie_with_eye(
    movie: torch.Tensor,          # (T,H,W) or (T,C,H,W)
    eye_xy: torch.Tensor,         # (T,2) in [-1,1], (x,y)
    out_size=(100, 100),          # (outH,outW)
    center=(0.0, 0.0),            # (cx,cy) in [-1,1]
    mode="bilinear",
    padding_mode="zeros",
    align_corners=True,
):
    """
    Returns an eye-shifted crop sampled from `movie` using grid_sample.

    Convention:
      - eye_xy[t] is the eye position in normalized coords [-1,1].
      - The returned movie is sampled around `center`, with the image shifted by -eye_xy
        (i.e., stabilizing the movie in eye-centered coordinates).
      - The output window spans from -outW/(2*W) to +outW/(2*W) in normalized coords
        (and similarly for height), so it represents the actual pixel extent.
    """
    if movie.dim() == 3:
        # (T,H,W) -> (T,1,H,W)
        movie = movie.unsqueeze(1)
        squeeze_C = True
    elif movie.dim() == 4:
        squeeze_C = False
    else:
        raise ValueError("movie must have shape (T,H,W) or (T,C,H,W)")

    T, C, H, W = movie.shape
    device = movie.device
    dtype = movie.dtype

    eye_xy = eye_xy.to(device=device, dtype=dtype)
    outH, outW = out_size
    cx, cy = center

    # Base sampling grid scaled by actual pixel dimensions
    # Grid spans from -outW/(2*W) to +outW/(2*W) in normalized coords
    x_extent = outW / W  # extent in normalized coords [-1,1]
    y_extent = outH / H

    ys = torch.linspace(-y_extent, y_extent, outH, device=device, dtype=dtype)
    xs = torch.linspace(-x_extent, x_extent, outW, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack((grid_x + cx, grid_y + cy), dim=-1).unsqueeze(0)  # (1,outH,outW,2)

    # Stabilize: sample from movie at (base_grid - eye_xy[t])
    # eye_xy: (T,2) -> (T,1,1,2) broadcast to (T,outH,outW,2)
    grid = base_grid - eye_xy.view(T, 1, 1, 2)

    # grid_sample expects input (N,C,H,W) and grid (N,outH,outW,2)
    out = F.grid_sample(
        movie,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )  # (T,C,outH,outW)

    if squeeze_C:
        out = out[:, 0]  # (T,outH,outW)

    return out


def save_eye_movies(
    full_stack: torch.Tensor,      # (T,H,W) full stimulus
    eye_movie: torch.Tensor,        # (T,outH,outW) shifted ROI
    eye_pos_pix: torch.Tensor,      # (T,2) eye position in pixels (x,y)
    save_prefix: str = "eye_movie",
    fps: int = 30,
    trail_length: int = 10,
    dot_size: int = 8,
    trail_alpha: float = 0.7,
):
    """
    Save two movies: overlay and ROI.

    Parameters:
    -----------
    full_stack : torch.Tensor (T,H,W)
        Full stimulus movie
    eye_movie : torch.Tensor (T,outH,outW)
        Eye-shifted ROI movie
    eye_pos_pix : torch.Tensor (T,2)
        Eye position in pixel coordinates (x,y) relative to image center
    save_prefix : str
        Prefix for output filenames (will create {prefix}_overlay.mp4 and {prefix}_roi.mp4)
    fps : int
        Frames per second for output videos
    trail_length : int
        Number of past positions to show as trail
    dot_size : int
        Size of the current position dot
    trail_alpha : float
        Transparency for the trail (0=transparent, 1=opaque)
    """
    from matplotlib.animation import FFMpegWriter
    import matplotlib.pyplot as plt
    import numpy as np

    # Convert to numpy
    full_stack_np = full_stack.cpu().numpy()
    eye_movie_np = eye_movie.cpu().numpy()
    eye_pos_pix_np = eye_pos_pix.cpu().numpy()

    T = full_stack_np.shape[0]
    H, W = full_stack_np.shape[1:3]

    # Normalize stimulus for display
    vmin, vmax = -1, 1

    # Convert eye position from center-relative to image coordinates
    # eye_pos_pix is (x,y) relative to center, convert to (row, col) in image coords
    eye_x = eye_pos_pix_np[:, 0] + W / 2  # x position in pixels
    eye_y = -eye_pos_pix_np[:, 1] + H / 2  # y position in pixels (flip y)

    # --- Movie 1: Overlay ---
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_aspect('equal')
    ax1.axis('off')

    writer1 = FFMpegWriter(fps=fps, codec='libx264',
                          extra_args=['-pix_fmt', 'yuv420p'],
                          bitrate=8000)

    with writer1.saving(fig1, f"{save_prefix}_overlay.mp4", dpi=100):
        for t in range(T):
            ax1.clear()
            ax1.imshow(full_stack_np[t], cmap='gray', vmin=vmin, vmax=vmax)
            ax1.axis('off')

            # Draw trail
            start_idx = max(0, t - trail_length)
            for i in range(start_idx, t):
                alpha = trail_alpha * (i - start_idx + 1) / trail_length
                ax1.plot(eye_x[i:i+2], eye_y[i:i+2],
                        color='cyan', linewidth=2, alpha=alpha)

            # Draw current position (brighter dot)
            ax1.plot(eye_x[t], eye_y[t], 'o',
                    color='cyan', markersize=dot_size,
                    markeredgecolor='white', markeredgewidth=1)

            ax1.set_xlim(0, W)
            ax1.set_ylim(H, 0)

            writer1.grab_frame()

    plt.close(fig1)
    print(f"Saved overlay movie to {save_prefix}_overlay.mp4")

    # --- Movie 2: ROI ---
    vmin_roi, vmax_roi = -1, 1

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.set_aspect('equal')
    ax2.axis('off')

    writer2 = FFMpegWriter(fps=fps, codec='libx264',
                          extra_args=['-pix_fmt', 'yuv420p'],
                          bitrate=8000)

    with writer2.saving(fig2, f"{save_prefix}_roi.mp4", dpi=100):
        for t in range(T):
            ax2.clear()
            ax2.imshow(eye_movie_np[t], cmap='gray', vmin=vmin_roi, vmax=vmax_roi)
            ax2.axis('off')
            ax2.set_title(f'Frame {t}/{T}', fontsize=10, color='white')

            writer2.grab_frame()

    plt.close(fig2)
    print(f"Saved ROI movie to {save_prefix}_roi.mp4")


def build_temporal_kernel(kernel_size=16, dt=1/240, tau_fast=0.008, tau_slow=0.022, a=0.9, n=2):
    """
    Build biphasic temporal kernel using difference of gamma functions.

    Parameters:
    -----------
    kernel_size : int
        Number of time points in kernel
    dt : float
        Time step in seconds
    tau_fast : float
        Fast gamma time constant (seconds)
    tau_slow : float
        Slow gamma time constant (seconds)
    a : float
        Amplitude ratio of slow to fast component
    n : int
        Gamma function order

    Returns:
    --------
    temporal_kernel : torch.Tensor (kernel_size,)
        Normalized biphasic temporal kernel
    """
    import math

    def gamma_kernel(t, tau, n=2):
        t = torch.clamp(t, min=0.)
        coef = 1.0 / (tau**n * math.gamma(n))
        return coef * (t**(n-1)) * torch.exp(-t/tau)

    t = torch.arange(kernel_size) * dt
    k = gamma_kernel(t, tau_fast, n) - a * gamma_kernel(t, tau_slow, n)
    # Enforce ~zero-mean (remove DC) and scale positive lobe to 1
    k = k - k.mean()
    pos_sum = k.clamp_min(0).sum().clamp_min(1e-12)
    temporal_kernel = k / pos_sum

    return temporal_kernel


def simulate_responses(
    n_trials=100,
    trial_duration=1.0,
    dt=1/240,
    ppd=60,

    # Eye position simulation parameters
    fix_dist="lognormal",
    fix_mean=0.2,
    fix_spread=0.45,
    D=0.001,
    ms_dur_mean=0.020,
    ms_dur_jitter=0.02,
    ms_amp_mean=0.3,
    ms_amp_spread=0.55,

    # Stimulus parameters
    stim_size=600,
    frames_per_im=8,

    # Pyramid parameters
    image_shape=(51, 51),
    num_ori=8,
    num_scales=3,

    # Neuron sampling parameters
    n_neurons=10,
    scales=None,  # list of scale indices to sample from, None = all
    orientations=None,  # list of orientation indices to sample from, None = all
    spatial_scatter_pix=0.0,  # std dev of spatial scatter in pixels
    nonlinearity='complex',  # 'complex' (abs), 'simple' (rectified real or imag), or fraction e.g. 0.5 = 50% complex, 50% simple
    mean_firing_rate=15.0,  # spikes per second

    # Temporal filtering
    include_temporal_filter=True,
    kernel_size=16,

    # Random seeds
    neuron_seed=42,
    eye_seed=None,  # None = random

    # Output
    verbose=True,
):
    """
    Simulate neural responses to fixRSVP stimulus with eye movements.

    Parameters:
    -----------
    nonlinearity : str or float
        If str: 'complex' (all neurons use amplitude) or 'simple' (all neurons use rectified real/imag)
        If float: fraction of complex cells (e.g., 0.7 = 70% complex, 30% simple)

    Returns:
    --------
    robs : np.ndarray (n_trials, n_time_bins, n_neurons)
        Simulated spike counts
    eyepos : np.ndarray (n_trials, n_time_bins, 2)
        Eye position in degrees (x, y)
    params : dict
        Dictionary of all parameters used for simulation
    """
    from plenoptic.simulate import SteerablePyramidFreq

    # Set up random number generators
    neuron_rng = np.random.default_rng(neuron_seed)
    if eye_seed is None:
        eye_seed = np.random.randint(0, 2**32 - 1)
    eye_rng = np.random.default_rng(eye_seed)

    # Parse nonlinearity specification
    if isinstance(nonlinearity, str):
        if nonlinearity not in ['complex', 'simple']:
            raise ValueError("nonlinearity must be 'complex', 'simple', or a float")
        nonlinearity_types = [nonlinearity] * n_neurons
        simple_phases = [None] * n_neurons  # not used for complex
    elif isinstance(nonlinearity, (float, int)):
        # Fraction of complex cells
        frac_complex = float(nonlinearity)
        if not 0 <= frac_complex <= 1:
            raise ValueError("nonlinearity fraction must be between 0 and 1")
        n_complex = int(np.round(frac_complex * n_neurons))
        nonlinearity_types = ['complex'] * n_complex + ['simple'] * (n_neurons - n_complex)
        # Shuffle to randomize order
        neuron_rng.shuffle(nonlinearity_types)
        simple_phases = [None] * n_neurons
    else:
        raise ValueError("nonlinearity must be str or float")

    # For simple cells, randomly assign real or imaginary phase
    for i in range(n_neurons):
        if nonlinearity_types[i] == 'simple':
            simple_phases[i] = neuron_rng.choice(['real', 'imag'])

    # Build steerable pyramid
    order = num_ori - 1
    pyr = SteerablePyramidFreq(
        image_shape, order=order, height=num_scales, is_complex=True,
        downsample=False, tight_frame=False
    )

    # Build temporal kernel if needed
    if include_temporal_filter:
        temporal_kernel = build_temporal_kernel(kernel_size=kernel_size, dt=dt)
    else:
        temporal_kernel = None
        kernel_size = 1

    # Get stimulus
    full_stack = get_fixrsvp_stack(full_size=stim_size, frames_per_im=frames_per_im)
    full_stack = torch.from_numpy(full_stack).float()

    # Determine number of time bins per trial
    n_time_bins = int(trial_duration / dt)

    # Sample neurons from pyramid
    if scales is None:
        scales = list(range(num_scales))
    if orientations is None:
        orientations = list(range(num_ori))

    # Create list of all possible (scale, orientation, y, x) combinations
    mid_y, mid_x = image_shape[0] // 2, image_shape[1] // 2
    neuron_locations = []
    for scale in scales:
        for ori in orientations:
            # Sample spatial locations with scatter
            for _ in range(n_neurons * 2):  # oversample to ensure we get enough
                if spatial_scatter_pix > 0:
                    dy = neuron_rng.normal(0, spatial_scatter_pix)
                    dx = neuron_rng.normal(0, spatial_scatter_pix)
                    y = int(np.round(mid_y + dy))
                    x = int(np.round(mid_x + dx))
                    # Clamp to valid range
                    y = np.clip(y, 0, image_shape[0] - 1)
                    x = np.clip(x, 0, image_shape[1] - 1)
                else:
                    y, x = mid_y, mid_x
                neuron_locations.append((scale, ori, y, x))

    # Sample n_neurons without replacement
    sampled_indices = neuron_rng.choice(len(neuron_locations), size=n_neurons, replace=False)
    sampled_neurons = [neuron_locations[i] for i in sampled_indices]

    if verbose:
        print(f"Simulating {n_trials} trials with {n_neurons} neurons")
        print(f"Neuron locations: {sampled_neurons}")

    # Initialize output arrays
    robs = np.full((n_trials, n_time_bins, n_neurons), np.nan, dtype=float)
    eyepos = np.full((n_trials, n_time_bins, 2), np.nan, dtype=float)

    # Simulate each trial
    for itrial in range(n_trials):
        if verbose and itrial % 10 == 0:
            print(f"Trial {itrial}/{n_trials}")

        # Simulate eye trace
        _, pos, _, _ = simulate_eye_trace(
            T_total=trial_duration,
            dt=dt,
            fix_dist=fix_dist,
            fix_mean=fix_mean,
            fix_spread=fix_spread,
            D=D,
            ms_dur_mean=ms_dur_mean,
            ms_dur_jitter=ms_dur_jitter,
            ms_amp_dist="lognormal",
            ms_amp_mean=ms_amp_mean,
            ms_amp_spread=ms_amp_spread,
            rng=eye_rng,
        )

        # Truncate to n_time_bins
        pos = pos[:n_time_bins]
        eyepos[itrial, :len(pos)] = pos

        # Determine how many frames we can actually use from the stimulus
        # (need to match eye position length)
        T_eye = len(pos)
        T_stim = full_stack.shape[0]

        # Use the minimum of the two, cycling stimulus if needed
        if T_eye > T_stim:
            # Repeat stimulus to cover eye trace
            n_repeats = int(np.ceil(T_eye / T_stim))
            stim_for_trial = full_stack.repeat(n_repeats, 1, 1)[:T_eye]
        else:
            stim_for_trial = full_stack[:T_eye]

        # Convert eye position to normalized coordinates
        eye_norm = eye_deg_to_norm(torch.from_numpy(pos), ppd, stim_for_trial.shape[1:3])

        # Get eye-shifted stimulus
        eye_movie = shift_movie_with_eye(
            stim_for_trial,
            eye_norm,
            out_size=image_shape,
            center=(0.0, 0.0),
            mode="bilinear"
        )  # (T_eye, H, W)

        # Add channel dimension for pyramid
        eye_movie = eye_movie.unsqueeze(1)  # (T, 1, H, W)

        # Get pyramid coefficients
        if include_temporal_filter:
            # Need to get lagged stimulus for temporal filtering
            T_trial = eye_movie.shape[0]
            pyr_responses = torch.zeros((T_trial, n_neurons), dtype=torch.float32)

            for t in range(kernel_size - 1, T_trial):
                # Get lagged frames
                stim_lagged = eye_movie[t - kernel_size + 1:t + 1]  # (kernel_size, 1, H, W)
                pyr_coeffs = pyr(stim_lagged)

                # Extract responses for each neuron
                for ineuron, (scale, ori, y, x) in enumerate(sampled_neurons):
                    coeff = pyr_coeffs[(scale, ori)][:, :, y, x].squeeze()  # (kernel_size,)
                    # Apply nonlinearity based on neuron type
                    nl_type = nonlinearity_types[ineuron]
                    if nl_type == 'complex':
                        coeff = torch.abs(coeff)
                    elif nl_type == 'simple':
                        # Rectified real or imaginary
                        if simple_phases[ineuron] == 'real':
                            coeff = torch.clamp(torch.real(coeff), min=0)
                        else:  # 'imag'
                            coeff = torch.clamp(torch.imag(coeff), min=0)
                    # Apply temporal filter
                    response = (coeff * temporal_kernel).sum()
                    # Ensure non-negative (temporal filter can make it negative)
                    pyr_responses[t, ineuron] = torch.clamp(response, min=0)
        else:
            # No temporal filtering
            pyr_coeffs = pyr(eye_movie)
            pyr_responses = torch.zeros((eye_movie.shape[0], n_neurons), dtype=torch.float32)

            for ineuron, (scale, ori, y, x) in enumerate(sampled_neurons):
                coeff = pyr_coeffs[(scale, ori)][:, :, y, x].squeeze()  # (T,)
                # Apply nonlinearity based on neuron type
                nl_type = nonlinearity_types[ineuron]
                if nl_type == 'complex':
                    coeff = torch.abs(coeff)
                elif nl_type == 'simple':
                    # Rectified real or imaginary
                    if simple_phases[ineuron] == 'real':
                        coeff = torch.clamp(torch.real(coeff), min=0)
                    else:  # 'imag'
                        coeff = torch.clamp(torch.imag(coeff), min=0)
                pyr_responses[:, ineuron] = coeff

        # Convert to firing rates and sample spikes
        pyr_responses = pyr_responses.numpy()
        for ineuron in range(n_neurons):
            r = pyr_responses[:, ineuron]
            # Ensure non-negative (should already be from nonlinearity, but double-check)
            r = np.maximum(r, 0)
            # Normalize to mean firing rate
            mu = np.nanmean(r)
            if mu > 0:
                r = r / mu * mean_firing_rate * dt
            # Sample Poisson spikes
            valid = np.isfinite(r) & (r >= 0)
            spikes = np.zeros_like(r)
            if np.any(valid):
                spikes[valid] = r[valid]
            robs[itrial, :len(spikes), ineuron] = spikes

    # Store parameters
    params = {
        'n_trials': n_trials,
        'trial_duration': trial_duration,
        'dt': dt,
        'ppd': ppd,
        'eye_params': {
            'fix_dist': fix_dist,
            'fix_mean': fix_mean,
            'fix_spread': fix_spread,
            'D': D,
            'ms_dur_mean': ms_dur_mean,
            'ms_dur_jitter': ms_dur_jitter,
            'ms_amp_mean': ms_amp_mean,
            'ms_amp_spread': ms_amp_spread,
        },
        'stim_params': {
            'stim_size': stim_size,
            'frames_per_im': frames_per_im,
        },
        'pyramid_params': {
            'image_shape': image_shape,
            'num_ori': num_ori,
            'num_scales': num_scales,
        },
        'neuron_params': {
            'n_neurons': n_neurons,
            'scales': scales,
            'orientations': orientations,
            'spatial_scatter_pix': spatial_scatter_pix,
            'nonlinearity': nonlinearity,
            'nonlinearity_types': nonlinearity_types,  # 'complex' or 'simple' for each neuron
            'simple_phases': simple_phases,  # 'real' or 'imag' for simple cells, None for complex
            'mean_firing_rate': mean_firing_rate,
            'sampled_neurons': sampled_neurons,
        },
        'temporal_params': {
            'include_temporal_filter': include_temporal_filter,
            'kernel_size': kernel_size,
        },
        'seeds': {
            'neuron_seed': neuron_seed,
            'eye_seed': eye_seed,
        },
    }

    return robs, eyepos, params


#%%
full_stack = get_fixrsvp_stack(frames_per_im=8)
full_stack = torch.from_numpy(full_stack)
ppd = 60
dt = 1/240
t, pos, vel, state = simulate_eye_trace(
        T_total=full_stack.shape[0]*dt, dt=dt,
        fix_dist="lognormal",
        fix_mean=0.2, fix_spread=0.45,
        D=0.001,
        ms_dur_mean=0.020, ms_dur_jitter=0.02,
        ms_amp_mean=0.3, ms_amp_spread=0.55,
        use_sigmoid_gate=True
    )

plt.plot(pos)
plt.ylim(-1, 1)

#%%


eye_norm = eye_deg_to_norm(torch.from_numpy(pos), ppd, full_stack.shape[1:3])
eye_movie = shift_movie_with_eye(
    full_stack,
    eye_norm,
    out_size=(100, 100),          # (outH,outW)
    center=(0.0, 0.0),            # (cx,cy) in [-1,1]
    mode="bilinear")

plt.imshow(eye_movie[7].numpy(), cmap='gray')

# %%
# Save movies
eye_pos_pix = eye_deg_to_pix(torch.from_numpy(pos), ppd)

save_eye_movies(
    full_stack=full_stack,
    eye_movie=eye_movie,
    eye_pos_pix=eye_pos_pix,
    save_prefix="fixrsvp_eye_sim",
    fps=30,
    trail_length=10,
    dot_size=8,
    trail_alpha=0.7
)

# %%
# Simulate with mixed nonlinearities: 70% complex, 30% simple
robs_mixed, eyepos_mixed, params_mixed = simulate_responses(
    n_trials=50,
    trial_duration=1.0,
    dt=1/240,
    ppd=60,

    # Eye movement parameters
    fix_mean=0.2,
    fix_spread=0.45,
    D=0.001,
    ms_amp_mean=0.3,
    ms_amp_spread=0.55,

    # Neuron parameters
    n_neurons=40,
    scales=[0, 1, 2],
    orientations=[0, 1, 2, 3],
    spatial_scatter_pix=2.0,
    nonlinearity=0.7,  # 70% complex, 30% simple
    mean_firing_rate=15.0,

    # Temporal filtering
    include_temporal_filter=True,

    # Seeds
    neuron_seed=42,
    eye_seed=123,

    verbose=True,
)

print(f"Nonlinearity types: {params_mixed['neuron_params']['nonlinearity_types']}")
print(f"Simple cell phases: {params_mixed['neuron_params']['simple_phases']}")
# Count each type
from collections import Counter
print(f"Type counts: {Counter(params_mixed['neuron_params']['nonlinearity_types'])}")
# Count simple cell phases
simple_phases_only = [p for p in params_mixed['neuron_params']['simple_phases'] if p is not None]
print(f"Simple phase counts: {Counter(simple_phases_only)}")

# %%

plt.imshow(robs_mixed[:,:,0], cmap='gray_r')
# %%
