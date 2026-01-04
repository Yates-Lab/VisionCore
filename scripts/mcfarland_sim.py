
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

from scipy.optimize import curve_fit
from tqdm import tqdm
import time
from scipy.signal import savgol_filter

# embed TrueType fonts in PDF/PS
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

# (optional) pick a clean sans‐serif
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

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

    # Initial fixation phase
    initial_fix_phase=None,   # If provided, sets the duration of the first fixation (seconds)

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
    first_fixation = True
    while i < n_steps - 1:
        # --- Fixation segment ---
        if first_fixation and initial_fix_phase is not None:
            # Use provided initial fixation duration
            Tf = initial_fix_phase
            first_fixation = False
        else:
            Tf = sample_fix_durations(
                n=1, dist=fix_dist, mean=fix_mean / max(ms_rate_boost, 1e-9), sigma=fix_spread, rng=rng
            )[0]
            first_fixation = False
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
    scale_factor=1.0,
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
    x_extent = (outW / W) * scale_factor  # extent in normalized coords [-1,1]
    y_extent = (outH / H) * scale_factor

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


def build_temporal_kernel(kernel_size=16, 
        dt=1/240, 
        tau_fast=0.004, 
        tau_slow=0.011, 
        a=0.9, n=2):
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

# PyramidSimulator Class
class PyramidSimulator:
    """
    Steerable Pyramid for simulating neural responses.

    This class encapsulates a steerable pyramid and provides methods to:
    - Compute RF properties (size, spatial frequency) for all units
    - Simulate responses from movies with optional temporal filtering
    - Visualize filters and RFs

    The simulator computes responses for ALL levels, orientations, and spatial positions
    in the pyramid. Units are indexed by (scale, orientation, y, x) tuples.

    Parameters:
    -----------
    image_shape : tuple of int
        (H, W) shape of input images
    num_ori : int
        Number of orientations
    num_scales : int
        Number of spatial scales
    temporal_kernel : torch.Tensor, optional
        Temporal filter kernel. If None, no temporal filtering is applied.
        Use build_temporal_kernel() to create a biphasic temporal kernel.

    Attributes:
    -----------
    pyr : SteerablePyramidFreq
        The underlying steerable pyramid
    rf_size : dict
        RF size (sqrt of area in pixels) for each (scale, ori) key
    rf_contour : dict
        RF contour points for each (scale, ori) key
    rf_center : dict
        RF center position for each (scale, ori) key
    filter_im : dict
        Reconstructed filter image for each (scale, ori) key
    freq_rad : dict
        Preferred spatial frequency (cycles/pixel) for each (scale, ori) key

    Example:
    --------
    >>> # Create simulator with temporal filtering
    >>> temporal_kernel = build_temporal_kernel(kernel_size=16, dt=1/240)
    >>> simulator = PyramidSimulator(
    ...     image_shape=(51, 51),
    ...     num_ori=8,
    ...     num_scales=3,
    ...     temporal_kernel=temporal_kernel
    ... )
    >>>
    >>> # Visualize filters and RFs
    >>> simulator.plot_filters(scales=[0, 1, 2])
    >>> simulator.plot_rfs(scales=[0, 1, 2])
    >>>
    >>> # Query properties for a specific unit
    >>> props = simulator.get_unit_properties(scale=1, ori=0)
    >>> print(f"RF size: {props['rf_size']:.2f} pixels")
    >>>
    >>> # Simulate responses from a movie
    >>> movie = torch.randn(100, 51, 51)  # 100 frames
    >>> units = [(0, 0, 25, 25), (1, 0, 25, 25)]  # Two units
    >>> responses = simulator.simulate(movie, units=units)
    >>> print(responses.shape)  # (100, 2)
    """

    def __init__(self, image_shape=(51, 51), num_ori=8, num_scales=3, temporal_kernel=None):
        from plenoptic.simulate import SteerablePyramidFreq

        self.image_shape = image_shape
        self.num_ori = num_ori
        self.num_scales = num_scales
        self.temporal_kernel = temporal_kernel

        # Build steerable pyramid
        order = num_ori - 1
        self.pyr = SteerablePyramidFreq(
            image_shape, order=order, height=num_scales, is_complex=True,
            downsample=False, tight_frame=False
        )

        # Compute RF properties for all units
        self._compute_rf_properties()

    def _compute_rf_properties(self):
        """Compute RF size and preferred spatial frequency for all scales and orientations."""
        from DataYatesV1.utils.rf import get_contour

        mid_y, mid_x = self.image_shape[0] // 2, self.image_shape[1] // 2

        # Get RF size using impulse response
        point = torch.zeros((1, 1, self.image_shape[0], self.image_shape[1]), dtype=torch.float32)
        point[0, 0, mid_y, mid_x] = 1
        pyr_coeffs = self.pyr.forward(point)

        def minmax(x):
            return (x - x.min()) / (x.max() - x.min())

        # Store RF properties for each (scale, orientation, y, x)
        self.rf_size = {}
        self.rf_contour = {}
        self.rf_center = {}

        for ilevel in range(self.num_scales):
            for iori in range(self.num_ori):
                I_for_contour = np.abs(pyr_coeffs[(ilevel, iori)].squeeze())
                I_for_contour = minmax(I_for_contour)
                contour, area_, center = get_contour(I_for_contour.numpy(), 0.5)

                self.rf_size[(ilevel, iori)] = np.sqrt(area_)
                self.rf_contour[(ilevel, iori)] = contour
                self.rf_center[(ilevel, iori)] = center

        # Get RF spatial frequency using filter reconstruction
        empty_image = torch.zeros((1, 1, self.image_shape[0], self.image_shape[1]), dtype=torch.float32)
        pyr_coeffs = self.pyr(empty_image)

        self.filter_im = {}
        self.freq_rad = {}

        for ilevel in range(self.num_scales):
            for iori in range(self.num_ori):
                # Set coefficient to 1 at center
                pyr_coeffs[(ilevel, iori)][:, :, mid_y, mid_x] = 1
                # Reconstruct filter
                filter_im = self.pyr.recon_pyr(pyr_coeffs, [ilevel], [iori]).squeeze()
                self.filter_im[(ilevel, iori)] = filter_im

                # Get spatial frequency tuning
                F = np.abs(np.fft.rfft2(filter_im.numpy()))
                fy = np.fft.fftfreq(self.image_shape[0], d=1)
                fx = np.fft.rfftfreq(self.image_shape[1], d=1)

                ky, kx = np.unravel_index(np.argmax(F), F.shape)
                self.freq_rad[(ilevel, iori)] = np.hypot(fy[ky], fx[kx])

                # Reset coefficient
                pyr_coeffs[(ilevel, iori)][:, :, mid_y, mid_x] = 0

    def simulate(self, movie):
        """
        Simulate responses from a movie.

        Parameters:
        -----------
        movie : torch.Tensor
            Input movie of shape (T, H, W) or (T, 1, H, W)
        
        Returns:
        --------
        responses : torch.Tensor
            Simulated responses of shape (T, n_units)
        """
        # Ensure movie has correct shape
        if movie.dim() == 3:
            movie = movie.unsqueeze(1)  # (T, 1, H, W)

        T, C, H, W = movie.shape
        
        # Simulate with or without temporal filtering
        responses = np.zeros((T, self.num_scales, self.num_ori, H, W))

        if self.temporal_kernel is not None:
            L = len(self.temporal_kernel)
            N = T-L+1
            iix = np.arange(N)[:, None] + np.arange(L)
            new_movie = torch.zeros((T, C, H, W), dtype=movie.dtype, device=movie.device)
            # print(movie[iix].shape)
            tmp = (movie[iix] * self.temporal_kernel[None, :, None, None, None]).sum(1)
            # print(tmp.shape)
            new_movie[L-1:] = tmp
            movie = new_movie

        # No temporal filtering
        pyr_coeffs = self.pyr(movie)
        for ilevel in range(self.num_scales):
            for iori in range(self.num_ori):
                responses[:, ilevel, iori] = pyr_coeffs[(ilevel, iori)].squeeze()

        # # old (apply temporal kernel AFTER)
        # if self.temporal_kernel is not None:
        #     L = len(self.temporal_kernel)
        #     N = T-L+1
        #     iix = np.arange(N)[:, None] + np.arange(L)
        #     coefs = self.pyr(movie[iix].reshape(N*L, 1, H, W)) #.reshape(N, L, H, W)
            
        #     for ilevel in range(self.num_scales):
        #         for iori in range(self.num_ori):
        #             co = coefs[(ilevel, iori)]
        #             responses[L-1:, ilevel, iori] = (co.reshape(N, L, H, W) * self.temporal_kernel[None, :, None, None]).sum(1)
        # else:
        #     # No temporal filtering
        #     pyr_coeffs = self.pyr(movie)
        #     for ilevel in range(self.num_scales):
        #         for iori in range(self.num_ori):
        #             responses[:, ilevel, iori] = pyr_coeffs[(ilevel, iori)].squeeze()
        
        return responses

    def plot_filters(self, scales=None, orientations=None, figsize=None, save_path=None):
        """
        Plot the spatial filters for requested scales and orientations.

        Parameters:
        -----------
        scales : list of int, optional
            Which scales to plot. If None, plots all scales.
        orientations : list of int, optional
            Which orientations to plot. If None, plots all orientations.
        figsize : tuple, optional
            Figure size (width, height)
        save_path : str, optional
            Path to save figure

        Returns:
        --------
        fig, axes : matplotlib figure and axes
        """
        if scales is None:
            scales = list(range(self.num_scales))
        if orientations is None:
            orientations = list(range(self.num_ori))

        n_scales = len(scales)
        n_ori = len(orientations)

        if figsize is None:
            figsize = (3 * n_ori, 3 * n_scales)

        fig, axes = plt.subplots(n_scales, n_ori, figsize=figsize)
        if n_scales == 1 and n_ori == 1:
            axes = np.array([[axes]])
        elif n_scales == 1:
            axes = axes[np.newaxis, :]
        elif n_ori == 1:
            axes = axes[:, np.newaxis]

        for i, scale in enumerate(scales):
            for j, ori in enumerate(orientations):
                ax = axes[i, j]
                filter_im = self.filter_im[(scale, ori)]
                ax.imshow(filter_im.numpy(), cmap='gray_r', interpolation='none')
                ax.set_title(f'Scale {scale}, Ori {ori}')
                ax.axis('off')

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig, axes

    def plot_rfs(self, scales=None, orientations=None, figsize=None, save_path=None):
        """
        Plot the receptive fields with contours for requested scales and orientations.

        Parameters:
        -----------
        scales : list of int, optional
            Which scales to plot. If None, plots all scales.
        orientations : list of int, optional
            Which orientations to plot. If None, plots all orientations.
        figsize : tuple, optional
            Figure size (width, height)
        save_path : str, optional
            Path to save figure

        Returns:
        --------
        fig, axes : matplotlib figure and axes
        """
        from DataYatesV1.utils.rf import get_contour

        if scales is None:
            scales = list(range(self.num_scales))
        if orientations is None:
            orientations = list(range(self.num_ori))

        n_scales = len(scales)
        n_ori = len(orientations)

        if figsize is None:
            figsize = (3 * n_ori, 3 * n_scales)

        fig, axes = plt.subplots(n_scales, n_ori, figsize=figsize)
        if n_scales == 1 and n_ori == 1:
            axes = np.array([[axes]])
        elif n_scales == 1:
            axes = axes[np.newaxis, :]
        elif n_ori == 1:
            axes = axes[:, np.newaxis]

        # Get impulse response
        mid_y, mid_x = self.image_shape[0] // 2, self.image_shape[1] // 2
        point = torch.zeros((1, 1, self.image_shape[0], self.image_shape[1]), dtype=torch.float32)
        point[0, 0, mid_y, mid_x] = 1
        pyr_coeffs = self.pyr.forward(point)

        def minmax(x):
            return (x - x.min()) / (x.max() - x.min())

        for i, scale in enumerate(scales):
            for j, ori in enumerate(orientations):
                ax = axes[i, j]
                I_for_contour = np.abs(pyr_coeffs[(scale, ori)].squeeze())
                I_for_contour = minmax(I_for_contour)

                ax.imshow(I_for_contour.numpy(), cmap='gray_r', interpolation='none')

                # Plot contour
                contour = self.rf_contour[(scale, ori)]
                ax.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)

                rf_size = self.rf_size[(scale, ori)]
                freq = self.freq_rad[(scale, ori)]
                ax.set_title(f'S{scale},O{ori}\nRF={rf_size:.1f}px, f={freq:.3f}c/px')
                ax.axis('off')

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig, axes

    def get_unit_properties(self, scale, ori):
        """
        Get RF properties for a specific unit.

        Parameters:
        -----------
        scale : int
            Scale index
        ori : int
            Orientation index

        Returns:
        --------
        dict with keys:
            'rf_size': RF size (sqrt of area in pixels)
            'freq_rad': Preferred spatial frequency (cycles per pixel)
            'rf_center': RF center position (y, x)
            'rf_contour': RF contour points
        """
        return {
            'rf_size': self.rf_size[(scale, ori)],
            'freq_rad': self.freq_rad[(scale, ori)],
            'rf_center': self.rf_center[(scale, ori)],
            'rf_contour': self.rf_contour[(scale, ori)],
        }



# Main simulation
def simulate_responses(
    pyr,
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

    # Get stimulus
    full_stack = get_fixrsvp_stack(full_size=stim_size, frames_per_im=frames_per_im)
    full_stack = torch.from_numpy(full_stack).float()

    # Determine number of time bins per trial
    n_time_bins = int(trial_duration / dt)

    # Initialize output arrays
    robs = np.full((n_trials, n_time_bins, pyr.num_scales, pyr.num_ori, pyr.image_shape[0], pyr.image_shape[1]), np.nan, dtype=float)
    eyepos = np.full((n_trials, n_time_bins, 2), np.nan, dtype=float)

    # Simulate each trial
    for itrial in range(n_trials):
        if verbose and itrial % 10 == 0:
            print(f"Trial {itrial}/{n_trials}")

        # Simulate eye trace with random initial position and phase
        x0 = eye_rng.normal(0, 0.1, size=2)  # Random initial position (mean=0, std=0.1 deg)
        # Random initial fixation phase to desynchronize microsaccade timing across trials
        initial_fix_phase = eye_rng.uniform(0, fix_mean * 2)  # Random phase in [0, 2*mean]
        _, pos, _, _ = simulate_eye_trace(
            T_total=trial_duration,
            dt=dt,
            x0=tuple(x0),
            fix_dist=fix_dist,
            fix_mean=fix_mean,
            fix_spread=fix_spread,
            D=D,
            ms_dur_mean=ms_dur_mean,
            ms_dur_jitter=ms_dur_jitter,
            ms_amp_dist="lognormal",
            ms_amp_mean=ms_amp_mean,
            ms_amp_spread=ms_amp_spread,
            initial_fix_phase=initial_fix_phase,
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
            out_size=pyr.image_shape,
            center=(0.0, 0.0),
            mode="bilinear"
        )  # (T_eye, H, W)

        # Add channel dimension for pyramid
        eye_movie = eye_movie.unsqueeze(1)  # (T, 1, H, W)

        # Simulate responses
        robs_for_trial = pyr.simulate(eye_movie)
        
        robs[itrial, :T_eye] = robs_for_trial
        eyepos[itrial, :T_eye] = pos
           
    return robs, eyepos

# Utility function for smoothing eye position
def _savgol_1d_nan(y, window_length=15, polyorder=3):
    """
    Apply Savitzky–Golay to a 1D array with NaNs.
    NaNs are interpolated for filtering and then restored.
    """
    y = np.asarray(y, float)
    mask = np.isfinite(y)

    # If too few valid points, just return original
    if mask.sum() < polyorder + 2:
        return y

    yy = y.copy()
    idx_valid = np.where(mask)[0]
    idx_nan   = np.where(~mask)[0]

    # Linear interp over NaNs so savgol_filter has no gaps
    yy[idx_nan] = np.interp(idx_nan, idx_valid, yy[idx_valid])

    # Apply SG filter
    ys = savgol_filter(
        yy,
        window_length=window_length,
        polyorder=polyorder,
        mode="interp"
    )

    # Restore original NaNs
    ys[~mask] = np.nan
    return ys


def savgol_nan_numpy(x, axis=1, window_length=15, polyorder=3):
    """
    NaN-tolerant Savitzky–Golay smoothing along a given axis for a NumPy array.
    """
    return np.apply_along_axis(
        _savgol_1d_nan,
        axis=axis,
        arr=x,
        window_length=window_length,
        polyorder=polyorder,
    )


def savgol_nan_torch(x, dim=1, window_length=15, polyorder=3):
    """
    NaN-tolerant Savitzky–Golay smoothing along dim for a torch.Tensor.
    - x: (..., T, ...) tensor
    - dim: time dimension (default 1)
    """
    # Move target dim to last for easier NumPy apply
    x_np = x.detach().cpu().numpy()
    x_np = np.moveaxis(x_np, dim, -1)

    y_np = savgol_nan_numpy(
        x_np,
        axis=-1,
        window_length=window_length,
        polyorder=polyorder,
    )

    # Move axis back and convert to torch
    y_np = np.moveaxis(y_np, -1, dim)
    y = torch.from_numpy(y_np).to(x.device).type_as(x)
    return y

def cov_to_corr(C):
    C = torch.tensor(C)
    # 1. Get the variances (diagonal elements)
    variances = torch.diag(C)
    
    # 2. Get standard deviations
    # Clamp to avoid division by zero if a neuron is silent
    std_devs = torch.sqrt(variances).clamp(min=1e-8)
    
    # 3. Outer product to create the denominator matrix
    # shape: (n, n) where entry (i, j) is sigma_i * sigma_j
    outer_std = torch.outer(std_devs, std_devs)
    
    # 4. Normalize
    R = C / outer_std
    
    # set diag to 0
    R = R - torch.diag(torch.diag(R))
    
    return R

def pava_nonincreasing_with_blocks(y, w, eps=1e-12):
    # weighted isotonic regression using PAVA (Pool-Adjacent-Violators Algorithm)
    # enforces the fitted sequence is non-increasing
    # in other words: covariance should not increase with eye distance.
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    means = []
    weights = []
    starts = []
    ends = []
    for i in range(len(y)):
        means.append(y[i])
        weights.append(w[i])
        starts.append(i)
        ends.append(i)
        while len(means) >= 2 and means[-2] < means[-1]:
            w_new = weights[-2] + weights[-1]
            m_new = (weights[-2] * means[-2] + weights[-1] * means[-1]) / (w_new + eps)
            means[-2] = m_new
            weights[-2] = w_new
            ends[-2] = ends[-1]
            means.pop(); weights.pop(); starts.pop(); ends.pop()
    yhat = np.empty_like(y)
    blocks = []
    for m, s, e, ww in zip(means, starts, ends, weights):
        yhat[s:e+1] = m
        blocks.append((s, e, float(m), float(ww)))
    return yhat, blocks

from scipy import stats
import numpy as np

def compute_robust_fano_statistics(variances, means, plot=False):
    """
    Computes robust population Fano Factor to avoid low-firing rate artifacts.
    
    Args:
        variances: (N,) array of spike count variances
        means:     (N,) array of mean spike counts
        
    Returns:
        dict containing robust metrics
    """
    # 1. Hard Filter: Discard wildly unstable estimates
    # We require at least 0.1 spikes per window on average to even consider the neuron.
    # Below this, the Fano Factor is mathematically meaningless (0/0 or noise/small).
    valid_mask = (means > 0.1) & np.isfinite(variances) & np.isfinite(means)
    
    v_clean = variances[valid_mask]
    m_clean = means[valid_mask]
    
    if len(v_clean) < 5:
        return {"FF_slope": np.nan, "FF_pop": np.nan, "FF_median": np.nan}

    # --- Metric 1: The Slope Estimator (The Gold Standard) ---
    # Fits Var = F * Mean + beta
    # This naturally weights high-firing neurons more (they have higher leverage)
    # and ignores the "explosion" at the low end.
    res = stats.linregress(m_clean, v_clean)
    slope = res.slope
    
    # --- Metric 2: The Population Ratio (Variance Weighted) ---
    # Equivalent to pooling all spikes from all neurons into one giant "super-neuron"
    # F_pop = Sum(Vars) / Sum(Means)
    ff_pop = np.sum(v_clean) / np.sum(m_clean)

    # --- Metric 3: The Median (Outlier rejection) ---
    # If you MUST use individual ratios, take the Median, never the Mean.
    ff_individual = v_clean / m_clean
    ff_median = np.median(ff_individual)

    return {
        "FF_slope": slope,
        "FF_pop": ff_pop,
        "FF_median": ff_median,
        "n_neurons": len(v_clean)
    }


# Law of total covariance decomposition    
class DualWindowAnalysis:
    """
    Covariance decomposition conditioned on eye trajectory similarity.
    
    - We estimate second moments E[S_i S_j | distance bin] (time matched), then fit intercept at d -> 0+
    - Covariance: Cov = E[SS^T] - E[S]E[S]^T
    - The Law of Total Covariance States:
        Cov[S] = E[Cov[S | d]] + Cov[E[S | d]]


    """

    def __init__(self, robs, eyepos, valid_mask,
                dt=1/240,
                min_seg_len=36,
                device="cuda"):
        '''
        robs: (tr, t, cells) spike counts
        eyepos: (tr, t, 2) eye positions
        valid_mask: (tr, t) boolean mask of valid times
        '''
        self.dt = float(dt)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"Initializing on {self.device}...")
        t0 = time.time()

        # sanitize
        if np.isnan(robs).any():
            robs = np.nan_to_num(robs, nan=0.0)
        eyepos = np.nan_to_num(eyepos, nan=0.0)

        self.robs = torch.tensor(robs, dtype=torch.float32, device=self.device)
        self.eyepos = torch.tensor(eyepos, dtype=torch.float32, device=self.device)
        self.valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)

        self.n_trials, self.n_time, self.n_cells = robs.shape

        # PSTH per time (mean across valid trials)
        valid_float = self.valid_mask.float().unsqueeze(-1)  # (tr, t, 1)
        sum_spikes = torch.sum(self.robs * valid_float, dim=0)  # (t, cells)
        cnt = torch.sum(valid_float, dim=0)  # (t, 1)

        # keep NaNs out of PSTH
        psth = torch.full((self.n_time, self.n_cells), float("nan"), device=self.device)
        ok = (cnt[:, 0] > 0)
        psth[ok] = sum_spikes[ok] / cnt[ok]
        self.psth = psth

        # break the data into valid contiguous segments
        self.segments = self._get_valid_segments(min_len_bins=min_seg_len)

        self.window_summaries = {}
        print(f"Loaded {len(self.segments)} valid segments. Init took {time.time()-t0:.2f}s")

    def _get_valid_segments(self, min_len_bins):
        segments = []
        mask_cpu = self.valid_mask.detach().cpu().numpy()
        for tr in range(self.n_trials):
            padded = np.concatenate(([False], mask_cpu[tr], [False]))
            diffs = np.diff(padded.astype(int))
            starts = np.where(diffs == 1)[0]
            stops = np.where(diffs == -1)[0]
            for s, e in zip(starts, stops):
                if (e - s) >= min_len_bins:
                    segments.append((tr, s, e))
        return segments

    # -------------------------
    # window extraction
    # -------------------------
    def _extract_windows(self, t_count, t_hist):
        """
        Inputs:
          - t_count: number of bins in count window
          - t_hist:  number of bins in history window (used for trajectory similarity)
        
        Returns:
          - SpikeCounts:  (N, cells) summed counts over count window
          - EyeTraj:      (N, t_hist, 2) eye positions over history
          - T_idx:        (N,) time index of start of count window (aligned label)
        """
        total_len = t_hist + t_count
        trial_indices, time_indices = [], []

        for (tr, start, stop) in self.segments:
            if (stop - start) < total_len:
                print(f"  Skipping trial {tr} - not enough time ({stop-start} < {total_len})")
                continue
            
            t_starts = np.arange(start, stop - total_len + 1, t_count)
            trial_indices.extend([tr] * len(t_starts))
            time_indices.extend(t_starts)

        if len(trial_indices) == 0:
            return None, None, None

        n_total = len(trial_indices)
        print(f"  Found {n_total} total windows before subsampling")
    
        idx_tr = torch.tensor(trial_indices, device=self.device, dtype=torch.long)
        idx_t0 = torch.tensor(time_indices, device=self.device, dtype=torch.long)

        # gather eye history+count then slice history
        offsets = torch.arange(total_len, device=self.device).unsqueeze(0)  # (1, total_len)
        gather_t = idx_t0.unsqueeze(1) + offsets                            # (N, total_len)
        gather_tr = idx_tr.unsqueeze(1).expand(-1, total_len)               # (N, total_len)

        EyeTraj = self.eyepos[gather_tr, gather_t, :]                             # (N, total_len, 2)

        # gather spikes only over count window
        spike_offsets = torch.arange(t_hist, total_len, device=self.device).unsqueeze(0)  # (1, t_count)
        gather_t_spk = idx_t0.unsqueeze(1) + spike_offsets                                 # (N, t_count)
        gather_tr_spk = idx_tr.unsqueeze(1).expand(-1, t_count)                            # (N, t_count)

        S_raw = self.robs[gather_tr_spk, gather_t_spk, :]                  # (N, t_count, cells)
        SpikeCounts = torch.sum(S_raw, dim=1)                                        # (N, cells)

        # aligned time label (start of count window)
        T_idx = idx_t0 + t_hist                                            # (N,)

        return SpikeCounts, EyeTraj, T_idx, idx_tr

    # 
    def _calculate_second_moment(self, SpikeCounts, EyeTraj, T_idx, n_bins=25):
        """
        Calculate second moment E[SS^T | d] for all pairs of samples
        use split half cross-validation to estimate E[SS^T]
    
        """
        
        # OLD: bins are mean euclidean distance. we had to move away from this because there's no way to do it on GPU without blowing up memory
        # diff = torch.sqrt( torch.sum((EyeTraj[:, None, :, :] - EyeTraj[None, :, :, :])**2,-1)).mean(2)       # (N, N, T, 2)
        # i, j = np.triu_indices_from(diff)
        # dist = diff[i,j]

        # bins are RMS distance. It's not an unreasonable metric for similarity, but we 
        # favor it over euclidean because there is a fast pytorch implementation on gpu (cdist)
        
        # Flatten time and coordinate dimensions: (N, T, 2) -> (N, 2T)
        N_samples, T, _ = EyeTraj.shape
        EyeFlat = EyeTraj.reshape(N_samples, -1) 

        # Compute RMS distance matrix
        dist_matrix = torch.cdist(EyeFlat, EyeFlat) / np.sqrt(T)

        # Extract upper triangle for percentiles
        i, j = torch.triu_indices(N_samples, N_samples, offset=1)
        dist = dist_matrix[i, j]

        bin_edges = np.percentile(dist.cpu().numpy(), np.arange(0, 100, 100/(n_bins+1)))

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        n_bins = len(bin_edges) - 1

        unique_times = np.unique(T_idx.detach().cpu().numpy())
        C = SpikeCounts.shape[1]
        T = EyeTraj.shape[1]
        device = EyeTraj.device

        bin_edges_t = torch.as_tensor(bin_edges, device=device, dtype=EyeTraj.dtype)
        inv_sqrt_T = (1.0 / torch.sqrt(torch.tensor(float(T), device=device, dtype=EyeTraj.dtype)))

        # keep accumulators on CPU as torch, convert to numpy at end
        SS_e_t = torch.zeros((n_bins, C, C), device='cpu', dtype=torch.float64)
        count_e_t = torch.zeros((n_bins,), device='cpu', dtype=torch.long)

        def accumulate_split(valid_idx, SS_e_t, count_e_t):
            # valid_idx: 1D numpy array of trial indices for this split
            N = len(valid_idx)
            if N < 2:
                return

            X = EyeTraj[valid_idx]                 # (N, T, 2)
            S = SpikeCounts[valid_idx]             # (N, C)

            # Pair list for cross-trial only
            ii, jj = torch.triu_indices(N, N, offset=1, device=device)  # (P,), (P,)

            # Eye distances on those pairs, without diff materialization
            Xflat = X.reshape(N, -1)                                   # (N, 2T)
            D = torch.cdist(Xflat, Xflat) * inv_sqrt_T                 # (N, N)
            d = D[ii, jj]                                              # (P,)

            # Bin IDs in 1..n_bins are interior bins (same convention as your np.digitize + (k+1))
            # bucketize returns in [0..len(edges)] where edges includes endpoints.
            bid = torch.bucketize(d, bin_edges_t, right=False)         # (P,)

            # Keep only pairs that fall into bins 1..n_bins
            ok = (bid >= 1) & (bid <= n_bins)
            if not ok.any():
                return
            ii = ii[ok]; jj = jj[ok]; bid = bid[ok]                    # (P',)

            # We’ll accumulate per bin with S_i^T @ S_j
            # (still no (P,C,C) tensor materialized)
            for k in range(1, n_bins + 1):
                mk = (bid == k)
                if not mk.any():
                    continue
                Si = S[ii[mk]]                                         # (P_k, C)
                Sj = S[jj[mk]]                                         # (P_k, C)

                # sum_p Si[p]^T Sj[p]  -> (C, C)
                # do on GPU, then move the (C,C) result to CPU accumulator
                M = Si.transpose(0, 1).matmul(Sj)                      # (C, C)

                SS_e_t[k-1] += M.detach().cpu().to(torch.float64)
                count_e_t[k-1] += mk.sum().detach().cpu()

        
        for t in unique_times:
            valid = np.where((T_idx == t).detach().cpu().numpy())[0]
            if len(valid) < 10:
                continue

            accumulate_split(valid, SS_e_t, count_e_t)

        # Convert to numpy and form split-half estimate
        SS_e = SS_e_t.numpy()
        count_e = count_e_t.numpy()

        MM = SS_e / count_e[:, None, None]
        # symmetrize
        MM = 0.5 * (MM + np.swapaxes(MM, -1, -2))

        return MM, bin_centers, count_e
    
    # -------------------------
    # unbiased PSTH covariance (split-half cross-covariance)
    # -------------------------
    def _split_half_psth_covariance(self, S, T_idx, min_trials_per_time=10, seed=0):
        '''
        Split-half cross-covariance to estimate PSTH covariance.
        Because we have finite sample size, we want a robust estimate of the PSTH covariance.
        The logic is as follows:
        Assume responses are û = u + ε. (u = true PSTH, ε = noise)
        Split data into independent halves A,B: û_A = u + ε_A, û_B = u + ε_B.
        Then cov(û_A, û_B) = cov(u,u) + cov(u,ε_B) + cov(ε_A,u) + cov(ε_A,ε_B).
        With independent, zero-mean noise: cov(û_A, û_B) = cov(u,u).
        '''

        # set random seed
        rng = np.random.default_rng(seed)

        unique_times = np.unique(T_idx.detach().cpu().numpy())
        NT = len(unique_times)
        N_cells = S.shape[1]
        N_samples = S.shape[0]

        # Pre-allocate masks (false by default)
        mask_A = np.zeros(N_samples, dtype=bool)
        mask_B = np.zeros(N_samples, dtype=bool)

        # iterate time points to ensure exactly 50/50 split per time bin
        # This minimizes the variance of the split means
        for t in unique_times:
            # Find indices for this specific time point
            # (Note: converting to numpy once outside loop would be faster, but this is clear)
            ix_t = np.where((T_idx == t).detach().cpu().numpy())[0]
            n_t = len(ix_t)

            if n_t < min_trials_per_time:
                continue

            # Shuffle indices for this time point
            perm = rng.permutation(n_t)
            
            # Split indices
            split_idx = n_t // 2
            idx_A_local = ix_t[perm[:split_idx]]
            idx_B_local = ix_t[perm[split_idx:]]

            mask_A[idx_A_local] = True
            mask_B[idx_B_local] = True

        # --- COMPUTE PSTH HALVES ---
        # Initialize with NaNs
        PSTH_A = np.full((NT, N_cells), np.nan)
        PSTH_B = np.full((NT, N_cells), np.nan)

        for it, t in enumerate(unique_times):
            # Intersect time mask with split masks
            # Since we built masks_A/B strictly on time indices, we can just check validity
            ix_t = (T_idx == t).detach().cpu().numpy()
            
            # We must re-verify the intersection to map to the correct row 'it'
            # (mask_A is global, ix_t is local time selector)
            m_A = mask_A & ix_t
            m_B = mask_B & ix_t
            
            # Check if we have data (redundant with loop above but safe)
            if not m_A.any() or not m_B.any():
                continue

            PSTH_A[it] = S[m_A].mean(0).detach().cpu().numpy()
            PSTH_B[it] = S[m_B].mean(0).detach().cpu().numpy()

        # --- UNBIASED COVARIANCE ---
        # Keep only times where both splits were valid
        finite_times = np.isfinite(PSTH_A).all(axis=1) & np.isfinite(PSTH_B).all(axis=1)
        
        if finite_times.sum() < 2:
            # Not enough time points to compute covariance
            return np.full((N_cells, N_cells), np.nan), PSTH_A, PSTH_B

        # Center across time
        # (N_time_valid, N_cells)
        XA = PSTH_A[finite_times] - PSTH_A[finite_times].mean(0, keepdims=True)
        XB = PSTH_B[finite_times] - PSTH_B[finite_times].mean(0, keepdims=True)

        # Unbiased estimator: Divide by (N_time - 1)
        n_time_bins = XA.shape[0]
        Ccv = (XA.T @ XB) / (n_time_bins - 1)

        # Symmetrize
        Ccv = 0.5 * (Ccv + Ccv.T)

        return Ccv, PSTH_A, PSTH_B

    def fit_best_monotonic(self, y, w):
        """
        Fits both non-increasing and non-decreasing PAVA.
        Returns the intercept (yhat[0]) of the fit with the lowest error.
        """
        # 1. Fit Non-Increasing (Classic PAVA)
        y_decr, _ = pava_nonincreasing_with_blocks(y, w)
        sse_decr = np.sum(w * (y - y_decr)**2)
        
        # 2. Fit Non-Decreasing
        # Trick: Negate y, fit non-increasing, then negate result
        y_incr_neg, _ = pava_nonincreasing_with_blocks(-y, w)
        y_incr = -y_incr_neg
        sse_incr = np.sum(w * (y - y_incr)**2)
        
        # 3. Model Selection
        if sse_decr < sse_incr:
            return y_decr[0]
        else:
            return y_incr[0]

    def _fit_intercepts_vectorized(self, Ceye, count_e):
        """
        Fits the intercept (d->0) for every element of the covariance matrix.
        Strictly enforces monotonicity (either increasing or decreasing) to handle
        both positive and negative correlations correctly.
        """
        n_bins, n_cells, _ = Ceye.shape
        C_intercept = np.zeros((n_cells, n_cells), dtype=Ceye.dtype)

        # Pre-calculate valid weights once
        # (Assuming count_e is consistent across pairs, which it is)
        # We need to handle potential NaNs in Ceye if binning failed for some reason
        
        for i in range(n_cells):
            # Diagonal: Variance must be non-increasing (Conditioning reduces variance)
            # Technically variance *could* increase if FEMs were suppressing noise, 
            # but physically FEMs add variance. So non-increasing is the correct physical prior for Diagonal.
            y_diag = Ceye[:, i, i]
            yhat, _ = pava_nonincreasing_with_blocks(y_diag, count_e)
            C_intercept[i, i] = yhat[0]

            # Off-Diagonals: Can be increasing OR decreasing
            for j in range(i + 1, n_cells):
                y = Ceye[:, i, j]
                
                # Handle NaNs if strictly necessary (though Ceye shouldn't have them if logic is tight)
                valid = np.isfinite(y)
                if not valid.any():
                    C_intercept[i, j] = np.nan
                    C_intercept[j, i] = np.nan
                    continue
                
                val = self.fit_best_monotonic(y[valid], count_e[valid])
                
                C_intercept[i, j] = val
                C_intercept[j, i] = val

        return C_intercept
    
    def _calculate_Crate(self, SpikeCounts, EyeTraj, T_idx, n_bins=25):
        """
        Calculate the eye-conditioned covariance matrix (Crate) using split-half cross-validation.

        Inputs:
        -------
        SpikeCounts : torch.Tensor (N, cells)
            Spike counts for each sample
        EyeTraj : torch.Tensor (N, t_hist, 2)
            Eye positions for each sample
        T_idx : torch.Tensor (N,)
            Time index of start of count window (aligned label)
        n_bins : int
            Number of bins to use for eye distance
        
        Returns:
        --------
        Crate : np.ndarray (cells, cells)
            Eye-conditioned covariance matrix
        Erate: np.ndarray (cells,)
            Mean spike counts per cell
        Ceye: np.ndarray (n_bins, cells, cells)
            Raw eye-conditioned covariance matrix (biased estimator)
        bin_centers: np.ndarray (n_bins,)
            Bin centers for eye distance
        count_e: np.ndarray (n_bins,)
            Number of pairs in each bin
        """
        MM, bin_centers, count_e = self._calculate_second_moment(SpikeCounts, EyeTraj, T_idx, n_bins=n_bins)
        Erate = torch.nanmean(SpikeCounts, 0).detach().cpu().numpy() # raw means
        Ceye = MM - Erate[:,None] * Erate[None,:] # raw rate covariances conditioned on eye trajectory

        Crate = self._fit_intercepts_vectorized(Ceye, count_e) # fit intercepts
        
        return Crate, Erate, Ceye, bin_centers, count_e

    # -------------------------
    # run_sweep
    # -------------------------
    def run_sweep(self, window_sizes_ms, t_hist_ms=10, n_bins=15):
        t_hist_bins = int(t_hist_ms / (self.dt * 1000))
        results = []
        mats_save = []

        print(f"Starting Sweep (Hist={t_hist_ms}ms)...")

        for win_ms in tqdm(window_sizes_ms):
            t_count_bins = int(win_ms / (self.dt * 1000))
            t_count_bins = max(1, t_count_bins)

            # extract windows
            SpikeCounts, EyeTraj, T_idx, _ = self._extract_windows(t_count_bins, np.maximum(t_hist_bins, t_count_bins))
            n_samples = SpikeCounts.shape[0]
            if SpikeCounts is None or n_samples < 100: continue # arbitrary threshold (how much data do we need?)

            # calculate eye conditioned covariance
            Crate, Erate, Ceye, bin_centers, count_e = self._calculate_Crate(SpikeCounts, EyeTraj, T_idx, n_bins=n_bins)
            
            # PSTH covariance
            Cpsth, PSTH_A, PSTH_B = self._split_half_psth_covariance(SpikeCounts, T_idx, min_trials_per_time=10, seed=0)

            # total covariance
            ix = np.isfinite(SpikeCounts.sum(1).detach().cpu().numpy())
            Ctotal = torch.cov(SpikeCounts[ix].T, correction=1).detach().cpu().numpy() # total covariance
            
            # covariance due to fixational eye movements
            Cfem = Crate - Cpsth
            Cfem = 0.5 * (Cfem + Cfem.T) # symmetrize

            # noise covariance
            CnoiseU = Ctotal - Cpsth
            CnoiseC = Ctotal - Crate

            # symmetrize
            CnoiseU = 0.5 * (CnoiseU + CnoiseU.T)
            CnoiseC = 0.5 * (CnoiseC + CnoiseC.T)
            
            # fano factors
            ff_uncorr = np.diag(CnoiseU) / Erate
            ff_corr = np.diag(CnoiseC) / Erate

            # noise correlation
            NoiseCorrU = cov_to_corr(CnoiseU)
            NoiseCorrC = cov_to_corr(CnoiseC)

            alpha = np.diag(Cpsth) / np.diag(Crate)

            if np.isnan(Cfem).any():
                rank = np.nan
            else:
                evals = np.linalg.eigvalsh(Cfem)[::-1]
                pos = evals[evals > 0]
                rank = (np.sum(pos[:2]) / np.sum(pos)) if len(pos) > 2 else 1.0

            results.append({
                "window_ms": win_ms,
                "ff_uncorr": ff_uncorr,
                "ff_corr": ff_corr,
                "ff_uncorr_mean": np.nanmean(ff_uncorr),
                "ff_corr_mean": np.nanmean(ff_corr),
                "alpha": alpha,
                "fem_rank_ratio": rank,
                "n_samples": n_samples,
                'Erates': Erate
            })

            mats_save.append({
                "Total": Ctotal,
                "PSTH": Cpsth,
                "FEM": Cfem,
                "Intercept": Crate,
                "NoiseCorrU": NoiseCorrU,
                "NoiseCorrC": NoiseCorrC,
                "PSTH_A": PSTH_A,
                "PSTH_B": PSTH_B,
            })

            win_key = float(win_ms)
            self.window_summaries[win_key] = {
                "bin_centers": bin_centers,
                "binned_covs": Ceye,          # SECOND MOMENTS (kept name for compatibility)
                "bin_counts": count_e,
                "Sigma_Intercept": Crate,          # COVARIANCE
                "Sigma_PSTH": Cpsth,            # COVARIANCE
                "Sigma_Total": Ctotal,
                "Sigma_FEM": Cfem,
                "mean_counts": Erate,
            }

        return results, mats_save

    # -------------------------
    # public API: inspect_neuron_pair (unchanged call signature)
    # -------------------------
    def inspect_neuron_pair(self, i, j, win_ms, ax=None, show=True):
        """
        Plots COVARIANCE vs distance by converting stored SECOND MOMENTS to covariance
        via subtracting global mean product (mu_i * mu_j), as in McFarland-style derivations.
        """
        import matplotlib.pyplot as plt

        if not self.window_summaries:
            raise RuntimeError("run_sweep must be called before inspecting neuron pairs.")

        win_key = float(win_ms)
        if win_key not in self.window_summaries:
            avail = ", ".join(str(k) for k in sorted(self.window_summaries.keys()))
            raise KeyError(f"Window {win_ms}ms not cached. Available: {avail}")

        summary = self.window_summaries[win_key]
        bin_centers = summary["bin_centers"]
        covs = summary["binned_covs"][:, i, j]     # SECOND MOMENT
        counts = summary["bin_counts"]
        valid = counts > 0
        if not np.any(valid):
            raise RuntimeError("No histogram bins with data for this neuron pair.")
        

        intercept_cov = summary["Sigma_Intercept"][i, j]
        psth_cov = summary["Sigma_PSTH"][i, j]

        created = False
        if ax is None:
            created = True
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
        else:
            fig = ax.figure

        ax.plot(bin_centers[valid], covs[valid], "o", alpha=0.6, label="Measured Covariance")


        ax.axhline(psth_cov, linestyle="--", linewidth=2, label="PSTH Covariance")
        ax.axhline(intercept_cov, linestyle=":", linewidth=2, label="Intercept")

        ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Δ Eye Trajectory (a.u.)")
        ax.set_ylabel("Covariance")
        ax.set_title(f"Neuron Pair ({i},{j}) | Window {win_ms} ms")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, loc="best")

        if show and created:
            plt.show()

        return fig, ax


#%%
if __name__ == "__main__":


#%%  Load stimuli, simulate eye trace
    dt = 1/240
    ppd = 60

    # get stimuli
    full_stack = get_fixrsvp_stack(frames_per_im=int(1/dt/30))
    full_stack = torch.from_numpy(full_stack)
    
    # example eye trace (for dialing in parameters)
    t, pos, vel, state = simulate_eye_trace(
            T_total=full_stack.shape[0]*dt, dt=dt,
            fix_dist="lognormal",
            fix_mean=0.4, fix_spread=0.45,
            D=0.001,
            ms_dur_mean=0.020, ms_dur_jitter=0.02,
            ms_amp_mean=0.15, ms_amp_spread=0.55,
            use_sigmoid_gate=True
        )

    plt.plot(pos)
    plt.ylim(-1, 1)



#%% shit movie

    eye_norm = eye_deg_to_norm(torch.from_numpy(pos), ppd, full_stack.shape[1:3])
    eye_movie = shift_movie_with_eye(
        full_stack,
        eye_norm,
        out_size=(101, 101),          # (outH,outW)
        center=(0.0, 0.0),            # (cx,cy) in [-1,1]
        scale_factor=1.0,
        mode="bilinear")

    plt.imshow(eye_movie[7].numpy(), cmap='gray')

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

# %% simulate responses

    temporal_kernel = build_temporal_kernel()

    pyr = PyramidSimulator(
        image_shape=(101, 101),
        num_ori=8,
        num_scales=4,
        temporal_kernel=None#torch.flip(temporal_kernel, dims=[0])
    )

    coefs_pyr, eyepos_sim = simulate_responses(pyr,
                n_trials=100,
                fix_mean=0.4, fix_spread=0.45,
                D=0.001,
                ms_amp_mean=0.15,
        )
#%% convert coefficients to spikes

    f = lambda x: np.maximum(0, x)
    driver = np.real(coefs_pyr[:,16:,3,0,3:-3,50])
    robs = f(driver)
    robs /= np.mean(robs, (0,1), keepdims=True)
    robs *= 10 # mean firing rate = 10 spikes/s

    robs = np.random.poisson(robs*dt)
    tracker_noise = np.random.randn(*eyepos_sim[:,16:].shape)*0
    eyepos = eyepos_sim[:,16:].copy() + tracker_noise


    sx = int(np.sqrt(robs.shape[-1]))
    sy = int(np.ceil(robs.shape[-1] / sx))
    fig, axs = plt.subplots(sy, sx, figsize=(3*sx, 2*sy), sharex=True, sharey=False)

    for cc in range(robs.shape[-1]):
        ax = axs.flatten()[cc]
        ax.imshow(robs[:, :, cc], aspect='auto', interpolation='none', cmap='gray_r')
        # axis off
        ax.set_xticks([])
        ax.set_yticks([])

#%% Run analysis

    # 1. Setup
    # Assuming 'robs', 'eyepos', 'valid_mask' are already loaded from your dataset code
    # valid_mask should be True where data is good (no fix breaks)
    valid_mask = np.isfinite(np.sum(robs, axis=2)) & np.isfinite(np.sum(eyepos, axis=2))
    
    analyzer = DualWindowAnalysis(robs, eyepos, valid_mask, dt=1/240)

    windows = [5, 10, 20, 40, 80, 100, 150]
    results, last_mats = analyzer.run_sweep(windows, t_hist_ms=5)

#%% inspect pairs
    # ii = 20
    # for jj in range(5):
    #     analyzer.inspect_neuron_pair(ii, jj, 20, ax=None, show=True)

#%%

    window_idx = 2
    Ctotal = last_mats[window_idx]['Total']
    Cfem = last_mats[window_idx]['Intercept']
    Cpsth = last_mats[window_idx]['PSTH']
    CnoiseU = last_mats[window_idx]['NoiseCorrU']
    CnoiseC = last_mats[window_idx]['NoiseCorrC']
    FF_uncorr = results[window_idx]['ff_uncorr']
    FF_corr = results[window_idx]['ff_corr']

    v = np.max(Ctotal.flatten())
    plt.subplot(1,3,1)
    plt.imshow(Ctotal, vmin=-v, vmax=v)
    plt.title('Total')
    plt.subplot(1,3,2)
    plt.imshow(Cfem, vmin=-v, vmax=v)
    plt.title('Eye')
    plt.subplot(1,3,3)
    plt.imshow(Cpsth, vmin=-v, vmax=v)
    plt.title('PSTH')

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(CnoiseU, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Noise (Uncorrected))')
    plt.subplot(1,2,2)
    plt.imshow(CnoiseC, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Noise (Corrected) ')


    plt.figure()
    plt.plot(FF_uncorr, FF_corr, '.')
    plt.plot(plt.xlim(), plt.xlim(), 'k')
    plt.xlabel('Fano Factor (Uncorrected)')
    plt.ylabel('Fano Factor (Corrected)')
    plt.title('Fano Factor vs Window Size')

#%%
    def get_upper_triangle(C):
        rows, cols = np.triu_indices_from(C, k=1)
        v = C[rows, cols]
        return v

    rho_uncorr = get_upper_triangle(CnoiseU)
    rho_corr = get_upper_triangle(CnoiseC)

    plt.figure()
    plt.plot(rho_uncorr, rho_corr, '.')
    plt.plot(plt.xlim(), plt.xlim(), 'k')
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel('Correlation (Uncorrected)')
    plt.ylabel('Correlation (Corrected)')
    plt.title('Correlation vs Window Size')


#%% 3. Plot Fano Factor Scaling
    window_ms = [results[i]['window_ms'] for i in range(len(results))]
    ff_uncorr = np.zeros_like(window_ms, dtype=np.float64)
    ff_uncorr_std = np.zeros_like(window_ms, dtype=np.float64)
    ff_corr = np.zeros_like(window_ms, dtype=np.float64)
    ff_corr_std = np.zeros_like(window_ms, dtype=np.float64)
    for iwindow in range(len(window_ms)):
        ff_uncorr[iwindow] = np.nanmean(results[iwindow]['ff_uncorr'])
        ff_corr[iwindow] = np.nanmean(results[iwindow]['ff_corr'])
        ff_uncorr_std[iwindow] = np.nanstd(results[iwindow]['ff_uncorr'])
        ff_corr_std[iwindow] = np.nanstd(results[iwindow]['ff_corr'])

    plt.figure(figsize=(8, 6))
    plt.plot(window_ms, ff_uncorr, 'o-', label='Standard (Uncorrected)')
    plt.plot(window_ms, ff_corr, 'o-', label='FEM-Corrected')
    # plot error bars
    plt.fill_between(window_ms, ff_uncorr - ff_uncorr_std, ff_uncorr + ff_uncorr_std, alpha=0.2)
    plt.fill_between(window_ms, ff_corr - ff_corr_std, ff_corr + ff_corr_std, alpha=0.2)

    plt.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Count Window (ms)')
    plt.ylabel('Mean Fano Factor')
    plt.title('Integration of Noise: FEM Correction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()