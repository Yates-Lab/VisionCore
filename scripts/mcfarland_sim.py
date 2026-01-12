
#%% Generic Imports

# this is to suppress errors in the attempts at compilation that happen for one of the loaded models because it crashed
import sys
sys.path.append('..')
import numpy as np

from DataYatesV1 import enable_autoreload, get_free_device

import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy import stats

import torch
import torch.nn.functional as F

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
def get_fixrsvp_stack(full_size=600, frames_per_im=3,
        bkgnd = 127.0,
        radius = 1.5,
        ppd = 37.50476617, prefix='im'):
    '''
    Utility for getting different types of stimuli.
    '''
    center_pix = [full_size / 2, full_size / 2]
    from DataYatesV1.exp.support import get_rsvp_fix_stim, get_face_library, get_backimage_directory
    from DataYatesV1.exp.general import gen_gauss_image_texture, place_gauss_image_texture

    if prefix == 'im':
        images = get_rsvp_fix_stim()
        num_images = len(images.keys())-3
    elif prefix == 'face':
        images = get_face_library()
        num_images = len(images.keys())-3
    elif prefix == 'nat':
        from PIL import Image
        directory = get_backimage_directory() # posix path

        # list all images in the directory
        image_files = [f for f in directory.iterdir() if f.is_file()]
        num_images = len(image_files)

        images = []
        for image_file in image_files:
            image = Image.open(image_file)
            W, H = 1280, 720
            ctr = [W//2, H//2]
            ix = np.arange(-full_size//2, full_size//2)
            image = image.resize((W, H), resample=2)
            image = np.array(image)[ix + ctr[1],:][:, ix + ctr[0]]
            if image.ndim == 3:
                image = np.mean(image, axis=2) #.astype(np.uint8)
            images.append(image)

        images = np.stack(images, axis=0)
        
    
    full_stack = []
    # frame_counter = 0
    for im_id in range(num_images):
        if isinstance(images, dict):
            
            im = images[f'{prefix}{im_id+1:02d}'].mean(axis=2).astype(np.float32)
            im_tex, alpha_tex = gen_gauss_image_texture(im, bkgnd)
        
            im, _ = place_gauss_image_texture(im_tex, alpha_tex, 
                                        np.array([0.,0.]), radius, center_pix,
                                        bkgnd, ppd, roi=np.array([[0, full_size-1], [0, full_size-1]]), binSize=1)

            im = (im + .5 + bkgnd).astype(np.uint8)
        else:
            im = images[im_id]
        
        # place in center
        for _ in range(frames_per_im):
            full_stack.append(im)
            
    full_stack = np.stack(full_stack, axis=0)
    
    return full_stack

# ----------------------------
# Simulating eye movements
# ----------------------------

# fixation duration
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


# Microsaccade size follows main sequence
def main_sequence_peak_vel(amp_deg, v0=20.0, vmax=200.0, a0=1.0):
    """
    Simple saturating main sequence: v_peak = v0 + (vmax - v0)*(1 - exp(-amp/a0))
    Units: deg/s.
    Reasonable microsaccade regime: amp ~ 0.05–1 deg, v_peak ~ 20–200 deg/s.
    """
    amp_deg = np.asarray(amp_deg)
    return v0 + (vmax - v0) * (1.0 - np.exp(-amp_deg / max(a0, 1e-9)))

# Core simulator
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

# ----------------------------
# Utilities for resampling images with gaze position
# ----------------------------

# convert degrees to normalized units (-1 to 1 range for image coordinates)
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

# convert degrees to pixels (relative to image center)
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

# resample gaze-contingent movie
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

# save out animations
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

# ----------------------------
# Simulate "neural" responses using steerable pyramid
# ----------------------------

# build biphasic temporal kernel
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

# ----------------------------
# General Utilities used throughout
# ----------------------------


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

# def cov_to_corr(C):
#     C = torch.tensor(C)
#     # 1. Get the variances (diagonal elements)
#     variances = torch.diag(C)
    
#     # 2. Get standard deviations
#     # Clamp to avoid division by zero if a neuron is silent
#     std_devs = torch.sqrt(variances).clamp(min=1e-8)
    
#     # 3. Outer product to create the denominator matrix
#     # shape: (n, n) where entry (i, j) is sigma_i * sigma_j
#     outer_std = torch.outer(std_devs, std_devs)
    
#     # 4. Normalize
#     R = C / outer_std
    
#     # set diag to 0
#     R = R - torch.diag(torch.diag(R))
    
#     return R
def cov_to_corr(C, min_var=1e-3):
    """
    Converts covariance to correlation (N x N).
    Returns NaNs for neurons with unstable, vanishing, or negative variance.
    """
    if not isinstance(C, torch.Tensor):
        C = torch.tensor(C, dtype=torch.float32)
    
    # 1. Get variances (diagonal)
    variances = torch.diag(C)
    
    # 2. Identify Valid Neurons
    # We require variance to be strictly positive and above the noise floor.
    # Neurons with NaN variance (from run_sweep) or tiny variance (survivors) fail this.
    valid_mask = variances > min_var
    
    # 3. Compute Standard Deviations
    # Initialize with NaNs so that invalid neurons automatically produce NaN correlations
    std_devs = torch.full_like(variances, float('nan'))
    std_devs[valid_mask] = torch.sqrt(variances[valid_mask])
    
    # 4. Outer Product (N x N)
    # Any row/col with a NaN std_dev will result in a NaN row/col in the denominator
    outer_std = torch.outer(std_devs, std_devs)
    
    # 5. Normalize
    # Division by NaN (or zero) results in NaN, which is exactly what we want.
    R = C / outer_std
    
    # 6. Clamp to [-1, 1]
    # torch.clamp passes NaNs through unchanged, but restricts valid values to physical limits.
    R = torch.clamp(R, -1.0, 1.0)
    
    # 7. Set diagonal to 0
    # (Standard practice for noise correlations)
    R.fill_diagonal_(0.0)
    
    return R.numpy()

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

def slope_ci_t(res, n, ci=0.95):
    """Parametric CI using linregress stderr and t critical value."""
    df = n - 2
    tcrit = stats.t.ppf(0.5 + ci/2, df)
    lo = res.slope - tcrit * res.stderr
    hi = res.slope + tcrit * res.stderr
    return lo, hi

def bootstrap_mean_ci(x, n_boot=5000, ci=95, seed=0):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot_means = x[idx].mean(axis=1)

    alpha = (100 - ci) / 2
    lo, hi = np.percentile(boot_means, [alpha, 100 - alpha])
    return x.mean(), (lo, hi)

def bootstrap_slope_ci(x, y, nboot=5000, ci=0.95, rng=0):
    """
    Nonparametric bootstrap: resample (x_i, y_i) pairs.
    Returns (slope_hat, lo, hi, slopes_boot).
    """
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    rng = np.random.default_rng(rng)

    slopes = np.empty(nboot, dtype=float)
    for b in range(nboot):
        idx = rng.integers(0, n, size=n)
        slopes[b] = stats.linregress(x[idx], y[idx]).slope

    alpha = 1 - ci
    lo, hi = np.quantile(slopes, [alpha/2, 1 - alpha/2])
    slope_hat = stats.linregress(x, y).slope
    return slope_hat, lo, hi, slopes

def plot_slope_estimation(ax, means, variances, title, color, label=''):
    # Filter
    valid = (means > 0.1) & np.isfinite(variances) & np.isfinite(means)
    x = np.asarray(means[valid])
    y = np.asarray(variances[valid])

    res = stats.linregress(x, y)
    
    ax.scatter(x, y, s=15, alpha=0.6, c=color, label=f'{label} FF = {res.slope:.2f}')

    x_line = np.linspace(0, x.max(), 100)
    y_line = res.slope * x_line + res.intercept
    ax.plot(x_line, y_line, 'k--', linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Mean Rate (spk/s)")
    ax.set_ylabel("Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return res, x, y  # return x,y too so we can bootstrap outside

def get_upper_triangle(C): # used to get the correlation values
    rows, cols = np.triu_indices_from(C, k=1)
    v = C[rows, cols]
    return v

def index_cov(cov_matrix, indices):
    # index into a square matrix
    return cov_matrix[indices][:, indices]
        
# ----------------------------
# Main analysis
# ----------------------------

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

    
    # window extraction 
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

    # Calculate second moment
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
    
    # unbiased PSTH covariance (split-half cross-covariance)
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
    
    def _calculate_Crate(self, SpikeCounts, EyeTraj, T_idx, n_bins=25, Ctotal=None):
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

        if Ctotal is not None:
            print("Ctotal is not None")
            # find neurons that violate the physical limit that the signal covariance cannot exceed the total covariance
            bad_mask = np.diag(Crate) > .99*np.diag(Ctotal)
            print(f"  Found {bad_mask.sum()} neurons violating physical limit")
            Crate[bad_mask,:] = np.nan
            Crate[:,bad_mask] = np.nan
            Ceye[:,bad_mask,:] = np.nan
            Ceye[:,:,bad_mask] = np.nan
        
        return Crate, Erate, Ceye, bin_centers, count_e

    def run_sweep(self, window_sizes_ms, t_hist_ms=10, n_bins=15, n_shuffles=0, seed=42):
        t_hist_bins = int(t_hist_ms / (self.dt * 1000))
        results = []
        mats_save = []

        print(f"Starting Sweep (Hist={t_hist_ms}ms) with {n_shuffles} shuffles...")
        
        # Generator for shuffling
        rng_shuffle = torch.Generator(device=self.device)
        rng_shuffle.manual_seed(seed)

        for win_ms in tqdm(window_sizes_ms):
            t_count_bins = int(win_ms / (self.dt * 1000))
            t_count_bins = max(1, t_count_bins)

            # 1. Extract Windows
            SpikeCounts, EyeTraj, T_idx, _ = self._extract_windows(t_count_bins, np.maximum(t_hist_bins, t_count_bins))
            
            if SpikeCounts is None:
                continue
                
            n_samples = SpikeCounts.shape[0]
            if n_samples < 100: 
                continue 

            # 2. Total Covariance (Fixed)
            ix = np.isfinite(SpikeCounts.sum(1).detach().cpu().numpy())
            Ctotal = torch.cov(SpikeCounts[ix].T, correction=1).detach().cpu().numpy() 

            # 3. PSTH Covariance (Fixed)
            # This is now calculated ONCE, guaranteeing identity between Real and Shuffled conditions.
            Cpsth, PSTH_A, PSTH_B = self._split_half_psth_covariance(SpikeCounts, T_idx, min_trials_per_time=10, seed=0)

            # 4. Real Analysis (Crate)
            Crate, Erate, Ceye, bin_centers, count_e = self._calculate_Crate(
                SpikeCounts, EyeTraj, T_idx, n_bins=n_bins, Ctotal=Ctotal
            )
            
            # 5. Shuffled Analysis (Loop)
            # We only re-calculate Crate (the intercept). Cfem_shuff will be derived later as (Crate_shuff - Cpsth).
            shuffled_intercepts = []
            
            if n_shuffles > 0:
                for k in range(n_shuffles):
                    # Permute EyeTraj relative to SpikeCounts
                    # This breaks the causal link but keeps valid trajectory statistics
                    perm = torch.randperm(n_samples, generator=rng_shuffle, device=self.device)
                    EyeTraj_shuff = EyeTraj[perm]
                    
                    # Calculate Intercept for shuffled data
                    # We pass Ctotal=None to avoid NaN-ing out violations (shuffles are noisy and might violate bounds)
                    Crate_shuff, _, _, _, _ = self._calculate_Crate(
                        SpikeCounts, EyeTraj_shuff, T_idx, n_bins=n_bins, Ctotal=None
                    )
                    shuffled_intercepts.append(Crate_shuff)

            # 6. Derived Real Metrics
            Cfem = Crate - Cpsth
            Cfem = 0.5 * (Cfem + Cfem.T) 

            CnoiseU = Ctotal - Cpsth
            CnoiseC = Ctotal - Crate
            CnoiseU = 0.5 * (CnoiseU + CnoiseU.T)
            CnoiseC = 0.5 * (CnoiseC + CnoiseC.T)
            
            ff_uncorr = np.diag(CnoiseU) / Erate
            ff_corr = np.diag(CnoiseC) / Erate

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
                'Erates': Erate,
                'count_e': count_e
            })

            mats_save.append({
                "Total": Ctotal,
                "PSTH": Cpsth,
                "FEM": Cfem,
                "Intercept": Crate,
                "Shuffled_Intercepts": shuffled_intercepts, # List of (N_cells, N_cells) arrays
                "NoiseCorrU": NoiseCorrU,
                "NoiseCorrC": NoiseCorrC,
                "PSTH_A": PSTH_A,
                "PSTH_B": PSTH_B,
            })

            # Store summary for plotting individual pairs
            win_key = float(win_ms)
            self.window_summaries[win_key] = {
                "bin_centers": bin_centers,
                "binned_covs": Ceye,           
                "bin_counts": count_e,
                "Sigma_Intercept": Crate,          
                "Sigma_PSTH": Cpsth,            
                "Sigma_Total": Ctotal,
                "Sigma_FEM": Cfem,
                "mean_counts": Erate,
            }

        return results, mats_save
    
    # run_sweep
    # def run_sweep(self, window_sizes_ms, t_hist_ms=10, n_bins=15):
    #     t_hist_bins = int(t_hist_ms / (self.dt * 1000))
    #     results = []
    #     mats_save = []

    #     print(f"Starting Sweep (Hist={t_hist_ms}ms)...")

    #     for win_ms in tqdm(window_sizes_ms):
    #         t_count_bins = int(win_ms / (self.dt * 1000))
    #         t_count_bins = max(1, t_count_bins)

    #         # extract windows
    #         SpikeCounts, EyeTraj, T_idx, _ = self._extract_windows(t_count_bins, np.maximum(t_hist_bins, t_count_bins))
    #         n_samples = SpikeCounts.shape[0]
    #         if SpikeCounts is None or n_samples < 100: continue # arbitrary threshold (how much data do we need?)

    #         # total covariance
    #         ix = np.isfinite(SpikeCounts.sum(1).detach().cpu().numpy())
    #         Ctotal = torch.cov(SpikeCounts[ix].T, correction=1).detach().cpu().numpy() # total covariance

    #         # calculate eye conditioned covariance
    #         Crate, Erate, Ceye, bin_centers, count_e = self._calculate_Crate(SpikeCounts, EyeTraj, T_idx, n_bins=n_bins, Ctotal=Ctotal)
            
    #         # PSTH covariance
    #         Cpsth, PSTH_A, PSTH_B = self._split_half_psth_covariance(SpikeCounts, T_idx, min_trials_per_time=10, seed=0)

    #         # covariance due to fixational eye movements
    #         Cfem = Crate - Cpsth
    #         Cfem = 0.5 * (Cfem + Cfem.T) # symmetrize

    #         # noise covariance
    #         CnoiseU = Ctotal - Cpsth
    #         CnoiseC = Ctotal - Crate

    #         # symmetrize
    #         CnoiseU = 0.5 * (CnoiseU + CnoiseU.T)
    #         CnoiseC = 0.5 * (CnoiseC + CnoiseC.T)
            
    #         # fano factors
    #         ff_uncorr = np.diag(CnoiseU) / Erate
    #         ff_corr = np.diag(CnoiseC) / Erate

    #         # noise correlation
    #         NoiseCorrU = cov_to_corr(CnoiseU)
    #         NoiseCorrC = cov_to_corr(CnoiseC)

    #         alpha = np.diag(Cpsth) / np.diag(Crate)

    #         if np.isnan(Cfem).any():
    #             rank = np.nan
    #         else:
    #             evals = np.linalg.eigvalsh(Cfem)[::-1]
    #             pos = evals[evals > 0]
    #             rank = (np.sum(pos[:2]) / np.sum(pos)) if len(pos) > 2 else 1.0

    #         results.append({
    #             "window_ms": win_ms,
    #             "ff_uncorr": ff_uncorr,
    #             "ff_corr": ff_corr,
    #             "ff_uncorr_mean": np.nanmean(ff_uncorr),
    #             "ff_corr_mean": np.nanmean(ff_corr),
    #             "alpha": alpha,
    #             "fem_rank_ratio": rank,
    #             "n_samples": n_samples,
    #             'Erates': Erate,
    #             'count_e': count_e
    #         })

    #         mats_save.append({
    #             "Total": Ctotal,
    #             "PSTH": Cpsth,
    #             "FEM": Cfem,
    #             "Intercept": Crate,
    #             "NoiseCorrU": NoiseCorrU,
    #             "NoiseCorrC": NoiseCorrC,
    #             "PSTH_A": PSTH_A,
    #             "PSTH_B": PSTH_B,
    #         })

    #         win_key = float(win_ms)
    #         self.window_summaries[win_key] = {
    #             "bin_centers": bin_centers,
    #             "binned_covs": Ceye,          # SECOND MOMENTS (kept name for compatibility)
    #             "bin_counts": count_e,
    #             "Sigma_Intercept": Crate,          # COVARIANCE
    #             "Sigma_PSTH": Cpsth,            # COVARIANCE
    #             "Sigma_Total": Ctotal,
    #             "Sigma_FEM": Cfem,
    #             "mean_counts": Erate,
    #         }

    #     return results, mats_save

    # utility for analyzing the analysis at the resolution of a single neuron or pair
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


# call analysis on a dataset
from eval.eval_stack_multidataset import load_model, load_single_dataset, run_bps_analysis, run_qc_analysis
from eval.eval_stack_utils import run_model, rescale_rhat, ccnorm_split_half_variable_trials

def run_mcfarland_on_dataset(model, dataset_idx, windows = [10, 20, 40, 80],
        plot=False, total_spikes_threshold=200, valid_time_bins=120, dt=1/120,
        rescale=True, batch_size=128,
        n_shuffles=0, seed=42):
    '''
    Run the Covariance Decomposition on a dataset.

    Inputs:
    -------
    dataset_configs : list
        List of dataset configuration dictionaries
    dataset_idx : int
        Index of the dataset to run on
    windows : list
        List of window sizes to run on (in ms)
    total_spikes_threshold : int
        Minimum number of spikes for a neuron to be included
    valid_time_bins : int
        Maximum number of time bins from each trial included
    dt : float
        Time bin size (in seconds)
    
    Returns:
    --------
    output : dict
        Dictionary containing the results of the analysis

    '''

    print(f"Dataset {dataset_idx}: {model.names[dataset_idx]}")
    train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
    dataset_name = model.names[dataset_idx]

    bps_results = run_bps_analysis(
            model, train_data, val_data, dataset_idx,
            batch_size=batch_size, rescale=rescale
        )
    
    qc_results = run_qc_analysis(
            dataset_name, dataset_config['cids'], dataset_idx
        )
        
    sess = train_data.dsets[0].metadata['sess']
    # ppd = train_data.dsets[0].metadata['ppd']
    cids = dataset_config['cids']
    print(f"Running on {sess.name}")

    # get fixrsvp inds and make one dataaset object
    inds = torch.concatenate([
            train_data.get_dataset_inds('fixrsvp'),
            val_data.get_dataset_inds('fixrsvp')
        ], dim=0)

    dataset = train_data.shallow_copy()
    dataset.inds = inds

    # Getting key variables
    dset_idx = inds[:,0].unique().item()
    trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
    trials = np.unique(trial_inds)

    NC = dataset.dsets[dset_idx]['robs'].shape[1]
    T = np.max(dataset.dsets[dset_idx].covariates['psth_inds'][:].numpy()).item() + 1
    NT = len(trials)

    fixation = np.hypot(dataset.dsets[dset_idx]['eyepos'][:,0].numpy(), dataset.dsets[dset_idx]['eyepos'][:,1].numpy()) < 1

    # Loop over trials and align responses
    robs = np.nan*np.zeros((NT, T, NC))
    rhat = np.nan*np.zeros((NT, T, NC))
    dfs = np.nan*np.zeros((NT, T, NC))
    eyepos = np.nan*np.zeros((NT, T, 2))
    fix_dur =np.nan*np.zeros((NT,))

    for itrial in tqdm(range(NT)):
        # print(f"Trial {itrial}/{NT}")
        ix = trials[itrial] == trial_inds
        ix = ix & fixation
        if np.sum(ix) == 0:
            continue
        
        # run model
        stim_inds = np.where(ix)[0]
        stim_inds = stim_inds[:,None] - np.array(dataset_config['keys_lags']['stim'])[None,:]
        stim = dataset.dsets[dset_idx]['stim'][stim_inds].permute(0,2,1,3,4)
        behavior = dataset.dsets[dset_idx]['behavior'][ix]

        out = run_model(model, {'stim': stim, 'behavior': behavior}, dataset_idx=dataset_idx)

        psth_inds = dataset.dsets[dset_idx].covariates['psth_inds'][ix].numpy()
        fix_dur[itrial] = len(psth_inds)
        robs[itrial][psth_inds] = dataset.dsets[dset_idx]['robs'][ix].numpy()
        rhat[itrial][psth_inds] = out['rhat'].detach().cpu().numpy()
        dfs[itrial][psth_inds] = dataset.dsets[dset_idx]['dfs'][ix].numpy()
        eyepos[itrial][psth_inds] = dataset.dsets[dset_idx]['eyepos'][ix].numpy()
    

    good_trials = fix_dur > 20
    robs = robs[good_trials]
    rhat = rhat[good_trials]
    dfs = dfs[good_trials]
    # robs[dfs!=True]=np.nan
    # rhat[dfs!=True]=np.nan
    eyepos = eyepos[good_trials]
    fix_dur = fix_dur[good_trials]

    if plot:
        ind = np.argsort(fix_dur)[::-1]
        plt.subplot(1,2,1)
        plt.imshow(eyepos[ind,:,0])
        plt.xlim(0, 160)
        plt.subplot(1,2,2)
        plt.imshow(np.nanmean(robs,2)[ind])
        plt.xlim(0, 160)

    # Run the analysis
    output = {}
    output['sess'] = sess.name
    output['cids'] = np.array(cids)


    # 1. Setup
    
    # valid_mask should be True where data is good (no fix breaks)
    neuron_mask = np.where(np.nansum(robs, (0,1))>total_spikes_threshold)[0]
    valid_mask = np.isfinite(np.sum(robs[:,:,neuron_mask], axis=2)) & np.isfinite(np.sum(eyepos, axis=2))
    
    NC = robs.shape[2]
    
    print(f"Using {len(neuron_mask)} neurons / {NC} total")
    iix = np.arange(valid_time_bins)

    robs_used = robs[:,iix][:,:,neuron_mask]
    eyepos_used = eyepos[:,iix]
    valid_mask_used = valid_mask[:,iix]
    analyzer = DualWindowAnalysis(robs_used, eyepos_used, valid_mask_used, dt=dt)

    # 2. Run Sweep
    results, last_mats = analyzer.run_sweep(windows, t_hist_ms=50, n_bins=15, n_shuffles=n_shuffles, seed=seed)
    
    output['neuron_mask'] = neuron_mask
    output['bps_results'] = bps_results
    output['qc_results'] = qc_results
    output['windows'] = windows
    output['cids_used'] = output['cids'][neuron_mask]
    output['results'] = results
    output['last_mats'] = last_mats

    rhat_used = rhat[:,iix][:,:,neuron_mask]
    dfs_used = dfs[:,iix][:,:,neuron_mask]

    rtrials, rtime, rneuron = rhat_used.shape
    if rescale:
        # reshape into (rtrials*rtime, rneuron)
        rhat_reshape = rhat_used.reshape(rtrials*rtime, rneuron)
        robs_reshape = robs_used.reshape(rtrials*rtime, rneuron)
        print(valid_mask.shape, valid_mask[:,iix].shape)
        valid_mask_reshape = valid_mask[:,iix].reshape(rtrials*rtime, 1).repeat(rneuron, axis=1)
        # rescale per neuron with affine transform
        print(robs_reshape.shape, rhat_reshape.shape, valid_mask_reshape.shape)
        rhat_rescaled, _ = rescale_rhat(torch.from_numpy(robs_reshape), torch.from_numpy(rhat_reshape), torch.from_numpy(valid_mask_reshape), mode='affine')
        rhat_used = rhat_rescaled.reshape(rtrials, rtime, rneuron).detach().cpu().numpy()
    
    # get ccnorm using split-half estimate of the max (Schoppe et al., 2016)
    # do it twice and take averate, set to nan if diff is excessive, which means our estimator is unreliable
    ccnorm, ccabs, ccmax, cchalf_mean, cchalf_n = ccnorm_split_half_variable_trials(robs_used, rhat_used, dfs_used, return_components=True, n_splits=500)
    ccnorm2, ccabs2, ccmax2, cchalf_mean2, cchalf_n2 = ccnorm_split_half_variable_trials(robs_used, rhat_used, dfs_used, return_components=True, n_splits=500)

    bad = (ccnorm - ccnorm2)**2 > .01
    ccnorm = 0.5 * (ccnorm + ccnorm2)
    ccabs = 0.5 * (ccabs + ccabs2)
    ccmax = 0.5 * (ccmax + ccmax2)
    cchalf_mean = 0.5 * (cchalf_mean + cchalf_mean2)
    cchalf_n = 0.5 * (cchalf_n + cchalf_n2)
    
    ccnorm[bad] = np.nan

    output['ccnorm'] = {'ccnorm': ccnorm, 'ccabs': ccabs, 'ccmax': ccmax, 'cchalf_mean': cchalf_mean, 'cchalf_n': cchalf_n}

    # redo variance analysis on residuals. We use this to establish variance explained metrics and subspace alignment.
    residuals = robs_used - rhat_used

    # plt.figure(figsize=(10,5))
    # plt.subplot(1,2,1)
    # plt.imshow(robs_used[:,:,0])
    # plt.subplot(1,2,2)
    # plt.imshow(rhat_used[:,:,0])
    # plt.show()
    analyzer_residuals = DualWindowAnalysis(residuals, eyepos[:,iix], valid_mask[:,iix], dt=dt)
    t_hist_ms = 50
    results_residuals, last_mats_residuals = analyzer_residuals.run_sweep(windows, t_hist_ms=t_hist_ms, n_bins=15)
    
    
    output['results_residuals'] = results_residuals
    output['last_mats_residuals'] = last_mats_residuals

    if plot:
        window_idx = 1
        Ctotal = last_mats[window_idx]['Total']
        Cfem = last_mats[window_idx]['FEM']
        Crate = last_mats[window_idx]['Intercept']
        Cpsth = last_mats[window_idx]['PSTH']
        CnoiseU = last_mats[window_idx]['NoiseCorrU']
        CnoiseC = last_mats[window_idx]['NoiseCorrC']
        FF_uncorr = results[window_idx]['ff_uncorr']
        FF_corr = results[window_idx]['ff_corr']
        Erates = results[window_idx]['Erates']


        v = np.max(Cfem.flatten())
        plt.figure()
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
        v = .2
        plt.imshow(CnoiseU, vmin=-v, vmax=v)
        plt.colorbar()
        plt.title('Noise (Uncorrected))')
        plt.subplot(1,2,2)
        plt.imshow(CnoiseC, vmin=-v, vmax=v)
        plt.colorbar()
        plt.title('Noise (Corrected) ')

        def get_upper_triangle(C):
            rows, cols = np.triu_indices_from(C, k=1)
            v = C[rows, cols]
            return v

        rho_uncorr = get_upper_triangle(CnoiseU)
        rho_corr = get_upper_triangle(CnoiseC)

        plt.figure()
        plt.plot(rho_uncorr, rho_corr, '.', alpha=0.1)
        # plot mean
        plt.plot(rho_uncorr.mean(), rho_corr.mean(), 'ro')
        plt.plot(plt.xlim(), plt.xlim(), 'k')
        plt.axhline(0, color='k', linestyle='--')
        plt.axvline(0, color='k', linestyle='--')
        plt.xlabel('Correlation (Uncorrected)')
        plt.ylabel('Correlation (Corrected)')
        plt.title('Correlation vs Window Size')
        plt.show()

        # Plot Fano Factor Scaling
        window_ms = [results[i]['window_ms'] for i in range(len(results))]
        ff_uncorr = np.zeros_like(window_ms, dtype=np.float64)
        ff_uncorr_std = np.zeros_like(window_ms, dtype=np.float64)
        ff_uncorr_se = np.zeros_like(window_ms, dtype=np.float64)
        ff_corr = np.zeros_like(window_ms, dtype=np.float64)
        ff_corr_std = np.zeros_like(window_ms, dtype=np.float64)
        ff_corr_se = np.zeros_like(window_ms, dtype=np.float64)

        for iwindow in range(len(window_ms)):
            Erates = results[iwindow]['Erates']
            good = Erates > 0.4
            ff_uncorr[iwindow] = np.nanmedian(results[iwindow]['ff_uncorr'][good])
            ff_corr[iwindow] = np.nanmedian(results[iwindow]['ff_corr'][good])
            ff_uncorr_std[iwindow] = np.nanstd(results[iwindow]['ff_uncorr'][good])
            ff_corr_std[iwindow] = np.nanstd(results[iwindow]['ff_corr'][good])
            ff_uncorr_se[iwindow] = ff_uncorr_std[iwindow] / np.sqrt(len(results[iwindow]['ff_uncorr'][good]))
            ff_corr_se[iwindow] = ff_corr_std[iwindow] / np.sqrt(len(results[iwindow]['ff_corr'][good]))

        plt.figure(figsize=(8, 6))
        plt.plot(window_ms, ff_uncorr, 'o-', label='Standard (Uncorrected)')
        plt.plot(window_ms, ff_corr, 'o-', label='FEM-Corrected')
        # plot error bars
        plt.fill_between(window_ms, ff_uncorr - ff_uncorr_se, ff_uncorr + ff_uncorr_se, alpha=0.2)
        plt.fill_between(window_ms, ff_corr - ff_corr_se, ff_corr + ff_corr_se, alpha=0.2)

        plt.axhline(1.0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Count Window (ms)')
        plt.ylabel('Mean Fano Factor')
        plt.title('Integration of Noise: FEM Correction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return output, analyzer


# extract metrics from the above analysis
# def extract_metrics(outputs):
#     n = len(outputs[0]['results'])
#     # fig, axs = plt.subplots(1,n, figsize=(3*n, 3), sharex=False, sharey=False)
#     metrics = []
#     for i in range(n):
        
#         ff_uncorrs = []
#         ff_corrs = []
#         erates = []
#         rhos_uncorr = []
#         rhos_corr = []
#         alphas = []
#         for j in range(len(outputs)):
#             window_ms = outputs[j]['results'][i]['window_ms']
#             ff_uncorr = outputs[j]['results'][i]['ff_uncorr']
#             ff_corr = outputs[j]['results'][i]['ff_corr']
#             Erates = outputs[j]['results'][i]['Erates']
#             alpha = outputs[j]['results'][i]['alpha']
            
#             CnoiseU = outputs[j]['last_mats'][i]['NoiseCorrU']
#             CnoiseC = outputs[j]['last_mats'][i]['NoiseCorrC']
#             rho_uncorr = get_upper_triangle(CnoiseU)
#             rho_corr = get_upper_triangle(CnoiseC)

#             valid = Erates > 0.1
#             ff_uncorrs.append(ff_uncorr[valid])
#             ff_corrs.append(ff_corr[valid])
#             erates.append(Erates[valid])
#             rhos_uncorr.append(rho_uncorr)
#             rhos_corr.append(rho_corr)
#             alphas.append(alpha[valid])

#             # axs[i].plot(Erates, ff_uncorr*Erates, 'r.', alpha=0.1)
#             # axs[i].plot(Erates, ff_corr*Erates, 'b.', alpha=0.1)
#             # xd = [0, np.percentile(Erates[valid], 99)]
#             # axs[i].plot(xd, xd, 'k--', alpha=0.5)
#             # axs[i].set_xlim(xd)
#             # axs[i].set_ylim(xd[0], xd[1]*2)
            
        
#         metrics.append({'window_ms': window_ms,
#                     'uncorr': np.concatenate(ff_uncorrs),
#                     'corr': np.concatenate(ff_corrs),
#                     'erate': np.concatenate(erates),
#                     'alpha': np.concatenate(alphas),
#                     'rho_uncorr': np.concatenate(rhos_uncorr),
#                     'rho_corr': np.concatenate(rhos_corr),
#                     })
#     return metrics

# def extract_metrics(outputs, min_total_spikes=50):
#     """
#     Extracts metrics for both Real and Shuffled data.
    
#     Returns metrics list where 'shuff_*' keys contain matrices of shape:
#     (Total_Valid_Neurons, N_shuffles)
#     """
#     n_windows = len(outputs[0]['results'])
#     metrics = []
    
#     for i in range(n_windows):
        
#         # Real Data Containers
#         ff_uncorrs, ff_corrs, erates, alphas = [], [], [], []
#         rhos_uncorr, rhos_corr = [], []
        
#         # Shuffled Data Containers (Blocks of (N_neurons, N_shuffles))
#         shuff_alphas_blocks = []
#         shuff_ff_uncorr_blocks = [] 
#         shuff_ff_corr_blocks = []
        
#         for j in range(len(outputs)):
#             # --- REAL DATA ---
#             res = outputs[j]['results'][i]
#             mats = outputs[j]['last_mats'][i]
            
#             window_ms = res['window_ms']
#             Erates = res['Erates']
#             n_samples = res['n_samples']
            
#             # Filter
#             total_spikes = Erates * n_samples
#             valid = total_spikes > min_total_spikes
            
#             # Store Real Metrics
#             ff_uncorrs.append(res['ff_uncorr'][valid])
#             ff_corrs.append(res['ff_corr'][valid])
#             erates.append(Erates[valid])
#             alphas.append(res['alpha'][valid])
            
#             # Covariances
#             rho_uncorr = get_upper_triangle(index_cov(mats['NoiseCorrU'], valid))
#             rho_corr = get_upper_triangle(index_cov(mats['NoiseCorrC'], valid))
#             rhos_uncorr.append(rho_uncorr)
#             rhos_corr.append(rho_corr)

#             # --- SHUFFLED DATA ---
#             if 'Shuffled_Intercepts' in mats and len(mats['Shuffled_Intercepts']) > 0:
                
#                 Ctotal = mats['Total']
#                 Cpsth = mats['PSTH']
                
#                 # Pre-calculate constants for this dataset
#                 var_psth = np.diag(Cpsth)
                
#                 # FF Uncorrected is constant across shuffles (depends only on Ctotal, Cpsth)
#                 # We calculate it once here
#                 ff_uncorr_const = res['ff_uncorr'][valid]
                
#                 # Containers for columns (shuffles) for this dataset
#                 ds_shuff_alphas_cols = []
#                 ds_shuff_ff_uncorr_cols = []
#                 ds_shuff_ff_corr_cols = []

#                 for Crate_s in mats['Shuffled_Intercepts']:
#                     # 1. Calculate Alpha
#                     var_rate_s = np.diag(Crate_s)
#                     with np.errstate(divide='ignore', invalid='ignore'):
#                         alpha_s = var_psth / var_rate_s
                    
#                     ds_shuff_alphas_cols.append(alpha_s[valid])

#                     # 2. Calculate Fano Factors
#                     # Corrected (Varies per shuffle)
#                     CnoiseC_s = Ctotal - Crate_s
#                     CnoiseC_s = 0.5 * (CnoiseC_s + CnoiseC_s.T)
#                     ff_corr_s = np.diag(CnoiseC_s) / Erates
                    
#                     ds_shuff_ff_corr_cols.append(ff_corr_s[valid])
                    
#                     # Uncorrected (Constant, but repeated for shape consistency)
#                     ds_shuff_ff_uncorr_cols.append(ff_uncorr_const)
                
#                 # Stack columns to make (N_valid_in_dataset, N_shuffles)
#                 if len(ds_shuff_alphas_cols) > 0:
#                     shuff_alphas_blocks.append(np.stack(ds_shuff_alphas_cols, axis=1))
#                     shuff_ff_uncorr_blocks.append(np.stack(ds_shuff_ff_uncorr_cols, axis=1))
#                     shuff_ff_corr_blocks.append(np.stack(ds_shuff_ff_corr_cols, axis=1))

#         # Concatenate across datasets (Axis 0 = Neurons)
#         if len(shuff_alphas_blocks) > 0:
#             cat_shuff_alphas = np.concatenate(shuff_alphas_blocks, axis=0)
#             cat_shuff_uncorr = np.concatenate(shuff_ff_uncorr_blocks, axis=0)
#             cat_shuff_corr = np.concatenate(shuff_ff_corr_blocks, axis=0)
#         else:
#             cat_shuff_alphas = np.array([])
#             cat_shuff_uncorr = np.array([])
#             cat_shuff_corr = np.array([])

#         metrics.append({
#             'window_ms': window_ms,
#             'uncorr': np.concatenate(ff_uncorrs),
#             'corr': np.concatenate(ff_corrs),
#             'erate': np.concatenate(erates),
#             'alpha': np.concatenate(alphas),
#             'rho_uncorr': np.concatenate(rhos_uncorr),
#             'rho_corr': np.concatenate(rhos_corr),
            
#             # Shuffled Stats: (Total_Valid_Neurons, N_shuffles)
#             'shuff_uncorr': cat_shuff_uncorr, 
#             'shuff_corr': cat_shuff_corr,
#             'shuff_alphas': cat_shuff_alphas, 
#         })
        
#     return metrics

import numpy as np

# ----------------------------
# guard-railed covariance->correlation
# ----------------------------
def cov_to_corr_safe(C, min_var=1e-3, eps=1e-6, set_diag_zero=True):
    """
    Converts covariance to correlation with guard rails.

    - Any neuron with variance <= min_var or non-finite variance becomes invalid.
    - Correlations involving invalid neurons are NaN.
    - Correlations are clipped to [-1+eps, 1-eps] to avoid atanh blowups.
    - Diagonal can be set to 0 (standard for noise corr).
    
    Returns
    -------
    R : (N,N) float array
    diag_var : (N,) float array of variances
    valid_neuron : (N,) bool array
    info : dict of diagnostics
    """
    C = np.asarray(C, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"C must be square (N,N), got {C.shape}")

    N = C.shape[0]
    diag_var = np.diag(C).copy()

    valid_neuron = np.isfinite(diag_var) & (diag_var > min_var)
    std = np.full(N, np.nan, dtype=np.float64)
    std[valid_neuron] = np.sqrt(diag_var[valid_neuron])

    outer = np.outer(std, std)  # rows/cols of invalid neurons become NaN

    with np.errstate(divide='ignore', invalid='ignore'):
        R = C / outer

    # clip *finite* entries only (leave NaNs)
    finite = np.isfinite(R)
    # count near-boundary before clip
    n_finite = int(finite.sum())
    n_over = int(np.sum(finite & (R > (1 - eps))))
    n_under = int(np.sum(finite & (R < (-1 + eps))))

    R = np.where(finite, np.clip(R, -1 + eps, 1 - eps), np.nan)

    if set_diag_zero:
        np.fill_diagonal(R, 0.0)

    info = dict(
        N=N,
        n_valid_neuron=int(valid_neuron.sum()),
        n_invalid_neuron=int((~valid_neuron).sum()),
        n_finite_entries=n_finite,
        n_clipped_high=n_over,
        n_clipped_low=n_under,
    )
    return R, diag_var, valid_neuron, info


# # ----------------------------
# # robust summaries for correlation distributions
# # ----------------------------
# def fisher_z_mean(rho, eps=1e-6):
#     """
#     Mean Fisher z of correlations (robust mean for rho).
#     Returns mean(z); you can back-transform via tanh(mean_z) if desired.
#     """
#     rho = np.asarray(rho, dtype=np.float64).reshape(-1)
#     rho = rho[np.isfinite(rho)]
#     if rho.size == 0:
#         return np.nan
#     rho = np.clip(rho, -1 + eps, 1 - eps)
#     z = np.arctanh(rho)
#     return np.nanmean(z)

# def fisher_z_ci_from_samples(z_samples, ci=0.95):
#     """
#     CI on mean-z or on per-shuffle mean-z distributions (already in z space).
#     """
#     z_samples = np.asarray(z_samples, dtype=np.float64).reshape(-1)
#     z_samples = z_samples[np.isfinite(z_samples)]
#     if z_samples.size == 0:
#         return np.nan, (np.nan, np.nan)
#     alpha = (1 - ci) / 2
#     lo, hi = np.percentile(z_samples, [100*alpha, 100*(1-alpha)])
#     return np.nanmean(z_samples), (lo, hi)


# def project_to_psd(C, eps=0.0):
#     C = 0.5 * (C + C.T)
#     w, V = np.linalg.eigh(C)
#     w = np.maximum(w, eps)
#     return (V * w) @ V.T

# def cov_diagnostics(C, name="C"):
#     C = 0.5*(C+C.T)
#     d = np.diag(C)
#     w = np.linalg.eigvalsh(C)
#     off = C.copy()
#     np.fill_diagonal(off, np.nan)
#     return dict(
#         name=name,
#         diag_min=float(np.nanmin(d)),
#         diag_med=float(np.nanmedian(d)),
#         diag_neg_frac=float(np.mean(d <= 0)),
#         off_med=float(np.nanmedian(off)),
#         off_q01=float(np.nanpercentile(off, 1)),
#         off_q99=float(np.nanpercentile(off, 99)),
#         eig_min=float(np.min(w)),
#         eig_neg_frac=float(np.mean(w < 0)),
#     )



# # ----------------------------
# # extract_metrics with shuffle nulls for noise correlations
# # ----------------------------
# def extract_metrics(outputs, min_total_spikes=50, min_var=1e-3, eps_rho=1e-6):
#     """
#     Extracts per-window metrics for REAL data and SHUFFLE controls.

#     Adds proper shuffle correction for noise correlations by computing, for each shuffle,
#     a summary statistic:
#         shuff_rho_c_meanz : mean Fisher-z of corrected noise correlations
#     (optionally also uncorrected if you want it)

#     Guard rails:
#       - Uses cov_to_corr_safe(Cnoise*, min_var=min_var, eps=eps_rho)
#       - Drops neurons with too few spikes (min_total_spikes)
#       - Correlations clipped to avoid atanh blowups
#       - Returns diagnostics to track invalid neurons/pairs and clipping rates.

#     Returns
#     -------
#     metrics : list of dict, one per window_ms, with keys including:
#       - 'rho_uncorr', 'rho_corr' : concatenated upper triangle rho arrays (real)
#       - 'rho_u_meanz', 'rho_c_meanz' : Fisher mean-z summaries (real)
#       - 'rho_u_meanz_by_ds', 'rho_c_meanz_by_ds' : per-dataset mean-z (better for CI)
#       - 'shuff_rho_c_meanz' : array of per-shuffle mean-z (concatenated across datasets)
#       - plus your existing FF-related keys
#       - 'diag' : diagnostics about validity/clipping/pair counts
#     """
#     n_windows = len(outputs[0]['results'])
#     metrics = []

#     for i in range(n_windows):

#         # Real data containers (concatenated across datasets)
#         ff_uncorrs, ff_corrs, erates, alphas = [], [], [], []
#         rhos_uncorr, rhos_corr = [], []

#         # Real per-dataset summaries (for proper CIs)
#         rho_u_meanz_by_ds = []
#         rho_c_meanz_by_ds = []

#         # Shuffle containers (concatenated across datasets)
#         shuff_alphas_blocks = []
#         shuff_ff_uncorr_blocks = []
#         shuff_ff_corr_blocks = []

#         # Shuffle noise-corr summaries (per shuffle)
#         shuff_rho_c_meanz_all = []  # concat across datasets
#         # If you later want an uncorrected shuffle too, you can add shuff_rho_u_meanz_all.

#         # Diagnostics accumulation
#         diag = dict(
#             window_ms=None,
#             real=dict(n_ds=0, n_pairs_total=0, n_pairs_finite=0, n_pairs_used=0,
#                       n_invalid_neuron_total=0, n_clipped_total=0),
#             shuff=dict(n_shuffles_total=0, n_pairs_total=0, n_pairs_used=0,
#                        n_invalid_neuron_total=0, n_clipped_total=0),
#         )

#         for j in range(len(outputs)):
#             # --- REAL DATA ---
#             res = outputs[j]['results'][i]
#             mats = outputs[j]['last_mats'][i]

#             window_ms = res['window_ms']
#             Erates = np.asarray(res['Erates'], dtype=np.float64).reshape(-1)
#             n_samples = res['n_samples']

#             diag["window_ms"] = window_ms

#             # Filter by total spikes (per neuron)
#             total_spikes = Erates * n_samples
#             valid = total_spikes > min_total_spikes
#             if valid.sum() < 5:
#                 # Too few neurons in this dataset/window for stable corr
#                 continue

#             diag["real"]["n_ds"] += 1

#             # Store real FF metrics
#             ff_uncorrs.append(np.asarray(res['ff_uncorr'])[valid])
#             ff_corrs.append(np.asarray(res['ff_corr'])[valid])
#             erates.append(Erates[valid])
#             alphas.append(np.asarray(res['alpha'])[valid])

#             # 
#             # Build real noise covariances
#             Ctotal = np.asarray(mats['Total'], dtype=np.float64)
#             Cpsth  = np.asarray(mats['PSTH'],  dtype=np.float64)
#             Crate  = np.asarray(mats['Intercept'],  dtype=np.float64)  # <-- you need the real Crate saved

#             CnoiseU = 0.5 * ((Ctotal - Cpsth) + (Ctotal - Cpsth).T)
#             CnoiseC = 0.5 * ((Ctotal - Crate) + (Ctotal - Crate).T)

#             # Fixed neuron set across conditions
#             valid_spikes = (Erates * n_samples) > min_total_spikes
#             validU = valid_spikes & np.isfinite(np.diag(CnoiseU)) & (np.diag(CnoiseU) > min_var)
#             validC = valid_spikes & np.isfinite(np.diag(CnoiseC)) & (np.diag(CnoiseC) > min_var)
#             valid_fixed = validU & validC

#             # If too few neurons, skip dataset/window
#             if valid_fixed.sum() < 5:
#                 continue

#             # PSD project (for correlation only)
#             CnoiseU_psd = project_to_psd(CnoiseU, eps=0.0)
#             CnoiseC_psd = project_to_psd(CnoiseC, eps=0.0)

#             RU, *_ = cov_to_corr_safe(CnoiseU_psd, min_var=min_var, eps=eps_rho)
#             RC, *_ = cov_to_corr_safe(CnoiseC_psd, min_var=min_var, eps=eps_rho)

#             rho_u = get_upper_triangle(index_cov(RU, valid_fixed))
#             rho_c = get_upper_triangle(index_cov(RC, valid_fixed))

#             z_u = fisher_z_mean(rho_u, eps=eps_rho)
#             z_c = fisher_z_mean(rho_c, eps=eps_rho)
#             delta_real = z_c - z_u

#             # Shuffle null on *delta*
#             delta_null = []
#             for Crate_s in mats['Shuffled_Intercepts']:
#                 Crate_s = np.asarray(Crate_s, dtype=np.float64)
#                 CnoiseC_s = 0.5 * ((Ctotal - Crate_s) + (Ctotal - Crate_s).T)

#                 CnoiseC_s_psd = project_to_psd(CnoiseC_s, eps=0.0)
#                 RCs, *_ = cov_to_corr_safe(CnoiseC_s_psd, min_var=min_var, eps=eps_rho)

#                 rho_cs = get_upper_triangle(index_cov(RCs, valid_fixed))
#                 z_cs = fisher_z_mean(rho_cs, eps=eps_rho)

#                 delta_null.append(z_cs - z_u)

#             delta_null = np.asarray(delta_null, dtype=float)
            

#             rhos_uncorr.append(rho_u)
#             rhos_corr.append(rho_c)

#             # per-dataset fisher means (for CI across datasets)
#             rho_u_meanz_by_ds.append(z_u)
#             rho_c_meanz_by_ds.append(z_c)

#             # # diagnostics (real)
#             # diag["real"]["n_invalid_neuron_total"] += int(np.size(valid_pair_neurons) - np.sum(valid_pair_neurons))
#             # diag["real"]["n_clipped_total"] += int(infoU.get("n_clipped_high", 0) + infoU.get("n_clipped_low", 0)
#             #                                      + infoC.get("n_clipped_high", 0) + infoC.get("n_clipped_low", 0))
#             # diag["real"]["n_pairs_total"] += int(np.sum(valid_pair_neurons) * (np.sum(valid_pair_neurons) - 1) / 2)
#             # diag["real"]["n_pairs_used"] += int(np.sum(np.isfinite(rho_uncorr)) + np.sum(np.isfinite(rho_corr)))

#             # --- SHUFFLED DATA ---
#             # Proper shuffle null for correlations must build CnoiseC_s and convert to corr
#             if 'Shuffled_Intercepts' in mats and len(mats['Shuffled_Intercepts']) > 0:

#                 Ctotal = np.asarray(mats['Total'], dtype=np.float64)
#                 Cpsth  = np.asarray(mats['PSTH'], dtype=np.float64)

#                 # For alpha + FFs (your existing behavior)
#                 var_psth = np.diag(Cpsth)
#                 ff_uncorr_const = np.asarray(res['ff_uncorr'], dtype=np.float64)  # full length

#                 ds_shuff_alphas_cols = []
#                 ds_shuff_ff_uncorr_cols = []
#                 ds_shuff_ff_corr_cols = []

#                 for Crate_s in mats['Shuffled_Intercepts']:
#                     Crate_s = np.asarray(Crate_s, dtype=np.float64)

#                     # 1) alpha
#                     var_rate_s = np.diag(Crate_s)
#                     with np.errstate(divide='ignore', invalid='ignore'):
#                         alpha_s = var_psth / var_rate_s
#                     ds_shuff_alphas_cols.append(alpha_s[valid])

#                     # 2) FFs
#                     CnoiseC_s = Ctotal - Crate_s
#                     CnoiseC_s = 0.5 * (CnoiseC_s + CnoiseC_s.T)
#                     ff_corr_s = np.diag(CnoiseC_s) / Erates  # full length
#                     ds_shuff_ff_corr_cols.append(ff_corr_s[valid])

#                     ds_shuff_ff_uncorr_cols.append(ff_uncorr_const[valid])  # shape consistency

#                     # 3) Noise correlation null summary (THIS is the missing piece)
#                     R_s, _, v_s, info_s = cov_to_corr_safe(CnoiseC_s, min_var=min_var, eps=eps_rho)

#                     # enforce same neuron inclusion rule as real: spike-valid AND variance-valid
#                     valid_s = valid & v_s
#                     rho_corr_s = get_upper_triangle(index_cov(R_s, valid_s))

#                     shuff_rho_c_meanz_all.append(fisher_z_mean(rho_corr_s, eps=eps_rho))

#                     # diagnostics (shuffle)
#                     diag["shuff"]["n_shuffles_total"] += 1
#                     diag["shuff"]["n_invalid_neuron_total"] += int(np.size(valid_s) - np.sum(valid_s))
#                     diag["shuff"]["n_clipped_total"] += int(info_s.get("n_clipped_high", 0) + info_s.get("n_clipped_low", 0))
#                     diag["shuff"]["n_pairs_total"] += int(np.sum(valid_s) * (np.sum(valid_s) - 1) / 2)
#                     diag["shuff"]["n_pairs_used"] += int(np.sum(np.isfinite(rho_corr_s)))

#                 # Stack columns -> (N_valid, N_shuffles)
#                 if len(ds_shuff_alphas_cols) > 0:
#                     shuff_alphas_blocks.append(np.stack(ds_shuff_alphas_cols, axis=1))
#                     shuff_ff_uncorr_blocks.append(np.stack(ds_shuff_ff_uncorr_cols, axis=1))
#                     shuff_ff_corr_blocks.append(np.stack(ds_shuff_ff_corr_cols, axis=1))

#         # concatenate across datasets (neurons)
#         if len(shuff_alphas_blocks) > 0:
#             cat_shuff_alphas = np.concatenate(shuff_alphas_blocks, axis=0)
#             cat_shuff_uncorr = np.concatenate(shuff_ff_uncorr_blocks, axis=0)
#             cat_shuff_corr   = np.concatenate(shuff_ff_corr_blocks, axis=0)
#         else:
#             cat_shuff_alphas = np.array([])
#             cat_shuff_uncorr = np.array([])
#             cat_shuff_corr   = np.array([])

#         # concatenate rhos (real)
#         rho_u_all = np.concatenate(rhos_uncorr) if len(rhos_uncorr) else np.array([])
#         rho_c_all = np.concatenate(rhos_corr)   if len(rhos_corr) else np.array([])

#         metrics.append({
#             "window_ms": diag["window_ms"],

#             # FF + rate metrics (your existing outputs)
#             "uncorr": np.concatenate(ff_uncorrs) if len(ff_uncorrs) else np.array([]),
#             "corr":   np.concatenate(ff_corrs)   if len(ff_corrs) else np.array([]),
#             "erate":  np.concatenate(erates)     if len(erates) else np.array([]),
#             "alpha":  np.concatenate(alphas)     if len(alphas) else np.array([]),

#             # Real noise correlations (raw for 2D hists)
#             "rho_uncorr": rho_u_all,
#             "rho_corr":   rho_c_all,

#             # Robust real summaries (z space)
#             "rho_u_meanz": fisher_z_mean(rho_u_all, eps=eps_rho),
#             "rho_c_meanz": fisher_z_mean(rho_c_all, eps=eps_rho),

#             # Per-dataset summaries (preferred for error bars)
#             "rho_u_meanz_by_ds": np.asarray(rho_u_meanz_by_ds, dtype=np.float64),
#             "rho_c_meanz_by_ds": np.asarray(rho_c_meanz_by_ds, dtype=np.float64),

#             # Shuffled stats (neurons x shuffles) — existing
#             "shuff_uncorr": cat_shuff_uncorr,
#             "shuff_corr":   cat_shuff_corr,
#             "shuff_alphas": cat_shuff_alphas,

#             # Shuffled noise-corr null summaries (one per shuffle)
#             "shuff_rho_c_meanz": np.asarray(shuff_rho_c_meanz_all, dtype=np.float64),

#             # Diagnostics
#             "diag": diag,
#             "params": dict(min_total_spikes=min_total_spikes, min_var=min_var, eps_rho=eps_rho),
#         })

#     return metrics

import numpy as np

# ----------------------------
# robust summaries for correlation distributions
# ----------------------------
def fisher_z_mean(rho, eps=1e-6):
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    rho = rho[np.isfinite(rho)]
    if rho.size == 0:
        return np.nan
    rho = np.clip(rho, -1 + eps, 1 - eps)
    return np.nanmean(np.arctanh(rho))

def project_to_psd(C, eps=0.0):
    C = np.asarray(C, dtype=np.float64)
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, eps)
    return (V * w) @ V.T

# def cov_diagnostics(C, name="C"):
#     C = np.asarray(C, dtype=np.float64)
#     C = 0.5 * (C + C.T)
#     d = np.diag(C)
#     w = np.linalg.eigvalsh(C)
#     off = C.copy()
#     np.fill_diagonal(off, np.nan)
#     return dict(
#         name=name,
#         N=int(C.shape[0]),
#         diag_min=float(np.nanmin(d)),
#         diag_med=float(np.nanmedian(d)),
#         diag_neg_frac=float(np.mean(d <= 0)),
#         off_med=float(np.nanmedian(off)),
#         off_q01=float(np.nanpercentile(off, 1)),
#         off_q99=float(np.nanpercentile(off, 99)),
#         eig_min=float(np.min(w)),
#         eig_med=float(np.median(w)),
#         eig_neg_frac=float(np.mean(w < 0)),
#     )

def cov_diagnostics(C, name="C", max_abs_warn=1e6):
    out = {"name": name, "status": "ok"}
    try:
        C = np.asarray(C, dtype=np.float64)
        out["shape"] = tuple(C.shape)

        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            out["status"] = "fail:not_square"
            return out

        # symmetrize
        C = 0.5 * (C + C.T)

        finite = np.isfinite(C)
        out["finite_frac"] = float(finite.mean())
        out["n_nonfinite"] = int(np.size(C) - finite.sum())

        # scale sanity
        absmax = np.nanmax(np.abs(C)) if finite.any() else np.nan
        out["abs_max"] = float(absmax)
        if np.isfinite(absmax) and absmax > max_abs_warn:
            out["status"] = "warn:huge_scale"

        d = np.diag(C)
        df = np.isfinite(d)
        out["diag_finite_frac"] = float(df.mean())
        out["diag_min"] = float(np.nanmin(d)) if df.any() else np.nan
        out["diag_med"] = float(np.nanmedian(d)) if df.any() else np.nan
        out["diag_neg_frac"] = float(np.mean((d <= 0) & df)) if df.any() else np.nan

        # off-diagonal summary
        off = C.copy()
        np.fill_diagonal(off, np.nan)
        off_f = np.isfinite(off)
        out["off_med"] = float(np.nanmedian(off)) if off_f.any() else np.nan
        out["off_q01"] = float(np.nanpercentile(off, 1)) if off_f.any() else np.nan
        out["off_q99"] = float(np.nanpercentile(off, 99)) if off_f.any() else np.nan

        # eigen diagnostics (can fail)
        try:
            # replace non-finite with 0 for eig attempt only (still marked in finite_frac)
            C_eig = np.where(np.isfinite(C), C, 0.0)
            w = np.linalg.eigvalsh(C_eig)
            out["eig_min"] = float(np.min(w))
            out["eig_med"] = float(np.median(w))
            out["eig_neg_frac"] = float(np.mean(w < 0))
        except Exception as e:
            out["status"] = "warn:eig_failed"
            out["eig_error"] = type(e).__name__

        return out

    except Exception as e:
        out["status"] = "fail:exception"
        out["error"] = type(e).__name__
        return out

def cov_to_corr_safe(C, min_var=1e-3, eps=1e-6, set_diag_zero=True):
    """
    Cov -> Corr with variance guard rails and clipping diagnostics.
    Returns:
      R, valid_neuron_mask, info
    """
    C = np.asarray(C, dtype=np.float64)
    C = 0.5 * (C + C.T)
    d = np.diag(C).copy()

    valid = np.isfinite(d) & (d > min_var)
    std = np.full_like(d, np.nan, dtype=np.float64)
    std[valid] = np.sqrt(d[valid])

    denom = np.outer(std, std)

    with np.errstate(divide='ignore', invalid='ignore'):
        R = C / denom

    finite = np.isfinite(R)
    # how many would exceed bounds prior to clipping?
    n_over = int(np.sum(finite & (R > (1 - eps))))
    n_under = int(np.sum(finite & (R < (-1 + eps))))

    R = np.where(finite, np.clip(R, -1 + eps, 1 - eps), np.nan)

    if set_diag_zero:
        np.fill_diagonal(R, 0.0)

    info = dict(
        n_valid_neuron=int(valid.sum()),
        n_invalid_neuron=int((~valid).sum()),
        n_clipped_high=n_over,
        n_clipped_low=n_under,
    )
    return R, valid, info


# ----------------------------
# extract_metrics with correct shuffle nulls for noise correlations
# ----------------------------
def extract_metrics(outputs, min_total_spikes=50, min_var=1e-3, eps_rho=1e-6,
                    psd_eps=0.0, diag_n_shuffles=0):
    """
    Extracts per-window metrics for REAL data and SHUFFLE controls.

    Noise-corr shuffle null is computed on the EFFECT:
        delta = mean_z(corrected) - mean_z(uncorrected)
    where mean_z is Fisher z mean of upper-triangle correlations
    computed on a FIXED neuron set (intersection across U/C).

    Parameters
    ----------
    min_total_spikes : neuron inclusion by spikes in this window
    min_var          : variance floor for cov->corr normalization
    eps_rho          : clip correlations to [-1+eps, 1-eps]
    psd_eps          : eigenvalue floor for PSD projection (0 is typical)
    diag_n_shuffles  : store diagnostics for up to this many shuffles per dataset/window
                       (0 disables to keep it light)

    Requires
    --------
    - mats keys: 'Total', 'PSTH', 'Intercept', optionally 'Shuffled_Intercepts'
    - helper funcs: index_cov(M, mask) and get_upper_triangle(M)
    """
    n_windows = len(outputs[0]['results'])
    metrics = []

    for i in range(n_windows):

        # --- aggregated real metrics across datasets ---
        ff_uncorrs, ff_corrs, erates, alphas = [], [], [], []
        rhos_uncorr, rhos_corr = [], []

        # per-dataset summaries (preferred for CI)
        z_u_by_ds = []
        z_c_by_ds = []
        delta_by_ds = []

        # shuffle FF blocks (your existing behavior)
        shuff_alphas_blocks = []
        shuff_ff_uncorr_blocks = []
        shuff_ff_corr_blocks = []

        # shuffle null summaries (concatenated across datasets)
        shuff_delta_meanz_all = []  # delta = z_cs - z_u
        shuff_zc_meanz_all = []     # optional: corrected z under shuffle

        # diagnostics
        diag = dict(
            window_ms=None,
            n_ds_used=0,
            real=dict(
                n_neuron_used_total=0,
                n_pairs_used_total=0,
                clipped_total=0,
                cov_stats=[],
            ),
            shuff=dict(
                n_shuffles_total=0,
                n_pairs_used_total=0,
                clipped_total=0,
                cov_stats_examples=[],  # only if diag_n_shuffles > 0
            ),
            params=dict(
                min_total_spikes=min_total_spikes,
                min_var=min_var,
                eps_rho=eps_rho,
                psd_eps=psd_eps,
                diag_n_shuffles=diag_n_shuffles,
            )
        )

        for j in range(len(outputs)):
            res = outputs[j]["results"][i]
            mats = outputs[j]["last_mats"][i]

            window_ms = res["window_ms"]
            diag["window_ms"] = window_ms

            Erates = np.asarray(res["Erates"], dtype=np.float64).reshape(-1)
            n_samples = float(res["n_samples"])

            # spike-count validity
            total_spikes = Erates * n_samples
            valid_spikes = total_spikes > min_total_spikes
            if valid_spikes.sum() < 5:
                continue

            # store FF metrics (neuron-level) on spike-valid neurons
            ff_uncorrs.append(np.asarray(res["ff_uncorr"], dtype=np.float64)[valid_spikes])
            ff_corrs.append(np.asarray(res["ff_corr"], dtype=np.float64)[valid_spikes])
            erates.append(Erates[valid_spikes])
            alphas.append(np.asarray(res["alpha"], dtype=np.float64)[valid_spikes])

            # --- build noise covariances ---
            Ctotal = np.asarray(mats["Total"], dtype=np.float64)
            Cpsth  = np.asarray(mats["PSTH"], dtype=np.float64)
            Crate  = np.asarray(mats["Intercept"], dtype=np.float64)  # <-- your real Crate

            CnoiseU = 0.5 * ((Ctotal - Cpsth) + (Ctotal - Cpsth).T)
            CnoiseC = 0.5 * ((Ctotal - Crate) + (Ctotal - Crate).T)

            # diagnostics on raw covariances (before PSD)
            diag["real"]["cov_stats"].append(cov_diagnostics(CnoiseU, name=f"ds{j}_CnoiseU"))
            diag["real"]["cov_stats"].append(cov_diagnostics(CnoiseC, name=f"ds{j}_CnoiseC"))

            # fixed neuron set across U/C, based on diagonal validity
            dU = np.diag(CnoiseU)
            dC = np.diag(CnoiseC)
            validU = valid_spikes & np.isfinite(dU) & (dU > min_var)
            validC = valid_spikes & np.isfinite(dC) & (dC > min_var)
            valid_fixed = validU & validC

            nN = int(valid_fixed.sum())
            if nN < 5:
                continue

            diag["n_ds_used"] += 1
            diag["real"]["n_neuron_used_total"] += nN
            diag["real"]["n_pairs_used_total"] += int(nN * (nN - 1) / 2)

            # PSD projection for correlation computation only
            CnoiseU_psd = project_to_psd(CnoiseU, eps=psd_eps)
            CnoiseC_psd = project_to_psd(CnoiseC, eps=psd_eps)

            RU, vU, infoU = cov_to_corr_safe(CnoiseU_psd, min_var=min_var, eps=eps_rho)
            RC, vC, infoC = cov_to_corr_safe(CnoiseC_psd, min_var=min_var, eps=eps_rho)

            # enforce our fixed mask (don’t let cov_to_corr redefine inclusion)
            rho_u = get_upper_triangle(index_cov(RU, valid_fixed))
            rho_c = get_upper_triangle(index_cov(RC, valid_fixed))

            rhos_uncorr.append(rho_u)
            rhos_corr.append(rho_c)

            z_u = fisher_z_mean(rho_u, eps=eps_rho)
            z_c = fisher_z_mean(rho_c, eps=eps_rho)
            z_u_by_ds.append(z_u)
            z_c_by_ds.append(z_c)
            delta_by_ds.append(z_c - z_u)

            diag["real"]["clipped_total"] += int(infoU["n_clipped_high"] + infoU["n_clipped_low"] +
                                                infoC["n_clipped_high"] + infoC["n_clipped_low"])

            # --- shuffle null on delta (z_cs - z_u) ---
            shuffs = mats.get("Shuffled_Intercepts", [])
            if shuffs is None or len(shuffs) == 0:
                continue

            # For your existing shuffle FF/alpha outputs (same as your old code)
            var_psth = np.diag(Cpsth)
            ff_uncorr_const = np.asarray(res["ff_uncorr"], dtype=np.float64)

            ds_shuff_alphas_cols = []
            ds_shuff_ff_uncorr_cols = []
            ds_shuff_ff_corr_cols = []

            # shuffle diagnostics examples
            shuff_diag_kept = 0

            for s_idx, Crate_s in enumerate(shuffs):
                Crate_s = np.asarray(Crate_s, dtype=np.float64)

                # alpha
                var_rate_s = np.diag(Crate_s)
                with np.errstate(divide="ignore", invalid="ignore"):
                    alpha_s = var_psth / var_rate_s
                ds_shuff_alphas_cols.append(alpha_s[valid_spikes])

                # corrected FF for this shuffle
                CnoiseC_s = 0.5 * ((Ctotal - Crate_s) + (Ctotal - Crate_s).T)
                ff_corr_s = np.diag(CnoiseC_s) / Erates
                ds_shuff_ff_corr_cols.append(ff_corr_s[valid_spikes])
                ds_shuff_ff_uncorr_cols.append(ff_uncorr_const[valid_spikes])

                # shuffle noise-corr effect (delta)
                CnoiseC_s_psd = project_to_psd(CnoiseC_s, eps=psd_eps)
                RCs, vCs, infoS = cov_to_corr_safe(CnoiseC_s_psd, min_var=min_var, eps=eps_rho)

                rho_cs = get_upper_triangle(index_cov(RCs, valid_fixed))
                z_cs = fisher_z_mean(rho_cs, eps=eps_rho)

                shuff_zc_meanz_all.append(z_cs)
                shuff_delta_meanz_all.append(z_cs - z_u)

                diag["shuff"]["n_shuffles_total"] += 1
                diag["shuff"]["n_pairs_used_total"] += int(np.sum(np.isfinite(rho_cs)))
                diag["shuff"]["clipped_total"] += int(infoS["n_clipped_high"] + infoS["n_clipped_low"])

                if diag_n_shuffles > 0 and shuff_diag_kept < diag_n_shuffles:
                    diag["shuff"]["cov_stats_examples"].append(cov_diagnostics(CnoiseC_s, name=f"ds{j}_shuff{s_idx}_CnoiseC"))
                    shuff_diag_kept += 1

            # stack shuffle FF/alpha columns -> (N_valid_spikes, N_shuffles)
            if len(ds_shuff_alphas_cols) > 0:
                shuff_alphas_blocks.append(np.stack(ds_shuff_alphas_cols, axis=1))
                shuff_ff_uncorr_blocks.append(np.stack(ds_shuff_ff_uncorr_cols, axis=1))
                shuff_ff_corr_blocks.append(np.stack(ds_shuff_ff_corr_cols, axis=1))

        # --- concatenate across datasets for this window ---
        rho_u_all = np.concatenate(rhos_uncorr) if len(rhos_uncorr) else np.array([])
        rho_c_all = np.concatenate(rhos_corr)   if len(rhos_corr) else np.array([])

        if len(shuff_alphas_blocks) > 0:
            cat_shuff_alphas = np.concatenate(shuff_alphas_blocks, axis=0)
            cat_shuff_uncorr = np.concatenate(shuff_ff_uncorr_blocks, axis=0)
            cat_shuff_corr   = np.concatenate(shuff_ff_corr_blocks, axis=0)
        else:
            cat_shuff_alphas = np.array([])
            cat_shuff_uncorr = np.array([])
            cat_shuff_corr   = np.array([])

        metrics.append({
            "window_ms": diag["window_ms"],

            # FF/rate metrics
            "uncorr": np.concatenate(ff_uncorrs) if len(ff_uncorrs) else np.array([]),
            "corr":   np.concatenate(ff_corrs)   if len(ff_corrs) else np.array([]),
            "erate":  np.concatenate(erates)     if len(erates) else np.array([]),
            "alpha":  np.concatenate(alphas)     if len(alphas) else np.array([]),

            # real noise corr raw
            "rho_uncorr": rho_u_all,
            "rho_corr":   rho_c_all,

            # robust pooled summaries (z space)
            "rho_u_meanz": fisher_z_mean(rho_u_all, eps=eps_rho),
            "rho_c_meanz": fisher_z_mean(rho_c_all, eps=eps_rho),

            # per-dataset summaries (preferred for CI/error bars)
            "rho_u_meanz_by_ds": np.asarray(z_u_by_ds, dtype=np.float64),
            "rho_c_meanz_by_ds": np.asarray(z_c_by_ds, dtype=np.float64),
            "rho_delta_meanz_by_ds": np.asarray(delta_by_ds, dtype=np.float64),

            # shuffle FF/alpha (as before)
            "shuff_uncorr": cat_shuff_uncorr,
            "shuff_corr":   cat_shuff_corr,
            "shuff_alphas": cat_shuff_alphas,

            # shuffle null summaries for noise corr
            "shuff_rho_c_meanz": np.asarray(shuff_zc_meanz_all, dtype=np.float64),
            "shuff_rho_delta_meanz": np.asarray(shuff_delta_meanz_all, dtype=np.float64),

            # diagnostics
            "diag": diag,
        })

    return metrics




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