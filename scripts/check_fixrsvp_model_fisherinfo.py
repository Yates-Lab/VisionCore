"""
Compute spatial information from the model using reconstructed stimuli.
Allows counterfactual analysis with real vs fake eye traces.
"""
#%% Imports
import sys
sys.path.append('..')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from DataYatesV1 import enable_autoreload, get_free_device
from eval.eval_stack_multidataset import load_model, load_single_dataset, scan_checkpoints
from mcfarland_sim import get_fixrsvp_stack, eye_deg_to_norm, shift_movie_with_eye
from spatial_info import make_stimulus_stack, make_counterfactual_stim
from spatial_info import get_spatial_readout
from spatial_info import compute_rate_map, compute_rate_map_batched
from spatial_info import spatial_ssi_population, make_movie

enable_autoreload()
device = get_free_device()

#%% Load model and dataset
checkpoint_dir = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_120_long/checkpoints"
models_by_type = scan_checkpoints(checkpoint_dir, verbose=False)

model_type = 'resnet_none_convgru'
model, model_info = load_model(
    model_type=model_type,
    model_index=0,
    checkpoint_path=None,
    checkpoint_dir=checkpoint_dir,
    device='cpu'
)
model.model.eval()
model.model.convnet.use_checkpointing = True  # Enable checkpointing to save GPU memory
model = model.to(device)



import dill
with open('mcfarland_outputs.pkl', 'rb') as f:
    outputs = dill.load(f)

readout = get_spatial_readout(model, outputs).to(device)

sessions = [outputs[i]['sess'] for i in range(len(outputs))]
#%%
dataset_idx = 10
print(f"Loading dataset {dataset_idx}: {model.names[dataset_idx]}")
train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)

#%% Get fixrsvp trial indices
inds = torch.concatenate([
    train_data.get_dataset_inds('fixrsvp'),
    val_data.get_dataset_inds('fixrsvp')
], dim=0)

dataset = train_data.shallow_copy()
dataset.inds = inds

dset_idx = inds[:,0].unique().item()
trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
trials = np.unique(trial_inds)
NT = len(trials)

fixation = np.hypot(
    dataset.dsets[dset_idx]['eyepos'][:,0].numpy(), 
    dataset.dsets[dset_idx]['eyepos'][:,1].numpy()
) < 1


#%% Fisher information
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# @torch.compile 
# def differentiable_grid_sample(image, grid):
#     """
#     Faster, compiled implementation of grid_sample supporting Forward AD.
#     """
#     B, C, H, W = image.shape
#     _, H_out, W_out, _ = grid.shape
    
#     # 1. Map grid coordinates [-1, 1] to pixel coordinates [0, H-1]
#     x = grid[..., 0]
#     y = grid[..., 1]
    
#     x_pix = (x + 1) * W * 0.5 - 0.5
#     y_pix = (y + 1) * H * 0.5 - 0.5
    
#     # 2. Get corner pixel coordinates
#     x0 = torch.floor(x_pix).long()
#     x1 = x0 + 1
#     y0 = torch.floor(y_pix).long()
#     y1 = y0 + 1
    
#     # 3. Clamp coords (border mode = zeros logic handled by mask later)
#     x0_c = torch.clamp(x0, 0, W - 1)
#     x1_c = torch.clamp(x1, 0, W - 1)
#     y0_c = torch.clamp(y0, 0, H - 1)
#     y1_c = torch.clamp(y1, 0, H - 1)
    
#     # 4. Gather pixel values (Optimized Flattening)
#     # We flatten only spatial dims to keep channel broadcast efficient
#     image_flat = image.view(B, C, -1)
    
#     # Stride arithmetic to replace the helper function
#     # index = y * W + x
#     base_idx = torch.arange(B, device=image.device).view(B, 1, 1) * (H * W) # Batch offset (unnecessary if viewing B,C,-1 but good for global linear)
#     # Actually, simpler: just gather on dim 2 of the (B, C, HW) view
    
#     def gather_flat(y_c, x_c):
#         lin_idx = y_c * W + x_c
#         # Expand to (B, C, H_out, W_out)
#         lin_idx = lin_idx.unsqueeze(1).expand(-1, C, -1, -1)
#         # Flatten the spatial part for gather
#         # lin_idx_flat: (B, C, H_out*W_out)
#         return torch.gather(image_flat, 2, lin_idx.reshape(B, C, -1)).reshape(B, C, H_out, W_out)

#     Ia = gather_flat(y0_c, x0_c)
#     Ib = gather_flat(y1_c, x0_c)
#     Ic = gather_flat(y0_c, x1_c)
#     Id = gather_flat(y1_c, x1_c)
    
#     # 5. Interpolation Weights
#     wa = (x1 - x_pix) * (y1 - y_pix)
#     wb = (x1 - x_pix) * (y_pix - y0)
#     wc = (x_pix - x0) * (y1 - y_pix)
#     wd = (x_pix - x0) * (y_pix - y0)
    
#     wa = wa.unsqueeze(1)
#     wb = wb.unsqueeze(1)
#     wc = wc.unsqueeze(1)
#     wd = wd.unsqueeze(1)
    
#     # 6. Compute
#     out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    
#     # 7. Zero Padding Mask
#     mask = (x_pix >= 0) & (x_pix < W - 1) & (y_pix >= 0) & (y_pix < H - 1)
#     mask = mask.unsqueeze(1)
    
#     return out * mask.float()

def differentiable_grid_sample(image, grid):
    """
    A pure PyTorch implementation of grid_sample that supports Forward AD.
    Assumes align_corners=False and padding_mode='zeros' (zeros everywhere outside).
    
    Args:
        image: (B, C, H, W)
        grid:  (B, H_out, W_out, 2) in range [-1, 1]
    """
    B, C, H, W = image.shape
    _, H_out, W_out, _ = grid.shape
    
    # 1. Map grid coordinates [-1, 1] to pixel coordinates [0, H-1]
    # Formula for align_corners=False: x_pix = (x_norm + 1) * W / 2 - 0.5
    x = grid[..., 0]
    y = grid[..., 1]
    
    x_pix = (x + 1) * W * 0.5 - 0.5
    y_pix = (y + 1) * H * 0.5 - 0.5
    
    # 2. Get corner pixel coordinates
    x0 = torch.floor(x_pix).long()
    x1 = x0 + 1
    y0 = torch.floor(y_pix).long()
    y1 = y0 + 1
    
    # 3. Clamp coords to be inside image for gathering (we will mask zeros later)
    x0_clamped = torch.clamp(x0, 0, W - 1)
    x1_clamped = torch.clamp(x1, 0, W - 1)
    y0_clamped = torch.clamp(y0, 0, H - 1)
    y1_clamped = torch.clamp(y1, 0, H - 1)
    
    # 4. Gather pixel values
    # Flatten image to (B, C, H*W) to use gather efficiently
    image_flat = image.view(B, C, -1)
    
    # Helper to calculate linear indices
    def get_pixel_value(idx_x, idx_y):
        # Linear index: y * W + x
        # dimensions: (B, H_out, W_out)
        lin_idx = idx_y * W + idx_x
        # Expand for channels: (B, C, H_out, W_out)
        lin_idx_expanded = lin_idx.unsqueeze(1).expand(-1, C, -1, -1)
        # Flatten spatial for gather: (B, C, H_out*W_out)
        lin_idx_flat = lin_idx_expanded.reshape(B, C, -1)
        
        gathered = torch.gather(image_flat, 2, lin_idx_flat)
        return gathered.reshape(B, C, H_out, W_out)

    Ia = get_pixel_value(x0_clamped, y0_clamped) # Top-Left
    Ib = get_pixel_value(x0_clamped, y1_clamped) # Bottom-Left
    Ic = get_pixel_value(x1_clamped, y0_clamped) # Top-Right
    Id = get_pixel_value(x1_clamped, y1_clamped) # Bottom-Right
    
    # 5. Calculate interpolation weights
    # wa = (x1 - x) * (y1 - y)
    wa = (x1 - x_pix) * (y1 - y_pix)
    wb = (x1 - x_pix) * (y_pix - y0)
    wc = (x_pix - x0) * (y1 - y_pix)
    wd = (x_pix - x0) * (y_pix - y0)
    
    # Expand weights for channels
    wa = wa.unsqueeze(1)
    wb = wb.unsqueeze(1)
    wc = wc.unsqueeze(1)
    wd = wd.unsqueeze(1)
    
    # 6. Compute interpolated value
    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    
    # 7. Apply Zero Padding (mask out values that were outside boundaries)
    mask = (x_pix >= 0) & (x_pix < W - 1) & (y_pix >= 0) & (y_pix < H - 1)
    mask = mask.unsqueeze(1) # (B, 1, H_out, W_out)
    
    return out * mask.float()

import torchvision.transforms.functional as TF

class DifferentiableStimulus(nn.Module):
    """
    Stage 1: Generates a high-resolution static 'world' image.
    Parameterized by position, orientation, and size (LogMAR).
    """
    def __init__(self, 
                 stim_type='E', 
                 ppd=120, 
                 canvas_size=(256, 256), 
                 template_res=1024, 
                 blur_sigma=1.0,
                 device='cuda'):
        super().__init__()
        self.stim_type = stim_type
        self.ppd = ppd
        self.canvas_size = canvas_size
        self.device = device
        self.blur_sigma = blur_sigma
        self.blur_kernel = 7
        sz_h = canvas_size[0] / ppd
        sz_w = canvas_size[1] / ppd
        self.extent = [-sz_w/2, sz_w/2, -sz_h/2, sz_h/2]
        
        self.register_buffer('template', self._make_template(stim_type, template_res))
        
    def _make_template(self, type, res):
        xx = torch.linspace(-1, 1, res)
        yy = torch.linspace(-1, 1, res)
        y, x = torch.meshgrid(yy, xx, indexing='ij')
        k = 200.0 
        
        if type == 'E':
            def box(x0, x1, y0, y1):
                return (torch.sigmoid(k * (x - x0)) * torch.sigmoid(k * (x1 - x)) *
                        torch.sigmoid(k * (y - y0)) * torch.sigmoid(k * (y1 - y)))
            shape = (box(-1.0, -0.6, -1.0, 1.0) +  
                     box(-0.6, 1.0, 0.6, 1.0) +    
                     box(-0.6, 1.0, -0.2, 0.2) +   
                     box(-0.6, 1.0, -1.0, -0.6))   
            shape = torch.clamp(shape, 0, 1)
        else: 
            shape = torch.sigmoid(k * (1 - x.abs())) * torch.sigmoid(k * (0.2 - y.abs()))

        return shape.unsqueeze(0).unsqueeze(0).to(self.device)

    def get_affine_matrix(self, theta, logmar):
        B = theta.shape[0]
        x_deg, y_deg, ori_deg = theta[:, 0], theta[:, 1], theta[:, 2]
        H, W = self.canvas_size
        
        # World Normalize
        tx = x_deg * self.ppd / (W / 2.0)
        ty = -y_deg * self.ppd / (H / 2.0)
        T_vec = torch.stack([tx, ty], dim=1).unsqueeze(2)

        angle = ori_deg * (np.pi / 180.0)
        c, s = torch.cos(angle), torch.sin(angle)
        row1 = torch.stack([c, s], dim=1)
        row2 = torch.stack([-s, c], dim=1)
        R_inv = torch.stack([row1, row2], dim=1) 

        if isinstance(logmar, float): logmar = torch.full((B,), logmar, device=self.device)
        size_pix = (5 * (10**logmar / 60.0)) * self.ppd
        sx_inv = W / (size_pix + 1e-8)
        sy_inv = H / (size_pix + 1e-8)
        S_inv = torch.zeros_like(R_inv)
        S_inv[:, 0, 0] = sx_inv
        S_inv[:, 1, 1] = sy_inv

        A = torch.bmm(S_inv, R_inv)
        b = -torch.bmm(A, T_vec)
        return torch.cat([A, b], dim=2)

    def forward(self, theta, logmar=0.0):
        B = theta.shape[0]
        affine = self.get_affine_matrix(theta, logmar)
        
        # NOTE: align_corners=False is critical to match the manual sampler logic
        grid = F.affine_grid(affine, (B, 1, *self.canvas_size), align_corners=False)

        # Replaced F.grid_sample with differentiable_grid_sample
        world = differentiable_grid_sample(self.template.expand(B,-1,-1,-1), grid)
        if self.blur_sigma > 0:
            world = TF.gaussian_blur(world, kernel_size=self.blur_kernel, sigma=self.blur_sigma)
        return world

class DifferentiableRetina(nn.Module):
    """
    Optimized Stage 2: Samples the high-res world image along a trajectory.
    Uses (Space, Time) grid trick to avoid expanding the source image.
    """
    def __init__(self, ppd, world_canvas_size, retina_size=(32, 32)):
        super().__init__()
        self.ppd = ppd
        self.world_h, self.world_w = world_canvas_size
        self.retina_h, self.retina_w = retina_size
        self.n_pixels = self.retina_h * self.retina_w
        
        # 1. Pre-compute flattened Base Grid (centered at 0)
        xs_pix = torch.linspace(-self.retina_w/2 + 0.5, self.retina_w/2 - 0.5, self.retina_w)
        ys_pix = torch.linspace(-self.retina_h/2 + 0.5, self.retina_h/2 - 0.5, self.retina_h)
        
        # Scale to World Norm coords [-1, 1]
        xs_norm = xs_pix * (2.0 / self.world_w)
        ys_norm = ys_pix * (2.0 / self.world_h)
        
        grid_y, grid_x = torch.meshgrid(ys_norm, xs_norm, indexing='ij')
        
        # Flatten to (P, 2)
        # We use P = H*W as the "Height" dimension for grid_sample
        self.register_buffer('base_grid_flat', torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1))

    def forward(self, images, trajectories):
        """
        Args:
            images: [B, 1, H_world, W_world] (High res world)
            trajectories: [B, T, 2] (Eye positions in degrees)
            
        Returns:
            retinal_movie: [B, C, T, H_ret, W_ret]
        """
        B, T, _ = trajectories.shape
        P = self.n_pixels
        
        # 1. Convert Eye Position (deg) -> World Normalized Shift
        x_deg = trajectories[:, :, 0]
        y_deg = trajectories[:, :, 1]
        
        shift_x = x_deg * self.ppd * (2.0 / self.world_w)
        shift_y = -y_deg * self.ppd * (2.0 / self.world_h)
        shifts = torch.stack([shift_x, shift_y], dim=-1) # [B, T, 2]
        
        # 2. Construct Spatiotemporal Grid [B, P, T, 2]
        # P (pixels) acts as "Height", T (time) acts as "Width" for grid_sample
        
        # Base: [1, P, 1, 2]
        base = self.base_grid_flat.unsqueeze(0).unsqueeze(2)
        
        # Shifts: [B, 1, T, 2]
        shifts = shifts.unsqueeze(1)
        
        # Broadcast Sum: [B, P, T, 2]
        # Every pixel P gets shifted by the eye position at time T
        grid = base + shifts
        
        # 3. Sample
        # Replaced F.grid_sample
        output = differentiable_grid_sample(images, grid)
        
        # 4. Unflatten Space and Permute
        output = output.view(B, 1, self.retina_h, self.retina_w, T)
        output = output.permute(0, 1, 4, 2, 3) 
        
        return output

# ==========================================
# Run Optimization Demo
# ==========================================

device = model.device

# 1. Init
# Canvas size needs to be large enough to contain the E and the full eye trace
world_gen = DifferentiableStimulus(ppd=120, canvas_size=(512, 512), device=device)
retina = DifferentiableRetina(ppd=37.50476617, world_canvas_size=(512, 512), retina_size=(151, 151))
retina.to(device)

# 2. Params
theta = torch.tensor([[-0.0, 0.0, 30.0]], device=device, requires_grad=True) 

# Random Walk Trace
T_len = 50
rw = torch.cumsum(torch.randn(1, T_len, 2, device=device)*0.05, dim=1)
eye_trace = rw.clone().detach().requires_grad_(True)

# 3. Forward (Fast!)
# Stage 1: One World Image
high_res_world = world_gen(theta, logmar=0.6) 

# Stage 2: One Grid Sample call for the whole video
movie = retina(high_res_world, eye_trace)

# 4. Viz
plt.figure(figsize=(10, 4))

plt.subplot(131)
world_np = high_res_world[0,0].detach().cpu().numpy()
trace_np = eye_trace[0].detach().cpu().numpy()
plt.imshow(world_np, extent=world_gen.extent, cmap='gray', origin='lower')
plt.plot(trace_np[:,0], trace_np[:,1], 'r-', alpha=0.6)
plt.title("World + Eye Trace")

plt.subplot(132)
# Show mean retinal activation over time
mean_retina = movie[0,0].mean(dim=0).detach().cpu().numpy()
plt.imshow(mean_retina, cmap='gray', origin='lower')
plt.title("Average Retinal Input")

# 5. Optimize Eye Trace to Maximize Energy
loss = -torch.sum(movie**2)
loss.backward()

plt.subplot(133)
# Visualize gradient on the eye trace itself
grad_trace = eye_trace.grad[0].cpu().numpy()
# Magnitude of gradient per time point
grad_mag = np.linalg.norm(grad_trace, axis=1)
plt.plot(grad_mag)
plt.title("Gradient Magnitude on Eye Trace")
plt.xlabel("Time")

plt.tight_layout()
plt.show()

print(f"Movie Shape: {movie.shape} (B, C, T, H, W)")


#%%

import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD
import numpy as np
import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD
import gc

def optimize_trajectory_chunked_amp(
    model, 
    readout, 
    stim_gen, 
    retina, 
    initial_eye_trace, 
    base_theta, 
    param_idx_to_maximize=2,
    n_steps=100,
    n_lags=32,
    lr=1e-3,
    logmar=0.6,
    chunk_size=5,  # You can likely increase this now
    reg_lambda=1e-3,
    noise_model='poisson'
):
    
    # 0. Safety Cleanup
    torch.cuda.empty_cache()
    gc.collect()

    # 1. Setup
    eye_trace = initial_eye_trace.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([eye_trace], lr=lr)
    scaler = torch.amp.GradScaler(enabled=True)
    
    T = eye_trace.shape[1]
    total_windows = T - n_lags
    
    model.eval()
    readout.eval()
    
    loss_history = []
    
    print(f"Optimizing with Chunk Size {chunk_size} (AMP Enabled)...")

    for step in range(n_steps):
        optimizer.zero_grad()
        total_fisher = 0
        
        with fwAD.dual_level():
            tangent = torch.zeros_like(base_theta)
            tangent[:, param_idx_to_maximize] = 1.0
            dual_theta = fwAD.make_dual(base_theta, tangent)
            
            # Keep world generation in FP32 for precision, or cast if needed
            world = stim_gen(dual_theta, logmar=logmar)
            
            for start_i in range(0, total_windows, chunk_size):
                end_i = min(start_i + chunk_size, total_windows)
                slice_end = (end_i - 1) + n_lags
                
                # <--- NEW: Autocast Context
                # Everything inside this block runs in Float16 where safe
                with torch.cuda.amp.autocast():
                    trace_chunk = eye_trace[:, start_i:slice_end, :]
                    movie_chunk = retina(world, trace_chunk) 
                    
                    movie_squeezed = movie_chunk.squeeze()
                    input_windows = movie_squeezed.unfold(0, n_lags, 1) \
                                                  .permute(0, 3, 1, 2) \
                                                  .unsqueeze(1)
                    
                    rates = compute_rate_map(model, readout, input_windows)
                    
                    # Unpack Dual
                    rates_dual = fwAD.unpack_dual(rates)
                    rates_primal = rates_dual.primal
                    d_rates_d_theta = rates_dual.tangent
                    
                    if d_rates_d_theta is None: 
                        d_rates_d_theta = torch.zeros_like(rates_primal)

                    # Compute Fisher
                    epsilon = 1e-6
                    if noise_model == 'gaussian':
                        fisher_per_element = d_rates_d_theta ** 2
                    elif noise_model == 'poisson':
                        fisher_per_element = (d_rates_d_theta ** 2) / (rates_primal + epsilon)
                    
                    chunk_fisher = fisher_per_element.sum()
                    
                    # We want to MAXIMIZE fisher, so minimize negative fisher
                    loss = -chunk_fisher

                # <--- NEW: Scaled Backward
                # Scales loss to prevent underflow in FP16 gradients
                scaler.scale(loss).backward()
                
                total_fisher += chunk_fisher.item()
                
                del rates, rates_dual, rates_primal, d_rates_d_theta, fisher_per_element, movie_chunk, input_windows
        
        # Regularization (Run in standard precision or autocast, straightforward here)
        accel = torch.diff(eye_trace, n=2, dim=1)
        reg_loss = reg_lambda * (accel ** 2).sum()
        
        # Backward Reg
        scaler.scale(reg_loss).backward()
        
        # <--- NEW: Scaler Step
        scaler.step(optimizer)
        scaler.update()
        
        loss_history.append(total_fisher)
        if step % 10 == 0:
            print(f"Step {step:03d} | Total Fisher: {total_fisher:.4f}")

    return eye_trace.detach(), loss_history

# def optimize_trajectory_chunked(
#     model, 
#     readout, 
#     stim_gen, 
#     retina, 
#     initial_eye_trace, 
#     base_theta, 
#     param_idx_to_maximize=2,
#     n_steps=100,
#     n_lags=32,
#     lr=1e-3,
#     logmar=0.6,
#     chunk_size=1,  # Keep small for safety
#     reg_lambda=1e-3,
#     noise_model='poisson'
# ):
    
#     # 0. Safety Cleanup
#     torch.cuda.empty_cache()
#     gc.collect()

#     # 1. Setup
#     eye_trace = initial_eye_trace.clone().detach().requires_grad_(True)
#     optimizer = torch.optim.Adam([eye_trace], lr=lr)
    
#     T = eye_trace.shape[1]
#     total_windows = T - n_lags
    
#     model.eval()
#     readout.eval()
    
#     loss_history = []
    
#     print(f"Optimizing with Chunk Size {chunk_size} (Retina-Inside-Loop)...")

#     for step in range(n_steps):
#         optimizer.zero_grad()
#         total_fisher = 0
        
#         # ==========================================================
#         # Forward Mode AD
#         # ==========================================================
#         with fwAD.dual_level():
#             tangent = torch.zeros_like(base_theta)
#             tangent[:, param_idx_to_maximize] = 1.0
#             dual_theta = fwAD.make_dual(base_theta, tangent)
            
#             # 1. Generate World (Global)
#             # This is static and cheap (single image), so we keep it outside.
#             world = stim_gen(dual_theta, logmar=logmar)
            
#             # ==========================================================
#             # Chunked Processing
#             # ==========================================================
#             for start_i in range(0, total_windows, chunk_size):
#                 end_i = min(start_i + chunk_size, total_windows)
                
#                 # A. Calculate Required Trace Slice
#                 # We need enough trace to cover 'chunk_size' windows + 'n_lags' history
#                 # Window 0: [start_i, start_i + n_lags]
#                 # Window N: [end_i-1, end_i-1 + n_lags]
#                 # So we slice from start_i to (end_i - 1 + n_lags)
#                 slice_end = (end_i - 1) + n_lags
                
#                 # B. Slice Trace
#                 # eye_trace is (Batch, Time, Coords). We slice Time (dim 1).
#                 # This slice creates a graph connection to the main eye_trace.
#                 trace_chunk = eye_trace[:, start_i:slice_end, :]
                
#                 # C. Generate Movie Chunk (Inside Loop!)
#                 # This creates a fresh, small graph for just this chunk.
#                 movie_chunk = retina(world, trace_chunk) # (1, 1, T_chunk, H, W)
                
#                 # D. Unfold to Windows
#                 # movie_chunk squeeze: (T_chunk, H, W)
#                 # unfold -> (Batch_Windows, H, W, Lags)
#                 # permute -> (Batch_Windows, Lags, H, W)
#                 # unsqueeze -> (Batch_Windows, 1, Lags, H, W)
#                 movie_squeezed = movie_chunk.squeeze()
                
#                 # Note: The 'step' in unfold is 1 because we want sliding windows.
#                 # The 'size' is n_lags.
#                 input_windows = movie_squeezed.unfold(0, n_lags, 1) \
#                                               .permute(0, 3, 1, 2) \
#                                               .unsqueeze(1)
                
#                 # E. Compute Rates
#                 rates = compute_rate_map(model, readout, input_windows)
                
#                 # F. Unpack Dual
#                 rates_dual = fwAD.unpack_dual(rates)
#                 rates_primal = rates_dual.primal
#                 d_rates_d_theta = rates_dual.tangent
                
#                 # Handle broken connections (rare but safe)
#                 if d_rates_d_theta is None:
#                     d_rates_d_theta = torch.zeros_like(rates_primal) 
                
#                 # G. Compute Fisher
#                 epsilon = 1e-6
#                 if noise_model == 'gaussian':
#                     fisher_per_element = d_rates_d_theta ** 2
#                 elif noise_model == 'poisson':
#                     fisher_per_element = (d_rates_d_theta ** 2) / (rates_primal + epsilon)
                
#                 chunk_fisher = fisher_per_element.sum()
                
#                 # H. Backward
#                 # CRITICAL: We do NOT use retain_graph=True here.
#                 # Because we re-ran 'retina', this chunk's graph is isolated.
#                 # Calling backward() calculates grads for eye_trace and DESTROYS the heavy ConvNet graph.
#                 (-chunk_fisher).backward()
                
#                 total_fisher += chunk_fisher.item()
                
#                 # Explicit cleanup
#                 del rates, rates_dual, rates_primal, d_rates_d_theta, fisher_per_element, movie_chunk, input_windows
        
#         # ==========================================================
#         # Regularization & Step
#         # ==========================================================
#         accel = torch.diff(eye_trace, n=2, dim=1)
#         reg_loss = reg_lambda * (accel ** 2).sum()
#         reg_loss.backward()
        
#         optimizer.step()
        
#         loss_history.append(total_fisher)
#         if step % 10 == 0:
#             print(f"Step {step:03d} | Total Fisher: {total_fisher:.4f}")

#     return eye_trace.detach(), loss_history

theta = torch.tensor([[0.0, 0.0, 0.0]], device=model.device) # Batch size 1

eye_trace = torch.zeros(1, 33, 2, device=model.device)
# eye_trace = torch.cumsum(torch.randn(1, 50, 2, device=model.device)*0.05, dim=1)


optimize_trajectory_chunked_amp(
    model=model,
    readout=readout,
    stim_gen=world_gen,
    retina=retina,
    initial_eye_trace=eye_trace, # From your snippet
    base_theta=theta
)

#%%

def optimize_short_trajectory(
    model, 
    readout, 
    stim_gen, 
    retina, 
    base_theta,
    initial_eye_trace=None, # Shape (1, 32, 2)
    param_idx_to_maximize=2, # 2 = Orientation, 0 = X, 1 = Y
    n_steps=100,
    lr=1e-3,
    logmar=0.6,
    reg_lambda=1e-3,
    n_lags = 32,
    noise_model='gaussian'
):
    """
    Optimizes a single short trajectory (e.g., 32 frames) to maximize 
    the Fisher Information of the resulting rate vector.
    """
     # 0. Safety Cleanup
    torch.cuda.empty_cache()
    gc.collect()

    
    # 1. Setup Snippet
    if initial_eye_trace is None:
        # Start with small random jitter around 0
        snippet = torch.randn(1, n_lags, 2, device=base_theta.device) * 0.01
    else:
        snippet = initial_eye_trace.clone().detach()
        
    snippet.requires_grad_(True)
    
    # Use standard Adam
    optimizer = torch.optim.Adam([snippet], lr=lr, weight_decay=0.001)
    
    model.eval()
    readout.eval()
    
    loss_history = []
    
    print(f"Optimizing single {n_lags}-frame snippet...")

    for step in range(n_steps):
        optimizer.zero_grad()
        
        # ==========================================================
        # Forward Mode AD (Single Pass)
        # ==========================================================
        with fwAD.dual_level():
            # 1. Dual Theta
            tangent = torch.zeros_like(base_theta)
            tangent[:, param_idx_to_maximize] = 1.0
            dual_theta = fwAD.make_dual(base_theta, tangent)
            
            # 2. Generate World (Dual)
            world = stim_gen(dual_theta, logmar=logmar)
            
            # 3. Retina (Single Call, No Loops)
            # Input: world (Dual), snippet (Primal)
            # Output: (1, 1, 32, H, W)
            movie_snippet = retina(world, snippet)
            
            # 4. Model (Single Call)
            # We don't need unfolding because we ARE the window.
            # Just ensure dims are (Batch, Channel, Time, H, W)
            # Retina output is usually (B, C, T, H, W) already.
            rates = compute_rate_map(model, readout, movie_snippet)
            
            # 5. Unpack Duals
            rates_dual = fwAD.unpack_dual(rates)
            rates_primal = rates_dual.primal
            d_rates_d_theta = rates_dual.tangent

            # 6. Fisher Info
            epsilon = 1e-6
            if noise_model == 'gaussian':
                fisher = (d_rates_d_theta ** 2).sum()
            elif noise_model == 'poisson':
                fisher = ((d_rates_d_theta ** 2) / (rates_primal + epsilon)).sum()
            
            # 7. Backward
            loss = -fisher
            loss.backward()
            
            # Clean up immediately
            del world, movie_snippet, rates, rates_dual
        
        # ==========================================================
        # Regularization (Smoothness + Centering)
        # ==========================================================
        # 1. Smoothness (minimize acceleration)
        accel = torch.diff(snippet, n=2, dim=1)
        reg_smooth = reg_lambda * (accel ** 2).sum()
        
        reg_smooth.backward()
        
        optimizer.step()
        loss_history.append(fisher.item())
        
        if step % 20 == 0:
            print(f"Step {step:03d} | Fisher: {fisher.item():.4f}")

    return snippet.detach(), loss_history


#%%

eye_trace = torch.randn(1, 32, 2, device=model.device) * 0.01

optimized_trace, history = optimize_short_trajectory(
    model=model,
    readout=readout,
    stim_gen=world_gen,
    retina=retina,
    n_steps=500,
    param_idx_to_maximize=0,
    initial_eye_trace=eye_trace, # From your snippet
    noise_model='poisson',
    base_theta=theta
)

# %%

plt.subplot(1,2,1)
plt.plot(eye_trace[0].detach().cpu().numpy()[:,0], 'b-', label='Initial X')
plt.plot(optimized_trace[0].detach().cpu().numpy()[:,0], 'b--', label='Optimized X')
plt.plot(eye_trace[0].detach().cpu().numpy()[:,1], 'r-', label='Initial Y')
plt.plot(optimized_trace[0].detach().cpu().numpy()[:,1], 'r--', label='Optimized Y')

plt.legend()
plt.xlabel('Time')
plt.ylabel('Eye Position (deg)')

# plot fourier power of the initial and optimized traces



#%%
plt.plot(history)
plt.ylabel('Fisher Information')
plt.xlabel('Optimization Step')


# %%

def visualize_results(stim_gen, retina, theta, initial_trace, optimized_trace, fisher_history):
    # 1. Generate the High-Res World
    # We detach everything to move to numpy
    with torch.no_grad():
        world = stim_gen(theta, logmar=0.6)
        world_np = world[0, 0].cpu().numpy()
    
    # 2. Process Traces
    # Convert trace from degrees to pixels for plotting over the image
    # We use the same logic as the DifferentiableStimulus to map deg -> pixels
    def trace_to_pixels(trace, canvas_size, ppd):
        trace_np = trace[0].detach().cpu().numpy()
        H, W = canvas_size
        
        # Invert the normalization done in get_affine_matrix
        # trace x (deg) * ppd = pixels from center
        x_pix = trace_np[:, 0] * ppd + (W / 2.0)
        y_pix = -trace_np[:, 1] * ppd + (H / 2.0) # Note the negative for y-flip
        return x_pix, y_pix

    x_init, y_init = trace_to_pixels(initial_trace, stim_gen.canvas_size, stim_gen.ppd)
    x_opt, y_opt = trace_to_pixels(optimized_trace, stim_gen.canvas_size, stim_gen.ppd)

    # 3. Plotting
    plt.figure(figsize=(12, 5))

    # Panel A: The Optimization Landscape
    plt.subplot(1, 2, 1)
    plt.plot(fisher_history, 'k-', lw=1.5)
    plt.title("Fisher Information Optimization")
    plt.xlabel("Step")
    plt.ylabel("Fisher Info (Dr/Dtheta)^2")
    plt.grid(True, alpha=0.3)

    # Panel B: The Trajectory
    plt.subplot(1, 2, 2)
    # Plot the world
    plt.imshow(world_np, cmap='gray', origin='upper', 
               extent=[0, stim_gen.canvas_size[1], stim_gen.canvas_size[0], 0])
    
    # Plot Initial Trace (faint red)
    plt.plot(x_init, y_init, 'r-', alpha=0.3, label='Initial (Random Walk)')
    plt.plot(x_init[0], y_init[0], 'ro', alpha=0.3) # Start point
    
    # Plot Optimized Trace (Green/Blue)
    # We use a scatter to show velocity (points closer together = slower speed)
    plt.plot(x_opt, y_opt, 'c-', lw=2, label='Optimized (Max Info)')
    # plt.scatter(x_opt, y_opt, c=np.arange(len(x_opt)), cmap='viridis', s=20, zorder=3)
    
    plt.legend(loc='upper right')
    plt.title("Eye Trajectory on Stimulus")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Run it
visualize_results(
    stim_gen=world_gen, 
    retina=retina, 
    theta=theta, 
    initial_trace=eye_trace, 
    optimized_trace=optimized_trace, 
    fisher_history=history
)


# %%
def visualize_rate_maps(model, readout, stim_gen, retina, theta, initial_trace, optimized_trace):
    
    # 1. Compute Rates for both traces
    model.eval()
    readout.eval()
    
    with torch.no_grad():
        # Generate Inputs
        world = stim_gen(theta, logmar=0.6)
        
        # A. Initial
        movie_init = retina(world, initial_trace)
        rates_init = compute_rate_map(model, readout, movie_init) # Shape (1, N_units)
        
        # B. Optimized
        movie_opt = retina(world, optimized_trace)
        rates_opt = compute_rate_map(model, readout, movie_opt)   # Shape (1, N_units)
        
        # Move to CPU
        r_init = rates_init[0].cpu()
        r_opt = rates_opt[0].cpu()

    # return r_init, r_opt
    # --- Plotting ---
    plt.figure(figsize=(14, 6))
    
    # Panel 1: Scatter Comparison
    plt.figure()
    plt.scatter(r_init, r_opt, alpha=0.15, s=5, c='k')
    plt.plot([0, r_opt.max()], [0, r_opt.max()], 'r--', alpha=0.5) # Identity line
    plt.title("Firing Rate Comparison")
    plt.xlabel("Initial Trace Rates")
    plt.ylabel("Optimized Trace Rates")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    plt.figure(figsize=(40,20))
    # Panel 2: Spatial Map (Initial)
    # make grid of 100 units 10x10 using torchvision make_grid
    from torchvision.utils import make_grid
    grid_init = make_grid(r_init[:25].unsqueeze(1), nrow=5, normalize=True, scale_each=False, padding=2)
    grid_opt = make_grid(r_opt[:25].unsqueeze(1), nrow=5, normalize=True, scale_each=False, padding=2)


    plt.subplot(1, 2, 1)
    plt.imshow(grid_init.permute(1, 2, 0).squeeze(), cmap="inferno", interpolation="none")
    plt.axis("off")
    
    # plt.imshow(r_init.mean(0), cmap='inferno', origin='lower')
    plt.title("Population Activity (Initial)")
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)

    # Panel 3: Spatial Map (Optimized)
    plt.subplot(1, 2, 2)
    # plt.imshow(r_opt.mean(0), cmap='inferno', origin='lower')
    plt.imshow(grid_opt.permute(1, 2, 0).squeeze(), cmap="inferno", interpolation="none")
    plt.title("Population Activity (Optimized)")
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

# Run it
visualize_rate_maps(model, readout, world_gen, retina, theta, eye_trace, optimized_trace)

#%%
import torch.autograd.forward_ad as fwAD
from torchvision.utils import make_grid

def visualize_fisher_spatial(model, readout, stim_gen, retina, theta, optimized_trace, noise_model='poisson'):
    """
    Visualizes where the model is looking (Rates) vs. where it is gathering Information (Fisher).
    """
    model.eval()
    readout.eval()
    
    # We need gradients, so we CANNOT use torch.no_grad()
    # We use Forward AD instead.
    with fwAD.dual_level():
        # 1. Create Dual Input (Theta + Tangent)
        tangent = torch.zeros_like(theta)
        # Assuming we want Fisher w.r.t the Orientation parameter (index 2)
        tangent[:, 2] = 1.0 
        dual_theta = fwAD.make_dual(theta, tangent)
        
        # 2. Generate World & Movie (Propagate Duals)
        # Note: Use same logmar/blur settings as your optimization
        world = stim_gen(dual_theta, logmar=0.6) 
        movie = retina(world, optimized_trace)
        
        # 3. Compute Rates (Dual)
        # Assumes snippet/trace is already correct length for the model
        rates = compute_rate_map(model, readout, movie)
        
        # 4. Unpack
        rates_dual = fwAD.unpack_dual(rates)
        r_primal = rates_dual.primal.detach() # The Firing Rates
        r_tangent = rates_dual.tangent.detach() # The Gradient (dr/dtheta)
        
        if r_tangent is None:
            r_tangent = torch.zeros_like(r_primal)

    # 5. Compute Fisher Information Map
    # Poisson: I = (f')^2 / f
    # Gaussian: I = (f')^2
    epsilon = 1e-6
    if noise_model == 'poisson':
        fisher_map = (r_tangent ** 2) / (r_primal + epsilon)
    else:
        fisher_map = (r_tangent ** 2)

    # --- Plotting ---
    # Assuming r_primal shape is (Batch, Channels, H, W) or (Channels, H, W)
    # We take the first batch item if needed
    if r_primal.ndim == 4:
        r_primal = r_primal[0]
        fisher_map = fisher_map[0]
    
    # If r is flattened (Neurons,), we can't visualize spatial maps easily 
    # unless we reshape. Assuming your readout is spatial based on your make_grid usage.
    
    # Move to CPU
    r_primal = r_primal.cpu()
    fisher_map = fisher_map.cpu()
    
    # 1. Select top channels to visualize (e.g., top 25 most active)
    # Or just the first 25
    top_k = 25
    
    # Sort by total activity to show the most interesting units
    total_activity = r_primal.view(r_primal.shape[0], -1).sum(1)
    _, idx = torch.sort(total_activity, descending=True)
    idx = idx[:top_k]
    
    r_viz = r_primal[idx].unsqueeze(1) # (25, 1, H, W)
    f_viz = fisher_map[idx].unsqueeze(1) # (25, 1, H, W)
    
    # Make Grids
    # normalize=True scales each map individually to [0,1] so we can see the pattern
    grid_rates = make_grid(r_viz, nrow=5, normalize=True, padding=2)
    grid_fisher = make_grid(f_viz, nrow=5, normalize=True, padding=2)

    plt.figure(figsize=(20, 10))
    
    # Plot Activity
    plt.subplot(1, 2, 1)
    plt.imshow(grid_rates.permute(1, 2, 0), cmap='inferno')
    plt.title(f"Firing Rates (Top {top_k} Active Units)\n'What the eye sees'")
    plt.axis('off')
    
    # Plot Information
    plt.subplot(1, 2, 2)
    plt.imshow(grid_fisher.permute(1, 2, 0), cmap='inferno')
    plt.title(f"Fisher Information Density (dr/dtheta)^2\n'Where the info is'")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return r_primal, fisher_map

# Run it
r_map, i_map_opt = visualize_fisher_spatial(model, readout, world_gen, retina, theta, optimized_trace )
r_map, i_map_init = visualize_fisher_spatial(model, readout, world_gen, retina, theta, eye_trace)


plt.figure()
plt.subplot(1,2,1)
vmin = max(i_map_init.mean(0).amax(), i_map_opt.mean(0).amax())
plt.imshow(i_map_init.mean(0), cmap='inferno', vmin=0, vmax=vmin)
plt.title('Initial')
plt.subplot(1,2,2)
plt.imshow(i_map_opt.mean(0), cmap='inferno', vmin=0, vmax=vmin)
plt.title('Optimized')

# %%

torch.cuda.empty_cache()
#%%
import torch.autograd.forward_ad as fwAD

def optimize_batch_trajectories(
    model, 
    readout, 
    stim_gen, 
    retina, 
    base_theta, 
    batch_size=32, 
    chunk_size=4,        # <--- Process only 4 walkers at a time to save memory
    n_steps=150, 
    lr=1e-2,
    param_idx=2
):
    """
    Runs batch_size independent trajectory optimizations using Gradient Accumulation.
    """
    
    # 1. Initialize Batch of Random Walks (Same as before)
    start_pos = (torch.rand(batch_size, 1, 2, device=base_theta.device) - 0.5) * 0.6
    T_len = 32
    walks = torch.cumsum(torch.randn(batch_size, T_len, 2, device=base_theta.device) * 0.05, dim=1)
    eye_traces = (start_pos + walks).detach().requires_grad_(True)
    
    optimizer = torch.optim.Adam([eye_traces], lr=lr)
    model.eval()
    readout.eval()
    
    best_history = []
    
    print(f"Launching {batch_size} optimizers (Chunk size {chunk_size})...")
    
    for step in range(n_steps):
        optimizer.zero_grad()
        
        step_best_val = -float('inf')
        
        # --- LOOP OVER CHUNKS ---
        # We process walkers [0:4], then [4:8], etc.
        for i in range(0, batch_size, chunk_size):
            
            # Slice the current chunk of trajectories
            # Gradients computed on 'trace_chunk' will flow back to 'eye_traces'
            trace_chunk = eye_traces[i : i + chunk_size]
            current_batch_size = trace_chunk.shape[0]
            
            # Forward AD Context (Fresh for each chunk to free memory after)
            with fwAD.dual_level():
                tangent = torch.zeros_like(base_theta)
                tangent[:, param_idx] = 1.0
                dual_theta = fwAD.make_dual(base_theta, tangent)
                
                # Generate World (1, 1, H, W) -> Expand to Chunk Size
                world_dual = stim_gen(dual_theta, logmar=0.6) 
                world_chunk = world_dual.expand(current_batch_size, -1, -1, -1)
                
                # Retina & Model (Process only chunk_size items)
                movie = retina(world_chunk, trace_chunk)
                rates = compute_rate_map(model, readout, movie)
                
                # Jacobian
                d_rates = fwAD.unpack_dual(rates).tangent
            
            # Compute Loss for this chunk
            # Fisher Info: Sum squares over (Time, Units, H, W)
            fisher_info_chunk = (d_rates ** 2).sum(dim=(1, 2, 3))
            
            # Accumulate Gradients
            # We sum losses so that gradients add up in eye_traces.grad
            loss = -fisher_info_chunk.sum()
            loss.backward() 
            
            # Track stats
            chunk_max = fisher_info_chunk.max().item()
            if chunk_max > step_best_val:
                step_best_val = chunk_max

        # --- UPDATE ---
        # After processing all chunks, eye_traces.grad contains gradients for all 32 walkers
        optimizer.step()
        
        best_history.append(step_best_val)
        if step % 20 == 0:
            print(f"Step {step:03d} | Best Fisher Info: {step_best_val:.4f}")

    # --- Select Winner ---
    # We need one final pass (chunked) to get the final scores without grad
    final_scores = []
    with torch.no_grad():
        for i in range(0, batch_size, chunk_size):
            trace_chunk = eye_traces[i : i + chunk_size]
            current_batch_size = trace_chunk.shape[0]
            
            # We can't use ForwardAD in no_grad mode easily for evaluation,
            # but we just need to know who won. 
            # Ideally, we track the winner index during the last optimization loop.
            # For simplicity here, we assume the last step's scores are close enough.
            pass

    # Note: To be perfectly accurate, we should have stored the indices in the loop.
    # But returning the trace at the index of the max *gradient magnitude* or just re-running
    # the Forward AD one last time is fine.
    # Let's just return the whole batch and let the visualizer pick the best?
    # Or simplified: We just return the last calculated best from the loop.
    
    # Quick fix to get exact winner index: Re-run Forward AD one last time (chunked)
    print("Selecting final winner...")
    all_scores = []
    for i in range(0, batch_size, chunk_size):
        with fwAD.dual_level():
            tangent = torch.zeros_like(base_theta); tangent[:, param_idx] = 1.0
            dual_theta = fwAD.make_dual(base_theta, tangent)
            world_chunk = stim_gen(dual_theta, logmar=0.6).expand(eye_traces[i:i+chunk_size].shape[0],-1,-1,-1)
            rates = compute_rate_map(model, readout, retina(world_chunk, eye_traces[i:i+chunk_size]))
            d_rates = fwAD.unpack_dual(rates).tangent
            scores = (d_rates ** 2).sum(dim=(1,2,3))
            all_scores.append(scores.detach())
    
    all_scores = torch.cat(all_scores)
    final_idx = torch.argmax(all_scores)
    winner_trace = eye_traces[final_idx].detach()
    
    print(f"Winner: Walker {final_idx} | Info: {all_scores[final_idx].item():.4f}")
    return winner_trace, eye_traces.detach(), best_history

def visualize_batch_results(stim_gen, theta, winner_trace, all_traces, history):
    # 1. Generate Static World for Background
    with torch.no_grad():
        world = stim_gen(theta, logmar=0.6)
        world_np = world[0, 0].cpu().numpy()

    # 2. Helper: Convert Trace to Pixels
    def to_pix(trace):
        # trace shape (T, 2)
        trace = trace.cpu().numpy()
        H, W = stim_gen.canvas_size
        x = trace[:, 0] * stim_gen.ppd + (W / 2.0)
        y = -trace[:, 1] * stim_gen.ppd + (H / 2.0)
        return x, y

    # 3. Plot
    plt.figure(figsize=(12, 5))

    # Panel A: Optimization History
    plt.subplot(1, 2, 1)
    plt.plot(history, 'k-', lw=2)
    plt.title(f"Optimization (Best of {len(all_traces)})")
    plt.xlabel("Step")
    plt.ylabel("Max Fisher Info")
    plt.grid(True, alpha=0.3)

    # Panel B: Trajectory Cloud
    plt.subplot(1, 2, 2)
    plt.imshow(world_np, cmap='gray', origin='upper', 
               extent=[0, stim_gen.canvas_size[1], stim_gen.canvas_size[0], 0])
    
    # A. Draw Losers (Faintly)
    for i in range(len(all_traces)):
        lx, ly = to_pix(all_traces[i])
        plt.plot(lx, ly, 'r-', alpha=0.1, lw=0.5)
        
    # B. Draw Winner (Bold Cyan)
    wx, wy = to_pix(winner_trace)
    plt.plot(wx, wy, 'c-', lw=2.5, label='Winning Strategy')
    plt.scatter(wx, wy, c=np.arange(len(wx)), cmap='cool', s=20, zorder=5)

    plt.legend(loc='upper right')
    plt.title("Batch Search Results")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Assuming model, readout, world_gen, retina are already defined and on device

# 1. Run Batch Optimization
winner, all_traces, history = optimize_batch_trajectories(
    model=model,
    readout=readout,
    stim_gen=world_gen,
    retina=retina,
    base_theta=theta, # (1, 3) tensor
    batch_size=32,    # Increased search space
    n_steps=150       # Fewer steps needed since we search parallel
)

# 2. Visualize
visualize_batch_results(world_gen, theta, winner, all_traces, history)
# %%

plt.plot(winner.detach().cpu().numpy())
# %%
