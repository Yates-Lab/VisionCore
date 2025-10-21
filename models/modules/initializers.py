# minimal, self-contained initializers for Conv3d
import math, torch
from torch.nn.utils import parametrize

# ---- generic helpers ----
def _param_list(m, name="weight"):
    return getattr(m.parametrizations, name) if (
        hasattr(m, "parametrizations") and hasattr(m.parametrizations, name)
    ) else None

def base_weight(m, name="weight"):
    P = _param_list(m, name)
    return P.original if P is not None else getattr(m, name)

def find_window_param(m, name="weight"):
    P = _param_list(m, name)
    if P is None: return None
    for p in P:
        if hasattr(p, "mask"): return p
    return None

def refresh_weightnorm_if_any(m, name="weight"):
    P = _param_list(m, name)
    if P is None: return
    bw = base_weight(m, name)
    for p in P:
        if hasattr(p, "v") and hasattr(p, "g") and hasattr(p, "dim"):
            p.v.copy_(bw)
            red = tuple(d for d in range(bw.ndim) if d != p.dim)
            if getattr(p, "keep_unit_norm", False): p.g.fill_(1.0)
            else: p.g.copy_(torch.linalg.vector_norm(p.v, dim=red))
            break

@torch.no_grad()
def init_like_no_param(m, init_fn, name="weight"):
    """Init *base* param as if no parametrizations existed; resync WN if present."""
    bw = base_weight(m, name)
    tmp = torch.empty_like(bw); init_fn(tmp); bw.copy_(tmp)
    refresh_weightnorm_if_any(m, name)

# ---- steerable Gaussian derivative bank (2D) ----
def _meshgrid(k, device, dtype):
    ax = torch.arange(k, device=device, dtype=dtype) - (k-1)/2
    return torch.meshgrid(ax, ax, indexing='ij')

def _gaussian(k, sigma, device, dtype):
    X, Y = _meshgrid(k, device, dtype)
    g = torch.exp(-(X**2 + Y**2) / (2*sigma**2))
    g /= g.sum().clamp_min(1e-12)
    return g

def _gauss_derivs(k, sigma, device, dtype):
    X, Y = _meshgrid(k, device, dtype); g = _gaussian(k, sigma, device, dtype)
    Gx  = -(X/(sigma**2)) * g
    Gy  = -(Y/(sigma**2)) * g
    Gxx = ((X**2 - sigma**2) / (sigma**4)) * g
    Gyy = ((Y**2 - sigma**2) / (sigma**4)) * g
    Gxy = (X*Y / (sigma**4)) * g
    return Gx, Gy, Gxx, Gxy, Gyy

def _l2norm(K): K = K - K.mean(); return K / (K.norm() + 1e-8)

@torch.no_grad()
def gaussian_steerable_kernels_2d(
    kernel_size=15, sigmas=(1.6,2.8,5.0),
    n_orient=8, orders=(1,2), normalize=True,
    device=None, dtype=None
):
    device = device or torch.device("cpu")
    dtype  = dtype or torch.float32
    k = kernel_size
    bank = []
    for s in sigmas:
        Gx, Gy, Gxx, Gxy, Gyy = _gauss_derivs(k, s, device, dtype)

        # --- order 0 (isotropic) ---
        if 0 in orders:
            K0 = _l2norm(_gaussian(k, s, device, dtype)) if normalize else _gaussian(k, s, device, dtype)
            bank.append(K0)

        # --- order 1 (oriented) ---
        if 1 in orders:
            for m in range(n_orient):
                th = 2*math.pi*m/n_orient
                # basis: [Gx, Gy]
                Kc =  math.cos(th)*Gx + math.sin(th)*Gy             # "cos" phase
                bank.append(_l2norm(Kc) if normalize else Kc)

        # --- order 2 (oriented) ---
        if 2 in orders:
            # Steer in Cartesian basis: G_θθ = cos²θ·Gxx + 2·sinθ·cosθ·Gxy + sin²θ·Gyy
            for m in range(n_orient):
                th = math.pi*m/n_orient  # orientation angle (0 to π for 2nd order)
                c, s = math.cos(th), math.sin(th)
                Kc = (c*c)*Gxx + 2*c*s*Gxy + (s*s)*Gyy
                bank.append(_l2norm(Kc) if normalize else Kc)

    return torch.stack(bank, 0).to(device=device, dtype=dtype)  # [N,k,k]

# ---- write a 2D bank into a Conv3d (handles parametrizations) ----
@torch.no_grad()
def init_conv3d_with_2d_kernels(conv3d: torch.nn.Conv3d,
                                kernels2d: torch.Tensor,        # [N,k,k]
                                temporal: str = "repeat"):
    device, dtype = conv3d.weight.device, conv3d.weight.dtype
    kt, (kh, kw) = conv3d.kernel_size[0], kernels2d.shape[-2:]
    cout, cin, groups = conv3d.out_channels, conv3d.in_channels, conv3d.groups
    cin_per_group = cin // groups
    N = kernels2d.shape[0]
    if cout % N != 0: raise ValueError(f"out_channels {cout} must be multiple of N={N}")

    # time profile
    if temporal == "delta":
        tt = torch.zeros(kt, device=device, dtype=dtype); tt[kt//2] = 1.0
    elif temporal == "gauss":
        t = torch.arange(kt, device=device, dtype=dtype) - (kt-1)/2
        sigma_t = max(1.0, kt/6.0); tt = torch.exp(-0.5*(t/sigma_t)**2); tt /= tt.sum().clamp_min(1e-12)
    elif temporal == "repeat":
        tt = torch.ones(kt, device=device, dtype=dtype) / kt
    else:
        raise ValueError("temporal must be 'delta', 'gauss', or 'repeat'")

    K3 = torch.einsum('t,nkh->ntkh', tt, kernels2d.to(device=device, dtype=dtype))  # [N,kt,kh,kw]
    stack = K3.unsqueeze(1).repeat(1, cin_per_group, 1, 1, 1) / (cin_per_group**0.5)
    base  = stack.repeat(cout // N, 1, 1, 1, 1)  # [Cout, Cin/groups, kt, kh, kw]

    # optional: precomp for AA mask (normalize after AA): not enabled by default
    # wp = find_window_param(conv3d, "weight");
    # if wp is not None: base = base / wp.mask.to(device=device, dtype=dtype).clamp_min(1e-4)

    bw = base_weight(conv3d, "weight")
    bw.copy_(base)
    refresh_weightnorm_if_any(conv3d, "weight")
