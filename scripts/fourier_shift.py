#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import matplotlib as mpl

# embed TrueType fonts in PDF/PS
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

# (optional) pick a clean sans‐serif
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# ----------------------------
# Controls
# ----------------------------
N        = 128
k1, k2   = 8, 24                 # DFT-bin-locked sinusoids
amps     = (1.0, 0.5)
snap_ts  = [-1.0, 0.0, +1.0]     # three snapshots
path_ts  = np.linspace(-1.0, +1.0, 101)

# ----------------------------
# Signal model: sum of two exact sinusoids, shifted by t samples
# ----------------------------
n = np.arange(N, dtype=float)
w1, w2 = 2*np.pi*k1/N, 2*np.pi*k2/N
a1, a2 = amps

# keep these next to your controls
snap_ts = [-1.0, 0.0, +1.0]
snap_colors = { -1.0: 'C0',  0.0: 'C1',  +1.0: 'C2' }  # consistent mapping


def signal_at_shift(t):
    return a1*np.sin(w1*(n - t)) + a2*np.sin(w2*(n - t))

def two_coeffs(x):
    X = np.fft.fft(x)
    return X[k1], X[k2]

# Build trajectories across shifts
X1_path, X2_path = [], []
for t in path_ts:
    X1, X2 = two_coeffs(signal_at_shift(t))
    X1_path.append(X1)
    X2_path.append(X2)
X1_path, X2_path = np.asarray(X1_path), np.asarray(X2_path)

# Snapshots
snap_data = []
for t in snap_ts:
    x = signal_at_shift(t)
    X1s, X2s = two_coeffs(x)
    snap_data.append((t, x, X1s, X2s))

# Convenience
f1, f2 = k1/N, k2/N
theta1 = np.unwrap(np.angle(X1_path))
theta2 = np.unwrap(np.angle(X2_path))
amp1   = np.abs(X1_path)  # constant over t

# ----------------------------
# Plot: 1 x 2 layout
# ----------------------------
fig = plt.figure(figsize=(8, 4.2))

# (a) Translating signal (zoomed view)
ax1 = fig.add_subplot(1, 2, 1)
for t, x, *_ in snap_data:
    ax1.plot(n, x, color=snap_colors[t], label=f't = {t:+g}')
ax1.set_xlim(45, 65)                    # << zoom as requested
ax1.set_title('(a) translating signal (zoomed)')
ax1.set_xlabel('space / sample (n)')
ax1.set_ylabel('amplitude')
ax1.legend(loc='upper right', fontsize=8, frameon=False)

# (b) Complex Fourier coefficients with full circles + axes
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
# trajectories
ax2.plot([f1]*len(path_ts), np.real(X1_path), np.imag(X1_path), lw=2)
ax2.plot([f2]*len(path_ts), np.real(X2_path), np.imag(X2_path), lw=2)

# snapshot points
colors = snap_colors.values() #colors = ['C1','C0','C2']
for c, (t, _, X1s, X2s) in zip(colors, snap_data):
    ax2.scatter([f1], [np.real(X1s)], [np.imag(X1s)], s=50, color=c)
    ax2.scatter([f2], [np.real(X2s)], [np.imag(X2s)], s=50, color=c)
    # plot line from the origin to the point
    ax2.plot([f1, f1], [0, np.real(X1s)], [0, np.imag(X1s)], color=c, linestyle='--', linewidth=1)
    ax2.plot([f2, f2], [0, np.real(X2s)], [0, np.imag(X2s)], color=c, linestyle='--', linewidth=1)

# full gray circles showing |X_k| at each frequency
def add_circle(ax, f_const, radius, npts=200):
    ang = np.linspace(0, 2*np.pi, npts)
    ax.plot([f_const]*npts, radius*np.cos(ang), radius*np.sin(ang),
            color='0.7', lw=1.5)

add_circle(ax2, f1, np.abs(X1_path[0]))
add_circle(ax2, f2, np.abs(X2_path[0]))

# real/imag axes at each frequency (to emphasize amplitude = distance to origin)
def add_axes(ax, f1, f2):
    ax.plot([f1, f2], [0, 0], [0, 0], color='k', lw=1)   # real axis
add_axes(ax2, f1, f2)
# add_axes(ax2, f2, np.abs(X2_path[0]) * 1.1)

ax2.set_title('(b) frequency coefficients (complex)')
ax2.set_xlabel('frequency (cycles/sample)')
ax2.set_ylabel('real')
ax2.set_zlabel('imag')
ax2.view_init(elev=18, azim=-55)


plt.tight_layout()
plt.savefig('../figures/fourier_shift.pdf')
plt.show()

# %%
