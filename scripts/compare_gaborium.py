#%%

from models.data import DictDataset
import numpy as np
import matplotlib.pyplot as plt

yates_dset = DictDataset.load('/mnt/ssd/YatesMarmoV1/processed/Allen_2022-04-13/datasets/gaborium.dset')
yates_stim = yates_dset['stim']
yates_eyepos = yates_dset['eyepos']
yates_ecc = np.hypot(yates_eyepos[:, 0], yates_eyepos[:, 1])
print('Yates dataset loaded.')
print(yates_dset)

rowley_dset = DictDataset.load('/mnt/ssd/RowleyMarmoV1V2/processed/Luke_2025-08-04/datasets/right_eye/gaborium.dset')
rowley_stim = rowley_dset['stim']
rowley_eyepos = rowley_dset['eyepos']
rowley_ecc = np.hypot(rowley_eyepos[:, 0], rowley_eyepos[:, 1])
print('Rowley dataset loaded.')
print(rowley_dset)
#%%
# Find the longest contiguous segment within a given eccentricity range
max_ecc = 5
yates_mask = yates_ecc < max_ecc
rowley_mask = rowley_ecc < max_ecc
def longest_contiguous_segment(mask):
    max_len = 0
    max_start = 0
    current_len = 0
    current_start = 0
    for i, val in enumerate(mask):
        if val:
            if current_len == 0:
                current_start = i
            current_len += 1
            if current_len > max_len:
                max_len = current_len
                max_start = current_start
        else:
            current_len = 0
    return max_start, max_start + max_len

yates_start, yates_end = longest_contiguous_segment(yates_mask)
print(f'Yates longest contiguous segment: {yates_start} to {yates_end} ({yates_end - yates_start} samples)')
rowley_start, rowley_end = longest_contiguous_segment(rowley_mask)
print(f'Rowley longest contiguous segment: {rowley_start} to {rowley_end} ({rowley_end - rowley_start} samples)')

#%%
n_frames = 240
rowley_frames = rowley_stim[rowley_start:rowley_start + n_frames]
yates_frames = yates_stim[yates_start:yates_start + n_frames]

# Squeeze channel dim if present (e.g. shape N x 1 x H x W)
if yates_frames.ndim == 4:
    yates_frames = np.asarray(yates_frames).squeeze(1)
if rowley_frames.ndim == 4:
    rowley_frames = np.asarray(rowley_frames).squeeze(1)

#%%
from matplotlib.animation import FuncAnimation

original_fps = 240  # Hz
playback_fps = 24   # 0.1x speed

fig, (ax_yates, ax_rowley) = plt.subplots(1, 2, figsize=(10, 5))

im_yates = ax_yates.imshow(yates_frames[0], cmap='gray', vmin=0, vmax=255)
ax_yates.set_title('Yates')
ax_yates.axis('off')

im_rowley = ax_rowley.imshow(rowley_frames[0], cmap='gray', vmin=0, vmax=255)
ax_rowley.set_title('Rowley')
ax_rowley.axis('off')

suptitle = fig.suptitle('Sample 0 | t = 0.000 s')

def update(frame_idx):
    im_yates.set_data(yates_frames[frame_idx])
    im_rowley.set_data(rowley_frames[frame_idx])
    t = frame_idx / original_fps
    suptitle.set_text(f'Sample {frame_idx} | t = {t:.3f} s')
    return im_yates, im_rowley, suptitle

anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / playback_fps, blit=False)

# Save as mp4
anim.save('scripts/compare_gaborium.mp4', writer='ffmpeg', fps=playback_fps)
print('Saved scripts/compare_gaborium.mp4')

plt.close(fig)

from IPython.display import HTML
HTML(anim.to_jshtml(fps=playback_fps))
