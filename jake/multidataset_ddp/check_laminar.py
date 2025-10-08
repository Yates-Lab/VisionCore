
#%%
import numpy as np
import matplotlib.pyplot as plt
from DataYatesV1 import get_complete_sessions
sessions = get_complete_sessions()

#%%
n_sessions = len(sessions)
sx = np.floor(np.sqrt(n_sessions)).astype(int)
sy = np.ceil(n_sessions / sx).astype(int)
fig, axs = plt.subplots(sy, sx, figsize=(25, 25))

for i, sess in enumerate(sessions):
    
    try:
        laminar_results = np.load(sess.sess_dir / 'laminar' / 'laminar.npz', allow_pickle=True)

        psd_dict = laminar_results['psd'].item()
        psd_data = psd_dict['psd']  # List of arrays, one per shank
        depths = psd_dict['depths']
        for j in range(len(psd_data)):
            psd_data[j] = psd_data[j].T
        ax = axs.flatten()[i]
        ax.imshow(np.concatenate(psd_data, 1), aspect='auto', cmap='YlOrBr', extent=[psd_dict['freqs'][0], psd_dict['freqs'][-1], depths[-1][-1], depths[0][0]], origin='lower')
        ax.set_title(sess.sess_dir.name)
        ax.axvline(psd_data[0].shape[1], color='k', linestyle='--')
          # List of depth arrays, one per shank
        # freqs = psd_dict['freqs']  # Frequency bins (406 frequencies from 300-3000 Hz)
        
    except Exception as e:
        print(f"Error processing session {sess}: {str(e)}")

plt.savefig('laminar_psd.png')
# %%
