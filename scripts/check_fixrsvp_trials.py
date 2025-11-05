## !/usr/bin/env python3


#%% Setup and Imports

# this is to suppress errors in the attempts at compilation that happen for one of the loaded models because it crashed
import sys
sys.path.append('..')
import numpy as np

from DataYatesV1 import enable_autoreload, get_free_device, get_session, get_complete_sessions
from models.data import prepare_data
from models.config_loader import load_dataset_configs

import matplotlib.pyplot as plt

import torch

enable_autoreload()
device = get_free_device()


# %% Try loading a model and predicting
from eval.eval_stack_multidataset import load_model, load_single_dataset, scan_checkpoints

import matplotlib as mpl

# embed TrueType fonts in PDF/PS
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

# (optional) pick a clean sans‐serif
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

enable_autoreload()

device = get_free_device()

#%% Load an example model (this will be provided in the logging)
print("Discovering available models...")
# checkpoint_dir = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120_backimage_history/checkpoints"
checkpoint_dir = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120_backimage_8/checkpoints"
models_by_type = scan_checkpoints(checkpoint_dir, verbose=False)

print(f"Found {len(models_by_type)} model types:")
for model_type, models in models_by_type.items():
    if models:
        best_model = models[0]
        if best_model.get('metric_type') == 'bps' and best_model.get('val_bps') is not None:
            best_metric = f"best BPS: {best_model['val_bps']:.4f}"
        else:
            best_metric = f"best loss: {best_model['val_loss']:.4f}"
        print(f"  {model_type}: {len(models)} models ({best_metric})")
    else:
        print(f"  {model_type}: 0 models")

# LOAD A MODEL
model_type = 'dense_concat_convgru'
model, model_info = load_model(
        model_type=model_type,
        model_index=0, # none for best model
        checkpoint_path=None,
        checkpoint_dir=checkpoint_dir,
        device='cpu'
    )

model.model.eval()
model.model.convnet.use_checkpointing = False 

model = model.to(device)

#%% Fast Logging code
if hasattr(model.model, 'frontend') and hasattr(model.model.frontend, 'temporal_conv'):
    fig_frontend = plt.plot(model.model.frontend.temporal_conv.weight.squeeze().detach().cpu().T)
    plt.title('Frontend Kernels')

if hasattr(model.model.convnet, 'stem'):
    fig_stem = model.model.convnet.stem.components.conv.plot_weights()
    plt.title('Stem Kernels')

# %%
from eval.eval_stack_utils import load_single_dataset
from DataYatesV1.exp.fix_rsvp import FixRsvpTrial
from DataYatesV1 import get_clock_functions
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
import torch
from eval.eval_stack_utils import run_model

dataset_idx = 10


#%%
for dataset_idx in range(len(model.names)):
    print(f"Dataset {dataset_idx}: {model.names[dataset_idx]}")
    
    try:    
        train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
        
        # Combine all indices (train + validation) for maximum data (we don't tend to train on gratings so this should be okay)

        inds = torch.concatenate([
            train_data.get_dataset_inds('fixrsvp'),
            val_data.get_dataset_inds('fixrsvp')
        ], dim=0)

        dataset = train_data.shallow_copy()
        dataset.inds = inds

        # Some warm up plotting and getting key variables
        dset_idx = inds[:,0].unique().item()
        trial_inds = dataset.dsets[dset_idx].covariates['trial_inds'].numpy()
        trials = np.unique(trial_inds)

        NC = dataset.dsets[dset_idx]['robs'].shape[1]
        T = np.max(dataset.dsets[dset_idx].covariates['psth_inds'][:].numpy()).item() + 1
        NT = len(trials)

        fixation = np.hypot(dataset.dsets[dset_idx]['eyepos'][:,0].numpy(), dataset.dsets[dset_idx]['eyepos'][:,1].numpy()) < 1

        #%%
        robs = np.nan*np.zeros((NT, T, NC))
        rhat = np.nan*np.zeros((NT, T, NC))
        eyepos = np.nan*np.zeros((NT, T, 2))
        fix_dur =np.nan*np.zeros((NT,))
        for itrial in range(NT):
            ix = trials[itrial] == trial_inds
            ix = ix & fixation

            stim_inds = np.where(ix)[0]
            stim_inds = stim_inds[:,None] - np.array(dataset_config['keys_lags']['stim'])[None,:]
            stim = dataset.dsets[dset_idx]['stim'][stim_inds].permute(0,2,1,3,4)
            behavior = dataset.dsets[dset_idx]['behavior'][ix]

            out = run_model(model, {'stim': stim, 'behavior': behavior}, dataset_idx=dataset_idx)

            psth_inds = dataset.dsets[dset_idx].covariates['psth_inds'][ix].numpy()
            robs[itrial][psth_inds] = dataset.dsets[dset_idx]['robs'][ix].numpy()
            rhat[itrial][psth_inds] = out['rhat'].detach().cpu().numpy()
            eyepos[itrial][psth_inds] = dataset.dsets[dset_idx]['eyepos'][ix].numpy()
            fix_dur[itrial] = len(psth_inds)

        #%%
        good_trials = fix_dur > 40
        robs = robs[good_trials]
        rhat = rhat[good_trials]
        eyepos = eyepos[good_trials]
        fix_dur = fix_dur[good_trials]

        ind = np.argsort(fix_dur)[::-1]
        plt.subplot(1,2,1)
        plt.imshow(eyepos[ind,:,0])
        plt.xlim(0, 60)
        plt.subplot(1,2,2)
        plt.imshow(np.nanmean(robs,2)[ind])
        plt.xlim(0, 60)

        sx = int(np.sqrt(NC))
        sy = int(np.ceil(NC / sx))
        fig, axs = plt.subplots(sy, sx, figsize=(2*sx, 2*sy), sharex=True, sharey=False)
        for i in range(sx*sy):
            if i >= NC:
                axs.flatten()[i].axis('off')
                continue
            # axs.flatten()[i].imshow(robs[:, :, i][ind], aspect='auto', interpolation='none', cmap='gray_r')
            rbar = np.nanmean(robs[:, :, i][ind], 0)[:80]
            rhatbar = np.nanmean(rhat[:, :, i][ind], 0)[:80]
            iix = np.isfinite(rbar) & np.isfinite(rhatbar)
            rbar = rbar[iix]
            rhatbar = rhatbar[iix]
            # zscore
            rbar = (rbar - rbar.mean()) / rbar.std()
            rhatbar = (rhatbar - rhatbar.mean()) / rhatbar.std()
            ax = axs.flatten()[i]
            ax.plot(rbar, 'k')
            ax.plot(rhatbar, 'r')
            rho = np.corrcoef(rbar, rhatbar)[0,1]
            ax.set_title(f'{i} rho={rho:.2f}')
            ax.axis('off')

        ax.set_xlim(0, 60)

        plt.savefig(f"../figures/fixrsvp_{dataset.dsets[dset_idx].metadata['sess'].name}_{dataset_idx}_PSTH.pdf")


        # load session
        sess = dataset.dsets[dset_idx].metadata['sess']
        ks_results = sess.ks_results
        st = ks_results.spike_times
        clu = ks_results.spike_clusters

        trials = dataset.dsets[dset_idx].covariates['trial_inds'].unique().numpy().astype(int)
        fixrsvp_trials = [FixRsvpTrial(sess.exp['D'][iT], sess.exp['S']) for iT in trials]
        ptb2ephys, _ = get_clock_functions(sess.exp)
        
        # DEBUGGING
        # i_trial = 0
        # #%%
        # i_trial += 1
        # if i_trial >= len(trials):
        #     i_trial = 0
        # this_trial = trials[i_trial]

        # ephys_start = sess.exp['D'][this_trial]['START_EPHYS']
        # ephys_end = sess.exp['D'][this_trial]['END_EPHYS']

        # ix = this_trial == trial_inds
        # print(f"Trial {this_trial} has {np.sum(ix)} frames")

        # eyepos = dataset.dsets[dset_idx]['eyepos'][ix].numpy()
        # stim = dataset.dsets[dset_idx]['stim'][ix]
        # robs = dataset.dsets[dset_idx]['robs'][ix].numpy()


        # plt.figure(figsize=(10,5))
        # plt.subplot(3,1,1)
        # grid = make_grid(stim, nrow=20, normalize=True, scale_each=True, padding=2, pad_value=1)
        # plt.imshow(grid.detach().cpu().permute(1, 2, 0).numpy(), aspect='auto', interpolation='none')
        # plt.axis('off')
        # plt.title(f"Trial {this_trial}")

        # plt.subplot(3,1,2)
        # plt.imshow(robs.T, aspect='auto',cmap='gray_r', interpolation='none', extent=[dataset.dsets[dset_idx]['t_bins'][ix][0], dataset.dsets[dset_idx]['t_bins'][ix][-1], 0, robs.shape[1]], origin='lower')
        # yd = plt.gca().get_ylim()
        # plt.gca().twinx
        # st_ix = (st >= ephys_start) & (st <= ephys_end)

        # for i, cid in enumerate(dataset.dsets[dset_idx].metadata['cids']):
        #     iix = st_ix * (clu == cid)
        #     plt.plot(st[iix], i*np.ones(np.sum(iix)), 'g.', markersize=1)

        # plt.ylim(yd)

        # plt.gca().twinx()
        # plt.plot(ephys_start + sess.exp['D'][this_trial]['eyeSmo'][:,0], sess.exp['D'][this_trial]['eyeSmo'][:,1])
        # plt.plot(dataset.dsets[dset_idx]['t_bins'][ix], eyepos[:,0], 'r')
        # plt.ylim(-2, 2)
        # plt.axvline(dataset.dsets[dset_idx]['t_bins'][ix][0], color='k', linestyle='--')
        # plt.axvline(dataset.dsets[dset_idx]['t_bins'][ix][-1], color='k', linestyle='--')

        # t0 = ptb2ephys(sess.exp['D'][this_trial]['PR']['NoiseHistory'][0][0])
        # plt.axvline(t0, color='y', linestyle='--')

        # plt.gca().twinx()
        # plt.plot(ptb2ephys(fixrsvp_trials[i_trial].flip_times), fixrsvp_trials[i_trial].image_ids, 'm.')
        # xd = plt.xlim()

        # plt.subplot(3,1,3)
        # stim_inds = np.where(ix)[0]
        # stim_inds = stim_inds[:,None] - np.array(dataset_config['keys_lags']['stim'])[None,:]
        # stim = dataset.dsets[dset_idx]['stim'][stim_inds].permute(0,2,1,3,4)
        # behavior = dataset.dsets[dset_idx]['behavior'][ix]

        # from eval.eval_stack_utils import run_model
        # out = run_model(model, {'stim': stim, 'behavior': behavior}, dataset_idx=dataset_idx)
        # # out['rhat']

        # plt.imshow(out['rhat'].detach().cpu().numpy().T, aspect='auto',cmap='gray_r', interpolation='none', extent=[dataset.dsets[dset_idx]['t_bins'][ix][0], dataset.dsets[dset_idx]['t_bins'][ix][-1], 0, robs.shape[1]], origin='lower')
        # plt.axvline(dataset.dsets[dset_idx]['t_bins'][ix][0], color='k', linestyle='--')
        # plt.axvline(dataset.dsets[dset_idx]['t_bins'][ix][-1], color='k', linestyle='--')
        # plt.xlim(xd)

        # #%%

        # # [plt.axvline(t, color='gray', alpha=.2) for t in dataset.dsets[dset_idx]['t_bins'][ix]]

        # --- helper: draw ONE trial into two axes on the current figure ---
        def draw_trial(ax_top, ax_middle, ax_bottom, this_trial, dset_idx, model, dataset_idx):
            
            ix = (trial_inds == this_trial)
            eyepos = dataset.dsets[dset_idx]['eyepos'][ix].numpy()
            stim   = dataset.dsets[dset_idx]['stim'][ix]
            robs   = dataset.dsets[dset_idx]['robs'][ix].numpy()
            t_bins = dataset.dsets[dset_idx]['t_bins'][ix]
            ephys_start = sess.exp['D'][this_trial]['START_EPHYS']
            ephys_end   = sess.exp['D'][this_trial]['END_EPHYS']

            # --- top panel: stimulus grid ---
            grid = make_grid(stim, nrow=20, normalize=True, scale_each=True, padding=2, pad_value=1)
            ax_top.imshow(grid.detach().cpu().permute(1, 2, 0).numpy(),
                        aspect='auto', interpolation='none')
            ax_top.set_axis_off()
            ax_top.set_title(f"Trial {this_trial}  |  {np.sum(ix)} frames", fontsize=10)

            # --- bottom panel: spikes + eye traces ---
            im = ax_middle.imshow(
                robs.T, aspect='auto', cmap='gray_r', interpolation='none',
                extent=[t_bins[0], t_bins[-1], 0, robs.shape[1]], origin='lower'
            )
            yd = ax_middle.get_ylim()

            # overlay spikes per cluster (assumes st, clu, and st_ix logic in scope)
            st_ix = (st >= ephys_start) & (st <= ephys_end)
            for i, cid in enumerate(dataset.dsets[dset_idx].metadata['cids']):
                iix = st_ix & (clu == cid)
                # tiny points for raster
                ax_middle.plot(st[iix], i*np.ones(np.sum(iix)), 'g.', markersize=1)

            ax_middle.set_ylim(yd)
            ax_middle.set_ylabel("Neuron", fontsize=8)
            ax_middle.set_xlabel("Time (s)", fontsize=8)

            # second y-axis for eye traces
            ax_eye = ax_middle.twinx()
            ax_eye.plot(ephys_start + sess.exp['D'][this_trial]['eyeSmo'][:,0],
                        sess.exp['D'][this_trial]['eyeSmo'][:,1])
            ax_eye.plot(t_bins, eyepos[:,0], 'r')
            ax_eye.set_ylim(-2, 2)
            ax_eye.set_yticks([])
            ax_eye.set_xlim(t_bins[0]-.4, t_bins[0]+1)

            # trial window & markers
            ax_middle.axvline(t_bins[0], color='k', linestyle='--', linewidth=0.8)
            ax_middle.axvline(t_bins[-1], color='k', linestyle='--', linewidth=0.8)

            t0 = ptb2ephys(sess.exp['D'][this_trial]['PR']['NoiseHistory'][0][0])
            ax_middle.axvline(t0, color='y', linestyle='--', linewidth=0.8)

            ax_frames = ax_middle.twinx()
            fixrsvp_trial = FixRsvpTrial(sess.exp['D'][this_trial], sess.exp['S'])
            ax_frames.plot(ptb2ephys(fixrsvp_trial.flip_times), fixrsvp_trial.image_ids, 'm.')
            ax_frames.set_yticks([])
            ax_frames.set_ylim(0, 25)

            stim_inds = np.where(ix)[0]
            stim_inds = stim_inds[:,None] - np.array(dataset_config['keys_lags']['stim'])[None,:]
            stim = dataset.dsets[dset_idx]['stim'][stim_inds].permute(0,2,1,3,4)
            behavior = dataset.dsets[dset_idx]['behavior'][ix]

            out = run_model(model, {'stim': stim, 'behavior': behavior}, dataset_idx=dataset_idx)
            print(out['rhat'].shape)
            
            ax_bottom.imshow(out['rhat'].detach().cpu().numpy().T, aspect='auto',cmap='gray_r', interpolation='none',
                            extent=[t_bins[0], t_bins[-1], 0, robs.shape[1]], origin='lower'
            )
            ax_bottom.axvline(t_bins[0], color='k', linestyle='--', linewidth=0.8)
            ax_bottom.axvline(t_bins[-1], color='k', linestyle='--', linewidth=0.8)
            ax_bottom.set_xlim(t_bins[0]-.4, t_bins[0]+1)

        # --- PDF builder: pack multiple trials per page ---
        pdf_path = f"../figures/FixRsvp_{dataset.dsets[dset_idx].metadata['sess'].name}_{model_type}.pdf"
        trials_per_page = 3
        page_size = (8.5, 11)   # inches (Letter)
        test_mode = False
        if test_mode:
            num_trials = 5
        else:
            num_trials = len(trials)
        with PdfPages(pdf_path) as pdf:
            # loop in pages
            for start in range(0, num_trials, trials_per_page):
                end = min(start + trials_per_page, len(trials))
                n_on_page = end - start

                fig = plt.figure(figsize=page_size)
                outer = gridspec.GridSpec(n_on_page, 1, hspace=0.35, top=0.96, bottom=0.06, left=0.06, right=0.97)

                for row, this_trial in enumerate(trials[start:end]):
                    # each trial block gets a 2x1 inner grid (top: stim, bottom: spikes+eye)
                    inner = gridspec.GridSpecFromSubplotSpec(
                        3, 1, subplot_spec=outer[row],
                        height_ratios=[1, 1.2, 1.2], hspace=0.15
                    )
                    ax_top = fig.add_subplot(inner[0])
                    ax_mid = fig.add_subplot(inner[1])
                    ax_bot = fig.add_subplot(inner[2], sharex=ax_mid)
                    try:
                        draw_trial(ax_top, ax_mid, ax_bot, this_trial, dset_idx, model, dataset_idx)
                        print(f"  ✓ Trial {this_trial} drawn successfully")
                    except Exception as e:
                        print(f"  ✗ Trial {this_trial} failed:")
                        print(f"    Error: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                pdf.savefig(fig)
                plt.close(fig)

        print(f"Wrote {pdf_path}")
    
    except Exception as e:
        print(f"✗ Failed to load dataset:")
        import traceback
        traceback.print_exc()
