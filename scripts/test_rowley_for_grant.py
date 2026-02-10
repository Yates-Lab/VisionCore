
#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
from tejas.rsvp_util import remove_duplicate_trials, align_image_ids

def fill_small_nan_gaps_prev_rowwise(X, k):
    """
    Row-wise: fill NaN runs of length <= k with the value immediately preceding the run.
    Leading NaNs stay NaN. Rows are independent. Gaps are along columns.

    Parameters
    ----------
    X : array-like, shape (R, C)
    k : int >= 0

    Returns
    -------
    Y : np.ndarray, shape (R, C)
    """
    X = np.asarray(X, dtype=float)
    Y = X.copy()
    if k <= 0:
        return Y

    R, C = Y.shape
    nan = np.isnan(Y)

    # Identify starts of NaN runs per row: nan & not nan immediately to the left
    left_nan = np.concatenate([np.zeros((R, 1), dtype=bool), nan[:, :-1]], axis=1)
    run_start = nan & ~left_nan

    # Label runs within each row: 1,2,3,... (0 for non-NaNs)
    run_id = np.cumsum(run_start, axis=1)
    run_id = run_id * nan  # 0 where not NaN

    max_runs = int(run_id.max())
    if max_runs == 0:
        return Y

    # Count run lengths: counts[r, rid] = number of NaNs in that run
    counts = np.zeros((R, max_runs + 1), dtype=np.int32)
    rr, cc = np.nonzero(nan)
    rid = run_id[rr, cc].astype(np.int32)
    np.add.at(counts, (rr, rid), 1)

    # For each run, store whether it has a valid previous value, and that fill value
    prev_ok = np.zeros((R, max_runs + 1), dtype=bool)
    fill_val = np.full((R, max_runs + 1), np.nan, dtype=float)

    rs_r, rs_c = np.nonzero(run_start)
    rs_id = run_id[rs_r, rs_c].astype(np.int32)

    has_prev = rs_c > 0
    prev_is_finite = np.zeros_like(has_prev, dtype=bool)
    prev_is_finite[has_prev] = np.isfinite(Y[rs_r[has_prev], rs_c[has_prev] - 1])

    ok = has_prev & prev_is_finite
    prev_ok[rs_r[ok], rs_id[ok]] = True
    fill_val[rs_r[ok], rs_id[ok]] = Y[rs_r[ok], rs_c[ok] - 1]

    # Decide which NaN entries to fill: run length <= k and run has valid previous value
    run_len = counts[rr, rid]
    do_fill = (run_len <= k) & prev_ok[rr, rid]

    Y[rr[do_fill], cc[do_fill]] = fill_val[rr[do_fill], rid[do_fill]]
    return Y


def plot_raster(ii, jj, height=1, ax=None, **kwargs):

    ii = np.stack([ii, ii, np.nan*np.ones_like(ii)], 1).flatten()
    jj = np.stack([jj, jj+height, np.nan*np.ones_like(jj)], 1).flatten()

    if ax is None:
        ax = plt.gca()
    ax.plot(ii, jj, 'k', **kwargs)


from scipy.sparse import coo_matrix
def bin_spikes(spike_times, spike_clusters, t_bins, n_units=None):
    time_bins = np.digitize(spike_times, t_bins) - 1
    spike_indices = spike_clusters  # Don't subtract 1 - use cluster IDs directly as indices

    if n_units is None:
        n_units = int(np.max(spike_clusters)) + 1  # +1 because 0-indexed

    # Filter out spikes outside valid bin range and invalid cluster indices
    n_time_bins = len(t_bins) - 1
    valid = (time_bins >= 0) & (time_bins < n_time_bins) & (spike_indices >= 0) & (spike_indices < n_units)
    time_bins = time_bins[valid]
    spike_indices = spike_indices[valid]
    n_spikes = len(time_bins)

    counts = coo_matrix(
            (np.ones(n_spikes, dtype=int), (time_bins.astype(int), spike_indices.astype(int))),
            shape=(n_time_bins, n_units),
        )
    return counts
#%%
# import dictdataset

fpath = '/mnt/ssd/RowleyMarmoV1V2/processed/Luke_2025-08-04/datasets/right_eye/fixrsvp.dset'

from models.data import DictDataset
dset = DictDataset.load(fpath)

use_rowley = True
if use_rowley:
    from DataRowleyV1V2.data.registry import get_session
    sess = get_session('Luke', '2025-08-04')
    spikes = sess.load_spikes()  
    st = spikes[0]      
    clu = spikes[1]
    exp = sess.load_exp()
    ptb2ephys = sess.load_ptb2ephys()[0]
else:
    from DataYatesV1 import get_session
    sess = get_session('Allen', '2022-02-18')
    st = sess.get_spike_times()
    clu = sess.get_spike_clusters()
    exp = sess.exp
    ptb2ephys = sess.ptb2ephys

#%%
def get_trial_aligned(exp, ptb2ephys, st, clu, dt=1/240):
    num_trials = len(exp['D'])
    trial_protocols = [exp['D'][i]['PR']['name'] for i in range(num_trials)]

    fixrsvp_trials = [i for i in range(num_trials) if trial_protocols[i] == 'FixRsvpStim']

    NT = len(fixrsvp_trials)
    nclu = np.max(clu)
    time_bins = np.arange(-1, 2, dt)

    binned_spikes = np.nan*np.zeros((NT, len(time_bins)-1, nclu))
    state = np.nan*np.zeros((NT, len(time_bins)-1))
    eyepos = np.nan*np.zeros((NT, len(time_bins)-1, 3))
    image_id = np.nan*np.zeros((NT, len(time_bins)-1))
    breakfix = np.nan*np.zeros(NT)
    ctrfix = np.nan*np.zeros((NT, 2))


    for i in range(NT):
        itrial = fixrsvp_trials[i]
        NH = exp['D'][itrial]['PR']['NoiseHistory']
        if NH is None or NH.shape[0] < 10:
            continue

        eye_data = exp['D'][itrial]['eyeData']

        c = exp['D'][itrial]['c']
        dx = exp['D'][itrial]['dx']
        dy = exp['D'][itrial]['dy']

        eye_data[:,2] = (eye_data[:,2] - c[1])*dy
        eye_data[:,1] = (eye_data[:,1] - c[0])*dx

        t2 = np.where(NH[:,3]==2)[0][0]
        eye_time = ptb2ephys(eye_data[:,0])
        frame_times = ptb2ephys(NH[:,0])

        eye_time = eye_time - frame_times[0]
        # digitize to time_bins
        eye_digi = np.digitize(eye_time, time_bins) - 1
        eye_iix = (eye_digi >= 0) & (eye_digi < len(time_bins)-1)
        nh_digi = np.digitize(frame_times-frame_times[0], time_bins) - 1
        nh_good = (nh_digi >= 0) & (nh_digi < len(time_bins)-1)
        
        image_id[i, nh_digi[nh_good]] = NH[nh_good,3]
        eyepos[i, eye_digi[eye_iix], :] = eye_data[eye_iix, 1:4]
        state[i, eye_digi[eye_iix]] = eye_data[eye_iix, 4]
        
        ctr_ = np.nanmean(eyepos[i][(time_bins[:-1]>0) & (time_bins[:-1]<.1)], 0)
        ctrfix[i] = ctr_[:2]
        st_ix = (st > frame_times[0]+time_bins[0]) & (st < frame_times[0]+time_bins[-1])

        trial_st = st[st_ix] - frame_times[0]
        trial_clu = clu[st_ix]
        st_binned = bin_spikes(trial_st, trial_clu, time_bins, n_units=nclu)
        binned_spikes[i] = st_binned.toarray()

        try:    
            breakfix[i] = np.where(state[i]==3)[0][0]
        except:
            breakfix[i] = np.nan

    ctr = np.nanmean(ctrfix, 0)
    nanfill = int(.010/dt)
    eyepos[:,:,0] = fill_small_nan_gaps_prev_rowwise(eyepos[:,:,0], nanfill) - ctr[0]
    eyepos[:,:,1] = fill_small_nan_gaps_prev_rowwise(eyepos[:,:,1], nanfill) - ctr[1]

    dfs = np.isfinite(image_id)[:,:,None].repeat(binned_spikes.shape[-1], axis=2)
    image_id = fill_small_nan_gaps_prev_rowwise(image_id, 2)

    # get fixation duration
    Y = image_id.copy()
    R, C = Y.shape
    nan = np.isnan(Y)

    # Identify starts of NaN runs per row: nan & not nan immediately to the left
    left_nan = np.concatenate([np.zeros((R, 1), dtype=bool), nan[:, :-1]], axis=1)
    run_start = nan & ~left_nan

    # Label runs within each row: 1,2,3,... (0 for non-NaNs)
    run_id = np.cumsum(run_start, axis=1)
    run_id = run_id * nan  # 0 where not NaN

    fix_dur = np.zeros(R)
    for i in range(R):
        try:
            fix_dur[i] = np.where(run_id[i]==2)[0][0]
        except:
            fix_dur[i] = np.nan

    good_trials = (fix_dur-np.where(time_bins[:-1]>0)[0][0]) > 20

    print(f"Found {good_trials.sum()}/{len(good_trials)} good trials")
    # # remove bad trials
    # binned_spikes = binned_spikes[good_trials]
    # eyepos = eyepos[good_trials][:,:,[0,1]]
    # image_id = image_id[good_trials]
    # fix_dur = fix_dur[good_trials]
    # dfs = dfs[good_trials]
    # breakfix = breakfix[good_trials]
    return binned_spikes, eyepos, image_id, fix_dur, dfs, breakfix


binned_spikes, eyepos, image_id, fix_dur, dfs, breakfix = get_trial_aligned(exp, ptb2ephys, st, clu, dt=1/240)

#%%
ind = np.argsort(fix_dur)

plt.subplot(1,2,1)
plt.imshow(eyepos[ind][:,:,0],
           vmin=-.5, vmax=.5,
           aspect='auto',
           cmap='coolwarm', interpolation='none',
           origin='lower',
           extent=[time_bins[0], time_bins[-1], 0, eyepos.shape[0]])

plt.xlim(-.1, 1.0)

plt.subplot(1,2,2)
plt.imshow(image_id[ind],
    aspect='auto',
    interpolation='none', origin='lower', vmin=0, vmax=20,
    extent=[time_bins[0], time_bins[-1], 0, eyepos.shape[0]])
plt.xlim(-.1, 1.0)
    
#%%
v1cids = [800, 801, 802, 805, 806, 807, 810, 815, 816, 818, 820, 823, 825, 828, 835, 841, 844, 846, 850, 852, 853, 854, 859, 863, 867, 875, 880, 883, 884, 887, 888, 894, 902, 904, 908, 911, 917, 923, 924, 925, 931, 933, 936, 938, 939, 940, 942, 943, 950]
v2cids = [356, 358, 359, 361, 362, 363, 364, 365, 367, 368, 370, 372, 373, 374, 377, 378, 379, 380, 381,  390, 398, 410, 411, 420, 421, 427, 428, 432, 445, 447, 458, 469, 472, 473, 477, 508, 536]

cids = np.concatenate((v1cids, v2cids))

#%% plot PSTHS

from scipy.signal import savgol_filter
plot_psths = False
plot_rasters = False

if plot_psths:
    for cc in cids:
        plt.plot(time_bins[:-1], savgol_filter(np.nanmean(binned_spikes, 0)[:,cc], 10, 1))
        # plt.plot(time_bins[:-1], np.nanmean(binned_spikes, 0)[:,cc])
        plt.axvline(0, color='r', linestyle='--')
        plt.title(f'Neuron {cc}')
        plt.show()

if plot_rasters:
    for cc in cids:
        good_trials = (np.isnan(binned_spikes[:,:,cc]).sum(1)==0) & (np.sum(np.diff(np.nansum(binned_spikes, 1), 0),1)>0)

        ind = np.argsort(fix_dur[good_trials])
        jj, ii = np.where(binned_spikes[:,:,cc][good_trials][ind])
        plot_raster(time_bins[ii], jj, height=1)
        plt.axvline(0, color='r', linestyle='--')

        plt.title(f'Neuron {cc}')
        plt.show()

#%% cleanup
robs, dfs, eyepos, dur, image_ids = remove_duplicate_trials(binned_spikes, dfs, eyepos, fix_dur, image_id)


#%% plot trials
NT = robs.shape[0]

for itrial in range(NT):

    ii, jj = np.where(robs[itrial])
    plot_raster(time_bins[:-1][ii], jj, height=1)
    plt.axis('off')
    plt.gca().twinx()
    plt.plot(time_bins[:-1], eyepos[itrial][:,0], '-r')
    plt.plot(time_bins[:-1], eyepos[itrial][:,1], '-g')
    plt.ylim(-50, 50)
    plt.gca().twinx()
    plt.plot(time_bins[:-1], image_id[good_trials][itrial], '-b')
    plt.title(f"Trial {itrial}")

    # plt.plot(time_bins[:-1], eyepos[good_trials][itrial][:,2], '.m')
    plt.xlim(-.250, 0.50)
    plt.show()

#%%

from mcfarland_sim import DualWindowAnalysis

iix = (time_bins[:-1] > 0) & (time_bins[:-1] < 1.5)
robs_used = robs[:,iix]
eyepos_used = eyepos[:,iix]
valid_mask = dfs[:,iix].sum(2)>0

# cids = np.where(robs.sum((0,1))>1000)[0] #np.concat([v1cids, v2cids])
analyzer = DualWindowAnalysis(robs_used[:,:,cids], eyepos_used, valid_mask, dt=dt)

#%%
results, last_mats = analyzer.run_sweep([10, 20, 40], t_hist_ms=50, n_bins=35)

#%%

from mcfarland_sim import project_to_psd

# import cmasher as cmr
# cmap = plt.get_cmap('cmr.prinsenvlag')   # MPL
cmap = plt.get_cmap('RdBu')   # MPL
window_idx = 0


Ctotal = last_mats[window_idx]['Total']
Cpsth = last_mats[window_idx]['PSTH']
Crate = last_mats[window_idx]['Intercept']
Cfem = last_mats[window_idx]['FEM']

MeanRates = results[window_idx]['Erates']

# project to psd
Ctotal = project_to_psd(Ctotal)
Cpsth = project_to_psd(Cpsth)
Crate = project_to_psd(Crate)
Cfem = project_to_psd(Cfem)
CnoiseC = Ctotal - Crate
CnoiseU = Ctotal - Cpsth
CnoiseU = project_to_psd(CnoiseU)
CnoiseC = project_to_psd(CnoiseC)

#%%
v = np.max(Ctotal.flatten()) * .5

fig, axs = plt.subplots(1,3, figsize=(20,5))
ax = axs[0]
ax.imshow(Ctotal, cmap=cmap, interpolation='nearest', vmin=-v, vmax=v)
ax.set_title('Total')
ax.axis('off')

ax = axs[1]
ax.imshow(Cpsth, cmap=cmap, interpolation='nearest', vmin=-v/2, vmax=v/2)
ax.set_title('PSTH')
ax.axis('off')

ax = axs[2]
ax.imshow(CnoiseU, cmap=cmap, interpolation='nearest', vmin=-v, vmax=v)
ax.set_title('Noise (Uncorrected)')
ax.axis('off')
# fig.savefig(f'../figures/mcfarland/covariance_decomposition_{i}_{window_idx}_psth.pdf', bbox_inches='tight', dpi=300) 



fig, axs = plt.subplots(1,4, figsize=(20,5))
ax = axs[0]
ax.imshow(Ctotal, cmap=cmap, interpolation='nearest', vmin=-v, vmax=v)
ax.set_title('Total')
ax.axis('off')
ax = axs[1]
ax.imshow(Cfem, cmap=cmap, interpolation='nearest', vmin=-v, vmax=v)
ax.set_title('FEM')
ax.axis('off')

ax = axs[2]
ax.imshow(Cpsth, cmap=cmap, interpolation='nearest', vmin=-v/2, vmax=v/2)
ax.set_title('PSTH')
ax.axis('off')

ax = axs[3]
ax.imshow(CnoiseC, cmap=cmap, interpolation='nearest', vmin=-v, vmax=v)
ax.set_title('Noise (Corrected)')
ax.axis('off')
    # fig.savefig(f'../figures/mcfarland/covariance_decomposition_{i}_{window_idx}_full.pdf', bbox_inches='tight', dpi=300) 

#%% Alpha
isv1 = np.isin(cids, v1cids)
isv2 = np.isin(cids, v2cids)
colors = ['b', 'g']

alpha = np.diag(Cpsth) / np.diag(Crate)
plt.hist(1-alpha[isv1], bins=np.linspace(0, 1, 50), color=colors[0], label='V1')
plt.hist(1-alpha[isv2], bins=np.linspace(0, 1, 50), color=colors[1], label='V2')
plt.xlabel('Frac. FEM (1-alpha)')
plt.ylabel('Count')
plt.legend()
plt.savefig('./grant_figs_alpha.pdf')
plt.show()

#%%


plt.figure(figsize=(4,2))
plt.subplot(1,2,1)

plt.plot(MeanRates[isv1], np.diag(CnoiseU)[isv1], '.', label='V1')
plt.plot(MeanRates[isv2], np.diag(CnoiseU)[isv2], '.', label='V2')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('Mean Rate')
plt.ylabel('Variance')

plt.subplot(1,2,2)

plt.plot(MeanRates[isv1], np.diag(CnoiseC)[isv1], '.', label='V1')
plt.plot(MeanRates[isv2], np.diag(CnoiseC)[isv2], '.', label='V2')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('Mean Rate')
plt.ylabel('Variance')


#%%
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages(f'unit_rasters_{sess.name}.pdf') as pdf:
    for i, cc in enumerate(cids):
        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        ax[0].set_title(f"Neuron {cc}")
        ind = np.argsort(dur)
        jj, ii = np.where(binned_spikes[:,:,cc][ind])
        plot_raster(time_bins[ii], jj, height=1, ax=ax[0])
        ax[0].axvline(0, color='r', linestyle='--')
        ax[0].set_xlim(-0.1, 1.5)

        analyzer.inspect_neuron_pair(i, i, 10, ax=ax[1], show=True)
        pdf.savefig(fig)
        plt.close(fig)
#%%

from matplotlib.backends.backend_pdf import PdfPages

with PdfPages(f'trial_rasters_{sess.name}.pdf') as pdf:
    for itrial in range(good_trials.sum()):
        if itrial >= good_trials.sum():
            itrial = 0
        fig, ax = plt.subplots()
        ii, jj = np.where(binned_spikes[good_trials][:,:,:][itrial])
        plot_raster(time_bins[:-1][ii], jj, height=1)
        plt.axis('off')
        plt.gca().twinx()
        plt.plot(time_bins[:-1], eyepos[good_trials][itrial][:,0], '.r')
        plt.plot(time_bins[:-1], eyepos[good_trials][itrial][:,1], '.g')
        plt.ylim(-150, 150)
        plt.gca().twinx()
        plt.plot(time_bins[:-1], image_id[good_trials][itrial], '.b')
        plt.ylim(0, 20)
        plt.title(f"Trial {itrial}")
        plt.xlim(-.250, 0.50)
        pdf.savefig(fig)
        plt.close(fig)


# %%
binned_spikes, eyepos, image_id, fix_dur, dfs, breakfix = get_trial_aligned(exp, ptb2ephys, st, clu, dt=1/1000)
eyepos = eyepos / np.nanstd(eyepos, (0,1), keepdims=True)
robs, dfs, eyepos, dur, image_ids = remove_duplicate_trials(binned_spikes, dfs, eyepos, fix_dur, image_id)
# %%
time_bins = np.arange(-1, 2, 1/1000)

NT = robs.shape[0]
sx = int(np.ceil(np.sqrt(NT)))
sy = int(np.ceil(NT / sx))

plt.figure(figsize=(sx*2, sy*2))
for itrial in range(robs.shape[0]):
    plt.subplot(sx, sy,itrial+1)
    ii, jj = np.where(robs[itrial][:,cids])
    plot_raster(time_bins[:-1][ii], jj, height=1)
    plt.axis('off')
    plt.gca().twinx()
    plt.plot(time_bins[:-1], eyepos[itrial][:,0], '-r')
    plt.plot(time_bins[:-1], eyepos[itrial][:,1], '-g')
    plt.ylim(-1, 1)
    plt.title(f"Trial {itrial}")
    # plt.plot(time_bins[:-1], eyepos[good_trials][itrial][:,2], '.m')
    plt.xlim(.25, 0.50)
    plt.axis('off')
# %%
group1 = [5, 10, 23, 32]
# group2 = [0, 2, 16, 44]
group2 = [20, 75, 16, 61]

fig, axs = plt.subplots(2,1, figsize=(10,5), sharex=True, sharey=False)
colors = ['b', 'r', 'b', 'm']
for i, group in enumerate([group1, group2]):
    for j in group:
        axs[0].plot(time_bins[:-1], eyepos[j,:,0], alpha=0.5, color=colors[i])
        axs[1].plot(time_bins[:-1], eyepos[j, :,1], alpha=0.5, color=colors[i])
 
axs[0].set_xlim(.15, 0.650)
axs[1].set_xlim(.15, 0.650)
axs[0].set_ylim(-1, 1)
axs[1].set_ylim(-1, 1)
fig.savefig('./grant_figs_eyepos.pdf')


fig, axs = plt.subplots(4,2, figsize=(5,4), sharex=True, sharey=False)
for i, group in enumerate([group1, group2]):
    for p, itrial in enumerate(group):
        ii, jj = np.where(robs[itrial][:,cids])
        ix = jj < len(v1cids)
        plot_raster(time_bins[:-1][ii[ix]], jj[ix], height=1, ax=axs[p, i], color='b')
        ix = jj > len(v1cids)
        plot_raster(time_bins[:-1][ii[ix]], jj[ix], height=1, ax=axs[p, i], color='g')
        axs[p,i].axis('off')

axs[0,0].set_xlim(.15, 0.650)
fig.savefig('./grant_figs_rasters.pdf')
# axs[1].set_title('Group 2')
# %%
robs_nan = robs.copy()

for itrial in range(robs.shape[0]):
    try:
        s = np.where(np.isfinite(image_id[itrial]))[0][-1]
        robs_nan[itrial][s:, :] = np.nan
        # eyepos[itrial][s:, :] = np.nan
    except:
        pass

psths = np.nanmean(robs_nan[:,:,cids], 0)

selectivity = np.mean(psths[time_bins[:-1]>-0.05], 0) - np.mean(psths[time_bins[:-1]<-0.2], 0)
# inds = np.argsort(selectivity)
# pick top selective
v1ix = np.zeros(len(cids), dtype=bool)
v1ix[:len(v1cids)] = True

inds = np.argsort(selectivity)[::-1]
psths = psths[:,inds]
v1ix = v1ix[inds]
selectivity = selectivity[inds]

plt.figure(figsize=(2,5))

ix = v1ix
for i, cc in enumerate(np.where(ix)[0]):
    if i > 10: break
    x = psths[:,cc]
    x = savgol_filter(x, 13, 1)
    x = x - x.min()
    x = x / x.max()
    
    color = 'b'
    plt.plot(time_bins[:-1], x + i, alpha=0.5, color=color)

maxi = i
ix = ~v1ix
for i, cc in enumerate(np.where(ix)[0]):
    if i > 10: break
    x = psths[:,cc]
    x = savgol_filter(x, 13, 1)
    x = x - x.min()
    x = x / x.max()
    
    color = 'g'
    plt.plot(time_bins[:-1], x + i + maxi, alpha=0.5, color=color)

plt.xlim(-.5, 1.5)
plt.savefig('./grant_figs_psths.pdf')

# _ = plt.plot(savgol_filter(psths/dt, 101, 2))
# %%
