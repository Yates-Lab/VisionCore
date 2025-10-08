#%%

# Note: works well for Allen datasets right now, but struggles with logan datasets where there are dead channels
# should try interpolating
import numpy as np
from DataYatesV1 import get_session, get_complete_sessions
import matplotlib.pyplot as plt
from DataYatesV1 import enable_autoreload
from DataYatesV1 import convert_time_to_samples
from DataYatesV1.exp.general import get_trial_protocols
from DataYatesV1.exp.csd import get_csd_flashes
from scipy.ndimage import gaussian_filter
enable_autoreload()

from DataYatesV1.utils import exp_get_trial_idx, get_clock_functions
from scipy.signal import welch
from scipy.signal import butter, filtfilt
import json

from pathlib import Path

sessions = get_complete_sessions()

#%%
from jake.detect_saccades import detect_saccades

def get_saccades_during_protocol(session, trial_type = 'BackImage'):
    exp = session.exp    
    trial_idx = exp_get_trial_idx(exp, trial_type)

    saccades = detect_saccades(session)
    try:
        saccade_onsets = np.array([s['start_time'] for s in saccades])
    except:
        saccade_onsets = np.array([s.start_time for s in saccades])

    trial_starts = np.asarray([exp['D'][i]['START_EPHYS'] for i in trial_idx])
    trial_stops = np.asarray([exp['D'][i]['END_EPHYS'] for i in trial_idx])

    sac_in_trial = np.where(np.sum((saccade_onsets[:,None] > trial_starts[None,:]) & (saccade_onsets[:,None] < trial_stops[None,:]), 1))[0]
    saccade_onsets = saccade_onsets[sac_in_trial]
    return saccade_onsets

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def get_flash_response(recording, flash_onset_samples, flash_window, ephys_metadata, f_range, ds):
    # good = ((flash_onset_samples + flash_window[0]) >= 0) & ((flash_onset_samples + flash_window[-1]) < recording.shape[0])
    # print(f"Number of good flashes: {np.sum(good)}")
    # print(f"Number of bad flashes: {np.sum(~good)}")
    # flash_onset_samples = flash_onset_samples[good]

    flash_responses = []
    for flash_onset in flash_onset_samples:
        try:
            response = np.abs(recording[flash_onset + flash_window])
            response_filt = butter_lowpass_filter(
                response.T, 
                f_range[1], 
                ephys_metadata['sample_rate'], 
                order=2
            ).T[::ds]
            flash_responses.append(response_filt)
        except Exception as e:
            print(f"Error: {e}")

    flash_responses = np.array(flash_responses)
    return flash_responses


def get_triggered_response(session, recording, event_times, t_win, f_range, ds, max_events=1000):

    # Get metadata and recordings
    ephys_metadata = session.ephys_metadata
    
    # Setup time conversion
    time_to_samples = lambda t: convert_time_to_samples(
    t, 
        ephys_metadata['sample_rate'], 
        ephys_metadata['block_start_times'], 
        ephys_metadata['block_n_samples']
        ).astype(int)


    flash_onset_samples = time_to_samples(event_times)

    if len(flash_onset_samples) > max_events:
        flash_onset_samples = np.random.choice(flash_onset_samples, max_events, replace=False)

    # Setup time window
    s0 = int(t_win[0] * ephys_metadata['sample_rate'])
    s1 = int(t_win[1] * ephys_metadata['sample_rate'])
    flash_window = np.arange(s0, s1)

    flash_responses = get_flash_response(recording, flash_onset_samples, flash_window, ephys_metadata, f_range, ds)
    flash_responses = flash_responses.mean(0)

    z_responses = flash_responses - flash_responses.mean(0)[None,:]
    z_responses = z_responses / z_responses.std(0)[None,:]

    # get probe geometry
    probe_spacing = ephys_metadata['probe_geometry_um'][1][1]-ephys_metadata['probe_geometry_um'][0][1]
        
    return {
        'responses': flash_responses,
        'z_responses': z_responses,
        'window': flash_window / ds,
        'sample_rate': ephys_metadata['sample_rate'] / ds,
        'probe_spacing': probe_spacing,
        'probe_geometry': np.array(ephys_metadata['probe_geometry_um'])
    }


def plot_triggered_response(response_dict, mode='raw', ax=None, average=False):
    responses = response_dict['responses']
    z_responses = response_dict['z_responses']
    window = response_dict['window'] / response_dict['sample_rate'] * 1000
    sample_rate = response_dict['sample_rate']

    if mode == 'raw':
        responses = responses
    elif mode == 'zscore':
        responses = z_responses
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if average:
        responses = responses[:,:responses.shape[1]//2] + responses[:,responses.shape[1]//2:]

    if ax is None:
        ax = plt.gca()

    dx = response_dict['probe_spacing']
    ax.imshow(responses.T, aspect='auto', origin='lower', interpolation='none', extent=[window[0], window[-1], 0, responses.shape[1]*dx])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Channel')
    return ax


#%%
fig_dir = Path('/home/jake/repos/DataYatesV1/jake/multidataset_ddp/laminar_figures')
for ii, session in enumerate(sessions):
    if ii < 4:
        continue
    try:

        f_range=[100, 100]
        ds=10

        flash_onsets = get_csd_flashes(session.exp)
        saccade_onsets = get_saccades_during_protocol(session, 'BackImage')

        recording = session.processed_recording

        flash_responses = get_triggered_response(session, recording, flash_onsets, [0, .1], f_range, ds)
        saccade_responses = get_triggered_response(session, recording, saccade_onsets, [-.1, .25], f_range, ds)
        

        fig, axs = plt.subplots(1, 4, figsize=(10, 5), sharey=True)
        plot_triggered_response(flash_responses, mode='raw', ax=axs[0], average=True)
        axs[0].set_title('Flash responses (raw)')

        plot_triggered_response(saccade_responses, mode='raw', ax=axs[1], average=True)
        axs[1].set_title('Saccade responses (raw)')
        plot_triggered_response(flash_responses, mode='zscore', ax=axs[2], average=True)
        axs[2].set_title('Flash responses (zscore)')
        plot_triggered_response(saccade_responses, mode='zscore', ax=axs[3], average=True)
        axs[3].set_title('Saccade responses (zscore)')

        for ax in axs:
            ax.axhline(32*35, color='k', linestyle='--')    
            ax.axvline(0, color='k', linestyle='--')  
        
        axs[0].set_yticks(np.arange(0, 64*35, 100))
        
        

        plt.suptitle(session.sess_dir.name.replace('_', ' '))
        plt.tight_layout()
        plt.savefig(fig_dir / f"{session.sess_dir.name}.png")
        plt.close(fig)

        print(f"Processed {session.sess_dir.name}")

    except Exception as e:
        print(f"Error processing session {session}: {str(e)}")  




# %%
for session in sessions:
    if session.sess_dir.name == 'Allen_2022-06-01':
        continue
    f_range=[100, 100]
    ds=10

    flash_onsets = get_csd_flashes(session.exp)
    # flash_onsets = get_saccades_during_protocol(session, 'Grating')
    saccade_onsets = get_saccades_during_protocol(session, 'BackImage')

    recording = session.recording

    flash_responses = get_triggered_response(session, recording, flash_onsets, [0, .1], f_range, ds)
    # flash_responses = get_triggered_response(session, session.recording, saccade_onsets, [0, .1], f_range, ds)
    saccade_responses = get_triggered_response(session, session.recording, saccade_onsets, [0, .1], f_range, ds)

    fig, axs = plt.subplots(1, 4, figsize=(10, 5), sharey=True)
    plot_triggered_response(flash_responses, mode='raw', ax=axs[0], average=True)
    axs[0].set_title('Flash responses (raw)')

    plot_triggered_response(saccade_responses, mode='raw', ax=axs[1], average=True)
    axs[1].set_title('Saccade responses (raw)')
    plot_triggered_response(flash_responses, mode='zscore', ax=axs[2], average=True)
    axs[2].set_title('Flash responses (zscore)')
    plot_triggered_response(saccade_responses, mode='zscore', ax=axs[3], average=True)
    axs[3].set_title('Saccade responses (zscore)')

    for ax in axs:
        ax.axhline(32*flash_responses['probe_spacing'], color='k', linestyle='--')
        ax.axvline(0, color='k', linestyle='--')  

    axs[0].set_yticks(np.arange(0, 32*flash_responses['probe_spacing'], 100))

    plt.suptitle(session.sess_dir.name.replace('_', ' '))
    plt.tight_layout()
    plt.savefig(fig_dir / f"{session.sess_dir.name}.png")
    # plt.close(fig)
    plt.show()

    print(f"Processed {session.sess_dir.name}")
# %%
saccades = detect_saccades(session)
# %%
saccades[0].start_time

# %%
