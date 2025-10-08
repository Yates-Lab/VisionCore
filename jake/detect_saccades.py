import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy import signal

def expand_binary(sig:np.ndarray, rad:int, pos:bool=True, axis:int=0):
    """
    Expand a binary signal by a certain radius.
    """

    n_dims = len(sig.shape)
    assert sig.dtype == bool, 'Input must be a binary array'
    assert axis < n_dims, f'axis must be valid in range [0-{n_dims-1}] for input array'

    if not pos:
        sig = ~sig
    sig = sig.astype(int)

    from scipy.signal import convolve
    k_shape = [rad*2+1 if iD == axis else 1 for iD in range(n_dims)]
    kernel = np.ones(k_shape, dtype=int)

    sig_conv = convolve(sig, kernel, mode='same')
    sig_exp = sig_conv >= 1

    if not pos:
        sig_exp = ~sig_exp

    return sig_exp

def butter_lp_filt(data, lp_cut, lp_ord, fs):
    b, a = signal.butter(lp_ord//2, lp_cut, 'low', fs = fs)
    return signal.filtfilt(b, a, data, axis=0)

def savgol_filt(data, window, order):
    return signal.savgol_filter(data, window, order, axis=0)

# Define the asymmetric Gaussian function
def asymmetric_gaussian(x, mu, A, sigma1, sigma2, b):
    g = np.where(
        x < mu,
        np.exp(-(x-mu)**2 / (2 * sigma1**2)),
        np.exp(-(x-mu)**2 / (2 * sigma2**2))
    )
    return A * g + b

# Fast fitting for one row
def fit_asymmetric_gaussian(x, y):

    # Initial guesses for [mu, A, sigma1, sigma2]
    mu_0 = x[np.argmax(y)]
    A_0 = y.max()
    sigma1_0 = sigma2_0 = np.sqrt(np.mean((x - mu_0)**2*y) / y.sum())
    b0 = (y[0] + y[-1]) / 2
    p0 = [mu_0, A_0, sigma1_0, sigma2_0, b0]

    # Fit the function
    popt, _ = curve_fit(asymmetric_gaussian, x, y, p0=p0, maxfev=5000)
    return popt  # Returns [mu, A, sigma1, sigma2]


def find_saccade_start_stop(mu, A, sigma1, sigma2, b, threshold=0.05):
    """
    Find the start and stop times of the saccade from the asymmetric Gaussian.

    Parameters:
    A : float
        Amplitude of the Gaussian (height).
    sigma1 : float
        Standard deviation for the left side of the peak.
    sigma2 : float
        Standard deviation for the right side of the peak.
    mu : float
        Mean or center of the Gaussian.
    threshold : float, optional
        Fraction of the maximum amplitude (default is 5%, or 0.05).

    Returns:
    start, stop : float
        Start and stop times (indices) of the saccade.
    """
    # Calculate the natural log of the threshold ratio
    log_term = np.log(threshold)

    # Ensure we are calculating the sqrt of a positive number
    start_offset = np.sqrt(-2 * sigma1**2 * log_term)
    stop_offset = np.sqrt(-2 * sigma2**2 * log_term)

    # Calculate start and stop times
    start = mu - start_offset
    stop = mu + stop_offset

    return start, stop

@dataclass
class AGSaccade:
    start_idx: int
    end_idx: int
    mu: float
    A: float
    sigma1: float
    sigma2: float
    b: float
    start_time: float
    end_time: float
    start_x: float
    start_y: float
    end_x: float
    end_y: float

    @property
    def slice(self):
        return slice(self.start_idx, self.end_idx)

    def __len__(self):
        return self.end_idx - self.start_idx

    @property
    def duration(self):
        return self.end_time - self.start_time

    @property
    def amplitude(self):
        return np.sqrt((self.end_x - self.start_x)**2 + (self.end_y - self.start_y)**2)

    @property
    def direction(self):
        return np.arctan2(self.end_y - self.start_y, self.end_x - self.start_x) * 180 / np.pi

    @property
    def velocity(self):
        return self.A

    def to_dict(self):
        return {
            'start_idx': int(self.start_idx),
            'end_idx': int(self.end_idx),
            'mu': float(self.mu),
            'A': float(self.A),
            'sigma1': float(self.sigma1),
            'sigma2': float(self.sigma2),
            'b': float(self.b),
            'start_time': float(self.start_time),
            'end_time': float(self.end_time),
            'start_x': float(self.start_x),
            'start_y': float(self.start_y),
            'end_x': float(self.end_x),
            'end_y': float(self.end_y),
        }

    def basis(self, x):
        return asymmetric_gaussian(x, self.mu, self.A, self.sigma1, self.sigma2, self.b)

    def __repr__(self):
        start = f'{self.start_time:.3f}' if self.start_time else 'None'
        end = f'{self.end_time:.3f}' if self.end_time else 'None'
        amp = f'{self.amplitude:.3f}' if self.amplitude else 'None'
        dir = f'{self.direction:.3f}' if self.direction else 'None'
        vel = f'{self.velocity:.3f}' if self.velocity else 'None'
        return f'Saccade: {start} - {end}, {amp} degrees, {dir} degrees, {vel} degrees/s'

def detect_saccades(sess):
    """
    Detect saccades from a session object.

    Parameters:
    sess : session object
        Session object containing exp and dpi data

    Returns:
    list : List of AGSaccade objects
    """

    # Check if saccades file already exists
    save_dir = sess.sess_dir / 'saccades'
    # if not exist, make direectory
    save_dir.mkdir(exist_ok=True)
    
    saccades_file = save_dir / 'saccades.json'

    if saccades_file.exists():
        print(f"Loading existing saccades from {saccades_file}")
        with open(saccades_file, 'r') as f:
            saccades_data = json.load(f)

        # Convert back to AGSaccade objects
        saccades = []
        for s_dict in saccades_data:
            saccades.append(AGSaccade(**s_dict))
        return saccades

    print("Computing saccades...")

    # Extract data from session
    exp = sess.exp
    dpi = sess.dpi

    ppd = exp['S']['pixPerDeg']
    center_pix = exp['S']['centerPix'][::-1]
    t_dpi = dpi['t_ephys'].values
    dpi_valid = dpi['valid'].values
    dpi_pix = dpi[['dpi_i','dpi_j']].to_numpy()
    dpi_deg = np.fliplr(dpi_pix - center_pix) / ppd * np.array([1, -1])

    # Output valid epochs
    n_samples = len(dpi_valid)
    dt = np.diff(t_dpi, append=np.inf)
    dxy = np.diff(dpi_deg, axis=0, append=0) / dt[:,None]
    speed = np.hypot(*dxy.T)
    max_speed = 5000 # deg/s Anything higher than this is nonsensical
    dpi_valid &= speed < max_speed
    frame_rate = 1/np.median(np.diff(t_dpi))
    print(f'Frame rate: {frame_rate:.2f} Hz')
    # check if nan
    if np.isnan(frame_rate):
        print('Frame rate is nan. using nanmedian, but there may be issues')
        frame_rate = 1/np.nanmedian(np.diff(t_dpi))

    expand_radius = .1 # expand invalid by this amount
    expand_radius_samps = int(expand_radius * frame_rate)
    dpi_valid = expand_binary(dpi_valid, expand_radius_samps, pos=False, axis=0)

    # A valid block is a contiguous segment
    # that has no invalid samples and
    # does not skip more than max_skip_samples
    # and has a duration of at least min_duration

    max_skip_samples = 2.5 # maximum number of samples that can be skipped
    max_skip_time = max_skip_samples / frame_rate # maximum time that can be skipped
    min_duration = 0.5 # minimum duration of a valid epoch

    valid_epochs = []

    start_idx = None
    for iS in range(n_samples):
        # Start valid epoch
        if start_idx is None and dpi_valid[iS]:
            start_idx = iS
            continue

        if start_idx is not None:
            if not dpi_valid[iS]:
                # End valid epoch because of an invalid sample
                if iS - start_idx > min_duration * frame_rate:
                    valid_epochs.append((start_idx, iS))
                start_idx = None
                continue

            if iS > 0 and dt[iS-1] > max_skip_time:
                # End valid epoch because of a skip
                if iS - start_idx > min_duration * frame_rate:
                    valid_epochs.append((start_idx, iS))
                start_idx = None
                continue

            if iS == n_samples - 1:
                # End valid epoch at the end of the data
                if iS - start_idx > min_duration * frame_rate:
                    valid_epochs.append((start_idx, iS))
                start_idx = None

    valid_epochs = np.array(valid_epochs)

    # Will apply a lowpass filter both before and after speed calculation
    savgol_samples = 11
    savgol_order = 3
    gauss_sigma = 2
    filt = lambda x: savgol_filt(x, savgol_samples, savgol_order)
    dpi_deg_filt = filt(dpi_deg)
    speed_filt = np.hypot(*np.diff(dpi_deg_filt, append=0,axis=0).T) / np.diff(t_dpi, append=np.inf)
    speed_filt = gaussian_filter1d(speed_filt, gauss_sigma, axis=0)

    speed_smooth = speed_filt
    valid_samples = np.zeros_like(t_dpi, dtype=bool)
    for s, e in valid_epochs:
        valid_samples[s:e] = True

    med_speed = np.median(speed_smooth[valid_samples])
    mad_speed = np.median(np.abs(speed_smooth[valid_samples] - med_speed))
    mad_multiplier = 21

    print(f'median speed: {med_speed}, mad: {mad_speed}, threshold: {med_speed + mad_multiplier * mad_speed}')
    saccade_threshold = med_speed + mad_multiplier * mad_speed

    sacc_min_dist = .03 # s

    # Detect peaks by finding the local maxima of the smoothed speed
    speed_thresh = (speed_smooth-saccade_threshold) * (speed_smooth > saccade_threshold).astype(speed_smooth.dtype)
    speed_thresh[~valid_samples] = 0
    speed_peak_signs = np.diff(np.sign(np.diff(speed_thresh)))
    speed_peaks = np.where(speed_peak_signs == -2)[0] + 1

    # remove peaks that are too close together
    # always removing the smaller of all peaks within the minimum distance
    saccade_peaks = []
    min_sample_dist = sacc_min_dist * frame_rate
    for i in range(len(speed_peaks)-1):
        peak_distances = np.abs(speed_peaks - speed_peaks[i])
        nearby_peaks = speed_peaks[peak_distances < min_sample_dist]
        this_peak = np.where(nearby_peaks == speed_peaks[i])[0][0]
        nearby_amps = speed_smooth[nearby_peaks]
        max_peak = np.argmax(nearby_amps)
        if this_peak == max_peak:
            saccade_peaks.append(speed_peaks[i])
    saccade_peaks = np.array(saccade_peaks)

    print(f'found {len(saccade_peaks)} saccade peaks')

    # march out from each peak to find the start and end of the saccade
    # then calculate the magnitude and direction of the saccade

    max_dur = .15 # s
    max_dur_samps = int(max_dur * frame_rate)
    sacc_pre_window = .01 # s
    pre_samps = int(sacc_pre_window * frame_rate)
    sacc_post_window = .01 # s
    post_samps = int(sacc_post_window * frame_rate)

    n_invalid_saccades = 0
    saccade_epochs = []
    for p in saccade_peaks:
        sacc_start = p
        sacc_end = p

        # find the start of the saccade
        stop = np.max((p-max_dur_samps // 2, 0))
        for start in range(p, stop-1, -1):
            if speed_smooth[start] < saccade_threshold or start == stop:
                sacc_start = np.max((start - pre_samps, 0))
                break

        # find the end of the saccade
        stop = np.min((p+max_dur_samps // 2, len(speed_smooth)-1))
        for end in range(p, stop+1):
            if speed_smooth[end] < saccade_threshold or end == stop:
                sacc_end = np.min((end + post_samps, len(speed_smooth)-1))
                break

        valid_win = dpi_valid[sacc_start:sacc_end]
        if not np.all(valid_win):
            n_invalid_saccades += 1
            valid_samples[sacc_start:sacc_end] = False
        else:
            saccade_epochs.append((sacc_start, sacc_end))

    print(f'found {n_invalid_saccades + len(saccade_epochs)} raw saccades')
    print(f'removed {n_invalid_saccades} saccades with invalid samples')

    # remove overlapping epochs
    overlap_threshold = .2 # fraction of saccade length
    merges = []
    for iS in range(0, len(saccade_epochs)-1):
        for jS in range(iS+1, len(saccade_epochs)):
            start1, end1 = saccade_epochs[iS]
            start2, end2 = saccade_epochs[jS]

            sacc_len = np.min((end1-start1, end2-start2))
            overlap = np.min((end1, end2)) - np.max((start1, start2))
            if overlap >= int(overlap_threshold * sacc_len+.5):
                merges.append((iS, jS))
            else:
                # since the saccades are ordered,
                # once the saccades no longer overlap we know
                # that all subsequent saccades will not overlap
                # so we can break out of the loop
                # if the saccades are no longer overlapping
                break

    print(f'merging {len(merges)} overlapping saccades')

    for iM in range(len(merges)):
        iS, jS = merges[iM]
        start1, end1 = saccade_epochs[iS]
        start2, end2 = saccade_epochs[jS]
        saccade_epochs[iS] = (np.min((start1, start2)), np.max((end1, end2)))
        saccade_epochs.pop(jS)

        # update merges now that saccade_epochs has changed
        for jM in range(iM+1, len(merges)):
            if merges[jM][0] > jS:
                merges[jM] = (merges[jM][0]-1, merges[jM][1])
            if merges[jM][0] == jS:
                merges[jM] = (iS, merges[jM][1])
            if merges[jM][1] > jS:
                merges[jM] = (merges[jM][0], merges[jM][1]-1)
            if merges[jM][1] == jS:
                merges[jM] = (merges[jM][0], iS)


    print(f'found {len(saccade_epochs)} total saccade epochs')

    # todo split saccade epochs if there are multiple peaks in an epoch (i.e. speed is multimodal)
    # (detect windows with multiple peaks with at least 20 deg/s trough between them)

    min_trough = .7 # % of the lower peak amplitude
    min_dist = .01 # minimum distance between peaks and troughs in seconds
    max_peak_ratio = 3 # maximum ratio of peak amplitudes
    multiple_peak_saccades = []
    saccade_peaks_tuple = []
    iS = 0
    while iS < len(saccade_epochs):
       s, e = saccade_epochs[iS]
       speed_win = speed_smooth[s:e]
       speed_signs = np.diff(np.sign(np.diff(speed_win)))
       peaks = np.where(speed_signs == -2)[0] + 1
       peak_amps = speed_win[peaks]
       if len(peaks) < 2:
           iS += 1
           continue

       # find the two largest peaks and the trough between them
       i1, i2 = np.argsort(peak_amps)[-2:]
       lower_peak_amp = np.min((peak_amps[i1], peak_amps[i2]))
       higher_peak_amp = np.max((peak_amps[i1], peak_amps[i2]))
       peak_ratio = higher_peak_amp / lower_peak_amp
       if peak_ratio > max_peak_ratio:
           iS += 1
           continue
       peak1, peak2 = peaks[np.min((i1, i2))], peaks[np.max((i1, i2))]
       peak_win = speed_win[peak1:peak2]
       interpeak_trough = np.min(peak_win)
       interpeak_trough_idx = np.argmin(peak_win) + peak1
       trough_dist = np.min((np.abs(peak1-interpeak_trough_idx), np.abs(peak2-interpeak_trough_idx))) / frame_rate
       if trough_dist < min_dist:
           iS += 1
           continue
       split_idx = s + interpeak_trough_idx
       if interpeak_trough < min_trough * lower_peak_amp:
           multiple_peak_saccades.append((s,e))
           saccade_peaks_tuple.append((peak1, peak2, interpeak_trough_idx))
           sacc1 = (s, split_idx)
           sacc2 = (s, split_idx)
           saccade_epochs.pop(iS)
           saccade_epochs.insert(iS, sacc1)
           saccade_epochs.insert(iS+1, sacc2)
           iS += 2
       else:
           iS += 1

    print(f'split {len(multiple_peak_saccades)} saccades with multiple peaks')

    saccades = []
    for s, e in saccade_epochs:
        speed_win = speed_smooth[s:e]
        t_win = t_dpi[s:e]
        try:
            popt = fit_asymmetric_gaussian(t_win, speed_win)
            t_start, t_stop = find_saccade_start_stop(*popt, threshold=.1)
            saccades.append(AGSaccade(
                s, e,
                popt[0], popt[1], popt[2], popt[3], popt[4],
                t_start, t_stop,
                dpi_deg[s,0], dpi_deg[s,1],
                dpi_deg[e,0], dpi_deg[e,1]
            ))
        except RuntimeError as e:
            print(f'Failed to fit saccade {s} - {e}')
            continue

    print(f'found {len(saccades)} saccades')

    # Save saccades
    try:
        save_dir.mkdir(exist_ok=True)
        with open(saccades_file, 'w') as f:
            json.dump([s.to_dict() for s in saccades], f)
        print(f"Saved saccades to {saccades_file}")
    except PermissionError:
        print(f"Warning: Could not save saccades to {saccades_file} due to permission error")
    except Exception as e:
        print(f"Warning: Could not save saccades to {saccades_file}: {e}")

    return saccades


# Example usage (for testing):
if __name__ == "__main__":
    from DataYatesV1 import get_session

    subject = 'Allen'
    date = '2022-04-13'
    sess = get_session(subject, date)

    saccades = detect_saccades(sess)
    print(f"Detected {len(saccades)} saccades")

    # Print first few saccades
    for i, sacc in enumerate(saccades[:5]):
        print(f"{i+1}: {sacc}")