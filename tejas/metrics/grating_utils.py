import numpy as np
import matplotlib.pyplot as plt
def plot_phase_tuning_with_fit(gratings_results, unit_idx, ax=None):
    """
    Plot phase tuning curve with sinusoidal fit overlay.

    Parameters:
    -----------
    fit_result : dict
        Results from sine fitting function
    phase_data : array
        Phase values
    spike_data : array
        Corresponding spike counts
    ax : matplotlib axis, optional
        Axis to plot on
    title : str
        Plot title
    """
    fit_result = gratings_results['sine_fit_results'][unit_idx]
    dt = gratings_results['dt']
    title = f'Phase Tuning for Unit {unit_idx}'
    if fit_result is None:
        if ax is not None:
            ax.text(0.5, 0.5, 'Insufficient data\nfor fitting',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
        return

    # Extract fit parameters
    amplitude = fit_result['amplitude']
    amplitude_se = fit_result['amplitude_se']
    phase_offset = fit_result['phase_offset_rad']
    baseline = fit_result['C']
    modulation_index = fit_result['modulation_index']
    modulation_index_se = fit_result['modulation_index_se']

    # Check for valid fit
    if (np.isnan(modulation_index) or np.isnan(modulation_index_se) or
        np.isnan(amplitude) or np.isnan(amplitude_se) or np.isnan(phase_offset)):
        if ax is not None:
            ax.text(0.5, 0.5, 'Fit failed\n(NaN values)',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
        return

    # Generate smooth curve for visualization
    smooth_phases = np.linspace(0, 2*np.pi, 100)
    smooth_fit = amplitude * np.sin(smooth_phases + phase_offset) + baseline
    smooth_fit_upper = (amplitude + amplitude_se) * np.sin(smooth_phases + phase_offset) + baseline
    smooth_fit_lower = (amplitude - amplitude_se) * np.sin(smooth_phases + phase_offset) + baseline

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # Plot fitted curve with confidence interval
    ax.plot(np.rad2deg(smooth_phases), smooth_fit / dt, 'red', linewidth=2, label='Fit')
    ax.fill_between(np.rad2deg(smooth_phases), smooth_fit_lower / dt, smooth_fit_upper / dt,
                   color='red', alpha=0.3, label='±1 SE')

    # Set axis properties
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Phase (degrees)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title(f'{title}\nMI: {modulation_index:.3f} ± {modulation_index_se:.3f}')
    ax.legend()

    phase_bin_centers = gratings_results['phase_bins']
    phase_response_corrected = gratings_results['phase_response'][unit_idx]
    phase_response_sem_corrected = gratings_results['phase_response_ste'][unit_idx]
    ax.errorbar(phase_bin_centers, phase_response_corrected,
                        yerr=phase_response_sem_corrected,
                        fmt='o-', capsize=5, label='Data', zorder=10)

def plot_ori_tuning(gratings_results, unit_idx, ax=None):
    ori_tuning = gratings_results['ori_tuning'][unit_idx]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gratings_results['oris'], ori_tuning)
    ax.set_xlabel('Orientation (degrees)')
    # ax.set_ylabel('Firing Rate (Hz)')
    title = f'Orientation Tuning for Unit {unit_idx}'
    ax.set_title(f'{title}')
    ax.legend()