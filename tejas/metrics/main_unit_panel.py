#%%

from inspect import getfile
from tejas.metrics.gratings import get_gratings_for_dataset, get_peak_sf_for_dataset
from tejas.metrics.fixrsvp import get_fixrsvp_for_dataset,plot_fixrsvp_psth, plot_fixrsvp_spike_raster, plot_fixrsvp_psth_and_spike_raster
from tejas.metrics.grating_utils import plot_phase_tuning_with_fit, plot_ori_tuning
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import os

# Configure matplotlib for PDF output with embedded fonts
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
from tejas.metrics.gaborium import plot_unit_sta_ste
from tejas.metrics.mcfarland import get_mcfarland_analysis_for_dataset, plot_mcfarland_analysis_for_unit, check_if_valid_mcfarland_analysis
from DataYatesV1 import  get_complete_sessions, get_session
from tejas.metrics.plot_variables import plot_variable1_vs_variable2
import numpy as np
from tejas.metrics.gaborium import get_rf_contour_metrics, get_rf_gaussian_fit_metrics
from pathlib import Path
from tejas.metrics.gaborium import show_stas_for_session_by_metric_order
from tejas.metrics.gaborium import get_unit_sta_ste_dict
#%%

#%%
def get_unit_info_panel_dict(date, subject, cache = True):
    gratings_info = get_gratings_for_dataset(date, subject, cache = cache)
    fixrsvp_info = get_fixrsvp_for_dataset(date, subject, cache = False)
    mcfarland_info = get_mcfarland_analysis_for_dataset(date, subject, cache = cache)
    rf_contour_metrics = get_rf_contour_metrics(date, subject)
    rf_gaussian_fit_metrics = get_rf_gaussian_fit_metrics(date, subject, cache = cache)
    unit_sta_ste_dict = get_unit_sta_ste_dict(date, subject)
    return {
        'gratings_info': gratings_info,
        'fixrsvp_info': fixrsvp_info,
        'mcfarland_info': mcfarland_info,
        'rf_contour_metrics': rf_contour_metrics,
        'rf_gaussian_fit_metrics': rf_gaussian_fit_metrics,
        'unit_sta_ste_dict': unit_sta_ste_dict
    }


def get_unit_info_panel(unit_idx, 
                        date, 
                        subject, 
                        unit_info_panel_dict,
                        cache = True, 
                        save = False, 
                        mcfarland_bins = 37, 
                        show_ln = False, 
                        show_energy = False, 
                        show_fit = True, 
                        show_ln_energy_fit = False, 
                        save_dir = None):
    # gratings_info = get_gratings_for_dataset(date, subject, cache = cache)
    # fixrsvp_info = get_fixrsvp_for_dataset(date, subject, cache = cache)
    # mcfarland_info = get_mcfarland_analysis_for_dataset(date, subject, cache = cache)
    # rf_contour_metrics = get_rf_contour_metrics(date, subject)
    # rf_gaussian_fit_metrics = get_rf_gaussian_fit_metrics(date, subject, cache = cache)
    # unit_info_panel_dict = get_unit_info_panel_dict(date, subject, cache = cache)
    gratings_info = unit_info_panel_dict['gratings_info']
    fixrsvp_info = unit_info_panel_dict['fixrsvp_info']
    mcfarland_info = unit_info_panel_dict['mcfarland_info']
    rf_contour_metrics = unit_info_panel_dict['rf_contour_metrics']
    rf_gaussian_fit_metrics = unit_info_panel_dict['rf_gaussian_fit_metrics']
    unit_sta_ste_dict = unit_info_panel_dict['unit_sta_ste_dict']



    

    # Create figure with subplot grid (3 rows x 2 cols)
    # fig = plt.figure(figsize=(15, 12))
    fig = plt.figure(figsize=(10, 8), dpi=50)
    gs = GridSpec(3, 2, figure=fig, hspace=0, wspace=0.3, top=0.95, bottom=0, 
                height_ratios=[1.2, 1, 1.1])

    # Top row: STA/STE (spans both columns)
    ax_sta = fig.add_subplot(gs[0, :])
    plot_unit_sta_ste(subject, date, unit_idx, unit_sta_ste_dict, contour_metrics=rf_contour_metrics, gaussian_fit_metrics=rf_gaussian_fit_metrics, sampling_rate=240, ax=ax_sta, show_ln_energy_fit=show_ln_energy_fit)

    # Middle row: Phase Tuning (left) and Orientation Tuning (right)
    ax_phase = fig.add_subplot(gs[1, 0])
    plot_phase_tuning_with_fit(gratings_info, unit_idx, ax=ax_phase)

    ax_ori = fig.add_subplot(gs[1, 1])
    plot_ori_tuning(gratings_info, unit_idx, ax=ax_ori)

    # Bottom row: FixRSVP PSTH (left) and McFarland Analysis (right)
    ax_fixrsvp = fig.add_subplot(gs[2, 0])
    plot_fixrsvp_psth_and_spike_raster(fixrsvp_info, unit_idx, ax=ax_fixrsvp)

    ax_mcfarland = fig.add_subplot(gs[2, 1])
    plot_mcfarland_analysis_for_unit(unit_idx, mcfarland_info, contour_metrics=rf_contour_metrics, gaussian_fit_metrics=None, ax=ax_mcfarland, bins=mcfarland_bins, show_ln=show_ln, show_energy=show_energy, show_fit=show_fit)
    # plot_mcfarland_analysis_for_unit(unit_idx, mcfarland_info, contour_metrics=rf_contour_metrics, gaussian_fit_metrics=rf_gaussian_fit_metrics, ax=ax_mcfarland, bins=mcfarland_bins, show_ln=show_ln, show_energy=show_energy, show_fit=show_fit)

    # Manually adjust positions to reduce gap between top and middle rows, and add space between middle and bottom rows
    # Get current positions
    pos_sta = ax_sta.get_position()
    pos_phase = ax_phase.get_position()
    pos_ori = ax_ori.get_position()
    pos_fixrsvp = ax_fixrsvp.get_position()
    pos_mcfarland = ax_mcfarland.get_position()

    # Move middle row plots upward to reduce gap with STA plot
    # When model fits are shown, the top panel has more rows, so don't reduce the gap as much
    middle_row_height = pos_phase.height
    space_to_reduce = -0.05 if show_ln_energy_fit else 0.03  # Add space when fits shown, reduce when not

    ax_phase.set_position([pos_phase.x0, pos_phase.y0 + space_to_reduce, pos_phase.width, middle_row_height])
    ax_ori.set_position([pos_ori.x0, pos_ori.y0 + space_to_reduce, pos_ori.width, middle_row_height])

    # Move bottom row plots down by increasing their bottom position
    bottom_row_height = pos_fixrsvp.height
    space_to_add = 0.05  # Additional space between middle and bottom rows

    ax_fixrsvp.set_position([pos_fixrsvp.x0, pos_fixrsvp.y0 - space_to_add, pos_fixrsvp.width, bottom_row_height])
    ax_mcfarland.set_position([pos_mcfarland.x0, pos_mcfarland.y0 - space_to_add, pos_mcfarland.width, bottom_row_height])
    if save:    
        # Save figure as PDF
        output_dir = Path('example_cells')
        output_dir.mkdir(exist_ok=True)
        fig.savefig(output_dir / f'Unit_{unit_idx}_{subject}_{date}.pdf', format='pdf', bbox_inches='tight', dpi=300)

    # print('square root of contour area: ', rf_contour_metrics[unit_idx]['sqrt_area_contour_deg'])
    # print('contour area: ', rf_contour_metrics[unit_idx]['area_contour'])
    # print('1-alpha: ', 1-mcfarland_info[37][unit_idx]['mcfarland_robs_out']['alpha'])
    # print('alpha: ', mcfarland_info[37][unit_idx]['mcfarland_robs_out']['alpha'])
    # print('1/tau: ', 1/mcfarland_info[37][unit_idx]['mcfarland_robs_out']['tau'])
    # print('tau: ', mcfarland_info[37][unit_idx]['mcfarland_robs_out']['tau'])
    print
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_dir / f'Unit_{unit_idx}_{subject}_{date}.pdf', format='pdf', bbox_inches='tight', dpi=300)
    return fig

if __name__ == '__main__':
    # %matplotlib inline
    date = '2022-04-08'
    subject = 'Allen'
    unit_idx = 16

    date = '2022-04-13'
    subject = 'Allen'
    unit_idx = 126

    date = '2022-04-15'
    subject = 'Allen'
    unit_idx = 122

    date = '2022-03-04'
    subject = 'Allen'
    unit_idx = 122

    gratings_info = get_gratings_for_dataset(date, subject, cache = True)
    plot_ori_tuning(gratings_info, unit_idx)

    # date = '2022-04-08'
    # subject = 'Allen'
    # unit_idx = 99

    # date = '2022-04-06'
    # subject = 'Allen'
    # unit_idx = 95

    cache = True

    # #throw out 118, 92, 65, 88 for Logan_2020-01-10'

    # # #steep fall off: 49, 126 # 126

    # #flatter: 57, 59, 60, 64, 72, 83, 96, 115 #96
    unit_info_panel_dict = get_unit_info_panel_dict(date, subject, cache = cache)
    _ = get_unit_info_panel(unit_idx, date, subject, unit_info_panel_dict, cache, 
    mcfarland_bins = 37, 
    show_ln = False, show_energy = True, show_fit = False,
    show_ln_energy_fit = True)
    # save_dir = Path('example_cells'))

    # # plot_fixrsvp_psth(fixrsvp_info, unit_idx)
    # # plt.show()

    # # plot_fixrsvp_spike_raster(fixrsvp_info, unit_idx)
    # # plt.show()

    # #%%
    # from tejas.metrics.qc import get_qc_units_for_session
    # _, l = get_qc_units_for_session(date, subject)

    #%%
    #%%
    # from DataYatesV1.utils.data import get_gaborium_sta_ste
    # from DataYatesV1.utils.plotting import plot_stas
    # sess = get_session(subject, date)

    # stas, stes = get_gaborium_sta_ste(sess, 20)
    # plt.imshow(np.abs(stes[unit_idx, 13]))
    # plt.colorbar()

    # NC = stes.shape[0]
    # sd = np.std(stes, axis=(2,3))
    # absmax = np.max(np.abs(stes), axis=(2,3))
    # sx = int(np.sqrt(NC))
    # sy = int(np.ceil(np.sqrt(NC)))
    # fig = plt.figure(figsize=(sx*3, sy*3))
    # for cc in range(NC):
    #     plt.subplot(sx, sy, cc+1)
    #     plt.plot(sd[cc])
    #     plt.axis('off')
    #     plt.gca().twinx()
    #     plt.plot(absmax[cc], 'r--')
    #     plt.axis('off')
    #     plt.title(f'Unit {cc}')
    # %%
