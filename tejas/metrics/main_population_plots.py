#%%

from inspect import getfile
from tejas.metrics.gratings import get_gratings_for_dataset
from tejas.metrics.fixrsvp import get_fixrsvp_for_dataset,plot_fixrsvp_psth, plot_fixrsvp_spike_raster
from tejas.metrics.grating_utils import plot_phase_tuning_with_fit, plot_ori_tuning
import matplotlib.pyplot as plt
from tejas.metrics.gaborium import plot_unit_sta_ste
from tejas.metrics.mcfarland import get_mcfarland_analysis_for_dataset_old, plot_mcfarland_analysis_for_unit, check_if_valid_mcfarland_analysis
from DataYatesV1 import  get_complete_sessions, get_session
from tejas.metrics.plot_variables import plot_variable1_vs_variable2, plot_variable1
from tejas.metrics.main_unit_panel import get_unit_info_panel
from tejas.metrics.plot_variables import plot_simple_complex_mcfarland_population
#%%
sessions_to_exclude = ['Allen_2022-06-10', 'Allen_2022-08-05']
all_sessions = [sess for sess in get_complete_sessions() if sess.name not in sessions_to_exclude]

# %%

#available variables: modulation_index, alpha, tau, ln_bps, energy_bps, linear_index, area_gaussian, area_contour, sqrt_area_gaussian_deg, sqrt_area_contour_deg, area_gaussian_deg2, area_contour_deg2, length_gaussian
# #%%
%matplotlib widget
# plot_variable1_vs_variable2('sqrt_area_contour_deg', 'tau', all_sessions, cache = True, interactive = True, cache_all = True, qc=False, outlier_method = 'mad', outlier_threshold = 7)
# plot_variable1_vs_variable2('sqrt_area_contour_deg', 'tau', all_sessions, cache = True, interactive = True, cache_all = True, qc=True, outlier_method = 'mad', outlier_threshold = 7)

# plot_variable1_vs_variable2('modulation_index', 'tau', all_sessions, cache = True, interactive = True, cache_all = True, qc=False)
#%%
x_variable = 'modulation_index'
y_variable = 'tau'
_, all_points, highlighted_points = plot_variable1_vs_variable2(x_variable, 
y_variable, all_sessions, cache = True, 
interactive = True, cache_all = True, 
qc=True, outlier_method = 'mad', outlier_threshold = 7,
points_to_highlight = ['further_from_y_axis', 'closest_to_y_axis'], num_points = 50)
#%%

%matplotlib inline
from matplotlib.backends.backend_pdf import PdfPages
mode = 'closest_to_origin'
save_to_one_pdf = True

if save_to_one_pdf:
    with PdfPages(f'{mode}_units.pdf') as pdf:
        for unit in highlighted_points[mode]:
            fig = get_unit_info_panel(unit[4], unit[2], unit[3], cache=True)
            fig.suptitle(f'{mode.replace("_", " ")}: {x_variable} vs {y_variable}', fontsize=16, fontweight='bold')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
else:
    for unit in highlighted_points[mode]:
        fig = get_unit_info_panel(unit[4], unit[2], unit[3], cache=True)
        fig.suptitle(f'{mode.replace("_", " ")}: {x_variable} vs {y_variable}', fontsize=16, fontweight='bold')
        plt.show()

# #%%
# # %matplotlib widget
# plot_variable1_vs_variable2('sqrt_area_contour_deg', 'sqrt_area_gaussian_deg', [get_session('Allen', '2022-04-13')], cache = True, interactive = True, cache_all = True)
# #%%
# %matplotlib widget
# plot_variable1_vs_variable2('sqrt_area_contour_deg', 'tau', 
# all_sessions, cache = True, interactive = True, cache_all = True, 
# outlier_method = 'mad', outlier_threshold = 7)
#%%
_, all_values = plot_variable1('modulation_index', all_sessions, cache_all=True, qc=True)

#%%
%matplotlib inline
from tejas.metrics.main_unit_panel import get_unit_info_panel
#top 10 units by alpha
for unit in all_values[-10:]:
    fig = get_unit_info_panel(unit[3], unit[1], unit[2], cache=True)
    plt.show()

#%%
#%%
_, all_values = plot_variable1('modulation_index', all_sessions, cache_all=True, qc=True)
#%%
plot_simple_complex_mcfarland_population(all_values, 50)
# %%
