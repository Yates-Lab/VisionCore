#%%

from inspect import getfile
from tejas.metrics.gratings import get_gratings_for_dataset
from tejas.metrics.fixrsvp import get_fixrsvp_for_dataset,plot_fixrsvp_psth, plot_fixrsvp_spike_raster
from tejas.metrics.grating_utils import plot_phase_tuning_with_fit, plot_ori_tuning
import matplotlib as mpl
import matplotlib.pyplot as plt
from tejas.metrics.gaborium import plot_unit_sta_ste
from tejas.metrics.mcfarland import get_mcfarland_analysis_for_dataset, plot_mcfarland_analysis_for_unit, check_if_valid_mcfarland_analysis
from DataYatesV1 import  get_complete_sessions, get_session
from tejas.metrics.plot_variables import plot_variable1_vs_variable2, plot_variable1
from tejas.metrics.main_unit_panel import get_unit_info_panel
from tejas.metrics.plot_variables import plot_simple_complex_mcfarland_population

# Configure matplotlib for PDF output with embedded fonts
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
#%%
sessions_to_exclude = ['Allen_2022-06-10', 'Allen_2022-08-05']
all_sessions = [sess for sess in get_complete_sessions() if sess.name not in sessions_to_exclude]

# %%

#available variables: modulation_index, tau_data_10_bins, alpha_data_10_bins, tau_ln_10_bins, alpha_ln_10_bins, tau_energy_10_bins, alpha_energy_10_bins, tau_data_37_bins, alpha_data_37_bins, tau_ln_37_bins, alpha_ln_37_bins, tau_energy_37_bins, alpha_energy_37_bins, ln_bps, energy_bps, linear_index, area_gaussian, area_contour, sqrt_area_gaussian_deg, sqrt_area_contour_deg, area_gaussian_deg2, area_contour_deg2, length_gaussian
# #%%
%matplotlib widget
# plot_variable1_vs_variable2('sqrt_area_contour_deg', 'tau_data_10_bins', all_sessions, cache = True, interactive = True, cache_all = True, qc=False, outlier_method = 'mad', outlier_threshold = 7)
# plot_variable1_vs_variable2('sqrt_area_contour_deg', 'tau_data_10_bins', all_sessions, cache = True, interactive = True, cache_all = True, qc=True, outlier_method = 'mad', outlier_threshold = 7)

# plot_variable1_vs_variable2('modulation_index', 'tau_data_10_bins', all_sessions, cache = True, interactive = True, cache_all = True, qc=False)
#%%
x_variable = 'sqrt_area_contour_deg'
y_variable = 'tau_data_37_bins'
# modes_to_highlight = ['furthest_from_origin', 'closest_to_origin', 'closest_to_median']
modes_to_highlight = None
_, all_points, highlighted_points = plot_variable1_vs_variable2(x_variable, 
y_variable, all_sessions, cache = True, 
interactive = True, cache_all = True, 
qc=True, outlier_method = 'mad', outlier_threshold = 7,
points_to_highlight = modes_to_highlight, num_points = 10)

#%%
#%%
_, all_values = plot_variable1('modulation_index', all_sessions, cache_all=True, qc=True)
#%%
plot_simple_complex_mcfarland_population(all_values, 50)
# %%
