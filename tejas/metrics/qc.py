#%%
import numpy as np
from DataYatesV1 import get_session, get_complete_sessions
from pathlib import Path

from tejas.metrics.fixrsvp import get_fixrsvp_for_dataset
from tejas.metrics.gaborium import get_rf_contour_metrics

#%%

def get_information_for_qc(date, subject, cache = False):
    cache_dir = Path(f'/mnt/ssd/YatesMarmoV1/metrics/qc/{subject}_{date}')
    cache_dir.mkdir(parents=True, exist_ok=True)
    information_for_qc_path = cache_dir / f'information_for_qc.npz'
    if cache and information_for_qc_path.exists():
        information_for_qc_npz = np.load(information_for_qc_path, allow_pickle=True)
        information_for_qc = information_for_qc_npz['information_for_qc']
        # If it's a 0-d array (pickled object), extract it with .item()
        if isinstance(information_for_qc, np.ndarray) and information_for_qc.ndim == 0:
            information_for_qc = information_for_qc.item()
        information_for_qc_npz.close()
        return information_for_qc

    sess = get_session(subject, date)
    cids = np.unique(sess.ks_results.spike_clusters)
    #contamination threshold
    inclusion_npz = np.load(sess.sess_dir / 'inclusion' / 'inclusion.npz')
    # Extract all arrays from NpzFile into a regular dictionary to avoid pickling file handles
    inclusion = {key: inclusion_npz[key] for key in inclusion_npz.keys()}
    inclusion_npz.close()  # Close the file handle


    #min firing rate in fixrsvp
    fixrsvp_info = get_fixrsvp_for_dataset(date, subject, cache = cache)


    #visually responsive units from gaborium
    rf_contour_metrics = get_rf_contour_metrics(date, subject)

    information_for_qc = {
        'cids': cids,
        'inclusion': inclusion,
        'fixrsvp_info': fixrsvp_info,
        'rf_contour_metrics': rf_contour_metrics,
    }
    #save qc results
    np.savez(information_for_qc_path, information_for_qc=information_for_qc)
    return information_for_qc

def get_qc_units_from_information_dict(information_for_qc, contamination_threshold = 60, firing_rate_threshold = 2.5, snr_value_threshold = 6):
    cids = information_for_qc['cids']
    inclusion = information_for_qc['inclusion']
    cids_contamination_qc = cids[inclusion['min_contam_pct'] < contamination_threshold]
    


    #min firing rate in fixrsvp
    fixrsvp_info = information_for_qc['fixrsvp_info']
    firing_rate_units = np.nanmean(fixrsvp_info['robs_unprocessed'], axis=0) * 240
    firing_rate_units_qc = cids[firing_rate_units > firing_rate_threshold]

    #visually responsive units from gaborium
    rf_contour_metrics = information_for_qc['rf_contour_metrics']
    assert len(rf_contour_metrics) == len(cids), f'{len(rf_contour_metrics)} != {len(cids)}'
    cids_snr_value_qc = np.array([uid for uid in rf_contour_metrics.keys() if rf_contour_metrics[uid]['snr_value'] > snr_value_threshold])

    cids_qc = np.intersect1d(cids_contamination_qc, firing_rate_units_qc)
    cids_qc = np.intersect1d(cids_qc, cids_snr_value_qc)
    
    return cids_qc

def get_qc_units_for_session(date, subject, contamination_threshold = 60, firing_rate_threshold = 2.5, snr_value_threshold = 6, cache = False):
    
    
    # sess = get_session(subject, date)
    # cids = np.unique(sess.ks_results.spike_clusters)
    # #contamination threshold
    # inclusion = np.load(sess.sess_dir / 'inclusion' / 'inclusion.npz')
    information_for_qc = get_information_for_qc(date, subject, cache = cache)
    cids_qc = get_qc_units_from_information_dict(information_for_qc, contamination_threshold = contamination_threshold, firing_rate_threshold = firing_rate_threshold, snr_value_threshold = snr_value_threshold)
    

    return cids_qc




# sessions_to_exclude = ['Allen_2022-06-10', 'Allen_2022-08-05']
# all_sessions = [sess for sess in get_complete_sessions() if sess.name not in sessions_to_exclude]
# total_qc_units = 0
# for sess in all_sessions:
#     date = sess.name.split('_')[1]
#     subject = sess.name.split('_')[0]
#     try:    
#         qc_units = get_qc_units_for_session(date, subject, cache = True)
#     except Exception as e:
#         print(f'Error exporting QC units for {sess}: {e}')
#         continue
#     print(f'{len(qc_units)} / {len(np.unique(sess.ks_results.spike_clusters))} units pass')
#     total_qc_units += len(qc_units)
# print(f'Total QC units: {total_qc_units}')

# #%%
# sessions_to_exclude = ['Allen_2022-06-10', 'Allen_2022-08-05', 'Allen_2022-06-01', 'Logan_2019-12-20', 'Logan_2019-12-31', 'Logan_2019-12-23', 'Allen_2022-02-16']
# all_sessions = [sess for sess in get_complete_sessions() if sess.name not in sessions_to_exclude]
# from tqdm import tqdm
# session_to_qc_information = {}
# for sess in tqdm(all_sessions):
#     date = sess.name.split('_')[1]
#     subject = sess.name.split('_')[0]
#     try:
#         qc_units = get_qc_units_for_session(date, subject, cache = True)
#         session_to_qc_information[sess.name] = {

#         }
#     except Exception as e:
#         print(f'Error exporting QC units for {sess}: {e}')
#         continue

# #%%

# sessions_to_exclude = ['Allen_2022-06-10', 'Allen_2022-08-05', 'Allen_2022-06-01', 'Logan_2019-12-20', 'Logan_2019-12-31', 'Logan_2019-12-23', 'Allen_2022-02-16']
# all_sessions = [sess for sess in get_complete_sessions() if sess.name not in sessions_to_exclude]
# from tqdm import tqdm
# session_to_qc_information = {}
# for sess in tqdm(all_sessions):
#     date = sess.name.split('_')[1]
#     subject = sess.name.split('_')[0]
#     try:
#         # qc_units = get_qc_units_for_session(date, subject, cache = True)
#         information_for_qc = get_information_for_qc(date, subject, cache = True)
    
#         session_to_qc_information[sess.name] = information_for_qc
#     except Exception as e:
#         print(f'Error exporting QC units for {sess}: {e}')
#         continue


# #%%
# for sess in tqdm(session_to_qc_information.keys()):
#     cids_qc = get_qc_units_from_information_dict(session_to_qc_information[sess])
# #%%