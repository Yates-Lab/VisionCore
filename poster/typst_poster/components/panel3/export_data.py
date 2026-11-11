"""Extract Figure 2A's original trial observations, never values from artwork.
Run with the analysis environment and the original fixrsvp.dset path.
The compact export lets the poster environment calculate annotations offline.
"""
from pathlib import Path
import argparse, ast, hashlib, json, sys
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
sys.path.insert(0,str(ROOT))
from models.config_loader import load_dataset_configs
from models.data.loading import apply_downsampling
from models.data.datasets import DictDataset
from VisionCore.covariance import align_fixrsvp_trials

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);args=p.parse_args()
source=ROOT/'paper/fig2/generate_panel_example.py'
# Read the figure's constants without executing unrelated plot/cache imports.
constants={}
for node in ast.parse(source.read_text()).body:
 if isinstance(node,ast.Assign):
  for target in node.targets:
   if isinstance(target,ast.Name):
    try: constants[target.id]=eval(compile(ast.Expression(node.value),str(source),'eval'),{'__builtins__':{}},constants)
    except (NameError,TypeError,AttributeError): pass
session=constants['SESSION'];trials=[constants['TRIAL_A'],constants['TRIAL_B']];unit=constants['EXAMPLE_UNIT'];n=constants['WINDOW_BINS']
config=ROOT/'experiments/dataset_configs/multi_basic_120_long.yaml'
cfg=next(c for c in load_dataset_configs(str(config)) if c['session']==session)
data=torch.load(args.dataset,map_location='cpu',weights_only=False)
# Match prepare_data's transformations for the four alignment inputs. The
# stimulus/behavior transforms and train split do not alter these covariates.
keys=['robs','eyepos','trial_inds','psth_inds']
d=DictDataset({k:data['covariates'][k] for k in keys},metadata=data.get('metadata',{}))
d=apply_downsampling(d,cfg['sampling']['source_rate']//cfg['sampling']['target_rate'])
d['robs']=d['robs'][:,cfg['cids']]
a,e,v,neuron_mask,meta=align_fixrsvp_trials(d,valid_time_bins=120,min_fix_dur=20,min_total_spikes=0)
j=np.flatnonzero(neuron_mask==unit).item()
counts=a[trials,:n,j];ep=e[trials,:n]
assert v[trials,:n].all();assert np.isfinite(counts).all();assert np.equal(counts,np.rint(counts)).all()
tids=np.asarray(d['trial_inds']).ravel();eye=np.asarray(d['eyepos']);fix=np.hypot(eye[:,0],eye[:,1])<1
kept=np.array([t for t in np.unique(tids) if ((tids==t)&fix).sum()>20])
windows=[constants['W1'],constants['W2']]
np.savez_compressed(OUT/'trial_data.npz',counts_120hz=counts,eyepos_deg=ep,trial_indices=trials,source_trial_ids=kept[trials],neuron_index=unit,source_cid=cfg['cids'][unit],dt=constants['DT'],windows_120hz=windows,rate_bin_factor=constants['RATE_BIN_FACTOR'])
provenance=dict(session=session,figure='Preprint Figure 2A',plotting_script=str(source.relative_to(ROOT)),plotting_script_sha256=sha(source),dataset=f'processed/{session}/datasets/{args.dataset.name}',dataset_sha256=sha(args.dataset),trial_indices=trials,source_trial_ids=kept[trials].tolist(),neuron_index=unit,source_cid=cfg['cids'][unit],eye_axis=constants['EYE_AXIS'],dt_seconds=constants['DT'],rate_bin_factor=constants['RATE_BIN_FACTOR'],window_bins_120hz=windows,alignment_metadata=meta,alignment=dict(valid_time_bins=120,min_fix_dur=20,min_total_spikes=0,fixation_radius=1.0,fixation_center='origin'),input_hashes={str(f.relative_to(ROOT)):sha(f) for f in [config,ROOT/f'experiments/dataset_configs/sessions/{session}.yaml',ROOT/'models/data/loading.py',ROOT/'VisionCore/covariance.py']},export_sha256=sha(OUT/'trial_data.npz'))
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(provenance,indent=2))
for lo,hi in windows:
 rates=counts[:,lo:hi].reshape(2,-1,constants['RATE_BIN_FACTOR']).sum(2)/(constants['RATE_BIN_FACTOR']*constants['DT'])
 window_mean_rates=counts[:,lo:hi].sum(axis=1)/((hi-lo)*constants['DT'])
 print('WINDOW',lo,hi,'rates',rates.tolist(),'delta',float(abs(window_mean_rates[0]-window_mean_rates[1])),'eye',np.abs(ep[0,lo:hi,0]-ep[1,lo:hi,0]).mean())
