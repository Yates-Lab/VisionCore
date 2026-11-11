"""Verify panel 3's plotted observations and 50-ms rate differences.

Uses only the bundled Figure 2A trial slice. SVG coordinates are checked against
observations; no scientific value is inferred from the artwork.
"""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as E
import numpy as np
OUT = Path(__file__).resolve().parent
HERE = OUT.parents[1]
root = E.parse(HERE/'assets/panel4.svg').getroot()
ns = '{http://www.w3.org/2000/svg}'
def path(id):
    group = next(e for e in root.iter() if e.get('id') == id)
    return next(e for e in group.iter() if e.tag == ns+'path')
# Scientific values come from the original trial export. SVG coordinates
# are used only to check that the existing artwork matches the observations.
source_file=OUT/'trial_data.npz'
provenance=json.loads((OUT/'provenance.json').read_text())
assert hashlib.sha256(source_file.read_bytes()).hexdigest()==provenance['export_sha256']
with np.load(source_file) as data:
 counts=data['counts_120hz'];eye=data['eyepos_deg'];dt=float(data['dt'])
 factor=int(data['rate_bin_factor']);window_bins=data['windows_120hz']
 assert data['trial_indices'].tolist()==provenance['trial_indices']==[35,41]
 assert int(data['neuron_index'])==provenance['neuron_index']==134
 assert counts.shape==(2,72) and eye.shape==(2,72,2)
 assert np.isfinite(counts).all() and np.equal(counts,np.rint(counts)).all()
bin_ms=factor*dt*1000
centers=(np.arange(counts.shape[1]//factor)+0.5)*bin_ms
raw_rates=counts.reshape(2,-1,factor).sum(2)/(factor*dt)
rates={'Trial 1':raw_rates[0],'Trial 2':raw_rates[1]}
svg_per_ms=(1593.72-1383.21)/400;bin_x=1383.21+svg_per_ms*centers
artwork_check={}
for trial,id in [('Trial 1','p4_a0_line2d_14'),('Trial 2','p4_a0_line2d_13')]:
 pts=np.array([float(v) for v in re.findall(r'[-+]?(?:\d*\.\d+|\d+\.?\d*)',path(id).get('d'))]).reshape(-1,2)
 segments=[(a,b) for a,b in zip(pts[:-1],pts[1:]) if b[0]>a[0]]
 values=[]
 for x in bin_x:
  candidates=[(a,b) for a,b in segments if a[0]-.02<=x<=b[0]+.02]
  assert len(candidates)==1,(trial,x,len(candidates))
  a,b=candidates[0];values.append((890.45-a[1])*200/(44.23*2))
 error=float(np.max(np.abs(np.array(values)-rates[trial])))
 assert error<.06,(trial,error)
 artwork_check[trial]={'all_24_rate_bins_match_original_data':True,'max_svg_rounding_error_spks':error}
# Verify the eye samples as well, allowing SVG coordinate rounding.
def eye_vertices(d):
 tokens=re.findall(r'[A-Za-z]|[-+]?(?:\d*\.\d+|\d+\.?\d*)',d)
 i=0;cmd=None;pos=np.zeros(2);points=[]
 while i<len(tokens):
  if tokens[i].isalpha():cmd=tokens[i];i+=1
  n={'M':2,'l':2,'h':1,'s':4}[cmd]
  vals=np.array([float(v) for v in tokens[i:i+n]]);i+=n
  if cmd=='M':pos=vals
  elif cmd=='l':pos=pos+vals
  elif cmd=='h':pos=pos+np.array([vals[0],0])
  elif cmd=='s':pos=pos+vals[-2:]
  points.append(pos.copy())
 return np.array(points)
for j,id in enumerate(['p4_a0_line2d_4','p4_a0_line2d_5']):
 pts=eye_vertices(path(id).get('d'));assert pts.shape==(72,2)
 expected_x=1383.21+svg_per_ms*np.arange(72)*dt*1000
 expected_y=641.65-(eye[j,:,0]+.5)*(45.83/.2)
 err=float(np.max(np.abs(pts-np.column_stack([expected_x,expected_y]))))
 assert err<.12,err
 artwork_check[f'Trial {j+1}']['all_72_eye_samples_match_original_data']=True
 artwork_check[f'Trial {j+1}']['max_eye_svg_coordinate_error_pt']=err

region = next(r for r in json.loads((HERE/'poster_content.json').read_text())['regions'] if r['name']=='panel4')
labels = {t['id']: t for t in region['labels']}
for (lo, hi), name in zip(window_bins, ['divergent', 'matched']):
    duration = (hi-lo)*dt
    assert np.isclose(duration, .050)
    mean_rates = counts[:, lo:hi].sum(axis=1)/duration
    delta = abs(mean_rates[0]-mean_rates[1])
    assert labels[f'delta_rate_{name}']['text'] == f'Δrate = {delta:g} spk/s'
print('PASS: panel 3 original trial data, both rate/eye traces and 50-ms mean-rate differences')
