"""Regenerate the approved Panel 5 plot geometry at its final physical sizes."""
from pathlib import Path
import os,sys,json
HERE=Path(__file__).resolve().parent
POSTER=HERE.parents[1]
sys.path.insert(0, str(POSTER))
from fonts import checked_font_paths
os.environ['MPLCONFIGDIR']=str(HERE/'.mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcdefaults()
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.container import ErrorbarContainer
from matplotlib.transforms import ScaledTranslation
from plot_helpers import plot_fano_population, plot_nc_violin
for p in checked_font_paths():
 fm.fontManager.addfont(str(p))
# Preserve the manuscript helper's original Matplotlib defaults.
plt.rcParams.update({'ps.fonttype':42, 'font.sans-serif':['Arial','Helvetica','DejaVu Sans']})
plt.rcParams.update({'font.family':'Arial','font.size':21,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.8,'axes.labelcolor':'#202632','xtick.color':'#202632','ytick.color':'#202632','svg.hashsalt':'panel5-typst-pilot'})
ASSETS=HERE/'assets';ASSETS.mkdir(exist_ok=True)
d=np.load(HERE/'plot_data.npz');navy='#193654'
# Corrected-residual accent, shared with panel5.typ (keep the two in sync).
gray=os.environ.get('POSTER_CORRECTED_ACCENT','#D62839')
muted='#526474';null_fill='#b7bfc8'
def save(fig,name):
 for ext in ['pdf','svg']:fig.savefig(ASSETS/f'{name}.{ext}',transparent=True,dpi=144)
 plt.close(fig)
for key in ['total','stimulus','uncorrected','fem','corrected']:
 fig=plt.figure(figsize=(190/72,190/72));ax=fig.add_axes([0,0,1,1]);ax.imshow(d[key],cmap='seismic_r',vmin=-float(d['vlim']),vmax=float(d['vlim']),interpolation='none');ax.axis('off');save(fig,key)
fig=plt.figure(figsize=(281.52/72,10.5/72));ax=fig.add_axes([0,0,1,1]);ax.imshow(np.linspace(-1,1,512)[None,:],cmap='seismic_r',aspect='auto',interpolation='none',extent=[0,1,0,1]);ax.set_xticks([]);ax.set_yticks([])
for spine in ax.spines.values():spine.set_linewidth(.8)
save(fig,'colorbar')
f={'sess_slope_unc':d['fano_unc'],'sess_slope_cor':d['fano_cor'],'mean_sess_cor_null_ci':d['fano_null'],'p_emp_mean_sess':1/1001,'n_shuff_sess':1000}
n={'rho_u':d['rho_unc'],'rho_c':d['rho_cor'],'r_u_mean':d['noise_mean'][0],'r_c_mean':d['noise_mean'][1],'r_u_sd':d['noise_sd'][0],'r_c_sd':d['noise_sd'][1],'null_dr_ci':d['noise_null']-d['noise_mean'][0],'p_emp_dr':1/1001,'n_shuff_dr':1000}
data={'WINDOWS_MS':np.array([25.]),'fano_stats':{25.:f},'nc_stats':{25.:n}}
audit={}
for name,draw in [('fano',plot_fano_population),('noise',plot_nc_violin)]:
 fig=plt.figure(figsize=(315/72,515/72));ax=fig.add_axes([50/315,60/515,230/315,455/515]);draw(ax=ax,data=data)
 # Existing manuscript helper draws data and defines ranges. All narrative text
 # is rebuilt using the poster's wording. Significance sits in the Typst header.
 for t in list(ax.texts):t.remove()
 ax.set_xlabel('');ax.set_ylabel('')
 for line in list(ax.lines):
  if len(line.get_xdata())==4 and np.allclose(line.get_xdata(),[0,0,1,1]):line.remove()
  elif line.get_marker()=='o' and len(line.get_xdata())==1:
   cor=float(line.get_xdata()[0])==1;line.set_markeredgecolor(gray if cor else navy);line.set_markerfacecolor(gray if cor else navy);line.set_markersize(12);line.set_markeredgewidth(3)
  elif len(line.get_xdata())==2 and line.get_color()=='tab:blue':line.set_color('#737373');line.set_linewidth(1.8)
 for cont in ax.containers:
  if isinstance(cont,ErrorbarContainer):
   for coll in cont.lines[2]:coll.set_color([navy,gray]);coll.set_linewidth(3.0)
 # Shuffle-null regions: light reference fill, no outline, labelled directly.
 from matplotlib.patches import Rectangle as _Rect
 for patch in ax.patches:
  if isinstance(patch,_Rect):patch.set_edgecolor('none');patch.set_facecolor(null_fill);patch.set_alpha(0.75);patch.set_zorder(2.5)
 # Per-session traces: strengthened just enough to survive print.
 for line in ax.lines:
  if len(line.get_xdata())==2 and line.get_linewidth()<1 :line.set_color('#a3acb6');line.set_linewidth(1.3);line.set_alpha(0.9)
 lo,hi=(d['fano_null'] if name=='fano' else d['noise_null'])
 ax.text(1,hi+(0.03 if name=='fano' else 0.012),'Shuffle null',ha='center',va='bottom',fontsize=18,color=muted,zorder=6)
 ax.set_xticks([0,1],['Uncorrected','Corrected']);ax.tick_params(axis='x',labelsize=21,pad=17,length=3)
 for tick,c,dx in zip(ax.get_xticklabels(),[navy,gray],[-7,7]):
  tick.set_color(c);tick.set_transform(tick.get_transform()+ScaledTranslation(dx/72,0,fig.dpi_scale_trans))
 ax.tick_params(axis='y',labelsize=21,pad=10,length=0)
 ax.set_ylim((.4,1.85) if name=='fano' else (-.2,.36))
 ticks=np.arange(.4,1.81,.2) if name=='fano' else np.arange(-.2,.31,.1)
 ax.set_yticks(ticks,[f'{v:.1f}' if abs(v)>1e-8 else '0.0' for v in ticks])
 if name=='fano':
  ax.annotate('Poisson\n= 1', xy=(.90, 1.0), xycoords=ax.get_yaxis_transform(),
              xytext=(0, 6), textcoords='offset points', fontsize=21,
              color='#656b73', ha='center', va='bottom', linespacing=1.0)
 audit[name]={'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim()),'axis_width_pt':230,'axis_height_pt':455,'plot_canvas_pt':[315,515],'n_session_lines':19 if name=='fano' else None}
 save(fig,name)
(HERE/'plot_layout.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
