import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def draw(data,out,palette=None):
 data=validate_summary(data)
 if palette is None:
  palette=json.loads(Path(__file__).with_name('palette.json').read_text(encoding='utf8'))
 if len(palette['colors']) != 4:
  raise ValueError('The palette must define four configuration colors')
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'#25303C','axes.labelcolor':'#25303C','xtick.color':'#64717D','ytick.color':'#64717D','svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 rows=data['rows'];labels=['Full PNV-USeg','w/o positive–negative','w/o dense guidance','Visual only']
 colors=palette['colors']
 fig=plt.figure(figsize=(13.5,4.6));gs=fig.add_gridspec(1,4,width_ratios=[1.6,1,1,1],left=.04,right=.975,bottom=.23,top=.70,wspace=.22)
 ax=fig.add_subplot(gs[0]);ax.set_xlim(0,4.2);ax.set_ylim(3.55,-.55);ax.axis('off')
 ax.axhspan(-.4,.4,color=palette['wash'])
 ax.text(.02,1.12,'Configuration',transform=ax.transAxes,fontsize=11,fontweight='bold')
 ax.text(.77,1.12,'PN',transform=ax.transAxes,fontsize=11,fontweight='bold',ha='center')
 ax.text(.96,1.12,'DG',transform=ax.transAxes,fontsize=11,fontweight='bold',ha='center')
 for i,(name,pn,dg) in enumerate(zip(labels,[1,0,1,0],[1,1,0,0])):
  ax.text(.04,i,name,fontsize=10,va='center',fontweight='bold' if i==0 else 'normal',color=colors[0] if i==0 else '#25303C')
  for x,on in [(3.23,pn),(4.03,dg)]:
   ax.text(x,i,'On' if on else 'Off',ha='center',va='center',fontsize=9,color=palette['on'] if on else '#9BA4AE',fontweight='bold' if on else 'normal')
 for k,(metric,title,unit) in enumerate(zip(['dice','iou','hd95'],['Dice decrease','IoU decrease','HD95 increase'],['pp','pp','px'])):
  ax=fig.add_subplot(gs[k+1]);ax.set_ylim(3.55,-.55);ax.set_yticks([])
  means=np.array([r['change_mean'][metric] for r in rows]);sd=np.array([r['change_sd'].get(metric,0) or 0 for r in rows]);extent=max(means+sd);low=min(0,float(min(means-sd)));span=max(extent-low,.05)
  ax.set_xlim(low-.08*span,extent+.42*span)
  ax.axhspan(-.4,.4,color=palette['wash'],zorder=0);ax.axvline(0,color='#C6CFD8',lw=.8)
  for i in range(4):
   if i==0:
    ax.scatter(0,0,s=70,marker='*',color=colors[i],zorder=4);ax.text(.08*span,0,'Reference',va='center',fontsize=9,color=colors[i]);continue
   ax.barh(i,means[i],height=.40,color=colors[i],zorder=3)
   if len(data['training_seeds'])==3:ax.errorbar(means[i],i,xerr=sd[i],color='#4B5563',capsize=2,elinewidth=.9,zorder=5,fmt='none')
   ax.text(means[i]+sd[i]+.06*span,i,f'{means[i]:.3f}',ha='left',va='center',fontsize=10,color='#25303C')
  ax.set_title(title,loc='left',fontsize=11,pad=20,fontweight='bold')
  ax.set_xlabel(unit,fontsize=9,labelpad=7);ax.tick_params(length=0,labelsize=8);ax.grid(axis='x',color='#E8EDF2',lw=.7);ax.set_axisbelow(True)
  for s in ax.spines.values():s.set_visible(False)
 fig.text(.04,.93,'Core module ablation',fontsize=18,fontweight='bold')
 final=len(data['training_seeds'])==3
 subtitle='Three training seeds · Mean paired change ± sample SD' if final else 'Seed42 test preview · Additional seeds 3407 and 2026 are in progress'
 fig.text(.04,.83,subtitle,fontsize=10,color='#64717D')
 fig.text(.04,.12,'PN: positive–negative semantics    DG: dense guidance    Visual prototypes, interaction and channel gating remain in all variants',fontsize=8.5,color='#64717D')
 fig.text(.04,.05,'Changes are relative to the full model at the same seed. Positive values indicate performance degradation. pp: percentage points.',fontsize=8.5,color='#64717D')
 for ext in ['png','svg','pdf']:fig.savefig(out/('PNV_USeg_Core_Ablation_'+('Three_Seeds' if final else 'Seed42_Preview')+'_EN.'+ext),dpi=300)
 plt.close(fig)

CONDITIONS = ['F', 'F_remove_dual', 'F_remove_summary', 'F_remove_all_text']
METRICS = ['dice', 'iou', 'hd95']

def validate_summary(data):
 import math
 if data.get('evaluation_split') != 'test' or data.get('result_status') != 'test_complete':
  raise ValueError('Only completed test results can be plotted')
 seeds = data.get('training_seeds', [])
 if seeds != [42] and not (len(seeds) == 3 and set(seeds) == {42, 3407, 2026}):
  raise ValueError('Use the seed42 preview or all three seeds: 42, 3407, 2026')
 rows = data.get('rows', [])
 if len(rows) != 4 or {r['condition'] for r in rows} != set(CONDITIONS):
  raise ValueError('The four matched core ablation configurations are required')
 rows = [next(r for r in rows if r['condition'] == c) for c in CONDITIONS]
 for row in rows:
  for metric in METRICS:
   value = float(row['test_mean'][metric])
   if not math.isfinite(value) or value < 0 or (metric != 'hd95' and value > 100):
    raise ValueError('Dice/IoU must be percentages; HD95 must be nonnegative pixels')
   expected = (rows[0]['test_mean'][metric] - value) * (-1 if metric == 'hd95' else 1)
   change = float(row['change_mean'][metric])
   if not math.isfinite(change) or abs(change - expected) > 1e-6:
    raise ValueError('Change does not match the full-model reference: ' + metric)
   if len(seeds) == 3:
    for key in ['test_sd', 'change_sd']:
     sd = float(row[key][metric])
     if not math.isfinite(sd) or sd < 0:
      raise ValueError('Three-seed sample SD must be finite and nonnegative')
 return dict(data, rows=rows)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--palette');a=p.parse_args()
 draw(json.loads(Path(a.data).read_text(encoding='utf8')),a.out,json.loads(Path(a.palette).read_text(encoding='utf8')) if a.palette else None)
