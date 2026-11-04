from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--data',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
D=json.loads(args.data.read_text(encoding='utf8'))
if D.get('evaluation_split')!='test' or D.get('result_status')!='test_complete':
 raise ValueError('Completed test results are required')
F=D['finetuning'];OUT=args.out;OUT.mkdir(parents=True,exist_ok=True)
MODELS=['PNV-USeg','TransUNet','SAM-Med2D','Iris-2D','U-Net++','U-Net','CLIPSeg','LViT','Swin-Unet']
ORGANS=['Left ventricle','Liver','Gallbladder']
INK='#253D49'; GRAY='#748690'; CORAL='#D66D62'; TEAL='#6BA89D'; BLUE='#6C99B6'; LGRAY='#BBC6CA'
COLORS=dict(zip(MODELS,[CORAL,'#6796B3','#A5B987','#A18AB6','#C3A16F','#6BA89D','#92A2B0','#BA93A3','#A9AA81']))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':INK,
 'axes.labelcolor':INK,'xtick.color':GRAY,'ytick.color':GRAY,'svg.fonttype':'none',
 'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
 'axes.titleweight':'bold','axes.titlecolor':INK,'axes.linewidth':.65,
 'savefig.facecolor':'white'})
def save(fig,name):
 for ext in ['png','svg','pdf']: fig.savefig(OUT/f'{name}.{ext}',dpi=300,facecolor='white')
 plt.close(fig)

# D. Separate radar panels avoid placing nine overlapping polygons on one chart.
# All panels use the same nine axes and the same 40–90% radial range.
theta=np.linspace(0,2*np.pi,9,endpoint=False)
closed=np.r_[theta,theta[0]]
axislabels=['LV\nK4','LV\nK8','LV\nK16','Liver\nK4','Liver\nK8','Liver\nK16','Gall\nK4','Gall\nK8','Gall\nK16']
def radar_values(m): return np.array([F[o][m][str(k)][0][0] for o in ORGANS for k in [4,8,16]])
pnv=radar_values('PNV-USeg')
fig,axs=plt.subplots(3,3,figsize=(11.6,11.8),subplot_kw={'projection':'polar'})
fig.subplots_adjust(left=.075,right=.925,bottom=.10,top=.865,hspace=.65,wspace=.45)
for ax,m in zip(axs.flat,MODELS):
 ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
 ax.set_ylim(40,90); ax.set_yticks([40,60,80,90],['40','60','80','90'],fontsize=6,color='#81939B'); ax.set_rlabel_position(20)
 ax.set_xticks(theta,axislabels,fontsize=7.2); ax.tick_params(axis='x',pad=7)
 ax.grid(color='#E2EAED',lw=.65); ax.spines['polar'].set_color('#DFE7EA')
 c=CORAL if m=='PNV-USeg' else BLUE; v=radar_values(m)
 ax.plot(closed,np.r_[v,v[0]],color=c,lw=2 if m=='PNV-USeg' else 1.5,zorder=4)
 ax.fill(closed,np.r_[v,v[0]],color=c,alpha=.19 if m=='PNV-USeg' else .13,zorder=2)
 ax.scatter(theta,v,s=12,color=c,edgecolor='white',linewidth=.35,zorder=5)
 if m!='PNV-USeg': ax.plot(closed,np.r_[pnv,pnv[0]],color=CORAL,lw=1.2,ls=(0,(3,2)),alpha=.85,zorder=3)
 ax.set_title(m,fontsize=11,fontweight='bold',color=CORAL if m=='PNV-USeg' else INK,pad=28)
fig.text(.075,.96,'Adaptation profiles across organs and support sizes',fontsize=18,fontweight='bold')
fig.text(.075,.92,'Mean test Dice (%) · Nine methods, nine matched settings · Shared radial range: 40–90%',fontsize=10,color=GRAY)
fig.legend(handles=[Line2D([],[],color=BLUE,lw=1.6,label='Baseline method'),Line2D([],[],color=CORAL,lw=1.4,ls='--',label='PNV-USeg reference')],loc='lower center',bbox_to_anchor=(.5,.04),ncol=2,frameon=False,fontsize=10)
fig.text(.075,.025,'LV: left ventricle · Gall: gallbladder · Source seeds: 42, 3407, 2026 · Support seed: 42 · Means only; polygon area is not an aggregate score',fontsize=8,color=GRAY)
save(fig,'D_Organ_finetuning_radar_profiles_EN')

