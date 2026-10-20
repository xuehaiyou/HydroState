"""English PNG slides from recorded validation metrics and checkpoint diagnostics."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

TASKS=['water','soil_moisture','et']
NAMES={'water':'Water fraction','soil_moisture':'Soil moisture','et':'Evapotranspiration',
 'trees':'Trees','grass':'Grass','flooded_vegetation':'Flooded vegetation','crops':'Crops',
 'shrub_and_scrub':'Shrub & scrub','built':'Built area','bare':'Bare ground','snow_and_ice':'Snow & ice',
 'lai':'Leaf area index','gpp':'Gross primary production','albedo_bsa':'Black-sky albedo',
 'albedo_wsa':'White-sky albedo','lst':'Land surface temperature'}
UNITS=['fraction','m³ m⁻³','mm day⁻¹']
COLORS=['#9BAEBF','#6A9D98','#2C5D7C']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,
 'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#25364A',
 'text.color':'#25364A','axes.edgecolor':'#ABB4BC','xtick.color':'#52606D','ytick.color':'#25364A',
 'figure.facecolor':'white','savefig.facecolor':'white'})

def save(fig,path):
 fig.savefig(path,dpi=220,facecolor='white');plt.close(fig)

def heading(fig,title,sub,foot):
 fig.text(.045,.955,title,fontsize=25,weight='bold',va='top')
 fig.text(.045,.898,sub,fontsize=13,va='top',color='#52606D')
 fig.text(.045,.025,foot,fontsize=11,color='#52606D',va='bottom')

def best(run):return max(run['history'],key=lambda x:x['hydro_score'])

def result_table(root, filename, title, subtitle, footnote, headers, rows, widths, bold, height=9):
 fig,ax=plt.subplots(figsize=(18,height));ax.axis('off')
 fig.subplots_adjust(left=.04,right=.96,top=.79,bottom=.16)
 table=ax.table(cellText=rows,colLabels=headers,colWidths=widths,cellLoc='center',
                loc='center',bbox=[0,0,1,1])
 table.auto_set_font_size(False);table.set_fontsize(15)
 for (r,c),cell in table.get_celld().items():
  cell.set_edgecolor('#DCE3E8');cell.set_linewidth(.6)
  cell.PAD=.04
  if r==0:
   cell.set_facecolor('#294F66');cell.set_text_props(color='white',weight='bold')
   cell.set_height(cell.get_height()*1.7)
  else:
   cell.set_facecolor('#F0F5F7' if r%2 else 'white')
   if c==0:cell.set_text_props(ha='left')
   if (r-1,c) in bold:cell.set_text_props(weight='bold',color='#183E4C')
 heading(fig,title,subtitle,footnote)
 save(fig,root/filename)

def table_figures(root):
 a=pd.read_csv(root/'source/architecture_metrics.csv')
 columns=[f'{t}_{m}' for t in TASKS for m in ['r2','rmse']]
 headers=['Model configuration']+[f'{("ET" if t=="et" else NAMES[t])}\n{m}' for t in TASKS for m in ['R² ↑','RMSE ↓']]
 rows=[[r['label'].replace(' + ',' +\n')]+[f'{r[k]:.3f}' for k in columns] for _,r in a.iterrows()]
 bold=set()
 for j,k in enumerate(columns,1):
  values=a[k].round(3);winner=values.max() if k.endswith('r2') else values.min()
  bold.update((i,j) for i,v in enumerate(values) if v==winner)
 common=json.loads((root/'source/common_sm_metrics.json').read_text())
 result_table(root,'01_architecture_r2.png','Validation performance across three model configurations',
              'Fixed original best checkpoints: epochs 17, 32 and 32. Bold = best value in each metric column (ties at displayed precision).',
              f'SM: common observed-land support, N = {common["common_sample_count"]}; water and ET retain original validation scores. Training recipes differ.\nRMSE units: water fraction (fraction), soil moisture (m³ m⁻³), evapotranspiration (mm day⁻¹). All R² values are dimensionless.',
              headers,rows,[.30]+[.70/6]*6,bold,height=8)
 sixteen_table(root)

def sixteen_table(root):
 a=pd.read_csv(root/'source/sixteen_variable_best.csv')
 rows=[[NAMES[r.task],f'{r.joint_r2:.3f}',f'{r.best_r2:.3f}',str(int(r.best_epoch))] for _,r in a.iterrows()]
 bold=set()
 for i,r in a.iterrows():
  vals=[round(r.joint_r2,3),round(r.best_r2,3)]
  bold.update((i,j+1) for j,v in enumerate(vals) if v==max(vals))
 result_table(root,'04_sixteen_variables_taskwise_best.png','Best observed validation accuracy for 16 variables',
              'Multilevel temporal model • 40 completed validation epochs • bold = higher R² within each row (ties at displayed precision)',
              'Joint best: one checkpoint selected by validation hydro_score. Task-specific best: maximum validation R² for each variable.\nTask-specific maxima come from different checkpoints and do not represent one deployable model.',
              ['Variable','Joint best R² ↑\nEpoch 33','Task-specific\nbest R² ↑','Best epoch\n(per variable)'],
              rows,[.37,.23,.23,.17],bold,height=12)

def history_figures(root):
 runs={r['run']:r for r in json.loads((root/'source/histories.json').read_text())}
 common_path=root/'source/common_sm_metrics.json'
 common_sm=json.loads(common_path.read_text()) if common_path.exists() else None
 selected=[('merged_n20000_4241518','Shared decoder'),
 ('staged_light_land_sm_b32_n20000_4245730','Light three-stage + land constraint'),
 ('multilevel_n20000_4251319','Multilevel fusion + land constraint')]
 records=[]
 for key,label in selected:
  r=runs[key]; b=best(r)
  records.append(dict(run=key,label=label,selected_epoch=int(b['step']),validated_epochs=len(r['history']),**{k:v for k,v in b.items() if any(k.startswith(t+'_') for t in TASKS)}))
  if common_sm:
   updated=common_sm['models'][key]
   assert updated['epoch']==int(b['step'])
   records[-1].update({f'soil_moisture_{k}':v for k,v in updated['common'].items() if k!='n'})
   records[-1]['soil_moisture_observation_count']=updated['common']['n']
   records[-1]['soil_moisture_prediction_coverage']=float('nan')
 pd.DataFrame(records).to_csv(root/'source/architecture_metrics.csv',index=False)
 labels=[f"{r['label']}\nEpoch {r['selected_epoch']} / {r['validated_epochs']} validated" for r in records]
 for metric,suffix in [('r2','R²  ↑'),('rmse','RMSE  ↓')]:
  fig,axes=plt.subplots(1,3,figsize=(17,8.7));fig.subplots_adjust(left=.32,right=.945,top=.80,bottom=.15,wspace=.34)
  for j,(ax,t) in enumerate(zip(axes,TASKS)):
   y=np.arange(len(records));vals=np.array([r[t+'_'+metric] for r in records])
   ax.barh(y,vals,color=COLORS,height=.57)
   for yy,v in zip(y,vals):ax.text(v+.012*(1 if metric=='r2' else vals.max()),yy,f'{v:.3f}',va='center',fontsize=13)
   ax.set_yticks(y,labels if j==0 else ['']*len(y),fontsize=12);ax.invert_yaxis()
   ax.set_title(NAMES[t],fontsize=17,pad=16);ax.set_xlabel(suffix+(f' ({UNITS[j]})' if metric=='rmse' else ''))
   ax.set_xlim(0,1.08 if metric=='r2' else vals.max()*1.25);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
  foot=(f'SM: identical observed-land weights and common valid samples (N = {common_sm["common_sample_count"]}); water and ET: original validation metrics.\nFixed original best checkpoints; all runs completed 50 epochs. Training recipes differ; this is not a controlled architecture ablation.'
        if common_sm else 'Same merged validation split; all runs completed 50 epochs. One validation-selected checkpoint per model.\nLand constraint changes soil-moisture support/aggregation; multilevel fusion also changes optimization and missing-data pooling.')
  heading(fig,'Validation performance across three model configurations','One checkpoint per run, selected by its original validation hydro_score.',foot)
  save(fig,root/f'01_architecture_{metric}.png')
 # Two 16-task models, selected checkpoint versus per-task envelope.
 keys=['ecosystem_temporal_4274950','ecosystem_multilevel_temporal_4275071']; ts=list(NAMES)
 fig,axes=plt.subplots(1,2,figsize=(17,10));fig.subplots_adjust(left=.235,right=.96,top=.80,bottom=.15,wspace=.45)
 for ax,key,label,col in zip(axes,keys,['Temporal queries','Multilevel temporal'],['#8B9FB5','#64A3A0']):
  r=runs[key];b=best(r); y=np.arange(16);values=[b[t+'_r2'] for t in ts]
  ax.barh(y,values,color=col,height=.65)
  for yy,v in zip(y,values):ax.text(v+.018,yy,f'{v:.3f}',va='center',fontsize=12)
  ax.set(yticks=y,yticklabels=[NAMES[t] for t in ts] if ax is axes[0] else ['']*16,xlim=(0,1.14),xlabel='Validation R²  ↑');ax.invert_yaxis();ax.tick_params(axis='y',length=0,labelsize=12)
  ax.set_title(f'{label}\nSelected epoch {int(b["step"])}',fontsize=17,pad=15);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
 heading(fig,'Sixteen-variable retrieval: best joint checkpoints','Each column is one deployable checkpoint, selected by the run’s validation hydro_score.',
 'Temporal queries: 50 validated epochs. Multilevel temporal: 40 validated epochs (wall-time limit reached).\nValidation-selected results from individual runs; task-specific best epochs are shown separately.')
 save(fig,root/'03_sixteen_variables_joint_best.png')
 r=runs[keys[1]];b=best(r);per=[max(r['history'],key=lambda x:x[t+'_r2']) for t in ts]
 pd.DataFrame([dict(task=t,best_epoch=int(p['step']),best_r2=p[t+'_r2'],best_rmse=p[t+'_rmse'],joint_epoch=int(b['step']),joint_r2=b[t+'_r2']) for t,p in zip(ts,per)]).to_csv(root/'source/sixteen_variable_best.csv',index=False)
 fig,ax=plt.subplots(figsize=(17,10));fig.subplots_adjust(left=.26,right=.88,top=.80,bottom=.16)
 for i,(t,p) in enumerate(zip(ts,per)):
  lo=b[t+'_r2'];hi=p[t+'_r2'];ax.plot([lo,hi],[i,i],color='#93AEBB',lw=4)
  ax.scatter(lo,i,s=65,facecolor='white',edgecolor='#50789B',zorder=3)
  ax.scatter(hi,i,s=75,color='#287E86',zorder=4)
  ax.text(1.015,i,f'{hi:.3f}   E{int(p["step"]):02}',va='center',fontsize=13)
 ax.scatter([],[],s=65,facecolor='white',edgecolor='#50789B',label='Joint best checkpoint (epoch 33)')
 ax.scatter([],[],s=75,color='#287E86',label='Task-specific best epoch')
 ax.set(yticks=np.arange(16),yticklabels=[NAMES[t] for t in ts],xlim=(.35,1.15),xticks=[.4,.5,.6,.7,.8,.9,1.0],xlabel='Validation R²  ↑');ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.tick_params(axis='y',length=0);ax.legend(loc='lower left',bbox_to_anchor=(0,-.17),ncol=2,frameon=False,fontsize=12)
 heading(fig,'Best observed accuracy for all 16 variables','Multilevel temporal model • highest validation R² for each variable across 40 completed validation epochs',
 'Filled points are a per-task upper envelope: they do not represent one checkpoint. E = task-specific selected epoch.')
 save(fig,root/'04_sixteen_variables_taskwise_best.png')
 # Learning curves convey unequal progress and plateaus.
 fig,axes=plt.subplots(1,3,figsize=(17,8));fig.subplots_adjust(left=.07,right=.98,top=.78,bottom=.25,wspace=.25)
 for (key,label),c in zip(selected,COLORS):
  h=runs[key]['history']
  for ax,t in zip(axes,TASKS):ax.plot([x['step'] for x in h],[x[t+'_r2'] for x in h],lw=2,label=label,color=c)
 for ax,t in zip(axes,TASKS):ax.set(title=NAMES[t],xlabel='Epoch',ylabel='Validation R²');ax.grid(alpha=.15)
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.10),ncol=3,frameon=False,fontsize=11)
 heading(fig,'Validation trajectories of the three model architectures','All completed validation epochs; curves are unsmoothed.',
 'All models use the merged split. Land constraint changes SM supervision; multilevel fusion also changes optimization and pooling.')
 save(fig,root/'05_validation_learning_curves.png')
 if common_sm:table_figures(root)


def diagnostics_figures(root, scatter_only=False):
 d=root/'diagnostics';z=np.load(d/'scatter_points.npz');metrics=json.loads((d/'metrics.json').read_text());prov=json.loads((d/'provenance.json').read_text())
 density_audit={}
 def scatter(axes,fig,tasks=TASKS):
  for ax,t in zip(axes,tasks):
   j=TASKS.index(t)
   p=z[t];lo=min(0,float(p.min()));hi=float(p.max())*1.03
   min_count=max(2,int(np.ceil(len(p)*.0001)))
   artist=ax.hexbin(p[:,0],p[:,1],gridsize=40 if t=='soil_moisture' else 65,mincnt=min_count,bins='log',cmap='viridis',extent=(lo,hi,lo,hi),linewidths=0)
   retained=int(np.asarray(artist.get_array()).sum())
   density_audit[t]=dict(sampled_observations=len(p),minimum_bin_count=min_count,visible_observations=retained,visible_fraction=retained/len(p))
   ax.plot([lo,hi],[lo,hi],ls='--',lw=1.2,color='#CC7766');ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel=f'Reference ({UNITS[j]})',ylabel=f'Prediction ({UNITS[j]})',title=NAMES[t]);ax.set_aspect('equal')
   ax.text(.04,.96,f'R² = {metrics[t+"_r2"]:.3f}\nRMSE = {metrics[t+"_rmse"]:.3f}\nN = {prov["counts"][t]:,}',transform=ax.transAxes,va='top',fontsize=13,bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
   ax.text(.97,.04,f'Bins ≥ {min_count} observations\n{retained/len(p):.1%} of sampled observations shown',transform=ax.transAxes,ha='right',va='bottom',fontsize=10,bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
   cb=fig.colorbar(artist,ax=ax,orientation='horizontal',fraction=.046,pad=.16);cb.set_label('Observations per visible hexagon (log scale)',fontsize=10)
 fig,axes=plt.subplots(1,3,figsize=(17,8.5));fig.subplots_adjust(left=.065,right=.98,top=.79,bottom=.18,wspace=.32);scatter(axes,fig)
 heading(fig,'Three-variable retrieval from the multilevel temporal model','Best joint checkpoint: epoch 33 • full validation split • predictions evaluated at each reference product’s support',
 'Metrics use all valid observations; low-density bins are hidden for display only (threshold shown in each panel).\nUniform sample: up to 50,000 per task. Water: output pixels; SM: window scalar; ET: native-cell means. References are not in-situ ground truth.')
 save(fig,root/'06_three_variable_scatter.png')
 for t in TASKS:
  fig,ax=plt.subplots(figsize=(10,10))
  fig.subplots_adjust(left=.12,right=.91,top=.79,bottom=.19)
  scatter([ax],fig,[t])
  heading(fig,f'{NAMES[t]} retrieval accuracy','Multilevel temporal model • epoch 33 • full validation split',
   'Metrics use all valid reference-support observations.\nUniformly sampled display; low-density bins are hidden using the threshold shown above.')
  save(fig,root/f'06_scatter_{t}.png')
 (root/'source'/'scatter_density_filter.json').write_text(json.dumps(density_audit,indent=2))
 if scatter_only:return
 # Import only the project's map conversion, preserving supervision contracts.
 from hydrostate.engine.visualization_hook import comparison_maps
 for i,entry in enumerate(prov['selected'],1):
  src=d/Path(entry['file']).with_suffix('.npz');sample=dict(np.load(src));meta=entry['metainfo']
  maps={t:comparison_maps(sample,t) for t in TASKS}
  fig,axes=plt.subplots(3,3,figsize=(17,12));fig.subplots_adjust(left=.07,right=.90,top=.79,bottom=.105,wspace=.18,hspace=.35)
  cmaps=['Blues','YlGnBu','YlOrRd'];limits=[(0,1),(0,.6),(0,10)]
  for row,t in enumerate(TASKS):
   for col,arr in enumerate(maps[t]):
    ax=axes[row,col];cmap=plt.get_cmap(cmaps[row]).with_extremes(bad='#E3E5E8')
    im=ax.imshow(arr,cmap=cmap,vmin=limits[row][0],vmax=limits[row][1],interpolation='nearest')
    ax.set_xticks([]);ax.set_yticks([])
    if col==0:ax.set_ylabel(f'{NAMES[t]}\n({UNITS[row]})',fontsize=13)
    if col==0:
     h,w=arr.shape
     ax.plot([w*.08,w*.08+10],[h*.90,h*.90],lw=4,color='white')
     ax.text(w*.08+5,h*.84,'300 m',color='white',ha='center',fontsize=11,
             bbox=dict(facecolor='#25364A',alpha=.55,edgecolor='none',pad=2))
    if col==0 and t in entry.get('target_time',{}):
     bounds=entry['target_time'][t];dates=[pd.Timestamp(v,unit='s').strftime('%Y-%m-%d') for v in bounds]
     ax.text(.5,-.06,dates[0] if dates[0]==dates[-1] else ' to '.join(dates),transform=ax.transAxes,ha='center',fontsize=10)
    if row==0:ax.set_title(['Retrieval at 30 m','Retrieval at reference support','Reference product'][col],fontsize=16,pad=14)
    if t=='soil_moisture' and col>0:
     v=np.asarray(arr).ravel();v=v[np.isfinite(v)];ax.text(.5,.5,f'{v.mean():.3f}\nwindow scalar' if len(v) else 'No reference',transform=ax.transAxes,ha='center',va='center',fontsize=16,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
   pos=axes[row,-1].get_position();cax=fig.add_axes([.925,pos.y0,.012,pos.height]);fig.colorbar(im,cax=cax,extend='max' if row else 'neither')
  loc=f"{float(meta['lat']):.3f}°, {float(meta['lon']):.3f}°" if 'lat' in meta and 'lon' in meta else f'Validation tile {i}'
  heading(fig,f'Spatial retrieval example {i}: {loc}',f'Multilevel temporal model • epoch 33 • validation sample date: {meta.get("target_time", "see source metadata")}',
   'Examples selected by seeded tile sampling, independent of accuracy. Grey = unsupported/missing.\nSM reference is a scalar, not a 30 m map. ET reference cells are projected to footprints for display; no interpolation of missing values.')
  save(fig,root/f'07_mapping_example_{i:02}.png')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/group_meeting_20261020'));p.add_argument('--diagnostics',action='store_true');p.add_argument('--scatter-only',action='store_true');p.add_argument('--tables-only',action='store_true');a=p.parse_args()
 if a.tables_only:table_figures(a.root)
 elif a.diagnostics or a.scatter_only:diagnostics_figures(a.root,scatter_only=a.scatter_only)
 else:history_figures(a.root)
