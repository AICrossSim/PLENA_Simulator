"""Aggregate only frozen-design heldout measurements; all ratios are paired."""
from collections import defaultdict
import math
import numpy as np
from .common import *
from .evaluate import selections

def group_rows(rows):
 out=defaultdict(list)
 for r in rows:out[r['macs'],r['batch'],r['org'],r['scheduler']].append(r)
 return out

def load_results(mode='full_scaled',bw=10,tag='frozen'):
 rows=[]
 for s in selections():
  rows+=read_json(ROOT/'E1/windows'/f"{s['budget']}_{s['org']}_{mode}_{bw}_{tag}.json.gz")
 return rows

def strongest_map():
 ss={(s['budget'],s['org']):s for s in selections()};out={};choices=[]
 for m in BUDGETS:
  for b in BATCHES:
   org=min(('H','H+'),key=lambda o:(ss[m,o]['dev_by_batch'][str(b)],o))
   out[m,b]=org;choices.append(dict(macs=m,batch=b,strongest_homo=org,
    H_dev_ms=ss[m,'H']['dev_by_batch'][str(b)]/1e6,Hplus_dev_ms=ss[m,'H+']['dev_by_batch'][str(b)]/1e6,selection_uses_heldout=False))
 write_csv(ROOT/'E1/strongest_homo_selection.csv',choices)
 return out

def online_best_map():
 delta=read_json(ROOT/'E2/delta_selection.json')['delta'];scores=defaultdict(list)
 for r in read_csv(ROOT/'E2/delta_development.csv.gz'):
  if float(r['delta'])==delta and r['org'] in ('H','H+'):scores[int(r['macs']),int(r['batch']),r['org']].append(float(r['cycles']))
 return {(m,b):min(('H','H+'),key=lambda o:gm(scores[m,b,o])) for m in BUDGETS for b in BATCHES}

def aggregate(rows,strongest):
 groups=group_rows(rows);cells=[];compare=[];balance=[];online_best=online_best_map()
 for m in BUDGETS:
  for b in BATCHES:
   h=strongest[m,b];ho=groups[m,b,h,'offline'];hn=groups[m,b,h,'online']
   observed_h=min(('H','H+'),key=lambda o:gm([r['ms'] for r in groups[m,b,o,'offline']]))
   observed_best=gm([r['ms'] for r in groups[m,b,observed_h,'offline']])
   for org in ('S','H','H+','P','P+'):
    rr=groups[m,b,org,'offline'];nn=groups[m,b,org,'online'];old=groups[m,b,org,'old'];nq=groups[m,b,org,'quota_ablation']
    off=gm([r['ms'] for r in rr]);lb=gm([r['offline_lb_cycles']/1e6 for r in rr]);on=gm([r['ms'] for r in nn]);hb=gm([r['ms'] for r in ho]);hon=gm([r['ms'] for r in hn])
    bw=gm([r['req_weight_TBps'] for r in rr]);speed=hb/off;speedon=hon/on
    cells.append(dict(macs=m,batch=b,org=org,granularity=rr[0]['granularity'],cores=rr[0]['cores'],offline_ms=off,offline_lb_ms=lb,online_ms=on,
     req_weight_TBps=bw,req_total_TBps=gm([r['req_total_TBps'] for r in rr]),peak_resident_MiB=max(r['peak_resident_MiB'] for r in rr),
     speedup_vs_strongest_homo_offline=speed,speedup_vs_strongest_homo_online=speedon,
     crit_A10=bw<=10,crit_A3p3=bw<=3.3,crit_B=speed>=1.1,crit_C=speedon>=1.05 and on/off<=1.05,
     strongest_homo=h,geometry=rr[0]['geometry'],used_macs=rr[0]['used_macs'],online_over_offline=on/off,
     heldout_observed_fastest_homo=observed_h,speedup_vs_heldout_fastest_homo_offline_diagnostic=observed_best/off,
     dev_online_best_homo=online_best[m,b],speedup_vs_dev_online_best_homo_diagnostic=gm([r['ms'] for r in groups[m,b,online_best[m,b],'online']])/on,
     offline_gap_pct=100*(off/lb-1),speedup_certified_lower=gm([r['offline_lb_cycles']/1e6 for r in ho])/off,
     crit_C_certified=speedon>=1.05 and on/lb<=1.05,
     speedup_possible_upper=hb/lb,req_weight_upper_from_lb_TBps=gm([r['weight_bytes']/r['offline_lb_cycles']/1000 for r in rr]),
     req_weight_at_owner_compute_TBps=gm([r['req_weight_at_owner_compute_TBps'] for r in rr]),
     req_weight_and_replicas_TBps=gm([r['req_weight_and_replicas_TBps'] for r in rr]),
     req_boundary_and_replicas_TBps=gm([r['req_boundary_and_replicas_TBps'] for r in rr]),
     weight_MiB=gm([r['weight_bytes']/2**20 for r in rr]),useful_gflops=gm([r['gflops'] for r in rr]),
     mac_util_pct=gm([r['mac_util_pct'] for r in rr]),broadcast_exposed_mean_ms=sum(r['broadcast_exposed_cycles'] for r in rr)/len(rr)/1e6,
     windows=len(rr),timing=f"pure compute, adequate ordinary supply; extra Shared broadcast at {rr[0]['broadcast_TBps']} TB/s (0 = ideal)"))
    cells[-1]['compute_mode']=rr[0]['mode']
    compare.append(dict(macs=m,batch=b,org=org,old_online_ms=gm([r['ms'] for r in old]),new_online_ms=on,offline_ms=off,
     old_over_offline=gm([r['ms'] for r in old])/off,new_over_offline=on/off,quota_ablation_ms=gm([r['ms'] for r in nq]),
     new_setup_mean_ms=sum(r['setup_cycles'] for r in nn)/len(nn)/1e6,
     new_assignment_critical_mean_ms=sum(r['owner_compute_cycles'] for r in nn)/len(nn)/1e6,
     offline_assignment_critical_mean_ms=sum(r['owner_compute_cycles'] for r in rr)/len(rr)/1e6,
     new_residual_wait_mean_ms=sum(r['cycles']-r['owner_compute_cycles']-r['setup_cycles'] for r in nn)/len(nn)/1e6,
     offline_residual_wait_mean_ms=sum(r['cycles']-r['owner_compute_cycles'] for r in rr)/len(rr)/1e6,
     quota_redirects_mean=sum(r['quota_redirects'] for r in nn)/len(nn),
     original_r5_no_broadcast_ms=gm([r['original_r5_zero_broadcast_cycles']/1e6 for r in old]),
     timing='pure compute, adequate ordinary supply; extra Shared broadcast10TB/s',
     old_definition='frozen R5 owners and routed order, replayed with common Shared multicast cohort'))
    offs={r['window_id']:r for r in rr}
    for r in nn:
     target=r['target_work'];actual=r['pool_work'];oracle=offs[r['window_id']]['pool_work'];total=sum(actual)
     balance.append(dict(macs=m,batch=b,org=org,window_id=r['window_id'],geometry=r['geometry'],
      actual_pool_work_pct=json.dumps([100*x/total for x in actual]),estimated_pool_target_pct=json.dumps([100*x/total for x in target]),
      offline_observed_pool_pct=json.dumps([100*x/total for x in oracle]),core_idle_pct=json.dumps(r['core_idle_pct']),
      core_finish_cycles=json.dumps(r['core_finish']),quota_redirects=r['quota_redirects'],quota_escapes=r['quota_escapes'],
      offline_targets_used_online=False))
 return cells,compare,balance

def figures(cells):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import LogNorm,ListedColormap
 from matplotlib.backends.backend_pdf import PdfPages
 lookup={(r['macs'],r['batch'],r['org']):r for r in cells};ss={(s['budget'],s['org']):s for s in selections()}
 chosen={(m,b):min(('P','P+'),key=lambda o:ss[m,o]['dev_by_batch'][str(b)]) for m in BUDGETS for b in BATCHES}
 path=ROOT/'figures';path.mkdir(exist_ok=True)
 def draw(org=None):
  fig,axes=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
  rs=[[lookup[m,b,org or chosen[m,b]] for b in BATCHES] for m in BUDGETS]
  arrays=[np.array([[r['req_weight_TBps'] for r in line] for line in rs]),np.array([[r['speedup_vs_strongest_homo_offline'] for r in line] for line in rs]),np.array([[r['crit_A10'] and r['crit_B'] and r['crit_C'] for r in line] for line in rs],int)]
  if org is None:
   arrays[2]=np.array([[any(all(lookup[m,b,o][k] for k in ('crit_A10','crit_B','crit_C')) for o in ('P','P+')) for b in BATCHES] for m in BUDGETS],int)
  titles=['Required weight bandwidth (TB/s)','Offline speedup / strongest homogeneous','A(10 TB/s) + B + C']
  if org is None:titles[2]='Any frozen P/P+ passes A + B + C'
  for i,(ax,a,title) in enumerate(zip(axes,arrays,titles)):
   kw=dict(norm=LogNorm(vmin=max(.1,a.min()),vmax=a.max()),cmap='viridis') if i==0 else dict(cmap='RdBu_r',vmin=.8,vmax=max(1.2,a.max())) if i==1 else dict(cmap=ListedColormap(['#eeeeee','#15803d']),vmin=0,vmax=1)
   im=ax.imshow(a,origin='lower',aspect='auto',**kw);ax.set_xticks(range(7),[str(b) for b in BATCHES],rotation=45);ax.set_yticks(range(5),['64K','128K','256K','512K','1M']);ax.set_xlabel('Prefill tokens / layer');ax.set_ylabel('Physical MAC budget');ax.set_title(title)
   for y in range(5):
    for x in range(7):
     rgba=im.cmap(im.norm(a[y,x]));lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
     label=('●' if a[y,x] else '–') if i==2 else f'{a[y,x]:.3f}' if i==1 else f'{a[y,x]:.2f}'
     ax.text(x,y,label,ha='center',va='center',fontsize=8.5,color='black' if lum>.55 or i==2 else 'white')
   levels=[3.3,10] if i==0 else [1.10] if i==1 else []
   for v in levels:
    if a.min()<v<a.max():
     cs=ax.contour(a,levels=[v],colors=['#ffcf00'],linewidths=1.5);ax.clabel(cs,fmt='%g',fontsize=8)
   if i!=2:fig.colorbar(im,ax=ax,shrink=.75)
  label=org or 'P/P+ selected by DEVELOPMENT batch score'
  fig.suptitle(f'{label}; frozen hardware across batches; adequate ordinary supply; extra Shared broadcast = 10 TB/s\nA is a necessary bandwidth condition, not a finite-memory E2E simulation.',fontsize=11)
  return fig
 with PdfPages(path/'fig_cells.pdf') as pdf:
  for org in (None,'P','P+'):
   fig=draw(org);pdf.savefig(fig)
   if org is None:fig.savefig(path/'fig_cells.png',dpi=160)
   plt.close(fig)

def build():
 strongest=strongest_map();cells,compare,balance=aggregate(load_results(),strongest)
 write_csv(ROOT/'E1/cells.csv',cells);write_csv(ROOT/'E2/online_compare.csv',compare);write_csv(ROOT/'E2/workload_balance.csv',balance)
 r5={(int(r['total_macs']),int(r['batch']),r['granularity']):r for r in read_csv(R5/'report/main.csv') if r['mode']=='full_scaled' and r['batch'].isdigit()}
 cmap={(r['macs'],r['batch'],r['org']):r for r in cells};effects=[]
 for m in BUDGETS:
  for b in BATCHES:
   a=cmap[m,b,'H']['offline_ms']/cmap[m,b,'P']['offline_ms'];z=cmap[m,b,'H+']['offline_ms']/cmap[m,b,'P']['offline_ms']
   row=dict(macs=m,batch=b,round6_H_over_P=a,round6_Hplus_over_P=z,round6_gain_removed_pp=100*(a-z),round6_strongest_homo=strongest[m,b],
    round6_strongest_over_P=cmap[m,b,'P']['speedup_vs_strongest_homo_offline'])
   if (m,b,'expert') in r5:
    x=r5[m,b,'expert'];y=r5[m,b,'shared64'];den=float(x['heterogeneous_ms'])
    row.update(round5_H_over_P=float(x['homogeneous_ms'])/den,round5_Hplus_over_same_P=float(y['homogeneous_ms'])/den,
     round5_gain_removed_pp=100*(float(x['homogeneous_ms'])-float(y['homogeneous_ms']))/den)
   else:row.update(round5_H_over_P='NA_new_batch',round5_Hplus_over_same_P='NA_new_batch',round5_gain_removed_pp='NA_new_batch')
   row['scope']='Within each round hold P whole-expert fixed and replace H with H+; R5 has ideal broadcast, R6 charges extra copies. Cross-round differences also include re-selection.'
   effects.append(row)
 write_csv(ROOT/'E1/shared_split_effect.csv',effects)
 matrix,_,_=aggregate(load_results('matrix'),strongest);write_csv(ROOT/'E1/matrix_appendix.csv',matrix)
 zero,_,_=aggregate(load_results(bw=0,tag='zero_broadcast'),strongest);write_csv(ROOT/'E1/zero_broadcast_appendix.csv',zero)
 hardware=[]
 for s in selections():hardware.append({k:v for k,v in s.items() if k not in ('cores','dev_by_batch','dev_by_batch_lb')})
 write_csv(ROOT/'E1/frozen_hardware.csv',hardware)
 refs=[]
 perdev={}
 for m in BUDGETS:
  for ch in (0,64):
   for s in read_json(ROOT/'E1/search'/f'{m}_{ch}'/'per_batch_selection.json'):perdev[m,s['batch'],s['org']]=s['dev_upper_cycles']
 for p in sorted((ROOT/'E1/windows').glob('*_per_batch_B*.json.gz')):
  rr=read_json(p);off=[r for r in rr if r['scheduler']=='offline'];on=[r for r in rr if r['scheduler']=='online']
  r=off[0];refs.append(dict(macs=r['macs'],batch=r['batch'],org=r['org'],geometry=r['geometry'],cores=r['cores'],offline_ms=gm([x['ms'] for x in off]),online_ms=gm([x['ms'] for x in on]),
   offline_lb_ms=gm([x['offline_lb_cycles']/1e6 for x in off]),req_weight_TBps=gm([x['req_weight_TBps'] for x in off]),req_total_TBps=gm([x['req_total_TBps'] for x in off]),
   peak_resident_MiB=max(x['peak_resident_MiB'] for x in off),scope='per-batch development selection; separate upper reference, not globally frozen hardware',timing='pure compute, adequate ordinary supply; extra Shared broadcast10TB/s'))
 refmap={(r['macs'],r['batch'],r['org']):r for r in refs}
 for r in refs:
  m,b=r['macs'],r['batch'];h=min(('H','H+'),key=lambda o:perdev[m,b,o]);base=refmap[m,b,h]
  r.update(strongest_per_batch_homo=h,speedup_offline=base['offline_ms']/r['offline_ms'],speedup_online=base['online_ms']/r['online_ms'],online_over_offline=r['online_ms']/r['offline_ms'])
  r.update(crit_A10=r['req_weight_TBps']<=10,crit_B=r['speedup_offline']>=1.10,crit_C=r['speedup_online']>=1.05 and r['online_over_offline']<=1.05)
 write_csv(ROOT/'E1/per_batch_hardware_reference.csv',refs)
 figures(cells)
 return cells
if __name__=='__main__':build()
