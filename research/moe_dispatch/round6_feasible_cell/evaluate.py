import argparse,math,time,hashlib,os,threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .common import *
from .inputs import load
from .model import Core,cost,make_costs,INDEX,matrix_for,replay
from .solver import optimize,fast
from .online import dispatch,old_dispatch
from .search import selection_weights

def offline_point(p,ns,g,d,ix,bw):
 h=hashlib.sha256(p.tobytes()+np.asarray(g).tobytes()+np.asarray(d).tobytes()+str((ns,ix,bw,.3)).encode()+
  (ROOT/'solver.py').read_bytes()+(ROOT/'kernel.cpp').read_bytes()).hexdigest()
 path=CACHE/'evaluation_offline'/f'{h}.json'
 if path.exists():return read_json(path)
 a=optimize(p,ns,g,d,ix,bw,work=.3);b=optimize(p,ns,g,d,ix,bw,work=.3);assert a==b
 temp=path.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp');write_json(temp,a);os.replace(temp,path)
 return a

def prewarm(budget):
 ws=load()['heldout']
 for chunk in (0,64):
  base=ROOT/'E1/search'/f'{budget}_{chunk}'
  configs=list(read_json(base/'selection.json').values())
  per=read_json(base/'per_batch_selection.json')
  for mode,bw,ss in [('full_scaled',10,configs+per),('matrix',10,configs),('full_scaled',0,configs)]:
   data,counts,ts,ns,gu,dn=make_costs(ws,chunk,mode,budget)
   for s in ss:
    ix=[INDEX[Core(**c)] for c in s['cores']];jobs=[j for j,w in enumerate(ws) if 'batch' not in s or w['batch']==s['batch']]
    def job(j):return offline_point(matrix_for(data,ix,j,counts),int(ns[j]),gu[ix,j],dn[ix,j],ix,bw)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(job,jobs))
   print('PREWARM',budget,chunk,mode,bw,flush=True)

def selections():
 out=[]
 for b in BUDGETS:
  for chunk in (0,64):out+=list(read_json(ROOT/'E1/search'/f'{b}_{chunk}'/'selection.json').values())
 return out

def tune_delta():
 path=ROOT/'E2/delta_selection.json'
 if path.exists():return read_json(path)['delta']
 ws=load()['development'];wt=selection_weights(ws);rows=[];objectives={d:[] for d in (.02,.05,.10,.20)}
 for b in BUDGETS:
  for chunk in (0,64):
   ss=read_json(ROOT/'E1/search'/f'{b}_{chunk}'/'selection.json')
   data,counts,ts,ns,gu,dn=make_costs(ws,chunk,budget=b)
   for org,s in ss.items():
    cs=[Core(**c) for c in s['cores']];ix=[INDEX[c] for c in cs]
    for delta in objectives:
     vals=[]
     for w in range(len(ws)):
      p=matrix_for(data,ix,w,counts);a=dispatch(p,ns[w],gu[ix,w],dn[ix,w],ix,ts[w],delta)
      assert a==dispatch(p,ns[w],gu[ix,w],dn[ix,w],ix,ts[w],delta)
      vals.append(a['cycles'])
      rows.append(dict(macs=b,org=org,window_id=ws[w]['id'],batch=ws[w]['batch'],delta=delta,cycles=a['cycles']))
     obj=float(np.exp(np.log(vals)@wt));objectives[delta].append(obj/s['dev_upper_cycles'])
   print('DELTA',b,chunk,flush=True)
 scores={d:gm(v) for d,v in objectives.items()};chosen=min(scores,key=lambda d:(scores[d],d))
 write_csv(ROOT/'E2/delta_development.csv.gz',rows)
 write_json(path,dict(delta=chosen,objective={str(d):v for d,v in scores.items()},objective_definition='Paired online/offline GM, equal weight per budget, organization and batch; development only',all_points_repeated_exactly=True))
 return chosen

def resident_peak(p,ns,cs,result):
 """Peak allocated live double buffers on actual task intervals.

 Shared streams use 2x256KiB per receiver and one source block. Routed
 experts use two natural BF16 W tiles per active core. Buffers are allocated
 at task start and freed at task end; this excludes activation/accumulator
 storage and therefore is a weight-buffer requirement, not total SRAM.
 """
 events=[];setup=result.get('cohort_setup_cycles',0);bc=result['broadcast']['cycles']
 for c,core in enumerate(cs):
  if result['shared_counts'][c]:events.extend([(setup,2*BLOCK),(setup+bc,-2*BLOCK)])
 if result['broadcast']['participants']>1:events.extend([(setup,BLOCK),(setup+bc,-BLOCK)])
 if 'starts' in result:starts=result['starts']
 else:
  starts=[0]*len(p);at=[bc if x else 0 for x in result['shared_counts']]
  for i,c in enumerate(result['owner'][:len(p)-ns]):starts[i]=at[c];at[c]+=int(p[i,c])
 for i,c in enumerate(result['owner'][:len(p)-ns]):
  size=4*cs[c].pn*cs[c].pk;events.extend([(starts[i],size),(starts[i]+int(p[i,c]),-size)])
 cur=peak=0
 for t,d in sorted(events,key=lambda x:(x[0],x[1])):cur+=d;peak=max(peak,cur)
 assert cur==0
 return peak

def detail(w,ts,cs,p,ns,result,mode,budget):
 useful=sum(3*t['m']*2048*t['f'] for t in ts);unique=sum(6*2048*e['F'] for e in w['experts'])
 boundary=sum(6*e['Me']*2048 for e in w['experts']) # BF16 input read + FP32 output write.
 cycles=int(result['cycles']);owner=result['owner'];services=[0]*len(cs);work=[0]*len(cs);issued=0;matrix=0;vector=0;flows={}
 for i,c in enumerate(owner):
  q=cost(cs[c],ts[i]['m'],ts[i]['f'],mode,budget);services[c]+=int(p[i,c]);work[c]+=q['useful'];issued+=q['issued'];matrix+=q['matrix_cycles'];vector+=q['vector_excess'];flows[q['flow']]=flows.get(q['flow'],0)+1
 unique_types=list(dict.fromkeys(cs));poolwork=[sum(work[c] for c in range(len(cs)) if cs[c]==t) for t in unique_types]
 rr=replay(p,ns,[0]*len(cs),[0]*len(cs),owner,0) if False else None
 compute_lower=max(services) # actual owner, with no broadcast / cohort / dispatch delay.
 return dict(cycles=cycles,ms=cycles/1e6,weight_bytes=unique,total_boundary_bytes=unique+boundary,
  req_weight_TBps=unique/cycles/1000,req_total_TBps=(unique+boundary)/cycles/1000,
  req_weight_and_replicas_TBps=(unique+result['broadcast']['extra_bytes'])/cycles/1000,
  req_boundary_and_replicas_TBps=(unique+boundary+result['broadcast']['extra_bytes'])/cycles/1000,
  req_weight_at_owner_compute_TBps=unique/compute_lower/1000,
  peak_resident_MiB=resident_peak(p,ns,cs,result)/2**20,
  useful_flops=2*useful,gflops=2*useful/cycles,mac_util_pct=100*useful/(budget*cycles),
  core_service=services,core_finish=result['core_finish'],core_work=work,pool_work=poolwork,
  core_idle_pct=[100*(1-s/cycles) for s in services],padding_waste_pct=100*(1-useful/issued),
  matrix_service_sum=matrix,vector_excess_sum=vector,flow_counts=flows,
  broadcast_extra_bytes=result['broadcast']['extra_bytes'],broadcast_exposed_cycles=result['broadcast']['broadcast_exposed_cycles'],
  shared_participants=result['broadcast']['participants'],owner_compute_cycles=compute_lower)

def evaluate_design(s,ws,delta,mode='full_scaled',bw=10,tag='frozen',only_batch=None):
 b=s['budget'];chunk=s['chunk'];org=s['org'];cs=[Core(**c) for c in s['cores']];ix=[INDEX[c] for c in cs]
 name=f'{b}_{org}_{mode}_{bw}_{tag}'+(f'_B{only_batch}' if only_batch else '')
 path=ROOT/'E1/windows'/f'{name}.json.gz'
 if path.exists():
  cached=read_json(path)
  assert all(r.get('delta')==delta and r.get('frozen_selection_digest')==digest(s) for r in cached),'cached parameter mismatch'
  return cached
 data,counts,ts,ns,gu,dn=make_costs(ws,chunk,mode,b)
 def job(j):
  w=ws[j];p=matrix_for(data,ix,j,counts);g=gu[ix,j];d=dn[ix,j];n=int(ns[j])
  offline=offline_point(p,n,g,d,ix,bw)
  new=dispatch(p,n,g,d,ix,ts[j],delta,bw=bw);assert new==dispatch(p,n,g,d,ix,ts[j],delta,bw=bw)
  old=old_dispatch(p,n,g,d,ix,ts[j],bw);assert old==old_dispatch(p,n,g,d,ix,ts[j],bw)
  noquota=dispatch(p,n,g,d,ix,ts[j],delta,new=False,bw=bw);assert noquota==dispatch(p,n,g,d,ix,ts[j],delta,new=False,bw=bw)
  # An online owner mapping is also a feasible offline incumbent. Never
  # publish an offline upper bound worse than a schedule already observed.
  for r in (new,old,noquota):
   v=replay(p,n,g,d,r['owner'],bw)['cycles']
   if v<offline['upper']:offline.update(upper=v,owner=r['owner'],status=offline['status']+'+runtime_incumbent')
  assert offline['lower']<=offline['upper']<=new['cycles']
  off=dict(owner=offline['owner'],**replay(p,n,g,d,offline['owner'],bw))
  base=dict(macs=b,batch=w['batch'],org=org,window_id=w['id'],group=w['group'],source_groups=w['source_groups'],geometry=s['geometry'],
   cores=len(cs),used_macs=s['used_macs'],granularity='shared64' if chunk else 'expert',mode=mode,broadcast_TBps=bw,selection=tag,delta=delta,frozen_selection_digest=digest(s),
   offline_lb_cycles=offline['lower'],offline_gap_pct=100*(offline['upper']/offline['lower']-1),solver_status=offline['status'])
  rows=[]
  for name,r in [('offline',off),('old',old),('online',new),('quota_ablation',noquota)]:
   rows.append(dict(base,scheduler=name,owner=r['owner'],**detail(w,ts[j],cs,p,n,r,mode,b),
    target_work=r.get('target_work'),quota_redirects=r.get('quota_redirects',0),quota_escapes=r.get('quota_escapes',0),
    setup_cycles=r.get('cohort_setup_cycles',0),estimator_cycles=r.get('estimator_cycles',0),shared_planner_cycles=r.get('shared_planner_cycles',0),quota_scan_cycles=r.get('quota_scan_cycles',0),
    original_r5_zero_broadcast_cycles=r.get('original_r5_zero_broadcast_cycles')))
  return rows
 chosen=[j for j,w in enumerate(ws) if only_batch is None or w['batch']==only_batch]
 with ThreadPoolExecutor(max_workers=4) as pool:rows=[r for result in pool.map(job,chosen) for r in result]
 write_json(path,rows)
 write_json(ROOT/'E1/receipts'/f'{name}.json',dict(windows=len(chosen),schedulers=4,repeats=2,bitwise_equal=True,rows_digest=digest(rows),mode=mode,bw=bw,delta=delta,selection_digest=digest(s)))
 print('EVALUATE',name,len(chosen),flush=True)
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--appendix',action='store_true');ap.add_argument('--budget',type=int);ap.add_argument('--tune-only',action='store_true');ap.add_argument('--prewarm',action='store_true');a=ap.parse_args()
 if a.prewarm:
  assert a.budget;prewarm(a.budget);return
 delta=tune_delta();ws=load()['heldout']
 if a.tune_only:return
 for s in selections():
  if a.budget and s['budget']!=a.budget:continue
  evaluate_design(s,ws,delta)
  evaluate_design(s,ws,delta,mode='matrix')
  evaluate_design(s,ws,delta,bw=0,tag='zero_broadcast')
 if a.appendix:
  for b in BUDGETS:
   if a.budget and b!=a.budget:continue
   for chunk in (0,64):
    for s in read_json(ROOT/'E1/search'/f'{b}_{chunk}'/'per_batch_selection.json'):
     evaluate_design(s,ws,delta,tag='per_batch',only_batch=s['batch'])
if __name__=='__main__':main()
