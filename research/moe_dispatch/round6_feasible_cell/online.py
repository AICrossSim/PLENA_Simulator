"""Work quotas from fractional pool capacities, no integer oracle mapping.

The escape clause compares against the projected LAYER drain deadline. Using
the current task's EFT would be tautological: every rejected alternative is
later than the earliest one, permanently disabling the quota.
"""
import math
import numpy as np
from .model import cohort
from .common import W_TBPS

def pool_targets(p,work,types,initial=None):
 """Continuous two-pool balance, return only aggregate work shares + horizon."""
 types=list(types);unique=list(dict.fromkeys(types));groups=[[c for c,t in enumerate(types) if t==q] for q in unique]
 initial=initial or [0]*len(types);total=sum(work)
 if len(groups)==1:
  return [float(total)],max(max(initial+[0]),(sum(initial)+sum(int(r[0]) for r in p))/len(types))
 assert len(groups)==2
 a,b=groups;ia,ib=a[0],b[0];na,nb=len(a),len(b)
 la=sum(initial[c] for c in a)/na;lb=(sum(initial[c] for c in b)+sum(int(row[ib]) for row in p))/nb
 qa=0.0
 if la>=lb:return [0.0,float(total)],la
 for i in sorted(range(len(p)),key=lambda i:(int(p[i,ia])/int(p[i,ib]),i)):
  da=int(p[i,ia])/na;db=int(p[i,ib])/nb
  fraction=min(1.0,max(0.0,(lb-la)/(da+db)))
  qa+=fraction*work[i];la+=fraction*da;lb-=fraction*db
  if fraction<1:return [qa,total-qa],max(la,lb)
 return [qa,total-qa],max(la,lb)

def shared_plan(p,ns,gu,dn,types,work,bw,new):
 """Plan a streamed Shared cohort before private expert issue.

New rule scans receiver counts with a fractional remaining-work forecast;
this does not call an integer solver or read its owner assignments.
"""
 n,nc=p.shape;rank=sorted(range(nc),key=lambda c:(gu[c]+dn[c],c));plans=[]
 choices=range(1,min(nc,ns)+1) if new else [min(nc,ns)]
 for k in choices:
  sc=[0]*nc;own=[]
  for j in range(ns):
   c=min(rank[:k],key=lambda c:((sc[c]+1)*(gu[c]+dn[c]),c));sc[c]+=1;own.append(c)
  bc=cohort(sc,gu,dn,bw);initial=[bc['cycles'] if sc[c] else 0 for c in range(nc)]
  _,forecast=pool_targets(p[:n-ns],work[:n-ns],types,initial)
  # Largest indivisible routed job and actual cohort completion remain floors.
  deadline=max(forecast,bc['cycles'],max((min(initial[c]+int(row[c]) for c in range(nc)) for row in p[:n-ns]),default=0))
  plans.append((deadline,bc['cycles'],k,sc,own,bc))
 return min(plans,key=lambda x:x[:3])

def dispatch(p,ns,gu,dn,types,tasks,delta=.05,new=True,bw=W_TBPS):
 p=np.asarray(p,np.int64);n,nc=p.shape;ns=int(ns);nr=n-ns
 work=[3*t['m']*2048*t['f'] for t in tasks];unique=list(dict.fromkeys(types));pool=[unique.index(t) for t in types]
 _,_,_,sc,shared_owner,bc=shared_plan(p,ns,gu,dn,types,work,bw,True)
 # A declared sequential-controller service model, not measured gate timing:
 # one (task,type) cost lookup/cycle; pipelined bitonic comparison stages;
 # one task per pool-balance scan. Shared receiver-count plans are evaluated
 # sequentially. Counting these prevents a free whole-layer oracle at t=0.
 levels=math.ceil(math.log2(max(1,nr)))
 estimate=n*len(unique)+levels*(levels+1)//2
 planning=(nr+ns)*min(nc,ns) if ns>1 else 0
 quota_scan=nr if len(unique)>1 else 0
 setup=ns+min(nc,ns)+2+estimate+planning+quota_scan
 initial=[bc['cycles'] if sc[c] else 0 for c in range(nc)]
 quota,forecast=pool_targets(p[:nr],work[:nr],types,initial)
 shared_work=[0]*len(unique)
 for j,c in enumerate(shared_owner):shared_work[pool[c]]+=work[nr+j]
 target=[q+s for q,s in zip(quota,shared_work)];accepted=shared_work.copy()
 deadline=setup+max(forecast,bc['cycles'])
 ends=[setup+initial[c] for c in range(nc)];queues=[[ends[c]] if sc[c] else [] for c in range(nc)]
 owner=[-1]*nr+shared_owner;starts=[-1]*nr+[setup]*ns
 pending=sorted(range(nr),key=lambda i:(-min(p[i]),i));now=setup;waits=0;quota_redirect=0;escapes=0;bypass=0
 while pending:
  for q in queues:
   while q and q[0]<=now:q.pop(0)
  choice=None
  for pos,i in enumerate(pending[:8]):
   opts=[]
   for c in range(nc):
    start=max(now+1,ends[c]);extra=max(0,2-(start-now-1));finish=start+extra+int(p[i,c])
    opts.append((finish,len(queues[c])>=2,c,start+extra))
   best=min(opts)
   if new and len(unique)>1:
    legal=[v for v in opts if accepted[pool[v[2]]]+work[i]<=(1+delta)*target[pool[v[2]]]+1e-6]
    if legal and best not in legal:
     alternative=min(legal)
     if alternative[0]<=(1+delta)*max(deadline,max(ends)):
      best=alternative;quota_redirect+=1
     else:escapes+=1
   finish,full,c,start=best
   if not full:choice=(i,c,start,finish);bypass+=int(pos>0);break
  if choice is None:
   future=[q[0] for q in queues if q];assert future and min(future)>now
   now=min(future);waits+=1;continue
  i,c,start,finish=choice;owner[i]=c;starts[i]=start;ends[c]=finish;queues[c].append(finish);pending.remove(i)
  accepted[pool[c]]+=work[i];now+=1
 cycles=max(ends)
 assert all(c>=0 for c in owner)
 return dict(cycles=cycles,owner=owner,starts=starts,core_finish=ends,shared_counts=sc,broadcast=bc,
  target_work=target,actual_work=accepted,quota_redirects=quota_redirect,quota_escapes=escapes,
  admission_waits=waits,lookahead_bypasses=bypass,cohort_setup_cycles=setup,
  estimator_cycles=estimate,shared_planner_cycles=planning,quota_scan_cycles=quota_scan,
  quota_target_source='fractional aggregate remaining-work balance; no offline integer assignments')

def old_dispatch(p,ns,gu,dn,types,tasks,bw=W_TBPS):
 """Frozen R5 LPT/EFT owners, replayed under the common streamed cohort.

 R5's original ordering cannot broadcast Shared blocks in a bounded buffer:
 receivers would demand different blocks at different times. Keep its owners
 and each core's routed order; move its Shared shards to the common initial
 cohort, and expose the precise compatibility adjustment in output metadata.
 """
 from research.moe_dispatch.round5_largebatch_compute.runtime_fix import admission
 from .model import replay
 r=admission(p,tasks,types);q=replay(p,ns,gu,dn,r['owner'],bw)
 levels=math.ceil(math.log2(max(1,len(tasks)-ns)))
 setup=ns+2+len(tasks)*len(set(types))+levels*(levels+1)//2;ends=[];starts=[setup]*len(tasks)
 for c in range(p.shape[1]):
  t=setup+(q['broadcast']['cycles'] if q['shared_counts'][c] else 0)
  for i in sorted((i for i in range(len(tasks)-ns) if r['owner'][i]==c),key=lambda i:r['starts'][i]):
   t=max(t,r['starts'][i]);starts[i]=t;t+=int(p[i,c])
  ends.append(t)
 return dict(cycles=max(ends),owner=r['owner'],starts=starts,core_finish=ends,shared_counts=q['shared_counts'],broadcast=q['broadcast'],
  original_r5_zero_broadcast_cycles=r['cycles'],cohort_setup_cycles=setup,
  contract='R5 admission owners and routed order; Shared phase moved to common streaming cohort for finite-buffer multicast')
