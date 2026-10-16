import itertools
import numpy as np
from .inputs import load
from .model import configurations,phase_duration,cohort,replay,Core,cost,phases,tasks
from .solver import fast,optimize
from .online import dispatch,pool_targets

def test_real_routes_and_splits():
 data=load()
 for ws in data.values():
  assert len(ws)==105
  for w in ws:
   routes=np.asarray(w['token_routes']);assert routes.shape==(w['batch'],6)
   assert all(len(set(row))==6 for row in routes)
   h=np.bincount(routes.ravel(),minlength=64)
   assert sum(h)==6*w['batch']
   assert sum(e['Me'] for e in w['experts'] if e['is_shared'])==w['batch']
   assert all(h[e['id']]==e['Me'] for e in w['experts'] if not e['is_shared'])
   if w['batch']==4096:
    sources=[next(x for x in ws if x['id']==s) for s in w['source_windows']]
    assert w['token_routes']==sources[0]['token_routes']+sources[1]['token_routes']
 a={r for w in data['development'] for r in w['source_requests']}
 assert not a&{r for w in data['heldout'] for r in w['source_requests']}

def test_budget_and_shapes():
 for budget in (65536,131072,262144,524288,1048576):
  configs=list(configurations(budget));assert {len(c) for f,c in configs if f=='H'}=={2,4,8,12,13,16}
  for f,cs in configs:
   n=sum(c.macs for c in cs)
   assert n<=budget
   assert n==budget or (f=='H' and len(cs) in (12,13))
   assert len(set(cs))==(1 if f in ('S','H') else 2)

def test_two_buffer_formula():
 for n,C,d in itertools.product(range(1,12),range(1,100),range(0,20)):
  q,r=divmod(C,n);end=0
  # producer cannot get more than one complete block ahead of consumer;
  # each block transfer uses the buffer freed at compute START of the last.
  arrival=0;last_start=0
  for i in range(n):
   arrival=max(arrival,last_start)+d
   start=max(end,arrival);end=start+q+(i<r);last_start=start
  assert end==phase_duration(C,n,d)

def test_exact_schedule_and_bounds():
 rng=np.random.default_rng(7)
 for nc,ns in [(2,1),(2,2),(3,2)]:
  for _ in range(4):
   p=rng.integers(50,1500,(5,nc));g=rng.integers(50,300,nc);dn=rng.integers(50,300,nc)
   p[-ns:]=g+dn;typ=list(range(nc)) if nc==2 else [0,0,1]
   if nc==3:p[:,1]=p[:,0];g[1]=g[0];dn[1]=dn[0]
   for bw in (0,10):
    truth=min(replay(p,ns,g,dn,o,bw)['cycles'] for o in itertools.product(range(nc),repeat=5))
    lo,hi,o=fast(p,ns,g,dn,typ,bw);assert lo<=truth<=hi
    a=optimize(p,ns,g,dn,typ,bw,work=2);b=optimize(p,ns,g,dn,typ,bw,work=2)
    assert a==b and a['upper']==truth and a['lower']==truth

def test_phase_contract_and_single_core():
 w=load()['heldout'][0];ts=tasks(w)
 for c in (Core(1,128,512),Core(4,128,64)):
  p=np.array([[cost(c,t['m'],t['f'],'full_scaled')['cycles']] for t in ts]);g,dn=phases(c,ts[-1],'full_scaled',65536)
  r=replay(p,1,[g],[dn],[0]*len(ts));assert r['cycles']==sum(p[:,0])

def test_online_repeat_and_nonoverlap():
 w=load()['heldout'][0];ts=tasks(w,64);cs=[Core(4,128,64),Core(1,128,64),Core(1,128,64)]
 p=np.array([[cost(c,t['m'],t['f'],'full_scaled')['cycles'] for c in cs] for t in ts]);ns=sum(t['shared'] for t in ts)
 gu,dn=np.array([phases(c,ts[-1],'full_scaled',65536) for c in cs]).T
 a=dispatch(p,ns,gu,dn,[0,1,1],ts);assert a==dispatch(p,ns,gu,dn,[0,1,1],ts)
 assert sum(a['actual_work'])==sum(3*t['m']*2048*t['f'] for t in ts)
 for c in range(3):
  spans=sorted((a['starts'][i],a['starts'][i]+p[i,c]) for i in range(len(ts)-ns) if a['owner'][i]==c)
  assert all(x[1]<=y[0] for x,y in zip(spans,spans[1:]))
  assert not spans or spans[0][0]>=a['cohort_setup_cycles']+(a['broadcast']['cycles'] if a['shared_counts'][c] else 0)

def test_fractional_targets_match_linear_program():
 from scipy.optimize import linprog
 rng=np.random.default_rng(17)
 for na,nb in ((1,1),(1,12),(4,3)):
  for _ in range(10):
   a=rng.integers(10,500,20);b=rng.integers(10,500,20);p=np.column_stack([a]*na+[b]*nb)
   share,t=pool_targets(p,[1]*20,[0]*na+[1]*nb)
   A=np.zeros((2,21));A[0,:20]=a/na;A[1,:20]=-b/nb;A[:,-1]=-1
   lp=linprog([0]*20+[1],A_ub=A,b_ub=[0,-sum(b)/nb],bounds=[(0,1)]*20+[(0,None)],method='highs')
   assert lp.success and abs(lp.fun-t)<1e-6
   assert abs(sum(share)-20)<1e-8

def test_peak_residency_uses_tiles_not_full_experts():
 from .evaluate import resident_peak
 cs=[Core(1,128,256)]*2;p=np.array([[100,100],[100,100]])
 r=dict(owner=[0,1],starts=[0,0],shared_counts=[1,1],broadcast=cohort([1,1],[60,60],[40,40]),cohort_setup_cycles=0)
 peak=resident_peak(p,2,cs,r)
 assert peak==5*262144 and peak<33*2**20
