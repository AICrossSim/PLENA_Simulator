import itertools, math
import numpy as np
from .model import Core,cost,configurations,projection,build_tasks
from .schedule import fast,optimize,online

def test_resource_conservation_and_common_domain():
    for budget in (65536,131072,262144,524288,1048576):
        seen=set()
        for family,cs in configurations(budget):
            assert sum(c.macs for c in cs)==budget
            assert all(c.pn<=256 and c.pm<=64 and c.pk<=512 for c in cs)
            key=tuple(sorted(cs));assert key not in seen;seen.add(key)

def test_projection_matches_round4_contract():
    from research.moe_dispatch.round4_compute_first.compute import IdealView
    from research.moe_dispatch.round4_compute_first.config import parameters
    from research.moe_dispatch.round4_compute_first.model import _projection
    for c in (Core(1,128,512),Core(4,128,128),Core(2,96,256)):
        for m in (1,2,17,64,128,512,2048):
            for flow in ('OS','WS','IS'):
                for n,k,paired in ((1408,2048,True),(2048,2816,False)):
                    view=IdealView((c,),(flow,))
                    a=_projection(m,n,k,view,0,parameters(control=False),paired)
                    b=projection(c,m,n,k,paired,flow)
                    assert a.compute_cycles==b['cycles']
                    assert a.issued_macs==b['issued']

def test_no_flops_padding_credit_and_vector_resource():
    for m in (1,65,2048):
        for c in (Core(4,128,64),Core(1,96,512)):
            x=cost(c,m,1408);y=cost(c,m,1408,'full_scaled')
            assert x['issued']>=x['useful']==3*m*2048*1408
            assert y['cycles']>=x['cycles']

def brute(p):
    return min(max(sum(p[i,c] for i in range(len(p)) if a[i]==c) for c in range(p.shape[1]))
               for a in itertools.product(range(p.shape[1]),repeat=len(p)))

def test_bounds_and_integer_optimizer_against_bruteforce():
    rng=np.random.default_rng(7)
    for ncore in (1,2,3,4):
        for _ in range(8):
            types=[0]*(ncore//2)+[1]*(ncore-ncore//2)
            q=rng.integers(1,40,(6,2));p=q[:,types]
            lb,ub,o=fast(p,types);truth=brute(p)
            assert lb<=truth+1e-6<=ub+1e-6
            assert max(sum(p[i,c] for i in range(6) if o[i]==c) for c in range(ncore))==ub
            r=optimize(p,types,work=.5)
            assert r['lower']<=truth<=r['upper']
            assert r['upper']==truth

def test_runtime_is_legal_and_does_not_use_future_oracle():
    p=np.array([[40,60],[20,5],[20,5],[30,30],[6,3],[2,4],[9,1],[10,10],[11,2]])
    tasks=[{} for _ in p];r=online(p,tasks,[0,1]);q=online(p,tasks,[0,1]);assert r==q
    for c in (0,1):
        ends=sorted((r['starts'][i],r['starts'][i]+p[i,c]) for i in range(len(p)) if r['owner'][i]==c)
        assert all(a[1]<=b[0] for a,b in zip(ends,ends[1:]))
    assert r['cycles']>=fast(p,[0,1])[0]

def test_shared_shards_conserve_work():
    w={'experts':[{'id':64,'Me':129,'F':2816,'is_shared':True},{'id':0,'Me':7,'F':1408,'is_shared':False}]}
    t=build_tasks(w,64);assert [x['m'] for x in t]==[64,64,1,7]
    assert sum(3*x['m']*2048*x['f'] for x in t)==3*2048*(129*2816+7*1408)

def test_exact_tiled_numeric_reduction():
    # Small integer inputs are exactly BF16-representable; FP32 accumulation
    # checks tile tails and reduction coverage, not full-model quality.
    rng=np.random.default_rng(16);x=rng.integers(-3,4,(7,19)).astype(np.float32);w=rng.integers(-3,4,(19,23)).astype(np.float32)
    for pm,pn,pk in ((1,8,4),(4,16,8),(16,32,32)):
        y=np.zeros((7,23),np.float32)
        for m in range(0,7,pm):
            for n in range(0,23,pn):
                for k in range(0,19,pk):y[m:m+pm,n:n+pn]+=x[m:m+pm,k:k+pk]@w[k:k+pk,n:n+pn]
        np.testing.assert_array_equal(y,x@w)

def test_vectorized_cost_table_matches_scalar_contract():
    from .search import costs,ALL_SHAPES
    windows=[{'batch':b,'experts':[{'id':64,'Me':b,'F':2816,'is_shared':True},
              {'id':0,'Me':max(1,b//7),'F':1408,'is_shared':False}]} for b in (64,128,2048)]
    for mode in ('matrix','full_scaled','full_fixed'):
        for chunk in (0,64):
            a,counts,tasks=costs(windows,mode,chunk,262144)
            for ci,c in enumerate(ALL_SHAPES):
                for d,ts in enumerate(tasks):
                    for i,t in enumerate(ts):
                        assert a[ci,d,i]==cost(c,t['m'],t['f'],mode,262144)['cycles']

def test_equal_batch_objective():
    from .search import weights,objective
    windows=[{'batch':b} for b in (64,128,256,512,1024,2048) for _ in range(1 if b>128 else 7)]
    w=weights(windows)
    for b in (64,128,256,512,1024,2048):
        assert abs(sum(x for x,y in zip(w,windows) if y['batch']==b)-1/6)<1e-12
    assert abs(objective(np.full((1,len(w)),100.),w)[0]-100)<1e-10

def test_runtime_priority_and_waiting_admission():
    from .runtime_fix import admission
    p=np.array([[100,1000],[90,1000],[80,1000],[1,1000]])
    r=admission(p,[{}]*4,[0,1])
    assert r['owner']==[0,0,0,0] and r['cycles']==274
    assert r['starts'][0]==3 and r['admission_wait_events']>0
    # Renaming cores cannot change the latency or give a preferred label
    # extra capabilities. All work stays on the physically faster core.
    q=admission(p[:,::-1],[{}]*4,[1,0])
    assert q['cycles']==r['cycles'] and q['owner']==[1,1,1,1]

def test_runtime_priority_keeps_large_task_first_and_all_cores_busy():
    from .runtime_fix import admission
    p=np.array([[100,400]]+[[30,40]]*7)
    r=admission(p,[{}]*8,[0,1]);assert r==admission(p,[{}]*8,[0,1])
    assert r['owner'][0]==0 and r['starts'][0]==3
    assert set(r['owner'])=={0,1}
    for c in (0,1):
        spans=sorted((r['starts'][i],r['starts'][i]+p[i,c]) for i in range(8) if r['owner'][i]==c)
        assert all(a[1]<=b[0] for a,b in zip(spans,spans[1:]))
