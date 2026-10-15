import math
from dataclasses import asdict,replace
import numpy as np
from .config import Core,parameters,TOTAL_MACS
from .compute import cost,pure,allocate_real
from .common import inputs,canonical
from .model import task_cost
from .runtime import simulate
from .search import candidates
from .crossover import crossing

def test_exact_domain_and_ratios():
    for cap in (256,512):
        count=0
        for family,cs,fs in candidates(cap):
            count+=1
            assert sum(c.macs for c in cs)==TOTAL_MACS
            assert all(c.pn<=cap for c in cs)
            assert family not in ('hetero21','hetero51')
            if family=='hetero11': assert cs[0]!=cs[1] and cs[0].macs==cs[1].macs
            if family.startswith('homo'):assert len(set(cs))==1 and len(set(fs))==1
        assert count>28000

def test_ports_and_capacity_conserved_four_cores():
    cs=(Core(1,64,256),)*4;d=allocate_real(cs,('WS',)*4)
    assert sum(sum(getattr(d,k)) for k in ('w_bytes','x_bytes','acc_bytes','z_bytes'))==532*1024
    assert (sum(d.w_banks),sum(d.x_banks),sum(d.acc_banks),sum(d.vector_lanes))==(64,24,12,64)
    assert d.ledger()['installed_storage_B']==2158592

def test_credit_capacity_reaches_requested_bandwidth():
    for b in (256,512,1229,2458,9830):
        p=parameters(b);assert p.hbm_bandwidth==b
        assert p.credits*32>=b*65
    assert math.isinf(parameters().hbm_bandwidth)

def test_flow_choice_no_worse_in_ideal_task():
    c=Core(1,128,512)
    for m in (1,2,8,128):
        qs=[cost(c,f,m,2048,1408,64) for f in ('OS','WS','IS')]
        assert cost(c,'PER_TASK',m,2048,1408,64)['cycles']==min(q['cycles'] for q in qs)
        assert all(q['macs']==3*m*2048*1408 for q in qs)
        assert all(q['issued']>=q['macs'] for q in qs)

def test_single_row_needs_no_whole_weight_cache():
    c=Core(1,128,512)
    assert cost(c,'IS',1,2048,1408,64)['W']==c.w_slice_bytes
    assert cost(c,'IS',2,2048,1408,64)['W']>c.w_slice_bytes

def test_refetch_present_only_when_loop_needs_it():
    c=Core(1,128,512);d=allocate_real((c,),('OS',))
    e=dict(Me=8,H=2048,F=1408)
    os=task_cost(e,d,0,parameters());ws=task_cost(e,replace(d,flows=('WS',)),0,parameters())
    assert os.hbm_bytes>=8*os.unique_hbm_bytes
    assert ws.hbm_bytes<os.hbm_bytes

def test_real_four_core_expert_coverage():
    w=inputs()['heldout'][0];d=allocate_real((Core(1,64,256),)*4,('WS',)*4)
    r=simulate(w,d,parameters(256),dispatch='fixed')
    assert sorted(t['expert_index'] for t in r['tasks'])==list(range(len(w['experts'])))
    assert r['useful_macs']==sum(3*e['Me']*e['H']*e['F'] for e in w['experts'])
    assert r['cycles']>=r['hbm_bytes']/256
    assert canonical(r)==canonical(simulate(w,d,parameters(256),dispatch='fixed'))

def test_frozen_engine_regression_two_cores():
    from research.moe_dispatch.round3.model import Design as OldDesign,Core as OldCore
    from research.moe_dispatch.round3.runtime import simulate as old_simulate
    from research.moe_dispatch.round3.config import parameters as old_parameters
    from .model import Design
    old=OldDesign((OldCore(3,16,128),OldCore(3,16,128)),flows=('WS','WS'))
    z=asdict(old);z['cores']=tuple(Core(**c) for c in z['cores']);new=Design(**z)
    p=old_parameters();w=inputs()['heldout'][0]
    assert canonical(simulate(w,new,p))==canonical(old_simulate(w,old,p))

def test_exact_crossover_and_nonexistence():
    c=[dict(Tc_ms=1,hbm_bytes=1e6)];b=[dict(Tc_ms=2,hbm_bytes=1e6)]
    x=crossing(c,[b],.1)
    assert abs(x-1/(2*.9))<1e-9
    assert crossing(c,[c],.05) is None
    # Traffic advantage can disappear at high BW: do not binary-search a
    # monotonicity assumption that does not hold for this ratio.
    slow=[dict(Tc_ms=4,hbm_bytes=5e5)]
    assert crossing(slow,[b],.1)==0

def test_bf16_exact_integer_tiling_and_padding():
    rng=np.random.default_rng(16)
    for shape in ((1,128,512),(3,96,64),(16,256,32),(2,512,64)):
        pm,pn,pk=shape;m,n,k=3,137,1408
        # Small integers are exactly BF16 representable; all sums fit FP32.
        x=rng.integers(-2,3,(m,k)).astype(np.float32)
        w=rng.integers(-2,3,(k,n)).astype(np.float32)
        out=np.zeros((m,n),np.float32)
        for mi in range(0,m,pm):
            for ni in range(0,n,pn):
                for ki in range(0,k,pk):
                    out[mi:mi+pm,ni:ni+pn]+=x[mi:mi+pm,ki:ki+pk]@w[ki:ki+pk,ni:ni+pn]
        assert np.array_equal(out,x.astype(np.int64)@w.astype(np.int64))
