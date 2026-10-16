import itertools
import random
from .common import CFG
from .model import Core, configurations, gemm, task_cost
from .scheduler import optimal, greedy
from .trace import load_inputs,validate_window

def test_formula_hand_calculation():
    c=Core(4,256)
    # M3 ->1, N1408 ->6, K2048 ->32, no padded MACs counted useful.
    q=gemm(c,3,2048,1408,'issue_formula',16)
    assert q['issues']==192 and q['cycles']==208
    assert q['sram_bytes']==192*(256+4)*64*2
    assert q['useful_macs']==3*2048*1408
    t=task_cost(c,3,'issue_formula',0)
    assert t['cycles']==192*2+8*22==560
    assert task_cost(c,3,'issue_formula',16)['cycles']==608

def test_reused_os_projection():
    from research.moe_dispatch.round5_largebatch_compute.model import Core as Old,projection
    c=Core(7,128)
    for m in (1,2,17,128):
        p=projection(Old(7,128,64),m,1408,2048,False,'OS')
        assert gemm(c,m,2048,1408,'plena_os',0)['cycles']==p['cycles']

def test_exact_matches_exhaustive():
    rng=random.Random(20261016)
    for n in range(1,11):
        for _ in range(12):
            p=[[rng.randint(1,80),rng.randint(1,80)] for _ in range(n)]
            truth=min(max(sum(p[i][0] for i,c in enumerate(o) if c==0),
                          sum(p[i][1] for i,c in enumerate(o) if c==1))
                      for o in itertools.product((0,1),repeat=n))
            result=optimal(p)
            assert result['cycles']==truth and result==optimal(p)
            assert greedy(p)['cycles']>=truth

def test_exact_large_duplicate_groups_and_66_bit_mask():
    p=[[10,7]]*66
    r=optimal(p)
    assert r['cycles']==min(max(10*k,7*(66-k)) for k in range(67))
    assert len(r['owner'])==66

def test_all_shapes_legal_and_complete():
    for cap in CFG['pn_caps']:
        configs=list(configurations(cap));keys=set()
        for org,cs in configs:
            assert cs not in keys;keys.add(cs)
            assert sum(c.macs for c in cs)==65536
            assert all(c.pm>=1 and c.pn%16==0 and c.pn<=cap and c.g==64 for c in cs)
            if org=='homo':assert cs[0]==cs[1] and cs[0].macs==32768
            if org=='hetero':assert cs[0]!=cs[1] and cs[1].macs in CFG['small_core_macs']
        # Independent brute enumeration of PM verifies no artificial PM cap.
        for budget in (65536,32768,8192,16384,24576,40960,49152,57344):
            from .model import shapes
            oracle={Core(m,n) for m in range(1,65) for n in range(16,cap+1,16) if m*n*64==budget}
            assert set(shapes(budget,cap))==oracle

def test_real_top6_and_two_shared_conservation():
    ws,meta=load_inputs(False)
    assert len(ws)==135 and meta['all_top6_checks_pass']
    for w in ws:
        assert sum(w['counts'])==6*w['batch']
        assert len(w['tasks'])==2+sum(m>0 for m in w['counts'])
        assert sum(t['m'] for t in w['tasks'])==8*w['batch']

def test_units_and_overhead_traffic_exclusion():
    from .metrics import window_metrics
    w=dict(id='hand',batch=2,tasks=[dict(m=2)]*8)
    cs=(Core(4,256),)
    for overhead in (0,16):
        p=[[task_cost(cs[0],2,'issue_formula',overhead)['cycles']]]*8
        r=window_metrics(cs,w,'issue_formula',overhead,optimal(p))
        assert r['useful_MACs']==8*2*3*2048*1408
        assert abs(r['throughput_TFLOPS']/131.072*100-r['utilization_pct'])<1e-10
        assert r['sram_MB']==8*560*(256+4)*64*2/1e6
