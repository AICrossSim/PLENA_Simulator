from .common import CFG, gm
from .model import task_cost, geometry

def window_metrics(cs,w,model,overhead,schedule):
    t=schedule['cycles']; owners=schedule['owner']; tasks=w['tasks']
    qs=[task_cost(cs[c],task['m'],model,overhead) for task,c in zip(tasks,owners)]
    useful=sum(q['useful_macs'] for q in qs)
    weight=len(tasks)*3*CFG['hidden']*CFG['intermediate']*CFG['element_bytes']
    sram=sum(q['sram_bytes'] for q in qs)
    # Literal full-read-width on all non-user-overhead cycles. In the reused
    # model this charges idle recurrence slots too, so it is an upper envelope.
    sram_fullslot=sum((q['cycles']-3*overhead)*cs[c].bytes_per_issue for q,c in zip(qs,owners))
    us=t/CFG['clock_hz']*1e6;flops=2*useful/1e9
    fetch=weight/(CFG['hbm_reference_TBps']*1e6)
    sfetch=sram/(CFG['sram_reference_TBps']*1e6)
    row=dict(window_id=w['id'],batch=w['batch'],compute_us=us,cycles=t,
        useful_MACs=useful,equivalent_GFLOP=flops,throughput_TFLOPS=flops*1000/us,
        utilization_pct=100*useful/(CFG['total_macs']*t),active_experts=len(tasks)-2,
        weight_MB=weight/1e6,hbm_req_TBps=weight/t/1000,hbm_fetch_us=fetch,
        roofline_us=max(us,fetch),bound='compute-bound' if us>=fetch else 'memory-bound',
        sram_peak_TBps=sum(c.peak_feed for c in cs),sram_MB=sram/1e6,
        sram_req_TBps=sram/t/1000,sram_fetch_us=sfetch,
        sram_die_equivalent=sram/t/1000/CFG['sram_reference_TBps'],
        sram_fullslot_upper_MB=sram_fullslot/1e6,
        sram_fullslot_upper_TBps=sram_fullslot/t/1000,
        hierarchy_roofline_us=max(us,fetch,sfetch),
        loads=schedule['loads'],owner=owners,status=schedule['status'],
        frontier_peak=schedule['frontier_peak'])
    assert useful==8*w['batch']*3*2048*1408
    assert 0<row['utilization_pct']<=100.0000001
    assert row['sram_req_TBps']<=row['sram_peak_TBps']+1e-9
    return row

MEAN_KEYS=('compute_us','useful_MACs','equivalent_GFLOP','throughput_TFLOPS','utilization_pct',
    'active_experts','weight_MB','hbm_req_TBps','hbm_fetch_us','roofline_us','sram_peak_TBps',
    'sram_MB','sram_req_TBps','sram_fetch_us','sram_die_equivalent',
    'sram_fullslot_upper_MB','sram_fullslot_upper_TBps','hierarchy_roofline_us')

def aggregate(rows):
    import math
    out=[]
    for b in CFG['batches']:
        rs=[r for r in rows if r['batch']==b]
        d=dict(batch=b,windows=len(rs))
        d.update({k:math.fsum(r[k] for r in rs)/len(rs) for k in MEAN_KEYS})
        cb=sum(r['bound']=='compute-bound' for r in rs)
        d['bound']='compute-bound' if cb==len(rs) else 'memory-bound' if cb==0 else 'mixed'
        d['compute_bound_windows']=cb
        out.append(d)
    return out

def gm_row(rows):
    return dict(batch='GM',windows=sum(r['windows'] for r in rows),
                **{k:gm([r[k] for r in rows]) for k in MEAN_KEYS})
