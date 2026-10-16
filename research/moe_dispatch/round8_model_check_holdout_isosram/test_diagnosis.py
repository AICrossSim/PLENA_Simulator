import csv
import json
from .diagnose import CFG, ROOT, decompose, ceildiv

def serial_group_oracle(q,nk,L):
    ready=[0]*q
    now=0
    for _ in range(nk):
        for i in range(q):
            now=max(now,ready[i])
            ready[i]=now+L
            now+=1
    return max(ready)

def test_recurrence_by_independent_event_replay():
    for shape in CFG['shapes']:
        pm,pn,g=shape
        for m,k,n in CFG['gemms_M_K_N']:
            nm,nn,nk=ceildiv(m,pm),ceildiv(n,pn),ceildiv(k,g)
            replay=nm*sum(serial_group_oracle(min(8,nn-j),nk,15) for j in range(0,nn,8))
            assert replay==decompose('round7_os',shape,m,k,n)['total_cycles']

def test_hand_calculated_gate_and_down():
    q=decompose('round7_os',[4,256,64],1,2048,1408)
    assert [q[x] for x in ('pure_cycles','reduction_drain_cycles','accumulation_dependency_cycles','other_cycles','total_cycles')]==[192,12,279,2,485]
    q=decompose('round7_os',[4,256,64],1,1408,2048)
    assert q['total_cycles']==176+12+147+2==337

def test_legacy_temporal_tail_differs_from_k_width():
    q=decompose('legacy_phase',[4,256,64],1,1408,2048)
    assert q['packing_extra_cycles']==16
    assert q['total_cycles']==176+16+4+11+6==213

def test_all_decompositions_add_up():
    for model in CFG['models']:
        for shape in CFG['shapes']:
            for m,k,n in CFG['gemms_M_K_N']:
                r=decompose(model,shape,m,k,n)
                assert r['total_cycles']==sum(r[x] for x in ('pure_cycles','reduction_drain_cycles','accumulation_dependency_cycles','other_cycles'))

def test_recorded_b2_reconciliation():
    summary=json.loads((ROOT/'results/b2_summary.json').read_text())['statistics']
    assert abs(summary['legacy_total_us']['AM_us']-7.672111111111111)<1e-12
    assert abs(summary['legacy_total_us']['GM_us']-7.605794395941989)<1e-12
    total=sum(summary[k]['AM_us'] for k in ['plena_os_pure_us','plena_os_reduction_drain_us','plena_os_accumulation_dependency_us','plena_os_other_us'])
    assert abs(total-summary['plena_os_total_us']['AM_us'])<1e-12

def test_confirmed_model_and_scope():
    assert CFG['main_model']=='issue_formula'
    assert CFG['sensitivity_model']=='plena_os'
    assert CFG['pn_caps']==[256,512]
    assert CFG['approval_status'].startswith('user_confirmed')
