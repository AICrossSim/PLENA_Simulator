import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate as q


def test_csv_checkpoint_preserves_previous_file_on_enospc(tmp_path,monkeypatch):
    import errno
    p=tmp_path/'verified.csv';p.write_text('prior verified checkpoint\n')
    class FailingWriter:
        def __init__(self,*args,**kwargs):pass
        def writeheader(self):pass
        def writerows(self,rows):raise OSError(errno.ENOSPC,'No space left on device')
    monkeypatch.setattr(q.csv,'DictWriter',FailingWriter)
    with pytest.raises(OSError):q.write_csv(p,[{'metric':1}])
    assert p.read_text()=='prior verified checkpoint\n'
    assert not list(tmp_path.glob('*.checkpoint.*'))


def test_capture_hash_and_rejection_of_smoke(tmp_path):
    p=tmp_path/'x.npz';np.savez(p,x_l13=np.zeros((2,3)),routes_l13=np.zeros((2,1),int),gates_l13=np.ones((2,1)))
    m={'layers':[13],'shards':[{'path':str(p),'sha256':q.sha(p),'tokens':2,'request_sha256':'abc'}]};mp=tmp_path/'m.json';mp.write_text(json.dumps(m))
    _,x,h,n=q.load_capture(mp);assert n==2 and h=={'abc'} and x[13]['x'].shape==(2,3)
    m['shards'][0]['sha256']='wrong';mp.write_text(json.dumps(m))
    with pytest.raises(ValueError):q.load_capture(mp)
    m['shards'][0].update({'sha256':q.sha(p),'prefix_truncated_smoke':True});mp.write_text(json.dumps(m))
    with pytest.raises(ValueError):q.load_capture(mp)


def test_mxint8s_factor_requires_negative8_support():
    x=np.array([[119.,-119.,8.,-8.]],np.float32).T
    f=q.factor_a(x,'mxint8s');assert np.array_equal(f,x)


def test_cosine_zero_stays_finite():
    assert np.isfinite(q.cosine(np.zeros(8),np.zeros(8)))


def test_rank_table_respects_each_projection_capacity():
    # Gate/Up use four K segments, Down uses three: one expert must expose 32/32/24.
    weights={e:{'g':np.zeros((4,2048),np.float32),'u':np.zeros((4,2048),np.float32),
                'd':np.zeros((4,1408),np.float32)} for e in range(64)}
    factors={e:{k:(np.zeros((w.shape[1],32 if k!='d' else 24)),np.zeros((32 if k!='d' else 24,4)),
                   np.linspace(32,1,40)) for k,w in weights[e].items()} for e in range(64)}
    cal={'x':np.zeros((16,2048)),'routes':np.tile(np.arange(6),(16,1)),'gates':np.full((16,6),1/6)}
    table,choices,energy,bytes_,caps,freq=q.rank_calibration(13,4,'qera_approx',weights,factors,cal,8)
    assert table['capacity_per_projection']['gate'][0]==32
    assert table['capacity_per_projection']['down'][0]==24
    assert table['tail_energy_per_projection']['down'][0][3]==table['tail_energy_per_projection']['down'][0][4]
    assert table['tail_energy_per_projection']['gate'][0][4]<table['tail_energy_per_projection']['gate'][0][3]
    assert np.array_equal(bytes_,sum(np.array(v) for v in table['factor_bytes_per_projection'].values()))
    assert caps[0]==4 and table['uniform_reference_rank']==32
    assert table['initial_control_state']['windows_observed']==0


def test_shared_source_statistics_preserve_reference_closed_form():
    rng=np.random.default_rng(23);X=rng.normal(size=(32,16));W=rng.normal(size=(12,16));Wq=q.rn.mx_quantize(W,4)[2]
    for method in ('lqer','l2qer','qera_approx','qera_exact'):
        cache={};a,b,s=q.cached_source_factors(W,Wq,X,8,method,cache,('e7','gu'))
        ar,br,sr=q.rn.lowrank_factors(W,Wq,X,8,method)
        assert np.allclose(a@b,ar@br,rtol=1e-10,atol=1e-10)
        assert np.allclose(s,sr,rtol=1e-10,atol=1e-10)
        q.cached_source_factors(W*2,Wq*2,X,8,method,cache,('e7','gu'))
        assert len(cache)==(0 if method=='lqer' else 1)


def test_cached_main_gate_up_has_identical_hardware_order():
    rng=np.random.default_rng(27);x=q.rn.bf16_round(rng.normal(size=(3,544)).astype(np.float32)*.1)
    w={k:q.rn.mx_quantize(rng.normal(size=(160,544) if k!='d' else (544,160)).astype(np.float32)*.02,4)[2] for k in ('g','u','d')}
    fs={k:(q.rn.quantize_factor(rng.normal(size=(W.shape[1],8)).astype(np.float32)*.02,'mxint4',0),
           q.rn.bf16_round(rng.normal(size=(8,W.shape[0])).astype(np.float32)*.02)) for k,W in w.items()}
    gate=np.array([.2,.4,.6],np.float32);actual=q.ffn_reuse_gate_up(x,w,fs,gate,4,q.gate_up_parts(x,w))
    gold=q.rn.expert_ffn_hw(x,w['g'],w['u'],w['d'],fs,gate,4)
    assert np.array_equal(actual,gold)


def test_q3_cached_uniform_outputs_match_direct_window(tmp_path):
    rng=np.random.default_rng(29);x=q.rn.bf16_round(rng.normal(size=(3,32)).astype(np.float32)*.1)
    routes=np.tile(np.arange(6),(3,1));scores=np.full((3,6),1/6,np.float32);ev={'x':x,'routes':routes,'gates':scores}
    ws={};fs={};problems={};direct=np.zeros_like(x);gold=np.zeros_like(x)
    for e in list(range(64))+[-1]:
        ws[e]={'g':rng.normal(size=(16,32)).astype(np.float32)*.02,'u':rng.normal(size=(16,32)).astype(np.float32)*.02,'d':rng.normal(size=(32,16)).astype(np.float32)*.02}
        fs[e]={k:(rng.normal(size=(W.shape[1],8)).astype(np.float32)*.02,
                  rng.normal(size=(8,W.shape[0])).astype(np.float32)*.02,np.linspace(16,1,16)) for k,W in ws[e].items()}
        indices=np.arange(3) if e<6 or e==-1 else np.array([],int);g=np.ones(len(indices),np.float32) if e==-1 else np.full(len(indices),1/6,np.float32)
        baseline=q.rn.expert_ffn_hw(x[indices],ws[e]['g'],ws[e]['u'],ws[e]['d'],gate=g)
        problems[e]={'ie':indices,'xe':x[indices],'ge':g,'gold':baseline};gold[indices]+=baseline
        if len(indices):
            deq={k:q.rn.mx_quantize(W,4)[2] for k,W in ws[e].items()};f={k:(q.factor_a(A,'mxint4'),q.rn.quantize_b_per_segment(B,q.rn.rank_segments(8,W.shape[1],8),'bf16')) for (k,(A,B,_)),W in zip(fs[e].items(),ws[e].values())}
            direct[indices]+=q.rn.expert_ffn_hw(x[indices],deq['g'],deq['u'],deq['d'],f,g,8)
    rows=q.evaluate_rank_policies(13,4,'qera_approx',problems,ws,fs,ev,ev,8,tmp_path,budget_ranks=(32,))
    row=next(r for r in rows if r['strategy']=='uniform')
    assert np.isclose(row['relative_error'],q.rn.rel_err(direct,gold),rtol=1e-8,atol=1e-10)
    assert row['factor_bytes']==row['uniform_budget_bytes']


def test_parallel_expert_bases_match_serial_reference():
    rng=np.random.default_rng(33);X=rng.normal(size=(32,16));Z=rng.normal(size=(32,12));ws={};p={}
    for e in (0,-1):
        ws[e]={'g':rng.normal(size=(12,16)),'u':rng.normal(size=(12,16)),'d':rng.normal(size=(16,12))}
        p[e]={'xc':X,'zc':Z}
    deq={e:{k:q.rn.mx_quantize(W,4)[2] for k,W in w.items()} for e,w in ws.items()}
    got=q.parallel_factor_bases(ws,deq,p,{'x':X},Z,'qera_exact',8,8,{},workers=2)
    assert list(got)==[0,-1]
    for e,w in ws.items():
        for k,W in w.items():
            A,B,_=got[e][k];ar,br,_=q.rn.lowrank_factors(W,deq[e][k],Z if k=='d' else X,8,'qera_exact')
            assert np.allclose(A@B,ar@br,rtol=1e-10,atol=1e-10)


def test_q1_coverage_checks_every_required_combination():
    c=q.q1_coverage([], [2,13,26],[4,3],[0,8,16,24,32,48,64],['rtn','qera_approx','qera_exact','lqer','l2qer'],[8,16],['mxint4','mxint8s','bf16'],['bf16','mxint8'])
    assert c['expected_candidates']==2178 and not c['complete']
    rows=[dict(scope='layer',layer=k[0],rank_lanes=k[1],bits=k[2],method=k[3],rank=k[4],factor_a=k[5],factor_b=k[6]) for k in c['missing_candidates']]
    assert q.q1_coverage(rows,[2,13,26],[4,3],[0,8,16,24,32,48,64],['rtn','qera_approx','qera_exact','lqer','l2qer'],[8,16],['mxint4','mxint8s','bf16'],['bf16','mxint8'])['complete']
    duplicate=q.q1_coverage(rows+rows[:1],[2,13,26],[4,3],[0,8,16,24,32,48,64],['rtn','qera_approx','qera_exact','lqer','l2qer'],[8,16],['mxint4','mxint8s','bf16'],['bf16','mxint8'])
    assert not duplicate['complete'] and len(duplicate['duplicate_candidates'])==1


def test_lambda_fit_uses_actual_development_window_allocations():
    ws={e:{k:np.zeros((4,K),np.float32) for k,K in [('g',2048),('u',2048),('d',1408)]} for e in range(64)}
    fs={e:{k:(np.zeros((W.shape[1],32)),np.zeros((32,4)),np.linspace(32,1,40)*(1+e%3)) for k,W in w.items()} for e,w in ws.items()}
    routes=np.concatenate([np.tile([0,1,2],(16,1)),np.tile([2,3,4],(16,1))]);gates=np.concatenate([np.tile([.98,.01,.01],(16,1)),np.tile([.1,.4,.5],(16,1))])
    cal={'x':np.zeros((32,2048)),'routes':routes,'gates':gates}
    tb,_,te,rb,caps,_=q.rank_calibration(13,4,'qera_approx',ws,fs,cal,8,uniform_rank=16)
    spent=[];budget=[]
    for start in (0,16):
        weight=np.zeros(64);active=np.unique(routes[start:start+16])
        for r,g in zip(routes[start:start+16],gates[start:start+16]):
            for e,s in zip(r,g):weight[e]+=s*s
        idx=q.rn.allocate_ranks(weight[active],te[active],rb[active],tb['lambda0'],caps[active])
        spent.append(rb[active,idx].sum());budget.append(rb[active,2].sum())
    assert tb['lambda_fit']['window_count']==2
    assert np.isclose(tb['lambda_fit']['mean_allocated_window_bytes'],np.mean(spent))
    assert np.isclose(tb['lambda_fit']['mean_uniform_window_bytes'],np.mean(budget))


def test_parallel_candidate_matches_serial_rows_and_actual_output():
    rng=np.random.default_rng(91);ws={};deq={};fs={};p={};gates={};x64={};original={};quant={};parts={};signatures={}
    for e,T in [(0,3),(7,2),(-1,3)]:
        X=q.rn.bf16_round(rng.normal(0,.1,(T,544)).astype(np.float32));gate=np.full(T,.3,np.float32)
        w={k:q.rn.bf16_round(rng.normal(0,.03,(160,544) if k!='d' else (544,160)).astype(np.float32)) for k in ('g','u','d')};ws[e]=w
        deq[e]={k:q.rn.mx_quantize(W,4)[2] for k,W in w.items()}
        fs[e]={k:(rng.normal(0,.02,(W.shape[1],8)),rng.normal(0,.02,(8,W.shape[0])),np.arange(8,0,-1)) for k,W in w.items()}
        Z=q.original_z_hw(X,w,gate);gold=q.rn.expert_ffn_hw(X,w['g'],w['u'],w['d'],gate=gate)
        p[e]={'xe':X,'ze':Z,'gold':gold};gates[e]=gate
        x64[e]={k:(Z if k=='d' else X).astype(np.float64) for k in ('g','u','d')}
        original[e]={k:x64[e][k]@W.T.astype(np.float64) for k,W in w.items()}
        quant[e]={k:x64[e][k]@W.T.astype(np.float64) for k,W in deq[e].items()}
        parts[e]=q.gate_up_parts(X,deq[e]);signatures[e]=q.Counter({(8,8,8):1})
    def run(workers):return q.parallel_candidate(13,4,'qera_approx',8,8,'mxint4','bf16',ws,deq,fs,p,gates,x64,original,quant,parts,{},{},{},signatures,workers)
    serial,srows,sbytes=run(1);parallel,prows,pbytes=run(8)
    assert list(serial)==list(parallel)==[0,7,-1] and srows==prows and sbytes==pbytes
    for e in serial:assert np.array_equal(serial[e],parallel[e])


def test_bf16_lut_can_change_a_discrete_rank_decision():
    energy=np.array([[1.003,1.002,1.001,1.000,.999]])
    costs=np.arange(5,dtype=np.float64)[None,:]
    original=q.rn.allocate_ranks(np.ones(1),energy,costs,.0001,np.array([4]))
    physical=q.rn.allocate_ranks(np.ones(1),q.rn.bf16_round(energy),costs,.0001,np.array([4]))
    assert original[0]==4 and physical[0]==0


def test_bf16_lut_actual_ffn_rows_use_same_resident_outputs(tmp_path,monkeypatch):
    import csv
    monkeypatch.setattr(q,'HARDWARE_BF16_POLICY',True)
    test_q3_cached_uniform_outputs_match_direct_window(tmp_path)
    rows=list(csv.DictReader((tmp_path/'q3_hardware_bf16_l13_w4_qera_approx_L8.csv').open()))
    uniform=next(r for r in rows if r['strategy']=='uniform')
    assert len(rows)==4 and uniform['same_rank_vector_as_float']=='True'
    comparisons=list(csv.DictReader((tmp_path/'q3_rank_vector_comparison_l13_w4_qera_approx_L8.csv').open()))
    assert next(r for r in comparisons if r['strategy']=='uniform')['same']=='True'
