"""Actual packed-byte values through timed Rust issues, vector and combine events.

Synthetic tensors are only a correctness check. Model quality is Q0--Q4.
Set PLENA_V3_BINARY and PLENA_V3_COMPILER to the final build/worktree.
"""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'v3_reference'))
sys.path.insert(0,os.environ.get('PLENA_V3_COMPILER','/scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-compiler/research/moe_dispatch'))
import ref_numerics as rn
import compiler_v3 as cv

BINARY=Path(os.environ.get('PLENA_V3_BINARY','/scratch/shared/mcl123/plena/build/moe_supply_v3/release/moe-dispatch-analytical-v1'))

def tensors(seed=31,rows=3,factor_a='mxint4',L=4,H=544,F=160,r=8,factor_b='bf16',main_bits=4):
    rng=np.random.default_rng(seed)
    x=rn.bf16_round(rng.normal(0,.1,(rows,H)).astype(np.float32))
    gate=rng.uniform(.1,.9,rows).astype(np.float32)
    ws={};fs={};payload={};hashes={}
    for name,K,N in [('g',H,F),('u',H,F),('d',F,H)]:
        w=rng.normal(0,.025,(N,K)).astype(np.float32)
        a=rng.normal(0,.02,(K,r)).astype(np.float32)
        b=rng.normal(0,.02,(r,N)).astype(np.float32)
        fmt=cv.bv.Fmt('timed-payload',main_bits,factor_b,L,factor_a=factor_a)
        raw,meta=cv.pack_projection(w,a,b,fmt,name)
        wq,aq,bq=cv.unpack_projection(raw,meta,fmt)
        hashes[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
        ws[name]=wq;fs[name]=(aq,bq)
        payload[name]={'a':aq.tolist(),'b':bq.tolist()}
        if factor_a=='mxint8s':
            hi,lo=cv.unpack_a8_passes(raw,meta)
            payload[name]['a_hi']=hi.tolist()
            payload[name]['a_lo']=lo.tolist()
    return dict(x=x.tolist(),weights={k:v.tolist() for k,v in ws.items()},
                factors=payload,gate=gate.tolist(),expert_id=-1),ws,fs,gate,hashes

def execute(tmp_path,payload,lanes=(4,2),precision='P2',comp='lanes',z_mode='full',factor_a='mxint4',second=False,factor_b='bf16',main_bits=4,overrides=None,second_rows=2):
    batch=len(payload['x']);H=len(payload['x'][0]);F=len(payload['weights']['g'])
    experts=[dict(id=-1,is_shared=True,Me=batch,H=H,F=F,token_indices=list(range(batch)))]
    values=[payload]
    if second:
        extra,_,_,_,_=tensors(41,second_rows,factor_a,H=H,F=F,r=len(payload['factors']['g']['b']),factor_b=factor_b,main_bits=main_bits);extra['expert_id']=7;values.append(extra)
        indices=[int(x) for x in np.linspace(0,batch-1,second_rows)]
        experts.append(dict(id=7,is_shared=False,Me=second_rows,H=H,F=F,token_indices=indices))
    workload=dict(id='packed_numeric',batch=batch,hidden=H,top_k=1,experts=experts)
    config=dict(arch='supply_v3',lanes=list(lanes),dataflow=['ws_group']*len(lanes),
                precision=precision,main_bits=main_bits,rank_lanes=4,
                ranks={kind:[len(payload['factors'][p]['b']) for p in ['g','u','d']] for kind in ['shared','routed']},
                comp_mode=comp,z_mode=z_mode,factor_a=factor_a,factor_b=factor_b,t_chunk=96,
                max_cycles=500000,record_trace=True,trace_limit=20000,
                numeric_payload={'experts':values})
    if overrides:
        config.update(overrides)
    a=tmp_path/'workload.json';b=tmp_path/'config.json';c=tmp_path/'result.json'
    a.write_text(json.dumps(workload));b.write_text(json.dumps(config))
    raw=[]
    for repeat in range(2):
        c=tmp_path/f'result{repeat}.json'
        result=subprocess.run([str(BINARY),str(a),str(b),str(c)],text=True,capture_output=True,timeout=60)
        assert result.returncode==0,result.stderr[-8000:]
        raw.append(c.read_bytes())
    assert raw[0]==raw[1], 'timed numeric run changed between identical reset executions'
    report=json.loads(raw[0])
    assert report['drained'] and all(report['invariants'].values())
    assert report['dma_transactions_accepted']==report['dma_transactions_landed']
    assert report['numeric_execution']['combine_events']
    return report,values

@pytest.mark.parametrize('lanes',[(6,),(3,3),(4,2)])
@pytest.mark.parametrize('overrides',[
    {'inline_silu':False},
    {'pipeline_supply':False},
    {'prefetch_quota':False},
    {'placement':'joint'},
])
def test_real_ablation_paths_preserve_packed_values(tmp_path,lanes,overrides):
    """Exercise the actual disabled mechanism, including concurrent consumers.

    Each case compares complete numerical outputs, drains real transfers and
    requires two identical timing runs; it does not use an injected cost oracle.
    """
    payload,_,_,_,_=tensors(rows=3,F=640,r=16)
    report,values=execute(tmp_path,payload,lanes=lanes,second=True,overrides=overrides)
    observed={x['expert_id']:np.array(x['output'],np.float32)
              for x in report['numeric_execution']['experts']}
    for value in values:
        assert rn.rel_err(observed[value['expert_id']],gold(value))<=1e-5
    if overrides.get('inline_silu') is False:
        assert sum(c['silu_spill_read_bytes'] for c in report['cores'])>0

@pytest.mark.parametrize('lanes',[(6,),(3,3),(4,2)])
@pytest.mark.parametrize('precision',['P1','P2'])
@pytest.mark.parametrize('overrides',[{'byte_pool':False},{'w_reuse':False}])
def test_slot_allocator_and_per_m_weight_reload_preserve_actual_values(
        tmp_path,lanes,precision,overrides):
    """Exercise finite slot fragmentation and real repeated M-block loads.

    Seven Shared rows require multiple M blocks on every comparison core.
    The two-core cases also have an independent routed consumer. Arithmetic
    and all leases must complete correctly under both compressed-native and
    decode-to-BF16 paths, with two exactly repeated executions.
    """
    payload,_,_,_,_=tensors(rows=7,F=640,r=16)
    report,values=execute(tmp_path,payload,lanes=lanes,precision=precision,
                          second=True,overrides=overrides)
    outputs={x['expert_id']:np.array(x['output'],np.float32)
             for x in report['numeric_execution']['experts']}
    for value in values:
        assert rn.rel_err(outputs[value['expert_id']],gold(value))<=1e-5
    if overrides.get('w_reuse') is False:
        assert report['pool_read_bytes']>report['unique_weight_bytes']
    if overrides.get('byte_pool') is False:
        assert report['pool_reservation_granularity_bytes']==4096
        assert sum(report['pool_partition_bytes_per_core'])==65536
    assert report['pool_peak_bytes']<=report['config'].get('pool_bytes',65536)

@pytest.mark.parametrize('flows',[
    ['ws_group','is_stream'],
    ['switchable','switchable'],
])
@pytest.mark.parametrize('z_mode',['full','streamed'])
@pytest.mark.parametrize('factor_a',['mxint4','mxint8s'])
def test_streaming_consumer_and_flexible_dataflow_execute_actual_values(tmp_path,flows,z_mode,factor_a):
    """Numerically cover the IS consumer used in the measured architecture.

    A Shared expert plus a two-row routed expert exercises both independent
    cores, including retained full-width partials and two-pass factor inputs.
    """
    payload,_,_,_,_=tensors(rows=3,F=640,r=16,factor_a=factor_a)
    report,values=execute(tmp_path,payload,second=True,factor_a=factor_a,z_mode=z_mode,
                          overrides={'dataflow':flows})
    outputs={x['expert_id']:np.array(x['output'],np.float32)
             for x in report['numeric_execution']['experts']}
    for value in values:
        assert rn.rel_err(outputs[value['expert_id']],gold(value,z_mode=z_mode))<=1e-5
        if z_mode=='full':
            assert rn.rel_err(outputs[value['expert_id']],gold(value))<=1e-5
        else:
            x=np.array(value['x'],np.float32)
            w={k:np.array(v,np.float32) for k,v in value['weights'].items()}
            f={k:(np.array(v['a'],np.float32),np.array(v['b'],np.float32))
               for k,v in value['factors'].items()}
            assert rn.rel_err(outputs[value['expert_id']],rn.expert_ffn(
                x,w['g'],w['u'],w['d'],f,np.array(value['gate'],np.float32)))<=5e-3
    assert any(b['dataflow']=='is_stream' for b in report['bindings'])

def gold(payload,L=4,z_mode='full',comp='lanes'):
    x=np.array(payload['x'],np.float32);w={k:np.array(v,np.float32) for k,v in payload['weights'].items()}
    f={k:(np.array(v['a'],np.float32),np.array(v['b'],np.float32)) for k,v in payload['factors'].items()}
    if comp=='none':f=None
    return rn.expert_ffn_hw(x,w['g'],w['u'],w['d'],f,np.array(payload['gate'],np.float32),L=L,z_mode=z_mode)

def test_offload_consumer_retirement_preserves_full_stream_output(tmp_path):
    """A Me4 IS consumer needs its full fixed arena while the owner offloads.

    Verify real numerical output and copies as well as progress; retiring the
    helper's lease before its result reaches U/backing/owner would fail this.
    """
    payload,_,_,_,_=tensors(rows=7,F=640,r=16)
    report,values=execute(tmp_path,payload,precision='P1',comp='offload',
                          second=True,second_rows=4,
                          overrides={'dataflow':['ws_group','is_stream']})
    outputs={v['expert_id']:np.array(v['output'],np.float32)
             for v in report['numeric_execution']['experts']}
    for value in values:
        assert rn.rel_err(outputs[value['expert_id']],gold(value))<=1e-5
    assert any(v['event']=='helper_acc_retire' for v in report['trace'])

@pytest.mark.parametrize('lanes',[(6,),(3,3),(4,2)])
@pytest.mark.parametrize('z_mode',['full','streamed'])
def test_real_packed_values_follow_timed_issue_and_combine(tmp_path,lanes,z_mode):
    payload,_,_,_,_=tensors()
    report,_=execute(tmp_path,payload,lanes=lanes,z_mode=z_mode)
    y=np.array(report['numeric_execution']['experts'][0]['output'],np.float32)
    expected=gold(payload,z_mode=z_mode)
    assert rn.rel_err(y,expected)<=1e-5
    assert rn.rel_err(np.array(report['numeric_execution']['combined_output']),expected)<=1e-5

@pytest.mark.parametrize('comp',['none','lanes','separate','kext','offload'])
def test_compensation_modes_execute_real_output_before_silu(tmp_path,comp):
    payload,ws,fs,g,_=tensors()
    report,_=execute(tmp_path,payload,precision='P1',comp=comp)
    y=np.array(report['numeric_execution']['experts'][0]['output'],np.float32)
    fp64=rn.expert_ffn(np.array(payload['x'],np.float32),ws['g'],ws['u'],ws['d'],None if comp=='none' else fs,g)
    assert rn.rel_err(y,fp64)<=5e-3
    if comp=='lanes':assert rn.rel_err(y,gold(payload))<=1e-5

@pytest.mark.parametrize('comp',['lanes','separate','offload'])
def test_p2_short_k_legal_int4_b_matches_its_own_quantized_model(tmp_path,comp):
    payload,ws,fs,g,_=tensors(factor_b='mxint4')
    report,_=execute(tmp_path,payload,comp=comp,factor_b='mxint4')
    output=np.array(report['numeric_execution']['experts'][0]['output'],np.float32)
    expected=rn.expert_ffn(np.array(payload['x'],np.float32),ws['g'],ws['u'],ws['d'],fs,g)
    assert rn.rel_err(output,expected)<=5e-3
    if comp=='lanes':assert rn.rel_err(output,gold(payload))<=1e-5

def test_packed_mxint3_values_follow_the_same_timed_contract(tmp_path):
    payload,ws,fs,g,_=tensors(main_bits=3)
    report,_=execute(tmp_path,payload,main_bits=3)
    output=np.array(report['numeric_execution']['experts'][0]['output'],np.float32)
    assert rn.rel_err(output,gold(payload))<=1e-5
    assert rn.rel_err(output,rn.expert_ffn(np.array(payload['x'],np.float32),ws['g'],ws['u'],ws['d'],fs,g))<=5e-3

def test_two_experts_combine_in_actual_recorded_column_order(tmp_path):
    payload,_,_,_,_=tensors(rows=3)
    report,values=execute(tmp_path,payload,second=True)
    outs={v['expert_id']:gold(v) for v in values}
    expected=np.zeros((3,544),np.float32)
    for ev in report['numeric_execution']['combine_events']:
        for row,t in enumerate(ev['tokens']):
            c=ev['col'];end=min(544,c+ev['cols'])
            expected[t,c:end]=np.add(expected[t,c:end],outs[ev['expert_id']][row,c:end],dtype=np.float32)
    actual=np.array(report['numeric_execution']['combined_output'],np.float32)
    assert rn.rel_err(actual,expected)<=1e-5

def test_a8_split_actual_high_low_payload_multiblock(tmp_path):
    payload,ws,fs,g,_=tensors(rows=7,factor_a='mxint8s')
    report,_=execute(tmp_path,payload,factor_a='mxint8s')
    y=np.array(report['numeric_execution']['experts'][0]['output'],np.float32)
    assert rn.rel_err(y,gold(payload))<=1e-5
    assert rn.rel_err(y,rn.expert_ffn(np.array(payload['x'],np.float32),ws['g'],ws['u'],ws['d'],fs,g))<=5e-3

@pytest.mark.parametrize('lanes',[(6,),(3,3),(4,2)])
def test_a8_prepass_groups_respect_actual_wor_byte_capacity(tmp_path,lanes):
    payload,_,_,_,_=tensors(rows=7,factor_a='mxint8s',r=32)
    report,_=execute(tmp_path,payload,lanes=lanes,factor_a='mxint8s')
    output=np.array(report['numeric_execution']['experts'][0]['output'],np.float32)
    assert rn.rel_err(output,gold(payload))<=1e-5

@pytest.mark.parametrize('lanes',[(6,),(3,3),(4,2)])
def test_streamed_multi_k_partial_combine_without_hidden_output_backing(tmp_path,lanes):
    payload,_,_,_,_=tensors(rows=3,F=640,r=16)
    report,values=execute(tmp_path,payload,lanes=lanes,z_mode='streamed',second=True)
    num=report['numeric_execution']
    events=num['combine_events']
    assert any(ev['partial'] for ev in events)
    assert all('m_start' in src and 'valid_rows' in src for ev in events for src in ev['sources'])
    states={v['expert_id']:{'X':np.array(v['x'],np.float32),
        'W':{p:np.array(w,np.float32) for p,w in v['weights'].items()},
        'factors':{p:(np.array(f['a'],np.float32),np.array(f['b'],np.float32)) for p,f in v['factors'].items()},
        'gate':np.array(v['gate'],np.float32)} for v in values}
    contributions={}
    for eid,state in states.items():
        _,hw=rn.expert_ffn_hw(state['X'],*[state['W'][p] for p in ['g','u','d']],
            state['factors'],state['gate'],L=4,z_mode='streamed',return_state=True)
        contributions[eid]=rn.down_partial_contributions_hw(hw['Z'],hw['U_d'],state['W']['d'],
            state['factors']['d'][1],hw['rank_segments']['d'],hw['tail_segments']['d'])
    expected=rn.combine_partial_hw(contributions,{e['expert_id']:e['tokens'] for e in events},events,token_count=3)
    assert rn.rel_err(np.array(num['combined_output'],np.float32),expected)<=1e-5
    for v in values:
        expected64=rn.expert_ffn(states[v['expert_id']]['X'],*[states[v['expert_id']]['W'][p] for p in ['g','u','d']],
            states[v['expert_id']]['factors'],states[v['expert_id']]['gate'])
        actual=next(e['output'] for e in num['experts'] if e['expert_id']==v['expert_id'])
        assert rn.rel_err(np.array(actual,np.float32),expected64)<=5e-3
