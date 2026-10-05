import math
from dataclasses import replace
import pytest
from research.moe_dispatch.geometry3d.compute import Core
from research.moe_dispatch.geometry3d.model import Settings as OldSettings, simulate_layer as old_simulate
from research.moe_dispatch.geometry3d.memory import FabricProfile
from research.moe_dispatch.regime.resources import Partition, memory_budget, packed_projection_bytes
from research.moe_dispatch.regime.model import Settings, simulate_layer, projection_phase
from research.moe_dispatch.regime.metrics import bounds, paired_bootstrap, paired_score


def workload(m=2):
    return {'id':'unit','batch':m,'hidden':2048,'top_k':1,'experts':[
        {'Me':m,'H':2048,'F':2816,'is_shared':True},
        {'Me':m,'H':2048,'F':1408,'is_shared':False}]}


@pytest.mark.parametrize('cores',[(Core(6,16,128),),(Core(3,4,512),Core(3,4,512)),(Core(2,23,32),Core(13,13,64))])
def test_BF16_reproduces_frozen_timing(cores):
    for m in (2,16,64,128):
        a=simulate_layer(workload(m),cores,Settings(),detail=True)
        b=old_simulate(workload(m),cores,OldSettings(),detail=True)
        for k in ('cycles','hbm_bytes','native_unique_bytes','useful_macs','issued_macs','core_finish_cycles','X_sram_bytes'):
            assert a[k]==b[k]
        assert bounds(a,Settings())['max_conditional_lb_ms']<=a['latency_ms']


def test_independent_partition_conserves_every_arena_and_port():
    cores=(Core(2,23,32),Core(13,13,64))
    for w in (.125,.5,.875):
        for bank in (.125,.5,.875):
            p=Partition(w=w,x=.75,acc=.25,z=.75,wb=bank,xb=.25,ab=.75)
            b=memory_budget(cores,128,2048,2816,partition=p)
            assert b.total_bytes==2158592 and sum(b.structures.values())==b.total_bytes
            assert sum(c.w_capacity_bytes for c in b.cores)==40960
            assert sum(c.x_register_bytes for c in b.cores)==12288
            assert sum(c.accumulator_bytes for c in b.cores)==98304
            assert sum(c.z_bytes for c in b.cores)==393216
            assert sum(c.w_banks for c in b.cores)==64
            assert sum(c.x_banks for c in b.cores)==24
            assert sum(c.accumulator_banks for c in b.cores)==12


@pytest.mark.parametrize('n,k',[(3,1),(16,128),(23,1408),(2816,2048)])
def test_packed_row_payload_and_scale_alignment(n,k):
    for fmt,bits in [('BF16',16),('W8',8),('W4',4)]:
        payload=((k*bits+7)//8+31)//32*32
        scales=0 if fmt=='BF16' else (((k+127)//128*2+31)//32)*32
        assert packed_projection_bytes(n,k,fmt)==n*(payload+scales)


def test_512_credit_cap_and_installed_storage():
    p=replace(FabricProfile(),hbm_credits=512)
    assert p.landing_credit_bandwidth_upper_bound==512*32/65
    s=Settings(fabric=p)
    r=simulate_layer(workload(),(Core(6,16,128),),s,detail=True)
    assert r['budget']['total_bytes']==2158592
    assert bounds(r,s)['hbm_cap_GBps']<256


def test_detail_recording_does_not_change_timing():
    for fmt in ('BF16','W8','W4'):
        s=Settings(weight_format=fmt)
        a=simulate_layer(workload(16),(Core(6,16,128),),s,detail=True)
        b=simulate_layer(workload(16),(Core(6,16,128),),s,detail=False)
        assert all(a[k]==v for k,v in b.items())


def test_shared_decoder_has_a_real_service_limit():
    s=Settings(weight_format='W4',decoder_elements_per_cycle=1)
    r=simulate_layer(workload(),(Core(6,16,128),),s,detail=True)
    elems=sum(p['decoder_elements'] for p in r['phases'])
    assert r['cycles']>=elems
    assert r['cycles']>simulate_layer(workload(),(Core(6,16,128),),replace(s,decoder_elements_per_cycle=0))['cycles']


def test_paired_bootstrap_and_identity_order():
    ref=[{'workload':str(i),'batch':2,'cycles':100+i} for i in range(8)]
    candidate=[{**r,'cycles':.9*r['cycles']} for r in ref]
    score=paired_bootstrap(candidate,ref,samples=100)
    assert score['bootstrap95_low_percent']==pytest.approx(10)
    assert score['bootstrap95_high_percent']==pytest.approx(10)
    assert paired_score(candidate,ref)==pytest.approx(.9)
    with pytest.raises(ValueError): paired_score(candidate,list(reversed(ref)))
    with pytest.raises(ValueError): paired_bootstrap(candidate,list(reversed(ref)),samples=10)


def test_search_floor_does_not_use_candidate_padding_or_reload_traffic():
    s=Settings(weight_format='W4')
    ws=workload(32)
    results=[simulate_layer(ws,g,s,detail=True) for g in ((Core(6,16,128),),(Core(16,12,64),))]
    a,b=[bounds(r,s) for r in results]
    assert results[0]['compulsory_traffic']==results[1]['compulsory_traffic']
    assert a['architecture_search_lb_ms']==b['architecture_search_lb_ms']
    assert a['port_compulsory_lb_ms']==b['port_compulsory_lb_ms']
    assert a['architecture_search_lb_ms']<=a['max_conditional_lb_ms']
    assert b['architecture_search_lb_ms']<=b['max_conditional_lb_ms']


def test_area_gate_refuses_mac_count_proxy():
    from research.moe_dispatch.regime.area import features,estimate_mm2
    g=(Core(6,16,128),);s=Settings()
    assert features(g,s)['main_BF16_multipliers']==12288
    with pytest.raises(ValueError):estimate_mm2(g,s,None)
    with pytest.raises(ValueError):estimate_mm2(g,s,{'synthesis_validated':True,'MAC_area':0.1})


def test_valid_sram_spans_preserve_physical_reservations_and_mac_slots():
    # M/N/K tails must mask payload traffic, never create extra capacity or
    # claim that inactive arithmetic lanes are useful work.
    c=Core(6,16,128)
    mem=memory_budget((c,),16,2048,2816).cores[0]
    padded=projection_phase(2,129,513,c,mem,Settings(),True)
    actual=projection_phase(2,129,513,c,mem,Settings(valid_operand_traffic=True),True)
    for name in ('peak_w_bytes','peak_x_bytes','peak_accumulator_bytes',
                 'hbm_bytes','weight_unique','useful_macs','issued_macs','issues'):
        assert getattr(actual,name)==getattr(padded,name)
    # Independently sum the valid spans: two projection planes; one M tile;
    # nine N tiles; five K tiles; final N record rounds to a 16B SRAM word.
    assert actual.w_port_bytes==actual.hbm_bytes+2*(129*513*2)*2
    assert actual.x_port_bytes==3*2*513*2+2*9*2*513*2
    assert actual.acc_port_bytes==2*2*(8*64+16)*(2*5-1)+8*2*129
    assert actual.w_port_bytes<padded.w_port_bytes
    assert actual.x_port_bytes<padded.x_port_bytes
    assert actual.acc_port_bytes<padded.acc_port_bytes


def test_compiler_group_limits_trade_lookahead_for_x_reuse_without_weight_cheating():
    c=Core(6,16,128)
    mem=memory_budget((c,),16,2048,2816).cores[0]
    auto=Settings(valid_operand_traffic=True)
    a=projection_phase(2,129,513,c,mem,auto,True)
    b=projection_phase(2,129,513,c,mem,replace(auto,gu_group_limit=1),True)
    for name in ('hbm_bytes','useful_macs','issued_macs','peak_w_bytes','peak_x_bytes'):
        assert getattr(a,name)==getattr(b,name)
    assert b.prefetch_bandwidth>a.prefetch_bandwidth
    assert b.x_port_bytes>a.x_port_bytes
    assert b.peak_accumulator_bytes<a.peak_accumulator_bytes
    # The GU knob must not alter the independent Down program.
    assert projection_phase(2,129,513,c,mem,auto)==projection_phase(2,129,513,c,mem,replace(auto,gu_group_limit=1))
    assert projection_phase(2,129,513,c,mem,replace(auto,down_group_limit=1)).prefetch_bandwidth>projection_phase(2,129,513,c,mem,auto).prefetch_bandwidth


def test_compulsory_weight_port_bound_covers_all_three_matrices_fill_and_read():
    s=Settings(valid_operand_traffic=True,weight_format='W4')
    r=simulate_layer(workload(2),(Core(6,16,128),),s,detail=True)
    minimum=r['compulsory_traffic'];hf=2048*(2816+1408)
    assert minimum['sum_HF']==hf
    weight=(r['native_unique_bytes']+3*hf*2*2)/(64*16)
    expected=max(weight,4*minimum['sum_MH_plus_MF']/(24*16),
                 minimum['accumulator_bytes']/(12*16),minimum['global_activation_bytes']/384)
    assert bounds(r,s)['port_compulsory_lb_ms']==expected/1e6


def test_valid_masking_never_creates_negative_x_fetch_accounting():
    ws=workload(2);core=Core(6,16,128)
    s=Settings(valid_operand_traffic=True)
    r=simulate_layer(ws,(core,),s,detail=True)
    # One M block, 88 routed / 176 shared N tiles; auto groups of four
    # paired N tiles, then eight Down N tiles. Each fetch spans valid rows.
    expected=sum(((e['F']+63)//64)*e['Me']*e['H']*2
                 +16*e['Me']*e['F']*2 for e in ws['experts'])
    assert r['X_sram_bytes']==expected>0


def test_near_optimal_export_accepts_mixed_screen_and_refinement_rows(tmp_path):
    from research.moe_dispatch.regime.search import point,csv_row
    from research.moe_dispatch.geometry3d.study import write_csv
    import csv
    base={**point((Core(6,16,128),)),'legal':True,'reason':'','score_ms':1,'window_cycles':[1,2]}
    screened={**base,'initial_profiles_tested':2,'initial_legal_profiles':2}
    p=tmp_path/'near.csv'
    write_csv(p,[csv_row(base),csv_row(screened)])
    rows=list(csv.DictReader(p.open()))
    assert len(rows)==2 and rows[1]['initial_profiles_tested']=='2'


def test_host_parallel_scoring_matches_serial_complete_repeats():
    from concurrent.futures import ProcessPoolExecutor
    from research.moe_dispatch.regime.search import point,score_point,baseline_one,init_worker
    dev=[workload(2),{**workload(16),'id':'unit16'}]
    jobs=[(point((Core(6,16,128),),gu=1,down=2),256,fmt) for fmt in ('BF16','W4')]
    jobs.append((point((Core(3,16,128),Core(3,16,128))),512,'BF16'))
    serial=[score_point(*args,dev=dev) for args in jobs]
    with ProcessPoolExecutor(max_workers=2,initializer=init_worker,initargs=(dev,)) as pool:
        parallel=list(pool.map(baseline_one,jobs))
    assert parallel==serial
