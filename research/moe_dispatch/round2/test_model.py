"""Model invariants, causal/dataflow cases, and online hardware contracts."""
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

import pytest

# Works both before integration (outside repo) and after integration.
try:
    from research.moe_dispatch.round2 import model as m
except ImportError:
    spec=importlib.util.spec_from_file_location("round2_draft_model",Path(__file__).with_name("model.py"))
    m=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=m
    spec.loader.exec_module(m)


def expert(rows=2,h=512,f=128,index=0,shared=False):
    return {"id":index,"Me":rows,"H":h,"F":f,"is_shared":shared}


def workload(experts,batch=8,topk=1):
    return {"id":"unit","batch":batch,"top_k":topk,"hidden":experts[0]["H"],"experts":experts}


def design(flow="OS",dual=True):
    cores=(m.Core(4,4,512),m.Core(2,4,512)) if dual else (m.Core(6,4,512),)
    return m.Design(cores,flows=(flow,)*len(cores))


def test_exact_ledger_and_decoupled_quota_split():
    d=m.Design((m.Core(4,4,512),m.Core(2,4,512)),w_bytes=(10*1024,30*1024),
               x_bytes=(8*1024,4*1024),acc_bytes=(40*1024,56*1024),
               z_bytes=(100*1024,284*1024),w_banks=(8,56),x_banks=(16,8),
               acc_banks=(5,7),vector_lanes=(12,52))
    assert d.total_macs==12288
    assert sum(d.w_banks)==64 and sum(d.x_banks)==24 and sum(d.acc_banks)==12
    assert d.ledger()["installed_storage_B"]==2158592
    assert d.w_banks!=(43,21)
    with pytest.raises(ValueError):
        m.Design(d.cores,w_bytes=(1024,1024))
    with pytest.raises(ValueError):
        m.Design((m.Core(4,4,512),))


@pytest.mark.parametrize("flow",["OS","WS","IS"])
@pytest.mark.parametrize("mode",["pipelined","port_tight","fixed_issue"])
def test_mac_and_sector_traffic_conservation(flow,mode):
    d=design(flow)
    p=m.Parameters(onchip_mode=mode)
    e=expert(3,h=513,f=129)
    co=m.task_cost(e,d,0,p)
    assert co.useful_macs==3*3*513*129
    assert co.issued_macs-co.useful_macs==co.padding_macs>=0
    assert co.hbm_bytes>=co.unique_hbm_bytes
    assert co.hbm_bytes%32==0
    assert co.w_sram_bytes>=co.unique_hbm_bytes
    assert co.irreducible_busy==co.issues*p.issue_interval
    assert 0<co.spatial_util<=1


def test_os_has_no_intermediate_acc_sram_rmw():
    os=design("OS",False)
    ws=design("WS",False)
    a=m.task_cost(expert(4,h=512,f=128),os,0).phases[0]
    b=m.task_cost(expert(4,h=2048,f=128),os,0).phases[0]
    c=m.task_cost(expert(4,h=2048,f=128),ws,0).phases[0]
    assert a.acc_sram_bytes==b.acc_sram_bytes
    assert b.rf_bytes>a.rf_bytes
    assert c.acc_sram_bytes>b.acc_sram_bytes


def test_ws_reuses_array_weight_across_m_waves():
    os=design("OS",False)
    ws=design("WS",False)
    a=m.task_cost(expert(16,h=2048,f=1408),os,0)
    b=m.task_cost(expert(16,h=2048,f=1408),ws,0)
    assert a.issues==b.issues
    assert a.w_sram_bytes>b.w_sram_bytes
    assert a.hbm_bytes>b.hbm_bytes


def test_is_reuses_array_x_across_n_tiles_and_spills_finitely():
    os=design("OS",False)
    isd=design("IS",False)
    a=m.task_cost(expert(16,h=2048,f=1408),os,0)
    b=m.task_cost(expert(16,h=2048,f=1408),isd,0)
    assert b.x_sram_bytes<a.x_sram_bytes
    # Force a narrow private accumulator in a dual design; backing is HBM.
    d=m.Design((m.Core(4,4,512),m.Core(2,4,512)),flows=("IS","IS"),
               acc_bytes=(1024,95*1024))
    cost=m.task_cost(expert(4,h=2048,f=1408),d,0)
    assert cost.spill_bytes>0
    assert cost.hbm_bytes>cost.unique_hbm_bytes


def test_z_chunking_does_not_allocate_extra_memory():
    d=design("WS",False)
    co=m.task_cost(expert(128,h=2048,f=2816,shared=True),d,0)
    assert co.z_chunks>=2 and len(co.phases)==2*co.z_chunks
    assert all(s.m*s.n*2<=d.z_bytes[0] for s in co.phases if s.paired)
    assert co.hbm_bytes>co.unique_hbm_bytes


def test_tiny_core_cannot_take_shared_without_one_z_row():
    d=m.Design((m.Core(1,2,64),m.Core(5,19,128)))
    m.task_cost(expert(2,f=1408),d,0)
    with pytest.raises(ValueError):
        m.task_cost(expert(2,f=2816,shared=True),d,0)


def test_credit_ceiling_and_port_tight_are_global():
    d=design("WS")
    p=m.Parameters()
    assert p.hbm_bandwidth==pytest.approx(256*32/65)
    assert m.Parameters(credits=512).hbm_bandwidth==pytest.approx(512*32/65)
    tight=m.Parameters(onchip_mode="port_tight")
    assert sum(tight.w_bandwidth(d,c) for c in range(2))==pytest.approx(4096/30.4)
    assert m.Parameters(onchip_mode="fixed_issue").issue_interval==30.4


def test_dot_stage_sensitivity_changes_every_pk_and_is_flat_tree():
    for pk in (32,64,128,256,512,1024):
        c=m.Core(1,1,pk)
        a=m.Parameters(dotstagecycles=1).dot_latency(c)
        b=m.Parameters(dotstagecycles=4).dot_latency(c)
        assert a==2+math.log2(pk)
        assert b>a
    assert m.Parameters().dot_latency(m.Core(1,1,512))==20


@pytest.mark.parametrize("flow",["OS","WS","IS"])
def test_shared_hbm_and_all_simple_resource_bounds(flow):
    d=design(flow)
    p=m.Parameters()
    w=workload([expert(4,h=2048,f=1408,index=0),expert(2,h=2048,f=1408,index=1)])
    r=m.simulate(w,d,p,owners=(0,1))
    assert r["cycles"]>=r["hbm_bytes"]/p.hbm_bandwidth-1e-7
    assert r["cycles"]>=r["useful_macs"]/12288-1e-7
    assert r["cycles"]>=r["w_sram_bytes"]/(64*p.bank_Bpc)-1e-7
    assert r["cycles"]>=r["x_sram_bytes"]/(24*p.bank_Bpc)-1e-7
    assert r["cycles"]>=r["acc_sram_bytes"]/(12*p.bank_Bpc)-1e-7
    assert r["hbm_busy_frac"]<=1+1e-10
    assert len(r["tasks"])==2


@pytest.mark.parametrize("policy",["eft","threshold_2","threshold_fallback_2","adaptive","random","idle"])
def test_online_queue_bound_and_repeatability(policy):
    d=design("WS")
    w=workload([expert((i%4)+1,index=i) for i in range(12)])
    a=m.simulate(w,d,policy=policy)
    b=m.simulate(w,d,policy=policy)
    assert json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)
    assert len(a["bindings"])==12
    assert all(x["online"] and x["bounded_core_queue_depth"]<=2 for x in a["bindings"])
    assert any(b["bind_cycle"]>a["tasks"][0]["start"] for b in a["bindings"][2:])


def test_predictor_feedback_and_progress_are_online():
    class Learning:
        def __init__(self):
            self.completed=[]
            self.progress=[]
            self.seen_updates=[]
        def predict(self,e,c,nominal):
            self.seen_updates.append(len(self.completed))
            return nominal*(1.25 if not self.completed else 1.0)
        def on_complete(self,e,c,predicted,actual):
            self.completed.append((e["id"],actual))
        def on_progress(self,e,c,elapsed,fraction,remaining):
            self.progress.append((e["id"],fraction))
            return elapsed/fraction*(1-fraction)
    pred=Learning()
    w=workload([expert(2,index=i) for i in range(8)])
    r=m.simulate(w,design("WS"),predictor=pred)
    assert len(pred.completed)==8
    assert len(pred.progress)==24
    assert set(q for _,q in pred.progress)=={.25,.5,.75}
    assert max(pred.seen_updates)>0
    assert len(r["tasks"])==8


def test_prefetch_uses_real_reserved_slot_and_can_be_disabled():
    d=design("WS",False)
    w=workload([expert(2,index=i) for i in range(3)])
    a=m.simulate(w,d,m.Parameters(prefetch=True))
    b=m.simulate(w,d,m.Parameters(prefetch=False))
    assert a["hbm_bytes"]==b["hbm_bytes"]
    assert a["prefetched_bytes"]>0
    assert b["prefetched_bytes"]==0
    assert all(t["finish"]>=t["start"] for t in a["tasks"])


def test_one_x_buffer_blocks_next_and_never_overwrites():
    d=m.Design((m.Core(4,4,512),m.Core(2,4,512)),x_bytes=(4*1024,8*1024))
    # Core0 has exactly one full 4x512 input slot; owner plan remains valid
    # but its Next cannot reserve X while Current is active.
    w=workload([expert(4,index=i) for i in range(3)])
    r=m.simulate(w,d,m.Parameters(),policy="threshold_2")
    assert len(r["tasks"])==3
    assert all(b["bounded_core_queue_depth"]==1 for b in r["bindings"] if b["core"]==0)


def test_large_synthetic_batch_chunks_input_combine_and_route_storage():
    es=[expert(256,h=2048,f=512,index=i) for i in range(6)]
    es.append(expert(256,h=2048,f=1024,index=6,shared=True))
    w=workload(es,batch=256,topk=6)
    r=m.simulate(w,design("WS"))
    assert r["storage_chunks"]>=2
    assert r["activation_spill_bytes"]==256*2048*6
    assert r["useful_macs"]==sum(3*e["Me"]*e["H"]*e["F"] for e in es)
    assert r["hbm_busy_frac"]<=1+1e-10
    assert r["hbm_bytes"]>r["native_unique_bytes"]


def test_micro_has_declared_quotas_and_is_not_main_iso_mac():
    cost=m.micro_cost(m.Core(2,4,512),2,flow="WS")
    assert cost.useful_macs==3*2*2048*1408
    assert cost.issued_macs>=cost.useful_macs


def test_random_concrete_design_lower_bounds_never_exceed_schedule():
    rng=random.Random(41)
    shapes=[(m.Core(6,16,128),),(m.Core(6,4,512),),
            (m.Core(3,4,512),m.Core(3,4,512)),
            (m.Core(4,4,512),m.Core(2,4,512))]
    for _ in range(16):
        cs=rng.choice(shapes)
        d=m.Design(cs,flows=tuple(rng.choice(("OS","WS","IS")) for _ in cs))
        p=m.Parameters(onchip_mode=rng.choice(("pipelined","port_tight","fixed_issue")))
        es=[expert(rng.randint(1,8),h=512,f=128,index=i) for i in range(4)]
        r=m.simulate(workload(es),d,p)
        assert r["hbm_bytes"]/p.hbm_bandwidth<=r["cycles"]+1e-7
        assert r["useful_macs"]/d.total_macs<=r["cycles"]+1e-7


def test_compute_recurrence_charges_final_drain_once():
    d=m.Design((m.Core(6,64,32),),flows=("OS",))
    p=m.Parameters(credits=10**9,hbm_Bpc=1e9,bank_Bpc=1e9,
                   hbm_latency=.001,dotstagecycles=.0001,vector_scale=1e9,
                   charge_control=False)
    e=expert(4,h=4,f=4)
    co=m.task_cost(e,d,0,p)
    r=m.simulate(workload([e],batch=4),d,p)
    expected=4*4*4/(12*p.bank_Bpc)+sum(s.compute_cycles+p.hbm_latency for s in co.phases)
    assert r["cycles"]==pytest.approx(expected,abs=1e-6)
    assert all(s.control_cycles==0 for s in co.phases)


def test_all_cache_initial_fill_can_use_reserved_empty_slots():
    d=design("OS",False)
    p=m.Parameters()
    s=m.task_cost(expert(2,h=512,f=4),d,0,p).phases[0]
    assert s.weight_loads==s.weight_live_slots
    average=(s.hbm_bytes-s.spill_bytes)/s.weight_loads
    assert m._window_bandwidth(s,p)==pytest.approx(s.w_slots*average/(p.hbm_latency+1))


def test_completion_callback_receives_immutable_nominal_and_late_waits():
    class Learning:
        def __init__(self): self.nominals=[]; self.bias=1
        def predict(self,e,c,nominal): return nominal*self.bias
        def on_complete(self,e,c,predicted,actual,nominal=None):
            self.nominals.append((e["id"],nominal))
            self.bias*=2
    d=design("WS",False)
    es=[expert(2,h=512,f=128,index=i) for i in range(3)]
    learner=Learning()
    r=m.simulate(workload(es),d,m.Parameters(binding_lead_cycles=0),predictor=learner)
    assert len(learner.nominals)==3
    nominal=m.task_cost(es[0],d,0).isolated_cycles
    assert all(n==nominal for _,n in learner.nominals)
    assert all(t["nominal_cycles"]==nominal for t in r["tasks"])
    assert r["tasks"][0]["current_end"] is None
    assert all(t["current_end"] is not None for t in r["tasks"][1:])


def test_u2_preserves_hbm_capacity_and_removes_only_repeated_array_reads():
    from dataclasses import replace
    d=design("OS",False)
    es=[expert(16,h=2048,f=1408)]
    p=m.Parameters()
    normal=m.simulate(workload(es,batch=16),d,p)
    oracle=m.simulate(workload(es,batch=16),d,replace(p,sram_weight_read_once_oracle=True))
    assert oracle["hbm_bytes"]==normal["hbm_bytes"]
    assert oracle["ledger"]==normal["ledger"]
    assert oracle["w_sram_bytes"]<normal["w_sram_bytes"]
    assert oracle["U2_weight_read_once_oracle"] is True


def test_u1_uses_same_frozen_ledger_and_exact_mac_budget():
    d=design("WS",False)
    w=workload([expert(2,h=512,f=128),expert(4,h=512,f=128,index=1)])
    r=m.shape_oracle(w,d)
    assert r["ledger"]==d.ledger()
    assert r["U1_per_expert_shape_oracle"]
    assert len(r["tasks"])==2
    for t in r["tasks"]:
        assert math.prod(map(int,t["shape"].split("x")))==12288

def test_whole_x_cache_reserves_full_resident_plane():
    d=m.Design((m.Core(6,16,128),),flows=("WS",))
    es=[expert(3,h=2048,f=128,index=i) for i in range(2)]
    co=m.task_cost(es[0],d,0)
    assert co.phases[0].peak_x_bytes==12*1024
    r=m.simulate(workload(es,batch=3),d,m.Parameters(binding_lead_cycles=1e12))
    assert r["bindings"][1]["bind_cycle"]>=r["tasks"][0]["finish"]


def test_ws_m_reuse_limited_by_actual_acc_sram_not_os_record_window():
    d=m.Design((m.Core(6,16,128),),flows=("WS",))
    e=expert(128,h=2048,f=1408)
    co=m.task_cost(e,d,0)
    gu=co.phases[0]
    assert m.ceildiv(128,6)>d.records
    assert gu.peak_acc_bytes==2*m.ceildiv(128,6)*d.cores[0].record_bytes
    assert gu.peak_acc_bytes<=d.acc_bytes[0]
    assert gu.hbm_bytes==gu.unique_hbm_bytes
    # The RF records are irrelevant to SRAM-backed WS residency.
    from dataclasses import replace
    fewer=m.task_cost(e,replace(d,records=1),0)
    assert fewer.hbm_bytes==co.hbm_bytes
    assert fewer.w_sram_bytes==co.w_sram_bytes
    assert fewer.issues==co.issues


def test_chunked_timeline_all_records_share_global_time_and_original_identity():
    es=[expert(256,h=2048,f=128,index=i) for i in range(8)]
    for e in es:e["token_indices"]=list(range(256))
    w=workload(es,batch=256,topk=8)
    d=m.Design((m.Core(6,16,128),),flows=("WS",))
    r=m.simulate(w,d)
    assert r["storage_chunks"]==3
    start=0
    for i,part in enumerate(r["chunk_results"]):
        for kind,timefield in (("tasks","start"),("bindings","bind_cycle"),("phases","start")):
            assert all(row[timefield]>=start for row in part[kind])
            assert all(row["chunk_index"]==i for row in part[kind])
            assert all(0<=row["original_expert_index"]<8 for row in part[kind])
        start+=part["cycles"]


def test_first_weight_prefix_observes_shared_two_core_service():
    d=design("WS")
    es=[expert(4,h=512,f=128,index=0),expert(2,h=512,f=128,index=1)]
    r=m.simulate(workload(es,batch=4),d,owners=(0,1))
    starts={ph["expert_index"]:ph for ph in r["phases"] if ph["phase"]==0}
    p=m.Parameters()
    ready=[]
    for t in r["tasks"]:
        ph=starts[t["expert_index"]]
        free_global=ph["start"]+ph["first_weight_bytes"]/p.hbm_bandwidth
        ready.append(t["first_weight_ready"]-free_global)
        assert ph["start"]<=t["first_weight_ready"]<=ph["finish"]
    assert max(ready)>1  # contention is observed, not estimated away.
    assert r["first_weight_timing_scope"].startswith("observed shared-fluid")
