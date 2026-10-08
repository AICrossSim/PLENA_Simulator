"""Independent resource/conservation regressions for the enlarged design API."""
from dataclasses import asdict, replace
import math

import pytest

from research.moe_dispatch.round3.config import parameters
from research.moe_dispatch.round3.model import (BytePoolAllocator, Core, Design, PoolRequest, SRAM_BYTES,
                    FIXED_STORAGE, task_cost)
from research.moe_dispatch.round3.runtime import Candidate, choose_candidate, simulate
from research.moe_dispatch.round2 import model as old_model
from research.moe_dispatch.round2.dispatch_fix.predictor_ablation_20261008.bounded_runtime import simulate as old_fixed
from research.moe_dispatch.round2.predictors import Predictor


def workload(batch=16):
    return {"id":"physical-test","batch":batch,"hidden":2048,"top_k":6,
            "experts":[{"id":"shared","Me":batch,"H":2048,"F":2816,"is_shared":True},
                       {"id":7,"Me":min(2,batch),"H":2048,"F":1408}]}


def private():
    return Design((Core(1,32,128),Core(2,32,128)),flows=("WS","WS"),
                  w_bytes=(24*1024,16*1024),x_bytes=(2*1024,10*1024),
                  acc_bytes=(16*1024,80*1024),z_bytes=(32*1024,352*1024),
                  w_banks=(32,32),x_banks=(4,20),acc_banks=(4,8),vector_lanes=(8,56))


def test_shared_pool_counts_once_and_rejects_free_storage():
    d=replace(private(),w_bytes=(0,0),landing_mode="shared",landing_pool_bytes=40*1024)
    ledger=d.ledger()
    assert sum(FIXED_STORAGE.values())+d.landing_pool_bytes+sum(sum(getattr(d,k))
                  for k in ("w_bytes","x_bytes","acc_bytes","z_bytes"))==SRAM_BYTES
    assert ledger["shared_pool_counted_once"] and d.w_bytes==(0,0)
    with pytest.raises(ValueError):
        replace(d,w_bytes=(1*1024,1*1024))
    with pytest.raises(ValueError):
        replace(d,landing_pool_bytes=48*1024)


def test_cross_type_h2_capacity_preserves_exact_aggregate():
    d=replace(private(),w_bytes=(24*1024,32*1024),acc_bytes=(16*1024,64*1024))
    assert sum(d.w_bytes)==56*1024
    for e in workload(128)["experts"]:
        for c in range(2):
            assert task_cost(e,d,c,parameters()).hbm_bytes <= task_cost(e,private(),c,parameters()).hbm_bytes
    with pytest.raises(ValueError):
        replace(private(),w_bytes=(24*1024,32*1024))


def test_byte_allocator_handles_unequal_tiles_and_deadlines():
    allocator=BytePoolAllocator(40*1024,(6*1024,8*1024))
    grants,ledger=allocator.allocate([
        PoolRequest("late",8*1024,20,20,9),
        PoolRequest("early",6*1024,0,100,9)],next_reservations=(8*1024,))
    assert grants["early"]==6*1024
    assert grants["late"]==8*1024
    assert ledger["used_bytes"]==36*1024
    assert ledger["used_bytes"]<=allocator.pool_bytes
    assert ledger["current_reserved_bytes"]==14*1024
    assert ledger["next_reserved_bytes"]==8*1024
    with pytest.raises(ValueError):
        allocator.allocate([],next_reservations=(32*1024,))


@pytest.mark.parametrize("mode",("pipelined","port_tight"))
@pytest.mark.parametrize("credits",(256,520))
def test_legacy_private_services_remain_bit_exact(mode,credits):
    p=parameters(mode,credits=credits);d=private();w=workload(16)
    old=old_fixed(w,d,p,t_big=3,large_first=True,predictor=Predictor("ours"),detail=True)
    new=simulate(w,d,p,t_big=3,large_first=True,predictor=Predictor("ours"),detail=True,
                 dispatch="fixed_legacy")
    for field in ("cycles","latency_ms","hbm_bytes","w_sram_bytes","x_sram_bytes",
                  "acc_sram_bytes","core_finish_cycles","hbm_busy_frac","core_compute_busy"):
        assert new[field]==old[field],field
    for e in w["experts"]:
        for c in range(2):
            assert asdict(task_cost(e,d,c,p))==asdict(old_model.task_cost(e,d,c,p))


@pytest.mark.parametrize("mode",("pipelined","port_tight"))
def test_single_new_fixed_and_old_eft_are_exact(mode):
    p=parameters(mode);d=Design((Core(3,32,128),),flows=("WS",));w=workload()
    a=simulate(w,d,p,dispatch="fixed",predictor=Predictor("ours"))
    b=simulate(w,d,p,dispatch="eft_old",predictor=Predictor("ours"))
    assert a==b


@pytest.mark.parametrize("mode",("pipelined","port_tight"))
def test_shared_simulation_conserves_bytes_and_installed_ports(mode):
    d=replace(private(),w_bytes=(0,0),landing_mode="shared",landing_pool_bytes=40*1024)
    p=parameters(mode);w=workload(16)
    a=simulate(w,d,p,dispatch="fixed_legacy",predictor=Predictor("ours"))
    b=simulate(w,d,p,dispatch="fixed_legacy",predictor=Predictor("ours"))
    assert a==b
    assert sum(d.w_banks)==64 and sum(d.x_banks)==24 and sum(d.acc_banks)==12
    integrated=sum((s["end"]-s["start"])*s["hbm_rate_Bpc"] for s in a["segments"])
    assert math.isclose(integrated,a["hbm_bytes"],rel_tol=1e-8,abs_tol=1)
    for s in a["segments"]:
        assert s["end"]<=a["cycles"] and s["start"]<=s["end"]
        assert s["hbm_rate_Bpc"]<=p.hbm_bandwidth*(1+1e-12)
        assert sum(s["inflight_bytes"])<=p.credits*p.request_bytes*(1+1e-12)
        if "pool_ledger" in s:
            assert s["pool_ledger"]["used_bytes"]<=d.landing_pool_bytes
    assert "not native" in a["supply_observation_scope"]


@pytest.mark.parametrize("mode",("pipelined","port_tight"))
def test_shared_physical_streams_support_different_tile_sizes(mode):
    # Equal multiplier halves with different geometries: W blocks are 12 KiB
    # and 4 KiB, so a nominal "slot count" cannot represent this shared pool.
    d=Design((Core(1,48,128),Core(3,16,128)),flows=("WS","WS"),
             w_bytes=(0,0),landing_mode="shared",landing_pool_bytes=40*1024)
    result=simulate(workload(),d,parameters(mode),dispatch="fixed_legacy",predictor=Predictor("ours"))
    assert result["hbm_bytes"]>=result["native_unique_bytes"]
    for segment in result["segments"]:
        if "pool_ledger" in segment:
            assert segment["pool_ledger"]["current_reserved_bytes"]==16*1024
            assert segment["pool_ledger"]["used_bytes"]<=40*1024


def test_wait_comparison_charges_only_externality_once():
    # Own refetch already included in the nominal 100-cycle task duration.
    # Extra shared-bus contention can make waiting for a clean core preferable.
    now_refetch=Candidate(0,100,0,2,True,1000,30)
    wait_clean=Candidate(1,20,90,1,False,0,0)
    assert choose_candidate([now_refetch,wait_clean],enhanced=False).chosen==now_refetch
    decision=choose_candidate([now_refetch,wait_clean],enhanced=True)
    assert decision.chosen is None and decision.deferred
    assert now_refetch.finish==100 and now_refetch.scored_finish==130


def test_all_refetch_selects_smallest_factor_or_waits_for_it():
    fast_bad=Candidate(0,10,0,3,True)
    slow_less=Candidate(1,100,0,2,True)
    assert choose_candidate([fast_bad,slow_less]).chosen==slow_less
    assert choose_candidate([fast_bad,replace(slow_less,admissible=False)]).chosen is None
    assert choose_candidate([fast_bad,slow_less],enhanced=False).chosen==fast_bad
