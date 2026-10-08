"""Conditional oracle recomputes physical service, never forces finishes."""
from copy import deepcopy
from dataclasses import replace
import pytest

from research.moe_dispatch.round3.config import parameters
from research.moe_dispatch.round3.model import Core,Design
from research.moe_dispatch.round3.runtime import simulate
from research.moe_dispatch.round3.oracle_replay import same_schedule_oracle
from research.moe_dispatch.round2.predictors import Predictor


def workload(batch=16):
    return {"id":"conditional-replay","batch":batch,"hidden":2048,"top_k":2,
            "experts":[{"id":0,"Me":batch,"H":2048,"F":1408},
                       {"id":1,"Me":batch,"H":2048,"F":1408},
                       {"id":-1,"Me":batch,"H":2048,"F":2816,"is_shared":True}]}


@pytest.mark.parametrize("mode",("pipelined","port_tight"))
@pytest.mark.parametrize("landing",("private","shared"))
@pytest.mark.parametrize("prefetch",(False,True))
def test_second_pass_recomputes_same_finite_services(mode,landing,prefetch):
    d=Design((Core(1,48,128),Core(3,16,128)),flows=("WS","IS"))
    if landing=="shared":
        d=replace(d,w_bytes=(0,0),landing_mode="shared",landing_pool_bytes=sum(d.w_bytes))
    p=parameters(mode,prefetch=prefetch)
    plan=simulate(workload(),d,p,predictor=Predictor("ours"));before=deepcopy(plan)
    first=same_schedule_oracle(plan,d,p);second=same_schedule_oracle(plan,d,p)
    assert first==second and plan==before
    check=first["oracle_replay"]
    assert check["physically_replayed"] and check["mae_pct"]<1e-8
    assert check["max_timing_difference_cycles"]<1e-4
    assert check["max_hbm_difference_bytes"]<1
    assert check["max_pool_used_bytes"]<=d.landing_pool_bytes
    assert first["cycles"]==pytest.approx(plan["cycles"],rel=1e-10,abs=1e-5)
    assert check["scope"].endswith("not a scheduling upper bound")


def single_plan():
    d=Design((Core(3,32,128),),flows=("WS",));p=parameters()
    return d,p,simulate(workload(),d,p,predictor=Predictor("ours"))


def test_replay_rejects_forged_finish_instead_of_forcing_it():
    d,p,plan=single_plan();plan["phases"][0]["finish"]+=100
    with pytest.raises(AssertionError,match="physical replay phase timing mismatch"):
        same_schedule_oracle(plan,d,p)


def test_replay_rejects_incorrect_physical_demands():
    d,p,plan=single_plan();plan["phases"][0]["w_sram_bytes"]+=1024
    with pytest.raises(AssertionError,match="installed cost model"):
        same_schedule_oracle(plan,d,p)


def test_replay_rejects_changed_bandwidth_and_first_arrival():
    d,p,plan=single_plan()
    with pytest.raises(AssertionError):
        same_schedule_oracle(plan,d,replace(p,credits=256))
    plan["tasks"][0]["first_weight_ready"]+=100
    with pytest.raises(AssertionError,match="physical replay task/prefix timing mismatch"):
        same_schedule_oracle(plan,d,p)


def test_chunked_global_storage_replay_pays_spill():
    d=Design((Core(3,32,128),),flows=("WS",));p=parameters()
    plan=simulate(workload(256),d,p,predictor=Predictor("ours"))
    assert plan["storage_chunks"]>1 and plan["activation_spill_bytes"]>0
    replay=same_schedule_oracle(plan,d,p)
    assert replay["cycles"]==pytest.approx(plan["cycles"],rel=1e-10)
    assert replay["oracle_replay"]["max_hbm_difference_bytes"]<1
