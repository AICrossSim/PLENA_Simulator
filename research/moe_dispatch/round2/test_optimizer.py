"""Assignment relaxation and regional-bound legality checks."""
import importlib.util
import itertools
from pathlib import Path
import random
import sys

import pytest

try:
    from research.moe_dispatch.round2 import optimizer as o
except ImportError:
    spec=importlib.util.spec_from_file_location("round2_draft_optimizer",Path(__file__).with_name("optimizer.py"))
    o=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=o
    spec.loader.exec_module(o)


def w(rows=(1,2,4,3),h=512,f=128):
    return {"id":"opt_unit","batch":8,"top_k":1,"hidden":h,"experts":[
        {"id":i,"Me":m,"H":h,"F":f,"is_shared":False} for i,m in enumerate(rows)]}


def d(flows=("OS","WS")):
    return o.Design((o.Core(4,4,512),o.Core(2,4,512)),flows=flows)


@pytest.mark.parametrize("mode",["pipelined","port_tight","fixed_issue"])
def test_cpsat_relaxation_matches_bruteforce_assignment(mode):
    ww=w()
    dd=d()
    pp=o.Parameters(onchip_mode=mode)
    exact=min(o.allocation_objective(ww,dd,pp,owners)
              for owners in itertools.product((0,1),repeat=4))
    r=o.solve_assignment(ww,dd,pp)
    assert r["status"]=="OPTIMAL"
    assert r["lb_cycles"]<=exact+1e-7
    assert exact-r["lb_cycles"]<=4*r["quantum"]+1e-6
    assert len(r["owners"])==4


def test_grouped_equal_task_counts_preserve_assignment_optimum():
    ww=w((2,2,2,2,2,2))
    dd=d(("WS","IS"))
    r=o.solve_assignment(ww,dd)
    assert r["aggregated_task_types"]==1
    exact=min(o.allocation_objective(ww,dd,o.Parameters(),owners)
              for owners in itertools.product((0,1),repeat=6))
    assert exact-r["lb_cycles"]<=6*r["quantum"]+1e-6


def test_isolated_cold_task_time_is_not_summed_as_core_load():
    ww=w((1,1))
    dd=o.Design((o.Core(6,4,512),),flows=("WS",))
    isolated=sum(o.task_cost(e,dd,0).isolated_cycles for e in ww["experts"])
    r=o.solve_assignment(ww,dd)
    assert r["lb_cycles"]<isolated
    assert r["scope"].startswith("exact quantized assignment")


@pytest.mark.parametrize("flow",["OS","WS","IS"])
@pytest.mark.parametrize("mode",["pipelined","port_tight","fixed_issue"])
def test_lb_is_no_greater_than_executable_schedule(flow,mode):
    rr=o.evaluate_design(w(),d((flow,flow)),o.Parameters(onchip_mode=mode))
    assert rr["legal"]
    assert rr["lb_cycles"]<=rr["milp_sched"]["cycles"]+1e-7
    assert rr["gap_sched_pct"]>=-1e-8
    assert len(rr["runtime"]["tasks"])==4


def test_assignment_repeat_is_bit_identical():
    a=o.solve_assignment(w(),d())
    b=o.solve_assignment(w(),d())
    assert a==b


def test_infeasible_task_is_reported_without_fake_owner():
    dd=o.Design((o.Core(6,4,512),),flows=("WS",))
    # A single Z row is larger than the installed384KiB pool.
    rr=o.solve_assignment(w((1,),f=300000),dd)
    assert rr["status"]=="INFEASIBLE"
    assert rr["owners"] is None and rr["lb_cycles"] is None


def test_root_floor_uses_shared_credit_bandwidth_and_three_w_transfers():
    ww=w()
    pp=o.Parameters(onchip_mode="port_tight")
    r=o.universal_bound(ww,pp)
    unique=o.unique_hbm_bytes(ww)
    assert r["terms"]["hbm_unique"]==pytest.approx(unique/(256*32/65))
    assert r["terms"]["W_mandatory"]==pytest.approx(3*unique/(4096/30.4))


def test_region_does_not_force_slow_core_and_uses_maximum_buffer():
    ww=w((1,2,4))
    dd=d()
    intervals={"w_bytes":[(1024,40*1024),(1024,40*1024)],
               "x_bytes":[(1024,12*1024),(1024,12*1024)],
               "acc_bytes":[(1024,96*1024),(1024,96*1024)],
               "z_bytes":[(1024,384*1024),(1024,384*1024)]}
    r=o.region_bound(ww,geometries=[dd.cores],intervals=intervals)
    sim=o.simulate(ww,dd)
    assert r["lb_cycles"]<=sim["cycles"]+1e-7
    assert r["forced_rule"]=="no merely-slower-core forcing"
    assert "optimistic maxima" in r["reload_rule"]


def test_random_regional_lb_legality_over_all_onchip_modes():
    rng=random.Random(2041)
    for mode in ("pipelined","port_tight","fixed_issue"):
        for _ in range(8):
            dd=d(tuple(rng.choice(("OS","WS","IS")) for _ in range(2)))
            ww=w(tuple(rng.randint(1,8) for _ in range(5)))
            pp=o.Parameters(onchip_mode=mode)
            region=o.region_bound(ww,pp,geometries=[dd.cores])
            result=o.simulate(ww,dd,pp)
            assert region["lb_cycles"]<=result["cycles"]+1e-7


def test_paired_geomean_uses_ratios_not_sum_ms():
    # Fast on a tiny window and slow on a large one: sums are dominated by
    # the large window, paired geometric ratio remains balanced.
    assert o.paired_geomean([1,200],[2,100])==pytest.approx(1)



def test_b256_assignment_uses_actual_global_storage_chunks_and_cached_small_me():
    # topk8 route store limits chunks to124 rows. The cold originalMe3 task
    # crosses a boundary as2+1; each cached plane differs from unchunkedMe3.
    batch=256;h=2048;f=128
    ex=[{"id":i,"Me":batch,"H":h,"F":f,"token_indices":list(range(batch))} for i in range(7)]
    # Fill the eighth route with disjoint degrees, keeping genuine pertoken topk8.
    ex += [{"id":7,"Me":3,"H":h,"F":f,"token_indices":[122,123,124]},
           {"id":8,"Me":253,"H":h,"F":f,"token_indices":[t for t in range(batch) if t not in (122,123,124)]}]
    ww={"id":"chunked_cold","batch":batch,"hidden":h,"top_k":8,"experts":ex}
    dd=o.Design((o.Core(6,16,128),),flows=("WS",))
    p=o.Parameters()
    cs=o.storage_chunks(ww)
    assert len(cs)==3
    table=o._costs(ww,dd,p)
    coldparts=[e for cw in cs for e in cw["experts"] if e["original_expert_index"]==7]
    assert [e["Me"] for e in coldparts]==[2,1]
    expected=sum(o.task_cost(e,dd,0,p).x_sram_bytes for e in coldparts)
    assert table[7][0].x_sram_bytes==expected
    rr=o.evaluate_design(ww,dd,p,detail=False)
    assert rr["lb_cycles"]<=rr["milp_sched"]["cycles"]+1e-7
    assert rr["assignment"]["common_activation_spill_bytes"]==batch*h*6
    assert rr["assignment"]["storage_chunks"]==3


def test_chunk_cost_grouping_does_not_merge_equal_original_me_different_boundaries():
    ww={"id":"chunk_type","batch":256,"hidden":2048,"top_k":8,"experts":[
        {"id":0,"Me":3,"H":2048,"F":128,"token_indices":[0,1,2]},
        {"id":1,"Me":3,"H":2048,"F":128,"token_indices":[122,123,124]}]}
    dd=o.Design((o.Core(6,16,128),),flows=("WS",))
    table=o._costs(ww,dd,o.Parameters())
    assert table[0][0].hbm_bytes!=table[1][0].hbm_bytes
    assert len(o._group_tasks(ww,table))==2


def test_max_z_rowchunk_floor_is_safe_and_retained_w_is_optimistic():
    ww=w((128,),h=2048,f=2816);ww["batch"]=128;ww["top_k"]=1
    dd=o.Design((o.Core(6,16,128),),flows=("WS",))
    p=o.Parameters()
    unique=o.unique_hbm_bytes(ww)
    mandatory=o.rowchunk_hbm_floor_bytes(ww)
    assert mandatory==2*unique-40*1024
    r=o.simulate(ww,dd,p)
    assert mandatory<=r["hbm_bytes"]
    assert o.universal_bound(ww,p)["lb_cycles"]<=r["cycles"]+1e-7
    assert o.rowchunk_hbm_floor_bytes(ww,max_z_bytes=384*1024,max_retained_w_bytes=80*1024)<mandatory


def test_regional_rowchunk_uses_largest_allowed_z_not_smallest():
    ww=w((64,),h=2048,f=2816);ww["batch"]=64
    cs=(o.Core(4,16,128),o.Core(2,16,128))
    dd=o.Design(cs,flows=("WS","WS"),z_bytes=(352*1024,32*1024))
    region=o.region_bound(ww,geometries=(cs,),intervals={"z_bytes":((1*1024,352*1024),(1*1024,32*1024)),
        "w_bytes":((1024,39*1024),(1024,39*1024))})
    r=o.simulate(ww,dd)
    assert region["terms"]["HBM_rowchunk_regional"]==o.unique_hbm_bytes(ww)/o.Parameters().hbm_bandwidth
    assert region["lb_cycles"]<=r["cycles"]+1e-7
