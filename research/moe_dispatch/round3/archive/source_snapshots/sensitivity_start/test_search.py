from dataclasses import replace
import itertools
import math

from research.moe_dispatch.round3.model import Core, Design, KIB
from research.moe_dispatch.round3.config import parameters
from research.moe_dispatch.round3.optimizer import universal_bound, evaluate_design
from research.moe_dispatch.round3.search import (composition_count, roots, split, c1_legal, search_family,
                     canonical, geometries, matches, CAPACITY_KIB, _ordered_geometry_cache)


def fixture():
    return {"id":"development_fixture", "batch":4, "hidden":2048, "top_k":1,
            "experts":[{"id":-1,"Me":4,"H":2048,"F":2816,"is_shared":True,
                        "token_indices":[0,1,2,3]},
                       {"id":0,"Me":2,"H":2048,"F":1408,"is_shared":False,
                        "token_indices":[0,1]},
                       {"id":1,"Me":2,"H":2048,"F":1408,"is_shared":False,
                        "token_indices":[2,3]}]}


def test_compositions_and_splits_conserve_enlarged_domain():
    ranges=((1,4),(1,5),(2,6))
    for total in range(4,17):
        exact=sum(sum(x)==total for x in itertools.product(*(range(a,b+1) for a,b in ranges)))
        assert composition_count(ranges,total)==exact
    for family in ("single","homogeneous","5+1","4+2","3+3"):
        for r in roots(family):
            assert r.cardinality>0
            assert sum(c.cardinality for c in split(r))==r.cardinality
            assert sum(math.comb(531,7) for _ in [0])>0


def test_capacity_total_cross_category_and_current_reservation():
    cores=(Core(1,32,128),Core(2,32,128))
    shared=Design(cores,flows=("WS","WS"),landing_mode="shared",w_bytes=(0,0),
                  landing_pool_bytes=40*KIB,x_bytes=(4*KIB,8*KIB),
                  acc_bytes=(32*KIB,64*KIB),z_bytes=(128*KIB,256*KIB))
    assert c1_legal(shared)
    too_small=replace(shared,landing_pool_bytes=24*KIB,z_bytes=(128*KIB,272*KIB))
    assert not c1_legal(too_small)
    shifted=replace(shared,landing_pool_bytes=64*KIB,z_bytes=(128*KIB,232*KIB))
    assert shifted.landing_pool_bytes+sum(shifted.x_bytes+shifted.acc_bytes+shifted.z_bytes)==CAPACITY_KIB*KIB


def test_new_bound_and_exact_repeat_with_open_frontier():
    w=fixture();p=parameters()
    d=Design((Core(3,32,128),),flows=("WS",),w_bytes=(64*KIB,),x_bytes=(32*KIB,),
             acc_bytes=(64*KIB,),z_bytes=(372*KIB,))
    result=evaluate_design(w,d,p)
    assert result["legal"]
    assert universal_bound(w,p)["lb_cycles"]<=result["milp_sched"]["cycles"]+1e-5
    a=search_family([w],p,"homogeneous",candidate_budget=3,node_budget=8)
    b=search_family([w],p,"homogeneous",candidate_budget=3,node_budget=8)
    assert canonical(a)==canonical(b)
    assert not a["proof_B_closed"]
    assert a["declared_lattice_points"]==a["pruned_lattice_points"]+sum(x["cardinality"] for x in a["open_regions"])
    assert a["evaluated_points"]==3


def test_ratio_family_3plus3_keeps_equal_geometry_option():
    inventory=geometries("3+3")
    assert any(c[0]==c[1] for c in inventory)
    assert any(c[0]!=c[1] for c in inventory)
    assert all(matches(c,"3+3") for c in inventory)


def test_geometry_order_cache_depends_on_workload_and_family_only():
    _ordered_geometry_cache.cache_clear()
    a = canonical([fixture()])
    changed = fixture()
    changed["experts"][0]["Me"] = 1
    b = canonical([changed])
    first = _ordered_geometry_cache("single", a)
    assert _ordered_geometry_cache("single", a) == first
    _ordered_geometry_cache("single", b)
    _ordered_geometry_cache("homogeneous", a)
    info = _ordered_geometry_cache.cache_info()
    assert info.hits == 1 and info.misses == 3
