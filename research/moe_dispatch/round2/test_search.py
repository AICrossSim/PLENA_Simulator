import importlib.util
from pathlib import Path
import sys,json,math
import pytest
try:
    from research.moe_dispatch.round2 import search as s
except ImportError:
    spec=importlib.util.spec_from_file_location("round2_draft_search",Path(__file__).with_name("search.py"))
    s=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s;spec.loader.exec_module(s)


def w(m=2):return {"id":"unit","batch":m,"hidden":512,"top_k":1,
                    "experts":[{"id":0,"Me":m,"H":512,"F":128}]}


def test_domain_contains_all_geometry_and_independent_resources():
    single=s._root("single");homo=s._root("homogeneous");hetero=s._root("heterogeneous")
    assert len(single.indices)==44 and single.cardinality==132
    assert len(homo.indices)==41 and len(hetero.indices)==16678
    cuts=math.prod(total-1 for _,total,_ in s.RESOURCE_SPECS)
    assert hetero.cardinality==16678*9*cuts
    assert cuts==39*11*95*383*63*23*11*63
    assert set(homo.indices).isdisjoint(hetero.indices)


def test_every_region_split_conserves_full_lattice():
    r=s._root("heterogeneous")
    for _ in range(30):
        children=s._split(r)
        assert sum(c.cardinality for c in children)==r.cardinality
        assert children[0].depth==r.depth+1
        r=children[0]
        if r.singleton:break


def test_time_cap_leaves_domain_open_no_shortlist_certificate():
    r=s.search_workloads([w()],time_limit_s=.000001)
    assert not r["proof_complete"]
    for fam,x in r["families"].items():
        assert x["design"] is not None
        assert x["coverage_pct"]==0
        assert sum(y["lattice_points"] for y in x["open_regions"])==x["declared_lattice_points"]
        assert x["proof_status"]=="time_or_node_limited_open_regions"
    json.dumps(r,allow_nan=False)


def test_node_cap_reproducible_results_and_real_full_design():
    a=s.search_workloads([w()],time_limit_s=30,max_nodes=2)
    b=s.search_workloads([w()],time_limit_s=30,max_nodes=2)
    for f,x in a["families"].items():
        assert x["design"] is not None
        assert x["design"]==b["families"][f]["design"]
        assert x["geomean_ms"]==b["families"][f]["geomean_ms"]
        assert x["covered_lattice_points"]+sum(y["lattice_points"] for y in x["open_regions"])==x["declared_lattice_points"]
        assert x["root_lb_ms"]<=x["geomean_ms"]+1e-10
        assert x["score_ms"]==x["geomean_ms"]
    assert all(row["repeat_identical"] for row in a["leaves"] if row["status"]=="evaluated")
    json.dumps(a,allow_nan=False)


def test_mirror_ratio_families_are_declared_aliases():
    assert s._root("4+2").indices==s._root("2+4").indices


def test_optimistic_region_buffers_not_minimum_force_rereads():
    r=s._root("homogeneous")
    intervals=s._intervals(r)
    assert intervals["w_bytes"]==((1024,39*1024),(1024,39*1024))
    val,_=s._bound(r,[w(16)],s.Parameters(),[s.universal_bound(w(16))["lb_cycles"]/1e6])
    for cs in [s._inventory()[i] for i in r.indices[:3]]:
        d=s.Design(cs,flows=("WS","WS"))
        result=s.evaluate_design(w(16),d,s.Parameters(),detail=False)
        if result["legal"]:assert val<=result["milp_sched"]["latency_ms"]


def test_real_resume_retains_partition_and_advances_frontier():
    first=s.search_workloads([w()],delta=0,time_limit_s=30,max_nodes=1,target_families=("single",))
    second=s.search_workloads([w()],delta=0,time_limit_s=30,max_nodes=1,target_families=("single",),resume_state=first)
    assert second["families"]["single"]["design"]==first["families"]["single"]["design"]
    assert second["families"]["single"]["declared_lattice_points"]==132
    assert second["resume"]["open_regions"]!=first["resume"]["open_regions"]
    json.dumps(second,allow_nan=False)
    with pytest.raises(ValueError):s.search_workloads([w(3)],resume_state=first,target_families=("single",),delta=0)


def test_cached_metrics_from_different_workloads_never_prune():
    first=s.search_workloads([w()],time_limit_s=30,max_nodes=0,target_families=("single",))
    cached=first["families"]["single"]
    cached["geomean_ms"]=1e-100;cached["score_ms"]=1e-100
    cached["workload_sha256"]="wrong_frozen_inputs"
    second=s.search_workloads([w()],time_limit_s=30,max_nodes=0,target_families=("single",),initial_designs=[cached])
    assert second["families"]["single"]["geomean_ms"]>1e-10


def test_ratio_families_have_real_full_domain_frontiers_not_seed_only():
    r=s.search_workloads([w()],time_limit_s=.000001,max_nodes=0,
        target_families=("single","homogeneous","heterogeneous","5+1","4+2"))
    for family in ("5+1","4+2"):
        x=r["families"][family]
        assert x["design"] is not None
        assert x["declared_lattice_points"]==s._root(family).cardinality
        assert sum(y["lattice_points"] for y in x["open_regions"])==x["declared_lattice_points"]
        assert not x["proof_complete"]
