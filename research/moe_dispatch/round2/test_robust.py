import importlib.util,json,sys
from pathlib import Path
import numpy as np
import pytest
for name in ("robust","extreme"):
    sp=importlib.util.spec_from_file_location("round2_draft_"+name,Path(__file__).with_name(name+".py"))
    module=importlib.util.module_from_spec(sp);sys.modules[sp.name]=module;sp.loader.exec_module(module)
r=sys.modules["round2_draft_robust"];e=sys.modules["round2_draft_extreme"]


def test_objectives_distinguish_tail_and_batch_without_time_sum():
    a=r.objectives([1]*9+[2],[2]*9+[128])
    assert a["cvar10"]==2 and a["minimax"]==2
    assert a["geomean"]==pytest.approx(2**.1)


def test_bootstrap_exact_200_draws_and_reproducible():
    a=r.bootstrap_choices([[.9,.9,1,1],[1,1,1,1]],[2,2,16,16])
    b=r.bootstrap_choices([[.9,.9,1,1],[1,1,1,1]],[2,2,16,16])
    assert a==b and all(sum(x)==200 for x in a.values())
    assert a["geomean"][0]==200


def test_nonpositive_or_unpaired_objectives_rejected():
    with pytest.raises(ValueError):r.objectives([1,0],[2,2])
    with pytest.raises(ValueError):r.objectives([1],[2,2])


def test_continuous_workload_decode_preserves_integer_tokens_and_domains():
    a=e.decode_parameters(np.zeros(6),.01,100)
    b=e.decode_parameters(np.ones(6),.01,100)
    assert a["batch"]==2 and b["batch"]==256
    assert a["E"]==64 and b["E"]==256
    assert a["F"]==512 and b["F"]==2048
    assert a["shared_units"]==0 and b["shared_units"]==4
    assert a["alpha"]==pytest.approx(.01) and b["alpha"]==pytest.approx(100)
    assert b["bw"]==pytest.approx(4*a["bw"])


def test_candidate_rows_include_both_bnb_proofs_and_anchor_near_set_at_all_evaluations(tmp_path):
    from research.moe_dispatch.round2 import robust
    from research.moe_dispatch.round2.common import Core,Design,encode_design

    single=(Core(6,16,128),)
    alternate=(Core(16,24,32),)
    third=(Core(12,8,128),)
    homogeneous=(Core(3,16,128),Core(3,16,128))
    hetero42=(Core(4,16,128),Core(2,16,128))
    hetero51=(Core(5,8,256),Core(1,8,256))

    def point(cores,flows,score,label=""):
        return {"design":encode_design(Design(cores,flows=flows,label=label)),"geomean_ms":score}

    def leaf(row,family="single",status="evaluated",repeat=True):
        return {**row,"design":json.dumps(row["design"]),"family":family,
                "status":status,"repeat_identical":repeat,"allocations_optimal":False}

    seed_single=point(single,("WS",),110)
    seed_homo=point(homogeneous,("WS","WS"),70)
    seed_hetero=point(hetero42,("WS","WS"),90)
    a_best=point(single,("OS",),100,"A witness")
    a_near=point(single,("IS",),100.9)
    a_incumbent=point(alternate,("IS",),100.6)
    a_homo=point(homogeneous,("OS","OS"),69.8)
    a_hetero=point(hetero51,("IS","WS"),89.5)
    b_near=point(alternate,("OS",),100.8)
    b_incumbent=point(third,("WS",),100.5)
    b_hetero=point(hetero42,("OS","WS"),89.6)
    frozen_single=point(alternate,("WS",),100.7)
    frozen={"single":frozen_single,"homogeneous":seed_homo,"heterogeneous":seed_hetero}

    (tmp_path/"seed_points_pipelined.json").write_text(json.dumps([
        seed_single,seed_homo,seed_hetero,{**seed_single,"invalid":"unresolved"}]))
    (tmp_path/"bnb_pipelined_A.json").write_text(json.dumps({
        "families":{"single":a_incumbent,"homogeneous":a_homo},
        "leaves":[leaf(a_best),leaf(a_near),leaf(a_hetero,"5+1")]}))
    duplicate=point(single,("OS",),100.2,"B duplicate")
    (tmp_path/"bnb_pipelined_B.json").write_text(json.dumps({
        "families":{"single":b_incumbent},
        "leaves":[leaf(b_near),leaf(b_hetero,"4+2"),leaf(duplicate,"homogeneous"),
                  leaf(point(single,("WS",),1),status="invalid_or_unresolved"),
                  leaf(point(single,("WS",),.1),repeat=False),
                  leaf(point(single,("WS",),None))]}))

    rows=robust._candidate_rows(tmp_path,"pipelined",frozen,near=.01)
    expected=[a_best,a_near,a_incumbent,a_homo,a_hetero,b_near,b_incumbent,b_hetero,
              frozen_single,seed_homo,seed_hetero]
    by_key={robust._candidate_key(row["design"]):row for row in rows}
    assert len(rows)==len(by_key)==len(expected)
    assert set(by_key)=={robust._candidate_key(row["design"]) for row in expected}
    assert robust._candidate_key(seed_single["design"]) not in by_key
    assert by_key[robust._candidate_key(a_best["design"])]["geomean_ms"]==100
    assert all(row["design"]["label"]=="" for row in rows)
    assert all(row["family"]==Design(tuple(Core(**c) for c in row["design"]["cores"])).family
               for row in rows)


def test_objective_agreement_distinguishes_resource_cuts_with_identical_geometry_and_flows():
    from dataclasses import replace
    from research.moe_dispatch.round2 import robust
    from research.moe_dispatch.round2.common import Core,Design,encode_design

    first=Design((Core(3,16,128),Core(3,16,128)),flows=("WS","WS"),label="mean winner")
    second=replace(first,w_bytes=(19*1024,21*1024),w_banks=(31,33),label="tail winner")
    candidates=[{"design":encode_design(d),"geometry":d.geometry,"flows":d.flows}
                for d in (first,second)]
    metrics=[robust.objectives(ratios,[2,2,2,2,128])
             for ratios in ([.7,.7,.7,.7,1.4],[1,1,1,1,1])]
    winners={name:min(range(2),key=lambda i:metrics[i][name])
             for name in ("geomean","cvar10","minimax")}
    assert winners=={"geomean":0,"cvar10":1,"minimax":1}

    summary=robust._winner_summary(candidates,winners)
    assert len(set(summary["winners"].values()))==1
    assert summary["objectives_agree"] is False
    assert summary["winning_designs"]["geomean"]["w_bytes"]==[20*1024,20*1024]
    assert summary["winning_designs"]["cvar10"]["w_bytes"]==[19*1024,21*1024]
    assert summary["winning_designs"]["minimax"]["w_banks"]==[31,33]
    assert all(d["label"]=="" for d in summary["winning_designs"].values())

    candidates[1]={**candidates[0],"design":encode_design(replace(first,label="renamed witness"))}
    assert robust._winner_summary(candidates,winners)["objectives_agree"] is True
