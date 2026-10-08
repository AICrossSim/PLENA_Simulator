"""Checks for the parameterized frontend and interval interpretation."""
from research.moe_dispatch.round3.model import Core, Design, task_cost
from research.moe_dispatch.round3.sensitivity import (
    SensitivityParameters, _result_interval, clear_parameter_cost_cache,
)


def test_frontend_service_and_credit_limits_are_global():
    design=Design((Core(1,48,256),))
    low=SensitivityParameters(credits=256,weight_tile_service_cycles=30.4)
    high=SensitivityParameters(credits=520,weight_tile_service_cycles=1)
    assert low.hbm_bandwidth==256*32/65
    assert high.hbm_bandwidth==256
    assert low.w_bandwidth(design,0)==4096/30.4
    assert high.w_bandwidth(design,0)==64*high.bank_Bpc
    expert={"Me":2,"H":2048,"F":1408}
    cost=task_cost(expert,design,0,low)
    assert clear_parameter_cost_cache()>0
    assert task_cost(expert,design,0,low)==cost


def test_ratio_interval_does_not_mislabel_witness_as_proof():
    def row(family,upper,lower):
        return {"family":family,"selected":{"score_ms":upper,"design":{}},
                "global_lb_ms":lower,"gap_pct":100*(upper/lower-1),
                "evaluated_points":8,"simulator_calls":16}
    raw={"families":{
        "single":row("single",10,8),"homogeneous":row("homogeneous",10,8),
        "5+1":row("5+1",9,7),"4+2":row("4+2",11,8),"3+3":row("3+3",12,8)},
        "proof_complete":False}
    result=_result_interval(raw)
    assert result["delta"]<0
    assert result["delta_lower"]==7/10-1
    assert result["delta_upper"]==9/8-1
    assert result["delta_upper"]>0  # Faster witness does not certify a reversal.
    assert result["proof_complete"] is False
    assert result["simulator_calls"]==5*16*2
    raw["families"]["5+1"]["selected"]["score_ms"]=11
    raw["families"]["3+3"]["selected"]["score_ms"]=9
    same={"pm":2,"pn":24,"pk":128}
    raw["families"]["3+3"]["selected"]["design"]={"cores":[same,same]}
    assert _result_interval(raw)["compute_shapes_distinct"] is False
    raw["families"]["3+3"]["selected"]["design"]["cores"]=[same,{"pm":4,"pn":12,"pk":128}]
    assert _result_interval(raw)["compute_shapes_distinct"] is True
