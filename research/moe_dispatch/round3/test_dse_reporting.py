from research.moe_dispatch.round3.finalize_dse import excluded_joint_five_percent, union_witnesses


def test_union_preserves_repeated_actual_witnesses_and_rejects_conflicts():
    from copy import deepcopy
    from dataclasses import asdict, replace
    import pytest
    from research.moe_dispatch.round3.model import Core, Design
    first=Design((Core(6,4,512),))
    second=replace(first,flows=("WS",))
    def row(design,score):
        return {"design":asdict(design),"status":"evaluated","score_ms":score,
                "latencies_ms":[score]*18,"repeat_identical":True,
                "solver_statuses":["OPTIMAL"]*18,"allocation_optimal":True}
    one=row(first,2);two=row(second,1)
    c0={"entire_search_repeat_identical":True,"witnesses":[one]}
    c1={"entire_search_repeat_identical":True,"witnesses":[deepcopy(one),two]}
    before=deepcopy((c0,c1))
    pool=union_witnesses(c0,c1)
    assert len(pool)==2
    assert min(r["score_ms"] for r in pool)==1
    assert sorted(len(r["generated_by"]) for r in pool)==[1,2]
    assert (c0,c1)==before  # Original actual-run certificates remain intact.
    c1["witnesses"][0]["latencies_ms"][0]+=1
    with pytest.raises(AssertionError,match="duplicate numerical"):
        union_witnesses(c0,c1)


def test_joint_gain_requires_both_baselines_and_preserves_boundary():
    assert excluded_joint_five_percent(95,100,200)==(False,False,False)
    assert excluded_joint_five_percent(96,100,200)==(True,False,True)
    assert excluded_joint_five_percent(192,100,200)==(True,True,True)
