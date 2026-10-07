import importlib.util,sys
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
