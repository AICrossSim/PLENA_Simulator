"""Independent resource/footprint checks for the experimental Vector modes."""
import pytest
from .ltile_cost import Machine, assembly_cost
from .vector_recurrence import Config, Coefficient, primitive_cost, resources


def cv(base, hs=1, rs=0, repeat=0, heads=32, extent=2048):
    return Coefficient(base, hs, rs, repeat, heads, extent)


def test_shared_units_are_not_unlimited_fusion_or_matrix_reads():
    cfg=Config(1,64,2,0)
    fields=[cv(16*2048),cv(17*2048,128,1,3),cv(18*2048,128,1,3)]
    bank,arith,dep,counts,_,_=primitive_cost(Machine(),0,cfg,fields,0,0,3*2048,0,False)
    assert counts["mul_operations"]==16 and counts["add_operations"]==16
    assert counts["vector_reads"]==32 and counts["vector_writes"]==8
    assert "matrix_reads" not in counts and bank+arith+dep>0


def test_invariant_latch_removes_reads_only_after_explicit_hold():
    fields=[cv(16*2048),cv(17*2048,128,1,3),cv(18*2048,128,1,3)]
    cfg=Config(1,64,3,0)
    with pytest.raises(ValueError,match="HOLD"):
        primitive_cost(Machine(),0,cfg,fields,0,0,3*2048,0,False)
    result=primitive_cost(Machine(),0,cfg,fields,0,0,3*2048,0,True)
    assert result[3]["vector_reads"]==24
    assert resources(3,Machine())["invariant_latch_bytes"]==4096


def test_fsm_has_same_per_element_work_and_bounded_capacity():
    fields=[cv(16*2048),cv(17*2048,128,1,3),cv(18*2048,128,1,3)]
    r=primitive_cost(Machine(),0,Config(32,64,4,96),fields,20*2048,20*2048,3*2048,0,True)
    assert r[3]["mul_operations"]==32*16 and r[3]["vector_writes"]==32*8
    with pytest.raises(ValueError):Config.decode(31,4|97<<3)
    with pytest.raises(ValueError):Coefficient.decode(131070|(31<<44)|(3<<50))


def test_reserved_configuration_and_explicit_forms():
    with pytest.raises(ValueError):Config.decode(64,1)
    with pytest.raises(ValueError):Config.decode(31,1)
    text="S_ADDI_INT gp12, gp0, 0\nS_ADDI_INT gp13, gp0, 3\nV_REC_CFG 0, gp12, gp13\nV_REC_EXEC gp0, gp0, gp0, 4\n"
    c=assembly_cost(text)
    assert c.accesses["vector_reads"]==1 and c.sram==1


def test_tree_overflow_is_rejected_before_partial_fsm_work():
    fields=[cv(16*2048),cv(17*2048,128,1,3),cv(18*2048,128,1,3)]
    for op in [1,2]:
        with pytest.raises(ValueError,match="exceeds 128"):
            primitive_cost(Machine(),op,Config(32,64,4,0),fields,5*2048,20*2048,3*2048,97,True)


def test_original_compiler_services_do_not_require_experimental_module(tmp_path):
    from pathlib import Path
    from types import SimpleNamespace
    from .ltile_services import Services
    from .ltile_platform import ExecutionProfile
    root=Path(__file__).resolve().parents[2]/"PLENA_Compiler"
    service=Services(root,ExecutionProfile(),SimpleNamespace(identity={}),tmp_path)
    if not service.vector_candidate_source.exists():
        with pytest.raises(ValueError,match="matching experimental"):
            service.layer("mamba",1,"row",vector_candidate_level=2)
    with pytest.raises(ValueError,match="0..4"):
        service.layer("mamba",1,"row",vector_candidate_level=5)
