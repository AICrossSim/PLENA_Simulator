"""Software schedules must expose reload costs and preserve private lifetimes."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from .ltile_cost import assembly_cost
from .ltile_layers import build_batch, compiler_api
from .ltile_platform import ExecutionProfile
from .ltile_services import Services
from .matrix_service import MatrixService
from .projection_software import software_profile


def fixture():
    root = Path(__file__).resolve().parents[2] / "PLENA_Compiler"
    compiler_api(root)
    from compiler.aten.plena.isa_matrix_projection import Projection
    from compiler.aten.plena.isa_projection_software import lower_software_projection

    spec = Projection(0x10000, 0x100000, 0x40000, 0, 1025, 65, 256)
    xs = [spec.inputs + i * 0x8000 for i in range(4)]
    ys = [spec.outputs + i * 0x8000 for i in range(4)]
    return spec, xs, ys, lower_software_projection


def test_request_grouping_exposes_weight_reloads_and_preserves_compute_count():
    p, xs, ys, lower = fixture()
    full = assembly_cost(lower(p, xs, ys, request_tile=4), matrix_service=MatrixService(), trace_memory=True)
    split = assembly_cost(lower(p, xs, ys, request_tile=1), matrix_service=MatrixService(), trace_memory=True)
    assert full.opcodes["M_MV"] == split.opcodes["M_MV"] == 4 * 5 * 3
    assert full.arithmetic == split.arithmetic
    for cost in (full, split):
        assert not cost.opcodes["M_MM.P"]
        assert not cost.accesses["projection_latch_service_cycles"]
    weight_bytes = lambda c: sum(n for op, a, n in c.memory_trace
                                if op == "r" and p.weights <= a < p.weights + p.weight_bytes)
    assert weight_bytes(full) == p.weight_bytes
    assert weight_bytes(split) == 4 * p.weight_bytes


def test_cross_group_aliases_and_invalid_budget_fail_before_emission():
    p, xs, ys, lower = fixture()
    with pytest.raises(ValueError, match="overlap"):
        lower(p, xs, [xs[3], *ys[1:]], request_tile=1)
    for tile in (0, 3, 32, True):
        with pytest.raises(ValueError):
            lower(p, xs, ys, request_tile=tile)
    with pytest.raises(ValueError):
        lower(p, xs, ys, vector_rows=4)


def test_larger_workspace_is_existing_capacity_not_extra_storage():
    p, xs, ys, lower = fixture()
    p = replace(p, k=7168)
    costs = [assembly_cost(lower(p, xs, ys, vector_rows=n), matrix_service=MatrixService(), trace_memory=True)
             for n in (58, 64)]
    assert costs[0].arithmetic == costs[1].arithmetic
    # Retaining more input rows also removes their DMA-to-SRAM writes.
    # Compute service stays identical; memory service must not increase.
    assert costs[1].sram <= costs[0].sram
    assert sum(n for op, _, n in costs[1].memory_trace if op == "r") < sum(
        n for op, _, n in costs[0].memory_trace if op == "r")


def test_codec_workspace_cannot_be_reclaimed_for_compressed_weights(tmp_path):
    h = software_profile(16, 2, 64)
    assert not h.matrix.weight_replay and h.matrix.projection_segments == 1
    assert h.matrix.resources() == ExecutionProfile().matrix.resources()
    root = Path(__file__).resolve().parents[2] / "PLENA_Compiler"
    s = Services(root, h, SimpleNamespace(identity={}), tmp_path)
    with pytest.raises(ValueError, match="codec"):
        s.layer("mamba", 1, "fsm", "NVFP4", supply="native")
    with pytest.raises(ValueError, match="codec"):
        s.projection(1, 128, 32, "NVFP4")
    with pytest.raises(ValueError, match="workspace"):
        ExecutionProfile(projection_vector_rows=64)


def test_standalone_projection_uses_the_same_software_request_tile(tmp_path, monkeypatch):
    """A changed profile must affect emission, not only the cache label."""
    root = Path(__file__).resolve().parents[2] / "PLENA_Compiler"
    compiler_api(root)
    costs = []
    for tile in (1, 4):
        service = Services(root, software_profile(16, tile, 58), SimpleNamespace(identity={}), tmp_path)
        monkeypatch.setattr(service, "cached", lambda spec, generate: generate())
        program, _, metadata = service.projection(4, 1025, 65)
        assert metadata["projection_request_tile"] == tile
        costs.append(assembly_cost(program, matrix_service=MatrixService(), trace_memory=True))
    assert costs[0].arithmetic == costs[1].arithmetic
    assert sum(n for op, _, n in costs[0].memory_trace if op == "r") > sum(
        n for op, _, n in costs[1].memory_trace if op == "r")


def test_transposed_packets_have_bounded_storage_and_no_weight_expansion():
    p, xs, ys, _ = fixture()
    from compiler.aten.plena.isa_projection_software import lower_transposed_projection

    p = replace(p, k_tile=1024)
    text = lower_transposed_projection(p, xs, ys)
    cost = assembly_cost(text, matrix_service=MatrixService(), trace_memory=True)
    assert cost.opcodes["M_TMV"] == 4 * 2 * 3
    assert not cost.opcodes["M_MM.P"] and not cost.opcodes["M_MV"]
    # The final K/N tails occupy whole words; layout never expands weights 8x.
    assert p.weight_bytes == 1056 * 96 * 2
    assert sum(n for op, a, n in cost.memory_trace
               if op == "r" and p.weights <= a < p.weights + p.weight_bytes) == p.weight_bytes
    with pytest.raises(ValueError, match="exceeds existing Matrix"):
        lower_transposed_projection(replace(p, k=16385, weights=0x10000000))


@pytest.mark.parametrize("batch", [1, 16])
def test_transposed_layer_composition_replaces_every_projection(batch):
    root = Path(__file__).resolve().parents[2] / "PLENA_Compiler"
    plan = build_batch("mamba", batch, root, gather="cached", native_coefficients=True,
                       projection_schedule="transposed", projection_k_tile=1024)
    projections = [s for s in plan.stages if s.matrix_shape]
    assert len(projections) == 2
    for stage in projections:
        assert "M_TMV " in stage.assembly
        assert "M_MV " not in stage.assembly and "M_MM.P " not in stage.assembly
        assert "transposed K=1024" in stage.assembly
        assert stage.matrix_shape[0] == batch


def test_transposed_profile_rejects_unvalidated_codec_and_extra_datapaths(tmp_path):
    root = Path(__file__).resolve().parents[2] / "PLENA_Compiler"
    h = ExecutionProfile(projection_schedule="transposed", projection_k_tile=1024)
    assert h.matrix.resources() == ExecutionProfile().matrix.resources()
    service = Services(root, h, SimpleNamespace(identity={}), tmp_path)
    with pytest.raises(ValueError, match="only for BF16"):
        service.layer("mamba", 1, "fsm", "NVFP4", supply="native")
    with pytest.raises(ValueError, match="only for BF16"):
        service.projection(4, 1024, 32, "NVFP4")
    with pytest.raises(ValueError, match="no projection extensions"):
        replace(h, matrix=replace(h.matrix, weight_replay=True))
