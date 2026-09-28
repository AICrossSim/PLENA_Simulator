"""Resource and address checks for the finite projection replay candidate."""

from dataclasses import replace
from pathlib import Path

import pytest

from .ltile_cost import assembly_cost
from .ltile_platform import ExecutionProfile
from .matrix_service import MatrixService, matrix_cost


def _load(reg, value):
    return f"S_LUI_INT {reg}, {value >> 12}\nS_ADDI_INT {reg}, {reg}, {value & 4095}\n"


def _packet(rows=1, k=256, source=2048, destination=16384, cfg=None):
    config = (rows - 1) | (8 << 2) | (64 << 10) if cfg is None else cfg
    return (
        _load("gp9", (31 << 12) | (k - 1))
        + _load("gp10", k)
        + "L_TILE_CFG 0, gp9, gp10\n"
        + _load("gp1", destination)
        + _load("gp2", 0)
        + _load("gp3", source)
        + _load("gp4", config)
        + "M_MM.P gp1, gp2, gp3, gp4, 0\n"
    )


def test_replay_retains_payload_but_still_pays_latch_feed_and_final_writes():
    h = MatrixService(weight_replay=True)
    prefix = _packet().rsplit("M_MM.P", 1)[0]
    old_asm = prefix + "M_MV 0, gp2, gp3, 0\nM_MV_WO gp1, 0\n"
    old = assembly_cost(old_asm, matrix_service=replace(h, weight_replay=False))
    replay = assembly_cost(old_asm, matrix_service=h)
    assert old.arithmetic == replay.arithmetic == 256
    assert old.sram == 42
    assert replay.sram == 23  # four preload, one row, sixteen latch feeds, RMW
    assert replay.accesses["projection_latch_service_cycles"] == 16
    assert h.projection_resources()["payload_bytes"] == 22784


def test_batch_packet_shares_arithmetic_but_preserves_per_request_merges():
    h = MatrixService()
    one = assembly_cost(_packet(1), matrix_service=h)
    four = assembly_cost(_packet(4), matrix_service=h)
    assert one.arithmetic == 256
    assert four.arithmetic == 304  # finite row latch serializes final merges
    assert one.sram == 23
    assert four.sram == 32
    assert four.accesses["vector_read_rows"] == 8
    assert four.accesses["vector_write_rows"] == 4
    assert one.accesses["matrix_bank_words"] == four.accesses["matrix_bank_words"] == 256


def test_narrow_ports_and_feedback_reduce_the_candidate_throughput():
    asm = _packet(4)
    h = MatrixService()
    fast = assembly_cost(asm, matrix_service=h)
    slow = assembly_cost(asm, matrix_service=replace(
        h, matrix_read_elements=128, vector_read_elements=128, vector_write_elements=128, mac_latency=4,
    ))
    assert slow.sram > fast.sram
    assert slow.arithmetic > fast.arithmetic
    assert slow.accesses["matrix_bank_words"] == fast.accesses["matrix_bank_words"]
    with pytest.raises(ValueError, match="Matrix SRAM"):
        assembly_cost(_packet(4), matrix_service=replace(h, matrix_capacity_bytes=524288))


@pytest.mark.parametrize("kwargs", [
    dict(cfg=(1 << 18) | (8 << 2) | (64 << 10)),
    dict(cfg=64 << 10),  # no input stride
    dict(source=2049),
    dict(source=131072),
    dict(destination=2050),
    dict(source=2048, destination=2048),
    dict(k=257),
])
def test_malformed_projection_packet_fails_closed(kwargs):
    with pytest.raises(ValueError):
        assembly_cost(_packet(**kwargs), matrix_service=MatrixService())


def test_profile_pins_schedule_and_capacities_and_refuses_fp32_mislabelling():
    p = ExecutionProfile()
    assert p.projection_schedule == "resident"
    assert not p.matrix.weight_replay
    for q in (
        replace(p, matrix=replace(p.matrix, weight_replay=True)),
        replace(p, projection_schedule="compact"),
        replace(p, projection_schedule="batch"),
        replace(p, projection_vector_rows=48, gather_vector_rows=48),
    ):
        assert q.identity != p.identity
    with pytest.raises(ValueError, match="BF16"):
        replace(p, projection_schedule="batch", matrix=replace(p.matrix, accumulator="FP32"))
    with pytest.raises(ValueError, match="boolean"):
        MatrixService(weight_replay=1)
    with pytest.raises(ValueError, match="does not model projection replay"):
        matrix_cost(4, 32, 256, MatrixService(weight_replay=True))


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("batch_tile", [1, 4])
def test_compiler_shares_weight_panels_and_charges_real_private_request_work(batch, batch_tile):
    from .ltile_layers import compiler_api
    from .ltile_program import ShapeArena

    root = Path(__file__).resolve().parents[2] / "PLENA_Compiler"
    _, Projection, _, _, _ = compiler_api(root)
    from compiler.aten.plena.isa_matrix_projection import lower_compact_projection

    arena = ShapeArena()
    zero = arena.add(2048)
    shape = Projection(0, 0, 0, zero, 7168, 64, 256)
    inputs = [arena.add(shape.input_values) for _ in range(batch)]
    weights = arena.add(shape.weight_bytes // 2)
    outputs = [arena.add(shape.output_values) for _ in range(batch)]
    p = replace(shape, inputs=inputs[0], weights=weights, outputs=outputs[0])
    assembly = lower_compact_projection(p, inputs, outputs, batch_tile=batch_tile)
    cost = assembly_cost(assembly, matrix_service=MatrixService(), trace_memory=True)
    weight_reads = sum(
        size for op, address, size in cost.memory_trace
        if op == "r" and weights <= address < weights + shape.weight_bytes
    )
    assert weight_reads == shape.weight_bytes
    assert cost.opcodes["M_MM.P"] == 2 * 28 * ((batch + batch_tile - 1) // batch_tile)
    assert cost.accesses["vector_write_rows"] == 2 * 28 * batch
    assert sum(size for op, _, size in cost.memory_trace if op == "w") == batch * 4096
