"""Regression checks for accounting errors which can invent a speedup."""

from pathlib import Path
import os
import json
from types import SimpleNamespace

import pytest

from .ltile_cost import Machine, View, assembly_cost, bank_service, dma_features
from .ltile_decode import unique_prepared_bytes, run as decode_run
from .ltile_program import build_program


@pytest.mark.parametrize("kind", ["mamba", "kda"])
def test_access_candidate_counts_real_bank_words_and_equal_capacity(kind):
    from .ltile_access import Shape, bank_waves

    shape = Shape(kind)
    assert shape.state_bytes == 512 * 1024
    for layout in ("head_major", "head_phase"):
        words = [word for row in range(128) for word in shape.stripe(row, 0, 2048, layout)]
        assert len(set(words)) == 8192
        assert max(address for bank, address in words) == 127
    assert len(bank_waves(shape.stripe(0, 0, 2048, "head_major"))) == 2
    assert len(bank_waves(shape.stripe(0, 0, 2048, "head_phase"))) == 1
    for layout in ("head_major", "head_phase"):
        assert len(bank_waves(shape.stripe(0, 0, 256, layout))) == 1


@pytest.mark.parametrize("kind,rows", [("mamba", 128), ("kda", 128), ("kda", 33)])
def test_native_coefficient_fetch_preserves_bf16_update_bits(kind, rows):
    from .ltile_access import Shape, verify_native_update

    result = verify_native_update(Shape(kind, rows), seed=20260924, tokens=4)
    assert result["coefficient_bits_exact"] and result["update_bits_exact"]


@pytest.mark.parametrize("width,credits,policy,blocked", [
    (256, 1, "shared", 0), (512, 2, "shared", 4),
    (2048, 4, "1r1w", 8), (1024, 3, "shared", 0),
])
def test_access_candidate_enforces_ports_backpressure_tags_and_tail(width, credits, policy, blocked):
    from .ltile_access import Shape, AccessConfig, simulate_update, validate_trace

    cfg = AccessConfig(read_values=width, output_slots=credits, port_policy=policy, blocked_every=blocked)
    result = simulate_update(Shape("kda", 33), cfg, layout="head_major", trace=True)
    assert validate_trace(result)
    assert result["peak_occupancy"]["input_slots"] <= cfg.input_slots
    assert result["peak_occupancy"]["output_slots"] <= credits


def test_wide_read_cannot_remove_compute_bound_or_invent_end_to_end_speedup():
    from .ltile_access import Shape, AccessConfig, simulate_update

    narrow = simulate_update(Shape("kda"), AccessConfig(read_values=256))
    wide = simulate_update(Shape("kda"), AccessConfig(read_values=2048))
    assert wide["counters"]["state_read_waves"] * 8 == narrow["counters"]["state_read_waves"]
    assert wide["counters"]["state_read_words"] == narrow["counters"]["state_read_words"]
    assert abs(wide["cycles"] - narrow["cycles"]) < 10
    assert wide["cycles"] >= wide["compute_lower_bound"]
    assert "whole layer" in wide["excluded"]


def test_native_supply_refills_are_not_free_and_result_credits_limit_throughput():
    from .ltile_access import Shape, AccessConfig, simulate_update

    shape = Shape("kda")
    native = simulate_update(shape)
    packed = simulate_update(shape, coefficients="packed")
    limited = simulate_update(shape, AccessConfig(output_slots=1))
    assert native["counters"]["coefficient_read_words"] == packed["counters"]["coefficient_read_words"]
    assert native["counters"]["coefficient_read_waves"] == 9  # input + four blocks of two fields
    assert native["cycles"] < packed["cycles"] < limited["cycles"]


def test_native_layout_cannot_reinterpret_already_written_state():
    from .ltile_access import Shape, Coefficients, AccessConfig, physical_word

    shape = Shape("kda")
    assert shape.state_word(0, 1, 0, "head_major") != shape.state_word(0, 1, 0, "head_phase")
    with pytest.raises(ValueError, match="512 KiB"):
        Shape("kda", 256)
    with pytest.raises(ValueError, match="multiple"):
        AccessConfig(lanes=512, read_values=256)
    with pytest.raises(ValueError, match="outside"):
        physical_word(16384)
    assert Coefficients(shape, "native").allocated_bytes <= 1024 * 1024


def test_native_descriptor_is_generic_bounded_and_rejects_reserved_bits():
    from .ltile_access import CoefficientView, Coefficients, Shape

    for kind in ("mamba", "kda"):
        source = Coefficients(Shape(kind), "native")
        for view in source.views.values():
            assert CoefficientView.unpack(view.pack()) == view
            for row in (0, 31, 32, 127):
                for head in range(view.heads):
                    assert CoefficientView.unpack(view.pack()).location(row, head) == view.location(row, head)
    with pytest.raises(ValueError, match="reserved"):
        CoefficientView.unpack(1 << 62)
    with pytest.raises(ValueError, match="extent"):
        CoefficientView(0, 128, 1, 0, 16, 2048).location(128, 15)


def test_candidate_state_phase_matches_compiled_view_addresses():
    from .ltile_access import Shape, bank_waves

    for kind in ("mamba", "kda"):
        shape = Shape(kind)
        view = View(shape.rows, shape.width, shape.heads, 0, shape.width // 32, False)
        for row in (0, 1, 63, 127):
            for first in range(0, shape.heads, 256 // shape.width):
                lines = tuple((h, row) for h in range(first, first + 256 // shape.width))
                words = shape.stripe(row, first * shape.width, 256, "head_phase")
                assert bank_service(0, view, lines) == (len(bank_waves(words)), len(words))


def test_sfu_drains_last_partial_subchunk_and_prices_both_sram_accesses():
    machine = Machine(sfu_lanes=300, sfu_ii=2, vector_softplus_cycles=13)
    cost = assembly_cost('V_SOFTPLUS_V gp1, gp2, 0\n', machine)
    assert cost.sram == 2
    assert cost.arithmetic == 6*2 + 13  # seven subchunks, no floor division
    assert cost.issue == 1


def test_sfu_does_not_silently_price_masked_execution():
    with pytest.raises(ValueError, match='masked SFU'):
        assembly_cost('V_SOFTPLUS_V gp1, gp2, 1\n')


def test_sfu_legacy_reciprocal_contract_unchanged():
    cost = assembly_cost('V_RECI_V gp1, gp2, 0\n')
    assert (cost.issue, cost.sram, cost.arithmetic) == (1, 2, 2)


def test_gate_cannot_inherit_one_cycle_whole_vector_sfu_by_accident():
    with pytest.raises(ValueError, match='finite-width'):
        assembly_cost('V_SOFTPLUS_V gp1, gp2, 0\n')


def test_original_dma_is_not_optimized_window_one():
    transfers = {("read", 4096): 1, ("write", 4096): 1}
    original = dma_features(transfers, "review")
    optimized = dma_features(transfers, "1")
    assert original["rmw_blocks"] == 64 and original["write_batches"] == 0
    assert optimized["rmw_blocks"] == 0 and optimized["write_batches"] == 64


def test_reused_kda_key_is_materialized_once():
    trace = [
        ("r", 0, 4096),
        ("w", 0, 4096),
        ("r", 8192, 4096),
        ("r", 8192, 4096),
        ("r", 12288, 128),
        ("r", 12288, 64),
        ("d", 100, 0),
    ]
    assert unique_prepared_bytes(trace) == 4224


def test_decode_rejects_an_unrelated_pass_label(tmp_path):
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"gate_passed": True, "model_contract": "instruction_count_only"}))
    with pytest.raises(ValueError, match="different predictor"):
        decode_run(tmp_path, tmp_path, SimpleNamespace(identity={}), tmp_path, gate_path=gate)


@pytest.mark.parametrize("width,heads,phase,conflict", [(64, 32, 2, 32), (128, 16, 4, 16)])
def test_fixed_diagonal_bank_phase_removes_same_packet_conflicts(width, heads, phase, conflict):
    # Address-only padded view, NOT the 512 KiB capacity-matched comparison:
    # heads*pitch=2048/4096 addresses per bank exceeds the actual depth=256.
    # Do not reuse its 16/32 conflict factor as a feasible full-tile baseline.
    flat = View(128, width, heads, 128, 0, False)
    phased = View(128, width, heads, 128, phase, False)
    lines = tuple((h, 0) for h in range(heads))
    assert bank_service(0, flat, lines) == (conflict, 64)
    assert bank_service(0, phased, lines) == (1, 64)


def test_unsupported_hardware_and_zero_loop_fail_closed():
    with pytest.raises(ValueError):
        Machine(lanes=2048)
    with pytest.raises(ValueError):
        Machine(banks=128)
    with pytest.raises(ValueError):
        assembly_cost("C_LOOP_START gp1, 0\nC_LOOP_END")
    with pytest.raises(ValueError):
        assembly_cost("UNKNOWN gp1")


def test_loop_counter_is_visible_in_nested_dma_addresses_and_zero_after_exit():
    # START initializes the ISA-visible counter; END decrements it. The outer
    # counter survives the inner loop. After either loop exits its counter is 0.
    scale_outer = "S_ADDI_INT gp3, gp5, 0\n" + "S_ADD_INT gp3, gp3, gp3\n" * 12
    scale_inner = "S_ADDI_INT gp4, gp6, 0\n" + "S_ADD_INT gp4, gp4, gp4\n" * 6
    asm = (
        "C_LOOP_START gp5, 2\n"
        + scale_outer
        + "C_LOOP_START gp6, 3\n"
        + scale_inner
        + "S_ADD_INT gp7, gp3, gp4\n"
        + "H_PREFETCH_V gp0, gp7, a0, 0, 2\n"
        + "C_LOOP_END gp6\n"
        + "H_PREFETCH_V gp0, gp6, a0, 0, 2\n"
        + "C_LOOP_END gp5\n"
        + "H_PREFETCH_V gp0, gp5, a0, 0, 2\n"
    )
    cost = assembly_cost(asm, trace_memory=True)
    assert [(a, n) for op, a, n in cost.memory_trace if op == "r"] == [
        (8384, 4096), (8320, 4096), (8256, 4096), (0, 4096),
        (4288, 4096), (4224, 4096), (4160, 4096), (0, 4096), (0, 4096),
    ]
    assert (cost.issue, cost.scalar, cost.sram) == (94, 85, 9)
    assert cost.opcodes["C_LOOP_START"] == 3
    assert cost.opcodes["C_LOOP_END"] == 8
    assert sum(a for op, a, _ in cost.memory_trace if op == "d") == cost.total


@pytest.mark.parametrize("asm,match", [
    ("C_LOOP_END gp1", "matching"),
    ("C_LOOP_START gp1, 2", "unterminated"),
    ("C_LOOP_START gp1, 2\nC_LOOP_END gp2", "innermost"),
    ("C_LOOP_START gp1, 2\nC_LOOP_END", "matching"),
    ("C_LOOP_START gp16, 2\nC_LOOP_END gp16", "GP register"),
    ("C_LOOP_START gp-1, 2\nC_LOOP_END gp-1", "GP register"),
    ("C_LOOP_START gp0, 2\nC_LOOP_END gp0", "nonzero"),
    ("C_LOOP_START gp1, 4194304\nC_LOOP_END gp1", "22-bit"),
    ("C_LOOP_START gp1, -1\nC_LOOP_END gp1", "22-bit"),
    ("C_LOOP_START gp1, 2\nC_LOOP_START gp1, 3\nC_LOOP_END gp1\nC_LOOP_END gp1", "distinct"),
    ("C_LOOP_START gp1, 2\nC_LOOP_START gp2, 3\nC_LOOP_END gp1\nC_LOOP_END gp2", "innermost"),
    ("C_LOOP_START gp1, 2\nS_ADDI_INT gp1, gp1, 1\nC_LOOP_END gp1", "mutate"),
    ("C_LOOP_START gp1, 2\nS_LUI_INT gp1, 0\nC_LOOP_END gp1", "mutate"),
    ("C_LOOP_START gp1, 2\nS_ADD_INT gp1, gp2, gp2\nC_LOOP_END gp1", "mutate"),
])
def test_loop_unsupported_control_fails_closed(asm, match):
    with pytest.raises(ValueError, match=match):
        assembly_cost(asm)


def test_loop_predictor_has_explicit_dynamic_instruction_bound():
    asm = "C_LOOP_START gp1, 10\nS_ADDI_INT gp2, gp1, 0\nC_LOOP_END gp1\n"
    assert assembly_cost(asm, max_instructions=21).issue == 21
    with pytest.raises(ValueError, match="exceeds max_instructions"):
        assembly_cost(asm, max_instructions=20)
    with pytest.raises(ValueError, match="positive integer"):
        assembly_cost(asm, max_instructions=True)


@pytest.mark.parametrize(
    "kind,control,expected",
    [
        ("mamba", "fsm", (836, 724, 48816, 10056, 0)),
        ("mamba", "row", (9028, 6884, 48816, 16152, 0)),
        ("kda", "fsm", (2882, 2546, 206592, 61728, 0)),
        ("kda", "row", (39722, 30242, 206592, 80016, 0)),
        ("mamba", "old_isa", (33575, 24287, 19576, 5144, 0)),
        ("kda", "old_isa", (165167, 118991, 89184, 21504, 0)),
    ],
)
def test_published_r3_component_oracles(kind, control, expected):
    root = os.environ.get("PLENA_E_COMPILER")
    if not root:
        pytest.skip("set PLENA_E_COMPILER to the pinned E compiler worktree")
    asm, _ = build_program(kind, 1, 4, control, compiler_root=Path(root))
    cost = assembly_cost(asm, Machine(), trace_memory=True)
    actual = tuple(cost.components()[k] for k in ("issue", "scalar", "sram", "arithmetic", "dependency"))
    assert actual == expected
    assert sum(a for op, a, _ in cost.memory_trace if op == "d") == cost.total
