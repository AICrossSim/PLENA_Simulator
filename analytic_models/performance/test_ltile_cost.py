"""Regression checks for accounting errors which can invent a speedup."""

from pathlib import Path
import os
import json
from types import SimpleNamespace

import pytest

from .ltile_cost import Machine, View, assembly_cost, bank_service, dma_features
from .ltile_decode import unique_prepared_bytes, run as decode_run
from .ltile_program import build_program


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
