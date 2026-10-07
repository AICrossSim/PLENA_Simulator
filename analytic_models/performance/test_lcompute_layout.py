import json
import sys
from pathlib import Path

import pytest

from .lcompute_layout import (
    BankGeometry,
    BankedLayoutBuffer,
    LayoutConfig,
    LayoutKind,
    LogicalCoord,
    multirow_word_packet,
    service_packet,
    simulate_fifo,
)


def test_physical_buffer_roundtrips_affine_layout():
    geometry = BankGeometry(16, 4)
    layout = LayoutConfig(LayoutKind.AFFINE_SKEW, 2, 3, 8, 64, alpha=1, beta=5, gamma=7)
    assert BankedLayoutBuffer(layout, geometry).roundtrip_all() == 2 * 3 * 8 * 64


def test_negative_alias_and_read_before_write_are_detected():
    geometry = BankGeometry(4, 1)
    with pytest.raises(ValueError, match="aliases"):
        BankedLayoutBuffer(LayoutConfig(LayoutKind.ROW_MAJOR, 1, 1, 2, 8, bank_row_pitch=1), geometry)

    buffer = BankedLayoutBuffer(LayoutConfig(LayoutKind.ROW_MAJOR, 1, 1, 1, 4), geometry)
    with pytest.raises(ValueError, match="before write"):
        buffer.read(LogicalCoord(0, 0, 0, 0))


def test_mamba_real_shape_multihead_word_conflict_is_removed_without_extra_bandwidth():
    # Nemotron 3 Nano: 64 Mamba heads, head_dim 64, state size 128.
    geometry = BankGeometry(16, 4)
    row = LayoutConfig(LayoutKind.ROW_MAJOR, groups=128, fields=1, majors=64, minors=64)
    skew = LayoutConfig(LayoutKind.AFFINE_SKEW, groups=128, fields=1, majors=64, minors=64, alpha=1)
    packet = multirow_word_packet(
        group=0,
        field=0,
        major_start=0,
        parallel_majors=8,
        minor_start=0,
        bank_width=geometry.bank_width,
    )
    assert len(packet) == 32 <= geometry.row_elements
    row_stats = service_packet(row, geometry, packet)
    skew_stats = service_packet(skew, geometry, packet)
    assert (row_stats.service_cycles, row_stats.bandwidth_floor_cycles) == (8, 1)
    assert (skew_stats.service_cycles, skew_stats.bandwidth_floor_cycles) == (1, 1)
    assert skew_stats.conflict_stall_cycles == 0


def test_kda_real_shape_multikey_word_conflict_is_removed():
    # Kimi K3: 96 heads, key/value dim 128. A state tile stores 64 value lanes;
    # the packet takes one four-value word from eight key rows.
    geometry = BankGeometry(16, 4)
    row = LayoutConfig(LayoutKind.ROW_MAJOR, groups=96, fields=2, majors=128, minors=64)
    skew = LayoutConfig(LayoutKind.AFFINE_SKEW, groups=96, fields=2, majors=128, minors=64, alpha=1, beta=8)
    packet = multirow_word_packet(
        group=0,
        field=0,
        major_start=0,
        parallel_majors=8,
        minor_start=0,
        bank_width=geometry.bank_width,
    )
    assert service_packet(row, geometry, packet).service_cycles == 8
    assert service_packet(skew, geometry, packet).service_cycles == 1


def test_two_full_rows_take_two_cycles_and_have_no_conflict_stall():
    geometry = BankGeometry(16, 4)
    layout = LayoutConfig(LayoutKind.AFFINE_SKEW, 1, 1, 2, 64, alpha=1)
    packet = list(layout.iter_coords())
    stats = service_packet(layout, geometry, packet)
    assert stats.values == 128
    assert stats.bandwidth_floor_cycles == 2
    assert stats.service_cycles == 2
    assert stats.conflict_stall_cycles == 0


def test_fifo_reports_backpressure_and_spill_separately():
    stalled = simulate_fifo(
        total_values=256,
        producer_values_per_cycle=64,
        consumer_values_per_cycle=16,
        capacity_values=64,
    )
    spilled = simulate_fifo(
        total_values=256,
        producer_values_per_cycle=64,
        consumer_values_per_cycle=16,
        capacity_values=64,
        spill_values_per_cycle=48,
    )
    assert stalled.stall_cycles > 0
    assert stalled.spilled_values == 0
    assert spilled.spilled_values > 0
    assert spilled.stall_cycles < stalled.stall_cycles


def test_compiler_and_simulator_execute_the_same_contract(tmp_path):
    compiler_root = Path(__file__).resolve().parents[2] / "PLENA_Compiler"
    if not compiler_root.exists():
        pytest.skip("PLENA_Compiler submodule is unavailable")
    if not (compiler_root / "aten/plena/affine_layout.py").exists():
        pytest.skip("pinned PLENA_Compiler predates the affine-layout contract")
    sys.path.insert(0, str(compiler_root))
    try:
        from compiler.aten.plena.affine_layout import (  # type: ignore[import-not-found]
            AffineLayout,
            BankGeometry as CompilerGeometry,
            LayoutKind,
        )
    finally:
        sys.path.pop(0)

    compiler_layout = AffineLayout(
        LayoutKind.AFFINE_SKEW,
        groups=2,
        fields=3,
        majors=8,
        minors=64,
        alpha=1,
        beta=5,
        gamma=7,
    )
    contract = compiler_layout.to_contract_dict(CompilerGeometry(16, 4))
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(contract))
    simulator_layout, simulator_geometry = LayoutConfig.from_contract(json.loads(path.read_text()))
    for logical in simulator_layout.iter_coords():
        c = compiler_layout.place(
            # Separate class on purpose: this catches field-order drift.
            __import__("compiler.aten.plena.affine_layout", fromlist=["LogicalCoord"]).LogicalCoord(
                logical.group, logical.field, logical.major, logical.minor
            ),
            CompilerGeometry(16, 4),
        )
        s = simulator_layout.place(logical, simulator_geometry)
        assert (c.bank, c.bank_row, c.sublane) == (s.bank, s.bank_row, s.sublane)
