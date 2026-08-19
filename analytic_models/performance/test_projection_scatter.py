from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from . import projection_scatter as scatter_module
from .projection_scatter import (
    BankedProjectionBuffer,
    ProjectionFifoSpillModel,
    ProjectionFlow,
    ScatterField,
    ScatterPlan,
    ServiceRun,
    StreamRun,
    _consumer_bank_stats,
    simulate_lowered_trace,
    verify_scatter_roundtrip,
)


FIXTURE = Path(__file__).parents[2] / "transactional_emulator/testdata/projection_scatter_v1_nemotron_decode.json"
KDA_FIXTURE = Path(__file__).parents[2] / "transactional_emulator/testdata/projection_scatter_v1_kimi_k3_decode.json"


def _document() -> dict:
    return json.loads(FIXTURE.read_text())


def _kda_document() -> dict:
    return json.loads(KDA_FIXTURE.read_text())


def test_compiler_golden_plan_is_bijective_and_checksum_valid() -> None:
    plan = ScatterPlan.from_dict(_document()["projection_scatters"][0]["plan"])
    assert plan.algorithm == "mamba2"
    assert plan.source_values_per_token == 10304
    assert plan.physical_values_per_token == 10368
    assert plan.compute_mapping_sha256() == plan.mapping_sha256


def test_ready_consumer_bypasses_state_fields_but_materializes_gate() -> None:
    report = simulate_lowered_trace(_document(), consumer_start_cycle=0)
    summary = report["summary"]
    assert summary["produced_values"] == 10304
    assert summary["direct_values"] == 6208
    assert summary["spill_values"] == 4096
    assert summary["spill_bytes"] == 8192
    assert summary["fifo_high_watermark"] <= 256
    assert summary["state_read_bank_stall_cycles"] == 0


def test_delayed_consumer_spills_packet_and_counts_bc_broadcast() -> None:
    report = simulate_lowered_trace(_document(), consumer_start_cycle=10000)
    summary = report["summary"]
    assert summary["direct_values"] == 0
    assert summary["spill_values"] == 10304
    assert summary["bc_value_reads"] == 2048
    assert summary["bc_broadcast_saved_reads"] == 14336

    without_broadcast = simulate_lowered_trace(_document(), consumer_start_cycle=10000, bc_broadcast=False)
    assert without_broadcast["summary"]["bc_value_reads"] == 16384
    assert without_broadcast["summary"]["bc_broadcast_saved_reads"] == 0


def test_fifo_backpressure_is_visible_when_spill_sink_is_too_narrow() -> None:
    model = ProjectionFifoSpillModel(
        flow=ProjectionFlow.BUFFERED,
        fifo_capacity_values=64,
        producer_burst_values=64,
        spill_write_values_per_cycle=1,
        consumer_start_cycle=0,
        consumer_values_per_cycle=16,
        activation_bytes=2,
    )
    stats = model.simulate(
        (StreamRun(0, 256, False),),
        producer_cycles=4,
    )
    assert stats.spill_values == 256
    assert stats.direct_values == 0
    assert stats.fifo_stall_cycles > 0
    assert stats.fifo_high_watermark == 64


def test_mapping_corruption_is_rejected() -> None:
    raw = _document()["projection_scatters"][0]["plan"]
    raw["mapping_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        ScatterPlan.from_dict(raw)


def test_contract_corruption_is_rejected() -> None:
    document = _document()
    document["projection_scatter_contract"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="contract checksum"):
        simulate_lowered_trace(document)


def _skewed_plan() -> ScatterPlan:
    return ScatterPlan.from_dict(_document()["projection_scatters"][0]["plan"])


def _row_major_plan() -> ScatterPlan:
    """The same packet, re-planned row-major, so the two can be compared."""
    raw = dict(_document()["projection_scatters"][0]["plan"])
    raw["layout"] = "row_major"
    raw["fields"] = [{**field, "skew_kind": "none", "skew_stride": 0} for field in raw["fields"]]
    plan = ScatterPlan(
        **{
            key: value
            for key, value in raw.items()
            if key in ScatterPlan.__dataclass_fields__ and key not in {"fields", "flow"}
        },
        fields=tuple(
            ScatterField(**{key: value for key, value in field.items() if key in ScatterField.__dataclass_fields__})
            for field in raw["fields"]
        ),
        flow=ProjectionFlow(raw["flow"]),
    )
    return replace(plan, mapping_sha256=plan.compute_mapping_sha256())


def test_skewed_mapping_delivers_every_value_to_the_right_lane() -> None:
    # mapping_sha256 proves both sides agree on the mapping and the stall counter
    # proves a packet could be serviced in one cycle. Neither shows the consumer
    # receives the value the projection wrote, so move real values through it.
    result = verify_scatter_roundtrip(_skewed_plan())
    assert result.written_values == 10304
    assert result.direct_values == 0
    assert result.banked_values == 10304
    assert result.read_values == 10304
    assert result.conflict_free
    assert result.stall_cycles == 0
    assert result.max_bank_multiplicity == 2  # 32-value packets over 16 banks


def test_row_major_returns_the_same_values_but_serializes_the_banks() -> None:
    skewed = verify_scatter_roundtrip(_skewed_plan())
    row_major = verify_scatter_roundtrip(_row_major_plan())
    # Same payload delivered either way: the layouts differ only in service time.
    assert row_major.read_values == skewed.read_values
    assert row_major.ideal_cycles == skewed.ideal_cycles
    assert not row_major.conflict_free
    assert row_major.max_bank_multiplicity == 8
    assert row_major.service_cycles > skewed.service_cycles


def test_roundtrip_service_cycles_match_the_counted_bank_stats() -> None:
    # The physical read and the analytic counter must not drift apart.
    plan = _skewed_plan()
    result = verify_scatter_roundtrip(plan)
    spilled = {0: set(range(plan.source_values_per_token))}
    state_reads, gate_reads, _, _ = _consumer_bank_stats(plan, spilled, True)
    assert result.service_cycles == state_reads.service_cycles + gate_reads.service_cycles
    assert result.ideal_cycles == state_reads.ideal_cycles + gate_reads.ideal_cycles


def test_roundtrip_catches_a_mapping_that_aliases_two_sources() -> None:
    class _AliasingPlan(ScatterPlan):
        def address(self, field_name, group, local_row, lane):
            source, row, _ = super().address(field_name, group, local_row, lane)
            return source, row, 0

    plan = _skewed_plan()
    broken = _AliasingPlan(**{key: getattr(plan, key) for key in ScatterPlan.__dataclass_fields__})
    with pytest.raises(ValueError, match="twice"):
        verify_scatter_roundtrip(broken)


def test_banked_buffer_rejects_double_writes_and_unwritten_reads() -> None:
    buffer = BankedProjectionBuffer(4, 8)
    buffer.write(0, 0, 42)
    with pytest.raises(ValueError, match="twice"):
        buffer.write(0, 0, 43)
    with pytest.raises(ValueError, match="unwritten"):
        buffer.read_packet(((0, 1),), 1)
    values, cycles, worst = buffer.read_packet(((0, 0),), 1)
    assert values == [42] and cycles == 1 and worst == 1


def test_roundtrip_reconstructs_fifo_direct_and_spilled_values() -> None:
    report = simulate_lowered_trace(
        _document(),
        consumer_start_cycle=0,
        roundtrip_tokens=1,
    )
    event = report["events"][0]
    roundtrip = event["roundtrip"]
    assert roundtrip["written_values"] == 10304
    assert roundtrip["direct_values"] == event["fifo"]["direct_values"] == 6208
    assert roundtrip["banked_values"] == event["fifo"]["spill_values"] == 4096
    assert roundtrip["read_values"] == 10304
    assert roundtrip["service_cycles"] == (
        event["state_reads"]["service_cycles"] + event["gate_reads"]["service_cycles"]
    )


def test_roundtrip_rejects_duplicate_and_omitted_consumer_sources(monkeypatch) -> None:
    plan = _skewed_plan()
    packets = list(scatter_module.consumer_packets(plan))
    index = next(index for index, packet in enumerate(packets) if len(packet.reads) >= 2)
    packet = packets[index]
    packets[index] = replace(
        packet,
        reads=(packet.reads[0], packet.reads[0], *packet.reads[2:]),
    )
    monkeypatch.setattr(scatter_module, "consumer_packets", lambda _: tuple(packets))
    with pytest.raises(ValueError, match=r"packet coverage.*missing=.*duplicated"):
        verify_scatter_roundtrip(plan)


def test_roundtrip_rejects_incomplete_service_runs() -> None:
    plan = _skewed_plan()
    with pytest.raises(ValueError, match="service runs cover"):
        verify_scatter_roundtrip(
            plan,
            service_runs=(ServiceRun(0, plan.total_values - 1, True),),
        )


def test_kda_k8_rotation_reaches_ideal_service_with_real_values() -> None:
    rotated = ScatterPlan.from_dict(_kda_document()["projection_scatters"][0]["plan"])
    row_major = _row_major_plan_for(rotated)
    rotated_result = verify_scatter_roundtrip(rotated)
    row_result = verify_scatter_roundtrip(row_major)
    assert rotated_result.read_values == row_result.read_values == 49248
    assert rotated_result.max_bank_multiplicity == 2
    assert rotated_result.service_cycles == rotated_result.ideal_cycles == 6240
    assert row_result.max_bank_multiplicity == 3
    assert row_result.service_cycles == 7776


def _row_major_plan_for(plan: ScatterPlan) -> ScatterPlan:
    fields = tuple(replace(field, skew_kind="none", skew_stride=0) for field in plan.fields)
    candidate = replace(plan, layout="row_major", fields=fields, mapping_sha256="")
    return replace(candidate, mapping_sha256=candidate.compute_mapping_sha256())
