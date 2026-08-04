"""Contracts for explicit TP and KV-cache-parallel decode."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ANALYTIC = Path(__file__).resolve().parents[1]
for _name in ("performance", "memory", "utilisation", "roofline", "disagg_serve"):
    _path = str(_ANALYTIC / _name)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import disagg_decode as decode  # noqa: E402
from compiler_trace_timing import (  # noqa: E402
    FULL_MODEL_DECODE_SCOPE,
    REFERENCE_DECODE_SCOPE,
)
from decode_results_table import build_parser, build_point  # noqa: E402
from physical_ledger import DecodeStepTrafficLedger  # noqa: E402


def _traffic() -> DecodeStepTrafficLedger:
    return DecodeStepTrafficLedger(
        weight_element_read_bytes=800,
        weight_scale_read_bytes=80,
        bf16_weight_read_bytes=160,
        activation_read_bytes=0,
        activation_write_bytes=0,
        kv_element_read_bytes=640,
        kv_scale_read_bytes=64,
        kv_element_write_bytes=32,
        kv_scale_write_bytes=8,
    )


class _TraceTimingProvider:
    """Small deterministic provider for decode-loop integration contracts."""

    def __init__(self, artifact_scope: str, *, result_scope: str | None = None):
        self.artifact_scope = artifact_scope
        self.result_scope = result_scope or artifact_scope
        self._results: dict[int, SimpleNamespace] = {}

    def _result(self, context: int) -> SimpleNamespace:
        digest = f"{context:064x}"
        compiler_digest = "1" * 64
        latency_digest = "2" * 64
        sidecar_digest = "3" * 64
        calibration_id = "request-latency-" + "4" * 64
        provenance = {
            "request_id": digest,
            "context_tokens": context,
            "artifact_scope": self.result_scope,
        }
        return SimpleNamespace(
            compiler_inputs_sha256="5" * 64,
            compiler_lowering_sha256="6" * 64,
            artifact_record_sha256=digest,
            context_tokens=context,
            batch=1,
            artifact_scope=self.result_scope,
            compiler_source_sha256=compiler_digest,
            latency_library_sha256=latency_digest,
            request_memory_sidecar_sha256=sidecar_digest,
            memory_calibration_id=calibration_id,
            provenance=provenance,
            reason="compiler_trace_timing_validated",
            frequency_hz=1.0e9,
            compute_cycles=6_000,
            memory_cycles=6_000,
            # Two sequential stages each take max(compute, memory):
            # max(5k, 1k) + max(1k, 5k) = 10k cycles. This is deliberately
            # different from max(sum(compute), sum(memory)) = 6k cycles.
            total_seconds=10_000 / 1.0e9,
        )

    def prepare(self, requests):
        results = []
        for request in requests:
            result = self._result(int(request))
            self._results[int(request)] = result
            results.append(result)
        return tuple(results)

    def evaluate(self, request):
        return self._results[int(request)]


class _LegacyBandwidthPoison:
    """Prove that compiler mode never touches aggregate-BW compatibility."""

    def operating_point_calibration_id(self, *_args, **_kwargs):
        raise AssertionError("compiler timing requested aggregate-BW evidence")

    def memory_time(self, *_args, **_kwargs):
        raise AssertionError("compiler timing used aggregate-BW pricing")


def test_tp_shards_weights_while_kvp_replicates_them() -> None:
    rank, system = decode._partition_step_traffic(_traffic(), tp=2, kvp=4)
    assert rank["weight_element_read_bytes"] == 400
    assert system["weight_element_read_bytes"] == 3_200
    assert rank["kv_element_read_bytes"] == 80
    assert system["kv_element_read_bytes"] == 640


def test_kv_residency_removes_the_declared_hbm_fraction() -> None:
    reduced = decode._traffic_for_policy(
        _traffic(),
        weights=type(
            "Weights",
            (),
            {"attention": type("Plane", (), {"element_aligned": 0, "scale_aligned": 0})()},
        )(),
        policy="kv_resident_50",
    )
    assert reduced.kv_element_read_bytes == 320
    assert reduced.kv_scale_write_bytes == 4
    assert reduced.weight_element_read_bytes == 800


def test_collectives_are_zero_only_for_single_chip() -> None:
    shape = {
        "hidden": 4096,
        "heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "layers": 32,
    }
    single = decode.collective_cost_per_step(
        shape,
        batch=8,
        tp=1,
        kvp=1,
        link_ports=0,
    )
    tensor = decode.collective_cost_per_step(
        shape,
        batch=8,
        tp=2,
        kvp=1,
        link_ports=1,
    )
    cache = decode.collective_cost_per_step(
        shape,
        batch=8,
        tp=1,
        kvp=2,
        link_ports=1,
    )
    mesh = decode.collective_cost_per_step(
        shape,
        batch=8,
        tp=2,
        kvp=2,
        link_ports=2,
    )
    assert single["time_s"] == single["total_bytes"] == 0.0
    assert tensor["tp_bytes"] > 0 and tensor["kvp_bytes"] == 0
    assert cache["kvp_bytes"] > 0 and cache["tp_bytes"] == 0
    assert mesh["total_bytes"] == pytest.approx(
        (mesh["tp_bytes"] + mesh["kvp_bytes"]) * 4
    )


def test_explicit_topology_must_match_chip_count() -> None:
    with pytest.raises(ValueError, match=r"differs from TP \* KVP"):
        decode._parallel_topology({"TP": 2, "KVP": 2}, n_chips=8)


def test_legacy_chip_count_remains_on_the_compatibility_path() -> None:
    topology = decode._parallel_topology({}, n_chips=7)
    assert topology["legacy_ideal_parallelism"] is True
    explicit = decode._parallel_topology(
        {"TP": 1, "KVP": 7, "LINK_PORTS": 1},
        n_chips=7,
    )
    assert explicit["legacy_ideal_parallelism"] is False


def test_explicit_single_chip_is_bit_identical_to_legacy_decode() -> None:
    args = build_parser().parse_args([])
    model, shape, hardware, memory, precision = build_point(
        args,
        "qwen3-32b",
    )
    common = {
        "model_path": model,
        "dims": shape,
        "hw_cfg": hardware,
        "isa_path": args.isa_lib,
        "base_mem": memory,
        "prec": precision,
        "batch": 1,
        "input_seq": 16,
        "output_seq": 2,
        "stride": 1,
        "n_chips": 1,
        "hbm_gen": args.hbm_gen,
        "hbm_channels": args.hbm_channels,
    }
    legacy = decode.evaluate(**common)
    explicit = decode.evaluate(
        **common,
        hw_over={
            "TP": 1,
            "KVP": 1,
            "LINK_PORTS": 0,
            "SRAM_POLICY": "streaming",
        },
    )
    invariant_keys = (
        "tpot",
        "tps",
        "total_time",
        "first_step",
        "avg_bytes_per_batch_step",
        "traffic_breakdown_per_batch_step",
        "hbm_required",
        "fits_runtime",
        "max_runtime_batch",
        "avg_peak_compute_seconds",
        "avg_ideal_compute_seconds",
        "avg_realized_compute_seconds",
        "avg_memory_seconds",
    )
    assert {key: explicit[key] for key in invariant_keys} == {
        key: legacy[key] for key in invariant_keys
    }


def test_explicit_reuse_switch_prices_per_head_kv_reads_only() -> None:
    legacy = decode._traffic_for_kv_head_reuse(
        _traffic(),
        kv_heads=4,
        kv_head_reuse=None,
    )
    reused = decode._traffic_for_kv_head_reuse(
        _traffic(),
        kv_heads=4,
        kv_head_reuse=True,
    )
    per_head = decode._traffic_for_kv_head_reuse(
        _traffic(),
        kv_heads=4,
        kv_head_reuse=False,
    )
    assert legacy == reused == _traffic()
    assert per_head.kv_element_read_bytes == 4 * legacy.kv_element_read_bytes
    assert per_head.kv_scale_read_bytes == 4 * legacy.kv_scale_read_bytes
    assert per_head.kv_element_write_bytes == legacy.kv_element_write_bytes
    assert per_head.weight_element_read_bytes == legacy.weight_element_read_bytes


def test_reuse_evidence_and_fp_sram_legality_are_explicit() -> None:
    hkv2 = decode.kv_head_reuse_status(
        enabled=True,
        mlen=1024,
        hlen=128,
        blen=2,
        kv_heads=2,
    )
    hkv4 = decode.kv_head_reuse_status(
        enabled=True,
        mlen=1024,
        hlen=128,
        blen=2,
        kv_heads=4,
    )
    qwen_legal = decode.kv_head_reuse_status(
        enabled=True,
        mlen=1024,
        hlen=128,
        blen=2,
        kv_heads=8,
    )
    qwen_illegal = decode.kv_head_reuse_status(
        enabled=True,
        mlen=1024,
        hlen=128,
        blen=4,
        kv_heads=8,
    )
    assert hkv2["measured_latency_delta_fraction"] == pytest.approx(-0.0061)
    assert hkv4["measured_latency_delta_fraction"] == pytest.approx(-0.0116)
    assert hkv2["evidence_tier"] == "transactional_emulator_measured"
    assert qwen_legal["required_fp_sram_slots"] == 390
    assert qwen_legal["supported"] is True
    assert qwen_illegal["required_fp_sram_slots"] == 774
    assert qwen_illegal["supported"] is False


def test_architecture_option_area_names_control_and_accumulator_bank() -> None:
    area = decode.architecture_option_area_mm2(
        mlen=1024,
        hlen=128,
        kv_heads=8,
        kv_head_reuse=True,
        drain_overlapped=True,
    )
    assert set(area["breakdown_mm2_per_chip"]) == {
        "KVHeadReuseControl",
        "DrainOverlapAccumulatorBank",
    }
    assert all(value > 0 for value in area["breakdown_mm2_per_chip"].values())
    bank = area["evidence"]["DrainOverlapAccumulatorBank"]
    assert bank["capacity_bytes"] == 576
    assert bank["tier"] == "published_sram_macro_geometry"
    assert area["evidence"]["KVHeadReuseControl"]["tier"] == (
        "declared_structural_estimate"
    )


def test_system_area_charges_each_option_on_every_chip() -> None:
    args = build_parser().parse_args([])
    _, shape, hardware, _, precision = build_point(args, "qwen3-32b")
    base = decode.system_area(
        hardware,
        precision,
        chip_count=2,
        link_ports=1,
        kv_heads=shape["kv_heads"],
    )
    enhanced = decode.system_area(
        hardware,
        precision,
        chip_count=2,
        link_ports=1,
        kv_head_reuse=True,
        drain_overlapped=True,
        kv_heads=shape["kv_heads"],
    )
    option = enhanced["architecture_options"]
    assert enhanced["area_mm2"] - base["area_mm2"] == pytest.approx(
        2 * option["area_mm2_per_chip"]
    )
    assert enhanced["chip_area_mm2"] - base["chip_area_mm2"] == (
        pytest.approx(option["area_mm2_per_chip"])
    )
    assert {
        "KVHeadReuseControl",
        "DrainOverlapAccumulatorBank",
    }.issubset(enhanced["chip"]["breakdown"])


def test_explicit_false_changes_kv_traffic_without_changing_storage() -> None:
    args = build_parser().parse_args([])
    model, shape, hardware, memory, precision = build_point(
        args,
        "qwen3-32b",
    )
    common = {
        "model_path": model,
        "dims": shape,
        "hw_cfg": hardware,
        "isa_path": args.isa_lib,
        "base_mem": memory,
        "prec": precision,
        "batch": 1,
        "input_seq": 16,
        "output_seq": 2,
        "stride": 1,
        "n_chips": 1,
        "hbm_gen": args.hbm_gen,
        "hbm_channels": args.hbm_channels,
    }
    legacy = decode.evaluate(**common)
    per_head = decode.evaluate(
        **common,
        hw_over={
            "TP": 1,
            "KVP": 1,
            "LINK_PORTS": 0,
            "SRAM_POLICY": "streaming",
            "KV_HEAD_REUSE": False,
            "DRAIN_OVERLAPPED": False,
        },
    )
    legacy_traffic = legacy["traffic_breakdown_per_generated_token"]
    per_head_traffic = per_head["traffic_breakdown_per_generated_token"]
    assert per_head["kv_footprint"] == legacy["kv_footprint"]
    assert per_head_traffic["kv_element_read_bytes"] == (
        shape["kv_heads"] * legacy_traffic["kv_element_read_bytes"]
    )
    assert per_head["architecture_options"]["explicit"] is True
    assert per_head["capacity_throughput_chain"]["max_feasible_batch"] == (
        per_head["max_runtime_batch"]
    )
    assert per_head["packed_q1_timing_reason"] == (
        "not_required_for_per_head_schedule"
    )


def test_explicit_reuse_binds_legal_schedule_timing_and_control_area() -> None:
    args = build_parser().parse_args([])
    model, shape, hardware, memory, precision = build_point(
        args,
        "qwen3-32b",
    )
    hardware = hardware.model_copy(update={"BLEN": 2})
    result = decode.evaluate(
        model,
        shape,
        hardware,
        args.isa_lib,
        memory,
        precision,
        1,
        16,
        2,
        hw_over={
            "TP": 1,
            "KVP": 1,
            "LINK_PORTS": 0,
            "SRAM_POLICY": "streaming",
            "KV_HEAD_REUSE": True,
            "DRAIN_OVERLAPPED": False,
        },
        stride=1,
        n_chips=1,
    )
    reuse = result["architecture_options"]["kv_head_reuse"]
    assert reuse["supported"] is True
    assert reuse["required_fp_sram_slots"] == 390
    assert reuse["traffic_reduction_vs_per_head"] == shape["kv_heads"]
    assert result["timing_calibrated"] is False
    assert result["timing_reason"] == "missing_packed_q1_timing_contract"
    assert "KVHeadReuseControl" in result[
        "architecture_options"
    ]["area"]["breakdown_mm2_per_chip"]


def test_drain_overlap_is_unrankable_without_matching_timing_evidence() -> None:
    args = build_parser().parse_args([])
    model, shape, hardware, memory, precision = build_point(
        args,
        "qwen3-32b",
    )
    result = decode.evaluate(
        model,
        shape,
        hardware,
        args.isa_lib,
        memory,
        precision,
        1,
        16,
        2,
        hw_over={
            "TP": 1,
            "KVP": 1,
            "LINK_PORTS": 0,
            "SRAM_POLICY": "streaming",
            "KV_HEAD_REUSE": False,
            "DRAIN_OVERLAPPED": True,
        },
        stride=1,
        n_chips=1,
    )
    drain = result["architecture_options"]["drain_overlapped"]
    assert result["timing_mode"] == "drain_overlapped"
    assert result["timing_calibrated"] is False
    assert drain["second_accumulator_bank_bytes_per_chip"] == 576
    assert drain["evidence_tier"] == "analytic_codesign_unrankable"
    assert "DrainOverlapAccumulatorBank" in result[
        "architecture_options"
    ]["area"]["breakdown_mm2_per_chip"]


def _evaluate_compiler_trace(provider: _TraceTimingProvider):
    args = build_parser().parse_args([])
    model, shape, hardware, memory, precision = build_point(
        args,
        "qwen3-32b",
    )
    hardware = hardware.model_copy(update={"BLEN": 2})
    return decode.evaluate(
        model,
        shape,
        hardware,
        args.isa_lib,
        memory,
        precision,
        1,
        16,
        2,
        hw_over={
            "TP": 1,
            "KVP": 1,
            "LINK_PORTS": 0,
            "SRAM_POLICY": "streaming",
            "KV_HEAD_REUSE": True,
            "DRAIN_OVERLAPPED": False,
        },
        stride=1,
        n_chips=1,
        bw_model=_LegacyBandwidthPoison(),
        execution_mode=decode.COMPILER_TRACE,
        trace_timing_provider=provider,
        trace_request_factory=lambda context: context,
    )


def test_compiler_trace_prices_sequential_stages_without_legacy_gates() -> None:
    result = _evaluate_compiler_trace(
        _TraceTimingProvider(FULL_MODEL_DECODE_SCOPE)
    )

    assert result["execution_mode"] == decode.COMPILER_TRACE
    assert result["tpot"] == pytest.approx(10_000 / 1.0e9)
    assert result["total_time"] == pytest.approx(20_000 / 1.0e9)
    assert result["avg_realized_compute_seconds"] == pytest.approx(
        6_000 / 1.0e9
    )
    assert result["avg_ideal_compute_seconds"] == pytest.approx(
        6_000 / 1.0e9
    )
    assert result["avg_memory_seconds"] == pytest.approx(6_000 / 1.0e9)
    assert result["tpot"] > max(
        result["avg_realized_compute_seconds"],
        result["avg_memory_seconds"],
    )
    assert result["timing_calibrated"] is True
    assert result["packed_q1_timing_reason"] == (
        "covered_by_full_model_compiler_trace"
    )
    assert result["bandwidth_calibration_id"] is None
    assert result["bandwidth_reason"] == "not_applicable_compiler_trace"
    assert decode.decode_bound_label(result) == "memory"
    assert decode.architecture_issue_bound_label(result) == "memory"
    evidence = result["compiler_trace_timing"]
    assert evidence["schema_version"] == (
        "plena-compiler-trace-timing-set-v1"
    )
    assert evidence["artifact_scope"] == FULL_MODEL_DECODE_SCOPE
    assert evidence["request_count"] == 2
    assert evidence["compiler_input_descriptor_sha256"] == "5" * 64
    assert evidence["compiler_lowering_key_sha256"] == "6" * 64
    assert evidence["request_memory_calibration_ids"] == [
        "request-latency-" + "4" * 64
    ]
    assert result["timing_evidence_id"] == (
        "compiler-trace-timing-" + decode.canonical_sha256(evidence)
    )


def test_compiler_trace_rejects_reference_decode_scope() -> None:
    with pytest.raises(ValueError, match="full-model independent-request"):
        _evaluate_compiler_trace(
            _TraceTimingProvider(REFERENCE_DECODE_SCOPE)
        )
