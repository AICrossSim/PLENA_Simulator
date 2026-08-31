"""Claim guards for the unified hybrid L-Compute campaign."""

from __future__ import annotations

import csv
import json
import os
from functools import cache, lru_cache
from pathlib import Path

from .hybrid_lcompute_campaign import (
    HardwarePoint,
    Variant,
    _sha256_json,
    _gpu_summary,
    _model,
    _packet_recurrence_service,
    build_layout_evidence,
    load_compiler_evidence,
    paper_2048_hardware_point,
    run_ablation,
    run_lane_dse,
    write_campaign_tables,
)
from .gpu_evidence import build_report as build_gpu_report
from .nemotron3_workload import InferencePhase, Precision


COMPILER_ROOT = Path(
    os.environ.get(
        "PLENA_COMPILER_ROOT",
        Path(__file__).resolve().parents[2] / "PLENA_Compiler",
    )
)
SIMULATOR_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _compiler() -> dict:
    return load_compiler_evidence(COMPILER_ROOT)


@cache
def _layout(hardware: HardwarePoint) -> dict:
    return build_layout_evidence(hardware, COMPILER_ROOT)


@lru_cache(maxsize=1)
def _paper_compiler() -> dict:
    return load_compiler_evidence(COMPILER_ROOT, paper_2048_hardware_point())


def _records(result: dict) -> dict[str, dict]:
    return {str(record["variant"]): record for record in result["records"]}


def test_base_geometry_preserves_regular_vector_rows() -> None:
    base = HardwarePoint()
    assert base.ordinary_row_read_cycles == 1
    assert base.binary_row_operand_cycles == 1
    assert not base.regular_vector_regression

    one_read_port = HardwarePoint(read_ports_per_bank=1)
    assert one_read_port.binary_row_operand_cycles == 2
    assert one_read_port.regular_vector_regression
    assert not HardwarePoint(layout_slots=2).stream_slots_sufficient


def test_paper_2048_geometry_preserves_regular_rows_and_recompiles_exact_packets() -> None:
    hardware = paper_2048_hardware_point()
    assert (hardware.blen, hardware.mlen, hardware.vector_lanes) == (32, 2048, 2048)
    assert (hardware.banks, hardware.bank_width) == (32, 64)
    assert hardware.ordinary_row_read_cycles == 1
    assert hardware.binary_row_operand_cycles == 1
    assert not hardware.regular_vector_regression
    proxies = hardware.resource_proxies()
    assert proxies["row_major_short_row_physical_rows_per_packet"] == 32
    assert proxies["affine_compact_physical_rows_per_packet"] == 1
    assert proxies["affine_packet_footprint_reduction"] == 32

    compiler = _paper_compiler()
    assert compiler["execution_config"]["packet_elements"] == 2048
    assert compiler["execution_config"]["recurrent_storage_row_elements"] == {
        "nemotron3": 64,
        "kimi_k3": 128,
    }
    assert compiler["assembly"]["nemotron_mamba_decode_recurrence"]["packet_affine"]["packetized_opcode_census"] == {
        "V_FMA_VF": 256,
        "V_MUL_VF": 256,
    }
    assert compiler["assembly"]["kimi_k3_decode_recurrent_mixer"]["packet_affine"]["packetized_opcode_census"] == {
        "V_FMA_VF": 768,
        "V_MUL_VF": 768,
    }


def test_paper_2048_full_decode_timeline_uses_exact_packets_and_removes_conflicts() -> None:
    hardware = paper_2048_hardware_point()
    compiler = _paper_compiler()
    layout = _layout(hardware)
    for model_name, layers, packet_ops_per_layer in (
        ("nemotron3", 52, 512),
        ("kimi_k3", 93, 1536),
    ):
        result = run_ablation(
            model_name,
            phase=InferencePhase.DECODE,
            tokens=1,
            context_length=2048,
            decode_tokens=4,
            hardware=hardware,
            compiler_root=COMPILER_ROOT,
            compiler_evidence=compiler,
            layout_evidence=layout,
        )
        assert result["schedule_validation"]["manifest_layers"] == layers
        records = _records(result)
        row = records[str(Variant.H_PACKET_ROW)]
        affine = records[str(Variant.I_PACKET_AFFINE)]
        expected = packet_ops_per_layer * (23 if model_name == "nemotron3" else 69) * 4
        assert row["packet_ops"] == affine["packet_ops"] == expected
        assert row["bank_conflict_stall_cycles"] > 0
        assert affine["bank_conflict_stall_cycles"] == 0
        assert affine["cycles"] < row["cycles"]
        assert affine["resource_utilization"]["matrix"] <= 1.0
        assert affine["resource_utilization"]["vector"] <= 1.0


def test_lane_dse_recompiles_each_width_instead_of_scaling_counts() -> None:
    hardware = paper_2048_hardware_point()
    dse = run_lane_dse(
        hardware,
        compiler_root=COMPILER_ROOT,
        base_compiler_evidence=_paper_compiler(),
    )
    assert [row["hardware"]["vector_lanes"] for row in dse["records"]] == [
        64,
        128,
        256,
        512,
        1024,
        2048,
    ]
    for model_name in ("nemotron3", "kimi_k3"):
        packet_ops = [row["models"][model_name]["I_packet_ops"] for row in dse["records"]]
        assert all(packet_ops[index] == 2 * packet_ops[index + 1] for index in range(len(packet_ops) - 1))


def test_compiler_and_gpu_evidence_are_the_pinned_real_shapes() -> None:
    compiler = _compiler()
    assert compiler["workloads"]["nemotron3"]["layer_counts"] == {
        "gqa": 6,
        "mamba": 23,
        "moe": 23,
    }
    assert compiler["workloads"]["kimi_k3"]["layer_counts"] == {
        "dense_ffn": 1,
        "kda": 69,
        "latent_moe": 92,
        "mla": 24,
    }
    assert compiler["workloads"]["kimi_k3"]["dimensions"]["kda_heads"] == 96
    assert compiler["workloads"]["nemotron3"]["dimensions"]["mamba_projection_width"] == 10_304

    gpu = _gpu_summary(build_gpu_report())
    assert gpu["kda_shape"]["num_heads"] == 96
    assert gpu["kda_shape"]["head_dim"] == 128
    assert gpu["nemotron_decode_itl_median_ms"] > 0
    assert "not a PLENA cycle" in gpu["evidence_use"]


def test_default_full_model_runs_use_pinned_checkpoint_mixed_weight_precision() -> None:
    nemotron = _model(
        "nemotron3",
        COMPILER_ROOT,
        activation_precision=Precision.BF16,
        weight_precision=None,
        state_precision=Precision.FP32,
    )
    n_policy = nemotron.weight_precision_policy
    assert n_policy is not None
    assert n_policy.precision_for(0, "mamba_in_projection") == Precision.NVFP4
    assert n_policy.precision_for(4, "mamba_in_projection") == Precision.BF16
    assert n_policy.precision_for(5, "attention_qkv_projection") == Precision.BF16

    kimi = _model(
        "kimi_k3",
        COMPILER_ROOT,
        activation_precision=Precision.BF16,
        weight_precision=None,
        state_precision=Precision.FP32,
    )
    k_policy = kimi.weight_precision_policy
    assert k_policy is not None
    assert k_policy.precision_for(0, "kda_qkv_projection") == Precision.BF16
    assert k_policy.precision_for(1, "latent_moe_routed_experts") == Precision.MXFP4
    assert k_policy.precision_for(1, "latent_moe_shared_experts") == Precision.BF16


def test_ablation_separates_arlo_stride_from_consumer_major_and_affine() -> None:
    hardware = HardwarePoint()
    result = run_ablation(
        "nemotron3",
        phase=InferencePhase.DECODE,
        tokens=1,
        context_length=2048,
        decode_tokens=1,
        hardware=hardware,
        compiler_root=COMPILER_ROOT,
        compiler_evidence=_compiler(),
        layout_evidence=_layout(hardware),
    )
    records = _records(result)
    a = records[str(Variant.A_ROW_GATHER)]
    b = records[str(Variant.B_ARLO_POSTINC)]
    c = records[str(Variant.C_CONSUMER_MAJOR)]
    d = records[str(Variant.D_AFFINE_LAYOUT)]
    e = records[str(Variant.E_STREAM_ADDRESSING)]
    f = records[str(Variant.F_AFFINE_STREAM)]
    g = records[str(Variant.G_OVERLAP)]
    h = records[str(Variant.H_PACKET_ROW)]
    i = records[str(Variant.I_PACKET_AFFINE)]
    j = records[str(Variant.J_PACKET_AFFINE_OVERLAP)]

    # B and C remove the same explicit gather, but only C changes producer
    # order.  Neither receives affine bank-conflict credit.
    assert b["bank_conflict_stall_cycles"] == c["bank_conflict_stall_cycles"] == 0
    assert a["layout_service_cycles"] > b["layout_service_cycles"]
    # C differs from B only in physical write order.  On the current serial
    # one-row consumer there is no executable bank benefit to claim.
    assert c["cycles"] == b["cycles"]
    local = _layout(hardware)["nemotron_mamba_projection"]
    assert (
        local["scores"][local["selected"]]["conflict_stall_cycles"]
        < local["scores"]["row_major"]["conflict_stall_cycles"]
    )
    assert d["lane_restore_cycles"] > 0
    assert e["cycles"] < b["cycles"]
    assert f["cycles"] >= e["cycles"]
    assert g["cycles"] <= f["cycles"]

    # H and I execute the same actual Mamba packet arithmetic. Only the
    # physical row rotation changes: identity placement conflicts, alpha=1
    # does not. E remains the fair ordinary-row baseline.
    assert h["packet_ops"] == i["packet_ops"] == 23 * 16_384
    assert h["bank_conflict_stall_cycles"] > 0
    assert i["bank_conflict_stall_cycles"] == 0
    assert i["cycles"] < h["cycles"]
    assert j["cycles"] <= i["cycles"]
    assert e["cycles"] <= i["cycles"]

    # D is a projection-only layout ablation. Recurrent packet costs belong to
    # H/I/J and must not leak into this earlier variant.
    selected = local["scores"][local["selected"]]
    expected = (selected["write_cycles"] - selected["write_floor_cycles"] + selected["lane_restore_cycles"]) * 23
    assert d["layout_service_cycles"] == expected


def test_packet_service_matches_the_rust_16_bank_counter_contract() -> None:
    hardware = HardwarePoint()
    pair = {
        "packet_row_major": {
            "opcode_census": {"V_FMA_VF": 16},
            "packetized_opcode_census": {"V_FMA_VF": 16},
        },
        "packet_affine": {
            "opcode_census": {"V_FMA_VF": 16},
            "packetized_opcode_census": {"V_FMA_VF": 16},
        },
    }
    row = _packet_recurrence_service(pair, "packet_row_major", hardware)
    affine = _packet_recurrence_service(pair, "packet_affine", hardware)

    # Pinned by the Rust connected dispatch test: 32 reads and 16 writes.
    assert row["read_packets"] == affine["read_packets"] == 32
    assert row["write_packets"] == affine["write_packets"] == 16
    assert row["service_cycles"] == 400
    assert row["bandwidth_floor_cycles"] == 48
    assert row["conflict_stall_cycles"] == 352
    assert affine["service_cycles"] == affine["bandwidth_floor_cycles"] == 48
    assert affine["conflict_stall_cycles"] == 0


def test_explicit_state_tile_transfers_only_at_decode_boundaries() -> None:
    hardware = HardwarePoint(explicit_state_resident_bytes=3 * 1024 * 1024)
    result = run_ablation(
        "nemotron3",
        phase=InferencePhase.DECODE,
        tokens=1,
        context_length=2048,
        decode_tokens=4,
        hardware=hardware,
        compiler_root=COMPILER_ROOT,
        compiler_evidence=_compiler(),
        layout_evidence=_layout(hardware),
    )
    resident = result["residency"]["resident_bytes"]
    assert resident > 0
    for record in result["records"]:
        assert record["explicit_state_transfer_bytes"] == 2 * resident


def test_zero_fifo_is_legal_and_serializes_the_overlap_boundary() -> None:
    no_fifo = HardwarePoint(fifo_values=0)
    full_packet_fifo = HardwarePoint(fifo_values=64)
    common = {
        "model_name": "nemotron3",
        "phase": InferencePhase.DECODE,
        "tokens": 1,
        "context_length": 2048,
        "decode_tokens": 1,
        "compiler_root": COMPILER_ROOT,
        "compiler_evidence": _compiler(),
    }
    no_fifo_result = run_ablation(
        hardware=no_fifo,
        layout_evidence=_layout(no_fifo),
        **common,
    )
    fifo_result = run_ablation(
        hardware=full_packet_fifo,
        layout_evidence=_layout(full_packet_fifo),
        **common,
    )
    no_fifo_g = _records(no_fifo_result)[str(Variant.G_OVERLAP)]
    fifo_g = _records(fifo_result)[str(Variant.G_OVERLAP)]
    assert no_fifo_g["fifo_stall_cycles"] > 0
    assert no_fifo_g["cycles"] >= fifo_g["cycles"]


def test_full_schedules_and_compressed_mla_are_validated_in_both_phases() -> None:
    hardware = HardwarePoint()
    for model_name, layers in (("nemotron3", 52), ("kimi_k3", 93)):
        for phase, tokens in (
            (InferencePhase.PREFILL, 16),
            (InferencePhase.DECODE, 1),
        ):
            result = run_ablation(
                model_name,
                phase=phase,
                tokens=tokens,
                context_length=2048,
                decode_tokens=1,
                hardware=hardware,
                compiler_root=COMPILER_ROOT,
                compiler_evidence=_compiler(),
                layout_evidence=_layout(hardware),
            )
            validation = result["schedule_validation"]
            assert validation["validated"]
            assert validation["manifest_layers"] == layers
            if phase == InferencePhase.DECODE:
                records = _records(result)
                packet_ops = 23 * 16_384 if model_name == "nemotron3" else 69 * 49_152
                row = records[str(Variant.H_PACKET_ROW)]
                affine = records[str(Variant.I_PACKET_AFFINE)]
                assert row["packet_ops"] == affine["packet_ops"] == packet_ops
                assert row["bank_conflict_stall_cycles"] > 0
                assert affine["bank_conflict_stall_cycles"] == 0
                assert affine["cycles"] < row["cycles"]
            if model_name == "kimi_k3":
                assert validation["compressed_mla"] == {
                    "elements_per_token": 576,
                    "bytes_per_token": 1152,
                    "expanded_96_head_kv_materialized": False,
                }


def test_full_model_uses_one_shared_hbm_timeline() -> None:
    hardware = HardwarePoint()
    result = run_ablation(
        "nemotron3",
        phase=InferencePhase.DECODE,
        tokens=1,
        context_length=2048,
        decode_tokens=1,
        hardware=hardware,
        compiler_root=COMPILER_ROOT,
        compiler_evidence=_compiler(),
        layout_evidence=_layout(hardware),
    )
    record = _records(result)[str(Variant.G_OVERLAP)]
    assert record["timeline_event_count"] > result["schedule_validation"]["stage_count"]
    assert record["resource_queue_wait_cycles"]["hbm"] > 0
    assert set(record["resource_busy_cycles"]) >= {"hbm", "matrix", "vector"}
    assert all(value <= 1.0 for value in record["resource_utilization"].values())


def test_campaign_tables_are_flat_and_nonempty(tmp_path: Path) -> None:
    record = {
        "variant": "B",
        "cycles": 10,
        "speedup_vs_arlo_B": 1.0,
        "logical_hbm_read_bytes": 64,
        "logical_hbm_write_bytes": 0,
        "physical_hbm_read_bytes": 64,
        "physical_hbm_write_bytes": 0,
        "layout_service_cycles": 0,
        "bank_conflict_stall_cycles": 0,
        "lane_restore_cycles": 0,
        "fifo_stall_cycles": 0,
        "timeline_event_count": 2,
        "resource_busy_cycles": {"hbm": 1},
        "resource_queue_wait_cycles": {"hbm": 0},
    }
    report = {
        "experiments": {
            "model": {
                "decode": {
                    "phase": "decode",
                    "tokens": 1,
                    "records": [record],
                    "schedule_validation": {
                        "validated": True,
                        "manifest_layers": 1,
                    },
                }
            }
        },
        "dse": {
            "records": [
                {
                    "hardware": {"name": "base", "banks": 16},
                    "eligible_for_freeze": True,
                    "models": {"model": {"B_cycles": 10}},
                }
            ]
        },
        "precision_dse": {
            "records": [
                {
                    "model": "model",
                    "state_precision": "fp32",
                    "accuracy": {"status": "measured"},
                }
            ]
        },
    }
    write_campaign_tables(report, tmp_path)
    for name in ("ablation.csv", "dse.csv", "precision.csv", "schedule_validation.csv"):
        with (tmp_path / name).open(newline="") as source:
            assert len(list(csv.DictReader(source))) == 1


def test_checked_in_long_campaign_is_self_consistent() -> None:
    path = SIMULATOR_ROOT / "artifacts/hybrid_lcompute_packet_v2/campaign.json"
    report = json.loads(path.read_text())
    claimed_hash = report.pop("report_sha256")
    assert claimed_hash == _sha256_json(report)
    assert report["schema_version"] == 4
    assert report["compiler_report_sha256"] == _sha256_json(_compiler())

    decision = report["dse"]["base_decision"]
    assert decision["stream_addressing_earns_isa"]
    assert decision["affine_packet_eliminates_conflicts"]
    assert not decision["affine_packet_beats_best_ordinary_row"]

    for model in ("nemotron3", "kimi_k3"):
        decode = _records(report["experiments"][model]["decode_32"])
        assert decode[str(Variant.H_PACKET_ROW)]["bank_conflict_stall_cycles"] > 0
        assert decode[str(Variant.I_PACKET_AFFINE)]["bank_conflict_stall_cycles"] == 0
        assert decode[str(Variant.I_PACKET_AFFINE)]["packet_ops"] > 0
        prefill = _records(report["experiments"][model]["prefill_s128"])
        assert prefill[str(Variant.I_PACKET_AFFINE)]["packet_ops"] == 0


def test_checked_in_paper_2048_campaign_is_self_consistent() -> None:
    path = SIMULATOR_ROOT / "artifacts/hybrid_lcompute_paper2048_v1/campaign.json"
    report = json.loads(path.read_text())
    claimed_hash = report.pop("report_sha256")
    assert claimed_hash == _sha256_json(report)
    assert report["schema_version"] == 4
    assert report["compiler_report_sha256"] == _sha256_json(_paper_compiler())
    assert report["paper_alignment"]["matched_by_this_run"]
    assert (
        report["hardware"]["blen"],
        report["hardware"]["mlen"],
        report["hardware"]["vector_lanes"],
    ) == (32, 2048, 2048)
    assert report["hardware"]["mamba_recurrent_row_elements"] == 64
    assert report["hardware"]["kda_recurrent_row_elements"] == 128
    assert not report["resource_proxies"]["regular_vector_regression"]

    decision = report["dse"]["base_decision"]
    assert decision["stream_addressing_earns_isa"]
    assert decision["affine_packet_eliminates_conflicts"]
    assert decision["affine_packet_beats_best_ordinary_row"]

    lanes = [point["hardware"]["vector_lanes"] for point in report["lane_dse"]["records"]]
    assert lanes == [64, 128, 256, 512, 1024, 2048]
    for model in ("nemotron3", "kimi_k3"):
        decode = _records(report["experiments"][model]["decode_32"])
        row = decode[str(Variant.H_PACKET_ROW)]
        affine = decode[str(Variant.I_PACKET_AFFINE)]
        ordinary = decode[str(Variant.E_STREAM_ADDRESSING)]
        assert row["bank_conflict_stall_cycles"] > 0
        assert affine["bank_conflict_stall_cycles"] == 0
        assert affine["cycles"] < row["cycles"]
        assert affine["cycles"] < ordinary["cycles"]
        prefill = _records(report["experiments"][model]["prefill_s128"])
        assert prefill[str(Variant.I_PACKET_AFFINE)]["packet_ops"] == 0
