"""Claim guards for the unified hybrid L-Compute campaign."""

from __future__ import annotations

import csv
from functools import cache, lru_cache
from pathlib import Path

from .hybrid_lcompute_campaign import (
    HardwarePoint,
    Variant,
    _gpu_summary,
    _model,
    build_layout_evidence,
    load_compiler_evidence,
    run_ablation,
    write_campaign_tables,
)
from .gpu_evidence import build_report as build_gpu_report
from .nemotron3_workload import InferencePhase, Precision


COMPILER_ROOT = Path(__file__).resolve().parents[2] / "PLENA_Compiler"


@lru_cache(maxsize=1)
def _compiler() -> dict:
    return load_compiler_evidence(COMPILER_ROOT)


@cache
def _layout(hardware: HardwarePoint) -> dict:
    return build_layout_evidence(hardware, COMPILER_ROOT)


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

    # Only incremental projection write/restore cost is executable. The state
    # multirow packet upper bound and counterfactual row-packet conflicts must
    # not leak into the end-to-end result.
    selected = local["scores"][local["selected"]]
    expected = (selected["write_cycles"] - selected["write_floor_cycles"] + selected["lane_restore_cycles"]) * 23
    assert d["layout_service_cycles"] == expected


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
