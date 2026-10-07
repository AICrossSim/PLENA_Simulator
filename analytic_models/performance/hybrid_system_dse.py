"""Run the pre-RTL full-system DSE for Nemotron 3 and Kimi K3.

This driver executes each model's complete ordered text backbone on the shared
resource timeline. It reports assumed PLENA cycles and byte traffic separately
from measured GPU evidence; GPU time is never used as a PLENA latency constant.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Any

from transactional_emulator.testbench.model_configs.loader import load_model_config

from .hybrid_routing_trace import load_kimi_routing_trace
from .hybrid_system_timeline import (
    HybridSystemTimelineModel,
    ModelFamily,
    PrecisionConfig,
    SystemDesign,
    SystemTimelineReport,
)
from .kimi_k3_precision import run_kda_precision_experiment
from .kimi_k3_workload import KimiK3Architecture
from .nemotron3_dse import ProjectionLayout, StateCachePolicy
from .nemotron3_precision import run_state_precision_experiment
from .nemotron3_workload import InferencePhase, Precision, storage_bytes


MIB = 1024 * 1024


def formal_precision(model: ModelFamily) -> PrecisionConfig:
    if model == ModelFamily.NEMOTRON3:
        return PrecisionConfig(
            activation=Precision.BF16,
            weight=Precision.NVFP4,
            state=Precision.FP32,
            conv_state=Precision.FP32,
        )
    return PrecisionConfig(
        activation=Precision.MXFP8,
        weight=Precision.MXFP4,
        state=Precision.FP32,
        conv_state=Precision.BF16,
    )


def persistent_state_bytes(model: ModelFamily, precision: PrecisionConfig) -> int:
    if model == ModelFamily.NEMOTRON3:
        arch = load_model_config("nemotron3_nano_30b_a3b").arch
        assert arch.mamba is not None
        layer_bytes = storage_bytes(arch.mamba.state_elements, precision.state)
        layer_bytes += storage_bytes(
            arch.mamba.conv_channels * arch.mamba.conv_kernel,
            precision.resolved_conv_state(model),
        )
        return sum(kind == "mamba" for kind in arch.layer_types) * layer_bytes
    arch = KimiK3Architecture()
    return arch.recurrent_state_bytes(precision.state) + arch.conv_state_bytes(precision.resolved_conv_state(model))


def selected_design(model: ModelFamily, precision: PrecisionConfig) -> SystemDesign:
    del precision
    return SystemDesign(
        name="shared_hybrid_pre_rtl_candidate",
        matrix_macs_per_cycle=4096,
        vector_ops_per_cycle=256,
        state_dim_lanes=8,
        projection_buffer_banks=16,
        projection_fifo_values=64,
        projection_layout=ProjectionLayout.GROUP_MAJOR_SKEWED,
        projection_direct_bypass=True,
        bc_broadcast=True,
        state_cache_bytes=32 * MIB,
        state_cache_policy=StateCachePolicy.PINNED,
        kv_cache_bytes=64 * MIB,
        moe_weight_cache_bytes=256 * MIB,
    )


def _summary(report: SystemTimelineReport) -> dict[str, Any]:
    document = report.to_dict(include_stages=False)
    metrics = document["metrics"]
    return {
        "model": report.model.value,
        "phase": report.phase.value,
        "design": document["design"],
        "precision": document["precision"],
        "model_ready_cycles": metrics["model_ready_cycles"],
        "final_cycle": metrics["final_cycle"],
        "ttft_us": metrics["ttft_us"],
        "tpot_us": metrics["tpot_us"],
        "prefill_cycles_per_prompt_token": metrics["prefill_cycles_per_prompt_token"],
        "logical_hbm_read_bytes": metrics["logical_hbm_read_bytes"],
        "logical_hbm_write_bytes": metrics["logical_hbm_write_bytes"],
        "physical_burst_hbm_read_bytes": metrics["physical_burst_hbm_read_bytes"],
        "physical_burst_hbm_write_bytes": metrics["physical_burst_hbm_write_bytes"],
        "bank_service_cycles": metrics["bank_service_cycles"],
        "bank_stall_cycles": metrics["bank_stall_cycles"],
        "resource_busy_cycles": metrics["resource_busy_cycles"],
        "resource_queue_wait_cycles": metrics["resource_queue_wait_cycles"],
        "resource_utilization": metrics["resource_utilization"],
        "state_cache": document["state_cache"],
        "kv_cache": document["kv_cache"],
        "moe_weight_cache": document["moe_weight_cache"],
    }


def _simulate(
    model: ModelFamily,
    design: SystemDesign,
    precision: PrecisionConfig,
    phase: InferencePhase,
    *,
    context_length: int,
    decode_tokens: int,
    prefill_tokens: int,
    kimi_routing_trace_path: Path | None = None,
) -> SystemTimelineReport:
    return HybridSystemTimelineModel(
        model,
        design,
        precision,
        kimi_routing_trace_path=kimi_routing_trace_path,
    ).simulate(
        phase,
        context_length=context_length,
        decode_tokens=decode_tokens,
        sequence_length=prefill_tokens,
        include_embedding=True,
        include_lm_head=True,
    )


def _unique_designs(designs: Iterable[SystemDesign]) -> list[SystemDesign]:
    result = []
    seen = set()
    for design in designs:
        key = tuple(sorted((key, str(value)) for key, value in asdict(design).items() if key != "name"))
        if key not in seen:
            seen.add(key)
            result.append(design)
    return result


def _candidate_designs(
    base: SystemDesign,
    *,
    grid: str,
) -> list[SystemDesign]:
    if grid == "quick":
        candidates = [base]
        candidates.extend(
            replace(base, name=f"matrix_{value}", matrix_macs_per_cycle=value) for value in (2048, 4096, 8192)
        )
        candidates.extend(replace(base, name=f"state_lanes_{value}", state_dim_lanes=value) for value in (4, 8, 16))
        candidates.extend(replace(base, name=f"banks_{value}", projection_buffer_banks=value) for value in (16, 32))
        candidates.extend(replace(base, name=f"fifo_{value}", projection_fifo_values=value) for value in (64, 128, 256))
        # HBM bandwidth is the one parameter the rest of this grid holds fixed,
        # and it decides roughly 97% of decode time at the 64 B/cycle candidate.
        # Without these points every other sensitivity here is quoted at a single
        # unverified bandwidth, and the on-chip mechanisms look uniformly
        # irrelevant because the memory server hides them.
        candidates.extend(
            replace(base, name=f"hbm_{value}", hbm_bytes_per_cycle=value)
            for value in (64, 128, 256, 512, 1024, 2048, 4096, 8192)
        )
        for capacity_mib in (0, 24, 32, 64):
            capacity = capacity_mib * MIB
            candidates.append(
                replace(
                    base,
                    name=f"state_cache_{capacity_mib}m",
                    state_cache_bytes=capacity,
                    state_cache_policy=(StateCachePolicy.NONE if capacity == 0 else StateCachePolicy.PINNED),
                )
            )
        return _unique_designs(candidates)

    values = product(
        (2048, 4096, 8192),
        (4, 8, 16),
        (16, 32),
        (64, 128, 256),
        tuple(value * MIB for value in (0, 16, 24, 32, 48, 64)),
    )
    candidates = []
    for matrix, state_lanes, banks, fifo, cache_bytes in values:
        candidates.append(
            replace(
                base,
                name=(f"m{matrix}_s{state_lanes}_b{banks}_f{fifo}_c{cache_bytes // MIB}m"),
                matrix_macs_per_cycle=matrix,
                state_dim_lanes=state_lanes,
                projection_buffer_banks=banks,
                projection_fifo_values=fifo,
                state_cache_bytes=cache_bytes,
                state_cache_policy=(StateCachePolicy.NONE if cache_bytes == 0 else StateCachePolicy.PINNED),
            )
        )
    return candidates


def _resource_vector(record: dict[str, Any]) -> tuple[float, ...]:
    design = record["design"]
    return (
        record["model_ready_cycles"],
        record["logical_hbm_read_bytes"] + record["logical_hbm_write_bytes"],
        design["matrix_macs_per_cycle"],
        design["state_macs_per_cycle"],
        design["projection_buffer_banks"],
        design["projection_fifo_values"],
        design["activation_sram_bytes"],
        design["state_cache_bytes"],
    )


def _pareto(records: list[dict[str, Any]]) -> list[str]:
    vectors = [_resource_vector(record) for record in records]
    result = []
    for index, candidate in enumerate(vectors):
        dominated = any(
            all(other_item <= candidate_item for other_item, candidate_item in zip(other, candidate, strict=True))
            and any(other_item < candidate_item for other_item, candidate_item in zip(other, candidate, strict=True))
            for other_index, other in enumerate(vectors)
            if other_index != index
        )
        if not dominated:
            result.append(records[index]["design"]["name"])
    return result


def _dse(
    model: ModelFamily,
    base: SystemDesign,
    precision: PrecisionConfig,
    *,
    grid: str,
    context_length: int,
    decode_tokens: int,
    prefill_tokens: int,
    kimi_routing_trace_path: Path | None,
) -> dict[str, Any]:
    records = [
        _summary(
            _simulate(
                model,
                design,
                precision,
                InferencePhase.DECODE,
                context_length=context_length,
                decode_tokens=decode_tokens,
                prefill_tokens=prefill_tokens,
                kimi_routing_trace_path=kimi_routing_trace_path,
            )
        )
        for design in _candidate_designs(base, grid=grid)
    ]
    return {
        "grid": grid,
        "candidate_count": len(records),
        "records": records,
        "pareto_designs": _pareto(records),
        "pareto_objectives": (
            "minimize cycles, HBM bytes, Matrix MAC lanes, state MAC lanes, banks, FIFO, activation SRAM, and state-cache bytes; "
            "no arbitrary area-weighted score is used"
        ),
    }


def _ablation(
    model: ModelFamily,
    base: SystemDesign,
    precision: PrecisionConfig,
    *,
    context_length: int,
    decode_tokens: int,
    prefill_tokens: int,
    kimi_routing_trace_path: Path | None,
) -> dict[str, Any]:
    disabled_design = replace(
        base,
        name="all_features_disabled",
        projection_layout=ProjectionLayout.ROW_MAJOR,
        projection_direct_bypass=False,
        bc_broadcast=False,
        state_cache_bytes=0,
        state_cache_policy=StateCachePolicy.NONE,
        fused_layer_dataflow=False,
    )
    variants: list[tuple[str, SystemDesign]] = [
        ("all_features", base),
        ("all_features_disabled", disabled_design),
        (
            "without_l_compute_layout",
            replace(base, name="without_l_compute_layout", projection_layout=ProjectionLayout.ROW_MAJOR),
        ),
        (
            "without_state_cache",
            replace(
                base,
                name="without_state_cache",
                state_cache_bytes=0,
                state_cache_policy=StateCachePolicy.NONE,
            ),
        ),
        (
            "without_projection_bypass",
            replace(base, name="without_projection_bypass", projection_direct_bypass=False),
        ),
        (
            "without_fused_layer_dataflow",
            replace(base, name="without_fused_layer_dataflow", fused_layer_dataflow=False),
        ),
    ]
    if model == ModelFamily.NEMOTRON3:
        variants.append(
            (
                "without_bc_broadcast",
                replace(base, name="without_bc_broadcast", bc_broadcast=False),
            )
        )
    records = []
    for name, design in variants:
        record = _summary(
            _simulate(
                model,
                design,
                precision,
                InferencePhase.DECODE,
                context_length=context_length,
                decode_tokens=decode_tokens,
                prefill_tokens=prefill_tokens,
                kimi_routing_trace_path=kimi_routing_trace_path,
            )
        )
        record["ablation"] = name
        records.append(record)
    baseline = records[0]
    disabled = next(record for record in records if record["ablation"] == "all_features_disabled")
    for record in records:
        record["slowdown_vs_all_features"] = record["model_ready_cycles"] / baseline["model_ready_cycles"]
        record["speedup_vs_all_features_disabled"] = disabled["model_ready_cycles"] / record["model_ready_cycles"]
    prefill_enabled = _summary(
        _simulate(
            model,
            base,
            precision,
            InferencePhase.PREFILL,
            context_length=context_length,
            decode_tokens=decode_tokens,
            prefill_tokens=prefill_tokens,
            kimi_routing_trace_path=kimi_routing_trace_path,
        )
    )
    prefill_disabled = _summary(
        _simulate(
            model,
            disabled_design,
            precision,
            InferencePhase.PREFILL,
            context_length=context_length,
            decode_tokens=decode_tokens,
            prefill_tokens=prefill_tokens,
            kimi_routing_trace_path=kimi_routing_trace_path,
        )
    )

    def ab_pair(enabled: dict[str, Any], disabled_record: dict[str, Any]) -> dict[str, Any]:
        enabled_cycles = int(enabled["model_ready_cycles"])
        disabled_cycles = int(disabled_record["model_ready_cycles"])
        return {
            "all_features_cycles": enabled_cycles,
            "all_features_disabled_cycles": disabled_cycles,
            "speedup": disabled_cycles / enabled_cycles,
            "all_features_hbm_bytes": int(enabled["logical_hbm_read_bytes"]) + int(enabled["logical_hbm_write_bytes"]),
            "all_features_disabled_hbm_bytes": int(disabled_record["logical_hbm_read_bytes"])
            + int(disabled_record["logical_hbm_write_bytes"]),
        }

    prefill_ab = ab_pair(prefill_enabled, prefill_disabled)
    decode_ab = ab_pair(baseline, disabled)
    enabled_end_to_end = prefill_ab["all_features_cycles"] + decode_ab["all_features_cycles"]
    disabled_end_to_end = prefill_ab["all_features_disabled_cycles"] + decode_ab["all_features_disabled_cycles"]
    end_to_end_ab = {
        "prefill": prefill_ab,
        "decode": decode_ab,
        "prefill_plus_decode": {
            "all_features_cycles": enabled_end_to_end,
            "all_features_disabled_cycles": disabled_end_to_end,
            "speedup": disabled_end_to_end / enabled_end_to_end,
            "decode_tokens": decode_tokens,
            "prefill_tokens": prefill_tokens,
            "decode_initial_context_tokens": context_length,
            "is_continuous_request": context_length == prefill_tokens,
        },
        "scope": (
            "Same real-shape layer timeline and hardware throughput; only L-Compute layout, "
            "projection bypass, state cache, fused layer dataflow, and B/C broadcast change. "
            "The prefill-plus-decode sum is one continuous request only when the decode "
            "initial context equals the prefill token count."
        ),
    }
    compact = {
        "ablation": "without_compact_matrix_loops",
        "timing_delta": None,
        "reason": (
            "Compact loops reduce static machine code and host compilation cost; they do not remove GEMM MACs or HBM weight bytes, "
            "so this service model does not invent a cycle speedup."
        ),
        "compiler_evidence": (
            {
                "compact_instructions": 6_202_993,
                "compact_binary_mib": 23.662540435791016,
            }
            if model == ModelFamily.NEMOTRON3
            else {
                "legacy_one_head_instructions": 100_221_916,
                "compact_96_head_instructions": 11_662_716,
                "compact_binary_mib": 44.48973083496094,
            }
        ),
    }
    bandwidth = _ablation_vs_bandwidth(
        model,
        base,
        precision,
        context_length=context_length,
        decode_tokens=decode_tokens,
        prefill_tokens=prefill_tokens,
        kimi_routing_trace_path=kimi_routing_trace_path,
    )
    return {
        "cycle_ablations": records,
        "all_features_speedup_vs_disabled": disabled["model_ready_cycles"] / baseline["model_ready_cycles"],
        "end_to_end_all_on_vs_all_off": end_to_end_ab,
        "code_generation_ablation": compact,
        "ablation_vs_hbm_bandwidth": bandwidth,
    }


def _ablation_vs_bandwidth(
    model: ModelFamily,
    base: SystemDesign,
    precision: PrecisionConfig,
    *,
    context_length: int,
    decode_tokens: int,
    prefill_tokens: int,
    kimi_routing_trace_path: Path | None,
) -> dict[str, Any]:
    """Re-run the two load-bearing ablations at every candidate HBM bandwidth.

    Quoting a single slowdown per mechanism hides that the ranking inverts. At
    the 64 B/cycle candidate the memory server is 97% busy and absorbs almost
    every on-chip win, so the layout looks irrelevant and the state cache looks
    decisive; once bandwidth stops being the bottleneck the two swap places.
    A reader given only the 64 B/cycle column would draw the opposite
    conclusion about where on-chip area belongs.
    """
    rows = []
    for value in (64, 128, 256, 512, 1024, 2048, 4096, 8192):
        design = replace(base, name=f"hbm_{value}", hbm_bytes_per_cycle=value)

        def _cycles(variant: SystemDesign) -> int:
            return _simulate(
                model,
                variant,
                precision,
                InferencePhase.DECODE,
                context_length=context_length,
                decode_tokens=decode_tokens,
                prefill_tokens=prefill_tokens,
                kimi_routing_trace_path=kimi_routing_trace_path,
            ).model_ready_cycle

        report = _simulate(
            model,
            design,
            precision,
            InferencePhase.DECODE,
            context_length=context_length,
            decode_tokens=decode_tokens,
            prefill_tokens=prefill_tokens,
            kimi_routing_trace_path=kimi_routing_trace_path,
        )
        baseline = report.model_ready_cycle
        metrics = report.to_dict(include_stages=False)["metrics"]
        rows.append(
            {
                "hbm_bytes_per_cycle": value,
                "hbm_utilization": metrics["resource_utilization"]["hbm"],
                "tpot_us": metrics["tpot_us"],
                "without_l_compute_layout": _cycles(
                    replace(design, name="row", projection_layout=ProjectionLayout.ROW_MAJOR)
                )
                / baseline,
                "without_state_cache": _cycles(
                    replace(
                        design,
                        name="nosc",
                        state_cache_bytes=0,
                        state_cache_policy=StateCachePolicy.NONE,
                    )
                )
                / baseline,
            }
        )
    return {
        "status": "bandwidth_is_the_only_unswept_parameter_that_decides_most_of_decode_time",
        "rows": rows,
        "limits": [
            "The 64 B/cycle candidate is a pre-RTL placeholder, not a measured PLENA memory controller.",
            "Every row keeps compute, SRAM, cache capacity and routing fixed; only the memory server changes.",
            "These remain analytic ablation ratios, not a PLENA-versus-GPU speedup.",
        ],
    }


def _mixed_precision(
    model: ModelFamily,
    base: SystemDesign,
    *,
    context_length: int,
    decode_tokens: int,
    prefill_tokens: int,
    kimi_routing_trace_path: Path | None,
) -> dict[str, Any]:
    formal = formal_precision(model)
    candidates = {
        "formal_gpu_dtype": formal,
        "bf16_weight_activation_fp32_state": replace(
            formal,
            activation=Precision.BF16,
            weight=Precision.BF16,
            state=Precision.FP32,
        ),
        "formal_weight_bf16_state": replace(formal, state=Precision.BF16),
        "formal_weight_mx8_b128_state": replace(formal, state=Precision.MX8),
    }
    records = []
    for name, precision in candidates.items():
        design = replace(base, name=name)
        record = _summary(
            _simulate(
                model,
                design,
                precision,
                InferencePhase.DECODE,
                context_length=context_length,
                decode_tokens=decode_tokens,
                prefill_tokens=prefill_tokens,
                kimi_routing_trace_path=kimi_routing_trace_path,
            )
        )
        record["candidate"] = name
        records.append(record)
    return {
        "records": records,
        "accuracy_scope": (
            "This table is a storage/traffic sensitivity. State numerical error is reported separately; "
            "task accuracy for alternative weight formats requires real checkpoints and benchmark prompts."
        ),
    }


def precision_error_sweep(token_counts: tuple[int, ...]) -> dict[str, Any]:
    return {
        "nemotron3_mamba2": [
            run_state_precision_experiment(
                tokens=tokens,
                num_heads=4,
                head_dim=2,
                state_dim=128,
                groups=2,
                seed=17,
            )
            for tokens in token_counts
        ],
        "kimi_k3_kda": [
            run_kda_precision_experiment(
                tokens=tokens,
                num_heads=2,
                key_dim=128,
                value_dim=8,
                seed=29,
            )
            for tokens in token_counts
        ],
        "scope": (
            "Real recurrence formulas and state dimension 128 are retained; head/value parallelism is reduced because each head is independent. "
            "Weights and language-task quality are outside this CPU state-storage experiment."
        ),
    }


def _shared_device_recommendation(results: dict[str, Any]) -> dict[str, Any]:
    probes = (
        "matrix_2048",
        "matrix_8192",
        "state_lanes_4",
        "state_lanes_16",
        "banks_32",
        "fifo_128",
        "fifo_256",
        "state_cache_0m",
        "state_cache_24m",
        "state_cache_64m",
        "hbm_128",
        "hbm_256",
        "hbm_512",
        "hbm_1024",
        "hbm_2048",
        "hbm_4096",
        "hbm_8192",
    )
    sensitivity: dict[str, dict[str, float]] = {}
    for model, section in results.items():
        records = {record["design"]["name"]: record for record in section["dse"]["records"]}
        baseline = records["shared_hybrid_pre_rtl_candidate"]["model_ready_cycles"]
        sensitivity[model] = {
            name: records[name]["model_ready_cycles"] / baseline for name in probes if name in records
        }
    return {
        "status": "recommended_pre_rtl_contract_not_ppa_frozen",
        "one_device_for_both_models": True,
        "parameters": {
            "matrix_macs_per_cycle": 4096,
            "state_lanes": "8 head x 4 head-dim x 8 state-dim = 256 state MACs/cycle",
            "projection_banks": "16 single-port banks",
            "matrix_result_burst_values": 64,
            "projection_fifo_values": 64,
            "state_cache_mib": 32,
            "activation_sram_mib": 4,
            "kv_read_cache_mib": 64,
            "moe_weight_cache_mib": 256,
        },
        "normalized_decode_sensitivity": sensitivity,
        "decision": (
            "Keep the 4096-MAC Matrix, 256-MAC state path, 16 banks, and 64-value FIFO at the pre-RTL knee. "
            "The same fixed 32 MiB state cache is used for both models; capacity-aware pinned residency streams the remainder."
        ),
        "not_frozen": (
            "frequency, SRAM macro/port PPA, HBM controller efficiency, Kimi expert placement, and task-accuracy precision policy"
        ),
    }


def build_report(
    *,
    models: tuple[ModelFamily, ...] = tuple(ModelFamily),
    grid: str = "quick",
    context_length: int = 2048,
    decode_tokens: int = 4,
    prefill_tokens: int = 128,
    precision_error_tokens: tuple[int, ...] = (),
    kimi_routing_trace_path: Path | None = None,
) -> dict[str, Any]:
    if grid not in {"quick", "full"}:
        raise ValueError("grid must be quick or full")
    if min(context_length, decode_tokens, prefill_tokens) <= 0:
        raise ValueError("context, decode tokens, and prefill tokens must be positive")
    kimi_routing_coverage = (
        load_kimi_routing_trace(kimi_routing_trace_path).coverage() if kimi_routing_trace_path is not None else None
    )
    results = {}
    for model in models:
        precision = formal_precision(model)
        design = selected_design(model, precision)
        results[model.value] = {
            "persistent_state_bytes": persistent_state_bytes(model, precision),
            "formal_precision": {
                "activation": precision.activation.value,
                "weight": precision.weight.value,
                "state": precision.state.value,
                "conv_state": precision.resolved_conv_state(model).value,
            },
            "full_model": {
                "prefill": _summary(
                    _simulate(
                        model,
                        design,
                        precision,
                        InferencePhase.PREFILL,
                        context_length=context_length,
                        decode_tokens=decode_tokens,
                        prefill_tokens=prefill_tokens,
                        kimi_routing_trace_path=kimi_routing_trace_path,
                    )
                ),
                "decode": _summary(
                    _simulate(
                        model,
                        design,
                        precision,
                        InferencePhase.DECODE,
                        context_length=context_length,
                        decode_tokens=decode_tokens,
                        prefill_tokens=prefill_tokens,
                        kimi_routing_trace_path=kimi_routing_trace_path,
                    )
                ),
            },
            "dse": _dse(
                model,
                design,
                precision,
                grid=grid,
                context_length=context_length,
                decode_tokens=decode_tokens,
                prefill_tokens=prefill_tokens,
                kimi_routing_trace_path=kimi_routing_trace_path,
            ),
            "ablation": _ablation(
                model,
                design,
                precision,
                context_length=context_length,
                decode_tokens=decode_tokens,
                prefill_tokens=prefill_tokens,
                kimi_routing_trace_path=kimi_routing_trace_path,
            ),
            "mixed_precision": _mixed_precision(
                model,
                design,
                context_length=context_length,
                decode_tokens=decode_tokens,
                prefill_tokens=prefill_tokens,
                kimi_routing_trace_path=kimi_routing_trace_path,
            ),
        }
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "pre_rtl_dse_not_rtl_calibrated",
        "scenario": {
            "batch_size": 1,
            "context_length": context_length,
            "decode_tokens": decode_tokens,
            "prefill_tokens": prefill_tokens,
        },
        "models": results,
        "shared_device_recommendation": _shared_device_recommendation(results),
        "data_coverage": {
            "nemotron3": ("complete B200 NVFP4 model latency/NSYS/NCU plus exact S2048+127-step top-6 routing"),
            "kimi_k3_kda": "complete B200 single-layer KDA stage NCU and numerical official-wrapper comparison",
            "kimi_k3_full_routing": (
                kimi_routing_coverage
                if kimi_routing_coverage is not None
                else "missing; current top-16 expert IDs are deterministic sensitivity traffic"
            ),
            "additional_gpu_required_for_pre_rtl_code": False,
            "additional_gpu_required_for_paper_generalization": (
                "yes: empirical Kimi routing and real-checkpoint task accuracy for alternative weight/state precision"
            ),
        },
        "limits": [
            "Cycles are produced by explicit candidate throughput/bandwidth parameters, not measured RTL frequency or GPU-time fitting.",
            "The full logical layer sequence is executed, but complete checkpoint tensors are not numerically replayed through Rust.",
            "HBM is one burst-rounded non-preemptive server; future RTL arbitration must calibrate it.",
            "PPA and PLENA-vs-GPU speedup remain undefined until RTL synthesis and frequency calibration.",
        ],
    }
    if precision_error_tokens:
        report["state_precision_error"] = precision_error_sweep(precision_error_tokens)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", *ModelFamily], default="all")
    parser.add_argument("--grid", choices=("quick", "full"), default="quick")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--prefill-tokens", type=int, default=128)
    parser.add_argument(
        "--kimi-routing-trace",
        type=Path,
        help="validated empirical Kimi K3 prefill+decode routing JSON",
    )
    parser.add_argument(
        "--precision-error-tokens",
        type=int,
        nargs="*",
        default=(),
        help="also run CPU state-error experiments at these sequence lengths",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    models = tuple(ModelFamily) if args.model == "all" else (ModelFamily(args.model),)
    report = build_report(
        models=models,
        grid=args.grid,
        context_length=args.context_length,
        decode_tokens=args.decode_tokens,
        prefill_tokens=args.prefill_tokens,
        precision_error_tokens=tuple(args.precision_error_tokens),
        kimi_routing_trace_path=args.kimi_routing_trace,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_report",
    "formal_precision",
    "persistent_state_bytes",
    "precision_error_sweep",
    "selected_design",
]
