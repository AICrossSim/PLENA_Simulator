"""Build the calibrated-workload, pre-RTL Nemotron 3 DSE report."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from transactional_emulator.testbench.model_configs.loader import load_model_config

from .b200_formal_campaign import PINNED_SUMMARY, build_report as build_campaign_report
from .layout_mode_dse import build_report as build_layout_report
from .nemotron3_dse import (
    HardwareDesign,
    Nemotron3DseModel,
    PersistentStateCacheModel,
    ProjectionLayout,
    StateCachePolicy,
)
from .nemotron3_moe_event_dse import build_report as build_moe_event_report
from .nemotron3_routing_dse import PINNED_TRACE, build_report as build_routing_report
from .nemotron3_workload import (
    InferencePhase,
    Nemotron3WorkloadModel,
    Precision,
    ScanStrategy,
    WorkloadReport,
    WorkloadScenario,
    formal_nemotron_nvfp4_weight_policy,
)


MIB = 1024 * 1024
GIB = 1024**3
MODEL_KEY = "nemotron3_nano_30b_a3b"


def _logical_by_layer_type(workload: WorkloadReport) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for stage in workload.stages:
        if stage.layer_type not in {"mamba", "attention", "moe"}:
            continue
        totals = result.setdefault(
            stage.layer_type,
            {"read_bytes": 0, "write_bytes": 0, "weight_read_bytes": 0},
        )
        totals["read_bytes"] += stage.traffic.logical_hbm_read_bytes
        totals["write_bytes"] += stage.traffic.logical_hbm_write_bytes
        totals["weight_read_bytes"] += stage.traffic.weight_read_bytes
    return result


def _gpu_logical_crosscheck(
    workload: WorkloadReport,
    ncu_phase: dict[str, Any],
) -> dict[str, Any]:
    logical = _logical_by_layer_type(workload)
    rows = {}
    for layer_type in ("mamba", "attention", "moe"):
        observed = ncu_phase["layer_types"][layer_type]
        expected = logical[layer_type]
        rows[layer_type] = {
            "logical_read_bytes": expected["read_bytes"],
            "logical_write_bytes": expected["write_bytes"],
            "logical_weight_read_bytes": expected["weight_read_bytes"],
            "b200_physical_read_bytes": observed["dram_read_bytes"],
            "b200_physical_write_bytes": observed["dram_write_bytes"],
            "b200_duration_ns": observed["duration_ns"],
            "b200_duration_fraction": observed["duration_fraction"],
            "physical_to_logical_read_ratio": (
                observed["dram_read_bytes"] / expected["read_bytes"] if expected["read_bytes"] else None
            ),
        }
    return {
        "layer_types": rows,
        "interpretation": (
            "The ratio cross-checks workload accounting against B200 physical DRAM. "
            "It is GPU-cache/runtime specific and is not applied as a PLENA bandwidth multiplier."
        ),
    }


def _cache_capacity_sweep(arch, *, decode_tokens: int) -> dict[str, Any]:
    rows = []
    grids = {
        Precision.FP32: (0, 16 * MIB, 32 * MIB, 48 * MIB, 50_495_488, 64 * MIB),
        Precision.BF16: (0, 16 * MIB, 24 * MIB, 25_247_744, 32 * MIB),
    }
    requirements = {}
    for precision, capacities in grids.items():
        probe = PersistentStateCacheModel(arch, precision, 0, StateCachePolicy.NONE)
        required_bytes = probe.entry_bytes * len(probe.mamba_layer_ids)
        requirements[str(precision)] = {
            "entry_bytes": probe.entry_bytes,
            "entries": len(probe.mamba_layer_ids),
            "required_bytes": required_bytes,
            "required_mib": required_bytes / MIB,
            "integer_mib_to_fit": math.ceil(required_bytes / MIB),
            "status": "formal_gpu_baseline" if precision == Precision.FP32 else "numerical_candidate_only",
        }
        for capacity in capacities:
            policies = (StateCachePolicy.NONE,) if capacity == 0 else (StateCachePolicy.LRU, StateCachePolicy.PINNED)
            for policy in policies:
                stats = PersistentStateCacheModel(arch, precision, capacity, policy).simulate(
                    batch_size=1,
                    decode_tokens=decode_tokens,
                )
                rows.append(
                    {
                        "precision": precision,
                        "capacity_bytes": capacity,
                        "capacity_mib": capacity / MIB,
                        **stats.to_dict(),
                    }
                )
    return {"requirements": requirements, "records": rows}


def _designs(fp32_state_bytes: int) -> tuple[HardwareDesign, ...]:
    base = HardwareDesign(
        frequency_hz=1_000_000_000,
        matrix_macs_per_cycle=4096,
        vector_ops_per_cycle=256,
        conv_macs_per_cycle=256,
        exp_ops_per_cycle=16,
        hbm_bytes_per_cycle=64,
        projection_buffer_banks=16,
        projection_buffer_ports_per_bank=1,
        matrix_result_burst_values=64,
        projection_buffer_write_values_per_cycle=16,
        projection_fifo_values=64,
        projection_consumer_start_cycles=0,
        projection_consume_values_per_cycle=16,
        head_lanes=8,
        head_dim_lanes=4,
        state_dim_lanes=8,
    )
    return (
        replace(
            base,
            name="row_buffered_state_stream",
            projection_layout=ProjectionLayout.ROW_MAJOR,
            projection_direct_bypass=False,
            bc_broadcast=False,
        ),
        replace(
            base,
            name="skew_buffered_state_stream",
            projection_layout=ProjectionLayout.GROUP_MAJOR_SKEWED,
            projection_direct_bypass=False,
            bc_broadcast=True,
        ),
        replace(
            base,
            name="row_bypass_state_stream",
            projection_layout=ProjectionLayout.ROW_MAJOR,
            projection_direct_bypass=True,
            bc_broadcast=False,
        ),
        replace(
            base,
            name="skew_bypass_state_stream",
            projection_layout=ProjectionLayout.GROUP_MAJOR_SKEWED,
            projection_direct_bypass=True,
            bc_broadcast=True,
        ),
        replace(
            base,
            name="skew_bypass_fp32_state_resident",
            projection_layout=ProjectionLayout.GROUP_MAJOR_SKEWED,
            projection_direct_bypass=True,
            bc_broadcast=True,
            state_cache_bytes=fp32_state_bytes,
            state_cache_policy=StateCachePolicy.PINNED,
        ),
    )


def _system_dse(arch, policy, fp32_state_bytes: int) -> dict[str, Any]:
    model = Nemotron3DseModel(arch)
    decode = WorkloadScenario(
        phase=InferencePhase.DECODE,
        batch_size=1,
        sequence_length=1,
        context_length=2048,
        decode_tokens=127,
        include_embedding=False,
        include_lm_head=False,
        moe_unique_experts=6,
    )
    prefill = WorkloadScenario(
        phase=InferencePhase.PREFILL,
        batch_size=1,
        sequence_length=128,
        context_length=128,
        decode_tokens=1,
        scan_strategy=ScanStrategy.CHUNKED_AFFINE,
        include_embedding=False,
        include_lm_head=False,
        moe_unique_experts=95,
    )

    def evaluate(scenario: WorkloadScenario, design: HardwareDesign) -> dict[str, Any]:
        return model.evaluate(
            scenario,
            design,
            activation_precision=Precision.BF16,
            weight_precision=Precision.NVFP4,
            state_precision=Precision.FP32,
            weight_precision_policy=policy,
        ).to_dict(include_stages=False)

    designs = _designs(fp32_state_bytes)
    decode_results = [evaluate(decode, design) for design in designs]
    prefill_results = [evaluate(prefill, design) for design in designs[:4]]
    return {
        "hardware_assumption": {
            "frequency_hz": 1_000_000_000,
            "matrix_macs_per_cycle": 4096,
            "hbm_bytes_per_cycle": 64,
            "projection_fifo_values": 64,
            "projection_banks": 16,
            "projection_ports_per_bank": 1,
            "calibrated": False,
        },
        "decode_b1_context2048_steps127": decode_results,
        "prefill_b1_s128": prefill_results,
        "limits": [
            "The mixed checkpoint map, state dtype, layer pattern, and decode expert count are calibrated from the campaign.",
            "Compute throughput, PLENA HBM bandwidth, overlap, and frequency are assumptions awaiting RTL/FPGA calibration.",
            "The system timing still streams routed/shared expert weights every step; profile-driven cross-step weight-cache hits are reported separately.",
        ],
    }


def _guardrails(campaign: dict[str, Any], layout: dict[str, Any]) -> dict[str, Any]:
    decode = campaign["nemotron"]["ncu"]["decode_step_s2048"]
    mamba_fraction = decode["layer_types"]["mamba"]["duration_fraction"]
    local_speedup = layout["nemotron3_mamba_decode"]["comparison"]["read_write_service_speedup"]
    optimistic_system_speedup = 1 / ((1 - mamba_fraction) + mamba_fraction / local_speedup)
    kda = {}
    for case in campaign["kda"]["cases"]:
        core_fraction = case["state_core_time_fraction"]
        kda[case["case"]] = {
            "state_core_time_fraction": core_fraction,
            "infinite_state_core_speedup_upper_bound": 1 / (1 - core_fraction),
            "matrix_path_time_fraction": case["matrix_path_time_fraction"],
        }
    return {
        "nemotron_decode": {
            "b200_mamba_time_fraction": mamba_fraction,
            "local_mamba_read_write_service_speedup": local_speedup,
            "optimistic_whole_system_speedup_upper_bound": optimistic_system_speedup,
            "meaning": (
                "This unrealistically applies the local L-Compute speedup to every Mamba GPU cycle. "
                "The real system gain must be smaller; it is not a PLENA-vs-B200 speedup claim."
            ),
        },
        "kda": kda,
    }


def build_report(
    campaign_path: Path = PINNED_SUMMARY,
    routing_trace_path: Path = PINNED_TRACE,
) -> dict[str, Any]:
    model = load_model_config(MODEL_KEY)
    arch = model.arch
    campaign = build_campaign_report(campaign_path)
    policy = formal_nemotron_nvfp4_weight_policy(
        arch,
        campaign["nemotron"]["checkpoint_quantization"],
    )
    decode_scenario = WorkloadScenario(
        phase=InferencePhase.DECODE,
        context_length=2048,
        include_embedding=False,
        include_lm_head=False,
        moe_unique_experts=6,
    )
    prefill_scenario = WorkloadScenario(
        phase=InferencePhase.PREFILL,
        sequence_length=128,
        context_length=128,
        scan_strategy=ScanStrategy.CHUNKED_AFFINE,
        include_embedding=False,
        include_lm_head=False,
        moe_unique_experts=95,
    )
    workload_model = Nemotron3WorkloadModel(
        arch,
        activation_precision=Precision.BF16,
        weight_precision=Precision.NVFP4,
        state_precision=Precision.FP32,
        weight_precision_policy=policy,
    )
    decode_workload = workload_model.build(decode_scenario)
    prefill_workload = workload_model.build(prefill_scenario)
    cache = _cache_capacity_sweep(arch, decode_tokens=127)
    fp32_state_bytes = cache["requirements"]["fp32"]["required_bytes"]
    layout = build_layout_report()
    moe_event_dse = build_moe_event_report(arch, routing_trace_path)
    return {
        "schema_version": 1,
        "status": "workload_and_gpu_baseline_calibrated_plena_timing_uncalibrated",
        "model": {
            "key": MODEL_KEY,
            "architecture_config_id": model.model_id,
            "measured_checkpoint_id": campaign["nemotron"]["model"],
            "measured_checkpoint_revision": campaign["nemotron"]["revision"],
        },
        "evidence": {
            "campaign_status": campaign["campaign_status"],
            "campaign_source_status": campaign["source_status"],
            "gpu": campaign["gpu"],
            "gpu_uuids": campaign["gpu_uuids"],
            "raw_archive": campaign["source"]["archive"],
            "top_level_checksums_verified": campaign["source"]["top_level_checksums_verified"],
            "collection_artifacts_verified": campaign["source"]["collection_artifacts_verified"],
            "plena_cycle_calibrated": False,
            "rtl_ppa_calibrated": False,
        },
        "calibrated_contract": {
            "layer_counts": {"mamba": 23, "moe": 23, "attention": 6},
            "activation_precision": Precision.BF16,
            "state_precision": Precision.FP32,
            "weight_precision_policy": policy.to_dict(),
            "decode_routed_experts_per_layer": 6,
            "prefill_s128_mean_active_routed_experts_per_layer": 95,
        },
        "gpu_baseline": {
            "latency": campaign["nemotron"]["latency"],
            "ncu": campaign["nemotron"]["ncu"],
            "kda": campaign["kda"],
        },
        "gpu_logical_traffic_crosscheck": {
            "decode_step_s2048": _gpu_logical_crosscheck(
                decode_workload,
                campaign["nemotron"]["ncu"]["decode_step_s2048"],
            ),
            "prefill_s128": _gpu_logical_crosscheck(
                prefill_workload,
                campaign["nemotron"]["ncu"]["prefill_s128"],
            ),
        },
        "l_compute_layout": layout,
        "mamba_state_cache": cache,
        "moe_weight_cache": build_routing_report(arch, routing_trace_path),
        "moe_event_dse": moe_event_dse,
        "system_dse": _system_dse(arch, policy, fp32_state_bytes),
        "guardrails": _guardrails(campaign, layout),
        "remaining_before_rtl": [
            "Replace the transferred Shared-MoE Matrix/HBM constants with direct Nemotron expert, dequant, reduction, and overlap cycles from Rust/RTL/FPGA counters.",
            "Run long-context accuracy/perplexity for BF16/MX8 state before selecting a non-FP32 default.",
            "Add a physical NVFP4 decode/dequantization path; current transactional Matrix execution is not NVIDIA NVFP4.",
            "Collect representative Kimi MLA/LatentMoE or full-model GPU data before making a 93-layer Kimi system claim.",
            "Synthesize the L-Compute address generator and bank mux to measure area, frequency, and energy.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    latency = report["gpu_baseline"]["latency"]
    decode_ncu = report["gpu_baseline"]["ncu"]["decode_step_s2048"]["layer_types"]
    l_compute = report["l_compute_layout"]["nemotron3_mamba_decode"]
    state = report["mamba_state_cache"]["requirements"]
    routed = report["moe_weight_cache"]["routed_expert"]
    routed_rows = {
        row["capacity_entries"]: row for row in routed["access_orders"]["expert_id"]
    }
    moe_event_rows = [
        row
        for row in report["moe_event_dse"]["records"]
        if row["capacity_entries"] == 138 and row["shared_resident"]
    ]
    system = report["system_dse"]["decode_b1_context2048_steps127"]
    lines = [
        "# Nemotron 3 Simulator Calibration and DSE",
        "",
        "## 证据边界",
        "",
        "- B200 完整 campaign、24 个 NCU、4 个 NSYS、80 条 latency 和 3,013 个 routing event 已做哈希校验。",
        "- 已校准真实层结构、shape、FP32 Mamba state、混合 NVFP4/BF16 权重映射、routing 和 GPU baseline。",
        "- PLENA 周期与 PPA 尚未校准；下表中的 PLENA 时间只能比较候选设计，不能声称相对 B200 加速。",
        "",
        "## GPU 基线",
        "",
        "| 项目 | 实测 |",
        "|---|---:|",
        f"| B200 Decode ITL median | {latency['decode_s2048_128']['itl_median_ms']:.6f} ms |",
        f"| B200 Decode ITL P95 | {latency['decode_s2048_128']['itl_p95_ms']:.6f} ms |",
        f"| Decode Mamba NCU 时间占比 | {100 * decode_ncu['mamba']['duration_fraction']:.2f}% |",
        f"| Decode MoE NCU 时间占比 | {100 * decode_ncu['moe']['duration_fraction']:.2f}% |",
        f"| Decode Attention NCU 时间占比 | {100 * decode_ncu['attention']['duration_fraction']:.2f}% |",
        "",
        "## L-Compute",
        "",
        "| Layout | 23 层读周期 | 读写总周期 | bank stall |",
        "|---|---:|---:|---:|",
    ]
    for row in l_compute["cases"]:
        lines.append(
            f"| {row['layout']} | {row['read_service_cycles']:,} | {row['total_service_cycles']:,} | "
            f"{row['read_stall_cycles'] + row['write_stall_cycles']:,} |"
        )
    lines.extend(
        [
            "",
            f"Mamba 本地 projection SRAM 读写服务从 {l_compute['cases'][0]['total_service_cycles']:,} "
            f"降到 {l_compute['cases'][1]['total_service_cycles']:,} cycles，局部为 "
            f"{l_compute['comparison']['read_write_service_speedup']:.3f}x；这不是整层或整机加速。",
            "",
            "## Persistent State",
            "",
            "| state 格式 | 23 层准确容量 | 整数 MiB 配置 | 状态 |",
            "|---|---:|---:|---|",
            f"| FP32 | {state['fp32']['required_mib']:.5f} MiB | {state['fp32']['integer_mib_to_fit']} MiB | GPU baseline |",
            f"| BF16 | {state['bf16']['required_mib']:.5f} MiB | {state['bf16']['integer_mib_to_fit']} MiB | 仅数值候选 |",
            "",
            "48 MiB 不能容纳完整 FP32 state；准确需要 48.15625 MiB，按整数配置至少 49 MiB。",
            "",
            "## MoE Weight Cache",
            "",
            "| routed slots | 容量 | 命中率（expert-ID 调度） |",
            "|---:|---:|---:|",
        ]
    )
    for capacity in (0, 92, 137, 138, 256, 512, 1024, 2048, 2944):
        if capacity not in routed_rows:
            continue
        row = routed_rows[capacity]
        lines.append(f"| {capacity} | {row['capacity_mib']:.1f} MiB | {100 * row['hit_rate']:.2f}% |")
    lines.extend(
        [
            "",
            f"每层 shared expert 全驻留还需 {report['moe_weight_cache']['shared_expert']['all_layers_resident_mib']:.2f} MiB。",
            f"138 routed slots 加全部 shared expert 共需 {report['moe_event_dse']['architecture']['routed138_plus_shared_mib']:.2f} MiB；这是容量上界，不是 FPGA 片上 SRAM 方案。",
            "138-slot 附近存在明显容量拐点，但命中率依赖 expert 的实际执行顺序；报告同时保存 top-k rank 顺序结果。",
            "",
            "## MoE Event Scheduler",
            "",
            "| 周期模型 | 最佳 4096-PE 候选 | Expert-body 局部加速 | HBM GiB/token |",
            "|---|---|---:|---:|",
        ]
    )
    for calibration in report["moe_event_dse"]["calibrations"]:
        best = min(
            (
                row
                for row in moe_event_rows
                if row["calibration"]["name"] == calibration["name"]
            ),
            key=lambda row: row["rank"],
        )
        lines.append(
            f"| {calibration['name']} | {best['candidate']['name']} | "
            f"{best['speedup_vs_baseline']:.3f}x | {best['hbm_gib_per_decode_token']:.3f} |"
        )
    lines.extend(
        [
            "",
            "真实 cache miss 已接入有限 buffer、共享 HBM、异步 Matrix 和共享 reduction 时间线。"
            "两套周期假设差异很大，因此这些数字只能用于候选排序敏感性，不能当成 RTL speedup。",
            "",
            "## PLENA 候选（未做周期校准）",
            "",
            "| 设计 | analytic us/token | HBM GiB/token | state hit | bank stall/token |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in system:
        metrics = item["metrics"]
        lines.append(
            f"| {item['design']['name']} | {metrics['latency_us_per_step']:.2f} | "
            f"{metrics['hbm_bytes_per_step'] / GIB:.3f} | {100 * metrics['state_cache_hit_rate']:.1f}% | "
            f"{metrics['bank_stall_cycles_per_step']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "1. L-Compute 的 bank-conflict 改善已由真实 packet、物理 roundtrip 和 service cycle 三层证明。",
            "2. 完整 Nemotron 的主系统瓶颈仍是 MoE/权重流量；只加 state engine 或斜存不能带来数量级整机加速。",
            "3. FP32 Mamba state 全驻留容量应按 48.15625 MiB 设计，不能写成 48 MiB。",
            "4. 真实 routing miss 已进入 Expert/M/K 事件调度；B1 下 M-split 退化，K-split 暂为两套敏感性中的共同第一名。",
            "5. 最终 speedup 必须等 RTL/FPGA 校准 Matrix、Vector、HBM、state engine 与 L-Compute 后再报告。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=PINNED_SUMMARY)
    parser.add_argument("--routing-trace", type=Path, default=PINNED_TRACE)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args(argv)
    report = build_report(args.campaign, args.routing_trace)
    rendered = json.dumps(report, indent=2) + "\n"
    markdown = render_markdown(report)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown)
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
