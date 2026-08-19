"""Staged projection-path and full-system sensitivity sweep for Nemotron 3."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

from transactional_emulator.testbench.model_configs.loader import (
    ModelArchConfig,
    load_model_config,
)

from .nemotron3_dse import (
    HardwareDesign,
    Nemotron3DseModel,
    PersistentStateCacheModel,
    ProjectionBankModel,
    ProjectionLayout,
    ProjectionWriteBufferModel,
    StateCachePolicy,
)
from .nemotron3_workload import (
    InferencePhase,
    Precision,
    ScanStrategy,
    WorkloadScenario,
)


MIB = 1024 * 1024
MODEL_KEY = "nemotron3_nano_30b_a3b"


def _projection_grid(quick: bool) -> dict[str, tuple[Any, ...]]:
    if quick:
        return {
            "layouts": (ProjectionLayout.ROW_MAJOR, ProjectionLayout.GROUP_MAJOR_SKEWED),
            "broadcasts": (False, True),
            "banks": (8, 16),
            "ports": (1,),
            "fifo_values": (64, 256),
            "bypasses": (False, True),
            "consumer_delays": (0, 10_000),
            "consumer_widths": (8, 16),
        }
    return {
        "layouts": (ProjectionLayout.ROW_MAJOR, ProjectionLayout.GROUP_MAJOR_SKEWED),
        "broadcasts": (False, True),
        "banks": (8, 16, 32),
        "ports": (1, 2),
        "fifo_values": (64, 128, 256, 512),
        "bypasses": (False, True),
        "consumer_delays": (0, 1_000, 10_000, 32_768, 100_000),
        "consumer_widths": (8, 16, 32),
    }


def _system_grid(quick: bool) -> dict[str, tuple[Any, ...]]:
    if quick:
        return {
            "precisions": (Precision.BF16,),
            "banks": (8, 16),
            "state_dim_lanes": (8,),
            "cache_mib": (0, 32),
            "resident_policies": (StateCachePolicy.PINNED,),
        }
    return {
        "precisions": (Precision.BF16, Precision.FP32),
        "banks": (8, 16, 32),
        "state_dim_lanes": (4, 8, 16),
        "cache_mib": (0, 16, 24, 32, 48, 64),
        "resident_policies": (StateCachePolicy.LRU, StateCachePolicy.PINNED),
    }


def sweep_projection_path(arch: ModelArchConfig, *, quick: bool = False) -> list[dict[str, Any]]:
    mamba = arch.mamba
    if mamba is None:
        raise ValueError("projection sensitivity requires a Mamba architecture")
    grid = _projection_grid(quick)
    producer_cycles = math.ceil(
        arch.hidden_size * mamba.projection_size / HardwareDesign().matrix_macs_per_cycle
    )
    records = []
    bank_cache = {}
    for layout, broadcast, banks, ports, fifo, bypass, delay, consumer_width in product(
        grid["layouts"],
        grid["broadcasts"],
        grid["banks"],
        grid["ports"],
        grid["fifo_values"],
        grid["bypasses"],
        grid["consumer_delays"],
        grid["consumer_widths"],
    ):
        design = HardwareDesign(
            name=(
                f"{layout}-bc{int(broadcast)}-bypass{int(bypass)}-b{banks}p{ports}"
                f"-fifo{fifo}-delay{delay}-consume{consumer_width}"
            ),
            projection_buffer_banks=banks,
            projection_buffer_ports_per_bank=ports,
            projection_fifo_values=fifo,
            projection_direct_bypass=bypass,
            projection_consumer_start_cycles=delay,
            projection_consume_values_per_cycle=consumer_width,
            projection_layout=layout,
            bc_broadcast=broadcast,
        )
        key = (layout, broadcast, banks, ports)
        bank_stats = bank_cache.get(key)
        if bank_stats is None:
            bank_stats = ProjectionBankModel(arch, design).simulate_one_token_request_layer()
            bank_cache[key] = bank_stats
        write = ProjectionWriteBufferModel(design).simulate(
            values=mamba.projection_size,
            producer_cycles=producer_cycles,
            values_per_token=mamba.projection_size,
            forced_spill_values_per_token=mamba.d_inner,
            activation_bytes=2,
        )
        state_values = mamba.projection_size - mamba.d_inner
        spilled_state = max(0, write.spill_values - mamba.d_inner)
        buffered_fraction = spilled_state / state_values
        actual_state = bank_stats.state_input.scaled_fraction(buffered_fraction)
        actual_gate = bank_stats.gate
        actual_bc_reads = round(bank_stats.bc_value_reads * buffered_fraction)
        read_service_cycles = actual_state.service_cycles + actual_gate.service_cycles
        read_stall_cycles = actual_state.stall_cycles + actual_gate.stall_cycles
        records.append(
            {
                "design": {
                    "name": design.name,
                    "layout": layout,
                    "bc_broadcast": broadcast,
                    "banks": banks,
                    "ports_per_bank": ports,
                    "fifo_values": fifo,
                    "direct_bypass": bypass,
                    "consumer_start_cycles": delay,
                    "consumer_values_per_cycle": consumer_width,
                },
                "metrics": {
                    "producer_cycles": producer_cycles,
                    "completion_cycles": write.completion_cycles,
                    "estimated_path_cycles": max(write.completion_cycles, read_service_cycles),
                    "direct_values": write.direct_values,
                    "spill_values": write.spill_values,
                    "spill_bytes": write.spill_bytes,
                    "state_buffered_fraction": buffered_fraction,
                    "fifo_stall_cycles": write.fifo_stall_cycles,
                    "fifo_high_watermark": write.fifo_high_watermark,
                    "buffer_read_service_cycles": read_service_cycles,
                    "buffer_read_stall_cycles": read_stall_cycles,
                    "state_read_stall_cycles": actual_state.stall_cycles,
                    "gate_read_stall_cycles": actual_gate.stall_cycles,
                    "bc_value_reads": actual_bc_reads,
                    "resource_bank_ports": banks * ports,
                },
            }
        )
    return records


def _cache_combinations(grid: dict[str, tuple[Any, ...]]) -> list[tuple[int, StateCachePolicy]]:
    combinations = [(0, StateCachePolicy.NONE)]
    for cache_mib in grid["cache_mib"]:
        if cache_mib == 0:
            continue
        combinations.extend((cache_mib, policy) for policy in grid["resident_policies"])
    return combinations


def sweep_decode_system(arch: ModelArchConfig, *, quick: bool = False) -> list[dict[str, Any]]:
    grid = _system_grid(quick)
    model = Nemotron3DseModel(arch)
    scenario = WorkloadScenario(
        phase=InferencePhase.DECODE,
        batch_size=1,
        sequence_length=1,
        context_length=2048,
        decode_tokens=4,
        include_embedding=False,
        include_lm_head=False,
    )
    records = []
    for precision, banks, state_lanes, (cache_mib, policy) in product(
        grid["precisions"],
        grid["banks"],
        grid["state_dim_lanes"],
        _cache_combinations(grid),
    ):
        design = HardwareDesign(
            name=(
                f"decode-{precision}-b{banks}-n{state_lanes}"
                f"-cache{cache_mib}-{policy}"
            ),
            projection_buffer_banks=banks,
            projection_buffer_ports_per_bank=1,
            projection_fifo_values=64,
            projection_direct_bypass=True,
            projection_consumer_start_cycles=0,
            projection_consume_values_per_cycle=16,
            projection_layout=ProjectionLayout.GROUP_MAJOR_SKEWED,
            bc_broadcast=True,
            state_dim_lanes=state_lanes,
            state_cache_bytes=cache_mib * MIB,
            state_cache_policy=policy,
        )
        result = model.evaluate(
            scenario,
            design,
            activation_precision=Precision.BF16,
            weight_precision=Precision.BF16,
            state_precision=precision,
        )
        rendered = result.to_dict(include_stages=False)
        records.append(
            {
                "design": rendered["design"],
                "metrics": rendered["metrics"],
                "state_cache": rendered["state_cache"],
            }
        )
    return records


def sweep_prefill_candidates(arch: ModelArchConfig, *, quick: bool = False) -> list[dict[str, Any]]:
    model = Nemotron3DseModel(arch)
    scenario = WorkloadScenario(
        phase=InferencePhase.PREFILL,
        batch_size=1,
        sequence_length=128,
        context_length=128,
        decode_tokens=1,
        scan_strategy=ScanStrategy.CHUNKED_AFFINE,
        include_embedding=False,
        include_lm_head=False,
    )
    banks_values = (8,) if quick else (8, 16)
    lane_values = (8,) if quick else (8, 16)
    records = []
    for banks, state_lanes, broadcast, bypass in product(
        banks_values,
        lane_values,
        (False, True),
        (False, True),
    ):
        design = HardwareDesign(
            name=f"prefill-b{banks}-n{state_lanes}-bc{int(broadcast)}-bypass{int(bypass)}",
            projection_buffer_banks=banks,
            projection_fifo_values=64,
            projection_direct_bypass=bypass,
            projection_layout=ProjectionLayout.GROUP_MAJOR_SKEWED,
            bc_broadcast=broadcast,
            state_dim_lanes=state_lanes,
        )
        result = model.evaluate(
            scenario,
            design,
            activation_precision=Precision.BF16,
            weight_precision=Precision.BF16,
            state_precision=Precision.BF16,
        )
        rendered = result.to_dict(include_stages=False)
        records.append(
            {
                "design": rendered["design"],
                "metrics": rendered["metrics"],
            }
        )
    return records


def _projection_best_by_delay(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    delays = sorted({record["design"]["consumer_start_cycles"] for record in records})
    for delay in delays:
        candidates = [record for record in records if record["design"]["consumer_start_cycles"] == delay]
        candidates.sort(
            key=lambda record: (
                record["metrics"]["estimated_path_cycles"],
                record["metrics"]["fifo_stall_cycles"],
                record["metrics"]["buffer_read_stall_cycles"],
                record["metrics"]["spill_bytes"],
                record["metrics"]["resource_bank_ports"],
                record["design"]["fifo_values"],
                record["design"]["consumer_values_per_cycle"],
            )
        )
        result.append(candidates[0])
    return result


def _cache_thresholds(
    arch: ModelArchConfig,
    records: list[dict[str, Any]],
    *,
    batch_size: int = 1,
) -> dict[str, dict[str, int | None]]:
    """Report the real full-residency requirement alongside the swept grid point.

    The smallest swept size that reaches a 100% hit rate is an artifact of
    ``cache_mib``: BF16 needs 24.08 MiB but the grid jumps 24 -> 32, so reporting
    the grid point alone overstates the SRAM requirement by a third. Size
    decisions must read ``required_mib``; ``smallest_swept_full_mib`` stays only
    to locate the record inside this report.
    """
    thresholds: dict[str, dict[str, int | None]] = {}
    for precision in (Precision.BF16, Precision.FP32):
        cache_model = PersistentStateCacheModel(
            arch, precision, 0, StateCachePolicy.NONE
        )
        required_bytes = (
            cache_model.entry_bytes * len(cache_model.mamba_layer_ids) * batch_size
        )
        swept = [
            record
            for record in records
            if record["design"]["state_cache_policy"] == StateCachePolicy.PINNED
            and record["design"]["name"].startswith(f"decode-{precision}-")
            and record["state_cache"]["hit_rate"] == 1.0
        ]
        thresholds[str(precision)] = {
            "entry_bytes": cache_model.entry_bytes,
            "resident_entries": len(cache_model.mamba_layer_ids) * batch_size,
            "required_bytes": required_bytes,
            "required_mib": math.ceil(required_bytes / MIB),
            "smallest_swept_full_mib": min(
                (record["design"]["state_cache_bytes"] // MIB for record in swept),
                default=None,
            ),
        }
    return thresholds


def build_report(*, quick: bool = False) -> dict[str, Any]:
    model_config = load_model_config(MODEL_KEY)
    arch = model_config.arch
    projection = sweep_projection_path(arch, quick=quick)
    decode = sweep_decode_system(arch, quick=quick)
    prefill = sweep_prefill_candidates(arch, quick=quick)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "model_key": MODEL_KEY,
        "model_id": model_config.model_id,
        "mode": "quick" if quick else "full",
        "status": "uncalibrated_sensitivity_not_rtl_performance",
        "compiler_scope": (
            "X_STATE and L_SCATTER_M descriptors are executable; general Matrix/Vector "
            "events and full-checkpoint lowering remain incomplete."
        ),
        "assumptions": [
            "Projection micro-sweep models one real Nemotron Mamba layer/token with BF16 activations.",
            "Matrix projection takes ceil(2688*10304/4096)=6762 ideal MAC cycles before RTL calibration.",
            "Gate is always materialized for Vector; x/B/C/dt may bypass or spill.",
            "Decode full-system sweep uses four generated tokens and excludes embedding/LM head.",
            "Prefill candidate sweep uses B1/S128 chunked-affine work accounting.",
            "GPU data validates workload shape and stage dominance, not PLENA cycle calibration.",
        ],
        "projection_path": {
            "design_count": len(projection),
            "best_by_consumer_delay": _projection_best_by_delay(projection),
            "records": projection,
        },
        "decode_system": {
            "design_count": len(decode),
            "full_cache_threshold": _cache_thresholds(arch, decode),
            "records": decode,
        },
        "prefill_candidates": {
            "design_count": len(prefill),
            "records": prefill,
        },
    }


def _best_decode_by_precision(report: dict[str, Any], precision: Precision) -> dict[str, Any]:
    records = [
        record
        for record in report["decode_system"]["records"]
        if record["design"]["name"].startswith(f"decode-{precision}-")
    ]
    return min(records, key=lambda record: record["metrics"]["total_cycles"])


def _bf16_decode_decision_points(report: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    records = [
        record
        for record in report["decode_system"]["records"]
        if record["design"]["name"].startswith(f"decode-{Precision.BF16}-")
    ]
    fastest = min(records, key=lambda record: record["metrics"]["total_cycles"])
    pinned = [
        record
        for record in records
        if record["design"]["state_cache_policy"] == StateCachePolicy.PINNED
    ]
    capacity_knee = min(
        (record for record in pinned if record["state_cache"]["hit_rate"] >= 0.95),
        key=lambda record: (
            record["design"]["state_cache_bytes"],
            record["design"]["projection_buffer_banks"],
            record["design"]["state_macs_per_cycle"],
        ),
    )
    minimum_full_resident = min(
        (record for record in pinned if record["state_cache"]["hit_rate"] == 1.0),
        key=lambda record: (
            record["design"]["projection_buffer_banks"],
            record["design"]["state_macs_per_cycle"],
            record["design"]["state_cache_bytes"],
        ),
    )
    no_cache = min(
        (record for record in records if record["design"]["state_cache_bytes"] == 0),
        key=lambda record: (
            record["design"]["projection_buffer_banks"],
            record["design"]["state_macs_per_cycle"],
        ),
    )
    return [
        ("No cache baseline", no_cache),
        (">=95% pinned capacity knee", capacity_knee),
        ("Minimum sampled full-resident", minimum_full_resident),
        ("Fastest sampled", fastest),
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Nemotron 3 Sensitivity Sweep",
        "",
        f"生成时间：`{report['generated_at']}`。结果状态：`{report['status']}`。",
        "",
        "## 易懂结论",
        "",
    ]
    projection_records = report["projection_path"]["records"]
    minimum_fifo = min(
        record["design"]["fifo_values"]
        for record in projection_records
        if record["metrics"]["fifo_stall_cycles"] == 0
    )
    maximum_high_watermark = max(
        record["metrics"]["fifo_high_watermark"] for record in projection_records
    )
    thresholds = report["decode_system"]["full_cache_threshold"]
    decision_points = _bf16_decode_decision_points(report)
    fastest = decision_points[-1][1]
    minimum_full_resident = decision_points[-2][1]
    fastest_gain_us = (
        minimum_full_resident["metrics"]["latency_us_per_step"]
        - fastest["metrics"]["latency_us_per_step"]
    )
    fastest_gain_percent = (
        fastest_gain_us / minimum_full_resident["metrics"]["latency_us_per_step"] * 100
    )
    lines.extend(
        [
            f"- 本轮最小无停顿 FIFO 是 **{minimum_fifo} values**；全 sweep 最大 high-watermark 是 **{maximum_high_watermark}**。",
            "- consumer 未准备好时 bypass 会自动退化为 spill；skewed layout 仍可消除读取 bank conflict。",
            f"- BF16 state 全驻留需要 **{thresholds['bf16']['required_mib']} MiB**"
            f"（{thresholds['bf16']['resident_entries']} 个 entry × "
            f"{thresholds['bf16']['entry_bytes']} B；本轮网格最小全命中采样点是 "
            f"{thresholds['bf16']['smallest_swept_full_mib']} MiB）。",
            f"- FP32 state 全驻留需要 **{thresholds['fp32']['required_mib']} MiB**"
            f"（本轮网格最小全命中采样点是 "
            f"{thresholds['fp32']['smallest_swept_full_mib']} MiB）。",
            f"- 最大采样点只比最小全驻留采样点快 **{fastest_gain_us:.2f} us/token ({fastest_gain_percent:.3f}%)**；当前不能用这点收益证明 32 banks/16 state lanes 合理。",
            "- 这些是敏感度和临界点，不是 RTL 校准后的绝对 latency。",
            "",
            "## Projection 最佳点（按 consumer delay）",
            "",
            "| Delay cycles | Layout | Banks x ports | FIFO | Bypass | Spill B | Read stall | Path cycles |",
            "|---:|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for record in report["projection_path"]["best_by_consumer_delay"]:
        design = record["design"]
        metrics = record["metrics"]
        lines.append(
            f"| {design['consumer_start_cycles']} | {design['layout']} | "
            f"{design['banks']}x{design['ports_per_bank']} | {design['fifo_values']} | "
            f"{design['direct_bypass']} | {metrics['spill_bytes']} | "
            f"{metrics['buffer_read_stall_cycles']} | {metrics['estimated_path_cycles']} |"
        )
    lines.extend(
        [
            "",
            "## Decode 全模型候选",
            "",
            "| State precision | Design | us/token | HBM MiB/token | Cache hit | Spill MiB/token |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    available_precisions = [
        precision
        for precision in (Precision.BF16, Precision.FP32)
        if any(
            record["design"]["name"].startswith(f"decode-{precision}-")
            for record in report["decode_system"]["records"]
        )
    ]
    for precision in available_precisions:
        record = _best_decode_by_precision(report, precision)
        design = record["design"]
        metrics = record["metrics"]
        lines.append(
            f"| {precision} | {design['name']} | {metrics['latency_us_per_step']:.2f} | "
            f"{metrics['hbm_bytes_per_step'] / MIB:.2f} | {record['state_cache']['hit_rate']:.3f} | "
            f"{metrics['projection_spill_bytes'] / 4 / MIB:.3f} |"
        )
    lines.extend(
        [
            "",
            "## BF16 Decode 决策点",
            "",
            "| Point | Design | us/token | HBM MiB/token | Cache hit |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for label, record in decision_points:
        lines.append(
            f"| {label} | {record['design']['name']} | "
            f"{record['metrics']['latency_us_per_step']:.2f} | "
            f"{record['metrics']['hbm_bytes_per_step'] / MIB:.2f} | "
            f"{record['state_cache']['hit_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 对 Compiler/Simulator 的直接要求",
            "",
            "- Compiler scheduler 已能发出 `PRELOAD/COMMIT/EVICT` 并固定前 N 个 layer；下一步要把 DSE 容量自动转换成 layer residency map，替代手填 cache entry 数。",
            "- 第一版物理参数应保留 8 banks 和较小 state-lane 配置作为基线；更多 banks/lanes 只有在 PPA 和并发 workload 证明后才能增加。",
            "- `PROJECTION_SCATTER` 的 64-value FIFO 是当前模型的下界，不是 RTL 保证；后续要用 cycle-accurate Matrix producer 校准 burst arrival。",
            "- KDA 当前共享 `X_STATE` 生命周期并保留 row-major fallback；RTX 5090 profile 已校准 state dtype/容量和 mixer latency，但专用 bank/layout 仍需分 stage traffic 后单独 DSE。",
            "",
            "## 不能从本轮得出的结论",
            "",
            "- 不能据此声称 FPGA/ASIC 的绝对加速比或 PPA。",
            "- 不能据此冻结 KDA skewed layout；现有 KDA profile 缺少 NSYS stage breakdown 和 NCU DRAM counters。",
            "- 不能声称 Compiler 已能从完整 Nemotron/Kimi checkpoint 生成最终二进制。",
            "- bypass 降低 SRAM traffic 不代表端到端 latency 一定下降；projection weight/compute 仍可能主导。",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("artifacts/dse/nemotron3_sensitivity_round1.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("artifacts/dse/nemotron3_sensitivity_round1.md"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = build_report(quick=args.quick)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    args.markdown_out.write_text(render_markdown(report))
    print(render_markdown(report))
    print(f"JSON report: {args.json_out}")
    print(f"Markdown report: {args.markdown_out}")


if __name__ == "__main__":
    main()
