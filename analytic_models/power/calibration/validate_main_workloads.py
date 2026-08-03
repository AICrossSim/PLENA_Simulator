"""Generate compact main-compiler action-mix coverage evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from compiler.aten.cost_frontend import (
    CompilerHardwareSpec,
    DecoderModelSpec,
    RoutingHistogram,
    compile_dense_decoder_trace,
    compile_routed_moe_trace,
)
from compiler.aten.program_sink import COST_TRACE_GRANULARITY_SUMMARY

from analytic_models.power import ActionHardwareConfig, estimate_action_energy
from analytic_models.power.calibration import DEFAULT_LOGIC_ENERGY, DEFAULT_POWER_VALIDATION


def _load(path: Path):
    return json.loads(path.read_text())


def _gpt_oss(path: Path) -> DecoderModelSpec:
    value = _load(path)
    return DecoderModelSpec(
        hidden_size=value["hidden_size"],
        intermediate_size=value["intermediate_size"],
        num_attention_heads=value["num_attention_heads"],
        num_key_value_heads=value["num_key_value_heads"],
        head_dim=value["head_dim"],
        num_hidden_layers=value["num_hidden_layers"],
        model_type=value["model_type"],
        rms_norm_eps=value["rms_norm_eps"],
        num_experts=value["num_local_experts"],
        experts_per_token=value["num_experts_per_tok"],
        moe_intermediate_size=value["intermediate_size"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen32", type=Path, required=True)
    parser.add_argument("--qwen235", type=Path, required=True)
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--gpt-oss", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    coefficients = _load(DEFAULT_LOGIC_ENERGY)
    inherited = _load(DEFAULT_POWER_VALIDATION)
    compiler_hardware = CompilerHardwareSpec(mlen=512, blen=64, mram_tile_capacity=4)
    action_hardware = ActionHardwareConfig(mlen=512, blen=64, vlen=512)
    qwen235 = DecoderModelSpec.load(args.qwen235)
    gpt_oss = _gpt_oss(args.gpt_oss)

    scenarios = [
        ("qwen3-32b-482x16", lambda: compile_dense_decoder_trace(args.qwen32, compiler_hardware, seq_len=482, batch_size=16, cost_trace_granularity=COST_TRACE_GRANULARITY_SUMMARY)),
        ("qwen3-32b-32768x1", lambda: compile_dense_decoder_trace(args.qwen32, compiler_hardware, seq_len=32768, batch_size=1, cost_trace_granularity=COST_TRACE_GRANULARITY_SUMMARY)),
        ("llama-3.1-8b-4096x1", lambda: compile_dense_decoder_trace(args.llama, compiler_hardware, seq_len=4096, batch_size=1, cost_trace_granularity=COST_TRACE_GRANULARITY_SUMMARY)),
        ("qwen3-235b-balanced", lambda: compile_routed_moe_trace(qwen235, compiler_hardware, RoutingHistogram.balanced(token_count=482 * 16, top_k=8, num_experts=128), cost_trace_granularity=COST_TRACE_GRANULARITY_SUMMARY)),
        ("qwen3-235b-skewed", lambda: compile_routed_moe_trace(qwen235, compiler_hardware, RoutingHistogram.skewed(token_count=482 * 16, top_k=8, num_experts=128), cost_trace_granularity=COST_TRACE_GRANULARITY_SUMMARY)),
        ("gpt-oss-20b-balanced", lambda: compile_routed_moe_trace(gpt_oss, compiler_hardware, RoutingHistogram.balanced(token_count=4096, top_k=4, num_experts=32), cost_trace_granularity=COST_TRACE_GRANULARITY_SUMMARY)),
        ("gpt-oss-20b-skewed", lambda: compile_routed_moe_trace(gpt_oss, compiler_hardware, RoutingHistogram.skewed(token_count=4096, top_k=4, num_experts=32), cost_trace_granularity=COST_TRACE_GRANULARITY_SUMMARY)),
    ]
    results = []
    family_union: set[str] = set()
    for name, compile_scenario in scenarios:
        started = time.perf_counter()
        compiled = compile_scenario()
        report = estimate_action_energy(compiled.trace, action_hardware, coefficients)
        families = sorted(
            {f"{item.component}.{item.action}" for item in report.actions if not item.component.endswith("_sram")}
        )
        family_union.update(families)
        results.append(
            {
                "scenario": name,
                "elapsed_seconds": time.perf_counter() - started,
                "dynamic_instructions": sum(compiled.trace.dynamic_opcode_counts.values()),
                "unique_trace_instructions": len(compiled.trace.instructions),
                "dynamic_opcodes": dict(sorted(compiled.trace.dynamic_opcode_counts.items())),
                "action_families": families,
                "opcode_coverage": report.opcode_coverage,
                "active_shape_coverage": report.active_shape_coverage,
                "sram_descriptor_coverage": report.sram_descriptor_coverage,
                "nominal_logic_energy_pj": report.nominal_energy_pj,
                "trace_isa_hash": compiled.trace.isa_hash,
                "route_objects": compiled.trace.metadata.get("route_object_count"),
            }
        )
    output = {
        "schema_version": "plena-main-power-workload-coverage-v1",
        "logic_calibration": coefficients["calibration_status"],
        "inherited_holdout_median_ape": inherited["holdout_median_ape"],
        "inherited_holdout_p95_ape": inherited["holdout_p95_ape"],
        "scenario_count": len(results),
        "action_family_union": sorted(family_union),
        "all_opcode_coverage_complete": all(item["opcode_coverage"] == 1.0 for item in results),
        "scenarios": results,
        "interpretation": (
            "These scenarios validate main compiler action-family coverage and compression. "
            "Numerical energy error remains the independent mapped-DC holdout metric."
        ),
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
