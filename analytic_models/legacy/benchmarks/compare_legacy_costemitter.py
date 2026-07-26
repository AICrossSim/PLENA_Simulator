"""Compare the historical Qwen analytic model with a CostEmitter trace.

This utility separates two effects that are otherwise easy to conflate:

1. The compiler emits more work than the closed-form legacy model represents.
2. Vector and scalar operations take more than one cycle once full-machine
   latency, dependencies, and resource hazards are respected.

The ideal counterfactual keeps the real compiler opcode counts and the current
MatrixMachine resource latency, but assigns one cycle to every VectorMachine,
ScalarMachine, and control instruction. It is therefore an architectural lower
bound for the existing instruction stream, not a prediction of current RTL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from analytic_models.performance.compiler_cost_model import TransactionalCycleModel
from analytic_models.performance.perf_model import load_hardware_config_from_toml


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _unwrap_report(value: dict[str, Any]) -> dict[str, Any]:
    report = value.get("report", value)
    if not isinstance(report, dict):
        raise ValueError("CostEmitter report must be an object")
    return report


def _prefix_count(counts: dict[str, int], prefix: str) -> int:
    return sum(int(count) for opcode, count in counts.items() if opcode.startswith(prefix))


def _category(opcode: str) -> str:
    if opcode.startswith("M_"):
        return "matrix"
    if opcode.startswith("V_"):
        return "vector"
    if opcode.startswith("S_"):
        return "scalar"
    return "control"


def _matrix_cycles_per_opcode(
    report: dict[str, Any],
    counts: dict[str, int],
) -> dict[str, float]:
    work = report["compute_opcode_work_cycles"]
    result: dict[str, float] = {}
    for opcode, count in counts.items():
        if opcode.startswith("M_") and count:
            result[opcode] = float(work[opcode]) / int(count)
    return result


def _matrix_work(
    counts: dict[str, int],
    cycles_per_opcode: dict[str, float],
) -> float:
    return sum(
        int(count) * cycles_per_opcode[opcode]
        for opcode, count in counts.items()
        if opcode.startswith("M_")
    )


def _same_trace_ideal_one_cycle(
    report: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the real trace with ideal one-cycle Vector/Scalar/control."""

    full_counts = trace["dynamic_opcodes"]
    matrix_cycles = _matrix_cycles_per_opcode(report, full_counts)
    categories = {
        "matrix": _matrix_work(full_counts, matrix_cycles),
        "vector": float(_prefix_count(full_counts, "V_")),
        "scalar": float(_prefix_count(full_counts, "S_")),
        "control": float(_prefix_count(full_counts, "C_")),
    }

    stage_results: dict[str, dict[str, float]] = {}
    roofline_ns = 0.0
    serial_ns = 0.0
    for stage, stage_trace in trace["stage_breakdown"].items():
        counts = stage_trace["dynamic_opcodes"]
        compute_cycles = (
            _matrix_work(counts, matrix_cycles)
            + _prefix_count(counts, "V_")
            + _prefix_count(counts, "S_")
            + _prefix_count(counts, "C_")
        )
        memory_ns = float(report["hbm_stage_latency_ns"][stage])
        stage_roofline_ns = max(compute_cycles, memory_ns)
        stage_results[stage] = {
            "compute_cycles": compute_cycles,
            "memory_ns": memory_ns,
            "roofline_ns": stage_roofline_ns,
        }
        roofline_ns += stage_roofline_ns
        serial_ns += compute_cycles + memory_ns

    return {
        "semantics": (
            "real compiler opcode counts; current matrix resource cycles; "
            "one cycle per vector, scalar, and control instruction; no "
            "inter-instruction hazard penalty"
        ),
        "compute_cycles": sum(categories.values()),
        "stage_roofline_ns": roofline_ns,
        "serial_compute_plus_memory_ns": serial_ns,
        "category_cycles": categories,
        "stage_breakdown": stage_results,
        "matrix_cycles_per_opcode": matrix_cycles,
    }


def _same_trace_legacy_opcode_timing(
    settings_path: Path,
    trace: dict[str, Any],
) -> dict[str, Any]:
    """Apply the historical transactional latency constants to the real trace."""

    timing = TransactionalCycleModel.load(settings_path)
    categories = {"matrix": 0, "vector": 0, "scalar": 0, "control": 0}
    for opcode, count in trace["dynamic_opcodes"].items():
        if opcode.startswith("H_"):
            continue
        categories[_category(opcode)] += int(count) * int(timing.instruction_cycles(opcode))
    return {
        "semantics": "real compiler opcode counts with legacy per-opcode latency constants",
        "compute_cycles": sum(categories.values()),
        "category_cycles": categories,
    }


def _legacy_qwen_model(
    model_config_path: Path,
    settings_path: Path,
    isa_path: Path,
    *,
    mlen: int,
    blen: int,
    vlen: int,
    hlen: int,
    seq_len: int,
    batch_size: int,
    frequency_hz: float,
) -> dict[str, Any]:
    """Run the historical Qwen3 closed-form model with controlled dimensions."""

    # qwen3_model.py and llama_model.py were originally executable scripts and
    # intentionally use sibling imports. Preserve that historical environment
    # rather than changing their behavior for this diagnostic.
    performance_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(performance_dir))
    try:
        from qwen3_model import Qwen3DenseModel
    finally:
        sys.path.pop(0)

    hardware = load_hardware_config_from_toml(str(settings_path)).model_copy(
        update={
            "MLEN": mlen,
            "BLEN": blen,
            "VLEN": vlen,
            "HLEN": hlen,
            "HBM_M_Prefetch_Amount": mlen,
            "HBM_V_Prefetch_Amount": blen,
            "HBM_V_Writeback_Amount": blen,
        }
    )
    model = Qwen3DenseModel(
        str(model_config_path),
        hardware,
        str(isa_path),
        batch_size=batch_size,
        input_seq_len=seq_len,
        output_seq_len=1,
        device_num=1,
        frequency_hz=frequency_hz,
    )

    perf = model.perf
    mode = "prefill"
    rms = perf.rms_layer(model.hidden_size, seq_len, batch_size, mode)
    projection = perf.projection(
        model.hidden_size,
        model.num_attention_heads,
        model.num_key_value_heads,
        model.head_dim,
        seq_len,
        batch_size,
        mode,
    )
    qk_norm = model._qk_norm_cycles(seq_len, batch_size, mode)
    attention = perf.flash_attention(
        model.num_attention_heads,
        model.num_key_value_heads,
        model.head_dim,
        seq_len,
        seq_len,
        batch_size,
        mode,
    )
    residual = perf.residual(model.hidden_size, seq_len, batch_size, mode)
    ffn = perf.feed_forward(model.hidden_size, model.intermediate_size, seq_len, batch_size, mode)
    embedding = perf.embeddings(model.hidden_size, seq_len, batch_size, mode)
    layer_cycles = 2 * rms + projection + qk_norm + attention + residual + ffn
    full_cycles = embedding + model.num_hidden_layers * layer_cycles

    return {
        "semantics": "historical closed-form Qwen3 dense analytic model",
        "frequency_hz": frequency_hz,
        "instruction_latency_cycles": dict(perf.instr.items()),
        "embedding_cycles": embedding,
        "one_layer_cycles": layer_cycles,
        "full_model_cycles": full_cycles,
        "full_model_latency_s": full_cycles / frequency_hz,
        "one_layer_breakdown_cycles": {
            "rms_norm_each": rms,
            "projection": projection,
            "qk_norm": qk_norm,
            "attention": attention,
            "residual": residual,
            "ffn": ffn,
        },
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    report = _unwrap_report(_load_json(args.cost_report))
    trace = _load_json(args.trace_summary)
    hardware = trace["hardware"]
    workload = trace["workload"]

    expected_hardware = {
        "mlen": args.mlen,
        "blen": args.blen,
        "vlen": args.vlen,
        "hlen": args.hlen,
    }
    for key, expected in expected_hardware.items():
        if int(hardware[key]) != expected:
            raise ValueError(f"trace {key}={hardware[key]} does not match requested {expected}")
    if int(workload["seq_len"]) != args.seq_len or int(workload["batch_size"]) != args.batch_size:
        raise ValueError("trace workload does not match requested sequence length and batch size")

    old = _legacy_qwen_model(
        args.model_config,
        args.settings,
        args.isa,
        mlen=args.mlen,
        blen=args.blen,
        vlen=args.vlen,
        hlen=args.hlen,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        frequency_hz=args.frequency_hz,
    )
    ideal = _same_trace_ideal_one_cycle(report, trace)
    ideal_scheduled = None
    if args.ideal_scheduled_report is not None:
        ideal_bundle = _load_json(args.ideal_scheduled_report)
        ideal_scheduled = ideal_bundle.get(
            "ideal_vector_scalar_one_cycle_scheduled",
            ideal_bundle.get("report", ideal_bundle),
        )
        if not isinstance(ideal_scheduled, dict):
            raise ValueError("ideal scheduled report must contain a report object")
    same_trace_legacy = _same_trace_legacy_opcode_timing(args.settings, trace)

    actual_ns = float(report["roofline_latency_ns"])
    old_ns = float(old["full_model_cycles"]) * 1e9 / args.frequency_hz
    ideal_ns = float(
        ideal["stage_roofline_ns"]
        if ideal_scheduled is None
        else ideal_scheduled["roofline_latency_ns"]
    )
    total_gap = actual_ns - old_ns
    timing_gap = actual_ns - ideal_ns
    represented_work_gap = ideal_ns - old_ns

    current_categories = report["category_latency_ns"]
    opcode_counts = {
        "matrix": _prefix_count(trace["dynamic_opcodes"], "M_"),
        "vector": _prefix_count(trace["dynamic_opcodes"], "V_"),
        "scalar": _prefix_count(trace["dynamic_opcodes"], "S_"),
        "control": _prefix_count(trace["dynamic_opcodes"], "C_"),
    }
    average_resource_cycles = {
        "matrix": float(current_categories["matrix_compute"]) / opcode_counts["matrix"],
        "vector": float(current_categories["vector_compute"]) / opcode_counts["vector"],
        "scalar": float(current_categories["scalar_compute"]) / opcode_counts["scalar"],
        "control": float(current_categories["control"]) / opcode_counts["control"],
    }

    return {
        "schema_version": 1,
        "configuration": {
            "model_config": str(args.model_config),
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "mlen": args.mlen,
            "blen": args.blen,
            "vlen": args.vlen,
            "hlen": args.hlen,
            "frequency_hz": args.frequency_hz,
        },
        "current_costemitter": {
            "compute_pipeline_ns": report["compute_latency_ns"],
            "stage_roofline_ns": actual_ns,
            "memory_latency_ns": report["memory_latency_ns"],
            "serial_resource_work_cycles": report["compute_resource_work_cycles"],
            "category_resource_work_cycles": current_categories,
            "compute_pipeline_fidelity": report["compute_pipeline_fidelity"],
            "compute_validation": report["compute_validation"],
        },
        "legacy_closed_form": old,
        "same_trace_legacy_opcode_timing": same_trace_legacy,
        "same_trace_ideal_vector_scalar_one_cycle_resource_work": ideal,
        "same_trace_ideal_vector_scalar_one_cycle_scheduled": ideal_scheduled,
        "opcode_counts": opcode_counts,
        "average_current_resource_cycles_per_opcode": average_resource_cycles,
        "mismatch_decomposition": {
            "current_over_legacy_ratio": actual_ns / old_ns,
            "ideal_same_trace_over_legacy_ratio": ideal_ns / old_ns,
            "current_over_ideal_same_trace_ratio": actual_ns / ideal_ns,
            "total_gap_ns": total_gap,
            "timing_and_hazard_gap_ns": timing_gap,
            "represented_work_gap_ns": represented_work_gap,
            "timing_and_hazard_share_of_gap": timing_gap / total_gap,
            "represented_work_share_of_gap": represented_work_gap / total_gap,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost-report", type=Path, required=True)
    parser.add_argument("--trace-summary", type=Path, required=True)
    parser.add_argument(
        "--ideal-scheduled-report",
        type=Path,
        help=(
            "optional report produced by replaying the same ordered CostTrace "
            "with a one-cycle Vector/Scalar timing artifact"
        ),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("Workspace/qwen3_32b_dense_analytic/qwen3-32b.json"),
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path("Workspace/gqa_pipeline_validation/target_smoke/settings/m2048_b1024.toml"),
    )
    parser.add_argument(
        "--isa",
        type=Path,
        default=Path("analytic_models/performance/customISA_lib.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mlen", type=int, default=2048)
    parser.add_argument("--blen", type=int, default=1024)
    parser.add_argument("--vlen", type=int, default=2048)
    parser.add_argument("--hlen", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=482)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frequency-hz", type=float, default=1e9)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = compare(args)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
