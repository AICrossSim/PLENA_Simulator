#!/usr/bin/env python3
"""Validate HBM V4 with Ramulator arrivals from the rtl-v1 schedule.

The functional transactional emulator executes opcodes serially.  Its DMA
service interval is therefore measured after functional/legacy compute time
has advanced the event runtime, which is not the same arrival process as the
separate rtl-v1 scoreboard.  This runner closes that time-domain gap:

1. schedule the compressed CostTrace with V4 occurrence latencies;
2. replay the exact production DMA sequence at those absolute service starts;
3. reschedule with the observed occurrence durations; and
4. repeat until arrival cycles and service durations stabilize.

Only DMA requests and timing are replayed. Numerical execution and the
transactional comparison gate are deliberately untouched.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
for dependency in (ROOT, ROOT / "PLENA_Compiler", ROOT / "PLENA_Tools"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from analytic_models.performance.compiler_cost_model import (  # noqa: E402
    TransactionalCycleModel,
    _actual_dma_service_provider,
    _build_compute_timing_context,
    _evaluate_scheduled_shadow,
    _hardware_from_settings,
    validate_hbm_service_v4_system_case,
)
from analytic_models.performance.hbm_service_model import (  # noqa: E402
    HbmConfig,
    MemoryPrecisionConfig,
)
from analytic_models.performance.hbm_service_v4 import (  # noqa: E402
    DMA_SEMANTIC_VERSION,
    MANIFEST_HASH_ALGORITHM,
    PHYSICAL_BURST_BYTES,
    REQUEST_BYTES,
    HbmServiceModelV4,
    V4DmaServiceProvider,
    combined_request_manifest_hash,
    request_manifest_fixture_hash,
    stream_occurrence_transfer,
)
from compiler.aten.cost_frontend import compile_native_decoder_cost_trace  # noqa: E402


DEFAULT_TIMING = ROOT / "transactional_emulator/calibration/rtl_opcode_timing_v1.json"
DEFAULT_DRIVER = ROOT / "transactional_emulator/target/release/hbm_dma_calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--precision-config", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument(
        "--initial-observed-dma-trace",
        type=Path,
        help=(
            "seed the fixed-point iteration from a prior observed replay; the "
            "seed is never accepted directly and is re-executed by Ramulator"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed iteration_N_results.json files in --out-dir",
    )
    parser.add_argument(
        "--convergence-cycle-tolerance",
        type=int,
        default=0,
        help="fallback tolerance for all fixed-point stability checks",
    )
    parser.add_argument("--arrival-convergence-cycle-tolerance", type=int)
    parser.add_argument("--duration-convergence-cycle-tolerance", type=int)
    parser.add_argument("--physical-start-convergence-cycle-tolerance", type=int)
    parser.add_argument("--makespan-convergence-cycle-tolerance", type=int)
    parser.add_argument("--hbm-work-convergence-cycle-tolerance", type=int)
    return parser.parse_args()


def _format_payload(precision_payload: dict[str, Any], fmt: Any) -> dict[str, Any]:
    name = str(precision_payload["weight"]).upper().replace("_", "")
    exponent_bits = mantissa_bits = 0
    if fmt.family == "mxfp":
        if name.startswith("MXFPE") and "M" in name:
            exponent, mantissa = name.removeprefix("MXFPE").split("M", 1)
            exponent_bits, mantissa_bits = int(exponent), int(mantissa)
        elif fmt.element_bits == 4:
            exponent_bits, mantissa_bits = 1, 2
        elif fmt.element_bits == 8:
            exponent_bits, mantissa_bits = 4, 3
        else:
            raise ValueError(f"unsupported MXFP element width {fmt.element_bits}")
    return {
        "family": fmt.family,
        "element_bits": fmt.element_bits,
        "scale_bits": fmt.scale_bits,
        "block": fmt.block,
        "exponent_bits": exponent_bits,
        "mantissa_bits": mantissa_bits,
    }


def _max_delta(left: Sequence[int], right: Sequence[int]) -> int:
    if len(left) != len(right):
        raise ValueError(f"sequence length mismatch: {len(left)} != {len(right)}")
    return max((abs(int(a) - int(b)) for a, b in zip(left, right, strict=True)), default=0)


def _stream_durations(
    ordered: Sequence[tuple[Any, Any, int, Any]], durations: Sequence[int]
) -> dict[int, tuple[int, ...]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for (stream, _fmt, _position, _estimate), duration in zip(
        ordered, durations, strict=True
    ):
        grouped[int(stream.stream_index)].append(int(duration))
    return {stream: tuple(values) for stream, values in grouped.items()}


def _schedule(
    trace: Any,
    settings: TransactionalCycleModel,
    timing: Any,
    provider: Any,
    fidelity: str,
) -> Any:
    result = _evaluate_scheduled_shadow(
        trace,
        settings,
        timing,
        enabled=True,
        hbm_service_cycles=provider,
        hbm_fidelity=fidelity,
    )
    if result.status != "complete":
        raise RuntimeError(f"rtl-v1 schedule unavailable: {result.reason}")
    provider.assert_consumed()
    return result


def _write_plan(
    path: Path,
    *,
    channels: int,
    clock_period_ps: int,
    format_payload: dict[str, Any],
    transfers: Sequence[dict[str, Any]],
    arrivals: Sequence[int],
    iteration: int,
    makespan_cycles: int | None,
) -> None:
    if len(transfers) != len(arrivals):
        raise ValueError("transfer and arrival counts differ")
    pattern = {
        "id": f"rtl-v1-arrival-replay-iteration-{iteration}",
        "channels": channels,
        "repetitions": 1,
        "warmup": 0,
        "format": format_payload,
        "transfer": transfers[0],
        "transfer_sequence": list(transfers),
        "transfer_arrival_cycles": [int(value) for value in arrivals],
        "arrival_clock_period_ps": clock_period_ps,
        "repeat_axes": [],
        "conditioner_addresses": [],
        "run_transactional": True,
        "run_raw": False,
        "record_occurrence_cycles": True,
    }
    payload = {
        "schema_version": 4,
        "ramulator_preset": "HBM2_2Gbps",
        "mapper": "MOP4CLXOR",
        "request_bytes": REQUEST_BYTES,
        "physical_burst_bytes": PHYSICAL_BURST_BYTES,
        "dma_semantic_version": DMA_SEMANTIC_VERSION,
        "request_manifest_hash_algorithm": MANIFEST_HASH_ALGORITHM,
        "request_manifest_fixture_hash": request_manifest_fixture_hash(),
        "arrival_timing_semantics": "rtl-v1-service-start-fixed-point-v1",
        "arrival_iteration": iteration,
        "scheduled_shadow_makespan_cycles": makespan_cycles,
        "patterns": [pattern],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if args.max_iterations <= 0:
        raise ValueError("--max-iterations must be positive")
    if args.convergence_cycle_tolerance < 0:
        raise ValueError("--convergence-cycle-tolerance cannot be negative")
    tolerance_names = (
        "arrival_convergence_cycle_tolerance",
        "duration_convergence_cycle_tolerance",
        "physical_start_convergence_cycle_tolerance",
        "makespan_convergence_cycle_tolerance",
        "hbm_work_convergence_cycle_tolerance",
    )
    if any(getattr(args, name) is not None and getattr(args, name) < 0 for name in tolerance_names):
        raise ValueError("fixed-point convergence tolerances cannot be negative")
    tolerances = {
        name.removesuffix("_convergence_cycle_tolerance"): (
            args.convergence_cycle_tolerance
            if getattr(args, name) is None
            else int(getattr(args, name))
        )
        for name in tolerance_names
    }
    if not args.driver.is_file():
        raise FileNotFoundError(
            f"production DMA driver not found at {args.driver}; build it with "
            "cargo build --release --bin hbm_dma_calibration"
        )

    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = TransactionalCycleModel.load(args.settings)
    precision_payload = json.loads(args.precision_config.read_text())
    precision = MemoryPrecisionConfig.from_mapping(precision_payload)
    model = HbmServiceModelV4.load(args.calibration)
    trace = compile_native_decoder_cost_trace(
        args.model_config,
        _hardware_from_settings(args.model_config, settings),
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
    )
    timing = _build_compute_timing_context(
        settings,
        precision,
        compute_timing_mode="rtl-v1",
        rtl_timing_calibration=DEFAULT_TIMING,
    )

    hbm = HbmConfig(channels=settings.hbm_channels)
    ordered_provider = V4DmaServiceProvider(
        trace, precision, hbm, model, settings.clock_period_ps
    )
    ordered = tuple(ordered_provider.ordered_estimates())
    ordered_manifests = tuple(ordered_provider.ordered_manifests())
    if not ordered:
        raise ValueError("CostTrace has no production DMA occurrences")
    if len(ordered_manifests) != len(ordered):
        raise ValueError("V4 estimate and manifest occurrence counts differ")
    signatures = {fmt.request_signature() for _stream, fmt, _pos, _est in ordered}
    if len(signatures) != 1:
        raise ValueError(
            "arrival replay currently requires one physical format, got "
            f"{sorted(signatures)}"
        )
    transfers = tuple(
        stream_occurrence_transfer(stream, position)
        for stream, _fmt, position, _estimate in ordered
    )
    format_payload = _format_payload(precision_payload, ordered[0][1])
    expected_read_lines = sum(len(item.read_lines) for item in ordered_manifests)
    expected_write_lines = sum(len(item.write_lines) for item in ordered_manifests)
    expected_manifest_hash = combined_request_manifest_hash(ordered_manifests)

    seed_trace_path: str | None = None
    seed_durations: list[int] | None = None
    if args.initial_observed_dma_trace is not None:
        seed_trace_path = str(args.initial_observed_dma_trace.resolve())
        seed_payload = json.loads(args.initial_observed_dma_trace.read_text())
        seed_events = tuple(seed_payload.get("events", ()))
        if len(seed_events) != len(ordered):
            raise ValueError(
                "initial observed DMA count differs from CostTrace: "
                f"seed={len(seed_events)}, expected={len(ordered)}"
            )
        seed_durations = []
        for index, (event, (stream, _fmt, _position, _estimate)) in enumerate(
            zip(seed_events, ordered, strict=True)
        ):
            if str(event["opcode"]) != stream.opcode:
                raise ValueError(
                    "initial observed DMA opcode order differs at "
                    f"{index}: seed={event['opcode']}, expected={stream.opcode}"
                )
            start = int(event["start_cycle"])
            completion = int(event["completion_cycle"])
            if completion < start:
                raise ValueError(f"initial observed DMA completes before start: {event!r}")
            seed_durations.append(max(1, completion - start))
        seed_provider = _actual_dma_service_provider(
            _stream_durations(ordered, seed_durations)
        )
        scheduled = _schedule(
            trace, settings, timing, seed_provider, "ramulator_observed_seed"
        )
    else:
        prediction_provider = V4DmaServiceProvider(
            trace, precision, hbm, model, settings.clock_period_ps
        )
        scheduled = _schedule(
            trace, settings, timing, prediction_provider, "post_hoc_v4"
        )
    if len(scheduled.dma_occurrences) != len(ordered):
        raise ValueError("predicted schedule DMA count differs from physical work")
    arrivals = [event.start_cycle for event in scheduled.dma_occurrences]
    previous_durations: list[int] | None = seed_durations
    previous_makespan = scheduled.makespan_cycles
    previous_hbm_work = None if seed_durations is None else sum(seed_durations)
    iteration_rows = []
    final_events: list[dict[str, Any]] = []
    converged = False
    canonical_manifest_hash: str | None = None

    for iteration in range(args.max_iterations):
        plan_path = output / f"iteration_{iteration}_plan.json"
        results_path = output / f"iteration_{iteration}_results.json"
        _write_plan(
            plan_path,
            channels=settings.hbm_channels,
            clock_period_ps=settings.clock_period_ps,
            format_payload=format_payload,
            transfers=transfers,
            arrivals=arrivals,
            iteration=iteration,
            makespan_cycles=scheduled.makespan_cycles,
        )
        if not (args.resume and results_path.is_file()):
            subprocess.run(
                [
                    str(args.driver),
                    "--input",
                    str(plan_path),
                    "--output",
                    str(results_path),
                    "--checkpoint-every",
                    "1",
                ],
                check=True,
            )
        result = json.loads(results_path.read_text())["patterns"][0]
        if canonical_manifest_hash is None:
            canonical_manifest_hash = str(result["request_manifest_hash"])
        parity = {
            "request_manifest_hash": str(result["request_manifest_hash"])
            == expected_manifest_hash,
            "request_manifest_hash_stable": str(result["request_manifest_hash"])
            == canonical_manifest_hash,
            "read_lines": int(result["read_lines"]) == expected_read_lines,
            "write_lines": int(result["write_lines"]) == expected_write_lines,
            "read_bytes": int(result["request_read_bytes"])
            == expected_read_lines * REQUEST_BYTES,
            "write_bytes": int(result["request_write_bytes"])
            == expected_write_lines * REQUEST_BYTES,
        }
        if not all(parity.values()):
            raise ValueError(f"Rust/Python production DMA manifest mismatch: {parity}")
        durations = [
            int(round(value))
            for value in result["production_dma_median_occurrence_cycles"]
        ]
        actual_starts = [
            int(round(value))
            for value in result["production_dma_median_occurrence_start_cycles"]
        ]
        actual_completions = [
            int(round(value))
            for value in result[
                "production_dma_median_occurrence_completion_cycles"
            ]
        ]
        if not (
            len(durations)
            == len(actual_starts)
            == len(actual_completions)
            == len(ordered)
        ):
            raise ValueError("arrival replay result count differs from CostTrace")

        observed_provider = _actual_dma_service_provider(
            _stream_durations(ordered, durations)
        )
        next_scheduled = _schedule(
            trace, settings, timing, observed_provider, "ramulator_observed"
        )
        next_arrivals = [
            event.start_cycle for event in next_scheduled.dma_occurrences
        ]
        arrival_delta = _max_delta(arrivals, next_arrivals)
        duration_delta = (
            None
            if previous_durations is None
            else _max_delta(previous_durations, durations)
        )
        physical_start_delay = max(
            (actual - target for actual, target in zip(actual_starts, arrivals, strict=True)),
            default=0,
        )
        hbm_work = sum(durations)
        makespan_delta = (
            None
            if previous_makespan is None or next_scheduled.makespan_cycles is None
            else abs(next_scheduled.makespan_cycles - previous_makespan)
        )
        hbm_work_delta = (
            None
            if previous_hbm_work is None
            else abs(hbm_work - previous_hbm_work)
        )
        iteration_rows.append(
            {
                "iteration": iteration,
                "plan": str(plan_path),
                "results": str(results_path),
                "scheduled_makespan_cycles": next_scheduled.makespan_cycles,
                "max_arrival_change_cycles": arrival_delta,
                "max_duration_change_cycles": duration_delta,
                "max_physical_start_delay_cycles": physical_start_delay,
                "makespan_change_cycles": makespan_delta,
                "hbm_work_change_cycles": hbm_work_delta,
                "hbm_work_cycles": hbm_work,
                "request_parity": parity,
            }
        )
        final_events = [
            {
                "sequence": event.sequence,
                "stream_index": event.stream_index,
                "opcode": event.opcode,
                "start_cycle": event.start_cycle,
                "completion_cycle": event.completion_cycle,
                "resource": "ramulator_arrival_replay",
            }
            for event in next_scheduled.dma_occurrences
        ]
        converged = (
            arrival_delta <= tolerances["arrival"]
            and duration_delta is not None
            and duration_delta <= tolerances["duration"]
            and physical_start_delay <= tolerances["physical_start"]
            and makespan_delta is not None
            and makespan_delta <= tolerances["makespan"]
            and hbm_work_delta is not None
            and hbm_work_delta <= tolerances["hbm_work"]
        )
        arrivals = next_arrivals
        previous_durations = durations
        previous_makespan = next_scheduled.makespan_cycles
        previous_hbm_work = hbm_work
        scheduled = next_scheduled
        if converged:
            break

    observed_trace = {
        "schema_version": 2,
        "timing_mode": "rtl-v1",
        "clock_period_ps": settings.clock_period_ps,
        "dma_timing_semantics": "rtl-v1-arrival-aware-ramulator-replay-v1",
        "converged": converged,
        "iterations": len(iteration_rows),
        "events": final_events,
    }
    observed_path = output / "observed_dma_arrival_replay.json"
    observed_path.write_text(json.dumps(observed_trace, indent=2, sort_keys=True) + "\n")

    validation = validate_hbm_service_v4_system_case(
        trace,
        settings,
        model,
        precision,
        observed_trace,
    )
    validation["arrival_replay"] = {
        "converged": converged,
        "convergence_definition": "bounded_fixed_point_stability_v1",
        "convergence_cycle_tolerances": tolerances,
        "initial_observed_dma_trace": seed_trace_path,
        "initial_trace_used_as_seed_only": seed_trace_path is not None,
        "iterations": iteration_rows,
        "observed_trace": str(observed_path),
    }
    # A model cannot be promoted from a replay that has not reached a stable
    # arrival/service fixed point, even if aggregate error happens to pass.
    validation["acceptance"]["arrival_replay_converged"] = converged
    validation["accepted"] = all(validation["acceptance"].values())
    validation_path = output / "system_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    summary = {
        "accepted": validation["accepted"],
        "converged": converged,
        "iterations": len(iteration_rows),
        "dma_occurrences": len(ordered),
        "system_validation": str(validation_path),
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if validation["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
