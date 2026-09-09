#!/usr/bin/env python3
"""Run an equal-multiplier fixed-route MoE comparison with mandatory gates.

No cycle estimate or speedup is produced unless every architecture and repeat
passes numerical, identity, resource and deterministic-repeat checks. This is
an operator experiment; it does not measure router execution or full requests.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import struct
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numerical_gate(actual, expected, atol, rtol):
    require(len(actual) == len(expected), "output token count differs from golden")
    maximum_error = 0.0
    for token, (row, reference) in enumerate(zip(actual, expected)):
        require(len(row) == len(reference), "output width differs from golden")
        for column, (value, target) in enumerate(zip(row, reference)):
            require(math.isfinite(value) and math.isfinite(target), "nonfinite output")
            error = abs(value - target)
            maximum_error = max(maximum_error, error)
            require(
                error <= atol + rtol * abs(target),
                f"golden mismatch at token {token}, column {column}: {value} != {target}",
            )
    return maximum_error


def validate_output_pair(bits, values):
    require(len(bits) == len(values), "BF16/FP32 token count mismatch")
    for bit_row, value_row in zip(bits, values):
        require(len(bit_row) == len(value_row), "BF16/FP32 width mismatch")
        for bit, value in zip(bit_row, value_row):
            require(
                isinstance(bit, int) and not isinstance(bit, bool) and 0 <= bit <= 65535, "output BF16 must be uint16"
            )
            decoded = struct.unpack("<f", struct.pack("<I", bit << 16))[0]
            require(math.isfinite(decoded) and decoded == value, "BF16 bits disagree with FP32 output")


def validate_output_shape(values, tokens, width):
    require(isinstance(values, list) and len(values) == tokens, "output token count differs from workload")
    require(all(isinstance(row, list) and len(row) == width for row in values), "output width differs from workload")


def validate_native(envelope, architecture, channels):
    cal = envelope["memory_model"]["calibration"]
    result = envelope["result"]
    require(
        cal["capi_version"] == 2
        and cal["native_transaction_bytes"] == 32
        and cal["mapper"] == "CacheLineInterleave"
        and cal["channel_shift"] == 5
        and cal["channels"] == channels,
        "native geometry is not calibrated HBM2",
    )
    require(
        cal["issue_policy"] == architecture.get("dma", {}).get("issue_policy", "global_fifo")
        and cal["issue_period_ps"] == architecture["clock_period_ps"],
        "native injection configuration differs",
    )
    require(
        cal["native_pending"] == 0 and 0 <= cal["native_inflight_peak"] <= 256 and cal["submission_entries"] == 256,
        "native requests not drained or finite tracker bound exceeded",
    )
    if architecture.get("dma", {}).get("issue_policy") == "demand_aware":
        require(cal["submission_entries_available"] == 256 and cal["submission_inflight_peak"] <= 256,
                "demand-aware native descriptors leaked or exceeded their bound")
        require(cal["demand_accepted"] + cal["prefetch_accepted"] == sum(cal["accepted_per_channel"]),
                "demand/prefetch classes do not reconcile with native requests")
    for key in ("accepted_per_channel", "rejected_per_channel"):
        require(
            len(cal[key]) == channels and all(type(v) is int and v >= 0 for v in cal[key]),
            "invalid native channel counter",
        )
    native = cal["native_stats"]["memory_system"]
    controllers = native["controller"]
    require(len(controllers) == channels, "native controller count differs")
    for port, controller in enumerate(controllers):
        require(controller["id"] == "Channel " + str(port), "native channel identity mismatch")
        require(
            controller["num_read_reqs"] == controller["num_read_reqs_served"] == cal["accepted_per_channel"][port],
            "accepted/read-command counts differ",
        )
        require(
            controller["num_read_reqs_forwarded"] == controller["num_write_reqs"] == 0,
            "unexpected forwarding or writes in read-only weight experiment",
        )
    require(
        sum(cal["accepted_per_channel"]) == native["total_num_read_requests"]
        and native["total_num_read_requests"] * 32 == result["hbm_read_bytes"],
        "native byte accounting mismatch",
    )
    identity = envelope["provenance"]
    require(digest(identity["native_library_path"]) == identity["native_library_sha256"], "native library hash differs")
    if architecture.get("dma"):
        dma = result["dma_frontend"]
        require(all(type(v) is int and v >= 0 for v in dma.values()), "invalid DMA metric")
        require(dma["reserved_bytes"] <= architecture["dma"]["frontend_sram_bytes"], "DMA metadata SRAM exceeded")
        require(dma["mshr_peak"] <= result["global_dma_inflight_peak"], "MSHR entries exceed admitted waiters")
        require(
            dma["sector_requests"] >= dma["merged_sectors"]
            and (dma["sector_requests"] - dma["merged_sectors"]) * 32 == result["hbm_read_bytes"],
            "MSHR sectors do not reconcile",
        )
        require(
            dma["lookup_busy_ps"]
            == architecture["dma"].get("lookup_ii_cycles", 2) * dma["line_requests"] * architecture["clock_period_ps"],
            "lookup service unaccounted",
        )
        require(
            dma["copy_busy_ps"] * 32 >= dma["useful_copy_bytes"] * architecture["clock_period_ps"],
            "copy bandwidth exceeded",
        )
        require(
            dma["lookup_busy_ps"] <= 4 * result["total_ps"] and dma["copy_busy_ps"] <= 8 * result["total_ps"],
            "DMA port occupancy exceeds elapsed time",
        )


def validate_refined_core(observed, core, architecture, workload, jobs, total_ps):
    """Reconstruct charged storage; Q is output state, never free operand SRAM.

    This checks the jobs actually assigned to a core. A work-conserving dispatch
    may change assignment when Q changes the set of jobs that fit; passing these
    gates alone does not establish an identical schedule in an A/B experiment.
    """
    config = core.get("refinement")
    detail = observed.get("refinement")
    if config is None:
        require(detail is None, "unexpected refined timing report")
        return
    require(architecture["schema_version"] == 2 and detail is not None, "missing refined V2 report")
    require(all(type(config[field]) is int and config[field] > 0 for field in (
        "m_rows", "active_n_tiles", "weight_read_elements_per_cycle", "accumulator_elements_per_cycle"
    )), "invalid refined dimensions or ports")
    require(config["tail_policy"] in ("padded", "valid_rows"), "invalid M tail policy")
    require(detail["m_rows"] == config["m_rows"], "reported M tile differs from configuration")
    p, r, mt = core["blen"], core["mlen"], config["m_rows"]
    slots = core.get("weight_slots", 2)
    pool = config.get("output_pool")
    pool_report = detail.get("output_pool")
    if pool is not None:
        require(all(type(pool[field]) is int and pool[field] > 0 for field in (
            "output_contexts", "operand_stages", "scheduler_cycles"
        )), "pool dimensions and scheduler cost must be positive integers")
        require(pool["output_contexts"] <= 256 and pool["operand_stages"] <= 2 and config["active_n_tiles"] == 2,
                "pool mode needs one or two stages and active_n_tiles=2")
        require(type(slots) is int and 1 <= slots <= 4 and pool["operand_stages"] <= slots,
                "pool stages exceed finite weight slots")
        q, stages = pool["output_contexts"], pool["operand_stages"]
        control = 128 * q + 64 * (slots + stages)
        pending = q * mt * p * 4
        require(pool_report is not None, "missing bounded output-pool report")
        require(all(type(v) is int and v >= 0 for v in pool_report.values()), "invalid output-pool metric")
        for field, expected in (("contexts_capacity", q), ("operand_stages_capacity", stages),
                                ("control_reserved_bytes", control), ("pending_result_reserved_bytes", pending)):
            require(pool_report[field] == expected, "output-pool reservation differs: " + field)
        require(pool_report["contexts_peak"] <= q and pool_report["pending_contexts_peak"] <= pool_report["contexts_peak"],
                "output or pending contexts exceed bounded pool")
        require(pool_report["operand_stages_peak"] <= stages, "operand stages exceed physical latches")
        require(pool_report["scheduler_busy_ps"] == pool_report["scheduler_visits"] * pool["scheduler_cycles"]
                * architecture["clock_period_ps"], "output scheduler service is unaccounted")
        require(pool_report["scheduler_busy_ps"] <= total_ps, "output scheduler occupancy exceeds run")
        require(not jobs or pool_report["scheduler_visits"] > 0, "output scheduler has zero cost for useful work")
        require(all((job["rows"] + mt - 1) // mt <= q for job in jobs),
                "assigned expert does not fit its whole M cohort in output pool")
    else:
        require(pool_report is None, "unexpected output-pool report")
        stages = config["active_n_tiles"]
        require(type(slots) is int and 2 <= slots <= 4 and 1 <= stages <= 3 and stages <= slots,
                "legacy N contexts exceed resident weight slots")
    latch = config["operand_latch_bytes"]
    require(type(latch) is int and latch >= stages * p * r * 2, "operand latches are underfunded")
    require(slots * p * (r // 8) * 25 + latch <= core["weight_sram_bytes"],
            "packed/decoded weight slots plus latches exceed SRAM")
    require(detail["operand_latch_reserved_bytes"] == latch, "reported operand latch reservation differs")
    require(observed["weight_sram_peak_bytes"] == observed["weight_slots_peak"] * p * (r // 8) * 25 + latch,
            "weight peak omits packed scales, decoded operands or reserved latches")
    pipeline = p * ((r - 1).bit_length() + architecture["mac_pipeline_cycles"]) * 4
    require(observed["pipeline_register_bytes"] == pipeline, "pipeline register reservation differs")
    expected_acc = expected_context = expected_pending = expected_contexts = 0
    for job in jobs:
        rows = job["rows"]
        if pool is None:
            contexts = stages * ((rows + mt - 1) // mt)
            control, pending = 32 * contexts, stages * rows * p * 4
        else:
            contexts = q
        expected_acc = max(expected_acc, rows * max(workload["input_dim"], workload["expert_hidden_dim"]) * 4
                           + pipeline + control + pending)
        expected_context = max(expected_context, control)
        expected_pending = max(expected_pending, pending)
        expected_contexts = max(expected_contexts, contexts)
    require(observed["accumulator_peak_bytes"] == expected_acc,
            "accumulator reservation omits output data, pipeline, contexts or pending results")
    require(expected_acc <= core["accumulator_bytes"], "modeled output pool exceeds accumulator SRAM")
    require(detail["output_context_peak_bytes"] == expected_context
            and detail["pending_result_peak_bytes"] == expected_pending, "output state accounting differs")
    require(detail["output_contexts_peak"] <= expected_contexts, "output-context peak exceeds reservation")
    for field in ("weight_port_busy_ps", "accumulator_port_busy_ps", "output_finalize_elapsed_ps"):
        require(type(detail[field]) is int and 0 <= detail[field] <= total_ps,
                "refined service occupancy exceeds elapsed run: " + field)
    require(detail["finalized_elements"] == sum(job["rows"] for job in jobs)
            * (2 * workload["expert_hidden_dim"] + workload["input_dim"]), "final output drain is incomplete")
    if pool is not None or "projections" in observed:
        validate_projection_reports(observed, core, architecture, workload, jobs, total_ps)


def validate_projection_reports(observed, core, architecture, workload, jobs, total_ps):
    """Projection intervals are disjoint per core; individual waits may overlap.

    Tile-load sums include concurrent requests and must not be interpreted as
    an additive share of wall time. Each serialized port has its own bound.
    """
    projections = observed["projections"]
    expected = {(job["job"], name): job for job in jobs for name in ("gate", "up", "down")}
    identities = [(entry["job"], entry["projection"]) for entry in projections]
    require(len(identities) == len(set(identities)) and set(identities) == set(expected),
            "missing or duplicate projection service record")
    last_end = 0
    config = core["refinement"]
    pool = config.get("output_pool")
    for entry in projections:
        job = expected[(entry["job"], entry["projection"])]
        n, k = workload["expert_hidden_dim"], workload["input_dim"]
        if entry["projection"] == "down":
            n, k = k, n
        require((entry["expert"], entry["m"], entry["n"], entry["k"]) == (job["expert"], job["rows"], n, k),
                "projection dimensions differ from completed expert")
        require(all(type(entry[field]) is int for field in ("start_ps", "end_ps")) and
                job["start_ps"] <= entry["start_ps"] <= entry["end_ps"] <= job["compute_done_ps"]
                and last_end <= entry["start_ps"], "projection service intervals overlap or escape their job")
        last_end = entry["end_ps"]
        elapsed = entry["end_ps"] - entry["start_ps"]
        metrics = entry["metrics"]
        require(all(type(v) is int and v >= 0 for key, v in metrics.items() if key != "tile_loads"),
                "invalid projection counter")
        require(metrics["useful_macs"] == job["rows"] * n * k and metrics["issued_macs"] >= metrics["useful_macs"],
                "projection MAC count differs from matrix dimensions")
        for field in ("compute_busy_ps", "weight_port_busy_ps", "accumulator_port_busy_ps", "scheduler_busy_ps",
                      "output_finalize_elapsed_ps"):
            require(metrics[field] <= elapsed, "projection port occupancy exceeds interval: " + field)
        loads = metrics["tile_loads"]
        count = ((n + core["blen"] - 1) // core["blen"]) * ((k + core["mlen"] - 1) // core["mlen"])
        require(all(type(v) is int and v >= 0 for v in loads.values()) and loads["count"] == count,
                "weight load count must be one per (N,K) tile across all M rows")
        require(0 <= loads["min_ps"] <= loads["max_ps"] <= elapsed
                and count * loads["min_ps"] <= loads["total_ps"] <= count * loads["max_ps"],
                "invalid tile-load service distribution")
        require(metrics["weight_slots_peak"] <= core.get("weight_slots", 2), "projection exceeds weight slots")
        if pool is not None:
            require(metrics["contexts_peak"] <= pool["output_contexts"]
                    and metrics["pending_contexts_peak"] <= metrics["contexts_peak"]
                    and metrics["operand_stages_peak"] <= pool["operand_stages"], "projection exceeds finite pool")
            require(metrics["tile_admissions"] == count, "pool tile admission/load counts differ")
            require(metrics["band_admissions"] == (n + core["blen"] - 1) // core["blen"],
                    "pool must admit each N band exactly once")
            require(metrics["context_updates"] == count * ((job["rows"] + config["m_rows"] - 1) // config["m_rows"]),
                    "pool did not execute all M contexts in each K tile")
            require(metrics["scheduler_visits"] > 0 and metrics["scheduler_busy_ps"]
                    == metrics["scheduler_visits"] * pool["scheduler_cycles"] * architecture["clock_period_ps"],
                    "projection scheduler service differs")
    for field in ("useful_macs", "issued_macs", "compute_busy_ps", "hbm_read_bytes",
                  "accumulator_dependency_stall_ps", "pipeline_drain_ps", "weight_ready_wait_ps"):
        require(sum(e["metrics"][field] for e in projections) == observed[field],
                "projection/core totals differ: " + field)
    for field in ("weight_port_busy_ps", "weight_port_wait_ps", "accumulator_port_busy_ps", "accumulator_port_wait_ps",
                  "output_finalize_elapsed_ps"):
        require(sum(e["metrics"][field] for e in projections) == observed["refinement"][field],
                "projection/refinement totals differ: " + field)
    if pool is not None:
        for field in ("scheduler_busy_ps", "scheduler_visits", "band_admissions", "tile_admissions", "context_updates"):
            require(sum(e["metrics"][field] for e in projections) == observed["refinement"]["output_pool"][field],
                    "projection/output-pool totals differ: " + field)
    if pool is not None or "tile_loads" in observed:
        loads = [e["metrics"]["tile_loads"] for e in projections]
        expected_loads = dict(count=sum(v["count"] for v in loads), total_ps=sum(v["total_ps"] for v in loads),
                              min_ps=min((v["min_ps"] for v in loads), default=0),
                              max_ps=max((v["max_ps"] for v in loads), default=0))
        require(observed["tile_loads"] == expected_loads, "projection/core tile-load distributions differ")


def validate_run(envelope, golden, workload, architecture, atol, rtol, hbm_channels=8):
    result = envelope["result"]
    require(result.get("numerical_execution", True) is True, "timing-only output is not numerical evidence")
    require(
        envelope.get("evidence_level") != "fixed_route_timing_only_no_numerical_validation",
        "timing-only report rejected",
    )
    require(
        envelope["memory_model"]["channels"] == hbm_channels
        and envelope["memory_model"]["upper_burst_bytes"] == 64
        and envelope["memory_model"]["name"] == "Ramulator HBM2 preset",
        "unexpected memory model or bandwidth configuration",
    )
    for field in (
        "total_ps",
        "multipliers",
        "useful_macs",
        "issued_macs",
        "hbm_read_bytes",
        "hbm_write_bytes",
        "global_dma_inflight_peak",
        "global_dma_staging_peak_bytes",
        "combine_sram_peak_bytes",
        "shared_vector_busy_ps",
    ):
        value = result[field]
        require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            "invalid nonnegative integer metric: " + field,
        )
    require(envelope["workload_manifest"] == workload, "workload was changed")
    require(envelope["architecture_manifest"] == architecture, "architecture was changed")
    expected_multipliers = sum(core["blen"] * core["mlen"] for core in architecture["cores"])
    require(result["multipliers"] == expected_multipliers, "incorrect multiplier count")
    token_count = len(workload["inputs_bf16"])
    rows = len(workload["routes"])
    if workload.get("shared_expert") is not None:
        rows += token_count
    expected_macs = 3 * rows * workload["input_dim"] * workload["expert_hidden_dim"]
    require(result["useful_macs"] == expected_macs, "useful MAC count differs from MoE work")
    require(result["issued_macs"] >= expected_macs, "issued MAC count is too small")
    require(result["global_dma_inflight_peak"] <= architecture["global_dma_credits"], "global DMA credits exceeded")
    require(
        result["global_dma_staging_peak_bytes"] <= architecture["global_dma_staging_bytes"],
        "DMA staging capacity exceeded",
    )
    require(result["combine_sram_peak_bytes"] <= architecture["combine_sram_bytes"], "combine SRAM capacity exceeded")
    require(len(result["cores"]) == len(architecture["cores"]), "core count mismatch")
    for observed, core in zip(result["cores"], architecture["cores"]):
        require(observed["id"] == core["id"], "core identity mismatch")
        require(observed["multipliers"] == core["blen"] * core["mlen"], "core multiplier mismatch")
        for metric in (
            "useful_macs",
            "issued_macs",
            "jobs",
            "hbm_read_bytes",
            "compute_busy_ps",
            "vector_sram_peak_bytes",
            "accumulator_peak_bytes",
            "weight_sram_peak_bytes",
            "weight_slots_peak",
        ):
            require(
                isinstance(observed[metric], int) and not isinstance(observed[metric], bool) and observed[metric] >= 0,
                "invalid per-core metric: " + metric,
            )
        require(observed["issued_macs"] >= observed["useful_macs"], "per-core issued MAC deficit")
        require(observed["compute_busy_ps"] <= result["total_ps"], "core busy time exceeds run")
        for peak, capacity in (
            ("vector_sram_peak_bytes", "vector_sram_bytes"),
            ("accumulator_peak_bytes", "accumulator_bytes"),
            ("weight_sram_peak_bytes", "weight_sram_bytes"),
        ):
            require(observed[peak] <= core[capacity], core["id"] + " exceeded " + capacity)
        require(observed["weight_slots_peak"] <= core.get("weight_slots", 2), "weight slot count exceeded")
        if "read_cache_bytes" in core and not architecture.get("dma"):
            for metric in ("cache_requests", "cache_hits", "cache_port_busy_ps", "cache_peak_bytes"):
                require(type(observed[metric]) is int and observed[metric] >= 0, "invalid cache metric: " + metric)
            require(
                observed["cache_requests"] == observed["cache_hits"] + observed["hbm_read_bytes"] // 64,
                "cache requests do not reconcile with real HBM reads",
            )
            require(
                observed["cache_peak_bytes"] <= core["read_cache_bytes"] and observed["cache_peak_bytes"] % 80 == 0,
                "cache data/tag capacity exceeded",
            )
            expected_cache_cycles = (
                (observed["cache_requests"] + observed["hbm_read_bytes"] // 64) if core["read_cache_bytes"] else 0
            )
            require(
                observed["cache_port_busy_ps"] == expected_cache_cycles * architecture["clock_period_ps"],
                "cache port service is not accounted",
            )
            require(observed["cache_port_busy_ps"] <= result["total_ps"], "cache port busy time exceeds run")
        for metric in ("compute_busy_fraction", "mac_utilization"):
            require(math.isfinite(observed[metric]) and 0 <= observed[metric] <= 1.000000001, "invalid " + metric)
    jobs = result["job_completions"]
    if "dispatch_queue_bytes" in architecture:
        require(
            result["dispatch_queue_peak_bytes"] == len(jobs) * 64
            and result["dispatch_queue_peak_bytes"] <= architecture["dispatch_queue_bytes"],
            "ready-job descriptor capacity mismatch",
        )
        require(
            result["dispatcher_busy_ps"]
            == len(jobs) * architecture["dispatch_cycles"] * architecture["clock_period_ps"]
            and result["dispatcher_busy_ps"] <= result["total_ps"],
            "dispatcher service mismatch",
        )
    identities = [(job["expert"], job["shared"]) for job in jobs]
    expected_rows = {}
    for route in workload["routes"]:
        identity = (route["expert"], False)
        expected_rows[identity] = expected_rows.get(identity, 0) + 1
    if workload.get("shared_expert") is not None and token_count:
        expected_rows[(workload["shared_expert"]["expert"], True)] = token_count
    require(
        len(identities) == len(set(identities)) and set(identities) == set(expected_rows),
        "missing or duplicate expert completion",
    )
    require(sum(core["jobs"] for core in result["cores"]) == len(jobs), "job count mismatch")
    for field in ("useful_macs", "issued_macs", "hbm_read_bytes"):
        require(
            sum(core[field] for core in result["cores"]) == result[field], "per-core totals do not reconcile: " + field
        )
    jobs_per_core = {core["id"]: 0 for core in architecture["cores"]}
    rows_per_core = {core["id"]: 0 for core in architecture["cores"]}
    for job in jobs:
        require(job["core"] in jobs_per_core, "job completion names an unknown core")
        require(
            type(job["rows"]) is int and job["rows"] == expected_rows[(job["expert"], job["shared"])],
            "expert completion row count differs from workload",
        )
        jobs_per_core[job["core"]] += 1
        rows_per_core[job["core"]] += job["rows"]
        require(
            all(type(job[field]) is int for field in ("start_ps", "compute_done_ps", "output_copied_ps")),
            "job completion timestamps must be integers",
        )
        require(
            0 <= job["start_ps"] <= job["compute_done_ps"] <= job["output_copied_ps"] <= result["total_ps"],
            "invalid job completion timestamps",
        )
    for core in result["cores"]:
        require(core["jobs"] == jobs_per_core[core["id"]], "per-core job count mismatch")
        require(
            core["useful_macs"]
            == rows_per_core[core["id"]] * 3 * workload["input_dim"] * workload["expert_hidden_dim"],
            "per-core useful MACs differ from completed jobs",
        )
    for observed, core in zip(result["cores"], architecture["cores"]):
        validate_refined_core(observed, core, architecture, workload,
                              [job for job in jobs if job["core"] == core["id"]], result["total_ps"])
    require(
        result["hbm_read_bytes"] % (32 if architecture.get("dma") else 64) == 0,
        "HBM request bytes must match transfer granularity",
    )
    require(result["hbm_write_bytes"] == 0, "unexpected HBM output/intermediate write")
    require(result["shared_vector_busy_ps"] <= result["total_ps"], "vector busy time exceeds run")
    for source in (result, golden):
        for field in ("output_bf16", "output_f32"):
            validate_output_shape(source[field], token_count, workload["input_dim"])
    validate_output_pair(result["output_bf16"], result["output_f32"])
    validate_output_pair(golden["output_bf16"], golden["output_f32"])
    if architecture.get("dma") or "calibration" in envelope["memory_model"]:
        validate_native(envelope, architecture, hbm_channels)
    reference = golden["output_f32"]
    error = numerical_gate(result["output_f32"], reference, atol, rtol)
    bit_exact = result["output_bf16"] == golden["output_bf16"]
    if atol == 0 and rtol == 0:
        require(bit_exact, "exact comparison requires identical BF16 bits")
    return {
        "passed": True,
        "max_absolute_error": error,
        "output_bit_exact": bit_exact,
    }


def _run_comparison(
    binary,
    workload_path,
    golden_path,
    architecture_paths,
    output_dir,
    repeats=2,
    hbm_channels=8,
    atol=1e-5,
    rtol=0.01,
    timeout=180,
    workers=1,
    max_hbm_bytes=536_870_912,
):
    require(type(workers) is int and 1 <= workers <= 8, "workers must be an integer between 1 and 8")
    require(type(repeats) is int and repeats >= 2, "at least two integer repeats are required")
    require(
        type(hbm_channels) is int and 1 <= hbm_channels <= 32 and hbm_channels & (hbm_channels - 1) == 0,
        "HBM channels must be a power of two between 1 and 32",
    )
    require(
        type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0, "timeout must be finite and positive"
    )
    require(len(architecture_paths) >= 2, "provide baseline and at least one candidate")
    require(type(max_hbm_bytes) is int and max_hbm_bytes > 0, "max_hbm_bytes must be a positive integer")
    require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in (atol, rtol)), "invalid tolerance")
    binary = Path(binary).resolve()
    workload_path = Path(workload_path).resolve()
    golden_path = Path(golden_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Bind decoded manifests and the numerical reference to the same bytes
    # whose hashes will be published, even if another process edits a file.
    workload_bytes, golden_bytes = workload_path.read_bytes(), golden_path.read_bytes()
    workload, golden = json.loads(workload_bytes), json.loads(golden_bytes)
    architecture_bytes = [Path(path).read_bytes() for path in architecture_paths]
    architectures = [json.loads(payload) for payload in architecture_bytes]
    architecture_hashes = [hashlib.sha256(payload).hexdigest() for payload in architecture_bytes]
    golden_hash = hashlib.sha256(golden_bytes).hexdigest()
    names = [arch["name"] for arch in architectures]
    require(len(names) == len(set(names)), "architecture names must be unique")
    total_pes = [sum(c["blen"] * c["mlen"] for c in a["cores"]) for a in architectures]
    require(len(set(total_pes)) == 1, "comparison requires equal multiplier totals")
    # Match Rust's Option<usize>: missing/null activation supply defaults to MLEN.
    activation_totals = [
        sum(
            c["mlen"] if c.get("activation_elements_per_cycle") is None else c["activation_elements_per_cycle"]
            for c in a["cores"]
        )
        for a in architectures
    ]
    require(len(set(activation_totals)) == 1, "comparison requires equal total activation supply")
    for a in architectures:
        modes = [c.get("refinement") is not None for c in a["cores"]]
        require(all(modes) or not any(modes), "all cores in an architecture must use the same refinement mode")
    refined = [all(c.get("refinement") is not None for c in a["cores"]) for a in architectures]
    require(all(refined) or not any(refined), "compare legacy and refined timing in separate experiments")
    if all(refined):
        for field in ("weight_read_elements_per_cycle", "accumulator_elements_per_cycle"):
            require(
                len({sum(c["refinement"][field] for c in a["cores"]) for a in architectures}) == 1,
                "comparison requires equal total modeled SRAM port throughput: " + field,
            )
    for field in (
        "clock_period_ps",
        "mac_pipeline_cycles",
        "vector_elements_per_cycle",
        "global_dma_credits",
        "global_dma_staging_bytes",
        "combine_sram_bytes",
    ):
        require(len({a[field] for a in architectures}) == 1, "comparison requires same shared resource: " + field)
    for field, default in (("dispatch_cycles", 1), ("dispatch_queue_bytes", 262144), ("matrix_timing", "pipelined")):
        require(
            len({a.get(field, default) for a in architectures}) == 1,
            "comparison requires same shared resource/timing: " + field,
        )
    for field in ("vector_sram_bytes", "accumulator_bytes", "weight_sram_bytes"):
        require(
            len({sum(c[field] for c in a["cores"]) for a in architectures}) == 1,
            "comparison requires equal total configured SRAM: " + field,
        )
    require(
        len({sum(c.get("read_cache_bytes", 0) for c in a["cores"]) for a in architectures}) == 1,
        "comparison requires equal total configured SRAM: read_cache_bytes",
    )
    dma_configs = [a.get("dma") for a in architectures]
    require(
        all(d is None for d in dma_configs) or all(d is not None for d in dma_configs),
        "DMA configuration must be explicit for all architectures",
    )
    if all(d is not None for d in dma_configs):
        require(
            len({json.dumps(d, sort_keys=True) for d in dma_configs}) == 1,
            "comparison requires identical DMA policy and budget",
        )
    hbm_path = workload_path.parent / workload["hbm_file"]
    expected_hashes = {
        "workload_sha256": hashlib.sha256(workload_bytes).hexdigest(),
        "hbm_sha256": digest(hbm_path),
        "executable_sha256": digest(binary),
    }
    for field in ("workload_sha256", "hbm_sha256"):
        require(golden[field] == expected_hashes[field], "golden identity mismatch: " + field)
    bank_path = None
    bank_hash = None
    if workload.get("weight_bank") is not None:
        bank_path = workload_path.parent / workload["weight_bank"]["manifest"]
        bank_hash = digest(bank_path)
        require(bank_hash == workload["weight_bank"]["sha256"] == golden["bank_sha256"],
                "immutable weight bank identity differs from golden/window")
    run_dir = output_dir / ("run_" + uuid.uuid4().hex)
    run_dir.mkdir()

    def execute_architecture(index):
        path, architecture = architecture_paths[index], architectures[index]
        repeat_results, gates, native_repeats, library_hashes = [], [], [], []
        architecture_hash = architecture_hashes[index]
        for repeat in range(repeats):
            report_path = run_dir / f"arch{index:02d}_repeat{repeat:02d}.json"
            log_path = report_path.with_suffix(".log")
            command = [
                str(binary),
                "--workload",
                str(workload_path),
                "--architecture",
                str(Path(path).resolve()),
                "--output",
                str(report_path),
                "--hbm-channels",
                str(hbm_channels),
                "--max-hbm-bytes",
                str(max_hbm_bytes),
            ]
            with log_path.open("wb") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
            envelope = read_json(report_path)
            for field, value in dict(expected_hashes, architecture_sha256=architecture_hash).items():
                require(envelope["provenance"][field] == value, "input/binary identity changed: " + field)
            if bank_hash is not None:
                require(envelope["provenance"]["weight_bank"]["manifest_sha256"] == bank_hash,
                        "runner did not bind the expected immutable bank")
            gates.append(validate_run(envelope, golden, workload, architecture, atol, rtol, hbm_channels))
            repeat_results.append(envelope["result"])
            native_repeats.append(envelope["memory_model"].get("calibration"))
            library_hashes.append(envelope["provenance"].get("native_library_sha256"))
        require(
            all(result == repeat_results[0] for result in repeat_results[1:]),
            "repeat mismatch for " + architecture["name"],
        )
        require(all(n == native_repeats[0] for n in native_repeats), "native counters changed between repeats")
        require(len(set(library_hashes)) == 1, "native library changed between repeats")
        result = repeat_results[0]
        require(result["total_ps"] > 0, "nonempty comparison must advance time")
        return {
            "architecture": architecture,
            "result": result,
            "gates": gates,
            "native": native_repeats[0],
            "native_library_sha256": library_hashes[0],
        }

    # Each process owns an independent Ramulator/executor. Preserve manifest
    # order and propagate every exception before publishing any speedup.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(execute_architecture, range(len(architectures))))
    if bank_path is not None:
        require(digest(bank_path) == bank_hash, "weight bank catalog changed during comparison")
    for path, expected, label in (
        (workload_path, expected_hashes["workload_sha256"], "workload"),
        (golden_path, golden_hash, "golden"),
        (hbm_path, expected_hashes["hbm_sha256"], "HBM"),
        (binary, expected_hashes["executable_sha256"], "executable"),
    ):
        require(digest(path) == expected, label + " changed during comparison")
    for path, expected in zip(architecture_paths, architecture_hashes):
        require(digest(path) == expected, "architecture changed during comparison")
    require(len({r["native_library_sha256"] for r in results}) == 1, "architectures used different native libraries")
    baseline_ps = results[0]["result"]["total_ps"]
    for row in results:
        row["speedup_vs_baseline"] = baseline_ps / row["result"]["total_ps"]
    summary = {
        "schema_version": 1,
        "all_gates_passed": True,
        "status": "passed",
        "run_directory": str(run_dir),
        "scope": "Fixed-route numerical MoE operator; analytical core timing and shared Ramulator. "
        "Not RTL-calibrated or whole-model/request latency.",
        "source_metadata": workload.get("metadata", {}),
        "workload_sha256": expected_hashes["workload_sha256"],
        "golden_sha256": golden_hash,
        "hbm_sha256": expected_hashes["hbm_sha256"],
        "executable_sha256": expected_hashes["executable_sha256"],
        "repeats": repeats,
        "hbm_channels": hbm_channels,
        "numerical_tolerance": {"atol": atol, "rtol": rtol},
        "comparisons": results,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def run_comparison(
    binary,
    workload_path,
    golden_path,
    architecture_paths,
    output_dir,
    repeats=2,
    hbm_channels=8,
    atol=1e-5,
    rtol=0.01,
    timeout=180,
    workers=1,
    max_hbm_bytes=536_870_912,
):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "comparison.json"
    summary_path.write_text(
        json.dumps({"schema_version": 1, "status": "running", "all_gates_passed": False}) + "\n", encoding="utf-8"
    )
    try:
        return _run_comparison(
            binary,
            workload_path,
            golden_path,
            architecture_paths,
            output_dir,
            repeats,
            hbm_channels,
            atol,
            rtol,
            timeout,
            workers,
            max_hbm_bytes,
        )
    except Exception as error:
        summary_path.write_text(
            json.dumps({"schema_version": 1, "status": "failed", "all_gates_passed": False, "error": str(error)})
            + "\n",
            encoding="utf-8",
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--golden", required=True)
    parser.add_argument("--architecture", action="append", required=True, help="Baseline first, then candidate(s)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--hbm-channels", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    try:
        summary = run_comparison(
            args.binary,
            args.workload,
            args.golden,
            args.architecture,
            args.output_dir,
            args.repeats,
            args.hbm_channels,
            args.atol,
            args.rtol,
            args.timeout,
            args.workers,
        )
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    for row in summary["comparisons"]:
        print(
            "{}: {} ps, {:.4f}x, gates passed".format(
                row["architecture"]["name"], row["result"]["total_ps"], row["speedup_vs_baseline"]
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
