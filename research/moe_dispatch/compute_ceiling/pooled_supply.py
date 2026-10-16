"""Reverse requirements for completed cross-expert ready-pool evaluations.

The ready-record RR loop differs from the primary K-major full-projection
loop.  This module therefore does NOT reuse primary pool-to-local reuse
rates.  It derives invariant issue payloads, exact partial-sum traffic and
conservative resident capacities from frozen ownership/event outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .supply import BF16_BYTES, FP32_BYTES, _core_shape, _shape_traffic
from .trace_inputs import load_primary_inputs

BATCHES = (2, 4, 8, 16, 64, 96, 128)
FAMILIES = ("single", "homogeneous", "heterogeneous")
POOL_RATE_STATUS = "未定，需按RR loop供数验证"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gm(values: list[float]) -> float:
    if not values or any(value < 0 or not math.isfinite(value) for value in values):
        raise ValueError("GM needs nonnegative finite values")
    if any(value == 0 for value in values):
        return 0.0
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty pooled-supply table")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError("pooled-supply rows do not have a consistent schema")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _capacity_all_weights(window: dict[str, Any]) -> int:
    routed = [e for e in window["experts"] if not e.get("is_shared", False)]
    shared = [e for e in window["experts"] if e.get("is_shared", False)]
    dims = {(int(e["H"]), int(e["F"])) for e in routed}
    if len(dims) != 1 or len(shared) != 1:
        raise ValueError("this report requires frozen DS-V2-Lite layer dimensions")
    h, f = next(iter(dims))
    expert_count = int(window.get("expert_count", window.get("routed_expert_count", 64)))
    return expert_count * 3 * h * f * BF16_BYTES + sum(
        3 * int(e["H"]) * int(e["F"]) * BF16_BYTES for e in shared)


def _window_requirements(window: dict[str, Any], result: dict[str, Any],
                         cores: list[Any], family: str) -> dict[str, Any]:
    """Dimension/count arithmetic only; do not change or replay the schedule."""
    shapes = [_core_shape(core) for core in cores]
    cycles = int(result["cycles"])
    if cycles <= 0 or result["window_id"] != window["id"]:
        raise ValueError("positive cycles and matching window ID required")
    if int(result["batch"]) != int(window["batch"]):
        raise ValueError("pooled result batch disagrees with trace")
    expected_geometry = "+".join("x".join(str(value) for value in shape) for shape in shapes)
    if result["geometry"] != expected_geometry:
        raise ValueError("pooled result geometry disagrees with frozen shapes")
    active = {e["id"]: e for e in window["experts"] if int(e["Me"]) > 0}
    ownership = result["ownership"]
    if len(ownership) != len(active) or {o["expert_id"] for o in ownership} != set(active):
        raise ValueError("every active expert must have exactly one frozen owner")
    counters = [defaultdict(int) for _ in cores]
    for owner_record in ownership:
        e = active[owner_record["expert_id"]]
        owner = int(owner_record["owner"])
        if owner not in range(len(cores)):
            raise ValueError("invalid owner index")
        if int(owner_record["Me"]) != int(e["Me"]) or bool(owner_record["is_shared"]) != bool(e.get("is_shared", False)):
            raise ValueError("ownership metadata disagree with the real routing input")
        m, h, f = int(e["Me"]), int(e["H"]), int(e["F"])
        projections = [_shape_traffic(m, n, k, shapes[owner])
                       for n, k in ((f, h), (f, h), (h, f))]
        expert_issues = sum(p["issues"] for p in projections)
        if expert_issues != int(owner_record["issues"]):
            raise ValueError("frozen owner issue count does not match ceil tile arithmetic")
        counters[owner]["issues"] += expert_issues
        counters[owner]["useful_macs"] += 3 * m * h * f
        for projection in projections:
            for key, value in projection["array_operand_and_partial_traffic"].items():
                counters[owner][key] += value
    if [c["issues"] for c in counters] != result["core_issues"]:
        raise ValueError("per-core issue arithmetic disagrees with the event replay")
    if sum(c["issues"] for c in counters) != int(result["issues"]):
        raise ValueError("layer issue arithmetic disagrees with the event replay")
    if sum(c["useful_macs"] for c in counters) != int(result["useful_macs"]):
        raise ValueError("logical useful MAC arithmetic disagrees with the event replay")
    traffic = defaultdict(int)
    for c in counters:
        for key, value in c.items():
            traffic[key] += value
    weight_bytes = sum(3 * int(e["H"]) * int(e["F"]) * BF16_BYTES for e in active.values())
    gather_x = sum(int(e["Me"]) * int(e["H"]) * BF16_BYTES for e in active.values())
    all_z = sum(int(e["Me"]) * int(e["F"]) * BF16_BYTES for e in active.values())
    hidden = {int(e["H"]) for e in active.values()}
    if len(hidden) != 1:
        raise ValueError("all experts in this MoE layer must share H")
    h = hidden.pop()
    batch = int(window["batch"])
    global_x = batch * h * BF16_BYTES
    combine_y = batch * h * FP32_BYTES
    chosen_partial = [int(value) for value in result["peak_FP32_partial_bytes_per_core"]]
    rr_partial = [int(value) for value in result["ready_pool_peak_FP32_partial_bytes_per_core"]]
    sources = result["core_schedule_source"]
    if len(chosen_partial) != len(cores) or len(rr_partial) != len(cores) or len(sources) != len(cores):
        raise ValueError("missing per-core event capacity/source data")
    if any(value < 0 for value in chosen_partial + rr_partial):
        raise ValueError("partial payload requirements cannot be negative")
    # These are sufficient payload allocations, not a promise of enough
    # ports to feed an arbitrarily interleaved pool from two local slots.
    live_core = [index for index, c in enumerate(counters) if c["issues"]]
    w_double = sum(2 * shapes[i][1] * shapes[i][2] * BF16_BYTES for i in live_core)
    x_double = sum(2 * shapes[i][0] * shapes[i][2] * BF16_BYTES for i in live_core)
    full_model_weights = _capacity_all_weights(window)
    # All expert X and Z payloads are retained concurrently.  Gathered X
    # includes the Shared copy of the complete input, so adding global_X
    # again would double count that copy.  Aliasing could reduce this bound.
    common_bytes = gather_x + all_z + combine_y + sum(chosen_partial) + w_double + x_double
    row = {
        "family": family, "geometry_M_N_K": result["geometry"],
        "window_id": window["id"], "batch": batch,
        "compute_cycles": cycles, "compute_ms": cycles / 1_000_000,
        "ready_pool_cycles": int(result["ready_pool_cycles"]),
        "core_schedule_source_json": json.dumps(sources, separators=(",", ":")),
        "core_issues_json": json.dumps(result["core_issues"], separators=(",", ":")),
        "issues": int(result["issues"]),
        "selected_unique_bf16_weight_bytes": weight_bytes,
        "selected_unique_bf16_weight_MiB": weight_bytes / 2**20,
        "cold_external_unique_weight_necessary_average_GB_s": weight_bytes / cycles,
        "resident_weight_HBM_bytes_per_replay_after_prewarm": 0,
        "pool_to_local_weight_refill_GB_s": POOL_RATE_STATUS,
        "pool_to_local_X_refill_GB_s": POOL_RATE_STATUS,
        "array_weight_full_slice_bytes": traffic["weight_full_slice_bytes"],
        "array_weight_masked_bytes": traffic["weight_masked_bytes"],
        "array_X_full_slice_bytes": traffic["x_full_slice_bytes"],
        "array_X_masked_bytes": traffic["x_masked_bytes"],
        "FP32_partial_read_full_record_bytes": traffic["partial_read_full_record_bytes"],
        "FP32_partial_write_full_record_bytes": traffic["partial_write_full_record_bytes"],
        "FP32_partial_read_masked_bytes": traffic["partial_read_masked_bytes"],
        "FP32_partial_write_masked_bytes": traffic["partial_write_masked_bytes"],
        "array_weight_full_slice_average_GB_s": traffic["weight_full_slice_bytes"] / cycles,
        "array_X_full_slice_average_GB_s": traffic["x_full_slice_bytes"] / cycles,
        "FP32_partial_read_full_record_average_GB_s": traffic["partial_read_full_record_bytes"] / cycles,
        "FP32_partial_write_full_record_average_GB_s": traffic["partial_write_full_record_bytes"] / cycles,
        "array_weight_full_slice_burst_envelope_GB_s": sum(shapes[i][1] * shapes[i][2] * BF16_BYTES for i in live_core),
        "array_X_full_slice_burst_envelope_GB_s": sum(shapes[i][0] * shapes[i][2] * BF16_BYTES for i in live_core),
        "FP32_partial_read_full_record_burst_envelope_GB_s": sum(shapes[i][0] * shapes[i][1] * FP32_BYTES for i in live_core),
        "FP32_partial_write_full_record_burst_envelope_GB_s": sum(shapes[i][0] * shapes[i][1] * FP32_BYTES for i in live_core),
        "whole_model_layer_weights_bytes": full_model_weights,
        "whole_model_layer_weights_MiB": full_model_weights / 2**20,
        "chosen_per_core_partial_peak_bytes_json": json.dumps(chosen_partial, separators=(",", ":")),
        "chosen_sum_core_partial_peaks_bytes": sum(chosen_partial),
        "chosen_sum_core_partial_peaks_KiB": sum(chosen_partial) / 1024,
        "RR_only_per_core_partial_peak_bytes_json": json.dumps(rr_partial, separators=(",", ":")),
        "RR_only_sum_core_partial_peaks_bytes": sum(rr_partial),
        "RR_only_sum_core_partial_peaks_KiB": sum(rr_partial) / 1024,
        "global_input_unique_X_bf16_bytes": global_x,
        "all_expert_gathered_X_bf16_resident_bytes_conservative": gather_x,
        "all_expert_Z_bf16_resident_bytes_conservative": all_z,
        "combined_Y_fp32_bytes": combine_y,
        "sum_local_weight_double_buffer_bytes": w_double,
        "sum_local_X_tile_double_buffer_bytes": x_double,
        "fully_resident_selected_weight_operand_and_context_bytes_conservative": weight_bytes + common_bytes,
        "fully_resident_all_model_weight_operand_and_context_bytes_conservative": full_model_weights + common_bytes,
        "installed_record_descriptor_count": sum(result.get("installed_record_descriptors_per_core", [])),
        "descriptor_metadata_bytes_counted": False,
        "frequency_hz": 1_000_000_000, "II_cycles": 1,
        "native_calibrated": False,
    }
    return row


def _aggregate(rows: list[dict[str, Any]], families: tuple[str, ...] = FAMILIES) -> list[dict[str, Any]]:
    rate_fields = (
        "compute_ms", "cold_external_unique_weight_necessary_average_GB_s",
        "array_weight_full_slice_average_GB_s", "array_X_full_slice_average_GB_s",
        "FP32_partial_read_full_record_average_GB_s", "FP32_partial_write_full_record_average_GB_s",
    )
    peak_fields = (
        "chosen_sum_core_partial_peaks_bytes", "RR_only_sum_core_partial_peaks_bytes",
        "all_expert_gathered_X_bf16_resident_bytes_conservative",
        "all_expert_Z_bf16_resident_bytes_conservative", "combined_Y_fp32_bytes",
        "sum_local_weight_double_buffer_bytes", "sum_local_X_tile_double_buffer_bytes",
        "fully_resident_selected_weight_operand_and_context_bytes_conservative",
        "fully_resident_all_model_weight_operand_and_context_bytes_conservative",
        "installed_record_descriptor_count",
        "array_weight_full_slice_burst_envelope_GB_s", "array_X_full_slice_burst_envelope_GB_s",
        "FP32_partial_read_full_record_burst_envelope_GB_s", "FP32_partial_write_full_record_burst_envelope_GB_s",
    )
    out = []
    for family in families:
        family_rows = [row for row in rows if row["family"] == family]
        for batch in (*BATCHES, "all"):
            selected = [row for row in family_rows if batch == "all" or row["batch"] == batch]
            group = defaultdict(list)
            for row in selected:
                group[row["batch"]].append(row)
            out_row = {"family": family, "geometry_M_N_K": selected[0]["geometry_M_N_K"],
                       "batch": batch, "windows": len(selected)}
            for field in rate_fields:
                # Equal batch log weighting matches the compute experiment.
                per_batch = [_gm([float(row[field]) for row in rr]) for rr in group.values()]
                out_row[f"{field}_GM"] = _gm(per_batch)
                out_row[f"{field}_max_window"] = max(row[field] for row in selected)
            out_row["selected_unique_bf16_weight_MiB_AM"] = math.fsum(
                math.fsum(row["selected_unique_bf16_weight_MiB"] for row in rr) / len(rr)
                for rr in group.values()) / len(group)
            out_row["whole_model_layer_weights_MiB"] = selected[0]["whole_model_layer_weights_MiB"]
            for field in peak_fields:
                out_row[f"{field}_max_window"] = max(row[field] for row in selected)
            out_row["pool_to_local_weight_refill_GB_s"] = POOL_RATE_STATUS
            out_row["pool_to_local_X_refill_GB_s"] = POOL_RATE_STATUS
            out_row["native_calibrated"] = False
            out.append(out_row)
    return out


def generate_pooled_supply(out: str | Path) -> dict[str, Any]:
    """Only read completed pooled results; never mutate frozen source files."""
    out = Path(out)
    completion_path = out / "COMPLETION.json"
    if not completion_path.exists():
        raise ValueError("pooled COMPLETION.json is required before reverse supply reporting")
    completion = json.loads(completion_path.read_text())
    if completion.get("heldout_windows") != 135:
        raise ValueError("expected all 135 primary heldout windows")
    frozen_path = out / "FROZEN_SELECTION.json"
    frozen = json.loads(frozen_path.read_text())
    source_paths = [completion_path, frozen_path, out / "CONTRACT.json"]
    configs = dict(frozen["configs"])
    families = FAMILIES
    # The original physical single receives the same interleaving capability
    # in a separately frozen reference replay.  Do not rewrite the search's
    # three-family development selection to insert this fixed baseline.
    reference_config = out / "ORIGINAL_REFERENCE.json"
    reference_completion = out / "COMPLETION_REFERENCE.json"
    if reference_config.exists() or reference_completion.exists():
        if not reference_config.exists() or not reference_completion.exists():
            raise ValueError("original reference must finish before merged pooled supply reporting")
        reference = json.loads(reference_config.read_text())
        configs["original_single"] = reference["configs"]["original_single"]
        families = ("original_single", *FAMILIES)
        source_paths.extend((reference_config, reference_completion))
    source_hashes = {str(path): _hash(path) for path in source_paths}
    inputs = load_primary_inputs(split="heldout")
    windows = {w["id"]: w for w in inputs["heldout"]}
    if len(windows) != 135:
        raise ValueError("primary frozen trace does not have 135 distinct heldout windows")
    rows = []
    for family in families:
        cores = configs[family]
        path = out / f"heldout_{family}.json"
        source_paths.append(path)
        source_hashes[str(path)] = _hash(path)
        source = json.loads(path.read_text())
        if len(source) != 135 or len({r["window_id"] for r in source}) != 135:
            raise ValueError("each family must contain 135 distinct completed windows")
        if {r["window_id"] for r in source} != set(windows):
            raise ValueError("pooled and routing window IDs do not match")
        rows.extend(_window_requirements(windows[result["window_id"]], result, cores, family)
                    for result in source)
    if len(rows) != 135 * len(families):
        raise ValueError("expected 135 windows for each frozen family/reference")
    if any(row["whole_model_layer_weights_MiB"] != 1089 for row in rows):
        raise ValueError("DS-V2-Lite all64+Shared weights should be 1089 MiB BF16")
    summary = _aggregate(rows, families)
    if len(summary) != 8 * len(families):
        raise ValueError("expected 7 batches + all, for each of 3 families")
    per_window_path = out / "reverse_supply_per_window.csv"
    by_batch_path = out / "reverse_supply_by_batch.csv"
    _csv(per_window_path, rows)
    _csv(by_batch_path, summary)
    if any(_hash(Path(path)) != digest for path, digest in source_hashes.items()):
        raise AssertionError("a frozen pooled source changed while generating supply requirements")
    receipt = {
        "status": "PASS", "scope": "reverse requirements from completed RR/fallback event schedules; no memory timing simulation",
        "window_rows": len(rows), "by_batch_rows": len(summary),
        "families": list(families), "assumed_clock_GHz": 1, "II_cycles": 1,
        "original_reference_included": "original_single" in families,
        "native_calibrated": False,
        "source_sha256": source_hashes, "sources_unchanged": True,
        "routing_manifest": inputs["manifest"],
        "source_formulas": {
            "unique_selected_weight_bytes": "sum active experts 3*H*F*2",
            "external_required_average_GB_s": "unique selected BF16 weight bytes / chosen compute cycles, at 1GHz",
            "array_W_full_bytes": "sum cores core_issues*PN*PK*2",
            "array_X_full_bytes": "sum cores core_issues*PM*PK*2",
            "partial_write_full_bytes": "sum projections nm*nn*nk*PM*PN*4",
            "partial_read_full_bytes": "sum projections nm*nn*(nk-1)*PM*PN*4; no old read for first K",
            "gathered_X_resident_bytes": "sum active experts Me*H*2; conservative per-expert copies, Shared includes the complete global X",
            "Z_resident_bytes": "sum active experts Me*F*2",
            "combined_output_bytes": "B*H*4",
            "local_W_double_buffer_bytes": "sum active cores 2*PN*PK*2",
            "local_X_double_buffer_bytes": "sum active cores 2*PM*PK*2",
            "context_partial_capacity": "sum event-measured per-core peaks for the chosen RR/fallback policy; conservative simultaneous allocation",
            "all_model_weights": "64*(3*2048*1408*2) + 3*2048*2816*2 =1089MiB",
            "all_batch_GM": "GM of per-batch GMs, matching equal-batch log weighting",
        },
        "limitations": [
            "Cold loading average is a necessary rate if all selected weights must arrive within the compute window; it cannot certify issue deadlines or startup overlap.",
            "After full persistent weight prewarm, weight HBM bytes per replay can be zero; initial loading is outside that replay's timing.",
            "Pool-to-local W/X reuse rates are intentionally undetermined until the RR operand-fetch/buffer loop is specified and verified.",
            "Array per-issue delivery can be served by latches/registers; it is not automatically an HBM or shared-SRAM read rate.",
            "Burst envelope sums active cores' full payloads at II1, not necessarily their observed simultaneous issue peaks.",
            "Context payload capacity is required, but metadata/ready queues/tags, alignment, banking and pipeline internal state are not counted.",
            "All X/Z copies are a conservative resident allocation, not a proven minimal SRAM footprint; aliasing/in-place release may reduce it.",
            "SiLU, gate weighting and combine traffic/service remain outside compute objective; their ports and timing are not certified here.",
            "Neither RR scheduling nor its source issue-assignment is a proof of globally optimal pooled instruction scheduling.",
        ],
        "outputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": _hash(path)}
                    for path in (per_window_path, by_batch_path)],
        "per_window": rows, "by_batch": summary,
    }
    (out / "REVERSE_SUPPLY.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = generate_pooled_supply(args.out)
    print(json.dumps({key: receipt[key] for key in ("status", "window_rows", "by_batch_rows", "sources_unchanged", "outputs")},
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
