"""Reverse OS supply ledger for a declared ideal-compute expert schedule.

The three byte boundaries are deliberately separate:

* cold unique external weights (each active expert matrix counted once);
* shared resident pool -> local operand buffers (OS loop refills);
* local buffers -> array broadcast ingress (before replicated PE fanout).

The last two have equal operand *counts* under the inherited OS m,n,k loop,
but they are separate interfaces, not additive HBM traffic.  This contract
does not assume free fused Gate/Up X reuse.  Full-tile counts charge padding;
masked counts assume free valid-element transfers and are lower bounds.

Exact scheduled peaks below refer ONLY to padded array-ingress consumption.
They follow the same fused-Gate/Up feed/drain timeline as the compute model.
A shared-pool refill peak is not determined by a compute schedule: it also
depends on prefetch distance, initial occupancy, transaction alignment,
banks and latency.  We report a clearly named continuous-refill provisioning
upper bound, NOT a measured SRAM peak or a proof of stall-free execution.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from research.moe_dispatch.plena65536_dedicated.geometry import Core
from research.moe_dispatch.plena65536_moe.dataflow import plena_mixed_traffic
from research.moe_dispatch.plena65536_shared.timing import PROFILES, Profile, phase

MIB = 1024**2
KIB = 1024
FREQUENCY_HZ = 1_000_000_000


def _core(value: Any) -> Core:
    if isinstance(value, Core):
        return value
    if isinstance(value, Mapping):
        names = ("b", "r", "rm", "rn") if "r" in value else ("b", "R", "rM", "rN")
        return Core(*(int(value[key]) for key in names))
    return Core(value.b, value.r, value.rm, value.rn)


def _profile(value: Profile | str) -> Profile:
    if isinstance(value, str):
        choices = {item.name: item for item in PROFILES}
        if value not in choices:
            raise ValueError(f"unknown timing profile: {value}")
        return choices[value]
    return value


def _cycle(job: Mapping[str, Any], names: Sequence[str]) -> int:
    for key in names:
        if key in job:
            value = job[key]
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer cycle")
            return int(value)
    raise ValueError(f"job requires a cycle endpoint from {names}")


def _phase_feed_intervals(core: Core, start: int, m: int, n: int, k: int,
                          factor: int, profile: Profile):
    """Half-open full-width feed intervals; no event for drain cycles."""
    records = factor * ((m + core.pm - 1) // core.pm) * ((n + core.pn - 1) // core.pn)
    feed = ((k + core.r - 1) // core.r) * core.b
    if profile.serialized:
        period = feed + profile.drain(core)
        for tile in range(records):
            yield start + tile * period, start + tile * period + feed
    elif records:
        yield start, start + records * feed


def array_feed_peak(cores: Sequence[Any], window: Mapping[str, Any],
                    plan: Mapping[str, Any], profile: Profile | str = PROFILES[0]
                    ) -> dict[str, Any]:
    """Exact padded array broadcast-ingress peak for whole-expert schedules.

    A b x b mini-array consumes b temporal steps per logical R=b*G segment.
    Per feeding cycle each core therefore accepts 2*G*PN weight bytes and
    2*G*PM activation bytes, not 2*R*(PM+PN) bytes.  Operand broadcasts to
    the replicated output lanes happen after this counted interface.
    """
    cores, profile = tuple(_core(c) for c in cores), _profile(profile)
    if not cores:
        raise ValueError("at least one core is required")
    active = {int(e["id"]): e for e in window["experts"] if int(e["Me"]) > 0}
    jobs = list(plan["jobs"])
    if sorted(int(j["expert_id"]) for j in jobs) != sorted(active):
        raise ValueError("schedule must execute every active expert exactly once")
    events: dict[int, list[int]] = {}
    feed_cycles = [0] * len(cores)
    full_weight = full_x = 0
    end_max = 0
    occupancy: list[list[tuple[int, int]]] = [[] for _ in cores]

    def add(time: int, weight: int, x: int) -> None:
        delta = events.setdefault(time, [0, 0])
        delta[0] += weight
        delta[1] += x

    for job in jobs:
        if job.get("partitions") is not None:
            raise ValueError("this phase requires whole-expert jobs, not partitioned experts")
        owner = int(job["core_id"])
        if not 0 <= owner < len(cores):
            raise ValueError("job references a nonexistent core")
        core = cores[owner]
        expert = active[int(job["expert_id"])]
        m, h, f = (int(expert[key]) for key in ("Me", "H", "F"))
        start = _cycle(job, ("start", "start_cycles", "start_cycle"))
        end = _cycle(job, ("end", "finish_cycles", "end_cycles", "finish", "end_cycle"))
        gu = phase(core, m, f, h, 2, profile)
        down = phase(core, m, h, f, 1, profile)
        if end - start != gu + down:
            raise ValueError("job duration disagrees with declared OS compute timing")
        occupancy[owner].append((start, end))
        end_max = max(end_max, end)
        w_rate, x_rate = 2 * core.groups * core.pn, 2 * core.groups * core.pm
        for offset, n, k, factor in ((start, f, h, 2), (start + gu, h, f, 1)):
            for begin, finish in _phase_feed_intervals(core, offset, m, n, k, factor, profile):
                add(begin, w_rate, x_rate)
                add(finish, -w_rate, -x_rate)
                feed_cycles[owner] += finish - begin
                full_weight += (finish - begin) * w_rate
                full_x += (finish - begin) * x_rate
    for intervals in occupancy:
        intervals.sort()
        if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])):
            raise ValueError("expert jobs overlap on one core")
    duration = int(plan.get("cycles", end_max))
    if duration <= 0 or duration != end_max:
        raise ValueError("plan cycles must equal the final job completion cycle")
    weight = x = peak_total = peak_w = peak_x = peak_cycle = 0
    for time, (dw, dx) in sorted(events.items()):
        weight += dw
        x += dx
        if weight < 0 or x < 0:
            raise AssertionError("feed interval event accounting is inconsistent")
        peak_w, peak_x = max(peak_w, weight), max(peak_x, x)
        if weight + x > peak_total:
            peak_total, peak_cycle = weight + x, time
    if weight or x:
        raise AssertionError("feed intervals must close")
    provision = sum(2 * c.groups * (c.pm + c.pn) for c in cores)
    return {
        "cycles": duration,
        "per_core_feed_cycles": feed_cycles,
        "array_full_weight_bytes": full_weight,
        "array_full_x_bytes": full_x,
        "array_full_operand_bytes": full_weight + full_x,
        "array_full_scheduled_peak_bytes_per_cycle": peak_total,
        "array_full_weight_scheduled_peak_bytes_per_cycle": peak_w,
        "array_full_x_scheduled_peak_bytes_per_cycle": peak_x,
        "array_full_scheduled_peak_first_cycle": peak_cycle,
        "array_full_scheduled_peak_TB_s": peak_total / 1000,
        "array_full_provision_peak_TB_s": provision / 1000,
        "peak_exact_for_padded_ingress_schedule": True,
        "peak_valid_element_only": False,
        "PE_fanout_traffic_included_in_ingress": False,
        "feed_interval_event_count": len(events),
    }


def supply_ledger(cores: Sequence[Any], window: Mapping[str, Any],
                  plan: Mapping[str, Any], profile: Profile | str = PROFILES[0]
                  ) -> dict[str, Any]:
    """Flat report fields plus explicit assumptions; all rates at 1 GHz.

    Full-layer weight capacity is provided for the fixed DeepSeek-V2-Lite
    model only: 64 routed F=1408 experts and one merged F=2816 Shared expert.
    The resident-payload envelope also charges intermediates, expanded PE
    contexts, reduction-tree registers and double buffers.  It is not a
    demonstrated minimal live set and excludes SRAM implementation overhead.
    """
    cores = tuple(_core(c) for c in cores)
    peak = array_feed_peak(cores, window, plan, profile)
    traffic = plena_mixed_traffic(window, cores, "OS", plan)
    pool, array = traffic["pool_to_local"], traffic["array_delivery"]
    if peak["array_full_weight_bytes"] != array["weight_full_bytes"]:
        raise AssertionError("padded weight feed intervals do not conserve traffic bytes")
    if peak["array_full_x_bytes"] != array["x_full_bytes"]:
        raise AssertionError("padded activation feed intervals do not conserve traffic bytes")
    for suffix in ("masked_bytes", "full_bytes"):
        for kind in ("weight", "x"):
            if pool[f"{kind}_{suffix}"] != array[f"{kind}_{suffix}"]:
                raise AssertionError("OS refills and broadcast ingress must conserve operands")
    cycles = peak["cycles"]
    rate = lambda byte_count: byte_count / cycles / 1000
    context = pool["partial_read_masked_bytes"] + pool["partial_write_masked_bytes"]
    if context:
        raise AssertionError("OS must retain expanded PE partials across K segments")
    operands = pool["weight_masked_bytes"] + pool["x_masked_bytes"]
    full_operands = pool["weight_full_bytes"] + pool["x_full_bytes"]
    output = pool["final_output_masked_bytes"]
    post = sum(traffic["postprocess"][key] for key in (
        "GU_read_fp32_bytes", "Z_write_bf16_bytes", "expert_Y_combine_read_fp32_bytes",
        "combined_Y_write_fp32_bytes"))
    active = [e for e in window["experts"] if int(e["Me"]) > 0]
    deepseek = (int(window.get("hidden", 2048)) == 2048 and
                all(int(e["H"]) == 2048 and int(e["F"]) ==
                    (2816 if e.get("is_shared", int(e["id"]) == -1) else 1408)
                    for e in active))
    full_layer = (3 * 2048 * (2816 + 64 * 1408) * 2) if deepseek else None
    cap, resource = traffic["capacity"], traffic["resource_ledger"]
    return {
        **peak,
        "window_id": window.get("id", ""), "batch": int(window["batch"]),
        "compute_ms": cycles / 1e6,
        "unique_weight_bytes": cap["active_unique_weights_bytes"],
        "unique_weight_MiB": cap["active_unique_weights_bytes"] / MIB,
        "full_layer_weight_MiB": full_layer / MIB if full_layer is not None else None,
        "unique_weight_average_TB_s": rate(cap["active_unique_weights_bytes"]),
        "unique_weight_fetch_ms_at256": cap["active_unique_weights_bytes"] / 256e6,
        "pool_weight_bytes": pool["weight_masked_bytes"],
        "pool_x_bytes": pool["x_masked_bytes"],
        "pool_context_bytes": context,
        "pool_operand_bytes": operands,
        "pool_operand_full_bytes": full_operands,
        "pool_output_bytes": output,
        "pool_postprocess_bytes": post,
        "pool_weight_average_TB_s": rate(pool["weight_masked_bytes"]),
        "pool_x_average_TB_s": rate(pool["x_masked_bytes"]),
        "pool_context_average_TB_s": rate(context),
        "pool_operand_average_TB_s": rate(operands),
        "pool_operand_full_average_TB_s": rate(full_operands),
        "pool_output_average_TB_s": rate(output),
        "pool_postprocess_average_TB_s": rate(post),
        "pool_all_average_TB_s": rate(operands + context + output + post),
        "array_operand_average_TB_s": rate(operands),
        "array_operand_full_average_TB_s": rate(full_operands),
        "pool_minimum_required_peak_TB_s": None,
        "pool_continuous_refill_provision_upper_TB_s": peak["array_full_provision_peak_TB_s"],
        "local_double_buffer_KiB": resource["full_operand_double_buffer_bytes"] / KIB,
        "static_local_storage_KiB": resource["static_storage_bytes"] / KIB,
        "resident_payload_upper_MiB": cap["unified_resident_payload_upper_bytes"] / MIB,
        "resident_pool_payload_upper_MiB": (cap["active_unique_weights_bytes"] +
                                             cap["activation_liveness_upper_bytes"]) / MIB,
        "physical_multipliers": resource["multipliers"],
        "model": "OS whole-expert loop; ideal supply; BF16 operands; FP32 partials; 1GHz",
        "pool_peak_contract": "Provision upper bound: both cores continuously refill one padded R segment "
                              "per b cycles into independent ping-pong operand buffers; first segment "
                              "preloaded. Assumes zero transport latency and unrestricted banks. "
                              "Does not cover output/postprocess port demand or prove finite-latency feasibility.",
        "average_contract": "Masked W+X bytes divided by complete layer compute makespan; necessary average "
                             "service only. Padding, transaction granularity, bank conflicts and refill deadlines "
                             "can raise the physical requirement.",
        "array_peak_contract": "Exact simultaneous full-width array broadcast-ingress feed cycles for the "
                                "specified analytical schedule, before replicated PE fanout; padding included.",
        "end_to_end_validated": False,
    }


# Readable window-first alias for callers outside the report module.
def os_supply_ledger(window: Mapping[str, Any], cores: Sequence[Any],
                     plan: Mapping[str, Any], profile: Profile | str = PROFILES[0]):
    return supply_ledger(cores, window, plan, profile)
