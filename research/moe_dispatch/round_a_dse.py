#!/usr/bin/env python3
"""Prospective equal-MAC MoE geometry screening; not the frozen v3 experiment.

The PLENA array has spatial K=512 dot products, not a 2-D systolic K loop.
The default model preserves II=1, dot latency=20 and one-cycle result commit.
HBM, operand ports, dispatch service and physical bank timing are deliberately
absent. New N widths are hypothetical hardware, not supported v3 datapaths.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Any

TOTAL_MACS = 12_288
PK = 512
TOTAL_UNITS = TOTAL_MACS // PK
SRAM_BUDGET = 2_158_592
THRESHOLDS = (1, 2, 3, 4, 6, 8, 12, 16)
DEFAULT_INPUT = Path(__file__).resolve().parent / "outputs"  # CLI requires an explicit source


def ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical(value))


@dataclass(frozen=True, order=True)
class Core:
    pm: int
    pn: int
    pk: int = PK

    @property
    def macs(self) -> int:
        return self.pm * self.pn * self.pk


@dataclass(frozen=True)
class Model:
    dot_latency: int = 20
    initiation_interval: int = 1
    commit_cycles: int = 1
    n_group_tiles: int = 8
    vector_elements_per_cycle: int = 64

    @property
    def completion_latency(self) -> int:
        return self.dot_latency + self.commit_cycles


def geometry_id(cores: tuple[Core, ...]) -> str:
    return "+".join(f"{c.pm}x{c.pn}x{c.pk}" for c in cores)


def enumerate_geometries() -> list[tuple[Core, ...]]:
    """All integer factorizations in declared PM[1,8], PN[1,24], PK=512.

    Mirrors are canonicalized; 2+4 and 4+2 are the same physical design.
    PM<=8 is a bounded design hypothesis, not an exhaustive NPU search.
    """
    shapes = [Core(m, n) for m in range(1, 9) for n in range(1, 25)
              if m * n <= TOTAL_UNITS]
    singles = [(c,) for c in shapes if c.macs == TOTAL_MACS]
    pairs = [(a, b) for i, a in enumerate(shapes) for b in shapes[i:]
             if a.macs + b.macs == TOTAL_MACS]
    return singles + pairs


def family(cores: tuple[Core, ...]) -> str:
    if len(cores) == 1:
        return "single"
    return "homogeneous" if cores[0] == cores[1] else "heterogeneous"


@lru_cache(maxsize=None)
def projection(m: int, n: int, k: int, core: Core, model: Model) -> dict[str, int]:
    """Exact timing of the declared bounded output-group issue sequence.

    Within each N group, K segments traverse all M/N records in the same
    order. A record's next K segment cannot issue before its previous result
    commits. Independent records overlap. Groups retire before replacement.
    For q records, s K segments, the final commit is
      (s-1)*max(q*II, dot+commit) + (q-1)*II + dot+commit.
    This is not waves multiplied by pipeline depth.
    """
    if min(m, n, k, core.pm, core.pn, core.pk) <= 0:
        raise ValueError("positive GEMM and array dimensions required")
    nm, nn, nk = ceil(m, core.pm), ceil(n, core.pn), ceil(k, core.pk)
    cycles = 0
    for first in range(0, nn, model.n_group_tiles):
        q = nm * min(model.n_group_tiles, nn - first)
        cycles += ((nk - 1) * max(q * model.initiation_interval, model.completion_latency)
                   + (q - 1) * model.initiation_interval + model.completion_latency)
    issues = nm * nn * nk
    issued = issues * core.macs
    useful = m * n * k
    return {"cycles": cycles, "issues": issues, "useful_macs": useful,
            "issued_macs": issued, "padding_macs": issued - useful,
            "m_waves": nm, "n_tiles": nn, "k_segments": nk}


@lru_cache(maxsize=None)
def expert_cost(m: int, h: int, f: int, core: Core, model: Model) -> dict[str, int]:
    # Gate and Up use concatenated W=[W_gate|W_up], a valid same-X GEMM.
    # Width boundaries are packed identically for every architecture.
    gu = projection(m, 2 * f, h, core, model)
    down = projection(m, h, f, core, model)
    silu = ceil(m * f, model.vector_elements_per_cycle)
    combine = ceil(m * h, model.vector_elements_per_cycle)
    return {"gate_up_cycles": gu["cycles"], "down_cycles": down["cycles"],
            "silu_cycles": silu, "combine_cycles": combine,
            "private_estimate_cycles": gu["cycles"] + down["cycles"] + silu + combine,
            "useful_macs": gu["useful_macs"] + down["useful_macs"],
            "issued_macs": gu["issued_macs"] + down["issued_macs"],
            "padding_macs": gu["padding_macs"] + down["padding_macs"],
            "main_issues": gu["issues"] + down["issues"]}


def eligible_cores(m: int, cores: tuple[Core, ...], policy: str) -> list[int]:
    if policy == "eft" or len(cores) == 1 or cores[0] == cores[1]:
        return list(range(len(cores)))
    threshold = int(policy.removeprefix("threshold_"))
    # "Stream" means narrower token axis; does not mean fewer total MACs.
    stream = min(range(len(cores)), key=lambda c: (cores[c].pm, cores[c].macs, c))
    dense = 1 - stream
    return [stream if m <= threshold else dense]


def assign_tasks(workload: dict, cores: tuple[Core, ...], policy: str,
                 model: Model) -> tuple[list[list[int]], list[dict]]:
    available = [0] * len(cores)
    queues: list[list[int]] = [[] for _ in cores]
    records = []
    for i, e in enumerate(workload["experts"]):
        candidates = eligible_cores(e["Me"], cores, policy)
        c = min(candidates, key=lambda c: (
            available[c] + expert_cost(e["Me"], e["H"], e["F"], cores[c], model)["private_estimate_cycles"],
            c))
        cost = expert_cost(e["Me"], e["H"], e["F"], cores[c], model)
        start = available[c]
        available[c] += cost["private_estimate_cycles"]
        queues[c].append(i)
        records.append({"expert_id": e["id"], "is_shared": e["is_shared"],
                        "Me": e["Me"], "core": c, "legal_core_count": len(candidates),
                        "estimated_queue_start": start, "estimated_finish": available[c]})
    return queues, records


def simulate_layer(workload: dict, cores: tuple[Core, ...], policy: str = "eft",
                   model: Model = Model(), detail: bool = False) -> dict:
    """Whole-expert FIFO queues plus one shared vector engine event model.

    Each core executes Gate/Up -> shared SiLU -> Down -> shared combine, then
    its next whole expert. No row co-packing, expert splitting or privileged
    stealing. Tasks all become known at post-Router time zero. EFT uses a
    private-cost estimate; actual SiLU/combiner contention is scheduled below.
    """
    assert sum(c.macs for c in cores) == TOTAL_MACS
    queues, bindings = assign_tasks(workload, cores, policy, model)
    pending: list[tuple[int, int, int, int]] = []
    next_in_queue = [0] * len(cores)
    core_finish = [0] * len(cores)
    vector_free = 0
    useful = issued = issues = 0
    completion = []

    def begin(c: int, start: int) -> None:
        nonlocal useful, issued, issues
        p = next_in_queue[c]
        if p == len(queues[c]):
            core_finish[c] = start
            return
        i = queues[c][p]
        next_in_queue[c] += 1
        e = workload["experts"][i]
        cost = expert_cost(e["Me"], e["H"], e["F"], cores[c], model)
        useful += cost["useful_macs"]
        issued += cost["issued_macs"]
        issues += cost["main_issues"]
        bindings[i]["actual_start"] = start
        heapq.heappush(pending, (start + cost["gate_up_cycles"], 0, c, i))

    for c in range(len(cores)):
        begin(c, 0)
    while pending:
        now, phase, c, i = heapq.heappop(pending)
        e = workload["experts"][i]
        cost = expert_cost(e["Me"], e["H"], e["F"], cores[c], model)
        if phase == 0:
            start = max(now, vector_free)
            vector_free = start + cost["silu_cycles"]
            heapq.heappush(pending, (vector_free + cost["down_cycles"], 1, c, i))
        elif phase == 1:
            start = max(now, vector_free)
            vector_free = start + cost["combine_cycles"]
            heapq.heappush(pending, (vector_free, 2, c, i))
        else:
            bindings[i]["actual_finish"] = now
            completion.append({"expert_id": e["id"], "core": c, "finish_cycle": now})
            begin(c, now)
    cycles = max(core_finish)
    expected = sum(3 * e["Me"] * e["H"] * e["F"] for e in workload["experts"])
    assert useful == expected
    assert len(completion) == len(workload["experts"])
    assert useful <= issued <= TOTAL_MACS * cycles
    out = {"workload": workload["id"], "batch": workload["batch"],
           "geometry": geometry_id(cores), "family": family(cores), "policy": policy,
           "cycles": cycles, "latency_ms_at_1ghz": cycles / 1_000_000,
           "useful_macs": useful, "issued_macs": issued, "padding_macs": issued - useful,
           "main_issues": issues, "spatial_utilization": useful / issued,
           "wall_mac_utilization": useful / (TOTAL_MACS * cycles),
           "core_finish_cycles": core_finish,
           "core_idle_tail_cycles": [cycles - t for t in core_finish]}
    if detail:
        out.update(bindings=bindings, completion_order=completion)
    return out


def verify_trace(workloads: list[dict]) -> dict:
    samples, useful = set(), 0
    for w in workloads:
        provenance = w.get("provenance")
        captured = provenance == "captured_decode" or (
            isinstance(provenance, dict) and provenance.get("origin") == "captured_mixed")
        assert captured, "Round A final inputs must be real captured routes, not constructed/Zipf workloads"
        tokens = w["tokens"]
        assert len(tokens) == w["batch"]
        ids = [t["token_index"] for t in tokens]
        assert ids == list(range(w["batch"]))
        routed = {}
        for t in tokens:
            samples.add(t["sample_id"])
            assert len(t["routes"]) == w["top_k"]
            assert len({r["expert_id"] for r in t["routes"]}) == w["top_k"]
            for route in t["routes"]:
                routed.setdefault(route["expert_id"], []).append(t["token_index"])
        seen_routed = {}
        for e in w["experts"]:
            assert e["H"] == w["hidden"] and e["Me"] == len(e["token_indices"])
            assert len(set(e["token_indices"])) == e["Me"]
            if e["is_shared"]:
                assert e["token_indices"] == ids
            else:
                assert sorted(e["token_indices"]) == routed[e["id"]]
                seen_routed[e["id"]] = sorted(e["token_indices"])
            useful += 3 * e["Me"] * e["H"] * e["F"]
        assert seen_routed == routed
        assert sum(e["Me"] for e in w["experts"] if not e["is_shared"]) == w["batch"] * w["top_k"]
    return {"workloads": len(workloads), "distinct_request_identifiers": len(samples),
            "token_id_and_expert_population_coherent": True, "useful_macs": useful,
            "real_capture_routes_only": True,
            "independence_warning": "decode windows/prefixes/layers are correlated; workload count is not independent model executions"}


def p95(values: list[int]) -> int:
    return sorted(values)[math.ceil(.95 * len(values)) - 1]


def aggregate(rows: list[dict]) -> dict:
    useful = sum(r["useful_macs"] for r in rows)
    issued = sum(r["issued_macs"] for r in rows)
    cycles = sum(r["cycles"] for r in rows)
    return {"layers": len(rows), "total_compute_layer_cycles": cycles,
            "total_compute_layer_ms_at_1ghz": cycles / 1_000_000,
            "mean_layer_cycles": cycles / len(rows),
            "p95_layer_cycles": p95([r["cycles"] for r in rows]),
            "p95_layer_ms_at_1ghz": p95([r["cycles"] for r in rows]) / 1_000_000,
            "useful_macs": useful, "issued_macs": issued,
            "padding_macs": issued - useful,
            "spatial_utilization": useful / issued,
            "wall_mac_utilization": useful / (TOTAL_MACS * cycles)}


def minimum_footprint(cores: tuple[Core, ...], max_m: int, model: Model) -> dict:
    # Necessary lower bound only: ignores persistent Z/U, output combine and
    # physical banks/ports. Passing cannot certify a feasible complete machine.
    weight_double = sum(2 * c.pn * c.pk * 2 for c in cores)
    x_double = sum(2 * c.pm * c.pk * 2 for c in cores)
    group_acc = sum(max_m * c.pn * model.n_group_tiles * 4 for c in cores)
    total = weight_double + x_double + group_acc + 4096 + 8192
    return {"double_weight_min_bytes": weight_double, "double_x_min_bytes": x_double,
            "bounded_group_accumulator_min_bytes": group_acc,
            "minimum_accounted_bytes": total,
            "fixed_total_sram_budget_bytes": SRAM_BUDGET,
            "necessary_lower_bound_fits": total <= SRAM_BUDGET,
            "full_physical_feasibility": "unknown; Round B required"}


def csv_out(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in r.items()})


def toy_table(model: Model) -> list[dict]:
    result = []
    # One projection only, exact teacher mechanism. Fixed A/B ownership.
    for cores, owners in [((Core(6, 4),), (0, 0)),
                          ((Core(3, 4), Core(3, 4)), (0, 1)),
                          ((Core(4, 4), Core(2, 4)), (0, 1))]:
        finish = [0] * len(cores)
        useful = issued = issue_count = 0
        m_waves = []
        for m, owner in zip((4, 2), owners):
            p = projection(m, 128, 512, cores[owner], model)
            finish[owner] += p["cycles"]
            useful += p["useful_macs"]
            issued += p["issued_macs"]
            issue_count += p["issues"]
            m_waves.append(p["m_waves"])
        cycles = max(finish)
        result.append({"geometry": geometry_id(cores), "scope": "teacher fixed-owner projection; ideal operands",
                       "MAC_budget": TOTAL_MACS, "M_experts": [4, 2], "N": 128, "K": 512,
                       "M_waves_per_expert": m_waves, "issued_tiles": issue_count,
                       "cycles": cycles, "latency_ms_at_1ghz": cycles / 1_000_000,
                       "useful_macs": useful, "padding_macs": issued - useful,
                       "spatial_utilization": useful / issued,
                       "wall_mac_utilization": useful / (TOTAL_MACS * cycles)})
    return result


def run(inputs: Path, output: Path, model: Model) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parent
    timing_sources = [source_root / "rust/src/v3/mod.rs",
                      source_root / "rust/src/v3/IMPLEMENTATION_CONTRACT.md",
                      source_root / "rust/src/compute.rs"]
    source_pins = {"scope": "only dot20/II1/resultcommit1/vector64 constants are inherited; not exact Rust event timing",
                   "timing_sources": [{"path": str(p.resolve()), "sha256": sha(p)} for p in timing_sources],
                   "python_model": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
                   "test_source": {"path": str((source_root / "test_round_a_dse.py").resolve()),
                                   "sha256": sha(source_root / "test_round_a_dse.py")},
                   "context_limit": "up to8 N tiles times ceil(Me/PM) independent output records per group; no separate8-in-flight cap",
                   "old_projection_oracle_limit_difference": "old compute.rs has8 in-flight slots; this is a different prospective context hypothesis"}
    write_json(output / "SOURCE_PINS.json", source_pins)
    development_paths = [inputs / "development.json", inputs / "mixed_development.json"]
    dev_groups = {p.stem: json.loads(p.read_text())["workloads"] for p in development_paths}
    dev = sum(dev_groups.values(), [])
    dev_receipts = {p.stem: {"source_path": str(p.resolve()), "sha256": sha(p),
                           **verify_trace(dev_groups[p.stem])} for p in development_paths}
    geometries = enumerate_geometries()
    max_m = max(e["Me"] for w in dev for e in w["experts"])
    policies = ["eft"] + [f"threshold_{t}" for t in THRESHOLDS]
    search = []
    rows_by_config = {}
    repeated_points = 0
    for cores in geometries:
        footprint = minimum_footprint(cores, max_m, model)
        assert footprint["necessary_lower_bound_fits"]
        for policy in policies:
            # Every point is evaluated twice; the complete serialized layer
            # result must match, not just an aggregate statistic.
            one = [simulate_layer(w, cores, policy, model) for w in dev]
            two = [simulate_layer(w, cores, policy, model) for w in dev]
            assert canonical(one) == canonical(two)
            repeated_points += len(one)
            ag = aggregate(one)
            row = {"geometry": geometry_id(cores), "family": family(cores),
                   "cores_M_N_K": [asdict(c) for c in cores], "policy": policy,
                   "main_MAC_budget": TOTAL_MACS, **ag, **footprint}
            search.append(row)
            rows_by_config[(geometry_id(cores), policy)] = one
    write_json(output / "development_search.json", search)
    csv_out(output / "development_search.csv", search)

    # Architecture and runtime searches are explicitly separate. Geometry is
    # selected using common EFT, not whichever threshold makes hetero win.
    selected = {}
    for group in ("single", "homogeneous", "heterogeneous"):
        eligible = [r for r in search if r["family"] == group and r["policy"] == "eft"]
        selected[group] = min(eligible, key=lambda r: (r["total_compute_layer_cycles"], r["geometry"]))
    fixed = {"fixed_single_6": (Core(6, 4),),
             "fixed_homogeneous_3_3": (Core(3, 4), Core(3, 4)),
             "fixed_heterogeneous_4_2": (Core(2, 4), Core(4, 4))}
    lookup = {geometry_id(g): g for g in geometries}
    frozen_points = {}
    for name, row in selected.items():
        gid = row["geometry"]
        frozen_points[f"best_{name}_eft"] = {"geometry": gid, "policy": "eft"}
        runtime = min((r for r in search if r["geometry"] == gid and r["policy"] != "eft"),
                      key=lambda r: (r["total_compute_layer_cycles"], int(r["policy"].split("_")[-1])))
        frozen_points[f"best_{name}_threshold"] = {"geometry": gid, "policy": runtime["policy"]}
    for name, g in fixed.items():
        frozen_points[name] = {"geometry": geometry_id(g), "policy": "eft"}
    scope = {
        "scope": "post-Router compute/vector analytical layer makespan, ideal memory; not HBM E2E or complete inference",
        "main_MAC_budget": TOTAL_MACS, "precision": "BF16 input/weights, FP32 accumulation",
        "clock_hz": 1_000_000_000, "total_SRAM_bytes_nominal_only": SRAM_BUDGET,
        "HBM": "ideal/infinite; no bandwidth performance comparison in Round A",
        "geometry_search": {"PK": 512, "PM": [1, 8], "PN": [1, 24],
                            "single_geometries": 6, "two_core_geometries": 90,
                            "homogeneous_geometries": 5, "asymmetric_geometries": 85,
                            "mirrors_canonicalized": True},
        "model": asdict(model),
        "dataflow": "bounded output-group ideal-operand ordering, not a WS/OS/RS sweep",
        "functional_work": "Gate/Up concatenated same-X GEMM, SiLU-product, Down GEMM, weighted combine",
        "runtime": "FIFO descriptor order, whole-expert owner, no cross-expert row packing, no splitting/steal",
        "limitations": ["new PN widths are prospective, unsupported by frozen compiler/Rust datapath",
                        "minimum footprint is necessary only; actual X ports, accumulator banking and persistent Z/U not modeled",
                        "logical MAC equality is not physical area/PPA equality",
                        "vector uses one shared64-element/cycle service; no nonlinear numerical replay",
                        "EFT ignores future shared vector contention while binding; contention is scheduled in actual layer time",
                        "same v3 heldout traces were exposed for correctness; not pristine blind holdout",
                        "trace datasets are correlated captured windows/prefixes, not whole models"],
    }
    freeze = {"schema": "plena_round_a_geometry_freeze_v1", "scope": scope,
              "source_sha256": sha(Path(__file__)), "development_sources": dev_receipts,
              "source_pins_sha256": sha(output / "SOURCE_PINS.json"),
              "architecture_selection": "minimum sum of development makespans, EFT only, tie by geometry name",
              "runtime_selection": "threshold tuning only after geometry selection, development only",
              "selected_points": frozen_points,
              "heldout_opened_in_this_run_before_freeze": False,
              "development_points_repeated_twice": repeated_points,
              "development_repeat_full_json_exact": True}
    write_json(output / "FROZEN_ROUND_A.json", freeze)
    freeze_sha = sha(output / "FROZEN_ROUND_A.json")

    # The first heldout read occurs only AFTER the immutable selection file.
    heldout_paths = [inputs / "heldout.json", inputs / "mixed_heldout.json"]
    evaluation = []
    heldout_receipts = {}
    for path in heldout_paths:
        workloads = json.loads(path.read_text())["workloads"]
        heldout_receipts[path.stem] = {"source_path": str(path.resolve()), "sha256": sha(path),
                                     **verify_trace(workloads)}
        for label, selection in frozen_points.items():
            cores = lookup[selection["geometry"]]
            one = [simulate_layer(w, cores, selection["policy"], model) for w in workloads]
            two = [simulate_layer(w, cores, selection["policy"], model) for w in workloads]
            assert canonical(one) == canonical(two)
            evaluation += [{"selection": label, "split": path.stem, **r} for r in one]
    csv_out(output / "heldout_layers.csv", evaluation)
    write_json(output / "heldout_layers.json", evaluation)
    summaries = []
    for split in ("heldout", "mixed_heldout", "all_heldout"):
        data = [r for r in evaluation if split == "all_heldout" or r["split"] == split]
        reference = {}
        for label in frozen_points:
            rr = [r for r in data if r["selection"] == label]
            reference[label] = aggregate(rr)
        for label, ag in reference.items():
            summaries.append({"split": split, "selection": label,
                              **frozen_points[label], **ag,
                              "speedup_vs_fixed_single_6": reference["fixed_single_6"]["total_compute_layer_cycles"] / ag["total_compute_layer_cycles"],
                              "speedup_vs_best_single_eft": reference["best_single_eft"]["total_compute_layer_cycles"] / ag["total_compute_layer_cycles"],
                              "speedup_vs_best_homogeneous_eft": reference["best_homogeneous_eft"]["total_compute_layer_cycles"] / ag["total_compute_layer_cycles"]})
    csv_out(output / "heldout_summary.csv", summaries)
    write_json(output / "heldout_summary.json", summaries)
    toy = toy_table(model)
    csv_out(output / "teacher_toy.csv", toy)
    write_json(output / "teacher_toy.json", toy)
    # This is a latency/utilization frontier; area is not available.
    eft = [r for r in search if r["policy"] == "eft"]
    frontier = [r for r in eft if not any(
        q["total_compute_layer_cycles"] <= r["total_compute_layer_cycles"]
        and q["spatial_utilization"] >= r["spatial_utilization"]
        and (q["total_compute_layer_cycles"] < r["total_compute_layer_cycles"]
             or q["spatial_utilization"] > r["spatial_utilization"])
        for q in eft)]
    csv_out(output / "latency_utilization_frontier.csv", frontier)
    write_json(output / "input_provenance.json", {"development": dev_receipts, "heldout": heldout_receipts})
    best_heldout = next(r for r in summaries if r["split"] == "all_heldout" and r["selection"] == "best_heterogeneous_eft")
    conclusion = {"complete": True, "scope": scope,
                  "freeze_sha256": freeze_sha,
                  "geometry_configs": len(geometries), "development_geometry_policy_points": len(search),
                  "development_layer_evaluations_x2": 2 * repeated_points,
                  "heldout_selected_layer_evaluations_x2": 2 * len(evaluation),
                  "repeat_full_json_exact": True,
                  "five_requested_outputs": {"best_heterogeneous_config": frozen_points["best_heterogeneous_eft"],
                      "total_MoE_compute_layer_latency_ms": best_heldout["total_compute_layer_ms_at_1ghz"],
                      "speedup_vs_6U_single": best_heldout["speedup_vs_fixed_single_6"],
                      "average_MAC_spatial_utilization": best_heldout["spatial_utilization"],
                      "p95_layer_latency_ms": best_heldout["p95_layer_ms_at_1ghz"]},
                  "best_heterogeneous_beats_best_single_eft": best_heldout["speedup_vs_best_single_eft"] > 1,
                  "best_heterogeneous_beats_best_homogeneous_eft": best_heldout["speedup_vs_best_homogeneous_eft"] > 1,
                  "full_system_or_architectural_necessity_proven": False}
    write_json(output / "CONCLUSIONS.json", conclusion)
    lines = ["# Round A: prospective equal-MAC geometry screening", "", scope["scope"], "",
             f"96 geometries ×9 policies ×18 development layers ×2 repeats = {2*repeated_points:,} evaluations.",
             f"Frozen selection, then135 heldout layers ×9 selected points ×2 = {2*len(evaluation):,} evaluations.",
             "", "| Selection | M×N×K | Policy | Heldout total ms | p95 ms | Spatial util | Speedup vs6×4×512 | vsbest single | vsbest homo |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        if r["split"] == "all_heldout":
            lines.append(f"|{r['selection']}|{r['geometry']}|{r['policy']}|{r['total_compute_layer_ms_at_1ghz']:.6f}|{r['p95_layer_ms_at_1ghz']:.6f}|{r['spatial_utilization']:.3%}|{r['speedup_vs_fixed_single_6']:.3f}×|{r['speedup_vs_best_single_eft']:.3f}×|{r['speedup_vs_best_homogeneous_eft']:.3f}×|")
    lines += ["", "The fastest heldout point is not used to revise the development-selected hardware.",
              "A new PN width changes buffers and ports. All rows have12,288 main MACs, but full SRAM/port feasibility and PPA are deferred to Round B.",
              "Latency includes the modeled dot/result dependency and one global vector queue. It excludes HBM, SRAM bank service, paid runtime decision service, attention and Router.",
              "The captured heldout data was previously exposed for correctness in the v3 campaign; the new geometry selection is development-only but this is not a pristine blind test.",
              "No quantitative full-model accuracy claim is made by this metadata-only screening."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    assert sha(output / "FROZEN_ROUND_A.json") == freeze_sha
    assert all(sha(Path(pin["path"])) == pin["sha256"] for pin in source_pins["timing_sources"])
    return conclusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dot-latency", type=int, default=20)
    parser.add_argument("--n-group-tiles", type=int, default=8)
    args = parser.parse_args()
    if args.dot_latency < 1 or args.n_group_tiles < 1:
        parser.error("positive dot latency and bounded N-group required")
    model = Model(dot_latency=args.dot_latency, n_group_tiles=args.n_group_tiles)
    print(json.dumps(run(args.inputs, args.output, model), indent=2))


if __name__ == "__main__":
    main()
