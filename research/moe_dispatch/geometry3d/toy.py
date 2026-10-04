"""Versioned teacher example for finite-eight-record 3D compute semantics.

Two whole experts, M=(4,2), each execute one N=128/K=512 projection.
The single core executes both serially; dual cores execute one fixed expert
each.  Operands are ideal.  This isolates padding, pipeline and context
effects; it is not a whole-MoE or physical supply comparison.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import heapq
import json
from pathlib import Path
from typing import Any

from .compute import (Core, ContextLimits, DEFAULT_TIMING, TOTAL_MACS,
                      TimingProfile, geometry_id, projection)


def issue_oracle(m: int, n: int, k: int, core: Core, timing: TimingProfile,
                 limits: ContextLimits) -> dict[str, Any]:
    """Independent issue/commit scoreboard, including actual logical tails.

    Does not call projection/group_cost or their ceil/closed-form helpers.
    Each resident output tile retains its FP32 result until all K segments
    have committed; replacement occurs after the group's final result.
    """
    if min(m, n, k) <= 0:
        raise ValueError("positive projection dimensions required")
    capacity = min(limits.max_records, limits.accumulator_bytes // (core.pm * core.pn * 4))
    if capacity < 1:
        raise ValueError("one physical FP32 output record does not fit")
    records = [(mi, ni) for ni in range(0, n, core.pn) for mi in range(0, m, core.pm)]
    latency = dict(timing.latency_by_pk)[core.pk] + timing.commit_cycles
    next_issue = 0
    previous_group_done = 0
    events = []
    useful_macs = 0
    peak_outstanding = 0
    for first in range(0, len(records), capacity):
        group = records[first:first + capacity]
        previous = {record: 0 for record in group}
        pending = []
        for ki in range(0, k, core.pk):
            for mi, ni in group:
                record = (mi, ni)
                now = max(next_issue, previous_group_done, previous[record])
                while pending and pending[0][0] <= now:
                    heapq.heappop(pending)
                commit = now + latency
                heapq.heappush(pending, (commit, record))
                peak_outstanding = max(peak_outstanding, len(pending))
                previous[record] = commit
                next_issue = now + timing.initiation_interval
                useful = min(core.pm, m - mi) * min(core.pn, n - ni) * min(core.pk, k - ki)
                useful_macs += useful
                events.append({"m0": mi, "n0": ni, "k0": ki,
                               "issue": now, "commit": commit, "useful_macs": useful})
        previous_group_done = max(previous.values())
    issued_macs = len(events) * core.macs
    return {"cycles": previous_group_done, "issues": len(events),
            "useful_macs": useful_macs, "issued_macs": issued_macs,
            "padding_macs": issued_macs - useful_macs,
            "peak_outstanding_results": peak_outstanding, "events": events}


def teacher_table(timing: TimingProfile = DEFAULT_TIMING,
                  limits: ContextLimits = ContextLimits()) -> list[dict[str, Any]]:
    rows = []
    cases = (("fixed_6", (Core(6, 4, 512),), (0, 0)),
             ("fixed_3+3", (Core(3, 4, 512), Core(3, 4, 512)), (0, 1)),
             ("fixed_4+2", (Core(2, 4, 512), Core(4, 4, 512)), (1, 0)))
    for label, cores, owners in cases:
        finish = [0] * len(cores)
        useful = issued = issues = 0
        experts = []
        for expert, (m, owner) in enumerate(zip((4, 2), owners)):
            p = projection(m, 128, 512, cores[owner], timing, limits)
            oracle = issue_oracle(m, 128, 512, cores[owner], timing, limits)
            for key in ("cycles", "issues", "useful_macs", "issued_macs", "padding_macs"):
                if getattr(p, key) != oracle[key]:
                    raise AssertionError(f"compute projection disagrees with explicit issue oracle: {key}")
            start = finish[owner]
            finish[owner] += p.cycles
            useful += p.useful_macs
            issued += p.issued_macs
            issues += p.issues
            experts.append({"expert": "AB"[expert], "m": m, "owner": owner,
                            "core": asdict(cores[owner]), "start_cycle": start,
                            "finish_cycle": finish[owner], "projection": asdict(p),
                            "oracle_peak_outstanding_results": oracle["peak_outstanding_results"]})
        cycles = max(finish)
        rows.append({"label": label, "geometry": geometry_id(cores),
                     "main_multipliers": sum(c.macs for c in cores),
                     "M_experts": [4, 2], "N": 128, "K": 512,
                     "M_waves_per_expert": [e["projection"]["m_waves"] for e in experts],
                     "groups_per_expert": [e["projection"]["group_count"] for e in experts],
                     "cycles_per_expert": [e["projection"]["cycles"] for e in experts],
                     "core_finish_cycles": finish, "cycles": cycles,
                     "estimated_latency_ms_at_1ghz": cycles / 1_000_000,
                     "issues": issues, "useful_macs": useful,
                     "issued_macs": issued, "padding_macs": issued - useful,
                     "spatial_utilization": useful / issued,
                     "wall_mac_utilization": useful / (TOTAL_MACS * cycles),
                     "oracle_verified": True, "experts": experts})
    return rows


def write_toy(directory: Path) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    rows = teacher_table()
    report = {
        "schema": "plena_geometry3d_teacher_finite8_v1",
        "scope": "ideal-operand single projection; fixed whole-expert ownership; prospective analytical 1GHz",
        "semantics": "finite8 output records; N-major/M-minor group; same-output K waits prior commit; groups retire before replacement",
        "timing": asdict(DEFAULT_TIMING), "limits": asdict(ContextLimits()),
        "not_claimed": ["whole expert Gate/Up/Down latency", "HBM/port/vector/control timing", "silicon speedup"],
        "rows": rows,
    }
    (directory / "teacher_toy_geometry3d.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    csv_rows = [{k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items() if k != "experts"}
                for r in rows]
    with (directory / "teacher_toy_geometry3d.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = write_toy(args.out)
    print(json.dumps({"schema": report["schema"],
                      "rows": [{"label": r["label"], "cycles": r["cycles"],
                                "spatial_utilization": r["spatial_utilization"]}
                               for r in report["rows"]]}, sort_keys=True))


if __name__ == "__main__":
    main()
