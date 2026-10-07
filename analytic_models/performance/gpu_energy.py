"""Windowed NVML energy integration and immutable legacy-campaign reanalysis.

Legacy captures did not retain batch boundaries or timestamp the power read
itself. Their request-window reanalysis is explicitly approximate, never a
replacement capture or an exact corrected batch-energy measurement.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any


class PowerTrace:
    """A time-ordered power trace; integration never extrapolates its edges."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        readings: dict[int, float] = {}
        for row in rows:
            power = row.get("power_w")
            if power is None or power == "N/A":
                continue
            timestamp, watts = int(row["timestamp_ns"]), float(power)
            if timestamp < 0 or not math.isfinite(watts) or watts < 0:
                raise ValueError("power samples must have non-negative finite timestamps and watts")
            if timestamp in readings and readings[timestamp] != watts:
                raise ValueError("conflicting power readings at the same timestamp")
            readings[timestamp] = watts
        self.timestamps = sorted(readings)
        self.watts = [readings[t] for t in self.timestamps]
        if len(self.timestamps) < 2:
            raise ValueError("energy integration requires at least two power samples")

    def _power_at(self, timestamp: int) -> float:
        index = bisect.bisect_left(self.timestamps, timestamp)
        if index < len(self.timestamps) and self.timestamps[index] == timestamp:
            return self.watts[index]
        if index == 0 or index == len(self.timestamps):
            raise ValueError("power samples do not bracket the integration window")
        left, right = self.timestamps[index - 1], self.timestamps[index]
        fraction = (timestamp - left) / (right - left)
        return self.watts[index - 1] + fraction * (self.watts[index] - self.watts[index - 1])

    def integrate(self, start_ns: int, end_ns: int) -> float:
        if not 0 <= start_ns < end_ns:
            raise ValueError("integration requires a positive-duration window")
        left = bisect.bisect_right(self.timestamps, start_ns)
        right = bisect.bisect_left(self.timestamps, end_ns)
        times = [start_ns, *self.timestamps[left:right], end_ns]
        values = [self._power_at(start_ns), *self.watts[left:right], self._power_at(end_ns)]
        return math.fsum((b - a) / 1e9 * (x + y) / 2 for a, b, x, y in zip(times, times[1:], values, values[1:]))


def _verified_sources(root: Path) -> dict[str, str]:
    checksums = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64 or name in checksums:
            raise ValueError("malformed campaign SHA256SUMS")
        checksums[name] = digest
    verified = {}
    for name in ("latency_raw.jsonl", "power_raw.csv"):
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if checksums.get(name) != digest:
            raise ValueError(f"campaign checksum mismatch: {name}")
        verified[name] = digest
    return verified


def reanalyze_campaign(root: Path) -> dict[str, Any]:
    sources = _verified_sources(root)
    with (root / "power_raw.csv").open(newline="") as source:
        power_rows = list(csv.DictReader(source))
    trace = PowerTrace(power_rows)
    power_by_trial: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in power_rows:
        power_by_trial[row["trial_id"]].append(row)
    requests: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (root / "latency_raw.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            if row.get("mode") == "batch_sweep" and row.get("phase") == "measurement":
                requests[row["trial_id"]].append(row)
    trials = []
    for trial_id, members in sorted(requests.items()):
        record = members[0]
        if len(members) != int(record["batch_size"]):
            raise ValueError(f"incomplete batch trial: {trial_id}")
        boundaries = {(r.get("group_start_ns"), r.get("group_end_ns")) for r in members}
        if len(boundaries) != 1:
            raise ValueError(f"inconsistent batch boundaries: {trial_id}")
        start, end = boundaries.pop()
        if start is None and end is None:
            start = min(r["arrival_ns"] for r in members)
            end = max(r["finish_ns"] for r in members)
            window_kind = "observed_request_window_approximation"
        elif start is None or end is None:
            raise ValueError(f"incomplete batch boundaries: {trial_id}")
        else:
            window_kind = "recorded_batch_window"
        count = int(record["power_sample_count"])
        original = (
            [row for row in power_rows if start <= int(row["timestamp_ns"]) <= end]
            if record.get("energy_method") == "sorted_interpolated_recorded_batch_window_v2"
            else power_by_trial[trial_id][:count]
        )
        if len(original) != count:
            raise ValueError(f"power sample count exceeds archived rows: {trial_id}")
        times = [int(row["timestamp_ns"]) for row in original]
        trials.append(
            {
                "trial_id": trial_id,
                "benchmark": record["benchmark"],
                "batch_size": int(record["batch_size"]),
                "group_index": int(record["group_index"]),
                "window_kind": window_kind,
                "archived_energy_joules": float(record["batch_energy_joules"]),
                "window_reanalysis_joules": trace.integrate(start, end),
                "archived_sample_order_nonmonotonic": any(b < a for a, b in pairwise(times)),
            }
        )
    if not trials:
        raise ValueError("campaign contains no measured batch-sweep trials")
    buckets: dict[tuple[str, int, int | None], list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        for key in (
            (trial["benchmark"], trial["batch_size"], trial["group_index"]),
            ("all", trial["batch_size"], None),
        ):
            buckets[key].append(trial)
    summaries = []
    for (benchmark, batch, group), items in sorted(buckets.items()):
        kinds = sorted({item["window_kind"] for item in items})
        summaries.append(
            {
                "benchmark": benchmark,
                "batch_size": batch,
                "group_index": group,
                "trial_count": len(items),
                "window_kinds": kinds,
                "archived_batch_energy_joules_median": statistics.median(
                    item["archived_energy_joules"] for item in items
                ),
                "window_reanalysis_joules_median": statistics.median(
                    item["window_reanalysis_joules"] for item in items
                ),
                "nonmonotonic_archived_trials": sum(item["archived_sample_order_nonmonotonic"] for item in items),
            }
        )
    return {
        "schema_version": 1,
        "method": "sorted_linear_interpolation_clipped_to_explicit_window_v1",
        "sources_sha256": sources,
        "claim_boundary": (
            "Offline reanalysis, not a new GPU capture. Legacy logs use observed request boundaries because "
            "batch start/end and the precise power-read time were not recorded; these values are approximate "
            "and must not be substituted for an exact corrected batch-energy baseline."
        ),
        "trial_count": len(trials),
        "summary": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.campaign_root.resolve(), args.output.resolve()
    if output.is_relative_to(root):
        raise ValueError("write the revision outside the immutable raw campaign")
    revision = reanalyze_campaign(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(revision, indent=2) + "\n")
    print(json.dumps({"trial_count": revision["trial_count"], "output": str(output)}))


if __name__ == "__main__":
    main()
