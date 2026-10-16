"""Two fresh schedule evaluations for frozen hardware and profile optima."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from research.moe_dispatch.plena65536_shared.geometry import Core
from research.moe_dispatch.plena65536_shared.timing import PROFILES
from .model import schedule
from .report import canonical, frozen_designs, gm, sha256, write_csv, write_json
from .workloads import load_workloads


def main(out):
    selection_path = out / "FROZEN_SELECTION.json"
    selection = json.loads(selection_path.read_text())
    inputs = load_workloads()
    configurations = []
    for design, scope, cores, source in frozen_designs(selection):
        for profile in PROFILES:
            configurations.append(("primary_frozen", design, cores, profile))
    for profile in PROFILES:
        for design, record in selection["profile_optima"][profile.name].items():
            cores = tuple(Core(**{key: core[key] for key in ("b", "r", "rm", "rn")})
                          for core in record["cores"])
            configurations.append(("profile_selected", design, cores, profile))
    rows, summaries, unique = [], [], {}
    for kind, design, cores, profile in configurations:
        for split in ("development", "heldout"):
            grouped, values = defaultdict(list), []
            for window in inputs[split]:
                key = (cores, profile, str(window["id"]))
                if key not in unique:
                    first, second = schedule(cores, window, profile), schedule(cores, window, profile)
                    if canonical(first) != canonical(second):
                        raise AssertionError("Two independently reconstructed timing schedules differ")
                    unique[key] = (first["cycles"], hashlib.sha256(canonical(first).encode()).hexdigest())
                cycles, digest = unique[key]
                grouped[int(window["batch"])].append(cycles)
                values.append(cycles)
                rows.append({"selection": kind, "design": design, "profile": profile.name,
                             "split": split, "window_id": window["id"], "batch": int(window["batch"]),
                             "compute_cycles": cycles, "compute_ms": cycles / 1e6,
                             "schedule_sha256": digest, "repetitions": 2, "repeat_identical": True})
            summaries.append({"selection": kind, "design": design, "profile": profile.name,
                              "split": split, "window_count": len(values),
                              "GM_compute_ms": gm(gm(group) for group in grouped.values()) / 1e6,
                              "sample_GM_compute_ms": gm(values) / 1e6,
                              "geometry": " + ".join(str(core) for core in cores), "repeat_identical": True})
    write_csv(out / "timing_profile_points.csv", rows)
    write_csv(out / "timing_profile_summary.csv", summaries)
    write_json(out / "TIMING_REPEAT_AUDIT.json", {
        "status": "PASS", "unique_freshly_repeated_schedules": len(unique), "repetitions_each": 2,
        "all_independent_schedule_comparisons_identical": True,
        "rows": len(rows), "summary_rows": len(summaries),
        "scope": "Primary default-profile frozen geometries under all three timings; each profile's independently development-selected geometries under its own timing",
        "source_sha256": {path.name: sha256(path) for path in (
            selection_path, Path(__file__), Path(__file__).with_name("model.py"))},
    })
    print(json.dumps({"status": "PASS", "unique_repeat_points": len(unique), "rows": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
