"""Read-only physical/search cProfile diagnostic; never changes main evidence."""
from __future__ import annotations
import cProfile
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import pstats
import time

from research.moe_dispatch.round3 import sensitivity
from research.moe_dispatch.round3.common import (ROOT, inputs, frozen_designs,
                                                source_manifest, write_json)


def main():
    out = ROOT / "diagnostics/profile"
    out.mkdir(parents=True, exist_ok=True)
    # Exactly reproduce search agent's full smoke, including two full searches
    # and the two physical replays inside every successful point evaluation.
    spec = (0, [15, 16, 2, 520, 1], inputs()["development"], 8, 32,
            list(frozen_designs("pipelined").values()))
    before = source_manifest()
    p = cProfile.Profile()
    started = time.monotonic()
    write_json(out / "STARTED.json", {"sample": spec[1], "candidate_budget_per_family": 8,
               "node_budget_per_family": 32, "development_windows": len(spec[2]),
               "initials": "old frozen pipelined hardware; diagnostic only, not final Sobol selection",
               "source_sha256": before, "complete_search_repeats": 2,
               "physical_replays_per_evaluated_window": 2})
    # Output redirection is local to this diagnostic process; sources untouched.
    sensitivity.ROOT = out / "outputs"
    p.enable()
    result = sensitivity._sobol_job(spec)
    p.disable()
    elapsed = time.monotonic() - started
    p.dump_stats(str(out / "sample_0000.prof"))
    streams = {}
    for key in ("cumulative", "tottime"):
        stream = io.StringIO()
        pstats.Stats(p, stream=stream).strip_dirs().sort_stats(key).print_stats(80)
        streams[key] = stream.getvalue()
        (out / (key + ".txt")).write_text(streams[key])
    stats = pstats.Stats(p)
    functions = [{"filename": f[0], "line": f[1], "function": f[2], "primitive_calls": v[0],
                  "calls": v[1], "self_seconds": v[2], "cumulative_seconds": v[3]}
                 for f, v in stats.stats.items()]
    functions.sort(key=lambda x: (-x["self_seconds"], x["filename"], x["line"]))
    after = source_manifest()
    executed = {"config.py", "common.py", "model.py", "runtime.py", "optimizer.py", "search.py", "sensitivity.py"}
    changed = [k for k in set(before) | set(after) if before.get(k) != after.get(k)]
    if any(Path(k).name in executed for k in changed):
        raise AssertionError("Physical/search source changed during profile")
    write_json(out / "PROFILE_RESULT.json", {"elapsed_seconds_with_cprofile": elapsed,
               "result": result, "physical_search_source_unchanged": True, "source_sha256": before,
               "changed_nonexecuted_report_files": changed,
               "functions": functions, "total_calls": stats.total_calls,
               "total_primitive_calls": stats.prim_calls, "total_profiled_seconds": stats.total_tt,
               "scope": "Full exact smoke algorithm, two searches/two replays preserved; old initial witnesses and profiler overhead, not a final sensitivity result"})
    print(json.dumps({"elapsed_seconds": elapsed, "result": result,
                     "largest_self_functions": functions[:12]}), flush=True)


if __name__ == "__main__":
    main()
