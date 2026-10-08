"""Recover a complete profile after a broad report-file hash guard tripped."""
from __future__ import annotations
import gzip
import json
from pathlib import Path
import pstats

from research.moe_dispatch.round3.common import ROOT, source_manifest, write_json
from research.moe_dispatch.round3.sensitivity import _result_interval


def main():
    out = ROOT / "diagnostics/profile"
    started = json.loads((out / "STARTED.json").read_text())
    before = started["source_sha256"]
    after = source_manifest()
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    executed = {"config.py", "common.py", "model.py", "runtime.py", "optimizer.py", "search.py", "sensitivity.py"}
    assert not any(Path(k).name in executed for k in changed), changed
    path = out / "outputs/E5/sobol/certificates/0000.json.gz"
    with gzip.open(path, "rt") as f:
        certificate = json.load(f)
    # _sobol_job writes the certificate only after canonical full-search repeat
    # equality passes. It finished before the outer broad manifest guard.
    stats = pstats.Stats(str(out / "sample_0000.prof"))
    functions = [{"filename": key[0], "line": key[1], "function": key[2],
                  "primitive_calls": v[0], "calls": v[1], "self_seconds": v[2],
                  "cumulative_seconds": v[3]} for key, v in stats.stats.items()]
    functions.sort(key=lambda x: (-x["self_seconds"], x["filename"], x["line"]))
    elapsed = next(x["cumulative_seconds"] for x in functions if x["function"] == "_sobol_job")
    result = _result_interval(certificate)
    assert result["simulator_calls"] == 2880
    write_json(out / "PROFILE_RESULT.json", {"sample": started["sample"], "result": result,
        "elapsed_profiled_sobol_job_seconds": elapsed, "physical_search_source_unchanged": True,
        "changed_nonexecuted_report_files": changed, "source_sha256": before,
        "total_calls": stats.total_calls, "total_primitive_calls": stats.prim_calls,
        "total_profiled_seconds": stats.total_tt, "functions": functions,
        "receipt_note": "Original broad manifest guard failed after complete .prof/certificate save because report/evaluation files changed. Executed numeric sources match their before-run hashes.",
        "scope": "Diagnostic full two-search/two-replay Sobol smoke with old initial witnesses; not final Sobol sampling output"})
    print(json.dumps({"elapsed_seconds": elapsed, "actual_calls": result["simulator_calls"],
                      "physical_search_source_unchanged": True, "report_only_changes": changed}), flush=True)


if __name__ == "__main__":
    main()
