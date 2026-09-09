#!/usr/bin/env python3
"""Post-B8 diagnostic of a newly legal P=6, R=512 large core.

This is one explicitly selected hypothesis, not a DSE or a predeclared winner.
All workloads reuse a previously exported full expert bank.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True, help="existing fixed_bank_full_qwen directory")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepared = args.prepared_root.resolve(strict=True)
    binary = args.binary.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent / "moe_timing/replay"))
    from compare_moe_normal import require, run_comparison

    original = prepared / "architectures/normal_v2_n2_heterogeneous.json"
    source_single = prepared / "architectures/normal_v2_n2_single.json"
    candidate = json.loads(original.read_text())
    require(candidate["schema_version"] == 2 and len(candidate["cores"]) == 2, "requires existing V2 heterogeneous case")
    large, small = candidate["cores"]
    require((large["blen"], large["mlen"], small["blen"], small["mlen"]) == (4, 768, 4, 256),
            "source shape differs from diagnostic hypothesis")
    require([c["refinement"]["m_rows"] for c in candidate["cores"]] == [4, 1], "source token tiles differ")
    require(all(c["refinement"]["active_n_tiles"] == 2 and c["weight_slots"] == 3 for c in candidate["cores"]),
            "requires two independent N groups and three slots per core")
    require(candidate["dma"]["lookup_ii_cycles"] == 1, "probe keeps one-cycle DMA initiation interval")
    before_multipliers = large["blen"] * large["mlen"]
    before_latches = large["refinement"]["operand_latch_bytes"]
    large.update(blen=6, mlen=512)
    large["refinement"]["operand_latch_bytes"] = 2 * large["blen"] * large["mlen"] * 2
    require(large["blen"] * large["mlen"] == before_multipliers == 3072, "large multiplier budget changed")
    require(large["refinement"]["operand_latch_bytes"] == before_latches, "operand latch budget changed")
    require(large["mlen"] % large["blen"] != 0, "probe must exercise shape forbidden by V1")
    configs = {}
    for policy in ("per_channel", "demand_aware"):
        directory = output / "architectures" / policy
        directory.mkdir(parents=True)
        single = json.loads(source_single.read_text())
        tested = copy.deepcopy(candidate)
        single["name"] = f"nonsquare_probe_{policy}_single"
        tested["name"] = f"nonsquare_probe_{policy}_p6_r512"
        for config in (single, tested):
            config["dma"]["issue_policy"] = policy
        configs[policy] = [directory / "single.json", directory / "candidate.json"]
        for path, config in zip(configs[policy], (single, tested), strict=True):
            write_json(path, config)
    windows = [prepared / "windows" / name for name in ("qwen_full_decode_b8", "qwen_full_decode_b32")]
    bank = prepared / "banks/qwen/bank.json"
    image = bank.parent / "weights.bin"
    hypothesis = dict(
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        selection="post-B8 diagnostic after observing original V2 candidates; not predeclared search or DSE",
        hypothesis="P6,R512 preserves3072 large multipliers while reducing R768 padding for Qwen D2048/F512; P6 introduces small N tails. SRAM ports, scheduling and shared DMA may offset any compute benefit.",
        changes=dict(large_before=dict(P=4, R=768), large_after=dict(P=6, R=512), small=dict(P=4, R=256),
                     temporal_M=[4, 1], active_n_tiles=2, weight_slots=[3, 3]),
        controls="same bank/routes, 4096 total multipliers, SRAM capacities, activation/weight/accumulator port totals, vector service, DMA credits and policy within each paired comparison",
        repeats=2, successful_native_runs_expected=16, reuses_cached_baseline_reports=False,
        source_sha256={str(p): digest(p) for p in (Path(__file__), original, source_single,
                                                here.parent / "moe_timing/replay/compare_moe_normal.py")},
        bank_sha256=digest(bank), hbm_sha256=digest(image), binary_sha256=digest(binary),
        window_sha256={str(w / name): digest(w / name) for w in windows for name in ("workload.json", "golden.json")},
        architecture_sha256={str(p): digest(p) for group in configs.values() for p in group})
    write_json(output / "hypothesis.json", hypothesis)  # Written before any native execution.
    shutil.copyfile(original, output / "original_heterogeneous.json")
    rows = []
    for window in windows:
        for policy, architectures in configs.items():
            result_dir = output / "comparisons" / window.name / policy
            summary = run_comparison(binary, window / "workload.json", window / "golden.json", architectures,
                                     result_dir, repeats=2, atol=0, rtol=0, max_hbm_bytes=1 << 30, timeout=1800)
            require(summary["all_gates_passed"], "nonsquare comparison gates failed")
            single, dual = summary["comparisons"]
            require(all(c["jobs"] > 0 for c in dual["result"]["cores"]), "both cores must execute work")
            rows.append(dict(window=window.name, policy=policy,
                             baseline_total_ps=single["result"]["total_ps"], candidate_total_ps=dual["result"]["total_ps"],
                             speedup=single["result"]["total_ps"] / dual["result"]["total_ps"],
                             baseline_useful_macs=single["result"]["useful_macs"], candidate_useful_macs=dual["result"]["useful_macs"],
                             candidate_issued_macs=dual["result"]["issued_macs"],
                             result=str(result_dir / "comparison.json")))
            write_json(output / "progress.json", dict(status="running", completed=rows))
    for group in ("source_sha256", "window_sha256", "architecture_sha256"):
        require(all(digest(Path(p)) == sha for p, sha in hypothesis[group].items()), "diagnostic inputs changed: " + group)
    require(digest(bank) == hypothesis["bank_sha256"] and digest(image) == hypothesis["hbm_sha256"], "weight bank changed")
    require(digest(binary) == hypothesis["binary_sha256"], "native binary changed")
    final = dict(status="passed", all_gates_passed=True, successful_native_runs=16, rows=rows,
                 hypothesis_sha256=digest(output / "hypothesis.json"),
                 scope="one post-B8 shape diagnostic on two full-dimension synthetic-value fixed-bank route windows; not DSE or full-model inference")
    write_json(output / "summary.json", final)
    write_json(output / "progress.json", final)
    print(json.dumps(dict(status="passed", result=str(output / "summary.json"))))


if __name__ == "__main__":
    main()
