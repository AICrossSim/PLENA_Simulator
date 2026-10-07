#!/usr/bin/env python3
"""Run the Qwen nonzero functional gate and its fault-injection self-checks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from transactional_emulator.testbench.window1_p2.build_route_traces import trace_from_record
from transactional_emulator.testbench.window1_p2.p2_utils import iter_jsonl, write_csv, write_json


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEED = 20260709
FAULTS = (
    "expert_base_plus_one_block",
    "swap_first_two_experts",
    "shift_token_expert_map",
    "scale_element_misaligned",
)


def _detect_plena_root() -> Path:
    for parent in REPO_ROOT.parents:
        if (parent / "outputs" / "decode_sweep50").exists() and (parent / "weights").exists():
            return parent
    return REPO_ROOT.parents[2]


def _routing_files(plena_root: Path) -> list[Path]:
    root = plena_root / "outputs" / "decode_sweep50"
    return [
        root / "bfcl_s0_routing.jsonl",
        root / "bfcl_s1_routing.jsonl",
        root / "gpqa_s0_routing.jsonl",
    ]


def _record_expert_counts(record: dict[str, Any]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for row in record["routes"]:
        for expert_id in row:
            counts[int(expert_id)] += 1
    return counts


def _unique_experts(record: dict[str, Any]) -> int:
    return len(_record_expert_counts(record))


def _max_expert_count(record: dict[str, Any]) -> int:
    counts = _record_expert_counts(record)
    return max(counts.values()) if counts else 0


def _iter_records(paths: list[Path]):
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"missing routing file: {path}")
        for record in iter_jsonl(path):
            if record.get("model_key") == "qwen3" and "routes" in record and record.get("route_weights") is not None:
                yield path, record


def _choose_scenarios(paths: list[Path]) -> list[tuple[str, Path, dict[str, Any]]]:
    records = list(_iter_records(paths))
    single = next(
        (item for item in records if item[1].get("phase") == "decode" and int(item[1].get("tokens", 0)) == 1),
        None,
    )
    same_expert = min(
        (
            item
            for item in records
            if item[1].get("phase") == "prefill"
            and int(item[1].get("tokens", 0)) >= 2
            and _max_expert_count(item[1]) >= 2
        ),
        key=lambda item: (int(item[1].get("tokens", 10**9)), int(item[1].get("layer", 10**9))),
        default=None,
    )
    cross_expert = min(
        (
            item
            for item in records
            if item[1].get("phase") == "prefill"
            and int(item[1].get("tokens", 0)) >= 20
            and _unique_experts(item[1]) >= 10
        ),
        key=lambda item: (int(item[1].get("tokens", 10**9)), int(item[1].get("layer", 10**9))),
        default=None,
    )
    missing = [
        name
        for name, item in (
            ("single_token_top8", single),
            ("same_expert_multi_token", same_expert),
            ("cross_expert_distribution", cross_expert),
        )
        if item is None
    ]
    if missing:
        raise ValueError("could not find required scenario(s): " + ", ".join(missing))
    return [
        ("single_token_top8", single[0], single[1]),
        ("same_expert_multi_token", same_expert[0], same_expert[1]),
        ("cross_expert_distribution", cross_expert[0], cross_expert[1]),
    ]


def _write_trace(
    *,
    scenario: str,
    record: dict[str, Any],
    out_dir: Path,
    mlen: int,
    blen: int,
    emu_threads: int,
) -> Path:
    trace = trace_from_record(
        record,
        mlen=mlen,
        blen=blen,
        emu_threads=emu_threads,
        allow_uniform_weights=False,
    )
    trace["trace_id"] = f"nonzero_gate_{scenario}_{trace['trace_id']}"
    path = out_dir / f"{trace['trace_id']}.json"
    write_json(path, trace)
    return path


def _load_run_summary(build_dir: Path) -> dict[str, Any]:
    path = build_dir / "qwen3_trace_replay_results.json"
    if not path.exists():
        raise FileNotFoundError(f"missing replay summary: {path}")
    return json.loads(path.read_text())


def _run_trace(
    *,
    trace_path: Path,
    build_dir: Path,
    log_path: Path,
    qwen_snapshot: Path,
    seed: int,
    mlen: int,
    blen: int,
    emu_threads: int,
    fault: str = "none",
    expect_failure: bool = False,
    stage_profile: bool = False,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "transactional_emulator.testbench.window1_p2.qwen3_trace_replay_test",
        str(trace_path),
        "--build-dir",
        str(build_dir),
        "--input-mode",
        "random",
        "--seed",
        str(seed),
        "--functional-golden-mode",
        "nonzero-real",
        "--qwen-snapshot",
        str(qwen_snapshot),
        "--nonzero-rel-rms-threshold",
        "0.01",
        "--fault-injection",
        fault,
        "--mlen",
        str(mlen),
        "--blen",
        str(blen),
        "--emu-threads",
        str(emu_threads),
    ]
    if expect_failure:
        cmd.append("--expect-nonzero-gate-fail")
    if stage_profile:
        cmd.append("--stage-profile")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"trace run failed rc={proc.returncode}; see {log_path}")
    summary = _load_run_summary(build_dir)
    return {
        "build_dir": str(build_dir),
        "log": str(log_path),
        "trace": str(trace_path),
        "fault": fault,
        "expect_failure": expect_failure,
        "returncode": proc.returncode,
        "summary": summary,
    }


def _existing_run(
    *,
    trace_path: Path,
    build_dir: Path,
    fault: str,
    expect_failure: bool,
) -> dict[str, Any]:
    summary = _load_run_summary(build_dir)
    run_log = summary.get("run_metrics", {}).get("log_path")
    return {
        "build_dir": str(build_dir),
        "log": str(run_log or build_dir / "rust_emulator_stdout.log"),
        "trace": str(trace_path),
        "fault": fault,
        "expect_failure": expect_failure,
        "returncode": 0,
        "summary": summary,
    }


def _scenario_row(scenario: str, source_path: Path, record: dict[str, Any], trace_path: Path) -> dict[str, Any]:
    counts = _record_expert_counts(record)
    return {
        "scenario": scenario,
        "source": str(source_path),
        "benchmark": record.get("benchmark"),
        "sample_id": record.get("sample_id"),
        "sample_index": record.get("sample_index"),
        "phase": record.get("phase"),
        "layer": record.get("layer"),
        "tokens": record.get("tokens"),
        "unique_experts": len(counts),
        "max_expert_count": max(counts.values()) if counts else 0,
        "trace_path": str(trace_path),
    }


def _result_row(kind: str, name: str, run: dict[str, Any]) -> dict[str, Any]:
    summary = run["summary"]
    gate = summary.get("nonzero_functional_gate") or {}
    metrics = summary.get("run_metrics") or {}
    return {
        "kind": kind,
        "name": name,
        "fault": run["fault"],
        "expected_failure": run["expect_failure"],
        "gate_passed": gate.get("passed"),
        "rel_rms": gate.get("rel_rms"),
        "threshold": gate.get("threshold"),
        "sim_latency_cycles": metrics.get("sim_latency_cycles"),
        "hbm_bytes_read": metrics.get("hbm_bytes_read"),
        "hbm_bytes_written": metrics.get("hbm_bytes_written"),
        "build_dir": run["build_dir"],
        "log": run["log"],
    }


def _write_report(
    *,
    path: Path,
    scenario_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    qwen_snapshot: Path,
    seed: int,
) -> None:
    pass_rows = [row for row in result_rows if row["kind"] == "golden_pass"]
    fault_rows = [row for row in result_rows if row["kind"] == "fault_injection"]
    passed = all(bool(row["gate_passed"]) for row in pass_rows)
    caught = all(row["gate_passed"] is False for row in fault_rows)
    lines = [
        "# Qwen 非零权重 Functional Gate 停车点 A",
        "",
        "## 结论",
        "",
        f"- 非零真实权重 gate: {'PASS' if passed else 'FAIL'} ({len(pass_rows)}/{len(pass_rows)})",
        f"- 故障注入自检: {'PASS' if caught else 'FAIL'} ({sum(row['gate_passed'] is False for row in fault_rows)}/{len(fault_rows)} caught)",
        "- 阶段二 S1 grouped execution: 未开始，等待人工确认。",
        "",
        "## 方法",
        "",
        "- 输入路由来自 `outputs/decode_sweep50` 的 Qwen3 sweep50 真实 router trace。",
        "- hidden states 未在 sweep50 产物中找到，本 gate 使用固定 seed 随机 BF16 activation；这只影响功能覆盖输入值，不影响专家寻址/scale/路由校验目的。",
        f"- Qwen 权重目录: `{qwen_snapshot}`",
        f"- random activation seed: `{seed}`",
        "- Golden 由 PyTorch CPU 路径生成，使用真实 Qwen expert weights、真实 route weights、标准 SwiGLU、MXFP8 权重取数路径和 BF16 VRAM-style accumulation。",
        "- Emulator 运行同一 route replay，输出矩阵与 PyTorch golden 做逐元素 rel_rms 比对，阈值 `0.01`。",
        "",
        "## 三个真实负载",
        "",
        "| 场景 | benchmark | sample | phase | layer | tokens | unique experts | max expert count |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scenario_rows:
        lines.append(
            f"| {row['scenario']} | {row['benchmark']} | {row['sample_id']} | {row['phase']} | "
            f"{row['layer']} | {row['tokens']} | {row['unique_experts']} | {row['max_expert_count']} |"
        )
    lines.extend(["", "## Gate 结果", "", "| kind | name | fault | gate | rel_rms | cycles | log |", "|---|---|---|---:|---:|---:|---|"])
    for row in result_rows:
        gate = "PASS" if row["gate_passed"] else "FAIL"
        lines.append(
            f"| {row['kind']} | {row['name']} | {row['fault']} | {gate} | {row['rel_rms']} | "
            f"{row['sim_latency_cycles']} | `{row['log']}` |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- `golden_pass` 必须 PASS，证明真实权重、真实路由、非零 activation 下 emulator 输出与 PyTorch golden 对齐。",
            "- `fault_injection` 必须 FAIL，且 runner 把 FAIL 作为“抓住故障”处理；如果任何一个故障仍 PASS，则 gate 覆盖不足。",
            "- 四类故障分别覆盖专家权重基址、专家 id、token 到专家映射、MXFP8 scale 与 element 对齐。",
            "",
            "## 停车点",
            "",
            "阶段一到这里结束。按任务约束，S1 grouped execution 需要人工确认后再开始。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plena-root", type=Path, default=_detect_plena_root())
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--qwen-snapshot", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mlen", type=int, default=128)
    parser.add_argument("--blen", type=int, default=4)
    parser.add_argument("--emu-threads", type=int, default=1)
    parser.add_argument("--skip-faults", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args()

    plena_root = args.plena_root.expanduser().resolve()
    out_dir = (args.out_dir or plena_root / "outputs" / "nonzero_functional_gate").expanduser().resolve()
    qwen_snapshot = (args.qwen_snapshot or plena_root / "weights" / "qwen3-30b-a3b").expanduser().resolve()
    if not qwen_snapshot.exists():
        raise FileNotFoundError(f"Qwen snapshot does not exist: {qwen_snapshot}")

    route_dir = out_dir / "route_traces"
    runs_dir = out_dir / "runs"
    logs_dir = out_dir / "logs"
    scenarios = _choose_scenarios(_routing_files(plena_root))
    scenario_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    trace_paths: dict[str, Path] = {}
    for scenario, source_path, record in scenarios:
        trace_path = _write_trace(
            scenario=scenario,
            record=record,
            out_dir=route_dir,
            mlen=args.mlen,
            blen=args.blen,
            emu_threads=args.emu_threads,
        )
        trace_paths[scenario] = trace_path
        scenario_rows.append(_scenario_row(scenario, source_path, record, trace_path))

    if args.summarize_existing:
        for scenario, _, _ in scenarios:
            run = _existing_run(
                trace_path=trace_paths[scenario],
                build_dir=runs_dir / scenario / "pass",
                fault="none",
                expect_failure=False,
            )
            result_rows.append(_result_row("golden_pass", scenario, run))
        fault_scenarios = {
            "expert_base_plus_one_block": "single_token_top8",
            "swap_first_two_experts": "single_token_top8",
            "shift_token_expert_map": "same_expert_multi_token",
            "scale_element_misaligned": "single_token_top8",
        }
        for fault in FAULTS:
            scenario = fault_scenarios[fault]
            run = _existing_run(
                trace_path=trace_paths[scenario],
                build_dir=runs_dir / "faults" / fault,
                fault=fault,
                expect_failure=True,
            )
            result_rows.append(_result_row("fault_injection", scenario, run))
        write_json(out_dir / "scenario_selection.json", scenario_rows)
        write_json(out_dir / "stage1_summary.json", {"scenarios": scenario_rows, "results": result_rows})
        write_csv(out_dir / "nonzero_functional_gate_results.csv", result_rows)
        _write_report(
            path=plena_root / "analysis" / "NONZERO_FUNCTIONAL_GATE_REPORT.md",
            scenario_rows=scenario_rows,
            result_rows=result_rows,
            qwen_snapshot=qwen_snapshot,
            seed=args.seed,
        )
        print(json.dumps({"out_dir": str(out_dir), "results": result_rows}, indent=2, sort_keys=True))
        return 0

    for scenario, _, _ in scenarios:
        run = _run_trace(
            trace_path=trace_paths[scenario],
            build_dir=runs_dir / scenario / "pass",
            log_path=logs_dir / f"{scenario}.pass.log",
            qwen_snapshot=qwen_snapshot,
            seed=args.seed,
            mlen=args.mlen,
            blen=args.blen,
            emu_threads=args.emu_threads,
        )
        result_rows.append(_result_row("golden_pass", scenario, run))

    if not args.skip_faults:
        fault_scenarios = {
            "expert_base_plus_one_block": "single_token_top8",
            "swap_first_two_experts": "single_token_top8",
            "shift_token_expert_map": "same_expert_multi_token",
            "scale_element_misaligned": "single_token_top8",
        }
        for fault in FAULTS:
            scenario = fault_scenarios[fault]
            run = _run_trace(
                trace_path=trace_paths[scenario],
                build_dir=runs_dir / "faults" / fault,
                log_path=logs_dir / f"fault.{fault}.log",
                qwen_snapshot=qwen_snapshot,
                seed=args.seed,
                mlen=args.mlen,
                blen=args.blen,
                emu_threads=args.emu_threads,
                fault=fault,
                expect_failure=True,
            )
            result_rows.append(_result_row("fault_injection", scenario, run))

    write_json(out_dir / "scenario_selection.json", scenario_rows)
    write_json(out_dir / "stage1_summary.json", {"scenarios": scenario_rows, "results": result_rows})
    write_csv(out_dir / "nonzero_functional_gate_results.csv", result_rows)
    _write_report(
        path=plena_root / "analysis" / "NONZERO_FUNCTIONAL_GATE_REPORT.md",
        scenario_rows=scenario_rows,
        result_rows=result_rows,
        qwen_snapshot=qwen_snapshot,
        seed=args.seed,
    )
    print(json.dumps({"out_dir": str(out_dir), "results": result_rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
