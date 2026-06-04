#!/usr/bin/env python3
"""Aggregate TTFT (prefill) vs TPOT (decode) latency from a sweep_ttft_tpot.sh run.

Scans emulator run-stats and emits BOTH a CSV and a markdown table with columns:
    board, preset, case(lang/vision), phase(TTFT/TPOT), latency_ns, cycles
plus a pivoted markdown table with TTFT and TPOT side-by-side per (board,preset,case).

Sources, in priority order (a stats blob is keyed by board/preset/case/phase, so a
later source never overwrites an earlier one):
  1. build/ttft_tpot_results/*.json  -- board-tagged snapshots written by the sweep
     (each carries a "__sweep__" meta block with board + clock_mhz). Preferred,
     because the build-dir name does NOT encode the board and the A7 boards share
     an identical 100 MHz profile, so raw build dirs alone cannot tell them apart.
  2. build/*/rust_emulator_run_stats.json with an adjacent sweep_meta.json sidecar.
  3. build/*/rust_emulator_run_stats.json with NO sidecar -- board parsed as "?"
     (clock inferred: only useful for single-board manual runs).

cycles = sim_latency_ns / ns_per_cycle, ns_per_cycle = 1000 / clock_mhz
(10 ns @ 100 MHz, 2.5 ns @ 400 MHz). Clock comes from the sweep meta, else from the
board YAML (board_configs/<board>.yaml clock_mhz), else defaults to 100 MHz.

Usage:
    python aggregate_ttft_tpot.py [--build-dir DIR] [--out-prefix PREFIX]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BUILD = SCRIPT_DIR / "build"
BOARD_CFG_DIR = SCRIPT_DIR / "board_configs"

# native_64x64x16_b1 / sliced_... etc. dir: <nick>_<mode>_<MxNxK>_b<N>_<case>[_decode_p<past>]
_DIR_RE = re.compile(
    r"^(?P<nick>[^_]+)_(?P<mode>native|sliced)_(?P<preset>\d+x\d+x\d+_b\d+)_"
    r"(?P<case>decoder|vision-layers|vision-connector|vlm-e2e|ffn|decoder-layer|decoder-chain)"
    r"(?:_decode_p(?P<past>\d+))?$"
)

# case -> friendly label
_CASE_LABEL = {"decoder": "lang", "vision-layers": "vision"}

_CLOCK_CACHE: dict[str, float] = {}


def _board_clock_mhz(board: str | None) -> float:
    """Clock for a board, read from board_configs/<board>.yaml (cached). Default 100."""
    if not board or board == "?":
        return 100.0
    if board in _CLOCK_CACHE:
        return _CLOCK_CACHE[board]
    clk = 100.0
    yml = BOARD_CFG_DIR / f"{board}.yaml"
    if yml.exists():
        try:
            import yaml

            with yml.open() as f:
                cfg = yaml.safe_load(f) or {}
            clk = float(cfg.get("clock_mhz", 100.0))
        except Exception:
            clk = 100.0
    _CLOCK_CACHE[board] = clk
    return clk


def _phase_label(phase: str | None, past_len: int) -> str:
    """prefill -> TTFT, decode -> TPOT."""
    if phase == "decode" or past_len > 0:
        return "TPOT"
    return "TTFT"


def _parse_dir_name(name: str) -> dict | None:
    m = _DIR_RE.match(name)
    if not m:
        return None
    g = m.groupdict()
    past = int(g["past"]) if g["past"] else 0
    return {
        "preset": f"{g['mode']}_{g['preset']}",
        "case": g["case"],
        "past_len": past,
        "phase": "decode" if past > 0 else "prefill",
    }


def _record(board, preset, case, phase, past_len, latency_ns, source):
    clock = _board_clock_mhz(board)
    ns_per_cycle = 1000.0 / clock
    cycles = latency_ns / ns_per_cycle if latency_ns is not None else None
    return {
        "board": board or "?",
        "preset": preset,
        "case": _CASE_LABEL.get(case, case),
        "phase": _phase_label(phase, past_len),
        "latency_ns": latency_ns,
        "cycles": cycles,
        "clock_mhz": clock,
        "_case_raw": case,
        "_source": source,
    }


def collect(build_dir: Path) -> list[dict]:
    """Collect one record per (board, preset, case, phase). Earlier sources win."""
    by_key: dict[tuple, dict] = {}

    def add(rec: dict):
        # Defense-in-depth: decode/TPOT is decoder-only. The vision encoder has no
        # KV cache and ignores past_len, so a stray vision-decode snapshot is a
        # degenerate seq_len=1 vision prefill mislabeled as TPOT -- drop it so it
        # can never surface as a "vision TPOT" row.
        if rec["phase"] == "TPOT" and rec["_case_raw"] != "decoder":
            return
        key = (rec["board"], rec["preset"], rec["case"], rec["phase"])
        # priority: snapshot (1) < sidecar (2) < bare (3); lower number wins.
        prio = {"snapshot": 1, "sidecar": 2, "bare": 3}[rec["_source"]]
        existing = by_key.get(key)
        if existing is None or prio < {"snapshot": 1, "sidecar": 2, "bare": 3}[existing["_source"]]:
            by_key[key] = rec

    # 1. board-tagged snapshots
    results_dir = build_dir / "ttft_tpot_results"
    if results_dir.is_dir():
        for jp in sorted(results_dir.glob("*.json")):
            try:
                blob = json.loads(jp.read_text())
            except Exception:
                continue
            meta = blob.get("__sweep__")
            if not meta:
                continue
            add(
                _record(
                    board=meta.get("board"),
                    preset=meta.get("preset"),
                    case=meta.get("case"),
                    phase=meta.get("phase"),
                    past_len=int(meta.get("past_len", 0) or 0),
                    latency_ns=blob.get("sim_latency_ns"),
                    source="snapshot",
                )
            )

    # 2/3. raw build dirs with optional sidecar
    for stats_p in sorted(build_dir.glob("*/rust_emulator_run_stats.json")):
        d = stats_p.parent
        try:
            blob = json.loads(stats_p.read_text())
        except Exception:
            continue
        latency_ns = blob.get("sim_latency_ns")
        sidecar = d / "sweep_meta.json"
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text())
            except Exception:
                meta = {}
            add(
                _record(
                    board=meta.get("board"),
                    preset=meta.get("preset"),
                    case=meta.get("case"),
                    phase=meta.get("phase"),
                    past_len=int(meta.get("past_len", 0) or 0),
                    latency_ns=latency_ns,
                    source="sidecar",
                )
            )
            continue
        parsed = _parse_dir_name(d.name)
        if not parsed:
            continue
        add(
            _record(
                board=None,  # board not encoded in dir name
                preset=parsed["preset"],
                case=parsed["case"],
                phase=parsed["phase"],
                past_len=parsed["past_len"],
                latency_ns=latency_ns,
                source="bare",
            )
        )

    rows = list(by_key.values())
    rows.sort(key=lambda r: (r["board"], r["preset"], r["case"], r["phase"]))
    return rows


def _fmt_ns(v):
    return "" if v is None else f"{v:.0f}"


def _fmt_cyc(v):
    return "" if v is None else f"{v:.0f}"


def write_csv(rows: list[dict], path: Path) -> None:
    cols = ["board", "preset", "case", "phase", "latency_ns", "cycles", "clock_mhz"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(
                [
                    r["board"],
                    r["preset"],
                    r["case"],
                    r["phase"],
                    _fmt_ns(r["latency_ns"]),
                    _fmt_cyc(r["cycles"]),
                    f"{r['clock_mhz']:.0f}",
                ]
            )


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def write_markdown(rows: list[dict], path: Path) -> None:
    parts: list[str] = ["# TTFT vs TPOT decode-latency sweep", ""]

    # flat table
    parts.append("## Per-run latency")
    parts.append("")
    flat = [
        [
            r["board"],
            r["preset"],
            r["case"],
            r["phase"],
            _fmt_ns(r["latency_ns"]),
            _fmt_cyc(r["cycles"]),
        ]
        for r in rows
    ]
    parts.append(_md_table(["board", "preset", "case", "phase", "latency_ns", "cycles"], flat))
    parts.append("")

    # pivot: TTFT and TPOT side by side per (board, preset, case)
    parts.append("## TTFT vs TPOT (side by side)")
    parts.append("")
    pivot: dict[tuple, dict] = {}
    for r in rows:
        key = (r["board"], r["preset"], r["case"])
        slot = pivot.setdefault(key, {})
        slot[r["phase"]] = r
    prows = []
    for (board, preset, case) in sorted(pivot):
        ttft = pivot[(board, preset, case)].get("TTFT")
        tpot = pivot[(board, preset, case)].get("TPOT")
        prows.append(
            [
                board,
                preset,
                case,
                _fmt_ns(ttft["latency_ns"]) if ttft else "",
                _fmt_cyc(ttft["cycles"]) if ttft else "",
                _fmt_ns(tpot["latency_ns"]) if tpot else "",
                _fmt_cyc(tpot["cycles"]) if tpot else "",
            ]
        )
    parts.append(
        _md_table(
            [
                "board",
                "preset",
                "case",
                "TTFT_latency_ns",
                "TTFT_cycles",
                "TPOT_latency_ns",
                "TPOT_cycles",
            ],
            prows,
        )
    )
    parts.append("")
    path.write_text("\n".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default=str(DEFAULT_BUILD), help="testbench/build directory to scan")
    ap.add_argument(
        "--out-prefix",
        default=None,
        help="output path prefix (default: <build-dir>/ttft_tpot_results/ttft_tpot)",
    )
    args = ap.parse_args()

    build_dir = Path(args.build_dir).resolve()
    rows = collect(build_dir)

    if args.out_prefix:
        prefix = Path(args.out_prefix)
    else:
        out_dir = build_dir / "ttft_tpot_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = out_dir / "ttft_tpot"
    prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    print(f"Collected {len(rows)} run(s) from {build_dir}")
    print(f"CSV : {csv_path}")
    print(f"MD  : {md_path}")
    if rows:
        # echo the markdown to stdout for quick eyeballing
        print()
        print(md_path.read_text())
    else:
        print("No rust_emulator_run_stats.json found -- run sweep_ttft_tpot.sh first.")


if __name__ == "__main__":
    main()
