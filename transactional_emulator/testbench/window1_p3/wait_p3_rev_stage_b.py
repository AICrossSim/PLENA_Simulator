#!/usr/bin/env python3
"""Wait for P3-rev Stage A gates, then launch Stage B replay/export.

This is intended to run inside a detached tmux session.  It does not change
timing semantics; it only polls checkpoint files and starts the existing
P3-rev sidecar once route traces and determinism evidence are ready.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from transactional_emulator.testbench.window1_p1.p1_utils import PLENA_ROOT


DEFAULT_OUT_ROOT = PLENA_ROOT / "outputs" / "window1_p3"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{time.time_ns()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _tmux_session_exists(name: str) -> bool:
    if shutil.which("tmux") is None:
        return False
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _kill_tmux_session(name: str) -> int:
    if shutil.which("tmux") is None:
        return 127
    pane_pids = _tmux_pane_pids(name)
    descendant_pids = _descendant_pids(pane_pids)
    rc = subprocess.run(
        ["tmux", "kill-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode
    _terminate_pids(descendant_pids)
    return rc


def _tmux_pane_pids(name: str) -> list[int]:
    if shutil.which("tmux") is None:
        return []
    result = subprocess.run(
        ["tmux", "list-panes", "-t", name, "-F", "#{pane_pid}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if result.returncode != 0:
        return []
    pids = []
    for line in result.stdout.splitlines():
        try:
            pids.append(int(line.strip()))
        except ValueError:
            pass
    return pids


def _descendant_pids(root_pids: list[int]) -> list[int]:
    if not root_pids:
        return []
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid="],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if result.returncode != 0:
        return []
    children: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
    roots = set(root_pids)
    seen: set[int] = set()
    stack = list(root_pids)
    while stack:
        pid = stack.pop()
        for child in children.get(pid, []):
            if child in seen or child in roots:
                continue
            seen.add(child)
            stack.append(child)
    return sorted(seen, reverse=True)


def _terminate_pids(pids: list[int]) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        live = []
        for pid in pids:
            try:
                os.kill(pid, 0)
            except OSError:
                continue
            live.append(pid)
        if not live:
            return
        for pid in live:
            try:
                os.kill(pid, sig)
            except OSError:
                pass
        time.sleep(1)


def _run_progress(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "transactional_emulator.testbench.window1_p3.run_p3_rev_subset",
        "--out-root",
        str(args.out_root),
        "--actions",
        "build,progress,report",
        "--stage-profile",
        "--skip-existing",
        "--keep-going",
        "--prune-success-artifacts",
        "--no-stage-status",
    ]
    subprocess.run(cmd, cwd=args.cwd, check=True)


def _ready(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    _run_progress(args)
    progress = _load_json(args.out_root / "p3_rev_progress_snapshot.json") or {}
    determinism = _load_json(args.out_root / "p3_rev_determinism.json")
    failure_counts = {
        path.name: _line_count(path)
        for path in sorted((args.out_root / "failures").glob("*.jsonl"))
    }
    expected = int(progress.get("expected_representative_runs") or 0)
    traces = int(progress.get("selected_trace_files") or 0)
    route_ready = expected > 0 and traces >= expected
    determinism_ready = (
        (not args.require_determinism)
        or (determinism is not None and determinism.get("status") == "passed")
    )
    failure_free = all(count == 0 for count in failure_counts.values())

    stopped_sessions: dict[str, int] = {}
    if route_ready:
        for name in args.stop_session_on_route_ready:
            if _tmux_session_exists(name):
                stopped_sessions[name] = _kill_tmux_session(name)
        if stopped_sessions:
            time.sleep(2)

    blocking_sessions = [name for name in args.wait_session if _tmux_session_exists(name)]
    sessions_ready = not blocking_sessions
    state = {
        "schema_version": 1,
        "updated_at_unix": time.time(),
        "expected_representative_runs": expected,
        "selected_trace_files": traces,
        "route_ready": route_ready,
        "determinism_status": None if determinism is None else determinism.get("status"),
        "determinism_ready": determinism_ready,
        "failure_counts": failure_counts,
        "failure_free": failure_free,
        "stopped_sessions_on_route_ready": stopped_sessions,
        "blocking_sessions": blocking_sessions,
        "sessions_ready": sessions_ready,
        "ready": route_ready and determinism_ready and failure_free and sessions_ready,
    }
    return bool(state["ready"]), state


def _launch_stage_b(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "-m",
        "transactional_emulator.testbench.window1_p3.run_p3_rev_subset",
        "--out-root",
        str(args.out_root),
        "--actions",
        "replay,export,report",
        "--stage-profile",
        "--skip-existing",
        "--keep-going",
        "--prune-success-artifacts",
        "--width",
        str(args.width),
        "--report-name",
        "p3_rev_report",
    ]
    print("launching Stage B:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=args.cwd).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--cwd", type=Path, default=REPO_ROOT)
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--require-determinism", action="store_true")
    parser.add_argument("--wait-session", action="append", default=["plena_p3_rev_route_build", "plena_p3_rev_gpqa_route"])
    parser.add_argument(
        "--stop-session-on-route-ready",
        action="append",
        default=[],
        help=(
            "tmux session to stop once all route traces are present. "
            "Use this for low-priority partial replay sessions so final Stage B "
            "can safely take over with skip-existing instead of waiting for the "
            "partial queue to drain."
        ),
    )
    parser.add_argument("--max-polls", type=int)
    args = parser.parse_args()

    status_path = args.out_root / "p3_rev_stage_b_waiter_status.json"
    poll = 0
    while True:
        poll += 1
        try:
            ready, state = _ready(args)
        except Exception as exc:
            state = {
                "schema_version": 1,
                "updated_at_unix": time.time(),
                "ready": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            ready = False
        state["poll"] = poll
        state["poll_seconds"] = args.poll_seconds
        _write_json(status_path, state)
        print(json.dumps(state, sort_keys=True), flush=True)
        if ready:
            state["stage_b_started_at_unix"] = time.time()
            _write_json(status_path, state)
            rc = _launch_stage_b(args)
            state["stage_b_return_code"] = rc
            state["stage_b_ended_at_unix"] = time.time()
            _write_json(status_path, state)
            return rc
        if args.max_polls is not None and poll >= args.max_polls:
            return 2
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
