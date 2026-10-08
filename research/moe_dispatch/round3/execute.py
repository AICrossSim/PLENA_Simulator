"""Capture the real command, status, log, and immutable input/source receipts."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
import subprocess
import time
from .common import ROOT, REPO, metadata, sha, write_json, check_round2_readonly
from .config import SEED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("Supply a command after --")
    directory = ROOT / "executions"
    directory.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    receipt = directory / (args.label + "_" + stamp + ".json")
    log = receipt.with_suffix(".log")
    env = os.environ.copy()
    env.update(PYTHONHASHSEED=str(SEED), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    record = metadata({"command": command, "label": args.label,
                       "started_utc": datetime.now(timezone.utc).isoformat(),
                       "returncode": None, "log": str(log.relative_to(ROOT))})
    write_json(receipt, record)
    start = time.monotonic()
    with log.open("w") as output:
        proc = subprocess.run(command, cwd=REPO, env=env, stdout=output,
                              stderr=subprocess.STDOUT)
    check_round2_readonly()
    record.update(returncode=proc.returncode, elapsed_seconds=time.monotonic()-start,
                  finished_utc=datetime.now(timezone.utc).isoformat(), log_sha256=sha(log))
    write_json(receipt, record)
    print({"command": command, "returncode": proc.returncode,
           "elapsed_seconds": record["elapsed_seconds"], "log": str(log)}, flush=True)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
