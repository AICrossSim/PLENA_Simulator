"""Restore missing frozen input files from the identical checked-in snapshot.

Existing files are verified, never overwritten. The frozen manifest and all
experimental window lists/hashes remain unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify only; never create files")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "results/E0/frozen_inputs.json").read_text())
    snapshot = root / "results/E0/input_snapshot"
    target = Path(manifest["input_directory"])
    verified = []
    for name, expected in manifest["input_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Input names must be basenames")
        source = snapshot / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError("Snapshot SHA mismatch: " + name)
        destination = target / name
        if not destination.exists():
            if args.check:
                raise FileNotFoundError(destination)
            target.mkdir(parents=True, exist_ok=True)
            # Exclusive creation prevents replacing an existing capture.
            with source.open("rb") as src, destination.open("xb") as dst:
                shutil.copyfileobj(src, dst)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise ValueError("Existing frozen input differs; not overwritten: " + name)
        verified.append(name)
    print(json.dumps({"verified": verified, "input_directory": str(target), "manifest_unchanged": True}))


if __name__ == "__main__":
    main()
