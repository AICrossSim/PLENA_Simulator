"""Memory-only Ramulator backend for the compositional analytical model.

The Python model determines addresses and non-memory delays. This backend runs
no numerical operations and never reads Rust timing observations. Sharing the
DRAM implementation validates integration, not the DRAM model independently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import uuid


def prepare_backend(output, controllers=8):
    """Build the memory-only reference in the repo's Nix development shell.

    This explicit profile matches the runner's eight-controller HBM2 preset.
    It does not select a new global hardware configuration or physical stack
    count. The generated artifact records the exact compiler/config hashes.
    """
    if controllers not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError("unsupported HBM controller count")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("ltile_memory.cc")
    profile = Path(__file__).with_name("profiles") / "hbm2_8controllers_1ghz.json"
    binary = output / "ltile_memory"
    command = ["c++", "-std=c++17", "-O2", str(source), "-lramulator", "-o", str(binary)]
    subprocess.run(command, check=True)
    config = output / "ramulator.json"
    if controllers == 8:
        config.write_bytes(profile.read_bytes())
    else:
        geometry = json.loads(profile.read_text())
        preset = geometry["memory_system"]["controllers"][0]
        geometry["memory_system"]["controllers"] = [preset for _ in range(controllers)]
        config.write_text(json.dumps(geometry) + "\n")
    (output / "cache").mkdir()
    manifest = dict(command=command, source_sha256=sha(source), binary_sha256=sha(binary), config_sha256=sha(config))
    (output / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DmaBackend:
    def __init__(self, binary, config, cache):
        self.binary, self.config, self.cache = map(Path, (binary, config, cache))
        self.cache.mkdir(parents=True, exist_ok=True)
        self.identity = dict(binary_sha256=sha(binary), config_sha256=sha(config))

    def price(self, cost, service):
        if not cost.memory_trace:
            raise ValueError("address/delay trace required; aggregate bytes are insufficient")
        trace = "".join(f"{op} {addr} {size}\n" for op, addr, size in cost.memory_trace)
        identity = dict(self.identity, service=service, trace_sha256=hashlib.sha256(trace.encode()).hexdigest())
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        output = self.cache / f"{key}.json"
        if output.exists():
            result = json.loads(output.read_text())
        else:
            unique = uuid.uuid4().hex
            path = self.cache / f"{key}.{unique}.trace"
            path.write_text(trace)
            try:
                run = subprocess.run(
                    [str(self.binary.resolve()), str(self.config.resolve()), str(path.resolve()), str(service)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                result = dict(json.loads(run.stdout), **identity)
                temporary = self.cache / f"{key}.{unique}.tmp"
                temporary.write_text(json.dumps(result, indent=2) + "\n")
                temporary.replace(output)
            finally:
                path.unlink(missing_ok=True)
        cost.dma = result["dma_cycles"]
        if cost.total != result["total_cycles"]:
            raise AssertionError("compute/delay ledger does not reconcile with memory time")
        expected_write = sum(n * c for (d, n), c in cost.transfers.items() if d == "write")
        expected_read = sum(n * c for (d, n), c in cost.transfers.items() if d == "read")
        if service == "review":
            expected_read += expected_write
        if (expected_read, expected_write) != (result["read_bytes"], result["write_bytes"]):
            raise AssertionError("DMA trace traffic differs from compiler transfer counts")
        if cost.sections:
            observed = result.get("sections", [])
            if len(observed) != len(cost.sections):
                raise ValueError("memory backend must support operator section markers")
            for section, timing in zip(cost.sections, observed):
                section["dma"] = timing["dma_cycles"]
                section["total"] += timing["dma_cycles"]
                section["hbm_read_bytes"] = timing["read_bytes"]
                section["hbm_write_bytes"] = timing["write_bytes"]
                if section["total"] != timing["total_cycles"]:
                    raise AssertionError("operator ledger differs from memory trace")
            if sum(s["total"] for s in cost.sections) != cost.total:
                raise AssertionError("operator sections do not cover the entire program")
        return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the pinned HBM2 timing reference in a fresh artifact directory")
    parser.add_argument("--prepare", required=True, type=Path)
    parser.add_argument("--controllers", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(prepare_backend(args.prepare, args.controllers), indent=2))
