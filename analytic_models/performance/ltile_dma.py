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
            path = self.cache / f"{key}.trace"
            path.write_text(trace)
            try:
                run = subprocess.run(
                    [str(self.binary.resolve()), str(self.config.resolve()), str(path.resolve()), str(service)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                result = dict(json.loads(run.stdout), **identity)
                output.write_text(json.dumps(result, indent=2) + "\n")
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
        return result
