"""Address-backed numerical audit for the proposed dispatch contract.

This is a small correctness reference, NOT a performance model, trained-model
quality measurement, or guarantee about a hardware BF16 implementation.

Reference order: BF16 round-to-nearest-even operands; within each ascending
K-tile, separate FP32 multiply then a fixed balanced 512-leaf FP32 addition
tree, with absent K lanes zero-padded; add each ascending K segment to the
FP32 accumulator; round gate/up/down outputs to BF16. SiLU and
its product are separately evaluated in NumPy FP32 and separately rounded to
BF16. Down outputs are BF16-rounded values held in FP32 result storage. This explicitly specified
order specifies the prior dot-tree contract but does not claim equivalence
to a fused hardware MAC or a library GEMM/reduction implementation.
No compute operation reads the host weights/input directly: each tile first
passes through address-backed W/X SRAM. All cross-core payload moves use copy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

K_TILE = 512
N_TILE = 4
DTYPE_BYTES = {"bf16": 2, "f32": 4}


def bf16_bits(values: Any) -> np.ndarray:
    a = np.asarray(values, dtype=np.float32)
    u = a.view(np.uint32)
    rounded = (u + np.uint32(0x7FFF) + ((u >> 16) & 1)) >> 16
    # Preserve a NaN rather than rounding a very small NaN payload to infinity.
    nan = ((u & 0x7F800000) == 0x7F800000) & ((u & 0x007FFFFF) != 0)
    return np.where(nan, (u >> 16) | 0x0040, rounded).astype(np.uint16)


def from_bf16(bits: Any) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def bf16(values: Any) -> np.ndarray:
    return from_bf16(bf16_bits(values))


def dot_tree_512(products: np.ndarray) -> np.ndarray:
    """One separately-rounded product per leaf, adjacent balanced FP32 sums."""
    if products.shape[-1] > K_TILE:
        raise ValueError("one physical K tile supports at most 512 leaves")
    values = np.zeros(products.shape[:-1] + (K_TILE,), dtype=np.float32)
    values[..., :products.shape[-1]] = products
    while values.shape[-1] > 1:
        values = np.add(values[..., 0::2], values[..., 1::2], dtype=np.float32)
    return values[..., 0]


@dataclass(frozen=True)
class Tensor:
    owner: int
    memory_id: int
    allocation: int
    name: str
    base: int
    shape: tuple[int, ...]
    dtype: str

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * DTYPE_BYTES[self.dtype]


class SRAM:
    """Finite byte-addressed private memory with lifetime/readiness checks."""
    def __init__(self, owner: int, name: str, capacity: int, counters: Counter):
        self.owner, self.name, self.capacity = owner, name, capacity
        self._data = bytearray(capacity)
        self._initialized = np.zeros(capacity, dtype=np.bool_)
        self._live: dict[int, Tensor] = {}
        self._serial = 0
        self.counters = counters
        self.peak_bytes = 0
        self.peak_address_extent = 0

    def alloc(self, name: str, shape: tuple[int, ...], dtype: str) -> Tensor:
        if dtype not in DTYPE_BYTES or not shape or any(d <= 0 for d in shape):
            raise ValueError("invalid tensor shape or dtype")
        size = int(np.prod(shape)) * DTYPE_BYTES[dtype]
        base = 0
        for t in sorted(self._live.values(), key=lambda a: a.base):
            base = (base + 15) // 16 * 16
            if base + size <= t.base:
                break
            base = t.base + t.nbytes
        base = (base + 15) // 16 * 16
        if base + size > self.capacity:
            raise MemoryError(f"core {self.owner} {self.name}: need {size} bytes at {base}, capacity {self.capacity}")
        self._serial += 1
        t = Tensor(self.owner, id(self), self._serial, name, base, tuple(shape), dtype)
        self._live[t.allocation] = t
        self._initialized[base:base + size] = False
        self.peak_bytes = max(self.peak_bytes, sum(v.nbytes for v in self._live.values()))
        self.peak_address_extent = max(self.peak_address_extent, max(v.base + v.nbytes for v in self._live.values()))
        return t

    def _check(self, actor: int, tensor: Tensor) -> None:
        if actor != self.owner or tensor.owner != self.owner:
            raise PermissionError("private SRAM cannot be read/written by another core")
        if tensor.memory_id != id(self) or self._live.get(tensor.allocation) != tensor:
            raise RuntimeError("wrong memory, released tensor, or stale allocation")

    def _selection(self, tensor: Tensor, key: Any) -> tuple[np.ndarray, tuple[int, ...]]:
        elements = np.arange(int(np.prod(tensor.shape))).reshape(tensor.shape)
        selected = elements if key is None else elements[key]
        width = DTYPE_BYTES[tensor.dtype]
        indices = tensor.base + selected.reshape(-1, 1) * width + np.arange(width)
        return indices.reshape(-1), selected.shape

    def write(self, actor: int, tensor: Tensor, values: Any, key: Any = None) -> None:
        self._check(actor, tensor)
        idx, shape = self._selection(tensor, key)
        a = np.asarray(values, dtype=np.float32)
        if a.shape != shape:
            raise ValueError(f"write shape {a.shape} does not match {shape}")
        encoded = bf16_bits(a).astype('<u2') if tensor.dtype == "bf16" else a.astype('<f4')
        np.frombuffer(self._data, dtype=np.uint8)[idx] = encoded.reshape(-1).view(np.uint8)
        self._initialized[idx] = True
        self.counters[f"{self.name}_write_bytes"] += int(idx.size)

    def read(self, actor: int, tensor: Tensor, key: Any = None) -> np.ndarray:
        self._check(actor, tensor)
        idx, shape = self._selection(tensor, key)
        if not np.all(self._initialized[idx]):
            raise RuntimeError(f"uninitialized payload read: {tensor.name}")
        raw = np.frombuffer(self._data, dtype=np.uint8)[idx].copy()
        self.counters[f"{self.name}_read_bytes"] += int(idx.size)
        decoded = from_bf16(raw.view('<u2')) if tensor.dtype == "bf16" else raw.view('<f4')
        return decoded.reshape(shape).copy()

    def free(self, actor: int, tensor: Tensor) -> None:
        self._check(actor, tensor)
        del self._live[tensor.allocation]
        self._initialized[tensor.base:tensor.base + tensor.nbytes] = False

    @property
    def live_bytes(self) -> int:
        return sum(t.nbytes for t in self._live.values())


def copy_payload(src: SRAM, source: Tensor, dst: SRAM, dest: Tensor,
                 *, src_key: Any = None, dst_key: Any = None,
                 counters: Counter, kind: str) -> None:
    """Authorized explicit transfer: both source read and destination write count."""
    if source.dtype != dest.dtype:
        raise ValueError("copy cannot silently change precision")
    values = src.read(src.owner, source, src_key)
    dst.write(dst.owner, dest, values, dst_key)
    payload = values.size * DTYPE_BYTES[source.dtype]
    counters[f"{kind}_payload_bytes"] += int(payload)
    if src.owner != dst.owner:
        counters["cross_core_payload_bytes"] += int(payload)


class Core:
    def __init__(self, index: int, m: int, workspace_bytes: int, counters: Counter, weight_slots: int = 5):
        self.index, self.m, self.counters = index, m, counters
        self.workspace = SRAM(index, "workspace", workspace_bytes, counters)
        self.x = SRAM(index, "x", 2 * m * K_TILE * 2, counters)
        self.w = SRAM(index, "weight", weight_slots * N_TILE * K_TILE * 2, counters)
        self.x_tile = self.x.alloc("x_tile", (m, K_TILE), "bf16")
        self.w_tile = self.w.alloc("w_tile", (N_TILE, K_TILE), "bf16")

    def stage_input(self, values: np.ndarray) -> Tensor:
        t = self.workspace.alloc("input", values.shape, "bf16")
        self.workspace.write(self.index, t, values)
        self.counters["input_ingress_payload_bytes"] += t.nbytes
        return t

    def project(self, x: Tensor, weights_nk: np.ndarray, n0: int, n1: int,
                name: str, output_dtype: str = "bf16",
                output: Tensor | None = None, output_key: Any = None) -> Tensor:
        """HBM weights are N,K; every multiplication reads staged W/X payload."""
        me, k_size = x.shape
        if weights_nk.shape[1] != k_size or not 0 <= n0 < n1 <= weights_nk.shape[0]:
            raise ValueError("projection dimension mismatch")
        acc = self.workspace.alloc(name + "_fp32", (me, n1 - n0), "f32")
        self.workspace.write(self.index, acc, np.zeros(acc.shape, dtype=np.float32))
        # K-major gives the same per-output K-tile order for every organization.
        for kt in range(0, k_size, K_TILE):
            k_valid = min(K_TILE, k_size - kt)
            for nt in range(n0, n1, N_TILE):
                n_valid = min(N_TILE, n1 - nt)
                weight_tile = np.zeros((N_TILE, K_TILE), dtype=np.float32)
                weight_tile[:n_valid, :k_valid] = weights_nk[nt:nt + n_valid, kt:kt + k_valid]
                self.w.write(self.index, self.w_tile, weight_tile)
                # These are logical BF16 HBM payload bytes, not padded 32B transactions.
                self.counters["hbm_weight_payload_bytes"] += n_valid * k_valid * 2
                self.counters["weight_tiles"] += 1
                for mt in range(0, me, self.m):
                    m_valid = min(self.m, me - mt)
                    input_tile = np.zeros((self.m, K_TILE), dtype=np.float32)
                    input_tile[:m_valid, :k_valid] = self.workspace.read(
                        self.index, x, (slice(mt, mt + m_valid), slice(kt, kt + k_valid)))
                    self.x.write(self.index, self.x_tile, input_tile)
                    actual_x = self.x.read(self.index, self.x_tile)[:m_valid, :k_valid]
                    actual_w = self.w.read(self.index, self.w_tile)[:n_valid, :k_valid]
                    products = np.multiply(actual_x[:, None, :], actual_w[None, :, :], dtype=np.float32)
                    segment = dot_tree_512(products)
                    key = (slice(mt, mt + m_valid), slice(nt - n0, nt - n0 + n_valid))
                    before = self.workspace.read(self.index, acc, key)
                    self.workspace.write(self.index, acc, np.add(before, segment, dtype=np.float32), key)
                    self.counters["useful_macs"] += m_valid * n_valid * k_valid
                    self.counters["issued_mac_slots"] += self.m * N_TILE * K_TILE
                    self.counters["invocations"] += 1
        if output_dtype not in ("bf16", "f32"):
            raise ValueError("unsupported output storage dtype")
        result = output if output is not None else self.workspace.alloc(name + "_rounded", acc.shape, output_dtype)
        if result.dtype != output_dtype:
            raise ValueError("projection output storage dtype mismatch")
        self.workspace.write(self.index, result, bf16(self.workspace.read(self.index, acc)), output_key)
        self.workspace.free(self.index, acc)
        return result

    def activate(self, gate: Tensor, up: Tensor, columns: tuple[int, int] | None = None) -> Tensor:
        """Overwrite the gate arena with Z, after reading both input payloads.

        With paired N splitting the gate arena is full-F sized; only this core's
        local columns are initialized. Unreceived remote columns remain poison.
        """
        key = None if columns is None else (slice(None), slice(*columns))
        g = self.workspace.read(self.index, gate, key)
        u = self.workspace.read(self.index, up)
        if g.shape != u.shape or gate.dtype != "bf16" or up.dtype != "bf16":
            raise ValueError("gate/up columns and BF16 payload must match")
        result = activation(g, u)
        # Both reads are complete before any aliased write or up release.
        self.workspace.write(self.index, gate, result, key)
        self.workspace.free(self.index, up)
        self.counters["gate_z_alias_reuses"] += 1
        return gate


def activation(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    one = np.float32(1.0)
    denominator = np.add(one, np.exp(np.negative(gate), dtype=np.float32), dtype=np.float32)
    silu_bf16 = bf16(np.divide(gate, denominator, dtype=np.float32))
    return np.multiply(silu_bf16, up, dtype=np.float32)


def reference_projection(x: np.ndarray, w_nk: np.ndarray) -> np.ndarray:
    """Independent untiled-M/N reference, identical explicit K arithmetic order."""
    x, w_nk = bf16(x), bf16(w_nk)
    output = np.zeros((x.shape[0], w_nk.shape[0]), dtype=np.float32)
    for start in range(0, x.shape[1], K_TILE):
        # Independent list-of-matrices tree, rather than the tiled 3-D kernel.
        leaves = [np.multiply(x[:, k, None], w_nk[None, :, k], dtype=np.float32)
                  if k < x.shape[1] else np.zeros_like(output)
                  for k in range(start, start + K_TILE)]
        while len(leaves) > 1:
            leaves = [np.add(leaves[i], leaves[i + 1], dtype=np.float32)
                      for i in range(0, len(leaves), 2)]
        output = np.add(output, leaves[0], dtype=np.float32)
    return bf16(output)


def reference_expert(x: np.ndarray, wg: np.ndarray, wu: np.ndarray, wd: np.ndarray) -> np.ndarray:
    g, u = reference_projection(x, wg), reference_projection(x, wu)
    return reference_projection(bf16(activation(g, u)), wd)


@dataclass
class Execution:
    cores: list[Core]
    result: Tensor
    result_core: Core
    counters: Counter
    mode: str

    def output(self) -> np.ndarray:
        # Observation after execution is not another simulated payload movement.
        before = self.counters["workspace_read_bytes"]
        value = self.result_core.workspace.read(self.result_core.index, self.result)
        self.counters["workspace_read_bytes"] = before
        return value

    def summary(self) -> dict[str, Any]:
        values = self.output()
        return {"mode": self.mode, "shape": list(values.shape),
                "sha256_bf16": hashlib.sha256(bf16_bits(values).astype('<u2').tobytes()).hexdigest(),
                "counters": dict(self.counters),
                "core_peaks": [{"core": c.index, "m": c.m,
                                "workspace_bytes": c.workspace.peak_bytes,
                                "workspace_capacity": c.workspace.capacity,
                                "workspace_peak_address_extent": c.workspace.peak_address_extent,
                                "x_bytes": c.x.peak_bytes,
                                "weight_bytes": c.w.peak_bytes} for c in self.cores]}


def aligned_split(size: int, fraction: float = 2 / 3) -> int:
    if size <= N_TILE:
        raise ValueError("split requires at least two nonempty N tiles")
    return min(((size - 1) // N_TILE) * N_TILE,
               max(N_TILE, int(round(size * fraction / N_TILE)) * N_TILE))


def execute_expert(x: np.ndarray, wg: np.ndarray, wu: np.ndarray, wd: np.ndarray,
                   *, mode: str = "whole", core_m: tuple[int, ...] = (4, 2),
                   workspace_bytes: tuple[int, ...] | None = None) -> Execution:
    me, h = x.shape
    f = wg.shape[0]
    if wg.shape != (f, h) or wu.shape != (f, h) or wd.shape != (h, f):
        raise ValueError("expert weights must be gate/up[F,H], down[H,F]")
    if not core_m or any(m <= 0 for m in core_m):
        raise ValueError("invalid core dimensions")
    if workspace_bytes is None:
        total_m = sum(core_m)
        workspace_bytes = tuple((2 * 1024 * 1024 * m) // total_m for m in core_m)
    if len(workspace_bytes) != len(core_m):
        raise ValueError("one private workspace capacity required per core")
    counters: Counter = Counter()
    cores = [Core(i, m, cap, counters, weight_slots=10 if len(core_m) == 1 else 5)
             for i, (m, cap) in enumerate(zip(core_m, workspace_bytes))]
    wg, wu, wd = bf16(wg), bf16(wu), bf16(wd)
    if mode == "whole":
        c = cores[0]
        local_x = c.stage_input(x)
        g, u = c.project(local_x, wg, 0, f, "gate"), c.project(local_x, wu, 0, f, "up")
        z = c.activate(g, u)
        c.workspace.free(c.index, local_x)
        y = c.project(z, wd, 0, h, "down", output_dtype="f32")
        c.workspace.free(c.index, z)
        return Execution(cores, y, c, counters, mode)
    if mode != "split_n" or len(cores) != 2:
        raise ValueError("split_n requires exactly two cores")
    fs = aligned_split(f, core_m[0] / sum(core_m))
    hs = aligned_split(h, core_m[0] / sum(core_m))
    full_z: list[Tensor] = []
    for c, (start, end) in zip(cores, ((0, fs), (fs, f))):
        local_x = c.stage_input(x)
        gate_z = c.workspace.alloc("gate_z_alias", (me, f), "bf16")
        c.project(local_x, wg, start, end, "gate", output=gate_z,
                  output_key=(slice(None), slice(start, end)))
        u = c.project(local_x, wu, start, end, "up")
        full_z.append(c.activate(gate_z, u, columns=(start, end)))
        c.workspace.free(c.index, local_x)
    # Local Z already occupies its final arena. Remote slices are still poison;
    # two explicit copies make the full down-input readable on each core.
    for src_i, (start, end) in enumerate(((0, fs), (fs, f))):
        dst_i = 1 - src_i
        key = (slice(None), slice(start, end))
        copy_payload(cores[src_i].workspace, full_z[src_i], cores[dst_i].workspace, full_z[dst_i],
                     src_key=key, dst_key=key, counters=counters, kind="z_exchange")
    partial_y = []
    for c, z, (start, end) in zip(cores, full_z, ((0, hs), (hs, h))):
        partial_y.append(c.project(z, wd, start, end, "down", output_dtype="f32"))
        c.workspace.free(c.index, z)
    # Explicit destination: core 0's private workspace, not an implicit shared array.
    c0 = cores[0]
    y = c0.workspace.alloc("gathered_y", (me, h), "f32")
    for i, (start, end) in enumerate(((0, hs), (hs, h))):
        copy_payload(cores[i].workspace, partial_y[i], c0.workspace, y,
                     dst_key=(slice(None), slice(start, end)), counters=counters,
                     kind="y_local" if i == 0 else "y_gather")
        cores[i].workspace.free(cores[i].index, partial_y[i])
    return Execution(cores, y, c0, counters, mode)


@dataclass
class Contribution:
    rank: int | tuple[int, ...]
    source: SRAM
    tensor: Tensor
    token_ids: tuple[int, ...]
    route_weights: np.ndarray
    is_shared: bool = False


def combine_rank_order(destination: Core, contributions: list[Contribution],
                       token_count: int, hidden: int) -> Tensor:
    """Frozen router-rank FP32 sums -> BF16 -> add shared -> BF16.

    This is an explicit reference reduction order, not a promise that a library
    torch.sum implementation makes identical FP32 rounding choices. Final BF16
    values are widened to the FP32 storage format used by result inboxes.
    """
    ordered_rows, shared_rows = [], []
    seen, shared_tokens = set(), set()
    for index, item in enumerate(contributions):
        if item.tensor.shape != (len(item.token_ids), hidden) or item.route_weights.shape != (len(item.token_ids),):
            raise ValueError("combine contribution shape mismatch")
        if len(set(item.token_ids)) != len(item.token_ids) or any(t < 0 or t >= token_count for t in item.token_ids):
            raise ValueError("invalid route token index")
        ranks = (item.rank,) * len(item.token_ids) if isinstance(item.rank, int) else item.rank
        if len(ranks) != len(item.token_ids):
            raise ValueError("one router rank required per token row")
        for row, (token_id, rank) in enumerate(zip(item.token_ids, ranks)):
            if item.is_shared:
                if token_id in shared_tokens or item.route_weights[row] != 1:
                    raise ValueError("one unweighted shared output per token required")
                shared_tokens.add(token_id)
                shared_rows.append((token_id, index, row))
            else:
                if rank < 0 or (token_id, rank) in seen:
                    raise ValueError("duplicate or negative per-token router rank")
                seen.add((token_id, rank))
                ordered_rows.append((rank, token_id, index, row))
    result = destination.workspace.alloc("combined_fp32", (token_count, hidden), "f32")
    destination.workspace.write(destination.index, result, np.zeros(result.shape, dtype=np.float32))

    def land_row(item: Contribution, row: int) -> tuple[Tensor, np.ndarray]:
        landed = destination.workspace.alloc("combine_landed_row", (1, hidden), item.tensor.dtype)
        copy_payload(item.source, item.tensor, destination.workspace, landed,
                     src_key=(slice(row, row + 1), slice(None)),
                     counters=destination.counters, kind="combine_transfer")
        return landed, destination.workspace.read(destination.index, landed)

    for _, token_id, index, row in sorted(ordered_rows):
        item = contributions[index]
        landed, payload = land_row(item, row)
        key = (slice(token_id, token_id + 1), slice(None))
        old = destination.workspace.read(destination.index, result, key)
        product = np.multiply(payload, np.float32(item.route_weights[row]), dtype=np.float32)
        destination.workspace.write(destination.index, result, np.add(old, product, dtype=np.float32), key)
        destination.workspace.free(destination.index, landed)
    # Mandatory boundary: route reduction is rounded before shared addition.
    destination.workspace.write(destination.index, result,
                                bf16(destination.workspace.read(destination.index, result)))
    for token_id, index, row in sorted(shared_rows):
        landed, payload = land_row(contributions[index], row)
        key = (slice(token_id, token_id + 1), slice(None))
        old = destination.workspace.read(destination.index, result, key)
        destination.workspace.write(destination.index, result,
                                    bf16(np.add(old, payload, dtype=np.float32)), key)
        destination.workspace.free(destination.index, landed)
    return result


def validation_report(seed: int = 20260928) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    cases = []
    for me, h, f in ((1, 13, 9), (2, 17, 11), (3, 13, 7), (4, 513, 9), (4, 9, 513)):
        x = rng.normal(0, 0.15, (me, h)).astype(np.float32)
        weights = [rng.normal(0, 0.1, shape).astype(np.float32) for shape in ((f, h), (f, h), (h, f))]
        ref = reference_expert(x, *weights)
        executions = [execute_expert(x, *weights, core_m=(6,), mode="whole"),
                      execute_expert(x, *weights, core_m=(3, 3), mode="whole"),
                      execute_expert(x, *weights, core_m=(4, 2), mode="whole"),
                      execute_expert(x, *weights, core_m=(3, 3), mode="split_n"),
                      execute_expert(x, *weights, core_m=(4, 2), mode="split_n")]
        for execution in executions:
            if not np.array_equal(bf16_bits(execution.output()), bf16_bits(ref)):
                raise AssertionError(f"bit mismatch: Me/H/F={me}/{h}/{f}, {execution.mode}")
            if execution.counters["hbm_weight_payload_bytes"] != 6 * h * f:
                raise AssertionError("weight payload duplicated or missing")
            if execution.mode == "split_n" and execution.counters["z_exchange_payload_bytes"] != 2 * me * f:
                raise AssertionError("Z cross-core payload mismatch")
        cases.append({"me": me, "h": h, "f": f, "bit_exact": True,
                      "executions": [execution.summary() for execution in executions]})
    return {"scope": "small explicit-SRAM reference arithmetic audit; not hardware/model quality validation",
            "seed": seed, "numpy_version": np.__version__, "cases": cases,
            "case_count": len(cases), "execution_count": sum(len(c["executions"]) for c in cases),
            "all_bit_exact": True,
            "timing_measured": False,
            "weight_byte_semantics": "logical BF16 payload; not native 32-byte burst/row padding",
            "workspace_scope": "standalone expert numerical scratch only; compiler separately reserves global outputs/control",
            "arithmetic_contract": "BF16 RNE X/W, balanced FP32 tree512 with zero tail then ascending FP32 K-segment adds, gate/up BF16, SiLU BF16, product BF16, down BF16-rounded in FP32 storage, router-rank FP32 sum -> BF16 -> add shared -> BF16 in FP32 storage"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="write the JSON audit report to this path")
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    report = validation_report(args.seed)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
