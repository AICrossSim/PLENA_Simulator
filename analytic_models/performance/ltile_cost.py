"""Finite-resource analytical cost of compiled recurrent programs.

No tensors, Rust results, or per-configuration measured cycles are used by the
predictor. The compiler's assembly supplies addresses, shapes and loop bounds.
Each primitive is priced by a bounded resource schedule. DMA service parameters
are calibrated separately; measured instruction counts are never model inputs.

This is the accepted R3 execution contract: serial instruction retirement,
shared Matrix packet service, independent Vector service, and overlap only
inside a primitive. It is not a claim of an integrated RTL implementation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import math
import re


@dataclass(frozen=True)
class Machine:
    lanes: int = 256
    update_ii: int = 2
    update_latency: int = 6
    dot_ii: int = 2
    dot_latency: int = 6
    sram_cycles: int = 1
    context_cycles: int = 1
    dot: str = "tree"
    banks: int = 64
    bank_width: int = 32
    vector_width: int = 2048
    vector_reciprocal_cycles: int = 2
    clock_hz: int = 1_000_000_000

    def __post_init__(self):
        if self.lanes not in (128, 256, 512):
            raise ValueError("validated lane search is 128/256/512")
        if self.dot not in ("tree", "fp32"):
            raise ValueError("dot must be tree or fp32")
        if (
            min(
                self.update_ii,
                self.update_latency,
                self.sram_cycles,
                self.dot_ii,
                self.dot_latency,
                self.context_cycles,
                self.vector_reciprocal_cycles,
                self.clock_hz,
            )
            < 1
        ):
            raise ValueError("resource service times must be positive")
        if (self.banks, self.bank_width, self.vector_width) != (64, 32, 2048):
            raise ValueError("only the audited 64 x 32 Matrix / 2048 Vector geometry is supported")


@dataclass(frozen=True)
class View:
    rows: int
    cols: int
    heads: int
    pitch: int
    phase: int
    broadcast: bool

    @classmethod
    def decode(cls, shape, mapping):
        if mapping & (63 << 16) or (mapping >> 28) & 7:
            raise ValueError("reserved Matrix-view bits")
        return cls(
            (shape & 4095) + 1,
            ((shape >> 12) & 4095) + 1,
            (shape >> 24) + 1,
            mapping & 65535,
            (mapping >> 22) & 63,
            bool(mapping & (1 << 31)),
        )

    @property
    def values(self):
        return self.rows * self.cols * self.heads


@lru_cache(maxsize=32768)
def bank_service(base, view, lines=None):
    """Distinct addressed words per bank, under the fixed diagonal mapping."""
    words = view.cols // 32
    if view.cols % 32:
        raise ValueError("views must contain whole bank words")
    counts = [0] * 64
    seen = set()
    if lines is None:
        lines = ((h, r) for h in range(view.heads) for r in range(view.rows))
    for h, r in lines:
        for word in range(words):
            address = base // 2048 + h * view.pitch + r * math.ceil(words / 64) + word // 64
            bank = (base % 2048 // 32 + address + h * view.phase + word) % 64
            key = (bank, address)
            if key not in seen:
                seen.add(key)
                counts[bank] += 1
    return max(1, max(counts)), len(seen)


class Schedule:
    """Integer bitsets represent occupied service slots, not runtime queues."""

    def __init__(self):
        self.ports = dict(matrix=0, vector=0, context_r=0, context_w=0)
        self.arithmetic = 0
        self.finish = self.next_launch = 0
        self.counts = Counter()

    def reserve(self, resource, ready, duration, count=None, words=0):
        duration = max(1, duration)
        occupied = self.ports[resource]
        mask = (1 << duration) - 1
        t = ready
        while overlap := ((occupied >> t) & mask):
            t += (overlap & -overlap).bit_length()
        self.ports[resource] = occupied | (mask << t)
        self.finish = max(self.finish, t + duration)
        if count:
            self.counts[count] += 1
        if resource == "matrix":
            self.counts["matrix_words"] += words
        return t + duration

    def compute(self, ready, latency, ii, feedback=0, existing=False):
        possible = ready if existing else max(ready, self.next_launch)
        t = max(possible, feedback)
        self.counts["feedback_wait"] += t - possible
        if not existing:
            self.next_launch = t + ii
            self.counts["launches"] += 1
        self.arithmetic |= ((1 << latency) - 1) << t
        self.finish = max(self.finish, t + latency)
        return t + latency

    def totals(self):
        bank_mask = 0
        for mask in self.ports.values():
            bank_mask |= mask
        bank = bank_mask.bit_count()
        arithmetic = (self.arithmetic & ~bank_mask).bit_count()
        return bank, arithmetic, self.finish - bank - arithmetic, dict(self.counts)


@lru_cache(maxsize=32768)
def primitive_cost(machine, op, dst, src, coeff, db, sb, cb, reduction_rows):
    """Shape-derived stage schedule; arithmetic values do not enter timing."""
    m = machine
    width, heads = dst.cols, dst.heads
    if width > m.lanes or m.lanes % width:
        raise ValueError("partial head subgroups are outside this contract")
    group = m.lanes // width
    c = Schedule()

    def matrix(base, view, lines, ready, write=False):
        cycles, words = bank_service(base, view, tuple(lines))
        return c.reserve("matrix", ready, cycles * m.sram_cycles, "matrix_writes" if write else "matrix_reads", words)

    def vector(ready, write=False):
        return c.reserve("vector", ready, m.sram_cycles, "vector_writes" if write else "vector_reads")

    def context(ready, write=False):
        return c.reserve(
            "context_w" if write else "context_r",
            ready,
            m.context_cycles,
            "context_writes" if write else "context_reads",
        )

    if op == 4:
        if m.dot == "fp32":
            for _ in range(math.ceil(heads * width / m.lanes)):
                context(0, True)
        return (*c.totals(), 0)

    if op in (7, 8):
        if reduction_rows < 1 or reduction_rows & (reduction_rows - 1):
            raise ValueError("reduction must finish on a power of two")
        scalar_ready = matrix(cb, coeff, [(0, 0)], 0) if op == 8 else 0
        for first in range(0, heads, group):
            lines = [(h, 0) for h in range(first, min(heads, first + group))]
            ready = scalar_ready
            if m.dot == "tree":
                ready = vector(ready)
            if op == 8:
                ready = matrix(sb, src, lines, ready)
            if m.dot == "fp32":
                ready = context(ready)
            done = c.compute(ready, 4 if op == 8 else 1, m.dot_ii)
            matrix(db, dst, lines, done, True)
        return (*c.totals(), 0)

    if op == 0:
        # Legacy affine interface drains each L-wide packet and reads its
        # operands explicitly; it cannot get a free 2048-wide FP32 datapath.
        bank = arithmetic = 0
        for row in range(dst.rows):
            for first in range(0, heads, group):
                lines = tuple((h, row) for h in range(first, min(heads, first + group)))
                source_lines = tuple((0 if src.heads == 1 else h, 0 if src.rows == 1 else row) for h, _ in lines)
                scalar_lines = ((0, row),) if coeff.heads == 1 else lines
                bank += 2 * bank_service(db, dst, lines)[0]
                bank += bank_service(sb, src, source_lines)[0]
                bank += bank_service(cb, coeff, scalar_lines)[0]
                arithmetic += m.update_latency
        return bank, arithmetic, 0, {}, reduction_rows

    if op not in (3, 5, 6):
        raise ValueError(f"primitive {op} is not in the v2 cost contract")
    reducing = op in (5, 6)
    rows = src.rows if reducing else dst.rows
    if reducing and reduction_rows + rows > 128:
        raise ValueError("reduction exceeds 128 rows")
    latency = (
        m.update_latency
        if op == 3
        else (m.update_latency + 3 if op == 6 else 3)
        if m.dot == "tree"
        else m.dot_latency + (m.update_latency if op == 6 else 0)
    )
    interval = m.update_ii if op == 3 else m.dot_ii
    credits = [0] * math.ceil(latency / interval)
    feedback = [0] * math.ceil(heads * width / m.lanes)
    supply = row_ready = launches = 0
    for row in range(rows):
        compact_ready = (
            matrix(cb, coeff, [(0, 0 if coeff.rows == 1 else row)], max(row_ready, supply))
            if coeff.broadcast
            else row_ready
        )
        leaves_ready = row_ready
        for first in range(0, heads, group):
            last = min(heads, first + group)
            chunk, slot = first * width // m.lanes, launches % len(credits)
            operands_ready = max(row_ready, supply, credits[slot])
            lines = tuple((h, row) for h in range(first, last))
            ready = matrix(sb if reducing else db, src if reducing else dst, lines, operands_ready)
            if coeff.broadcast:
                scalar_ready = compact_ready
            else:
                sliced = View(1, 2 * (last - first) * width, 1, coeff.pitch, coeff.phase, False)
                offset = row * math.ceil(coeff.cols / 2048) * 2048 + 2 * first * width
                scalar_ready = matrix(cb + offset, sliced, [(0, 0)], operands_ready)
            ready = max(ready, scalar_ready)
            if op == 3:
                sources = tuple((h, 0 if src.rows == 1 else row) for h in range(first, last))
                ready = max(ready, matrix(sb, src, sources, operands_ready))
                done = c.compute(ready, m.update_latency, m.update_ii)
                credits[slot] = matrix(db, dst, lines, done, True)
            elif m.dot == "tree":
                done = c.compute(ready, latency, m.dot_ii)
                credits[slot] = vector(done, True)
                leaves_ready = max(leaves_ready, credits[slot])
            else:
                c.counts["feedback_wait"] += max(0, feedback[chunk] - ready)
                ready = context(max(ready, feedback[chunk]))
                done = c.compute(ready, latency, m.dot_ii, feedback[chunk])
                credits[slot] = feedback[chunk] = context(done, True)
            launches += 1
            supply = done - latency
        if reducing:
            if m.dot == "tree":
                ready = leaves_ready
                merges = (reduction_rows ^ (reduction_rows + 1)).bit_length() - 1
                for _ in range(merges):
                    ready = vector(vector(ready))
                    ready = c.compute(ready, 1, 1, existing=True)
                    ready = vector(ready, True)
                row_ready = ready
            reduction_rows += 1
    return (*c.totals(), reduction_rows)


@dataclass
class ProgramCost:
    issue: int = 0
    scalar: int = 0
    sram: int = 0
    arithmetic: int = 0
    dependency: int = 0
    dma: float = 0.0
    transfers: Counter = field(default_factory=Counter)
    accesses: Counter = field(default_factory=Counter)
    opcodes: Counter = field(default_factory=Counter)
    memory_trace: list = field(default_factory=list)

    @property
    def total(self):
        return self.issue + self.scalar + self.sram + self.arithmetic + self.dependency + self.dma

    def components(self):
        return dict(
            issue=self.issue,
            scalar=self.scalar,
            frontend=self.issue + self.scalar,
            sram=self.sram,
            arithmetic=self.arithmetic,
            dependency=self.dependency,
            dma=self.dma,
            total=self.total,
        )


def assembly_cost(assembly: str, machine: Machine = Machine(), *, trace_memory=False):
    """Evaluate only instruction control/addresses, not model arithmetic."""
    code = []
    for raw in assembly.splitlines():
        line = raw.split(";")[0].strip()
        if line:
            op, *args = re.split(r"[\s,]+", line)
            code.append((op, args))
    cost = ProgramCost()
    registers = [0] * 16
    views = {}
    loops = []
    reduction_rows = 0
    pc = 0
    traced_cycles = 0
    while pc < len(code):
        op, args = code[pc]

        def gp(s):
            if not s.startswith("gp"):
                raise ValueError(f"expected GP register, got {s}")
            return int(s[2:])

        def value(s):
            return registers[gp(s)]

        cost.issue += 1
        cost.opcodes[op] += 1
        if op.startswith("S_") or op.startswith("C_"):
            cost.scalar += 1
        if op == "S_LUI_INT":
            registers[gp(args[0])] = (int(args[1], 0) << 12) & 0xFFFFFFFF
        elif op == "S_ADDI_INT":
            registers[gp(args[0])] = (value(args[1]) + int(args[2], 0)) & 0xFFFFFFFF
        elif op == "C_LOOP_START":
            if int(args[1], 0) <= 0:
                raise ValueError("loop must have a positive finite trip count")
            loops.append([pc + 1, int(args[1], 0)])
        elif op == "C_LOOP_END":
            loops[-1][1] -= 1
            if loops[-1][1]:
                pc = loops[-1][0]
                continue
            loops.pop()
        elif op == "L_TILE_CFG":
            views[int(args[0])] = View.decode(value(args[1]), value(args[2]))
            cost.scalar += 1
        elif op == "L_TILE_EXEC":
            primitive = int(args[3], 0)
            bank, arith, dep, counts, reduction_rows = primitive_cost(
                machine,
                primitive,
                views[0],
                views[1],
                views[2],
                value(args[0]),
                value(args[1]),
                value(args[2]),
                reduction_rows,
            )
            cost.sram += bank
            cost.arithmetic += arith
            cost.dependency += dep
            cost.accesses.update(counts)
        elif op in ("H_PREFETCH_V", "H_STORE_V", "H_PREFETCH_V.MV", "H_STORE_V.MV"):
            write = op.startswith("H_STORE")
            before_sram = cost.sram
            if op.endswith(".MV"):
                view = views[int(args[5])]
                length = view.values * 2
                cost.sram += bank_service(value(args[0]), view)[0]
            else:
                length = machine.vector_width * 2
                cost.sram += 1
            base = value(args[1])
            if base % 64 or length % 64:
                raise ValueError("unaligned/partial BF16 DMA needs an explicit adapter")
            if trace_memory:
                # Reads complete before SRAM placement; writes read SRAM first.
                memory_start = cost.total if write else cost.total - (cost.sram - before_sram)
                cost.memory_trace.append(("d", int(memory_start - traced_cycles), 0))
                cost.memory_trace.append(("w" if write else "r", base, length))
                traced_cycles = memory_start
            # R3 view writeback divides its packet into <=4096-byte rows.
            if write:
                for offset in range(0, length, machine.vector_width * 2):
                    n = min(machine.vector_width * 2, length - offset)
                    cost.transfers[("write", n)] += 1
            else:
                cost.transfers[("read", length)] += 1
        elif op in ("V_ADD_VV", "V_SUB_VV", "V_MUL_VV"):
            cost.sram += 3
            cost.arithmetic += 1
        elif op == "V_RECI_V":
            if int(args[2]) != 0:
                raise ValueError("masked reciprocal needs a separate service contract")
            cost.sram += 2
            cost.arithmetic += machine.vector_reciprocal_cycles
        else:
            raise ValueError(f"unpriced opcode {op} at instruction {pc}")
        registers[0] = 0
        pc += 1
    if loops:
        raise ValueError("unterminated loop")
    if trace_memory:
        cost.memory_trace.append(("d", int(cost.total - traced_cycles), 0))
    return cost


def dma_features(transfers, service):
    """Protocol-derived features; fitted coefficients are shared by all arms.

    read payload/gather startup, write payload, and finite write batches are
    separate. The review writer adds a serialized read for every write block.
    It must never be mislabeled as the optimized writer's window=1.
    """
    if service != "review" and int(service) < 1:
        raise ValueError("write window must be positive")
    f = Counter(read_calls=0.0, read_kib=0.0, write_kib=0.0, write_batches=0.0, rmw_blocks=0.0)
    for (direction, nbytes), count in transfers.items():
        blocks = nbytes // 64
        if direction == "read":
            f["read_calls"] += count
            f["read_kib"] += count * nbytes / 1024
        else:
            f["write_kib"] += count * nbytes / 1024
            if service == "review":
                f["rmw_blocks"] += count * blocks
            else:
                f["write_batches"] += count * math.ceil(blocks / int(service))
    return dict(f)


def apply_dma(cost, service, coefficients):
    features = dma_features(cost.transfers, service)
    cost.dma = sum(features[k] * coefficients[k] for k in features)
    if cost.dma < 0:
        raise ValueError("negative DMA cost")
    return cost
