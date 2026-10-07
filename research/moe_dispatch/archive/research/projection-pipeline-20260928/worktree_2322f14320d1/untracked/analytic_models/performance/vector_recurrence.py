"""Finite Vector-only experimental recurrence modes; no Matrix SRAM credit.

Expanded supply, compact selection, an explicit invariant latch and a bounded
walker have separate levels. The shared arithmetic variant reserves one
L-wide multiplier and one L-wide FP32 add/sub pipeline; provision of their
precision/interfaces is an unvalidated hardware extension, not free reuse.
"""
from dataclasses import dataclass
from collections import Counter

W = 2048


@dataclass(frozen=True)
class Config:
    rows: int
    width: int
    level: int
    origin: int

    @classmethod
    def decode(cls, shape, control):
        if shape >> 6 or control >> 10:
            raise ValueError("reserved Vector recurrence configuration bits")
        c = cls((shape & 31) + 1, 128 if shape & 32 else 64, control & 7, control >> 3)
        if not 1 <= c.level <= 4 or c.origin + c.rows > 128 or (c.level != 4 and c.rows != 1):
            raise ValueError("Vector recurrence exceeds bounded 32-row workspace")
        return c


@dataclass(frozen=True)
class Coefficient:
    base: int
    hs: int
    rs: int
    repeat: int
    heads: int
    extent: int

    @classmethod
    def decode(cls, word):
        if word >> 62:
            raise ValueError("reserved Vector coefficient bits")
        c = cls(word & 0x7ffff, word >> 19 & 8191, word >> 32 & 511,
                word >> 41 & 7, (word >> 44 & 63) + 1, (word >> 50 & 4095) + 1)
        if c.base + c.extent > 64 * W:
            raise ValueError("Vector coefficient exceeds 256-KiB capacity")
        return c

    def address(self, row, lane, cfg):
        if self.heads != W // cfg.width:
            raise ValueError("Vector coefficient head count differs")
        offset = lane if cfg.level == 1 else (lane // cfg.width >> self.repeat) * self.hs + row * self.rs
        if offset >= self.extent:
            raise ValueError("Vector coefficient extent")
        return self.base + offset


def resources(level, machine):
    return dict(
        storage="Vector SRAM only; private state streamed from HBM",
        vector_capacity_bytes=256*1024, state_chunk_rows=32 if level == 4 else 1,
        port="one shared read/write Vector port; each L-wide selected read charged",
        input_latches_bytes=4 * machine.lanes * 2,
        result_payload_bytes=4 * machine.lanes * 2,
        pipeline_partial_bytes=3 * machine.lanes * 4,
        invariant_latch_bytes=4096 if level >= 3 else 0,
        descriptors_bytes=32,
        arithmetic_mode=machine.vector_rec_alu,
        arithmetic=("one L-wide multiplier, one L-wide FP32 add/sub pipeline, II1 each"
                    if machine.vector_rec_alu == "shared" else "matched dedicated fused pipeline, II2"),
        arithmetic_latency=machine.vector_rec_alu_latency,
        unvalidated="raw BF16 product tap, mixed FP32-by-BF16 prediction multiply, FP32 add/sub, selected bank-write enables and wiring",
        area="not measured; register bytes and unit counts are not area",
    )


def primitive_cost(machine, op, cfg, coefficients, destination, source, invariant, tree_rows, held):
    from .ltile_cost import Schedule
    if cfg is None:
        raise ValueError("V_REC_EXEC requires CFG")
    m, c = machine, Schedule()
    if cfg.width > m.lanes or m.lanes % cfg.width:
        raise ValueError("Vector lanes must contain whole heads")
    issues = dict(mul=0, add=0)

    def vector(ready, write=False):
        return c.reserve("vector", ready, m.sram_cycles, "vector_writes" if write else "vector_reads")

    def alu(ready, multiply):
        name = "mul" if multiply else "add"
        t = ready
        while issues[name] >> t & 1:
            t += 1
        issues[name] |= 1 << t
        end = t + m.vector_rec_alu_latency
        c.arithmetic |= ((1 << m.vector_rec_alu_latency) - 1) << t
        c.finish = max(c.finish, end)
        c.counts[name + "_operations"] += 1
        c.counts["launches"] += 1
        return end

    def arithmetic(ready):
        if m.vector_rec_alu == "dedicated":
            return c.compute(ready, {0: 6, 1: 9, 2: 3, 3: 4, 8: 2, 9: 6}[op], 2)
        if op == 0:
            p, q = alu(ready, True), alu(ready, True)
            u = alu(p, False)
            return alu(max(u, q), False)
        if op == 1:
            return alu(alu(alu(ready, True), False), True)
        if op in (2, 8):
            return alu(ready, True)
        if op == 3:
            return alu(alu(ready, False), True)
        if op == 9:
            return alu(alu(ready, True), False)
        raise ValueError("unsupported shared Vector operation")

    def address(a):
        if a % W or a < 0 or a + W > 64*W:
            raise ValueError("Vector row exceeds SRAM capacity/alignment")

    if op == 4:
        if cfg.level < 3:
            raise ValueError("invariant latch exists only at levels3/4")
        address(source); vector(0)
        return (*c.totals(), tree_rows, True)
    if op == 5:
        return (*c.totals(), 0, held)
    if op == 7:
        if tree_rows != 128:
            raise ValueError("tree write requires 128 leaves")
        address(destination)
        if destination != 4*W:
            vector(vector(0), True)
        return (*c.totals(), tree_rows, held)

    def fold(count, ready):
        merges = 0
        while count >> merges & 1:
            merges += 1
        for _ in range(merges):
            ready = vector(vector(ready))
            ready = c.compute(ready, 1, 1, existing=True)
            ready = vector(ready, True)
        return ready

    if op == 6:
        if tree_rows >= 128:
            raise ValueError("too many tree leaves")
        fold(tree_rows, 0)
        return (*c.totals(), tree_rows+1, held)
    if op not in (0, 1, 2, 3, 8, 9):
        raise ValueError("reserved Vector recurrence primitive")
    rows = cfg.rows if cfg.level == 4 and op in (0, 1, 2) else 1
    if cfg.level == 4 and op in (1, 2) and tree_rows + rows > 128:
        raise ValueError("Vector FSM exceeds 128 tree leaves")
    ready, credits = 0, [0] * 4
    for row in range(rows):
        address(source + row*W)
        if not (cfg.level == 4 and op in (1, 2)):
            address(destination + row*W)
        for chunk, first in enumerate(range(0, W, m.lanes)):
            ready = vector(ready)
            fields = (0, 1) if op in (0, 1) else (2,) if op == 2 else (0,)
            blocks = set()
            for field in fields:
                cv = coefficients[field]
                if cv is None:
                    raise ValueError("Vector EXEC without CCFG")
                blocks.update(cv.address(cfg.origin+row, lane, cfg)//W
                              for lane in range(first, min(W, first+m.lanes)))
            for _ in sorted(blocks):
                ready = vector(ready)
            if cfg.level >= 2:
                ready += 1
                c.finish = max(c.finish, ready)
            if op == 0:
                if cfg.level >= 3:
                    if not held:
                        raise ValueError("Vector invariant requires HOLD")
                else:
                    address(invariant); ready = vector(ready)
            elif op in (3, 9):
                address(destination if op == 9 else invariant); ready = vector(ready)
            launch = max(ready, credits[chunk % 4])
            done = arithmetic(launch)
            credits[chunk % 4] = vector(done, True)
            # Retain the sole input bundle through its last consumer issue.
            # Dedicated fusion captures all operands at the first launch.
            ready = done-m.vector_rec_alu_latency+1 if m.vector_rec_alu == "shared" else launch+1
        if cfg.level == 4 and op in (1, 2):
            if tree_rows >= 128:
                raise ValueError("too many tree leaves")
            ready = fold(tree_rows, c.finish)
            tree_rows += 1
        else:
            ready = c.finish
    return (*c.totals(), tree_rows, held)
