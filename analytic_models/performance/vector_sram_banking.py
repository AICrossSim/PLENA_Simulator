"""Would a banked Vector SRAM stall on the static Mamba/KDA sweeps?

The emulator does not model banking: `VectorSram` is a flat `Vec<Mutex<Cell>>`,
one entry per row, so every access costs the same whatever its address. The
question is therefore analytic, and it is worth answering before anyone builds
banking into the RTL: **do the sweeps this lowering emits have strides that would
conflict?**

The model
---------
A vector row is `mlen` words. Under row-interleaved banking with `B` banks, row
`r` lives in bank `r % B`, and a sweep that advances by `stride` rows visits
banks `r, r+stride, r+2*stride, ...` (mod B). It touches `B / gcd(stride, B)`
distinct banks before repeating, so the fraction of the array it can keep busy
is `gcd(stride, B) / B` -- and the worst case, `stride % B == 0`, pins every
access to one bank.

Word-interleaved banking is the other plausible mapping, and under it a whole
`mlen`-word row spans `min(mlen, B)` banks, so a row-sequential sweep never
conflicts regardless of stride. That mapping makes this whole question moot;
the study is about the row-interleaved one.

Where the strides come from
---------------------------
Not from a model of the algorithm -- from the emitted assembly. Every hardware
loop this lowering produces advances its pointers with
`S_ADDI_INT gpN, gpN, step`, and `step` is in words. :func:`sweep_strides` reads
them out of a compiled program, so the study tracks the code rather than a
description of it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

#: `S_ADDI_INT gpN, gpN, step` -- a pointer advancing by itself is a sweep step.
#: The initialising form is `S_ADDI_INT gpN, gp0, addr`, which this does not match.
_STEP = re.compile(r"S_ADDI_INT\s+gp(\d+)\s*,\s*gp(\d+)\s*,\s*(-?\d+)")

#: `C_LOOP_START gpN, count`
_LOOP = re.compile(r"C_LOOP_START\s+gp(\d+)\s*,\s*(\d+)")


@dataclass(frozen=True)
class Sweep:
    """One hardware loop's pointer advance, in rows."""

    stride_rows: int
    trips: int

    def distinct_banks(self, banks: int) -> int:
        """Banks this sweep visits before it repeats."""
        if self.stride_rows == 0:
            return 1  # a pinned pointer sits on one bank
        return banks // math.gcd(abs(self.stride_rows), banks)

    def conflict_fraction(self, banks: int) -> float:
        """Share of accesses that land on an already-busy bank.

        ``0.0`` means the sweep walks every bank in turn; ``1 - 1/banks`` is the
        worst case, every access on one bank.
        """
        distinct = self.distinct_banks(banks)
        return 1.0 - distinct / banks


def sweep_strides(asm: str, mlen: int) -> list[Sweep]:
    """Extract each hardware loop's pointer stride, in rows, from ``asm``.

    A loop body may advance several pointers; the one that matters for banking
    is the largest stride, because that is what determines how far apart
    consecutive accesses land. A pinned pointer (no advance) is reported as
    stride 0, which is the worst case under row interleaving.
    """
    sweeps: list[Sweep] = []
    lines = [ln.strip() for ln in asm.splitlines() if ln.strip()]
    i = 0
    while i < len(lines):
        loop = _LOOP.search(lines[i])
        if not loop:
            i += 1
            continue
        loop_reg, trips = int(loop.group(1)), int(loop.group(2))
        steps: list[int] = []
        j = i + 1
        depth = 1
        while j < len(lines) and depth:
            if _LOOP.search(lines[j]):
                depth += 1
            elif f"C_LOOP_END gp{loop_reg}" in lines[j]:
                depth -= 1
                if not depth:
                    break
            m = _STEP.search(lines[j])
            if m and m.group(1) == m.group(2):
                steps.append(int(m.group(3)))
            j += 1
        # Pointers that advance by whole rows are the vector-SRAM walkers;
        # anything else is an FPRAM pointer and does not touch this array.
        row_steps = [s // mlen for s in steps if s and s % mlen == 0]
        sweeps.append(Sweep(max(row_steps, default=0), trips))
        i = j + 1
    return sweeps


def report(named: dict[str, str], mlen: int, bank_counts=(2, 4, 8, 16, 32)) -> str:
    """A table of stalled-access fractions per kernel per bank count."""
    out = [
        f"Vector SRAM banking, row-interleaved, mlen={mlen}",
        "",
        f"{'kernel':<28} {'loops':>6} {'worst stride':>13} " + " ".join(f"{'B=' + str(b):>7}" for b in bank_counts),
        "-" * (28 + 7 + 14 + 8 * len(bank_counts)),
    ]
    for name, asm in named.items():
        sweeps = sweep_strides(asm, mlen)
        if not sweeps:
            out.append(f"{name:<28} {'0':>6}  (no hardware loops)")
            continue
        # Weight by trip count: a 128-trip conflicted loop matters more than a
        # 2-trip one.
        total = sum(s.trips for s in sweeps) or 1
        cells = []
        for b in bank_counts:
            stalled = sum(s.trips * s.conflict_fraction(b) for s in sweeps)
            cells.append(f"{stalled / total:>7.1%}")
        worst = max(sweeps, key=lambda s: s.conflict_fraction(max(bank_counts)))
        out.append(f"{name:<28} {len(sweeps):>6} {worst.stride_rows:>13} " + " ".join(cells))
    return "\n".join(out)


def _lowered_kernels(mlen: int = 64) -> dict[str, str]:
    """Compile the kernels this study is about."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for candidate in (root / "PLENA_Compiler", root.parent / "PLENA_Compiler"):
        if (candidate / "aten" / "plena" / "compiler.py").exists():
            sys.path.insert(0, str(candidate))
            break

    from compiler.aten.models.kda.shape import KdaShape
    from compiler.aten.plena import PlenaCompiler
    from compiler.aten.plena.program_kda_common import kda_state_rows, kda_vector_rows
    from compiler.aten.plena.program_kda_gates import kda_head_blocks, kda_key_blocks
    from compiler.aten.plena.program_kda_mixer import KdaMixerBuffers
    from compiler.aten.plena.program_mamba_common import Mamba2Shape

    out: dict[str, str] = {}
    up = lambda n: ((n + mlen - 1) // mlen) * mlen  # noqa: E731

    # KDA: the recurrence and the whole mixer, at Kimi K3's shape.
    shape = KdaShape.kimi_k3()
    kb = kda_key_blocks(shape, mlen)
    p = PlenaCompiler(mlen=mlen, blen=4)
    a = lambda n, r: p.alloc(n, up(r), mlen, strict=False)  # noqa: E731
    decay = p.fp_var("decay", size=shape.key_dim)
    buffers = KdaMixerBuffers(
        q=a("q", shape.num_heads * kb),
        k=a("k", shape.num_heads * kb),
        v=a("v", kda_vector_rows(shape, mlen)),
        gate=a("g", shape.num_heads * kb),
        dt_bias=a("dtb", shape.num_heads * kb),
        beta_logit=a("bl", kda_head_blocks(shape, mlen)),
        state=a("st", kda_state_rows(shape, mlen)),
        out=a("o", kda_vector_rows(shape, mlen)),
        pred=a("pr", kda_vector_rows(shape, mlen)),
        err=a("er", kda_vector_rows(shape, mlen)),
        sq_scratch=a("sq", shape.num_heads * kb),
        decay_fp=decay,
        q_hat_fp=decay,
        k_hat_fp=p.fp_var("kh", size=shape.key_dim),
        beta_fp=p.fp_var("b", size=kda_head_blocks(shape, mlen) * mlen),
        part_fp=p.fp_var("pt", size=kb),
        acc_fp=p.fp_var("ac", size=1),
        output_scale_fp=p.fp_var("os", size=1),
        rate_fp=p.fp_var("rt", size=shape.num_heads),
        lower_bound_fp=p.fp_var("lb", size=1),
        consts=p.kda_fp_constants(),
    )
    mark = len(p.get_code())
    p.kda_decode_step_v0(
        state=buffers.state,
        q_fp=buffers.q_hat_fp,
        k_fp=buffers.k_hat_fp,
        decay_fp=buffers.decay_fp,
        beta_fp=buffers.beta_fp,
        v=buffers.v,
        o=buffers.out,
        pred=buffers.pred,
        err=buffers.err,
        shape=shape,
        output_scale_fp=buffers.output_scale_fp,
        head_rows=[0],
        fp_head_stride=0,
    )
    out["KDA recurrence (Kimi K3)"] = p.get_code()[mark:]

    mark = len(p.get_code())
    p.kda_mixer_step_v0(buffers=buffers, shape=shape, head_rows=[0])
    out["KDA mixer, one head"] = p.get_code()[mark:]

    # Mamba: the decode step.
    m_shape = Mamba2Shape(
        hidden_size=mlen,
        num_heads=1,
        head_dim=mlen,
        state_size=mlen,
        n_groups=1,
        conv_kernel=4,
        chunk_size=16,
        seq_len=1,
    )
    q = PlenaCompiler(mlen=mlen, blen=4)
    b = lambda n, r: q.alloc(n, up(r), mlen, strict=False)  # noqa: E731
    consts = q.mamba_fp_constants()
    mark = len(q.get_code())
    q.ssm_decode_step_v0(
        state=b("st", m_shape.state_size),
        x=b("x", 1),
        y=b("y", 1),
        b_fp=q.fp_var("b", size=m_shape.state_size),
        c_fp=q.fp_var("c", size=m_shape.state_size),
        da_fp=q.fp_var("da", size=1),
        dt_fp=q.fp_var("dt", size=1),
        d_fp=q.fp_var("d", size=1),
        scratch=b("sc", 1),
        shape=m_shape,
        consts=consts,
    )
    out["Mamba-2 decode step"] = q.get_code()[mark:]
    return out


if __name__ == "__main__":  # pragma: no cover - a report, not a test
    MLEN = 64
    print(report(_lowered_kernels(MLEN), MLEN))
