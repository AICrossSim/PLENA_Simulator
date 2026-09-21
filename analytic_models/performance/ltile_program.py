"""Shape-only recurrent compilation: no checkpoint, tensor or result lookup.

Address allocation follows the accepted E ABI, including 64-byte guards. The
compiler supplies the optimized old ISA and the row/FSM programs. All batch
members have private HBM allocations; no timing is extrapolated from B1.
"""

from dataclasses import replace
from pathlib import Path
import sys


class ShapeArena:
    def __init__(self):
        self.size = 0

    def add(self, elements):
        self.size += (-self.size) % 64 + 64
        base = self.size
        self.size += elements * 2
        self.size += (-self.size) % 64
        return base


def build_program(
    kind,
    batch=1,
    tokens=4,
    control="fsm",
    *,
    compiler_root,
    phased=True,
    broadcast=True,
    resident=True,
    diagnostic=False,
    decay_is_delta=False,
):
    if kind not in ("mamba", "kda") or batch < 1 or tokens < 1:
        raise ValueError("invalid recurrent workload")
    root = str(Path(compiler_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from compiler.aten.plena.ltile_v2 import Options, Emitter, lower_group, views
    from compiler.aten.plena.matrix_recurrence_lowering import (
        NEMOTRON_MAMBA,
        KIMI_KDA,
    )
    from compiler.aten.plena.prepared_vector_recurrence import (
        PreparedVectorGroup,
        lower_prepared_vector_recurrence,
    )

    arena = ShapeArena()
    heads, width = (64, 64) if kind == "mamba" else (96, 128)
    groups = batch * heads * width // 2048
    if control == "old_isa":
        spec = NEMOTRON_MAMBA if kind == "mamba" else KIMI_KDA
        spec = replace(spec, heads=spec.heads * batch)
        state_base = arena.add(groups * 128 * 2048)
        names = ("a", "b", "c", "x", "dt", "d") if kind == "mamba" else ("decay", "key", "query", "value", "beta")
        programs = []
        for _ in range(tokens):
            packed = []
            for g in range(groups):
                fields = {name: arena.add(128 * 2048 if name in names[:3] else 2048) for name in names}
                fields["zero"] = arena.add(2048)
                fields["output"] = arena.add(2048)
                packed.append(PreparedVectorGroup(state_base + g * 128 * 4096, fields))
            programs.append(
                lower_prepared_vector_recurrence(
                    spec,
                    tuple(packed),
                    static_address_reuse=True,
                    pairwise_bf16_dot=True,
                    mamba_decay_row_invariant=kind == "mamba",
                    decay_is_delta=decay_is_delta,
                )
            )
        return "\n".join(programs), arena.size
    o = Options(kind, control, phased, broadcast, resident)
    chunk = o.chunk_rows
    coefficients = views(o)["coeff"].descriptor.shape.cols
    scalars = ((2 * o.heads + 31) // 32) * 32
    states = [[arena.add(chunk * 2048) for _ in range(128 // chunk)] for _ in range(groups)]
    emitter = Emitter()
    for _ in range(tokens):
        for g in range(groups):
            memory = dict(states=states[g], update=[], dot=[])
            for _ in range(128 // chunk):
                memory["update"].append(arena.add(chunk * coefficients))
                memory["dot"].append(arena.add(chunk * coefficients))
            for name, elements in [("input", 2048), ("scalar", scalars), ("skip", scalars), ("output", 2048)]:
                memory[name] = arena.add(elements)
            if diagnostic:
                memory["snapshots"] = [arena.add(chunk * 2048) for _ in range(128 // chunk)]
            lower_group(o, memory, emitter)
    return "\n".join(emitter.lines) + "\n", arena.size
