"""Executable recurrent sublayer plans, generated from dimensions only.

No checkpoints or measured cycle tables enter this module. The reference B1
plans are instruction-identical to the connected numerical tests. Stages own
their padded allocations; weights may be shared, state is request private.
"""

from dataclasses import dataclass
from pathlib import Path
import sys

from .ltile_program import ShapeArena


def compiler_api(root):
    root = Path(root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from compiler.aten.plena import recurrent_coefficients as c
    from compiler.aten.plena.isa_matrix_projection import Projection, lower_b1_projection
    from compiler.aten.plena.ltile_v2 import Options, lower_group

    # Refuse the old frozen E compiler accidentally imported by another runner.
    if root not in Path(c.__file__).resolve().parents:
        raise ValueError(f"Compiler import belongs to another checkout: {c.__file__}")
    return c, Projection, lower_b1_projection, Options, lower_group


def lane(base, index):
    return base + index // 2048 * 4096, index % 2048


@dataclass(frozen=True)
class Stage:
    name: str
    assembly: str
    category: str
    matrix_shape: tuple = ()
    weight_base: int = 0
    weight_bytes: int = 0
    input_base: int = 0
    output_base: int = 0
    zero_base: int = 0


class LayerPlan:
    def __init__(
        self, compiler_root, *, gather="reference", projection_schedule="stream", arena=None, shared_weights=None,
        vector_rows=58, gather_vector_rows=64, projection_n_panel_tile=1,
    ):
        self.c, self.Projection, self.lower_projection, self.Options, self.lower_group = compiler_api(compiler_root)
        if projection_schedule not in ("stream", "resident", "compact", "batch"):
            raise ValueError("unknown projection schedule")
        self.projection_schedule = projection_schedule
        self.vector_rows = vector_rows
        self.gather_vector_rows = gather_vector_rows
        if projection_schedule == "resident":
            from functools import partial
            from compiler.aten.plena.isa_matrix_projection import lower_resident_projection

            self.lower_projection = partial(lower_resident_projection, vector_rows=vector_rows)
        elif projection_schedule in ("compact", "batch"):
            from functools import partial
            from compiler.aten.plena.isa_matrix_projection import lower_compact_projection

            self.lower_projection = partial(
                lower_compact_projection, batch_tile=1 if projection_schedule == "compact" else 4,
                vector_rows=vector_rows, n_panel_tile=projection_n_panel_tile,
            )
        self.arena = arena if arena is not None else ShapeArena()
        self.shared_weights = shared_weights if shared_weights is not None else {}
        self.stages = []
        self.gather_mode = gather
        self.zero = self.arena.add(2048)
        self.onehot = self.arena.add(2048)
        self.constants = self.arena.add(len(self.c.GATE_CONSTANTS) * 2048)
        self.masks = self.arena.add(16 * 2048)

    def output(self, count, extra=0):
        return self.arena.add((count + extra + 2047) // 2048 * 2048)

    def emit(self, name, text, category=None, **metadata):
        if category is None:
            category = (
                "recurrence"
                if name.startswith("recurrence")
                else "coefficient_layout"
                if name.startswith("pack_")
                else "conv"
                if "conv" in name
                else "norm"
                if "norm" in name or "rms" in name
                else "gate_prepare"
            )
        self.stages.append(Stage(name, text, category, **metadata))

    def projection(self, name, source, k, n):
        key = (name, k, n)
        if key not in self.shared_weights:
            self.shared_weights[key] = self.arena.add(((k + 31) // 32 * 32) * ((n + 31) // 32 * 32))
        weights = self.shared_weights[key]
        out = self.output(n, 2048)
        spec = self.Projection(source, weights, out, self.zero, k, n, 256)
        self.emit(
            name,
            self.lower_projection(spec),
            "projection",
            matrix_shape=(1, n, k),
            weight_base=weights,
            weight_bytes=spec.weight_bytes,
            input_base=source,
            output_base=out,
            zero_base=self.zero,
        )
        return out

    def gather(self, name, mapping):
        out = self.output(len(mapping))
        kwargs = {} if self.gather_mode == "reference" else {"strategy": self.gather_mode}
        if self.gather_vector_rows != 64:
            kwargs["vector_rows"] = self.gather_vector_rows
        self.emit(name, self.c.lower_bf16_gather(mapping, out, self.zero, self.onehot, **kwargs))
        return out

    def assembly(self, *, marked=False):
        return "".join((f"; @operator={s.name}\n" if marked else "") + s.assembly for s in self.stages)

    def old_group(self, kind, group, fields, mappings):
        from dataclasses import replace
        from compiler.aten.plena.prepared_vector_recurrence import PreparedVectorGroup, lower_prepared_vector_recurrence
        from compiler.aten.plena.matrix_recurrence_lowering import NEMOTRON_MAMBA, KIMI_KDA

        intervals = sorted({(start, count) for entries in mappings.values() for _, _, start, count in entries})
        masks = {key: self.arena.add(2048) for key in intervals}
        loader = self.c.CompactCoefficientLoader(mappings, masks, self.onehot)
        state = self.arena.add(128 * 2048)
        spec = replace(NEMOTRON_MAMBA if kind == "mamba" else KIMI_KDA, heads=32 if kind == "mamba" else 16)
        self.emit(
            f"recurrence_{group}",
            lower_prepared_vector_recurrence(
                spec,
                (PreparedVectorGroup(state, dict(fields, zero=self.zero)),),
                static_address_reuse=True,
                pairwise_bf16_dot=True,
                mamba_decay_row_invariant=kind == "mamba",
                decay_is_delta=True,
                coefficient_loaders=(loader,),
            ),
        )


def build_layer(
    kind,
    compiler_root,
    *,
    control="fsm",
    gather="reference",
    projection_schedule="stream",
    arena=None,
    shared_weights=None,
    native_coefficients=False,
    vector_rows=58,
    gather_vector_rows=64,
    projection_n_panel_tile=1,
):
    if native_coefficients and control != "fsm":
        raise ValueError("native prototype requires FSM")
    if kind not in ("mamba", "kda") or control not in ("old_isa", "row", "fsm"):
        raise ValueError("expected Mamba/KDA and matched row/FSM control")
    p = LayerPlan(
        compiler_root,
        gather=gather,
        projection_schedule=projection_schedule,
        arena=arena,
        shared_weights=shared_weights,
        vector_rows=vector_rows,
        gather_vector_rows=gather_vector_rows,
        projection_n_panel_tile=projection_n_panel_tile,
    )
    c, a = p.c, p.arena
    if kind == "mamba":
        source, nw = a.add(4096), a.add(4096)
        norm = p.output(2688, 2048)
        p.emit("input_norm", c.lower_global_rms(source, nw, norm, 2688))
        projected = p.projection("input_projection", norm, 2688, 10304)
        wb, bias, history, conv = a.add(4 * 6144), a.add(6144), a.add(4 * 6144), p.output(6144)
        p.emit(
            "convolution",
            c.lower_conv_steps([c.ConvStep(projected + 8192, history, wb, conv, 6144, bias)], p.constants),
        )
        db, ab, dt, delta = a.add(2048), a.add(2048), p.output(2048), p.output(2048)
        p.emit(
            "gate_production",
            c.lower_mamba_gate_rows([c.MambaGateRow(projected + 20480, db, ab, dt, delta)], p.constants),
        )
        skipbase, onebase, raw = a.add(2048), a.add(2048), p.output(4096)
        for group in range(2):
            heads = range(group * 32, (group + 1) * 32)
            if control == "old_isa":
                mapping = {}
                for name, base in (("a", delta), ("dt", dt), ("d", skipbase)):
                    mapping[name, 0] = [(*lane(base, h), i * 64, 64) for i, h in enumerate(heads)]
                for row in range(128):
                    for name, offset in (("b", 4096), ("c", 5120)):
                        mapping[name, row] = [
                            (*lane(conv, offset + h // 8 * 128 + row), i * 64, 512)
                            for i, h in enumerate(heads)
                            if i % 8 == 0
                        ]
                p.old_group(kind, group, dict(x=conv + group * 4096, output=raw + group * 4096), mapping)
                continue
            update, dot = [], []
            for row in range(128):
                for h in heads:
                    update.extend([lane(delta, h), lane(conv, 4096 + h // 8 * 128 + row)])
                    dot.extend([lane(conv, 5120 + h // 8 * 128 + row), None])
            # Keep historical fixture addresses stable. Native execution never
            # reads/writes these reserved HBM holes; no packing cost is hidden.
            u = p.output(len(update)) if native_coefficients else p.gather(f"pack_update_{group}", update)
            d = p.output(len(dot)) if native_coefficients else p.gather(f"pack_dot_{group}", dot)
            scalar = p.gather(f"pack_dt_{group}", [s for h in heads for s in (None, lane(dt, h))])
            skip = p.gather(f"pack_skip_{group}", [s for h in heads for s in (lane(onebase, 0), lane(skipbase, h))])
            state = a.add(32 * 128 * 64)
            memory = dict(
                states=[state],
                update=[u],
                dot=[d],
                input=conv + group * 4096,
                scalar=scalar,
                skip=skip,
                output=raw + group * 4096,
            )
            if native_coefficients:
                memory["native"] = [
                    (delta, group*32, 1, 0, 0),
                    (conv + 8192, group*512, 128, 1, 3),
                    (conv + 8192, 1024 + group*512, 128, 1, 3),
                ]
            p.emit(f"recurrence_{group}", "\n".join(p.lower_group(p.Options(kind, control), memory).lines) + "\n")
        silu, gated = p.output(4096), p.output(4096)
        p.emit(
            "silu_gate",
            c.lower_pointwise_rows(projected, projected, silu, 2, sigmoid_input=True, constants_base=p.constants),
        )
        p.emit("gate_product", c.lower_pointwise_rows(raw, silu, gated, 2))
        masks, rms = a.add(4 * 2048), p.output(4096)
        p.emit("gated_rms", c.lower_l2norm_rows(c.L2NormRows(gated, rms, 4096, masks, p.zero, 2, 3, 512)))
        mn, weighted = a.add(4096), p.output(4096, 2048)
        p.emit("norm_weight", c.lower_pointwise_rows(rms, mn, weighted, 2))
        p.projection("output_projection", weighted, 4096, 2688)
    else:
        source, nw, norm = a.add(8192), a.add(8192), p.output(7168, 2048)
        p.emit("input_norm", c.lower_global_rms(source, nw, norm, 7168))
        projected = {
            name: p.projection(name + "_projection", norm, 7168, n)
            for name, n in (("q", 12288), ("k", 12288), ("v", 12288), ("f_a", 128), ("b", 96), ("g", 12288))
        }
        gate = p.projection("f_b_projection", projected["f_a"], 128, 12288)
        conv = {}
        for name in ("q", "k", "v"):
            wb, history, out = a.add(4 * 12288), a.add(4 * 12288), p.output(12288)
            p.emit(
                name + "_conv", c.lower_conv_steps([c.ConvStep(projected[name], history, wb, out, 12288)], p.constants)
            )
            conv[name] = out
        vectors = {}
        for name, slot in (("q", 3), ("k", 4)):
            out = p.output(12288)
            p.emit(name + "_norm", c.lower_l2norm_rows(c.L2NormRows(conv[name], out, 12288, p.masks, p.zero, 2, slot)))
            vectors[name] = out
        bias, alog, delta = a.add(12288), a.add(12288), p.output(12288)
        beta = [p.output(2048) for _ in range(6)]
        p.emit(
            "decay_beta",
            c.lower_kda_gate_rows(
                [
                    c.KdaGateRow(
                        gate + i * 4096, bias + i * 4096, alog + i * 4096, projected["b"], delta + i * 4096, beta[i]
                    )
                    for i in range(6)
                ],
                p.constants,
            ),
        )
        raw = p.output(12288, 2048)
        for group in range(6):
            heads = range(group * 16, (group + 1) * 16)
            if control == "old_isa":
                mapping = {("beta", 0): [(*lane(beta[0], h), i * 128, 128) for i, h in enumerate(heads)]}
                for row in range(128):
                    for name, base in (("decay", delta), ("key", vectors["k"]), ("query", vectors["q"])):
                        mapping[name, row] = [(*lane(base, h * 128 + row), i * 128, 128) for i, h in enumerate(heads)]
                p.old_group(kind, group, dict(value=conv["v"] + group * 4096, output=raw + group * 4096), mapping)
                continue
            indices = [h * 128 + r for r in range(128) for h in heads]
            umap = [s for i in indices for s in (lane(delta, i), lane(vectors["k"], i))]
            dmap = [s for i in indices for s in (lane(vectors["q"], i), None)]
            update = p.output(len(umap)) if native_coefficients else p.gather(f"pack_update_{group}", umap)
            dot = p.output(len(dmap)) if native_coefficients else p.gather(f"pack_dot_{group}", dmap)
            scalar = p.gather(f"pack_beta_{group}", [s for h in heads for s in (lane(beta[0], h), None)])
            state = a.add(16 * 128 * 128)
            memory = dict(
                states=[state],
                update=[update],
                dot=[dot],
                input=conv["v"] + group * 4096,
                scalar=scalar,
                output=raw + group * 4096,
            )
            if native_coefficients:
                memory["native"] = [(base + group*4096, 0, 128, 1, 0)
                    for base in (delta, vectors["k"], vectors["q"])]
            p.emit(f"recurrence_{group}", "\n".join(p.lower_group(p.Options(kind, control), memory).lines) + "\n")
        rms = p.output(12288)
        p.emit("output_rms", c.lower_l2norm_rows(c.L2NormRows(raw, rms, 12288, p.masks, p.zero, 5, 6)))
        wb, weighted = a.add(12288), p.output(12288)
        p.emit("output_norm_weight", c.lower_pointwise_rows(rms, wb, weighted, 6))
        gated = p.output(12288, 2048)
        p.emit(
            "output_gate",
            c.lower_pointwise_rows(projected["g"], weighted, gated, 6, sigmoid_input=True, constants_base=p.constants),
        )
        p.projection("output_projection", gated, 12288, 7168)
    return p


def build_batch(kind, batch, compiler_root, *, control="fsm", gather="grouped", projection_schedule="stream", native_coefficients=False,
                vector_rows=58, gather_vector_rows=64, projection_n_panel_tile=1, projection_panel_overrides=(),
                projection_request_tile=16, projection_k_tile=256):
    """Compile batch stages with shared weight panels and private state.

    No B1 cycle multiplication. Matrix SRAM weights stay resident across
    requests within a panel; recurrent tiles are processed one request at a
    time. This legal sequential schedule has no implicit concurrent DMA.
    """
    from dataclasses import replace

    if type(batch) is not int or not 1 <= batch <= 16:
        raise ValueError("batch must be 1..16")
    if type(projection_request_tile) is not int or projection_request_tile not in (1, 2, 4, 8, 16):
        raise ValueError("projection request tile must be 1/2/4/8/16")
    if projection_request_tile != 16 and projection_schedule != "resident":
        raise ValueError("software request grouping requires resident projection")
    if type(projection_k_tile) is not int or projection_k_tile not in (256, 512, 1024) or (
        projection_schedule != "transposed" and projection_k_tile != 256
    ):
        raise ValueError("larger K grouping requires the transposed software path")
    overrides = dict(projection_panel_overrides)
    if len(overrides) != len(projection_panel_overrides):
        raise ValueError("duplicate projection stage override")
    if overrides and projection_schedule not in ("compact", "batch"):
        raise ValueError("projection overrides require compact/batch lowering")
    arena, weights = ShapeArena(), {}
    plans = [
        build_layer(
            kind,
            compiler_root,
            control=control,
            gather=gather,
            projection_schedule="resident" if projection_schedule == "transposed" else projection_schedule,
            arena=arena,
            shared_weights=weights,
            native_coefficients=native_coefficients,
            vector_rows=vector_rows,
            gather_vector_rows=gather_vector_rows,
            projection_n_panel_tile=projection_n_panel_tile,
        )
        for _ in range(batch)
    ]
    names = {s.name for s in plans[0].stages if s.matrix_shape}
    if overrides.keys() - names:
        raise ValueError(f"unknown projection stage override: {sorted(overrides.keys() - names)}")
    if batch == 1 and projection_schedule != "transposed":
        result = plans[0]
        for index, s in enumerate(result.stages):
            if s.name in overrides:
                _, n, k = s.matrix_shape
                spec = result.Projection(s.input_base, s.weight_base, s.output_base, s.zero_base, k, n, 256)
                result.stages[index] = replace(s, assembly=result.lower_projection(spec, n_panel_tile=overrides[s.name]))
        return result
    from compiler.aten.plena.isa_matrix_projection import lower_batch_projection, lower_resident_projection

    result = plans[0]
    all_stages = [p.stages for p in plans]
    result.stages = []
    for columns in zip(*all_stages):
        s = columns[0]
        if any(x.name != s.name for x in columns):
            raise AssertionError("batch stage order differs")
        if s.matrix_shape:
            _, n, k = s.matrix_shape
            spec = result.Projection(s.input_base, s.weight_base, s.output_base, s.zero_base, k, n, 256)
            if projection_schedule == "transposed":
                from compiler.aten.plena.isa_projection_software import lower_transposed_projection

                text = lower_transposed_projection(
                    replace(spec, k_tile=projection_k_tile),
                    [x.input_base for x in columns], [x.output_base for x in columns], vector_rows=vector_rows,
                )
            elif projection_schedule in ("compact", "batch"):
                from compiler.aten.plena.isa_matrix_projection import lower_compact_projection

                text = lower_compact_projection(
                    spec, [x.input_base for x in columns], [x.output_base for x in columns],
                    batch_tile=1 if projection_schedule == "compact" else 4,
                    vector_rows=vector_rows,
                    n_panel_tile=overrides.get(s.name, projection_n_panel_tile),
                )
            else:
                lower = lower_resident_projection if projection_schedule == "resident" else lower_batch_projection
                kwargs = {"vector_rows": vector_rows} if projection_schedule == "resident" else {}
                if projection_schedule == "resident" and projection_request_tile < batch:
                    from compiler.aten.plena.isa_projection_software import lower_software_projection

                    lower = lower_software_projection
                    kwargs["request_tile"] = projection_request_tile
                text = lower(spec, [x.input_base for x in columns], [x.output_base for x in columns], **kwargs)
            result.stages.append(replace(s, assembly=text, matrix_shape=(batch, n, k)))
        else:
            result.stages.extend(replace(x, name=f"r{i}/{x.name}") for i, x in enumerate(columns))
    return result
