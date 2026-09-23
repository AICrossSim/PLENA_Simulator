"""Reusable compiled operator services and version-bound analytical cache."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import uuid

from .ltile_layers import build_batch, compiler_api
from .ltile_program import ShapeArena
from .ltile_execution import price_program
from .ltile_dma import sha
from .ltile_cost import ProgramCost


class Services:
    def __init__(self, compiler, profile, backend, cache):
        self.compiler = Path(compiler).resolve()
        self.profile = profile
        if profile.codec.input_bytes + profile.codec.output_bytes > 6 * 4096:
            raise ValueError("weight decoder exceeds the six reserved Vector SRAM rows")
        self.backend = backend
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        here = Path(__file__).parent
        paths = [
            here / name
            for name in (
                "ltile_cost.py",
                "ltile_dma.py",
                "ltile_program.py",
                "ltile_execution.py",
                "ltile_layers.py",
                "ltile_services.py",
                "ltile_platform.py",
                "weight_codec.py",
                "matrix_service.py",
            )
        ]
        paths += [
            self.compiler / "aten/plena" / name
            for name in (
                "ltile_v2.py",
                "isa_matrix_projection.py",
                "recurrent_coefficients.py",
                "prepared_vector_recurrence.py",
                "mview.py",
            )
        ]
        self.sources = {str(p): sha(p) for p in paths}
        self.identity = dict(profile=profile.identity, memory=backend.identity, sources=self.sources)

    def cached(self, spec, generate):
        identity = dict(self.identity, operator=spec)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.cache / (key + ".json")
        if path.exists():
            return json.loads(path.read_text())
        text, regions, metadata = generate()
        cost, result = price_program(text, self.profile, self.backend, nvfp4_regions=regions)
        result.update(
            identity=identity,
            sections=cost.sections,
            metadata=metadata,
            sram_accesses=dict(cost.accesses),
            cache_key=key,
        )
        temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(path)
        print(json.dumps(dict(operator=spec, cycles=cost.total)), flush=True)
        return result

    def layer(self, kind, batch, control, weight="BF16"):
        if weight not in ("BF16", "NVFP4"):
            raise ValueError("unsupported weight contract")

        def generate():
            plan = build_batch(
                kind, batch, self.compiler, control=control, gather="cached", projection_schedule="resident"
            )
            regions = (
                {(s.weight_base, s.weight_bytes) for s in plan.stages if s.matrix_shape} if weight == "NVFP4" else set()
            )
            return (
                plan.assembly(marked=True),
                sorted(regions),
                dict(
                    hbm_allocated_bytes=plan.arena.size,
                    stages=[{k: v for k, v in asdict(s).items() if k != "assembly"} for s in plan.stages],
                    boundaries="input norm through recurrent output projection; excludes outer residual/MoE",
                    batch_mapping="shared Matrix weight panels; private input/output/state; serial request tiles",
                    projection_workspace_bytes=58 * 4096,
                    projection_reserved_codec_bytes=6 * 4096,
                    baseline="compact coefficients cached in existing Vector SRAM; ordinary BF16 row/tree instructions"
                    if control == "old_isa"
                    else "SRAM-reused software coefficient packing; fused update; BF16 tree",
                ),
            )

        return self.cached(
            dict(
                type="layer",
                kind=kind,
                batch=batch,
                control=control,
                weight=weight,
                gather="cached",
                projection_schedule="resident",
            ),
            generate,
        )

    def projection(self, batch, k, n, weight="BF16"):
        if weight not in ("BF16", "NVFP4"):
            raise ValueError("unsupported weight contract")

        def generate():
            _, Projection, _, _, _ = compiler_api(self.compiler)
            from compiler.aten.plena.isa_matrix_projection import lower_resident_projection

            a = ShapeArena()
            zero = a.add(2048)
            spec = Projection(0, 0, 0, zero, k, n, 256)
            inputs = [a.add(spec.input_values) for _ in range(batch)]
            weights = a.add(spec.weight_bytes // 2)
            outputs = [a.add(spec.output_values) for _ in range(batch)]
            p = Projection(inputs[0], weights, outputs[0], zero, k, n, 256)
            text = lower_resident_projection(p, inputs, outputs)
            return (
                "; @operator=projection\n" + text,
                [(weights, p.weight_bytes)] if weight == "NVFP4" else [],
                dict(
                    shape=[batch, n, k],
                    hbm_allocated_bytes=a.size,
                    weight_bytes=p.weight_bytes,
                    matrix_capacity_bytes=self.profile.matrix.matrix_capacity_bytes,
                    vector_live_bytes=58 * 4096
                    + (self.profile.codec.input_bytes + self.profile.codec.output_bytes if weight == "NVFP4" else 0),
                    mapping="existing M_MV; output32 panels, K256; bounded input-window cache and private output rows; shared weight panels when they fit",
                ),
            )

        return self.cached(dict(type="projection", batch=batch, k=k, n=n, weight=weight, schedule="resident"), generate)

    def vector(self, kind, values, *, groups=1):
        """Compile actual norm/elementwise/gate programs including their DMA.

        Each group is private. No surrounding stage is charged by MAC count.
        Large working sets are HBM-backed; existing Vector rows are reused.
        """

        def generate():
            c, _, _, _, _ = compiler_api(self.compiler)
            from compiler.aten.plena.prepared_vector_recurrence import _Emitter

            a = ShapeArena()
            rows = (values + 2047) // 2048
            a.add(2048)  # reserved zero row in the common Vector ABI
            constants = a.add(len(c.GATE_CONSTANTS) * 2048)
            weights = a.add(rows * 2048)
            text = []
            for i in range(groups):
                x, y, out = (a.add(rows * 2048) for _ in range(3))
                if kind == "norm":
                    code = c.lower_global_rms(x, weights, out, values)
                elif kind in ("mul", "sigmoid_mul", "silu"):
                    code = c.lower_pointwise_rows(
                        x, x if kind == "silu" else y, out, rows, sigmoid_input=kind != "mul", constants_base=constants
                    )
                elif kind == "softmax":
                    mask = a.add(2048)
                    code = c.lower_softmax_rows(x, y, out, values, mask)
                elif kind == "dot":
                    onehot = a.add(2048)
                    code = c.lower_dot_rows(x, weights, out, onehot, values)
                elif kind == "positive_normalize":
                    code = c.lower_positive_normalize(x, out, values)
                elif kind == "rope64":
                    if values != 64:
                        raise ValueError("this rotary mapping requires head dimension 64")
                    onehot = a.add(2048)
                    zero = a.add(2048)
                    rotated = a.add(2048)
                    product = a.add(2048)
                    # Signed sine encodes the negate on the first half.
                    # cos/sin are static position tables, not computed for free.
                    sine = a.add(2048)
                    mapping = [(x, (i + 32) % 64) for i in range(64)]
                    code = c.lower_bf16_gather(mapping, rotated, zero, onehot, strategy="pattern")
                    code += c.lower_pointwise_rows(x, weights, product, 1)
                    code += c.lower_pointwise_rows(rotated, sine, out, 1)
                    e = _Emitter(2048, True)
                    e.transfer(0, product)
                    e.transfer(1, out)
                    e.binary("ADD", 0, 0, 1)
                    e.transfer(0, out, store=True)
                    code += "\n".join(e.lines) + "\n"
                else:
                    e = _Emitter(2048, True)
                    for r in range(rows):
                        e.transfer(0, x + r * 4096)
                        if kind == "copy":
                            pass
                        elif kind == "relu2":
                            e.address(1, 0)
                            e.address(2, 0)
                            e.lines.append("V_MAX_VF gp1, gp2, f0, 0")
                            e.binary("MUL", 0, 0, 0)
                        elif kind == "add":
                            e.transfer(1, y + r * 4096)
                            e.binary("ADD", 0, 0, 1)
                        else:
                            raise ValueError("unknown vector service")
                        e.transfer(0, out + r * 4096, store=True)
                    code = "\n".join(e.lines) + "\n"
                text.append(f"; @operator={kind}_{i}\n" + code)
            return (
                "".join(text),
                [],
                dict(
                    values=values,
                    groups=groups,
                    hbm_allocated_bytes=a.size,
                    storage="HBM intermediates; up to 26 existing Vector rows",
                ),
            )

        return self.cached(dict(type="vector", kind=kind, values=values, groups=groups), generate)

    def kv_append(self, batch, heads, key_dim, value_dim, context):
        """Finite masked word RMW for packed K; contiguous words for packed V.

        Memory service is executed, while 12 serial issue/port/ALU cycles per
        masked word are an explicit orchestration budget. No free append or
        free layout conversion. K/V live in separate packed allocations.
        """
        cost = ProgramCost()
        base = 64
        for _ in range(batch * heads):
            for r in range(key_dim):
                address = base + (context // 32 * key_dim + r) * 64
                cost.memory_trace.extend([("d", 4, 0), ("r", address, 64), ("d", 8, 0), ("w", address, 64)])
                cost.transfers["read", 64] += 1
                cost.transfers["write", 64] += 1
                cost.issue += 4
                cost.scalar += 2
                cost.sram += 4
                cost.arithmetic += 2
            base += ((context + 32) // 32) * key_dim * 64 + 4096
            for c in range((value_dim + 31) // 32):
                address = base + (c * (context + 32) + context) * 64
                cost.memory_trace.extend([("d", 6, 0), ("w", address, 64)])
                cost.transfers["write", 64] += 1
                cost.issue += 2
                cost.scalar += 1
                cost.sram += 2
                cost.arithmetic += 1
            base += ((value_dim + 31) // 32) * (context + 32) * 64 + 4096
        self.backend.price(cost, self.profile.dma.service)
        return dict(
            components=cost.components(),
            hbm_read_bytes=sum(n * c for (d, n), c in cost.transfers.items() if d == "read"),
            hbm_write_bytes=sum(n * c for (d, n), c in cost.transfers.items() if d == "write"),
            metadata=dict(
                validation="Ramulator memory + finite masked-word orchestration budget",
                input_buffer_bytes=4096,
                word_buffer_bytes=64,
            ),
        )
