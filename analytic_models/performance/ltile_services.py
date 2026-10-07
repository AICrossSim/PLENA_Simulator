"""Reusable compiled operator services and version-bound analytical cache."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import uuid

from .ltile_layers import build_batch, compiler_api
from .ltile_program import ShapeArena
from .ltile_execution import price_program
from .ltile_dma import sha


def select_projection_panels(stages, candidates):
    """Rank only like-for-like stages; the caller must reprice their composition."""
    if set(candidates) != {1, 2, 4, 8}:
        raise ValueError("projection search requires the four legal panel schedules")
    names = [s["name"] for s in stages]
    if len(set(names)) != len(names):
        raise ValueError("duplicate stage names")
    sections = {}
    for panels, costs in candidates.items():
        sections[panels] = {s["name"]: s for s in costs}
        if len(sections[panels]) != len(costs) or set(sections[panels]) != set(names):
            raise ValueError("projection candidate stages differ")
        if any(not isinstance(s["total"], (int, float)) or not 0 <= s["total"] < float("inf") for s in costs):
            raise ValueError("projection costs must be finite and nonnegative")
    return {
        s["name"]: min(candidates, key=lambda n: (sections[n][s["name"]]["total"], n))
        for s in stages if s["matrix_shape"]
    }


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
                "ltile_access.py",
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
                "ltile_native.py",
                "isa_matrix_projection.py",
                "isa_projection_software.py",
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

    def layer(self, kind, batch, control, weight="BF16", *, supply="packed"):
        if weight not in ("BF16", "NVFP4"):
            raise ValueError("unsupported weight contract")
        if self.profile.projection_schedule == "transposed" and weight != "BF16":
            raise ValueError("transposed static weight packing is validated only for BF16")
        if weight != "BF16" and self.profile.projection_codec_rows != 6:
            raise ValueError("compressed weights require the reserved codec workspace")
        if supply not in ("packed", "native") or (supply == "native" and control != "fsm"):
            raise ValueError("native coefficient supply requires the resident FSM interface")

        def generate():
            plan = build_batch(
                kind, batch, self.compiler, control=control, gather="cached", projection_schedule=self.profile.projection_schedule,
                native_coefficients=supply == "native",
                vector_rows=self.profile.projection_vector_rows,
                gather_vector_rows=self.profile.gather_vector_rows,
                projection_n_panel_tile=self.profile.projection_n_panel_tile,
                projection_panel_overrides=self.profile.projection_panel_overrides,
                projection_request_tile=self.profile.projection_request_tile,
                projection_k_tile=self.profile.projection_k_tile,
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
                    batch_mapping="shared Matrix weight panels; private input/output/state; bounded request tiles",
                    projection_schedule=self.profile.projection_schedule,
                    projection_n_panel_tile=self.profile.projection_n_panel_tile,
                    projection_panel_overrides=dict(self.profile.projection_panel_overrides),
                    projection_resources=self.profile.matrix.projection_resources()
                    if self.profile.projection_schedule in ("compact", "batch") or self.profile.matrix.weight_replay
                    else None,
                    projection_workspace_bytes=self.profile.projection_vector_rows * 4096,
                    projection_reserved_codec_bytes=self.profile.projection_codec_rows * 4096,
                    projection_request_tile=self.profile.projection_request_tile,
                    projection_k_tile=self.profile.projection_k_tile,
                    coefficient_supply=supply,
                    baseline="compact coefficients cached in existing Vector SRAM; ordinary BF16 row/tree instructions"
                    if control == "old_isa"
                    else ("native coefficient descriptors; bounded sector/state/result buffers; fused update; BF16 tree"
                          if supply == "native" else "SRAM-reused software coefficient packing; fused update; BF16 tree"),
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
                projection_schedule=self.profile.projection_schedule,
                coefficient_supply=supply,
            ),
            generate,
        )

    def tuned_layer(self, kind, batch, control, weight="BF16", *, supply="packed"):
        """Choose bounded per-projection loop nests, then reprice the program.

        No Rust observations or saved speedup tables enter this search. Four
        uniform compiled schedules provide local analytical candidates. Their
        best individual stages form one mixed program, which is executed by
        the memory model as a whole; stage times are never simply summed.
        Keep the best uniform program if memory-history interactions make the
        mixed program slower. This is offline compilation, not runtime logic.
        """
        if self.profile.projection_schedule not in ("compact", "batch"):
            raise ValueError("projection tuning requires compact/batch lowering")
        if self.profile.projection_panel_overrides:
            raise ValueError("projection tuning starts from uniform candidate schedules")
        variants = {}
        for panels in (1, 2, 4, 8):
            profile = replace(self.profile, projection_n_panel_tile=panels)
            variants[panels] = Services(self.compiler, profile, self.backend, self.cache).layer(
                kind, batch, control, weight, supply=supply,
            )
        reference = variants[1]["metadata"]["stages"]
        if any(result["metadata"]["stages"] != reference for result in variants.values()):
            raise ValueError("projection candidates changed stage contracts or HBM allocations")
        choices = select_projection_panels(reference, {n: r["sections"] for n, r in variants.items()})
        profile = replace(self.profile, projection_n_panel_tile=1, projection_panel_overrides=tuple(choices.items()))
        mixed = Services(self.compiler, profile, self.backend, self.cache).layer(
            kind, batch, control, weight, supply=supply,
        )
        uniform = min(variants, key=lambda n: (variants[n]["components"]["total"], n))
        use_mixed = mixed["components"]["total"] < variants[uniform]["components"]["total"]
        selected = dict(mixed if use_mixed else variants[uniform])
        selected["projection_search"] = dict(
            policy="per_operator_analytical_v1",
            candidate_totals={str(n): r["components"]["total"] for n, r in variants.items()},
            candidate_assembly_sha256={str(n): r["assembly_sha256"] for n, r in variants.items()},
            mixed_panels=choices,
            mixed_total=mixed["components"]["total"],
            mixed_assembly_sha256=mixed["assembly_sha256"],
            best_uniform_panels=uniform,
            selected="mixed" if use_mixed else "uniform",
            no_regression_guard="whole-program analytical total including memory service",
        )
        return selected

    def projection(self, batch, k, n, weight="BF16"):
        if weight not in ("BF16", "NVFP4"):
            raise ValueError("unsupported weight contract")
        if self.profile.projection_schedule == "transposed" and weight != "BF16":
            raise ValueError("transposed static weight packing is validated only for BF16")
        if weight != "BF16" and self.profile.projection_codec_rows != 6:
            raise ValueError("compressed weights require the reserved codec workspace")
        overrides = dict(self.profile.projection_panel_overrides)
        if overrides.keys() - {"projection"}:
            raise ValueError("standalone projection accepts only the projection stage override")

        def generate():
            _, Projection, _, _, _ = compiler_api(self.compiler)
            from compiler.aten.plena.isa_projection_software import lower_software_projection, lower_transposed_projection

            a = ShapeArena()
            zero = a.add(2048)
            spec = Projection(0, 0, 0, zero, k, n, self.profile.projection_k_tile)
            inputs = [a.add(spec.input_values) for _ in range(batch)]
            weights = a.add(spec.weight_bytes // 2)
            outputs = [a.add(spec.output_values) for _ in range(batch)]
            p = Projection(inputs[0], weights, outputs[0], zero, k, n, self.profile.projection_k_tile)
            if self.profile.projection_schedule == "transposed":
                text = lower_transposed_projection(p, inputs, outputs, vector_rows=self.profile.projection_vector_rows)
            elif self.profile.projection_schedule == "resident":
                text = lower_software_projection(
                    p, inputs, outputs, vector_rows=self.profile.projection_vector_rows,
                    request_tile=self.profile.projection_request_tile,
                )
            else:
                from compiler.aten.plena.isa_matrix_projection import lower_compact_projection

                text = lower_compact_projection(
                    p, inputs, outputs,
                    batch_tile=1 if self.profile.projection_schedule == "compact" else 4,
                    vector_rows=self.profile.projection_vector_rows,
                    n_panel_tile=overrides.get("projection", self.profile.projection_n_panel_tile),
                )
            return (
                "; @operator=projection\n" + text,
                [(weights, p.weight_bytes)] if weight == "NVFP4" else [],
                dict(
                    shape=[batch, n, k],
                    hbm_allocated_bytes=a.size,
                    weight_bytes=p.weight_bytes,
                    projection_request_tile=self.profile.projection_request_tile,
                    projection_k_tile=self.profile.projection_k_tile,
                    matrix_capacity_bytes=self.profile.matrix.matrix_capacity_bytes,
                    vector_live_bytes=self.profile.projection_vector_rows * 4096
                    + (self.profile.codec.input_bytes + self.profile.codec.output_bytes if weight == "NVFP4" else 0),
                    mapping=(
                        "existing M_MV; output32 panels, K256; bounded input-window cache; shared weight panels"
                        if self.profile.projection_schedule == "resident"
                        else "existing M_TMV; static N-by-K packets; serial row reads; no replay; explicit K reduction contract"
                        if self.profile.projection_schedule == "transposed"
                        else "M_MM.P; K256/output32; aligned compact input rows; finite replay; masked tails; bounded request tiles"
                    ),
                    projection_resources=self.profile.matrix.projection_resources()
                    if self.profile.projection_schedule in ("compact", "batch") or self.profile.matrix.weight_replay
                    else None,
                ),
            )

        return self.cached(dict(type="projection", batch=batch, k=k, n=n, weight=weight, schedule=self.profile.projection_schedule), generate)

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
        """Actual old-ISA append to packed K/V, shared by all comparison arms.

        Preserves whole owned Vector rows using immutable compile-time masks.
        Only the new values are gathered; no invented masked-word opcode.
        Allocations are local service addresses, not a whole-model address map.
        """
        if min(batch, heads, key_dim, value_dim) < 1 or context < 0:
            raise ValueError("invalid KV append shape")

        def generate():
            c, _, _, _, _ = compiler_api(self.compiler)
            a = ShapeArena()
            zero, one_hot, scratch = (a.add(2048) for _ in range(3))
            code = []
            for request in range(batch * heads):
                for k, n, column in ((key_dim, context + 1, True), (context + 1, value_dim, False)):
                    packed = ((k + 31)//32*32) * ((n + 31)//32*32)
                    base = a.add(((packed + 2047)//2048 + 1)*2048)
                    source = a.add(((k if column else n) + 2047)//2048*2048)
                    masks = {}
                    constants = {}
                    for row, edits in c.packed_append_edits(k,n,column=column).items():
                        pattern = tuple(edits)
                        if pattern not in constants:
                            constants[pattern] = a.add(2048)
                        masks[row] = constants[pattern]
                    code.append(c.lower_packed_matrix_append(
                        base,k,n,source,zero,one_hot,scratch,column=column,keep_masks=masks))
            return "; @operator=kv_append\n" + "".join(code), [], dict(
                validation="compiled gather + masked Vector RMW; connected Rust numerical checks",
                hbm_allocated_bytes=a.size, vector_live_bytes=64*4096,
                constants="BF16 keep masks: ones except updated lanes; immutable shape metadata",
                precision="finite BF16; zero-sign preservation not guaranteed",
            )

        return self.cached(dict(type="kv_append",batch=batch,heads=heads,key_dim=key_dim,
                                value_dim=value_dim,context=context,schedule="masked_vector_rmw"), generate)
