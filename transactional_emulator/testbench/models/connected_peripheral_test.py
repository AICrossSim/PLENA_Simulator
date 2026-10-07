"""Connected peripheral checks with owned buffers and independent references.

Programs execute all intermediate production/packing in Rust. These are
representative operator compositions, not full-checkpoint quality tests.
"""

import argparse
import json
from pathlib import Path
import numpy as np

from .unified_service_test import profile_for
from transactional_emulator.testbench.aten.recurrent_gate_test import Arena, bf, exp
from transactional_emulator.testbench.aten.recurrent_conv_test import run_program, read
from transactional_emulator.testbench.aten.matrix_projection_test import reference
from compiler.aten.plena.isa_matrix_projection import Projection, lower_resident_projection
from compiler.aten.plena.recurrent_coefficients import (
    lower_bf16_gather,
    lower_softmax_rows,
    lower_global_rms,
    lower_pointwise_rows,
    lower_packed_matrix_append,
    packed_append_edits,
)
from compiler.aten.plena.prepared_vector_recurrence import _Emitter


def tree(x):
    x = np.asarray(x, np.float32)
    while x.size > 1:
        x = bf(x[::2] + x[1::2])
    return x[0]


class Program:
    def __init__(self, memory, seed):
        self.a, self.code, self.refs = Arena(), [], []
        self.p = profile_for(memory)
        self.rng = np.random.default_rng(seed)
        self.fp = [-16384.0]
        self.zero = self.a.add(np.zeros(2048))
        self.onehot = self.a.add(np.eye(1, 2048).ravel())

    def data(self, x, output=False):
        x = np.asarray(x).ravel()
        # Matrix K windows require an extra owned Vector row beyond valid K.
        return self.a.add(np.pad(x, (0, ((x.size + 2047) // 2048 + 1) * 2048 - x.size)), output=output)

    def out(self, n):
        return self.data(np.zeros(n), output=True)

    def emit(self, name, text):
        self.code.append("; @operator=" + name + "\n" + text)

    def check(self, name, address, x):
        self.refs.append((name, address, np.asarray(x, np.float32).ravel()))

    def weights(self, w, mutable=False):
        k, n = w.shape
        padded = np.pad(w, ((0, (-k) % 32), (0, (-n) % 32)))
        packed = padded.reshape(len(padded), -1, 32).transpose(1, 0, 2).ravel()
        return self.data(packed, output=mutable)

    def project(self, name, sources, xs, w, weight_base=None):
        k, n = w.shape
        wb = self.weights(w) if weight_base is None else weight_base
        outputs = [self.out(n) for _ in sources]
        spec = Projection(sources[0], wb, outputs[0], self.zero, k, n, 256)
        self.emit(name, lower_resident_projection(spec, sources, outputs))
        ys = [reference(x, w, 256, self.p.matrix) for x in xs]
        for i, (address, y) in enumerate(zip(outputs, ys)):
            self.check(f"{name}/{i}", address, y)
        return outputs, ys

    def random_weight(self, k, n):
        return bf(self.rng.normal(0, 0.08, (k, n)))

    def gather(self, name, mapping, expected):
        out = self.out(len(mapping))
        self.emit(name, lower_bf16_gather(mapping, out, self.zero, self.onehot, strategy="cached"))
        self.check(name, out, expected)
        return out

    def scale(self, name, source, x, factor):
        constant = self.data(np.full(len(x), factor))
        out = self.out(len(x))
        self.emit(name, lower_pointwise_rows(source, constant, out, (len(x) + 2047) // 2048))
        y = bf(x * bf(factor))
        self.check(name, out, y)
        return out, y

    def normalize(self, name, source, x):
        n = len(x)
        out = self.out(n)
        weight = self.data(np.ones(n))
        epsilon = len(self.fp)
        self.fp += [n * 1e-5, float(np.sqrt(n))]
        self.emit(name, lower_global_rms(source, weight, out, n, epsilon_slot=epsilon, scale_slot=epsilon + 1))
        padded = np.pad(bf(x * x), (0, (-n) % 2048))
        total = np.float32(0)
        for row in padded.reshape(-1, 2048):
            total = bf(total + tree(row))[0]
        inv = bf(1 / bf(np.sqrt(bf(total + bf(n * 1e-5)))))[0]
        y = bf(x * bf(inv * bf(np.sqrt(n))))
        self.check(name, out, y)
        return out, y

    def rope(self, name, source, x, position):
        assert len(x) == 64
        angle = position / (10000 ** (np.arange(32, dtype=np.float32) / 32))
        cosine = bf(np.tile(np.cos(angle), 2))
        sine = bf(np.concatenate((-np.sin(angle), np.sin(angle))))
        rotated = self.gather(name + "_swap", [(source, (i + 32) % 64) for i in range(64)], np.roll(x, 32))
        prod, out = self.out(64), self.out(64)
        self.emit(name + "_cos", lower_pointwise_rows(source, self.data(cosine), prod, 1))
        self.emit(name + "_sin", lower_pointwise_rows(rotated, self.data(sine), out, 1))
        e = _Emitter(2048, True)
        e.transfer(0, prod)
        e.transfer(1, out)
        e.binary("ADD", 0, 0, 1)
        e.transfer(0, out, store=True)
        self.emit(name + "_add", "\n".join(e.lines) + "\n")
        y = bf(bf(x * cosine) + bf(np.roll(x, 32) * sine))
        self.check(name, out, y)
        return out, y

    def append(self, name, base, w, source, values, *, column):
        """Masked append to owned packed K/V, using actual gather/store ISA."""
        k, n = w.shape
        masks = {}
        for row, edits in packed_append_edits(k, n, column=column).items():
            mask = np.ones(2048)
            mask[list(edits)] = 0
            masks[row] = self.data(mask)
        self.emit(
            name,
            lower_packed_matrix_append(
                base, k, n, source, self.zero, self.onehot, self.out(2048), column=column, keep_masks=masks
            ),
        )
        if column:
            w[:, -1] = values
        else:
            w[-1, :] = values
        padded = np.pad(w, ((0, (-k) % 32), (0, (-n) % 32)))
        self.check(name, base, padded.reshape(len(padded), -1, 32).transpose(1, 0, 2))

    def attention(self, name, query, q, kbase, key, vbase, value):
        logits, scores = self.project(name + "_qk", [query], [q], key, kbase)
        n = key.shape[1]
        tmp, prob = self.out(n), self.out(n)
        mask = self.data(np.ones(n % 2048 or 2048))
        self.emit(name + "_softmax", lower_softmax_rows(logits[0], tmp, prob, n, mask))
        full = np.pad(scores[0], (0, (-n) % 2048))
        ex = exp(bf(full - full.max()))
        ex[n:] = 0
        total = np.float32(0)
        for row in ex.reshape(-1, 2048):
            total = bf(total + tree(row))[0]
        p = bf(ex[:n] * bf(1 / total))
        self.check(name + "_probability", prob, p)
        outputs, ys = self.project(name + "_pv", [prob], [p], value, vbase)
        return outputs[0], ys[0]

    def run(self, root, runtime, memory, scope):
        image, result = run_program(
            root,
            runtime,
            memory,
            self.a,
            "".join(self.code),
            profile=self.p,
            fp_constants=self.fp,
            references=self.refs,
        )
        for name, address, x in self.refs:
            np.testing.assert_array_equal(read(image, address, x.size), x, err_msg=name)
        return dict(
            status="passed",
            scope=scope,
            checked_values=sum(x.size for _, _, x in self.refs),
            checked_boundaries=[n for n, _, _ in self.refs],
            hbm_owned_bytes=len(self.a.data),
            **result,
        )


def attention_case(root, runtime, memory, *, batch=2, keys=65, model_shape=False):
    p = Program(memory, 915 + batch + keys)
    hidden, width, heads, kv_heads = (2688, 128, 32, 2) if model_shape else (64, 128, 2, 1)
    xs = [bf(p.rng.normal(0, 0.1, hidden)) for _ in range(batch)]
    inputs = [p.data(x) for x in xs]
    if model_shape:
        combined, values = p.project(
            "qkv_projection", inputs, xs, p.random_weight(hidden, (heads + 2 * kv_heads) * width)
        )
        qs, qv = combined, [x[: heads * width] for x in values]
        ks = [address + 2 * heads * width for address in combined]
        kv = [x[heads * width : (heads + kv_heads) * width] for x in values]
        vs = [address + 2 * (heads + kv_heads) * width for address in combined]
        vv = [x[(heads + kv_heads) * width :] for x in values]
    else:
        qs, qv = p.project("q_projection", inputs, xs, p.random_weight(hidden, heads * width))
        ks, kv = p.project("k_projection", inputs, xs, p.random_weight(hidden, kv_heads * width))
        vs, vv = p.project("v_projection", inputs, xs, p.random_weight(hidden, kv_heads * width))
    concatenated, expected = [], []
    for request in range(batch):
        caches = []
        for group in range(kv_heads):
            key = bf(p.rng.normal(0, 0.1, (width, keys)))
            key[:, -1] = 0
            value = bf(p.rng.normal(0, 0.1, (keys, width)))
            value[-1] = 0
            kb, vb = p.weights(key, True), p.weights(value, True)
            start = group * width
            p.append(
                f"r{request}/g{group}/key_append",
                kb,
                key,
                ks[request] + 2 * start,
                kv[request][start : start + width],
                column=True,
            )
            p.append(
                f"r{request}/g{group}/value_append",
                vb,
                value,
                vs[request] + 2 * start,
                vv[request][start : start + width],
                column=False,
            )
            caches.append((kb, key, vb, value))
        sources, ys = [], []
        for h in range(heads):
            q = qv[request][h * width : (h + 1) * width]
            qb = p.gather(
                f"r{request}/q{h}",
                [(qs[request] + (h * width + i) // 2048 * 4096, (h * width + i) % 2048) for i in range(width)],
                q,
            )
            qb, q = p.scale(f"r{request}/scale{h}", qb, q, 1 / np.sqrt(width))
            kb, key, vb, value = caches[h // (heads // kv_heads)]
            ob, y = p.attention(f"r{request}/head{h}", qb, q, kb, key, vb, value)
            sources.append(ob)
            ys.append(y)
        expected.append(np.concatenate(ys))
        concatenated.append(
            p.gather(f"r{request}/concat", [(a, i) for a in sources for i in range(width)], expected[-1])
        )
    p.project("output_projection", concatenated, expected, p.random_weight(heads * width, hidden))
    result = p.run(
        root,
        runtime,
        memory,
        "GQA projections -> request-private KV append -> scaled QK/softmax/PV -> output projection; shared KV groups; excludes outer norm/residual; synthetic inputs and weights",
    )
    result.update(
        batch=batch,
        keys=keys,
        hidden=hidden,
        query_heads=heads,
        kv_heads=kv_heads,
        head_dim=width,
        model_shape=model_shape,
    )
    return result


def mla_case(root, runtime, memory, *, keys=65):
    p = Program(memory, 4261 + keys)
    hidden, latent, nope, rope, vd, heads = 64, 512, 128, 64, 128, 2
    x = bf(p.rng.normal(0, 0.1, hidden))
    xb = p.data(x)
    low, lv = p.project("q_low", [xb], [x], p.random_weight(hidden, 128))
    low, l = p.normalize("q_norm", low[0], lv[0])
    qbase, qvalue = p.project("q_up", [low], [l], p.random_weight(128, heads * (nope + rope)))
    cb, cv = p.project("kv_latent", [xb], [x], p.random_weight(hidden, latent))
    cb, c = p.normalize("kv_norm", cb[0], cv[0])
    rb, rv = p.project("key_rope_projection", [xb], [x], p.random_weight(hidden, rope))
    rb, r = p.rope("key_rope", rb[0], rv[0], keys - 1)
    newk = np.concatenate((c, r))
    newkb = p.gather("key_concat", [(cb, i) for i in range(latent)] + [(rb, i) for i in range(rope)], newk)
    cache = bf(p.rng.normal(0, 0.1, (latent, keys)))
    cache[:, -1] = 0
    positions = bf(p.rng.normal(0, 0.1, (rope, keys)))
    positions[:, -1] = 0
    key = np.concatenate((cache, positions))
    value = cache.T.copy()
    kb, vb = p.weights(key, True), p.weights(value, True)
    p.append("latent_key_append", kb, key, newkb, newk, column=True)
    p.append("latent_value_append", vb, value, cb, c, column=False)
    outputs, ys = [], []
    for h in range(heads):
        q = qvalue[0][h * (nope + rope) : (h + 1) * (nope + rope)]
        nb = p.gather(f"q_nope{h}", [(qbase[0], h * (nope + rope) + i) for i in range(nope)], q[:nope])
        rqb = p.gather(f"q_rope{h}", [(qbase[0], h * (nope + rope) + nope + i) for i in range(rope)], q[nope:])
        rqb, rq = p.rope(f"query_rope{h}", rqb, q[nope:], keys - 1)
        ab, av = p.project(f"absorb_key{h}", [nb], [q[:nope]], p.random_weight(nope, latent))
        aq = np.concatenate((av[0], rq))
        aqb = p.gather(f"absorbed_query{h}", [(ab[0], i) for i in range(latent)] + [(rqb, i) for i in range(rope)], aq)
        aqb, aq = p.scale(f"query_scale{h}", aqb, aq, 1 / np.sqrt(nope + rope))
        ob, y = p.attention(f"latent_attention{h}", aqb, aq, kb, key, vb, value)
        ob, yv = p.project(f"value_expand{h}", [ob], [y], p.random_weight(latent, vd))
        outputs.append(ob[0])
        ys.append(yv[0])
    y = np.concatenate(ys)
    yb = p.gather("heads_concat", [(a, i) for a in outputs for i in range(vd)], y)
    p.project("mla_output_projection", [yb], [y], p.random_weight(heads * vd, hidden))
    return p.run(
        root,
        runtime,
        memory,
        "absorbed MLA: q low/norm/up, latent KV norm, RoPE, private incremental dual-orientation KV, absorbed QK/softmax/PV/value expansion/output; representative two-head synthetic block, excludes output gate and outer residual",
    )


def experts_case(root, runtime, memory, *, batch=4, model_shape=False):
    """Uneven fixed routes, shared panels, two expert activation contracts."""
    from compiler.aten.plena.recurrent_coefficients import GATE_CONSTANTS
    from transactional_emulator.testbench.aten.recurrent_gate_test import softplus

    p = Program(memory, 527 + batch)
    hidden, mid = (2688, 1856) if model_shape else (65, 257)
    p.fp = [0.0]
    xs = [bf(p.rng.normal(0, 0.15, hidden)) for _ in range(batch)]
    inputs = [p.data(x) for x in xs]
    constants = p.data(np.repeat(np.array(GATE_CONSTANTS), 2048))
    contributions = [[] for _ in xs]
    for expert in range(3):
        tokens = list(range(batch)) if expert == 0 else [i for i in range(batch) if i % 2 == expert - 1]
        addresses, values = p.project(
            f"expert{expert}_up", [inputs[i] for i in tokens], [xs[i] for i in tokens], p.random_weight(hidden, mid)
        )
        active, expected = [], []
        if expert == 0 and not model_shape:
            gate, gv = p.project(
                "expert0_gate", [inputs[i] for i in tokens], [xs[i] for i in tokens], p.random_weight(hidden, mid)
            )
        for index, (address, value) in enumerate(zip(addresses, values)):
            out = p.out(mid)
            if expert == 0 and not model_shape:
                temp = p.out(mid)
                p.emit(
                    "silu",
                    lower_pointwise_rows(
                        gate[index], gate[index], temp, 1, sigmoid_input=True, constants_base=constants
                    ),
                )
                p.emit("swiglu", lower_pointwise_rows(temp, address, out, 1))
                expected.append(bf(bf(exp(bf(-softplus(bf(-gv[index])))) * gv[index]) * value))
            else:
                e = _Emitter(2048, True)
                e.transfer(0, address)
                e.address(1, 0)
                e.address(2, 0)
                e.lines.append("V_MAX_VF gp1, gp2, f0, 0")
                e.binary("MUL", 0, 0, 0)
                e.transfer(0, out, store=True)
                p.emit("relu2", "\n".join(e.lines) + "\n")
                expected.append(bf(np.maximum(value, 0) ** 2))
            active.append(out)
            p.check(f"expert{expert}_activation{index}", out, expected[-1])
        outputs, ys = p.project(f"expert{expert}_down", active, expected, p.random_weight(mid, hidden))
        for token, address, y in zip(tokens, outputs, ys):
            contributions[token].append((address, y, 0.25 if expert == 0 else 0.75))
    for token, parts in enumerate(contributions):
        if model_shape:
            out = p.out(hidden)
            factors = [p.data(np.full(2048, factor)) for _, _, factor in parts]
            e = _Emitter(2048, True)
            for row in range((hidden + 2047) // 2048):
                e.transfer(0, p.zero)
                for (address, _, _), factor_base in zip(parts, factors):
                    e.transfer(1, address + row * 4096)
                    e.transfer(2, factor_base)
                    e.binary("MUL", 1, 1, 2)
                    e.binary("ADD", 0, 0, 1)
                e.transfer(0, out + row * 4096, store=True)
            p.emit("weighted_combine", "\n".join(e.lines) + "\n")
            expected = np.zeros(hidden, np.float32)
            for _, y, factor in parts:
                expected = bf(expected + bf(y * factor))
            p.check(f"combined{token}", out, expected)
            continue
        e = _Emitter(2048, True)
        e.transfer(0, p.zero)
        expected = np.zeros(hidden, np.float32)
        for address, y, factor in parts:
            e.transfer(1, address)
            e.transfer(2, p.data(np.full(hidden, factor)))
            e.binary("MUL", 1, 1, 2)
            e.binary("ADD", 0, 0, 1)
            expected = bf(expected + bf(y * factor))
        out = p.out(hidden)
        e.transfer(0, out, store=True)
        p.emit("weighted_combine", "\n".join(e.lines) + "\n")
        p.check(f"combined{token}", out, expected)
    result = p.run(
        root,
        runtime,
        memory,
        (
            "Nemotron expert widths 2688/1856, ReLU2, B16-capable private tokens/shared weights; "
            "three-expert subset with fixed two contributions/token; not a full top6 router; synthetic inputs/weights"
            if model_shape
            else "fixed uneven routes; private tokens, shared resident weights; tail K/N, SwiGLU/ReLU2 and weighted combine; top-k selection supplied, not executed"
        ),
    )
    result.update(batch=batch, hidden=hidden, intermediate=mid, model_shape=model_shape)
    return result


def codec_case(root, runtime, memory):
    """Functional NVFP4 decode -> Rust projection; decoder service is modeled."""
    from analytic_models.performance.weight_codec import decode_nvfp4, decode_e4m3_scale, WeightPacket

    p = Program(memory, 8501)
    k, n, batch = 273, 35, 4
    padded = (k + 31) // 32 * 32
    packed = p.rng.integers(0, 256, (n, padded // 2), dtype=np.uint8)
    scale_bits = p.rng.integers(8, 80, (n, padded // 16), dtype=np.uint8)
    w = bf(decode_nvfp4(packed, decode_e4m3_scale(scale_bits), 0.125))[:k]
    # Scalar formula independent of the table/vectorized codec implementation.
    independent = np.zeros((k, n), np.float32)
    for col in range(n):
        for row in range(k):
            bits = int(packed[col, row // 2])
            code = (bits >> (4 * (row % 2))) & 15
            exponent = (code >> 1) & 3
            mantissa = code & 1
            value = mantissa * 0.5 if exponent == 0 else (1 + mantissa * 0.5) * 2 ** (exponent - 1)
            sb = int(scale_bits[col, row // 16])
            se = (sb >> 3) & 15
            sm = sb & 7
            scale = sm * 2**-9 if se == 0 else (1 + sm / 8) * 2 ** (se - 7)
            independent[row, col] = (-1 if code & 8 else 1) * value * scale * 0.125
    np.testing.assert_array_equal(w, bf(independent))
    xs = [bf(p.rng.normal(0, 0.15, k)) for _ in range(batch)]
    p.project("decoded_projection", [p.data(x) for x in xs], xs, w)
    result = p.run(
        root,
        runtime,
        memory,
        "NVFP4 block16/E4M3 scales -> BF16 -> shared-panel tail projection; decode is Python functional, projection is Rust; no claim of executed hardware decoder",
    )
    result["codec_packet_services"] = {str(count): WeightPacket(count).service(p.p.codec) for count in (1024, 8192)}
    result["decoded_weight_values"] = k * n
    result["codec_cost_scope"] = "finite packet buffer/throughput model, independently swept; not RTL calibrated"
    return result


def private_state_case(root, runtime, memory, *, batch, supply="native"):
    """Two tokens, distinct states/coefficients, reversed second-token order."""
    from compiler.aten.plena.ltile_v2 import Options, lower_group

    p = Program(memory, 7211 + batch)
    o = Options("kda")
    h, w = o.heads, o.width
    states = [bf(p.rng.normal(0, 0.2, (h, 128, w))) for _ in range(batch)]
    bases = [p.data(s, output=True) for s in states]
    for token in range(2):
        for request in range(batch) if token == 0 else reversed(range(batch)):
            delta = bf(p.rng.uniform(0.0001, 0.03, (h, 128)))
            key = bf(p.rng.normal(0, 0.1, (h, 128)))
            query = bf(p.rng.normal(0, 0.1, (h, 128)))
            x = bf(p.rng.normal(0, 0.2, (h, w)))
            beta = bf(p.rng.uniform(0.1, 0.9, h))
            decayed = states[request] - delta[:, :, None] * states[request]

            def reduce_rows(v):
                v = bf(v)
                while v.shape[1] > 1:
                    v = bf(v[:, ::2] + v[:, 1::2])
                return v[:, 0]

            pred = reduce_rows(decayed * key[:, :, None])
            residual = bf(beta[:, None] * (x - pred))
            state = bf(decayed + key[:, :, None] * residual[:, None, :])
            out = reduce_rows(state * query[:, :, None])
            states[request] = state
            coeff = [p.data(a) for a in (delta, key, query)]
            output, snapshot = p.out(h * w), p.out(h * 128 * w)
            memory_map = dict(
                states=[bases[request]],
                input=p.data(x),
                scalar=p.data(np.stack((beta, np.zeros(h)), -1)),
                output=output,
                snapshots=[snapshot],
            )
            if supply == "native":
                memory_map["native"] = [(address, 0, 128, 1, 0) for address in coeff]
            else:
                memory_map["update"] = [p.data(np.stack((delta.T, key.T), -1))]
                memory_map["dot"] = [p.data(np.stack((query.T, np.zeros_like(query.T)), -1))]
            p.emit(f"token{token}/request{request}", "\n".join(lower_group(o, memory_map).lines) + "\n")
            p.check(f"t{token}/r{request}/state", snapshot, state)
            p.check(f"t{token}/r{request}/output", output, out)
    for request in range(batch):
        p.check(f"final/r{request}", bases[request], states[request])
    result = p.run(
        root,
        runtime,
        memory,
        f"KDA {supply} supply, one full 16-head tile group per private request; two tokens, distinct inputs and states, reversed execution order; all state/output boundaries checked",
    )
    result.update(batch=batch, tokens=2, request_state_addresses=bases, supply=supply)
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("output", "runtime", "memory-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--case", choices=("attention", "mla", "experts", "codec", "private-state"), required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--keys", type=int, default=65)
    parser.add_argument(
        "--model-shape", action="store_true", help="Nemotron GQA/expert dimensions, synthetic weights/inputs"
    )
    parser.add_argument("--supply", choices=("native", "packed"), default="native")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.case == "attention":
        result = attention_case(
            args.output, args.runtime, args.memory_root, batch=args.batch, keys=args.keys, model_shape=args.model_shape
        )
    elif args.case == "mla":
        result = mla_case(args.output, args.runtime, args.memory_root, keys=args.keys)
    elif args.case == "experts":
        result = experts_case(
            args.output, args.runtime, args.memory_root, batch=args.batch, model_shape=args.model_shape
        )
    elif args.case == "private-state":
        result = private_state_case(args.output, args.runtime, args.memory_root, batch=args.batch, supply=args.supply)
    else:
        result = codec_case(args.output, args.runtime, args.memory_root)
    (args.output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
