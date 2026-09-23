"""Held-out machine-code checks for shared analytical services and schedules."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np

from analytic_models.performance.ltile_platform import ExecutionProfile
from transactional_emulator.testbench.aten.recurrent_gate_test import Arena, bf, digest, exp
from transactional_emulator.testbench.aten.recurrent_conv_test import run_program, read
from transactional_emulator.testbench.aten.matrix_projection_test import reference
from compiler.aten.plena.isa_matrix_projection import Projection, lower_batch_projection
from compiler.aten.plena.recurrent_coefficients import CompactCoefficientLoader, lower_bf16_gather, lower_softmax_rows
from compiler.aten.plena.prepared_vector_recurrence import PreparedVectorGroup, lower_prepared_vector_recurrence
from compiler.aten.plena.matrix_recurrence_lowering import NEMOTRON_MAMBA, KIMI_KDA


def profile_for(memory):
    config = json.loads((memory / "ramulator.json").read_text())
    return ExecutionProfile(hbm_controllers=len(config["memory_system"]["controllers"]))


def matrix_case(root, runtime, memory, b, k, n):
    rng = np.random.default_rng(3107 + b + k + n)
    arena = Arena()
    zero = arena.add(np.zeros(2048))
    h = profile_for(memory)
    spec = Projection(0, 0, 0, 0, k, n, 256)
    xs = [bf(rng.normal(0, 0.2, k)) for _ in range(b)]
    inputs = [arena.add(np.pad(x, (0, spec.input_values - k))) for x in xs]
    w = bf(rng.normal(0, 0.1, (k, n)))
    padded = np.pad(w, ((0, (-k) % 32), (0, (-n) % 32)))
    weights = arena.add(padded.reshape(len(padded), -1, 32).transpose(1, 0, 2).copy())
    outputs = [arena.add(np.full(spec.output_values, 7), output=True) for _ in xs]
    spec = replace(spec, inputs=inputs[0], weights=weights, outputs=outputs[0], zero=zero)
    image, result = run_program(root, runtime, memory, arena, lower_batch_projection(spec, inputs, outputs), profile=h)
    for x, address in zip(xs, outputs):
        if not np.array_equal(read(image, address, n), reference(x, w, 256, h.matrix)):
            raise AssertionError("shared-panel numerical mismatch")
    return dict(
        case=root.name,
        **result,
        checked_values=b * n,
        status="passed",
        scope="M_MV panel reuse, private requests, tail K/N, final writeback",
    )


def gather_case(root, runtime, memory, strategy="grouped"):
    rng = np.random.default_rng(741)
    arena = Arena()
    zero = arena.add(np.zeros(2048))
    one = arena.add(np.eye(1, 2048).ravel())
    source = bf(rng.normal(size=(3, 2048)))
    bases = [arena.add(row) for row in source]
    maps = [None if i % 7 == 0 else (bases[(i * 5 + 1) % 3], (i // 19) % 47) for i in range(2301)]
    dst = arena.add(np.full(4096, 7), output=True)
    expected = np.zeros(4096, np.float32)
    for i, pair in enumerate(maps):
        if pair:
            expected[i] = source[bases.index(pair[0]), pair[1]]
    image, result = run_program(
        root,
        runtime,
        memory,
        arena,
        lower_bf16_gather(maps, dst, zero, one, strategy=strategy),
        profile=profile_for(memory),
    )
    if not np.array_equal(read(image, dst, 4096), expected):
        raise AssertionError("gather mismatch")
    return dict(case=root.name, **result, status="passed", checked_values=4096)


def compact_case(root, runtime, memory, kind):
    rng = np.random.default_rng(194 + len(kind))
    arena = Arena()
    zero = arena.add(np.zeros(2048))
    one = arena.add(np.eye(1, 2048).ravel())
    heads, width = (32, 64) if kind == "mamba" else (16, 128)
    spec = replace(NEMOTRON_MAMBA if kind == "mamba" else KIMI_KDA, heads=heads)
    fields = ("a", "b", "c", "dt", "d") if kind == "mamba" else ("decay", "key", "query", "beta")
    maps = {}
    expanded = {}
    for name in fields:
        rows = 128 if name in ("b", "c", "decay", "key", "query") else 1
        values = bf(rng.uniform(0.001, 0.02, (rows, heads)))
        # Compact row-major capture; each cached row owns its padded tail.
        flat = np.pad(values.ravel(), (0, (-values.size) % 2048))
        base = arena.add(flat)
        for r in range(rows):
            maps[name, r] = [
                (base + (r * heads + h) // 2048 * 4096, (r * heads + h) % 2048, h * width, width) for h in range(heads)
            ]
        expanded[name] = arena.add(np.repeat(values, width, axis=1))
    masks = {}
    for h in range(heads):
        mask = np.zeros(2048)
        mask[h * width : (h + 1) * width] = 1
        masks[h * width, width] = arena.add(mask)
    loader = CompactCoefficientLoader(maps, masks, one)
    state = bf(rng.normal(0, 0.02, (128, 2048)))
    states = [arena.add(state.copy(), output=True) for _ in range(2)]
    value = arena.add(bf(rng.normal(0, 0.2, 2048)))
    outputs = [arena.add(np.full(2048, 7), output=True) for _ in range(2)]
    text = []
    for index in range(2):
        f = dict(expanded, zero=zero, output=outputs[index], **{("x" if kind == "mamba" else "value"): value})
        text.append(
            lower_prepared_vector_recurrence(
                spec,
                (PreparedVectorGroup(states[index], f),),
                static_address_reuse=True,
                pairwise_bf16_dot=True,
                mamba_decay_row_invariant=kind == "mamba",
                decay_is_delta=True,
                coefficient_loaders=(loader,) if index else None,
            )
        )
    image, result = run_program(root, runtime, memory, arena, "".join(text), profile=profile_for(memory))
    for addresses, count in ((states, 128 * 2048), (outputs, 2048)):
        if not np.array_equal(read(image, addresses[0], count), read(image, addresses[1], count)):
            raise AssertionError("compact baseline differs from ordinary expanded baseline")
    return dict(
        case=root.name,
        **result,
        status="passed",
        checked_values=128 * 2048 + 2048,
        scope="identical ordinary BF16 arithmetic, HBM expansion versus finite Vector software broadcast",
    )


def softmax_case(root, runtime, memory, values, max_latency):
    rng = np.random.default_rng(88 + values)
    a = Arena()
    rows = (values + 2047) // 2048
    x = np.full(rows * 2048, -16384, np.float32)
    x[:values] = bf(rng.normal(0, 0.6, values))
    source = a.add(x)
    tmp = a.add(np.full_like(x, 7), output=True)
    out = a.add(np.full_like(x, 7), output=True)
    mask = np.ones(2048, np.float32)
    if values % 2048:
        mask[values % 2048 :] = 0
    mb = a.add(mask)
    profile = profile_for(memory)
    profile = replace(profile, machine=replace(profile.machine, vector_max_cycles=max_latency))
    image, result = run_program(
        root,
        runtime,
        memory,
        a,
        lower_softmax_rows(source, tmp, out, values, mb),
        profile=profile,
        fp_constants=[-16384] + [0] * 31,
    )
    ex = exp(bf(x - np.max(x)))
    ex[values:] = 0
    total = np.float32(0)
    for r in ex.reshape(rows, 2048):
        while len(r) > 1:
            r = bf(r[::2] + r[1::2])
        total = bf(total + r[0])[0]
    expected = bf(ex * bf(1 / total))
    actual = read(image, out, len(x))
    if not np.array_equal(actual, expected):
        raise AssertionError(f"softmax rounding mismatch: {np.max(np.abs(actual - expected))}")
    return dict(
        case=root.name,
        **result,
        status="passed",
        checked_values=len(x),
        scope="stable BF16 softmax, finite SFU, SRAM/HBM spills, masked tail",
    )


def vector_case(root, runtime, memory):
    from compiler.aten.plena.recurrent_coefficients import lower_dot_rows, lower_positive_normalize
    from compiler.aten.plena.prepared_vector_recurrence import _Emitter

    rng = np.random.default_rng(980)
    a = Arena()
    x, w = bf(rng.normal(0, 0.2, 4096)), bf(rng.normal(0, 0.2, 4096))
    x[2307:] = 0
    w[2307:] = 0
    source, weights = a.add(x), a.add(w)
    onehot = a.add(np.eye(1, 2048).ravel())
    dot = a.add(np.full(2048, 7), output=True)
    relu = a.add(np.full(4096, 7), output=True)
    scores = np.zeros(2048, np.float32)
    scores[:6] = bf(rng.uniform(0.05, 0.9, 6))
    sb, norm = a.add(scores), a.add(np.full(2048, 7), output=True)
    text = lower_dot_rows(source, weights, dot, onehot, 2307)
    text += lower_positive_normalize(sb, norm, 6)
    e = _Emitter(2048, True)
    for r in range(2):
        e.transfer(0, source + r * 4096)
        e.address(1, 0)
        e.address(2, 0)
        e.lines.append("V_MAX_VF gp1, gp2, f0, 0")
        e.binary("MUL", 0, 0, 0)
        e.transfer(0, relu + r * 4096, store=True)
    text += "\n".join(e.lines) + "\n"
    image, result = run_program(root, runtime, memory, a, text, profile=profile_for(memory))

    def tree(v):
        while len(v) > 1:
            v = bf(v[::2] + v[1::2])
        return v[0]

    total = np.float32(0)
    for r in bf(x * w).reshape(2, 2048):
        total = bf(total + tree(r))[0]
    expected = np.zeros(2048, np.float32)
    expected[0] = total
    np.testing.assert_array_equal(read(image, dot, 2048), expected)
    np.testing.assert_array_equal(read(image, norm, 2048), bf(scores * bf(1 / tree(scores))))
    np.testing.assert_array_equal(read(image, relu, 4096), bf(np.maximum(x, 0) ** 2))
    return dict(
        case=root.name,
        **result,
        status="passed",
        checked_values=8192,
        scope="BF16 dot tail, positive router-score normalization, relu2",
    )


def attention_core_case(root, runtime, memory):
    """Connect QK -> softmax -> PV without host writes between operators.

    Queries are already scaled. Two query heads share one static KV group;
    projection/RoPE and incremental KV packing are outside this core check.
    Extra owned padding covers full Vector DMA reads of the final K tile.
    """
    rng = np.random.default_rng(739)
    a = Arena()
    profile = profile_for(memory)
    b, keys, width = 2, 4096, 128
    q = bf(rng.normal(0, 0.1, (b, width)))
    key = bf(rng.normal(0, 0.2, (width, keys)))
    value = bf(rng.normal(0, 0.2, (keys, width)))
    zero = a.add(np.zeros(2048))
    mask = a.add(np.ones(2048))
    qk = Projection(0, 0, 0, zero, width, keys, 256)
    pv = Projection(0, 0, 0, zero, keys, width, 256)

    def weights(w):
        return a.add(w.reshape(w.shape[0], -1, 32).transpose(1, 0, 2).copy())

    queries = [a.add(np.pad(row, (0, qk.input_values - width))) for row in q]
    key_base, value_base = weights(key), weights(value)
    logits = [a.add(np.full(qk.output_values, 7), output=True) for _ in q]
    scratch = [a.add(np.full(qk.output_values, 7), output=True) for _ in q]
    probabilities = [a.add(np.zeros(pv.input_values), output=True) for _ in q]
    outputs = [a.add(np.full(pv.output_values, 7), output=True) for _ in q]
    qk = replace(qk, inputs=queries[0], weights=key_base, outputs=logits[0])
    pv = replace(pv, inputs=probabilities[0], weights=value_base, outputs=outputs[0])
    text = lower_batch_projection(qk, queries, logits)
    for source, temporary, destination in zip(logits, scratch, probabilities):
        text += lower_softmax_rows(source, temporary, destination, keys, mask)
    text += lower_batch_projection(pv, probabilities, outputs)
    image, result = run_program(
        root, runtime, memory, a, text, profile=profile, fp_constants=[-16384] + [0] * 31
    )
    for query, lb, pb, ob in zip(q, logits, probabilities, outputs):
        scores = reference(query, key, 256, profile.matrix)
        exponentials = exp(bf(scores - np.max(scores)))
        total = np.float32(0)
        for row in exponentials.reshape(-1, 2048):
            while len(row) > 1:
                row = bf(row[::2] + row[1::2])
            total = bf(total + row[0])[0]
        probability = bf(exponentials * bf(1 / total))
        expected = reference(probability, value, 256, profile.matrix)
        np.testing.assert_array_equal(read(image, lb, keys), scores)
        np.testing.assert_array_equal(read(image, pb, keys), probability)
        np.testing.assert_array_equal(read(image, ob, width), expected)
    return dict(
        case=root.name, **result, status="passed", checked_values=b * (keys * 2 + width),
        scope="connected prepared-Q/static-KV attention core; no host intermediate injection; no KV append/router/MLA claim",
    )


def main():
    p = argparse.ArgumentParser()
    for arg in ("output", "runtime", "memory-root"):
        p.add_argument("--" + arg, required=True, type=Path)
    p.add_argument("--only", default="all", choices=("all", "auxiliary", "attention"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    jobs = [
        ("matrix_b2_tail", lambda d: matrix_case(d, args.runtime, args.memory_root, 2, 289, 65)),
        ("matrix_b4_tail", lambda d: matrix_case(d, args.runtime, args.memory_root, 4, 97, 33)),
        ("gather_grouped", lambda d: gather_case(d, args.runtime, args.memory_root)),
        ("gather_pattern", lambda d: gather_case(d, args.runtime, args.memory_root, "pattern")),
        *[
            ("compact_" + kind, lambda d, k=kind: compact_case(d, args.runtime, args.memory_root, k))
            for kind in ("mamba", "kda")
        ],
    ]
    auxiliary = [
        ("vector_auxiliary", lambda d: vector_case(d, args.runtime, args.memory_root)),
        ("softmax_4096", lambda d: softmax_case(d, args.runtime, args.memory_root, 4096, 4)),
        ("softmax_tail_latency8", lambda d: softmax_case(d, args.runtime, args.memory_root, 2307, 8)),
    ]
    attention = [("attention_core_4096", lambda d: attention_core_case(d, args.runtime, args.memory_root))]
    jobs = auxiliary if args.only == "auxiliary" else attention if args.only == "attention" else jobs + auxiliary + attention
    for name, run in jobs:
        result = run(args.output / name)
        results.append(result)
        (args.output / "validation.json").write_text(
            json.dumps(dict(status="running", results=results), indent=2) + "\n"
        )
        print(name, result["prediction"]["total"], "passed", flush=True)
    (args.output / "validation.json").write_text(
        json.dumps(dict(status="passed", results=results, runtime_sha256=digest(args.runtime)), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
