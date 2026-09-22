"""Connected real conv -> L2 norm -> runtime coefficient packing validation."""

import argparse
import base64
import json
import shlex
import sys
from pathlib import Path
import numpy as np
from transactional_emulator.testbench.aten.recurrent_gate_test import Arena, bf, digest, metric, sigmoid, COMPILER_ROOT
from transactional_emulator.testbench.aten.recurrent_conv_test import run_program, read
from compiler.aten.plena.recurrent_coefficients import (
    GATE_CONSTANTS,
    ConvStep,
    L2NormRows,
    lower_conv_steps,
    lower_l2norm_rows,
    lower_bf16_gather,
)
from analytic_models.performance.ltile_cost import Machine


def norm_reference(x, scale):
    values = bf(x * x).reshape(-1, 128)
    while values.shape[1] > 1:
        values = bf(values[:, ::2] + values[:, 1::2])
    denom = bf(np.sqrt(bf(values[:, 0] + bf(1e-6))))
    inv = bf(bf(1 / denom) * bf(scale))
    return bf(x.reshape(-1, 128) * inv[:, None]).ravel()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("capture", "weights", "runtime", "memory-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--kind", choices=["q", "k"], default="q")
    p.add_argument("--gather-heads", type=int, default=2)
    a = p.parse_args()
    archive = np.load(a.capture / "kda_inputs_0000.npz")
    spec = json.loads(a.weights.read_text())["language_model.model.layers.0.self_attn." + a.kind + "_conv1d.weight"]
    weight = bf(np.frombuffer(base64.b64decode(spec["data"]), "<f4").reshape(12288, 4).T)
    stage = 1 if a.kind == "q" else 2
    x = bf(archive[f"stage_{stage}_out"][0].ravel())
    arena = Arena()
    constants = arena.add(np.repeat(np.asarray(GATE_CONSTANTS)[:, None], 2048, axis=1))
    weights = arena.add(weight)
    history = arena.add(np.zeros_like(weight), output=True)
    src = arena.add(x)
    conv = arena.add(np.full(12288, 7), output=True)
    norm = arena.add(np.full(12288, 7), output=True)
    zero = arena.add(np.zeros(2048))
    onehot = arena.add(np.eye(1, 2048).ravel())
    masks = np.zeros((16, 2048), np.float32)
    for h in range(16):
        masks[h, h * 128 : (h + 1) * 128] = 1
    maskbase = arena.add(masks)
    program = lower_conv_steps([ConvStep(src, history, weights, conv, 12288)], constants)
    program += lower_l2norm_rows(L2NormRows(conv, norm, 12288, maskbase, zero))
    if not 1 <= a.gather_heads <= 16:
        raise ValueError("head count outside test group")
    sources = [
        (norm + (h * 128 + r) // 2048 * 4096, (h * 128 + r) % 2048) for r in range(128) for h in range(a.gather_heads)
    ]
    destination = arena.add(np.full((len(sources) + 2047) // 2048 * 2048, 7), output=True)
    program += lower_bf16_gather(sources, destination, zero, onehot)
    machine = Machine(
        sfu_lanes=32,
        vector_exp_cycles=8,
        vector_softplus_cycles=16,
        vector_reciprocal_cycles=8,
        reduction_tree_bf16=True,
    )
    scale = 128**-0.5 if a.kind == "q" else 1.0
    image, result = run_program(
        a.output, a.runtime, a.memory_root, arena, program, machine, fp_constants=[1e-6, scale] + [0.0] * 30
    )
    pre = bf(x * weight[3])
    conv_ref = bf(pre * sigmoid(pre))
    norm_ref = norm_reference(conv_ref, scale)
    actual_conv = read(image, conv, 12288)
    actual_norm = read(image, norm, 12288)
    packed = np.zeros((len(sources) + 2047) // 2048 * 2048, np.float32)
    packed[: len(sources)] = norm_ref.reshape(96, 128)[: a.gather_heads].T.ravel()
    actual_gather = read(image, destination, len(packed))
    for name, actual, expected in [
        ("conv", actual_conv, conv_ref),
        ("norm", actual_norm, norm_ref),
        ("gather", actual_gather, packed),
    ]:
        if not np.array_equal(actual, expected):
            idx = np.flatnonzero(actual != expected)
            raise AssertionError(f"{name} mismatch {len(idx)} at {idx[:8]}: {actual[idx[:8]]} != {expected[idx[:8]]}")
    native = archive["q" if a.kind == "q" else "b"][0].ravel()
    result.update(
        status="passed_connected_conv_norm_gather",
        exact=True,
        kind=a.kind,
        tokens=1,
        heads=96,
        gather_heads=a.gather_heads,
        vs_native_prepared=metric(actual_norm, native),
        source_sha256={
            str(x): digest(x)
            for x in (
                a.weights,
                a.capture / "kda_inputs_0000.npz",
                COMPILER_ROOT / "aten/plena/recurrent_coefficients.py",
            )
        },
        reproduce=shlex.join(
            [sys.executable, "-m", "transactional_emulator.testbench.aten.recurrent_prepare_test", *sys.argv[1:]]
        ),
        exclusions=[
            "projection producer",
            "full coefficient set",
            "recurrence connection",
            "full layer",
            "task quality",
        ],
        assumptions=[
            "BF16 tree/scalar norm candidate; one-cycle scalar sqrt/reci inherited, not synthesized",
            "software one-hot gather is correctness reference, not optimized baseline",
        ],
    )
    (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "cycles": result["observed"]["total"],
                "rel_l2": result["vs_native_prepared"]["rel_l2"],
            }
        )
    )


if __name__ == "__main__":
    main()
