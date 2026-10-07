"""Actual rectangular Matrix instructions: tails, K accumulation, DMA and stores.

Synthetic deterministic inputs validate the executable projection ABI. They do
not certify checkpoint weights, compressed weight decoding or a complete layer.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
from transactional_emulator.testbench.aten.recurrent_gate_test import (
    Arena,
    AssemblyToBinary,
    COMPILER_ROOT,
    MatrixSramPoint,
    _write_settings,
    bf,
    digest,
    metric,
)
from compiler.aten.plena.isa_matrix_projection import Projection, lower_b1_projection
from analytic_models.performance.ltile_cost import assembly_cost
from analytic_models.performance.ltile_dma import DmaBackend
from analytic_models.performance.matrix_service import MatrixService


def reference(x, w, k_tile, h):
    """Independent array-vector reference with specified local/tree rounding."""
    result = np.zeros(w.shape[1], np.float32)
    rounded = bf if h.accumulator == "BF16" else lambda x: np.asarray(x, dtype=np.float32)
    for k0 in range(0, len(x), k_tile):
        count = min(k_tile, len(x) - k0)
        products = np.zeros((h.reduction_lanes, w.shape[1]), np.float32)
        products[:count] = x[k0 : k0 + count, None] * w[k0 : k0 + count]
        groups = products.reshape(h.groups, h.edge, -1)
        partial = np.zeros((h.groups, w.shape[1]), np.float32)
        for k in range(h.edge):
            partial = rounded(partial + groups[:, k])
        while len(partial) > 1:
            partial = rounded(partial[::2] + partial[1::2])
        result = rounded(result + partial[0])
    return bf(result)


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    x = bf(rng.normal(0, 0.2, args.k))
    w = bf(rng.normal(0, 0.1, (args.k, args.n)))
    h = MatrixService(accumulator=args.accumulator)
    arena = Arena()
    template = Projection(0, 0, 0, 0, args.k, args.n, args.k_tile)
    inputs = arena.add(np.pad(x, (0, template.input_values - args.k)))
    weight_values = []
    for col, k0, rows, _ in template.packets():
        packet = np.zeros((rows, 32), np.float32)
        nr, nc = min(args.k - k0, rows), min(args.n - col, 32)
        packet[:nr, :nc] = w[k0 : k0 + nr, col : col + nc]
        weight_values.append(packet.ravel())
    weights = arena.add(np.concatenate(weight_values))
    outputs = arena.add(np.full(template.output_values, 7), output=True)
    zero = arena.add(np.zeros(2048, np.float32))
    p = Projection(inputs, weights, outputs, zero, args.k, args.n, args.k_tile)
    assembly = lower_b1_projection(p)
    profile = out / "matrix_profile.json"
    profile.write_text(json.dumps(asdict(h), indent=2))
    cost = assembly_cost(assembly, trace_memory=True, matrix_service=h)
    memory = DmaBackend(
        args.memory_root / "ltile_memory", args.memory_root / "ramulator.json", args.memory_root / "cache"
    ).price(cost, "bounded:32:32")
    predicted = cost.components()
    (out / "prediction.json").write_text(json.dumps(predicted, indent=2))
    asm = out / "generated_asm_code.asm"
    asm.write_text(assembly)
    binary = out / "generated_machine_code.mem"
    AssemblyToBinary(
        str(COMPILER_ROOT / "doc/operation.svh"), str(COMPILER_ROOT / "doc/configuration.svh")
    ).generate_binary(str(asm), str(binary))
    (out / "hbm_for_behave_sim.bin").write_bytes(arena.data)
    for name in ("fp_sram.bin", "int_sram.bin"):
        (out / name).write_bytes(bytes(64))
    _write_settings(out, MatrixSramPoint())
    env = dict(
        os.environ,
        PLENA_UNIFIED_SERIAL_TIMING="1",
        PLENA_DMA_READ_WINDOW="32",
        PLENA_DMA_WRITE_WINDOW="32",
        PLENA_EXACT_VIEW_DMA="1",
        PLENA_MATRIX_SERVICE_PROFILE=str(profile),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        RUST_LOG="warn,transactional_emulator=info",
    )
    command = [
        str(args.runtime.resolve()),
        "--opcode",
        str(binary),
        "--hbm",
        str(out / "hbm_for_behave_sim.bin"),
        "--fpsram",
        str(out / "fp_sram.bin"),
        "--intsram",
        str(out / "int_sram.bin"),
        "--settings",
        str(out / "plena_settings.toml"),
        "--hbm-size",
        str(len(arena.data)),
        "--hbm-dump",
        str(out / "hbm_dump.bin"),
    ]
    with (out / "run.log").open("w") as log:
        subprocess.run(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
    image = (out / "hbm_dump.bin").read_bytes()
    arena.verify_guards(image)
    actual = (np.frombuffer(image, "<u2", p.output_values, p.outputs).astype(np.uint32) << 16).view(np.float32)
    expected = np.pad(reference(x, w, p.k_tile, h), (0, p.output_values - p.n))
    if not np.array_equal(actual.view(np.uint32), expected.view(np.uint32)):
        idx = np.flatnonzero(actual != expected)
        raise AssertionError(
            f"projection mismatch {len(idx)}: {idx[:8]}, actual={actual[idx[:8]]}, expected={expected[idx[:8]]}"
        )
    t = json.loads((out / "execution_timing.json").read_text())
    c = t["counters"]
    observed = dict(
        issue=c["issue_cycles"],
        scalar=t["scalar_and_control_cycles"],
        sram=c["bank_service_cycles"],
        arithmetic=c["arithmetic_cycles"],
        dependency=c["dependency_cycles"],
        dma=t["dma_and_memory_wait_picos"] / t["period_picos"],
        total=t["total_picos"] / t["period_picos"],
    )
    errors = {key: abs(predicted[key] - v) for key, v in observed.items()}
    if any(errors.values()):
        raise AssertionError(f"projection cycle calibration failed: {errors}")
    result = dict(
        status="passed_matrix_isa_candidate",
        shape=[1, args.k, args.n],
        projection=asdict(p),
        profile=asdict(h),
        seed=args.seed,
        checked_values=p.output_values,
        exact=True,
        versus_fp32=metric(actual[: p.n], x.astype(np.float64) @ w.astype(np.float64)),
        prediction=predicted,
        observed=observed,
        errors=errors,
        memory=memory,
        bytes=dict(
            hbm_read=sum(n * c for (d, n), c in cost.transfers.items() if d == "read"),
            hbm_write=sum(n * c for (d, n), c in cost.transfers.items() if d == "write"),
        ),
        source_sha256={
            str(f): digest(f)
            for f in [args.runtime, binary, profile, COMPILER_ROOT / "aten/plena/isa_matrix_projection.py"]
        },
        reproduce=shlex.join(
            [sys.executable, "-m", "transactional_emulator.testbench.aten.matrix_projection_test", *sys.argv[1:]]
        ),
        exclusions=["NVFP4 codec", "checkpoint weights", "complete layer", "RTL timing proof"],
        resource_note="Serial reads; repeated words charged; existing BLEN accumulator; candidate mini-array/tree and operand latches",
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"shape": result["shape"], "cycles": observed["total"], "exact": True}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=257)
    p.add_argument("--n", type=int, default=65)
    p.add_argument("--k-tile", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--accumulator", choices=["BF16", "FP32"], default="BF16")
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--memory-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
