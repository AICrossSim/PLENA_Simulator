"""Raw captured gates -> Compiler -> machine code -> Rust -> HBM.

This tests the ordinary-Vector producer, not a complete recurrent layer. Input
packing is performed by the fixture and explicitly excluded from its cycles.
The ideal prepared GPU fields are comparison data, never program operands.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import torch

from transactional_emulator.testbench.aten.matrix_lcompute_recurrence_test import (
    AssemblyToBinary,
    COMPILER_ROOT,
    MatrixSramPoint,
    _write_settings,
)
from compiler.aten.plena.recurrent_coefficients import (
    GATE_CONSTANTS,
    KdaGateRow,
    MambaGateRow,
    lower_kda_gate_rows,
    lower_mamba_gate_rows,
)
from analytic_models.performance.ltile_cost import Machine, assembly_cost
from analytic_models.performance.ltile_dma import DmaBackend


def bf(x):
    a = np.ascontiguousarray(x, dtype=np.float32)
    bits = a.view(np.uint32)
    return ((bits + np.uint32(32767) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def softplus(x):
    t = torch.from_numpy(np.ascontiguousarray(x))
    return bf((t.clamp_min(0) + t.abs().neg().exp().log1p()).numpy())


def exp(x):
    return bf(torch.from_numpy(np.ascontiguousarray(x)).clamp(-88, 88).exp().numpy())


def sigmoid(x):
    return exp(bf(-softplus(bf(-x))))


def delta(log):
    if not np.isfinite(log).all() or log.min() < -32768 or log.max() > 0:
        raise ValueError("rounded gate is outside rational candidate domain")
    a, b, c, one, two = (bf(x) for x in GATE_CONSTANTS[:5])
    t = bf(log * a)
    u = bf(t * bf(one + bf(t * bf(c + bf(t * b)))))
    p = bf(u * bf(torch.from_numpy(bf(one + u)).reciprocal().numpy()))
    for _ in range(4):
        p = bf(p * bf(two - p))
    return p


class Arena:
    def __init__(self):
        self.data = bytearray()
        self.outputs = []

    def add(self, values, output=False):
        payload = (bf(values).view(np.uint32) >> 16).astype("<u2").tobytes()
        self.data.extend(b"\xa5" * ((-len(self.data)) % 64 + 64))
        base = len(self.data)
        self.data.extend(payload)
        self.data.extend(b"\xa5" * ((-len(self.data)) % 64))
        if output:
            self.outputs.append((base, base + len(payload)))
        return base

    def verify_guards(self, image):
        restored = bytearray(image[: len(self.data)])
        for start, end in self.outputs:
            restored[start:end] = self.data[start:end]
        if restored != self.data:
            raise AssertionError("producer overwrote immutable inputs or guard bytes")


def metric(actual, expected):
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    norm = float(np.linalg.norm(expected))
    return dict(
        rel_l2=float(np.linalg.norm(diff) / max(norm, 1e-30)),
        abs_l2=float(np.linalg.norm(diff)),
        max_abs=float(np.max(np.abs(diff))),
        reference_norm=norm,
    )


def fixture(kind, capture, layer, tokens):
    path = capture / (f"model.layers.{layer}.mixer_inputs_0000.npz" if kind == "mamba" else "kda_inputs_0000.npz")
    data = np.load(path)
    if not 1 <= tokens <= len(data["delta"]):
        raise ValueError("token count exceeds capture shard")
    pins = {str(path): digest(path)}
    if kind == "kda":
        static = np.load(capture / "static.npz")
        flags = json.loads((capture / "flags.json").read_text())
        if flags["lower_bound"] != -5 or not flags["gate_in_kernel"] or not flags["beta_sigmoid_in_kernel"]:
            raise ValueError("KDA gate formula differs from this producer")
        for name in ("static.npz", "flags.json"):
            pins[str(capture / name)] = digest(capture / name)
    rows = []
    for token in range(tokens):
        if kind == "mamba":
            dt = data["source_dt"][token, 0]
            bias = data["source_dt_bias"][token]
            a = data["source_A"][token]
            if not (np.all(dt == dt[:, :1]) and np.all(bias == bias[:, :1]) and np.all(a == a[:, :1, :1])):
                raise ValueError("Mamba dt/A is not head-invariant")
            inputs = [dt[:, 0], bias[:, 0], a[:, 0, 0]]
            ideals = [data["dt"][token], data["delta"][token, :, 0]]
        else:
            inputs = [
                data["source_g"][token].reshape(-1),
                static["dt_bias"].reshape(-1),
                np.repeat(static["A_log"].reshape(-1), 128),
                np.repeat(data["source_beta"][token].reshape(-1), 128),
            ]
            ideals = [data["delta"][token].reshape(-1), np.repeat(data["beta"][token], 128)]
        for start in range(0, len(inputs[0]), 2048):
            count = min(2048, len(inputs[0]) - start)
            arrays = [bf(np.pad(x[start : start + count], (0, 2048 - count))) for x in inputs]
            if kind == "mamba":
                computed_dt = softplus(bf(arrays[0] + arrays[1]))
                expected = [computed_dt, delta(bf(computed_dt * arrays[2]))]
            else:
                log = bf(-5 * sigmoid(bf(exp(arrays[2]) * bf(arrays[0] + arrays[1]))))
                expected = [delta(log), sigmoid(arrays[3])]
            rows.append((arrays, expected, [x[start : start + count] for x in ideals], count))
    return rows, pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["mamba", "kda"], required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--layer", type=int, help="Nemotron layer only (default 46); KDA capture is first attention")
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--memory-root", type=Path, required=True)
    parser.add_argument("--sfu-lanes", type=int, default=32)
    parser.add_argument("--sfu-ii", type=int, default=1)
    args = parser.parse_args()
    if args.kind == "kda" and args.layer is not None:
        parser.error("--layer is only valid for Mamba; KDA capture is first attention only")
    if args.kind == "mamba" and args.layer is None:
        args.layer = 46
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, pins = fixture(args.kind, args.capture, args.layer, args.tokens)
    arena = Arena()
    constants = arena.add(np.repeat(np.asarray(GATE_CONSTANTS)[:, None], 2048, axis=1))
    requests, destinations = [], []
    for inputs, expected, _, _ in rows:
        input_addresses = [arena.add(x) for x in inputs]
        output_addresses = [arena.add(np.full(2048, 7), output=True) for _ in expected]
        destinations.append(output_addresses)
        row_type = MambaGateRow if args.kind == "mamba" else KdaGateRow
        requests.append(row_type(*input_addresses, *output_addresses))
    lowering = lower_mamba_gate_rows if args.kind == "mamba" else lower_kda_gate_rows
    assembly = lowering(requests, constants)
    machine = Machine(
        sfu_lanes=args.sfu_lanes,
        sfu_ii=args.sfu_ii,
        vector_exp_cycles=8,
        vector_softplus_cycles=16,
        vector_reciprocal_cycles=8,
    )
    cost = assembly_cost(assembly, machine, trace_memory=True)
    memory = DmaBackend(
        args.memory_root / "ltile_memory", args.memory_root / "ramulator.json", args.memory_root / "cache"
    ).price(cost, "bounded:32:32")
    prediction = cost.components()
    (args.output / "prediction.json").write_text(json.dumps(prediction, indent=2) + "\n")
    asm = args.output / "generated_asm_code.asm"
    asm.write_text(assembly)
    binary = args.output / "generated_machine_code.mem"
    AssemblyToBinary(
        str(COMPILER_ROOT / "doc/operation.svh"), str(COMPILER_ROOT / "doc/configuration.svh")
    ).generate_binary(str(asm), str(binary))
    (args.output / "hbm_for_behave_sim.bin").write_bytes(arena.data)
    for name in ("fp_sram.bin", "int_sram.bin"):
        (args.output / name).write_bytes(bytes(64))
    _write_settings(args.output, MatrixSramPoint())
    env = dict(
        os.environ,
        PLENA_UNIFIED_SERIAL_TIMING="1",
        PLENA_DMA_READ_WINDOW="32",
        PLENA_DMA_WRITE_WINDOW="32",
        PLENA_EXACT_VIEW_DMA="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        RUST_LOG="warn,transactional_emulator=info",
        PLENA_VECTOR_SFU_LANES=str(machine.sfu_lanes),
        PLENA_VECTOR_SFU_II=str(machine.sfu_ii),
        PLENA_VECTOR_SFU_EXP_LATENCY="8",
        PLENA_VECTOR_SFU_SOFTPLUS_LATENCY="16",
        PLENA_VECTOR_SFU_RECI_LATENCY="8",
    )
    command = [
        str(args.runtime),
        "--opcode",
        str(binary),
        "--hbm",
        str(args.output / "hbm_for_behave_sim.bin"),
        "--fpsram",
        str(args.output / "fp_sram.bin"),
        "--intsram",
        str(args.output / "int_sram.bin"),
        "--settings",
        str(args.output / "plena_settings.toml"),
        "--hbm-size",
        str(len(arena.data)),
        "--hbm-dump",
        str(args.output / "hbm_dump.bin"),
    ]
    with (args.output / "run.log").open("w") as log:
        subprocess.run(command, env=env, cwd=args.output, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    image = (args.output / "hbm_dump.bin").read_bytes()
    arena.verify_guards(image)
    actual_by_field, ideal_by_field = [[], []], [[], []]
    checked = 0
    for addresses, (_, expected, ideal, count) in zip(destinations, rows):
        for index, (address, reference) in enumerate(zip(addresses, expected)):
            bits = np.frombuffer(image, dtype="<u2", count=2048, offset=address)
            actual = (bits.astype(np.uint32) << 16).view(np.float32)
            if not np.array_equal(actual.view(np.uint32), reference.view(np.uint32)):
                mismatch = np.flatnonzero(actual.view(np.uint32) != reference.view(np.uint32))
                raise AssertionError(
                    f"ISA contract mismatch field={index}, count={len(mismatch)}, first={mismatch[:8]}"
                )
            checked += len(actual)
            actual_by_field[index].append(actual[:count])
            ideal_by_field[index].append(ideal[index])
    timing = json.loads((args.output / "execution_timing.json").read_text())
    c = timing["counters"]
    observed = dict(
        issue=c["issue_cycles"],
        scalar=timing["scalar_and_control_cycles"],
        sram=c["bank_service_cycles"],
        arithmetic=c["arithmetic_cycles"],
        dependency=c["dependency_cycles"],
        dma=timing["dma_and_memory_wait_picos"] / timing["period_picos"],
        total=timing["total_picos"] / timing["period_picos"],
    )
    errors = {key: abs(prediction[key] - value) for key, value in observed.items()}
    if any(errors.values()):
        raise AssertionError(f"cycle prediction failed: {errors}")
    fields = ["dt", "delta"] if args.kind == "mamba" else ["delta", "beta"]
    result = dict(
        status="passed_gate_candidate_only",
        kind=args.kind,
        source_layer=args.layer if args.kind == "mamba" else "first_attention_only",
        tokens=args.tokens,
        capture_sha256=pins,
        runtime_sha256=digest(args.runtime),
        compiler_sha256=digest(COMPILER_ROOT / "aten/plena/recurrent_coefficients.py"),
        program_sha256=digest(binary),
        settings_sha256=digest(args.output / "plena_settings.toml"),
        machine=asdict(machine),
        prediction=prediction,
        observed=observed,
        component_errors=errors,
        bitwise_checked_values=checked,
        guard_bytes_unchanged=True,
        memory=memory,
        vs_gpu_prepared={
            name: metric(np.concatenate(a), np.concatenate(b))
            for name, a, b in zip(fields, actual_by_field, ideal_by_field)
        },
        reproduce=shlex.join(
            [sys.executable, "-m", "transactional_emulator.testbench.aten.recurrent_gate_test", *sys.argv[1:]]
        ),
        exclusions=[
            "projection",
            "conv",
            "q/k normalization",
            "coefficient packing/broadcast",
            "recurrence connection",
            "task quality",
            "SFU RTL approximation/throughput proof",
        ],
        assumptions=[
            "SFU 8/16/8-cycle subchunk latencies are candidate budgets, not synthesis measurements",
            "libtorch transcendental functions shared by functional oracle and emulator",
            "finite 32-read/32-write DMA; analytical and Rust DMA share Ramulator",
        ],
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    for name in (
        "hbm_for_behave_sim.bin",
        "hbm_dump.bin",
        "mram_dump.bin",
        "vram_dump.bin",
        "fpsram_dump.bin",
        "intsram_dump.bin",
    ):
        (args.output / name).unlink(missing_ok=True)
    for path in (asm, binary):
        path.with_suffix(path.suffix + ".gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
        path.unlink()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
