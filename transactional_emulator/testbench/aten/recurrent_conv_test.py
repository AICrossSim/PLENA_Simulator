"""Real captured KDA projection -> four-tap conv -> SiLU via actual ISA."""

import argparse
import base64
import gzip
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
    sigmoid,
)
from compiler.aten.plena.recurrent_coefficients import GATE_CONSTANTS, ConvStep, lower_conv_steps
from analytic_models.performance.ltile_cost import Machine
from analytic_models.performance.ltile_dma import DmaBackend
from analytic_models.performance.ltile_platform import ExecutionProfile
from analytic_models.performance.ltile_execution import price_program


def read_image(path):
    """Completed captures may be losslessly compressed after verification."""
    if path.exists():
        return path.read_bytes()
    with gzip.open(str(path) + ".gz", "rb") as stream:
        return stream.read()


def run_program(
    output,
    runtime,
    memory_root,
    arena,
    assembly,
    machine=None,
    matrix_service=None,
    fp_constants=None,
    recheck_only=False,
    references=None,
    profile=None,
):
    """Run one connected program; no host writes after execution starts."""
    machine = (
        machine
        or (profile.machine if profile else None)
        or Machine(sfu_lanes=32, vector_exp_cycles=8, vector_softplus_cycles=16, vector_reciprocal_cycles=8)
    )
    if profile is None:
        profile = ExecutionProfile(machine=machine, **({"matrix": matrix_service} if matrix_service else {}))
    elif profile.machine != machine or (matrix_service is not None and profile.matrix != matrix_service):
        raise ValueError("runner arguments differ from the common execution profile")
    elif matrix_service is None:
        matrix_service = profile.matrix
    output = output.resolve()
    if recheck_only:
        # Re-evaluate reference/diagnostic metrics without pretending to rerun
        # machine code. Refuse reuse if any execution input has changed.
        result_path = output / "result.json"
        if not result_path.exists():
            result_path = output / "execution_result.json"
        result = json.loads(result_path.read_text())
        profile_path = output / "execution_profile.json"
        if profile_path.exists() and json.loads(profile_path.read_text()) != asdict(profile):
            raise ValueError("cannot reuse a different execution profile")
        if not profile_path.exists() and (
            profile.dma.service != "bounded:32:32" or profile.state_rounding != "rn" or profile.hbm_controllers != 8
        ):
            raise ValueError("legacy execution only certifies bounded:32:32 / RN")
        if (output / "generated_asm_code.asm").read_text() != assembly:
            raise ValueError("cannot reuse execution with a different program")
        if read_image(output / "hbm_for_behave_sim.bin") != arena.data:
            raise ValueError("cannot reuse execution with different initial HBM")
        if result["runtime_sha256"] != digest(runtime) or Machine(**result["machine"]) != machine:
            raise ValueError("cannot reuse a different runtime or service contract")
        if result["program_sha256"] != digest(output / "generated_machine_code.mem"):
            raise ValueError("cannot reuse modified machine code")
        if matrix_service is not None and json.loads((output / "matrix_profile.json").read_text()) != asdict(
            matrix_service
        ):
            raise ValueError("cannot reuse a different Matrix service contract")
        if (
            fp_constants is not None
            and (output / "fp_sram.bin").read_bytes()
            != (bf(fp_constants).view(np.uint32) >> 16).astype("<u2").tobytes()
        ):
            raise ValueError("cannot reuse different FP constants")
        image = read_image(output / "hbm_dump.bin")
        if len(image) != len(arena.data):
            raise ValueError("truncated or oversized HBM dump")
        arena.verify_guards(image)
        return image, result
    output.mkdir(parents=True, exist_ok=False)
    cost, priced = price_program(
        assembly,
        profile,
        DmaBackend(memory_root / "ltile_memory", memory_root / "ramulator.json", memory_root / "cache"),
        matrix=matrix_service is not None,
    )
    memory = priced["memory"]
    (output / "execution_profile.json").write_text(json.dumps(asdict(profile), indent=2) + "\n")
    predicted = cost.components()
    (output / "prediction.json").write_text(json.dumps(predicted, indent=2))
    asm = output / "generated_asm_code.asm"
    asm.write_text(assembly)
    binary = output / "generated_machine_code.mem"
    AssemblyToBinary(
        str(COMPILER_ROOT / "doc/operation.svh"), str(COMPILER_ROOT / "doc/configuration.svh")
    ).generate_binary(str(asm), str(binary))
    (output / "hbm_for_behave_sim.bin").write_bytes(arena.data)
    for name in ("fp_sram.bin", "int_sram.bin"):
        (output / name).write_bytes(bytes(64))
    if fp_constants is not None:
        (output / "fp_sram.bin").write_bytes((bf(fp_constants).view(np.uint32) >> 16).astype("<u2").tobytes())
    _write_settings(output, MatrixSramPoint())
    import re

    settings = output / "plena_settings.toml"
    content, count = re.subn(
        r"(\[TRANSACTIONAL\.LATENCY\.VECTOR_MAX_CYCLES\]\s*)dc_lib_en = \d+\s*dc_lib_dis = \d+",
        lambda m: m[1] + f"dc_lib_en = {machine.vector_max_cycles}\ndc_lib_dis = {machine.vector_max_cycles}",
        settings.read_text(),
    )
    if count != 1:
        raise ValueError("missing transactional max-reduction latency setting")
    settings.write_text(content)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        RUST_LOG="warn,transactional_emulator=info",
    )
    env.update(profile.runtime_environment())
    if matrix_service is not None:
        path = output / "matrix_profile.json"
        path.write_text(json.dumps(asdict(matrix_service), indent=2))
        env["PLENA_MATRIX_SERVICE_PROFILE"] = str(path)
    command = [
        str(runtime.resolve()),
        "--opcode",
        str(binary),
        "--hbm",
        str(output / "hbm_for_behave_sim.bin"),
        "--fpsram",
        str(output / "fp_sram.bin"),
        "--intsram",
        str(output / "int_sram.bin"),
        "--settings",
        str(output / "plena_settings.toml"),
        "--hbm-size",
        str(len(arena.data)),
        "--hbm-dump",
        str(output / "hbm_dump.bin"),
    ]
    if fp_constants is not None:
        command.append("--fpsram-bf16")
    manifest = {
        "runtime_sha256": digest(runtime),
        "program_sha256": digest(binary),
        "machine": asdict(machine),
        "command": command,
        "inputs_sha256": {
            name: digest(output / name)
            for name in ("hbm_for_behave_sim.bin", "fp_sram.bin", "int_sram.bin", "plena_settings.toml")
        },
    }
    (output / "execution_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if references is not None:
        (output / "reference_layout.json").write_text(
            json.dumps(
                [dict(name=name, address=address, values=len(value)) for name, address, value in references], indent=2
            )
            + "\n"
        )
        np.savez_compressed(output / "reference_values.npz", **{name: value for name, _, value in references})
    with (output / "run.log").open("w") as log:
        subprocess.run(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
    image = (output / "hbm_dump.bin").read_bytes()
    if len(image) != len(arena.data):
        raise ValueError("truncated or oversized HBM dump")
    arena.verify_guards(image)
    t = json.loads((output / "execution_timing.json").read_text())
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
        raise AssertionError(f"component timing mismatch: {errors}")
    result = dict(
        prediction=predicted,
        observed=observed,
        component_errors=errors,
        memory=memory,
        hbm_read_bytes=sum(n * c for (d, n), c in cost.transfers.items() if d == "read"),
        hbm_write_bytes=sum(n * c for (d, n), c in cost.transfers.items() if d == "write"),
        runtime_sha256=digest(runtime),
        program_sha256=digest(binary),
        machine=asdict(machine),
    )
    for name in ("hbm_read_bytes", "hbm_write_bytes"):
        if result[name] != t[name]:
            raise AssertionError(f"physical HBM accounting mismatch: {name}")
    result["traffic_source"] = "Rust physical HBM counters, checked against emitted DMA transfers"
    (output / "execution_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return image, result


def read(image, base, count):
    return (np.frombuffer(image, "<u2", count, base).astype(np.uint32) << 16).view(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "weights", "runtime", "memory-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--kind", choices=["q", "k", "v"], default="q")
    p.add_argument("--tokens", type=int, default=32)
    args = p.parse_args()
    archive = np.load(args.capture / "kda_inputs_0000.npz")
    raw_weights = json.loads(args.weights.read_text())
    key = "language_model.model.layers.0.self_attn." + args.kind + "_conv1d.weight"
    spec = raw_weights[key]
    if spec["dtype"] != "F32" or spec["shape"] != [12288, 1, 4]:
        raise ValueError("unexpected checkpoint conv format")
    weight = bf(np.frombuffer(base64.b64decode(spec["data"]), "<f4").reshape(12288, 4).T)
    stages = {"q": (1, 4), "k": (2, 5), "v": (3, 6)}
    projection, conv = stages[args.kind]
    inputs = bf(archive[f"stage_{projection}_out"][: args.tokens].reshape(args.tokens, -1))
    native = archive[f"stage_{conv}_out_0"][: args.tokens].reshape(args.tokens, -1)
    if not 1 <= args.tokens <= len(archive["delta"]):
        raise ValueError("missing continuous tokens")
    arena = Arena()
    constants = arena.add(np.repeat(np.asarray(GATE_CONSTANTS)[:, None], 2048, axis=1))
    weights = arena.add(weight)
    history = arena.add(np.zeros_like(weight), output=True)
    expected = []
    steps = []
    state = np.zeros_like(weight)
    for x in inputs:
        src = arena.add(x)
        dst = arena.add(np.full(len(x), 7), output=True)
        steps.append(ConvStep(src, history, weights, dst, len(x)))
        state = np.concatenate((state[1:], x[None]), axis=0)
        products = bf(state * weight)
        pre = bf(bf(products[0] + products[1]) + bf(products[2] + products[3]))
        expected.append(bf(pre * sigmoid(pre)))
    assembly = lower_conv_steps(steps, constants)
    image, result = run_program(args.output, args.runtime, args.memory_root, arena, assembly)
    actual = np.stack([read(image, s.output, 12288) for s in steps])
    if not np.array_equal(actual, np.stack(expected)):
        raise AssertionError("conv arithmetic differs")
    actual_state = read(image, history, 4 * 12288).reshape(4, 12288)
    if not np.array_equal(actual_state, state):
        raise AssertionError("carried conv state differs")
    result.update(
        status="passed_real_conv_candidate",
        kind=args.kind,
        tokens=args.tokens,
        channels=12288,
        exact=True,
        checked_values=actual.size + state.size,
        versus_native=metric(actual, native),
        capture_sha256=digest(args.capture / "kda_inputs_0000.npz"),
        weight_sha256=digest(args.weights),
        compiler_sha256=digest(COMPILER_ROOT / "aten/plena/recurrent_coefficients.py"),
        reproduce=shlex.join(
            [sys.executable, "-m", "transactional_emulator.testbench.aten.recurrent_conv_test", *sys.argv[1:]]
        ),
        exclusions=["projection producer connection", "q/k norm", "recurrence", "full layer", "long-chain acceptance"],
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"kind": args.kind, "cycles": result["observed"]["total"], "error": result["versus_native"]["rel_l2"]}
        )
    )


if __name__ == "__main__":
    main()
