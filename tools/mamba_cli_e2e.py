#!/usr/bin/env python3
"""Compiler -> CLI simulator differential test for one complete Mamba mixer."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


HBM_SIZE = 64 * 1024


def values(length: int, scale: float, offset: int) -> torch.Tensor:
    return torch.tensor(
        [(((index * 17 + offset) % 23) - 11) * scale for index in range(length)],
        dtype=torch.float32,
    )


def install_compiler_imports(compiler_root: Path) -> None:
    sys.path.insert(0, str(compiler_root))


def apply_image(hbm: bytearray, image: object) -> None:
    start = image.region.address
    stop = start + len(image.data)
    if stop > len(hbm):
        raise AssertionError(f"{image.region.name} exceeds the test HBM image")
    hbm[start:stop] = image.data


def assemble_program(
    compiler_root: Path,
    assembly: str,
    directory: Path,
    stem: str,
) -> Path:
    from assembler.assembly_to_binary import AssemblyToBinary

    asm_path = directory / f"{stem}.asm"
    opcode_path = directory / f"{stem}.mem"
    asm_path.write_text(assembly, encoding="ascii")
    assembler = AssemblyToBinary(
        str(compiler_root / "doc/operation.svh"),
        str(compiler_root / "doc/configuration.svh"),
    )
    assembler.generate_binary(str(asm_path), str(opcode_path))
    return opcode_path


def run_emulator(
    simulator_root: Path,
    emulator: Path,
    opcode: Path,
    hbm: bytearray,
    directory: Path,
    stem: str,
    timing_config: Path | None = None,
) -> bytearray:
    hbm_path = directory / f"{stem}.hbm.bin"
    dump_path = directory / f"{stem}.hbm.out.bin"
    fpsram_path = directory / "fpsram.bin"
    hbm_path.write_bytes(hbm)
    fpsram_path.write_bytes(b"")
    command = [
            str(emulator),
            "--opcode",
            str(opcode),
            "--hbm",
            str(hbm_path),
            "--fpsram",
            str(fpsram_path),
            "--hbm-size",
            str(HBM_SIZE),
            "--hbm-dump",
            str(dump_path),
            "--settings",
            str(simulator_root / "plena_settings.toml"),
            "--log-level",
            "warn",
        ]
    if timing_config is not None:
        command.extend(
            [
                "--mamba-timing-config",
                str(timing_config),
                "--mamba-profile-out",
                str(directory / f"{stem}.mamba-profile.json"),
            ]
        )
    result = subprocess.run(
        command,
        cwd=directory,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise AssertionError(
            f"emulator failed ({result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    data = dump_path.read_bytes()
    if len(data) != HBM_SIZE:
        raise AssertionError(f"HBM dump has {len(data)} bytes, expected {HBM_SIZE}")
    return bytearray(data)


def decode_storage(data: bytes, policy: object) -> torch.Tensor:
    from aten.models.nemotron3_mamba2.reference import PrecisionPolicy

    if policy == PrecisionPolicy.FP32_REFERENCE:
        return torch.frombuffer(bytearray(data), dtype=torch.float32).clone()
    if policy == PrecisionPolicy.BF16_ACTIVATION_FP32_STATE:
        return (
            torch.frombuffer(bytearray(data), dtype=torch.uint16)
            .view(torch.bfloat16)
            .float()
            .clone()
        )
    raise AssertionError(f"unsupported test policy {policy!r}")


def read_output(hbm: bytearray, bindings: object, batch: int, sequence: int) -> torch.Tensor:
    config = bindings.config
    element_bytes = 4 if int(bindings.precision_policy) == 0 else 2
    row_bytes = config.d_model * element_bytes
    output = torch.empty(batch, sequence, config.d_model, dtype=torch.float32)
    for batch_index in range(batch):
        for token in range(sequence):
            start = (
                bindings.output.address
                + batch_index * bindings.output_batch_stride
                + token * bindings.output_token_stride
            )
            output[batch_index, token] = decode_storage(
                hbm[start : start + row_bytes], bindings.precision_policy
            )
    return output


def read_state(hbm: bytearray, state: object, config: object, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    from aten.models.nemotron3_mamba2.reference import PrecisionPolicy

    ssm_elements = batch * config.num_heads * config.head_dim * config.state_dim
    conv_elements = batch * config.conv_channels * config.conv_kernel
    ssm = decode_storage(
        hbm[state.ssm.address : state.ssm.address + ssm_elements * 4],
        PrecisionPolicy.FP32_REFERENCE,
    ).reshape(batch, config.num_heads, config.head_dim, config.state_dim)
    conv = decode_storage(
        hbm[state.conv.address : state.conv.address + conv_elements * 4],
        PrecisionPolicy.FP32_REFERENCE,
    ).reshape(batch, config.conv_channels, config.conv_kernel)
    return ssm, conv


def read_completion(hbm: bytearray, program: object) -> tuple[int, int, int]:
    if program.completion is None:
        raise AssertionError("test command did not allocate a completion record")
    address = program.completion.region.address
    return struct.unpack_from("<IIQ", hbm, address)


def assert_bf16_ulps(actual: torch.Tensor, expected: torch.Tensor, max_ulps: int) -> None:
    actual_bits = actual.to(torch.bfloat16).view(torch.uint16).to(torch.int32)
    expected_bits = expected.to(torch.bfloat16).view(torch.uint16).to(torch.int32)
    if not torch.equal(actual_bits >> 15, expected_bits >> 15):
        raise AssertionError("BF16 result has a sign mismatch")
    distance = (actual_bits - expected_bits).abs()
    if int(distance.max()) > max_ulps:
        index = int(distance.argmax())
        raise AssertionError(
            f"BF16 result exceeds {max_ulps} ULPs at flat index {index}: "
            f"actual={actual.flatten()[index].item()}, "
            f"expected={expected.flatten()[index].item()}, "
            f"distance={distance.flatten()[index].item()}"
        )


def assert_result(
    actual: torch.Tensor,
    expected: torch.Tensor,
    policy: object,
    *,
    state: bool = False,
) -> None:
    from aten.models.nemotron3_mamba2.reference import PrecisionPolicy

    if policy == PrecisionPolicy.BF16_ACTIVATION_FP32_STATE and not state:
        assert_bf16_ulps(actual, expected, 2)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def run_policy(
    compiler_root: Path,
    simulator_root: Path,
    emulator: Path,
    policy: object,
) -> None:
    from assembler.generated_contract import (
        CONTRACT_SHA256,
        MAMBA_COMPLETION_STATUS,
        MAMBA_DESCRIPTOR_FIELDS,
    )
    from aten.models.nemotron3_mamba2.lowering import (
        MambaCommandCompiler,
        MambaCommandSpec,
        MambaSubop,
        allocate_mamba_tensor_bindings,
        materialize_mamba_io_images,
        materialize_mamba_weight_images,
    )
    from aten.models.nemotron3_mamba2.memory import (
        ByteAddressArena,
        MambaPersistentStateAllocator,
    )
    from aten.models.nemotron3_mamba2.reference import (
        Mamba2Config,
        Mamba2Weights,
        mamba2_prefill,
        mamba2_step,
    )

    config = Mamba2Config(
        d_model=4,
        d_inner=4,
        num_heads=2,
        head_dim=2,
        state_dim=2,
        groups=1,
        chunk_size=2,
        conv_kernel=2,
    )
    weights = Mamba2Weights(
        in_proj_weight=values(14 * 4, 0.015, 3).reshape(14, 4),
        in_proj_bias=values(14, 0.01, 5),
        conv_weight=values(8 * 2, 0.025, 7).reshape(8, 2),
        conv_bias=values(8, 0.01, 11),
        a_log=values(2, 0.02, 13),
        dt_bias=values(2, 0.03, 17),
        d_skip=values(2, 0.04, 19),
        norm_weight=torch.tensor([0.8, 0.9, 1.0, 1.1]),
        out_proj_weight=values(4 * 4, 0.02, 2).reshape(4, 4),
        out_proj_bias=values(4, 0.01, 4),
    )
    prefill_input = values(3 * 4, 0.05, 1).reshape(1, 3, 4)
    step_inputs = [
        values(4, 0.04, 9).reshape(1, 1, 4),
        values(4, 0.035, 14).reshape(1, 1, 4),
        values(4, 0.045, 20).reshape(1, 1, 4),
    ]
    golden_prefill = mamba2_prefill(
        prefill_input, weights, config, policy=policy, scan="sequential"
    )
    golden_steps = []
    golden_state = golden_prefill.state
    for step_input in step_inputs:
        golden_step = mamba2_step(
            step_input, weights, config, golden_state, policy=policy
        )
        golden_steps.append(golden_step)
        golden_state = golden_step.state

    arena = ByteAddressArena(0x1000, HBM_SIZE - 0x1000)
    state_allocator = MambaPersistentStateAllocator(arena)
    bindings = allocate_mamba_tensor_bindings(
        arena,
        config,
        batch_capacity=1,
        sequence_capacity=3,
        precision_policy=policy,
        prefix=f"e2e.{policy.name.lower()}",
        include_in_proj_bias=True,
        include_conv_bias=True,
        include_out_proj_bias=True,
    )
    compiler = MambaCommandCompiler(arena, state_allocator)
    prefill_program = compiler.compile(
        config,
        bindings,
        MambaCommandSpec(
            subop=MambaSubop.PREFILL,
            context_id=7,
            layer_id=3,
            batch_size=1,
            sequence_length=3,
            precision_policy=policy,
            completion_event=11,
        ),
    )
    step_programs = [
        compiler.compile(
            config,
            bindings,
            MambaCommandSpec(
                subop=MambaSubop.STEP,
                context_id=7,
                layer_id=3,
                batch_size=1,
                sequence_length=1,
                precision_policy=policy,
                continue_state=True,
                completion_event=12 + index,
            ),
        )
        for index in range(len(step_inputs))
    ]
    if any(prefill_program.state != program.state for program in step_programs):
        raise AssertionError("compiler did not reuse persistent state across commands")

    hbm = bytearray(HBM_SIZE)
    for image in materialize_mamba_weight_images(bindings, weights):
        apply_image(hbm, image)
    for image in materialize_mamba_io_images(bindings, prefill_input):
        apply_image(hbm, image)
    apply_image(hbm, prefill_program.descriptor)
    apply_image(hbm, prefill_program.completion)

    with tempfile.TemporaryDirectory(prefix=f"plena-mamba-{policy.name.lower()}-") as tmp:
        directory = Path(tmp)
        prefill_opcode = assemble_program(
            compiler_root, prefill_program.assembly, directory, "prefill"
        )

        invalid_hbm = bytearray(hbm)
        descriptor_base = prefill_program.descriptor.region.address
        state_stride_offset = MAMBA_DESCRIPTOR_FIELDS["state_head_stride"][0]
        struct.pack_into("<I", invalid_hbm, descriptor_base + state_stride_offset, 4)
        invalid_result = run_emulator(
            simulator_root,
            emulator,
            prefill_opcode,
            invalid_hbm,
            directory,
            "invalid-prefill",
        )
        status, event, elapsed = read_completion(invalid_result, prefill_program)
        if (status, event, elapsed) != (
            MAMBA_COMPLETION_STATUS["INVALID_DESCRIPTOR"],
            11,
            0,
        ):
            raise AssertionError(f"invalid error completion {(status, event, elapsed)}")
        if any(
            invalid_result[
                bindings.output.address : bindings.output.address
                + bindings.output.size_bytes
            ]
        ):
            raise AssertionError("invalid descriptor partially modified output")
        for region in (prefill_program.state.ssm, prefill_program.state.conv):
            if any(invalid_result[region.address : region.end_address]):
                raise AssertionError(
                    f"invalid descriptor partially modified {region.name}"
                )

        if int(policy) == 0:
            timing_config = simulator_root / "config/mamba_timing_design_point.toml"
            timed_result = run_emulator(
                simulator_root,
                emulator,
                prefill_opcode,
                hbm,
                directory,
                "timed-prefill",
                timing_config,
            )
            status, event, elapsed = read_completion(timed_result, prefill_program)
            if status != MAMBA_COMPLETION_STATUS["SUCCESS"] or event != 11 or elapsed == 0:
                raise AssertionError(f"invalid timed completion {(status, event, elapsed)}")
            profile = json.loads(
                (directory / "timed-prefill.mamba-profile.json").read_text(
                    encoding="utf-8"
                )
            )
            if profile["config"]["calibrated"] is not False:
                raise AssertionError("design-point timing profile must be marked uncalibrated")
            if profile["config"]["hardware_profile"] != "sim_reference_mlen64":
                raise AssertionError("timing profile omitted its hardware profile")
            if profile["contract_sha256"] != CONTRACT_SHA256:
                raise AssertionError("timing profile used a different architecture contract")
            summary = profile["summary"]
            if (
                summary["command_count"] != 1
                or summary["total_span_cycles"] <= 0
                or summary["max_queue_depth_after_issue"] != 1
            ):
                raise AssertionError(f"invalid timing summary {summary}")
            resources = {item["resource"]: item for item in summary["resources"]}
            if not {"hbm", "matrix", "conv", "exp", "state", "elementwise"}.issubset(
                resources
            ):
                raise AssertionError("timing summary is missing a modeled resource")
            if any(
                item["busy_cycles"] <= 0 or not 0.0 < item["utilization"] <= 1.0
                for item in resources.values()
            ):
                raise AssertionError(f"invalid resource utilization {resources}")
            command = profile["commands"][0]
            stage_names = {stage["name"] for stage in command["stages"]}
            required_stages = {
                "descriptor_dma",
                "tensor_state_read",
                "input_projection",
                "depthwise_conv",
                "dt_a_exp",
                "selective_state_scan",
                "gate_group_rmsnorm",
                "output_projection",
                "output_state_write",
                "completion_write",
            }
            if not required_stages.issubset(stage_names):
                raise AssertionError(f"timing profile is missing {required_stages - stage_names}")
            if command["external_scratch_bytes"] != 0:
                raise AssertionError("timing model exposed forbidden external scan scratch")

        hbm = run_emulator(
            simulator_root, emulator, prefill_opcode, hbm, directory, "prefill"
        )
        assert_result(read_output(hbm, bindings, 1, 3), golden_prefill.output, policy)
        actual_ssm, actual_conv = read_state(hbm, prefill_program.state, config, 1)
        assert_result(actual_ssm, golden_prefill.state.ssm, policy, state=True)
        assert_result(actual_conv, golden_prefill.state.conv, policy, state=True)
        status, event, elapsed = read_completion(hbm, prefill_program)
        if (status, event, elapsed) != (MAMBA_COMPLETION_STATUS["SUCCESS"], 11, 0):
            raise AssertionError(f"invalid prefill completion {(status, event, elapsed)}")

        for index, (step_input, step_program, golden_step) in enumerate(
            zip(step_inputs, step_programs, golden_steps, strict=True)
        ):
            for image in materialize_mamba_io_images(bindings, step_input):
                apply_image(hbm, image)
            apply_image(hbm, step_program.descriptor)
            apply_image(hbm, step_program.completion)
            stem = f"step-{index}"
            step_opcode = assemble_program(
                compiler_root, step_program.assembly, directory, stem
            )
            hbm = run_emulator(
                simulator_root, emulator, step_opcode, hbm, directory, stem
            )
            assert_result(
                read_output(hbm, bindings, 1, 1), golden_step.output, policy
            )
            actual_ssm, actual_conv = read_state(
                hbm, step_program.state, config, 1
            )
            assert_result(actual_ssm, golden_step.state.ssm, policy, state=True)
            assert_result(actual_conv, golden_step.state.conv, policy, state=True)
            status, event, elapsed = read_completion(hbm, step_program)
            expected_completion = (
                MAMBA_COMPLETION_STATUS["SUCCESS"],
                12 + index,
                0,
            )
            if (status, event, elapsed) != expected_completion:
                raise AssertionError(f"invalid step completion {(status, event, elapsed)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler-root", type=Path, required=True)
    parser.add_argument("--simulator-root", type=Path, required=True)
    parser.add_argument("--emulator", type=Path, required=True)
    args = parser.parse_args()
    compiler_root = args.compiler_root.resolve()
    simulator_root = args.simulator_root.resolve()
    emulator = args.emulator.resolve()
    install_compiler_imports(compiler_root)
    torch.set_num_threads(1)

    from aten.models.nemotron3_mamba2.reference import PrecisionPolicy

    for policy in (
        PrecisionPolicy.FP32_REFERENCE,
        PrecisionPolicy.BF16_ACTIVATION_FP32_STATE,
    ):
        run_policy(compiler_root, simulator_root, emulator, policy)
        print(f"PASS {policy.name}")


if __name__ == "__main__":
    main()
