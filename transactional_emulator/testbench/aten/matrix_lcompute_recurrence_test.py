"""Compiler-to-Rust BF16 L-Compute recurrence verification at official shapes.

This is deliberately not a hand-written Opcode unit test.  The active Compiler
emits four complete token programs, its assembler produces the machine words,
and the Rust transactional emulator decodes and executes those words.  State
and every token result are read back from HBM and compared with an independent
BF16 reference.  No cache or private state memory is involved.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import tomlkit
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
COMPILER_ROOT = Path(
    os.environ.get("PLENA_COMPILER_ROOT", REPO_ROOT / "PLENA_Compiler")
).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(COMPILER_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPILER_ROOT))

from compiler.assembler.assembly_to_binary import AssemblyToBinary  # noqa: E402
from compiler.aten.plena.matrix_recurrence_lowering import (  # noqa: E402
    BF16_BYTES,
    KIMI_KDA,
    NEMOTRON_MAMBA,
    MatrixRecurrenceSpec,
    MatrixSramPoint,
    RecurrenceFieldManifest,
    RecurrenceFieldPacket,
    RecurrenceLayout,
    RecurrenceWorkingSet,
    build_recurrence_field_manifest,
    build_recurrence_working_set,
    lower_matrix_recurrence,
    validate_recurrence_output_stores,
)
from compiler.aten.plena.mview import validate_matrix_view_dominance  # noqa: E402
from transactional_emulator.testbench.emulator_runner import run_emulator  # noqa: E402


TOKENS = 4
SEED = 20260903


def _bf16(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().to(torch.bfloat16).float().contiguous()


def _bf16_bytes(value: torch.Tensor) -> bytes:
    bits = _bf16(value).to(torch.bfloat16).view(torch.uint16).numpy()
    return bits.astype("<u2", copy=False).tobytes()


def _read_bf16(image: bytes, offset: int, count: int) -> torch.Tensor:
    bits = np.frombuffer(image, dtype="<u2", count=count, offset=offset).copy()
    return torch.from_numpy(bits).view(torch.bfloat16).float()


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _write_packet(
    image: bytearray,
    packet: RecurrenceFieldPacket,
    values: torch.Tensor,
) -> None:
    flat = _bf16(values).flatten()
    if flat.numel() != packet.logical_values:
        raise AssertionError(
            f"{packet.key}: generated {flat.numel()} values, expected "
            f"{packet.logical_values}"
        )
    padded = torch.zeros(packet.transfer_values, dtype=torch.float32)
    padded[: flat.numel()] = flat
    payload = _bf16_bytes(padded)
    begin = packet.hbm_byte_offset
    end = begin + len(payload)
    image[begin:end] = payload


def _mamba_inputs(token: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(SEED + 101 * token)
    heads, rows, width = (
        NEMOTRON_MAMBA.heads,
        NEMOTRON_MAMBA.recurrence_rows,
        NEMOTRON_MAMBA.row_elements,
    )
    return {
        "x": _bf16(torch.randn(heads, width, generator=generator) * 0.08),
        "dt": _bf16(0.08 + torch.rand(heads, generator=generator) * 0.08),
        "a": _bf16(0.82 + torch.rand(heads, rows, generator=generator) * 0.12),
        "b": _bf16(torch.randn(heads, rows, generator=generator) * 0.025),
        "c": _bf16(torch.randn(heads, rows, generator=generator) * 0.03),
        "d": _bf16(0.15 + torch.rand(heads, generator=generator) * 0.15),
    }


def _mamba_reference(
    state: torch.Tensor,
    operands: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    x, dt = operands["x"], operands["dt"]
    scratch = _bf16(dt[:, None] * x)
    state = _bf16(
        operands["a"][:, :, None] * state
        + operands["b"][:, :, None] * scratch[:, None, :]
    )
    accumulator = torch.zeros_like(x)
    for row in range(NEMOTRON_MAMBA.recurrence_rows):
        accumulator += state[:, row, :] * operands["c"][:, row, None]
    output = _bf16(accumulator)
    output = _bf16(output + operands["d"][:, None] * x)
    return output, state


def _mamba_packet_values(
    packet: RecurrenceFieldPacket,
    operands: dict[str, torch.Tensor],
    working_set: RecurrenceWorkingSet,
) -> torch.Tensor:
    group_heads = working_set.group_heads
    first = packet.group * group_heads
    last = first + group_heads
    chunk = 0 if packet.chunk is None else packet.chunk
    if packet.field in {"x", "value"}:
        return operands["x"][first:last]
    if packet.field in {"scratch_zero", "output_zero", "output_result"}:
        return torch.zeros(packet.logical_values)
    if packet.field == "dt":
        values = torch.zeros(2 * group_heads)
        values[1::2] = operands["dt"][first:last]
        return values
    if packet.field == "d":
        values = torch.zeros(2 * group_heads)
        values[0::2] = 1.0
        values[1::2] = operands["d"][first:last]
        return values
    row_first = chunk * working_set.state_rows_per_chunk
    row_last = row_first + working_set.state_rows_per_chunk
    if packet.field == "update":
        values = torch.empty(working_set.state_rows_per_chunk, 2 * group_heads)
        values[:, 0::2] = operands["a"][first:last, row_first:row_last].T
        values[:, 1::2] = operands["b"][first:last, row_first:row_last].T
        return values
    if packet.field == "c":
        values = torch.zeros(working_set.state_rows_per_chunk, 2 * group_heads)
        values[:, :group_heads] = operands["c"][first:last, row_first:row_last].T
        return values
    raise KeyError(packet.field)


def _kda_inputs(token: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(SEED + 1009 + 101 * token)
    heads, keys, values = KIMI_KDA.heads, KIMI_KDA.recurrence_rows, KIMI_KDA.row_elements
    return {
        "decay": _bf16(0.84 + torch.rand(heads, keys, generator=generator) * 0.12),
        "key": _bf16(torch.randn(heads, keys, generator=generator) * 0.025),
        "query": _bf16(torch.randn(heads, keys, generator=generator) * 0.025),
        "value": _bf16(torch.randn(heads, values, generator=generator) * 0.08),
        "beta": _bf16(0.2 + torch.rand(heads, generator=generator) * 0.35),
    }


def _kda_reference(
    state: torch.Tensor,
    operands: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    state = _bf16(operands["decay"][:, :, None] * state)
    prediction = torch.zeros_like(operands["value"])
    for key in range(KIMI_KDA.recurrence_rows):
        prediction += state[:, key, :] * operands["key"][:, key, None]
    prediction = _bf16(prediction)
    error = _bf16(
        operands["beta"][:, None]
        * (operands["value"] - prediction)
    )
    state = _bf16(state + operands["key"][:, :, None] * error[:, None, :])
    output = torch.zeros_like(error)
    for key in range(KIMI_KDA.recurrence_rows):
        output += state[:, key, :] * operands["query"][:, key, None]
    return _bf16(output), state


def _kda_packet_values(
    packet: RecurrenceFieldPacket,
    operands: dict[str, torch.Tensor],
    working_set: RecurrenceWorkingSet,
) -> torch.Tensor:
    group_heads = working_set.group_heads
    first = packet.group * group_heads
    last = first + group_heads
    if packet.field in {"prediction_zero", "output_zero", "output_result"}:
        return torch.zeros(packet.logical_values)
    if packet.field == "value":
        return operands["value"][first:last]
    if packet.field == "beta":
        values = torch.empty(2 * group_heads)
        values[0::2] = operands["beta"][first:last]
        values[1::2] = -operands["beta"][first:last]
        return values
    chunk = 0 if packet.chunk is None else packet.chunk
    row_first = chunk * working_set.state_rows_per_chunk
    row_last = row_first + working_set.state_rows_per_chunk
    descriptor = working_set.allocation(packet.target).descriptor
    if packet.field == "decay":
        values = torch.zeros(group_heads, 2, descriptor.shape.cols)
        values[:, 0, : working_set.state_rows_per_chunk] = operands["decay"][
            first:last, row_first:row_last
        ]
        return values
    if packet.field in {"key", "query"}:
        values = torch.zeros(group_heads, 1, descriptor.shape.cols)
        values[:, 0, : working_set.state_rows_per_chunk] = operands[packet.field][
            first:last, row_first:row_last
        ]
        return values
    raise KeyError(packet.field)


def _state_seed(spec: MatrixRecurrenceSpec) -> torch.Tensor:
    generator = torch.Generator().manual_seed(
        SEED + (0 if spec is NEMOTRON_MAMBA else 5003)
    )
    return _bf16(
        torch.randn(
            spec.heads,
            spec.recurrence_rows,
            spec.row_elements,
            generator=generator,
        )
        * 0.04
    )


def _pack_state_hbm(state: torch.Tensor, working_set: RecurrenceWorkingSet) -> bytes:
    """Pack logical [head,row,lane] state into the Compiler's DMA packet ABI."""

    packets = []
    for group in range(working_set.groups):
        head_first = group * working_set.group_heads
        head_last = head_first + working_set.group_heads
        for chunk in range(working_set.chunks):
            row_first = chunk * working_set.state_rows_per_chunk
            row_last = row_first + working_set.state_rows_per_chunk
            packets.append(state[head_first:head_last, row_first:row_last, :])
    return b"".join(_bf16_bytes(packet) for packet in packets)


def _unpack_state_hbm(
    image: bytes,
    working_set: RecurrenceWorkingSet,
) -> torch.Tensor:
    """Restore logical [head,row,lane] state from the packet-major DMA ABI."""

    spec = working_set.spec
    state = torch.empty(
        spec.heads,
        spec.recurrence_rows,
        spec.row_elements,
        dtype=torch.float32,
    )
    packet_values = (
        working_set.group_heads
        * working_set.state_rows_per_chunk
        * spec.row_elements
    )
    packet_bytes = packet_values * BF16_BYTES
    packet_index = 0
    for group in range(working_set.groups):
        head_first = group * working_set.group_heads
        head_last = head_first + working_set.group_heads
        for chunk in range(working_set.chunks):
            row_first = chunk * working_set.state_rows_per_chunk
            row_last = row_first + working_set.state_rows_per_chunk
            values = _read_bf16(
                image,
                packet_index * packet_bytes,
                packet_values,
            ).reshape(
                working_set.group_heads,
                working_set.state_rows_per_chunk,
                spec.row_elements,
            )
            state[head_first:head_last, row_first:row_last, :] = values
            packet_index += 1
    return state


def _write_settings(build_dir: Path, point: MatrixSramPoint) -> Path:
    with (REPO_ROOT / "plena_settings.toml").open() as file:
        config = tomlkit.load(file)
    txn = config["TRANSACTIONAL"]["CONFIG"]
    txn["MLEN"]["value"] = point.mlen
    txn["VLEN"]["value"] = point.mlen
    txn["BLEN"]["value"] = point.bank_width
    txn["HLEN"]["value"] = 128
    txn["BROADCAST_AMOUNT"]["value"] = point.bank_width
    # The transactional Matrix SRAM setting is a count of MLEN-wide physical
    # rows, not a count of scalar elements.  At the paper point this is
    # 1 MiB / (2048 values * 2 B) = 256 rows.
    txn["MATRIX_SRAM_SIZE"]["value"] = point.depth_rows
    # This connected program never uses ordinary Vector-SRAM operands.  Keep a
    # small legal instance so the mandatory post-run dump stays bounded.
    txn["VECTOR_SRAM_SIZE"]["value"] = 64
    txn["HBM_V_Prefetch_Amount"]["value"] = 1
    txn["HBM_V_Writeback_Amount"]["value"] = 1
    path = build_dir / "plena_settings.toml"
    with path.open("w") as file:
        tomlkit.dump(config, file)
    return path


@contextmanager
def _setting_override(path: Path) -> Iterator[None]:
    previous = os.environ.get("PLENA_SETTINGS_TOML")
    os.environ["PLENA_SETTINGS_TOML"] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PLENA_SETTINGS_TOML", None)
        else:
            os.environ["PLENA_SETTINGS_TOML"] = previous


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    error = (actual - expected).abs()
    max_abs = float(error.max()) if error.numel() else 0.0
    relative_l2 = float(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(expected).clamp_min(1e-12))
    if not torch.allclose(actual, expected, atol=1e-2, rtol=1e-2):
        mismatch = int((error > (1e-2 + 1e-2 * expected.abs())).sum())
        raise AssertionError(
            f"{name}: {mismatch}/{actual.numel()} values mismatch; "
            f"max_abs={max_abs}, relative_l2={relative_l2}"
        )
    return {"max_abs": max_abs, "relative_l2": relative_l2}


def run_case(
    spec: MatrixRecurrenceSpec,
    layout: RecurrenceLayout,
    output_root: Path,
    *,
    keep_build: bool = False,
) -> dict[str, object]:
    output_root = output_root.resolve()
    point = MatrixSramPoint()
    working_set = build_recurrence_working_set(spec, layout=layout, point=point)
    state = _state_seed(spec)
    expected_state = state.clone()
    expected_outputs: list[torch.Tensor] = []
    manifests: list[RecurrenceFieldManifest] = []
    assemblies: list[str] = []
    field_base = _round_up(spec.state_bytes_per_layer, 64)

    for token in range(TOKENS):
        manifest = build_recurrence_field_manifest(
            working_set,
            field_hbm_base=field_base,
        )
        assembly = lower_matrix_recurrence(
            spec,
            layout=layout,
            point=point,
            state_hbm_base=0,
            field_hbm_base=field_base,
        )
        validate_recurrence_output_stores(
            assembly,
            expected_groups=working_set.groups,
        )
        operands = _mamba_inputs(token) if spec is NEMOTRON_MAMBA else _kda_inputs(token)
        if spec is NEMOTRON_MAMBA:
            expected_output, expected_state = _mamba_reference(expected_state, operands)
        else:
            expected_output, expected_state = _kda_reference(expected_state, operands)
        expected_outputs.append(expected_output)
        manifests.append(manifest)
        assemblies.append(assembly)
        field_base = _round_up(manifest.end, 64)

    program = "\n".join(assemblies)
    validate_matrix_view_dominance(program)
    assembly_sha256 = hashlib.sha256(program.encode()).hexdigest()
    build_dir = output_root / spec.name / layout.value
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    (build_dir / "generated_asm_code.asm").write_text(program)
    assembler = AssemblyToBinary(
        str(COMPILER_ROOT / "doc" / "operation.svh"),
        str(COMPILER_ROOT / "doc" / "configuration.svh"),
    )
    assembler.generate_binary(
        str(build_dir / "generated_asm_code.asm"),
        str(build_dir / "generated_machine_code.mem"),
    )
    machine_code = (build_dir / "generated_machine_code.mem").read_bytes()
    machine_code_sha256 = hashlib.sha256(machine_code).hexdigest()
    machine_words = sum(1 for line in machine_code.splitlines() if line.strip())

    image = bytearray(field_base)
    image[: spec.state_bytes_per_layer] = _pack_state_hbm(state, working_set)
    for token, manifest in enumerate(manifests):
        operands = _mamba_inputs(token) if spec is NEMOTRON_MAMBA else _kda_inputs(token)
        for packet in manifest.packets:
            values = (
                _mamba_packet_values(packet, operands, working_set)
                if spec is NEMOTRON_MAMBA
                else _kda_packet_values(packet, operands, working_set)
            )
            _write_packet(image, packet, values)
    (build_dir / "hbm_for_behave_sim.bin").write_bytes(image)
    input_hbm_sha256 = hashlib.sha256(image).hexdigest()
    (build_dir / "fp_sram.bin").write_bytes(bytes(64))
    (build_dir / "int_sram.bin").write_bytes(bytes(64))
    settings = _write_settings(build_dir, point)
    with _setting_override(settings):
        metrics = run_emulator(
            build_dir,
            hbm_size=_round_up(len(image), 64),
            threads=1,
            dump_cwd=build_dir,
            dump_hbm=True,
        )

    post = (build_dir / "hbm_dump.bin").read_bytes()
    actual_state = _unpack_state_hbm(post, working_set)
    state_error = _assert_close("final recurrent state", actual_state, expected_state)
    output_errors = []
    for token, (manifest, expected) in enumerate(zip(manifests, expected_outputs, strict=True)):
        actual_groups = []
        for group in range(working_set.groups):
            packet = manifest.packet("output_result", group=group)
            values = _read_bf16(post, packet.hbm_byte_offset, packet.logical_values)
            actual_groups.append(
                values.reshape(working_set.group_heads, spec.row_elements)
            )
        actual = torch.cat(actual_groups, dim=0)
        output_errors.append(_assert_close(f"token {token} output", actual, expected))

    counters = metrics.get("matrix_view_packet_counters", {})
    result: dict[str, object] = {
        "schema_version": 1,
        "model": spec.name,
        "layout": str(layout),
        "precision": "bf16_uniform_matrix_recurrence",
        "tokens": TOKENS,
        "official_shape": {
            "heads": spec.heads,
            "state_rows": spec.recurrence_rows,
            "row_elements": spec.row_elements,
        },
        "compiler_generated_machine_words": machine_words,
        "assembly_sha256": assembly_sha256,
        "machine_code_sha256": machine_code_sha256,
        "input_hbm_sha256": input_hbm_sha256,
        "l_tile_exec_count": program.count("L_TILE_EXEC"),
        "state_values_compared": state.numel(),
        "output_values_compared": sum(output.numel() for output in expected_outputs),
        "state_error": state_error,
        "output_error": {
            "max_abs": max(error["max_abs"] for error in output_errors),
            "relative_l2": max(error["relative_l2"] for error in output_errors),
        },
        "rust_simulation_cycles": metrics.get("sim_latency_cycles"),
        "matrix_view_packet_counters": counters,
        "evidence": (
            "Compiler assembly -> Compiler assembler -> Rust opcode decoder -> "
            "BF16 banked Matrix SRAM -> explicit HBM state/output readback"
        ),
        "temporary_build_retained": keep_build,
    }
    (build_dir / "connected_result.json").write_text(json.dumps(result, indent=2) + "\n")
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / f"{spec.name}_{layout.value}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    if not keep_build:
        shutil.rmtree(build_dir)
        model_dir = build_dir.parent
        if model_dir.exists() and not any(model_dir.iterdir()):
            model_dir.rmdir()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "matrix_lcompute_connected_bf16",
    )
    parser.add_argument(
        "--model",
        choices=("mamba", "kda", "both"),
        default="both",
    )
    parser.add_argument(
        "--layout",
        choices=("fixed", "affine", "both"),
        default="both",
        help="Run the single-base fixed descriptor, affine descriptor, or both.",
    )
    parser.add_argument(
        "--keep-build",
        action="store_true",
        help="Keep generated assembly, machine code and HBM dumps for debugging.",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    specs = {
        "mamba": NEMOTRON_MAMBA,
        "kda": KIMI_KDA,
    }
    selected = specs.values() if args.model == "both" else (specs[args.model],)
    layouts = (
        (RecurrenceLayout.FIXED, RecurrenceLayout.AFFINE)
        if args.layout == "both"
        else (RecurrenceLayout(args.layout),)
    )
    results = [
        run_case(spec, layout, args.output_dir, keep_build=args.keep_build)
        for spec in selected
        for layout in layouts
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
