"""Independent BF16 references and memory fixtures for Matrix L-Compute tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path

import numpy as np
import tomlkit
import torch

from compiler.aten.plena.program_lcompute import (
    BF16_BYTES,
    KIMI_KDA,
    NEMOTRON_MAMBA,
    MatrixRecurrenceSpec,
    MatrixSramPoint,
    RecurrenceFieldPacket,
    RecurrenceKind,
    RecurrenceWorkingSet,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
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
        raise AssertionError(f"{packet.key}: generated {flat.numel()} values, expected {packet.logical_values}")
    padded = torch.zeros(packet.transfer_values, dtype=torch.float32)
    padded[: flat.numel()] = flat
    payload = _bf16_bytes(padded)
    begin = packet.hbm_byte_offset
    end = begin + len(payload)
    image[begin:end] = payload


def _mamba_inputs(token: int, seed: int = SEED) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 101 * token)
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
    state = _bf16(operands["a"][:, :, None] * state + operands["b"][:, :, None] * scratch[:, None, :])
    accumulator = torch.zeros_like(x)
    for row in range(state.shape[1]):
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
        values = torch.zeros(packet.logical_values)
        values[1 : 2 * group_heads : 2] = operands["dt"][first:last]
        return values
    if packet.field == "d":
        values = torch.zeros(packet.logical_values)
        values[0 : 2 * group_heads : 2] = 1.0
        values[1 : 2 * group_heads : 2] = operands["d"][first:last]
        return values
    row_first = chunk * working_set.state_rows_per_chunk
    row_last = row_first + working_set.state_rows_per_chunk
    if packet.field == "update":
        values = torch.zeros(
            working_set.state_rows_per_chunk,
            working_set.allocation(packet.target).descriptor.shape.cols,
        )
        values[:, 0 : 2 * group_heads : 2] = operands["a"][first:last, row_first:row_last].T
        values[:, 1 : 2 * group_heads : 2] = operands["b"][first:last, row_first:row_last].T
        return values
    if packet.field == "c":
        values = torch.zeros(
            working_set.state_rows_per_chunk,
            working_set.allocation(packet.target).descriptor.shape.cols,
        )
        values[:, :group_heads] = operands["c"][first:last, row_first:row_last].T
        return values
    raise KeyError(packet.field)


def _kda_inputs(token: int, seed: int = SEED) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 1009 + 101 * token)
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
    for key in range(state.shape[1]):
        prediction += state[:, key, :] * operands["key"][:, key, None]
    prediction = _bf16(prediction)
    error = _bf16(operands["beta"][:, None] * (operands["value"] - prediction))
    state = _bf16(state + operands["key"][:, :, None] * error[:, None, :])
    output = torch.zeros_like(error)
    for key in range(state.shape[1]):
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
        values[:, 0, : working_set.state_rows_per_chunk] = operands["decay"][first:last, row_first:row_last]
        return values
    if packet.field in {"key", "query"}:
        values = torch.zeros(group_heads, 1, descriptor.shape.cols)
        values[:, 0, : working_set.state_rows_per_chunk] = operands[packet.field][first:last, row_first:row_last]
        return values
    raise KeyError(packet.field)


def _state_seed(spec: MatrixRecurrenceSpec, seed: int = SEED) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + (0 if spec.kind is RecurrenceKind.MAMBA else 5003))
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
    packet_values = working_set.group_heads * working_set.state_rows_per_chunk * spec.row_elements
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


# BF16 recurrence acceptance policy: keep the existing per-element outlier
# bound and additionally cap aggregate relative L2 error at 1%. Fixed/chunked
# reductions round at different boundaries; this is an explicit error budget,
# not a claim that their observed error is the ISA's mathematical tolerance.
# For effectively zero tensors, permit at most 1e-7 RMS absolute error so the
# norm test does not divide by zero or reject harmless sub-signal roundoff.
RECURRENCE_RELATIVE_L2_LIMIT = 1e-2
RECURRENCE_ZERO_RMS_LIMIT = 1e-7


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, *, exact: bool = False) -> dict[str, float]:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError(f"{name}: non-finite actual or reference values")
    if exact and not torch.equal(actual.view(torch.int32), expected.view(torch.int32)):
        raise AssertionError(f"{name}: exact BF16 comparison failed")
    error = (actual - expected).abs()
    max_abs = float(error.max()) if error.numel() else 0.0
    error_norm = torch.linalg.vector_norm(error)
    expected_norm = torch.linalg.vector_norm(expected)
    relative_l2 = float(error_norm / expected_norm.clamp_min(1e-12))
    norm_budget = max(
        RECURRENCE_RELATIVE_L2_LIMIT * float(expected_norm),
        RECURRENCE_ZERO_RMS_LIMIT * error.numel() ** 0.5,
    )
    mismatch = int((error > (1e-2 + 1e-2 * expected.abs())).sum())
    if mismatch or float(error_norm) > norm_budget:
        raise AssertionError(
            f"{name}: {mismatch}/{actual.numel()} values mismatch; max_abs={max_abs}, "
            f"relative_l2={relative_l2}, error_l2={float(error_norm)}, l2_budget={norm_budget}"
        )
    return {"max_abs": max_abs, "relative_l2": relative_l2}
