"""Shared helpers for the GPT-OSS routed-MoE testbench scripts.

These helpers were previously copy-pasted (byte-identical) across the
``routed_moe/`` and ``models/gpt_oss/`` test scripts. They are consolidated
here so there is a single source of truth. Names keep their historical leading
underscore so existing call sites in the test scripts need no changes beyond
importing them from this module.

Only helpers that were identical across every defining file live here; helpers
that legitimately differ between tests (e.g. ``_activation_golden``,
``_linear_projection_golden``) are intentionally left in their own files.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import torch

from transactional_emulator.testbench.aten.golden import quantize_to_mxfp


class MxfpWeightCache:
    """Cache immutable synthetic weights after hardware MX quantization.

    Full-backbone references reuse each layer weight for every token. Tracking
    the tensor version keeps the cache correct if a test mutates a weight in
    place while avoiding thousands of identical quantizer calls.
    """

    def __init__(self) -> None:
        self._entries: dict[int, tuple[torch.Tensor, int, torch.Tensor]] = {}

    def quantize(self, weight: torch.Tensor) -> torch.Tensor:
        key = id(weight)
        version = weight._version
        entry = self._entries.get(key)
        if entry is not None and entry[0] is weight and entry[1] == version:
            return entry[2]
        quantized = quantize_to_mxfp(weight)
        self._entries[key] = (weight, version, quantized)
        return quantized


def _scan_cache_append_tokens(
    assembly: str,
    backing_names: Iterable[str],
) -> dict[str, list[int]]:
    """Collect cache-append token indices without materializing all ASM lines."""
    tokens = {name: [] for name in backing_names}
    marker = "DECODE_CACHE_APPEND "
    cursor = 0
    while True:
        marker_start = assembly.find(marker, cursor)
        if marker_start < 0:
            return tokens
        line_end = assembly.find("\n", marker_start)
        if line_end < 0:
            line_end = len(assembly)
        name_start = marker_start + len(marker)
        name_end = assembly.find(" ", name_start, line_end)
        if name_end > name_start:
            name = assembly[name_start:name_end]
            if name in tokens:
                token_start = assembly.find("token=", name_end, line_end)
                if token_start >= 0:
                    token_start += len("token=")
                    token_end = token_start
                    while token_end < line_end and assembly[token_end].isdigit():
                        token_end += 1
                    if token_end == token_start:
                        raise ValueError(f"cache append for {name} has no numeric token")
                    tokens[name].append(int(assembly[token_start:token_end]))
        cursor = line_end + 1


def _machine_code_line_count(metrics: Mapping[str, object]) -> int:
    """Read the instruction count already collected by the emulator runner."""
    artifacts = metrics.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("emulator metrics do not contain an artifact summary")
    count = artifacts.get("machine_code_lines")
    if not isinstance(count, int) or count < 0:
        raise ValueError("emulator metrics do not contain a valid machine-code line count")
    return count


def _activation_golden(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """BF16 GPT-OSS clamp-gated SwiGLU activation, matching the vector-ISA emit sequence."""
    neg_alpha = torch.tensor(-1.702, dtype=torch.bfloat16).float()

    gate = _bf16(torch.clamp(gate.float(), max=7.0))
    up = _bf16(torch.clamp(up.float(), max=7.0))
    up = _bf16(torch.clamp(up.float(), min=-7.0))

    sigmoid = _bf16(gate.float())
    sigmoid = _bf16(sigmoid.float() * neg_alpha)
    sigmoid = _bf16(torch.exp(torch.clamp(sigmoid.float(), -88.0, 88.0)))
    sigmoid = _bf16(sigmoid.float() + 1.0)
    sigmoid = _bf16(torch.reciprocal(sigmoid.float()))

    glu = _bf16(gate.float() * sigmoid.float())
    up_plus_one = _bf16(up.float() + 1.0)
    return _bf16(up_plus_one.float() * glu.float())


def _linear_projection_golden(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    mlen: int,
    mram_tile_capacity: int = 4,
    hbm_input: bool = True,
    weight_cache: MxfpWeightCache | None = None,
) -> torch.Tensor:
    """Hardware-aware linear projection golden, including compiler K-split.

    HBM-loaded activations and weights are MX-quantized before matmul. VRAM
    intermediate activations are already BF16 and must not be MX-quantized
    again. When K exceeds MRAM tile capacity, the compiler emits partial matmuls
    and BF16 VRAM adds; mirror that rounding order here.
    """
    x_q = quantize_to_mxfp(x) if hbm_input else x.to(torch.bfloat16)
    w_q = quantize_to_mxfp(w) if weight_cache is None else weight_cache.quantize(w)
    k_total = x_q.shape[1]
    chunk = mlen * mram_tile_capacity

    acc = None
    for k_start in range(0, k_total, chunk):
        k_end = min(k_start + chunk, k_total)
        partial = torch.matmul(x_q[:, k_start:k_end].float(), w_q[k_start:k_end, :].float()).to(torch.bfloat16)
        acc = partial if acc is None else (acc.float() + partial.float()).to(torch.bfloat16)
    assert acc is not None
    return acc


def _align_to(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def _exact_mxfp8_tensor(shape: tuple[int, ...], *, stride: int, offset: int = 0) -> torch.Tensor:
    values = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float32)
    idx = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.long)
    return values[(idx * stride + offset) % values.numel()].reshape(shape)


def _decode_bf16_dump(path: Path) -> torch.Tensor:
    raw = np.fromfile(path, dtype="<u2")
    return torch.tensor(raw.astype(np.uint16), dtype=torch.uint16).view(torch.bfloat16)


def _decode_u32_dump(path: Path) -> torch.Tensor:
    raw = np.fromfile(path, dtype="<u4")
    return torch.tensor(raw.astype(np.uint32), dtype=torch.int64)


def _expanded_bias(row_bias: torch.Tensor, rows: int) -> torch.Tensor:
    return row_bias.to(torch.bfloat16).reshape(1, -1).repeat(rows, 1)


def _stats_dict(stats) -> dict:
    return {
        "rel_rms": stats.rel_rms,
        "atol": stats.atol,
        "rtol": stats.rtol,
        "allclose": stats.allclose,
        "pass_rate": stats.pass_rate,
        "max_abs_error": stats.max_abs_error,
    }


def _comparison_params_for(output_vram, *, rows: int, hidden: int, mlen: int, golden: torch.Tensor) -> dict:
    output_vram_addr = output_vram._program._compiler.get_vram_addr(output_vram.name)
    output_physical_rows = output_vram.physical_shape[0]
    chunks_per_batch = math.ceil(hidden / mlen)
    rows_to_read = (chunks_per_batch - 1) * output_physical_rows + rows
    return {
        "start_row_idx": output_vram_addr // mlen,
        "num_rows": rows_to_read,
        "num_batches": rows,
        "elements_per_batch": hidden,
        "row_dim": mlen,
        "physical_rows": output_physical_rows,
        "atol": float((golden.float().std(unbiased=False) * 0.01).item()),
        "rtol": 0.02,
    }


def _comparison_diagnostics(
    results: dict,
    *,
    checkpoint_stages: int,
    total_tokens: int,
    hidden: int,
    prefill_tokens: int,
) -> dict[str, float | int]:
    golden = results["golden_values"].float()
    simulated = results["simulated_values"].float()
    expected_values = checkpoint_stages * total_tokens * hidden
    if golden.numel() != expected_values or simulated.numel() != expected_values:
        raise ValueError(
            "comparison shape does not match checkpoint topology: "
            f"expected={expected_values}, golden={golden.numel()}, "
            f"simulated={simulated.numel()}"
        )
    golden = golden.reshape(checkpoint_stages, total_tokens, hidden)
    simulated = simulated.reshape_as(golden)
    errors = (golden - simulated).abs()
    within = errors <= float(results["atol"]) + float(results["rtol"]) * golden.abs()
    delta = simulated - golden

    def rate(mask: torch.Tensor) -> float:
        return float(mask.float().mean().item() * 100.0)

    denominator = torch.linalg.vector_norm(golden)
    relative_l2 = torch.linalg.vector_norm(delta) / denominator.clamp_min(1e-12)
    final_denominator = torch.linalg.vector_norm(golden[-1])
    final_relative_l2 = torch.linalg.vector_norm(delta[-1]) / final_denominator.clamp_min(1e-12)
    per_token_rate = within.float().mean(dim=(0, 2)) * 100.0
    per_stage_rate = within.float().mean(dim=(1, 2)) * 100.0
    return {
        "compared_values": int(golden.numel()),
        "mismatch_values": int((~within).sum().item()),
        "relative_l2_error": float(relative_l2.item()),
        "prefill_allclose_match_rate": rate(within[:, :prefill_tokens]),
        "decode_allclose_match_rate": rate(within[:, prefill_tokens:]),
        "final_layer_allclose_match_rate": float(per_stage_rate[-1].item()),
        "final_layer_relative_l2_error": float(final_relative_l2.item()),
        "last_token_allclose_match_rate": float(per_token_rate[-1].item()),
        "worst_token_allclose_match_rate": float(per_token_rate.min().item()),
        "worst_layer_allclose_match_rate": float(per_stage_rate.min().item()),
    }
