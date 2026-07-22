"""Pure, self-contained helpers for attention_semantics_test.

Golden references, summary/verdict builders, and small tensor utilities that
take all their inputs as arguments (no module-level state). Split out of
attention_semantics_test.py to separate reusable utilities from the large
runnable scenarios. Behaviour is unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def _rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float()
    b_f = b.float()
    return float(torch.linalg.vector_norm(a_f - b_f) / torch.clamp(torch.linalg.vector_norm(b_f), min=1e-12))


def _router_bias_block_rows(router_bias: torch.Tensor, *, rows: int, num_experts: int, mlen: int) -> torch.Tensor:
    expert_blocks = math.ceil(num_experts / mlen)
    if expert_blocks == 1:
        return router_bias.reshape(1, -1).repeat(rows, 1).to(torch.bfloat16)
    out = torch.zeros(rows * expert_blocks, mlen, dtype=torch.bfloat16)
    for token_idx in range(rows):
        for block_idx in range(expert_blocks):
            col_start = block_idx * mlen
            col_end = min(col_start + mlen, num_experts)
            out[token_idx * expert_blocks + block_idx, : col_end - col_start] = router_bias[col_start:col_end]
    return out


def _strict_tail_summary(actual: torch.Tensor, reference: torch.Tensor, *, rtol: float = 0.02) -> dict:
    actual_f = actual.float()
    ref_f = reference.float()
    diff = actual_f - ref_f
    finite = torch.isfinite(actual_f) & torch.isfinite(ref_f) & torch.isfinite(diff)
    abs_diff = diff.abs()
    safe_abs_diff = torch.where(torch.isfinite(abs_diff), abs_diff, torch.full_like(abs_diff, float("inf")))
    atol = ref_f.std(unbiased=False) * 0.01
    allowed = atol + rtol * ref_f.abs()
    failed = (~finite) | (safe_abs_diff > allowed)
    finite_abs = safe_abs_diff[finite]
    sigma = finite_abs.std(unbiased=False) if finite_abs.numel() else torch.tensor(float("inf"))
    k_sweep = []
    for k in (2.0, 2.5, 3.0, 3.5, 4.0):
        k_allowed = k * sigma + rtol * ref_f.abs()
        k_failed = (~finite) | (safe_abs_diff > k_allowed)
        k_sweep.append({"k": k, "fail_count": int(k_failed.sum().item())})
    failed_signed = diff[failed]
    finite_safe_abs = safe_abs_diff[torch.isfinite(safe_abs_diff)]
    return {
        "atol": float(atol.item()),
        "rtol": rtol,
        "fail_count": int(failed.sum().item()),
        "numel": int(failed.numel()),
        "pass_rate": float((~failed).float().mean().item()),
        "allclose": bool((~failed).all().item()),
        "nonfinite_count": int((~finite).sum().item()),
        "max_abs_error": float(finite_safe_abs.max().item()) if finite_safe_abs.numel() else float("inf"),
        "sigma_abs_error": float(sigma.item()),
        "k_sweep": k_sweep,
        "sign": {
            "positive": int((failed_signed > 0).sum().item()) if failed_signed.numel() else 0,
            "negative": int((failed_signed < 0).sum().item()) if failed_signed.numel() else 0,
            "zero": int((failed_signed == 0).sum().item()) if failed_signed.numel() else 0,
        },
    }


def _topk_match_summary(
    *,
    host_indices: torch.Tensor,
    host_weights: torch.Tensor,
    device_indices: torch.Tensor,
    device_weights: torch.Tensor,
) -> dict:
    set_matches = []
    order_matches = []
    for token_idx in range(host_indices.shape[0]):
        host = [int(v) for v in host_indices[token_idx].tolist()]
        dev = [int(v) for v in device_indices[token_idx].tolist()]
        set_matches.append(set(host) == set(dev))
        order_matches.append(host == dev)
    weight_diff = (device_weights.float() - host_weights.float()).abs()
    return {
        "host_indices": host_indices.cpu().tolist(),
        "device_indices": device_indices.cpu().tolist(),
        "host_weights": host_weights.cpu().float().tolist(),
        "device_weights": device_weights.cpu().float().tolist(),
        "set_matches_by_token": set_matches,
        "order_matches_by_token": order_matches,
        "set_match_count": int(sum(set_matches)),
        "order_match_count": int(sum(order_matches)),
        "num_tokens": int(host_indices.shape[0]),
        "weight_rel_rms": _rel_rms(device_weights, host_weights),
        "weight_max_abs_error": float(weight_diff.max().item()) if weight_diff.numel() else 0.0,
    }


def _router_margin_summary(logits: torch.Tensor, top_k: int) -> dict:
    logits_f = logits.float()
    sorted_vals, sorted_idx = torch.sort(logits_f, dim=-1, descending=True, stable=True)
    if sorted_vals.shape[-1] <= top_k:
        gaps = torch.full((sorted_vals.shape[0],), float("inf"))
        rank_next = torch.full((sorted_vals.shape[0],), -1, dtype=torch.long)
        next_vals = torch.full((sorted_vals.shape[0],), float("-inf"))
    else:
        gaps = sorted_vals[:, top_k - 1] - sorted_vals[:, top_k]
        rank_next = sorted_idx[:, top_k]
        next_vals = sorted_vals[:, top_k]
    top_vals = sorted_vals[:, :top_k]
    top_idx = sorted_idx[:, :top_k]
    return {
        "top_k": int(top_k),
        "min_rank_k_to_next_gap": float(gaps.min().item()) if gaps.numel() else float("inf"),
        "per_token": [
            {
                "token_index": int(token_idx),
                "top_indices": [int(v) for v in top_idx[token_idx].tolist()],
                "top_values": [float(v) for v in top_vals[token_idx].tolist()],
                "rank_k_index": int(top_idx[token_idx, top_k - 1].item()),
                "rank_k_value": float(top_vals[token_idx, top_k - 1].item()),
                "rank_next_index": int(rank_next[token_idx].item()),
                "rank_next_value": float(next_vals[token_idx].item()),
                "rank_k_to_next_gap": float(gaps[token_idx].item()),
            }
            for token_idx in range(logits.shape[0])
        ],
    }


def _comparison_params(vram_addr: int, rows: int, cols: int, mlen: int, physical_rows: int | None = None) -> dict:
    physical_rows = physical_rows or rows
    num_col_blocks = math.ceil(cols / mlen)
    rows_to_read = (num_col_blocks - 1) * physical_rows + rows
    return {
        "start_row_idx": vram_addr // mlen,
        "num_rows": rows_to_read,
        "num_batches": rows,
        "elements_per_batch": cols,
        "row_dim": mlen,
        "physical_rows": physical_rows,
        "atol": 0.02,
        "rtol": 0.02,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _resolve_sliding_window(args: argparse.Namespace) -> tuple[int | None, str]:
    if args.sliding_window is not None:
        return args.sliding_window, "explicit"
    if args.config_json is None:
        return None, "none"
    if args.layer_idx is None:
        raise ValueError("--layer-idx is required when --config-json is used")
    with args.config_json.expanduser().open() as f:
        config = json.load(f)
    layer_types = config.get("layer_types")
    if not layer_types:
        return None, "config:no_layer_types"
    layer_type = layer_types[args.layer_idx]
    if "sliding" not in layer_type:
        return None, f"config:{layer_type}"
    return int(config["sliding_window"]), f"config:{layer_type}"


def _bias_parts(args: argparse.Namespace) -> set[str]:
    if not getattr(args, "projection_bias", False):
        return set()
    parts = (getattr(args, "bias_parts", "qkvo") or "").lower()
    aliases = {"all": "qkvo", "none": ""}
    parts = aliases.get(parts, parts)
    allowed = set("qkvo")
    unknown = set(parts) - allowed
    if unknown:
        raise ValueError(f"unknown --bias-parts entries: {sorted(unknown)}")
    return set(parts)


def _make_rotate_half_matrix(head_dim: int) -> torch.Tensor:
    rotate = torch.zeros(head_dim, head_dim, dtype=torch.bfloat16)
    half = head_dim // 2
    for i in range(half):
        rotate[i + half, i] = -1.0
        rotate[i, i + half] = 1.0
    return rotate


def _make_packed_rope_inputs(
    seq: int, packed_heads: int, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = packed_heads * head_dim
    rotate = torch.zeros(width, width, dtype=torch.bfloat16)
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half).float() / half))
    angles = torch.outer(torch.arange(seq).float(), freqs)
    cos_head = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1).to(torch.bfloat16)
    sin_head = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1).to(torch.bfloat16)
    cos = torch.zeros(seq, width, dtype=torch.bfloat16)
    sin = torch.zeros(seq, width, dtype=torch.bfloat16)
    head_rotate = _make_rotate_half_matrix(head_dim)
    for head in range(packed_heads):
        start = head * head_dim
        rotate[start : start + head_dim, start : start + head_dim] = head_rotate
        cos[:, start : start + head_dim] = cos_head
        sin[:, start : start + head_dim] = sin_head
    return rotate.contiguous(), cos.contiguous(), sin.contiguous()


def _align_to_tile(value: int, mlen: int) -> int:
    return math.ceil(value / (mlen * mlen)) * (mlen * mlen)
