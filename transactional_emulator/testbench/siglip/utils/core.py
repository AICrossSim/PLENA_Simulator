"""Shared helper utilities for SigLIP testbench scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


ENCODER_HBM_DATA_ORDER = ["WQ", "K", "V", "WO", "W1", "W2"]


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def padded_head_dim(config: dict, mlen: int) -> int:
    """Per-head dimension padded up to the matrix tile size (mlen)."""
    head_dim = int(config["hidden_size"]) // int(config["num_attention_heads"])
    return align_up(head_dim, mlen)


def kv_activation_numel(config: dict, mlen: int, seq_len_kernel: int) -> int:
    """Element count of one layer's K (or V) flash-attn activation tile.

    This equals num_kv_heads * seq_len_kernel * d_padded, i.e. the size of the
    K/V HBM slot the flash kernel prefetches, which must match reset_kv_prefetch's
    SCALE_REG so the interleaved MXFP scales align with the prefetch's scale read.
    """
    num_kv_heads = int(config["num_key_value_heads"])
    return num_kv_heads * seq_len_kernel * padded_head_dim(config, mlen)


def resolve_vision_encoder_layer(model, layer_idx: int = 0):
    """Get SigLIP vision encoder layer by index across HF structure variants."""
    vision_root = getattr(model, "vision_model", model)
    encoder = getattr(vision_root, "encoder", None)
    if encoder is not None and hasattr(encoder, "layers"):
        return encoder.layers[layer_idx]
    if hasattr(vision_root, "layers"):
        return vision_root.layers[layer_idx]
    raise AttributeError("Could not locate SigLIP vision encoder layer")


def resolve_position_embedding(vision_root):
    """Extract position embedding table from a SigLIP vision model root."""
    embeddings = vision_root.embeddings
    for attr_name in ("position_embedding", "position_embeddings", "position_embed"):
        if hasattr(embeddings, attr_name):
            position_embedding = getattr(embeddings, attr_name)
            if hasattr(position_embedding, "weight"):
                return position_embedding.weight.detach()
            return position_embedding.detach()
    raise AttributeError("Could not locate SigLIP position embedding table")


def json_default(value):
    """JSON serializer fallback for numpy and torch values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def pad_to_batch_boundary(value: int, batch_boundary: int) -> int:
    """Pad value up to nearest multiple of batch_boundary."""
    return align_up(value, batch_boundary)


def pad_to_alignment(value: int, alignment: int) -> int:
    """Pad value up to nearest multiple of alignment."""
    return align_up(value, alignment)


def compute_padding_amount(current: int, target_alignment: int) -> int:
    """Compute number of elements needed to pad to alignment."""
    aligned = align_up(current, target_alignment)
    return aligned - current


def tensor_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    visible_lane_positions: list[int] | None = None,
) -> dict:
    """Compute compact tensor similarity metrics used by test harnesses."""
    pred_f = pred.float()
    target_f = target.float()
    abs_err = (pred_f - target_f).abs()

    metrics = {
        "allclose_pass": bool(torch.isclose(pred_f, target_f, atol=atol, rtol=rtol).all().item()),
        "mse": float(torch.mean((pred_f - target_f) ** 2).item()),
        "mae": float(torch.mean(abs_err).item()),
        "max_error": float(torch.max(abs_err).item()),
        "match_rate": float(torch.isclose(pred_f, target_f, atol=atol, rtol=rtol).float().mean().item() * 100.0),
    }

    if visible_lane_positions is not None and pred_f.ndim >= 2 and pred_f.shape == target_f.shape:
        hidden_dim = pred_f.shape[-1]
        idx = torch.tensor(visible_lane_positions, dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < hidden_dim)]
        if idx.numel() > 0:
            idx = torch.unique(idx, sorted=True)
            visible_pred = pred_f[..., idx]
            visible_target = target_f[..., idx]
            visible_err = (visible_pred - visible_target).abs().reshape(-1)
            visible_close = torch.isclose(visible_pred, visible_target, atol=atol, rtol=rtol)
            visible_stats = {
                "count": int(visible_err.numel()),
                "mae": float(torch.mean(visible_err).item()),
                "max_error": float(torch.max(visible_err).item()),
                "match_rate": float(visible_close.float().mean().item() * 100.0),
                "p99_error": float(
                    torch.quantile(visible_err.float(), 0.99).item()
                    if visible_err.numel() > 1
                    else torch.max(visible_err).item()
                ),
            }
            metrics["visible_lane_metrics"] = visible_stats

            padded_mask = torch.ones(hidden_dim, dtype=torch.bool)
            padded_mask[idx] = False
            padded_idx = torch.where(padded_mask)[0]
            if padded_idx.numel() > 0:
                padded_pred = pred_f[..., padded_idx]
                padded_target = target_f[..., padded_idx]
                padded_err = (padded_pred - padded_target).abs().reshape(-1)
                padded_close = torch.isclose(padded_pred, padded_target, atol=atol, rtol=rtol)
                metrics["padded_lane_metrics"] = {
                    "count": int(padded_err.numel()),
                    "mae": float(torch.mean(padded_err).item()),
                    "max_error": float(torch.max(padded_err).item()),
                    "match_rate": float(padded_close.float().mean().item() * 100.0),
                    "p99_error": float(
                        torch.quantile(padded_err.float(), 0.99).item()
                        if padded_err.numel() > 1
                        else torch.max(padded_err).item()
                    ),
                    "pred_nonzero_rate": float((padded_pred.abs() > 0).float().mean().item() * 100.0),
                }

    return metrics


def write_golden_values_file(golden_path: Path, values: torch.Tensor) -> None:
    """Write flattened golden tensor values in a stable text format."""
    values_np = values.detach().cpu().float().reshape(-1).numpy()
    with open(golden_path, "w") as f:
        f.write("Original Output:\n")
        f.write(
            np.array2string(
                values_np,
                separator=", ",
                max_line_width=120,
                threshold=values_np.size,
            )
        )

__all__ = [
    "ENCODER_HBM_DATA_ORDER",
    "align_up",
    "json_default",
    "resolve_position_embedding",
    "resolve_vision_encoder_layer",
    "tensor_metrics",
    "write_golden_values_file",
]
