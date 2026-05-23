"""Shared helper utilities for SigLIP testbench scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


ENCODER_HBM_DATA_ORDER = ["WQ", "K", "V", "W1", "W2"]


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


def tensor_metrics(pred: torch.Tensor, target: torch.Tensor, *, atol: float = 1e-2, rtol: float = 1e-2) -> dict:
    """Compute compact tensor similarity metrics used by test harnesses."""
    pred_f = pred.float()
    target_f = target.float()
    abs_err = (pred_f - target_f).abs()
    return {
        "allclose_pass": bool(torch.isclose(pred_f, target_f, atol=atol, rtol=rtol).all().item()),
        "mse": float(torch.mean((pred_f - target_f) ** 2).item()),
        "mae": float(torch.mean(abs_err).item()),
        "max_error": float(torch.max(abs_err).item()),
        "match_rate": float(torch.isclose(pred_f, target_f, atol=atol, rtol=rtol).float().mean().item() * 100.0),
    }


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
    "json_default",
    "resolve_position_embedding",
    "resolve_vision_encoder_layer",
    "tensor_metrics",
    "write_golden_values_file",
]
