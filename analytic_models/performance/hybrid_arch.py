"""Pinned, dependency-free architecture records for hybrid workload models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MambaArchConfig:
    num_heads: int
    head_dim: int
    state_dim: int
    groups: int
    conv_kernel: int
    chunk_size: int
    state_dtype: str = "float32"

    @property
    def d_inner(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def heads_per_group(self) -> int:
        return self.num_heads // self.groups

    @property
    def conv_channels(self) -> int:
        return self.d_inner + 2 * self.groups * self.state_dim

    @property
    def projection_size(self) -> int:
        return self.d_inner + self.conv_channels + self.num_heads

    @property
    def state_elements(self) -> int:
        return self.num_heads * self.head_dim * self.state_dim


@dataclass(frozen=True)
class MoeArchConfig:
    num_experts: int
    experts_per_token: int
    intermediate_size: int
    shared_experts: int
    shared_intermediate_size: int


@dataclass(frozen=True)
class ModelArchConfig:
    hidden_size: int
    inter_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int
    rope_theta: float
    rms_norm_eps: float
    vocab_size: int
    model_type: str
    layer_pattern: str
    mamba: MambaArchConfig
    moe: MoeArchConfig

    def __post_init__(self) -> None:
        if len(self.layer_pattern) != self.num_layers:
            raise ValueError("hybrid layer pattern length does not match layer count")
        if self.mamba.num_heads % self.mamba.groups:
            raise ValueError("Mamba heads must be divisible by groups")

    @property
    def layer_types(self) -> list[str]:
        names = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}
        unknown = set(self.layer_pattern) - set(names)
        if unknown:
            raise ValueError(f"unsupported hybrid layer markers: {sorted(unknown)}")
        return [names[symbol] for symbol in self.layer_pattern]


def load_nemotron3_arch(path: str | Path) -> ModelArchConfig:
    raw = json.loads(Path(path).read_text())
    arch = ModelArchConfig(
        hidden_size=int(raw["hidden_size"]),
        inter_dim=int(raw["intermediate_size"]),
        num_heads=int(raw["num_attention_heads"]),
        num_kv_heads=int(raw["num_key_value_heads"]),
        head_dim=int(raw["head_dim"]),
        num_layers=int(raw["num_hidden_layers"]),
        rope_theta=float(raw["rope_theta"]),
        rms_norm_eps=float(raw.get("norm_eps", raw["layer_norm_epsilon"])),
        vocab_size=int(raw["vocab_size"]),
        model_type=str(raw["model_type"]),
        layer_pattern=str(raw["hybrid_override_pattern"]),
        mamba=MambaArchConfig(
            num_heads=int(raw["mamba_num_heads"]),
            head_dim=int(raw["mamba_head_dim"]),
            state_dim=int(raw["ssm_state_size"]),
            groups=int(raw["n_groups"]),
            conv_kernel=int(raw["conv_kernel"]),
            chunk_size=int(raw["chunk_size"]),
            # The checkpoint calls this a cache dtype; architecturally it is
            # the persistent recurrent-state precision, not a hardware cache.
            state_dtype=str(raw["mamba_ssm_cache_dtype"]),
        ),
        moe=MoeArchConfig(
            num_experts=int(raw["n_routed_experts"]),
            experts_per_token=int(raw["num_experts_per_tok"]),
            intermediate_size=int(raw["moe_intermediate_size"]),
            shared_experts=int(raw["n_shared_experts"]),
            shared_intermediate_size=int(raw["moe_shared_expert_intermediate_size"]),
        ),
    )
    counts = {kind: arch.layer_types.count(kind) for kind in set(arch.layer_types)}
    if counts != {"mamba": 23, "moe": 23, "attention": 6}:
        raise ValueError(f"unexpected Nemotron layer census: {counts}")
    if arch.mamba.projection_size != 10_304:
        raise ValueError("Nemotron Mamba projection width must be 10,304")
    return arch


__all__ = [
    "MambaArchConfig",
    "ModelArchConfig",
    "MoeArchConfig",
    "load_nemotron3_arch",
]
