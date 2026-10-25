"""Decoder structure read from a HuggingFace-style config.

Two layer compositions mirror PLENA's single-chip scripts so that a one-device
system reproduces them:

* ``"llama"`` (``analytic_models/performance/llama_model.py``): dense layers of
  RMSNorm, QKV projection, attention, residual, RMSNorm and FFN;
* ``"moe"`` (``analytic_models/performance/gpt_oss_model.py``): per-layer full
  or sliding-window attention, a residual after attention and after the MLP,
  routed experts (or a dense FFN where ``mlp_types`` says so), and the LM head
  in prefill.

A config with routed experts, sliding-window layers or ``model_type ==
"gpt_oss"`` uses ``"moe"``. Shared experts are not modelled.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..stacked_dram._validate import nonnegative_int, positive_int

FAMILIES = ("llama", "moe")
_EXPERT_KEYS = ("num_local_experts", "num_experts", "n_routed_experts")
_SHARED_EXPERT_KEYS = ("n_shared_experts", "num_shared_experts", "shared_expert_intermediate_size")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    intermediate_size: int
    vocab_size: int
    tie_word_embeddings: bool
    num_experts: int
    experts_per_token: int
    moe_intermediate_size: int
    layer_types: tuple[str, ...]
    mlp_types: tuple[str, ...]
    sliding_window: int

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {', '.join(FAMILIES)}")
        for name in (
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "num_hidden_layers",
            "intermediate_size",
            "vocab_size",
            "experts_per_token",
            "moe_intermediate_size",
        ):
            positive_int(getattr(self, name), name)
        nonnegative_int(self.num_experts, "num_experts")
        nonnegative_int(self.sliding_window, "sliding_window")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        for name, allowed in (("layer_types", {"full_attention", "sliding_attention"}), ("mlp_types", {"ffn", "moe"})):
            values = getattr(self, name)
            if len(values) != self.num_hidden_layers:
                raise ValueError(f"{name} has {len(values)} entries for {self.num_hidden_layers} layers")
            if set(values) - allowed:
                raise ValueError(f"{name} entries must be in {sorted(allowed)}")
        if "sliding_attention" in self.layer_types and self.sliding_window <= 0:
            raise ValueError("sliding-window layers need a positive sliding_window")
        if "moe" in self.mlp_types:
            if self.num_experts < 2:
                raise ValueError("MoE layers need at least two routed experts")
            if self.experts_per_token > self.num_experts:
                raise ValueError("experts_per_token exceeds num_experts")
        if self.family == "llama" and ("moe" in self.mlp_types or "sliding_attention" in self.layer_types):
            raise ValueError("the llama composition has dense full-attention layers only")

    @classmethod
    def from_hf_config(cls, config: Mapping[str, Any], *, name: str) -> ModelSpec:
        if any(config.get(key) for key in _SHARED_EXPERT_KEYS):
            raise ValueError(f"{name}: shared experts are not modelled yet")
        experts = max((config.get(key) or 0) for key in _EXPERT_KEYS)
        experts = experts if experts > 1 else 0
        layers = config["num_hidden_layers"]
        hidden = config["hidden_size"]
        heads = config["num_attention_heads"]
        layer_types = tuple(config.get("layer_types") or ["full_attention"] * layers)
        mlp_types = tuple(config.get("mlp_types") or ["moe" if experts else "ffn"] * layers)
        family = (
            "moe" if experts or "sliding_attention" in layer_types or config.get("model_type") == "gpt_oss" else "llama"
        )
        return cls(
            name=name,
            family=family,
            hidden_size=hidden,
            num_attention_heads=heads,
            num_key_value_heads=config.get("num_key_value_heads", heads),
            head_dim=config.get("head_dim") or hidden // heads,
            num_hidden_layers=layers,
            intermediate_size=config["intermediate_size"],
            vocab_size=config["vocab_size"],
            tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
            num_experts=experts,
            experts_per_token=config.get("experts_per_token", config.get("num_experts_per_tok", 1)) if experts else 1,
            moe_intermediate_size=config.get("moe_intermediate_size") or config["intermediate_size"],
            layer_types=layer_types,
            mlp_types=mlp_types,
            sliding_window=config.get("sliding_window") or 0,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ModelSpec:
        config_path = Path(path)
        with config_path.open() as handle:
            return cls.from_hf_config(json.load(handle), name=config_path.stem)

    @property
    def has_moe(self) -> bool:
        return "moe" in self.mlp_types

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "heads": self.num_attention_heads,
            "kv_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "moe_layers": sum(kind == "moe" for kind in self.mlp_types),
            "num_experts": self.num_experts,
            "experts_per_token": self.experts_per_token,
            "sliding_layers": sum(kind == "sliding_attention" for kind in self.layer_types),
        }
