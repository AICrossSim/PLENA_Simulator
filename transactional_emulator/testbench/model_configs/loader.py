"""Model-aware YAML config loader with hardware assertions.

Loads per-model YAML configurations and auto-detects model architectures
from HuggingFace configs. Validates hardware constraints (hlen >= head_dim,
broadcast >= GQA ratio, hlen * broadcast == MLEN).
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_CONFIG_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Model config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MambaArchConfig:
    num_heads: int
    head_dim: int
    state_dim: int
    groups: int
    conv_kernel: int
    chunk_size: int
    cache_dtype: str = "float32"

    def __post_init__(self) -> None:
        for name in ("num_heads", "head_dim", "state_dim", "groups", "conv_kernel", "chunk_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"Mamba {name} must be positive")
        if self.num_heads % self.groups:
            raise ValueError("Mamba num_heads must be divisible by groups")

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
        # Nemotron 3 Nano has d_mlp=0: [gate, x/B/C, dt].
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

    def __post_init__(self) -> None:
        for name in (
            "num_experts",
            "experts_per_token",
            "intermediate_size",
            "shared_intermediate_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"MoE {name} must be positive")
        if self.shared_experts < 0:
            raise ValueError("MoE shared_experts must be non-negative")
        if self.experts_per_token > self.num_experts:
            raise ValueError("MoE experts_per_token cannot exceed num_experts")


@dataclass
class ModelArchConfig:
    hidden_size: int
    inter_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int
    rope_theta: float
    rms_norm_eps: float
    vocab_size: int | None = None
    model_type: str = "llama"
    layer_pattern: str | None = None
    mamba: MambaArchConfig | None = None
    moe: MoeArchConfig | None = None

    def __post_init__(self) -> None:
        if self.layer_pattern is None:
            return
        if len(self.layer_pattern) != self.num_layers:
            raise ValueError(f"layer_pattern length ({len(self.layer_pattern)}) != num_layers ({self.num_layers})")
        invalid = sorted(set(self.layer_pattern) - {"M", "E", "*", "-"})
        if invalid:
            raise ValueError(f"unsupported layer_pattern symbols: {invalid}")
        if "M" in self.layer_pattern and self.mamba is None:
            raise ValueError("layer_pattern contains Mamba layers but no Mamba config")
        if "E" in self.layer_pattern and self.moe is None:
            raise ValueError("layer_pattern contains MoE layers but no MoE config")

    @property
    def gqa_ratio(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def layer_types(self) -> list[str]:
        if self.layer_pattern is None:
            return ["attention"] * self.num_layers
        names = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}
        return [names[symbol] for symbol in self.layer_pattern]

    def count_layers(self, layer_type: str) -> int:
        return self.layer_types.count(layer_type)

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> ModelArchConfig:
        """Extract architecture config from a HuggingFace model config."""
        # VLMs wrap their text model under text_config
        cfg = getattr(hf_config, "text_config", hf_config)
        hidden = cfg.hidden_size
        heads = cfg.num_attention_heads

        inter = getattr(cfg, "intermediate_size", None)
        if inter is None:
            inter = getattr(cfg, "mlp_hidden_size", None)
        if inter is None:
            inter = 4 * hidden

        kv_heads = getattr(cfg, "num_key_value_heads", None)
        if kv_heads is None:
            kv_heads = getattr(cfg, "n_kv_heads", heads)

        layer_pattern = getattr(cfg, "hybrid_override_pattern", None)
        mamba = None
        if layer_pattern and "M" in layer_pattern:
            mamba = MambaArchConfig(
                num_heads=int(cfg.mamba_num_heads),
                head_dim=int(cfg.mamba_head_dim),
                state_dim=int(cfg.ssm_state_size),
                groups=int(getattr(cfg, "n_groups", getattr(cfg, "mamba_n_groups", 1))),
                conv_kernel=int(getattr(cfg, "conv_kernel", getattr(cfg, "mamba_d_conv", 4))),
                chunk_size=int(getattr(cfg, "chunk_size", getattr(cfg, "mamba_chunk_size", 128))),
                cache_dtype=str(getattr(cfg, "mamba_ssm_cache_dtype", "float32")),
            )

        moe = None
        if layer_pattern and "E" in layer_pattern:
            moe = MoeArchConfig(
                num_experts=int(cfg.n_routed_experts),
                experts_per_token=int(cfg.num_experts_per_tok),
                intermediate_size=int(cfg.moe_intermediate_size),
                shared_experts=int(getattr(cfg, "n_shared_experts", 0)),
                shared_intermediate_size=int(
                    getattr(cfg, "moe_shared_expert_intermediate_size", cfg.moe_intermediate_size)
                ),
            )

        return cls(
            hidden_size=hidden,
            inter_dim=inter,
            num_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=int(getattr(cfg, "head_dim", hidden // heads)),
            num_layers=getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layers", 0)),
            rope_theta=getattr(cfg, "rope_theta", 10000.0),
            rms_norm_eps=getattr(
                cfg,
                "rms_norm_eps",
                getattr(cfg, "layer_norm_epsilon", getattr(cfg, "norm_eps", 1e-5)),
            ),
            vocab_size=getattr(cfg, "vocab_size", None),
            model_type=getattr(cfg, "model_type", "unknown"),
            layer_pattern=layer_pattern,
            mamba=mamba,
            moe=moe,
        )


@dataclass
class HardwarePreset:
    mlen: int = 64
    vlen: int = 64
    blen: int = 4
    batch_size: int = 1
    hlen: int = 64
    broadcast: int = 1
    mram_tile_capacity: int = 4
    mode: str = "native"


@dataclass
class ModelConfig:
    model_id: str
    nickname: str
    trust_remote_code: bool
    family: str
    arch: ModelArchConfig
    hardware: HardwarePreset
    hardware_presets: dict[str, HardwarePreset] = field(default_factory=dict)
    raw: dict = field(default_factory=dict, repr=False)

    def get_preset(self, name: str) -> HardwarePreset:
        if name not in self.hardware_presets:
            raise KeyError(f"Unknown hardware preset '{name}'. Available: {list(self.hardware_presets)}")
        return self.hardware_presets[name]


# ---------------------------------------------------------------------------
# Known models registry
# ---------------------------------------------------------------------------

KNOWN_MODELS = {
    "smolvlm2_256m": "smolvlm2_256m.yaml",  # text decoder (default)
    "smolvlm2_256m_text": "smolvlm2_256m_text.yaml",
    "smollm2_135m": "smollm2_135m.yaml",
    "llada_8b": "llada_8b.yaml",  # Instruct (default)
    "llada_8b_instruct": "llada_8b_instruct.yaml",
    "llada_8b_base": "llada_8b_base.yaml",
    "clm_60m": "clm_60m.yaml",
    "nemotron3_nano_30b_a3b": "nemotron3_nano_30b_a3b.yaml",
}


def load_model_config(model_key: str, model_id_override: str | None = None) -> ModelConfig:
    """Load a known model config by key (e.g. 'llada_8b')."""
    if model_key not in KNOWN_MODELS:
        raise KeyError(f"Unknown model key '{model_key}'. Known: {list(KNOWN_MODELS)}")
    path = _CONFIG_DIR / KNOWN_MODELS[model_key]
    with open(path) as f:
        raw = yaml.safe_load(f)

    if model_id_override:
        raw["model_id"] = model_id_override

    presets = {}
    for name, preset_raw in raw.get("hardware_presets", {}).items():
        presets[name] = HardwarePreset(**preset_raw)

    arch_raw = dict(raw["architecture"]["text"])
    mamba_raw = arch_raw.pop("mamba", None)
    moe_raw = arch_raw.pop("moe", None)
    arch = ModelArchConfig(
        **arch_raw,
        mamba=MambaArchConfig(**mamba_raw) if mamba_raw is not None else None,
        moe=MoeArchConfig(**moe_raw) if moe_raw is not None else None,
    )

    return ModelConfig(
        model_id=raw["model_id"],
        nickname=raw.get("nickname", model_key),
        trust_remote_code=raw.get("trust_remote_code", False),
        family=raw["family"],
        arch=arch,
        hardware=HardwarePreset(**raw["hardware"]),
        hardware_presets=presets,
        raw=raw,
    )


def validate_hardware(arch: ModelArchConfig, hw: HardwarePreset, mlen: int) -> list[str]:
    """Validate hardware config against architecture. Returns list of issues."""
    issues = []

    if hw.hlen < arch.head_dim:
        issues.append(f"hlen={hw.hlen} < head_dim={arch.head_dim}: head slots too small for attention heads")

    expected_mlen = hw.hlen * hw.broadcast
    if expected_mlen != mlen:
        issues.append(f"hlen*broadcast = {hw.hlen}*{hw.broadcast} = {expected_mlen} != MLEN={mlen}")

    if hw.broadcast < arch.gqa_ratio:
        issues.append(
            f"broadcast={hw.broadcast} < GQA ratio={arch.gqa_ratio}: "
            f"insufficient broadcast for {arch.num_heads}/{arch.num_kv_heads} heads"
        )

    return issues


def resolve_hardware(
    arch: ModelArchConfig,
    mlen: int,
    vlen: int | None = None,
    blen: int | None = None,
    batch_size: int = 1,
    mram_tile_capacity: int = 16,
    mode: str = "native",
) -> HardwarePreset:
    """Auto-compute valid hardware config given architecture and MLEN.

    Rules:
      - hlen >= head_dim
      - hlen * broadcast == MLEN
      - broadcast >= GQA ratio
    """
    gqa = arch.gqa_ratio

    for broadcast in range(gqa, mlen + 1):
        if mlen % broadcast == 0:
            hlen = mlen // broadcast
            if hlen >= arch.head_dim:
                return HardwarePreset(
                    mlen=mlen,
                    vlen=vlen if vlen is not None else mlen,
                    blen=blen if blen is not None else 4,
                    batch_size=batch_size,
                    hlen=hlen,
                    broadcast=broadcast,
                    mram_tile_capacity=mram_tile_capacity,
                    mode=mode,
                )

    return HardwarePreset(
        mlen=mlen,
        vlen=vlen if vlen is not None else mlen,
        blen=blen if blen is not None else 4,
        batch_size=batch_size,
        hlen=mlen,
        broadcast=1,
        mram_tile_capacity=mram_tile_capacity,
        mode=mode,
    )


_NICKNAME_MAP: dict[str, str] = {}


def _ensure_nickname_map() -> None:
    if _NICKNAME_MAP:
        return
    for key, filename in KNOWN_MODELS.items():
        path = _CONFIG_DIR / filename
        with open(path) as f:
            raw = yaml.safe_load(f)
        nick = raw.get("nickname")
        if nick and nick not in _NICKNAME_MAP:
            _NICKNAME_MAP[nick] = key


def load_model_config_by_nickname(nickname: str, model_id_override: str | None = None) -> ModelConfig:
    """Load a model config by its nickname (e.g. 'smollm2', 'llada-8b')."""
    _ensure_nickname_map()
    if nickname not in _NICKNAME_MAP:
        raise KeyError(f"Unknown nickname '{nickname}'. Known: {list(_NICKNAME_MAP)}")
    return load_model_config(_NICKNAME_MAP[nickname], model_id_override)


def arch_from_hf(model_id: str, trust_remote_code: bool = False) -> ModelArchConfig:
    """Probe HuggingFace model config and extract architecture (no weight download)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    return ModelArchConfig.from_hf_config(cfg)
