"""Model-aware YAML config loader with hardware assertions.

Loads per-model YAML configurations and auto-detects model architectures
from HuggingFace configs. Validates hardware constraints (hlen >= head_dim,
broadcast >= GQA ratio, hlen * broadcast == MLEN).
"""

from __future__ import annotations

import math
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_CONFIG_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Model config dataclass
# ---------------------------------------------------------------------------


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

    @property
    def gqa_ratio(self) -> int:
        return self.num_heads // self.num_kv_heads

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

        return cls(
            hidden_size=hidden,
            inter_dim=inter,
            num_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=hidden // heads,
            num_layers=getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layers", 0)),
            rope_theta=getattr(cfg, "rope_theta", 10000.0),
            rms_norm_eps=getattr(cfg, "rms_norm_eps", 1e-5),
            vocab_size=getattr(cfg, "vocab_size", None),
            model_type=getattr(cfg, "model_type", "unknown"),
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

    return ModelConfig(
        model_id=raw["model_id"],
        nickname=raw.get("nickname", model_key),
        trust_remote_code=raw.get("trust_remote_code", False),
        family=raw["family"],
        arch=ModelArchConfig(**raw["architecture"]["text"]),
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


def _projection_k_dims(mc: "ModelConfig") -> list[int]:
    """Contraction (K) dimensions of every K-tiled projection in the network.

    A projection ``out = act @ weight`` over ``K`` input columns is tiled into
    ``ceil(K / mlen)`` MLEN-tiles. The K-tiled ops are the attention QKVO
    projections (K = hidden_size) and the FFN up/gate (K = hidden_size) and down
    (K = inter_dim) projections; the down-projection's ``inter_dim`` is the
    largest K in a transformer block. For a VLM both the text decoder and the
    vision encoder contribute, so the network-wide maximum spans both towers.
    """
    dims = [mc.arch.hidden_size, mc.arch.inter_dim]
    vision = (mc.raw.get("architecture") or {}).get("vision") or {}
    for key in ("hidden_size", "inter_dim"):
        v = vision.get(key)
        if v:
            dims.append(int(v))
    return [d for d in dims if d]


def recommended_mram_tile_capacity(mc: "ModelConfig", mlen: int) -> int:
    """Smallest matrix-SRAM tile capacity that keeps every projection single-pass.

    When a projection's ``ceil(K / mlen)`` tiles exceed the resident matrix-SRAM
    pool (``mram_tile_capacity``), the compiler must K-split it: the contraction
    is broken into chunks whose partial sums are accumulated back through VRAM
    with extra full-output ``V_ADD_VV`` passes (``ffn_asm.py`` K-split path). A
    capacity ``>= max_K ceil(K / mlen)`` makes every projection fit in one pass
    and eliminates that overhead. The binding op is the FFN down-projection
    (K = inter_dim); for a VLM, the larger of the text and vision towers wins.
    """
    return max(math.ceil(k / mlen) for k in _projection_k_dims(mc))


def recommended_matrix_sram_depth(mc: "ModelConfig") -> int:
    """MLEN-invariant matrix-SRAM depth (rows) that keeps every projection single-pass.

    Equals the largest projection contraction dimension (the FFN down-projection's
    inter_dim). A matrix SRAM at least this deep holds the full K of every projection
    at any MLEN, so the compiler never K-splits; the per-MLEN tile cap is then
    mram_tile_capacity = matrix_sram_depth // MLEN.
    """
    return max(_projection_k_dims(mc))


def ksplit_advisory(mc: "ModelConfig", mlen: int, mram_tile_capacity: int) -> str | None:
    """Human-readable advisory comparing the active capacity to the K-split-free
    minimum. Returns ``None`` when the capacity already eliminates the K-split.

    K-split is numerically identical to single-pass (it only adds VRAM
    accumulation traffic), so this is advisory, not a hard error.
    """
    rec = recommended_mram_tile_capacity(mc, mlen)
    if mram_tile_capacity >= rec:
        return None
    return (
        f"mram_tile_capacity={mram_tile_capacity} forces an FFN/projection K-split "
        f"at mlen={mlen}: the largest projection needs {rec} resident matrix-SRAM "
        f"tiles (K={max(_projection_k_dims(mc))}). Raise the board's matrix_sram_depth to "
        f">= {recommended_matrix_sram_depth(mc)} rows (>= {rec} tiles at MLEN={mlen}) to make "
        f"every projection single-pass."
    )


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
