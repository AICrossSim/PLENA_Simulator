"""
PLENA Memory Analytic Model.

Provides per-layer memory footprint, bandwidth, and traffic analysis.
This module models:
- KV-cache footprint (per-layer and total)
- HBM memory traffic (reads and writes)
- Memory bandwidth utilization
- Weight memory footprint
- Activation memory footprint
"""

import math
from dataclasses import dataclass, field
from enum import Enum

import toml
from pydantic import BaseModel, Field, model_validator


# =============================================================================
# Data Type Definitions
# =============================================================================


class QuantFormat(Enum):
    """Quantization format types."""

    PLAIN = "Plain"  # Standard format (FP16, BF16, FP32, INT8, etc.)
    MX = "Mx"  # Microscaling format (MXFP4, MXFP8, etc.)


@dataclass
class DataTypeSpec:
    """Specification for a data type."""

    format: QuantFormat
    bits_per_element: float  # Average bits per element (including scale overhead for MX)

    @classmethod
    def from_toml_config(cls, config: dict) -> "DataTypeSpec":
        """Create DataTypeSpec from TOML precision config."""
        fmt = QuantFormat(config.get("format", "Plain"))

        if fmt == QuantFormat.PLAIN:
            # Plain format: compute bits from type specification
            data_type = config.get("DATA_TYPE", {})
            if data_type.get("type") == "Fp":
                # FP: sign + exponent + mantissa
                bits = 1 + data_type.get("exponent", 8) + data_type.get("mantissa", 7)
            elif data_type.get("type") == "Int":
                bits = data_type.get("width", 32)
            else:
                bits = 16  # Default to 16-bit
            return cls(format=fmt, bits_per_element=bits)

        elif fmt == QuantFormat.MX:
            # MX format: element bits + scale bits amortized over block
            block_size = config.get("block", 8)
            elem_config = config.get("ELEM", {})
            scale_config = config.get("SCALE", {})

            # Element bits
            if elem_config.get("type") == "Fp":
                elem_bits = 1 + elem_config.get("exponent", 4) + elem_config.get("mantissa", 3)
            else:
                elem_bits = 8

            # Scale bits (amortized over block)
            if scale_config.get("type") == "Fp":
                scale_bits = (
                    (0 if not scale_config.get("sign", False) else 1)
                    + scale_config.get("exponent", 8)
                    + scale_config.get("mantissa", 0)
                )
            else:
                scale_bits = 8

            # Average bits = element_bits + scale_bits/block_size
            avg_bits = elem_bits + scale_bits / block_size
            return cls(format=fmt, bits_per_element=avg_bits)

        return cls(format=QuantFormat.PLAIN, bits_per_element=16)


# =============================================================================
# Memory Configuration Schema
# =============================================================================


class MemoryConfig(BaseModel):
    """Validated memory configuration for PLENA accelerator."""

    # HBM configuration
    HBM_SIZE: int = Field(gt=0, description="HBM capacity in bytes")
    HBM_WIDTH: int = Field(gt=0, description="HBM bus width in bits")

    # SRAM configuration
    MATRIX_SRAM_SIZE: int = Field(gt=0, description="Matrix SRAM size in elements")
    VECTOR_SRAM_SIZE: int = Field(gt=0, description="Vector SRAM size in elements")

    # Hardware dimensions (needed for memory calculations)
    MLEN: int = Field(gt=0, description="Matrix unit length")
    BLEN: int = Field(gt=0, description="Block length")
    VLEN: int = Field(gt=0, description="Vector length")
    HLEN: int = Field(gt=0, description="Head dimension length")

    # Data type specifications (bits per element)
    weight_bits: float = Field(default=8.0, description="Bits per weight element")
    kv_cache_bits: float = Field(default=8.0, description="Bits per KV cache element")
    activation_bits: float = Field(default=16.0, description="Bits per activation element")

    # Allow extra fields
    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def validate_dimensions(self) -> "MemoryConfig":
        """Validate memory configuration relationships."""
        if self.MLEN % self.BLEN != 0:
            raise ValueError(f"MLEN ({self.MLEN}) must be divisible by BLEN ({self.BLEN})")
        return self


def load_memory_config_from_toml(toml_path: str) -> MemoryConfig:
    """
    Load memory configuration from plena_settings.toml.

    Args:
        toml_path: Path to the TOML configuration file

    Returns:
        MemoryConfig: Validated memory configuration
    """
    with open(toml_path) as f:
        data = toml.load(f)

    config_dict = {}
    analytic_data = data.get("ANALYTIC", {})

    # Extract CONFIG section values
    config_section = analytic_data.get("CONFIG", {})
    for param_name, val in config_section.items():
        if isinstance(val, dict) and "value" in val:
            config_dict[param_name] = val["value"]

    # Extract precision specifications
    precision_section = analytic_data.get("PRECISION", {})

    # Weight precision (from HBM_M_WEIGHT_TYPE)
    if "HBM_M_WEIGHT_TYPE" in precision_section:
        weight_spec = DataTypeSpec.from_toml_config(precision_section["HBM_M_WEIGHT_TYPE"])
        config_dict["weight_bits"] = weight_spec.bits_per_element

    # KV cache precision (from HBM_V_KV_TYPE or HBM_M_KV_TYPE)
    if "HBM_V_KV_TYPE" in precision_section:
        kv_spec = DataTypeSpec.from_toml_config(precision_section["HBM_V_KV_TYPE"])
        config_dict["kv_cache_bits"] = kv_spec.bits_per_element
    elif "HBM_M_KV_TYPE" in precision_section:
        kv_spec = DataTypeSpec.from_toml_config(precision_section["HBM_M_KV_TYPE"])
        config_dict["kv_cache_bits"] = kv_spec.bits_per_element

    # Activation precision (from HBM_V_ACT_TYPE)
    if "HBM_V_ACT_TYPE" in precision_section:
        act_spec = DataTypeSpec.from_toml_config(precision_section["HBM_V_ACT_TYPE"])
        config_dict["activation_bits"] = act_spec.bits_per_element

    return MemoryConfig(**config_dict)


# =============================================================================
# Memory Traffic Result Classes
# =============================================================================


@dataclass
class MemoryTraffic:
    """Memory traffic for a single operation."""

    read_bytes: int = 0
    write_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return self.read_bytes + self.write_bytes

    def __add__(self, other: "MemoryTraffic") -> "MemoryTraffic":
        return MemoryTraffic(
            read_bytes=self.read_bytes + other.read_bytes, write_bytes=self.write_bytes + other.write_bytes
        )

    def __iadd__(self, other: "MemoryTraffic") -> "MemoryTraffic":
        self.read_bytes += other.read_bytes
        self.write_bytes += other.write_bytes
        return self

    def __mul__(self, factor: int) -> "MemoryTraffic":
        return MemoryTraffic(read_bytes=self.read_bytes * factor, write_bytes=self.write_bytes * factor)


@dataclass
class KVCacheFootprint:
    """KV cache memory footprint."""

    bytes_per_layer: int = 0
    total_bytes: int = 0


@dataclass
class WeightFootprint:
    """Model weight memory footprint."""

    embedding_bytes: int = 0
    attention_bytes: int = 0  # QKV projections + output projection
    ffn_bytes: int = 0  # Up, gate, down projections
    lm_head_bytes: int = 0
    other_bytes: int = 0  # Layer norms, etc.

    # MoE specific
    router_bytes: int = 0
    expert_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return (
            self.embedding_bytes
            + self.attention_bytes
            + self.ffn_bytes
            + self.lm_head_bytes
            + self.other_bytes
            + self.router_bytes
            + self.expert_bytes
        )


@dataclass
class MemoryFootprint:
    """Complete memory footprint analysis."""

    weights: WeightFootprint = field(default_factory=WeightFootprint)
    kv_cache: KVCacheFootprint = field(default_factory=KVCacheFootprint)

    # Activation memory (peak during inference)
    activation_bytes: int = 0

    # HBM capacity analysis
    hbm_capacity_bytes: int = 0
    total_required_bytes: int = 0
    utilization_ratio: float = 0.0

    @property
    def fits_in_hbm(self) -> bool:
        return self.total_required_bytes <= self.hbm_capacity_bytes


@dataclass
class MemoryUtilization:
    """Memory utilization metrics."""

    # Execution time (from performance model)
    execution_cycles: int = 0
    execution_time_seconds: float = 0.0

    # Off-chip (HBM) metrics
    hbm_traffic: MemoryTraffic = field(default_factory=MemoryTraffic)
    hbm_peak_bandwidth_gbps: float = 0.0
    hbm_achieved_bandwidth_gbps: float = 0.0
    hbm_utilization: float = 0.0  # ratio (0-1)

    # Bottleneck analysis
    is_hbm_bound: bool = False
    arithmetic_intensity: float = 0.0  # FLOPs per byte


@dataclass
class BandwidthAnalysis:
    """Memory bandwidth utilization analysis."""

    # Traffic during operation
    prefill_traffic: MemoryTraffic = field(default_factory=MemoryTraffic)
    decode_traffic: MemoryTraffic = field(default_factory=MemoryTraffic)

    # Per-token analysis (decode phase)
    bytes_per_token: int = 0

    # Bandwidth metrics
    peak_bandwidth_gbps: float = 0.0  # GB/s
    achieved_bandwidth_gbps: float = 0.0  # GB/s
    bandwidth_utilization: float = 0.0  # ratio (0-1)

    # Bottleneck analysis
    is_memory_bound: bool = False
    arithmetic_intensity: float = 0.0  # FLOPs per byte


# =============================================================================
# MemoryModel: Per-Layer Memory Analysis
# =============================================================================


class MemoryModel:
    """
    Per-layer memory footprint and traffic model for PLENA accelerator.

    Computes:
    - Weight memory footprint per layer type
    - KV cache footprint with reduction analysis
    - Memory traffic (read/write bytes) per operation
    - Bandwidth utilization estimation
    """

    def __init__(self, memory_config: MemoryConfig):
        """
        Initialize MemoryModel.

        Args:
            memory_config: Validated memory configuration
        """
        self.config = memory_config
        self.mlen = memory_config.MLEN
        self.blen = memory_config.BLEN
        self.vlen = memory_config.VLEN
        self.hlen = memory_config.HLEN

        self.hbm_size = memory_config.HBM_SIZE
        self.hbm_width_bytes = memory_config.HBM_WIDTH // 8

        self.weight_bits = memory_config.weight_bits
        self.kv_cache_bits = memory_config.kv_cache_bits
        self.activation_bits = memory_config.activation_bits

        self.vector_sram_bytes = memory_config.VECTOR_SRAM_SIZE * self.vlen * (self.activation_bits / 8)
        self.matrix_sram_bytes = memory_config.MATRIX_SRAM_SIZE * self.mlen * self.mlen * (self.weight_bits / 8)
        # Alias for decode batch size calculations
        self.vector_sram_size = self.vector_sram_bytes

    def _bits_to_bytes(self, num_elements: int, bits_per_element: float) -> int:
        """Convert element count to bytes."""
        return math.ceil(num_elements * bits_per_element / 8)

    # -------------------------------------------------------------------------
    # Weight Memory Footprint Methods
    # -------------------------------------------------------------------------

    def embedding_weights(self, vocab_size: int, hidden_size: int) -> int:
        """Embedding layer weight bytes."""
        return self._bits_to_bytes(vocab_size * hidden_size, self.weight_bits)

    def qkv_weights(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> int:
        """Q, K, V projection weight bytes."""
        q_params = hidden_size * num_attention_heads * head_dim
        k_params = hidden_size * num_kv_heads * head_dim
        v_params = hidden_size * num_kv_heads * head_dim
        return self._bits_to_bytes(q_params + k_params + v_params, self.weight_bits)

    def output_projection_weights(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
    ) -> int:
        """Output projection (O) weight bytes."""
        o_params = num_attention_heads * head_dim * hidden_size
        return self._bits_to_bytes(o_params, self.weight_bits)

    def attention_weights(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> int:
        """Attention layer weight bytes (Q, K, V, O projections)."""
        return self.qkv_weights(
            hidden_size, num_attention_heads, num_kv_heads, head_dim
        ) + self.output_projection_weights(hidden_size, num_attention_heads, head_dim)

    def ffn_weights(self, hidden_size: int, intermediate_size: int) -> int:
        """
        Feed-forward layer weight bytes (gate, up, down projections).

        Gate: hidden_size -> intermediate_size
        Up: hidden_size -> intermediate_size
        Down: intermediate_size -> hidden_size
        """
        gate_params = hidden_size * intermediate_size
        up_params = hidden_size * intermediate_size
        down_params = intermediate_size * hidden_size

        total_params = gate_params + up_params + down_params
        return self._bits_to_bytes(total_params, self.weight_bits)

    def moe_router_weights(self, hidden_size: int, num_experts: int) -> int:
        """MoE router weight bytes."""
        return self._bits_to_bytes(hidden_size * num_experts, self.weight_bits)

    def moe_expert_weights(self, hidden_size: int, intermediate_size: int, num_experts: int) -> int:
        """MoE expert weights (all experts)."""
        expert_params_each = 3 * hidden_size * intermediate_size
        return self._bits_to_bytes(expert_params_each * num_experts, self.weight_bits)

    def moe_weights(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
    ) -> tuple[int, int]:
        """MoE layer weight bytes. Returns: (router_bytes, total_expert_bytes)."""
        return (
            self.moe_router_weights(hidden_size, num_experts),
            self.moe_expert_weights(hidden_size, intermediate_size, num_experts),
        )

    def layer_norm_weights(self, hidden_size: int) -> int:
        """Layer norm weight bytes (scale + bias, or just scale for RMSNorm)."""
        # RMSNorm typically just has scale (gamma)
        return self._bits_to_bytes(hidden_size, self.weight_bits)

    def lm_head_weights(self, hidden_size: int, vocab_size: int, tie_embeddings: bool = False) -> int:
        """LM head weight bytes."""
        if tie_embeddings:
            return 0  # Weights shared with embedding
        return self._bits_to_bytes(hidden_size * vocab_size, self.weight_bits)

    # -------------------------------------------------------------------------
    # KV Cache Footprint Methods
    # -------------------------------------------------------------------------

    def kv_cache_per_token(
        self,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
    ) -> int:
        """
        KV cache bytes per token (summed over all layers).

        Per layer: 2 * num_kv_heads * head_dim (K and V)
        """
        elements_per_layer = 2 * num_kv_heads * head_dim
        total_elements = elements_per_layer * num_layers
        return self._bits_to_bytes(total_elements, self.kv_cache_bits)

    def kv_cache_footprint(
        self,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        seq_len: int,
        batch_size: int,
    ) -> KVCacheFootprint:
        """
        Compute KV cache footprint.

        KV cache size = 2 * num_kv_heads * head_dim * seq_len * batch_size * num_layers * (bits/8)
        """
        elements_per_layer = 2 * num_kv_heads * head_dim * seq_len * batch_size
        total_elements = elements_per_layer * num_layers

        return KVCacheFootprint(
            bytes_per_layer=self._bits_to_bytes(elements_per_layer, self.kv_cache_bits),
            total_bytes=self._bits_to_bytes(total_elements, self.kv_cache_bits),
        )

    # -------------------------------------------------------------------------
    # HBM (Off-Chip) Memory Traffic Methods
    # -------------------------------------------------------------------------
    # HBM traffic only includes: weights (read) and KV cache (read/write)
    # Activations are stored on-chip (SRAM) and don't contribute to HBM traffic

    def embedding_traffic(
        self,
        hidden_size: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> MemoryTraffic:
        """Embedding layer HBM traffic (weight reads only)."""
        if mode == "prefill":
            num_tokens = seq_len * batch_size
            act_bytes = self._bits_to_bytes(num_tokens * hidden_size, self.activation_bits)
            return MemoryTraffic(read_bytes=act_bytes, write_bytes=act_bytes)
        else:
            # Stored everything on-chip
            return MemoryTraffic(read_bytes=0, write_bytes=0)

    def projection_traffic(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> MemoryTraffic:
        """QKV projection HBM traffic (weights + KV cache write)."""
        num_tokens = seq_len * batch_size

        # HBM Read: Q, K, V weights (reuse footprint method)
        weight_bytes = self.qkv_weights(hidden_size, num_attention_heads, num_kv_heads, head_dim)
        # HBM Write: KV cache (K and V projections stored to HBM)
        kv_write_bytes = self._bits_to_bytes(num_tokens * num_kv_heads * head_dim * 2, self.kv_cache_bits)

        if mode == "prefill":
            act_bytes = self._bits_to_bytes(num_tokens * hidden_size, self.activation_bits)
            return MemoryTraffic(read_bytes=act_bytes + weight_bytes, write_bytes=act_bytes + kv_write_bytes)
        else:
            # Stored everything on-chip
            return MemoryTraffic(read_bytes=weight_bytes, write_bytes=kv_write_bytes)

    def attention_traffic(
        self,
        _num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        _seq_len: int,
        kv_size: int,
        batch_size: int,
        _mode: str = "prefill",
    ) -> MemoryTraffic:
        """Self-attention HBM traffic (KV cache reads only, Q and output are on-chip)."""
        read_bytes = 0
        write_bytes = 0
        decode_max_batch_size = self.vector_sram_size // (_num_attention_heads * head_dim * (self.activation_bits / 8))
        if _mode == "prefill" or (batch_size >= decode_max_batch_size and _mode == "decode"):
            # QKT
            q_read_bytes = self._bits_to_bytes(
                _seq_len * batch_size * _num_attention_heads * head_dim, self.activation_bits
            )
            kt_read_bytes = self._bits_to_bytes(kv_size * batch_size * num_kv_heads * head_dim, self.kv_cache_bits)
            read_bytes = q_read_bytes + kt_read_bytes
            s_bytes = self._bits_to_bytes(_seq_len * kv_size * batch_size * _num_attention_heads, self.activation_bits)
            write_bytes += s_bytes
            # Softmax
            read_bytes += s_bytes * 2
            write_bytes += s_bytes * 2
            # PV
            v_read_bytes = self._bits_to_bytes(batch_size * kv_size * num_kv_heads * head_dim, self.kv_cache_bits)
            pv_write_bytes = self._bits_to_bytes(
                _seq_len * batch_size * _num_attention_heads * head_dim, self.activation_bits
            )
            read_bytes += v_read_bytes + s_bytes
            write_bytes += pv_write_bytes
        else:
            # Stored activations on-chip
            # QKT
            kt_read_bytes = self._bits_to_bytes(kv_size * batch_size * num_kv_heads * head_dim, self.kv_cache_bits)
            read_bytes += kt_read_bytes
            # PV
            v_read_bytes = self._bits_to_bytes(batch_size * kv_size * num_kv_heads * head_dim, self.kv_cache_bits)
            pv_write_bytes = self._bits_to_bytes(
                _seq_len * batch_size * _num_attention_heads * head_dim, self.activation_bits
            )
            read_bytes += v_read_bytes
            write_bytes += pv_write_bytes

        return MemoryTraffic(read_bytes=read_bytes, write_bytes=write_bytes)

    def flash_attention_traffic(
        self,
        _num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        _seq_len: int,
        kv_size: int,
        batch_size: int,
        _mode: str = "prefill",
    ) -> MemoryTraffic:
        """Flash-attention HBM traffic (KV cache reads only, Q and output are on-chip)."""
        read_bytes = 0
        write_bytes = 0
        decode_max_batch_size = self.vector_sram_size // (_num_attention_heads * head_dim * (self.activation_bits / 8))
        if _mode == "prefill" or (batch_size >= decode_max_batch_size and _mode == "decode"):
            # QKT
            q_read_bytes = self._bits_to_bytes(
                _seq_len * batch_size * _num_attention_heads * head_dim, self.activation_bits
            )
            kt_read_bytes = self._bits_to_bytes(kv_size * batch_size * num_kv_heads * head_dim, self.kv_cache_bits)
            read_bytes += q_read_bytes + kt_read_bytes
            # PV
            v_read_bytes = self._bits_to_bytes(batch_size * kv_size * num_kv_heads * head_dim, self.kv_cache_bits)
            pv_write_bytes = self._bits_to_bytes(
                _seq_len * batch_size * _num_attention_heads * head_dim, self.activation_bits
            )
            read_bytes += v_read_bytes
            write_bytes += pv_write_bytes
            return MemoryTraffic(read_bytes=read_bytes, write_bytes=write_bytes)
        else:
            # Stored activations on-chip
            # QKT
            kt_read_bytes = self._bits_to_bytes(kv_size * batch_size * num_kv_heads * head_dim, self.kv_cache_bits)
            read_bytes += kt_read_bytes
            # PV
            v_read_bytes = self._bits_to_bytes(batch_size * kv_size * num_kv_heads * head_dim, self.kv_cache_bits)
            pv_write_bytes = self._bits_to_bytes(
                _seq_len * batch_size * _num_attention_heads * head_dim, self.activation_bits
            )
            read_bytes += v_read_bytes
            write_bytes += pv_write_bytes
            return MemoryTraffic(read_bytes=read_bytes, write_bytes=write_bytes)

    def ffn_traffic(
        self,
        hidden_size: int,
        intermediate_size: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> MemoryTraffic:
        """Feed-forward layer HBM traffic (gate/up/down weights only).
        Args:
            hidden_size: The hidden size of the model.
            intermediate_size: The intermediate size of the model.
            seq_len: The sequence length of the input.
            batch_size: The batch size of the input.
            mode: The mode of the input.
        Returns:
            The HBM traffic for the feed-forward layer.
        Assumption: We ignore the case where the intermediate can partially stored on-chip (either all or none)
        """
        read_bytes = 0
        write_bytes = 0
        decode_max_batch_size = self.vector_sram_size // (
            (hidden_size * 2 + intermediate_size * 2) * (self.activation_bits / 8)
        )

        if mode == "prefill" or (batch_size >= decode_max_batch_size and mode == "decode"):
            act_bytes = self._bits_to_bytes(hidden_size * batch_size * seq_len, self.activation_bits)
            intermediate_bytes = self._bits_to_bytes(intermediate_size * batch_size * seq_len, self.activation_bits)
            up_gate_bytes = self._bits_to_bytes(intermediate_size * hidden_size, self.weight_bits)
            down_bytes = self._bits_to_bytes(hidden_size * intermediate_size, self.weight_bits)
            # Need to store the intermedate results offchip.
            write_bytes += intermediate_bytes * 2 + act_bytes
            read_bytes += act_bytes * 2 + up_gate_bytes * 2 + down_bytes + intermediate_bytes
        else:
            # Stored activations on-chip
            intermediate_bytes = self._bits_to_bytes(intermediate_size * hidden_size, self.weight_bits)
            read_bytes += intermediate_bytes * 3

        return MemoryTraffic(read_bytes=read_bytes, write_bytes=write_bytes)

    def sliding_attention_traffic(
        self,
        _num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        _seq_len: int,
        kv_size: int,
        batch_size: int,
        sliding_window_size: int,
        _mode: str = "prefill",
    ) -> MemoryTraffic:
        """Sliding window attention HBM traffic (windowed KV cache reads).

        Similar to flash_attention_traffic but with a limited window size.
        """
        effective_kv_size = min(kv_size, sliding_window_size)
        read_bytes = 0
        write_bytes = 0
        decode_max_batch_size = self.vector_sram_size // (_num_attention_heads * head_dim * (self.activation_bits / 8))

        if _mode == "prefill" or (batch_size >= decode_max_batch_size and _mode == "decode"):
            # QKT
            q_read_bytes = self._bits_to_bytes(
                _seq_len * batch_size * _num_attention_heads * head_dim, self.activation_bits
            )
            kt_read_bytes = self._bits_to_bytes(
                effective_kv_size * batch_size * num_kv_heads * head_dim, self.kv_cache_bits
            )
            read_bytes += q_read_bytes + kt_read_bytes
            # PV
            v_read_bytes = self._bits_to_bytes(
                batch_size * effective_kv_size * num_kv_heads * head_dim, self.kv_cache_bits
            )
            pv_write_bytes = self._bits_to_bytes(
                _seq_len * batch_size * _num_attention_heads * head_dim, self.activation_bits
            )
            read_bytes += v_read_bytes
            write_bytes += pv_write_bytes
            return MemoryTraffic(read_bytes=read_bytes, write_bytes=write_bytes)
        else:
            # Stored activations on-chip
            # QKT
            kt_read_bytes = self._bits_to_bytes(
                effective_kv_size * batch_size * num_kv_heads * head_dim, self.kv_cache_bits
            )
            read_bytes += kt_read_bytes
            # PV
            v_read_bytes = self._bits_to_bytes(
                batch_size * effective_kv_size * num_kv_heads * head_dim, self.kv_cache_bits
            )
            read_bytes += v_read_bytes
            return MemoryTraffic(read_bytes=read_bytes, write_bytes=write_bytes)

    def output_projection_traffic(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> MemoryTraffic:
        """Output projection HBM traffic (O weight matrix + activations).

        Args:
            hidden_size: The hidden size of the model.
            num_attention_heads: Number of attention heads.
            head_dim: Dimension per head.
            seq_len: The sequence length of the input.
            batch_size: The batch size of the input.
            mode: The mode of the input (prefill or decode).
        Returns:
            The HBM traffic for the output projection.
        """
        read_bytes = 0
        write_bytes = 0
        # Output projection input: num_attention_heads * head_dim, output: hidden_size
        input_size = num_attention_heads * head_dim
        decode_max_batch_size = self.vector_sram_size // ((input_size + hidden_size) * (self.activation_bits / 8))

        # HBM Read: O weights
        weight_bytes = self.output_projection_weights(hidden_size, num_attention_heads, head_dim)

        if mode == "prefill" or (batch_size >= decode_max_batch_size and mode == "decode"):
            num_tokens = seq_len * batch_size
            input_act_bytes = self._bits_to_bytes(num_tokens * input_size, self.activation_bits)
            output_act_bytes = self._bits_to_bytes(num_tokens * hidden_size, self.activation_bits)
            read_bytes = weight_bytes + input_act_bytes
            write_bytes = output_act_bytes
        else:
            # Stored activations on-chip, only need weight reads
            read_bytes = weight_bytes

        return MemoryTraffic(read_bytes=read_bytes, write_bytes=write_bytes)

    def moe_traffic(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        experts_per_token: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> MemoryTraffic:
        """MoE layer HBM traffic (router + activated expert weights + activations).

        Args:
            hidden_size: The hidden size of the model.
            intermediate_size: The intermediate size per expert.
            num_experts: Total number of experts.
            experts_per_token: Number of experts activated per token.
            seq_len: The sequence length of the input.
            batch_size: The batch size of the input.
            mode: The mode of the input (prefill or decode).
        Returns:
            The HBM traffic for the MoE layer.
        Assumption: Similar to FFN, we ignore the case where intermediate can partially stored on-chip.
        """
        read_bytes = 0
        write_bytes = 0
        # Similar to FFN: input hidden + intermediate (for gelu) + output hidden
        decode_max_batch_size = self.vector_sram_size // (
            (hidden_size * 2 + intermediate_size * 2) * (self.activation_bits / 8)
        )

        # Router weights (always read)
        router_bytes = self.moe_router_weights(hidden_size, num_experts)

        # Expert weights for activated experts only
        # Each expert: gate + up + down = 3 * hidden * intermediate
        expert_weight_per = self._bits_to_bytes(3 * hidden_size * intermediate_size, self.weight_bits)
        expert_weights_bytes = expert_weight_per * experts_per_token

        if mode == "prefill" or (batch_size >= decode_max_batch_size and mode == "decode"):
            num_tokens = seq_len * batch_size
            act_bytes = self._bits_to_bytes(num_tokens * hidden_size, self.activation_bits)
            intermediate_bytes = self._bits_to_bytes(num_tokens * intermediate_size, self.activation_bits)

            # Read: router weights + expert weights + input activations (for each expert)
            # Write: intermediate results + output activations
            # Note: For MoE, tokens are routed to different experts, so we multiply by experts_per_token
            read_bytes = router_bytes + expert_weights_bytes + act_bytes * experts_per_token
            write_bytes = (intermediate_bytes * 2 + act_bytes) * experts_per_token
            # Also need to read intermediate for down projection
            read_bytes += intermediate_bytes * experts_per_token
        else:
            # Stored activations on-chip, only need weight reads
            read_bytes = router_bytes + expert_weights_bytes

        return MemoryTraffic(read_bytes=read_bytes, write_bytes=write_bytes)

    def lm_head_traffic(self, hidden_size: int, vocab_size: int) -> MemoryTraffic:
        """LM head HBM traffic (weight reads only)."""
        # HBM Read: LM head weights (reuse footprint method)
        read_bytes = self.lm_head_weights(hidden_size, vocab_size, tie_embeddings=False)
        return MemoryTraffic(read_bytes=read_bytes, write_bytes=0)

    # -------------------------------------------------------------------------
    # Bandwidth Analysis Methods
    # -------------------------------------------------------------------------

    def compute_peak_bandwidth(self, frequency_hz: float = 1e9) -> float:
        """
        Compute peak HBM bandwidth in GB/s.

        Args:
            frequency_hz: Accelerator clock frequency

        Returns:
            Peak bandwidth in GB/s
        """
        # HBM bandwidth = bus_width * frequency (simplified)
        # Typical HBM2e: 3.2 GT/s per pin, 1024-bit bus = 409.6 GB/s
        # For PLENA, estimate based on HBM_WIDTH
        bytes_per_cycle = self.hbm_width_bytes
        bandwidth_bytes_per_sec = bytes_per_cycle * frequency_hz
        return bandwidth_bytes_per_sec / 1e9  # GB/s

    def compute_bandwidth_utilization(
        self,
        traffic: MemoryTraffic,
        execution_cycles: int,
        frequency_hz: float = 1e9,
    ) -> BandwidthAnalysis:
        """
        Compute memory bandwidth utilization.

        Args:
            traffic: Memory traffic for the operation
            execution_cycles: Number of cycles for the operation
            frequency_hz: Clock frequency

        Returns:
            BandwidthAnalysis with utilization metrics
        """
        result = BandwidthAnalysis()

        execution_time_sec = execution_cycles / frequency_hz
        result.peak_bandwidth_gbps = self.compute_peak_bandwidth(frequency_hz)

        if execution_time_sec > 0:
            result.achieved_bandwidth_gbps = traffic.total_bytes / execution_time_sec / 1e9
            result.bandwidth_utilization = result.achieved_bandwidth_gbps / result.peak_bandwidth_gbps

        return result

    def compute_arithmetic_intensity(
        self,
        flops: int,
        traffic: MemoryTraffic,
    ) -> float:
        """
        Compute arithmetic intensity (FLOPs per byte).

        Args:
            flops: Number of floating-point operations
            traffic: Memory traffic

        Returns:
            Arithmetic intensity (FLOPs/byte)
        """
        if traffic.total_bytes > 0:
            return flops / traffic.total_bytes
        return 0.0

    # -------------------------------------------------------------------------
    # Memory Utilization Computation
    # -------------------------------------------------------------------------

    def compute_memory_utilization(
        self,
        hbm_traffic: MemoryTraffic,
        execution_cycles: int,
        frequency_hz: float = 1e9,
    ) -> MemoryUtilization:
        """
        Compute memory utilization given traffic and execution time.

        Args:
            hbm_traffic: Off-chip HBM traffic
            execution_cycles: Total execution cycles from performance model
            frequency_hz: Clock frequency

        Returns:
            MemoryUtilization with all metrics
        """
        result = MemoryUtilization()

        result.execution_cycles = execution_cycles
        result.execution_time_seconds = execution_cycles / frequency_hz

        # HBM metrics
        result.hbm_traffic = hbm_traffic
        result.hbm_peak_bandwidth_gbps = self.compute_peak_bandwidth(frequency_hz)

        if result.execution_time_seconds > 0:
            result.hbm_achieved_bandwidth_gbps = hbm_traffic.total_bytes / result.execution_time_seconds / 1e9
            result.hbm_utilization = result.hbm_achieved_bandwidth_gbps / result.hbm_peak_bandwidth_gbps

        # Bottleneck analysis
        result.is_hbm_bound = result.hbm_utilization > 0.7

        return result
