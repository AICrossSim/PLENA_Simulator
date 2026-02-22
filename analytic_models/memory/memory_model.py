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
from typing import Optional

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
                scale_bits = (0 if not scale_config.get("sign", False) else 1) + \
                            scale_config.get("exponent", 8) + scale_config.get("mantissa", 0)
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

    # Prefetch/writeback configuration
    HBM_M_Prefetch_Amount: int = Field(gt=0, description="Matrix prefetch amount")
    HBM_V_Prefetch_Amount: int = Field(gt=0, description="Vector prefetch amount")
    HBM_V_Writeback_Amount: int = Field(gt=0, description="Vector writeback amount")

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
            read_bytes=self.read_bytes + other.read_bytes,
            write_bytes=self.write_bytes + other.write_bytes
        )

    def __iadd__(self, other: "MemoryTraffic") -> "MemoryTraffic":
        self.read_bytes += other.read_bytes
        self.write_bytes += other.write_bytes
        return self

    def __mul__(self, factor: int) -> "MemoryTraffic":
        return MemoryTraffic(
            read_bytes=self.read_bytes * factor,
            write_bytes=self.write_bytes * factor
        )


@dataclass
class KVCacheFootprint:
    """KV cache memory footprint analysis."""
    # Per-layer footprint
    k_cache_bytes_per_layer: int = 0
    v_cache_bytes_per_layer: int = 0

    # Total footprint
    total_k_cache_bytes: int = 0
    total_v_cache_bytes: int = 0

    # Reduction analysis
    baseline_bytes: int = 0  # Without GQA, sliding window, etc.
    actual_bytes: int = 0
    reduction_ratio: float = 1.0

    # Breakdown by technique
    gqa_reduction_bytes: int = 0  # Savings from grouped-query attention
    sliding_window_reduction_bytes: int = 0  # Savings from sliding window
    quantization_reduction_bytes: int = 0  # Savings from KV cache quantization

    @property
    def total_per_layer_bytes(self) -> int:
        return self.k_cache_bytes_per_layer + self.v_cache_bytes_per_layer

    @property
    def total_bytes(self) -> int:
        return self.total_k_cache_bytes + self.total_v_cache_bytes


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
        return (self.embedding_bytes + self.attention_bytes + self.ffn_bytes +
                self.lm_head_bytes + self.other_bytes + self.router_bytes + self.expert_bytes)


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
class OnChipMemoryTraffic:
    """On-chip SRAM memory traffic."""
    # Matrix SRAM traffic (weights, intermediate results)
    matrix_sram_read_bytes: int = 0
    matrix_sram_write_bytes: int = 0

    # Vector SRAM traffic (activations, KV cache tiles)
    vector_sram_read_bytes: int = 0
    vector_sram_write_bytes: int = 0

    @property
    def total_matrix_sram_bytes(self) -> int:
        return self.matrix_sram_read_bytes + self.matrix_sram_write_bytes

    @property
    def total_vector_sram_bytes(self) -> int:
        return self.vector_sram_read_bytes + self.vector_sram_write_bytes

    @property
    def total_bytes(self) -> int:
        return self.total_matrix_sram_bytes + self.total_vector_sram_bytes

    def __add__(self, other: "OnChipMemoryTraffic") -> "OnChipMemoryTraffic":
        return OnChipMemoryTraffic(
            matrix_sram_read_bytes=self.matrix_sram_read_bytes + other.matrix_sram_read_bytes,
            matrix_sram_write_bytes=self.matrix_sram_write_bytes + other.matrix_sram_write_bytes,
            vector_sram_read_bytes=self.vector_sram_read_bytes + other.vector_sram_read_bytes,
            vector_sram_write_bytes=self.vector_sram_write_bytes + other.vector_sram_write_bytes,
        )

    def __iadd__(self, other: "OnChipMemoryTraffic") -> "OnChipMemoryTraffic":
        self.matrix_sram_read_bytes += other.matrix_sram_read_bytes
        self.matrix_sram_write_bytes += other.matrix_sram_write_bytes
        self.vector_sram_read_bytes += other.vector_sram_read_bytes
        self.vector_sram_write_bytes += other.vector_sram_write_bytes
        return self


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

    # On-chip (SRAM) metrics
    onchip_traffic: OnChipMemoryTraffic = field(default_factory=OnChipMemoryTraffic)
    matrix_sram_capacity_bytes: int = 0
    vector_sram_capacity_bytes: int = 0
    matrix_sram_peak_bandwidth_gbps: float = 0.0
    vector_sram_peak_bandwidth_gbps: float = 0.0
    matrix_sram_achieved_bandwidth_gbps: float = 0.0
    vector_sram_achieved_bandwidth_gbps: float = 0.0
    matrix_sram_utilization: float = 0.0
    vector_sram_utilization: float = 0.0

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

        self.vector_sram_bytes = memory_config.VECTOR_SRAM_SIZE * self.vlen * 2  # 2 bytes per element
        self.matrix_sram_bytes = memory_config.MATRIX_SRAM_SIZE * self.mlen * self.mlen * 2

        self.weight_bits = memory_config.weight_bits
        self.kv_cache_bits = memory_config.kv_cache_bits
        self.activation_bits = memory_config.activation_bits

    def _bits_to_bytes(self, num_elements: int, bits_per_element: float) -> int:
        """Convert element count to bytes."""
        return math.ceil(num_elements * bits_per_element / 8)

    # -------------------------------------------------------------------------
    # Weight Memory Footprint Methods
    # -------------------------------------------------------------------------

    def embedding_weights(self, vocab_size: int, hidden_size: int) -> int:
        """Embedding layer weight bytes."""
        num_params = vocab_size * hidden_size
        return self._bits_to_bytes(num_params, self.weight_bits)

    def attention_weights(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> int:
        """
        Attention layer weight bytes (Q, K, V, O projections).

        Q projection: hidden_size -> num_attention_heads * head_dim
        K projection: hidden_size -> num_kv_heads * head_dim
        V projection: hidden_size -> num_kv_heads * head_dim
        O projection: num_attention_heads * head_dim -> hidden_size
        """
        q_params = hidden_size * num_attention_heads * head_dim
        k_params = hidden_size * num_kv_heads * head_dim
        v_params = hidden_size * num_kv_heads * head_dim
        o_params = num_attention_heads * head_dim * hidden_size

        total_params = q_params + k_params + v_params + o_params
        return self._bits_to_bytes(total_params, self.weight_bits)

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

    def moe_weights(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
    ) -> tuple[int, int]:
        """
        MoE layer weight bytes.

        Returns:
            tuple: (router_bytes, total_expert_bytes)
        """
        # Router: hidden_size -> num_experts
        router_params = hidden_size * num_experts
        router_bytes = self._bits_to_bytes(router_params, self.weight_bits)

        # Each expert has gate, up, down projections
        expert_params_each = 3 * hidden_size * intermediate_size
        total_expert_params = expert_params_each * num_experts
        expert_bytes = self._bits_to_bytes(total_expert_params, self.weight_bits)

        return router_bytes, expert_bytes

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
        num_attention_heads: int,
        head_dim: int,
        num_layers: int,
        seq_len: int,
        batch_size: int,
        sliding_window_size: Optional[int] = None,
        num_sliding_layers: int = 0,
        baseline_bits: float = 16.0,  # FP16 baseline for reduction calculation
    ) -> KVCacheFootprint:
        """
        Compute KV cache footprint with reduction analysis.

        Args:
            num_kv_heads: Number of KV heads (for GQA)
            num_attention_heads: Number of query heads
            head_dim: Dimension per head
            num_layers: Total number of layers
            seq_len: Sequence length
            batch_size: Batch size
            sliding_window_size: Sliding window size (None for full attention)
            num_sliding_layers: Number of layers using sliding window attention
            baseline_bits: Bits per element for baseline comparison
        """
        result = KVCacheFootprint()

        # Per-layer footprint (K + V for one layer)
        k_elements_per_layer = num_kv_heads * head_dim * seq_len * batch_size
        v_elements_per_layer = num_kv_heads * head_dim * seq_len * batch_size

        result.k_cache_bytes_per_layer = self._bits_to_bytes(k_elements_per_layer, self.kv_cache_bits)
        result.v_cache_bytes_per_layer = self._bits_to_bytes(v_elements_per_layer, self.kv_cache_bits)

        # Full attention layers
        num_full_layers = num_layers - num_sliding_layers
        full_attn_elements = 2 * num_kv_heads * head_dim * seq_len * batch_size * num_full_layers

        # Sliding window layers (capped sequence length)
        if sliding_window_size and num_sliding_layers > 0:
            effective_seq_len = min(seq_len, sliding_window_size)
            sliding_elements = 2 * num_kv_heads * head_dim * effective_seq_len * batch_size * num_sliding_layers
        else:
            sliding_elements = 0

        total_elements = full_attn_elements + sliding_elements
        result.total_k_cache_bytes = self._bits_to_bytes(total_elements // 2, self.kv_cache_bits)
        result.total_v_cache_bytes = self._bits_to_bytes(total_elements // 2, self.kv_cache_bits)
        result.actual_bytes = result.total_bytes

        # Baseline: MHA (num_attention_heads instead of num_kv_heads), full attention, FP16
        baseline_elements = 2 * num_attention_heads * head_dim * seq_len * batch_size * num_layers
        result.baseline_bytes = self._bits_to_bytes(baseline_elements, baseline_bits)

        # GQA reduction
        if num_kv_heads < num_attention_heads:
            mha_elements = 2 * num_attention_heads * head_dim * seq_len * batch_size * num_layers
            gqa_elements = 2 * num_kv_heads * head_dim * seq_len * batch_size * num_layers
            result.gqa_reduction_bytes = self._bits_to_bytes(mha_elements - gqa_elements, self.kv_cache_bits)

        # Sliding window reduction
        if sliding_window_size and num_sliding_layers > 0 and seq_len > sliding_window_size:
            full_sliding_elements = 2 * num_kv_heads * head_dim * seq_len * batch_size * num_sliding_layers
            actual_sliding_elements = sliding_elements
            result.sliding_window_reduction_bytes = self._bits_to_bytes(
                full_sliding_elements - actual_sliding_elements, self.kv_cache_bits
            )

        # Quantization reduction (compared to FP16 baseline)
        if self.kv_cache_bits < baseline_bits:
            current_bytes = self._bits_to_bytes(total_elements, self.kv_cache_bits)
            fp16_bytes = self._bits_to_bytes(total_elements, baseline_bits)
            result.quantization_reduction_bytes = fp16_bytes - current_bytes

        # Overall reduction ratio
        if result.baseline_bytes > 0:
            result.reduction_ratio = result.actual_bytes / result.baseline_bytes

        return result

    # -------------------------------------------------------------------------
    # Memory Traffic Methods
    # -------------------------------------------------------------------------

    def embedding_traffic(
        self,
        vocab_size: int,
        hidden_size: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> MemoryTraffic:
        """Embedding layer memory traffic."""
        if mode == "prefill":
            num_tokens = seq_len * batch_size
        else:
            num_tokens = batch_size

        # Read: embedding vectors for each token
        # Assuming random access, read full embedding per token
        read_bytes = self._bits_to_bytes(num_tokens * hidden_size, self.activation_bits)

        # Write: output activations
        write_bytes = self._bits_to_bytes(num_tokens * hidden_size, self.activation_bits)

        return MemoryTraffic(read_bytes=read_bytes, write_bytes=write_bytes)

    def rms_norm_traffic(
        self,
        hidden_size: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> MemoryTraffic:
        """RMS normalization memory traffic."""
        if mode == "prefill":
            num_tokens = seq_len * batch_size
        else:
            num_tokens = batch_size

        # Read: input activations + norm weights
        read_bytes = self._bits_to_bytes(num_tokens * hidden_size, self.activation_bits)
        read_bytes += self._bits_to_bytes(hidden_size, self.weight_bits)  # scale

        # Write: normalized activations
        write_bytes = self._bits_to_bytes(num_tokens * hidden_size, self.activation_bits)

        return MemoryTraffic(read_bytes=read_bytes, write_bytes=write_bytes)

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
        """QKV projection + KV cache write traffic."""
        if mode == "prefill":
            num_tokens = seq_len * batch_size
        else:
            num_tokens = batch_size

        # Read: input activations + Q, K, V weight matrices
        read_bytes = self._bits_to_bytes(num_tokens * hidden_size, self.activation_bits)
        read_bytes += self._bits_to_bytes(hidden_size * num_attention_heads * head_dim, self.weight_bits)  # Q
        read_bytes += self._bits_to_bytes(hidden_size * num_kv_heads * head_dim, self.weight_bits)  # K
        read_bytes += self._bits_to_bytes(hidden_size * num_kv_heads * head_dim, self.weight_bits)  # V

        # Write: Q, K, V outputs + KV cache
        write_bytes = self._bits_to_bytes(num_tokens * num_attention_heads * head_dim, self.activation_bits)  # Q
        write_bytes += self._bits_to_bytes(num_tokens * num_kv_heads * head_dim, self.kv_cache_bits)  # K cache
        write_bytes += self._bits_to_bytes(num_tokens * num_kv_heads * head_dim, self.kv_cache_bits)  # V cache

        return MemoryTraffic(read_bytes=read_bytes, write_bytes=write_bytes)

    def attention_traffic(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        kv_size: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> MemoryTraffic:
        """Self-attention memory traffic (QKT + softmax + PV)."""
        if mode == "prefill":
            num_tokens = seq_len * batch_size

            # Read: Q, K cache, V cache
            read_bytes = self._bits_to_bytes(num_tokens * num_attention_heads * head_dim, self.activation_bits)  # Q
            read_bytes += self._bits_to_bytes(kv_size * batch_size * num_kv_heads * head_dim, self.kv_cache_bits)  # K
            read_bytes += self._bits_to_bytes(kv_size * batch_size * num_kv_heads * head_dim, self.kv_cache_bits)  # V

            # Write: attention output
            write_bytes = self._bits_to_bytes(num_tokens * num_attention_heads * head_dim, self.activation_bits)

        else:  # decode
            # Read: Q (1 token), full K cache, full V cache
            read_bytes = self._bits_to_bytes(batch_size * num_attention_heads * head_dim, self.activation_bits)
            read_bytes += self._bits_to_bytes(kv_size * batch_size * num_kv_heads * head_dim, self.kv_cache_bits)
            read_bytes += self._bits_to_bytes(kv_size * batch_size * num_kv_heads * head_dim, self.kv_cache_bits)

            # Write: attention output (1 token per batch)
            write_bytes = self._bits_to_bytes(batch_size * num_attention_heads * head_dim, self.activation_bits)

        return MemoryTraffic(read_bytes=read_bytes, write_bytes=write_bytes)

    def sliding_attention_traffic(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        kv_size: int,
        batch_size: int,
        sliding_window_size: int,
        mode: str = "prefill",
    ) -> MemoryTraffic:
        """Sliding window attention memory traffic."""
        effective_kv_size = min(kv_size, sliding_window_size)

        if mode == "prefill":
            num_tokens = seq_len * batch_size

            # Read: Q, K cache (window), V cache (window)
            read_bytes = self._bits_to_bytes(num_tokens * num_attention_heads * head_dim, self.activation_bits)
            read_bytes += self._bits_to_bytes(effective_kv_size * batch_size * num_kv_heads * head_dim, self.kv_cache_bits)
            read_bytes += self._bits_to_bytes(effective_kv_size * batch_size * num_kv_heads * head_dim, self.kv_cache_bits)

            # Write: attention output
            write_bytes = self._bits_to_bytes(num_tokens * num_attention_heads * head_dim, self.activation_bits)

        else:  # decode
            read_bytes = self._bits_to_bytes(batch_size * num_attention_heads * head_dim, self.activation_bits)
            read_bytes += self._bits_to_bytes(effective_kv_size * batch_size * num_kv_heads * head_dim, self.kv_cache_bits)
            read_bytes += self._bits_to_bytes(effective_kv_size * batch_size * num_kv_heads * head_dim, self.kv_cache_bits)

            write_bytes = self._bits_to_bytes(batch_size * num_attention_heads * head_dim, self.activation_bits)

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
        """Output projection after attention."""
        if mode == "prefill":
            num_tokens = seq_len * batch_size
        else:
            num_tokens = batch_size

        # Read: attention output + O weight matrix
        read_bytes = self._bits_to_bytes(num_tokens * num_attention_heads * head_dim, self.activation_bits)
        read_bytes += self._bits_to_bytes(num_attention_heads * head_dim * hidden_size, self.weight_bits)

        # Write: projected output
        write_bytes = self._bits_to_bytes(num_tokens * hidden_size, self.activation_bits)

        return MemoryTraffic(read_bytes=read_bytes, write_bytes=write_bytes)

    def ffn_traffic(
        self,
        hidden_size: int,
        intermediate_size: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> MemoryTraffic:
        """Feed-forward layer memory traffic."""
        if mode == "prefill":
            num_tokens = seq_len * batch_size
        else:
            num_tokens = batch_size

        # Read: input + gate/up/down weights
        read_bytes = self._bits_to_bytes(num_tokens * hidden_size, self.activation_bits)
        read_bytes += self._bits_to_bytes(hidden_size * intermediate_size, self.weight_bits)  # gate
        read_bytes += self._bits_to_bytes(hidden_size * intermediate_size, self.weight_bits)  # up
        read_bytes += self._bits_to_bytes(intermediate_size * hidden_size, self.weight_bits)  # down

        # Write: FFN output
        write_bytes = self._bits_to_bytes(num_tokens * hidden_size, self.activation_bits)

        return MemoryTraffic(read_bytes=read_bytes, write_bytes=write_bytes)

    def moe_traffic(
        self,
        hidden_size: int,
        intermediate_size: int,
        seq_len: int,
        batch_size: int,
        num_experts: int,
        experts_per_token: int,
        mode: str = "prefill",
    ) -> MemoryTraffic:
        """MoE layer memory traffic."""
        if mode == "prefill":
            num_tokens = seq_len * batch_size
        else:
            num_tokens = batch_size

        # Router: read input, router weights
        read_bytes = self._bits_to_bytes(num_tokens * hidden_size, self.activation_bits)
        read_bytes += self._bits_to_bytes(hidden_size * num_experts, self.weight_bits)

        # Expert computation: each token activates experts_per_token experts
        # On average, each expert processes (num_tokens * experts_per_token / num_experts) tokens
        # But we read ALL expert weights (they're accessed)
        # Conservative: assume we read weights for all activated experts per token

        # Read expert weights (gate + up + down for each activated expert)
        expert_weight_bytes = self._bits_to_bytes(3 * hidden_size * intermediate_size, self.weight_bits)
        # Each token activates experts_per_token experts
        read_bytes += expert_weight_bytes * experts_per_token  # Amortized across batch

        # Write: final output
        write_bytes = self._bits_to_bytes(num_tokens * hidden_size, self.activation_bits)

        return MemoryTraffic(read_bytes=read_bytes, write_bytes=write_bytes)

    def residual_traffic(
        self,
        hidden_size: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> MemoryTraffic:
        """Residual connection memory traffic."""
        if mode == "prefill":
            num_tokens = seq_len * batch_size
        else:
            num_tokens = batch_size

        # Read: residual input + current output
        read_bytes = self._bits_to_bytes(2 * num_tokens * hidden_size, self.activation_bits)

        # Write: summed output
        write_bytes = self._bits_to_bytes(num_tokens * hidden_size, self.activation_bits)

        return MemoryTraffic(read_bytes=read_bytes, write_bytes=write_bytes)

    def lm_head_traffic(
        self,
        hidden_size: int,
        vocab_size: int,
        batch_size: int,
    ) -> MemoryTraffic:
        """LM head memory traffic."""
        # Read: final hidden states + LM head weights
        read_bytes = self._bits_to_bytes(batch_size * hidden_size, self.activation_bits)
        read_bytes += self._bits_to_bytes(hidden_size * vocab_size, self.weight_bits)

        # Write: logits
        write_bytes = self._bits_to_bytes(batch_size * vocab_size, self.activation_bits)

        return MemoryTraffic(read_bytes=read_bytes, write_bytes=write_bytes)

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
    # On-Chip (SRAM) Traffic Methods
    # -------------------------------------------------------------------------

    def projection_onchip_traffic(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> OnChipMemoryTraffic:
        """On-chip SRAM traffic for QKV projection."""
        if mode == "prefill":
            num_tokens = seq_len * batch_size
        else:
            num_tokens = batch_size

        # Matrix SRAM: weight tiles loaded from HBM, processed in tiles of MLEN x MLEN
        # Weights are tiled and each tile is read once from HBM to SRAM
        # Q weight: hidden_size x (num_attention_heads * head_dim)
        # K weight: hidden_size x (num_kv_heads * head_dim)
        # V weight: hidden_size x (num_kv_heads * head_dim)
        q_weight_tiles = math.ceil(hidden_size / self.mlen) * math.ceil(num_attention_heads * head_dim / self.mlen)
        k_weight_tiles = math.ceil(hidden_size / self.mlen) * math.ceil(num_kv_heads * head_dim / self.mlen)
        v_weight_tiles = math.ceil(hidden_size / self.mlen) * math.ceil(num_kv_heads * head_dim / self.mlen)

        tile_size_bytes = self._bits_to_bytes(self.mlen * self.mlen, self.weight_bits)
        matrix_sram_read = (q_weight_tiles + k_weight_tiles + v_weight_tiles) * tile_size_bytes

        # Vector SRAM: input activations read, output activations written
        # Input: num_tokens x hidden_size
        # Q output: num_tokens x (num_attention_heads * head_dim)
        # K output: num_tokens x (num_kv_heads * head_dim)
        # V output: num_tokens x (num_kv_heads * head_dim)
        vector_sram_read = self._bits_to_bytes(num_tokens * hidden_size, self.activation_bits)
        vector_sram_write = self._bits_to_bytes(
            num_tokens * (num_attention_heads + 2 * num_kv_heads) * head_dim,
            self.activation_bits
        )

        return OnChipMemoryTraffic(
            matrix_sram_read_bytes=matrix_sram_read,
            matrix_sram_write_bytes=0,  # Weights are read-only
            vector_sram_read_bytes=vector_sram_read,
            vector_sram_write_bytes=vector_sram_write,
        )

    def attention_onchip_traffic(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        kv_size: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> OnChipMemoryTraffic:
        """On-chip SRAM traffic for self-attention."""
        if mode == "prefill":
            num_tokens = seq_len * batch_size

            # Vector SRAM: Q read, attention scores intermediate, output written
            vector_sram_read = self._bits_to_bytes(num_tokens * num_attention_heads * head_dim, self.activation_bits)
            vector_sram_read += self._bits_to_bytes(kv_size * batch_size * num_kv_heads * head_dim * 2, self.kv_cache_bits)  # K + V

            # Attention scores: (num_attention_heads, seq_len, kv_size)
            attn_scores_bytes = self._bits_to_bytes(num_attention_heads * seq_len * kv_size, self.activation_bits)

            vector_sram_write = self._bits_to_bytes(num_tokens * num_attention_heads * head_dim, self.activation_bits)
            vector_sram_write += attn_scores_bytes  # Intermediate attention scores

        else:  # decode
            # Q: (batch_size, num_attention_heads * head_dim)
            vector_sram_read = self._bits_to_bytes(batch_size * num_attention_heads * head_dim, self.activation_bits)
            # K, V cache: (kv_size * batch_size, num_kv_heads * head_dim)
            vector_sram_read += self._bits_to_bytes(kv_size * batch_size * num_kv_heads * head_dim * 2, self.kv_cache_bits)

            # Output: (batch_size, num_attention_heads * head_dim)
            vector_sram_write = self._bits_to_bytes(batch_size * num_attention_heads * head_dim, self.activation_bits)

        return OnChipMemoryTraffic(
            matrix_sram_read_bytes=0,  # Attention doesn't use matrix weights
            matrix_sram_write_bytes=0,
            vector_sram_read_bytes=vector_sram_read,
            vector_sram_write_bytes=vector_sram_write,
        )

    def ffn_onchip_traffic(
        self,
        hidden_size: int,
        intermediate_size: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> OnChipMemoryTraffic:
        """On-chip SRAM traffic for FFN layer."""
        if mode == "prefill":
            num_tokens = seq_len * batch_size
        else:
            num_tokens = batch_size

        # Matrix SRAM: gate, up, down weights tiled
        gate_tiles = math.ceil(hidden_size / self.mlen) * math.ceil(intermediate_size / self.mlen)
        up_tiles = math.ceil(hidden_size / self.mlen) * math.ceil(intermediate_size / self.mlen)
        down_tiles = math.ceil(intermediate_size / self.mlen) * math.ceil(hidden_size / self.mlen)

        tile_size_bytes = self._bits_to_bytes(self.mlen * self.mlen, self.weight_bits)
        matrix_sram_read = (gate_tiles + up_tiles + down_tiles) * tile_size_bytes

        # Vector SRAM: input, intermediate activations, output
        vector_sram_read = self._bits_to_bytes(num_tokens * hidden_size, self.activation_bits)
        # Intermediate: gate output + up output (both intermediate_size)
        vector_sram_write = self._bits_to_bytes(num_tokens * intermediate_size * 2, self.activation_bits)
        # Final output
        vector_sram_write += self._bits_to_bytes(num_tokens * hidden_size, self.activation_bits)

        return OnChipMemoryTraffic(
            matrix_sram_read_bytes=matrix_sram_read,
            matrix_sram_write_bytes=0,
            vector_sram_read_bytes=vector_sram_read,
            vector_sram_write_bytes=vector_sram_write,
        )

    def moe_onchip_traffic(
        self,
        hidden_size: int,
        intermediate_size: int,
        seq_len: int,
        batch_size: int,
        num_experts: int,
        experts_per_token: int,
        mode: str = "prefill",
    ) -> OnChipMemoryTraffic:
        """On-chip SRAM traffic for MoE layer."""
        if mode == "prefill":
            num_tokens = seq_len * batch_size
        else:
            num_tokens = batch_size

        # Router weight tiles
        router_tiles = math.ceil(hidden_size / self.mlen) * math.ceil(num_experts / self.mlen)

        # Expert weight tiles (per activated expert): gate + up + down
        expert_tiles_per = (
            2 * math.ceil(hidden_size / self.mlen) * math.ceil(intermediate_size / self.mlen) +  # gate + up
            math.ceil(intermediate_size / self.mlen) * math.ceil(hidden_size / self.mlen)  # down
        )

        tile_size_bytes = self._bits_to_bytes(self.mlen * self.mlen, self.weight_bits)

        # Each token activates experts_per_token experts
        matrix_sram_read = router_tiles * tile_size_bytes
        matrix_sram_read += expert_tiles_per * experts_per_token * tile_size_bytes

        # Vector SRAM
        vector_sram_read = self._bits_to_bytes(num_tokens * hidden_size, self.activation_bits)
        # Intermediate per expert
        vector_sram_write = self._bits_to_bytes(
            num_tokens * experts_per_token * intermediate_size * 2,
            self.activation_bits
        )
        vector_sram_write += self._bits_to_bytes(num_tokens * hidden_size, self.activation_bits)

        return OnChipMemoryTraffic(
            matrix_sram_read_bytes=matrix_sram_read,
            matrix_sram_write_bytes=0,
            vector_sram_read_bytes=vector_sram_read,
            vector_sram_write_bytes=vector_sram_write,
        )

    # -------------------------------------------------------------------------
    # Memory Utilization Computation
    # -------------------------------------------------------------------------

    def compute_memory_utilization(
        self,
        hbm_traffic: MemoryTraffic,
        onchip_traffic: OnChipMemoryTraffic,
        execution_cycles: int,
        frequency_hz: float = 1e9,
        sram_bandwidth_multiplier: float = 4.0,  # SRAM typically 4x faster than HBM
    ) -> MemoryUtilization:
        """
        Compute memory utilization given traffic and execution time.

        Args:
            hbm_traffic: Off-chip HBM traffic
            onchip_traffic: On-chip SRAM traffic
            execution_cycles: Total execution cycles from performance model
            frequency_hz: Clock frequency
            sram_bandwidth_multiplier: SRAM bandwidth relative to HBM

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

        # On-chip SRAM metrics
        result.onchip_traffic = onchip_traffic
        result.matrix_sram_capacity_bytes = self.matrix_sram_bytes
        result.vector_sram_capacity_bytes = self.vector_sram_bytes

        # SRAM peak bandwidth (higher than HBM)
        result.matrix_sram_peak_bandwidth_gbps = result.hbm_peak_bandwidth_gbps * sram_bandwidth_multiplier
        result.vector_sram_peak_bandwidth_gbps = result.hbm_peak_bandwidth_gbps * sram_bandwidth_multiplier

        if result.execution_time_seconds > 0:
            result.matrix_sram_achieved_bandwidth_gbps = (
                onchip_traffic.total_matrix_sram_bytes / result.execution_time_seconds / 1e9
            )
            result.vector_sram_achieved_bandwidth_gbps = (
                onchip_traffic.total_vector_sram_bytes / result.execution_time_seconds / 1e9
            )

            result.matrix_sram_utilization = (
                result.matrix_sram_achieved_bandwidth_gbps / result.matrix_sram_peak_bandwidth_gbps
            )
            result.vector_sram_utilization = (
                result.vector_sram_achieved_bandwidth_gbps / result.vector_sram_peak_bandwidth_gbps
            )

        # Bottleneck analysis
        result.is_hbm_bound = result.hbm_utilization > 0.7

        return result
