"""
LLM Memory Model for PLENA Simulator.

Provides end-to-end memory analysis for LLM inference workloads including:
- Total weight memory footprint
- KV cache footprint with reduction analysis (GQA, sliding window, quantization)
- Memory traffic per phase (prefill, decode)
- Memory bandwidth utilization
- HBM capacity requirements

Usage:
    python llm_memory_model.py --model llama-3.1-8b --model-lib ./Model_Lib --config ./plena_settings.toml
    python llm_memory_model.py --model gpt-oss-20b --model-lib ./Model_Lib --config ./plena_settings.toml --json
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from memory_model import (
    BandwidthAnalysis,
    KVCacheFootprint,
    MemoryConfig,
    MemoryModel,
    MemoryTraffic,
    WeightFootprint,
    load_memory_config_from_toml,
)


# =============================================================================
# Result Data Classes
# =============================================================================


@dataclass
class PhaseMemoryAnalysis:
    """Memory analysis for a single phase (prefill or decode).

    Only tracks HBM traffic: weights + KV cache.
    Activations (norm, residual) stay on SRAM and are not counted here.
    """

    total_traffic: MemoryTraffic = field(default_factory=MemoryTraffic)

    # Traffic breakdown by component (HBM only)
    embedding_traffic: MemoryTraffic = field(default_factory=MemoryTraffic)
    attention_traffic: MemoryTraffic = field(default_factory=MemoryTraffic)
    ffn_traffic: MemoryTraffic = field(default_factory=MemoryTraffic)
    lm_head_traffic: MemoryTraffic = field(default_factory=MemoryTraffic)

    # Bytes per token (for decode phase)
    bytes_per_output_token: int = 0


@dataclass
class PhaseUtilizationAnalysis:
    """Utilization analysis for a single phase (prefill or decode one token)."""

    # Execution metrics (from performance model)
    execution_cycles: int = 0
    execution_time_seconds: float = 0.0

    # Off-chip (HBM) traffic and utilization
    hbm_traffic: MemoryTraffic = field(default_factory=MemoryTraffic)
    hbm_read_bytes: int = 0
    hbm_write_bytes: int = 0
    hbm_total_bytes: int = 0
    hbm_peak_bandwidth_gbps: float = 0.0
    hbm_achieved_bandwidth_gbps: float = 0.0
    hbm_utilization: float = 0.0  # Average utilization (0-1)

    # Component breakdown
    component_traffic: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "execution": {
                "cycles": self.execution_cycles,
                "time_seconds": self.execution_time_seconds,
                "time_ms": self.execution_time_seconds * 1000,
            },
            "hbm": {
                "read_bytes": self.hbm_read_bytes,
                "write_bytes": self.hbm_write_bytes,
                "total_bytes": self.hbm_total_bytes,
                "read_mb": self.hbm_read_bytes / 1e6,
                "write_mb": self.hbm_write_bytes / 1e6,
                "total_mb": self.hbm_total_bytes / 1e6,
                "peak_bandwidth_gbps": self.hbm_peak_bandwidth_gbps,
                "achieved_bandwidth_gbps": self.hbm_achieved_bandwidth_gbps,
                "utilization_percent": self.hbm_utilization * 100,
            },
            "component_breakdown": self.component_traffic,
        }


@dataclass
class LLMMemoryAnalysis:
    """Complete LLM memory analysis results."""

    model_name: str
    batch_size: int
    input_seq_len: int
    output_seq_len: int

    # Memory footprint
    weight_footprint: WeightFootprint = field(default_factory=WeightFootprint)
    kv_cache_footprint: KVCacheFootprint = field(default_factory=KVCacheFootprint)
    peak_activation_bytes: int = 0

    # HBM capacity analysis
    hbm_capacity_bytes: int = 0
    total_required_bytes: int = 0
    hbm_utilization_ratio: float = 0.0
    fits_in_hbm: bool = True

    # Phase analysis
    prefill_analysis: PhaseMemoryAnalysis = field(default_factory=PhaseMemoryAnalysis)
    decode_analysis: PhaseMemoryAnalysis = field(default_factory=PhaseMemoryAnalysis)

    # Bandwidth analysis
    prefill_bandwidth: BandwidthAnalysis = field(default_factory=BandwidthAnalysis)
    decode_bandwidth: BandwidthAnalysis = field(default_factory=BandwidthAnalysis)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "input_seq_len": self.input_seq_len,
            "output_seq_len": self.output_seq_len,
            "weight_footprint": {
                "embedding_mb": self.weight_footprint.embedding_bytes / 1e6,
                "attention_mb": self.weight_footprint.attention_bytes / 1e6,
                "ffn_mb": self.weight_footprint.ffn_bytes / 1e6,
                "lm_head_mb": self.weight_footprint.lm_head_bytes / 1e6,
                "router_mb": self.weight_footprint.router_bytes / 1e6,
                "expert_mb": self.weight_footprint.expert_bytes / 1e6,
                "other_mb": self.weight_footprint.other_bytes / 1e6,
                "total_mb": self.weight_footprint.total_bytes / 1e6,
                "total_gb": self.weight_footprint.total_bytes / 1e9,
            },
            "kv_cache_footprint": {
                "per_layer_mb": self.kv_cache_footprint.bytes_per_layer / 1e6,
                "total_mb": self.kv_cache_footprint.total_bytes / 1e6,
                "total_gb": self.kv_cache_footprint.total_bytes / 1e9,
            },
            "hbm_analysis": {
                "capacity_gb": self.hbm_capacity_bytes / 1e9,
                "required_gb": self.total_required_bytes / 1e9,
                "utilization_percent": self.hbm_utilization_ratio * 100,
                "fits_in_hbm": self.fits_in_hbm,
                "headroom_gb": (self.hbm_capacity_bytes - self.total_required_bytes) / 1e9,
            },
            "prefill_traffic": {
                "read_mb": self.prefill_analysis.total_traffic.read_bytes / 1e6,
                "write_mb": self.prefill_analysis.total_traffic.write_bytes / 1e6,
                "total_mb": self.prefill_analysis.total_traffic.total_bytes / 1e6,
            },
            "decode_traffic": {
                "read_mb": self.decode_analysis.total_traffic.read_bytes / 1e6,
                "write_mb": self.decode_analysis.total_traffic.write_bytes / 1e6,
                "total_mb": self.decode_analysis.total_traffic.total_bytes / 1e6,
                "bytes_per_token": self.decode_analysis.bytes_per_output_token,
                "mb_per_token": self.decode_analysis.bytes_per_output_token / 1e6,
            },
            "bandwidth_utilization": {
                "prefill": {
                    "peak_gbps": self.prefill_bandwidth.peak_bandwidth_gbps,
                    "achieved_gbps": self.prefill_bandwidth.achieved_bandwidth_gbps,
                    "utilization_percent": self.prefill_bandwidth.bandwidth_utilization * 100,
                },
                "decode": {
                    "peak_gbps": self.decode_bandwidth.peak_bandwidth_gbps,
                    "achieved_gbps": self.decode_bandwidth.achieved_bandwidth_gbps,
                    "utilization_percent": self.decode_bandwidth.bandwidth_utilization * 100,
                },
            },
        }


# =============================================================================
# LLM Memory Model
# =============================================================================


class LLMMemoryModel:
    """
    End-to-end LLM memory analysis model.

    Computes memory footprint, traffic, and bandwidth for LLM inference
    on PLENA hardware. Supports both standard transformer and MoE architectures.
    """

    def __init__(
        self,
        model_config_path: str,
        memory_config: MemoryConfig,
        batch_size: int = 1,
        input_seq_len: int = 2048,
        output_seq_len: int = 128,
        device_num: int = 1,
        frequency_hz: float = 1e9,
        use_flash_attention: bool = True,
    ):
        """
        Initialize LLM memory model.

        Args:
            model_config_path: Path to model config JSON
            memory_config: PLENA memory configuration
            batch_size: Inference batch size
            input_seq_len: Input/prompt sequence length
            output_seq_len: Output/generation sequence length
            device_num: Number of devices for parallelism
            frequency_hz: Clock frequency for bandwidth calculations
            use_flash_attention: Use flash attention (no materialized attention matrix)
        """
        with open(model_config_path) as f:
            model_param = json.load(f)

        # Model architecture parameters
        self.hidden_size = model_param["hidden_size"]
        self.num_attention_heads = model_param["num_attention_heads"]
        self.num_hidden_layers = model_param["num_hidden_layers"]
        self.intermediate_size = model_param["intermediate_size"]
        self.num_key_value_heads = model_param["num_key_value_heads"]
        self.vocab_size = model_param["vocab_size"]
        self.head_dim = model_param.get("head_dim", self.hidden_size // self.num_attention_heads)

        # MoE parameters
        self.num_experts = model_param.get("num_local_experts", 1)
        self.experts_per_token = model_param.get("experts_per_token", model_param.get("num_experts_per_tok", 1))

        # Per-layer MLP types: "ffn" or "moe"
        # Default: all "moe" if num_experts > 1, else all "ffn"
        default_mlp_type = "moe" if self.num_experts > 1 else "ffn"
        self.mlp_types = model_param.get("mlp_types", [default_mlp_type] * self.num_hidden_layers)
        self.num_moe_layers = sum(1 for mt in self.mlp_types if mt == "moe")
        self.num_ffn_layers = self.num_hidden_layers - self.num_moe_layers

        # Attention parameters
        self.sliding_window = model_param.get("sliding_window", 0)
        self.layer_types = model_param.get("layer_types", ["full_attention"] * self.num_hidden_layers)
        self.num_sliding_layers = sum(1 for lt in self.layer_types if lt == "sliding_attention")
        self.num_full_layers = self.num_hidden_layers - self.num_sliding_layers

        # Embedding tie
        self.tie_embeddings = model_param.get("tie_word_embeddings", False)

        # Inference parameters
        self.batch_size = batch_size
        self.device_batch_size = batch_size // device_num
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.device_num = device_num
        self.frequency = frequency_hz
        self.use_flash_attention = use_flash_attention

        # Memory model
        self.memory_config = memory_config
        self.mem = MemoryModel(memory_config)

        # Model name from path
        self.model_name = Path(model_config_path).stem

    def print_config(self):
        """Print model and memory configuration."""
        print("=" * 70)
        print("LLM Memory Model Configuration")
        print("=" * 70)
        print(f"Model:                {self.model_name}")
        print("-" * 70)
        print("Architecture")
        print("-" * 70)
        print(f"Hidden size:          {self.hidden_size}")
        print(f"Num attention heads:  {self.num_attention_heads}")
        print(f"Num KV heads:         {self.num_key_value_heads}")
        print(f"Head dim:             {self.head_dim}")
        print(f"Num hidden layers:    {self.num_hidden_layers}")
        print(f"Intermediate size:    {self.intermediate_size}")
        print(f"Vocab size:           {self.vocab_size}")

        if self.num_moe_layers > 0:
            print("-" * 70)
            print("MLP Configuration")
            print("-" * 70)
            print(f"FFN layers:           {self.num_ffn_layers}")
            print(f"MoE layers:           {self.num_moe_layers}")
            print(f"Num experts:          {self.num_experts}")
            print(f"Experts per token:    {self.experts_per_token}")

        if self.sliding_window > 0:
            print("-" * 70)
            print("Attention Configuration")
            print("-" * 70)
            print(f"Sliding window size:  {self.sliding_window}")
            print(f"Sliding attn layers:  {self.num_sliding_layers}")
            print(f"Full attn layers:     {self.num_full_layers}")

        print("-" * 70)
        print("Inference Settings")
        print("-" * 70)
        print(f"Batch size:           {self.batch_size}")
        print(f"Input seq len:        {self.input_seq_len}")
        print(f"Output seq len:       {self.output_seq_len}")
        print(f"Device num:           {self.device_num}")
        print(f"Flash attention:      {'Enabled' if self.use_flash_attention else 'Disabled'}")
        print("-" * 70)
        print("Memory Configuration")
        print("-" * 70)
        print(f"HBM capacity:         {self.memory_config.HBM_SIZE / 1e9:.2f} GB")

        # Helper to format precision with format info
        def fmt_precision(bits: float, fmt: str) -> str:
            if fmt == "Mx":
                return f"{bits:.1f} bits avg (MX format)"
            return f"{bits:.1f} bits ({fmt})"

        print(f"Weight precision:     {fmt_precision(self.memory_config.weight_bits, self.memory_config.weight_format)}")
        print(f"KV cache precision:   {fmt_precision(self.memory_config.kv_cache_bits, self.memory_config.kv_cache_format)}")
        print(f"Activation precision: {fmt_precision(self.memory_config.activation_bits, self.memory_config.activation_format)}")
        print("=" * 70)

    # -------------------------------------------------------------------------
    # Weight Footprint Computation
    # -------------------------------------------------------------------------

    def compute_weight_footprint(self) -> WeightFootprint:
        """Compute total model weight memory footprint."""
        result = WeightFootprint()

        # Embedding
        result.embedding_bytes = self.mem.embedding_weights(self.vocab_size, self.hidden_size)

        # Per-layer attention weights
        attention_per_layer = self.mem.attention_weights(
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )
        result.attention_bytes = attention_per_layer * self.num_hidden_layers

        # Per-layer FFN or MoE weights (based on mlp_types)
        ffn_per_layer = self.mem.ffn_weights(self.hidden_size, self.intermediate_size)
        router_per_layer, expert_per_layer = self.mem.moe_weights(
            self.hidden_size,
            self.intermediate_size,
            self.num_experts,
        )

        result.ffn_bytes = ffn_per_layer * self.num_ffn_layers
        result.router_bytes = router_per_layer * self.num_moe_layers
        result.expert_bytes = expert_per_layer * self.num_moe_layers

        # Layer norms (2 per layer: pre-attention and pre-FFN)
        norm_per_layer = self.mem.layer_norm_weights(self.hidden_size) * 2
        result.other_bytes = norm_per_layer * self.num_hidden_layers

        # LM head
        result.lm_head_bytes = self.mem.lm_head_weights(self.hidden_size, self.vocab_size, self.tie_embeddings)

        return result

    # -------------------------------------------------------------------------
    # KV Cache Footprint Computation
    # -------------------------------------------------------------------------

    def compute_kv_cache_footprint(self, total_seq_len: Optional[int] = None) -> KVCacheFootprint:
        """Compute KV cache memory footprint."""
        if total_seq_len is None:
            total_seq_len = self.input_seq_len + self.output_seq_len

        return self.mem.kv_cache_footprint(
            num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            num_layers=self.num_hidden_layers,
            seq_len=total_seq_len,
            batch_size=self.device_batch_size,
        )

    # -------------------------------------------------------------------------
    # Memory Traffic Computation
    # -------------------------------------------------------------------------

    def compute_prefill_traffic(self) -> PhaseMemoryAnalysis:
        """Compute HBM memory traffic for prefill phase (weights + KV cache only)."""
        result = PhaseMemoryAnalysis()
        mode = "prefill"
        kv_size = self.input_seq_len
        num_layers = self.num_hidden_layers

        # Embedding weights
        result.embedding_traffic = self.mem.embedding_traffic(
            self.hidden_size, self.input_seq_len, self.device_batch_size, mode
        )

        # QKV projection weights + KV cache write
        proj_per_layer = self.mem.projection_traffic(
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.input_seq_len,
            self.device_batch_size,
            mode,
        )
        # Output projection weights
        out_proj_per_layer = self.mem.output_projection_traffic(
            self.hidden_size,
            self.num_attention_heads,
            self.head_dim,
            self.input_seq_len,
            self.device_batch_size,
            mode,
        )
        result.attention_traffic = (proj_per_layer + out_proj_per_layer) * num_layers

        # Attention: KV cache reads (sliding vs full)
        # Use flash attention or standard attention based on setting
        if self.use_flash_attention:
            full_attn_traffic = self.mem.flash_attention_traffic(
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.input_seq_len,
                kv_size,
                self.device_batch_size,
                mode,
            )
        else:
            full_attn_traffic = self.mem.attention_traffic(
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.input_seq_len,
                kv_size,
                self.device_batch_size,
                mode,
            )

        if self.sliding_window > 0:
            # Sliding attention is inherently flash-style (no full attention matrix)
            sliding_attn_traffic = self.mem.sliding_attention_traffic(
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.input_seq_len,
                kv_size,
                self.device_batch_size,
                self.sliding_window,
                mode,
            )
        else:
            sliding_attn_traffic = full_attn_traffic

        result.attention_traffic += full_attn_traffic * self.num_full_layers
        result.attention_traffic += sliding_attn_traffic * self.num_sliding_layers

        # MLP weights: FFN vs MoE
        ffn_traffic = self.mem.ffn_traffic(
            self.hidden_size,
            self.intermediate_size,
            self.input_seq_len,
            self.device_batch_size,
            mode,
        )
        moe_traffic = self.mem.moe_traffic(
            self.hidden_size,
            self.intermediate_size,
            self.num_experts,
            self.experts_per_token,
            self.input_seq_len,
            self.device_batch_size,
            mode,
        )
        result.ffn_traffic = ffn_traffic * self.num_ffn_layers + moe_traffic * self.num_moe_layers

        # LM head weights
        result.lm_head_traffic = self.mem.lm_head_traffic(self.hidden_size, self.vocab_size)

        # Total HBM traffic
        result.total_traffic = (
            result.embedding_traffic + result.attention_traffic + result.ffn_traffic + result.lm_head_traffic
        )

        return result

    def compute_decode_traffic(self, num_output_tokens: Optional[int] = None) -> PhaseMemoryAnalysis:
        """
        Compute HBM memory traffic for a single decode step (generating one token).

        Args:
            num_output_tokens: Number of tokens already generated before this step.
                               KV cache size = input_seq_len + num_output_tokens.
                               If None, uses 0 (first decode token after prefill).
        """
        if num_output_tokens is None:
            num_output_tokens = 0

        result = PhaseMemoryAnalysis()
        mode = "decode"
        num_layers = self.num_hidden_layers

        # KV cache size: prefill tokens + already generated tokens
        kv_size = self.input_seq_len + num_output_tokens

        # QKV projection weights + KV cache write (for 1 token)
        proj_per_layer = self.mem.projection_traffic(
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            1,  # decode generates 1 token
            self.device_batch_size,
            mode,
        )
        # Output projection weights (for 1 token)
        out_proj_per_layer = self.mem.output_projection_traffic(
            self.hidden_size,
            self.num_attention_heads,
            self.head_dim,
            1,  # decode generates 1 token
            self.device_batch_size,
            mode,
        )
        proj_total = (proj_per_layer + out_proj_per_layer) * num_layers

        # MLP weights: FFN vs MoE (for 1 token)
        ffn_per_layer = self.mem.ffn_traffic(
            self.hidden_size,
            self.intermediate_size,
            1,  # decode generates 1 token
            self.device_batch_size,
            mode,
        )
        moe_per_layer = self.mem.moe_traffic(
            self.hidden_size,
            self.intermediate_size,
            self.num_experts,
            self.experts_per_token,
            1,  # decode generates 1 token
            self.device_batch_size,
            mode,
        )
        result.ffn_traffic = ffn_per_layer * self.num_ffn_layers + moe_per_layer * self.num_moe_layers

        # LM head weights (for 1 token)
        result.lm_head_traffic = self.mem.lm_head_traffic(self.hidden_size, self.vocab_size)

        # Attention: KV cache reads for current kv_size
        # Use flash attention or standard attention based on setting
        if self.use_flash_attention:
            full_attn = self.mem.flash_attention_traffic(
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                1,  # decode generates 1 token
                kv_size,
                self.device_batch_size,
                mode,
            )
        else:
            full_attn = self.mem.attention_traffic(
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                1,  # decode generates 1 token
                kv_size,
                self.device_batch_size,
                mode,
            )
        attn_traffic = full_attn * self.num_full_layers

        # Sliding attention layers (inherently flash-style)
        if self.sliding_window > 0 and self.num_sliding_layers > 0:
            sliding_attn = self.mem.sliding_attention_traffic(
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                1,  # decode generates 1 token
                kv_size,
                self.device_batch_size,
                self.sliding_window,
                mode,
            )
            attn_traffic += sliding_attn * self.num_sliding_layers

        result.attention_traffic = proj_total + attn_traffic

        # Total HBM traffic (for this single decode step)
        result.total_traffic = result.attention_traffic + result.ffn_traffic + result.lm_head_traffic

        # Bytes for this single token
        result.bytes_per_output_token = result.total_traffic.total_bytes

        return result

    # -------------------------------------------------------------------------
    # Utilization Analysis (One Token)
    # -------------------------------------------------------------------------

    def compute_prefill_utilization(
        self,
        execution_cycles: int,
        frequency_hz: Optional[float] = None,
    ) -> PhaseUtilizationAnalysis:
        """
        Compute memory utilization for prefill phase (generating one token).

        Args:
            execution_cycles: Total execution cycles from performance model
            frequency_hz: Clock frequency (defaults to self.frequency)

        Returns:
            PhaseUtilizationAnalysis with traffic and utilization metrics
        """
        if frequency_hz is None:
            frequency_hz = self.frequency

        result = PhaseUtilizationAnalysis()
        result.execution_cycles = execution_cycles
        result.execution_time_seconds = execution_cycles / frequency_hz

        # Compute HBM (off-chip) traffic
        hbm_analysis = self.compute_prefill_traffic()
        result.hbm_traffic = hbm_analysis.total_traffic
        result.hbm_read_bytes = hbm_analysis.total_traffic.read_bytes
        result.hbm_write_bytes = hbm_analysis.total_traffic.write_bytes
        result.hbm_total_bytes = hbm_analysis.total_traffic.total_bytes

        # Compute HBM bandwidth and utilization
        result.hbm_peak_bandwidth_gbps = self.mem.compute_peak_bandwidth(frequency_hz)
        if result.execution_time_seconds > 0:
            result.hbm_achieved_bandwidth_gbps = result.hbm_total_bytes / result.execution_time_seconds / 1e9
            result.hbm_utilization = result.hbm_achieved_bandwidth_gbps / result.hbm_peak_bandwidth_gbps

        # Component breakdown (HBM only: weights + KV cache)
        result.component_traffic = {
            "embedding": {
                "read_mb": hbm_analysis.embedding_traffic.read_bytes / 1e6,
                "write_mb": hbm_analysis.embedding_traffic.write_bytes / 1e6,
            },
            "attention": {
                "read_mb": hbm_analysis.attention_traffic.read_bytes / 1e6,
                "write_mb": hbm_analysis.attention_traffic.write_bytes / 1e6,
            },
            "ffn": {
                "read_mb": hbm_analysis.ffn_traffic.read_bytes / 1e6,
                "write_mb": hbm_analysis.ffn_traffic.write_bytes / 1e6,
            },
            "lm_head": {
                "read_mb": hbm_analysis.lm_head_traffic.read_bytes / 1e6,
                "write_mb": hbm_analysis.lm_head_traffic.write_bytes / 1e6,
            },
        }

        return result

    def compute_decode_utilization(
        self,
        execution_cycles: int,
        kv_size: Optional[int] = None,
        frequency_hz: Optional[float] = None,
    ) -> PhaseUtilizationAnalysis:
        """
        Compute memory utilization for decode phase (generating one token).

        Args:
            execution_cycles: Total execution cycles from performance model
            kv_size: KV cache size at this decode step. If None, uses input_seq_len.
            frequency_hz: Clock frequency (defaults to self.frequency)

        Returns:
            PhaseUtilizationAnalysis with traffic and utilization metrics
        """
        if frequency_hz is None:
            frequency_hz = self.frequency
        if kv_size is None:
            kv_size = self.input_seq_len

        result = PhaseUtilizationAnalysis()
        result.execution_cycles = execution_cycles
        result.execution_time_seconds = execution_cycles / frequency_hz

        # Compute HBM (off-chip) traffic for one decode token
        # num_output_tokens = tokens already generated = kv_size - input_seq_len
        num_output_tokens = kv_size - self.input_seq_len
        hbm_analysis = self.compute_decode_traffic(num_output_tokens=num_output_tokens)
        result.hbm_traffic = hbm_analysis.total_traffic
        result.hbm_read_bytes = hbm_analysis.total_traffic.read_bytes
        result.hbm_write_bytes = hbm_analysis.total_traffic.write_bytes
        result.hbm_total_bytes = hbm_analysis.total_traffic.total_bytes

        # Compute HBM bandwidth and utilization
        result.hbm_peak_bandwidth_gbps = self.mem.compute_peak_bandwidth(frequency_hz)
        if result.execution_time_seconds > 0:
            result.hbm_achieved_bandwidth_gbps = result.hbm_total_bytes / result.execution_time_seconds / 1e9
            result.hbm_utilization = result.hbm_achieved_bandwidth_gbps / result.hbm_peak_bandwidth_gbps

        # Component breakdown (HBM only: weights + KV cache)
        result.component_traffic = {
            "attention": {
                "read_mb": hbm_analysis.attention_traffic.read_bytes / 1e6,
                "write_mb": hbm_analysis.attention_traffic.write_bytes / 1e6,
            },
            "ffn": {
                "read_mb": hbm_analysis.ffn_traffic.read_bytes / 1e6,
                "write_mb": hbm_analysis.ffn_traffic.write_bytes / 1e6,
            },
            "lm_head": {
                "read_mb": hbm_analysis.lm_head_traffic.read_bytes / 1e6,
                "write_mb": hbm_analysis.lm_head_traffic.write_bytes / 1e6,
            },
        }

        return result

    # -------------------------------------------------------------------------
    # Complete Analysis
    # -------------------------------------------------------------------------

    def analyze(self, prefill_cycles: Optional[int] = None, decode_cycles: Optional[int] = None) -> LLMMemoryAnalysis:
        """
        Perform complete memory analysis.

        Args:
            prefill_cycles: Number of cycles for prefill phase (for bandwidth calculation)
            decode_cycles: Number of cycles for decode phase (for bandwidth calculation)

        Returns:
            LLMMemoryAnalysis with all metrics
        """
        result = LLMMemoryAnalysis(
            model_name=self.model_name,
            batch_size=self.batch_size,
            input_seq_len=self.input_seq_len,
            output_seq_len=self.output_seq_len,
        )

        # Weight footprint
        result.weight_footprint = self.compute_weight_footprint()

        # KV cache footprint
        result.kv_cache_footprint = self.compute_kv_cache_footprint()

        # Peak activation memory (rough estimate: one layer's intermediate activations)
        # In practice, activation checkpointing affects this
        peak_seq = max(self.input_seq_len, 1)  # Prefill has larger activations
        result.peak_activation_bytes = int(
            self.device_batch_size
            * peak_seq
            * self.hidden_size
            * (self.memory_config.activation_bits / 8)
            * 4  # Factor for intermediate results
        )

        # HBM capacity analysis
        result.hbm_capacity_bytes = self.memory_config.HBM_SIZE
        result.total_required_bytes = (
            result.weight_footprint.total_bytes + result.kv_cache_footprint.total_bytes + result.peak_activation_bytes
        )
        result.hbm_utilization_ratio = result.total_required_bytes / result.hbm_capacity_bytes
        result.fits_in_hbm = result.total_required_bytes <= result.hbm_capacity_bytes

        # Traffic analysis
        result.prefill_analysis = self.compute_prefill_traffic()
        result.decode_analysis = self.compute_decode_traffic()

        # Bandwidth analysis (if cycles provided)
        if prefill_cycles and prefill_cycles > 0:
            result.prefill_bandwidth = self.mem.compute_bandwidth_utilization(
                result.prefill_analysis.total_traffic, prefill_cycles, self.frequency
            )

        if decode_cycles and decode_cycles > 0:
            result.decode_bandwidth = self.mem.compute_bandwidth_utilization(
                result.decode_analysis.total_traffic, decode_cycles, self.frequency
            )

        return result

    def print_analysis(self, analysis: LLMMemoryAnalysis):
        """Print formatted analysis results."""
        print("\n" + "=" * 70)
        print("MEMORY ANALYSIS RESULTS")
        print("=" * 70)

        # Weight footprint
        print("\n[WEIGHT MEMORY FOOTPRINT]")
        print("-" * 70)
        wf = analysis.weight_footprint
        print(f"  Embedding:      {wf.embedding_bytes / 1e9:8.3f} GB")
        print(f"  Attention:      {wf.attention_bytes / 1e9:8.3f} GB")
        if wf.ffn_bytes > 0:
            print(f"  FFN:            {wf.ffn_bytes / 1e9:8.3f} GB")
        if wf.router_bytes > 0:
            print(f"  MoE Router:     {wf.router_bytes / 1e9:8.3f} GB")
        if wf.expert_bytes > 0:
            print(f"  MoE Experts:    {wf.expert_bytes / 1e9:8.3f} GB")
        print(f"  LM Head:        {wf.lm_head_bytes / 1e9:8.3f} GB")
        print(f"  Other (norms):  {wf.other_bytes / 1e9:8.3f} GB")
        print(f"  ----------------------------------------")
        print(f"  TOTAL WEIGHTS:  {wf.total_bytes / 1e9:8.3f} GB")

        # KV cache footprint
        print("\n[KV CACHE FOOTPRINT]")
        print("-" * 70)
        kv = analysis.kv_cache_footprint
        print(f"  Per-layer (K+V):    {kv.bytes_per_layer / 1e6:8.2f} MB")
        print(f"  Total (all layers): {kv.total_bytes / 1e9:8.3f} GB")

        # HBM capacity
        print("\n[HBM CAPACITY ANALYSIS]")
        print("-" * 70)
        print(f"  HBM Capacity:       {analysis.hbm_capacity_bytes / 1e9:8.3f} GB")
        print(f"  Weights:            {wf.total_bytes / 1e9:8.3f} GB")
        print(f"  KV Cache:           {kv.total_bytes / 1e9:8.3f} GB")
        print(f"  Peak Activations:   {analysis.peak_activation_bytes / 1e9:8.3f} GB")
        print(f"  ----------------------------------------")
        print(f"  Total Required:     {analysis.total_required_bytes / 1e9:8.3f} GB")
        print(f"  HBM Utilization:    {analysis.hbm_utilization_ratio * 100:8.1f}%")
        print(f"  Fits in HBM:        {'Yes' if analysis.fits_in_hbm else 'NO - EXCEEDS CAPACITY'}")

        # Memory traffic
        print("\n[MEMORY TRAFFIC]")
        print("-" * 70)
        pf = analysis.prefill_analysis.total_traffic
        dc = analysis.decode_analysis.total_traffic
        print("  Prefill Phase:")
        print(f"    Read:     {pf.read_bytes / 1e9:8.3f} GB")
        print(f"    Write:    {pf.write_bytes / 1e9:8.3f} GB")
        print(f"    Total:    {pf.total_bytes / 1e9:8.3f} GB")
        print("  Decode Phase:")
        print(f"    Read:     {dc.read_bytes / 1e9:8.3f} GB")
        print(f"    Write:    {dc.write_bytes / 1e9:8.3f} GB")
        print(f"    Total:    {dc.total_bytes / 1e9:8.3f} GB")
        print(f"    Per token:{analysis.decode_analysis.bytes_per_output_token / 1e6:8.3f} MB")

        # Bandwidth utilization
        if analysis.prefill_bandwidth.peak_bandwidth_gbps > 0:
            print("\n[BANDWIDTH UTILIZATION]")
            print("-" * 70)
            print(f"  Peak Bandwidth:     {analysis.prefill_bandwidth.peak_bandwidth_gbps:8.1f} GB/s")
            if analysis.prefill_bandwidth.achieved_bandwidth_gbps > 0:
                print(
                    f"  Prefill Achieved:   {analysis.prefill_bandwidth.achieved_bandwidth_gbps:8.1f} GB/s ({analysis.prefill_bandwidth.bandwidth_utilization * 100:.1f}%)"
                )
            if analysis.decode_bandwidth.achieved_bandwidth_gbps > 0:
                print(
                    f"  Decode Achieved:    {analysis.decode_bandwidth.achieved_bandwidth_gbps:8.1f} GB/s ({analysis.decode_bandwidth.bandwidth_utilization * 100:.1f}%)"
                )

        print("\n" + "=" * 70)

    def print_utilization(self, phase: str, util: PhaseUtilizationAnalysis):
        """Print formatted utilization analysis for a phase."""
        print("\n" + "=" * 70)
        print(f"MEMORY UTILIZATION - {phase.upper()} (One Token)")
        print("=" * 70)

        # Execution time
        print("\n[EXECUTION]")
        print("-" * 70)
        print(f"  Cycles:         {util.execution_cycles:,}")
        print(f"  Time:           {util.execution_time_seconds * 1e6:.2f} us")

        # Off-chip (HBM) traffic and utilization
        print("\n[OFF-CHIP (HBM) MEMORY]")
        print("-" * 70)
        print(f"  Read:           {util.hbm_read_bytes / 1e6:8.2f} MB")
        print(f"  Write:          {util.hbm_write_bytes / 1e6:8.2f} MB")
        print(f"  Total:          {util.hbm_total_bytes / 1e6:8.2f} MB")
        print(f"  Peak BW:        {util.hbm_peak_bandwidth_gbps:8.1f} GB/s")
        print(f"  Achieved BW:    {util.hbm_achieved_bandwidth_gbps:8.1f} GB/s")
        print(f"  Utilization:    {util.hbm_utilization * 100:8.1f}%")

        # On-chip (SRAM) traffic and utilization
        print("\n[ON-CHIP (SRAM) MEMORY]")
        print("-" * 70)
        print("  Matrix SRAM:")
        print(f"    Read:         {util.matrix_sram_read_bytes / 1e6:8.2f} MB")
        print(f"    Write:        {util.matrix_sram_write_bytes / 1e6:8.2f} MB")
        print(f"    Peak BW:      {util.matrix_sram_peak_bandwidth_gbps:8.1f} GB/s")
        print(f"    Achieved BW:  {util.matrix_sram_achieved_bandwidth_gbps:8.1f} GB/s")
        print(f"    Utilization:  {util.matrix_sram_utilization * 100:8.1f}%")
        print("  Vector SRAM:")
        print(f"    Read:         {util.vector_sram_read_bytes / 1e6:8.2f} MB")
        print(f"    Write:        {util.vector_sram_write_bytes / 1e6:8.2f} MB")
        print(f"    Peak BW:      {util.vector_sram_peak_bandwidth_gbps:8.1f} GB/s")
        print(f"    Achieved BW:  {util.vector_sram_achieved_bandwidth_gbps:8.1f} GB/s")
        print(f"    Utilization:  {util.vector_sram_utilization * 100:8.1f}%")

        # Component breakdown
        if util.component_traffic:
            print("\n[COMPONENT BREAKDOWN (HBM)]")
            print("-" * 70)
            for comp, traffic in util.component_traffic.items():
                total = traffic["read_mb"] + traffic["write_mb"]
                print(
                    f"  {comp:12s}:  R={traffic['read_mb']:7.2f} MB  W={traffic['write_mb']:7.2f} MB  Total={total:7.2f} MB"
                )

        print("\n" + "=" * 70)


# =============================================================================
# Model Library Utilities
# =============================================================================


def list_available_models(model_lib_path: Path) -> list:
    """List all available model configs in Model_Lib."""
    if not model_lib_path.exists():
        return []
    return sorted([f.stem for f in model_lib_path.glob("*.json")])


def resolve_model_path(model_name: str, model_lib_path: Path) -> Path:
    """Resolve model name to full path."""
    model_path = model_lib_path / f"{model_name}.json"
    if not model_path.exists():
        available = list_available_models(model_lib_path)
        raise FileNotFoundError(f"Model '{model_name}' not found.\nAvailable models: {', '.join(available)}")
    return model_path


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="LLM Memory Model - Analyze memory footprint, traffic, and bandwidth for LLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python llm_memory_model.py --model llama-3.1-8b --model-lib ./Model_Lib --config ./plena_settings.toml
  python llm_memory_model.py --model gpt-oss-20b --model-lib ./Model_Lib --config ./plena_settings.toml --json
  python llm_memory_model.py --list-models --model-lib ./Model_Lib
""",
    )

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", "-m", help="Model name from Model_Lib")
    model_group.add_argument("--model-path", help="Full path to model config JSON")
    model_group.add_argument("--list-models", "-l", action="store_true", help="List available models")

    parser.add_argument(
        "--model-lib", required=False, help="Path to Model_Lib directory (required for --model and --list-models)"
    )
    parser.add_argument("--config", "-c", required=False, help="Path to hardware config TOML (required for analysis)")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--input-seq", "-i", type=int, default=2048, help="Input sequence length (default: 2048)")
    parser.add_argument("--output-seq", "-o", type=int, default=128, help="Output sequence length (default: 128)")
    parser.add_argument("--device-num", "-d", type=int, default=1, help="Number of devices (default: 1)")
    parser.add_argument("--no-flash-attn", action="store_true", help="Disable flash attention (use standard attention)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress configuration output")

    args = parser.parse_args()

    if args.list_models:
        if not args.model_lib:
            parser.error("--model-lib is required for --list-models")
        model_lib_path = Path(args.model_lib)
        print("Available models:")
        for model in list_available_models(model_lib_path):
            print(f"  {model}")
        return

    # Validate required args
    if not args.config:
        parser.error("--config is required for analysis")

    # Resolve model path
    if args.model:
        if not args.model_lib:
            parser.error("--model-lib is required when using --model")
        model_lib_path = Path(args.model_lib)
        model_path = str(resolve_model_path(args.model, model_lib_path))
    else:
        model_path = args.model_path

    # Load memory config
    memory_config = load_memory_config_from_toml(args.config)

    # Create and run model
    model = LLMMemoryModel(
        model_config_path=model_path,
        memory_config=memory_config,
        batch_size=args.batch_size,
        input_seq_len=args.input_seq,
        output_seq_len=args.output_seq,
        device_num=args.device_num,
        use_flash_attention=not args.no_flash_attn,
    )

    if not args.quiet:
        model.print_config()

    analysis = model.analyze()

    if args.json:
        print(json.dumps(analysis.to_dict(), indent=2))
    else:
        model.print_analysis(analysis)


if __name__ == "__main__":
    main()
