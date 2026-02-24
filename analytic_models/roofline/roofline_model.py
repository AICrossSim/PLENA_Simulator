"""
Roofline Model for PLENA Simulator.

Combines memory and performance models to analyze computational intensity
and determine if workloads are memory-bound or compute-bound.

The roofline model plots:
- X-axis: Operational Intensity (FLOPs/byte)
- Y-axis: Performance (FLOPs/second)
- Memory bandwidth ceiling (sloped line from origin)
- Compute ceiling (horizontal line at peak compute)
- Ridge point: Where the two ceilings meet

Usage:
    python roofline_model.py --model llama-3.1-8b --model-lib ./Model_Lib --config ./plena_settings.toml --isa-lib ./customISA_lib.json
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "memory"))
sys.path.insert(0, str(Path(__file__).parent.parent / "performance"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utilisation"))

from memory_model import MemoryConfig, MemoryModel, MemoryTraffic, load_memory_config_from_toml
from perf_model import HardwareConfig, PerfModel, load_hardware_config_from_toml
from utilisation_model import PLENAUtilization


# =============================================================================
# Data Classes for Roofline Analysis
# =============================================================================


@dataclass
class UtilizationMetrics:
    """Hardware utilization metrics from utilization model."""

    # Attention utilization
    attention_attainable: float = 0.0
    attention_theoretical: float = 0.0
    attention_utilization: float = 0.0

    # FFN utilization
    ffn_attainable: float = 0.0
    ffn_theoretical: float = 0.0
    ffn_utilization: float = 0.0

    # Overall
    total_attainable: float = 0.0
    total_theoretical: float = 0.0
    overall_utilization: float = 0.0

    def compute_overall(self):
        """Compute overall utilization from components."""
        self.total_attainable = self.attention_attainable + self.ffn_attainable
        self.total_theoretical = self.attention_theoretical + self.ffn_theoretical
        if self.total_theoretical > 0:
            self.overall_utilization = self.total_attainable / self.total_theoretical


@dataclass
class OperationProfile:
    """Profile for a single operation or phase."""

    name: str = ""

    # Compute metrics
    flops: int = 0  # Total floating-point operations
    execution_cycles: int = 0  # Cycles from performance model

    # Memory metrics
    memory_traffic: MemoryTraffic = field(default_factory=MemoryTraffic)

    # Utilization metrics (from utilization model)
    utilization: UtilizationMetrics = field(default_factory=UtilizationMetrics)

    # Derived metrics (computed)
    operational_intensity: float = 0.0  # FLOPs/byte
    achieved_flops_per_cycle: float = 0.0  # FLOPs/cycle
    achieved_flops_per_second: float = 0.0  # FLOPs/second

    # Effective metrics (accounting for utilization)
    effective_flops: float = 0.0  # FLOPs adjusted by utilization
    effective_operational_intensity: float = 0.0  # Effective FLOPs/byte

    def compute_metrics(self, frequency_hz: float):
        """Compute derived metrics."""
        if self.memory_traffic.total_bytes > 0:
            self.operational_intensity = self.flops / self.memory_traffic.total_bytes

        if self.execution_cycles > 0:
            self.achieved_flops_per_cycle = self.flops / self.execution_cycles
            execution_time = self.execution_cycles / frequency_hz
            self.achieved_flops_per_second = self.flops / execution_time

        # Compute effective metrics using utilization
        if self.utilization.overall_utilization > 0:
            self.effective_flops = self.flops * self.utilization.overall_utilization
            if self.memory_traffic.total_bytes > 0:
                self.effective_operational_intensity = self.effective_flops / self.memory_traffic.total_bytes


@dataclass
class RooflinePoint:
    """A point on the roofline plot."""

    name: str
    operational_intensity: float  # FLOPs/byte (x-axis)
    achieved_performance: float  # FLOPs/second (y-axis)
    is_memory_bound: bool = True
    bottleneck_ratio: float = 0.0  # How close to the roofline (0-1)


@dataclass
class HardwareLimits:
    """Hardware performance limits for roofline."""

    # Peak compute (FLOPs/second) - theoretical maximum
    peak_flops_per_second: float = 0.0
    peak_flops_per_cycle: float = 0.0

    # Effective peak compute (accounting for typical utilization)
    effective_peak_flops_per_second: float = 0.0
    hardware_utilization: float = 1.0  # Utilization factor (0-1)

    # Peak memory bandwidth (bytes/second)
    peak_memory_bandwidth: float = 0.0  # bytes/second

    # Ridge point (where memory and compute ceilings meet)
    ridge_point_intensity: float = 0.0  # FLOPs/byte (theoretical)
    effective_ridge_point_intensity: float = 0.0  # FLOPs/byte (accounting for utilization)

    # Frequency
    frequency_hz: float = 1e9


@dataclass
class RooflineAnalysis:
    """Complete roofline analysis results."""

    hardware_limits: HardwareLimits = field(default_factory=HardwareLimits)

    # Phase analysis
    prefill_profile: OperationProfile = field(default_factory=OperationProfile)
    decode_profile: OperationProfile = field(default_factory=OperationProfile)

    # Roofline points
    prefill_point: RooflinePoint = field(default_factory=lambda: RooflinePoint("prefill", 0, 0))
    decode_point: RooflinePoint = field(default_factory=lambda: RooflinePoint("decode", 0, 0))

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "hardware_limits": {
                "peak_tflops": self.hardware_limits.peak_flops_per_second / 1e12,
                "effective_peak_tflops": self.hardware_limits.effective_peak_flops_per_second / 1e12,
                "peak_flops_per_cycle": self.hardware_limits.peak_flops_per_cycle,
                "peak_memory_bandwidth_gbps": self.hardware_limits.peak_memory_bandwidth / 1e9,
                "ridge_point_intensity": self.hardware_limits.ridge_point_intensity,
                "effective_ridge_point_intensity": self.hardware_limits.effective_ridge_point_intensity,
                "frequency_ghz": self.hardware_limits.frequency_hz / 1e9,
            },
            "prefill": {
                "flops": self.prefill_profile.flops,
                "effective_flops": self.prefill_profile.effective_flops,
                "memory_bytes": self.prefill_profile.memory_traffic.total_bytes,
                "execution_cycles": self.prefill_profile.execution_cycles,
                "operational_intensity": self.prefill_profile.operational_intensity,
                "effective_operational_intensity": self.prefill_profile.effective_operational_intensity,
                "achieved_tflops": self.prefill_profile.achieved_flops_per_second / 1e12,
                "is_memory_bound": self.prefill_point.is_memory_bound,
                "bottleneck_ratio": self.prefill_point.bottleneck_ratio,
                "utilization": {
                    "attention": self.prefill_profile.utilization.attention_utilization,
                    "ffn": self.prefill_profile.utilization.ffn_utilization,
                    "overall": self.prefill_profile.utilization.overall_utilization,
                },
            },
            "decode": {
                "flops": self.decode_profile.flops,
                "effective_flops": self.decode_profile.effective_flops,
                "memory_bytes": self.decode_profile.memory_traffic.total_bytes,
                "execution_cycles": self.decode_profile.execution_cycles,
                "operational_intensity": self.decode_profile.operational_intensity,
                "effective_operational_intensity": self.decode_profile.effective_operational_intensity,
                "achieved_tflops": self.decode_profile.achieved_flops_per_second / 1e12,
                "is_memory_bound": self.decode_point.is_memory_bound,
                "bottleneck_ratio": self.decode_point.bottleneck_ratio,
                "utilization": {
                    "attention": self.decode_profile.utilization.attention_utilization,
                    "ffn": self.decode_profile.utilization.ffn_utilization,
                    "overall": self.decode_profile.utilization.overall_utilization,
                },
            },
        }


# =============================================================================
# Roofline Model
# =============================================================================


class RooflineModel:
    """
    Roofline model combining memory and performance analysis.

    Computes operational intensity and determines memory-bound vs compute-bound
    for LLM inference phases (prefill and decode).
    """

    def __init__(
        self,
        model_config_path: str,
        hardware_config: HardwareConfig,
        memory_config: MemoryConfig,
        custom_isa_path: str,
        batch_size: int = 1,
        input_seq_len: int = 2048,
        output_seq_len: int = 128,
        device_num: int = 1,
        frequency_hz: float = 1e9,
        partitioned_matrix: bool = True,
    ):
        """
        Initialize RooflineModel.

        Args:
            model_config_path: Path to model config JSON
            hardware_config: PLENA hardware configuration
            memory_config: PLENA memory configuration
            custom_isa_path: Path to customISA_lib.json
            batch_size: Inference batch size
            input_seq_len: Input/prompt sequence length
            output_seq_len: Output/generation sequence length (for decode kv_size)
            device_num: Number of devices for parallelism
            frequency_hz: Clock frequency
            partitioned_matrix: Use partitioned matrix optimization for attention
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

        # Inference parameters
        self.batch_size = batch_size
        self.device_batch_size = batch_size // device_num
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.device_num = device_num
        self.frequency = frequency_hz
        self.partitioned_matrix = partitioned_matrix

        # Hardware and memory configs
        self.hardware_config = hardware_config
        self.memory_config = memory_config

        # Initialize models
        self.perf = PerfModel(hardware_config, custom_isa_path)
        self.mem = MemoryModel(memory_config)

        # Initialize utilization model
        # Convert HardwareConfig to dict for PLENAUtilization
        util_config = {
            "MLEN": hardware_config.MLEN,
            "BLEN": hardware_config.BLEN,
            "VLEN": hardware_config.VLEN,
        }
        self.util = PLENAUtilization(util_config)

        # Model name
        self.model_name = Path(model_config_path).stem

    def compute_hardware_limits(self, utilization_factor: float = 1.0) -> HardwareLimits:
        """
        Compute hardware performance limits.

        Args:
            utilization_factor: Hardware utilization factor (0-1) to compute effective peak
        """
        limits = HardwareLimits()
        limits.frequency_hz = self.frequency

        # Peak compute: MLEN x MLEN MACs per cycle for matrix unit
        # Each MAC = 2 FLOPs (multiply + add)
        mlen = self.hardware_config.MLEN
        limits.peak_flops_per_cycle = mlen * mlen * 2  # MACs * 2

        limits.peak_flops_per_second = limits.peak_flops_per_cycle * self.frequency

        # Effective peak accounting for utilization
        limits.hardware_utilization = utilization_factor
        limits.effective_peak_flops_per_second = limits.peak_flops_per_second * utilization_factor

        # Peak memory bandwidth from memory config
        limits.peak_memory_bandwidth = self.mem.compute_peak_bandwidth(self.frequency) * 1e9  # Convert GB/s to bytes/s

        # Ridge point: where memory ceiling meets compute ceiling
        # peak_flops = peak_bw * intensity => intensity = peak_flops / peak_bw
        if limits.peak_memory_bandwidth > 0:
            limits.ridge_point_intensity = limits.peak_flops_per_second / limits.peak_memory_bandwidth
            limits.effective_ridge_point_intensity = limits.effective_peak_flops_per_second / limits.peak_memory_bandwidth

        return limits

    # -------------------------------------------------------------------------
    # Utilization Computation Methods (from utilization model)
    # -------------------------------------------------------------------------

    def compute_prefill_utilization(self) -> UtilizationMetrics:
        """Compute hardware utilization for prefill phase using utilization model."""
        mode = "prefill"
        metrics = UtilizationMetrics()

        # Projection utilization
        proj_att, proj_theo = self.util.projection_utilization(
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.input_seq_len,
            self.device_batch_size,
            mode,
        )

        # Flash attention utilization
        fa_att, fa_theo = self.util.flash_attention_utilization(
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.input_seq_len,
            self.input_seq_len,  # kv_size = input_seq_len for prefill
            self.device_batch_size,
            mode,
            self.partitioned_matrix,
        )

        # Combined attention utilization
        metrics.attention_attainable = proj_att + fa_att
        metrics.attention_theoretical = proj_theo + fa_theo
        if metrics.attention_theoretical > 0:
            metrics.attention_utilization = metrics.attention_attainable / metrics.attention_theoretical

        # FFN utilization
        ffn_att, ffn_theo = self.util.ffn_utilization(
            self.hidden_size,
            self.intermediate_size,
            self.input_seq_len,
            self.device_batch_size,
            mode,
        )
        metrics.ffn_attainable = ffn_att
        metrics.ffn_theoretical = ffn_theo
        if metrics.ffn_theoretical > 0:
            metrics.ffn_utilization = metrics.ffn_attainable / metrics.ffn_theoretical

        # Scale by number of layers
        metrics.attention_attainable *= self.num_hidden_layers
        metrics.attention_theoretical *= self.num_hidden_layers
        metrics.ffn_attainable *= self.num_hidden_layers
        metrics.ffn_theoretical *= self.num_hidden_layers

        # Compute overall
        metrics.compute_overall()

        return metrics

    def compute_decode_utilization(self, kv_size: int) -> UtilizationMetrics:
        """Compute hardware utilization for single token decode using utilization model."""
        mode = "decode"
        metrics = UtilizationMetrics()

        # Projection utilization
        proj_att, proj_theo = self.util.projection_utilization(
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            1,  # seq_len = 1 for decode
            self.device_batch_size,
            mode,
        )

        # Flash attention utilization
        fa_att, fa_theo = self.util.flash_attention_utilization(
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            1,  # seq_len = 1 for decode
            kv_size,
            self.device_batch_size,
            mode,
            self.partitioned_matrix,
        )

        # Combined attention utilization
        metrics.attention_attainable = proj_att + fa_att
        metrics.attention_theoretical = proj_theo + fa_theo
        if metrics.attention_theoretical > 0:
            metrics.attention_utilization = metrics.attention_attainable / metrics.attention_theoretical

        # FFN utilization
        ffn_att, ffn_theo = self.util.ffn_utilization(
            self.hidden_size,
            self.intermediate_size,
            1,  # seq_len = 1 for decode
            self.device_batch_size,
            mode,
        )
        metrics.ffn_attainable = ffn_att
        metrics.ffn_theoretical = ffn_theo
        if metrics.ffn_theoretical > 0:
            metrics.ffn_utilization = metrics.ffn_attainable / metrics.ffn_theoretical

        # Scale by number of layers
        metrics.attention_attainable *= self.num_hidden_layers
        metrics.attention_theoretical *= self.num_hidden_layers
        metrics.ffn_attainable *= self.num_hidden_layers
        metrics.ffn_theoretical *= self.num_hidden_layers

        # Compute overall
        metrics.compute_overall()

        return metrics

    # -------------------------------------------------------------------------
    # FLOPs Computation Methods
    # -------------------------------------------------------------------------

    def compute_embedding_flops(self, seq_len: int, batch_size: int) -> int:
        """Embedding lookup FLOPs (essentially 0, just memory access)."""
        return 0

    def compute_rms_norm_flops(self, hidden_size: int, seq_len: int, batch_size: int) -> int:
        """
        RMSNorm FLOPs.

        RMSNorm: x / sqrt(mean(x^2) + eps) * gamma
        Per element: 1 square, 1 add (for mean), 1 div, 1 mul = ~4 ops
        But parallelized: O(hidden_size) per token
        """
        # Per token: hidden_size squares, 1 mean (hidden_size adds + 1 div),
        # 1 sqrt, hidden_size divs, hidden_size muls
        tokens = seq_len * batch_size
        flops_per_token = hidden_size * 2 + hidden_size + 1 + hidden_size * 2
        return tokens * flops_per_token

    def compute_projection_flops(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        batch_size: int,
    ) -> int:
        """
        Q, K, V, O projection FLOPs.

        Q: (batch*seq, hidden) @ (hidden, num_heads*head_dim) = 2 * batch*seq * hidden * num_heads*head_dim
        K: (batch*seq, hidden) @ (hidden, num_kv_heads*head_dim)
        V: (batch*seq, hidden) @ (hidden, num_kv_heads*head_dim)
        O: (batch*seq, num_heads*head_dim) @ (num_heads*head_dim, hidden)
        """
        tokens = seq_len * batch_size

        # Q projection
        q_flops = 2 * tokens * hidden_size * (num_attention_heads * head_dim)

        # K projection
        k_flops = 2 * tokens * hidden_size * (num_kv_heads * head_dim)

        # V projection
        v_flops = 2 * tokens * hidden_size * (num_kv_heads * head_dim)

        # O projection (output projection after attention)
        o_flops = 2 * tokens * (num_attention_heads * head_dim) * hidden_size

        return q_flops + k_flops + v_flops + o_flops

    def compute_attention_flops(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        kv_size: int,
        batch_size: int,
    ) -> int:
        """
        Self-attention FLOPs.

        QK^T: (batch, heads, seq, head_dim) @ (batch, heads, head_dim, kv_size) = 2 * batch * heads * seq * head_dim * kv_size
        Softmax: ~5 ops per element (exp, sum, div) = 5 * batch * heads * seq * kv_size
        PV: (batch, heads, seq, kv_size) @ (batch, heads, kv_size, head_dim) = 2 * batch * heads * seq * kv_size * head_dim
        """
        # GQA: kv_heads are repeated for q_heads
        gqa_ratio = num_attention_heads // num_kv_heads

        # QK^T
        qkt_flops = 2 * batch_size * num_attention_heads * seq_len * head_dim * kv_size

        # Softmax (approximately)
        softmax_flops = 5 * batch_size * num_attention_heads * seq_len * kv_size

        # PV
        pv_flops = 2 * batch_size * num_attention_heads * seq_len * kv_size * head_dim

        return qkt_flops + softmax_flops + pv_flops

    def compute_ffn_flops(
        self,
        hidden_size: int,
        intermediate_size: int,
        seq_len: int,
        batch_size: int,
    ) -> int:
        """
        Feed-forward network FLOPs.

        Gate: (batch*seq, hidden) @ (hidden, intermediate) = 2 * tokens * hidden * intermediate
        Up: (batch*seq, hidden) @ (hidden, intermediate) = 2 * tokens * hidden * intermediate
        SiLU: ~4 ops per element = 4 * tokens * intermediate
        Multiply gate * up: tokens * intermediate
        Down: (batch*seq, intermediate) @ (intermediate, hidden) = 2 * tokens * intermediate * hidden
        """
        tokens = seq_len * batch_size

        gate_flops = 2 * tokens * hidden_size * intermediate_size
        up_flops = 2 * tokens * hidden_size * intermediate_size
        silu_flops = 4 * tokens * intermediate_size
        mul_flops = tokens * intermediate_size
        down_flops = 2 * tokens * intermediate_size * hidden_size

        return gate_flops + up_flops + silu_flops + mul_flops + down_flops

    def compute_lm_head_flops(self, hidden_size: int, vocab_size: int, batch_size: int) -> int:
        """
        LM head FLOPs.

        (batch, hidden) @ (hidden, vocab) = 2 * batch * hidden * vocab
        """
        return 2 * batch_size * hidden_size * vocab_size

    # -------------------------------------------------------------------------
    # Phase-level FLOPs and Memory Traffic
    # -------------------------------------------------------------------------

    def compute_prefill_profile(self) -> OperationProfile:
        """Compute FLOPs, memory traffic, and cycles for prefill phase."""
        profile = OperationProfile(name="prefill")
        mode = "prefill"
        kv_size = self.input_seq_len

        # Compute total FLOPs
        total_flops = 0

        # Embedding (minimal FLOPs)
        total_flops += self.compute_embedding_flops(self.input_seq_len, self.device_batch_size)

        # Per-layer computations
        for _ in range(self.num_hidden_layers):
            # RMS norm (pre-attention)
            total_flops += self.compute_rms_norm_flops(self.hidden_size, self.input_seq_len, self.device_batch_size)

            # QKV + O projections
            total_flops += self.compute_projection_flops(
                self.hidden_size,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.input_seq_len,
                self.device_batch_size,
            )

            # Attention
            total_flops += self.compute_attention_flops(
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.input_seq_len,
                kv_size,
                self.device_batch_size,
            )

            # RMS norm (pre-FFN)
            total_flops += self.compute_rms_norm_flops(self.hidden_size, self.input_seq_len, self.device_batch_size)

            # FFN
            total_flops += self.compute_ffn_flops(
                self.hidden_size, self.intermediate_size, self.input_seq_len, self.device_batch_size
            )

        # LM head
        total_flops += self.compute_lm_head_flops(self.hidden_size, self.vocab_size, self.device_batch_size)

        profile.flops = total_flops

        # Compute memory traffic
        total_traffic = MemoryTraffic()

        # Embedding traffic
        total_traffic += self.mem.embedding_traffic(self.hidden_size, self.input_seq_len, self.device_batch_size, mode)

        # Per-layer traffic
        for _ in range(self.num_hidden_layers):
            # Projection traffic (weights + KV cache write)
            total_traffic += self.mem.projection_traffic(
                self.hidden_size,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.input_seq_len,
                self.device_batch_size,
                mode,
            )

            # Attention traffic (KV cache reads)
            total_traffic += self.mem.flash_attention_traffic(
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.input_seq_len,
                kv_size,
                self.device_batch_size,
                mode,
            )

            # Output projection traffic
            total_traffic += self.mem.output_projection_traffic(
                self.hidden_size,
                self.num_attention_heads,
                self.head_dim,
                self.input_seq_len,
                self.device_batch_size,
                mode,
            )

            # FFN traffic
            total_traffic += self.mem.ffn_traffic(
                self.hidden_size, self.intermediate_size, self.input_seq_len, self.device_batch_size, mode
            )

        # LM head traffic
        total_traffic += self.mem.lm_head_traffic(self.hidden_size, self.vocab_size)

        profile.memory_traffic = total_traffic

        # Compute execution cycles
        total_cycles = 0

        total_cycles += self.perf.embeddings(self.hidden_size, self.input_seq_len, self.device_batch_size, mode)

        rms = self.perf.rms_layer(self.hidden_size, self.input_seq_len, self.device_batch_size, mode)
        proj = self.perf.projection(
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.input_seq_len,
            self.device_batch_size,
            mode,
        )
        attn = self.perf.flash_attention(
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.input_seq_len,
            kv_size,
            self.device_batch_size,
            mode,
        )
        res = self.perf.residual(self.hidden_size, self.input_seq_len, self.device_batch_size, mode)
        ffn = self.perf.feed_forward(
            self.hidden_size, self.intermediate_size, self.input_seq_len, self.device_batch_size, mode
        )

        transformer_block_cycles = rms + proj + attn + res + rms + ffn
        total_cycles += transformer_block_cycles * self.num_hidden_layers
        total_cycles += self.perf.lm_head(self.hidden_size, self.vocab_size, self.device_batch_size)

        profile.execution_cycles = total_cycles

        # Compute utilization metrics
        profile.utilization = self.compute_prefill_utilization()

        # Compute derived metrics
        profile.compute_metrics(self.frequency)

        return profile

    def compute_single_token_decode_profile(self, output_token_size: int = 0) -> OperationProfile:
        """
        Compute FLOPs, memory traffic, and cycles for single token decode.

        Args:
            output_token_size: Number of tokens already generated (kv_size = input_seq_len + output_token_size)
        """
        profile = OperationProfile(name="decode")
        mode = "decode"
        kv_size = self.input_seq_len + output_token_size

        # Compute total FLOPs (seq_len = 1 for decode)
        total_flops = 0

        # Per-layer computations
        for _ in range(self.num_hidden_layers):
            # RMS norm (pre-attention)
            total_flops += self.compute_rms_norm_flops(self.hidden_size, 1, self.device_batch_size)

            # QKV + O projections
            total_flops += self.compute_projection_flops(
                self.hidden_size,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                1,
                self.device_batch_size,
            )

            # Attention (attends to all kv_size tokens)
            total_flops += self.compute_attention_flops(
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                1,
                kv_size,
                self.device_batch_size,
            )

            # RMS norm (pre-FFN)
            total_flops += self.compute_rms_norm_flops(self.hidden_size, 1, self.device_batch_size)

            # FFN
            total_flops += self.compute_ffn_flops(self.hidden_size, self.intermediate_size, 1, self.device_batch_size)

        # LM head
        total_flops += self.compute_lm_head_flops(self.hidden_size, self.vocab_size, self.device_batch_size)

        profile.flops = total_flops

        # Compute memory traffic
        total_traffic = MemoryTraffic()

        # Per-layer traffic
        for _ in range(self.num_hidden_layers):
            # Projection traffic
            total_traffic += self.mem.projection_traffic(
                self.hidden_size,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                1,
                self.device_batch_size,
                mode,
            )

            # Attention traffic
            total_traffic += self.mem.flash_attention_traffic(
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                1,
                kv_size,
                self.device_batch_size,
                mode,
            )

            # Output projection traffic
            total_traffic += self.mem.output_projection_traffic(
                self.hidden_size,
                self.num_attention_heads,
                self.head_dim,
                1,
                self.device_batch_size,
                mode,
            )

            # FFN traffic
            total_traffic += self.mem.ffn_traffic(
                self.hidden_size, self.intermediate_size, 1, self.device_batch_size, mode
            )

        # LM head traffic
        total_traffic += self.mem.lm_head_traffic(self.hidden_size, self.vocab_size)

        profile.memory_traffic = total_traffic

        # Compute execution cycles (matches compute_single_token_decode_time structure)
        total_cycles = 0

        rms = self.perf.rms_layer(self.hidden_size, 1, self.device_batch_size, mode)
        proj = self.perf.projection(
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            1,
            self.device_batch_size,
            mode,
        )
        attn = self.perf.flash_attention(
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            1,
            kv_size,
            self.device_batch_size,
            mode,
        )
        res = self.perf.residual(self.hidden_size, 1, self.device_batch_size, mode)
        ffn = self.perf.feed_forward(self.hidden_size, self.intermediate_size, 1, self.device_batch_size, mode)

        transformer_block_cycles = rms + proj + attn + res + rms + ffn
        total_cycles = transformer_block_cycles * self.num_hidden_layers

        profile.execution_cycles = total_cycles

        # Compute utilization metrics
        profile.utilization = self.compute_decode_utilization(kv_size)

        # Compute derived metrics
        profile.compute_metrics(self.frequency)

        return profile

    def analyze(self, decode_kv_size: int | None = None) -> RooflineAnalysis:
        """
        Perform complete roofline analysis.

        Args:
            decode_kv_size: KV cache size for decode analysis (default: input_seq_len)

        Returns:
            RooflineAnalysis with all metrics
        """
        if decode_kv_size is None:
            decode_kv_size = self.input_seq_len

        result = RooflineAnalysis()

        # Prefill analysis (compute first to get utilization)
        result.prefill_profile = self.compute_prefill_profile()

        # Decode analysis (single token)
        output_tokens = decode_kv_size - self.input_seq_len
        result.decode_profile = self.compute_single_token_decode_profile(output_tokens)

        # Hardware limits - use average utilization from both phases
        avg_utilization = (
            result.prefill_profile.utilization.overall_utilization
            + result.decode_profile.utilization.overall_utilization
        ) / 2
        result.hardware_limits = self.compute_hardware_limits(utilization_factor=avg_utilization)
        limits = result.hardware_limits

        # Create roofline points
        result.prefill_point = self._create_roofline_point("prefill", result.prefill_profile, limits)
        result.decode_point = self._create_roofline_point("decode", result.decode_profile, limits)

        return result

    def _create_roofline_point(
        self, name: str, profile: OperationProfile, limits: HardwareLimits
    ) -> RooflinePoint:
        """Create a roofline point from operation profile."""
        # Use effective operational intensity that accounts for utilization
        effective_intensity = profile.effective_operational_intensity or profile.operational_intensity

        point = RooflinePoint(
            name=name,
            operational_intensity=effective_intensity,
            achieved_performance=profile.achieved_flops_per_second,
        )

        # Determine if memory-bound or compute-bound using effective ridge point
        point.is_memory_bound = effective_intensity < limits.effective_ridge_point_intensity

        # Compute bottleneck ratio (how close to the roofline ceiling)
        if point.is_memory_bound:
            # Memory-bound: ceiling = bandwidth * intensity
            ceiling = limits.peak_memory_bandwidth * effective_intensity
        else:
            # Compute-bound: ceiling = effective peak compute
            ceiling = limits.effective_peak_flops_per_second

        if ceiling > 0:
            point.bottleneck_ratio = profile.achieved_flops_per_second / ceiling

        return point

    def print_analysis(self, analysis: RooflineAnalysis):
        """Print formatted roofline analysis."""
        print("\n" + "=" * 70)
        print("ROOFLINE ANALYSIS")
        print("=" * 70)

        # Hardware limits
        limits = analysis.hardware_limits
        print("\n[HARDWARE LIMITS]")
        print("-" * 70)
        print(f"  Peak Compute:         {limits.peak_flops_per_second / 1e12:8.2f} TFLOP/s (theoretical)")
        print(f"  Effective Peak:       {limits.effective_peak_flops_per_second / 1e12:8.2f} TFLOP/s (with utilization)")
        print(f"  HW Utilization:       {limits.hardware_utilization * 100:8.1f}%")
        print(f"  Peak FLOPs/cycle:     {limits.peak_flops_per_cycle:8.0f}")
        print(f"  Peak Memory BW:       {limits.peak_memory_bandwidth / 1e9:8.2f} GB/s")
        print(f"  Ridge Point:          {limits.ridge_point_intensity:8.2f} FLOPs/byte (theoretical)")
        print(f"  Effective Ridge:      {limits.effective_ridge_point_intensity:8.2f} FLOPs/byte")
        print(f"  Frequency:            {limits.frequency_hz / 1e9:8.2f} GHz")

        # Prefill analysis
        pf = analysis.prefill_profile
        pp = analysis.prefill_point
        print("\n[PREFILL PHASE]")
        print("-" * 70)
        print(f"  Total FLOPs:          {pf.flops / 1e9:12.2f} GFLOPs")
        print(f"  Effective FLOPs:      {pf.effective_flops / 1e9:12.2f} GFLOPs")
        print(f"  Memory Traffic:       {pf.memory_traffic.total_bytes / 1e9:12.2f} GB")
        print(f"    Read:               {pf.memory_traffic.read_bytes / 1e9:12.2f} GB")
        print(f"    Write:              {pf.memory_traffic.write_bytes / 1e9:12.2f} GB")
        print(f"  Execution Cycles:     {pf.execution_cycles:12,}")
        print(f"  Operational Intensity:{pf.operational_intensity:12.2f} FLOPs/byte")
        print(f"  Effective Intensity:  {pf.effective_operational_intensity:12.2f} FLOPs/byte")
        print(f"  Achieved Performance: {pf.achieved_flops_per_second / 1e12:12.2f} TFLOP/s")
        print(f"  Bound Type:           {'MEMORY-BOUND' if pp.is_memory_bound else 'COMPUTE-BOUND'}")
        print(f"  Efficiency:           {pp.bottleneck_ratio * 100:11.1f}%")
        print("  Hardware Utilization:")
        print(f"    Attention:          {pf.utilization.attention_utilization * 100:8.1f}%")
        print(f"    FFN:                {pf.utilization.ffn_utilization * 100:8.1f}%")
        print(f"    Overall:            {pf.utilization.overall_utilization * 100:8.1f}%")

        # Decode analysis
        dc = analysis.decode_profile
        dp = analysis.decode_point
        print("\n[DECODE PHASE (Single Token)]")
        print("-" * 70)
        print(f"  Total FLOPs:          {dc.flops / 1e9:12.4f} GFLOPs")
        print(f"  Effective FLOPs:      {dc.effective_flops / 1e9:12.4f} GFLOPs")
        print(f"  Memory Traffic:       {dc.memory_traffic.total_bytes / 1e6:12.2f} MB")
        print(f"    Read:               {dc.memory_traffic.read_bytes / 1e6:12.2f} MB")
        print(f"    Write:              {dc.memory_traffic.write_bytes / 1e6:12.2f} MB")
        print(f"  Execution Cycles:     {dc.execution_cycles:12,}")
        print(f"  Operational Intensity:{dc.operational_intensity:12.2f} FLOPs/byte")
        print(f"  Effective Intensity:  {dc.effective_operational_intensity:12.2f} FLOPs/byte")
        print(f"  Achieved Performance: {dc.achieved_flops_per_second / 1e12:12.4f} TFLOP/s")
        print(f"  Bound Type:           {'MEMORY-BOUND' if dp.is_memory_bound else 'COMPUTE-BOUND'}")
        print(f"  Efficiency:           {dp.bottleneck_ratio * 100:11.1f}%")
        print("  Hardware Utilization:")
        print(f"    Attention:          {dc.utilization.attention_utilization * 100:8.1f}%")
        print(f"    FFN:                {dc.utilization.ffn_utilization * 100:8.1f}%")
        print(f"    Overall:            {dc.utilization.overall_utilization * 100:8.1f}%")

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
        description="Roofline Model - Analyze memory vs compute boundedness for LLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python roofline_model.py --model llama-3.1-8b --model-lib ./Model_Lib --config ./plena_settings.toml --isa-lib ./customISA_lib.json
  python roofline_model.py --model llama-3.1-8b --model-lib ./Model_Lib --config ./plena_settings.toml --isa-lib ./customISA_lib.json --json
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
    parser.add_argument("--isa-lib", required=False, help="Path to customISA_lib.json (required for analysis)")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--input-seq", "-i", type=int, default=2048, help="Input sequence length (default: 2048)")
    parser.add_argument("--output-seq", "-o", type=int, default=128, help="Output sequence length (default: 128)")
    parser.add_argument("--device-num", "-d", type=int, default=1, help="Number of devices (default: 1)")
    parser.add_argument("--no-partition", action="store_true", help="Disable partitioned matrix optimization")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress detailed output")

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
    if not args.isa_lib:
        parser.error("--isa-lib is required for analysis")

    # Resolve model path
    if args.model:
        if not args.model_lib:
            parser.error("--model-lib is required when using --model")
        model_lib_path = Path(args.model_lib)
        model_path = str(resolve_model_path(args.model, model_lib_path))
    else:
        model_path = args.model_path

    # Load configs
    hardware_config = load_hardware_config_from_toml(args.config)
    memory_config = load_memory_config_from_toml(args.config)

    # Create and run model
    model = RooflineModel(
        model_config_path=model_path,
        hardware_config=hardware_config,
        memory_config=memory_config,
        custom_isa_path=args.isa_lib,
        batch_size=args.batch_size,
        input_seq_len=args.input_seq,
        output_seq_len=args.output_seq,
        device_num=args.device_num,
        partitioned_matrix=not args.no_partition,
    )

    analysis = model.analyze()

    if args.json:
        print(json.dumps(analysis.to_dict(), indent=2))
    else:
        model.print_analysis(analysis)


if __name__ == "__main__":
    main()
