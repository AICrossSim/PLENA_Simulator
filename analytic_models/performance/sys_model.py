"""
PLENA System Model - Fused Performance and Memory Model.

Combines compute and memory modeling into a single unified model.
All computations are modeled at the TILE level, then aggregated into segments.

Output format matches perf_model.py for compatibility with existing plotting tools:
- segments: list of dicts with name, cycles, systolic_utilization, bandwidth_utilization,
  memory_read_bytes, memory_write_bytes, compute_time_us, memory_time_us
"""

from __future__ import annotations

import json
import math

import toml
from pydantic import BaseModel, Field, model_validator


# =============================================================================
# Configuration Classes
# =============================================================================


class SystemConfig(BaseModel):
    """Unified system configuration for PLENA accelerator."""

    # Core hardware dimensions
    MLEN: int = Field(gt=0, description="Matrix unit length")
    BLEN: int = Field(gt=0, description="Block length")
    VLEN: int = Field(gt=0, description="Vector length")
    HLEN: int = Field(gt=0, description="Head dimension length")

    # Memory configuration
    HBM_SIZE: int = Field(gt=0, description="HBM capacity in bytes")
    HBM_WIDTH: int = Field(gt=0, description="HBM bus width in bits")
    MATRIX_SRAM_SIZE: int = Field(gt=0, description="Matrix SRAM size in elements")
    VECTOR_SRAM_SIZE: int = Field(gt=0, description="Vector SRAM size in elements")
    HBM_V_Prefetch_Amount: int = Field(default=1, gt=0, description="HBM vector prefetch amount")

    # Data type specifications (bits per element)
    weight_bits: float = Field(default=8.0, description="Bits per weight element")
    kv_cache_bits: float = Field(default=8.0, description="Bits per KV cache element")
    activation_bits: float = Field(default=16.0, description="Bits per activation element")

    # Allow extra fields for latency parameters
    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def validate_dimensions(self) -> "SystemConfig":
        """Validate hardware dimension relationships."""
        if self.MLEN % self.BLEN != 0:
            raise ValueError(f"MLEN ({self.MLEN}) must be divisible by BLEN ({self.BLEN})")
        return self


def load_system_config_from_toml(toml_path: str) -> SystemConfig:
    """Load unified system configuration from plena_settings.toml."""
    with open(toml_path) as f:
        data = toml.load(f)

    config_dict = {}
    analytic_data = data.get("ANALYTIC", {})

    # Extract CONFIG section values
    config_section = analytic_data.get("CONFIG", {})
    for param_name, val in config_section.items():
        if isinstance(val, dict) and "value" in val:
            config_dict[param_name] = val["value"]

    # Extract LATENCY section values
    latency_section = analytic_data.get("LATENCY", {})
    for param_name, val in latency_section.items():
        if isinstance(val, dict):
            if "dc_lib_en" in val:
                config_dict[param_name] = val["dc_lib_en"]
            elif "value" in val:
                config_dict[param_name] = val["value"]

    # Extract precision specifications
    precision_section = analytic_data.get("PRECISION", {})
    if "HBM_M_WEIGHT_TYPE" in precision_section:
        config_dict["weight_bits"] = _extract_bits(precision_section["HBM_M_WEIGHT_TYPE"])
    if "HBM_V_KV_TYPE" in precision_section:
        config_dict["kv_cache_bits"] = _extract_bits(precision_section["HBM_V_KV_TYPE"])
    elif "HBM_M_KV_TYPE" in precision_section:
        config_dict["kv_cache_bits"] = _extract_bits(precision_section["HBM_M_KV_TYPE"])
    if "HBM_V_ACT_TYPE" in precision_section:
        config_dict["activation_bits"] = _extract_bits(precision_section["HBM_V_ACT_TYPE"])

    return SystemConfig(**config_dict)


def _extract_bits(precision_config: dict) -> float:
    """Extract bits per element from precision config."""
    fmt = precision_config.get("format", "Plain")
    if fmt == "Plain":
        data_type = precision_config.get("DATA_TYPE", {})
        if data_type.get("type") == "Fp":
            return 1 + data_type.get("exponent", 8) + data_type.get("mantissa", 7)
        elif data_type.get("type") == "Int":
            return data_type.get("width", 32)
        return 16
    elif fmt == "Mx":
        block_size = precision_config.get("block", 8)
        elem_config = precision_config.get("ELEM", {})
        scale_config = precision_config.get("SCALE", {})
        if elem_config.get("type") == "Fp":
            elem_bits = 1 + elem_config.get("exponent", 4) + elem_config.get("mantissa", 3)
        else:
            elem_bits = 8
        if scale_config.get("type") == "Fp":
            scale_bits = (
                (0 if not scale_config.get("sign", False) else 1)
                + scale_config.get("exponent", 8)
                + scale_config.get("mantissa", 0)
            )
        else:
            scale_bits = 8
        return elem_bits + scale_bits / block_size
    return 16


# =============================================================================
# System Model
# =============================================================================


class SystemModel:
    """
    Unified system model with tile-level compute and memory analysis.

    Output format matches perf_model.py:
    {
        "total_cycles": int,
        "segments": [
            {
                "name": str,
                "cycles": int,
                "systolic_utilization": float,
                "bandwidth_utilization": float,
                "memory_read_bytes": int,
                "memory_write_bytes": int,
                "compute_time_us": float,
                "memory_time_us": float,
            },
            ...
        ],
        "tiles": {...}  # Optional tile-level detail
    }
    """

    def __init__(
        self,
        config: SystemConfig,
        isa_lib_path: str,
        frequency_hz: float = 1e9,
    ):
        """
        Initialize SystemModel.

        Args:
            config: Unified system configuration
            isa_lib_path: Path to customISA_lib.json
            frequency_hz: Clock frequency in Hz
        """
        self.config = config
        self.frequency_hz = frequency_hz

        # Hardware dimensions
        self.mlen = config.MLEN
        self.blen = config.BLEN
        self.vlen = config.VLEN
        self.hlen = config.HLEN

        # Memory parameters
        self.hbm_width_bytes = config.HBM_WIDTH
        self.vector_sram_bytes = config.VECTOR_SRAM_SIZE * self.vlen * (config.activation_bits / 8)
        self.prefetch_v_amount = config.HBM_V_Prefetch_Amount

        # Data type bits
        self.weight_bits = config.weight_bits
        self.kv_cache_bits = config.kv_cache_bits
        self.activation_bits = config.activation_bits

        # Peak bandwidth (bytes per microsecond)
        self.peak_bw_bytes_per_us = self.hbm_width_bytes * frequency_hz / 1e6

        # Build instruction latencies
        self.instr = self._build_instruction_latencies(isa_lib_path)

    def _build_instruction_latencies(self, isa_lib_path: str) -> dict[str, int]:
        """Build instruction latency map from customISA_lib.json."""
        with open(isa_lib_path) as f:
            custom_isa_lib = json.load(f)

        configs = self.config.model_dump()
        configs["SA_ACC_CYCLES"] = int(math.log2(self.mlen / self.blen) + 1)

        latencies = {}
        for instr_name, instr_data in custom_isa_lib.items():
            if "pipelined" in instr_data:
                latencies[instr_name] = eval(instr_data["pipelined"], {}, configs)

        return latencies

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _bits_to_bytes(self, num_elements: int, bits_per_element: float) -> int:
        """Convert element count to bytes."""
        return math.ceil(num_elements * bits_per_element / 8)

    def _cycles_to_us(self, cycles: int) -> float:
        """Convert cycles to microseconds."""
        return cycles / self.frequency_hz * 1e6

    def _bytes_to_us(self, total_bytes: int, actual_bw_amt: int) -> float:
        """Convert bytes to transfer time in microseconds."""
        if total_bytes <= 0:
            return 0.0
        actual_bw_us = actual_bw_amt * self.frequency_hz / 1e6
        return total_bytes / actual_bw_us

    def _compute_systolic_util(self, mode: str, batch_size: int) -> float:
        """Compute systolic array utilization percentage."""
        if mode == "prefill":
            return 100.0
        else:
            return min(batch_size / self.blen, 1.0) * 100.0

    def _compute_bw_util(self, actual_bw_amt: int) -> float:
        """Compute bandwidth utilization percentage."""
        peak_bw_amt = self.hbm_width_bytes
        return min((actual_bw_amt / peak_bw_amt) * 100.0, 100.0)

    # -------------------------------------------------------------------------
    # RMS Norm Layer
    # -------------------------------------------------------------------------

    def rms_norm(
        self,
        hidden_size: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> dict:
        """RMSNorm layer analysis at tile level.

        Optimized: computes aggregates directly.
        """
        return {"total_cycles": 0, "segments": [], "tiles": {}}
    # -------------------------------------------------------------------------
    # QKV Projection Layer
    # -------------------------------------------------------------------------

    def projection(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> dict:
        """QKV projection + RoPE layer analysis at tile level.

        Optimized: computes aggregates directly instead of creating tile objects.
        """
        return {"total_cycles": 0, "segments": [], "tiles": {}}

    # -------------------------------------------------------------------------
    # Self-Attention Layer
    # -------------------------------------------------------------------------

    def self_attention(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        kv_size: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> dict:
        """Self-attention layer analysis at tile level.

        Optimized: computes aggregates directly instead of creating millions of tile objects.
        """
        m_util = self._compute_systolic_util(mode, seq_len)
        kv_head_loop = num_kv_heads
        inner_q_head_loop = num_attention_heads // num_kv_heads

        if mode == "prefill":
            # ----- QKT -----
            qkt_total_cycles = (
                    (
                        4
                        + self.instr["M_BTMV"] * math.ceil(seq_len / self.mlen) * math.ceil(kv_size / self.mlen)
                        + self.instr["H_PREFETCH_M"]
                    )
                    * kv_head_loop
                    * math.ceil(inner_q_head_loop / math.ceil((self.mlen / self.hlen)))
                ) * batch_size


            s_bytes = self._bits_to_bytes(seq_len * kv_size * num_attention_heads * batch_size, self.activation_bits)
            qkt_total_read = 0
            qkt_total_write = s_bytes

            # ----- Scaling -----
            scaling_total_cycles = (
                num_attention_heads * seq_len * math.ceil(kv_size / self.vlen) * self.instr["V_BASIC"]
            ) * batch_size
            scaling_total_read = 2 * s_bytes
            scaling_total_write = 2 * s_bytes

            # ----- Softmax -----
            softmax_total_cycles = (
                num_attention_heads * seq_len * math.ceil(kv_size / self.vlen)
                * (6 * self.instr["V_BASIC"] + self.instr["V_EXP_V"] + self.instr["V_RED_MAX"])
            ) * batch_size
            softmax_total_read = 4 * s_bytes
            softmax_total_write = 4 * s_bytes

            # ----- PV -----
            pv_total_cycles = (
                (4 + self.instr["M_MV"] * math.ceil(kv_size / self.mlen))
                * math.ceil(seq_len / self.blen)
                * math.ceil(head_dim / self.blen)
                * num_attention_heads
            ) * batch_size

            v_bytes = self._bits_to_bytes(kv_size * num_kv_heads * head_dim * batch_size, self.kv_cache_bits)
            out_bytes = self._bits_to_bytes(seq_len * num_attention_heads * head_dim * batch_size, self.activation_bits)
            pv_total_read = (s_bytes + v_bytes)
            pv_total_write = out_bytes * batch_size
            print("m_util", m_util)
            quit()

        else:  # decode mode

            # ----- QKT -----
            qkt_total_cycles = (
                (4 + self.instr["M_BTMV"] * math.ceil(kv_size / self.mlen) + self.instr["H_PREFETCH_M"])
                * kv_head_loop
                * math.ceil(inner_q_head_loop / math.ceil((self.mlen / self.hlen)))
            ) * batch_size

            s_bytes = self._bits_to_bytes(kv_size * num_attention_heads * batch_size, self.activation_bits)
            
            qkt_total_read = 0
            qkt_total_write = s_bytes

            # ----- Scaling -----
            scaling_total_cycles = (num_attention_heads * math.ceil(kv_size / self.vlen) * (6 * self.instr["V_BASIC"])) * batch_size
            scaling_total_read = 0
            scaling_total_write = s_bytes

            # ----- Softmax -----
            softmax_total_cycles = (
                (math.ceil(kv_size / self.vlen)
                * (6 * self.instr["V_BASIC"] + self.instr["V_EXP_V"] + self.instr["V_RED_MAX"])
                * num_attention_heads) * batch_size
            )
            softmax_total_read = s_bytes
            softmax_total_write = s_bytes

            # ----- PV -----
            pv_total_cycles = (
                (4 + self.instr["M_MV"] + self.instr["H_PREFETCH_M"])
                * math.ceil(head_dim / self.blen)
                * math.ceil(kv_size / self.mlen)
                * num_attention_heads
            ) * batch_size

            pv_total_read = self._bits_to_bytes(kv_size * num_kv_heads * head_dim, self.kv_cache_bits)
            pv_total_write = s_bytes

        # Build segments with multi-section timing control
        # Each segment can have multiple compute/memory/systolic/bandwidth sections

        # Compute timing values
        qkt_compute_time = self._cycles_to_us(qkt_total_cycles)
        qkt_write_time = self._bytes_to_us(qkt_total_write, self.vlen)

        scaling_compute_time = self._cycles_to_us(scaling_total_cycles)
        # scaling_read_time = self._bytes_to_us(scaling_total_read, self.vlen)
        scaling_write_time = self._bytes_to_us(scaling_total_write, self.vlen)

        softmax_compute_time = self._cycles_to_us(softmax_total_cycles)
        # softmax_read_time = self._bytes_to_us(softmax_total_read, self.vlen)
        softmax_write_time = self._bytes_to_us(softmax_total_write, self.vlen)

        pv_compute_time = self._cycles_to_us(pv_total_cycles)
        pv_write_time = self._bytes_to_us(pv_total_write, self.vlen)

        segments = [
            {
                "name": "QKT",
                "cycles": qkt_total_cycles,
                "memory_read_bytes": qkt_total_read,
                "memory_write_bytes": qkt_total_write,
                # Multiple compute sections (just one here: main compute)
                "compute_sections": [
                    {"offset_us": 0.0, "duration_us": qkt_compute_time},
                ],
                # Multiple memory sections: writeback after compute
                "memory_sections": [
                    {"offset_us": 0, "duration_us": qkt_write_time + qkt_compute_time},
                ],
                # Systolic util during compute phase
                "systolic_sections": [
                    {"offset_us": 0.0, "duration_us": qkt_compute_time, "value": m_util},
                ],
                # Bandwidth util during writeback
                "bandwidth_sections": [
                    {"offset_us": 0, "duration_us": qkt_compute_time, "value": self._compute_bw_util(2 * self.mlen)},
                    {"offset_us": qkt_compute_time, "duration_us": qkt_write_time, "value": self._compute_bw_util(self.vlen)},
                ],
            },
            {
                "name": "scaling",
                "cycles": scaling_total_cycles,
                "memory_read_bytes": scaling_total_read,
                "memory_write_bytes": scaling_total_write,
                # Compute centered between read and write
                "compute_sections": [
                    {"offset_us": 0, "duration_us": scaling_compute_time},
                ],
                # Memory: read first, then write after compute
                "memory_sections": [
                    {"offset_us": 0.0, "duration_us": scaling_compute_time},  # Read phase
                    {"offset_us": scaling_compute_time, "duration_us": scaling_write_time},  # Write phase
                ],
                # No systolic util (vector ops)
                "systolic_sections": [],
                # Bandwidth: during read and write phases with different utilization
                "bandwidth_sections": [
                    {"offset_us": 0.0, "duration_us": scaling_compute_time, "value": self._compute_bw_util(self.vlen)},
                    {"offset_us": scaling_compute_time, "duration_us": scaling_write_time, "value": self._compute_bw_util(self.vlen)},
                ],
            },
            {
                "name": "softmax",
                "cycles": softmax_total_cycles,
                "memory_read_bytes": softmax_total_read,
                "memory_write_bytes": softmax_total_write,
                # Compute after read
                "compute_sections": [
                    {"offset_us": 0, "duration_us": softmax_compute_time},
                ],
                # Memory: read, then write after compute
                "memory_sections": [
                    {"offset_us": 0.0, "duration_us": softmax_compute_time},
                    {"offset_us": softmax_compute_time, "duration_us": softmax_write_time},
                ],
                # No systolic util (vector ops)
                "systolic_sections": [],
                # Bandwidth during memory phases
                "bandwidth_sections": [
                    {"offset_us": 0.0, "duration_us": softmax_compute_time, "value": self._compute_bw_util(self.vlen)},
                    {"offset_us": softmax_compute_time, "duration_us": softmax_write_time, "value": self._compute_bw_util(self.vlen)},
                ],
            },
            {
                "name": "PV",
                "cycles": pv_total_cycles,
                "memory_read_bytes": pv_total_read,
                "memory_write_bytes": pv_total_write,
                # Multiple compute sections (just one here: main compute)
                "compute_sections": [
                    {"offset_us": 0.0, "duration_us": pv_compute_time},
                ],
                # Multiple memory sections: writeback after compute
                "memory_sections": [
                    {"offset_us": 0, "duration_us": pv_compute_time + pv_write_time},
                ],
                # Systolic util during compute phase
                "systolic_sections": [
                    {"offset_us": 0.0, "duration_us": pv_compute_time, "value": m_util},
                ],
                # Bandwidth util during writeback
                "bandwidth_sections": [
                    {"offset_us": 0, "duration_us": pv_compute_time, "value": self._compute_bw_util(2 * self.mlen)},
                    {"offset_us": pv_compute_time, "duration_us": pv_write_time, "value": self._compute_bw_util(self.vlen)},
                ],
            },
        ]

        total_cycles = sum(seg["cycles"] for seg in segments)

        return {
            "total_cycles": total_cycles,
            "segments": segments,
        }

    # -------------------------------------------------------------------------
    # Self-Attention Layer (No Prefetching)
    # -------------------------------------------------------------------------

    def self_attention_no_prefetch(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        kv_size: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> dict:
        """Self-attention layer analysis WITHOUT prefetching/preloading.

        Memory operations stall compute - data must be loaded before compute can proceed.
        TODO: Implement the no-prefetch version with memory stalls.
        """
        m_util = self._compute_systolic_util(mode, seq_len)
        kv_head_loop = num_kv_heads
        inner_q_head_loop = num_attention_heads // num_kv_heads

        # TODO: Implement no-prefetch logic
        # Key differences from prefetched version:
        # 1. Memory read time adds to total time (not overlapped)
        # 2. Compute sections start after memory read completes
        # 3. Lower effective utilization due to stalls

        if mode == "prefill":
            # ----- QKT -----
            qkt_total_cycles = (
                    (
                        4
                        + self.instr["M_BTMV"] * math.ceil(seq_len / self.mlen) * math.ceil(kv_size / self.mlen)
                        + self.instr["H_PREFETCH_M"]
                    )
                    * kv_head_loop
                    * math.ceil(inner_q_head_loop / math.ceil((self.mlen / self.hlen)))
                ) * batch_size

            s_bytes = self._bits_to_bytes(seq_len * kv_size * num_attention_heads * batch_size, self.activation_bits)
            qkt_total_read = self._bits_to_bytes(kv_size * num_kv_heads * head_dim * batch_size, self.kv_cache_bits)  + s_bytes * 8
            qkt_total_write = s_bytes

            # ----- Scaling -----
            scaling_total_cycles = (
                num_attention_heads * seq_len * math.ceil(kv_size / self.vlen) * self.instr["V_BASIC"]
            ) * batch_size
            scaling_total_read = 2 * s_bytes
            scaling_total_write = 2 * s_bytes

            # ----- Softmax -----
            softmax_total_cycles = (
                num_attention_heads * seq_len * math.ceil(kv_size / self.vlen)
                * (6 * self.instr["V_BASIC"] + self.instr["V_EXP_V"] + self.instr["V_RED_MAX"])
            ) * batch_size
            softmax_total_read = 4 * s_bytes
            softmax_total_write = 4 * s_bytes

            # ----- PV -----
            pv_total_cycles = (
                (4 + self.instr["M_MV"] * math.ceil(kv_size / self.mlen))
                * math.ceil(seq_len / self.blen)
                * math.ceil(head_dim / self.blen)
                * num_attention_heads
            ) * batch_size

            v_bytes = self._bits_to_bytes(kv_size * num_kv_heads * head_dim * batch_size, self.kv_cache_bits)
            out_bytes = self._bits_to_bytes(seq_len * num_attention_heads * head_dim * batch_size, self.activation_bits)
            pv_total_read = v_bytes + s_bytes * 8
            pv_total_write = out_bytes * batch_size

        else:  # decode mode
            # ----- QKT -----
            qkt_total_cycles = (
                (self.instr["M_BTMV"] * math.ceil(kv_size / self.mlen))
                * kv_head_loop
                * math.ceil(inner_q_head_loop / math.ceil((self.mlen / self.hlen)))
            ) * batch_size

            s_bytes = self._bits_to_bytes(kv_size * num_attention_heads * batch_size, self.activation_bits)
            qkt_total_read = self._bits_to_bytes(kv_size * num_kv_heads * head_dim * batch_size, self.kv_cache_bits) * 32
            qkt_total_write = s_bytes

            # ----- Scaling -----
            scaling_total_cycles = (num_attention_heads * math.ceil(kv_size / self.vlen) * (6 * self.instr["V_BASIC"])) * batch_size
            scaling_total_read = 0
            scaling_total_write = s_bytes

            # ----- Softmax -----
            softmax_total_cycles = (
                (math.ceil(kv_size / self.vlen)
                * (6 * self.instr["V_BASIC"] + self.instr["V_EXP_V"] + self.instr["V_RED_MAX"])
                * num_attention_heads) * batch_size
            )
            softmax_total_read = s_bytes * 4
            softmax_total_write = s_bytes * 4

            # ----- PV -----
            pv_total_cycles = (
                (4 + self.instr["M_MV"] + self.instr["H_PREFETCH_M"])
                * math.ceil(head_dim / self.blen)
                * math.ceil(kv_size / self.mlen)
                * num_attention_heads
            ) * batch_size

            pv_total_read = self._bits_to_bytes(kv_size * num_kv_heads * head_dim * batch_size, self.kv_cache_bits) *32
            pv_total_write = s_bytes

        # Compute timing values - NO PREFETCH: memory read time adds to offset
        qkt_compute_time = self._cycles_to_us(qkt_total_cycles)
        qkt_read_time = self._bytes_to_us(qkt_total_read, self.vlen)
        qkt_write_time = self._bytes_to_us(qkt_total_write, self.vlen)

        scaling_compute_time = self._cycles_to_us(scaling_total_cycles)
        scaling_read_time = self._bytes_to_us(scaling_total_read, self.vlen)
        scaling_write_time = self._bytes_to_us(scaling_total_write, self.vlen)

        softmax_compute_time = self._cycles_to_us(softmax_total_cycles)
        softmax_read_time = self._bytes_to_us(softmax_total_read, self.vlen)
        softmax_write_time = self._bytes_to_us(softmax_total_write, self.vlen)

        pv_compute_time = self._cycles_to_us(pv_total_cycles)
        pv_read_time = self._bytes_to_us(pv_total_read, self.vlen)
        pv_write_time = self._bytes_to_us(pv_total_write, self.vlen)

        # TODO: Build segments with stalls - compute starts AFTER memory read
        segments = [
            {
                "name": "QKT",
                "cycles": qkt_total_cycles,
                "memory_read_bytes": qkt_total_read,
                "memory_write_bytes": qkt_total_write,
                "compute_sections": [
                    {"offset_us": qkt_read_time, "duration_us": qkt_compute_time},
                ],
                "memory_sections": [
                    {"offset_us": 0, "duration_us": qkt_read_time},
                    {"offset_us": qkt_read_time + qkt_compute_time, "duration_us": qkt_write_time},
                ],
                "systolic_sections": [
                    {"offset_us": qkt_read_time, "duration_us": qkt_compute_time, "value": m_util},
                ],
                "bandwidth_sections": [
                    {"offset_us": 0, "duration_us": qkt_read_time, "value": self._compute_bw_util(self.hbm_width_bytes)},
                    {"offset_us": qkt_read_time + qkt_compute_time, "duration_us": qkt_write_time, "value": self._compute_bw_util(self.vlen)},
                ],
            },
            {
                "name": "scaling",
                "cycles": scaling_total_cycles,
                "memory_read_bytes": scaling_total_read,
                "memory_write_bytes": scaling_total_write,
                "compute_sections": [
                    {"offset_us": scaling_read_time, "duration_us": scaling_compute_time},
                ],
                "memory_sections": [
                    {"offset_us": 0.0, "duration_us": scaling_read_time},
                    {"offset_us": scaling_read_time + scaling_compute_time, "duration_us": scaling_write_time},
                ],
                "systolic_sections": [],
                "bandwidth_sections": [
                    {"offset_us": 0.0, "duration_us": scaling_read_time, "value": self._compute_bw_util(self.vlen)},
                    {"offset_us": scaling_read_time + scaling_compute_time, "duration_us": scaling_write_time, "value": self._compute_bw_util(self.vlen)},
                ],
            },
            {
                "name": "softmax",
                "cycles": softmax_total_cycles,
                "memory_read_bytes": softmax_total_read,
                "memory_write_bytes": softmax_total_write,
                "compute_sections": [
                    {"offset_us": softmax_read_time, "duration_us": softmax_compute_time},
                ],
                "memory_sections": [
                    {"offset_us": 0.0, "duration_us": softmax_read_time},
                    {"offset_us": softmax_read_time + softmax_compute_time, "duration_us": softmax_write_time},
                ],
                "systolic_sections": [],
                "bandwidth_sections": [
                    {"offset_us": 0.0, "duration_us": softmax_read_time, "value": self._compute_bw_util(self.vlen)},
                    {"offset_us": softmax_read_time + softmax_compute_time, "duration_us": softmax_write_time, "value": self._compute_bw_util(self.vlen)},
                ],
            },
            {
                "name": "PV",
                "cycles": pv_total_cycles,
                "memory_read_bytes": pv_total_read,
                "memory_write_bytes": pv_total_write,
                "compute_sections": [
                    {"offset_us": pv_read_time, "duration_us": pv_compute_time},
                ],
                "memory_sections": [
                    {"offset_us": 0, "duration_us": pv_read_time},
                    {"offset_us": pv_read_time + pv_compute_time, "duration_us": pv_write_time},
                ],
                "systolic_sections": [
                    {"offset_us": pv_read_time, "duration_us": pv_compute_time, "value": m_util},
                ],
                "bandwidth_sections": [
                    {"offset_us": 0, "duration_us": pv_read_time, "value": self._compute_bw_util(self.hbm_width_bytes)},
                    {"offset_us": pv_read_time + pv_compute_time, "duration_us": pv_write_time, "value": self._compute_bw_util(self.vlen)},
                ],
            },
        ]

        total_cycles = sum(seg["cycles"] for seg in segments)

        return {
            "total_cycles": total_cycles,
            "segments": segments,
        }

    # -------------------------------------------------------------------------
    # Flash Attention Layer (placeholder)
    # -------------------------------------------------------------------------

    def flash_attention(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        kv_size: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> dict:
        """Flash attention layer analysis at tile level.

        Flash attention processes tiles iteratively without storing full attention
        matrices to HBM. Each iteration processes QKT -> online softmax -> PV.
        """
        inner_q_head_loop = math.ceil(num_attention_heads / num_kv_heads)
        tr = math.ceil(seq_len / self.mlen)
        tc = math.ceil(kv_size / self.mlen)
        num_iterations = tr * tc * num_kv_heads * batch_size
        

        if mode == "prefill":
            m_util = self._compute_systolic_util(mode, batch_size * seq_len)
            # QKT (M-related: M_BTMM) - per iteration
            qkt_cycles = (
                (4 + self.instr["M_BTMM"] + self.instr["H_PREFETCH_M"])
                * math.ceil(inner_q_head_loop / math.ceil(self.mlen / self.hlen))
            )
            print(f"inner_q_head_loop / math.ceil(self.mlen / self.hlen): {inner_q_head_loop / math.ceil(self.mlen / self.hlen)}")
            # Read Q tile and K tile from SRAM (already prefetched)
            # qkt_total_read = self._bits_to_bytes(self.mlen * head_dim, self.kv_cache_bits)
            qkt_total_read = self._bits_to_bytes(self.mlen * head_dim * inner_q_head_loop, self.activation_bits) * 16
            qkt_total_write = 0

            # Online softmax (not M-related) - per iteration
            softmax_cycles = (
                self.mlen
                * (8 * self.instr["V_BASIC"] + 2 * self.instr["S_BASIC"] + self.instr["S_EXP_FP"])
                * inner_q_head_loop
            ) / math.ceil(self.vlen / self.mlen)
            softmax_total_read = 0
            softmax_total_write = 0

            # PV (M-related: M_MM) - per iteration
            pv_cycles = (
                (4 + math.ceil(head_dim / self.blen) * math.ceil(self.mlen / self.blen) * self.instr["M_MM"])
                * inner_q_head_loop
            )
            pv_total_read = self._bits_to_bytes(self.mlen * head_dim , self.kv_cache_bits) * math.ceil(self.mlen / self.blen)
            pv_total_write = 0

        else:  # decode mode
            m_util = self._compute_systolic_util(mode, 1)
            # QKT (M-related: M_BTMV) - per iteration
            qkt_cycles = (
                4 + self.instr["M_BTMV"] + self.instr["H_PREFETCH_M"]
            ) * math.ceil(inner_q_head_loop / math.ceil(self.mlen / self.hlen)) * math.ceil(self.mlen / self.blen)

            qkt_total_read = self._bits_to_bytes(self.mlen * head_dim, self.kv_cache_bits) * 4
            # qkt_total_read += self._bits_to_bytes(self.mlen * head_dim * inner_q_head_loop, self.activation_bits) * 8
            qkt_total_write = 0

            # Online softmax (not M-related) - per iteration
            softmax_cycles = (
                (6 * self.instr["V_BASIC"] + 2 * self.instr["S_BASIC"] + self.instr["S_EXP_FP"])
                * inner_q_head_loop
            )
            softmax_total_read = 0
            softmax_total_write = 0

            # PV (M-related: M_MV) - per iteration
            pv_cycles = (
                (4 + math.ceil(head_dim / self.blen) * self.instr["M_MV"])
                * inner_q_head_loop
            )
            pv_total_read = self._bits_to_bytes(self.mlen * head_dim, self.kv_cache_bits) * 4
            pv_total_write = 0

        # Per-iteration cycles (QKT -> softmax -> PV)
        iter_cycles = qkt_cycles + softmax_cycles + pv_cycles
        total_cycles = iter_cycles * num_iterations


        # Compute timing values per iteration
        qkt_compute_time = self._cycles_to_us(qkt_cycles)
        qkt_read_time = self._bytes_to_us(qkt_total_read, self.mlen)
        qkt_write_time = self._bytes_to_us(qkt_total_write, self.vlen)
        softmax_compute_time = self._cycles_to_us(softmax_cycles)
        pv_compute_time = self._cycles_to_us(pv_cycles)
        softmax_write_time = self._bytes_to_us(softmax_total_write, self.vlen)
        pv_read_time = self._bytes_to_us(pv_total_read, self.mlen)
        pv_write_time = self._bytes_to_us(pv_total_write, self.vlen)
        iter_time_us = (qkt_compute_time + qkt_write_time +
                        softmax_compute_time + softmax_write_time +
                        pv_compute_time + pv_write_time)

        # Total time for all iterations
        total_time_us = iter_time_us * num_iterations

        # Build separate segments for QKT, Softmax, PV
        segments = [
            {
                "name": "QKT",
                "compute_sections": [
                    {"offset_us": 0.0, "duration_us": qkt_compute_time * num_iterations},
                ],
                "memory_sections": [
                    {"offset_us": 0, "duration_us": qkt_read_time * num_iterations},
                ],
                "systolic_sections": [
                    {"offset_us": 0.0, "duration_us": qkt_compute_time * num_iterations, "value": m_util},
                ],
                "bandwidth_sections": [
                    {"offset_us": 0, "duration_us": qkt_compute_time * num_iterations, "value": self._compute_bw_util(2 * math.sqrt(self.mlen * self.blen))},
                ],
            },
            {
                "name": "Softmax",
                "compute_sections": [
                    {"offset_us": 0.0, "duration_us": softmax_compute_time * num_iterations},
                ],
                "memory_sections": [
                    {"offset_us": 0, "duration_us": 0},
                ],
                "systolic_sections": [],  
                "bandwidth_sections": [],
            },
            {
                "name": "PV",
                "compute_sections": [
                    {"offset_us": 0.0, "duration_us": pv_compute_time * num_iterations},
                ],
                "memory_sections": [
                    {"offset_us": 0, "duration_us": (pv_read_time) * num_iterations},
                    # {"offset_us": (pv_read_time + pv_compute_time) * num_iterations, "duration_us": (pv_write_time) * num_iterations},
                ],
                "systolic_sections": [
                    {"offset_us": 0.0, "duration_us": pv_compute_time * num_iterations, "value": m_util},
                ],
                "bandwidth_sections": [
                    {"offset_us": 0, "duration_us": pv_read_time * num_iterations, "value": self._compute_bw_util(2 * math.sqrt(self.mlen * self.blen))},
                    # {"offset_us": (pv_read_time + pv_compute_time) * num_iterations, "duration_us": (pv_write_time) * num_iterations, "value": self._compute_bw_util(self.vlen)},
                ],
            },
        ]

        return {
            "total_cycles": total_cycles,
            "segments": segments,
            "tiles": {
                "tr": tr,
                "tc": tc,
                "num_iterations": num_iterations,
            },
        }

    # -------------------------------------------------------------------------
    # Flash Attention Layer (No Prefetching)
    # -------------------------------------------------------------------------

    def flash_attention_no_prefetch(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        kv_size: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> dict:
        """Flash attention layer analysis WITHOUT prefetching/preloading.

        Memory operations stall compute - KV tiles must be loaded before compute can proceed.
        TODO: Implement the no-prefetch version with memory stalls.
        """
        inner_q_head_loop = math.ceil(num_attention_heads / num_kv_heads)
        tr = math.ceil(seq_len / self.mlen)
        tc = math.ceil(kv_size / self.mlen)
        num_iterations = tr * tc * num_kv_heads * batch_size

        # KV tile size per iteration
        kv_tile_bytes = self._bits_to_bytes(self.mlen * head_dim * 2, self.kv_cache_bits)  # K and V tiles

        if mode == "prefill":
            m_util = self._compute_systolic_util(mode, batch_size * seq_len)
            # QKT cycles per iteration
            qkt_cycles = (
                (4 + self.instr["M_BTMM"] + self.instr["H_PREFETCH_M"])
                * math.ceil(inner_q_head_loop / math.ceil(self.mlen / self.hlen))
            )

            qkt_total_read = self._bits_to_bytes(self.mlen * head_dim, self.kv_cache_bits) * (self.mlen / self.blen)
            qkt_total_read += self._bits_to_bytes(self.mlen * head_dim * inner_q_head_loop, self.activation_bits) * math.ceil(self.mlen / self.blen)
            qkt_total_write = 0
            # Online softmax cycles per iteration
            softmax_cycles = (
                self.mlen
                * (8 * self.instr["V_BASIC"] + 2 * self.instr["S_BASIC"] + self.instr["S_EXP_FP"])
                * inner_q_head_loop
            ) / math.ceil(self.vlen / self.mlen)

            # PV cycles per iteration
            pv_cycles = (
                (4 + math.ceil(head_dim / self.blen) * math.ceil(self.mlen / self.blen) * self.instr["M_MM"])
                * inner_q_head_loop
            )
            pv_total_read = self._bits_to_bytes(self.mlen * head_dim, self.kv_cache_bits) * 2
            pv_total_write = self._bits_to_bytes(self.mlen * inner_q_head_loop, self.activation_bits)

        else:  # decode mode
            m_util = self._compute_systolic_util(mode, 1)
            # QKT cycles per iteration
            qkt_total_read = self._bits_to_bytes(self.mlen * head_dim, self.kv_cache_bits) * 4
            # qkt_total_read += self._bits_to_bytes(head_dim * inner_q_head_loop, self.activation_bits) 
            qkt_total_write = 0
            qkt_cycles = (
                4 + self.instr["M_BTMV"] + self.instr["H_PREFETCH_M"]
            ) * math.ceil(inner_q_head_loop / math.ceil(self.mlen / self.hlen))

            # Online softmax cycles per iteration
            softmax_cycles = (
                (6 * self.instr["V_BASIC"] + 2 * self.instr["S_BASIC"] + self.instr["S_EXP_FP"])
                * inner_q_head_loop
            )

            # PV cycles per iteration
            pv_cycles = (
                (4 + math.ceil(head_dim / self.blen) * self.instr["M_MV"])
                * inner_q_head_loop
            )
            pv_total_read = self._bits_to_bytes(self.mlen * head_dim, self.kv_cache_bits) * 4
            pv_total_write = self._bits_to_bytes(self.mlen * inner_q_head_loop, self.activation_bits)
        
        # Per-iteration timing
        qkt_compute_time = self._cycles_to_us(qkt_cycles)
        qkt_read_time = self._bytes_to_us(qkt_total_read, self.mlen)
        qkt_write_time = self._bytes_to_us(qkt_total_write, self.mlen)
        softmax_compute_time = self._cycles_to_us(softmax_cycles)
        pv_compute_time = self._cycles_to_us(pv_cycles)
        pv_read_time = self._bytes_to_us(pv_total_read, self.mlen)
        pv_write_time = self._bytes_to_us(pv_total_write, self.mlen)

        # Total cycles (compute only, memory stalls are separate)
        iter_cycles = qkt_cycles + softmax_cycles + pv_cycles
        total_cycles = iter_cycles * num_iterations

        # TODO: Build segments with memory stalls
        # Key: compute sections have offset = kv_read_time (stall for memory)
        total_qkt_compute = qkt_compute_time * num_iterations
        total_qkt_read = qkt_read_time * num_iterations
        total_qkt_write = qkt_write_time * num_iterations
        total_softmax_compute = softmax_compute_time * num_iterations
        total_pv_compute = pv_compute_time * num_iterations
        total_pv_read = pv_read_time * num_iterations
        total_pv_write = pv_write_time * num_iterations
        segments = [
            {
                "name": "QKT",
                "compute_sections": [
                    {"offset_us": total_qkt_read, "duration_us": total_qkt_compute},
                ],
                "memory_sections": [
                    {"offset_us": 0, "duration_us": total_qkt_read},
                    {"offset_us": total_qkt_read + total_qkt_compute, "duration_us": total_qkt_write},
                ],
                "systolic_sections": [
                    {"offset_us": total_qkt_read, "duration_us": total_qkt_compute, "value": m_util},
                ],
                "bandwidth_sections": [
                    {"offset_us": 0, "duration_us": qkt_read_time, "value": self._compute_bw_util(self.vlen)},
                ],
            },
            {
                "name": "Softmax",
                "compute_sections": [
                    {"offset_us": 0.0, "duration_us": total_softmax_compute},
                ],
                "memory_sections": [],
                "systolic_sections": [],
                "bandwidth_sections": [],
            },
            {
                "name": "PV",
                "compute_sections": [
                    {"offset_us": total_pv_read, "duration_us": total_pv_compute},
                ],
                "memory_sections": [
                    {"offset_us": 0, "duration_us": total_pv_read},
                    {"offset_us": total_pv_read + total_pv_compute, "duration_us": total_pv_write},
                ],
                "systolic_sections": [
                    {"offset_us": total_pv_read, "dur   ation_us": total_pv_compute, "value": m_util},
                ],
                "bandwidth_sections": [
                    {"offset_us": 0, "duration_us": total_pv_read, "value": self._compute_bw_util(2 * math.sqrt(self.mlen * self.blen))},
                    {"offset_us": total_pv_read + total_pv_compute, "duration_us": total_pv_write, "value": self._compute_bw_util(self.vlen)},
                ],
            },
        ]

        return {
            "total_cycles": total_cycles,
            "segments": segments,
            "tiles": {
                "tr": tr,
                "tc": tc,
                "num_iterations": num_iterations,
            },
        }

    # -------------------------------------------------------------------------
    # FFN Layer (placeholder)
    # -------------------------------------------------------------------------

    def feed_forward(
        self,
        hidden_size: int,
        intermediate_size: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> dict:
        """Feed-forward layer analysis."""
        m_util = self._compute_systolic_util(mode, batch_size * seq_len)
        if mode == "prefill":
            up_cycles = (
                math.ceil((seq_len * batch_size) / self.blen)
                * math.ceil(hidden_size / self.mlen)
                * math.ceil(intermediate_size / self.blen)
                * self.instr["M_MM"]
            )
            up_total_read = 0
            up_total_write = self._bits_to_bytes(intermediate_size * batch_size * seq_len, self.activation_bits)

            gate_cycles = (
                math.ceil((seq_len * batch_size) / self.blen)
                * math.ceil(hidden_size / self.mlen)
                * math.ceil(intermediate_size / self.blen)
                * self.instr["M_MM"]
            )
            gate_total_read = 0
            gate_total_write = self._bits_to_bytes(intermediate_size * batch_size * seq_len, self.activation_bits)

            silu_cycles = (
                math.ceil(intermediate_size / self.vlen) * 6 * self.instr["V_BASIC"] * seq_len * batch_size
            )

            silu_total_read = 0
            silu_total_write = self._bits_to_bytes(intermediate_size * batch_size * seq_len, self.activation_bits)

            down_cycles = (
                math.ceil((seq_len * batch_size) / self.blen)
                * math.ceil(intermediate_size / self.mlen)
                * math.ceil(hidden_size / self.blen)
                * self.instr["M_MM"]
            )
            down_total_read = 0
            down_total_write = self._bits_to_bytes(hidden_size * intermediate_size, self.activation_bits)

        else:
            up_cycles = (
                math.ceil(intermediate_size / self.blen)
                * math.ceil(hidden_size / self.mlen)
                * math.ceil(batch_size / self.blen)
                * self.instr["M_MM"]
            )
            up_total_read = 0
            up_total_write = 0

            gate_cycles = (
                math.ceil(intermediate_size / self.blen)
                * math.ceil(hidden_size / self.mlen)
                * math.ceil(batch_size / self.blen)
                * self.instr["M_MM"]
            )
            gate_total_read = 0
            gate_total_write = 0

            silu_cycles = (
                math.ceil(intermediate_size / self.vlen) * 6 * self.instr["V_BASIC"] * batch_size
            )
            silu_total_read = 0
            silu_total_write = 0

            down_cycles = (
                math.ceil(batch_size / self.blen)
                * math.ceil(intermediate_size / self.mlen)
                * math.ceil(hidden_size / self.blen)
                * self.instr["M_MM"]
            )
            down_total_read = 0
            down_total_write = 0

        # Total cycles
        total_cycles = up_cycles + gate_cycles + silu_cycles + down_cycles

        # Compute timing values
        up_compute_time = self._cycles_to_us(up_cycles)
        up_write_time = self._bytes_to_us(up_total_write, self.vlen)
        gate_compute_time = self._cycles_to_us(gate_cycles)
        gate_write_time = self._bytes_to_us(gate_total_write, self.vlen)
        silu_compute_time = self._cycles_to_us(silu_cycles)
        silu_write_time = self._bytes_to_us(silu_total_write, self.vlen)
        down_compute_time = self._cycles_to_us(down_cycles)
        down_write_time = self._bytes_to_us(down_total_write, self.vlen)

        # Build segments for Up, Gate, SiLU, Down
        segments = [
            {
                "name": "Up",
                "compute_sections": [
                    {"offset_us": 0.0, "duration_us": up_compute_time},
                ],
                "memory_sections": [
                    {"offset_us": 0, "duration_us": up_compute_time + up_write_time},
                ],
                "systolic_sections": [
                    {"offset_us": 0.0, "duration_us": up_compute_time, "value": m_util},
                ],
                "bandwidth_sections": [
                    {"offset_us": 0, "duration_us": up_compute_time, "value": self._compute_bw_util(2 * self.mlen)},
                    {"offset_us": up_compute_time, "duration_us": up_write_time, "value": self._compute_bw_util(self.vlen)},
                ],
            },
            {
                "name": "Gate",
                "compute_sections": [
                    {"offset_us": 0.0, "duration_us": gate_compute_time},
                ],
                "memory_sections": [
                    {"offset_us": 0, "duration_us": gate_compute_time + gate_write_time},
                ],
                "systolic_sections": [
                    {"offset_us": 0.0, "duration_us": gate_compute_time, "value": m_util},
                ],
                "bandwidth_sections": [
                    {"offset_us": 0, "duration_us": gate_compute_time, "value": self._compute_bw_util(2 * self.mlen)},
                    {"offset_us": gate_compute_time, "duration_us": gate_write_time, "value": self._compute_bw_util(self.vlen)},
                ],
            },
            {
                "name": "SiLU",
                "compute_sections": [
                    {"offset_us": 0.0, "duration_us": silu_compute_time},
                ],
                "memory_sections": [
                    {"offset_us": 0, "duration_us": silu_compute_time + silu_write_time},
                ],
                "systolic_sections": [],  # SiLU is vector op, no systolic
                "bandwidth_sections": [
                    {"offset_us": 0, "duration_us": silu_compute_time + silu_write_time, "value": self._compute_bw_util(self.vlen)},
                ],
            },
            {
                "name": "Down",
                "compute_sections": [
                    {"offset_us": 0.0, "duration_us": down_compute_time},
                ],
                "memory_sections": [
                    {"offset_us": 0, "duration_us": down_compute_time + down_write_time},
                ],
                "systolic_sections": [
                    {"offset_us": 0.0, "duration_us": down_compute_time, "value": m_util},
                ],
                "bandwidth_sections": [
                    {"offset_us": 0, "duration_us": down_compute_time, "value": self._compute_bw_util(2 * self.mlen)},
                    {"offset_us": down_compute_time, "duration_us": down_write_time, "value": self._compute_bw_util(self.vlen)},
                ],
            },
        ]

        return {"total_cycles": total_cycles, "segments": segments}

    # -------------------------------------------------------------------------
    # FFN Layer (No Prefetching)
    # -------------------------------------------------------------------------

    def feed_forward_no_prefetch(
        self,
        hidden_size: int,
        intermediate_size: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> dict:
        """Feed-forward layer analysis WITHOUT prefetching/preloading.

        Memory operations stall compute - weights must be loaded before compute can proceed.
        Similar to self_attention_no_prefetch, memory read happens first, then compute, then write.
        """
        m_util = self._compute_systolic_util(mode, batch_size * seq_len)

        if mode == "prefill":
            up_cycles = (
                math.ceil((seq_len * batch_size) / self.blen)
                * math.ceil(hidden_size / self.mlen)
                * math.ceil(intermediate_size / self.blen)
                * self.instr["M_MM"]
            )
            s_bytes = self._bits_to_bytes(hidden_size * batch_size * seq_len, self.activation_bits)
            up_total_read = self._bits_to_bytes(hidden_size * intermediate_size, self.weight_bits)
            up_total_read += s_bytes * math.ceil(intermediate_size / self.blen)
            up_total_write = self._bits_to_bytes(intermediate_size * batch_size * seq_len, self.activation_bits) * math.ceil(intermediate_size / self.blen)

            gate_cycles = (
                math.ceil((seq_len * batch_size) / self.blen)
                * math.ceil(hidden_size / self.mlen)
                * math.ceil(intermediate_size / self.blen)
                * self.instr["M_MM"]
            )
            gate_total_read = self._bits_to_bytes(hidden_size * intermediate_size, self.weight_bits)
            gate_total_read += s_bytes * math.ceil(intermediate_size / self.blen)
            gate_total_write = self._bits_to_bytes(intermediate_size * batch_size * seq_len, self.activation_bits) * math.ceil(intermediate_size / self.blen)

            silu_cycles = (
                math.ceil(intermediate_size / self.vlen) * 6 * self.instr["V_BASIC"] * seq_len * batch_size
            )
            silu_total_read = 4 * self._bits_to_bytes(intermediate_size * batch_size * seq_len, self.activation_bits)
            silu_total_write = 4 * self._bits_to_bytes(intermediate_size * batch_size * seq_len, self.activation_bits)

            down_cycles = (
                math.ceil((seq_len * batch_size) / self.blen)
                * math.ceil(intermediate_size / self.mlen)
                * math.ceil(hidden_size / self.blen)
                * self.instr["M_MM"]
            )
            down_total_read = self._bits_to_bytes(intermediate_size * hidden_size, self.weight_bits)
            down_total_read += self._bits_to_bytes(intermediate_size * batch_size * seq_len, self.activation_bits) * math.ceil(hidden_size / self.blen)
            down_total_write = self._bits_to_bytes(hidden_size * batch_size * seq_len, self.activation_bits) * math.ceil(hidden_size / self.blen)

        else:  # decode mode
            up_cycles = (
                math.ceil(intermediate_size / self.blen)
                * math.ceil(hidden_size / self.mlen)
                * math.ceil(batch_size / self.blen)
                * self.instr["M_MM"]
            )
            up_total_read = self._bits_to_bytes(hidden_size * intermediate_size, self.weight_bits)
            up_total_read += self._bits_to_bytes(intermediate_size * batch_size, self.activation_bits) * math.ceil(intermediate_size / self.blen)
            up_total_write = self._bits_to_bytes(intermediate_size * batch_size, self.activation_bits) * math.ceil(intermediate_size / self.blen)

            gate_cycles = (
                math.ceil(intermediate_size / self.blen)
                * math.ceil(hidden_size / self.mlen)
                * math.ceil(batch_size / self.blen)
                * self.instr["M_MM"]
            )
            gate_total_read = self._bits_to_bytes(hidden_size * intermediate_size, self.weight_bits)
            gate_total_read += self._bits_to_bytes(intermediate_size * batch_size, self.activation_bits) * math.ceil(intermediate_size / self.blen)
            gate_total_write = self._bits_to_bytes(intermediate_size * batch_size, self.activation_bits) * math.ceil(intermediate_size / self.blen)

            silu_cycles = (
                math.ceil(intermediate_size / self.vlen) * 6 * self.instr["V_BASIC"] * batch_size
            )
            silu_total_read = 4 * self._bits_to_bytes(intermediate_size * batch_size, self.activation_bits)
            silu_total_write = 4 * self._bits_to_bytes(intermediate_size * batch_size, self.activation_bits)

            down_cycles = (
                math.ceil(batch_size / self.blen)
                * math.ceil(intermediate_size / self.mlen)
                * math.ceil(hidden_size / self.blen)
                * self.instr["M_MM"]
            )
            down_total_read = self._bits_to_bytes(intermediate_size * hidden_size, self.weight_bits)
            down_total_read += self._bits_to_bytes(intermediate_size * batch_size, self.activation_bits) * math.ceil(hidden_size / self.blen)
            down_total_write = self._bits_to_bytes(hidden_size * batch_size, self.activation_bits) * math.ceil(hidden_size / self.blen)

        # Total cycles
        total_cycles = up_cycles + gate_cycles + silu_cycles + down_cycles

        # Compute timing values - NO PREFETCH: memory read time adds to offset
        up_compute_time = self._cycles_to_us(up_cycles)
        up_read_time = self._bytes_to_us(up_total_read, 2 * self.mlen)
        up_write_time = self._bytes_to_us(up_total_write, self.vlen)

        gate_compute_time = self._cycles_to_us(gate_cycles)
        gate_read_time = self._bytes_to_us(gate_total_read, 2 * self.mlen)
        gate_write_time = self._bytes_to_us(gate_total_write, self.vlen)

        silu_compute_time = self._cycles_to_us(silu_cycles)
        silu_read_time = self._bytes_to_us(silu_total_read, self.vlen)
        silu_write_time = self._bytes_to_us(silu_total_write, self.vlen)

        down_compute_time = self._cycles_to_us(down_cycles)
        down_read_time = self._bytes_to_us(down_total_read, 2 * self.mlen)
        down_write_time = self._bytes_to_us(down_total_write, self.mlen)

        # Build segments with stalls - compute starts AFTER memory read
        segments = [
            {
                "name": "Up",
                "cycles": up_cycles,
                "memory_read_bytes": up_total_read,
                "memory_write_bytes": up_total_write,
                "compute_sections": [
                    {"offset_us": up_read_time, "duration_us": up_compute_time},
                ],
                "memory_sections": [
                    {"offset_us": 0, "duration_us": up_read_time},
                    {"offset_us": up_read_time + up_compute_time, "duration_us": up_write_time},
                ],
                "systolic_sections": [
                    {"offset_us": up_read_time, "duration_us": up_compute_time, "value": m_util},
                ],
                "bandwidth_sections": [
                    {"offset_us": 0, "duration_us": up_read_time, "value": self._compute_bw_util(2 * self.vlen)},
                    {"offset_us": up_read_time + up_compute_time, "duration_us": up_write_time, "value": self._compute_bw_util(self.vlen)},
                ],
            },
            {
                "name": "Gate",
                "cycles": gate_cycles,
                "memory_read_bytes": gate_total_read,
                "memory_write_bytes": gate_total_write,
                "compute_sections": [
                    {"offset_us": gate_read_time, "duration_us": gate_compute_time},
                ],
                "memory_sections": [
                    {"offset_us": 0, "duration_us": gate_read_time},
                    {"offset_us": gate_read_time + gate_compute_time, "duration_us": gate_write_time},
                ],
                "systolic_sections": [
                    {"offset_us": gate_read_time, "duration_us": gate_compute_time, "value": m_util},
                ],
                "bandwidth_sections": [
                    {"offset_us": 0, "duration_us": gate_read_time, "value": self._compute_bw_util(2 * self.vlen)},
                    {"offset_us": gate_read_time + gate_compute_time, "duration_us": gate_write_time, "value": self._compute_bw_util(self.vlen)},
                ],
            },
            {
                "name": "SiLU",
                "cycles": silu_cycles,
                "memory_read_bytes": silu_total_read,
                "memory_write_bytes": silu_total_write,
                "compute_sections": [
                    {"offset_us": silu_read_time, "duration_us": silu_compute_time},
                ],
                "memory_sections": [
                    {"offset_us": 0, "duration_us": silu_read_time},
                    {"offset_us": silu_read_time + silu_compute_time, "duration_us": silu_write_time},
                ],
                "systolic_sections": [],  # SiLU is vector op, no systolic
                "bandwidth_sections": [
                    {"offset_us": 0, "duration_us": silu_read_time, "value": self._compute_bw_util(self.vlen)},
                    {"offset_us": silu_read_time + silu_compute_time, "duration_us": silu_write_time, "value": self._compute_bw_util(self.vlen)},
                ],
            },
            {
                "name": "Down",
                "cycles": down_cycles,
                "memory_read_bytes": down_total_read,
                "memory_write_bytes": down_total_write,
                "compute_sections": [
                    {"offset_us": down_read_time, "duration_us": down_compute_time},
                ],
                "memory_sections": [
                    {"offset_us": 0, "duration_us": down_read_time},
                    {"offset_us": down_read_time + down_compute_time, "duration_us": down_write_time},
                ],
                "systolic_sections": [
                    {"offset_us": down_read_time, "duration_us": down_compute_time, "value": m_util},
                ],
                "bandwidth_sections": [
                    {"offset_us": 0, "duration_us": down_read_time, "value": self._compute_bw_util(2 * self.vlen)},
                    {"offset_us": down_read_time + down_compute_time, "duration_us": down_write_time, "value": self._compute_bw_util(self.vlen)},
                ],
            },
        ]

        return {"total_cycles": total_cycles, "segments": segments}

    # -------------------------------------------------------------------------
    # Output Projection Layer (placeholder)
    # -------------------------------------------------------------------------

    def output_projection(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> dict:
        """Output projection layer analysis. TODO: Implement."""
        return {"total_cycles": 0, "segments": [], "tiles": {}}

    # -------------------------------------------------------------------------
    # Residual Layer (placeholder)
    # -------------------------------------------------------------------------

    def residual(
        self,
        hidden_size: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> dict:
        """Residual connection analysis. TODO: Implement."""
        return {"total_cycles": 0, "segments": [], "tiles": {}}
