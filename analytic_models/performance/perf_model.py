"""
PLENA Hardware Performance Model.

Provides per-layer hardware latency modeling using instruction latencies.
This module is used by llama_model.py for LLM-level performance estimation.
"""

import json
import math

import toml
from pydantic import BaseModel, Field, model_validator

# =============================================================================
# Hardware Configuration Schema
# =============================================================================


class HardwareConfig(BaseModel):
    """Validated hardware configuration for PLENA accelerator."""

    # Core hardware dimensions
    MLEN: int = Field(gt=0, description="Matrix unit length")
    BLEN: int = Field(gt=0, description="Block length")
    VLEN: int = Field(gt=0, description="Vector length")
    HLEN: int = Field(gt=0, description="Head dimension length")

    # Memory configuration
    VECTOR_SRAM_SIZE: int = Field(gt=0, description="Vector SRAM size in elements")
    HBM_V_Prefetch_Amount: int = Field(gt=0, description="HBM vector prefetch amount")

    # Allow extra fields for latency parameters (dynamically loaded)
    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def validate_dimensions(self) -> "HardwareConfig":
        """Validate hardware dimension relationships."""
        if self.MLEN % self.BLEN != 0:
            raise ValueError(f"MLEN ({self.MLEN}) must be divisible by BLEN ({self.BLEN})")
        if self.VLEN < self.BLEN:
            raise ValueError(f"VLEN ({self.VLEN}) must be >= BLEN ({self.BLEN})")
        return self


class InstructionLatency(BaseModel):
    """Validated instruction latency map (instruction name -> pipelined cycles)."""

    latencies: dict[str, int]

    @model_validator(mode="after")
    def validate_latencies(self) -> "InstructionLatency":
        """Validate all latencies are positive."""
        for name, cycles in self.latencies.items():
            if cycles <= 0:
                raise ValueError(f"Instruction '{name}' has invalid latency: {cycles}")
        return self

    def __getitem__(self, key: str) -> int:
        """Allow dict-like access: instr['M_MM']."""
        if key not in self.latencies:
            raise KeyError(f"Unknown instruction: {key}. Available: {list(self.latencies.keys())}")
        return self.latencies[key]

    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator."""
        return key in self.latencies

    def items(self):
        """Allow iteration over items."""
        return self.latencies.items()


# =============================================================================
# Hardware Configuration Loading
# =============================================================================


def load_hardware_config_from_toml(toml_path: str) -> HardwareConfig:
    """
    Load hardware configuration from plena_settings.toml.
    Always reads from the ANALYTIC section for the latency model.

    Args:
        toml_path: Path to the TOML configuration file

    Returns:
        HardwareConfig: Validated hardware configuration
    """
    with open(toml_path) as f:
        data = toml.load(f)

    config_dict = {}

    # Always read from ANALYTIC section for the latency model
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

    return HardwareConfig(**config_dict)


# =============================================================================
# Instruction Latency Model (Pipelined)
# =============================================================================


def build_pipelined_latency(hardware_config: HardwareConfig, custom_isa_path: str) -> InstructionLatency:
    """
    Build pipelined instruction latency from customISA_lib.json.
    Evaluates expressions using hardware config values.

    Args:
        hardware_config: Validated hardware configuration
        custom_isa_path: Path to customISA_lib.json

    Returns:
        InstructionLatency: Validated instruction latencies
    """
    with open(custom_isa_path) as f:
        custom_isa_lib = json.load(f)

    # Build config dict for eval (convert pydantic model to dict)
    configs = hardware_config.model_dump()
    configs["SA_ACC_CYCLES"] = int(math.log2(hardware_config.MLEN / hardware_config.BLEN) + 1)

    latencies = {}
    for instr_name, instr_data in custom_isa_lib.items():
        if "pipelined" in instr_data:
            latencies[instr_name] = eval(instr_data["pipelined"], {}, configs)
        else:
            raise ValueError(f"Instruction '{instr_name}' missing 'pipelined' field.")

    return InstructionLatency(latencies=latencies)


# =============================================================================
# PerfModel: Per-Layer Hardware Latency Model
# =============================================================================


class PerfModel:
    """
    Per-layer hardware latency model for PLENA accelerator.

    On init, builds pipelined instruction latency from customISA_lib.json
    using hardware config values (MLEN, BLEN, VLEN, HLEN, latency params).

    Attributes:
        config: Validated HardwareConfig
        instr: Validated InstructionLatency (access via instr["M_MM"])
        mlen, blen, vlen, hlen: Hardware dimensions
    """

    config: HardwareConfig
    instr: InstructionLatency

    def __init__(self, hardware_config: HardwareConfig, custom_isa_path: str):
        """
        Initialize PerfModel.

        Args:
            hardware_config: Validated hardware configuration
            custom_isa_path: Path to customISA_lib.json
        """
        self.config = hardware_config
        self.mlen = hardware_config.MLEN
        self.blen = hardware_config.BLEN
        self.vlen = hardware_config.VLEN
        self.hlen = hardware_config.HLEN
        self.vector_sram_size = hardware_config.VECTOR_SRAM_SIZE * self.vlen
        self.prefetch_v_amount = hardware_config.HBM_V_Prefetch_Amount

        # Build validated instruction latencies
        self.instr = build_pipelined_latency(hardware_config, custom_isa_path)

    # -------------------------------------------------------------------------
    # Layer-level latency computation methods
    # -------------------------------------------------------------------------

    def rms_layer(self, hidden_size: int, seq_len: int, batch_size: int, mode: str = "prefill") -> int:
        """RMSNorm layer cycle count."""
        setting_inst_num = 10
        loop_inst_num = 8
        loop_num = hidden_size // self.vlen
        overall_cycles = 0

        if mode == "prefill":
            compute_cycle = (
                setting_inst_num * self.instr["S_BASIC"]
                + loop_num * loop_inst_num * seq_len * self.instr["V_BASIC"] * batch_size
            )
            if hidden_size * seq_len * batch_size > self.vector_sram_size:
                overall_cycles += (
                    compute_cycle
                    + (
                        (hidden_size * seq_len * batch_size - self.vector_sram_size)
                        // (self.vlen * self.prefetch_v_amount)
                    )
                    * self.instr["H_PREFETCH_V"]
                    * 2
                )
            else:
                overall_cycles += compute_cycle
        else:  # decode
            overall_cycles = (
                setting_inst_num * self.instr["S_BASIC"] + loop_num * loop_inst_num * self.instr["V_BASIC"] * batch_size
            )

        return overall_cycles

    def projection(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> int:
        """Q, K, V projection + RoPE cycle count."""
        compute_cycle = 0
        overall_cycles = 0
        bs_dim = seq_len * batch_size

        if mode == "prefill":
            # Projection of Q
            compute_cycle += (
                math.ceil(bs_dim / self.blen)
                * (math.ceil(hidden_size / self.mlen) * (math.ceil(hidden_size / self.blen)))
                * self.instr["M_MM"]
            )
            # RoPE of Q
            compute_cycle += num_attention_heads * math.ceil(bs_dim / self.vlen) * self.instr["V_BASIC"]
            # Projection of K
            compute_cycle += (
                math.ceil(bs_dim / self.blen)
                * (
                    math.ceil((num_kv_heads * head_dim) / self.mlen)
                    * (math.ceil((num_kv_heads * head_dim) / self.blen))
                )
                * self.instr["M_MM"]
            )
            # RoPE of K
            compute_cycle += num_kv_heads * math.ceil(bs_dim / self.vlen) * self.instr["V_BASIC"]
            # Projection of V
            compute_cycle += (
                math.ceil(bs_dim / self.blen)
                * (
                    math.ceil((num_kv_heads * head_dim) / self.mlen)
                    * (math.ceil((num_kv_heads * head_dim) / self.blen))
                )
                * self.instr["M_MM"]
            )
            # Activation (Q) Manipulation
            if hidden_size * seq_len * batch_size > self.vector_sram_size:
                overall_cycles += (
                    compute_cycle
                    + (
                        (hidden_size * seq_len * batch_size - self.vector_sram_size)
                        // (self.vlen * self.prefetch_v_amount)
                    )
                    * self.instr["H_PREFETCH_V"]
                    * 2
                )
            else:
                overall_cycles += compute_cycle
            # Store K,V Cache
            overall_cycles += (
                2
                * ((batch_size * seq_len * num_kv_heads * head_dim) // (self.vlen * self.prefetch_v_amount))
                * self.instr["H_STORE_V"]
            )
        else:  # decode
            # Projection of Q
            compute_cycle += (
                math.ceil(batch_size / self.blen)
                * math.ceil(hidden_size / self.mlen)
                * math.ceil(hidden_size / self.blen)
                * self.instr["M_MM"]
            )
            # RoPE of Q
            compute_cycle += num_attention_heads * math.ceil(bs_dim / self.vlen) * self.instr["V_BASIC"]
            # Projection of K
            compute_cycle += (
                math.ceil(batch_size / self.blen)
                * (
                    math.ceil((num_kv_heads * head_dim) / self.mlen)
                    * (math.ceil((num_kv_heads * head_dim) / self.blen))
                )
                * self.instr["M_MM"]
            )
            # RoPE of K
            compute_cycle += num_kv_heads * math.ceil(bs_dim / self.vlen) * self.instr["V_BASIC"]
            # Projection of V
            compute_cycle += (
                math.ceil(batch_size / self.blen)
                * (
                    math.ceil((num_kv_heads * head_dim) / self.mlen)
                    * (math.ceil((num_kv_heads * head_dim) / self.blen))
                )
                * self.instr["M_MM"]
            )
            overall_cycles += compute_cycle
            # Store K,V Cache
            overall_cycles += (
                2
                * ((batch_size * num_kv_heads * head_dim) // (self.vlen * self.prefetch_v_amount))
                * self.instr["H_STORE_V"]
            )
        return overall_cycles

    def flash_attention(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        kv_size: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> int:
        """Flash attention cycle count (assumes GQA mode)."""
        inner_compute_cycles = 0
        overall_cycles = 0

        kv_head_loop = num_kv_heads
        inner_q_head_loop = num_attention_heads // num_kv_heads
        tr = math.ceil(seq_len / self.mlen)
        tc = math.ceil(kv_size / self.mlen)

        if mode == "prefill":
            # QKT (per KV head and Grouped Q heads)
            inner_compute_cycles += (4 + self.instr["M_BTMM"] + self.instr["H_PREFETCH_M"]) * math.ceil(
                (self.mlen // self.hlen) // inner_q_head_loop
            )
            # online softmax
            inner_compute_cycles += (
                self.mlen
                * (8 * self.instr["V_BASIC"] + 2 * self.instr["S_BASIC"] + self.instr["S_EXP_FP"])
                * inner_q_head_loop
            )
            # Compute PV
            inner_compute_cycles += (
                4
                + math.ceil(head_dim / self.blen)
                * (math.ceil(self.mlen / self.blen))
                * self.instr["M_MM"]
                * inner_q_head_loop
            )
            # Compute O
            inner_compute_cycles += self.mlen * (2 * self.instr["V_BASIC"] + 4) * inner_q_head_loop
            # Compute Scaling
            inner_compute_cycles += (
                self.mlen * (1 * self.instr["V_BASIC"] + 4 + self.instr["S_RECI_FP"]) * inner_q_head_loop
            )
            overall_cycles = inner_compute_cycles * tr * tc * kv_head_loop * batch_size
        else:  # decode
            # QKT (per KV head and Grouped Q heads)
            inner_compute_cycles += 4 + self.instr["M_BTMV"] + self.instr["H_PREFETCH_M"]
            # online softmax
            inner_compute_cycles += (
                8 * self.instr["V_BASIC"] + 2 * self.instr["S_BASIC"] + self.instr["S_EXP_FP"]
            ) * inner_q_head_loop
            # Compute PV
            inner_compute_cycles += (
                4
                + math.ceil(head_dim / self.blen)
                * (self.instr["M_MV"] + 2 * self.instr["S_ADDI_INT"])
                * inner_q_head_loop
            )
            # Compute O
            inner_compute_cycles += (2 * self.instr["V_BASIC"] + 1) * inner_q_head_loop
            # Compute Scaling
            inner_compute_cycles += (1 * self.instr["V_BASIC"] + self.instr["S_RECI_FP"]) * inner_q_head_loop
            overall_cycles = inner_compute_cycles * tr * tc * kv_head_loop * batch_size

        return overall_cycles

    def self_attention(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        kv_size: int,
        batch_size: int,
        mode: str = "prefill",
        multi_core_mode: bool = False,
    ) -> int:
        """Self-attention cycle count."""
        overall_cycles = 0
        single_batch_compute_cycles = 0
        kv_head_loop = num_kv_heads
        inner_q_head_loop = num_attention_heads // num_kv_heads

        if mode == "prefill":
            # S = Q (seq_len, num_attention_heads, head_dim) @ K^T (seq_len, num_kv_heads, head_dim) = (seq_len, seq_len, num_attention_heads)
            if multi_core_mode:
                single_batch_compute_cycles += (
                    (
                        4
                        + self.instr["M_BTMM"] * math.ceil(seq_len / self.mlen) * math.ceil(seq_len / self.mlen)
                        + self.instr["H_PREFETCH_M"]
                    )
                    * kv_head_loop
                    * (math.ceil((self.mlen // self.hlen) // inner_q_head_loop))
                )
            else:
                single_batch_compute_cycles += (
                    4
                    + self.instr["M_MM"] * math.ceil(seq_len / self.blen) * math.ceil(seq_len / self.blen)
                    + self.instr["H_PREFETCH_M"]
                ) * num_attention_heads
            # QKT / const
            single_batch_compute_cycles += num_attention_heads * (seq_len * math.ceil(seq_len / self.vlen))
            # P= Softmax
            single_batch_compute_cycles += (
                seq_len
                * math.ceil(seq_len / self.vlen)
                * (self.instr["V_EXP_FP"] + self.instr["V_RED_MAX"] + self.instr["V_BASIC"])
            )
            # PV = P (seq_len, seq_len, num_attention_heads) @ V (seq_len, num_kv_heads, head_dim) = (seq_len, num_attention_heads, head_dim)
            single_batch_compute_cycles += (
                (4 + self.instr["M_MM"] * math.ceil(seq_len / self.mlen) + self.instr["H_PREFETCH_M"])
                * math.ceil(head_dim / self.blen)
                * math.ceil(seq_len / self.blen)
                * num_attention_heads
            )
        overall_cycles = single_batch_compute_cycles * batch_size
        return overall_cycles

    def moe(self, hidden_size: int, seq_len: int, batch_size: int, mode: str = "prefill") -> int:
        """MoE cycle count."""
        overall_cycles = 0
        return overall_cycles

    def residual(self, hidden_size: int, seq_len: int, batch_size: int, mode: str = "prefill") -> int:
        """Residual connection cycle count."""
        iteration = hidden_size // self.vlen
        overall_cycles = 0

        if mode == "prefill":
            compute_cycle = (self.instr["V_ADD_VV"] + 3) * seq_len * iteration * batch_size
            if hidden_size * seq_len * batch_size > self.vector_sram_size:
                overall_cycles += (
                    compute_cycle
                    + (
                        (hidden_size * seq_len * batch_size - self.vector_sram_size)
                        // (self.vlen * self.prefetch_v_amount)
                    )
                    * self.instr["H_PREFETCH_V"]
                    * 2
                )
            else:
                overall_cycles += compute_cycle
        else:
            compute_cycle = (self.instr["V_ADD_VV"] + 3) * iteration * batch_size
            overall_cycles += compute_cycle
        return overall_cycles

    def feed_forward(
        self, hidden_size: int, intermediate_size: int, seq_len: int, batch_size: int, mode: str = "prefill"
    ) -> int:
        """Feed-forward (MLP) layer cycle count."""
        overall_cycles = 0

        if mode == "prefill":
            # Upsize Linear and Gate
            overall_cycles += (
                2
                * math.ceil((seq_len * batch_size) / self.blen)
                * math.ceil(hidden_size / self.mlen)
                * math.ceil(intermediate_size / self.blen)
                * self.instr["M_MM"]
            )
            # SiLU
            overall_cycles += (
                math.ceil(intermediate_size / self.vlen) * 6 * self.instr["V_BASIC"] * seq_len * batch_size
            )
            # Downsize Linear
            overall_cycles += (
                math.ceil((seq_len * batch_size) / self.blen)
                * math.ceil(intermediate_size / self.mlen)
                * math.ceil(hidden_size / self.blen)
                * self.instr["M_MM"]
            )
        else:
            # Upsize Linear and Gate
            overall_cycles += (
                2
                * math.ceil(intermediate_size / self.blen)
                * math.ceil(hidden_size / self.mlen)
                * math.ceil(batch_size / self.blen)
                * self.instr["M_MM"]
            )
            # SiLU
            overall_cycles += math.ceil(intermediate_size / self.vlen) * 6 * self.instr["V_BASIC"] * batch_size
            # Downsize Linear
            overall_cycles += (
                math.ceil(batch_size / self.blen)
                * math.ceil(intermediate_size / self.mlen)
                * math.ceil(hidden_size / self.blen)
                * self.instr["M_MM"]
            )

        return overall_cycles

    def embeddings(self, hidden_size: int, seq_len: int, batch_size: int, mode: str = "prefill") -> int:
        """Embedding layer cycle count."""
        setting_inst_num = 3
        overall_cycles = setting_inst_num * self.instr["S_BASIC"]

        if mode == "prefill":
            overall_cycles += seq_len * batch_size * math.ceil(hidden_size / self.vlen) * self.instr["H_PREFETCH_V"]
        else:  # decode
            overall_cycles += batch_size * math.ceil(hidden_size / self.vlen) * self.instr["H_PREFETCH_V"]

        return overall_cycles

    def lm_head(self, hidden_size: int, vocab_size: int, batch_size: int) -> int:
        """LM head cycle count (linear projection to vocab)."""
        setting_inst_num = 3
        overall_cycles = setting_inst_num * self.instr["S_BASIC"]

        # Matrix multiply: [batch_size, hidden_size] x [hidden_size, vocab_size]
        overall_cycles += (
            math.ceil(batch_size / self.blen)
            * math.ceil(hidden_size / self.mlen)
            * math.ceil(vocab_size / self.blen)
            * self.instr["M_MM"]
        )

        return overall_cycles
