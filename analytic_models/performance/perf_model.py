"""
PLENA Hardware Performance Model.

Provides per-layer hardware latency modeling using instruction latencies.
This module is used by llama_model.py for LLM-level performance estimation.
"""

import json
import math
from typing import TypedDict

import toml
from pydantic import BaseModel, Field, model_validator


# =============================================================================
# Layer Performance Result with Utilization Segments
# =============================================================================


class UtilizationSegment(TypedDict):
    """A segment of execution: cycles and systolic array utilization percentage."""

    cycles: int
    systolic_utilization: float  # 0.0 to 100.0 percentage


# LayerPerfResult is a dict: {"total_cycles": int, "segments": list[UtilizationSegment]}
# segments is ordered list representing execution timeline
# Example for prefill: {"total_cycles": 2300, "segments": [
#     {"cycles": 1000, "systolic_utilization": 100.0},  # QKT - full utilization
#     {"cycles": 500, "systolic_utilization": 0.0},     # softmax - no systolic
#     {"cycles": 800, "systolic_utilization": 100.0},   # PV - full utilization
# ]}
# Example for decode (batch=4, blen=16): {"total_cycles": 1500, "segments": [
#     {"cycles": 600, "systolic_utilization": 25.0},    # QKT - batch/blen = 4/16 = 25%
#     {"cycles": 300, "systolic_utilization": 0.0},     # softmax - no systolic
#     {"cycles": 600, "systolic_utilization": 25.0},    # PV - batch/blen = 25%
# ]}

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
                inner_q_head_loop / (self.mlen // self.hlen)
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
            # S = Q (seq_len, num_attention_heads, head_dim) @ K^T (seq_len, num_kv_heads, head_dim) = (num_attention_heads, seq_len, seq_len)
            if multi_core_mode:
                single_batch_compute_cycles += (
                    (
                        4
                        + self.instr["M_BTMM"] * math.ceil(seq_len / self.mlen) * math.ceil(kv_size / self.mlen)
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
            # QKT / const (num_attention_heads, seq_len, seq_len)
            single_batch_compute_cycles += num_attention_heads * (seq_len * math.ceil(seq_len / self.vlen))
            # P= Softmax (num_attention_heads, seq_len, seq_len)
            single_batch_compute_cycles += (
                seq_len
                * math.ceil(seq_len / self.vlen)
                * (self.instr["V_EXP_V"] + self.instr["V_RED_MAX"] + self.instr["V_BASIC"])
            )
            # PV = P (seq_len, seq_len, num_attention_heads) @ V (seq_len, num_kv_heads, head_dim) = (seq_len, num_attention_heads, head_dim)
            single_batch_compute_cycles += (
                (4 + self.instr["M_MM"] * math.ceil(seq_len / self.mlen) + self.instr["H_PREFETCH_M"])
                * math.ceil(seq_len / self.blen)
                * math.ceil(head_dim / self.blen)
                * num_attention_heads
            )
        else:  # decode
            # S = Q (1, num_attention_heads, head_dim) @ K^T (kv_size, num_kv_heads, head_dim) = (num_attention_heads, kv_size)
            if multi_core_mode:
                single_batch_compute_cycles += (
                    (4 + self.instr["M_BTMV"] * math.ceil(kv_size / self.mlen) + self.instr["H_PREFETCH_M"])
                    * kv_head_loop
                    * (math.ceil((self.mlen // self.hlen) // inner_q_head_loop))
                )
            else:
                single_batch_compute_cycles += (
                    4 + self.instr["M_MV"] * math.ceil(kv_size / self.blen) + self.instr["H_PREFETCH_M"]
                ) * num_attention_heads

            # QKT / const (num_attention_heads, kv_size)
            single_batch_compute_cycles += num_attention_heads * (math.ceil(kv_size / self.vlen))

            # P= Softmax (num_attention_heads, kv_size)
            single_batch_compute_cycles += math.ceil(kv_size / self.vlen) * (
                self.instr["V_EXP_V"] + self.instr["V_RED_MAX"] + self.instr["V_BASIC"]
            )

            # PV = P (kv_size, num_attention_heads, head_dim) @ V (kv_size, num_kv_heads, head_dim) = (1, num_attention_heads, head_dim)
            single_batch_compute_cycles += (
                (4 + self.instr["M_MV"] * math.ceil(kv_size / self.mlen) + self.instr["H_PREFETCH_M"])
                * math.ceil(head_dim / self.blen)
                * num_attention_heads
            )
        overall_cycles = single_batch_compute_cycles * batch_size
        return overall_cycles

    def mlp_moe(
        self,
        hidden_size: int,
        seq_len: int,
        batch_size: int,
        num_experts: int,
        expert_per_token: int,
        intermediate_size: int,
        mode: str = "prefill",
    ) -> int:
        """
        MoE cycle count.

        In MoE, tokens are routed to experts and batched per expert.
        Each expert processes its batch of tokens using M_MM (not per-token M_MV).
        Average tokens per expert = (total_tokens * expert_per_token) / num_experts
        """
        overall_cycles = 0

        if mode == "prefill":
            # Total tokens being processed
            total_tokens = batch_size * seq_len

            # Average tokens routed to each expert (for batched processing)
            # Each token selects expert_per_token experts, distributed across num_experts
            tokens_per_expert = math.ceil((total_tokens * expert_per_token) / num_experts)

            # Normalize (b, s, h) -> (b, s, h)
            overall_cycles += (math.ceil(hidden_size / self.vlen) * self.instr["V_BASIC"] * 4) * total_tokens

            # Router / Gate: (b*s, h) @ (h, num_experts) -> (b*s, num_experts)
            # Using M_MM for batch matrix multiply
            overall_cycles += (
                (4 + math.ceil(hidden_size / self.mlen) * self.instr["M_MM"] + self.instr["H_PREFETCH_M"])
                * math.ceil(total_tokens / self.blen)
                * math.ceil(num_experts / self.blen)
            )

            # TOP K: (b*s, num_experts) -> (b*s, expert_per_token)
            overall_cycles += (4 + math.ceil(num_experts / self.vlen) * self.instr["V_TOPK"]) * total_tokens

            # Softmax over selected experts: (b*s, expert_per_token) -> (b*s, expert_per_token)
            overall_cycles += (
                total_tokens
                * math.ceil(expert_per_token / self.vlen)
                * (self.instr["V_EXP_V"] + self.instr["V_RED_MAX"] + self.instr["V_BASIC"])
            )

            # Expert FFN Computation - MLP1 (Gate + Up projection)
            # Tokens are grouped by expert and processed in batches using M_MM
            # Each expert: (tokens_per_expert, hidden) @ (hidden, 2*intermediate) -> (tokens_per_expert, 2*intermediate)
            # Run for all num_experts experts
            overall_cycles += (
                num_experts
                * (4 + math.ceil(hidden_size / self.mlen) * self.instr["M_MM"] + self.instr["H_PREFETCH_M"])
                * math.ceil(tokens_per_expert / self.blen)
                * math.ceil(2 * intermediate_size / self.blen)
            )

            # SiLU activation + element-wise multiply (gate * up)
            # Total activations = total_tokens * expert_per_token (each token activates expert_per_token experts)
            overall_cycles += (
                total_tokens * expert_per_token * math.ceil(intermediate_size / self.vlen) * 6 * self.instr["V_BASIC"]
            )

            # Expert FFN Computation - MLP2 (Down projection)
            # Each expert: (tokens_per_expert, intermediate) @ (intermediate, hidden) -> (tokens_per_expert, hidden)
            overall_cycles += (
                num_experts
                * (4 + math.ceil(intermediate_size / self.mlen) * self.instr["M_MM"] + self.instr["H_PREFETCH_M"])
                * math.ceil(tokens_per_expert / self.blen)
                * math.ceil(hidden_size / self.blen)
            )

            # Weighted sum of experts
            # Per token: sum over expert_per_token weighted vectors of size hidden_size
            overall_cycles += (
                total_tokens
                * expert_per_token
                * math.ceil(hidden_size / self.vlen)
                * (self.instr["V_MUL_VV"] + self.instr["V_ADD_VV"])
            )

        else:  # decode mode: seq_len = 1, few tokens - use M_MV per token
            total_tokens = batch_size

            # Normalize (b, h) -> (b, h)
            overall_cycles += (math.ceil(hidden_size / self.vlen) * self.instr["V_BASIC"] * 4) * total_tokens

            # Router / Gate: (b, h) @ (h, num_experts) -> (b, num_experts)
            # For small batch, use M_MV per token
            overall_cycles += (
                total_tokens
                * (4 + math.ceil(hidden_size / self.mlen) * self.instr["M_MV"] + self.instr["H_PREFETCH_M"])
                * math.ceil(num_experts / self.blen)
            )

            # TOP K: (b, num_experts) -> (b, expert_per_token)
            overall_cycles += (4 + math.ceil(num_experts / self.vlen) * self.instr["V_TOPK"]) * total_tokens

            # Softmax over selected experts: (b, expert_per_token) -> (b, expert_per_token)
            overall_cycles += (
                total_tokens
                * math.ceil(expert_per_token / self.vlen)
                * (self.instr["V_EXP_V"] + self.instr["V_RED_MAX"] + self.instr["V_BASIC"])
            )

            # Expert FFN Computation - MLP1 (Gate + Up projection)
            # In decode, few tokens so use M_MV per (token, expert) pair
            overall_cycles += (
                total_tokens
                * expert_per_token
                * (4 + math.ceil(hidden_size / self.mlen) * self.instr["M_MV"] + self.instr["H_PREFETCH_M"])
                * math.ceil(2 * intermediate_size / self.blen)
            )

            # SiLU activation + element-wise multiply
            overall_cycles += (
                total_tokens * expert_per_token * math.ceil(intermediate_size / self.vlen) * 6 * self.instr["V_BASIC"]
            )

            # Expert FFN Computation - MLP2 (Down projection)
            overall_cycles += (
                total_tokens
                * expert_per_token
                * (4 + math.ceil(intermediate_size / self.mlen) * self.instr["M_MV"] + self.instr["H_PREFETCH_M"])
                * math.ceil(hidden_size / self.blen)
            )

            # Weighted sum of experts
            overall_cycles += (
                total_tokens
                * expert_per_token
                * math.ceil(hidden_size / self.vlen)
                * (self.instr["V_MUL_VV"] + self.instr["V_ADD_VV"])
            )

        return overall_cycles

    def sliding_window_attention(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        kv_size: int,
        batch_size: int,
        sliding_window_size: int,
        num_sink_tokens: int = 1,
        mode: str = "prefill",
        multi_core_mode: bool = False,
    ) -> int:
        """
        Sliding window attention cycle count.

        Based on Aria sliding attention pattern:
        - Q: (seq_len, num_kv_heads, q_mult, head_dim)
        - K: (seq_len, num_kv_heads, head_dim)
        - V: (seq_len, num_kv_heads, head_dim)
        - S (sinks): (num_attention_heads,) - sink token scores

        Each query attends to at most sliding_window_size keys plus sink tokens.
        """
        overall_cycles = 0
        single_batch_compute_cycles = 0
        kv_head_loop = num_kv_heads
        inner_q_head_loop = num_attention_heads // num_kv_heads

        if mode == "prefill":
            # Effective attention window per query position
            # Each query at position i attends to keys in range [max(0, i - sliding_window_size + 1), i]
            # Average effective KV length = (1 + sliding_window_size) / 2 for early tokens, sliding_window_size for later tokens
            # Simplified: use min(seq_len, sliding_window_size) as effective KV dimension
            effective_kv_len = min(seq_len, sliding_window_size)

            # QK^T: Q (seq_len, num_kv_heads, q_mult, head_dim) @ K^T (seq_len, num_kv_heads, head_dim)
            # Output shape: (num_kv_heads, q_mult, seq_len, effective_kv_len) = (num_attention_heads, seq_len, effective_kv_len)
            if multi_core_mode:
                single_batch_compute_cycles += (
                    (
                        4
                        + self.instr["M_BTMM"]
                        * math.ceil(seq_len / self.mlen)
                        * math.ceil(effective_kv_len / self.mlen)
                        + self.instr["H_PREFETCH_M"]
                    )
                    * kv_head_loop
                    * (math.ceil((self.mlen // self.hlen) // inner_q_head_loop))
                )
            else:
                single_batch_compute_cycles += (
                    4
                    + self.instr["M_MM"] * math.ceil(seq_len / self.blen) * math.ceil(effective_kv_len / self.blen)
                    + self.instr["H_PREFETCH_M"]
                ) * num_attention_heads

            # QKT scaling: / sqrt(head_dim) - (num_attention_heads, seq_len, effective_kv_len)
            single_batch_compute_cycles += num_attention_heads * (seq_len * math.ceil(effective_kv_len / self.vlen))

            # ==================== UNSUPPORTED OPERATIONS ====================
            # TODO: Sliding window mask application
            # Apply causal mask + sliding window mask (set positions outside window to -inf)
            # mask = triu(-inf, diagonal=1) + tril(-inf, diagonal=-sliding_window_size)
            # Cycles: ___FILL_MASK_CYCLES___

            # TODO: Sink token concatenation
            # Concatenate sink scores S to attention: QK = cat([QK, S], dim=-1)
            # Shape: (num_attention_heads, seq_len, effective_kv_len) -> (num_attention_heads, seq_len, effective_kv_len + num_sink_tokens)
            # Cycles: ___FILL_CONCAT_CYCLES___
            # ================================================================

            # Softmax: (num_attention_heads, seq_len, effective_kv_len + num_sink_tokens)
            single_batch_compute_cycles += (
                seq_len
                * math.ceil((effective_kv_len + num_sink_tokens) / self.vlen)
                * (self.instr["V_EXP_V"] + self.instr["V_RED_MAX"] + self.instr["V_BASIC"])
            )

            # ==================== UNSUPPORTED OPERATIONS ====================
            # TODO: Remove sink dimension from attention weights
            # W = W[..., :-num_sink_tokens]
            # Shape: (num_attention_heads, seq_len, effective_kv_len + num_sink_tokens) -> (num_attention_heads, seq_len, effective_kv_len)
            # Cycles: ___FILL_SLICE_CYCLES___
            # ================================================================

            # P @ V: P (seq_len, effective_kv_len, num_attention_heads) @ V (effective_kv_len, num_kv_heads, head_dim)
            # Output: (seq_len, num_attention_heads, head_dim)
            single_batch_compute_cycles += (
                (4 + self.instr["M_MM"] * math.ceil(effective_kv_len / self.mlen) + self.instr["H_PREFETCH_M"])
                * math.ceil(seq_len / self.blen)
                * math.ceil(head_dim / self.blen)
                * num_attention_heads
            )

        else:  # decode
            # In decode mode, query has seq_len=1
            # Attend to min(kv_size, sliding_window_size) keys
            effective_kv_len = min(kv_size, sliding_window_size)

            # QK^T: Q (1, num_attention_heads, head_dim) @ K^T (effective_kv_len, num_kv_heads, head_dim)
            # Output: (num_attention_heads, effective_kv_len)
            if multi_core_mode:
                single_batch_compute_cycles += (
                    (4 + self.instr["M_BTMV"] * math.ceil(effective_kv_len / self.mlen) + self.instr["H_PREFETCH_M"])
                    * kv_head_loop
                    * (math.ceil((self.mlen // self.hlen) // inner_q_head_loop))
                )
            else:
                single_batch_compute_cycles += (
                    4 + self.instr["M_MV"] * math.ceil(effective_kv_len / self.blen) + self.instr["H_PREFETCH_M"]
                ) * num_attention_heads

            # QKT scaling: / sqrt(head_dim) - (num_attention_heads, effective_kv_len)
            single_batch_compute_cycles += num_attention_heads * (math.ceil(effective_kv_len / self.vlen))

            # ==================== UNSUPPORTED OPERATIONS ====================
            # TODO: Sink token concatenation for decode
            # Concatenate sink scores: (num_attention_heads, effective_kv_len) -> (num_attention_heads, effective_kv_len + num_sink_tokens)
            # Cycles: ___FILL_CONCAT_CYCLES___
            # ================================================================

            # Softmax: (num_attention_heads, effective_kv_len + num_sink_tokens)
            single_batch_compute_cycles += math.ceil((effective_kv_len + num_sink_tokens) / self.vlen) * (
                self.instr["V_EXP_V"] + self.instr["V_RED_MAX"] + self.instr["V_BASIC"]
            )

            # ==================== UNSUPPORTED OPERATIONS ====================
            # TODO: Remove sink dimension from attention weights
            # Cycles: ___FILL_SLICE_CYCLES___
            # ================================================================

            # P @ V: P (effective_kv_len, num_attention_heads) @ V (effective_kv_len, num_kv_heads, head_dim)
            # Output: (1, num_attention_heads, head_dim)
            single_batch_compute_cycles += (
                (4 + self.instr["M_MV"] * math.ceil(effective_kv_len / self.mlen) + self.instr["H_PREFETCH_M"])
                * math.ceil(head_dim / self.blen)
                * num_attention_heads
            )

        overall_cycles = single_batch_compute_cycles * batch_size
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

    # -------------------------------------------------------------------------
    # Layer-level methods with utilization segments
    # Returns dict: {"total_cycles": int, "segments": list[UtilizationSegment]}
    # -------------------------------------------------------------------------

    def rms_layer_with_util(
        self, hidden_size: int, seq_len: int, batch_size: int, mode: str = "prefill"
    ) -> dict:
        """RMSNorm layer with utilization segments. No M-related ops."""
        total_cycles = self.rms_layer(hidden_size, seq_len, batch_size, mode)
        # RMSNorm uses only V_BASIC and H_PREFETCH_V - no systolic array
        return {
            "total_cycles": total_cycles,
            "segments": [{"cycles": total_cycles, "systolic_utilization": 0.0}],
        }

    def projection_with_util(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> dict:
        """Q, K, V projection + RoPE with utilization segments."""
        segments: list[UtilizationSegment] = []
        bs_dim = seq_len * batch_size

        # Systolic utilization: 100% for prefill, batch/blen ratio for decode
        if mode == "prefill":
            m_util = 100.0
        else:
            m_util = min(batch_size / self.blen, 1.0) * 100.0

        if mode == "prefill":
            # Projection of Q (M-related)
            q_proj_cycles = (
                math.ceil(bs_dim / self.blen)
                * (math.ceil(hidden_size / self.mlen) * (math.ceil(hidden_size / self.blen)))
                * self.instr["M_MM"]
            )
            segments.append({"cycles": q_proj_cycles, "systolic_utilization": m_util})

            # RoPE of Q (not M-related)
            q_rope_cycles = num_attention_heads * math.ceil(bs_dim / self.vlen) * self.instr["V_BASIC"]
            segments.append({"cycles": q_rope_cycles, "systolic_utilization": 0.0})

            # Projection of K (M-related)
            k_proj_cycles = (
                math.ceil(bs_dim / self.blen)
                * (
                    math.ceil((num_kv_heads * head_dim) / self.mlen)
                    * (math.ceil((num_kv_heads * head_dim) / self.blen))
                )
                * self.instr["M_MM"]
            )
            segments.append({"cycles": k_proj_cycles, "systolic_utilization": m_util})

            # RoPE of K (not M-related)
            k_rope_cycles = num_kv_heads * math.ceil(bs_dim / self.vlen) * self.instr["V_BASIC"]
            segments.append({"cycles": k_rope_cycles, "systolic_utilization": 0.0})

            # Projection of V (M-related)
            v_proj_cycles = (
                math.ceil(bs_dim / self.blen)
                * (
                    math.ceil((num_kv_heads * head_dim) / self.mlen)
                    * (math.ceil((num_kv_heads * head_dim) / self.blen))
                )
                * self.instr["M_MM"]
            )
            segments.append({"cycles": v_proj_cycles, "systolic_utilization": m_util})

            # Memory operations (not M-related)
            compute_cycle = q_proj_cycles + q_rope_cycles + k_proj_cycles + k_rope_cycles + v_proj_cycles
            mem_cycles = 0
            if hidden_size * seq_len * batch_size > self.vector_sram_size:
                mem_cycles = (
                    (hidden_size * seq_len * batch_size - self.vector_sram_size)
                    // (self.vlen * self.prefetch_v_amount)
                ) * self.instr["H_PREFETCH_V"] * 2
            # Store K,V Cache
            store_cycles = (
                2
                * ((batch_size * seq_len * num_kv_heads * head_dim) // (self.vlen * self.prefetch_v_amount))
                * self.instr["H_STORE_V"]
            )
            if mem_cycles + store_cycles > 0:
                segments.append({"cycles": mem_cycles + store_cycles, "systolic_utilization": 0.0})

        else:  # decode
            # Projection of Q (M-related)
            q_proj_cycles = (
                math.ceil(batch_size / self.blen)
                * math.ceil(hidden_size / self.mlen)
                * math.ceil(hidden_size / self.blen)
                * self.instr["M_MM"]
            )
            segments.append({"cycles": q_proj_cycles, "systolic_utilization": m_util})

            # RoPE of Q (not M-related)
            q_rope_cycles = num_attention_heads * math.ceil(bs_dim / self.vlen) * self.instr["V_BASIC"]
            segments.append({"cycles": q_rope_cycles, "systolic_utilization": 0.0})

            # Projection of K (M-related)
            k_proj_cycles = (
                math.ceil(batch_size / self.blen)
                * (
                    math.ceil((num_kv_heads * head_dim) / self.mlen)
                    * (math.ceil((num_kv_heads * head_dim) / self.blen))
                )
                * self.instr["M_MM"]
            )
            segments.append({"cycles": k_proj_cycles, "systolic_utilization": m_util})

            # RoPE of K (not M-related)
            k_rope_cycles = num_kv_heads * math.ceil(bs_dim / self.vlen) * self.instr["V_BASIC"]
            segments.append({"cycles": k_rope_cycles, "systolic_utilization": 0.0})

            # Projection of V (M-related)
            v_proj_cycles = (
                math.ceil(batch_size / self.blen)
                * (
                    math.ceil((num_kv_heads * head_dim) / self.mlen)
                    * (math.ceil((num_kv_heads * head_dim) / self.blen))
                )
                * self.instr["M_MM"]
            )
            segments.append({"cycles": v_proj_cycles, "systolic_utilization": m_util})

            # Store K,V Cache (not M-related)
            store_cycles = (
                2
                * ((batch_size * num_kv_heads * head_dim) // (self.vlen * self.prefetch_v_amount))
                * self.instr["H_STORE_V"]
            )
            if store_cycles > 0:
                segments.append({"cycles": store_cycles, "systolic_utilization": 0.0})

        total_cycles = sum(seg["cycles"] for seg in segments)
        return {"total_cycles": total_cycles, "segments": segments}

    def flash_attention_with_util(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        kv_size: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> dict:
        """Flash attention with utilization segments."""
        segments: list[UtilizationSegment] = []

        kv_head_loop = num_kv_heads
        inner_q_head_loop = num_attention_heads // num_kv_heads
        tr = math.ceil(seq_len / self.mlen)
        tc = math.ceil(kv_size / self.mlen)
        outer_loops = tr * tc * kv_head_loop * batch_size

        # Systolic utilization: 100% for prefill, batch/blen ratio for decode
        if mode == "prefill":
            m_util = 100.0
        else:
            m_util = min(batch_size / self.blen, 1.0) * 100.0

        if mode == "prefill":
            # QKT (M-related: M_BTMM)
            qkt_cycles = (
                (4 + self.instr["M_BTMM"] + self.instr["H_PREFETCH_M"])
                * math.ceil(inner_q_head_loop / (self.mlen // self.hlen))
                * outer_loops
            )
            segments.append({"cycles": qkt_cycles, "systolic_utilization": m_util})

            # Online softmax (not M-related)
            softmax_cycles = (
                self.mlen
                * (8 * self.instr["V_BASIC"] + 2 * self.instr["S_BASIC"] + self.instr["S_EXP_FP"])
                * inner_q_head_loop
                * outer_loops
            )
            segments.append({"cycles": softmax_cycles, "systolic_utilization": 0.0})

            # Compute PV (M-related: M_MM)
            pv_cycles = (
                (4 + math.ceil(head_dim / self.blen) * math.ceil(self.mlen / self.blen) * self.instr["M_MM"])
                * inner_q_head_loop
                * outer_loops
            )
            segments.append({"cycles": pv_cycles, "systolic_utilization": m_util})

            # Compute O + Scaling (not M-related)
            o_scaling_cycles = (
                (self.mlen * (2 * self.instr["V_BASIC"] + 4) * inner_q_head_loop
                 + self.mlen * (1 * self.instr["V_BASIC"] + 4 + self.instr["S_RECI_FP"]) * inner_q_head_loop)
                * outer_loops
            )
            segments.append({"cycles": o_scaling_cycles, "systolic_utilization": 0.0})

        else:  # decode
            # QKT (M-related: M_BTMV)
            qkt_cycles = (4 + self.instr["M_BTMV"] + self.instr["H_PREFETCH_M"]) * outer_loops
            segments.append({"cycles": qkt_cycles, "systolic_utilization": m_util})

            # Online softmax (not M-related)
            softmax_cycles = (
                (8 * self.instr["V_BASIC"] + 2 * self.instr["S_BASIC"] + self.instr["S_EXP_FP"])
                * inner_q_head_loop
                * outer_loops
            )
            segments.append({"cycles": softmax_cycles, "systolic_utilization": 0.0})

            # Compute PV (M-related: M_MV)
            pv_cycles = (
                (4 + math.ceil(head_dim / self.blen) * (self.instr["M_MV"] + 2 * self.instr["S_ADDI_INT"]))
                * inner_q_head_loop
                * outer_loops
            )
            segments.append({"cycles": pv_cycles, "systolic_utilization": m_util})

            # Compute O + Scaling (not M-related)
            o_scaling_cycles = (
                ((2 * self.instr["V_BASIC"] + 1) * inner_q_head_loop
                 + (1 * self.instr["V_BASIC"] + self.instr["S_RECI_FP"]) * inner_q_head_loop)
                * outer_loops
            )
            segments.append({"cycles": o_scaling_cycles, "systolic_utilization": 0.0})

        total_cycles = sum(seg["cycles"] for seg in segments)
        return {"total_cycles": total_cycles, "segments": segments}

    def residual_with_util(
        self, hidden_size: int, seq_len: int, batch_size: int, mode: str = "prefill"
    ) -> dict:
        """Residual connection with utilization segments. No M-related ops."""
        total_cycles = self.residual(hidden_size, seq_len, batch_size, mode)
        return {
            "total_cycles": total_cycles,
            "segments": [{"cycles": total_cycles, "systolic_utilization": 0.0}],
        }

    def feed_forward_with_util(
        self, hidden_size: int, intermediate_size: int, seq_len: int, batch_size: int, mode: str = "prefill"
    ) -> dict:
        """Feed-forward (MLP) layer with utilization segments."""
        segments: list[UtilizationSegment] = []

        # Systolic utilization: 100% for prefill, batch/blen ratio for decode
        if mode == "prefill":
            m_util = 100.0
        else:
            m_util = min(batch_size / self.blen, 1.0) * 100.0

        if mode == "prefill":
            # Upsize Linear and Gate (M-related)
            upsize_cycles = (
                2
                * math.ceil((seq_len * batch_size) / self.blen)
                * math.ceil(hidden_size / self.mlen)
                * math.ceil(intermediate_size / self.blen)
                * self.instr["M_MM"]
            )
            segments.append({"cycles": upsize_cycles, "systolic_utilization": m_util})

            # SiLU (not M-related)
            silu_cycles = (
                math.ceil(intermediate_size / self.vlen) * 6 * self.instr["V_BASIC"] * seq_len * batch_size
            )
            segments.append({"cycles": silu_cycles, "systolic_utilization": 0.0})

            # Downsize Linear (M-related)
            downsize_cycles = (
                math.ceil((seq_len * batch_size) / self.blen)
                * math.ceil(intermediate_size / self.mlen)
                * math.ceil(hidden_size / self.blen)
                * self.instr["M_MM"]
            )
            segments.append({"cycles": downsize_cycles, "systolic_utilization": m_util})

        else:  # decode
            # Upsize Linear and Gate (M-related)
            upsize_cycles = (
                2
                * math.ceil(intermediate_size / self.blen)
                * math.ceil(hidden_size / self.mlen)
                * math.ceil(batch_size / self.blen)
                * self.instr["M_MM"]
            )
            segments.append({"cycles": upsize_cycles, "systolic_utilization": m_util})

            # SiLU (not M-related)
            silu_cycles = math.ceil(intermediate_size / self.vlen) * 6 * self.instr["V_BASIC"] * batch_size
            segments.append({"cycles": silu_cycles, "systolic_utilization": 0.0})

            # Downsize Linear (M-related)
            downsize_cycles = (
                math.ceil(batch_size / self.blen)
                * math.ceil(intermediate_size / self.mlen)
                * math.ceil(hidden_size / self.blen)
                * self.instr["M_MM"]
            )
            segments.append({"cycles": downsize_cycles, "systolic_utilization": m_util})

        total_cycles = sum(seg["cycles"] for seg in segments)
        return {"total_cycles": total_cycles, "segments": segments}
