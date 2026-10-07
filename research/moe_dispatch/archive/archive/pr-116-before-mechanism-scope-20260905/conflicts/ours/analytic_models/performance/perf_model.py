"""
PLENA Hardware Performance Model.

Provides per-layer hardware latency modeling using instruction latencies.
This module is used by llama_model.py / gpt_oss_model.py / mamba2_model.py for
LLM-level performance estimation.

Two cost axes are modelled per layer stage:

1. *Compute / instruction-issue cycles* — tile counts multiplied by the
   ``pipelined`` cycle cost of each opcode in ``customISA_lib.json``.
2. *Memory cycles* — the bytes each stage has to move across the HBM interface
   divided by ``HBM_BANDWIDTH_BYTES_PER_CYCLE`` (see ``PerfModel`` docstring).

A stage costs ``max(compute_cycles, mem_cycles)``, not the sum. Rationale: the
PLENA datapath double-buffers HBM prefetch against systolic/vector execution
(``H_PREFETCH_*`` issues asynchronously and the consumer stalls only if the data
has not landed), so a perfectly pipelined stage runs at whichever of the two
limits is worse. ``max`` is the standard roofline bound; summing would model a
machine that never overlaps DMA with compute and would roughly double every
memory-bound stage. The cost of ``max`` is that it ignores prologue/epilogue
ramp (the first tile's fetch is genuinely exposed), so the model is optimistic
by roughly one tile-fetch per stage. Set ``enable_bandwidth=False`` to recover
the pre-bandwidth (compute-only) behaviour exactly.
"""

import json
import math

import toml
from pydantic import BaseModel, Field, model_validator

# =============================================================================
# Numeric format parsing ([ANALYTIC.PRECISION.*])
# =============================================================================


def _scalar_type_bits(type_spec: dict) -> int:
    """Bit width of a single scalar element described by a PRECISION type table.

    Handles the two shapes used in plena_settings.toml:
      {type="Fp", sign=<bool>, exponent=<int>, mantissa=<int>}
      {type="Int", width=<int>}
    """
    kind = str(type_spec.get("type", "Fp")).lower()
    if kind == "int":
        return int(type_spec["width"])
    if kind == "fp":
        return int(bool(type_spec.get("sign", False))) + int(type_spec["exponent"]) + int(type_spec["mantissa"])
    raise ValueError(f"Unknown PRECISION element type: {type_spec.get('type')!r}")


def _entry_bytes_per_elem(entry: dict) -> float:
    """Bytes per element for one [ANALYTIC.PRECISION.<NAME>] entry.

    ``format = "Plain"``  -> DATA_TYPE bits / 8.
    ``format = "Mx"``     -> a block of ``block`` elements shares ONE scale, so the
                             amortised width is (elem_bits*block + scale_bits) / block.
                             MXFP8 at block=8 with an 8-bit scale is therefore
                             (8*8 + 8) / (8*8) = 1.125 bytes/elem, not 1.0.
    bare type table       -> (e.g. SCALAR_FP) treated as Plain.
    """
    fmt = entry.get("format")
    if fmt == "Mx":
        block = int(entry.get("block", 1))
        elem_bits = _scalar_type_bits(entry["ELEM"])
        scale_bits = _scalar_type_bits(entry["SCALE"])
        return (elem_bits * block + scale_bits) / (8.0 * block)
    if fmt == "Plain":
        return _scalar_type_bits(entry["DATA_TYPE"]) / 8.0
    if "type" in entry:  # bare type table, e.g. [ANALYTIC.PRECISION.SCALAR_FP]
        return _scalar_type_bits(entry) / 8.0
    raise ValueError(f"Unrecognised PRECISION entry: {entry!r}")


def parse_precision_bytes(precision_section: dict) -> dict[str, float]:
    """Parse a whole [<MODE>.PRECISION] section into {name: bytes_per_element}."""
    out: dict[str, float] = {}
    for name, entry in precision_section.items():
        if not isinstance(entry, dict):
            continue
        out[name] = _entry_bytes_per_elem(entry)
    return out


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
    HBM_V_Writeback_Amount: int = Field(default=16, gt=0, description="HBM vector writeback amount")

    # HBM interface. HBM_WIDTH is a row/burst width in BYTES — this matches every
    # other consumer in the tree (address_alloc.py's `hbm_row_width: int = 512
    # # bytes`, PLENA_Tools/memory_mapping/rand_gen.py's `HBM_WIDTH // 4` floats
    # per line). One row per cycle is the assumed peak.
    HBM_WIDTH: int = Field(default=512, gt=0, description="HBM row width in bytes")
    HBM_BANDWIDTH_BYTES_PER_CYCLE: float = Field(
        default=0.0,
        ge=0.0,
        description="Explicit HBM bandwidth override in bytes/cycle. 0 = derive from HBM_WIDTH.",
    )

    # bytes-per-element per memory class, parsed from [<MODE>.PRECISION.*]
    PRECISION_BYTES: dict[str, float] = Field(default_factory=dict, description="Bytes per element by memory class")

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

    # Extract PRECISION section -> bytes per element per memory class.
    # Previously ignored entirely, which left the model with no notion of how
    # many bytes a tensor actually occupies in HBM.
    config_dict["PRECISION_BYTES"] = parse_precision_bytes(analytic_data.get("PRECISION", {}))

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
            latencies[instr_name] = eval(instr_data["pipelined"], {"__builtins__": {}}, configs)
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

    Memory-bandwidth model
    ----------------------
    Peak HBM bandwidth is ``HBM_BANDWIDTH_BYTES_PER_CYCLE`` if set in the config,
    otherwise ``HBM_WIDTH`` bytes/cycle (one HBM row per cycle). With the shipped
    analytic config that is 512 B/cycle, i.e. 512 GB/s at the 1 GHz clock
    ``llama_model.py`` assumes.

    Bytes per element come from ``[ANALYTIC.PRECISION.*]``; MX formats amortise
    their shared block scale, so MXFP8 at block=8 is 1.125 B/elem.

    Each layer stage returns ``max(compute_cycles, mem_cycles)`` — see the module
    docstring for why ``max`` and not a sum, and what that approximation costs.

    Assumptions worth knowing before trusting a number:
      * Weights are streamed from HBM on *every* invocation of a stage. At batch 1
        decode this is the dominant term and is realistic; at large batch, a real
        scheduler would amortise one weight fetch over the whole batch, and this
        model does too (weight bytes do not scale with batch).
      * No cache/SRAM residency model for weights across layers: a model small
        enough to stay resident in on-chip SRAM would be over-charged here.
      * ``max`` per stage, summed across stages: it cannot express overlap
        *between* stages (prefetching layer N+1's weights during layer N).

    Attributes:
        config: Validated HardwareConfig
        instr: Validated InstructionLatency (access via instr["M_MM"])
        mlen, blen, vlen, hlen: Hardware dimensions
        hbm_bytes_per_cycle: Peak HBM bandwidth used by the roofline
        weight_bytes / kv_bytes / act_bytes / state_bytes: bytes per element
    """

    config: HardwareConfig
    instr: InstructionLatency

    def __init__(self, hardware_config: HardwareConfig, custom_isa_path: str, enable_bandwidth: bool = True):
        """
        Initialize PerfModel.

        Args:
            hardware_config: Validated hardware configuration
            custom_isa_path: Path to customISA_lib.json
            enable_bandwidth: If False, every stage returns its compute-only cycle
                count (the pre-bandwidth behaviour). Used by the regression test
                that pins backward compatibility.
        """
        self.config = hardware_config
        self.mlen = hardware_config.MLEN
        self.blen = hardware_config.BLEN
        self.vlen = hardware_config.VLEN
        self.hlen = hardware_config.HLEN
        self.vector_sram_size = hardware_config.VECTOR_SRAM_SIZE * self.vlen
        self.prefetch_v_amount = hardware_config.HBM_V_Prefetch_Amount
        # BUGFIX: stores are writebacks and must use the writeback amount. The
        # previous code used HBM_V_Prefetch_Amount for H_STORE_V while
        # HBM_V_Writeback_Amount sat unused in plena_settings.toml.
        self.writeback_v_amount = hardware_config.HBM_V_Writeback_Amount

        # ---- memory bandwidth / precision -----------------------------------
        self.enable_bandwidth = enable_bandwidth
        self.precision_bytes: dict[str, float] = dict(hardware_config.PRECISION_BYTES)
        # bf16 (2 B) fallbacks if a config predates the PRECISION section
        self.weight_bytes = self.precision_bytes.get("HBM_M_WEIGHT_TYPE", 2.0)
        self.kv_bytes = self.precision_bytes.get("HBM_M_KV_TYPE", 2.0)
        self.act_bytes = self.precision_bytes.get("HBM_V_ACT_TYPE", 2.0)
        # Recurrent state is not an attention KV cache.  Keep its precision
        # independent from KV: the evaluated PLENA design point selects BF16,
        # while the official GPU implementations' FP32 state remains external
        # baseline/accuracy evidence.  Older configs still fall back to KV.
        self.state_bytes = self.precision_bytes.get(
            "HBM_STATE_TYPE",
            self.precision_bytes.get("HBM_V_KV_TYPE", self.kv_bytes),
        )

        bw_override = float(getattr(hardware_config, "HBM_BANDWIDTH_BYTES_PER_CYCLE", 0.0) or 0.0)
        self.hbm_bytes_per_cycle = bw_override if bw_override > 0 else float(hardware_config.HBM_WIDTH)

        self.reset_traffic()

        # Build validated instruction latencies
        self.instr = build_pipelined_latency(hardware_config, custom_isa_path)

    # -------------------------------------------------------------------------
    # Bandwidth / roofline helpers
    # -------------------------------------------------------------------------

    def reset_traffic(self) -> None:
        """Zero the traffic/roofline accounting counters."""
        self.traffic_bytes = 0.0
        self.compute_only_cycles = 0
        self.memory_only_cycles = 0
        self.roofline_cycles = 0
        self.memory_bound_stages = 0
        self.total_stages = 0

    def mem_cycles(self, num_bytes: float) -> int:
        """Cycles to move ``num_bytes`` across the HBM interface at peak bandwidth."""
        if num_bytes <= 0:
            return 0
        return math.ceil(num_bytes / self.hbm_bytes_per_cycle)

    def _roofline(self, compute_cycles: int, mem_bytes: float = 0.0) -> int:
        """Combine a stage's compute cost and its HBM traffic into one cycle count.

        Returns ``max(compute, memory)`` (see module docstring). Also accumulates
        traffic statistics so callers can report how memory-bound a phase is.
        """
        compute_cycles = int(compute_cycles)
        mc = self.mem_cycles(mem_bytes)

        self.traffic_bytes += mem_bytes
        self.compute_only_cycles += compute_cycles
        self.memory_only_cycles += mc
        self.total_stages += 1
        if mc > compute_cycles:
            self.memory_bound_stages += 1

        result = max(compute_cycles, mc) if self.enable_bandwidth else compute_cycles
        self.roofline_cycles += result
        return result

    def traffic_summary(self) -> dict:
        """Snapshot of the traffic counters accumulated since ``reset_traffic()``."""
        return {
            "hbm_bytes_per_cycle": self.hbm_bytes_per_cycle,
            "traffic_bytes": self.traffic_bytes,
            "compute_only_cycles": self.compute_only_cycles,
            "memory_only_cycles": self.memory_only_cycles,
            "roofline_cycles": self.roofline_cycles,
            "memory_bound_stages": self.memory_bound_stages,
            "total_stages": self.total_stages,
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _spill_elems(self, hidden_size: int, seq_len: int, batch_size: int) -> int:
        """Activation elements that do NOT fit in vector SRAM and spill to HBM."""
        total = hidden_size * seq_len * batch_size
        return max(0, total - self.vector_sram_size)

    def _effective_btmm(self, eff_rows: int, eff_cols: int) -> int:
        """Effective BTMM cost for a tile of actual size eff_rows × eff_cols.

        The full M_BTMM = (MLEN//BLEN)² × BLEN assumes the systolic array
        processes a complete MLEN×MLEN tile.  When the actual data occupies
        fewer rows/cols (e.g. seq_len < MLEN), only ceil(dim/BLEN) blocks
        per axis are active and the array completes proportionally faster.
        """
        return math.ceil(eff_rows / self.blen) * math.ceil(eff_cols / self.blen) * self.blen

    # -------------------------------------------------------------------------
    # Layer-level latency computation methods
    # -------------------------------------------------------------------------

    def rms_layer(self, hidden_size: int, seq_len: int, batch_size: int, mode: str = "prefill") -> int:
        """RMSNorm layer cycle count."""
        setting_inst_num = 10
        loop_inst_num = 8
        loop_num = hidden_size // self.vlen
        overall_cycles = 0
        mem_bytes = 0.0

        if mode == "prefill":
            compute_cycle = (
                setting_inst_num * self.instr["S_BASIC"]
                + loop_num * loop_inst_num * seq_len * self.instr["V_BASIC"] * batch_size
            )
            spill = self._spill_elems(hidden_size, seq_len, batch_size)
            if spill:
                overall_cycles += (
                    compute_cycle + (spill // (self.vlen * self.prefetch_v_amount)) * self.instr["H_PREFETCH_V"] * 2
                )
                # spilled activations are read back and written out again
                mem_bytes = 2 * spill * self.act_bytes
            else:
                overall_cycles += compute_cycle
        else:  # decode
            # A single token's activations always fit in vector SRAM -> no HBM traffic.
            overall_cycles = (
                setting_inst_num * self.instr["S_BASIC"] + loop_num * loop_inst_num * self.instr["V_BASIC"] * batch_size
            )

        return self._roofline(overall_cycles, mem_bytes)

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

        # HBM traffic shared by both modes: the Q/K/V weight matrices are streamed
        # once per invocation, and the newly produced K,V are appended to the cache.
        # (o_proj is not modelled here — the callers never charge for it.)
        weight_elems = hidden_size * hidden_size + 2 * hidden_size * (num_kv_heads * head_dim)
        kv_written_elems = 2 * batch_size * (seq_len if mode == "prefill" else 1) * num_kv_heads * head_dim
        mem_bytes = weight_elems * self.weight_bytes + kv_written_elems * self.kv_bytes

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
            spill = self._spill_elems(hidden_size, seq_len, batch_size)
            if spill:
                overall_cycles += (
                    compute_cycle + (spill // (self.vlen * self.prefetch_v_amount)) * self.instr["H_PREFETCH_V"] * 2
                )
                mem_bytes += 2 * spill * self.act_bytes
            else:
                overall_cycles += compute_cycle
            # Store K,V Cache
            # BUGFIX (a): math.ceil, not floor. Floor division silently charged ZERO
            # cycles whenever the cache slice was smaller than one writeback burst.
            # BUGFIX (b): a store is metered by HBM_V_Writeback_Amount, not the
            # prefetch amount.
            overall_cycles += (
                2
                * math.ceil((batch_size * seq_len * num_kv_heads * head_dim) / (self.vlen * self.writeback_v_amount))
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
            # Store K,V Cache — same two bugfixes as the prefill path. At the
            # default config this term used to evaluate to
            #   (4 * 8 * 128) // (2048 * 16) == 4096 // 32768 == 0
            # i.e. the decode cache write was free.
            overall_cycles += (
                2
                * math.ceil((batch_size * num_kv_heads * head_dim) / (self.vlen * self.writeback_v_amount))
                * self.instr["H_STORE_V"]
            )
        return self._roofline(overall_cycles, mem_bytes)

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

        # HBM traffic: the KV cache is streamed once per Q block (FlashAttention's
        # outer loop is over Q, inner over KV), so tr passes over the whole cache.
        # In decode tr == 1, giving exactly one pass — the term that makes decode
        # attention cost grow linearly with context.
        # Approximation: causal masking would remove ~half the tiles in prefill;
        # neither the compute nor the memory term models that (both charge tr*tc).
        kv_elems = 2 * batch_size * kv_size * num_kv_heads * head_dim
        mem_bytes = tr * kv_elems * self.kv_bytes

        if mode == "prefill":
            # Effective BTMM cost — scales with actual tile fill, not full MLEN×MLEN
            eff_btmm = self._effective_btmm(min(seq_len, self.mlen), min(kv_size, self.mlen))
            # QKT (per KV head and Grouped Q heads)
            inner_compute_cycles += (4 + eff_btmm + self.instr["H_PREFETCH_M"]) * math.ceil(
                inner_q_head_loop / (self.mlen // self.hlen)
            )
            # online softmax
            inner_compute_cycles += (
                min(self.mlen, kv_size)
                * (4 * self.instr["V_BASIC"] + 2 * self.instr["S_BASIC"] + self.instr["S_EXP_FP"])
                * inner_q_head_loop
            )
            # Compute PV
            inner_compute_cycles += (
                4
                + math.ceil(head_dim / self.blen)
                * (math.ceil(min(self.mlen, kv_size) / self.blen))
                * self.instr["M_MM"]
                * inner_q_head_loop
            )
            # Compute O and scaling
            inner_compute_cycles += (
                min(self.mlen, kv_size) * (1 * self.instr["V_BASIC"] + 4 + self.instr["S_RECI_FP"]) * inner_q_head_loop
            )
            overall_cycles = inner_compute_cycles * tr * tc * kv_head_loop * batch_size
        else:  # decode
            # QKT (per KV head and Grouped Q heads)
            inner_compute_cycles += (self.instr["M_BTMV"] + self.instr["H_PREFETCH_M"]) * math.ceil(
                inner_q_head_loop / (self.mlen // self.hlen)
            )
            # online softmax
            inner_compute_cycles += (
                4 * self.instr["V_BASIC"] + 2 * self.instr["S_BASIC"] + self.instr["S_EXP_FP"]
            ) * inner_q_head_loop
            # Compute PV
            inner_compute_cycles += math.ceil(head_dim / self.blen) * (self.instr["M_MV"]) * inner_q_head_loop
            # Compute O
            inner_compute_cycles += (2 * self.instr["V_BASIC"] + 1) * inner_q_head_loop
            overall_cycles = inner_compute_cycles * tr * tc * kv_head_loop * batch_size

        return self._roofline(overall_cycles, mem_bytes)

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

        # Unfused attention materialises the full S/P matrix, but this model does
        # not charge for spilling it; only the KV cache read is counted (one pass).
        mem_bytes = 2 * batch_size * kv_size * num_kv_heads * head_dim * self.kv_bytes

        if mode == "prefill":
            # S = Q (seq_len, num_attention_heads, head_dim) @ K^T (seq_len, num_kv_heads, head_dim) = (num_attention_heads, seq_len, seq_len)
            eff_btmm = self._effective_btmm(min(seq_len, self.mlen), min(kv_size, self.mlen))
            if multi_core_mode:
                single_batch_compute_cycles += (
                    (
                        4
                        + eff_btmm * math.ceil(seq_len / self.mlen) * math.ceil(kv_size / self.mlen)
                        + self.instr["H_PREFETCH_M"]
                    )
                    * kv_head_loop
                    * math.ceil(inner_q_head_loop / (self.mlen // self.hlen))
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
                    * math.ceil(inner_q_head_loop / (self.mlen // self.hlen))
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
        return self._roofline(overall_cycles, mem_bytes)

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

        # HBM traffic: router weights plus the weights of every expert actually
        # touched. Prefill touches all of them; decode touches at most
        # batch*top_k distinct experts (capped by num_experts).
        expert_weight_elems = 3 * hidden_size * intermediate_size
        touched_experts = num_experts if mode == "prefill" else min(num_experts, max(1, batch_size * expert_per_token))
        mem_bytes = (hidden_size * num_experts + touched_experts * expert_weight_elems) * self.weight_bytes

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

        return self._roofline(overall_cycles, mem_bytes)

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

        # HBM traffic: only the in-window slice of the KV cache is ever read, which
        # is what bounds sliding-window decode to O(1) memory per token.
        window_kv = min(seq_len if mode == "prefill" else kv_size, sliding_window_size)
        tr = math.ceil(seq_len / self.mlen) if mode == "prefill" else 1
        mem_bytes = tr * 2 * batch_size * window_kv * num_kv_heads * head_dim * self.kv_bytes

        if mode == "prefill":
            # Effective attention window per query position
            # Each query at position i attends to keys in range [max(0, i - sliding_window_size + 1), i]
            # Average effective KV length = (1 + sliding_window_size) / 2 for early tokens, sliding_window_size for later tokens
            # Simplified: use min(seq_len, sliding_window_size) as effective KV dimension
            effective_kv_len = min(seq_len, sliding_window_size)

            # QK^T: Q (seq_len, num_kv_heads, q_mult, head_dim) @ K^T (seq_len, num_kv_heads, head_dim)
            # Output shape: (num_kv_heads, q_mult, seq_len, effective_kv_len) = (num_attention_heads, seq_len, effective_kv_len)
            eff_btmm = self._effective_btmm(min(seq_len, self.mlen), min(effective_kv_len, self.mlen))
            if multi_core_mode:
                single_batch_compute_cycles += (
                    (
                        4
                        + eff_btmm * math.ceil(seq_len / self.mlen) * math.ceil(effective_kv_len / self.mlen)
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
        return self._roofline(overall_cycles, mem_bytes)

    def residual(self, hidden_size: int, seq_len: int, batch_size: int, mode: str = "prefill") -> int:
        """Residual connection cycle count."""
        iteration = hidden_size // self.vlen
        overall_cycles = 0
        mem_bytes = 0.0

        if mode == "prefill":
            compute_cycle = (self.instr["V_ADD_VV"] + 3) * seq_len * iteration * batch_size
            spill = self._spill_elems(hidden_size, seq_len, batch_size)
            if spill:
                overall_cycles += (
                    compute_cycle + (spill // (self.vlen * self.prefetch_v_amount)) * self.instr["H_PREFETCH_V"] * 2
                )
                mem_bytes = 2 * spill * self.act_bytes
            else:
                overall_cycles += compute_cycle
        else:
            compute_cycle = (self.instr["V_ADD_VV"] + 3) * iteration * batch_size
            overall_cycles += compute_cycle
        return self._roofline(overall_cycles, mem_bytes)

    def feed_forward(
        self, hidden_size: int, intermediate_size: int, seq_len: int, batch_size: int, mode: str = "prefill"
    ) -> int:
        """Feed-forward (MLP) layer cycle count."""
        overall_cycles = 0

        # HBM traffic: gate + up + down weight matrices, streamed once per call.
        mem_bytes = 3 * hidden_size * intermediate_size * self.weight_bytes

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

        return self._roofline(overall_cycles, mem_bytes)

    def embeddings(self, hidden_size: int, seq_len: int, batch_size: int, mode: str = "prefill") -> int:
        """Embedding layer cycle count."""
        setting_inst_num = 3
        overall_cycles = setting_inst_num * self.instr["S_BASIC"]
        tokens = seq_len * batch_size if mode == "prefill" else batch_size

        # HBM traffic: one gathered embedding row per token.
        mem_bytes = tokens * hidden_size * self.act_bytes

        overall_cycles += tokens * math.ceil(hidden_size / self.vlen) * self.instr["H_PREFETCH_V"]

        return self._roofline(overall_cycles, mem_bytes)

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

        # HBM traffic: the (hidden x vocab) unembedding matrix.
        mem_bytes = hidden_size * vocab_size * self.weight_bytes

        return self._roofline(overall_cycles, mem_bytes)

    def lm_head_full_seq(self, hidden_size: int, vocab_size: int, seq_len: int, batch_size: int) -> int:
        """LM head cycle count over full sequence (used by LLaDA: all positions need logits)."""
        setting_inst_num = 3
        overall_cycles = setting_inst_num * self.instr["S_BASIC"]

        # Matrix multiply: [batch_size * seq_len, hidden_size] x [hidden_size, vocab_size]
        overall_cycles += (
            math.ceil((batch_size * seq_len) / self.blen)
            * math.ceil(hidden_size / self.mlen)
            * math.ceil(vocab_size / self.blen)
            * self.instr["M_MM"]
        )

        mem_bytes = hidden_size * vocab_size * self.weight_bytes

        return self._roofline(overall_cycles, mem_bytes)

    def softmax_full_seq(self, vocab_size: int, seq_len: int, batch_size: int) -> int:
        """Softmax over vocab for all sequence positions (used by LLaDA for confidence scoring)."""
        # One softmax row per token position: batch_size * seq_len rows of length vocab_size
        loop_num = math.ceil(vocab_size / self.vlen)
        # softmax: max-reduce + exp + sum + divide = ~6 V_ ops per chunk
        overall_cycles = batch_size * seq_len * loop_num * 6 * self.instr["V_BASIC"]
        # The full logit tensor is far larger than vector SRAM: read once, write once.
        mem_bytes = 2 * batch_size * seq_len * vocab_size * self.act_bytes
        return self._roofline(overall_cycles, mem_bytes)

    # -------------------------------------------------------------------------
    # Mamba-2 / selective state-space primitives
    # -------------------------------------------------------------------------

    def linear(self, in_features: int, out_features: int, seq_len: int, batch_size: int, mode: str = "prefill") -> int:
        """Generic dense projection [tokens, in_features] @ [in_features, out_features].

        Used for Mamba-2's in_proj / out_proj, which are plain GEMMs with no RoPE
        and no KV cache, so `projection()` (attention-shaped) does not apply.
        """
        tokens = seq_len * batch_size if mode == "prefill" else batch_size
        overall_cycles = (
            math.ceil(tokens / self.blen)
            * math.ceil(in_features / self.mlen)
            * math.ceil(out_features / self.blen)
            * self.instr["M_MM"]
        )
        mem_bytes = in_features * out_features * self.weight_bytes
        return self._roofline(overall_cycles, mem_bytes)

    def causal_conv1d(
        self,
        conv_dim: int,
        conv_kernel: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
        activation: bool = True,
    ) -> int:
        """Causal depthwise conv1d over ``conv_dim`` channels with kernel ``conv_kernel``.

        Depthwise means there is no channel mixing, so this is NOT a GEMM: it is
        ``conv_kernel`` fused multiply-adds per (channel, token), vectorised across
        channels on the vector unit. In decode the (conv_kernel-1)-deep rolling
        window of past inputs is part of the recurrent cache: it is read, shifted
        and written back every token.
        """
        tokens = seq_len * batch_size if mode == "prefill" else batch_size
        chunks = math.ceil(conv_dim / self.vlen)

        # kernel taps: multiply-accumulate per tap, then bias. The depthwise weight
        # for tap k is a per-channel *vector*, hence V_MUL_VV not V_MUL_VF.
        overall_cycles = tokens * chunks * conv_kernel * (self.instr["V_MUL_VV"] + self.instr["V_ADD_VV"])
        overall_cycles += tokens * chunks * self.instr["V_ADD_VF"]
        if activation:  # SiLU, same 6-op budget the FFN uses
            overall_cycles += tokens * chunks * 6 * self.instr["V_BASIC"]

        conv_state_elems = batch_size * conv_dim * (conv_kernel - 1)
        if mode == "prefill":
            # window slides inside the vector unit; only the final state is written out
            mem_bytes = conv_dim * conv_kernel * self.weight_bytes + conv_state_elems * self.state_bytes
        else:
            # shift the rolling window by one position, then read + write it
            overall_cycles += tokens * chunks * self.instr["V_SHFT_V"]
            mem_bytes = conv_dim * conv_kernel * self.weight_bytes + 2 * conv_state_elems * self.state_bytes

        return self._roofline(overall_cycles, mem_bytes)

    def dt_activation(self, num_heads: int, seq_len: int, batch_size: int, mode: str = "prefill") -> int:
        """dt = clamp(softplus(dt_raw + dt_bias), dt_min, dt_max), one value per head per token.

        softplus is a single V_SOFTPLUS_V (ISA 0x3D); the clamp is V_MAX_VF + V_MIN_VF.
        Purely on-chip: dt is produced by in_proj and consumed by the scan.

        dt is [tokens, num_heads] and contiguous, so it is charged as
        ceil(tokens*num_heads / VLEN) full-width vector ops rather than one
        (mostly empty) op per token — num_heads is typically far below VLEN.
        """
        tokens = seq_len * batch_size if mode == "prefill" else batch_size
        vecs = math.ceil(tokens * num_heads / self.vlen)
        overall_cycles = vecs * (
            self.instr["V_ADD_VF"] + self.instr["V_SOFTPLUS_V"] + self.instr["V_MAX_VF"] + self.instr["V_MIN_VF"]
        )
        return self._roofline(overall_cycles, 0.0)

    def ssd_chunk_scan(
        self,
        num_heads: int,
        head_dim: int,
        state_size: int,
        n_groups: int,
        chunk_size: int,
        seq_len: int,
        batch_size: int,
    ) -> int:
        """Mamba-2 SSD chunked scan — the prefill form of the selective recurrence.

        Per chunk of ``chunk_size`` tokens the algorithm is (Dao & Gu, 2024 §6):

          cs      = cumsum(dt * A) within the chunk           [chunk] per head
          G       = C @ B^T                                   [chunk, chunk] per group
          L       = exp(segsum(cs)) with a causal mask        [chunk, chunk] per head
          Y_intra = (L o G) @ X                               [chunk, head_dim] per head
          h       = decay*h_prev + (B*decay)^T @ X_scaled     [state_size, head_dim] per head
          Y_inter = C @ h_prev                                [chunk, head_dim] per head
          Y       = Y_intra + Y_inter + D*X

        G is computed once per *group* (B and C are shared across the heads of a
        group); everything else is per head.

        Vector-lane convention: elementwise work over a dense tile is charged as
        ceil(elems/VLEN) full-width ops. Work that needs a *different broadcast
        scalar per row* (the decay subtraction, the X rescale) stays row-serial,
        because each row needs its own S_MAP_FP_V extraction first — which is why
        S_MAP_FP_V, not the matmuls, dominates this kernel at chunk_size=256.
        """
        num_chunks = math.ceil(seq_len / chunk_size)
        chunk = min(chunk_size, seq_len)

        c_blocks = math.ceil(chunk / self.blen)
        p_blocks = math.ceil(head_dim / self.blen)
        n_blocks_m = math.ceil(state_size / self.mlen)
        c_blocks_m = math.ceil(chunk / self.mlen)
        p_vec = math.ceil(head_dim / self.vlen)
        c_vec = math.ceil(chunk / self.vlen)

        # --- per group ---------------------------------------------------------
        # G = C @ B^T : [chunk, state_size] @ [state_size, chunk]
        per_group = c_blocks * n_blocks_m * c_blocks * self.instr["M_MM"]

        # --- per head ----------------------------------------------------------
        per_head = 0
        # cumulative sum of dt*A inside the chunk (prefix scan), then one broadcast
        # scalar per row of the decay matrix (S_MAP_FP_V, ISA 0x3E)
        per_head += c_vec * (self.instr["V_MUL_VV"] + self.instr["V_PS_V"])
        per_head += chunk * self.instr["S_MAP_FP_V"]
        # decay mask L = exp(cs_i - cs_j), causal: [chunk, chunk].
        # subtraction is row-serial (per-row broadcast cs_i), exp is packed.
        per_head += chunk * c_vec * self.instr["V_SUB_VF"]
        per_head += math.ceil(chunk * chunk / self.vlen) * self.instr["V_EXP_V"]
        # L o G : dense [chunk, chunk] elementwise
        per_head += math.ceil(chunk * chunk / self.vlen) * self.instr["V_MUL_VV"]
        # Y_intra = (L o G) @ X : [chunk, chunk] @ [chunk, head_dim]
        per_head += c_blocks * c_blocks_m * p_blocks * self.instr["M_MM"]
        # X_scaled = X * exp(cs_chunk - cs_t): one broadcast scalar per row
        per_head += chunk * self.instr["S_MAP_FP_V"] + chunk * p_vec * self.instr["V_MUL_VF"]
        # state update h_new = B_scaled^T @ X_scaled : [state_size, chunk] @ [chunk, head_dim]
        per_head += math.ceil(state_size / self.blen) * c_blocks_m * p_blocks * self.instr["M_TMM"]
        # h = h_prev * decay + h_new  over [state_size, head_dim]
        per_head += math.ceil(state_size * head_dim / self.vlen) * (self.instr["V_MUL_VF"] + self.instr["V_ADD_VV"])
        # Y_inter = C @ h_prev : [chunk, state_size] @ [state_size, head_dim]
        per_head += c_blocks * n_blocks_m * p_blocks * self.instr["M_MM"]
        # Y = Y_intra + Y_inter + D*X : dense [chunk, head_dim] elementwise
        per_head += math.ceil(chunk * head_dim / self.vlen) * (2 * self.instr["V_ADD_VV"] + self.instr["V_MUL_VF"])

        overall_cycles = num_chunks * batch_size * (n_groups * per_group + num_heads * per_head)

        # --- HBM traffic -------------------------------------------------------
        # X, B, C, dt for the whole sequence, plus the final recurrent state.
        # Only the part that does not fit in vector SRAM actually goes to HBM.
        d_inner = num_heads * head_dim
        act_elems = batch_size * seq_len * (d_inner + 2 * n_groups * state_size + num_heads)
        spill = max(0, act_elems - self.vector_sram_size)
        state_elems = batch_size * num_heads * head_dim * state_size
        mem_bytes = 2 * spill * self.act_bytes + state_elems * self.state_bytes

        return self._roofline(overall_cycles, mem_bytes)

    def ssd_recurrence_decode(
        self,
        num_heads: int,
        head_dim: int,
        state_size: int,
        n_groups: int,
        batch_size: int,
    ) -> int:
        """Mamba-2 single-token recurrence: h = dA*h + dB*x ; y = C.h + D*x.

        The state is ``num_heads * head_dim * state_size`` elements and is read AND
        written for every token — the O(1)-in-context term that replaces the
        linearly growing KV cache. This is where the bandwidth model matters: with
        no memory term the state update looks free.

        Approximation: the C.h contraction is charged head-serially as one M_MV per
        head over a [head_dim, state_size] tile. On a machine with MLEN=2048 and
        state_size=128 that leaves the matrix unit mostly idle; a real kernel would
        batch heads into the array, so this term is pessimistic by up to MLEN/state_size.
        """
        state_vec = math.ceil(head_dim * state_size / self.vlen)

        # dA = exp(dt * A), one scalar per head
        overall_cycles = math.ceil(num_heads / self.vlen) * (self.instr["V_MUL_VV"] + self.instr["V_EXP_V"])
        # dB*x outer product, dA*h decay, and the add — over the [head_dim, state_size]
        # state block of each head
        overall_cycles += num_heads * state_vec * (2 * self.instr["V_MUL_VF"] + self.instr["V_ADD_VV"])
        # y = C.h : [head_dim, state_size] @ [state_size]
        overall_cycles += (
            num_heads * math.ceil(head_dim / self.blen) * math.ceil(state_size / self.mlen) * self.instr["M_MV"]
        )
        # y += D*x
        overall_cycles += (
            num_heads * math.ceil(head_dim / self.vlen) * (self.instr["V_MUL_VF"] + self.instr["V_ADD_VV"])
        )
        overall_cycles *= batch_size

        # state read + state write, every token
        state_elems = batch_size * num_heads * head_dim * state_size
        mem_bytes = 2 * state_elems * self.state_bytes

        return self._roofline(overall_cycles, mem_bytes)

    # -------------------------------------------------------------------------
    # KDA (Kimi Delta Attention) — the gated delta rule
    # -------------------------------------------------------------------------
    #
    # Both methods below are structural derivations in the same style as the SSD
    # pair above, but unlike those they have a **measured** counterpart: the KDA
    # lowering exists, compiles, and runs as machine code on the transactional
    # emulator, so `test_kda_stage_instruction_mix.py` checks these formulas
    # against the compiler's real instruction counts at MLEN=64 rather than
    # leaving them unfalsifiable. Where the two disagree the formula is wrong.
    #
    # KDA's decay is **channel-wise on the key axis** -- one scalar per (token,
    # key), not one per token. That is what makes the state update a per-key
    # sweep rather than a single scaled add, and it is the single biggest
    # difference from Mamba-2's cost shape.

    def kda_recurrence_decode(
        self,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        batch_size: int,
    ) -> int:
        """KDA single-token gated delta rule, per token.

        With state ``S[key, value]`` held transposed (the layout the lowering
        uses so every sweep is a row progression and therefore a hardware loop)::

            a       = exp(gate)                      per key channel
            k_hat   = k * a ;  q_hat = q * a
            o       = S^T q_hat                      contraction over key
            e       = beta * (v - S^T k_hat)
            S       = S * a + k_hat (x) e

        Each of the three sweeps walks ``key_dim`` rows of ``value_dim`` lanes.
        The predict and update sweeps are `V_FMA_VF` — one instruction per row,
        which is what makes the cost constant in ``key_dim`` per row rather than
        the copy/multiply/add triple it replaced.

        **The scalar overhead is not a rounding term and is billed here.** Every
        sweep iteration advances its pointers with an explicit `S_ADDI_INT` per
        operand and reloads its FPRAM scalar with `S_LD_FP`. On the compiled
        kernel that is 50% and 25% of the dynamic instruction stream against
        18% for the arithmetic itself, so a model counting only the FMAs would
        be optimistic by roughly 4x in instruction count. Post-increment
        addressing would remove most of it; until the ISA has it, this is the
        machine that exists.

        Gates, L2 normalisation and the output norm are billed separately by the
        caller, the same split `ssd_recurrence_decode` uses for `dt`.
        """
        value_vec = math.ceil(value_dim / self.vlen)

        # One sweep iteration: the vector op, its FPRAM scalar, and the pointer
        # arithmetic for both operands.
        sweep_row = self.instr["S_LD_FP"] + 2 * self.instr["S_ADDI_INT"]
        fma_row = self.instr["V_FMA_VF"] * value_vec + sweep_row
        mul_row = self.instr["V_MUL_VF"] * value_vec + sweep_row

        # a = exp(gate), one value per (head, key)
        overall_cycles = math.ceil(num_heads * key_dim / self.vlen) * (self.instr["V_MUL_VV"] + self.instr["V_EXP_V"])
        # k_hat and q_hat: two elementwise products over the key-width vectors
        overall_cycles += 2 * math.ceil(num_heads * key_dim / self.vlen) * self.instr["V_MUL_VV"]
        # predict o = S^T q_hat, and the error term e's S^T k_hat: two sweeps
        overall_cycles += 2 * num_heads * key_dim * fma_row
        # state decay S *= a, then the rank-1 accumulate S += k_hat (x) e
        overall_cycles += num_heads * key_dim * (mul_row + fma_row)
        overall_cycles *= batch_size

        # The state is read and written every token. This is the O(1)-in-context
        # term that replaces a growing KV cache -- and the reason the bandwidth
        # model matters: with no memory term the update looks free.
        state_elems = batch_size * num_heads * key_dim * value_dim
        mem_bytes = 2 * state_elems * self.state_bytes

        return self._roofline(overall_cycles, mem_bytes)

    def kda_chunk_prefill(
        self,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        chunk_size: int,
        seq_len: int,
        batch_size: int,
        row_granular_prefetch: bool = False,
    ) -> int:
        """KDA chunked prefill: seven matrix products per chunk, per head.

        The chunk form collapses ``C`` sequential rank-1 updates into::

            M   = tril(k_hat @ k_tilde^T, -1)        [C, C]
            N   = tril(q_hat @ k_tilde^T)            [C, C], diagonal included
            T   = (I + tril(diag(beta) M, -1))^-1 diag(beta)
            W   = V - k_hat @ S_0^T                  [C, value]
            E   = T @ W
            out = scale * (q_hat @ S_0^T + N @ E)
            S_C = A_C * S_0 + E^T @ (k * A_C / A)

        **Every one of the seven has a dynamic second operand** -- there is no
        weight here, only activations and state -- so each is spilled to HBM and
        re-prefetched into MRAM, because MRAM is writable only by
        `H_PREFETCH_M`. That spill traffic is the dominant memory term and is
        billed below; a model that assumed a weight-resident operand would
        under-count it by the number of products.

        The UT transform is a forward substitution, one `V_FMA_VF` sweep per row
        of the chunk, so it costs ``C`` sweeps and not ``C^2`` instructions.

        ``chunk_size`` is bounded by bf16 range, not by tiling: ``1/A`` reaches
        ``exp(chunk * |gate_lower_bound|)`` and overflows past 17 at Kimi's
        gate floor of -5. 16 is what the lowering uses.
        """
        chunk = min(chunk_size, seq_len)
        num_chunks = math.ceil(seq_len / chunk_size)

        c_blocks = math.ceil(chunk / self.blen)
        c_blocks_m = math.ceil(chunk / self.mlen)
        k_blocks_m = math.ceil(key_dim / self.mlen)
        v_blocks_m = math.ceil(value_dim / self.mlen)
        k_vec = math.ceil(key_dim / self.vlen)
        v_vec = math.ceil(value_dim / self.vlen)
        c_vec = math.ceil(chunk / self.vlen)

        # --- per chunk, per head ----------------------------------------------
        # The cumulative decay is a running product, not exp of a running sum:
        # the cumulative log-decay reaches chunk * gate_lower_bound = -80, where
        # bf16's ulp is 0.31, so exponentiating a stored sum costs 17% relative
        # error on A. That makes it a sequential scan of `chunk` steps, not one
        # matmul against a triangular ones matrix the way Mamba-2's cumsum is.
        # The scan stages the previous timestep at the current row before the
        # multiply, because the row ops take one row list for both operands.
        # That staging copy is a zero-fill plus an add, per step, per key block.
        per_chunk = (
            chunk * k_blocks_m * (self.instr["V_MUL_VV"] + self.instr["V_ADD_VV"] + 2 * self.instr["S_ADDI_INT"])
        )
        # k_tilde = k / A, then k_hat / q_hat
        per_chunk += chunk * k_vec * (self.instr["V_RECI_V"] + 3 * self.instr["V_MUL_VV"])

        # The two grams, contracting over key: [C, key] @ [C, key]^T -> [C, C]
        per_chunk += 2 * c_blocks * k_blocks_m * c_blocks_m * self.instr["M_TMM"]
        # The causal mask on N
        per_chunk += chunk * c_vec * self.instr["V_MUL_VV"]

        # UT transform: one FMA sweep per row, row i walking j < i
        per_chunk += sum(
            i * (self.instr["V_FMA_VF"] * c_vec + self.instr["S_LD_FP"] + 2 * self.instr["S_ADDI_INT"])
            for i in range(chunk)
        )
        per_chunk += chunk * (self.instr["S_MAP_FP_V"] + c_vec * self.instr["V_MUL_VF"])

        # W^T = v^T - S_0 @ k_hat^T ; E^T = W^T @ T^T
        per_chunk += 2 * math.ceil(value_dim / self.blen) * c_blocks_m * k_blocks_m * self.instr["M_TMM"]
        per_chunk += value_dim * c_vec * self.instr["V_SUB_VV"]
        # out = scale * (q_hat @ S_0^T + N @ E)
        per_chunk += 2 * c_blocks * v_blocks_m * k_blocks_m * self.instr["M_TMM"]
        per_chunk += chunk * v_vec * (self.instr["V_ADD_VV"] + self.instr["V_MUL_VF"])
        # S_C = A_C * S_0 + E^T @ k_end
        per_chunk += math.ceil(value_dim / self.blen) * c_blocks_m * k_blocks_m * self.instr["M_MM"]
        per_chunk += value_dim * k_vec * (self.instr["V_MUL_VV"] + self.instr["V_ADD_VV"])

        # The six spills are instructions too, not only bytes. Each one zero-fills
        # the rows past its live data (the prefetch pulls a whole mlen x mlen
        # block regardless), stores the tile a writeback-amount of rows at a
        # time, and is prefetched back into MRAM. Leaving this out is what made a
        # first version of this method 30% under the compiler's measured count
        # while the decode form was within 2%.
        rows_per_store = max(1, self.writeback_v_amount)

        def spill(col_blocks: int, live_rows: int) -> int:
            """One spill of a tile `col_blocks` column blocks wide.

            The width matters and is not a detail. A VRAM matrix wider than
            `mlen` is column-block-major, and both the zero-fill and the store
            walk every block -- so a `[chunk, key_dim]` tile at `key_dim` 128
            against `mlen` 64 costs twice what the same tile one block wide
            does. Billing every spill at one block under-counted the key axis by
            half, which showed up as the model scaling with `key_dim` at half
            the rate the compiler does.

            **The height is where the machine's width leaks in.** Today the
            prefetch pulls a whole `mlen x mlen` MRAM block -- `k_block_count`
            selects whole blocks and cannot trim a partial one -- so the tile has
            to be `mlen` rows tall and every row past `live_rows` must be zeroed
            before the store, or the matmul contracts against whatever was there.
            The fill therefore scales with `mlen` while the data scales with
            `chunk`, and KDA's chunk cannot grow to meet it: `1/A` overflows bf16
            past 17.

            `row_granular_prefetch` models the instruction set that does not have
            that constraint -- a prefetch taking a row count. Then only the live
            rows are stored and none need zeroing, and the term stops tracking
            the machine's width. It is a hypothetical: no such instruction
            exists, and this switch is here to price it rather than to assume it.
            """
            rows = live_rows if row_granular_prefetch else self.mlen
            fill = 0 if row_granular_prefetch else self.mlen * self.instr["V_MUL_VF"]
            return col_blocks * (
                fill + math.ceil(rows / rows_per_store) * self.instr["H_STORE_V"] + self.instr["H_PREFETCH_M"]
            )

        # k_tilde, k_hat and k_end are [chunk, key]; t_mat is [chunk, chunk];
        # the state is [value, key] and the error [value, chunk].
        per_chunk += (
            3 * spill(k_blocks_m, chunk)
            + spill(c_blocks_m, chunk)
            + spill(k_blocks_m, value_dim)
            + spill(c_blocks_m, value_dim)
        )

        # `A_C` is broadcast down a tile by doubling the filled span, which is
        # ceil(log2(n)) + 1 copies rather than n. Twice per chunk.
        per_chunk += (
            2 * (math.ceil(math.log2(max(chunk, value_dim))) + 1) * (self.instr["V_MUL_VF"] + self.instr["V_ADD_VV"])
        )

        overall_cycles = per_chunk * num_chunks * num_heads * batch_size

        # --- memory ------------------------------------------------------------
        # Six spills per chunk per head, each written once and prefetched once.
        # Sized from the tiles the lowering actually spills: k_tilde, k_hat and
        # k_end are [C, key]; t_mat is [C, C]; state and err are [value, *].
        spill_elems = 3 * chunk * key_dim + chunk * chunk + value_dim * key_dim + value_dim * chunk
        mem_bytes = 2 * spill_elems * self.act_bytes * num_chunks * num_heads * batch_size
        # The carried state, read at the start of each chunk and written at its end.
        mem_bytes += 2 * num_chunks * num_heads * key_dim * value_dim * self.state_bytes * batch_size

        return self._roofline(overall_cycles, mem_bytes)

    def gated_rms_norm(self, dim: int, seq_len: int, batch_size: int, mode: str = "prefill") -> int:
        """RMSNorm(x * silu(z)) — Mamba-2's output norm, gated by the z branch."""
        tokens = seq_len * batch_size if mode == "prefill" else batch_size
        chunks = math.ceil(dim / self.vlen)
        setting_inst_num = 10
        loop_inst_num = 8

        overall_cycles = setting_inst_num * self.instr["S_BASIC"]
        # SiLU on the gate branch + the elementwise product
        overall_cycles += tokens * chunks * (6 * self.instr["V_BASIC"] + self.instr["V_MUL_VV"])
        # the RMSNorm itself, same 8-op-per-chunk budget as rms_layer()
        overall_cycles += tokens * chunks * loop_inst_num * self.instr["V_BASIC"]

        mem_bytes = 0.0
        if mode == "prefill":
            spill = self._spill_elems(dim, seq_len, batch_size)
            if spill:
                overall_cycles += (spill // (self.vlen * self.prefetch_v_amount)) * self.instr["H_PREFETCH_V"] * 2
                mem_bytes = 2 * spill * self.act_bytes

        return self._roofline(overall_cycles, mem_bytes)
