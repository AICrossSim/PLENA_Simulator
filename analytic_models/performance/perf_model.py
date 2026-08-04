"""
PLENA Hardware Performance Model.

The per-operation cycle library for one decoder layer: each method returns the cycle
count for an operation (RMSNorm, Q/K/V projection, attention output projection,
flash-attention, residual, MLP, embedding, LM head, softmax) in either "prefill" or
"decode" mode. Cycle costs come from the pipelined instruction latencies built from
customISA_lib.json for the current hardware config (MLEN/BLEN/VLEN/HLEN).

Consumed by the disaggregated decode-serving model in disagg_decode.py.
"""

import json
import math

import toml
from pydantic import BaseModel, Field, model_validator

try:
    from .decode_timing import (
        DRAIN_OVERLAPPED,
        IDEAL_MATRIX_PIPELINE,
        RTL_SERIALIZED,
        TIMING_MODES,
        matrix_issue_cycles,
        projection_mm_events,
    )
except ImportError:
    from decode_timing import (
        DRAIN_OVERLAPPED,
        IDEAL_MATRIX_PIPELINE,
        RTL_SERIALIZED,
        TIMING_MODES,
        matrix_issue_cycles,
        projection_mm_events,
    )

try:
    from .packed_q1_timing import (
        PackedQ1TimingContract,
        packed_q1_matrix_histogram,
    )
except ImportError:
    from packed_q1_timing import (
        PackedQ1TimingContract,
        packed_q1_matrix_histogram,
    )

# Address cursors a matmul loop body advances per matrix instruction: the
# operand it reads and the result it writes.
MATMUL_ADDRESS_CURSORS = 2

# SiLU evaluates x / (1 + exp(-x)) over a VLEN-wide chunk: negate, exponentiate,
# add one, reciprocate, and two multiplies against the gate. Five of those are
# elementwise ops priced at V_BASIC; the reciprocal runs on the vector special
# function unit and costs V_RECI_V, which is the more expensive of the two.
SILU_BASIC_OPS = 5

# Setup issues the LM head runs once: the HBM scale and stride registers and the
# activation base.
LM_HEAD_SETTING_INSTRUCTIONS = 3

# Address cursors one `M_BTMM` + `M_BMM_WO` pair advances inside the score-tile
# loop: the key operand it reads and the score column it drains into.
BROADCAST_TILE_ADDRESS_CURSORS = 2
# Address cursors one query-row block of that nest sets up: the key operand base,
# the score column base, and the query and score row advances.
BROADCAST_TILE_BASE_CURSORS = 4

# Address cursors an RMSNorm row sets up before its two passes: the read base,
# the sum-of-squares base, and, out of place, the write base.
RMSNORM_BASE_CURSORS_IN_PLACE = 2
RMSNORM_BASE_CURSORS_OUT_OF_PLACE = 3
# Address cursors one VLEN chunk advances: one over the sum-of-squares pass and,
# in the scaling pass, the read cursor plus the write cursor when they differ.
RMSNORM_CHUNK_CURSORS_IN_PLACE = 2
RMSNORM_CHUNK_CURSORS_OUT_OF_PLACE = 3
# The sum-of-squares pass and the scaling pass are separate hardware loops.
RMSNORM_LOOP_LEVELS = 2

# Address cursors a residual loop body advances: the two operands it reads and
# the result it writes.
RESIDUAL_ADDRESS_CURSORS = 3

# Address cursors an online-softmax row advances: the score row it walks and the
# three running-state slots (running max, its residual, running sum).
SOFTMAX_ADDRESS_CURSORS = 4

# Address cursors the output update advances per row: the accumulator, the PV
# result it folds in, and the running-max residual it scales by.
OUTPUT_UPDATE_ADDRESS_CURSORS = 3


def ffn_decode_auxiliary_histogram(
    *,
    mlen: int,
    blen: int,
    vlen: int,
    hidden_size: int,
    intermediate_size: int,
    rows: int,
) -> dict[str, int]:
    """Dynamic non-compute instructions emitted by the looped decode FFN.

    The looped template has three nested projection loops: MLEN-wide output
    blocks, BLEN-wide output tiles, and BLEN-row activation blocks.  Reduction
    tiles are unrolled inside the innermost loop.  Counting those nests gives
    the address, control, setting, and weight-prefetch issues below without
    using an emulator-derived correction factor.

    Each logical address materialisation is one issue here.  Compiler
    immediate legalisation can expand a large address into additional scalar
    instructions; those geometry- and allocation-dependent instructions are
    intentionally outside this shape-only model.
    """
    dimensions = {
        "mlen": mlen,
        "blen": blen,
        "vlen": vlen,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "rows": rows,
    }
    if any(value <= 0 for value in dimensions.values()):
        raise ValueError("FFN dimensions must be positive")
    if mlen % blen:
        raise ValueError("mlen must be divisible by blen")

    output_tiles = mlen // blen
    row_tiles = math.ceil(rows / blen)
    hidden_tiles = math.ceil(hidden_size / mlen)
    intermediate_tiles = math.ceil(intermediate_size / mlen)

    def projection_histogram(
        *,
        output_blocks: int,
        reduction_tiles: int,
        setup_addi: int,
        activation_setup_addi: int,
    ) -> dict[str, int]:
        # Two prefetch cursors, three compute cursors, and two outer-loop
        # advances surround each MLEN output block.  Each activation row body
        # has two base cursors, two advances, and two more advances between
        # consecutive reduction tiles.
        addi_per_row = 2 * reduction_tiles + 2
        addi_per_output_block = (
            2
            + 2 * reduction_tiles
            + 3
            + output_tiles
            * (
                activation_setup_addi
                + row_tiles * addi_per_row
                + 3
            )
            + 2
        )
        return {
            "S_ADDI_INT": setup_addi + output_blocks * addi_per_output_block,
            "S_ADD_INT": output_blocks * output_tiles,
            "C_LOOP_START": 1 + output_blocks + output_blocks * output_tiles,
            "C_LOOP_END": output_blocks
            + output_blocks * output_tiles
            + output_blocks * output_tiles * row_tiles,
            "H_PREFETCH_M": output_blocks * reduction_tiles,
        }

    up = projection_histogram(
        output_blocks=intermediate_tiles,
        reduction_tiles=hidden_tiles,
        setup_addi=1,
        activation_setup_addi=1,
    )
    # Gate resets both result bases before entering the same projection nest.
    gate = projection_histogram(
        output_blocks=intermediate_tiles,
        reduction_tiles=hidden_tiles,
        setup_addi=3,
        activation_setup_addi=1,
    )
    # Down projection setup includes scale/stride values, both result bases,
    # and its HBM cursor.  Its activation base is loaded and then copied.
    down = projection_histogram(
        output_blocks=hidden_tiles,
        reduction_tiles=intermediate_tiles,
        setup_addi=6,
        activation_setup_addi=2,
    )

    histogram = {
        "S_ADDI_INT": 5,  # common scale, stride, reset, and result bases
        "S_ADD_INT": 0,
        "S_LD_FP": 1,
        "C_SET_SCALE_REG": 2,
        "C_SET_STRIDE_REG": 2,
        "C_LOOP_START": 0,
        "C_LOOP_END": 0,
        "H_PREFETCH_M": 0,
    }
    for projection in (up, gate, down):
        for opcode, count in projection.items():
            histogram[opcode] += count

    silu_iterations = rows * math.ceil(intermediate_size / vlen)
    histogram["S_ADDI_INT"] += 3 + 2 * silu_iterations
    histogram["C_LOOP_START"] += 1
    histogram["C_LOOP_END"] += silu_iterations
    return histogram

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
        matrix_depth = getattr(self, "MATRIX_SRAM_SIZE", None)
        if matrix_depth is not None and matrix_depth < 4 * self.MLEN:
            raise ValueError(
                "MATRIX_SRAM_SIZE must hold at least four MLEN tiles"
            )
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


def build_pipelined_latency(
    hardware_config: HardwareConfig,
    custom_isa_path: str,
    timing_mode: str = RTL_SERIALIZED,
) -> InstructionLatency:
    """
    Build pipelined instruction latency from customISA_lib.json.
    Evaluates expressions using hardware config values.

    The "pipelined" column is the effective per-instruction cost inside a
    back-to-back sequence. For matrix accumulates (M_MM/M_TMM/M_MV/...) it
    equals "alone": the RTL pipeline controller stalls every matrix op behind
    an active MCU, so consecutive accumulates serialize at the full
    feed+wavefront cost. Only the *_WO writeouts pipeline (they fold into the
    accumulate's drain, hence cost 1). JSON cannot carry comments, so this is
    the authoritative note for those values.

    Args:
        hardware_config: Validated hardware configuration
        custom_isa_path: Path to customISA_lib.json

    Returns:
        InstructionLatency: Validated instruction latencies
    """
    if timing_mode not in TIMING_MODES:
        raise ValueError(f"unknown timing mode {timing_mode!r}")
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
    matrix_opcodes = (
        "M_MM",
        "M_TMM",
        "M_BMM",
        "M_BTMM",
        "M_MM_WO",
        "M_BMM_WO",
        "M_MV",
        "M_TMV",
        "M_BMV",
        "M_BTMV",
        "M_MV_WO",
        "M_BMV_WO",
    )
    # `rtl_serialized` is the only contract whose latencies must equal the
    # standalone ISA table; the other two deliberately depart from it.
    if timing_mode != RTL_SERIALIZED:
        for opcode in matrix_opcodes:
            latencies[opcode] = matrix_issue_cycles(
                opcode,
                hardware_config.BLEN,
                timing_mode,
                mlen=hardware_config.MLEN,
                hlen=hardware_config.HLEN,
            )
    else:
        for opcode in matrix_opcodes:
            expected = matrix_issue_cycles(
                opcode,
                hardware_config.BLEN,
                timing_mode,
                mlen=hardware_config.MLEN,
                hlen=hardware_config.HLEN,
            )
            observed = eval(
                custom_isa_lib[opcode]["alone"],
                {"__builtins__": {}},
                configs,
            )
            if observed != expected:
                raise ValueError(
                    f"standalone ISA latency for {opcode} differs from the "
                    f"{timing_mode} timing contract"
                )
            latencies[opcode] = expected

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

    def __init__(
        self,
        hardware_config: HardwareConfig,
        custom_isa_path: str,
        timing_mode: str = RTL_SERIALIZED,
    ):
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
        self.timing_mode = timing_mode

        # Build validated instruction latencies
        self.instr = build_pipelined_latency(
            hardware_config,
            custom_isa_path,
            timing_mode=timing_mode,
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _issue_overhead(self, matrix_instructions: int, loop_bodies: int) -> int:
        """Address and loop issues serialized around a matmul loop body.

        A matmul loop body walks two cursors — the operand it reads and the
        result it writes — and advances each with an `S_ADDI_INT` per matrix
        instruction it issues, then closes the loop once per body with a
        `C_LOOP_END`. The pipeline controller issues these ahead of the matrix
        op rather than hiding them behind it, so they are part of the stage's
        serialized cost. Costing only the matrix instructions leaves this out
        and understates every matmul-bearing stage by the same shape.
        """
        return (
            matrix_instructions
            * MATMUL_ADDRESS_CURSORS
            * self.instr["S_ADDI_INT"]
            + loop_bodies * self.instr["C_LOOP_END"]
        )

    def _matmul_compute_and_drain(self, mm_events: int, reduction_tiles: int) -> int:
        """Matrix accumulate and writeout cycles without loop bookkeeping."""
        if reduction_tiles <= 0:
            raise ValueError("reduction tiles must be positive")
        writeouts = mm_events // reduction_tiles
        return (
            mm_events * self.instr["M_MM"]
            + writeouts * self.instr["M_MM_WO"]
        )

    def _matmul_and_drain(self, mm_events: int, reduction_tiles: int) -> int:
        """Cycles for `mm_events` accumulate issues plus the writeouts draining them.

        One `M_MM_WO` drains each accumulation group, and a group spans
        `reduction_tiles` accumulate issues over the reduction dimension, so the
        drain count is `mm_events / reduction_tiles`. The RTL serializes the
        drain behind the accumulate; the pipeline oracle prices it at 1 cycle.
        The loop that issues them also costs its address and loop instructions.
        """
        writeouts = mm_events // reduction_tiles
        compute = self._matmul_compute_and_drain(mm_events, reduction_tiles)
        return compute + self._issue_overhead(mm_events + writeouts, writeouts)

    def _effective_btmm(self, eff_rows: int, eff_cols: int) -> int:
        """Effective BTMM cost for a tile of actual size eff_rows × eff_cols.

        The RTL retires BLEN rows per per-head matmul issue (`PH_DRAIN_ROWS` in
        `mxint_systolic_mcu.sv`), so a tile of eff_rows × eff_cols costs
        ceil(eff_rows/BLEN) × ceil(eff_cols/BLEN) serialized issues. Each issue
        is priced like any other matmul, which keeps QK^T in the same timing
        mode as the PV term below rather than at the MAC-limited floor.
        """
        return (
            math.ceil(eff_rows / self.blen)
            * math.ceil(eff_cols / self.blen)
            * self.instr["M_MM"]
        )

    def _broadcast_qkt(self, eff_rows: int, eff_cols: int) -> int:
        """Cycles for one score tile of `eff_rows` queries by `eff_cols` keys.

        `M_BTMM` reduces over the head dimension inside a single issue, so every
        issue completes a BLEN x BLEN block for each head lane and is drained by
        its own `M_BMM_WO`; there is no accumulation group to amortize the drain
        over. The two-deep loop nest that walks the tile advances the key
        operand and the score column per inner iteration, and reloads its bases
        per outer iteration.
        """
        row_blocks = math.ceil(eff_rows / self.blen)
        column_blocks = math.ceil(eff_cols / self.blen)
        issues = row_blocks * column_blocks
        compute = issues * (self.instr["M_BTMM"] + self.instr["M_BMM_WO"])
        inner = issues * (
            BROADCAST_TILE_ADDRESS_CURSORS * self.instr["S_ADDI_INT"]
            + self.instr["C_LOOP_END"]
        )
        outer = row_blocks * (
            BROADCAST_TILE_BASE_CURSORS * self.instr["S_ADDI_INT"]
            + self.instr["C_LOOP_START"]
            + self.instr["C_LOOP_END"]
        )
        return compute + inner + outer

    # -------------------------------------------------------------------------
    # Layer-level latency computation methods
    # -------------------------------------------------------------------------

    def rms_layer(
        self,
        hidden_size: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
        out_of_place: bool = True,
    ) -> int:
        """RMSNorm layer cycle count.

        Each row runs two passes over its VLEN chunks — a vector sum-reduction
        for the mean square and a vector scale — separated by the scalar
        reciprocal-sqrt chain. The chain runs once per row, not once per chunk,
        so it is charged outside the chunk term; conflating the two overstates
        every row whose hidden size exceeds VLEN.

        `out_of_place` writes the normalized row to a separate tensor, which is
        what a transformer block needs so the pre-norm activation survives as
        its residual. It costs one extra address cursor per chunk and one more
        base register per row, and saves the whole copy pass a residual snapshot
        would otherwise take.
        """
        setting_inst_num = 10
        chunk_cursors = (
            RMSNORM_CHUNK_CURSORS_OUT_OF_PLACE
            if out_of_place
            else RMSNORM_CHUNK_CURSORS_IN_PLACE
        )
        base_cursors = (
            RMSNORM_BASE_CURSORS_OUT_OF_PLACE
            if out_of_place
            else RMSNORM_BASE_CURSORS_IN_PLACE
        )
        chunk_inst_num = (
            self.instr["V_RED_SUM"]
            + self.instr["V_MUL_VV"]
            # Scaling the row by the reciprocal RMS is the pass that writes the
            # normalized result, so it is part of every chunk.
            + self.instr["V_MUL_VF"]
            + chunk_cursors * self.instr["S_ADDI_INT"]
        )
        row_inst_num = (
            # The +eps add and the accumulator reset that closes the row.
            2 * self.instr["S_ADD_FP"]
            + self.instr["S_MUL_FP"]
            + self.instr["S_SQRT_FP"]
            + self.instr["S_RECI_FP"]
            + base_cursors * self.instr["S_ADDI_INT"]
            + RMSNORM_LOOP_LEVELS
            * (self.instr["C_LOOP_START"] + self.instr["C_LOOP_END"])
        )
        loop_num = hidden_size // self.vlen
        row_cycles = row_inst_num + loop_num * chunk_inst_num
        overall_cycles = 0

        if mode == "prefill":
            compute_cycle = (
                setting_inst_num * self.instr["S_BASIC"]
                + row_cycles * seq_len * batch_size
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
                setting_inst_num * self.instr["S_BASIC"] + row_cycles * batch_size
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
        kv_projection_width: int | None = None,
    ) -> int:
        """Q, K, V projection + RoPE cycle count.

        `kv_projection_width` is the K/V out_features the lowering actually
        computes. It defaults to the logical `num_kv_heads * head_dim`; a
        lowering that zero-pads K/V to a tile boundary passes the padded width,
        which costs real matmul issues without changing attention's GQA shape.
        """
        compute_cycle = 0
        overall_cycles = 0
        bs_dim = seq_len * batch_size
        input_width = hidden_size
        query_width = num_attention_heads * head_dim
        kv_width = (
            num_kv_heads * head_dim
            if kv_projection_width is None
            else kv_projection_width
        )
        rows = bs_dim if mode == "prefill" else batch_size
        q_events, k_events, v_events = projection_mm_events(
            rows=rows,
            input_width=hidden_size,
            query_width=query_width,
            kv_width=kv_width,
            mlen=self.mlen,
            blen=self.blen,
        )

        if mode == "prefill":
            # Projection of Q
            compute_cycle += self._matmul_and_drain(
                q_events, math.ceil(input_width / self.mlen)
            )
            # RoPE of Q
            compute_cycle += num_attention_heads * math.ceil(bs_dim / self.vlen) * self.instr["V_BASIC"]
            # Projection of K
            compute_cycle += self._matmul_and_drain(
                k_events, math.ceil(input_width / self.mlen)
            )
            # RoPE of K
            compute_cycle += num_kv_heads * math.ceil(bs_dim / self.vlen) * self.instr["V_BASIC"]
            # Projection of V
            compute_cycle += self._matmul_and_drain(
                v_events, math.ceil(input_width / self.mlen)
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
                * math.ceil(
                    batch_size * seq_len * kv_width
                    / (self.vlen * self.prefetch_v_amount)
                )
                * self.instr["H_STORE_V"]
            )
        else:  # decode
            # Projection of Q
            compute_cycle += self._matmul_and_drain(
                q_events, math.ceil(input_width / self.mlen)
            )
            # RoPE of Q
            compute_cycle += num_attention_heads * math.ceil(bs_dim / self.vlen) * self.instr["V_BASIC"]
            # Projection of K
            compute_cycle += self._matmul_and_drain(
                k_events, math.ceil(input_width / self.mlen)
            )
            # RoPE of K
            compute_cycle += num_kv_heads * math.ceil(bs_dim / self.vlen) * self.instr["V_BASIC"]
            # Projection of V
            compute_cycle += self._matmul_and_drain(
                v_events, math.ceil(input_width / self.mlen)
            )
            overall_cycles += compute_cycle
            # Store K,V Cache
            overall_cycles += (
                2
                * math.ceil(
                    batch_size * kv_width
                    / (self.vlen * self.prefetch_v_amount)
                )
                * self.instr["H_STORE_V"]
            )
        return overall_cycles

    def output_projection(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill",
    ) -> int:
        """Output projection cycle count: A_o = A_w @ W_O

        Maps the attention output (num_attention_heads * head_dim) back to hidden_size.
        Same systolic tiling as `projection`'s Q matmul, with the contraction dimension
        = attn_dim (heads*head_dim) and the output dimension = hidden_size.
        """
        attn_dim = num_attention_heads * head_dim
        rows = (seq_len * batch_size) if mode == "prefill" else batch_size
        reduction_tiles = math.ceil(attn_dim / self.mlen)
        return self._matmul_and_drain(
            math.ceil(rows / self.blen)
            * reduction_tiles
            * math.ceil(hidden_size / self.blen),
            reduction_tiles,
        )

    def flash_attention(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        kv_size: int,
        batch_size: int,
        mode: str = "prefill",
        packed_q1: bool = False,
        packed_q1_contract: PackedQ1TimingContract | None = None,
        batch_packed: bool = False,
    ) -> int:
        """Flash attention cycle count (assumes GQA mode)."""
        inner_compute_cycles = 0
        overall_cycles = 0

        kv_head_loop = num_kv_heads
        inner_q_head_loop = num_attention_heads // num_kv_heads
        tr = math.ceil(seq_len / self.mlen)
        tc = math.ceil(kv_size / self.mlen)
        if mode == "prefill":
            # Rows of the query tile actually occupied. The softmax, PV and O
            # terms below are per query row: each row carries its own running
            # max/sum and its own output tile. Sizing them by the key count
            # instead only agrees when seq_len == kv_size, which holds for
            # prefill but not for a packed decode tile, where a batch of q=1
            # tokens fills far fewer rows than the cache has keys.
            query_rows = min(seq_len, self.mlen)
            key_rows = min(kv_size, self.mlen)
            # QKT: the tile-covering M_BTMM nest, one broadcast group at a time.
            inner_compute_cycles += (
                4 + self._broadcast_qkt(query_rows, key_rows)
                + self.instr["H_PREFETCH_M"]
            ) * math.ceil(inner_q_head_loop / (self.mlen // self.hlen))
            # Online softmax, one pass per query row per Q head. The opcodes are
            # the loop body of `asm_templates/flashattn/online_softmax.py`: the
            # running max and sum are vector reductions, not V_BASIC ops, and
            # each row carries its running (m, l) state through FP SRAM.
            softmax_row = (
                2 * self.instr["V_BASIC"]
                + self.instr["V_RED_MAX"]
                + self.instr["V_EXP_V"]
                + self.instr["V_RED_SUM"]
                + 2 * self.instr["S_LD_FP"]
                + 3 * self.instr["S_ST_FP"]
                + 2 * self.instr["S_ADD_FP"]
                + self.instr["S_SUB_FP"]
                + self.instr["S_MUL_FP"]
                + self.instr["S_EXP_FP"]
                # The row walks its score row and its three state slots, and
                # closes the hardware loop once.
                + SOFTMAX_ADDRESS_CURSORS * self.instr["S_ADDI_INT"]
                + self.instr["C_LOOP_END"]
            )
            inner_compute_cycles += query_rows * softmax_row * inner_q_head_loop
            # PV, and the writeout that drains each accumulation group. The
            # reduction is one MLEN-wide key block, so every accumulate is
            # followed by its own drain.
            pv_events = (
                math.ceil(head_dim / self.blen)
                * math.ceil(query_rows / self.blen)
                * inner_q_head_loop
            )
            inner_compute_cycles += 4 + self._matmul_and_drain(pv_events, 1)
            # Compute O and the 1/l row scaling.
            inner_compute_cycles += (
                query_rows
                * (
                    2 * self.instr["V_BASIC"]
                    + self.instr["S_LD_FP"]
                    + self.instr["S_RECI_FP"]
                    + OUTPUT_UPDATE_ADDRESS_CURSORS * self.instr["S_ADDI_INT"]
                    + self.instr["C_LOOP_END"]
                )
                * inner_q_head_loop
            )
            overall_cycles = inner_compute_cycles * tr * tc * kv_head_loop * batch_size
        elif packed_q1:
            if seq_len != 1:
                raise ValueError("PackedKV cached timing requires q_len=1")
            if packed_q1_contract is not None:
                point = packed_q1_contract.point(kv_size)
                missing = [
                    opcode
                    for opcode, count in point.opcode_histogram
                    if count and opcode not in self.instr
                ]
                if missing:
                    raise ValueError(
                        f"PackedKV compiler trace has unpriced opcodes {missing}"
                    )
                return sum(
                    count * self.instr[opcode]
                    for opcode, count in point.opcode_histogram
                )

            matrix_histogram = packed_q1_matrix_histogram(
                cache_tokens=kv_size,
                batch=batch_size,
                mlen=self.mlen,
                blen=self.blen,
                hlen=self.hlen,
                query_heads=num_attention_heads,
                kv_heads=num_kv_heads,
                head_dim=head_dim,
                batch_packed=batch_packed,
            )
            matrix_cycles = sum(
                count * self.instr[opcode]
                for opcode, count in matrix_histogram
            )
            vector_cycles = (
                (
                    4 * self.instr["V_BASIC"]
                    + 2 * self.instr["S_BASIC"]
                    + self.instr["S_EXP_FP"]
                    + 2 * self.instr["V_BASIC"]
                    + 1
                )
                * num_attention_heads
                * tc
                * batch_size
            )
            prefetch_cycles = (
                (num_kv_heads + num_attention_heads)
                * tc
                * batch_size
                * self.instr["H_PREFETCH_M"]
            )
            return matrix_cycles + vector_cycles + prefetch_cycles
        elif batch_packed:
            # The compiler picks the instruction shape from the query-row count,
            # not from the workload: `asm_templates/flashattn/overall.py` emits
            # the matrix-shaped mix (M_BTMM + M_MM) whenever q_len > 1, and the
            # decode-shaped mix (M_BTMV + M_MV) only for a single query row.
            # Packing the batch's single tokens into query rows therefore runs
            # the same lowering as a q_len-row prefill tile, so cost it that way.
            packed_rows = min(batch_size, self.mlen)
            return self.flash_attention(
                num_attention_heads,
                num_kv_heads,
                head_dim,
                packed_rows,
                kv_size,
                math.ceil(batch_size / packed_rows),
                mode="prefill",
            )
        else:  # decode sensitivity for layouts without PackedKV selector timing
            # QKT (per KV head and grouped Q heads). The single-row emitter
            # drains every broadcast result with M_BMV_WO; omitting that
            # MLEN-wide stream reverses the single-row/batch-packed ordering.
            qkt_groups = math.ceil(
                inner_q_head_loop / (self.mlen // self.hlen)
            )
            inner_compute_cycles += (
                self.instr["M_BTMV"]
                + self.instr["M_BMV_WO"]
                + self.instr["H_PREFETCH_M"]
            ) * qkt_groups
            # online softmax
            inner_compute_cycles += (
                4 * self.instr["V_BASIC"] + 2 * self.instr["S_BASIC"] + self.instr["S_EXP_FP"]
            ) * inner_q_head_loop
            # Compute PV. Each column block is likewise retired by M_MV_WO.
            inner_compute_cycles += (
                math.ceil(head_dim / self.blen)
                * (self.instr["M_MV"] + self.instr["M_MV_WO"])
                * inner_q_head_loop
            )
            # Compute O
            inner_compute_cycles += (2 * self.instr["V_BASIC"] + 1) * inner_q_head_loop
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
                total_tokens * expert_per_token * math.ceil(intermediate_size / self.vlen) * (SILU_BASIC_OPS * self.instr["V_BASIC"] + self.instr["V_RECI_V"])
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
                total_tokens * expert_per_token * math.ceil(intermediate_size / self.vlen) * (SILU_BASIC_OPS * self.instr["V_BASIC"] + self.instr["V_RECI_V"])
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
        return overall_cycles

    def _residual_body(self) -> int:
        """Cycles for one residual loop body: the add plus its address and loop issues."""
        return (
            self.instr["V_ADD_VV"]
            + RESIDUAL_ADDRESS_CURSORS * self.instr["S_ADDI_INT"]
            + self.instr["C_LOOP_END"]
        )

    def residual(self, hidden_size: int, seq_len: int, batch_size: int, mode: str = "prefill") -> int:
        """Residual connection cycle count."""
        iteration = hidden_size // self.vlen
        overall_cycles = 0

        if mode == "prefill":
            compute_cycle = self._residual_body() * seq_len * iteration * batch_size
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
            compute_cycle = self._residual_body() * iteration * batch_size
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
                math.ceil(intermediate_size / self.vlen) * (SILU_BASIC_OPS * self.instr["V_BASIC"] + self.instr["V_RECI_V"]) * seq_len * batch_size
            )
            # Downsize Linear
            overall_cycles += (
                math.ceil((seq_len * batch_size) / self.blen)
                * math.ceil(intermediate_size / self.mlen)
                * math.ceil(hidden_size / self.blen)
                * self.instr["M_MM"]
            )
        else:
            auxiliary = ffn_decode_auxiliary_histogram(
                mlen=self.mlen,
                blen=self.blen,
                vlen=self.vlen,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                rows=batch_size,
            )
            # Upsize Linear and Gate
            up_gate_reduction = math.ceil(hidden_size / self.mlen)
            overall_cycles += self._matmul_compute_and_drain(
                2
                * math.ceil(intermediate_size / self.blen)
                * up_gate_reduction
                * math.ceil(batch_size / self.blen),
                up_gate_reduction,
            )
            # SiLU
            overall_cycles += math.ceil(intermediate_size / self.vlen) * (SILU_BASIC_OPS * self.instr["V_BASIC"] + self.instr["V_RECI_V"]) * batch_size
            # Downsize Linear
            down_reduction = math.ceil(intermediate_size / self.mlen)
            overall_cycles += self._matmul_compute_and_drain(
                math.ceil(batch_size / self.blen)
                * down_reduction
                * math.ceil(hidden_size / self.blen),
                down_reduction,
            )
            overall_cycles += sum(
                count * self.instr[opcode]
                for opcode, count in auxiliary.items()
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

    def _lm_head_rows(self, hidden_size: int, vocab_size: int, rows: int) -> int:
        """LM head cost for `rows` hidden-state rows.

        `asm_templates/lm_head.py` streams the checkpoint's native
        `(vocab_size, hidden_size)` weight through the transposed projection: one
        matrix issue per (BLEN row block, BLEN vocabulary block, MLEN reduction
        tile), a writeout draining each reduction group, and one weight prefetch
        per MLEN-wide vocabulary group per reduction tile.
        """
        reduction_tiles = math.ceil(hidden_size / self.mlen)
        mm_events = (
            math.ceil(rows / self.blen)
            * math.ceil(vocab_size / self.blen)
            * reduction_tiles
        )
        weight_tiles = math.ceil(vocab_size / self.mlen) * reduction_tiles
        return (
            LM_HEAD_SETTING_INSTRUCTIONS * self.instr["S_BASIC"]
            + self._matmul_and_drain(mm_events, reduction_tiles)
            + weight_tiles * self.instr["H_PREFETCH_M"]
        )

    def lm_head(self, hidden_size: int, vocab_size: int, batch_size: int) -> int:
        """LM head cycle count (linear projection to vocab)."""
        return self._lm_head_rows(hidden_size, vocab_size, batch_size)

    def lm_head_full_seq(self, hidden_size: int, vocab_size: int, seq_len: int, batch_size: int) -> int:
        """LM head cycle count over full sequence (used by LLaDA: all positions need logits)."""
        return self._lm_head_rows(hidden_size, vocab_size, batch_size * seq_len)

    def softmax_full_seq(self, vocab_size: int, seq_len: int, batch_size: int) -> int:
        """Softmax over vocab for all sequence positions (used by LLaDA for confidence scoring)."""
        # One softmax row per token position: batch_size * seq_len rows of length vocab_size
        loop_num = math.ceil(vocab_size / self.vlen)
        # softmax: max-reduce + exp + sum + divide = ~6 V_ ops per chunk
        overall_cycles = batch_size * seq_len * loop_num * (SILU_BASIC_OPS * self.instr["V_BASIC"] + self.instr["V_RECI_V"])
        return overall_cycles
