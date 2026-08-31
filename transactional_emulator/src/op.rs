#[derive(Debug, Clone, Copy)]
pub enum MatrixPrecision {
    Weights,
    KeyValue,
}

#[derive(Debug, Clone, Copy)]
pub enum VectorPrecision {
    Activation,
    KeyValue,
}

#[derive(Debug, Clone, Copy)]
pub enum VectorOrder {
    Normal,
    Reverse,
}

#[allow(non_camel_case_types)]
#[derive(Debug)]
pub enum Opcode {
    Invalid,
    M_MM {
        rs1: u8,
        rs2: u8,
    },
    M_TMM {
        rs1: u8,
        rs2: u8,
    },
    M_BMM {
        rs1: u8,
        rs2: u8,
    },
    M_BTMM {
        rs1: u8,
        rs2: u8,
    },
    M_BMM_WO {
        rd: u8,
        imm: u32,
    },
    M_MM_WO {
        rd: u8,
        rstride: u8,
        imm: u32,
    },
    M_MV {
        rs1: u8,
        rs2: u8,
    },
    M_TMV {
        rs1: u8,
        rs2: u8,
    },
    M_BMV {
        rs1: u8,
        rs2: u8,
        rd: u8,
    },
    M_BTMV {
        rs1: u8,
        rs2: u8,
        rd: u8,
    },
    M_MV_WO {
        rd: u8,
        imm: u32,
    },
    M_BMV_WO {
        rd: u8,
        imm: u32,
    },
    V_ADD_VV {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rmask: u8,
        lmask: u8,
    },
    V_ADD_VF {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rmask: u8,
        lmask: u8,
    },
    V_SUB_VV {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rmask: u8,
        lmask: u8,
    },
    V_SUB_VF {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rmask: u8,
        rorder: VectorOrder,
    },
    V_MUL_VV {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rmask: u8,
        lmask: u8,
    },
    V_MUL_VF {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rmask: u8,
        lmask: u8,
    },
    /// `Vector[rd] += Vector[rs1] * fp_reg<rs2>` -- fused broadcast multiply-add.
    ///
    /// Unlike every other V-type op, `rd` is **read** as well as written. That
    /// is the whole point: it collapses the `copy + multiply + add` triple that
    /// a rank-1 state update or a state contraction otherwise costs, and by
    /// removing the scratch row it lets a whole key sweep become one arithmetic
    /// row progression -- which the compiler turns into a hardware loop.
    V_FMA_VF {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rmask: u8,
        lmask: u8,
    },
    V_MAX_VF {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rmask: u8,
        lmask: u8,
    },
    V_MIN_VF {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rmask: u8,
        lmask: u8,
    },
    /// Routed-MoE router helper.
    ///
    /// Encoding follows the regular vector register form:
    /// - `rs1`: VRAM row containing router logits.
    /// - `rd`: GP register whose value is the FP SRAM base for selected route
    ///   weights.  Policy 0 stores GPT-OSS 32-way/top-4 weights.  Policy 1
    ///   stores Qwen 128-way/top-8 weights.  Qwen computes full-expert softmax
    ///   before top-k in HF, but `norm_topk_prob=true` renormalizes the selected
    ///   entries, making the final route weights equivalent to selected-logit
    ///   softmax.
    /// - `rs2`: GP register whose value is the INT SRAM base for the selected
    ///   expert indices.
    /// - `rmask`: policy selector (`0` = 32 experts/top-4, `1` = 128
    ///   experts/top-8, `15` = take `(num_experts, top_k)` from the
    ///   `C_SET_TOPK_REG` control register).
    V_TOPK {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rmask: u8,
    },
    V_EXP_V {
        rd: u8,
        rs1: u8,
        rmask: u8,
        lmask: u8,
    },
    V_RECI_V {
        rd: u8,
        rs1: u8,
        rmask: u8,
        lmask: u8,
    },
    V_RED_SUM {
        rd: u8,
        rs1: u8,
        rmask: u8,
        lmask: u8,
    },
    V_RED_MAX {
        rd: u8,
        rs1: u8,
        rmask: u8,
        lmask: u8,
    },
    /// `V_SOFTPLUS_V rd, rs1, rmask` — elementwise `log(1 + exp(x))` over one VLEN tile.
    ///
    /// Mamba/Mamba-2 needs `dt = softplus(dt_raw + dt_bias)` on the critical path of a
    /// multiplicative recurrence, and the ISA has no logarithm, so there is no exact
    /// software lowering. Evaluated as the range-safe identity
    /// `softplus(x) = relu(x) + log1p(exp(-|x|))`, which is what the RTL is expected to
    /// implement: it never evaluates `exp` on a positive argument, so it cannot overflow
    /// for any finite input. `rmask` follows the usual masked-vector convention (0 = whole
    /// tile, otherwise the per-head bitmask in `V_MASK`).
    V_SOFTPLUS_V {
        rd: u8,
        rs1: u8,
        rmask: u8,
        lmask: u8,
    },
    S_ADD_FP {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    S_SUB_FP {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    S_MAX_FP {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    S_MUL_FP {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    S_EXP_FP {
        rd: u8,
        rs1: u8,
    },
    S_RECI_FP {
        rd: u8,
        rs1: u8,
    },
    S_SQRT_FP {
        rd: u8,
        rs1: u8,
    },
    S_LD_FP {
        rd: u8,
        rs1: u8,
        imm: u32,
    },
    S_ST_FP {
        rd: u8,
        rs1: u8,
        imm: u32,
    },
    S_MAP_V_FP {
        rd: u8,
        rs1: u8,
        imm: u32,
    },
    /// `S_MAP_FP_V rd, rs1, imm` — the exact inverse of `S_MAP_V_FP`.
    ///
    /// Copies one VLEN-wide Vector SRAM row at `gp[rs1]` into VLEN consecutive FP_MEM
    /// slots starting at `gp[rd] + imm`. Before this existed the only VRAM-to-scalar path
    /// was `V_RED_SUM`/`V_RED_MAX`, which collapse the whole row, so extracting a single
    /// lane cost a one-hot `V_MUL_VV` + `V_RED_SUM` + `S_ST_FP` triple. Mamba-2's chunked
    /// scan needs one broadcast scalar per row (`cs_i`, `exp(cs_C - cs_t)`, `exp(cs_i)`),
    /// which made that the dominant instruction cost of the whole kernel.
    S_MAP_FP_V {
        rd: u8,
        rs1: u8,
        imm: u32,
    },

    S_ADD_INT {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    S_ADDI_INT {
        rd: u8,
        rs1: u8,
        imm: u32,
    },
    S_SUB_INT {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    S_MUL_INT {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    S_LUI_INT {
        rd: u8,
        imm: u32,
    },
    S_LD_INT {
        rd: u8,
        rs1: u8,
        imm: u32,
    },
    S_ST_INT {
        rd: u8,
        rs1: u8,
        imm: u32,
    },

    H_PREFETCH_M {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rstride: u8,
        precision: MatrixPrecision,
    },
    H_PREFETCH_V {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rstride: u8,
        precision: VectorPrecision,
    },
    H_STORE_V {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rstride: u8,
        precision: VectorPrecision,
    },

    C_SET_ADDR_REG {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    C_SET_SCALE_REG {
        rd: u8,
    },
    C_SET_STRIDE_REG {
        rd: u8,
    },
    C_SET_V_MASK_REG {
        rd: u8,
    },
    /// Set the sticky routed-MoE top-k policy read by `V_TOPK rmask=15`.
    ///
    /// `rd` names a GP register holding `(num_experts << 8) | top_k`. Sticky like
    /// the other `C_SET_*_REG` registers, so a single-policy program sets it once.
    C_SET_TOPK_REG {
        rd: u8,
    },
    /// Configure one field of a compiler-managed affine operand stream.
    L_CFG {
        value: u8,
        target: u8,
        slot: u8,
        field: u8,
    },
    C_LOOP_START {
        rd: u8,
        imm: u32,
    },
    C_LOOP_END {
        rd: u8,
    },
    // Extensions
    V_SHFT_V {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    C_BREAK,
}

const OPERAND_WIDTH: u32 = 4;
const OPCODE_WIDTH: u32 = 6;
const IMM_WIDTH: u32 = 22;
const IMM_2_WIDTH: u32 = 18;
const LSTREAM_CONSUMER_MASK: u8 = 0x7;
const VECTOR_ACCUMULATE_MODE: u8 = 0x8;

const fn mask(width: u32) -> u32 {
    ((1 << width) - 1) as u32
}

impl Opcode {
    #[inline]
    fn matrix_precision_from(funct1: u8) -> MatrixPrecision {
        if funct1 == 0 {
            MatrixPrecision::Weights
        } else {
            MatrixPrecision::KeyValue
        }
    }

    #[inline]
    fn vector_precision_from(funct1: u8) -> VectorPrecision {
        if funct1 == 0 {
            VectorPrecision::Activation
        } else {
            VectorPrecision::KeyValue
        }
    }

    #[inline]
    fn vector_order_from(funct1: u8) -> VectorOrder {
        if funct1 == 0 {
            VectorOrder::Normal
        } else {
            VectorOrder::Reverse
        }
    }

    pub fn decode(instr: u32) -> Self {
        let opcode = instr & mask(OPCODE_WIDTH);
        let rd = ((instr >> OPCODE_WIDTH) & mask(OPERAND_WIDTH)) as u8;
        let rs1 = ((instr >> (OPCODE_WIDTH + OPERAND_WIDTH)) & mask(OPERAND_WIDTH)) as u8;
        let rs2 = ((instr >> (OPCODE_WIDTH + OPERAND_WIDTH * 2)) & mask(OPERAND_WIDTH)) as u8;
        let rs3 = ((instr >> (OPCODE_WIDTH + OPERAND_WIDTH * 3)) & mask(OPERAND_WIDTH)) as u8;
        let funct1 = ((instr >> (OPCODE_WIDTH + OPERAND_WIDTH * 4)) & mask(OPERAND_WIDTH)) as u8;
        let imm = (instr >> (OPCODE_WIDTH + OPERAND_WIDTH)) & mask(IMM_WIDTH);
        let imm2 = (instr >> (OPCODE_WIDTH + OPERAND_WIDTH * 2)) & mask(IMM_2_WIDTH);

        match opcode {
            0x00 => Self::Invalid,
            // Matrix Operations
            0x01 => Self::M_MM { rs1, rs2 },
            0x02 => Self::M_TMM { rs1, rs2 },
            0x03 => {
                // ISA spec defines matrix address as `gp_reg<rs1> + gp_reg<rd>` but
                // this emulator only consumes `rs1`. M_BMV/M_BTMV honor `rd`; until
                // M_BMM/M_BTMM follow suit, refuse encodings that would otherwise
                // silently drop the rd offset.
                assert_eq!(
                    rd, 0,
                    "M_BMM rd must be 0: emulator does not honor the spec's `gp_reg<rd>` matrix offset"
                );
                Self::M_BMM { rs1, rs2 }
            }
            0x04 => {
                assert_eq!(
                    rd, 0,
                    "M_BTMM rd must be 0: emulator does not honor the spec's `gp_reg<rd>` matrix offset"
                );
                Self::M_BTMM { rs1, rs2 }
            }
            0x05 => Self::M_BMM_WO { rd, imm: imm2 },
            0x06 => Self::M_MM_WO {
                rd,
                rstride: rs1,
                imm: imm2,
            },
            0x07 => Self::M_MV { rs1, rs2 },
            0x08 => Self::M_TMV { rs1, rs2 },
            0x09 => Self::M_BMV { rs1, rs2, rd },
            0x0A => Self::M_BTMV { rs1, rs2, rd },
            0x0B => Self::M_MV_WO { rd, imm: imm2 },
            0x0C => Self::M_BMV_WO { rd, imm: imm2 },

            // Vector Operations
            0x0D if funct1 <= LSTREAM_CONSUMER_MASK => Self::V_ADD_VV {
                rd,
                rs1,
                rs2,
                rmask: rs3,
                lmask: funct1 & LSTREAM_CONSUMER_MASK,
            },
            0x0E if funct1 <= LSTREAM_CONSUMER_MASK => Self::V_ADD_VF {
                rd,
                rs1,
                rs2,
                rmask: rs3,
                lmask: funct1 & LSTREAM_CONSUMER_MASK,
            },
            0x0F if funct1 <= LSTREAM_CONSUMER_MASK => Self::V_SUB_VV {
                rd,
                rs1,
                rs2,
                rmask: rs3,
                lmask: funct1 & LSTREAM_CONSUMER_MASK,
            },
            0x10 => Self::V_SUB_VF {
                rd,
                rs1,
                rs2,
                rmask: rs3,
                rorder: Self::vector_order_from(funct1),
            },
            0x11 if funct1 <= LSTREAM_CONSUMER_MASK => Self::V_MUL_VV {
                rd,
                rs1,
                rs2,
                rmask: rs3,
                lmask: funct1 & LSTREAM_CONSUMER_MASK,
            },
            0x12 if funct1 & VECTOR_ACCUMULATE_MODE == 0 => Self::V_MUL_VF {
                rd,
                rs1,
                rs2,
                rmask: rs3,
                lmask: funct1 & LSTREAM_CONSUMER_MASK,
            },
            0x12 => Self::V_FMA_VF {
                rd,
                rs1,
                rs2,
                rmask: rs3,
                lmask: funct1 & LSTREAM_CONSUMER_MASK,
            },
            0x13 if funct1 <= LSTREAM_CONSUMER_MASK => Self::V_EXP_V {
                rd,
                rs1,
                rmask: rs3,
                lmask: funct1 & LSTREAM_CONSUMER_MASK,
            },
            0x14 if funct1 <= LSTREAM_CONSUMER_MASK => Self::V_RECI_V {
                rd,
                rs1,
                rmask: rs3,
                lmask: funct1 & LSTREAM_CONSUMER_MASK,
            },
            0x15 if funct1 <= LSTREAM_CONSUMER_MASK => Self::V_RED_SUM {
                rd,
                rs1,
                rmask: rs3,
                lmask: funct1 & LSTREAM_CONSUMER_MASK,
            },
            0x16 if funct1 <= LSTREAM_CONSUMER_MASK => Self::V_RED_MAX {
                rd,
                rs1,
                rmask: rs3,
                lmask: funct1 & LSTREAM_CONSUMER_MASK,
            },
            0x35 if funct1 <= LSTREAM_CONSUMER_MASK => Self::V_MAX_VF {
                rd,
                rs1,
                rs2,
                rmask: rs3,
                lmask: funct1 & LSTREAM_CONSUMER_MASK,
            },
            0x36 if funct1 <= LSTREAM_CONSUMER_MASK => Self::V_MIN_VF {
                rd,
                rs1,
                rs2,
                rmask: rs3,
                lmask: funct1 & LSTREAM_CONSUMER_MASK,
            },
            0x37 if funct1 == 0 => Self::V_TOPK {
                rd,
                rs1,
                rs2,
                rmask: rs3,
            },
            0x37 => {
                tracing::error!(instr, "non-canonical V_TOPK encoding");
                Self::Invalid
            }
            0x0D | 0x0E | 0x0F | 0x11 | 0x13 | 0x14 | 0x15 | 0x16 | 0x35 | 0x36
                if funct1 & VECTOR_ACCUMULATE_MODE != 0 =>
            {
                tracing::error!(instr, opcode, "reserved vector arithmetic variant");
                Self::Invalid
            }

            // Scalar Operations (Floating-Point)
            0x17 => Self::S_ADD_FP { rd, rs1, rs2 },
            0x18 => Self::S_SUB_FP { rd, rs1, rs2 },
            0x19 => Self::S_MAX_FP { rd, rs1, rs2 },
            0x1A => Self::S_MUL_FP { rd, rs1, rs2 },
            0x1B => Self::S_EXP_FP { rd, rs1 },
            0x1C => Self::S_RECI_FP { rd, rs1 },
            0x1D => Self::S_SQRT_FP { rd, rs1 },
            0x1E => Self::S_LD_FP { rd, rs1, imm: imm2 },
            0x1F => Self::S_ST_FP { rd, rs1, imm: imm2 },
            0x20 => Self::S_MAP_V_FP { rd, rs1, imm: imm2 },

            // Scalar Operations (INT)
            0x21 => Self::S_ADD_INT { rd, rs1, rs2 },
            0x22 => Self::S_ADDI_INT { rd, rs1, imm: imm2 },
            0x23 => Self::S_SUB_INT { rd, rs1, rs2 },
            0x24 => Self::S_MUL_INT { rd, rs1, rs2 },
            0x25 => Self::S_LUI_INT { rd, imm },
            0x26 => Self::S_LD_INT { rd, rs1, imm: imm2 },
            0x27 => Self::S_ST_INT { rd, rs1, imm: imm2 },

            0x28 => Self::H_PREFETCH_M {
                rd,
                rs1,
                rs2,
                rstride: rs3,
                precision: Self::matrix_precision_from(funct1),
            },
            // 0x29 => Self::H_PREFETCH_M { rd, rs1, rs2, rstride: rs3, precision: MatrixPrecision::KeyValue },
            0x29 => Self::H_PREFETCH_V {
                rd,
                rs1,
                rs2,
                rstride: rs3,
                precision: Self::vector_precision_from(funct1),
            },
            // 0x2A => Self::H_PREFETCH_V { rd, rs1, rs2, rstride: rs3, precision: VectorPrecision::KeyValue },
            0x2A => Self::H_STORE_V {
                rd,
                rs1,
                rs2,
                rstride: rs3,
                precision: Self::vector_precision_from(funct1),
            },
            // 0x2B => Self::H_STORE_V { rd, rs1, rs2, rstride: rs3, precision: VectorPrecision::KeyValue },
            0x2B => Self::C_SET_ADDR_REG { rd, rs1, rs2 },
            0x2C => Self::C_SET_SCALE_REG { rd },
            0x2D => Self::C_SET_STRIDE_REG { rd },
            0x2E => Self::C_SET_V_MASK_REG { rd },
            0x2F => Self::C_LOOP_START { rd, imm },
            0x30 => Self::C_LOOP_END { rd },
            // 0x31 and 0x33 are intentionally unassigned gaps: V_SHFT_V and
            // C_BREAK were renumbered (0x31->0x32, 0x32->0x34) to keep the
            // routed-MoE vector ops (V_MAX_VF/V_MIN_VF/V_TOPK at 0x35..=0x37)
            // contiguous with the other masked vector ops. Encodings must stay
            // in sync with PLENA_Compiler's assembler (isa_definitions).
            0x32 => Self::V_SHFT_V { rd, rs1, rs2 },
            0x34 => Self::C_BREAK,
            // 0x35..=0x37 (V_MAX_VF/V_MIN_VF/V_TOPK) are decoded with the other
            // masked vector ops above.
            0x38 => Self::C_SET_TOPK_REG { rd },
            // 0x39..=0x3C are reserved for the Shared Expert route dispatcher.
            // This branch does not emit them; keeping them invalid here is safer
            // than silently executing a different recurrent operation.
            0x39..=0x3C => {
                tracing::error!(
                    instr,
                    opcode,
                    "reserved routed-MoE opcode is not implemented here"
                );
                Self::Invalid
            }
            // General static-recurrent extensions. Encodings must stay in sync
            // with PLENA_Compiler's doc/operation.svh and assembler.
            0x3D if funct1 <= LSTREAM_CONSUMER_MASK => Self::V_SOFTPLUS_V {
                rd,
                rs1,
                rmask: rs3,
                lmask: funct1 & LSTREAM_CONSUMER_MASK,
            },
            0x3D => {
                tracing::error!(instr, opcode, "reserved vector arithmetic variant");
                Self::Invalid
            }
            0x3E => Self::S_MAP_FP_V { rd, rs1, imm: imm2 },
            0x3F => {
                if instr >> 22 != 0 || rs2 >= 4 {
                    tracing::error!(instr, "non-canonical L_CFG encoding");
                    Self::Invalid
                } else {
                    Self::L_CFG {
                        value: rd,
                        target: rs1,
                        slot: rs2,
                        field: rs3,
                    }
                }
            }
            _ => {
                tracing::error!("Unknown opcode {opcode:#x}");
                Self::Invalid
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a register-form instruction word matching `decode`'s field layout:
    /// opcode[0..6], rd[6..10], rs1[10..14], rs2[14..18], rs3[18..22], funct1[22..26].
    fn rform(opcode: u32, rd: u32, rs1: u32, rs2: u32, rs3: u32, funct1: u32) -> u32 {
        opcode | (rd << 6) | (rs1 << 10) | (rs2 << 14) | (rs3 << 18) | (funct1 << 22)
    }

    #[test]
    fn test_decode_register_fields() {
        // V_ADD_VV packs rd, rs1, rs2, and rmask (= rs3).
        match Opcode::decode(rform(0x0D, 1, 2, 3, 4, 0)) {
            Opcode::V_ADD_VV {
                rd,
                rs1,
                rs2,
                rmask,
                lmask,
            } => assert_eq!((rd, rs1, rs2, rmask, lmask), (1, 2, 3, 4, 0)),
            other => panic!("expected V_ADD_VV, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_two_register_matrix_op() {
        // M_MM consumes only rs1 and rs2.
        match Opcode::decode(rform(0x01, 0, 5, 6, 0, 0)) {
            Opcode::M_MM { rs1, rs2 } => assert_eq!((rs1, rs2), (5, 6)),
            other => panic!("expected M_MM, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_invalid_and_unknown_are_invalid() {
        assert!(matches!(Opcode::decode(0x00), Opcode::Invalid));
        // V_PS_V remains declared but deliberately unimplemented.
        assert!(matches!(Opcode::decode(0x31), Opcode::Invalid));
    }

    #[test]
    fn test_decode_c_set_topk_reg() {
        match Opcode::decode(0x38 | (7 << 6)) {
            Opcode::C_SET_TOPK_REG { rd } => assert_eq!(rd, 7),
            other => panic!("expected C_SET_TOPK_REG, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_imm22_field() {
        // S_LUI_INT carries the wide 22-bit immediate (bits 10..32).
        let instr = 0x25 | (5 << 6) | (0x2ABCD << 10);
        match Opcode::decode(instr) {
            Opcode::S_LUI_INT { rd, imm } => assert_eq!((rd, imm), (5, 0x2ABCD)),
            other => panic!("expected S_LUI_INT, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_imm18_field() {
        // S_ADDI_INT carries rs1 plus the 18-bit immediate (bits 14..32).
        let instr = 0x22 | (1 << 6) | (2 << 10) | (0x1ABCD << 14);
        match Opcode::decode(instr) {
            Opcode::S_ADDI_INT { rd, rs1, imm } => assert_eq!((rd, rs1, imm), (1, 2, 0x1ABCD)),
            other => panic!("expected S_ADDI_INT, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_matrix_precision_from_funct1() {
        assert!(matches!(
            Opcode::decode(rform(0x28, 0, 0, 0, 0, 0)),
            Opcode::H_PREFETCH_M {
                precision: MatrixPrecision::Weights,
                ..
            }
        ));
        assert!(matches!(
            Opcode::decode(rform(0x28, 0, 0, 0, 0, 1)),
            Opcode::H_PREFETCH_M {
                precision: MatrixPrecision::KeyValue,
                ..
            }
        ));
    }

    #[test]
    fn test_decode_vector_order_from_funct1() {
        assert!(matches!(
            Opcode::decode(rform(0x10, 0, 0, 0, 0, 0)),
            Opcode::V_SUB_VF {
                rorder: VectorOrder::Normal,
                ..
            }
        ));
        assert!(matches!(
            Opcode::decode(rform(0x10, 0, 0, 0, 0, 1)),
            Opcode::V_SUB_VF {
                rorder: VectorOrder::Reverse,
                ..
            }
        ));
    }

    #[test]
    fn test_decode_m_bmm_rd_zero_ok() {
        match Opcode::decode(rform(0x03, 0, 7, 8, 0, 0)) {
            Opcode::M_BMM { rs1, rs2 } => assert_eq!((rs1, rs2), (7, 8)),
            other => panic!("expected M_BMM, got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "M_BMM rd must be 0")]
    fn test_decode_m_bmm_rd_nonzero_panics() {
        // The emulator does not honor the spec's gp_reg<rd> matrix offset, so a
        // non-zero rd is refused at decode time.
        let _ = Opcode::decode(rform(0x03, 1, 7, 8, 0, 0));
    }

    /// Build an imm2-form word: opcode[0..6], rd[6..10], rs1[10..14],
    /// imm2[14..32] (the 18-bit immediate used by LD/ST/WO/MAP ops).
    fn i2form(opcode: u32, rd: u32, rs1: u32, imm2: u32) -> u32 {
        opcode | (rd << 6) | (rs1 << 10) | (imm2 << 14)
    }

    /// Build an imm22-form word: opcode[0..6], rd[6..10], imm[10..32].
    fn i22form(opcode: u32, rd: u32, imm: u32) -> u32 {
        opcode | (rd << 6) | (imm << 10)
    }

    // ---------- scalar ops (rd, rs1[, rs2]) ----------

    #[test]
    fn test_decode_scalar_fp_three_register() {
        match Opcode::decode(rform(0x17, 1, 2, 3, 0, 0)) {
            Opcode::S_ADD_FP { rd, rs1, rs2 } => assert_eq!((rd, rs1, rs2), (1, 2, 3)),
            other => panic!("expected S_ADD_FP, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_scalar_fp_two_register() {
        // S_EXP_FP consumes only rd and rs1 (rs2 ignored).
        match Opcode::decode(rform(0x1B, 4, 5, 9, 0, 0)) {
            Opcode::S_EXP_FP { rd, rs1 } => assert_eq!((rd, rs1), (4, 5)),
            other => panic!("expected S_EXP_FP, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_scalar_int_three_register() {
        match Opcode::decode(rform(0x21, 6, 7, 8, 0, 0)) {
            Opcode::S_ADD_INT { rd, rs1, rs2 } => assert_eq!((rd, rs1, rs2), (6, 7, 8)),
            other => panic!("expected S_ADD_INT, got {other:?}"),
        }
    }

    // ---------- imm2 (18-bit) field on LD/ST/MAP ----------

    #[test]
    fn test_decode_scalar_ld_fp_imm2() {
        match Opcode::decode(i2form(0x1E, 2, 3, 0x1ABCD)) {
            Opcode::S_LD_FP { rd, rs1, imm } => assert_eq!((rd, rs1, imm), (2, 3, 0x1ABCD)),
            other => panic!("expected S_LD_FP, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_scalar_st_int_imm2() {
        match Opcode::decode(i2form(0x27, 1, 2, 0x3FFFF)) {
            Opcode::S_ST_INT { rd, rs1, imm } => assert_eq!((rd, rs1, imm), (1, 2, 0x3FFFF)),
            other => panic!("expected S_ST_INT, got {other:?}"),
        }
    }

    // ---------- matrix write-out (imm2) and strided variants ----------

    #[test]
    fn test_decode_m_mm_wo_carries_rstride_and_imm2() {
        // M_MM_WO packs rd, rstride (= rs1 field), and the 18-bit imm2.
        match Opcode::decode(i2form(0x06, 5, 6, 0x2BEEF)) {
            Opcode::M_MM_WO { rd, rstride, imm } => {
                assert_eq!((rd, rstride, imm), (5, 6, 0x2BEEF))
            }
            other => panic!("expected M_MM_WO, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_m_bmv_carries_rd() {
        // M_BMV honors rd (unlike M_BMM); decode keeps all three.
        match Opcode::decode(rform(0x09, 9, 7, 8, 0, 0)) {
            Opcode::M_BMV { rs1, rs2, rd } => assert_eq!((rs1, rs2, rd), (7, 8, 9)),
            other => panic!("expected M_BMV, got {other:?}"),
        }
    }

    // ---------- HBM prefetch/store precision-from-funct1 ----------

    #[test]
    fn test_decode_prefetch_v_precision_from_funct1() {
        assert!(matches!(
            Opcode::decode(rform(0x29, 0, 0, 0, 0, 0)),
            Opcode::H_PREFETCH_V {
                precision: VectorPrecision::Activation,
                ..
            }
        ));
        assert!(matches!(
            Opcode::decode(rform(0x29, 0, 0, 0, 0, 1)),
            Opcode::H_PREFETCH_V {
                precision: VectorPrecision::KeyValue,
                ..
            }
        ));
    }

    #[test]
    fn test_decode_store_v_precision_and_fields() {
        match Opcode::decode(rform(0x2A, 1, 2, 3, 4, 1)) {
            Opcode::H_STORE_V {
                rd,
                rs1,
                rs2,
                rstride,
                precision: VectorPrecision::KeyValue,
            } => assert_eq!((rd, rs1, rs2, rstride), (1, 2, 3, 4)),
            other => panic!("expected H_STORE_V KeyValue, got {other:?}"),
        }
    }

    // ---------- control ops ----------

    #[test]
    fn test_decode_control_set_addr_reg() {
        match Opcode::decode(rform(0x2B, 1, 2, 3, 0, 0)) {
            Opcode::C_SET_ADDR_REG { rd, rs1, rs2 } => assert_eq!((rd, rs1, rs2), (1, 2, 3)),
            other => panic!("expected C_SET_ADDR_REG, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_loop_start_imm22() {
        match Opcode::decode(i22form(0x2F, 3, 0x2ABCD)) {
            Opcode::C_LOOP_START { rd, imm } => assert_eq!((rd, imm), (3, 0x2ABCD)),
            other => panic!("expected C_LOOP_START, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_break_is_unit() {
        assert!(matches!(Opcode::decode(0x34), Opcode::C_BREAK));
    }

    // ---------- Mamba / selective-SSM extensions ----------

    #[test]
    fn test_decode_v_softplus_v_reads_rmask_from_rs3() {
        // rs3 (not rs2) carries rmask, matching every other masked vector op and the
        // compiler's `_RMASK_VECTOR_OPS` encoding.
        match Opcode::decode(rform(0x3D, 1, 2, 0, 5, 0)) {
            Opcode::V_SOFTPLUS_V {
                rd,
                rs1,
                rmask,
                lmask,
            } => assert_eq!((rd, rs1, rmask, lmask), (1, 2, 5, 0)),
            other => panic!("expected V_SOFTPLUS_V, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_s_map_fp_v_imm2() {
        // Same operand shape as its S_MAP_V_FP mirror: rd, rs1 and the 18-bit imm2.
        match Opcode::decode(i2form(0x3E, 4, 5, 0x1F00F)) {
            Opcode::S_MAP_FP_V { rd, rs1, imm } => assert_eq!((rd, rs1, imm), (4, 5, 0x1F00F)),
            other => panic!("expected S_MAP_FP_V, got {other:?}"),
        }
    }

    /// Every opcode PLENA_Compiler declares must decode to something here.
    ///
    /// `decode` carries three separate comments saying encodings "must stay in
    /// sync with PLENA_Compiler's doc/operation.svh". That was enforced by
    /// author diligence alone -- exactly the arrangement that let the compiler's
    /// FPRAM depth (1024) drift from the SystemVerilog's (512) unnoticed. The
    /// submodule is checked out recursively in every CI job, so the header can
    /// simply be read.
    #[test]
    fn every_compiler_opcode_decodes_to_something() {
        let svh = include_str!("../../PLENA_Compiler/doc/operation.svh");

        // Declared in the header but deliberately not modelled. A new entry is
        // a decision to argue for in review, not a default.
        const NOT_MODELLED: &[&str] = &[
            // The sentinel. Invalid is precisely what it must decode to.
            "INVALID_OPCODE",
            // Declared by PLENA but never implemented, in RTL or here. The
            // compiler knows and refuses to emit it -- see program_ssd.py's
            // ssd_chunk_cumsum docstring: it "assembles silently and then
            // decodes to Invalid, so emitting it is worse than not emitting
            // it". Its PREFIX_SCAN_V_ELEMENT micro-op scans within a row, which
            // under any head-on-lanes layout would cumsum across heads.
            "V_PS_V",
            // Likewise declared and unimplemented; nothing emits it.
            "C_HADAMARD_TRANSFORM",
            // Owned by the Shared Expert branch. Their numeric reservation is
            // part of this branch's conflict-free ABI, but route execution is
            // merged independently from L-Compute.
            "C_ROUTE_BEGIN",
            "C_ROUTE_LOOP_START",
            "C_ROUTE_LOOP_END",
            "V_ROUTE_MUL",
        ];

        let mut checked = 0;
        for line in svh.lines() {
            let line = line.trim();
            if line.starts_with("//") {
                continue;
            }
            let Some((name, rest)) = line.split_once('=') else {
                continue;
            };
            let name = name.trim();
            let Some(hex) = rest.trim().strip_prefix("6'h") else {
                continue;
            };
            let hex: String = hex.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
            let Ok(opcode) = u32::from_str_radix(&hex, 16) else {
                continue;
            };
            if NOT_MODELLED.contains(&name) {
                continue;
            }
            checked += 1;
            assert!(
                !matches!(
                    Opcode::decode(rform(opcode, 0, 0, 0, 0, 0)),
                    Opcode::Invalid
                ),
                "{name} = 6'h{opcode:02X} is declared in PLENA_Compiler's \
                 doc/operation.svh but decodes to Invalid here"
            );
        }
        assert!(
            checked > 40,
            "only {checked} opcodes parsed out of the header -- the parse broke, \
             so this guard was passing vacuously"
        );
    }

    #[test]
    fn v_fma_vf_decodes_like_v_mul_vf() {
        // Same physical opcode and R-type slots as V_MUL_VF. funct1[3] selects
        // read-modify-write FMA; funct1[2:0] remains the affine view mask.
        match Opcode::decode(rform(0x12, 3, 4, 2, 0, 0x8)) {
            Opcode::V_FMA_VF {
                rd,
                rs1,
                rs2,
                rmask,
                lmask,
            } => {
                assert_eq!((rd, rs1, rs2, rmask, lmask), (3, 4, 2, 0, 0));
            }
            other => panic!("expected V_FMA_VF, got {other:?}"),
        }
        match Opcode::decode(rform(0x12, 1, 2, 3, 5, 0xD)) {
            Opcode::V_FMA_VF { rmask, lmask, .. } => {
                assert_eq!(rmask, 5, "rs3 is the mask");
                assert_eq!(lmask, 5, "low funct bits are the view mask");
            }
            other => panic!("expected V_FMA_VF, got {other:?}"),
        }
    }

    #[test]
    fn test_mamba_opcodes_do_not_collide_with_existing_encodings() {
        // 0x39..0x3C belong to Shared Expert. Static recurrent operations occupy
        // 0x3D..0x3F, while FMA reuses V_MUL_VF's operation-variant bit.
        assert!(matches!(
            Opcode::decode(rform(0x38, 1, 0, 0, 0, 0)),
            Opcode::C_SET_TOPK_REG { .. }
        ));
        assert!(matches!(
            Opcode::decode(rform(0x12, 1, 2, 0, 0, 8)),
            Opcode::V_FMA_VF { .. }
        ));
        assert!(matches!(
            Opcode::decode(rform(0x3D, 1, 2, 0, 0, 0)),
            Opcode::V_SOFTPLUS_V { .. }
        ));
        assert!(matches!(
            Opcode::decode(rform(0x3E, 1, 2, 0, 0, 0)),
            Opcode::S_MAP_FP_V { .. }
        ));
        assert!(matches!(
            Opcode::decode(rform(0x3F, 1, 2, 3, 4, 0)),
            Opcode::L_CFG {
                value: 1,
                target: 2,
                slot: 3,
                field: 4
            }
        ));
        assert!(matches!(
            Opcode::decode(rform(0x39, 0, 0, 0, 0, 0)),
            Opcode::Invalid
        ));
    }

    #[test]
    fn l_cfg_accepts_field_15_and_rejects_high_bits_and_slots() {
        assert!(matches!(
            Opcode::decode(rform(0x3F, 1, 2, 3, 4, 1)),
            Opcode::Invalid
        ));
        assert!(matches!(
            Opcode::decode(rform(0x3F, 1, 2, 4, 4, 0)),
            Opcode::Invalid
        ));
        assert!(matches!(
            Opcode::decode(0x003C_C87F),
            Opcode::L_CFG {
                value: 1,
                target: 2,
                slot: 3,
                field: 15
            }
        ));
    }

    #[test]
    fn test_decode_v_shft_v() {
        match Opcode::decode(rform(0x32, 1, 2, 3, 0, 0)) {
            Opcode::V_SHFT_V { rd, rs1, rs2 } => assert_eq!((rd, rs1, rs2), (1, 2, 3)),
            other => panic!("expected V_SHFT_V, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_m_btmv_carries_rd() {
        // M_BTMV (unlike M_BTMM) honors rd; decode keeps all three fields, and
        // unlike M_BMM/M_BTMM it does not assert rd == 0.
        match Opcode::decode(rform(0x0A, 9, 7, 8, 0, 0)) {
            Opcode::M_BTMV { rs1, rs2, rd } => assert_eq!((rs1, rs2, rd), (7, 8, 9)),
            other => panic!("expected M_BTMV, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_s_map_v_fp_imm2() {
        // S_MAP_V_FP is the only S_MAP op; it carries rd, rs1, and the 18-bit imm2.
        match Opcode::decode(i2form(0x20, 4, 5, 0x1F00F)) {
            Opcode::S_MAP_V_FP { rd, rs1, imm } => assert_eq!((rd, rs1, imm), (4, 5, 0x1F00F)),
            other => panic!("expected S_MAP_V_FP, got {other:?}"),
        }
    }

    // ---------- field isolation (no cross-field bleed) ----------

    #[test]
    fn test_decode_operand_fields_are_masked_to_4_bits() {
        // All four operand fields set to 0xF must read back as 15 each, proving
        // each is masked to its own 4-bit window with no bleed.
        match Opcode::decode(rform(0x0D, 0xF, 0xF, 0xF, 0xF, 0)) {
            Opcode::V_ADD_VV {
                rd,
                rs1,
                rs2,
                rmask,
                ..
            } => assert_eq!((rd, rs1, rs2, rmask), (15, 15, 15, 15)),
            other => panic!("expected V_ADD_VV, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_lmask_does_not_bleed_into_rmask() {
        // funct1[2:0] must not leak into rmask (= rs3, bits 18..22).
        match Opcode::decode(rform(0x0D, 0, 0, 0, 0, 0x7)) {
            Opcode::V_ADD_VV { rmask, lmask, .. } => {
                assert_eq!(rmask, 0);
                assert_eq!(lmask, 0x7);
            }
            other => panic!("expected V_ADD_VV, got {other:?}"),
        }
    }

    #[test]
    fn non_mul_vector_ops_reject_the_reserved_variant_bit() {
        assert!(matches!(
            Opcode::decode(rform(0x0D, 0, 0, 0, 0, 0x8)),
            Opcode::Invalid
        ));
        assert!(matches!(
            Opcode::decode(rform(0x3D, 0, 0, 0, 0, 0x8)),
            Opcode::Invalid
        ));
    }

    #[test]
    fn test_decode_vector_scalar_minmax() {
        match Opcode::decode(rform(0x35, 1, 2, 3, 4, 0)) {
            Opcode::V_MAX_VF {
                rd,
                rs1,
                rs2,
                rmask,
                ..
            } => {
                assert_eq!((rd, rs1, rs2, rmask), (1, 2, 3, 4));
            }
            other => panic!("expected V_MAX_VF, got {other:?}"),
        }
        match Opcode::decode(rform(0x36, 5, 6, 7, 8, 0)) {
            Opcode::V_MIN_VF {
                rd,
                rs1,
                rs2,
                rmask,
                ..
            } => {
                assert_eq!((rd, rs1, rs2, rmask), (5, 6, 7, 8));
            }
            other => panic!("expected V_MIN_VF, got {other:?}"),
        }
        match Opcode::decode(rform(0x37, 9, 10, 11, 12, 0)) {
            Opcode::V_TOPK {
                rd,
                rs1,
                rs2,
                rmask,
            } => {
                assert_eq!((rd, rs1, rs2, rmask), (9, 10, 11, 12));
            }
            other => panic!("expected V_TOPK, got {other:?}"),
        }
        assert!(matches!(
            Opcode::decode(rform(0x37, 9, 10, 11, 12, 1)),
            Opcode::Invalid
        ));
    }
}
