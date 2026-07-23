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
    },
    V_ADD_VF {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rmask: u8,
    },
    V_SUB_VV {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rmask: u8,
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
    },
    V_MUL_VF {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rmask: u8,
    },
    V_EXP_V {
        rd: u8,
        rs1: u8,
        rmask: u8,
    },
    V_RECI_V {
        rd: u8,
        rs1: u8,
        rmask: u8,
    },
    V_RED_SUM {
        rd: u8,
        rs1: u8,
        rmask: u8,
    },
    V_RED_MAX {
        rd: u8,
        rs1: u8,
        rmask: u8,
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
    C_LOOP_START {
        rd: u8,
        imm: u32,
    },
    C_LOOP_END {
        rd: u8,
    },
    // Extensions
    V_SHIFT_V {
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

const fn mask(width: u32) -> u32 {
    ((1 << width) - 1) as u32
}

impl Opcode {
    /// ISA mnemonic of this opcode, without operands.
    ///
    /// Used as the per-opcode key in `--op-stats` output; must stay in sync
    /// with the variant names.
    pub(crate) fn mnemonic(&self) -> &'static str {
        match self {
            Self::Invalid => "Invalid",
            Self::M_MM { .. } => "M_MM",
            Self::M_TMM { .. } => "M_TMM",
            Self::M_BMM { .. } => "M_BMM",
            Self::M_BTMM { .. } => "M_BTMM",
            Self::M_BMM_WO { .. } => "M_BMM_WO",
            Self::M_MM_WO { .. } => "M_MM_WO",
            Self::M_MV { .. } => "M_MV",
            Self::M_TMV { .. } => "M_TMV",
            Self::M_BMV { .. } => "M_BMV",
            Self::M_BTMV { .. } => "M_BTMV",
            Self::M_MV_WO { .. } => "M_MV_WO",
            Self::M_BMV_WO { .. } => "M_BMV_WO",
            Self::V_ADD_VV { .. } => "V_ADD_VV",
            Self::V_ADD_VF { .. } => "V_ADD_VF",
            Self::V_SUB_VV { .. } => "V_SUB_VV",
            Self::V_SUB_VF { .. } => "V_SUB_VF",
            Self::V_MUL_VV { .. } => "V_MUL_VV",
            Self::V_MUL_VF { .. } => "V_MUL_VF",
            Self::V_EXP_V { .. } => "V_EXP_V",
            Self::V_RECI_V { .. } => "V_RECI_V",
            Self::V_RED_SUM { .. } => "V_RED_SUM",
            Self::V_RED_MAX { .. } => "V_RED_MAX",
            Self::S_ADD_FP { .. } => "S_ADD_FP",
            Self::S_SUB_FP { .. } => "S_SUB_FP",
            Self::S_MAX_FP { .. } => "S_MAX_FP",
            Self::S_MUL_FP { .. } => "S_MUL_FP",
            Self::S_EXP_FP { .. } => "S_EXP_FP",
            Self::S_RECI_FP { .. } => "S_RECI_FP",
            Self::S_SQRT_FP { .. } => "S_SQRT_FP",
            Self::S_LD_FP { .. } => "S_LD_FP",
            Self::S_ST_FP { .. } => "S_ST_FP",
            Self::S_MAP_V_FP { .. } => "S_MAP_V_FP",
            Self::S_ADD_INT { .. } => "S_ADD_INT",
            Self::S_ADDI_INT { .. } => "S_ADDI_INT",
            Self::S_SUB_INT { .. } => "S_SUB_INT",
            Self::S_MUL_INT { .. } => "S_MUL_INT",
            Self::S_LUI_INT { .. } => "S_LUI_INT",
            Self::S_LD_INT { .. } => "S_LD_INT",
            Self::S_ST_INT { .. } => "S_ST_INT",
            Self::H_PREFETCH_M { .. } => "H_PREFETCH_M",
            Self::H_PREFETCH_V { .. } => "H_PREFETCH_V",
            Self::H_STORE_V { .. } => "H_STORE_V",
            Self::C_SET_ADDR_REG { .. } => "C_SET_ADDR_REG",
            Self::C_SET_SCALE_REG { .. } => "C_SET_SCALE_REG",
            Self::C_SET_STRIDE_REG { .. } => "C_SET_STRIDE_REG",
            Self::C_SET_V_MASK_REG { .. } => "C_SET_V_MASK_REG",
            Self::C_LOOP_START { .. } => "C_LOOP_START",
            Self::C_LOOP_END { .. } => "C_LOOP_END",
            Self::V_SHIFT_V { .. } => "V_SHIFT_V",
            Self::C_BREAK => "C_BREAK",
        }
    }

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
        let imm = ((instr >> (OPCODE_WIDTH + OPERAND_WIDTH)) & mask(IMM_WIDTH)) as u32;
        let imm2 = ((instr >> (OPCODE_WIDTH + OPERAND_WIDTH * 2)) & mask(IMM_2_WIDTH)) as u32;

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
            0x0D => Self::V_ADD_VV {
                rd,
                rs1,
                rs2,
                rmask: rs3,
            },
            0x0E => Self::V_ADD_VF {
                rd,
                rs1,
                rs2,
                rmask: rs3,
            },
            0x0F => Self::V_SUB_VV {
                rd,
                rs1,
                rs2,
                rmask: rs3,
            },
            0x10 => Self::V_SUB_VF {
                rd,
                rs1,
                rs2,
                rmask: rs3,
                rorder: Self::vector_order_from(funct1),
            },
            0x11 => Self::V_MUL_VV {
                rd,
                rs1,
                rs2,
                rmask: rs3,
            },
            0x12 => Self::V_MUL_VF {
                rd,
                rs1,
                rs2,
                rmask: rs3,
            },
            0x13 => Self::V_EXP_V {
                rd,
                rs1,
                rmask: rs3,
            },
            0x14 => Self::V_RECI_V {
                rd,
                rs1,
                rmask: rs3,
            },
            0x15 => Self::V_RED_SUM {
                rd,
                rs1,
                rmask: rs3,
            },
            0x16 => Self::V_RED_MAX {
                rd,
                rs1,
                rmask: rs3,
            },

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
            // 0x31 (V_PS_V) and 0x33 (C_HADAMARD_TRANSFORM) are in the ISA
            // spec but not implemented here; they fall through to Invalid.
            0x32 => Self::V_SHIFT_V { rd, rs1, rs2 },
            0x34 => Self::C_BREAK,
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
            } => assert_eq!((rd, rs1, rs2, rmask), (1, 2, 3, 4)),
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
        // 0x3F is past the highest defined opcode (0x32).
        assert!(matches!(Opcode::decode(0x3F), Opcode::Invalid));
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
        // Spec encoding (operation.svh): C_BREAK = 6'h34.
        assert!(matches!(Opcode::decode(0x34), Opcode::C_BREAK));
    }

    #[test]
    fn test_decode_v_shift_v() {
        // Spec encoding (operation.svh): V_SHFT_V = 6'h32.
        match Opcode::decode(rform(0x32, 1, 2, 3, 0, 0)) {
            Opcode::V_SHIFT_V { rd, rs1, rs2 } => assert_eq!((rd, rs1, rs2), (1, 2, 3)),
            other => panic!("expected V_SHIFT_V, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_spec_unimplemented_extensions_are_invalid() {
        // V_PS_V (0x31) and C_HADAMARD_TRANSFORM (0x33) exist in the ISA spec
        // but not in this emulator; they must decode Invalid, never alias
        // another op.
        assert!(matches!(Opcode::decode(0x31), Opcode::Invalid));
        assert!(matches!(Opcode::decode(0x33), Opcode::Invalid));
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
            } => assert_eq!((rd, rs1, rs2, rmask), (15, 15, 15, 15)),
            other => panic!("expected V_ADD_VV, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_funct1_does_not_bleed_into_rmask() {
        // funct1 (bits 22..26) must not leak into rmask (= rs3, bits 18..22).
        match Opcode::decode(rform(0x0D, 0, 0, 0, 0, 0xF)) {
            Opcode::V_ADD_VV { rmask, .. } => assert_eq!(rmask, 0),
            other => panic!("expected V_ADD_VV, got {other:?}"),
        }
    }
}
