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
        rd: u8,
    },
    M_BTMM {
        rs1: u8,
        rs2: u8,
        rd: u8,
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
    // VBcF {
    //     rd: u8,
    //     rs1: u8,
    // },
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
        // eprintln!(
        //     "decode(): instr = 0x{instr:08X} ({instr:032b})"
        // );
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
            0x03 => Self::M_BMM { rs1, rs2, rd },
            0x04 => Self::M_BTMM { rs1, rs2, rd },
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
            0x31 => Self::C_BREAK,
            _ => {
                eprintln!("Unknown opcode {opcode:#x}");
                Self::Invalid
            }
        }
    }
}
