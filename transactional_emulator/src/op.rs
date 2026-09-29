#[derive(Debug, Clone, Copy)]
pub enum MatrixPrecision {
    Weights,
    KeyValue,
}

#[derive(Debug, Clone, Copy)]
pub enum VectorPrecision {
    Activation,
    KeyValue,
    State,
}

#[derive(Debug, Clone, Copy)]
pub enum VectorOrder {
    Normal,
    Reverse,
}

/// Model-independent algebraic forms executed over configured Matrix views.
/// Loop bounds and broadcasting come from the views, never from a model ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LTilePrimitive {
    ScaleAccum,
    DotReduce,
    OuterUpdate,
}

impl TryFrom<u8> for LTilePrimitive {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::ScaleAccum),
            1 => Ok(Self::DotReduce),
            2 => Ok(Self::OuterUpdate),
            _ => Err(()),
        }
    }
}

/// Logical line direction selected independently for each `L_TILE_EXEC` input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LTileAxis {
    Row,
    Column,
}

impl From<u8> for LTileAxis {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Row,
            1 => Self::Column,
            _ => unreachable!("L_TILE axis is one bit"),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug)]
pub enum Opcode {
    Invalid,
    M_MM {
        rs1: u8,
        rs2: u8,
        view: Option<u8>,
    },
    M_TMM {
        rs1: u8,
        rs2: u8,
        view: Option<u8>,
    },
    M_BMM {
        rs1: u8,
        rs2: u8,
        view: Option<u8>,
    },
    M_BTMM {
        rs1: u8,
        rs2: u8,
        view: Option<u8>,
    },
    M_BMM_WO {
        rd: u8,
        imm: u32,
    },
    M_MM_WO {
        rd: u8,
        rstride: u8,
        imm: u32,
        view: Option<u8>,
    },
    M_MV {
        rs1: u8,
        rs2: u8,
        view: Option<u8>,
    },
    M_TMV {
        rs1: u8,
        rs2: u8,
        view: Option<u8>,
    },
    M_BMV {
        rs1: u8,
        rs2: u8,
        rd: u8,
        view: Option<u8>,
    },
    M_BTMV {
        rs1: u8,
        rs2: u8,
        rd: u8,
        view: Option<u8>,
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
        view_mask: u8,
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
        view_mask: u8,
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
        view_mask: u8,
    },
    V_MUL_VF {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rmask: u8,
    },
    V_MAX_VF {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rmask: u8,
    },
    V_MIN_VF {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rmask: u8,
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
    /// Set the sticky routed-MoE top-k policy read by `V_TOPK rmask=15`.
    ///
    /// `rd` names a GP register holding `(num_experts << 8) | top_k`. Sticky like
    /// the other `C_SET_*_REG` registers, so a single-policy program sets it once.
    C_SET_TOPK_REG {
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
    V_SHFT_V {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    C_BREAK,
    H_PREFETCH_V_MV {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rstride: u8,
        precision: VectorPrecision,
        view: u8,
    },
    H_STORE_V_MV {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rstride: u8,
        precision: VectorPrecision,
        view: u8,
    },
    L_TILE_CFG {
        shape: u8,
        mapping: u8,
        slot: u8,
    },
    L_TILE_EXEC {
        rd: u8,
        rs1: u8,
        rs2: u8,
        primitive: LTilePrimitive,
        source_axis: LTileAxis,
        scale_axis: LTileAxis,
    },
}

const OPERAND_WIDTH: u32 = 4;
const OPCODE_WIDTH: u32 = 6;
const IMM_WIDTH: u32 = 22;
const IMM_2_WIDTH: u32 = 18;

const fn mask(width: u32) -> u32 {
    ((1 << width) - 1) as u32
}

impl Opcode {
    fn matrix_view_from(funct1: u8) -> Option<u8> {
        match funct1 {
            0 => None,
            1..=4 => Some(funct1 - 1),
            _ => unreachable!("caller must reject reserved Matrix-view selector"),
        }
    }

    fn matrix_writeback_view(immediate: u32) -> (u32, Option<u8>) {
        const VIEW_MARKER: u32 = 1 << 17;
        if immediate & VIEW_MARKER == 0 {
            (immediate, None)
        } else {
            (
                immediate & ((1 << 15) - 1),
                Some(((immediate >> 15) & 0x3) as u8),
            )
        }
    }

    fn matrix_view_vector_precision_from(funct1: u8) -> Option<VectorPrecision> {
        // Explicit Matrix-view DMA adds an independent state selector. Ordinary
        // DMA preserves its original 0=activation/nonzero=KV interpretation.
        match funct1 {
            0 => Some(VectorPrecision::Activation),
            1 => Some(VectorPrecision::KeyValue),
            2 => Some(VectorPrecision::State),
            _ => None,
        }
    }

    fn matrix_view_dma_slot(instr: u32) -> Result<Option<u8>, ()> {
        let high = (instr >> 26) as u8;
        if high == 0 {
            return Ok(None);
        }
        // bit31 marks the form, bits30:29 are the slot, bits28:26 are zero.
        if high & 0b100000 == 0 || high & 0b000111 != 0 {
            return Err(());
        }
        Ok(Some((high >> 3) & 0b11))
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
        let imm = (instr >> (OPCODE_WIDTH + OPERAND_WIDTH)) & mask(IMM_WIDTH);
        let imm2 = (instr >> (OPCODE_WIDTH + OPERAND_WIDTH * 2)) & mask(IMM_2_WIDTH);

        match opcode {
            0x00 => Self::Invalid,
            // Matrix Operations
            0x01 if funct1 <= 4 => Self::M_MM {
                rs1,
                rs2,
                view: Self::matrix_view_from(funct1),
            },
            0x02 if funct1 <= 4 => Self::M_TMM {
                rs1,
                rs2,
                view: Self::matrix_view_from(funct1),
            },
            0x03 if funct1 <= 4 => {
                // ISA spec defines matrix address as `gp_reg<rs1> + gp_reg<rd>` but
                // this emulator only consumes `rs1`. M_BMV/M_BTMV honor `rd`; until
                // M_BMM/M_BTMM follow suit, refuse encodings that would otherwise
                // silently drop the rd offset.
                assert_eq!(
                    rd, 0,
                    "M_BMM rd must be 0: emulator does not honor the spec's `gp_reg<rd>` matrix offset"
                );
                Self::M_BMM {
                    rs1,
                    rs2,
                    view: Self::matrix_view_from(funct1),
                }
            }
            0x04 if funct1 <= 4 => {
                assert_eq!(
                    rd, 0,
                    "M_BTMM rd must be 0: emulator does not honor the spec's `gp_reg<rd>` matrix offset"
                );
                Self::M_BTMM {
                    rs1,
                    rs2,
                    view: Self::matrix_view_from(funct1),
                }
            }
            0x05 => Self::M_BMM_WO { rd, imm: imm2 },
            0x06 => {
                let (imm, view) = Self::matrix_writeback_view(imm2);
                Self::M_MM_WO {
                    rd,
                    rstride: rs1,
                    imm,
                    view,
                }
            }
            0x07 if funct1 <= 4 => Self::M_MV {
                rs1,
                rs2,
                view: Self::matrix_view_from(funct1),
            },
            0x08 if funct1 <= 4 => Self::M_TMV {
                rs1,
                rs2,
                view: Self::matrix_view_from(funct1),
            },
            0x09 if funct1 <= 4 => Self::M_BMV {
                rs1,
                rs2,
                rd,
                view: Self::matrix_view_from(funct1),
            },
            0x0A if funct1 <= 4 => Self::M_BTMV {
                rs1,
                rs2,
                rd,
                view: Self::matrix_view_from(funct1),
            },
            0x01..=0x04 | 0x07..=0x0A => {
                tracing::error!(instr, funct1, "reserved Matrix-view selector");
                Self::Invalid
            }
            0x0B => Self::M_MV_WO { rd, imm: imm2 },
            0x0C => Self::M_BMV_WO { rd, imm: imm2 },

            // Vector Operations
            0x0D if funct1 == 0 || funct1 >= 9 => Self::V_ADD_VV {
                rd,
                rs1,
                rs2,
                rmask: rs3,
                view_mask: funct1 & 7,
            },
            0x0E => Self::V_ADD_VF {
                rd,
                rs1,
                rs2,
                rmask: rs3,
            },
            0x0F if funct1 == 0 || funct1 >= 9 => Self::V_SUB_VV {
                rd,
                rs1,
                rs2,
                rmask: rs3,
                view_mask: funct1 & 7,
            },
            0x10 => Self::V_SUB_VF {
                rd,
                rs1,
                rs2,
                rmask: rs3,
                rorder: Self::vector_order_from(funct1),
            },
            0x11 if funct1 == 0 || funct1 >= 9 => Self::V_MUL_VV {
                rd,
                rs1,
                rs2,
                rmask: rs3,
                view_mask: funct1 & 7,
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
            0x35 => Self::V_MAX_VF {
                rd,
                rs1,
                rs2,
                rmask: rs3,
            },
            0x36 => Self::V_MIN_VF {
                rd,
                rs1,
                rs2,
                rmask: rs3,
            },
            0x37 => Self::V_TOPK {
                rd,
                rs1,
                rs2,
                rmask: rs3,
            },

            0x0D | 0x0F | 0x11 => Self::Invalid,

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
            0x29 => match Self::matrix_view_dma_slot(instr) {
                Ok(None) => Self::H_PREFETCH_V {
                    rd,
                    rs1,
                    rs2,
                    rstride: rs3,
                    precision: Self::vector_precision_from(funct1),
                },
                Ok(Some(view)) => match Self::matrix_view_vector_precision_from(funct1) {
                    Some(precision) => Self::H_PREFETCH_V_MV {
                        rd,
                        rs1,
                        rs2,
                        rstride: rs3,
                        precision,
                        view,
                    },
                    None => {
                        tracing::error!(instr, funct1, "reserved Matrix-view prefetch precision");
                        Self::Invalid
                    }
                },
                Err(()) => {
                    tracing::error!(instr, funct1, "reserved H_PREFETCH_V encoding");
                    Self::Invalid
                }
            },
            // 0x2A => Self::H_PREFETCH_V { rd, rs1, rs2, rstride: rs3, precision: VectorPrecision::KeyValue },
            0x2A => match Self::matrix_view_dma_slot(instr) {
                Ok(None) => Self::H_STORE_V {
                    rd,
                    rs1,
                    rs2,
                    rstride: rs3,
                    precision: Self::vector_precision_from(funct1),
                },
                Ok(Some(view)) => match Self::matrix_view_vector_precision_from(funct1) {
                    Some(precision) => Self::H_STORE_V_MV {
                        rd,
                        rs1,
                        rs2,
                        rstride: rs3,
                        precision,
                        view,
                    },
                    None => {
                        tracing::error!(instr, funct1, "reserved Matrix-view store precision");
                        Self::Invalid
                    }
                },
                Err(()) => {
                    tracing::error!(instr, funct1, "reserved H_STORE_V encoding");
                    Self::Invalid
                }
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
            0x3F if funct1 == 1 => {
                if instr >> 26 != 0 || rs2 >= 4 || rs3 != 0 {
                    tracing::error!(instr, "non-canonical L_TILE_CFG encoding");
                    Self::Invalid
                } else {
                    Self::L_TILE_CFG {
                        shape: rd,
                        mapping: rs1,
                        slot: rs2,
                    }
                }
            }
            0x3F if funct1 == 3 => {
                if instr >> 28 != 0 {
                    tracing::error!(instr, "non-canonical L_TILE_EXEC encoding");
                    Self::Invalid
                } else if let Ok(primitive) = LTilePrimitive::try_from(rs3) {
                    Self::L_TILE_EXEC {
                        rd,
                        rs1,
                        rs2,
                        primitive,
                        source_axis: LTileAxis::from(((instr >> 26) & 1) as u8),
                        scale_axis: LTileAxis::from(((instr >> 27) & 1) as u8),
                    }
                } else {
                    tracing::error!(instr, rs3, "reserved L_TILE primitive");
                    Self::Invalid
                }
            }
            0x3F => {
                tracing::error!(instr, funct1, "reserved L_TILE form");
                Self::Invalid
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
                view_mask,
            } => assert_eq!((rd, rs1, rs2, rmask, view_mask), (1, 2, 3, 4, 0)),
            other => panic!("expected V_ADD_VV, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_two_register_matrix_op() {
        // M_MM consumes only rs1 and rs2.
        match Opcode::decode(rform(0x01, 0, 5, 6, 0, 0)) {
            Opcode::M_MM { rs1, rs2, .. } => assert_eq!((rs1, rs2), (5, 6)),
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
            Opcode::M_BMM { rs1, rs2, .. } => assert_eq!((rs1, rs2), (7, 8)),
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
        match Opcode::decode(i2form(0x06, 5, 6, 0x0BEEF)) {
            Opcode::M_MM_WO {
                rd,
                rstride,
                imm,
                view,
            } => {
                assert_eq!((rd, rstride, imm, view), (5, 6, 0x0BEEF, None))
            }
            other => panic!("expected M_MM_WO, got {other:?}"),
        }
    }

    #[test]
    fn test_decode_m_bmv_carries_rd() {
        // M_BMV honors rd (unlike M_BMM); decode keeps all three.
        match Opcode::decode(rform(0x09, 9, 7, 8, 0, 0)) {
            Opcode::M_BMV { rs1, rs2, rd, .. } => assert_eq!((rs1, rs2, rd), (7, 8, 9)),
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
        for legacy_nonzero in 2..=15 {
            assert!(matches!(
                Opcode::decode(rform(0x29, 0, 0, 0, 0, legacy_nonzero)),
                Opcode::H_PREFETCH_V {
                    precision: VectorPrecision::KeyValue,
                    ..
                }
            ));
        }
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
        for legacy_nonzero in 2..=15 {
            assert!(matches!(
                Opcode::decode(rform(0x2A, 1, 2, 3, 4, legacy_nonzero)),
                Opcode::H_STORE_V {
                    precision: VectorPrecision::KeyValue,
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_decode_vector_dma_matrix_view_form_and_reject_reserved_high_bits() {
        let marker = 1_u32 << 31;
        let slot = 3_u32 << 29;
        match Opcode::decode(rform(0x29, 1, 2, 3, 4, 2) | marker | slot) {
            Opcode::H_PREFETCH_V_MV {
                rd,
                rs1,
                rs2,
                rstride,
                precision: VectorPrecision::State,
                view,
            } => assert_eq!((rd, rs1, rs2, rstride, view), (1, 2, 3, 4, 3)),
            other => panic!("expected Matrix-view prefetch, got {other:?}"),
        }
        assert!(matches!(
            Opcode::decode(rform(0x2A, 1, 2, 3, 4, 2) | marker | (2 << 29)),
            Opcode::H_STORE_V_MV {
                precision: VectorPrecision::State,
                view: 2,
                ..
            }
        ));
        assert!(matches!(
            Opcode::decode(rform(0x29, 1, 2, 3, 4, 2) | marker | (1 << 28)),
            Opcode::Invalid
        ));
        assert!(matches!(
            Opcode::decode(rform(0x29, 1, 2, 3, 4, 3) | marker),
            Opcode::Invalid
        ));
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
                    Opcode::decode(rform(
                        opcode,
                        0,
                        0,
                        0,
                        0,
                        if opcode == 0x3F { 1 } else { 0 }
                    )),
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
    fn l_tile_forms_and_explicit_matrix_consumer_match_compiler_words() {
        match Opcode::decode(rform(0x3F, 7, 9, 2, 0, 1)) {
            Opcode::L_TILE_CFG {
                shape,
                mapping,
                slot,
            } => assert_eq!((shape, mapping, slot), (7, 9, 2)),
            other => panic!("expected L_TILE_CFG, got {other:?}"),
        }
        assert!(matches!(
            Opcode::decode(rform(0x3F, 9, 2, 2, 0, 2)),
            Opcode::Invalid
        ));
        match Opcode::decode(rform(0x09, 9, 5, 6, 0, 3)) {
            Opcode::M_BMV { rd, rs1, rs2, view } => {
                assert_eq!((rd, rs1, rs2, view), (9, 5, 6, Some(2)));
            }
            other => panic!("expected viewed M_BMV, got {other:?}"),
        }
        match Opcode::decode(i2form(0x06, 4, 0, (1 << 17) | (2 << 15) | 5)) {
            Opcode::M_MM_WO {
                rd,
                rstride,
                imm,
                view,
            } => assert_eq!((rd, rstride, imm, view), (4, 0, 5, Some(2))),
            other => panic!("expected viewed M_MM_WO, got {other:?}"),
        }
        match Opcode::decode(rform(0x0D, 4, 5, 6, 0, 0x8 | 0b110)) {
            Opcode::V_ADD_VV {
                rd,
                rs1,
                rs2,
                view_mask,
                ..
            } => assert_eq!((rd, rs1, rs2, view_mask), (4, 5, 6, 0b110)),
            other => panic!("expected Matrix-view V_ADD_VV, got {other:?}"),
        }
    }

    #[test]
    fn l_mview_rejects_reserved_bits_forms_and_consumer_slots() {
        assert!(matches!(
            Opcode::decode(rform(0x3F, 7, 9, 2, 1, 1)),
            Opcode::Invalid
        ));
        assert!(matches!(
            Opcode::decode(rform(0x3F, 7, 9, 2, 0, 6)),
            Opcode::Invalid
        ));
        assert!(matches!(
            Opcode::decode(rform(0x01, 0, 5, 6, 0, 5)),
            Opcode::Invalid
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
            Opcode::M_BTMV { rs1, rs2, rd, .. } => assert_eq!((rs1, rs2, rd), (7, 8, 9)),
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
    }
}
