//! Opcode execution for [`Accelerator`].
//!
//! The public accelerator facade stays in `mod.rs`; this module owns the ISA
//! match and dispatch-only helpers.

use half::bf16;
use quantize::MxDataType;

use crate::runtime_config::{
    HLEN, MATRIX_KV_TYPE, MATRIX_WEIGHT_TYPE, MLEN, PREFETCH_M_AMOUNT, PREFETCH_V_AMOUNT,
    SCALAR_FP_BASIC_CYCLES, SCALAR_FP_EXP_CYCLES, SCALAR_FP_RECI_CYCLES, SCALAR_FP_SQRT_CYCLES,
    SCALAR_INT_BASIC_CYCLES, STORE_V_AMOUNT, VECTOR_ACTIVATION_TYPE, VECTOR_KV_TYPE, VLEN,
};
use crate::{cycle, dma, op};

use super::Accelerator;
use super::loop_state::LoopDecision;

impl Accelerator {
    /// Resolve the V_* opcode mask.
    ///
    /// When `rmask == 0`, the opcode operates on all HLEN heads of the VLEN
    /// vector (mask = all-ones over `*HLEN` bits). Otherwise the per-head mask
    /// stored in `reg_file.v_mask` is used directly.
    fn resolve_v_mask(&self, rmask: u8) -> u32 {
        if rmask == 0 {
            (1 << *HLEN) - 1
        } else {
            self.reg_file.v_mask()
        }
    }

    fn mx_region(&self, dtype: MxDataType, addr: u64, offset: u32, rstride: u8) -> dma::MxRegion {
        let scale = match dtype {
            MxDataType::Plain(_) => 0,
            MxDataType::Mx { .. } => offset / dtype.element_scale_ratio(),
        };

        dma::MxRegion {
            hbm_type: dtype,
            index: addr + offset as u64,
            // Scales are stored AFTER elements, so scale_index =
            // element_index + scale_reg + scale, where scale_reg is the offset
            // from element start to scale start.
            scale_index: addr + self.reg_file.scale() as u64 + scale as u64,
            rstride,
            stride: self.reg_file.stride(),
        }
    }

    pub(crate) async fn do_ops(&mut self, ops: &[op::Opcode]) {
        let mut pc: usize = 0; // Program counter

        while pc < ops.len() {
            let op = &ops[pc];

            self.loop_state.record_instruction();

            tracing::debug!(pc, ?op, "execute op");

            let mut jump_pc: Option<usize> = None;

            match op {
                op::Opcode::Invalid => {
                    tracing::error!(pc, "invalid opcode reached in dispatch");
                    panic!("invalid opcode at pc {pc}");
                }

                op::Opcode::M_MM { rs1, rs2 } => {
                    self.m_machine
                        .mm(self.reg_file.read_gp(*rs1), self.reg_file.read_gp(*rs2))
                        .await;
                }
                op::Opcode::M_MM_WO { rd, rstride, imm } => {
                    let stride_len = if *rstride == 0 {
                        1
                    } else {
                        self.reg_file.read_gp(*rstride)
                    };
                    self.m_machine
                        .mm_wo(self.reg_file.read_gp(*rd) + *imm, stride_len)
                        .await;
                }
                op::Opcode::M_TMM { rs1, rs2 } => {
                    self.m_machine
                        .tmm(self.reg_file.read_gp(*rs1), self.reg_file.read_gp(*rs2))
                        .await;
                }
                op::Opcode::M_BMM { rs1, rs2 } => {
                    self.m_machine
                        .bmm(
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_gp(*rs2),
                            self.reg_file.bmm_scale(),
                        )
                        .await;
                }
                op::Opcode::M_BTMM { rs1, rs2 } => {
                    self.m_machine
                        .btmm(
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_gp(*rs2),
                            self.reg_file.bmm_scale(),
                        )
                        .await;
                }
                op::Opcode::M_BMM_WO { rd, imm } => {
                    self.m_machine
                        .bmm_wo(self.reg_file.read_gp(*rd) + *imm)
                        .await;
                }
                op::Opcode::M_MV { rs1, rs2 } => {
                    self.m_machine
                        .mv(self.reg_file.read_gp(*rs1), self.reg_file.read_gp(*rs2))
                        .await;
                }
                op::Opcode::M_TMV { rs1, rs2 } => {
                    self.m_machine
                        .tmv(self.reg_file.read_gp(*rs1), self.reg_file.read_gp(*rs2))
                        .await;
                }
                op::Opcode::M_BMV { rs1, rs2, rd } => {
                    self.m_machine
                        .bmv(
                            self.reg_file.read_gp(*rs1) + self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs2),
                            self.reg_file.bmm_scale(),
                        )
                        .await;
                }
                op::Opcode::M_BTMV { rs1, rs2, rd } => {
                    self.m_machine
                        .btmv(
                            self.reg_file.read_gp(*rs1) + self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs2),
                            self.reg_file.bmm_scale(),
                        )
                        .await;
                }
                op::Opcode::M_MV_WO { rd, imm } => {
                    self.m_machine
                        .mv_wo(self.reg_file.read_gp(*rd) + *imm)
                        .await;
                }
                op::Opcode::M_BMV_WO { rd, imm } => {
                    self.m_machine
                        .bmv_wo(self.reg_file.read_gp(*rd) + *imm)
                        .await;
                }

                op::Opcode::V_ADD_VV {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .add(
                            self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_gp(*rs2),
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_ADD_VF {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .add_scalar(
                            self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_fp(*rs2).into(),
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_SUB_VV {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .sub(
                            self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_gp(*rs2),
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_SUB_VF {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                    rorder,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .sub_scalar(
                            self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_fp(*rs2).into(),
                            *rmask,
                            mask,
                            *rorder,
                        )
                        .await;
                }
                op::Opcode::V_MUL_VV {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .mul(
                            self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_gp(*rs2),
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_MUL_VF {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .mul_scalar(
                            self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_fp(*rs2).into(),
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_EXP_V { rd, rs1, rmask } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .exp(
                            self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs1),
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_RECI_V { rd, rs1, rmask } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .reciprocal(
                            self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs1),
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_SHIFT_V { rd, rs1, rs2 } => {
                    self.v_machine
                        .shift_scalar(
                            self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_gp(*rs2),
                        )
                        .await;
                }
                // Write to fp0 is a no-op.
                op::Opcode::V_RED_SUM { rd: 0, .. } | op::Opcode::V_RED_MAX { rd: 0, .. } => (),

                op::Opcode::V_RED_SUM { rd, rs1, rmask } => {
                    let mask = self.resolve_v_mask(*rmask);
                    let result = self
                        .v_machine
                        .reduce_sum(
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_fp(*rd).into(),
                            *rmask,
                            mask,
                        )
                        .await;
                    self.reg_file.write_fp(*rd, bf16::from_f32(result));
                }
                op::Opcode::V_RED_MAX { rd, rs1, rmask } => {
                    let mask = self.resolve_v_mask(*rmask);
                    let result = self
                        .v_machine
                        .reduce_max(
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_fp(*rd).into(),
                            *rmask,
                            mask,
                        )
                        .await;
                    self.reg_file.write_fp(*rd, bf16::from_f32(result));
                }

                // Write to fp0 is a no-op.
                op::Opcode::S_ADD_FP { rd: 0, .. }
                | op::Opcode::S_SUB_FP { rd: 0, .. }
                | op::Opcode::S_MAX_FP { rd: 0, .. }
                | op::Opcode::S_MUL_FP { rd: 0, .. }
                | op::Opcode::S_EXP_FP { rd: 0, .. }
                | op::Opcode::S_RECI_FP { rd: 0, .. }
                | op::Opcode::S_SQRT_FP { rd: 0, .. } => {}

                op::Opcode::S_ADD_FP { rd, rs1, rs2 } => {
                    self.reg_file.binop_fp(*rd, *rs1, *rs2, std::ops::Add::add);
                    cycle!(*SCALAR_FP_BASIC_CYCLES);
                }
                op::Opcode::S_SUB_FP { rd, rs1, rs2 } => {
                    self.reg_file.binop_fp(*rd, *rs1, *rs2, std::ops::Sub::sub);
                    cycle!(*SCALAR_FP_BASIC_CYCLES);
                }
                op::Opcode::S_MAX_FP { rd, rs1, rs2 } => {
                    self.reg_file.binop_fp(*rd, *rs1, *rs2, bf16::max);
                    cycle!(*SCALAR_FP_BASIC_CYCLES);
                }
                op::Opcode::S_MUL_FP { rd, rs1, rs2 } => {
                    self.reg_file.binop_fp(*rd, *rs1, *rs2, std::ops::Mul::mul);
                    cycle!(*SCALAR_FP_BASIC_CYCLES);
                }
                op::Opcode::S_EXP_FP { rd, rs1 } => {
                    let val: f32 = self.reg_file.read_fp(*rs1).into();
                    let clamped = val.clamp(-88.0, 88.0);
                    self.reg_file.write_fp(*rd, bf16::from_f32(clamped.exp()));
                    cycle!(*SCALAR_FP_EXP_CYCLES);
                }
                op::Opcode::S_RECI_FP { rd, rs1 } => {
                    self.reg_file
                        .write_fp(*rd, bf16::ONE / self.reg_file.read_fp(*rs1));
                    cycle!(*SCALAR_FP_RECI_CYCLES);
                }
                op::Opcode::S_SQRT_FP { rd, rs1 } => {
                    self.reg_file.write_fp(
                        *rd,
                        bf16::from_f32(f32::from(self.reg_file.read_fp(*rs1)).sqrt()),
                    );
                    cycle!(*SCALAR_FP_SQRT_CYCLES);
                }
                op::Opcode::S_LD_FP { rd, rs1, imm } => {
                    self.reg_file.write_fp(
                        *rd,
                        self.scalar_sram
                            .read_fp((self.reg_file.read_gp(*rs1) + *imm) as usize),
                    );
                    cycle!(1);
                }
                op::Opcode::S_ST_FP { rd, rs1, imm } => {
                    self.scalar_sram.write_fp(
                        (self.reg_file.read_gp(*rs1) + *imm) as usize,
                        self.reg_file.read_fp(*rd),
                    );
                    cycle!(1);
                }
                op::Opcode::S_MAP_V_FP { rd, rs1, imm } => {
                    let start_idx = (self.reg_file.read_gp(*rs1) + *imm) as usize;
                    let f = self.scalar_sram.read_fp_window(start_idx, *VLEN as usize);
                    self.v_machine
                        .vector_transfer_fp(self.reg_file.read_gp(*rd), f)
                        .await;
                    cycle!(*VLEN);
                }
                op::Opcode::S_ADD_INT { rd, rs1, rs2 } => {
                    self.reg_file.binop_gp(*rd, *rs1, *rs2, u32::wrapping_add);
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::S_ADDI_INT { rd, rs1, imm } => {
                    self.reg_file
                        .write_gp(*rd, self.reg_file.read_gp(*rs1).wrapping_add(*imm));
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::S_SUB_INT { rd, rs1, rs2 } => {
                    self.reg_file.binop_gp(*rd, *rs1, *rs2, u32::wrapping_sub);
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::S_MUL_INT { rd, rs1, rs2 } => {
                    self.reg_file.binop_gp(*rd, *rs1, *rs2, u32::wrapping_mul);
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::S_LUI_INT { rd, imm } => {
                    self.reg_file.write_gp(*rd, *imm << 12);
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::S_LD_INT { rd, rs1, imm } => {
                    self.reg_file.write_gp(
                        *rd,
                        self.scalar_sram
                            .read_int((self.reg_file.read_gp(*rs1) + *imm) as usize),
                    );
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::S_ST_INT { rd, rs1, imm } => {
                    self.scalar_sram.write_int(
                        (self.reg_file.read_gp(*rs1) + *imm) as usize,
                        self.reg_file.read_gp(*rd),
                    );
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::H_PREFETCH_M {
                    rd,
                    rs1,
                    rs2,
                    rstride,
                    precision,
                } => {
                    // TODO: rstride support to be added
                    let offset = self.reg_file.read_gp(*rs1);
                    let addr = self.reg_file.read_hbm(*rs2);
                    let dtype = match precision {
                        op::MatrixPrecision::Weights => *MATRIX_WEIGHT_TYPE,
                        op::MatrixPrecision::KeyValue => *MATRIX_KV_TYPE,
                    };

                    let region = self.mx_region(dtype, addr, offset, *rstride);
                    let xfer = dma::transfer_mx_from_hbm(
                        &self.hbm,
                        region,
                        self.m_machine.mram.ty(),
                        *MLEN,
                        *PREFETCH_M_AMOUNT,
                        *MLEN,
                    );

                    self.m_machine
                        .mram
                        .continous_write_delayed(
                            self.reg_file.read_gp(*rd),
                            *PREFETCH_M_AMOUNT,
                            xfer,
                        )
                        .await;
                }
                op::Opcode::H_PREFETCH_V {
                    rd,
                    rs1,
                    rs2,
                    rstride,
                    precision,
                } => {
                    // TODO: rstride support to be added
                    let offset = self.reg_file.read_gp(*rs1);
                    let addr = self.reg_file.read_hbm(*rs2);
                    let dtype = match precision {
                        op::VectorPrecision::Activation => *VECTOR_ACTIVATION_TYPE,
                        op::VectorPrecision::KeyValue => *VECTOR_KV_TYPE,
                    };

                    let region = self.mx_region(dtype, addr, offset, *rstride);
                    let xfer = dma::transfer_mx_from_hbm(
                        &self.hbm,
                        region,
                        self.v_machine.vram.ty(),
                        *VLEN,
                        *PREFETCH_V_AMOUNT,
                        1,
                    );

                    let dest = self.reg_file.read_gp(*rd);
                    self.v_machine
                        .vram
                        .continous_write_delayed(dest, *PREFETCH_V_AMOUNT, xfer)
                        .await;
                }
                op::Opcode::H_STORE_V {
                    rd,
                    rs1,
                    rs2,
                    rstride,
                    precision,
                } => {
                    let src_addr = self.reg_file.read_gp(*rd);
                    let offset = self.reg_file.read_gp(*rs1);
                    let addr = self.reg_file.read_hbm(*rs2);
                    let dtype = match precision {
                        op::VectorPrecision::Activation => *VECTOR_ACTIVATION_TYPE,
                        op::VectorPrecision::KeyValue => *VECTOR_KV_TYPE,
                    };

                    let region = self.mx_region(dtype, addr, offset, *rstride);

                    dma::transfer_mx_to_hbm(
                        &self.hbm,
                        &self.v_machine.vram,
                        region,
                        src_addr,
                        *VLEN,
                        *STORE_V_AMOUNT,
                    )
                    .await;
                }
                op::Opcode::C_SET_ADDR_REG { rd, rs1, rs2 } => {
                    let imm = ((self.reg_file.read_gp(*rs1) as u64) << 32)
                        | (self.reg_file.read_gp(*rs2) as u64);
                    self.reg_file.write_hbm(*rd, imm);
                    cycle!(1);
                }
                op::Opcode::C_SET_SCALE_REG { rd } => {
                    self.reg_file.set_scale(self.reg_file.read_gp(*rd));
                    cycle!(1);
                }
                op::Opcode::C_SET_STRIDE_REG { rd } => {
                    self.reg_file.set_stride(self.reg_file.read_gp(*rd));
                    cycle!(1);
                }
                op::Opcode::C_SET_V_MASK_REG { rd } => {
                    self.reg_file.set_v_mask(self.reg_file.read_gp(*rd));
                    cycle!(1);
                }
                op::Opcode::C_LOOP_START { rd, imm } => {
                    self.loop_state.start(pc, *rd, *imm, &mut self.reg_file);
                    cycle!(1);
                }
                op::Opcode::C_LOOP_END { rd } => {
                    if let LoopDecision::JumpTo(target_pc) =
                        self.loop_state.end(*rd, &mut self.reg_file)
                    {
                        jump_pc = Some(target_pc);
                    }
                    cycle!(1);
                }
                op::Opcode::C_BREAK => {
                    self.loop_state.break_innermost(&mut self.reg_file);
                    cycle!(1);
                }
            }

            // Handle loop jumps
            if let Some(target_pc) = jump_pc {
                pc = target_pc;
            } else {
                pc += 1;
            }
        }
    }
}
