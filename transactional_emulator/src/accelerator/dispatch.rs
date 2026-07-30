//! Opcode execution for [`Accelerator`].
//!
//! The public accelerator facade stays in `mod.rs`; this module owns the ISA
//! match and dispatch-only helpers.

use half::bf16;
use quantize::MxDataType;

use crate::runtime_config::{
    BLEN, HLEN, MATRIX_KV_TYPE, MATRIX_WEIGHT_TYPE, MLEN, PREFETCH_M_AMOUNT, PREFETCH_V_AMOUNT,
    SCALAR_FP_BASIC_CYCLES, SCALAR_FP_EXP_CYCLES, SCALAR_FP_RECI_CYCLES, SCALAR_FP_SQRT_CYCLES,
    SCALAR_INT_BASIC_CYCLES, STORE_V_AMOUNT, VECTOR_ACTIVATION_TYPE, VECTOR_KV_TYPE, VLEN,
};
use crate::stage_profile::{ResourceKind, StageProfiler};
use crate::timing_overlay::{SramRange, SramSpace, TimingAccess, TimingOverlay};
use crate::{cycle, dma, op};
use runtime::Executor;

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

    pub(crate) async fn do_ops(
        &mut self,
        ops: &[op::Opcode],
        mut stage_profiler: Option<&mut StageProfiler>,
        mut timing_overlay: Option<&mut TimingOverlay>,
    ) {
        let mut pc: usize = 0; // Program counter

        while pc < ops.len() {
            let executed_pc = pc;
            let op = &ops[pc];
            let capture_elapsed = stage_profiler.is_some() || timing_overlay.is_some();
            let timing_access = if timing_overlay.is_some() {
                Some(self.timing_access_for_opcode(op))
            } else {
                None
            };
            let profile_start_instant = if capture_elapsed {
                Some(Executor::current().now())
            } else {
                None
            };
            let profile_start_hbm = if stage_profiler.is_some() {
                self.hbm.statistics()
            } else {
                None
            };

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
                op::Opcode::V_MAX_VF {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .max_scalar(
                            self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_fp(*rs2).into(),
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_MIN_VF {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .min_scalar(
                            self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_fp(*rs2).into(),
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_TOPK {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                } => {
                    let (expert_count, topk) = match *rmask {
                        0 => (32, 4),
                        1 => (128, 8),
                        // Escape to the sticky C_SET_TOPK_REG policy. Any shape that
                        // is not GPT-OSS or Qwen3-30B-A3B arrives here.
                        15 => match self.reg_file.topk_policy() {
                            Some(policy) => policy,
                            None => {
                                tracing::error!(pc, "V_TOPK rmask=15 with no C_SET_TOPK_REG");
                                panic!(
                                    "V_TOPK rmask=15 at pc {pc} requires a preceding \
                                     C_SET_TOPK_REG; the policy register is unset"
                                );
                            }
                        },
                        other => {
                            // Consistent with the Opcode::Invalid handler: a
                            // malformed-but-encodable field is a bad-program error,
                            // logged with the pc before aborting.
                            tracing::error!(pc, rmask = other, "unsupported V_TOPK rmask policy");
                            panic!(
                                "unsupported V_TOPK rmask policy {other} at pc {pc}; \
                                 expected 0=32/top4, 1=128/top8, or 15=C_SET_TOPK_REG"
                            );
                        }
                    };
                    let fp_base = self.reg_file.read_gp(*rd) as usize;
                    let int_base = self.reg_file.read_gp(*rs2) as usize;
                    let (indices, weights) = self
                        .v_machine
                        .topk_softmax(self.reg_file.read_gp(*rs1), expert_count, topk)
                        .await;
                    for (offset, (idx, weight)) in indices.iter().zip(weights.iter()).enumerate() {
                        self.scalar_sram.write_int(int_base + offset, *idx);
                        self.scalar_sram.write_fp(fp_base + offset, *weight);
                    }
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
                op::Opcode::V_SHFT_V { rd, rs1, rs2 } => {
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
                op::Opcode::C_SET_TOPK_REG { rd } => {
                    self.reg_file.set_topk_policy(self.reg_file.read_gp(*rd));
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

            if let Some(start_instant) = profile_start_instant {
                let elapsed_duration = Executor::current().now() - start_instant;
                let elapsed_picos = elapsed_duration.as_picos();
                let elapsed_secs = elapsed_picos as f64 / 1_000_000_000_000.0;
                if let (Some(access), Some(overlay)) =
                    (timing_access, timing_overlay.as_deref_mut())
                {
                    overlay.record(access, elapsed_picos);
                }
                if let Some(profiler) = stage_profiler.as_deref_mut() {
                    let (hbm_bytes_read, hbm_bytes_written) = if let (Some(before), Some(after)) =
                        (profile_start_hbm, self.hbm.statistics())
                    {
                        (
                            after
                                .total_bytes_read
                                .saturating_sub(before.total_bytes_read),
                            after
                                .total_bytes_written
                                .saturating_sub(before.total_bytes_written),
                        )
                    } else {
                        (0, 0)
                    };
                    profiler.record(
                        executed_pc,
                        elapsed_secs,
                        elapsed_picos,
                        resource_kind_for_opcode(op),
                        hbm_bytes_read,
                        hbm_bytes_written,
                    );
                }
            }
        }
    }

    fn timing_access_for_opcode(&self, op: &op::Opcode) -> TimingAccess {
        classify_timing_access(op, &|reg| self.reg_file.read_gp(reg), &|| {
            self.reg_file.topk_policy()
        })
    }
}

/// Classify an opcode's matrix/vector SRAM footprint for the overlap model.
///
/// # Invariant: this mirrors the execution arms in `do_ops`
///
/// The addresses and extents here are hand-derived from the corresponding
/// execution arm and the machine method it calls. There is no shared source of
/// truth, so **any change to how an execution arm addresses SRAM must be
/// reflected here**. Review found two arms that had already drifted (`M_MM_WO` /
/// `M_MV_WO` read-modify-write their destination row; `V_TOPK` reads
/// `ceil(expert_count / VLEN)` rows, not one), both of which made the overlay
/// optimistic.
///
/// Three guards limit the damage, none of which catches a wrong address:
///
/// - The match is exhaustive with no `_` arm, so a newly added opcode fails to
///   compile until it is classified.
/// - Zero-cost no-op arms in `do_ops` are mirrored explicitly (see the
///   `V_RED_SUM { rd: 0 }` arm). An earlier attempt to infer this from elapsed
///   time was unsound: SRAM reads resolve a `Cell::Ready` without awaiting, so
///   elapsed time reflects `cycle!` calls, not memory traffic.
/// - `classify_timing_access` is a free function over a register accessor so the
///   mapping is unit-testable; see `mod tests` below.
///
/// Replacing this with an access descriptor owned by `op::Opcode` and consumed by
/// both paths is the real fix; see the module docs on `timing_overlay`.
fn classify_timing_access(
    op: &op::Opcode,
    gp: &dyn Fn(u8) -> u32,
    topk_policy: &dyn Fn() -> Option<(usize, usize)>,
) -> TimingAccess {
    let matrix_tile = *MLEN * *MLEN;
    let vector_tile = *VLEN;
    let matrix_prefetch = *MLEN * *PREFETCH_M_AMOUNT;
    let vector_prefetch = *VLEN * *PREFETCH_V_AMOUNT;
    let vector_store = *VLEN * *STORE_V_AMOUNT;
    let matrix_vector_batch = *MLEN * *BLEN;

    let matrix = |start: u32, len: u32| SramRange::new(SramSpace::Matrix, start, len);
    let vector = |start: u32, len: u32| SramRange::new(SramSpace::Vector, start, len);
    // `MatrixMachine` aligns a vector write-out address down to a whole row before
    // touching vram (`multiple_and_offset(v_addr, mlen)`), so mirror that here
    // rather than using the raw register value.
    let row_base = |addr: u32| addr - (addr % *MLEN);

    match *op {
        op::Opcode::H_PREFETCH_M { rd, .. } => {
            TimingAccess::prefetch(vec![matrix(gp(rd), matrix_prefetch)])
        }
        op::Opcode::H_PREFETCH_V { rd, .. } => {
            TimingAccess::prefetch(vec![vector(gp(rd), vector_prefetch)])
        }
        op::Opcode::H_STORE_V { rd, .. } => {
            // Mirrors the `H_STORE_V` execution arm: `src_addr = gp(rd)`, then
            // `transfer_mx_to_hbm(.., src_addr, *VLEN, *STORE_V_AMOUNT)`. A store is a
            // *read* of that vector region, not a barrier -- modelling it as one used
            // to abandon every pending prefetch, so any program that interleaved
            // stores got essentially no overlap credit.
            TimingAccess::store(vec![vector(gp(rd), vector_store)])
        }

        op::Opcode::M_MM { rs1, rs2 } | op::Opcode::M_TMM { rs1, rs2 } => {
            TimingAccess::compute(vec![
                matrix(gp(rs1), matrix_tile),
                vector(gp(rs2), matrix_vector_batch),
            ])
        }
        op::Opcode::M_BMM { rs1, rs2 } | op::Opcode::M_BTMM { rs1, rs2 } => {
            TimingAccess::compute(vec![
                matrix(gp(rs1), matrix_tile),
                vector(gp(rs2), matrix_tile),
            ])
        }
        op::Opcode::M_MV { rs1, rs2 } | op::Opcode::M_TMV { rs1, rs2 } => {
            TimingAccess::compute(vec![
                matrix(gp(rs1), matrix_tile),
                vector(gp(rs2), vector_tile),
            ])
        }
        op::Opcode::M_BMV { rs1, rs2, rd } | op::Opcode::M_BTMV { rs1, rs2, rd } => {
            TimingAccess::compute(vec![
                matrix(gp(rs1).wrapping_add(gp(rd)), matrix_tile),
                vector(gp(rs2), vector_tile),
            ])
        }

        // `mm_wo` is a read-modify-write: for each of `blen` rows it reads
        // `vec_base + i * mlen * stride_len`, splices the accumulator into a
        // `blen`-wide slot, and writes the row back. The read is a genuine RAW
        // dependency on any prefetch that filled those rows.
        op::Opcode::M_MM_WO { rd, rstride, imm } => {
            let stride_len = if rstride == 0 { 1 } else { gp(rstride) };
            let base = row_base(gp(rd).wrapping_add(imm));
            let span = (*BLEN)
                .saturating_sub(1)
                .saturating_mul(*MLEN)
                .saturating_mul(stride_len)
                .saturating_add(vector_tile);
            TimingAccess::compute(vec![vector(base, span)])
        }
        // `mv_wo` reads exactly the one destination row before splicing.
        op::Opcode::M_MV_WO { rd, imm } => TimingAccess::compute(vec![vector(
            row_base(gp(rd).wrapping_add(imm)),
            vector_tile,
        )]),
        // `bmm_wo` / `bmv_wo` overwrite whole rows and read nothing, so they carry no
        // dependency -- but their compute time can still hide independent prefetches.
        op::Opcode::M_BMM_WO { .. } | op::Opcode::M_BMV_WO { .. } => TimingAccess::compute(vec![]),

        op::Opcode::V_ADD_VV { rs1, rs2, .. }
        | op::Opcode::V_SUB_VV { rs1, rs2, .. }
        | op::Opcode::V_MUL_VV { rs1, rs2, .. } => TimingAccess::compute(vec![
            vector(gp(rs1), vector_tile),
            vector(gp(rs2), vector_tile),
        ]),

        // Writing to fp0 is discarded, and the execution arm returns without touching
        // vram or the clock. Mirrors `do_ops`'s `V_RED_SUM { rd: 0 }` no-op arm.
        op::Opcode::V_RED_SUM { rd: 0, .. } | op::Opcode::V_RED_MAX { rd: 0, .. } => {
            TimingAccess::Other
        }

        // `topk_softmax` walks the logits in VLEN-sized chunks, so it touches
        // `ceil(expert_count / VLEN)` rows -- two of them under the 128-expert policy,
        // not the one row this used to report.
        op::Opcode::V_TOPK { rs1, rmask, .. } => {
            let expert_count: u32 = match rmask {
                0 => 32,
                1 => 128,
                // The rmask=15 escape takes its shape from C_SET_TOPK_REG, so the
                // read extent is only knowable at execution time. Falling back to a
                // literal here would under-report every model wider than 128
                // experts -- DeepSeek-V3 and Kimi K2 are 256 -- and the overlay would
                // credit the missing rows as free hiding capacity.
                15 => topk_policy().map_or(128, |(experts, _)| experts as u32),
                // Unsupported policies panic in the execution arm; classify
                // conservatively rather than duplicating the panic here.
                _ => 128,
            };
            let rows = expert_count.div_ceil(vector_tile.max(1));
            TimingAccess::compute(vec![vector(gp(rs1), rows * vector_tile)])
        }

        op::Opcode::V_ADD_VF { rs1, .. }
        | op::Opcode::V_SUB_VF { rs1, .. }
        | op::Opcode::V_MUL_VF { rs1, .. }
        | op::Opcode::V_MAX_VF { rs1, .. }
        | op::Opcode::V_MIN_VF { rs1, .. }
        | op::Opcode::V_EXP_V { rs1, .. }
        | op::Opcode::V_RECI_V { rs1, .. }
        | op::Opcode::V_RED_SUM { rs1, .. }
        | op::Opcode::V_RED_MAX { rs1, .. }
        | op::Opcode::V_SHFT_V { rs1, .. } => {
            TimingAccess::compute(vec![vector(gp(rs1), vector_tile)])
        }

        op::Opcode::C_BREAK => TimingAccess::Barrier,

        // These neither read matrix/vector SRAM nor depend on a prefetch, so they add
        // no hiding capacity and retire nothing. `S_MAP_V_FP` is the one member that
        // does touch vector SRAM -- it *writes* a VLEN row -- but the model tracks
        // only prefetch-write -> compute-read (RAW), so a write is out of scope; the
        // cost is that its cycles are not available as hiding capacity.
        //
        // Listed exhaustively rather than caught by `_` on purpose: this match
        // hand-mirrors the SRAM addressing of the execution arms, and a wildcard
        // would silently absorb a newly added opcode into "no SRAM access" instead of
        // failing the build until someone classifies it.
        op::Opcode::S_ADD_FP { .. }
        | op::Opcode::S_SUB_FP { .. }
        | op::Opcode::S_MAX_FP { .. }
        | op::Opcode::S_MUL_FP { .. }
        | op::Opcode::S_EXP_FP { .. }
        | op::Opcode::S_RECI_FP { .. }
        | op::Opcode::S_SQRT_FP { .. }
        | op::Opcode::S_LD_FP { .. }
        | op::Opcode::S_ST_FP { .. }
        | op::Opcode::S_MAP_V_FP { .. }
        | op::Opcode::S_ADD_INT { .. }
        | op::Opcode::S_ADDI_INT { .. }
        | op::Opcode::S_SUB_INT { .. }
        | op::Opcode::S_MUL_INT { .. }
        | op::Opcode::S_LUI_INT { .. }
        | op::Opcode::S_LD_INT { .. }
        | op::Opcode::S_ST_INT { .. }
        | op::Opcode::C_SET_ADDR_REG { .. }
        | op::Opcode::C_SET_SCALE_REG { .. }
        | op::Opcode::C_SET_STRIDE_REG { .. }
        | op::Opcode::C_SET_V_MASK_REG { .. }
        | op::Opcode::C_SET_TOPK_REG { .. }
        | op::Opcode::C_LOOP_START { .. }
        | op::Opcode::C_LOOP_END { .. }
        | op::Opcode::Invalid => TimingAccess::Other,
    }
}

fn resource_kind_for_opcode(op: &op::Opcode) -> ResourceKind {
    match op {
        op::Opcode::M_MM { .. }
        | op::Opcode::M_TMM { .. }
        | op::Opcode::M_BMM { .. }
        | op::Opcode::M_BTMM { .. }
        | op::Opcode::M_BMM_WO { .. }
        | op::Opcode::M_MM_WO { .. }
        | op::Opcode::M_MV { .. }
        | op::Opcode::M_TMV { .. }
        | op::Opcode::M_BMV { .. }
        | op::Opcode::M_BTMV { .. }
        | op::Opcode::M_MV_WO { .. }
        | op::Opcode::M_BMV_WO { .. } => ResourceKind::Matrix,

        op::Opcode::V_ADD_VV { .. }
        | op::Opcode::V_ADD_VF { .. }
        | op::Opcode::V_SUB_VV { .. }
        | op::Opcode::V_SUB_VF { .. }
        | op::Opcode::V_MUL_VV { .. }
        | op::Opcode::V_MUL_VF { .. }
        | op::Opcode::V_MAX_VF { .. }
        | op::Opcode::V_MIN_VF { .. }
        | op::Opcode::V_TOPK { .. }
        | op::Opcode::V_EXP_V { .. }
        | op::Opcode::V_RECI_V { .. }
        | op::Opcode::V_RED_SUM { .. }
        | op::Opcode::V_RED_MAX { .. }
        | op::Opcode::V_SHFT_V { .. } => ResourceKind::Vector,

        op::Opcode::S_ADD_FP { .. }
        | op::Opcode::S_SUB_FP { .. }
        | op::Opcode::S_MAX_FP { .. }
        | op::Opcode::S_MUL_FP { .. }
        | op::Opcode::S_EXP_FP { .. }
        | op::Opcode::S_RECI_FP { .. }
        | op::Opcode::S_SQRT_FP { .. }
        | op::Opcode::S_LD_FP { .. }
        | op::Opcode::S_ST_FP { .. }
        | op::Opcode::S_MAP_V_FP { .. }
        | op::Opcode::S_ADD_INT { .. }
        | op::Opcode::S_ADDI_INT { .. }
        | op::Opcode::S_SUB_INT { .. }
        | op::Opcode::S_MUL_INT { .. }
        | op::Opcode::S_LUI_INT { .. }
        | op::Opcode::S_LD_INT { .. }
        | op::Opcode::S_ST_INT { .. }
        | op::Opcode::C_SET_ADDR_REG { .. }
        | op::Opcode::C_SET_SCALE_REG { .. }
        | op::Opcode::C_SET_STRIDE_REG { .. }
        | op::Opcode::C_SET_V_MASK_REG { .. }
        | op::Opcode::C_SET_TOPK_REG { .. }
        | op::Opcode::C_LOOP_START { .. }
        | op::Opcode::C_LOOP_END { .. }
        | op::Opcode::C_BREAK => ResourceKind::Scalar,

        op::Opcode::H_PREFETCH_M { .. }
        | op::Opcode::H_PREFETCH_V { .. }
        | op::Opcode::H_STORE_V { .. } => ResourceKind::Dma,

        op::Opcode::Invalid => ResourceKind::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::VectorPrecision;
    use crate::timing_overlay::{SramSpace, TimingAccess};

    /// Register file stand-in: `gp(n)` returns `n as u32 * 1024`, so every operand
    /// lands at a distinct, easily-recognised address.
    fn gp_stub(reg: u8) -> u32 {
        reg as u32 * 1024
    }

    fn classify(op: op::Opcode) -> TimingAccess {
        classify_timing_access(&op, &gp_stub, &|| None)
    }

    /// `classify` with a `C_SET_TOPK_REG` policy in effect, for the `rmask=15` arm.
    fn classify_with_topk(op: op::Opcode, policy: (usize, usize)) -> TimingAccess {
        classify_timing_access(&op, &gp_stub, &|| Some(policy))
    }

    fn read_ranges(access: &TimingAccess) -> Vec<(SramSpace, u32, u32)> {
        match access {
            TimingAccess::Compute { read_ranges } | TimingAccess::Store { read_ranges } => {
                read_ranges
                    .iter()
                    .map(|r| (r.space, r.start, r.len))
                    .collect()
            }
            other => panic!("expected an access with read ranges, got {other:?}"),
        }
    }

    fn write_ranges(access: &TimingAccess) -> Vec<(SramSpace, u32, u32)> {
        match access {
            TimingAccess::Prefetch { write_ranges } => write_ranges
                .iter()
                .map(|r| (r.space, r.start, r.len))
                .collect(),
            other => panic!("expected a prefetch, got {other:?}"),
        }
    }

    #[test]
    fn store_reads_the_region_it_drains_rather_than_acting_as_a_barrier() {
        // Mirrors `transfer_mx_to_hbm(.., src_addr = gp(rd), *VLEN, *STORE_V_AMOUNT)`.
        let access = classify(op::Opcode::H_STORE_V {
            rd: 3,
            rs1: 0,
            rs2: 0,
            rstride: 0,
            precision: VectorPrecision::Activation,
        });
        assert!(matches!(access, TimingAccess::Store { .. }));
        assert_eq!(
            read_ranges(&access),
            vec![(SramSpace::Vector, gp_stub(3), *VLEN * *STORE_V_AMOUNT)]
        );
    }

    #[test]
    fn prefetch_write_extents_match_the_dma_transfer_sizes() {
        assert_eq!(
            write_ranges(&classify(op::Opcode::H_PREFETCH_V {
                rd: 2,
                rs1: 0,
                rs2: 0,
                rstride: 0,
                precision: VectorPrecision::Activation,
            })),
            vec![(SramSpace::Vector, gp_stub(2), *VLEN * *PREFETCH_V_AMOUNT)]
        );
        assert_eq!(
            write_ranges(&classify(op::Opcode::H_PREFETCH_M {
                rd: 2,
                rs1: 0,
                rs2: 0,
                rstride: 0,
                precision: op::MatrixPrecision::Weights,
            })),
            vec![(SramSpace::Matrix, gp_stub(2), *MLEN * *PREFETCH_M_AMOUNT)]
        );
    }

    #[test]
    fn vector_write_out_ops_read_their_destination_row() {
        // `mv_wo` reads the destination row before splicing the accumulator in, so
        // it is a genuine RAW dependency -- it used to report an empty read set.
        let access = classify(op::Opcode::M_MV_WO { rd: 2, imm: 0 });
        assert_eq!(
            read_ranges(&access),
            vec![(SramSpace::Vector, gp_stub(2), *VLEN)]
        );

        // `mm_wo` does the same for each of `blen` rows at `mlen * stride_len`.
        let access = classify(op::Opcode::M_MM_WO {
            rd: 2,
            rstride: 0,
            imm: 0,
        });
        let expected_span = (*BLEN - 1) * *MLEN + *VLEN;
        assert_eq!(
            read_ranges(&access),
            vec![(SramSpace::Vector, gp_stub(2), expected_span)]
        );
    }

    #[test]
    fn batched_write_out_ops_read_nothing_but_still_provide_hiding_capacity() {
        // `bmm_wo` / `bmv_wo` overwrite whole rows, so an empty read set is correct
        // here -- but they must stay `Compute`, not `Other`, to contribute time.
        for op in [
            op::Opcode::M_BMM_WO { rd: 2, imm: 0 },
            op::Opcode::M_BMV_WO { rd: 2, imm: 0 },
        ] {
            let access = classify(op);
            assert!(matches!(access, TimingAccess::Compute { .. }));
            assert!(read_ranges(&access).is_empty());
        }
    }

    #[test]
    fn topk_reads_every_row_the_expert_policy_spans() {
        // rmask=1 selects the 128-expert policy; `topk_softmax` walks the logits in
        // VLEN-sized chunks, so it touches ceil(128 / VLEN) rows.
        let access = classify(op::Opcode::V_TOPK {
            rd: 1,
            rs1: 2,
            rs2: 0,
            rmask: 1,
        });
        let expected = 128u32.div_ceil(*VLEN) * *VLEN;
        assert_eq!(
            read_ranges(&access),
            vec![(SramSpace::Vector, gp_stub(2), expected)]
        );
        assert!(expected >= 128, "128 experts must span at least 128 units");

        // rmask=0 is the 32-expert policy: a single (partial) row.
        let access = classify(op::Opcode::V_TOPK {
            rd: 1,
            rs1: 2,
            rs2: 0,
            rmask: 0,
        });
        assert_eq!(
            read_ranges(&access),
            vec![(SramSpace::Vector, gp_stub(2), 32u32.div_ceil(*VLEN) * *VLEN)]
        );
    }

    #[test]
    fn topk_escape_policy_takes_its_read_extent_from_the_control_register() {
        // DeepSeek-V3 / Kimi K2: 256 experts, top-8. Before the escape existed this
        // shape was unencodable; if the arm fell back to the 128 literal the overlay
        // would model half the logit rows and credit the rest as free.
        let deepseek = classify_with_topk(
            op::Opcode::V_TOPK {
                rd: 1,
                rs1: 2,
                rs2: 0,
                rmask: 15,
            },
            (256, 8),
        );
        assert_eq!(
            read_ranges(&deepseek),
            vec![(
                SramSpace::Vector,
                gp_stub(2),
                256u32.div_ceil(*VLEN) * *VLEN
            )]
        );

        // Llama-4 Scout sits at the other end: 16 experts, top-1.
        let scout = classify_with_topk(
            op::Opcode::V_TOPK {
                rd: 1,
                rs1: 2,
                rs2: 0,
                rmask: 15,
            },
            (16, 1),
        );
        assert_eq!(
            read_ranges(&scout),
            vec![(SramSpace::Vector, gp_stub(2), 16u32.div_ceil(*VLEN) * *VLEN)]
        );
    }

    #[test]
    fn fp0_reductions_are_no_ops_and_must_not_retire_prefetches() {
        // `do_ops` returns immediately for these without touching vram; classifying
        // them as Compute would let a no-op retire a prefetch it never read.
        for op in [
            op::Opcode::V_RED_SUM {
                rd: 0,
                rs1: 2,
                rmask: 0,
            },
            op::Opcode::V_RED_MAX {
                rd: 0,
                rs1: 2,
                rmask: 0,
            },
        ] {
            assert!(
                matches!(
                    classify_timing_access(&op, &gp_stub, &|| None),
                    TimingAccess::Other
                ),
                "{op:?} is a no-op in do_ops and must not participate"
            );
        }

        // A non-zero rd is a real reduction and does read vram.
        let access = classify(op::Opcode::V_RED_SUM {
            rd: 1,
            rs1: 2,
            rmask: 0,
        });
        assert_eq!(
            read_ranges(&access),
            vec![(SramSpace::Vector, gp_stub(2), *VLEN)]
        );
    }

    #[test]
    fn batched_matrix_vector_mirrors_the_rs1_plus_rd_addressing() {
        // `MatrixMachine::bmv` is called with `read_gp(rs1) + read_gp(rd)`.
        let access = classify(op::Opcode::M_BMV {
            rs1: 1,
            rs2: 2,
            rd: 3,
        });
        assert_eq!(
            read_ranges(&access),
            vec![
                (SramSpace::Matrix, gp_stub(1) + gp_stub(3), *MLEN * *MLEN),
                (SramSpace::Vector, gp_stub(2), *VLEN),
            ]
        );
    }

    #[test]
    fn control_flow_barriers_and_scalar_ops_stay_out_of_the_model() {
        assert!(matches!(
            classify(op::Opcode::C_BREAK),
            TimingAccess::Barrier
        ));
        assert!(matches!(
            classify(op::Opcode::S_ADDI_INT {
                rd: 1,
                rs1: 0,
                imm: 4
            }),
            TimingAccess::Other
        ));
    }
}
