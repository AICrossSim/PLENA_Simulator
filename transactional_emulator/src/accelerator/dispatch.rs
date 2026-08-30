//! Opcode execution for [`Accelerator`].
//!
//! The public accelerator facade stays in `mod.rs`; this module owns the ISA
//! match and dispatch-only helpers.

use half::bf16;
use quantize::MxDataType;

use crate::runtime_config::PERIOD;
use crate::runtime_config::{
    HLEN, MATRIX_KV_TYPE, MATRIX_WEIGHT_TYPE, MLEN, PREFETCH_M_AMOUNT, PREFETCH_V_AMOUNT,
    SCALAR_FP_BASIC_CYCLES, SCALAR_FP_EXP_CYCLES, SCALAR_FP_RECI_CYCLES, SCALAR_FP_SQRT_CYCLES,
    SCALAR_INT_BASIC_CYCLES, STORE_V_AMOUNT, VECTOR_ACTIVATION_TYPE, VECTOR_KV_TYPE, VLEN,
};
use crate::stage_profile::{ResourceKind, StageProfiler};
use crate::vector_machine::VectorOperandViews;
use crate::{cycle, dma, op, timing};
use runtime::{Executor, Instant};

use super::Accelerator;
use super::access::{self, OpAccess};
use super::loop_state::LoopDecision;
use super::lstream::{ConfigField, StreamTarget};
use super::scoreboard::{DmaKind, Scoreboard};

/// How `do_ops` charges time.
///
/// `Serial` is the historical model: every opcode sleeps for its full latency
/// on the dispatch task, so total cycles are the sum of all per-instruction
/// latencies. `Scoreboard` keeps functional execution serial but models
/// pipelined issue via the analytic scoreboard: dispatch sleeps only to each
/// op's modeled issue instant and the final clock is the scoreboard's
/// max-finish.
pub(crate) enum TimingDriver<'a> {
    Serial,
    Scoreboard { scoreboard: &'a mut Scoreboard },
}

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
        mut timing: TimingDriver<'_>,
    ) {
        let mut pc: usize = 0; // Program counter

        while pc < ops.len() {
            let executed_pc = pc;
            let op = &ops[pc];

            self.loop_state.record_instruction();
            tracing::debug!(pc, ?op, "execute op");

            // Affine streams change operand addressing, not arithmetic.  Take
            // the access snapshot before execution so every bound target is
            // hydrated and advanced exactly once even when the same register
            // is both source and destination (V_FMA_VF).
            let stream_access = opcode_uses_lstream(op).then(|| self.op_access_for_opcode(op));
            if let Some(access) = &stream_access {
                self.hydrate_lstream_fp_operands(access);
                self.validate_affine_opcode(op, access, pc);
            }

            // Scoreboard mode: resolve hazards, then sleep to the modeled
            // issue instant so everything the arm does (in particular HBM
            // traffic) happens at model-consistent virtual times.
            let mut issued: Option<(Instant, OpAccess)> = None;
            if let TimingDriver::Scoreboard { scoreboard } = &mut timing {
                let access = stream_access
                    .clone()
                    .unwrap_or_else(|| self.op_access_for_opcode(op));
                // Drain policy: HBM-side ordering is not range-tracked, so
                // H_* ops order conservatively against in-flight DMAs.
                // - Barrier / H_STORE_V: everything (a store overwrites HBM an
                //   in-flight prefetch may read, and earlier stores' HBM
                //   writes may overlap its own).
                // - H_PREFETCH_*: all outstanding stores (HBM RAW) plus
                //   prefetches overlapping its SRAM destination (WAW).
                // - Everything else: SRAM overlaps only.
                let pending = if access.barrier || matches!(op, op::Opcode::H_STORE_V { .. }) {
                    scoreboard.take_all_dma()
                } else if matches!(
                    op,
                    op::Opcode::H_PREFETCH_M { .. } | op::Opcode::H_PREFETCH_V { .. }
                ) {
                    scoreboard.take_dma_for_prefetch(&access)
                } else {
                    scoreboard.take_overlapping_dma(&access)
                };
                for dma in pending {
                    let wait_start = Executor::current().now();
                    match dma.done.await {
                        Ok(completed_at) => scoreboard.retire_dma(&dma.writes, completed_at),
                        Err(_) => tracing::error!(
                            pc,
                            "pending DMA completer dropped its channel; timing may be optimistic"
                        ),
                    }
                    scoreboard.note_dma_wait(Executor::current().now() - wait_start);
                }
                let now = Executor::current().now();
                let issue = scoreboard.issue_bound(&access, now);
                if issue > now {
                    Executor::current().resolve_at(issue).await;
                }
                debug_assert_eq!(
                    timing::pending_charge(),
                    0,
                    "latency charge leaked across an instruction boundary"
                );
                // DMA goes asynchronous in scoreboard mode: a prefetch's fill
                // is spawned (destination cells parked Pending), a store's
                // vram rows are snapshotted here and its HBM writes spawned;
                // either way the op only occupies the DMA issue slot for one
                // cycle and its completion is carried by the registered
                // PendingDma. In serialize (serial-equivalence validation)
                // mode DMA stays inline like every other op, so the run
                // reproduces serial timing exactly.
                if !scoreboard.is_serialize() && self.issue_async_dma(op, &access, scoreboard).await
                {
                    let finish = scoreboard.commit(&access, issue, *PERIOD);
                    scoreboard.trace_op(executed_pc, op, access.unit, issue, finish);
                    if let Some(profiler) = stage_profiler.as_deref_mut() {
                        let elapsed_picos = (finish - issue).as_picos();
                        profiler.record(
                            executed_pc,
                            elapsed_picos as f64 / 1_000_000_000_000.0,
                            elapsed_picos,
                            ResourceKind::Dma,
                            // The transfer's HBM bytes land while the fill is
                            // in flight; they show up in the run-level HBM
                            // statistics, not in this op's delta.
                            0,
                            0,
                        );
                    }
                    pc += 1;
                    continue;
                }
                issued = Some((issue, access));
            }

            // Serial mode: snapshot the clock for the profiler epilogue.
            let profile_start_instant = if issued.is_none() && stage_profiler.is_some() {
                Some(Executor::current().now())
            } else {
                None
            };
            let profile_start_hbm = if stage_profiler.is_some() {
                self.hbm.statistics()
            } else {
                None
            };

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
                        .mm_wo(
                            self.reg_file.read_gp(*rd) + *imm,
                            stride_len,
                            self.reg_file.lstream_gp_affine_view(*rd),
                        )
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
                            VectorOperandViews {
                                destination: self.reg_file.lstream_gp_affine_view(*rd),
                                source: self.reg_file.lstream_gp_affine_view(*rs1),
                            },
                        )
                        .await;
                }
                // The only V-type op that reads `rd`: `V[rd] += V[rs1] * fp[rs2]`.
                // `rd` is the destination *and* an operand, so it goes first --
                // swapping the first two arguments computes `V[rs1] += V[rd]*f`,
                // which is finite, plausible and wrong.
                op::Opcode::V_FMA_VF {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .fma_scalar(
                            self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_fp(*rs2).into(),
                            *rmask,
                            mask,
                            VectorOperandViews {
                                destination: self.reg_file.lstream_gp_affine_view(*rd),
                                source: self.reg_file.lstream_gp_affine_view(*rs1),
                            },
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
                op::Opcode::V_SOFTPLUS_V { rd, rs1, rmask } => {
                    let mask = self.resolve_v_mask(*rmask);
                    self.v_machine
                        .softplus(
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
                            self.reg_file.lstream_gp_affine_view(*rs1),
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
                op::Opcode::S_MAP_FP_V { rd, rs1, imm } => {
                    // Mirror of S_MAP_V_FP: VRAM row -> VLEN consecutive FP_MEM slots.
                    // Note the operand roles are the mirror image too: `rs1` is the
                    // VRAM source row and `rd` is the FP_MEM base, so that both
                    // instructions keep "rd names the destination memory".
                    let values = self
                        .v_machine
                        .vector_read_fp(self.reg_file.read_gp(*rs1))
                        .await;
                    let start_idx = (self.reg_file.read_gp(*rd) + *imm) as usize;
                    self.scalar_sram.write_fp_window(start_idx, &values);
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
                op::Opcode::L_STREAM_CFG {
                    value,
                    target,
                    slot,
                    field,
                } => {
                    let field = ConfigField::try_from(*field).unwrap_or_else(|error| {
                        tracing::error!(pc, %error, "invalid L_STREAM_CFG field");
                        panic!("{error} at pc {pc}");
                    });
                    let value = self.reg_file.read_gp(*value);
                    self.reg_file
                        .configure_lstream(value, *target, *slot, field)
                        .unwrap_or_else(|error| {
                            tracing::error!(pc, %error, "invalid L_STREAM_CFG value");
                            panic!("{error} at pc {pc}");
                        });
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

            if let Some(access) = &stream_access {
                self.advance_lstream_operands(access);
            }

            // Handle loop jumps
            if let Some(target_pc) = jump_pc {
                pc = target_pc;
            } else {
                pc += 1;
            }

            // Epilogue: commit/charge modeled time and feed the profilers.
            let mut profiled_elapsed_picos: Option<u64> = None;
            match &mut timing {
                TimingDriver::Scoreboard { scoreboard } => {
                    let (issue, access) = issued.take().expect("scoreboard pre-issue ran");
                    let charged = timing::take_charged();
                    let after = Executor::current().now();
                    let is_dma_op = matches!(
                        op,
                        op::Opcode::H_PREFETCH_M { .. }
                            | op::Opcode::H_PREFETCH_V { .. }
                            | op::Opcode::H_STORE_V { .. }
                    );
                    if after > issue && !is_dma_op {
                        tracing::warn!(
                            pc = executed_pc,
                            ?op,
                            "unclassified dependency stalled dispatch past its modeled issue"
                        );
                    }
                    // Inline HBM transfers advance the clock themselves
                    // (`after - issue`); everything else charges through
                    // `timing::charge_cycles`.
                    let latency = (after - issue) + *PERIOD * charged;
                    let finish = scoreboard.commit(&access, issue, latency);
                    scoreboard.trace_op(executed_pc, op, access.unit, issue, finish);
                    profiled_elapsed_picos = Some((finish - issue).as_picos());
                }
                TimingDriver::Serial => {
                    if let Some(start_instant) = profile_start_instant {
                        profiled_elapsed_picos =
                            Some((Executor::current().now() - start_instant).as_picos());
                    }
                }
            }
            if let (Some(elapsed_picos), Some(profiler)) =
                (profiled_elapsed_picos, stage_profiler.as_deref_mut())
            {
                let elapsed_secs = elapsed_picos as f64 / 1_000_000_000_000.0;
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

        // End of program in scoreboard mode: land every in-flight DMA, then
        // advance the clock to the modeled finish so `executor.now()` (the
        // reported total) reflects pipelined completion.
        if let TimingDriver::Scoreboard { scoreboard } = &mut timing {
            for dma in scoreboard.take_all_dma() {
                match dma.done.await {
                    Ok(completed_at) => scoreboard.retire_dma(&dma.writes, completed_at),
                    Err(_) => tracing::error!(
                        "pending DMA completer dropped its channel at end of program"
                    ),
                }
            }
            let now = Executor::current().now();
            let finish = scoreboard.max_finish().max(now);
            if finish > now {
                Executor::current().resolve_at(finish).await;
            }
        }
    }

    fn op_access_for_opcode(&self, op: &op::Opcode) -> OpAccess {
        access::op_access(op, &|reg| self.reg_file.read_gp(reg), &|| {
            self.reg_file.topk_policy()
        })
    }

    fn hydrate_lstream_fp_operands(&mut self, access: &OpAccess) {
        let mut registers = std::collections::BTreeSet::new();
        for resource in &access.reads {
            if let access::Resource::Fp(register) = resource {
                registers.insert(*register);
            }
        }
        for register in registers {
            if let Some(address) = self.reg_file.lstream_fp_address(register) {
                let value = self.scalar_sram.read_fp(address as usize);
                self.reg_file.write_fp(register, value);
            }
        }
    }

    fn validate_affine_opcode(&self, op: &op::Opcode, access: &OpAccess, pc: usize) {
        let affine_targets: Vec<_> = access
            .reads
            .iter()
            .filter_map(|resource| match resource {
                access::Resource::Gp(register)
                    if self.reg_file.lstream_gp_affine_view(*register).is_some() =>
                {
                    Some(*register)
                }
                _ => None,
            })
            .collect();
        if affine_targets.is_empty() {
            return;
        }
        if matches!(
            op,
            op::Opcode::M_MM_WO { .. }
                | op::Opcode::V_MUL_VF { .. }
                | op::Opcode::V_FMA_VF { .. }
                | op::Opcode::V_RED_SUM { .. }
        ) {
            return;
        }
        panic!(
            "affine L-stream targets {affine_targets:?} on unsupported opcode {op:?} at pc {pc}; \
             use the identity stream or the static fallback"
        );
    }

    fn advance_lstream_operands(&mut self, access: &OpAccess) {
        let targets = access.reads.iter().filter_map(|resource| match resource {
            access::Resource::Gp(register) => Some(StreamTarget::Gp(*register)),
            access::Resource::Fp(register) => Some(StreamTarget::Fp(*register)),
            _ => None,
        });
        self.reg_file.advance_lstream_targets(targets);
    }

    /// Scoreboard-mode asynchronous DMA issue: launch the HBM traffic at the
    /// current (modeled issue) instant and spawn a completer that reports the
    /// real completion instant. For a prefetch, the destination SRAM cells
    /// are parked as `Cell::Pending` first; for a store, the source vram rows
    /// are snapshotted here (functional WAR safety) before the HBM writes are
    /// spawned. Returns `false` for any other opcode, leaving it to the
    /// normal inline path.
    ///
    /// The `Cell::Pending` parking is the functional safety net: even a
    /// dependency the access descriptor misses still blocks on first read
    /// instead of observing stale data.
    async fn issue_async_dma(
        &mut self,
        op: &op::Opcode,
        access: &OpAccess,
        scoreboard: &mut Scoreboard,
    ) -> bool {
        let (kind, done_rx) = match op {
            op::Opcode::H_PREFETCH_M {
                rd,
                rs1,
                rs2,
                rstride,
                precision,
            } => {
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
                let dest = self.reg_file.read_gp(*rd);
                // The transfer produces MLEN * PREFETCH_M_AMOUNT elements.
                let cells = (*MLEN * *PREFETCH_M_AMOUNT).div_ceil(*MLEN * *MLEN);
                let senders = self.m_machine.mram.mark_pending_tiles(dest, cells).await;
                let mram = self.m_machine.mram.clone();
                let (done_tx, done_rx) = tokio::sync::oneshot::channel();
                Executor::current().spawn(async move {
                    mram.fill_pending(senders, xfer).await;
                    let _ = done_tx.send(Executor::current().now());
                });
                (DmaKind::Prefetch, done_rx)
            }
            op::Opcode::H_PREFETCH_V {
                rd,
                rs1,
                rs2,
                rstride,
                precision,
            } => {
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
                let senders = self
                    .v_machine
                    .vram
                    .mark_pending_rows(dest, *PREFETCH_V_AMOUNT)
                    .await;
                let vram = self.v_machine.vram.clone();
                let (done_tx, done_rx) = tokio::sync::oneshot::channel();
                Executor::current().spawn(async move {
                    vram.fill_pending(senders, xfer).await;
                    let _ = done_tx.send(Executor::current().now());
                });
                (DmaKind::Prefetch, done_rx)
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
                // Snapshot the source rows at issue: later instructions may
                // overwrite them while the HBM writes are still in flight.
                let rows =
                    dma::snapshot_vram_rows(&self.v_machine.vram, src_addr, *VLEN, *STORE_V_AMOUNT)
                        .await;
                let hbm = self.hbm.clone();
                let (done_tx, done_rx) = tokio::sync::oneshot::channel();
                Executor::current().spawn(async move {
                    dma::store_rows_to_hbm(&hbm, region, rows, *VLEN).await;
                    let _ = done_tx.send(Executor::current().now());
                });
                (DmaKind::Store, done_rx)
            }
            _ => return false,
        };
        let writes: Vec<access::SramRange> = access
            .writes
            .iter()
            .filter_map(|r| match r {
                access::Resource::Sram(range) => Some(*range),
                _ => None,
            })
            .collect();
        scoreboard.register_dma(kind, writes, done_rx);
        true
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
        | op::Opcode::V_FMA_VF { .. }
        | op::Opcode::V_SOFTPLUS_V { .. }
        // S_MAP_FP_V drives the vector SRAM read port for a whole VLEN row, so it
        // contends with the vector unit even though its destination is FP_MEM. Its
        // mirror S_MAP_V_FP is classified Scalar for the same reason inverted.
        | op::Opcode::S_MAP_FP_V { .. }
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
        | op::Opcode::L_STREAM_CFG { .. }
        | op::Opcode::C_LOOP_START { .. }
        | op::Opcode::C_LOOP_END { .. }
        | op::Opcode::C_BREAK => ResourceKind::Scalar,

        op::Opcode::H_PREFETCH_M { .. }
        | op::Opcode::H_PREFETCH_V { .. }
        | op::Opcode::H_STORE_V { .. } => ResourceKind::Dma,

        op::Opcode::Invalid => ResourceKind::Other,
    }
}

/// Operations whose register operands may be backed by an affine stream.
///
/// Control, scalar-address arithmetic and DMA stay explicit.  This prevents a
/// stray S_ADDI on a bound pointer from advancing it twice and keeps stream
/// semantics orthogonal to HBM transfers.
fn opcode_uses_lstream(op: &op::Opcode) -> bool {
    matches!(
        resource_kind_for_opcode(op),
        ResourceKind::Matrix | ResourceKind::Vector
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `V_FMA_VF` execution arm must pass `rd` as the destination.
    ///
    /// Nothing in this crate executes a *decoded* instruction -- every
    /// `fma_scalar` test calls the method directly -- so swapping the first two
    /// arguments in the dispatch arm, which turns the instruction into
    /// `V[rs1] += V[rd] * f`, left the whole suite green. Every other opcode has
    /// the same gap; closing it properly means building an Accelerator in a test,
    /// which needs Ramulator and a MatrixMachine.
    ///
    /// Parsing our own source is crude. `stage_profile.rs` already does it for
    /// the same reason: it is the only way to tie two things together without a
    /// macro, and a crude guard beats none on an argument order that is a
    /// silently different instruction when wrong.
    #[test]
    fn the_fma_dispatch_arm_passes_rd_as_the_destination() {
        let source = include_str!("dispatch.rs");
        let arm = source
            .find("op::Opcode::V_FMA_VF {")
            .expect("the V_FMA_VF execution arm must exist");
        let body = &source[arm..arm + 600];
        let call = body
            .find(".fma_scalar(")
            .expect("the arm must call fma_scalar");
        let args = &body[call..];
        let rd = args.find("read_gp(*rd)").expect("rd must be passed");
        let rs1 = args.find("read_gp(*rs1)").expect("rs1 must be passed");
        assert!(
            rd < rs1,
            "fma_scalar takes (vd, vs1): rd must come first, or the instruction \
             computes V[rs1] += V[rd] * f instead"
        );
        assert!(
            args.find("read_fp(*rs2)").is_some_and(|f| f > rs1),
            "rs2 is the FP scalar and must come third"
        );
    }

    /// `V_FMA_VF` belongs to the vector unit, not the scalar one.
    #[test]
    fn fma_is_billed_to_the_vector_unit() {
        assert_eq!(
            resource_kind_for_opcode(&op::Opcode::V_FMA_VF {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rmask: 0,
            }),
            ResourceKind::Vector
        );
    }
}
