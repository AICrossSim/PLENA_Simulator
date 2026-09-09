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
use crate::{cycle, dma, op};

use super::Accelerator;
use super::loop_state::LoopDecision;
use super::qwen3_moe;

impl Accelerator {
    /// Resolve the V_* opcode mask.
    ///
    /// When `rmask == 0`, the opcode operates on every HLEN-wide segment.
    /// Otherwise the mask register is used directly.
    fn resolve_v_mask(&self, rmask: u8) -> u32 {
        if rmask == 0 {
            let segments = *VLEN / *HLEN;
            if segments >= u32::BITS {
                u32::MAX
            } else {
                (1_u32 << segments) - 1
            }
        } else {
            self.reg_file.v_mask()
        }
    }

    fn mx_region(&self, dtype: MxDataType, addr: u64, offset: u32, rstride: u8) -> dma::MxRegion {
        assert!(rstride <= 1, "HBM rstride must be 0 or 1");
        let scale = match dtype {
            MxDataType::Plain(_) => 0,
            MxDataType::Mx { .. } => {
                let ratio = dtype.element_scale_ratio();
                assert!(
                    offset.is_multiple_of(ratio),
                    "MX HBM byte offset must align with its scale stream"
                );
                offset / ratio
            }
        };

        dma::MxRegion {
            hbm_type: dtype,
            index: addr
                .checked_add(offset as u64)
                .expect("HBM element address overflow"),
            // Scales are stored AFTER elements, so scale_index =
            // element_index + scale_reg + scale, where scale_reg is the offset
            // from element start to scale start.
            scale_index: addr
                .checked_add(self.reg_file.scale() as u64)
                .and_then(|value| value.checked_add(scale as u64))
                .expect("HBM scale address overflow"),
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

            // Snapshot clock + HBM counters so the post-dispatch record()
            // captures exactly this instruction's time and traffic.
            let mut op_mark = self.op_stats.as_ref().map(|r| r.begin());
            let mut hbm_issue_read_bytes = 0;
            let mut hbm_issue_written_bytes = 0;

            // Structural hazard stalls (mirrors RTL pipeline_control): matrix
            // ops consume both SRAMs; vector ops, S_MAP_V_FP and H_STORE_V
            // consume Vector SRAM. Consumers wait here for any in-flight
            // background prefetch, so the DMA latency is charged to the first
            // dependent instruction, not the prefetch itself.
            {
                let name = op.mnemonic();
                if name.starts_with("M_") {
                    self.drain_m_load().await;
                    self.drain_v_load().await;
                } else if name.starts_with("V_") || name == "S_MAP_V_FP" || name == "H_STORE_V" {
                    self.drain_v_load().await;
                }
            }

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
                op::Opcode::M_BMM { rd, rs1, rs2 } => {
                    self.m_machine
                        .bmm(
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_gp(*rs2),
                            *rd as u32,
                        )
                        .await;
                }
                op::Opcode::M_BTMM { rd, rs1, rs2 } => {
                    self.m_machine
                        .btmm(
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_gp(*rs2),
                            *rd as u32,
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
                        )
                        .await;
                }
                op::Opcode::M_BTMV { rs1, rs2, rd } => {
                    self.m_machine
                        .btmv(
                            self.reg_file.read_gp(*rs1) + self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs2),
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
                op::Opcode::V_ROUTER_LINEAR_BF16 {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                } => {
                    let hidden = match *rmask {
                        0 => 64,
                        1 => 2048,
                        other => {
                            tracing::error!(
                                pc,
                                policy = other,
                                "unsupported V_ROUTER_LINEAR_BF16 policy"
                            );
                            panic!(
                                "unsupported V_ROUTER_LINEAR_BF16 policy {other} at pc {pc}; expected 0=hidden64 or 1=hidden2048"
                            );
                        }
                    };
                    self.v_machine
                        .router_linear_bf16(
                            self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_gp(*rs2),
                            hidden,
                            128,
                            *BLEN as usize,
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
                        other => {
                            tracing::error!(pc, rmask = other, "unsupported V_TOPK policy");
                            panic!(
                                "unsupported V_TOPK policy {other} at pc {pc}; expected 0=32/top4 or 1=128/top8"
                            );
                        }
                    };
                    let route_base = self.reg_file.read_gp(*rd) as usize;
                    let int_base = self.reg_file.read_gp(*rs2) as usize;
                    let (indices, weights) = self
                        .v_machine
                        .topk_softmax(self.reg_file.read_gp(*rs1), expert_count, topk)
                        .await;
                    for (offset, (index, weight)) in indices.iter().zip(weights.iter()).enumerate()
                    {
                        self.scalar_sram.write_int(int_base + offset, *index);
                        self.scalar_sram
                            .write_route_f32(route_base + offset, *weight);
                    }
                }
                op::Opcode::V_MUL_ROUTE_F32 {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                } => {
                    let mask = self.resolve_v_mask(*rmask);
                    let route_score = self
                        .scalar_sram
                        .read_route_f32(self.reg_file.read_gp(*rs2) as usize);
                    assert!(
                        route_score.is_finite(),
                        "V_MUL_ROUTE_F32 route score must be finite"
                    );
                    self.v_machine
                        .mul_route_f32(
                            self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs1),
                            route_score,
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_QWEN3_EXPERT_COMBINE_BF16 {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                } => {
                    let descriptor_base = self.reg_file.read_hbm(*rmask);
                    let descriptor = qwen3_moe::read_descriptor(&self.hbm, descriptor_base).await;
                    hbm_issue_read_bytes = qwen3_moe::descriptor_physical_read_bytes();
                    let pair_base = self.reg_file.read_gp(*rs2) as usize;
                    let mut pairs = (0..8)
                        .map(|offset| {
                            (
                                self.scalar_sram.read_int(pair_base + offset),
                                self.scalar_sram.read_route_f32(pair_base + offset),
                            )
                        })
                        .collect::<Vec<_>>();
                    assert!(
                        pairs.iter().all(|(expert, score)| {
                            *expert < 128 && score.is_finite() && *score >= 0.0
                        }),
                        "V_QWEN3_EXPERT_COMBINE_BF16 route pair is invalid"
                    );
                    pairs.sort_by_key(|(expert, _)| *expert);
                    assert!(
                        pairs.windows(2).all(|pair| pair[0].0 < pair[1].0),
                        "V_QWEN3_EXPERT_COMBINE_BF16 expert IDs must be unique"
                    );
                    let route_sum = pairs.iter().map(|(_, score)| score).sum::<f32>();
                    assert!(
                        (route_sum - 1.0).abs() <= 1.0e-5,
                        "V_QWEN3_EXPERT_COMBINE_BF16 route scores are not normalized"
                    );
                    for (ordinal, (expert_id, route_score)) in pairs.into_iter().enumerate() {
                        hbm_issue_read_bytes = hbm_issue_read_bytes
                            .checked_add(descriptor.expert_physical_read_bytes())
                            .expect("Qwen3 expert issue-origin traffic overflow");
                        let (gate_up, down) =
                            qwen3_moe::read_expert(&self.hbm, &descriptor, expert_id).await;
                        self.v_machine
                            .qwen3_expert_accumulate_bf16(
                                self.reg_file.read_gp(*rd),
                                self.reg_file.read_gp(*rs1),
                                descriptor.hidden,
                                descriptor.intermediate,
                                *BLEN as usize,
                                expert_id,
                                route_score,
                                gate_up,
                                down,
                                ordinal == 0,
                            )
                            .await;
                    }
                }
                op::Opcode::V_QWEN3_RMSNORM_BF16 {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                } => {
                    let hidden = match *rmask {
                        0 => 64,
                        1 => 2048,
                        other => {
                            tracing::error!(
                                pc,
                                policy = other,
                                "unsupported V_QWEN3_RMSNORM_BF16 policy"
                            );
                            panic!(
                                "unsupported V_QWEN3_RMSNORM_BF16 policy {other} at pc {pc}; expected 0=hidden64 or 1=hidden2048"
                            );
                        }
                    };
                    self.v_machine
                        .qwen3_rmsnorm_bf16(
                            self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_gp(*rs2),
                            hidden,
                            *BLEN as usize,
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
                    let offset = self.reg_file.read_gp(*rs1);
                    let addr = self.reg_file.read_hbm(*rs2);
                    let dtype = match precision {
                        op::MatrixPrecision::Weights => *MATRIX_WEIGHT_TYPE,
                        op::MatrixPrecision::KeyValue => *MATRIX_KV_TYPE,
                    };

                    // Single-slot load engine: wait for any previous M
                    // prefetch, then issue this DMA in the background. The
                    // instruction itself only pays the 1-cycle issue; the DMA
                    // latency is paid by the first matrix op that stalls on
                    // `m_load_barrier` (see the hazard block above).
                    self.drain_m_load().await;

                    let region = self.mx_region(dtype, addr, offset, *rstride);
                    let transfer = dma::transfer_mx_from_hbm(
                        &self.hbm,
                        region,
                        self.m_machine.mram.ty(),
                        *MLEN,
                        *PREFETCH_M_AMOUNT,
                        *MLEN,
                    );
                    hbm_issue_read_bytes = transfer.traffic.read_bytes;
                    let xfer = transfer.result;

                    let dest = self.reg_file.read_gp(*rd);
                    let amount = *PREFETCH_M_AMOUNT;
                    if self.blocking_prefetch {
                        // Calibration mode: pay the DMA here so --op-stats
                        // charges the exact service time to this instruction.
                        self.m_machine
                            .mram
                            .continous_write_delayed(dest, amount, xfer)
                            .await;
                    } else {
                        let mram = self.m_machine.mram.clone();
                        let (done_tx, done_rx) = tokio::sync::oneshot::channel();
                        runtime::Executor::current().spawn(async move {
                            mram.continous_write_delayed(dest, amount, xfer).await;
                            let _ = done_tx.send(());
                        });
                        self.m_load_barrier = Some(done_rx);
                        cycle!(1);
                    }
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

                    // Single-slot load engine, V side: issue in the background;
                    // VRAM consumers stall on `v_load_barrier`.
                    self.drain_v_load().await;

                    let region = self.mx_region(dtype, addr, offset, *rstride);
                    let transfer = dma::transfer_mx_from_hbm(
                        &self.hbm,
                        region,
                        self.v_machine.vram.ty(),
                        *VLEN,
                        *PREFETCH_V_AMOUNT,
                        1,
                    );
                    hbm_issue_read_bytes = transfer.traffic.read_bytes;
                    let xfer = transfer.result;

                    let dest = self.reg_file.read_gp(*rd);
                    let amount = *PREFETCH_V_AMOUNT;
                    if self.blocking_prefetch {
                        // Calibration mode: pay the DMA here (see H_PREFETCH_M).
                        self.v_machine
                            .vram
                            .continous_write_delayed(dest, amount, xfer)
                            .await;
                    } else {
                        let vram = self.v_machine.vram.clone();
                        let (done_tx, done_rx) = tokio::sync::oneshot::channel();
                        runtime::Executor::current().spawn(async move {
                            vram.continous_write_delayed(dest, amount, xfer).await;
                            let _ = done_tx.send(());
                        });
                        self.v_load_barrier = Some(done_rx);
                        cycle!(1);
                    }
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

                    let traffic = dma::transfer_mx_to_hbm(
                        &self.hbm,
                        &self.v_machine.vram,
                        region,
                        src_addr,
                        *VLEN,
                        *STORE_V_AMOUNT,
                    )
                    .await;
                    hbm_issue_read_bytes = traffic.read_bytes;
                    hbm_issue_written_bytes = traffic.written_bytes;
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
                    let broke_loop = self.loop_state.break_innermost(&mut self.reg_file);
                    cycle!(1);
                    if !broke_loop {
                        // Top-level C_BREAK = end of program (RTL halt marker).
                        if let (Some(mark), Some(recorder)) =
                            (op_mark.take(), self.op_stats.as_mut())
                        {
                            recorder.record(
                                pc,
                                op.mnemonic(),
                                mark,
                                hbm_issue_read_bytes,
                                hbm_issue_written_bytes,
                            );
                        }
                        break;
                    }
                }
            }

            if let (Some(mark), Some(recorder)) = (op_mark, self.op_stats.as_mut()) {
                recorder.record(
                    pc,
                    op.mnemonic(),
                    mark,
                    hbm_issue_read_bytes,
                    hbm_issue_written_bytes,
                );
            }

            // Handle loop jumps
            if let Some(target_pc) = jump_pc {
                pc = target_pc;
            } else {
                pc += 1;
            }
        }

        // Let any still-in-flight prefetch land before the caller reads the
        // SRAM dumps.
        self.drain_m_load().await;
        self.drain_v_load().await;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use half::bf16;
    use memory::{ErasedMemoryModel, MemoryBacked, NoData};
    use quantize::{DataType, FpType, MxDataType, QuantTensor};
    use runtime::{Executor, Instant};
    use sram::{MatrixSram, VectorSram};
    use tch::Tensor;

    use crate::matrix_machine::MatrixMachine;
    use crate::op::Opcode;
    use crate::vector_machine::VectorMachine;

    use super::Accelerator;

    fn fnv1a_bf16(values: &[f32]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for value in values {
            for byte in bf16::from_f32(*value).to_bits().to_le_bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn put_bf16(bytes: &mut [u8], offset: usize, value: f32) {
        bytes[offset..offset + 2].copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
    }

    #[tokio::test]
    async fn qwen_topk_dispatch_writes_expert_ids_and_route_weights() {
        let executor = Executor::new();
        let got = Arc::new(Mutex::new(None));
        let got_task = got.clone();

        executor.spawn(async move {
            let fp_type = DataType::Fp(FpType::BF16);
            let storage_type = MxDataType::Plain(fp_type);
            let vram = Arc::new(VectorSram::new(64, 4, fp_type, 4));
            let mut first = vec![-100.0f32; 64];
            let mut second = vec![-100.0f32; 64];
            first[2] = 7.0;
            first[5] = 7.0;
            first[63] = 3.0;
            second[0] = 4.0;
            second[7] = 5.0;
            second[20] = 2.5;
            second[21] = 2.25;
            second[63] = 6.0;
            vram.write(
                0,
                QuantTensor::quantize(Tensor::from_slice(&first), storage_type),
            )
            .await;
            vram.write(
                64,
                QuantTensor::quantize(Tensor::from_slice(&second), storage_type),
            )
            .await;

            let mram = Arc::new(MatrixSram::new(64, 4096, storage_type));
            let matrix = MatrixMachine::new(mram, vram.clone(), 64, 16, 4, 4, storage_type, "mxfp");
            let vector = VectorMachine::new(vram, 64, 16);
            let hbm: Arc<dyn ErasedMemoryModel> = Arc::new(NoData);
            let mut accelerator = Accelerator::new(matrix, vector, hbm);
            accelerator.reg_file.write_gp(1, 20);
            accelerator.reg_file.write_gp(2, 0);
            accelerator.reg_file.write_gp(3, 40);
            accelerator
                .do_ops(&[Opcode::V_TOPK {
                    rd: 1,
                    rs1: 2,
                    rs2: 3,
                    rmask: 1,
                }])
                .await;

            let indices = (40..48)
                .map(|address| accelerator.scalar_sram.read_int(address))
                .collect::<Vec<_>>();
            let weights = accelerator
                .scalar_sram
                .read_route_f32_window(20, 8)
                .to_vec();
            *got_task.lock().unwrap() = Some((indices, weights));
        });

        executor.enter(Instant::ETERNITY).await;
        let (indices, weights) = got.lock().unwrap().take().unwrap();
        assert_eq!(indices, vec![2, 5, 127, 71, 64, 63, 84, 85]);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 0.01);
        assert!(weights.windows(2).all(|pair| pair[0] >= pair[1]));
        assert!(
            weights
                .iter()
                .any(|weight| { bf16::from_f32(*weight).to_f32().to_bits() != weight.to_bits() })
        );
    }

    #[tokio::test]
    async fn qwen_router_linear_to_topk_matches_transformers_fixture() {
        let executor = Executor::new();
        let got = Arc::new(Mutex::new(None));
        let got_task = got.clone();

        executor.spawn(async move {
            const TILE: usize = 64;
            const EXPERTS: usize = 128;
            const INPUT_ROWS: usize = 4;
            let fp_type = DataType::Fp(FpType::BF16);
            let storage_type = MxDataType::Plain(fp_type);
            let weight_base = (INPUT_ROWS * TILE) as u32;
            let output_base = weight_base + (EXPERTS * TILE) as u32;
            let vram = Arc::new(VectorSram::new(TILE as u32, 134, fp_type, 4));
            let input = (0..TILE)
                .map(|index| {
                    let raw = ((index * 37 + 11) % 2001) as i32 - 1000;
                    bf16::from_f32(raw as f32 / 257.0).to_f32()
                })
                .collect::<Vec<_>>();
            vram.write(
                0,
                QuantTensor::quantize(Tensor::from_slice(&input), storage_type),
            )
            .await;
            for expert in 0..EXPERTS {
                let weight = (0..TILE)
                    .map(|index| {
                        let raw = ((expert * 97 + index * 53 + 19) % 2001) as i32 - 1000;
                        bf16::from_f32(raw as f32 / 263.0).to_f32()
                    })
                    .collect::<Vec<_>>();
                vram.write(
                    weight_base + (expert * TILE) as u32,
                    QuantTensor::quantize(Tensor::from_slice(&weight), storage_type),
                )
                .await;
            }

            let mram = Arc::new(MatrixSram::new(64, 4096, storage_type));
            let matrix = MatrixMachine::new(mram, vram.clone(), 64, 16, 4, 4, storage_type, "mxfp");
            let vector = VectorMachine::new(vram, 64, 16);
            let hbm: Arc<dyn ErasedMemoryModel> = Arc::new(NoData);
            let mut accelerator = Accelerator::new(matrix, vector, hbm);
            accelerator.reg_file.write_gp(1, output_base);
            accelerator.reg_file.write_gp(2, 0);
            accelerator.reg_file.write_gp(3, weight_base);
            accelerator.reg_file.write_gp(4, 20);
            accelerator.reg_file.write_gp(5, 40);
            accelerator
                .do_ops(&[
                    Opcode::V_ROUTER_LINEAR_BF16 {
                        rd: 1,
                        rs1: 2,
                        rs2: 3,
                        rmask: 0,
                    },
                    Opcode::V_TOPK {
                        rd: 4,
                        rs1: 1,
                        rs2: 5,
                        rmask: 1,
                    },
                ])
                .await;
            let indices = (40..48)
                .map(|address| accelerator.scalar_sram.read_int(address))
                .collect::<Vec<_>>();
            let weights = accelerator
                .scalar_sram
                .read_route_f32_window(20, 8)
                .to_vec();
            *got_task.lock().unwrap() = Some((indices, weights));
        });

        executor.enter(Instant::ETERNITY).await;
        let (indices, weights) = got.lock().unwrap().take().unwrap();
        assert_eq!(indices, vec![53, 115, 12, 33, 95, 74, 32, 94]);
        assert_eq!(
            weights
                .iter()
                .map(|weight| weight.to_bits())
                .collect::<Vec<_>>(),
            vec![
                0x3f5d_e77b,
                0x3df0_4073,
                0x3c82_0ee4,
                0x2e0f_eb3d,
                0x2d00_735a,
                0x2730_7f89,
                0x24aa_8d95,
                0x23fa_f8cc,
            ]
        );
    }

    #[tokio::test]
    async fn qwen_post_attention_moe_tail_matches_transformers_5_5_fixture() {
        let executor = Executor::new();
        let got = Arc::new(Mutex::new(None));
        let got_task = got.clone();

        executor.spawn(async move {
            const TILE: usize = 64;
            const BLEN_VALUE: usize = 4;
            const EXPERTS: usize = 128;
            const INTERMEDIATE: usize = 64;
            const DESCRIPTOR_BYTES: usize = 64;
            const GATE_UP_STRIDE: usize = 2 * INTERMEDIATE * TILE * 2;
            const DOWN_STRIDE: usize = TILE * INTERMEDIATE * 2;
            const GATE_UP_BASE: usize = DESCRIPTOR_BYTES;
            const DOWN_BASE: usize = GATE_UP_BASE + EXPERTS * GATE_UP_STRIDE;
            const HBM_END: usize = DOWN_BASE + EXPERTS * DOWN_STRIDE;
            const ATTENTION: u32 = 0;
            const RESIDUAL: u32 = 256;
            const POST_ATTENTION: u32 = 512;
            const MOE_RESIDUAL: u32 = 768;
            const NORM_WEIGHT: u32 = 1024;
            const NORMALIZED: u32 = 1280;
            const ROUTER_WEIGHT: u32 = 1536;
            const LOGITS: u32 = 9728;
            const COMBINED: u32 = 9856;

            let hbm_backing = Arc::new(MemoryBacked::with_capacity(HBM_END));
            hbm_backing.with_data(|bytes| {
                bytes[..8].copy_from_slice(b"Q3MOEBF1");
                put_u32(bytes, 8, 1);
                put_u32(bytes, 12, TILE as u32);
                put_u32(bytes, 16, INTERMEDIATE as u32);
                put_u32(bytes, 20, EXPERTS as u32);
                put_u64(bytes, 24, GATE_UP_BASE as u64);
                put_u64(bytes, 32, DOWN_BASE as u64);
                put_u64(bytes, 40, GATE_UP_STRIDE as u64);
                put_u64(bytes, 48, DOWN_STRIDE as u64);
                put_u64(bytes, 56, HBM_END as u64);
                for expert in 0..EXPERTS {
                    let expert_gate_up = GATE_UP_BASE + expert * GATE_UP_STRIDE;
                    for output in 0..INTERMEDIATE {
                        for input in 0..TILE {
                            let gate_raw =
                                ((expert * 13 + output * 17 + input * 19 + 5) % 257) as i32 - 128;
                            let up_raw =
                                ((expert * 23 + output * 29 + input * 31 + 7) % 257) as i32 - 128;
                            put_bf16(
                                bytes,
                                expert_gate_up + (output * TILE + input) * 2,
                                gate_raw as f32 / 509.0,
                            );
                            put_bf16(
                                bytes,
                                expert_gate_up + ((INTERMEDIATE + output) * TILE + input) * 2,
                                up_raw as f32 / 521.0,
                            );
                        }
                    }
                    let expert_down = DOWN_BASE + expert * DOWN_STRIDE;
                    for output in 0..TILE {
                        for input in 0..INTERMEDIATE {
                            let raw =
                                ((expert * 37 + output * 41 + input * 43 + 11) % 257) as i32 - 128;
                            put_bf16(
                                bytes,
                                expert_down + (output * INTERMEDIATE + input) * 2,
                                raw as f32 / 523.0,
                            );
                        }
                    }
                }
            });
            let hbm: Arc<dyn ErasedMemoryModel> = hbm_backing;
            let fp_type = DataType::Fp(FpType::BF16);
            let storage_type = MxDataType::Plain(fp_type);
            let vram = Arc::new(VectorSram::new(TILE as u32, 160, fp_type, BLEN_VALUE));
            let attention = (0..TILE)
                .map(|index| {
                    let raw = ((index * 17 + 3) % 127) as i32 - 63;
                    bf16::from_f32(raw as f32 / 41.0).to_f32()
                })
                .collect::<Vec<_>>();
            let residual = (0..TILE)
                .map(|index| {
                    let raw = ((index * 29 + 7) % 131) as i32 - 65;
                    bf16::from_f32(raw as f32 / 43.0).to_f32()
                })
                .collect::<Vec<_>>();
            let norm_weight = (0..TILE)
                .map(|index| bf16::from_f32(0.75 + ((index * 11) % 17) as f32 / 64.0).to_f32())
                .collect::<Vec<_>>();
            for (address, values) in [
                (ATTENTION, attention),
                (RESIDUAL, residual),
                (NORM_WEIGHT, norm_weight),
            ] {
                vram.write(
                    address,
                    QuantTensor::quantize(Tensor::from_slice(&values), storage_type),
                )
                .await;
            }
            for expert in 0..EXPERTS {
                let values = (0..TILE)
                    .map(|index| {
                        let raw = ((expert * 97 + index * 53 + 19) % 2001) as i32 - 1000;
                        bf16::from_f32(raw as f32 / 263.0).to_f32()
                    })
                    .collect::<Vec<_>>();
                vram.write(
                    ROUTER_WEIGHT + (expert * TILE) as u32,
                    QuantTensor::quantize(Tensor::from_slice(&values), storage_type),
                )
                .await;
            }

            let mram = Arc::new(MatrixSram::new(64, 4096, storage_type));
            let matrix = MatrixMachine::new(
                mram,
                vram.clone(),
                64,
                16,
                BLEN_VALUE as u32,
                BLEN_VALUE as u32,
                storage_type,
                "mxfp",
            );
            let vector = VectorMachine::new(vram.clone(), 64, 16);
            let mut accelerator = Accelerator::new(matrix, vector, hbm);
            for (register, value) in [
                (1, POST_ATTENTION),
                (2, ATTENTION),
                (3, RESIDUAL),
                (4, MOE_RESIDUAL),
                (5, NORM_WEIGHT),
                (6, NORMALIZED),
                (7, ROUTER_WEIGHT),
                (8, LOGITS),
                (9, 0),
                (10, COMBINED),
                (11, 0),
            ] {
                accelerator.reg_file.write_gp(register, value);
            }
            accelerator
                .do_ops(&[
                    Opcode::V_ADD_VV {
                        rd: 1,
                        rs1: 2,
                        rs2: 3,
                        rmask: 0,
                    },
                    Opcode::V_ADD_VF {
                        rd: 4,
                        rs1: 1,
                        rs2: 0,
                        rmask: 0,
                    },
                    Opcode::V_QWEN3_RMSNORM_BF16 {
                        rd: 6,
                        rs1: 1,
                        rs2: 5,
                        rmask: 0,
                    },
                    Opcode::V_ROUTER_LINEAR_BF16 {
                        rd: 8,
                        rs1: 6,
                        rs2: 7,
                        rmask: 0,
                    },
                    Opcode::V_TOPK {
                        rd: 9,
                        rs1: 8,
                        rs2: 9,
                        rmask: 1,
                    },
                    Opcode::C_SET_ADDR_REG {
                        rd: 1,
                        rs1: 0,
                        rs2: 11,
                    },
                    Opcode::V_QWEN3_EXPERT_COMBINE_BF16 {
                        rd: 10,
                        rs1: 6,
                        rs2: 9,
                        rmask: 1,
                    },
                    Opcode::V_ADD_VV {
                        rd: 10,
                        rs1: 10,
                        rs2: 4,
                        rmask: 0,
                    },
                ])
                .await;
            let output = vram.read(COMBINED).await;
            let output_values =
                Vec::<f32>::try_from(output.as_tensor().to_kind(tch::Kind::Float)).unwrap();
            let normalized = vram.read(NORMALIZED).await;
            let normalized_values =
                Vec::<f32>::try_from(normalized.as_tensor().to_kind(tch::Kind::Float)).unwrap();
            let indices = (0..8)
                .map(|address| accelerator.scalar_sram.read_int(address))
                .collect::<Vec<_>>();
            let scores = accelerator
                .scalar_sram
                .read_route_f32_window(0, 8)
                .iter()
                .map(|score| score.to_bits())
                .collect::<Vec<_>>();
            *got_task.lock().unwrap() = Some((
                fnv1a_bf16(&output_values),
                fnv1a_bf16(&normalized_values),
                indices,
                scores,
            ));
        });

        executor.enter(Instant::ETERNITY).await;
        let (output_hash, normalized_hash, indices, scores) = got.lock().unwrap().take().unwrap();
        assert_eq!(output_hash, 0x910c_569b_727d_0efc);
        assert_eq!(normalized_hash, 0x07eb_bab0_6edc_1369);
        assert_eq!(indices, vec![29, 99, 33, 95, 8, 70, 91, 50]);
        assert_eq!(
            scores,
            vec![
                0x3f7e_d0dc,
                0x3b0e_b240,
                0x3aad_196f,
                0x3a86_cf57,
                0x3856_c6fd,
                0x383d_8a52,
                0x35cf_8af5,
                0x358e_a456,
            ]
        );
    }

    #[tokio::test]
    #[should_panic(expected = "unsupported V_ROUTER_LINEAR_BF16 policy 2")]
    async fn qwen_router_linear_unknown_policy_fails_closed() {
        let executor = Executor::new();
        executor.spawn(async move {
            let fp_type = DataType::Fp(FpType::BF16);
            let storage_type = MxDataType::Plain(fp_type);
            let vram = Arc::new(VectorSram::new(64, 1, fp_type, 4));
            let mram = Arc::new(MatrixSram::new(64, 4096, storage_type));
            let matrix = MatrixMachine::new(mram, vram.clone(), 64, 16, 4, 4, storage_type, "mxfp");
            let vector = VectorMachine::new(vram, 64, 16);
            let hbm: Arc<dyn ErasedMemoryModel> = Arc::new(NoData);
            let mut accelerator = Accelerator::new(matrix, vector, hbm);
            accelerator
                .do_ops(&[Opcode::V_ROUTER_LINEAR_BF16 {
                    rd: 1,
                    rs1: 2,
                    rs2: 3,
                    rmask: 2,
                }])
                .await;
        });
        executor.enter(Instant::ETERNITY).await;
    }

    #[tokio::test]
    async fn qwen_route_multiply_retains_fp32_score_until_bf16_output_cast() {
        let executor = Executor::new();
        let got = Arc::new(Mutex::new(None));
        let got_task = got.clone();

        executor.spawn(async move {
            let fp_type = DataType::Fp(FpType::BF16);
            let storage_type = MxDataType::Plain(fp_type);
            let vram = Arc::new(VectorSram::new(64, 4, fp_type, 4));
            vram.write(
                0,
                QuantTensor::quantize(Tensor::from_slice(&vec![1.625f32; 64]), storage_type),
            )
            .await;

            let mram = Arc::new(MatrixSram::new(64, 4096, storage_type));
            let matrix = MatrixMachine::new(mram, vram.clone(), 64, 16, 4, 4, storage_type, "mxfp");
            let vector = VectorMachine::new(vram.clone(), 64, 16);
            let hbm: Arc<dyn ErasedMemoryModel> = Arc::new(NoData);
            let mut accelerator = Accelerator::new(matrix, vector, hbm);
            let score = 0.123_456_79f32;
            accelerator.scalar_sram.write_route_f32(12, score);
            accelerator.reg_file.write_gp(1, 64);
            accelerator.reg_file.write_gp(2, 0);
            accelerator.reg_file.write_gp(3, 12);
            accelerator
                .do_ops(&[Opcode::V_MUL_ROUTE_F32 {
                    rd: 1,
                    rs1: 2,
                    rs2: 3,
                    rmask: 0,
                }])
                .await;
            let output = vram.read(64).await;
            let first = output.as_tensor().double_value(&[0]) as f32;
            *got_task.lock().unwrap() = Some(first);
        });

        executor.enter(Instant::ETERNITY).await;
        let actual = got.lock().unwrap().take().unwrap();
        let expected = bf16::from_f32(1.625 * 0.123_456_79).to_f32();
        let prematurely_rounded =
            bf16::from_f32(1.625 * bf16::from_f32(0.123_456_79).to_f32()).to_f32();
        assert_eq!(actual.to_bits(), expected.to_bits());
        assert_ne!(actual.to_bits(), prematurely_rounded.to_bits());
    }
}
