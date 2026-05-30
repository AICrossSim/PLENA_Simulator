mod cli;
mod dma;
mod load_config;
mod matrix_machine;
mod op;
mod vector_machine;

use std::io::Write;
use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::sync::LazyLock;

use half::{bf16, f16};
use matrix_machine::MatrixMachine;
use memory::ErasedMemoryModel;
use quantize::MxDataType;
use runtime::{Duration, Executor, Instant};
use sram::{MatrixSram, VectorSram};
use vector_machine::VectorMachine;

use tracing_subscriber::prelude::*;

use cli::{Opts, Parser};

// Import the configuration functions
use load_config::*;

// Replace the const declarations with function calls to the config
// These functions will be called at runtime to get the configured values

const PERIOD: Duration = Duration::from_nanos(1);
static SYSTOLIC_PROCESSING_OVERHEAD: LazyLock<u32> =
    LazyLock::new(|| systolic_processing_overhead());
static VECTOR_ADD_CYCLES: LazyLock<u32> = LazyLock::new(|| vector_add_cycles());
static VECTOR_MUL_CYCLES: LazyLock<u32> = LazyLock::new(|| vector_mul_cycles());
static VECTOR_EXP_CYCLES: LazyLock<u32> = LazyLock::new(|| vector_exp_cycles());
static VECTOR_RECI_CYCLES: LazyLock<u32> = LazyLock::new(|| vector_reci_cycles());
static VECTOR_MAX_CYCLES: LazyLock<u32> = LazyLock::new(|| vector_max_cycles());
static VECTOR_SUM_CYCLES: LazyLock<u32> = LazyLock::new(|| vector_sum_cycles());
static SCALAR_FP_BASIC_CYCLES: LazyLock<u32> = LazyLock::new(|| scalar_fp_basic_cycles());
static SCALAR_FP_EXP_CYCLES: LazyLock<u32> = LazyLock::new(|| scalar_fp_exp_cycles());
static SCALAR_FP_SQRT_CYCLES: LazyLock<u32> = LazyLock::new(|| scalar_fp_sqrt_cycles());
static SCALAR_FP_RECI_CYCLES: LazyLock<u32> = LazyLock::new(|| scalar_fp_reci_cycles());
static SCALAR_INT_BASIC_CYCLES: LazyLock<u32> = LazyLock::new(|| scalar_int_basic_cycles());
static MAX_LOOP_INSTRUCTIONS: LazyLock<usize> = LazyLock::new(|| max_loop_instructions());

static MLEN: LazyLock<u32> = LazyLock::new(|| mlen());
static VLEN: LazyLock<u32> = LazyLock::new(|| vlen());
static BLEN: LazyLock<u32> = LazyLock::new(|| blen());
static HLEN: LazyLock<u32> = LazyLock::new(|| hlen());
static BROADCAST_AMOUNT: LazyLock<u32> = LazyLock::new(|| broadcast_amount());
static HBM_SIZE: LazyLock<usize> = LazyLock::new(|| hbm_size());
static MATRIX_SRAM_SIZE: LazyLock<usize> = LazyLock::new(|| matrix_sram_size());
static VECTOR_SRAM_SIZE: LazyLock<usize> = LazyLock::new(|| vector_sram_size());
static MATRIX_SRAM_TYPE: LazyLock<MxDataType> = LazyLock::new(|| matrix_sram_type());
static VECTOR_SRAM_TYPE: LazyLock<MxDataType> = LazyLock::new(|| vector_sram_type());
static MATRIX_WEIGHT_TYPE: LazyLock<MxDataType> = LazyLock::new(|| matrix_weight_type());
static MATRIX_KV_TYPE: LazyLock<MxDataType> = LazyLock::new(|| matrix_kv_type());
static VECTOR_ACTIVATION_TYPE: LazyLock<MxDataType> = LazyLock::new(|| vector_activation_type());
static VECTOR_KV_TYPE: LazyLock<MxDataType> = LazyLock::new(|| vector_kv_type());
static PREFETCH_M_AMOUNT: LazyLock<u32> = LazyLock::new(|| {
    let raw = hbm_m_prefetch_amount();
    let mlen = mlen();
    // Must be a multiple of MLEN (one full matrix tile per write).
    // Round up to the nearest multiple of MLEN if needed.
    if raw < mlen {
        tracing::warn!(
            "HBM_M_Prefetch_Amount ({}) < MLEN ({}); clamping to MLEN",
            raw,
            mlen
        );
        mlen
    } else if raw % mlen != 0 {
        let clamped = ((raw + mlen - 1) / mlen) * mlen;
        tracing::warn!(
            "HBM_M_Prefetch_Amount ({}) not a multiple of MLEN ({}); rounding up to {}",
            raw,
            mlen,
            clamped
        );
        clamped
    } else {
        raw
    }
});
static PREFETCH_V_AMOUNT: LazyLock<u32> = LazyLock::new(|| hbm_v_prefetch_amount());
static STORE_V_AMOUNT: LazyLock<u32> = LazyLock::new(|| hbm_v_writeback_amount());

#[macro_export]
macro_rules! cycle {
    ($cycle: expr) => {
        ::runtime::Executor::current()
            .resolve_at($crate::PERIOD * ($cycle as u32))
            .await;
    };
}

/// Information about an active loop
struct LoopInfo {
    start_pc: usize,          // Program counter of C_LOOP_START
    iteration_count: u32,     // Total number of iterations (from imm)
    current_iteration: u32,   // Current iteration (starts at iteration_count, decrements)
    instruction_count: usize, // Number of instructions executed in current iteration
    loop_reg: u8,             // Register used for loop counter (rd from C_LOOP_START)
}

struct Accelerator {
    m_machine: MatrixMachine,
    v_machine: VectorMachine,
    hbm: Arc<dyn ErasedMemoryModel>,
    reg_file: AcceleratorRegFile,
    intsram: Vec<u32>,
    fpsram: Vec<bf16>,
    loop_stack: Vec<LoopInfo>, // Stack for nested loops
}

struct AcceleratorRegFile {
    // === ISA-indexed register banks ===
    gp_reg: [u32; 16],
    fp_reg: [bf16; 8],
    hbm_addr_reg: [u64; 16],

    // === Global config registers ===
    scale: u32,
    stride: u32,
    bmm_scale: f32, // Scale factor during the BMM operation, apply to every element in the matrix operation.
    v_mask: u32,    // HLEN Head Mask for VLEN Vector
}

impl AcceleratorRegFile {
    /// Read a general-purpose register by its 4-bit ISA encoding.
    fn read_gp(&self, r: u8) -> u32 {
        self.gp_reg[r as usize]
    }

    /// Read a floating-point register by its 3-bit ISA encoding.
    fn read_fp(&self, r: u8) -> bf16 {
        self.fp_reg[r as usize]
    }

    /// Read an HBM address register by its 4-bit ISA encoding.
    fn read_hbm(&self, r: u8) -> u64 {
        self.hbm_addr_reg[r as usize]
    }

    /// Write a general-purpose register by its 4-bit ISA encoding.
    fn write_gp(&mut self, r: u8, v: u32) {
        self.gp_reg[r as usize] = v;
    }

    /// Write a floating-point register by its 3-bit ISA encoding.
    fn write_fp(&mut self, r: u8, v: bf16) {
        self.fp_reg[r as usize] = v;
    }

    /// Write an HBM address register by its 4-bit ISA encoding.
    fn write_hbm(&mut self, r: u8, v: u64) {
        self.hbm_addr_reg[r as usize] = v;
    }

    /// `dst_gp = op(read_gp(src1), read_gp(src2))`. Helper for binary GP-to-GP
    /// instructions (S_ADD_INT / S_SUB_INT / S_MUL_INT).
    fn binop_gp<F: FnOnce(u32, u32) -> u32>(&mut self, dst: u8, src1: u8, src2: u8, op: F) {
        let v = op(self.read_gp(src1), self.read_gp(src2));
        self.write_gp(dst, v);
    }

    /// `dst_fp = op(read_fp(src1), read_fp(src2))`. Helper for binary FP-to-FP
    /// instructions (S_ADD_FP / S_SUB_FP / S_MAX_FP / S_MUL_FP).
    fn binop_fp<F: FnOnce(bf16, bf16) -> bf16>(&mut self, dst: u8, src1: u8, src2: u8, op: F) {
        let v = op(self.read_fp(src1), self.read_fp(src2));
        self.write_fp(dst, v);
    }
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
            self.reg_file.v_mask
        }
    }

    async fn do_ops(&mut self, ops: &[op::Opcode]) {
        let mut pc: usize = 0; // Program counter

        while pc < ops.len() {
            let op = &ops[pc];

            // Update instruction count for active loops
            for loop_info in &mut self.loop_stack {
                loop_info.instruction_count += 1;
                // Check if we've exceeded the max instructions limit
                if loop_info.instruction_count > *MAX_LOOP_INSTRUCTIONS {
                    tracing::error!(
                        loop_pc = loop_info.start_pc,
                        max = *MAX_LOOP_INSTRUCTIONS,
                        current_iter = loop_info.current_iteration,
                        instructions = loop_info.instruction_count,
                        "Loop exceeded max instructions limit"
                    );
                    panic!(
                        "Loop at PC {} exceeded max instructions limit ({}). Current iteration: {}, Instructions in this iteration: {}",
                        loop_info.start_pc,
                        *MAX_LOOP_INSTRUCTIONS,
                        loop_info.current_iteration,
                        loop_info.instruction_count
                    );
                }
            }

            tracing::debug!(pc, ?op, "execute op");

            let mut jump_pc: Option<usize> = None;

            match op {
                op::Opcode::Invalid => todo!(),

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
                        .mm_wo(self.reg_file.read_gp(*rd) + *imm as u32, stride_len as u32)
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
                            self.reg_file.bmm_scale,
                        )
                        .await;
                }
                op::Opcode::M_BTMM { rs1, rs2 } => {
                    self.m_machine
                        .btmm(
                            self.reg_file.read_gp(*rs1),
                            self.reg_file.read_gp(*rs2),
                            self.reg_file.bmm_scale,
                        )
                        .await;
                }
                op::Opcode::M_BMM_WO { rd, imm } => {
                    self.m_machine
                        .bmm_wo(self.reg_file.read_gp(*rd) + *imm as u32)
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
                            self.reg_file.bmm_scale,
                        )
                        .await;
                }
                op::Opcode::M_BTMV { rs1, rs2, rd } => {
                    self.m_machine
                        .btmv(
                            self.reg_file.read_gp(*rs1) + self.reg_file.read_gp(*rd),
                            self.reg_file.read_gp(*rs2),
                            self.reg_file.bmm_scale,
                        )
                        .await;
                }
                op::Opcode::M_MV_WO { rd, imm } => {
                    self.m_machine
                        .mv_wo(self.reg_file.read_gp(*rd) + *imm as u32)
                        .await;
                }
                op::Opcode::M_BMV_WO { rd, imm } => {
                    self.m_machine
                        .bmv_wo(self.reg_file.read_gp(*rd) + *imm as u32)
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
                        self.fpsram[(self.reg_file.read_gp(*rs1) + *imm) as usize],
                    );
                    cycle!(1);
                }
                op::Opcode::S_ST_FP { rd, rs1, imm } => {
                    self.fpsram[(self.reg_file.read_gp(*rs1) + *imm) as usize] =
                        self.reg_file.read_fp(*rd);
                    cycle!(1);
                }
                op::Opcode::S_MAP_V_FP { rd, rs1, imm } => {
                    let start_idx = (self.reg_file.read_gp(*rs1) + *imm) as usize;
                    let end_idx = start_idx + *VLEN as usize;
                    let f = &self.fpsram[start_idx..end_idx];
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
                        .write_gp(*rd, self.reg_file.read_gp(*rs1).wrapping_add(*imm as u32));
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
                    self.reg_file.write_gp(*rd, (*imm as u32) << 12);
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::S_LD_INT { rd, rs1, imm } => {
                    self.reg_file.write_gp(
                        *rd,
                        self.intsram[(self.reg_file.read_gp(*rs1) + *imm) as usize],
                    );
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::S_ST_INT { rd, rs1, imm } => {
                    self.intsram[(self.reg_file.read_gp(*rs1) + *imm) as usize] =
                        self.reg_file.read_gp(*rd);
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

                    // Element addr shifted by (element to scale ratio)
                    let scale = match dtype {
                        MxDataType::Plain(_) => 0,
                        MxDataType::Mx { .. } => offset / dtype.element_scale_ratio(),
                    };
                    let region = dma::MxRegion {
                        hbm_type: dtype,
                        index: addr + offset as u64,
                        scale_index: addr + self.reg_file.scale as u64 + scale as u64,
                        rstride: *rstride,
                        stride: self.reg_file.stride,
                    };
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

                    let scale = match dtype {
                        MxDataType::Plain(_) => 0,
                        MxDataType::Mx { .. } => offset / dtype.element_scale_ratio(),
                    };
                    let region = dma::MxRegion {
                        hbm_type: dtype,
                        index: addr + offset as u64,
                        scale_index: addr + self.reg_file.scale as u64 + scale as u64,
                        rstride: *rstride,
                        stride: self.reg_file.stride,
                    };
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

                    let scale = match dtype {
                        MxDataType::Plain(_) => 0,
                        MxDataType::Mx { .. } => offset / dtype.element_scale_ratio(),
                    };

                    let region = dma::MxRegion {
                        hbm_type: dtype,
                        index: addr + offset as u64,
                        // Scales are stored AFTER elements, so scale_index =
                        // element_index + scale_reg + scale, where scale_reg is
                        // the offset from element start to scale start.
                        scale_index: addr + self.reg_file.scale as u64 + scale as u64,
                        rstride: *rstride,
                        stride: self.reg_file.stride,
                    };

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
                    self.reg_file.scale = self.reg_file.read_gp(*rd);
                    cycle!(1);
                }
                op::Opcode::C_SET_STRIDE_REG { rd } => {
                    self.reg_file.stride = self.reg_file.read_gp(*rd);
                    cycle!(1);
                }
                op::Opcode::C_SET_V_MASK_REG { rd } => {
                    self.reg_file.v_mask = self.reg_file.read_gp(*rd);
                    cycle!(1);
                }
                op::Opcode::C_LOOP_START { rd, imm } => {
                    // Store iteration count in register
                    assert!(*imm > 0, "Iteration count must be greater than 0");
                    let iteration_count = *imm as u32;
                    self.reg_file.write_gp(*rd, iteration_count);

                    // Push new loop onto stack
                    self.loop_stack.push(LoopInfo {
                        start_pc: pc,
                        iteration_count,
                        current_iteration: iteration_count,
                        instruction_count: 0,
                        loop_reg: *rd,
                    });

                    tracing::debug!(
                        "C_LOOP_START: Starting loop at PC {} with {} iterations",
                        pc,
                        iteration_count
                    );
                    cycle!(1);
                }
                op::Opcode::C_LOOP_END { rd } => {
                    // Find the matching loop (most recent loop with matching register)
                    if let Some(loop_info) =
                        self.loop_stack.iter_mut().rev().find(|l| l.loop_reg == *rd)
                    {
                        // Decrement the register (as per spec)
                        let reg_value = self.reg_file.read_gp(*rd);
                        if reg_value > 1 {
                            // More iterations remaining, loop back
                            self.reg_file.write_gp(*rd, reg_value - 1);

                            // Update loop state
                            loop_info.current_iteration = reg_value - 1;
                            loop_info.instruction_count = 0; // Reset instruction count for next iteration

                            // Jump back to C_LOOP_START + 1 (skip the C_LOOP_START instruction itself)
                            jump_pc = Some(loop_info.start_pc + 1);

                            tracing::debug!(
                                "C_LOOP_END: Looping back to PC {} (remaining iterations: {})",
                                loop_info.start_pc + 1,
                                reg_value - 1
                            );
                        } else {
                            // Last iteration (reg_value == 1) or already done (reg_value == 0)
                            // Decrement to 0 and exit the loop
                            self.reg_file.write_gp(*rd, 0);

                            // Loop is complete, pop it from stack
                            tracing::debug!(
                                "C_LOOP_END: Loop at PC {} completed (executed {} times)",
                                loop_info.start_pc,
                                loop_info.iteration_count
                            );
                            // Remove this loop from the stack
                            let loop_reg = loop_info.loop_reg;
                            let pos = self
                                .loop_stack
                                .iter()
                                .rposition(|l| l.loop_reg == loop_reg)
                                .unwrap();
                            self.loop_stack.remove(pos);
                        }
                    } else {
                        tracing::error!(
                            rd = *rd,
                            loop_stack_depth = self.loop_stack.len(),
                            "C_LOOP_END: No matching C_LOOP_START found"
                        );
                        panic!(
                            "C_LOOP_END: No matching C_LOOP_START found for register {}",
                            *rd
                        );
                    }
                    cycle!(1);
                }
                op::Opcode::C_BREAK => {
                    // Break out of the innermost loop
                    if let Some(loop_info) = self.loop_stack.pop() {
                        tracing::debug!(
                            "C_BREAK: Breaking out of loop at PC {}",
                            loop_info.start_pc
                        );
                        // Set the loop register to 0 to indicate loop is done
                        self.reg_file.write_gp(loop_info.loop_reg, 0);
                    } else {
                        tracing::error!("C_BREAK: No active loop to break out of");
                        panic!("C_BREAK: No active loop to break out of");
                    }
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

/// Write `bytes` to `path` as a diagnostic dump.
///
/// Dumps are post-run artifacts, so a write failure (e.g. read-only cwd, full
/// disk) is logged as a warning and the run continues rather than panicking and
/// discarding an already-completed simulation.
fn dump_to_file(path: &str, bytes: &[u8]) {
    match std::fs::File::create(path).and_then(|mut f| f.write_all(bytes)) {
        Ok(()) => tracing::info!(path, bytes = bytes.len(), "dumped content"),
        Err(err) => tracing::warn!(path, %err, "failed to write dump file"),
    }
}

async fn start() {
    let opts = Opts::parse();

    // If --settings is given, set PLENA_SETTINGS_TOML env var BEFORE any
    // LazyLock access (which triggers load_config()). This ensures the
    // per-build TOML is used for all config values.
    if let Some(ref settings_path) = opts.settings {
        // SAFETY: set_var is called before any threads are spawned and before
        // LazyLock statics are accessed, so no concurrent readers exist.
        unsafe { std::env::set_var("PLENA_SETTINGS_TOML", settings_path.as_os_str()) };
    }

    // Initialize tracing subscriber.
    //
    // Filter precedence: `--log-level` (full override) > `RUST_LOG` > default (debug).
    // Output: stderr by default; if `--log-file` is given, also writes to that
    // file (non-blocking appender, no ANSI codes in file).
    let env_filter: tracing_subscriber::EnvFilter = match opts.log_level {
        Some(level) => tracing_subscriber::EnvFilter::new(level.as_level_filter().to_string()),
        None => tracing_subscriber::EnvFilter::builder()
            .with_default_directive(tracing_subscriber::filter::LevelFilter::DEBUG.into())
            .from_env_lossy(),
    };

    let stderr_layer = tracing_subscriber::fmt::layer().with_writer(std::io::stderr);

    // Hold the worker guard for the rest of `start()` so the appender's
    // background thread isn't dropped before logs are flushed.
    let (file_layer, _file_guard) = match opts.log_file.as_ref() {
        Some(path) => {
            let target = cli::validate_log_file_path(path).unwrap_or_else(|err| {
                // Bootstrap error: the tracing subscriber is not installed yet
                // (we are still building its layers below), so write to stderr.
                eprintln!("error: {}", err);
                std::process::exit(1);
            });
            let appender = tracing_appender::rolling::never(&target.parent, &target.filename);
            let (non_blocking, guard) = tracing_appender::non_blocking(appender);
            let layer = tracing_subscriber::fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false);
            (Some(layer), Some(guard))
        }
        None => (None, None),
    };

    tracing_subscriber::registry()
        .with(env_filter)
        .with(stderr_layer)
        .with(file_layer)
        .init();

    tracing::warn!(
        mlen = *MLEN,
        vlen = *VLEN,
        hlen = *HLEN,
        blen = *BLEN,
        broadcast_amount = *BROADCAST_AMOUNT,
        "Topology"
    );
    tracing::info!(
        matrix_sram_size = *MATRIX_SRAM_SIZE,
        vector_sram_size = *VECTOR_SRAM_SIZE,
        matrix_type = ?*MATRIX_SRAM_TYPE,
        vector_type = ?*VECTOR_SRAM_TYPE,
        "SRAM"
    );
    tracing::info!(
        prefetch_m = *PREFETCH_M_AMOUNT,
        prefetch_v = *PREFETCH_V_AMOUNT,
        store_v = *STORE_V_AMOUNT,
        max_loop_instructions = *MAX_LOOP_INSTRUCTIONS,
        "Pipeline"
    );
    tracing::info!(
        settings = %std::env::var("PLENA_SETTINGS_TOML")
            .unwrap_or_else(|_| "default (../plena_settings.toml)".to_string()),
        "Config source"
    );

    let mram = Arc::new(MatrixSram::new(*MLEN, *MATRIX_SRAM_SIZE, *MATRIX_SRAM_TYPE)); // Matrix SRAM
    let vram = Arc::new(VectorSram::from_mx_type(
        *VLEN,
        *VECTOR_SRAM_SIZE,
        *VECTOR_SRAM_TYPE,
    )); // Vector SRAM

    let m_machine = MatrixMachine::new(mram, vram.clone(), *MLEN, *HLEN, *BLEN, *BROADCAST_AMOUNT);

    let v_machine = VectorMachine::new(vram, *VLEN, *HLEN); // Share same dim with VSRAM

    // Allow CLI override of HBM size. The default (from plena_settings.toml)
    // can be 128 GiB to fit large models like LLaDA-8B; tests with smaller
    // preloads should pass --hbm-size to bound the steady-state RSS.
    let effective_hbm_size = opts.hbm_size.unwrap_or(*HBM_SIZE);
    tracing::info!(
        "HBM size: {} bytes ({:.2} GiB)",
        effective_hbm_size,
        effective_hbm_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    let hbm = Arc::new(memory::WithStats::new(memory::WithTiming::new(
        ManuallyDrop::new(ramulator::Ramulator::hbm2_preset(8).unwrap()),
        memory::MemoryBacked::with_capacity(effective_hbm_size),
    )));

    let mut accelerator = Accelerator {
        m_machine,
        v_machine,
        hbm: hbm.clone(),
        reg_file: AcceleratorRegFile {
            gp_reg: [0; 16],
            fp_reg: [bf16::ZERO; 8],
            hbm_addr_reg: [0; 16],
            scale: 0,
            stride: 1,
            // bmm_scale = 0.25 corresponds to 1/sqrt(head_dim=16).
            // For other head dimensions, the ISA program must set this via
            // the appropriate scalar register instruction before M_BMM/M_BTMM.
            bmm_scale: 0.25,
            v_mask: 0,
        },
        intsram: vec![0; 1024],
        fpsram: vec![bf16::ZERO; 1024],
        loop_stack: Vec::new(),
    };

    use std::fs;
    // Panic (rather than exit) on these fatal startup errors so the stack
    // unwinds: that runs the tracing-appender WorkerGuard's Drop, flushing any
    // buffered --log-file output, and preserves the prior exit-101 behavior.
    let op_file = fs::read_to_string(&opts.opcode)
        .unwrap_or_else(|err| panic!("failed to read opcode file {:?}: {err}", opts.opcode));

    let op: Vec<u32> = op_file
        .split_whitespace() // split by spaces/newlines
        .map(|tok| {
            u32::from_str_radix(tok.trim_start_matches("0x"), 16)
                .unwrap_or_else(|err| panic!("failed to parse opcode hex token {tok:?}: {err}"))
        })
        .collect();

    // Memory Initialization
    // - HBM Preload
    let hbm_data = std::fs::read(&opts.hbm)
        .unwrap_or_else(|err| panic!("failed to read HBM preload file {:?}: {err}", opts.hbm));
    hbm.model().data().with_data(|f| {
        f[..hbm_data.len()].copy_from_slice(&hbm_data);
    });

    // Load fpsram and intsram as raw bytes and map to the vector files.
    // - fpsram Preload
    let fpsram_data = std::fs::read(&opts.fpsram).unwrap_or_else(|err| {
        panic!(
            "failed to read FP SRAM preload file {:?}: {err}",
            opts.fpsram
        )
    });
    let fp_vals: Vec<bf16> = {
        let n = fpsram_data.len() / std::mem::size_of::<f16>();
        let f16_slice: &[f16] =
            unsafe { std::slice::from_raw_parts(fpsram_data.as_ptr() as *const f16, n) };
        f16_slice
            .iter()
            .map(|x| bf16::from_f32(f32::from(*x)))
            .collect()
    };

    // Replace the beginning of accelerator.fpsram with fp_vals
    accelerator.fpsram[..fp_vals.len()].copy_from_slice(&fp_vals[..fp_vals.len()]);

    // - INT SRAM Preload
    if let Some(intsram_path) = opts.intsram {
        let intsram_data = std::fs::read(&intsram_path).unwrap_or_else(|err| {
            panic!(
                "failed to read INT SRAM preload file {:?}: {err}",
                intsram_path
            )
        });
        let int_vals: &[u32] = unsafe {
            std::slice::from_raw_parts(
                intsram_data.as_ptr() as *const u32,
                intsram_data.len() / std::mem::size_of::<u32>(),
            )
        };
        accelerator.intsram[..int_vals.len()].copy_from_slice(&int_vals[..int_vals.len()]);
    }
    // - VRAM Preload (if provided)
    if let Some(vram_path) = opts.vram {
        let vram_data = std::fs::read(&vram_path).unwrap_or_else(|err| {
            panic!("failed to read VRAM preload file {:?}: {err}", vram_path)
        });
        accelerator.v_machine.vram.load_from_bytes(&vram_data).await;
    }

    // - Execute Instructions
    // accelerator
    //     .do_ops(&dbg!(
    //         op.into_iter().map(op::Opcode::decode).collect::<Vec<_>>()
    //     ))
    //     .await;
    let decoded_ops = op.into_iter().map(op::Opcode::decode).collect::<Vec<_>>();
    accelerator.do_ops(&decoded_ops).await;

    tracing::debug!("gp1 = {:x}", accelerator.reg_file.gp_reg[1]);
    tracing::debug!("scale = {}", accelerator.reg_file.scale);
    tracing::debug!(
        "Vector SRAM Contents: \n {}",
        accelerator.v_machine.vram.read(0x0000).await.as_tensor()
    );

    tracing::debug!(
        "Matrix SRAM Contents: \n {}",
        accelerator.m_machine.mram.read(0x0000).await.as_tensor()
    );

    tracing::debug!("INT SRAM Contents: \n {:?}", accelerator.intsram);
    tracing::debug!("FP SRAM Contents: \n {:?}", accelerator.fpsram);

    // Dump MRAM
    let mram_bytes = accelerator.m_machine.mram.as_bytes().await;
    dump_to_file("mram_dump.bin", &mram_bytes);

    // Dump VRAM
    let vram_bytes = accelerator.v_machine.vram.as_bytes().await;
    dump_to_file("vram_dump.bin", &vram_bytes);

    // Dump FPSRAM
    let fpsram_bytes: Vec<u8> = accelerator
        .fpsram
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    dump_to_file("fpsram_dump.bin", &fpsram_bytes);

    // Dump HBM — skipped unless DEBUG tracing is enabled because HBM_SIZE may
    // be 128 GiB+. Tests run with --log-level warn and don't need hbm_dump.bin;
    // only manual debug runs dump HBM.
    if tracing::enabled!(tracing::Level::DEBUG) {
        let hbm_size = effective_hbm_size;
        let mut hbm_bytes = vec![0u8; hbm_size];
        hbm.model().data().with_data(|f| {
            let len = std::cmp::min(hbm_size, f.len());
            hbm_bytes[..len].copy_from_slice(&f[..len]);
        });
        dump_to_file("hbm_dump.bin", &hbm_bytes);
    }

    let memory_stats = hbm.statistics();
    let utilization = (memory_stats.total_bytes_read + memory_stats.total_bytes_written) as f64
        / Executor::current().now().to_secs();
    tracing::info!(
        "HBM Statistics - Bytes read: {:?} | Bytes written: {:?} | Utilization: {:.2e} bytes/sec",
        memory_stats.total_bytes_read,
        memory_stats.total_bytes_written,
        utilization
    );
}

#[tokio::main]
async fn main() {
    let executor = Executor::new();
    executor.spawn(start());
    executor.enter(Instant::ETERNITY).await;
    tracing::info!("Simulation completed. Latency {:?}", executor.now());
}
