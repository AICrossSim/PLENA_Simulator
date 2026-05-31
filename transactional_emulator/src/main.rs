mod cli;
mod dma;
mod load_config;
mod matrix_machine;
mod op;
mod runner;
mod vector_machine;

use std::sync::Arc;
use std::sync::LazyLock;

use half::bf16;
use matrix_machine::MatrixMachine;
use memory::ErasedMemoryModel;
use quantize::MxDataType;
use runtime::{Duration, Executor, Instant};
use vector_machine::VectorMachine;

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

#[tokio::main]
async fn main() {
    let executor = Executor::new();
    executor.spawn(runner::run_from_cli());
    executor.enter(Instant::ETERNITY).await;
    tracing::info!("Simulation completed. Latency {:?}", executor.now());
}
