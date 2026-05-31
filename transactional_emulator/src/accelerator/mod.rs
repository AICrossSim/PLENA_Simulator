mod execution;

use std::sync::Arc;

use crate::matrix_machine::MatrixMachine;
use crate::vector_machine::VectorMachine;
use half::bf16;
use memory::ErasedMemoryModel;

/// Information about an active loop.
struct LoopInfo {
    start_pc: usize,          // Program counter of C_LOOP_START
    iteration_count: u32,     // Total number of iterations (from imm)
    current_iteration: u32,   // Current iteration (starts at iteration_count, decrements)
    instruction_count: usize, // Number of instructions executed in current iteration
    loop_reg: u8,             // Register used for loop counter (rd from C_LOOP_START)
}

enum DispatchOutcome {
    Next,
    Jump(usize),
}

pub(crate) struct Accelerator {
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
    pub(crate) fn new(
        m_machine: MatrixMachine,
        v_machine: VectorMachine,
        hbm: Arc<dyn ErasedMemoryModel>,
    ) -> Self {
        Self {
            m_machine,
            v_machine,
            hbm,
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
        }
    }

    pub(crate) fn load_fpsram(&mut self, values: &[bf16]) {
        self.fpsram[..values.len()].copy_from_slice(&values[..values.len()]);
    }

    pub(crate) fn load_intsram(&mut self, values: &[u32]) {
        self.intsram[..values.len()].copy_from_slice(&values[..values.len()]);
    }

    pub(crate) async fn load_vram_from_bytes(&mut self, bytes: &[u8]) {
        self.v_machine.vram.load_from_bytes(bytes).await;
    }

    pub(crate) async fn log_debug_state(&mut self) {
        tracing::debug!("gp1 = {:x}", self.reg_file.gp_reg[1]);
        tracing::debug!("scale = {}", self.reg_file.scale);
        tracing::debug!(
            "Vector SRAM Contents: \n {}",
            self.v_machine.vram.read(0x0000).await.as_tensor()
        );

        tracing::debug!(
            "Matrix SRAM Contents: \n {}",
            self.m_machine.mram.read(0x0000).await.as_tensor()
        );

        tracing::debug!("INT SRAM Contents: \n {:?}", self.intsram);
        tracing::debug!("FP SRAM Contents: \n {:?}", self.fpsram);
    }

    pub(crate) async fn mram_bytes(&mut self) -> Vec<u8> {
        self.m_machine.mram.as_bytes().await
    }

    pub(crate) async fn vram_bytes(&mut self) -> Vec<u8> {
        self.v_machine.vram.as_bytes().await
    }

    pub(crate) fn fpsram_bytes(&self) -> Vec<u8> {
        self.fpsram.iter().flat_map(|f| f.to_le_bytes()).collect()
    }
}
