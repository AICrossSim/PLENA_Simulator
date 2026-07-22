//! `Accelerator` — the top-level ISA dispatcher.
//!
//! Holds the register file, the matrix/vector machines, the scalar SRAMs, the
//! HBM handle, and the loop stack; [`Accelerator::do_ops`] decodes and executes
//! an opcode stream one instruction at a time. The runner-facing state API
//! lives here; opcode execution lives in `dispatch.rs`.

use std::sync::Arc;

use memory::ErasedMemoryModel;

use crate::matrix_machine::MatrixMachine;
use crate::vector_machine::VectorMachine;

mod dispatch;
mod loop_state;
mod registers;
mod scalar_sram;

use loop_state::LoopState;
use registers::AcceleratorRegFile;
use scalar_sram::ScalarSram;

pub(crate) struct Accelerator {
    m_machine: MatrixMachine,
    v_machine: VectorMachine,
    hbm: Arc<dyn ErasedMemoryModel>,
    reg_file: AcceleratorRegFile,
    scalar_sram: ScalarSram,
    loop_state: LoopState,
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
            reg_file: AcceleratorRegFile::new(),
            scalar_sram: ScalarSram::new(),
            loop_state: LoopState::new(),
        }
    }

    pub(crate) fn load_fpsram_from_f16_bytes(&mut self, bytes: &[u8]) {
        self.scalar_sram.load_fpsram_from_f16_bytes(bytes);
    }

    pub(crate) fn load_intsram_from_u32_bytes(&mut self, bytes: &[u8]) {
        self.scalar_sram.load_intsram_from_u32_bytes(bytes);
    }

    pub(crate) async fn load_vram_from_bytes(&mut self, bytes: &[u8]) {
        self.v_machine.vram.load_from_bytes(bytes).await;
    }

    pub(crate) async fn log_debug_state(&mut self) {
        tracing::debug!("gp1 = {:x}", self.reg_file.read_gp(1));
        tracing::debug!("scale = {}", self.reg_file.scale());
        tracing::debug!(
            "Vector SRAM Contents: \n {}",
            self.v_machine.vram.read(0x0000).await.as_tensor()
        );
        tracing::debug!(
            "Matrix SRAM Contents: \n {}",
            self.m_machine.mram.read(0x0000).await.as_tensor()
        );
        self.scalar_sram.log_debug_contents();
    }

    pub(crate) async fn mram_dump_bytes(&mut self) -> Vec<u8> {
        self.m_machine.mram.as_bytes().await
    }

    pub(crate) async fn vram_dump_bytes(&mut self) -> Vec<u8> {
        self.v_machine.vram.as_bytes().await
    }

    pub(crate) fn fpsram_dump_bytes(&self) -> Vec<u8> {
        self.scalar_sram.fpsram_to_le_bytes()
    }

    pub(crate) fn intsram_dump_bytes(&self) -> Vec<u8> {
        self.scalar_sram.intsram_to_le_bytes()
    }
}
