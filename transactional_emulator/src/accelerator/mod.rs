//! `Accelerator` — the top-level ISA dispatcher.
//!
//! Holds the register file, the matrix/vector machines, the scalar SRAMs, the
//! HBM handle, and the loop stack; [`Accelerator::do_ops`] decodes and executes
//! an opcode stream one instruction at a time. The runner-facing state API
//! lives here; opcode execution lives in `dispatch.rs`.

use std::sync::Arc;

use memory::ErasedMemoryModel;

use crate::matrix_machine::MatrixMachine;
use crate::op_stats::OpStatsRecorder;
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
    /// Per-instruction time/HBM sampling; `None` unless `--op-stats` is given.
    op_stats: Option<OpStatsRecorder>,
    /// Completion signal for the in-flight background `H_PREFETCH_M` DMA.
    /// `H_PREFETCH_*` issues its DMA and returns immediately. The load engine
    /// has one slot, so a new prefetch waits on the previous one and SRAM
    /// consumers stall here before executing (the RTL's `m_load_in_process`
    /// stall). DMA latency is charged at that stall, not at issue.
    m_load_barrier: Option<tokio::sync::oneshot::Receiver<()>>,
    /// Same as `m_load_barrier`, for the `H_PREFETCH_V` engine.
    v_load_barrier: Option<tokio::sync::oneshot::Receiver<()>>,
    /// `--blocking-prefetch`: calibration mode where `H_PREFETCH_M/V` pay the
    /// DMA inline instead of issuing in the background.
    blocking_prefetch: bool,
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
            op_stats: None,
            m_load_barrier: None,
            v_load_barrier: None,
            blocking_prefetch: false,
        }
    }

    /// Calibration mode: `H_PREFETCH_M/V` block until the DMA lands instead of
    /// issuing in the background, so `--op-stats` charges the exact DMA service
    /// time to the prefetch instruction itself.
    pub(crate) fn set_blocking_prefetch(&mut self, blocking: bool) {
        self.blocking_prefetch = blocking;
    }

    /// Block until the in-flight `H_PREFETCH_M` DMA (if any) has landed in
    /// Matrix SRAM. Called by matrix ops and by the next `H_PREFETCH_M`.
    pub(crate) async fn drain_m_load(&mut self) {
        if let Some(rx) = self.m_load_barrier.take() {
            let _ = rx.await;
        }
    }

    /// Block until the in-flight `H_PREFETCH_V` DMA (if any) has landed in
    /// Vector SRAM. Called by every VRAM consumer and the next `H_PREFETCH_V`.
    pub(crate) async fn drain_v_load(&mut self) {
        if let Some(rx) = self.v_load_barrier.take() {
            let _ = rx.await;
        }
    }

    /// Attach an `--op-stats` recorder; every instruction executed by
    /// [`Accelerator::do_ops`] is then sampled.
    pub(crate) fn set_op_stats(&mut self, recorder: OpStatsRecorder) {
        self.op_stats = Some(recorder);
    }

    /// Write the aggregate line and flush the `--op-stats` file, if enabled.
    pub(crate) fn finish_op_stats(&mut self) {
        if let Some(recorder) = self.op_stats.as_mut() {
            recorder.finish();
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
}
