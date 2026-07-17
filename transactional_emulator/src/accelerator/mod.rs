//! `Accelerator` — the top-level ISA dispatcher.
//!
//! Holds the register file, the matrix/vector machines, the scalar SRAMs, the
//! HBM handle, and the loop stack; [`Accelerator::do_ops`] decodes and executes
//! an opcode stream one instruction at a time. The runner-facing state API
//! lives here; opcode execution lives in `dispatch.rs`.

use std::sync::Arc;

use memory::ErasedMemoryModel;

use crate::matrix_machine::MatrixMachine;
use crate::scheduler::RtlScheduler;
use crate::timing::{EventTrace, RtlValidationSummary, TimingMode};
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
    event_trace: Option<EventTrace>,
    event_sequence: u64,
    dma_statistics: Arc<crate::dma::DmaStatistics>,
    rtl_scheduler: Option<RtlScheduler>,
}

impl Accelerator {
    pub(crate) fn new(
        m_machine: MatrixMachine,
        v_machine: VectorMachine,
        hbm: Arc<dyn ErasedMemoryModel>,
        timing_mode: TimingMode,
        event_trace_enabled: bool,
    ) -> Self {
        Self {
            m_machine,
            v_machine,
            hbm,
            reg_file: AcceleratorRegFile::new(),
            scalar_sram: ScalarSram::new(),
            loop_state: LoopState::new(),
            event_trace: event_trace_enabled.then(|| EventTrace::new(timing_mode)),
            event_sequence: 0,
            dma_statistics: Arc::new(crate::dma::DmaStatistics::default()),
            rtl_scheduler: matches!(timing_mode, TimingMode::RtlV1).then(RtlScheduler::default),
        }
    }

    pub(crate) fn event_trace(&self) -> Option<&EventTrace> {
        self.event_trace.as_ref()
    }

    pub(crate) fn modeled_makespan_cycles(&self) -> Option<u64> {
        self.rtl_scheduler
            .as_ref()
            .map(RtlScheduler::makespan_cycles)
    }

    pub(crate) fn rtl_validation_summary(&self) -> Option<RtlValidationSummary> {
        self.rtl_scheduler
            .as_ref()
            .map(RtlScheduler::validation_summary)
            .or_else(|| {
                self.event_trace
                    .as_ref()
                    .map(EventTrace::rtl_validation_summary)
            })
    }

    pub(crate) fn dma_statistics(&self) -> crate::dma::DmaStatisticsSnapshot {
        self.dma_statistics.snapshot()
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
