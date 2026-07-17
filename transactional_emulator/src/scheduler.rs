//! In-order, hazard-aware logical scheduler for `rtl-v1` timing.
//!
//! Functional opcode execution remains deliberately unchanged and therefore
//! still occurs serially.  After each opcode completes functionally, this
//! scoreboard places the instruction on an independent RTL cycle timeline.
//! HBM service duration is taken from the real Ramulator completion interval;
//! compute duration comes from `opcode_timing`.  This separation lets timing
//! overlap without allowing asynchronous tensor writes to perturb the existing
//! numerical comparison path.

use crate::op::Opcode;
use crate::opcode_timing::{OpcodeTimingEstimate, calibrated_timing};
use crate::timing::{
    EventRecord, Resource, RtlValidationAccumulator, RtlValidationSummary, resource_for,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct AddressRange {
    pub(crate) start: u64,
    pub(crate) end: u64,
}

impl AddressRange {
    pub(crate) fn new(start: u64, length: u64) -> Self {
        Self {
            start,
            end: start.saturating_add(length),
        }
    }

    fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct InstructionAccesses {
    pub(crate) matrix_reads: Vec<AddressRange>,
    pub(crate) matrix_writes: Vec<AddressRange>,
    pub(crate) vector_reads: Vec<AddressRange>,
    pub(crate) vector_writes: Vec<AddressRange>,
    /// Architectural FP register dependencies. Register zero is omitted by
    /// dispatch because reads are constant-zero and writes are discarded.
    pub(crate) scalar_fp_reads: Vec<u8>,
    pub(crate) scalar_fp_writes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default)]
struct Slot {
    until: u64,
    sequence: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
struct PendingWrite {
    range: AddressRange,
    slot: Slot,
}

#[derive(Clone, Copy, Debug)]
struct BusyInterval {
    start: u64,
    end: u64,
    sequence: Option<u64>,
}

#[derive(Clone, Debug)]
pub(crate) struct RtlScheduler {
    next_issue_cycle: u64,
    makespan_cycles: u64,
    hbm_shared: Slot,
    hbm_matrix: Slot,
    hbm_vector: Slot,
    hbm_store: Slot,
    matrix_compute: Slot,
    matrix_writeout: Slot,
    vector_pipeline: Slot,
    scalar_fp_compute: Slot,
    scalar_sram: Slot,
    scalar_fp_results: [Slot; 32],
    vector_reduction_result: Slot,
    vector_element_result: Slot,
    vector_element_latency: Option<u64>,
    matrix_writes: Vec<PendingWrite>,
    vector_writes: Vec<PendingWrite>,
    vector_port_a_writes: Vec<BusyInterval>,
    validation: RtlValidationAccumulator,
}

impl Default for RtlScheduler {
    fn default() -> Self {
        Self {
            next_issue_cycle: 0,
            makespan_cycles: 0,
            hbm_shared: Slot::default(),
            hbm_matrix: Slot::default(),
            hbm_vector: Slot::default(),
            hbm_store: Slot::default(),
            matrix_compute: Slot::default(),
            matrix_writeout: Slot::default(),
            vector_pipeline: Slot::default(),
            scalar_fp_compute: Slot::default(),
            scalar_sram: Slot::default(),
            scalar_fp_results: [Slot::default(); 32],
            vector_reduction_result: Slot::default(),
            vector_element_result: Slot::default(),
            vector_element_latency: None,
            matrix_writes: Vec::new(),
            vector_writes: Vec::new(),
            vector_port_a_writes: Vec::new(),
            validation: RtlValidationAccumulator::default(),
        }
    }
}

#[derive(Clone, Debug)]
struct StartConstraint {
    cycle: u64,
    reason: Option<&'static str>,
    dependency: Option<u64>,
}

impl StartConstraint {
    fn at_issue(cycle: u64) -> Self {
        Self {
            cycle,
            reason: None,
            dependency: None,
        }
    }

    fn include_slot(&mut self, slot: Slot, reason: &'static str) {
        if slot.until > self.cycle {
            self.cycle = slot.until;
            self.reason = Some(reason);
            self.dependency = slot.sequence;
        }
    }

    fn include_pending(
        &mut self,
        reads: &[AddressRange],
        pending: &[PendingWrite],
        reason: &'static str,
    ) {
        for write in pending {
            if reads.iter().any(|read| read.overlaps(write.range)) {
                self.include_slot(write.slot, reason);
            }
        }
    }

    fn include_intervals(&mut self, intervals: &[BusyInterval], reason: &'static str) {
        // Advancing past one write pulse may land on the next pulse from a
        // pipelined vector stream, so continue until the candidate cycle is
        // outside every known interval.
        loop {
            let conflict = intervals
                .iter()
                .filter(|interval| interval.start <= self.cycle && self.cycle < interval.end)
                .max_by_key(|interval| interval.end);
            let Some(interval) = conflict else {
                break;
            };
            self.cycle = interval.end;
            self.reason = Some(reason);
            self.dependency = interval.sequence;
        }
    }
}

impl RtlScheduler {
    pub(crate) fn makespan_cycles(&self) -> u64 {
        self.makespan_cycles
    }

    pub(crate) fn validation_summary(&self) -> RtlValidationSummary {
        self.validation.summary()
    }

    pub(crate) fn schedule(
        &mut self,
        sequence: u64,
        pc: usize,
        opcode: &Opcode,
        accesses: &InstructionAccesses,
        observed_service_cycles: u64,
    ) -> EventRecord {
        let issue_cycle = self.next_issue_cycle;
        let mut accepted = StartConstraint::at_issue(issue_cycle);
        let resource = resource_for(opcode);
        let timing = calibrated_timing(opcode)
            .unwrap_or_else(|| OpcodeTimingEstimate::observed(observed_service_cycles));

        // Completed writes cannot constrain future instructions and retaining
        // them would turn long compiler traces into an O(N^2) scan.
        self.matrix_writes
            .retain(|write| write.slot.until > issue_cycle);
        self.vector_writes
            .retain(|write| write.slot.until > issue_cycle);
        self.vector_port_a_writes
            .retain(|interval| interval.end > issue_cycle);

        accepted.include_pending(
            &accesses.matrix_reads,
            &self.matrix_writes,
            "matrix_sram_operand_not_ready",
        );
        accepted.include_pending(
            &accesses.vector_reads,
            &self.vector_writes,
            "vector_sram_operand_not_ready",
        );
        accepted.include_pending(
            &accesses.matrix_writes,
            &self.matrix_writes,
            "matrix_sram_write_collision",
        );
        accepted.include_pending(
            &accesses.vector_writes,
            &self.vector_writes,
            "vector_sram_write_collision",
        );
        if uses_vector_sram_port_a(opcode) {
            accepted.include_intervals(&self.vector_port_a_writes, "vector_sram_port_a_write");
        }

        match resource {
            Resource::HbmMatrixDma => {
                accepted.include_slot(self.hbm_shared, "hbm_request_port_busy");
                accepted.include_slot(self.hbm_matrix, "matrix_dma_busy");
            }
            Resource::HbmVectorDma => {
                accepted.include_slot(self.hbm_shared, "hbm_request_port_busy");
                accepted.include_slot(self.hbm_vector, "vector_dma_busy");
                accepted.include_slot(self.matrix_compute, "matrix_vector_sram_conflict");
                accepted.include_slot(self.vector_pipeline, "vector_sram_port_busy");
            }
            Resource::HbmVectorStore => {
                accepted.include_slot(self.hbm_shared, "hbm_request_port_busy");
                accepted.include_slot(self.hbm_store, "vector_store_busy");
                accepted.include_slot(self.matrix_compute, "matrix_vector_sram_conflict");
                accepted.include_slot(self.vector_pipeline, "vector_sram_port_busy");
            }
            Resource::MatrixCompute => {
                accepted.include_slot(self.matrix_compute, "matrix_mcu_active");
                accepted.include_slot(self.matrix_writeout, "matrix_writeout_active");
                accepted.include_slot(self.hbm_vector, "vector_prefetch_in_progress");
                accepted.include_slot(self.hbm_store, "vector_store_in_progress");
            }
            Resource::MatrixWriteout => {
                // pipeline_control explicitly lets MM_WO/MV_WO enter while the
                // MCU is active. It queues behind compute in the backend, so
                // compute constrains service start below, not frontend accept.
                accepted.include_slot(self.matrix_writeout, "matrix_writeout_active");
            }
            Resource::VectorPipeline => {
                accepted.include_slot(self.vector_pipeline, "vector_pipeline_busy");
                accepted.include_slot(self.hbm_vector, "vector_prefetch_in_progress");
                accepted.include_slot(self.hbm_store, "vector_store_in_progress");
                if is_vector_scalar_broadcast(opcode) {
                    // pipeline_control.sv stalls LD_OUT_FP while *any* scalar
                    // FP ALU/SFU operation is active, even when its register is
                    // unrelated to the vector broadcast operand.
                    accepted.include_slot(self.scalar_fp_compute, "scalar_fp_compute_in_progress");
                }
                if is_vector_element_opcode(opcode)
                    && self
                        .vector_element_latency
                        .is_some_and(|latency| latency != timing.result_ready_cycles)
                {
                    // vector_machine.sv requires differently-latent element
                    // operations to retire in issue order.  Same-latency ops
                    // still use the measured initiation interval.
                    accepted
                        .include_slot(self.vector_element_result, "vector_mixed_latency_in_order");
                }
                self.include_scalar_fp_dependencies(&mut accepted, accesses);
            }
            Resource::ScalarPipeline => {
                if is_scalar_fp_opcode(opcode) {
                    // pipeline_control condition 8 applies to every scalar-FP
                    // opcode, including SRAM load/store and vector map.
                    accepted.include_slot(
                        self.vector_reduction_result,
                        "vector_reduction_result_not_ready",
                    );
                }
                if is_scalar_fp_compute(opcode) {
                    accepted.include_slot(self.scalar_fp_compute, "scalar_fp_compute_in_progress");
                } else if uses_scalar_sram(opcode) {
                    accepted.include_slot(self.scalar_sram, "scalar_fp_sram_busy");
                }
                self.include_scalar_fp_dependencies(&mut accepted, accesses);
            }
            Resource::ControlFrontend => {
                if matches!(opcode, Opcode::C_BREAK) {
                    accepted.include_slot(self.hbm_vector, "vector_prefetch_in_progress");
                }
            }
            Resource::Invalid => {}
        }

        // A held instruction does not execute on the first cycle in which the
        // raw hazard disappears. pipeline_control's b1_pipeline_stall replays
        // it into determine stage for one registered recovery cycle.
        let recovery_cycles = u64::from(accepted.cycle > issue_cycle);
        accepted.cycle = accepted.cycle.saturating_add(recovery_cycles);

        let mut service_start_cycle = accepted.cycle;
        let mut service_dependency = accepted.dependency;
        let mut service_reason = accepted.reason;
        if matches!(resource, Resource::MatrixWriteout)
            && self.matrix_compute.until > service_start_cycle
        {
            service_start_cycle = self.matrix_compute.until;
            service_dependency = self.matrix_compute.sequence;
            service_reason = Some("matrix_result_not_ready");
        }
        let completion_cycle = service_start_cycle.saturating_add(timing.resource_cycles);
        let result_ready_cycle = service_start_cycle.saturating_add(timing.result_ready_cycles);
        let resource_slot = Slot {
            until: completion_cycle,
            sequence: Some(sequence),
        };
        let result_slot = Slot {
            until: result_ready_cycle,
            sequence: Some(sequence),
        };

        match resource {
            Resource::HbmMatrixDma => {
                self.hbm_shared = resource_slot;
                self.hbm_matrix = resource_slot;
            }
            Resource::HbmVectorDma => {
                self.hbm_shared = resource_slot;
                self.hbm_vector = resource_slot;
            }
            Resource::HbmVectorStore => {
                self.hbm_shared = resource_slot;
                self.hbm_store = resource_slot;
            }
            Resource::MatrixCompute => self.matrix_compute = resource_slot,
            Resource::MatrixWriteout => self.matrix_writeout = resource_slot,
            Resource::VectorPipeline => {
                let is_reduction =
                    matches!(opcode, Opcode::V_RED_SUM { .. } | Opcode::V_RED_MAX { .. });
                // Elementwise units have an RTL-measured latency but accept a
                // new independent vector at their calibrated initiation
                // interval. Reductions remain non-pipelined.
                self.vector_pipeline = if is_reduction {
                    resource_slot
                } else {
                    Slot {
                        until: service_start_cycle
                            .saturating_add(timing.initiation_interval_cycles),
                        sequence: Some(sequence),
                    }
                };
                if is_reduction {
                    self.vector_reduction_result = result_slot;
                } else if is_vector_element_opcode(opcode) {
                    self.vector_element_result = result_slot;
                    self.vector_element_latency = Some(timing.result_ready_cycles);
                }
            }
            Resource::ScalarPipeline => {
                if is_scalar_fp_compute(opcode) {
                    self.scalar_fp_compute = resource_slot;
                } else if uses_scalar_sram(opcode) {
                    self.scalar_sram = resource_slot;
                }
            }
            Resource::ControlFrontend | Resource::Invalid => {}
        }

        for range in &accesses.matrix_writes {
            self.matrix_writes.push(PendingWrite {
                range: *range,
                slot: result_slot,
            });
        }
        for (row_index, range) in accesses.vector_writes.iter().enumerate() {
            let slot = if matches!(resource, Resource::MatrixWriteout) {
                timing
                    .first_result_ready_cycles
                    .zip(timing.result_cadence_cycles)
                    .map(|(first, cadence)| Slot {
                        until: service_start_cycle.saturating_add(
                            first.saturating_add(cadence.saturating_mul(row_index as u64)),
                        ),
                        sequence: Some(sequence),
                    })
                    .unwrap_or(result_slot)
            } else {
                result_slot
            };
            self.vector_writes.push(PendingWrite {
                range: *range,
                slot,
            });
        }
        for register in &accesses.scalar_fp_writes {
            if let Some(slot) = self.scalar_fp_results.get_mut(usize::from(*register)) {
                *slot = result_slot;
            }
        }
        if matches!(resource, Resource::VectorPipeline)
            && !matches!(opcode, Opcode::V_RED_SUM { .. } | Opcode::V_RED_MAX { .. })
            && !accesses.vector_writes.is_empty()
        {
            self.vector_port_a_writes.push(BusyInterval {
                start: result_ready_cycle,
                end: result_ready_cycle.saturating_add(1),
                sequence: Some(sequence),
            });
        }

        // Once the held instruction has completed replay and entered execute,
        // the following instruction may advance on the next cycle.
        self.next_issue_cycle = accepted.cycle.saturating_add(1);
        self.makespan_cycles = self.makespan_cycles.max(completion_cycle);

        let record = EventRecord {
            sequence,
            pc,
            opcode: crate::profiler::opcode_mnemonic(opcode),
            issue_cycle,
            accepted_cycle: accepted.cycle,
            recovery_cycles,
            start_cycle: service_start_cycle,
            result_ready_cycle,
            completion_cycle,
            first_result_ready_cycle: timing
                .first_result_ready_cycles
                .map(|cycles| service_start_cycle.saturating_add(cycles)),
            result_cadence_cycles: timing.result_cadence_cycles,
            resource,
            calibration_status: timing.calibration_status,
            rtl_supported: timing.rtl_supported,
            calibration_in_domain: timing.calibration_in_domain,
            stall_reason: service_reason.map(str::to_owned),
            dependency: service_dependency,
        };
        self.validation.observe(&record);
        record
    }

    fn include_scalar_fp_dependencies(
        &self,
        accepted: &mut StartConstraint,
        accesses: &InstructionAccesses,
    ) {
        for register in &accesses.scalar_fp_reads {
            if let Some(slot) = self.scalar_fp_results.get(usize::from(*register)) {
                accepted.include_slot(*slot, "scalar_fp_operand_not_ready");
            }
        }
        // A write-after-write collision can otherwise make a later, shorter
        // operation publish before the older producer of the same register.
        for register in &accesses.scalar_fp_writes {
            if let Some(slot) = self.scalar_fp_results.get(usize::from(*register)) {
                accepted.include_slot(*slot, "scalar_fp_write_port_busy");
            }
        }
    }
}

fn is_scalar_fp_compute(opcode: &Opcode) -> bool {
    matches!(
        opcode,
        Opcode::S_ADD_FP { .. }
            | Opcode::S_SUB_FP { .. }
            | Opcode::S_MAX_FP { .. }
            | Opcode::S_MUL_FP { .. }
            | Opcode::S_EXP_FP { .. }
            | Opcode::S_RECI_FP { .. }
            | Opcode::S_SQRT_FP { .. }
    )
}

fn is_scalar_fp_opcode(opcode: &Opcode) -> bool {
    is_scalar_fp_compute(opcode) || uses_scalar_sram(opcode)
}

fn uses_scalar_sram(opcode: &Opcode) -> bool {
    matches!(
        opcode,
        Opcode::S_LD_FP { .. } | Opcode::S_ST_FP { .. } | Opcode::S_MAP_V_FP { .. }
    )
}

fn uses_vector_sram_port_a(opcode: &Opcode) -> bool {
    // pipeline_control.sv condition 4 stalls every VectorMachine operation
    // and every MatrixMachine operation while vector SRAM port A is being
    // written.  This is a physical port conflict, so address disjointness
    // does not make the access legal.
    matches!(
        opcode,
        Opcode::M_MM { .. }
            | Opcode::M_TMM { .. }
            | Opcode::M_BMM { .. }
            | Opcode::M_BTMM { .. }
            | Opcode::M_BMM_WO { .. }
            | Opcode::M_MM_WO { .. }
            | Opcode::M_MV { .. }
            | Opcode::M_TMV { .. }
            | Opcode::M_BMV { .. }
            | Opcode::M_BTMV { .. }
            | Opcode::M_MV_WO { .. }
            | Opcode::M_BMV_WO { .. }
            | Opcode::V_ADD_VV { .. }
            | Opcode::V_ADD_VF { .. }
            | Opcode::V_SUB_VV { .. }
            | Opcode::V_SUB_VF { .. }
            | Opcode::V_MUL_VV { .. }
            | Opcode::V_MUL_VF { .. }
            | Opcode::V_EXP_V { .. }
            | Opcode::V_RECI_V { .. }
            | Opcode::V_RED_SUM { .. }
            | Opcode::V_RED_MAX { .. }
            | Opcode::V_SHIFT_V { .. }
    )
}

fn is_vector_scalar_broadcast(opcode: &Opcode) -> bool {
    matches!(
        opcode,
        Opcode::V_ADD_VF { .. } | Opcode::V_SUB_VF { .. } | Opcode::V_MUL_VF { .. }
    )
}

fn is_vector_element_opcode(opcode: &Opcode) -> bool {
    matches!(
        opcode,
        Opcode::V_ADD_VV { .. }
            | Opcode::V_ADD_VF { .. }
            | Opcode::V_SUB_VV { .. }
            | Opcode::V_SUB_VF { .. }
            | Opcode::V_MUL_VV { .. }
            | Opcode::V_MUL_VF { .. }
            | Opcode::V_EXP_V { .. }
            | Opcode::V_RECI_V { .. }
            | Opcode::V_SHIFT_V { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace_case(name: &str, events: Vec<EventRecord>) -> serde_json::Value {
        serde_json::json!({
            "name": name,
            "events": events,
        })
    }

    fn schedule(
        scheduler: &mut RtlScheduler,
        sequence: u64,
        opcode: &Opcode,
        accesses: InstructionAccesses,
        observed: u64,
    ) -> EventRecord {
        scheduler.schedule(sequence, sequence as usize, opcode, &accesses, observed)
    }

    #[test]
    fn compact_validation_summary_does_not_require_event_trace_storage() {
        let mut scheduler = RtlScheduler::default();
        schedule(
            &mut scheduler,
            0,
            &Opcode::S_MAX_FP {
                rd: 1,
                rs1: 2,
                rs2: 3,
            },
            InstructionAccesses::default(),
            1,
        );

        let summary = scheduler.validation_summary();
        assert_eq!(
            summary.status,
            crate::timing::RtlValidationStatus::UnsupportedOpcodes
        );
        assert_eq!(summary.total_opcodes, 1);
        assert_eq!(summary.unsupported_opcodes, 1);
        assert_eq!(summary.unsupported_opcode_counts["S_MAX_FP"], 1);
    }

    #[test]
    fn matrix_prefetch_can_overlap_unrelated_matrix_compute() {
        let mut scheduler = RtlScheduler::default();
        let prefetch = schedule(
            &mut scheduler,
            0,
            &Opcode::H_PREFETCH_M {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rstride: 0,
                precision: crate::op::MatrixPrecision::Weights,
            },
            InstructionAccesses {
                matrix_writes: vec![AddressRange::new(0, 1024)],
                ..Default::default()
            },
            100,
        );
        let compute = schedule(
            &mut scheduler,
            1,
            &Opcode::M_MM { rs1: 1, rs2: 2 },
            InstructionAccesses {
                matrix_reads: vec![AddressRange::new(4096, 1024)],
                ..Default::default()
            },
            1,
        );
        assert!(compute.start_cycle < prefetch.completion_cycle);
    }

    #[test]
    fn overlapping_matrix_operand_waits_for_prefetch() {
        let mut scheduler = RtlScheduler::default();
        let prefetch = schedule(
            &mut scheduler,
            0,
            &Opcode::H_PREFETCH_M {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rstride: 0,
                precision: crate::op::MatrixPrecision::Weights,
            },
            InstructionAccesses {
                matrix_writes: vec![AddressRange::new(0, 1024)],
                ..Default::default()
            },
            100,
        );
        let compute = schedule(
            &mut scheduler,
            1,
            &Opcode::M_MM { rs1: 1, rs2: 2 },
            InstructionAccesses {
                matrix_reads: vec![AddressRange::new(512, 1024)],
                ..Default::default()
            },
            1,
        );
        assert_eq!(compute.start_cycle, prefetch.completion_cycle + 1);
        assert_eq!(compute.dependency, Some(prefetch.sequence));
        assert_eq!(
            compute.stall_reason.as_deref(),
            Some("matrix_sram_operand_not_ready")
        );
    }

    #[test]
    fn vector_dma_stalls_conflicting_vector_pipeline() {
        let mut scheduler = RtlScheduler::default();
        let dma = schedule(
            &mut scheduler,
            0,
            &Opcode::H_PREFETCH_V {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rstride: 0,
                precision: crate::op::VectorPrecision::Activation,
            },
            InstructionAccesses::default(),
            40,
        );
        let vector = schedule(
            &mut scheduler,
            1,
            &Opcode::V_ADD_VV {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rmask: 0,
            },
            InstructionAccesses::default(),
            1,
        );
        assert_eq!(vector.start_cycle, dma.completion_cycle + 1);
        assert_eq!(
            vector.stall_reason.as_deref(),
            Some("vector_prefetch_in_progress")
        );
    }

    #[test]
    fn frontend_inserts_one_recovery_cycle_after_a_stall() {
        let mut scheduler = RtlScheduler::default();
        let _dma = schedule(
            &mut scheduler,
            0,
            &Opcode::H_PREFETCH_V {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rstride: 0,
                precision: crate::op::VectorPrecision::Activation,
            },
            InstructionAccesses::default(),
            20,
        );
        let blocked = schedule(
            &mut scheduler,
            1,
            &Opcode::V_ADD_VV {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rmask: 0,
            },
            InstructionAccesses::default(),
            1,
        );
        let following = schedule(
            &mut scheduler,
            2,
            &Opcode::C_SET_SCALE_REG { rd: 1 },
            InstructionAccesses::default(),
            1,
        );

        assert!(blocked.accepted_cycle > blocked.issue_cycle);
        assert_eq!(blocked.recovery_cycles, 1);
        assert_eq!(following.issue_cycle, blocked.accepted_cycle + 1);
    }

    #[test]
    fn vector_reduction_blocks_scalar_fp_load_and_map() {
        let mut scheduler = RtlScheduler::default();
        let reduction = schedule(
            &mut scheduler,
            0,
            &Opcode::V_RED_SUM {
                rd: 1,
                rs1: 2,
                rmask: 0,
            },
            InstructionAccesses::default(),
            1,
        );
        let load = schedule(
            &mut scheduler,
            1,
            &Opcode::S_LD_FP {
                rd: 3,
                rs1: 4,
                imm: 0,
            },
            InstructionAccesses::default(),
            1,
        );
        let map = schedule(
            &mut scheduler,
            2,
            &Opcode::S_MAP_V_FP {
                rd: 2,
                rs1: 3,
                imm: 0,
            },
            InstructionAccesses::default(),
            1,
        );

        assert_eq!(load.start_cycle, reduction.result_ready_cycle + 1);
        assert_eq!(
            load.stall_reason.as_deref(),
            Some("vector_reduction_result_not_ready")
        );
        assert!(map.start_cycle >= reduction.result_ready_cycle);
    }

    #[test]
    fn different_latency_vector_ops_retire_in_issue_order() {
        let mut scheduler = RtlScheduler::default();
        let add = schedule(
            &mut scheduler,
            0,
            &Opcode::V_ADD_VV {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rmask: 0,
            },
            InstructionAccesses::default(),
            1,
        );
        let multiply = schedule(
            &mut scheduler,
            1,
            &Opcode::V_MUL_VV {
                rd: 4,
                rs1: 5,
                rs2: 6,
                rmask: 0,
            },
            InstructionAccesses::default(),
            1,
        );

        assert_eq!(multiply.start_cycle, add.result_ready_cycle + 1);
        assert_eq!(
            multiply.stall_reason.as_deref(),
            Some("vector_mixed_latency_in_order")
        );
    }

    #[test]
    fn unrelated_vector_dma_and_matrix_writeout_can_overlap() {
        let mut scheduler = RtlScheduler::default();
        let writeout = schedule(
            &mut scheduler,
            0,
            &Opcode::M_MM_WO {
                rd: 1,
                rstride: 0,
                imm: 0,
            },
            InstructionAccesses {
                vector_writes: vec![AddressRange::new(0, 64)],
                ..Default::default()
            },
            1,
        );
        let dma = schedule(
            &mut scheduler,
            1,
            &Opcode::H_PREFETCH_V {
                rd: 2,
                rs1: 3,
                rs2: 4,
                rstride: 0,
                precision: crate::op::VectorPrecision::Activation,
            },
            InstructionAccesses {
                vector_writes: vec![AddressRange::new(4096, 64)],
                ..Default::default()
            },
            40,
        );

        assert!(dma.start_cycle < writeout.completion_cycle);
    }

    #[test]
    fn same_vector_destination_waits_for_older_pipelined_write() {
        let mut scheduler = RtlScheduler::default();
        let first = schedule(
            &mut scheduler,
            0,
            &Opcode::V_ADD_VV {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rmask: 0,
            },
            InstructionAccesses {
                vector_writes: vec![AddressRange::new(0, 64)],
                ..Default::default()
            },
            1,
        );
        let second = schedule(
            &mut scheduler,
            1,
            &Opcode::V_MUL_VV {
                rd: 1,
                rs1: 4,
                rs2: 5,
                rmask: 0,
            },
            InstructionAccesses {
                vector_writes: vec![AddressRange::new(0, 64)],
                ..Default::default()
            },
            1,
        );

        // The result becomes ready on the same cycle that it writes vector
        // SRAM port A. The dependent op therefore waits through that physical
        // write pulse and then incurs the registered recovery cycle.
        assert_eq!(second.start_cycle, first.result_ready_cycle + 2);
        assert_eq!(
            second.stall_reason.as_deref(),
            Some("vector_sram_port_a_write")
        );
    }

    #[test]
    fn matrix_writeout_is_accepted_then_waits_in_backend() {
        let mut scheduler = RtlScheduler::default();
        let compute = schedule(
            &mut scheduler,
            0,
            &Opcode::M_MM { rs1: 1, rs2: 2 },
            InstructionAccesses::default(),
            1,
        );
        let writeout = schedule(
            &mut scheduler,
            1,
            &Opcode::M_MM_WO {
                rd: 1,
                rstride: 0,
                imm: 0,
            },
            InstructionAccesses::default(),
            1,
        );

        assert_eq!(writeout.accepted_cycle, writeout.issue_cycle);
        assert_eq!(writeout.start_cycle, compute.completion_cycle);
        assert_eq!(writeout.dependency, Some(compute.sequence));
    }

    #[test]
    fn independent_vector_ops_use_one_cycle_initiation_interval() {
        let mut scheduler = RtlScheduler::default();
        let first = schedule(
            &mut scheduler,
            0,
            &Opcode::V_RECI_V {
                rd: 1,
                rs1: 2,
                rmask: 0,
            },
            InstructionAccesses {
                vector_reads: vec![AddressRange::new(0, 64)],
                vector_writes: vec![AddressRange::new(64, 64)],
                ..Default::default()
            },
            1,
        );
        let second = schedule(
            &mut scheduler,
            1,
            &Opcode::V_RECI_V {
                rd: 3,
                rs1: 4,
                rmask: 0,
            },
            InstructionAccesses {
                vector_reads: vec![AddressRange::new(256, 64)],
                vector_writes: vec![AddressRange::new(320, 64)],
                ..Default::default()
            },
            1,
        );

        assert_eq!(second.start_cycle, first.start_cycle + 1);
        assert!(second.start_cycle < first.completion_cycle);
    }

    #[test]
    fn scalar_sram_op_on_another_register_can_overlap_fp_compute() {
        let mut scheduler = RtlScheduler::default();
        let producer = schedule(
            &mut scheduler,
            0,
            &Opcode::S_EXP_FP { rd: 1, rs1: 2 },
            InstructionAccesses {
                scalar_fp_reads: vec![2],
                scalar_fp_writes: vec![1],
                ..Default::default()
            },
            1,
        );
        let load = schedule(
            &mut scheduler,
            1,
            &Opcode::S_LD_FP {
                rd: 3,
                rs1: 4,
                imm: 0,
            },
            InstructionAccesses {
                scalar_fp_writes: vec![3],
                ..Default::default()
            },
            1,
        );

        assert!(load.start_cycle < producer.completion_cycle);
    }

    #[test]
    fn scalar_store_waits_for_its_specific_fp_operand() {
        let mut scheduler = RtlScheduler::default();
        let producer = schedule(
            &mut scheduler,
            0,
            &Opcode::S_RECI_FP { rd: 5, rs1: 4 },
            InstructionAccesses {
                scalar_fp_reads: vec![4],
                scalar_fp_writes: vec![5],
                ..Default::default()
            },
            1,
        );
        let store = schedule(
            &mut scheduler,
            1,
            &Opcode::S_ST_FP {
                rd: 5,
                rs1: 1,
                imm: 0,
            },
            InstructionAccesses {
                scalar_fp_reads: vec![5],
                ..Default::default()
            },
            1,
        );

        assert_eq!(store.start_cycle, producer.result_ready_cycle + 1);
        assert_eq!(store.dependency, Some(producer.sequence));
        assert_eq!(
            store.stall_reason.as_deref(),
            Some("scalar_fp_operand_not_ready")
        );
    }

    #[test]
    fn scalar_sfu_blocks_vector_scalar_broadcast_until_backend_idle() {
        let mut scheduler = RtlScheduler::default();
        let producer = schedule(
            &mut scheduler,
            0,
            &Opcode::S_EXP_FP { rd: 1, rs1: 2 },
            InstructionAccesses {
                scalar_fp_reads: vec![2],
                scalar_fp_writes: vec![1],
                ..Default::default()
            },
            1,
        );
        let broadcast = schedule(
            &mut scheduler,
            1,
            &Opcode::V_MUL_VF {
                rd: 3,
                rs1: 4,
                rs2: 5,
                rmask: 0,
            },
            InstructionAccesses {
                vector_reads: vec![AddressRange::new(0, 64)],
                vector_writes: vec![AddressRange::new(64, 64)],
                scalar_fp_reads: vec![5],
                ..Default::default()
            },
            1,
        );

        assert_eq!(broadcast.start_cycle, producer.completion_cycle + 1);
        assert_eq!(broadcast.dependency, Some(producer.sequence));
        assert_eq!(
            broadcast.stall_reason.as_deref(),
            Some("scalar_fp_compute_in_progress")
        );
    }

    #[test]
    fn vector_port_a_write_pulses_block_disjoint_vector_reads() {
        let mut scheduler = RtlScheduler::default();
        let mut events = Vec::new();
        for sequence in 0..12 {
            events.push(schedule(
                &mut scheduler,
                sequence,
                &Opcode::V_ADD_VV {
                    rd: 1,
                    rs1: 2,
                    rs2: 3,
                    rmask: 0,
                },
                InstructionAccesses {
                    vector_reads: vec![AddressRange::new(4096 + sequence * 256, 64)],
                    vector_writes: vec![AddressRange::new(8192 + sequence * 256, 64)],
                    ..Default::default()
                },
                1,
            ));
        }

        // The thirteenth ADD reaches determine when the first ADD writes
        // vector SRAM port A. Twelve contiguous result pulses occupy cycles
        // [12, 24), then the RTL inserts one recovery cycle before execute.
        let blocked = schedule(
            &mut scheduler,
            12,
            &Opcode::V_ADD_VV {
                rd: 4,
                rs1: 5,
                rs2: 6,
                rmask: 0,
            },
            InstructionAccesses {
                vector_reads: vec![AddressRange::new(16384, 64)],
                vector_writes: vec![AddressRange::new(16640, 64)],
                ..Default::default()
            },
            1,
        );

        assert_eq!(events[0].result_ready_cycle, 12);
        assert_eq!(events[11].result_ready_cycle, 23);
        assert_eq!(blocked.issue_cycle, 12);
        assert_eq!(blocked.start_cycle, 25);
        assert_eq!(blocked.recovery_cycles, 1);
        assert_eq!(blocked.dependency, Some(11));
        assert_eq!(
            blocked.stall_reason.as_deref(),
            Some("vector_sram_port_a_write")
        );
    }

    #[test]
    fn matrix_output_read_waits_for_backend_idle_without_row_valid() {
        let mut scheduler = RtlScheduler::default();
        let _compute = schedule(
            &mut scheduler,
            0,
            &Opcode::M_MM { rs1: 1, rs2: 2 },
            InstructionAccesses::default(),
            1,
        );
        let output = AddressRange::new(0, u64::from(*crate::runtime_config::MLEN));
        let writeout = schedule(
            &mut scheduler,
            1,
            &Opcode::M_MM_WO {
                rd: 1,
                rstride: 0,
                imm: 0,
            },
            InstructionAccesses {
                vector_writes: vec![output],
                ..Default::default()
            },
            1,
        );
        let consumer = schedule(
            &mut scheduler,
            2,
            &Opcode::V_ADD_VV {
                rd: 3,
                rs1: 1,
                rs2: 2,
                rmask: 0,
            },
            InstructionAccesses {
                vector_reads: vec![output],
                ..Default::default()
            },
            1,
        );

        assert_eq!(consumer.start_cycle, writeout.result_ready_cycle + 1);
        assert_eq!(writeout.result_ready_cycle, writeout.completion_cycle);
        assert_eq!(consumer.start_cycle, writeout.completion_cycle + 1);
    }

    #[test]
    #[ignore = "writes an evidence artifact when RTL_SCHEDULER_TRACE_OUT is set"]
    fn emit_rtl_scheduler_validation_trace() {
        let output = std::env::var_os("RTL_SCHEDULER_TRACE_OUT")
            .map(std::path::PathBuf::from)
            .expect("set RTL_SCHEDULER_TRACE_OUT to the target JSON path");
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }

        let mut cases = Vec::new();

        let mut scheduler = RtlScheduler::default();
        let matrix_output = AddressRange::new(0, u64::from(*crate::runtime_config::MLEN));
        let matrix = schedule(
            &mut scheduler,
            0,
            &Opcode::M_MM { rs1: 1, rs2: 2 },
            InstructionAccesses::default(),
            1,
        );
        let writeout = schedule(
            &mut scheduler,
            1,
            &Opcode::M_MM_WO {
                rd: 1,
                rstride: 0,
                imm: 0,
            },
            InstructionAccesses {
                vector_writes: vec![matrix_output],
                ..Default::default()
            },
            1,
        );
        let row_consumer = schedule(
            &mut scheduler,
            2,
            &Opcode::V_ADD_VV {
                rd: 3,
                rs1: 1,
                rs2: 2,
                rmask: 0,
            },
            InstructionAccesses {
                vector_reads: vec![matrix_output],
                vector_writes: vec![AddressRange::new(4096, 64)],
                ..Default::default()
            },
            1,
        );
        cases.push(trace_case(
            "matrix_compute_writeout_row_consumer",
            vec![matrix, writeout, row_consumer],
        ));

        let mut scheduler = RtlScheduler::default();
        let reduction = schedule(
            &mut scheduler,
            0,
            &Opcode::V_RED_SUM {
                rd: 1,
                rs1: 2,
                rmask: 0,
            },
            InstructionAccesses {
                scalar_fp_writes: vec![1],
                ..Default::default()
            },
            1,
        );
        let scalar = schedule(
            &mut scheduler,
            1,
            &Opcode::S_LD_FP {
                rd: 3,
                rs1: 4,
                imm: 0,
            },
            InstructionAccesses {
                scalar_fp_writes: vec![3],
                ..Default::default()
            },
            1,
        );
        cases.push(trace_case(
            "vector_reduction_to_scalar_fp",
            vec![reduction, scalar],
        ));

        let mut scheduler = RtlScheduler::default();
        let sfu = schedule(
            &mut scheduler,
            0,
            &Opcode::S_EXP_FP { rd: 1, rs1: 2 },
            InstructionAccesses {
                scalar_fp_writes: vec![1],
                ..Default::default()
            },
            1,
        );
        let broadcast = schedule(
            &mut scheduler,
            1,
            &Opcode::V_MUL_VF {
                rd: 3,
                rs1: 4,
                rs2: 5,
                rmask: 0,
            },
            InstructionAccesses {
                scalar_fp_reads: vec![5],
                vector_writes: vec![AddressRange::new(64, 64)],
                ..Default::default()
            },
            1,
        );
        cases.push(trace_case(
            "scalar_sfu_to_vector_broadcast",
            vec![sfu, broadcast],
        ));

        let mut scheduler = RtlScheduler::default();
        let dma = schedule(
            &mut scheduler,
            0,
            &Opcode::H_PREFETCH_V {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rstride: 0,
                precision: crate::op::VectorPrecision::Activation,
            },
            InstructionAccesses::default(),
            20,
        );
        let after_dma = schedule(
            &mut scheduler,
            1,
            &Opcode::V_ADD_VV {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rmask: 0,
            },
            InstructionAccesses {
                vector_writes: vec![AddressRange::new(0, 64)],
                ..Default::default()
            },
            1,
        );
        cases.push(trace_case(
            "hbm_vector_prefetch_to_vector_compute",
            vec![dma, after_dma],
        ));

        let mut scheduler = RtlScheduler::default();
        let first = schedule(
            &mut scheduler,
            0,
            &Opcode::V_ADD_VV {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rmask: 0,
            },
            InstructionAccesses {
                vector_writes: vec![AddressRange::new(0, 64)],
                ..Default::default()
            },
            1,
        );
        let dependent = schedule(
            &mut scheduler,
            1,
            &Opcode::V_ADD_VV {
                rd: 2,
                rs1: 1,
                rs2: 4,
                rmask: 0,
            },
            InstructionAccesses {
                vector_reads: vec![AddressRange::new(0, 64)],
                vector_writes: vec![AddressRange::new(128, 64)],
                ..Default::default()
            },
            1,
        );
        cases.push(trace_case("stall_recovery_reissue", vec![first, dependent]));

        let mut scheduler = RtlScheduler::default();
        let add = schedule(
            &mut scheduler,
            0,
            &Opcode::V_ADD_VV {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rmask: 0,
            },
            InstructionAccesses {
                vector_writes: vec![AddressRange::new(0, 64)],
                ..Default::default()
            },
            1,
        );
        let multiply = schedule(
            &mut scheduler,
            1,
            &Opcode::V_MUL_VV {
                rd: 4,
                rs1: 5,
                rs2: 6,
                rmask: 0,
            },
            InstructionAccesses {
                vector_writes: vec![AddressRange::new(128, 64)],
                ..Default::default()
            },
            1,
        );
        cases.push(trace_case(
            "mixed_vector_latency_in_order",
            vec![add, multiply],
        ));

        let artifact = serde_json::json!({
            "schema_version": 1,
            "model": "rtl_v1_hazard_aware_scheduler",
            "clock_period_ps": *crate::runtime_config::CLOCK_PERIOD_PS,
            "cases": cases,
        });
        std::fs::write(
            output,
            serde_json::to_string_pretty(&artifact).unwrap() + "\n",
        )
        .unwrap();
    }
}
