//! Shared timing-mode and instruction timeline types.
//!
//! `legacy` preserves the original execute-and-await timing. `rtl-v1` is the
//! hazard-aware model introduced incrementally in this directory. Keeping the
//! mode explicit lets numerical regression tests continue to exercise the same
//! functional datapath while timing changes are validated independently.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use clap::ValueEnum;
use runtime::{Executor, Instant};
use serde::Serialize;

use crate::op::Opcode;
use crate::opcode_timing::{CalibrationStatus, TimingCalibrationMetadata, calibration_metadata};
use crate::profiler::opcode_mnemonic;
use crate::runtime_config::CLOCK_PERIOD_PS;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum TimingMode {
    Legacy,
    RtlV1,
}

impl TimingMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::RtlV1 => "rtl-v1",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Resource {
    HbmMatrixDma,
    HbmVectorDma,
    HbmVectorStore,
    MatrixCompute,
    MatrixWriteout,
    VectorPipeline,
    ScalarPipeline,
    ControlFrontend,
    Invalid,
}

impl Resource {
    pub(crate) fn as_key(self) -> &'static str {
        match self {
            Self::HbmMatrixDma => "hbm_matrix_dma",
            Self::HbmVectorDma => "hbm_vector_dma",
            Self::HbmVectorStore => "hbm_vector_store",
            Self::MatrixCompute => "matrix_compute",
            Self::MatrixWriteout => "matrix_writeout",
            Self::VectorPipeline => "vector_pipeline",
            Self::ScalarPipeline => "scalar_pipeline",
            Self::ControlFrontend => "control_frontend",
            Self::Invalid => "invalid",
        }
    }

    fn is_memory(self) -> bool {
        matches!(
            self,
            Self::HbmMatrixDma | Self::HbmVectorDma | Self::HbmVectorStore
        )
    }

    fn is_compute(self) -> bool {
        matches!(
            self,
            Self::MatrixCompute
                | Self::MatrixWriteout
                | Self::VectorPipeline
                | Self::ScalarPipeline
        )
    }
}

pub(crate) fn resource_for(opcode: &Opcode) -> Resource {
    match opcode {
        Opcode::H_PREFETCH_M { .. } => Resource::HbmMatrixDma,
        Opcode::H_PREFETCH_V { .. } => Resource::HbmVectorDma,
        Opcode::H_STORE_V { .. } => Resource::HbmVectorStore,
        Opcode::M_MM_WO { .. }
        | Opcode::M_BMM_WO { .. }
        | Opcode::M_MV_WO { .. }
        | Opcode::M_BMV_WO { .. } => Resource::MatrixWriteout,
        Opcode::M_MM { .. }
        | Opcode::M_TMM { .. }
        | Opcode::M_BMM { .. }
        | Opcode::M_BTMM { .. }
        | Opcode::M_MV { .. }
        | Opcode::M_TMV { .. }
        | Opcode::M_BMV { .. }
        | Opcode::M_BTMV { .. } => Resource::MatrixCompute,
        Opcode::V_ADD_VV { .. }
        | Opcode::V_ADD_VF { .. }
        | Opcode::V_SUB_VV { .. }
        | Opcode::V_SUB_VF { .. }
        | Opcode::V_MUL_VV { .. }
        | Opcode::V_MUL_VF { .. }
        | Opcode::V_EXP_V { .. }
        | Opcode::V_RECI_V { .. }
        | Opcode::V_RED_SUM { .. }
        | Opcode::V_RED_MAX { .. }
        | Opcode::V_SHIFT_V { .. } => Resource::VectorPipeline,
        Opcode::S_ADD_FP { .. }
        | Opcode::S_SUB_FP { .. }
        | Opcode::S_MAX_FP { .. }
        | Opcode::S_MUL_FP { .. }
        | Opcode::S_EXP_FP { .. }
        | Opcode::S_RECI_FP { .. }
        | Opcode::S_SQRT_FP { .. }
        | Opcode::S_LD_FP { .. }
        | Opcode::S_ST_FP { .. }
        | Opcode::S_MAP_V_FP { .. }
        | Opcode::S_ADD_INT { .. }
        | Opcode::S_ADDI_INT { .. }
        | Opcode::S_SUB_INT { .. }
        | Opcode::S_MUL_INT { .. }
        | Opcode::S_LUI_INT { .. }
        | Opcode::S_LD_INT { .. }
        | Opcode::S_ST_INT { .. } => Resource::ScalarPipeline,
        Opcode::C_SET_ADDR_REG { .. }
        | Opcode::C_SET_SCALE_REG { .. }
        | Opcode::C_SET_STRIDE_REG { .. }
        | Opcode::C_SET_V_MASK_REG { .. }
        | Opcode::C_LOOP_START { .. }
        | Opcode::C_LOOP_END { .. }
        | Opcode::C_BREAK => Resource::ControlFrontend,
        Opcode::Invalid => Resource::Invalid,
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct EventRecord {
    pub(crate) sequence: u64,
    pub(crate) pc: usize,
    pub(crate) opcode: &'static str,
    pub(crate) issue_cycle: u64,
    /// Cycle at which the in-order frontend accepts the instruction.
    pub(crate) accepted_cycle: u64,
    /// Registered replay interval after the raw hazard deasserts. Current RTL
    /// uses one `b1_pipeline_stall` cycle for every instruction that stalled.
    pub(crate) recovery_cycles: u64,
    /// Cycle at which the selected backend resource begins service. This can
    /// be later than acceptance for queued operations such as MM_WO/MV_WO.
    pub(crate) start_cycle: u64,
    /// Cycle at which all architectural outputs are safe for a dependent
    /// consumer. It may precede backend idle/completion.
    pub(crate) result_ready_cycle: u64,
    pub(crate) completion_cycle: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) first_result_ready_cycle: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) result_cadence_cycles: Option<u64>,
    pub(crate) resource: Resource,
    pub(crate) calibration_status: CalibrationStatus,
    pub(crate) rtl_supported: bool,
    pub(crate) calibration_in_domain: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stall_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) dependency: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RtlValidationStatus {
    #[default]
    Validated,
    UnsupportedOpcodes,
    OutOfCalibrationDomain,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct RtlValidationSummary {
    pub(crate) status: RtlValidationStatus,
    pub(crate) total_opcodes: u64,
    pub(crate) validated_opcodes: u64,
    pub(crate) unsupported_opcodes: u64,
    pub(crate) out_of_domain_opcodes: u64,
    pub(crate) unsupported_opcode_counts: BTreeMap<&'static str, u64>,
    pub(crate) out_of_domain_opcode_counts: BTreeMap<&'static str, u64>,
    pub(crate) unsupported_resource_cycles: BTreeMap<&'static str, u64>,
    pub(crate) out_of_domain_resource_cycles: BTreeMap<&'static str, u64>,
    /// Tail cycles that disappear if unsupported events are omitted. This is
    /// a conservative local sensitivity, not a rescheduled counterfactual.
    pub(crate) unsupported_makespan_sensitivity_cycles: u64,
}

/// Constant-space validation coverage for runs that do not request a full
/// instruction trace. Production Qwen traces contain millions of opcodes, so
/// retaining every [`EventRecord`] merely to discover an unsupported opcode is
/// unnecessary. The scheduler updates this accumulator as records are formed;
/// traced/profiled runs still retain the detailed events independently.
#[derive(Clone, Debug, Default)]
pub(crate) struct RtlValidationAccumulator {
    total_opcodes: u64,
    validated_opcodes: u64,
    unsupported_opcodes: u64,
    out_of_domain_opcodes: u64,
    unsupported_opcode_counts: BTreeMap<&'static str, u64>,
    out_of_domain_opcode_counts: BTreeMap<&'static str, u64>,
    unsupported_resource_cycles: BTreeMap<&'static str, u64>,
    out_of_domain_resource_cycles: BTreeMap<&'static str, u64>,
    makespan_cycles: u64,
    supported_makespan_cycles: u64,
}

impl RtlValidationAccumulator {
    pub(crate) fn observe(&mut self, event: &EventRecord) {
        self.total_opcodes += 1;
        self.makespan_cycles = self.makespan_cycles.max(event.completion_cycle);
        let cycles = event.completion_cycle.saturating_sub(event.start_cycle);
        if !event.rtl_supported {
            self.unsupported_opcodes += 1;
            *self
                .unsupported_opcode_counts
                .entry(event.opcode)
                .or_insert(0) += 1;
            *self
                .unsupported_resource_cycles
                .entry(event.resource.as_key())
                .or_insert(0) += cycles;
        } else {
            self.supported_makespan_cycles =
                self.supported_makespan_cycles.max(event.completion_cycle);
            if !event.calibration_in_domain {
                self.out_of_domain_opcodes += 1;
                *self
                    .out_of_domain_opcode_counts
                    .entry(event.opcode)
                    .or_insert(0) += 1;
                *self
                    .out_of_domain_resource_cycles
                    .entry(event.resource.as_key())
                    .or_insert(0) += cycles;
            } else {
                self.validated_opcodes += 1;
            }
        }
    }

    pub(crate) fn summary(&self) -> RtlValidationSummary {
        let status = if self.unsupported_opcodes > 0 {
            RtlValidationStatus::UnsupportedOpcodes
        } else if self.out_of_domain_opcodes > 0 {
            RtlValidationStatus::OutOfCalibrationDomain
        } else {
            RtlValidationStatus::Validated
        };
        RtlValidationSummary {
            status,
            total_opcodes: self.total_opcodes,
            validated_opcodes: self.validated_opcodes,
            unsupported_opcodes: self.unsupported_opcodes,
            out_of_domain_opcodes: self.out_of_domain_opcodes,
            unsupported_opcode_counts: self.unsupported_opcode_counts.clone(),
            out_of_domain_opcode_counts: self.out_of_domain_opcode_counts.clone(),
            unsupported_resource_cycles: self.unsupported_resource_cycles.clone(),
            out_of_domain_resource_cycles: self.out_of_domain_resource_cycles.clone(),
            unsupported_makespan_sensitivity_cycles: self
                .makespan_cycles
                .saturating_sub(self.supported_makespan_cycles),
        }
    }
}

#[derive(Debug)]
pub(crate) struct EventTrace {
    schema_version: u32,
    timing_mode: TimingMode,
    clock_period_ps: u32,
    timing_calibration: Option<TimingCalibrationMetadata<'static>>,
    events: Vec<EventRecord>,
}

#[derive(Serialize)]
struct EventTraceDocument<'a> {
    schema_version: u32,
    timing_mode: TimingMode,
    clock_period_ps: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    timing_calibration: Option<&'a TimingCalibrationMetadata<'static>>,
    rtl_validation: RtlValidationSummary,
    events: &'a [EventRecord],
}

#[derive(Serialize)]
struct DmaEventTraceDocument<'a> {
    schema_version: u32,
    timing_mode: TimingMode,
    clock_period_ps: u32,
    /// DMA intervals come from the serial functional executor and are then
    /// replayed by the independent rtl-v1 scoreboard. They are useful for
    /// debugging, but are not arrival-time-coupled rtl-v1 Ramulator evidence.
    dma_timing_semantics: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    timing_calibration: Option<&'a TimingCalibrationMetadata<'static>>,
    rtl_validation: RtlValidationSummary,
    events: Vec<&'a EventRecord>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct TimelineProfile {
    pub(crate) total_makespan_cycles: u64,
    pub(crate) total_makespan_picos: u64,
    pub(crate) total_makespan_ns: f64,
    pub(crate) frontend_issue_cycles: u64,
    pub(crate) stall_cycles_by_reason: BTreeMap<String, u64>,
    pub(crate) backend_wait_cycles_by_reason: BTreeMap<String, u64>,
    /// Number of instructions in each timing-calibration confidence class.
    pub(crate) timing_calibration_status_counts: BTreeMap<&'static str, u64>,
    /// Sum of event service intervals. Concurrent resources intentionally make
    /// the total exceed the program makespan.
    pub(crate) resource_work_cycles: BTreeMap<&'static str, u64>,
    pub(crate) resource_utilization: BTreeMap<&'static str, f64>,
    pub(crate) hbm_request_completion_wait_cycles: u64,
    pub(crate) memory_compute_overlap_cycles: u64,
    pub(crate) memory_compute_overlap_pct: f64,
    /// A mutually exclusive attribution obtained by following the blocking
    /// dependency chain of the event that determines the final makespan.
    pub(crate) critical_path_cycles: BTreeMap<&'static str, u64>,
    pub(crate) rtl_validation: RtlValidationSummary,
}

impl EventTrace {
    pub(crate) fn new(timing_mode: TimingMode) -> Self {
        Self {
            schema_version: 3,
            timing_mode,
            clock_period_ps: *CLOCK_PERIOD_PS,
            timing_calibration: matches!(timing_mode, TimingMode::RtlV1).then(calibration_metadata),
            events: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, record: EventRecord) {
        self.events.push(record);
    }

    pub(crate) fn events(&self) -> &[EventRecord] {
        &self.events
    }

    pub(crate) fn timing_calibration(&self) -> Option<&TimingCalibrationMetadata<'static>> {
        self.timing_calibration.as_ref()
    }

    pub(crate) fn timeline_profile(&self) -> TimelineProfile {
        timeline_profile(&self.events)
    }

    pub(crate) fn rtl_validation_summary(&self) -> RtlValidationSummary {
        rtl_validation_summary(&self.events)
    }

    pub(crate) fn write_json(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let document = EventTraceDocument {
            schema_version: self.schema_version,
            timing_mode: self.timing_mode,
            clock_period_ps: self.clock_period_ps,
            timing_calibration: self.timing_calibration.as_ref(),
            rtl_validation: self.rtl_validation_summary(),
            events: &self.events,
        };
        let json = serde_json::to_string_pretty(&document).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    pub(crate) fn write_dma_json(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let events = self
            .events
            .iter()
            .filter(|event| event.resource.is_memory())
            .collect();
        let document = DmaEventTraceDocument {
            schema_version: 2,
            timing_mode: self.timing_mode,
            clock_period_ps: self.clock_period_ps,
            dma_timing_semantics: "functional-executor-service-interval-replayed-on-rtl-v1",
            timing_calibration: self.timing_calibration.as_ref(),
            rtl_validation: self.rtl_validation_summary(),
            events,
        };
        let json = serde_json::to_string_pretty(&document).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }
}

pub(crate) fn rtl_validation_summary(events: &[EventRecord]) -> RtlValidationSummary {
    let mut accumulator = RtlValidationAccumulator::default();
    for event in events {
        accumulator.observe(event);
    }
    accumulator.summary()
}

fn merged_intervals(
    events: &[EventRecord],
    predicate: impl Fn(Resource) -> bool,
) -> Vec<(u64, u64)> {
    let mut intervals = events
        .iter()
        .filter(|event| predicate(event.resource))
        .map(|event| (event.start_cycle, event.completion_cycle))
        .collect::<Vec<_>>();
    intervals.sort_unstable();

    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (start, end) in intervals {
        if let Some(last) = merged.last_mut()
            && start <= last.1
        {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    merged
}

fn intersection_cycles(lhs: &[(u64, u64)], rhs: &[(u64, u64)]) -> u64 {
    let (mut i, mut j, mut total) = (0, 0, 0);
    while i < lhs.len() && j < rhs.len() {
        let start = lhs[i].0.max(rhs[j].0);
        let end = lhs[i].1.min(rhs[j].1);
        total += end.saturating_sub(start);
        if lhs[i].1 <= rhs[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    total
}

pub(crate) fn timeline_profile(events: &[EventRecord]) -> TimelineProfile {
    let makespan = events
        .iter()
        .map(|event| event.completion_cycle)
        .max()
        .unwrap_or(0);
    let mut stalls = BTreeMap::new();
    let mut backend_waits = BTreeMap::new();
    let mut work = BTreeMap::new();
    let mut calibration_status_counts = BTreeMap::new();
    let mut hbm_wait = 0;
    for event in events {
        if let Some(reason) = &event.stall_reason {
            let frontend_wait = event
                .accepted_cycle
                .saturating_sub(event.issue_cycle)
                .saturating_sub(event.recovery_cycles);
            if frontend_wait > 0 {
                *stalls.entry(reason.clone()).or_insert(0) += frontend_wait;
            }
            let backend_wait = event.start_cycle.saturating_sub(event.accepted_cycle);
            if backend_wait > 0 {
                *backend_waits.entry(reason.clone()).or_insert(0) += backend_wait;
            }
        }
        if event.recovery_cycles > 0 {
            *stalls.entry("pipeline_recovery".to_owned()).or_insert(0) += event.recovery_cycles;
        }
        let duration = event.completion_cycle.saturating_sub(event.start_cycle);
        *work.entry(event.resource.as_key()).or_insert(0) += duration;
        *calibration_status_counts
            .entry(event.calibration_status.as_key())
            .or_insert(0) += 1;
        if event.resource.is_memory() {
            hbm_wait += duration;
        }
    }

    let resources = events
        .iter()
        .map(|event| event.resource)
        .collect::<BTreeSet<_>>();
    let utilization = resources
        .into_iter()
        .map(|resource| {
            let busy_cycles = merged_intervals(events, |candidate| candidate == resource)
                .into_iter()
                .map(|(start, end)| end.saturating_sub(start))
                .sum::<u64>();
            (
                resource.as_key(),
                if makespan == 0 {
                    0.0
                } else {
                    busy_cycles as f64 * 100.0 / makespan as f64
                },
            )
        })
        .collect();

    let memory_intervals = merged_intervals(events, Resource::is_memory);
    let compute_intervals = merged_intervals(events, Resource::is_compute);
    let overlap = intersection_cycles(&memory_intervals, &compute_intervals);

    let by_sequence = events
        .iter()
        .map(|event| (event.sequence, event))
        .collect::<BTreeMap<_, _>>();
    let mut critical = BTreeMap::new();
    let mut issue_order = events.iter().collect::<Vec<_>>();
    issue_order.sort_unstable_by_key(|event| (event.accepted_cycle, event.sequence));

    // The scheduler issues at most one instruction per cycle. Partition that
    // in-order issue timeline into one frontend cycle per accepted instruction
    // and the gaps that blocked the next instruction. A gap belongs to the
    // dependency resource recorded by the scoreboard, not to the frontend that
    // was merely waiting for it. This produces a mutually exclusive and useful
    // memory/matrix/vector/scalar/control critical-path breakdown.
    let mut issue_cursor = 0;
    for event in issue_order {
        let blocked_cycles = event.accepted_cycle.saturating_sub(issue_cursor);
        if blocked_cycles > 0 {
            let owner = event
                .dependency
                .and_then(|sequence| by_sequence.get(&sequence))
                .map_or(Resource::ControlFrontend, |dependency| dependency.resource);
            *critical.entry(owner.as_key()).or_insert(0) += blocked_cycles;
        }

        if event.accepted_cycle < makespan {
            *critical
                .entry(Resource::ControlFrontend.as_key())
                .or_insert(0) += 1;
        }
        issue_cursor = event.accepted_cycle.saturating_add(1).min(makespan);
    }

    // Once issue stops, only backend drain can extend the makespan. Attribute
    // that tail to the event whose completion defines the final cycle.
    let drain_cycles = makespan.saturating_sub(issue_cursor);
    if drain_cycles > 0 {
        let drain_owner = events
            .iter()
            .max_by_key(|event| (event.completion_cycle, event.sequence))
            .map_or(Resource::ControlFrontend, |event| event.resource);
        *critical.entry(drain_owner.as_key()).or_insert(0) += drain_cycles;
    }

    let clock_period_ps = u64::from(*CLOCK_PERIOD_PS);
    TimelineProfile {
        total_makespan_cycles: makespan,
        total_makespan_picos: makespan.saturating_mul(clock_period_ps),
        total_makespan_ns: makespan as f64 * clock_period_ps as f64 / 1000.0,
        frontend_issue_cycles: events.len() as u64,
        stall_cycles_by_reason: stalls,
        backend_wait_cycles_by_reason: backend_waits,
        timing_calibration_status_counts: calibration_status_counts,
        resource_work_cycles: work,
        resource_utilization: utilization,
        hbm_request_completion_wait_cycles: hbm_wait,
        memory_compute_overlap_cycles: overlap,
        memory_compute_overlap_pct: if makespan == 0 {
            0.0
        } else {
            overlap as f64 * 100.0 / makespan as f64
        },
        critical_path_cycles: critical,
        rtl_validation: rtl_validation_summary(events),
    }
}

pub(crate) fn current_cycle() -> u64 {
    (Executor::current().now() - Instant::INIT).as_picos() / u64::from(*CLOCK_PERIOD_PS)
}

pub(crate) fn completed_record(
    sequence: u64,
    pc: usize,
    opcode: &Opcode,
    issue_cycle: u64,
    start_cycle: u64,
) -> EventRecord {
    EventRecord {
        sequence,
        pc,
        opcode: opcode_mnemonic(opcode),
        issue_cycle,
        accepted_cycle: start_cycle,
        recovery_cycles: 0,
        start_cycle,
        result_ready_cycle: current_cycle(),
        completion_cycle: current_cycle(),
        first_result_ready_cycle: None,
        result_cadence_cycles: None,
        resource: resource_for(opcode),
        calibration_status: CalibrationStatus::LegacyFunctionalObserved,
        rtl_supported: true,
        calibration_in_domain: false,
        stall_reason: None,
        dependency: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(sequence: u64, start: u64, completion: u64, resource: Resource) -> EventRecord {
        EventRecord {
            sequence,
            pc: sequence as usize,
            opcode: "test",
            issue_cycle: start,
            accepted_cycle: start,
            recovery_cycles: 0,
            start_cycle: start,
            result_ready_cycle: completion,
            completion_cycle: completion,
            first_result_ready_cycle: None,
            result_cadence_cycles: None,
            resource,
            calibration_status: CalibrationStatus::FullMachineMeasured,
            rtl_supported: true,
            calibration_in_domain: true,
            stall_reason: None,
            dependency: sequence.checked_sub(1),
        }
    }

    #[test]
    fn timeline_counts_memory_compute_overlap_without_double_counting_makespan() {
        let profile = timeline_profile(&[
            event(0, 0, 10, Resource::HbmMatrixDma),
            event(1, 2, 8, Resource::MatrixCompute),
        ]);
        assert_eq!(profile.total_makespan_cycles, 10);
        assert_eq!(profile.memory_compute_overlap_cycles, 6);
        assert_eq!(profile.resource_work_cycles["hbm_matrix_dma"], 10);
        assert_eq!(profile.resource_work_cycles["matrix_compute"], 6);
        assert_eq!(profile.critical_path_cycles.values().sum::<u64>(), 10);
    }

    #[test]
    fn overlap_scan_counts_multiple_compute_intervals_inside_one_dma() {
        let profile = timeline_profile(&[
            event(0, 0, 20, Resource::HbmMatrixDma),
            event(1, 2, 5, Resource::MatrixCompute),
            event(2, 7, 11, Resource::VectorPipeline),
            event(3, 13, 18, Resource::ScalarPipeline),
        ]);
        assert_eq!(profile.memory_compute_overlap_cycles, 12);
    }

    #[test]
    fn critical_path_partitions_issue_stall_and_drain_cycles() {
        let producer = EventRecord {
            result_ready_cycle: 10,
            completion_cycle: 15,
            dependency: None,
            ..event(0, 0, 15, Resource::MatrixWriteout)
        };
        let consumer = EventRecord {
            dependency: Some(0),
            ..event(1, 10, 20, Resource::VectorPipeline)
        };

        let profile = timeline_profile(&[producer, consumer]);
        assert_eq!(profile.total_makespan_cycles, 20);
        assert_eq!(profile.frontend_issue_cycles, 2);
        assert_eq!(profile.critical_path_cycles["control_frontend"], 2);
        assert_eq!(profile.critical_path_cycles["matrix_writeout"], 9);
        assert_eq!(profile.critical_path_cycles["vector_pipeline"], 9);
        assert_eq!(profile.critical_path_cycles.values().sum::<u64>(), 20);
    }

    #[test]
    fn unsupported_opcode_takes_precedence_over_domain_warning() {
        let unsupported = EventRecord {
            rtl_supported: false,
            calibration_in_domain: false,
            ..event(0, 0, 8, Resource::ScalarPipeline)
        };
        let out_of_domain = EventRecord {
            calibration_in_domain: false,
            ..event(1, 1, 12, Resource::MatrixCompute)
        };

        let summary = rtl_validation_summary(&[unsupported, out_of_domain]);
        assert_eq!(summary.status, RtlValidationStatus::UnsupportedOpcodes);
        assert_eq!(summary.unsupported_opcodes, 1);
        assert_eq!(summary.out_of_domain_opcodes, 1);
        assert_eq!(summary.unsupported_opcode_counts["test"], 1);
    }
}
