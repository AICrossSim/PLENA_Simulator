//! Off-by-default matrix-architecture experiments for Shared-MoE.
//!
//! Legacy modes retain the original post-run matrix-cycle replacement. The
//! request-replay modes instead run a baseline and candidate on isolated
//! simulation executors. Each replay owns a fresh eight-channel Ramulator and
//! issues the same physical 64-byte MX weight bursts. This keeps experimental
//! traffic out of the functional run while preserving request ordering,
//! contention, finite weight-buffer depth, compute overlap, and completion
//! skew.
//!
//! Candidate compute timers transfer the existing 2048x1408 calibration by
//! the actual tiled M_MM work; they are not RTL-measured for every model shape.
//! Request replay is therefore stronger than the legacy arithmetic overlay,
//! but it is still not RTL sign-off for a new matrix geometry.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque};
use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::{Arc, Mutex};

use clap::ValueEnum;
use futures_util::future::join_all;
use memory::{ErasedMemoryModel, MemoryModel};
use runtime::{Executor, Instant};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::runtime_config::PERIOD;

const BASELINE_ROWS: u64 = 4;
const BASELINE_COLS: u64 = 1024;
const BASELINE_PES: u64 = BASELINE_ROWS * BASELINE_COLS;
const COMPILER_MLEN: u64 = 64;
const COMPILER_BLEN: u64 = 4;
const MX_BLOCK: u64 = 8;
const HBM_BURST_BYTES: u64 = 64;
const HBM_CHANNELS: usize = 8;
const HBM_ISSUE_BATCH: usize = 4096;
const WEIGHT_TILE_N: u64 = 1024;
const WEIGHT_TILE_STORED_BYTES: u64 = 73_728;

const ROUTED_MATRIX_WAVE_CYCLES: u64 = 2_171_392;
const SHARED_MATRIX_WAVE_CYCLES: u64 = 2_171_136;
// The two measured wave constants above came from the 2048x1408
// Shared-MoE shape.  A request replay may use a different model shape, so the
// calibration must scale with the actual tiled M_MM work instead of treating
// 2,171,392 cycles as a shape-independent opcode cost.
const CALIBRATION_HIDDEN: u64 = 2048;
const CALIBRATION_INTERMEDIATE: u64 = 1408;
const CALIBRATION_MM_OPS_PER_WAVE: u64 =
    3 * (CALIBRATION_HIDDEN / COMPILER_MLEN) * (CALIBRATION_INTERMEDIATE / COMPILER_BLEN);
const PARTITION_LAUNCH_CYCLES: u64 = 128;
const COMPLETION_CYCLES: u64 = 32;
const VECTOR_REDUCE_WIDTH: u64 = 64;
const INTERCONNECT_BYTES_PER_CYCLE: u64 = 64;
const SYNC_CYCLES_PER_PART: u64 = 32;
const ONLINE_DISPATCH_CYCLES: u64 = 8;
const ONLINE_CROSSBAR_BYTES_PER_CYCLE: u64 = 64;

static REPORT_CYCLES: Mutex<Option<u64>> = Mutex::new(None);

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum ExperimentalMatrixArchitecture {
    Baseline,
    FixedBigTail,
    GangableRowSlices,
    #[value(name = "dual-2x1024-nsplit-depth2")]
    Dual2x1024NsplitDepth2,
    #[value(name = "dual-2x1024-k-split-depth2")]
    Dual2x1024KsplitDepth2,
    #[value(name = "dual-2x1024-online-stream-k-depth2")]
    Dual2x1024OnlineStreamKDepth2,
    #[value(name = "asym-2-1-1-stream-k-depth3")]
    Asym211StreamKDepth3,
    #[value(name = "asym-2-1-1-online-stream-k-depth3")]
    Asym211OnlineStreamKDepth3,
    #[value(name = "asym-2-1-1-k-split-depth3")]
    Asym211KsplitDepth3,
    #[value(name = "quad-1x1024-m-split-depth4")]
    Quad1x1024MsplitDepth4,
    #[value(name = "quad-1x1024-n-split-depth4")]
    Quad1x1024NsplitDepth4,
    #[value(name = "quad-1x1024-k-split-depth4")]
    Quad1x1024KsplitDepth4,
    #[value(name = "quad-1x1024-stage-n-to-k-depth4")]
    Quad1x1024StageNToKDepth4,
    #[value(name = "quad-1x1024-stream-k-depth4")]
    Quad1x1024StreamKDepth4,
    #[value(name = "quad-1x1024-online-stream-k-depth4")]
    Quad1x1024OnlineStreamKDepth4,
}

impl ExperimentalMatrixArchitecture {
    fn is_request_replay(self) -> bool {
        matches!(
            self,
            Self::Dual2x1024NsplitDepth2
                | Self::Dual2x1024KsplitDepth2
                | Self::Dual2x1024OnlineStreamKDepth2
                | Self::Asym211StreamKDepth3
                | Self::Asym211OnlineStreamKDepth3
                | Self::Asym211KsplitDepth3
                | Self::Quad1x1024MsplitDepth4
                | Self::Quad1x1024NsplitDepth4
                | Self::Quad1x1024KsplitDepth4
                | Self::Quad1x1024StageNToKDepth4
                | Self::Quad1x1024StreamKDepth4
                | Self::Quad1x1024OnlineStreamKDepth4
        )
    }
}

#[derive(Debug, Deserialize)]
struct GroupedManifest {
    model: String,
    architecture: String,
    #[serde(default)]
    workload: String,
    #[serde(default)]
    layer: u64,
    #[serde(default)]
    step: u64,
    batch: u64,
    top_k: u64,
    num_experts: u64,
    hidden: u64,
    routed_intermediate: u64,
    shared_intermediate: u64,
    #[serde(default)]
    selected_experts: Vec<u64>,
    group_sizes: BTreeMap<String, u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum JobKind {
    Shared,
    Routed,
}

#[derive(Clone, Debug)]
struct Projection {
    name: &'static str,
    k: u64,
    n: u64,
    base: u64,
}

#[derive(Clone, Debug)]
struct LogicalJob {
    id: usize,
    kind: JobKind,
    expert_id: Option<u64>,
    tokens: u64,
    projections: [Projection; 3],
}

#[derive(Clone, Copy, Debug)]
enum ReplayArchitecture {
    Baseline { buffer_depth: usize },
    DualNsplitDepth2,
    DualKsplitDepth2,
    DualOnlineStreamKDepth2,
    Asym211StreamKDepth3,
    Asym211OnlineStreamKDepth3,
    Asym211KsplitDepth3,
    QuadMsplitDepth4,
    QuadNsplitDepth4,
    QuadKsplitDepth4,
    QuadStageNToKDepth4,
    QuadStreamKDepth4,
    QuadOnlineStreamKDepth4,
}

impl ReplayArchitecture {
    fn name(self) -> &'static str {
        match self {
            Self::Baseline { buffer_depth: 2 } => "baseline_4x1024_depth2",
            Self::Baseline { buffer_depth: 3 } => "baseline_4x1024_depth3",
            Self::Baseline { buffer_depth: 4 } => "baseline_4x1024_depth4",
            Self::Baseline { buffer_depth } => {
                panic!("unsupported request-replay baseline depth {buffer_depth}")
            }
            Self::DualNsplitDepth2 => "dual_2x1024_nsplit_depth2",
            Self::DualKsplitDepth2 => "dual_2x1024_k_split_depth2",
            Self::DualOnlineStreamKDepth2 => "dual_2x1024_online_stream_k_depth2",
            Self::Asym211StreamKDepth3 => "asym_2_1_1_stream_k_depth3",
            Self::Asym211OnlineStreamKDepth3 => "asym_2_1_1_online_stream_k_depth3",
            Self::Asym211KsplitDepth3 => "asym_2_1_1_k_split_depth3",
            Self::QuadMsplitDepth4 => "quad_1x1024_m_split_depth4",
            Self::QuadNsplitDepth4 => "quad_1x1024_n_split_depth4",
            Self::QuadKsplitDepth4 => "quad_1x1024_k_split_depth4",
            Self::QuadStageNToKDepth4 => "quad_1x1024_stage_n_to_k_depth4",
            Self::QuadStreamKDepth4 => "quad_1x1024_stream_k_depth4",
            Self::QuadOnlineStreamKDepth4 => "quad_1x1024_online_stream_k_depth4",
        }
    }

    fn buffer_depth(self) -> usize {
        match self {
            Self::Baseline { buffer_depth } => buffer_depth,
            Self::DualNsplitDepth2 => 2,
            Self::DualKsplitDepth2 => 2,
            Self::DualOnlineStreamKDepth2 => 2,
            Self::Asym211StreamKDepth3 => 3,
            Self::Asym211OnlineStreamKDepth3 => 3,
            Self::Asym211KsplitDepth3 => 3,
            Self::QuadMsplitDepth4
            | Self::QuadNsplitDepth4
            | Self::QuadKsplitDepth4
            | Self::QuadStageNToKDepth4
            | Self::QuadStreamKDepth4 => 4,
            Self::QuadOnlineStreamKDepth4 => 4,
        }
    }

    fn partition_count(self) -> usize {
        match self {
            Self::Baseline { .. } => 1,
            Self::DualNsplitDepth2 | Self::DualKsplitDepth2 | Self::DualOnlineStreamKDepth2 => 2,
            Self::Asym211StreamKDepth3
            | Self::Asym211OnlineStreamKDepth3
            | Self::Asym211KsplitDepth3 => 3,
            Self::QuadMsplitDepth4
            | Self::QuadNsplitDepth4
            | Self::QuadKsplitDepth4
            | Self::QuadStageNToKDepth4
            | Self::QuadStreamKDepth4
            | Self::QuadOnlineStreamKDepth4 => 4,
        }
    }

    fn core_rows(self) -> &'static [u64] {
        match self {
            Self::Baseline { .. } => &[4],
            Self::DualNsplitDepth2 | Self::DualKsplitDepth2 | Self::DualOnlineStreamKDepth2 => {
                &[2, 2]
            }
            Self::Asym211StreamKDepth3
            | Self::Asym211OnlineStreamKDepth3
            | Self::Asym211KsplitDepth3 => &[2, 1, 1],
            Self::QuadMsplitDepth4
            | Self::QuadNsplitDepth4
            | Self::QuadKsplitDepth4
            | Self::QuadStageNToKDepth4
            | Self::QuadStreamKDepth4
            | Self::QuadOnlineStreamKDepth4 => &[1, 1, 1, 1],
        }
    }

    fn mapping(self) -> &'static str {
        match self {
            Self::Baseline { .. } => "monolithic",
            Self::DualNsplitDepth2 => "uniform_n_split_with_intermediate_exchange",
            Self::DualKsplitDepth2 | Self::Asym211KsplitDepth3 => {
                "static_contiguous_k_split_with_projection_reduction"
            }
            Self::DualOnlineStreamKDepth2 => {
                "two_way_completion_driven_online_stream_k_with_tagged_reduction"
            }
            Self::Asym211StreamKDepth3 => "work_centric_stream_k_with_split_output_tile_reduction",
            Self::Asym211OnlineStreamKDepth3 => {
                "completion_driven_online_stream_k_with_tagged_reduction"
            }
            Self::QuadMsplitDepth4 => "four_way_token_row_split_with_weight_multicast",
            Self::QuadNsplitDepth4 => "four_way_n_split_with_intermediate_exchange",
            Self::QuadKsplitDepth4 => {
                "four_way_static_contiguous_k_split_with_projection_reduction"
            }
            Self::QuadStageNToKDepth4 => "four_way_gate_up_n_split_then_down_k_reduction",
            Self::QuadStreamKDepth4 => {
                "four_way_work_centric_stream_k_with_split_output_tile_reduction"
            }
            Self::QuadOnlineStreamKDepth4 => {
                "four_way_completion_driven_online_stream_k_with_tagged_reduction"
            }
        }
    }

    fn is_online_scheduler(self) -> bool {
        matches!(
            self,
            Self::DualOnlineStreamKDepth2
                | Self::Asym211OnlineStreamKDepth3
                | Self::QuadOnlineStreamKDepth4
        )
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ComputeBreakdown {
    compute_cycles: u64,
    finalize_cycles: u64,
    aggregate_core_busy_cycles: u64,
    aggregate_feed_cycles: u64,
    aggregate_fixed_cycles: u64,
    aggregate_launch_cycles: u64,
    aggregate_partition_wait_cycles: u64,
    max_partition_skew_cycles: u64,
    reducer_cycles: u64,
    interconnect_cycles: u64,
    synchronization_cycles: u64,
    interconnect_bytes: u64,
    reduction_output_elements: u64,
    reduction_add_elements: u64,
    // Backward-compatible alias for reduction_output_elements.
    reduction_elements: u64,
}

#[derive(Debug, Serialize)]
struct RequestReplayRun {
    architecture: String,
    mapping: &'static str,
    core_rows: Vec<u64>,
    buffer_depth: usize,
    buffer_unit: &'static str,
    max_weight_buffer_bytes: u64,
    weight_buffer_banks: usize,
    matrix_read_ports: usize,
    matrix_write_ports: usize,
    equal_area_pes: u64,
    completed_jobs: usize,
    total_cycles: u64,
    executor_drained_cycles: u64,
    accounted_controller_cycles: u64,
    cycle_accounting_passed: bool,
    hbm_physical_bytes: u64,
    hbm_read_requests_64b: u64,
    hbm_last_completion_cycle: u64,
    hbm_starved_cycles: u64,
    compute_wall_cycles: u64,
    aggregate_core_busy_cycles: u64,
    aggregate_feed_cycles: u64,
    aggregate_fixed_cycles: u64,
    aggregate_launch_cycles: u64,
    aggregate_partition_wait_cycles: u64,
    max_partition_skew_cycles: u64,
    reducer_busy_cycles: u64,
    interconnect_cycles: u64,
    synchronization_cycles: u64,
    completion_overhead_cycles: u64,
    interconnect_bytes: u64,
    reduction_output_elements: u64,
    reduction_add_elements: u64,
    // Backward-compatible alias for reduction_output_elements.
    reduction_elements: u64,
    // Backward-compatible alias; now correctly means partition/core skew.
    completion_skew_cycles: u64,
    job_completion_span_cycles: u64,
    scheduler_kind: &'static str,
    scheduler_dispatch_cycles: u64,
    online_work_items: u64,
    online_completion_events: u64,
    online_work_steal_events: u64,
    max_ready_queue_depth: usize,
    scoreboard_output_entries: u64,
    core_busy_cycles_by_core: Vec<u64>,
    core_completed_work_items: Vec<u64>,
    aggregate_partial_sum_wait_cycles: u64,
    max_partial_sum_lifetime_cycles: u64,
    aggregate_core_idle_cycles: u64,
    crossbar_busy_cycles: u64,
    crossbar_queue_wait_cycles: u64,
    reducer_queue_wait_cycles: u64,
    bank_port_stall_cycles: u64,
}

#[derive(Debug, Serialize)]
struct RequestReplayComparison {
    timing_source: &'static str,
    address_source: &'static str,
    physical_byte_contract: &'static str,
    tensor_partition_contract: &'static str,
    baseline: RequestReplayRun,
    candidate: RequestReplayRun,
    speedup_request_region: f64,
    hbm_byte_equality_passed: bool,
    completed_job_equality_passed: bool,
    equal_pe_count_passed: bool,
    equal_weight_buffer_bytes_passed: bool,
    tensor_slice_alignment_passed: bool,
    request_block_coverage_passed: bool,
    cycle_accounting_passed: bool,
    executor_drain_equality_passed: bool,
}

#[derive(Debug, Serialize)]
#[allow(dead_code)]
struct RequestReplayOnlyReport {
    schema_version: u32,
    evidence_tier: &'static str,
    model: String,
    model_architecture: String,
    workload: String,
    layer: u64,
    step: u64,
    batch: u64,
    top_k: u64,
    active_routed_experts: usize,
    routed_pairs: u64,
    comparison: RequestReplayComparison,
    limitations: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
pub(crate) struct MatrixArchitectureReport {
    schema_version: u32,
    evidence_tier: String,
    measurement_mode: &'static str,
    pub(crate) architecture: String,
    model: String,
    model_architecture: String,
    workload: String,
    layer: u64,
    step: u64,
    batch: u64,
    top_k: u64,
    active_routed_experts: usize,
    routed_pairs: u64,
    pub(crate) serial_total_cycles: u64,
    measured_baseline_matrix_cycles: u64,
    reconstructed_baseline_matrix_cycles: u64,
    candidate_matrix_cycles: Option<u64>,
    pub(crate) adjusted_total_cycles: Option<u64>,
    pub(crate) speedup_vs_measured_baseline: Option<f64>,
    request_replay: Option<RequestReplayComparison>,
    equal_area_pes: u64,
    big_rows: u64,
    big_cols: u64,
    tail_rows: u64,
    tail_cols: u64,
    tail_threshold: u64,
    slice_count: u64,
    slice_rows: u64,
    slice_cols: u64,
    routed_matrix_wave_cycles: u64,
    shared_matrix_wave_cycles: u64,
    assumptions: Vec<&'static str>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FixedBigTailConfig {
    pub(crate) big_cols: u64,
    pub(crate) tail_rows: u64,
    pub(crate) tail_cols: u64,
    pub(crate) tail_threshold: u64,
}

pub(crate) fn clear_report_cycles() {
    *REPORT_CYCLES.lock().unwrap() = None;
}

pub(crate) fn report_cycles() -> Option<u64> {
    *REPORT_CYCLES.lock().unwrap()
}

fn div_ceil(value: u64, divisor: u64) -> u64 {
    value.div_ceil(divisor.max(1))
}

fn align_up(value: u64, alignment: u64) -> u64 {
    div_ceil(value, alignment) * alignment
}

fn shared_width_multiple(manifest: &GroupedManifest) -> u64 {
    assert!(manifest.routed_intermediate > 0);
    assert!(
        manifest
            .shared_intermediate
            .is_multiple_of(manifest.routed_intermediate)
    );
    manifest.shared_intermediate / manifest.routed_intermediate
}

fn baseline_matrix_cycles(manifest: &GroupedManifest) -> u64 {
    let routed = manifest
        .group_sizes
        .values()
        .map(|tokens| div_ceil(*tokens, BASELINE_ROWS))
        .sum::<u64>()
        * ROUTED_MATRIX_WAVE_CYCLES;
    let shared = shared_width_multiple(manifest)
        * div_ceil(manifest.batch, BASELINE_ROWS)
        * SHARED_MATRIX_WAVE_CYCLES;
    routed + shared
}

fn gangable_matrix_cycles(manifest: &GroupedManifest) -> u64 {
    let mut jobs = Vec::new();
    jobs.extend(std::iter::repeat_n(
        shared_width_multiple(manifest) * SHARED_MATRIX_WAVE_CYCLES,
        manifest.batch as usize,
    ));
    jobs.extend(std::iter::repeat_n(
        ROUTED_MATRIX_WAVE_CYCLES,
        manifest.group_sizes.values().sum::<u64>() as usize,
    ));
    jobs.sort_unstable_by(|left, right| right.cmp(left));

    let mut lanes = BinaryHeap::from([Reverse(0_u64); 4]);
    for job in jobs {
        let Reverse(ready) = lanes.pop().unwrap();
        lanes.push(Reverse(ready + job));
    }
    lanes
        .into_iter()
        .map(|Reverse(value)| value)
        .max()
        .unwrap_or(0)
}

fn fixed_big_tail_matrix_cycles(manifest: &GroupedManifest, config: FixedBigTailConfig) -> u64 {
    assert!(config.big_cols > 0 && config.tail_rows > 0 && config.tail_cols > 0);
    assert!(config.tail_threshold > 0);
    assert_eq!(
        BASELINE_ROWS * config.big_cols + config.tail_rows * config.tail_cols,
        BASELINE_PES,
        "fixed big+tail candidate must be equal-area (4096 PEs)"
    );
    assert!(config.big_cols <= BASELINE_COLS && config.tail_cols <= BASELINE_COLS);

    let mut big_routed_waves = 0;
    let mut tail_routed_waves = 0;
    for tokens in manifest.group_sizes.values() {
        if *tokens <= config.tail_threshold {
            tail_routed_waves += div_ceil(*tokens, config.tail_rows);
        } else {
            big_routed_waves += div_ceil(*tokens, BASELINE_ROWS);
        }
    }
    let big_unscaled = big_routed_waves * ROUTED_MATRIX_WAVE_CYCLES
        + shared_width_multiple(manifest)
            * div_ceil(manifest.batch, BASELINE_ROWS)
            * SHARED_MATRIX_WAVE_CYCLES;
    let tail_unscaled = tail_routed_waves * ROUTED_MATRIX_WAVE_CYCLES;
    let big = div_ceil(big_unscaled * BASELINE_COLS, config.big_cols);
    let tail = div_ceil(tail_unscaled * BASELINE_COLS, config.tail_cols);
    big.max(tail)
}

fn stored_mx_bytes(k: u64, n: u64) -> u64 {
    assert!(k.is_multiple_of(COMPILER_MLEN));
    assert!(n.is_multiple_of(COMPILER_MLEN));
    let elements = k * n;
    assert!(elements.is_multiple_of(MX_BLOCK));
    align_up(elements + elements / MX_BLOCK, HBM_BURST_BYTES)
}

#[cfg(test)]
fn projection_physical_bytes(projection: &Projection) -> u64 {
    let tiles = (projection.k / COMPILER_MLEN) * (projection.n / COMPILER_MLEN);
    // A 64x64 MX tile causes 64 element bursts plus 64 scale bursts. Scale
    // bytes are sparse/strided, so 64-byte physical rounding applies per row.
    tiles * 2 * COMPILER_MLEN * HBM_BURST_BYTES
}

fn jobs_from_manifest(manifest: &GroupedManifest) -> Vec<LogicalJob> {
    assert!(manifest.hidden.is_multiple_of(COMPILER_MLEN));
    assert!(manifest.routed_intermediate.is_multiple_of(COMPILER_MLEN));
    assert!(manifest.shared_intermediate.is_multiple_of(COMPILER_MLEN));
    assert!(manifest.num_experts > 0);

    let routed_gate_stride = stored_mx_bytes(manifest.hidden, manifest.routed_intermediate);
    let routed_up_stride = routed_gate_stride;
    let routed_down_stride = stored_mx_bytes(manifest.routed_intermediate, manifest.hidden);
    let routed_gate_table = 0;
    let routed_up_table = routed_gate_table + routed_gate_stride * manifest.num_experts;
    let routed_down_table = routed_up_table + routed_up_stride * manifest.num_experts;
    let shared_gate_base = routed_down_table + routed_down_stride * manifest.num_experts;
    let shared_gate_size = stored_mx_bytes(manifest.hidden, manifest.shared_intermediate);
    let shared_up_base = shared_gate_base + shared_gate_size;
    let shared_up_size = shared_gate_size;
    let shared_down_base = shared_up_base + shared_up_size;

    let mut jobs = vec![LogicalJob {
        id: 0,
        kind: JobKind::Shared,
        expert_id: None,
        tokens: manifest.batch,
        projections: [
            Projection {
                name: "shared_gate",
                k: manifest.hidden,
                n: manifest.shared_intermediate,
                base: shared_gate_base,
            },
            Projection {
                name: "shared_up",
                k: manifest.hidden,
                n: manifest.shared_intermediate,
                base: shared_up_base,
            },
            Projection {
                name: "shared_down",
                k: manifest.shared_intermediate,
                n: manifest.hidden,
                base: shared_down_base,
            },
        ],
    }];

    let mut routed = manifest
        .group_sizes
        .iter()
        .map(|(expert, tokens)| {
            let expert_id = expert
                .parse::<u64>()
                .unwrap_or_else(|err| panic!("invalid expert id {expert:?}: {err}"));
            assert!(expert_id < manifest.num_experts);
            (expert_id, *tokens)
        })
        .collect::<Vec<_>>();
    routed.sort_unstable_by_key(|(expert, tokens)| (Reverse(*tokens), *expert));
    for (expert_id, tokens) in routed {
        let id = jobs.len();
        jobs.push(LogicalJob {
            id,
            kind: JobKind::Routed,
            expert_id: Some(expert_id),
            tokens,
            projections: [
                Projection {
                    name: "routed_gate",
                    k: manifest.hidden,
                    n: manifest.routed_intermediate,
                    base: routed_gate_table + expert_id * routed_gate_stride,
                },
                Projection {
                    name: "routed_up",
                    k: manifest.hidden,
                    n: manifest.routed_intermediate,
                    base: routed_up_table + expert_id * routed_up_stride,
                },
                Projection {
                    name: "routed_down",
                    k: manifest.routed_intermediate,
                    n: manifest.hidden,
                    base: routed_down_table + expert_id * routed_down_stride,
                },
            ],
        });
    }
    jobs
}

fn mm_ops_per_wave(job: &LogicalJob) -> u64 {
    job.projections
        .iter()
        .map(|projection| (projection.k / COMPILER_MLEN) * (projection.n / COMPILER_BLEN))
        .sum()
}

fn calibrated_wave_cycles(job: &LogicalJob) -> u64 {
    let reference = match job.kind {
        JobKind::Routed => ROUTED_MATRIX_WAVE_CYCLES,
        JobKind::Shared => SHARED_MATRIX_WAVE_CYCLES,
    };
    div_ceil(
        reference * mm_ops_per_wave(job),
        CALIBRATION_MM_OPS_PER_WAVE,
    )
}

fn fixed_cycles_per_output_tile(job: &LogicalJob) -> f64 {
    let feed = mm_ops_per_wave(job) * COMPILER_MLEN;
    let wave = calibrated_wave_cycles(job);
    assert!(
        wave >= feed,
        "calibrated matrix wave is smaller than feed work"
    );
    let output_tiles = job
        .projections
        .iter()
        .map(|projection| projection.n / COMPILER_BLEN)
        .sum::<u64>();
    (wave - feed) as f64 / output_tiles as f64
}

fn baseline_compute(job: &LogicalJob) -> ComputeBreakdown {
    let row_waves = div_ceil(job.tokens, BASELINE_ROWS);
    let feed = row_waves * mm_ops_per_wave(job) * COMPILER_MLEN;
    let cycles = row_waves * calibrated_wave_cycles(job);
    let fixed = cycles - feed;
    ComputeBreakdown {
        compute_cycles: cycles,
        aggregate_core_busy_cycles: cycles,
        aggregate_feed_cycles: feed,
        aggregate_fixed_cycles: fixed,
        ..Default::default()
    }
}

fn split_aligned(total: u64, parts: u64, alignment: u64) -> Vec<(u64, u64)> {
    assert!(total.is_multiple_of(alignment));
    let units = total / alignment;
    let quotient = units / parts;
    let remainder = units % parts;
    let mut cursor = 0;
    let mut result = Vec::new();
    for index in 0..parts {
        let extent = (quotient + u64::from(index < remainder)) * alignment;
        result.push((cursor, cursor + extent));
        cursor += extent;
    }
    assert_eq!(cursor, total);
    result
}

fn account_parallel_durations(result: &mut ComputeBreakdown, durations: &[u64]) {
    let wall = *durations
        .iter()
        .max()
        .expect("partition list must be non-empty");
    let earliest = *durations.iter().min().unwrap();
    let busy = durations.iter().sum::<u64>();
    result.compute_cycles += wall;
    result.aggregate_core_busy_cycles += busy;
    result.aggregate_partition_wait_cycles += wall * durations.len() as u64 - busy;
    result.max_partition_skew_cycles = result
        .max_partition_skew_cycles
        .max(wall.saturating_sub(earliest));
}

fn n_split_compute(job: &LogicalJob, core_rows: &[u64]) -> ComputeBreakdown {
    assert!(core_rows.len() > 1);
    let fixed_per_output = fixed_cycles_per_output_tile(job);
    let mut result = ComputeBreakdown::default();
    for projection in &job.projections {
        let mut durations = Vec::new();
        // A compiler N tile is BLEN-wide, but a physical MX element request is
        // one 64-byte row burst.  Keeping the split on 64-column boundaries
        // guarantees that the two cores neither duplicate nor share a burst.
        for ((start, stop), rows) in
            split_aligned(projection.n, core_rows.len() as u64, COMPILER_MLEN)
                .into_iter()
                .zip(core_rows.iter())
        {
            let row_waves = div_ceil(job.tokens, *rows);
            let n_tiles = (stop - start) / COMPILER_BLEN;
            let k_tiles = projection.k / COMPILER_MLEN;
            let mm_ops = row_waves * n_tiles * k_tiles;
            let feed = mm_ops * COMPILER_MLEN;
            let fixed = ((row_waves * n_tiles) as f64 * fixed_per_output).ceil() as u64;
            let launch = PARTITION_LAUNCH_CYCLES;
            durations.push(feed + fixed + launch);
            result.aggregate_feed_cycles += feed;
            result.aggregate_fixed_cycles += fixed;
            result.aggregate_launch_cycles += launch;
        }
        account_parallel_durations(&mut result, &durations);
    }
    let exchange_bytes = job.tokens * job.projections[0].n * 2 * (core_rows.len() as u64 - 1);
    let exchange = div_ceil(exchange_bytes, INTERCONNECT_BYTES_PER_CYCLE);
    result.finalize_cycles = exchange;
    result.interconnect_cycles = exchange;
    result.interconnect_bytes = exchange_bytes;
    result
}

fn m_token_counts(tokens: u64, core_rows: &[u64]) -> Vec<u64> {
    assert_eq!(core_rows.iter().sum::<u64>(), BASELINE_ROWS);
    let full_waves = tokens / BASELINE_ROWS;
    let mut remainder = tokens % BASELINE_ROWS;
    let mut counts = core_rows
        .iter()
        .map(|rows| rows * full_waves)
        .collect::<Vec<_>>();
    for (count, rows) in counts.iter_mut().zip(core_rows.iter()) {
        let assigned = remainder.min(*rows);
        *count += assigned;
        remainder -= assigned;
    }
    assert_eq!(remainder, 0);
    assert_eq!(counts.iter().sum::<u64>(), tokens);
    counts
}

fn m_split_compute(job: &LogicalJob, core_rows: &[u64]) -> ComputeBreakdown {
    assert!(core_rows.len() > 1);
    let fixed_per_output = fixed_cycles_per_output_tile(job);
    let token_counts = m_token_counts(job.tokens, core_rows);
    let mut result = ComputeBreakdown::default();
    for projection in &job.projections {
        let n_tiles = projection.n / COMPILER_BLEN;
        let k_tiles = projection.k / COMPILER_MLEN;
        let mut durations = Vec::new();
        for (tokens, rows) in token_counts.iter().zip(core_rows.iter()) {
            if *tokens == 0 {
                continue;
            }
            let row_waves = div_ceil(*tokens, *rows);
            let feed = row_waves * n_tiles * k_tiles * COMPILER_MLEN;
            let fixed = ((row_waves * n_tiles) as f64 * fixed_per_output).ceil() as u64;
            let launch = PARTITION_LAUNCH_CYCLES;
            durations.push(feed + fixed + launch);
            result.aggregate_feed_cycles += feed;
            result.aggregate_fixed_cycles += fixed;
            result.aggregate_launch_cycles += launch;
        }
        account_parallel_durations(&mut result, &durations);
    }
    result
}

fn split_integer_weighted(total: u64, weights: &[f64]) -> Vec<u64> {
    let sum = weights.iter().sum::<f64>();
    let exact = weights
        .iter()
        .map(|weight| total as f64 * weight / sum)
        .collect::<Vec<_>>();
    let mut result = exact
        .iter()
        .map(|value| value.floor() as u64)
        .collect::<Vec<_>>();
    let remainder = total - result.iter().sum::<u64>();
    let mut order = (0..weights.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        let left_fraction = exact[*left] - result[*left] as f64;
        let right_fraction = exact[*right] - result[*right] as f64;
        right_fraction
            .total_cmp(&left_fraction)
            .then_with(|| weights[*right].total_cmp(&weights[*left]))
            .then_with(|| left.cmp(right))
    });
    for index in order.into_iter().take(remainder as usize) {
        result[index] += 1;
    }
    assert_eq!(result.iter().sum::<u64>(), total);
    result
}

fn k_block_counts(projection: &Projection, tokens: u64, core_rows: &[u64]) -> Vec<u64> {
    assert!(projection.k.is_multiple_of(COMPILER_MLEN));
    let blocks = projection.k / COMPILER_MLEN;
    assert!(blocks >= core_rows.len() as u64);
    let weights = core_rows
        .iter()
        .map(|rows| 1.0 / div_ceil(tokens, *rows) as f64)
        .collect::<Vec<_>>();
    let counts = split_integer_weighted(blocks, &weights);
    assert!(counts.iter().all(|count| *count > 0));
    counts
}

fn block_partition(block: u64, counts: &[u64]) -> usize {
    let mut stop = 0_u64;
    for (partition, count) in counts.iter().enumerate() {
        stop += count;
        if block < stop {
            return partition;
        }
    }
    panic!("block {block} lies outside {} assigned blocks", stop);
}

fn static_k_split_compute(job: &LogicalJob, core_rows: &[u64]) -> ComputeBreakdown {
    let fixed_per_output = fixed_cycles_per_output_tile(job);
    let mut result = ComputeBreakdown::default();
    for projection in &job.projections {
        let counts = k_block_counts(projection, job.tokens, core_rows);
        let n_tiles = projection.n / COMPILER_BLEN;
        let mut durations = Vec::with_capacity(core_rows.len());
        for (rows, k_tiles) in core_rows.iter().zip(counts.iter()) {
            let row_waves = div_ceil(job.tokens, *rows);
            let feed = row_waves * n_tiles * k_tiles * COMPILER_MLEN;
            // Every K participant produces a partial output tile, so fixed
            // fill/drain/writeback is paid by every active participant.  Only
            // the feed term scales with the K slice extent.
            let fixed = ((row_waves * n_tiles) as f64 * fixed_per_output).ceil() as u64;
            let launch = PARTITION_LAUNCH_CYCLES;
            durations.push(feed + fixed + launch);
            result.aggregate_feed_cycles += feed;
            result.aggregate_fixed_cycles += fixed;
            result.aggregate_launch_cycles += launch;
        }
        account_parallel_durations(&mut result, &durations);

        let elements = job.tokens * projection.n;
        let active_parts = counts.iter().filter(|count| **count > 0).count() as u64;
        let (reduce, interconnect, sync, bytes) = reduction_cost(elements, active_parts);
        result.finalize_cycles += reduce + interconnect + sync;
        result.reducer_cycles += reduce;
        result.interconnect_cycles += interconnect;
        result.synchronization_cycles += sync;
        result.interconnect_bytes += bytes;
        result.reduction_output_elements += elements;
        result.reduction_add_elements += elements * active_parts.saturating_sub(1);
        result.reduction_elements += elements;
    }
    result
}

fn stage_n_to_k_compute(job: &LogicalJob, core_rows: &[u64]) -> ComputeBreakdown {
    assert!(core_rows.len() > 1);
    assert!(core_rows.iter().all(|rows| *rows == core_rows[0]));
    let fixed_per_output = fixed_cycles_per_output_tile(job);
    let mut result = ComputeBreakdown::default();

    // Gate and up produce disjoint intermediate columns. The same aligned
    // slices become the down projection's K slices, so no all-to-all
    // intermediate exchange is required between stages.
    for projection in &job.projections[..2] {
        let mut durations = Vec::new();
        for ((start, stop), rows) in
            split_aligned(projection.n, core_rows.len() as u64, COMPILER_MLEN)
                .into_iter()
                .zip(core_rows.iter())
        {
            let row_waves = div_ceil(job.tokens, *rows);
            let n_tiles = (stop - start) / COMPILER_BLEN;
            let k_tiles = projection.k / COMPILER_MLEN;
            let feed = row_waves * n_tiles * k_tiles * COMPILER_MLEN;
            let fixed = ((row_waves * n_tiles) as f64 * fixed_per_output).ceil() as u64;
            let launch = PARTITION_LAUNCH_CYCLES;
            durations.push(feed + fixed + launch);
            result.aggregate_feed_cycles += feed;
            result.aggregate_fixed_cycles += fixed;
            result.aggregate_launch_cycles += launch;
        }
        account_parallel_durations(&mut result, &durations);
    }

    let down = &job.projections[2];
    let counts = split_aligned(down.k, core_rows.len() as u64, COMPILER_MLEN)
        .into_iter()
        .map(|(start, stop)| (stop - start) / COMPILER_MLEN)
        .collect::<Vec<_>>();
    let n_tiles = down.n / COMPILER_BLEN;
    let mut durations = Vec::new();
    for (rows, k_tiles) in core_rows.iter().zip(counts.iter()) {
        let row_waves = div_ceil(job.tokens, *rows);
        let feed = row_waves * n_tiles * k_tiles * COMPILER_MLEN;
        let fixed = ((row_waves * n_tiles) as f64 * fixed_per_output).ceil() as u64;
        let launch = PARTITION_LAUNCH_CYCLES;
        durations.push(feed + fixed + launch);
        result.aggregate_feed_cycles += feed;
        result.aggregate_fixed_cycles += fixed;
        result.aggregate_launch_cycles += launch;
    }
    account_parallel_durations(&mut result, &durations);
    let elements = job.tokens * down.n;
    let parts = core_rows.len() as u64;
    let (reduce, interconnect, sync, bytes) = reduction_cost(elements, parts);
    result.finalize_cycles += reduce + interconnect + sync;
    result.reducer_cycles += reduce;
    result.interconnect_cycles += interconnect;
    result.synchronization_cycles += sync;
    result.interconnect_bytes += bytes;
    result.reduction_output_elements += elements;
    result.reduction_add_elements += elements * (parts - 1);
    result.reduction_elements += elements;
    result
}

fn reduction_cost(elements: u64, parts: u64) -> (u64, u64, u64, u64) {
    if parts <= 1 || elements == 0 {
        return (0, 0, 0, 0);
    }
    let levels = parts.next_power_of_two().ilog2() as u64;
    let reduce = div_ceil(elements, VECTOR_REDUCE_WIDTH) * levels;
    let bytes = elements * 2 * (parts - 1);
    let interconnect = div_ceil(bytes, INTERCONNECT_BYTES_PER_CYCLE);
    let sync = SYNC_CYCLES_PER_PART * parts;
    (reduce, interconnect, sync, bytes)
}

fn stream_k_compute(job: &LogicalJob, rows: &[u64]) -> ComputeBreakdown {
    assert!(rows.len() > 1);
    let fixed_per_output = fixed_cycles_per_output_tile(job);
    let mut result = ComputeBreakdown::default();
    for projection in &job.projections {
        let n_tiles = projection.n / COMPILER_BLEN;
        let k_tiles = projection.k / COMPILER_MLEN;
        let mut core_feed = vec![0_u64; rows.len()];
        let mut core_drain = vec![0_f64; rows.len()];
        let full_waves = job.tokens / BASELINE_ROWS;
        let tail_rows = job.tokens % BASELINE_ROWS;
        let mut categories = vec![(BASELINE_ROWS, full_waves)];
        if tail_rows > 0 {
            categories.push((tail_rows, 1));
        }
        for (valid_rows, wave_count) in categories {
            if wave_count == 0 {
                continue;
            }
            let output_tiles = wave_count * n_tiles;
            let task_count = output_tiles * k_tiles;
            let task_costs = rows
                .iter()
                .map(|core_rows| div_ceil(valid_rows, *core_rows))
                .collect::<Vec<_>>();
            let weights = task_costs
                .iter()
                .map(|cost| 1.0 / *cost as f64)
                .collect::<Vec<_>>();
            let assigned = split_integer_weighted(task_count, &weights);
            let mut cursor = 0_u64;
            let mut split_outputs = BTreeMap::<u64, u64>::new();
            for index in 0..rows.len() {
                let count = assigned[index];
                let row_passes = task_costs[index];
                if count == 0 {
                    continue;
                }
                let first_output = cursor / k_tiles;
                let last_output = (cursor + count - 1) / k_tiles;
                let touched = last_output - first_output + 1;
                core_feed[index] += count * row_passes * COMPILER_MLEN;
                core_drain[index] += touched as f64 * row_passes as f64 * fixed_per_output;
                cursor += count;
                if cursor > 0 && cursor < task_count && !cursor.is_multiple_of(k_tiles) {
                    *split_outputs.entry(cursor / k_tiles).or_default() += 1;
                }
            }
            for boundary_count in split_outputs.values() {
                let elements = valid_rows * COMPILER_BLEN;
                let parts = boundary_count + 1;
                let (reduce, interconnect, sync, bytes) = reduction_cost(elements, parts);
                result.finalize_cycles += reduce + interconnect + sync;
                result.reducer_cycles += reduce;
                result.interconnect_cycles += interconnect;
                result.synchronization_cycles += sync;
                result.interconnect_bytes += bytes;
                result.reduction_output_elements += elements;
                result.reduction_add_elements += elements * (parts - 1);
                result.reduction_elements += elements;
            }
        }
        let durations = (0..rows.len())
            .map(|index| {
                core_feed[index] + core_drain[index].ceil() as u64 + PARTITION_LAUNCH_CYCLES
            })
            .collect::<Vec<_>>();
        account_parallel_durations(&mut result, &durations);
        result.aggregate_feed_cycles += core_feed.iter().sum::<u64>();
        result.aggregate_fixed_cycles += core_drain
            .iter()
            .map(|cycles| cycles.ceil() as u64)
            .sum::<u64>();
        result.aggregate_launch_cycles += PARTITION_LAUNCH_CYCLES * rows.len() as u64;
    }
    result
}

fn compute_for(job: &LogicalJob, architecture: ReplayArchitecture) -> ComputeBreakdown {
    match architecture {
        ReplayArchitecture::Baseline { .. } => baseline_compute(job),
        ReplayArchitecture::DualNsplitDepth2 => n_split_compute(job, architecture.core_rows()),
        ReplayArchitecture::DualKsplitDepth2 => {
            static_k_split_compute(job, architecture.core_rows())
        }
        ReplayArchitecture::DualOnlineStreamKDepth2 => {
            stream_k_compute(job, architecture.core_rows())
        }
        ReplayArchitecture::Asym211StreamKDepth3 => stream_k_compute(job, architecture.core_rows()),
        ReplayArchitecture::Asym211OnlineStreamKDepth3 => {
            stream_k_compute(job, architecture.core_rows())
        }
        ReplayArchitecture::Asym211KsplitDepth3 => {
            static_k_split_compute(job, architecture.core_rows())
        }
        ReplayArchitecture::QuadMsplitDepth4 => m_split_compute(job, architecture.core_rows()),
        ReplayArchitecture::QuadNsplitDepth4 => n_split_compute(job, architecture.core_rows()),
        ReplayArchitecture::QuadKsplitDepth4 => {
            static_k_split_compute(job, architecture.core_rows())
        }
        ReplayArchitecture::QuadStageNToKDepth4 => {
            stage_n_to_k_compute(job, architecture.core_rows())
        }
        ReplayArchitecture::QuadStreamKDepth4 => stream_k_compute(job, architecture.core_rows()),
        ReplayArchitecture::QuadOnlineStreamKDepth4 => {
            stream_k_compute(job, architecture.core_rows())
        }
    }
}

#[derive(Clone, Debug)]
struct WeightTile {
    id: usize,
    job_id: usize,
    expert_id: Option<u64>,
    tokens: u64,
    projection: Projection,
    col_start: u64,
    col_stop: u64,
    k_start: u64,
    compute: ComputeBreakdown,
    job_last: bool,
}

fn allocate_weighted_exact(total: u64, weights: &[f64]) -> Vec<u64> {
    if total == 0 {
        return vec![0; weights.len()];
    }
    split_integer_weighted(total, weights)
}

fn build_weight_tiles(jobs: &[LogicalJob], architecture: ReplayArchitecture) -> Vec<WeightTile> {
    let mut result = Vec::new();
    for job in jobs {
        let mut specs = Vec::new();
        for projection in &job.projections {
            let mut col_start = 0;
            while col_start < projection.n {
                let col_stop = (col_start + WEIGHT_TILE_N).min(projection.n);
                assert!(col_start.is_multiple_of(COMPILER_MLEN));
                assert!(col_stop.is_multiple_of(COMPILER_MLEN));
                for k_start in (0..projection.k).step_by(COMPILER_MLEN as usize) {
                    // Feed work scales with N extent. The final K tile also
                    // carries drain/writeback for those output columns.
                    let mut weight = ((col_stop - col_start) / COMPILER_BLEN) as f64;
                    if k_start + COMPILER_MLEN == projection.k {
                        weight *= 1.0 + fixed_cycles_per_output_tile(job) / COMPILER_MLEN as f64;
                    }
                    specs.push((projection.clone(), col_start, col_stop, k_start, weight));
                }
                col_start = col_stop;
            }
        }
        let total = compute_for(job, architecture);
        let weights = specs.iter().map(|item| item.4).collect::<Vec<_>>();
        let compute_alloc = allocate_weighted_exact(total.compute_cycles, &weights);
        let busy_alloc = allocate_weighted_exact(total.aggregate_core_busy_cycles, &weights);
        let feed_alloc = allocate_weighted_exact(total.aggregate_feed_cycles, &weights);
        let fixed_alloc = allocate_weighted_exact(total.aggregate_fixed_cycles, &weights);
        let launch_alloc = allocate_weighted_exact(total.aggregate_launch_cycles, &weights);
        let wait_alloc = allocate_weighted_exact(total.aggregate_partition_wait_cycles, &weights);
        let spec_count = specs.len();
        for (index, (projection, col_start, col_stop, k_start, _)) in specs.into_iter().enumerate()
        {
            let job_last = index + 1 == spec_count;
            result.push(WeightTile {
                id: result.len(),
                job_id: job.id,
                expert_id: job.expert_id,
                tokens: job.tokens,
                projection,
                col_start,
                col_stop,
                k_start,
                compute: ComputeBreakdown {
                    compute_cycles: compute_alloc[index],
                    aggregate_core_busy_cycles: busy_alloc[index],
                    aggregate_feed_cycles: feed_alloc[index],
                    aggregate_fixed_cycles: fixed_alloc[index],
                    aggregate_launch_cycles: launch_alloc[index],
                    aggregate_partition_wait_cycles: wait_alloc[index],
                    max_partition_skew_cycles: if job_last {
                        total.max_partition_skew_cycles
                    } else {
                        0
                    },
                    finalize_cycles: if job_last { total.finalize_cycles } else { 0 },
                    reducer_cycles: if job_last { total.reducer_cycles } else { 0 },
                    interconnect_cycles: if job_last {
                        total.interconnect_cycles
                    } else {
                        0
                    },
                    synchronization_cycles: if job_last {
                        total.synchronization_cycles
                    } else {
                        0
                    },
                    interconnect_bytes: if job_last {
                        total.interconnect_bytes
                    } else {
                        0
                    },
                    reduction_output_elements: if job_last {
                        total.reduction_output_elements
                    } else {
                        0
                    },
                    reduction_add_elements: if job_last {
                        total.reduction_add_elements
                    } else {
                        0
                    },
                    reduction_elements: if job_last {
                        total.reduction_elements
                    } else {
                        0
                    },
                },
                job_last,
            });
        }
    }
    result
}

fn tile_partition(
    architecture: ReplayArchitecture,
    tile: &WeightTile,
    local_col_block: u64,
) -> usize {
    let global_col = tile.col_start + local_col_block * COMPILER_MLEN;
    match architecture {
        ReplayArchitecture::Baseline { .. } => 0,
        ReplayArchitecture::DualNsplitDepth2 | ReplayArchitecture::QuadNsplitDepth4 => {
            split_aligned(
                tile.projection.n,
                architecture.partition_count() as u64,
                COMPILER_MLEN,
            )
            .iter()
            .position(|(start, stop)| global_col >= *start && global_col < *stop)
            .expect("N request block lies outside the candidate tensor slices")
        }
        ReplayArchitecture::DualKsplitDepth2
        | ReplayArchitecture::Asym211KsplitDepth3
        | ReplayArchitecture::QuadKsplitDepth4 => {
            let counts = k_block_counts(&tile.projection, tile.tokens, architecture.core_rows());
            block_partition(tile.k_start / COMPILER_MLEN, &counts)
        }
        ReplayArchitecture::QuadStageNToKDepth4 => {
            if tile.projection.name.ends_with("down") {
                let counts = split_aligned(
                    tile.projection.k,
                    architecture.partition_count() as u64,
                    COMPILER_MLEN,
                )
                .into_iter()
                .map(|(start, stop)| (stop - start) / COMPILER_MLEN)
                .collect::<Vec<_>>();
                block_partition(tile.k_start / COMPILER_MLEN, &counts)
            } else {
                split_aligned(
                    tile.projection.n,
                    architecture.partition_count() as u64,
                    COMPILER_MLEN,
                )
                .iter()
                .position(|(start, stop)| global_col >= *start && global_col < *stop)
                .expect("stage-aware N request block lies outside the tensor slices")
            }
        }
        ReplayArchitecture::QuadMsplitDepth4 => {
            let n_blocks = tile.projection.n / COMPILER_MLEN;
            let k_blocks = tile.projection.k / COMPILER_MLEN;
            let request_block =
                (global_col / COMPILER_MLEN) * k_blocks + tile.k_start / COMPILER_MLEN;
            (request_block % (n_blocks * k_blocks).min(architecture.partition_count() as u64))
                as usize
        }
        ReplayArchitecture::DualOnlineStreamKDepth2
        | ReplayArchitecture::Asym211StreamKDepth3
        | ReplayArchitecture::Asym211OnlineStreamKDepth3
        | ReplayArchitecture::QuadStreamKDepth4
        | ReplayArchitecture::QuadOnlineStreamKDepth4 => {
            // Stream-K computes at BLEN x MLEN granularity, while a physical
            // element request contains MLEN output columns.  Stage each 64x64
            // request block exactly once, using the same inverse-row-pass
            // weighting as the compute scheduler.  A later SRAM crossbar may
            // distribute the sixteen BLEN output tiles inside that burst.
            let n_blocks = tile.projection.n / COMPILER_MLEN;
            let k_blocks = tile.projection.k / COMPILER_MLEN;
            let request_block =
                (global_col / COMPILER_MLEN) * k_blocks + tile.k_start / COMPILER_MLEN;
            let weights = architecture
                .core_rows()
                .iter()
                .map(|rows| 1.0 / div_ceil(tile.tokens, *rows) as f64)
                .collect::<Vec<_>>();
            let counts = split_integer_weighted(n_blocks * k_blocks, &weights);
            block_partition(request_block, &counts)
        }
    }
}

fn request_block_owners(architecture: ReplayArchitecture, tile: &WeightTile) -> Vec<(u64, usize)> {
    let blocks = (tile.col_stop - tile.col_start) / COMPILER_MLEN;
    (0..blocks)
        .map(|block| (block, tile_partition(architecture, tile, block)))
        .collect()
}

fn validate_block_owners(
    owners: &[(u64, usize)],
    expected_blocks: u64,
    partitions: usize,
) -> Result<(), String> {
    if owners.len() as u64 != expected_blocks {
        return Err(format!(
            "request block count mismatch: got {}, expected {expected_blocks}",
            owners.len()
        ));
    }
    let mut seen = vec![false; expected_blocks as usize];
    for (block, partition) in owners {
        if *block >= expected_blocks {
            return Err(format!("request block {block} is out of range"));
        }
        if *partition >= partitions {
            return Err(format!("request partition {partition} is out of range"));
        }
        if std::mem::replace(&mut seen[*block as usize], true) {
            return Err(format!("request block {block} is assigned more than once"));
        }
    }
    if let Some(block) = seen.iter().position(|present| !present) {
        return Err(format!("request block {block} is unassigned"));
    }
    Ok(())
}

fn validate_replay_contract(
    jobs: &[LogicalJob],
    architecture: ReplayArchitecture,
) -> Result<(), String> {
    let rows = architecture.core_rows();
    if rows.iter().sum::<u64>() * BASELINE_COLS != BASELINE_PES {
        return Err(format!("{} is not a 4096-PE design", architecture.name()));
    }
    if rows.len() != architecture.partition_count() {
        return Err(format!(
            "{} has {} cores but {} request partitions",
            architecture.name(),
            rows.len(),
            architecture.partition_count()
        ));
    }
    for job in jobs {
        for projection in &job.projections {
            if !projection.k.is_multiple_of(COMPILER_MLEN)
                || !projection.n.is_multiple_of(COMPILER_MLEN)
            {
                return Err(format!(
                    "{} {} shape K={} N={} is not 64-aligned",
                    architecture.name(),
                    projection.name,
                    projection.k,
                    projection.n
                ));
            }
            match architecture {
                ReplayArchitecture::DualNsplitDepth2 | ReplayArchitecture::QuadNsplitDepth4 => {
                    let slices = split_aligned(
                        projection.n,
                        architecture.partition_count() as u64,
                        COMPILER_MLEN,
                    );
                    if slices.first().map(|item| item.0) != Some(0)
                        || slices.last().map(|item| item.1) != Some(projection.n)
                        || slices.iter().any(|(start, stop)| {
                            start == stop
                                || !start.is_multiple_of(COMPILER_MLEN)
                                || !stop.is_multiple_of(COMPILER_MLEN)
                        })
                    {
                        return Err(format!("{} has an illegal N slice", projection.name));
                    }
                }
                ReplayArchitecture::DualKsplitDepth2
                | ReplayArchitecture::Asym211KsplitDepth3
                | ReplayArchitecture::QuadKsplitDepth4 => {
                    let counts = k_block_counts(projection, job.tokens, rows);
                    if counts.iter().sum::<u64>() != projection.k / COMPILER_MLEN {
                        return Err(format!("{} K slices do not cover K", projection.name));
                    }
                }
                ReplayArchitecture::QuadStageNToKDepth4 => {
                    let axis = if projection.name.ends_with("down") {
                        projection.k
                    } else {
                        projection.n
                    };
                    let slices =
                        split_aligned(axis, architecture.partition_count() as u64, COMPILER_MLEN);
                    if slices.iter().any(|(start, stop)| {
                        start == stop
                            || !start.is_multiple_of(COMPILER_MLEN)
                            || !stop.is_multiple_of(COMPILER_MLEN)
                    }) {
                        return Err(format!(
                            "{} has an illegal stage-aware slice",
                            projection.name
                        ));
                    }
                }
                ReplayArchitecture::Baseline { .. }
                | ReplayArchitecture::DualOnlineStreamKDepth2
                | ReplayArchitecture::Asym211StreamKDepth3
                | ReplayArchitecture::Asym211OnlineStreamKDepth3
                | ReplayArchitecture::QuadMsplitDepth4
                | ReplayArchitecture::QuadStreamKDepth4
                | ReplayArchitecture::QuadOnlineStreamKDepth4 => {}
            }
        }
    }
    for tile in build_weight_tiles(jobs, architecture) {
        let blocks = (tile.col_stop - tile.col_start) / COMPILER_MLEN;
        validate_block_owners(
            &request_block_owners(architecture, &tile),
            blocks,
            architecture.partition_count(),
        )?;
    }
    Ok(())
}

fn append_tile_bursts(addresses: &mut Vec<u64>, projection: &Projection, col: u64, row: u64) {
    let element_base = projection.base;
    let scale_base = projection.base + projection.k * projection.n;
    for offset in 0..COMPILER_MLEN {
        let logical = (row + offset) * projection.n + col;
        let element_addr = element_base + logical;
        assert!(element_addr.is_multiple_of(HBM_BURST_BYTES));
        addresses.push(element_addr);
        let scale_addr = scale_base + logical / MX_BLOCK;
        addresses.push((scale_addr / HBM_BURST_BYTES) * HBM_BURST_BYTES);
    }
}

async fn issue_address_batch(hbm: &Arc<dyn ErasedMemoryModel>, addresses: &mut Vec<u64>) {
    let reads = addresses
        .drain(..)
        .map(|addr| {
            let hbm = hbm.clone();
            async move {
                let _ = hbm.read(addr).await;
            }
        })
        .collect::<Vec<_>>();
    join_all(reads).await;
}

async fn transfer_partition(
    hbm: Arc<dyn ErasedMemoryModel>,
    tile: WeightTile,
    architecture: ReplayArchitecture,
    partition: usize,
) {
    let mut addresses = Vec::with_capacity(HBM_ISSUE_BATCH);
    for (local_col_block, owner) in request_block_owners(architecture, &tile) {
        if owner != partition {
            continue;
        }
        append_tile_bursts(
            &mut addresses,
            &tile.projection,
            tile.col_start + local_col_block * COMPILER_MLEN,
            tile.k_start,
        );
        if addresses.len() >= HBM_ISSUE_BATCH {
            issue_address_batch(&hbm, &mut addresses).await;
        }
    }
    if !addresses.is_empty() {
        issue_address_batch(&hbm, &mut addresses).await;
    }
}

#[derive(Debug)]
struct TransferComplete {
    launch_sequence: u64,
    tile: WeightTile,
    completion_cycle: u64,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct OnlineOutputKey {
    job_id: usize,
    projection: usize,
    wave: u64,
    n_tile: u64,
}

#[derive(Clone, Debug)]
struct OnlineWorkItem {
    id: u64,
    ready_sequence: u64,
    tile_id: usize,
    key: OnlineOutputKey,
    n_tile_count: u64,
    valid_rows: u64,
    home_partition: usize,
    fixed_cycles_per_output: f64,
}

impl OnlineWorkItem {
    fn output_keys(&self) -> impl Iterator<Item = OnlineOutputKey> + '_ {
        (0..self.n_tile_count).map(|offset| OnlineOutputKey {
            n_tile: self.key.n_tile + offset,
            ..self.key
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct OnlineTaskTiming {
    feed_cycles: u64,
    fixed_cycles: u64,
    launch_cycles: u64,
    dispatch_cycles: u64,
}

impl OnlineTaskTiming {
    fn total(self) -> u64 {
        self.feed_cycles + self.fixed_cycles + self.launch_cycles + self.dispatch_cycles
    }
}

#[derive(Debug)]
struct PendingCrossbar {
    sequence: u64,
    core_id: usize,
    item: OnlineWorkItem,
    timing: OnlineTaskTiming,
    cycles: u64,
    bytes: u64,
    ready_cycle: u64,
}

#[derive(Debug)]
struct PendingReduction {
    sequence: u64,
    key: OnlineOutputKey,
    reduce_cycles: u64,
    interconnect_cycles: u64,
    synchronization_cycles: u64,
    interconnect_bytes: u64,
    output_elements: u64,
    add_elements: u64,
    ready_cycle: u64,
}

impl PendingReduction {
    fn total_cycles(&self) -> u64 {
        self.reduce_cycles + self.interconnect_cycles + self.synchronization_cycles
    }
}

#[derive(Debug)]
enum OnlineEvent {
    Transfer {
        sequence: u64,
        cycle: u64,
        tile: WeightTile,
    },
    Crossbar {
        sequence: u64,
        cycle: u64,
        transfer: PendingCrossbar,
    },
    BankRelease {
        sequence: u64,
        cycle: u64,
        bank_id: usize,
    },
    Core {
        sequence: u64,
        cycle: u64,
        core_id: usize,
        item: OnlineWorkItem,
        timing: OnlineTaskTiming,
    },
    Reduction {
        sequence: u64,
        cycle: u64,
        reduction: PendingReduction,
    },
    Completion {
        sequence: u64,
        cycle: u64,
        job_id: usize,
    },
}

impl OnlineEvent {
    fn order_key(&self) -> (u64, u64) {
        match self {
            Self::Transfer {
                sequence, cycle, ..
            }
            | Self::Crossbar {
                sequence, cycle, ..
            }
            | Self::BankRelease {
                sequence, cycle, ..
            }
            | Self::Core {
                sequence, cycle, ..
            }
            | Self::Reduction {
                sequence, cycle, ..
            }
            | Self::Completion {
                sequence, cycle, ..
            } => (*cycle, *sequence),
        }
    }
}

#[derive(Debug)]
struct OnlineOutputState {
    remaining_k_tiles: u64,
    valid_rows: u64,
    participants: BTreeSet<usize>,
    first_partial_cycle: Option<u64>,
    last_partial_cycle: u64,
}

fn projection_index(projection: &Projection) -> usize {
    if projection.name.ends_with("gate") {
        0
    } else if projection.name.ends_with("up") {
        1
    } else if projection.name.ends_with("down") {
        2
    } else {
        panic!("unknown Shared-MoE projection {}", projection.name)
    }
}

fn online_output_counts(jobs: &[LogicalJob]) -> Vec<[u64; 3]> {
    jobs.iter()
        .map(|job| {
            std::array::from_fn(|projection| {
                div_ceil(job.tokens, BASELINE_ROWS)
                    * (job.projections[projection].n / COMPILER_BLEN)
            })
        })
        .collect()
}

fn interleave_tiles_by_job(tiles: Vec<WeightTile>, jobs: usize) -> VecDeque<WeightTile> {
    let mut per_job = vec![VecDeque::new(); jobs];
    for tile in tiles {
        per_job[tile.job_id].push_back(tile);
    }
    let mut result = VecDeque::new();
    loop {
        let mut moved = false;
        for queue in &mut per_job {
            if let Some(tile) = queue.pop_front() {
                result.push_back(tile);
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    result
}

fn work_items_for_tile(
    tile: &WeightTile,
    job: &LogicalJob,
    architecture: ReplayArchitecture,
    next_work_id: &mut u64,
    next_ready_sequence: &mut u64,
) -> Vec<OnlineWorkItem> {
    let projection = projection_index(&tile.projection);
    let mut items = Vec::new();
    let waves = div_ceil(tile.tokens, BASELINE_ROWS);
    for wave in 0..waves {
        let consumed = wave * BASELINE_ROWS;
        let valid_rows = (tile.tokens - consumed).min(BASELINE_ROWS);
        for n_start in (tile.col_start..tile.col_stop).step_by(COMPILER_MLEN as usize) {
            let local_col_block = (n_start - tile.col_start) / COMPILER_MLEN;
            *next_work_id += 1;
            *next_ready_sequence += 1;
            items.push(OnlineWorkItem {
                id: *next_work_id,
                ready_sequence: *next_ready_sequence,
                tile_id: tile.id,
                key: OnlineOutputKey {
                    job_id: tile.job_id,
                    projection,
                    wave,
                    n_tile: n_start / COMPILER_BLEN,
                },
                n_tile_count: COMPILER_MLEN / COMPILER_BLEN,
                valid_rows,
                home_partition: tile_partition(architecture, tile, local_col_block),
                fixed_cycles_per_output: fixed_cycles_per_output_tile(job),
            });
        }
    }
    items
}

struct OnlineScheduler {
    architecture: ReplayArchitecture,
    jobs: Vec<LogicalJob>,
    sender: mpsc::UnboundedSender<OnlineEvent>,
    next_sequence: u64,
    next_work_id: u64,
    next_ready_sequence: u64,
    free_cores: BTreeSet<usize>,
    free_banks: BTreeSet<usize>,
    ready_work: VecDeque<OnlineWorkItem>,
    blocked_down: Vec<Vec<OnlineWorkItem>>,
    tile_remaining: BTreeMap<usize, u64>,
    outputs: BTreeMap<OnlineOutputKey, OnlineOutputState>,
    projection_remaining: Vec<[u64; 3]>,
    core_touched_outputs: Vec<BTreeSet<OnlineOutputKey>>,
    core_touched_projections: Vec<BTreeSet<(usize, usize)>>,
    crossbar_queue: VecDeque<PendingCrossbar>,
    crossbar_busy: bool,
    reducer_queue: VecDeque<PendingReduction>,
    reducer_busy: bool,
    completion_queue: VecDeque<usize>,
    completion_busy: bool,
    completed_jobs: usize,
    completion_times: Vec<u64>,
    compute_first_start: Option<u64>,
    compute_last_completion: u64,
    aggregate_core_busy_cycles: u64,
    aggregate_feed_cycles: u64,
    aggregate_fixed_cycles: u64,
    aggregate_launch_cycles: u64,
    aggregate_partition_wait_cycles: u64,
    max_partition_skew_cycles: u64,
    aggregate_partial_sum_wait_cycles: u64,
    max_partial_sum_lifetime_cycles: u64,
    crossbar_busy_cycles: u64,
    crossbar_queue_wait_cycles: u64,
    reducer_queue_wait_cycles: u64,
    reducer_busy_cycles: u64,
    interconnect_cycles: u64,
    synchronization_cycles: u64,
    completion_overhead_cycles: u64,
    interconnect_bytes: u64,
    reduction_output_elements: u64,
    reduction_add_elements: u64,
    scheduler_dispatch_cycles: u64,
    online_work_items: u64,
    online_completion_events: u64,
    online_work_steal_events: u64,
    max_ready_queue_depth: usize,
    scoreboard_output_entries: u64,
    core_busy_cycles_by_core: Vec<u64>,
    core_completed_work_items: Vec<u64>,
}

impl OnlineScheduler {
    fn new(
        architecture: ReplayArchitecture,
        jobs: Vec<LogicalJob>,
        sender: mpsc::UnboundedSender<OnlineEvent>,
    ) -> Self {
        assert!(architecture.is_online_scheduler());
        let core_count = architecture.partition_count();
        Self {
            architecture,
            projection_remaining: online_output_counts(&jobs),
            blocked_down: vec![Vec::new(); jobs.len()],
            jobs,
            sender,
            next_sequence: 0,
            next_work_id: 0,
            next_ready_sequence: 0,
            free_cores: (0..core_count).collect(),
            free_banks: (0..core_count).collect(),
            ready_work: VecDeque::new(),
            tile_remaining: BTreeMap::new(),
            outputs: BTreeMap::new(),
            core_touched_outputs: vec![BTreeSet::new(); core_count],
            core_touched_projections: vec![BTreeSet::new(); core_count],
            crossbar_queue: VecDeque::new(),
            crossbar_busy: false,
            reducer_queue: VecDeque::new(),
            reducer_busy: false,
            completion_queue: VecDeque::new(),
            completion_busy: false,
            completed_jobs: 0,
            completion_times: Vec::new(),
            compute_first_start: None,
            compute_last_completion: 0,
            aggregate_core_busy_cycles: 0,
            aggregate_feed_cycles: 0,
            aggregate_fixed_cycles: 0,
            aggregate_launch_cycles: 0,
            aggregate_partition_wait_cycles: 0,
            max_partition_skew_cycles: 0,
            aggregate_partial_sum_wait_cycles: 0,
            max_partial_sum_lifetime_cycles: 0,
            crossbar_busy_cycles: 0,
            crossbar_queue_wait_cycles: 0,
            reducer_queue_wait_cycles: 0,
            reducer_busy_cycles: 0,
            interconnect_cycles: 0,
            synchronization_cycles: 0,
            completion_overhead_cycles: 0,
            interconnect_bytes: 0,
            reduction_output_elements: 0,
            reduction_add_elements: 0,
            scheduler_dispatch_cycles: 0,
            online_work_items: 0,
            online_completion_events: 0,
            online_work_steal_events: 0,
            max_ready_queue_depth: 0,
            scoreboard_output_entries: 0,
            core_busy_cycles_by_core: vec![0; core_count],
            core_completed_work_items: vec![0; core_count],
        }
    }

    fn issue_sequence(&mut self) -> u64 {
        self.next_sequence += 1;
        self.next_sequence
    }

    fn down_ready(&self, job_id: usize) -> bool {
        self.projection_remaining[job_id][0] == 0 && self.projection_remaining[job_id][1] == 0
    }

    fn accept_tile(&mut self, tile: WeightTile) {
        let projection = projection_index(&tile.projection);
        let k_tiles = tile.projection.k / COMPILER_MLEN;
        let items = work_items_for_tile(
            &tile,
            &self.jobs[tile.job_id],
            self.architecture,
            &mut self.next_work_id,
            &mut self.next_ready_sequence,
        );
        assert!(!items.is_empty());
        assert!(
            self.tile_remaining
                .insert(tile.id, items.len() as u64)
                .is_none()
        );
        self.online_work_items += items.len() as u64;
        for item in items {
            for key in item.output_keys() {
                match self.outputs.entry(key) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(OnlineOutputState {
                            remaining_k_tiles: k_tiles,
                            valid_rows: item.valid_rows,
                            participants: BTreeSet::new(),
                            first_partial_cycle: None,
                            last_partial_cycle: 0,
                        });
                        self.scoreboard_output_entries += 1;
                    }
                    std::collections::btree_map::Entry::Occupied(entry) => {
                        assert!(entry.get().remaining_k_tiles <= k_tiles);
                        assert_eq!(entry.get().valid_rows, item.valid_rows);
                    }
                }
            }
            if projection == 2 && !self.down_ready(tile.job_id) {
                self.blocked_down[tile.job_id].push(item);
            } else {
                self.ready_work.push_back(item);
            }
        }
        self.max_ready_queue_depth = self.max_ready_queue_depth.max(self.ready_work.len());
    }

    fn task_timing(&self, core_id: usize, item: &OnlineWorkItem) -> OnlineTaskTiming {
        let rows = self.architecture.core_rows()[core_id];
        let row_passes = div_ceil(item.valid_rows, rows);
        let new_outputs = item
            .output_keys()
            .filter(|key| !self.core_touched_outputs[core_id].contains(key))
            .count() as u64;
        let fixed_cycles =
            (item.fixed_cycles_per_output * row_passes as f64 * new_outputs as f64).ceil() as u64;
        let projection = (item.key.job_id, item.key.projection);
        let launch_cycles = if self.core_touched_projections[core_id].contains(&projection) {
            0
        } else {
            PARTITION_LAUNCH_CYCLES
        };
        OnlineTaskTiming {
            feed_cycles: row_passes * COMPILER_MLEN * item.n_tile_count,
            fixed_cycles,
            launch_cycles,
            dispatch_cycles: ONLINE_DISPATCH_CYCLES,
        }
    }

    fn crossbar_transfer(&self, core_id: usize, item: &OnlineWorkItem) -> (u64, u64) {
        if core_id == item.home_partition {
            return (0, 0);
        }
        let rows = self.architecture.core_rows()[core_id];
        let row_passes = div_ceil(item.valid_rows, rows);
        let elements = COMPILER_MLEN * COMPILER_BLEN * item.n_tile_count;
        let bytes = row_passes * (elements + elements / MX_BLOCK);
        (div_ceil(bytes, ONLINE_CROSSBAR_BYTES_PER_CYCLE), bytes)
    }

    fn launch_core(&mut self, core_id: usize, item: OnlineWorkItem, timing: OnlineTaskTiming) {
        if core_id == item.home_partition {
            let bank_sequence = self.issue_sequence();
            let bank_sender = self.sender.clone();
            let bank_id = item.home_partition;
            let bank_hold_cycles =
                timing.dispatch_cycles + timing.launch_cycles + timing.feed_cycles;
            Executor::current().spawn(async move {
                Executor::current()
                    .resolve_at(PERIOD * bank_hold_cycles)
                    .await;
                let cycle = Executor::current()
                    .now()
                    .as_picos()
                    .div_ceil(PERIOD.as_picos().max(1));
                bank_sender
                    .send(OnlineEvent::BankRelease {
                        sequence: bank_sequence,
                        cycle,
                        bank_id,
                    })
                    .expect("online scheduler dropped before bank release");
            });
        }
        let sequence = self.issue_sequence();
        let sender = self.sender.clone();
        let start = Executor::current()
            .now()
            .as_picos()
            .div_ceil(PERIOD.as_picos().max(1));
        self.compute_first_start.get_or_insert(start);
        Executor::current().spawn(async move {
            Executor::current()
                .resolve_at(PERIOD * timing.total())
                .await;
            let cycle = Executor::current()
                .now()
                .as_picos()
                .div_ceil(PERIOD.as_picos().max(1));
            sender
                .send(OnlineEvent::Core {
                    sequence,
                    cycle,
                    core_id,
                    item,
                    timing,
                })
                .expect("online scheduler dropped before core completion");
        });
    }

    fn start_crossbar(&mut self) {
        if self.crossbar_busy {
            return;
        }
        let Some(transfer) = self.crossbar_queue.pop_front() else {
            return;
        };
        self.crossbar_busy = true;
        let now = Executor::current()
            .now()
            .as_picos()
            .div_ceil(PERIOD.as_picos().max(1));
        self.crossbar_queue_wait_cycles += now.saturating_sub(transfer.ready_cycle);
        self.crossbar_busy_cycles += transfer.cycles;
        let sender = self.sender.clone();
        let sequence = transfer.sequence;
        let cycles = transfer.cycles;
        Executor::current().spawn(async move {
            Executor::current().resolve_at(PERIOD * cycles).await;
            let cycle = Executor::current()
                .now()
                .as_picos()
                .div_ceil(PERIOD.as_picos().max(1));
            sender
                .send(OnlineEvent::Crossbar {
                    sequence,
                    cycle,
                    transfer,
                })
                .expect("online scheduler dropped before crossbar completion");
        });
    }

    fn dispatch(&mut self) {
        while !self.free_cores.is_empty() && !self.ready_work.is_empty() {
            let item_index = self
                .ready_work
                .iter()
                .enumerate()
                .filter(|(_, item)| self.free_banks.contains(&item.home_partition))
                .max_by_key(|(_, item)| {
                    (
                        item.valid_rows,
                        Reverse(item.ready_sequence),
                        Reverse(item.id),
                    )
                })
                .map(|(index, _)| index);
            let Some(item_index) = item_index else {
                break;
            };
            let item = self.ready_work.remove(item_index).unwrap();
            assert!(
                item.key.projection != 2 || self.down_ready(item.key.job_id),
                "down-projection work became schedulable before gate/up completed"
            );
            let (core_id, timing, crossbar_cycles, crossbar_bytes) = self
                .free_cores
                .iter()
                .map(|core_id| {
                    let timing = self.task_timing(*core_id, &item);
                    let (crossbar_cycles, crossbar_bytes) = self.crossbar_transfer(*core_id, &item);
                    (*core_id, timing, crossbar_cycles, crossbar_bytes)
                })
                .min_by_key(|(core_id, timing, crossbar, _)| (timing.total() + crossbar, *core_id))
                .unwrap();
            assert!(self.free_cores.remove(&core_id));
            assert!(self.free_banks.remove(&item.home_partition));
            self.core_touched_outputs[core_id].extend(item.output_keys());
            self.core_touched_projections[core_id].insert((item.key.job_id, item.key.projection));
            self.scheduler_dispatch_cycles += timing.dispatch_cycles;
            if crossbar_cycles > 0 {
                self.online_work_steal_events += 1;
                let sequence = self.issue_sequence();
                self.crossbar_queue.push_back(PendingCrossbar {
                    sequence,
                    core_id,
                    item,
                    timing,
                    cycles: crossbar_cycles,
                    bytes: crossbar_bytes,
                    ready_cycle: Executor::current()
                        .now()
                        .as_picos()
                        .div_ceil(PERIOD.as_picos().max(1)),
                });
            } else {
                self.launch_core(core_id, item, timing);
            }
        }
        self.max_ready_queue_depth = self.max_ready_queue_depth.max(self.ready_work.len());
        self.start_crossbar();
    }

    fn start_reducer(&mut self) {
        if self.reducer_busy {
            return;
        }
        let Some(reduction) = self.reducer_queue.pop_front() else {
            return;
        };
        self.reducer_busy = true;
        let now = Executor::current()
            .now()
            .as_picos()
            .div_ceil(PERIOD.as_picos().max(1));
        self.reducer_queue_wait_cycles += now.saturating_sub(reduction.ready_cycle);
        let sender = self.sender.clone();
        let sequence = reduction.sequence;
        let cycles = reduction.total_cycles();
        Executor::current().spawn(async move {
            Executor::current().resolve_at(PERIOD * cycles).await;
            let cycle = Executor::current()
                .now()
                .as_picos()
                .div_ceil(PERIOD.as_picos().max(1));
            sender
                .send(OnlineEvent::Reduction {
                    sequence,
                    cycle,
                    reduction,
                })
                .expect("online scheduler dropped before reduction completion");
        });
    }

    fn start_completion(&mut self) {
        if self.completion_busy {
            return;
        }
        let Some(job_id) = self.completion_queue.pop_front() else {
            return;
        };
        self.completion_busy = true;
        self.completion_overhead_cycles += COMPLETION_CYCLES;
        let sequence = self.issue_sequence();
        let sender = self.sender.clone();
        Executor::current().spawn(async move {
            Executor::current()
                .resolve_at(PERIOD * COMPLETION_CYCLES)
                .await;
            let cycle = Executor::current()
                .now()
                .as_picos()
                .div_ceil(PERIOD.as_picos().max(1));
            sender
                .send(OnlineEvent::Completion {
                    sequence,
                    cycle,
                    job_id,
                })
                .expect("online scheduler dropped before job completion");
        });
    }

    fn mark_output_complete(&mut self, key: OnlineOutputKey) {
        let projection_finished = {
            let remaining = &mut self.projection_remaining[key.job_id][key.projection];
            assert!(*remaining > 0);
            *remaining -= 1;
            *remaining == 0
        };
        if key.projection < 2 && self.down_ready(key.job_id) {
            let blocked = std::mem::take(&mut self.blocked_down[key.job_id]);
            self.ready_work.extend(blocked);
            self.max_ready_queue_depth = self.max_ready_queue_depth.max(self.ready_work.len());
        }
        if key.projection == 2 && projection_finished {
            self.completion_queue.push_back(key.job_id);
            self.start_completion();
        }
    }

    fn record_output_task_completion(&mut self, key: OnlineOutputKey, core_id: usize, cycle: u64) {
        let finished_output = {
            let output = self.outputs.get_mut(&key).unwrap();
            assert!(output.remaining_k_tiles > 0);
            output.remaining_k_tiles -= 1;
            output.participants.insert(core_id);
            output.first_partial_cycle.get_or_insert(cycle);
            output.last_partial_cycle = output.last_partial_cycle.max(cycle);
            if output.remaining_k_tiles == 0 {
                Some((
                    output.valid_rows,
                    output.participants.len() as u64,
                    output.first_partial_cycle.unwrap_or(cycle),
                    output.last_partial_cycle,
                ))
            } else {
                None
            }
        };
        let Some((valid_rows, parts, earliest, latest)) = finished_output else {
            return;
        };
        self.outputs.remove(&key);
        let lifetime = latest.saturating_sub(earliest);
        self.max_partial_sum_lifetime_cycles = self.max_partial_sum_lifetime_cycles.max(lifetime);
        self.aggregate_partial_sum_wait_cycles += lifetime;
        let elements = valid_rows * COMPILER_BLEN;
        let (reduce, interconnect, sync, bytes) = reduction_cost(elements, parts);
        if reduce + interconnect + sync > 0 {
            let sequence = self.issue_sequence();
            self.reducer_queue.push_back(PendingReduction {
                sequence,
                key,
                reduce_cycles: reduce,
                interconnect_cycles: interconnect,
                synchronization_cycles: sync,
                interconnect_bytes: bytes,
                output_elements: elements,
                add_elements: elements * parts.saturating_sub(1),
                ready_cycle: cycle,
            });
            self.start_reducer();
        } else {
            self.mark_output_complete(key);
        }
    }

    fn handle_core_completion(
        &mut self,
        cycle: u64,
        core_id: usize,
        item: OnlineWorkItem,
        timing: OnlineTaskTiming,
    ) -> usize {
        assert!(self.free_cores.insert(core_id));
        self.online_completion_events += 1;
        self.core_completed_work_items[core_id] += 1;
        self.core_busy_cycles_by_core[core_id] += timing.total();
        self.aggregate_core_busy_cycles += timing.total();
        self.aggregate_feed_cycles += timing.feed_cycles;
        self.aggregate_fixed_cycles += timing.fixed_cycles;
        self.aggregate_launch_cycles += timing.launch_cycles + timing.dispatch_cycles;
        self.compute_last_completion = self.compute_last_completion.max(cycle);

        let tile_remaining = self.tile_remaining.get_mut(&item.tile_id).unwrap();
        assert!(*tile_remaining > 0);
        *tile_remaining -= 1;
        let released_tiles = if *tile_remaining == 0 {
            self.tile_remaining.remove(&item.tile_id);
            1
        } else {
            0
        };

        let output_keys = item.output_keys().collect::<Vec<_>>();
        for key in output_keys {
            self.record_output_task_completion(key, core_id, cycle);
        }
        released_tiles
    }

    fn handle_event(&mut self, event: OnlineEvent) -> usize {
        match event {
            OnlineEvent::Transfer { tile, .. } => {
                self.accept_tile(tile);
                0
            }
            OnlineEvent::Crossbar { transfer, .. } => {
                assert!(self.crossbar_busy);
                self.crossbar_busy = false;
                assert!(self.free_banks.insert(transfer.item.home_partition));
                self.interconnect_cycles += transfer.cycles;
                self.interconnect_bytes += transfer.bytes;
                self.launch_core(transfer.core_id, transfer.item, transfer.timing);
                self.start_crossbar();
                0
            }
            OnlineEvent::BankRelease { bank_id, .. } => {
                assert!(self.free_banks.insert(bank_id));
                0
            }
            OnlineEvent::Core {
                cycle,
                core_id,
                item,
                timing,
                ..
            } => self.handle_core_completion(cycle, core_id, item, timing),
            OnlineEvent::Reduction { reduction, .. } => {
                assert!(self.reducer_busy);
                self.reducer_busy = false;
                self.reducer_busy_cycles += reduction.reduce_cycles;
                self.interconnect_cycles += reduction.interconnect_cycles;
                self.synchronization_cycles += reduction.synchronization_cycles;
                self.interconnect_bytes += reduction.interconnect_bytes;
                self.reduction_output_elements += reduction.output_elements;
                self.reduction_add_elements += reduction.add_elements;
                self.mark_output_complete(reduction.key);
                self.start_reducer();
                0
            }
            OnlineEvent::Completion { cycle, job_id, .. } => {
                assert!(self.completion_busy);
                self.completion_busy = false;
                assert_eq!(self.projection_remaining[job_id][2], 0);
                self.completed_jobs += 1;
                self.completion_times.push(cycle);
                self.start_completion();
                0
            }
        }
    }

    fn all_cores_free(&self) -> bool {
        self.free_cores.len() == self.architecture.partition_count()
    }

    fn has_scheduled_event(&self, transfer_events: usize) -> bool {
        transfer_events > 0
            || !self.all_cores_free()
            || self.crossbar_busy
            || self.reducer_busy
            || self.completion_busy
    }

    fn bank_port_blocked(&self) -> bool {
        !self.free_cores.is_empty()
            && !self.ready_work.is_empty()
            && !self
                .ready_work
                .iter()
                .any(|item| self.free_banks.contains(&item.home_partition))
    }
}

fn launch_transfer(
    hbm: Arc<dyn ErasedMemoryModel>,
    architecture: ReplayArchitecture,
    tile: WeightTile,
    launch_sequence: u64,
    sender: mpsc::UnboundedSender<TransferComplete>,
) {
    Executor::current().spawn(async move {
        let transfers = (0..architecture.partition_count())
            .map(|partition| transfer_partition(hbm.clone(), tile.clone(), architecture, partition))
            .collect::<Vec<_>>();
        join_all(transfers).await;
        let completion_cycle = Executor::current()
            .now()
            .as_picos()
            .div_ceil(PERIOD.as_picos().max(1));
        sender
            .send(TransferComplete {
                launch_sequence,
                tile,
                completion_cycle,
            })
            .expect("request replay controller dropped before HBM completion");
    });
}

fn launch_online_transfer(
    hbm: Arc<dyn ErasedMemoryModel>,
    architecture: ReplayArchitecture,
    tile: WeightTile,
    sequence: u64,
    sender: mpsc::UnboundedSender<OnlineEvent>,
) {
    Executor::current().spawn(async move {
        let transfers = (0..architecture.partition_count())
            .map(|partition| transfer_partition(hbm.clone(), tile.clone(), architecture, partition))
            .collect::<Vec<_>>();
        join_all(transfers).await;
        let cycle = Executor::current()
            .now()
            .as_picos()
            .div_ceil(PERIOD.as_picos().max(1));
        sender
            .send(OnlineEvent::Transfer {
                sequence,
                cycle,
                tile,
            })
            .expect("online scheduler dropped before HBM completion");
    });
}

fn launch_online_pending(
    hbm: &Arc<dyn ErasedMemoryModel>,
    architecture: ReplayArchitecture,
    pending: &mut VecDeque<WeightTile>,
    in_flight: &mut usize,
    transfer_events: &mut usize,
    scheduler: &mut OnlineScheduler,
) {
    while *in_flight < architecture.buffer_depth() {
        let Some(tile) = pending.pop_front() else {
            break;
        };
        let sequence = scheduler.issue_sequence();
        launch_online_transfer(
            hbm.clone(),
            architecture,
            tile,
            sequence,
            scheduler.sender.clone(),
        );
        *in_flight += 1;
        *transfer_events += 1;
    }
}

async fn online_replay_controller(
    hbm: Arc<dyn ErasedMemoryModel>,
    jobs: Vec<LogicalJob>,
    architecture: ReplayArchitecture,
) -> RequestReplayRun {
    assert!(architecture.is_online_scheduler());
    let total_jobs = jobs.len();
    let tiles = build_weight_tiles(&jobs, architecture);
    let total_tiles = tiles.len();
    let mut pending = interleave_tiles_by_job(tiles, total_jobs);
    let (sender, mut receiver) = mpsc::unbounded_channel();
    let mut scheduler = OnlineScheduler::new(architecture, jobs, sender);
    let mut in_flight = 0_usize;
    let mut transfer_events = 0_usize;
    let mut hbm_last_completion = 0_u64;
    let mut hbm_starved = 0_u64;
    let mut bank_port_stall = 0_u64;
    let mut released_tiles = 0_usize;
    let mut pending_events = BTreeMap::<(u64, u64), OnlineEvent>::new();
    let mut observation_cycle = 0_u64;

    launch_online_pending(
        &hbm,
        architecture,
        &mut pending,
        &mut in_flight,
        &mut transfer_events,
        &mut scheduler,
    );
    while scheduler.completed_jobs < total_jobs {
        scheduler.dispatch();
        if pending_events.is_empty() {
            assert!(
                scheduler.has_scheduled_event(transfer_events),
                "online scheduler deadlocked with no scheduled completion"
            );
            let first = receiver
                .recv()
                .await
                .expect("online scheduler workers exited before all jobs completed");
            assert!(pending_events.insert(first.order_key(), first).is_none());
            while let Ok(event) = receiver.try_recv() {
                assert!(pending_events.insert(event.order_key(), event).is_none());
            }
        }

        let next_cycle = pending_events.keys().next().unwrap().0;
        let all_idle_for_hbm =
            scheduler.all_cores_free() && scheduler.ready_work.is_empty() && transfer_events > 0;
        if all_idle_for_hbm {
            hbm_starved += next_cycle.saturating_sub(observation_cycle);
        }
        if scheduler.bank_port_blocked() {
            bank_port_stall += next_cycle.saturating_sub(observation_cycle);
        }
        observation_cycle = next_cycle;

        let keys = pending_events
            .range((next_cycle, 0)..=(next_cycle, u64::MAX))
            .map(|(key, _)| *key)
            .collect::<Vec<_>>();
        for key in keys {
            let event = pending_events.remove(&key).unwrap();
            if let OnlineEvent::Transfer { cycle, .. } = &event {
                transfer_events -= 1;
                hbm_last_completion = hbm_last_completion.max(*cycle);
            }
            let released = scheduler.handle_event(event);
            if released > 0 {
                assert!(in_flight >= released);
                in_flight -= released;
                released_tiles += released;
            }
        }
        launch_online_pending(
            &hbm,
            architecture,
            &mut pending,
            &mut in_flight,
            &mut transfer_events,
            &mut scheduler,
        );
    }

    assert!(pending.is_empty());
    assert!(pending_events.is_empty());
    assert_eq!(transfer_events, 0);
    assert_eq!(in_flight, 0);
    assert_eq!(released_tiles, total_tiles);
    assert!(scheduler.tile_remaining.is_empty());
    assert!(scheduler.outputs.is_empty());
    assert!(
        scheduler
            .projection_remaining
            .iter()
            .all(|remaining| remaining.iter().all(|count| *count == 0))
    );
    assert!(scheduler.ready_work.is_empty());
    assert!(scheduler.blocked_down.iter().all(Vec::is_empty));
    assert!(scheduler.all_cores_free());
    assert_eq!(scheduler.free_banks.len(), architecture.partition_count());
    assert!(!scheduler.crossbar_busy && scheduler.crossbar_queue.is_empty());
    assert!(!scheduler.reducer_busy && scheduler.reducer_queue.is_empty());
    assert!(!scheduler.completion_busy && scheduler.completion_queue.is_empty());

    let total_cycles = Executor::current()
        .now()
        .as_picos()
        .div_ceil(PERIOD.as_picos().max(1));
    let stats = hbm
        .statistics()
        .expect("online request replay HBM must expose stats");
    let min_completion = scheduler
        .completion_times
        .iter()
        .copied()
        .min()
        .unwrap_or(0);
    let max_completion = scheduler
        .completion_times
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let compute_wall_cycles = scheduler
        .compute_first_start
        .map(|start| scheduler.compute_last_completion.saturating_sub(start))
        .unwrap_or(0);
    let cycle_accounting_passed = scheduler.completed_jobs == total_jobs
        && released_tiles == total_tiles
        && scheduler.online_completion_events == scheduler.online_work_items;
    RequestReplayRun {
        architecture: architecture.name().to_string(),
        mapping: architecture.mapping(),
        core_rows: architecture.core_rows().to_vec(),
        buffer_depth: architecture.buffer_depth(),
        buffer_unit: "one 64x(up to 1024) MX weight macro-tile",
        max_weight_buffer_bytes: architecture.buffer_depth() as u64 * WEIGHT_TILE_STORED_BYTES,
        weight_buffer_banks: architecture.buffer_depth(),
        matrix_read_ports: architecture.buffer_depth(),
        matrix_write_ports: architecture.buffer_depth(),
        equal_area_pes: BASELINE_PES,
        completed_jobs: scheduler.completed_jobs,
        total_cycles,
        executor_drained_cycles: 0,
        // Online resources overlap, so the critical-path duration is the only
        // additive wall-clock quantity. Resource busy counters are reported
        // separately and may sum to more than total_cycles.
        accounted_controller_cycles: total_cycles,
        cycle_accounting_passed,
        hbm_physical_bytes: stats.total_bytes_read,
        hbm_read_requests_64b: stats.total_bytes_read / HBM_BURST_BYTES,
        hbm_last_completion_cycle: hbm_last_completion,
        hbm_starved_cycles: hbm_starved,
        compute_wall_cycles,
        aggregate_core_busy_cycles: scheduler.aggregate_core_busy_cycles,
        aggregate_feed_cycles: scheduler.aggregate_feed_cycles,
        aggregate_fixed_cycles: scheduler.aggregate_fixed_cycles,
        aggregate_launch_cycles: scheduler.aggregate_launch_cycles,
        aggregate_partition_wait_cycles: scheduler.aggregate_partition_wait_cycles,
        max_partition_skew_cycles: scheduler.max_partition_skew_cycles,
        reducer_busy_cycles: scheduler.reducer_busy_cycles,
        interconnect_cycles: scheduler.interconnect_cycles,
        synchronization_cycles: scheduler.synchronization_cycles,
        completion_overhead_cycles: scheduler.completion_overhead_cycles,
        interconnect_bytes: scheduler.interconnect_bytes,
        reduction_output_elements: scheduler.reduction_output_elements,
        reduction_add_elements: scheduler.reduction_add_elements,
        reduction_elements: scheduler.reduction_output_elements,
        completion_skew_cycles: scheduler.max_partition_skew_cycles,
        job_completion_span_cycles: max_completion.saturating_sub(min_completion),
        scheduler_kind: "completion_driven_earliest_finish_with_locality",
        scheduler_dispatch_cycles: scheduler.scheduler_dispatch_cycles,
        online_work_items: scheduler.online_work_items,
        online_completion_events: scheduler.online_completion_events,
        online_work_steal_events: scheduler.online_work_steal_events,
        max_ready_queue_depth: scheduler.max_ready_queue_depth,
        scoreboard_output_entries: scheduler.scoreboard_output_entries,
        core_busy_cycles_by_core: scheduler.core_busy_cycles_by_core,
        core_completed_work_items: scheduler.core_completed_work_items,
        aggregate_partial_sum_wait_cycles: scheduler.aggregate_partial_sum_wait_cycles,
        max_partial_sum_lifetime_cycles: scheduler.max_partial_sum_lifetime_cycles,
        aggregate_core_idle_cycles: total_cycles
            .saturating_mul(architecture.partition_count() as u64)
            .saturating_sub(scheduler.aggregate_core_busy_cycles),
        crossbar_busy_cycles: scheduler.crossbar_busy_cycles,
        crossbar_queue_wait_cycles: scheduler.crossbar_queue_wait_cycles,
        reducer_queue_wait_cycles: scheduler.reducer_queue_wait_cycles,
        bank_port_stall_cycles: bank_port_stall,
    }
}

async fn replay_controller(
    hbm: Arc<dyn ErasedMemoryModel>,
    jobs: Vec<LogicalJob>,
    architecture: ReplayArchitecture,
) -> RequestReplayRun {
    let (sender, mut receiver) = mpsc::unbounded_channel();
    let total_jobs = jobs.len();
    let tiles = build_weight_tiles(&jobs, architecture);
    let total_tiles = tiles.len();
    let mut pending = VecDeque::from(tiles);
    let mut ready = BTreeMap::new();
    let mut in_flight = 0_usize;
    let mut launch_sequence = 0_u64;
    let mut next_tile_id = 0_usize;
    let mut completed = 0_usize;
    let mut hbm_last_completion = 0_u64;
    let mut hbm_starved = 0_u64;
    let mut compute_wall = 0_u64;
    let mut aggregate_core_busy = 0_u64;
    let mut aggregate_feed = 0_u64;
    let mut aggregate_fixed = 0_u64;
    let mut aggregate_launch = 0_u64;
    let mut aggregate_partition_wait = 0_u64;
    let mut max_partition_skew = 0_u64;
    let mut reducer_busy = 0_u64;
    let mut interconnect_cycles = 0_u64;
    let mut synchronization_cycles = 0_u64;
    let mut interconnect_bytes = 0_u64;
    let mut reduction_output_elements = 0_u64;
    let mut reduction_add_elements = 0_u64;
    let mut reduction_elements = 0_u64;
    let mut completion_times = Vec::new();

    let launch_pending =
        |pending: &mut VecDeque<WeightTile>, in_flight: &mut usize, launch_sequence: &mut u64| {
            while *in_flight < architecture.buffer_depth() {
                let Some(tile) = pending.pop_front() else {
                    break;
                };
                *launch_sequence += 1;
                launch_transfer(
                    hbm.clone(),
                    architecture,
                    tile,
                    *launch_sequence,
                    sender.clone(),
                );
                *in_flight += 1;
            }
        };

    launch_pending(&mut pending, &mut in_flight, &mut launch_sequence);
    while next_tile_id < total_tiles {
        let wait_start = Executor::current().now();
        while !ready.contains_key(&next_tile_id) {
            let transfer = receiver
                .recv()
                .await
                .expect("all request replay HBM workers exited early");
            hbm_last_completion = hbm_last_completion.max(transfer.completion_cycle);
            let _ = transfer.launch_sequence;
            assert!(ready.insert(transfer.tile.id, transfer.tile).is_none());
        }
        let wait_end = Executor::current().now();
        hbm_starved += (wait_end - wait_start)
            .as_picos()
            .div_ceil(PERIOD.as_picos().max(1));
        let tile = ready.remove(&next_tile_id).unwrap();
        let _ = tile.job_id;
        let _ = tile.expert_id;
        let _ = tile.projection.name;
        let compute = tile.compute;
        if compute.compute_cycles > 0 {
            Executor::current()
                .resolve_at(PERIOD * compute.compute_cycles)
                .await;
        }
        if compute.finalize_cycles > 0 {
            Executor::current()
                .resolve_at(PERIOD * compute.finalize_cycles)
                .await;
        }
        if tile.job_last && !matches!(architecture, ReplayArchitecture::Baseline { .. }) {
            Executor::current()
                .resolve_at(PERIOD * COMPLETION_CYCLES)
                .await;
        }
        compute_wall += compute.compute_cycles;
        aggregate_core_busy += compute.aggregate_core_busy_cycles;
        aggregate_feed += compute.aggregate_feed_cycles;
        aggregate_fixed += compute.aggregate_fixed_cycles;
        aggregate_launch += compute.aggregate_launch_cycles;
        aggregate_partition_wait += compute.aggregate_partition_wait_cycles;
        max_partition_skew = max_partition_skew.max(compute.max_partition_skew_cycles);
        reducer_busy += compute.reducer_cycles;
        interconnect_cycles += compute.interconnect_cycles;
        synchronization_cycles += compute.synchronization_cycles;
        interconnect_bytes += compute.interconnect_bytes;
        reduction_output_elements += compute.reduction_output_elements;
        reduction_add_elements += compute.reduction_add_elements;
        reduction_elements += compute.reduction_elements;
        in_flight -= 1;
        if tile.job_last {
            completed += 1;
            completion_times.push(
                Executor::current()
                    .now()
                    .as_picos()
                    .div_ceil(PERIOD.as_picos().max(1)),
            );
        }
        next_tile_id += 1;
        launch_pending(&mut pending, &mut in_flight, &mut launch_sequence);
    }
    assert_eq!(completed, total_jobs);

    let total_cycles = Executor::current()
        .now()
        .as_picos()
        .div_ceil(PERIOD.as_picos().max(1));
    let stats = hbm
        .statistics()
        .expect("request replay HBM must expose stats");
    let min_completion = completion_times.iter().copied().min().unwrap_or(0);
    let max_completion = completion_times.iter().copied().max().unwrap_or(0);
    let completion_overhead_cycles = if matches!(architecture, ReplayArchitecture::Baseline { .. })
    {
        0
    } else {
        completed as u64 * COMPLETION_CYCLES
    };
    let accounted_controller_cycles = hbm_starved
        + compute_wall
        + reducer_busy
        + interconnect_cycles
        + synchronization_cycles
        + completion_overhead_cycles;
    let cycle_accounting_passed = accounted_controller_cycles == total_cycles;
    assert!(
        cycle_accounting_passed,
        "request replay cycle stack does not close: accounted={accounted_controller_cycles}, total={total_cycles}"
    );
    RequestReplayRun {
        architecture: architecture.name().to_string(),
        mapping: architecture.mapping(),
        core_rows: architecture.core_rows().to_vec(),
        buffer_depth: architecture.buffer_depth(),
        buffer_unit: "one 64x(up to 1024) MX weight macro-tile",
        max_weight_buffer_bytes: architecture.buffer_depth() as u64 * WEIGHT_TILE_STORED_BYTES,
        weight_buffer_banks: architecture.buffer_depth(),
        matrix_read_ports: architecture.buffer_depth(),
        matrix_write_ports: architecture.buffer_depth(),
        equal_area_pes: BASELINE_PES,
        completed_jobs: completed,
        total_cycles,
        executor_drained_cycles: 0,
        accounted_controller_cycles,
        cycle_accounting_passed,
        hbm_physical_bytes: stats.total_bytes_read,
        hbm_read_requests_64b: stats.total_bytes_read / HBM_BURST_BYTES,
        hbm_last_completion_cycle: hbm_last_completion,
        hbm_starved_cycles: hbm_starved,
        compute_wall_cycles: compute_wall,
        aggregate_core_busy_cycles: aggregate_core_busy,
        aggregate_feed_cycles: aggregate_feed,
        aggregate_fixed_cycles: aggregate_fixed,
        aggregate_launch_cycles: aggregate_launch,
        aggregate_partition_wait_cycles: aggregate_partition_wait,
        max_partition_skew_cycles: max_partition_skew,
        reducer_busy_cycles: reducer_busy,
        interconnect_cycles,
        synchronization_cycles,
        completion_overhead_cycles,
        interconnect_bytes,
        reduction_output_elements,
        reduction_add_elements,
        reduction_elements,
        completion_skew_cycles: max_partition_skew,
        job_completion_span_cycles: max_completion.saturating_sub(min_completion),
        scheduler_kind: "ordered_macro_tile_controller",
        scheduler_dispatch_cycles: 0,
        online_work_items: 0,
        online_completion_events: 0,
        online_work_steal_events: 0,
        max_ready_queue_depth: 0,
        scoreboard_output_entries: 0,
        core_busy_cycles_by_core: Vec::new(),
        core_completed_work_items: Vec::new(),
        aggregate_partial_sum_wait_cycles: 0,
        max_partial_sum_lifetime_cycles: 0,
        aggregate_core_idle_cycles: total_cycles
            .saturating_mul(architecture.partition_count() as u64)
            .saturating_sub(aggregate_core_busy),
        crossbar_busy_cycles: 0,
        crossbar_queue_wait_cycles: 0,
        reducer_queue_wait_cycles: 0,
        bank_port_stall_cycles: 0,
    }
}

async fn run_isolated_replay(
    jobs: Vec<LogicalJob>,
    architecture: ReplayArchitecture,
) -> RequestReplayRun {
    tokio::task::spawn_blocking(move || {
        let host_runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("failed to create isolated request-replay host runtime");
        host_runtime.block_on(async move {
            let executor = Executor::new();
            let result = Arc::new(Mutex::new(None));
            let output = result.clone();
            executor.spawn(async move {
                let typed = Arc::new(memory::WithStats::new(memory::WithTiming::new(
                    ManuallyDrop::new(
                        ramulator::Ramulator::hbm2_preset(HBM_CHANNELS)
                            .expect("failed to create request-replay Ramulator"),
                    ),
                    memory::NoData,
                )));
                let hbm: Arc<dyn ErasedMemoryModel> = typed;
                let run = if architecture.is_online_scheduler() {
                    online_replay_controller(hbm, jobs, architecture).await
                } else {
                    replay_controller(hbm, jobs, architecture).await
                };
                *output.lock().unwrap() = Some(run);
            });
            executor.enter(Instant::ETERNITY).await;
            let drained = executor.now().as_picos().div_ceil(PERIOD.as_picos().max(1));
            let mut run = result
                .lock()
                .unwrap()
                .take()
                .expect("request replay did not produce a result");
            run.executor_drained_cycles = drained;
            run
        })
    })
    .await
    .expect("isolated request replay worker panicked")
}

async fn request_replay_comparison(
    architecture: ExperimentalMatrixArchitecture,
    manifest: &GroupedManifest,
) -> RequestReplayComparison {
    let jobs = jobs_from_manifest(manifest);
    let expected_jobs = jobs.len();
    let (candidate_arch, baseline_arch) = match architecture {
        ExperimentalMatrixArchitecture::Dual2x1024NsplitDepth2 => (
            ReplayArchitecture::DualNsplitDepth2,
            ReplayArchitecture::Baseline { buffer_depth: 2 },
        ),
        ExperimentalMatrixArchitecture::Dual2x1024KsplitDepth2 => (
            ReplayArchitecture::DualKsplitDepth2,
            ReplayArchitecture::Baseline { buffer_depth: 2 },
        ),
        ExperimentalMatrixArchitecture::Dual2x1024OnlineStreamKDepth2 => (
            ReplayArchitecture::DualOnlineStreamKDepth2,
            ReplayArchitecture::Baseline { buffer_depth: 2 },
        ),
        ExperimentalMatrixArchitecture::Asym211StreamKDepth3 => (
            ReplayArchitecture::Asym211StreamKDepth3,
            ReplayArchitecture::Baseline { buffer_depth: 3 },
        ),
        ExperimentalMatrixArchitecture::Asym211OnlineStreamKDepth3 => (
            ReplayArchitecture::Asym211OnlineStreamKDepth3,
            ReplayArchitecture::Baseline { buffer_depth: 3 },
        ),
        ExperimentalMatrixArchitecture::Asym211KsplitDepth3 => (
            ReplayArchitecture::Asym211KsplitDepth3,
            ReplayArchitecture::Baseline { buffer_depth: 3 },
        ),
        ExperimentalMatrixArchitecture::Quad1x1024MsplitDepth4 => (
            ReplayArchitecture::QuadMsplitDepth4,
            ReplayArchitecture::Baseline { buffer_depth: 4 },
        ),
        ExperimentalMatrixArchitecture::Quad1x1024NsplitDepth4 => (
            ReplayArchitecture::QuadNsplitDepth4,
            ReplayArchitecture::Baseline { buffer_depth: 4 },
        ),
        ExperimentalMatrixArchitecture::Quad1x1024KsplitDepth4 => (
            ReplayArchitecture::QuadKsplitDepth4,
            ReplayArchitecture::Baseline { buffer_depth: 4 },
        ),
        ExperimentalMatrixArchitecture::Quad1x1024StageNToKDepth4 => (
            ReplayArchitecture::QuadStageNToKDepth4,
            ReplayArchitecture::Baseline { buffer_depth: 4 },
        ),
        ExperimentalMatrixArchitecture::Quad1x1024StreamKDepth4 => (
            ReplayArchitecture::QuadStreamKDepth4,
            ReplayArchitecture::Baseline { buffer_depth: 4 },
        ),
        ExperimentalMatrixArchitecture::Quad1x1024OnlineStreamKDepth4 => (
            ReplayArchitecture::QuadOnlineStreamKDepth4,
            ReplayArchitecture::Baseline { buffer_depth: 4 },
        ),
        _ => unreachable!(),
    };
    validate_replay_contract(&jobs, baseline_arch)
        .unwrap_or_else(|err| panic!("baseline replay contract failed: {err}"));
    validate_replay_contract(&jobs, candidate_arch)
        .unwrap_or_else(|err| panic!("candidate replay contract failed: {err}"));
    let (baseline, candidate) = tokio::join!(
        run_isolated_replay(jobs.clone(), baseline_arch),
        run_isolated_replay(jobs, candidate_arch),
    );
    let hbm_byte_equality_passed = baseline.hbm_physical_bytes == candidate.hbm_physical_bytes;
    let completed_job_equality_passed =
        baseline.completed_jobs == expected_jobs && candidate.completed_jobs == expected_jobs;
    let equal_pe_count_passed = baseline.equal_area_pes == candidate.equal_area_pes;
    let equal_weight_buffer_bytes_passed = baseline.max_weight_buffer_bytes
        == candidate.max_weight_buffer_bytes
        && baseline.weight_buffer_banks == candidate.weight_buffer_banks
        && baseline.matrix_read_ports == candidate.matrix_read_ports
        && baseline.matrix_write_ports == candidate.matrix_write_ports;
    let cycle_accounting_passed =
        baseline.cycle_accounting_passed && candidate.cycle_accounting_passed;
    let executor_drain_equality_passed = baseline.total_cycles == baseline.executor_drained_cycles
        && candidate.total_cycles == candidate.executor_drained_cycles;
    assert!(
        hbm_byte_equality_passed,
        "candidate changed physical HBM bytes"
    );
    assert!(
        completed_job_equality_passed,
        "request replay did not complete every shared/routed expert job"
    );
    assert!(equal_pe_count_passed, "candidate changed the PE count");
    assert!(
        equal_weight_buffer_bytes_passed,
        "candidate and baseline do not have the same global weight-buffer resource contract"
    );
    assert!(
        cycle_accounting_passed,
        "request replay cycle stack did not close"
    );
    assert!(
        executor_drain_equality_passed,
        "request replay left simulation work after the reported total"
    );
    RequestReplayComparison {
        timing_source: "isolated runtime::Executor + 8-channel Ramulator request/tick/callback",
        address_source: "compiler MX layout reconstructed from manifest dimensions and true expert ids",
        physical_byte_contract: "one 64B element burst and one 64B rounded scale burst per MX row",
        tensor_partition_contract: "M is integral; N and K request boundaries are 64-element aligned; every 64x64 request block is assigned exactly once",
        speedup_request_region: baseline.total_cycles as f64 / candidate.total_cycles as f64,
        baseline,
        candidate,
        hbm_byte_equality_passed,
        completed_job_equality_passed,
        equal_pe_count_passed,
        equal_weight_buffer_bytes_passed,
        tensor_slice_alignment_passed: true,
        request_block_coverage_passed: true,
        cycle_accounting_passed,
        executor_drain_equality_passed,
    }
}

fn read_and_validate_manifest(manifest_path: &Path) -> GroupedManifest {
    let manifest: GroupedManifest =
        serde_json::from_slice(&std::fs::read(manifest_path).unwrap_or_else(|err| {
            panic!("failed to read matrix DSE manifest {manifest_path:?}: {err}")
        }))
        .unwrap_or_else(|err| {
            panic!("failed to parse matrix DSE manifest {manifest_path:?}: {err}")
        });
    assert_eq!(
        manifest.group_sizes.values().sum::<u64>(),
        manifest.batch * manifest.top_k,
        "group sizes must account for every routed token/expert pair"
    );
    if !manifest.selected_experts.is_empty() {
        let selected = manifest
            .selected_experts
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>();
        assert!(
            selected
                .iter()
                .all(|expert| manifest.group_sizes.contains_key(expert)),
            "selected_experts and group_sizes disagree"
        );
    }
    manifest
}

#[allow(dead_code)]
pub(crate) async fn evaluate_request_replay_only(
    architecture: ExperimentalMatrixArchitecture,
    manifest_path: &Path,
    output_path: &Path,
) {
    assert!(
        architecture.is_request_replay(),
        "request-replay-only accepts only flag-isolated physical DSE architectures"
    );
    let manifest = read_and_validate_manifest(manifest_path);
    let comparison = request_replay_comparison(architecture, &manifest).await;
    let limitations = if matches!(
        architecture,
        ExperimentalMatrixArchitecture::Dual2x1024OnlineStreamKDepth2
            | ExperimentalMatrixArchitecture::Asym211OnlineStreamKDepth3
            | ExperimentalMatrixArchitecture::Quad1x1024OnlineStreamKDepth4
    ) {
        vec![
            "manifest was produced by a separately validated functional opcode run",
            "request replay models expert/shared weight traffic, not router/gather/scatter vector traffic",
            "shape-scaled compute and reducer fixed costs remain provisional until RTL calibration",
            "online scheduling uses physical 64x64 MX request-block tasks, an 8-cycle dispatch assumption, one read port per Weight SRAM bank, and one shared 64B/cycle SRAM crossbar",
            "completion scoreboard/reducer timing is modelled, but SRAM macro PPA and candidate frequency require RTL synthesis",
        ]
    } else {
        vec![
            "manifest was produced by a separately validated functional opcode run",
            "request replay models expert/shared weight traffic, not router/gather/scatter vector traffic",
            "shape-scaled compute and reducer fixed costs remain provisional until RTL calibration",
            "Stream-K stages each 64x64 HBM request once; sub-burst BLEN work distribution through the SRAM crossbar is not cycle-modelled",
            "M-split fetches each weight block once; on-chip multicast distribution is structural but not cycle-modelled",
        ]
    };
    let report = RequestReplayOnlyReport {
        schema_version: 5,
        evidence_tier: "request-level Ramulator replay with shape-scaled transferred compute calibration; not RTL sign-off",
        model: manifest.model,
        model_architecture: manifest.architecture,
        workload: manifest.workload,
        layer: manifest.layer,
        step: manifest.step,
        batch: manifest.batch,
        top_k: manifest.top_k,
        active_routed_experts: manifest.group_sizes.len(),
        routed_pairs: manifest.group_sizes.values().sum(),
        comparison,
        limitations,
    };
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).unwrap_or_else(|err| {
            panic!("failed to create request-replay output directory {parent:?}: {err}")
        });
    }
    std::fs::write(
        output_path,
        serde_json::to_string_pretty(&report).unwrap() + "\n",
    )
    .unwrap_or_else(|err| panic!("failed to write request-replay report {output_path:?}: {err}"));
}

pub(crate) async fn evaluate(
    architecture: ExperimentalMatrixArchitecture,
    manifest_path: &Path,
    output_path: &Path,
    fixed: FixedBigTailConfig,
    serial_total_picos: u64,
    measured_matrix_picos: u64,
) -> MatrixArchitectureReport {
    let manifest = read_and_validate_manifest(manifest_path);

    let period = PERIOD.as_picos().max(1);
    let serial_total_cycles = serial_total_picos.div_ceil(period);
    let measured_baseline_matrix_cycles = measured_matrix_picos.div_ceil(period);
    let reconstructed_baseline_matrix_cycles = baseline_matrix_cycles(&manifest);
    assert_eq!(
        measured_baseline_matrix_cycles, reconstructed_baseline_matrix_cycles,
        "calibrated matrix-wave reconstruction drifted from the measured stage profile"
    );

    let request_replay = if architecture.is_request_replay() {
        Some(request_replay_comparison(architecture, &manifest).await)
    } else {
        None
    };

    let (name, candidate_matrix_cycles, adjusted_total_cycles, speedup, geometry) =
        if let Some(comparison) = request_replay.as_ref() {
            (
                comparison.candidate.architecture.clone(),
                None,
                None,
                None,
                match architecture {
                    ExperimentalMatrixArchitecture::Dual2x1024NsplitDepth2
                    | ExperimentalMatrixArchitecture::Dual2x1024KsplitDepth2
                    | ExperimentalMatrixArchitecture::Dual2x1024OnlineStreamKDepth2 => {
                        (0, 0, 0, 0, 0, 2, 2, 1024)
                    }
                    ExperimentalMatrixArchitecture::Asym211StreamKDepth3
                    | ExperimentalMatrixArchitecture::Asym211OnlineStreamKDepth3
                    | ExperimentalMatrixArchitecture::Asym211KsplitDepth3 => {
                        (2, 1024, 1, 1024, 0, 3, 0, 1024)
                    }
                    ExperimentalMatrixArchitecture::Quad1x1024MsplitDepth4
                    | ExperimentalMatrixArchitecture::Quad1x1024NsplitDepth4
                    | ExperimentalMatrixArchitecture::Quad1x1024KsplitDepth4
                    | ExperimentalMatrixArchitecture::Quad1x1024StageNToKDepth4
                    | ExperimentalMatrixArchitecture::Quad1x1024StreamKDepth4
                    | ExperimentalMatrixArchitecture::Quad1x1024OnlineStreamKDepth4 => {
                        (0, 0, 0, 0, 0, 4, 1, 1024)
                    }
                    _ => unreachable!(),
                },
            )
        } else {
            let candidate = match architecture {
                ExperimentalMatrixArchitecture::Baseline => reconstructed_baseline_matrix_cycles,
                ExperimentalMatrixArchitecture::FixedBigTail => {
                    fixed_big_tail_matrix_cycles(&manifest, fixed)
                }
                ExperimentalMatrixArchitecture::GangableRowSlices => {
                    gangable_matrix_cycles(&manifest)
                }
                _ => unreachable!(),
            };
            let adjusted_picos = serial_total_picos
                .saturating_sub(measured_matrix_picos)
                .saturating_add(candidate * period);
            let adjusted = adjusted_picos.div_ceil(period);
            *REPORT_CYCLES.lock().unwrap() = Some(adjusted);
            let (name, geometry) = match architecture {
                ExperimentalMatrixArchitecture::Baseline => {
                    ("baseline_4x1024".to_string(), (4, 1024, 0, 0, 0, 0, 0, 0))
                }
                ExperimentalMatrixArchitecture::FixedBigTail => (
                    "fixed_big_tail".to_string(),
                    (
                        4,
                        fixed.big_cols,
                        fixed.tail_rows,
                        fixed.tail_cols,
                        fixed.tail_threshold,
                        0,
                        0,
                        0,
                    ),
                ),
                ExperimentalMatrixArchitecture::GangableRowSlices => (
                    "gangable_4x_1x1024".to_string(),
                    (0, 0, 0, 0, 0, 4, 1, 1024),
                ),
                _ => unreachable!(),
            };
            (
                name,
                Some(candidate),
                Some(adjusted),
                Some(serial_total_cycles as f64 / adjusted as f64),
                geometry,
            )
        };
    let (big_rows, big_cols, tail_rows, tail_cols, threshold, slices, slice_rows, slice_cols) =
        geometry;

    let report = MatrixArchitectureReport {
        schema_version: 2,
        evidence_tier: if architecture.is_request_replay() {
            "request-level Ramulator replay with shape-scaled transferred compute calibration; not RTL sign-off"
                .to_string()
        } else {
            "emulator-calibrated arithmetic matrix timing overlay; not RTL sign-off".to_string()
        },
        measurement_mode: if architecture.is_request_replay() {
            "isolated_request_replay"
        } else {
            "post_run_cycle_replacement"
        },
        architecture: name,
        model: manifest.model,
        model_architecture: manifest.architecture,
        workload: manifest.workload,
        layer: manifest.layer,
        step: manifest.step,
        batch: manifest.batch,
        top_k: manifest.top_k,
        active_routed_experts: manifest.group_sizes.len(),
        routed_pairs: manifest.group_sizes.values().sum(),
        serial_total_cycles,
        measured_baseline_matrix_cycles,
        reconstructed_baseline_matrix_cycles,
        candidate_matrix_cycles,
        adjusted_total_cycles,
        speedup_vs_measured_baseline: speedup,
        request_replay,
        equal_area_pes: BASELINE_PES,
        big_rows,
        big_cols,
        tail_rows,
        tail_cols,
        tail_threshold: threshold,
        slice_count: slices,
        slice_rows,
        slice_cols,
        routed_matrix_wave_cycles: ROUTED_MATRIX_WAVE_CYCLES,
        shared_matrix_wave_cycles: SHARED_MATRIX_WAVE_CYCLES,
        assumptions: if matches!(
            architecture,
            ExperimentalMatrixArchitecture::Dual2x1024OnlineStreamKDepth2
                | ExperimentalMatrixArchitecture::Asym211OnlineStreamKDepth3
                | ExperimentalMatrixArchitecture::Quad1x1024OnlineStreamKDepth4
        ) {
            vec![
                "functional opcode execution remains on the unchanged single 4x1024 MatrixMachine",
                "baseline and candidate replay use separate fresh Ramulators starting at cycle zero",
                "physical HBM bytes are invariant; only request order and compute overlap may change",
                "online work uses one physical 64x64 MX request block per dispatchable macro-task",
                "dispatch, crossbar, completion scoreboard, and tagged reduction are cycle-modelled but require RTL calibration and PPA synthesis",
            ]
        } else if architecture.is_request_replay() {
            vec![
                "functional opcode execution remains on the unchanged single 4x1024 MatrixMachine",
                "baseline and candidate replay use separate fresh Ramulators starting at cycle zero",
                "physical HBM bytes are invariant; only request order and compute overlap may change",
                "compute calibration scales from the 2048x1408 reference by actual tiled M_MM work",
                "candidate matrix/reducer timers require RTL calibration before absolute-cycle claims",
            ]
        } else {
            vec![
                "only measured matrix proxy cycles are replaced; all non-matrix cycles are unchanged",
                "all candidates contain exactly 4096 matrix PEs and share baseline HBM traffic",
                "matrix jobs are ready for scheduling; extra dispatch/SRAM-port/frequency costs are not modelled",
                "absolute accuracy remains pending RTL calibration for each candidate geometry",
            ]
        },
    };
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).unwrap_or_else(|err| {
            panic!("failed to create matrix DSE output directory {parent:?}: {err}")
        });
    }
    std::fs::write(
        output_path,
        serde_json::to_string_pretty(&report).unwrap() + "\n",
    )
    .unwrap_or_else(|err| panic!("failed to write matrix DSE report {output_path:?}: {err}"));
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shaped_job(kind: JobKind, hidden: u64, intermediate: u64) -> LogicalJob {
        LogicalJob {
            id: 0,
            kind,
            expert_id: None,
            tokens: 1,
            projections: [
                Projection {
                    name: "gate",
                    k: hidden,
                    n: intermediate,
                    base: 0,
                },
                Projection {
                    name: "up",
                    k: hidden,
                    n: intermediate,
                    base: 0,
                },
                Projection {
                    name: "down",
                    k: intermediate,
                    n: hidden,
                    base: 0,
                },
            ],
        }
    }

    fn manifest(groups: &[u64], architecture: &str, batch: u64, top_k: u64) -> GroupedManifest {
        let group_sizes = groups
            .iter()
            .enumerate()
            .map(|(id, size)| (id.to_string(), *size))
            .collect::<BTreeMap<_, _>>();
        GroupedManifest {
            model: "fixture".into(),
            architecture: architecture.into(),
            workload: "fixture".into(),
            layer: 0,
            step: 0,
            batch,
            top_k,
            num_experts: groups.len() as u64,
            hidden: 256,
            routed_intermediate: 256,
            shared_intermediate: if architecture == "deepseek" {
                512
            } else {
                1024
            },
            selected_experts: (0..groups.len() as u64).collect(),
            group_sizes,
        }
    }

    #[test]
    fn baseline_preserves_four_row_step_function() {
        for tokens in 1..=4 {
            let m = manifest(&[tokens], "deepseek", 2, 1);
            let routed = baseline_matrix_cycles(&m) - 2 * SHARED_MATRIX_WAVE_CYCLES;
            assert_eq!(routed, ROUTED_MATRIX_WAVE_CYCLES);
        }
        let m = manifest(&[5], "deepseek", 2, 1);
        let routed = baseline_matrix_cycles(&m) - 2 * SHARED_MATRIX_WAVE_CYCLES;
        assert_eq!(routed, 2 * ROUTED_MATRIX_WAVE_CYCLES);
    }

    #[test]
    fn fixed_candidates_are_rejected_unless_equal_area() {
        let valid = FixedBigTailConfig {
            big_cols: 768,
            tail_rows: 1,
            tail_cols: 1024,
            tail_threshold: 1,
        };
        let m = manifest(&[1, 1], "deepseek", 2, 1);
        assert!(fixed_big_tail_matrix_cycles(&m, valid) > 0);
    }

    #[test]
    fn gangable_slices_pack_four_singletons_in_one_wave() {
        let m = manifest(&[1, 1, 1, 1], "deepseek", 2, 2);
        assert_eq!(gangable_matrix_cycles(&m), 2 * ROUTED_MATRIX_WAVE_CYCLES);
    }

    #[test]
    fn reconstructed_mx_requests_match_calibrated_physical_bytes() {
        let m = manifest(&[1], "deepseek", 1, 1);
        let jobs = jobs_from_manifest(&m);
        let routed = jobs.iter().find(|job| job.kind == JobKind::Routed).unwrap();
        let bytes = routed
            .projections
            .iter()
            .map(projection_physical_bytes)
            .sum::<u64>();
        // Each 256x256 projection contains sixteen 64x64 request blocks. Every
        // block issues 64 element and 64 rounded-scale bursts.
        assert_eq!(bytes, 3 * 16 * 2 * 64 * 64);
    }

    #[test]
    fn transferred_wave_calibration_scales_with_actual_model_shape() {
        let reference_routed = shaped_job(
            JobKind::Routed,
            CALIBRATION_HIDDEN,
            CALIBRATION_INTERMEDIATE,
        );
        assert_eq!(
            calibrated_wave_cycles(&reference_routed),
            ROUTED_MATRIX_WAVE_CYCLES
        );

        let reference_shared = shaped_job(
            JobKind::Shared,
            CALIBRATION_HIDDEN,
            2 * CALIBRATION_INTERMEDIATE,
        );
        assert_eq!(
            calibrated_wave_cycles(&reference_shared),
            2 * SHARED_MATRIX_WAVE_CYCLES
        );

        for job in [
            shaped_job(JobKind::Routed, 2048, 512),
            shaped_job(JobKind::Shared, 2048, 512),
            shaped_job(JobKind::Routed, 2688, 1856),
            shaped_job(JobKind::Shared, 2688, 3712),
        ] {
            let feed = mm_ops_per_wave(&job) * COMPILER_MLEN;
            assert!(calibrated_wave_cycles(&job) >= feed);
            assert!(fixed_cycles_per_output_tile(&job) >= 0.0);
        }
    }

    #[test]
    fn shortlist_compute_is_equal_area_and_faster_for_singletons() {
        let m = manifest(&[1, 1, 1, 1], "deepseek", 4, 1);
        let routed = jobs_from_manifest(&m)
            .into_iter()
            .find(|job| job.kind == JobKind::Routed)
            .unwrap();
        let baseline = baseline_compute(&routed);
        let dual = n_split_compute(&routed, &[2, 2]);
        let stream = stream_k_compute(&routed, &[2, 1, 1]);
        let dual_k = static_k_split_compute(&routed, &[2, 2]);
        let asym_k = static_k_split_compute(&routed, &[2, 1, 1]);
        let quad_m = m_split_compute(&routed, &[1, 1, 1, 1]);
        let quad_n = n_split_compute(&routed, &[1, 1, 1, 1]);
        let quad_k = static_k_split_compute(&routed, &[1, 1, 1, 1]);
        let quad_stage = stage_n_to_k_compute(&routed, &[1, 1, 1, 1]);
        let quad_stream = stream_k_compute(&routed, &[1, 1, 1, 1]);
        assert!(dual.compute_cycles < baseline.compute_cycles);
        assert!(stream.compute_cycles < baseline.compute_cycles);
        assert!(dual_k.compute_cycles < baseline.compute_cycles);
        assert!(asym_k.compute_cycles < baseline.compute_cycles);
        for candidate in [
            baseline,
            dual,
            stream,
            dual_k,
            asym_k,
            quad_m,
            quad_n,
            quad_k,
            quad_stage,
            quad_stream,
        ] {
            assert_eq!(
                candidate.aggregate_core_busy_cycles,
                candidate.aggregate_feed_cycles
                    + candidate.aggregate_fixed_cycles
                    + candidate.aggregate_launch_cycles
            );
        }
        assert_eq!(BASELINE_PES, 2 * 2 * 1024);
        assert_eq!(BASELINE_PES, (2 + 1 + 1) * 1024);
    }

    #[test]
    fn request_and_tensor_slices_are_exact_for_uneven_model_shapes() {
        let mut job = shaped_job(JobKind::Routed, 2688, 1856);
        job.tokens = 3;
        let n_slices = split_aligned(job.projections[0].n, 2, COMPILER_MLEN);
        assert_eq!(n_slices, vec![(0, 960), (960, 1856)]);

        for architecture in [
            ReplayArchitecture::DualNsplitDepth2,
            ReplayArchitecture::DualKsplitDepth2,
            ReplayArchitecture::DualOnlineStreamKDepth2,
            ReplayArchitecture::Asym211StreamKDepth3,
            ReplayArchitecture::Asym211OnlineStreamKDepth3,
            ReplayArchitecture::Asym211KsplitDepth3,
            ReplayArchitecture::QuadMsplitDepth4,
            ReplayArchitecture::QuadNsplitDepth4,
            ReplayArchitecture::QuadKsplitDepth4,
            ReplayArchitecture::QuadStageNToKDepth4,
            ReplayArchitecture::QuadStreamKDepth4,
            ReplayArchitecture::QuadOnlineStreamKDepth4,
        ] {
            validate_replay_contract(std::slice::from_ref(&job), architecture).unwrap();
            for tile in build_weight_tiles(std::slice::from_ref(&job), architecture) {
                let blocks = (tile.col_stop - tile.col_start) / COMPILER_MLEN;
                validate_block_owners(
                    &request_block_owners(architecture, &tile),
                    blocks,
                    architecture.partition_count(),
                )
                .unwrap();
            }
        }
    }

    #[test]
    fn static_k_split_preserves_feed_and_reduces_every_projection() {
        let job = shaped_job(JobKind::Routed, 2048, 1408);
        let baseline = baseline_compute(&job);
        for rows in [&[2_u64, 2][..], &[2_u64, 1, 1][..]] {
            let split = static_k_split_compute(&job, rows);
            assert_eq!(split.aggregate_feed_cycles, baseline.aggregate_feed_cycles);
            assert!(split.aggregate_fixed_cycles > baseline.aggregate_fixed_cycles);
            assert_eq!(split.reduction_elements, job.tokens * (2 * 1408 + 2048));
            assert_eq!(split.reduction_output_elements, split.reduction_elements);
            assert_eq!(
                split.reduction_add_elements,
                split.reduction_output_elements * (rows.len() as u64 - 1)
            );
            assert_eq!(split.interconnect_bytes, 2 * split.reduction_add_elements);
            assert!(split.reducer_cycles > 0);
            assert!(split.interconnect_cycles > 0);
        }
    }

    #[test]
    fn stream_k_counts_split_output_and_partial_additions_once() {
        for tokens in 1..=8 {
            let mut job = shaped_job(JobKind::Routed, 2048, 1408);
            job.tokens = tokens;
            let split = stream_k_compute(&job, &[2, 1, 1]);
            assert_eq!(split.reduction_output_elements, split.reduction_elements);
            assert!(split.reduction_add_elements >= split.reduction_output_elements);
            assert!(split.reduction_add_elements <= 2 * split.reduction_output_elements);
            assert_eq!(split.interconnect_bytes, 2 * split.reduction_add_elements);
        }
    }

    #[test]
    fn partition_skew_is_not_confused_with_job_completion_span() {
        let mut job = shaped_job(JobKind::Routed, 2688, 1856);
        job.tokens = 3;
        let baseline = baseline_compute(&job);
        let n_split = n_split_compute(&job, &[2, 2]);
        let k_split = static_k_split_compute(&job, &[2, 1, 1]);
        assert_eq!(baseline.aggregate_partition_wait_cycles, 0);
        assert_eq!(baseline.max_partition_skew_cycles, 0);
        assert!(n_split.aggregate_partition_wait_cycles > 0);
        assert!(n_split.max_partition_skew_cycles > 0);
        assert!(k_split.aggregate_partition_wait_cycles > 0);
        assert!(k_split.max_partition_skew_cycles > 0);
    }

    #[test]
    fn request_block_gate_catches_drop_duplicate_and_bad_partition() {
        assert!(validate_block_owners(&[(0, 0), (1, 1)], 2, 2).is_ok());
        assert!(validate_block_owners(&[(0, 0)], 2, 2).is_err());
        assert!(validate_block_owners(&[(0, 0), (0, 1)], 2, 2).is_err());
        assert!(validate_block_owners(&[(0, 0), (1, 2)], 2, 2).is_err());
    }

    #[test]
    fn online_work_items_follow_physical_64x64_request_blocks() {
        let m = manifest(&[1], "deepseek", 1, 1);
        let jobs = jobs_from_manifest(&m);
        for architecture in [
            ReplayArchitecture::DualOnlineStreamKDepth2,
            ReplayArchitecture::Asym211OnlineStreamKDepth3,
            ReplayArchitecture::QuadOnlineStreamKDepth4,
        ] {
            let tile = build_weight_tiles(&jobs, architecture)
                .into_iter()
                .find(|tile| tile.job_id == 0)
                .unwrap();
            let mut work_id = 0;
            let mut ready_sequence = 0;
            let items = work_items_for_tile(
                &tile,
                &jobs[tile.job_id],
                architecture,
                &mut work_id,
                &mut ready_sequence,
            );
            assert_eq!(
                items.len() as u64,
                (tile.col_stop - tile.col_start) / COMPILER_MLEN
            );
            assert!(
                items
                    .iter()
                    .all(|item| item.n_tile_count == COMPILER_MLEN / COMPILER_BLEN)
            );
            let keys = items
                .iter()
                .flat_map(OnlineWorkItem::output_keys)
                .collect::<BTreeSet<_>>();
            assert_eq!(
                keys.len() as u64,
                (tile.col_stop - tile.col_start) / COMPILER_BLEN
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tiny_request_replay_is_three_repeat_deterministic_and_byte_exact() {
        let m = manifest(&[1, 1], "deepseek", 2, 1);
        let jobs = jobs_from_manifest(&m);
        for architecture in [
            ReplayArchitecture::DualNsplitDepth2,
            ReplayArchitecture::DualKsplitDepth2,
            ReplayArchitecture::DualOnlineStreamKDepth2,
            ReplayArchitecture::Asym211StreamKDepth3,
            ReplayArchitecture::Asym211OnlineStreamKDepth3,
            ReplayArchitecture::Asym211KsplitDepth3,
            ReplayArchitecture::QuadMsplitDepth4,
            ReplayArchitecture::QuadNsplitDepth4,
            ReplayArchitecture::QuadKsplitDepth4,
            ReplayArchitecture::QuadStageNToKDepth4,
            ReplayArchitecture::QuadStreamKDepth4,
            ReplayArchitecture::QuadOnlineStreamKDepth4,
        ] {
            let mut repeats = Vec::new();
            for _ in 0..3 {
                repeats.push(run_isolated_replay(jobs.clone(), architecture).await);
            }
            let baseline = run_isolated_replay(
                jobs.clone(),
                ReplayArchitecture::Baseline {
                    buffer_depth: architecture.buffer_depth(),
                },
            )
            .await;
            assert!(
                repeats
                    .iter()
                    .all(|run| run.total_cycles == repeats[0].total_cycles)
            );
            assert!(
                repeats
                    .iter()
                    .all(|run| run.hbm_physical_bytes == baseline.hbm_physical_bytes)
            );
            assert!(repeats.iter().all(|run| run.completed_jobs == 3));
            assert!(repeats.iter().all(|run| run.cycle_accounting_passed));
            assert!(
                repeats
                    .iter()
                    .all(|run| run.total_cycles == run.accounted_controller_cycles)
            );
            assert!(
                repeats
                    .iter()
                    .all(|run| run.total_cycles == run.executor_drained_cycles)
            );
            if architecture.is_online_scheduler() {
                let expected_outputs = jobs
                    .iter()
                    .map(|job| {
                        div_ceil(job.tokens, BASELINE_ROWS)
                            * job
                                .projections
                                .iter()
                                .map(|projection| projection.n / COMPILER_BLEN)
                                .sum::<u64>()
                    })
                    .sum::<u64>();
                assert!(repeats.iter().all(|run| {
                    run.scheduler_kind == "completion_driven_earliest_finish_with_locality"
                        && run.online_work_items > 0
                        && run.online_completion_events == run.online_work_items
                        && run.scoreboard_output_entries > 0
                        && run.scoreboard_output_entries == expected_outputs
                        && (run.online_work_steal_events == 0 || run.crossbar_busy_cycles > 0)
                        && run.core_busy_cycles_by_core.len() == architecture.partition_count()
                        && run.core_completed_work_items.iter().sum::<u64>()
                            == run.online_work_items
                }));
            }
        }
    }
}
