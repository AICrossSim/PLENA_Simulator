//! Versioned input and output contract for the normal-buffer MoE experiment.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MatrixRegion {
    pub rows: usize,
    pub cols: usize,
    pub element_base: u64,
    pub scale_base: u64,
    pub element_row_stride: u64,
    pub scale_row_stride: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Expert {
    pub id: usize,
    pub gate: MatrixRegion,
    pub up: MatrixRegion,
    pub down: MatrixRegion,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Route {
    pub token: usize,
    pub slot: usize,
    pub expert: usize,
    pub weight: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SharedExpert {
    pub expert: usize,
    pub weight: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WeightBankReference {
    pub manifest: String,
    /// SHA256 of the exact immutable bank manifest bytes.
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub schema_version: u32,
    pub name: String,
    pub hbm_file: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_bank: Option<WeightBankReference>,
    pub input_dim: usize,
    pub expert_hidden_dim: usize,
    pub inputs_bf16: Vec<Vec<u16>>,
    pub routes: Vec<Route>,
    pub experts: Vec<Expert>,
    #[serde(default)]
    pub shared_expert: Option<SharedExpert>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub grouped_routes: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CoreConfig {
    pub id: String,
    /// Physical output lanes P. The legacy mapping also uses BLEN for its
    /// temporal M extent; refinement supplies m_rows independently.
    pub blen: usize,
    /// Physical K tile dimension; a multiple of local MX block size 8.
    pub mlen: usize,
    /// BF16 X, gate, up, Z and output, all reserved for one expert group.
    pub vector_sram_bytes: usize,
    /// FP32 output accumulator, reused across the three projections.
    pub accumulator_bytes: usize,
    /// Configurable slots, each holding packed MX ingress and decoded BF16 tile.
    pub weight_sram_bytes: usize,
    /// FIFO read cache: each entry reserves 64 data + 16 tag/control bytes.
    #[serde(default)]
    pub read_cache_bytes: usize,
    #[serde(default = "default_weight_slots")]
    pub weight_slots: usize,
    /// Explicit BF16 activation supply per cycle. None retains the legacy
    /// full-width analytical assumption; this is not a physical area estimate.
    #[serde(default)]
    pub activation_elements_per_cycle: Option<usize>,
    /// V2 makes token scheduling independent of BLEN output lanes / MLEN K lanes.
    #[serde(default)]
    pub refinement: Option<CoreRefinement>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TailPolicy {
    Padded,
    ValidRows,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CoreRefinement {
    pub m_rows: usize,
    pub tail_policy: TailPolicy,
    /// Legacy path: one to three resident N tiles. Pool mode requires 2 here;
    /// only output_pool controls its independent context/stage capacities.
    pub active_n_tiles: usize,
    /// Shared decoded-SRAM read/write port; BF16 elements per cycle.
    pub weight_read_elements_per_cycle: usize,
    /// One shared FP32 accumulator read/write port, elements per cycle.
    pub accumulator_elements_per_cycle: usize,
    /// Stationary BF16 operands, charged inside weight_sram_bytes, not extra SRAM.
    pub operand_latch_bytes: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_pool: Option<OutputPoolConfig>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OutputPoolConfig {
    /// Total (temporal M block, N band) records. An N band reserves all its
    /// ceil(Me / m_rows) records together, preserving one weight load per K.
    pub output_contexts: usize,
    /// Independently owned BF16 operand latches (1 or 2).
    pub operand_stages: usize,
    /// Nonzero cost of each bounded scheduler descriptor visit/admission.
    pub scheduler_cycles: u64,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DispatchPolicy {
    #[default]
    Threshold,
    WorkConserving,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MatrixTiming {
    #[default]
    Pipelined,
    /// Legacy MatrixMachine-like per-instruction K + overhead serialization.
    LegacySerialized,
}

fn default_weight_slots() -> usize {
    2
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DmaConfig {
    pub issue_policy: ramulator::model::IssuePolicy,
    pub sector_reads: bool,
    pub coalesce: bool,
    #[serde(default)]
    pub fair_credits: bool,
    /// Lookup latency is two cycles; initiation interval is independently 1/2.
    #[serde(default = "default_weight_slots")]
    pub lookup_ii_cycles: usize,
    /// Includes response data, MSHRs, waiters, native trackers, and tile descriptors.
    pub frontend_sram_bytes: usize,
}

fn default_queue_bytes() -> usize {
    262144
}
fn default_dispatch_cycles() -> u64 {
    1
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Architecture {
    pub schema_version: u32,
    pub name: String,
    pub cores: Vec<CoreConfig>,
    /// Groups with M >= threshold select large_core, otherwise small_core.
    pub dispatch_threshold: usize,
    pub large_core: usize,
    pub small_core: usize,
    pub global_dma_credits: usize,
    /// Each in-flight 64-byte HBM request retains its credit until copied.
    pub global_dma_staging_bytes: usize,
    /// Ready input BF16 + all BF16 route outputs + FP32 combine + BF16 final.
    pub combine_sram_bytes: usize,
    pub clock_period_ps: u64,
    /// Analytical fill/drain overhead, in addition to BLEN + ceil(log2 MLEN).
    pub mac_pipeline_cycles: u64,
    /// One shared vector actor serves gathers, SwiGLU, result copies/combine.
    pub vector_elements_per_cycle: usize,
    #[serde(default)]
    pub dispatch_policy: DispatchPolicy,
    /// Fixed 64-byte descriptor per ready expert job.
    #[serde(default = "default_queue_bytes")]
    pub dispatch_queue_bytes: usize,
    #[serde(default = "default_dispatch_cycles")]
    pub dispatch_cycles: u64,
    #[serde(default)]
    pub matrix_timing: MatrixTiming,
    #[serde(default)]
    pub dma: Option<DmaConfig>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CoreReport {
    pub id: String,
    pub blen: usize,
    pub mlen: usize,
    pub multipliers: u64,
    pub jobs: usize,
    pub useful_macs: u64,
    pub issued_macs: u64,
    pub compute_busy_ps: u64,
    pub accumulator_dependency_stall_ps: u64,
    pub pipeline_drain_ps: u64,
    pub pipeline_register_bytes: usize,
    pub weight_ready_wait_ps: u64,
    pub vector_wait_ps: u64,
    pub vector_sram_peak_bytes: usize,
    pub accumulator_peak_bytes: usize,
    pub weight_sram_peak_bytes: usize,
    pub weight_slots_peak: usize,
    pub hbm_read_bytes: u64,
    pub compute_busy_fraction: f64,
    pub mac_utilization: f64,
    pub cache_requests: u64,
    pub cache_hits: u64,
    pub cache_port_busy_ps: u64,
    pub cache_peak_bytes: usize,
    pub refinement: Option<RefinementReport>,
    pub tile_loads: TileLoadStats,
    /// Observational records; not architectural queues or on-chip storage.
    pub projections: Vec<ProjectionReport>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RefinementReport {
    pub m_rows: usize,
    pub operand_latch_reserved_bytes: usize,
    pub output_context_peak_bytes: usize,
    pub pending_result_peak_bytes: usize,
    pub weight_port_busy_ps: u64,
    pub weight_port_wait_ps: u64,
    pub accumulator_port_busy_ps: u64,
    pub accumulator_port_wait_ps: u64,
    pub output_context_stall_ps: u64,
    pub independent_output_switches: u64,
    pub output_contexts_peak: usize,
    pub finalized_elements: u64,
    pub output_finalize_elapsed_ps: u64,
    pub output_pool: Option<OutputPoolReport>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct OutputPoolReport {
    pub contexts_capacity: usize,
    pub operand_stages_capacity: usize,
    pub control_reserved_bytes: usize,
    pub pending_result_reserved_bytes: usize,
    pub scheduler_busy_ps: u64,
    pub scheduler_visits: u64,
    pub band_admissions: u64,
    pub tile_admissions: u64,
    pub context_updates: u64,
    pub contexts_peak: usize,
    pub operand_stages_peak: usize,
    pub pending_contexts_peak: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TileLoadStats {
    pub count: u64,
    pub total_ps: u64,
    pub min_ps: u64,
    pub max_ps: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ProjectionMetrics {
    pub useful_macs: u64,
    pub issued_macs: u64,
    pub compute_busy_ps: u64,
    pub accumulator_dependency_stall_ps: u64,
    pub pipeline_drain_ps: u64,
    pub weight_ready_wait_ps: u64,
    pub vector_wait_ps: u64,
    pub hbm_read_bytes: u64,
    pub weight_port_busy_ps: u64,
    pub weight_port_wait_ps: u64,
    pub accumulator_port_busy_ps: u64,
    pub accumulator_port_wait_ps: u64,
    pub output_finalize_elapsed_ps: u64,
    pub scheduler_busy_ps: u64,
    pub scheduler_visits: u64,
    pub band_admissions: u64,
    pub tile_admissions: u64,
    pub context_updates: u64,
    pub weight_slots_peak: usize,
    pub contexts_peak: usize,
    pub operand_stages_peak: usize,
    pub pending_contexts_peak: usize,
    pub tile_loads: TileLoadStats,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProjectionReport {
    pub job: usize,
    pub expert: usize,
    pub projection: String,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub start_ps: u64,
    pub end_ps: u64,
    pub metrics: ProjectionMetrics,
}

#[derive(Clone, Debug, Serialize)]
pub struct JobCompletion {
    pub job: usize,
    pub expert: usize,
    pub shared: bool,
    pub core: String,
    pub rows: usize,
    pub start_ps: u64,
    pub compute_done_ps: u64,
    pub output_copied_ps: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunReport {
    pub schema_version: u32,
    pub workload: String,
    pub architecture: String,
    pub timing_model: String,
    pub timing_boundary: String,
    pub weight_format: String,
    pub total_ps: u64,
    pub multipliers: u64,
    pub useful_macs: u64,
    pub issued_macs: u64,
    pub hbm_read_bytes: u64,
    pub hbm_write_bytes: u64,
    pub global_dma_inflight_peak: usize,
    pub global_dma_staging_peak_bytes: usize,
    pub combine_sram_peak_bytes: usize,
    pub shared_vector_busy_ps: u64,
    pub dispatch_queue_peak_bytes: usize,
    pub dispatcher_busy_ps: u64,
    pub dma_frontend: Option<DmaReport>,
    pub cores: Vec<CoreReport>,
    /// Completion order, which may differ from deterministic reduction order.
    pub job_completions: Vec<JobCompletion>,
    pub output_bf16: Vec<Vec<u16>>,
    pub output_f32: Vec<Vec<f32>>,
    pub pre_round_output_f32: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DmaReport {
    pub reserved_bytes: usize,
    pub line_requests: u64,
    pub sector_requests: u64,
    pub merged_sectors: u64,
    pub useful_copy_bytes: u64,
    pub lookup_busy_ps: u64,
    pub copy_busy_ps: u64,
    pub mshr_peak: usize,
    pub fair_credit_reserve_per_core: usize,
    pub fair_credit_wait_ps: u64,
}
