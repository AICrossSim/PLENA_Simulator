use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use runtime::Duration;
use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};

use crate::runtime_config::PERIOD;

/// Serialize an ordered slice of `(name, value)` entries as a JSON object,
/// preserving insertion order rather than sorting keys. Used for the per-stage
/// maps so they stay in `StageKind::ALL` logical order (a plain
/// `BTreeMap<&str, _>` would reorder them alphabetically).
fn serialize_ordered<S, T>(entries: &[(&'static str, T)], serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    T: Serialize,
{
    let mut map = serializer.serialize_map(Some(entries.len()))?;
    for (name, value) in entries {
        map.serialize_entry(name, value)?;
    }
    map.end()
}

const PROFILE_CAVEAT: &str = "Stage and routed-pair labels are derived from generated ASM comments. Cycles use simulator time only. physical_hbm_bytes_* are measured from the global WithStats 64B HBM deltas before/after each opcode. logical_bytes_* are intentionally null here and must be joined from workload shape/route formulas. resource_proxy_cycles are first-pass opcode-class wall-cycle attribution, not calibrated component busy counters. ramulator_proxy is a sub-view of dma, not an additive peer; totals should use matrix+vector+scalar+dma+other. Current do_ops still awaits each opcode, so this profile does not by itself prove cross-op overlap. Pair labels identify static routed pair slots, not necessarily unique expert IDs without joining the routing dump.";

const LOGICAL_BYTE_STATUS: &str =
    "not_declared_by_opcode_profile; join benchmark route/shape formulas for logical bytes";
const PHYSICAL_BYTE_STATUS: &str = "HBM bytes are emulator WithStats 64B physical transfer deltas";
const RESOURCE_CYCLE_STATUS: &str =
    "first-pass opcode-class wall-cycle proxy, not calibrated per-component busy counters";

/// Opcode-class this instruction is attributed to for the first-pass resource
/// proxy. This is a coarse wall-cycle attribution keyed on opcode family, NOT a
/// calibrated per-component busy counter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResourceKind {
    Matrix,
    Vector,
    Scalar,
    Dma,
    Other,
}

/// Per-resource wall-cycle accumulator. `ramulator_proxy_cycles` is a *sub-view*
/// of `dma_cycles` (a memory proxy for readers), never an additive peer: totals
/// must use matrix+vector+scalar+dma+other.
#[derive(Clone, Copy, Debug, Default, Serialize)]
struct ResourceRuntime {
    #[serde(rename = "matrix")]
    matrix_cycles: u64,
    #[serde(rename = "vector")]
    vector_cycles: u64,
    #[serde(rename = "scalar")]
    scalar_cycles: u64,
    #[serde(rename = "dma")]
    dma_cycles: u64,
    #[serde(rename = "ramulator_proxy")]
    ramulator_proxy_cycles: u64,
    #[serde(rename = "other")]
    other_cycles: u64,
}

impl ResourceRuntime {
    fn add(&mut self, resource: ResourceKind, cycles: u64) {
        match resource {
            ResourceKind::Matrix => self.matrix_cycles += cycles,
            ResourceKind::Vector => self.vector_cycles += cycles,
            ResourceKind::Scalar => self.scalar_cycles += cycles,
            ResourceKind::Dma => {
                self.dma_cycles += cycles;
                // Sub-view only: ramulator_proxy_cycles is included in dma_cycles.
                // Totals should use matrix+vector+scalar+dma+other, not both.
                self.ramulator_proxy_cycles += cycles;
            }
            ResourceKind::Other => self.other_cycles += cycles,
        }
    }

    fn add_runtime(&mut self, other: Self) {
        self.matrix_cycles += other.matrix_cycles;
        self.vector_cycles += other.vector_cycles;
        self.scalar_cycles += other.scalar_cycles;
        self.dma_cycles += other.dma_cycles;
        self.ramulator_proxy_cycles += other.ramulator_proxy_cycles;
        self.other_cycles += other.other_cycles;
    }
}

#[derive(Serialize)]
struct StageStatsJson {
    instructions: u64,
    wall_cycles: u64,
    seconds: f64,
    instruction_fraction: f64,
    time_fraction: f64,
    cycle_fraction: f64,
    logical_bytes_read: Option<u64>,
    logical_bytes_written: Option<u64>,
    physical_hbm_bytes_read: u64,
    physical_hbm_bytes_written: u64,
    hbm_bytes_read: u64,
    hbm_bytes_written: u64,
    resource_proxy_cycles: ResourceRuntime,
}

#[derive(Serialize)]
struct PairStageStatsJson {
    instructions: u64,
    wall_cycles: u64,
    seconds: f64,
    logical_bytes_read: Option<u64>,
    logical_bytes_written: Option<u64>,
    physical_hbm_bytes_read: u64,
    physical_hbm_bytes_written: u64,
    hbm_bytes_read: u64,
    hbm_bytes_written: u64,
    resource_proxy_cycles: ResourceRuntime,
}

#[derive(Serialize)]
struct PairStatsJson {
    instructions: u64,
    wall_cycles: u64,
    seconds: f64,
    logical_bytes_read: Option<u64>,
    logical_bytes_written: Option<u64>,
    physical_hbm_bytes_read: u64,
    physical_hbm_bytes_written: u64,
    hbm_bytes_read: u64,
    hbm_bytes_written: u64,
    resource_proxy_cycles: ResourceRuntime,
    #[serde(serialize_with = "serialize_ordered")]
    stages: Vec<(&'static str, PairStageStatsJson)>,
}

#[derive(Serialize)]
struct ProfileJson {
    schema_version: u32,
    label_count: usize,
    total_instructions_executed: u64,
    total_simulation_cycles: Option<u64>,
    total_profiled_cycles: u64,
    total_stage_wall_cycles: u64,
    total_unprofiled_cycles: u64,
    cycle_accounting_status: &'static str,
    total_profiled_seconds: f64,
    total_hbm_bytes_read: u64,
    total_hbm_bytes_written: u64,
    total_resource_proxy_cycles: ResourceRuntime,
    logical_byte_status: &'static str,
    physical_byte_status: &'static str,
    resource_cycle_status: &'static str,
    #[serde(serialize_with = "serialize_ordered")]
    stages: Vec<(&'static str, StageStatsJson)>,
    // Keyed by u32 so serde emits the pair objects in numeric order
    // ("2" before "10"); a String key would sort them lexicographically.
    pairs: BTreeMap<u32, PairStatsJson>,
    caveat: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StageKind {
    RouterTopk,
    AccumulatorInit,
    Gather,
    ExpertWeightAddress,
    ExpertWeightPrefetch,
    ExpertProjection,
    ExpertActivation,
    ExpertBias,
    ExpertRouteWeight,
    ScatterCombine,
    Other,
}

impl StageKind {
    const ALL: [StageKind; 11] = [
        StageKind::RouterTopk,
        StageKind::AccumulatorInit,
        StageKind::Gather,
        StageKind::ExpertWeightAddress,
        StageKind::ExpertWeightPrefetch,
        StageKind::ExpertProjection,
        StageKind::ExpertActivation,
        StageKind::ExpertBias,
        StageKind::ExpertRouteWeight,
        StageKind::ScatterCombine,
        StageKind::Other,
    ];

    fn name(self) -> &'static str {
        match self {
            StageKind::RouterTopk => "router_topk",
            StageKind::AccumulatorInit => "accumulator_init",
            StageKind::Gather => "gather",
            StageKind::ExpertWeightAddress => "expert_weight_address",
            StageKind::ExpertWeightPrefetch => "expert_weight_prefetch",
            StageKind::ExpertProjection => "expert_projection",
            StageKind::ExpertActivation => "expert_activation",
            StageKind::ExpertBias => "expert_bias",
            StageKind::ExpertRouteWeight => "expert_route_weight",
            StageKind::ScatterCombine => "scatter_combine",
            StageKind::Other => "other",
        }
    }

    fn index(self) -> usize {
        match self {
            StageKind::RouterTopk => 0,
            StageKind::AccumulatorInit => 1,
            StageKind::Gather => 2,
            StageKind::ExpertWeightAddress => 3,
            StageKind::ExpertWeightPrefetch => 4,
            StageKind::ExpertProjection => 5,
            StageKind::ExpertActivation => 6,
            StageKind::ExpertBias => 7,
            StageKind::ExpertRouteWeight => 8,
            StageKind::ScatterCombine => 9,
            StageKind::Other => 10,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct StageRuntime {
    instructions: u64,
    wall_cycles: u64,
    seconds: f64,
    hbm_bytes_read: u64,
    hbm_bytes_written: u64,
    resource_proxy: ResourceRuntime,
}

pub(crate) struct StageProfiler {
    labels: Vec<StageKind>,
    pair_labels: Vec<Option<u32>>,
    stages: [StageRuntime; 11],
    pair_stages: BTreeMap<u32, [StageRuntime; 11]>,
    total_instructions: u64,
    total_profiled_cycles: u64,
    total_simulation_cycles: Option<u64>,
    total_seconds: f64,
    total_hbm_bytes_read: u64,
    total_hbm_bytes_written: u64,
    total_resource_proxy: ResourceRuntime,
}

impl StageProfiler {
    pub(crate) fn from_asm(path: &Path, expected_ops: usize) -> std::io::Result<Self> {
        let asm = fs::read_to_string(path)?;
        let mut labels = Vec::with_capacity(expected_ops);
        let mut pair_labels = Vec::with_capacity(expected_ops);
        let mut stage = StageKind::Other;
        let mut pair_id = None;

        for raw_line in asm.lines() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with(';') {
                stage = classify_comment(line, stage);
                pair_id = extract_pair_id(line).or_else(|| {
                    if matches!(stage, StageKind::RouterTopk | StageKind::AccumulatorInit) {
                        None
                    } else {
                        pair_id
                    }
                });
            } else if is_opcode_line(line) {
                labels.push(stage);
                pair_labels.push(pair_id);
            }
        }

        if labels.len() != expected_ops {
            tracing::warn!(
                asm = %path.display(),
                labels = labels.len(),
                expected_ops,
                "stage profile ASM label count differs from decoded opcode count"
            );
        }

        Ok(Self {
            labels,
            pair_labels,
            stages: [StageRuntime::default(); 11],
            pair_stages: BTreeMap::new(),
            total_instructions: 0,
            total_profiled_cycles: 0,
            total_simulation_cycles: None,
            total_seconds: 0.0,
            total_hbm_bytes_read: 0,
            total_hbm_bytes_written: 0,
            total_resource_proxy: ResourceRuntime::default(),
        })
    }

    // First-pass proxy: per-op div_ceil can systematically overcount when op
    // durations are not exact cycle multiples. Calibrate this with RTL primitive
    // measurements before treating stage-profile cycle sums as final timing.
    pub(crate) fn duration_to_cycles(duration: Duration) -> u64 {
        let period_picos = PERIOD.as_picos().max(1);
        duration.as_picos().div_ceil(period_picos)
    }

    pub(crate) fn set_total_simulation_duration(&mut self, duration: Duration) {
        self.total_simulation_cycles = Some(Self::duration_to_cycles(duration));
    }

    pub(crate) fn record(
        &mut self,
        pc: usize,
        seconds: f64,
        wall_cycles: u64,
        resource: ResourceKind,
        hbm_bytes_read: u64,
        hbm_bytes_written: u64,
    ) {
        let stage = self.labels.get(pc).copied().unwrap_or(StageKind::Other);
        let bucket = &mut self.stages[stage.index()];
        bucket.instructions += 1;
        bucket.wall_cycles += wall_cycles;
        bucket.seconds += seconds;
        bucket.hbm_bytes_read += hbm_bytes_read;
        bucket.hbm_bytes_written += hbm_bytes_written;
        bucket.resource_proxy.add(resource, wall_cycles);
        self.total_instructions += 1;
        self.total_profiled_cycles += wall_cycles;
        self.total_seconds += seconds;
        self.total_hbm_bytes_read += hbm_bytes_read;
        self.total_hbm_bytes_written += hbm_bytes_written;
        self.total_resource_proxy.add(resource, wall_cycles);

        if let Some(pair_id) = self.pair_labels.get(pc).copied().flatten() {
            let pair_buckets = self
                .pair_stages
                .entry(pair_id)
                .or_insert([StageRuntime::default(); 11]);
            let pair_bucket = &mut pair_buckets[stage.index()];
            pair_bucket.instructions += 1;
            pair_bucket.wall_cycles += wall_cycles;
            pair_bucket.seconds += seconds;
            pair_bucket.hbm_bytes_read += hbm_bytes_read;
            pair_bucket.hbm_bytes_written += hbm_bytes_written;
            pair_bucket.resource_proxy.add(resource, wall_cycles);
        }
    }

    fn to_json(&self) -> ProfileJson {
        let stages = StageKind::ALL
            .iter()
            .map(|stage| {
                let stats = self.stages[stage.index()];
                let instruction_fraction = if self.total_instructions == 0 {
                    0.0
                } else {
                    stats.instructions as f64 / self.total_instructions as f64
                };
                let time_fraction = if self.total_seconds == 0.0 {
                    0.0
                } else {
                    stats.seconds / self.total_seconds
                };
                let cycle_fraction = if self.total_profiled_cycles == 0 {
                    0.0
                } else {
                    stats.wall_cycles as f64 / self.total_profiled_cycles as f64
                };
                (
                    stage.name(),
                    StageStatsJson {
                        instructions: stats.instructions,
                        wall_cycles: stats.wall_cycles,
                        seconds: stats.seconds,
                        instruction_fraction,
                        time_fraction,
                        cycle_fraction,
                        logical_bytes_read: None,
                        logical_bytes_written: None,
                        physical_hbm_bytes_read: stats.hbm_bytes_read,
                        physical_hbm_bytes_written: stats.hbm_bytes_written,
                        hbm_bytes_read: stats.hbm_bytes_read,
                        hbm_bytes_written: stats.hbm_bytes_written,
                        resource_proxy_cycles: stats.resource_proxy,
                    },
                )
            })
            .collect();

        let pairs = self
            .pair_stages
            .iter()
            .map(|(pair_id, stages)| {
                let totals = sum_stage_runtimes(stages);
                let per_stage = StageKind::ALL
                    .iter()
                    .map(|stage| {
                        let stats = stages[stage.index()];
                        (
                            stage.name(),
                            PairStageStatsJson {
                                instructions: stats.instructions,
                                wall_cycles: stats.wall_cycles,
                                seconds: stats.seconds,
                                logical_bytes_read: None,
                                logical_bytes_written: None,
                                physical_hbm_bytes_read: stats.hbm_bytes_read,
                                physical_hbm_bytes_written: stats.hbm_bytes_written,
                                hbm_bytes_read: stats.hbm_bytes_read,
                                hbm_bytes_written: stats.hbm_bytes_written,
                                resource_proxy_cycles: stats.resource_proxy,
                            },
                        )
                    })
                    .collect();
                (
                    *pair_id,
                    PairStatsJson {
                        instructions: totals.instructions,
                        wall_cycles: totals.wall_cycles,
                        seconds: totals.seconds,
                        logical_bytes_read: None,
                        logical_bytes_written: None,
                        physical_hbm_bytes_read: totals.hbm_bytes_read,
                        physical_hbm_bytes_written: totals.hbm_bytes_written,
                        hbm_bytes_read: totals.hbm_bytes_read,
                        hbm_bytes_written: totals.hbm_bytes_written,
                        resource_proxy_cycles: totals.resource_proxy,
                        stages: per_stage,
                    },
                )
            })
            .collect();

        let total_stage_wall_cycles = sum_stage_runtimes(&self.stages).wall_cycles;
        let total_simulation_cycles = self.total_simulation_cycles;
        let total_unprofiled_cycles = total_simulation_cycles
            .map(|cycles| cycles.saturating_sub(self.total_profiled_cycles))
            .unwrap_or(0);
        let cycle_accounting_status = match total_simulation_cycles {
            Some(cycles) if cycles == self.total_profiled_cycles => "profiled_cycles_match_total",
            Some(_) => "profiled_cycles_do_not_match_total",
            None => "total_simulation_cycles_unset",
        };

        ProfileJson {
            schema_version: 2,
            label_count: self.labels.len(),
            total_instructions_executed: self.total_instructions,
            total_simulation_cycles,
            total_profiled_cycles: self.total_profiled_cycles,
            total_stage_wall_cycles,
            total_unprofiled_cycles,
            cycle_accounting_status,
            total_profiled_seconds: self.total_seconds,
            total_hbm_bytes_read: self.total_hbm_bytes_read,
            total_hbm_bytes_written: self.total_hbm_bytes_written,
            total_resource_proxy_cycles: self.total_resource_proxy,
            logical_byte_status: LOGICAL_BYTE_STATUS,
            physical_byte_status: PHYSICAL_BYTE_STATUS,
            resource_cycle_status: RESOURCE_CYCLE_STATUS,
            stages,
            pairs,
            caveat: PROFILE_CAVEAT,
        }
    }

    pub(crate) fn write_json(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.to_json()).map_err(std::io::Error::other)?;
        fs::write(path, json + "\n")
    }
}

fn sum_stage_runtimes(stages: &[StageRuntime; 11]) -> StageRuntime {
    let mut total = StageRuntime::default();
    for stats in stages {
        total.instructions += stats.instructions;
        total.wall_cycles += stats.wall_cycles;
        total.seconds += stats.seconds;
        total.hbm_bytes_read += stats.hbm_bytes_read;
        total.hbm_bytes_written += stats.hbm_bytes_written;
        total.resource_proxy.add_runtime(stats.resource_proxy);
    }
    total
}

fn is_opcode_line(line: &str) -> bool {
    line.as_bytes()
        .first()
        .copied()
        .map(|byte| byte.is_ascii_uppercase())
        .unwrap_or(false)
}

/// Map a generated-ASM comment line to the routed-MoE stage it introduces.
///
/// WARNING: this is a best-effort heuristic that matches on substrings of the
/// comment text emitted by PLENA_Compiler's routed-MoE emitter. It is NOT a
/// stable contract — renaming a comment on the compiler side will silently
/// reclassify (or drop) instructions here. This is acceptable only because the
/// stage profile is diagnostic-only, gated behind `--stage-profile-asm`, and
/// never affects functional emulation. Keep the matched phrases in sync with
/// `aten/plena/program_routed_moe.py` (and related emitters) in the compiler.
fn classify_comment(comment: &str, current: StageKind) -> StageKind {
    let text = comment.to_ascii_lowercase();
    if text.contains("gpt-oss router")
        || text.contains("router token")
        || text.contains("router dot token")
    {
        StageKind::RouterTopk
    } else if text.contains("gpt-oss vram scatter-add") || text.contains("_scatter") {
        StageKind::ScatterCombine
    } else if text.contains("allocate vram matrix step6_pair") && text.contains("_route") {
        StageKind::ExpertRouteWeight
    } else if text.contains("materialize route weight")
        || text.contains("vram matrix mul")
        || (text.contains("true-zero vram rows") && matches!(current, StageKind::ExpertRouteWeight))
    {
        StageKind::ExpertRouteWeight
    } else if text.contains("step6_device_routing_acc") || text.contains("true-zero vram rows") {
        StageKind::AccumulatorInit
    } else if text.contains("gpt-oss gather token rows")
        || text.contains("gather pair")
        || text.contains("clear gather padding")
        || (text.contains("allocate vram matrix step6_pair") && text.contains("_gather"))
    {
        StageKind::Gather
    } else if text.contains("dynamic expert bias add") {
        StageKind::ExpertBias
    } else if text.contains("allocate vram matrix step6_pair") && text.contains("_sigmoid") {
        StageKind::ExpertActivation
    } else if text.contains("tile row min fp")
        || text.contains("tile row max fp")
        || matches!(current, StageKind::ExpertActivation)
            && (text.contains("vram fill zero")
                || text.contains("vram matrix add")
                || text.contains("vram matrix mul"))
    {
        StageKind::ExpertActivation
    } else if text.contains("dynamic hbm weight prefetch")
        || text.contains("expert_id_to_weight_base")
    {
        StageKind::ExpertWeightAddress
    } else if text.contains("subblock [") {
        StageKind::ExpertWeightPrefetch
    } else if text.contains("sub projection")
        || text.contains("vram block add")
        || text.contains("vram block")
        || (text.contains("allocate vram matrix step6_pair") && !text.contains("_gather"))
    {
        StageKind::ExpertProjection
    } else {
        current
    }
}

fn extract_pair_id(comment: &str) -> Option<u32> {
    let bytes = comment.as_bytes();
    for prefix in [b"step6_pair".as_slice(), b"pair=".as_slice()] {
        let mut start = 0;
        while let Some(pos) = find_subslice(&bytes[start..], prefix) {
            let digit_start = start + pos + prefix.len();
            let digit_end = bytes[digit_start..]
                .iter()
                .position(|byte| !byte.is_ascii_digit())
                .map(|offset| digit_start + offset)
                .unwrap_or(bytes.len());
            if digit_end > digit_start {
                if let Ok(id) = comment[digit_start..digit_end].parse::<u32>() {
                    return Some(id);
                }
            }
            start = digit_start;
        }
    }
    None
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize)]
    struct OrderWrapper {
        #[serde(serialize_with = "serialize_ordered")]
        stages: Vec<(&'static str, u32)>,
        pairs: BTreeMap<u32, u32>,
    }

    #[test]
    fn json_preserves_logical_stage_and_numeric_pair_order() {
        // Insertion order is deliberately non-alphabetical; pair ids are chosen so
        // lexicographic ("10" < "2") differs from numeric ("2" < "10").
        let wrapper = OrderWrapper {
            stages: vec![("router_topk", 1), ("accumulator_init", 2), ("gather", 3)],
            pairs: BTreeMap::from([(10u32, 0u32), (2u32, 0u32)]),
        };
        let json = serde_json::to_string(&wrapper).unwrap();

        let pos = |needle: &str| json.find(needle).expect(needle);
        // stages keep StageKind::ALL insertion order, not alphabetical.
        assert!(pos("router_topk") < pos("accumulator_init"));
        assert!(pos("accumulator_init") < pos("gather"));
        // pair keys serialize in numeric order ("2" before "10").
        assert!(pos("\"2\"") < pos("\"10\""));
    }

    #[test]
    fn duration_to_cycles_rounds_up_to_period() {
        assert_eq!(
            StageProfiler::duration_to_cycles(Duration::from_picos(0)),
            0
        );
        assert_eq!(
            StageProfiler::duration_to_cycles(Duration::from_picos(999)),
            1
        );
        assert_eq!(
            StageProfiler::duration_to_cycles(Duration::from_picos(1000)),
            1
        );
        assert_eq!(
            StageProfiler::duration_to_cycles(Duration::from_picos(1001)),
            2
        );
    }
}
