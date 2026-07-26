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

const PROFILE_CAVEAT: &str = "Stage and routed-pair labels are derived from generated ASM comments. Time uses simulator time only. *_picos fields are exact and additive; *_cycles fields are that picosecond value rounded up to whole clock periods and are therefore NOT additive (each level rounds independently, so per-stage or per-bucket cycles can exceed their parent by up to one cycle each) -- do arithmetic on picos, display cycles. physical_hbm_bytes_* are measured from the global WithStats 64B HBM deltas before/after each opcode. logical_bytes_* are intentionally null here and must be joined from workload shape/route formulas. resource_proxy_* are first-pass opcode-class wall-time attribution, not calibrated component busy counters; its buckets are disjoint, so a total is matrix+vector+scalar+dma+other. Current do_ops still awaits each opcode, so this profile does not by itself prove cross-op overlap. Pair labels identify static routed pair slots, not necessarily unique expert IDs without joining the routing dump.";

const LOGICAL_BYTE_STATUS: &str =
    "not_declared_by_opcode_profile; join benchmark route/shape formulas for logical bytes";
const PHYSICAL_BYTE_STATUS: &str = "HBM bytes are emulator WithStats 64B physical transfer deltas";
const RESOURCE_CYCLE_STATUS: &str =
    "first-pass opcode-class wall-time proxy, not calibrated per-component busy counters";

/// Explains the picos/cycles relationship to anyone reading the JSON without the
/// full caveat string.
const TIME_UNIT_STATUS: &str = "picos are exact and additive; cycles are picos rounded up to whole PERIOD and are not additive across levels";

/// Warn when a routed-MoE program leaves more than this fraction of opcodes
/// unclassified (a strong signal the compiler comment vocabulary drifted out of
/// sync with `classify_comment`).
const UNCLASSIFIED_WARN_THRESHOLD: f64 = 0.5;

/// Substrings that mark a program as routed-MoE / expert-computation (so a high
/// unclassified fraction is a drift signal rather than expected). Broad on
/// purpose: matching more of the stage vocabulary makes the warning robust to a
/// *partial* comment rename. Kept lowercase; matched against the lowercased ASM.
const ROUTED_MOE_MARKERS: [&str; 4] = ["step6_pair", "gpt-oss", "sub projection", "expert_"];

/// Every comment substring `classify_comment` keys on, in the order the rules are
/// applied.
///
/// This is declared as data purely so the vocabulary is auditable, reportable and
/// testable. `classify_comment` still owns the rules themselves: it combines these
/// with conjunctions, one negation, and a stateful carry-over that a flat table
/// cannot express, so this list must not be turned into the classifier.
///
/// Its job is to catch the drift `unclassified_fraction` cannot see. If the
/// compiler renames a comment to something that happens to match a *different*
/// rule, opcodes keep getting classified (just wrongly) and the unclassified
/// fraction never moves — but the term it used to match vanishes from
/// `vocabulary_terms_present`.
const STAGE_VOCABULARY: [&str; 27] = [
    "gpt-oss router",
    "router token",
    "router dot token",
    "gpt-oss vram scatter-add",
    "_scatter",
    "allocate vram matrix step6_pair",
    "_route",
    "materialize route weight",
    "vram matrix mul",
    "true-zero vram rows",
    "step6_device_routing_acc",
    "gpt-oss gather token rows",
    "gather pair",
    "clear gather padding",
    "_gather",
    "dynamic expert bias add",
    "_sigmoid",
    "tile row min fp",
    "tile row max fp",
    "vram fill zero",
    "vram matrix add",
    "dynamic hbm weight prefetch",
    "expert_id_to_weight_base",
    "subblock [",
    "sub projection",
    "vram block add",
    "vram block",
];

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

/// Per-resource wall-time accumulator, in **picoseconds**. These are disjoint
/// buckets: every opcode lands in exactly one, so a total is
/// `matrix+vector+scalar+dma+other`.
///
/// Picoseconds rather than cycles because rounding per opcode systematically
/// over-counts: an opcode shorter than one clock period used to bill a whole
/// cycle, and `n` such opcodes billed `n` cycles instead of
/// `ceil(n * duration / period)`. Accumulate exactly, round once per reported
/// quantity. See [`ResourceRuntime::to_cycles_json`] for the consequence.
#[derive(Clone, Copy, Debug, Default)]
struct ResourceRuntime {
    matrix_picos: u64,
    vector_picos: u64,
    scalar_picos: u64,
    dma_picos: u64,
    other_picos: u64,
}

/// Serialized view of a [`ResourceRuntime`], in whichever unit the caller picked.
#[derive(Serialize)]
struct ResourceProxyJson {
    matrix: u64,
    vector: u64,
    scalar: u64,
    dma: u64,
    other: u64,
}

impl ResourceRuntime {
    fn add(&mut self, resource: ResourceKind, picos: u64) {
        match resource {
            ResourceKind::Matrix => self.matrix_picos += picos,
            ResourceKind::Vector => self.vector_picos += picos,
            ResourceKind::Scalar => self.scalar_picos += picos,
            ResourceKind::Dma => self.dma_picos += picos,
            ResourceKind::Other => self.other_picos += picos,
        }
    }

    fn add_runtime(&mut self, other: Self) {
        self.matrix_picos += other.matrix_picos;
        self.vector_picos += other.vector_picos;
        self.scalar_picos += other.scalar_picos;
        self.dma_picos += other.dma_picos;
        self.other_picos += other.other_picos;
    }

    fn total_picos(&self) -> u64 {
        self.matrix_picos
            + self.vector_picos
            + self.scalar_picos
            + self.dma_picos
            + self.other_picos
    }

    fn to_picos_json(self) -> ResourceProxyJson {
        ResourceProxyJson {
            matrix: self.matrix_picos,
            vector: self.vector_picos,
            scalar: self.scalar_picos,
            dma: self.dma_picos,
            other: self.other_picos,
        }
    }

    /// Cycles are a *rounded view*: each bucket rounds up independently, so bucket
    /// cycles do not necessarily sum to the parent's cycle count (they can exceed
    /// it by up to one cycle per bucket). The `_picos` fields are exact and do sum
    /// — use those whenever the arithmetic matters.
    fn to_cycles_json(self) -> ResourceProxyJson {
        ResourceProxyJson {
            matrix: picos_to_cycles(self.matrix_picos),
            vector: picos_to_cycles(self.vector_picos),
            scalar: picos_to_cycles(self.scalar_picos),
            dma: picos_to_cycles(self.dma_picos),
            other: picos_to_cycles(self.other_picos),
        }
    }
}

fn picos_to_cycles(picos: u64) -> u64 {
    picos.div_ceil(PERIOD.as_picos().max(1))
}

#[derive(Serialize)]
struct StageStatsJson {
    instructions: u64,
    wall_picos: u64,
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
    resource_proxy_picos: ResourceProxyJson,
    resource_proxy_cycles: ResourceProxyJson,
}

#[derive(Serialize)]
struct PairStageStatsJson {
    instructions: u64,
    wall_picos: u64,
    wall_cycles: u64,
    seconds: f64,
    logical_bytes_read: Option<u64>,
    logical_bytes_written: Option<u64>,
    physical_hbm_bytes_read: u64,
    physical_hbm_bytes_written: u64,
    hbm_bytes_read: u64,
    hbm_bytes_written: u64,
    resource_proxy_picos: ResourceProxyJson,
    resource_proxy_cycles: ResourceProxyJson,
}

#[derive(Serialize)]
struct PairStatsJson {
    instructions: u64,
    wall_picos: u64,
    wall_cycles: u64,
    seconds: f64,
    logical_bytes_read: Option<u64>,
    logical_bytes_written: Option<u64>,
    physical_hbm_bytes_read: u64,
    physical_hbm_bytes_written: u64,
    hbm_bytes_read: u64,
    hbm_bytes_written: u64,
    resource_proxy_picos: ResourceProxyJson,
    resource_proxy_cycles: ResourceProxyJson,
    #[serde(serialize_with = "serialize_ordered")]
    stages: Vec<(&'static str, PairStageStatsJson)>,
}

/// Coverage of the ASM-comment stage classifier.
///
/// Two independent drift signals, because they fail in different ways:
///
/// - `unclassified_fraction` catches a rename that leaves opcodes matching
///   *nothing*, so they fall into `Other`.
/// - `vocabulary_terms_present` catches a rename that leaves opcodes matching
///   *something else*. That case is invisible to `unclassified_fraction` — the
///   opcodes are still classified, just wrongly — but the term they used to match
///   disappears from this list.
#[derive(Serialize)]
struct ClassificationJson {
    routed_moe_markers_present: bool,
    label_count: usize,
    unclassified_labels: usize,
    unclassified_fraction: f64,
    vocabulary_terms_total: usize,
    vocabulary_terms_present: Vec<&'static str>,
    vocabulary_terms_absent: Vec<&'static str>,
}

#[derive(Serialize)]
struct ProfileJson {
    schema_version: u32,
    label_count: usize,
    total_instructions_executed: u64,
    total_simulation_picos: Option<u64>,
    total_simulation_cycles: Option<u64>,
    total_profiled_picos: u64,
    total_profiled_cycles: u64,
    total_stage_wall_picos: u64,
    total_stage_wall_cycles: u64,
    total_unprofiled_picos: u64,
    total_unprofiled_cycles: u64,
    cycle_accounting_status: &'static str,
    resource_accounting_status: &'static str,
    total_profiled_seconds: f64,
    total_hbm_bytes_read: u64,
    total_hbm_bytes_written: u64,
    total_resource_proxy_picos: ResourceProxyJson,
    total_resource_proxy_cycles: ResourceProxyJson,
    logical_byte_status: &'static str,
    physical_byte_status: &'static str,
    resource_cycle_status: &'static str,
    time_unit_status: &'static str,
    classification: ClassificationJson,
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
    wall_picos: u64,
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
    total_profiled_picos: u64,
    total_simulation_picos: Option<u64>,
    total_seconds: f64,
    total_hbm_bytes_read: u64,
    total_hbm_bytes_written: u64,
    total_resource_proxy: ResourceRuntime,
    routed_moe_markers: bool,
    vocabulary_terms_present: Vec<&'static str>,
}

impl StageProfiler {
    pub(crate) fn from_asm(path: &Path, expected_ops: usize) -> std::io::Result<Self> {
        let asm = fs::read_to_string(path)?;
        let asm_lower = asm.to_ascii_lowercase();
        let routed_moe_markers = ROUTED_MOE_MARKERS.iter().any(|m| asm_lower.contains(m));
        let vocabulary_terms_present: Vec<&'static str> = STAGE_VOCABULARY
            .iter()
            .copied()
            .filter(|term| asm_lower.contains(term))
            .collect();
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

        if routed_moe_markers && !labels.is_empty() {
            let unclassified = labels
                .iter()
                .filter(|label| matches!(label, StageKind::Other))
                .count();
            let fraction = unclassified as f64 / labels.len() as f64;
            if fraction > UNCLASSIFIED_WARN_THRESHOLD {
                tracing::warn!(
                    asm = %path.display(),
                    unclassified,
                    labels = labels.len(),
                    fraction,
                    "routed-MoE stage classification coverage low: compiler ASM comment \
                     vocabulary may have drifted out of sync with classify_comment"
                );
            }
        }

        Ok(Self {
            labels,
            pair_labels,
            stages: [StageRuntime::default(); 11],
            pair_stages: BTreeMap::new(),
            total_instructions: 0,
            total_profiled_picos: 0,
            total_simulation_picos: None,
            total_seconds: 0.0,
            total_hbm_bytes_read: 0,
            total_hbm_bytes_written: 0,
            total_resource_proxy: ResourceRuntime::default(),
            routed_moe_markers,
            vocabulary_terms_present,
        })
    }

    pub(crate) fn set_total_simulation_duration(&mut self, duration: Duration) {
        self.total_simulation_picos = Some(duration.as_picos());
    }

    pub(crate) fn record(
        &mut self,
        pc: usize,
        seconds: f64,
        wall_picos: u64,
        resource: ResourceKind,
        hbm_bytes_read: u64,
        hbm_bytes_written: u64,
    ) {
        let stage = self.labels.get(pc).copied().unwrap_or(StageKind::Other);
        let bucket = &mut self.stages[stage.index()];
        bucket.instructions += 1;
        bucket.wall_picos += wall_picos;
        bucket.seconds += seconds;
        bucket.hbm_bytes_read += hbm_bytes_read;
        bucket.hbm_bytes_written += hbm_bytes_written;
        bucket.resource_proxy.add(resource, wall_picos);
        self.total_instructions += 1;
        self.total_profiled_picos += wall_picos;
        self.total_seconds += seconds;
        self.total_hbm_bytes_read += hbm_bytes_read;
        self.total_hbm_bytes_written += hbm_bytes_written;
        self.total_resource_proxy.add(resource, wall_picos);

        if let Some(pair_id) = self.pair_labels.get(pc).copied().flatten() {
            let pair_buckets = self
                .pair_stages
                .entry(pair_id)
                .or_insert([StageRuntime::default(); 11]);
            let pair_bucket = &mut pair_buckets[stage.index()];
            pair_bucket.instructions += 1;
            pair_bucket.wall_picos += wall_picos;
            pair_bucket.seconds += seconds;
            pair_bucket.hbm_bytes_read += hbm_bytes_read;
            pair_bucket.hbm_bytes_written += hbm_bytes_written;
            pair_bucket.resource_proxy.add(resource, wall_picos);
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
                // Fraction from the exact picosecond values, not the rounded cycle
                // views, so the fractions across stages still sum to 1.0.
                let cycle_fraction = if self.total_profiled_picos == 0 {
                    0.0
                } else {
                    stats.wall_picos as f64 / self.total_profiled_picos as f64
                };
                (
                    stage.name(),
                    StageStatsJson {
                        instructions: stats.instructions,
                        wall_picos: stats.wall_picos,
                        wall_cycles: picos_to_cycles(stats.wall_picos),
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
                        resource_proxy_picos: stats.resource_proxy.to_picos_json(),
                        resource_proxy_cycles: stats.resource_proxy.to_cycles_json(),
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
                                wall_picos: stats.wall_picos,
                                wall_cycles: picos_to_cycles(stats.wall_picos),
                                seconds: stats.seconds,
                                logical_bytes_read: None,
                                logical_bytes_written: None,
                                physical_hbm_bytes_read: stats.hbm_bytes_read,
                                physical_hbm_bytes_written: stats.hbm_bytes_written,
                                hbm_bytes_read: stats.hbm_bytes_read,
                                hbm_bytes_written: stats.hbm_bytes_written,
                                resource_proxy_picos: stats.resource_proxy.to_picos_json(),
                                resource_proxy_cycles: stats.resource_proxy.to_cycles_json(),
                            },
                        )
                    })
                    .collect();
                (
                    *pair_id,
                    PairStatsJson {
                        instructions: totals.instructions,
                        wall_picos: totals.wall_picos,
                        wall_cycles: picos_to_cycles(totals.wall_picos),
                        seconds: totals.seconds,
                        logical_bytes_read: None,
                        logical_bytes_written: None,
                        physical_hbm_bytes_read: totals.hbm_bytes_read,
                        physical_hbm_bytes_written: totals.hbm_bytes_written,
                        hbm_bytes_read: totals.hbm_bytes_read,
                        hbm_bytes_written: totals.hbm_bytes_written,
                        resource_proxy_picos: totals.resource_proxy.to_picos_json(),
                        resource_proxy_cycles: totals.resource_proxy.to_cycles_json(),
                        stages: per_stage,
                    },
                )
            })
            .collect();

        let total_stage_wall_picos = sum_stage_runtimes(&self.stages).wall_picos;
        let total_simulation_picos = self.total_simulation_picos;
        let total_unprofiled_picos = total_simulation_picos
            .map(|picos| picos.saturating_sub(self.total_profiled_picos))
            .unwrap_or(0);
        // Compared in picoseconds, so the verdict is exact and independent of the
        // clock period. The old cycle-domain comparison only held because the DRAM
        // tCK happened to equal PERIOD; any preset or frequency change made it fail
        // for rounding reasons alone.
        let cycle_accounting_status = match total_simulation_picos {
            Some(picos) if picos == self.total_profiled_picos => "profiled_time_matches_total",
            Some(_) => "profiled_time_does_not_match_total",
            None => "total_simulation_time_unset",
        };

        // The resource buckets partition every profiled opcode, so their picosecond
        // sum must equal the profiled total. A mismatch means resource_kind_for_opcode
        // double-counted or dropped an opcode class -- report it rather than let a
        // silently lossy attribution look authoritative.
        let resource_accounting_status =
            if self.total_resource_proxy.total_picos() == self.total_profiled_picos {
                "resource_buckets_sum_to_profiled_time"
            } else {
                "resource_buckets_do_not_sum_to_profiled_time"
            };

        let unclassified_labels = self
            .labels
            .iter()
            .filter(|label| matches!(label, StageKind::Other))
            .count();
        let label_count = self.labels.len();
        let unclassified_fraction = if label_count == 0 {
            0.0
        } else {
            unclassified_labels as f64 / label_count as f64
        };

        ProfileJson {
            schema_version: 3,
            label_count: self.labels.len(),
            total_instructions_executed: self.total_instructions,
            total_simulation_picos,
            total_simulation_cycles: total_simulation_picos.map(picos_to_cycles),
            total_profiled_picos: self.total_profiled_picos,
            total_profiled_cycles: picos_to_cycles(self.total_profiled_picos),
            total_stage_wall_picos,
            total_stage_wall_cycles: picos_to_cycles(total_stage_wall_picos),
            total_unprofiled_picos,
            total_unprofiled_cycles: picos_to_cycles(total_unprofiled_picos),
            cycle_accounting_status,
            resource_accounting_status,
            total_profiled_seconds: self.total_seconds,
            total_hbm_bytes_read: self.total_hbm_bytes_read,
            total_hbm_bytes_written: self.total_hbm_bytes_written,
            total_resource_proxy_picos: self.total_resource_proxy.to_picos_json(),
            total_resource_proxy_cycles: self.total_resource_proxy.to_cycles_json(),
            logical_byte_status: LOGICAL_BYTE_STATUS,
            physical_byte_status: PHYSICAL_BYTE_STATUS,
            resource_cycle_status: RESOURCE_CYCLE_STATUS,
            time_unit_status: TIME_UNIT_STATUS,
            classification: ClassificationJson {
                routed_moe_markers_present: self.routed_moe_markers,
                vocabulary_terms_total: STAGE_VOCABULARY.len(),
                vocabulary_terms_present: self.vocabulary_terms_present.clone(),
                vocabulary_terms_absent: STAGE_VOCABULARY
                    .iter()
                    .copied()
                    .filter(|term| !self.vocabulary_terms_present.contains(term))
                    .collect(),
                label_count,
                unclassified_labels,
                unclassified_fraction,
            },
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
        total.wall_picos += stats.wall_picos;
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

    /// Pins the classifier's behaviour on representative real comment lines.
    ///
    /// `classify_comment` is a priority-ordered if/else chain over compiler comment
    /// substrings, with conjunctions, a negation, and stateful carry-over. Inserting
    /// or reordering a rule can silently re-home opcodes that already matched an
    /// earlier one (`"vram matrix mul"` alone appears in two branches, resolved only
    /// by order). These cases lock that in.
    #[test]
    fn classify_comment_pins_the_stage_vocabulary() {
        use StageKind::*;
        let cases: &[(&str, StageKind, StageKind)] = &[
            // (comment, incoming stage, expected stage)
            ("; gpt-oss router dot token 0", Other, RouterTopk),
            ("; gpt-oss vram scatter-add pair3", Other, ScatterCombine),
            ("; trace_pair7_scatter", Other, ScatterCombine),
            (
                "; allocate vram matrix step6_pair2_route",
                Other,
                ExpertRouteWeight,
            ),
            ("; materialize route weight", Other, ExpertRouteWeight),
            ("; step6_device_routing_acc", Other, AccumulatorInit),
            ("; gpt-oss gather token rows", Other, Gather),
            ("; allocate vram matrix step6_pair2_gather", Other, Gather),
            ("; clear gather padding", Other, Gather),
            ("; dynamic expert bias add", Other, ExpertBias),
            (
                "; allocate vram matrix step6_pair2_sigmoid",
                Other,
                ExpertActivation,
            ),
            ("; tile row max fp", Other, ExpertActivation),
            ("; dynamic hbm weight prefetch", Other, ExpertWeightAddress),
            ("; expert_id_to_weight_base", Other, ExpertWeightAddress),
            ("; subblock [0]", Other, ExpertWeightPrefetch),
            ("; sub projection 0", Other, ExpertProjection),
            ("; vram block add", Other, ExpertProjection),
            // The negation branch: step6_pair without _gather is a projection.
            (
                "; allocate vram matrix step6_pair2_up",
                Other,
                ExpertProjection,
            ),
            // Order-sensitive: "vram matrix mul" resolves to ExpertRouteWeight
            // because that branch precedes the ExpertActivation one.
            ("; vram matrix mul", Other, ExpertRouteWeight),
            ("; vram matrix mul", ExpertActivation, ExpertRouteWeight),
            // Stateful carry-over: same text, different incoming stage.
            ("; true-zero vram rows", Other, AccumulatorInit),
            (
                "; true-zero vram rows",
                ExpertRouteWeight,
                ExpertRouteWeight,
            ),
            ("; vram fill zero", ExpertActivation, ExpertActivation),
            // Unrecognised comments inherit the current stage rather than resetting.
            ("; something the compiler invented", Gather, Gather),
        ];
        for (comment, incoming, expected) in cases {
            assert_eq!(
                classify_comment(comment, *incoming),
                *expected,
                "classify_comment({comment:?}, {incoming:?})"
            );
        }
    }

    /// Every declared vocabulary term must actually influence classification,
    /// otherwise `vocabulary_terms_present` would report drift in a term the
    /// classifier no longer cares about (or miss one it does).
    #[test]
    fn stage_vocabulary_terms_are_wellformed_and_unique() {
        let mut seen = std::collections::HashSet::new();
        for term in STAGE_VOCABULARY {
            assert!(!term.is_empty(), "empty vocabulary term");
            assert_eq!(
                term,
                term.to_ascii_lowercase(),
                "vocabulary terms are matched against lowercased ASM, so {term:?} \
                 must itself be lowercase or it can never match"
            );
            assert!(seen.insert(term), "duplicate vocabulary term {term:?}");
        }
        assert_eq!(seen.len(), STAGE_VOCABULARY.len());
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
    fn picos_to_cycles_rounds_up_to_period() {
        assert_eq!(picos_to_cycles(0), 0);
        assert_eq!(picos_to_cycles(999), 1);
        assert_eq!(picos_to_cycles(1000), 1);
        assert_eq!(picos_to_cycles(1001), 2);
    }

    #[test]
    fn picosecond_accumulation_does_not_round_per_opcode() {
        // Three 400 ps opcodes total 1200 ps -> 2 cycles. Rounding each opcode to a
        // whole cycle first (the pre-schema-v3 behaviour) would have billed 3.
        let mut runtime = ResourceRuntime::default();
        for _ in 0..3 {
            runtime.add(ResourceKind::Matrix, 400);
        }
        assert_eq!(runtime.total_picos(), 1200);
        assert_eq!(runtime.to_picos_json().matrix, 1200);
        assert_eq!(runtime.to_cycles_json().matrix, 2);
    }

    #[test]
    fn resource_buckets_are_disjoint_and_sum_to_the_total() {
        let mut runtime = ResourceRuntime::default();
        runtime.add(ResourceKind::Matrix, 1500);
        runtime.add(ResourceKind::Vector, 700);
        runtime.add(ResourceKind::Scalar, 250);
        runtime.add(ResourceKind::Dma, 3300);
        runtime.add(ResourceKind::Other, 50);
        assert_eq!(runtime.total_picos(), 1500 + 700 + 250 + 3300 + 50);

        // Picos are additive...
        let picos = runtime.to_picos_json();
        assert_eq!(
            picos.matrix + picos.vector + picos.scalar + picos.dma + picos.other,
            runtime.total_picos()
        );
        // ...cycles are not: each bucket rounds up independently, so the bucket sum
        // (2+1+1+4+1 = 9) exceeds the rounded total (ceil(5800/1000) = 6). This is
        // why consumers must do arithmetic on the picosecond fields.
        let cycles = runtime.to_cycles_json();
        assert_eq!(
            cycles.matrix + cycles.vector + cycles.scalar + cycles.dma + cycles.other,
            9
        );
        assert_eq!(picos_to_cycles(runtime.total_picos()), 6);
    }

    #[test]
    fn add_runtime_merges_every_bucket() {
        let mut a = ResourceRuntime::default();
        a.add(ResourceKind::Matrix, 10);
        a.add(ResourceKind::Dma, 40);
        let mut b = ResourceRuntime::default();
        b.add(ResourceKind::Matrix, 5);
        b.add(ResourceKind::Vector, 7);
        b.add(ResourceKind::Scalar, 9);
        b.add(ResourceKind::Other, 11);
        a.add_runtime(b);
        let picos = a.to_picos_json();
        assert_eq!(picos.matrix, 15);
        assert_eq!(picos.vector, 7);
        assert_eq!(picos.scalar, 9);
        assert_eq!(picos.dma, 40);
        assert_eq!(picos.other, 11);
        assert_eq!(a.total_picos(), 82);
    }
}
