use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};

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

const PROFILE_CAVEAT: &str = "Stage and routed-pair labels are derived from generated ASM comments. Time and HBM bytes are attributed by dynamically executed opcode. Bytes are measured from the global HBM stats delta before/after each opcode. Pair labels identify static routed pair slots, not necessarily unique expert IDs without joining the routing dump.";

#[derive(Serialize)]
struct StageStatsJson {
    instructions: u64,
    seconds: f64,
    instruction_fraction: f64,
    time_fraction: f64,
    hbm_bytes_read: u64,
    hbm_bytes_written: u64,
}

#[derive(Serialize)]
struct PairStageStatsJson {
    instructions: u64,
    seconds: f64,
    hbm_bytes_read: u64,
    hbm_bytes_written: u64,
}

#[derive(Serialize)]
struct PairStatsJson {
    instructions: u64,
    seconds: f64,
    hbm_bytes_read: u64,
    hbm_bytes_written: u64,
    #[serde(serialize_with = "serialize_ordered")]
    stages: Vec<(&'static str, PairStageStatsJson)>,
}

#[derive(Serialize)]
struct ProfileJson {
    schema_version: u32,
    label_count: usize,
    total_instructions_executed: u64,
    total_profiled_seconds: f64,
    total_hbm_bytes_read: u64,
    total_hbm_bytes_written: u64,
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
    seconds: f64,
    hbm_bytes_read: u64,
    hbm_bytes_written: u64,
}

pub(crate) struct StageProfiler {
    labels: Vec<StageKind>,
    pair_labels: Vec<Option<u32>>,
    stages: [StageRuntime; 11],
    pair_stages: BTreeMap<u32, [StageRuntime; 11]>,
    total_instructions: u64,
    total_seconds: f64,
    total_hbm_bytes_read: u64,
    total_hbm_bytes_written: u64,
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
            total_seconds: 0.0,
            total_hbm_bytes_read: 0,
            total_hbm_bytes_written: 0,
        })
    }

    pub(crate) fn record(
        &mut self,
        pc: usize,
        seconds: f64,
        hbm_bytes_read: u64,
        hbm_bytes_written: u64,
    ) {
        let stage = self.labels.get(pc).copied().unwrap_or(StageKind::Other);
        let bucket = &mut self.stages[stage.index()];
        bucket.instructions += 1;
        bucket.seconds += seconds;
        bucket.hbm_bytes_read += hbm_bytes_read;
        bucket.hbm_bytes_written += hbm_bytes_written;
        self.total_instructions += 1;
        self.total_seconds += seconds;
        self.total_hbm_bytes_read += hbm_bytes_read;
        self.total_hbm_bytes_written += hbm_bytes_written;

        if let Some(pair_id) = self.pair_labels.get(pc).copied().flatten() {
            let pair_buckets = self
                .pair_stages
                .entry(pair_id)
                .or_insert([StageRuntime::default(); 11]);
            let pair_bucket = &mut pair_buckets[stage.index()];
            pair_bucket.instructions += 1;
            pair_bucket.seconds += seconds;
            pair_bucket.hbm_bytes_read += hbm_bytes_read;
            pair_bucket.hbm_bytes_written += hbm_bytes_written;
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
                (
                    stage.name(),
                    StageStatsJson {
                        instructions: stats.instructions,
                        seconds: stats.seconds,
                        instruction_fraction,
                        time_fraction,
                        hbm_bytes_read: stats.hbm_bytes_read,
                        hbm_bytes_written: stats.hbm_bytes_written,
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
                                seconds: stats.seconds,
                                hbm_bytes_read: stats.hbm_bytes_read,
                                hbm_bytes_written: stats.hbm_bytes_written,
                            },
                        )
                    })
                    .collect();
                (
                    *pair_id,
                    PairStatsJson {
                        instructions: totals.instructions,
                        seconds: totals.seconds,
                        hbm_bytes_read: totals.hbm_bytes_read,
                        hbm_bytes_written: totals.hbm_bytes_written,
                        stages: per_stage,
                    },
                )
            })
            .collect();

        ProfileJson {
            schema_version: 1,
            label_count: self.labels.len(),
            total_instructions_executed: self.total_instructions,
            total_profiled_seconds: self.total_seconds,
            total_hbm_bytes_read: self.total_hbm_bytes_read,
            total_hbm_bytes_written: self.total_hbm_bytes_written,
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
        total.seconds += stats.seconds;
        total.hbm_bytes_read += stats.hbm_bytes_read;
        total.hbm_bytes_written += stats.hbm_bytes_written;
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
}
