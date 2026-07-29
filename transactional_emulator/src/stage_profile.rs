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

const PROFILE_CAVEAT: &str = "Stage and routed-pair labels are derived from generated ASM comments. Time uses simulator time only. UNITS: *_picos fields are exact and additive -- do all arithmetic on them. *_cycles fields are the corresponding picosecond value rounded up to whole clock periods (see period_picos) and are NOT additive: each is rounded independently, so a set of n sibling values can sum to as much as n-1 cycles more than their parent (an individual child never exceeds its parent). seconds/total_profiled_seconds are the same quantity as *_picos in f64 and carry accumulation drift; prefer picos. BYTES: physical_hbm_bytes_* are measured from the global WithStats 64B HBM deltas before/after each opcode; hbm_bytes_read/hbm_bytes_written are older aliases of the physical figures, kept for compatibility; logical_bytes_* are intentionally null here and must be joined from workload shape/route formulas. resource_proxy_picos/resource_proxy_cycles are first-pass opcode-class wall-time attribution, not calibrated component busy counters; the buckets are disjoint, so a total is matrix+vector+scalar+dma+other -- but only in the picos view. Current do_ops still awaits each opcode, so this profile does not by itself prove cross-op overlap. Pair labels identify static routed pair slots, not necessarily unique expert IDs without joining the routing dump.";

const LOGICAL_BYTE_STATUS: &str =
    "not_declared_by_opcode_profile; join benchmark route/shape formulas for logical bytes";
const PHYSICAL_BYTE_STATUS: &str = "HBM bytes are emulator WithStats 64B physical transfer deltas";
const RESOURCE_CYCLE_STATUS: &str =
    "first-pass opcode-class wall-time proxy, not calibrated per-component busy counters";

/// Explains the picos/cycles relationship to anyone reading the JSON without the
/// full caveat string.
const TIME_UNIT_STATUS: &str = "picos are exact and additive; cycles are picos rounded up to whole period_picos and are not additive across levels";

/// Warn when a routed-MoE program leaves more than this fraction of opcodes
/// unclassified (a strong signal the compiler comment vocabulary drifted out of
/// sync with `classify_comment`).
const UNCLASSIFIED_WARN_THRESHOLD: f64 = 0.5;

/// Substrings that mark a program as routed-MoE / expert-computation (so a high
/// unclassified fraction is a drift signal rather than expected). Broad on
/// purpose: matching more of the stage vocabulary makes the warning robust to a
/// *partial* comment rename. Kept lowercase; matched against the lowercased ASM.
const ROUTED_MOE_MARKERS: [&str; 5] = [
    "step6_pair",
    "gpt-oss",
    "sub projection",
    "expert_",
    // Marker-emitting programs need not contain any of the prose above: once the
    // compiler routes attribution through `@stage=`, "gpt-oss" in particular
    // disappears from policy-agnostic emitters.
    "@stage=",
];

/// Substrings `extract_pair_id` keys on. Separate from `STAGE_VOCABULARY` because
/// they drive *pair attribution* rather than stage labelling, but they are the
/// same kind of undeclared cross-repo contract: if the compiler renames `pair=`,
/// every `pair_id` silently becomes `None`, the `pairs` map empties, and no
/// coverage or vocabulary guard would otherwise notice.
const PAIR_ID_VOCABULARY: [&str; 2] = ["step6_pair", "pair="];

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
///
/// Note this signal is per-term over the whole comment stream, so it cannot see a
/// *partial* rename (one surviving occurrence keeps a term "present") or a broken
/// conjunction (both conjuncts present, never co-occurring on a line). Those are
/// what `stage_instruction_counts` is for.
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
/// proxy. This is a coarse wall-time attribution keyed on opcode family, NOT a
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
    /// Share of `total_profiled_picos`. Exact (u64 ratio), so these sum to 1.0
    /// across stages -- unlike anything derived from the f64 `seconds`.
    time_fraction: f64,
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
    /// How stages were attributed.
    ///
    /// `"explicit_stage_markers"` -- the ASM carried `@stage=` markers and the
    /// legacy substring rules were disabled entirely. Attribution is then a
    /// declared contract rather than an inferred one.
    ///
    /// `"legacy_comment_substrings"` -- no markers present, so stages were guessed
    /// from comment prose. Every `vocabulary_*` field below only means anything in
    /// this mode; under markers they describe rules that never ran.
    stage_attribution: &'static str,
    /// `@stage=` names the compiler emitted that no `StageKind` matches.
    ///
    /// Non-empty means the two repos' stage vocabularies have drifted: the
    /// compiler validates names against `MOE_STAGES` before emitting, so anything
    /// here was known to *it*. Those regions silently inherit the previous stage,
    /// which is why this is reported rather than only logged.
    unresolved_stage_markers: Vec<String>,
    vocabulary_terms_total: usize,
    vocabulary_terms_present: Vec<&'static str>,
    vocabulary_terms_absent: Vec<&'static str>,
    /// Presence of the substrings `extract_pair_id` depends on. A rename here
    /// empties the `pairs` map without moving any other reported number.
    pair_id_terms_present: Vec<&'static str>,
    /// Executed instructions per stage. This is the strongest available drift
    /// signal: a rename that re-homes opcodes into a *different* stage moves these
    /// counts even when every vocabulary term is still present and the unclassified
    /// fraction is unchanged. Consumers should assert the stages a program cannot
    /// possibly contain are zero.
    #[serde(serialize_with = "serialize_ordered")]
    stage_instruction_counts: Vec<(&'static str, u64)>,
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
    total_unprofiled_picos: u64,
    total_unprofiled_cycles: u64,
    cycle_accounting_status: &'static str,
    /// Picoseconds per cycle. Emitted so a consumer holding only this JSON can
    /// convert between the `_picos` and `_cycles` views without knowing the
    /// emulator's clock.
    period_picos: u64,
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
    /// Shared-expert gate/up/down projections. A shared expert is dense over every
    /// token, so this is the architecturally-unavoidable half of MoE cost, as
    /// against `ExpertProjection`'s routing-dependent half.
    SharedExpertProjection,
    SharedExpertActivation,
    /// Qwen2-MoE's `sigmoid(x @ w_gate)` scalar gate on the shared output. No other
    /// shared-expert architecture has one, so this stage is empty for DeepSeek,
    /// Llama-4 and GLM.
    SharedExpertGate,
    Other,
}

impl StageKind {
    const ALL: [StageKind; STAGE_COUNT] = [
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
        StageKind::SharedExpertProjection,
        StageKind::SharedExpertActivation,
        StageKind::SharedExpertGate,
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
            StageKind::SharedExpertProjection => "shared_expert_projection",
            StageKind::SharedExpertActivation => "shared_expert_activation",
            StageKind::SharedExpertGate => "shared_expert_gate",
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
            StageKind::SharedExpertProjection => 10,
            StageKind::SharedExpertActivation => 11,
            StageKind::SharedExpertGate => 12,
            StageKind::Other => 13,
        }
    }

    /// Resolve the name carried by a `; @stage=<name>` marker.
    ///
    /// `None` for an unrecognised name, which the caller keeps as the current
    /// stage. The compiler validates marker names against its own `MOE_STAGES` set
    /// before emitting, so an unknown one here means the two repos have drifted --
    /// `unresolved_stage_markers` in the JSON reports exactly that, since silently
    /// falling through would look identical to a program that never marked at all.
    fn from_tag(tag: &str) -> Option<StageKind> {
        StageKind::ALL
            .iter()
            .copied()
            .find(|stage| stage.name() == tag && !matches!(stage, StageKind::Other))
    }
}

/// Number of `StageKind` variants, including `Other`.
const STAGE_COUNT: usize = 14;

/// Comment prefix carrying an explicit stage attribution, emitted by the
/// compiler's `moe_stage_marker`. Kept in sync by
/// `stage_marker_vocabulary_matches_the_compiler` and the compiler-side
/// `MOE_STAGES` set.
const STAGE_MARKER_PREFIX: &str = "@stage=";

/// Extract the stage name from a marker comment, if the line is one.
///
/// Accepts `; @stage=gather pairs=8` and `;@stage=gather`; the name runs to the
/// first whitespace so markers can carry human-readable detail after it.
fn extract_stage_tag(comment: &str) -> Option<&str> {
    let rest = comment.trim_start_matches(';').trim_start();
    let rest = rest.strip_prefix(STAGE_MARKER_PREFIX)?;
    Some(rest.split_whitespace().next().unwrap_or(""))
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
    stages: [StageRuntime; STAGE_COUNT],
    pair_stages: BTreeMap<u32, [StageRuntime; STAGE_COUNT]>,
    total_instructions: u64,
    total_profiled_picos: u64,
    total_simulation_picos: Option<u64>,
    total_seconds: f64,
    total_hbm_bytes_read: u64,
    total_hbm_bytes_written: u64,
    total_resource_proxy: ResourceRuntime,
    routed_moe_markers: bool,
    /// Whether stages came from explicit `@stage=` markers rather than the legacy
    /// substring rules. Reported so a consumer can tell a genuinely-unclassified
    /// program from one whose compiler simply predates marker emission.
    marker_authoritative: bool,
    unresolved_stage_markers: Vec<String>,
    vocabulary_terms_present: Vec<&'static str>,
    pair_id_terms_present: Vec<&'static str>,
}

impl StageProfiler {
    pub(crate) fn from_asm(path: &Path, expected_ops: usize) -> std::io::Result<Self> {
        let asm = fs::read_to_string(path)?;
        let asm_lower = asm.to_ascii_lowercase();
        let routed_moe_markers = ROUTED_MOE_MARKERS.iter().any(|m| asm_lower.contains(m));
        // Scan only comment lines: `classify_comment` is called for nothing else,
        // so matching a term in an opcode, label or operand would report a term as
        // "live" that the classifier never sees.
        let comment_text: String = asm_lower
            .lines()
            .map(str::trim_start)
            .filter(|line| line.starts_with(';'))
            .collect::<Vec<_>>()
            .join("\n");
        let vocabulary_terms_present: Vec<&'static str> = STAGE_VOCABULARY
            .iter()
            .copied()
            .filter(|term| comment_text.contains(term))
            .collect();
        // A program that carries any `@stage=` marker is classified *only* by its
        // markers; the legacy substring rules are switched off for the whole file.
        //
        // Mixing the two is not a conservative middle ground, it is wrong. Markers
        // are sticky, so a marked region is meant to hold until the next marker --
        // but a marked region's body still contains ordinary comments from
        // general-purpose helpers ("sub projection", "subblock ["), and a live
        // substring rule would immediately steal the region back. Marking
        // `shared_expert_projection` and then having `linear_projection`'s own
        // comments reclassify the work as routed `expert_projection` is precisely
        // the bug the markers exist to remove.
        let marker_authoritative = asm
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with(';'))
            .any(|line| extract_stage_tag(line).is_some());

        let mut labels = Vec::with_capacity(expected_ops);
        let mut pair_labels = Vec::with_capacity(expected_ops);
        let mut stage = StageKind::Other;
        let mut pair_id = None;
        let mut unresolved_stage_markers: Vec<String> = Vec::new();

        for raw_line in asm.lines() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with(';') {
                if marker_authoritative {
                    if let Some(tag) = extract_stage_tag(line) {
                        match StageKind::from_tag(tag) {
                            Some(marked) => stage = marked,
                            None => {
                                // Keep the current stage, but surface the name: the
                                // compiler validates against its own MOE_STAGES set,
                                // so reaching here means the two vocabularies drifted.
                                if !unresolved_stage_markers.iter().any(|seen| seen == tag) {
                                    unresolved_stage_markers.push(tag.to_string());
                                }
                            }
                        }
                    }
                } else {
                    stage = classify_comment(line, stage);
                }
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

        if !unresolved_stage_markers.is_empty() {
            tracing::warn!(
                asm = %path.display(),
                markers = ?unresolved_stage_markers,
                "compiler emitted @stage= names the emulator does not know; \
                 StageKind and the compiler's MOE_STAGES have drifted apart"
            );
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
            stages: [StageRuntime::default(); STAGE_COUNT],
            pair_stages: BTreeMap::new(),
            total_instructions: 0,
            total_profiled_picos: 0,
            total_simulation_picos: None,
            total_seconds: 0.0,
            total_hbm_bytes_read: 0,
            total_hbm_bytes_written: 0,
            total_resource_proxy: ResourceRuntime::default(),
            routed_moe_markers,
            marker_authoritative,
            unresolved_stage_markers,
            vocabulary_terms_present,
            pair_id_terms_present: PAIR_ID_VOCABULARY
                .iter()
                .copied()
                .filter(|term| comment_text.contains(term))
                .collect(),
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
                .or_insert([StageRuntime::default(); STAGE_COUNT]);
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
        // The resource buckets partition every profiled opcode: `record` adds the
        // same `wall_picos` to one bucket and to the profiled total. That makes the
        // equality hold by construction, so it is asserted here rather than emitted
        // as a status field pretending to be a live check. It earns its keep the
        // moment someone adds a bucket that is a *sub-view* of another (an earlier
        // revision had `ramulator_proxy` double-counting `dma`), which is exactly
        // when the "disjoint" claim in PROFILE_CAVEAT would stop being true.
        debug_assert_eq!(
            self.total_resource_proxy.total_picos(),
            self.total_profiled_picos,
            "resource proxy buckets must partition profiled time"
        );
        let stages = StageKind::ALL
            .iter()
            .map(|stage| {
                let stats = self.stages[stage.index()];
                let instruction_fraction = if self.total_instructions == 0 {
                    0.0
                } else {
                    stats.instructions as f64 / self.total_instructions as f64
                };
                // From the exact picosecond values -- not the rounded cycle views
                // and not the f64 `seconds` accumulator -- so the fractions across
                // stages sum to exactly 1.0. Pre-v3 this was two fields
                // (`time_fraction` from `seconds`, `cycle_fraction` from per-op
                // rounded cycles); in the picosecond domain they are the same ratio,
                // so only one is emitted.
                let time_fraction = if self.total_profiled_picos == 0 {
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
            schema_version: 4,
            label_count: self.labels.len(),
            total_instructions_executed: self.total_instructions,
            total_simulation_picos,
            total_simulation_cycles: total_simulation_picos.map(picos_to_cycles),
            total_profiled_picos: self.total_profiled_picos,
            total_profiled_cycles: picos_to_cycles(self.total_profiled_picos),
            total_unprofiled_picos,
            // Deliberately the additive complement rather than
            // picos_to_cycles(total_unprofiled_picos), so that
            // profiled_cycles + unprofiled_cycles == simulation_cycles holds. These
            // three are the one place a reader will expect cycles to add up.
            total_unprofiled_cycles: total_simulation_picos
                .map(picos_to_cycles)
                .unwrap_or(0)
                .saturating_sub(picos_to_cycles(self.total_profiled_picos)),
            cycle_accounting_status,
            period_picos: PERIOD.as_picos().max(1),
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
                stage_attribution: if self.marker_authoritative {
                    "explicit_stage_markers"
                } else {
                    "legacy_comment_substrings"
                },
                unresolved_stage_markers: self.unresolved_stage_markers.clone(),
                vocabulary_terms_total: STAGE_VOCABULARY.len(),
                vocabulary_terms_present: self.vocabulary_terms_present.clone(),
                vocabulary_terms_absent: STAGE_VOCABULARY
                    .iter()
                    .copied()
                    .filter(|term| !self.vocabulary_terms_present.contains(term))
                    .collect(),
                pair_id_terms_present: self.pair_id_terms_present.clone(),
                stage_instruction_counts: StageKind::ALL
                    .iter()
                    .map(|stage| (stage.name(), self.stages[stage.index()].instructions))
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

fn sum_stage_runtimes(stages: &[StageRuntime; STAGE_COUNT]) -> StageRuntime {
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
        || (text.contains("true-zero vram rows") && matches!(current, StageKind::ExpertRouteWeight))
    {
        // NOTE: `"vram matrix mul"` used to be an unconditional disjunct here. That
        // made the guarded copy further down (which keeps activation-region
        // multiplies in `ExpertActivation`) unreachable, and it claimed comments
        // emitted by the compiler's *general-purpose* `VRAM Matrix Mul` helper --
        // not the routed-MoE emitter. On gpt_oss_moe_expert, a single-expert
        // program with no routing at all, it flipped the stage mid-activation and
        // then carried over, billing ~25% of opcodes (53 instructions) to
        // `expert_route_weight`. The genuine route-weight comment is
        // "materialize route weight", which the first disjunct already catches.
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
            // "vram matrix mul" comes from the compiler's *general-purpose* matrix
            // helper, so on its own it says nothing about the stage and must leave
            // the current one alone. Inside an activation region it keeps the
            // opcodes in ExpertActivation -- the rule that an unconditional
            // route-weight disjunct used to make unreachable.
            ("; vram matrix mul", Other, Other),
            ("; vram matrix mul", ExpertActivation, ExpertActivation),
            // ...but a route-weight region still carries over, because unmatched
            // comments inherit the current stage.
            ("; vram matrix mul", ExpertRouteWeight, ExpertRouteWeight),
            // Stateful carry-over: same text, different incoming stage.
            ("; true-zero vram rows", Other, AccumulatorInit),
            (
                "; true-zero vram rows",
                ExpertRouteWeight,
                ExpertRouteWeight,
            ),
            ("; vram fill zero", ExpertActivation, ExpertActivation),
            // Terms that were previously shadowed by another case in this list, so
            // deleting them from classify_comment left the suite green.
            ("; gpt-oss router pair0", Other, RouterTopk),
            ("; router token 3", Other, RouterTopk),
            ("; gather pair slot 1", Other, Gather),
            ("; tile row min fp", Other, ExpertActivation),
            ("; vram matrix add", ExpertActivation, ExpertActivation),
            ("; vram block 2", Other, ExpertProjection),
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

    /// `STAGE_VOCABULARY` must list *every* substring `classify_comment` keys on.
    ///
    /// This is the property the whole drift guard rests on, and it was previously
    /// enforced only by author diligence: adding a `text.contains("...")` rule and
    /// forgetting the array silently reopens the blind spot the guard exists to
    /// close. Parsing our own source is crude, but it is the only way to tie the
    /// two together without a macro.
    #[test]
    fn stage_vocabulary_lists_every_literal_classify_comment_matches() {
        let source = include_str!("stage_profile.rs");
        let body_start = source
            .find("fn classify_comment(")
            .expect("classify_comment must exist");
        // The function ends at the next top-level item.
        let body_end = body_start
            + source[body_start..]
                .find("\n}\n")
                .expect("classify_comment must be brace-terminated");
        let body = &source[body_start..body_end];

        let mut literals = Vec::new();
        let mut rest = body;
        while let Some(idx) = rest.find("text.contains(\"") {
            rest = &rest[idx + "text.contains(\"".len()..];
            let end = rest.find('"').expect("unterminated string literal");
            literals.push(&rest[..end]);
            rest = &rest[end..];
        }
        assert!(
            literals.len() >= 20,
            "parser found only {} literals; it has probably broken",
            literals.len()
        );

        let declared: std::collections::HashSet<&str> = STAGE_VOCABULARY.iter().copied().collect();
        let missing: Vec<&str> = literals
            .iter()
            .copied()
            .filter(|lit| !declared.contains(lit))
            .collect();
        assert!(
            missing.is_empty(),
            "classify_comment matches {missing:?}, which STAGE_VOCABULARY does not declare; \
             vocabulary_terms_present would be blind to a rename of those"
        );

        let found: std::collections::HashSet<&str> = literals.into_iter().collect();
        let unused: Vec<&str> = STAGE_VOCABULARY
            .iter()
            .copied()
            .filter(|term| !found.contains(term))
            .collect();
        assert!(
            unused.is_empty(),
            "STAGE_VOCABULARY declares {unused:?}, which classify_comment never matches; \
             those would be reported as drift in terms the classifier does not use"
        );
    }

    #[test]
    fn vocabulary_terms_are_lowercase_and_unique() {
        // Terms are matched against lowercased ASM, so an uppercase character makes
        // a term dead on arrival.
        let mut seen = std::collections::HashSet::new();
        for term in STAGE_VOCABULARY.iter().chain(PAIR_ID_VOCABULARY.iter()) {
            assert!(!term.is_empty(), "empty vocabulary term");
            assert_eq!(
                *term,
                term.to_ascii_lowercase(),
                "{term:?} can never match lowercased ASM"
            );
            assert!(seen.insert(*term), "duplicate vocabulary term {term:?}");
        }
    }

    /// Build a profiler over a synthetic label list, bypassing `from_asm`.
    fn profiler_with_labels(labels: Vec<StageKind>) -> StageProfiler {
        StageProfiler {
            pair_labels: vec![None; labels.len()],
            labels,
            stages: [StageRuntime::default(); STAGE_COUNT],
            pair_stages: BTreeMap::new(),
            total_instructions: 0,
            total_profiled_picos: 0,
            total_simulation_picos: None,
            total_seconds: 0.0,
            total_hbm_bytes_read: 0,
            total_hbm_bytes_written: 0,
            total_resource_proxy: ResourceRuntime::default(),
            routed_moe_markers: false,
            marker_authoritative: false,
            unresolved_stage_markers: Vec::new(),
            vocabulary_terms_present: Vec::new(),
            pair_id_terms_present: Vec::new(),
        }
    }

    /// End-to-end over the path that actually held the bug.
    ///
    /// The pre-v3 defect was in the *caller*: `do_ops` rounded each opcode to whole
    /// cycles before handing it to `record`. Every unit test lived below that
    /// boundary, so reverting the caller left the suite green. This drives
    /// `record` -> `to_json` with sub-period durations, which is the only shape that
    /// distinguishes "accumulate then round" from "round then accumulate".
    #[test]
    fn sub_period_opcodes_accumulate_before_rounding() {
        use StageKind::*;
        let mut profiler = profiler_with_labels(vec![Gather, Gather, Gather, ExpertProjection]);
        // Three 400 ps gathers (1200 ps -> 2 cycles, not 3) plus one 1600 ps
        // projection (-> 2 cycles). Total 2800 ps -> 3 cycles, not 5.
        for pc in 0..3 {
            profiler.record(pc, 400e-12, 400, ResourceKind::Dma, 64, 0);
        }
        profiler.record(3, 1600e-12, 1600, ResourceKind::Matrix, 0, 128);
        profiler.set_total_simulation_duration(Duration::from_picos(2800));

        let json = serde_json::to_value(profiler.to_json()).unwrap();
        assert_eq!(json["schema_version"], 4);
        assert_eq!(json["total_profiled_picos"], 2800);
        assert_eq!(json["total_profiled_cycles"], 3);
        assert_eq!(json["period_picos"], 1000);

        let gather = &json["stages"]["gather"];
        assert_eq!(gather["wall_picos"], 1200);
        assert_eq!(gather["wall_cycles"], 2, "per-opcode rounding would give 3");
        assert_eq!(gather["resource_proxy_picos"]["dma"], 1200);
        assert_eq!(gather["resource_proxy_cycles"]["dma"], 2);
        assert_eq!(json["stages"]["expert_projection"]["wall_picos"], 1600);

        // Fractions come from the exact picosecond values and sum to 1.
        let sum: f64 = json["stages"]
            .as_object()
            .unwrap()
            .values()
            .map(|stage| stage["time_fraction"].as_f64().unwrap())
            .sum();
        assert!((sum - 1.0).abs() < 1e-12, "time_fraction sum was {sum}");

        // Accounting is exact in picoseconds regardless of the clock period.
        assert_eq!(
            json["cycle_accounting_status"],
            "profiled_time_matches_total"
        );
        assert_eq!(json["total_unprofiled_picos"], 0);
        // The cycle triple is the one place cycles are made to add up.
        assert_eq!(
            json["total_profiled_cycles"].as_u64().unwrap()
                + json["total_unprofiled_cycles"].as_u64().unwrap(),
            json["total_simulation_cycles"].as_u64().unwrap()
        );
        // Per-stage instruction counts are exported for the drift guard.
        assert_eq!(
            json["classification"]["stage_instruction_counts"]["gather"],
            3
        );
        assert_eq!(
            json["classification"]["stage_instruction_counts"]["router_topk"],
            0
        );
    }

    #[test]
    fn unprofiled_time_is_reported_when_the_profile_does_not_cover_the_run() {
        let mut profiler = profiler_with_labels(vec![StageKind::Gather]);
        profiler.record(0, 500e-12, 500, ResourceKind::Dma, 0, 0);
        profiler.set_total_simulation_duration(Duration::from_picos(2500));

        let json = serde_json::to_value(profiler.to_json()).unwrap();
        assert_eq!(
            json["cycle_accounting_status"],
            "profiled_time_does_not_match_total"
        );
        assert_eq!(json["total_unprofiled_picos"], 2000);
        assert_eq!(
            json["total_profiled_cycles"].as_u64().unwrap()
                + json["total_unprofiled_cycles"].as_u64().unwrap(),
            json["total_simulation_cycles"].as_u64().unwrap()
        );
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

    #[test]
    fn stage_tags_parse_with_and_without_trailing_detail() {
        assert_eq!(extract_stage_tag("; @stage=gather pairs=8"), Some("gather"));
        assert_eq!(extract_stage_tag(";@stage=scatter_combine"), Some("scatter_combine"));
        assert_eq!(
            extract_stage_tag("; @stage=shared_expert_gate [qwen2_moe] rows=4"),
            Some("shared_expert_gate")
        );
        // Not a marker: ordinary prose, and prose that merely mentions the prefix
        // somewhere other than at the start of the comment.
        assert_eq!(extract_stage_tag("; GPT-OSS gather token rows"), None);
        assert_eq!(extract_stage_tag("; see @stage=gather for details"), None);
    }

    #[test]
    fn every_stage_name_round_trips_through_from_tag() {
        for stage in StageKind::ALL {
            if matches!(stage, StageKind::Other) {
                // `other` is the fallback, not an emittable marker: resolving it
                // would let a marker *clear* attribution rather than set it.
                assert_eq!(StageKind::from_tag(stage.name()), None);
                continue;
            }
            assert_eq!(
                StageKind::from_tag(stage.name()),
                Some(stage),
                "{} does not round-trip",
                stage.name()
            );
        }
        assert_eq!(StageKind::from_tag("no_such_stage"), None);
    }

    /// The three shared-expert stages must be distinct buckets, not aliases of the
    /// routed ones -- separating dense-over-all-tokens cost from routing-dependent
    /// cost is the entire reason they exist.
    #[test]
    fn shared_expert_stages_are_distinct_buckets() {
        let shared = [
            StageKind::SharedExpertProjection,
            StageKind::SharedExpertActivation,
            StageKind::SharedExpertGate,
        ];
        let routed = [
            StageKind::ExpertProjection,
            StageKind::ExpertActivation,
            StageKind::ExpertRouteWeight,
        ];
        for s in shared {
            for r in routed {
                assert_ne!(s.index(), r.index(), "{} collides with {}", s.name(), r.name());
            }
        }
        let mut indices: Vec<usize> = StageKind::ALL.iter().map(|s| s.index()).collect();
        indices.sort_unstable();
        indices.dedup();
        assert_eq!(
            indices.len(),
            STAGE_COUNT,
            "StageKind indices must be dense and unique or the stage arrays alias"
        );
    }

    #[test]
    fn stage_marker_names_match_the_compiler_vocabulary() {
        // Mirrors MOE_STAGES in PLENA_Compiler/aten/plena/program_routed_moe.py.
        // A name the compiler emits but StageKind lacks lands in
        // `unresolved_stage_markers`; a name only the emulator knows is a stage no
        // program can ever reach. Both are silent failures without this list.
        const COMPILER_MOE_STAGES: [&str; 13] = [
            "router_topk",
            "accumulator_init",
            "gather",
            "expert_weight_address",
            "expert_weight_prefetch",
            "expert_projection",
            "expert_activation",
            "expert_bias",
            "expert_route_weight",
            "scatter_combine",
            "shared_expert_projection",
            "shared_expert_activation",
            "shared_expert_gate",
        ];
        for name in COMPILER_MOE_STAGES {
            assert!(
                StageKind::from_tag(name).is_some(),
                "compiler emits @stage={name} but no StageKind matches"
            );
        }
        let emulator_only: Vec<&str> = StageKind::ALL
            .iter()
            .filter(|s| !matches!(s, StageKind::Other))
            .map(|s| s.name())
            .filter(|name| !COMPILER_MOE_STAGES.contains(name))
            .collect();
        assert!(
            emulator_only.is_empty(),
            "StageKind has stages no compiler marker produces: {emulator_only:?}"
        );
    }
}
