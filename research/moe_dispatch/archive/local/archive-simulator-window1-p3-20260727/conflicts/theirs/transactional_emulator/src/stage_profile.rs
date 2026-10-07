use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use runtime::Duration;

use crate::runtime_config::PERIOD;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResourceKind {
    Matrix,
    Vector,
    Scalar,
    Dma,
    Other,
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

#[derive(Clone, Copy, Debug, Default)]
struct ResourceRuntime {
    matrix_cycles: u64,
    vector_cycles: u64,
    scalar_cycles: u64,
    dma_cycles: u64,
    ramulator_proxy_cycles: u64,
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

    pub(crate) fn write_json(&self, path: &Path) -> std::io::Result<()> {
        let mut out = String::new();
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

        out.push_str("{\n");
        out.push_str("  \"schema_version\": 2,\n");
        out.push_str(&format!("  \"label_count\": {},\n", self.labels.len()));
        out.push_str(&format!(
            "  \"total_instructions_executed\": {},\n",
            self.total_instructions
        ));
        match total_simulation_cycles {
            Some(cycles) => out.push_str(&format!("  \"total_simulation_cycles\": {},\n", cycles)),
            None => out.push_str("  \"total_simulation_cycles\": null,\n"),
        }
        out.push_str(&format!(
            "  \"total_profiled_cycles\": {},\n",
            self.total_profiled_cycles
        ));
        out.push_str(&format!(
            "  \"total_stage_wall_cycles\": {},\n",
            total_stage_wall_cycles
        ));
        out.push_str(&format!(
            "  \"total_unprofiled_cycles\": {},\n",
            total_unprofiled_cycles
        ));
        out.push_str(&format!(
            "  \"cycle_accounting_status\": \"{}\",\n",
            cycle_accounting_status
        ));
        out.push_str(&format!(
            "  \"total_profiled_seconds\": {:.12},\n",
            self.total_seconds
        ));
        out.push_str(&format!(
            "  \"total_hbm_bytes_read\": {},\n",
            self.total_hbm_bytes_read
        ));
        out.push_str(&format!(
            "  \"total_hbm_bytes_written\": {},\n",
            self.total_hbm_bytes_written
        ));
        out.push_str(&format!(
            "  \"total_resource_proxy_cycles\": {},\n",
            resource_json(self.total_resource_proxy)
        ));
        out.push_str("  \"logical_byte_status\": \"not_declared_by_opcode_profile; join benchmark route/shape formulas for logical bytes\",\n");
        out.push_str("  \"physical_byte_status\": \"HBM bytes are emulator WithStats 64B physical transfer deltas\",\n");
        out.push_str("  \"resource_cycle_status\": \"first-pass opcode-class wall-cycle proxy, not calibrated per-component busy counters\",\n");
        out.push_str("  \"stages\": {\n");

        for (idx, stage) in StageKind::ALL.iter().enumerate() {
            let stats = self.stages[stage.index()];
            let instr_fraction = if self.total_instructions == 0 {
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
            out.push_str(&format!(
                "    \"{}\": {{\"instructions\": {}, \"wall_cycles\": {}, \"seconds\": {:.12}, \"instruction_fraction\": {:.12}, \"time_fraction\": {:.12}, \"cycle_fraction\": {:.12}, \"logical_bytes_read\": null, \"logical_bytes_written\": null, \"physical_hbm_bytes_read\": {}, \"physical_hbm_bytes_written\": {}, \"hbm_bytes_read\": {}, \"hbm_bytes_written\": {}, \"resource_proxy_cycles\": {}}}",
                stage.name(),
                stats.instructions,
                stats.wall_cycles,
                stats.seconds,
                instr_fraction,
                time_fraction,
                cycle_fraction,
                stats.hbm_bytes_read,
                stats.hbm_bytes_written,
                stats.hbm_bytes_read,
                stats.hbm_bytes_written,
                resource_json(stats.resource_proxy)
            ));
            if idx + 1 != StageKind::ALL.len() {
                out.push(',');
            }
            out.push('\n');
        }

        out.push_str("  },\n");
        out.push_str("  \"pairs\": {\n");

        for (pair_idx, (pair_id, stages)) in self.pair_stages.iter().enumerate() {
            let totals = sum_stage_runtimes(stages);
            out.push_str(&format!(
                "    \"{}\": {{\"instructions\": {}, \"wall_cycles\": {}, \"seconds\": {:.12}, \"logical_bytes_read\": null, \"logical_bytes_written\": null, \"physical_hbm_bytes_read\": {}, \"physical_hbm_bytes_written\": {}, \"hbm_bytes_read\": {}, \"hbm_bytes_written\": {}, \"resource_proxy_cycles\": {}, \"stages\": {{\n",
                pair_id,
                totals.instructions,
                totals.wall_cycles,
                totals.seconds,
                totals.hbm_bytes_read,
                totals.hbm_bytes_written,
                totals.hbm_bytes_read,
                totals.hbm_bytes_written,
                resource_json(totals.resource_proxy)
            ));
            for (stage_idx, stage) in StageKind::ALL.iter().enumerate() {
                let stats = stages[stage.index()];
                out.push_str(&format!(
                    "        \"{}\": {{\"instructions\": {}, \"wall_cycles\": {}, \"seconds\": {:.12}, \"logical_bytes_read\": null, \"logical_bytes_written\": null, \"physical_hbm_bytes_read\": {}, \"physical_hbm_bytes_written\": {}, \"hbm_bytes_read\": {}, \"hbm_bytes_written\": {}, \"resource_proxy_cycles\": {}}}",
                    stage.name(),
                    stats.instructions,
                    stats.wall_cycles,
                    stats.seconds,
                    stats.hbm_bytes_read,
                    stats.hbm_bytes_written,
                    stats.hbm_bytes_read,
                    stats.hbm_bytes_written,
                    resource_json(stats.resource_proxy)
                ));
                if stage_idx + 1 != StageKind::ALL.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            out.push_str("      }");
            out.push_str("}");
            if pair_idx + 1 != self.pair_stages.len() {
                out.push(',');
            }
            out.push('\n');
        }

        out.push_str("  },\n");
        out.push_str(
            "  \"caveat\": \"Stage and routed-pair labels are derived from generated ASM comments. Cycles use simulator time only. physical_hbm_bytes_* are measured from the global WithStats 64B HBM deltas before/after each opcode. logical_bytes_* are intentionally null here and must be joined from workload shape/route formulas. resource_proxy_cycles are first-pass opcode-class wall-cycle attribution, not calibrated component busy counters. ramulator_proxy is a sub-view of dma, not an additive peer; totals should use matrix+vector+scalar+dma+other. Current do_ops still awaits each opcode, so this profile does not by itself prove cross-op overlap. Pair labels identify static routed pair slots, not necessarily unique expert IDs without joining the routing dump.\"\n",
        );
        out.push_str("}\n");
        fs::write(path, out)
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

fn resource_json(stats: ResourceRuntime) -> String {
    // ramulator_proxy is a DMA sub-view for readers that want a memory proxy;
    // do not add it to dma when computing total resource proxy cycles.
    format!(
        "{{\"matrix\": {}, \"vector\": {}, \"scalar\": {}, \"dma\": {}, \"ramulator_proxy\": {}, \"other\": {}}}",
        stats.matrix_cycles,
        stats.vector_cycles,
        stats.scalar_cycles,
        stats.dma_cycles,
        stats.ramulator_proxy_cycles,
        stats.other_cycles
    )
}

fn is_opcode_line(line: &str) -> bool {
    line.as_bytes()
        .first()
        .copied()
        .map(|byte| byte.is_ascii_uppercase())
        .unwrap_or(false)
}

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
