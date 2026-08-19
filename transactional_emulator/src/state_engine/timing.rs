use std::fs;
use std::path::Path;

use serde::Serialize;

use super::descriptor::{StateDescriptor, StatePayload};
use super::generated_contract::{self as wire, StateAlgorithm, StatePrecision, StateSubop};
use super::projection::ProjectionBufferStats;

const MX_BLOCK: u64 = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StateSramLayout {
    RowMajor,
    DualAxisCyclic,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct StateTimingConfig {
    pub head_lanes: u32,
    pub row_lanes: u32,
    pub column_lanes: u32,
    pub banks_per_head_lane: u32,
    pub ports_per_bank: u32,
    pub fma_lanes_per_head_lane: u32,
    pub hbm_bytes_per_cycle: u32,
    pub head_tile_slots: u32,
    pub layout: StateSramLayout,
}

impl Default for StateTimingConfig {
    fn default() -> Self {
        Self {
            head_lanes: 1,
            row_lanes: 4,
            column_lanes: 8,
            banks_per_head_lane: 32,
            ports_per_bank: 1,
            fma_lanes_per_head_lane: 32,
            hbm_bytes_per_cycle: 64,
            head_tile_slots: 2,
            layout: StateSramLayout::RowMajor,
        }
    }
}

impl StateTimingConfig {
    pub fn validate(self) -> Result<Self, String> {
        for (name, value) in [
            ("head_lanes", self.head_lanes),
            ("row_lanes", self.row_lanes),
            ("column_lanes", self.column_lanes),
            ("banks_per_head_lane", self.banks_per_head_lane),
            ("ports_per_bank", self.ports_per_bank),
            ("fma_lanes_per_head_lane", self.fma_lanes_per_head_lane),
            ("hbm_bytes_per_cycle", self.hbm_bytes_per_cycle),
            ("head_tile_slots", self.head_tile_slots),
        ] {
            if value == 0 {
                return Err(format!("X_STATE timing {name} must be positive"));
            }
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct StateExecutionRecord {
    pub subop: String,
    pub algorithm: String,
    pub context_id: u32,
    pub request_id: u32,
    pub layer_id: u32,
    pub state_id: u32,
    pub valid_tokens: u16,
    pub state_resident: bool,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub descriptor_hbm_read_bytes: u64,
    pub parameter_hbm_read_bytes: u64,
    pub state_hbm_read_bytes: u64,
    pub state_hbm_write_bytes: u64,
    pub completion_hbm_write_bytes: u64,
    pub projection_vram_read_bytes: u64,
    pub output_vram_write_bytes: u64,
    pub projection_buffer_values: u64,
    pub projection_fifo_capacity_values: u64,
    pub projection_fifo_peak_values: u64,
    pub projection_fifo_spill_values: u64,
    pub projection_fifo_backpressure_cycles: u64,
    pub projection_write_packets: u64,
    pub projection_write_ideal_cycles: u64,
    pub projection_write_service_cycles: u64,
    pub projection_write_stall_cycles: u64,
    pub projection_read_packets: u64,
    pub projection_read_ideal_cycles: u64,
    pub projection_read_service_cycles: u64,
    pub projection_read_stall_cycles: u64,
    pub state_sram_read_values: u64,
    pub state_sram_write_values: u64,
    pub bank_packets: u64,
    pub bank_ideal_cycles: u64,
    pub bank_service_cycles: u64,
    pub bank_stall_cycles: u64,
    pub arithmetic_cycles: u64,
    pub recurrent_cycles: u64,
    pub estimated_hbm_cycles: u64,
    pub estimated_total_cycles: u64,
}

impl StateExecutionRecord {
    pub fn charged_compute_cycles(&self) -> u64 {
        self.recurrent_cycles
    }

    pub fn record_projection_buffer(
        &mut self,
        stats: ProjectionBufferStats,
        config: StateTimingConfig,
    ) {
        self.projection_buffer_values = stats.values;
        self.projection_fifo_capacity_values = stats.fifo_capacity_values;
        self.projection_fifo_peak_values = stats.fifo_peak_values;
        self.projection_fifo_spill_values = stats.fifo_spill_values;
        self.projection_fifo_backpressure_cycles = stats.fifo_backpressure_cycles;
        self.projection_write_packets = stats.write_packets;
        self.projection_write_ideal_cycles = stats.write_ideal_cycles;
        self.projection_write_service_cycles = stats.write_service_cycles;
        self.projection_write_stall_cycles = stats
            .write_service_cycles
            .saturating_sub(stats.write_ideal_cycles);
        self.projection_read_packets = stats.read_packets;
        self.projection_read_ideal_cycles = stats.read_ideal_cycles;
        self.projection_read_service_cycles = stats.read_service_cycles;
        self.projection_read_stall_cycles = stats
            .read_service_cycles
            .saturating_sub(stats.read_ideal_cycles);

        // Producer writes belong to Matrix writeback and are reported without
        // charging X_STATE twice. Consumer reads can gate the recurrent core,
        // so they participate in the same max-overlapped recurrence bound as
        // arithmetic and persistent-state SRAM service.
        self.recurrent_cycles = self
            .arithmetic_cycles
            .max(self.bank_service_cycles)
            .max(stats.read_service_cycles);
        self.recompute_total(config);
    }

    fn recompute_total(&mut self, config: StateTimingConfig) {
        let fixed_hbm_bytes = self.descriptor_hbm_read_bytes
            + self.parameter_hbm_read_bytes
            + self.completion_hbm_write_bytes;
        let streamed_state_hbm_bytes = self.state_hbm_read_bytes + self.state_hbm_write_bytes;
        self.estimated_hbm_cycles = (fixed_hbm_bytes + streamed_state_hbm_bytes)
            .div_ceil(u64::from(config.hbm_bytes_per_cycle));
        self.estimated_total_cycles = if matches!(self.subop.as_str(), "prefill" | "step")
            && !self.state_resident
            && config.head_tile_slots >= 2
        {
            fixed_hbm_bytes.div_ceil(u64::from(config.hbm_bytes_per_cycle))
                + self
                    .recurrent_cycles
                    .max(streamed_state_hbm_bytes.div_ceil(u64::from(config.hbm_bytes_per_cycle)))
        } else {
            self.recurrent_cycles + self.estimated_hbm_cycles
        };
    }
}

#[derive(Debug, Default)]
pub struct StateEngineProfile {
    pub schema_version: u32,
    pub config: Option<StateTimingConfig>,
    pub commands: Vec<StateExecutionRecord>,
}

impl StateEngineProfile {
    pub fn new(config: StateTimingConfig) -> Self {
        Self {
            schema_version: 2,
            config: Some(config),
            commands: Vec::new(),
        }
    }

    pub fn record(&mut self, command: StateExecutionRecord) {
        self.commands.push(command);
    }

    pub fn write_json(&self, path: &Path) -> std::io::Result<()> {
        #[derive(Serialize)]
        struct ProfileJson<'a> {
            schema_version: u32,
            config: Option<StateTimingConfig>,
            summary: StateProfileSummary,
            commands: &'a [StateExecutionRecord],
        }
        let rendered = serde_json::to_string_pretty(&ProfileJson {
            schema_version: self.schema_version,
            config: self.config,
            summary: self.summary(),
            commands: &self.commands,
        })
        .expect("state profile is serializable");
        fs::write(path, rendered + "\n")
    }

    pub fn summary(&self) -> StateProfileSummary {
        let mut summary = StateProfileSummary {
            commands: self.commands.len() as u64,
            ..StateProfileSummary::default()
        };
        for command in &self.commands {
            summary.valid_tokens += u64::from(command.valid_tokens);
            summary.cache_hits += command.cache_hits;
            summary.cache_misses += command.cache_misses;
            summary.descriptor_hbm_read_bytes += command.descriptor_hbm_read_bytes;
            summary.parameter_hbm_read_bytes += command.parameter_hbm_read_bytes;
            summary.state_hbm_read_bytes += command.state_hbm_read_bytes;
            summary.state_hbm_write_bytes += command.state_hbm_write_bytes;
            summary.completion_hbm_write_bytes += command.completion_hbm_write_bytes;
            summary.projection_vram_read_bytes += command.projection_vram_read_bytes;
            summary.output_vram_write_bytes += command.output_vram_write_bytes;
            summary.projection_buffer_values += command.projection_buffer_values;
            summary.projection_fifo_capacity_values = summary
                .projection_fifo_capacity_values
                .max(command.projection_fifo_capacity_values);
            summary.projection_fifo_peak_values = summary
                .projection_fifo_peak_values
                .max(command.projection_fifo_peak_values);
            summary.projection_fifo_spill_values += command.projection_fifo_spill_values;
            summary.projection_fifo_backpressure_cycles +=
                command.projection_fifo_backpressure_cycles;
            summary.projection_write_packets += command.projection_write_packets;
            summary.projection_write_ideal_cycles += command.projection_write_ideal_cycles;
            summary.projection_write_service_cycles += command.projection_write_service_cycles;
            summary.projection_write_stall_cycles += command.projection_write_stall_cycles;
            summary.projection_read_packets += command.projection_read_packets;
            summary.projection_read_ideal_cycles += command.projection_read_ideal_cycles;
            summary.projection_read_service_cycles += command.projection_read_service_cycles;
            summary.projection_read_stall_cycles += command.projection_read_stall_cycles;
            summary.state_sram_read_values += command.state_sram_read_values;
            summary.state_sram_write_values += command.state_sram_write_values;
            summary.bank_packets += command.bank_packets;
            summary.bank_ideal_cycles += command.bank_ideal_cycles;
            summary.bank_service_cycles += command.bank_service_cycles;
            summary.bank_stall_cycles += command.bank_stall_cycles;
            summary.arithmetic_cycles += command.arithmetic_cycles;
            summary.recurrent_cycles += command.recurrent_cycles;
            summary.estimated_total_cycles += command.estimated_total_cycles;
        }
        let cache_accesses = summary.cache_hits + summary.cache_misses;
        summary.cache_hit_rate = if cache_accesses == 0 {
            0.0
        } else {
            summary.cache_hits as f64 / cache_accesses as f64
        };
        summary
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct StateProfileSummary {
    pub commands: u64,
    pub valid_tokens: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_hit_rate: f64,
    pub descriptor_hbm_read_bytes: u64,
    pub parameter_hbm_read_bytes: u64,
    pub state_hbm_read_bytes: u64,
    pub state_hbm_write_bytes: u64,
    pub completion_hbm_write_bytes: u64,
    pub projection_vram_read_bytes: u64,
    pub output_vram_write_bytes: u64,
    pub projection_buffer_values: u64,
    pub projection_fifo_capacity_values: u64,
    pub projection_fifo_peak_values: u64,
    pub projection_fifo_spill_values: u64,
    pub projection_fifo_backpressure_cycles: u64,
    pub projection_write_packets: u64,
    pub projection_write_ideal_cycles: u64,
    pub projection_write_service_cycles: u64,
    pub projection_write_stall_cycles: u64,
    pub projection_read_packets: u64,
    pub projection_read_ideal_cycles: u64,
    pub projection_read_service_cycles: u64,
    pub projection_read_stall_cycles: u64,
    pub state_sram_read_values: u64,
    pub state_sram_write_values: u64,
    pub bank_packets: u64,
    pub bank_ideal_cycles: u64,
    pub bank_service_cycles: u64,
    pub bank_stall_cycles: u64,
    pub arithmetic_cycles: u64,
    pub recurrent_cycles: u64,
    pub estimated_total_cycles: u64,
}

pub fn estimate(
    descriptor: &StateDescriptor,
    subop: StateSubop,
    config: StateTimingConfig,
) -> StateExecutionRecord {
    let state_resident = !descriptor.is_streaming();
    let state_payload_bytes = descriptor.resident_bytes();
    let mut state_hbm_read_bytes = 0;
    let mut state_hbm_write_bytes = 0;
    let mut parameter_hbm_read_bytes = 0;
    let mut projection_vram_read_bytes = 0;
    let mut output_vram_write_bytes = 0;
    let mut cache_hits = 0;
    let mut cache_misses = 0;
    let mut state_sram_read_values = 0;
    let mut state_sram_write_values = 0;
    let mut bank = BankStats::default();
    let mut arithmetic_cycles = 0;
    let tokens = u64::from(descriptor.valid_tokens) * u64::from(descriptor.batch_size);
    let state_elements =
        u64::from(descriptor.state_bytes) / descriptor.state_precision.element_bytes() as u64;
    let conv_elements = u64::from(descriptor.conv_state_bytes)
        / descriptor.conv_state_precision.element_bytes() as u64;
    let resident_values = state_elements + conv_elements;

    match subop {
        StateSubop::Preload => {
            state_hbm_read_bytes = state_payload_bytes;
            state_sram_write_values = resident_values;
            cache_misses = 1;
        }
        StateSubop::Reset => {
            if descriptor.is_streaming() {
                state_hbm_write_bytes = state_payload_bytes;
            } else {
                state_sram_write_values = resident_values;
            }
        }
        StateSubop::Commit => {
            state_hbm_write_bytes = state_payload_bytes;
            state_sram_read_values = resident_values;
        }
        StateSubop::Prefill | StateSubop::Step => {
            if state_resident {
                cache_hits = 1;
            } else {
                cache_misses = 1;
                state_hbm_read_bytes = state_payload_bytes;
                state_hbm_write_bytes = state_payload_bytes;
            }
            parameter_hbm_read_bytes = parameter_bytes(descriptor);
            projection_vram_read_bytes = tokens
                * storage_bytes(
                    u64::from(descriptor.input_token_stride),
                    descriptor.activation_precision,
                );
            output_vram_write_bytes = tokens
                * storage_bytes(
                    descriptor.output_elements(),
                    descriptor.activation_precision,
                );
            let (rows, columns, passes, fmas_per_element, extra_ops) = geometry(descriptor);
            state_sram_read_values = tokens * (passes * state_elements + conv_elements);
            state_sram_write_values = tokens * (state_elements + conv_elements);
            bank = bank_stats(descriptor, rows, columns, passes, tokens, config);
            let lanes = u64::from(config.head_lanes) * u64::from(config.fma_lanes_per_head_lane);
            let recurrence_ops = tokens * state_elements * fmas_per_element;
            let conv_ops = tokens * conv_elements;
            arithmetic_cycles = (recurrence_ops + conv_ops + tokens * extra_ops).div_ceil(lanes);
        }
        StateSubop::Evict | StateSubop::Fence => {}
    }

    let descriptor_hbm_read_bytes = if subop == StateSubop::Fence {
        0
    } else {
        wire::DESCRIPTOR_SIZE as u64
    };
    let completion_hbm_write_bytes = if descriptor.writes_completion() {
        16
    } else {
        0
    };
    let recurrent_cycles = arithmetic_cycles.max(bank.service_cycles);
    let fixed_hbm_bytes =
        descriptor_hbm_read_bytes + parameter_hbm_read_bytes + completion_hbm_write_bytes;
    let streamed_state_hbm_bytes = state_hbm_read_bytes + state_hbm_write_bytes;
    let total_hbm_bytes = fixed_hbm_bytes + streamed_state_hbm_bytes;
    let estimated_hbm_cycles = total_hbm_bytes.div_ceil(u64::from(config.hbm_bytes_per_cycle));
    let estimated_total_cycles = if matches!(subop, StateSubop::Prefill | StateSubop::Step)
        && !state_resident
        && config.head_tile_slots >= 2
    {
        fixed_hbm_bytes.div_ceil(u64::from(config.hbm_bytes_per_cycle))
            + recurrent_cycles
                .max(streamed_state_hbm_bytes.div_ceil(u64::from(config.hbm_bytes_per_cycle)))
    } else {
        recurrent_cycles + estimated_hbm_cycles
    };

    StateExecutionRecord {
        subop: format!("{subop:?}").to_ascii_lowercase(),
        algorithm: match descriptor.algorithm {
            StateAlgorithm::Mamba2 => "mamba2",
            StateAlgorithm::Kda => "kda",
        }
        .to_string(),
        context_id: descriptor.identity.context_id,
        request_id: descriptor.identity.request_id,
        layer_id: descriptor.identity.layer_id,
        state_id: descriptor.identity.state_id,
        valid_tokens: if matches!(subop, StateSubop::Prefill | StateSubop::Step) {
            descriptor.valid_tokens
        } else {
            0
        },
        state_resident,
        cache_hits,
        cache_misses,
        descriptor_hbm_read_bytes,
        parameter_hbm_read_bytes,
        state_hbm_read_bytes,
        state_hbm_write_bytes,
        completion_hbm_write_bytes,
        projection_vram_read_bytes,
        output_vram_write_bytes,
        projection_buffer_values: 0,
        projection_fifo_capacity_values: 0,
        projection_fifo_peak_values: 0,
        projection_fifo_spill_values: 0,
        projection_fifo_backpressure_cycles: 0,
        projection_write_packets: 0,
        projection_write_ideal_cycles: 0,
        projection_write_service_cycles: 0,
        projection_write_stall_cycles: 0,
        projection_read_packets: 0,
        projection_read_ideal_cycles: 0,
        projection_read_service_cycles: 0,
        projection_read_stall_cycles: 0,
        state_sram_read_values,
        state_sram_write_values,
        bank_packets: bank.packets,
        bank_ideal_cycles: bank.ideal_cycles,
        bank_service_cycles: bank.service_cycles,
        bank_stall_cycles: bank.service_cycles.saturating_sub(bank.ideal_cycles),
        arithmetic_cycles,
        recurrent_cycles,
        estimated_hbm_cycles,
        estimated_total_cycles,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct BankStats {
    packets: u64,
    ideal_cycles: u64,
    service_cycles: u64,
}

fn bank_stats(
    descriptor: &StateDescriptor,
    rows: u32,
    columns: u32,
    passes: u64,
    tokens: u64,
    config: StateTimingConfig,
) -> BankStats {
    let mut one_head = BankStats::default();
    for row_start in (0..rows).step_by(config.row_lanes as usize) {
        for column_start in (0..columns).step_by(config.column_lanes as usize) {
            let mut counts = vec![0u32; config.banks_per_head_lane as usize];
            let mut values = 0u64;
            for row in row_start..(row_start + config.row_lanes).min(rows) {
                for column in column_start..(column_start + config.column_lanes).min(columns) {
                    let bank = state_bank(row, column, columns, config);
                    counts[bank as usize] += 1;
                    values += 1;
                }
            }
            one_head.packets += 1;
            one_head.ideal_cycles += values
                .div_ceil(u64::from(config.banks_per_head_lane) * u64::from(config.ports_per_bank));
            one_head.service_cycles += counts
                .into_iter()
                .map(|count| u64::from(count).div_ceil(u64::from(config.ports_per_bank)))
                .max()
                .unwrap_or(0);
        }
    }
    let head_waves = u64::from(descriptor.num_heads).div_ceil(u64::from(config.head_lanes));
    let factor = head_waves * passes * tokens;
    BankStats {
        packets: one_head.packets * factor,
        ideal_cycles: one_head.ideal_cycles * factor,
        service_cycles: one_head.service_cycles * factor,
    }
}

fn state_bank(row: u32, column: u32, columns: u32, config: StateTimingConfig) -> u32 {
    match config.layout {
        StateSramLayout::RowMajor => (row * columns + column) % config.banks_per_head_lane,
        StateSramLayout::DualAxisCyclic => {
            let local_row = row % config.row_lanes;
            let local_column = column % config.column_lanes;
            (local_row * config.column_lanes + local_column) % config.banks_per_head_lane
        }
    }
}

fn geometry(descriptor: &StateDescriptor) -> (u32, u32, u64, u64, u64) {
    match &descriptor.payload {
        StatePayload::Mamba2(payload) => (
            u32::from(payload.head_dim),
            u32::from(payload.state_dim),
            1,
            2,
            u64::from(descriptor.num_heads),
        ),
        StatePayload::Kda(payload) => (
            u32::from(payload.value_dim),
            u32::from(payload.key_dim),
            2,
            3,
            4 * u64::from(descriptor.num_heads) * u64::from(payload.key_dim),
        ),
    }
}

fn parameter_bytes(descriptor: &StateDescriptor) -> u64 {
    let mut tensors: Vec<(u64, u64)> = Vec::new();
    match &descriptor.payload {
        StatePayload::Mamba2(payload) => {
            let heads = u64::from(descriptor.num_heads);
            let channels = heads * u64::from(payload.head_dim)
                + 2 * u64::from(payload.groups) * u64::from(payload.state_dim);
            tensors.push((
                channels * u64::from(payload.conv_kernel),
                u64::from(payload.conv_kernel),
            ));
            if payload.conv_bias_addr != 0 {
                tensors.push((channels, channels));
            }
            tensors.extend([(heads, heads), (heads, heads), (heads, heads)]);
        }
        StatePayload::Kda(payload) => {
            let heads = u64::from(descriptor.num_heads);
            let key = heads * u64::from(payload.key_dim);
            let value = heads * u64::from(payload.value_dim);
            let kernel = u64::from(payload.conv_kernel);
            tensors.extend([
                (key * kernel, kernel),
                (key * kernel, kernel),
                (value * kernel, kernel),
            ]);
            for (address, elements) in [
                (payload.q_conv_bias_addr, key),
                (payload.k_conv_bias_addr, key),
                (payload.v_conv_bias_addr, value),
            ] {
                if address != 0 {
                    tensors.push((elements, elements));
                }
            }
            tensors.push((heads, heads));
            tensors.push((key, u64::from(payload.key_dim)));
        }
    }
    let values = tensors.iter().map(|(elements, _)| elements).sum::<u64>()
        * descriptor.parameter_precision.element_bytes() as u64;
    let scales = if descriptor.parameter_precision == StatePrecision::Mx8B128 {
        tensors
            .iter()
            .map(|(elements, inner)| (elements / inner) * inner.div_ceil(MX_BLOCK))
            .sum()
    } else {
        0
    };
    values + scales
}

fn storage_bytes(elements: u64, precision: StatePrecision) -> u64 {
    elements * precision.element_bytes() as u64
        + if precision == StatePrecision::Mx8B128 {
            elements.div_ceil(MX_BLOCK)
        } else {
            0
        }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_engine::descriptor::StateDescriptor;

    fn decode_hex(value: &str) -> Vec<u8> {
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|digits| u8::from_str_radix(std::str::from_utf8(digits).unwrap(), 16).unwrap())
            .collect()
    }

    fn golden(name: &str) -> StateDescriptor {
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../../testdata/x_state_v2_golden.json")).unwrap();
        let bytes = decode_hex(data["descriptors"][name]["hex"].as_str().unwrap());
        StateDescriptor::parse(&bytes).unwrap()
    }

    #[test]
    fn cyclic_layout_removes_candidate_state_bank_conflicts() {
        for descriptor in [golden("mamba2_real"), golden("kda_real")] {
            let row = estimate(&descriptor, StateSubop::Step, StateTimingConfig::default());
            let cyclic = estimate(
                &descriptor,
                StateSubop::Step,
                StateTimingConfig {
                    layout: StateSramLayout::DualAxisCyclic,
                    ..StateTimingConfig::default()
                },
            );
            assert!(row.bank_stall_cycles > 0);
            assert_eq!(cyclic.bank_stall_cycles, 0);
            assert!(cyclic.recurrent_cycles <= row.recurrent_cycles);
        }
    }

    #[test]
    fn streaming_and_resident_have_same_work_but_different_state_hbm() {
        let streaming = golden("mamba2_real");
        let mut resident = streaming.clone();
        resident.state_sram_offset = 0;
        let streamed = estimate(&streaming, StateSubop::Step, StateTimingConfig::default());
        let cached = estimate(&resident, StateSubop::Step, StateTimingConfig::default());
        assert_eq!(streamed.arithmetic_cycles, cached.arithmetic_cycles);
        assert!(streamed.state_hbm_read_bytes > 0);
        assert_eq!(cached.state_hbm_read_bytes, 0);
        assert_eq!(streamed.cache_misses, 1);
        assert_eq!(cached.cache_hits, 1);
    }

    #[test]
    fn streaming_overlap_only_hides_state_dma() {
        let descriptor = golden("mamba2_real");
        let config = StateTimingConfig::default();
        let timing = estimate(&descriptor, StateSubop::Step, config);
        let fixed_bytes = timing.descriptor_hbm_read_bytes
            + timing.parameter_hbm_read_bytes
            + timing.completion_hbm_write_bytes;
        let state_bytes = timing.state_hbm_read_bytes + timing.state_hbm_write_bytes;
        let expected = fixed_bytes.div_ceil(u64::from(config.hbm_bytes_per_cycle))
            + timing
                .recurrent_cycles
                .max(state_bytes.div_ceil(u64::from(config.hbm_bytes_per_cycle)));
        assert_eq!(timing.estimated_total_cycles, expected);
        assert!(timing.estimated_total_cycles > timing.recurrent_cycles);
    }

    #[test]
    fn lifecycle_commands_do_not_count_workload_tokens() {
        let descriptor = golden("mamba2_real");
        let timing = estimate(
            &descriptor,
            StateSubop::Preload,
            StateTimingConfig::default(),
        );
        assert_eq!(timing.valid_tokens, 0);
    }

    #[test]
    fn profile_summary_aggregates_traffic_and_compute_tokens() {
        let descriptor = golden("mamba2_real");
        let mut profile = StateEngineProfile::new(StateTimingConfig::default());
        profile.record(estimate(
            &descriptor,
            StateSubop::Preload,
            StateTimingConfig::default(),
        ));
        profile.record(estimate(
            &descriptor,
            StateSubop::Step,
            StateTimingConfig::default(),
        ));
        let summary = profile.summary();
        assert_eq!(summary.commands, 2);
        assert_eq!(summary.valid_tokens, 1);
        assert_eq!(summary.cache_misses, 2);
        assert!(summary.descriptor_hbm_read_bytes > 0);
        assert!(summary.state_hbm_read_bytes > 0);
        assert!(summary.state_sram_read_values > 0);
        assert!(summary.state_sram_write_values > 0);
        assert!(summary.recurrent_cycles > 0);
    }
}
