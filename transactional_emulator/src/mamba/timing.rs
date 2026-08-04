//! Descriptor-derived event timing model for the Mamba State Engine.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::generated_contract as contract;

use super::{MambaDescriptor, MambaDescriptorEnvelope, MambaError};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MambaTimingConfig {
    pub(crate) schema_version: u32,
    pub(crate) source: String,
    pub(crate) hardware_profile: String,
    pub(crate) model_profile: String,
    pub(crate) calibrated: bool,
    pub(crate) hbm_bytes_per_cycle: u64,
    pub(crate) matrix_macs_per_cycle: u64,
    pub(crate) conv_macs_per_cycle: u64,
    pub(crate) exp_values_per_cycle: u64,
    pub(crate) exp_pipeline_latency: u64,
    pub(crate) state_fmas_per_cycle: u64,
    pub(crate) elementwise_values_per_cycle: u64,
}

impl MambaTimingConfig {
    pub(crate) fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let config: Self = toml::from_str(&text)
            .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err(format!(
                "unsupported Mamba timing schema version {}",
                self.schema_version
            ));
        }
        if self.source.trim().is_empty() {
            return Err("Mamba timing source must not be empty".to_string());
        }
        if self.hardware_profile.trim().is_empty() || self.model_profile.trim().is_empty() {
            return Err("Mamba timing profile names must not be empty".to_string());
        }
        for (name, value) in [
            ("hbm_bytes_per_cycle", self.hbm_bytes_per_cycle),
            ("matrix_macs_per_cycle", self.matrix_macs_per_cycle),
            ("conv_macs_per_cycle", self.conv_macs_per_cycle),
            ("exp_values_per_cycle", self.exp_values_per_cycle),
            ("state_fmas_per_cycle", self.state_fmas_per_cycle),
            (
                "elementwise_values_per_cycle",
                self.elementwise_values_per_cycle,
            ),
        ] {
            if value == 0 {
                return Err(format!("{name} must be positive"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct MambaStageEvent {
    pub(crate) name: &'static str,
    pub(crate) resource: &'static str,
    pub(crate) start_cycle: u64,
    pub(crate) end_cycle: u64,
    pub(crate) work_units: u64,
    pub(crate) unit: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct MambaTimingRecord {
    pub(crate) context_id: u32,
    pub(crate) layer_id: u32,
    pub(crate) queue_id: u8,
    pub(crate) subop: &'static str,
    pub(crate) status: u32,
    pub(crate) issue_cycle: u64,
    pub(crate) command_ready_cycle: u64,
    pub(crate) completion_cycle: u64,
    pub(crate) elapsed_cycles: u64,
    pub(crate) queue_depth_after_issue: usize,
    pub(crate) queue_wait_cycles: u64,
    pub(crate) dependency_wait_cycles: u64,
    pub(crate) state_wait_cycles: u64,
    pub(crate) resource_wait_cycles: u64,
    pub(crate) hbm_bytes_read: u64,
    pub(crate) hbm_bytes_written: u64,
    pub(crate) matrix_macs: u64,
    pub(crate) conv_macs: u64,
    pub(crate) state_fmas: u64,
    pub(crate) exp_values: u64,
    pub(crate) external_scratch_bytes: u64,
    pub(crate) stages: Vec<MambaStageEvent>,
}

#[derive(Clone, Debug, Serialize)]
struct MambaResourceSummary {
    resource: &'static str,
    busy_cycles: u64,
    utilization: f64,
}

#[derive(Clone, Debug, Serialize)]
struct MambaTimingSummary {
    command_count: usize,
    first_issue_cycle: u64,
    last_completion_cycle: u64,
    total_span_cycles: u64,
    max_queue_depth_after_issue: usize,
    resources: Vec<MambaResourceSummary>,
}

#[derive(Clone, Debug, Serialize)]
struct MambaTimingReport<'a> {
    contract_sha256: &'static str,
    config: &'a MambaTimingConfig,
    summary: MambaTimingSummary,
    commands: &'a [MambaTimingRecord],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MambaWorkload {
    pub(crate) context_id: u32,
    pub(crate) layer_id: u32,
    pub(crate) subop: contract::MambaSubop,
    pub(crate) dependency_event: u32,
    pub(crate) completion_event: u32,
    pub(crate) writes_completion: bool,
    pub(crate) continues_state: bool,
    pub(crate) batch: u64,
    pub(crate) sequence: u64,
    pub(crate) d_model: u64,
    pub(crate) d_inner: u64,
    pub(crate) heads: u64,
    pub(crate) head_dim: u64,
    pub(crate) state_dim: u64,
    pub(crate) conv_channels: u64,
    pub(crate) conv_kernel: u64,
    pub(crate) projection: u64,
    pub(crate) output_projection: u64,
    pub(crate) element_bytes: u64,
    pub(crate) state_request_stride: u64,
    pub(crate) conv_request_stride: u64,
    pub(crate) optional_bias_elements: u64,
}

fn checked_product(values: &[u64]) -> Result<u64, MambaError> {
    values.iter().try_fold(1u64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(MambaError::UnsupportedProfile(
                "timing workload overflows u64",
            ))
    })
}

fn checked_sum(values: &[u64]) -> Result<u64, MambaError> {
    values.iter().try_fold(0u64, |sum, value| {
        sum.checked_add(*value)
            .ok_or(MambaError::UnsupportedProfile(
                "timing workload overflows u64",
            ))
    })
}

impl MambaWorkload {
    pub(crate) fn from_descriptor(
        descriptor: &MambaDescriptor,
        subop: contract::MambaSubop,
    ) -> Result<Self, MambaError> {
        let base = Self {
            context_id: descriptor.context_id,
            layer_id: descriptor.layer_id,
            subop,
            dependency_event: descriptor.dependency_event,
            completion_event: descriptor.completion_event,
            writes_completion: descriptor.write_completion(),
            continues_state: descriptor.continue_state(),
            batch: 0,
            sequence: 0,
            d_model: 0,
            d_inner: 0,
            heads: 0,
            head_dim: 0,
            state_dim: 0,
            conv_channels: 0,
            conv_kernel: 0,
            projection: 0,
            output_projection: 0,
            element_bytes: descriptor.precision.element_bytes(),
            state_request_stride: 0,
            conv_request_stride: 0,
            optional_bias_elements: 0,
        };
        if subop == contract::MambaSubop::Wait {
            return Ok(base);
        }

        let state_workload = Self {
            batch: descriptor.batch_size as u64,
            d_inner: descriptor.d_inner as u64,
            heads: descriptor.num_heads as u64,
            head_dim: descriptor.head_dim as u64,
            state_dim: descriptor.state_dim as u64,
            conv_channels: descriptor.conv_channels()?,
            conv_kernel: descriptor.conv_kernel as u64,
            state_request_stride: descriptor.state_request_stride as u64,
            conv_request_stride: descriptor.conv_request_stride as u64,
            ..base
        };
        if subop == contract::MambaSubop::StateReset {
            return Ok(state_workload);
        }

        let optional_bias_elements = checked_sum(&[
            u64::from(descriptor.in_proj_bias_addr != 0) * descriptor.projection_size()?,
            u64::from(descriptor.conv_bias_addr != 0) * descriptor.conv_channels()?,
            u64::from(descriptor.out_proj_bias_addr != 0) * descriptor.d_model as u64,
        ])?;
        Ok(Self {
            sequence: descriptor.sequence_length as u64,
            d_model: descriptor.d_model as u64,
            projection: descriptor.projection_size()?,
            output_projection: descriptor.output_projection_size()?,
            optional_bias_elements,
            ..state_workload
        })
    }

    fn uses_state(self) -> bool {
        self.subop != contract::MambaSubop::Wait
    }
}

pub(crate) struct MambaTimingModel {
    config: MambaTimingConfig,
    resource_ready: HashMap<&'static str, u64>,
    queue_ready: [u64; 16],
    queue_completions: [Vec<u64>; 16],
    state_ready: HashMap<(u32, u32), u64>,
    event_ready: HashMap<u32, u64>,
    records: Vec<MambaTimingRecord>,
    resource_wait_accumulator: u64,
}

impl MambaTimingModel {
    pub(crate) fn new(config: MambaTimingConfig) -> Self {
        Self {
            config,
            resource_ready: HashMap::new(),
            queue_ready: [0; 16],
            queue_completions: std::array::from_fn(|_| Vec::new()),
            state_ready: HashMap::new(),
            event_ready: HashMap::new(),
            records: Vec::new(),
            resource_wait_accumulator: 0,
        }
    }

    fn cycles(work: u64, throughput: u64) -> u64 {
        work.div_ceil(throughput)
    }

    fn reserve(
        &mut self,
        stages: &mut Vec<MambaStageEvent>,
        name: &'static str,
        resource: &'static str,
        earliest: u64,
        cycles: u64,
        work_units: u64,
        unit: &'static str,
    ) -> u64 {
        let available = self.resource_ready.get(resource).copied().unwrap_or(0);
        let start = earliest.max(available);
        let end = start.saturating_add(cycles);
        self.resource_wait_accumulator = self
            .resource_wait_accumulator
            .saturating_add(start.saturating_sub(earliest));
        self.resource_ready.insert(resource, end);
        stages.push(MambaStageEvent {
            name,
            resource,
            start_cycle: start,
            end_cycle: end,
            work_units,
            unit,
        });
        end
    }

    fn subop_name(subop: Option<contract::MambaSubop>) -> &'static str {
        match subop {
            Some(contract::MambaSubop::Prefill) => "PREFILL",
            Some(contract::MambaSubop::Step) => "STEP",
            Some(contract::MambaSubop::StateReset) => "STATE_RESET",
            Some(contract::MambaSubop::Wait) => "WAIT",
            None => "INVALID",
        }
    }

    pub(crate) fn schedule_terminal(
        &mut self,
        envelope: MambaDescriptorEnvelope,
        subop: Option<contract::MambaSubop>,
        writes_completion: bool,
        status: u32,
        queue_id: u8,
        issue_cycle: u64,
    ) -> Result<MambaTimingRecord, MambaError> {
        let queue_depth_after_issue = {
            let queue_completions = &mut self.queue_completions[queue_id as usize];
            queue_completions.retain(|completion| *completion > issue_cycle);
            queue_completions.len() + 1
        };
        let queue_ready = self.queue_ready[queue_id as usize];
        if envelope.completion_event != contract::MAMBA_NO_EVENT
            && self.event_ready.contains_key(&envelope.completion_event)
        {
            return Err(MambaError::StateHazard(
                "completion event is already active",
            ));
        }
        let dependency_ready =
            if envelope.dependency_event == contract::MAMBA_NO_EVENT {
                issue_cycle
            } else {
                self.event_ready.remove(&envelope.dependency_event).ok_or(
                    MambaError::StateHazard("dependency event has no active producer"),
                )?
            };
        let command_ready = issue_cycle.max(queue_ready).max(dependency_ready);
        let queue_wait_cycles = queue_ready.saturating_sub(issue_cycle);
        let dependency_wait_cycles = dependency_ready.saturating_sub(issue_cycle);
        let mut stages = Vec::new();
        self.resource_wait_accumulator = 0;
        let descriptor_end = self.reserve(
            &mut stages,
            "descriptor_dma",
            "hbm",
            command_ready,
            Self::cycles(
                contract::MAMBA_DESCRIPTOR_BYTES as u64,
                self.config.hbm_bytes_per_cycle,
            ),
            contract::MAMBA_DESCRIPTOR_BYTES as u64,
            "bytes",
        );
        let completion_cycle = if writes_completion {
            self.reserve(
                &mut stages,
                "completion_write",
                "hbm",
                descriptor_end,
                Self::cycles(
                    contract::MAMBA_COMPLETION_BYTES as u64,
                    self.config.hbm_bytes_per_cycle,
                ),
                contract::MAMBA_COMPLETION_BYTES as u64,
                "bytes",
            )
        } else {
            descriptor_end
        };

        self.queue_ready[queue_id as usize] = completion_cycle;
        self.queue_completions[queue_id as usize].push(completion_cycle);
        if envelope.completion_event != contract::MAMBA_NO_EVENT {
            self.event_ready
                .insert(envelope.completion_event, completion_cycle);
        }
        let record = MambaTimingRecord {
            context_id: envelope.context_id,
            layer_id: envelope.layer_id,
            queue_id,
            subop: Self::subop_name(subop),
            status,
            issue_cycle,
            command_ready_cycle: command_ready,
            completion_cycle,
            elapsed_cycles: descriptor_end.saturating_sub(issue_cycle),
            queue_depth_after_issue,
            queue_wait_cycles,
            dependency_wait_cycles,
            state_wait_cycles: 0,
            resource_wait_cycles: self.resource_wait_accumulator,
            hbm_bytes_read: contract::MAMBA_DESCRIPTOR_BYTES as u64,
            hbm_bytes_written: u64::from(writes_completion)
                * contract::MAMBA_COMPLETION_BYTES as u64,
            matrix_macs: 0,
            conv_macs: 0,
            state_fmas: 0,
            exp_values: 0,
            external_scratch_bytes: 0,
            stages,
        };
        self.records.push(record.clone());
        Ok(record)
    }

    pub(crate) fn schedule(
        &mut self,
        workload: MambaWorkload,
        queue_id: u8,
        issue_cycle: u64,
    ) -> Result<MambaTimingRecord, MambaError> {
        let queue_depth_after_issue = {
            let queue_completions = &mut self.queue_completions[queue_id as usize];
            queue_completions.retain(|completion| *completion > issue_cycle);
            queue_completions.len() + 1
        };
        let queue_ready = self.queue_ready[queue_id as usize];
        if workload.completion_event != contract::MAMBA_NO_EVENT
            && self.event_ready.contains_key(&workload.completion_event)
        {
            return Err(MambaError::StateHazard(
                "completion event is already active",
            ));
        }
        let dependency_ready =
            if workload.dependency_event == contract::MAMBA_NO_EVENT {
                issue_cycle
            } else {
                self.event_ready.remove(&workload.dependency_event).ok_or(
                    MambaError::StateHazard("dependency event has no active producer"),
                )?
            };
        let state_ready = if workload.uses_state() {
            self.state_ready
                .get(&(workload.context_id, workload.layer_id))
                .copied()
                .unwrap_or(issue_cycle)
        } else {
            issue_cycle
        };
        let command_ready = issue_cycle
            .max(queue_ready)
            .max(dependency_ready)
            .max(state_ready);
        let queue_wait_cycles = queue_ready.saturating_sub(issue_cycle);
        let dependency_wait_cycles = dependency_ready.saturating_sub(issue_cycle);
        let state_wait_cycles = state_ready.saturating_sub(issue_cycle);
        let mut stages = Vec::new();
        self.resource_wait_accumulator = 0;

        let descriptor_cycles = Self::cycles(
            contract::MAMBA_DESCRIPTOR_BYTES as u64,
            self.config.hbm_bytes_per_cycle,
        );
        let descriptor_end = self.reserve(
            &mut stages,
            "descriptor_dma",
            "hbm",
            command_ready,
            descriptor_cycles,
            contract::MAMBA_DESCRIPTOR_BYTES as u64,
            "bytes",
        );

        let tokens = checked_product(&[workload.batch, workload.sequence])?;
        let state_bytes = checked_product(&[
            workload.batch,
            checked_sum(&[workload.state_request_stride, workload.conv_request_stride])?,
        ])?;
        let mut hbm_bytes_read = contract::MAMBA_DESCRIPTOR_BYTES as u64;
        let mut hbm_bytes_written =
            u64::from(workload.writes_completion) * contract::MAMBA_COMPLETION_BYTES as u64;
        let mut matrix_macs = 0u64;
        let mut conv_macs = 0u64;
        let mut state_fmas = 0u64;
        let mut exp_values = 0u64;
        let completion_earliest;

        match workload.subop {
            contract::MambaSubop::Wait => {
                completion_earliest = descriptor_end;
            }
            contract::MambaSubop::StateReset => {
                hbm_bytes_written = hbm_bytes_written.checked_add(state_bytes).ok_or(
                    MambaError::UnsupportedProfile("timing byte count overflows u64"),
                )?;
                let state_write_end = self.reserve(
                    &mut stages,
                    "state_reset_write",
                    "hbm",
                    descriptor_end,
                    Self::cycles(state_bytes, self.config.hbm_bytes_per_cycle),
                    state_bytes,
                    "bytes",
                );
                completion_earliest = state_write_end;
            }
            contract::MambaSubop::Prefill | contract::MambaSubop::Step => {
                let input_elements = checked_product(&[tokens, workload.d_model])?;
                let weight_elements = checked_sum(&[
                    checked_product(&[workload.projection, workload.d_model])?,
                    checked_product(&[workload.conv_channels, workload.conv_kernel])?,
                    workload.heads * 3,
                    workload.d_inner,
                    checked_product(&[workload.d_model, workload.output_projection])?,
                    workload.optional_bias_elements,
                ])?;
                let tensor_read_bytes = checked_product(&[
                    checked_sum(&[input_elements, weight_elements])?,
                    workload.element_bytes,
                ])?
                .checked_add(if workload.continues_state {
                    state_bytes
                } else {
                    0
                })
                .ok_or(MambaError::UnsupportedProfile(
                    "timing byte count overflows u64",
                ))?;
                hbm_bytes_read = hbm_bytes_read.checked_add(tensor_read_bytes).ok_or(
                    MambaError::UnsupportedProfile("timing byte count overflows u64"),
                )?;
                let output_bytes =
                    checked_product(&[tokens, workload.d_model, workload.element_bytes])?;
                let writeback_bytes =
                    output_bytes
                        .checked_add(state_bytes)
                        .ok_or(MambaError::UnsupportedProfile(
                            "timing byte count overflows u64",
                        ))?;
                hbm_bytes_written = hbm_bytes_written.checked_add(writeback_bytes).ok_or(
                    MambaError::UnsupportedProfile("timing byte count overflows u64"),
                )?;
                matrix_macs = checked_sum(&[
                    checked_product(&[tokens, workload.projection, workload.d_model])?,
                    checked_product(&[tokens, workload.d_model, workload.output_projection])?,
                ])?;
                conv_macs =
                    checked_product(&[tokens, workload.conv_channels, workload.conv_kernel])?;
                state_fmas = checked_product(&[
                    tokens,
                    workload.heads,
                    workload.head_dim,
                    workload.state_dim,
                    2,
                ])?;
                exp_values = checked_sum(&[
                    checked_product(&[tokens, workload.heads, 2])?,
                    workload.heads,
                ])?;

                let tensor_read_end = self.reserve(
                    &mut stages,
                    "tensor_state_read",
                    "hbm",
                    descriptor_end,
                    Self::cycles(tensor_read_bytes, self.config.hbm_bytes_per_cycle),
                    tensor_read_bytes,
                    "bytes",
                );
                let in_projection_macs =
                    checked_product(&[tokens, workload.projection, workload.d_model])?;
                let in_projection_end = self.reserve(
                    &mut stages,
                    "input_projection",
                    "matrix",
                    tensor_read_end,
                    Self::cycles(in_projection_macs, self.config.matrix_macs_per_cycle),
                    in_projection_macs,
                    "macs",
                );
                let conv_end = self.reserve(
                    &mut stages,
                    "depthwise_conv",
                    "conv",
                    in_projection_end,
                    Self::cycles(conv_macs, self.config.conv_macs_per_cycle),
                    conv_macs,
                    "macs",
                );
                let dt_end = self.reserve(
                    &mut stages,
                    "dt_a_exp",
                    "exp",
                    in_projection_end,
                    Self::cycles(exp_values, self.config.exp_values_per_cycle)
                        + self.config.exp_pipeline_latency,
                    exp_values,
                    "values",
                );
                let scan_ready = conv_end.max(dt_end);
                let scan_end = self.reserve(
                    &mut stages,
                    "selective_state_scan",
                    "state",
                    scan_ready,
                    Self::cycles(state_fmas, self.config.state_fmas_per_cycle),
                    state_fmas,
                    "fmas",
                );
                let norm_values = checked_product(&[tokens, workload.d_inner, 8])?;
                let norm_end = self.reserve(
                    &mut stages,
                    "gate_group_rmsnorm",
                    "elementwise",
                    scan_end,
                    Self::cycles(norm_values, self.config.elementwise_values_per_cycle),
                    norm_values,
                    "values",
                );
                let out_projection_macs =
                    checked_product(&[tokens, workload.d_model, workload.output_projection])?;
                let out_projection_end = self.reserve(
                    &mut stages,
                    "output_projection",
                    "matrix",
                    norm_end,
                    Self::cycles(out_projection_macs, self.config.matrix_macs_per_cycle),
                    out_projection_macs,
                    "macs",
                );
                completion_earliest = self.reserve(
                    &mut stages,
                    "output_state_write",
                    "hbm",
                    out_projection_end,
                    Self::cycles(writeback_bytes, self.config.hbm_bytes_per_cycle),
                    writeback_bytes,
                    "bytes",
                );
            }
        }

        let completion_cycle = if workload.writes_completion {
            self.reserve(
                &mut stages,
                "completion_write",
                "hbm",
                completion_earliest,
                Self::cycles(
                    contract::MAMBA_COMPLETION_BYTES as u64,
                    self.config.hbm_bytes_per_cycle,
                ),
                contract::MAMBA_COMPLETION_BYTES as u64,
                "bytes",
            )
        } else {
            completion_earliest
        };
        self.queue_ready[queue_id as usize] = completion_cycle;
        self.queue_completions[queue_id as usize].push(completion_cycle);
        if workload.uses_state() {
            self.state_ready
                .insert((workload.context_id, workload.layer_id), completion_cycle);
        }
        if workload.completion_event != contract::MAMBA_NO_EVENT {
            self.event_ready
                .insert(workload.completion_event, completion_cycle);
        }
        let record = MambaTimingRecord {
            context_id: workload.context_id,
            layer_id: workload.layer_id,
            queue_id,
            subop: Self::subop_name(Some(workload.subop)),
            status: contract::MAMBA_STATUS_SUCCESS,
            issue_cycle,
            command_ready_cycle: command_ready,
            completion_cycle,
            elapsed_cycles: completion_earliest.saturating_sub(issue_cycle),
            queue_depth_after_issue,
            queue_wait_cycles,
            dependency_wait_cycles,
            state_wait_cycles,
            resource_wait_cycles: self.resource_wait_accumulator,
            hbm_bytes_read,
            hbm_bytes_written,
            matrix_macs,
            conv_macs,
            state_fmas,
            exp_values,
            external_scratch_bytes: 0,
            stages,
        };
        self.records.push(record.clone());
        Ok(record)
    }

    pub(crate) fn write_json(&self, path: &Path) -> Result<(), String> {
        let first_issue_cycle = self
            .records
            .iter()
            .map(|record| record.issue_cycle)
            .min()
            .unwrap_or(0);
        let last_completion_cycle = self
            .records
            .iter()
            .map(|record| record.completion_cycle)
            .max()
            .unwrap_or(first_issue_cycle);
        let total_span_cycles = last_completion_cycle.saturating_sub(first_issue_cycle);
        let mut busy_by_resource: HashMap<&'static str, u64> = HashMap::new();
        for stage in self.records.iter().flat_map(|record| &record.stages) {
            *busy_by_resource.entry(stage.resource).or_default() +=
                stage.end_cycle.saturating_sub(stage.start_cycle);
        }
        let mut resources: Vec<_> = busy_by_resource
            .into_iter()
            .map(|(resource, busy_cycles)| MambaResourceSummary {
                resource,
                busy_cycles,
                utilization: if total_span_cycles == 0 {
                    0.0
                } else {
                    busy_cycles as f64 / total_span_cycles as f64
                },
            })
            .collect();
        resources.sort_unstable_by_key(|summary| summary.resource);
        let summary = MambaTimingSummary {
            command_count: self.records.len(),
            first_issue_cycle,
            last_completion_cycle,
            total_span_cycles,
            max_queue_depth_after_issue: self
                .records
                .iter()
                .map(|record| record.queue_depth_after_issue)
                .max()
                .unwrap_or(0),
            resources,
        };
        let report = MambaTimingReport {
            contract_sha256: contract::CONTRACT_SHA256,
            config: &self.config,
            summary,
            commands: &self.records,
        };
        let json = serde_json::to_string_pretty(&report)
            .map_err(|error| format!("failed to serialize Mamba timing profile: {error}"))?;
        std::fs::write(path, json + "\n")
            .map_err(|error| format!("failed to write {}: {error}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> MambaDescriptor {
        MambaDescriptor {
            flags: 0,
            context_id: 7,
            sequence_length: 1,
            batch_size: 1,
            d_model: 16,
            d_inner: 16,
            num_heads: 2,
            head_dim: 8,
            state_dim: 16,
            groups: 1,
            chunk_size: 8,
            conv_kernel: 4,
            precision: contract::MambaPrecisionPolicy::Fp32Reference,
            rms_norm_eps: 1e-5,
            input_addr: 0,
            output_addr: 0,
            in_proj_weight_addr: 0,
            in_proj_bias_addr: 0,
            conv_weight_addr: 0,
            conv_bias_addr: 0,
            a_log_addr: 0,
            d_skip_addr: 0,
            norm_weight_addr: 0,
            out_proj_weight_addr: 0,
            out_proj_bias_addr: 0,
            ssm_state_addr: 0,
            conv_state_addr: 0,
            scratch_addr: 0,
            completion_addr: 0,
            dt_bias_addr: 0,
            input_batch_stride: 0,
            input_token_stride: 0,
            output_batch_stride: 0,
            output_token_stride: 0,
            state_request_stride: 1024,
            state_head_stride: 512,
            conv_request_stride: 1280,
            scratch_bytes: 0,
            dependency_event: contract::MAMBA_NO_EVENT,
            completion_event: contract::MAMBA_NO_EVENT,
            layer_id: 3,
            dt_min: 0.0,
            dt_max: f32::INFINITY,
            d_mlp: 0,
        }
    }

    fn config() -> MambaTimingConfig {
        MambaTimingConfig {
            schema_version: 1,
            source: "unit-test".to_string(),
            hardware_profile: "unit-test-hardware".to_string(),
            model_profile: "mamba_rtl_smoke".to_string(),
            calibrated: false,
            hbm_bytes_per_cycle: 64,
            matrix_macs_per_cycle: 256,
            conv_macs_per_cycle: 32,
            exp_values_per_cycle: 4,
            exp_pipeline_latency: 3,
            state_fmas_per_cycle: 64,
            elementwise_values_per_cycle: 32,
        }
    }

    fn workload(context_id: u32, layer_id: u32, event: u32) -> MambaWorkload {
        MambaWorkload {
            context_id,
            layer_id,
            subop: contract::MambaSubop::Step,
            dependency_event: contract::MAMBA_NO_EVENT,
            completion_event: event,
            writes_completion: true,
            continues_state: true,
            batch: 1,
            sequence: 1,
            d_model: 16,
            d_inner: 16,
            heads: 2,
            head_dim: 8,
            state_dim: 16,
            conv_channels: 80,
            conv_kernel: 4,
            projection: 98,
            output_projection: 16,
            element_bytes: 2,
            state_request_stride: 1024,
            conv_request_stride: 1280,
            optional_bias_elements: 80,
        }
    }

    #[test]
    fn global_events_cross_queues_are_one_shot_and_state_keys_serialize() {
        let mut model = MambaTimingModel::new(config());
        let first = model.schedule(workload(7, 3, 10), 0, 0).unwrap();
        let mut second_workload = workload(8, 3, 11);
        second_workload.dependency_event = 10;
        let second = model.schedule(second_workload, 1, 0).unwrap();
        assert!(second.command_ready_cycle >= first.completion_cycle);
        assert!(second.dependency_wait_cycles >= first.completion_cycle);

        let mut consumed_again = workload(9, 3, 12);
        consumed_again.dependency_event = 10;
        assert!(matches!(
            model.schedule(consumed_again, 2, 0),
            Err(MambaError::StateHazard(_))
        ));

        let same_state = model.schedule(workload(7, 3, 13), 3, 0).unwrap();
        assert!(same_state.command_ready_cycle >= first.completion_cycle);
        assert!(same_state.state_wait_cycles >= first.completion_cycle);
        assert_eq!(same_state.external_scratch_bytes, 0);

        assert!(matches!(
            model.schedule(workload(10, 3, 11), 4, 0),
            Err(MambaError::StateHazard(_))
        ));
    }

    #[test]
    fn same_queue_dependency_and_queue_order_are_explicit() {
        let mut model = MambaTimingModel::new(config());
        let first = model.schedule(workload(1, 1, 20), 2, 5).unwrap();
        let mut next = workload(2, 1, 21);
        next.dependency_event = 20;
        let second = model.schedule(next, 2, 5).unwrap();
        assert_eq!(second.command_ready_cycle, first.completion_cycle);
        assert!(second.queue_wait_cycles > 0);
        assert!(second.dependency_wait_cycles > 0);
        assert!(second.hbm_bytes_read > contract::MAMBA_DESCRIPTOR_BYTES as u64);
        assert!(second.matrix_macs > 0);
        assert!(second.state_fmas > 0);
        assert_eq!(first.queue_depth_after_issue, 1);
        assert_eq!(second.queue_depth_after_issue, 2);
    }

    #[test]
    fn elapsed_cycles_excludes_the_completion_record_dma() {
        let mut model = MambaTimingModel::new(config());
        let record = model.schedule(workload(1, 1, 20), 0, 5).unwrap();
        let completion_write = record
            .stages
            .iter()
            .find(|stage| stage.name == "completion_write")
            .unwrap();

        assert_eq!(
            record.elapsed_cycles,
            completion_write.start_cycle - record.issue_cycle
        );
        assert_eq!(record.completion_cycle, completion_write.end_cycle);
        assert!(record.completion_cycle - record.issue_cycle > record.elapsed_cycles);
    }

    #[test]
    fn failed_terminal_commands_publish_and_consume_events() {
        let mut model = MambaTimingModel::new(config());
        let envelope = MambaDescriptorEnvelope {
            flags: 1 << contract::MAMBA_FLAG_WRITE_COMPLETION_BIT,
            context_id: 7,
            completion_addr: 0x1000,
            dependency_event: contract::MAMBA_NO_EVENT,
            completion_event: 30,
            layer_id: 3,
        };
        let first = model
            .schedule_terminal(
                envelope,
                Some(contract::MambaSubop::Wait),
                true,
                contract::MAMBA_STATUS_UNSUPPORTED_PROFILE,
                0,
                5,
            )
            .unwrap();
        assert_eq!(first.status, contract::MAMBA_STATUS_UNSUPPORTED_PROFILE);
        assert_eq!(first.subop, "WAIT");
        let completion_write = first
            .stages
            .iter()
            .find(|stage| stage.name == "completion_write")
            .unwrap();
        assert_eq!(
            first.elapsed_cycles,
            completion_write.start_cycle - first.issue_cycle
        );

        let dependent = MambaDescriptorEnvelope {
            dependency_event: 30,
            completion_event: 31,
            ..envelope
        };
        let second = model
            .schedule_terminal(
                dependent,
                None,
                true,
                contract::MAMBA_STATUS_INVALID_DESCRIPTOR,
                1,
                5,
            )
            .unwrap();
        assert_eq!(second.subop, "INVALID");
        assert!(second.command_ready_cycle >= first.completion_cycle);
        assert!(matches!(
            model.schedule_terminal(
                dependent,
                None,
                true,
                contract::MAMBA_STATUS_INVALID_DESCRIPTOR,
                2,
                5,
            ),
            Err(MambaError::StateHazard(_))
        ));
    }

    #[test]
    fn wait_workload_ignores_all_model_and_state_dimensions() {
        let mut descriptor = descriptor();
        descriptor.d_inner = u32::MAX;
        descriptor.num_heads = u32::MAX;
        descriptor.head_dim = u32::MAX;
        descriptor.state_dim = u32::MAX;
        descriptor.groups = u32::MAX;
        descriptor.d_mlp = u32::MAX;

        let wait = MambaWorkload::from_descriptor(&descriptor, contract::MambaSubop::Wait)
            .expect("WAIT must not derive a mixer shape");
        assert_eq!(wait.batch, 0);
        assert_eq!(wait.conv_channels, 0);
        assert_eq!(wait.projection, 0);
        assert_eq!(wait.state_request_stride, 0);
    }

    #[test]
    fn state_reset_workload_contains_only_persistent_state_shape() {
        let mut descriptor = descriptor();
        descriptor.d_model = u32::MAX;
        descriptor.d_mlp = u32::MAX;
        descriptor.in_proj_bias_addr = 64;

        let reset = MambaWorkload::from_descriptor(&descriptor, contract::MambaSubop::StateReset)
            .expect("STATE_RESET must not derive projection work");
        assert_eq!(reset.batch, 1);
        assert_eq!(reset.conv_channels, 48);
        assert_eq!(reset.projection, 0);
        assert_eq!(reset.output_projection, 0);
        assert_eq!(reset.optional_bias_elements, 0);
    }
}
