//! Functional command lifecycle for `X_MAMBA`.
//!
//! Functional tensor execution and event timing are deliberately independent:
//! host numerical work never becomes simulated hardware latency.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use memory::{ErasedMemoryModel, MemoryModel};
use runtime::Executor;

use crate::generated_contract as contract;
use crate::runtime_config::PERIOD;

use super::hbm::{execute_functional_hbm, read_bytes, write_completion};
use super::timing::{MambaTimingConfig, MambaTimingModel, MambaWorkload};
use super::{MambaDescriptor, MambaDescriptorEnvelope, MambaError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MambaInstruction {
    pub(crate) register_context: u32,
    pub(crate) descriptor_base: u64,
    pub(crate) descriptor_offset: u32,
    pub(crate) descriptor_hbm_register: u8,
    pub(crate) queue_id: u8,
    pub(crate) subop: u8,
    pub(crate) reserved: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MambaCommandOutcome {
    pub(crate) status: u32,
    pub(crate) completion_event: u32,
    /// Functional mode does not claim a hardware cycle estimate.
    pub(crate) elapsed_cycles: u64,
}

pub(crate) struct MambaEngine {
    hbm: Arc<dyn ErasedMemoryModel>,
    events: MambaFunctionalEvents,
    timing: Option<MambaTimingModel>,
}

#[derive(Default)]
struct MambaFunctionalEvents {
    completed: HashSet<u32>,
}

impl MambaFunctionalEvents {
    fn begin(&mut self, dependency_event: u32, completion_event: u32) -> Result<(), MambaError> {
        if completion_event != contract::MAMBA_NO_EVENT
            && self.completed.contains(&completion_event)
        {
            return Err(MambaError::StateHazard(
                "completion event is already active",
            ));
        }
        if dependency_event != contract::MAMBA_NO_EVENT && !self.completed.remove(&dependency_event)
        {
            return Err(MambaError::StateHazard(
                "dependency event has no active completed producer",
            ));
        }
        Ok(())
    }

    fn publish(&mut self, completion_event: u32) {
        if completion_event != contract::MAMBA_NO_EVENT {
            self.completed.insert(completion_event);
        }
    }
}

impl MambaEngine {
    pub(crate) fn new(
        hbm: Arc<dyn ErasedMemoryModel>,
        timing_config: Option<MambaTimingConfig>,
    ) -> Self {
        Self {
            hbm,
            events: MambaFunctionalEvents::default(),
            timing: timing_config.map(MambaTimingModel::new),
        }
    }

    fn current_cycle() -> u64 {
        Executor::current().now().to_picos() / PERIOD.to_picos()
    }

    pub(crate) fn write_timing_profile(&self, path: &Path) -> Result<(), String> {
        self.timing
            .as_ref()
            .ok_or_else(|| "Mamba timing profile requested without a timing config".to_string())?
            .write_json(path)
    }

    fn descriptor_address(&self, instruction: MambaInstruction) -> Result<u64, MambaError> {
        let address = instruction
            .descriptor_base
            .checked_add(instruction.descriptor_offset as u64)
            .ok_or(MambaError::AddressError(
                "descriptor base plus offset overflows u64",
            ))?;
        if !address.is_multiple_of(contract::MAMBA_DESCRIPTOR_ALIGNMENT as u64) {
            return Err(MambaError::AddressError(
                "descriptor pointer is not 64-byte aligned",
            ));
        }
        let end = address
            .checked_add(contract::MAMBA_DESCRIPTOR_BYTES as u64)
            .ok_or(MambaError::AddressError("descriptor range overflows u64"))?;
        if let Some(capacity) = self.hbm.capacity_bytes()
            && end > capacity
        {
            return Err(MambaError::AddressError(
                "descriptor range exceeds HBM capacity",
            ));
        }
        Ok(address)
    }

    fn completion_is_safe(
        &self,
        envelope: MambaDescriptorEnvelope,
        descriptor_address: u64,
    ) -> bool {
        if !envelope.write_completion()
            || envelope.completion_addr == 0
            || !envelope
                .completion_addr
                .is_multiple_of(contract::MAMBA_COMPLETION_ALIGNMENT as u64)
        {
            return false;
        }
        let Some(completion_end) = envelope
            .completion_addr
            .checked_add(contract::MAMBA_COMPLETION_BYTES as u64)
        else {
            return false;
        };
        let descriptor_end = descriptor_address + contract::MAMBA_DESCRIPTOR_BYTES as u64;
        if envelope.completion_addr < descriptor_end && descriptor_address < completion_end {
            return false;
        }
        self.hbm
            .capacity_bytes()
            .is_none_or(|capacity| completion_end <= capacity)
    }

    async fn finish(
        &mut self,
        descriptor_address: u64,
        envelope: MambaDescriptorEnvelope,
        status: u32,
        elapsed_cycles: u64,
        event_reserved: bool,
    ) -> MambaCommandOutcome {
        let mut final_status = status;
        if self.completion_is_safe(envelope, descriptor_address)
            && write_completion(
                &self.hbm,
                envelope.completion_addr,
                status,
                envelope.completion_event,
                elapsed_cycles,
            )
            .await
            .is_err()
        {
            final_status = contract::MAMBA_STATUS_INTERNAL_ERROR;
        }
        if event_reserved && self.timing.is_none() {
            self.events.publish(envelope.completion_event);
        }
        MambaCommandOutcome {
            status: final_status,
            completion_event: if event_reserved {
                envelope.completion_event
            } else {
                contract::MAMBA_NO_EVENT
            },
            elapsed_cycles,
        }
    }

    async fn finish_terminal_failure(
        &mut self,
        queue_id: u8,
        descriptor_address: u64,
        envelope: MambaDescriptorEnvelope,
        subop: Option<contract::MambaSubop>,
        status: u32,
        issue_cycle: u64,
    ) -> MambaCommandOutcome {
        let writes_completion = self.completion_is_safe(envelope, descriptor_address);
        let (terminal_status, elapsed_cycles, event_reserved) =
            if let Some(timing) = self.timing.as_mut() {
                match timing.schedule_terminal(
                    envelope,
                    subop,
                    writes_completion,
                    status,
                    queue_id,
                    issue_cycle,
                ) {
                    Ok(record) => (status, record.elapsed_cycles, true),
                    Err(error) => (error.status(), 0, false),
                }
            } else {
                match self
                    .events
                    .begin(envelope.dependency_event, envelope.completion_event)
                {
                    Ok(()) => (status, 0, true),
                    Err(error) => (error.status(), 0, false),
                }
            };
        self.finish(
            descriptor_address,
            envelope,
            terminal_status,
            elapsed_cycles,
            event_reserved,
        )
        .await
    }

    pub(crate) async fn execute(&mut self, instruction: MambaInstruction) -> MambaCommandOutcome {
        let issue_cycle = Self::current_cycle();
        let descriptor_address = match self.descriptor_address(instruction) {
            Ok(address) => address,
            Err(error) => {
                return MambaCommandOutcome {
                    status: error.status(),
                    completion_event: contract::MAMBA_NO_EVENT,
                    elapsed_cycles: 0,
                };
            }
        };
        let descriptor_bytes = match read_bytes(
            &self.hbm,
            descriptor_address,
            contract::MAMBA_DESCRIPTOR_BYTES,
        )
        .await
        {
            Ok(bytes) => bytes,
            Err(error) => {
                return MambaCommandOutcome {
                    status: error.status(),
                    completion_event: contract::MAMBA_NO_EVENT,
                    elapsed_cycles: 0,
                };
            }
        };
        let subop_hint = contract::MambaSubop::try_from(instruction.subop).ok();
        let envelope = match MambaDescriptorEnvelope::parse(&descriptor_bytes) {
            Ok(envelope) => envelope,
            Err(error) => {
                return MambaCommandOutcome {
                    status: error.status(),
                    completion_event: contract::MAMBA_NO_EVENT,
                    elapsed_cycles: 0,
                };
            }
        };
        let descriptor = match MambaDescriptor::parse(&descriptor_bytes) {
            Ok(descriptor) => descriptor,
            Err(error) => {
                return self
                    .finish_terminal_failure(
                        instruction.queue_id,
                        descriptor_address,
                        envelope,
                        subop_hint,
                        error.status(),
                        issue_cycle,
                    )
                    .await;
            }
        };

        if instruction.reserved != 0 || instruction.descriptor_hbm_register >= 8 {
            return self
                .finish_terminal_failure(
                    instruction.queue_id,
                    descriptor_address,
                    envelope,
                    subop_hint,
                    MambaError::InvalidDescriptor("invalid X_MAMBA instruction fields").status(),
                    issue_cycle,
                )
                .await;
        }
        let subop = match contract::MambaSubop::try_from(instruction.subop) {
            Ok(subop) => subop,
            Err(_) => {
                return self
                    .finish_terminal_failure(
                        instruction.queue_id,
                        descriptor_address,
                        envelope,
                        None,
                        MambaError::InvalidDescriptor("unknown X_MAMBA sub-operation").status(),
                        issue_cycle,
                    )
                    .await;
            }
        };
        if let Err(error) = descriptor.validate(subop, instruction.register_context) {
            return self
                .finish_terminal_failure(
                    instruction.queue_id,
                    descriptor_address,
                    envelope,
                    Some(subop),
                    error.status(),
                    issue_cycle,
                )
                .await;
        }
        if let Err(error) =
            descriptor.validate_memory_map(subop, descriptor_address, self.hbm.capacity_bytes())
        {
            return self
                .finish_terminal_failure(
                    instruction.queue_id,
                    descriptor_address,
                    envelope,
                    Some(subop),
                    error.status(),
                    issue_cycle,
                )
                .await;
        }

        let elapsed_cycles = if let Some(timing) = self.timing.as_mut() {
            let workload = match MambaWorkload::from_descriptor(&descriptor, subop) {
                Ok(workload) => workload,
                Err(error) => {
                    return self
                        .finish_terminal_failure(
                            instruction.queue_id,
                            descriptor_address,
                            envelope,
                            Some(subop),
                            error.status(),
                            issue_cycle,
                        )
                        .await;
                }
            };
            match timing.schedule(workload, instruction.queue_id, issue_cycle) {
                Ok(record) => record.elapsed_cycles,
                Err(error) => {
                    return self
                        .finish(descriptor_address, envelope, error.status(), 0, false)
                        .await;
                }
            }
        } else {
            if let Err(error) = self
                .events
                .begin(descriptor.dependency_event, descriptor.completion_event)
            {
                return self
                    .finish(descriptor_address, envelope, error.status(), 0, false)
                    .await;
            }
            0
        };

        let status = match execute_functional_hbm(&self.hbm, &descriptor, subop).await {
            Ok(()) => contract::MAMBA_STATUS_SUCCESS,
            Err(error) => error.status(),
        };
        self.finish(descriptor_address, envelope, status, elapsed_cycles, true)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn functional_events_are_global_one_shot_edges() {
        let mut events = MambaFunctionalEvents::default();
        events.begin(contract::MAMBA_NO_EVENT, 10).unwrap();
        events.publish(10);

        assert!(matches!(
            events.begin(contract::MAMBA_NO_EVENT, 10),
            Err(MambaError::StateHazard(_))
        ));
        events.begin(10, 11).unwrap();
        assert!(matches!(
            events.begin(10, 12),
            Err(MambaError::StateHazard(_))
        ));
        events.publish(11);
        events.begin(11, 10).unwrap();
    }
}
