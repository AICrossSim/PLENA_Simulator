use std::collections::BTreeMap;

use super::descriptor::{StateDescriptor, StateIdentity};
use super::error::StateEngineError;
use super::generated_contract::{StateAlgorithm, StatePrecision};

#[derive(Clone, Debug)]
pub struct ResidentState {
    pub identity: StateIdentity,
    pub algorithm: StateAlgorithm,
    pub precision: StatePrecision,
    pub conv_precision: StatePrecision,
    pub state_hbm_addr: u64,
    pub conv_state_hbm_addr: u64,
    pub state: Vec<u8>,
    pub conv_state: Vec<u8>,
    pub state_scales: Vec<u8>,
    pub conv_state_scales: Vec<u8>,
    pub dirty: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateBuffers {
    pub state: Vec<u8>,
    pub conv_state: Vec<u8>,
    pub state_scales: Vec<u8>,
    pub conv_state_scales: Vec<u8>,
}

impl StateBuffers {
    pub fn zeros(descriptor: &StateDescriptor) -> Self {
        Self {
            state: vec![0; descriptor.state_bytes as usize],
            conv_state: vec![0; descriptor.conv_state_bytes as usize],
            state_scales: vec![127; descriptor.state_scale_bytes as usize],
            conv_state_scales: vec![127; descriptor.conv_state_scale_bytes as usize],
        }
    }

    fn matches(&self, descriptor: &StateDescriptor) -> bool {
        self.state.len() == descriptor.state_bytes as usize
            && self.conv_state.len() == descriptor.conv_state_bytes as usize
            && self.state_scales.len() == descriptor.state_scale_bytes as usize
            && self.conv_state_scales.len() == descriptor.conv_state_scale_bytes as usize
    }
}

impl ResidentState {
    fn matches(&self, descriptor: &StateDescriptor) -> bool {
        self.identity == descriptor.identity
            && self.algorithm == descriptor.algorithm
            && self.precision == descriptor.state_precision
            && self.conv_precision == descriptor.conv_state_precision
            && self.state_hbm_addr == descriptor.state_hbm_addr
            && self.conv_state_hbm_addr == descriptor.conv_state_hbm_addr
            && self.state.len() == descriptor.state_bytes as usize
            && self.conv_state.len() == descriptor.conv_state_bytes as usize
            && self.state_scales.len() == descriptor.state_scale_bytes as usize
            && self.conv_state_scales.len() == descriptor.conv_state_scale_bytes as usize
    }
}

#[derive(Debug)]
pub struct StateCache {
    capacity_bytes: u64,
    entries: BTreeMap<u32, ResidentState>,
}

impl StateCache {
    pub fn new(capacity_bytes: u64) -> Self {
        assert!(capacity_bytes > 0);
        Self {
            capacity_bytes,
            entries: BTreeMap::new(),
        }
    }

    pub fn preload(
        &mut self,
        descriptor: &StateDescriptor,
        state: Vec<u8>,
        conv_state: Vec<u8>,
        state_scales: Vec<u8>,
        conv_state_scales: Vec<u8>,
    ) -> Result<(), StateEngineError> {
        self.require_resident(descriptor)?;
        if state.len() != descriptor.state_bytes as usize
            || conv_state.len() != descriptor.conv_state_bytes as usize
            || state_scales.len() != descriptor.state_scale_bytes as usize
            || conv_state_scales.len() != descriptor.conv_state_scale_bytes as usize
        {
            return Err(StateEngineError::internal(
                "preload data length does not match descriptor",
            ));
        }
        self.require_free_region(descriptor)?;
        self.entries.insert(
            descriptor.state_sram_offset,
            ResidentState {
                identity: descriptor.identity,
                algorithm: descriptor.algorithm,
                precision: descriptor.state_precision,
                conv_precision: descriptor.conv_state_precision,
                state_hbm_addr: descriptor.state_hbm_addr,
                conv_state_hbm_addr: descriptor.conv_state_hbm_addr,
                state,
                conv_state,
                state_scales,
                conv_state_scales,
                dirty: false,
            },
        );
        Ok(())
    }

    pub fn reset(&mut self, descriptor: &StateDescriptor) -> Result<(), StateEngineError> {
        self.require_resident(descriptor)?;
        if let Some(entry) = self.entries.get_mut(&descriptor.state_sram_offset) {
            Self::require_matching(entry, descriptor)?;
            if entry.dirty {
                return Err(StateEngineError::hazard(
                    "cannot reset a dirty resident state",
                ));
            }
            entry.state.fill(0);
            entry.conv_state.fill(0);
            entry.state_scales.fill(127);
            entry.conv_state_scales.fill(127);
            entry.dirty = true;
            return Ok(());
        }
        self.require_free_region(descriptor)?;
        self.entries.insert(
            descriptor.state_sram_offset,
            ResidentState {
                identity: descriptor.identity,
                algorithm: descriptor.algorithm,
                precision: descriptor.state_precision,
                conv_precision: descriptor.conv_state_precision,
                state_hbm_addr: descriptor.state_hbm_addr,
                conv_state_hbm_addr: descriptor.conv_state_hbm_addr,
                state: vec![0; descriptor.state_bytes as usize],
                conv_state: vec![0; descriptor.conv_state_bytes as usize],
                state_scales: vec![127; descriptor.state_scale_bytes as usize],
                conv_state_scales: vec![127; descriptor.conv_state_scale_bytes as usize],
                dirty: true,
            },
        );
        Ok(())
    }

    pub fn compute_data(
        &self,
        descriptor: &StateDescriptor,
    ) -> Result<StateBuffers, StateEngineError> {
        self.require_resident(descriptor)?;
        let entry = self
            .entries
            .get(&descriptor.state_sram_offset)
            .ok_or_else(|| StateEngineError::hazard("state is not resident"))?;
        Self::require_matching(entry, descriptor)?;
        Ok(StateBuffers {
            state: entry.state.clone(),
            conv_state: entry.conv_state.clone(),
            state_scales: entry.state_scales.clone(),
            conv_state_scales: entry.conv_state_scales.clone(),
        })
    }

    pub fn store_compute_data(
        &mut self,
        descriptor: &StateDescriptor,
        buffers: StateBuffers,
    ) -> Result<(), StateEngineError> {
        self.require_resident(descriptor)?;
        if !buffers.matches(descriptor) {
            return Err(StateEngineError::internal(
                "computed state length does not match descriptor",
            ));
        }
        let entry = self
            .entries
            .get_mut(&descriptor.state_sram_offset)
            .ok_or_else(|| StateEngineError::hazard("state is not resident"))?;
        Self::require_matching(entry, descriptor)?;
        entry.state = buffers.state;
        entry.conv_state = buffers.conv_state;
        entry.state_scales = buffers.state_scales;
        entry.conv_state_scales = buffers.conv_state_scales;
        entry.dirty = true;
        Ok(())
    }

    pub fn commit_data(
        &self,
        descriptor: &StateDescriptor,
    ) -> Result<StateBuffers, StateEngineError> {
        self.require_resident(descriptor)?;
        let entry = self
            .entries
            .get(&descriptor.state_sram_offset)
            .ok_or_else(|| StateEngineError::hazard("state is not resident"))?;
        Self::require_matching(entry, descriptor)?;
        if !entry.dirty {
            return Err(StateEngineError::hazard(
                "cannot commit a clean resident state",
            ));
        }
        Ok(StateBuffers {
            state: entry.state.clone(),
            conv_state: entry.conv_state.clone(),
            state_scales: entry.state_scales.clone(),
            conv_state_scales: entry.conv_state_scales.clone(),
        })
    }

    pub fn mark_clean(&mut self, descriptor: &StateDescriptor) -> Result<(), StateEngineError> {
        let entry = self
            .entries
            .get_mut(&descriptor.state_sram_offset)
            .ok_or_else(|| StateEngineError::hazard("state is not resident"))?;
        Self::require_matching(entry, descriptor)?;
        entry.dirty = false;
        Ok(())
    }

    pub fn evict(&mut self, descriptor: &StateDescriptor) -> Result<(), StateEngineError> {
        self.require_resident(descriptor)?;
        let entry = self
            .entries
            .get(&descriptor.state_sram_offset)
            .ok_or_else(|| StateEngineError::hazard("state is not resident"))?;
        Self::require_matching(entry, descriptor)?;
        if entry.dirty {
            return Err(StateEngineError::hazard(
                "cannot evict a dirty resident state",
            ));
        }
        self.entries.remove(&descriptor.state_sram_offset);
        Ok(())
    }

    fn require_resident(&self, descriptor: &StateDescriptor) -> Result<(), StateEngineError> {
        if descriptor.is_streaming() {
            return Err(StateEngineError::hazard(
                "operation requires a resident state SRAM offset",
            ));
        }
        Ok(())
    }

    fn require_free_region(&self, descriptor: &StateDescriptor) -> Result<(), StateEngineError> {
        if self
            .entries
            .values()
            .any(|entry| entry.identity == descriptor.identity)
        {
            return Err(StateEngineError::hazard(
                "state identity is already resident at another slot",
            ));
        }
        let start = u64::from(descriptor.state_sram_offset);
        let len = descriptor.resident_bytes();
        let end = start
            .checked_add(len)
            .ok_or_else(|| StateEngineError::address("state SRAM range overflows"))?;
        if end > self.capacity_bytes {
            return Err(StateEngineError::address(format!(
                "state SRAM allocation [{start}, {end}) exceeds {} bytes",
                self.capacity_bytes
            )));
        }
        for (other_start, entry) in &self.entries {
            let other_start = u64::from(*other_start);
            let other_end = other_start
                + entry.state.len() as u64
                + entry.conv_state.len() as u64
                + entry.state_scales.len() as u64
                + entry.conv_state_scales.len() as u64;
            if start < other_end && other_start < end {
                return Err(StateEngineError::hazard(format!(
                    "state SRAM allocation [{start}, {end}) aliases [{other_start}, {other_end})"
                )));
            }
        }
        Ok(())
    }

    fn require_matching(
        entry: &ResidentState,
        descriptor: &StateDescriptor,
    ) -> Result<(), StateEngineError> {
        if !entry.matches(descriptor) {
            return Err(StateEngineError::hazard(
                "state SRAM slot is owned by a different context or shape",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_engine::descriptor::{KdaPayload, Mamba2Payload, StatePayload};
    use crate::state_engine::generated_contract::{NO_EVENT, StateStatus};

    fn descriptor(offset: u32, context: u32) -> StateDescriptor {
        StateDescriptor {
            algorithm: StateAlgorithm::Mamba2,
            state_precision: StatePrecision::Fp32,
            conv_state_precision: StatePrecision::Fp32,
            activation_precision: StatePrecision::Bf16,
            parameter_precision: StatePrecision::Bf16,
            flags: 0,
            identity: StateIdentity {
                context_id: context,
                request_id: 2,
                layer_id: 3,
                state_id: 4,
            },
            batch_size: 1,
            num_heads: 1,
            sequence_length: 1,
            token_offset: 0,
            valid_tokens: 1,
            chunk_size: 1,
            state_sram_offset: offset,
            state_bytes: 16,
            conv_state_bytes: 16,
            input_vram_addr: 0,
            output_vram_addr: 0,
            input_token_stride: 7,
            output_token_stride: 2,
            state_hbm_addr: 0,
            conv_state_hbm_addr: 64,
            state_scale_addr: 0,
            state_scale_bytes: 0,
            conv_state_scale_bytes: 0,
            completion_addr: 0,
            dependency_event: NO_EVENT,
            completion_event: NO_EVENT,
            payload: StatePayload::Mamba2(Mamba2Payload {
                head_dim: 2,
                state_dim: 2,
                groups: 1,
                conv_kernel: 1,
                xbc_offset: 2,
                dt_offset: 6,
                conv_weight_addr: 0,
                conv_bias_addr: 0,
                a_log_addr: 0,
                dt_bias_addr: 0,
                d_skip_addr: 0,
                parameter_scale_addr: 0,
                dt_min: 0.0,
                dt_max: f32::INFINITY,
            }),
        }
    }

    /// One head of KDA at [V=2, K=2]: state is 1*2*2 FP32 values and conv state
    /// is 1*(2*K + V) = 6 FP32 values, so the two algorithms deliberately have
    /// different resident footprints.
    fn kda_descriptor(offset: u32, identity: StateIdentity) -> StateDescriptor {
        StateDescriptor {
            algorithm: StateAlgorithm::Kda,
            state_precision: StatePrecision::Fp32,
            conv_state_precision: StatePrecision::Fp32,
            activation_precision: StatePrecision::Bf16,
            parameter_precision: StatePrecision::Bf16,
            flags: 0,
            identity,
            batch_size: 1,
            num_heads: 1,
            sequence_length: 1,
            token_offset: 0,
            valid_tokens: 1,
            chunk_size: 1,
            state_sram_offset: offset,
            state_bytes: 16,
            conv_state_bytes: 24,
            input_vram_addr: 0,
            output_vram_addr: 0,
            input_token_stride: 9,
            output_token_stride: 2,
            state_hbm_addr: 256,
            conv_state_hbm_addr: 320,
            state_scale_addr: 0,
            state_scale_bytes: 0,
            conv_state_scale_bytes: 0,
            completion_addr: 0,
            dependency_event: NO_EVENT,
            completion_event: NO_EVENT,
            payload: StatePayload::Kda(KdaPayload {
                key_dim: 2,
                value_dim: 2,
                conv_kernel: 1,
                q_offset: 0,
                k_offset: 2,
                v_offset: 4,
                decay_offset: 6,
                beta_offset: 8,
                q_conv_weight_addr: 0,
                k_conv_weight_addr: 0,
                v_conv_weight_addr: 0,
                q_conv_bias_addr: 0,
                k_conv_bias_addr: 0,
                v_conv_bias_addr: 0,
                a_log_addr: 0,
                dt_bias_addr: 0,
                parameter_scale_addr: 0,
                output_scale: 1.0,
                gate_lower_bound: -5.0,
            }),
        }
    }

    #[test]
    fn mamba2_and_kda_states_are_resident_at_the_same_time() {
        let mut cache = StateCache::new(1024);
        let mamba = descriptor(0, 1);
        let kda = kda_descriptor(
            64,
            StateIdentity {
                context_id: 1,
                request_id: 5,
                layer_id: 6,
                state_id: 7,
            },
        );
        cache
            .preload(&mamba, vec![0xA1; 16], vec![0xA2; 16], vec![], vec![])
            .unwrap();
        cache
            .preload(&kda, vec![0xB1; 16], vec![0xB2; 24], vec![], vec![])
            .unwrap();

        // Advancing one algorithm's state must leave the other's untouched.
        let mut buffers = cache.compute_data(&mamba).unwrap();
        buffers.state.fill(0xCC);
        cache.store_compute_data(&mamba, buffers).unwrap();

        let kda_after = cache.compute_data(&kda).unwrap();
        assert_eq!(kda_after.state, vec![0xB1; 16]);
        assert_eq!(kda_after.conv_state, vec![0xB2; 24]);

        let mamba_after = cache.compute_data(&mamba).unwrap();
        assert_eq!(mamba_after.state, vec![0xCC; 16]);
        assert_eq!(mamba_after.conv_state, vec![0xA2; 16]);

        // Each algorithm keeps its own dirty bit and lifecycle.
        assert_eq!(
            cache.commit_data(&kda).unwrap_err().status,
            StateStatus::StateHazard
        );
        let buffers = cache.commit_data(&mamba).unwrap();
        assert_eq!(buffers.state, vec![0xCC; 16]);
        assert_eq!(buffers.conv_state, vec![0xA2; 16]);
        cache.mark_clean(&mamba).unwrap();
        cache.evict(&mamba).unwrap();

        // Evicting Mamba leaves the KDA slot intact.
        assert_eq!(cache.compute_data(&kda).unwrap().state, vec![0xB1; 16]);
    }

    #[test]
    fn a_kda_descriptor_cannot_reuse_a_mamba2_slot() {
        let mut cache = StateCache::new(1024);
        let mamba = descriptor(0, 1);
        cache
            .preload(&mamba, vec![0; 16], vec![0; 16], vec![], vec![])
            .unwrap();

        // Same identity and same slot, different algorithm: the resident entry
        // is owned by Mamba-2 and every access must be refused rather than
        // reinterpreting its bytes as a KDA state.
        let impostor = kda_descriptor(0, mamba.identity);
        assert_eq!(
            cache.compute_data(&impostor).unwrap_err().status,
            StateStatus::StateHazard
        );
        assert_eq!(
            cache.commit_data(&impostor).unwrap_err().status,
            StateStatus::StateHazard
        );
        assert_eq!(
            cache.evict(&impostor).unwrap_err().status,
            StateStatus::StateHazard
        );
        assert_eq!(
            cache.reset(&impostor).unwrap_err().status,
            StateStatus::StateHazard
        );
        assert_eq!(cache.compute_data(&mamba).unwrap().state, vec![0; 16]);
    }

    #[test]
    fn dirty_state_must_be_committed_before_evict() {
        let mut cache = StateCache::new(1024);
        let descriptor = descriptor(0, 1);
        cache
            .preload(&descriptor, vec![1; 16], vec![2; 16], vec![], vec![])
            .unwrap();
        let buffers = cache.compute_data(&descriptor).unwrap();
        cache.store_compute_data(&descriptor, buffers).unwrap();
        assert_eq!(
            cache.evict(&descriptor).unwrap_err().status,
            StateStatus::StateHazard
        );
        let _ = cache.commit_data(&descriptor).unwrap();
        cache.mark_clean(&descriptor).unwrap();
        cache.evict(&descriptor).unwrap();
    }

    #[test]
    fn alias_and_cross_context_access_are_rejected() {
        let mut cache = StateCache::new(1024);
        let first = descriptor(0, 1);
        cache
            .preload(&first, vec![0; 16], vec![0; 16], vec![], vec![])
            .unwrap();
        let alias = descriptor(0, 2);
        assert_eq!(
            cache
                .preload(&alias, vec![0; 16], vec![0; 16], vec![], vec![])
                .unwrap_err()
                .status,
            StateStatus::StateHazard
        );
        assert_eq!(
            cache.compute_data(&alias).unwrap_err().status,
            StateStatus::StateHazard
        );
    }
}
