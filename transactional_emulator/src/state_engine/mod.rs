mod cache;
pub mod descriptor;
mod error;
mod functional;
pub mod generated_contract;
pub mod hbm;
mod layout;
mod precision;
mod projection;
mod queue;
pub mod timing;
mod vram;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;

use memory::ErasedMemoryModel;
use runtime::Executor;
use sram::VectorSram;
use tokio::sync::Mutex as AsyncMutex;

use cache::{StateBuffers, StateCache};
use descriptor::StateDescriptor;
use error::StateEngineError;
use generated_contract::{self as wire, StateStatus, StateSubop};
use hbm::{read_bytes, write_bytes};
use layout::{LayoutDescriptor, LayoutMode, LayoutStore};
use projection::ProjectionBufferStats;
use queue::CompletionLatch;
use timing::{StateEngineProfile, StateTimingConfig};

#[cfg(test)]
const DEFAULT_STATE_SRAM_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub struct EncodedStateCommand {
    pub context_gp: u8,
    pub descriptor_offset_gp: u8,
    pub descriptor_hbm_reg: u8,
    pub queue_id: u8,
    pub subop: u8,
    pub context_id: u32,
    pub descriptor_address: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
pub struct EncodedLayoutCommand {
    pub context_gp: u8,
    pub descriptor_offset_gp: u8,
    pub descriptor_hbm_reg: u8,
    pub buffer_id: u8,
    pub mode: u8,
    pub context_id: u32,
    pub descriptor_address: Option<u64>,
}

pub struct StateEngine {
    hbm: Arc<dyn ErasedMemoryModel>,
    vram: Arc<VectorSram>,
    cache: Arc<AsyncMutex<StateCache>>,
    layouts: Arc<AsyncMutex<LayoutStore>>,
    engine_lane: Arc<AsyncMutex<()>>,
    timing_config: StateTimingConfig,
    profile: Arc<StdMutex<StateEngineProfile>>,
    async_queues: bool,
    queue_tails: Vec<Arc<CompletionLatch>>,
    events: HashMap<u32, Arc<CompletionLatch>>,
    event_producers: HashSet<u32>,
    last_charged_compute_cycles: u64,
    last_charged_layout_cycles: u64,
}

impl StateEngine {
    #[cfg(test)]
    pub fn new(hbm: Arc<dyn ErasedMemoryModel>, vram: Arc<VectorSram>) -> Self {
        Self::with_config(
            hbm,
            vram,
            DEFAULT_STATE_SRAM_BYTES,
            StateTimingConfig::default(),
        )
    }

    #[cfg(test)]
    pub fn with_config(
        hbm: Arc<dyn ErasedMemoryModel>,
        vram: Arc<VectorSram>,
        state_sram_bytes: u64,
        timing_config: StateTimingConfig,
    ) -> Self {
        Self::with_mode(hbm, vram, state_sram_bytes, timing_config, false)
    }

    pub fn with_mode(
        hbm: Arc<dyn ErasedMemoryModel>,
        vram: Arc<VectorSram>,
        state_sram_bytes: u64,
        timing_config: StateTimingConfig,
        async_queues: bool,
    ) -> Self {
        assert!(state_sram_bytes > 0);
        let timing_config = timing_config
            .validate()
            .expect("invalid X_STATE timing configuration");
        Self {
            hbm,
            vram,
            cache: Arc::new(AsyncMutex::new(StateCache::new(state_sram_bytes))),
            layouts: Arc::new(AsyncMutex::new(LayoutStore::default())),
            engine_lane: Arc::new(AsyncMutex::new(())),
            timing_config,
            profile: Arc::new(StdMutex::new(StateEngineProfile::new(timing_config))),
            async_queues,
            queue_tails: (0..16)
                .map(|_| Arc::new(CompletionLatch::completed(StateStatus::Success)))
                .collect(),
            events: HashMap::new(),
            event_producers: HashSet::new(),
            last_charged_compute_cycles: 0,
            last_charged_layout_cycles: 0,
        }
    }

    pub fn take_charged_compute_cycles(&mut self) -> u64 {
        std::mem::take(&mut self.last_charged_compute_cycles)
    }

    pub fn take_charged_layout_cycles(&mut self) -> u64 {
        std::mem::take(&mut self.last_charged_layout_cycles)
    }

    pub async fn execute_layout(&mut self, command: EncodedLayoutCommand) -> StateStatus {
        self.last_charged_layout_cycles = 0;
        if command.context_gp >= 16
            || command.descriptor_offset_gp >= 16
            || command.descriptor_hbm_reg >= 8
        {
            tracing::error!(?command, "non-canonical L_SCATTER_M register fields");
            return StateStatus::InvalidDescriptor;
        }
        let Some(descriptor_address) = command.descriptor_address else {
            tracing::error!("L_SCATTER_M descriptor address overflowed");
            return StateStatus::AddressError;
        };
        if !descriptor_address.is_multiple_of(layout::DESCRIPTOR_ALIGNMENT) {
            tracing::error!(descriptor_address, "unaligned L_SCATTER_M descriptor");
            return StateStatus::AddressError;
        }
        let raw = read_bytes(&self.hbm, descriptor_address, layout::DESCRIPTOR_SIZE).await;
        let descriptor = match LayoutDescriptor::parse(&raw) {
            Ok(descriptor) => descriptor,
            Err(error) => {
                tracing::error!(%error, "L_SCATTER_M failed");
                return error.status;
            }
        };
        let mode = match LayoutMode::try_from(command.mode) {
            Ok(mode) => mode,
            Err(()) => return StateStatus::InvalidDescriptor,
        };
        if descriptor.mode != mode
            || descriptor.buffer_id != command.buffer_id
            || descriptor.context_id != command.context_id
        {
            tracing::error!(
                descriptor_mode = ?descriptor.mode,
                instruction_mode = ?mode,
                descriptor_buffer = descriptor.buffer_id,
                instruction_buffer = command.buffer_id,
                "L_SCATTER_M instruction and descriptor disagree"
            );
            return StateStatus::StateHazard;
        }
        let result = {
            let _lane = self.engine_lane.lock().await;
            let mut layouts = self.layouts.lock().await;
            layouts.stage(descriptor, &self.vram).await
        };
        match result {
            Ok(stats) => {
                self.last_charged_layout_cycles = stats
                    .write_service_cycles
                    .saturating_add(stats.fifo_backpressure_cycles);
                StateStatus::Success
            }
            Err(error) => {
                tracing::error!(%error, "L_SCATTER_M failed");
                error.status
            }
        }
    }

    pub fn write_profile(&self, path: &std::path::Path) -> std::io::Result<()> {
        self.profile.lock().unwrap().write_json(path)
    }

    pub async fn execute(&mut self, command: EncodedStateCommand) -> StateStatus {
        self.last_charged_compute_cycles = 0;
        let subop = match StateSubop::try_from(command.subop) {
            Ok(subop) => subop,
            Err(()) => {
                tracing::error!(subop = command.subop, "invalid X_STATE subop");
                return StateStatus::InvalidDescriptor;
            }
        };
        if subop == StateSubop::Fence {
            return self.execute_fence(command).await;
        }

        let Some(descriptor_address) = command.descriptor_address else {
            tracing::error!("X_STATE descriptor address overflowed");
            return StateStatus::AddressError;
        };
        if !descriptor_address.is_multiple_of(wire::DESCRIPTOR_ALIGNMENT) {
            tracing::error!(descriptor_address, "unaligned X_STATE descriptor");
            return StateStatus::AddressError;
        }
        let raw = read_bytes(&self.hbm, descriptor_address, wire::DESCRIPTOR_SIZE).await;
        let completion_target = CompletionTarget::from_untrusted_descriptor(&raw);
        let descriptor = match StateDescriptor::parse(&raw) {
            Ok(descriptor) => descriptor,
            Err(error) => {
                let error = StateEngineError::from(error);
                tracing::error!(?subop, queue_id = command.queue_id, %error, "X_STATE failed");
                write_completion_target(&self.hbm, completion_target, error.status, 0).await;
                return error.status;
            }
        };
        if descriptor.identity.context_id != command.context_id {
            let error = StateEngineError::hazard(format!(
                "context register {} does not own descriptor context {}",
                command.context_id, descriptor.identity.context_id
            ));
            tracing::error!(?subop, queue_id = command.queue_id, %error, "X_STATE failed");
            write_completion_target(&self.hbm, completion_target, error.status, 0).await;
            return error.status;
        }

        let mut timing = timing::estimate(&descriptor, subop, self.timing_config);
        if self.async_queues {
            return self
                .issue_async(
                    command.queue_id,
                    descriptor,
                    subop,
                    timing,
                    completion_target,
                )
                .await;
        }
        if descriptor.dependency_event != wire::NO_EVENT {
            let error = StateEngineError::invalid("dependency_event requires --state-async-queues");
            tracing::error!(?subop, queue_id = command.queue_id, %error, "X_STATE failed");
            write_completion_target(&self.hbm, completion_target, error.status, 0).await;
            return error.status;
        }

        let result = {
            let _lane = self.engine_lane.lock().await;
            let mut cache = self.cache.lock().await;
            let mut layouts = self.layouts.lock().await;
            run_subop(
                &self.hbm,
                &self.vram,
                &mut cache,
                &mut layouts,
                &descriptor,
                subop,
            )
            .await
        };
        let (status, elapsed_cycles) = match result {
            Ok(projection_stats) => {
                timing.record_projection_buffer(projection_stats, self.timing_config);
                self.last_charged_compute_cycles = timing.charged_compute_cycles();
                let elapsed_cycles = timing.estimated_total_cycles;
                self.profile.lock().unwrap().record(timing);
                (StateStatus::Success, elapsed_cycles)
            }
            Err(error) => {
                tracing::error!(?subop, queue_id = command.queue_id, %error, "X_STATE failed");
                (error.status, 0)
            }
        };
        write_completion_target(&self.hbm, completion_target, status, elapsed_cycles).await;
        status
    }

    async fn execute_fence(&mut self, command: EncodedStateCommand) -> StateStatus {
        if command.context_gp != 0
            || command.descriptor_offset_gp != 0
            || command.descriptor_hbm_reg != 0
        {
            tracing::error!("X_STATE FENCE has nonzero descriptor operands");
            return StateStatus::InvalidDescriptor;
        }
        if !self.async_queues {
            return StateStatus::Success;
        }
        let tail = self.queue_tails[usize::from(command.queue_id)].clone();
        tail.wait().await
    }

    async fn issue_async(
        &mut self,
        queue_id: u8,
        descriptor: StateDescriptor,
        subop: StateSubop,
        mut timing: timing::StateExecutionRecord,
        completion_target: Option<CompletionTarget>,
    ) -> StateStatus {
        if descriptor.dependency_event == descriptor.completion_event
            && descriptor.dependency_event != wire::NO_EVENT
        {
            tracing::error!(
                event = descriptor.dependency_event,
                "X_STATE command cannot depend on its own completion event"
            );
            write_completion_target(&self.hbm, completion_target, StateStatus::StateHazard, 0)
                .await;
            return StateStatus::StateHazard;
        }
        let dependency = if descriptor.dependency_event == wire::NO_EVENT {
            None
        } else {
            let Some(dependency) = self.events.get(&descriptor.dependency_event).cloned() else {
                tracing::error!(
                    dependency_event = descriptor.dependency_event,
                    "X_STATE dependency event has no earlier producer"
                );
                write_completion_target(&self.hbm, completion_target, StateStatus::StateHazard, 0)
                    .await;
                return StateStatus::StateHazard;
            };
            Some(dependency)
        };
        let completion_event = if descriptor.completion_event == wire::NO_EVENT {
            None
        } else {
            if !self.event_producers.insert(descriptor.completion_event) {
                tracing::error!(
                    completion_event = descriptor.completion_event,
                    "duplicate X_STATE completion event producer"
                );
                write_completion_target(&self.hbm, completion_target, StateStatus::StateHazard, 0)
                    .await;
                return StateStatus::StateHazard;
            }
            Some(
                self.events
                    .entry(descriptor.completion_event)
                    .or_insert_with(|| Arc::new(CompletionLatch::pending()))
                    .clone(),
            )
        };
        let predecessor = self.queue_tails[usize::from(queue_id)].clone();
        let command_done = Arc::new(CompletionLatch::pending());
        self.queue_tails[usize::from(queue_id)] = command_done.clone();

        let hbm = self.hbm.clone();
        let vram = self.vram.clone();
        let cache = self.cache.clone();
        let layouts = self.layouts.clone();
        let engine_lane = self.engine_lane.clone();
        let profile = self.profile.clone();
        let timing_config = self.timing_config;
        Executor::current().spawn(async move {
            let mut status = predecessor.wait().await;
            if status == StateStatus::Success
                && let Some(dependency) = dependency
            {
                status = dependency.wait().await;
            }
            if status == StateStatus::Success {
                let result = {
                    let _lane = engine_lane.lock().await;
                    let mut cache = cache.lock().await;
                    let mut layouts = layouts.lock().await;
                    let result =
                        run_subop(&hbm, &vram, &mut cache, &mut layouts, &descriptor, subop).await;
                    if let Ok(projection_stats) = result.as_ref() {
                        timing.record_projection_buffer(*projection_stats, timing_config);
                        wait_cycles(timing.charged_compute_cycles()).await;
                    }
                    result
                };
                status = match result {
                    Ok(_) => {
                        profile.lock().unwrap().record(timing.clone());
                        StateStatus::Success
                    }
                    Err(error) => {
                        tracing::error!(?subop, queue_id, %error, "asynchronous X_STATE failed");
                        error.status
                    }
                };
            }
            let elapsed_cycles = if status == StateStatus::Success {
                timing.estimated_total_cycles
            } else {
                0
            };
            write_completion_target(&hbm, completion_target, status, elapsed_cycles).await;
            if let Some(completion_event) = completion_event {
                completion_event.signal(status);
            }
            command_done.signal(status);
        });
        StateStatus::Success
    }

    pub async fn fence_all(&mut self) -> StateStatus {
        let tails = self.queue_tails.clone();
        let mut aggregate = StateStatus::Success;
        for tail in tails {
            let status = tail.wait().await;
            if status != StateStatus::Success {
                aggregate = status;
            }
        }
        aggregate
    }
}

async fn run_subop(
    hbm: &Arc<dyn ErasedMemoryModel>,
    vram: &Arc<VectorSram>,
    cache: &mut StateCache,
    layouts: &mut LayoutStore,
    descriptor: &StateDescriptor,
    subop: StateSubop,
) -> Result<ProjectionBufferStats, StateEngineError> {
    match subop {
        StateSubop::Preload => {
            if descriptor.is_streaming() {
                return Err(StateEngineError::hazard(
                    "PRELOAD is invalid for streaming state",
                ));
            }
            let state = read_bytes(
                hbm,
                descriptor.state_hbm_addr,
                descriptor.state_bytes as usize,
            )
            .await;
            let conv_state = read_bytes(
                hbm,
                descriptor.conv_state_hbm_addr,
                descriptor.conv_state_bytes as usize,
            )
            .await;
            let scales = read_bytes(
                hbm,
                descriptor.state_scale_addr,
                descriptor.state_scale_bytes as usize + descriptor.conv_state_scale_bytes as usize,
            )
            .await;
            let split = descriptor.state_scale_bytes as usize;
            cache.preload(
                descriptor,
                state,
                conv_state,
                scales[..split].to_vec(),
                scales[split..].to_vec(),
            )?;
            Ok(ProjectionBufferStats::default())
        }
        StateSubop::Reset => {
            if descriptor.is_streaming() {
                let buffers = StateBuffers::zeros(descriptor);
                write_bytes(hbm, descriptor.state_hbm_addr, &buffers.state).await;
                write_bytes(hbm, descriptor.conv_state_hbm_addr, &buffers.conv_state).await;
                if !buffers.state_scales.is_empty() || !buffers.conv_state_scales.is_empty() {
                    let mut scales = buffers.state_scales;
                    scales.extend_from_slice(&buffers.conv_state_scales);
                    write_bytes(hbm, descriptor.state_scale_addr, &scales).await;
                }
                Ok(ProjectionBufferStats::default())
            } else {
                cache.reset(descriptor)?;
                Ok(ProjectionBufferStats::default())
            }
        }
        StateSubop::Commit => {
            if descriptor.is_streaming() {
                return Err(StateEngineError::hazard(
                    "COMMIT is invalid for streaming state",
                ));
            }
            let buffers = cache.commit_data(descriptor)?;
            write_bytes(hbm, descriptor.state_hbm_addr, &buffers.state).await;
            write_bytes(hbm, descriptor.conv_state_hbm_addr, &buffers.conv_state).await;
            if !buffers.state_scales.is_empty() || !buffers.conv_state_scales.is_empty() {
                let mut scales = buffers.state_scales;
                scales.extend_from_slice(&buffers.conv_state_scales);
                write_bytes(hbm, descriptor.state_scale_addr, &scales).await;
            }
            cache.mark_clean(descriptor)?;
            Ok(ProjectionBufferStats::default())
        }
        StateSubop::Evict => {
            if descriptor.is_streaming() {
                return Err(StateEngineError::hazard(
                    "EVICT is invalid for streaming state",
                ));
            }
            cache.evict(descriptor)?;
            Ok(ProjectionBufferStats::default())
        }
        StateSubop::Prefill | StateSubop::Step => {
            if subop == StateSubop::Step && descriptor.valid_tokens != 1 {
                return Err(StateEngineError::invalid(
                    "X_STATE STEP requires valid_tokens=1",
                ));
            }
            let mut buffers = if descriptor.is_streaming() {
                read_streaming_state(hbm, descriptor).await
            } else {
                cache.compute_data(descriptor)?
            };
            let projection_stats =
                functional::execute(hbm, vram, layouts, descriptor, &mut buffers).await?;
            if descriptor.is_streaming() {
                write_streaming_state(hbm, descriptor, &buffers).await;
            } else {
                cache.store_compute_data(descriptor, buffers)?;
            }
            Ok(projection_stats)
        }
        StateSubop::Fence => unreachable!("FENCE is handled before descriptor fetch"),
    }
}

async fn read_streaming_state(
    hbm: &Arc<dyn ErasedMemoryModel>,
    descriptor: &StateDescriptor,
) -> StateBuffers {
    let state = read_bytes(
        hbm,
        descriptor.state_hbm_addr,
        descriptor.state_bytes as usize,
    )
    .await;
    let conv_state = read_bytes(
        hbm,
        descriptor.conv_state_hbm_addr,
        descriptor.conv_state_bytes as usize,
    )
    .await;
    let scales = read_bytes(
        hbm,
        descriptor.state_scale_addr,
        descriptor.state_scale_bytes as usize + descriptor.conv_state_scale_bytes as usize,
    )
    .await;
    let split = descriptor.state_scale_bytes as usize;
    StateBuffers {
        state,
        conv_state,
        state_scales: scales[..split].to_vec(),
        conv_state_scales: scales[split..].to_vec(),
    }
}

async fn write_streaming_state(
    hbm: &Arc<dyn ErasedMemoryModel>,
    descriptor: &StateDescriptor,
    buffers: &StateBuffers,
) {
    write_bytes(hbm, descriptor.state_hbm_addr, &buffers.state).await;
    write_bytes(hbm, descriptor.conv_state_hbm_addr, &buffers.conv_state).await;
    if !buffers.state_scales.is_empty() || !buffers.conv_state_scales.is_empty() {
        let mut scales = buffers.state_scales.clone();
        scales.extend_from_slice(&buffers.conv_state_scales);
        write_bytes(hbm, descriptor.state_scale_addr, &scales).await;
    }
}

pub(crate) async fn wait_cycles(mut cycles: u64) {
    while cycles > 0 {
        let chunk = cycles.min(u64::from(u32::MAX)) as u32;
        crate::cycle!(chunk);
        cycles -= u64::from(chunk);
    }
}

async fn write_completion_target(
    hbm: &Arc<dyn ErasedMemoryModel>,
    target: Option<CompletionTarget>,
    status: StateStatus,
    elapsed_cycles: u64,
) {
    if let Some(target) = target {
        let completion = completion_bytes(status, target.completion_event, elapsed_cycles);
        write_bytes(hbm, target.address, &completion).await;
    }
}

#[derive(Clone, Copy, Debug)]
struct CompletionTarget {
    address: u64,
    completion_event: u32,
}

impl CompletionTarget {
    fn from_untrusted_descriptor(data: &[u8]) -> Option<Self> {
        if data.len() != wire::DESCRIPTOR_SIZE {
            return None;
        }
        let flags = u32::from_le_bytes(
            data[wire::common::FLAGS..wire::common::FLAGS + 4]
                .try_into()
                .unwrap(),
        );
        if flags & wire::FLAG_WRITE_COMPLETION == 0 {
            return None;
        }
        let address = u64::from_le_bytes(
            data[wire::common::COMPLETION_ADDR..wire::common::COMPLETION_ADDR + 8]
                .try_into()
                .unwrap(),
        );
        if address == 0 || !address.is_multiple_of(wire::COMPLETION_ALIGNMENT) {
            return None;
        }
        let completion_event = u32::from_le_bytes(
            data[wire::common::COMPLETION_EVENT..wire::common::COMPLETION_EVENT + 4]
                .try_into()
                .unwrap(),
        );
        Some(Self {
            address,
            completion_event,
        })
    }
}

fn completion_bytes(status: StateStatus, completion_event: u32, elapsed_cycles: u64) -> [u8; 16] {
    let mut data = [0u8; wire::COMPLETION_SIZE];
    data[wire::completion::STATUS..wire::completion::STATUS + 4]
        .copy_from_slice(&(status as u32).to_le_bytes());
    data[wire::completion::COMPLETION_EVENT..wire::completion::COMPLETION_EVENT + 4]
        .copy_from_slice(&completion_event.to_le_bytes());
    data[wire::completion::ELAPSED_CYCLES..wire::completion::ELAPSED_CYCLES + 8]
        .copy_from_slice(&elapsed_cycles.to_le_bytes());
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;
    use memory::{MemoryBacked, MemoryModel};
    use quantize::{DataType, FpType, QuantTensor, tensor_from_f32_slice, tensor_to_f32_vec};

    fn test_vram() -> Arc<VectorSram> {
        Arc::new(VectorSram::new(8, 128, DataType::Fp(FpType::BF16), 4))
    }

    fn put_u16(data: &mut [u8], offset: usize, value: u16) {
        data[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(data: &mut [u8], offset: usize, value: u32) {
        data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(data: &mut [u8], offset: usize, value: u64) {
        data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn put_bf16(data: &mut [u8], offset: usize, values: &[f32]) {
        for (index, value) in values.iter().enumerate() {
            data[offset + 2 * index..offset + 2 * index + 2]
                .copy_from_slice(&bf16::from_f32(*value).to_bits().to_le_bytes());
        }
    }

    async fn write_vram_record(vram: &Arc<VectorSram>, address: u32, values: &[f32]) {
        let vlen = vram.tile_size() as usize;
        for (row, chunk) in values.chunks(vlen).enumerate() {
            let mut padded = vec![0.0; vlen];
            padded[..chunk.len()].copy_from_slice(chunk);
            vram.write(
                address + (row * vlen) as u32,
                QuantTensor::quantize(tensor_from_f32_slice(&padded), vram.ty()),
            )
            .await;
        }
    }

    async fn write_vram_blocked_records(vram: &Arc<VectorSram>, base: u32, records: &[&[f32]]) {
        let vlen = vram.tile_size() as usize;
        let chunk_rows = records.len();
        for (record, values) in records.iter().enumerate() {
            for (feature_tile, chunk) in values.chunks(vlen).enumerate() {
                let address = base + (feature_tile * chunk_rows * vlen + record * vlen) as u32;
                let mut padded = vec![0.0; vlen];
                padded[..chunk.len()].copy_from_slice(chunk);
                vram.write(
                    address,
                    QuantTensor::quantize(tensor_from_f32_slice(&padded), vram.ty()),
                )
                .await;
            }
        }
    }

    async fn read_vram_values(vram: &Arc<VectorSram>, address: u32, count: usize) -> Vec<f32> {
        let vlen = vram.tile_size() as usize;
        let mut result = Vec::with_capacity(count);
        for row in 0..count.div_ceil(vlen) {
            let tensor = vram.read(address + (row * vlen) as u32).await;
            result.extend(tensor_to_f32_vec(tensor.as_tensor()));
        }
        result.truncate(count);
        result
    }

    fn numeric_common(
        algorithm: wire::StateAlgorithm,
        state_bytes: u32,
        conv_state_bytes: u32,
    ) -> [u8; wire::DESCRIPTOR_SIZE] {
        let mut data = [0u8; wire::DESCRIPTOR_SIZE];
        put_u32(&mut data, wire::common::MAGIC, wire::DESCRIPTOR_MAGIC);
        put_u16(&mut data, wire::common::VERSION, wire::DESCRIPTOR_VERSION);
        put_u16(
            &mut data,
            wire::common::SIZE_BYTES,
            wire::DESCRIPTOR_SIZE as u16,
        );
        data[wire::common::ALGORITHM] = algorithm as u8;
        data[wire::common::STATE_PRECISION] = wire::StatePrecision::Fp32 as u8;
        data[wire::common::ACTIVATION_PRECISION] = wire::StatePrecision::Bf16 as u8;
        data[wire::common::ACCUMULATOR_PRECISION] = wire::StatePrecision::Fp32 as u8;
        data[wire::common::PARAMETER_PRECISION] = wire::StatePrecision::Bf16 as u8;
        put_u32(&mut data, wire::common::CONTEXT_ID, 7);
        put_u32(&mut data, wire::common::REQUEST_ID, 11);
        put_u32(&mut data, wire::common::LAYER_ID, 13);
        put_u16(&mut data, wire::common::BATCH_SIZE, 1);
        put_u16(&mut data, wire::common::NUM_HEADS, 1);
        put_u32(&mut data, wire::common::SEQUENCE_LENGTH, 1);
        put_u16(&mut data, wire::common::VALID_TOKENS, 1);
        put_u16(&mut data, wire::common::CHUNK_SIZE, 1);
        put_u32(
            &mut data,
            wire::common::STATE_SRAM_OFFSET,
            wire::STREAMING_SRAM_OFFSET,
        );
        put_u32(&mut data, wire::common::STATE_BYTES, state_bytes);
        put_u32(&mut data, wire::common::CONV_STATE_BYTES, conv_state_bytes);
        put_u32(&mut data, wire::common::INPUT_VRAM_ADDR, 0);
        put_u32(&mut data, wire::common::OUTPUT_VRAM_ADDR, 32);
        put_u32(&mut data, wire::common::INPUT_TOKEN_STRIDE, 16);
        put_u32(&mut data, wire::common::OUTPUT_TOKEN_STRIDE, 16);
        put_u64(&mut data, wire::common::STATE_HBM_ADDR, 1024);
        put_u64(&mut data, wire::common::CONV_STATE_HBM_ADDR, 1088);
        put_u32(&mut data, wire::common::DEPENDENCY_EVENT, wire::NO_EVENT);
        put_u32(&mut data, wire::common::COMPLETION_EVENT, wire::NO_EVENT);
        data
    }

    fn compiler_golden_descriptor(name: &str) -> [u8; wire::DESCRIPTOR_SIZE] {
        let golden: serde_json::Value =
            serde_json::from_str(include_str!("../../testdata/x_state_v2_golden.json")).unwrap();
        let hex = golden["descriptors"][name]["hex"].as_str().unwrap();
        let bytes = hex
            .as_bytes()
            .chunks_exact(2)
            .map(|digits| u8::from_str_radix(std::str::from_utf8(digits).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        bytes.try_into().unwrap()
    }

    fn compiler_layout_golden(name: &str) -> [u8; layout::DESCRIPTOR_SIZE] {
        let golden: serde_json::Value =
            serde_json::from_str(include_str!("../../testdata/l_scatter_m_v1_golden.json"))
                .unwrap();
        let hex = golden[name]["layout_hex"].as_str().unwrap();
        let bytes = hex
            .as_bytes()
            .chunks_exact(2)
            .map(|digits| u8::from_str_radix(std::str::from_utf8(digits).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        bytes.try_into().unwrap()
    }

    fn resident_descriptor() -> [u8; wire::DESCRIPTOR_SIZE] {
        let mut data = [0u8; wire::DESCRIPTOR_SIZE];
        put_u32(&mut data, wire::common::MAGIC, wire::DESCRIPTOR_MAGIC);
        put_u16(&mut data, wire::common::VERSION, wire::DESCRIPTOR_VERSION);
        put_u16(
            &mut data,
            wire::common::SIZE_BYTES,
            wire::DESCRIPTOR_SIZE as u16,
        );
        data[wire::common::ALGORITHM] = wire::StateAlgorithm::Mamba2 as u8;
        data[wire::common::STATE_PRECISION] = wire::StatePrecision::Fp32 as u8;
        data[wire::common::ACTIVATION_PRECISION] = wire::StatePrecision::Bf16 as u8;
        data[wire::common::ACCUMULATOR_PRECISION] = wire::StatePrecision::Fp32 as u8;
        put_u32(&mut data, wire::common::FLAGS, wire::FLAG_WRITE_COMPLETION);
        put_u32(&mut data, wire::common::CONTEXT_ID, 7);
        put_u32(&mut data, wire::common::REQUEST_ID, 8);
        put_u32(&mut data, wire::common::LAYER_ID, 9);
        put_u16(&mut data, wire::common::BATCH_SIZE, 1);
        put_u16(&mut data, wire::common::NUM_HEADS, 1);
        put_u32(&mut data, wire::common::SEQUENCE_LENGTH, 1);
        put_u16(&mut data, wire::common::VALID_TOKENS, 1);
        put_u16(&mut data, wire::common::CHUNK_SIZE, 1);
        put_u32(&mut data, wire::common::STATE_SRAM_OFFSET, 0);
        put_u32(&mut data, wire::common::STATE_BYTES, 16);
        put_u32(&mut data, wire::common::CONV_STATE_BYTES, 24);
        put_u32(&mut data, wire::common::INPUT_TOKEN_STRIDE, 9);
        put_u32(&mut data, wire::common::OUTPUT_TOKEN_STRIDE, 2);
        put_u64(&mut data, wire::common::STATE_HBM_ADDR, 1024);
        put_u64(&mut data, wire::common::CONV_STATE_HBM_ADDR, 1088);
        put_u64(&mut data, wire::common::COMPLETION_ADDR, 512);
        put_u32(&mut data, wire::common::DEPENDENCY_EVENT, wire::NO_EVENT);
        put_u32(&mut data, wire::common::COMPLETION_EVENT, 123);
        put_u16(&mut data, wire::mamba2::HEAD_DIM, 2);
        put_u16(&mut data, wire::mamba2::STATE_DIM, 2);
        put_u16(&mut data, wire::mamba2::GROUPS, 1);
        put_u16(&mut data, wire::mamba2::CONV_KERNEL, 1);
        put_u32(&mut data, wire::mamba2::XBC_OFFSET, 2);
        put_u32(&mut data, wire::mamba2::DT_OFFSET, 8);
        put_u32(
            &mut data,
            wire::mamba2::DT_MAX_F32_BITS,
            f32::INFINITY.to_bits(),
        );
        data
    }

    fn command(subop: wire::StateSubop) -> EncodedStateCommand {
        EncodedStateCommand {
            context_gp: 1,
            descriptor_offset_gp: 2,
            descriptor_hbm_reg: 0,
            queue_id: 3,
            subop: subop as u8,
            context_id: 7,
            descriptor_address: Some(0),
        }
    }

    #[tokio::test]
    async fn descriptor_fetch_lifecycle_and_completion_are_end_to_end() {
        let backing = Arc::new(MemoryBacked::with_capacity(4096));
        let descriptor = resident_descriptor();
        backing.with_data(|data| {
            data[..wire::DESCRIPTOR_SIZE].copy_from_slice(&descriptor);
            data[1024..1040].fill(0x11);
            data[1088..1112].fill(0x22);
        });
        let hbm: Arc<dyn ErasedMemoryModel> = backing.clone();
        let mut engine = StateEngine::new(hbm.clone(), test_vram());

        assert_eq!(
            engine.execute(command(wire::StateSubop::Preload)).await,
            StateStatus::Success
        );
        let completion = read_bytes(&hbm, 512, wire::COMPLETION_SIZE).await;
        assert_eq!(u32::from_le_bytes(completion[0..4].try_into().unwrap()), 1);
        assert_eq!(
            u32::from_le_bytes(completion[4..8].try_into().unwrap()),
            123
        );

        assert_eq!(
            engine.execute(command(wire::StateSubop::Reset)).await,
            StateStatus::Success
        );
        assert_eq!(
            engine.execute(command(wire::StateSubop::Evict)).await,
            StateStatus::StateHazard
        );
        assert_eq!(
            engine.execute(command(wire::StateSubop::Commit)).await,
            StateStatus::Success
        );
        assert_eq!(backing.read(1024).await[..16], [0u8; 16]);
        assert_eq!(backing.read(1088).await[..16], [0u8; 16]);
        assert_eq!(
            engine.execute(command(wire::StateSubop::Evict)).await,
            StateStatus::Success
        );
    }

    #[tokio::test]
    async fn fence_is_descriptor_less_and_context_is_checked() {
        let backing = Arc::new(MemoryBacked::with_capacity(4096));
        backing.with_data(|data| {
            data[..wire::DESCRIPTOR_SIZE].copy_from_slice(&resident_descriptor())
        });
        let hbm: Arc<dyn ErasedMemoryModel> = backing;
        let mut engine = StateEngine::new(hbm, test_vram());
        assert_eq!(
            engine
                .execute(EncodedStateCommand {
                    context_gp: 0,
                    descriptor_offset_gp: 0,
                    descriptor_hbm_reg: 0,
                    queue_id: 2,
                    subop: wire::StateSubop::Fence as u8,
                    context_id: 0,
                    descriptor_address: None,
                })
                .await,
            StateStatus::Success
        );
        let mut wrong_context = command(wire::StateSubop::Preload);
        wrong_context.context_id = 99;
        assert_eq!(
            engine.execute(wrong_context).await,
            StateStatus::StateHazard
        );
    }

    #[tokio::test]
    async fn mamba_step_matches_python_core_reference() {
        let backing = Arc::new(MemoryBacked::with_capacity(8192));
        let descriptor = compiler_golden_descriptor("mamba2_tiny");
        backing.with_data(|data| {
            data[..wire::DESCRIPTOR_SIZE].copy_from_slice(&descriptor);
            put_bf16(data, 2048, &[0.0, 1.0].repeat(6));
            put_bf16(data, 2112, &[0.0]);
            put_bf16(data, 2176, &[0.0]);
            put_bf16(data, 2240, &[0.5]);
        });
        let hbm: Arc<dyn ErasedMemoryModel> = backing.clone();
        let vram = test_vram();
        write_vram_record(&vram, 0, &[0.0, 0.0, 1.0, 2.0, 0.5, -0.25, 0.2, 0.3, 0.0]).await;
        let mut engine = StateEngine::new(hbm.clone(), vram.clone());
        assert_eq!(
            engine.execute(command(wire::StateSubop::Step)).await,
            StateStatus::Success
        );

        assert_eq!(
            read_vram_values(&vram, 64, 2).await,
            [0.373_046_88, 0.8984375]
        );
        let state_bytes = read_bytes(&hbm, 1024, 16).await;
        let state =
            precision::decode_tensor(&state_bytes, &[], wire::StatePrecision::Fp32, 2).unwrap();
        let expected = [0.15770979, -0.055464707, 0.38002512, -0.13365044];
        for (actual, expected) in state.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
        }
    }

    #[tokio::test]
    async fn a_nan_dt_limit_is_rejected_rather_than_panicking() {
        // Descriptors come from untrusted HBM bytes, so a malformed dt limit has
        // to surface as a status. dt_max is compared with `dt_max < dt_min`,
        // which is false for NaN, and only dt_min is NaN-checked, so a NaN
        // dt_max reaches f32::clamp - and clamp panics on a NaN bound.
        let backing = Arc::new(MemoryBacked::with_capacity(8192));
        let mut descriptor = compiler_golden_descriptor("mamba2_tiny");
        put_u32(
            &mut descriptor,
            wire::mamba2::DT_MAX_F32_BITS,
            f32::NAN.to_bits(),
        );
        backing.with_data(|data| {
            data[..wire::DESCRIPTOR_SIZE].copy_from_slice(&descriptor);
        });
        let hbm: Arc<dyn ErasedMemoryModel> = backing.clone();
        let vram = test_vram();
        write_vram_record(&vram, 0, &[0.0; 9]).await;
        let mut engine = StateEngine::new(hbm.clone(), vram.clone());
        assert_eq!(
            engine.execute(command(wire::StateSubop::Step)).await,
            StateStatus::InvalidDescriptor
        );
    }

    #[tokio::test]
    async fn mamba_step_honours_an_independent_conv_state_precision() {
        // state_precision and conv_state_precision are separate wire fields, so
        // a descriptor may keep the recurrent state in FP32 while storing the
        // convolution window in BF16. Decoding either buffer with the other's
        // precision derives the element count from the wrong element size.
        let backing = Arc::new(MemoryBacked::with_capacity(8192));
        let mut descriptor = compiler_golden_descriptor("mamba2_tiny");
        // 6 channels x 2 taps, so the BF16 window is 24 bytes where FP32 is 48.
        descriptor[wire::common::CONV_STATE_PRECISION] = wire::StatePrecision::Bf16 as u8;
        put_u32(&mut descriptor, wire::common::CONV_STATE_BYTES, 24);
        backing.with_data(|data| {
            data[..wire::DESCRIPTOR_SIZE].copy_from_slice(&descriptor);
            put_bf16(data, 2048, &[0.0, 1.0].repeat(6));
            put_bf16(data, 2112, &[0.0]);
            put_bf16(data, 2176, &[0.0]);
            put_bf16(data, 2240, &[0.5]);
        });
        let hbm: Arc<dyn ErasedMemoryModel> = backing.clone();
        let vram = test_vram();
        write_vram_record(&vram, 0, &[0.0, 0.0, 1.0, 2.0, 0.5, -0.25, 0.2, 0.3, 0.0]).await;
        let mut engine = StateEngine::new(hbm.clone(), vram.clone());
        assert_eq!(
            engine.execute(command(wire::StateSubop::Step)).await,
            StateStatus::Success
        );

        // The convolution window starts at zero, which is exact in both
        // formats, so the first token reproduces the all-FP32 reference step.
        let state_bytes = read_bytes(&hbm, 1024, 16).await;
        let state =
            precision::decode_tensor(&state_bytes, &[], wire::StatePrecision::Fp32, 2).unwrap();
        let expected = [0.15770979, -0.055464707, 0.38002512, -0.13365044];
        for (actual, expected) in state.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
        }
    }

    #[tokio::test]
    async fn mamba_step_executes_mx8_parameters_and_state_scale_streams() {
        let backing = Arc::new(MemoryBacked::with_capacity(8192));
        let mut descriptor = compiler_golden_descriptor("mamba2_tiny");
        descriptor[wire::common::STATE_PRECISION] = wire::StatePrecision::Mx8B128 as u8;
        descriptor[wire::common::CONV_STATE_PRECISION] = wire::StatePrecision::Mx8B128 as u8;
        descriptor[wire::common::PARAMETER_PRECISION] = wire::StatePrecision::Mx8B128 as u8;
        put_u32(&mut descriptor, wire::common::STATE_BYTES, 4);
        put_u32(&mut descriptor, wire::common::CONV_STATE_BYTES, 12);
        put_u64(&mut descriptor, wire::common::STATE_SCALE_ADDR, 1152);
        put_u32(&mut descriptor, wire::common::STATE_SCALE_BYTES, 2);
        put_u32(&mut descriptor, wire::common::CONV_STATE_SCALE_BYTES, 6);
        put_u64(&mut descriptor, wire::mamba2::PARAMETER_SCALE_ADDR, 3072);

        let tensors = [
            (2048, [0.0, 1.0].repeat(6), 2),
            (2112, vec![0.0], 1),
            (2176, vec![0.0], 1),
            (2240, vec![0.5], 1),
        ];
        let encoded = tensors
            .iter()
            .map(|(_, values, inner)| {
                precision::encode_tensor(values, wire::StatePrecision::Mx8B128, *inner).unwrap()
            })
            .collect::<Vec<_>>();
        backing.with_data(|data| {
            data[..wire::DESCRIPTOR_SIZE].copy_from_slice(&descriptor);
            let mut scale_offset = 3072;
            for ((address, _, _), tensor) in tensors.iter().zip(&encoded) {
                data[*address..*address + tensor.values.len()].copy_from_slice(&tensor.values);
                data[scale_offset..scale_offset + tensor.scales.len()]
                    .copy_from_slice(&tensor.scales);
                scale_offset += tensor.scales.len();
            }
        });
        let hbm: Arc<dyn ErasedMemoryModel> = backing.clone();
        let vram = test_vram();
        write_vram_record(&vram, 0, &[0.0, 0.0, 1.0, 2.0, 0.5, -0.25, 0.2, 0.3, 0.0]).await;
        let mut engine = StateEngine::new(hbm.clone(), vram.clone());
        assert_eq!(
            engine.execute(command(wire::StateSubop::Step)).await,
            StateStatus::Success
        );
        assert_eq!(
            read_vram_values(&vram, 64, 2).await,
            [0.373_046_88, 0.8984375]
        );
        let state_values = read_bytes(&hbm, 1024, 4).await;
        let state_scales = read_bytes(&hbm, 1152, 2).await;
        let state = precision::decode_tensor(
            &state_values,
            &state_scales,
            wire::StatePrecision::Mx8B128,
            2,
        )
        .unwrap();
        assert_eq!(state.len(), 4);
        assert!(state.iter().any(|value| *value != 0.0));
    }

    #[tokio::test]
    async fn kda_step_honours_an_independent_conv_state_precision() {
        // KdaScheduleConfig now ships FP32 recurrent state with a BF16
        // convolution window, so the split-precision path is KDA's default. The
        // golden that carries that combination (`kda_real`) is only ever parsed,
        // never executed, and `kda_tiny` keeps both sides FP32 - so nothing
        // executed the combination the compiler actually emits.
        let backing = Arc::new(MemoryBacked::with_capacity(8192));
        let mut descriptor = compiler_golden_descriptor("kda_tiny");
        // 6 channels x 2 taps: 24 BF16 bytes where FP32 needs 48.
        descriptor[wire::common::CONV_STATE_PRECISION] = wire::StatePrecision::Bf16 as u8;
        put_u32(&mut descriptor, wire::common::CONV_STATE_BYTES, 24);
        backing.with_data(|data| {
            data[..wire::DESCRIPTOR_SIZE].copy_from_slice(&descriptor);
            put_bf16(data, 2048, &[0.0, 1.0].repeat(2));
            put_bf16(data, 2112, &[0.0, 1.0].repeat(2));
            put_bf16(data, 2176, &[0.0, 1.0].repeat(2));
            put_bf16(data, 2240, &[0.0]);
            put_bf16(data, 2304, &[0.0, 0.25]);
        });
        let hbm: Arc<dyn ErasedMemoryModel> = backing.clone();
        let vram = test_vram();
        write_vram_record(&vram, 0, &[1.0, 2.0, 0.5, -1.0, 0.25, 1.5, 0.1, -0.2, 0.3]).await;
        let mut engine = StateEngine::new(hbm.clone(), vram.clone());
        assert_eq!(
            engine.execute(command(wire::StateSubop::Step)).await,
            StateStatus::Success
        );

        // The window starts at zero, exact in both formats, so the recurrent
        // result matches the all-FP32 reference step.
        let state_bytes = read_bytes(&hbm, 1024, 16).await;
        let state =
            precision::decode_tensor(&state_bytes, &[], wire::StatePrecision::Fp32, 2).unwrap();
        let expected = [0.061107103, -0.052804194, 0.5332091, -0.46075943];
        for (actual, expected) in state.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
        }
    }

    #[tokio::test]
    async fn kda_step_matches_python_core_reference() {
        let backing = Arc::new(MemoryBacked::with_capacity(8192));
        let descriptor = compiler_golden_descriptor("kda_tiny");
        backing.with_data(|data| {
            data[..wire::DESCRIPTOR_SIZE].copy_from_slice(&descriptor);
            put_bf16(data, 2048, &[0.0, 1.0].repeat(2));
            put_bf16(data, 2112, &[0.0, 1.0].repeat(2));
            put_bf16(data, 2176, &[0.0, 1.0].repeat(2));
            put_bf16(data, 2240, &[0.0]);
            put_bf16(data, 2304, &[0.0, 0.25]);
        });
        let hbm: Arc<dyn ErasedMemoryModel> = backing.clone();
        let vram = test_vram();
        write_vram_record(&vram, 0, &[1.0, 2.0, 0.5, -1.0, 0.25, 1.5, 0.1, -0.2, 0.3]).await;
        let mut engine = StateEngine::new(hbm.clone(), vram.clone());
        assert_eq!(
            engine.execute(command(wire::StateSubop::Step)).await,
            StateStatus::Success
        );

        assert_eq!(
            read_vram_values(&vram, 64, 2).await,
            [-0.0126953125, -0.110_351_56]
        );
        let state_bytes = read_bytes(&hbm, 1024, 16).await;
        let state =
            precision::decode_tensor(&state_bytes, &[], wire::StatePrecision::Fp32, 2).unwrap();
        let expected = [0.061107103, -0.052804194, 0.5332091, -0.46075943];
        for (actual, expected) in state.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
        }
    }

    #[tokio::test]
    async fn l_scatter_m_to_mamba_x_state_is_numerically_connected() {
        let backing = Arc::new(MemoryBacked::with_capacity(8192));
        let state_descriptor = compiler_golden_descriptor("mamba2_tiny");
        let layout_descriptor = compiler_layout_golden("mamba2_tiny");
        backing.with_data(|data| {
            data[..wire::DESCRIPTOR_SIZE].copy_from_slice(&state_descriptor);
            data[256..256 + layout::DESCRIPTOR_SIZE].copy_from_slice(&layout_descriptor);
            put_bf16(data, 2048, &[0.0, 1.0].repeat(6));
            put_bf16(data, 2112, &[0.0]);
            put_bf16(data, 2176, &[0.0]);
            put_bf16(data, 2240, &[0.5]);
        });
        let hbm: Arc<dyn ErasedMemoryModel> = backing;
        let vram = test_vram();
        write_vram_record(&vram, 0, &[0.0, 0.0, 1.0, 2.0, 0.5, -0.25, 0.2, 0.3, 0.0]).await;
        let mut engine = StateEngine::new(hbm.clone(), vram.clone());
        assert_eq!(
            engine
                .execute_layout(EncodedLayoutCommand {
                    context_gp: 1,
                    descriptor_offset_gp: 2,
                    descriptor_hbm_reg: 0,
                    buffer_id: 0,
                    mode: LayoutMode::MambaSkew as u8,
                    context_id: 7,
                    descriptor_address: Some(256),
                })
                .await,
            StateStatus::Success
        );
        assert_eq!(engine.take_charged_layout_cycles(), 4);
        assert_eq!(
            engine.execute(command(wire::StateSubop::Step)).await,
            StateStatus::Success
        );

        assert_eq!(
            read_vram_values(&vram, 64, 2).await,
            [0.373_046_88, 0.8984375]
        );
        let state_bytes = read_bytes(&hbm, 1024, 16).await;
        let state =
            precision::decode_tensor(&state_bytes, &[], wire::StatePrecision::Fp32, 2).unwrap();
        let expected = [0.15770979, -0.055464707, 0.38002512, -0.13365044];
        for (actual, expected) in state.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
        }
        let summary = engine.profile.lock().unwrap().summary();
        assert_eq!(summary.projection_buffer_values, 9);
        assert_eq!(summary.projection_fifo_spill_values, 2);
        assert_eq!(summary.projection_write_service_cycles, 4);
        assert!(summary.projection_read_packets > 0);
    }

    #[tokio::test]
    async fn l_scatter_m_to_kda_x_state_is_numerically_connected() {
        let backing = Arc::new(MemoryBacked::with_capacity(8192));
        let state_descriptor = compiler_golden_descriptor("kda_tiny");
        let layout_descriptor = compiler_layout_golden("kda_tiny");
        backing.with_data(|data| {
            data[..wire::DESCRIPTOR_SIZE].copy_from_slice(&state_descriptor);
            data[256..256 + layout::DESCRIPTOR_SIZE].copy_from_slice(&layout_descriptor);
            put_bf16(data, 2048, &[0.0, 1.0].repeat(2));
            put_bf16(data, 2112, &[0.0, 1.0].repeat(2));
            put_bf16(data, 2176, &[0.0, 1.0].repeat(2));
            put_bf16(data, 2240, &[0.0]);
            put_bf16(data, 2304, &[0.0, 0.25]);
        });
        let hbm: Arc<dyn ErasedMemoryModel> = backing;
        let vram = test_vram();
        write_vram_record(&vram, 0, &[1.0, 2.0, 0.5, -1.0, 0.25, 1.5, 0.1, -0.2, 0.3]).await;
        let mut engine = StateEngine::new(hbm.clone(), vram.clone());
        assert_eq!(
            engine
                .execute_layout(EncodedLayoutCommand {
                    context_gp: 1,
                    descriptor_offset_gp: 2,
                    descriptor_hbm_reg: 0,
                    buffer_id: 0,
                    mode: LayoutMode::KdaSkew as u8,
                    context_id: 7,
                    descriptor_address: Some(256),
                })
                .await,
            StateStatus::Success
        );
        assert_eq!(engine.take_charged_layout_cycles(), 4);
        assert_eq!(
            engine.execute(command(wire::StateSubop::Step)).await,
            StateStatus::Success
        );

        assert_eq!(
            read_vram_values(&vram, 64, 2).await,
            [-0.0126953125, -0.110_351_56]
        );
        let state_bytes = read_bytes(&hbm, 1024, 16).await;
        let state =
            precision::decode_tensor(&state_bytes, &[], wire::StatePrecision::Fp32, 2).unwrap();
        let expected = [0.061107103, -0.052804194, 0.5332091, -0.46075943];
        for (actual, expected) in state.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
        }
        let summary = engine.profile.lock().unwrap().summary();
        assert_eq!(summary.projection_buffer_values, 9);
        assert_eq!(summary.projection_fifo_spill_values, 9);
        assert_eq!(summary.projection_write_service_cycles, 4);
        assert!(summary.projection_read_packets > 0);
    }

    #[tokio::test]
    async fn legacy_matrix_writeback_runs_without_fabricating_layout_stats() {
        use crate::matrix_machine::MatrixMachine;
        use runtime::{Executor, Instant};
        use sram::MatrixSram;

        let backing = Arc::new(MemoryBacked::with_capacity(8192));
        let mut descriptor = compiler_golden_descriptor("kda_tiny");
        // BLEN=2 reserves two physical rows even though decode has one live row.
        // The second feature tile therefore starts at element address 16.
        put_u16(&mut descriptor, wire::common::CHUNK_SIZE, 2);
        backing.with_data(|data| {
            data[..wire::DESCRIPTOR_SIZE].copy_from_slice(&descriptor);
            put_bf16(data, 2048, &[0.0, 1.0].repeat(2));
            put_bf16(data, 2112, &[0.0, 1.0].repeat(2));
            put_bf16(data, 2176, &[0.0, 1.0].repeat(2));
            put_bf16(data, 2240, &[0.0]);
            put_bf16(data, 2304, &[0.0, 0.25]);
        });
        let hbm: Arc<dyn ErasedMemoryModel> = backing;
        let vram = test_vram();
        let mram = Arc::new(MatrixSram::new(8, 64, vram.ty()));
        let identity = (0..8)
            .flat_map(|row| (0..8).map(move |column| f32::from(row == column)))
            .collect::<Vec<_>>();
        mram.write(
            0,
            QuantTensor::quantize(tensor_from_f32_slice(&identity), vram.ty()),
        )
        .await;

        // Two identity projections produce the nine live KDA fields. The
        // zero row is physical BLEN padding and must never become a live token.
        write_vram_record(&vram, 128, &[1.0, 2.0, 0.5, -1.0, 0.25, 1.5, 0.1, -0.2]).await;
        write_vram_record(&vram, 136, &[0.0; 8]).await;
        write_vram_record(&vram, 144, &[0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).await;
        write_vram_record(&vram, 152, &[0.0; 8]).await;

        let result = Arc::new(StdMutex::new(Vec::new()));
        let task_result = result.clone();
        let executor = Executor::new();
        executor.spawn(async move {
            let mut matrix = MatrixMachine::new(mram, vram.clone(), 8, 2, 2, 4);
            for column in (0..8).step_by(2) {
                matrix.mm(column * 8, 128).await;
                matrix.mm_wo(column, 1).await;
            }
            matrix.mm(0, 144).await;
            matrix.mm_wo(16, 1).await;

            let mut engine = StateEngine::new(hbm, vram.clone());
            assert_eq!(
                engine.execute(command(wire::StateSubop::Step)).await,
                StateStatus::Success
            );
            let summary = engine.profile.lock().unwrap().summary();
            assert_eq!(summary.projection_buffer_values, 0);
            assert_eq!(summary.projection_read_packets, 0);
            let output = read_vram_values(&vram, 64, 2).await;
            task_result.lock().unwrap().extend(output);
        });
        executor.enter(Instant::ETERNITY).await;
        assert_eq!(*result.lock().unwrap(), [-0.0126953125, -0.110_351_56]);
    }

    #[tokio::test]
    async fn mamba_prefill_is_identical_to_repeated_step() {
        let backing = Arc::new(MemoryBacked::with_capacity(8192));
        let mut raw = compiler_golden_descriptor("mamba2_tiny");
        put_u32(&mut raw, wire::common::SEQUENCE_LENGTH, 2);
        put_u16(&mut raw, wire::common::VALID_TOKENS, 2);
        put_u16(&mut raw, wire::common::CHUNK_SIZE, 2);
        put_u32(&mut raw, wire::common::OUTPUT_VRAM_ADDR, 128);
        backing.with_data(|data| {
            put_bf16(data, 2048, &[0.0, 1.0].repeat(6));
            put_bf16(data, 2112, &[0.0]);
            put_bf16(data, 2176, &[0.0]);
            put_bf16(data, 2240, &[0.5]);
        });
        let hbm: Arc<dyn ErasedMemoryModel> = backing;
        let projected = [
            [0.0, 0.0, 1.0, 2.0, 0.5, -0.25, 0.2, 0.3, 0.0],
            [0.0, 0.0, -0.5, 0.75, 0.1, 0.4, -0.2, 0.25, 0.2],
        ];
        let prefill_vram = test_vram();
        let step_vram = test_vram();
        write_vram_blocked_records(
            &prefill_vram,
            0,
            &[projected[0].as_slice(), projected[1].as_slice()],
        )
        .await;
        write_vram_record(&step_vram, 0, &projected[0]).await;
        let prefill_descriptor = StateDescriptor::parse(&raw).unwrap();
        let mut prefill_buffers = StateBuffers::zeros(&prefill_descriptor);
        let mut prefill_layouts = LayoutStore::default();
        functional::execute(
            &hbm,
            &prefill_vram,
            &mut prefill_layouts,
            &prefill_descriptor,
            &mut prefill_buffers,
        )
        .await
        .unwrap();

        let mut step_descriptor = prefill_descriptor.clone();
        step_descriptor.valid_tokens = 1;
        step_descriptor.chunk_size = 1;
        step_descriptor.output_vram_addr = 128;
        let mut step_buffers = StateBuffers::zeros(&step_descriptor);
        let mut step_layouts = LayoutStore::default();
        functional::execute(
            &hbm,
            &step_vram,
            &mut step_layouts,
            &step_descriptor,
            &mut step_buffers,
        )
        .await
        .unwrap();
        step_descriptor.token_offset = 1;
        step_descriptor.input_vram_addr = 64;
        step_descriptor.output_vram_addr = 136;
        write_vram_record(&step_vram, 64, &projected[1]).await;
        functional::execute(
            &hbm,
            &step_vram,
            &mut step_layouts,
            &step_descriptor,
            &mut step_buffers,
        )
        .await
        .unwrap();

        assert_eq!(prefill_buffers, step_buffers);
        assert_eq!(
            read_vram_values(&prefill_vram, 128, 2).await,
            read_vram_values(&step_vram, 128, 2).await
        );
        assert_eq!(
            read_vram_values(&prefill_vram, 136, 2).await,
            read_vram_values(&step_vram, 136, 2).await
        );
    }

    #[tokio::test]
    async fn kda_prefill_is_identical_to_repeated_step() {
        let backing = Arc::new(MemoryBacked::with_capacity(8192));
        let mut raw = compiler_golden_descriptor("kda_tiny");
        put_u32(&mut raw, wire::common::SEQUENCE_LENGTH, 2);
        put_u16(&mut raw, wire::common::VALID_TOKENS, 2);
        put_u16(&mut raw, wire::common::CHUNK_SIZE, 2);
        put_u32(&mut raw, wire::common::OUTPUT_VRAM_ADDR, 128);
        backing.with_data(|data| {
            put_bf16(data, 2048, &[0.0, 1.0].repeat(2));
            put_bf16(data, 2112, &[0.0, 1.0].repeat(2));
            put_bf16(data, 2176, &[0.0, 1.0].repeat(2));
            put_bf16(data, 2240, &[0.0]);
            put_bf16(data, 2304, &[0.0, 0.25]);
        });
        let hbm: Arc<dyn ErasedMemoryModel> = backing;
        let projected = [
            [1.0, 2.0, 0.5, -1.0, 0.25, 1.5, 0.1, -0.2, 0.3],
            [-0.5, 0.75, 1.25, 0.1, -0.4, 0.2, -0.3, 0.6, -0.1],
        ];
        let prefill_vram = test_vram();
        let step_vram = test_vram();
        write_vram_blocked_records(
            &prefill_vram,
            0,
            &[projected[0].as_slice(), projected[1].as_slice()],
        )
        .await;
        write_vram_record(&step_vram, 0, &projected[0]).await;
        let prefill_descriptor = StateDescriptor::parse(&raw).unwrap();
        let mut prefill_buffers = StateBuffers::zeros(&prefill_descriptor);
        let mut prefill_layouts = LayoutStore::default();
        functional::execute(
            &hbm,
            &prefill_vram,
            &mut prefill_layouts,
            &prefill_descriptor,
            &mut prefill_buffers,
        )
        .await
        .unwrap();

        let mut step_descriptor = prefill_descriptor.clone();
        step_descriptor.valid_tokens = 1;
        step_descriptor.chunk_size = 1;
        step_descriptor.output_vram_addr = 128;
        let mut step_buffers = StateBuffers::zeros(&step_descriptor);
        let mut step_layouts = LayoutStore::default();
        functional::execute(
            &hbm,
            &step_vram,
            &mut step_layouts,
            &step_descriptor,
            &mut step_buffers,
        )
        .await
        .unwrap();
        step_descriptor.token_offset = 1;
        step_descriptor.input_vram_addr = 64;
        step_descriptor.output_vram_addr = 136;
        write_vram_record(&step_vram, 64, &projected[1]).await;
        functional::execute(
            &hbm,
            &step_vram,
            &mut step_layouts,
            &step_descriptor,
            &mut step_buffers,
        )
        .await
        .unwrap();

        assert_eq!(prefill_buffers, step_buffers);
        assert_eq!(
            read_vram_values(&prefill_vram, 128, 2).await,
            read_vram_values(&step_vram, 128, 2).await
        );
        assert_eq!(
            read_vram_values(&prefill_vram, 136, 2).await,
            read_vram_values(&step_vram, 136, 2).await
        );
    }

    #[tokio::test]
    async fn asynchronous_issue_overlaps_independent_compute_until_fence() {
        use std::sync::Mutex;

        use runtime::{Executor, Instant};

        let backing = Arc::new(MemoryBacked::with_capacity(8192));
        let descriptor = compiler_golden_descriptor("mamba2_tiny");
        backing.with_data(|data| {
            data[..wire::DESCRIPTOR_SIZE].copy_from_slice(&descriptor);
            put_bf16(data, 2048, &[0.0, 1.0].repeat(6));
            put_bf16(data, 2112, &[0.0]);
            put_bf16(data, 2176, &[0.0]);
            put_bf16(data, 2240, &[0.5]);
        });
        let hbm: Arc<dyn ErasedMemoryModel> = backing;
        let vram = test_vram();
        write_vram_record(&vram, 0, &[0.0, 0.0, 1.0, 2.0, 0.5, -0.25, 0.2, 0.3, 0.0]).await;
        let parsed = StateDescriptor::parse(&descriptor).unwrap();
        // Pin a multi-cycle core instead of relying on the evolving default
        // geometry. The test needs work to remain in flight across the
        // independent one-cycle operation; its exact latency is not the API.
        let timing_config = StateTimingConfig {
            fma_lanes_per_head_lane: 1,
            ..StateTimingConfig::default()
        };
        let timing = timing::estimate(&parsed, wire::StateSubop::Step, timing_config);
        let charged_cycles = timing.charged_compute_cycles();
        assert!(charged_cycles > 1);

        let observed = Arc::new(Mutex::new(Vec::new()));
        let observed_task = observed.clone();
        let executor = Executor::new();
        executor.spawn(async move {
            let mut engine =
                StateEngine::with_mode(hbm, vram, DEFAULT_STATE_SRAM_BYTES, timing_config, true);
            assert_eq!(
                engine.execute(command(wire::StateSubop::Step)).await,
                StateStatus::Success
            );
            observed_task
                .lock()
                .unwrap()
                .push(Executor::current().now());
            crate::cycle!(1);
            assert_eq!(
                engine
                    .execute(EncodedStateCommand {
                        context_gp: 0,
                        descriptor_offset_gp: 0,
                        descriptor_hbm_reg: 0,
                        queue_id: 3,
                        subop: wire::StateSubop::Fence as u8,
                        context_id: 0,
                        descriptor_address: None,
                    })
                    .await,
                StateStatus::Success
            );
            observed_task
                .lock()
                .unwrap()
                .push(Executor::current().now());
        });
        executor.enter(Instant::ETERNITY).await;

        let observed = observed.lock().unwrap();
        assert_eq!(observed[0], Instant::INIT);
        assert_eq!(
            observed[1],
            Instant::INIT + crate::runtime_config::PERIOD * charged_cycles as u32
        );
        assert_eq!(
            executor.now(),
            Instant::INIT + crate::runtime_config::PERIOD * charged_cycles as u32
        );
    }

    #[tokio::test]
    async fn asynchronous_dependency_event_waits_for_an_earlier_issued_producer() {
        use runtime::{Executor, Instant};

        let backing = Arc::new(MemoryBacked::with_capacity(8192));
        let mut producer = numeric_common(wire::StateAlgorithm::Mamba2, 16, 24);
        put_u16(&mut producer, wire::mamba2::HEAD_DIM, 2);
        put_u16(&mut producer, wire::mamba2::STATE_DIM, 2);
        put_u16(&mut producer, wire::mamba2::GROUPS, 1);
        put_u16(&mut producer, wire::mamba2::CONV_KERNEL, 1);
        put_u32(&mut producer, wire::mamba2::XBC_OFFSET, 2);
        put_u32(&mut producer, wire::mamba2::DT_OFFSET, 8);
        put_u32(
            &mut producer,
            wire::mamba2::DT_MAX_F32_BITS,
            f32::INFINITY.to_bits(),
        );
        put_u32(&mut producer, wire::common::COMPLETION_EVENT, 41);

        let mut dependent = producer;
        put_u32(&mut dependent, wire::common::DEPENDENCY_EVENT, 41);
        put_u32(&mut dependent, wire::common::COMPLETION_EVENT, 42);
        put_u64(&mut dependent, wire::common::STATE_HBM_ADDR, 4096);
        put_u64(&mut dependent, wire::common::CONV_STATE_HBM_ADDR, 4160);
        backing.with_data(|data| {
            data[..wire::DESCRIPTOR_SIZE].copy_from_slice(&producer);
            data[wire::DESCRIPTOR_SIZE..2 * wire::DESCRIPTOR_SIZE].copy_from_slice(&dependent);
            data[4096..4112].fill(0xff);
            data[4160..4184].fill(0xff);
        });
        let hbm: Arc<dyn ErasedMemoryModel> = backing.clone();
        let executor = Executor::new();
        executor.spawn(async move {
            let mut engine = StateEngine::with_mode(
                hbm,
                test_vram(),
                DEFAULT_STATE_SRAM_BYTES,
                StateTimingConfig::default(),
                true,
            );
            let mut producer_command = command(wire::StateSubop::Reset);
            producer_command.queue_id = 0;
            assert_eq!(engine.execute(producer_command).await, StateStatus::Success);
            let mut dependent_command = command(wire::StateSubop::Reset);
            dependent_command.queue_id = 1;
            dependent_command.descriptor_address = Some(wire::DESCRIPTOR_SIZE as u64);
            assert_eq!(
                engine.execute(dependent_command).await,
                StateStatus::Success
            );
            assert_eq!(engine.fence_all().await, StateStatus::Success);
        });
        executor.enter(Instant::ETERNITY).await;
        assert_eq!(backing.read(4096).await[..16], [0u8; 16]);
        assert_eq!(backing.read(4160).await[..16], [0u8; 16]);
    }

    #[tokio::test]
    async fn compiler_bytes_execute_through_accelerator_dispatch() {
        use crate::accelerator::Accelerator;
        use crate::matrix_machine::MatrixMachine;
        use crate::op::Opcode;
        use crate::vector_machine::VectorMachine;
        use runtime::{Executor, Instant};
        use sram::MatrixSram;

        let backing = Arc::new(MemoryBacked::with_capacity(8192));
        let descriptor = compiler_golden_descriptor("mamba2_tiny");
        backing.with_data(|data| {
            data[..wire::DESCRIPTOR_SIZE].copy_from_slice(&descriptor);
            put_bf16(data, 2048, &[0.0, 1.0].repeat(6));
            put_bf16(data, 2112, &[0.0]);
            put_bf16(data, 2176, &[0.0]);
            put_bf16(data, 2240, &[0.5]);
        });
        let hbm: Arc<dyn ErasedMemoryModel> = backing;
        let vram = test_vram();
        write_vram_record(&vram, 0, &[0.0, 0.0, 1.0, 2.0, 0.5, -0.25, 0.2, 0.3, 0.0]).await;
        let mram = Arc::new(MatrixSram::new(
            8,
            64,
            quantize::MxDataType::Plain(DataType::Fp(FpType::BF16)),
        ));
        let m_machine = MatrixMachine::new(mram, vram.clone(), 8, 2, 2, 4);
        let v_machine = VectorMachine::new(vram.clone(), 8, 2);

        let result = Arc::new(StdMutex::new(Vec::new()));
        let task_result = result.clone();
        let executor = Executor::new();
        executor.spawn(async move {
            let mut accelerator = Accelerator::new(
                m_machine,
                v_machine,
                hbm,
                DEFAULT_STATE_SRAM_BYTES,
                StateTimingConfig::default(),
                true,
            );
            let set_context = 0x22 | (1 << 6) | (7 << 14);
            let step = 0x3d | (1 << 6) | (2 << 10) | (3 << 18) | (3 << 22);
            let fence = 0x3d | (3 << 18) | (6 << 22);
            let ops = [
                Opcode::decode(set_context),
                Opcode::decode(step),
                Opcode::decode(fence),
            ];
            accelerator.do_ops(&ops, None, None).await;
            let output = read_vram_values(&vram, 64, 2).await;
            task_result.lock().unwrap().extend(output);
        });
        executor.enter(Instant::ETERNITY).await;
        assert_eq!(*result.lock().unwrap(), [0.373_046_88, 0.8984375]);
    }
}
