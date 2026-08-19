mod kda;
mod mamba;

use std::sync::Arc;

use memory::ErasedMemoryModel;
use sram::VectorSram;

use super::cache::StateBuffers;
use super::descriptor::{StateDescriptor, StatePayload};
use super::error::StateEngineError;
use super::generated_contract::StatePrecision;
use super::hbm::read_bytes;
use super::layout::LayoutStore;
use super::precision::{decode_tensor, scale_count};
use super::projection::ProjectionBufferStats;

pub async fn execute(
    hbm: &Arc<dyn ErasedMemoryModel>,
    vram: &Arc<VectorSram>,
    layouts: &mut LayoutStore,
    descriptor: &StateDescriptor,
    buffers: &mut StateBuffers,
) -> Result<ProjectionBufferStats, StateEngineError> {
    match &descriptor.payload {
        StatePayload::Mamba2(payload) => {
            mamba::execute(hbm, vram, layouts, descriptor, payload, buffers).await
        }
        StatePayload::Kda(payload) => {
            kda::execute(hbm, vram, layouts, descriptor, payload, buffers).await
        }
    }
}

struct ParameterReader<'a> {
    hbm: &'a Arc<dyn ErasedMemoryModel>,
    precision: StatePrecision,
    scale_address: u64,
}

impl<'a> ParameterReader<'a> {
    fn new(
        hbm: &'a Arc<dyn ErasedMemoryModel>,
        precision: StatePrecision,
        scale_address: u64,
    ) -> Self {
        Self {
            hbm,
            precision,
            scale_address,
        }
    }

    async fn required(
        &mut self,
        name: &str,
        address: u64,
        elements: usize,
        inner_dim: usize,
    ) -> Result<Vec<f32>, StateEngineError> {
        if address == 0 {
            return Err(StateEngineError::invalid(format!(
                "{name} requires a nonzero HBM address"
            )));
        }
        self.read(address, elements, inner_dim).await
    }

    async fn optional(
        &mut self,
        address: u64,
        elements: usize,
        inner_dim: usize,
    ) -> Result<Option<Vec<f32>>, StateEngineError> {
        if address == 0 {
            return Ok(None);
        }
        self.read(address, elements, inner_dim).await.map(Some)
    }

    async fn read(
        &mut self,
        address: u64,
        elements: usize,
        inner_dim: usize,
    ) -> Result<Vec<f32>, StateEngineError> {
        let value_bytes = elements
            .checked_mul(self.precision.element_bytes())
            .ok_or_else(|| StateEngineError::address("parameter byte count overflows"))?;
        let values = read_bytes(self.hbm, address, value_bytes).await;
        let scales = if self.precision == StatePrecision::Mx8B128 {
            let count = scale_count(elements, inner_dim)?;
            let bytes = read_bytes(self.hbm, self.scale_address, count).await;
            self.scale_address = self
                .scale_address
                .checked_add(count as u64)
                .ok_or_else(|| StateEngineError::address("parameter scale address overflows"))?;
            bytes
        } else {
            Vec::new()
        };
        decode_tensor(&values, &scales, self.precision, inner_dim)
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}
