use std::sync::Arc;

use quantize::{
    DataType, FpType, MxDataType, QuantTensor, tensor_from_f32_slice, tensor_to_f32_vec,
};
use sram::VectorSram;

use super::descriptor::StateDescriptor;
use super::error::StateEngineError;
use super::generated_contract::StatePrecision;
use super::layout::LayoutStore;
use super::projection::ProjectionBufferStats;

pub struct VramAccess<'a> {
    vram: &'a Arc<VectorSram>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockedRecord {
    local_token: usize,
    batch_index: usize,
    batch_size: usize,
    valid_tokens: usize,
    chunk_size: usize,
}

impl BlockedRecord {
    pub(crate) fn new(
        local_token: usize,
        batch_index: usize,
        batch_size: usize,
        valid_tokens: usize,
        chunk_size: usize,
    ) -> Self {
        Self {
            local_token,
            batch_index,
            batch_size,
            valid_tokens,
            chunk_size,
        }
    }

    fn from_state_descriptor(
        descriptor: &StateDescriptor,
        local_token: usize,
        batch_index: usize,
    ) -> Self {
        Self::new(
            local_token,
            batch_index,
            usize::from(descriptor.batch_size),
            usize::from(descriptor.valid_tokens),
            usize::from(descriptor.chunk_size),
        )
    }

    fn geometry(self) -> Result<(usize, usize), StateEngineError> {
        record_geometry(
            self.local_token,
            self.batch_index,
            self.batch_size,
            self.valid_tokens,
            self.chunk_size,
        )
    }
}

impl<'a> VramAccess<'a> {
    pub fn new(
        vram: &'a Arc<VectorSram>,
        activation_precision: StatePrecision,
    ) -> Result<Self, StateEngineError> {
        let expected = match activation_precision {
            StatePrecision::Fp32 => FpType::F32,
            StatePrecision::Bf16 => FpType::BF16,
            StatePrecision::Fp16 => FpType::F16,
            StatePrecision::Mx8B128 => {
                return Err(StateEngineError::unsupported_precision(
                    "MX8 activation VRAM requires scale-aware Vector SRAM support",
                ));
            }
        };
        if vram.ty() != MxDataType::Plain(DataType::Fp(expected)) {
            return Err(StateEngineError::invalid(
                "descriptor activation precision does not match Vector SRAM",
            ));
        }
        Ok(Self { vram })
    }

    pub async fn read_projection_token(
        &self,
        descriptor: &StateDescriptor,
        local_token: usize,
        batch_index: usize,
        layouts: &mut LayoutStore,
    ) -> Result<(Vec<f32>, ProjectionBufferStats), StateEngineError> {
        if layouts.contains(descriptor) {
            return layouts.consume_record(descriptor, local_token, batch_index);
        }
        let projected = self
            .read_blocked_record(
                descriptor.input_vram_addr,
                descriptor.input_token_stride as usize,
                BlockedRecord::from_state_descriptor(descriptor, local_token, batch_index),
            )
            .await?;
        // Legacy X_STATE-only programs remain executable, but no layout or bank
        // benefit is fabricated when L_SCATTER_M was not issued.
        Ok((projected, ProjectionBufferStats::default()))
    }

    pub async fn write_output_token(
        &self,
        descriptor: &StateDescriptor,
        local_token: usize,
        batch_index: usize,
        output: &[f32],
    ) -> Result<(), StateEngineError> {
        if output.len() > descriptor.output_token_stride as usize {
            return Err(StateEngineError::invalid(
                "X_STATE output exceeds output_token_stride",
            ));
        }
        self.write_blocked_record(
            descriptor.output_vram_addr,
            BlockedRecord::from_state_descriptor(descriptor, local_token, batch_index),
            output,
        )
        .await
    }

    /// Read one logical row from PLENA's feature-tile-major Matrix writeback.
    ///
    /// A feature tile owns all descriptor chunk rows before the next feature
    /// tile: `base + feature_tile * chunk_rows * VLEN + row * VLEN`.
    pub(crate) async fn read_blocked_record(
        &self,
        base: u32,
        stride: usize,
        record: BlockedRecord,
    ) -> Result<Vec<f32>, StateEngineError> {
        let (record, padded_records) = record.geometry()?;
        let vlen = self.vram.tile_size() as usize;
        let mut output = Vec::with_capacity(stride);
        for feature_start in (0..stride).step_by(vlen) {
            let address = blocked_address(base, feature_start, record, padded_records, vlen)?;
            let remaining = stride - output.len();
            let values = self.read_elements(address, remaining.min(vlen)).await?;
            output.extend_from_slice(&values);
        }
        Ok(output)
    }

    async fn write_blocked_record(
        &self,
        base: u32,
        record: BlockedRecord,
        values: &[f32],
    ) -> Result<(), StateEngineError> {
        let (record, padded_records) = record.geometry()?;
        let vlen = self.vram.tile_size() as usize;
        for (feature_tile, chunk) in values.chunks(vlen).enumerate() {
            let feature_start = feature_tile * vlen;
            let address = blocked_address(base, feature_start, record, padded_records, vlen)?;
            self.write_elements(address, chunk).await?;
        }
        Ok(())
    }

    fn validate_range(&self, address: u32, elements: usize) -> Result<(), StateEngineError> {
        let vlen = self.vram.tile_size();
        if !address.is_multiple_of(vlen) {
            return Err(StateEngineError::address(format!(
                "Vector SRAM address {address} is not aligned to VLEN {vlen}"
            )));
        }
        let end = u64::from(address)
            .checked_add(elements as u64)
            .ok_or_else(|| StateEngineError::address("Vector SRAM range overflows"))?;
        if end > self.vram.capacity_elements() {
            return Err(StateEngineError::address(format!(
                "Vector SRAM range [{address}, {end}) exceeds {} elements",
                self.vram.capacity_elements()
            )));
        }
        Ok(())
    }

    async fn read_elements(
        &self,
        address: u32,
        elements: usize,
    ) -> Result<Vec<f32>, StateEngineError> {
        self.validate_range(address, elements)?;
        let vlen = self.vram.tile_size() as usize;
        let mut output = Vec::with_capacity(elements);
        for row in 0..elements.div_ceil(vlen) {
            let row_address = u64::from(address) + (row * vlen) as u64;
            let tensor = self.vram.read(row_address as u32).await;
            let values = tensor_to_f32_vec(tensor.as_tensor());
            let remaining = elements - output.len();
            output.extend_from_slice(&values[..remaining.min(vlen)]);
        }
        Ok(output)
    }

    async fn write_elements(&self, address: u32, values: &[f32]) -> Result<(), StateEngineError> {
        self.validate_range(address, values.len())?;
        let vlen = self.vram.tile_size() as usize;
        for (row, chunk) in values.chunks(vlen).enumerate() {
            let row_address = (u64::from(address) + (row * vlen) as u64) as u32;
            let mut row_values = if chunk.len() == vlen {
                vec![0.0; vlen]
            } else {
                let current = self.vram.read(row_address).await;
                tensor_to_f32_vec(current.as_tensor())
            };
            row_values[..chunk.len()].copy_from_slice(chunk);
            let tensor = QuantTensor::quantize(tensor_from_f32_slice(&row_values), self.vram.ty());
            self.vram.write(row_address, tensor).await;
        }
        Ok(())
    }
}

fn record_geometry(
    local_token: usize,
    batch_index: usize,
    batch_size: usize,
    valid_tokens: usize,
    chunk_size: usize,
) -> Result<(usize, usize), StateEngineError> {
    if batch_index >= batch_size {
        return Err(StateEngineError::internal(
            "batch index exceeds descriptor batch",
        ));
    }
    let record = local_token
        .checked_mul(batch_size)
        .and_then(|value| value.checked_add(batch_index))
        .ok_or_else(|| StateEngineError::address("Vector SRAM token index overflows"))?;
    let records = valid_tokens
        .checked_mul(batch_size)
        .ok_or_else(|| StateEngineError::address("Vector SRAM row count overflows"))?;
    if record >= records {
        return Err(StateEngineError::internal(
            "token index exceeds descriptor valid rows",
        ));
    }
    let capacity_records = chunk_size
        .checked_mul(batch_size)
        .ok_or_else(|| StateEngineError::address("Vector SRAM chunk row count overflows"))?;
    if records > capacity_records {
        return Err(StateEngineError::invalid(
            "valid token rows exceed descriptor chunk capacity",
        ));
    }
    Ok((record, capacity_records))
}

fn blocked_address(
    base: u32,
    feature_start: usize,
    record: usize,
    padded_records: usize,
    vlen: usize,
) -> Result<u32, StateEngineError> {
    let feature_offset = feature_start
        .checked_mul(padded_records)
        .ok_or_else(|| StateEngineError::address("Vector SRAM feature offset overflows"))?;
    let row_offset = record
        .checked_mul(vlen)
        .ok_or_else(|| StateEngineError::address("Vector SRAM row offset overflows"))?;
    let offset = feature_offset
        .checked_add(row_offset)
        .ok_or_else(|| StateEngineError::address("Vector SRAM blocked offset overflows"))?;
    u64::from(base)
        .checked_add(offset as u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| StateEngineError::address("Vector SRAM token address overflows u32"))
}
