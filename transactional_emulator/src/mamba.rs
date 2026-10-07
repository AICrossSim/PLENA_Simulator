//! Functional Mamba-2 command support.
//!
//! This module starts with a strict parser for the cross-repository descriptor
//! ABI. Opcode dispatch is connected only after the functional engine passes
//! differential tests.

use crate::generated_contract as contract;

mod engine;
mod functional;
mod hbm;
mod timing;

pub(crate) use engine::{MambaEngine, MambaInstruction};
pub(crate) use timing::MambaTimingConfig;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MambaError {
    InvalidDescriptor(&'static str),
    UnsupportedProfile(&'static str),
    AddressError(&'static str),
    StateHazard(&'static str),
    Internal(&'static str),
}

impl MambaError {
    pub(crate) fn status(self) -> u32 {
        match self {
            Self::InvalidDescriptor(_) => contract::MAMBA_STATUS_INVALID_DESCRIPTOR,
            Self::UnsupportedProfile(_) => contract::MAMBA_STATUS_UNSUPPORTED_PROFILE,
            Self::AddressError(_) => contract::MAMBA_STATUS_ADDRESS_ERROR,
            Self::StateHazard(_) => contract::MAMBA_STATUS_STATE_HAZARD,
            Self::Internal(_) => contract::MAMBA_STATUS_INTERNAL_ERROR,
        }
    }
}

pub(crate) type PrecisionPolicy = contract::MambaPrecisionPolicy;

impl contract::MambaPrecisionPolicy {
    pub(crate) fn element_bytes(self) -> u64 {
        match self {
            Self::Fp32Reference => 4,
            Self::Bf16ActivationFp32State => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MambaDescriptorEnvelope {
    pub(crate) flags: u32,
    pub(crate) context_id: u32,
    pub(crate) completion_addr: u64,
    pub(crate) dependency_event: u32,
    pub(crate) completion_event: u32,
    pub(crate) layer_id: u32,
}

impl MambaDescriptorEnvelope {
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, MambaError> {
        if bytes.len() != contract::MAMBA_DESCRIPTOR_BYTES {
            return Err(MambaError::InvalidDescriptor(
                "descriptor length is not 256 bytes",
            ));
        }
        if read_u32(bytes, contract::MAMBA_DESC_MAGIC_OFFSET) != contract::MAMBA_DESCRIPTOR_MAGIC {
            return Err(MambaError::InvalidDescriptor("descriptor magic mismatch"));
        }
        if read_u16(bytes, contract::MAMBA_DESC_VERSION_OFFSET)
            != contract::MAMBA_DESCRIPTOR_VERSION
        {
            return Err(MambaError::InvalidDescriptor("descriptor version mismatch"));
        }
        if read_u16(bytes, contract::MAMBA_DESC_SIZE_BYTES_OFFSET) as usize
            != contract::MAMBA_DESCRIPTOR_BYTES
        {
            return Err(MambaError::InvalidDescriptor(
                "descriptor size field mismatch",
            ));
        }
        if read_u64(bytes, contract::MAMBA_DESC_RESERVED3_OFFSET) != 0 {
            return Err(MambaError::InvalidDescriptor(
                "reserved descriptor bits are nonzero",
            ));
        }
        Ok(Self {
            flags: read_u32(bytes, contract::MAMBA_DESC_FLAGS_OFFSET),
            context_id: read_u32(bytes, contract::MAMBA_DESC_CONTEXT_ID_OFFSET),
            completion_addr: read_u64(bytes, contract::MAMBA_DESC_COMPLETION_ADDR_OFFSET),
            dependency_event: read_u32(bytes, contract::MAMBA_DESC_DEPENDENCY_EVENT_OFFSET),
            completion_event: read_u32(bytes, contract::MAMBA_DESC_COMPLETION_EVENT_OFFSET),
            layer_id: read_u32(bytes, contract::MAMBA_DESC_LAYER_ID_OFFSET),
        })
    }

    pub(crate) fn write_completion(self) -> bool {
        self.flags & (1 << contract::MAMBA_FLAG_WRITE_COMPLETION_BIT) != 0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MambaDescriptor {
    pub(crate) flags: u32,
    pub(crate) context_id: u32,
    pub(crate) sequence_length: u32,
    pub(crate) batch_size: u32,
    pub(crate) d_model: u32,
    pub(crate) d_inner: u32,
    pub(crate) num_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) state_dim: u32,
    pub(crate) groups: u32,
    pub(crate) chunk_size: u32,
    pub(crate) conv_kernel: u32,
    pub(crate) precision: PrecisionPolicy,
    pub(crate) rms_norm_eps: f32,
    pub(crate) input_addr: u64,
    pub(crate) output_addr: u64,
    pub(crate) in_proj_weight_addr: u64,
    pub(crate) in_proj_bias_addr: u64,
    pub(crate) conv_weight_addr: u64,
    pub(crate) conv_bias_addr: u64,
    pub(crate) a_log_addr: u64,
    pub(crate) d_skip_addr: u64,
    pub(crate) norm_weight_addr: u64,
    pub(crate) out_proj_weight_addr: u64,
    pub(crate) out_proj_bias_addr: u64,
    pub(crate) ssm_state_addr: u64,
    pub(crate) conv_state_addr: u64,
    pub(crate) scratch_addr: u64,
    pub(crate) completion_addr: u64,
    pub(crate) dt_bias_addr: u64,
    pub(crate) input_batch_stride: u32,
    pub(crate) input_token_stride: u32,
    pub(crate) output_batch_stride: u32,
    pub(crate) output_token_stride: u32,
    pub(crate) state_request_stride: u32,
    pub(crate) state_head_stride: u32,
    pub(crate) conv_request_stride: u32,
    pub(crate) scratch_bytes: u32,
    pub(crate) dependency_event: u32,
    pub(crate) completion_event: u32,
    pub(crate) layer_id: u32,
    pub(crate) dt_min: f32,
    pub(crate) dt_max: f32,
    pub(crate) d_mlp: u32,
}

impl From<&MambaDescriptor> for MambaDescriptorEnvelope {
    fn from(descriptor: &MambaDescriptor) -> Self {
        Self {
            flags: descriptor.flags,
            context_id: descriptor.context_id,
            completion_addr: descriptor.completion_addr,
            dependency_event: descriptor.dependency_event,
            completion_event: descriptor.completion_event,
            layer_id: descriptor.layer_id,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MemorySpan {
    pub(crate) name: &'static str,
    pub(crate) address: u64,
    pub(crate) size_bytes: u64,
}

impl MemorySpan {
    fn end(self) -> Result<u64, MambaError> {
        self.address
            .checked_add(self.size_bytes)
            .ok_or(MambaError::AddressError("HBM range overflows u64"))
    }
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_bits(read_u32(bytes, offset))
}

fn checked_product(values: &[u64]) -> Result<u64, MambaError> {
    values.iter().try_fold(1u64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(MambaError::InvalidDescriptor(
                "dimension product overflows u64",
            ))
    })
}

fn require_aligned(address: u64, optional: bool) -> Result<(), MambaError> {
    if address == 0 && optional {
        return Ok(());
    }
    if address == 0 {
        return Err(MambaError::AddressError("required pointer is zero"));
    }
    if !address.is_multiple_of(64) {
        return Err(MambaError::AddressError("pointer is not 64-byte aligned"));
    }
    Ok(())
}

impl MambaDescriptor {
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, MambaError> {
        MambaDescriptorEnvelope::parse(bytes)?;
        let precision = match read_u32(bytes, contract::MAMBA_DESC_PRECISION_POLICY_OFFSET) {
            0 => PrecisionPolicy::Fp32Reference,
            1 => PrecisionPolicy::Bf16ActivationFp32State,
            _ => {
                return Err(MambaError::UnsupportedProfile(
                    "unknown Mamba precision policy",
                ));
            }
        };
        Ok(Self {
            flags: read_u32(bytes, contract::MAMBA_DESC_FLAGS_OFFSET),
            context_id: read_u32(bytes, contract::MAMBA_DESC_CONTEXT_ID_OFFSET),
            sequence_length: read_u32(bytes, contract::MAMBA_DESC_SEQUENCE_LENGTH_OFFSET),
            batch_size: read_u32(bytes, contract::MAMBA_DESC_BATCH_SIZE_OFFSET),
            d_model: read_u32(bytes, contract::MAMBA_DESC_D_MODEL_OFFSET),
            d_inner: read_u32(bytes, contract::MAMBA_DESC_D_INNER_OFFSET),
            num_heads: read_u32(bytes, contract::MAMBA_DESC_NUM_HEADS_OFFSET),
            head_dim: read_u32(bytes, contract::MAMBA_DESC_HEAD_DIM_OFFSET),
            state_dim: read_u32(bytes, contract::MAMBA_DESC_STATE_DIM_OFFSET),
            groups: read_u32(bytes, contract::MAMBA_DESC_GROUPS_OFFSET),
            chunk_size: read_u32(bytes, contract::MAMBA_DESC_CHUNK_SIZE_OFFSET),
            conv_kernel: read_u32(bytes, contract::MAMBA_DESC_CONV_KERNEL_OFFSET),
            precision,
            rms_norm_eps: read_f32(bytes, contract::MAMBA_DESC_RMS_NORM_EPS_F32_BITS_OFFSET),
            input_addr: read_u64(bytes, contract::MAMBA_DESC_INPUT_ADDR_OFFSET),
            output_addr: read_u64(bytes, contract::MAMBA_DESC_OUTPUT_ADDR_OFFSET),
            in_proj_weight_addr: read_u64(bytes, contract::MAMBA_DESC_IN_PROJ_WEIGHT_ADDR_OFFSET),
            in_proj_bias_addr: read_u64(bytes, contract::MAMBA_DESC_IN_PROJ_BIAS_ADDR_OFFSET),
            conv_weight_addr: read_u64(bytes, contract::MAMBA_DESC_CONV_WEIGHT_ADDR_OFFSET),
            conv_bias_addr: read_u64(bytes, contract::MAMBA_DESC_CONV_BIAS_ADDR_OFFSET),
            a_log_addr: read_u64(bytes, contract::MAMBA_DESC_A_LOG_ADDR_OFFSET),
            d_skip_addr: read_u64(bytes, contract::MAMBA_DESC_D_SKIP_ADDR_OFFSET),
            norm_weight_addr: read_u64(bytes, contract::MAMBA_DESC_NORM_WEIGHT_ADDR_OFFSET),
            out_proj_weight_addr: read_u64(bytes, contract::MAMBA_DESC_OUT_PROJ_WEIGHT_ADDR_OFFSET),
            out_proj_bias_addr: read_u64(bytes, contract::MAMBA_DESC_OUT_PROJ_BIAS_ADDR_OFFSET),
            ssm_state_addr: read_u64(bytes, contract::MAMBA_DESC_SSM_STATE_ADDR_OFFSET),
            conv_state_addr: read_u64(bytes, contract::MAMBA_DESC_CONV_STATE_ADDR_OFFSET),
            scratch_addr: read_u64(bytes, contract::MAMBA_DESC_SCRATCH_ADDR_OFFSET),
            completion_addr: read_u64(bytes, contract::MAMBA_DESC_COMPLETION_ADDR_OFFSET),
            dt_bias_addr: read_u64(bytes, contract::MAMBA_DESC_DT_BIAS_ADDR_OFFSET),
            input_batch_stride: read_u32(bytes, contract::MAMBA_DESC_INPUT_BATCH_STRIDE_OFFSET),
            input_token_stride: read_u32(bytes, contract::MAMBA_DESC_INPUT_TOKEN_STRIDE_OFFSET),
            output_batch_stride: read_u32(bytes, contract::MAMBA_DESC_OUTPUT_BATCH_STRIDE_OFFSET),
            output_token_stride: read_u32(bytes, contract::MAMBA_DESC_OUTPUT_TOKEN_STRIDE_OFFSET),
            state_request_stride: read_u32(bytes, contract::MAMBA_DESC_STATE_REQUEST_STRIDE_OFFSET),
            state_head_stride: read_u32(bytes, contract::MAMBA_DESC_STATE_HEAD_STRIDE_OFFSET),
            conv_request_stride: read_u32(bytes, contract::MAMBA_DESC_CONV_REQUEST_STRIDE_OFFSET),
            scratch_bytes: read_u32(bytes, contract::MAMBA_DESC_SCRATCH_BYTES_OFFSET),
            dependency_event: read_u32(bytes, contract::MAMBA_DESC_DEPENDENCY_EVENT_OFFSET),
            completion_event: read_u32(bytes, contract::MAMBA_DESC_COMPLETION_EVENT_OFFSET),
            layer_id: read_u32(bytes, contract::MAMBA_DESC_LAYER_ID_OFFSET),
            dt_min: read_f32(bytes, contract::MAMBA_DESC_DT_MIN_F32_BITS_OFFSET),
            dt_max: read_f32(bytes, contract::MAMBA_DESC_DT_MAX_F32_BITS_OFFSET),
            d_mlp: read_u32(bytes, contract::MAMBA_DESC_D_MLP_OFFSET),
        })
    }

    pub(crate) fn continue_state(&self) -> bool {
        self.flags & (1 << contract::MAMBA_FLAG_CONTINUE_STATE_BIT) != 0
    }

    pub(crate) fn write_completion(&self) -> bool {
        self.flags & (1 << contract::MAMBA_FLAG_WRITE_COMPLETION_BIT) != 0
    }

    pub(crate) fn conv_channels(&self) -> Result<u64, MambaError> {
        (self.d_inner as u64)
            .checked_add(checked_product(&[
                2,
                self.groups as u64,
                self.state_dim as u64,
            ])?)
            .ok_or(MambaError::InvalidDescriptor(
                "conv channel count overflows",
            ))
    }

    pub(crate) fn projection_size(&self) -> Result<u64, MambaError> {
        let conv_channels = self.conv_channels()?;
        checked_product(&[2, self.d_mlp as u64])?
            .checked_add(self.d_inner as u64)
            .and_then(|value| value.checked_add(conv_channels))
            .and_then(|value| value.checked_add(self.num_heads as u64))
            .ok_or(MambaError::InvalidDescriptor(
                "input projection width overflows u64",
            ))
    }

    pub(crate) fn output_projection_size(&self) -> Result<u64, MambaError> {
        (self.d_inner as u64)
            .checked_add(self.d_mlp as u64)
            .ok_or(MambaError::InvalidDescriptor(
                "output projection width overflows u64",
            ))
    }

    fn tensor_extent(&self, batch_stride: u32, token_stride: u32) -> Result<u64, MambaError> {
        let row_bytes = checked_product(&[self.d_model as u64, self.precision.element_bytes()])?;
        checked_product(&[
            self.batch_size.saturating_sub(1) as u64,
            batch_stride as u64,
        ])?
        .checked_add(checked_product(&[
            self.sequence_length.saturating_sub(1) as u64,
            token_stride as u64,
        ])?)
        .and_then(|value| value.checked_add(row_bytes))
        .ok_or(MambaError::AddressError(
            "strided tensor extent overflows u64",
        ))
    }

    fn push_span(spans: &mut Vec<MemorySpan>, name: &'static str, address: u64, size_bytes: u64) {
        if address != 0 && size_bytes != 0 {
            spans.push(MemorySpan {
                name,
                address,
                size_bytes,
            });
        }
    }

    pub(crate) fn memory_spans(
        &self,
        subop: contract::MambaSubop,
        descriptor_address: u64,
    ) -> Result<Vec<MemorySpan>, MambaError> {
        let mut spans = vec![MemorySpan {
            name: "descriptor",
            address: descriptor_address,
            size_bytes: contract::MAMBA_DESCRIPTOR_BYTES as u64,
        }];
        if self.write_completion() {
            Self::push_span(
                &mut spans,
                "completion",
                self.completion_addr,
                contract::MAMBA_COMPLETION_BYTES as u64,
            );
        }
        if subop == contract::MambaSubop::Wait {
            return Ok(spans);
        }

        Self::push_span(
            &mut spans,
            "ssm_state",
            self.ssm_state_addr,
            checked_product(&[self.batch_size as u64, self.state_request_stride as u64])?,
        );
        Self::push_span(
            &mut spans,
            "conv_state",
            self.conv_state_addr,
            checked_product(&[self.batch_size as u64, self.conv_request_stride as u64])?,
        );
        if subop == contract::MambaSubop::StateReset {
            return Ok(spans);
        }

        let element_bytes = self.precision.element_bytes();
        let conv_channels = self.conv_channels()?;
        let projection_size = self.projection_size()?;
        let output_projection_size = self.output_projection_size()?;
        Self::push_span(
            &mut spans,
            "input",
            self.input_addr,
            self.tensor_extent(self.input_batch_stride, self.input_token_stride)?,
        );
        Self::push_span(
            &mut spans,
            "output",
            self.output_addr,
            self.tensor_extent(self.output_batch_stride, self.output_token_stride)?,
        );
        for (name, address, elements) in [
            (
                "in_proj_weight",
                self.in_proj_weight_addr,
                checked_product(&[projection_size, self.d_model as u64])?,
            ),
            ("in_proj_bias", self.in_proj_bias_addr, projection_size),
            (
                "conv_weight",
                self.conv_weight_addr,
                checked_product(&[conv_channels, self.conv_kernel as u64])?,
            ),
            ("conv_bias", self.conv_bias_addr, conv_channels),
            ("a_log", self.a_log_addr, self.num_heads as u64),
            ("d_skip", self.d_skip_addr, self.num_heads as u64),
            ("dt_bias", self.dt_bias_addr, self.num_heads as u64),
            ("norm_weight", self.norm_weight_addr, self.d_inner as u64),
            (
                "out_proj_weight",
                self.out_proj_weight_addr,
                checked_product(&[self.d_model as u64, output_projection_size])?,
            ),
            (
                "out_proj_bias",
                self.out_proj_bias_addr,
                self.d_model as u64,
            ),
        ] {
            Self::push_span(
                &mut spans,
                name,
                address,
                checked_product(&[elements, element_bytes])?,
            );
        }
        Ok(spans)
    }

    pub(crate) fn validate_memory_map(
        &self,
        subop: contract::MambaSubop,
        descriptor_address: u64,
        capacity_bytes: Option<u64>,
    ) -> Result<(), MambaError> {
        if !descriptor_address.is_multiple_of(contract::MAMBA_DESCRIPTOR_ALIGNMENT as u64) {
            return Err(MambaError::AddressError(
                "descriptor pointer is not 64-byte aligned",
            ));
        }
        let mut spans = self.memory_spans(subop, descriptor_address)?;
        for span in &spans {
            let end = span.end()?;
            if let Some(capacity) = capacity_bytes
                && end > capacity
            {
                return Err(MambaError::AddressError("HBM range exceeds capacity"));
            }
        }
        spans.sort_unstable_by_key(|span| span.address);
        for pair in spans.windows(2) {
            if pair[0].end()? > pair[1].address {
                return Err(MambaError::AddressError("HBM ranges overlap"));
            }
        }
        Ok(())
    }

    pub(crate) fn validate(
        &self,
        subop: contract::MambaSubop,
        register_context: u32,
    ) -> Result<(), MambaError> {
        let known_flags = (1 << contract::MAMBA_FLAG_CONTINUE_STATE_BIT)
            | (1 << contract::MAMBA_FLAG_WRITE_COMPLETION_BIT)
            | (1 << contract::MAMBA_FLAG_PROFILE_BIT);
        if self.flags & !known_flags != 0 {
            return Err(MambaError::InvalidDescriptor("unknown descriptor flags"));
        }
        if self.context_id != register_context {
            return Err(MambaError::InvalidDescriptor(
                "register and descriptor context IDs differ",
            ));
        }
        if (self.scratch_addr == 0) != (self.scratch_bytes == 0) {
            return Err(MambaError::InvalidDescriptor(
                "scratch address and size presence differ",
            ));
        }
        if self.scratch_bytes != 0 {
            return Err(MambaError::UnsupportedProfile(
                "functional baseline does not use external scratch",
            ));
        }
        if self.write_completion() {
            require_aligned(self.completion_addr, false)?;
        } else if self.completion_addr != 0 {
            return Err(MambaError::InvalidDescriptor(
                "completion pointer is set without WRITE_COMPLETION",
            ));
        }

        // WAIT has no tensor or state operands. Descriptor header, flags,
        // context, dependency/completion metadata, and completion address are
        // the only architected inputs for this sub-operation.
        if subop == contract::MambaSubop::Wait {
            return Ok(());
        }

        if self.batch_size == 0 {
            return Err(MambaError::InvalidDescriptor("batch size is zero"));
        }
        match subop {
            contract::MambaSubop::Prefill if self.sequence_length == 0 => {
                return Err(MambaError::InvalidDescriptor(
                    "MAMBA_PREFILL sequence length is zero",
                ));
            }
            contract::MambaSubop::Step if self.sequence_length != 1 || !self.continue_state() => {
                return Err(MambaError::InvalidDescriptor(
                    "MAMBA_STEP requires sequence length 1 and continuing state",
                ));
            }
            _ => {}
        }

        let state_dimensions = [
            self.d_inner,
            self.num_heads,
            self.head_dim,
            self.state_dim,
            self.groups,
            self.conv_kernel,
        ];
        if state_dimensions.contains(&0) {
            return Err(MambaError::InvalidDescriptor(
                "a persistent-state dimension is zero",
            ));
        }
        if self.num_heads.checked_mul(self.head_dim) != Some(self.d_inner) {
            return Err(MambaError::InvalidDescriptor(
                "d_inner does not equal num_heads * head_dim",
            ));
        }
        if !self.num_heads.is_multiple_of(self.groups) || !self.d_inner.is_multiple_of(self.groups)
        {
            return Err(MambaError::InvalidDescriptor(
                "head or inner dimension is not divisible by groups",
            ));
        }

        require_aligned(self.ssm_state_addr, false)?;
        require_aligned(self.conv_state_addr, false)?;
        let expected_head_stride =
            checked_product(&[self.head_dim as u64, self.state_dim as u64, 4])?;
        let expected_request_stride =
            checked_product(&[self.num_heads as u64, expected_head_stride])?;
        let conv_channels = self.conv_channels()?;
        let expected_conv_stride = checked_product(&[conv_channels, self.conv_kernel as u64, 4])?;
        if self.state_head_stride as u64 != expected_head_stride
            || self.state_request_stride as u64 != expected_request_stride
            || self.conv_request_stride as u64 != expected_conv_stride
        {
            return Err(MambaError::InvalidDescriptor(
                "persistent state stride does not match the canonical layout",
            ));
        }

        // STATE_RESET zeroes only the persistent state range. All model
        // tensors, numerical parameters, sequence length, and I/O strides are
        // deliberately ignored.
        if subop == contract::MambaSubop::StateReset {
            return Ok(());
        }

        if self.d_model == 0 || self.chunk_size == 0 {
            return Err(MambaError::InvalidDescriptor(
                "an execution dimension is zero",
            ));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(MambaError::InvalidDescriptor("invalid RMSNorm epsilon"));
        }
        if !self.dt_min.is_finite()
            || self.dt_min < 0.0
            || self.dt_max.is_nan()
            || self.dt_max < self.dt_min
        {
            return Err(MambaError::InvalidDescriptor("invalid dt clamp interval"));
        }
        for address in [
            self.input_addr,
            self.output_addr,
            self.in_proj_weight_addr,
            self.conv_weight_addr,
            self.a_log_addr,
            self.d_skip_addr,
            self.norm_weight_addr,
            self.out_proj_weight_addr,
            self.dt_bias_addr,
        ] {
            require_aligned(address, false)?;
        }
        for address in [
            self.in_proj_bias_addr,
            self.conv_bias_addr,
            self.out_proj_bias_addr,
        ] {
            require_aligned(address, true)?;
        }

        let element_bytes = self.precision.element_bytes();
        let row_bytes = checked_product(&[self.d_model as u64, element_bytes])?;
        for (token_stride, batch_stride) in [
            (self.input_token_stride, self.input_batch_stride),
            (self.output_token_stride, self.output_batch_stride),
        ] {
            let batch_extent =
                checked_product(&[token_stride as u64, self.sequence_length as u64])?;
            if (token_stride as u64) < row_bytes
                || !(token_stride as u64).is_multiple_of(element_bytes)
                || !(batch_stride as u64).is_multiple_of(element_bytes)
                || (batch_stride as u64) < batch_extent
            {
                return Err(MambaError::InvalidDescriptor("invalid input/output stride"));
            }
        }

        // Force all derived projection sizes through checked arithmetic here;
        // the functional engine can then allocate using these dimensions
        // without host-integer wraparound.
        let projection_size = self.projection_size()?;
        checked_product(&[projection_size, self.d_model as u64, element_bytes])?;
        checked_product(&[
            self.d_model as u64,
            self.output_projection_size()?,
            element_bytes,
        ])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn descriptor_bytes() -> Vec<u8> {
        let mut bytes = vec![0u8; contract::MAMBA_DESCRIPTOR_BYTES];
        write_u32(
            &mut bytes,
            contract::MAMBA_DESC_MAGIC_OFFSET,
            contract::MAMBA_DESCRIPTOR_MAGIC,
        );
        write_u16(
            &mut bytes,
            contract::MAMBA_DESC_VERSION_OFFSET,
            contract::MAMBA_DESCRIPTOR_VERSION,
        );
        write_u16(
            &mut bytes,
            contract::MAMBA_DESC_SIZE_BYTES_OFFSET,
            contract::MAMBA_DESCRIPTOR_BYTES as u16,
        );
        write_u32(
            &mut bytes,
            contract::MAMBA_DESC_FLAGS_OFFSET,
            1 << contract::MAMBA_FLAG_WRITE_COMPLETION_BIT,
        );
        write_u32(&mut bytes, contract::MAMBA_DESC_CONTEXT_ID_OFFSET, 7);
        write_u32(&mut bytes, contract::MAMBA_DESC_SEQUENCE_LENGTH_OFFSET, 5);
        write_u32(&mut bytes, contract::MAMBA_DESC_BATCH_SIZE_OFFSET, 2);
        write_u32(&mut bytes, contract::MAMBA_DESC_D_MODEL_OFFSET, 16);
        write_u32(&mut bytes, contract::MAMBA_DESC_D_INNER_OFFSET, 16);
        write_u32(&mut bytes, contract::MAMBA_DESC_NUM_HEADS_OFFSET, 2);
        write_u32(&mut bytes, contract::MAMBA_DESC_HEAD_DIM_OFFSET, 8);
        write_u32(&mut bytes, contract::MAMBA_DESC_STATE_DIM_OFFSET, 16);
        write_u32(&mut bytes, contract::MAMBA_DESC_GROUPS_OFFSET, 2);
        write_u32(&mut bytes, contract::MAMBA_DESC_CHUNK_SIZE_OFFSET, 8);
        write_u32(&mut bytes, contract::MAMBA_DESC_CONV_KERNEL_OFFSET, 4);
        write_u32(&mut bytes, contract::MAMBA_DESC_PRECISION_POLICY_OFFSET, 1);
        write_u32(
            &mut bytes,
            contract::MAMBA_DESC_RMS_NORM_EPS_F32_BITS_OFFSET,
            1e-5f32.to_bits(),
        );
        for offset in [
            contract::MAMBA_DESC_INPUT_ADDR_OFFSET,
            contract::MAMBA_DESC_OUTPUT_ADDR_OFFSET,
            contract::MAMBA_DESC_IN_PROJ_WEIGHT_ADDR_OFFSET,
            contract::MAMBA_DESC_CONV_WEIGHT_ADDR_OFFSET,
            contract::MAMBA_DESC_A_LOG_ADDR_OFFSET,
            contract::MAMBA_DESC_D_SKIP_ADDR_OFFSET,
            contract::MAMBA_DESC_NORM_WEIGHT_ADDR_OFFSET,
            contract::MAMBA_DESC_OUT_PROJ_WEIGHT_ADDR_OFFSET,
            contract::MAMBA_DESC_SSM_STATE_ADDR_OFFSET,
            contract::MAMBA_DESC_CONV_STATE_ADDR_OFFSET,
            contract::MAMBA_DESC_COMPLETION_ADDR_OFFSET,
            contract::MAMBA_DESC_DT_BIAS_ADDR_OFFSET,
        ] {
            write_u64(&mut bytes, offset, 0x1000 + offset as u64 * 64);
        }
        write_u32(
            &mut bytes,
            contract::MAMBA_DESC_INPUT_TOKEN_STRIDE_OFFSET,
            32,
        );
        write_u32(
            &mut bytes,
            contract::MAMBA_DESC_INPUT_BATCH_STRIDE_OFFSET,
            160,
        );
        write_u32(
            &mut bytes,
            contract::MAMBA_DESC_OUTPUT_TOKEN_STRIDE_OFFSET,
            32,
        );
        write_u32(
            &mut bytes,
            contract::MAMBA_DESC_OUTPUT_BATCH_STRIDE_OFFSET,
            160,
        );
        write_u32(
            &mut bytes,
            contract::MAMBA_DESC_STATE_HEAD_STRIDE_OFFSET,
            512,
        );
        write_u32(
            &mut bytes,
            contract::MAMBA_DESC_STATE_REQUEST_STRIDE_OFFSET,
            1024,
        );
        write_u32(
            &mut bytes,
            contract::MAMBA_DESC_CONV_REQUEST_STRIDE_OFFSET,
            1280,
        );
        write_u32(
            &mut bytes,
            contract::MAMBA_DESC_DEPENDENCY_EVENT_OFFSET,
            contract::MAMBA_NO_EVENT,
        );
        write_u32(&mut bytes, contract::MAMBA_DESC_COMPLETION_EVENT_OFFSET, 9);
        write_u32(
            &mut bytes,
            contract::MAMBA_DESC_DT_MIN_F32_BITS_OFFSET,
            0.0f32.to_bits(),
        );
        write_u32(
            &mut bytes,
            contract::MAMBA_DESC_DT_MAX_F32_BITS_OFFSET,
            f32::INFINITY.to_bits(),
        );
        bytes
    }

    fn non_overlapping_descriptor_bytes() -> Vec<u8> {
        let mut bytes = descriptor_bytes();
        for (offset, address) in [
            (contract::MAMBA_DESC_INPUT_ADDR_OFFSET, 0x1000),
            (contract::MAMBA_DESC_OUTPUT_ADDR_OFFSET, 0x2000),
            (contract::MAMBA_DESC_IN_PROJ_WEIGHT_ADDR_OFFSET, 0x3000),
            (contract::MAMBA_DESC_CONV_WEIGHT_ADDR_OFFSET, 0x4000),
            (contract::MAMBA_DESC_A_LOG_ADDR_OFFSET, 0x5000),
            (contract::MAMBA_DESC_D_SKIP_ADDR_OFFSET, 0x5100),
            (contract::MAMBA_DESC_NORM_WEIGHT_ADDR_OFFSET, 0x5200),
            (contract::MAMBA_DESC_OUT_PROJ_WEIGHT_ADDR_OFFSET, 0x6000),
            (contract::MAMBA_DESC_SSM_STATE_ADDR_OFFSET, 0x7000),
            (contract::MAMBA_DESC_CONV_STATE_ADDR_OFFSET, 0x8000),
            (contract::MAMBA_DESC_COMPLETION_ADDR_OFFSET, 0x9000),
            (contract::MAMBA_DESC_DT_BIAS_ADDR_OFFSET, 0x9100),
        ] {
            write_u64(&mut bytes, offset, address);
        }
        bytes
    }

    #[test]
    fn parses_and_validates_canonical_bf16_prefill() {
        let descriptor = MambaDescriptor::parse(&descriptor_bytes()).unwrap();
        assert_eq!(descriptor.context_id, 7);
        assert_eq!(
            descriptor.precision,
            PrecisionPolicy::Bf16ActivationFp32State
        );
        descriptor
            .validate(contract::MambaSubop::Prefill, 7)
            .unwrap();
    }

    #[test]
    fn rejects_bad_header_reserved_bits_and_context_mismatch() {
        let mut bytes = descriptor_bytes();
        write_u32(&mut bytes, contract::MAMBA_DESC_MAGIC_OFFSET, 0);
        assert!(matches!(
            MambaDescriptor::parse(&bytes),
            Err(MambaError::InvalidDescriptor("descriptor magic mismatch"))
        ));
        let mut bytes = descriptor_bytes();
        write_u64(&mut bytes, contract::MAMBA_DESC_RESERVED3_OFFSET, 1);
        assert!(MambaDescriptor::parse(&bytes).is_err());
        let descriptor = MambaDescriptor::parse(&descriptor_bytes()).unwrap();
        assert!(
            descriptor
                .validate(contract::MambaSubop::Prefill, 8)
                .is_err()
        );
    }

    #[test]
    fn envelope_separates_trusted_routing_fields_from_body_validation() {
        let mut bytes = descriptor_bytes();
        write_u32(&mut bytes, contract::MAMBA_DESC_PRECISION_POLICY_OFFSET, 99);
        let envelope = MambaDescriptorEnvelope::parse(&bytes).unwrap();
        assert_eq!(envelope.context_id, 7);
        assert_eq!(envelope.completion_event, 9);
        assert!(matches!(
            MambaDescriptor::parse(&bytes),
            Err(MambaError::UnsupportedProfile(_))
        ));

        write_u32(&mut bytes, contract::MAMBA_DESC_MAGIC_OFFSET, 0);
        assert!(MambaDescriptorEnvelope::parse(&bytes).is_err());
    }

    #[test]
    fn step_requires_one_token_and_continuing_state() {
        let descriptor = MambaDescriptor::parse(&descriptor_bytes()).unwrap();
        assert!(descriptor.validate(contract::MambaSubop::Step, 7).is_err());
        let mut bytes = descriptor_bytes();
        write_u32(&mut bytes, contract::MAMBA_DESC_SEQUENCE_LENGTH_OFFSET, 1);
        write_u32(
            &mut bytes,
            contract::MAMBA_DESC_FLAGS_OFFSET,
            (1 << contract::MAMBA_FLAG_CONTINUE_STATE_BIT)
                | (1 << contract::MAMBA_FLAG_WRITE_COMPLETION_BIT),
        );
        write_u32(
            &mut bytes,
            contract::MAMBA_DESC_INPUT_BATCH_STRIDE_OFFSET,
            32,
        );
        write_u32(
            &mut bytes,
            contract::MAMBA_DESC_OUTPUT_BATCH_STRIDE_OFFSET,
            32,
        );
        MambaDescriptor::parse(&bytes)
            .unwrap()
            .validate(contract::MambaSubop::Step, 7)
            .unwrap();
    }

    #[test]
    fn wait_ignores_tensor_and_state_fields() {
        let mut bytes = descriptor_bytes();
        for offset in [
            contract::MAMBA_DESC_INPUT_ADDR_OFFSET,
            contract::MAMBA_DESC_OUTPUT_ADDR_OFFSET,
            contract::MAMBA_DESC_IN_PROJ_WEIGHT_ADDR_OFFSET,
            contract::MAMBA_DESC_CONV_WEIGHT_ADDR_OFFSET,
            contract::MAMBA_DESC_A_LOG_ADDR_OFFSET,
            contract::MAMBA_DESC_D_SKIP_ADDR_OFFSET,
            contract::MAMBA_DESC_NORM_WEIGHT_ADDR_OFFSET,
            contract::MAMBA_DESC_OUT_PROJ_WEIGHT_ADDR_OFFSET,
            contract::MAMBA_DESC_SSM_STATE_ADDR_OFFSET,
            contract::MAMBA_DESC_CONV_STATE_ADDR_OFFSET,
            contract::MAMBA_DESC_DT_BIAS_ADDR_OFFSET,
        ] {
            write_u64(&mut bytes, offset, 0);
        }
        for offset in [
            contract::MAMBA_DESC_SEQUENCE_LENGTH_OFFSET,
            contract::MAMBA_DESC_BATCH_SIZE_OFFSET,
            contract::MAMBA_DESC_D_MODEL_OFFSET,
            contract::MAMBA_DESC_D_INNER_OFFSET,
            contract::MAMBA_DESC_NUM_HEADS_OFFSET,
            contract::MAMBA_DESC_HEAD_DIM_OFFSET,
            contract::MAMBA_DESC_STATE_DIM_OFFSET,
            contract::MAMBA_DESC_GROUPS_OFFSET,
            contract::MAMBA_DESC_CHUNK_SIZE_OFFSET,
            contract::MAMBA_DESC_CONV_KERNEL_OFFSET,
            contract::MAMBA_DESC_INPUT_BATCH_STRIDE_OFFSET,
            contract::MAMBA_DESC_INPUT_TOKEN_STRIDE_OFFSET,
            contract::MAMBA_DESC_OUTPUT_BATCH_STRIDE_OFFSET,
            contract::MAMBA_DESC_OUTPUT_TOKEN_STRIDE_OFFSET,
            contract::MAMBA_DESC_STATE_REQUEST_STRIDE_OFFSET,
            contract::MAMBA_DESC_STATE_HEAD_STRIDE_OFFSET,
            contract::MAMBA_DESC_CONV_REQUEST_STRIDE_OFFSET,
        ] {
            write_u32(&mut bytes, offset, 0);
        }
        MambaDescriptor::parse(&bytes)
            .unwrap()
            .validate(contract::MambaSubop::Wait, 7)
            .unwrap();
    }

    #[test]
    fn state_reset_validates_only_persistent_state_layout() {
        let mut bytes = descriptor_bytes();
        for offset in [
            contract::MAMBA_DESC_INPUT_ADDR_OFFSET,
            contract::MAMBA_DESC_OUTPUT_ADDR_OFFSET,
            contract::MAMBA_DESC_IN_PROJ_WEIGHT_ADDR_OFFSET,
            contract::MAMBA_DESC_CONV_WEIGHT_ADDR_OFFSET,
            contract::MAMBA_DESC_A_LOG_ADDR_OFFSET,
            contract::MAMBA_DESC_D_SKIP_ADDR_OFFSET,
            contract::MAMBA_DESC_NORM_WEIGHT_ADDR_OFFSET,
            contract::MAMBA_DESC_OUT_PROJ_WEIGHT_ADDR_OFFSET,
            contract::MAMBA_DESC_DT_BIAS_ADDR_OFFSET,
        ] {
            write_u64(&mut bytes, offset, 0);
        }
        for offset in [
            contract::MAMBA_DESC_SEQUENCE_LENGTH_OFFSET,
            contract::MAMBA_DESC_D_MODEL_OFFSET,
            contract::MAMBA_DESC_CHUNK_SIZE_OFFSET,
            contract::MAMBA_DESC_INPUT_BATCH_STRIDE_OFFSET,
            contract::MAMBA_DESC_INPUT_TOKEN_STRIDE_OFFSET,
            contract::MAMBA_DESC_OUTPUT_BATCH_STRIDE_OFFSET,
            contract::MAMBA_DESC_OUTPUT_TOKEN_STRIDE_OFFSET,
            contract::MAMBA_DESC_RMS_NORM_EPS_F32_BITS_OFFSET,
            contract::MAMBA_DESC_DT_MIN_F32_BITS_OFFSET,
            contract::MAMBA_DESC_DT_MAX_F32_BITS_OFFSET,
        ] {
            write_u32(&mut bytes, offset, 0);
        }
        MambaDescriptor::parse(&bytes)
            .unwrap()
            .validate(contract::MambaSubop::StateReset, 7)
            .unwrap();
    }

    #[test]
    fn rejects_batch_stride_product_outside_u32_range() {
        let mut bytes = descriptor_bytes();
        write_u32(
            &mut bytes,
            contract::MAMBA_DESC_INPUT_TOKEN_STRIDE_OFFSET,
            u32::MAX - 1,
        );
        write_u32(
            &mut bytes,
            contract::MAMBA_DESC_INPUT_BATCH_STRIDE_OFFSET,
            u32::MAX,
        );
        assert!(
            MambaDescriptor::parse(&bytes)
                .unwrap()
                .validate(contract::MambaSubop::Prefill, 7)
                .is_err()
        );
    }

    #[test]
    fn validates_complete_non_overlapping_hbm_map() {
        let descriptor = MambaDescriptor::parse(&non_overlapping_descriptor_bytes()).unwrap();
        descriptor
            .validate(contract::MambaSubop::Prefill, 7)
            .unwrap();
        descriptor
            .validate_memory_map(contract::MambaSubop::Prefill, 0xa000, Some(0xb000))
            .unwrap();
    }

    #[test]
    fn rejects_overlapping_out_of_capacity_and_unaligned_hbm_maps() {
        let mut bytes = non_overlapping_descriptor_bytes();
        write_u64(&mut bytes, contract::MAMBA_DESC_OUTPUT_ADDR_OFFSET, 0x1000);
        let descriptor = MambaDescriptor::parse(&bytes).unwrap();
        assert_eq!(
            descriptor
                .validate_memory_map(contract::MambaSubop::Prefill, 0xa000, Some(0xb000))
                .unwrap_err()
                .status(),
            contract::MAMBA_STATUS_ADDRESS_ERROR
        );

        let descriptor = MambaDescriptor::parse(&non_overlapping_descriptor_bytes()).unwrap();
        assert!(
            descriptor
                .validate_memory_map(contract::MambaSubop::Prefill, 0xa000, Some(0xa080))
                .is_err()
        );
        assert!(
            descriptor
                .validate_memory_map(contract::MambaSubop::Prefill, 0xa020, Some(0xb000))
                .is_err()
        );
    }

    #[test]
    fn rejects_alignment_scratch_and_state_stride_errors() {
        let mut bytes = descriptor_bytes();
        write_u64(&mut bytes, contract::MAMBA_DESC_INPUT_ADDR_OFFSET, 0x1001);
        assert_eq!(
            MambaDescriptor::parse(&bytes)
                .unwrap()
                .validate(contract::MambaSubop::Prefill, 7)
                .unwrap_err()
                .status(),
            contract::MAMBA_STATUS_ADDRESS_ERROR
        );
        let mut bytes = descriptor_bytes();
        write_u32(&mut bytes, contract::MAMBA_DESC_SCRATCH_BYTES_OFFSET, 64);
        assert!(
            MambaDescriptor::parse(&bytes)
                .unwrap()
                .validate(contract::MambaSubop::Prefill, 7)
                .is_err()
        );
        let mut bytes = descriptor_bytes();
        write_u32(
            &mut bytes,
            contract::MAMBA_DESC_STATE_HEAD_STRIDE_OFFSET,
            256,
        );
        assert!(
            MambaDescriptor::parse(&bytes)
                .unwrap()
                .validate(contract::MambaSubop::Prefill, 7)
                .is_err()
        );
    }
}
