use std::fmt;

use super::generated_contract::{self as wire, StateAlgorithm, StatePrecision, StateStatus};

const KNOWN_FLAGS: u32 = wire::FLAG_LAST_CHUNK | wire::FLAG_WRITE_COMPLETION | wire::FLAG_PROFILE;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StateIdentity {
    pub context_id: u32,
    pub request_id: u32,
    pub layer_id: u32,
    pub state_id: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mamba2Payload {
    pub head_dim: u16,
    pub state_dim: u16,
    pub groups: u16,
    pub conv_kernel: u16,
    pub xbc_offset: u32,
    pub dt_offset: u32,
    pub conv_weight_addr: u64,
    pub conv_bias_addr: u64,
    pub a_log_addr: u64,
    pub dt_bias_addr: u64,
    pub d_skip_addr: u64,
    pub parameter_scale_addr: u64,
    pub dt_min: f32,
    pub dt_max: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KdaPayload {
    pub key_dim: u16,
    pub value_dim: u16,
    pub conv_kernel: u16,
    pub q_offset: u32,
    pub k_offset: u32,
    pub v_offset: u32,
    pub decay_offset: u32,
    pub beta_offset: u32,
    pub q_conv_weight_addr: u64,
    pub k_conv_weight_addr: u64,
    pub v_conv_weight_addr: u64,
    pub q_conv_bias_addr: u64,
    pub k_conv_bias_addr: u64,
    pub v_conv_bias_addr: u64,
    pub a_log_addr: u64,
    pub dt_bias_addr: u64,
    pub parameter_scale_addr: u64,
    pub output_scale: f32,
    pub gate_lower_bound: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StatePayload {
    Mamba2(Mamba2Payload),
    Kda(KdaPayload),
}

#[derive(Clone, Debug, PartialEq)]
pub struct StateDescriptor {
    pub algorithm: StateAlgorithm,
    pub state_precision: StatePrecision,
    pub conv_state_precision: StatePrecision,
    pub activation_precision: StatePrecision,
    pub parameter_precision: StatePrecision,
    pub flags: u32,
    pub identity: StateIdentity,
    pub batch_size: u16,
    pub num_heads: u16,
    pub sequence_length: u32,
    pub token_offset: u32,
    pub valid_tokens: u16,
    pub chunk_size: u16,
    pub state_sram_offset: u32,
    pub state_bytes: u32,
    pub conv_state_bytes: u32,
    pub input_vram_addr: u32,
    pub output_vram_addr: u32,
    pub input_token_stride: u32,
    pub output_token_stride: u32,
    pub state_hbm_addr: u64,
    pub conv_state_hbm_addr: u64,
    pub state_scale_addr: u64,
    pub state_scale_bytes: u32,
    pub conv_state_scale_bytes: u32,
    pub completion_addr: u64,
    pub dependency_event: u32,
    pub completion_event: u32,
    pub payload: StatePayload,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DescriptorError {
    pub status: StateStatus,
    message: String,
}

impl DescriptorError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: StateStatus::InvalidDescriptor,
            message: message.into(),
        }
    }

    fn unsupported_algorithm(value: u8) -> Self {
        Self {
            status: StateStatus::UnsupportedAlgorithm,
            message: format!("unsupported state algorithm {value}"),
        }
    }

    fn unsupported_precision(value: u8) -> Self {
        Self {
            status: StateStatus::UnsupportedPrecision,
            message: format!("unsupported state precision {value}"),
        }
    }

    fn address(message: impl Into<String>) -> Self {
        Self {
            status: StateStatus::AddressError,
            message: message.into(),
        }
    }
}

impl fmt::Display for DescriptorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DescriptorError {}

impl StateDescriptor {
    pub fn parse(data: &[u8]) -> Result<Self, DescriptorError> {
        if data.len() != wire::DESCRIPTOR_SIZE {
            return Err(DescriptorError::invalid(format!(
                "descriptor must be exactly {} bytes",
                wire::DESCRIPTOR_SIZE
            )));
        }
        if u32_at(data, wire::common::MAGIC) != wire::DESCRIPTOR_MAGIC
            || u16_at(data, wire::common::VERSION) != wire::DESCRIPTOR_VERSION
            || usize::from(u16_at(data, wire::common::SIZE_BYTES)) != wire::DESCRIPTOR_SIZE
        {
            return Err(DescriptorError::invalid(
                "incompatible X_STATE descriptor header",
            ));
        }
        require_zero(data, 62..64, "common reserved0")?;

        let algorithm_raw = data[wire::common::ALGORITHM];
        let algorithm = StateAlgorithm::try_from(algorithm_raw)
            .map_err(|_| DescriptorError::unsupported_algorithm(algorithm_raw))?;
        let state_precision = precision_at(data, wire::common::STATE_PRECISION)?;
        let conv_state_precision = precision_at(data, wire::common::CONV_STATE_PRECISION)?;
        let activation_precision = precision_at(data, wire::common::ACTIVATION_PRECISION)?;
        let accumulator_precision = precision_at(data, wire::common::ACCUMULATOR_PRECISION)?;
        if accumulator_precision != StatePrecision::Fp32 {
            return Err(DescriptorError::invalid(
                "X_STATE accumulation precision must be FP32",
            ));
        }
        let flags = u32_at(data, wire::common::FLAGS);
        if flags & !KNOWN_FLAGS != 0 {
            return Err(DescriptorError::invalid("descriptor has unknown flag bits"));
        }

        let payload = match algorithm {
            StateAlgorithm::Mamba2 => {
                require_zero(data, 200..256, "Mamba2 reserved payload")?;
                StatePayload::Mamba2(Mamba2Payload {
                    head_dim: u16_at(data, wire::mamba2::HEAD_DIM),
                    state_dim: u16_at(data, wire::mamba2::STATE_DIM),
                    groups: u16_at(data, wire::mamba2::GROUPS),
                    conv_kernel: u16_at(data, wire::mamba2::CONV_KERNEL),
                    xbc_offset: u32_at(data, wire::mamba2::XBC_OFFSET),
                    dt_offset: u32_at(data, wire::mamba2::DT_OFFSET),
                    conv_weight_addr: u64_at(data, wire::mamba2::CONV_WEIGHT_ADDR),
                    conv_bias_addr: u64_at(data, wire::mamba2::CONV_BIAS_ADDR),
                    a_log_addr: u64_at(data, wire::mamba2::A_LOG_ADDR),
                    dt_bias_addr: u64_at(data, wire::mamba2::DT_BIAS_ADDR),
                    d_skip_addr: u64_at(data, wire::mamba2::D_SKIP_ADDR),
                    parameter_scale_addr: u64_at(data, wire::mamba2::PARAMETER_SCALE_ADDR),
                    dt_min: f32::from_bits(u32_at(data, wire::mamba2::DT_MIN_F32_BITS)),
                    dt_max: f32::from_bits(u32_at(data, wire::mamba2::DT_MAX_F32_BITS)),
                })
            }
            StateAlgorithm::Kda => {
                require_zero(data, 134..136, "KDA reserved0")?;
                require_zero(data, 156..160, "KDA reserved1")?;
                require_zero(data, 240..256, "KDA reserved payload")?;
                StatePayload::Kda(KdaPayload {
                    key_dim: u16_at(data, wire::kda::KEY_DIM),
                    value_dim: u16_at(data, wire::kda::VALUE_DIM),
                    conv_kernel: u16_at(data, wire::kda::CONV_KERNEL),
                    q_offset: u32_at(data, wire::kda::Q_OFFSET),
                    k_offset: u32_at(data, wire::kda::K_OFFSET),
                    v_offset: u32_at(data, wire::kda::V_OFFSET),
                    decay_offset: u32_at(data, wire::kda::DECAY_OFFSET),
                    beta_offset: u32_at(data, wire::kda::BETA_OFFSET),
                    q_conv_weight_addr: u64_at(data, wire::kda::Q_CONV_WEIGHT_ADDR),
                    k_conv_weight_addr: u64_at(data, wire::kda::K_CONV_WEIGHT_ADDR),
                    v_conv_weight_addr: u64_at(data, wire::kda::V_CONV_WEIGHT_ADDR),
                    q_conv_bias_addr: u64_at(data, wire::kda::Q_CONV_BIAS_ADDR),
                    k_conv_bias_addr: u64_at(data, wire::kda::K_CONV_BIAS_ADDR),
                    v_conv_bias_addr: u64_at(data, wire::kda::V_CONV_BIAS_ADDR),
                    a_log_addr: u64_at(data, wire::kda::A_LOG_ADDR),
                    dt_bias_addr: u64_at(data, wire::kda::DT_BIAS_ADDR),
                    parameter_scale_addr: u64_at(data, wire::kda::PARAMETER_SCALE_ADDR),
                    output_scale: f32::from_bits(u32_at(data, wire::kda::OUTPUT_SCALE_F32_BITS)),
                    gate_lower_bound: f32::from_bits(u32_at(
                        data,
                        wire::kda::GATE_LOWER_BOUND_F32_BITS,
                    )),
                })
            }
        };

        let descriptor = Self {
            algorithm,
            state_precision,
            conv_state_precision,
            activation_precision,
            parameter_precision: precision_at(data, wire::common::PARAMETER_PRECISION)?,
            flags,
            identity: StateIdentity {
                context_id: u32_at(data, wire::common::CONTEXT_ID),
                request_id: u32_at(data, wire::common::REQUEST_ID),
                layer_id: u32_at(data, wire::common::LAYER_ID),
                state_id: u32_at(data, wire::common::STATE_ID),
            },
            batch_size: u16_at(data, wire::common::BATCH_SIZE),
            num_heads: u16_at(data, wire::common::NUM_HEADS),
            sequence_length: u32_at(data, wire::common::SEQUENCE_LENGTH),
            token_offset: u32_at(data, wire::common::TOKEN_OFFSET),
            valid_tokens: u16_at(data, wire::common::VALID_TOKENS),
            chunk_size: u16_at(data, wire::common::CHUNK_SIZE),
            state_sram_offset: u32_at(data, wire::common::STATE_SRAM_OFFSET),
            state_bytes: u32_at(data, wire::common::STATE_BYTES),
            conv_state_bytes: u32_at(data, wire::common::CONV_STATE_BYTES),
            input_vram_addr: u32_at(data, wire::common::INPUT_VRAM_ADDR),
            output_vram_addr: u32_at(data, wire::common::OUTPUT_VRAM_ADDR),
            input_token_stride: u32_at(data, wire::common::INPUT_TOKEN_STRIDE),
            output_token_stride: u32_at(data, wire::common::OUTPUT_TOKEN_STRIDE),
            state_hbm_addr: u64_at(data, wire::common::STATE_HBM_ADDR),
            conv_state_hbm_addr: u64_at(data, wire::common::CONV_STATE_HBM_ADDR),
            state_scale_addr: u64_at(data, wire::common::STATE_SCALE_ADDR),
            state_scale_bytes: u32_at(data, wire::common::STATE_SCALE_BYTES),
            conv_state_scale_bytes: u32_at(data, wire::common::CONV_STATE_SCALE_BYTES),
            completion_addr: u64_at(data, wire::common::COMPLETION_ADDR),
            dependency_event: u32_at(data, wire::common::DEPENDENCY_EVENT),
            completion_event: u32_at(data, wire::common::COMPLETION_EVENT),
            payload,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    pub fn is_streaming(&self) -> bool {
        self.state_sram_offset == wire::STREAMING_SRAM_OFFSET
    }

    pub fn writes_completion(&self) -> bool {
        self.flags & wire::FLAG_WRITE_COMPLETION != 0
    }

    pub fn input_elements(&self) -> u64 {
        match &self.payload {
            StatePayload::Mamba2(payload) => {
                let d_inner = u64::from(self.num_heads) * u64::from(payload.head_dim);
                d_inner * 2
                    + 2 * u64::from(payload.groups) * u64::from(payload.state_dim)
                    + u64::from(self.num_heads)
            }
            StatePayload::Kda(payload) => {
                u64::from(self.num_heads)
                    * (3 * u64::from(payload.key_dim) + u64::from(payload.value_dim) + 1)
            }
        }
    }

    pub fn output_elements(&self) -> u64 {
        match &self.payload {
            StatePayload::Mamba2(payload) => {
                u64::from(self.num_heads) * u64::from(payload.head_dim)
            }
            StatePayload::Kda(payload) => u64::from(self.num_heads) * u64::from(payload.value_dim),
        }
    }

    pub fn resident_bytes(&self) -> u64 {
        u64::from(self.state_bytes)
            + u64::from(self.conv_state_bytes)
            + u64::from(self.state_scale_bytes)
            + u64::from(self.conv_state_scale_bytes)
    }

    fn validate(&self) -> Result<(), DescriptorError> {
        if self.batch_size == 0
            || self.num_heads == 0
            || self.sequence_length == 0
            || self.valid_tokens == 0
            || self.chunk_size == 0
        {
            return Err(DescriptorError::invalid(
                "batch, heads, sequence, valid tokens, and chunk size must be positive",
            ));
        }
        if self.valid_tokens > self.chunk_size {
            return Err(DescriptorError::invalid(
                "valid_tokens cannot exceed chunk_size",
            ));
        }
        if self
            .token_offset
            .checked_add(u32::from(self.valid_tokens))
            .is_none_or(|end| end > self.sequence_length)
        {
            return Err(DescriptorError::invalid(
                "descriptor token range exceeds sequence_length",
            ));
        }
        if self.input_token_stride < u32::try_from(self.input_elements()).unwrap_or(u32::MAX)
            || self.output_token_stride < u32::try_from(self.output_elements()).unwrap_or(u32::MAX)
        {
            return Err(DescriptorError::invalid(
                "VRAM token stride is smaller than the logical tensor",
            ));
        }
        if !self.is_streaming() && !self.state_sram_offset.is_multiple_of(64) {
            return Err(DescriptorError::address(
                "resident state SRAM offset must be 64-byte aligned",
            ));
        }
        for (name, address) in [
            ("state_hbm_addr", self.state_hbm_addr),
            ("conv_state_hbm_addr", self.conv_state_hbm_addr),
            ("state_scale_addr", self.state_scale_addr),
            ("completion_addr", self.completion_addr),
        ] {
            if !address.is_multiple_of(64) {
                return Err(DescriptorError::address(format!(
                    "{name} must be 64-byte aligned"
                )));
            }
        }
        if self.writes_completion() && self.completion_addr == 0 {
            return Err(DescriptorError::address(
                "WRITE_COMPLETION requires a nonzero completion_addr",
            ));
        }
        self.state_hbm_addr
            .checked_add(u64::from(self.state_bytes))
            .ok_or_else(|| DescriptorError::address("state HBM range overflows u64"))?;
        self.conv_state_hbm_addr
            .checked_add(u64::from(self.conv_state_bytes))
            .ok_or_else(|| DescriptorError::address("conv state HBM range overflows u64"))?;

        let expected_state_elements;
        let expected_conv_elements;
        let expected_state_scale_bytes;
        let expected_conv_scale_bytes;
        match &self.payload {
            StatePayload::Mamba2(payload) => {
                if payload.head_dim == 0
                    || payload.state_dim == 0
                    || payload.groups == 0
                    || payload.conv_kernel == 0
                    || !self.num_heads.is_multiple_of(payload.groups)
                    || payload.dt_min < 0.0
                    || payload.dt_max < payload.dt_min
                    // `dt_max < dt_min` is false when either side is NaN, and
                    // f32::clamp panics on a NaN bound, so both need checking.
                    || payload.dt_min.is_nan()
                    || payload.dt_max.is_nan()
                {
                    return Err(DescriptorError::invalid("invalid Mamba2 payload shape"));
                }
                let heads = u64::from(self.num_heads);
                let head_dim = u64::from(payload.head_dim);
                let state_dim = u64::from(payload.state_dim);
                let groups = u64::from(payload.groups);
                let d_inner = heads * head_dim;
                let xbc_elements = d_inner + 2 * groups * state_dim;
                validate_segments(
                    self.input_token_stride,
                    &[
                        (0, d_inner, "gate"),
                        (u64::from(payload.xbc_offset), xbc_elements, "xbc"),
                        (u64::from(payload.dt_offset), heads, "dt"),
                    ],
                )?;
                expected_state_elements = u64::from(self.batch_size) * heads * head_dim * state_dim;
                expected_conv_elements = u64::from(self.batch_size)
                    * (d_inner + 2 * groups * state_dim)
                    * u64::from(payload.conv_kernel);
                expected_state_scale_bytes = if self.state_precision == StatePrecision::Mx8B128 {
                    u64::from(self.batch_size) * heads * head_dim * state_dim.div_ceil(128)
                } else {
                    0
                };
                expected_conv_scale_bytes = if self.conv_state_precision == StatePrecision::Mx8B128
                {
                    u64::from(self.batch_size)
                        * (d_inner + 2 * groups * state_dim)
                        * u64::from(payload.conv_kernel).div_ceil(128)
                } else {
                    0
                };
                if self.parameter_precision == StatePrecision::Mx8B128
                    && payload.parameter_scale_addr == 0
                {
                    return Err(DescriptorError::invalid(
                        "MX8 Mamba2 parameters require parameter_scale_addr",
                    ));
                }
            }
            StatePayload::Kda(payload) => {
                if payload.key_dim == 0
                    || payload.value_dim == 0
                    || payload.conv_kernel == 0
                    || !payload.output_scale.is_finite()
                    || payload.output_scale <= 0.0
                    || !payload.gate_lower_bound.is_finite()
                    || payload.gate_lower_bound >= 0.0
                {
                    return Err(DescriptorError::invalid("invalid KDA payload shape"));
                }
                let heads = u64::from(self.num_heads);
                let key_elements = heads * u64::from(payload.key_dim);
                let value_elements = heads * u64::from(payload.value_dim);
                validate_segments(
                    self.input_token_stride,
                    &[
                        (u64::from(payload.q_offset), key_elements, "q"),
                        (u64::from(payload.k_offset), key_elements, "k"),
                        (u64::from(payload.v_offset), value_elements, "v"),
                        (u64::from(payload.decay_offset), key_elements, "decay"),
                        (u64::from(payload.beta_offset), heads, "beta"),
                    ],
                )?;
                expected_state_elements = u64::from(self.batch_size)
                    * heads
                    * u64::from(payload.value_dim)
                    * u64::from(payload.key_dim);
                expected_conv_elements = u64::from(self.batch_size)
                    * (2 * key_elements + value_elements)
                    * u64::from(payload.conv_kernel);
                expected_state_scale_bytes = if self.state_precision == StatePrecision::Mx8B128 {
                    u64::from(self.batch_size)
                        * heads
                        * u64::from(payload.value_dim)
                        * u64::from(payload.key_dim).div_ceil(128)
                } else {
                    0
                };
                expected_conv_scale_bytes = if self.conv_state_precision == StatePrecision::Mx8B128
                {
                    u64::from(self.batch_size)
                        * (2 * key_elements + value_elements)
                        * u64::from(payload.conv_kernel).div_ceil(128)
                } else {
                    0
                };
                if self.parameter_precision == StatePrecision::Mx8B128
                    && payload.parameter_scale_addr == 0
                {
                    return Err(DescriptorError::invalid(
                        "MX8 KDA parameters require parameter_scale_addr",
                    ));
                }
            }
        }
        if (self.state_precision == StatePrecision::Mx8B128
            || self.conv_state_precision == StatePrecision::Mx8B128)
            && self.state_scale_addr == 0
        {
            return Err(DescriptorError::invalid(
                "MX8 state storage requires state_scale_addr",
            ));
        }
        let element_bytes = self.state_precision.element_bytes() as u64;
        let conv_element_bytes = self.conv_state_precision.element_bytes() as u64;
        let expected_state_bytes = expected_state_elements
            .checked_mul(element_bytes)
            .ok_or_else(|| DescriptorError::invalid("state allocation overflows"))?;
        let expected_conv_bytes = expected_conv_elements
            .checked_mul(conv_element_bytes)
            .ok_or_else(|| DescriptorError::invalid("conv state allocation overflows"))?;
        if expected_state_bytes != u64::from(self.state_bytes) {
            return Err(DescriptorError::invalid(
                "state_bytes does not match shape and precision",
            ));
        }
        if expected_conv_bytes != u64::from(self.conv_state_bytes) {
            return Err(DescriptorError::invalid(
                "conv_state_bytes does not match shape and precision",
            ));
        }
        if expected_state_scale_bytes != u64::from(self.state_scale_bytes) {
            return Err(DescriptorError::invalid(
                "state_scale_bytes does not match shape and precision",
            ));
        }
        if expected_conv_scale_bytes != u64::from(self.conv_state_scale_bytes) {
            return Err(DescriptorError::invalid(
                "conv_state_scale_bytes does not match shape and precision",
            ));
        }
        u64::from(self.state_sram_offset)
            .checked_add(self.resident_bytes())
            .ok_or_else(|| DescriptorError::address("state SRAM allocation overflows u64"))?;
        Ok(())
    }
}

fn validate_segments(
    stride: u32,
    segments: &[(u64, u64, &'static str)],
) -> Result<(), DescriptorError> {
    for (index, (start, len, name)) in segments.iter().copied().enumerate() {
        let end = start
            .checked_add(len)
            .ok_or_else(|| DescriptorError::invalid(format!("{name} VRAM range overflows")))?;
        if end > u64::from(stride) {
            return Err(DescriptorError::invalid(format!(
                "{name} VRAM range exceeds input_token_stride"
            )));
        }
        for (other_start, other_len, other_name) in segments[..index].iter().copied() {
            let other_end = other_start + other_len;
            if start < other_end && other_start < end {
                return Err(DescriptorError::invalid(format!(
                    "{name} and {other_name} VRAM ranges overlap"
                )));
            }
        }
    }
    Ok(())
}

fn precision_at(data: &[u8], offset: usize) -> Result<StatePrecision, DescriptorError> {
    let value = data[offset];
    StatePrecision::try_from(value).map_err(|_| DescriptorError::unsupported_precision(value))
}

fn require_zero(
    data: &[u8],
    range: std::ops::Range<usize>,
    name: &str,
) -> Result<(), DescriptorError> {
    if data[range].iter().any(|value| *value != 0) {
        return Err(DescriptorError::invalid(format!("{name} must be zero")));
    }
    Ok(())
}

fn u16_at(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap())
}

fn u32_at(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

fn u64_at(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn put_u16(data: &mut [u8], offset: usize, value: u16) {
        data[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(data: &mut [u8], offset: usize, value: u32) {
        data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(data: &mut [u8], offset: usize, value: u64) {
        data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn decode_hex(value: &str) -> Vec<u8> {
        assert!(value.len().is_multiple_of(2));
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|digits| u8::from_str_radix(std::str::from_utf8(digits).unwrap(), 16).unwrap())
            .collect()
    }

    fn mamba_descriptor() -> [u8; wire::DESCRIPTOR_SIZE] {
        let mut data = [0u8; wire::DESCRIPTOR_SIZE];
        put_u32(&mut data, wire::common::MAGIC, wire::DESCRIPTOR_MAGIC);
        put_u16(&mut data, wire::common::VERSION, wire::DESCRIPTOR_VERSION);
        put_u16(
            &mut data,
            wire::common::SIZE_BYTES,
            wire::DESCRIPTOR_SIZE as u16,
        );
        data[wire::common::ALGORITHM] = StateAlgorithm::Mamba2 as u8;
        data[wire::common::STATE_PRECISION] = StatePrecision::Fp32 as u8;
        data[wire::common::ACTIVATION_PRECISION] = StatePrecision::Bf16 as u8;
        data[wire::common::ACCUMULATOR_PRECISION] = StatePrecision::Fp32 as u8;
        put_u16(&mut data, wire::common::BATCH_SIZE, 1);
        put_u16(&mut data, wire::common::NUM_HEADS, 64);
        put_u32(&mut data, wire::common::SEQUENCE_LENGTH, 1);
        put_u16(&mut data, wire::common::VALID_TOKENS, 1);
        put_u16(&mut data, wire::common::CHUNK_SIZE, 128);
        put_u32(
            &mut data,
            wire::common::STATE_SRAM_OFFSET,
            wire::STREAMING_SRAM_OFFSET,
        );
        put_u32(&mut data, wire::common::STATE_BYTES, 2 * 1024 * 1024);
        put_u32(&mut data, wire::common::CONV_STATE_BYTES, 96 * 1024);
        put_u32(&mut data, wire::common::INPUT_TOKEN_STRIDE, 10_304);
        put_u32(&mut data, wire::common::OUTPUT_TOKEN_STRIDE, 4_096);
        put_u64(&mut data, wire::common::STATE_HBM_ADDR, 0x10_0000);
        put_u64(&mut data, wire::common::CONV_STATE_HBM_ADDR, 0x40_0000);
        put_u32(&mut data, wire::common::DEPENDENCY_EVENT, wire::NO_EVENT);
        put_u32(&mut data, wire::common::COMPLETION_EVENT, wire::NO_EVENT);
        put_u16(&mut data, wire::mamba2::HEAD_DIM, 64);
        put_u16(&mut data, wire::mamba2::STATE_DIM, 128);
        put_u16(&mut data, wire::mamba2::GROUPS, 8);
        put_u16(&mut data, wire::mamba2::CONV_KERNEL, 4);
        put_u32(&mut data, wire::mamba2::XBC_OFFSET, 4096);
        put_u32(&mut data, wire::mamba2::DT_OFFSET, 10_240);
        put_u32(
            &mut data,
            wire::mamba2::DT_MAX_F32_BITS,
            f32::INFINITY.to_bits(),
        );
        data
    }

    #[test]
    fn parses_real_nemotron_shape() {
        let descriptor = StateDescriptor::parse(&mamba_descriptor()).unwrap();
        assert_eq!(descriptor.algorithm, StateAlgorithm::Mamba2);
        assert_eq!(descriptor.state_bytes, 2 * 1024 * 1024);
        assert_eq!(descriptor.conv_state_bytes, 96 * 1024);
        assert!(descriptor.is_streaming());
    }

    #[test]
    fn rejects_reserved_bytes_and_bad_state_size() {
        let mut reserved = mamba_descriptor();
        reserved[255] = 1;
        assert_eq!(
            StateDescriptor::parse(&reserved).unwrap_err().status,
            StateStatus::InvalidDescriptor
        );

        let mut wrong_size = mamba_descriptor();
        put_u32(&mut wrong_size, wire::common::STATE_BYTES, 64);
        assert!(
            StateDescriptor::parse(&wrong_size)
                .unwrap_err()
                .to_string()
                .contains("state_bytes")
        );
    }

    #[test]
    fn rejects_overlapping_projection_segments() {
        let mut data = mamba_descriptor();
        put_u32(&mut data, wire::mamba2::DT_OFFSET, 4096);
        assert!(
            StateDescriptor::parse(&data)
                .unwrap_err()
                .to_string()
                .contains("overlap")
        );
    }

    #[test]
    fn parses_compiler_generated_golden_descriptors() {
        let golden: Value =
            serde_json::from_str(include_str!("../../testdata/x_state_v2_golden.json")).unwrap();
        assert_eq!(golden["instruction_words"]["step"], "00d0c87d");
        assert_eq!(golden["instruction_words"]["fence"], "018c003d");

        let mamba = decode_hex(
            golden["descriptors"]["mamba2_real"]["hex"]
                .as_str()
                .unwrap(),
        );
        let mamba = StateDescriptor::parse(&mamba).unwrap();
        assert_eq!(mamba.algorithm, StateAlgorithm::Mamba2);
        assert_eq!(mamba.state_bytes, 2 * 1024 * 1024);
        assert_eq!(mamba.parameter_precision, StatePrecision::Bf16);

        let kda = decode_hex(golden["descriptors"]["kda_real"]["hex"].as_str().unwrap());
        let kda = StateDescriptor::parse(&kda).unwrap();
        assert_eq!(kda.algorithm, StateAlgorithm::Kda);
        assert_eq!(kda.state_bytes, 6 * 1024 * 1024);
        assert_eq!(kda.conv_state_bytes, 288 * 1024);
        assert_eq!(kda.state_precision, StatePrecision::Fp32);
        assert_eq!(kda.conv_state_precision, StatePrecision::Bf16);
    }
}
