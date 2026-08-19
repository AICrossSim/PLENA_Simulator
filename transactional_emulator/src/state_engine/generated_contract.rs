// Generated from spec/x_state_v2.json; do not edit by hand.
#![allow(dead_code)]

pub const CONTRACT_SHA256: &str =
    "2edcd9b6ceb22428680f27c9262b4c16015b07f30a26b6b4f23b83625b81b3fc";
pub const X_STATE_OPCODE: u8 = 61;
pub const DESCRIPTOR_MAGIC: u32 = 844387160;
pub const DESCRIPTOR_VERSION: u16 = 2;
pub const DESCRIPTOR_SIZE: usize = 256;
pub const DESCRIPTOR_ALIGNMENT: u64 = 64;
pub const STREAMING_SRAM_OFFSET: u32 = 4294967295;
pub const COMPLETION_SIZE: usize = 16;
pub const COMPLETION_ALIGNMENT: u64 = 64;
pub const NO_EVENT: u32 = 4294967295;

pub const FLAG_LAST_CHUNK: u32 = 1 << 0;
pub const FLAG_WRITE_COMPLETION: u32 = 1 << 1;
pub const FLAG_PROFILE: u32 = 1 << 2;

pub mod instruction {
    pub const OPCODE_LSB: u32 = 0;
    pub const OPCODE_WIDTH: u32 = 6;
    pub const CONTEXT_GP_LSB: u32 = 6;
    pub const CONTEXT_GP_WIDTH: u32 = 4;
    pub const DESCRIPTOR_OFFSET_GP_LSB: u32 = 10;
    pub const DESCRIPTOR_OFFSET_GP_WIDTH: u32 = 4;
    pub const DESCRIPTOR_HBM_REG_LSB: u32 = 14;
    pub const DESCRIPTOR_HBM_REG_WIDTH: u32 = 4;
    pub const QUEUE_ID_LSB: u32 = 18;
    pub const QUEUE_ID_WIDTH: u32 = 4;
    pub const SUBOP_LSB: u32 = 22;
    pub const SUBOP_WIDTH: u32 = 4;
    pub const RESERVED_LSB: u32 = 26;
    pub const RESERVED_WIDTH: u32 = 6;
}

pub mod common {
    pub const MAGIC: usize = 0;
    pub const VERSION: usize = 4;
    pub const SIZE_BYTES: usize = 6;
    pub const ALGORITHM: usize = 8;
    pub const STATE_PRECISION: usize = 9;
    pub const ACTIVATION_PRECISION: usize = 10;
    pub const ACCUMULATOR_PRECISION: usize = 11;
    pub const FLAGS: usize = 12;
    pub const CONTEXT_ID: usize = 16;
    pub const REQUEST_ID: usize = 20;
    pub const LAYER_ID: usize = 24;
    pub const STATE_ID: usize = 28;
    pub const BATCH_SIZE: usize = 32;
    pub const NUM_HEADS: usize = 34;
    pub const SEQUENCE_LENGTH: usize = 36;
    pub const TOKEN_OFFSET: usize = 40;
    pub const VALID_TOKENS: usize = 44;
    pub const CHUNK_SIZE: usize = 46;
    pub const STATE_SRAM_OFFSET: usize = 48;
    pub const STATE_BYTES: usize = 52;
    pub const CONV_STATE_BYTES: usize = 56;
    pub const PARAMETER_PRECISION: usize = 60;
    pub const CONV_STATE_PRECISION: usize = 61;
    pub const RESERVED0: usize = 62;
    pub const INPUT_VRAM_ADDR: usize = 64;
    pub const OUTPUT_VRAM_ADDR: usize = 68;
    pub const INPUT_TOKEN_STRIDE: usize = 72;
    pub const OUTPUT_TOKEN_STRIDE: usize = 76;
    pub const STATE_HBM_ADDR: usize = 80;
    pub const CONV_STATE_HBM_ADDR: usize = 88;
    pub const STATE_SCALE_ADDR: usize = 96;
    pub const COMPLETION_ADDR: usize = 104;
    pub const DEPENDENCY_EVENT: usize = 112;
    pub const COMPLETION_EVENT: usize = 116;
    pub const STATE_SCALE_BYTES: usize = 120;
    pub const CONV_STATE_SCALE_BYTES: usize = 124;
}

pub mod mamba2 {
    pub const HEAD_DIM: usize = 128;
    pub const STATE_DIM: usize = 130;
    pub const GROUPS: usize = 132;
    pub const CONV_KERNEL: usize = 134;
    pub const XBC_OFFSET: usize = 136;
    pub const DT_OFFSET: usize = 140;
    pub const CONV_WEIGHT_ADDR: usize = 144;
    pub const CONV_BIAS_ADDR: usize = 152;
    pub const A_LOG_ADDR: usize = 160;
    pub const DT_BIAS_ADDR: usize = 168;
    pub const D_SKIP_ADDR: usize = 176;
    pub const PARAMETER_SCALE_ADDR: usize = 184;
    pub const DT_MIN_F32_BITS: usize = 192;
    pub const DT_MAX_F32_BITS: usize = 196;
    pub const RESERVED_PAYLOAD: usize = 200;
}
pub mod kda {
    pub const KEY_DIM: usize = 128;
    pub const VALUE_DIM: usize = 130;
    pub const CONV_KERNEL: usize = 132;
    pub const RESERVED_KDA0: usize = 134;
    pub const Q_OFFSET: usize = 136;
    pub const K_OFFSET: usize = 140;
    pub const V_OFFSET: usize = 144;
    pub const DECAY_OFFSET: usize = 148;
    pub const BETA_OFFSET: usize = 152;
    pub const RESERVED_KDA1: usize = 156;
    pub const Q_CONV_WEIGHT_ADDR: usize = 160;
    pub const K_CONV_WEIGHT_ADDR: usize = 168;
    pub const V_CONV_WEIGHT_ADDR: usize = 176;
    pub const Q_CONV_BIAS_ADDR: usize = 184;
    pub const K_CONV_BIAS_ADDR: usize = 192;
    pub const V_CONV_BIAS_ADDR: usize = 200;
    pub const A_LOG_ADDR: usize = 208;
    pub const DT_BIAS_ADDR: usize = 216;
    pub const PARAMETER_SCALE_ADDR: usize = 224;
    pub const OUTPUT_SCALE_F32_BITS: usize = 232;
    pub const GATE_LOWER_BOUND_F32_BITS: usize = 236;
    pub const RESERVED_PAYLOAD: usize = 240;
}

pub mod completion {
    pub const STATUS: usize = 0;
    pub const COMPLETION_EVENT: usize = 4;
    pub const ELAPSED_CYCLES: usize = 8;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum StateAlgorithm {
    Mamba2 = 0,
    Kda = 1,
}

impl TryFrom<u8> for StateAlgorithm {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Mamba2),
            1 => Ok(Self::Kda),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum StateSubop {
    Preload = 0,
    Reset = 1,
    Prefill = 2,
    Step = 3,
    Commit = 4,
    Evict = 5,
    Fence = 6,
}

impl TryFrom<u8> for StateSubop {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Preload),
            1 => Ok(Self::Reset),
            2 => Ok(Self::Prefill),
            3 => Ok(Self::Step),
            4 => Ok(Self::Commit),
            5 => Ok(Self::Evict),
            6 => Ok(Self::Fence),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum StatePrecision {
    Fp32 = 0,
    Bf16 = 1,
    Fp16 = 2,
    Mx8B128 = 3,
}

impl TryFrom<u8> for StatePrecision {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Fp32),
            1 => Ok(Self::Bf16),
            2 => Ok(Self::Fp16),
            3 => Ok(Self::Mx8B128),
            _ => Err(()),
        }
    }
}

impl StatePrecision {
    pub const fn element_bytes(self) -> usize {
        match self {
            Self::Fp32 => 4,
            Self::Bf16 => 2,
            Self::Fp16 => 2,
            Self::Mx8B128 => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum StateStatus {
    Empty = 0,
    Success = 1,
    InvalidDescriptor = 2,
    UnsupportedAlgorithm = 3,
    AddressError = 4,
    StateHazard = 5,
    UnsupportedPrecision = 6,
    InternalError = 255,
}

impl TryFrom<u32> for StateStatus {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Empty),
            1 => Ok(Self::Success),
            2 => Ok(Self::InvalidDescriptor),
            3 => Ok(Self::UnsupportedAlgorithm),
            4 => Ok(Self::AddressError),
            5 => Ok(Self::StateHazard),
            6 => Ok(Self::UnsupportedPrecision),
            255 => Ok(Self::InternalError),
            _ => Err(()),
        }
    }
}
