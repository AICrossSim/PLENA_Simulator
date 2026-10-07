//! Generated from PLENA_RTL/spec/plena_contract.json; do not edit.

pub const CONTRACT_SHA256: &str =
    "09ca12098ad0c232f98de97bdfa145eddc27d06898d0b0f52c60ad14eed4ffc5";
pub const X_MAMBA_OPCODE: u8 = 0x39;
pub const MAMBA_DESCRIPTOR_MAGIC: u32 = 0x32424d50;
pub const MAMBA_DESCRIPTOR_VERSION: u16 = 1;
pub const MAMBA_DESCRIPTOR_BYTES: usize = 256;
pub const MAMBA_DESCRIPTOR_ALIGNMENT: usize = 64;
pub const MAMBA_COMPLETION_BYTES: usize = 16;
pub const MAMBA_COMPLETION_ALIGNMENT: usize = 64;
pub const MAMBA_NO_EVENT: u32 = 0xffffffff;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum MambaSubop {
    Prefill = 0,
    Step = 1,
    StateReset = 2,
    Wait = 3,
}

impl TryFrom<u8> for MambaSubop {
    type Error = u8;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Prefill),
            1 => Ok(Self::Step),
            2 => Ok(Self::StateReset),
            3 => Ok(Self::Wait),
            other => Err(other),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum MambaPrecisionPolicy {
    Fp32Reference = 0,
    Bf16ActivationFp32State = 1,
}

pub const MAMBA_FLAG_CONTINUE_STATE_BIT: u32 = 0;
pub const MAMBA_FLAG_WRITE_COMPLETION_BIT: u32 = 1;
pub const MAMBA_FLAG_PROFILE_BIT: u32 = 2;

pub const MAMBA_DESC_MAGIC_OFFSET: usize = 0;
pub const MAMBA_DESC_VERSION_OFFSET: usize = 4;
pub const MAMBA_DESC_SIZE_BYTES_OFFSET: usize = 6;
pub const MAMBA_DESC_FLAGS_OFFSET: usize = 8;
pub const MAMBA_DESC_CONTEXT_ID_OFFSET: usize = 12;
pub const MAMBA_DESC_SEQUENCE_LENGTH_OFFSET: usize = 16;
pub const MAMBA_DESC_BATCH_SIZE_OFFSET: usize = 20;
pub const MAMBA_DESC_D_MODEL_OFFSET: usize = 24;
pub const MAMBA_DESC_D_INNER_OFFSET: usize = 28;
pub const MAMBA_DESC_NUM_HEADS_OFFSET: usize = 32;
pub const MAMBA_DESC_HEAD_DIM_OFFSET: usize = 36;
pub const MAMBA_DESC_STATE_DIM_OFFSET: usize = 40;
pub const MAMBA_DESC_GROUPS_OFFSET: usize = 44;
pub const MAMBA_DESC_CHUNK_SIZE_OFFSET: usize = 48;
pub const MAMBA_DESC_CONV_KERNEL_OFFSET: usize = 52;
pub const MAMBA_DESC_PRECISION_POLICY_OFFSET: usize = 56;
pub const MAMBA_DESC_RMS_NORM_EPS_F32_BITS_OFFSET: usize = 60;
pub const MAMBA_DESC_INPUT_ADDR_OFFSET: usize = 64;
pub const MAMBA_DESC_OUTPUT_ADDR_OFFSET: usize = 72;
pub const MAMBA_DESC_IN_PROJ_WEIGHT_ADDR_OFFSET: usize = 80;
pub const MAMBA_DESC_IN_PROJ_BIAS_ADDR_OFFSET: usize = 88;
pub const MAMBA_DESC_CONV_WEIGHT_ADDR_OFFSET: usize = 96;
pub const MAMBA_DESC_CONV_BIAS_ADDR_OFFSET: usize = 104;
pub const MAMBA_DESC_A_LOG_ADDR_OFFSET: usize = 112;
pub const MAMBA_DESC_D_SKIP_ADDR_OFFSET: usize = 120;
pub const MAMBA_DESC_NORM_WEIGHT_ADDR_OFFSET: usize = 128;
pub const MAMBA_DESC_OUT_PROJ_WEIGHT_ADDR_OFFSET: usize = 136;
pub const MAMBA_DESC_OUT_PROJ_BIAS_ADDR_OFFSET: usize = 144;
pub const MAMBA_DESC_SSM_STATE_ADDR_OFFSET: usize = 152;
pub const MAMBA_DESC_CONV_STATE_ADDR_OFFSET: usize = 160;
pub const MAMBA_DESC_SCRATCH_ADDR_OFFSET: usize = 168;
pub const MAMBA_DESC_COMPLETION_ADDR_OFFSET: usize = 176;
pub const MAMBA_DESC_DT_BIAS_ADDR_OFFSET: usize = 184;
pub const MAMBA_DESC_INPUT_BATCH_STRIDE_OFFSET: usize = 192;
pub const MAMBA_DESC_INPUT_TOKEN_STRIDE_OFFSET: usize = 196;
pub const MAMBA_DESC_OUTPUT_BATCH_STRIDE_OFFSET: usize = 200;
pub const MAMBA_DESC_OUTPUT_TOKEN_STRIDE_OFFSET: usize = 204;
pub const MAMBA_DESC_STATE_REQUEST_STRIDE_OFFSET: usize = 208;
pub const MAMBA_DESC_STATE_HEAD_STRIDE_OFFSET: usize = 212;
pub const MAMBA_DESC_CONV_REQUEST_STRIDE_OFFSET: usize = 216;
pub const MAMBA_DESC_SCRATCH_BYTES_OFFSET: usize = 220;
pub const MAMBA_DESC_DEPENDENCY_EVENT_OFFSET: usize = 224;
pub const MAMBA_DESC_COMPLETION_EVENT_OFFSET: usize = 228;
pub const MAMBA_DESC_LAYER_ID_OFFSET: usize = 232;
pub const MAMBA_DESC_DT_MIN_F32_BITS_OFFSET: usize = 236;
pub const MAMBA_DESC_DT_MAX_F32_BITS_OFFSET: usize = 240;
pub const MAMBA_DESC_D_MLP_OFFSET: usize = 244;
pub const MAMBA_DESC_RESERVED3_OFFSET: usize = 248;

pub const MAMBA_CPL_STATUS_OFFSET: usize = 0;
pub const MAMBA_CPL_COMPLETION_EVENT_OFFSET: usize = 4;
pub const MAMBA_CPL_ELAPSED_CYCLES_OFFSET: usize = 8;

pub const MAMBA_STATUS_EMPTY: u32 = 0;
pub const MAMBA_STATUS_SUCCESS: u32 = 1;
pub const MAMBA_STATUS_INVALID_DESCRIPTOR: u32 = 2;
pub const MAMBA_STATUS_UNSUPPORTED_PROFILE: u32 = 3;
pub const MAMBA_STATUS_ADDRESS_ERROR: u32 = 4;
pub const MAMBA_STATUS_STATE_HAZARD: u32 = 5;
pub const MAMBA_STATUS_INTERNAL_ERROR: u32 = 255;
