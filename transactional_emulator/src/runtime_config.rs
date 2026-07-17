use std::sync::LazyLock;

use quantize::{DataType, MxDataType};
use runtime::Duration;

use crate::load_config::*;

pub(crate) static PERIOD: LazyLock<Duration> =
    LazyLock::new(|| Duration::from_picos(clock_period_ps() as u64));
pub(crate) static CLOCK_PERIOD_PS: LazyLock<u32> = LazyLock::new(clock_period_ps);

pub(crate) static SYSTOLIC_PROCESSING_OVERHEAD: LazyLock<u32> =
    LazyLock::new(|| systolic_processing_overhead());
pub(crate) static VECTOR_ADD_CYCLES: LazyLock<u32> = LazyLock::new(|| vector_add_cycles());
pub(crate) static VECTOR_MUL_CYCLES: LazyLock<u32> = LazyLock::new(|| vector_mul_cycles());
pub(crate) static VECTOR_EXP_CYCLES: LazyLock<u32> = LazyLock::new(|| vector_exp_cycles());
pub(crate) static VECTOR_RECI_CYCLES: LazyLock<u32> = LazyLock::new(|| vector_reci_cycles());
pub(crate) static VECTOR_MAX_CYCLES: LazyLock<u32> = LazyLock::new(|| vector_max_cycles());
pub(crate) static VECTOR_SUM_CYCLES: LazyLock<u32> = LazyLock::new(|| vector_sum_cycles());
pub(crate) static SCALAR_FP_BASIC_CYCLES: LazyLock<u32> =
    LazyLock::new(|| scalar_fp_basic_cycles());
pub(crate) static SCALAR_FP_EXP_CYCLES: LazyLock<u32> = LazyLock::new(|| scalar_fp_exp_cycles());
pub(crate) static SCALAR_FP_SQRT_CYCLES: LazyLock<u32> = LazyLock::new(|| scalar_fp_sqrt_cycles());
pub(crate) static SCALAR_FP_RECI_CYCLES: LazyLock<u32> = LazyLock::new(|| scalar_fp_reci_cycles());
pub(crate) static SCALAR_INT_BASIC_CYCLES: LazyLock<u32> =
    LazyLock::new(|| scalar_int_basic_cycles());
pub(crate) static MAX_LOOP_INSTRUCTIONS: LazyLock<usize> =
    LazyLock::new(|| max_loop_instructions());

pub(crate) static MLEN: LazyLock<u32> = LazyLock::new(|| mlen());
pub(crate) static VLEN: LazyLock<u32> = LazyLock::new(|| vlen());
pub(crate) static BLEN: LazyLock<u32> = LazyLock::new(|| blen());
pub(crate) static HLEN: LazyLock<u32> = LazyLock::new(|| hlen());
pub(crate) static BROADCAST_AMOUNT: LazyLock<u32> = LazyLock::new(|| broadcast_amount());
pub(crate) static HBM_SIZE: LazyLock<usize> = LazyLock::new(|| hbm_size());
pub(crate) static HBM_CHANNELS: LazyLock<u32> = LazyLock::new(|| hbm_channels());
pub(crate) static MATRIX_SRAM_SIZE: LazyLock<usize> = LazyLock::new(|| matrix_sram_size());
pub(crate) static VECTOR_SRAM_SIZE: LazyLock<usize> = LazyLock::new(|| vector_sram_size());
pub(crate) static MATRIX_SRAM_TYPE: LazyLock<MxDataType> = LazyLock::new(|| matrix_sram_type());
pub(crate) static VECTOR_SRAM_TYPE: LazyLock<MxDataType> = LazyLock::new(|| vector_sram_type());
pub(crate) static SCALAR_FP_TYPE: LazyLock<DataType> = LazyLock::new(|| scalar_fp_type());
pub(crate) static MATRIX_WEIGHT_TYPE: LazyLock<MxDataType> = LazyLock::new(|| matrix_weight_type());
pub(crate) static MATRIX_KV_TYPE: LazyLock<MxDataType> = LazyLock::new(|| matrix_kv_type());
pub(crate) static VECTOR_ACTIVATION_TYPE: LazyLock<MxDataType> =
    LazyLock::new(|| vector_activation_type());
pub(crate) static VECTOR_KV_TYPE: LazyLock<MxDataType> = LazyLock::new(|| vector_kv_type());
pub(crate) static PREFETCH_M_AMOUNT: LazyLock<u32> = LazyLock::new(hbm_m_prefetch_amount);
pub(crate) static PREFETCH_V_AMOUNT: LazyLock<u32> = LazyLock::new(|| hbm_v_prefetch_amount());
pub(crate) static STORE_V_AMOUNT: LazyLock<u32> = LazyLock::new(|| hbm_v_writeback_amount());
