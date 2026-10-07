use std::sync::LazyLock;

use quantize::MxDataType;
use runtime::Duration;

use crate::load_config::*;

/// The accelerator clock period, from `[<MODE>.CONFIG.CLOCK_PERIOD_PS]`.
///
/// This was a `const` of 1 ns with no stated basis, and every second,
/// microsecond and TPOT figure the emulator reports is that number's
/// consequence. It is still 1 ns by default -- nothing has measured it, because
/// no RTL has been synthesised and no critical path has set a frequency -- but
/// it now lives in a configuration file where the assumption is visible and
/// changeable, and `runner` checks its relationship to the DRAM model's own
/// clock at startup instead of leaving the two to coincide.
pub(crate) static PERIOD: LazyLock<Duration> =
    LazyLock::new(|| Duration::from_picos(clock_period_ps()));

pub(crate) static SYSTOLIC_PROCESSING_OVERHEAD: LazyLock<u32> =
    LazyLock::new(systolic_processing_overhead);
pub(crate) static VECTOR_ADD_CYCLES: LazyLock<u32> = LazyLock::new(vector_add_cycles);
pub(crate) static VECTOR_MUL_CYCLES: LazyLock<u32> = LazyLock::new(vector_mul_cycles);
pub(crate) static VECTOR_EXP_CYCLES: LazyLock<u32> = LazyLock::new(vector_exp_cycles);
pub(crate) static VECTOR_RECI_CYCLES: LazyLock<u32> = LazyLock::new(vector_reci_cycles);
pub(crate) static VECTOR_MAX_CYCLES: LazyLock<u32> = LazyLock::new(vector_max_cycles);
// V_MIN_VF is a single-pass scalar clamp with the same cost as V_MAX_VF; alias
// to the max latency rather than add a redundant knob to every config file.
pub(crate) static VECTOR_MIN_CYCLES: LazyLock<u32> = LazyLock::new(vector_max_cycles);
pub(crate) static VECTOR_SUM_CYCLES: LazyLock<u32> = LazyLock::new(vector_sum_cycles);
pub(crate) static VECTOR_SOFTPLUS_CYCLES: LazyLock<u32> = LazyLock::new(vector_softplus_cycles);

/// Optional finite-width SFU service contract. This is an implementation
/// candidate, not synthesis evidence. All five parameters are mandatory when
/// enabled; never interpret a historical whole-vector latency as a measured
/// per-lane latency. Instructions retire serially after the final subchunk.
pub(crate) static VECTOR_SFU: LazyLock<Option<VectorSfu>> = LazyLock::new(|| {
    let names = [
        "PLENA_VECTOR_SFU_LANES",
        "PLENA_VECTOR_SFU_II",
        "PLENA_VECTOR_SFU_EXP_LATENCY",
        "PLENA_VECTOR_SFU_SOFTPLUS_LATENCY",
        "PLENA_VECTOR_SFU_RECI_LATENCY",
    ];
    let values = names.map(|name| std::env::var(name).ok());
    if values.iter().all(Option::is_none) {
        return None;
    }
    let values = std::array::from_fn::<_, 5, _>(|i| {
        let value = values[i]
            .as_ref()
            .unwrap_or_else(|| panic!("{} required for finite SFU", names[i]));
        let value = value.parse::<u32>().expect("SFU parameter must be uint32");
        assert!(value > 0, "SFU parameter must be positive");
        value
    });
    Some(VectorSfu {
        lanes: values[0],
        ii: values[1],
        exp: values[2],
        softplus: values[3],
        reciprocal: values[4],
    })
});

pub(crate) struct VectorSfu {
    pub lanes: u32,
    pub ii: u32,
    pub exp: u32,
    pub softplus: u32,
    pub reciprocal: u32,
}

impl VectorSfu {
    pub fn service(&self, elements: u32, latency: u32) -> u32 {
        assert!(elements > 0 && self.lanes <= elements);
        (elements.div_ceil(self.lanes) - 1)
            .checked_mul(self.ii)
            .and_then(|n| n.checked_add(latency))
            .expect("SFU service overflow")
    }
}
pub(crate) static SCALAR_FP_BASIC_CYCLES: LazyLock<u32> = LazyLock::new(scalar_fp_basic_cycles);
pub(crate) static SCALAR_FP_EXP_CYCLES: LazyLock<u32> = LazyLock::new(scalar_fp_exp_cycles);
pub(crate) static SCALAR_FP_SQRT_CYCLES: LazyLock<u32> = LazyLock::new(scalar_fp_sqrt_cycles);
pub(crate) static SCALAR_FP_RECI_CYCLES: LazyLock<u32> = LazyLock::new(scalar_fp_reci_cycles);
pub(crate) static SCALAR_INT_BASIC_CYCLES: LazyLock<u32> = LazyLock::new(scalar_int_basic_cycles);
pub(crate) static MAX_LOOP_INSTRUCTIONS: LazyLock<usize> = LazyLock::new(max_loop_instructions);

pub(crate) static MLEN: LazyLock<u32> = LazyLock::new(mlen);
pub(crate) static VLEN: LazyLock<u32> = LazyLock::new(vlen);
pub(crate) static BLEN: LazyLock<u32> = LazyLock::new(blen);
pub(crate) static HLEN: LazyLock<u32> = LazyLock::new(hlen);
pub(crate) static BROADCAST_AMOUNT: LazyLock<u32> = LazyLock::new(broadcast_amount);
pub(crate) static HBM_SIZE: LazyLock<usize> = LazyLock::new(hbm_size);
pub(crate) static MATRIX_SRAM_SIZE: LazyLock<usize> = LazyLock::new(matrix_sram_size);
pub(crate) static VECTOR_SRAM_SIZE: LazyLock<usize> = LazyLock::new(vector_sram_size);
pub(crate) static MATRIX_SRAM_TYPE: LazyLock<MxDataType> = LazyLock::new(matrix_sram_type);
pub(crate) static VECTOR_SRAM_TYPE: LazyLock<MxDataType> = LazyLock::new(vector_sram_type);
pub(crate) static MATRIX_WEIGHT_TYPE: LazyLock<MxDataType> = LazyLock::new(matrix_weight_type);
pub(crate) static MATRIX_KV_TYPE: LazyLock<MxDataType> = LazyLock::new(matrix_kv_type);
pub(crate) static VECTOR_ACTIVATION_TYPE: LazyLock<MxDataType> =
    LazyLock::new(vector_activation_type);
pub(crate) static VECTOR_KV_TYPE: LazyLock<MxDataType> = LazyLock::new(vector_kv_type);
pub(crate) static STATE_TYPE: LazyLock<MxDataType> = LazyLock::new(state_type);
pub(crate) static PREFETCH_M_AMOUNT: LazyLock<u32> = LazyLock::new(|| {
    let raw = hbm_m_prefetch_amount();
    let mlen = mlen();
    // Must be a multiple of MLEN (one full matrix tile per write).
    // Round up to the nearest multiple of MLEN if needed.
    if raw < mlen {
        tracing::warn!(
            "HBM_M_Prefetch_Amount ({}) < MLEN ({}); clamping to MLEN",
            raw,
            mlen
        );
        mlen
    } else if !raw.is_multiple_of(mlen) {
        let clamped = raw.div_ceil(mlen) * mlen;
        tracing::warn!(
            "HBM_M_Prefetch_Amount ({}) not a multiple of MLEN ({}); rounding up to {}",
            raw,
            mlen,
            clamped
        );
        clamped
    } else {
        raw
    }
});
pub(crate) static PREFETCH_V_AMOUNT: LazyLock<u32> = LazyLock::new(hbm_v_prefetch_amount);
pub(crate) static STORE_V_AMOUNT: LazyLock<u32> = LazyLock::new(hbm_v_writeback_amount);
