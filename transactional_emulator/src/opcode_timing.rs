//! RTL-oriented opcode timing used by the `rtl-v1` scheduler.
//!
//! The functional emulator intentionally keeps its legacy delays: changing
//! them would mix numerical and timing changes.  This module instead maps an
//! opcode to two distinct points on the logical RTL timeline:
//!
//! * `resource_cycles`: how long the selected backend remains occupied;
//! * `result_ready_cycles`: when consumers may observe the complete result.
//!
//! The values are kept separate even when a measured path currently reports
//! the same cycle for both. This avoids encoding a false dependency between
//! architectural result visibility and backend-idle behavior.

use std::sync::LazyLock;

use quantize::{DataType, MxDataType};
use serde::{Deserialize, Serialize};

use crate::op::Opcode;
use crate::runtime_config::{
    BLEN, MATRIX_SRAM_TYPE, MATRIX_WEIGHT_TYPE, MLEN, SCALAR_FP_TYPE, VECTOR_SRAM_TYPE, VLEN,
};

const CALIBRATION_JSON: &str = include_str!("../calibration/rtl_opcode_timing_v1.json");

#[derive(Clone, Debug, Deserialize)]
struct TimingCalibration {
    schema_version: u32,
    model: String,
    measurement_boundary: String,
    clock_period_assumption_ps: u32,
    source: CalibrationSource,
    matrix: MatrixCalibration,
    vector: VectorCalibration,
    scalar: ScalarCalibration,
}

#[derive(Clone, Debug, Deserialize)]
struct CalibrationSource {
    rtl_repository: String,
    rtl_head: String,
    rtl_dirty: bool,
    rtl_diff_sha256: String,
    implementation_profile: String,
    dc_library_profile_verified: bool,
    note: String,
}

#[derive(Clone, Debug, Deserialize)]
struct MatrixCalibration {
    mxint: MxIntMatrixCalibration,
    mxfp: MxFpMatrixCalibration,
}

#[derive(Clone, Debug, Deserialize)]
struct MxIntMatrixCalibration {
    gemm_compute_blen_coefficient: u64,
    gemm_compute_fixed_cycles: u64,
    gemm_writeout_blen_coefficient: u64,
    gemm_writeout_fixed_cycles: u64,
    gemv_implemented: bool,
    broadcast_implemented: bool,
    #[allow(dead_code)]
    measured_shapes: Vec<serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize)]
struct MxFpMatrixCalibration {
    gemm_load_blen_coefficient: u64,
    gemm_load_fixed_cycles: u64,
    gemm_writeout_busy_blen_coefficient: u64,
    gemm_writeout_busy_overhead_coefficient: u64,
    gemm_writeout_busy_fixed_cycles: u64,
    gemm_writeout_ready_blen_coefficient: u64,
    gemm_writeout_ready_overhead_coefficient: u64,
    gemm_writeout_ready_fixed_cycles: u64,
    systolic_processing_overhead_cycles: u64,
    gemv_load_mlen_coefficient: u64,
    gemv_load_fixed_cycles: u64,
    gemv_writeout_busy_cycles: u64,
    gemv_writeout_ready_cycles: u64,
    broadcast_implemented: bool,
    gemm_writeout_rtl_supported: bool,
    gemm_writeout_row_writeback_supported: bool,
    measured_operand_format: FpFormat,
    measured_internal_format: FpFormat,
    measured_shapes: Vec<MatrixMeasuredShape>,
}

#[derive(Clone, Debug, Deserialize)]
struct VectorCalibration {
    add_vv_cycles: u64,
    add_vf_cycles: u64,
    sub_vv_cycles: u64,
    sub_vf_cycles: u64,
    mul_vv_cycles: u64,
    mul_vf_cycles: u64,
    exp_cycles: u64,
    reciprocal_cycles: u64,
    reduce_sum_base_cycles: u64,
    reduce_sum_per_level_cycles: u64,
    reduce_max_base_cycles: u64,
    reduce_max_per_level_cycles: u64,
    shift_implemented: bool,
    shift_conservative_cycles: u64,
    initiation_interval_cycles: u64,
    measured_points: Vec<FpMeasuredPoint>,
}

#[derive(Clone, Debug, Deserialize)]
struct ScalarCalibration {
    fp_add_ready_cycles: u64,
    fp_add_done_cycles: u64,
    fp_sub_ready_cycles: u64,
    fp_sub_done_cycles: u64,
    fp_mul_ready_cycles: u64,
    fp_mul_done_cycles: u64,
    fp_exp_ready_cycles: u64,
    fp_exp_done_cycles: u64,
    fp_reciprocal_ready_cycles: u64,
    fp_reciprocal_done_cycles: u64,
    fp_sqrt_ready_cycles: u64,
    fp_sqrt_done_cycles: u64,
    fp_max_implemented: bool,
    fp_max_conservative_cycles: u64,
    int_basic_cycles: u64,
    sram_cycles: u64,
    map_vector_ready_fixed_cycles: u64,
    map_vector_done_fixed_cycles: u64,
    measured_points: Vec<FpMeasuredPoint>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
struct FpFormat {
    exponent: u8,
    mantissa: u8,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct MatrixMeasuredShape {
    mlen: u64,
    blen: u64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct FpMeasuredPoint {
    vlen: u64,
    exponent: u8,
    mantissa: u8,
}

static CALIBRATION: LazyLock<TimingCalibration> = LazyLock::new(|| {
    let calibration: TimingCalibration = serde_json::from_str(CALIBRATION_JSON)
        .expect("embedded RTL opcode timing calibration is valid");
    assert_eq!(calibration.schema_version, 3);
    calibration
});

/// How strongly the selected cycle estimate is tied to implemented RTL.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CalibrationStatus {
    /// Legacy execute-and-await timing from the functional model.
    LegacyFunctionalObserved,
    /// Measured at the complete Machine execute/consumer boundary.
    FullMachineMeasured,
    /// Extrapolated from a measured Machine point and an RTL structure.
    StructuralExtrapolation,
    /// The software ISA has no implemented counterpart in the current RTL.
    UnsupportedRtl,
    /// Timing came from the actual Ramulator request/completion interval.
    RamulatorObserved,
}

impl CalibrationStatus {
    pub(crate) fn as_key(self) -> &'static str {
        match self {
            Self::LegacyFunctionalObserved => "legacy_functional_observed",
            Self::FullMachineMeasured => "full_machine_measured",
            Self::StructuralExtrapolation => "structural_extrapolation",
            Self::UnsupportedRtl => "unsupported_rtl",
            Self::RamulatorObserved => "ramulator_observed",
        }
    }
}

/// Scheduler-facing timing for one instruction.
#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct OpcodeTimingEstimate {
    pub(crate) resource_cycles: u64,
    pub(crate) result_ready_cycles: u64,
    pub(crate) initiation_interval_cycles: u64,
    pub(crate) calibration_status: CalibrationStatus,
    /// Relative cycle at which row zero is available. Non-row-producing
    /// instructions leave this unset and use `result_ready_cycles`.
    pub(crate) first_result_ready_cycles: Option<u64>,
    /// Relative spacing between successive output rows.
    pub(crate) result_cadence_cycles: Option<u64>,
    pub(crate) rtl_supported: bool,
    pub(crate) calibration_in_domain: bool,
}

impl OpcodeTimingEstimate {
    fn fixed(cycles: u64, status: CalibrationStatus) -> Self {
        let cycles = cycles.max(1);
        Self {
            resource_cycles: cycles,
            result_ready_cycles: cycles,
            initiation_interval_cycles: cycles,
            calibration_status: status,
            first_result_ready_cycles: None,
            result_cadence_cycles: None,
            rtl_supported: !matches!(status, CalibrationStatus::UnsupportedRtl),
            calibration_in_domain: matches!(
                status,
                CalibrationStatus::FullMachineMeasured | CalibrationStatus::RamulatorObserved
            ),
        }
    }

    fn with_validation(mut self, rtl_supported: bool, calibration_in_domain: bool) -> Self {
        self.rtl_supported = rtl_supported;
        self.calibration_in_domain = calibration_in_domain;
        self
    }

    fn split(
        resource_cycles: u64,
        result_ready_cycles: u64,
        initiation_interval_cycles: u64,
        status: CalibrationStatus,
    ) -> Self {
        Self {
            resource_cycles: resource_cycles.max(1),
            result_ready_cycles: result_ready_cycles.max(1),
            initiation_interval_cycles: initiation_interval_cycles.max(1),
            calibration_status: status,
            first_result_ready_cycles: None,
            result_cadence_cycles: None,
            rtl_supported: !matches!(status, CalibrationStatus::UnsupportedRtl),
            calibration_in_domain: matches!(status, CalibrationStatus::FullMachineMeasured),
        }
    }

    pub(crate) fn observed(cycles: u64) -> Self {
        Self::fixed(cycles, CalibrationStatus::RamulatorObserved)
    }
}

/// Named MatrixMachine phases, retained in traces/tests so a cycle estimate is
/// inspectable instead of being a single unexplained constant.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct MatrixTimingBreakdown {
    pub(crate) input_load_cycles: u64,
    pub(crate) systolic_pipeline_cycles: u64,
    pub(crate) reduction_drain_cycles: u64,
    pub(crate) writeout_cycles: u64,
    pub(crate) result_ready_cycles: u64,
    pub(crate) initiation_interval_cycles: u64,
    pub(crate) first_result_ready_cycles: Option<u64>,
    pub(crate) result_cadence_cycles: Option<u64>,
    pub(crate) rtl_supported: bool,
    pub(crate) calibration_in_domain: bool,
    pub(crate) calibration_status: Option<CalibrationStatus>,
}

impl MatrixTimingBreakdown {
    pub(crate) fn total_cycles(self) -> u64 {
        (self.input_load_cycles
            + self.systolic_pipeline_cycles
            + self.reduction_drain_cycles
            + self.writeout_cycles)
            .max(1)
    }

    fn estimate(self) -> OpcodeTimingEstimate {
        OpcodeTimingEstimate {
            resource_cycles: self.total_cycles(),
            result_ready_cycles: self.result_ready_cycles.max(1),
            initiation_interval_cycles: self.initiation_interval_cycles.max(1),
            calibration_status: self
                .calibration_status
                .unwrap_or(CalibrationStatus::StructuralExtrapolation),
            first_result_ready_cycles: self.first_result_ready_cycles,
            result_cadence_cycles: self.result_cadence_cycles,
            rtl_supported: self.rtl_supported,
            calibration_in_domain: self.calibration_in_domain,
        }
    }
}

fn ceil_log2(value: u64) -> u64 {
    if value <= 1 {
        0
    } else {
        u64::from(u64::BITS - (value - 1).leading_zeros())
    }
}

fn matrix_uses_mxint() -> bool {
    matches!(MATRIX_WEIGHT_TYPE.element_type(), DataType::Int(_))
}

fn fp_format(data_type: DataType) -> Option<FpFormat> {
    match data_type {
        DataType::Fp(format) => Some(FpFormat {
            exponent: format.exponent,
            mantissa: format.mantissa,
        }),
        DataType::Int(_) => None,
    }
}

fn mx_fp_format(data_type: MxDataType) -> Option<FpFormat> {
    fp_format(data_type.element_type())
}

fn measured_format(points: &[FpMeasuredPoint], candidate: Option<FpFormat>) -> bool {
    candidate.is_some_and(|candidate| {
        points.iter().any(|point| {
            point.exponent == candidate.exponent && point.mantissa == candidate.mantissa
        })
    })
}

fn measured_fp_point(points: &[FpMeasuredPoint], vlen: u64, candidate: Option<FpFormat>) -> bool {
    candidate.is_some_and(|candidate| {
        points.iter().any(|point| {
            point.vlen == vlen
                && point.exponent == candidate.exponent
                && point.mantissa == candidate.mantissa
        })
    })
}

fn measured_matrix_shape(calibration: &MxFpMatrixCalibration, mlen: u64, blen: u64) -> bool {
    calibration
        .measured_shapes
        .iter()
        .any(|shape| shape.mlen == mlen && shape.blen == blen)
}

fn measured_mxfp_profile(calibration: &MxFpMatrixCalibration) -> bool {
    mx_fp_format(*MATRIX_WEIGHT_TYPE) == Some(calibration.measured_operand_format)
        && mx_fp_format(*MATRIX_SRAM_TYPE) == Some(calibration.measured_internal_format)
}

fn is_broadcast_matrix_opcode(opcode: &Opcode) -> bool {
    matches!(
        opcode,
        Opcode::M_BMM { .. }
            | Opcode::M_BTMM { .. }
            | Opcode::M_BMM_WO { .. }
            | Opcode::M_BMV { .. }
            | Opcode::M_BTMV { .. }
            | Opcode::M_BMV_WO { .. }
    )
}

fn matrix_compute_timing(opcode: &Opcode, gemv: bool) -> MatrixTimingBreakdown {
    let mlen = u64::from(*MLEN);
    let blen = u64::from(*BLEN);
    let broadcast = is_broadcast_matrix_opcode(opcode);

    if matrix_uses_mxint() {
        let c = &CALIBRATION.matrix.mxint;
        if gemv {
            // mxint_systolic_mcu explicitly documents GEMM-only operation.
            // Keep software ISA execution usable with a conservative bound,
            // but make the lack of cycle-exact RTL support visible in trace.
            let cycles = mlen
                .saturating_add(c.gemm_compute_blen_coefficient * blen)
                .saturating_add(c.gemm_compute_fixed_cycles);
            MatrixTimingBreakdown {
                input_load_cycles: mlen,
                systolic_pipeline_cycles: c.gemm_compute_blen_coefficient * blen,
                reduction_drain_cycles: c.gemm_compute_fixed_cycles,
                result_ready_cycles: cycles,
                initiation_interval_cycles: cycles,
                calibration_status: Some(if c.gemv_implemented {
                    CalibrationStatus::StructuralExtrapolation
                } else {
                    CalibrationStatus::UnsupportedRtl
                }),
                rtl_supported: c.gemv_implemented,
                calibration_in_domain: false,
                ..Default::default()
            }
        } else {
            let cycles = c.gemm_compute_blen_coefficient * blen + c.gemm_compute_fixed_cycles;
            MatrixTimingBreakdown {
                // The measured total already contains the BLEN load stream;
                // split it out only for explanatory breakdown.
                input_load_cycles: blen,
                systolic_pipeline_cycles: blen,
                reduction_drain_cycles: c.gemm_compute_fixed_cycles,
                result_ready_cycles: cycles,
                initiation_interval_cycles: cycles,
                calibration_status: Some(if broadcast && !c.broadcast_implemented {
                    CalibrationStatus::UnsupportedRtl
                } else {
                    CalibrationStatus::StructuralExtrapolation
                }),
                rtl_supported: !broadcast || c.broadcast_implemented,
                // MXINT currently has leaf-MCU measurements only.  Do not
                // describe the complete MatrixMachine boundary as validated.
                calibration_in_domain: false,
                ..Default::default()
            }
        }
    } else {
        let c = &CALIBRATION.matrix.mxfp;
        let cycles = if gemv {
            c.gemv_load_mlen_coefficient * mlen + c.gemv_load_fixed_cycles
        } else {
            c.gemm_load_blen_coefficient * blen + c.gemm_load_fixed_cycles
        };
        let rtl_supported = !broadcast || c.broadcast_implemented;
        let profile_measured = measured_mxfp_profile(c);
        let in_domain = rtl_supported && !gemv && profile_measured && mlen <= 64 && blen <= 16;
        let directly_measured = matches!(opcode, Opcode::M_MM { .. })
            && profile_measured
            && measured_matrix_shape(c, mlen, blen);
        MatrixTimingBreakdown {
            input_load_cycles: cycles,
            result_ready_cycles: cycles,
            initiation_interval_cycles: cycles,
            calibration_status: Some(if !rtl_supported {
                CalibrationStatus::UnsupportedRtl
            } else if directly_measured {
                CalibrationStatus::FullMachineMeasured
            } else {
                CalibrationStatus::StructuralExtrapolation
            }),
            rtl_supported,
            calibration_in_domain: in_domain,
            ..Default::default()
        }
    }
}

fn matrix_writeout_timing(opcode: &Opcode, gemv: bool) -> MatrixTimingBreakdown {
    let blen = u64::from(*BLEN);
    let broadcast = is_broadcast_matrix_opcode(opcode);

    if matrix_uses_mxint() {
        let c = &CALIBRATION.matrix.mxint;
        if gemv {
            // See the GEMV note in `matrix_compute_timing`.
            let cycles = c.gemm_writeout_blen_coefficient * blen + c.gemm_writeout_fixed_cycles;
            MatrixTimingBreakdown {
                writeout_cycles: cycles,
                result_ready_cycles: cycles,
                initiation_interval_cycles: cycles,
                calibration_status: Some(if c.gemv_implemented {
                    CalibrationStatus::StructuralExtrapolation
                } else {
                    CalibrationStatus::UnsupportedRtl
                }),
                rtl_supported: c.gemv_implemented,
                calibration_in_domain: false,
                ..Default::default()
            }
        } else {
            let cycles = c.gemm_writeout_blen_coefficient * blen + c.gemm_writeout_fixed_cycles;
            MatrixTimingBreakdown {
                writeout_cycles: cycles,
                result_ready_cycles: cycles,
                initiation_interval_cycles: cycles,
                calibration_status: Some(if broadcast && !c.broadcast_implemented {
                    CalibrationStatus::UnsupportedRtl
                } else {
                    CalibrationStatus::StructuralExtrapolation
                }),
                rtl_supported: !broadcast || c.broadcast_implemented,
                calibration_in_domain: false,
                ..Default::default()
            }
        }
    } else {
        let c = &CALIBRATION.matrix.mxfp;
        if gemv {
            MatrixTimingBreakdown {
                writeout_cycles: c.gemv_writeout_busy_cycles,
                result_ready_cycles: c.gemv_writeout_ready_cycles,
                initiation_interval_cycles: c.gemv_writeout_busy_cycles,
                calibration_status: Some(if broadcast && !c.broadcast_implemented {
                    CalibrationStatus::UnsupportedRtl
                } else {
                    CalibrationStatus::StructuralExtrapolation
                }),
                rtl_supported: !broadcast || c.broadcast_implemented,
                calibration_in_domain: false,
                ..Default::default()
            }
        } else {
            let busy = c.gemm_writeout_busy_blen_coefficient * blen
                + c.gemm_writeout_busy_overhead_coefficient * c.systolic_processing_overhead_cycles
                + c.gemm_writeout_busy_fixed_cycles;
            let ready = c.gemm_writeout_ready_blen_coefficient * blen
                + c.gemm_writeout_ready_overhead_coefficient
                    * c.systolic_processing_overhead_cycles
                + c.gemm_writeout_ready_fixed_cycles;
            MatrixTimingBreakdown {
                systolic_pipeline_cycles: c.gemm_writeout_busy_blen_coefficient * blen,
                reduction_drain_cycles: c.gemm_writeout_busy_overhead_coefficient
                    * c.systolic_processing_overhead_cycles
                    + c.gemm_writeout_busy_fixed_cycles,
                // The current RTL emits one write-valid pulse for a BLEN-row
                // result.  Until row-valid generation is implemented, no
                // consumer-visible row can be proven ready before backend
                // idle. Keep the conservative all-result boundary here.
                result_ready_cycles: if c.gemm_writeout_row_writeback_supported {
                    ready
                } else {
                    busy
                },
                initiation_interval_cycles: busy,
                calibration_status: Some(
                    if !c.gemm_writeout_rtl_supported || (broadcast && !c.broadcast_implemented) {
                        CalibrationStatus::UnsupportedRtl
                    } else {
                        CalibrationStatus::StructuralExtrapolation
                    },
                ),
                rtl_supported: c.gemm_writeout_rtl_supported
                    && (!broadcast || c.broadcast_implemented),
                calibration_in_domain: false,
                ..Default::default()
            }
        }
    }
}

pub(crate) fn matrix_timing(opcode: &Opcode) -> Option<MatrixTimingBreakdown> {
    match opcode {
        Opcode::M_MM { .. }
        | Opcode::M_TMM { .. }
        | Opcode::M_BMM { .. }
        | Opcode::M_BTMM { .. } => Some(matrix_compute_timing(opcode, false)),
        Opcode::M_MV { .. }
        | Opcode::M_TMV { .. }
        | Opcode::M_BMV { .. }
        | Opcode::M_BTMV { .. } => Some(matrix_compute_timing(opcode, true)),
        Opcode::M_MM_WO { .. } | Opcode::M_BMM_WO { .. } => {
            Some(matrix_writeout_timing(opcode, false))
        }
        Opcode::M_MV_WO { .. } | Opcode::M_BMV_WO { .. } => {
            Some(matrix_writeout_timing(opcode, true))
        }
        _ => None,
    }
}

/// Return calibrated non-memory timing. HBM opcodes return `None`; their
/// service interval comes from actual Ramulator acceptance/completion.
pub(crate) fn calibrated_timing(opcode: &Opcode) -> Option<OpcodeTimingEstimate> {
    if let Some(timing) = matrix_timing(opcode) {
        return Some(timing.estimate());
    }

    let vector = &CALIBRATION.vector;
    let scalar = &CALIBRATION.scalar;
    let vlen = u64::from(*VLEN);
    let vector_point_measured = measured_fp_point(
        &vector.measured_points,
        vlen,
        mx_fp_format(*VECTOR_SRAM_TYPE),
    );
    let vector_status = if vector_point_measured {
        CalibrationStatus::FullMachineMeasured
    } else {
        CalibrationStatus::StructuralExtrapolation
    };
    let measured_vector = |cycles: u64| {
        OpcodeTimingEstimate::split(
            cycles,
            cycles,
            vector.initiation_interval_cycles,
            vector_status,
        )
        .with_validation(true, vector_point_measured)
    };
    let scalar_format_measured =
        measured_format(&scalar.measured_points, fp_format(*SCALAR_FP_TYPE));
    let scalar_point_measured =
        measured_fp_point(&scalar.measured_points, vlen, fp_format(*SCALAR_FP_TYPE));
    let scalar_status = if scalar_format_measured {
        CalibrationStatus::FullMachineMeasured
    } else {
        CalibrationStatus::StructuralExtrapolation
    };
    let measured_scalar = |ready: u64, done: u64| {
        OpcodeTimingEstimate::split(done, ready, done, scalar_status)
            .with_validation(true, scalar_format_measured)
    };

    Some(match opcode {
        Opcode::M_MM { .. }
        | Opcode::M_TMM { .. }
        | Opcode::M_BMM { .. }
        | Opcode::M_BTMM { .. }
        | Opcode::M_BMM_WO { .. }
        | Opcode::M_MM_WO { .. }
        | Opcode::M_MV { .. }
        | Opcode::M_TMV { .. }
        | Opcode::M_BMV { .. }
        | Opcode::M_BTMV { .. }
        | Opcode::M_MV_WO { .. }
        | Opcode::M_BMV_WO { .. } => unreachable!("matrix op handled above"),
        Opcode::H_PREFETCH_M { .. } | Opcode::H_PREFETCH_V { .. } | Opcode::H_STORE_V { .. } => {
            return None;
        }
        Opcode::V_ADD_VV { .. } => measured_vector(vector.add_vv_cycles),
        Opcode::V_ADD_VF { .. } => measured_vector(vector.add_vf_cycles),
        Opcode::V_SUB_VV { .. } => measured_vector(vector.sub_vv_cycles),
        Opcode::V_SUB_VF { .. } => measured_vector(vector.sub_vf_cycles),
        Opcode::V_MUL_VV { .. } => measured_vector(vector.mul_vv_cycles),
        Opcode::V_MUL_VF { .. } => measured_vector(vector.mul_vf_cycles),
        Opcode::V_EXP_V { .. } => measured_vector(vector.exp_cycles),
        Opcode::V_RECI_V { .. } => measured_vector(vector.reciprocal_cycles),
        Opcode::V_RED_SUM { .. } => OpcodeTimingEstimate::fixed(
            vector.reduce_sum_base_cycles
                + vector.reduce_sum_per_level_cycles * ceil_log2(vlen + 1),
            vector_status,
        )
        .with_validation(true, vector_point_measured),
        Opcode::V_RED_MAX { .. } => OpcodeTimingEstimate::fixed(
            vector.reduce_max_base_cycles
                + vector.reduce_max_per_level_cycles * ceil_log2(vlen + 1),
            vector_status,
        )
        .with_validation(true, vector_point_measured),
        Opcode::V_SHIFT_V { .. } => OpcodeTimingEstimate::fixed(
            vector.shift_conservative_cycles,
            if vector.shift_implemented {
                CalibrationStatus::StructuralExtrapolation
            } else {
                CalibrationStatus::UnsupportedRtl
            },
        )
        .with_validation(vector.shift_implemented, false),
        Opcode::S_MAP_V_FP { .. } => OpcodeTimingEstimate::split(
            vlen + scalar.map_vector_done_fixed_cycles,
            vlen + scalar.map_vector_ready_fixed_cycles,
            vlen + scalar.map_vector_done_fixed_cycles,
            if scalar_point_measured {
                CalibrationStatus::FullMachineMeasured
            } else {
                CalibrationStatus::StructuralExtrapolation
            },
        )
        .with_validation(true, scalar_point_measured),
        Opcode::S_ADD_FP { .. } => {
            measured_scalar(scalar.fp_add_ready_cycles, scalar.fp_add_done_cycles)
        }
        Opcode::S_SUB_FP { .. } => {
            measured_scalar(scalar.fp_sub_ready_cycles, scalar.fp_sub_done_cycles)
        }
        Opcode::S_MUL_FP { .. } => {
            measured_scalar(scalar.fp_mul_ready_cycles, scalar.fp_mul_done_cycles)
        }
        Opcode::S_MAX_FP { .. } => OpcodeTimingEstimate::fixed(
            scalar.fp_max_conservative_cycles,
            if scalar.fp_max_implemented {
                CalibrationStatus::StructuralExtrapolation
            } else {
                CalibrationStatus::UnsupportedRtl
            },
        )
        .with_validation(scalar.fp_max_implemented, false),
        Opcode::S_EXP_FP { .. } => {
            measured_scalar(scalar.fp_exp_ready_cycles, scalar.fp_exp_done_cycles)
        }
        Opcode::S_RECI_FP { .. } => measured_scalar(
            scalar.fp_reciprocal_ready_cycles,
            scalar.fp_reciprocal_done_cycles,
        ),
        Opcode::S_SQRT_FP { .. } => {
            measured_scalar(scalar.fp_sqrt_ready_cycles, scalar.fp_sqrt_done_cycles)
        }
        Opcode::S_LD_FP { .. } | Opcode::S_ST_FP { .. } => OpcodeTimingEstimate::fixed(
            scalar.sram_cycles,
            CalibrationStatus::StructuralExtrapolation,
        )
        .with_validation(true, false),
        Opcode::S_ADD_INT { .. }
        | Opcode::S_ADDI_INT { .. }
        | Opcode::S_SUB_INT { .. }
        | Opcode::S_MUL_INT { .. }
        | Opcode::S_LUI_INT { .. }
        | Opcode::S_LD_INT { .. }
        | Opcode::S_ST_INT { .. } => OpcodeTimingEstimate::fixed(
            scalar.int_basic_cycles,
            CalibrationStatus::StructuralExtrapolation,
        )
        .with_validation(true, false),
        Opcode::C_SET_ADDR_REG { .. }
        | Opcode::C_SET_SCALE_REG { .. }
        | Opcode::C_SET_STRIDE_REG { .. }
        | Opcode::C_SET_V_MASK_REG { .. }
        | Opcode::C_LOOP_START { .. }
        | Opcode::C_LOOP_END { .. }
        | Opcode::C_BREAK
        | Opcode::Invalid => {
            OpcodeTimingEstimate::fixed(1, CalibrationStatus::StructuralExtrapolation)
                .with_validation(!matches!(opcode, Opcode::Invalid), true)
        }
    })
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct TimingCalibrationMetadata<'a> {
    pub(crate) schema_version: u32,
    pub(crate) model: &'a str,
    pub(crate) measurement_boundary: &'a str,
    pub(crate) clock_period_assumption_ps: u32,
    pub(crate) rtl_repository: &'a str,
    pub(crate) rtl_head: &'a str,
    pub(crate) rtl_dirty: bool,
    pub(crate) rtl_diff_sha256: &'a str,
    pub(crate) implementation_profile: &'a str,
    pub(crate) dc_library_profile_verified: bool,
    pub(crate) note: &'a str,
}

pub(crate) fn calibration_metadata() -> TimingCalibrationMetadata<'static> {
    let c = &*CALIBRATION;
    TimingCalibrationMetadata {
        schema_version: c.schema_version,
        model: c.model.as_str(),
        measurement_boundary: c.measurement_boundary.as_str(),
        clock_period_assumption_ps: c.clock_period_assumption_ps,
        rtl_repository: c.source.rtl_repository.as_str(),
        rtl_head: c.source.rtl_head.as_str(),
        rtl_dirty: c.source.rtl_dirty,
        rtl_diff_sha256: c.source.rtl_diff_sha256.as_str(),
        implementation_profile: c.source.implementation_profile.as_str(),
        dc_library_profile_verified: c.source.dc_library_profile_verified,
        note: c.source.note.as_str(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_and_gemv_use_different_load_dimensions() {
        let gemm = matrix_timing(&Opcode::M_MM { rs1: 1, rs2: 2 }).unwrap();
        let gemv = matrix_timing(&Opcode::M_MV { rs1: 1, rs2: 2 }).unwrap();
        if !matrix_uses_mxint() {
            assert_eq!(
                gemm.input_load_cycles,
                u64::from(*BLEN) + CALIBRATION.matrix.mxfp.gemm_load_fixed_cycles
            );
            assert_eq!(
                gemv.input_load_cycles,
                u64::from(*MLEN) + CALIBRATION.matrix.mxfp.gemv_load_fixed_cycles
            );
        }
        assert!(gemm.total_cycles() > 0);
        assert!(gemv.total_cycles() > 0);
    }

    #[test]
    fn writeout_is_not_a_fixed_one_cycle() {
        let wo = matrix_timing(&Opcode::M_MM_WO {
            rd: 1,
            rstride: 0,
            imm: 0,
        })
        .unwrap();
        assert!(wo.total_cycles() > 1);
        assert!(wo.result_ready_cycles > 1);
    }

    #[test]
    fn mxfp_writeout_matches_measured_drain_formula() {
        if matrix_uses_mxint() {
            return;
        }
        let c = &CALIBRATION.matrix.mxfp;
        let expected = c.gemm_writeout_busy_blen_coefficient * u64::from(*BLEN)
            + c.gemm_writeout_busy_overhead_coefficient * c.systolic_processing_overhead_cycles
            + c.gemm_writeout_busy_fixed_cycles;
        let timing = matrix_timing(&Opcode::M_MM_WO {
            rd: 1,
            rstride: 0,
            imm: 0,
        })
        .unwrap();
        assert_eq!(timing.total_cycles(), expected);
        assert_eq!(timing.result_ready_cycles, expected);
    }

    #[test]
    fn reduction_latency_tracks_logarithmic_tree_depth() {
        let levels = ceil_log2(u64::from(*VLEN) + 1);
        let timing = calibrated_timing(&Opcode::V_RED_SUM {
            rd: 1,
            rs1: 2,
            rmask: 0,
        })
        .unwrap();
        assert_eq!(
            timing.resource_cycles,
            CALIBRATION.vector.reduce_sum_base_cycles
                + CALIBRATION.vector.reduce_sum_per_level_cycles * levels
        );
    }

    #[test]
    fn calibration_metadata_records_unverified_dc_profile() {
        let metadata = calibration_metadata();
        assert_eq!(metadata.schema_version, 3);
        assert_eq!(
            metadata.measurement_boundary,
            "full_machine_execute_accept_to_consumer_ready_and_backend_idle"
        );
        assert!(!metadata.dc_library_profile_verified);
    }

    #[test]
    fn scalar_full_machine_timing_distinguishes_ready_from_idle() {
        let timing = calibrated_timing(&Opcode::S_ADD_FP {
            rd: 1,
            rs1: 2,
            rs2: 3,
        })
        .unwrap();
        assert_eq!(
            timing.result_ready_cycles,
            CALIBRATION.scalar.fp_add_ready_cycles
        );
        assert_eq!(
            timing.resource_cycles,
            CALIBRATION.scalar.fp_add_done_cycles
        );
        assert!(timing.result_ready_cycles < timing.resource_cycles);
    }

    #[test]
    fn unsupported_rtl_opcodes_are_not_reported_as_validated() {
        let max = calibrated_timing(&Opcode::S_MAX_FP {
            rd: 1,
            rs1: 2,
            rs2: 3,
        })
        .unwrap();
        let shift = calibrated_timing(&Opcode::V_SHIFT_V {
            rd: 1,
            rs1: 2,
            rs2: 3,
        })
        .unwrap();
        assert!(!max.rtl_supported);
        assert!(!shift.rtl_supported);
        assert_eq!(max.calibration_status, CalibrationStatus::UnsupportedRtl);
        assert_eq!(shift.calibration_status, CalibrationStatus::UnsupportedRtl);
    }
}
