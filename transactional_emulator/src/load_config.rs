// load_config.rs
use serde::{Deserialize, Serialize};
use std::{env, fs, sync::LazyLock};

// Import the types from your main module
use quantize::{DataType, FpType, IntType, MxDataType};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConfigValue {
    pub value: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConfigValueUsize {
    pub value: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConfigValueString {
    pub value: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LatencyValue {
    pub dc_lib_en: u32,
    pub dc_lib_dis: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FpTypeConfig {
    pub sign: bool,
    pub exponent: u8,
    pub mantissa: u8,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IntTypeConfig {
    pub width: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum DataTypeConfig {
    Fp(FpTypeConfig),
    Int(IntTypeConfig),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MxDataTypeConfig {
    pub format: String,
    #[serde(flatten)]
    pub data: MxDataTypeData,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum MxDataTypeData {
    Plain {
        #[serde(rename = "DATA_TYPE")]
        data_type: DataTypeConfig,
    },
    Mx {
        block: u32,
        #[serde(rename = "ELEM")]
        elem: DataTypeConfig,
        #[serde(rename = "SCALE")]
        scale: DataTypeConfig,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AcceleratorConfig {
    #[serde(rename = "CONFIG")]
    pub config: ConfigSection,
    #[serde(rename = "PRECISION")]
    pub precision: PrecisionSection,
    #[serde(rename = "LATENCY")]
    pub latency: LatencySection,
}

/// Wrapper struct for parsing the new TOML structure with TRANSACTIONAL section
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PlenaSettings {
    #[serde(rename = "TRANSACTIONAL")]
    pub transactional: AcceleratorConfig,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConfigSection {
    #[serde(rename = "BLEN")]
    pub blen: ConfigValue,
    #[serde(rename = "HLEN")]
    pub hlen: ConfigValue,
    #[serde(rename = "MLEN")]
    pub mlen: ConfigValue,
    #[serde(rename = "VLEN")]
    pub vlen: ConfigValue,
    #[serde(rename = "BROADCAST_AMOUNT")]
    pub broadcast_amount: ConfigValue,
    #[serde(rename = "HBM_SIZE")]
    pub hbm_size: ConfigValueUsize,
    #[serde(rename = "MATRIX_SRAM_SIZE")]
    pub matrix_sram_size: ConfigValueUsize,
    #[serde(rename = "VECTOR_SRAM_SIZE")]
    pub vector_sram_size: ConfigValueUsize,
    #[serde(rename = "FP_SRAM_DEPTH", default = "default_fp_sram_depth")]
    pub fp_sram_depth: ConfigValueUsize,
    #[serde(rename = "DRAIN_OVERLAPPED", default = "default_drain_overlapped")]
    pub drain_overlapped: ConfigValueUsize,
    #[serde(rename = "HBM_M_Prefetch_Amount")]
    pub hbm_m_prefetch_amount: ConfigValue,
    #[serde(rename = "HBM_V_Prefetch_Amount")]
    pub hbm_v_prefetch_amount: ConfigValue,
    #[serde(rename = "HBM_V_Writeback_Amount")]
    pub hbm_v_writeback_amount: ConfigValue,
    #[serde(rename = "DC_EN")]
    pub dc_en: ConfigValue,
    #[serde(rename = "MAX_LOOP_INSTRUCTIONS")]
    pub max_loop_instructions: ConfigValueUsize,
    /// HBM generation for the Ramulator timing model ("HBM2" or "HBM3").
    /// Optional so older TOMLs keep working; defaults to HBM2.
    #[serde(rename = "HBM_GEN", default)]
    pub hbm_gen: Option<ConfigValueString>,
    /// Number of HBM channels in the Ramulator model. Optional; defaults to 8.
    #[serde(rename = "HBM_CHANNELS", default)]
    pub hbm_channels: Option<ConfigValueUsize>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PrecisionSection {
    #[serde(rename = "MATRIX_SRAM_TYPE")]
    pub matrix_sram_type: MxDataTypeConfig,
    #[serde(rename = "VECTOR_SRAM_TYPE")]
    pub vector_sram_type: MxDataTypeConfig,
    #[serde(rename = "HBM_M_WEIGHT_TYPE")]
    pub hbm_m_weight_type: MxDataTypeConfig,
    #[serde(rename = "HBM_M_KV_TYPE")]
    pub hbm_m_kv_type: MxDataTypeConfig,
    #[serde(rename = "HBM_V_ACT_TYPE")]
    pub hbm_v_act_type: MxDataTypeConfig,
    #[serde(rename = "HBM_V_KV_TYPE")]
    pub hbm_v_kv_type: MxDataTypeConfig,
    #[serde(rename = "HBM_V_INT_TYPE")]
    pub hbm_v_int_type: MxDataTypeConfig,
    #[serde(rename = "SCALAR_FP")]
    pub scalar_fp: DataTypeConfig,
    #[serde(rename = "MATRIX_SEMANTICS")]
    pub matrix_semantics: MatrixSemanticsDescriptor,
    #[serde(rename = "MX_PHYSICAL_SEMANTICS")]
    pub mx_physical_semantics: MxPhysicalSemanticsDescriptor,
}

impl PrecisionSection {
    fn validate(&self) -> Result<(), String> {
        let block_size = self.matrix_semantics.profile_contract.block_size;
        for (name, data_type) in [
            ("MATRIX_SRAM_TYPE", &self.matrix_sram_type),
            ("VECTOR_SRAM_TYPE", &self.vector_sram_type),
            ("HBM_M_WEIGHT_TYPE", &self.hbm_m_weight_type),
            ("HBM_M_KV_TYPE", &self.hbm_m_kv_type),
            ("HBM_V_ACT_TYPE", &self.hbm_v_act_type),
            ("HBM_V_KV_TYPE", &self.hbm_v_kv_type),
            ("HBM_V_INT_TYPE", &self.hbm_v_int_type),
        ] {
            validate_mx_storage_type(name, data_type, block_size)?;
        }
        self.mx_physical_semantics.validate()?;
        Ok(())
    }
}

fn validate_mx_storage_type(
    name: &str,
    data_type: &MxDataTypeConfig,
    block_size: u32,
) -> Result<(), String> {
    let MxDataTypeData::Mx { block, elem, scale } = &data_type.data else {
        return Ok(());
    };
    if *block != block_size {
        return Err(format!(
            "{name} block size {block} differs from native block size {block_size}"
        ));
    }
    if let DataTypeConfig::Int(int_type) = elem {
        if !matches!(int_type.width, 2 | 4 | 8) {
            return Err(format!(
                "{name} MXINT element width must be 2, 4, or 8 bits"
            ));
        }
    }
    if !matches!(
        scale,
        DataTypeConfig::Fp(FpTypeConfig {
            sign: false,
            exponent: 8,
            mantissa: 0,
        })
    ) {
        return Err(format!("{name} requires an E8M0 scale"));
    }
    Ok(())
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MxPhysicalSemanticsDescriptor {
    pub schema_version: String,
    pub block_size: u32,
    pub scale_format: String,
    pub scale_code_bias: u32,
    pub scale_code_min: u32,
    pub scale_code_max: u32,
    pub scale_exponent_min: i32,
    pub scale_exponent_max: i32,
    pub zero_scale_code: u32,
    pub element_bit_order: String,
    pub plane_order: Vec<String>,
    pub plane_alignment_bytes: u32,
    pub mxint_encoding: String,
    pub mxint_canonical_zero: String,
    pub mxint_rounding: String,
    pub mxint_scale_rule: String,
    pub mxfp_scale_rule: String,
}

impl MxPhysicalSemanticsDescriptor {
    fn expected() -> Self {
        Self {
            schema_version: "plena-mx-physical-semantics".to_string(),
            block_size: 8,
            scale_format: "E8M0".to_string(),
            scale_code_bias: 127,
            scale_code_min: 0,
            scale_code_max: 255,
            scale_exponent_min: -127,
            scale_exponent_max: 128,
            zero_scale_code: 127,
            element_bit_order: "little_endian_lsb_first".to_string(),
            plane_order: vec!["element".to_string(), "scale".to_string()],
            plane_alignment_bytes: 32,
            mxint_encoding: "sign_magnitude".to_string(),
            mxint_canonical_zero: "positive_zero".to_string(),
            mxint_rounding: "round_to_nearest_ties_to_even".to_string(),
            mxint_scale_rule: "ceil_log2_max_abs_over_qmax_fraction".to_string(),
            mxfp_scale_rule: "floor_log2_max_abs".to_string(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self != &Self::expected() {
            return Err("physical MX semantics differ from the PLENA contract".to_string());
        }
        Ok(())
    }
}

impl Default for MxPhysicalSemanticsDescriptor {
    fn default() -> Self {
        Self::expected()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MatrixSemanticsProfileContract {
    pub schema_version: String,
    pub block_size: u32,
    pub mxint_rule: String,
    pub mxint_max_shift: u32,
    pub mxint_vector_rounding: String,
    pub mxint_partial_conversion: String,
    pub mxint_cross_instruction_accumulation: String,
    pub mxfp_rule: String,
    pub m_fp_format_binding: String,
    pub matrix_storage_fp_binding: String,
    pub matrix_instruction_k_partition: String,
    pub qk_logical_k_partition: String,
    pub fixed_accumulator_integer_bits: u8,
    pub fixed_accumulator_fraction_bits: u8,
    pub accumulator_rule: String,
    pub output_rule: String,
    pub mixed_family_rule: String,
    pub mixed_family_deployment_supported: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MatrixOperationBinding {
    pub operation: String,
    pub left_role: String,
    pub right_role: String,
    pub family: String,
    pub rule: String,
    pub structurally_supported: bool,
    pub numerical_trace_conformance: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FixedAccumulatorBank {
    pub integer_bits: u8,
    pub fraction_bits: u8,
    pub accumulator_rule: String,
    pub writeout_rule: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InstructionReduction {
    pub partial_conversion: String,
    pub cross_instruction_accumulation: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MatrixStorageFp {
    pub format: String,
    pub exponent_bits: u8,
    pub mantissa_bits: u8,
    pub rtl_parameter_source: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MxIntPipeline {
    pub block_size: u32,
    pub block_mac: String,
    pub alignment: String,
    pub max_shift: u32,
    pub matrix_conversion: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MxFpPipeline {
    pub product_conversion: String,
    pub m_fp_format_binding: String,
    pub bank_conversion: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MixedFamilyBinding {
    pub rule: String,
    pub deployment_supported: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PackedKvSelectorCapability {
    pub supported: bool,
    pub reason: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NumericalTraceConformance {
    pub status: String,
    pub required_for_emulator_valid: bool,
    pub required_for_rtl_valid: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MatrixSemanticsDescriptor {
    pub schema_version: String,
    pub source_profile_schema: String,
    pub profile_contract: MatrixSemanticsProfileContract,
    pub active_family: String,
    pub active_rule: String,
    pub operation_bindings: Vec<MatrixOperationBinding>,
    pub fixed_accumulator_bank: FixedAccumulatorBank,
    pub instruction_reduction: InstructionReduction,
    pub matrix_storage_fp: MatrixStorageFp,
    pub mxint_pipeline: MxIntPipeline,
    pub mxfp_pipeline: MxFpPipeline,
    pub mixed_family: MixedFamilyBinding,
    pub packedkv_selector_rtl_capability: PackedKvSelectorCapability,
    pub structural_binding_valid: bool,
    pub numerical_trace_conformance: NumericalTraceConformance,
}

impl MatrixSemanticsDescriptor {
    fn for_binding(
        family: &str,
        vector_format: &str,
        vector_exponent_bits: u8,
        vector_mantissa_bits: u8,
    ) -> Result<Self, String> {
        let canonical_vector_format = if vector_exponent_bits == 8
            && vector_mantissa_bits == 7
        {
            "BF16".to_string()
        } else {
            format!("FP_E{vector_exponent_bits}M{vector_mantissa_bits}")
        };
        if vector_format != canonical_vector_format {
            return Err("matrix storage format differs from its bit fields".to_string());
        }
        if !matches!(
            vector_format,
            "BF16"
                | "FP_E3M2"
                | "FP_E2M3"
                | "FP_E6M5"
                | "FP_E5M6"
                | "FP_E4M7"
                | "FP_E8M5"
        ) {
            return Err(format!("unsupported matrix storage format {vector_format:?}"));
        }
        let active_rule = match family {
            "mxint" => {
                "block8_range_safe_scale_widened_mac_max_shift16_rne_vector"
            }
            "mxfp" => "product_cast_to_m_fp_then_fixed16_16_bank",
            _ => return Err(format!("unsupported matrix family {family:?}")),
        };
        let operation_bindings = [
            ("linear", "activation", "weight"),
            ("qk", "activation", "key"),
            ("pv", "activation", "value"),
        ]
        .into_iter()
        .map(|(operation, left_role, right_role)| MatrixOperationBinding {
            operation: operation.to_string(),
            left_role: left_role.to_string(),
            right_role: right_role.to_string(),
            family: family.to_string(),
            rule: active_rule.to_string(),
            structurally_supported: true,
            numerical_trace_conformance: "not_run".to_string(),
        })
        .collect();
        let descriptor = Self {
            schema_version: "plena-matrix-semantics".to_string(),
            source_profile_schema: "decode-precision-profile".to_string(),
            profile_contract: MatrixSemanticsProfileContract {
                schema_version: "plena-matrix-semantics".to_string(),
                block_size: 8,
                mxint_rule:
                    "block8_range_safe_scale_widened_mac_max_shift16_rne_vector"
                        .to_string(),
                mxint_max_shift: 16,
                mxint_vector_rounding: "round_to_nearest_even".to_string(),
                mxint_partial_conversion:
                    "per_mm_ic_integer_reduction_to_vector_storage_fp".to_string(),
                mxint_cross_instruction_accumulation:
                    "signed_fixed16_16_wraparound".to_string(),
                mxfp_rule: "product_cast_to_m_fp_then_fixed16_16_bank".to_string(),
                m_fp_format_binding: "profile.vector_format".to_string(),
                matrix_storage_fp_binding: "profile.vector_format".to_string(),
                matrix_instruction_k_partition: "MLEN".to_string(),
                qk_logical_k_partition: "HLEN".to_string(),
                fixed_accumulator_integer_bits: 16,
                fixed_accumulator_fraction_bits: 16,
                accumulator_rule: "plena_fixed16_16_accumulate_truncate".to_string(),
                output_rule: "truncate_to_vector_format".to_string(),
                mixed_family_rule: "deployment_unsupported_without_trace_evidence".to_string(),
                mixed_family_deployment_supported: false,
            },
            active_family: family.to_string(),
            active_rule: active_rule.to_string(),
            operation_bindings,
            fixed_accumulator_bank: FixedAccumulatorBank {
                integer_bits: 16,
                fraction_bits: 16,
                accumulator_rule: "plena_fixed16_16_accumulate_truncate".to_string(),
                writeout_rule: "truncate_to_vector_format".to_string(),
            },
            instruction_reduction: InstructionReduction {
                partial_conversion: "per_mm_ic_to_vector_storage_fp".to_string(),
                cross_instruction_accumulation:
                    "signed_fixed16_16_wraparound".to_string(),
            },
            matrix_storage_fp: MatrixStorageFp {
                format: vector_format.to_string(),
                exponent_bits: vector_exponent_bits,
                mantissa_bits: vector_mantissa_bits,
                rtl_parameter_source: "V_FP".to_string(),
            },
            mxint_pipeline: MxIntPipeline {
                block_size: 8,
                block_mac: "exact_signed_widened_integer".to_string(),
                alignment: "bounded_exponent_alignment".to_string(),
                max_shift: 16,
                matrix_conversion: "round_to_nearest_even_to_vector".to_string(),
            },
            mxfp_pipeline: MxFpPipeline {
                product_conversion: "cast_each_product_to_m_fp".to_string(),
                m_fp_format_binding: "profile.vector_format".to_string(),
                bank_conversion: "m_fp_to_fixed16_16".to_string(),
            },
            mixed_family: MixedFamilyBinding {
                rule: "deployment_unsupported_without_trace_evidence".to_string(),
                deployment_supported: false,
            },
            packedkv_selector_rtl_capability: PackedKvSelectorCapability {
                supported: family == "mxint",
                reason: if family == "mxint" {
                    "supported".to_string()
                } else {
                    "selector_is_mxint_only".to_string()
                },
            },
            structural_binding_valid: true,
            numerical_trace_conformance: NumericalTraceConformance {
                status: "not_run".to_string(),
                required_for_emulator_valid: true,
                required_for_rtl_valid: true,
            },
        };
        Ok(descriptor)
    }

    pub fn validate(&self) -> Result<(), String> {
        let expected = Self::for_binding(
            &self.active_family,
            &self.matrix_storage_fp.format,
            self.matrix_storage_fp.exponent_bits,
            self.matrix_storage_fp.mantissa_bits,
        )?;
        if self != &expected {
            return Err("matrix semantics differ from the PLENA contract".to_string());
        }
        Ok(())
    }

    pub fn validate_binding(
        &self,
        config: &ConfigSection,
        vector_sram_type: &MxDataTypeConfig,
    ) -> Result<(), String> {
        self.validate()?;
        if config.mlen.value == 0
            || config.hlen.value == 0
            || config.hlen.value > config.mlen.value
        {
            return Err("matrix reduction geometry is invalid".to_string());
        }
        let MxDataTypeData::Plain {
            data_type:
                DataTypeConfig::Fp(FpTypeConfig {
                    exponent,
                    mantissa,
                    ..
                }),
        } = &vector_sram_type.data
        else {
            return Err("VECTOR_SRAM_TYPE must be a plain FP format".to_string());
        };
        if self.matrix_storage_fp.exponent_bits != *exponent
            || self.matrix_storage_fp.mantissa_bits != *mantissa
        {
            return Err(
                "matrix storage contract differs from VECTOR_SRAM_TYPE".to_string(),
            );
        }
        Ok(())
    }
}

impl Default for MatrixSemanticsDescriptor {
    fn default() -> Self {
        Self::for_binding("mxfp", "FP_E6M5", 6, 5)
            .expect("the built-in matrix contract is valid")
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LatencySection {
    #[serde(rename = "SYSTOLIC_PROCESSING_OVERHEAD")]
    pub systolic_processing_overhead: LatencyValue,
    #[serde(rename = "VECTOR_ADD_CYCLES")]
    pub vector_add_cycles: LatencyValue,
    #[serde(rename = "VECTOR_MUL_CYCLES")]
    pub vector_mul_cycles: LatencyValue,
    #[serde(rename = "VECTOR_EXP_CYCLES")]
    pub vector_exp_cycles: LatencyValue,
    #[serde(rename = "VECTOR_PREFIX_SCAN_CYCLES")]
    pub vector_prefix_scan_cycles: LatencyValue,
    #[serde(rename = "VECTOR_SHIFT_CYCLES")]
    pub vector_shift_cycles: LatencyValue,
    #[serde(rename = "VECTOR_RECI_CYCLES")]
    pub vector_reci_cycles: LatencyValue,
    #[serde(rename = "VECTOR_MAX_CYCLES")]
    pub vector_max_cycles: LatencyValue,
    #[serde(rename = "VECTOR_SUM_CYCLES")]
    pub vector_sum_cycles: LatencyValue,
    #[serde(rename = "SCALAR_FP_LONGEST_OPERATE_CYCLES")]
    pub scalar_fp_longest_operate_cycles: LatencyValue,
    #[serde(rename = "SCALAR_FP_BASIC_CYCLES")]
    pub scalar_fp_basic_cycles: LatencyValue,
    #[serde(rename = "SCALAR_FP_EXP_CYCLES")]
    pub scalar_fp_exp_cycles: LatencyValue,
    #[serde(rename = "SCALAR_FP_SQRT_CYCLES")]
    pub scalar_fp_sqrt_cycles: LatencyValue,
    #[serde(rename = "SCALAR_FP_RECI_CYCLES")]
    pub scalar_fp_reci_cycles: LatencyValue,
    #[serde(rename = "SCALAR_INT_BASIC_CYCLES")]
    pub scalar_int_basic_cycles: LatencyValue,
}

impl Default for AcceleratorConfig {
    fn default() -> Self {
        AcceleratorConfig {
            config: ConfigSection {
                blen: ConfigValue { value: 32 },
                hlen: ConfigValue { value: 16 },
                mlen: ConfigValue { value: 32 },
                vlen: ConfigValue { value: 32 },
                broadcast_amount: ConfigValue { value: 2 },
                hbm_size: ConfigValueUsize { value: 1073741824 },
                matrix_sram_size: ConfigValueUsize { value: 1024 },
                vector_sram_size: ConfigValueUsize { value: 1024 },
                fp_sram_depth: default_fp_sram_depth(),
                drain_overlapped: default_drain_overlapped(),
                hbm_m_prefetch_amount: ConfigValue { value: 16 },
                hbm_v_prefetch_amount: ConfigValue { value: 16 },
                hbm_v_writeback_amount: ConfigValue { value: 16 },
                dc_en: ConfigValue { value: 1 },
                max_loop_instructions: ConfigValueUsize { value: 10000 },
                hbm_gen: None,
                hbm_channels: None,
            },
            precision: PrecisionSection {
                matrix_sram_type: MxDataTypeConfig {
                    format: "Plain".to_string(),
                    data: MxDataTypeData::Plain {
                        data_type: DataTypeConfig::Fp(FpTypeConfig {
                            sign: true,
                            exponent: 8,
                            mantissa: 7,
                        }),
                    },
                },
                vector_sram_type: MxDataTypeConfig {
                    format: "Plain".to_string(),
                    data: MxDataTypeData::Plain {
                        data_type: DataTypeConfig::Fp(FpTypeConfig {
                            sign: true,
                            exponent: 8,
                            mantissa: 7,
                        }),
                    },
                },
                hbm_m_weight_type: MxDataTypeConfig {
                    format: "Mx".to_string(),
                    data: MxDataTypeData::Mx {
                        block: 8,
                        elem: DataTypeConfig::Fp(FpTypeConfig {
                            sign: true,
                            exponent: 4,
                            mantissa: 3,
                        }),
                        scale: DataTypeConfig::Fp(FpTypeConfig {
                            sign: false,
                            exponent: 8,
                            mantissa: 0,
                        }),
                    },
                },
                hbm_m_kv_type: MxDataTypeConfig {
                    format: "Mx".to_string(),
                    data: MxDataTypeData::Mx {
                        block: 8,
                        elem: DataTypeConfig::Fp(FpTypeConfig {
                            sign: true,
                            exponent: 4,
                            mantissa: 3,
                        }),
                        scale: DataTypeConfig::Fp(FpTypeConfig {
                            sign: false,
                            exponent: 8,
                            mantissa: 0,
                        }),
                    },
                },
                hbm_v_act_type: MxDataTypeConfig {
                    format: "Mx".to_string(),
                    data: MxDataTypeData::Mx {
                        block: 8,
                        elem: DataTypeConfig::Fp(FpTypeConfig {
                            sign: true,
                            exponent: 4,
                            mantissa: 3,
                        }),
                        scale: DataTypeConfig::Fp(FpTypeConfig {
                            sign: false,
                            exponent: 8,
                            mantissa: 0,
                        }),
                    },
                },
                hbm_v_kv_type: MxDataTypeConfig {
                    format: "Mx".to_string(),
                    data: MxDataTypeData::Mx {
                        block: 8,
                        elem: DataTypeConfig::Fp(FpTypeConfig {
                            sign: true,
                            exponent: 4,
                            mantissa: 3,
                        }),
                        scale: DataTypeConfig::Fp(FpTypeConfig {
                            sign: false,
                            exponent: 8,
                            mantissa: 0,
                        }),
                    },
                },
                hbm_v_int_type: MxDataTypeConfig {
                    format: "Plain".to_string(),
                    data: MxDataTypeData::Plain {
                        data_type: DataTypeConfig::Int(IntTypeConfig { width: 32 }),
                    },
                },
                scalar_fp: DataTypeConfig::Fp(FpTypeConfig {
                    sign: true,
                    exponent: 8,
                    mantissa: 7,
                }),
                matrix_semantics: MatrixSemanticsDescriptor::default(),
                mx_physical_semantics: MxPhysicalSemanticsDescriptor::default(),
            },
            latency: LatencySection {
                systolic_processing_overhead: LatencyValue {
                    dc_lib_en: 0,
                    dc_lib_dis: 0,
                },
                vector_add_cycles: LatencyValue {
                    dc_lib_en: 2,
                    dc_lib_dis: 7,
                },
                vector_mul_cycles: LatencyValue {
                    dc_lib_en: 1,
                    dc_lib_dis: 5,
                },
                vector_exp_cycles: LatencyValue {
                    dc_lib_en: 1,
                    dc_lib_dis: 6,
                },
                vector_prefix_scan_cycles: LatencyValue {
                    dc_lib_en: 9,
                    dc_lib_dis: 9,
                },
                vector_shift_cycles: LatencyValue {
                    dc_lib_en: 1,
                    dc_lib_dis: 1,
                },
                vector_reci_cycles: LatencyValue {
                    dc_lib_en: 2,
                    dc_lib_dis: 7,
                },
                vector_max_cycles: LatencyValue {
                    dc_lib_en: 4,
                    dc_lib_dis: 4,
                },
                vector_sum_cycles: LatencyValue {
                    dc_lib_en: 8,
                    dc_lib_dis: 20,
                },
                scalar_fp_longest_operate_cycles: LatencyValue {
                    dc_lib_en: 4,
                    dc_lib_dis: 4,
                },
                scalar_fp_basic_cycles: LatencyValue {
                    dc_lib_en: 1,
                    dc_lib_dis: 1,
                },
                scalar_fp_exp_cycles: LatencyValue {
                    dc_lib_en: 1,
                    dc_lib_dis: 2,
                },
                scalar_fp_sqrt_cycles: LatencyValue {
                    dc_lib_en: 1,
                    dc_lib_dis: 2,
                },
                scalar_fp_reci_cycles: LatencyValue {
                    dc_lib_en: 1,
                    dc_lib_dis: 2,
                },
                scalar_int_basic_cycles: LatencyValue {
                    dc_lib_en: 1,
                    dc_lib_dis: 1,
                },
            },
        }
    }
}

// Conversion functions from config types to your actual types
impl From<FpTypeConfig> for FpType {
    fn from(config: FpTypeConfig) -> Self {
        FpType {
            sign: config.sign,
            exponent: config.exponent,
            mantissa: config.mantissa,
        }
    }
}

impl From<IntTypeConfig> for IntType {
    fn from(config: IntTypeConfig) -> Self {
        IntType {
            width: config.width,
        }
    }
}

impl From<DataTypeConfig> for DataType {
    fn from(config: DataTypeConfig) -> Self {
        match config {
            DataTypeConfig::Fp(fp_config) => DataType::Fp(fp_config.into()),
            DataTypeConfig::Int(int_config) => DataType::Int(int_config.into()),
        }
    }
}

impl From<MxDataTypeConfig> for MxDataType {
    fn from(config: MxDataTypeConfig) -> Self {
        match config.data {
            MxDataTypeData::Plain { data_type } => MxDataType::Plain(data_type.into()),
            MxDataTypeData::Mx { elem, scale, block } => MxDataType::Mx {
                elem: elem.into(),
                scale: scale.into(),
                block,
            },
        }
    }
}

// Global configuration loaded at runtime
pub static CONFIG: LazyLock<AcceleratorConfig> = LazyLock::new(|| {
    load_config().unwrap_or_else(|error| {
        panic!("invalid or unbound PLENA configuration: {error}")
    })
});

// Configuration loading functions
pub fn load_config() -> Result<AcceleratorConfig, Box<dyn std::error::Error>> {
    // 1. Check PLENA_SETTINGS_TOML env var (set by per-build test harness)
    if let Ok(path) = env::var("PLENA_SETTINGS_TOML") {
        return load_config_from_file(&path);
    }

    // 2. Fallback to hardcoded ../plena_settings.toml
    let config_path = env::current_dir()
        .unwrap()
        .parent()
        .unwrap()
        .join("plena_settings.toml");

    let config_path = config_path.to_str().unwrap();
    if let Ok(config) = load_config_from_file(config_path) {
        return Ok(config);
    }

    Err("No configuration file found".into())
}

pub fn load_config_from_file(path: &str) -> Result<AcceleratorConfig, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let settings: PlenaSettings = toml::from_str(&content)?;
    settings
        .transactional
        .precision
        .matrix_semantics
        .validate_binding(
            &settings.transactional.config,
            &settings.transactional.precision.vector_sram_type,
        )
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    settings
        .transactional
        .precision
        .validate()
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    Ok(settings.transactional)
}

// Helper function to check if DC library is enabled from config
pub fn is_dc_lib_enabled() -> bool {
    CONFIG.config.dc_en.value != 0
}

// Helper function to select DC library enabled or disabled values
pub fn get_dc_lib_value(latency_val: &LatencyValue) -> u32 {
    if is_dc_lib_enabled() {
        latency_val.dc_lib_en
    } else {
        latency_val.dc_lib_dis
    }
}

// Configuration accessor functions (automatically uses DC_EN setting from config)

pub fn hbm_size() -> usize {
    CONFIG.config.hbm_size.value
}

pub fn hbm_gen() -> String {
    CONFIG
        .config
        .hbm_gen
        .as_ref()
        .map(|v| v.value.clone())
        .unwrap_or_else(|| "HBM2".to_string())
}

pub fn hbm_channels() -> usize {
    CONFIG.config.hbm_channels.as_ref().map(|v| v.value).unwrap_or(8)
}

pub fn matrix_sram_size() -> usize {
    CONFIG.config.matrix_sram_size.value
}

pub fn vector_sram_size() -> usize {
    CONFIG.config.vector_sram_size.value
}

/// Scalar FP SRAM depth inherited from the RTL configuration.
fn default_fp_sram_depth() -> ConfigValueUsize {
    ConfigValueUsize { value: 512 }
}

/// The implemented matrix control unit accepts a new instruction only when it is
/// not draining, so a writeout is serialized ahead of the next accumulate.
/// Setting `DRAIN_OVERLAPPED = 1` prices the writeout at one cycle, which is the
/// behaviour of a control unit whose accumulator is double-banked. It is off by
/// default because it does not describe the current RTL.
fn default_drain_overlapped() -> ConfigValueUsize {
    ConfigValueUsize { value: 0 }
}

pub fn fp_sram_depth() -> usize {
    CONFIG.config.fp_sram_depth.value
}

pub fn drain_overlapped() -> bool {
    CONFIG.config.drain_overlapped.value != 0
}

pub fn matrix_sram_type() -> MxDataType {
    CONFIG.precision.matrix_sram_type.clone().into()
}

pub fn vector_sram_type() -> MxDataType {
    CONFIG.precision.vector_sram_type.clone().into()
}

pub fn matrix_weight_type() -> MxDataType {
    CONFIG.precision.hbm_m_weight_type.clone().into()
}

pub fn hbm_m_prefetch_amount() -> u32 {
    CONFIG.config.hbm_m_prefetch_amount.value
}

pub fn hbm_v_prefetch_amount() -> u32 {
    CONFIG.config.hbm_v_prefetch_amount.value
}

pub fn hbm_v_writeback_amount() -> u32 {
    CONFIG.config.hbm_v_writeback_amount.value
}

pub fn matrix_kv_type() -> MxDataType {
    CONFIG.precision.hbm_m_kv_type.clone().into()
}

pub fn vector_activation_type() -> MxDataType {
    CONFIG.precision.hbm_v_act_type.clone().into()
}

pub fn vector_kv_type() -> MxDataType {
    CONFIG.precision.hbm_v_kv_type.clone().into()
}

pub fn matrix_semantics() -> MatrixSemanticsDescriptor {
    CONFIG.precision.matrix_semantics.clone()
}

/// Reserved for future scalar FP ops; not yet wired into any opcode dispatch.
#[allow(dead_code)]
pub fn scalar_fp_type() -> DataType {
    CONFIG.precision.scalar_fp.clone().into()
}

// pub fn vector_int_type() -> MxDataType {
//     CONFIG.precision.hbm_v_int_type.clone().into()
// }

// Additional accessor functions for new parameters
pub fn mlen() -> u32 {
    CONFIG.config.mlen.value
}

pub fn hlen() -> u32 {
    CONFIG.config.hlen.value
}

pub fn broadcast_amount() -> u32 {
    CONFIG.config.broadcast_amount.value
}

pub fn vlen() -> u32 {
    CONFIG.config.vlen.value
}

pub fn blen() -> u32 {
    CONFIG.config.blen.value
}

// pub fn dc_en() -> u32 {
//     CONFIG.config.dc_en.value
// }

// Latency accessor functions (automatically uses DC_EN setting from config)
pub fn systolic_processing_overhead() -> u32 {
    get_dc_lib_value(&CONFIG.latency.systolic_processing_overhead)
}

// pub fn vector_ps_cycles() -> u32 {
//     get_dc_lib_value(&CONFIG.latency.vector_ps_cycles)
// }

// pub fn vector_shift_cycles() -> u32 {
//     get_dc_lib_value(&CONFIG.latency.vector_shift_cycles)
// }

pub fn vector_max_cycles() -> u32 {
    get_dc_lib_value(&CONFIG.latency.vector_max_cycles)
}

pub fn vector_sum_cycles() -> u32 {
    get_dc_lib_value(&CONFIG.latency.vector_sum_cycles)
}

pub fn vector_add_cycles() -> u32 {
    get_dc_lib_value(&CONFIG.latency.vector_add_cycles)
}

pub fn vector_mul_cycles() -> u32 {
    get_dc_lib_value(&CONFIG.latency.vector_mul_cycles)
}

pub fn vector_exp_cycles() -> u32 {
    get_dc_lib_value(&CONFIG.latency.vector_exp_cycles)
}

pub fn vector_reci_cycles() -> u32 {
    get_dc_lib_value(&CONFIG.latency.vector_reci_cycles)
}

pub fn scalar_fp_basic_cycles() -> u32 {
    get_dc_lib_value(&CONFIG.latency.scalar_fp_basic_cycles)
}

pub fn scalar_fp_exp_cycles() -> u32 {
    get_dc_lib_value(&CONFIG.latency.scalar_fp_exp_cycles)
}

pub fn scalar_fp_sqrt_cycles() -> u32 {
    get_dc_lib_value(&CONFIG.latency.scalar_fp_sqrt_cycles)
}

pub fn scalar_fp_reci_cycles() -> u32 {
    get_dc_lib_value(&CONFIG.latency.scalar_fp_reci_cycles)
}

pub fn scalar_int_basic_cycles() -> u32 {
    get_dc_lib_value(&CONFIG.latency.scalar_int_basic_cycles)
}

pub fn max_loop_instructions() -> usize {
    CONFIG.config.max_loop_instructions.value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp_type_config_conversion() {
        let c = FpTypeConfig {
            sign: true,
            exponent: 4,
            mantissa: 3,
        };
        assert_eq!(
            FpType::from(c),
            FpType {
                sign: true,
                exponent: 4,
                mantissa: 3,
            }
        );
    }

    #[test]
    fn test_int_type_config_conversion() {
        assert_eq!(
            IntType::from(IntTypeConfig { width: 32 }),
            IntType { width: 32 }
        );
    }

    #[test]
    fn test_data_type_config_conversion() {
        assert_eq!(
            DataType::from(DataTypeConfig::Fp(FpTypeConfig {
                sign: true,
                exponent: 8,
                mantissa: 7,
            })),
            DataType::Fp(FpType {
                sign: true,
                exponent: 8,
                mantissa: 7,
            })
        );
        assert_eq!(
            DataType::from(DataTypeConfig::Int(IntTypeConfig { width: 16 })),
            DataType::Int(IntType { width: 16 })
        );
    }

    #[test]
    fn test_mx_data_type_config_plain_conversion() {
        let c = MxDataTypeConfig {
            format: "Plain".to_string(),
            data: MxDataTypeData::Plain {
                data_type: DataTypeConfig::Fp(FpTypeConfig {
                    sign: true,
                    exponent: 8,
                    mantissa: 7,
                }),
            },
        };
        assert_eq!(
            MxDataType::from(c),
            MxDataType::Plain(DataType::Fp(FpType {
                sign: true,
                exponent: 8,
                mantissa: 7,
            }))
        );
    }

    #[test]
    fn test_mx_data_type_config_mx_conversion() {
        let c = MxDataTypeConfig {
            format: "Mx".to_string(),
            data: MxDataTypeData::Mx {
                block: 8,
                elem: DataTypeConfig::Fp(FpTypeConfig {
                    sign: true,
                    exponent: 4,
                    mantissa: 3,
                }),
                scale: DataTypeConfig::Fp(FpTypeConfig {
                    sign: false,
                    exponent: 8,
                    mantissa: 0,
                }),
            },
        };
        assert_eq!(
            MxDataType::from(c),
            MxDataType::Mx {
                elem: DataType::Fp(FpType {
                    sign: true,
                    exponent: 4,
                    mantissa: 3,
                }),
                scale: DataType::Fp(FpType {
                    sign: false,
                    exponent: 8,
                    mantissa: 0,
                }),
                block: 8,
            }
        );
    }

    #[test]
    fn test_mx_storage_rejects_non_power_of_two_integer_width() {
        let data_type = MxDataTypeConfig {
            format: "Mx".to_string(),
            data: MxDataTypeData::Mx {
                block: 8,
                elem: DataTypeConfig::Int(IntTypeConfig { width: 3 }),
                scale: DataTypeConfig::Fp(FpTypeConfig {
                    sign: false,
                    exponent: 8,
                    mantissa: 0,
                }),
            },
        };
        assert!(validate_mx_storage_type("HBM_M_KV_TYPE", &data_type, 8).is_err());
    }

    #[test]
    fn test_mx_storage_rejects_non_native_block_size() {
        let data_type = MxDataTypeConfig {
            format: "Mx".to_string(),
            data: MxDataTypeData::Mx {
                block: 16,
                elem: DataTypeConfig::Int(IntTypeConfig { width: 4 }),
                scale: DataTypeConfig::Fp(FpTypeConfig {
                    sign: false,
                    exponent: 8,
                    mantissa: 0,
                }),
            },
        };
        assert!(validate_mx_storage_type("HBM_M_KV_TYPE", &data_type, 8).is_err());
    }

    #[test]
    fn test_default_config_scalar_values() {
        let cfg = AcceleratorConfig::default();
        assert_eq!(cfg.config.blen.value, 32);
        assert_eq!(cfg.config.hlen.value, 16);
        assert_eq!(cfg.config.mlen.value, 32);
        assert_eq!(cfg.config.vlen.value, 32);
        assert_eq!(cfg.config.hbm_size.value, 1073741824);
        assert_eq!(cfg.config.fp_sram_depth.value, 512);
        assert_eq!(cfg.config.dc_en.value, 1);
        assert_eq!(cfg.config.max_loop_instructions.value, 10000);
    }

    #[test]
    fn test_default_precision_types_convert() {
        let cfg = AcceleratorConfig::default();
        // Matrix SRAM defaults to a plain bf16-shaped type (sign, 8 exp, 7 mantissa).
        assert_eq!(
            MxDataType::from(cfg.precision.matrix_sram_type.clone()),
            MxDataType::Plain(DataType::Fp(FpType {
                sign: true,
                exponent: 8,
                mantissa: 7,
            }))
        );
        // HBM matrix weights default to MXFP8 (e4m3 elements, e8m0 scale, block 8).
        assert_eq!(
            MxDataType::from(cfg.precision.hbm_m_weight_type.clone()),
            MxDataType::Mx {
                elem: DataType::Fp(FpType {
                    sign: true,
                    exponent: 4,
                    mantissa: 3,
                }),
                scale: DataType::Fp(FpType {
                    sign: false,
                    exponent: 8,
                    mantissa: 0,
                }),
                block: 8,
            }
        );
    }

    #[test]
    fn test_matrix_semantics_bind_per_family() {
        let mxfp =
            MatrixSemanticsDescriptor::for_binding("mxfp", "FP_E6M5", 6, 5)
                .unwrap();
        assert!(!mxfp.packedkv_selector_rtl_capability.supported);
        mxfp.validate().unwrap();

        let mxint =
            MatrixSemanticsDescriptor::for_binding("mxint", "FP_E6M5", 6, 5)
                .unwrap();
        assert!(mxint.packedkv_selector_rtl_capability.supported);
        mxint.validate().unwrap();
    }

    #[test]
    fn test_physical_semantics_reject_encoding_drift() {
        let descriptor = MxPhysicalSemanticsDescriptor::expected();
        descriptor.validate().unwrap();

        let mut changed = descriptor;
        changed.mxint_encoding = "twos_complement".to_string();
        assert!(changed.validate().is_err());
    }

    #[test]
    fn test_matrix_semantics_reject_drift_and_mixed_family() {
        let mut descriptor =
            MatrixSemanticsDescriptor::for_binding("mxint", "FP_E6M5", 6, 5)
                .unwrap();
        descriptor.mxint_pipeline.max_shift = 15;
        assert!(descriptor.validate().is_err());
        assert!(
            MatrixSemanticsDescriptor::for_binding("mixed", "FP_E6M5", 6, 5)
            .is_err()
        );
    }

    #[test]
    fn test_precision_section_requires_matrix_semantics() {
        let settings = PlenaSettings {
            transactional: AcceleratorConfig::default(),
        };
        let mut value = toml::Value::try_from(settings).unwrap();
        value
            .get_mut("TRANSACTIONAL")
            .and_then(toml::Value::as_table_mut)
            .and_then(|table| table.get_mut("PRECISION"))
            .and_then(toml::Value::as_table_mut)
            .unwrap()
            .remove("MATRIX_SEMANTICS");
        let unbound = toml::to_string(&value).unwrap();
        assert!(toml::from_str::<PlenaSettings>(&unbound).is_err());
    }

    #[test]
    fn test_precision_section_requires_physical_semantics() {
        let settings = PlenaSettings {
            transactional: AcceleratorConfig::default(),
        };
        let mut value = toml::Value::try_from(settings).unwrap();
        value
            .get_mut("TRANSACTIONAL")
            .and_then(toml::Value::as_table_mut)
            .and_then(|table| table.get_mut("PRECISION"))
            .and_then(toml::Value::as_table_mut)
            .unwrap()
            .remove("MX_PHYSICAL_SEMANTICS");
        let unbound = toml::to_string(&value).unwrap();
        assert!(toml::from_str::<PlenaSettings>(&unbound).is_err());
    }

    #[test]
    fn test_repository_config_has_a_valid_bound_contract() {
        let config = load_config_from_file("../plena_settings.toml").unwrap();
        config
            .precision
            .matrix_semantics
            .validate_binding(&config.config, &config.precision.vector_sram_type)
            .unwrap();
        config.precision.mx_physical_semantics.validate().unwrap();
    }
}
