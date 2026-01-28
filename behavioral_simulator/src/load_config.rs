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
                hbm_m_prefetch_amount: ConfigValue { value: 16 },
                hbm_v_prefetch_amount: ConfigValue { value: 16 },
                hbm_v_writeback_amount: ConfigValue { value: 16 },
                dc_en: ConfigValue { value: 1 },
                max_loop_instructions: ConfigValueUsize { value: 10000 },
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
                            sign: true,
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
                            sign: true,
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
                            sign: true,
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
                            sign: true,
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
    load_config().unwrap_or_else(|e| {
        eprintln!("Failed to load config: {}. Using defaults.", e);
        AcceleratorConfig::default()
    })
});

// Configuration loading functions
pub fn load_config() -> Result<AcceleratorConfig, Box<dyn std::error::Error>> {
    let config_path = env::current_dir()
        .unwrap()
        .parent()
        .unwrap()
        .join("src/definitions/plena_settings.toml");

    let config_path = config_path.to_str().unwrap();
    if let Ok(config) = load_config_from_file(config_path) {
        return Ok(config);
    }

    Err("No configuration file found".into())
}

pub fn load_config_from_file(path: &str) -> Result<AcceleratorConfig, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let config: AcceleratorConfig = toml::from_str(&content)?;
    Ok(config)
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

pub fn matrix_sram_size() -> usize {
    CONFIG.config.matrix_sram_size.value
}

pub fn vector_sram_size() -> usize {
    CONFIG.config.vector_sram_size.value
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
