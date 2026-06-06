use std::collections::BTreeMap;

use runtime::Duration;
use serde::Serialize;

use crate::cli::ProfileMemoryLevel;
use crate::op::Opcode;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ProfileCategory {
    Memory,
    MatrixCompute,
    VectorCompute,
    ScalarCompute,
    Control,
    Other,
}

impl ProfileCategory {
    fn as_key(self) -> &'static str {
        match self {
            Self::Memory => "memory",
            Self::MatrixCompute => "matrix_compute",
            Self::VectorCompute => "vector_compute",
            Self::ScalarCompute => "scalar_compute",
            Self::Control => "control",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct ProfileCounter {
    count: u64,
    total_picos: u64,
    total_ns: f64,
    max_picos: u64,
    percent_of_profiled_time: f64,
}

impl ProfileCounter {
    fn add(&mut self, delta: Duration) {
        let picos = delta.as_picos();
        self.count += 1;
        self.total_picos += picos;
        self.max_picos = self.max_picos.max(picos);
    }

    fn finalize(&mut self, profiled_picos: u64) {
        self.total_ns = self.total_picos as f64 / 1000.0;
        self.percent_of_profiled_time = if profiled_picos == 0 {
            0.0
        } else {
            self.total_picos as f64 * 100.0 / profiled_picos as f64
        };
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct OpcodeProfileCounter {
    category: ProfileCategory,
    #[serde(flatten)]
    counter: ProfileCounter,
}

impl OpcodeProfileCounter {
    fn new(category: ProfileCategory) -> Self {
        Self {
            category,
            counter: ProfileCounter::default(),
        }
    }

    fn add(&mut self, delta: Duration) {
        self.counter.add(delta);
    }

    fn finalize(&mut self, profiled_picos: u64) {
        self.counter.finalize(profiled_picos);
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct MemoryProfileReport {
    schema_version: u32,
    profile_level: &'static str,
    program_total_picos: u64,
    program_total_ns: f64,
    hbm_bytes_read: u64,
    hbm_bytes_written: u64,
    categories: BTreeMap<&'static str, ProfileCounter>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    opcodes: BTreeMap<&'static str, OpcodeProfileCounter>,
}

#[derive(Clone, Debug)]
pub(crate) struct MemoryProfiler {
    level: ProfileMemoryLevel,
    program_total_picos: u64,
    categories: BTreeMap<ProfileCategory, ProfileCounter>,
    opcodes: BTreeMap<&'static str, OpcodeProfileCounter>,
}

impl MemoryProfiler {
    pub(crate) fn new(level: ProfileMemoryLevel) -> Self {
        Self {
            level,
            program_total_picos: 0,
            categories: BTreeMap::new(),
            opcodes: BTreeMap::new(),
        }
    }

    pub(crate) fn record(&mut self, opcode: &Opcode, delta: Duration) {
        let category = category_for(opcode);
        self.program_total_picos += delta.as_picos();
        self.categories.entry(category).or_default().add(delta);

        if matches!(self.level, ProfileMemoryLevel::Opcode) {
            let mnemonic = opcode_mnemonic(opcode);
            self.opcodes
                .entry(mnemonic)
                .or_insert_with(|| OpcodeProfileCounter::new(category))
                .add(delta);
        }
    }

    pub(crate) fn report(
        &self,
        hbm_bytes_read: u64,
        hbm_bytes_written: u64,
    ) -> MemoryProfileReport {
        let mut categories = BTreeMap::new();
        for (category, counter) in &self.categories {
            let mut counter = counter.clone();
            counter.finalize(self.program_total_picos);
            categories.insert(category.as_key(), counter);
        }

        let mut opcodes = BTreeMap::new();
        if matches!(self.level, ProfileMemoryLevel::Opcode) {
            for (mnemonic, counter) in &self.opcodes {
                let mut counter = counter.clone();
                counter.finalize(self.program_total_picos);
                opcodes.insert(*mnemonic, counter);
            }
        }

        MemoryProfileReport {
            schema_version: 1,
            profile_level: self.level.as_str(),
            program_total_picos: self.program_total_picos,
            program_total_ns: self.program_total_picos as f64 / 1000.0,
            hbm_bytes_read,
            hbm_bytes_written,
            categories,
            opcodes,
        }
    }
}

pub(crate) fn category_for(opcode: &Opcode) -> ProfileCategory {
    match opcode {
        Opcode::H_PREFETCH_M { .. } | Opcode::H_PREFETCH_V { .. } | Opcode::H_STORE_V { .. } => {
            ProfileCategory::Memory
        }
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
        | Opcode::M_BMV_WO { .. } => ProfileCategory::MatrixCompute,
        Opcode::V_ADD_VV { .. }
        | Opcode::V_ADD_VF { .. }
        | Opcode::V_SUB_VV { .. }
        | Opcode::V_SUB_VF { .. }
        | Opcode::V_MUL_VV { .. }
        | Opcode::V_MUL_VF { .. }
        | Opcode::V_EXP_V { .. }
        | Opcode::V_RECI_V { .. }
        | Opcode::V_RED_SUM { .. }
        | Opcode::V_RED_MAX { .. }
        | Opcode::V_SHIFT_V { .. } => ProfileCategory::VectorCompute,
        Opcode::S_ADD_FP { .. }
        | Opcode::S_SUB_FP { .. }
        | Opcode::S_MAX_FP { .. }
        | Opcode::S_MUL_FP { .. }
        | Opcode::S_EXP_FP { .. }
        | Opcode::S_RECI_FP { .. }
        | Opcode::S_SQRT_FP { .. }
        | Opcode::S_LD_FP { .. }
        | Opcode::S_ST_FP { .. }
        | Opcode::S_MAP_V_FP { .. }
        | Opcode::S_ADD_INT { .. }
        | Opcode::S_ADDI_INT { .. }
        | Opcode::S_SUB_INT { .. }
        | Opcode::S_MUL_INT { .. }
        | Opcode::S_LUI_INT { .. }
        | Opcode::S_LD_INT { .. }
        | Opcode::S_ST_INT { .. } => ProfileCategory::ScalarCompute,
        Opcode::C_SET_ADDR_REG { .. }
        | Opcode::C_SET_SCALE_REG { .. }
        | Opcode::C_SET_STRIDE_REG { .. }
        | Opcode::C_SET_V_MASK_REG { .. }
        | Opcode::C_LOOP_START { .. }
        | Opcode::C_LOOP_END { .. }
        | Opcode::C_BREAK => ProfileCategory::Control,
        Opcode::Invalid => ProfileCategory::Other,
    }
}

pub(crate) fn opcode_mnemonic(opcode: &Opcode) -> &'static str {
    match opcode {
        Opcode::Invalid => "INVALID",
        Opcode::M_MM { .. } => "M_MM",
        Opcode::M_TMM { .. } => "M_TMM",
        Opcode::M_BMM { .. } => "M_BMM",
        Opcode::M_BTMM { .. } => "M_BTMM",
        Opcode::M_BMM_WO { .. } => "M_BMM_WO",
        Opcode::M_MM_WO { .. } => "M_MM_WO",
        Opcode::M_MV { .. } => "M_MV",
        Opcode::M_TMV { .. } => "M_TMV",
        Opcode::M_BMV { .. } => "M_BMV",
        Opcode::M_BTMV { .. } => "M_BTMV",
        Opcode::M_MV_WO { .. } => "M_MV_WO",
        Opcode::M_BMV_WO { .. } => "M_BMV_WO",
        Opcode::V_ADD_VV { .. } => "V_ADD_VV",
        Opcode::V_ADD_VF { .. } => "V_ADD_VF",
        Opcode::V_SUB_VV { .. } => "V_SUB_VV",
        Opcode::V_SUB_VF { .. } => "V_SUB_VF",
        Opcode::V_MUL_VV { .. } => "V_MUL_VV",
        Opcode::V_MUL_VF { .. } => "V_MUL_VF",
        Opcode::V_EXP_V { .. } => "V_EXP_V",
        Opcode::V_RECI_V { .. } => "V_RECI_V",
        Opcode::V_RED_SUM { .. } => "V_RED_SUM",
        Opcode::V_RED_MAX { .. } => "V_RED_MAX",
        Opcode::S_ADD_FP { .. } => "S_ADD_FP",
        Opcode::S_SUB_FP { .. } => "S_SUB_FP",
        Opcode::S_MAX_FP { .. } => "S_MAX_FP",
        Opcode::S_MUL_FP { .. } => "S_MUL_FP",
        Opcode::S_EXP_FP { .. } => "S_EXP_FP",
        Opcode::S_RECI_FP { .. } => "S_RECI_FP",
        Opcode::S_SQRT_FP { .. } => "S_SQRT_FP",
        Opcode::S_LD_FP { .. } => "S_LD_FP",
        Opcode::S_ST_FP { .. } => "S_ST_FP",
        Opcode::S_MAP_V_FP { .. } => "S_MAP_V_FP",
        Opcode::S_ADD_INT { .. } => "S_ADD_INT",
        Opcode::S_ADDI_INT { .. } => "S_ADDI_INT",
        Opcode::S_SUB_INT { .. } => "S_SUB_INT",
        Opcode::S_MUL_INT { .. } => "S_MUL_INT",
        Opcode::S_LUI_INT { .. } => "S_LUI_INT",
        Opcode::S_LD_INT { .. } => "S_LD_INT",
        Opcode::S_ST_INT { .. } => "S_ST_INT",
        Opcode::H_PREFETCH_M { .. } => "H_PREFETCH_M",
        Opcode::H_PREFETCH_V { .. } => "H_PREFETCH_V",
        Opcode::H_STORE_V { .. } => "H_STORE_V",
        Opcode::C_SET_ADDR_REG { .. } => "C_SET_ADDR_REG",
        Opcode::C_SET_SCALE_REG { .. } => "C_SET_SCALE_REG",
        Opcode::C_SET_STRIDE_REG { .. } => "C_SET_STRIDE_REG",
        Opcode::C_SET_V_MASK_REG { .. } => "C_SET_V_MASK_REG",
        Opcode::C_LOOP_START { .. } => "C_LOOP_START",
        Opcode::C_LOOP_END { .. } => "C_LOOP_END",
        Opcode::V_SHIFT_V { .. } => "V_SHIFT_V",
        Opcode::C_BREAK => "C_BREAK",
    }
}

#[cfg(test)]
mod tests {
    use runtime::Duration;

    use super::*;

    #[test]
    fn categorizes_representative_opcodes() {
        assert_eq!(
            category_for(&Opcode::H_PREFETCH_M {
                rd: 0,
                rs1: 0,
                rs2: 0,
                rstride: 0,
                precision: crate::op::MatrixPrecision::Weights,
            }),
            ProfileCategory::Memory
        );
        assert_eq!(
            category_for(&Opcode::M_MM { rs1: 0, rs2: 0 }),
            ProfileCategory::MatrixCompute
        );
        assert_eq!(
            category_for(&Opcode::V_EXP_V {
                rd: 0,
                rs1: 0,
                rmask: 0,
            }),
            ProfileCategory::VectorCompute
        );
        assert_eq!(
            category_for(&Opcode::S_ADD_INT {
                rd: 0,
                rs1: 0,
                rs2: 0,
            }),
            ProfileCategory::ScalarCompute
        );
        assert_eq!(category_for(&Opcode::C_BREAK), ProfileCategory::Control);
    }

    #[test]
    fn aggregates_and_serializes_profile() {
        let mut profiler = MemoryProfiler::new(ProfileMemoryLevel::Opcode);
        profiler.record(&Opcode::M_MM { rs1: 0, rs2: 0 }, Duration::from_nanos(10));
        profiler.record(
            &Opcode::H_STORE_V {
                rd: 0,
                rs1: 0,
                rs2: 0,
                rstride: 0,
                precision: crate::op::VectorPrecision::Activation,
            },
            Duration::from_nanos(30),
        );

        let report = profiler.report(128, 64);
        let json = serde_json::to_string(&report).expect("profile serializes");
        assert!(json.contains("\"profile_level\":\"opcode\""));
        assert!(json.contains("\"matrix_compute\""));
        assert!(json.contains("\"H_STORE_V\""));
        assert_eq!(report.program_total_picos, 40_000);
        assert_eq!(report.hbm_bytes_read, 128);
        assert_eq!(report.hbm_bytes_written, 64);
    }
}
