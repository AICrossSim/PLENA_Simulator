//! Emit the transactional emulator's calibrated opcode timing as JSON.
//!
//! This binary is intentionally small and read-only.  Cross-language parity
//! tests point `PLENA_SETTINGS_TOML` at a generated configuration, invoke this
//! executable, and compare every field with CostEmitter's Python timing model.

#![allow(dead_code)]

#[path = "../load_config.rs"]
mod load_config;
#[path = "../op.rs"]
mod op;
#[path = "../opcode_timing.rs"]
mod opcode_timing;
#[path = "../runtime_config.rs"]
mod runtime_config;

use std::collections::BTreeMap;

use op::{MatrixPrecision, Opcode, VectorOrder, VectorPrecision};
use serde::Serialize;

#[derive(Serialize)]
struct TimingDump {
    hardware: Hardware,
    calibration: opcode_timing::TimingCalibrationMetadata<'static>,
    opcodes: BTreeMap<&'static str, Option<opcode_timing::OpcodeTimingEstimate>>,
}

#[derive(Serialize)]
struct Hardware {
    mlen: u32,
    blen: u32,
    vlen: u32,
    hlen: u32,
    broadcast_amount: u32,
}

fn opcode_samples() -> Vec<(&'static str, Opcode)> {
    vec![
        ("M_MM", Opcode::M_MM { rs1: 1, rs2: 2 }),
        ("M_TMM", Opcode::M_TMM { rs1: 1, rs2: 2 }),
        ("M_BMM", Opcode::M_BMM { rs1: 1, rs2: 2 }),
        ("M_BTMM", Opcode::M_BTMM { rs1: 1, rs2: 2 }),
        (
            "M_MM_WO",
            Opcode::M_MM_WO {
                rd: 1,
                rstride: 2,
                imm: 0,
            },
        ),
        ("M_BMM_WO", Opcode::M_BMM_WO { rd: 1, imm: 0 }),
        ("M_MV", Opcode::M_MV { rs1: 1, rs2: 2 }),
        ("M_TMV", Opcode::M_TMV { rs1: 1, rs2: 2 }),
        (
            "M_BMV",
            Opcode::M_BMV {
                rs1: 1,
                rs2: 2,
                rd: 3,
            },
        ),
        (
            "M_BTMV",
            Opcode::M_BTMV {
                rs1: 1,
                rs2: 2,
                rd: 3,
            },
        ),
        ("M_MV_WO", Opcode::M_MV_WO { rd: 1, imm: 0 }),
        ("M_BMV_WO", Opcode::M_BMV_WO { rd: 1, imm: 0 }),
        (
            "V_ADD_VV",
            Opcode::V_ADD_VV {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rmask: 0,
            },
        ),
        (
            "V_ADD_VF",
            Opcode::V_ADD_VF {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rmask: 0,
            },
        ),
        (
            "V_SUB_VV",
            Opcode::V_SUB_VV {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rmask: 0,
            },
        ),
        (
            "V_SUB_VF",
            Opcode::V_SUB_VF {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rmask: 0,
                rorder: VectorOrder::Normal,
            },
        ),
        (
            "V_MUL_VV",
            Opcode::V_MUL_VV {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rmask: 0,
            },
        ),
        (
            "V_MUL_VF",
            Opcode::V_MUL_VF {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rmask: 0,
            },
        ),
        (
            "V_EXP_V",
            Opcode::V_EXP_V {
                rd: 1,
                rs1: 2,
                rmask: 0,
            },
        ),
        (
            "V_RECI_V",
            Opcode::V_RECI_V {
                rd: 1,
                rs1: 2,
                rmask: 0,
            },
        ),
        (
            "V_RED_SUM",
            Opcode::V_RED_SUM {
                rd: 1,
                rs1: 2,
                rmask: 0,
            },
        ),
        (
            "V_RED_MAX",
            Opcode::V_RED_MAX {
                rd: 1,
                rs1: 2,
                rmask: 0,
            },
        ),
        (
            "V_SHIFT_V",
            Opcode::V_SHIFT_V {
                rd: 1,
                rs1: 2,
                rs2: 3,
            },
        ),
        (
            "S_ADD_FP",
            Opcode::S_ADD_FP {
                rd: 1,
                rs1: 2,
                rs2: 3,
            },
        ),
        (
            "S_SUB_FP",
            Opcode::S_SUB_FP {
                rd: 1,
                rs1: 2,
                rs2: 3,
            },
        ),
        (
            "S_MAX_FP",
            Opcode::S_MAX_FP {
                rd: 1,
                rs1: 2,
                rs2: 3,
            },
        ),
        (
            "S_MUL_FP",
            Opcode::S_MUL_FP {
                rd: 1,
                rs1: 2,
                rs2: 3,
            },
        ),
        ("S_EXP_FP", Opcode::S_EXP_FP { rd: 1, rs1: 2 }),
        ("S_RECI_FP", Opcode::S_RECI_FP { rd: 1, rs1: 2 }),
        ("S_SQRT_FP", Opcode::S_SQRT_FP { rd: 1, rs1: 2 }),
        (
            "S_LD_FP",
            Opcode::S_LD_FP {
                rd: 1,
                rs1: 2,
                imm: 0,
            },
        ),
        (
            "S_ST_FP",
            Opcode::S_ST_FP {
                rd: 1,
                rs1: 2,
                imm: 0,
            },
        ),
        (
            "S_MAP_V_FP",
            Opcode::S_MAP_V_FP {
                rd: 1,
                rs1: 2,
                imm: 0,
            },
        ),
        (
            "S_ADD_INT",
            Opcode::S_ADD_INT {
                rd: 1,
                rs1: 2,
                rs2: 3,
            },
        ),
        (
            "S_ADDI_INT",
            Opcode::S_ADDI_INT {
                rd: 1,
                rs1: 2,
                imm: 0,
            },
        ),
        (
            "S_SUB_INT",
            Opcode::S_SUB_INT {
                rd: 1,
                rs1: 2,
                rs2: 3,
            },
        ),
        (
            "S_MUL_INT",
            Opcode::S_MUL_INT {
                rd: 1,
                rs1: 2,
                rs2: 3,
            },
        ),
        ("S_LUI_INT", Opcode::S_LUI_INT { rd: 1, imm: 0 }),
        (
            "S_LD_INT",
            Opcode::S_LD_INT {
                rd: 1,
                rs1: 2,
                imm: 0,
            },
        ),
        (
            "S_ST_INT",
            Opcode::S_ST_INT {
                rd: 1,
                rs1: 2,
                imm: 0,
            },
        ),
        (
            "H_PREFETCH_M",
            Opcode::H_PREFETCH_M {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rstride: 4,
                precision: MatrixPrecision::Weights,
            },
        ),
        (
            "H_PREFETCH_V",
            Opcode::H_PREFETCH_V {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rstride: 4,
                precision: VectorPrecision::Activation,
            },
        ),
        (
            "H_STORE_V",
            Opcode::H_STORE_V {
                rd: 1,
                rs1: 2,
                rs2: 3,
                rstride: 4,
                precision: VectorPrecision::Activation,
            },
        ),
        (
            "C_SET_ADDR_REG",
            Opcode::C_SET_ADDR_REG {
                rd: 1,
                rs1: 2,
                rs2: 3,
            },
        ),
        ("C_SET_SCALE_REG", Opcode::C_SET_SCALE_REG { rd: 1 }),
        ("C_SET_STRIDE_REG", Opcode::C_SET_STRIDE_REG { rd: 1 }),
        ("C_SET_V_MASK_REG", Opcode::C_SET_V_MASK_REG { rd: 1 }),
        ("C_LOOP_START", Opcode::C_LOOP_START { rd: 1, imm: 1 }),
        ("C_LOOP_END", Opcode::C_LOOP_END { rd: 1 }),
        ("C_BREAK", Opcode::C_BREAK),
    ]
}

fn main() {
    let opcodes = opcode_samples()
        .into_iter()
        .map(|(name, opcode)| (name, opcode_timing::calibrated_timing(&opcode)))
        .collect();
    let dump = TimingDump {
        hardware: Hardware {
            mlen: *runtime_config::MLEN,
            blen: *runtime_config::BLEN,
            vlen: *runtime_config::VLEN,
            hlen: *runtime_config::HLEN,
            broadcast_amount: *runtime_config::BROADCAST_AMOUNT,
        },
        calibration: opcode_timing::calibration_metadata(),
        opcodes,
    };
    println!("{}", serde_json::to_string_pretty(&dump).unwrap());
}
