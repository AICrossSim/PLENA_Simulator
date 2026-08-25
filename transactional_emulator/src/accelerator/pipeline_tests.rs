//! End-to-end tests of the scoreboard (pipelined) timing model against the
//! serial baseline: overlap where hardware overlaps, stalls where data
//! dependencies demand, and bit-identical numerics in both modes.

use std::sync::{Arc, Mutex};

use memory::{ErasedMemoryModel, MemoryBacked, NaiveTiming, WithTiming};
use runtime::{Executor, Instant};
use sram::{MatrixSram, VectorSram};

use super::{Accelerator, Scoreboard, TimingDriver};
use crate::matrix_machine::MatrixMachine;
use crate::op;
use crate::runtime_config::{
    BLEN, BROADCAST_AMOUNT, HLEN, MATRIX_SRAM_TYPE, MLEN, PERIOD, PREFETCH_V_AMOUNT,
    SCALAR_INT_BASIC_CYCLES, SYSTOLIC_PROCESSING_OVERHEAD, VECTOR_SRAM_TYPE, VLEN,
};
use crate::timing::{TimingMode, set_timing_mode};
use crate::vector_machine::VectorMachine;

#[derive(Clone, Copy, PartialEq)]
enum RunMode {
    Serial,
    Scoreboard,
    ScoreboardSerialize,
}

struct RunResult {
    now: Instant,
    vram: Vec<u8>,
    mram: Vec<u8>,
    /// First 16 KiB of HBM after the run (covers every address the test
    /// programs touch).
    hbm: Vec<u8>,
}

/// Execute `ops` on a fresh accelerator (NaiveTiming-backed HBM preloaded
/// with `hbm_data`) under the given timing mode and return the final virtual
/// instant plus SRAM dumps.
async fn run_program(ops: Vec<op::Opcode>, mode: RunMode, hbm_data: Vec<u8>) -> RunResult {
    set_timing_mode(match mode {
        RunMode::Serial => TimingMode::Serial,
        _ => TimingMode::Scoreboard,
    });

    let executor = Executor::new();
    let result = Arc::new(Mutex::new(None));
    let result_task = result.clone();

    executor.spawn(async move {
        let mram = Arc::new(MatrixSram::new(
            *MLEN,
            (*MLEN as usize) * 64,
            *MATRIX_SRAM_TYPE,
        ));
        let vram = Arc::new(VectorSram::from_mx_type(*VLEN, 4096, *VECTOR_SRAM_TYPE));
        let m_machine =
            MatrixMachine::new(mram, vram.clone(), *MLEN, *HLEN, *BLEN, *BROADCAST_AMOUNT);
        let v_machine = VectorMachine::new(vram, *VLEN, *HLEN);
        let hbm_concrete = Arc::new(WithTiming::new(
            NaiveTiming::preset_ddr4_2400p(4),
            MemoryBacked::with_capacity(1 << 20),
        ));
        hbm_concrete
            .data()
            .with_data(|f| f[..hbm_data.len()].copy_from_slice(&hbm_data));
        let hbm: Arc<dyn ErasedMemoryModel> = hbm_concrete.clone();

        let mut accelerator = Accelerator::new(m_machine, v_machine, hbm);
        match mode {
            RunMode::Serial => {
                accelerator.do_ops(&ops, None, TimingDriver::Serial).await;
            }
            RunMode::Scoreboard | RunMode::ScoreboardSerialize => {
                let mut scoreboard = Scoreboard::new(mode == RunMode::ScoreboardSerialize);
                accelerator
                    .do_ops(
                        &ops,
                        None,
                        TimingDriver::Scoreboard {
                            scoreboard: &mut scoreboard,
                        },
                    )
                    .await;
            }
        }

        let vram_bytes = accelerator.vram_dump_bytes().await;
        let mram_bytes = accelerator.mram_dump_bytes().await;
        let mut hbm_bytes = vec![0u8; 16 * 1024];
        hbm_concrete
            .data()
            .with_data(|f| hbm_bytes.copy_from_slice(&f[..16 * 1024]));
        *result_task.lock().unwrap() = Some((vram_bytes, mram_bytes, hbm_bytes));
    });

    executor.enter(Instant::ETERNITY).await;
    // Reset for other tests on this thread.
    set_timing_mode(TimingMode::Serial);

    let (vram, mram, hbm) = result.lock().unwrap().take().expect("program completed");
    RunResult {
        now: executor.now(),
        vram,
        mram,
        hbm,
    }
}

fn cycles_of(now: Instant) -> u64 {
    (now - Instant::INIT).as_picos() / PERIOD.as_picos()
}

fn independent_scalar_op() -> op::Opcode {
    // Reads gp1/gp2 (never written here), writes gp3: no RAW with the matrix
    // op or with earlier copies of itself.
    op::Opcode::S_ADD_INT {
        rd: 3,
        rs1: 1,
        rs2: 2,
    }
}

fn matrix_plus_scalars(scalars: usize) -> Vec<op::Opcode> {
    let mut ops = vec![op::Opcode::M_MM { rs1: 1, rs2: 2 }];
    for _ in 0..scalars {
        ops.push(independent_scalar_op());
    }
    ops
}

#[tokio::test]
async fn independent_scalar_work_hides_under_a_matrix_op() {
    let serial = run_program(matrix_plus_scalars(10), RunMode::Serial, vec![]).await;
    let pipelined = run_program(matrix_plus_scalars(10), RunMode::Scoreboard, vec![]).await;

    let matrix_latency = (*SYSTOLIC_PROCESSING_OVERHEAD + *MLEN) as u64;
    let scalar_latency = *SCALAR_INT_BASIC_CYCLES as u64;
    assert_eq!(cycles_of(serial.now), matrix_latency + 10 * scalar_latency);
    // Pipelined: the matrix op owns the systolic array; the ten scalar ops
    // issue one per cycle underneath it.
    assert_eq!(
        cycles_of(pipelined.now),
        matrix_latency.max(10 + scalar_latency)
    );
    assert_eq!(serial.vram, pipelined.vram);
    assert_eq!(serial.mram, pipelined.mram);
}

#[tokio::test]
async fn dependent_scalar_chain_does_not_pipeline() {
    // Each op reads and writes gp3: a pure RAW chain.
    let chain = || -> Vec<op::Opcode> {
        (0..6)
            .map(|_| op::Opcode::S_ADD_INT {
                rd: 3,
                rs1: 3,
                rs2: 3,
            })
            .collect()
    };

    let serial = run_program(chain(), RunMode::Serial, vec![]).await;
    let pipelined = run_program(chain(), RunMode::Scoreboard, vec![]).await;

    // Reproduce the scoreboard recurrence: issue_i = max(prev_issue + 1,
    // ready(gp3)); finish_i = issue_i + latency.
    let latency = *SCALAR_INT_BASIC_CYCLES as u64;
    let mut issue_prev: Option<u64> = None;
    let mut ready = 0u64;
    let mut finish = 0u64;
    for _ in 0..6 {
        let issue = issue_prev.map_or(0, |p| p + 1).max(ready);
        finish = issue + latency;
        ready = finish;
        issue_prev = Some(issue);
    }
    assert_eq!(cycles_of(serial.now), 6 * latency);
    assert_eq!(cycles_of(pipelined.now), finish);
}

#[tokio::test]
async fn serialize_mode_reproduces_serial_cycle_counts() {
    let make_ops = || {
        let mut ops = matrix_plus_scalars(7);
        ops.push(op::Opcode::V_ADD_VV {
            rd: 5,
            rs1: 5,
            rs2: 5,
            rmask: 0,
        });
        ops
    };

    let serial = run_program(make_ops(), RunMode::Serial, vec![]).await;
    let serialized = run_program(make_ops(), RunMode::ScoreboardSerialize, vec![]).await;
    assert_eq!(
        serial.now, serialized.now,
        "--scoreboard-serialize must reproduce serial timing exactly"
    );
    assert_eq!(serial.vram, serialized.vram);

    // Also with a prefetch in the stream: serialize mode keeps DMA inline, so
    // the HBM transfer is issued at the same instant as in serial mode and
    // the totals must still match exactly.
    let serial = run_program(prefetch_program(8, true), RunMode::Serial, patterned_hbm()).await;
    let serialized = run_program(
        prefetch_program(8, true),
        RunMode::ScoreboardSerialize,
        patterned_hbm(),
    )
    .await;
    assert_eq!(
        serial.now, serialized.now,
        "--scoreboard-serialize must keep prefetches inline"
    );
    assert_eq!(serial.vram, serialized.vram);
}

/// Program: point gp5 at a vram row far from the prefetch destination, issue
/// an H_PREFETCH_V into rows 0..PREFETCH_V_AMOUNT, run `independent` vector
/// adds on the far row, then (optionally) a dependent add that reads the
/// prefetched row 0.
fn prefetch_program(independent: usize, dependent: bool) -> Vec<op::Opcode> {
    let mut ops = vec![
        // gp5 = 16 rows past the prefetch destination.
        op::Opcode::S_ADDI_INT {
            rd: 5,
            rs1: 0,
            imm: *VLEN * (*PREFETCH_V_AMOUNT + 16),
        },
        // gp4 = 0 (default): prefetch fills rows 0..PREFETCH_V_AMOUNT.
        op::Opcode::H_PREFETCH_V {
            rd: 4,
            rs1: 0,
            rs2: 0,
            rstride: 0,
            precision: op::VectorPrecision::Activation,
        },
    ];
    for _ in 0..independent {
        ops.push(op::Opcode::V_ADD_VV {
            rd: 5,
            rs1: 5,
            rs2: 5,
            rmask: 0,
        });
    }
    if dependent {
        // Reads the prefetched row 0 (gp4 = 0), writes the far row.
        ops.push(op::Opcode::V_ADD_VV {
            rd: 5,
            rs1: 4,
            rs2: 4,
            rmask: 0,
        });
    }
    ops
}

fn patterned_hbm() -> Vec<u8> {
    (0..4096u32).map(|i| (i % 251) as u8).collect()
}

#[tokio::test]
async fn prefetch_hides_under_independent_vector_work_and_stalls_dependents() {
    let serial = run_program(prefetch_program(8, true), RunMode::Serial, patterned_hbm()).await;
    let pipelined = run_program(
        prefetch_program(8, true),
        RunMode::Scoreboard,
        patterned_hbm(),
    )
    .await;

    assert!(
        pipelined.now < serial.now,
        "prefetch latency must overlap independent vector work \
         (pipelined {:?} vs serial {:?})",
        pipelined.now,
        serial.now
    );
    // Numerics must be bit-identical: the dependent add consumed the
    // Cell::Pending fill path instead of the inline continous_write_delayed.
    assert_eq!(serial.vram, pipelined.vram);
    assert_eq!(serial.mram, pipelined.mram);
}

#[tokio::test]
async fn dependent_read_waits_for_the_prefetch_it_consumes() {
    // With no independent work, the dependent read exposes the full DMA
    // latency: total time cannot be shorter than the prefetch alone.
    let prefetch_only = run_program(
        prefetch_program(0, false),
        RunMode::Scoreboard,
        patterned_hbm(),
    )
    .await;
    let with_dependent = run_program(
        prefetch_program(0, true),
        RunMode::Scoreboard,
        patterned_hbm(),
    )
    .await;

    assert!(
        with_dependent.now >= prefetch_only.now,
        "a dependent read cannot finish before the transfer it consumes"
    );
    // And it must match serial numerics.
    let serial = run_program(prefetch_program(0, true), RunMode::Serial, patterned_hbm()).await;
    assert_eq!(serial.vram, with_dependent.vram);
}

/// Program: prefetch HBM into rows 0..PREFETCH_V_AMOUNT, store those rows
/// back to a disjoint HBM region, then immediately overwrite the store's
/// source row (WAR against the in-flight store) and run independent vector
/// work the store latency can hide under.
fn store_program(independent: usize) -> Vec<op::Opcode> {
    let mut ops = vec![
        // gp5 = far row base for independent work.
        op::Opcode::S_ADDI_INT {
            rd: 5,
            rs1: 0,
            imm: *VLEN * (*PREFETCH_V_AMOUNT + 16),
        },
        // gp7 = HBM byte offset for the store destination (disjoint from the
        // prefetch source region at offset 0).
        op::Opcode::S_ADDI_INT {
            rd: 7,
            rs1: 0,
            imm: 8192,
        },
        // Fill rows 0.. with real data from HBM offset 0.
        op::Opcode::H_PREFETCH_V {
            rd: 4,
            rs1: 0,
            rs2: 0,
            rstride: 0,
            precision: op::VectorPrecision::Activation,
        },
        // Store rows starting at gp0 = 0 to HBM offset gp7.
        op::Opcode::H_STORE_V {
            rd: 0,
            rs1: 7,
            rs2: 0,
            rstride: 0,
            precision: op::VectorPrecision::Activation,
        },
        // WAR: overwrite the store's source row 0 (gp4 = 0) while the store's
        // HBM writes may still be in flight — the issue-time snapshot must
        // protect the stored values.
        op::Opcode::V_ADD_VV {
            rd: 4,
            rs1: 5,
            rs2: 5,
            rmask: 0,
        },
    ];
    for _ in 0..independent {
        ops.push(op::Opcode::V_ADD_VV {
            rd: 5,
            rs1: 5,
            rs2: 5,
            rmask: 0,
        });
    }
    ops
}

/// HBM preload of small-magnitude values: whatever element/scale format the
/// configured activation type decodes these bytes as, the results stay far
/// from overflow, so the store's re-quantization to the HBM type is safe.
fn small_valued_hbm() -> Vec<u8> {
    vec![0x3C; 8192]
}

#[tokio::test]
async fn async_store_overlaps_and_snapshots_against_war() {
    let serial = run_program(store_program(8), RunMode::Serial, small_valued_hbm()).await;
    let pipelined = run_program(store_program(8), RunMode::Scoreboard, small_valued_hbm()).await;

    assert!(
        pipelined.now < serial.now,
        "store latency must overlap independent vector work \
         (pipelined {:?} vs serial {:?})",
        pipelined.now,
        serial.now
    );
    // The WAR overwrite must not corrupt the stored HBM bytes: the snapshot
    // taken at issue is what lands in HBM, exactly as in the serial run.
    assert_eq!(serial.hbm, pipelined.hbm);
    assert_eq!(serial.vram, pipelined.vram);

    // And serialize mode (inline DMA) must still reproduce serial exactly.
    let serialized = run_program(
        store_program(8),
        RunMode::ScoreboardSerialize,
        small_valued_hbm(),
    )
    .await;
    assert_eq!(serial.now, serialized.now);
    assert_eq!(serial.hbm, serialized.hbm);
}

#[tokio::test]
async fn back_to_back_matrix_ops_serialize_on_the_matrix_unit() {
    let ops = vec![
        op::Opcode::M_MM { rs1: 1, rs2: 2 },
        op::Opcode::M_MM { rs1: 1, rs2: 2 },
        op::Opcode::M_MM_WO {
            rd: 1,
            rstride: 0,
            imm: 0,
        },
    ];
    let pipelined = run_program(ops, RunMode::Scoreboard, vec![]).await;
    let matrix_latency = (*SYSTOLIC_PROCESSING_OVERHEAD + *MLEN) as u64;
    // Two accumulates serialize on the (single) systolic array; the write-out
    // costs its captured latency (compute(1)) after them.
    assert_eq!(cycles_of(pipelined.now), 2 * matrix_latency + 1);
}
