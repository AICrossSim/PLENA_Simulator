//! End-to-end tests of the scoreboard (pipelined) timing model against the
//! serial baseline: overlap where hardware overlaps, stalls where data
//! dependencies demand, and bit-identical numerics in both modes.

use std::sync::{Arc, Mutex};

use half::bf16;
use memory::{ErasedMemoryModel, MemoryBacked, NaiveTiming, WithTiming};
use quantize::{QuantTensor, tensor_to_f32_vec};
use runtime::{Executor, Instant};
use sram::{MatrixSram, VectorSram};
use tch::Tensor;

use super::{Accelerator, Scoreboard, TimingDriver};
use crate::matrix_machine::MatrixMachine;
use crate::op;
use crate::runtime_config::{
    BLEN, BROADCAST_AMOUNT, HLEN, MATRIX_SRAM_TYPE, MLEN, PERIOD, PREFETCH_V_AMOUNT,
    SCALAR_INT_BASIC_CYCLES, SYSTOLIC_PROCESSING_OVERHEAD, VECTOR_SRAM_TYPE, VLEN,
};
use crate::timing::{TimingMode, set_timing_mode};
use crate::vector_machine::{PacketCounterSnapshot, VectorMachine};

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

fn cfg(value: u8, target: u8, slot: u8, field: u8) -> op::Opcode {
    op::Opcode::L_CFG {
        value,
        target,
        slot,
        field,
    }
}

fn set_gp(rd: u8, value: u32) -> op::Opcode {
    op::Opcode::S_ADDI_INT {
        rd,
        rs1: 0,
        imm: value,
    }
}

fn stream_field(ops: &mut Vec<op::Opcode>, value: u32, target: u8, slot: u8, field: u8) {
    ops.push(set_gp(1, value));
    ops.push(cfg(1, target, slot, field));
}

async fn run_matrix_view_packet_roundtrip(tile_pitch_rows: u32) -> (Vec<f32>, u64, u64) {
    assert_eq!((*MLEN, *BLEN, *VLEN), (64, 4, 64));
    let mram = Arc::new(MatrixSram::with_banks(
        *MLEN,
        (*MLEN as usize) * 64,
        *BLEN,
        *MATRIX_SRAM_TYPE,
    ));
    let vram = Arc::new(VectorSram::from_mx_type(*VLEN, 8, *VECTOR_SRAM_TYPE));
    let values = (0..*VLEN)
        .map(|value| value as f32 + 1.0)
        .collect::<Vec<_>>();
    let layout = sram::matrix::MatrixLayout {
        rows: 1,
        cols: 2 * *BLEN,
        tile_count: *VLEN / (2 * *BLEN),
        tile_pitch_rows,
        alpha: 1,
        tile_skew: 0,
    };
    mram.write_layout_packet(
        0,
        layout,
        QuantTensor::quantize(Tensor::from_slice(&values), *MATRIX_SRAM_TYPE),
    )
    .await;
    mram.reset_packet_counters();

    let m_machine = MatrixMachine::new(
        mram.clone(),
        vram.clone(),
        *MLEN,
        *HLEN,
        *BLEN,
        *BROADCAST_AMOUNT,
    );
    let v_machine = VectorMachine::new(vram.clone(), *VLEN, *HLEN);
    let hbm: Arc<dyn ErasedMemoryModel> = Arc::new(WithTiming::new(
        NaiveTiming::preset_ddr4_2400p(4),
        MemoryBacked::with_capacity(4096),
    ));
    let mut accelerator = Accelerator::new(m_machine, v_machine, hbm);
    // Eight 1x8 tiles form one 64-value packet. Pitch 1 overlaps adjacent
    // two-word tiles on the fixed diagonal banks; pitch 2 uses every bank once.
    let shape = (7 << 12) | (7 << 24);
    let mapping = tile_pitch_rows | (1 << 28);
    let ops = vec![
        set_gp(1, shape),
        set_gp(2, mapping),
        set_gp(3, *VLEN),
        set_gp(4, 0),
        set_gp(5, 0),
        op::Opcode::L_TILE_CFG {
            shape: 1,
            mapping: 2,
            slot: 1,
        },
        op::Opcode::V_ADD_VV {
            rd: 3,
            rs1: 4,
            rs2: 5,
            rmask: 0,
            // Explicit Matrix marker + source-1 slot.
            lmask: 0x8 | 0b010,
        },
    ];
    let start = Executor::current().now();
    accelerator.do_ops(&ops, None, TimingDriver::Serial).await;
    let cycles = (Executor::current().now() - start).as_picos() / PERIOD.as_picos();
    let counters = accelerator.matrix_view_packet_counters();
    (
        tensor_to_f32_vec(vram.read(*VLEN).await.as_tensor()),
        counters.bank_stall_cycles,
        cycles,
    )
}

#[tokio::test]
async fn l_mview_dispatch_roundtrips_values_and_removes_real_matrix_bank_conflicts() {
    set_timing_mode(TimingMode::Serial);
    let executor = Executor::new();
    let result = Arc::new(Mutex::new(None));
    let result_task = result.clone();
    executor.spawn(async move {
        let pitch_one = run_matrix_view_packet_roundtrip(1).await;
        let co_layout = run_matrix_view_packet_roundtrip(2).await;
        *result_task.lock().unwrap() = Some((pitch_one, co_layout));
    });
    executor.enter(Instant::ETERNITY).await;
    let ((row_values, row_stalls, row_cycles), (affine_values, affine_stalls, affine_cycles)) =
        result.lock().unwrap().take().unwrap();
    assert_eq!(row_values, affine_values);
    assert!(row_stalls > 0);
    assert_eq!(affine_stalls, 0);
    assert!(row_cycles > affine_cycles);
}

async fn run_matrix_accumulator_view_writeback(tile_pitch_rows: u32) -> (Vec<f32>, u64, u64) {
    assert_eq!((*MLEN, *BLEN, *VLEN), (64, 4, 64));
    let mram = Arc::new(MatrixSram::with_banks(
        *MLEN,
        (*MLEN as usize) * 64,
        *BLEN,
        *MATRIX_SRAM_TYPE,
    ));
    let vram = Arc::new(VectorSram::from_mx_type(*VLEN, 128, *VECTOR_SRAM_TYPE));

    let mut identity = vec![0.0_f32; (*MLEN * *MLEN) as usize];
    for index in 0..*MLEN as usize {
        identity[index * *MLEN as usize + index] = 1.0;
    }
    mram.write(
        0,
        QuantTensor::quantize(Tensor::from_slice(&identity), *MATRIX_SRAM_TYPE),
    )
    .await;

    let output_blocks = *VLEN / *BLEN;
    for tile in 0..output_blocks {
        for row in 0..*BLEN {
            let mut values = vec![0.0_f32; *VLEN as usize];
            if row == 0 {
                for column in 0..*BLEN {
                    values[column as usize] = (tile * *BLEN + column + 1) as f32;
                }
            }
            vram.write(
                (tile * *BLEN + row) * *VLEN,
                QuantTensor::quantize(Tensor::from_slice(&values), *VECTOR_SRAM_TYPE),
            )
            .await;
        }
    }

    let m_machine = MatrixMachine::new(mram, vram.clone(), *MLEN, *HLEN, *BLEN, *BROADCAST_AMOUNT);
    let v_machine = VectorMachine::new(vram.clone(), *VLEN, *HLEN);
    let hbm: Arc<dyn ErasedMemoryModel> = Arc::new(WithTiming::new(
        NaiveTiming::preset_ddr4_2400p(4),
        MemoryBacked::with_capacity(4096),
    ));
    let mut accelerator = Accelerator::new(m_machine, v_machine, hbm);

    // The producer writes sixteen BLEN-wide fragments. The consumer sees eight
    // logical heads with two fragments per head, so this catches the old but
    // incorrect 16x4 producer-only descriptor.
    let consumer_cols = 2 * *BLEN;
    let consumer_tiles = *VLEN / consumer_cols;
    let shape = (consumer_cols - 1) << 12 | ((consumer_tiles - 1) << 24);
    let mapping = tile_pitch_rows | (1 << 28);
    let matrix_output_base = *MLEN * *MLEN;
    let vector_output_base = output_blocks * *BLEN * *VLEN;
    let mut ops = Vec::new();
    // Leave a valid legacy producer stream on the same GP as the explicit
    // Matrix-view writeback. The explicit view must fully determine placement
    // and must not consume or advance this stale stream state.
    for (field, value) in [
        (0, 0),
        (2, matrix_output_base),
        (3, *VLEN),
        (4, *BLEN),
        (5, 1),
        (6, 1),
        (8, 1),
        (12, *VLEN),
        (13, *BLEN),
        (14, matrix_output_base / *VLEN),
        (1, 1 | 4 | 16 | 32 | 64),
    ] {
        stream_field(&mut ops, value, 4, 3, field);
    }
    ops.extend([
        set_gp(1, shape),
        set_gp(2, mapping),
        set_gp(3, 0),
        set_gp(4, matrix_output_base),
        set_gp(5, vector_output_base),
        set_gp(8, vector_output_base + *VLEN),
        op::Opcode::L_TILE_CFG {
            shape: 1,
            mapping: 2,
            slot: 0,
        },
        op::Opcode::L_TILE_CFG {
            shape: 1,
            mapping: 2,
            slot: 1,
        },
    ]);
    for tile in 0..output_blocks {
        ops.extend([
            set_gp(6, tile * *BLEN * *VLEN),
            set_gp(7, tile * *BLEN),
            op::Opcode::M_MM {
                rs1: 3,
                rs2: 6,
                view: None,
            },
            op::Opcode::M_MM_WO {
                rd: 4,
                rstride: 7,
                imm: 0,
                view: Some(0),
            },
        ]);
    }
    ops.push(op::Opcode::V_ADD_VV {
        rd: 5,
        rs1: 4,
        rs2: 8,
        rmask: 0,
        lmask: 0x8 | 0b010,
    });

    let start = Executor::current().now();
    accelerator.do_ops(&ops, None, TimingDriver::Serial).await;
    assert_eq!(
        accelerator.reg_file.read_gp_view(4, 1 << 3),
        matrix_output_base,
        "explicit Matrix-view writeback advanced stale legacy L_CFG state"
    );
    let cycles = (Executor::current().now() - start).as_picos() / PERIOD.as_picos();
    let counters = accelerator.matrix_view_packet_counters();
    (
        tensor_to_f32_vec(vram.read(vector_output_base).await.as_tensor()),
        counters.bank_stall_cycles,
        cycles,
    )
}

#[tokio::test]
async fn matrix_accumulator_writes_skewed_tiles_consumed_without_bank_conflicts() {
    set_timing_mode(TimingMode::Serial);
    let executor = Executor::new();
    let result = Arc::new(Mutex::new(None));
    let result_task = result.clone();
    executor.spawn(async move {
        let fixed = run_matrix_accumulator_view_writeback(1).await;
        let affine = run_matrix_accumulator_view_writeback(2).await;
        *result_task.lock().unwrap() = Some((fixed, affine));
    });
    executor.enter(Instant::ETERNITY).await;

    let ((fixed_values, fixed_stalls, fixed_cycles), (affine_values, affine_stalls, affine_cycles)) =
        result.lock().unwrap().take().unwrap();
    let expected = (1..=*VLEN).map(|value| value as f32).collect::<Vec<_>>();
    assert_eq!(fixed_values, expected);
    assert_eq!(affine_values, expected);
    assert!(fixed_stalls > 0);
    assert_eq!(affine_stalls, 0);
    assert!(fixed_cycles > affine_cycles);
}

async fn run_dispatched_packet_rank_update(
    alpha: u32,
    decays: Option<Vec<f32>>,
) -> (Vec<f32>, PacketCounterSnapshot, u64) {
    assert_eq!(*VLEN, 64);
    assert_eq!(*BLEN, 4);
    let rows = 16;
    let state_base = 0;
    let source_base = rows * *VLEN;
    let mram = Arc::new(MatrixSram::new(
        *MLEN,
        (*MLEN as usize) * 64,
        *MATRIX_SRAM_TYPE,
    ));
    let vram = Arc::new(VectorSram::from_mx_type_with_banks(
        *VLEN,
        64,
        *VECTOR_SRAM_TYPE,
        16,
    ));
    let logical_state: Vec<f32> = (0..rows)
        .flat_map(|row| (0..*VLEN).map(move |column| (row * 100 + column) as f32))
        .collect();
    if alpha == 0 {
        for row in 0..rows {
            let begin = (row * *VLEN) as usize;
            vram.write(
                state_base + row * *VLEN,
                QuantTensor::quantize(
                    Tensor::from_slice(&logical_state[begin..begin + *VLEN as usize]),
                    *VECTOR_SRAM_TYPE,
                ),
            )
            .await;
        }
    } else {
        let minor_steps = *VLEN / *BLEN;
        let mut compact = vec![vec![0.0_f32; *VLEN as usize]; minor_steps as usize];
        for row in 0..rows {
            for stripe in 0..minor_steps {
                let bank = (stripe + alpha * row) % rows;
                let logical_begin = (row * *VLEN + stripe * *BLEN) as usize;
                let physical_begin = (bank * *BLEN) as usize;
                compact[stripe as usize][physical_begin..physical_begin + *BLEN as usize]
                    .copy_from_slice(&logical_state[logical_begin..logical_begin + *BLEN as usize]);
            }
        }
        for (physical_row, values) in compact.iter().enumerate() {
            vram.write(
                physical_row as u32 * *VLEN,
                QuantTensor::quantize(Tensor::from_slice(values), *VECTOR_SRAM_TYPE),
            )
            .await;
        }
    }
    let source: Vec<f32> = (0..*VLEN).map(|column| (column + 1) as f32).collect();
    vram.write(
        source_base,
        QuantTensor::quantize(Tensor::from_slice(&source), *VECTOR_SRAM_TYPE),
    )
    .await;

    let m_machine = MatrixMachine::new(mram, vram.clone(), *MLEN, *HLEN, *BLEN, *BROADCAST_AMOUNT);
    let v_machine = VectorMachine::new(vram.clone(), *VLEN, *HLEN);
    let hbm: Arc<dyn ErasedMemoryModel> = Arc::new(WithTiming::new(
        NaiveTiming::preset_ddr4_2400p(4),
        MemoryBacked::with_capacity(4096),
    ));
    let mut accelerator = Accelerator::new(m_machine, v_machine, hbm);
    for row in 0..rows {
        accelerator
            .scalar_sram
            .write_fp(row as usize, bf16::from_f32((row + 1) as f32));
    }
    if let Some(decays) = &decays {
        assert_eq!(decays.len(), rows as usize);
        for (row, decay) in decays.iter().enumerate() {
            accelerator
                .scalar_sram
                .write_fp(rows as usize + row, bf16::from_f32(*decay));
        }
    }

    let mut ops = Vec::new();
    // The compact physical layout is an independent contract from affine
    // address generation.  Set both bits when the fixture is initialized in
    // major-packed form; otherwise the consumer would correctly interpret the
    // bytes as the regular affine layout and expose a fixture mismatch.
    let gp_flags = 1 | 16 | 32 | 64 | 128 | if alpha != 0 { 2 | 4 } else { 0 };
    if decays.is_some() {
        // state[row] *= decay[row] -- Mamba uses repeated values for one head,
        // while KDA supplies one value per key row. Both are the same generic
        // segmented-scalar packet at the ISA boundary.
        for (field, value) in [
            (0, 0),
            (2, state_base),
            (3, *VLEN),
            (4, rows),
            (5, 1),
            (6, 1),
            (8, alpha),
            (11, *BLEN),
            (12, *VLEN),
            (13, *BLEN),
            (14, state_base / *VLEN),
            (15, *VLEN),
            (1, gp_flags),
        ] {
            stream_field(&mut ops, value, 3, 0, field);
        }
        for (field, value) in [
            (0, 0),
            (2, rows),
            (3, *VLEN),
            (4, rows),
            (5, 1),
            (6, 1),
            (11, 0),
            (12, *VLEN),
            (13, *BLEN),
            (15, 1),
            (1, 1 | 8 | 32 | 64 | 128),
        ] {
            stream_field(&mut ops, value, 1, 1, field);
        }
        ops.extend([
            op::Opcode::C_LOOP_START { rd: 5, imm: rows },
            op::Opcode::V_MUL_VF {
                rd: 3,
                rs1: 3,
                rs2: 1,
                rmask: 0,
                lmask: 3,
            },
            op::Opcode::C_LOOP_END { rd: 5 },
        ]);
    }
    for (field, value) in [
        (0, 0),
        (2, state_base),
        (3, *VLEN),
        (4, rows),
        (5, 1),
        (6, 1),
        (8, alpha),
        (11, *BLEN),
        (12, *VLEN),
        (13, *BLEN),
        (14, state_base / *VLEN),
        (15, *VLEN),
        (1, gp_flags),
    ] {
        stream_field(&mut ops, value, 3, 0, field);
    }
    for (field, value) in [
        (0, 0),
        (2, source_base),
        (3, *VLEN),
        (4, rows),
        (5, 1),
        (6, 1),
        (11, *BLEN),
        (12, *VLEN),
        (13, *BLEN),
        (14, source_base / *VLEN),
        (15, 0),
        (1, 1 | 32 | 64 | 128),
    ] {
        stream_field(&mut ops, value, 4, 1, field);
    }
    for (field, value) in [
        (0, 0),
        (2, 0),
        (3, *VLEN),
        (4, rows),
        (5, 1),
        (6, 1),
        (11, 0),
        (12, *VLEN),
        (13, *BLEN),
        (15, 1),
        (1, 1 | 8 | 32 | 64 | 128),
    ] {
        stream_field(&mut ops, value, 1, 2, field);
    }
    ops.extend([
        op::Opcode::C_LOOP_START { rd: 5, imm: rows },
        op::Opcode::V_FMA_VF {
            rd: 3,
            rs1: 4,
            rs2: 1,
            rmask: 0,
            lmask: 7,
        },
        op::Opcode::C_LOOP_END { rd: 5 },
    ]);

    let start = Executor::current().now();
    accelerator.do_ops(&ops, None, TimingDriver::Serial).await;
    let elapsed = (Executor::current().now() - start).as_picos();
    let counters = accelerator.lstream_packet_counters();
    let mut output = vec![0.0_f32; logical_state.len()];
    if alpha == 0 {
        for row in 0..rows {
            let physical = tensor_to_f32_vec(vram.read(row * *VLEN).await.as_tensor());
            let begin = (row * *VLEN) as usize;
            output[begin..begin + *VLEN as usize].copy_from_slice(&physical);
        }
    } else {
        let minor_steps = *VLEN / *BLEN;
        let mut physical = Vec::with_capacity(minor_steps as usize);
        for row in 0..minor_steps {
            physical.push(tensor_to_f32_vec(vram.read(row * *VLEN).await.as_tensor()));
        }
        for row in 0..rows {
            for stripe in 0..minor_steps {
                let bank = (stripe + alpha * row) % rows;
                let logical_begin = (row * *VLEN + stripe * *BLEN) as usize;
                let physical_begin = (bank * *BLEN) as usize;
                output[logical_begin..logical_begin + *BLEN as usize].copy_from_slice(
                    &physical[stripe as usize][physical_begin..physical_begin + *BLEN as usize],
                );
            }
        }
    }
    (output, counters, elapsed)
}

#[tokio::test]
async fn l_cfg_dispatches_conflict_free_affine_packets_into_existing_fma() {
    // This is an Accelerator dispatch test, not a direct VectorMachine call.
    // Distinct destination/source contents make a swapped rd/rs1 argument
    // produce a different tensor, so it also pins the V_FMA_VF dispatch order.
    set_timing_mode(TimingMode::Serial);
    let executor = Executor::new();
    let result = Arc::new(Mutex::new(None));
    let result_task = result.clone();
    executor.spawn(async move {
        let row = run_dispatched_packet_rank_update(0, None).await;
        let affine = run_dispatched_packet_rank_update(1, None).await;
        *result_task.lock().unwrap() = Some((row, affine));
    });
    executor.enter(Instant::ETERNITY).await;

    let ((row_values, row_count, row_time), (affine_values, affine_count, affine_time)) =
        result.lock().unwrap().take().unwrap();
    assert_eq!(row_values, affine_values);
    assert!(row_count.conflict_stall_cycles > 0);
    assert_eq!(affine_count.conflict_stall_cycles, 0);
    assert!(row_time > affine_time);
    assert_eq!(row_count.read_packets, 32);
    assert_eq!(row_count.write_packets, 16);
    // Each destination packet touches 16 logical rows. Row-major sends all
    // 16 bank words to one bank (8 cycles at 2R, 16 cycles at 1W), while the
    // affine alpha=1 placement sends one word to every bank. The pinned source
    // is one deduplicated word and therefore adds no conflict stall.
    assert_eq!(row_count.service_cycles, 400);
    assert_eq!(row_count.bandwidth_floor_cycles, 48);
    assert_eq!(row_count.conflict_stall_cycles, 352);
    assert_eq!(affine_count.service_cycles, 48);
    assert_eq!(affine_count.bandwidth_floor_cycles, 48);
    assert_eq!(affine_count.conflict_stall_cycles, 0);
}

async fn run_paper_width_dispatched_rank_update(
    alpha: u32,
) -> (Vec<f32>, PacketCounterSnapshot, u64) {
    const PACKET: u32 = 2048;
    const ROW_ELEMENTS: u32 = 64;
    const ROWS: u32 = PACKET / ROW_ELEMENTS;
    const BANKS: u32 = 32;
    let state_base = 0;
    let source_base = ROWS * ROW_ELEMENTS;
    let source_physical_row = ROWS;
    let mram = Arc::new(MatrixSram::new(
        *MLEN,
        (*MLEN as usize) * 64,
        *MATRIX_SRAM_TYPE,
    ));
    let vram = Arc::new(VectorSram::from_mx_type_with_banks(
        PACKET,
        64,
        *VECTOR_SRAM_TYPE,
        BANKS,
    ));
    let mut initial = Vec::with_capacity(PACKET as usize);
    let mut compact_physical = vec![0.0_f32; PACKET as usize];
    for row in 0..ROWS {
        let logical: Vec<f32> = (0..ROW_ELEMENTS)
            .map(|column| (row * 100 + column) as f32)
            .collect();
        initial.extend_from_slice(&logical);
        let bank = (alpha * row) % BANKS;
        let begin = (bank * ROW_ELEMENTS) as usize;
        if alpha == 0 {
            let mut physical = vec![0.0_f32; PACKET as usize];
            physical[begin..begin + ROW_ELEMENTS as usize].copy_from_slice(&logical);
            vram.write(
                row * PACKET,
                QuantTensor::quantize(Tensor::from_slice(&physical), *VECTOR_SRAM_TYPE),
            )
            .await;
        } else {
            compact_physical[begin..begin + ROW_ELEMENTS as usize].copy_from_slice(&logical);
        }
    }
    if alpha != 0 {
        vram.write(
            0,
            QuantTensor::quantize(Tensor::from_slice(&compact_physical), *VECTOR_SRAM_TYPE),
        )
        .await;
    }
    let source: Vec<f32> = (0..ROW_ELEMENTS)
        .map(|column| (column + 1) as f32)
        .collect();
    let mut source_physical = vec![0.0_f32; PACKET as usize];
    source_physical[..ROW_ELEMENTS as usize].copy_from_slice(&source);
    vram.write(
        source_physical_row * PACKET,
        QuantTensor::quantize(Tensor::from_slice(&source_physical), *VECTOR_SRAM_TYPE),
    )
    .await;

    let m_machine = MatrixMachine::new(mram, vram.clone(), *MLEN, *HLEN, *BLEN, *BROADCAST_AMOUNT);
    let v_machine = VectorMachine::new(vram.clone(), PACKET, ROW_ELEMENTS);
    let hbm: Arc<dyn ErasedMemoryModel> = Arc::new(WithTiming::new(
        NaiveTiming::preset_ddr4_2400p(4),
        MemoryBacked::with_capacity(4096),
    ));
    let mut accelerator = Accelerator::new(m_machine, v_machine, hbm);
    for row in 0..ROWS {
        accelerator
            .scalar_sram
            .write_fp(row as usize, bf16::from_f32((row + 1) as f32 / 32.0));
    }

    let mut ops = Vec::new();
    let destination_flags = 1 | 16 | 32 | 64 | 128 | if alpha != 0 { 2 | 4 } else { 0 };
    for (field, value) in [
        (0, 0),
        (2, state_base),
        (3, ROW_ELEMENTS),
        (4, ROWS),
        (5, 1),
        (6, 1),
        (8, alpha),
        (11, ROW_ELEMENTS),
        (12, PACKET),
        (13, ROW_ELEMENTS),
        (14, 0),
        (15, ROW_ELEMENTS),
        (1, destination_flags),
    ] {
        stream_field(&mut ops, value, 3, 0, field);
    }
    for (field, value) in [
        (0, 0),
        (2, source_base),
        (3, ROW_ELEMENTS),
        (4, ROWS),
        (5, 1),
        (6, 1),
        (11, 0),
        (12, PACKET),
        (13, ROW_ELEMENTS),
        (14, source_physical_row),
        (15, 0),
        (1, 1 | 32 | 64 | 128),
    ] {
        stream_field(&mut ops, value, 4, 1, field);
    }
    for (field, value) in [
        (0, 0),
        (2, 0),
        (3, ROW_ELEMENTS),
        (4, ROWS),
        (5, 1),
        (6, 1),
        (11, 0),
        (12, PACKET),
        (13, ROW_ELEMENTS),
        (15, 1),
        (1, 1 | 8 | 32 | 64 | 128),
    ] {
        stream_field(&mut ops, value, 1, 2, field);
    }
    ops.push(op::Opcode::V_FMA_VF {
        rd: 3,
        rs1: 4,
        rs2: 1,
        rmask: 0,
        lmask: 7,
    });

    let start = Executor::current().now();
    accelerator.do_ops(&ops, None, TimingDriver::Serial).await;
    let elapsed = (Executor::current().now() - start).as_picos();
    let counters = accelerator.lstream_packet_counters();
    let mut output = Vec::with_capacity(PACKET as usize);
    for row in 0..ROWS {
        let physical_row = if alpha == 0 { row } else { 0 };
        let physical = tensor_to_f32_vec(vram.read(physical_row * PACKET).await.as_tensor());
        let bank = (alpha * row) % BANKS;
        let begin = (bank * ROW_ELEMENTS) as usize;
        output.extend_from_slice(&physical[begin..begin + ROW_ELEMENTS as usize]);
    }
    let expected: Vec<f32> = initial
        .chunks_exact(ROW_ELEMENTS as usize)
        .enumerate()
        .flat_map(|(row, values)| {
            let scalar = bf16::from_f32((row + 1) as f32 / 32.0).to_f32();
            values.iter().zip(&source).map(move |(state, update)| {
                bf16::from_f32(
                    bf16::from_f32(*state).to_f32() + bf16::from_f32(*update).to_f32() * scalar,
                )
                .to_f32()
            })
        })
        .collect();
    assert_eq!(output, expected);
    (output, counters, elapsed)
}

#[tokio::test]
async fn paper_2048_l_cfg_dispatches_short_rows_without_conflicts() {
    set_timing_mode(TimingMode::Serial);
    let executor = Executor::new();
    let result = Arc::new(Mutex::new(None));
    let result_task = result.clone();
    executor.spawn(async move {
        let row = run_paper_width_dispatched_rank_update(0).await;
        let affine = run_paper_width_dispatched_rank_update(1).await;
        *result_task.lock().unwrap() = Some((row, affine));
    });
    executor.enter(Instant::ETERNITY).await;

    let ((row_values, row_count, row_time), (affine_values, affine_count, affine_time)) =
        result.lock().unwrap().take().unwrap();
    assert_eq!(row_values, affine_values);
    assert_eq!(row_count.read_packets, 2);
    assert_eq!(row_count.write_packets, 1);
    assert_eq!(row_count.conflict_stall_cycles, 46);
    assert_eq!(affine_count.conflict_stall_cycles, 0);
    assert!(row_time > affine_time);
}

#[tokio::test]
async fn paper_2048_kda_packets_advance_scalars_across_two_atom_rows() {
    const PACKET: u32 = 2048;
    const ROW_ELEMENTS: u32 = 128;
    const ROWS: u32 = 96;
    const ATOM: u32 = 64;
    const BANKS: u32 = 32;
    const PACKETS: u32 = 6;
    set_timing_mode(TimingMode::Serial);
    let executor = Executor::new();
    let result = Arc::new(Mutex::new(None));
    let result_task = result.clone();
    executor.spawn(async move {
        let state_base = 0;
        let source_base = ROWS * ROW_ELEMENTS;
        let source_physical_row = PACKETS;
        let mram = Arc::new(MatrixSram::new(
            *MLEN,
            (*MLEN as usize) * 64,
            *MATRIX_SRAM_TYPE,
        ));
        let vram = Arc::new(VectorSram::from_mx_type_with_banks(
            PACKET,
            16,
            *VECTOR_SRAM_TYPE,
            BANKS,
        ));

        let state: Vec<f32> = (0..ROWS * ROW_ELEMENTS)
            .map(|index| (index % 251) as f32 / 16.0)
            .collect();
        let source: Vec<f32> = (0..ROW_ELEMENTS)
            .map(|index| (index + 1) as f32 / 128.0)
            .collect();
        let mut physical_state = vec![vec![0.0_f32; PACKET as usize]; PACKETS as usize];
        for row in 0..ROWS {
            for stripe in 0..ROW_ELEMENTS / ATOM {
                let bank = (row + stripe) % BANKS;
                let bank_row = (row / BANKS) * (ROW_ELEMENTS / ATOM) + stripe;
                let logical_begin = (row * ROW_ELEMENTS + stripe * ATOM) as usize;
                let physical_begin = (bank * ATOM) as usize;
                physical_state[bank_row as usize][physical_begin..physical_begin + ATOM as usize]
                    .copy_from_slice(&state[logical_begin..logical_begin + ATOM as usize]);
            }
        }
        for (row, values) in physical_state.iter().enumerate() {
            vram.write(
                row as u32 * PACKET,
                QuantTensor::quantize(Tensor::from_slice(values), *VECTOR_SRAM_TYPE),
            )
            .await;
        }
        let mut physical_source = vec![0.0_f32; PACKET as usize];
        physical_source[..ROW_ELEMENTS as usize].copy_from_slice(&source);
        vram.write(
            source_physical_row * PACKET,
            QuantTensor::quantize(Tensor::from_slice(&physical_source), *VECTOR_SRAM_TYPE),
        )
        .await;

        let m_machine =
            MatrixMachine::new(mram, vram.clone(), *MLEN, *HLEN, *BLEN, *BROADCAST_AMOUNT);
        let v_machine = VectorMachine::new(vram.clone(), PACKET, ROW_ELEMENTS);
        let hbm: Arc<dyn ErasedMemoryModel> = Arc::new(WithTiming::new(
            NaiveTiming::preset_ddr4_2400p(4),
            MemoryBacked::with_capacity(4096),
        ));
        let mut accelerator = Accelerator::new(m_machine, v_machine, hbm);
        for row in 0..ROWS {
            accelerator
                .scalar_sram
                .write_fp(row as usize, bf16::from_f32((row + 1) as f32 / 64.0));
        }

        let mut ops = Vec::new();
        for (field, value) in [
            (0, 0),
            (2, state_base),
            (3, ROW_ELEMENTS),
            (4, ROWS),
            (5, 1),
            (6, 1),
            (8, 1),
            (11, ATOM),
            (12, PACKET),
            (13, ATOM),
            (14, 0),
            (15, ROW_ELEMENTS),
            (1, 1 | 2 | 4 | 16 | 32 | 64 | 128),
        ] {
            stream_field(&mut ops, value, 3, 0, field);
        }
        for (field, value) in [
            (0, 0),
            (2, source_base),
            (3, ROW_ELEMENTS),
            (4, ROWS),
            (5, 1),
            (6, 1),
            (11, ATOM),
            (12, PACKET),
            (13, ATOM),
            (14, source_physical_row),
            (15, 0),
            (1, 1 | 32 | 64 | 128),
        ] {
            stream_field(&mut ops, value, 4, 1, field);
        }
        for (field, value) in [
            (0, 0),
            (2, 0),
            (3, ROW_ELEMENTS),
            (4, ROWS),
            (5, 1),
            (6, 1),
            (11, 0),
            (12, PACKET),
            (13, ATOM),
            (15, 1),
            (1, 1 | 8 | 32 | 64 | 128),
        ] {
            stream_field(&mut ops, value, 1, 2, field);
        }
        ops.extend([
            op::Opcode::C_LOOP_START {
                rd: 5,
                imm: PACKETS,
            },
            op::Opcode::V_FMA_VF {
                rd: 3,
                rs1: 4,
                rs2: 1,
                rmask: 0,
                lmask: 7,
            },
            op::Opcode::C_LOOP_END { rd: 5 },
        ]);
        accelerator.do_ops(&ops, None, TimingDriver::Serial).await;

        let mut output = vec![0.0_f32; state.len()];
        for row in 0..ROWS {
            for stripe in 0..ROW_ELEMENTS / ATOM {
                let bank = (row + stripe) % BANKS;
                let bank_row = (row / BANKS) * (ROW_ELEMENTS / ATOM) + stripe;
                let physical = tensor_to_f32_vec(vram.read(bank_row * PACKET).await.as_tensor());
                let logical_begin = (row * ROW_ELEMENTS + stripe * ATOM) as usize;
                let physical_begin = (bank * ATOM) as usize;
                output[logical_begin..logical_begin + ATOM as usize]
                    .copy_from_slice(&physical[physical_begin..physical_begin + ATOM as usize]);
            }
        }
        let expected: Vec<f32> = state
            .chunks_exact(ROW_ELEMENTS as usize)
            .enumerate()
            .flat_map(|(row, values)| {
                let scalar = bf16::from_f32((row + 1) as f32 / 64.0).to_f32();
                values.iter().zip(&source).map(move |(state, update)| {
                    bf16::from_f32(
                        bf16::from_f32(*state).to_f32() + bf16::from_f32(*update).to_f32() * scalar,
                    )
                    .to_f32()
                })
            })
            .collect();
        *result_task.lock().unwrap() =
            Some((output, expected, accelerator.lstream_packet_counters()));
    });
    executor.enter(Instant::ETERNITY).await;

    let (output, expected, counters) = result.lock().unwrap().take().unwrap();
    assert_eq!(output, expected);
    assert_eq!(counters.conflict_stall_cycles, 0);
    assert_eq!(counters.write_packets, PACKETS as u64);
}

fn packet_state_step_golden(decays: &[f32]) -> Vec<f32> {
    let mut expected = Vec::with_capacity(16 * *VLEN as usize);
    for row in 0..16_u32 {
        let decay = bf16::from_f32(decays[row as usize]).to_f32();
        let update = bf16::from_f32((row + 1) as f32).to_f32();
        for column in 0..*VLEN {
            let state = bf16::from_f32((row * 100 + column) as f32).to_f32();
            let source = bf16::from_f32((column + 1) as f32).to_f32();
            let decayed = bf16::from_f32(state * decay).to_f32();
            expected.push(bf16::from_f32(decayed + source * update).to_f32());
        }
    }
    expected
}

#[tokio::test]
async fn mamba_and_kda_state_steps_execute_through_conflict_free_packets() {
    set_timing_mode(TimingMode::Serial);
    let executor = Executor::new();
    let result = Arc::new(Mutex::new(None));
    let result_task = result.clone();
    executor.spawn(async move {
        let cases = [
            ("mamba", vec![0.75; 16]),
            ("kda", (0..16).map(|row| 0.50 + row as f32 / 64.0).collect()),
        ];
        let mut outputs = Vec::new();
        for (name, decays) in cases {
            let row = run_dispatched_packet_rank_update(0, Some(decays.clone())).await;
            let affine = run_dispatched_packet_rank_update(1, Some(decays.clone())).await;
            outputs.push((name, decays, row, affine));
        }
        *result_task.lock().unwrap() = Some(outputs);
    });
    executor.enter(Instant::ETERNITY).await;

    for (name, decays, row, affine) in result.lock().unwrap().take().unwrap() {
        let (row_values, row_count, row_time) = row;
        let (affine_values, affine_count, affine_time) = affine;
        assert_eq!(row_values, packet_state_step_golden(&decays), "{name}");
        assert_eq!(affine_values, row_values, "{name}");
        assert_eq!(row_count.read_packets, 48, "{name}");
        assert_eq!(row_count.write_packets, 32, "{name}");
        assert_eq!(row_count.service_cycles, 784, "{name}");
        assert_eq!(row_count.bandwidth_floor_cycles, 80, "{name}");
        assert_eq!(row_count.conflict_stall_cycles, 704, "{name}");
        assert_eq!(affine_count.service_cycles, 80, "{name}");
        assert_eq!(affine_count.bandwidth_floor_cycles, 80, "{name}");
        assert_eq!(affine_count.conflict_stall_cycles, 0, "{name}");
        assert!(row_time > affine_time, "{name}");
    }
}

#[tokio::test]
async fn lstream_executes_a_loop_without_explicit_pointer_or_scalar_loads() {
    let executor = Executor::new();
    executor.spawn(async move {
        let mram = Arc::new(MatrixSram::new(
            *MLEN,
            (*MLEN as usize) * 64,
            *MATRIX_SRAM_TYPE,
        ));
        let vram = Arc::new(VectorSram::from_mx_type(*VLEN, 16, *VECTOR_SRAM_TYPE));
        let m_machine =
            MatrixMachine::new(mram, vram.clone(), *MLEN, *HLEN, *BLEN, *BROADCAST_AMOUNT);
        let v_machine = VectorMachine::new(vram, *VLEN, *HLEN);
        let hbm: Arc<dyn ErasedMemoryModel> = Arc::new(WithTiming::new(
            NaiveTiming::preset_ddr4_2400p(4),
            MemoryBacked::with_capacity(4096),
        ));
        let mut accelerator = Accelerator::new(m_machine, v_machine, hbm);

        let mut ops = vec![];
        // Slot 0: GP3 supplies two consecutive Vector-SRAM rows.
        for (field, value) in [
            (0, 0), // reset
            (2, 0), // base
            (3, *VLEN),
            // Keep one packet beyond the two loop iterations so the ordinary
            // register-read API can inspect the post-loop cursor. Exact
            // exhaustion is covered by strict_bounds_trap_before_an_extra_packet.
            (4, 3),
            (5, 1),
            (6, 1),
            (11, *VLEN),
            (12, *VLEN),
            (13, 4),
            (1, 1 | 64), // enable | strict bounds
        ] {
            stream_field(&mut ops, value, 3, 0, field);
        }
        // Slot 1: FP1 is hydrated from consecutive scalar-SRAM entries.
        for (field, value) in [
            (0, 0),
            (2, 0),
            (3, 3),
            (4, 1),
            (5, 1),
            (6, 1),
            (11, 1),
            (12, 1),
            (13, 1),
            (1, 1 | 8 | 64), // target is FP
        ] {
            stream_field(&mut ops, value, 1, 1, field);
        }
        ops.extend([
            op::Opcode::C_LOOP_START { rd: 4, imm: 2 },
            op::Opcode::V_FMA_VF {
                rd: 3,
                rs1: 3,
                rs2: 1,
                rmask: 0,
                lmask: 3,
            },
            op::Opcode::C_LOOP_END { rd: 4 },
        ]);

        accelerator.do_ops(&ops, None, TimingDriver::Serial).await;
        assert_eq!(accelerator.reg_file.read_gp(3), 0);
        assert_eq!(accelerator.reg_file.read_gp_view(3, 3), 2 * *VLEN);
        assert_eq!(accelerator.reg_file.lstream_fp_address(3, 1), Some(2));
        // The loop body contains no S_ADDI_INT and no S_LD_FP. Setup is outside
        // the loop and amortizes across the repeated arithmetic operation.
        assert!(matches!(ops[ops.len() - 2], op::Opcode::V_FMA_VF { .. }));
    });
    executor.enter(Instant::ETERNITY).await;
}

#[tokio::test]
async fn matrix_affine_writeback_roundtrips_through_streamed_vector_consumption() {
    let executor = Executor::new();
    executor.spawn(async move {
        assert_eq!(*VLEN, *MLEN);
        assert_eq!(*VLEN / *BLEN, 16);
        let mram = Arc::new(MatrixSram::new(
            *MLEN,
            (*MLEN as usize) * 64,
            *MATRIX_SRAM_TYPE,
        ));
        let vram = Arc::new(VectorSram::from_mx_type_with_banks(
            *VLEN,
            64,
            *VECTOR_SRAM_TYPE,
            16,
        ));

        let mut identity = vec![0.0_f32; (*MLEN * *MLEN) as usize];
        for lane in 0..*MLEN as usize {
            identity[lane * *MLEN as usize + lane] = 1.0;
        }
        mram.write(
            0,
            QuantTensor::quantize(Tensor::from_slice(&identity), *MATRIX_SRAM_TYPE),
        )
        .await;
        for row in 0..*BLEN {
            let mut values = vec![0.0_f32; *VLEN as usize];
            for (column, value) in values.iter_mut().take(*BLEN as usize).enumerate() {
                *value = (row * 10 + column as u32 + 1) as f32;
            }
            vram.write(
                row * *VLEN,
                QuantTensor::quantize(Tensor::from_slice(&values), *VECTOR_SRAM_TYPE),
            )
            .await;
        }

        let m_machine =
            MatrixMachine::new(mram, vram.clone(), *MLEN, *HLEN, *BLEN, *BROADCAST_AMOUNT);
        let v_machine = VectorMachine::new(vram.clone(), *VLEN, *HLEN);
        let hbm: Arc<dyn ErasedMemoryModel> = Arc::new(WithTiming::new(
            NaiveTiming::preset_ddr4_2400p(4),
            MemoryBacked::with_capacity(4096),
        ));
        let mut accelerator = Accelerator::new(m_machine, v_machine, hbm);
        accelerator.reg_file.write_fp(1, bf16::ONE);

        let physical_base = 8 * *VLEN;
        let logical_output = 16 * *VLEN;
        let mut ops = Vec::new();
        // Matrix producer: four output rows are skewed by alpha=1.  One BLEN
        // block is exactly one of the sixteen physical banks.
        for (field, value) in [
            (0, 0),
            (2, physical_base),
            (3, *VLEN),
            (4, *BLEN),
            (5, 1),
            (6, 1),
            (8, 1),
            (12, *VLEN),
            (13, *BLEN),
            (14, physical_base / *VLEN),
            (1, 1 | 4 | 16 | 32 | 64),
        ] {
            stream_field(&mut ops, value, 3, 3, field);
        }
        ops.extend([
            set_gp(1, 0),
            set_gp(2, 0),
            set_gp(3, physical_base),
            op::Opcode::M_MM {
                rs1: 1,
                rs2: 2,
                view: None,
            },
            op::Opcode::M_MM_WO {
                rd: 3,
                rstride: 0,
                imm: 0,
                view: None,
            },
        ]);

        // Vector consumer sees the same affine tensor through GP4.  GP5 is an
        // identity output stream, so the loop leaves a conventional row-major
        // result that can be checked directly.
        for (field, value) in [
            (0, 0),
            (2, physical_base),
            (3, *VLEN),
            (4, *BLEN),
            (5, 1),
            (6, 1),
            (8, 1),
            (11, *VLEN),
            (12, *VLEN),
            (13, *BLEN),
            (14, physical_base / *VLEN),
            (1, 1 | 4 | 32 | 64),
        ] {
            stream_field(&mut ops, value, 4, 1, field);
        }
        for (field, value) in [
            (0, 0),
            (2, logical_output),
            (3, *VLEN),
            (4, *BLEN),
            (5, 1),
            (6, 1),
            (11, *VLEN),
            (12, *VLEN),
            (13, *BLEN),
            (1, 1 | 64),
        ] {
            stream_field(&mut ops, value, 5, 2, field);
        }
        ops.extend([
            op::Opcode::C_LOOP_START { rd: 6, imm: *BLEN },
            op::Opcode::V_FMA_VF {
                rd: 5,
                rs1: 4,
                rs2: 1,
                rmask: 0,
                lmask: 6,
            },
            op::Opcode::C_LOOP_END { rd: 6 },
        ]);

        accelerator.do_ops(&ops, None, TimingDriver::Serial).await;

        for row in 0..*BLEN {
            let physical = vram.read(physical_base + row * *VLEN).await;
            let physical_values = tensor_to_f32_vec(physical.as_tensor());
            let physical_column = (row * *BLEN) as usize;
            assert_eq!(
                &physical_values[physical_column..physical_column + *BLEN as usize],
                &[
                    (row * 10 + 1) as f32,
                    (row * 10 + 2) as f32,
                    (row * 10 + 3) as f32,
                    (row * 10 + 4) as f32,
                ]
            );

            let restored = vram.read(logical_output + row * *VLEN).await;
            let restored_values = tensor_to_f32_vec(restored.as_tensor());
            assert_eq!(
                &restored_values[..*BLEN as usize],
                &[
                    (row * 10 + 1) as f32,
                    (row * 10 + 2) as f32,
                    (row * 10 + 3) as f32,
                    (row * 10 + 4) as f32,
                ]
            );
        }
    });
    executor.enter(Instant::ETERNITY).await;
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
    let mut ops = vec![op::Opcode::M_MM {
        rs1: 1,
        rs2: 2,
        view: None,
    }];
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
            lmask: 0,
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
            lmask: 0,
        });
    }
    if dependent {
        // Reads the prefetched row 0 (gp4 = 0), writes the far row.
        ops.push(op::Opcode::V_ADD_VV {
            rd: 5,
            rs1: 4,
            rs2: 4,
            rmask: 0,
            lmask: 0,
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
            lmask: 0,
        },
    ];
    for _ in 0..independent {
        ops.push(op::Opcode::V_ADD_VV {
            rd: 5,
            rs1: 5,
            rs2: 5,
            rmask: 0,
            lmask: 0,
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
        op::Opcode::M_MM {
            rs1: 1,
            rs2: 2,
            view: None,
        },
        op::Opcode::M_MM {
            rs1: 1,
            rs2: 2,
            view: None,
        },
        op::Opcode::M_MM_WO {
            rd: 1,
            rstride: 0,
            imm: 0,
            view: None,
        },
    ];
    let pipelined = run_program(ops, RunMode::Scoreboard, vec![]).await;
    let matrix_latency = (*SYSTOLIC_PROCESSING_OVERHEAD + *MLEN) as u64;
    // Two accumulates serialize on the (single) systolic array; the write-out
    // costs its captured latency (compute(1)) after them.
    assert_eq!(cycles_of(pipelined.now), 2 * matrix_latency + 1);
}
