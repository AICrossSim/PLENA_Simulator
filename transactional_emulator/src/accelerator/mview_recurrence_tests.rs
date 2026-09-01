//! Numerical recurrence tests that consume real multi-tile Matrix-SRAM packets.
//!
//! The ISA remains model independent: Matrix views only place and restore data;
//! every arithmetic step below is an existing Vector opcode.  Row-major and
//! affine runs execute the same operation stream and differ only in `alpha`.

use std::sync::{Arc, Mutex};

use memory::{ErasedMemoryModel, MemoryBacked, NaiveTiming, WithTiming};
use quantize::{QuantTensor, tensor_to_f32_vec};
use runtime::{Executor, Instant};
use sram::matrix::{MatrixLayout, MatrixPacketCounterSnapshot};
use sram::{MatrixSram, VectorSram};
use tch::Tensor;

use super::{Accelerator, TimingDriver};
use crate::matrix_machine::MatrixMachine;
use crate::op;
use crate::runtime_config::{
    BLEN, BROADCAST_AMOUNT, HLEN, MATRIX_SRAM_TYPE, MLEN, PERIOD, VECTOR_SRAM_TYPE, VLEN,
};
use crate::timing::{TimingMode, set_timing_mode};
use crate::vector_machine::{PacketCounterSnapshot, VectorMachine};

const HEADS: usize = 16;
const WIDTH: usize = 4;
const STATE_AXES: usize = 2;
const TOKENS: usize = 4;

#[derive(Debug)]
struct RecurrenceResult {
    output: Vec<f32>,
    state: Vec<f32>,
    matrix: MatrixPacketCounterSnapshot,
    vector: PacketCounterSnapshot,
    cycles: u64,
}

fn set_gp(rd: u8, value: u32) -> op::Opcode {
    op::Opcode::S_ADDI_INT {
        rd,
        rs1: 0,
        imm: value,
    }
}

fn add(rd: u8, rs1: u8, rs2: u8) -> op::Opcode {
    op::Opcode::V_ADD_VV {
        rd,
        rs1,
        rs2,
        rmask: 0,
        lmask: 0,
    }
}

fn sub(rd: u8, rs1: u8, rs2: u8) -> op::Opcode {
    op::Opcode::V_SUB_VV {
        rd,
        rs1,
        rs2,
        rmask: 0,
        lmask: 0,
    }
}

fn mul(rd: u8, rs1: u8, rs2: u8) -> op::Opcode {
    op::Opcode::V_MUL_VV {
        rd,
        rs1,
        rs2,
        rmask: 0,
        lmask: 0,
    }
}

fn add_mv(rd: u8, rs1: u8, rs2: u8, operand_mask: u8) -> op::Opcode {
    assert!((1..=7).contains(&operand_mask));
    op::Opcode::V_ADD_VV {
        rd,
        rs1,
        rs2,
        rmask: 0,
        lmask: 0x8 | operand_mask,
    }
}

fn mul_mv(rd: u8, rs1: u8, rs2: u8, operand_mask: u8) -> op::Opcode {
    assert!((1..=7).contains(&operand_mask));
    op::Opcode::V_MUL_VV {
        rd,
        rs1,
        rs2,
        rmask: 0,
        lmask: 0x8 | operand_mask,
    }
}

fn packet_layout(alpha: u32) -> MatrixLayout {
    MatrixLayout {
        rows: 1,
        cols: WIDTH as u32,
        tile_count: HEADS as u32,
        tile_pitch_rows: 1,
        alpha,
    }
}

fn packet_shape_word() -> u32 {
    ((WIDTH as u32 - 1) << 12) | ((HEADS as u32 - 1) << 24)
}

fn packet_map_word(alpha: u32) -> u32 {
    1 | (alpha << 16) | (1 << 28)
}

fn state_base(axis: usize) -> u32 {
    packet_base(axis)
}

fn packet_base(packet: usize) -> u32 {
    packet as u32 * HEADS as u32 * *MLEN
}

async fn write_vector(vram: &VectorSram, row: u32, values: &[f32]) {
    assert_eq!(values.len(), *VLEN as usize);
    vram.write(
        row * *VLEN,
        QuantTensor::quantize(Tensor::from_slice(values), *VECTOR_SRAM_TYPE),
    )
    .await;
}

async fn write_state_packet(mram: &MatrixSram, layout: MatrixLayout, axis: usize, values: &[f32]) {
    write_matrix_packet(mram, layout, axis, values).await;
}

async fn write_matrix_packet(
    mram: &MatrixSram,
    layout: MatrixLayout,
    packet: usize,
    values: &[f32],
) {
    mram.write_layout_packet(
        packet_base(packet),
        layout,
        QuantTensor::quantize(Tensor::from_slice(values), *MATRIX_SRAM_TYPE),
    )
    .await;
}

fn repeated_by_head<F>(mut value: F) -> Vec<f32>
where
    F: FnMut(usize) -> f32,
{
    (0..HEADS)
        .flat_map(|head| std::iter::repeat_n(value(head), WIDTH))
        .collect()
}

fn initial_state() -> Vec<Vec<f32>> {
    (0..STATE_AXES)
        .map(|axis| {
            (0..HEADS)
                .flat_map(|head| {
                    (0..WIDTH).map(move |lane| {
                        (head % 4 + 1) as f32 + axis as f32 * 0.5 + lane as f32 * 0.25
                    })
                })
                .collect()
        })
        .collect()
}

fn flatten_state(state: &[Vec<f32>]) -> Vec<f32> {
    state.iter().flatten().copied().collect()
}

async fn new_accelerator(mram: Arc<MatrixSram>, vram: Arc<VectorSram>) -> Accelerator {
    let m_machine = MatrixMachine::new(mram, vram.clone(), *MLEN, *HLEN, *BLEN, *BROADCAST_AMOUNT);
    let v_machine = VectorMachine::new(vram, *VLEN, *HLEN);
    let hbm: Arc<dyn ErasedMemoryModel> = Arc::new(WithTiming::new(
        NaiveTiming::preset_ddr4_2400p(4),
        MemoryBacked::with_capacity(4096),
    ));
    Accelerator::new(m_machine, v_machine, hbm)
}

async fn read_state(mram: &MatrixSram, layout: MatrixLayout) -> Vec<f32> {
    let mut result = Vec::new();
    for axis in 0..STATE_AXES {
        let (packet, _) = mram.read_layout_packet(state_base(axis), layout).await;
        result.extend(tensor_to_f32_vec(packet.as_tensor()));
    }
    result
}

async fn run_kda(alpha: u32) -> RecurrenceResult {
    assert_eq!((*MLEN, *BLEN, *VLEN), (64, 4, 64));
    let layout = packet_layout(alpha);
    let mram = Arc::new(MatrixSram::with_banks(
        *MLEN,
        *MLEN as usize * 64,
        *BLEN,
        *MATRIX_SRAM_TYPE,
    ));
    let vram = Arc::new(VectorSram::from_mx_type(*VLEN, 32, *VECTOR_SRAM_TYPE));
    let state = initial_state();
    for (axis, values) in state.iter().enumerate() {
        write_state_packet(&mram, layout, axis, values).await;
    }

    let value = (0..HEADS)
        .flat_map(|head| (0..WIDTH).map(move |lane| (head % 3 + 2) as f32 + lane as f32 * 0.25))
        .collect::<Vec<_>>();
    let beta = repeated_by_head(|head| if head % 2 == 0 { 0.5 } else { 0.25 });
    let decay = [
        repeated_by_head(|head| if head % 2 == 0 { 0.5 } else { 1.0 }),
        repeated_by_head(|head| if head % 2 == 0 { 1.0 } else { 0.5 }),
    ];
    let key = [repeated_by_head(|_| 0.5), repeated_by_head(|_| 0.25)];
    let query = [repeated_by_head(|_| 0.25), repeated_by_head(|_| 0.5)];

    // Vector rows hold only ordinary temporaries/value/beta/output. Decay,
    // key and query are direct Matrix packet operands.
    for (row, values) in [(3, &value), (4, &beta)] {
        write_vector(&vram, row, values).await;
    }
    for (packet, values) in [
        (2, &decay[0]),
        (3, &decay[1]),
        (4, &key[0]),
        (5, &key[1]),
        (6, &query[0]),
        (7, &query[1]),
    ] {
        write_matrix_packet(&mram, layout, packet, values).await;
    }
    let mut accelerator = new_accelerator(mram.clone(), vram.clone()).await;
    let mut ops = vec![
        set_gp(1, packet_shape_word()),
        set_gp(2, packet_map_word(alpha)),
        op::Opcode::L_MVIEW_FULL {
            shape: 1,
            mapping: 2,
            slot: 0,
        },
        op::Opcode::L_MVIEW_FULL {
            shape: 1,
            mapping: 2,
            slot: 1,
        },
        op::Opcode::L_MVIEW_FULL {
            shape: 1,
            mapping: 2,
            slot: 2,
        },
        set_gp(3, 0),
        set_gp(4, *VLEN),
        set_gp(5, 2 * *VLEN),
        set_gp(6, 3 * *VLEN),
        set_gp(7, 4 * *VLEN),
        set_gp(8, 5 * *VLEN),
    ];
    for axis in 0..STATE_AXES {
        ops.extend([
            set_gp(1, state_base(axis)),
            set_gp(2, packet_base(2 + axis)),
            // state = state * decay; destination and both sources use Matrix views.
            mul_mv(1, 1, 2, 0b111),
            set_gp(2, packet_base(4 + axis)),
            // scratch = state * key; output remains an ordinary Vector row.
            mul_mv(3, 1, 2, 0b110),
            add(4, 4, 3),
        ]);
    }
    ops.extend([sub(5, 6, 4), mul(5, 5, 7)]);
    for axis in 0..STATE_AXES {
        ops.extend([
            set_gp(1, state_base(axis)),
            set_gp(2, packet_base(4 + axis)),
            // scratch = error * key, then state = state + scratch.
            mul_mv(3, 5, 2, 0b100),
            add_mv(1, 1, 3, 0b011),
            set_gp(2, packet_base(6 + axis)),
            mul_mv(3, 1, 2, 0b110),
            add(8, 8, 3),
        ]);
    }

    mram.reset_packet_counters();
    let start = Executor::current().now();
    for _ in 0..TOKENS {
        // Prediction and readout are reductions. They are architectural
        // temporaries, not recurrent state, and must start at zero for every
        // token while the Matrix-resident state deliberately persists.
        write_vector(&vram, 1, &vec![0.0; *VLEN as usize]).await;
        write_vector(&vram, 5, &vec![0.0; *VLEN as usize]).await;
        accelerator.do_ops(&ops, None, TimingDriver::Serial).await;
    }
    let cycles = (Executor::current().now() - start).as_picos() / PERIOD.as_picos();
    let matrix = accelerator.matrix_view_packet_counters();
    let vector = accelerator.lstream_packet_counters();
    let output = tensor_to_f32_vec(vram.read(5 * *VLEN).await.as_tensor());
    let state = read_state(&mram, layout).await;
    RecurrenceResult {
        output,
        state,
        matrix,
        vector,
        cycles,
    }
}

fn kda_reference() -> (Vec<f32>, Vec<f32>) {
    let mut state = initial_state();
    let value = (0..HEADS)
        .flat_map(|head| (0..WIDTH).map(move |lane| (head % 3 + 2) as f32 + lane as f32 * 0.25))
        .collect::<Vec<_>>();
    let mut output = vec![0.0; *VLEN as usize];
    for _ in 0..TOKENS {
        for head in 0..HEADS {
            let beta = if head % 2 == 0 { 0.5 } else { 0.25 };
            for lane in 0..WIDTH {
                let index = head * WIDTH + lane;
                let d0 = if head % 2 == 0 { 0.5 } else { 1.0 };
                let d1 = if head % 2 == 0 { 1.0 } else { 0.5 };
                state[0][index] *= d0;
                state[1][index] *= d1;
                let prediction = state[0][index] * 0.5 + state[1][index] * 0.25;
                let error = beta * (value[index] - prediction);
                state[0][index] += error * 0.5;
                state[1][index] += error * 0.25;
                output[index] = state[0][index] * 0.25 + state[1][index] * 0.5;
            }
        }
    }
    (output, flatten_state(&state))
}

async fn run_mamba(alpha: u32) -> RecurrenceResult {
    assert_eq!((*MLEN, *BLEN, *VLEN), (64, 4, 64));
    let layout = packet_layout(alpha);
    let mram = Arc::new(MatrixSram::with_banks(
        *MLEN,
        *MLEN as usize * 64,
        *BLEN,
        *MATRIX_SRAM_TYPE,
    ));
    let vram = Arc::new(VectorSram::from_mx_type(*VLEN, 32, *VECTOR_SRAM_TYPE));
    let state = initial_state();
    for (axis, values) in state.iter().enumerate() {
        write_state_packet(&mram, layout, axis, values).await;
    }
    let x = (0..HEADS)
        .flat_map(|head| (0..WIDTH).map(move |lane| (head % 3 + 1) as f32 + lane as f32 * 0.25))
        .collect::<Vec<_>>();
    let dt = repeated_by_head(|head| if head % 2 == 0 { 0.5 } else { 0.25 });
    let d = repeated_by_head(|_| 0.5);
    let decay = [
        repeated_by_head(|head| if head % 2 == 0 { 0.5 } else { 1.0 }),
        repeated_by_head(|head| if head % 2 == 0 { 1.0 } else { 0.5 }),
    ];
    let b = [repeated_by_head(|_| 0.5), repeated_by_head(|_| 0.25)];
    let c = [repeated_by_head(|_| 0.25), repeated_by_head(|_| 0.5)];
    for (row, values) in [(2, &x), (3, &dt), (4, &d)] {
        write_vector(&vram, row, values).await;
    }
    for (packet, values) in [
        (2, &decay[0]),
        (3, &decay[1]),
        (4, &b[0]),
        (5, &b[1]),
        (6, &c[0]),
        (7, &c[1]),
    ] {
        write_matrix_packet(&mram, layout, packet, values).await;
    }
    let mut accelerator = new_accelerator(mram.clone(), vram.clone()).await;
    let mut ops = vec![
        set_gp(1, packet_shape_word()),
        set_gp(2, packet_map_word(alpha)),
        op::Opcode::L_MVIEW_FULL {
            shape: 1,
            mapping: 2,
            slot: 0,
        },
        op::Opcode::L_MVIEW_FULL {
            shape: 1,
            mapping: 2,
            slot: 1,
        },
        op::Opcode::L_MVIEW_FULL {
            shape: 1,
            mapping: 2,
            slot: 2,
        },
        set_gp(3, 0),
        set_gp(4, *VLEN),
        set_gp(5, 2 * *VLEN),
        set_gp(6, 3 * *VLEN),
        set_gp(7, 4 * *VLEN),
    ];
    for axis in 0..STATE_AXES {
        ops.extend([
            set_gp(1, state_base(axis)),
            set_gp(2, packet_base(2 + axis)),
            mul_mv(1, 1, 2, 0b111),
            mul(3, 5, 6),
            set_gp(2, packet_base(4 + axis)),
            mul_mv(3, 3, 2, 0b100),
            add_mv(1, 1, 3, 0b011),
            set_gp(2, packet_base(6 + axis)),
            mul_mv(3, 1, 2, 0b110),
            add(4, 4, 3),
        ]);
    }
    ops.extend([mul(3, 5, 7), add(4, 4, 3)]);

    mram.reset_packet_counters();
    let start = Executor::current().now();
    for _ in 0..TOKENS {
        // The output reduction is per-token; recurrent state is intentionally
        // left in Matrix SRAM between invocations.
        write_vector(&vram, 1, &vec![0.0; *VLEN as usize]).await;
        accelerator.do_ops(&ops, None, TimingDriver::Serial).await;
    }
    let cycles = (Executor::current().now() - start).as_picos() / PERIOD.as_picos();
    let matrix = accelerator.matrix_view_packet_counters();
    let vector = accelerator.lstream_packet_counters();
    let output = tensor_to_f32_vec(vram.read(*VLEN).await.as_tensor());
    let state = read_state(&mram, layout).await;
    RecurrenceResult {
        output,
        state,
        matrix,
        vector,
        cycles,
    }
}

fn mamba_reference() -> (Vec<f32>, Vec<f32>) {
    let mut state = initial_state();
    let x = (0..HEADS)
        .flat_map(|head| (0..WIDTH).map(move |lane| (head % 3 + 1) as f32 + lane as f32 * 0.25))
        .collect::<Vec<_>>();
    let mut output = vec![0.0; *VLEN as usize];
    for _ in 0..TOKENS {
        for head in 0..HEADS {
            let dt = if head % 2 == 0 { 0.5 } else { 0.25 };
            for lane in 0..WIDTH {
                let index = head * WIDTH + lane;
                let d0 = if head % 2 == 0 { 0.5 } else { 1.0 };
                let d1 = if head % 2 == 0 { 1.0 } else { 0.5 };
                state[0][index] = d0 * state[0][index] + dt * 0.5 * x[index];
                state[1][index] = d1 * state[1][index] + dt * 0.25 * x[index];
                output[index] = state[0][index] * 0.25 + state[1][index] * 0.5 + x[index] * 0.5;
            }
        }
    }
    (output, flatten_state(&state))
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 1e-2 + 1e-2 * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index}: expected {expected}, got {actual}, tolerance {tolerance}"
        );
    }
}

#[tokio::test]
async fn kda_recurrence_uses_affine_matrix_packets_without_bank_stalls() {
    set_timing_mode(TimingMode::Serial);
    let executor = Executor::new();
    let result = Arc::new(Mutex::new(None));
    let task_result = result.clone();
    executor.spawn(async move {
        *task_result.lock().unwrap() = Some((run_kda(0).await, run_kda(1).await));
    });
    executor.enter(Instant::ETERNITY).await;
    let (fixed, affine) = result.lock().unwrap().take().unwrap();
    let (expected_output, expected_state) = kda_reference();

    assert_close(&fixed.output, &expected_output);
    assert_close(&fixed.state, &expected_state);
    assert_eq!(fixed.output, affine.output);
    assert_eq!(fixed.state, affine.state);
    // Joint source reads and state writeback all use the same physical bank
    // model. The fixed map concentrates every 16-head packet in four banks.
    assert_eq!(fixed.matrix.bank_stall_cycles, 300 * TOKENS as u64);
    assert_eq!(affine.matrix.bank_stall_cycles, 0);
    assert_eq!(fixed.vector, affine.vector);
    assert!(fixed.cycles > affine.cycles);
}

#[tokio::test]
async fn mamba_recurrence_uses_affine_matrix_packets_without_bank_stalls() {
    set_timing_mode(TimingMode::Serial);
    let executor = Executor::new();
    let result = Arc::new(Mutex::new(None));
    let task_result = result.clone();
    executor.spawn(async move {
        *task_result.lock().unwrap() = Some((run_mamba(0).await, run_mamba(1).await));
    });
    executor.enter(Instant::ETERNITY).await;
    let (fixed, affine) = result.lock().unwrap().take().unwrap();
    let (expected_output, expected_state) = mamba_reference();

    assert_close(&fixed.output, &expected_output);
    assert_close(&fixed.state, &expected_state);
    assert_eq!(fixed.output, affine.output);
    assert_eq!(fixed.state, affine.state);
    assert_eq!(fixed.matrix.bank_stall_cycles, 240 * TOKENS as u64);
    assert_eq!(affine.matrix.bank_stall_cycles, 0);
    assert_eq!(fixed.vector, affine.vector);
    assert!(fixed.cycles > affine.cycles);
}
