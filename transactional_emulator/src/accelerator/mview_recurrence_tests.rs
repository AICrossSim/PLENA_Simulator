//! Numerical recurrence tests that consume real multi-tile Matrix-SRAM packets.
//!
//! The ISA remains model independent: Matrix views only place and restore data;
//! every arithmetic step below is an existing Vector opcode.  Row-major and
//! pitch-1 and co-layout runs execute the same operation stream and differ
//! only in the compiler-selected tile pitch.

use std::sync::{Arc, Mutex};

use half::bf16;
use memory::{ErasedMemoryModel, MemoryBacked, NaiveTiming, WithTiming};
use quantize::{DataType, FpType, MxDataType, QuantTensor, tensor_to_f32_vec};
use runtime::{Executor, Instant};
use sram::matrix::{MatrixLayout, MatrixPacketCounterSnapshot};
use sram::{MatrixSram, VectorSram};
use tch::Tensor;

use super::{Accelerator, Scoreboard, TimingDriver};
use crate::matrix_machine::MatrixMachine;
use crate::op;
use crate::runtime_config::{
    BLEN, BROADCAST_AMOUNT, HLEN, MATRIX_SRAM_TYPE, MLEN, PERIOD, VECTOR_SRAM_TYPE, VLEN,
};
use crate::timing::{TimingMode, set_timing_mode};
use crate::vector_machine::{PacketCounterSnapshot, VectorMachine};

const HEADS: usize = 8;
const WIDTH: usize = 8;
const STATE_AXES: usize = 2;
const TOKENS: usize = 4;
const PACKET_ALLOCATION_PITCH: u32 = 2;
const HBM_PACKET_STRIDE_BYTES: u32 = 8192;
const HBM_TEST_CAPACITY: usize = 1 << 20;

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

fn packet_layout(tile_pitch_rows: u32) -> MatrixLayout {
    MatrixLayout {
        rows: 1,
        cols: WIDTH as u32,
        tile_count: HEADS as u32,
        tile_pitch_rows,
        alpha: 1,
        tile_skew: 0,
    }
}

fn packet_shape_word() -> u32 {
    ((WIDTH as u32 - 1) << 12) | ((HEADS as u32 - 1) << 24)
}

fn packet_map_word(tile_pitch_rows: u32) -> u32 {
    tile_pitch_rows
}

fn ltile_shape_word(rows: u32, cols: u32, tiles: u32) -> u32 {
    (rows - 1) | ((cols - 1) << 12) | ((tiles - 1) << 24)
}

fn ltile_map_word(pitch: u32, tile_phase_stride: Option<u32>, broadcast_minor: bool) -> u32 {
    let mut flags = 0_u32;
    let mut word = pitch;
    if let Some(phase) = tile_phase_stride {
        // Match the physical layout used to seed the SRAM: treatment changes
        // only the per-tile phase, while retaining PLENA's diagonal row term.
        word |= phase << 22;
    }
    if broadcast_minor {
        flags |= 1 << 3;
    }
    word | (flags << 28)
}

fn configure_ltile_view(ops: &mut Vec<op::Opcode>, slot: u8, shape: u32, mapping: u32) {
    ops.extend([
        set_gp(10, shape),
        set_gp(11, mapping),
        op::Opcode::L_TILE_CFG {
            shape: 10,
            mapping: 11,
            slot,
        },
    ]);
}

fn state_base(axis: usize) -> u32 {
    packet_base(axis)
}

fn packet_base(packet: usize) -> u32 {
    packet as u32 * HEADS as u32 * PACKET_ALLOCATION_PITCH * *MLEN
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

fn new_accelerator_with_hbm(
    mram: Arc<MatrixSram>,
    vram: Arc<VectorSram>,
    image: &[u8],
) -> (Accelerator, Arc<WithTiming<NaiveTiming, MemoryBacked>>) {
    assert!(image.len() <= HBM_TEST_CAPACITY);
    let m_machine = MatrixMachine::new(mram, vram.clone(), *MLEN, *HLEN, *BLEN, *BROADCAST_AMOUNT);
    let v_machine = VectorMachine::new(vram, *VLEN, *HLEN);
    let hbm = Arc::new(WithTiming::new(
        NaiveTiming::preset_ddr4_2400p(4),
        MemoryBacked::with_capacity(HBM_TEST_CAPACITY),
    ));
    hbm.data()
        .with_data(|bytes| bytes[..image.len()].copy_from_slice(image));
    let erased: Arc<dyn ErasedMemoryModel> = hbm.clone();
    (Accelerator::new(m_machine, v_machine, erased), hbm)
}

fn hbm_packet_offset(region: u32) -> u32 {
    region * HBM_PACKET_STRIDE_BYTES
}

fn write_hbm_state_packet(image: &mut [u8], region: u32, values: &[f32]) {
    let offset = hbm_packet_offset(region) as usize;
    let mut packet = QuantTensor::quantize(Tensor::from_slice(values), full_state_type());
    let (bytes, scale_bytes) = packet.into_bytes();
    assert!(scale_bytes.is_empty());
    assert!(bytes.len() <= HBM_PACKET_STRIDE_BYTES as usize);
    image[offset..offset + bytes.len()].copy_from_slice(&bytes);
}

fn read_hbm_state_packet(
    hbm: &WithTiming<NaiveTiming, MemoryBacked>,
    region: u32,
    values: usize,
) -> Vec<f32> {
    let offset = hbm_packet_offset(region) as usize;
    let bytes_per_value = full_state_type().element_type().size_in_bits() as usize / 8;
    let byte_len = values * bytes_per_value;
    let mut bytes = vec![0_u8; byte_len];
    hbm.data().with_data(|image| {
        bytes.copy_from_slice(&image[offset..offset + byte_len]);
    });
    let packet = QuantTensor::from_bytes(&bytes, &[], values, full_state_type());
    tensor_to_f32_vec(packet.as_tensor())
}

#[allow(clippy::too_many_arguments)]
fn append_matrix_view_dma(
    ops: &mut Vec<op::Opcode>,
    load: bool,
    matrix_base: u32,
    hbm_region: u32,
    rows: u32,
    cols: u32,
    affine: bool,
    broadcast_minor: bool,
) {
    const DMA_VIEW: u8 = 3;
    configure_ltile_view(
        ops,
        DMA_VIEW,
        ltile_shape_word(rows, cols, if broadcast_minor { 1 } else { FULL_TILES }),
        full_map(rows, cols, affine, broadcast_minor),
    );
    ops.extend([
        set_gp(12, matrix_base),
        set_gp(13, hbm_packet_offset(hbm_region)),
    ]);
    if load {
        ops.push(op::Opcode::H_PREFETCH_V_MV {
            rd: 12,
            rs1: 13,
            rs2: 0,
            rstride: 0,
            precision: op::VectorPrecision::State,
            view: DMA_VIEW,
        });
    } else {
        ops.push(op::Opcode::H_STORE_V_MV {
            rd: 12,
            rs1: 13,
            rs2: 0,
            rstride: 0,
            precision: op::VectorPrecision::State,
            view: DMA_VIEW,
        });
    }
}

async fn read_state(mram: &MatrixSram, layout: MatrixLayout) -> Vec<f32> {
    let mut result = Vec::new();
    for axis in 0..STATE_AXES {
        let (packet, _) = mram.read_layout_packet(state_base(axis), layout).await;
        result.extend(tensor_to_f32_vec(packet.as_tensor()));
    }
    result
}

async fn run_kda(tile_pitch_rows: u32) -> RecurrenceResult {
    assert_eq!((*MLEN, *BLEN, *VLEN), (64, 4, 64));
    let layout = packet_layout(tile_pitch_rows);
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
        set_gp(2, packet_map_word(tile_pitch_rows)),
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
        op::Opcode::L_TILE_CFG {
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

async fn run_mamba(tile_pitch_rows: u32) -> RecurrenceResult {
    assert_eq!((*MLEN, *BLEN, *VLEN), (64, 4, 64));
    let layout = packet_layout(tile_pitch_rows);
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
        set_gp(2, packet_map_word(tile_pitch_rows)),
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
        op::Opcode::L_TILE_CFG {
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
async fn kda_recurrence_uses_fixed_diagonal_pitch_without_bank_stalls() {
    set_timing_mode(TimingMode::Serial);
    let executor = Executor::new();
    let result = Arc::new(Mutex::new(None));
    let task_result = result.clone();
    executor.spawn(async move {
        *task_result.lock().unwrap() = Some((run_kda(1).await, run_kda(2).await));
    });
    executor.enter(Instant::ETERNITY).await;
    let (pitch_one, co_layout) = result.lock().unwrap().take().unwrap();
    let (expected_output, expected_state) = kda_reference();

    assert_close(&pitch_one.output, &expected_output);
    assert_close(&pitch_one.state, &expected_state);
    assert_eq!(pitch_one.output, co_layout.output);
    assert_eq!(pitch_one.state, co_layout.state);
    // Joint source reads and state writeback all use the same physical bank
    // model. The fixed map concentrates every 16-head packet in four banks.
    assert!(pitch_one.matrix.bank_stall_cycles > 0);
    assert_eq!(co_layout.matrix.bank_stall_cycles, 0);
    assert_eq!(pitch_one.vector, co_layout.vector);
    assert!(pitch_one.cycles > co_layout.cycles);
}

#[tokio::test]
async fn mamba_recurrence_uses_fixed_diagonal_pitch_without_bank_stalls() {
    set_timing_mode(TimingMode::Serial);
    let executor = Executor::new();
    let result = Arc::new(Mutex::new(None));
    let task_result = result.clone();
    executor.spawn(async move {
        *task_result.lock().unwrap() = Some((run_mamba(1).await, run_mamba(2).await));
    });
    executor.enter(Instant::ETERNITY).await;
    let (pitch_one, co_layout) = result.lock().unwrap().take().unwrap();
    let (expected_output, expected_state) = mamba_reference();

    assert_close(&pitch_one.output, &expected_output);
    assert_close(&pitch_one.state, &expected_state);
    assert_eq!(pitch_one.output, co_layout.output);
    assert_eq!(pitch_one.state, co_layout.state);
    assert!(pitch_one.matrix.bank_stall_cycles > 0);
    assert_eq!(co_layout.matrix.bank_stall_cycles, 0);
    assert_eq!(pitch_one.vector, co_layout.vector);
    assert!(pitch_one.cycles > co_layout.cycles);
}

#[derive(Debug)]
struct LTileResult {
    state: Vec<f32>,
    output: Vec<f32>,
    matrix: MatrixPacketCounterSnapshot,
}

async fn run_ltile_primitives(tile_skew: Option<u32>) -> LTileResult {
    const ROWS: u32 = 2;
    const TILES: u32 = 8;
    const COLS: u32 = 8;
    const SCALE_COLS: u32 = 4;
    const PITCH: u32 = 8;
    assert_eq!((*MLEN, *BLEN, *VLEN), (64, 4, 64));

    let state_layout = MatrixLayout {
        rows: ROWS,
        cols: COLS,
        tile_count: TILES,
        tile_pitch_rows: PITCH,
        alpha: 1,
        tile_skew: tile_skew.unwrap_or(0),
    };
    let source_layout = MatrixLayout {
        rows: 1,
        cols: COLS,
        tile_count: TILES,
        tile_pitch_rows: PITCH,
        alpha: 1,
        tile_skew: tile_skew.unwrap_or(0),
    };
    let scale_layout = MatrixLayout {
        rows: ROWS,
        cols: SCALE_COLS,
        tile_count: TILES,
        tile_pitch_rows: PITCH,
        alpha: 1,
        tile_skew: tile_skew.unwrap_or(0),
    };
    let output_layout = MatrixLayout {
        rows: 1,
        cols: COLS,
        tile_count: TILES,
        tile_pitch_rows: PITCH,
        alpha: 1,
        tile_skew: tile_skew.unwrap_or(0),
    };
    let state_base = 0;
    let source_base = 64 * *MLEN;
    let scale_base = 128 * *MLEN;
    let output_base = 192 * *MLEN;
    let state_input = (0..TILES * ROWS * COLS)
        .map(|index| 0.25 + index as f32 / 16.0)
        .collect::<Vec<_>>();
    let source = (0..TILES * COLS)
        .map(|index| 1.0 + index as f32 / 32.0)
        .collect::<Vec<_>>();
    let scales = (0..TILES)
        .flat_map(|tile| {
            (0..ROWS)
                .flat_map(move |row| [0.5 + row as f32 / 32.0 + tile as f32 / 64.0, 0.25, 0.0, 0.0])
        })
        .collect::<Vec<_>>();

    let mram = Arc::new(MatrixSram::with_banks(
        *MLEN,
        *MLEN as usize * 64,
        *BLEN,
        *MATRIX_SRAM_TYPE,
    ));
    let vram = Arc::new(VectorSram::from_mx_type(*VLEN, 32, *VECTOR_SRAM_TYPE));
    mram.write_layout_packet(
        state_base,
        state_layout,
        QuantTensor::quantize(Tensor::from_slice(&state_input), *MATRIX_SRAM_TYPE),
    )
    .await;
    mram.write_layout_packet(
        source_base,
        source_layout,
        QuantTensor::quantize(Tensor::from_slice(&source), *MATRIX_SRAM_TYPE),
    )
    .await;
    mram.write_layout_packet(
        scale_base,
        scale_layout,
        QuantTensor::quantize(Tensor::from_slice(&scales), *MATRIX_SRAM_TYPE),
    )
    .await;
    mram.write_layout_packet(
        output_base,
        output_layout,
        QuantTensor::quantize(
            Tensor::zeros(
                [(TILES * COLS) as i64],
                (tch::Kind::Float, tch::Device::Cpu),
            ),
            *MATRIX_SRAM_TYPE,
        ),
    )
    .await;

    let mut accelerator = new_accelerator(mram.clone(), vram).await;
    let state_map = ltile_map_word(PITCH, tile_skew, false);
    let scale_map = ltile_map_word(PITCH, tile_skew, true);
    let source_map = ltile_map_word(PITCH, tile_skew, false);
    let output_map = ltile_map_word(PITCH, tile_skew, false);
    let mut ops = Vec::new();

    configure_ltile_view(&mut ops, 0, ltile_shape_word(ROWS, COLS, TILES), state_map);
    configure_ltile_view(&mut ops, 1, ltile_shape_word(1, COLS, TILES), source_map);
    configure_ltile_view(
        &mut ops,
        2,
        ltile_shape_word(ROWS, SCALE_COLS, TILES),
        scale_map,
    );
    ops.extend([
        set_gp(1, state_base),
        set_gp(2, source_base),
        set_gp(3, scale_base),
        op::Opcode::L_TILE_EXEC {
            rd: 1,
            rs1: 2,
            rs2: 3,
            primitive: op::LTilePrimitive::ScaleAccum,
            source_axis: op::LTileAxis::Row,
            scale_axis: op::LTileAxis::Row,
        },
    ]);

    configure_ltile_view(&mut ops, 0, ltile_shape_word(1, COLS, TILES), output_map);
    configure_ltile_view(&mut ops, 1, ltile_shape_word(ROWS, COLS, TILES), state_map);
    configure_ltile_view(
        &mut ops,
        2,
        ltile_shape_word(ROWS, SCALE_COLS, TILES),
        scale_map,
    );
    ops.extend([
        set_gp(1, output_base),
        set_gp(2, state_base),
        set_gp(3, scale_base),
        op::Opcode::L_TILE_EXEC {
            rd: 1,
            rs1: 2,
            rs2: 3,
            primitive: op::LTilePrimitive::DotReduce,
            source_axis: op::LTileAxis::Row,
            scale_axis: op::LTileAxis::Row,
        },
    ]);

    configure_ltile_view(&mut ops, 0, ltile_shape_word(ROWS, COLS, TILES), state_map);
    configure_ltile_view(&mut ops, 1, ltile_shape_word(1, COLS, TILES), source_map);
    configure_ltile_view(
        &mut ops,
        2,
        ltile_shape_word(ROWS, SCALE_COLS, TILES),
        scale_map,
    );
    ops.extend([
        set_gp(1, state_base),
        set_gp(2, source_base),
        set_gp(3, scale_base),
        op::Opcode::L_TILE_EXEC {
            rd: 1,
            rs1: 2,
            rs2: 3,
            primitive: op::LTilePrimitive::OuterUpdate,
            source_axis: op::LTileAxis::Row,
            scale_axis: op::LTileAxis::Row,
        },
    ]);

    mram.reset_packet_counters();
    accelerator.do_ops(&ops, None, TimingDriver::Serial).await;
    let matrix = accelerator.matrix_view_packet_counters();
    let state = tensor_to_f32_vec(
        mram.read_layout_packet(state_base, state_layout)
            .await
            .0
            .as_tensor(),
    );
    let output = tensor_to_f32_vec(
        mram.read_layout_packet(output_base, output_layout)
            .await
            .0
            .as_tensor(),
    );
    LTileResult {
        state,
        output,
        matrix,
    }
}

#[tokio::test]
async fn l_tile_exec_runs_row_and_column_primitives_through_the_same_banks() {
    set_timing_mode(TimingMode::Serial);
    let executor = Executor::new();
    let result = Arc::new(Mutex::new(None));
    let task_result = result.clone();
    executor.spawn(async move {
        *task_result.lock().unwrap() = Some((
            run_ltile_primitives(None).await,
            run_ltile_primitives(Some(2)).await,
        ));
    });
    executor.enter(Instant::ETERNITY).await;
    let (fixed, affine) = result.lock().unwrap().take().unwrap();
    let rows = 2_usize;
    let tiles = 8_usize;
    let cols = 8_usize;
    let original = (0..tiles * rows * cols)
        .map(|index| 0.25 + index as f32 / 16.0)
        .collect::<Vec<_>>();
    let source = (0..tiles * cols)
        .map(|index| 1.0 + index as f32 / 32.0)
        .collect::<Vec<_>>();
    let mut after_scale = vec![0_f32; tiles * rows * cols];
    for tile in 0..tiles {
        for row in 0..rows {
            let a = 0.5 + row as f32 / 32.0 + tile as f32 / 64.0;
            for col in 0..cols {
                let index = (tile * rows + row) * cols + col;
                after_scale[index] = a * original[index] + 0.25 * source[tile * cols + col];
            }
        }
    }
    let mut expected_output = Vec::with_capacity(tiles * cols);
    for tile in 0..tiles {
        for col in 0..cols {
            expected_output.push(
                (0..rows)
                    .map(|row| {
                        let scale = 0.5 + row as f32 / 32.0 + tile as f32 / 64.0;
                        after_scale[(tile * rows + row) * cols + col] * scale
                    })
                    .sum::<f32>(),
            );
        }
    }
    let mut expected_state = after_scale;
    for tile in 0..tiles {
        for row in 0..rows {
            let scale = 0.5 + row as f32 / 32.0 + tile as f32 / 64.0;
            for col in 0..cols {
                expected_state[(tile * rows + row) * cols + col] +=
                    source[tile * cols + col] * scale;
            }
        }
    }

    assert_close(&fixed.output, &expected_output);
    assert_close(&fixed.state, &expected_state);
    assert_eq!(fixed.output, affine.output);
    assert_eq!(fixed.state, affine.state);
    assert!(fixed.matrix.bank_stall_cycles > affine.matrix.bank_stall_cycles);
}

// A full-state tile deliberately uses a pitch that is a multiple of the bank
// count, matching the real 128-row recurrent state.  Under fixed wiring, equal
// state rows from every head therefore land on the same bank words.  The
// treatment adds a field-specific tile phase: two banks per 8-value state row,
// one bank per scalar row.
const FULL_ROWS: u32 = 16;
const FULL_TILES: u32 = 8;
const FULL_COLS: u32 = 8;
const FULL_SCALE_COLS: u32 = 4;
const FULL_COMPACT_SCALE_COLS: u32 = 32;
const FULL_PITCH: u32 = 16;

#[derive(Debug)]
struct FullLTileResult {
    state: Vec<f32>,
    output: Vec<f32>,
    matrix: MatrixPacketCounterSnapshot,
    cycles: u64,
}

fn full_region_base(region: u32) -> u32 {
    region * 128 * *MLEN
}

fn full_layout(rows: u32, cols: u32, affine: bool) -> MatrixLayout {
    let words_per_row = cols / *BLEN;
    MatrixLayout {
        rows,
        cols,
        tile_count: FULL_TILES,
        tile_pitch_rows: if affine {
            2 * words_per_row
        } else {
            FULL_PITCH
        },
        alpha: 1,
        tile_skew: if affine { words_per_row } else { 0 },
    }
}

fn full_head_major_layout(rows: u32, cols: u32, affine: bool) -> MatrixLayout {
    if !affine {
        return full_layout(rows, cols, false);
    }
    MatrixLayout {
        rows,
        cols,
        tile_count: FULL_TILES,
        // Consecutive heads occupy consecutive groups of physical rows. For
        // a two-row coefficient pair this makes bank=(2*head+field+word), so
        // one key-column read uses each bank at most once.
        tile_pitch_rows: rows,
        alpha: 1,
        tile_skew: 0,
    }
}

fn full_map(rows: u32, cols: u32, affine: bool, broadcast_minor: bool) -> u32 {
    let _ = rows;
    let words_per_row = cols / *BLEN;
    ltile_map_word(
        if affine {
            2 * words_per_row
        } else {
            FULL_PITCH
        },
        affine.then_some(words_per_row),
        broadcast_minor,
    )
}

fn full_head_major_map(rows: u32, cols: u32, affine: bool) -> u32 {
    if !affine {
        return full_map(rows, cols, false, true);
    }
    let _ = cols;
    ltile_map_word(rows, Some(0), true)
}

fn append_ltile_exec(
    ops: &mut Vec<op::Opcode>,
    destination: (u32, u32, u32),
    source: (u32, u32, u32),
    scale: (u32, u32),
    affine: bool,
    primitive: op::LTilePrimitive,
) {
    let (dst_base, dst_rows, dst_cols) = destination;
    let (src_base, src_rows, src_cols) = source;
    let (scale_base, scale_rows) = scale;
    append_ltile_exec_with_scale_shape(
        ops,
        dst_base,
        dst_rows,
        dst_cols,
        src_base,
        src_rows,
        src_cols,
        scale_base,
        scale_rows,
        FULL_SCALE_COLS,
        FULL_TILES,
        affine,
        primitive,
        op::LTileAxis::Row,
    );
}

#[allow(clippy::too_many_arguments)]
fn append_ltile_exec_with_scale_shape(
    ops: &mut Vec<op::Opcode>,
    dst_base: u32,
    dst_rows: u32,
    dst_cols: u32,
    src_base: u32,
    src_rows: u32,
    src_cols: u32,
    scale_base: u32,
    scale_rows: u32,
    scale_cols: u32,
    scale_tiles: u32,
    affine: bool,
    primitive: op::LTilePrimitive,
    scale_axis: op::LTileAxis,
) {
    configure_ltile_view(
        ops,
        0,
        ltile_shape_word(dst_rows, dst_cols, FULL_TILES),
        full_map(dst_rows, dst_cols, affine, false),
    );
    configure_ltile_view(
        ops,
        1,
        ltile_shape_word(src_rows, src_cols, FULL_TILES),
        full_map(src_rows, src_cols, affine, false),
    );
    configure_ltile_view(
        ops,
        2,
        ltile_shape_word(scale_rows, scale_cols, scale_tiles),
        if scale_axis == op::LTileAxis::Column {
            full_head_major_map(scale_rows, scale_cols, affine)
        } else {
            full_map(scale_rows, scale_cols, affine, true)
        },
    );
    ops.extend([
        set_gp(1, dst_base),
        set_gp(2, src_base),
        set_gp(3, scale_base),
        op::Opcode::L_TILE_EXEC {
            rd: 1,
            rs1: 2,
            rs2: 3,
            primitive,
            source_axis: op::LTileAxis::Row,
            scale_axis,
        },
    ]);
}

#[allow(clippy::too_many_arguments)]
fn append_ltile_exec_compact(
    ops: &mut Vec<op::Opcode>,
    dst_base: u32,
    dst_rows: u32,
    dst_cols: u32,
    src_base: u32,
    src_rows: u32,
    src_cols: u32,
    scale_base: u32,
    scale_rows: u32,
    affine: bool,
    primitive: op::LTilePrimitive,
) {
    append_ltile_exec_with_scale_shape(
        ops,
        dst_base,
        dst_rows,
        dst_cols,
        src_base,
        src_rows,
        src_cols,
        scale_base,
        scale_rows,
        FULL_COMPACT_SCALE_COLS,
        1,
        affine,
        primitive,
        op::LTileAxis::Row,
    );
}

async fn seed_full_packet(
    mram: &MatrixSram,
    base: u32,
    rows: u32,
    cols: u32,
    affine: bool,
    values: &[f32],
) {
    mram.write_layout_packet(
        base,
        full_layout(rows, cols, affine),
        QuantTensor::quantize(Tensor::from_slice(values), mram.ty()),
    )
    .await;
}

async fn seed_full_head_major_packet(
    mram: &MatrixSram,
    base: u32,
    rows: u32,
    cols: u32,
    affine: bool,
    values: &[f32],
) {
    mram.write_layout_packet(
        base,
        full_head_major_layout(rows, cols, affine),
        QuantTensor::quantize(Tensor::from_slice(values), mram.ty()),
    )
    .await;
}

fn full_state_type() -> MxDataType {
    MxDataType::Plain(DataType::Fp(FpType::BF16))
}

fn full_state_seed() -> Vec<f32> {
    (0..FULL_TILES * FULL_ROWS * FULL_COLS)
        .map(|index| 0.5 + index as f32 / 64.0)
        .collect()
}

fn round_bf16(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

fn full_vector_seed(offset: f32) -> Vec<f32> {
    (0..FULL_TILES * FULL_COLS)
        .map(|index| offset + index as f32 / 128.0)
        .collect()
}

fn full_scales<F>(rows: u32, mut values: F) -> Vec<f32>
where
    F: FnMut(u32, u32) -> (f32, f32),
{
    let mut packed = Vec::with_capacity((FULL_TILES * rows * FULL_SCALE_COLS) as usize);
    for tile in 0..FULL_TILES {
        for row in 0..rows {
            let (a, b) = values(tile, row);
            packed.extend([a, b, 0.0, 0.0]);
        }
    }
    packed
}

/// Keep projected per-head fields in their natural `[head][field][key]`
/// order. `L_TILE_EXEC` selects a key column and restores one scalar (or one
/// `[a,b]` pair) per head, so no copied key-major transpose is involved.
fn full_head_major_fields<F>(field_rows: u32, mut value: F) -> Vec<f32>
where
    F: FnMut(u32, u32, u32) -> f32,
{
    let mut packed = Vec::with_capacity((FULL_TILES * field_rows * FULL_ROWS) as usize);
    for tile in 0..FULL_TILES {
        for field in 0..field_rows {
            for key in 0..FULL_ROWS {
                packed.push(value(tile, field, key));
            }
        }
    }
    packed
}

fn full_compact_scale_pairs<F>(rows: u32, mut values: F) -> Vec<f32>
where
    F: FnMut(u32, u32) -> (f32, f32),
{
    let mut packed = Vec::with_capacity((rows * FULL_COMPACT_SCALE_COLS) as usize);
    for row in 0..rows {
        for tile in 0..FULL_TILES {
            let (a, b) = values(tile, row);
            packed.extend([a, b]);
        }
        packed.resize(
            packed.len() + (FULL_COMPACT_SCALE_COLS - 2 * FULL_TILES) as usize,
            0.0,
        );
    }
    packed
}

fn full_compact_scalars<F>(rows: u32, mut value: F) -> Vec<f32>
where
    F: FnMut(u32, u32) -> f32,
{
    let mut packed = Vec::with_capacity((rows * FULL_COMPACT_SCALE_COLS) as usize);
    for row in 0..rows {
        for tile in 0..FULL_TILES {
            packed.push(value(tile, row));
        }
        packed.resize(
            packed.len() + (FULL_COMPACT_SCALE_COLS - FULL_TILES) as usize,
            0.0,
        );
    }
    packed
}

async fn read_full_packet(
    mram: &MatrixSram,
    base: u32,
    rows: u32,
    cols: u32,
    affine: bool,
) -> Vec<f32> {
    tensor_to_f32_vec(
        mram.read_layout_packet(base, full_layout(rows, cols, affine))
            .await
            .0
            .as_tensor(),
    )
}

async fn run_full_mamba_ltile(affine: bool) -> FullLTileResult {
    assert_eq!((*MLEN, *BLEN, *VLEN), (64, 4, 64));
    let state_base = full_region_base(0);
    let x_base = full_region_base(1);
    let scratch_base = full_region_base(2);
    let dt_base = full_region_base(3);
    let update_base = full_region_base(4);
    let c_base = full_region_base(5);
    let output_base = full_region_base(6);
    let skip_base = full_region_base(7);
    let state = full_state_seed();
    let x = full_vector_seed(1.0);
    let zeros = vec![0.0; (FULL_TILES * FULL_COLS) as usize];
    let dt = full_scales(1, |tile, _| (0.0, 0.25 + tile as f32 / 64.0));
    let update = full_scales(FULL_ROWS, |tile, row| {
        (
            0.75 + row as f32 / 16.0,
            0.125 + tile as f32 / 128.0 + row as f32 / 64.0,
        )
    });
    let c = full_scales(FULL_ROWS, |tile, row| {
        (0.25 + tile as f32 / 128.0 + row as f32 / 16.0, 0.0)
    });
    let skip = full_scales(1, |tile, _| (1.0, 0.5 + tile as f32 / 128.0));

    let mram = Arc::new(MatrixSram::with_banks(
        *MLEN,
        *MLEN as usize * 1024,
        *BLEN,
        full_state_type(),
    ));
    let vram = Arc::new(VectorSram::from_mx_type(*VLEN, 32, *VECTOR_SRAM_TYPE));
    for (base, rows, cols, values) in [
        (state_base, FULL_ROWS, FULL_COLS, state.as_slice()),
        (x_base, 1, FULL_COLS, x.as_slice()),
        (scratch_base, 1, FULL_COLS, zeros.as_slice()),
        (dt_base, 1, FULL_SCALE_COLS, dt.as_slice()),
        (update_base, FULL_ROWS, FULL_SCALE_COLS, update.as_slice()),
        (c_base, FULL_ROWS, FULL_SCALE_COLS, c.as_slice()),
        (output_base, 1, FULL_COLS, zeros.as_slice()),
        (skip_base, 1, FULL_SCALE_COLS, skip.as_slice()),
    ] {
        seed_full_packet(&mram, base, rows, cols, affine, values).await;
    }

    let mut ops = Vec::new();
    append_ltile_exec(
        &mut ops,
        (scratch_base, 1, FULL_COLS),
        (x_base, 1, FULL_COLS),
        (dt_base, 1),
        affine,
        op::LTilePrimitive::ScaleAccum,
    );
    append_ltile_exec(
        &mut ops,
        (state_base, FULL_ROWS, FULL_COLS),
        (scratch_base, 1, FULL_COLS),
        (update_base, FULL_ROWS),
        affine,
        op::LTilePrimitive::ScaleAccum,
    );
    append_ltile_exec(
        &mut ops,
        (output_base, 1, FULL_COLS),
        (state_base, FULL_ROWS, FULL_COLS),
        (c_base, FULL_ROWS),
        affine,
        op::LTilePrimitive::DotReduce,
    );
    append_ltile_exec(
        &mut ops,
        (output_base, 1, FULL_COLS),
        (x_base, 1, FULL_COLS),
        (skip_base, 1),
        affine,
        op::LTilePrimitive::ScaleAccum,
    );

    let mut accelerator = new_accelerator(mram.clone(), vram).await;
    mram.reset_packet_counters();
    let start = Executor::current().now();
    for _ in 0..TOKENS {
        // State persists across tokens, but the reduction target is a
        // token-local value. DOT_REDUCE reads the old destination so that
        // several state chunks can accumulate into it; therefore the program
        // must explicitly initialise the first chunk's accumulator.
        seed_full_packet(&mram, output_base, 1, FULL_COLS, affine, &zeros).await;
        accelerator.do_ops(&ops, None, TimingDriver::Serial).await;
    }
    let cycles = (Executor::current().now() - start).as_picos() / PERIOD.as_picos();
    let matrix = mram.packet_counter_snapshot();
    FullLTileResult {
        state: read_full_packet(&mram, state_base, FULL_ROWS, FULL_COLS, affine).await,
        output: read_full_packet(&mram, output_base, 1, FULL_COLS, affine).await,
        matrix,
        cycles,
    }
}

fn full_mamba_reference() -> (Vec<f32>, Vec<f32>) {
    let mut state = full_state_seed()
        .into_iter()
        .map(round_bf16)
        .collect::<Vec<_>>();
    let x = full_vector_seed(1.0)
        .into_iter()
        .map(round_bf16)
        .collect::<Vec<_>>();
    let mut output = vec![0.0; (FULL_TILES * FULL_COLS) as usize];
    for _ in 0..TOKENS {
        for tile in 0..FULL_TILES as usize {
            let dt = round_bf16(0.25 + tile as f32 / 64.0);
            for col in 0..FULL_COLS as usize {
                let vector_index = tile * FULL_COLS as usize + col;
                let scratch = round_bf16(dt * x[vector_index]);
                let mut y = 0.0;
                for row in 0..FULL_ROWS as usize {
                    let state_index = (tile * FULL_ROWS as usize + row) * FULL_COLS as usize + col;
                    let decay = round_bf16(0.75 + row as f32 / 16.0);
                    let b = round_bf16(0.125 + tile as f32 / 128.0 + row as f32 / 64.0);
                    state[state_index] = round_bf16(decay * state[state_index] + b * scratch);
                    let c = round_bf16(0.25 + tile as f32 / 128.0 + row as f32 / 16.0);
                    y += c * state[state_index];
                }
                let reduced = round_bf16(y);
                let d = round_bf16(0.5 + tile as f32 / 128.0);
                output[vector_index] = round_bf16(reduced + d * x[vector_index]);
            }
        }
    }
    (output, state)
}

async fn run_full_kda_ltile(affine: bool) -> FullLTileResult {
    assert_eq!((*MLEN, *BLEN, *VLEN), (64, 4, 64));
    let state_base = full_region_base(0);
    let zero_base = full_region_base(1);
    let decay_base = full_region_base(2);
    let pred_base = full_region_base(3);
    let k_base = full_region_base(4);
    let value_base = full_region_base(5);
    let error_base = full_region_base(6);
    let beta_base = full_region_base(7);
    let q_base = full_region_base(8);
    let output_base = full_region_base(9);
    let state = full_state_seed();
    let zero = vec![0.0; (FULL_TILES * FULL_COLS) as usize];
    let value = full_vector_seed(1.5);
    let decay = full_head_major_fields(2, |tile, field, key| match field {
        0 => 0.75 + tile as f32 / 128.0 + key as f32 / 32.0,
        1 => 0.0,
        _ => unreachable!(),
    });
    let k = full_head_major_fields(1, |tile, _, key| {
        0.125 + tile as f32 / 256.0 + key as f32 / 16.0
    });
    let beta = full_scales(1, |tile, _| {
        let beta = 0.25 + tile as f32 / 128.0;
        (beta, -beta)
    });
    let q = full_head_major_fields(1, |tile, _, key| {
        0.25 + tile as f32 / 256.0 + key as f32 / 32.0
    });

    let mram = Arc::new(MatrixSram::with_banks(
        *MLEN,
        *MLEN as usize * 1408,
        *BLEN,
        full_state_type(),
    ));
    let vram = Arc::new(VectorSram::from_mx_type(*VLEN, 32, *VECTOR_SRAM_TYPE));
    for (base, rows, cols, values) in [
        (state_base, FULL_ROWS, FULL_COLS, state.as_slice()),
        (zero_base, 1, FULL_COLS, zero.as_slice()),
        (pred_base, 1, FULL_COLS, zero.as_slice()),
        (value_base, 1, FULL_COLS, value.as_slice()),
        (error_base, 1, FULL_COLS, value.as_slice()),
        (beta_base, 1, FULL_SCALE_COLS, beta.as_slice()),
        (output_base, 1, FULL_COLS, zero.as_slice()),
    ] {
        seed_full_packet(&mram, base, rows, cols, affine, values).await;
    }
    for (base, rows, values) in [
        (decay_base, 2, decay.as_slice()),
        (k_base, 1, k.as_slice()),
        (q_base, 1, q.as_slice()),
    ] {
        seed_full_head_major_packet(&mram, base, rows, FULL_ROWS, affine, values).await;
    }

    let mut ops = Vec::new();
    append_ltile_exec_with_scale_shape(
        &mut ops,
        state_base,
        FULL_ROWS,
        FULL_COLS,
        zero_base,
        1,
        FULL_COLS,
        decay_base,
        2,
        FULL_ROWS,
        FULL_TILES,
        affine,
        op::LTilePrimitive::ScaleAccum,
        op::LTileAxis::Column,
    );
    append_ltile_exec_with_scale_shape(
        &mut ops,
        pred_base,
        1,
        FULL_COLS,
        state_base,
        FULL_ROWS,
        FULL_COLS,
        k_base,
        1,
        FULL_ROWS,
        FULL_TILES,
        affine,
        op::LTilePrimitive::DotReduce,
        op::LTileAxis::Column,
    );
    append_ltile_exec(
        &mut ops,
        (error_base, 1, FULL_COLS),
        (pred_base, 1, FULL_COLS),
        (beta_base, 1),
        affine,
        op::LTilePrimitive::ScaleAccum,
    );
    append_ltile_exec_with_scale_shape(
        &mut ops,
        state_base,
        FULL_ROWS,
        FULL_COLS,
        error_base,
        1,
        FULL_COLS,
        k_base,
        1,
        FULL_ROWS,
        FULL_TILES,
        affine,
        op::LTilePrimitive::OuterUpdate,
        op::LTileAxis::Column,
    );
    append_ltile_exec_with_scale_shape(
        &mut ops,
        output_base,
        1,
        FULL_COLS,
        state_base,
        FULL_ROWS,
        FULL_COLS,
        q_base,
        1,
        FULL_ROWS,
        FULL_TILES,
        affine,
        op::LTilePrimitive::DotReduce,
        op::LTileAxis::Column,
    );

    let mut accelerator = new_accelerator(mram.clone(), vram).await;
    mram.reset_packet_counters();
    let start = Executor::current().now();
    for _ in 0..TOKENS {
        // Projection produces a fresh v tensor for every token; seed the
        // destination through the same affine Matrix-SRAM mapping before the
        // error primitive consumes it.
        seed_full_packet(&mram, error_base, 1, FULL_COLS, affine, &value).await;
        // Prediction and readout reduce across one or more state chunks. They
        // are token-local accumulators, unlike the persistent recurrent state.
        seed_full_packet(&mram, pred_base, 1, FULL_COLS, affine, &zero).await;
        seed_full_packet(&mram, output_base, 1, FULL_COLS, affine, &zero).await;
        accelerator.do_ops(&ops, None, TimingDriver::Serial).await;
    }
    let cycles = (Executor::current().now() - start).as_picos() / PERIOD.as_picos();
    let matrix = mram.packet_counter_snapshot();
    FullLTileResult {
        state: read_full_packet(&mram, state_base, FULL_ROWS, FULL_COLS, affine).await,
        output: read_full_packet(&mram, output_base, 1, FULL_COLS, affine).await,
        matrix,
        cycles,
    }
}

fn full_kda_reference() -> (Vec<f32>, Vec<f32>) {
    let mut state = full_state_seed()
        .into_iter()
        .map(round_bf16)
        .collect::<Vec<_>>();
    let value = full_vector_seed(1.5)
        .into_iter()
        .map(round_bf16)
        .collect::<Vec<_>>();
    let mut output = vec![0.0; (FULL_TILES * FULL_COLS) as usize];
    for _ in 0..TOKENS {
        for tile in 0..FULL_TILES as usize {
            let beta = round_bf16(0.25 + tile as f32 / 128.0);
            let negative_beta = round_bf16(-beta);
            for col in 0..FULL_COLS as usize {
                let vector_index = tile * FULL_COLS as usize + col;
                let mut prediction = 0.0;
                for row in 0..FULL_ROWS as usize {
                    let state_index = (tile * FULL_ROWS as usize + row) * FULL_COLS as usize + col;
                    let decay = round_bf16(0.75 + tile as f32 / 128.0 + row as f32 / 32.0);
                    let k = round_bf16(0.125 + tile as f32 / 256.0 + row as f32 / 16.0);
                    state[state_index] = round_bf16(state[state_index] * decay);
                    prediction += state[state_index] * k;
                }
                let prediction = round_bf16(prediction);
                let error = round_bf16(beta * value[vector_index] + negative_beta * prediction);
                let mut readout = 0.0;
                for row in 0..FULL_ROWS as usize {
                    let state_index = (tile * FULL_ROWS as usize + row) * FULL_COLS as usize + col;
                    let k = round_bf16(0.125 + tile as f32 / 256.0 + row as f32 / 16.0);
                    let q = round_bf16(0.25 + tile as f32 / 256.0 + row as f32 / 32.0);
                    state[state_index] = round_bf16(state[state_index] + error * k);
                    readout += state[state_index] * q;
                }
                output[vector_index] = round_bf16(readout);
            }
        }
    }
    (output, state)
}

#[derive(Debug)]
struct ConnectedLTileResult {
    state: Vec<f32>,
    output: Vec<f32>,
    matrix: MatrixPacketCounterSnapshot,
    cycles: u64,
}

#[allow(clippy::too_many_arguments)]
async fn execute_hbm_connected_program(
    ops: Vec<op::Opcode>,
    mram: Arc<MatrixSram>,
    image: Vec<u8>,
    state_output_region: u32,
    state_values: usize,
    output_region: u32,
    output_values: usize,
    scoreboard: bool,
) -> ConnectedLTileResult {
    set_timing_mode(if scoreboard {
        TimingMode::Scoreboard
    } else {
        TimingMode::Serial
    });
    let vram = Arc::new(VectorSram::from_mx_type(*VLEN, 32, *VECTOR_SRAM_TYPE));
    let (mut accelerator, hbm) = new_accelerator_with_hbm(mram.clone(), vram, &image);
    mram.reset_packet_counters();
    let start = Executor::current().now();
    if scoreboard {
        let mut dependencies = Scoreboard::new(false);
        accelerator
            .do_ops(
                &ops,
                None,
                TimingDriver::Scoreboard {
                    scoreboard: &mut dependencies,
                },
            )
            .await;
    } else {
        accelerator.do_ops(&ops, None, TimingDriver::Serial).await;
    }
    let cycles = (Executor::current().now() - start).as_picos() / PERIOD.as_picos();
    ConnectedLTileResult {
        state: read_hbm_state_packet(&hbm, state_output_region, state_values),
        output: read_hbm_state_packet(&hbm, output_region, output_values),
        matrix: mram.packet_counter_snapshot(),
        cycles,
    }
}

async fn run_hbm_connected_mamba(affine: bool, scoreboard: bool) -> ConnectedLTileResult {
    const STATE_IN: u32 = 0;
    const X_IN: u32 = 1;
    const ZERO_IN: u32 = 2;
    const DT_IN: u32 = 3;
    const UPDATE_IN: u32 = 4;
    const C_IN: u32 = 5;
    const SKIP_IN: u32 = 6;
    const OUTPUT_BASE: u32 = 16;
    const STATE_OUT: u32 = 31;

    let state_base = full_region_base(0);
    let x_base = full_region_base(1);
    let scratch_base = full_region_base(2);
    let dt_base = full_region_base(3);
    let update_base = full_region_base(4);
    let c_base = full_region_base(5);
    let output_base = full_region_base(6);
    let skip_base = full_region_base(7);
    let state = full_state_seed();
    let x = full_vector_seed(1.0);
    let zeros = vec![0.0; (FULL_TILES * FULL_COLS) as usize];
    let dt = full_compact_scale_pairs(1, |tile, _| (0.0, 0.25 + tile as f32 / 64.0));
    let update = full_compact_scale_pairs(FULL_ROWS, |tile, row| {
        (
            0.75 + row as f32 / 16.0,
            0.125 + tile as f32 / 128.0 + row as f32 / 64.0,
        )
    });
    let c = full_compact_scalars(FULL_ROWS, |tile, row| {
        0.25 + tile as f32 / 128.0 + row as f32 / 16.0
    });
    let skip = full_compact_scale_pairs(1, |tile, _| (1.0, 0.5 + tile as f32 / 128.0));
    let mut image = vec![0_u8; HBM_TEST_CAPACITY];
    for (region, values) in [
        (STATE_IN, state.as_slice()),
        (X_IN, x.as_slice()),
        (ZERO_IN, zeros.as_slice()),
        (DT_IN, dt.as_slice()),
        (UPDATE_IN, update.as_slice()),
        (C_IN, c.as_slice()),
        (SKIP_IN, skip.as_slice()),
    ] {
        write_hbm_state_packet(&mut image, region, values);
    }

    let mut ops = Vec::new();
    append_matrix_view_dma(
        &mut ops, true, state_base, STATE_IN, FULL_ROWS, FULL_COLS, affine, false,
    );
    for token in 0..TOKENS as u32 {
        for (base, region, rows, cols, broadcast) in [
            (x_base, X_IN, 1, FULL_COLS, false),
            (scratch_base, ZERO_IN, 1, FULL_COLS, false),
            (dt_base, DT_IN, 1, FULL_COMPACT_SCALE_COLS, true),
            (
                update_base,
                UPDATE_IN,
                FULL_ROWS,
                FULL_COMPACT_SCALE_COLS,
                true,
            ),
            (c_base, C_IN, FULL_ROWS, FULL_COMPACT_SCALE_COLS, true),
            (output_base, ZERO_IN, 1, FULL_COLS, false),
            (skip_base, SKIP_IN, 1, FULL_COMPACT_SCALE_COLS, true),
        ] {
            append_matrix_view_dma(&mut ops, true, base, region, rows, cols, affine, broadcast);
        }
        append_ltile_exec_compact(
            &mut ops,
            scratch_base,
            1,
            FULL_COLS,
            x_base,
            1,
            FULL_COLS,
            dt_base,
            1,
            affine,
            op::LTilePrimitive::ScaleAccum,
        );
        append_ltile_exec_compact(
            &mut ops,
            state_base,
            FULL_ROWS,
            FULL_COLS,
            scratch_base,
            1,
            FULL_COLS,
            update_base,
            FULL_ROWS,
            affine,
            op::LTilePrimitive::ScaleAccum,
        );
        append_ltile_exec_compact(
            &mut ops,
            output_base,
            1,
            FULL_COLS,
            state_base,
            FULL_ROWS,
            FULL_COLS,
            c_base,
            FULL_ROWS,
            affine,
            op::LTilePrimitive::DotReduce,
        );
        append_ltile_exec_compact(
            &mut ops,
            output_base,
            1,
            FULL_COLS,
            x_base,
            1,
            FULL_COLS,
            skip_base,
            1,
            affine,
            op::LTilePrimitive::ScaleAccum,
        );
        append_matrix_view_dma(
            &mut ops,
            false,
            output_base,
            OUTPUT_BASE + token,
            1,
            FULL_COLS,
            affine,
            false,
        );
    }
    append_matrix_view_dma(
        &mut ops, false, state_base, STATE_OUT, FULL_ROWS, FULL_COLS, affine, false,
    );

    let mram = Arc::new(MatrixSram::with_banks(
        *MLEN,
        *MLEN as usize * 1024,
        *BLEN,
        full_state_type(),
    ));
    execute_hbm_connected_program(
        ops,
        mram,
        image,
        STATE_OUT,
        state.len(),
        OUTPUT_BASE + TOKENS as u32 - 1,
        zeros.len(),
        scoreboard,
    )
    .await
}

async fn run_hbm_connected_kda(affine: bool, scoreboard: bool) -> ConnectedLTileResult {
    const STATE_IN: u32 = 0;
    const ZERO_IN: u32 = 1;
    const DECAY_IN: u32 = 2;
    const K_IN: u32 = 3;
    const VALUE_IN: u32 = 4;
    const BETA_IN: u32 = 5;
    const Q_IN: u32 = 6;
    const OUTPUT_BASE: u32 = 16;
    const STATE_OUT: u32 = 31;

    let state_base = full_region_base(0);
    let zero_base = full_region_base(1);
    let decay_base = full_region_base(2);
    let pred_base = full_region_base(3);
    let k_base = full_region_base(4);
    let value_base = full_region_base(5);
    let error_base = full_region_base(6);
    let beta_base = full_region_base(7);
    let q_base = full_region_base(8);
    let output_base = full_region_base(9);
    let state = full_state_seed();
    let zero = vec![0.0; (FULL_TILES * FULL_COLS) as usize];
    let value = full_vector_seed(1.5);
    let decay = full_compact_scale_pairs(FULL_ROWS, |tile, row| {
        (0.75 + tile as f32 / 128.0 + row as f32 / 32.0, 0.0)
    });
    let k = full_compact_scalars(FULL_ROWS, |tile, row| {
        0.125 + tile as f32 / 256.0 + row as f32 / 16.0
    });
    let beta = full_compact_scale_pairs(1, |tile, _| {
        let beta = 0.25 + tile as f32 / 128.0;
        (beta, -beta)
    });
    let q = full_compact_scalars(FULL_ROWS, |tile, row| {
        0.25 + tile as f32 / 256.0 + row as f32 / 32.0
    });
    let mut image = vec![0_u8; HBM_TEST_CAPACITY];
    for (region, values) in [
        (STATE_IN, state.as_slice()),
        (ZERO_IN, zero.as_slice()),
        (DECAY_IN, decay.as_slice()),
        (K_IN, k.as_slice()),
        (VALUE_IN, value.as_slice()),
        (BETA_IN, beta.as_slice()),
        (Q_IN, q.as_slice()),
    ] {
        write_hbm_state_packet(&mut image, region, values);
    }

    let mut ops = Vec::new();
    append_matrix_view_dma(
        &mut ops, true, state_base, STATE_IN, FULL_ROWS, FULL_COLS, affine, false,
    );
    append_matrix_view_dma(
        &mut ops, true, zero_base, ZERO_IN, 1, FULL_COLS, affine, false,
    );
    for token in 0..TOKENS as u32 {
        for (base, region, rows, cols, broadcast) in [
            (
                decay_base,
                DECAY_IN,
                FULL_ROWS,
                FULL_COMPACT_SCALE_COLS,
                true,
            ),
            (pred_base, ZERO_IN, 1, FULL_COLS, false),
            (k_base, K_IN, FULL_ROWS, FULL_COMPACT_SCALE_COLS, true),
            (value_base, VALUE_IN, 1, FULL_COLS, false),
            (error_base, VALUE_IN, 1, FULL_COLS, false),
            (beta_base, BETA_IN, 1, FULL_COMPACT_SCALE_COLS, true),
            (q_base, Q_IN, FULL_ROWS, FULL_COMPACT_SCALE_COLS, true),
            (output_base, ZERO_IN, 1, FULL_COLS, false),
        ] {
            append_matrix_view_dma(&mut ops, true, base, region, rows, cols, affine, broadcast);
        }
        append_ltile_exec_compact(
            &mut ops,
            state_base,
            FULL_ROWS,
            FULL_COLS,
            zero_base,
            1,
            FULL_COLS,
            decay_base,
            FULL_ROWS,
            affine,
            op::LTilePrimitive::ScaleAccum,
        );
        append_ltile_exec_compact(
            &mut ops,
            pred_base,
            1,
            FULL_COLS,
            state_base,
            FULL_ROWS,
            FULL_COLS,
            k_base,
            FULL_ROWS,
            affine,
            op::LTilePrimitive::DotReduce,
        );
        append_ltile_exec_compact(
            &mut ops,
            error_base,
            1,
            FULL_COLS,
            pred_base,
            1,
            FULL_COLS,
            beta_base,
            1,
            affine,
            op::LTilePrimitive::ScaleAccum,
        );
        append_ltile_exec_compact(
            &mut ops,
            state_base,
            FULL_ROWS,
            FULL_COLS,
            error_base,
            1,
            FULL_COLS,
            k_base,
            FULL_ROWS,
            affine,
            op::LTilePrimitive::OuterUpdate,
        );
        append_ltile_exec_compact(
            &mut ops,
            output_base,
            1,
            FULL_COLS,
            state_base,
            FULL_ROWS,
            FULL_COLS,
            q_base,
            FULL_ROWS,
            affine,
            op::LTilePrimitive::DotReduce,
        );
        append_matrix_view_dma(
            &mut ops,
            false,
            output_base,
            OUTPUT_BASE + token,
            1,
            FULL_COLS,
            affine,
            false,
        );
    }
    append_matrix_view_dma(
        &mut ops, false, state_base, STATE_OUT, FULL_ROWS, FULL_COLS, affine, false,
    );

    let mram = Arc::new(MatrixSram::with_banks(
        *MLEN,
        *MLEN as usize * 1408,
        *BLEN,
        full_state_type(),
    ));
    execute_hbm_connected_program(
        ops,
        mram,
        image,
        STATE_OUT,
        state.len(),
        OUTPUT_BASE + TOKENS as u32 - 1,
        zero.len(),
        scoreboard,
    )
    .await
}

#[tokio::test]
async fn dot_reduce_accumulates_across_two_state_chunks() {
    const CHUNK_ROWS: u32 = FULL_ROWS / 2;
    set_timing_mode(TimingMode::Serial);
    let executor = Executor::new();
    let result = Arc::new(Mutex::new(None));
    let task_result = result.clone();
    executor.spawn(async move {
        let first_state_base = full_region_base(0);
        let second_state_base = full_region_base(1);
        let first_scale_base = full_region_base(2);
        let second_scale_base = full_region_base(3);
        let output_base = full_region_base(4);
        let first = (0..FULL_TILES * CHUNK_ROWS * FULL_COLS)
            .map(|index| 0.25 + index as f32 / 64.0)
            .collect::<Vec<_>>();
        let second = (0..FULL_TILES * CHUNK_ROWS * FULL_COLS)
            .map(|index| 1.0 + index as f32 / 32.0)
            .collect::<Vec<_>>();
        let first_scale = full_scales(CHUNK_ROWS, |tile, row| {
            (0.5 + tile as f32 / 128.0 + row as f32 / 64.0, 0.0)
        });
        let second_scale = full_scales(CHUNK_ROWS, |tile, row| {
            (0.75 + tile as f32 / 128.0 + row as f32 / 32.0, 0.0)
        });
        let zero = vec![0.0; (FULL_TILES * FULL_COLS) as usize];
        let mram = Arc::new(MatrixSram::with_banks(
            *MLEN,
            *MLEN as usize * 640,
            *BLEN,
            full_state_type(),
        ));
        let vram = Arc::new(VectorSram::from_mx_type(*VLEN, 32, *VECTOR_SRAM_TYPE));
        for (base, rows, cols, values) in [
            (first_state_base, CHUNK_ROWS, FULL_COLS, first.as_slice()),
            (second_state_base, CHUNK_ROWS, FULL_COLS, second.as_slice()),
            (
                first_scale_base,
                CHUNK_ROWS,
                FULL_SCALE_COLS,
                first_scale.as_slice(),
            ),
            (
                second_scale_base,
                CHUNK_ROWS,
                FULL_SCALE_COLS,
                second_scale.as_slice(),
            ),
            (output_base, 1, FULL_COLS, zero.as_slice()),
        ] {
            seed_full_packet(&mram, base, rows, cols, true, values).await;
        }
        let mut ops = Vec::new();
        for (state_base, scale_base) in [
            (first_state_base, first_scale_base),
            (second_state_base, second_scale_base),
        ] {
            append_ltile_exec(
                &mut ops,
                (output_base, 1, FULL_COLS),
                (state_base, CHUNK_ROWS, FULL_COLS),
                (scale_base, CHUNK_ROWS),
                true,
                op::LTilePrimitive::DotReduce,
            );
        }
        let mut accelerator = new_accelerator(mram.clone(), vram).await;
        accelerator.do_ops(&ops, None, TimingDriver::Serial).await;
        let output = read_full_packet(&mram, output_base, 1, FULL_COLS, true).await;
        let mut expected = vec![0.0; output.len()];
        for tile in 0..FULL_TILES as usize {
            for col in 0..FULL_COLS as usize {
                let output_index = tile * FULL_COLS as usize + col;
                for row in 0..CHUNK_ROWS as usize {
                    let index = (tile * CHUNK_ROWS as usize + row) * FULL_COLS as usize + col;
                    expected[output_index] +=
                        first[index] * (0.5 + tile as f32 / 128.0 + row as f32 / 64.0);
                    expected[output_index] +=
                        second[index] * (0.75 + tile as f32 / 128.0 + row as f32 / 32.0);
                }
            }
        }
        *task_result.lock().unwrap() = Some((output, expected));
    });
    executor.enter(Instant::ETERNITY).await;
    let (output, expected) = result.lock().unwrap().take().unwrap();
    assert_close(&output, &expected);
}

#[tokio::test]
async fn full_mamba_recurrence_uses_four_ltile_execs_across_four_tokens() {
    set_timing_mode(TimingMode::Serial);
    let executor = Executor::new();
    let result = Arc::new(Mutex::new(None));
    let task_result = result.clone();
    executor.spawn(async move {
        *task_result.lock().unwrap() = Some((
            run_full_mamba_ltile(false).await,
            run_full_mamba_ltile(true).await,
        ));
    });
    executor.enter(Instant::ETERNITY).await;
    let (fixed, affine) = result.lock().unwrap().take().unwrap();
    eprintln!(
        "Mamba full recurrence: fixed cycles={} stalls={}, affine cycles={} stalls={}",
        fixed.cycles,
        fixed.matrix.bank_stall_cycles,
        affine.cycles,
        affine.matrix.bank_stall_cycles,
    );
    let (expected_output, expected_state) = full_mamba_reference();
    assert_close(&fixed.output, &expected_output);
    assert_close(&fixed.state, &expected_state);
    assert_eq!(fixed.output, affine.output);
    assert_eq!(fixed.state, affine.state);
    assert!(fixed.matrix.bank_stall_cycles > 0);
    assert_eq!(affine.matrix.bank_stall_cycles, 0);
    assert!(fixed.cycles > affine.cycles);
}

#[tokio::test]
async fn full_kda_recurrence_uses_five_ltile_execs_across_four_tokens() {
    set_timing_mode(TimingMode::Serial);
    let executor = Executor::new();
    let result = Arc::new(Mutex::new(None));
    let task_result = result.clone();
    executor.spawn(async move {
        *task_result.lock().unwrap() = Some((
            run_full_kda_ltile(false).await,
            run_full_kda_ltile(true).await,
        ));
    });
    executor.enter(Instant::ETERNITY).await;
    let (fixed, affine) = result.lock().unwrap().take().unwrap();
    eprintln!(
        "KDA full recurrence: fixed cycles={} stalls={}, affine cycles={} stalls={}",
        fixed.cycles,
        fixed.matrix.bank_stall_cycles,
        affine.cycles,
        affine.matrix.bank_stall_cycles,
    );
    let (expected_output, expected_state) = full_kda_reference();
    assert_close(&fixed.output, &expected_output);
    assert_close(&fixed.state, &expected_state);
    assert_eq!(fixed.output, affine.output);
    assert_eq!(fixed.state, affine.state);
    assert!(fixed.matrix.bank_stall_cycles > 0);
    assert_eq!(affine.matrix.bank_stall_cycles, 0);
    assert!(fixed.cycles > affine.cycles);
}

#[tokio::test]
async fn mamba_hbm_to_affine_matrix_to_hbm_is_numerically_connected() {
    let executor = Executor::new();
    let result = Arc::new(Mutex::new(None));
    let task_result = result.clone();
    executor.spawn(async move {
        *task_result.lock().unwrap() = Some((
            run_hbm_connected_mamba(false, false).await,
            run_hbm_connected_mamba(true, false).await,
            run_hbm_connected_mamba(true, true).await,
        ));
    });
    executor.enter(Instant::ETERNITY).await;
    let (fixed, affine, pipelined) = result.lock().unwrap().take().unwrap();
    let (expected_output, expected_state) = full_mamba_reference();

    assert_close(&fixed.output, &expected_output);
    assert_close(&fixed.state, &expected_state);
    assert_eq!(fixed.output, affine.output);
    assert_eq!(fixed.state, affine.state);
    assert_eq!(affine.output, pipelined.output);
    assert_eq!(affine.state, pipelined.state);
    assert!(fixed.matrix.bank_stall_cycles > 0);
    assert_eq!(affine.matrix.bank_stall_cycles, 0);
    assert!(fixed.cycles > affine.cycles);
}

#[tokio::test]
async fn kda_hbm_to_affine_matrix_to_hbm_is_numerically_connected() {
    let executor = Executor::new();
    let result = Arc::new(Mutex::new(None));
    let task_result = result.clone();
    executor.spawn(async move {
        *task_result.lock().unwrap() = Some((
            run_hbm_connected_kda(false, false).await,
            run_hbm_connected_kda(true, false).await,
            run_hbm_connected_kda(true, true).await,
        ));
    });
    executor.enter(Instant::ETERNITY).await;
    let (fixed, affine, pipelined) = result.lock().unwrap().take().unwrap();
    let (expected_output, expected_state) = full_kda_reference();

    assert_close(&fixed.output, &expected_output);
    assert_close(&fixed.state, &expected_state);
    assert_eq!(fixed.output, affine.output);
    assert_eq!(fixed.state, affine.state);
    assert_eq!(affine.output, pipelined.output);
    assert_eq!(affine.state, pipelined.state);
    assert!(fixed.matrix.bank_stall_cycles > 0);
    assert_eq!(affine.matrix.bank_stall_cycles, 0);
    assert!(fixed.cycles > affine.cycles);
}
