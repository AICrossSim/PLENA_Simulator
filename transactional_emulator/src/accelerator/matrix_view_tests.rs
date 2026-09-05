//! Matrix SRAM view, projection and L_TILE recurrence integration tests.
//!
//! Tests execute real banked storage and inspect state/output values, packet
//! service, and Serial/Scoreboard behavior using prepared BF16 operands.

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
use crate::vector_machine::VectorMachine;

const TOKENS: usize = 4;
const HBM_PACKET_STRIDE_BYTES: u32 = 8192;
const HBM_TEST_CAPACITY: usize = 1 << 20;

fn set_gp(rd: u8, value: u32) -> op::Opcode {
    op::Opcode::S_ADDI_INT {
        rd,
        rs1: 0,
        imm: value,
    }
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

async fn execute_oversized_ltile_line(primitive: op::LTilePrimitive, axis: op::LTileAxis) {
    set_timing_mode(TimingMode::Serial);
    let executor = Executor::new();
    executor.spawn(async move {
        let ty = MxDataType::Plain(DataType::Fp(FpType::BF16));
        let mram = Arc::new(MatrixSram::with_banks(64, 256, 4, ty));
        let vram = Arc::new(VectorSram::from_mx_type(64, 8, ty));
        let m_machine = MatrixMachine::new(mram, vram.clone(), 64, 16, 4, 4);
        let v_machine = VectorMachine::new(vram, 64, 16);
        let hbm: Arc<dyn ErasedMemoryModel> = Arc::new(WithTiming::new(
            NaiveTiming::preset_ddr4_2400p(4),
            MemoryBacked::with_capacity(4096),
        ));
        let mut accelerator = Accelerator::new(m_machine, v_machine, hbm);
        let (source_rows, source_cols) = match axis {
            op::LTileAxis::Row => (1, 128),
            op::LTileAxis::Column => (128, 4),
        };
        let mut ops = Vec::new();
        configure_ltile_view(&mut ops, 0, ltile_shape_word(1, 128, 1), 0);
        configure_ltile_view(
            &mut ops,
            1,
            ltile_shape_word(source_rows, source_cols, 1),
            0,
        );
        configure_ltile_view(
            &mut ops,
            2,
            ltile_shape_word(4, 4, 1),
            ltile_map_word(0, None, true),
        );
        ops.push(op::Opcode::L_TILE_EXEC {
            rd: 1,
            rs1: 2,
            rs2: 3,
            primitive,
            source_axis: axis,
            scale_axis: op::LTileAxis::Row,
        });
        accelerator.do_ops(&ops, None, TimingDriver::Serial).await;
    });
    executor.enter(Instant::ETERNITY).await;
}

#[tokio::test]
#[should_panic(expected = "L_TILE logical line width exceeds VLEN")]
async fn ltile_scale_accum_rejects_a_row_wider_than_vlen() {
    execute_oversized_ltile_line(op::LTilePrimitive::ScaleAccum, op::LTileAxis::Row).await;
}

#[tokio::test]
#[should_panic(expected = "L_TILE logical line width exceeds VLEN")]
async fn ltile_outer_update_rejects_a_row_wider_than_vlen() {
    execute_oversized_ltile_line(op::LTilePrimitive::OuterUpdate, op::LTileAxis::Row).await;
}

#[tokio::test]
#[should_panic(expected = "L_TILE logical line width exceeds VLEN")]
async fn ltile_dot_reduce_rejects_a_column_wider_than_vlen() {
    execute_oversized_ltile_line(op::LTilePrimitive::DotReduce, op::LTileAxis::Column).await;
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

#[derive(Debug)]
struct LTileResult {
    state: Vec<f32>,
    output: Vec<f32>,
    matrix: MatrixPacketCounterSnapshot,
}

async fn check_ltile_coefficient_packets(primitive: op::LTilePrimitive) {
    assert_eq!((*MLEN, *BLEN, *VLEN), (64, 4, 64));
    set_timing_mode(TimingMode::Serial);
    let executor = Executor::new();
    let results = Arc::new(Mutex::new(Vec::new()));
    let task_results = results.clone();
    executor.spawn(async move {
        // Three full single-tile packets; a full packet plus a one-tile tail;
        // and two full multi-tile packets. Different coefficients per tile
        // make the global compact offset observable in every case.
        for (cols, tiles) in [(64_u32, 3_u32), (32, 3), (32, 4)] {
            for compact in [true, false] {
                for shared_source in [false, true] {
                    let is_dot = matches!(primitive, op::LTilePrimitive::DotReduce);
                    if is_dot && shared_source {
                        // DOT_REDUCE requires one source tile per output tile.
                        continue;
                    }
                    let is_scale = matches!(primitive, op::LTilePrimitive::ScaleAccum);
                    let dst_rows = if is_dot { 1 } else { 2 };
                    let src_rows = if is_dot { 2 } else { 1 };
                    let src_tiles = if shared_source { 1 } else { tiles };
                    let scale_tiles = if compact { 1 } else { tiles };
                    let per_tile = if is_scale { 2 } else { 1 };
                    let scale_cols = if compact {
                        (per_tile * tiles).div_ceil(*BLEN) * *BLEN
                    } else {
                        *BLEN
                    };
                    let make_layout = |rows, cols, tile_count| MatrixLayout {
                        rows,
                        cols,
                        tile_count,
                        tile_pitch_rows: 2,
                        alpha: 1,
                        tile_skew: 0,
                    };
                    let dst_layout = make_layout(dst_rows, cols, tiles);
                    let src_layout = make_layout(src_rows, cols, src_tiles);
                    let scale_layout = make_layout(2, scale_cols, scale_tiles);
                    let source_value = |tile: u32, row: u32, col: u32| {
                        (1 + if shared_source { 0 } else { tile } + row + col % 3) as f32
                    };
                    let mut source = Vec::new();
                    for tile in 0..src_tiles {
                        for row in 0..src_rows {
                            for col in 0..cols {
                                source.push(source_value(tile, row, col));
                            }
                        }
                    }
                    let mut scales = vec![-16.0; (scale_tiles * 2 * scale_cols) as usize];
                    for tile in 0..tiles {
                        for row in 0..2 {
                            let index = if compact {
                                row * scale_cols + per_tile * tile
                            } else {
                                (tile * 2 + row) * scale_cols
                            } as usize;
                            if is_scale {
                                scales[index] = 0.5;
                                scales[index + 1] = (tile + row + 1) as f32;
                            } else {
                                scales[index] = (tile + row + 1) as f32;
                            }
                        }
                    }
                    let destination = vec![1.0; (tiles * dst_rows * cols) as usize];
                    let mram = Arc::new(MatrixSram::with_banks(
                        *MLEN,
                        *MLEN as usize * 64,
                        *BLEN,
                        full_state_type(),
                    ));
                    let vram = Arc::new(VectorSram::from_mx_type(*VLEN, 32, *VECTOR_SRAM_TYPE));
                    let mut ops = Vec::new();
                    for (slot, base, layout, values) in [
                        (0, 0, dst_layout, &destination),
                        (1, 1024, src_layout, &source),
                        (2, 2048, scale_layout, &scales),
                    ] {
                        mram.write_layout_packet(
                            base,
                            layout,
                            QuantTensor::quantize(Tensor::from_slice(values), full_state_type()),
                        )
                        .await;
                        configure_ltile_view(
                            &mut ops,
                            slot,
                            ltile_shape_word(layout.rows, layout.cols, layout.tile_count),
                            ltile_map_word(layout.tile_pitch_rows, None, slot == 2),
                        );
                    }
                    ops.extend([
                        set_gp(1, 0),
                        set_gp(2, 1024),
                        set_gp(3, 2048),
                        op::Opcode::L_TILE_EXEC {
                            rd: 1,
                            rs1: 2,
                            rs2: 3,
                            primitive,
                            source_axis: op::LTileAxis::Row,
                            scale_axis: op::LTileAxis::Row,
                        },
                    ]);
                    let mut accelerator = new_accelerator(mram.clone(), vram).await;
                    accelerator.do_ops(&ops, None, TimingDriver::Serial).await;
                    let actual = tensor_to_f32_vec(
                        mram.read_layout_packet(0, dst_layout).await.0.as_tensor(),
                    );
                    let mut expected = Vec::with_capacity(destination.len());
                    for tile in 0..tiles {
                        for row in 0..dst_rows {
                            for col in 0..cols {
                                let value = match primitive {
                                    op::LTilePrimitive::ScaleAccum => {
                                        0.5 + (tile + row + 1) as f32 * source_value(tile, 0, col)
                                    }
                                    op::LTilePrimitive::OuterUpdate => {
                                        1.0 + (tile + row + 1) as f32 * source_value(tile, 0, col)
                                    }
                                    op::LTilePrimitive::DotReduce => {
                                        1.0 + (0..src_rows)
                                            .map(|r| {
                                                (tile + r + 1) as f32 * source_value(tile, r, col)
                                            })
                                            .sum::<f32>()
                                    }
                                };
                                expected.push(value);
                            }
                        }
                    }
                    task_results.lock().unwrap().push((
                        format!(
                            "cols={cols}, tiles={tiles}, compact={compact}, shared={shared_source}"
                        ),
                        actual,
                        expected,
                    ));
                }
            }
        }
    });
    executor.enter(Instant::ETERNITY).await;
    let results = results.lock().unwrap();
    let expected_cases = if matches!(primitive, op::LTilePrimitive::DotReduce) {
        6
    } else {
        12
    };
    assert_eq!(results.len(), expected_cases);
    for (case, actual, expected) in results.iter() {
        // All inputs and outputs are BF16-exact; no tolerance can hide a
        // coefficient, tile, row, or broadcast selection error.
        assert_eq!(actual, expected, "{case}");
    }
}

#[tokio::test]
async fn l_tile_scale_accum_preserves_coefficient_layout_across_packets() {
    check_ltile_coefficient_packets(op::LTilePrimitive::ScaleAccum).await;
}

#[tokio::test]
async fn l_tile_dot_reduce_preserves_coefficient_layout_across_packets() {
    check_ltile_coefficient_packets(op::LTilePrimitive::DotReduce).await;
}

#[tokio::test]
async fn l_tile_outer_update_preserves_coefficient_layout_across_packets() {
    check_ltile_coefficient_packets(op::LTilePrimitive::OuterUpdate).await;
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
    let mapping = tile_pitch_rows;
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
            view_mask: 0b010,
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
    let mapping = tile_pitch_rows;
    let matrix_output_base = *MLEN * *MLEN;
    let vector_output_base = output_blocks * *BLEN * *VLEN;
    let mut ops = Vec::new();
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
        view_mask: 0b010,
    });

    let start = Executor::current().now();
    accelerator.do_ops(&ops, None, TimingDriver::Serial).await;
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
