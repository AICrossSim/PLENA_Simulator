//! Compiler-programmable views over PLENA's fixed-diagonal Matrix SRAM.
//!
//! A view is architectural placement metadata, not a cache or a traversal
//! engine.  Existing Matrix operations name one of four slots explicitly.
//! There is no implicit selection, replacement, auto-advance, or model state.

use super::Accelerator;
use crate::vector_machine::{TileScaleLayout, VectorBinaryOp};
use crate::{op, timing};
use quantize::{QuantTensor, tensor_from_f32_slice, tensor_to_f32_vec};
use sram::matrix::MatrixLayout;
use sram::matrix::MatrixPacketService;

const VIEW_SLOTS: usize = 4;
const DIM_MASK: u32 = (1 << 12) - 1;
const TILE_COUNT_MASK: u32 = (1 << 8) - 1;
const PITCH_MASK: u32 = (1 << 16) - 1;
const PHASE_MASK: u32 = (1 << 6) - 1;
const BROADCAST_MINOR: u8 = 1 << 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MatrixViewShape {
    pub(crate) rows: u32,
    pub(crate) cols: u32,
    pub(crate) tile_count: u32,
}

impl MatrixViewShape {
    pub(crate) fn unpack(word: u32) -> Self {
        Self {
            rows: (word & DIM_MASK) + 1,
            cols: ((word >> 12) & DIM_MASK) + 1,
            tile_count: ((word >> 24) & TILE_COUNT_MASK) + 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MatrixViewMap {
    /// Distance between consecutive logical tiles, measured in physical rows.
    pub(crate) tile_pitch_rows: u32,
    /// Compiler-selected bank phase stride between consecutive logical tiles.
    pub(crate) tile_phase_stride: u32,
    pub(crate) flags: u8,
}

impl MatrixViewMap {
    pub(crate) fn unpack(word: u32) -> Result<Self, String> {
        if (word >> 16) & PHASE_MASK != 0 {
            return Err("Matrix-view mapping bits [21:16] are reserved".into());
        }
        let mapping = Self {
            tile_pitch_rows: word & PITCH_MASK,
            tile_phase_stride: (word >> 22) & PHASE_MASK,
            flags: ((word >> 28) & 0xf) as u8,
        };
        if mapping.flags & !BROADCAST_MINOR != 0 {
            return Err(format!(
                "Matrix-view flags contain reserved bits: {:#x}",
                mapping.flags
            ));
        }
        Ok(mapping)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MatrixViewDescriptor {
    pub(crate) shape: MatrixViewShape,
    pub(crate) mapping: MatrixViewMap,
}

impl MatrixViewDescriptor {
    fn unpack(shape_word: u32, map_word: u32) -> Result<Self, String> {
        Ok(Self {
            shape: MatrixViewShape::unpack(shape_word),
            mapping: MatrixViewMap::unpack(map_word)?,
        })
    }

    fn validate(self, banks: u32, bank_width: u32) -> Result<Self, String> {
        if !banks.is_power_of_two() || banks > 64 {
            return Err(format!(
                "Matrix-view bank count must be a power of two in 1..=64, got {banks}"
            ));
        }
        if bank_width == 0 {
            return Err("Matrix-view bank width must be positive".into());
        }
        if !self.shape.cols.is_multiple_of(bank_width) {
            return Err(format!(
                "Matrix-view width {} is not a multiple of bank width {bank_width}",
                self.shape.cols
            ));
        }
        let words_per_row = self.shape.cols / bank_width;
        let row_groups = words_per_row.div_ceil(banks);
        let alpha = 1;
        let tile_phase_stride = self.mapping.tile_phase_stride;
        let mut occupied = std::collections::HashMap::new();
        for tile in 0..self.shape.tile_count {
            for row in 0..self.shape.rows {
                for word in 0..words_per_row {
                    let bank_row =
                        tile * self.mapping.tile_pitch_rows + row * row_groups + word / banks;
                    let bank = (alpha * bank_row + tile_phase_stride * tile + word) % banks;
                    if let Some(previous) = occupied.insert((bank, bank_row), (tile, row, word)) {
                        return Err(format!(
                            "Matrix view aliases logical bank words: {previous:?} and {:?} at bank={bank}, row={bank_row}",
                            (tile, row, word)
                        ));
                    }
                }
            }
        }
        Ok(self)
    }

    pub(crate) fn layout(self) -> MatrixLayout {
        MatrixLayout {
            rows: self.shape.rows,
            cols: self.shape.cols,
            tile_count: self.shape.tile_count,
            tile_pitch_rows: self.mapping.tile_pitch_rows,
            // The row term always uses PLENA's prior-work diagonal wiring.
            // Only the inter-tile phase is compiler selected.
            alpha: 1,
            tile_skew: self.mapping.tile_phase_stride,
        }
    }

    pub(crate) fn broadcast_minor(self) -> bool {
        self.mapping.flags & BROADCAST_MINOR != 0
    }

    pub(crate) fn values(self) -> u32 {
        self.shape
            .rows
            .checked_mul(self.shape.cols)
            .and_then(|value| value.checked_mul(self.shape.tile_count))
            .expect("validated Matrix-view dimensions overflowed u32")
    }
}

pub(super) struct MatrixViewTable {
    banks: u32,
    bank_width: u32,
    slots: [Option<MatrixViewDescriptor>; VIEW_SLOTS],
}

impl MatrixViewTable {
    pub(super) fn new(banks: u32, bank_width: u32) -> Self {
        assert!(banks.is_power_of_two());
        assert!(banks <= 64);
        assert!(bank_width > 0);
        Self {
            banks,
            bank_width,
            slots: [None; VIEW_SLOTS],
        }
    }

    pub(super) fn configure(
        &mut self,
        slot: u8,
        shape_word: u32,
        map_word: u32,
    ) -> Result<(), String> {
        let index = self.slot_index(slot)?;
        let descriptor = MatrixViewDescriptor::unpack(shape_word, map_word)?
            .validate(self.banks, self.bank_width)?;
        self.slots[index] = Some(descriptor);
        Ok(())
    }

    pub(super) fn get(&self, slot: u8) -> Result<MatrixViewDescriptor, String> {
        let index = self.slot_index(slot)?;
        self.slots[index].ok_or_else(|| format!("Matrix-view slot {slot} is not configured"))
    }

    fn slot_index(&self, slot: u8) -> Result<usize, String> {
        let index = usize::from(slot);
        if index >= VIEW_SLOTS {
            Err(format!(
                "Matrix-view slot {slot} is outside 0..{VIEW_SLOTS}"
            ))
        } else {
            Ok(index)
        }
    }
}

pub(super) struct MatrixViewBinaryArgs {
    pub(super) operation: VectorBinaryOp,
    pub(super) rd: u8,
    pub(super) rs1: u8,
    pub(super) rs2: u8,
    pub(super) rmask: u8,
    pub(super) view_mask: u8,
    pub(super) mask: u32,
    pub(super) pc: usize,
}

pub(super) struct LTileExecArgs {
    pub(super) destination_register: u8,
    pub(super) source_register: u8,
    pub(super) scale_register: u8,
    pub(super) primitive: op::LTilePrimitive,
    pub(super) source_axis: op::LTileAxis,
    pub(super) scale_axis: op::LTileAxis,
}

impl Accelerator {
    pub(super) fn resolve_matrix_view(
        &self,
        slot: Option<u8>,
        pc: usize,
    ) -> Option<MatrixViewDescriptor> {
        slot.map(|slot| {
            self.reg_file.matrix_view(slot).unwrap_or_else(|error| {
                tracing::error!(pc, slot, %error, "invalid Matrix-view consumer");
                panic!("{error} at pc {pc}")
            })
        })
    }

    pub(super) async fn read_l_tile_lines(
        &mut self,
        base: u32,
        view: MatrixViewDescriptor,
        axis: op::LTileAxis,
        lines: &[(u32, u32)],
    ) -> (QuantTensor, MatrixPacketService) {
        match axis {
            op::LTileAxis::Row => {
                self.m_machine
                    .mram
                    .read_layout_indexed_rows(base, view.layout(), lines)
                    .await
            }
            op::LTileAxis::Column => {
                self.m_machine
                    .mram
                    .read_layout_indexed_columns(base, view.layout(), lines)
                    .await
            }
        }
    }

    /// Decode the explicit Matrix-view operand marker carried by the VV
    /// family. Bits 0/1/2 select destination/source-1/source-2 slots. Keeping
    /// the marker in the instruction avoids inferring addressing semantics
    /// from whichever configuration registers happen to be live.
    pub(super) fn matrix_view_operand_mask(view_mask: u8) -> Option<u8> {
        assert!(view_mask < 8, "Matrix-view operand mask exceeds three bits");
        (view_mask != 0).then_some(view_mask)
    }

    pub(super) async fn vector_binary_with_matrix_views(&mut self, args: MatrixViewBinaryArgs) {
        let MatrixViewBinaryArgs {
            operation,
            rd,
            rs1,
            rs2,
            rmask,
            view_mask,
            mask,
            pc,
        } = args;
        let view_mask = Self::matrix_view_operand_mask(view_mask)
            .expect("Matrix-view vector helper requires the explicit marker");
        assert_ne!(view_mask, 0, "Matrix-view operand mask cannot be zero");

        let destination_view =
            (view_mask & 0b001 != 0).then(|| self.resolve_matrix_view(Some(0), pc).unwrap());
        let source1_view =
            (view_mask & 0b010 != 0).then(|| self.resolve_matrix_view(Some(1), pc).unwrap());
        let source2_view =
            (view_mask & 0b100 != 0).then(|| self.resolve_matrix_view(Some(2), pc).unwrap());

        for descriptor in [destination_view, source1_view, source2_view]
            .into_iter()
            .flatten()
        {
            assert_eq!(
                descriptor.values(),
                self.v_machine.tile_size(),
                "a Vector Matrix-view operand must restore exactly VLEN values"
            );
        }

        let mut requests = Vec::with_capacity(2);
        if let Some(view) = source1_view {
            requests.push((self.reg_file.read_gp(rs1), view.layout()));
        }
        if let Some(view) = source2_view {
            requests.push((self.reg_file.read_gp(rs2), view.layout()));
        }
        let matrix_packets = if requests.is_empty() {
            Vec::new()
        } else {
            let (packets, service) = self.m_machine.mram.read_layout_packets(&requests).await;
            timing::charge_bank_cycles(service.service_cycles.max(1)).await;
            packets
        };
        let mut matrix_packets = matrix_packets.into_iter();
        let lhs = if source1_view.is_some() {
            matrix_packets
                .next()
                .expect("missing Matrix source-1 packet")
        } else {
            self.v_machine.vram.read(self.reg_file.read_gp(rs1)).await
        };
        let rhs = if source2_view.is_some() {
            matrix_packets
                .next()
                .expect("missing Matrix source-2 packet")
        } else {
            self.v_machine.vram.read(self.reg_file.read_gp(rs2)).await
        };
        debug_assert!(matrix_packets.next().is_none());

        let result = self
            .v_machine
            .binary_packet(operation, lhs, rhs, rmask, mask)
            .await;
        if let Some(view) = destination_view {
            let service = self
                .m_machine
                .mram
                .write_layout_packet(self.reg_file.read_gp(rd), view.layout(), result)
                .await;
            timing::charge_bank_cycles(service.service_cycles.max(1)).await;
        } else {
            self.v_machine
                .vram
                .write(self.reg_file.read_gp(rd), result)
                .await;
        }
    }

    /// Execute one model-independent recurrence primitive over Matrix views.
    ///
    /// Views 0/1/2 are destination/source/scalars.  The decoder owns only a
    /// deterministic row/column walk; all bases, shapes and layouts remain
    /// compiler-visible architectural state. Matrix-view storage is BF16.
    pub(super) async fn execute_l_tile(&mut self, args: LTileExecArgs, pc: usize) {
        let LTileExecArgs {
            destination_register,
            source_register,
            scale_register,
            primitive,
            source_axis,
            scale_axis,
        } = args;
        let destination = self.resolve_matrix_view(Some(0), pc).unwrap();
        let source = self.resolve_matrix_view(Some(1), pc).unwrap();
        let scales = self.resolve_matrix_view(Some(2), pc).unwrap();
        let dst_base = self.reg_file.read_gp(destination_register);
        let src_base = self.reg_file.read_gp(source_register);
        let scale_base = self.reg_file.read_gp(scale_register);

        if !scales.broadcast_minor() {
            panic!("L_TILE scale view must set BROADCAST_MINOR");
        }
        if scales.shape.tile_count != 1 && scales.shape.tile_count != destination.shape.tile_count {
            panic!("L_TILE scale tiles must be one or match destination tiles");
        }
        if source.shape.tile_count != 1 && source.shape.tile_count != destination.shape.tile_count {
            panic!("L_TILE source tiles must be one or match destination tiles");
        }

        let dst_layout = destination.layout();
        let source_line_count = match source_axis {
            op::LTileAxis::Row => source.shape.rows,
            op::LTileAxis::Column => source.shape.cols,
        };
        let source_line_width = match source_axis {
            op::LTileAxis::Row => source.shape.cols,
            op::LTileAxis::Column => source.shape.rows,
        };
        let scale_line_count = match scale_axis {
            op::LTileAxis::Row => scales.shape.rows,
            op::LTileAxis::Column => scales.shape.cols,
        };
        let scale_line_width = match scale_axis {
            op::LTileAxis::Row => scales.shape.cols,
            op::LTileAxis::Column => scales.shape.rows,
        };

        // A recurrence line is serviced by one existing Vector operation.
        // Wider Matrix views remain legal for DMA/Matrix consumers, but this
        // controller does not split a logical line across multiple VLEN ops.
        assert!(
            source_line_width <= self.v_machine.tile_size(),
            "L_TILE logical line width exceeds VLEN; compiler must tile the columns"
        );

        match primitive {
            op::LTilePrimitive::ScaleAccum | op::LTilePrimitive::OuterUpdate => {
                if source_line_width != destination.shape.cols {
                    panic!("row-wise L_TILE source/destination widths differ");
                }
                if source_line_count != 1 && source_line_count != destination.shape.rows {
                    panic!("row-wise L_TILE source rows must be one or match destination");
                }
                if scale_line_count < destination.shape.rows {
                    panic!("L_TILE scale view has fewer logical lines than destination");
                }
                let tiles_per_packet = (self.v_machine.tile_size() / destination.shape.cols).max(1);
                for row in 0..destination.shape.rows {
                    for first_tile in
                        (0..destination.shape.tile_count).step_by(tiles_per_packet as usize)
                    {
                        let tile_count =
                            tiles_per_packet.min(destination.shape.tile_count - first_tile);
                        let scale_layout = if scales.shape.tile_count == 1 {
                            TileScaleLayout::Compact { first_tile }
                        } else {
                            TileScaleLayout::Expanded
                        };
                        let destination_lines = (first_tile..first_tile + tile_count)
                            .map(|tile| (tile, row))
                            .collect::<Vec<_>>();
                        let source_lines = destination_lines
                            .iter()
                            .map(|&(tile, destination_row)| {
                                (
                                    if source.shape.tile_count == 1 {
                                        0
                                    } else {
                                        tile
                                    },
                                    if source_line_count == 1 {
                                        0
                                    } else {
                                        destination_row
                                    },
                                )
                            })
                            .collect::<Vec<_>>();
                        let scale_lines = if scales.shape.tile_count == 1 {
                            // Compact per-segment scalars are fetched once,
                            // one cycle ahead of the all-bank state packet.
                            vec![(0, row)]
                        } else {
                            destination_lines
                                .iter()
                                .map(|&(tile, destination_row)| (tile, destination_row))
                                .collect::<Vec<_>>()
                        };

                        let (dst_packet, dst_service) = self
                            .m_machine
                            .mram
                            .read_layout_indexed_rows(dst_base, dst_layout, &destination_lines)
                            .await;
                        timing::charge_bank_cycles(dst_service.service_cycles.max(1)).await;
                        let (src_packet, src_service) = self
                            .read_l_tile_lines(src_base, source, source_axis, &source_lines)
                            .await;
                        timing::charge_bank_cycles(src_service.service_cycles.max(1)).await;
                        let (scale_packet, scale_service) = self
                            .read_l_tile_lines(scale_base, scales, scale_axis, &scale_lines)
                            .await;
                        // Scalar bank words are deliberately charged separately:
                        // a full state packet already consumes every bank word.
                        timing::charge_bank_cycles(scale_service.service_cycles.max(1)).await;

                        let result = match primitive {
                            op::LTilePrimitive::ScaleAccum => {
                                self.v_machine
                                    .tile_scale_accum(
                                        dst_packet,
                                        src_packet,
                                        scale_packet,
                                        destination.shape.cols,
                                        scale_line_width,
                                        scale_layout,
                                    )
                                    .await
                            }
                            op::LTilePrimitive::OuterUpdate => {
                                self.v_machine
                                    .tile_outer_update(
                                        dst_packet,
                                        src_packet,
                                        scale_packet,
                                        destination.shape.cols,
                                        scale_line_width,
                                        scale_layout,
                                    )
                                    .await
                            }
                            op::LTilePrimitive::DotReduce => unreachable!(),
                        };
                        let service = self
                            .m_machine
                            .mram
                            .write_layout_indexed_rows(
                                dst_base,
                                dst_layout,
                                &destination_lines,
                                result,
                            )
                            .await;
                        timing::charge_bank_cycles(service.service_cycles.max(1)).await;
                    }
                }
            }
            op::LTilePrimitive::DotReduce => {
                if destination.shape.rows != 1
                    || destination.shape.cols != source_line_width
                    || destination.shape.tile_count != source.shape.tile_count
                {
                    panic!("DOT_REDUCE destination must be one row per source tile");
                }
                if scale_line_count < source_line_count {
                    panic!("DOT_REDUCE scale view has fewer lines than reduction rows");
                }
                let tiles_per_packet = (self.v_machine.tile_size() / source_line_width).max(1);
                for first_tile in (0..source.shape.tile_count).step_by(tiles_per_packet as usize) {
                    let tile_count = tiles_per_packet.min(source.shape.tile_count - first_tile);
                    let scale_layout = if scales.shape.tile_count == 1 {
                        TileScaleLayout::Compact { first_tile }
                    } else {
                        TileScaleLayout::Expanded
                    };
                    let destination_lines = (first_tile..first_tile + tile_count)
                        .map(|tile| (tile, 0))
                        .collect::<Vec<_>>();
                    let (destination_packet, destination_service) = self
                        .m_machine
                        .mram
                        .read_layout_indexed_rows(dst_base, dst_layout, &destination_lines)
                        .await;
                    timing::charge_bank_cycles(destination_service.service_cycles.max(1)).await;
                    let mut accumulator = tensor_to_f32_vec(destination_packet.as_tensor());
                    assert_eq!(accumulator.len(), (tile_count * source_line_width) as usize);
                    for row in 0..source_line_count {
                        let source_lines = (first_tile..first_tile + tile_count)
                            .map(|tile| (tile, row))
                            .collect::<Vec<_>>();
                        let scale_lines = if scales.shape.tile_count == 1 {
                            vec![(0, row)]
                        } else {
                            source_lines
                                .iter()
                                .map(|&(tile, source_row)| (tile, source_row))
                                .collect::<Vec<_>>()
                        };
                        let (source_packet, source_service) = self
                            .read_l_tile_lines(src_base, source, source_axis, &source_lines)
                            .await;
                        timing::charge_bank_cycles(source_service.service_cycles.max(1)).await;
                        let (scale_packet, scale_service) = self
                            .read_l_tile_lines(scale_base, scales, scale_axis, &scale_lines)
                            .await;
                        timing::charge_bank_cycles(scale_service.service_cycles.max(1)).await;
                        self.v_machine
                            .tile_dot_accumulate(
                                &mut accumulator,
                                source_packet,
                                scale_packet,
                                source_line_width,
                                scale_line_width,
                                scale_layout,
                            )
                            .await;
                    }
                    let result = QuantTensor::quantize(
                        tensor_from_f32_slice(&accumulator),
                        self.m_machine.mram.ty(),
                    );
                    let service = self
                        .m_machine
                        .mram
                        .write_layout_indexed_rows(dst_base, dst_layout, &destination_lines, result)
                        .await;
                    timing::charge_bank_cycles(service.service_cycles.max(1)).await;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(rows: u32, cols: u32, tiles: u32) -> u32 {
        (rows - 1) | ((cols - 1) << 12) | ((tiles - 1) << 24)
    }

    fn mapping(pitch: u32) -> u32 {
        pitch
    }

    #[test]
    fn v3_mapping_words_match_the_python_contract() {
        assert_eq!(mapping(64), 0x0000_0040);
        assert_eq!(mapping(0) | (4 << 22), 0x0100_0000);
        assert_eq!(mapping(0) | (4 << 22) | (8 << 28), 0x8100_0000);
    }

    #[test]
    fn configuration_matches_the_python_contract() {
        let mut table = MatrixViewTable::new(16, 4);
        table.configure(2, shape(64, 64, 3), mapping(64)).unwrap();
        let view = table.get(2).unwrap();
        assert_eq!(
            view.shape,
            MatrixViewShape {
                rows: 64,
                cols: 64,
                tile_count: 3
            }
        );
        assert_eq!(view.mapping.tile_pitch_rows, 64);
        assert_eq!(view.mapping.tile_phase_stride, 0);
    }

    #[test]
    fn rejects_aliasing_pitch_and_reserved_mapping_bits() {
        let mut table = MatrixViewTable::new(16, 4);
        assert!(table.configure(0, shape(64, 64, 2), mapping(63)).is_err());
        assert!(
            table
                .configure(0, shape(64, 64, 2), mapping(64) | (1 << 16))
                .is_err()
        );
        let phased = mapping(64) | (5 << 22);
        table.configure(0, shape(64, 64, 2), phased).unwrap();
        let view = table.get(0).unwrap();
        assert_eq!((view.layout().alpha, view.layout().tile_skew), (1, 5));

        let programmable_row = mapping(64) | (3 << 16) | (5 << 22);
        assert!(
            table
                .configure(0, shape(64, 64, 2), programmable_row)
                .is_err()
        );
        for reserved_flag in [1_u32, 2, 4] {
            let invalid = mapping(64) | (5 << 22) | (reserved_flag << 28);
            assert!(table.configure(0, shape(64, 64, 2), invalid).is_err());
        }
    }

    #[test]
    fn zero_pitch_is_legal_only_when_tile_phase_prevents_aliasing() {
        let mut table = MatrixViewTable::new(64, 32);
        assert!(table.configure(0, shape(128, 128, 8), mapping(0)).is_err());
        let compact = mapping(0) | (4 << 22);
        table.configure(0, shape(128, 128, 8), compact).unwrap();
        let view = table.get(0).unwrap();
        assert_eq!(view.mapping.tile_pitch_rows, 0);
        assert_eq!((view.layout().alpha, view.layout().tile_skew), (1, 4));
    }
}
