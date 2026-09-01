use quantize::{tensor_from_f32_slice, tensor_to_f32_vec, DataType, MxDataType, QuantTensor};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::oneshot::{Receiver, Sender};
use tokio::sync::Mutex;

use crate::{addr_to_cell, Cell};

/// Logical Matrix-SRAM view interpreted by the physical bank mapper.
///
/// `tile_pitch_rows` is measured in physical rows inside each bank.  The
/// arithmetic operation is deliberately absent: this structure describes
/// placement only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixLayout {
    pub rows: u32,
    pub cols: u32,
    pub tile_count: u32,
    pub tile_pitch_rows: u32,
    /// Compiler-selected skew for this logical tensor view.
    pub alpha: u32,
}

/// Logical direction serviced from the same physical Matrix-SRAM cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixAccessAxis {
    Row,
    Column,
}

/// One physical location in a banked Matrix SRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatrixPhysicalCoord {
    pub bank: u32,
    pub bank_row: u32,
    pub lane: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MatrixPacketService {
    pub values: u64,
    pub bank_words: u64,
    pub ideal_cycles: u64,
    pub service_cycles: u64,
    pub bank_stall_cycles: u64,
    pub worst_bank_words: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MatrixPacketCounterSnapshot {
    pub packets: u64,
    pub values: u64,
    pub bank_words: u64,
    pub ideal_cycles: u64,
    pub service_cycles: u64,
    pub bank_stall_cycles: u64,
}

#[derive(Default)]
struct MatrixPacketCounters {
    packets: AtomicU64,
    values: AtomicU64,
    bank_words: AtomicU64,
    ideal_cycles: AtomicU64,
    service_cycles: AtomicU64,
    bank_stall_cycles: AtomicU64,
}

struct PendingWord {
    sender: Sender<QuantTensor>,
    tile: usize,
    row: usize,
    word: usize,
}

/// Opaque completion handle returned when a Matrix DMA parks physical words.
pub struct PendingMatrixTiles {
    words: Vec<PendingWord>,
}

/// Physically banked Matrix SRAM.
///
/// A cell is one `bank_width`-element bank word, not a whole matrix tile.  A
/// logical tile is reconstructed through the affine placement function on
/// read.  Consequently a wrong view changes values, not merely a timing
/// counter, which is the evidence needed for lane-restoration correctness.
pub struct MatrixSram {
    tile_size: u32,
    /// Number of MLEN-wide logical rows physically implemented.
    depth_rows: usize,
    /// Complete square tiles addressable by the legacy Matrix APIs.
    full_tile_count: usize,
    ty: MxDataType,
    element_type: DataType,
    banks: u32,
    bank_width: u32,
    alpha: u32,
    gamma: u32,
    rows_per_tile: u32,
    bank_rows: Vec<Vec<Mutex<Cell<Vec<u8>>>>>,
    packet_counters: MatrixPacketCounters,
}

impl MatrixSram {
    /// Backwards-compatible constructor: one scalar lane per physical bank.
    pub fn new(tile_size: u32, depth: usize, ty: MxDataType) -> Self {
        Self::with_banks_and_map(tile_size, depth, 1, 1, 0, ty)
    }

    /// Construct the Matrix SRAM with `tile_size / bank_width` physical banks.
    pub fn with_banks(tile_size: u32, depth: usize, bank_width: u32, ty: MxDataType) -> Self {
        Self::with_banks_and_map(tile_size, depth, bank_width, 1, 0, ty)
    }

    /// Construct a fixed-wiring control point (`alpha`, `gamma`) for D'.
    pub fn with_banks_and_map(
        tile_size: u32,
        depth: usize,
        bank_width: u32,
        alpha: u32,
        gamma: u32,
        ty: MxDataType,
    ) -> Self {
        assert!(tile_size > 0);
        assert!(bank_width > 0);
        assert!(tile_size.is_multiple_of(bank_width));
        assert!(
            depth > 0,
            "Matrix SRAM must contain at least one physical row"
        );
        let banks = tile_size / bank_width;
        assert!(
            banks.is_power_of_two(),
            "Matrix bank count must be a power of two"
        );
        assert!(banks <= 64, "Matrix view skew has a 6-bit bank contract");
        let element_type = match ty {
            MxDataType::Plain(data_type) => data_type,
            MxDataType::Mx { .. } => {
                panic!("Matrix SRAM bank words require a plain element type")
            }
        };
        assert!(
            (bank_width as usize * element_type.size_in_bits() as usize).is_multiple_of(8),
            "a Matrix bank word must contain whole bytes"
        );

        let full_tile_count = depth / tile_size as usize;
        let words_per_row = tile_size / bank_width;
        let rows_per_tile = tile_size * words_per_row.div_ceil(banks);
        // MATRIX_SRAM_SIZE is measured in MLEN-wide rows.  Views may use a
        // proper subset of those rows even when the legacy square-tile API
        // cannot fit one complete MLEN x MLEN tile.
        let physical_rows = depth;
        let bytes_per_word =
            (bank_width as usize * element_type.size_in_bits() as usize).div_ceil(8);
        let bank_rows = (0..banks)
            .map(|_| {
                (0..physical_rows)
                    .map(|_| Mutex::new(Cell::Ready(vec![0_u8; bytes_per_word])))
                    .collect()
            })
            .collect();

        Self {
            tile_size,
            depth_rows: depth,
            full_tile_count,
            ty,
            element_type,
            banks,
            bank_width,
            alpha: alpha % banks,
            gamma: gamma % banks,
            rows_per_tile,
            bank_rows,
            packet_counters: MatrixPacketCounters::default(),
        }
    }

    pub fn tile_size(&self) -> u32 {
        self.tile_size
    }

    pub fn depth_rows(&self) -> usize {
        self.depth_rows
    }

    pub fn ty(&self) -> MxDataType {
        self.ty
    }

    pub fn banks(&self) -> u32 {
        self.banks
    }

    pub fn bank_width(&self) -> u32 {
        self.bank_width
    }

    pub fn fixed_map(&self) -> (u32, u32) {
        (self.alpha, self.gamma)
    }

    /// Actual byte capacity of all physical bank words.
    pub fn size_in_bytes(&self) -> usize {
        let bits =
            self.depth_rows * self.tile_size as usize * self.element_type.size_in_bits() as usize;
        bits.div_ceil(8)
    }

    pub fn default_layout(&self) -> MatrixLayout {
        MatrixLayout {
            rows: self.tile_size,
            cols: self.tile_size,
            tile_count: 1,
            tile_pitch_rows: self.rows_per_tile,
            alpha: self.alpha,
        }
    }

    pub fn packet_counter_snapshot(&self) -> MatrixPacketCounterSnapshot {
        MatrixPacketCounterSnapshot {
            packets: self.packet_counters.packets.load(Ordering::Relaxed),
            values: self.packet_counters.values.load(Ordering::Relaxed),
            bank_words: self.packet_counters.bank_words.load(Ordering::Relaxed),
            ideal_cycles: self.packet_counters.ideal_cycles.load(Ordering::Relaxed),
            service_cycles: self.packet_counters.service_cycles.load(Ordering::Relaxed),
            bank_stall_cycles: self
                .packet_counters
                .bank_stall_cycles
                .load(Ordering::Relaxed),
        }
    }

    pub fn reset_packet_counters(&self) {
        self.packet_counters.packets.store(0, Ordering::Relaxed);
        self.packet_counters.values.store(0, Ordering::Relaxed);
        self.packet_counters.bank_words.store(0, Ordering::Relaxed);
        self.packet_counters
            .ideal_cycles
            .store(0, Ordering::Relaxed);
        self.packet_counters
            .service_cycles
            .store(0, Ordering::Relaxed);
        self.packet_counters
            .bank_stall_cycles
            .store(0, Ordering::Relaxed);
    }

    pub fn physical_coord(
        &self,
        addr: u32,
        layout: MatrixLayout,
        tile: u32,
        row: u32,
        col: u32,
    ) -> MatrixPhysicalCoord {
        self.validate_layout(addr, layout);
        assert!(
            tile < layout.tile_count,
            "Matrix-view tile index out of bounds"
        );
        assert!(row < layout.rows, "Matrix-view row out of bounds");
        assert!(col < layout.cols, "Matrix-view column out of bounds");

        let full_row_elements = self.banks * self.bank_width;
        assert!(
            addr.is_multiple_of(full_row_elements),
            "Matrix-view base address {addr} is not aligned to a physical row"
        );
        let base_bank_row = addr / full_row_elements;
        let word = col / self.bank_width;
        let words_per_row = layout.cols / self.bank_width;
        let row_groups = words_per_row.div_ceil(self.banks);
        let bank_row =
            base_bank_row + tile * layout.tile_pitch_rows + row * row_groups + word / self.banks;
        // The address already contains the allocation base, tile pitch, row,
        // and wide-row word group.  Using the tile-local `row` here discards
        // that information and falsely credits a per-tile offset with restoring
        // it.  The programmable term is the tensor view's skew (`alpha`).
        let bank =
            (layout.alpha * bank_row + self.gamma * (bank_row / self.banks) + word) % self.banks;
        assert!(
            (bank_row as usize) < self.bank_rows[bank as usize].len(),
            "Matrix-view physical row {bank_row} exceeds bank capacity"
        );
        MatrixPhysicalCoord {
            bank,
            bank_row,
            lane: col % self.bank_width,
        }
    }

    pub async fn read(&self, addr: u32) -> QuantTensor {
        self.read_layout_tile(addr, self.default_layout(), 0).await
    }

    pub async fn write(&self, addr: u32, tensor: QuantTensor) {
        self.write_layout_tile(addr, self.default_layout(), 0, tensor)
            .await;
    }

    /// Read one logical tile through a configured placement and restore lanes.
    pub async fn read_layout_tile(
        &self,
        addr: u32,
        layout: MatrixLayout,
        tile: u32,
    ) -> QuantTensor {
        self.validate_layout(addr, layout);
        let mut logical = vec![0_f32; (layout.rows * layout.cols) as usize];
        let words_per_row = layout.cols / self.bank_width;
        for row in 0..layout.rows {
            for word in 0..words_per_row {
                let col = word * self.bank_width;
                let coord = self.physical_coord(addr, layout, tile, row, col);
                let bytes = {
                    let mut guard = self.bank_rows[coord.bank as usize][coord.bank_row as usize]
                        .lock()
                        .await;
                    guard
                        .resolve_with(|tensor| self.tensor_to_word_bytes(&tensor))
                        .await
                        .clone()
                };
                let values = self.word_bytes_to_values(&bytes);
                let start = (row * layout.cols + col) as usize;
                logical[start..start + self.bank_width as usize].copy_from_slice(&values);
            }
        }
        QuantTensor::quantize(tensor_from_f32_slice(&logical), self.ty)
    }

    /// Read one logical row or column from the same physical cells.
    ///
    /// A bank contributes at most one word per cycle.  Column reads select one
    /// lane from each returned bank word and restore logical row order; there
    /// is no transposed copy or hidden transpose buffer.
    pub async fn read_layout_line(
        &self,
        addr: u32,
        layout: MatrixLayout,
        tile: u32,
        index: u32,
        axis: MatrixAccessAxis,
    ) -> (QuantTensor, MatrixPacketService) {
        self.validate_layout(addr, layout);
        let positions = match axis {
            MatrixAccessAxis::Row => {
                assert!(index < layout.rows, "Matrix-view row out of bounds");
                (0..layout.cols)
                    .step_by(self.bank_width as usize)
                    .map(|col| (index, col, true))
                    .collect::<Vec<_>>()
            }
            MatrixAccessAxis::Column => {
                assert!(index < layout.cols, "Matrix-view column out of bounds");
                (0..layout.rows)
                    .map(|row| (row, index, false))
                    .collect::<Vec<_>>()
            }
        };

        let mut logical = Vec::with_capacity(match axis {
            MatrixAccessAxis::Row => layout.cols as usize,
            MatrixAccessAxis::Column => layout.rows as usize,
        });
        let mut per_bank = vec![0_u64; self.banks as usize];
        for (row, col, whole_word) in positions {
            let word_col = col - col % self.bank_width;
            let coord = self.physical_coord(addr, layout, tile, row, word_col);
            let bytes = {
                let mut guard = self.bank_rows[coord.bank as usize][coord.bank_row as usize]
                    .lock()
                    .await;
                guard
                    .resolve_with(|tensor| self.tensor_to_word_bytes(&tensor))
                    .await
                    .clone()
            };
            let values = self.word_bytes_to_values(&bytes);
            if whole_word {
                logical.extend(values);
            } else {
                logical.push(values[coord.lane as usize]);
            }
            per_bank[coord.bank as usize] += 1;
        }
        let service = self.packet_service(&per_bank, logical.len() as u64);
        self.record_packet(service);
        (
            QuantTensor::quantize(tensor_from_f32_slice(&logical), self.ty),
            service,
        )
    }

    /// Reconstruct a complete tile while accounting for row- or column-wise
    /// service.  The returned tensor is always logical row-major data.
    pub async fn read_layout_tile_axis(
        &self,
        addr: u32,
        layout: MatrixLayout,
        tile: u32,
        axis: MatrixAccessAxis,
    ) -> (QuantTensor, MatrixPacketService) {
        let mut logical = vec![0_f32; (layout.rows * layout.cols) as usize];
        let mut total = MatrixPacketService::default();
        let lines = match axis {
            MatrixAccessAxis::Row => layout.rows,
            MatrixAccessAxis::Column => layout.cols,
        };
        for index in 0..lines {
            let (line, service) = self.read_layout_line(addr, layout, tile, index, axis).await;
            let values = tensor_to_f32_vec(line.as_tensor());
            match axis {
                MatrixAccessAxis::Row => {
                    let start = (index * layout.cols) as usize;
                    logical[start..start + layout.cols as usize].copy_from_slice(&values);
                }
                MatrixAccessAxis::Column => {
                    for (row, value) in values.into_iter().enumerate() {
                        logical[row * layout.cols as usize + index as usize] = value;
                    }
                }
            }
            total.values += service.values;
            total.bank_words += service.bank_words;
            total.ideal_cycles += service.ideal_cycles;
            total.service_cycles += service.service_cycles;
            total.bank_stall_cycles += service.bank_stall_cycles;
            total.worst_bank_words = total.worst_bank_words.max(service.worst_bank_words);
        }
        (
            QuantTensor::quantize(tensor_from_f32_slice(&logical), self.ty),
            total,
        )
    }

    /// Write one logical tile through the affine mapper into physical banks.
    pub async fn write_layout_tile(
        &self,
        addr: u32,
        layout: MatrixLayout,
        tile: u32,
        tensor: QuantTensor,
    ) {
        self.validate_layout(addr, layout);
        let mut logical = tensor_to_f32_vec(tensor.as_tensor());
        let expected = (layout.rows * layout.cols) as usize;
        assert!(
            logical.len() <= expected,
            "Matrix tile is larger than its view"
        );
        logical.resize(expected, 0.0);
        let words_per_row = layout.cols / self.bank_width;
        for row in 0..layout.rows {
            for word in 0..words_per_row {
                let col = word * self.bank_width;
                let coord = self.physical_coord(addr, layout, tile, row, col);
                let start = (row * layout.cols + col) as usize;
                let bytes =
                    self.values_to_word_bytes(&logical[start..start + self.bank_width as usize]);
                *self.bank_rows[coord.bank as usize][coord.bank_row as usize]
                    .lock()
                    .await = Cell::Ready(bytes);
            }
        }
    }

    /// Read every tile in one logical packet and restore tile/row/column order.
    ///
    /// One bank word per bank may be served each cycle. The returned service
    /// record is calculated from the same physical coordinates that supplied
    /// the returned values, so timing and correctness cannot drift apart.
    pub async fn read_layout_packet(
        &self,
        addr: u32,
        layout: MatrixLayout,
    ) -> (QuantTensor, MatrixPacketService) {
        let (packet, per_bank) = self.read_layout_packet_raw(addr, layout).await;
        let service = self.packet_service(&per_bank, packet.as_tensor().numel() as u64);
        self.record_packet(service);
        (packet, service)
    }

    /// Read several logical operands in one Vector issue slot.
    ///
    /// Their bank loads are combined before service time is calculated. This is
    /// what distinguishes a genuinely same-cycle cross-field access from a list
    /// of independent one-packet microbenchmarks.
    pub async fn read_layout_packets(
        &self,
        requests: &[(u32, MatrixLayout)],
    ) -> (Vec<QuantTensor>, MatrixPacketService) {
        assert!(
            !requests.is_empty(),
            "a Matrix packet group cannot be empty"
        );
        let mut packets = Vec::with_capacity(requests.len());
        let mut per_bank = vec![0_u64; self.banks as usize];
        let mut values = 0_u64;
        for &(addr, layout) in requests {
            let (packet, loads) = self.read_layout_packet_raw(addr, layout).await;
            values += packet.as_tensor().numel() as u64;
            for (total, load) in per_bank.iter_mut().zip(loads) {
                *total += load;
            }
            packets.push(packet);
        }
        let service = self.packet_service(&per_bank, values);
        self.record_packet_count(service, requests.len() as u64);
        (packets, service)
    }

    /// Scatter one tile-major packet through the same affine map used by read.
    pub async fn write_layout_packet(
        &self,
        addr: u32,
        layout: MatrixLayout,
        tensor: QuantTensor,
    ) -> MatrixPacketService {
        self.validate_layout(addr, layout);
        let expected = (layout.tile_count * layout.rows * layout.cols) as usize;
        let values = tensor_to_f32_vec(tensor.as_tensor());
        assert_eq!(
            values.len(),
            expected,
            "Matrix packet contains {} values, view requires {expected}",
            values.len()
        );
        let words_per_row = layout.cols / self.bank_width;
        let mut per_bank = vec![0_u64; self.banks as usize];
        for tile in 0..layout.tile_count {
            for row in 0..layout.rows {
                for word in 0..words_per_row {
                    let col = word * self.bank_width;
                    let coord = self.physical_coord(addr, layout, tile, row, col);
                    let start = ((tile * layout.rows + row) * layout.cols + col) as usize;
                    let bytes =
                        self.values_to_word_bytes(&values[start..start + self.bank_width as usize]);
                    *self.bank_rows[coord.bank as usize][coord.bank_row as usize]
                        .lock()
                        .await = Cell::Ready(bytes);
                    per_bank[coord.bank as usize] += 1;
                }
            }
        }
        let bank_words = per_bank.iter().sum::<u64>();
        let service_cycles = per_bank.iter().copied().max().unwrap_or(0);
        let ideal_cycles = bank_words.div_ceil(u64::from(self.banks));
        let service = MatrixPacketService {
            values: values.len() as u64,
            bank_words,
            ideal_cycles,
            service_cycles,
            bank_stall_cycles: service_cycles.saturating_sub(ideal_cycles),
            worst_bank_words: service_cycles,
        };
        self.record_packet(service);
        service
    }

    /// Write one dense microtile into a logical view at `logical_offset`.
    ///
    /// This is the direct Matrix-accumulator writeback path. The offset is in
    /// logical elements, so physical placement remains entirely a property of
    /// the configured view. One accumulator row is exactly one physical bank
    /// word; keeping that invariant explicit avoids hiding a crossbar or a
    /// read-modify-write in the timing model.
    pub async fn write_layout_microtile(
        &self,
        addr: u32,
        layout: MatrixLayout,
        logical_offset: u32,
        tensor: QuantTensor,
        micro_rows: u32,
        micro_cols: u32,
    ) -> MatrixPacketService {
        self.validate_layout(addr, layout);
        assert_eq!(
            micro_cols, self.bank_width,
            "Matrix accumulator row must equal one Matrix bank word"
        );
        let values = tensor_to_f32_vec(tensor.as_tensor());
        assert_eq!(values.len(), (micro_rows * micro_cols) as usize);
        let tile_values = layout.rows * layout.cols;
        let total_values = tile_values * layout.tile_count;
        assert!(logical_offset < total_values);

        let start_in_tile = logical_offset % tile_values;
        let start_col = start_in_tile % layout.cols;
        assert!(start_col.is_multiple_of(self.bank_width));
        assert!(start_col + micro_cols <= layout.cols);

        let mut per_bank = vec![0_u64; self.banks as usize];
        for micro_row in 0..micro_rows {
            let flat = logical_offset + micro_row * layout.cols;
            assert!(flat + micro_cols <= total_values);
            let tile = flat / tile_values;
            let within = flat % tile_values;
            let row = within / layout.cols;
            let col = within % layout.cols;
            assert!(row < layout.rows);
            assert_eq!(col, start_col, "microtile may not wrap a logical row");
            let coord = self.physical_coord(addr, layout, tile, row, col);
            assert_eq!(coord.lane, 0);
            let value_start = (micro_row * micro_cols) as usize;
            let bytes =
                self.values_to_word_bytes(&values[value_start..value_start + micro_cols as usize]);
            *self.bank_rows[coord.bank as usize][coord.bank_row as usize]
                .lock()
                .await = Cell::Ready(bytes);
            per_bank[coord.bank as usize] += 1;
        }

        let bank_words = per_bank.iter().sum::<u64>();
        let service_cycles = per_bank.iter().copied().max().unwrap_or(0);
        let ideal_cycles = bank_words.div_ceil(u64::from(self.banks));
        let service = MatrixPacketService {
            values: values.len() as u64,
            bank_words,
            ideal_cycles,
            service_cycles,
            bank_stall_cycles: service_cycles.saturating_sub(ideal_cycles),
            worst_bank_words: service_cycles,
        };
        self.record_packet(service);
        service
    }

    /// Preserve the historical odd address divisor of this legacy API.
    pub async fn write_delayed(&self, addr: u32, tensor: Receiver<QuantTensor>) {
        assert!(
            self.full_tile_count > 0,
            "legacy Matrix tile API requires at least MLEN physical rows"
        );
        let index = addr_to_cell(addr, self.tile_size, self.full_tile_count);
        let tensor = tensor.await.expect("delayed Matrix write sender dropped");
        self.write(index as u32 * self.tile_size * self.tile_size, tensor)
            .await;
    }

    /// Mark every physical bank word in `cells` default-layout tiles pending.
    pub async fn mark_pending_tiles(&self, addr: u32, cells: u32) -> PendingMatrixTiles {
        assert!(
            self.full_tile_count > 0,
            "legacy Matrix tile API requires at least MLEN physical rows"
        );
        let start_tile = addr_to_cell(addr, self.tile_size * self.tile_size, self.full_tile_count);
        let count = (cells as usize).min(self.full_tile_count.saturating_sub(start_tile));
        let layout = self.default_layout();
        let mut pending = Vec::with_capacity(count * self.tile_size as usize * self.banks as usize);
        for tile in 0..count {
            for row in 0..self.tile_size as usize {
                for word in 0..self.banks as usize {
                    let col = word as u32 * self.bank_width;
                    let coord = self.physical_coord(addr, layout, tile as u32, row as u32, col);
                    let (sender, receiver) = tokio::sync::oneshot::channel();
                    *self.bank_rows[coord.bank as usize][coord.bank_row as usize]
                        .lock()
                        .await = Cell::Pending(receiver);
                    pending.push(PendingWord {
                        sender,
                        tile,
                        row,
                        word,
                    });
                }
            }
        }
        PendingMatrixTiles { words: pending }
    }

    pub async fn fill_pending(&self, pending: PendingMatrixTiles, tensor: Receiver<QuantTensor>) {
        let tensor = tensor
            .await
            .unwrap_or_else(|error| panic!("delayed Matrix fill sender dropped: {error}"));
        let values = tensor_to_f32_vec(tensor.as_tensor());
        let tile_elements = (self.tile_size * self.tile_size) as usize;
        for word in pending.words {
            let source = word.tile * tile_elements
                + word.row * self.tile_size as usize
                + word.word * self.bank_width as usize;
            let mut padded = vec![0_f32; self.bank_width as usize];
            if source < values.len() {
                let end = (source + self.bank_width as usize).min(values.len());
                padded[..end - source].copy_from_slice(&values[source..end]);
            }
            let quantized = QuantTensor::quantize(tensor_from_f32_slice(&padded), self.ty);
            let _ = word.sender.send(quantized);
        }
    }

    pub async fn continous_write_delayed(
        &self,
        addr: u32,
        write_amount: u32,
        tensor: Receiver<QuantTensor>,
    ) {
        let tensor = tensor
            .await
            .unwrap_or_else(|error| panic!("delayed Matrix write sender dropped: {error}"));
        let values = tensor_to_f32_vec(tensor.as_tensor());
        let tile_elements = (self.tile_size * self.tile_size) as usize;
        let count = (write_amount as usize)
            .min(values.len().div_ceil(tile_elements))
            .min(self.full_tile_count);
        for tile in 0..count {
            let start = tile * tile_elements;
            let end = (start + tile_elements).min(values.len());
            let mut padded = vec![0_f32; tile_elements];
            padded[..end - start].copy_from_slice(&values[start..end]);
            let quantized = QuantTensor::quantize(tensor_from_f32_slice(&padded), self.ty);
            self.write(
                addr + tile as u32 * self.tile_size * self.tile_size,
                quantized,
            )
            .await;
        }
    }

    pub async fn as_bytes(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(self.size_in_bytes());
        // Export the complete physical capacity in default logical row order,
        // including a final partial-square region.  This keeps dump size equal
        // to MATRIX_SRAM_SIZE * MLEN rather than silently dropping spare rows.
        for row in 0..self.depth_rows as u32 {
            for word in 0..self.banks {
                let bank = (self.alpha * row + self.gamma * (row / self.banks) + word) % self.banks;
                let bytes = {
                    let mut guard = self.bank_rows[bank as usize][row as usize].lock().await;
                    guard
                        .resolve_with(|tensor| self.tensor_to_word_bytes(&tensor))
                        .await
                        .clone()
                };
                result.extend_from_slice(&bytes);
            }
        }
        debug_assert_eq!(result.len(), self.size_in_bytes());
        result
    }

    fn validate_layout(&self, addr: u32, layout: MatrixLayout) {
        assert!(layout.rows > 0 && layout.cols > 0 && layout.tile_count > 0);
        assert!(layout.cols.is_multiple_of(self.bank_width));
        assert!(layout.alpha < self.banks);
        let words_per_row = layout.cols / self.bank_width;
        let minimum_pitch = layout.rows * words_per_row.div_ceil(self.banks);
        assert!(layout.tile_pitch_rows >= minimum_pitch);
        let full_row_elements = self.banks * self.bank_width;
        assert!(addr.is_multiple_of(full_row_elements));
        let base_bank_row = addr / full_row_elements;
        let final_row =
            base_bank_row + (layout.tile_count - 1) * layout.tile_pitch_rows + minimum_pitch;
        assert!(
            final_row <= self.bank_rows[0].len() as u32,
            "Matrix view exceeds SRAM capacity"
        );
    }

    async fn read_layout_packet_raw(
        &self,
        addr: u32,
        layout: MatrixLayout,
    ) -> (QuantTensor, Vec<u64>) {
        self.validate_layout(addr, layout);
        let mut logical =
            Vec::with_capacity((layout.tile_count * layout.rows * layout.cols) as usize);
        let words_per_row = layout.cols / self.bank_width;
        let mut per_bank = vec![0_u64; self.banks as usize];
        for tile in 0..layout.tile_count {
            for row in 0..layout.rows {
                for word in 0..words_per_row {
                    let col = word * self.bank_width;
                    let coord = self.physical_coord(addr, layout, tile, row, col);
                    let bytes = {
                        let mut guard = self.bank_rows[coord.bank as usize]
                            [coord.bank_row as usize]
                            .lock()
                            .await;
                        guard
                            .resolve_with(|tensor| self.tensor_to_word_bytes(&tensor))
                            .await
                            .clone()
                    };
                    logical.extend(self.word_bytes_to_values(&bytes));
                    per_bank[coord.bank as usize] += 1;
                }
            }
        }
        (
            QuantTensor::quantize(tensor_from_f32_slice(&logical), self.ty),
            per_bank,
        )
    }

    fn packet_service(&self, per_bank: &[u64], values: u64) -> MatrixPacketService {
        let bank_words = per_bank.iter().sum::<u64>();
        let service_cycles = per_bank.iter().copied().max().unwrap_or(0);
        let ideal_cycles = bank_words.div_ceil(u64::from(self.banks));
        MatrixPacketService {
            values,
            bank_words,
            ideal_cycles,
            service_cycles,
            bank_stall_cycles: service_cycles.saturating_sub(ideal_cycles),
            worst_bank_words: service_cycles,
        }
    }

    fn record_packet(&self, service: MatrixPacketService) {
        self.record_packet_count(service, 1);
    }

    fn record_packet_count(&self, service: MatrixPacketService, packets: u64) {
        self.packet_counters
            .packets
            .fetch_add(packets, Ordering::Relaxed);
        self.packet_counters
            .values
            .fetch_add(service.values, Ordering::Relaxed);
        self.packet_counters
            .bank_words
            .fetch_add(service.bank_words, Ordering::Relaxed);
        self.packet_counters
            .ideal_cycles
            .fetch_add(service.ideal_cycles, Ordering::Relaxed);
        self.packet_counters
            .service_cycles
            .fetch_add(service.service_cycles, Ordering::Relaxed);
        self.packet_counters
            .bank_stall_cycles
            .fetch_add(service.bank_stall_cycles, Ordering::Relaxed);
    }

    fn bytes_per_word(&self) -> usize {
        (self.bank_width as usize * self.element_type.size_in_bits() as usize).div_ceil(8)
    }

    fn tensor_to_word_bytes(&self, tensor: &QuantTensor) -> Vec<u8> {
        let mut values = tensor_to_f32_vec(tensor.as_tensor());
        values.resize(self.bank_width as usize, 0.0);
        values.truncate(self.bank_width as usize);
        self.values_to_word_bytes(&values)
    }

    fn values_to_word_bytes(&self, values: &[f32]) -> Vec<u8> {
        let mut bytes = vec![0_u8; self.bytes_per_word()];
        self.element_type.bytes_from_f32(values, &mut bytes);
        bytes
    }

    fn word_bytes_to_values(&self, bytes: &[u8]) -> Vec<f32> {
        let mut values = vec![0_f32; self.bank_width as usize];
        self.element_type
            .convert_bytes_to_f32_vec(bytes, &mut values);
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantize::{FpType, MxDataType};
    use tch::Tensor;
    use tokio::sync::oneshot;

    fn f32_plain() -> MxDataType {
        MxDataType::Plain(DataType::Fp(FpType::F32))
    }

    fn bf16_plain() -> MxDataType {
        MxDataType::Plain(DataType::Fp(FpType::BF16))
    }

    fn tile(ty: MxDataType, vals: &[f32]) -> QuantTensor {
        QuantTensor::new_assuming_quantized(Tensor::from_slice(vals), ty).unwrap()
    }

    #[test]
    fn test_matrix_new_dimensions() {
        let m = MatrixSram::new(2, 8, f32_plain());
        assert_eq!(m.tile_size(), 2);
        assert_eq!(m.banks(), 2);
        assert_eq!(m.bank_width(), 1);
        assert_eq!(m.size_in_bytes(), 64);
    }

    #[tokio::test]
    async fn test_matrix_write_read_roundtrip() {
        let ty = f32_plain();
        let m = MatrixSram::new(2, 8, ty);
        let qt = tile(ty, &[1.0, 2.0, 3.0, 4.0]);
        m.write(4, qt.clone()).await;
        let got = m.read(4).await;
        assert!(got.as_tensor().equal(qt.as_tensor()));
        assert_eq!(
            m.packet_counter_snapshot(),
            MatrixPacketCounterSnapshot::default()
        );
    }

    #[tokio::test]
    async fn paper_depth_is_mlen_wide_rows_and_supports_compact_views() {
        const MLEN: u32 = 2048;
        const DEPTH_ROWS: usize = 256;
        const BLEN: u32 = 32;
        let ty = bf16_plain();
        let m = MatrixSram::with_banks(MLEN, DEPTH_ROWS, BLEN, ty);
        assert_eq!(m.depth_rows(), DEPTH_ROWS);
        assert_eq!(m.size_in_bytes(), 1024 * 1024);

        let view = MatrixLayout {
            rows: 1,
            cols: 64,
            tile_count: 32,
            tile_pitch_rows: 1,
            alpha: 2,
        };
        let values = (0..MLEN)
            .map(|index| ((index % 127) as f32 - 63.0) / 16.0)
            .collect::<Vec<_>>();
        let input = QuantTensor::quantize(Tensor::from_slice(&values), ty);
        let expected = tensor_to_f32_vec(input.as_tensor());
        let write = m.write_layout_packet(0, view, input).await;
        let (output, read) = m.read_layout_packet(0, view).await;
        assert_eq!(tensor_to_f32_vec(output.as_tensor()), expected);
        assert_eq!((write.service_cycles, read.service_cycles), (1, 1));
        assert_eq!(read.bank_stall_cycles, 0);
    }

    #[tokio::test]
    async fn test_matrix_write_delayed_preserves_legacy_address_divisor() {
        let ty = f32_plain();
        let m = MatrixSram::new(2, 8, ty);
        let qt = tile(ty, &[5.0, 6.0, 7.0, 8.0]);
        let (tx, rx) = oneshot::channel();
        assert!(tx.send(qt.clone()).is_ok());
        m.write_delayed(2, rx).await;
        let got = m.read(4).await;
        assert!(got.as_tensor().equal(qt.as_tensor()));
    }

    #[tokio::test]
    async fn non_identity_skew_moves_physical_banks_and_roundtrips() {
        let ty = f32_plain();
        let m = MatrixSram::with_banks(4, 16, 1, ty);
        let view = MatrixLayout {
            rows: 4,
            cols: 4,
            tile_count: 1,
            tile_pitch_rows: 4,
            alpha: 1,
        };
        let values = (0..16).map(|v| v as f32 + 1.0).collect::<Vec<_>>();
        m.write_layout_tile(0, view, 0, tile(ty, &values)).await;
        let row_major = MatrixLayout { alpha: 0, ..view };
        assert_ne!(
            m.physical_coord(0, view, 0, 1, 0),
            m.physical_coord(0, row_major, 0, 1, 0)
        );
        let got = m.read_layout_tile(0, view, 0).await;
        assert_eq!(tensor_to_f32_vec(got.as_tensor()), values);
    }

    #[tokio::test]
    async fn wrong_skew_returns_wrong_values_not_only_extra_cycles() {
        let ty = f32_plain();
        let m = MatrixSram::with_banks(4, 16, 1, ty);
        let placed = MatrixLayout {
            rows: 4,
            cols: 4,
            tile_count: 1,
            tile_pitch_rows: 4,
            alpha: 1,
        };
        let wrong = MatrixLayout { alpha: 0, ..placed };
        let values = (0..16).map(|v| v as f32 + 1.0).collect::<Vec<_>>();
        m.write_layout_tile(0, placed, 0, tile(ty, &values)).await;
        let got = tensor_to_f32_vec(m.read_layout_tile(0, wrong, 0).await.as_tensor());
        assert_ne!(got, values);
    }

    #[tokio::test]
    async fn pending_dma_fills_physical_words() {
        let ty = f32_plain();
        let m = MatrixSram::with_banks(4, 16, 1, ty);
        let pending = m.mark_pending_tiles(0, 1).await;
        let values = (0..16).map(|v| v as f32 + 1.0).collect::<Vec<_>>();
        let (tx, rx) = oneshot::channel();
        assert!(tx.send(tile(ty, &values)).is_ok());
        m.fill_pending(pending, rx).await;
        assert_eq!(tensor_to_f32_vec(m.read(0).await.as_tensor()), values);
    }

    #[tokio::test]
    async fn per_view_skew_removes_cross_tile_bank_conflict_with_same_values() {
        let ty = f32_plain();
        let m = MatrixSram::with_banks(4, 16, 1, ty);
        let row_major = MatrixLayout {
            rows: 1,
            cols: 1,
            tile_count: 4,
            tile_pitch_rows: 1,
            alpha: 0,
        };
        let affine = MatrixLayout {
            alpha: 1,
            ..row_major
        };
        let values = vec![11.0, 22.0, 33.0, 44.0];

        let row_write = m.write_layout_packet(0, row_major, tile(ty, &values)).await;
        let (row_values, row_read) = m.read_layout_packet(0, row_major).await;
        assert_eq!(tensor_to_f32_vec(row_values.as_tensor()), values);
        assert_eq!((row_write.service_cycles, row_read.service_cycles), (4, 4));

        let affine_base = 4 * 4;
        let affine_write = m
            .write_layout_packet(affine_base, affine, tile(ty, &values))
            .await;
        let (affine_values, affine_read) = m.read_layout_packet(affine_base, affine).await;
        assert_eq!(tensor_to_f32_vec(affine_values.as_tensor()), values);
        assert_eq!(
            (affine_write.service_cycles, affine_read.service_cycles),
            (1, 1)
        );
        assert_eq!(affine_read.bank_stall_cycles, 0);
    }

    #[tokio::test]
    async fn diagonal_placement_serves_rows_and_columns_at_the_bank_floor() {
        let ty = f32_plain();
        let values = (0..16).map(|value| value as f32 + 1.0).collect::<Vec<_>>();
        let row_major = MatrixLayout {
            rows: 4,
            cols: 4,
            tile_count: 1,
            tile_pitch_rows: 4,
            alpha: 0,
        };
        let diagonal = MatrixLayout {
            alpha: 1,
            ..row_major
        };

        let row_sram = MatrixSram::with_banks_and_map(4, 16, 1, 1, 0, ty);
        row_sram
            .write_layout_tile(0, row_major, 0, tile(ty, &values))
            .await;
        let (row, row_service) = row_sram
            .read_layout_line(0, row_major, 0, 2, MatrixAccessAxis::Row)
            .await;
        let (column, column_service) = row_sram
            .read_layout_line(0, row_major, 0, 1, MatrixAccessAxis::Column)
            .await;
        assert_eq!(
            tensor_to_f32_vec(row.as_tensor()),
            vec![9.0, 10.0, 11.0, 12.0]
        );
        assert_eq!(
            tensor_to_f32_vec(column.as_tensor()),
            vec![2.0, 6.0, 10.0, 14.0]
        );
        assert_eq!(row_service.service_cycles, 1);
        assert_eq!(column_service.service_cycles, 4);
        assert_eq!(column_service.bank_stall_cycles, 3);

        let diagonal_sram = MatrixSram::with_banks_and_map(4, 16, 1, 1, 0, ty);
        diagonal_sram
            .write_layout_tile(0, diagonal, 0, tile(ty, &values))
            .await;
        let (row, row_service) = diagonal_sram
            .read_layout_line(0, diagonal, 0, 2, MatrixAccessAxis::Row)
            .await;
        let (column, column_service) = diagonal_sram
            .read_layout_line(0, diagonal, 0, 1, MatrixAccessAxis::Column)
            .await;
        assert_eq!(
            tensor_to_f32_vec(row.as_tensor()),
            vec![9.0, 10.0, 11.0, 12.0]
        );
        assert_eq!(
            tensor_to_f32_vec(column.as_tensor()),
            vec![2.0, 6.0, 10.0, 14.0]
        );
        assert_eq!(row_service.service_cycles, 1);
        assert_eq!(column_service.service_cycles, 1);
        assert_eq!(column_service.bank_stall_cycles, 0);

        let (by_row, _) = diagonal_sram
            .read_layout_tile_axis(0, diagonal, 0, MatrixAccessAxis::Row)
            .await;
        let (by_column, _) = diagonal_sram
            .read_layout_tile_axis(0, diagonal, 0, MatrixAccessAxis::Column)
            .await;
        assert_eq!(tensor_to_f32_vec(by_row.as_tensor()), values);
        assert_eq!(tensor_to_f32_vec(by_column.as_tensor()), values);
    }

    #[tokio::test]
    async fn kda_prefill_state_becomes_decode_state_by_column_view_not_transpose_copy() {
        const DIM: u32 = 8;
        let ty = f32_plain();
        let sram = MatrixSram::with_banks_and_map(DIM, 64, 1, 1, 0, ty);
        let view = MatrixLayout {
            rows: DIM,
            cols: DIM,
            tile_count: 1,
            tile_pitch_rows: DIM,
            alpha: 1,
        };
        // Logical storage is [value, key]. It is intentionally non-symmetric,
        // because Kimi's 128x128 real shape cannot detect an axis error by shape.
        let prefill = (0..DIM)
            .flat_map(|value| (0..DIM).map(move |key| (value * 100 + key * 3 + 1) as f32))
            .collect::<Vec<_>>();
        sram.write_layout_tile(0, view, 0, tile(ty, &prefill)).await;

        let mut decode = Vec::with_capacity((DIM * DIM) as usize);
        let mut total_service = 0;
        for key in 0..DIM {
            let (line, service) = sram
                .read_layout_line(0, view, 0, key, MatrixAccessAxis::Column)
                .await;
            assert_eq!(service.service_cycles, service.ideal_cycles);
            total_service += service.service_cycles;
            decode.extend(tensor_to_f32_vec(line.as_tensor()));
        }
        let expected = (0..DIM)
            .flat_map(|key| (0..DIM).map(move |value| (value * 100 + key * 3 + 1) as f32))
            .collect::<Vec<_>>();
        assert_eq!(decode, expected);
        assert_ne!(prefill, expected, "row/column mismatch must be observable");
        assert_eq!(total_service, u64::from(DIM));
    }

    async fn paper_packet_roundtrip(
        tiles: u32,
        values_per_tile: u32,
        affine_alpha: u32,
        fixed_cycles: u64,
    ) {
        const MLEN: u32 = 2048;
        const BLEN: u32 = 32;
        let ty = bf16_plain();
        let values = (0..MLEN)
            .map(|index| ((index % 127) as f32 - 63.0) / 16.0)
            .collect::<Vec<_>>();
        let input = QuantTensor::quantize(Tensor::from_slice(&values), ty);
        let expected = tensor_to_f32_vec(input.as_tensor());
        // D' is one fixed map for every tensor in the machine. (alpha=2,
        // gamma=1) keeps an ordinary 128-value column read at its two-cycle
        // bank floor, reaches the Mamba floor, and is the best non-regressing
        // fixed point across the paper access families. D may instead select
        // alpha from each tensor's logical row width.
        let fixed = MatrixLayout {
            rows: 1,
            cols: values_per_tile,
            tile_count: tiles,
            tile_pitch_rows: 1,
            alpha: 2,
        };
        let affine = MatrixLayout {
            alpha: affine_alpha,
            ..fixed
        };

        // Both paths use the corrected physical row (allocation + tile pitch +
        // row + wide-row group). Only the compiler-selected skew changes.
        let fixed_sram = MatrixSram::with_banks_and_map(MLEN, 256, BLEN, 1, 1, ty);
        let fixed_write = fixed_sram
            .write_layout_packet(0, fixed, input.clone())
            .await;
        let (fixed_output, fixed_read) = fixed_sram.read_layout_packet(0, fixed).await;
        assert_eq!(tensor_to_f32_vec(fixed_output.as_tensor()), expected);
        assert_eq!(fixed_write.service_cycles, fixed_cycles);
        assert_eq!(fixed_read.service_cycles, fixed_cycles);
        assert_eq!(fixed_read.ideal_cycles, 1);

        let affine_sram = MatrixSram::with_banks_and_map(MLEN, 256, BLEN, 1, 1, ty);
        let affine_write = affine_sram.write_layout_packet(0, affine, input).await;
        let (affine_output, affine_read) = affine_sram.read_layout_packet(0, affine).await;
        assert_eq!(tensor_to_f32_vec(affine_output.as_tensor()), expected);
        assert_eq!(
            (affine_write.service_cycles, affine_read.service_cycles),
            (1, 1)
        );
        assert_eq!(affine_read.bank_stall_cycles, 0);
        assert_eq!(affine_read.bank_words, 64);
    }

    async fn paper_two_source_read(
        tiles: u32,
        values_per_tile: u32,
        affine_alpha: u32,
        fixed_cycles: u64,
    ) {
        const MLEN: u32 = 2048;
        const BLEN: u32 = 32;
        let ty = bf16_plain();
        let first = (0..MLEN)
            .map(|index| ((index % 113) as f32 - 56.0) / 8.0)
            .collect::<Vec<_>>();
        let second = (0..MLEN)
            .map(|index| ((index % 97) as f32 + 1.0) / 16.0)
            .collect::<Vec<_>>();
        let first = QuantTensor::quantize(Tensor::from_slice(&first), ty);
        let second = QuantTensor::quantize(Tensor::from_slice(&second), ty);
        let expected = [
            tensor_to_f32_vec(first.as_tensor()),
            tensor_to_f32_vec(second.as_tensor()),
        ];
        let fixed = MatrixLayout {
            rows: 1,
            cols: values_per_tile,
            tile_count: tiles,
            tile_pitch_rows: 1,
            alpha: 2,
        };
        let affine = MatrixLayout {
            alpha: affine_alpha,
            ..fixed
        };
        let second_base = tiles * MLEN;

        for (layout, expected_cycles) in [(fixed, fixed_cycles), (affine, 2)] {
            let sram = MatrixSram::with_banks_and_map(MLEN, 256, BLEN, 1, 1, ty);
            sram.write_layout_packet(0, layout, first.clone()).await;
            sram.write_layout_packet(second_base, layout, second.clone())
                .await;
            sram.reset_packet_counters();
            let (operands, service) = sram
                .read_layout_packets(&[(0, layout), (second_base, layout)])
                .await;
            assert_eq!(operands.len(), 2);
            assert_eq!(tensor_to_f32_vec(operands[0].as_tensor()), expected[0]);
            assert_eq!(tensor_to_f32_vec(operands[1].as_tensor()), expected[1]);
            assert_eq!(service.service_cycles, expected_cycles);
            assert_eq!(service.ideal_cycles, 2);
            assert_eq!(service.bank_stall_cycles, expected_cycles - 2);
            assert_eq!(service.values, 2 * u64::from(MLEN));
        }
    }

    async fn paper_projection_fragments_feed_consumer_view(
        tiles: u32,
        values_per_tile: u32,
        alpha: u32,
    ) {
        const MLEN: u32 = 2048;
        const BLEN: u32 = 32;
        let ty = bf16_plain();
        let values = (0..MLEN)
            .map(|index| ((index % 109) as f32 - 54.0) / 8.0)
            .collect::<Vec<_>>();
        let input = QuantTensor::quantize(Tensor::from_slice(&values), ty);
        let expected = tensor_to_f32_vec(input.as_tensor());
        let consumer = MatrixLayout {
            rows: 1,
            cols: values_per_tile,
            tile_count: tiles,
            tile_pitch_rows: 1,
            alpha,
        };
        let producer_only = MatrixLayout {
            rows: 1,
            cols: BLEN,
            tile_count: MLEN / BLEN,
            tile_pitch_rows: 1,
            alpha: 1,
        };
        let sram = MatrixSram::with_banks_and_map(MLEN, 256, BLEN, 1, 1, ty);

        for fragment in 0..MLEN / BLEN {
            let start = (fragment * BLEN) as usize;
            let word = QuantTensor::quantize(
                Tensor::from_slice(&expected[start..start + BLEN as usize]),
                ty,
            );
            let service = sram
                .write_layout_microtile(0, consumer, fragment * BLEN, word, 1, BLEN)
                .await;
            assert_eq!(service.service_cycles, 1);
        }

        let (output, service) = sram.read_layout_packet(0, consumer).await;
        assert_eq!(tensor_to_f32_vec(output.as_tensor()), expected);
        assert_eq!(service.service_cycles, 1);
        assert_eq!(service.bank_stall_cycles, 0);

        let wrong = tensor_to_f32_vec(
            sram.read_layout_packet(0, producer_only)
                .await
                .0
                .as_tensor(),
        );
        assert_ne!(wrong, expected);
    }

    #[tokio::test]
    async fn paper_nemotron_packet_is_one_cycle_for_global_and_per_view_maps() {
        // 32 Mamba heads x one 64-value state row. Each head contributes two
        // 32-value bank words, so the tensor-specific conflict-free alpha is 2.
        paper_packet_roundtrip(32, 64, 2, 1).await;
    }

    #[tokio::test]
    async fn paper_kimi_packet_is_two_cycles_global_and_one_cycle_per_view() {
        // 16 KDA heads x one 128-value state row. Each head contributes four
        // bank words, so the tensor-specific conflict-free alpha is 4.
        paper_packet_roundtrip(16, 128, 4, 2).await;
    }

    #[tokio::test]
    async fn paper_nemotron_two_source_read_reaches_two_cycle_floor() {
        paper_two_source_read(32, 64, 2, 2).await;
    }

    #[tokio::test]
    async fn paper_kimi_two_source_read_reaches_two_cycle_floor() {
        paper_two_source_read(16, 128, 4, 2).await;
    }

    #[tokio::test]
    async fn paper_nemotron_projection_fragments_fill_true_head_tiles() {
        paper_projection_fragments_feed_consumer_view(32, 64, 2).await;
    }

    #[tokio::test]
    async fn paper_kimi_projection_fragments_fill_true_head_tiles() {
        paper_projection_fragments_feed_consumer_view(16, 128, 4).await;
    }
}
