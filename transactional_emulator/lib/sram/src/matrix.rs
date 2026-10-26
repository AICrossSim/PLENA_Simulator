use quantize::{tensor_from_f32_slice, tensor_to_f32_vec, DataType, MxDataType, QuantTensor};
use std::collections::{hash_map::Entry, HashMap};
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
    /// Bank skew used by the physical machine. Public Matrix views fix this to 1;
    /// tests may vary it only for a non-architectural upper-bound control.
    pub alpha: u32,
    /// Additional compiler-selected phase between logical tiles.
    pub tile_skew: u32,
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
    source_offset: usize,
}

/// Opaque completion handle returned when a Matrix DMA parks physical words.
pub struct PendingMatrixTiles {
    words: Vec<PendingWord>,
}

/// Physically banked Matrix SRAM.
///
/// A cell is one `bank_width`-element bank word, not a whole matrix tile.  A
/// logical tile is reconstructed through the fixed-diagonal placement function on
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
    rows_per_tile: u32,
    bank_rows: Vec<Vec<Mutex<Cell<Vec<u8>>>>>,
    packet_counters: MatrixPacketCounters,
}

impl MatrixSram {
    /// Backwards-compatible constructor with at most 64 physical banks.
    ///
    /// Small test geometries retain one scalar lane per bank.  Wider, paper-
    /// scale rows widen each bank word instead of exceeding the six-bit bank
    /// index used by Matrix views.
    pub fn new(tile_size: u32, depth: usize, ty: MxDataType) -> Self {
        assert!(
            tile_size.is_power_of_two(),
            "legacy Matrix tile size must be a power of two"
        );
        let bank_width = (tile_size / 64).max(1);
        Self::with_banks(tile_size, depth, bank_width, ty)
    }

    /// Construct the Matrix SRAM with fixed diagonal placement and
    /// `tile_size / bank_width` physical banks.
    pub fn with_banks(tile_size: u32, depth: usize, bank_width: u32, ty: MxDataType) -> Self {
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

    pub fn element_bits(&self) -> u32 {
        u32::from(self.element_type.size_in_bits())
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
            alpha: 1,
            tile_skew: 0,
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
        self.physical_coord_unchecked(addr, layout, tile, row, col)
    }

    fn physical_coord_unchecked(
        &self,
        addr: u32,
        layout: MatrixLayout,
        tile: u32,
        row: u32,
        col: u32,
    ) -> MatrixPhysicalCoord {
        assert!(
            tile < layout.tile_count,
            "Matrix-view tile index out of bounds"
        );
        assert!(row < layout.rows, "Matrix-view row out of bounds");
        assert!(col < layout.cols, "Matrix-view column out of bounds");

        let full_row_elements = self.banks * self.bank_width;
        assert!(
            addr.is_multiple_of(self.bank_width),
            "Matrix-view base address {addr} is not aligned to a bank word"
        );
        let base_bank_row = addr / full_row_elements;
        let base_bank = (addr % full_row_elements) / self.bank_width;
        let word = col / self.bank_width;
        let words_per_row = layout.cols / self.bank_width;
        let row_groups = words_per_row.div_ceil(self.banks);
        let bank_row =
            base_bank_row + tile * layout.tile_pitch_rows + row * row_groups + word / self.banks;
        // The address already contains the allocation base, tile pitch, row,
        // and wide-row word group. Using the tile-local `row` here discards
        // that information. ISA-visible Matrix views always use alpha=1.
        let bank =
            (base_bank + layout.alpha * bank_row + layout.tile_skew * tile + word) % self.banks;
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
        assert_eq!(
            tensor.data_type(),
            self.ty,
            "legacy Matrix write must match the SRAM data type"
        );
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
                let coord = self.physical_coord_unchecked(addr, layout, tile, row, col);
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
            let coord = self.physical_coord_unchecked(addr, layout, tile, row, word_col);
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
                logical.push(values[(col % self.bank_width) as usize]);
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

    /// Read several logical rows or columns in one bank-service packet.
    ///
    /// For column groups, lanes that share a physical bank word are fetched
    /// once and then restored to their logical columns.  The service record is
    /// derived from those exact physical words, so diagonal-read correctness
    /// and bank-conflict timing cannot diverge.
    pub async fn read_layout_lines(
        &self,
        addr: u32,
        layout: MatrixLayout,
        tile: u32,
        first: u32,
        count: u32,
        axis: MatrixAccessAxis,
    ) -> (QuantTensor, MatrixPacketService) {
        self.validate_layout(addr, layout);
        assert!(count > 0, "Matrix line packet must be non-empty");
        let limit = match axis {
            MatrixAccessAxis::Row => layout.rows,
            MatrixAccessAxis::Column => layout.cols,
        };
        assert!(
            first + count <= limit,
            "Matrix line packet exceeds its view"
        );

        let line_len = match axis {
            MatrixAccessAxis::Row => layout.cols,
            MatrixAccessAxis::Column => layout.rows,
        };
        let mut logical = vec![0_f32; (count * line_len) as usize];
        let mut per_bank = vec![0_u64; self.banks as usize];
        let mut words: HashMap<(u32, u32), Vec<f32>> = HashMap::new();

        for line_offset in 0..count {
            let line = first + line_offset;
            match axis {
                MatrixAccessAxis::Row => {
                    for col in (0..layout.cols).step_by(self.bank_width as usize) {
                        let coord = self.physical_coord_unchecked(addr, layout, tile, line, col);
                        let key = (coord.bank, coord.bank_row);
                        if let Entry::Vacant(entry) = words.entry(key) {
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
                            entry.insert(self.word_bytes_to_values(&bytes));
                            per_bank[coord.bank as usize] += 1;
                        }
                        let values = &words[&key];
                        let start = (line_offset * line_len + col) as usize;
                        logical[start..start + self.bank_width as usize].copy_from_slice(values);
                    }
                }
                MatrixAccessAxis::Column => {
                    for row in 0..layout.rows {
                        let word_col = line - line % self.bank_width;
                        let coord =
                            self.physical_coord_unchecked(addr, layout, tile, row, word_col);
                        let key = (coord.bank, coord.bank_row);
                        if let Entry::Vacant(entry) = words.entry(key) {
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
                            entry.insert(self.word_bytes_to_values(&bytes));
                            per_bank[coord.bank as usize] += 1;
                        }
                        logical[(line_offset * line_len + row) as usize] =
                            words[&key][(line % self.bank_width) as usize];
                    }
                }
            }
        }

        let service = self.packet_service(&per_bank, logical.len() as u64);
        self.record_packet(service);
        (
            QuantTensor::quantize(tensor_from_f32_slice(&logical), self.ty),
            service,
        )
    }

    /// Read an explicitly ordered packet of logical rows across one or more
    /// tiles.  `lines` contains `(tile, row)` pairs and the returned tensor is
    /// the concatenation of those rows in exactly that order.
    ///
    /// L-Tile recurrence packets use row-major tile order: for a fixed state
    /// index they request the same logical row from several head tiles.  Bank
    /// service is computed from the physical words that supplied those values,
    /// including de-duplication for an explicitly broadcast source row.
    pub async fn read_layout_indexed_rows(
        &self,
        addr: u32,
        layout: MatrixLayout,
        lines: &[(u32, u32)],
    ) -> (QuantTensor, MatrixPacketService) {
        self.validate_layout(addr, layout);
        assert!(
            !lines.is_empty(),
            "Matrix indexed-row packet must be non-empty"
        );

        let mut logical = vec![0_f32; lines.len() * layout.cols as usize];
        let mut per_bank = vec![0_u64; self.banks as usize];
        let mut words: HashMap<(u32, u32), Vec<f32>> = HashMap::new();

        for (line_index, &(tile, row)) in lines.iter().enumerate() {
            assert!(
                tile < layout.tile_count,
                "Matrix indexed-row tile out of bounds"
            );
            assert!(row < layout.rows, "Matrix indexed-row row out of bounds");
            for col in (0..layout.cols).step_by(self.bank_width as usize) {
                let coord = self.physical_coord_unchecked(addr, layout, tile, row, col);
                let key = (coord.bank, coord.bank_row);
                if let Entry::Vacant(entry) = words.entry(key) {
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
                    entry.insert(self.word_bytes_to_values(&bytes));
                    per_bank[coord.bank as usize] += 1;
                }
                let start = line_index * layout.cols as usize + col as usize;
                logical[start..start + self.bank_width as usize].copy_from_slice(&words[&key]);
            }
        }

        let service = self.packet_service(&per_bank, logical.len() as u64);
        self.record_packet(service);
        (
            QuantTensor::quantize(tensor_from_f32_slice(&logical), self.ty),
            service,
        )
    }

    /// Read an explicitly ordered packet of logical columns across tiles.
    ///
    /// This is the transpose counterpart of `read_layout_indexed_rows`.  It
    /// reads the same physical cells and returns `[requested_line][logical_row]`
    /// order.  A head-major `[head][key]` field can therefore feed a key-major
    /// recurrence packet without a copied transpose.
    pub async fn read_layout_indexed_columns(
        &self,
        addr: u32,
        layout: MatrixLayout,
        lines: &[(u32, u32)],
    ) -> (QuantTensor, MatrixPacketService) {
        self.validate_layout(addr, layout);
        assert!(
            !lines.is_empty(),
            "Matrix indexed-column packet must be non-empty"
        );

        let mut logical = vec![0_f32; lines.len() * layout.rows as usize];
        let mut per_bank = vec![0_u64; self.banks as usize];
        let mut words: HashMap<(u32, u32), Vec<f32>> = HashMap::new();

        for (line_index, &(tile, col)) in lines.iter().enumerate() {
            assert!(
                tile < layout.tile_count,
                "Matrix indexed-column tile out of bounds"
            );
            assert!(
                col < layout.cols,
                "Matrix indexed-column column out of bounds"
            );
            let word_col = col - col % self.bank_width;
            for row in 0..layout.rows {
                let coord = self.physical_coord_unchecked(addr, layout, tile, row, word_col);
                let key = (coord.bank, coord.bank_row);
                if let Entry::Vacant(entry) = words.entry(key) {
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
                    entry.insert(self.word_bytes_to_values(&bytes));
                    per_bank[coord.bank as usize] += 1;
                }
                logical[line_index * layout.rows as usize + row as usize] =
                    words[&key][(col % self.bank_width) as usize];
            }
        }

        let service = self.packet_service(&per_bank, logical.len() as u64);
        self.record_packet(service);
        (
            QuantTensor::quantize(tensor_from_f32_slice(&logical), self.ty),
            service,
        )
    }

    /// Write several logical rows from a lane-restored packet.
    pub async fn write_layout_rows(
        &self,
        addr: u32,
        layout: MatrixLayout,
        tile: u32,
        first_row: u32,
        row_count: u32,
        tensor: QuantTensor,
    ) -> MatrixPacketService {
        self.validate_layout(addr, layout);
        assert!(row_count > 0 && first_row + row_count <= layout.rows);
        let values = tensor_to_f32_vec(tensor.as_tensor());
        assert_eq!(values.len(), (row_count * layout.cols) as usize);
        let mut per_bank = vec![0_u64; self.banks as usize];
        for row_offset in 0..row_count {
            let row = first_row + row_offset;
            for col in (0..layout.cols).step_by(self.bank_width as usize) {
                let coord = self.physical_coord_unchecked(addr, layout, tile, row, col);
                let start = (row_offset * layout.cols + col) as usize;
                let bytes =
                    self.values_to_word_bytes(&values[start..start + self.bank_width as usize]);
                *self.bank_rows[coord.bank as usize][coord.bank_row as usize]
                    .lock()
                    .await = Cell::Ready(bytes);
                per_bank[coord.bank as usize] += 1;
            }
        }
        let service = self.packet_service(&per_bank, values.len() as u64);
        self.record_packet(service);
        service
    }

    /// Write an explicitly ordered packet of logical rows across head tiles.
    /// Distinct destination words are required to map to distinct physical
    /// cells; otherwise a purported layout silently aliases recurrent state.
    pub async fn write_layout_indexed_rows(
        &self,
        addr: u32,
        layout: MatrixLayout,
        lines: &[(u32, u32)],
        tensor: QuantTensor,
    ) -> MatrixPacketService {
        self.validate_layout(addr, layout);
        assert!(
            !lines.is_empty(),
            "Matrix indexed-row write must be non-empty"
        );
        let values = tensor_to_f32_vec(tensor.as_tensor());
        assert_eq!(values.len(), lines.len() * layout.cols as usize);
        let mut per_bank = vec![0_u64; self.banks as usize];
        let mut destinations = HashMap::new();

        for (line_index, &(tile, row)) in lines.iter().enumerate() {
            assert!(
                tile < layout.tile_count,
                "Matrix indexed-row tile out of bounds"
            );
            assert!(row < layout.rows, "Matrix indexed-row row out of bounds");
            for col in (0..layout.cols).step_by(self.bank_width as usize) {
                let coord = self.physical_coord_unchecked(addr, layout, tile, row, col);
                let key = (coord.bank, coord.bank_row);
                assert!(
                    destinations.insert(key, (tile, row, col)).is_none(),
                    "Matrix view aliases two logical destination words at bank {} row {}",
                    coord.bank,
                    coord.bank_row
                );
                let start = line_index * layout.cols as usize + col as usize;
                let bytes =
                    self.values_to_word_bytes(&values[start..start + self.bank_width as usize]);
                *self.bank_rows[coord.bank as usize][coord.bank_row as usize]
                    .lock()
                    .await = Cell::Ready(bytes);
                per_bank[coord.bank as usize] += 1;
            }
        }

        let service = self.packet_service(&per_bank, values.len() as u64);
        self.record_packet(service);
        service
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
                let coord = self.physical_coord_unchecked(addr, layout, tile, row, col);
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
                    let coord = self.physical_coord_unchecked(addr, layout, tile, row, col);
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
            let coord = self.physical_coord_unchecked(addr, layout, tile, row, col);
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
        let layout = MatrixLayout {
            tile_count: count as u32,
            ..self.default_layout()
        };
        let mut pending = Vec::with_capacity(count * self.tile_size as usize * self.banks as usize);
        for tile in 0..count {
            for row in 0..self.tile_size as usize {
                for word in 0..self.banks as usize {
                    let col = word as u32 * self.bank_width;
                    let coord =
                        self.physical_coord_unchecked(addr, layout, tile as u32, row as u32, col);
                    let (sender, receiver) = tokio::sync::oneshot::channel();
                    *self.bank_rows[coord.bank as usize][coord.bank_row as usize]
                        .lock()
                        .await = Cell::Pending(receiver);
                    pending.push(PendingWord {
                        sender,
                        source_offset: tile * self.tile_size as usize * self.tile_size as usize
                            + row * self.tile_size as usize
                            + word * self.bank_width as usize,
                    });
                }
            }
        }
        PendingMatrixTiles { words: pending }
    }

    /// Park every physical bank word selected by an explicit Matrix view.
    ///
    /// The completion handle preserves each word's logical packet offset, so
    /// the DMA result is scattered through the exact same affine map later
    /// consumed by `read_layout_packet`/`L_TILE_EXEC`.
    pub async fn mark_pending_layout_packet(
        &self,
        addr: u32,
        layout: MatrixLayout,
    ) -> (PendingMatrixTiles, MatrixPacketService) {
        self.validate_layout(addr, layout);
        let words_per_row = layout.cols / self.bank_width;
        let mut pending =
            Vec::with_capacity((layout.tile_count * layout.rows * words_per_row) as usize);
        let mut destinations = HashMap::new();
        let mut per_bank = vec![0_u64; self.banks as usize];
        for tile in 0..layout.tile_count {
            for row in 0..layout.rows {
                for word in 0..words_per_row {
                    let col = word * self.bank_width;
                    let coord = self.physical_coord_unchecked(addr, layout, tile, row, col);
                    assert!(
                        destinations
                            .insert((coord.bank, coord.bank_row), (tile, row, word))
                            .is_none(),
                        "Matrix view aliases pending DMA destinations"
                    );
                    let (sender, receiver) = tokio::sync::oneshot::channel();
                    *self.bank_rows[coord.bank as usize][coord.bank_row as usize]
                        .lock()
                        .await = Cell::Pending(receiver);
                    pending.push(PendingWord {
                        source_offset: ((tile * layout.rows + row) * layout.cols + col) as usize,
                        sender,
                    });
                    per_bank[coord.bank as usize] += 1;
                }
            }
        }
        let values = u64::from(layout.tile_count) * u64::from(layout.rows) * u64::from(layout.cols);
        let service = self.packet_service(&per_bank, values);
        self.record_packet(service);
        (PendingMatrixTiles { words: pending }, service)
    }

    pub async fn fill_pending(&self, pending: PendingMatrixTiles, tensor: Receiver<QuantTensor>) {
        let tensor = tensor
            .await
            .unwrap_or_else(|error| panic!("delayed Matrix fill sender dropped: {error}"));
        let values = tensor_to_f32_vec(tensor.as_tensor());
        for word in pending.words {
            let source = word.source_offset;
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
                let bank = (row + word) % self.banks;
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
        let row_groups = words_per_row.div_ceil(self.banks);
        let full_row_elements = self.banks * self.bank_width;
        assert!(addr.is_multiple_of(self.bank_width));
        let base_bank_row = addr / full_row_elements;
        let base_bank = (addr % full_row_elements) / self.bank_width;
        let mut occupied = HashMap::new();
        let mut final_row = base_bank_row;
        for tile in 0..layout.tile_count {
            for row in 0..layout.rows {
                for word in 0..words_per_row {
                    let bank_row = base_bank_row
                        + tile * layout.tile_pitch_rows
                        + row * row_groups
                        + word / self.banks;
                    let bank =
                        (base_bank + layout.alpha * bank_row + layout.tile_skew * tile + word)
                            % self.banks;
                    let previous = occupied.insert((bank, bank_row), (tile, row, word));
                    assert!(
                        previous.is_none(),
                        "Matrix view aliases logical words {:?} and {:?} at bank {bank}, row {bank_row}",
                        previous.unwrap_or_default(),
                        (tile, row, word),
                    );
                    final_row = final_row.max(bank_row + 1);
                }
            }
        }
        assert!(
            final_row <= self.bank_rows[0].len() as u32,
            "Matrix view exceeds SRAM capacity"
        );
    }

    /// Check that several compiler-managed views can coexist in this SRAM.
    ///
    /// A descriptor proves only that a tensor does not alias itself.  A static
    /// allocator must additionally prove that two live tensors never claim the
    /// same physical bank word.  This is deliberately not a cache check: there
    /// are no tags, misses or replacement decisions, only explicit addresses.
    pub fn validate_disjoint_layouts(
        &self,
        views: &[(&str, u32, MatrixLayout)],
    ) -> Result<(), String> {
        let mut occupied: HashMap<(u32, u32), (&str, u32, u32, u32)> = HashMap::new();
        for &(name, addr, layout) in views {
            self.validate_layout(addr, layout);
            let words_per_row = layout.cols / self.bank_width;
            for tile in 0..layout.tile_count {
                for row in 0..layout.rows {
                    for word in 0..words_per_row {
                        let coord = self.physical_coord_unchecked(
                            addr,
                            layout,
                            tile,
                            row,
                            word * self.bank_width,
                        );
                        let key = (coord.bank, coord.bank_row);
                        if let Some(previous) = occupied.insert(key, (name, tile, row, word)) {
                            return Err(format!(
                                "Matrix views alias physical bank word bank={}, row={}: {:?} and {:?}",
                                coord.bank,
                                coord.bank_row,
                                previous,
                                (name, tile, row, word),
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
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
                    let coord = self.physical_coord_unchecked(addr, layout, tile, row, col);
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

    #[test]
    fn default_constructor_supports_paper_mlen() {
        let m = MatrixSram::new(2048, 256, bf16_plain());
        assert_eq!(m.banks(), 64);
        assert_eq!(m.bank_width(), 32);
        assert_eq!(m.size_in_bytes(), 1024 * 1024);
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
    #[should_panic(expected = "legacy Matrix write must match the SRAM data type")]
    async fn legacy_write_rejects_a_mismatched_element_type() {
        let m = MatrixSram::new(2, 8, bf16_plain());
        m.write(0, tile(f32_plain(), &[1.0, 2.0, 3.0, 4.0])).await;
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
            tile_skew: 0,
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
            tile_skew: 0,
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
            tile_skew: 0,
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
    async fn pending_dma_fills_multiple_tiles() {
        let ty = f32_plain();
        let m = MatrixSram::with_banks(4, 16, 1, ty);
        let pending = m.mark_pending_tiles(0, 2).await;
        let values = (0..32).map(|value| value as f32 + 1.0).collect::<Vec<_>>();
        let (tx, rx) = oneshot::channel();
        assert!(tx.send(tile(ty, &values)).is_ok());
        m.fill_pending(pending, rx).await;

        assert_eq!(tensor_to_f32_vec(m.read(0).await.as_tensor()), values[..16]);
        assert_eq!(
            tensor_to_f32_vec(m.read(16).await.as_tensor()),
            values[16..]
        );
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
            tile_skew: 0,
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
    async fn bank_word_aligned_base_supplies_a_constant_field_phase() {
        let ty = f32_plain();
        let sram = MatrixSram::with_banks(16, 32, 4, ty);
        let view = MatrixLayout {
            rows: 1,
            cols: 4,
            tile_count: 1,
            tile_pitch_rows: 1,
            alpha: 1,
            tile_skew: 0,
        };
        let unphased = sram.physical_coord(0, view, 0, 0, 0);
        let phased = sram.physical_coord(3 * sram.bank_width(), view, 0, 0, 0);
        assert_eq!(unphased.bank_row, phased.bank_row);
        assert_eq!(phased.bank, (unphased.bank + 3) % sram.banks());

        let left = tile(ty, &[1.0, 2.0, 3.0, 4.0]);
        let right = tile(ty, &[5.0, 6.0, 7.0, 8.0]);
        sram.write_layout_packet(0, view, left).await;
        sram.write_layout_packet(3 * sram.bank_width(), view, right)
            .await;
        assert_eq!(
            tensor_to_f32_vec(sram.read_layout_packet(0, view).await.0.as_tensor()),
            [1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            tensor_to_f32_vec(
                sram.read_layout_packet(3 * sram.bank_width(), view)
                    .await
                    .0
                    .as_tensor()
            ),
            [5.0, 6.0, 7.0, 8.0]
        );
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
            tile_skew: 0,
        };
        let diagonal = MatrixLayout {
            alpha: 1,
            ..row_major
        };

        let row_sram = MatrixSram::with_banks(4, 16, 1, ty);
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

        let diagonal_sram = MatrixSram::with_banks(4, 16, 1, ty);
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
    async fn column_reads_restore_every_lane_for_real_bank_widths() {
        const MLEN: u32 = 32;
        const ROWS: u32 = 8;
        let ty = bf16_plain();
        let values = (0..ROWS * MLEN)
            .map(|index| index as f32)
            .collect::<Vec<_>>();

        for bank_width in [1, 4, 32] {
            let banks = MLEN / bank_width;
            let sram = MatrixSram::with_banks(MLEN, 64, bank_width, ty);
            let view = MatrixLayout {
                rows: ROWS,
                cols: MLEN,
                tile_count: 1,
                tile_pitch_rows: ROWS,
                alpha: u32::from(banks > 1),
                tile_skew: 0,
            };
            sram.write_layout_tile(0, view, 0, tile(ty, &values)).await;

            for col in 0..MLEN {
                let (line, _) = sram
                    .read_layout_line(0, view, 0, col, MatrixAccessAxis::Column)
                    .await;
                let expected = (0..ROWS)
                    .map(|row| values[(row * MLEN + col) as usize])
                    .collect::<Vec<_>>();
                assert_eq!(
                    tensor_to_f32_vec(line.as_tensor()),
                    expected,
                    "column {col} failed at bank_width={bank_width}"
                );
            }
        }
    }

    #[tokio::test]
    async fn f32_matrix_words_roundtrip_with_multiple_lanes() {
        const MLEN: u32 = 16;
        let ty = f32_plain();
        let sram = MatrixSram::with_banks(MLEN, 32, 4, ty);
        let view = MatrixLayout {
            rows: 4,
            cols: MLEN,
            tile_count: 1,
            tile_pitch_rows: 4,
            alpha: 1,
            tile_skew: 0,
        };
        let values = (0..4 * MLEN).map(|index| index as f32).collect::<Vec<_>>();
        sram.write_layout_tile(0, view, 0, tile(ty, &values)).await;
        let got = sram.read_layout_tile(0, view, 0).await;
        assert_eq!(tensor_to_f32_vec(got.as_tensor()), values);
    }

    #[tokio::test]
    async fn kda_prefill_state_becomes_decode_state_by_column_view_not_transpose_copy() {
        const DIM: u32 = 8;
        let ty = f32_plain();
        let sram = MatrixSram::with_banks(DIM, 64, 1, ty);
        let view = MatrixLayout {
            rows: DIM,
            cols: DIM,
            tile_count: 1,
            tile_pitch_rows: DIM,
            alpha: 1,
            tile_skew: 0,
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

    fn official_recurrent_layout(
        rows: u32,
        cols: u32,
        tiles: u32,
        bank_width: u32,
    ) -> MatrixLayout {
        let words_per_row = cols / bank_width;
        MatrixLayout {
            rows,
            cols,
            tile_count: tiles,
            // Both the tile phase and the phase seen by an equal-row packet
            // must traverse every bank-word group.  Twice the row width is the
            // smallest pitch that satisfies both constraints for the official
            // power-of-two Mamba/KDA shapes.
            tile_pitch_rows: 2 * words_per_row,
            alpha: 1,
            tile_skew: words_per_row,
        }
    }

    async fn official_recurrent_group_roundtrip(cols: u32, tiles: u32, expected_last_row: u32) {
        const MLEN: u32 = 2048;
        const DEPTH_ROWS: usize = 256;
        const STATE_ROWS: u32 = 128;
        const BF16_VALUES_PER_BANK_WORD: u32 = 32;

        let ty = bf16_plain();
        let sram = MatrixSram::with_banks(MLEN, DEPTH_ROWS, BF16_VALUES_PER_BANK_WORD, ty);
        let view = official_recurrent_layout(STATE_ROWS, cols, tiles, BF16_VALUES_PER_BANK_WORD);
        let final_row = (tiles - 1) * view.tile_pitch_rows + STATE_ROWS;
        assert_eq!(final_row, expected_last_row);
        assert!(final_row <= DEPTH_ROWS as u32);

        let value_count = tiles * STATE_ROWS * cols;
        let values = (0..value_count)
            .map(|index| ((index % 257) as f32 - 128.0) / 64.0)
            .collect::<Vec<_>>();
        let input = QuantTensor::quantize(Tensor::from_slice(&values), ty);
        let expected = tensor_to_f32_vec(input.as_tensor());
        let write = sram.write_layout_packet(0, view, input).await;
        assert_eq!(write.bank_stall_cycles, 0);

        // This is the recurrence access: the same logical state row from all
        // heads in the group must fill all 64 bank words exactly once.
        for row in [0, STATE_ROWS / 2, STATE_ROWS - 1] {
            let lines = (0..tiles).map(|tile| (tile, row)).collect::<Vec<_>>();
            let (packet, service) = sram.read_layout_indexed_rows(0, view, &lines).await;
            assert_eq!(service.ideal_cycles, 1);
            assert_eq!(service.service_cycles, 1);
            assert_eq!(service.bank_stall_cycles, 0);
            let got = tensor_to_f32_vec(packet.as_tensor());
            let expected_row = (0..tiles)
                .flat_map(|tile| {
                    let start = ((tile * STATE_ROWS + row) * cols) as usize;
                    expected[start..start + cols as usize].iter().copied()
                })
                .collect::<Vec<_>>();
            assert_eq!(got, expected_row);
        }

        let (roundtrip, read) = sram.read_layout_packet(0, view).await;
        assert_eq!(tensor_to_f32_vec(roundtrip.as_tensor()), expected);
        assert_eq!(read.bank_stall_cycles, 0);

        // A *single-base descriptor* with no per-tile phase can hold only two
        // full-height heads. The stronger fixed-wiring D' control is tested
        // separately below and must not be confused with this ISA limitation.
        let single_descriptor_required_rows = STATE_ROWS * tiles;
        assert!(single_descriptor_required_rows > DEPTH_ROWS as u32);
        assert_eq!(DEPTH_ROWS as u32 / STATE_ROWS, 2);
    }

    async fn fixed_phased_official_state_roundtrip(cols: u32, tiles: u32) {
        const MLEN: u32 = 2048;
        const DEPTH_ROWS: usize = 256;
        const STATE_ROWS: u32 = 128;
        const BF16_VALUES_PER_BANK_WORD: u32 = 32;

        let ty = bf16_plain();
        let sram = MatrixSram::with_banks(MLEN, DEPTH_ROWS, BF16_VALUES_PER_BANK_WORD, ty);
        let words_per_head = cols / BF16_VALUES_PER_BANK_WORD;
        assert_eq!(tiles * words_per_head, 64);
        let fixed = MatrixLayout {
            rows: STATE_ROWS,
            cols,
            tile_count: 1,
            tile_pitch_rows: STATE_ROWS,
            alpha: 1,
            tile_skew: 0,
        };
        let affine = MatrixLayout {
            rows: STATE_ROWS,
            cols,
            tile_count: tiles,
            tile_pitch_rows: 0,
            alpha: 1,
            tile_skew: words_per_head,
        };

        let names = (0..tiles)
            .map(|tile| format!("head_{tile}"))
            .collect::<Vec<_>>();
        let bases = (0..tiles)
            .map(|tile| tile * words_per_head * BF16_VALUES_PER_BANK_WORD)
            .collect::<Vec<_>>();
        let views = names
            .iter()
            .zip(&bases)
            .map(|(name, &base)| (name.as_str(), base, fixed))
            .collect::<Vec<_>>();
        sram.validate_disjoint_layouts(&views)
            .expect("fixed per-head phases must fit without aliases");

        // D' and D occupy exactly the same bank word for every official state
        // value. D uses one compact descriptor; D' uses one ordinary base per
        // head. Therefore programmable skew has no pure bank-service credit.
        for tile in 0..tiles {
            for row in 0..STATE_ROWS {
                for word in 0..words_per_head {
                    let col = word * BF16_VALUES_PER_BANK_WORD;
                    let fixed_coord =
                        sram.physical_coord_unchecked(bases[tile as usize], fixed, 0, row, col);
                    let affine_coord = sram.physical_coord_unchecked(0, affine, tile, row, col);
                    assert_eq!(fixed_coord, affine_coord);
                }
            }
        }

        let mut expected = Vec::with_capacity(tiles as usize);
        for tile in 0..tiles {
            let value_count = (STATE_ROWS * cols) as usize;
            let values = (0..value_count)
                .map(|index| {
                    let code = (tile as usize * 131 + index) % 257;
                    (code as f32 - 128.0) / 64.0
                })
                .collect::<Vec<_>>();
            let input = QuantTensor::quantize(Tensor::from_slice(&values), ty);
            expected.push(tensor_to_f32_vec(input.as_tensor()));
            let write = sram
                .write_layout_packet(bases[tile as usize], fixed, input)
                .await;
            assert_eq!(write.bank_stall_cycles, 0);
        }

        sram.reset_packet_counters();
        let requests = bases.iter().map(|&base| (base, fixed)).collect::<Vec<_>>();
        let (actual, service) = sram.read_layout_packets(&requests).await;
        assert_eq!(service.ideal_cycles, u64::from(STATE_ROWS));
        assert_eq!(service.service_cycles, u64::from(STATE_ROWS));
        assert_eq!(service.bank_stall_cycles, 0);
        assert_eq!(service.values, u64::from(tiles * STATE_ROWS * cols));
        for (packet, expected) in actual.iter().zip(expected) {
            assert_eq!(tensor_to_f32_vec(packet.as_tensor()), expected);
        }
    }

    fn paper_addr(row: u32, bank_phase: u32) -> u32 {
        const MLEN: u32 = 2048;
        const BLEN: u32 = 32;
        row * MLEN + bank_phase * BLEN
    }

    fn expected_after_sram_quantization(sram: &MatrixSram, values: &[f32]) -> Vec<f32> {
        values
            .chunks(sram.bank_width as usize)
            .flat_map(|word| {
                let bytes = sram.values_to_word_bytes(word);
                sram.word_bytes_to_values(&bytes)
                    .into_iter()
                    .take(word.len())
            })
            .collect()
    }

    async fn assert_colocated_views_roundtrip(
        sram: &MatrixSram,
        views: &[(&str, u32, MatrixLayout)],
    ) {
        sram.validate_disjoint_layouts(views)
            .expect("compiler placement must be physically disjoint");
        let mut expected = Vec::with_capacity(views.len());
        for (view_index, &(_, base, layout)) in views.iter().enumerate() {
            let count = (layout.rows * layout.cols * layout.tile_count) as usize;
            let values = (0..count)
                .map(|index| (view_index * 100_000 + index) as f32)
                .collect::<Vec<_>>();
            let quantized = expected_after_sram_quantization(sram, &values);
            let input = QuantTensor::quantize(Tensor::from_slice(&values), sram.ty());
            sram.write_layout_packet(base, layout, input).await;
            expected.push(quantized);
        }
        for ((name, base, layout), expected) in views.iter().zip(expected) {
            let (output, _) = sram.read_layout_packet(*base, *layout).await;
            assert_eq!(
                tensor_to_f32_vec(output.as_tensor()),
                expected,
                "co-resident view {name} did not round-trip",
            );
        }
    }

    fn kda_chunk_views(affine: bool) -> Vec<(&'static str, u32, MatrixLayout)> {
        let (state, scalar, vector, scalar_row, vector_row) = if affine {
            (
                MatrixLayout {
                    rows: 16,
                    cols: 128,
                    tile_count: 16,
                    tile_pitch_rows: 8,
                    alpha: 1,
                    tile_skew: 4,
                },
                MatrixLayout {
                    rows: 16,
                    cols: 32,
                    tile_count: 16,
                    tile_pitch_rows: 1,
                    alpha: 1,
                    tile_skew: 3,
                },
                MatrixLayout {
                    rows: 1,
                    cols: 128,
                    tile_count: 16,
                    tile_pitch_rows: 1,
                    alpha: 1,
                    tile_skew: 3,
                },
                136,
                168,
            )
        } else {
            (
                MatrixLayout {
                    rows: 16,
                    cols: 128,
                    tile_count: 16,
                    tile_pitch_rows: 16,
                    alpha: 1,
                    tile_skew: 0,
                },
                MatrixLayout {
                    rows: 16,
                    cols: 32,
                    tile_count: 16,
                    tile_pitch_rows: 16,
                    alpha: 1,
                    tile_skew: 0,
                },
                MatrixLayout {
                    rows: 1,
                    cols: 128,
                    tile_count: 16,
                    tile_pitch_rows: 16,
                    alpha: 1,
                    tile_skew: 0,
                },
                0,
                0,
            )
        };
        vec![
            ("state", paper_addr(0, 0), state),
            (
                "decay",
                paper_addr(scalar_row, if affine { 0 } else { 4 }),
                scalar,
            ),
            (
                "key",
                paper_addr(scalar_row, if affine { 1 } else { 5 }),
                scalar,
            ),
            (
                "query",
                paper_addr(scalar_row, if affine { 2 } else { 6 }),
                scalar,
            ),
            (
                "value_or_error",
                paper_addr(vector_row, if affine { 0 } else { 8 }),
                vector,
            ),
            (
                "prediction_or_output",
                paper_addr(vector_row, if affine { 4 } else { 12 }),
                vector,
            ),
        ]
    }

    fn mamba_chunk_views(affine: bool) -> Vec<(&'static str, u32, MatrixLayout)> {
        let tiles = if affine { 32 } else { 16 };
        let (state, scalar, vector, scalar_row, vector_row) = if affine {
            (
                MatrixLayout {
                    rows: 16,
                    cols: 64,
                    tile_count: tiles,
                    tile_pitch_rows: 4,
                    alpha: 1,
                    tile_skew: 2,
                },
                MatrixLayout {
                    rows: 16,
                    cols: 32,
                    tile_count: tiles,
                    tile_pitch_rows: 1,
                    alpha: 1,
                    tile_skew: 1,
                },
                MatrixLayout {
                    rows: 1,
                    cols: 64,
                    tile_count: tiles,
                    tile_pitch_rows: 1,
                    alpha: 1,
                    tile_skew: 1,
                },
                140,
                188,
            )
        } else {
            (
                MatrixLayout {
                    rows: 16,
                    cols: 64,
                    tile_count: tiles,
                    tile_pitch_rows: 16,
                    alpha: 1,
                    tile_skew: 0,
                },
                MatrixLayout {
                    rows: 16,
                    cols: 32,
                    tile_count: tiles,
                    tile_pitch_rows: 16,
                    alpha: 1,
                    tile_skew: 0,
                },
                MatrixLayout {
                    rows: 1,
                    cols: 64,
                    tile_count: tiles,
                    tile_pitch_rows: 16,
                    alpha: 1,
                    tile_skew: 0,
                },
                0,
                0,
            )
        };
        vec![
            ("state", paper_addr(0, 0), state),
            (
                "decay_and_b",
                paper_addr(scalar_row, if affine { 0 } else { 2 }),
                scalar,
            ),
            (
                "c",
                paper_addr(scalar_row, if affine { 16 } else { 3 }),
                scalar,
            ),
            (
                "dt_and_skip",
                paper_addr(scalar_row, if affine { 32 } else { 4 }),
                scalar,
            ),
            (
                "x",
                paper_addr(vector_row, if affine { 0 } else { 8 }),
                vector,
            ),
            (
                "scratch",
                paper_addr(vector_row, if affine { 2 } else { 10 }),
                vector,
            ),
            (
                "output",
                paper_addr(vector_row, if affine { 4 } else { 12 }),
                vector,
            ),
        ]
    }

    #[tokio::test]
    async fn official_nemotron_state_group_fits_and_reads_32_heads_without_conflict() {
        // 32 heads x [128,64] BF16 state. The compact affine layout occupies
        // 252 physical rows; a fixed full-height layout can hold only 2 heads.
        official_recurrent_group_roundtrip(64, 32, 252).await;
    }

    #[tokio::test]
    async fn official_kimi_state_group_fits_and_reads_16_heads_without_conflict() {
        // 16 heads x [128,128] BF16 state. The compact affine layout occupies
        // 248 physical rows; a fixed full-height layout can hold only 2 heads.
        official_recurrent_group_roundtrip(128, 16, 248).await;
    }

    #[tokio::test]
    async fn fixed_phased_nemotron_state_matches_affine_without_programmable_skew() {
        fixed_phased_official_state_roundtrip(64, 32).await;
    }

    #[tokio::test]
    async fn fixed_phased_kimi_state_matches_affine_without_programmable_skew() {
        fixed_phased_official_state_roundtrip(128, 16).await;
    }

    #[tokio::test]
    async fn official_kda_chunk_and_all_live_fields_share_one_matrix_sram() {
        const MLEN: u32 = 2048;
        const DEPTH_ROWS: usize = 256;
        const BLEN: u32 = 32;
        for affine in [false, true] {
            let sram = MatrixSram::with_banks(MLEN, DEPTH_ROWS, BLEN, bf16_plain());
            let views = kda_chunk_views(affine);
            assert_colocated_views_roundtrip(&sram, &views).await;
            let state = views[0];
            let lines = (0..state.2.tile_count)
                .map(|tile| (tile, 0))
                .collect::<Vec<_>>();
            let (_, service) = sram
                .read_layout_indexed_rows(state.1, state.2, &lines)
                .await;
            assert_eq!(service.service_cycles, if affine { 1 } else { 4 });
        }
    }

    #[tokio::test]
    async fn official_mamba_chunk_and_all_live_fields_share_one_matrix_sram() {
        const MLEN: u32 = 2048;
        const DEPTH_ROWS: usize = 256;
        const BLEN: u32 = 32;
        for affine in [false, true] {
            let sram = MatrixSram::with_banks(MLEN, DEPTH_ROWS, BLEN, bf16_plain());
            let views = mamba_chunk_views(affine);
            assert_colocated_views_roundtrip(&sram, &views).await;
            let state = views[0];
            let lines = (0..state.2.tile_count)
                .map(|tile| (tile, 0))
                .collect::<Vec<_>>();
            let (_, service) = sram
                .read_layout_indexed_rows(state.1, state.2, &lines)
                .await;
            assert_eq!(service.service_cycles, if affine { 1 } else { 4 });
        }
    }

    #[test]
    fn colocated_view_validator_rejects_cross_tensor_aliases() {
        let sram = MatrixSram::with_banks(2048, 256, 32, bf16_plain());
        let view = MatrixLayout {
            rows: 1,
            cols: 128,
            tile_count: 16,
            tile_pitch_rows: 16,
            alpha: 1,
            tile_skew: 0,
        };
        let error = sram
            .validate_disjoint_layouts(&[("first", 0, view), ("second", 0, view)])
            .unwrap_err();
        assert!(error.contains("first"));
        assert!(error.contains("second"));
    }
}
