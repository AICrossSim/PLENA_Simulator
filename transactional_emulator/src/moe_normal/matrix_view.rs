//! Standalone matrix-view and transpose-SRAM primitives.
//!
//! These are not connected to the normal MoE runner or an Attention pipeline.
//! A view maps logical coordinates back to the original matrix *before* looking
//! up its block-8 scale. Transposition does not regroup or requantize MX blocks.
//! The SRAM stores decoded BF16, with finite, shared read/write bank ports.

use half::bf16;
use quantize::{DataType, FpType};

use super::types::MatrixRegion;

pub const LOCAL_MX_BLOCK: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    Normal,
    Transposed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceElement {
    pub row: usize,
    pub col: usize,
    pub element_address: u64,
    pub scale_address: u64,
}

/// A rectangular crop in source coordinates, optionally viewed transposed.
/// Extents describe the crop *before* transposition; `shape()` is the result.
#[derive(Clone, Debug)]
pub struct MatrixView {
    source: MatrixRegion,
    origin: [usize; 2],
    extent: [usize; 2],
    orientation: Orientation,
}

impl MatrixView {
    pub fn new(
        source: &MatrixRegion,
        origin: [usize; 2],
        extent: [usize; 2],
        orientation: Orientation,
        image_bytes: u64,
    ) -> Result<Self, String> {
        if source.rows == 0 || source.cols == 0 || extent.contains(&0) {
            return Err("matrix and view extents must be nonzero".into());
        }
        for axis in 0..2 {
            let end = origin[axis]
                .checked_add(extent[axis])
                .ok_or("view extent overflow")?;
            if end > [source.rows, source.cols][axis] {
                return Err("view exceeds source matrix".into());
            }
        }
        let blocks = source.cols.div_ceil(LOCAL_MX_BLOCK) as u64;
        let padded_cols = blocks
            .checked_mul(LOCAL_MX_BLOCK as u64)
            .ok_or("MX padded width overflow")?;
        let mut ranges = Vec::with_capacity(2);
        for (base, stride, width) in [
            (source.element_base, source.element_row_stride, padded_cols),
            (source.scale_base, source.scale_row_stride, blocks),
        ] {
            if stride < width {
                return Err("source row stride must cover padded MX rows".into());
            }
            let end = (source.rows as u64 - 1)
                .checked_mul(stride)
                .and_then(|v| v.checked_add(base))
                .and_then(|v| v.checked_add(width))
                .ok_or("source address overflow")?;
            if end > image_bytes {
                return Err("source matrix exceeds HBM image".into());
            }
            ranges.push(base..end);
        }
        // This contract uses distinct element and scale planes; interleaving
        // either plane into the other's row padding is deliberately excluded.
        if ranges[0].start < ranges[1].end && ranges[1].start < ranges[0].end {
            return Err("source element and scale planes overlap".into());
        }
        Ok(Self {
            source: source.clone(),
            origin,
            extent,
            orientation,
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        match self.orientation {
            Orientation::Normal => self.extent,
            Orientation::Transposed => [self.extent[1], self.extent[0]],
        }
    }

    /// Returns source addresses without reading memory, for a future DMA caller.
    /// An unaligned crop may touch two source scales within eight view elements.
    pub fn source_element(&self, row: usize, col: usize) -> Result<SourceElement, String> {
        let [rows, cols] = self.shape();
        if row >= rows || col >= cols {
            return Err("view coordinate out of bounds".into());
        }
        let [r, c] = match self.orientation {
            Orientation::Normal => [self.origin[0] + row, self.origin[1] + col],
            Orientation::Transposed => [self.origin[0] + col, self.origin[1] + row],
        };
        Ok(SourceElement {
            row: r,
            col: c,
            element_address: self.source.element_base
                + r as u64 * self.source.element_row_stride
                + c as u64,
            scale_address: self.source.scale_base
                + r as u64 * self.source.scale_row_stride
                + (c / LOCAL_MX_BLOCK) as u64,
        })
    }

    /// Functional helper only: a caller must model DMA and decode latency.
    pub fn decode_bf16(&self, image: &[u8], row: usize, col: usize) -> Result<u16, String> {
        let location = self.source_element(row, col)?;
        let byte = |address| {
            usize::try_from(address)
                .ok()
                .and_then(|index| image.get(index))
                .copied()
                .ok_or_else(|| "source byte missing from HBM image".to_string())
        };
        let element_type = DataType::Fp(FpType {
            sign: true,
            exponent: 4,
            mantissa: 3,
        });
        let element = element_type.convert_bits_to_f32(byte(location.element_address)? as u32);
        let scale =
            DataType::Fp(FpType::E8M0).convert_bits_to_f32(byte(location.scale_address)? as u32);
        let value = bf16::from_f32(element * scale);
        if !value.is_finite() {
            return Err("decoded matrix value is not finite BF16".into());
        }
        Ok(value.to_bits())
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct BankPort {
    cycle: u64,
    used: usize,
}

/// A single resident tile. Banks have `words_per_cycle` shared read/write ports.
/// Fill and drain use diagonal banking; there is no free transpose or broadcast.
/// Times are integral simulator cycles, with a one-cycle access latency.
pub struct TransposeBuffer {
    rows: usize,
    cols: usize,
    banks: usize,
    words_per_bank_row: usize,
    words_per_cycle: usize,
    data: Vec<Vec<u16>>,
    ports: Vec<BankPort>,
    ready_cycle: Option<u64>,
    last_completion: u64,
    storage_bytes: usize,
}

impl TransposeBuffer {
    /// Storage includes padded BF16 bank words and conservative control bytes:
    /// 16 per bank (cycle/port counters) plus 16 for tile timing/state.
    pub fn required_bytes(rows: usize, cols: usize, banks: usize) -> Result<usize, String> {
        if rows == 0 || cols == 0 || banks == 0 {
            return Err("transpose buffer dimensions and bank count must be nonzero".into());
        }
        rows.checked_mul(cols.div_ceil(banks))
            .and_then(|v| v.checked_mul(banks))
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| banks.checked_add(1)?.checked_mul(16)?.checked_add(v))
            .ok_or_else(|| "transpose storage overflow".into())
    }

    pub fn new(
        rows: usize,
        cols: usize,
        banks: usize,
        words_per_cycle: usize,
        capacity_bytes: usize,
    ) -> Result<Self, String> {
        let storage_bytes = Self::required_bytes(rows, cols, banks)?;
        if words_per_cycle == 0 || storage_bytes > capacity_bytes {
            return Err("transpose buffer exceeds capacity or has no ports".into());
        }
        let words_per_bank_row = cols.div_ceil(banks);
        Ok(Self {
            rows,
            cols,
            banks,
            words_per_bank_row,
            words_per_cycle,
            data: vec![vec![0; rows * words_per_bank_row]; banks],
            ports: vec![BankPort::default(); banks],
            ready_cycle: None,
            last_completion: 0,
            storage_bytes,
        })
    }

    pub fn storage_bytes(&self) -> usize {
        self.storage_bytes
    }

    /// Consecutive row and column entries rotate through the banks. Within a
    /// row, every group of `banks` columns gets a distinct physical word.
    pub fn bank_location(&self, row: usize, col: usize) -> Result<(usize, usize), String> {
        if row >= self.rows || col >= self.cols {
            return Err("transpose SRAM coordinate out of bounds".into());
        }
        let bank = ((row % self.banks) + (col % self.banks)) % self.banks;
        Ok((bank, row * self.words_per_bank_row + col / self.banks))
    }

    fn access(&mut self, bank: usize, now: u64) -> Result<u64, String> {
        let port = &mut self.ports[bank];
        if now > port.cycle {
            port.cycle = now;
            port.used = 0;
        }
        if port.used == self.words_per_cycle {
            port.cycle = port.cycle.checked_add(1).ok_or("bank cycle overflow")?;
            port.used = 0;
        }
        let completion = port.cycle.checked_add(1).ok_or("bank cycle overflow")?;
        port.used += 1;
        self.last_completion = self.last_completion.max(completion);
        Ok(completion)
    }

    /// Atomically accepts one complete BF16 tile; reports the last write cycle.
    /// The data is inaccessible until that cycle, even though host writes happen
    /// synchronously. No second tile can overwrite it until `release` succeeds.
    pub fn fill(&mut self, values: &[u16], now: u64) -> Result<u64, String> {
        if self.ready_cycle.is_some() {
            return Err("transpose buffer already owns a tile".into());
        }
        if now < self.last_completion {
            return Err("new transpose fill precedes previous tile release".into());
        }
        if values.len() != self.rows * self.cols {
            return Err("BF16 tile length does not match transpose SRAM shape".into());
        }
        // A conservative bound rejects overflow before any storage is changed.
        now.max(self.last_completion)
            .checked_add(values.len() as u64)
            .ok_or("transpose fill cycle overflow")?;
        let mut completion = now;
        for row in 0..self.rows {
            for col in 0..self.cols {
                let (bank, word) = self.bank_location(row, col)?;
                completion = completion.max(self.access(bank, now)?);
                self.data[bank][word] = values[row * self.cols + col];
            }
        }
        self.ready_cycle = Some(completion);
        Ok(completion)
    }

    /// Reads one contiguous logical row, normal or transposed. Returned BF16
    /// words become available only at the reported completion cycle. Repeated
    /// reads share the same bank-port schedule, including calls at the same time.
    pub fn read_row(
        &mut self,
        orientation: Orientation,
        row: usize,
        start: usize,
        count: usize,
        now: u64,
    ) -> Result<(Vec<u16>, u64), String> {
        let ready = self.ready_cycle.ok_or("transpose buffer is empty")?;
        if now < ready {
            return Err("transpose tile is not ready".into());
        }
        let (rows, cols) = match orientation {
            Orientation::Normal => (self.rows, self.cols),
            Orientation::Transposed => (self.cols, self.rows),
        };
        let end = start.checked_add(count).ok_or("read extent overflow")?;
        if row >= rows || count == 0 || end > cols {
            return Err("transpose row read exceeds logical matrix".into());
        }
        now.max(self.last_completion)
            .checked_add(count as u64)
            .ok_or("transpose read cycle overflow")?;
        let mut result = Vec::with_capacity(count);
        let mut completion = now;
        for col in start..end {
            let (r, c) = match orientation {
                Orientation::Normal => (row, col),
                Orientation::Transposed => (col, row),
            };
            let (bank, word) = self.bank_location(r, c)?;
            completion = completion.max(self.access(bank, now)?);
            result.push(self.data[bank][word]);
        }
        Ok((result, completion))
    }

    pub fn release(&mut self, now: u64) -> Result<(), String> {
        if self.ready_cycle.is_none() || now < self.last_completion {
            return Err("cannot release empty or still-active transpose tile".into());
        }
        self.ready_cycle = None;
        self.last_completion = now;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn source() -> (MatrixRegion, Vec<u8>) {
        let region = MatrixRegion {
            rows: 3,
            cols: 13,
            element_base: 8,
            scale_base: 96,
            element_row_stride: 24,
            scale_row_stride: 4,
        };
        let mut bytes = vec![0xa5; 112];
        for row in 0..3 {
            for col in 0..13 {
                bytes[8 + row * 24 + col] = if col % 2 == 0 { 0x38 } else { 0xb8 };
            }
            for block in 0..2 {
                bytes[96 + row * 4 + block] = 126 + (row * 2 + block) as u8;
            }
        }
        (region, bytes)
    }

    #[test]
    fn transposed_crop_preserves_original_scale_ownership() {
        let (region, bytes) = source();
        let view = MatrixView::new(&region, [1, 6], [2, 7], Orientation::Transposed, 112).unwrap();
        assert_eq!(view.shape(), [7, 2]);
        for row in 0..7 {
            for col in 0..2 {
                let source = view.source_element(row, col).unwrap();
                assert_eq!((source.row, source.col), (1 + col, 6 + row));
                assert_eq!(
                    source.scale_address,
                    96 + (1 + col) as u64 * 4 + ((6 + row) / 8) as u64
                );
                let sign = if (6 + row) % 2 == 0 { 1.0 } else { -1.0 };
                let exponent = (2 * (1 + col) + (6 + row) / 8) as i32 - 1;
                let expected = bf16::from_f32(sign * 2f32.powi(exponent)).to_bits();
                assert_eq!(view.decode_bf16(&bytes, row, col).unwrap(), expected);
            }
        }
        assert!(view.source_element(7, 0).is_err());
        assert!(view.source_element(0, 2).is_err());
    }

    #[test]
    fn normal_unaligned_tail_uses_two_scales_and_ignores_padding() {
        let (region, bytes) = source();
        let view = MatrixView::new(&region, [0, 7], [1, 6], Orientation::Normal, 112).unwrap();
        assert_eq!(view.shape(), [1, 6]);
        assert_eq!(view.source_element(0, 0).unwrap().scale_address, 96);
        assert_eq!(view.source_element(0, 1).unwrap().scale_address, 97);
        assert_eq!(
            bf16::from_bits(view.decode_bf16(&bytes, 0, 0).unwrap()).to_f32(),
            -0.5
        );
        assert_eq!(
            bf16::from_bits(view.decode_bf16(&bytes, 0, 5).unwrap()).to_f32(),
            1.0
        );
        assert!(view.decode_bf16(&bytes[..16], 0, 5).is_err());
    }

    #[test]
    fn rejects_invalid_extents_strides_overlaps_and_address_overflow() {
        let (region, _) = source();
        for (origin, extent) in [
            ([0, 0], [0, 1]),
            ([2, 0], [2, 1]),
            ([0, usize::MAX], [1, 1]),
        ] {
            assert!(MatrixView::new(&region, origin, extent, Orientation::Normal, 112).is_err());
        }
        let mut bad = region.clone();
        bad.element_row_stride = 13;
        assert!(MatrixView::new(&bad, [0, 0], [1, 1], Orientation::Normal, 112).is_err());
        bad = region.clone();
        bad.scale_base = 9;
        assert!(MatrixView::new(&bad, [0, 0], [1, 1], Orientation::Normal, 112).is_err());
        bad = region;
        bad.element_base = u64::MAX - 8;
        assert!(MatrixView::new(&bad, [0, 0], [1, 1], Orientation::Normal, u64::MAX).is_err());
    }

    #[test]
    fn non_square_bank_mapping_has_no_aliases_and_charges_padding() {
        for rows in 1..8 {
            for cols in 1..11 {
                for banks in 1..6 {
                    let required = TransposeBuffer::required_bytes(rows, cols, banks).unwrap();
                    assert!(TransposeBuffer::new(rows, cols, banks, 1, required - 1).is_err());
                    let buffer = TransposeBuffer::new(rows, cols, banks, 1, required).unwrap();
                    let mut addresses = BTreeSet::new();
                    for row in 0..rows {
                        for col in 0..cols {
                            assert!(addresses.insert(buffer.bank_location(row, col).unwrap()));
                        }
                    }
                    assert_eq!(addresses.len(), rows * cols);
                    assert_eq!(
                        buffer.storage_bytes(),
                        rows * cols.div_ceil(banks) * banks * 2 + (banks + 1) * 16
                    );
                }
            }
        }
        assert!(TransposeBuffer::required_bytes(usize::MAX, 4, 2).is_err());
    }

    #[test]
    fn transpose_matches_source_view_after_finite_write_latency() {
        let (region, bytes) = source();
        let normal = MatrixView::new(&region, [1, 6], [2, 7], Orientation::Normal, 112).unwrap();
        let transposed =
            MatrixView::new(&region, [1, 6], [2, 7], Orientation::Transposed, 112).unwrap();
        let mut values = Vec::new();
        for row in 0..2 {
            for col in 0..7 {
                values.push(normal.decode_bf16(&bytes, row, col).unwrap());
            }
        }
        let capacity = TransposeBuffer::required_bytes(2, 7, 2).unwrap();
        let mut buffer = TransposeBuffer::new(2, 7, 2, 1, capacity).unwrap();
        assert!(buffer.read_row(Orientation::Normal, 0, 0, 1, 0).is_err());
        let ready = buffer.fill(&values, 10).unwrap();
        assert_eq!(ready, 17); // 14 writes / 2 balanced single-port banks.
        assert!(
            buffer
                .read_row(Orientation::Transposed, 0, 0, 2, 16)
                .is_err()
        );
        assert!(buffer.fill(&values, 17).is_err());
        let mut completion = ready;
        for row in 0..7 {
            let (actual, end) = buffer
                .read_row(Orientation::Transposed, row, 0, 2, completion)
                .unwrap();
            let expected: Vec<_> = (0..2)
                .map(|col| transposed.decode_bf16(&bytes, row, col).unwrap())
                .collect();
            assert_eq!(actual, expected);
            assert_eq!(end, completion + 1);
            completion = end;
        }
        assert!(buffer.release(completion - 1).is_err());
        buffer.release(completion).unwrap();
        assert!(buffer.fill(&values, completion - 1).is_err());
        assert!(
            buffer
                .read_row(Orientation::Normal, 0, 0, 1, completion)
                .is_err()
        );
        assert!(buffer.fill(&values, completion).unwrap() > completion);
    }

    #[test]
    fn repeated_reads_share_ports_and_wider_ports_reduce_cycles() {
        let capacity = TransposeBuffer::required_bytes(1, 8, 2).unwrap();
        let values: Vec<u16> = (0..8).collect();
        let mut narrow = TransposeBuffer::new(1, 8, 2, 1, capacity).unwrap();
        let mut wide = TransposeBuffer::new(1, 8, 2, 2, capacity).unwrap();
        assert_eq!(narrow.fill(&values, 0).unwrap(), 4);
        assert_eq!(wide.fill(&values, 0).unwrap(), 2);
        let (_, first) = narrow.read_row(Orientation::Normal, 0, 0, 8, 4).unwrap();
        let (_, second) = narrow.read_row(Orientation::Normal, 0, 0, 8, 4).unwrap();
        assert_eq!((first, second), (8, 12));
        assert_eq!(wide.read_row(Orientation::Normal, 0, 0, 8, 2).unwrap().1, 4);
        assert!(narrow.read_row(Orientation::Normal, 0, 7, 2, 12).is_err());
        assert!(TransposeBuffer::new(1, 8, 2, 0, capacity).is_err());
    }
}
