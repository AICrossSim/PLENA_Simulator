//! Compiler-programmable views over PLENA's fixed-diagonal Matrix SRAM.
//!
//! A view is architectural placement metadata, not a cache or a traversal
//! engine.  Existing Matrix operations name one of four slots explicitly.
//! There is no implicit selection, replacement, auto-advance, or model state.

use sram::matrix::MatrixLayout;

const VIEW_SLOTS: usize = 4;
const DIM_MASK: u32 = (1 << 12) - 1;
const TILE_COUNT_MASK: u32 = (1 << 8) - 1;
const PITCH_MASK: u32 = (1 << 16) - 1;
const SKEW_MASK: u32 = (1 << 6) - 1;
const STRICT_BOUNDS: u8 = 1;
const AFFINE: u8 = 1 << 1;
const FP32: u8 = 1 << 2;
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
    /// Affine coefficient applied to the logical/physical row term.
    pub(crate) row_skew: u32,
    /// Additional affine phase applied to the logical tile term.
    pub(crate) tile_skew: u32,
    pub(crate) flags: u8,
}

impl MatrixViewMap {
    pub(crate) fn unpack(word: u32) -> Result<Self, String> {
        let mapping = Self {
            tile_pitch_rows: word & PITCH_MASK,
            row_skew: (word >> 16) & SKEW_MASK,
            tile_skew: (word >> 22) & SKEW_MASK,
            flags: ((word >> 28) & 0xf) as u8,
        };
        if mapping.flags & !(STRICT_BOUNDS | AFFINE | FP32 | BROADCAST_MINOR) != 0 {
            return Err(format!(
                "Matrix-view flags contain reserved bits: {:#x}",
                mapping.flags
            ));
        }
        if mapping.flags & AFFINE == 0 && (mapping.row_skew != 0 || mapping.tile_skew != 0) {
            return Err("non-zero Matrix-view skew requires the AFFINE flag".into());
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
        let affine = self.mapping.flags & AFFINE != 0;
        let alpha = if affine { self.mapping.row_skew } else { 1 };
        let tile_skew = if affine { self.mapping.tile_skew } else { 0 };
        let mut occupied = std::collections::HashMap::new();
        for tile in 0..self.shape.tile_count {
            for row in 0..self.shape.rows {
                for word in 0..words_per_row {
                    let bank_row =
                        tile * self.mapping.tile_pitch_rows + row * row_groups + word / banks;
                    let bank = (alpha * bank_row + tile_skew * tile + word) % banks;
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
        let affine = self.mapping.flags & AFFINE != 0;
        MatrixLayout {
            rows: self.shape.rows,
            cols: self.shape.cols,
            tile_count: self.shape.tile_count,
            tile_pitch_rows: self.mapping.tile_pitch_rows,
            // The control uses PLENA's prior-work diagonal wiring. Treatment
            // views make the row/tile phases explicit and compiler-selected.
            alpha: if affine { self.mapping.row_skew } else { 1 },
            tile_skew: if affine { self.mapping.tile_skew } else { 0 },
        }
    }

    pub(crate) fn fp32(self) -> bool {
        self.mapping.flags & FP32 != 0
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

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(rows: u32, cols: u32, tiles: u32) -> u32 {
        (rows - 1) | ((cols - 1) << 12) | ((tiles - 1) << 24)
    }

    fn mapping(pitch: u32) -> u32 {
        pitch | (u32::from(STRICT_BOUNDS) << 28)
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
        assert_eq!((view.mapping.row_skew, view.mapping.tile_skew), (0, 0));
    }

    #[test]
    fn rejects_aliasing_pitch_and_noncanonical_affine_bits() {
        let mut table = MatrixViewTable::new(16, 4);
        assert!(table.configure(0, shape(64, 64, 2), mapping(63)).is_err());
        assert!(
            table
                .configure(0, shape(64, 64, 2), mapping(64) | (1 << 16))
                .is_err()
        );
        let affine = mapping(64) | (1 << 16) | (5 << 22) | (u32::from(AFFINE) << 28);
        table.configure(0, shape(64, 64, 2), affine).unwrap();
        let view = table.get(0).unwrap();
        assert_eq!((view.layout().alpha, view.layout().tile_skew), (1, 5));
    }

    #[test]
    fn zero_pitch_is_legal_only_when_tile_phase_prevents_aliasing() {
        let mut table = MatrixViewTable::new(64, 32);
        assert!(table.configure(0, shape(128, 128, 8), mapping(0)).is_err());
        let compact = mapping(0) | (1 << 16) | (4 << 22) | (u32::from(AFFINE) << 28);
        table.configure(0, shape(128, 128, 8), compact).unwrap();
        let view = table.get(0).unwrap();
        assert_eq!(view.mapping.tile_pitch_rows, 0);
        assert_eq!((view.layout().alpha, view.layout().tile_skew), (1, 4));
    }
}
