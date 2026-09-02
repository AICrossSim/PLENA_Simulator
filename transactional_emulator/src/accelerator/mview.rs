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
const RESERVED_MAP_MASK: u32 = ((1 << 12) - 1) << 16;
const STRICT_BOUNDS: u8 = 1;

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
    pub(crate) flags: u8,
}

impl MatrixViewMap {
    pub(crate) fn unpack(word: u32) -> Result<Self, String> {
        let mapping = Self {
            tile_pitch_rows: word & PITCH_MASK,
            flags: ((word >> 28) & 0xf) as u8,
        };
        if mapping.tile_pitch_rows == 0 {
            return Err("Matrix-view tile pitch must be non-zero".into());
        }
        if mapping.flags & !STRICT_BOUNDS != 0 {
            return Err(format!(
                "Matrix-view flags contain reserved bits: {:#x}",
                mapping.flags
            ));
        }
        if word & RESERVED_MAP_MASK != 0 {
            return Err(format!(
                "Matrix-view mapping uses reserved bits [27:16]: {word:#010x}"
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
        let bank_words_per_row = self.shape.cols / bank_width;
        let physical_rows_per_logical_row = bank_words_per_row.div_ceil(banks);
        let required_pitch = self.shape.rows * physical_rows_per_logical_row;
        if self.mapping.tile_pitch_rows < required_pitch {
            return Err(format!(
                "Matrix-view tile pitch {} aliases tiles; need at least {required_pitch}",
                self.mapping.tile_pitch_rows
            ));
        }
        Ok(self)
    }

    pub(crate) fn layout(self) -> MatrixLayout {
        MatrixLayout {
            rows: self.shape.rows,
            cols: self.shape.cols,
            tile_count: self.shape.tile_count,
            tile_pitch_rows: self.mapping.tile_pitch_rows,
            // PLENA's prior-work diagonal wiring is fixed at alpha=1.
            alpha: 1,
        }
    }

    pub(crate) fn values(self) -> u32 {
        self.shape
            .rows
            .checked_mul(self.shape.cols)
            .and_then(|value| value.checked_mul(self.shape.tile_count))
            .expect("validated Matrix-view dimensions overflowed u32")
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct PartialView {
    shape_word: Option<u32>,
    map_word: Option<u32>,
}

pub(super) struct MatrixViewTable {
    banks: u32,
    bank_width: u32,
    slots: [Option<MatrixViewDescriptor>; VIEW_SLOTS],
    partial: [PartialView; VIEW_SLOTS],
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
            partial: [PartialView::default(); VIEW_SLOTS],
        }
    }

    pub(super) fn configure_full(
        &mut self,
        slot: u8,
        shape_word: u32,
        map_word: u32,
    ) -> Result<(), String> {
        let index = self.slot_index(slot)?;
        let descriptor = MatrixViewDescriptor::unpack(shape_word, map_word)?
            .validate(self.banks, self.bank_width)?;
        self.slots[index] = Some(descriptor);
        self.partial[index] = PartialView {
            shape_word: Some(shape_word),
            map_word: Some(map_word),
        };
        Ok(())
    }

    pub(super) fn configure_field(
        &mut self,
        slot: u8,
        field: u8,
        value: u32,
    ) -> Result<(), String> {
        let index = self.slot_index(slot)?;
        match field {
            0 => {
                self.slots[index] = None;
                self.partial[index] = PartialView::default();
                return Ok(());
            }
            1 => self.partial[index].shape_word = Some(value),
            2 => self.partial[index].map_word = Some(value),
            _ => return Err(format!("reserved Matrix-view field {field}")),
        }

        let PartialView {
            shape_word: Some(shape_word),
            map_word: Some(map_word),
        } = self.partial[index]
        else {
            self.slots[index] = None;
            return Ok(());
        };
        self.slots[index] = Some(
            MatrixViewDescriptor::unpack(shape_word, map_word)?
                .validate(self.banks, self.bank_width)?,
        );
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
    fn full_configuration_matches_the_python_contract() {
        let mut table = MatrixViewTable::new(16, 4);
        table
            .configure_full(2, shape(64, 64, 3), mapping(64))
            .unwrap();
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
    }

    #[test]
    fn field_configuration_is_incomplete_until_both_words_arrive() {
        let mut table = MatrixViewTable::new(16, 4);
        table.configure_field(0, 1, shape(64, 64, 1)).unwrap();
        assert!(table.get(0).is_err());
        table.configure_field(0, 2, mapping(64)).unwrap();
        assert!(table.get(0).is_ok());
        table.configure_field(0, 0, 0).unwrap();
        assert!(table.get(0).is_err());
    }

    #[test]
    fn rejects_aliasing_pitch_and_reserved_mapping_bits() {
        let mut table = MatrixViewTable::new(16, 4);
        assert!(
            table
                .configure_full(0, shape(64, 64, 2), mapping(63))
                .is_err()
        );
        assert!(
            table
                .configure_full(0, shape(64, 64, 2), mapping(64) | (1 << 16))
                .is_err()
        );
        assert!(
            table
                .configure_full(0, shape(64, 64, 2), mapping(64) | (1 << 22))
                .is_err()
        );
    }
}
