//! Model-independent affine operand streams for `L_STREAM_CFG`.
//!
//! This is explicit compiler-managed configuration, not a cache: there are no
//! tags, lookups, replacement decisions, dirty bits, or implicit transfers.

use std::collections::BTreeSet;

#[cfg(test)]
use std::collections::{BTreeMap, btree_map::Entry};

pub(super) const MAX_STREAM_SLOTS: usize = 4;

const ENABLE: u32 = 1 << 0;
const AUTO_ADVANCE: u32 = 1 << 1;
const AFFINE: u32 = 1 << 2;
const TARGET_FP: u32 = 1 << 3;
const WRITE: u32 = 1 << 4;
const LANE_RESTORE: u32 = 1 << 5;
const STRICT_BOUNDS: u32 = 1 << 6;
const PACKETIZED: u32 = 1 << 7;
const KNOWN_FLAGS: u32 =
    ENABLE | AUTO_ADVANCE | AFFINE | TARGET_FP | WRITE | LANE_RESTORE | STRICT_BOUNDS | PACKETIZED;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(super) enum ConfigField {
    Reset = 0,
    Flags = 1,
    Base = 2,
    ExtentMinor = 3,
    ExtentMajor = 4,
    ExtentField = 5,
    ExtentGroup = 6,
    BankRowPitch = 7,
    Alpha = 8,
    Beta = 9,
    Gamma = 10,
    Advance = 11,
    PacketElements = 12,
    StorageAtom = 13,
    PhysicalBaseRow = 14,
    PacketStride = 15,
}

impl TryFrom<u8> for ConfigField {
    type Error = String;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Reset),
            1 => Ok(Self::Flags),
            2 => Ok(Self::Base),
            3 => Ok(Self::ExtentMinor),
            4 => Ok(Self::ExtentMajor),
            5 => Ok(Self::ExtentField),
            6 => Ok(Self::ExtentGroup),
            7 => Ok(Self::BankRowPitch),
            8 => Ok(Self::Alpha),
            9 => Ok(Self::Beta),
            10 => Ok(Self::Gamma),
            11 => Ok(Self::Advance),
            12 => Ok(Self::PacketElements),
            13 => Ok(Self::StorageAtom),
            14 => Ok(Self::PhysicalBaseRow),
            15 => Ok(Self::PacketStride),
            _ => Err(format!("reserved L_STREAM_CFG field {value}")),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(super) enum StreamTarget {
    Gp(u8),
    Fp(u8),
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct PhysicalCoord {
    pub(crate) bank: u32,
    pub(crate) bank_row: u32,
    pub(crate) sublane: u32,
}

/// Immutable affine view consumed by Matrix writeback and Vector lane restore.
///
/// It contains only compiler-written address metadata.  There are no tags,
/// lookup outcomes, replacement decisions, or implicit data movement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AffineView {
    base: u32,
    extent_minor: u32,
    extent_major: u32,
    extent_field: u32,
    extent_group: u32,
    bank_row_pitch: u32,
    alpha: u32,
    beta: u32,
    gamma: u32,
    storage_atom: u32,
    physical_base_row: u32,
    write: bool,
    lane_restore: bool,
    packet_elements: u32,
    packet_stride: u32,
    packetized: bool,
}

impl AffineView {
    #[cfg(test)]
    pub(crate) fn packet_test_view(spec: PacketTestView) -> Self {
        Self {
            base: spec.base,
            extent_minor: spec.extent_minor,
            extent_major: spec.extent_major,
            extent_field: 1,
            extent_group: 1,
            bank_row_pitch: 0,
            alpha: spec.alpha,
            beta: 0,
            gamma: 0,
            storage_atom: spec.storage_atom,
            physical_base_row: spec.physical_base_row,
            write: spec.write,
            lane_restore: true,
            packet_elements: spec.extent_minor,
            packet_stride: spec.packet_stride,
            packetized: spec.packetized,
        }
    }

    fn logical_coord(self, address: u32) -> Result<(u32, u32, u32, u32), String> {
        let mut relative = address
            .checked_sub(self.base)
            .ok_or_else(|| format!("affine address {address} precedes base {}", self.base))?;
        let elements = self
            .extent_minor
            .checked_mul(self.extent_major)
            .and_then(|value| value.checked_mul(self.extent_field))
            .and_then(|value| value.checked_mul(self.extent_group))
            .ok_or_else(|| "affine extent product overflow".to_string())?;
        if relative >= elements {
            return Err(format!(
                "affine address {address} is outside [{}, {})",
                self.base,
                self.base + elements
            ));
        }
        let minor = relative % self.extent_minor;
        relative /= self.extent_minor;
        let major = relative % self.extent_major;
        relative /= self.extent_major;
        let field = relative % self.extent_field;
        let group = relative / self.extent_field;
        Ok((group, field, major, minor))
    }

    pub(crate) fn place(self, address: u32, banks: u32) -> Result<PhysicalCoord, String> {
        let (group, field, major, minor) = self.logical_coord(address)?;
        let stripe = minor / self.storage_atom;
        let phase = (self.alpha * major + self.beta * field + self.gamma * group) % banks;
        let outer = (group * self.extent_field + field) * self.extent_major + major;
        let minimum_pitch = self
            .extent_minor
            .div_ceil(self.storage_atom)
            .div_ceil(banks);
        let pitch = if self.bank_row_pitch == 0 {
            minimum_pitch
        } else {
            self.bank_row_pitch
        };
        Ok(PhysicalCoord {
            bank: (stripe + phase) % banks,
            bank_row: self.physical_base_row + outer * pitch + stripe / banks,
            sublane: minor % self.storage_atom,
        })
    }

    pub(crate) fn storage_atom(self) -> u32 {
        self.storage_atom
    }

    pub(crate) fn is_write(self) -> bool {
        self.write
    }

    pub(crate) fn restores_lanes(self) -> bool {
        self.lane_restore
    }

    pub(crate) fn is_packetized(self) -> bool {
        self.packetized
    }

    pub(crate) fn packet_elements(self) -> u32 {
        self.packet_elements
    }

    pub(crate) fn packet_stride(self) -> u32 {
        self.packet_stride
    }
}

#[cfg(test)]
pub(crate) struct PacketTestView {
    pub(crate) base: u32,
    pub(crate) extent_minor: u32,
    pub(crate) extent_major: u32,
    pub(crate) alpha: u32,
    pub(crate) storage_atom: u32,
    pub(crate) physical_base_row: u32,
    pub(crate) packet_stride: u32,
    pub(crate) packetized: bool,
    pub(crate) write: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PacketService {
    pub(crate) values: usize,
    pub(crate) bank_words: usize,
    pub(crate) bandwidth_floor_cycles: u32,
    pub(crate) service_cycles: u32,
}

impl PacketService {
    pub(crate) fn conflict_stall_cycles(self) -> u32 {
        self.service_cycles - self.bandwidth_floor_cycles
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ScalarPacketView {
    pub(crate) origin: u32,
    pub(crate) packet_elements: u32,
    pub(crate) storage_atom: u32,
    pub(crate) packet_stride: u32,
}

/// Sparse executable view of the candidate banked output SRAM.
///
/// It stores values at physical coordinates, so aliasing, duplicate writes and
/// read-before-write errors cannot be hidden by a logical tensor dictionary.
/// This is a Simulator proof object, not an implicit cache or a new memory
/// hierarchy visible to programs.
#[cfg(test)]
#[derive(Debug)]
pub(super) struct BankedLayoutBuffer {
    words: BTreeMap<PhysicalCoord, u64>,
}

#[cfg(test)]
impl BankedLayoutBuffer {
    pub(super) fn new() -> Self {
        Self {
            words: BTreeMap::new(),
        }
    }

    pub(super) fn write(&mut self, coord: PhysicalCoord, value: u64) -> Result<(), String> {
        match self.words.entry(coord) {
            Entry::Vacant(entry) => {
                entry.insert(value);
                Ok(())
            }
            Entry::Occupied(_) => Err(format!(
                "banked output SRAM received a duplicate write at {coord:?}"
            )),
        }
    }

    pub(super) fn read(&self, coord: PhysicalCoord) -> Result<u64, String> {
        self.words
            .get(&coord)
            .copied()
            .ok_or_else(|| format!("banked output SRAM read before write at {coord:?}"))
    }
}

fn div_ceil(value: usize, divisor: usize) -> usize {
    value.div_ceil(divisor)
}

pub(crate) fn packet_service(
    coords: &[PhysicalCoord],
    banks: u32,
    ports_per_bank: u32,
) -> Result<PacketService, String> {
    if coords.is_empty() || banks == 0 || ports_per_bank == 0 {
        return Err("packet service requires values, banks, and ports".to_string());
    }
    let words: BTreeSet<_> = coords
        .iter()
        .map(|coord| (coord.bank, coord.bank_row))
        .collect();
    let mut bank_words = vec![0_usize; banks as usize];
    for (bank, _) in &words {
        let Some(count) = bank_words.get_mut(*bank as usize) else {
            return Err(format!(
                "physical packet references bank {bank} outside {banks}"
            ));
        };
        *count += 1;
    }
    let floor = div_ceil(words.len(), (banks * ports_per_bank) as usize) as u32;
    let service = bank_words
        .into_iter()
        .map(|count| div_ceil(count, ports_per_bank as usize) as u32)
        .max()
        .unwrap_or(0);
    Ok(PacketService {
        values: coords.len(),
        bank_words: words.len(),
        bandwidth_floor_cycles: floor,
        service_cycles: service,
    })
}

#[derive(Clone, Copy, Debug)]
struct StreamSlot {
    target: Option<u8>,
    flags: u32,
    base: u32,
    current: u32,
    packet_index: u32,
    extent_minor: u32,
    extent_major: u32,
    extent_field: u32,
    extent_group: u32,
    bank_row_pitch: u32,
    alpha: u32,
    beta: u32,
    gamma: u32,
    advance: u32,
    packet_elements: u32,
    storage_atom: u32,
    physical_base_row: u32,
    packet_stride: u32,
}

impl Default for StreamSlot {
    fn default() -> Self {
        Self {
            target: None,
            flags: 0,
            base: 0,
            current: 0,
            packet_index: 0,
            extent_minor: 1,
            extent_major: 1,
            extent_field: 1,
            extent_group: 1,
            bank_row_pitch: 0,
            alpha: 0,
            beta: 0,
            gamma: 0,
            advance: 0,
            packet_elements: 1,
            storage_atom: 1,
            physical_base_row: 0,
            packet_stride: 1,
        }
    }
}

impl StreamSlot {
    fn enabled(self) -> bool {
        self.flags & ENABLE != 0
    }

    fn auto_advance(self) -> bool {
        self.flags & AUTO_ADVANCE != 0
    }

    fn packetized(self) -> bool {
        self.flags & PACKETIZED != 0
    }

    fn packet_segments(self) -> u32 {
        self.packet_elements / self.storage_atom
    }

    fn packet_minor_steps(self) -> u32 {
        self.extent_minor / self.storage_atom
    }

    fn packet_steps(self) -> u32 {
        self.packet_minor_steps()
            * (self.extent_major / self.packet_segments())
            * self.extent_field
            * self.extent_group
    }

    fn packet_origin(self) -> u32 {
        let minor_steps = self.packet_minor_steps();
        let minor_index = self.packet_index % minor_steps;
        let major_block = self.packet_index / minor_steps;
        self.base
            .checked_add(
                minor_index
                    .checked_mul(self.advance)
                    .expect("L-stream packet minor address overflow"),
            )
            .and_then(|value| {
                value.checked_add(
                    major_block
                        .checked_mul(self.packet_segments())
                        .and_then(|count| count.checked_mul(self.packet_stride))
                        .expect("L-stream packet major address overflow"),
                )
            })
            .expect("L-stream packet origin overflow")
    }

    fn target(self) -> Option<StreamTarget> {
        if !self.enabled() {
            return None;
        }
        self.target.map(|target| {
            if self.flags & TARGET_FP != 0 {
                StreamTarget::Fp(target)
            } else {
                StreamTarget::Gp(target)
            }
        })
    }

    fn minimum_pitch(self, banks: u32) -> u32 {
        let stripes = self.extent_minor.div_ceil(self.storage_atom);
        stripes.div_ceil(banks)
    }

    fn pitch(self, banks: u32) -> u32 {
        if self.bank_row_pitch == 0 {
            self.minimum_pitch(banks)
        } else {
            self.bank_row_pitch
        }
    }

    fn validate(self, target: u8, banks: u32) -> Result<(), String> {
        if self.flags & !KNOWN_FLAGS != 0 {
            return Err(format!(
                "unknown L-stream flags {:#x}",
                self.flags & !KNOWN_FLAGS
            ));
        }
        let target_limit = if self.flags & TARGET_FP != 0 { 8 } else { 16 };
        if target as usize >= target_limit {
            return Err(format!(
                "L-stream target {target} exceeds register file size {target_limit}"
            ));
        }
        for (name, value) in [
            ("extent_minor", self.extent_minor),
            ("extent_major", self.extent_major),
            ("extent_field", self.extent_field),
            ("extent_group", self.extent_group),
            ("packet_elements", self.packet_elements),
            ("storage_atom", self.storage_atom),
        ] {
            if value == 0 {
                return Err(format!("enabled L-stream requires nonzero {name}"));
            }
        }
        if !self.packet_elements.is_multiple_of(self.storage_atom) {
            return Err("L-stream packet_elements must be divisible by storage_atom".to_string());
        }
        if self.packetized() {
            let segments = self.packet_segments();
            if !self.extent_minor.is_multiple_of(self.storage_atom) {
                return Err("packetized L-stream minor extent must contain whole atoms".to_string());
            }
            if !self.extent_major.is_multiple_of(segments) {
                return Err(format!(
                    "packetized L-stream major extent {} must be divisible by {segments} segments",
                    self.extent_major
                ));
            }
            if self.flags & TARGET_FP != 0 && self.flags & WRITE != 0 {
                return Err("packetized FP streams are read-only scalar operands".to_string());
            }
        }
        if banks == 0 || self.pitch(banks) < self.minimum_pitch(banks) {
            return Err("L-stream bank-row pitch aliases logical rows".to_string());
        }
        let elements = self
            .extent_minor
            .checked_mul(self.extent_major)
            .and_then(|v| v.checked_mul(self.extent_field))
            .and_then(|v| v.checked_mul(self.extent_group))
            .ok_or_else(|| "L-stream extent product overflow".to_string())?;
        self.base
            .checked_add(elements)
            .ok_or_else(|| "L-stream bound overflow".to_string())?;
        Ok(())
    }

    #[cfg(test)]
    fn place(
        self,
        group: u32,
        field: u32,
        major: u32,
        minor: u32,
        banks: u32,
    ) -> Result<PhysicalCoord, String> {
        if group >= self.extent_group
            || field >= self.extent_field
            || major >= self.extent_major
            || minor >= self.extent_minor
        {
            return Err("L-stream logical coordinate out of range".to_string());
        }
        let stripe = minor / self.storage_atom;
        let sublane = minor % self.storage_atom;
        let phase = (self.alpha * major + self.beta * field + self.gamma * group) % banks;
        let bank = (stripe + phase) % banks;
        let outer = (group * self.extent_field + field) * self.extent_major + major;
        let bank_row = self.physical_base_row + outer * self.pitch(banks) + stripe / banks;
        Ok(PhysicalCoord {
            bank,
            bank_row,
            sublane,
        })
    }

    fn advance(&mut self) {
        if !self.enabled() || self.flags & AUTO_ADVANCE == 0 {
            return;
        }
        if self.packetized() {
            self.packet_index = self
                .packet_index
                .checked_add(1)
                .expect("L-stream packet index overflow");
            if self.flags & STRICT_BOUNDS != 0 {
                assert!(
                    self.packet_index <= self.packet_steps(),
                    "L-stream packet index {} exceeds {} configured packets",
                    self.packet_index,
                    self.packet_steps()
                );
            }
            return;
        }
        let next = self
            .current
            .checked_add(self.advance)
            .expect("L-stream address advance overflow");
        if self.flags & STRICT_BOUNDS != 0 {
            let elements =
                self.extent_minor * self.extent_major * self.extent_field * self.extent_group;
            let end = self.base + elements;
            assert!(
                next <= end,
                "L-stream advance moved address {next} beyond [{}, {end}]",
                self.base
            );
        }
        self.current = next;
    }

    fn checked_current(self) -> u32 {
        if self.packetized() {
            assert!(
                self.packet_index < self.packet_steps(),
                "L-stream packet index {} is outside {} configured packets",
                self.packet_index,
                self.packet_steps()
            );
            let origin = self.packet_origin();
            if self.flags & STRICT_BOUNDS != 0 {
                let last = origin
                    .checked_add(
                        (self.packet_segments() - 1)
                            .checked_mul(self.packet_stride)
                            .expect("L-stream packet stride overflow"),
                    )
                    .and_then(|value| value.checked_add(self.storage_atom))
                    .expect("L-stream packet bound overflow");
                let end = self
                    .base
                    .checked_add(
                        self.extent_minor
                            .checked_mul(self.extent_major)
                            .and_then(|value| value.checked_mul(self.extent_field))
                            .and_then(|value| value.checked_mul(self.extent_group))
                            .expect("L-stream extent product overflow"),
                    )
                    .expect("L-stream bound overflow");
                assert!(
                    origin >= self.base && last <= end,
                    "L-stream packet [{origin}, {last}) is outside [{}, {end})",
                    self.base
                );
            }
            return origin;
        }
        if self.flags & STRICT_BOUNDS != 0 {
            let elements =
                self.extent_minor * self.extent_major * self.extent_field * self.extent_group;
            let end = self.base + elements;
            let packet_end = self
                .current
                .checked_add(self.packet_elements)
                .expect("L-stream packet bound overflow");
            assert!(
                self.current >= self.base && packet_end <= end,
                "L-stream packet [{}, {packet_end}) is outside [{}, {end})",
                self.current,
                self.base
            );
        }
        self.current
    }

    fn affine_view(self) -> Option<AffineView> {
        (self.enabled() && self.flags & (AFFINE | PACKETIZED) != 0).then_some(AffineView {
            base: self.base,
            extent_minor: self.extent_minor,
            extent_major: self.extent_major,
            extent_field: self.extent_field,
            extent_group: self.extent_group,
            bank_row_pitch: self.bank_row_pitch,
            alpha: self.alpha,
            beta: self.beta,
            gamma: self.gamma,
            storage_atom: self.storage_atom,
            physical_base_row: self.physical_base_row,
            write: self.flags & WRITE != 0,
            lane_restore: self.flags & LANE_RESTORE != 0,
            packet_elements: self.packet_elements,
            packet_stride: self.packet_stride,
            packetized: self.packetized(),
        })
    }

    fn scalar_packet(self) -> Option<ScalarPacketView> {
        (self
            .target()
            .is_some_and(|target| matches!(target, StreamTarget::Fp(_)))
            && self.packetized())
        .then_some(ScalarPacketView {
            origin: self.checked_current(),
            packet_elements: self.packet_elements,
            storage_atom: self.storage_atom,
            packet_stride: self.packet_stride,
        })
    }
}

#[derive(Debug, Default)]
pub(super) struct StreamTable {
    slots: [StreamSlot; MAX_STREAM_SLOTS],
    banks: u32,
}

impl StreamTable {
    pub(super) fn new(banks: u32) -> Self {
        assert!(banks > 0, "L-stream bank count must be positive");
        Self {
            slots: [StreamSlot::default(); MAX_STREAM_SLOTS],
            banks,
        }
    }

    pub(super) fn configure(
        &mut self,
        value: u32,
        target: u8,
        slot: u8,
        field: ConfigField,
    ) -> Result<(), String> {
        let slot_index = slot as usize;
        let Some(entry) = self.slots.get(slot_index).copied() else {
            return Err(format!(
                "L-stream slot {slot} is outside 0..{MAX_STREAM_SLOTS}"
            ));
        };
        if field == ConfigField::Reset {
            self.slots[slot_index] = StreamSlot::default();
            return Ok(());
        }
        if let Some(previous) = entry.target
            && previous != target
        {
            return Err(format!(
                "L-stream slot {slot} changed target from {previous} to {target} without reset"
            ));
        }
        let mut candidate = entry;
        candidate.target = Some(target);
        match field {
            ConfigField::Reset => unreachable!(),
            ConfigField::Flags => candidate.flags = value,
            ConfigField::Base => {
                candidate.base = value;
                candidate.current = value;
                candidate.packet_index = 0;
            }
            ConfigField::ExtentMinor => candidate.extent_minor = value,
            ConfigField::ExtentMajor => candidate.extent_major = value,
            ConfigField::ExtentField => candidate.extent_field = value,
            ConfigField::ExtentGroup => candidate.extent_group = value,
            ConfigField::BankRowPitch => candidate.bank_row_pitch = value,
            ConfigField::Alpha => candidate.alpha = value,
            ConfigField::Beta => candidate.beta = value,
            ConfigField::Gamma => candidate.gamma = value,
            ConfigField::Advance => candidate.advance = value,
            ConfigField::PacketElements => candidate.packet_elements = value,
            ConfigField::StorageAtom => candidate.storage_atom = value,
            ConfigField::PhysicalBaseRow => candidate.physical_base_row = value,
            ConfigField::PacketStride => candidate.packet_stride = value,
        }
        if candidate.enabled() {
            candidate.validate(target, self.banks)?;
            let candidate_target = candidate
                .target()
                .expect("an enabled configured stream has a target");
            if self.slots.iter().enumerate().any(|(index, other)| {
                index != slot_index && other.target() == Some(candidate_target)
            }) {
                return Err(format!(
                    "L-stream target {candidate_target:?} is already bound by another slot"
                ));
            }
        }
        self.slots[slot_index] = candidate;
        Ok(())
    }

    pub(super) fn resolve_gp(&self, register: u8, fallback: u32) -> u32 {
        self.slots
            .iter()
            .find(|slot| slot.target() == Some(StreamTarget::Gp(register)) && slot.auto_advance())
            .map_or(fallback, |slot| slot.checked_current())
    }

    pub(super) fn fp_address(&self, register: u8) -> Option<u32> {
        self.slots
            .iter()
            .find(|slot| slot.target() == Some(StreamTarget::Fp(register)))
            .filter(|slot| !slot.packetized())
            .map(|slot| slot.checked_current())
    }

    pub(super) fn fp_packet(&self, register: u8) -> Option<ScalarPacketView> {
        self.slots
            .iter()
            .find(|slot| slot.target() == Some(StreamTarget::Fp(register)))
            .and_then(|slot| slot.scalar_packet())
    }

    pub(super) fn gp_affine_view(&self, register: u8) -> Option<AffineView> {
        self.slots
            .iter()
            .find(|slot| slot.target() == Some(StreamTarget::Gp(register)))
            .and_then(|slot| slot.affine_view())
    }

    pub(super) fn advance_targets(&mut self, targets: impl IntoIterator<Item = StreamTarget>) {
        let unique: BTreeSet<_> = targets.into_iter().collect();
        for slot in &mut self.slots {
            if slot.target().is_some_and(|target| unique.contains(&target)) {
                slot.advance();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn configured_slot(table: &mut StreamTable, target: u8, flags: u32) {
        for (field, value) in [
            (ConfigField::Base, 4096),
            (ConfigField::ExtentMinor, 64),
            (ConfigField::ExtentMajor, 8),
            (ConfigField::ExtentField, 1),
            (ConfigField::ExtentGroup, 1),
            (ConfigField::Alpha, 1),
            (ConfigField::Advance, 64),
            (ConfigField::PacketElements, 64),
            (ConfigField::StorageAtom, 4),
            (ConfigField::Flags, flags),
        ] {
            table.configure(value, target, 0, field).unwrap();
        }
    }

    #[test]
    fn affine_map_is_bijective_and_spreads_multirow_words() {
        let mut table = StreamTable::new(16);
        configured_slot(
            &mut table,
            3,
            ENABLE | AUTO_ADVANCE | AFFINE | LANE_RESTORE | STRICT_BOUNDS,
        );
        let slot = table.slots[0];
        let mut seen = BTreeSet::new();
        for major in 0..8 {
            for minor in 0..64 {
                assert!(seen.insert(slot.place(0, 0, major, minor, 16).unwrap()));
            }
        }
        let banks: BTreeSet<_> = (0..8)
            .map(|major| slot.place(0, 0, major, 0, 16).unwrap().bank)
            .collect();
        assert_eq!(banks.len(), 8);
    }

    #[test]
    fn gp_and_fp_targets_resolve_and_advance_once() {
        let mut gp_table = StreamTable::new(16);
        configured_slot(&mut gp_table, 3, ENABLE | AUTO_ADVANCE | STRICT_BOUNDS);
        assert_eq!(gp_table.resolve_gp(3, 7), 4096);
        gp_table.advance_targets([StreamTarget::Gp(3), StreamTarget::Gp(3)]);
        assert_eq!(gp_table.resolve_gp(3, 7), 4160);

        let mut fp_table = StreamTable::new(16);
        configured_slot(
            &mut fp_table,
            1,
            ENABLE | AUTO_ADVANCE | TARGET_FP | STRICT_BOUNDS,
        );
        assert_eq!(fp_table.fp_address(1), Some(4096));
        fp_table.advance_targets([StreamTarget::Fp(1)]);
        assert_eq!(fp_table.fp_address(1), Some(4160));
    }

    #[test]
    fn packet_cursor_walks_minor_atoms_then_the_next_row_block() {
        fn configure_packet(table: &mut StreamTable, target: u8, base: u32, stride: u32, fp: bool) {
            for (field, value) in [
                (ConfigField::Base, base),
                (ConfigField::ExtentMinor, 16),
                (ConfigField::ExtentMajor, 8),
                (ConfigField::Advance, if fp { 0 } else { 4 }),
                (ConfigField::PacketElements, 16),
                (ConfigField::StorageAtom, 4),
                (ConfigField::PacketStride, stride),
                (
                    ConfigField::Flags,
                    ENABLE
                        | AUTO_ADVANCE
                        | STRICT_BOUNDS
                        | LANE_RESTORE
                        | PACKETIZED
                        | if fp { TARGET_FP } else { 0 },
                ),
            ] {
                table.configure(value, target, 0, field).unwrap();
            }
        }

        let mut gp = StreamTable::new(4);
        configure_packet(&mut gp, 3, 4096, 16, false);
        let mut origins = Vec::new();
        for _ in 0..8 {
            origins.push(gp.resolve_gp(3, 0));
            gp.advance_targets([StreamTarget::Gp(3)]);
        }
        assert_eq!(
            origins,
            vec![4096, 4100, 4104, 4108, 4160, 4164, 4168, 4172]
        );

        let mut fp = StreamTable::new(4);
        configure_packet(&mut fp, 1, 100, 1, true);
        let mut scalar_origins = Vec::new();
        for _ in 0..8 {
            scalar_origins.push(fp.fp_packet(1).unwrap().origin);
            fp.advance_targets([StreamTarget::Fp(1)]);
        }
        assert_eq!(scalar_origins, vec![100, 100, 100, 100, 104, 104, 104, 104]);
    }

    #[test]
    fn packet_cursor_covers_field_and_group_extents() {
        let mut table = StreamTable::new(4);
        for (field, value) in [
            (ConfigField::Base, 4096),
            (ConfigField::ExtentMinor, 16),
            (ConfigField::ExtentMajor, 8),
            (ConfigField::ExtentField, 2),
            (ConfigField::ExtentGroup, 2),
            (ConfigField::Advance, 4),
            (ConfigField::PacketElements, 16),
            (ConfigField::StorageAtom, 4),
            (ConfigField::PacketStride, 16),
            (
                ConfigField::Flags,
                ENABLE | AUTO_ADVANCE | STRICT_BOUNDS | LANE_RESTORE | PACKETIZED,
            ),
        ] {
            table.configure(value, 3, 0, field).unwrap();
        }

        let mut origins = Vec::new();
        for _ in 0..32 {
            origins.push(table.resolve_gp(3, 0));
            table.advance_targets([StreamTarget::Gp(3)]);
        }
        assert_eq!(
            &origins[..8],
            &[4096, 4100, 4104, 4108, 4160, 4164, 4168, 4172]
        );
        assert_eq!(
            &origins[24..],
            &[4480, 4484, 4488, 4492, 4544, 4548, 4552, 4556]
        );
        assert_eq!(table.slots[0].packet_steps(), 32);
    }

    #[test]
    fn affine_only_stream_preserves_the_compiler_written_gp_pointer() {
        let mut table = StreamTable::new(16);
        configured_slot(
            &mut table,
            3,
            ENABLE | AFFINE | WRITE | LANE_RESTORE | STRICT_BOUNDS,
        );

        // Without AUTO_ADVANCE, ordinary compiler address arithmetic remains
        // authoritative. The stream contributes only the physical bank view.
        assert_eq!(table.resolve_gp(3, 4_224), 4_224);
        assert!(table.gp_affine_view(3).is_some());
        table.advance_targets([StreamTarget::Gp(3)]);
        assert_eq!(table.resolve_gp(3, 4_288), 4_288);
    }

    #[test]
    fn reset_defaults_form_a_valid_scalar_stream_and_target_change_fails() {
        let mut table = StreamTable::new(16);
        assert!(table.configure(ENABLE, 1, 0, ConfigField::Flags).is_ok());
        table.configure(0, 1, 0, ConfigField::Reset).unwrap();
        configured_slot(&mut table, 1, ENABLE);
        assert!(table.configure(ENABLE, 2, 0, ConfigField::Flags).is_err());
    }

    #[test]
    fn enabled_stream_revalidates_updates_without_poisoning_the_slot() {
        let mut table = StreamTable::new(16);
        configured_slot(&mut table, 3, ENABLE | AFFINE | LANE_RESTORE);
        let before = table.gp_affine_view(3).unwrap();

        assert!(table.configure(0, 3, 0, ConfigField::StorageAtom).is_err());
        assert_eq!(table.gp_affine_view(3), Some(before));
    }

    #[test]
    fn two_enabled_slots_cannot_ambiguously_bind_the_same_target() {
        let mut table = StreamTable::new(16);
        configured_slot(&mut table, 3, ENABLE | AUTO_ADVANCE | STRICT_BOUNDS);
        for (field, value) in [
            (ConfigField::Base, 8192),
            (ConfigField::ExtentMinor, 64),
            (ConfigField::PacketElements, 64),
            (ConfigField::StorageAtom, 4),
        ] {
            table.configure(value, 3, 1, field).unwrap();
        }
        assert!(
            table
                .configure(
                    ENABLE | AUTO_ADVANCE | STRICT_BOUNDS,
                    3,
                    1,
                    ConfigField::Flags
                )
                .is_err()
        );
        assert!(table.slots[1].target().is_none());
    }

    #[test]
    #[should_panic(expected = "L-stream packet")]
    fn strict_bounds_trap_before_an_extra_packet_is_consumed() {
        let mut table = StreamTable::new(16);
        configured_slot(&mut table, 3, ENABLE | AUTO_ADVANCE | STRICT_BOUNDS);
        for _ in 0..8 {
            assert!(table.resolve_gp(3, 0) >= 4096);
            table.advance_targets([StreamTarget::Gp(3)]);
        }
        let _ = table.resolve_gp(3, 0);
    }

    #[test]
    fn physical_buffer_roundtrips_values_and_detects_bad_accesses() {
        let mut table = StreamTable::new(16);
        configured_slot(
            &mut table,
            3,
            ENABLE | AUTO_ADVANCE | AFFINE | LANE_RESTORE | STRICT_BOUNDS,
        );
        let slot = table.slots[0];
        let mut buffer = BankedLayoutBuffer::new();
        for major in 0..8 {
            for minor in 0..64 {
                let value = (major * 64 + minor) as u64;
                let physical = slot.place(0, 0, major, minor, 16).unwrap();
                buffer.write(physical, value).unwrap();
            }
        }
        for major in 0..8 {
            for minor in 0..64 {
                let expected = (major * 64 + minor) as u64;
                let physical = slot.place(0, 0, major, minor, 16).unwrap();
                assert_eq!(buffer.read(physical).unwrap(), expected);
                assert!(buffer.write(physical, expected).is_err());
            }
        }
        assert!(
            buffer
                .read(PhysicalCoord {
                    bank: 15,
                    bank_row: 999,
                    sublane: 3,
                })
                .is_err()
        );
    }

    #[test]
    fn skewed_multirow_packet_reaches_the_bandwidth_floor() {
        let mut table = StreamTable::new(16);
        configured_slot(
            &mut table,
            3,
            ENABLE | AUTO_ADVANCE | AFFINE | LANE_RESTORE | STRICT_BOUNDS,
        );
        let slot = table.slots[0];
        let row: Vec<_> = (0..8)
            .flat_map(|major| (0..4).map(move |minor| slot.place(0, 0, major, minor, 16).unwrap()))
            .collect();
        let stats = packet_service(&row, 16, 1).unwrap();
        assert_eq!(stats.values, 32);
        assert_eq!(stats.bank_words, 8);
        assert_eq!(stats.bandwidth_floor_cycles, 1);
        assert_eq!(stats.service_cycles, 1);
        assert_eq!(stats.conflict_stall_cycles(), 0);

        let row_major: Vec<_> = (0..8)
            .flat_map(|major| {
                (0..4).map(move |minor| PhysicalCoord {
                    bank: 0,
                    bank_row: major,
                    sublane: minor,
                })
            })
            .collect();
        let baseline = packet_service(&row_major, 16, 1).unwrap();
        assert_eq!(baseline.bandwidth_floor_cycles, 1);
        assert_eq!(baseline.service_cycles, 8);
        assert_eq!(baseline.conflict_stall_cycles(), 7);
    }
}
