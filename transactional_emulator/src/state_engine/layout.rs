//! Programmable Matrix-writeback layout engine for L_SCATTER_M.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use sram::VectorSram;

use super::descriptor::StateDescriptor;
use super::error::StateEngineError;
use super::generated_contract::StatePrecision;
use super::projection::ProjectionBufferStats;
use super::vram::{BlockedRecord, VramAccess};

pub const DESCRIPTOR_MAGIC: u32 = 0x314d_534c;
pub const DESCRIPTOR_VERSION: u16 = 1;
pub const DESCRIPTOR_SIZE: usize = 256;
pub const DESCRIPTOR_ALIGNMENT: u64 = 64;
const FIELD_OFFSET: usize = 80;
const FIELD_SIZE: usize = 24;
const MAX_FIELDS: usize = 7;
const HEAD_LANES: usize = 8;
const HEAD_DIM_LANES: usize = 4;
const STATE_DIM_LANES: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum LayoutMode {
    RowMajor = 0,
    Transpose = 1,
    MambaSkew = 2,
    KdaSkew = 3,
    Custom = 4,
}

impl TryFrom<u8> for LayoutMode {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::RowMajor),
            1 => Ok(Self::Transpose),
            2 => Ok(Self::MambaSkew),
            3 => Ok(Self::KdaSkew),
            4 => Ok(Self::Custom),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum LayoutFlow {
    Buffered = 0,
    FifoWithSpill = 1,
}

impl TryFrom<u8> for LayoutFlow {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Buffered),
            1 => Ok(Self::FifoWithSpill),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum LayoutConsumer {
    State = 0,
    Vector = 1,
}

impl TryFrom<u8> for LayoutConsumer {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::State),
            1 => Ok(Self::Vector),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum LayoutSkew {
    None = 0,
    LocalRow = 1,
    Field = 2,
    Group = 3,
}

impl TryFrom<u8> for LayoutSkew {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::LocalRow),
            2 => Ok(Self::Field),
            3 => Ok(Self::Group),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
enum FieldId {
    MambaGate = 0,
    MambaX = 1,
    MambaB = 2,
    MambaC = 3,
    MambaDt = 4,
    KdaQ = 16,
    KdaK = 17,
    KdaV = 18,
    KdaDecay = 19,
    KdaBeta = 20,
}

impl TryFrom<u8> for FieldId {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::MambaGate),
            1 => Ok(Self::MambaX),
            2 => Ok(Self::MambaB),
            3 => Ok(Self::MambaC),
            4 => Ok(Self::MambaDt),
            16 => Ok(Self::KdaQ),
            17 => Ok(Self::KdaK),
            18 => Ok(Self::KdaV),
            19 => Ok(Self::KdaDecay),
            20 => Ok(Self::KdaBeta),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LayoutField {
    id: FieldId,
    consumer: LayoutConsumer,
    skew: LayoutSkew,
    skew_stride: usize,
    source_offset: usize,
    values_per_group: usize,
    physical_offset: usize,
    physical_span: usize,
    local_rows: usize,
    local_lanes: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayoutDescriptor {
    pub context_id: u32,
    request_id: u32,
    layer_id: u32,
    token_offset: u32,
    source_vram_addr: u32,
    source_token_stride: usize,
    source_values_per_token: usize,
    logical_rows: usize,
    logical_cols: usize,
    valid_tokens: usize,
    chunk_size: usize,
    batch_size: usize,
    groups: usize,
    physical_buffer_base_row: usize,
    physical_token_stride_rows: usize,
    physical_buffer_rows: usize,
    group_span_values: usize,
    banks: usize,
    ports_per_bank: usize,
    pub mode: LayoutMode,
    activation_precision: StatePrecision,
    pub buffer_id: u8,
    flow: LayoutFlow,
    spill_write_values_per_cycle: usize,
    producer_burst_values: usize,
    fifo_capacity_values: usize,
    fields: Vec<LayoutField>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Mapping {
    source: usize,
    row: usize,
    bank: usize,
}

impl LayoutDescriptor {
    pub fn parse(data: &[u8]) -> Result<Self, StateEngineError> {
        if data.len() != DESCRIPTOR_SIZE {
            return Err(StateEngineError::invalid(
                "L_SCATTER_M descriptor must be exactly 256 bytes",
            ));
        }
        if u32_at(data, 0) != DESCRIPTOR_MAGIC
            || u16_at(data, 4) != DESCRIPTOR_VERSION
            || usize::from(u16_at(data, 6)) != DESCRIPTOR_SIZE
        {
            return Err(StateEngineError::invalid(
                "incompatible L_SCATTER_M descriptor header",
            ));
        }
        let mode_raw = data[67];
        let precision_raw = data[68];
        let flow_raw = data[70];
        let mode = LayoutMode::try_from(mode_raw)
            .map_err(|_| StateEngineError::invalid("unknown layout mode"))?;
        let activation_precision = StatePrecision::try_from(precision_raw)
            .map_err(|_| StateEngineError::unsupported_precision("unknown layout precision"))?;
        if activation_precision == StatePrecision::Mx8B128 {
            return Err(StateEngineError::unsupported_precision(
                "MX8 layout ingress requires scale-aware Vector SRAM support",
            ));
        }
        let flow = LayoutFlow::try_from(flow_raw)
            .map_err(|_| StateEngineError::invalid("unknown layout flow mode"))?;
        let field_count = usize::from(data[66]);
        if field_count > MAX_FIELDS {
            return Err(StateEngineError::invalid(
                "layout descriptor has too many fields",
            ));
        }
        let mut fields = Vec::with_capacity(field_count);
        for index in 0..field_count {
            let start = FIELD_OFFSET + index * FIELD_SIZE;
            let field = LayoutField {
                id: FieldId::try_from(data[start])
                    .map_err(|_| StateEngineError::invalid("unknown layout field id"))?,
                consumer: LayoutConsumer::try_from(data[start + 1])
                    .map_err(|_| StateEngineError::invalid("unknown layout consumer"))?,
                skew: LayoutSkew::try_from(data[start + 2])
                    .map_err(|_| StateEngineError::invalid("unknown layout skew"))?,
                skew_stride: usize::from(data[start + 3]),
                source_offset: usize_at_u32(data, start + 4)?,
                values_per_group: usize_at_u32(data, start + 8)?,
                physical_offset: usize_at_u32(data, start + 12)?,
                physical_span: usize_at_u32(data, start + 16)?,
                local_rows: usize::from(u16_at(data, start + 20)),
                local_lanes: usize::from(u16_at(data, start + 22)),
            };
            if field.values_per_group == 0
                || field.local_rows.checked_mul(field.local_lanes) != Some(field.values_per_group)
                || field.physical_span < field.values_per_group
            {
                return Err(StateEngineError::invalid("invalid layout field shape"));
            }
            fields.push(field);
        }
        let used = FIELD_OFFSET + field_count * FIELD_SIZE;
        if data[used..].iter().any(|&value| value != 0) {
            return Err(StateEngineError::invalid(
                "layout descriptor unused field bytes must be zero",
            ));
        }
        let descriptor = Self {
            context_id: u32_at(data, 8),
            request_id: u32_at(data, 12),
            layer_id: u32_at(data, 16),
            token_offset: u32_at(data, 20),
            source_vram_addr: u32_at(data, 24),
            source_token_stride: usize_at_u32(data, 28)?,
            source_values_per_token: usize_at_u32(data, 32)?,
            logical_rows: usize::from(u16_at(data, 36)),
            logical_cols: usize::from(u16_at(data, 38)),
            valid_tokens: usize::from(u16_at(data, 40)),
            chunk_size: usize::from(u16_at(data, 42)),
            batch_size: usize::from(u16_at(data, 44)),
            groups: usize::from(u16_at(data, 46)),
            physical_buffer_base_row: usize_at_u32(data, 48)?,
            physical_token_stride_rows: usize_at_u32(data, 52)?,
            physical_buffer_rows: usize_at_u32(data, 56)?,
            group_span_values: usize_at_u32(data, 60)?,
            banks: usize::from(data[64]),
            ports_per_bank: usize::from(data[65]),
            mode,
            activation_precision,
            buffer_id: data[69],
            flow,
            spill_write_values_per_cycle: usize::from(data[71]),
            producer_burst_values: usize::from(u16_at(data, 72)),
            fifo_capacity_values: usize::from(u16_at(data, 74)),
            fields,
        };
        descriptor.validate()?;
        let expected_crc = u32_at(data, 76);
        if descriptor.mapping_crc32()? != expected_crc {
            return Err(StateEngineError::invalid(
                "layout descriptor mapping CRC does not match",
            ));
        }
        Ok(descriptor)
    }

    fn validate(&self) -> Result<(), StateEngineError> {
        if self.banks == 0 || !self.banks.is_power_of_two() {
            return Err(StateEngineError::invalid(
                "layout banks must be a nonzero power of two",
            ));
        }
        if self.ports_per_bank == 0
            || self.valid_tokens == 0
            || self.chunk_size == 0
            || self.batch_size == 0
            || self.groups == 0
            || self.producer_burst_values == 0
            || self.fifo_capacity_values == 0
            || self.spill_write_values_per_cycle == 0
        {
            return Err(StateEngineError::invalid(
                "layout dimensions and service widths must be positive",
            ));
        }
        if self.valid_tokens > self.chunk_size
            || self.source_token_stride < self.source_values_per_token
            || self.logical_rows.checked_mul(self.logical_cols)
                != Some(self.source_values_per_token)
            || self.physical_buffer_rows
                < self
                    .batch_size
                    .checked_mul(self.chunk_size)
                    .and_then(|value| value.checked_mul(self.physical_token_stride_rows))
                    .ok_or_else(|| StateEngineError::address("layout dimensions overflow"))?
        {
            return Err(StateEngineError::invalid("inconsistent layout dimensions"));
        }
        if self.mode == LayoutMode::Transpose && !self.fields.is_empty() {
            return Err(StateEngineError::invalid(
                "transpose layout must not carry fields",
            ));
        }
        if matches!(
            self.mode,
            LayoutMode::MambaSkew | LayoutMode::KdaSkew | LayoutMode::Custom
        ) && self.fields.is_empty()
        {
            return Err(StateEngineError::invalid("skewed layout requires fields"));
        }
        let mapping = self.mapping()?;
        let sources = mapping
            .iter()
            .map(|item| item.source)
            .collect::<HashSet<_>>();
        let cells = mapping
            .iter()
            .map(|item| (item.row, item.bank))
            .collect::<HashSet<_>>();
        if sources.len() != self.source_values_per_token
            || !(0..self.source_values_per_token).all(|source| sources.contains(&source))
        {
            return Err(StateEngineError::invalid(
                "layout does not cover each source exactly once",
            ));
        }
        if cells.len() != mapping.len() {
            return Err(StateEngineError::invalid("layout aliases two sources"));
        }
        let token_end = self
            .physical_buffer_base_row
            .checked_add(self.physical_token_stride_rows)
            .ok_or_else(|| StateEngineError::address("layout row range overflows"))?;
        if mapping.iter().any(|item| {
            item.row < self.physical_buffer_base_row
                || item.row >= token_end
                || item.bank >= self.banks
        }) {
            return Err(StateEngineError::invalid(
                "layout mapping exceeds one physical token slot",
            ));
        }
        Ok(())
    }

    fn mapping(&self) -> Result<Vec<Mapping>, StateEngineError> {
        match self.mode {
            LayoutMode::RowMajor => Ok((0..self.source_values_per_token)
                .map(|source| Mapping {
                    source,
                    row: self.physical_buffer_base_row + source / self.banks,
                    bank: source % self.banks,
                })
                .collect()),
            LayoutMode::Transpose => {
                let mut result = Vec::with_capacity(self.source_values_per_token);
                for row in 0..self.logical_rows {
                    for column in 0..self.logical_cols {
                        let source = row * self.logical_cols + column;
                        let physical = column * self.logical_rows + row;
                        result.push(Mapping {
                            source,
                            row: self.physical_buffer_base_row + physical / self.banks,
                            bank: physical % self.banks,
                        });
                    }
                }
                Ok(result)
            }
            LayoutMode::MambaSkew | LayoutMode::KdaSkew | LayoutMode::Custom => {
                let mut result = Vec::with_capacity(self.source_values_per_token);
                for field in &self.fields {
                    for group in 0..self.groups {
                        for local_row in 0..field.local_rows {
                            for lane in 0..field.local_lanes {
                                let local = local_row * field.local_lanes + lane;
                                let source =
                                    field.source_offset + group * field.values_per_group + local;
                                let physical =
                                    group * self.group_span_values + field.physical_offset + local;
                                let skew = match field.skew {
                                    LayoutSkew::None => 0,
                                    LayoutSkew::LocalRow => local_row * field.skew_stride,
                                    LayoutSkew::Field => field.skew_stride,
                                    LayoutSkew::Group => group * field.skew_stride,
                                };
                                result.push(Mapping {
                                    source,
                                    row: self.physical_buffer_base_row + physical / self.banks,
                                    bank: (local % self.banks + skew) % self.banks,
                                });
                            }
                        }
                    }
                }
                Ok(result)
            }
        }
    }

    fn mapping_crc32(&self) -> Result<u32, StateEngineError> {
        let mut crc = 0xffff_ffffu32;
        for item in self.mapping()? {
            for byte in (item.source as u32)
                .to_le_bytes()
                .into_iter()
                .chain((item.row as u32).to_le_bytes())
                .chain((item.bank as u32).to_le_bytes())
            {
                crc ^= u32::from(byte);
                for _ in 0..8 {
                    crc = if crc & 1 != 0 {
                        (crc >> 1) ^ 0xedb8_8320
                    } else {
                        crc >> 1
                    };
                }
            }
        }
        Ok(!crc)
    }

    fn field(&self, id: FieldId) -> Result<&LayoutField, StateEngineError> {
        self.fields
            .iter()
            .find(|field| field.id == id)
            .ok_or_else(|| StateEngineError::invalid(format!("layout omits field {id:?}")))
    }

    fn address(&self, field: &LayoutField, group: usize, local_row: usize, lane: usize) -> Mapping {
        let local = local_row * field.local_lanes + lane;
        let physical = group * self.group_span_values + field.physical_offset + local;
        let skew = match field.skew {
            LayoutSkew::None => 0,
            LayoutSkew::LocalRow => local_row * field.skew_stride,
            LayoutSkew::Field => field.skew_stride,
            LayoutSkew::Group => group * field.skew_stride,
        };
        Mapping {
            source: field.source_offset + group * field.values_per_group + local,
            row: self.physical_buffer_base_row + physical / self.banks,
            bank: (local % self.banks + skew) % self.banks,
        }
    }

    fn consumer_packets(&self) -> Result<Vec<Vec<Mapping>>, StateEngineError> {
        if !matches!(
            self.mode,
            LayoutMode::MambaSkew | LayoutMode::KdaSkew | LayoutMode::Custom | LayoutMode::RowMajor
        ) {
            return Err(StateEngineError::invalid(
                "transpose layout has no X_STATE consumer packet definition",
            ));
        }
        if self.fields.is_empty() {
            return Err(StateEngineError::invalid(
                "X_STATE layout requires field metadata",
            ));
        }
        if self.fields.iter().any(|field| {
            matches!(
                field.id,
                FieldId::KdaQ
                    | FieldId::KdaK
                    | FieldId::KdaV
                    | FieldId::KdaDecay
                    | FieldId::KdaBeta
            )
        }) {
            self.kda_packets()
        } else {
            self.mamba_packets()
        }
    }

    fn resolve(
        &self,
        coordinates: &[(FieldId, usize, usize, usize)],
    ) -> Result<Vec<Mapping>, StateEngineError> {
        coordinates
            .iter()
            .map(|&(id, group, local_row, lane)| {
                let field = self.field(id)?;
                if group >= self.groups
                    || local_row >= field.local_rows
                    || lane >= field.local_lanes
                {
                    return Err(StateEngineError::invalid(
                        "consumer coordinate exceeds layout field",
                    ));
                }
                if self.mode == LayoutMode::RowMajor {
                    let source = field.source_offset
                        + group * field.values_per_group
                        + local_row * field.local_lanes
                        + lane;
                    return Ok(Mapping {
                        source,
                        row: self.physical_buffer_base_row + source / self.banks,
                        bank: source % self.banks,
                    });
                }
                Ok(self.address(field, group, local_row, lane))
            })
            .collect()
    }

    fn kda_packets(&self) -> Result<Vec<Vec<Mapping>>, StateEngineError> {
        let key_dim = self.field(FieldId::KdaQ)?.local_lanes;
        let value_dim = self.field(FieldId::KdaV)?.local_lanes;
        let mut packets = Vec::new();
        for head in 0..self.groups {
            packets.push(self.resolve(&[(FieldId::KdaBeta, head, 0, 0)])?);
            for key_start in (0..key_dim).step_by(STATE_DIM_LANES) {
                let mut coordinates = Vec::new();
                for id in [FieldId::KdaQ, FieldId::KdaK, FieldId::KdaDecay] {
                    for key in key_start..(key_start + STATE_DIM_LANES).min(key_dim) {
                        coordinates.push((id, head, 0, key));
                    }
                }
                packets.push(self.resolve(&coordinates)?);
            }
            for value_start in (0..value_dim).step_by(HEAD_DIM_LANES) {
                let coordinates = (value_start..(value_start + HEAD_DIM_LANES).min(value_dim))
                    .map(|value| (FieldId::KdaV, head, 0, value))
                    .collect::<Vec<_>>();
                packets.push(self.resolve(&coordinates)?);
            }
        }
        Ok(packets)
    }

    fn mamba_packets(&self) -> Result<Vec<Vec<Mapping>>, StateEngineError> {
        let heads = self.field(FieldId::MambaX)?.local_rows;
        let head_dim = self.field(FieldId::MambaX)?.local_lanes;
        let state_dim = self.field(FieldId::MambaB)?.local_lanes;
        let mut packets = Vec::new();
        for group in 0..self.groups {
            for head_start in (0..heads).step_by(HEAD_LANES) {
                let local_heads = head_start..(head_start + HEAD_LANES).min(heads);
                packets.push(
                    self.resolve(
                        &local_heads
                            .clone()
                            .map(|head| (FieldId::MambaDt, group, head, 0))
                            .collect::<Vec<_>>(),
                    )?,
                );
                for dim_start in (0..head_dim).step_by(HEAD_DIM_LANES) {
                    for id in [FieldId::MambaX, FieldId::MambaGate] {
                        let mut coordinates = Vec::new();
                        for head in local_heads.clone() {
                            for dim in dim_start..(dim_start + HEAD_DIM_LANES).min(head_dim) {
                                coordinates.push((id, group, head, dim));
                            }
                        }
                        packets.push(self.resolve(&coordinates)?);
                    }
                }
            }
            for state_start in (0..state_dim).step_by(STATE_DIM_LANES) {
                let mut coordinates = Vec::new();
                for id in [FieldId::MambaB, FieldId::MambaC] {
                    for state in state_start..(state_start + STATE_DIM_LANES).min(state_dim) {
                        coordinates.push((id, group, 0, state));
                    }
                }
                packets.push(self.resolve(&coordinates)?);
            }
        }
        Ok(packets)
    }

    fn key(&self) -> LayoutKey {
        LayoutKey {
            context_id: self.context_id,
            request_id: self.request_id,
            layer_id: self.layer_id,
            token_offset: self.token_offset,
            source_vram_addr: self.source_vram_addr,
        }
    }

    fn matches_state(&self, state: &StateDescriptor) -> bool {
        self.context_id == state.identity.context_id
            && self.request_id == state.identity.request_id
            && self.layer_id == state.identity.layer_id
            && self.token_offset == state.token_offset
            && self.source_vram_addr == state.input_vram_addr
            && self.source_token_stride == state.input_token_stride as usize
            && self.source_values_per_token == state.input_elements() as usize
            && self.valid_tokens == usize::from(state.valid_tokens)
            && self.chunk_size == usize::from(state.chunk_size)
            && self.batch_size == usize::from(state.batch_size)
            && self.activation_precision == state.activation_precision
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct LayoutKey {
    context_id: u32,
    request_id: u32,
    layer_id: u32,
    token_offset: u32,
    source_vram_addr: u32,
}

struct BankedLayoutBuffer {
    base_row: usize,
    banks: usize,
    cells: Vec<Option<f32>>,
}

impl BankedLayoutBuffer {
    fn new(base_row: usize, rows: usize, banks: usize) -> Result<Self, StateEngineError> {
        let cells = rows
            .checked_mul(banks)
            .ok_or_else(|| StateEngineError::address("banked layout allocation overflows"))?;
        Ok(Self {
            base_row,
            banks,
            cells: vec![None; cells],
        })
    }

    fn index(&self, row: usize, bank: usize) -> Result<usize, StateEngineError> {
        let local_row = row
            .checked_sub(self.base_row)
            .ok_or_else(|| StateEngineError::address("layout row precedes buffer"))?;
        let index = local_row
            .checked_mul(self.banks)
            .and_then(|value| value.checked_add(bank))
            .ok_or_else(|| StateEngineError::address("layout cell index overflows"))?;
        if bank >= self.banks || index >= self.cells.len() {
            return Err(StateEngineError::address("layout cell exceeds buffer"));
        }
        Ok(index)
    }

    fn write(&mut self, row: usize, bank: usize, value: f32) -> Result<(), StateEngineError> {
        let index = self.index(row, bank)?;
        if self.cells[index].replace(value).is_some() {
            return Err(StateEngineError::internal(format!(
                "layout writes cell ({row}, {bank}) twice"
            )));
        }
        Ok(())
    }

    fn read(&self, row: usize, bank: usize) -> Result<f32, StateEngineError> {
        let index = self.index(row, bank)?;
        self.cells[index].ok_or_else(|| {
            StateEngineError::internal(format!("layout reads unwritten cell ({row}, {bank})"))
        })
    }
}

struct StagedLayout {
    descriptor: LayoutDescriptor,
    buffer: BankedLayoutBuffer,
    consumed: Vec<bool>,
    write_stats: Option<ProjectionBufferStats>,
}

impl StagedLayout {
    async fn from_vram(
        descriptor: LayoutDescriptor,
        vram: &Arc<VectorSram>,
    ) -> Result<Self, StateEngineError> {
        let access = VramAccess::new(vram, descriptor.activation_precision)?;
        let records = descriptor
            .valid_tokens
            .checked_mul(descriptor.batch_size)
            .ok_or_else(|| StateEngineError::address("layout record count overflows"))?;
        let mapping = descriptor.mapping()?;
        let mut buffer = BankedLayoutBuffer::new(
            descriptor.physical_buffer_base_row,
            descriptor.physical_buffer_rows,
            descriptor.banks,
        )?;
        let mut stats = ProjectionBufferStats {
            fifo_capacity_values: descriptor.fifo_capacity_values as u64,
            ..ProjectionBufferStats::default()
        };
        for local_token in 0..descriptor.valid_tokens {
            for batch_index in 0..descriptor.batch_size {
                let record = local_token * descriptor.batch_size + batch_index;
                let values = access
                    .read_blocked_record(
                        descriptor.source_vram_addr,
                        descriptor.source_token_stride,
                        BlockedRecord::new(
                            local_token,
                            batch_index,
                            descriptor.batch_size,
                            descriptor.valid_tokens,
                            descriptor.chunk_size,
                        ),
                    )
                    .await?;
                let row_offset = record * descriptor.physical_token_stride_rows;
                let mut source_banks = vec![usize::MAX; descriptor.source_values_per_token];
                for item in &mapping {
                    buffer.write(item.row + row_offset, item.bank, values[item.source])?;
                    source_banks[item.source] = item.bank;
                }
                stats.values += descriptor.source_values_per_token as u64;
                let mut record_write_packets = 0_u64;
                for banks in source_banks.chunks(descriptor.producer_burst_values) {
                    let (ideal, service) =
                        service_cycles(banks, descriptor.banks, descriptor.ports_per_bank);
                    record_write_packets += 1;
                    stats.write_packets += 1;
                    stats.write_ideal_cycles += ideal;
                    stats.write_service_cycles += service;
                }
                let forced_spill = match descriptor.flow {
                    LayoutFlow::Buffered => descriptor.source_values_per_token,
                    LayoutFlow::FifoWithSpill => descriptor
                        .fields
                        .iter()
                        .filter(|field| field.consumer == LayoutConsumer::Vector)
                        .map(|field| field.values_per_group * descriptor.groups)
                        .sum(),
                };
                stats.fifo_spill_values += forced_spill as u64;
                stats.fifo_peak_values = stats.fifo_peak_values.max(
                    descriptor
                        .producer_burst_values
                        .min(descriptor.fifo_capacity_values) as u64,
                );
                let spill_cycles =
                    forced_spill.div_ceil(descriptor.spill_write_values_per_cycle) as u64;
                stats.fifo_backpressure_cycles +=
                    spill_cycles.saturating_sub(record_write_packets.min(spill_cycles));
            }
        }
        Ok(Self {
            descriptor,
            buffer,
            consumed: vec![false; records],
            write_stats: Some(stats),
        })
    }

    fn consume_record(
        &mut self,
        state: &StateDescriptor,
        local_token: usize,
        batch_index: usize,
    ) -> Result<(Vec<f32>, ProjectionBufferStats), StateEngineError> {
        if !self.descriptor.matches_state(state) {
            return Err(StateEngineError::hazard(
                "L_SCATTER_M descriptor does not match X_STATE descriptor",
            ));
        }
        if local_token >= self.descriptor.valid_tokens || batch_index >= self.descriptor.batch_size
        {
            return Err(StateEngineError::internal(
                "layout consumer record is out of range",
            ));
        }
        let record = local_token * self.descriptor.batch_size + batch_index;
        if std::mem::replace(&mut self.consumed[record], true) {
            return Err(StateEngineError::hazard(
                "X_STATE consumed one layout record twice",
            ));
        }
        let mut restored = vec![0.0; self.descriptor.source_token_stride];
        let mut seen = vec![false; self.descriptor.source_values_per_token];
        let mut stats = self.write_stats.take().unwrap_or_default();
        let row_offset = record * self.descriptor.physical_token_stride_rows;
        for packet in self.descriptor.consumer_packets()? {
            let mut banks = Vec::with_capacity(packet.len());
            for item in packet {
                if std::mem::replace(&mut seen[item.source], true) {
                    return Err(StateEngineError::internal(
                        "layout consumer reads one source twice",
                    ));
                }
                restored[item.source] = self.buffer.read(item.row + row_offset, item.bank)?;
                banks.push(item.bank);
            }
            let (ideal, service) = service_cycles(
                &banks,
                self.descriptor.banks,
                self.descriptor.ports_per_bank,
            );
            stats.read_packets += 1;
            stats.read_ideal_cycles += ideal;
            stats.read_service_cycles += service;
        }
        if seen.iter().any(|&value| !value) {
            return Err(StateEngineError::internal(
                "layout consumer omitted live values",
            ));
        }
        Ok((restored, stats))
    }

    fn complete(&self) -> bool {
        self.consumed.iter().all(|&value| value)
    }
}

#[derive(Default)]
pub struct LayoutStore {
    staged: HashMap<LayoutKey, StagedLayout>,
}

impl LayoutStore {
    pub async fn stage(
        &mut self,
        descriptor: LayoutDescriptor,
        vram: &Arc<VectorSram>,
    ) -> Result<ProjectionBufferStats, StateEngineError> {
        let key = descriptor.key();
        if self.staged.contains_key(&key) {
            return Err(StateEngineError::hazard(
                "L_SCATTER_M overwrites an unconsumed layout",
            ));
        }
        let staged = StagedLayout::from_vram(descriptor, vram).await?;
        let stats = staged.write_stats.unwrap_or_default();
        self.staged.insert(key, staged);
        Ok(stats)
    }

    pub fn contains(&self, descriptor: &StateDescriptor) -> bool {
        self.staged.contains_key(&state_key(descriptor))
    }

    pub fn consume_record(
        &mut self,
        descriptor: &StateDescriptor,
        local_token: usize,
        batch_index: usize,
    ) -> Result<(Vec<f32>, ProjectionBufferStats), StateEngineError> {
        let key = state_key(descriptor);
        let staged = self
            .staged
            .get_mut(&key)
            .ok_or_else(|| StateEngineError::hazard("X_STATE has no staged layout"))?;
        let result = staged.consume_record(descriptor, local_token, batch_index)?;
        if staged.complete() {
            self.staged.remove(&key);
        }
        Ok(result)
    }
}

fn state_key(descriptor: &StateDescriptor) -> LayoutKey {
    LayoutKey {
        context_id: descriptor.identity.context_id,
        request_id: descriptor.identity.request_id,
        layer_id: descriptor.identity.layer_id,
        token_offset: descriptor.token_offset,
        source_vram_addr: descriptor.input_vram_addr,
    }
}

fn service_cycles(banks: &[usize], bank_count: usize, ports: usize) -> (u64, u64) {
    if banks.is_empty() {
        return (0, 0);
    }
    let mut counts = vec![0usize; bank_count];
    for &bank in banks {
        counts[bank] += 1;
    }
    let ideal = banks.len().div_ceil(bank_count * ports) as u64;
    let service = counts.into_iter().max().unwrap_or(0).div_ceil(ports) as u64;
    (ideal, service)
}

fn u16_at(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap())
}

fn u32_at(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

fn usize_at_u32(data: &[u8], offset: usize) -> Result<usize, StateEngineError> {
    usize::try_from(u32_at(data, offset))
        .map_err(|_| StateEngineError::address("layout u32 does not fit host usize"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;
    use quantize::{DataType, FpType, QuantTensor, tensor_from_f32_slice};

    fn decode_hex(value: &str) -> Vec<u8> {
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|digits| u8::from_str_radix(std::str::from_utf8(digits).unwrap(), 16).unwrap())
            .collect()
    }

    fn golden() -> (StateDescriptor, Vec<u8>, Vec<f32>) {
        let document: serde_json::Value =
            serde_json::from_str(include_str!("../../testdata/l_scatter_m_v1_golden.json"))
                .unwrap();
        let item = &document["kda_tiny"];
        let state =
            StateDescriptor::parse(&decode_hex(item["state_hex"].as_str().unwrap())).unwrap();
        let layout = decode_hex(item["layout_hex"].as_str().unwrap());
        let projected = item["projected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect();
        (state, layout, projected)
    }

    async fn write_blocked_record(vram: &Arc<VectorSram>, values: &[f32]) {
        let vlen = vram.tile_size() as usize;
        let padded_records = 1;
        for (feature_tile, chunk) in values.chunks(vlen).enumerate() {
            let address = (feature_tile * padded_records * vlen) as u32;
            let mut padded = vec![0.0; vlen];
            padded[..chunk.len()].copy_from_slice(chunk);
            vram.write(
                address,
                QuantTensor::quantize(tensor_from_f32_slice(&padded), vram.ty()),
            )
            .await;
        }
    }

    async fn write_blocked_records(vram: &Arc<VectorSram>, records: &[&[f32]]) {
        let vlen = vram.tile_size() as usize;
        let physical_rows = records.len();
        for (record, values) in records.iter().enumerate() {
            for (feature_tile, chunk) in values.chunks(vlen).enumerate() {
                let address = (feature_tile * physical_rows * vlen + record * vlen) as u32;
                let mut padded = vec![0.0; vlen];
                padded[..chunk.len()].copy_from_slice(chunk);
                vram.write(
                    address,
                    QuantTensor::quantize(tensor_from_f32_slice(&padded), vram.ty()),
                )
                .await;
            }
        }
    }

    #[test]
    fn compiler_descriptor_parses_and_crc_is_live() {
        let (_, bytes, _) = golden();
        let descriptor = LayoutDescriptor::parse(&bytes).unwrap();
        assert_eq!(descriptor.mode, LayoutMode::KdaSkew);
        assert_eq!(descriptor.source_values_per_token, 9);
        assert_eq!(descriptor.banks, 16);

        let mut corrupted = bytes;
        corrupted[82] = LayoutSkew::Field as u8;
        corrupted[83] = 1;
        assert!(LayoutDescriptor::parse(&corrupted).is_err());
    }

    #[test]
    fn parser_rejects_a_mapping_that_aliases_two_sources() {
        let (_, mut bytes, _) = golden();
        let second = FIELD_OFFSET + FIELD_SIZE;
        bytes[second + 2] = LayoutSkew::None as u8;
        bytes[second + 3] = 0;
        bytes[second + 12..second + 16].copy_from_slice(&0u32.to_le_bytes());
        let error = LayoutDescriptor::parse(&bytes).unwrap_err();
        assert!(error.to_string().contains("aliases"));
    }

    #[test]
    fn physical_buffer_rejects_duplicate_write_and_unwritten_read() {
        let mut buffer = BankedLayoutBuffer::new(10, 2, 16).unwrap();
        buffer.write(10, 3, 1.0).unwrap();
        assert!(buffer.write(10, 3, 2.0).is_err());
        assert!(buffer.read(10, 4).is_err());
    }

    #[test]
    fn transpose_mapping_roundtrips_a_dense_tile_and_makes_columns_conflict_free() {
        let (_, bytes, _) = golden();
        let mut descriptor = LayoutDescriptor::parse(&bytes).unwrap();
        descriptor.mode = LayoutMode::Transpose;
        descriptor.fields.clear();
        descriptor.source_values_per_token = 16 * 128;
        descriptor.source_token_stride = 16 * 128;
        descriptor.logical_rows = 16;
        descriptor.logical_cols = 128;
        descriptor.physical_token_stride_rows = 128;
        descriptor.physical_buffer_rows = 128;

        let mapping = descriptor.mapping().unwrap();
        assert_eq!(mapping.len(), 16 * 128);
        assert_eq!(
            mapping
                .iter()
                .map(|item| (item.row, item.bank))
                .collect::<std::collections::HashSet<_>>()
                .len(),
            mapping.len()
        );
        let mut buffer = BankedLayoutBuffer::new(0, 128, 16).unwrap();
        for item in &mapping {
            buffer
                .write(item.row, item.bank, item.source as f32)
                .unwrap();
        }
        for item in &mapping {
            assert_eq!(
                buffer.read(item.row, item.bank).unwrap(),
                item.source as f32
            );
        }

        let by_source = mapping.iter().map(|item| item.bank).collect::<Vec<_>>();
        let mut ideal = 0;
        let mut service = 0;
        for column in 0..128 {
            let banks = (0..16)
                .map(|row| by_source[row * 128 + column])
                .collect::<Vec<_>>();
            let cycles = service_cycles(&banks, 16, 1);
            ideal += cycles.0;
            service += cycles.1;
        }
        assert_eq!((ideal, service), (128, 128));
    }

    #[tokio::test]
    async fn fifo_backpressure_is_counted_per_record_not_cumulatively() {
        let (_, bytes, projected) = golden();
        let mut descriptor = LayoutDescriptor::parse(&bytes).unwrap();
        descriptor.valid_tokens = 2;
        descriptor.chunk_size = 2;
        descriptor.physical_buffer_rows = 10;
        descriptor.spill_write_values_per_cycle = 1;
        let vram = Arc::new(VectorSram::new(8, 128, DataType::Fp(FpType::BF16), 4));
        write_blocked_records(&vram, &[&projected, &projected]).await;

        let staged = StagedLayout::from_vram(descriptor, &vram).await.unwrap();
        let stats = staged.write_stats.unwrap();
        assert_eq!(stats.write_packets, 2);
        assert_eq!(stats.fifo_spill_values, 18);
        assert_eq!(stats.fifo_backpressure_cycles, 16);
    }

    #[tokio::test]
    async fn compiler_layout_physically_roundtrips_every_kda_value() {
        let (state, bytes, projected) = golden();
        let descriptor = LayoutDescriptor::parse(&bytes).unwrap();
        let vram = Arc::new(VectorSram::new(8, 128, DataType::Fp(FpType::BF16), 4));
        write_blocked_record(&vram, &projected).await;
        let mut store = LayoutStore::default();
        let write_stats = store.stage(descriptor, &vram).await.unwrap();
        assert_eq!(write_stats.values, projected.len() as u64);
        let (restored, stats) = store.consume_record(&state, 0, 0).unwrap();
        let quantized = projected
            .iter()
            .map(|&value| bf16::from_f32(value).to_f32())
            .collect::<Vec<_>>();
        assert_eq!(&restored[..projected.len()], quantized.as_slice());
        assert!(stats.write_packets > 0);
        assert!(stats.read_packets > 0);
        assert!(!store.contains(&state));
    }
}
