//! `MatrixMachine` — executes PLENA matrix ISA opcodes against `MatrixSram`
//! and `VectorSram`.
//!
//! Covers the systolic-array operations:
//! - `M_MM` / `M_TMM` — matrix multiply (and its transposed variant)
//! - `M_BMM` / `M_BTMM` — broadcast matmul: applies the same matrix slice to
//!   `broadcast_amount` independent vector heads
//! - `M_MV` / `M_TMV` / `M_BMV` / `M_BTMV` — single-vector variants
//! - `M_*_WO` — write-out: flushes the corresponding accumulator into vram
//!
//! Each compute op accumulates into one of `m_accum` / `hm_accum` /
//! `hv_accum` / `v_accum`; the matching `*_WO` op flushes the accumulator
//! and resets it to zeros.
//!
//! # Timing model
//!
//! A plain matmul's cycle count is **BLEN-based, not MLEN-based**: the RTL
//! spreads the MLEN reduction spatially across MLEN/BLEN parallel sub-arrays,
//! so the per-instruction latency is set by the BLEN×BLEN tile feed/drain, and
//! the pipeline controller serializes matrix ops behind an active MCU (only the
//! writeout folds into the drain). A broadcast matmul retires the same
//! BLEN×BLEN tile, once for each of the MLEN/HLEN head lanes, so it costs the
//! same single issue. Costs here mirror the `alone` column of
//! `analytic_models/performance/customISA_lib.json` so the emulator and the
//! analytic model charge identical per-instruction cycles:
//!
//! - accumulate (`M_MM`/`M_TMM`):  3·BLEN + 11  (RTL-measured)
//! - broadcast accumulate (`M_BMM`/`M_BTMM`):  3·BLEN + 11. Each head lane is a
//!   separate (BLEN, HLEN)×(HLEN, BLEN) core running in parallel with the rest,
//!   so the issue latency does not scale with the head count.
//! - vector accumulate (`M_MV`/`M_TMV`):  BLEN + 9
//! - broadcast vector accumulate (`M_BMV`/`M_BTMV`):  (MLEN/BLEN)·(BLEN + 9)
//! - matrix writeout: `M_MM_WO` costs BLEN + 6; `M_BMM_WO` costs
//!   (MLEN/HLEN)·BLEN + 6, the `PH_DRAIN_ROWS` rows it commits
//! - vector writeout (`M_MV_WO`/`M_BMV_WO`):  MLEN + 6

use std::sync::Arc;

use quantize::{DataType, FpType, MxDataType, QuantTensor};
use sram::{MatrixSram, VectorSram, multiple_and_offset};
use tch::{IndexOp, Tensor};

use crate::cycle;

/// Serialized cycle cost of one matrix-matrix accumulate (`M_MM` family):
/// BLEN-row feed + BLEN-deep wavefront + fixed pipeline/fetch overhead.
/// RTL-microbenchmark-measured serialized cost: 23 cycles at BLEN=4 and
/// 35 at BLEN=8 pin the unique linear fit 3*BLEN + 11 (BLEN-row feed +
/// 2*BLEN drain + fixed overhead). Mirrors customISA_lib.json
/// `M_MM = BLEN*3 + 11`.
macro_rules! mm_cycles {
    ($self:ident) => {
        3 * $self.blen + 11
    };
}

/// Serialized cycle cost of one matrix-vector accumulate (`M_MV` family):
/// single-row feed + BLEN-deep wavefront + fixed pipeline overhead.
/// Mirrors customISA_lib.json `M_MV.alone = 6 + BLEN + 3`.
macro_rules! mv_cycles {
    ($self:ident) => {
        $self.blen + 9
    };
}

/// Serialized cycle cost of one broadcast matrix accumulate (`M_BMM`/`M_BTMM`).
///
/// The array is partitioned into MLEN/HLEN cores, each running a
/// (BLEN, HLEN) x (HLEN, BLEN) GEMM, and the cores run concurrently. One issue
/// therefore retires one BLEN x BLEN tile per head lane at the same latency as
/// a plain `M_MM`, and covering a whole MLEN x MLEN score tile takes
/// `(MLEN/BLEN)^2` issues emitted by the compiler rather than one wide issue.
macro_rules! bmm_cycles {
    ($self:ident) => {
        3 * $self.blen + 11
    };
}

/// Serialized cycle cost of one broadcast vector accumulate
/// (`M_BMV`/`M_BTMV`): a single activation row against every broadcast head
/// over an MLEN-wide result, i.e. MLEN/BLEN serialized matrix-vector issues.
macro_rules! bmv_cycles {
    ($self:ident) => {
        ($self.mlen / $self.blen) * ($self.blen + 9)
    };
}

/// Report the operand magnitudes when a matrix product overflows.
///
/// A non-finite product is either a genuinely unbounded input or an operand
/// read from the wrong place; the two are told apart by how large the operands
/// are, so both are printed with the addresses they came from.
fn debug_assert_operands_finite(
    result: &Tensor,
    activation: &Tensor,
    matrix: &Tensor,
    opcode: &str,
    m_addr: u32,
    v_addr: u32,
) {
    if bool::try_from(result.isfinite().all().logical_not()).unwrap_or(false) {
        let peak = |tensor: &Tensor| -> f64 {
            f64::try_from(tensor.abs().max()).unwrap_or(f64::NAN)
        };
        panic!(
            "{opcode} produced a non-finite result from finite-shaped operands: \
             activation at VRAM {v_addr} peaks at {:.3e} {:?}, matrix at MRAM \
             {m_addr} peaks at {:.3e} {:?}",
            peak(activation),
            activation.size(),
            peak(matrix),
            matrix.size(),
        );
    }
}

/// Cycles a matrix writeout costs.
///
/// The implemented control unit accepts a new instruction only when it is not
/// draining, so the writeout is serialized ahead of the next accumulate. With
/// `DRAIN_OVERLAPPED` the drain streams behind that accumulate instead and costs
/// one issue slot, which is what a double-banked accumulator buys.
fn writeout_cycles(rows: u32) -> u32 {
    if *crate::runtime_config::DRAIN_OVERLAPPED {
        1
    } else {
        rows + 6
    }
}

/// Tensor allocation options used for every accumulator buffer (f32 on CPU).
const ACCUM_OPTS: (tch::Kind, tch::Device) = (tch::Kind::Float, tch::Device::Cpu);
const FIXED_ACCUM_OPTS: (tch::Kind, tch::Device) =
    (tch::Kind::Int64, tch::Device::Cpu);
const FIXED_SCALE: f64 = 65_536.0;
const FIXED_MODULUS: f64 = 4_294_967_296.0;

fn round_matrix_activation(tensor: &Tensor, ty: MxDataType) -> Tensor {
    let shape = tensor.size();
    let flat = tensor
        .to_kind(tch::Kind::Float)
        .contiguous()
        .view([-1]);
    QuantTensor::quantize_materialized(flat, ty)
        .as_tensor()
        .view(shape.as_slice())
        .shallow_clone()
}

fn storage_fp_type(ty: MxDataType) -> FpType {
    let MxDataType::Plain(DataType::Fp(fp_type)) = ty else {
        panic!("matrix result storage must be a plain FP format");
    };
    fp_type
}

fn materialize_storage_fp(tensor: &Tensor, ty: MxDataType) -> Tensor {
    let shape = tensor.size();
    let flat = tensor
        .to_kind(tch::Kind::Float)
        .contiguous()
        .view([-1]);
    QuantTensor::quantize_materialized(flat, ty)
        .as_tensor()
        .view(shape.as_slice())
        .shallow_clone()
}

fn fp_to_fixed16_16(value: f32) -> i64 {
    assert!(value.is_finite(), "matrix partial must be finite");
    let scaled = (f64::from(value) * FIXED_SCALE).trunc();
    let bits = scaled.rem_euclid(FIXED_MODULUS) as u32;
    i64::from(bits as i32)
}

fn fixed16_16_partial(tensor: &Tensor, storage_type: MxDataType) -> Tensor {
    let shape = tensor.size();
    let rounded = materialize_storage_fp(tensor, storage_type);
    let values = Vec::<f32>::try_from(rounded.contiguous().view([-1]))
        .expect("matrix partial must convert to host FP32");
    let fixed: Vec<i64> = values.into_iter().map(fp_to_fixed16_16).collect();
    Tensor::from_slice(&fixed).view(shape.as_slice())
}

fn accumulate_fixed16_16(
    accumulator: &Tensor,
    partial: &Tensor,
    storage_type: MxDataType,
) -> Tensor {
    assert_eq!(accumulator.kind(), tch::Kind::Int64);
    assert_eq!(accumulator.size(), partial.size());
    let fixed_partial = fixed16_16_partial(partial, storage_type);
    let accumulated = Vec::<i64>::try_from(accumulator.contiguous().view([-1]))
        .expect("fixed accumulator must convert to host integers");
    let incoming = Vec::<i64>::try_from(fixed_partial.contiguous().view([-1]))
        .expect("fixed partial must convert to host integers");
    let wrapped: Vec<i64> = accumulated
        .into_iter()
        .zip(incoming)
        .map(|(left, right)| {
            i64::from((left as i32).wrapping_add(right as i32))
        })
        .collect();
    Tensor::from_slice(&wrapped).view(accumulator.size().as_slice())
}

fn truncate_fp_scalar(value: f32, fp_type: FpType) -> f32 {
    if value == 0.0 || !value.is_finite() {
        return value;
    }
    let magnitude = value.abs();
    let bias = if fp_type.exponent == 1 {
        1
    } else {
        (1_i32 << (fp_type.exponent - 1)) - 1
    };
    let min_exponent = 1 - bias;
    let max_exponent = (1_i32 << fp_type.exponent) - 2 - bias;
    let normal_floor = 2.0_f32.powi(min_exponent);
    let exponent = magnitude
        .log2()
        .floor()
        .clamp(min_exponent as f32, max_exponent as f32) as i32;
    let step = if magnitude < normal_floor {
        2.0_f32.powi(min_exponent - i32::from(fp_type.mantissa))
    } else {
        2.0_f32.powi(exponent - i32::from(fp_type.mantissa))
    };
    let maximum = (2.0 - 2.0_f32.powi(-i32::from(fp_type.mantissa)))
        * 2.0_f32.powi(max_exponent);
    let truncated = ((magnitude / step).floor() * step).min(maximum);
    truncated.copysign(value)
}

fn fixed16_16_to_storage_fp(
    accumulator: &Tensor,
    storage_type: MxDataType,
) -> Tensor {
    assert_eq!(accumulator.kind(), tch::Kind::Int64);
    let shape = accumulator.size();
    let fp_type = storage_fp_type(storage_type);
    let values = Vec::<i64>::try_from(accumulator.contiguous().view([-1]))
        .expect("fixed accumulator must convert to host integers");
    let output: Vec<f32> = values
        .into_iter()
        .map(|value| {
            truncate_fp_scalar(value as f32 / FIXED_SCALE as f32, fp_type)
        })
        .collect();
    Tensor::from_slice(&output).view(shape.as_slice())
}

/// Executes matrix opcodes by reading tiles/vectors from `mram`/`vram`,
/// running each materialized matrix product, and accumulating into per-shape
/// buffers that are later flushed via the `*_wo` ops. MXINT matrix products
/// cross the storage-FP and signed 16.16 boundaries after every instruction.
pub(crate) struct MatrixMachine {
    pub(crate) mram: Arc<MatrixSram>,
    vram: Arc<VectorSram>,
    m_accum: Tensor,
    hm_accum: Tensor,
    hv_accum: Tensor,
    v_accum: Tensor,
    mlen: u32,
    hlen: u32,
    blen: u32,
    broadcast_amount: u32,
    activation_type: MxDataType,
    mxint_fixed_bank: bool,
}

impl MatrixMachine {
    /// Create a `MatrixMachine` and initialize its four accumulators to zeros.
    ///
    /// `broadcast_amount` must equal `mlen / hlen` (the ratio of full-tile
    /// length to per-head length); this invariant is asserted at runtime in
    /// every broadcast op (`bmm` / `btmm` / `bmv` / `btmv`).
    pub(crate) fn new(
        mram: Arc<MatrixSram>,
        vram: Arc<VectorSram>,
        mlen: u32,
        hlen: u32,
        blen: u32,
        broadcast_amount: u32,
        activation_type: MxDataType,
        matrix_family: &str,
    ) -> Self {
        let mxint_fixed_bank = match matrix_family {
            "mxint" => true,
            "mxfp" => false,
            _ => panic!("unsupported matrix arithmetic family {matrix_family:?}"),
        };
        let matrix_accum_opts = if mxint_fixed_bank {
            FIXED_ACCUM_OPTS
        } else {
            ACCUM_OPTS
        };
        Self {
            m_accum: Tensor::zeros(
                [blen as i64, blen as i64],
                matrix_accum_opts,
            ),
            hm_accum: Tensor::zeros(
                [broadcast_amount as i64, blen as i64, blen as i64],
                matrix_accum_opts,
            ),
            hv_accum: Tensor::zeros([broadcast_amount as i64, mlen as i64], ACCUM_OPTS),
            v_accum: Tensor::zeros([blen as i64], ACCUM_OPTS),
            mram,
            vram,
            mlen,
            hlen,
            blen,
            broadcast_amount,
            activation_type,
            mxint_fixed_bank,
        }
    }

    /// Round a matrix operand's activation block to the array's input format.
    ///
    /// `v_addr` is the VRAM base the block was read from. A non-finite value
    /// here cannot be quantized, and the address is the only thing that
    /// identifies which tensor produced it, so it is reported rather than left
    /// to the serializer's shapeless assertion.
    fn round_activation_at(&self, tensor: &Tensor, v_addr: u32) -> Tensor {
        let flat = tensor.reshape([-1]).to_kind(tch::Kind::Float);
        let values = Vec::<f32>::try_from(&flat)
            .expect("matrix operand must convert to host floats");
        if let Some(first) = values.iter().position(|value| !value.is_finite()) {
            let columns = tensor.size().last().copied().unwrap_or(1) as usize;
            panic!(
                "matrix operand read from VRAM address {v_addr} is not finite: \
                 element {first} (row {}, column {}) of the {:?} block is {}",
                first / columns,
                first % columns,
                tensor.size(),
                values[first],
            );
        }
        self.round_activation(tensor)
    }

    fn round_activation(&self, tensor: &Tensor) -> Tensor {
        round_matrix_activation(tensor, self.activation_type)
    }

    fn accumulate_matrix(&self, accumulator: &Tensor, partial: &Tensor) -> Tensor {
        if self.mxint_fixed_bank {
            accumulate_fixed16_16(accumulator, partial, self.vram.ty())
        } else {
            accumulator + partial
        }
    }

    fn matrix_writeout(&self, accumulator: &Tensor) -> Tensor {
        if self.mxint_fixed_bank {
            fixed16_16_to_storage_fp(accumulator, self.vram.ty())
        } else {
            accumulator.shallow_clone()
        }
    }

    /// Multiply a BLEN-row activation block by a BLEN-column matrix tile.
    ///
    /// The matrix SRAM is addressed in element units but selects whole vectors:
    /// `matrix_sram_without_rounding.sv` shifts the read address right by
    /// `log2(MLEN * PARALLEL_DIM)`, and the sub-SRAM bank skew in `subsram.sv`
    /// turns the low bits of that shifted address into the column index of a
    /// column gather. The column-group index is therefore scaled by MLEN in the
    /// operand address, exactly as `tmm`'s row index is, and the compiler
    /// advances the operand by `BLEN * MLEN` per column group.
    pub(crate) async fn mm(&mut self, m_addr: u32, v_addr: u32) {
        let (mat_base, mat_offset) = multiple_and_offset(m_addr, self.mlen * self.mlen);
        assert!(
            mat_offset.is_multiple_of(self.mlen),
            "M_MM operand offset {mat_offset} (m_addr {m_addr}) is not a multiple \
             of MLEN {}: the column index is MLEN-scaled, so the emitter must \
             advance this operand by BLEN * MLEN per column group",
            self.mlen
        );
        let column = mat_offset / self.mlen;
        assert!(
            column.is_multiple_of(self.blen),
            "M_MM column {column} (m_addr {m_addr}) is not a multiple of BLEN {}",
            self.blen
        );
        assert!(
            column + self.blen <= self.mlen,
            "M_MM column group [{column}, {}) (m_addr {m_addr}, base {mat_base}) \
             exceeds the {} columns of the matrix tile",
            column + self.blen,
            self.mlen
        );

        let full_mat = self.mram.read(mat_base).await;
        // Take the addressed BLEN-column group: [mlen, blen]
        let mat = full_mat
            .as_tensor()
            .view([self.mlen as i64, self.mlen as i64])
            .i((
                ..,
                column as i64..(column as i64 + self.blen as i64),
            ));
        let mut tensors = Vec::with_capacity(self.blen as usize);
        cycle!(mm_cycles!(self));
        for i in 0..self.blen {
            tensors.push(
                self.vram
                    .read(v_addr + i * self.mlen)
                    .await
                    .as_tensor()
                    .shallow_clone(),
            );
        }
        // Stack along dimension 0 to get [blen, mlen]
        let vec = self.round_activation_at(&Tensor::stack(&tensors, 0), v_addr);
        // Convert to float32 before matmul to match PyTorch golden reference
        let vec_f32 = vec.to_kind(tch::Kind::Float);
        let mat_f32 = mat.to_kind(tch::Kind::Float);
        // Now vec @ mat: [blen, mlen] @ [mlen, blen] = [blen, blen]
        let partial = vec_f32.matmul(&mat_f32);
        self.m_accum = self.accumulate_matrix(&self.m_accum, &partial);
    }

    /// Broadcast matmul, the non-transposed counterpart of [`Self::btmm`]: one
    /// `(BLEN, HLEN) x (HLEN, BLEN)` GEMM per head lane.
    ///
    /// The operand's sub-MLEN bits select the HLEN row window of the matrix
    /// tile (the head), and the MLEN-aligned part divided by MLEN is the BLEN
    /// column group, as in [`Self::mm`]. The activation supplies BLEN rows read
    /// as `broadcast_amount` HLEN-wide head lanes.
    pub(crate) async fn bmm(
        &mut self,
        m_addr: u32,
        v_addr: u32,
        head_selector: u32,
    ) {
        assert!(self.broadcast_amount * self.hlen == self.mlen);
        assert!(head_selector < self.broadcast_amount);
        let m_addr = m_addr
            .checked_add(head_selector * self.hlen)
            .expect("packed matrix head address overflow");
        // Load matrix from matrix SRAM.
        let (mat_base, mat_offset) = multiple_and_offset(m_addr, self.mlen * self.mlen);
        let (column_offset, head_offset) = multiple_and_offset(mat_offset, self.mlen);
        assert!(head_offset.is_multiple_of(self.hlen));

        let column = column_offset / self.mlen;
        assert!(
            column.is_multiple_of(self.blen),
            "M_BMM column {column} (m_addr {m_addr}) is not a multiple of BLEN {}",
            self.blen
        );
        assert!(
            column + self.blen <= self.mlen,
            "M_BMM column group [{column}, {}) (m_addr {m_addr}) exceeds the {} \
             columns of the matrix tile",
            column + self.blen,
            self.mlen
        );

        let full_mat = self.mram.read(mat_base).await;
        // This head's HLEN rows, restricted to the addressed BLEN columns.
        let mat = full_mat
            .as_tensor()
            .view([self.mlen as i64, self.mlen as i64])
            .i((
                head_offset as i64..(head_offset + self.hlen) as i64,
                column as i64..(column + self.blen) as i64,
            ));

        let mut tensors = Vec::with_capacity(self.blen as usize);
        cycle!(bmm_cycles!(self));
        for i in 0..self.blen {
            tensors.push(
                self.vram
                    .read(v_addr + i * self.mlen)
                    .await
                    .as_tensor()
                    .shallow_clone(),
            );
        }
        // [blen, mlen] read as [blen, broadcast_amount, hlen] head lanes.
        let vec = self.round_activation_at(&Tensor::stack(&tensors, 0), v_addr).view([
            self.blen as i64,
            self.broadcast_amount as i64,
            self.hlen as i64,
        ]);

        let mat_f32 = mat.to_kind(tch::Kind::Float); // [hlen, blen]
        let mut result_tensors = Vec::with_capacity(self.broadcast_amount as usize);
        for i in 0..self.broadcast_amount {
            let vec_i = vec.i((.., i as i64, ..)).squeeze_dim(1); // [blen, hlen]
            let vec_i_f32 = vec_i.to_kind(tch::Kind::Float);
            result_tensors.push(vec_i_f32.matmul(&mat_f32)); // [blen, blen]
        }
        let result_tensor = Tensor::stack(&result_tensors, 0); // [broadcast, blen, blen]

        self.hm_accum = self.accumulate_matrix(
            &self.hm_accum,
            &result_tensor,
        );
        tracing::trace!("hm_accum = {}", self.hm_accum);
    }

    pub(crate) async fn bmv(&mut self, m_addr: u32, v_addr: u32) {
        assert!(self.broadcast_amount * self.hlen == self.mlen);
        // Load matrix from matrix SRAM.
        let (mat_base, mat_offset) = multiple_and_offset(m_addr, self.mlen * self.blen);
        let (mat_offset, head_offset) = multiple_and_offset(mat_offset, self.mlen);

        assert!(mat_offset.is_multiple_of(self.blen));
        assert!(head_offset.is_multiple_of(self.hlen));
        let full_mat = self.mram.read(mat_base).await;

        // Slice columns instead of rows: [hlen, mlen]
        let mat = full_mat
            .as_tensor()
            .view([self.mlen as i64, self.mlen as i64])
            .i((
                head_offset as i64..(head_offset + self.hlen) as i64,
                mat_offset as i64..(mat_offset + self.mlen) as i64,
            ));

        // For bmv, only read 1 vector (not mlen like bmm)
        let mut tensors = Vec::with_capacity(1);
        cycle!(bmv_cycles!(self));
        for i in 0..1 {
            tensors.push(
                self.vram
                    .read(v_addr + i * self.mlen)
                    .await
                    .as_tensor()
                    .shallow_clone(),
            );
        }
        // Stack along dimension 0 to get [1, hlen, broadcast_amount]
        let vec = self.round_activation_at(&Tensor::stack(&tensors, 0), v_addr).view([
            1_i64,
            self.hlen as i64,
            self.broadcast_amount as i64,
        ]);

        // Now vec @ mat: [broadcast_amount, 1, hlen] @ [hlen, mlen] = [broadcast_amount, 1, mlen]
        let mut result_tensors = Vec::with_capacity(self.broadcast_amount as usize);
        for i in 0..self.broadcast_amount {
            // vec: [1, hlen, broadcast_amount]
            // For each i, select the corresponding slice along broadcast_amount
            let vec_i = vec.i((.., .., i as i64)).squeeze_dim(-1); // [1, hlen]
            // mat: [hlen, mlen]
            // Convert to float32 before matmul to match PyTorch golden reference
            let vec_i_f32 = vec_i.to_kind(tch::Kind::Float);
            let mat_f32 = mat.to_kind(tch::Kind::Float);
            let result = vec_i_f32.matmul(&mat_f32); // [1, mlen]
            result_tensors.push(result);
        }
        // [broadcast_amount, 1, mlen] -> [broadcast_amount, mlen], the shape of
        // the per-head vector accumulator.
        let result_tensor = Tensor::stack(&result_tensors, 0).squeeze_dim(1);

        self.hv_accum += result_tensor;
        tracing::trace!("hv_accum = {}", self.hv_accum);
    }

    /// Broadcast transposed matmul: one `(BLEN, HLEN) x (HLEN, BLEN)` GEMM per
    /// head lane, all `MLEN/HLEN` lanes concurrently.
    ///
    /// The operand address below `MLEN^2` splits at MLEN exactly as `M_TMM`'s
    /// does: the MLEN-aligned part divided by MLEN is the BLEN-wide row block of
    /// the matrix tile, and the remainder is the HLEN-wide head-column window,
    /// to which `head_selector * HLEN` is added. The activation operand supplies
    /// BLEN rows, each read as `broadcast_amount` HLEN-wide head lanes, so the
    /// result is `[broadcast_amount, BLEN, BLEN]` — the tile
    /// `mxint_systolic_mcu.sv` drains as `PH_DRAIN_ROWS = HEAD_COUNT * BLEN`.
    pub(crate) async fn btmm(
        &mut self,
        m_addr: u32,
        v_addr: u32,
        head_selector: u32,
    ) {
        assert!(self.broadcast_amount * self.hlen == self.mlen);
        assert!(head_selector < self.broadcast_amount);
        let m_addr = m_addr
            .checked_add(head_selector * self.hlen)
            .expect("packed matrix head address overflow");
        // Load matrix from matrix SRAM.
        let (mat_base, mat_offset) = multiple_and_offset(m_addr, self.mlen * self.mlen);
        let (row_offset, head_offset) = multiple_and_offset(mat_offset, self.mlen);
        assert!(head_offset.is_multiple_of(self.hlen));

        let row_block = row_offset / self.mlen;
        assert!(
            row_block.is_multiple_of(self.blen),
            "M_BTMM row block {row_block} (m_addr {m_addr}) is not a multiple of \
             BLEN {}: the row index is MLEN-scaled, so the emitter must advance \
             this operand by BLEN * MLEN per row block",
            self.blen
        );
        assert!(
            row_block + self.blen <= self.mlen,
            "M_BTMM row block [{row_block}, {}) (m_addr {m_addr}) exceeds the {} \
             rows of the matrix tile",
            row_block + self.blen,
            self.mlen
        );

        let full_mat = self.mram.read(mat_base).await;
        // The addressed BLEN rows, restricted to this head's HLEN columns.
        let mat = full_mat
            .as_tensor()
            .view([self.mlen as i64, self.mlen as i64])
            .i((
                row_block as i64..(row_block + self.blen) as i64,
                head_offset as i64..(head_offset + self.hlen) as i64,
            ));

        let mut tensors = Vec::with_capacity(self.blen as usize);
        cycle!(bmm_cycles!(self));
        for i in 0..self.blen {
            tensors.push(
                self.vram
                    .read(v_addr + i * self.mlen)
                    .await
                    .as_tensor()
                    .shallow_clone(),
            );
        }
        // [blen, mlen] read as [blen, broadcast_amount, hlen] head lanes.
        let vec = self.round_activation_at(&Tensor::stack(&tensors, 0), v_addr).view([
            self.blen as i64,
            self.broadcast_amount as i64,
            self.hlen as i64,
        ]);

        tracing::trace!("btmm vec = {}", vec);
        tracing::trace!("btmm mat = {}", mat);

        let mat_t_f32 = mat.transpose(-1, -2).to_kind(tch::Kind::Float); // [hlen, blen]
        let mut result_tensors = Vec::with_capacity(self.broadcast_amount as usize);
        for i in 0..self.broadcast_amount {
            let vec_i = vec.i((.., i as i64, ..)).squeeze_dim(1); // [blen, hlen]
            let vec_i_f32 = vec_i.to_kind(tch::Kind::Float);
            result_tensors.push(vec_i_f32.matmul(&mat_t_f32)); // [blen, blen]
        }
        let result_tensor = Tensor::stack(&result_tensors, 0); // [broadcast, blen, blen]
        debug_assert_operands_finite(
            &result_tensor,
            &vec,
            &mat,
            "M_BTMM",
            m_addr,
            v_addr,
        );

        self.hm_accum = self.accumulate_matrix(
            &self.hm_accum,
            &result_tensor,
        );
    }

    pub(crate) async fn btmv(&mut self, m_addr: u32, v_addr: u32) {
        assert!(self.broadcast_amount * self.hlen == self.mlen);
        // Load matrix from matrix SRAM.
        let (mat_base, mat_offset) = multiple_and_offset(m_addr, self.mlen * self.mlen);
        let (mat_offset, head_offset) = multiple_and_offset(mat_offset, self.mlen);

        assert!(mat_offset.is_multiple_of(self.blen));
        assert!(head_offset.is_multiple_of(self.hlen));
        let full_mat = self.mram.read(mat_base).await;

        // Slice columns instead of rows: [mlen, hlen]
        let mat = full_mat
            .as_tensor()
            .view([self.mlen as i64, self.mlen as i64])
            // .transpose(-1, -2)
            .i((
                mat_offset as i64..(mat_offset + self.mlen) as i64,
                head_offset as i64..(head_offset + self.hlen) as i64,
            ));

        // For btmv, only read 1 vector (not mlen like btmm)
        let mut tensors = Vec::with_capacity(1);
        cycle!(bmv_cycles!(self));
        // B, S, H, D - only 1 query token for decode
        for i in 0..1 {
            tensors.push(
                self.vram
                    .read(v_addr + i * self.mlen)
                    .await
                    .as_tensor()
                    .shallow_clone(),
            );
        }
        // Stack along dimension 0 to get [1, broadcast_amount, hlen]
        let vec = self.round_activation_at(&Tensor::stack(&tensors, 0), v_addr).view([
            1_i64,
            self.broadcast_amount as i64,
            self.hlen as i64,
        ]);

        tracing::trace!("btmv vec = {}", vec);
        tracing::trace!("btmv mat = {}", mat);
        tracing::debug!("broadcast_amount = {:?}", self.broadcast_amount);

        // Now vec @ mat: [broadcast_amount, 1, hlen] @ [hlen, mlen] = [broadcast_amount, 1, mlen]
        let mut result_tensors = Vec::with_capacity(self.broadcast_amount as usize);
        for i in 0..self.broadcast_amount {
            // vec: [1, broadcast_amount, hlen]
            // For each i, select the corresponding slice along broadcast_amount
            let vec_i = vec.i((.., i as i64, ..)).squeeze_dim(1); // [1, hlen]
            // mat: [mlen, hlen]
            tracing::trace!("vec_i = {}", vec_i);
            // Convert to float32 before matmul to match PyTorch golden reference
            let vec_i_f32 = vec_i.to_kind(tch::Kind::Float);
            let mat_t_f32 = mat.transpose(-1, -2).to_kind(tch::Kind::Float);
            let result = vec_i_f32.matmul(&mat_t_f32); // [1, mlen]
            tracing::trace!("result = {}", result);
            result_tensors.push(result);
        }
        let result_tensor = Tensor::stack(&result_tensors, 0).squeeze_dim(1); // [broadcast_amount, mlen]

        self.hv_accum += result_tensor;
    }

    pub(crate) async fn tmm(&mut self, v_addr: u32, m_addr: u32) {
        let (mat_base, mat_offset) = multiple_and_offset(m_addr, self.mlen * self.mlen);
        assert!(
            mat_offset.is_multiple_of(self.mlen),
            "M_TMM operand offset {mat_offset} (m_addr {m_addr}) is not a multiple \
             of MLEN {}: the row index is MLEN-scaled",
            self.mlen
        );
        let mat_offset = mat_offset / self.mlen;
        assert!(
            mat_offset.is_multiple_of(self.blen),
            "M_TMM row {mat_offset} (m_addr {m_addr}) is not a multiple of BLEN {}",
            self.blen
        );
        let full_mat = self.mram.read(mat_base).await;
        // Transpose then slice columns: [mlen, blen]
        let mat = full_mat
            .as_tensor()
            .view([self.mlen as i64, self.mlen as i64])
            .transpose(-1, -2)
            .i((.., mat_offset as i64..(mat_offset + self.blen) as i64));
        let mut tensors = Vec::with_capacity(self.blen as usize);
        cycle!(mm_cycles!(self));
        for i in 0..self.blen {
            tensors.push(
                self.vram
                    .read(v_addr + i * self.mlen)
                    .await
                    .as_tensor()
                    .shallow_clone(),
            );
        }
        // Stack along dimension 0 to get [blen, mlen]
        let vec = self.round_activation_at(&Tensor::stack(&tensors, 0), v_addr);
        // Convert to float32 before matmul to match PyTorch golden reference
        let vec_f32 = vec.to_kind(tch::Kind::Float);
        let mat_f32 = mat.to_kind(tch::Kind::Float);
        // Now vec @ mat: [blen, mlen] @ [mlen, blen] = [blen, blen]
        let partial = vec_f32.matmul(&mat_f32);
        self.m_accum = self.accumulate_matrix(&self.m_accum, &partial);
    }

    pub(crate) async fn mm_wo(&mut self, v_addr: u32, stride_len: u32) {
        let (vec_base, vec_offset) = multiple_and_offset(v_addr, self.mlen);
        assert!(vec_offset.is_multiple_of(self.blen));
        // Drains the BLEN result rows; `M_MM_WO.alone = 6 + BLEN`.
        cycle!(writeout_cycles(self.blen));
        let writeout = self.matrix_writeout(&self.m_accum);
        for i in 0..self.blen {
            let tensor = writeout.i((i as i64, ..));
            let old = self.vram.read(vec_base + i * self.mlen * stride_len).await;
            let new = old.as_tensor().copy();
            new.i(vec_offset as i64..(vec_offset + self.blen) as i64)
                .copy_(&tensor);
            self.vram
                .write(
                    vec_base + i * self.mlen * stride_len,
                    QuantTensor::quantize(new, old.data_type()),
                )
                .await;
        }

        let matrix_accum_opts = if self.mxint_fixed_bank {
            FIXED_ACCUM_OPTS
        } else {
            ACCUM_OPTS
        };
        self.m_accum = Tensor::zeros(
            [self.blen as i64, self.blen as i64],
            matrix_accum_opts,
        );
    }

    /// Drain the broadcast accumulator: `PH_DRAIN_ROWS = HEAD_COUNT * BLEN`
    /// rows, BLEN of them for each head lane.
    ///
    /// The address decomposes as `M_MM_WO`'s does — the MLEN-aligned part is the
    /// destination row and the remainder is the BLEN-wide column group — and
    /// each head's tile sits one MLEN-row block further on, so head `j` row `i`
    /// lands at `base + (j * MLEN + i) * MLEN`.
    pub(crate) async fn bmm_wo(&mut self, v_addr: u32) {
        let (vec_base, vec_offset) = multiple_and_offset(v_addr, self.mlen);
        assert!(
            vec_offset.is_multiple_of(self.blen),
            "M_BMM_WO column {vec_offset} (v_addr {v_addr}) is not a multiple of \
             BLEN {}",
            self.blen
        );
        cycle!(writeout_cycles(self.broadcast_amount * self.blen));
        let writeout = self.matrix_writeout(&self.hm_accum);
        for j in 0..self.broadcast_amount {
            for i in 0..self.blen {
                let tensor = writeout.i((j as i64, i as i64, ..));
                let address = vec_base + (j * self.mlen + i) * self.mlen;
                let old = self.vram.read(address).await;
                let new = old.as_tensor().copy();
                new.i(vec_offset as i64..(vec_offset + self.blen) as i64)
                    .copy_(&tensor);
                self.vram
                    .write(address, QuantTensor::quantize(new, old.data_type()))
                    .await;
            }
        }
        let matrix_accum_opts = if self.mxint_fixed_bank {
            FIXED_ACCUM_OPTS
        } else {
            ACCUM_OPTS
        };
        self.hm_accum = Tensor::zeros(
            [
                self.broadcast_amount as i64,
                self.blen as i64,
                self.blen as i64,
            ],
            matrix_accum_opts,
        );
    }

    pub(crate) async fn bmv_wo(&mut self, v_addr: u32) {
        let (vec_base, vec_offset) = multiple_and_offset(v_addr, self.mlen);
        assert!(vec_offset.is_multiple_of(self.mlen));
        // Streams one MLEN-wide row; `M_BMV_WO.alone = 6 + MLEN`.
        cycle!(writeout_cycles(self.mlen));
        for j in 0..self.broadcast_amount {
            let tensor = self.hv_accum.i((j as i64, ..));
            self.vram
                .write(
                    vec_base + (j * self.mlen),
                    QuantTensor::quantize(tensor, self.vram.ty()),
                )
                .await;
        }
        self.hv_accum = Tensor::zeros([self.broadcast_amount as i64, self.mlen as i64], ACCUM_OPTS);
    }

    /// `M_MV` selects a column group the same way `mm` does: the operand offset
    /// carries the column index scaled by MLEN.
    pub(crate) async fn mv(&mut self, m_addr: u32, v_addr: u32) {
        let (mat_base, mat_offset) = multiple_and_offset(m_addr, self.mlen * self.mlen);
        assert!(
            mat_offset.is_multiple_of(self.mlen),
            "M_MV operand offset {mat_offset} (m_addr {m_addr}) is not a multiple \
             of MLEN {}: the column index is MLEN-scaled",
            self.mlen
        );
        let mat_offset = mat_offset / self.mlen;
        tracing::debug!("======================== MV ==========================");
        tracing::debug!("m_addr = {:?}", m_addr);
        tracing::debug!("column = {:?}", mat_offset);
        tracing::debug!("blen = {:?}", self.blen);
        assert!(mat_offset.is_multiple_of(self.blen));
        assert!(mat_offset + self.blen <= self.mlen);

        let mat = self.mram.read(mat_base).await;
        let vec = self.vram.read(v_addr).await;
        cycle!(mv_cycles!(self));
        // vec @ mat: [1, mlen] @ [mlen, mlen] = [1, mlen], then squeeze
        // Convert to float32 before matmul to match PyTorch golden reference
        let vec_f32 = self
            .round_activation_at(vec.as_tensor(), v_addr)
            .unsqueeze(0)
            .to_kind(tch::Kind::Float);
        let mat_t_f32 = mat
            .as_tensor()
            .view([self.mlen as i64, self.mlen as i64])
            .i((
                ..,
                mat_offset as i64..(mat_offset as i64 + self.blen as i64),
            ))
            .to_kind(tch::Kind::Float);
        let result = vec_f32.matmul(&mat_t_f32).squeeze_dim(0);
        self.v_accum += result;
    }

    /// `M_TMV` addresses rows exactly as `M_TMM` does: the tile offset carries
    /// the row index scaled by MLEN. `asm_templates/flashattn/qkt.py` advances
    /// this operand by `BLEN * MLEN` per row group.
    pub(crate) async fn tmv(&mut self, m_addr: u32, v_addr: u32) {
        let (mat_base, mat_offset) = multiple_and_offset(m_addr, self.mlen * self.mlen);
        assert!(
            mat_offset.is_multiple_of(self.mlen),
            "M_TMV operand offset {mat_offset} (m_addr {m_addr}) is not a \
             multiple of MLEN {}: the row index is MLEN-scaled",
            self.mlen
        );
        let mat_offset = mat_offset / self.mlen;
        assert!(
            mat_offset.is_multiple_of(self.blen),
            "M_TMV row {mat_offset} (m_addr {m_addr}) is not a multiple of BLEN {}",
            self.blen
        );
        assert!(
            mat_offset + self.blen <= self.mlen,
            "M_TMV row group [{mat_offset}, {}) (m_addr {m_addr}) exceeds the {} \
             rows of the matrix tile",
            mat_offset + self.blen,
            self.mlen
        );
        let mat = self.mram.read(mat_base).await;
        let vec = self.vram.read(v_addr).await;
        cycle!(mv_cycles!(self));
        // vec @ transpose(mat): [1, mlen] @ [mlen, mlen] = [1, mlen], then squeeze
        // Convert to float32 before matmul to match PyTorch golden reference
        let vec_f32 = self
            .round_activation_at(vec.as_tensor(), v_addr)
            .unsqueeze(0)
            .to_kind(tch::Kind::Float);
        let mat_t_f32 = mat
            .as_tensor()
            .view([self.mlen as i64, self.mlen as i64])
            .transpose(-1, -2)
            .i((
                ..,
                mat_offset as i64..(mat_offset as i64 + self.blen as i64),
            ))
            .to_kind(tch::Kind::Float);
        let result = vec_f32.matmul(&mat_t_f32).squeeze_dim(0);
        self.v_accum += result;
    }

    pub(crate) async fn mv_wo(&mut self, v_addr: u32) {
        let (vec_base, vec_offset) = multiple_and_offset(v_addr, self.mlen);
        assert!(vec_offset.is_multiple_of(self.blen));
        // Vector writeout streams one MLEN-wide row; customISA_lib `M_MV_WO.alone = 6 + MLEN`.
        cycle!(self.mlen + 6);
        let old = self.vram.read(vec_base).await;
        let new = old.as_tensor().copy();
        new.i(vec_offset as i64..(vec_offset + self.blen) as i64)
            .copy_(&self.v_accum);
        self.vram
            .write(vec_base, QuantTensor::quantize(new, old.data_type()))
            .await;
        self.v_accum = Tensor::zeros([self.blen as i64], ACCUM_OPTS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantize::{DataType, FpType, IntType};

    fn plain_fp(exponent: u8, mantissa: u8) -> MxDataType {
        MxDataType::Plain(DataType::Fp(FpType {
            sign: true,
            exponent,
            mantissa,
        }))
    }

    /// BF16 storage: the small integers used in the addressing tests are exact,
    /// so any mismatch is an addressing error rather than a rounding one.
    fn bf16_storage() -> MxDataType {
        MxDataType::Plain(DataType::Fp(FpType { sign: true, exponent: 8, mantissa: 7 }))
    }

    /// Build a machine over an MRAM holding one MLEN x MLEN matrix and a VRAM
    /// holding MLEN rows, both in exact FP32 so the check is on addressing only.
    fn build_machine(mlen: u32, blen: u32) -> (MatrixMachine, Arc<MatrixSram>, Arc<VectorSram>) {
        let ty = bf16_storage();
        let mram = Arc::new(MatrixSram::new(mlen, (mlen * mlen) as usize, ty));
        let vram = Arc::new(VectorSram::new(
            mlen,
            mlen as usize,
            DataType::Fp(FpType { sign: true, exponent: 8, mantissa: 7 }),
            4,
        ));
        let machine = MatrixMachine::new(
            Arc::clone(&mram),
            Arc::clone(&vram),
            mlen,
            mlen,
            blen,
            1,
            ty,
            "mxfp",
        );
        (machine, mram, vram)
    }

    /// `M_MM` must compute `A @ W[:, group]` for the BLEN-column group the
    /// operand address names, with the column index scaled by MLEN. The RTL
    /// shifts the matrix read address by `log2(MLEN * PARALLEL_DIM)`
    /// (`matrix_sram_without_rounding.sv`) and advances it by MLEN per column
    /// (`data_flow_control.sv`), so the compiler drives a `BLEN * MLEN` column
    /// stride.
    #[tokio::test]
    async fn mm_multiplies_the_addressed_column_group() {
        let (mlen, blen) = (4u32, 2u32);
        let ty = bf16_storage();
        for group in 0..(mlen / blen) {
            let (mut unit, mram, vram) = build_machine(mlen, blen);
            let weights: Vec<f32> = (0..(mlen * mlen)).map(|i| i as f32).collect();
            mram.write(
                0,
                QuantTensor::quantize_materialized(Tensor::from_slice(&weights), ty),
            )
            .await;
            for row in 0..blen {
                let mut activation = vec![0.0f32; mlen as usize];
                activation[row as usize] = 1.0;
                vram.write(
                    row * mlen,
                    QuantTensor::quantize_materialized(
                        Tensor::from_slice(&activation),
                        ty,
                    ),
                )
                .await;
            }
            // Matrix ops charge simulated cycles, so they need an executor.
            let column = group * blen;
            let operand_address = column * mlen; // MLEN-scaled: the RTL drops low bits
            let executor = runtime::Executor::new();
            let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let sink = std::sync::Arc::clone(&captured);
            executor.spawn(async move {
                unit.mm(operand_address, 0).await;
                *sink.lock().unwrap() = Vec::<f32>::try_from(
                    unit.m_accum.to_kind(tch::Kind::Float).contiguous().view([-1]),
                )
                .unwrap();
            });
            executor.enter(runtime::Instant::ETERNITY).await;
            let got = captured.lock().unwrap().clone();
            let mut want = Vec::new();
            for row in 0..blen {
                for offset in 0..blen {
                    want.push((row * mlen + column + offset) as f32);
                }
            }
            assert_eq!(got, want, "column group {group} (operand address {operand_address})");
        }
    }

    /// Run one matrix op inside an executor and return the flushed accumulator.
    async fn run_op<F>(unit: MatrixMachine, op: F) -> Vec<f32>
    where
        F: FnOnce(
                MatrixMachine,
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = MatrixMachine> + Send>>
            + Send
            + 'static,
    {
        let executor = runtime::Executor::new();
        let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = std::sync::Arc::clone(&captured);
        executor.spawn(async move {
            let unit = op(unit).await;
            *sink.lock().unwrap() = Vec::<f32>::try_from(
                unit.m_accum.to_kind(tch::Kind::Float).contiguous().view([-1]),
            )
            .unwrap();
        });
        executor.enter(runtime::Instant::ETERNITY).await;
        let out = captured.lock().unwrap().clone();
        out
    }

    /// Fill MRAM with W[r][c] = r * MLEN + c and VRAM with an identity block, so
    /// the accumulator directly reveals which slice of W an opcode addressed.
    async fn seed(mram: &MatrixSram, vram: &VectorSram, mlen: u32, rows: u32) {
        let ty = bf16_storage();
        let weights: Vec<f32> = (0..(mlen * mlen)).map(|i| i as f32).collect();
        mram.write(0, QuantTensor::quantize_materialized(
            Tensor::from_slice(&weights), ty,
        )).await;
        for row in 0..rows {
            let mut activation = vec![0.0f32; mlen as usize];
            activation[row as usize] = 1.0;
            vram.write(row * mlen, QuantTensor::quantize_materialized(
                Tensor::from_slice(&activation), ty,
            )).await;
        }
    }

    /// `M_TMM` addresses rows: the operand offset is divided by MLEN, so the
    /// address advances a BLEN-row group by `BLEN * MLEN`. This is the
    /// convention the RTL implements and that `isa_matrix` emits.
    #[tokio::test]
    async fn tmm_addresses_rows_with_an_mlen_scaled_offset() {
        let (mlen, blen) = (4u32, 2u32);
        for group in 0..(mlen / blen) {
            let (unit, mram, vram) = build_machine(mlen, blen);
            seed(&mram, &vram, mlen, blen).await;
            let row = group * blen;
            let operand_address = row * mlen; // MLEN-scaled: the RTL drops low bits
            let got = run_op(unit, move |mut u| {
                Box::pin(async move {
                    u.tmm(0, operand_address).await;
                    u
                })
            })
            .await;
            // W^T columns [row, row+BLEN) == W rows [row, row+BLEN), so the
            // identity activation picks W[row + k][j] for j in 0..BLEN.
            let mut want = Vec::new();
            for k in 0..blen {
                for j in 0..blen {
                    want.push(((row + j) * mlen + k) as f32);
                }
            }
            assert_eq!(got, want, "row group {group} (operand address {operand_address})");
        }
    }

    /// Build a broadcast-configured machine: HLEN-wide heads, `MLEN / HLEN`
    /// of them, so `M_BMM` / `M_BMV` / `M_BTMM` accumulate into `hm_accum`.
    fn build_broadcast_machine(
        mlen: u32,
        blen: u32,
        hlen: u32,
    ) -> (MatrixMachine, Arc<MatrixSram>, Arc<VectorSram>) {
        let ty = bf16_storage();
        let mram = Arc::new(MatrixSram::new(mlen, (mlen * mlen) as usize, ty));
        let vram = Arc::new(VectorSram::new(
            mlen,
            mlen as usize,
            DataType::Fp(FpType { sign: true, exponent: 8, mantissa: 7 }),
            4,
        ));
        let machine = MatrixMachine::new(
            Arc::clone(&mram),
            Arc::clone(&vram),
            mlen,
            hlen,
            blen,
            mlen / hlen,
            ty,
            "mxfp",
        );
        (machine, mram, vram)
    }

    /// Run one matrix op inside an executor and return the flushed per-head
    /// accumulator, which is where the broadcast opcodes deposit their result.
    async fn run_broadcast_op<F>(unit: MatrixMachine, op: F) -> Vec<f32>
    where
        F: FnOnce(
                MatrixMachine,
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = MatrixMachine> + Send>>
            + Send
            + 'static,
    {
        let executor = runtime::Executor::new();
        let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = std::sync::Arc::clone(&captured);
        executor.spawn(async move {
            let unit = op(unit).await;
            *sink.lock().unwrap() = Vec::<f32>::try_from(
                unit.hm_accum.to_kind(tch::Kind::Float).contiguous().view([-1]),
            )
            .unwrap();
        });
        executor.enter(runtime::Instant::ETERNITY).await;
        let out = captured.lock().unwrap().clone();
        out
    }

    /// `M_MV` addresses the same column group as `M_MM`, with the column index
    /// scaled by MLEN, and applies it to a single activation row.
    #[tokio::test]
    async fn mv_addresses_the_column_group_with_an_mlen_scaled_offset() {
        let (mlen, blen) = (4u32, 2u32);
        for group in 0..(mlen / blen) {
            let (unit, mram, vram) = build_machine(mlen, blen);
            seed(&mram, &vram, mlen, 1).await;
            let column = group * blen;
            let operand_address = column * mlen;
            let executor = runtime::Executor::new();
            let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let sink = std::sync::Arc::clone(&captured);
            let mut unit = unit;
            executor.spawn(async move {
                unit.mv(operand_address, 0).await;
                *sink.lock().unwrap() = Vec::<f32>::try_from(
                    unit.v_accum.to_kind(tch::Kind::Float).contiguous().view([-1]),
                )
                .unwrap();
            });
            executor.enter(runtime::Instant::ETERNITY).await;
            let got = captured.lock().unwrap().clone();
            // The activation selects row 0, so the result is W[0][column + j].
            let want: Vec<f32> = (0..blen).map(|j| (column + j) as f32).collect();
            assert_eq!(got, want, "column group {group} (operand address {operand_address})");
        }
    }

    /// `M_BMM` splits the operand offset at MLEN: the MLEN-aligned part is the
    /// row block and the remainder is a head offset that must be a multiple of
    /// HLEN. Selecting head `h` must apply `W[h*HLEN .. (h+1)*HLEN, :]`, so the
    /// sub-MLEN bits are a deliberate head offset and not an addressing error.
    #[tokio::test]
    async fn bmm_head_offset_selects_the_hlen_row_block() {
        let (mlen, blen, hlen) = (4u32, 2u32, 2u32);
        for head in 0..(mlen / hlen) {
            let (unit, mram, vram) = build_broadcast_machine(mlen, blen, hlen);
            seed(&mram, &vram, mlen, mlen).await;
            let operand_address = head * hlen;
            let got = run_broadcast_op(unit, move |mut u| {
                Box::pin(async move {
                    u.bmm(operand_address, 0, 0).await;
                    u
                })
            })
            .await;
            // The activation is an identity block, so lane `l` of broadcast
            // group `g` reads W[head*HLEN + (l*broadcast + g) selected row].
            // Only the addressed HLEN rows can contribute, so every non-zero
            // entry must lie inside [head*HLEN*MLEN, (head+1)*HLEN*MLEN).
            let lo = (head * hlen * mlen) as f32;
            let hi = ((head + 1) * hlen * mlen) as f32;
            let peak = got.iter().cloned().fold(f32::MIN, f32::max);
            assert!(
                peak >= lo && peak < hi,
                "head {head} read outside its HLEN row block \
                 (peak {peak}, expected [{lo}, {hi}))"
            );
        }
    }

    /// `M_BMV` decomposes its operand exactly like `M_BMM` but consumes a single
    /// activation row, so the same head offset must select the same row block.
    #[tokio::test]
    async fn bmv_head_offset_matches_bmm() {
        let (mlen, blen, hlen) = (4u32, 2u32, 2u32);
        for head in 0..(mlen / hlen) {
            let (unit, mram, vram) = build_broadcast_machine(mlen, blen, hlen);
            seed(&mram, &vram, mlen, 1).await;
            let operand_address = head * hlen;
            let executor = runtime::Executor::new();
            let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let sink = std::sync::Arc::clone(&captured);
            let mut unit = unit;
            executor.spawn(async move {
                unit.bmv(operand_address, 0).await;
                *sink.lock().unwrap() = Vec::<f32>::try_from(
                    unit.hv_accum.to_kind(tch::Kind::Float).contiguous().view([-1]),
                )
                .unwrap();
            });
            executor.enter(runtime::Instant::ETERNITY).await;
            let got = captured.lock().unwrap().clone();
            // Broadcast group 0 reads activation lane 0, which is 1.0 at index
            // 0, so it emits W[head*HLEN, :] exactly.
            let want: Vec<f32> =
                (0..mlen).map(|c| (head * hlen * mlen + c) as f32).collect();
            assert_eq!(
                got[..mlen as usize],
                want[..],
                "head {head} (operand address {operand_address})"
            );
        }
    }

    /// `M_TMV` addresses rows with the same MLEN-scaled offset as `M_TMM`,
    /// applied to a single activation row. `flashattn/qkt.py` drives it with a
    /// `BLEN * MLEN` row stride.
    #[tokio::test]
    async fn tmv_addresses_rows_with_an_mlen_scaled_offset() {
        let (mlen, blen) = (4u32, 2u32);
        for group in 0..(mlen / blen) {
            let (unit, mram, vram) = build_machine(mlen, blen);
            seed(&mram, &vram, mlen, 1).await;
            let row = group * blen;
            let operand_address = row * mlen;
            let executor = runtime::Executor::new();
            let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let sink = std::sync::Arc::clone(&captured);
            let mut unit = unit;
            executor.spawn(async move {
                unit.tmv(operand_address, 0).await;
                *sink.lock().unwrap() = Vec::<f32>::try_from(
                    unit.v_accum.to_kind(tch::Kind::Float).contiguous().view([-1]),
                )
                .unwrap();
            });
            executor.enter(runtime::Instant::ETERNITY).await;
            let got = captured.lock().unwrap().clone();
            // The activation selects column 0, so the result is W[row + j][0].
            let want: Vec<f32> = (0..blen).map(|j| ((row + j) * mlen) as f32).collect();
            assert_eq!(got, want, "row group {group} (operand address {operand_address})");
        }
    }

    /// One `M_BTMM` retires a BLEN x BLEN block for each of the `MLEN / HLEN`
    /// head lanes — the `PH_DRAIN_ROWS = HEAD_COUNT * BLEN` rows the RTL drains
    /// and the granularity the SimTop probe measured. Consuming MLEN activation
    /// rows and producing an MLEN x MLEN tile per head instead would let the
    /// compiler cover a score tile with one issue where hardware needs
    /// `(MLEN / BLEN)^2`.
    #[tokio::test]
    async fn btmm_retires_a_blen_by_blen_block_per_head() {
        let (mlen, blen, hlen) = (8u32, 2u32, 4u32);
        let broadcast = mlen / hlen;
        for row_block in (0..mlen).step_by(blen as usize) {
            let (unit, mram, vram) = build_broadcast_machine(mlen, blen, hlen);
            seed(&mram, &vram, mlen, blen).await;
            let operand_address = row_block * mlen;
            let got = run_broadcast_op(unit, move |mut u| {
                Box::pin(async move {
                    u.btmm(operand_address, 0, 0).await;
                    u
                })
            })
            .await;

            assert_eq!(
                got.len(),
                (broadcast * blen * blen) as usize,
                "row block {row_block}: the accumulator must hold one BLEN x BLEN \
                 block per head lane, not a whole tile"
            );
            // Activation rows 0..BLEN are the identity, so head lane 0 sees a
            // BLEN x BLEN identity in its HLEN columns and every later lane sees
            // zeros. result[0][i][j] is then W[row_block + j][i].
            let mut want = vec![0.0f32; got.len()];
            for i in 0..blen {
                for j in 0..blen {
                    want[(i * blen + j) as usize] = ((row_block + j) * mlen + i) as f32;
                }
            }
            assert_eq!(got, want, "row block {row_block} (operand {operand_address})");
        }
    }

    /// `M_BMM_WO` commits `HEAD_COUNT * BLEN` rows: BLEN rows for each head lane,
    /// one MLEN-row block apart, each carrying the BLEN-wide column group the
    /// low address bits select.
    #[tokio::test]
    async fn bmm_wo_commits_ph_drain_rows() {
        let (mlen, blen, hlen) = (8u32, 2u32, 4u32);
        let broadcast = mlen / hlen;
        let ty = bf16_storage();
        for column_block in (0..mlen).step_by(blen as usize) {
            let mram = Arc::new(MatrixSram::new(mlen, (mlen * mlen) as usize, ty));
            // One MLEN-row block for the activation, then one per head lane.
            let vram = Arc::new(VectorSram::new(
                mlen,
                ((broadcast + 1) * mlen) as usize,
                DataType::Fp(FpType { sign: true, exponent: 8, mantissa: 7 }),
                4,
            ));
            let unit = MatrixMachine::new(
                Arc::clone(&mram),
                Arc::clone(&vram),
                mlen,
                hlen,
                blen,
                broadcast,
                ty,
                "mxfp",
            );
            seed(&mram, &vram, mlen, blen).await;

            // Drain past the activation block so the seeded rows are not mistaken
            // for written ones.
            let destination_row = mlen;
            let executor = runtime::Executor::new();
            let sink = Arc::clone(&vram);
            let mut unit = unit;
            executor.spawn(async move {
                unit.btmm(0, 0, 0).await;
                unit.bmm_wo(destination_row * mlen + column_block).await;
            });
            executor.enter(runtime::Instant::ETERNITY).await;

            let mut written = Vec::new();
            for row in destination_row..((broadcast + 1) * mlen) {
                let values = Vec::<f32>::try_from(
                    sink.read(row * mlen)
                        .await
                        .as_tensor()
                        .to_kind(tch::Kind::Float)
                        .contiguous()
                        .view([-1]),
                )
                .unwrap();
                if values.iter().any(|&v| v != 0.0) {
                    written.push((row, values));
                }
            }

            // Head lane 0 alone sees a non-zero activation, so it is the only
            // lane whose BLEN rows carry data; the drain still walks all of them.
            let rows: Vec<u32> = written.iter().map(|(row, _)| *row).collect();
            assert_eq!(
                rows,
                (destination_row..destination_row + blen).collect::<Vec<_>>(),
                "column block {column_block}: expected BLEN rows of head lane 0"
            );
            for (row, values) in &written {
                let live: Vec<usize> = values
                    .iter()
                    .enumerate()
                    .filter(|(_, value)| **value != 0.0)
                    .map(|(index, _)| index)
                    .collect();
                assert!(
                    live.iter().all(|&index| {
                        index >= column_block as usize
                            && index < (column_block + blen) as usize
                    }),
                    "row {row} wrote outside the addressed BLEN column group \
                     [{column_block}, {}): {live:?}",
                    column_block + blen
                );
            }
        }
    }

    /// `M_BTMM` splits the operand offset at MLEN: the MLEN-aligned part selects
    /// the row block and the remainder is a head-column offset, which must be a
    /// multiple of HLEN. The `head_selector` immediate adds a further
    /// `selector * HLEN`, so sub-MLEN address bits are meaningful here by design
    /// and must not be treated as an addressing error.
    #[tokio::test]
    async fn btmm_head_selector_shifts_the_column_window_by_hlen() {
        let (mlen, blen, hlen) = (4u32, 2u32, 2u32);
        let ty = bf16_storage();
        for selector in 0..(mlen / hlen) {
            let mram = Arc::new(MatrixSram::new(mlen, (mlen * mlen) as usize, ty));
            let vram = Arc::new(VectorSram::new(
                mlen,
                mlen as usize,
                DataType::Fp(FpType { sign: true, exponent: 8, mantissa: 7 }),
                4,
            ));
            let unit = MatrixMachine::new(
                Arc::clone(&mram),
                Arc::clone(&vram),
                mlen,
                hlen,
                blen,
                mlen / hlen,
                ty,
                "mxfp",
            );
            seed(&mram, &vram, mlen, mlen).await;
            // Broadcast ops accumulate into the per-head bank, not `m_accum`.
            let executor = runtime::Executor::new();
            let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let sink = std::sync::Arc::clone(&captured);
            let mut unit = unit;
            executor.spawn(async move {
                unit.btmm(0, 0, selector).await;
                *sink.lock().unwrap() = Vec::<f32>::try_from(
                    unit.hm_accum.to_kind(tch::Kind::Float).contiguous().view([-1]),
                )
                .unwrap();
            });
            executor.enter(runtime::Instant::ETERNITY).await;
            let got = captured.lock().unwrap().clone();
            assert!(!got.is_empty(), "selector {selector} produced no result");
            // The selector must move the addressed window by exactly HLEN
            // columns, so a later selector must reach larger entries of W.
            let peak = got.iter().cloned().fold(f32::MIN, f32::max);
            assert!(
                peak >= (selector * hlen) as f32,
                "selector {selector} did not shift the column window (peak {peak})"
            );
        }
    }

    /// `M_BTMV` splits the operand offset at `MLEN * MLEN` to pick the matrix
    /// tile; the remainder below MLEN is a head-column offset and must be a
    /// multiple of HLEN. It is the single-vector form of `M_BTMM` and is what
    /// `flashattn/qkt.py` emits on the per-head decode path, so the column
    /// window it selects decides which head's QK^T a decode step computes.
    #[tokio::test]
    async fn btmv_head_offset_selects_the_hlen_column_window() {
        let (mlen, blen, hlen) = (4u32, 2u32, 2u32);
        for head in 0..(mlen / hlen) {
            let (unit, mram, vram) = build_broadcast_machine(mlen, blen, hlen);
            seed(&mram, &vram, mlen, 1).await;
            let operand_address = head * hlen;
            let executor = runtime::Executor::new();
            let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let sink = std::sync::Arc::clone(&captured);
            let mut unit = unit;
            executor.spawn(async move {
                unit.btmv(operand_address, 0).await;
                *sink.lock().unwrap() = Vec::<f32>::try_from(
                    unit.hv_accum.to_kind(tch::Kind::Float).contiguous().view([-1]),
                )
                .unwrap();
            });
            executor.enter(runtime::Instant::ETERNITY).await;
            let got = captured.lock().unwrap().clone();
            // The activation is read as [broadcast_amount, HLEN] lanes. Lane 0
            // holds the single 1.0, so broadcast group 0 emits the addressed
            // column of W and every later group contributes nothing.
            let want: Vec<f32> = (0..mlen)
                .map(|r| (r * mlen + head * hlen) as f32)
                .collect();
            assert_eq!(
                got[..mlen as usize],
                want[..],
                "head {head} (operand address {operand_address}) selected the wrong column window"
            );
            assert!(
                got[mlen as usize..].iter().all(|&v| v == 0.0),
                "head {head}: broadcast groups beyond lane 0 must not contribute, got {got:?}"
            );
        }
    }

    /// The `MLEN * MLEN` multiple of the operand address selects the matrix
    /// SRAM tile, which is what lets a key tile past the first be addressed.
    #[tokio::test]
    async fn btmv_reads_the_tile_the_mlen_squared_multiple_selects() {
        let (mlen, blen, hlen) = (4u32, 2u32, 2u32);
        let ty = bf16_storage();
        let mram = Arc::new(MatrixSram::new(mlen, (2 * mlen * mlen) as usize, ty));
        let vram = Arc::new(VectorSram::new(
            mlen,
            mlen as usize,
            DataType::Fp(FpType { sign: true, exponent: 8, mantissa: 7 }),
            4,
        ));
        let unit = MatrixMachine::new(
            Arc::clone(&mram),
            Arc::clone(&vram),
            mlen,
            hlen,
            blen,
            mlen / hlen,
            ty,
            "mxfp",
        );
        seed(&mram, &vram, mlen, 1).await;
        // Second tile is offset by a constant, so reading it is unambiguous.
        let second: Vec<f32> = (0..(mlen * mlen)).map(|i| (i + 100) as f32).collect();
        mram.write(
            mlen * mlen,
            QuantTensor::quantize_materialized(Tensor::from_slice(&second), ty),
        )
        .await;

        let executor = runtime::Executor::new();
        let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = std::sync::Arc::clone(&captured);
        let mut unit = unit;
        executor.spawn(async move {
            unit.btmv(mlen * mlen, 0).await;
            *sink.lock().unwrap() = Vec::<f32>::try_from(
                unit.hv_accum.to_kind(tch::Kind::Float).contiguous().view([-1]),
            )
            .unwrap();
        });
        executor.enter(runtime::Instant::ETERNITY).await;
        let got = captured.lock().unwrap().clone();
        let want: Vec<f32> = (0..mlen).map(|r| (r * mlen + 100) as f32).collect();
        assert_eq!(
            got[..mlen as usize],
            want[..],
            "the MLEN*MLEN multiple did not select the second matrix tile"
        );
    }

    #[test]
    fn matrix_activation_rounding_preserves_shape_and_uses_a_format() {
        let ty = MxDataType::Mx {
            elem: DataType::Int(IntType { width: 2 }),
            scale: DataType::Fp(FpType::E8M0),
            block: 8,
        };
        let input = Tensor::from_slice(&[
            -0.40f32, -0.30, -0.20, -0.10, 0.10, 0.20, 0.30, 0.40,
        ])
        .view([2, 4]);
        let rounded = round_matrix_activation(&input, ty);
        let values: Vec<f32> =
            Vec::<f32>::try_from(rounded.view([-1])).unwrap();
        assert_eq!(rounded.size(), vec![2, 4]);
        // MXINT2 carries one magnitude bit, so qmax = 1 and the shared scale is
        // 2^ceil(log2(0.40 / 1)) = 0.5. Rounding to nearest even then keeps the
        // block maximum representable instead of clipping it.
        assert_eq!(
            values,
            vec![-0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]
        );
    }

    #[test]
    fn matrix_partial_rounding_precedes_fixed_bank_accumulation() {
        let storage = plain_fp(3, 2);
        let mut accumulator = Tensor::zeros([1], FIXED_ACCUM_OPTS);
        accumulator = accumulate_fixed16_16(
            &accumulator,
            &Tensor::from_slice(&[0.5f32]),
            storage,
        );
        accumulator = accumulate_fixed16_16(
            &accumulator,
            &Tensor::from_slice(&[-0.54f32]),
            storage,
        );
        let output = fixed16_16_to_storage_fp(&accumulator, storage);
        let values = Vec::<f32>::try_from(output).unwrap();
        assert_eq!(values, vec![0.0]);

        let full_product = Tensor::from_slice(&[0.5f32 - 0.54]);
        let one_partial = accumulate_fixed16_16(
            &Tensor::zeros([1], FIXED_ACCUM_OPTS),
            &full_product,
            storage,
        );
        let one_partial = fixed16_16_to_storage_fp(&one_partial, storage);
        assert!(!one_partial.equal(&Tensor::from_slice(&values)));
    }

    #[test]
    fn fixed_bank_addition_wraps_as_signed_16_16() {
        let storage = plain_fp(8, 7);
        let accumulator = Tensor::from_slice(&[i64::from(i32::MAX)]);
        let wrapped = accumulate_fixed16_16(
            &accumulator,
            &Tensor::from_slice(&[1.0f32 / 65_536.0]),
            storage,
        );
        let values = Vec::<i64>::try_from(wrapped).unwrap();
        assert_eq!(values, vec![i64::from(i32::MIN)]);
    }

    #[test]
    fn fixed_bank_writeout_truncates_to_storage_fp() {
        let storage = plain_fp(3, 2);
        let fixed = Tensor::from_slice(&[fp_to_fixed16_16(1.219)]);
        let output = fixed16_16_to_storage_fp(&fixed, storage);
        let values = Vec::<f32>::try_from(output).unwrap();
        assert_eq!(values, vec![1.0]);
    }
}
