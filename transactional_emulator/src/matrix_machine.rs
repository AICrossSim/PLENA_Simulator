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

use std::sync::Arc;

use quantize::QuantTensor;
use sram::{MatrixSram, VectorSram, assert_multiple_of, multiple_and_offset};
use tch::{IndexOp, Tensor};

use crate::matrix_core::{MatrixCore, MatrixCoreProfile};
use crate::runtime_config::SYSTOLIC_PROCESSING_OVERHEAD;

/// Tensor allocation options used for every accumulator buffer (f32 on CPU).
const ACCUM_OPTS: (tch::Kind, tch::Device) = (tch::Kind::Float, tch::Device::Cpu);

/// Executes matrix opcodes by reading tiles/vectors from `mram`/`vram`,
/// running matmul in f32, and accumulating into per-shape buffers that are
/// later flushed via the `*_wo` ops.
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
    core: MatrixCore,
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
    ) -> Self {
        Self::new_with_core(
            mram,
            vram,
            mlen,
            hlen,
            blen,
            broadcast_amount,
            MatrixCoreProfile::big_default(),
        )
    }

    pub(crate) fn new_with_core(
        mram: Arc<MatrixSram>,
        vram: Arc<VectorSram>,
        mlen: u32,
        hlen: u32,
        blen: u32,
        broadcast_amount: u32,
        core_profile: MatrixCoreProfile,
    ) -> Self {
        Self {
            m_accum: Tensor::zeros([blen as i64, blen as i64], ACCUM_OPTS),
            hm_accum: Tensor::zeros(
                [broadcast_amount as i64, mlen as i64, mlen as i64],
                ACCUM_OPTS,
            ),
            hv_accum: Tensor::zeros([broadcast_amount as i64, mlen as i64], ACCUM_OPTS),
            v_accum: Tensor::zeros([blen as i64], ACCUM_OPTS),
            mram,
            vram,
            mlen,
            hlen,
            blen,
            broadcast_amount,
            core: MatrixCore::new(core_profile),
        }
    }

    pub(crate) fn core_profile(&self) -> MatrixCoreProfile {
        self.core.profile()
    }

    fn core(&self) -> MatrixCore {
        self.core
    }

    pub(crate) async fn mm(&mut self, m_addr: u32, v_addr: u32) {
        let (mat_base, mat_offset) = multiple_and_offset(m_addr, self.mlen * self.mlen);
        // Row-granular projection ABI: the M_MM column stride is `blen * mlen`
        // (compiler e852c88, isa_matrix.py vram_sub_projection*), so the within-tile
        // offset is `col_block * blen * mlen`. Divide out the mlen row width to
        // recover the column-block offset, mirroring `tmm` below. (Before e852c88 the
        // stride was `blen` and this decode was element-granular.)
        let mat_offset = assert_multiple_of(mat_offset, self.mlen);
        assert!(mat_offset.is_multiple_of(self.blen));
        assert!(mat_offset < self.mlen);

        let full_mat = self.mram.read(mat_base).await;
        // Slice columns instead of rows: [mlen, blen]
        let mat = full_mat
            .as_tensor()
            .view([self.mlen as i64, self.mlen as i64])
            .i((
                ..,
                mat_offset as i64..(mat_offset as i64 + self.blen as i64),
            ));
        let mut tensors = Vec::with_capacity(self.blen as usize);
        self.core()
            .compute(*SYSTOLIC_PROCESSING_OVERHEAD + self.mlen)
            .await;
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
        let vec = Tensor::stack(&tensors, 0);
        // Convert to float32 before matmul to match PyTorch golden reference
        let vec_f32 = vec.to_kind(tch::Kind::Float);
        let mat_f32 = mat.to_kind(tch::Kind::Float);
        // Now vec @ mat: [blen, mlen] @ [mlen, blen] = [blen, blen]
        self.m_accum += vec_f32.matmul(&mat_f32);
    }

    pub(crate) async fn bmm(&mut self, m_addr: u32, v_addr: u32, bmm_scale: f32) {
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

        let mut tensors = Vec::with_capacity(self.mlen as usize);
        self.core()
            .compute(*SYSTOLIC_PROCESSING_OVERHEAD + self.mlen)
            .await;
        for i in 0..self.mlen {
            tensors.push(
                self.vram
                    .read(v_addr + i * self.mlen)
                    .await
                    .as_tensor()
                    .shallow_clone(),
            );
        }
        // Stack along dimension 0 to get [mlen, hlen, broadcast_amount]
        let vec = Tensor::stack(&tensors, 0).view([
            self.mlen as i64,
            self.hlen as i64,
            self.broadcast_amount as i64,
        ]);

        // Now vec @ mat: [broadcast_amount, mlen, hlen] @ [hlen, mlen] = [broadcast_amount, mlen, mlen]
        let mut result_tensors = Vec::with_capacity(self.broadcast_amount as usize);
        for i in 0..self.broadcast_amount {
            // vec: [mlen, hlen, broadcast_amount]
            // For each i, select the corresponding slice along broadcast_amount
            let vec_i = vec.i((.., .., i as i64)).squeeze_dim(-1); // [mlen, hlen]
            // mat: [hlen, mlen]
            // Convert to float32 before matmul to match PyTorch golden reference
            let vec_i_f32 = vec_i.to_kind(tch::Kind::Float);
            let mat_f32 = mat.to_kind(tch::Kind::Float);
            let mut result = vec_i_f32.matmul(&mat_f32); // [mlen, mlen]
            result = &result * (bmm_scale as f64);
            result_tensors.push(result);
        }
        let result_tensor = Tensor::stack(&result_tensors, 0); // [broadcast_amount, mlen, mlen]

        self.hm_accum += result_tensor;
        tracing::trace!("hm_accum = {}", self.hm_accum);
    }

    pub(crate) async fn bmv(&mut self, m_addr: u32, v_addr: u32, bmm_scale: f32) {
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
        self.core().compute(*SYSTOLIC_PROCESSING_OVERHEAD + 1).await;
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
        let vec = Tensor::stack(&tensors, 0).view([
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
            let mut result = vec_i_f32.matmul(&mat_f32); // [1, mlen]
            result = &result * (bmm_scale as f64);
            result_tensors.push(result);
        }
        let result_tensor = Tensor::stack(&result_tensors, 0); // [broadcast_amount, 1, mlen]

        self.hv_accum += result_tensor;
        tracing::trace!("hv_accum = {}", self.hv_accum);
    }

    pub(crate) async fn btmm(&mut self, m_addr: u32, v_addr: u32, bmm_scale: f32) {
        assert!(self.broadcast_amount * self.hlen == self.mlen);
        // Load matrix from matrix SRAM.
        let (mat_base, mat_offset) = multiple_and_offset(m_addr, self.mlen * self.mlen);
        let (mat_offset, head_offset) = multiple_and_offset(mat_offset, self.mlen);

        assert!(mat_offset.is_multiple_of(self.blen));
        assert!(head_offset.is_multiple_of(self.hlen));
        let full_mat = self.mram.read(mat_base).await;

        // Slice columns instead of rows: [hlen, mlen]
        let mat = full_mat
            .as_tensor()
            .view([self.mlen as i64, self.mlen as i64])
            // .transpose(-1, -2)
            .i((
                mat_offset as i64..(mat_offset + self.mlen) as i64,
                head_offset as i64..(head_offset + self.hlen) as i64,
            ));

        let mut tensors = Vec::with_capacity(self.mlen as usize);
        self.core()
            .compute(*SYSTOLIC_PROCESSING_OVERHEAD + self.mlen)
            .await;
        // B, S, H, D
        for i in 0..self.mlen {
            tensors.push(
                self.vram
                    .read(v_addr + i * self.mlen)
                    .await
                    .as_tensor()
                    .shallow_clone(),
            );
        }
        // Stack along dimension 0 to get [mlen, hlen, broadcast_amount]
        let vec = Tensor::stack(&tensors, 0).view([
            self.mlen as i64,
            self.broadcast_amount as i64,
            self.hlen as i64,
        ]);

        tracing::trace!("btmm vec = {}", vec);
        tracing::trace!("btmm mat = {}", mat);
        tracing::debug!("broadcast_amount = {:?}", self.broadcast_amount);

        // Now vec @ mat: [broadcast_amount, mlen, hlen] @ [hlen, mlen] = [broadcast_amount, mlen, mlen]
        let mut result_tensors = Vec::with_capacity(self.broadcast_amount as usize);
        for i in 0..self.broadcast_amount {
            // vec: [mlen, hlen, broadcast_amount]
            // For each i, select the corresponding slice along broadcast_amount
            let vec_i = vec.i((.., i as i64, ..)).squeeze_dim(1); // [mlen, hlen]
            // mat: [hlen, mlen]
            tracing::trace!("vec_i = {}", vec_i);
            // Convert to float32 before matmul to match PyTorch golden reference
            let vec_i_f32 = vec_i.to_kind(tch::Kind::Float);
            let mat_t_f32 = mat.transpose(-1, -2).to_kind(tch::Kind::Float);
            let result = vec_i_f32.matmul(&mat_t_f32); // [mlen, mlen]
            let result = &result * (bmm_scale as f64);
            tracing::trace!("result = {}", result);
            result_tensors.push(result);
        }
        let result_tensor = Tensor::stack(&result_tensors, 0); // [broadcast_amount, mlen, mlen]

        self.hm_accum += result_tensor;
    }

    pub(crate) async fn btmv(&mut self, m_addr: u32, v_addr: u32, bmm_scale: f32) {
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
        self.core().compute(*SYSTOLIC_PROCESSING_OVERHEAD + 1).await;
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
        let vec = Tensor::stack(&tensors, 0).view([
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
            let result = &result * (bmm_scale as f64);
            tracing::trace!("result = {}", result);
            result_tensors.push(result);
        }
        let result_tensor = Tensor::stack(&result_tensors, 0).squeeze_dim(1); // [broadcast_amount, mlen]

        self.hv_accum += result_tensor;
    }

    pub(crate) async fn tmm(&mut self, v_addr: u32, m_addr: u32) {
        let (mat_base, mat_offset) = multiple_and_offset(m_addr, self.mlen * self.mlen);
        let mat_offset = assert_multiple_of(mat_offset, self.mlen);
        assert!(mat_offset.is_multiple_of(self.blen));
        let full_mat = self.mram.read(mat_base).await;
        // Transpose then slice columns: [mlen, blen]
        let mat = full_mat
            .as_tensor()
            .view([self.mlen as i64, self.mlen as i64])
            .transpose(-1, -2)
            .i((.., mat_offset as i64..(mat_offset + self.blen) as i64));
        let mut tensors = Vec::with_capacity(self.blen as usize);
        self.core()
            .compute(*SYSTOLIC_PROCESSING_OVERHEAD + self.mlen)
            .await;
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
        let vec = Tensor::stack(&tensors, 0);
        // Convert to float32 before matmul to match PyTorch golden reference
        let vec_f32 = vec.to_kind(tch::Kind::Float);
        let mat_f32 = mat.to_kind(tch::Kind::Float);
        // Now vec @ mat: [blen, mlen] @ [mlen, blen] = [blen, blen]
        self.m_accum += vec_f32.matmul(&mat_f32);
    }

    pub(crate) async fn mm_wo(&mut self, v_addr: u32, stride_len: u32) {
        let (vec_base, vec_offset) = multiple_and_offset(v_addr, self.mlen);
        assert!(vec_offset.is_multiple_of(self.blen));
        self.core().compute(1).await;
        for i in 0..self.blen {
            let tensor = self.m_accum.select_copy(0, i as i64).contiguous();
            let old = self.vram.read(vec_base + i * self.mlen * stride_len).await;
            let new = old.as_tensor().contiguous();
            let mut slot = new.i(vec_offset as i64..(vec_offset + self.blen) as i64);
            slot.copy_(&tensor);
            self.vram
                .write(
                    vec_base + i * self.mlen * stride_len,
                    QuantTensor::quantize(new, old.data_type()),
                )
                .await;
        }

        self.m_accum = Tensor::zeros([self.blen as i64, self.blen as i64], ACCUM_OPTS);
    }

    pub(crate) async fn bmm_wo(&mut self, v_addr: u32) {
        let (vec_base, vec_offset) = multiple_and_offset(v_addr, self.mlen);
        assert!(vec_offset.is_multiple_of(self.mlen));
        self.core().compute(1).await;
        for j in 0..self.broadcast_amount {
            for i in 0..self.mlen {
                let tensor = self
                    .hm_accum
                    .select_copy(0, j as i64)
                    .select_copy(0, i as i64)
                    .contiguous();
                self.vram
                    .write(
                        vec_base + (j * self.mlen + i) * self.mlen,
                        QuantTensor::quantize(tensor, self.vram.ty()),
                    )
                    .await;
            }
        }
        self.hm_accum = Tensor::zeros(
            [
                self.broadcast_amount as i64,
                self.mlen as i64,
                self.mlen as i64,
            ],
            ACCUM_OPTS,
        );
    }

    pub(crate) async fn bmv_wo(&mut self, v_addr: u32) {
        let (vec_base, vec_offset) = multiple_and_offset(v_addr, self.mlen);
        assert!(vec_offset.is_multiple_of(self.mlen));
        self.core().compute(1).await;
        for j in 0..self.broadcast_amount {
            let tensor = self.hv_accum.select_copy(0, j as i64).contiguous();
            self.vram
                .write(
                    vec_base + (j * self.mlen),
                    QuantTensor::quantize(tensor, self.vram.ty()),
                )
                .await;
        }
        self.hv_accum = Tensor::zeros([self.broadcast_amount as i64, self.mlen as i64], ACCUM_OPTS);
    }

    pub(crate) async fn mv(&mut self, m_addr: u32, v_addr: u32) {
        let (mat_base, mat_offset) = multiple_and_offset(m_addr, self.mlen * self.mlen);
        tracing::debug!("======================== MV ==========================");
        tracing::debug!("m_addr = {:?}", m_addr);
        tracing::debug!("mat_offset = {:?}", mat_offset);
        tracing::debug!("blen = {:?}", self.blen);
        assert!(mat_offset.is_multiple_of(self.blen));
        assert!(mat_offset < self.mlen);

        let mat = self.mram.read(mat_base).await;
        let vec = self.vram.read(v_addr).await;
        self.core().compute(self.mlen).await;
        // vec @ mat: [1, mlen] @ [mlen, mlen] = [1, mlen], then squeeze
        // Convert to float32 before matmul to match PyTorch golden reference
        let vec_f32 = vec.as_tensor().unsqueeze(0).to_kind(tch::Kind::Float);
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

    pub(crate) async fn tmv(&mut self, m_addr: u32, v_addr: u32) {
        // TODO: `_mat_base` is computed for the assertion below but the read
        // uses `m_addr` directly. For tile-aligned reads they're equivalent
        // (integer division), but other matrix ops here use `mat_base`. Worth
        // investigating whether this should be `mram.read(mat_base)`.
        let (_mat_base, mat_offset) = multiple_and_offset(m_addr, self.mlen * self.mlen);
        assert!(mat_offset.is_multiple_of(self.blen));
        let mat = self.mram.read(m_addr).await;
        let vec = self.vram.read(v_addr).await;
        self.core().compute(self.mlen).await;
        // vec @ transpose(mat): [1, mlen] @ [mlen, mlen] = [1, mlen], then squeeze
        // Convert to float32 before matmul to match PyTorch golden reference
        let vec_f32 = vec.as_tensor().unsqueeze(0).to_kind(tch::Kind::Float);
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
        self.core().compute(1).await;
        let old = self.vram.read(vec_base).await;
        let new = old.as_tensor().contiguous();
        let source = self.v_accum.contiguous();
        let mut slot = new.i(vec_offset as i64..(vec_offset + self.blen) as i64);
        slot.copy_(&source);
        self.vram
            .write(vec_base, QuantTensor::quantize(new, old.data_type()))
            .await;
        self.v_accum = Tensor::zeros([self.blen as i64], ACCUM_OPTS);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use quantize::{DataType, FpType, MxDataType, QuantTensor};
    use runtime::{Duration, Executor, Instant};
    use sram::{MatrixSram, VectorSram};
    use tch::Tensor;

    use super::*;
    use crate::matrix_core::MatrixCoreProfile;
    use crate::runtime_config::SYSTOLIC_PROCESSING_OVERHEAD;

    fn bf16_plain() -> MxDataType {
        MxDataType::Plain(DataType::Fp(FpType::BF16))
    }

    fn quant(vals: &[f32]) -> QuantTensor {
        QuantTensor::quantize(Tensor::from_slice(vals), bf16_plain())
    }

    async fn make_machine(output_base: u32) -> (MatrixMachine, Arc<VectorSram>, u32) {
        let mram = Arc::new(MatrixSram::new(4, 64, bf16_plain()));
        let vram = Arc::new(VectorSram::from_mx_type(4, 64, bf16_plain()));

        mram.write(
            0,
            quant(&[
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0,
            ]),
        )
        .await;
        vram.write(0, quant(&[1.0, 2.0, 3.0, 4.0])).await;
        vram.write(4, quant(&[5.0, 6.0, 7.0, 8.0])).await;

        (
            MatrixMachine::new_with_core(
                mram,
                vram.clone(),
                4,
                2,
                2,
                2,
                MatrixCoreProfile::big_default(),
            ),
            vram,
            output_base,
        )
    }

    #[tokio::test]
    async fn independent_matrix_machines_overlap_actual_gemm_work() {
        let executor = Executor::new();
        let observed = Arc::new(Mutex::new(Vec::new()));

        let (mut machine_a, vram_a, out_a) = make_machine(8).await;
        let (mut machine_b, vram_b, out_b) = make_machine(16).await;

        let observed_a = observed.clone();
        executor.spawn(async move {
            machine_a.mm(0, 0).await;
            machine_a.mm_wo(out_a, 1).await;
            observed_a.lock().unwrap().push(Executor::current().now());
        });

        let observed_b = observed.clone();
        executor.spawn(async move {
            machine_b.mm(0, 0).await;
            machine_b.mm_wo(out_b, 1).await;
            observed_b.lock().unwrap().push(Executor::current().now());
        });

        executor.enter(Instant::ETERNITY).await;

        let expected_cycles = *SYSTOLIC_PROCESSING_OVERHEAD + 4 + 1;
        let expected = Instant::INIT + Duration::from_nanos(expected_cycles as u64);
        assert_eq!(executor.now(), expected);
        assert_eq!(observed.lock().unwrap().as_slice(), &[expected, expected]);

        let a0 = vram_a.read(8).await.as_tensor().shallow_clone();
        let a1 = vram_a.read(12).await.as_tensor().shallow_clone();
        let b0 = vram_b.read(16).await.as_tensor().shallow_clone();
        let b1 = vram_b.read(20).await.as_tensor().shallow_clone();

        assert!(a0.equal(&Tensor::from_slice(&[1.0f32, 2.0, 0.0, 0.0])));
        assert!(a1.equal(&Tensor::from_slice(&[5.0f32, 6.0, 0.0, 0.0])));
        assert!(b0.equal(&Tensor::from_slice(&[1.0f32, 2.0, 0.0, 0.0])));
        assert!(b1.equal(&Tensor::from_slice(&[5.0f32, 6.0, 0.0, 0.0])));
    }

    #[tokio::test]
    async fn mm_row_granular_offset_selects_column_block() {
        // Row-granular M_MM ABI (compiler e852c88): the within-tile matrix offset is
        // `col_block * blen * mlen`. With mlen=4, blen=2, m_addr=8 selects column
        // block 1, i.e. matrix columns [2..4]. Build a 4x4 tile whose columns 2 and 3
        // are e1 and e2, so mm(8, _) must reproduce the column-block-0 identity result
        // ([1,2,0,0] / [5,6,0,0]). A pre-e852c88 element-granular decode would instead
        // read columns [8..10] (out of range) or trip `mat_offset < mlen`.
        let executor = Executor::new();

        let mram = Arc::new(MatrixSram::new(4, 64, bf16_plain()));
        let vram = Arc::new(VectorSram::from_mx_type(4, 64, bf16_plain()));
        // 4x4 row-major; column 2 = [1,0,0,0]^T, column 3 = [0,1,0,0]^T.
        mram.write(
            0,
            quant(&[
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0, //
                0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0,
            ]),
        )
        .await;
        vram.write(0, quant(&[1.0, 2.0, 3.0, 4.0])).await;
        vram.write(4, quant(&[5.0, 6.0, 7.0, 8.0])).await;

        let mut machine = MatrixMachine::new_with_core(
            mram,
            vram.clone(),
            4,
            2,
            2,
            2,
            MatrixCoreProfile::big_default(),
        );

        executor.spawn(async move {
            machine.mm(8, 0).await; // within-offset 8 -> column block 1 (columns [2..4])
            machine.mm_wo(8, 1).await;
        });
        executor.enter(Instant::ETERNITY).await;

        let a0 = vram.read(8).await.as_tensor().shallow_clone();
        let a1 = vram.read(12).await.as_tensor().shallow_clone();
        assert!(a0.equal(&Tensor::from_slice(&[1.0f32, 2.0, 0.0, 0.0])));
        assert!(a1.equal(&Tensor::from_slice(&[5.0f32, 6.0, 0.0, 0.0])));
    }
}
