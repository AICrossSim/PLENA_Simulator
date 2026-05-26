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

use crate::{SYSTOLIC_PROCESSING_OVERHEAD, cycle};

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
        }
    }

    pub(crate) async fn mm(&mut self, m_addr: u32, v_addr: u32) {
        let (mat_base, mat_offset) = multiple_and_offset(m_addr, self.mlen * self.mlen);
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
        cycle!(*SYSTOLIC_PROCESSING_OVERHEAD + self.mlen);
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
        cycle!(*SYSTOLIC_PROCESSING_OVERHEAD + self.mlen);
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
        tracing::debug!("hm_accum = {}", self.hm_accum);
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
        cycle!(*SYSTOLIC_PROCESSING_OVERHEAD + 1);
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
        tracing::debug!("hv_accum = {}", self.hv_accum);
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
        cycle!(*SYSTOLIC_PROCESSING_OVERHEAD + self.mlen);
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

        tracing::debug!("btmm vec = {}", vec);
        tracing::debug!("btmm mat = {}", mat);
        tracing::debug!("broadcast_amount = {:?}", self.broadcast_amount);

        // Now vec @ mat: [broadcast_amount, mlen, hlen] @ [hlen, mlen] = [broadcast_amount, mlen, mlen]
        let mut result_tensors = Vec::with_capacity(self.broadcast_amount as usize);
        for i in 0..self.broadcast_amount {
            // vec: [mlen, hlen, broadcast_amount]
            // For each i, select the corresponding slice along broadcast_amount
            let vec_i = vec.i((.., i as i64, ..)).squeeze_dim(1); // [mlen, hlen]
            // mat: [hlen, mlen]
            tracing::debug!("vec_i = {}", vec_i);
            // Convert to float32 before matmul to match PyTorch golden reference
            let vec_i_f32 = vec_i.to_kind(tch::Kind::Float);
            let mat_t_f32 = mat.transpose(-1, -2).to_kind(tch::Kind::Float);
            let result = vec_i_f32.matmul(&mat_t_f32); // [mlen, mlen]
            let result = &result * (bmm_scale as f64);
            tracing::debug!("result = {}", result);
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
        cycle!(*SYSTOLIC_PROCESSING_OVERHEAD + 1);
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

        tracing::debug!("btmv vec = {}", vec);
        tracing::debug!("btmv mat = {}", mat);
        tracing::debug!("broadcast_amount = {:?}", self.broadcast_amount);

        // Now vec @ mat: [broadcast_amount, 1, hlen] @ [hlen, mlen] = [broadcast_amount, 1, mlen]
        let mut result_tensors = Vec::with_capacity(self.broadcast_amount as usize);
        for i in 0..self.broadcast_amount {
            // vec: [1, broadcast_amount, hlen]
            // For each i, select the corresponding slice along broadcast_amount
            let vec_i = vec.i((.., i as i64, ..)).squeeze_dim(1); // [1, hlen]
            // mat: [mlen, hlen]
            tracing::debug!("vec_i = {}", vec_i);
            // Convert to float32 before matmul to match PyTorch golden reference
            let vec_i_f32 = vec_i.to_kind(tch::Kind::Float);
            let mat_t_f32 = mat.transpose(-1, -2).to_kind(tch::Kind::Float);
            let result = vec_i_f32.matmul(&mat_t_f32); // [1, mlen]
            let result = &result * (bmm_scale as f64);
            tracing::debug!("result = {}", result);
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
        cycle!(*SYSTOLIC_PROCESSING_OVERHEAD + self.mlen);
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
        cycle!(1);
        for i in 0..self.blen {
            let tensor = self.m_accum.i((i as i64, ..));
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

        self.m_accum = Tensor::zeros([self.blen as i64, self.blen as i64], ACCUM_OPTS);
    }

    pub(crate) async fn bmm_wo(&mut self, v_addr: u32) {
        let (vec_base, vec_offset) = multiple_and_offset(v_addr, self.mlen);
        assert!(vec_offset.is_multiple_of(self.mlen));
        cycle!(1);
        for j in 0..self.broadcast_amount {
            for i in 0..self.mlen {
                let tensor = self.hm_accum.i((j as i64, i as i64, ..));
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
        cycle!(1);
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
        cycle!(self.mlen);
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
        let (mat_base, mat_offset) = multiple_and_offset(m_addr, self.mlen * self.mlen);
        assert!(mat_offset.is_multiple_of(self.blen));
        let mat = self.mram.read(m_addr).await;
        let vec = self.vram.read(v_addr).await;
        cycle!(self.mlen);
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
        cycle!(1);
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
