#![allow(unused_variables, unused_mut, dead_code)]

mod load_config;
mod op; // Add this line to include the config module

use std::future::Future;
use std::io::Write;
use std::mem::ManuallyDrop;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::LazyLock;

use clap::Parser;
use futures::StreamExt;
use futures::stream::FuturesUnordered;
use half::f16;
use memory::{ErasedMemoryModel, MemoryModel};
use quantize::{MxDataType, QuantTensor};
use runtime::{Duration, Executor, Instant};
use tch::{IndexOp, Tensor};
use vector_sram::VectorSram;

use tokio::sync::Mutex;
use tokio::sync::oneshot::{self, Receiver};

// Import the configuration functions
use load_config::*;

// Replace the const declarations with function calls to the config
// These functions will be called at runtime to get the configured values

const PERIOD: Duration = Duration::from_nanos(1);
static SYSTOLIC_PROCESSING_OVERHEAD: LazyLock<u32> =
    LazyLock::new(|| systolic_processing_overhead());
static VECTOR_ADD_CYCLES: LazyLock<u32> = LazyLock::new(|| vector_add_cycles());
static VECTOR_MUL_CYCLES: LazyLock<u32> = LazyLock::new(|| vector_mul_cycles());
static VECTOR_EXP_CYCLES: LazyLock<u32> = LazyLock::new(|| vector_exp_cycles());
static VECTOR_RECI_CYCLES: LazyLock<u32> = LazyLock::new(|| vector_reci_cycles());
static VECTOR_MAX_CYCLES: LazyLock<u32> = LazyLock::new(|| vector_max_cycles());
static VECTOR_SUM_CYCLES: LazyLock<u32> = LazyLock::new(|| vector_sum_cycles());
static SCALAR_FP_BASIC_CYCLES: LazyLock<u32> = LazyLock::new(|| scalar_fp_basic_cycles());
static SCALAR_FP_EXP_CYCLES: LazyLock<u32> = LazyLock::new(|| scalar_fp_exp_cycles());
static SCALAR_FP_SQRT_CYCLES: LazyLock<u32> = LazyLock::new(|| scalar_fp_sqrt_cycles());
static SCALAR_FP_RECI_CYCLES: LazyLock<u32> = LazyLock::new(|| scalar_fp_reci_cycles());
static SCALAR_INT_BASIC_CYCLES: LazyLock<u32> = LazyLock::new(|| scalar_int_basic_cycles());
static MAX_LOOP_INSTRUCTIONS: LazyLock<usize> = LazyLock::new(|| max_loop_instructions());

static MLEN: LazyLock<u32> = LazyLock::new(|| mlen());
static VLEN: LazyLock<u32> = LazyLock::new(|| vlen());
static BLEN: LazyLock<u32> = LazyLock::new(|| blen());
static HLEN: LazyLock<u32> = LazyLock::new(|| hlen());
static BROADCAST_AMOUNT: LazyLock<u32> = LazyLock::new(|| broadcast_amount());
static HBM_SIZE: LazyLock<usize> = LazyLock::new(|| hbm_size());
static MATRIX_SRAM_SIZE: LazyLock<usize> = LazyLock::new(|| matrix_sram_size());
static VECTOR_SRAM_SIZE: LazyLock<usize> = LazyLock::new(|| vector_sram_size());
static MATRIX_SRAM_TYPE: LazyLock<MxDataType> = LazyLock::new(|| matrix_sram_type());
static VECTOR_SRAM_TYPE: LazyLock<MxDataType> = LazyLock::new(|| vector_sram_type());
static MATRIX_WEIGHT_TYPE: LazyLock<MxDataType> = LazyLock::new(|| matrix_weight_type());
static MATRIX_KV_TYPE: LazyLock<MxDataType> = LazyLock::new(|| matrix_kv_type());
static VECTOR_ACTIVATION_TYPE: LazyLock<MxDataType> = LazyLock::new(|| vector_activation_type());
static VECTOR_KV_TYPE: LazyLock<MxDataType> = LazyLock::new(|| vector_kv_type());
static PREFETCH_M_AMOUNT: LazyLock<u32> = LazyLock::new(|| hbm_m_prefetch_amount());
static PREFETCH_V_AMOUNT: LazyLock<u32> = LazyLock::new(|| hbm_v_prefetch_amount());
static STORE_V_AMOUNT: LazyLock<u32> = LazyLock::new(|| hbm_v_writeback_amount());

/// Address handling utilities.
///
/// Many operations on matrix and vector SRAM operate on entire tiles so it needs to be multiple, but some aren't, so we use
/// element indexing. This utility provides some helper functions for address handling.
trait AddrUtils: Sized {
    fn assert_multiple_of(self, mul: Self) -> Self;

    fn multiple_and_offset(self, mul: Self) -> (Self, Self);
}

impl AddrUtils for u32 {
    fn assert_multiple_of(self, mul: u32) -> u32 {
        assert!(self.is_multiple_of(mul));
        self / mul
    }

    fn multiple_and_offset(self, mul: u32) -> (u32, u32) {
        let d = self / mul;
        let r = self % mul;
        (d * mul, r)
    }
}

macro_rules! cycle {
    ($cycle: expr) => {
        runtime::Executor::current()
            .resolve_at(PERIOD * ($cycle as u32))
            .await;
    };
}

/// Behaviour modelling of matrix SRAM.
///
/// The timing aspect is to be considered by the matrix machine itself.
struct MatrixSram {
    tile_size: u32,
    tiles: Vec<Mutex<Result<QuantTensor, Receiver<QuantTensor>>>>,
    ty: MxDataType,
}

impl MatrixSram {
    /// Creata a matrix SRAM with given tile size and depth.
    fn new(tile_size: u32, depth: usize, ty: MxDataType) -> Self {
        let tiles = (0..(depth / tile_size as usize))
            .map(|_| Mutex::new(Ok(QuantTensor::zeros((tile_size * tile_size) as usize, ty))))
            .collect();
        Self {
            tile_size,
            tiles,
            ty,
        }
    }

    fn size_in_bytes(&self) -> usize {
        (self.tile_size * self.tile_size) as usize * self.tiles.len()
    }

    async fn read(&self, addr: u32) -> QuantTensor {
        let addr_in_tiles = addr.assert_multiple_of(self.tile_size * self.tile_size);

        let mut guard = self.tiles[addr_in_tiles as usize].lock().await;
        if let Err(ref mut fut) = *guard {
            *guard = Ok(fut.await.unwrap());
        }

        guard.as_ref().map_err(|_| ()).unwrap().clone()
    }

    async fn write(&self, addr: u32, tensor: QuantTensor) {
        let addr_in_tiles = addr.assert_multiple_of(self.tile_size * self.tile_size);

        assert!(tensor.data_type() == self.ty);
        *self.tiles[addr_in_tiles as usize].lock().await = Ok(tensor);
    }

    async fn write_delayed(&self, addr: u32, tensor: Receiver<QuantTensor>) {
        let addr_in_tiles = addr.assert_multiple_of(self.tile_size);

        *self.tiles[addr_in_tiles as usize].lock().await = Err(tensor);
    }

    async fn continous_write_delayed(
        &self,
        addr: u32,
        write_amount: u32,
        tensor: Receiver<QuantTensor>,
    ) {
        let addr_in_tiles = addr.assert_multiple_of(self.tile_size * self.tile_size);
        // Await the tensor from the channel (blocks until data arrives)
        if let Ok(tensor) = tensor.await {
            let dims = tensor.as_tensor().size();
            let chunk_size = (self.tile_size * self.tile_size) as i64;
            let total = dims[0];

            // Split the tensor into chunks of self.tile_size and store each in self.tiles.
            for i in 0..write_amount.min(
                (total as u32 + self.tile_size * self.tile_size - 1)
                    / (self.tile_size * self.tile_size),
            ) {
                let start = (i as i64) * chunk_size;
                let end = ((i as i64 + 1) * chunk_size).min(total);
                let chunk = tensor
                    .as_tensor()
                    .narrow(0, start, end - start)
                    .shallow_clone();
                let chunk_qt = QuantTensor::quantize(chunk, self.ty);
                *self.tiles[(addr_in_tiles + i) as usize].lock().await = Ok(chunk_qt);
            }
        }
    }

    async fn as_bytes(&self) -> Vec<u8> {
        let element_ty = self.ty.element_type();
        let mut result = Vec::new();

        for tile_mutex in &self.tiles {
            let mut guard = tile_mutex.lock().await;
            if let Err(ref mut fut) = *guard {
                *guard = Ok(fut.await.unwrap());
            }
            let tensor = guard.as_ref().map_err(|_| ()).unwrap();
            let tensor_data = tensor.as_tensor();
            let len = tensor_data.size1().unwrap() as usize;
            let f32_slice =
                unsafe { core::slice::from_raw_parts(tensor_data.data_ptr() as *const f32, len) };
            // Calculate bytes needed for THIS tile's actual size
            let total_bits = len * element_ty.size_in_bits() as usize;
            let bytes_needed = (total_bits + 7) / 8;
            let mut tile_bytes = vec![0u8; bytes_needed];
            element_ty.bytes_from_f32(f32_slice, &mut tile_bytes);
            result.extend_from_slice(&tile_bytes);
        }

        result
    }
}

// VectorSram is now imported from the vector_sram library

struct MatrixMachine {
    mram: Arc<MatrixSram>,
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
    async fn mm(&mut self, m_addr: u32, v_addr: u32) {
        // println!("======================== M_MM ==========================");
        // println!("m_addr = {:?}", m_addr);
        // println!("v_addr = {:?}", v_addr);
        let (mat_base, mat_offset) = m_addr.multiple_and_offset(self.mlen * self.mlen);
        // println!("mat_base = {:?}", mat_base);
        // println!("mat_offset = {:?}", mat_offset);
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
        let vec = tch::Tensor::stack(&tensors, 0);
        // Convert to float32 before matmul to match PyTorch golden reference
        let vec_f32 = vec.to_kind(tch::Kind::Float);
        let mat_f32 = mat.to_kind(tch::Kind::Float);
        // println!("vec = {}", vec);
        // println!("mat = {}", mat);
        // Now vec @ mat: [blen, mlen] @ [mlen, blen] = [blen, blen]
        self.m_accum += vec_f32.matmul(&mat_f32);
    }

    async fn bmm(&mut self, m_addr: u32, v_addr: u32, bmm_scale: f32) {
        // println!("m_addr = {:?}", m_addr);
        // println!("v_addr = {:?}", v_addr);
        assert!(self.broadcast_amount * self.hlen == self.mlen);
        // Load matrix from matrix SRAM.
        let (mat_base, mat_offset) = m_addr.multiple_and_offset(self.mlen * self.blen);
        let (mat_offset, head_offset) = mat_offset.multiple_and_offset(self.mlen);

        // println!("mat_offset = {:?}", mat_offset);
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
                    .read(v_addr + i * self.mlen )
                    .await
                    .as_tensor()
                    .shallow_clone(),
            );
        }
        // Stack along dimension 0 to get [mlen, hlen, broadcast_amount]
        let vec = tch::Tensor::stack(&tensors, 0).view([
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
        let result_tensor = tch::Tensor::stack(&result_tensors, 0); // [broadcast_amount, mlen, mlen]

        self.hm_accum += result_tensor;
        if !is_quiet() {
            println!("hm_accum = {}", self.hm_accum);
        }
    }

    async fn bmv(&mut self, m_addr: u32, v_addr: u32, bmm_scale: f32) {
        // println!("m_addr = {:?}", m_addr);
        // println!("v_addr = {:?}", v_addr);
        assert!(self.broadcast_amount * self.hlen == self.mlen);
        // Load matrix from matrix SRAM.
        let (mat_base, mat_offset) = m_addr.multiple_and_offset(self.mlen * self.blen);
        let (mat_offset, head_offset) = mat_offset.multiple_and_offset(self.mlen);

        // println!("mat_offset = {:?}", mat_offset);
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
        let vec = tch::Tensor::stack(&tensors, 0).view([
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
        let result_tensor = tch::Tensor::stack(&result_tensors, 0); // [broadcast_amount, 1, mlen]

        self.hv_accum += result_tensor;
        if !is_quiet() {
            println!("hv_accum = {}", self.hv_accum);
        }
    }

    async fn btmm(&mut self, m_addr: u32, v_addr: u32, bmm_scale: f32) {
        // println!("======================== BTMM ==========================");
        // println!("m_addr = {:?}", m_addr);
        // println!("v_addr = {:?}", v_addr);
        // println!("bmm_scale = {:?}", bmm_scale);
        assert!(self.broadcast_amount * self.hlen == self.mlen);
        // Load matrix from matrix SRAM.
        let (mat_base, mat_offset) = m_addr.multiple_and_offset(self.mlen * self.mlen);
        let (mat_offset, head_offset) = mat_offset.multiple_and_offset(self.mlen);

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
                    .read(v_addr + i * self.mlen )
                    .await
                    .as_tensor()
                    .shallow_clone(),
            );
        }
        // Stack along dimension 0 to get [mlen, hlen, broadcast_amount]
        let vec = tch::Tensor::stack(&tensors, 0).view([
            self.mlen as i64,
            self.broadcast_amount as i64,
            self.hlen as i64,
        ]);

        if !is_quiet() {
            println!("btmm vec = {}", vec);
            println!("btmm mat = {}", mat);
            println!("broadcast_amount = {:?}", self.broadcast_amount);
        }

        // Now vec @ mat: [broadcast_amount, mlen, hlen] @ [hlen, mlen] = [broadcast_amount, mlen, mlen]
        let mut result_tensors = Vec::with_capacity(self.broadcast_amount as usize);
        for i in 0..self.broadcast_amount {
            // vec: [mlen, hlen, broadcast_amount]
            // For each i, select the corresponding slice along broadcast_amount
            let vec_i = vec.i((.., i as i64, ..)).squeeze_dim(1); // [mlen, hlen]
            // mat: [hlen, mlen]
            if !is_quiet() {
                println!("vec_i = {}", vec_i);
            }
            // Convert to float32 before matmul to match PyTorch golden reference
            let vec_i_f32 = vec_i.to_kind(tch::Kind::Float);
            let mat_t_f32 = mat.transpose(-1, -2).to_kind(tch::Kind::Float);
            let result = vec_i_f32.matmul(&mat_t_f32); // [mlen, mlen]
            let result = &result * (bmm_scale as f64);
            if !is_quiet() {
                println!("result = {}", result);
            }
            result_tensors.push(result);
        }
        let result_tensor = tch::Tensor::stack(&result_tensors, 0); // [broadcast_amount, mlen, mlen]

        self.hm_accum += result_tensor;
    }

    async fn btmv(&mut self, m_addr: u32, v_addr: u32, bmm_scale: f32) {
        // println!("======================== BTMV ==========================");
        // println!("m_addr = {:?}", m_addr);
        // println!("v_addr = {:?}", v_addr);
        // println!("bmm_scale = {:?}", bmm_scale);
        assert!(self.broadcast_amount * self.hlen == self.mlen);
        // Load matrix from matrix SRAM.
        let (mat_base, mat_offset) = m_addr.multiple_and_offset(self.mlen * self.mlen);
        let (mat_offset, head_offset) = mat_offset.multiple_and_offset(self.mlen);

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
        let vec = tch::Tensor::stack(&tensors, 0).view([
            1_i64,
            self.broadcast_amount as i64,
            self.hlen as i64,
        ]);

        if !is_quiet() {
            println!("btmv vec = {}", vec);
            println!("btmv mat = {}", mat);
            println!("broadcast_amount = {:?}", self.broadcast_amount);
        }

        // Now vec @ mat: [broadcast_amount, 1, hlen] @ [hlen, mlen] = [broadcast_amount, 1, mlen]
        let mut result_tensors = Vec::with_capacity(self.broadcast_amount as usize);
        for i in 0..self.broadcast_amount {
            // vec: [1, broadcast_amount, hlen]
            // For each i, select the corresponding slice along broadcast_amount
            let vec_i = vec.i((.., i as i64, ..)).squeeze_dim(1); // [1, hlen]
            // mat: [mlen, hlen]
            if !is_quiet() {
                println!("vec_i = {}", vec_i);
            }
            // Convert to float32 before matmul to match PyTorch golden reference
            let vec_i_f32 = vec_i.to_kind(tch::Kind::Float);
            let mat_t_f32 = mat.transpose(-1, -2).to_kind(tch::Kind::Float);
            let result = vec_i_f32.matmul(&mat_t_f32); // [1, mlen]
            let result = &result * (bmm_scale as f64);
            if !is_quiet() {
                println!("result = {}", result);
            }
            result_tensors.push(result);
        }
        let result_tensor = tch::Tensor::stack(&result_tensors, 0).squeeze_dim(1); // [broadcast_amount, mlen]

        self.hv_accum += result_tensor;
    }

    async fn tmm(&mut self, v_addr: u32, m_addr: u32) {
        let (mat_base, mat_offset) = m_addr.multiple_and_offset(self.mlen * self.mlen);
        let mat_offset = mat_offset.assert_multiple_of(self.mlen);
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
        let vec = tch::Tensor::stack(&tensors, 0);
        // Convert to float32 before matmul to match PyTorch golden reference
        let vec_f32 = vec.to_kind(tch::Kind::Float);
        let mat_f32 = mat.to_kind(tch::Kind::Float);
        // Now vec @ mat: [blen, mlen] @ [mlen, blen] = [blen, blen]
        self.m_accum += vec_f32.matmul(&mat_f32);
    }

    async fn mm_wo(&mut self, v_addr: u32, stride_len: u32) {
        let (vec_base, vec_offset) = v_addr.multiple_and_offset(self.mlen);
        assert!(vec_offset.is_multiple_of(self.blen));
        cycle!(1);
        // println!("======================== MM_WO ==========================");
        // println!("m accum = {}", self.m_accum);
        // println!("vec_base = {}, vec_offset = {}, stride_len = {}", vec_base, vec_offset, stride_len);
        for i in 0..self.blen {
            let tensor = self.m_accum.i((i as i64, ..));
            let old = self.vram.read(vec_base + i * self.mlen * stride_len).await;
            // println!("old = {}", old.as_tensor());
            let new = old.as_tensor().copy();
            new.i(vec_offset as i64..(vec_offset + self.blen) as i64)
                .copy_(&tensor);
            // println!("new = {}", new);
            self.vram
                .write(
                    vec_base + i * self.mlen * stride_len,
                    QuantTensor::quantize(new, old.data_type()),
                )
                .await;
        }

        self.m_accum = Tensor::zeros(
            [self.blen as i64, self.blen as i64],
            (tch::Kind::Float, tch::Device::Cpu),
        );
    }

    async fn bmm_wo(&mut self, v_addr: u32) {
        let (vec_base, vec_offset) = v_addr.multiple_and_offset(self.mlen);
        // println!("======================== BMM_WO ==========================");
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
            (tch::Kind::Float, tch::Device::Cpu),
        );
    }

    async fn bmv_wo(&mut self, v_addr: u32) {
        let (vec_base, vec_offset) = v_addr.multiple_and_offset(self.mlen);
        // println!("======================== BMV_WO ==========================");
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
        self.hv_accum = Tensor::zeros(
            [
                self.broadcast_amount as i64,
                self.mlen as i64,
            ],
            (tch::Kind::Float, tch::Device::Cpu),
        );
    }

    async fn mv(&mut self, m_addr: u32, v_addr: u32) {
        let (mat_base, mat_offset) = m_addr.multiple_and_offset(self.mlen * self.mlen);
        println!("======================== MV ==========================");
        println!("m_addr = {:?}", m_addr);
        println!("mat_offset = {:?}", mat_offset);
        println!("blen = {:?}", self.blen);
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

    async fn tmv(&mut self, m_addr: u32, v_addr: u32) {
        let (mat_base, mat_offset) = m_addr.multiple_and_offset(self.mlen * self.mlen);
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

    async fn mv_wo(&mut self, v_addr: u32) {
        let (vec_base, vec_offset) = v_addr.multiple_and_offset(self.mlen);
        assert!(vec_offset.is_multiple_of(self.blen));
        cycle!(1);
        let old = self.vram.read(vec_base).await;
        let new = old.as_tensor().copy();
        new.i(vec_offset as i64..(vec_offset + self.blen) as i64)
            .copy_(&self.v_accum);
        self.vram
            .write(
                vec_base,
                QuantTensor::quantize(new, old.data_type()),
            )
            .await;
        self.v_accum = Tensor::zeros([self.blen as i64], (tch::Kind::Float, tch::Device::Cpu));
    }
}

struct VectorMachine {
    vram: Arc<VectorSram>,
    tile_size: u32,
    mask_unit: u32,
}

impl VectorMachine {
    async fn add_scalar(&mut self, vd: u32, vs1: u32, f: f32, rmask: u8, mask: u32) {
        let a = self.vram.read(vs1).await;
        if rmask == 0 {
            let c = QuantTensor::quantize(a.as_tensor() + (f as f64), a.data_type());
            cycle!(*VECTOR_ADD_CYCLES);
            self.vram.write(vd, c).await;
        } else {
            // mask is a bitmask; each bit controls whether to apply 'f' to corresponding mask_unit-section
            let result = a.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if (mask & (1 << head)) != 0 {
                    // Mask is set for this head
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = result.narrow(0, start, end - start);
                    let updated = &sliced + (f as f64);
                    // Overwrite this section with calculated values
                    result.narrow(0, start, end - start).copy_(&updated);
                }
                // else leave unchanged
            }
            let c = QuantTensor::quantize(result, a.data_type());
            cycle!(*VECTOR_ADD_CYCLES);
            self.vram.write(vd, c).await;
        }
    }

    async fn sub_scalar(
        &mut self,
        vd: u32,
        vs1: u32,
        f: f32,
        rmask: u8,
        mask: u32,
        rorder: op::VectorOrder,
    ) {
        let a = self.vram.read(vs1).await;
        if rmask == 0 {
            if matches!(rorder, op::VectorOrder::Normal) {
                let c = QuantTensor::quantize(a.as_tensor() - (f as f64), a.data_type());
                cycle!(*VECTOR_ADD_CYCLES);
                self.vram.write(vd, c).await;
            } else {
                let c = QuantTensor::quantize((f as f64) - a.as_tensor(), a.data_type());
                cycle!(*VECTOR_ADD_CYCLES);
                self.vram.write(vd, c).await;
            }
        } else {
            // mask is a bitmask; each bit controls whether to apply 'f' to corresponding mask_unit-section
            let result = a.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if (mask & (1 << head)) != 0 {
                    // Mask is set for this head
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = result.narrow(0, start, end - start);
                    let updated = if matches!(rorder, op::VectorOrder::Normal) {
                        &sliced - (f as f64)
                    } else {
                        (f as f64) - &sliced
                    };
                    // Overwrite this section with calculated values
                    result.narrow(0, start, end - start).copy_(&updated);
                }
                // else leave unchanged
            }
            let c = QuantTensor::quantize(result, a.data_type());
            cycle!(*VECTOR_ADD_CYCLES);
            self.vram.write(vd, c).await;
        }
    }

    async fn mul_scalar(&mut self, vd: u32, vs1: u32, f: f32, rmask: u8, mask: u32) {
        let a = self.vram.read(vs1).await;
        if rmask == 0 {
            let c = QuantTensor::quantize(a.as_tensor() * (f as f64), a.data_type());
            cycle!(*VECTOR_MUL_CYCLES);
            self.vram.write(vd, c).await;
        } else {
            // println!("======================== V_MUL_VF ==========================");
            // println!("add: mask = {:?}", mask);
            // println!("a = {}", a.as_tensor());
            // println!("f = {}", f);
            let result = a.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if (mask & (1 << head)) != 0 {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = result.narrow(0, start, end - start);
                    let updated = &sliced * (f as f64);
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let c = QuantTensor::quantize(result, a.data_type());
            cycle!(*VECTOR_MUL_CYCLES);
            self.vram.write(vd, c).await;
        }
    }

    async fn add(&mut self, vd: u32, vs1: u32, vs2: u32, rmask: u8, mask: u32) {
        let (a, b) = tokio::join!(self.vram.read(vs1), self.vram.read(vs2));
        if rmask == 0 {
            let c = QuantTensor::quantize(a.as_tensor() + b.as_tensor(), a.data_type());
            cycle!(*VECTOR_ADD_CYCLES);
            self.vram.write(vd, c).await;
        } else {
            // println!("======================== V_ADD ==========================");
            // println!("add: mask = {:?}", mask);
            // println!("a = {}", a.as_tensor());
            // println!("b = {}", b.as_tensor());
            let result = a.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if (mask & (1 << head)) != 0 {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = result.narrow(0, start, end - start);
                    let updated = &sliced + b.as_tensor().narrow(0, start, end - start);
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let c = QuantTensor::quantize(result, a.data_type());
            cycle!(*VECTOR_ADD_CYCLES);
            self.vram.write(vd, c).await;
        }
    }

    async fn sub(&mut self, vd: u32, vs1: u32, vs2: u32, rmask: u8, mask: u32) {
        let (a, b) = tokio::join!(self.vram.read(vs1), self.vram.read(vs2));
        if rmask == 0 {
            let c = QuantTensor::quantize(a.as_tensor() - b.as_tensor(), a.data_type());
            cycle!(*VECTOR_ADD_CYCLES);
            self.vram.write(vd, c).await;
        } else {
            let result = a.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if (mask & (1 << head)) != 0 {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = result.narrow(0, start, end - start);
                    let updated = &sliced - b.as_tensor().narrow(0, start, end - start);
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let c = QuantTensor::quantize(result, a.data_type());
            cycle!(*VECTOR_ADD_CYCLES);
            self.vram.write(vd, c).await;
        }
    }

    async fn mul(&mut self, vd: u32, vs1: u32, vs2: u32, rmask: u8, mask: u32) {
        let (a, b) = tokio::join!(self.vram.read(vs1), self.vram.read(vs2));
        if rmask == 0 {
            let c = QuantTensor::quantize(a.as_tensor() * b.as_tensor(), a.data_type());
            cycle!(*VECTOR_MUL_CYCLES);
            self.vram.write(vd, c).await;
        } else {
            let result = a.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if (mask & (1 << head)) != 0 {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = result.narrow(0, start, end - start);
                    let updated = &sliced * b.as_tensor().narrow(0, start, end - start);
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let c = QuantTensor::quantize(result, a.data_type());
            cycle!(*VECTOR_MUL_CYCLES);
            self.vram.write(vd, c).await;
        }
    }

    async fn exp(&mut self, vd: u32, vs1: u32, rmask: u8, mask: u32) {
        let a = self.vram.read(vs1).await;
        if rmask == 0 {
            let c = QuantTensor::quantize(a.as_tensor().exp(), a.data_type());
            cycle!(*VECTOR_EXP_CYCLES);
            self.vram.write(vd, c).await;
        } else {
            let result = a.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if (mask & (1 << head)) != 0 {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = result.narrow(0, start, end - start);
                    let updated = &sliced.exp();
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let c = QuantTensor::quantize(result, a.data_type());
            cycle!(*VECTOR_EXP_CYCLES);
            self.vram.write(vd, c).await;
        }
    }

    async fn reciprocal(&mut self, vd: u32, vs1: u32, rmask: u8, mask: u32) {
        let a = self.vram.read(vs1).await;
        if rmask == 0 {
            let c = QuantTensor::quantize(a.as_tensor().reciprocal(), a.data_type());
            cycle!(*VECTOR_RECI_CYCLES);
            self.vram.write(vd, c).await;
        } else {
            let result = a.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if (mask & (1 << head)) != 0 {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = result.narrow(0, start, end - start);
                    let updated = &sliced.reciprocal();
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let c = QuantTensor::quantize(result, a.data_type());
            cycle!(*VECTOR_RECI_CYCLES);
            self.vram.write(vd, c).await;
        }
    }

    async fn vector_transfer_fp(&mut self, vd: u32, f: &[f16]) {
        assert_eq!(
            f.len(),
            self.vram.tile_size() as usize,
            "Input vector length must match tile_size"
        );
        // Convert f16 slice to f32 vector
        let f32_vec: Vec<f32> = f.iter().map(|x| f32::from(*x)).collect();
        // Create tensor from f32 vector
        let tensor = tch::Tensor::from_slice(&f32_vec);
        // Quantize the tensor according to vram data type
        let c = QuantTensor::quantize(tensor, self.vram.ty());
        cycle!(*VLEN);
        self.vram.write(vd, c).await;
    }

    async fn reduce_sum(&mut self, vs1: u32, f: f32, rmask: u8, mask: u32) -> f32 {
        let a = self.vram.read(vs1).await;
        cycle!(*VECTOR_SUM_CYCLES);
        if rmask == 0 {
            let val: f32 = a.as_tensor().sum(tch::Kind::Float).try_into().unwrap();
            f + val
        } else {
            let result = a.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if (mask & (1 << head)) != 0 {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = result.narrow(0, start, end - start);
                    let updated = &sliced.sum(tch::Kind::Float);
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let val: f32 = result.sum(tch::Kind::Float).try_into().unwrap();
            f + val
        }
    }

    async fn reduce_max(&mut self, vs1: u32, f: f32, rmask: u8, mask: u32) -> f32 {
        let a = self.vram.read(vs1).await;
        cycle!(*VECTOR_MAX_CYCLES);
        if rmask == 0 {
            let val: f32 = a.as_tensor().max().try_into().unwrap();
            f32::max(val, f)
        } else {
            let result = a.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if (mask & (1 << head)) != 0 {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = result.narrow(0, start, end - start);
                    let updated = &sliced.max();
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let val: f32 = result.max().try_into().unwrap();
            f32::max(val, f)
        }
    }
}

/// Information about an active loop
struct LoopInfo {
    start_pc: usize,          // Program counter of C_LOOP_START
    iteration_count: u32,     // Total number of iterations (from imm)
    current_iteration: u32,   // Current iteration (starts at iteration_count, decrements)
    instruction_count: usize, // Number of instructions executed in current iteration
    loop_reg: u8,             // Register used for loop counter (rd from C_LOOP_START)
}

struct Accelerator {
    m_machine: MatrixMachine,
    v_machine: VectorMachine,
    hbm: Arc<dyn ErasedMemoryModel>,
    reg_file: AcceeleratorRegFile,
    intsram: Vec<u32>,
    fpsram: Vec<f16>,
    loop_stack: Vec<LoopInfo>, // Stack for nested loops
}

struct AcceeleratorRegFile {
    gp_reg: [u32; 16],
    fp_reg: [f16; 8],
    hbm_addr_reg: [u64; 8],
    scale: u32,
    stride: u32,
    bmm_scale: f32, // Scale factor during the BMM operation, apply to every element in the matrix operation.
    v_mask: u32,    // HLEN Head Mask for VLEN Vector
}

impl Accelerator {
    /// Transfer a vector from HBM to host.
    /// Transfer data from HBM with strided loading pattern.
    /// Parameters:
    /// - index: Starting address for element data in HBM
    /// - scale_index: Starting address for scale data in HBM (for MXFP/MXINT types)
    /// - hbm_type: Data type format in HBM
    /// - sram_type: Target data type format for SRAM
    /// - stride: Byte offset between consecutive loads
    /// - load_dim: Number of elements to load per iteration
    /// - load_amount: Number of strided loads to perform
    fn transfer_mx_from_hbm(
        &mut self,
        index: u64,
        scale_index: u64,
        hbm_type: MxDataType,
        sram_type: MxDataType,
        rstride: u8,
        load_dim: u32,
        load_amount: u32,
        write_amount: u32,
    ) -> Receiver<QuantTensor> {
        // input: load_amount is how many "reads", write_amount is how many sram writes
        // write_dim = load_dim * write_amount per write, repeat for (load_amount / write_amount) times
        assert!(load_dim.is_multiple_of(write_amount));
        assert!(load_amount % write_amount == 0); // must divide evenly

        let write_dim = load_dim * write_amount; // Number of elements per write to sram
        let num_writes = load_amount / write_amount;
        let (sender, receiver) = oneshot::channel();

        let hbm_clone = self.hbm.clone();
        let stride = if rstride == 1 {
            self.reg_file.stride
        } else {
            load_dim
        };

        Executor::current().spawn(async move {
            let element_ty = hbm_type.element_type();
            let element_bits = element_ty.size_in_bits();

            // Extract scale bits and block size if Mx type, otherwise use element_bits/1 as default
            let (scale_bits, blocksize) = match hbm_type {
                MxDataType::Mx {
                    elem: _,
                    scale,
                    block,
                } => (scale.size_in_bits(), block),
                _ => (element_bits, 1), // Plain type: each element is "scaled" by 1
            };

            let element_scale_ratio = (element_bits * blocksize as u8) / scale_bits;
            let stride_scale = stride as f32 / element_scale_ratio as f32;
            assert!(element_bits.is_power_of_two());

            let len_in_bits_per_load = element_bits as u32 * load_dim;
            assert!(len_in_bits_per_load.is_multiple_of(8 * 64));
            let len_in_bytes_per_load = len_in_bits_per_load / 8;

            // Calculate scale bytes per load iteration (for Mx types)
            let (scale_len_in_bytes_per_load, block) = if let MxDataType::Mx {
                elem: _,
                scale,
                block,
            } = hbm_type
            {
                let scale_bits = scale.size_in_bits();
                assert!(scale_bits.is_power_of_two());
                let scale_len_in_bits_per_load = scale_bits as u32 * (load_dim / block);
                assert!(scale_len_in_bits_per_load.is_multiple_of(8));
                (scale_len_in_bits_per_load / 8, block as usize)
            } else {
                (0, usize::MAX)
            };

            // Total elements/bytes for all writes:
            let total_elements = (write_dim * num_writes) as usize;
            let total_bytes = (len_in_bytes_per_load * write_amount * num_writes) as usize;
            let total_scale_bytes =
                (scale_len_in_bytes_per_load * write_amount * num_writes) as usize;

            let mut bytes = vec![0u8; total_bytes];
            let mut scale_bytes = vec![0u8; total_scale_bytes];
            let hbm_clone = &hbm_clone;

            enum ChunkType {
                Element(usize, [u8; 64], usize), // (offset, data, size)
                Scale(usize, [u8; 64], usize),
            }
            let mut futures =
                FuturesUnordered::<Pin<Box<dyn Future<Output = ChunkType> + Send>>>::new();

            // Outer loop: For each "write". Inner: gather blocks for all loads for this write.
            for write_idx in 0..num_writes {
                for block_idx in 0..write_amount {
                    // println!("stride = {:?}, stride_scale = {:?}", stride, stride_scale);
                    let load_iter = write_idx * write_amount + block_idx;
                    let element_addr = index + (load_iter * stride) as u64;
                    let scale_addr = scale_index + (load_iter as f32 * stride_scale) as u64;
                    // println!("element_addr = {:?}, scale_addr = {:?}", element_addr, scale_addr);
                    let byte_offset = (write_idx * write_amount * len_in_bytes_per_load) as usize
                        + block_idx as usize * len_in_bytes_per_load as usize;
                    let scale_byte_offset = (write_idx * write_amount * scale_len_in_bytes_per_load)
                        as usize
                        + block_idx as usize * scale_len_in_bytes_per_load as usize;

                    // Element chunks:
                    for i in 0..(len_in_bytes_per_load as usize + 63) / 64 {
                        let chunk_offset = byte_offset + i * 64;
                        let chunk_size = std::cmp::min(64, total_bytes - chunk_offset);
                        let addr = element_addr + (i * 64) as u64;
                        assert!(addr.is_multiple_of(64));
                        futures.push(Box::pin(async move {
                            let data = hbm_clone.read(addr).await;
                            ChunkType::Element(chunk_offset, data, chunk_size)
                        }));
                    }

                    // Scale chunks (if Mx type)
                    if scale_len_in_bytes_per_load > 0 {
                        // Always align to 64-byte chunk boundary for loading
                        // For scale_addr, we fetch the aligned 64-byte block, and mask/select out only what is needed
                        let aligned_scale_addr = (scale_addr / 64) * 64;
                        let within_chunk_offset = (scale_addr % 64) as usize;
                        let chunk_offset = scale_byte_offset; // where to write in scale_bytes
                        let chunk_size = std::cmp::min(64, total_scale_bytes - chunk_offset);
                        futures.push(Box::pin(async move {
                            let data = hbm_clone.read(aligned_scale_addr).await;
                            // println!("aligned_scale_addr = {:?}", aligned_scale_addr);
                            // Copy out only the relevant bytes for this scale_addr
                            // scale_len_in_bytes_per_load says how many bytes to copy from within the chunk
                            let end_offset = std::cmp::min(
                                within_chunk_offset + scale_len_in_bytes_per_load as usize,
                                64,
                            );
                            let mut selected = [0u8; 64];
                            let len_to_copy = end_offset - within_chunk_offset;
                            selected[..len_to_copy]
                                .copy_from_slice(&data[within_chunk_offset..end_offset]);
                            // println!("selected scale = {:?}", selected);
                            ChunkType::Scale(chunk_offset, selected, len_to_copy)
                        }));
                    }
                }
            }

            // Collect all HBM reads
            while let Some(chunk_result) = futures.next().await {
                match chunk_result {
                    ChunkType::Element(offset, data, size) => {
                        bytes[offset..offset + size].copy_from_slice(&data[..size]);
                    }
                    ChunkType::Scale(offset, data, size) => {
                        scale_bytes[offset..offset + size].copy_from_slice(&data[..size]);
                    }
                }
            }

            // Process each write batch
            let mut all_results: Vec<QuantTensor> = Vec::with_capacity(num_writes as usize);
            for write_idx in 0..num_writes {
                let offset = write_idx * write_dim;
                let write_elements = write_dim as usize;

                let mut vec = vec![0f32; write_elements];

                // Fill `vec` with elements for this write
                let elements_offset = write_idx * write_amount * load_dim;
                let bytes_start =
                    (write_idx * write_amount) as usize * len_in_bytes_per_load as usize;

                element_ty.convert_bytes_to_f32_vec(
                    &bytes[bytes_start..bytes_start + write_elements * (element_bits as usize / 8)],
                    &mut vec,
                );

                // Apply scaling if needed
                if let MxDataType::Mx {
                    elem: _,
                    scale,
                    block,
                } = hbm_type
                {
                    let nblocks = write_elements / block as usize;
                    let scale_bytes_start =
                        (write_idx * write_amount) as usize * scale_len_in_bytes_per_load as usize;
                    let mut scale_vec = vec![0f32; nblocks];
                    scale.convert_bytes_to_f32_vec(
                        &scale_bytes[scale_bytes_start
                            ..scale_bytes_start + nblocks * (scale_bits as usize / 8)],
                        &mut scale_vec,
                    );
                    for (elem_block, scale_val) in vec
                        .chunks_mut(block as usize)
                        .zip(scale_vec.iter().copied())
                    {
                        for elem in elem_block.iter_mut() {
                            *elem *= scale_val;
                        }
                    }
                }

                let tensor = tch::Tensor::from_slice(&vec);
                all_results.push(QuantTensor::quantize(tensor, sram_type));
            }

            // Send all results as a concatenated tensor
            // (To maintain compatibility: flatten and send as one QuantTensor)
            let full_tensor = tch::Tensor::cat(
                &all_results
                    .iter()
                    .map(|qt| qt.as_tensor())
                    .collect::<Vec<_>>(),
                0,
            );
            let _ = sender.send(QuantTensor::quantize(full_tensor, sram_type));
        });

        receiver
    }

    /// Transfer data from SRAM to HBM with strided writing pattern.
    /// Parameters:
    /// - src_addr: Starting address in Vector SRAM
    /// - index: Starting address for element data in HBM
    /// - scale_index: Starting address for scale data in HBM (for MXFP/MXINT types)
    /// - sram_type: Source data type format in SRAM
    /// - hbm_type: Target data type format for HBM
    /// - rstride: Stride mode selector
    /// - store_dim: Number of elements to store per iteration (VLEN)
    /// - store_amount: Number of strided stores to perform
    async fn transfer_mx_to_hbm(
        &mut self,
        src_addr: u32,
        index: u64,
        scale_index: u64,
        sram_type: MxDataType,
        hbm_type: MxDataType,
        rstride: u8,
        store_dim: u32,
        store_amount: u32,
    ) {
        let hbm_clone = self.hbm.clone();
        let stride = if rstride == 1 {
            self.reg_file.stride
        } else {
            store_dim
        };

        let element_ty = hbm_type.element_type();
        let element_bits = element_ty.size_in_bits();

        // Extract scale bits and block size if Mx type
        let (scale_bits, blocksize) = match hbm_type {
            MxDataType::Mx {
                elem: _,
                scale,
                block,
            } => (scale.size_in_bits(), block),
            _ => (element_bits, 1),
        };

        let element_scale_ratio = (element_bits * blocksize as u8) / scale_bits;
        let stride_scale = stride as f32 / element_scale_ratio as f32;
        assert!(element_bits.is_power_of_two());

        let len_in_bits_per_store = element_bits as u32 * store_dim;
        assert!(len_in_bits_per_store.is_multiple_of(8 * 64));
        let len_in_bytes_per_store = len_in_bits_per_store / 8;

        // Calculate scale bytes per store iteration (for Mx types)
        let (scale_len_in_bytes_per_store, block) = if let MxDataType::Mx {
            elem: _,
            scale,
            block,
        } = hbm_type
        {
            let scale_bits = scale.size_in_bits();
            assert!(scale_bits.is_power_of_two());
            let scale_len_in_bits_per_store = scale_bits as u32 * (store_dim / block);
            assert!(scale_len_in_bits_per_store.is_multiple_of(8));
            (scale_len_in_bits_per_store / 8, block as usize)
        } else {
            (0, usize::MAX)
        };

        // Read data from VRAM and convert to HBM format
        for store_iter in 0..store_amount {
            // Read from VRAM
            let src_vram_addr = src_addr + store_iter * store_dim;
            let sram_tensor = self.v_machine.vram.read(src_vram_addr).await;

            // Debug: Print VRAM data read
            if !is_quiet() {
                let vram_data = sram_tensor.as_tensor();
                let vram_size = vram_data.size1().unwrap() as usize;
                let vram_slice = unsafe {
                    core::slice::from_raw_parts(
                        vram_data.data_ptr() as *const f32,
                        vram_size.min(store_dim as usize),
                    )
                };
                eprintln!(
                    "[H_STORE_V] Store iter {}: VRAM[{}] -> {} FP32 values",
                    store_iter,
                    src_vram_addr,
                    vram_slice.len()
                );
                eprintln!(
                    "  VRAM data (first 8): {:?}",
                    &vram_slice[..vram_slice.len().min(8)]
                );
            }

            // Convert from SRAM type to HBM type
            let mut hbm_tensor =
                QuantTensor::quantize(sram_tensor.as_tensor().shallow_clone(), hbm_type);

            // Convert to bytes (element bytes + scale bytes)
            let (element_bytes, scale_bytes) = hbm_tensor.into_bytes();

            // Verify scale bytes length matches expected
            // if scale_len_in_bytes_per_store > 0 {
            //     assert_eq!(
            //         scale_bytes.len(),
            //         scale_len_in_bytes_per_store as usize,
            //         scale_len_in_bytes_per_store,
            //         scale_bytes.len()
            //     );
            // }

            // Debug: Print converted HBM data
            if !is_quiet() {
                eprintln!("  Converted to HBM format:");
                eprintln!(
                    "    Element bytes: {} bytes (first 16): {:?}",
                    element_bytes.len(),
                    &element_bytes[..element_bytes.len().min(16)]
                );
                if !scale_bytes.is_empty() {
                    eprintln!(
                        "    Scale bytes: {} bytes (expected {}): {:?}",
                        scale_bytes.len(),
                        scale_len_in_bytes_per_store,
                        &scale_bytes[..scale_bytes.len().min(8)]
                    );
                }
            }

            // Calculate HBM addresses
            let element_addr = index + (store_iter * stride) as u64;
            let scale_addr = scale_index + (store_iter as f32 * stride_scale) as u64;

            // Write element bytes to HBM (64-byte aligned chunks)
            for i in 0..(len_in_bytes_per_store as usize + 63) / 64 {
                let chunk_offset = i * 64;
                let chunk_size = std::cmp::min(64, len_in_bytes_per_store as usize - chunk_offset);
                let addr = element_addr + (i * 64) as u64;
                assert!(addr.is_multiple_of(64));

                let mut chunk = [0u8; 64];
                if chunk_offset < element_bytes.len() {
                    let copy_len = std::cmp::min(chunk_size, element_bytes.len() - chunk_offset);
                    chunk[..copy_len]
                        .copy_from_slice(&element_bytes[chunk_offset..chunk_offset + copy_len]);
                }
                hbm_clone.write(addr, chunk).await;
            }

            // Write scale bytes to HBM (if Mx type)
            // Handle scales that may span multiple 64-byte chunks
            if scale_len_in_bytes_per_store > 0 {
                let mut scale_bytes_written = 0;
                let total_scale_bytes = scale_len_in_bytes_per_store as usize;

                // Write scales in 64-byte chunks, handling unaligned addresses
                while scale_bytes_written < total_scale_bytes {
                    let current_scale_addr = scale_addr + scale_bytes_written as u64;
                    let aligned_scale_addr = (current_scale_addr / 64) * 64;
                    let within_chunk_offset = (current_scale_addr % 64) as usize;

                    // Read existing chunk
                    let mut existing_chunk = hbm_clone.read(aligned_scale_addr).await;

                    // Calculate how many bytes we can write in this chunk
                    let bytes_remaining = total_scale_bytes - scale_bytes_written;
                    let bytes_in_chunk = std::cmp::min(64 - within_chunk_offset, bytes_remaining);
                    let bytes_to_copy =
                        std::cmp::min(bytes_in_chunk, scale_bytes.len() - scale_bytes_written);

                    if bytes_to_copy > 0 {
                        // Debug: Print scale write info
                        if !is_quiet() && scale_bytes_written == 0 {
                            eprintln!(
                                "    Writing scale: {} total bytes starting at HBM[0x{:x}]",
                                total_scale_bytes, scale_addr
                            );
                            eprintln!(
                                "      First chunk: {} bytes at HBM[0x{:x}] (offset within chunk: {})",
                                bytes_to_copy, aligned_scale_addr, within_chunk_offset
                            );
                            eprintln!(
                                "      Scale data (hex): {:02x?}",
                                &scale_bytes[scale_bytes_written
                                    ..(scale_bytes_written + bytes_to_copy)
                                        .min(scale_bytes_written + 8)]
                            );
                        }

                        // Copy scale bytes into the chunk
                        existing_chunk[within_chunk_offset..within_chunk_offset + bytes_to_copy]
                            .copy_from_slice(
                                &scale_bytes
                                    [scale_bytes_written..scale_bytes_written + bytes_to_copy],
                            );

                        // Write the modified chunk back
                        hbm_clone.write(aligned_scale_addr, existing_chunk).await;

                        // Verify: Read back from both hbm_clone and self.hbm to check consistency
                        if !is_quiet() && scale_bytes_written == 0 {
                            let verify_chunk_clone = hbm_clone.read(aligned_scale_addr).await;
                            let verify_chunk_self = self.hbm.read(aligned_scale_addr).await;
                            let verify_slice_clone = &verify_chunk_clone
                                [within_chunk_offset..within_chunk_offset + bytes_to_copy];
                            let verify_slice_self = &verify_chunk_self
                                [within_chunk_offset..within_chunk_offset + bytes_to_copy];
                            let expected_slice = &scale_bytes
                                [scale_bytes_written..scale_bytes_written + bytes_to_copy];

                            if verify_slice_clone != expected_slice {
                                eprintln!(
                                    "    WARNING: Scale write verification failed (hbm_clone)!"
                                );
                                eprintln!("      Expected: {:02x?}", expected_slice);
                                eprintln!("      Got:      {:02x?}", verify_slice_clone);
                            } else if verify_slice_self != expected_slice {
                                eprintln!(
                                    "    WARNING: Scale write to hbm_clone succeeded, but self.hbm doesn't match!"
                                );
                                eprintln!("      Expected: {:02x?}", expected_slice);
                                eprintln!("      hbm_clone: {:02x?}", verify_slice_clone);
                                eprintln!("      self.hbm:  {:02x?}", verify_slice_self);
                            } else {
                                eprintln!(
                                    "    Scale write verified successfully (both hbm_clone and self.hbm)"
                                );
                            }
                        }

                        scale_bytes_written += bytes_to_copy;
                    } else {
                        break;
                    }
                }

                if !is_quiet() {
                    eprintln!(
                        "    Wrote {} scale bytes total (expected {})",
                        scale_bytes_written, total_scale_bytes
                    );
                    if scale_bytes_written != total_scale_bytes {
                        eprintln!("    ERROR: Scale bytes written mismatch!");
                    }
                }
            }

            if !is_quiet() {
                eprintln!("[H_STORE_V] Store iter {} completed\n", store_iter);
            }
        }

        // Final verification: Read back all scales that were written
        // Read from self.hbm (not clone) to verify writes are visible
        if !is_quiet() && scale_len_in_bytes_per_store > 0 {
            eprintln!("[H_STORE_V] Final scale verification (reading from self.hbm):");
            for store_iter in 0..store_amount {
                let scale_addr = scale_index + (store_iter as f32 * stride_scale) as u64;
                let aligned_scale_addr = (scale_addr / 64) * 64;
                let within_chunk_offset = (scale_addr % 64) as usize;
                let verify_chunk = self.hbm.read(aligned_scale_addr).await;
                let verify_len = std::cmp::min(
                    scale_len_in_bytes_per_store as usize,
                    64 - within_chunk_offset,
                );
                if verify_len > 0 {
                    let verify_slice =
                        &verify_chunk[within_chunk_offset..within_chunk_offset + verify_len];
                    eprintln!(
                        "  Store iter {}: Scale at HBM[0x{:x}]: {:02x?} (first {} bytes)",
                        store_iter,
                        scale_addr,
                        &verify_slice[..verify_len.min(8)],
                        verify_len
                    );
                }
            }
        }
    }

    async fn do_ops(&mut self, ops: &[op::Opcode]) {
        let mut pc: usize = 0; // Program counter

        while pc < ops.len() {
            let op = &ops[pc];

            // Update instruction count for active loops
            for loop_info in &mut self.loop_stack {
                loop_info.instruction_count += 1;
                // Check if we've exceeded the max instructions limit
                if loop_info.instruction_count > *MAX_LOOP_INSTRUCTIONS {
                    panic!(
                        "Loop at PC {} exceeded max instructions limit ({}). Current iteration: {}, Instructions in this iteration: {}",
                        loop_info.start_pc,
                        *MAX_LOOP_INSTRUCTIONS,
                        loop_info.current_iteration,
                        loop_info.instruction_count
                    );
                }
            }

            if !is_quiet() {
                println!("execute op[{pc}] = {:?}", op);
            }

            let mut jump_pc: Option<usize> = None;

            match op {
                op::Opcode::Invalid => todo!(),

                op::Opcode::M_MM { rs1, rs2 } => {
                    self.m_machine
                        .mm(
                            self.reg_file.gp_reg[*rs1 as usize],
                            self.reg_file.gp_reg[*rs2 as usize],
                        )
                        .await;
                }
                op::Opcode::M_MM_WO { rd, rstride, imm } => {
                    let stride_len = if *rstride == 0 {
                        1
                    } else {
                        self.reg_file.gp_reg[*rstride as usize]
                    };
                    println!("stride_len = {:?}", stride_len);
                    self.m_machine
                        .mm_wo(
                            self.reg_file.gp_reg[*rd as usize] + *imm as u32,
                            stride_len as u32,
                        )
                        .await;
                }
                op::Opcode::M_TMM { rs1, rs2 } => {
                    self.m_machine
                        .tmm(
                            self.reg_file.gp_reg[*rs1 as usize],
                            self.reg_file.gp_reg[*rs2 as usize],
                        )
                        .await;
                }
                op::Opcode::M_BMM { rs1, rs2, rd } => {
                    self.m_machine
                        .bmm(
                            self.reg_file.gp_reg[*rs1 as usize],
                            self.reg_file.gp_reg[*rs2 as usize],
                            self.reg_file.bmm_scale,
                        )
                        .await;
                }
                op::Opcode::M_BTMM { rs1, rs2, rd } => {
                    self.m_machine
                        .btmm(
                            self.reg_file.gp_reg[*rs1 as usize],
                            self.reg_file.gp_reg[*rs2 as usize],
                            self.reg_file.bmm_scale,
                        )
                        .await;
                }
                op::Opcode::M_BMM_WO { rd, imm } => {
                    self.m_machine
                        .bmm_wo(self.reg_file.gp_reg[*rd as usize] + *imm as u32)
                        .await;
                }
                op::Opcode::M_MV { rs1, rs2 } => {
                    self.m_machine
                        .mv(
                            self.reg_file.gp_reg[*rs1 as usize],
                            self.reg_file.gp_reg[*rs2 as usize],
                        )
                        .await;
                }
                op::Opcode::M_TMV { rs1, rs2 } => {
                    self.m_machine
                        .tmv(
                            self.reg_file.gp_reg[*rs1 as usize],
                            self.reg_file.gp_reg[*rs2 as usize],
                        )
                        .await;
                }
                op::Opcode::M_BMV { rs1, rs2, rd } => {
                    self.m_machine
                        .bmv(
                            self.reg_file.gp_reg[*rs1 as usize]
                                + self.reg_file.gp_reg[*rd as usize],
                            self.reg_file.gp_reg[*rs2 as usize],
                            self.reg_file.bmm_scale,
                        )
                        .await;
                }
                op::Opcode::M_BTMV { rs1, rs2, rd } => {
                    self.m_machine
                        .btmv(
                            self.reg_file.gp_reg[*rs1 as usize]
                                + self.reg_file.gp_reg[*rd as usize],
                            self.reg_file.gp_reg[*rs2 as usize],
                            self.reg_file.bmm_scale,
                        )
                        .await;
                }
                op::Opcode::M_MV_WO { rd, imm } => {
                    self.m_machine
                        .mv_wo(self.reg_file.gp_reg[*rd as usize] + *imm as u32)
                        .await;
                }
                op::Opcode::M_BMV_WO { rd, imm } => {
                    self.m_machine
                        .bmv_wo(self.reg_file.gp_reg[*rd as usize] + *imm as u32)
                        .await;
                }

                op::Opcode::V_ADD_VV {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                } => {
                    let mask = if *rmask == 0 {
                        (1 << *HLEN as u32) - 1
                    } else {
                        self.reg_file.v_mask
                    };
                    self.v_machine
                        .add(
                            self.reg_file.gp_reg[*rd as usize],
                            self.reg_file.gp_reg[*rs1 as usize],
                            self.reg_file.gp_reg[*rs2 as usize],
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_ADD_VF {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                } => {
                    let mask = if *rmask == 0 {
                        (1 << *HLEN as u32) - 1
                    } else {
                        self.reg_file.v_mask
                    };
                    self.v_machine
                        .add_scalar(
                            self.reg_file.gp_reg[*rd as usize],
                            self.reg_file.gp_reg[*rs1 as usize],
                            self.reg_file.fp_reg[*rs2 as usize].into(),
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_SUB_VV {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                } => {
                    let mask = if *rmask == 0 {
                        (1 << *HLEN as u32) - 1
                    } else {
                        self.reg_file.v_mask
                    };
                    self.v_machine
                        .sub(
                            self.reg_file.gp_reg[*rd as usize],
                            self.reg_file.gp_reg[*rs1 as usize],
                            self.reg_file.gp_reg[*rs2 as usize],
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_SUB_VF {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                    rorder,
                } => {
                    let mask = if *rmask == 0 {
                        (1 << *HLEN as u32) - 1
                    } else {
                        self.reg_file.v_mask
                    };
                    self.v_machine
                        .sub_scalar(
                            self.reg_file.gp_reg[*rd as usize],
                            self.reg_file.gp_reg[*rs1 as usize],
                            self.reg_file.fp_reg[*rs2 as usize].into(),
                            *rmask,
                            mask,
                            *rorder,
                        )
                        .await;
                }
                op::Opcode::V_MUL_VV {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                } => {
                    let mask = if *rmask == 0 {
                        (1 << *HLEN as u32) - 1
                    } else {
                        self.reg_file.v_mask
                    };
                    self.v_machine
                        .mul(
                            self.reg_file.gp_reg[*rd as usize],
                            self.reg_file.gp_reg[*rs1 as usize],
                            self.reg_file.gp_reg[*rs2 as usize],
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_MUL_VF {
                    rd,
                    rs1,
                    rs2,
                    rmask,
                } => {
                    let mask = if *rmask == 0 {
                        (1 << *HLEN as u32) - 1
                    } else {
                        self.reg_file.v_mask
                    };
                    self.v_machine
                        .mul_scalar(
                            self.reg_file.gp_reg[*rd as usize],
                            self.reg_file.gp_reg[*rs1 as usize],
                            self.reg_file.fp_reg[*rs2 as usize].into(),
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_EXP_V { rd, rs1, rmask } => {
                    let mask = if *rmask == 0 {
                        (1 << *HLEN as u32) - 1
                    } else {
                        self.reg_file.v_mask
                    };
                    self.v_machine
                        .exp(
                            self.reg_file.gp_reg[*rd as usize],
                            self.reg_file.gp_reg[*rs1 as usize],
                            *rmask,
                            mask,
                        )
                        .await;
                }
                op::Opcode::V_RECI_V { rd, rs1, rmask } => {
                    let mask = if *rmask == 0 {
                        (1 << *HLEN as u32) - 1
                    } else {
                        self.reg_file.v_mask
                    };
                    self.v_machine
                        .reciprocal(
                            self.reg_file.gp_reg[*rd as usize],
                            self.reg_file.gp_reg[*rs1 as usize],
                            *rmask,
                            mask,
                        )
                        .await;
                }

                // Write to fp0 is a no-op.
                op::Opcode::V_RED_SUM { rd: 0, .. } | op::Opcode::V_RED_MAX { rd: 0, .. } => (),

                op::Opcode::V_RED_SUM { rd, rs1, rmask } => {
                    let mask = if *rmask == 0 {
                        (1 << *HLEN as u32) - 1
                    } else {
                        self.reg_file.v_mask
                    };
                    let result = self
                        .v_machine
                        .reduce_sum(
                            self.reg_file.gp_reg[*rs1 as usize],
                            self.reg_file.fp_reg[*rd as usize].into(),
                            *rmask,
                            mask,
                        )
                        .await;
                    self.reg_file.fp_reg[*rd as usize] = f16::from_f32(result);
                }
                op::Opcode::V_RED_MAX { rd, rs1, rmask } => {
                    let mask = if *rmask == 0 {
                        (1 << *HLEN as u32) - 1
                    } else {
                        self.reg_file.v_mask
                    };
                    let result = self
                        .v_machine
                        .reduce_max(
                            self.reg_file.gp_reg[*rs1 as usize],
                            self.reg_file.fp_reg[*rd as usize].into(),
                            *rmask,
                            mask,
                        )
                        .await;
                    self.reg_file.fp_reg[*rd as usize] = f16::from_f32(result);
                }

                // Write to fp0 is a no-op.
                op::Opcode::S_ADD_FP { rd: 0, .. }
                | op::Opcode::S_SUB_FP { rd: 0, .. }
                | op::Opcode::S_MAX_FP { rd: 0, .. }
                | op::Opcode::S_MUL_FP { rd: 0, .. }
                | op::Opcode::S_EXP_FP { rd: 0, .. }
                | op::Opcode::S_RECI_FP { rd: 0, .. }
                | op::Opcode::S_SQRT_FP { rd: 0, .. } => {}

                op::Opcode::S_ADD_FP { rd, rs1, rs2 } => {
                    self.reg_file.fp_reg[*rd as usize] =
                        self.reg_file.fp_reg[*rs1 as usize] + self.reg_file.fp_reg[*rs2 as usize];
                    cycle!(*SCALAR_FP_BASIC_CYCLES);
                }
                op::Opcode::S_SUB_FP { rd, rs1, rs2 } => {
                    self.reg_file.fp_reg[*rd as usize] =
                        self.reg_file.fp_reg[*rs1 as usize] - self.reg_file.fp_reg[*rs2 as usize];
                    cycle!(*SCALAR_FP_BASIC_CYCLES);
                }
                op::Opcode::S_MAX_FP { rd, rs1, rs2 } => {
                    self.reg_file.fp_reg[*rd as usize] = f16::max(
                        self.reg_file.fp_reg[*rs1 as usize],
                        self.reg_file.fp_reg[*rs2 as usize],
                    );
                    cycle!(*SCALAR_FP_BASIC_CYCLES);
                }
                op::Opcode::S_MUL_FP { rd, rs1, rs2 } => {
                    self.reg_file.fp_reg[*rd as usize] =
                        self.reg_file.fp_reg[*rs1 as usize] * self.reg_file.fp_reg[*rs2 as usize];
                    cycle!(*SCALAR_FP_BASIC_CYCLES);
                }
                op::Opcode::S_EXP_FP { rd, rs1 } => {
                    self.reg_file.fp_reg[*rd as usize] =
                        f16::from_f32(f32::exp(self.reg_file.fp_reg[*rs1 as usize].into()));
                    cycle!(*SCALAR_FP_EXP_CYCLES);
                }
                op::Opcode::S_RECI_FP { rd, rs1 } => {
                    self.reg_file.fp_reg[*rd as usize] =
                        f16::ONE / self.reg_file.fp_reg[*rs1 as usize];
                    cycle!(*SCALAR_FP_RECI_CYCLES);
                }
                op::Opcode::S_SQRT_FP { rd, rs1 } => {
                    self.reg_file.fp_reg[*rd as usize] =
                        f16::from_f32(f32::from(self.reg_file.fp_reg[*rs1 as usize]).sqrt());
                    cycle!(*SCALAR_FP_SQRT_CYCLES);
                }
                op::Opcode::S_LD_FP { rd, rs1, imm } => {
                    self.reg_file.fp_reg[*rd as usize] =
                        self.fpsram[(self.reg_file.gp_reg[*rs1 as usize] + *imm) as usize];
                    cycle!(1);
                }
                op::Opcode::S_ST_FP { rd, rs1, imm } => {
                    self.fpsram[(self.reg_file.gp_reg[*rs1 as usize] + *imm) as usize] =
                        self.reg_file.fp_reg[*rd as usize];
                    cycle!(1);
                }
                op::Opcode::S_MAP_V_FP { rd, rs1, imm } => {
                    let start_idx = (self.reg_file.gp_reg[*rs1 as usize] + *imm) as usize;
                    let end_idx = start_idx + *VLEN as usize;
                    let f = &self.fpsram[start_idx..end_idx];
                    self.v_machine
                        .vector_transfer_fp(self.reg_file.gp_reg[*rd as usize], f)
                        .await;
                    cycle!(*VLEN);
                }
                op::Opcode::S_ADD_INT { rd, rs1, rs2 } => {
                    self.reg_file.gp_reg[*rd as usize] = self.reg_file.gp_reg[*rs1 as usize]
                        .wrapping_add(self.reg_file.gp_reg[*rs2 as usize]);
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::S_ADDI_INT { rd, rs1, imm } => {
                    self.reg_file.gp_reg[*rd as usize] =
                        self.reg_file.gp_reg[*rs1 as usize].wrapping_add(*imm as u32);
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::S_SUB_INT { rd, rs1, rs2 } => {
                    self.reg_file.gp_reg[*rd as usize] = self.reg_file.gp_reg[*rs1 as usize]
                        .wrapping_sub(self.reg_file.gp_reg[*rs2 as usize]);
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::S_MUL_INT { rd, rs1, rs2 } => {
                    self.reg_file.gp_reg[*rd as usize] = self.reg_file.gp_reg[*rs1 as usize]
                        .wrapping_mul(self.reg_file.gp_reg[*rs2 as usize]);
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::S_LUI_INT { rd, imm } => {
                    self.reg_file.gp_reg[*rd as usize] = (*imm as u32) << 12;
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::S_LD_INT { rd, rs1, imm } => {
                    self.reg_file.gp_reg[*rd as usize] =
                        self.intsram[(self.reg_file.gp_reg[*rs1 as usize] + *imm) as usize];
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::S_ST_INT { rd, rs1, imm } => {
                    self.intsram[(self.reg_file.gp_reg[*rs1 as usize] + *imm) as usize] =
                        self.reg_file.gp_reg[*rd as usize];
                    cycle!(*SCALAR_INT_BASIC_CYCLES);
                }
                op::Opcode::H_PREFETCH_M {
                    rd,
                    rs1,
                    rs2,
                    rstride,
                    precision,
                } => {
                    // TODO: rstride support to be added
                    let offset = self.reg_file.gp_reg[*rs1 as usize];
                    let addr = self.reg_file.hbm_addr_reg[*rs2 as usize];
                    let dtype = match precision {
                        op::MatrixPrecision::Weights => *MATRIX_WEIGHT_TYPE,
                        op::MatrixPrecision::KeyValue => *MATRIX_KV_TYPE,
                    };

                    let scale = match dtype {
                        MxDataType::Plain(_) => 0,
                        MxDataType::Mx { elem, scale, block } => {
                            offset
                                / (elem.size_in_bits() as u32 * block / scale.size_in_bits() as u32)
                        } // Element addr shifted by (element to scale ratio)
                    };
                    let xfer = self.transfer_mx_from_hbm(
                        addr + offset as u64,
                        addr + self.reg_file.scale as u64 + scale as u64,
                        dtype,
                        self.m_machine.mram.ty,
                        *rstride,
                        *MLEN,
                        *PREFETCH_M_AMOUNT,
                        *MLEN,
                    );

                    self.m_machine
                        .mram
                        .continous_write_delayed(
                            self.reg_file.gp_reg[*rd as usize],
                            *PREFETCH_M_AMOUNT,
                            xfer,
                        )
                        .await;
                }
                op::Opcode::H_PREFETCH_V {
                    rd,
                    rs1,
                    rs2,
                    rstride,
                    precision,
                } => {
                    // TODO: rstride support to be added
                    let offset = self.reg_file.gp_reg[*rs1 as usize];
                    let addr = self.reg_file.hbm_addr_reg[*rs2 as usize];
                    let dtype = match precision {
                        op::VectorPrecision::Activation => *VECTOR_ACTIVATION_TYPE,
                        op::VectorPrecision::KeyValue => *VECTOR_KV_TYPE,
                    };

                    let scale = match dtype {
                        MxDataType::Plain(_) => 0,
                        MxDataType::Mx { elem, scale, block } => {
                            offset
                                / (elem.size_in_bits() as u32 * block / scale.size_in_bits() as u32)
                        }
                    };
                    let xfer = self.transfer_mx_from_hbm(
                        addr + offset as u64,
                        addr + self.reg_file.scale as u64 + scale as u64,
                        dtype,
                        self.v_machine.vram.ty(),
                        *rstride,
                        *VLEN,
                        *PREFETCH_V_AMOUNT,
                        1,
                    );

                    let dest = self.reg_file.gp_reg[*rd as usize];
                    self.v_machine
                        .vram
                        .continous_write_delayed(dest, *PREFETCH_V_AMOUNT, xfer)
                        .await;
                }
                op::Opcode::H_STORE_V {
                    rd,
                    rs1,
                    rs2,
                    rstride,
                    precision,
                } => {
                    let src_addr = self.reg_file.gp_reg[*rd as usize];
                    let offset = self.reg_file.gp_reg[*rs1 as usize];
                    let addr = self.reg_file.hbm_addr_reg[*rs2 as usize];
                    let dtype = match precision {
                        op::VectorPrecision::Activation => *VECTOR_ACTIVATION_TYPE,
                        op::VectorPrecision::KeyValue => *VECTOR_KV_TYPE,
                    };

                    let scale = match dtype {
                        MxDataType::Plain(_) => 0,
                        MxDataType::Mx { elem, scale, block } => {
                            offset
                                / (elem.size_in_bits() as u32 * block / scale.size_in_bits() as u32)
                        }
                    };

                    let element_index = addr + offset as u64;
                    // Scales are stored AFTER elements, so scale_index = element_index + scale_reg + scale
                    // where scale_reg is the offset from element start to scale start
                    let scale_index = addr + self.reg_file.scale as u64 + scale as u64;

                    self.transfer_mx_to_hbm(
                        src_addr,
                        element_index,
                        scale_index,
                        self.v_machine.vram.ty(),
                        dtype,
                        *rstride,
                        *VLEN,
                        *STORE_V_AMOUNT,
                    )
                    .await;
                }
                op::Opcode::C_SET_ADDR_REG { rd, rs1, rs2 } => {
                    let imm = ((self.reg_file.gp_reg[*rs1 as usize] as u64) << 32)
                        | (self.reg_file.gp_reg[*rs2 as usize] as u64);
                    self.reg_file.hbm_addr_reg[*rd as usize] = imm;
                    cycle!(1);
                }
                op::Opcode::C_SET_SCALE_REG { rd } => {
                    self.reg_file.scale = self.reg_file.gp_reg[*rd as usize];
                    cycle!(1);
                }
                op::Opcode::C_SET_STRIDE_REG { rd } => {
                    self.reg_file.stride = self.reg_file.gp_reg[*rd as usize];
                    cycle!(1);
                }
                op::Opcode::C_SET_V_MASK_REG { rd } => {
                    self.reg_file.v_mask = self.reg_file.gp_reg[*rd as usize];
                    cycle!(1);
                }
                op::Opcode::C_LOOP_START { rd, imm } => {
                    // Store iteration count in register
                    assert!(*imm > 0, "Iteration count must be greater than 0");
                    let iteration_count = *imm as u32;
                    self.reg_file.gp_reg[*rd as usize] = iteration_count;

                    // Push new loop onto stack
                    self.loop_stack.push(LoopInfo {
                        start_pc: pc,
                        iteration_count,
                        current_iteration: iteration_count,
                        instruction_count: 0,
                        loop_reg: *rd,
                    });

                    if !is_quiet() {
                        println!(
                            "C_LOOP_START: Starting loop at PC {} with {} iterations",
                            pc, iteration_count
                        );
                    }
                    cycle!(1);
                }
                op::Opcode::C_LOOP_END { rd } => {
                    // Find the matching loop (most recent loop with matching register)
                    if let Some(loop_info) =
                        self.loop_stack.iter_mut().rev().find(|l| l.loop_reg == *rd)
                    {
                        // Decrement the register (as per spec)
                        let reg_value = self.reg_file.gp_reg[*rd as usize];
                        if reg_value > 1 {
                            // More iterations remaining, loop back
                            self.reg_file.gp_reg[*rd as usize] = reg_value - 1;

                            // Update loop state
                            loop_info.current_iteration = reg_value - 1;
                            loop_info.instruction_count = 0; // Reset instruction count for next iteration

                            // Jump back to C_LOOP_START + 1 (skip the C_LOOP_START instruction itself)
                            jump_pc = Some(loop_info.start_pc + 1);

                            if !is_quiet() {
                                println!(
                                    "C_LOOP_END: Looping back to PC {} (remaining iterations: {})",
                                    loop_info.start_pc + 1,
                                    reg_value - 1
                                );
                            }
                        } else {
                            // Last iteration (reg_value == 1) or already done (reg_value == 0)
                            // Decrement to 0 and exit the loop
                            self.reg_file.gp_reg[*rd as usize] = 0;

                            // Loop is complete, pop it from stack
                            if !is_quiet() {
                                println!(
                                    "C_LOOP_END: Loop at PC {} completed (executed {} times)",
                                    loop_info.start_pc, loop_info.iteration_count
                                );
                            }
                            // Remove this loop from the stack
                            let loop_reg = loop_info.loop_reg;
                            let pos = self
                                .loop_stack
                                .iter()
                                .rposition(|l| l.loop_reg == loop_reg)
                                .unwrap();
                            self.loop_stack.remove(pos);
                        }
                    } else {
                        panic!(
                            "C_LOOP_END: No matching C_LOOP_START found for register {}",
                            *rd
                        );
                    }
                    cycle!(1);
                }
                op::Opcode::C_BREAK => {
                    // Break out of the innermost loop
                    if let Some(loop_info) = self.loop_stack.pop() {
                        if !is_quiet() {
                            println!("C_BREAK: Breaking out of loop at PC {}", loop_info.start_pc);
                        }
                        // Set the loop register to 0 to indicate loop is done
                        self.reg_file.gp_reg[loop_info.loop_reg as usize] = 0;
                    } else {
                        panic!("C_BREAK: No active loop to break out of");
                    }
                    cycle!(1);
                }
            }

            // Handle loop jumps
            if let Some(target_pc) = jump_pc {
                pc = target_pc;
            } else {
                pc += 1;
            }
        }
    }
}

#[derive(Parser)]
struct Opts {
    #[arg(long)]
    /// Path to Instruction to be executed.
    opcode: PathBuf,

    #[arg(long)]
    /// Path to HBM contents for preloading.
    hbm: PathBuf,

    #[arg(long)]
    /// Path to FP SRAM contents for preloading.
    fpsram: PathBuf,

    #[arg(long)]
    /// Path to INT SRAM contents for preloading.
    intsram: Option<PathBuf>,

    #[arg(long)]
    /// Path to file storing Vector SRAM contents (optional).
    vram: Option<PathBuf>,

    #[arg(long, short)]
    /// Quiet mode: only output final latency and statistics.
    quiet: bool,
}

static QUIET_MODE: LazyLock<std::sync::atomic::AtomicBool> =
    LazyLock::new(|| std::sync::atomic::AtomicBool::new(false));

fn is_quiet() -> bool {
    QUIET_MODE.load(std::sync::atomic::Ordering::Relaxed)
}

async fn start() {
    let opts = Opts::parse();
    QUIET_MODE.store(opts.quiet, std::sync::atomic::Ordering::Relaxed);
    let mram = Arc::new(MatrixSram::new(*MLEN, *MATRIX_SRAM_SIZE, *MATRIX_SRAM_TYPE)); // Matrix SRAM
    let vram = Arc::new(VectorSram::from_mx_type(
        *VLEN,
        *VECTOR_SRAM_SIZE,
        *VECTOR_SRAM_TYPE,
    )); // Vector SRAM

    let m_machine = MatrixMachine {
        mram,
        vram: vram.clone(),
        mlen: *MLEN,
        hlen: *HLEN,
        blen: *BLEN,
        m_accum: Tensor::zeros(
            [*BLEN as i64, *BLEN as i64],
            (tch::Kind::Float, tch::Device::Cpu),
        ),
        hm_accum: Tensor::zeros(
            [*BROADCAST_AMOUNT as i64, *MLEN as i64, *MLEN as i64],
            (tch::Kind::Float, tch::Device::Cpu),
        ),
        hv_accum: Tensor::zeros(
            [*BROADCAST_AMOUNT as i64, *MLEN as i64],
            (tch::Kind::Float, tch::Device::Cpu),
        ),
        v_accum: Tensor::zeros([*BLEN as i64], (tch::Kind::Float, tch::Device::Cpu)),
        broadcast_amount: *BROADCAST_AMOUNT,
    };

    let v_machine = VectorMachine {
        vram,
        tile_size: *VLEN,
        mask_unit: *HLEN,
    }; // Share same dim with VSRAM

    let hbm = Arc::new(memory::WithStats::new(memory::WithTiming::new(
        ManuallyDrop::new(ramulator::Ramulator::hbm2_preset(8).unwrap()),
        memory::MemoryBacked::with_capacity(*HBM_SIZE),
    )));

    let mut accelerator = Accelerator {
        m_machine,
        v_machine,
        hbm: hbm.clone(),
        reg_file: AcceeleratorRegFile {
            gp_reg: [0; 16],
            fp_reg: [f16::ZERO; 8],
            hbm_addr_reg: [0; 8],
            scale: 0,
            stride: 1,
            bmm_scale: 1.0,
            v_mask: 0,
        },
        intsram: vec![0; 1024],
        fpsram: vec![f16::ZERO; 1024],
        loop_stack: Vec::new(),
    };

    use std::fs;
    let op_file = fs::read_to_string(opts.opcode).unwrap();

    let op: Vec<u32> = op_file
        .split_whitespace() // split by spaces/newlines
        .map(|tok| u32::from_str_radix(tok.trim_start_matches("0x"), 16).unwrap())
        .collect();

    // Memory Initialization
    // - HBM Preload
    let hbm_data = std::fs::read(opts.hbm).unwrap();
    hbm.model().data().with_data(|f| {
        f[..hbm_data.len()].copy_from_slice(&hbm_data);
    });

    // Load fpsram and intsram as raw bytes and map to the vector files.
    // - fpsram Preload
    let fpsram_data = std::fs::read(opts.fpsram).unwrap();
    let fp_vals: &[f16] = unsafe {
        std::slice::from_raw_parts(
            fpsram_data.as_ptr() as *const f16,
            fpsram_data.len() / std::mem::size_of::<f16>(),
        )
    };

    // Replace the beginning of accelerator.fpsram with fp_vals
    accelerator.fpsram[..fp_vals.len()].copy_from_slice(&fp_vals[..fp_vals.len()]);

    // - INT SRAM Preload
    if let Some(intsram_path) = opts.intsram {
        let intsram_data = std::fs::read(intsram_path).unwrap();
        let int_vals: &[u32] = unsafe {
            std::slice::from_raw_parts(
                intsram_data.as_ptr() as *const u32,
                intsram_data.len() / std::mem::size_of::<u32>(),
            )
        };
        accelerator.intsram[..int_vals.len()].copy_from_slice(&int_vals[..int_vals.len()]);
    }
    // - VRAM Preload (if provided)
    if let Some(vram_path) = opts.vram {
        let vram_data = std::fs::read(vram_path).unwrap();
        accelerator.v_machine.vram.load_from_bytes(&vram_data).await;
    }

    // - Execute Instructions
    // accelerator
    //     .do_ops(&dbg!(
    //         op.into_iter().map(op::Opcode::decode).collect::<Vec<_>>()
    //     ))
    //     .await;
    let decoded_ops = op.into_iter().map(op::Opcode::decode).collect::<Vec<_>>();
    accelerator.do_ops(&decoded_ops).await;

    println!("gp1 = {:x}", accelerator.reg_file.gp_reg[1]);
    println!("scale = {}", accelerator.reg_file.scale);
    println!(
        "Vector SRAM Contents: \n {}",
        accelerator.v_machine.vram.read(0x0000).await.as_tensor()
    );

    println!(
        "Matrix SRAM Contents: \n {}",
        accelerator.m_machine.mram.read(0x0000).await.as_tensor()
    );

    println!("INT SRAM Contents: \n {:?}", accelerator.intsram);
    println!("FP SRAM Contents: \n {:?}", accelerator.fpsram);

    // Dump MRAM
    let mram_dump_path = "mram_dump.bin";
    let mram_bytes = accelerator.m_machine.mram.as_bytes().await;
    let mut mram_file = std::fs::File::create(mram_dump_path).unwrap();
    mram_file.write_all(&mram_bytes).unwrap();
    if !is_quiet() {
        eprintln!("Dumped MRAM content to: {:?}", mram_dump_path);
    }

    // Dump VRAM
    let vram_dump_path = "vram_dump.bin";
    let vram_bytes = accelerator.v_machine.vram.as_bytes().await;
    let mut vram_file = std::fs::File::create(vram_dump_path).unwrap();
    vram_file.write_all(&vram_bytes).unwrap();
    if !is_quiet() {
        eprintln!("Dumped VRAM content to: {:?}", vram_dump_path);
    }

    // Dump FPSRAM
    let fpsram_dump_path = "fpsram_dump.bin";
    let fpsram_bytes: Vec<u8> = accelerator
        .fpsram
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let mut fpsram_file = std::fs::File::create(fpsram_dump_path).unwrap();
    fpsram_file.write_all(&fpsram_bytes).unwrap();
    if !is_quiet() {
        eprintln!("Dumped FPSRAM content to: {:?}", fpsram_dump_path);
    }

    // Dump HBM
    let hbm_dump_path = "hbm_dump.bin";
    let hbm_size = *HBM_SIZE;
    let mut hbm_bytes = vec![0u8; hbm_size];
    hbm.model().data().with_data(|f| {
        let len = std::cmp::min(hbm_size, f.len());
        hbm_bytes[..len].copy_from_slice(&f[..len]);
    });
    let mut hbm_file = std::fs::File::create(hbm_dump_path).unwrap();
    hbm_file.write_all(&hbm_bytes).unwrap();

    let memory_stats = hbm.statistics();
    let utilization = (memory_stats.total_bytes_read + memory_stats.total_bytes_written) as f64
        / Executor::current().now().to_secs();
    eprintln!(
        "HBM Statistics - Bytes read: {:?} | Bytes written: {:?} | Utilization: {:.2e} bytes/sec",
        memory_stats.total_bytes_read, memory_stats.total_bytes_written, utilization
    );
}

#[tokio::main]
async fn main() {
    let executor = Executor::new();
    executor.spawn(start());
    executor.enter(Instant::ETERNITY).await;
    eprintln!("Simulation completed. Latency {:?}", executor.now());
}
