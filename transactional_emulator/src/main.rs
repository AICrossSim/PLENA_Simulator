#![allow(unused_variables, unused_mut, dead_code)]

mod load_config;
mod op; // Add this line to include the config module

use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::io::Write;
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, anyhow, bail};
use clap::Parser;
use futures::FutureExt;
use futures::StreamExt;
use futures::stream::FuturesUnordered;
use half::f16;
use memory::{ErasedMemoryModel, MemoryModel};
use quantize::{MxDataType, QuantTensor};
use runtime::{Duration, Executor, Instant};
use serde::Deserialize;
use serde_json::{Value, json};
use tch::{IndexOp, Tensor};
use vector_sram::VectorSram;

use tokio::io::{AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use tokio::sync::RwLock;
use tokio::sync::broadcast;
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
static FP_SRAM_SIZE: LazyLock<usize> = LazyLock::new(|| fp_sram_size());
static INT_SRAM_SIZE: LazyLock<usize> = LazyLock::new(|| int_sram_size());
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
        assert!(
            self.is_multiple_of(mul),
            "MRAM/VRAM address {} is not a multiple of {} \
             (this usually means the kernel was compiled for a different \
             config — e.g. kernel built with smaller MLEN/VLEN but simulator \
             running with the current value; rebuild the kernel against the \
             active config or run the simulator with the matching PLENA_CONFIG)",
            self, mul
        );
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
                    .read(v_addr + i * self.mlen)
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
                    .read(v_addr + i * self.mlen)
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
            [self.broadcast_amount as i64, self.mlen as i64],
            (tch::Kind::Float, tch::Device::Cpu),
        );
    }

    async fn mv(&mut self, m_addr: u32, v_addr: u32) {
        let (mat_base, mat_offset) = m_addr.multiple_and_offset(self.mlen * self.mlen);
        if !is_quiet() {
            println!("======================== MV ==========================");
            println!("m_addr = {:?}", m_addr);
            println!("mat_offset = {:?}", mat_offset);
            println!("blen = {:?}", self.blen);
        }
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
            .write(vec_base, QuantTensor::quantize(new, old.data_type()))
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

    async fn shift(&mut self, vd: u32, vs1: u32, shift_amount: u32) {
        let a = self.vram.read(vs1).await;
        let vlen = self.tile_size as i64;
        let shift = shift_amount as i64;
        let src_tensor = a.as_tensor();
        let result = tch::Tensor::zeros(&[vlen], (tch::Kind::Float, tch::Device::Cpu));
        if shift < vlen {
            let src_len = vlen - shift;
            let src_slice = src_tensor.narrow(0, 0, src_len);
            result.narrow(0, shift, src_len).copy_(&src_slice);
        }
        let c = QuantTensor::quantize(result, a.data_type());
        cycle!(*VECTOR_ADD_CYCLES);
        self.vram.write(vd, c).await;
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

// ===========================================================================
// Per-op execution trace (Gantt visualisation)
// ===========================================================================
//
// When trace collection is enabled on `EmulatorState`, every iteration of
// `Accelerator::do_ops` appends one `TraceEntry` (24 bytes, repr(C)) to
// `Accelerator.trace`. The buffer is drained back into `EmulatorState`
// after `execute_batch` completes; the client retrieves it via the
// `dump_trace { path }` protocol command. On-disk layout matches the
// in-memory struct byte-for-byte so Python can `np.fromfile` it.

/// Cap to prevent a runaway kernel OOM-ing the backend. 10M × 24 B ≈ 240 MB.
/// Beyond this the buffer silently stops growing and we set an `overflowed`
/// flag in the sidecar metadata.
const TRACE_MAX_ENTRIES: usize = 10_000_000;

/// Engine lane in the Gantt. PrefetchM and PrefetchV are separate lanes
/// because real PLENA hardware can issue them concurrently (different
/// buses + bank groups), even though the current serial dispatcher does
/// not exploit that overlap yet.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EngineKind {
    Matrix = 0,
    Vector = 1,
    Scalar = 2,
    /// `H_PREFETCH_M` — HBM → MRAM, bulk weight loads.
    PrefetchM = 3,
    /// `H_PREFETCH_V` + `H_STORE_V` — HBM ↔ VRAM, activation pre/store.
    /// Lumped because they share the V-channel + scale-table pipeline.
    PrefetchV = 4,
    Control = 5,
}

/// One row in the trace ring buffer. `repr(C)` + explicit padding so the
/// in-memory layout matches what `np.fromfile(dtype=...)` reads on disk —
/// no serialisation step needed.
#[repr(C)]
#[derive(Clone, Copy)]
struct TraceEntry {
    /// Sim-time at which this op started (picoseconds since INIT).
    start_picos: u64,
    /// Sim-time spent in this op. `u32` covers up to ~4 ms per op; no
    /// single PLENA op approaches that ceiling.
    duration_picos: u32,
    /// `EngineKind as u8`.
    engine: u8,
    /// Tag for the opcode family (see `OP_TAG_TABLE`). The sidecar JSON
    /// dumps the tag→name mapping so old traces keep rendering after
    /// new variants are added.
    op_tag: u8,
    _pad: u16,
    /// HBM bytes READ during this op (delta of the global `WithStats`
    /// counter). Reserved; currently always 0 because `Accelerator.hbm`
    /// is `dyn ErasedMemoryModel` which erases the `statistics()`
    /// accessor — we'll wire this up when the Ramulator bank-stat hook
    /// lands and we extend the trait.
    hbm_bytes_read: u32,
    /// HBM bytes WRITTEN during this op. Reserved (see above).
    hbm_bytes_written: u32,
}

const _: () = assert!(core::mem::size_of::<TraceEntry>() == 24);

/// Numeric tag for each opcode family. Stable across builds (sidecar JSON
/// carries the mapping so traces dumped today render with renderers shipped
/// later, even after new opcodes are added).
fn op_tag_of(op: &op::Opcode) -> u8 {
    match op {
        op::Opcode::Invalid => 0,
        op::Opcode::M_MM { .. } => 1,
        op::Opcode::M_TMM { .. } => 2,
        op::Opcode::M_BMM { .. } => 3,
        op::Opcode::M_BTMM { .. } => 4,
        op::Opcode::M_BMM_WO { .. } => 5,
        op::Opcode::M_MM_WO { .. } => 6,
        op::Opcode::M_MV { .. } => 7,
        op::Opcode::M_TMV { .. } => 8,
        op::Opcode::M_BMV { .. } => 9,
        op::Opcode::M_BTMV { .. } => 10,
        op::Opcode::M_MV_WO { .. } => 11,
        op::Opcode::M_BMV_WO { .. } => 12,
        op::Opcode::V_ADD_VV { .. } => 20,
        op::Opcode::V_ADD_VF { .. } => 21,
        op::Opcode::V_SUB_VV { .. } => 22,
        op::Opcode::V_SUB_VF { .. } => 23,
        op::Opcode::V_MUL_VV { .. } => 24,
        op::Opcode::V_MUL_VF { .. } => 25,
        op::Opcode::V_EXP_V { .. } => 26,
        op::Opcode::V_RECI_V { .. } => 27,
        op::Opcode::V_RED_SUM { .. } => 28,
        op::Opcode::V_RED_MAX { .. } => 29,
        op::Opcode::V_SHFT_V { .. } => 30,
        op::Opcode::S_ADD_FP { .. } => 40,
        op::Opcode::S_SUB_FP { .. } => 41,
        op::Opcode::S_MAX_FP { .. } => 42,
        op::Opcode::S_MUL_FP { .. } => 43,
        op::Opcode::S_EXP_FP { .. } => 44,
        op::Opcode::S_RECI_FP { .. } => 45,
        op::Opcode::S_SQRT_FP { .. } => 46,
        op::Opcode::S_LD_FP { .. } => 47,
        op::Opcode::S_ST_FP { .. } => 48,
        op::Opcode::S_MAP_V_FP { .. } => 49,
        op::Opcode::S_ADD_INT { .. } => 50,
        op::Opcode::S_ADDI_INT { .. } => 51,
        op::Opcode::S_SUB_INT { .. } => 52,
        op::Opcode::S_MUL_INT { .. } => 53,
        op::Opcode::S_LUI_INT { .. } => 54,
        op::Opcode::S_LD_INT { .. } => 55,
        op::Opcode::S_ST_INT { .. } => 56,
        op::Opcode::H_PREFETCH_M { .. } => 60,
        op::Opcode::H_PREFETCH_V { .. } => 61,
        op::Opcode::H_STORE_V { .. } => 62,
        op::Opcode::C_SET_ADDR_REG { .. } => 70,
        op::Opcode::C_SET_SCALE_REG { .. } => 71,
        op::Opcode::C_SET_STRIDE_REG { .. } => 72,
        op::Opcode::C_SET_V_MASK_REG { .. } => 73,
        op::Opcode::C_LOOP_START { .. } => 74,
        op::Opcode::C_LOOP_END { .. } => 75,
        op::Opcode::C_BREAK => 76,
    }
}

/// Coarse engine lane. `_WO` (write-out) matrix variants stay in the
/// matrix lane because they're issued by the matrix unit even though
/// they end up writing to VRAM. `H_PREFETCH_V` and `H_STORE_V` share
/// the V-channel and are both binned under `PrefetchV`.
fn engine_kind_of(op: &op::Opcode) -> EngineKind {
    match op {
        op::Opcode::Invalid => EngineKind::Control,
        op::Opcode::M_MM { .. }
        | op::Opcode::M_TMM { .. }
        | op::Opcode::M_BMM { .. }
        | op::Opcode::M_BTMM { .. }
        | op::Opcode::M_BMM_WO { .. }
        | op::Opcode::M_MM_WO { .. }
        | op::Opcode::M_MV { .. }
        | op::Opcode::M_TMV { .. }
        | op::Opcode::M_BMV { .. }
        | op::Opcode::M_BTMV { .. }
        | op::Opcode::M_MV_WO { .. }
        | op::Opcode::M_BMV_WO { .. } => EngineKind::Matrix,
        op::Opcode::V_ADD_VV { .. }
        | op::Opcode::V_ADD_VF { .. }
        | op::Opcode::V_SUB_VV { .. }
        | op::Opcode::V_SUB_VF { .. }
        | op::Opcode::V_MUL_VV { .. }
        | op::Opcode::V_MUL_VF { .. }
        | op::Opcode::V_EXP_V { .. }
        | op::Opcode::V_RECI_V { .. }
        | op::Opcode::V_RED_SUM { .. }
        | op::Opcode::V_RED_MAX { .. }
        | op::Opcode::V_SHFT_V { .. } => EngineKind::Vector,
        op::Opcode::S_ADD_FP { .. }
        | op::Opcode::S_SUB_FP { .. }
        | op::Opcode::S_MAX_FP { .. }
        | op::Opcode::S_MUL_FP { .. }
        | op::Opcode::S_EXP_FP { .. }
        | op::Opcode::S_RECI_FP { .. }
        | op::Opcode::S_SQRT_FP { .. }
        | op::Opcode::S_LD_FP { .. }
        | op::Opcode::S_ST_FP { .. }
        | op::Opcode::S_MAP_V_FP { .. }
        | op::Opcode::S_ADD_INT { .. }
        | op::Opcode::S_ADDI_INT { .. }
        | op::Opcode::S_SUB_INT { .. }
        | op::Opcode::S_MUL_INT { .. }
        | op::Opcode::S_LUI_INT { .. }
        | op::Opcode::S_LD_INT { .. }
        | op::Opcode::S_ST_INT { .. } => EngineKind::Scalar,
        op::Opcode::H_PREFETCH_M { .. } => EngineKind::PrefetchM,
        op::Opcode::H_PREFETCH_V { .. } | op::Opcode::H_STORE_V { .. } => {
            EngineKind::PrefetchV
        }
        op::Opcode::C_SET_ADDR_REG { .. }
        | op::Opcode::C_SET_SCALE_REG { .. }
        | op::Opcode::C_SET_STRIDE_REG { .. }
        | op::Opcode::C_SET_V_MASK_REG { .. }
        | op::Opcode::C_LOOP_START { .. }
        | op::Opcode::C_LOOP_END { .. }
        | op::Opcode::C_BREAK => EngineKind::Control,
    }
}

/// `(op_tag, opcode_name, engine_kind)` table dumped into the sidecar JSON
/// so the Python renderer can resolve labels and lanes without a
/// schema-synced enum on its side.
const OP_TAG_TABLE: &[(u8, &str, EngineKind)] = &[
    (0, "Invalid", EngineKind::Control),
    (1, "M_MM", EngineKind::Matrix),
    (2, "M_TMM", EngineKind::Matrix),
    (3, "M_BMM", EngineKind::Matrix),
    (4, "M_BTMM", EngineKind::Matrix),
    (5, "M_BMM_WO", EngineKind::Matrix),
    (6, "M_MM_WO", EngineKind::Matrix),
    (7, "M_MV", EngineKind::Matrix),
    (8, "M_TMV", EngineKind::Matrix),
    (9, "M_BMV", EngineKind::Matrix),
    (10, "M_BTMV", EngineKind::Matrix),
    (11, "M_MV_WO", EngineKind::Matrix),
    (12, "M_BMV_WO", EngineKind::Matrix),
    (20, "V_ADD_VV", EngineKind::Vector),
    (21, "V_ADD_VF", EngineKind::Vector),
    (22, "V_SUB_VV", EngineKind::Vector),
    (23, "V_SUB_VF", EngineKind::Vector),
    (24, "V_MUL_VV", EngineKind::Vector),
    (25, "V_MUL_VF", EngineKind::Vector),
    (26, "V_EXP_V", EngineKind::Vector),
    (27, "V_RECI_V", EngineKind::Vector),
    (28, "V_RED_SUM", EngineKind::Vector),
    (29, "V_RED_MAX", EngineKind::Vector),
    (30, "V_SHFT_V", EngineKind::Vector),
    (40, "S_ADD_FP", EngineKind::Scalar),
    (41, "S_SUB_FP", EngineKind::Scalar),
    (42, "S_MAX_FP", EngineKind::Scalar),
    (43, "S_MUL_FP", EngineKind::Scalar),
    (44, "S_EXP_FP", EngineKind::Scalar),
    (45, "S_RECI_FP", EngineKind::Scalar),
    (46, "S_SQRT_FP", EngineKind::Scalar),
    (47, "S_LD_FP", EngineKind::Scalar),
    (48, "S_ST_FP", EngineKind::Scalar),
    (49, "S_MAP_V_FP", EngineKind::Scalar),
    (50, "S_ADD_INT", EngineKind::Scalar),
    (51, "S_ADDI_INT", EngineKind::Scalar),
    (52, "S_SUB_INT", EngineKind::Scalar),
    (53, "S_MUL_INT", EngineKind::Scalar),
    (54, "S_LUI_INT", EngineKind::Scalar),
    (55, "S_LD_INT", EngineKind::Scalar),
    (56, "S_ST_INT", EngineKind::Scalar),
    (60, "H_PREFETCH_M", EngineKind::PrefetchM),
    (61, "H_PREFETCH_V", EngineKind::PrefetchV),
    (62, "H_STORE_V", EngineKind::PrefetchV),
    (70, "C_SET_ADDR_REG", EngineKind::Control),
    (71, "C_SET_SCALE_REG", EngineKind::Control),
    (72, "C_SET_STRIDE_REG", EngineKind::Control),
    (73, "C_SET_V_MASK_REG", EngineKind::Control),
    (74, "C_LOOP_START", EngineKind::Control),
    (75, "C_LOOP_END", EngineKind::Control),
    (76, "C_BREAK", EngineKind::Control),
];

struct Accelerator {
    m_machine: MatrixMachine,
    v_machine: VectorMachine,
    hbm: Arc<dyn ErasedMemoryModel>,
    reg_file: AcceeleratorRegFile,
    intsram: Vec<u32>,
    fpsram: Vec<f16>,
    loop_stack: Vec<LoopInfo>, // Stack for nested loops
    /// When `Some(_)`, `do_ops` appends one entry per dispatched op.
    /// `execute_batch` is responsible for plumbing the buffer in before
    /// the run and draining it back into `EmulatorState.last_trace`
    /// after, so subsequent runs don't see stale entries.
    trace: Option<Vec<TraceEntry>>,
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
                    // Patch: scale span can cross multiple 64B chunks when
                    // scale_len_in_bytes_per_load > 64 - (scale_addr % 64).
                    // The original code issued only ONE read and capped the
                    // copy at 64B, silently dropping the rest of the scale
                    // stream — which corrupts MXFP dequantization on
                    // config_2 (HLEN=128 ⇒ ≥16 scale bytes per row).
                    if scale_len_in_bytes_per_load > 0 {
                        let total_to_load = scale_len_in_bytes_per_load as usize;
                        let within_chunk_offset0 = (scale_addr % 64) as usize;
                        let aligned_scale_addr_base = (scale_addr / 64) * 64;
                        let mut remaining = total_to_load;
                        let mut dst_pos = scale_byte_offset;
                        let mut chunk_i: u64 = 0;
                        while remaining > 0 {
                            let chunk_addr = aligned_scale_addr_base + chunk_i * 64;
                            let src_start = if chunk_i == 0 { within_chunk_offset0 } else { 0 };
                            let copy_len = std::cmp::min(64 - src_start, remaining);
                            let dst_pos_local = dst_pos;
                            futures.push(Box::pin(async move {
                                let data = hbm_clone.read(chunk_addr).await;
                                let mut selected = [0u8; 64];
                                selected[..copy_len]
                                    .copy_from_slice(&data[src_start..src_start + copy_len]);
                                ChunkType::Scale(dst_pos_local, selected, copy_len)
                            }));
                            remaining -= copy_len;
                            dst_pos += copy_len;
                            chunk_i += 1;
                        }
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

            // Trace collection: capture sim-time *before* dispatching this
            // op. Lazy — when trace is None we don't even read the clock,
            // so the overhead is one `is_some()` branch per op.
            let trace_active = self.trace.is_some();
            let op_start_picos = if trace_active {
                Executor::current().now().as_picos()
            } else {
                0
            };

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
                            self.reg_file.gp_reg[*rs1 as usize]
                                + self.reg_file.gp_reg[*rd as usize],
                            self.reg_file.gp_reg[*rs2 as usize],
                            self.reg_file.bmm_scale,
                        )
                        .await;
                }
                op::Opcode::M_BTMM { rs1, rs2, rd } => {
                    self.m_machine
                        .btmm(
                            self.reg_file.gp_reg[*rs1 as usize]
                                + self.reg_file.gp_reg[*rd as usize],
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
                op::Opcode::V_SHFT_V { rd, rs1, rs2 } => {
                    self.v_machine
                        .shift(
                            self.reg_file.gp_reg[*rd as usize],
                            self.reg_file.gp_reg[*rs1 as usize],
                            self.reg_file.gp_reg[*rs2 as usize],
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
            // Trace collection (append side). Looped ops show up as
            // repeated entries with distinct timestamps — that's exactly
            // what the Gantt needs.
            if trace_active {
                let op_end_picos = Executor::current().now().as_picos();
                let entry = TraceEntry {
                    start_picos: op_start_picos,
                    duration_picos: op_end_picos
                        .saturating_sub(op_start_picos)
                        .min(u32::MAX as u64) as u32,
                    engine: engine_kind_of(op) as u8,
                    op_tag: op_tag_of(op),
                    _pad: 0,
                    hbm_bytes_read: 0, // reserved — see TraceEntry doc
                    hbm_bytes_written: 0,
                };
                if let Some(buf) = self.trace.as_mut() {
                    if buf.len() < TRACE_MAX_ENTRIES {
                        buf.push(entry);
                    }
                }
            }

            if let Some(target_pc) = jump_pc {
                pc = target_pc;
            } else {
                pc += 1;
            }

            // Bump the global execution-progress counter to "furthest
            // instruction reached so far". `pc` is the **next** instruction
            // index (post-increment / post-jump); fetch_max keeps the
            // counter monotonic even when a C_LOOP_END jumps backward,
            // so the GUI progress bar never rewinds during a loop body.
            // Relaxed atomic on a single-writer counter is essentially a
            // plain RMW — negligible overhead per op.
            EXECUTION_PROGRESS_DONE.fetch_max(pc as u64, AtomicOrdering::Relaxed);
        }
    }
}

type ConcreteHbm =
    memory::WithStats<memory::WithTiming<ManuallyDrop<ramulator::Ramulator>, memory::MemoryBacked>>;

/// Shared hardware emulator state — exactly one instance per server.
/// Mutating commands take a write lock; pure queries take a read lock.
struct EmulatorState {
    executor: Executor,
    accelerator: Option<Accelerator>,
    hbm: Arc<ConcreteHbm>,
    /// When true, the next `execute_batch` / `execute_file` allocates a
    /// trace buffer on the accelerator and `do_ops` populates it.
    trace_enabled: bool,
    /// Most recently collected trace. Replaced (not appended) at the end
    /// of every executed batch.
    last_trace: Vec<TraceEntry>,
    /// Whether the most recent trace ran out of buffer space.
    last_trace_overflowed: bool,
    /// Sim time (picoseconds) at which the most recent trace started.
    last_trace_start_picos: u64,
}

/// Per-connection session record kept in the shared registry.
#[derive(Clone)]
struct ClientInfo {
    id: u64,
    peer_addr: String,
    connected_at_secs: u64,
    label: Option<String>,
    last_cmd: Option<String>,
    last_cmd_at_secs: Option<u64>,
    cmds_executed: u64,
    /// Sum of `executor.now()` deltas across every successful execute_*
    /// command this session ran. Lets the monitor GUI attribute simulator
    /// time to short-lived clients that connect, fire one kernel, and
    /// disconnect again before a manual refresh would catch them.
    sim_time_advanced_picos: u128,
    /// Number of execute_batch / execute_file commands attributed.
    execute_commands: u64,
    /// Module currently being executed, if the client used the
    /// start_module / end_module bracket protocol. Surfaced in
    /// `list_sessions` so the monitor GUI can show which Transformer
    /// module each session is on right now.
    current_module: Option<ModuleInProgress>,
    /// Ring buffer of recently-completed modules. Lets the GUI display
    /// "modules completed so far" for sessions that have already run
    /// many short kernels too fast for individual polls.
    module_log: VecDeque<ModuleEntry>,
}

const MODULE_LOG_CAPACITY: usize = 64;

#[derive(Clone)]
struct ModuleInProgress {
    name: String,
    started_at_secs: u64,
    /// Snapshot of `sim_time_advanced_picos` at start, so end_module can
    /// derive the sim-time delta for this module.
    sim_time_before_picos: u128,
    /// Snapshot of `execute_commands` at start.
    execute_count_before: u64,
}

#[derive(Clone)]
struct ModuleEntry {
    name: String,
    started_at_secs: u64,
    ended_at_secs: u64,
    sim_time_picos: u128,
    execute_count: u64,
    ok: bool,
    error: Option<String>,
}

/// Snapshot of a session pushed to the closed-sessions ring buffer when
/// the client disconnects. Keeps the GUI's view useful for runs that
/// complete in milliseconds.
#[derive(Clone)]
struct ClosedSession {
    info: ClientInfo,
    closed_at_secs: u64,
}

const CLOSED_SESSIONS_CAPACITY: usize = 100;

/// Granularity of the execute_* progress counter. The server processes
/// instruction batches in chunks of this many opcodes; after each chunk
/// it bumps `EXECUTION_PROGRESS_DONE` so a concurrently-polled client can
/// observe progress without holding any state lock. Smaller chunks =
/// smoother progress bar; larger chunks = lower bookkeeping overhead.
/// 64 is a balance — for a JIT block of a few thousand instructions
/// the GUI sees ~50 progress updates.
const EXECUTION_CHUNK_SIZE: usize = 64;

/// Single-writer progress telemetry for the currently-running execute_*
/// command. Only one execute_* runs at a time (it holds the state write
/// lock), so a flat set of atomics with relaxed ordering is enough — no
/// need for a Mutex. Readers see a self-consistent snapshot per field;
/// rare torn reads across fields just make the GUI estimate stale for one
/// poll, which is harmless.
static EXECUTION_PROGRESS_TOTAL: AtomicU64 = AtomicU64::new(0);
static EXECUTION_PROGRESS_DONE: AtomicU64 = AtomicU64::new(0);
static EXECUTION_PROGRESS_SESSION_ID: AtomicU64 = AtomicU64::new(0);
static EXECUTION_PROGRESS_STARTED_AT_SECS: AtomicU64 = AtomicU64::new(0);
/// Snapshot of `executor.now()` when the current batch started, used by
/// the GUI to display "X ms of simulated time elapsed so far".
static EXECUTION_PROGRESS_SIM_TIME_BEFORE: AtomicU64 = AtomicU64::new(0);

fn progress_begin(session_id: u64, total: usize, sim_time_before_picos: u64) {
    EXECUTION_PROGRESS_SESSION_ID.store(session_id, AtomicOrdering::Relaxed);
    EXECUTION_PROGRESS_STARTED_AT_SECS.store(now_unix_secs(), AtomicOrdering::Relaxed);
    EXECUTION_PROGRESS_TOTAL.store(total as u64, AtomicOrdering::Relaxed);
    EXECUTION_PROGRESS_DONE.store(0, AtomicOrdering::Relaxed);
    EXECUTION_PROGRESS_SIM_TIME_BEFORE.store(sim_time_before_picos, AtomicOrdering::Relaxed);
}

fn progress_advance(done: usize) {
    EXECUTION_PROGRESS_DONE.store(done as u64, AtomicOrdering::Relaxed);
}

fn progress_end() {
    EXECUTION_PROGRESS_TOTAL.store(0, AtomicOrdering::Relaxed);
    EXECUTION_PROGRESS_DONE.store(0, AtomicOrdering::Relaxed);
    EXECUTION_PROGRESS_SESSION_ID.store(0, AtomicOrdering::Relaxed);
    EXECUTION_PROGRESS_STARTED_AT_SECS.store(0, AtomicOrdering::Relaxed);
    EXECUTION_PROGRESS_SIM_TIME_BEFORE.store(0, AtomicOrdering::Relaxed);
}

/// Single entry in the global execution history ring buffer.
#[derive(Clone)]
struct HistoryEntry {
    session_id: u64,
    cmd: String,
    at_secs: u64,
    ok: bool,
    detail: Option<String>,
}

const HISTORY_CAPACITY: usize = 256;

/// Server-wide shared bundle: hardware state + session registry + history.
struct ServerCtx {
    state: RwLock<EmulatorState>,
    sessions: std::sync::Mutex<HashMap<u64, ClientInfo>>,
    closed_sessions: std::sync::Mutex<VecDeque<ClosedSession>>,
    history: std::sync::Mutex<VecDeque<HistoryEntry>>,
    next_session_id: AtomicU64,
    started_at_secs: u64,
    shutdown_tx: broadcast::Sender<()>,
}

struct ServiceOutcome {
    response: Value,
    shutdown: bool,
    /// Picoseconds that `executor.now()` advanced while handling this
    /// request (non-zero only for successful execute_* commands).
    sim_time_advanced_picos: u128,
}

impl ServiceOutcome {
    fn plain(response: Value) -> Self {
        Self {
            response,
            shutdown: false,
            sim_time_advanced_picos: 0,
        }
    }
    fn shutting_down(response: Value) -> Self {
        Self {
            response,
            shutdown: true,
            sim_time_advanced_picos: 0,
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum ServiceRequest {
    Ping,
    Reset,
    GetConfig,
    GetState,
    ExecuteBatch {
        opcodes: Vec<String>,
    },
    ExecuteFile {
        path: String,
    },
    LoadHbmFile {
        path: String,
    },
    LoadFpSramFile {
        path: String,
    },
    LoadIntSramFile {
        path: String,
    },
    LoadVramFile {
        path: String,
    },
    ReadVram {
        addr: u32,
    },
    ReadMram {
        addr: u32,
    },
    ReadHbm {
        addr: u64,
        len: usize,
    },
    /// Returns this connection's own session id (and metadata).
    WhoAmI,
    /// Lists every active session connected to the server.
    ListSessions,
    /// Lists recently-closed sessions (newest first). Useful when
    /// short-lived simulations finish before the monitor polls.
    ListClosedSessions {
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Tags the current session with a human-readable label.
    SetLabel {
        label: String,
    },
    /// Returns up to `limit` most-recent history entries (newest first).
    GetHistory {
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Read-only summary used by the monitoring GUI (no heavy memory reads).
    GetServerStatus,
    /// Returns the progress of the currently-running execute_* command,
    /// or `running=false` if nothing is executing. Cheap: reads atomics,
    /// touches no state lock — safe to poll while a heavy kernel runs.
    GetExecutionProgress,
    /// Mark the start of a logical "module" — a named scope around one
    /// or more execute_* calls. Used by the TileLang adapter so the
    /// monitor can show "currently running q_proj_linear" rather than
    /// just an opaque opcode count.
    StartModule {
        name: String,
    },
    /// Finalize the current module, push it to the per-session module
    /// log with the elapsed wall + sim time.
    EndModule {
        #[serde(default = "default_true")]
        ok: bool,
        #[serde(default)]
        error: Option<String>,
    },
    /// Returns this session's recently-completed modules.
    ListSessionModules {
        #[serde(default)]
        id: Option<u64>,
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Toggle per-op trace collection. When `enabled=true`, the next
    /// `execute_batch` / `execute_file` allocates a `TraceEntry` buffer
    /// and `do_ops` records one entry per dispatched opcode. Off by
    /// default for zero overhead; stays on across multiple executes
    /// until disabled.
    EnableTrace {
        #[serde(default = "default_true")]
        enabled: bool,
    },
    /// Write the most-recently-collected trace to `path` as a packed
    /// stream of 24-byte little-endian `TraceEntry` records. A sidecar
    /// `<path>.meta.json` is also written with the op_tag / engine
    /// mapping + run metadata. Returns entry count + overflow flag.
    DumpTrace {
        path: String,
    },
    Shutdown,
}

fn default_true() -> bool {
    true
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn cmd_name(req: &ServiceRequest) -> &'static str {
    match req {
        ServiceRequest::Ping => "ping",
        ServiceRequest::Reset => "reset",
        ServiceRequest::GetConfig => "get_config",
        ServiceRequest::GetState => "get_state",
        ServiceRequest::ExecuteBatch { .. } => "execute_batch",
        ServiceRequest::ExecuteFile { .. } => "execute_file",
        ServiceRequest::LoadHbmFile { .. } => "load_hbm_file",
        ServiceRequest::LoadFpSramFile { .. } => "load_fp_sram_file",
        ServiceRequest::LoadIntSramFile { .. } => "load_int_sram_file",
        ServiceRequest::LoadVramFile { .. } => "load_vram_file",
        ServiceRequest::ReadVram { .. } => "read_vram",
        ServiceRequest::ReadMram { .. } => "read_mram",
        ServiceRequest::ReadHbm { .. } => "read_hbm",
        ServiceRequest::WhoAmI => "who_am_i",
        ServiceRequest::ListSessions => "list_sessions",
        ServiceRequest::ListClosedSessions { .. } => "list_closed_sessions",
        ServiceRequest::SetLabel { .. } => "set_label",
        ServiceRequest::GetHistory { .. } => "get_history",
        ServiceRequest::GetServerStatus => "get_server_status",
        ServiceRequest::GetExecutionProgress => "get_execution_progress",
        ServiceRequest::StartModule { .. } => "start_module",
        ServiceRequest::EndModule { .. } => "end_module",
        ServiceRequest::ListSessionModules { .. } => "list_session_modules",
        ServiceRequest::EnableTrace { .. } => "enable_trace",
        ServiceRequest::DumpTrace { .. } => "dump_trace",
        ServiceRequest::Shutdown => "shutdown",
    }
}

/// Whether a command needs an exclusive (write) lock on emulator state.
fn cmd_needs_write_lock(req: &ServiceRequest) -> bool {
    match req {
        ServiceRequest::Reset
        | ServiceRequest::ExecuteBatch { .. }
        | ServiceRequest::ExecuteFile { .. }
        | ServiceRequest::LoadHbmFile { .. }
        | ServiceRequest::LoadFpSramFile { .. }
        | ServiceRequest::LoadIntSramFile { .. }
        | ServiceRequest::LoadVramFile { .. }
        | ServiceRequest::EnableTrace { .. }
        | ServiceRequest::DumpTrace { .. }
        | ServiceRequest::Shutdown => true,
        _ => false,
    }
}

fn build_accelerator() -> (Accelerator, Arc<ConcreteHbm>) {
    let mram = Arc::new(MatrixSram::new(*MLEN, *MATRIX_SRAM_SIZE, *MATRIX_SRAM_TYPE));
    let vram = Arc::new(VectorSram::from_mx_type(
        *VLEN,
        *VECTOR_SRAM_SIZE,
        *VECTOR_SRAM_TYPE,
    ));

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
    };

    let hbm = Arc::new(memory::WithStats::new(memory::WithTiming::new(
        ManuallyDrop::new(ramulator::Ramulator::hbm2_preset(8).unwrap()),
        memory::MemoryBacked::with_capacity(*HBM_SIZE),
    )));

    let accelerator = Accelerator {
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
        intsram: vec![0; *INT_SRAM_SIZE],
        fpsram: vec![f16::ZERO; *FP_SRAM_SIZE],
        loop_stack: Vec::new(),
        trace: None,
    };

    (accelerator, hbm)
}

fn tensor_to_f32_vec(tensor: &Tensor) -> Vec<f32> {
    let len = tensor.numel();
    let slice = unsafe { core::slice::from_raw_parts(tensor.data_ptr() as *const f32, len) };
    slice.to_vec()
}

fn parse_opcode_token(token: &str) -> anyhow::Result<u32> {
    let trimmed = token.trim();
    if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        Ok(u32::from_str_radix(hex, 16)?)
    } else {
        Ok(trimmed.parse()?)
    }
}

fn load_opcode_file(path: &Path) -> anyhow::Result<Vec<u32>> {
    let op_file = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read opcode file {}", path.display()))?;
    op_file
        .split_whitespace()
        .map(parse_opcode_token)
        .collect::<anyhow::Result<Vec<_>>>()
}

fn decode_ops(opcodes: &[u32]) -> Vec<op::Opcode> {
    opcodes.iter().copied().map(op::Opcode::decode).collect()
}

fn bytes_to_f16_vec(bytes: &[u8]) -> anyhow::Result<Vec<f16>> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<u16>()) {
        bail!("FP SRAM file size must be a multiple of 2 bytes");
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn bytes_to_u32_vec(bytes: &[u8]) -> anyhow::Result<Vec<u32>> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<u32>()) {
        bail!("INT SRAM file size must be a multiple of 4 bytes");
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else if let Some(msg) = payload.downcast_ref::<&str>() {
        (*msg).to_string()
    } else {
        "unknown panic".to_string()
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

impl EmulatorState {
    fn new() -> Self {
        let (accelerator, hbm) = build_accelerator();
        Self {
            executor: Executor::new(),
            accelerator: Some(accelerator),
            hbm,
            trace_enabled: false,
            last_trace: Vec::new(),
            last_trace_overflowed: false,
            last_trace_start_picos: 0,
        }
    }

    fn accelerator(&self) -> &Accelerator {
        self.accelerator
            .as_ref()
            .expect("accelerator must be available")
    }

    fn accelerator_mut(&mut self) -> &mut Accelerator {
        self.accelerator
            .as_mut()
            .expect("accelerator must be available")
    }

    fn reset(&mut self) {
        let (accelerator, hbm) = build_accelerator();
        self.executor = Executor::new();
        self.accelerator = Some(accelerator);
        self.hbm = hbm;
    }

    async fn preload_from_opts(&mut self, opts: &Opts) -> anyhow::Result<()> {
        if let Some(path) = opts.hbm.as_deref() {
            self.load_hbm_file(path)?;
        }
        if let Some(path) = opts.fpsram.as_deref() {
            self.load_fpsram_file(path)?;
        }
        if let Some(path) = opts.intsram.as_deref() {
            self.load_intsram_file(path)?;
        }
        if let Some(path) = opts.vram.as_deref() {
            self.load_vram_file(path).await?;
        }
        Ok(())
    }

    fn load_hbm_file(&mut self, path: &Path) -> anyhow::Result<usize> {
        let hbm_data = std::fs::read(path)
            .with_context(|| format!("failed to read HBM file {}", path.display()))?;
        if hbm_data.len() > *HBM_SIZE {
            bail!(
                "HBM preload is too large: {} bytes > configured HBM size {} bytes",
                hbm_data.len(),
                *HBM_SIZE
            );
        }
        self.hbm.model().data().with_data(|f| {
            f.fill(0);
            f[..hbm_data.len()].copy_from_slice(&hbm_data);
        });
        Ok(hbm_data.len())
    }

    fn load_fpsram_file(&mut self, path: &Path) -> anyhow::Result<usize> {
        let fpsram_data = std::fs::read(path)
            .with_context(|| format!("failed to read FP SRAM file {}", path.display()))?;
        let fp_vals = bytes_to_f16_vec(&fpsram_data)?;
        let accel = self.accelerator_mut();
        if fp_vals.len() > accel.fpsram.len() {
            bail!(
                "FP SRAM preload is too large: {} values > capacity {}",
                fp_vals.len(),
                accel.fpsram.len()
            );
        }
        accel.fpsram.fill(f16::ZERO);
        accel.fpsram[..fp_vals.len()].copy_from_slice(&fp_vals);
        Ok(fp_vals.len())
    }

    fn load_intsram_file(&mut self, path: &Path) -> anyhow::Result<usize> {
        let intsram_data = std::fs::read(path)
            .with_context(|| format!("failed to read INT SRAM file {}", path.display()))?;
        let int_vals = bytes_to_u32_vec(&intsram_data)?;
        let accel = self.accelerator_mut();
        if int_vals.len() > accel.intsram.len() {
            bail!(
                "INT SRAM preload is too large: {} values > capacity {}",
                int_vals.len(),
                accel.intsram.len()
            );
        }
        accel.intsram.fill(0);
        accel.intsram[..int_vals.len()].copy_from_slice(&int_vals);
        Ok(int_vals.len())
    }

    async fn load_vram_file(&mut self, path: &Path) -> anyhow::Result<usize> {
        let vram_data = std::fs::read(path)
            .with_context(|| format!("failed to read VRAM file {}", path.display()))?;
        self.accelerator()
            .v_machine
            .vram
            .load_from_bytes(&vram_data)
            .await;
        Ok(vram_data.len())
    }

    async fn execute_file(&mut self, session_id: u64, path: &Path) -> anyhow::Result<usize> {
        let opcodes = load_opcode_file(path)?;
        self.execute_batch(session_id, &opcodes).await
    }

    /// Run a batch of opcodes, exposing per-instruction progress via the
    /// global `EXECUTION_PROGRESS_*` atomics so a concurrently-polled
    /// client can observe completion without contending for the state lock.
    ///
    /// IMPORTANT: the full op slice is passed to `do_ops` in a SINGLE call.
    /// `do_ops` maintains internal pc / loop_stack that are scoped to the
    /// slice it receives, so splitting the call into chunks would corrupt
    /// any branch or loop spanning a chunk boundary. Per-instruction
    /// progress is bumped from inside `do_ops` itself.
    async fn execute_batch(&mut self, session_id: u64, opcodes: &[u32]) -> anyhow::Result<usize> {
        let total = opcodes.len();
        let decoded_ops = decode_ops(opcodes);
        let mut accelerator = self
            .accelerator
            .take()
            .ok_or_else(|| anyhow!("accelerator is busy"))?;

        // If trace collection is on, hand the accelerator a fresh buffer.
        // Capacity is just a hint; we grow until the global TRACE_MAX
        // cap, after which `do_ops` silently stops appending and we set
        // `last_trace_overflowed` when draining.
        let trace_enabled = self.trace_enabled;
        let trace_start_picos = self.executor.now().as_picos();
        if trace_enabled {
            accelerator.trace = Some(Vec::with_capacity(total.min(1 << 16)));
        }

        let (sender, receiver) =
            oneshot::channel::<Result<Accelerator, Box<dyn Any + Send + 'static>>>();

        progress_begin(session_id, total, self.executor.now().as_picos());

        self.executor.spawn(async move {
            let result = std::panic::AssertUnwindSafe(async move {
                let mut accelerator = accelerator;
                accelerator.do_ops(&decoded_ops).await;
                accelerator
            })
            .catch_unwind()
            .await;
            let _ = sender.send(result);
        });

        self.executor.enter(Instant::ETERNITY).await;

        let outcome = receiver.await.context("simulation worker dropped")?;
        // Clear progress before returning so a follow-up read doesn't see
        // a stale 100%-but-still-running snapshot.
        progress_end();
        match outcome {
            Ok(mut accelerator) => {
                // Drain trace into EmulatorState before parking the
                // accelerator back into its slot, so the next execute
                // can overwrite the buffer without losing this run's.
                if let Some(buf) = accelerator.trace.take() {
                    let overflowed = buf.len() >= TRACE_MAX_ENTRIES;
                    self.last_trace = buf;
                    self.last_trace_overflowed = overflowed;
                    self.last_trace_start_picos = trace_start_picos;
                }
                self.accelerator = Some(accelerator);
                Ok(opcodes.len())
            }
            Err(payload) => {
                let message = panic_message(payload);
                self.reset();
                bail!("emulator panicked during execution: {message}");
            }
        }
    }

    fn config_json(&self) -> Value {
        json!({
            "mlen": *MLEN,
            "vlen": *VLEN,
            "blen": *BLEN,
            "hlen": *HLEN,
            "broadcast_amount": *BROADCAST_AMOUNT,
            "hbm_size_bytes": *HBM_SIZE,
            "matrix_sram_size": *MATRIX_SRAM_SIZE,
            "vector_sram_size": *VECTOR_SRAM_SIZE,
            "fp_sram_size": *FP_SRAM_SIZE,
            "int_sram_size": *INT_SRAM_SIZE,
            "matrix_sram_type": format!("{:?}", *MATRIX_SRAM_TYPE),
            "vector_sram_type": format!("{:?}", *VECTOR_SRAM_TYPE),
            "matrix_weight_type": format!("{:?}", *MATRIX_WEIGHT_TYPE),
            "matrix_kv_type": format!("{:?}", *MATRIX_KV_TYPE),
            "vector_activation_type": format!("{:?}", *VECTOR_ACTIVATION_TYPE),
            "vector_kv_type": format!("{:?}", *VECTOR_KV_TYPE),
        })
    }

    fn stats_json(&self) -> Value {
        let memory_stats = self.hbm.statistics();
        let sim_time = self.executor.now();
        let sim_time_secs = sim_time.to_secs();
        let utilization = if sim_time_secs > 0.0 {
            (memory_stats.total_bytes_read + memory_stats.total_bytes_written) as f64
                / sim_time_secs
        } else {
            0.0
        };
        json!({
            "sim_time_picos": sim_time.as_picos(),
            "sim_time_debug": format!("{sim_time:?}"),
            "hbm_bytes_read": memory_stats.total_bytes_read,
            "hbm_bytes_written": memory_stats.total_bytes_written,
            "hbm_utilization_bytes_per_sec": utilization,
        })
    }

    fn state_json(&self) -> Value {
        let accelerator = self.accelerator();
        let loop_stack = accelerator
            .loop_stack
            .iter()
            .map(|loop_info| {
                json!({
                    "start_pc": loop_info.start_pc,
                    "iteration_count": loop_info.iteration_count,
                    "current_iteration": loop_info.current_iteration,
                    "instruction_count": loop_info.instruction_count,
                    "loop_reg": loop_info.loop_reg,
                })
            })
            .collect::<Vec<_>>();

        json!({
            "gp_reg": accelerator.reg_file.gp_reg,
            "fp_reg": accelerator
                .reg_file
                .fp_reg
                .iter()
                .map(|value| f32::from(*value))
                .collect::<Vec<_>>(),
            "hbm_addr_reg": accelerator.reg_file.hbm_addr_reg,
            "scale": accelerator.reg_file.scale,
            "stride": accelerator.reg_file.stride,
            "bmm_scale": accelerator.reg_file.bmm_scale,
            "v_mask": accelerator.reg_file.v_mask,
            "loop_stack": loop_stack,
            "stats": self.stats_json(),
        })
    }

    async fn read_vram_json(&self, addr: u32) -> anyhow::Result<Value> {
        let tensor = self.accelerator().v_machine.vram.read(addr).await;
        Ok(json!({
            "addr": addr,
            "data_type": format!("{:?}", tensor.data_type()),
            "values": tensor_to_f32_vec(tensor.as_tensor()),
        }))
    }

    async fn read_mram_json(&self, addr: u32) -> anyhow::Result<Value> {
        let tensor = self.accelerator().m_machine.mram.read(addr).await;
        Ok(json!({
            "addr": addr,
            "data_type": format!("{:?}", tensor.data_type()),
            "values": tensor_to_f32_vec(tensor.as_tensor()),
        }))
    }

    fn read_hbm_json(&self, addr: u64, len: usize) -> anyhow::Result<Value> {
        let start = usize::try_from(addr).context("HBM read address does not fit in usize")?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| anyhow!("HBM read range overflow"))?;
        if end > *HBM_SIZE {
            bail!(
                "HBM read out of bounds: [{start}, {end}) exceeds configured size {}",
                *HBM_SIZE
            );
        }

        let mut bytes = Vec::with_capacity(len);
        self.hbm.model().data().with_data(|f| {
            bytes.extend_from_slice(&f[start..end]);
        });

        Ok(json!({
            "addr": addr,
            "len": len,
            "hex": encode_hex(&bytes),
        }))
    }

    async fn print_batch_summary(&self) {
        let accelerator = self.accelerator();
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
    }

    async fn dump_batch_files(&self) -> anyhow::Result<()> {
        let accelerator = self.accelerator();

        let mram_dump_path = "mram_dump.bin";
        let mram_bytes = accelerator.m_machine.mram.as_bytes().await;
        let mut mram_file = std::fs::File::create(mram_dump_path)?;
        mram_file.write_all(&mram_bytes)?;
        if !is_quiet() {
            eprintln!("Dumped MRAM content to: {:?}", mram_dump_path);
        }

        let vram_dump_path = "vram_dump.bin";
        let vram_bytes = accelerator.v_machine.vram.as_bytes().await;
        let mut vram_file = std::fs::File::create(vram_dump_path)?;
        vram_file.write_all(&vram_bytes)?;
        if !is_quiet() {
            eprintln!("Dumped VRAM content to: {:?}", vram_dump_path);
        }

        let fpsram_dump_path = "fpsram_dump.bin";
        let fpsram_bytes: Vec<u8> = accelerator
            .fpsram
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let mut fpsram_file = std::fs::File::create(fpsram_dump_path)?;
        fpsram_file.write_all(&fpsram_bytes)?;
        if !is_quiet() {
            eprintln!("Dumped FPSRAM content to: {:?}", fpsram_dump_path);
        }

        let hbm_dump_path = "hbm_dump.bin";
        let mut hbm_bytes = vec![0u8; *HBM_SIZE];
        self.hbm.model().data().with_data(|f| {
            let len = std::cmp::min(*HBM_SIZE, f.len());
            hbm_bytes[..len].copy_from_slice(&f[..len]);
        });
        let mut hbm_file = std::fs::File::create(hbm_dump_path)?;
        hbm_file.write_all(&hbm_bytes)?;
        Ok(())
    }

    /// Handle a write-locked command (caller already holds `RwLock::write()`).
    async fn handle_write_request(
        &mut self,
        session_id: u64,
        request: ServiceRequest,
    ) -> anyhow::Result<ServiceOutcome> {
        let outcome = match request {
            ServiceRequest::Reset => {
                self.reset();
                ServiceOutcome::plain(json!({"message": "emulator state reset"}))
            }
            ServiceRequest::ExecuteBatch { opcodes } => {
                let opcodes = opcodes
                    .iter()
                    .map(|opcode| parse_opcode_token(opcode))
                    .collect::<anyhow::Result<Vec<_>>>()?;
                let before = self.executor.now().as_picos();
                let executed = self.execute_batch(session_id, &opcodes).await?;
                let after = self.executor.now().as_picos();
                let delta = after.saturating_sub(before);
                ServiceOutcome {
                    response: json!({
                        "executed": executed,
                        "sim_time_advanced_picos": delta,
                        "stats": self.stats_json(),
                    }),
                    shutdown: false,
                    sim_time_advanced_picos: u128::from(delta),
                }
            }
            ServiceRequest::ExecuteFile { path } => {
                let before = self.executor.now().as_picos();
                let executed = self.execute_file(session_id, Path::new(&path)).await?;
                let after = self.executor.now().as_picos();
                let delta = after.saturating_sub(before);
                ServiceOutcome {
                    response: json!({
                        "executed": executed,
                        "path": path,
                        "sim_time_advanced_picos": delta,
                        "stats": self.stats_json(),
                    }),
                    shutdown: false,
                    sim_time_advanced_picos: u128::from(delta),
                }
            }
            ServiceRequest::LoadHbmFile { path } => {
                let bytes = self.load_hbm_file(Path::new(&path))?;
                ServiceOutcome::plain(json!({"loaded_bytes": bytes, "path": path}))
            }
            ServiceRequest::LoadFpSramFile { path } => {
                let values = self.load_fpsram_file(Path::new(&path))?;
                ServiceOutcome::plain(json!({"loaded_values": values, "path": path}))
            }
            ServiceRequest::LoadIntSramFile { path } => {
                let values = self.load_intsram_file(Path::new(&path))?;
                ServiceOutcome::plain(json!({"loaded_values": values, "path": path}))
            }
            ServiceRequest::LoadVramFile { path } => {
                let bytes = self.load_vram_file(Path::new(&path)).await?;
                ServiceOutcome::plain(json!({"loaded_bytes": bytes, "path": path}))
            }
            ServiceRequest::EnableTrace { enabled } => {
                let prev = self.trace_enabled;
                self.trace_enabled = enabled;
                ServiceOutcome::plain(json!({
                    "enabled": enabled,
                    "previously_enabled": prev,
                    "last_trace_entries": self.last_trace.len(),
                }))
            }
            ServiceRequest::DumpTrace { path } => {
                let response = self.dump_trace_to_path(Path::new(&path))?;
                ServiceOutcome::plain(response)
            }
            ServiceRequest::Shutdown => {
                ServiceOutcome::shutting_down(json!({"message": "server shutting down"}))
            }
            other => bail!(
                "internal error: {} routed to write handler",
                cmd_name(&other)
            ),
        };

        Ok(outcome)
    }

    /// Serialise `self.last_trace` to `path` as packed 24-byte little-
    /// endian `TraceEntry` records, plus a sidecar `<path>.meta.json`
    /// with the engine + op_tag table and run metadata. The Python
    /// renderer (`tools/render_trace.py`) reads both.
    fn dump_trace_to_path(&self, path: &Path) -> anyhow::Result<Value> {
        use std::io::Write as _;

        let entries = &self.last_trace;
        // Safety: TraceEntry is repr(C) + Copy + size_of == 24 (asserted
        // at compile time). The only padding is the explicit `_pad: u16`,
        // so we can reinterpret the slice as raw bytes.
        let bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                entries.as_ptr() as *const u8,
                entries.len() * core::mem::size_of::<TraceEntry>(),
            )
        };

        let mut f = std::fs::File::create(path)
            .with_context(|| format!("failed to create trace file {}", path.display()))?;
        f.write_all(bytes)
            .with_context(|| format!("failed to write trace to {}", path.display()))?;

        // Sidecar metadata next to the binary.
        let meta_path = {
            let new_name = format!(
                "{}.meta.json",
                path.file_name().and_then(|s| s.to_str()).unwrap_or("trace")
            );
            let mut p = path.to_path_buf();
            p.set_file_name(new_name);
            p
        };
        let engine_names = ["Matrix", "Vector", "Scalar", "PrefetchM", "PrefetchV", "Control"];
        let engines: Vec<Value> = engine_names
            .iter()
            .enumerate()
            .map(|(i, n)| json!({"id": i, "name": n}))
            .collect();
        let op_tags: Vec<Value> = OP_TAG_TABLE
            .iter()
            .map(|(tag, name, engine)| {
                json!({
                    "tag": tag,
                    "name": name,
                    "engine_id": *engine as u8,
                })
            })
            .collect();
        let total_sim_picos = entries
            .last()
            .map(|e| e.start_picos.saturating_add(u64::from(e.duration_picos)))
            .unwrap_or(0)
            .saturating_sub(self.last_trace_start_picos);
        let meta = json!({
            "version": 1,
            "entry_struct_bytes": core::mem::size_of::<TraceEntry>(),
            "entry_count": entries.len(),
            "overflowed": self.last_trace_overflowed,
            "overflow_cap": TRACE_MAX_ENTRIES,
            "trace_start_picos": self.last_trace_start_picos,
            "total_sim_picos": total_sim_picos,
            "engines": engines,
            "op_tags": op_tags,
            "hardware": {
                "mlen": *MLEN,
                "vlen": *VLEN,
                "blen": *BLEN,
                "hlen": *HLEN,
            },
        });
        std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)
            .with_context(|| {
                format!("failed to write sidecar metadata to {}", meta_path.display())
            })?;

        Ok(json!({
            "path": path.display().to_string(),
            "meta_path": meta_path.display().to_string(),
            "entry_count": entries.len(),
            "bytes_written": bytes.len(),
            "overflowed": self.last_trace_overflowed,
        }))
    }

    /// Handle a read-locked command (caller holds `RwLock::read()`).
    async fn handle_read_request(&self, request: ServiceRequest) -> anyhow::Result<ServiceOutcome> {
        let response = match request {
            ServiceRequest::Ping => json!({"message": "pong"}),
            ServiceRequest::GetConfig => self.config_json(),
            ServiceRequest::GetState => self.state_json(),
            ServiceRequest::ReadVram { addr } => self.read_vram_json(addr).await?,
            ServiceRequest::ReadMram { addr } => self.read_mram_json(addr).await?,
            ServiceRequest::ReadHbm { addr, len } => self.read_hbm_json(addr, len)?,
            other => bail!(
                "internal error: {} routed to read handler",
                cmd_name(&other)
            ),
        };
        Ok(ServiceOutcome::plain(response))
    }
}

#[derive(Parser)]
struct Opts {
    #[arg(long)]
    /// Run the emulator as an online TCP service instead of a one-shot batch job.
    serve: bool,

    #[arg(long, conflicts_with = "serve")]
    /// Run as a gateway: accept client connections on `--bind`, spawn an
    /// isolated `--serve` backend subprocess per session, proxy commands.
    /// Backends are reaped automatically on client disconnect. Each backend
    /// has its own ~1 GB HBM (see `plena_settings.toml`) — size accordingly.
    gateway: bool,

    #[arg(long, default_value = "127.0.0.1:7878")]
    /// TCP bind address for service / gateway mode.
    bind: String,

    #[arg(long, default_value_t = 40000)]
    /// First TCP port the gateway hands out to spawned backends. The
    /// gateway always asks the kernel for an ephemeral free port via
    /// bind(":0"), so this is only a starting hint kept for clarity.
    gateway_port_base: u16,

    #[arg(long)]
    /// Path to Instruction to be executed.
    opcode: Option<PathBuf>,

    #[arg(long)]
    /// Path to HBM contents for preloading.
    hbm: Option<PathBuf>,

    #[arg(long)]
    /// Path to FP SRAM contents for preloading.
    fpsram: Option<PathBuf>,

    #[arg(long)]
    /// Path to INT SRAM contents for preloading.
    intsram: Option<PathBuf>,

    #[arg(long)]
    /// Path to file storing Vector SRAM contents (optional).
    vram: Option<PathBuf>,

    #[arg(long, short)]
    /// Quiet mode: only output final latency and statistics.
    quiet: bool,

    #[arg(long)]
    /// Verbose mode: re-enable per-instruction and per-op debug prints.
    /// Has no effect unless `--serve` is set — in batch mode the existing
    /// `--quiet` flag is the only knob. Under `--serve` we default to
    /// quiet because the per-instruction println! dominates simulation
    /// time for any non-trivial workload.
    verbose: bool,
}

static QUIET_MODE: LazyLock<std::sync::atomic::AtomicBool> =
    LazyLock::new(|| std::sync::atomic::AtomicBool::new(false));

fn is_quiet() -> bool {
    QUIET_MODE.load(std::sync::atomic::Ordering::Relaxed)
}

fn require_batch_path<'a>(name: &str, path: &'a Option<PathBuf>) -> anyhow::Result<&'a Path> {
    path.as_deref()
        .ok_or_else(|| anyhow!("--{name} is required in batch mode"))
}

fn ok_response(data: Value) -> Value {
    json!({ "ok": true, "data": data })
}

fn error_response(message: impl Into<String>) -> Value {
    json!({ "ok": false, "error": message.into() })
}

async fn write_json_line<W: AsyncWrite + Unpin>(
    writer: &mut W,
    value: &Value,
) -> anyhow::Result<()> {
    writer.write_all(value.to_string().as_bytes()).await?;
    writer.write_all(b"\n").await?;
    Ok(())
}

fn module_entry_to_json(m: &ModuleEntry) -> Value {
    json!({
        "name": m.name,
        "started_at_secs": m.started_at_secs,
        "ended_at_secs": m.ended_at_secs,
        "duration_secs": m.ended_at_secs.saturating_sub(m.started_at_secs),
        "sim_time_picos": m.sim_time_picos as u64,
        "execute_count": m.execute_count,
        "ok": m.ok,
        "error": m.error,
    })
}

fn session_to_json(info: &ClientInfo, self_id: u64, now_secs: u64) -> Value {
    let current_module = info.current_module.as_ref().map(|m| {
        json!({
            "name": m.name,
            "started_at_secs": m.started_at_secs,
            "elapsed_secs": now_secs.saturating_sub(m.started_at_secs),
        })
    });
    json!({
        "id": info.id,
        "peer_addr": info.peer_addr,
        "label": info.label,
        "connected_at_secs": info.connected_at_secs,
        "uptime_secs": now_secs.saturating_sub(info.connected_at_secs),
        "last_cmd": info.last_cmd,
        "last_cmd_at_secs": info.last_cmd_at_secs,
        "cmds_executed": info.cmds_executed,
        "execute_commands": info.execute_commands,
        "sim_time_advanced_picos": info.sim_time_advanced_picos as u64,
        "sim_time_advanced_picos_str": info.sim_time_advanced_picos.to_string(),
        "is_self": info.id == self_id,
        "current_module": current_module,
        "modules_completed": info.module_log.len(),
    })
}

fn closed_session_to_json(closed: &ClosedSession) -> Value {
    let info = &closed.info;
    json!({
        "id": info.id,
        "peer_addr": info.peer_addr,
        "label": info.label,
        "connected_at_secs": info.connected_at_secs,
        "closed_at_secs": closed.closed_at_secs,
        "lifetime_secs": closed.closed_at_secs.saturating_sub(info.connected_at_secs),
        "last_cmd": info.last_cmd,
        "last_cmd_at_secs": info.last_cmd_at_secs,
        "cmds_executed": info.cmds_executed,
        "execute_commands": info.execute_commands,
        "sim_time_advanced_picos": info.sim_time_advanced_picos as u64,
        "sim_time_advanced_picos_str": info.sim_time_advanced_picos.to_string(),
    })
}

fn list_closed_sessions_json(ctx: &ServerCtx, limit: Option<usize>) -> Value {
    let closed = ctx.closed_sessions.lock().unwrap();
    let limit = limit.unwrap_or(closed.len()).min(closed.len());
    let entries: Vec<Value> = closed
        .iter()
        .rev()
        .take(limit)
        .map(closed_session_to_json)
        .collect();
    json!({
        "count": entries.len(),
        "total_recorded": closed.len(),
        "capacity": CLOSED_SESSIONS_CAPACITY,
        "sessions": entries,
    })
}

fn history_entry_to_json(entry: &HistoryEntry) -> Value {
    json!({
        "session_id": entry.session_id,
        "cmd": entry.cmd,
        "at_secs": entry.at_secs,
        "ok": entry.ok,
        "detail": entry.detail,
    })
}

fn list_sessions_json(ctx: &ServerCtx, self_id: u64) -> Value {
    let now = now_unix_secs();
    let snapshot: Vec<ClientInfo> = {
        let sessions = ctx.sessions.lock().unwrap();
        sessions.values().cloned().collect()
    };
    let mut entries: Vec<Value> = snapshot
        .iter()
        .map(|info| session_to_json(info, self_id, now))
        .collect();
    entries.sort_by_key(|v| v["id"].as_u64().unwrap_or(0));
    json!({
        "count": entries.len(),
        "sessions": entries,
        "server_now_secs": now,
        "server_started_at_secs": ctx.started_at_secs,
    })
}

fn get_history_json(ctx: &ServerCtx, limit: Option<usize>) -> Value {
    let history = ctx.history.lock().unwrap();
    let limit = limit.unwrap_or(history.len()).min(history.len());
    let entries: Vec<Value> = history
        .iter()
        .rev()
        .take(limit)
        .map(history_entry_to_json)
        .collect();
    json!({
        "count": entries.len(),
        "capacity": HISTORY_CAPACITY,
        "entries": entries,
    })
}

fn execution_progress_json() -> Value {
    let total = EXECUTION_PROGRESS_TOTAL.load(AtomicOrdering::Relaxed);
    if total == 0 {
        return json!({"running": false});
    }
    let done = EXECUTION_PROGRESS_DONE.load(AtomicOrdering::Relaxed);
    let session_id = EXECUTION_PROGRESS_SESSION_ID.load(AtomicOrdering::Relaxed);
    let started_at_secs = EXECUTION_PROGRESS_STARTED_AT_SECS.load(AtomicOrdering::Relaxed);
    let sim_time_before = EXECUTION_PROGRESS_SIM_TIME_BEFORE.load(AtomicOrdering::Relaxed);
    let now = now_unix_secs();
    json!({
        "running": true,
        "session_id": session_id,
        "total": total,
        "done": done.min(total),
        "fraction": (done as f64) / (total as f64),
        "started_at_secs": started_at_secs,
        "elapsed_secs": now.saturating_sub(started_at_secs),
        "sim_time_before_picos": sim_time_before,
    })
}

fn server_status_json(ctx: &ServerCtx) -> Value {
    let now = now_unix_secs();
    let session_count = ctx.sessions.lock().unwrap().len();
    let closed_count = ctx.closed_sessions.lock().unwrap().len();
    json!({
        "server_now_secs": now,
        "server_started_at_secs": ctx.started_at_secs,
        "uptime_secs": now.saturating_sub(ctx.started_at_secs),
        "session_count": session_count,
        "closed_session_count": closed_count,
        "closed_session_capacity": CLOSED_SESSIONS_CAPACITY,
    })
}

/// Apply a session-scoped request that touches only the registry (not emulator state).
async fn handle_session_request(
    ctx: &Arc<ServerCtx>,
    self_id: u64,
    request: ServiceRequest,
) -> anyhow::Result<ServiceOutcome> {
    let response = match request {
        ServiceRequest::WhoAmI => {
            let now = now_unix_secs();
            let info_opt = {
                let sessions = ctx.sessions.lock().unwrap();
                sessions.get(&self_id).cloned()
            };
            match info_opt {
                Some(info) => session_to_json(&info, self_id, now),
                None => json!({"id": self_id, "error": "session not registered"}),
            }
        }
        ServiceRequest::ListSessions => list_sessions_json(ctx, self_id),
        ServiceRequest::ListClosedSessions { limit } => list_closed_sessions_json(ctx, limit),
        ServiceRequest::SetLabel { label } => {
            let trimmed = label.trim();
            let stored = if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            };
            let mut sessions = ctx.sessions.lock().unwrap();
            if let Some(info) = sessions.get_mut(&self_id) {
                info.label = stored.clone();
            }
            json!({"id": self_id, "label": stored})
        }
        ServiceRequest::GetHistory { limit } => get_history_json(ctx, limit),
        ServiceRequest::GetServerStatus => server_status_json(ctx),
        ServiceRequest::GetExecutionProgress => execution_progress_json(),
        ServiceRequest::StartModule { name } => {
            let mut sessions = ctx.sessions.lock().unwrap();
            let info = sessions
                .get_mut(&self_id)
                .ok_or_else(|| anyhow!("session not registered"))?;
            info.current_module = Some(ModuleInProgress {
                name: name.clone(),
                started_at_secs: now_unix_secs(),
                sim_time_before_picos: info.sim_time_advanced_picos,
                execute_count_before: info.execute_commands,
            });
            json!({"name": name, "started_at_secs": now_unix_secs()})
        }
        ServiceRequest::EndModule { ok, error } => {
            let mut sessions = ctx.sessions.lock().unwrap();
            let info = sessions
                .get_mut(&self_id)
                .ok_or_else(|| anyhow!("session not registered"))?;
            let m = info
                .current_module
                .take()
                .ok_or_else(|| anyhow!("end_module without start_module"))?;
            let ended_at = now_unix_secs();
            let entry = ModuleEntry {
                name: m.name,
                started_at_secs: m.started_at_secs,
                ended_at_secs: ended_at,
                sim_time_picos: info
                    .sim_time_advanced_picos
                    .saturating_sub(m.sim_time_before_picos),
                execute_count: info.execute_commands.saturating_sub(m.execute_count_before),
                ok,
                error,
            };
            if info.module_log.len() >= MODULE_LOG_CAPACITY {
                info.module_log.pop_front();
            }
            let response = module_entry_to_json(&entry);
            info.module_log.push_back(entry);
            response
        }
        ServiceRequest::ListSessionModules { id, limit } => {
            let target = id.unwrap_or(self_id);
            let sessions = ctx.sessions.lock().unwrap();
            match sessions.get(&target) {
                Some(info) => {
                    let limit = limit
                        .unwrap_or(info.module_log.len())
                        .min(info.module_log.len());
                    let entries: Vec<Value> = info
                        .module_log
                        .iter()
                        .rev()
                        .take(limit)
                        .map(module_entry_to_json)
                        .collect();
                    let current = info.current_module.as_ref().map(|m| {
                        json!({
                            "name": m.name,
                            "started_at_secs": m.started_at_secs,
                            "elapsed_secs": now_unix_secs().saturating_sub(m.started_at_secs),
                        })
                    });
                    json!({
                        "session_id": target,
                        "count": entries.len(),
                        "capacity": MODULE_LOG_CAPACITY,
                        "current_module": current,
                        "modules": entries,
                    })
                }
                None => bail!("session {target} not found"),
            }
        }
        other => bail!(
            "internal error: {} routed to session handler",
            cmd_name(&other)
        ),
    };
    Ok(ServiceOutcome::plain(response))
}

/// Whether a command only touches the session registry (never the emulator state lock).
fn cmd_is_session_only(req: &ServiceRequest) -> bool {
    matches!(
        req,
        ServiceRequest::WhoAmI
            | ServiceRequest::ListSessions
            | ServiceRequest::ListClosedSessions { .. }
            | ServiceRequest::SetLabel { .. }
            | ServiceRequest::GetHistory { .. }
            | ServiceRequest::GetServerStatus
            | ServiceRequest::GetExecutionProgress
            | ServiceRequest::StartModule { .. }
            | ServiceRequest::EndModule { .. }
            | ServiceRequest::ListSessionModules { .. }
    )
}

fn record_history(ctx: &ServerCtx, entry: HistoryEntry) {
    let mut history = ctx.history.lock().unwrap();
    if history.len() >= HISTORY_CAPACITY {
        history.pop_front();
    }
    history.push_back(entry);
}

fn record_request(ctx: &ServerCtx, session_id: u64, cmd: &str) {
    let mut sessions = ctx.sessions.lock().unwrap();
    if let Some(info) = sessions.get_mut(&session_id) {
        info.last_cmd = Some(cmd.to_string());
        info.last_cmd_at_secs = Some(now_unix_secs());
        info.cmds_executed = info.cmds_executed.saturating_add(1);
    }
}

async fn handle_client(
    stream: TcpStream,
    ctx: Arc<ServerCtx>,
    mut shutdown_rx: broadcast::Receiver<()>,
) -> anyhow::Result<()> {
    let peer = stream
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| "<unknown>".into());

    let session_id = ctx.next_session_id.fetch_add(1, AtomicOrdering::SeqCst);
    let connected_at = now_unix_secs();
    {
        let mut sessions = ctx.sessions.lock().unwrap();
        sessions.insert(
            session_id,
            ClientInfo {
                id: session_id,
                peer_addr: peer.clone(),
                connected_at_secs: connected_at,
                label: None,
                last_cmd: None,
                last_cmd_at_secs: None,
                cmds_executed: 0,
                sim_time_advanced_picos: 0,
                execute_commands: 0,
                current_module: None,
                module_log: VecDeque::with_capacity(MODULE_LOG_CAPACITY),
            },
        );
    }

    if !is_quiet() {
        eprintln!("Client connected: session_id={session_id} peer={peer}");
    }

    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();

    let outcome: anyhow::Result<()> = async {
        loop {
            let next = tokio::select! {
                line = lines.next_line() => line?,
                _ = shutdown_rx.recv() => {
                    let bye = error_response("server is shutting down");
                    let _ = write_json_line(&mut writer, &bye).await;
                    return Ok(());
                }
            };

            let line = match next {
                Some(line) => line,
                None => return Ok(()),
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let parsed: Result<ServiceRequest, _> = serde_json::from_str(line);
            let (response_value, shutdown_after, cmd_label, ok_flag, detail) = match parsed {
                Ok(request) => {
                    let cmd_label = cmd_name(&request).to_string();
                    record_request(&ctx, session_id, &cmd_label);

                    let outcome_res = if cmd_is_session_only(&request) {
                        handle_session_request(&ctx, session_id, request).await
                    } else if cmd_needs_write_lock(&request) {
                        let mut guard = ctx.state.write().await;
                        guard.handle_write_request(session_id, request).await
                    } else {
                        let guard = ctx.state.read().await;
                        guard.handle_read_request(request).await
                    };

                    match outcome_res {
                        Ok(outcome) => {
                            // Attribute simulator time advanced by execute_*
                            // commands to this session, and surface it in
                            // the history entry for tiny runs.
                            let sim_delta = outcome.sim_time_advanced_picos;
                            if sim_delta > 0 {
                                let mut sessions = ctx.sessions.lock().unwrap();
                                if let Some(info) = sessions.get_mut(&session_id) {
                                    info.sim_time_advanced_picos =
                                        info.sim_time_advanced_picos.saturating_add(sim_delta);
                                    info.execute_commands = info.execute_commands.saturating_add(1);
                                }
                            }
                            let detail = if sim_delta > 0 {
                                Some(format!("sim_time +{} ps", sim_delta))
                            } else {
                                None
                            };
                            (
                                ok_response(outcome.response),
                                outcome.shutdown,
                                cmd_label,
                                true,
                                detail,
                            )
                        }
                        Err(err) => {
                            let msg = err.to_string();
                            (
                                error_response(msg.clone()),
                                false,
                                cmd_label,
                                false,
                                Some(msg),
                            )
                        }
                    }
                }
                Err(err) => {
                    let msg = format!("invalid request JSON: {err}");
                    (
                        error_response(msg.clone()),
                        false,
                        "<invalid>".to_string(),
                        false,
                        Some(msg),
                    )
                }
            };

            record_history(
                &ctx,
                HistoryEntry {
                    session_id,
                    cmd: cmd_label,
                    at_secs: now_unix_secs(),
                    ok: ok_flag,
                    detail,
                },
            );

            write_json_line(&mut writer, &response_value).await?;
            if shutdown_after {
                let _ = ctx.shutdown_tx.send(());
                return Ok(());
            }
        }
    }
    .await;

    let info_snapshot = {
        let mut sessions = ctx.sessions.lock().unwrap();
        sessions.remove(&session_id)
    };
    if let Some(info) = info_snapshot {
        let mut closed = ctx.closed_sessions.lock().unwrap();
        if closed.len() >= CLOSED_SESSIONS_CAPACITY {
            closed.pop_front();
        }
        closed.push_back(ClosedSession {
            info,
            closed_at_secs: now_unix_secs(),
        });
    }

    if !is_quiet() {
        eprintln!("Client disconnected: session_id={session_id} peer={peer}");
    }

    outcome
}

async fn serve(opts: &Opts) -> anyhow::Result<()> {
    let listener = TcpListener::bind(&opts.bind)
        .await
        .with_context(|| format!("failed to bind {}", opts.bind))?;

    let mut initial_state = EmulatorState::new();
    initial_state.preload_from_opts(opts).await?;

    let (shutdown_tx, _) = broadcast::channel::<()>(16);
    let ctx = Arc::new(ServerCtx {
        state: RwLock::new(initial_state),
        sessions: std::sync::Mutex::new(HashMap::new()),
        closed_sessions: std::sync::Mutex::new(VecDeque::with_capacity(CLOSED_SESSIONS_CAPACITY)),
        history: std::sync::Mutex::new(VecDeque::with_capacity(HISTORY_CAPACITY)),
        next_session_id: AtomicU64::new(1),
        started_at_secs: now_unix_secs(),
        shutdown_tx,
    });

    // Always emit this — it's a once-at-startup message and operators
    // running with the default (--serve auto-quiet) still want to see
    // that the bind succeeded.
    eprintln!("Online emulator listening on {}", opts.bind);

    let mut shutdown_rx = ctx.shutdown_tx.subscribe();
    loop {
        tokio::select! {
            accept = listener.accept() => {
                let (stream, _) = accept?;
                let ctx = ctx.clone();
                let client_shutdown_rx = ctx.shutdown_tx.subscribe();
                // spawn_local keeps the !Send simulator state on the current
                // thread while still allowing concurrent client connections.
                tokio::task::spawn_local(async move {
                    if let Err(err) = handle_client(stream, ctx, client_shutdown_rx).await {
                        if !is_quiet() {
                            eprintln!("Client task error: {err}");
                        }
                    }
                });
            }
            _ = shutdown_rx.recv() => {
                if !is_quiet() {
                    eprintln!("Shutdown signal received; stopping accept loop");
                }
                break;
            }
        }
    }

    Ok(())
}

// ===========================================================================
// Gateway mode
// ===========================================================================
//
// The gateway accepts client connections on `--bind` and proxies them to a
// dedicated `--serve` backend subprocess per session. Each backend has its
// own EmulatorState, so multiple clients can run independent kernels in true
// isolation — the inverse of `--serve` where everyone shares one HBM/MRAM.
//
// Backends are spawned **lazily**: a session that only issues session-only
// commands (e.g. the WebGUI monitor calling `list_sessions`) never costs a
// child process. On the first hardware-touching command (`execute_*`,
// `load_*`, `read_*`, `get_state`, `get_config`, `reset`, `ping`) the
// gateway spawns `--serve` on a kernel-assigned ephemeral port and opens
// a persistent connection to it. On client disconnect the gateway sends
// `{"cmd":"shutdown"}` to the backend, waits for the child to exit, and
// pushes the session info into the closed-sessions buffer with a final
// sim-time delta scraped from the response stream.

/// Session metadata kept locally by the gateway. Mirrors `ClientInfo` plus
/// gateway-only fields (backend pid/port, currently-executing flag).
#[derive(Clone)]
struct GwClientInfo {
    base: ClientInfo,
    backend_port: Option<u16>,
    backend_pid: Option<u32>,
    /// Set true while an execute_batch / execute_file is in flight, so
    /// `list_sessions` can show which session is busy without polling
    /// every backend on every call.
    is_executing: bool,
    /// Per-session config TOML supplied by the client at handshake time
    /// (see `set_session_config` command). When set, the gateway exports
    /// it as `PLENA_CONFIG=<config_path>` when spawning this session's
    /// backend subprocess.
    config_temp_path: Option<PathBuf>,
    config_name: Option<String>,
    config_summary: Option<Value>,
}

struct GatewayCtx {
    sessions: std::sync::Mutex<HashMap<u64, GwClientInfo>>,
    closed_sessions: std::sync::Mutex<VecDeque<ClosedSession>>,
    history: std::sync::Mutex<VecDeque<HistoryEntry>>,
    next_session_id: AtomicU64,
    started_at_secs: u64,
    self_path: PathBuf,
    /// Optional `--quiet` propagation to backends. Always true under the
    /// gateway since backends produce noise nobody reads.
    backend_quiet: bool,
}

impl GatewayCtx {
    fn snapshot_session_jsons(&self, self_id: u64, now_secs: u64) -> Vec<Value> {
        let sessions = self.sessions.lock().unwrap();
        let mut entries: Vec<Value> = sessions
            .values()
            .map(|gw| {
                let mut obj = session_to_json(&gw.base, self_id, now_secs);
                if let Value::Object(ref mut map) = obj {
                    map.insert("is_executing".to_string(), Value::Bool(gw.is_executing));
                    map.insert(
                        "backend_port".to_string(),
                        gw.backend_port
                            .map(|p| Value::from(p))
                            .unwrap_or(Value::Null),
                    );
                    map.insert(
                        "backend_pid".to_string(),
                        gw.backend_pid
                            .map(|p| Value::from(p))
                            .unwrap_or(Value::Null),
                    );
                    // Per-session config (see `set_session_config`). When
                    // unset the WebGUI knows the session is using the
                    // gateway-default config.
                    map.insert(
                        "config_path".to_string(),
                        gw.config_temp_path
                            .as_ref()
                            .map(|p| Value::from(p.to_string_lossy().to_string()))
                            .unwrap_or(Value::Null),
                    );
                    map.insert(
                        "config_name".to_string(),
                        gw.config_name
                            .clone()
                            .map(Value::from)
                            .unwrap_or(Value::Null),
                    );
                    map.insert(
                        "config_summary".to_string(),
                        gw.config_summary.clone().unwrap_or(Value::Null),
                    );
                }
                obj
            })
            .collect();
        entries.sort_by_key(|v| v["id"].as_u64().unwrap_or(0));
        entries
    }
}

/// Live TCP connection to a backend subprocess plus its handle.
struct Backend {
    child: tokio::process::Child,
    port: u16,
    pid: u32,
    /// Newline-delimited JSON reader/writer split halves.
    reader: tokio::io::Lines<BufReader<tokio::net::tcp::OwnedReadHalf>>,
    writer: tokio::net::tcp::OwnedWriteHalf,
}

impl Backend {
    /// Send a request line, await one response line.
    async fn round_trip(&mut self, request_line: &str) -> anyhow::Result<String> {
        self.writer.write_all(request_line.as_bytes()).await?;
        if !request_line.ends_with('\n') {
            self.writer.write_all(b"\n").await?;
        }
        let line = self
            .reader
            .next_line()
            .await?
            .ok_or_else(|| anyhow!("backend closed the connection"))?;
        Ok(line)
    }

    /// Best-effort graceful shutdown then SIGKILL fallback.
    async fn shutdown(mut self) {
        let _ = self.writer.write_all(b"{\"cmd\":\"shutdown\"}\n").await;
        // Give the backend up to 2s to exit cleanly; otherwise kill it.
        let wait = tokio::time::timeout(std::time::Duration::from_secs(2), self.child.wait());
        match wait.await {
            Ok(Ok(_)) => {}
            _ => {
                let _ = self.child.kill().await;
                let _ = self.child.wait().await;
            }
        }
    }
}

/// Ask the kernel for a free ephemeral port. There is a small race window
/// between dropping the listener here and binding it inside the spawned
/// backend, but the port hasn't been TIME_WAIT'd yet (we never `accept()`)
/// so collisions are exceedingly unlikely in practice.
async fn allocate_ephemeral_port() -> anyhow::Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

async fn spawn_backend(
    ctx: &GatewayCtx,
    session_config_path: Option<&Path>,
) -> anyhow::Result<Backend> {
    use tokio::io::AsyncReadExt as _;
    let port = allocate_ephemeral_port().await?;
    let bind_addr = format!("127.0.0.1:{port}");

    let mut cmd = tokio::process::Command::new(&ctx.self_path);
    cmd.arg("--serve").arg("--bind").arg(&bind_addr);
    if ctx.backend_quiet {
        cmd.arg("--quiet");
    }
    // Per-session config override (see `set_session_config`). When the
    // client has handed us a TOML for this session we pin it via the
    // `PLENA_CONFIG` env var, which `load_config.rs::load_config` reads
    // at backend startup. Otherwise the backend falls back to the
    // gateway's own PLENA_CONFIG (inherited) or to plena_settings.toml.
    if let Some(path) = session_config_path {
        cmd.env("PLENA_CONFIG", path);
    }
    // Backends inherit our env (RUST_BACKTRACE, DYLD_*, etc.) but we drop
    // their stdio — nobody reads it and the per-instruction printlns get
    // muted anyway since we pass --quiet.
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::null());
    // Keep stderr captured into a pipe so we can include a snippet in
    // error reports if the backend fails to start.
    cmd.stderr(std::process::Stdio::piped());
    // The child should die if the gateway is killed without cleanup; on
    // unix this is best-effort via setsid+inherit, on macOS it's just
    // "kill child explicitly in our Drop / shutdown path".
    cmd.kill_on_drop(true);

    let mut child = cmd
        .spawn()
        .with_context(|| format!("failed to spawn backend at {bind_addr}"))?;
    let pid = child.id().unwrap_or(0);

    // Poll until the backend's TCP listener is up, or it dies, or we time out.
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(15);
    let mut stderr_buf = Vec::new();
    let mut stderr = child.stderr.take();
    let stream = loop {
        if tokio::time::Instant::now() > deadline {
            if let Some(mut s) = stderr {
                let _ = s.read_to_end(&mut stderr_buf).await;
            }
            let _ = child.kill().await;
            bail!(
                "backend did not become ready within 15s on {bind_addr}; \
                 stderr tail: {}",
                String::from_utf8_lossy(&stderr_buf)
            );
        }
        match TcpStream::connect(&bind_addr).await {
            Ok(s) => break s,
            Err(_) => {
                // Check if backend died early.
                if let Some(status) = child.try_wait()? {
                    if let Some(mut s) = stderr {
                        let _ = s.read_to_end(&mut stderr_buf).await;
                    }
                    bail!(
                        "backend exited early ({:?}) before listening on {bind_addr}; \
                         stderr tail: {}",
                        status,
                        String::from_utf8_lossy(&stderr_buf)
                    );
                }
                tokio::time::sleep(std::time::Duration::from_millis(80)).await;
            }
        }
    };
    // Stuff stderr back so it stays drained by the OS pipe buffer; we
    // don't actively read it anymore, but kill_on_drop will release the fd.
    child.stderr = stderr;

    let (reader, writer) = stream.into_split();
    Ok(Backend {
        child,
        port,
        pid,
        reader: BufReader::new(reader).lines(),
        writer,
    })
}

/// Write a client-supplied TOML config to a tempfile keyed by session id.
///
/// Returns the on-disk path (we pass this as `PLENA_CONFIG` to the
/// session's backend) and a small JSON summary of the BEHAVIOR.CONFIG
/// section for the WebGUI to display.
fn persist_session_config(
    session_id: u64,
    toml_text: &str,
) -> Result<(PathBuf, Option<Value>), String> {
    let parsed: toml::Value = toml::from_str(toml_text)
        .map_err(|e| format!("TOML parse error: {e}"))?;
    let summary = extract_behavior_summary(&parsed);

    let path = std::env::temp_dir().join(format!("plena_session_{session_id}.toml"));
    std::fs::write(&path, toml_text)
        .map_err(|e| format!("failed to write {path:?}: {e}"))?;
    Ok((path, summary))
}

/// Pluck a few headline numbers out of `[BEHAVIOR.CONFIG.<KEY>] value = <int>`
/// so the WebGUI can show what this session's hardware looks like.
fn extract_behavior_summary(parsed: &toml::Value) -> Option<Value> {
    let cfg = parsed.get("BEHAVIOR")?.get("CONFIG")?.as_table()?;
    let pick = |k: &str| -> Option<Value> {
        let entry = cfg.get(k)?;
        let v = entry
            .get("value")
            .or(Some(entry))
            .and_then(|x| x.as_integer())?;
        Some(Value::from(v))
    };
    let mut summary = serde_json::Map::new();
    for key in [
        "BLEN",
        "HLEN",
        "MLEN",
        "VLEN",
        "BROADCAST_AMOUNT",
        "HBM_WIDTH",
        "MATRIX_SRAM_SIZE",
        "VECTOR_SRAM_SIZE",
        "FP_SRAM_SIZE",
        "INT_SRAM_SIZE",
        "HBM_M_Prefetch_Amount",
        "HBM_V_Prefetch_Amount",
        "HBM_V_Writeback_Amount",
    ] {
        if let Some(v) = pick(key) {
            summary.insert(key.to_string(), v);
        }
    }
    if summary.is_empty() {
        None
    } else {
        Some(Value::Object(summary))
    }
}

fn is_session_only_cmd(cmd: &str) -> bool {
    matches!(
        cmd,
        "who_am_i"
            | "list_sessions"
            | "list_closed_sessions"
            | "set_label"
            | "set_session_config"
            | "get_history"
            | "get_server_status"
            | "get_execution_progress"
            | "get_session_state"
            | "get_session_config"
            | "get_session_progress"
            | "enable_session_trace"
            | "dump_session_trace"
            | "start_module"
            | "end_module"
            | "list_session_modules"
    )
}

/// Open a one-shot TCP connection to a session's backend, send a single
/// JSON command, return the parsed response. Used by the gateway to
/// service "drill into session N" UI requests without holding the
/// session's primary connection (which may be busy executing a kernel).
async fn proxy_to_backend(port: u16, command: &str) -> anyhow::Result<Value> {
    let mut stream = tokio::time::timeout(
        std::time::Duration::from_millis(500),
        TcpStream::connect(format!("127.0.0.1:{port}")),
    )
    .await
    .context("timed out connecting to backend")?
    .context("backend connect failed")?;
    let (reader, mut writer) = stream.split();
    writer.write_all(command.as_bytes()).await?;
    if !command.ends_with('\n') {
        writer.write_all(b"\n").await?;
    }
    let mut lines = BufReader::new(reader).lines();
    let response_line = tokio::time::timeout(std::time::Duration::from_secs(5), lines.next_line())
        .await
        .context("timed out waiting for backend response")?
        .context("backend read error")?
        .ok_or_else(|| anyhow!("backend closed connection"))?;
    let v: Value = serde_json::from_str(&response_line).context("invalid backend response")?;
    Ok(v)
}

fn is_execute_cmd(cmd: &str) -> bool {
    matches!(cmd, "execute_batch" | "execute_file")
}

fn gateway_server_status_json(ctx: &GatewayCtx) -> Value {
    let now = now_unix_secs();
    let session_count = ctx.sessions.lock().unwrap().len();
    let closed_count = ctx.closed_sessions.lock().unwrap().len();
    let executing_count = ctx
        .sessions
        .lock()
        .unwrap()
        .values()
        .filter(|s| s.is_executing)
        .count();
    json!({
        "mode": "gateway",
        "server_now_secs": now,
        "server_started_at_secs": ctx.started_at_secs,
        "uptime_secs": now.saturating_sub(ctx.started_at_secs),
        "session_count": session_count,
        "executing_count": executing_count,
        "closed_session_count": closed_count,
        "closed_session_capacity": CLOSED_SESSIONS_CAPACITY,
    })
}

fn gateway_list_sessions_json(ctx: &GatewayCtx, self_id: u64) -> Value {
    let now = now_unix_secs();
    let entries = ctx.snapshot_session_jsons(self_id, now);
    json!({
        "count": entries.len(),
        "sessions": entries,
        "server_now_secs": now,
        "server_started_at_secs": ctx.started_at_secs,
    })
}

fn gateway_record_request(ctx: &GatewayCtx, session_id: u64, cmd: &str) {
    let mut sessions = ctx.sessions.lock().unwrap();
    if let Some(info) = sessions.get_mut(&session_id) {
        info.base.last_cmd = Some(cmd.to_string());
        info.base.last_cmd_at_secs = Some(now_unix_secs());
        info.base.cmds_executed = info.base.cmds_executed.saturating_add(1);
    }
}

fn gateway_record_history(ctx: &GatewayCtx, entry: HistoryEntry) {
    let mut history = ctx.history.lock().unwrap();
    if history.len() >= HISTORY_CAPACITY {
        history.pop_front();
    }
    history.push_back(entry);
}

fn parse_execute_response_for_sim_time(line: &str) -> Option<u64> {
    let v: Value = serde_json::from_str(line).ok()?;
    let data = v.get("data")?;
    data.get("sim_time_advanced_picos")?.as_u64()
}

/// Helper used by `handle_gateway_local_cmd` for the proxy commands.
/// Returns Ok(data Value) on success, Err(string) on failure — caller
/// decides how to wrap into an ok/err response envelope.
async fn proxy_session_cmd(
    ctx: &Arc<GatewayCtx>,
    target_id: u64,
    request_line: &str,
) -> Result<Value, String> {
    let port = {
        let sessions = ctx.sessions.lock().unwrap();
        match sessions.get(&target_id) {
            Some(info) => info.backend_port,
            None => return Err(format!("session {target_id} not found")),
        }
    };
    let port = match port {
        Some(p) => p,
        None => {
            // Session is registered but has no backend yet (only ran
            // session-only commands). Caller probably wants a friendly
            // "idle" indication rather than an error.
            return Ok(json!({
                "session_id": target_id,
                "running": false,
                "no_backend": true,
                "message": "session has no backend (only ran session-only commands)",
            }));
        }
    };
    match proxy_to_backend(port, request_line).await {
        Ok(v) => {
            // Backend reply envelope is {"ok": bool, "data": ...} — pass
            // the inner `data` through so the gateway-level envelope can
            // wrap it consistently with other session-only responses.
            if v.get("ok").and_then(|b| b.as_bool()).unwrap_or(false) {
                Ok(v.get("data").cloned().unwrap_or(Value::Null))
            } else {
                Err(v
                    .get("error")
                    .and_then(|e| e.as_str())
                    .unwrap_or("backend error")
                    .to_string())
            }
        }
        Err(err) => Err(format!(
            "proxy to session {target_id} backend failed: {err}"
        )),
    }
}

async fn handle_gateway_local_cmd(
    ctx: &Arc<GatewayCtx>,
    session_id: u64,
    raw: &Value,
    cmd_name: &str,
) -> Value {
    let result: Result<Value, String> = match cmd_name {
        "who_am_i" => {
            let now = now_unix_secs();
            let entries = ctx.snapshot_session_jsons(session_id, now);
            let me = entries
                .into_iter()
                .find(|v| v["is_self"].as_bool().unwrap_or(false))
                .unwrap_or_else(|| json!({"id": session_id, "error": "session not registered"}));
            Ok(me)
        }
        "list_sessions" => Ok(gateway_list_sessions_json(ctx, session_id)),
        "list_closed_sessions" => {
            let limit = raw
                .get("limit")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize);
            let closed = ctx.closed_sessions.lock().unwrap();
            let limit = limit.unwrap_or(closed.len()).min(closed.len());
            let entries: Vec<Value> = closed
                .iter()
                .rev()
                .take(limit)
                .map(closed_session_to_json)
                .collect();
            Ok(json!({
                "count": entries.len(),
                "total_recorded": closed.len(),
                "capacity": CLOSED_SESSIONS_CAPACITY,
                "sessions": entries,
            }))
        }
        "set_label" => {
            let label = raw.get("label").and_then(|v| v.as_str()).unwrap_or("");
            let trimmed = label.trim();
            let stored = if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            };
            let mut sessions = ctx.sessions.lock().unwrap();
            if let Some(info) = sessions.get_mut(&session_id) {
                info.base.label = stored.clone();
            }
            Ok(json!({"id": session_id, "label": stored}))
        }
        "set_session_config" => {
            // Per-session hardware config supplied by the client. Must be
            // called *before* any hardware-touching command (which is what
            // lazily spawns the backend) — otherwise the backend already
            // loaded the default config and we can't reconfigure it.
            let sessions_guard = ctx.sessions.lock().unwrap();
            let already_spawned = sessions_guard
                .get(&session_id)
                .map(|s| s.backend_port.is_some())
                .unwrap_or(false);
            drop(sessions_guard);
            if already_spawned {
                Err("set_session_config must be called before the backend is spawned; \
                     this session's backend is already running with its current config"
                    .to_string())
            } else {
                let toml_text = raw
                    .get("config_toml")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let name = raw
                    .get("config_name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());
                if toml_text.is_empty() {
                    Err("set_session_config: missing required field `config_toml`".to_string())
                } else {
                    match persist_session_config(session_id, toml_text) {
                        Err(e) => Err(format!("failed to persist session config: {e}")),
                        Ok((path, summary)) => {
                            let mut sessions = ctx.sessions.lock().unwrap();
                            if let Some(info) = sessions.get_mut(&session_id) {
                                // Replace any earlier per-session config (idempotent
                                // re-handshake). Unlink the previous tempfile.
                                if let Some(old) = info.config_temp_path.take() {
                                    let _ = std::fs::remove_file(&old);
                                }
                                info.config_temp_path = Some(path.clone());
                                info.config_name = name.clone();
                                info.config_summary = summary.clone();
                            }
                            Ok(json!({
                                "id": session_id,
                                "config_path": path.to_string_lossy().to_string(),
                                "config_name": name,
                                "config_summary": summary,
                            }))
                        }
                    }
                }
            }
        }
        "get_history" => {
            let limit = raw
                .get("limit")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize);
            let history = ctx.history.lock().unwrap();
            let limit = limit.unwrap_or(history.len()).min(history.len());
            let entries: Vec<Value> = history
                .iter()
                .rev()
                .take(limit)
                .map(history_entry_to_json)
                .collect();
            Ok(json!({
                "count": entries.len(),
                "capacity": HISTORY_CAPACITY,
                "entries": entries,
            }))
        }
        "get_server_status" => Ok(gateway_server_status_json(ctx)),
        "get_execution_progress" => {
            // Proxy to the targeted session's backend (or self if no id given).
            let target = raw.get("id").and_then(|v| v.as_u64()).unwrap_or(session_id);
            proxy_session_cmd(ctx, target, r#"{"cmd":"get_execution_progress"}"#).await
        }
        "get_session_state" => match raw.get("id").and_then(|v| v.as_u64()) {
            None => Err("missing required field `id`".to_string()),
            Some(target) => proxy_session_cmd(ctx, target, r#"{"cmd":"get_state"}"#).await,
        },
        "get_session_config" => match raw.get("id").and_then(|v| v.as_u64()) {
            None => Err("missing required field `id`".to_string()),
            Some(target) => proxy_session_cmd(ctx, target, r#"{"cmd":"get_config"}"#).await,
        },
        "get_session_progress" => match raw.get("id").and_then(|v| v.as_u64()) {
            None => Err("missing required field `id`".to_string()),
            Some(target) => {
                proxy_session_cmd(ctx, target, r#"{"cmd":"get_execution_progress"}"#).await
            }
        },
        // Toggle trace collection on session `id`'s backend.
        "enable_session_trace" => match raw.get("id").and_then(|v| v.as_u64()) {
            None => Err("missing required field `id`".to_string()),
            Some(target) => {
                let enabled = raw.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true);
                let cmd_line = format!(
                    "{{\"cmd\":\"enable_trace\",\"enabled\":{}}}",
                    if enabled { "true" } else { "false" }
                );
                proxy_session_cmd(ctx, target, &cmd_line).await
            }
        },
        // Dump session `id`'s collected trace to `path` on the host.
        // WebGUI's `/api/session_trace` uses this and then invokes the
        // Python renderer over the resulting binary.
        "dump_session_trace" => match (
            raw.get("id").and_then(|v| v.as_u64()),
            raw.get("path").and_then(|v| v.as_str()),
        ) {
            (None, _) => Err("missing required field `id`".to_string()),
            (_, None) => Err("missing required field `path`".to_string()),
            (Some(target), Some(path)) => {
                let cmd_line = serde_json::to_string(&serde_json::json!({
                    "cmd": "dump_trace",
                    "path": path,
                }))
                .expect("dump_trace command JSON is always serializable");
                proxy_session_cmd(ctx, target, &cmd_line).await
            }
        },
        "start_module" => {
            let name = raw
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            match name {
                None => Err("missing required field `name`".to_string()),
                Some(name) => {
                    let mut sessions = ctx.sessions.lock().unwrap();
                    match sessions.get_mut(&session_id) {
                        Some(gw) => {
                            gw.base.current_module = Some(ModuleInProgress {
                                name: name.clone(),
                                started_at_secs: now_unix_secs(),
                                sim_time_before_picos: gw.base.sim_time_advanced_picos,
                                execute_count_before: gw.base.execute_commands,
                            });
                            Ok(json!({"name": name, "started_at_secs": now_unix_secs()}))
                        }
                        None => Err("session not registered".to_string()),
                    }
                }
            }
        }
        "end_module" => {
            let ok_flag = raw.get("ok").and_then(|v| v.as_bool()).unwrap_or(true);
            let err_msg = raw
                .get("error")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let mut sessions = ctx.sessions.lock().unwrap();
            match sessions.get_mut(&session_id) {
                None => Err("session not registered".to_string()),
                Some(gw) => match gw.base.current_module.take() {
                    None => Err("end_module without start_module".to_string()),
                    Some(m) => {
                        let ended_at = now_unix_secs();
                        let entry = ModuleEntry {
                            name: m.name,
                            started_at_secs: m.started_at_secs,
                            ended_at_secs: ended_at,
                            sim_time_picos: gw
                                .base
                                .sim_time_advanced_picos
                                .saturating_sub(m.sim_time_before_picos),
                            execute_count: gw
                                .base
                                .execute_commands
                                .saturating_sub(m.execute_count_before),
                            ok: ok_flag,
                            error: err_msg,
                        };
                        if gw.base.module_log.len() >= MODULE_LOG_CAPACITY {
                            gw.base.module_log.pop_front();
                        }
                        let response = module_entry_to_json(&entry);
                        gw.base.module_log.push_back(entry);
                        Ok(response)
                    }
                },
            }
        }
        "list_session_modules" => {
            let target = raw.get("id").and_then(|v| v.as_u64()).unwrap_or(session_id);
            let limit = raw
                .get("limit")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize);
            let sessions = ctx.sessions.lock().unwrap();
            match sessions.get(&target) {
                None => Err(format!("session {target} not found")),
                Some(gw) => {
                    let log = &gw.base.module_log;
                    let limit = limit.unwrap_or(log.len()).min(log.len());
                    let entries: Vec<Value> = log
                        .iter()
                        .rev()
                        .take(limit)
                        .map(module_entry_to_json)
                        .collect();
                    let current = gw.base.current_module.as_ref().map(|m| {
                        json!({
                            "name": m.name,
                            "started_at_secs": m.started_at_secs,
                            "elapsed_secs": now_unix_secs().saturating_sub(m.started_at_secs),
                        })
                    });
                    Ok(json!({
                        "session_id": target,
                        "count": entries.len(),
                        "capacity": MODULE_LOG_CAPACITY,
                        "current_module": current,
                        "modules": entries,
                    }))
                }
            }
        }
        _ => Err(format!("internal error: unknown local cmd {cmd_name}")),
    };

    match result {
        Ok(data) => ok_response(data),
        Err(msg) => error_response(msg),
    }
}

async fn handle_gateway_client(
    client_stream: TcpStream,
    ctx: Arc<GatewayCtx>,
    mut shutdown_rx: broadcast::Receiver<()>,
) -> anyhow::Result<()> {
    let peer = client_stream
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| "<unknown>".into());

    let session_id = ctx.next_session_id.fetch_add(1, AtomicOrdering::SeqCst);
    let connected_at = now_unix_secs();
    {
        let mut sessions = ctx.sessions.lock().unwrap();
        sessions.insert(
            session_id,
            GwClientInfo {
                base: ClientInfo {
                    id: session_id,
                    peer_addr: peer.clone(),
                    connected_at_secs: connected_at,
                    label: None,
                    last_cmd: None,
                    last_cmd_at_secs: None,
                    cmds_executed: 0,
                    sim_time_advanced_picos: 0,
                    execute_commands: 0,
                    current_module: None,
                    module_log: VecDeque::with_capacity(MODULE_LOG_CAPACITY),
                },
                backend_port: None,
                backend_pid: None,
                is_executing: false,
                config_temp_path: None,
                config_name: None,
                config_summary: None,
            },
        );
    }

    // Lifecycle messages are once-per-session (not hot path), so we always
    // emit them regardless of --quiet.
    eprintln!("[gateway] client connected: session_id={session_id} peer={peer}");

    let (reader, mut writer) = client_stream.into_split();
    let mut lines = BufReader::new(reader).lines();

    // Lazily spawned backend for this session.
    let mut backend: Option<Backend> = None;

    let result: anyhow::Result<()> = async {
        loop {
            let next = tokio::select! {
                line = lines.next_line() => line?,
                _ = shutdown_rx.recv() => {
                    let bye = error_response("gateway is shutting down");
                    let _ = write_json_line(&mut writer, &bye).await;
                    return Ok(());
                }
            };
            let line = match next {
                Some(l) => l,
                None => return Ok(()),
            };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Parse the JSON envelope. We need the `cmd` field to route.
            let cmd_value: Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(err) => {
                    let msg = format!("invalid request JSON: {err}");
                    let resp = error_response(msg.clone());
                    gateway_record_history(
                        &ctx,
                        HistoryEntry {
                            session_id,
                            cmd: "<invalid>".to_string(),
                            at_secs: now_unix_secs(),
                            ok: false,
                            detail: Some(msg),
                        },
                    );
                    write_json_line(&mut writer, &resp).await?;
                    continue;
                }
            };
            let cmd_name = cmd_value
                .get("cmd")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            gateway_record_request(&ctx, session_id, &cmd_name);

            // Session-only commands are answered by the gateway itself.
            if is_session_only_cmd(&cmd_name) {
                let resp = handle_gateway_local_cmd(&ctx, session_id, &cmd_value, &cmd_name).await;
                gateway_record_history(
                    &ctx,
                    HistoryEntry {
                        session_id,
                        cmd: cmd_name.clone(),
                        at_secs: now_unix_secs(),
                        ok: resp["ok"].as_bool().unwrap_or(false),
                        detail: None,
                    },
                );
                write_json_line(&mut writer, &resp).await?;
                continue;
            }

            // Handle `shutdown`: in gateway mode this only tears down the
            // CURRENT session's backend, not the gateway itself.
            if cmd_name == "shutdown" {
                let resp = ok_response(json!({"message": "session shutting down"}));
                write_json_line(&mut writer, &resp).await?;
                return Ok(());
            }

            // Hardware-touching command — ensure a backend exists.
            if backend.is_none() {
                // Snapshot the session's per-session config path (if any)
                // before spawn_backend yields.
                let session_cfg_path: Option<PathBuf> = {
                    let sessions = ctx.sessions.lock().unwrap();
                    sessions
                        .get(&session_id)
                        .and_then(|s| s.config_temp_path.clone())
                };
                match spawn_backend(&ctx, session_cfg_path.as_deref()).await {
                    Ok(b) => {
                        eprintln!(
                            "[gateway] session {session_id}: spawned backend pid={} on port {}",
                            b.pid, b.port
                        );
                        {
                            let mut sessions = ctx.sessions.lock().unwrap();
                            if let Some(info) = sessions.get_mut(&session_id) {
                                info.backend_port = Some(b.port);
                                info.backend_pid = Some(b.pid);
                            }
                        }
                        backend = Some(b);
                    }
                    Err(err) => {
                        let msg = format!("failed to spawn backend: {err}");
                        let resp = error_response(msg.clone());
                        gateway_record_history(
                            &ctx,
                            HistoryEntry {
                                session_id,
                                cmd: cmd_name.clone(),
                                at_secs: now_unix_secs(),
                                ok: false,
                                detail: Some(msg),
                            },
                        );
                        write_json_line(&mut writer, &resp).await?;
                        continue;
                    }
                }
            }
            let bk = backend.as_mut().unwrap();

            // Mark executing if applicable, then forward.
            let was_execute = is_execute_cmd(&cmd_name);
            if was_execute {
                let mut sessions = ctx.sessions.lock().unwrap();
                if let Some(info) = sessions.get_mut(&session_id) {
                    info.is_executing = true;
                }
            }

            let forward_result = bk.round_trip(trimmed).await;

            if was_execute {
                let mut sessions = ctx.sessions.lock().unwrap();
                if let Some(info) = sessions.get_mut(&session_id) {
                    info.is_executing = false;
                }
            }

            match forward_result {
                Ok(response_line) => {
                    // Attribute sim time advancement to this session.
                    if was_execute {
                        if let Some(picos) = parse_execute_response_for_sim_time(&response_line) {
                            let mut sessions = ctx.sessions.lock().unwrap();
                            if let Some(info) = sessions.get_mut(&session_id) {
                                info.base.sim_time_advanced_picos = info
                                    .base
                                    .sim_time_advanced_picos
                                    .saturating_add(u128::from(picos));
                                info.base.execute_commands =
                                    info.base.execute_commands.saturating_add(1);
                            }
                        }
                    }
                    let detail = if was_execute {
                        parse_execute_response_for_sim_time(&response_line)
                            .map(|p| format!("sim_time +{p} ps"))
                    } else {
                        None
                    };
                    let ok_flag = serde_json::from_str::<Value>(&response_line)
                        .ok()
                        .and_then(|v| v["ok"].as_bool())
                        .unwrap_or(false);
                    gateway_record_history(
                        &ctx,
                        HistoryEntry {
                            session_id,
                            cmd: cmd_name.clone(),
                            at_secs: now_unix_secs(),
                            ok: ok_flag,
                            detail,
                        },
                    );
                    writer.write_all(response_line.as_bytes()).await?;
                    writer.write_all(b"\n").await?;
                }
                Err(err) => {
                    // Backend died mid-command. Surface the error, drop the
                    // backend (a follow-up command will respawn).
                    let msg = format!("backend communication error: {err}");
                    if !is_quiet() {
                        eprintln!("[gateway] session {session_id}: {msg}");
                    }
                    let resp = error_response(msg.clone());
                    gateway_record_history(
                        &ctx,
                        HistoryEntry {
                            session_id,
                            cmd: cmd_name.clone(),
                            at_secs: now_unix_secs(),
                            ok: false,
                            detail: Some(msg),
                        },
                    );
                    write_json_line(&mut writer, &resp).await?;
                    backend = None;
                    {
                        let mut sessions = ctx.sessions.lock().unwrap();
                        if let Some(info) = sessions.get_mut(&session_id) {
                            info.backend_port = None;
                            info.backend_pid = None;
                        }
                    }
                }
            }
        }
    }
    .await;

    // Tear down the backend (if any).
    if let Some(b) = backend {
        b.shutdown().await;
    }

    // Snapshot + move to closed buffer. While we hold the session record
    // unlink the per-session config tempfile (if any) — see
    // `set_session_config` / `persist_session_config`.
    let info_snapshot = {
        let mut sessions = ctx.sessions.lock().unwrap();
        sessions.remove(&session_id).map(|gw| {
            if let Some(path) = gw.config_temp_path {
                let _ = std::fs::remove_file(&path);
            }
            gw.base
        })
    };
    if let Some(info) = info_snapshot {
        let mut closed = ctx.closed_sessions.lock().unwrap();
        if closed.len() >= CLOSED_SESSIONS_CAPACITY {
            closed.pop_front();
        }
        closed.push_back(ClosedSession {
            info,
            closed_at_secs: now_unix_secs(),
        });
    }

    eprintln!("[gateway] client disconnected: session_id={session_id} peer={peer}");

    result
}

async fn gateway(opts: &Opts) -> anyhow::Result<()> {
    let self_path = std::env::current_exe()
        .context("failed to resolve current executable path for backend spawning")?;
    let listener = TcpListener::bind(&opts.bind)
        .await
        .with_context(|| format!("failed to bind {}", opts.bind))?;

    let (shutdown_tx, _) = broadcast::channel::<()>(16);
    let ctx = Arc::new(GatewayCtx {
        sessions: std::sync::Mutex::new(HashMap::new()),
        closed_sessions: std::sync::Mutex::new(VecDeque::with_capacity(CLOSED_SESSIONS_CAPACITY)),
        history: std::sync::Mutex::new(VecDeque::with_capacity(HISTORY_CAPACITY)),
        next_session_id: AtomicU64::new(1),
        started_at_secs: now_unix_secs(),
        self_path,
        backend_quiet: true,
    });

    eprintln!("PLENA gateway listening on {}", opts.bind);
    eprintln!("[gateway] backends spawned per session");

    // Listen for ctrl-c so we can clean up children. Best-effort — kill_on_drop
    // on the Child takes care of the actual reaping when the gateway exits.
    let _ = shutdown_tx.clone();

    loop {
        let (stream, _) = listener.accept().await?;
        let ctx = ctx.clone();
        let client_shutdown_rx = shutdown_tx.subscribe();
        tokio::task::spawn_local(async move {
            if let Err(err) = handle_gateway_client(stream, ctx, client_shutdown_rx).await {
                if !is_quiet() {
                    eprintln!("[gateway] client task error: {err}");
                }
            }
        });
    }
}

async fn run_batch(opts: &Opts) -> anyhow::Result<()> {
    require_batch_path("opcode", &opts.opcode)?;
    require_batch_path("hbm", &opts.hbm)?;
    require_batch_path("fpsram", &opts.fpsram)?;

    let mut state = EmulatorState::new();
    state.preload_from_opts(opts).await?;

    let opcode_path = require_batch_path("opcode", &opts.opcode)?;
    // session_id = 0 is the sentinel for "not a TCP session" — the
    // progress atomics still get bumped but nobody is polling.
    state.execute_file(0, opcode_path).await?;
    state.print_batch_summary().await;
    state.dump_batch_files().await?;

    let stats = state.stats_json();
    let hbm_bytes_read = stats["hbm_bytes_read"].as_u64().unwrap_or(0);
    let hbm_bytes_written = stats["hbm_bytes_written"].as_u64().unwrap_or(0);
    let hbm_utilization = stats["hbm_utilization_bytes_per_sec"]
        .as_f64()
        .unwrap_or(0.0);
    eprintln!(
        "HBM Statistics - Bytes read: {:?} | Bytes written: {:?} | Utilization: {:.2e} bytes/sec",
        hbm_bytes_read, hbm_bytes_written, hbm_utilization
    );
    eprintln!("Simulation completed. Latency {:?}", state.executor.now());
    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let opts = Opts::parse();
    // In --serve / --gateway modes the per-instruction print is in the hot
    // path. Default to quiet there unless the user explicitly requested
    // --verbose. Batch mode keeps the existing --quiet behavior unchanged.
    let quiet = opts.quiet || ((opts.serve || opts.gateway) && !opts.verbose);
    QUIET_MODE.store(quiet, std::sync::atomic::Ordering::Relaxed);

    // The simulator (Executor, tch::Tensor, ...) is not `Send`, so the entire
    // server runs on a single LocalSet — concurrent connections cooperate via
    // tokio's async scheduler rather than across OS threads.
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async move {
            if opts.gateway {
                gateway(&opts).await
            } else if opts.serve {
                serve(&opts).await
            } else {
                run_batch(&opts).await
            }
        })
        .await
}
