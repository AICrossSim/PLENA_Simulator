//! `VectorMachine` — executes PLENA vector ISA opcodes (V_ADD, V_SUB, V_MUL,
//! V_EXP, V_RECI, V_RED_SUM, V_RED_MAX, etc.) against an underlying
//! [`VectorSram`].
//!
//! Vector ops operate on tiles of `tile_size` elements; the `mask_unit`-sized
//! sub-sections within each tile can be selectively included/excluded via a
//! per-head bitmask (the `mask` argument). When `rmask == 0` the op runs on
//! the full tile; otherwise only heads whose bit is set in `mask` are
//! updated.

use std::sync::Arc;

use half::bf16;
use quantize::{DataType, FpType, MxDataType, QuantTensor};
use sram::VectorSram;
use tch::{Kind, Tensor};

use crate::runtime_config::{
    VECTOR_ADD_CYCLES, VECTOR_EXP_CYCLES, VECTOR_MAX_CYCLES, VECTOR_MUL_CYCLES, VECTOR_RECI_CYCLES,
    VECTOR_SUM_CYCLES, VLEN,
};
use crate::{cycle, op};

fn head_is_selected(mask: u32, head: u32) -> bool {
    head < u32::BITS && (mask & (1_u32 << head)) != 0
}

/// Executes vector opcodes against `vram`. Cell payloads inside `vram` use
/// interior mutability (Mutex), so all methods only need `&self`.
pub(crate) struct VectorMachine {
    pub(crate) vram: Arc<VectorSram>,
    tile_size: u32,
    mask_unit: u32,
}

impl VectorMachine {
    pub(crate) fn new(vram: Arc<VectorSram>, tile_size: u32, mask_unit: u32) -> Self {
        Self {
            vram,
            tile_size,
            mask_unit,
        }
    }

    pub(crate) async fn add_scalar(&self, vd: u32, vs1: u32, f: f32, rmask: u8, mask: u32) {
        let a = self.vram.read(vs1).await;
        if rmask == 0 {
            let c = QuantTensor::quantize(a.as_tensor() + (f as f64), a.data_type());
            cycle!(*VECTOR_ADD_CYCLES);
            self.vram.write(vd, c).await;
        } else {
            let destination = self.vram.read(vd).await;
            let result = destination.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if head_is_selected(mask, head) {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = a.as_tensor().narrow(0, start, end - start);
                    let updated = &sliced + (f as f64);
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let c = QuantTensor::quantize(result, a.data_type());
            cycle!(*VECTOR_ADD_CYCLES);
            self.vram.write(vd, c).await;
        }
    }

    pub(crate) async fn sub_scalar(
        &self,
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
            let destination = self.vram.read(vd).await;
            let result = destination.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if head_is_selected(mask, head) {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = a.as_tensor().narrow(0, start, end - start);
                    let updated = if matches!(rorder, op::VectorOrder::Normal) {
                        &sliced - (f as f64)
                    } else {
                        (f as f64) - &sliced
                    };
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let c = QuantTensor::quantize(result, a.data_type());
            cycle!(*VECTOR_ADD_CYCLES);
            self.vram.write(vd, c).await;
        }
    }

    pub(crate) async fn mul_scalar(&self, vd: u32, vs1: u32, f: f32, rmask: u8, mask: u32) {
        let a = self.vram.read(vs1).await;
        if rmask == 0 {
            let c = QuantTensor::quantize(a.as_tensor() * (f as f64), a.data_type());
            cycle!(*VECTOR_MUL_CYCLES);
            self.vram.write(vd, c).await;
        } else {
            let destination = self.vram.read(vd).await;
            let result = destination.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if head_is_selected(mask, head) {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = a.as_tensor().narrow(0, start, end - start);
                    let updated = &sliced * (f as f64);
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let c = QuantTensor::quantize(result, a.data_type());
            cycle!(*VECTOR_MUL_CYCLES);
            self.vram.write(vd, c).await;
        }
    }

    pub(crate) async fn mul_route_f32(
        &self,
        vd: u32,
        vs1: u32,
        route_score: f32,
        rmask: u8,
        mask: u32,
    ) {
        let source = self.vram.read(vs1).await;
        assert_eq!(
            source.data_type(),
            MxDataType::Plain(DataType::Fp(FpType::BF16)),
            "V_MUL_ROUTE_F32 requires BF16 expert output storage"
        );
        drop(source);
        self.mul_scalar(vd, vs1, route_score, rmask, mask).await;
    }

    async fn read_bf16_row(
        &self,
        base: u32,
        width: usize,
        physical_rows: usize,
        operation: &str,
    ) -> Tensor {
        let tile_size = self.tile_size as usize;
        assert!(
            width > 0 && width.is_multiple_of(tile_size),
            "{operation} width must be a positive multiple of VLEN"
        );
        assert!(physical_rows > 0, "{operation} physical row count is zero");
        let storage_type = MxDataType::Plain(DataType::Fp(FpType::BF16));
        let stride = physical_rows
            .checked_mul(tile_size)
            .expect("BF16 row stride overflow");
        let mut values = Vec::with_capacity(width);
        for block in 0..width / tile_size {
            let offset = block
                .checked_mul(stride)
                .and_then(|value| u32::try_from(value).ok())
                .expect("BF16 row address offset overflow");
            let address = base.checked_add(offset).expect("BF16 row address overflow");
            let tile = self.vram.read(address).await;
            assert_eq!(
                tile.data_type(),
                storage_type,
                "{operation} requires BF16 VRAM storage"
            );
            for index in 0..tile_size {
                let value = tile.as_tensor().double_value(&[index as i64]) as f32;
                assert!(value.is_finite(), "{operation} input is non-finite");
                values.push(value);
            }
        }
        Tensor::from_slice(&values)
            .reshape([1, width as i64])
            .to_kind(Kind::BFloat16)
    }

    async fn write_bf16_row(
        &self,
        base: u32,
        row: Tensor,
        width: usize,
        physical_rows: usize,
        operation: &str,
    ) {
        let tile_size = self.tile_size as usize;
        assert_eq!(
            row.kind(),
            Kind::BFloat16,
            "{operation} must produce BF16 output"
        );
        assert_eq!(
            row.numel(),
            width,
            "{operation} output width does not match its ABI"
        );
        let values = Vec::<f32>::try_from(row.reshape([width as i64]).to_kind(Kind::Float))
            .expect("BF16 output conversion failed");
        assert!(
            values.iter().all(|value| value.is_finite()),
            "{operation} output is non-finite"
        );
        let storage_type = MxDataType::Plain(DataType::Fp(FpType::BF16));
        let stride = physical_rows
            .checked_mul(tile_size)
            .expect("BF16 output stride overflow");
        for (block, chunk) in values.chunks(tile_size).enumerate() {
            let offset = block
                .checked_mul(stride)
                .and_then(|value| u32::try_from(value).ok())
                .expect("BF16 output address offset overflow");
            let address = base
                .checked_add(offset)
                .expect("BF16 output address overflow");
            self.vram
                .write(
                    address,
                    QuantTensor::quantize(Tensor::from_slice(chunk), storage_type),
                )
                .await;
        }
    }

    /// Exact post-attention Qwen3 RMSNorm used by the executable MoE-tail
    /// validation path. The operation order matches Transformers 5.5.
    pub(crate) async fn qwen3_rmsnorm_bf16(
        &self,
        vd: u32,
        vs1: u32,
        vs2: u32,
        hidden: usize,
        physical_rows: usize,
    ) {
        let input = self
            .read_bf16_row(vs1, hidden, physical_rows, "V_QWEN3_RMSNORM_BF16")
            .await;
        let weight = self
            .read_bf16_row(vs2, hidden, physical_rows, "V_QWEN3_RMSNORM_BF16")
            .await;
        let input_fp32 = input.to_kind(Kind::Float);
        let variance = (&input_fp32 * &input_fp32).mean(Kind::Float);
        let normalized = input_fp32 * (variance + 1.0e-6f64).rsqrt();
        let output = weight * normalized.to_kind(Kind::BFloat16);
        self.write_bf16_row(vd, output, hidden, physical_rows, "V_QWEN3_RMSNORM_BF16")
            .await;
    }

    /// Execute one fused Qwen3 expert and accumulate its weighted output.
    /// Dispatch invokes this in ascending expert-id order, matching the
    /// Transformers 5.5 expert loop while loading one HBM bank at a time.
    pub(crate) async fn qwen3_expert_accumulate_bf16(
        &self,
        vd: u32,
        vs1: u32,
        hidden: usize,
        intermediate: usize,
        physical_rows: usize,
        expert_id: u32,
        route_score: f32,
        gate_up_weight: Tensor,
        down_weight: Tensor,
        reset: bool,
    ) {
        assert!(expert_id < 128, "Qwen3 expert ID is out of range");
        assert!(
            route_score.is_finite() && route_score >= 0.0,
            "Qwen3 route score is invalid"
        );
        assert_eq!(
            gate_up_weight.size(),
            [2 * intermediate as i64, hidden as i64],
            "Qwen3 fused gate/up weight shape mismatch"
        );
        assert_eq!(
            down_weight.size(),
            [hidden as i64, intermediate as i64],
            "Qwen3 down weight shape mismatch"
        );
        assert_eq!(gate_up_weight.kind(), Kind::BFloat16);
        assert_eq!(down_weight.kind(), Kind::BFloat16);
        let input = self
            .read_bf16_row(vs1, hidden, physical_rows, "V_QWEN3_EXPERT_COMBINE_BF16")
            .await;
        let mut final_output = if reset {
            Tensor::zeros([1, hidden as i64], (Kind::BFloat16, tch::Device::Cpu))
        } else {
            self.read_bf16_row(vd, hidden, physical_rows, "V_QWEN3_EXPERT_COMBINE_BF16")
                .await
        };
        let token_index = Tensor::zeros([1], (Kind::Int64, tch::Device::Cpu));
        let gate_up = input.linear(&gate_up_weight, Option::<&Tensor>::None);
        let chunks = gate_up.chunk(2, -1);
        let activated = chunks[0].silu() * &chunks[1];
        let expert_output = activated.linear(&down_weight, Option::<&Tensor>::None);
        let score = Tensor::from_slice(&[route_score]).to_kind(Kind::Float);
        let weighted = (expert_output * score).to_kind(Kind::BFloat16);
        let _ = final_output.index_add_(0, &token_index, &weighted);
        self.write_bf16_row(
            vd,
            final_output,
            hidden,
            physical_rows,
            "V_QWEN3_EXPERT_COMBINE_BF16",
        )
        .await;

        // Functional parity is exact; this is only a structural compute charge.
        let macs = 3usize.saturating_mul(hidden).saturating_mul(intermediate);
        let tile_macs = (self.tile_size as usize).saturating_mul(self.tile_size as usize);
        let structural_tiles = macs.div_ceil(tile_macs);
        cycle!(u32::try_from(structural_tiles).unwrap_or(u32::MAX));
    }

    pub(crate) async fn shift_scalar(&self, vd: u32, vs1: u32, shift: u32) {
        let a = self.vram.read(vs1).await;
        let tensor = a.as_tensor();
        let len = tensor.size()[0];
        let shift_amount = shift as i64;

        // Element shift (right): [a0, a1, a2, ...] -> [0, 0, ..., a0, a1, a2, ...]
        // Shift elements right by shift_amount, filling with zeros from the left
        let result = if shift_amount >= len {
            // Shift amount >= length, result is all zeros
            Tensor::zeros_like(tensor)
        } else if shift_amount == 0 {
            tensor.shallow_clone()
        } else {
            // Pad with zeros at the beginning, take elements from start to (len - shift_amount)
            let remaining = len - shift_amount;
            let shifted_part = tensor.narrow(0, 0, remaining);
            let zeros = Tensor::zeros([shift_amount], (tensor.kind(), tensor.device()));
            Tensor::cat(&[zeros, shifted_part], 0)
        };
        let c = QuantTensor::quantize(result, a.data_type());
        cycle!(*VECTOR_MUL_CYCLES);
        self.vram.write(vd, c).await;
    }

    pub(crate) async fn add(&self, vd: u32, vs1: u32, vs2: u32, rmask: u8, mask: u32) {
        let (a, b) = tokio::join!(self.vram.read(vs1), self.vram.read(vs2));
        if rmask == 0 {
            let c = QuantTensor::quantize(a.as_tensor() + b.as_tensor(), a.data_type());
            cycle!(*VECTOR_ADD_CYCLES);
            self.vram.write(vd, c).await;
        } else {
            let destination = self.vram.read(vd).await;
            let result = destination.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if head_is_selected(mask, head) {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let updated = a.as_tensor().narrow(0, start, end - start)
                        + b.as_tensor().narrow(0, start, end - start);
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let c = QuantTensor::quantize(result, a.data_type());
            cycle!(*VECTOR_ADD_CYCLES);
            self.vram.write(vd, c).await;
        }
    }

    pub(crate) async fn sub(&self, vd: u32, vs1: u32, vs2: u32, rmask: u8, mask: u32) {
        let (a, b) = tokio::join!(self.vram.read(vs1), self.vram.read(vs2));
        if rmask == 0 {
            let c = QuantTensor::quantize(a.as_tensor() - b.as_tensor(), a.data_type());
            cycle!(*VECTOR_ADD_CYCLES);
            self.vram.write(vd, c).await;
        } else {
            let destination = self.vram.read(vd).await;
            let result = destination.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if head_is_selected(mask, head) {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let updated = a.as_tensor().narrow(0, start, end - start)
                        - b.as_tensor().narrow(0, start, end - start);
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let c = QuantTensor::quantize(result, a.data_type());
            cycle!(*VECTOR_ADD_CYCLES);
            self.vram.write(vd, c).await;
        }
    }

    pub(crate) async fn mul(&self, vd: u32, vs1: u32, vs2: u32, rmask: u8, mask: u32) {
        let (a, b) = tokio::join!(self.vram.read(vs1), self.vram.read(vs2));
        if rmask == 0 {
            let c = QuantTensor::quantize(a.as_tensor() * b.as_tensor(), a.data_type());
            cycle!(*VECTOR_MUL_CYCLES);
            self.vram.write(vd, c).await;
        } else {
            let destination = self.vram.read(vd).await;
            let result = destination.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if head_is_selected(mask, head) {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let updated = a.as_tensor().narrow(0, start, end - start)
                        * b.as_tensor().narrow(0, start, end - start);
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let c = QuantTensor::quantize(result, a.data_type());
            cycle!(*VECTOR_MUL_CYCLES);
            self.vram.write(vd, c).await;
        }
    }

    pub(crate) async fn exp(&self, vd: u32, vs1: u32, rmask: u8, mask: u32) {
        let a = self.vram.read(vs1).await;
        // Clamp inputs to [-88, 88] to prevent bf16 overflow (exp(89) > bf16_max).
        // This matches what hardware exp units do (saturate instead of producing inf/NaN).
        let clamped = a.as_tensor().clamp(-88.0f64, 88.0f64);
        if rmask == 0 {
            let c = QuantTensor::quantize(clamped.exp(), a.data_type());
            cycle!(*VECTOR_EXP_CYCLES);
            self.vram.write(vd, c).await;
        } else {
            let destination = self.vram.read(vd).await;
            let result = destination.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if head_is_selected(mask, head) {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = clamped.narrow(0, start, end - start);
                    let updated = &sliced.exp();
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let c = QuantTensor::quantize(result, a.data_type());
            cycle!(*VECTOR_EXP_CYCLES);
            self.vram.write(vd, c).await;
        }
    }

    pub(crate) async fn reciprocal(&self, vd: u32, vs1: u32, rmask: u8, mask: u32) {
        let a = self.vram.read(vs1).await;
        if rmask == 0 {
            let c = QuantTensor::quantize(a.as_tensor().reciprocal(), a.data_type());
            cycle!(*VECTOR_RECI_CYCLES);
            self.vram.write(vd, c).await;
        } else {
            let destination = self.vram.read(vd).await;
            let result = destination.as_tensor().shallow_clone();
            let total_heads = self.tile_size / self.mask_unit;
            for head in 0..total_heads {
                if head_is_selected(mask, head) {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let sliced = a.as_tensor().narrow(0, start, end - start);
                    let updated = &sliced.reciprocal();
                    result.narrow(0, start, end - start).copy_(&updated);
                }
            }
            let c = QuantTensor::quantize(result, a.data_type());
            cycle!(*VECTOR_RECI_CYCLES);
            self.vram.write(vd, c).await;
        }
    }

    pub(crate) async fn vector_transfer_fp(&self, vd: u32, f: &[bf16]) {
        assert_eq!(
            f.len(),
            self.vram.tile_size() as usize,
            "Input vector length must match tile_size"
        );
        // Convert bf16 slice to f32 vector
        let f32_vec: Vec<f32> = f.iter().map(|x| f32::from(*x)).collect();
        // Create tensor from f32 vector
        let tensor = Tensor::from_slice(&f32_vec);
        // Quantize the tensor according to vram data type
        let c = QuantTensor::quantize(tensor, self.vram.ty());
        cycle!(*VLEN);
        self.vram.write(vd, c).await;
    }

    pub(crate) async fn reduce_sum(&self, vs1: u32, f: f32, rmask: u8, mask: u32) -> f32 {
        let a = self.vram.read(vs1).await;
        cycle!(*VECTOR_SUM_CYCLES);
        if rmask == 0 {
            let val: f32 = a.as_tensor().sum(tch::Kind::Float).try_into().unwrap();
            f + val
        } else {
            let total_heads = self.tile_size / self.mask_unit;
            let mut result = f;
            for head in 0..total_heads {
                if head_is_selected(mask, head) {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let value: f32 = a
                        .as_tensor()
                        .narrow(0, start, end - start)
                        .sum(tch::Kind::Float)
                        .try_into()
                        .unwrap();
                    result += value;
                }
            }
            result
        }
    }

    pub(crate) async fn reduce_max(&self, vs1: u32, f: f32, rmask: u8, mask: u32) -> f32 {
        let a = self.vram.read(vs1).await;
        cycle!(*VECTOR_MAX_CYCLES);
        if rmask == 0 {
            let val: f32 = a.as_tensor().max().try_into().unwrap();
            f32::max(val, f)
        } else {
            let total_heads = self.tile_size / self.mask_unit;
            let mut result = f;
            for head in 0..total_heads {
                if head_is_selected(mask, head) {
                    let start = (head * self.mask_unit) as i64;
                    let end = ((head + 1) * self.mask_unit) as i64;
                    let value: f32 = a
                        .as_tensor()
                        .narrow(0, start, end - start)
                        .max()
                        .try_into()
                        .unwrap();
                    result = f32::max(result, value);
                }
            }
            result
        }
    }

    /// Execute the BF16 Qwen3 router linear operation with one output cast.
    ///
    /// The input row uses the compiler's column-tiled layout with
    /// `input_physical_rows` rows. Router weights use 128 physical rows. Both
    /// are reconstructed as BF16 tensors before calling the same libtorch
    /// `linear` operation used by the sealed CPU oracle.
    pub(crate) async fn router_linear_bf16(
        &self,
        vd: u32,
        vs1: u32,
        vs2: u32,
        hidden: usize,
        expert_count: usize,
        input_physical_rows: usize,
    ) {
        let tile_size = self.tile_size as usize;
        assert_eq!(expert_count, 128, "Qwen3 router requires 128 experts");
        assert!(
            hidden > 0 && hidden.is_multiple_of(tile_size),
            "router hidden size must be a positive multiple of VLEN"
        );
        assert!(
            input_physical_rows > 0,
            "router input physical row count must be positive"
        );
        let storage_type = MxDataType::Plain(DataType::Fp(FpType::BF16));
        let hidden_blocks = hidden / tile_size;
        let input_stride = input_physical_rows
            .checked_mul(tile_size)
            .expect("router input stride overflow");
        let weight_stride = expert_count
            .checked_mul(tile_size)
            .expect("router weight stride overflow");

        let mut input_values = Vec::with_capacity(hidden);
        for block in 0..hidden_blocks {
            let offset = block
                .checked_mul(input_stride)
                .and_then(|value| u32::try_from(value).ok())
                .expect("router input address offset overflow");
            let address = vs1
                .checked_add(offset)
                .expect("router input address overflow");
            let tile = self.vram.read(address).await;
            assert_eq!(
                tile.data_type(),
                storage_type,
                "V_ROUTER_LINEAR_BF16 requires BF16 input storage"
            );
            for index in 0..tile_size {
                let value = tile.as_tensor().double_value(&[index as i64]) as f32;
                assert!(
                    value.is_finite(),
                    "router input contains a non-finite value"
                );
                input_values.push(value);
            }
        }

        let mut weight_values = vec![0.0f32; expert_count * hidden];
        for block in 0..hidden_blocks {
            let block_offset = block
                .checked_mul(weight_stride)
                .and_then(|value| u32::try_from(value).ok())
                .expect("router weight block offset overflow");
            for expert in 0..expert_count {
                let row_offset = expert
                    .checked_mul(tile_size)
                    .and_then(|value| u32::try_from(value).ok())
                    .expect("router weight row offset overflow");
                let address = vs2
                    .checked_add(block_offset)
                    .and_then(|value| value.checked_add(row_offset))
                    .expect("router weight address overflow");
                let tile = self.vram.read(address).await;
                assert_eq!(
                    tile.data_type(),
                    storage_type,
                    "V_ROUTER_LINEAR_BF16 requires BF16 weight storage"
                );
                let destination = expert * hidden + block * tile_size;
                for index in 0..tile_size {
                    let value = tile.as_tensor().double_value(&[index as i64]) as f32;
                    assert!(
                        value.is_finite(),
                        "router weight contains a non-finite value"
                    );
                    weight_values[destination + index] = value;
                }
            }
        }

        let input = Tensor::from_slice(&input_values).to_kind(tch::Kind::BFloat16);
        let weights = Tensor::from_slice(&weight_values)
            .reshape([expert_count as i64, hidden as i64])
            .to_kind(tch::Kind::BFloat16);
        let logits = input.linear(&weights, Option::<&Tensor>::None);
        assert_eq!(
            logits.kind(),
            tch::Kind::BFloat16,
            "router linear must cast logits once to BF16"
        );
        let logits = Vec::<f32>::try_from(logits.to_kind(tch::Kind::Float)).unwrap();
        assert_eq!(logits.len(), expert_count);
        assert!(
            logits.iter().all(|value| value.is_finite()),
            "router linear produced a non-finite logit"
        );
        for (chunk_index, chunk) in logits.chunks(tile_size).enumerate() {
            let mut padded = vec![0.0f32; tile_size];
            padded[..chunk.len()].copy_from_slice(chunk);
            let offset = chunk_index
                .checked_mul(tile_size)
                .and_then(|value| u32::try_from(value).ok())
                .expect("router output offset overflow");
            let address = vd
                .checked_add(offset)
                .expect("router output address overflow");
            self.vram
                .write(
                    address,
                    QuantTensor::quantize(Tensor::from_slice(&padded), storage_type),
                )
                .await;
        }

        // Functional execution is exact against the libtorch oracle. The
        // cycle charge is only a structural composition of existing ops.
        let dot_tiles =
            u32::try_from(expert_count.saturating_mul(hidden_blocks)).unwrap_or(u32::MAX);
        let cycles_per_tile = (*VECTOR_MUL_CYCLES).saturating_add(*VECTOR_SUM_CYCLES);
        cycle!(dot_tiles.saturating_mul(cycles_per_tile));
    }

    /// Read one routed-expert BF16 logit row and execute the FP32
    /// softmax/top-k/renormalization sequence.
    ///
    /// These tensor operations intentionally mirror Transformers. Replacing
    /// the two normalizations with a selected-logit softmax is algebraically
    /// equivalent but can round differently in FP32.
    pub(crate) async fn topk_softmax(
        &self,
        vs1: u32,
        expert_count: usize,
        topk: usize,
    ) -> (Vec<u32>, Vec<f32>) {
        assert!(topk > 0, "topk must be positive");
        assert!(
            topk <= expert_count,
            "topk={} exceeds expert_count={}",
            topk,
            expert_count
        );

        let tile_size = self.tile_size as usize;
        let mut logits = Vec::with_capacity(expert_count);
        for chunk_start in (0..expert_count).step_by(tile_size) {
            let row = self.vram.read(vs1 + chunk_start as u32).await;
            let chunk_len = (expert_count - chunk_start).min(tile_size);
            for idx in 0..chunk_len {
                let value = row.as_tensor().double_value(&[idx as i64]) as f32;
                assert!(
                    value.is_finite(),
                    "V_TOPK input contains a non-finite logit"
                );
                logits.push(value);
            }
        }

        let logits = Tensor::from_slice(&logits);
        let probabilities = logits.softmax(-1, tch::Kind::Float);
        let (selected, indices) = probabilities.topk(topk as i64, -1, true, true);
        let denominator = selected.sum(tch::Kind::Float);
        let denominator_value: f32 = (&denominator).try_into().unwrap();
        assert!(
            denominator_value.is_finite() && denominator_value > 0.0,
            "V_TOPK selected-probability denominator is invalid"
        );
        let normalized = selected / denominator;
        let weights = Vec::<f32>::try_from(normalized).unwrap();
        let indices = Vec::<i64>::try_from(indices)
            .unwrap()
            .into_iter()
            .map(|index| u32::try_from(index).unwrap())
            .collect();

        // Functional execution uses libtorch FP32 kernels. Emulated latency is
        // expressed in existing vector primitives: one selection pass per VRAM
        // row plus normalization. This is structural, not calibrated timing.
        let selection_rows = expert_count.div_ceil(tile_size) as u32;
        let topk_cycles = (*VECTOR_MAX_CYCLES)
            .saturating_mul(selection_rows)
            .saturating_add(*VECTOR_ADD_CYCLES)
            .saturating_add(*VECTOR_EXP_CYCLES)
            .saturating_add(*VECTOR_SUM_CYCLES)
            .saturating_add(*VECTOR_RECI_CYCLES)
            .saturating_add(*VECTOR_MUL_CYCLES);
        cycle!(topk_cycles);
        (indices, weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantize::{DataType, FpType, MxDataType};
    use runtime::{Executor, Instant};
    use std::sync::Mutex;

    #[test]
    fn head_mask_selects_only_requested_segments() {
        let mask = 0b0101;
        assert!(head_is_selected(mask, 0));
        assert!(!head_is_selected(mask, 1));
        assert!(head_is_selected(mask, 2));
        assert!(!head_is_selected(mask, 3));
        assert!(!head_is_selected(u32::MAX, 32));
    }

    async fn router_linear_fixture_hash(hidden: usize) -> u64 {
        const TILE_SIZE: usize = 64;
        const INPUT_PHYSICAL_ROWS: usize = 4;
        const EXPERTS: usize = 128;
        let blocks = hidden / TILE_SIZE;
        let input_span = blocks * INPUT_PHYSICAL_ROWS * TILE_SIZE;
        let weight_base = input_span as u32;
        let weight_span = blocks * EXPERTS * TILE_SIZE;
        let output_base = (input_span + weight_span) as u32;
        let depth = output_base as usize / TILE_SIZE + EXPERTS.div_ceil(TILE_SIZE);
        let fp_type = DataType::Fp(FpType::BF16);
        let storage_type = MxDataType::Plain(fp_type);
        let vram = Arc::new(VectorSram::new(TILE_SIZE as u32, depth, fp_type, 4));
        let machine = VectorMachine::new(vram.clone(), TILE_SIZE as u32, 16);

        for block in 0..blocks {
            let values = (0..TILE_SIZE)
                .map(|offset| {
                    let index = block * TILE_SIZE + offset;
                    let raw = ((index * 37 + 11) % 2001) as i32 - 1000;
                    bf16::from_f32(raw as f32 / 257.0).to_f32()
                })
                .collect::<Vec<_>>();
            vram.write(
                (block * INPUT_PHYSICAL_ROWS * TILE_SIZE) as u32,
                QuantTensor::quantize(Tensor::from_slice(&values), storage_type),
            )
            .await;
        }
        for block in 0..blocks {
            for expert in 0..EXPERTS {
                let values = (0..TILE_SIZE)
                    .map(|offset| {
                        let index = block * TILE_SIZE + offset;
                        let raw = ((expert * 97 + index * 53 + 19) % 2001) as i32 - 1000;
                        bf16::from_f32(raw as f32 / 263.0).to_f32()
                    })
                    .collect::<Vec<_>>();
                let address =
                    weight_base + (block * EXPERTS * TILE_SIZE + expert * TILE_SIZE) as u32;
                vram.write(
                    address,
                    QuantTensor::quantize(Tensor::from_slice(&values), storage_type),
                )
                .await;
            }
        }

        machine
            .router_linear_bf16(
                output_base,
                0,
                weight_base,
                hidden,
                EXPERTS,
                INPUT_PHYSICAL_ROWS,
            )
            .await;
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for chunk in 0..EXPERTS.div_ceil(TILE_SIZE) {
            let output = vram.read(output_base + (chunk * TILE_SIZE) as u32).await;
            for index in 0..(EXPERTS - chunk * TILE_SIZE).min(TILE_SIZE) {
                let value = output.as_tensor().double_value(&[index as i64]) as f32;
                for byte in bf16::from_f32(value).to_bits().to_le_bytes() {
                    hash ^= byte as u64;
                    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
        }
        hash
    }

    #[tokio::test]
    async fn router_linear_matches_transformers_bf16_oracles() {
        let executor = Executor::new();
        let got = Arc::new(Mutex::new(None));
        let got_task = got.clone();
        executor.spawn(async move {
            *got_task.lock().unwrap() = Some((
                router_linear_fixture_hash(64).await,
                router_linear_fixture_hash(2048).await,
            ));
        });

        executor.enter(Instant::ETERNITY).await;
        assert_eq!(
            got.lock().unwrap().take().unwrap(),
            (0xb678_a0d7_fe63_550a, 0x81e3_82ff_292f_fd8e)
        );
    }

    #[tokio::test]
    #[should_panic(expected = "router input contains a non-finite value")]
    async fn router_linear_nonfinite_input_fails_closed() {
        let executor = Executor::new();
        executor.spawn(async move {
            let fp_type = DataType::Fp(FpType::BF16);
            let storage_type = MxDataType::Plain(fp_type);
            let vram = Arc::new(VectorSram::new(64, 4, fp_type, 4));
            vram.write(
                0,
                QuantTensor::quantize(Tensor::from_slice(&vec![f32::NAN; 64]), storage_type),
            )
            .await;
            VectorMachine::new(vram, 64, 16)
                .router_linear_bf16(0, 0, 0, 64, 128, 4)
                .await;
        });
        executor.enter(Instant::ETERNITY).await;
    }

    #[tokio::test]
    #[should_panic(expected = "out of bounds")]
    async fn router_linear_weight_address_fails_closed() {
        let executor = Executor::new();
        executor.spawn(async move {
            let fp_type = DataType::Fp(FpType::BF16);
            let storage_type = MxDataType::Plain(fp_type);
            let vram = Arc::new(VectorSram::new(64, 1, fp_type, 4));
            vram.write(
                0,
                QuantTensor::quantize(Tensor::from_slice(&vec![0.0f32; 64]), storage_type),
            )
            .await;
            VectorMachine::new(vram, 64, 16)
                .router_linear_bf16(0, 0, 64, 64, 128, 4)
                .await;
        });
        executor.enter(Instant::ETERNITY).await;
    }

    #[tokio::test]
    async fn topk_softmax_uses_descending_logits_and_low_index_ties() {
        let executor = Executor::new();
        let got = Arc::new(Mutex::new(None));
        let got_task = got.clone();

        executor.spawn(async move {
            let fp_type = DataType::Fp(FpType::BF16);
            let vram = Arc::new(VectorSram::new(64, 4, fp_type, 4));
            let machine = VectorMachine::new(vram.clone(), 64, 16);
            let ty = MxDataType::Plain(fp_type);
            let mut input = vec![-100.0f32; 64];
            input[7] = 4.0;
            input[3] = 4.0;
            input[9] = 2.0;
            input[0] = 1.0;
            vram.write(0, QuantTensor::quantize(Tensor::from_slice(&input), ty))
                .await;

            let (indices, weights) = machine.topk_softmax(0, 32, 4).await;
            *got_task.lock().unwrap() = Some((indices, weights));
        });

        executor.enter(Instant::ETERNITY).await;
        let (indices, weights) = got.lock().unwrap().take().unwrap();
        assert_eq!(indices, vec![3, 7, 9, 0]);
        let denominator = 2.0 + f32::exp(-2.0) + f32::exp(-3.0);
        let expected = [
            1.0 / denominator,
            1.0 / denominator,
            f32::exp(-2.0) / denominator,
            f32::exp(-3.0) / denominator,
        ];
        for (actual, expected) in weights.iter().zip(expected) {
            assert!((actual - expected).abs() < 0.003);
        }
    }

    #[tokio::test]
    async fn topk_softmax_matches_transformers_5_5_fp32_oracle_fixture() {
        let executor = Executor::new();
        let got = Arc::new(Mutex::new(None));
        let got_task = got.clone();

        executor.spawn(async move {
            let fp_type = DataType::Fp(FpType::BF16);
            let vram = Arc::new(VectorSram::new(64, 4, fp_type, 4));
            let machine = VectorMachine::new(vram.clone(), 64, 16);
            let ty = MxDataType::Plain(fp_type);
            let mut first = vec![-100.0f32; 64];
            let mut second = vec![-100.0f32; 64];
            first[2] = 7.0;
            first[5] = 7.0;
            first[63] = 3.0;
            second[0] = 4.0;
            second[7] = 5.0;
            second[20] = 2.5;
            second[21] = 2.25;
            second[63] = 6.0;
            vram.write(0, QuantTensor::quantize(Tensor::from_slice(&first), ty))
                .await;
            vram.write(64, QuantTensor::quantize(Tensor::from_slice(&second), ty))
                .await;

            let (indices, weights) = machine.topk_softmax(0, 128, 8).await;
            *got_task.lock().unwrap() = Some((indices, weights));
        });

        executor.enter(Instant::ETERNITY).await;
        let (indices, weights) = got.lock().unwrap().take().unwrap();
        assert_eq!(indices, vec![2, 5, 127, 71, 64, 63, 84, 85]);
        // Generated by transformers==5.5.0 Qwen3MoeTopKRouter using the BF16
        // logits above. Pinning bits catches algebraic rewrites that alter FP32
        // rounding even when the selected experts remain unchanged.
        let expected_bits = [
            0x3ec5_99e4,
            0x3ec5_99e4,
            0x3e11_6305,
            0x3d55_f072,
            0x3c9d_685f,
            0x3be7_a0d4,
            0x3b8c_7d58,
            0x3b5a_d3ad,
        ];
        assert_eq!(
            weights
                .iter()
                .map(|weight| weight.to_bits())
                .collect::<Vec<_>>(),
            expected_bits
        );
    }

    #[tokio::test]
    #[should_panic(expected = "V_TOPK input contains a non-finite logit")]
    async fn topk_softmax_nonfinite_row_fails_closed() {
        let executor = Executor::new();
        let got = Arc::new(Mutex::new(None));
        let got_task = got.clone();

        executor.spawn(async move {
            let fp_type = DataType::Fp(FpType::BF16);
            let vram = Arc::new(VectorSram::new(64, 4, fp_type, 4));
            let machine = VectorMachine::new(vram.clone(), 64, 16);
            let ty = MxDataType::Plain(fp_type);
            let input = vec![f32::NAN; 64];
            vram.write(0, QuantTensor::quantize(Tensor::from_slice(&input), ty))
                .await;
            *got_task.lock().unwrap() = Some(machine.topk_softmax(0, 32, 4).await);
        });

        executor.enter(Instant::ETERNITY).await;
        let _ = got.lock().unwrap().take();
    }
}
